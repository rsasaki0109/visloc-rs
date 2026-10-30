//! Online camera-IMU time offset (and focal length) estimation.
//!
//! A browser gives no common clock for camera frames and motion events:
//! Chrome reports frame capture times, Safari only presentation times, a
//! few tens of milliseconds later. On ADVIO's iPhone recordings a 30 ms
//! error doubles the trajectory error and 60 ms breaks tracking repeatedly.
//!
//! The camera's rotation between consecutive frames is measured from the
//! tracked points alone (Kabsch on their bearings; rotation dominates the
//! image motion of a hand-held phone), and the gyroscope's rotation over the
//! same interval is integrated with the frame times shifted by a candidate
//! offset. The offset whose gyroscope rotations best match the visual ones
//! is the residual time offset of the frame timestamps. The match is the
//! normalized correlation of the rotation vectors after the best rotation
//! between the IMU and camera frames (hand-eye calibration, Kabsch again),
//! so a wrong camera-IMU rotation does not hide the offset, and that best
//! rotation is reported against the assumed one. The ratio of the visual
//! to the gyroscope rotation about the image axes measures the focal length
//! error (a focal length `k` times too long shrinks visual rotations by
//! `1 / k`).
//!
//! On ADVIO this recovers the dataset's ~300 ms frame-clock offset with a
//! correlation above 0.99.

use std::collections::{HashMap, VecDeque};

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};

/// Seconds of motion the estimate is computed over.
const WINDOW_S: f64 = 12.0;
/// Offsets searched, seconds either side of the current timestamps.
const SEARCH_S: f64 = 0.2;
const STEP_S: f64 = 0.002;
/// Tracks two frames must share for a rotation measurement.
const MIN_COMMON: usize = 15;

/// What [`TimeSync::estimate`] found.
#[derive(Debug, Clone, Copy)]
pub struct SyncEstimate {
    /// Add this to the frame timestamps (ms) to put them on the IMU clock.
    pub offset_ms: f64,
    /// Peak normalized correlation of visual and gyroscope rotations, with
    /// the best camera-IMU rotation.
    pub correlation: f64,
    /// The same correlation with the assumed camera-IMU rotation.
    pub correlation_assumed: f64,
    /// The best camera-IMU rotation (`T_imu_cam`'s rotation), and its angle
    /// from the assumed one (degrees).
    pub imu_from_camera: UnitQuaternion<f64>,
    pub rotation_error_deg: f64,
    /// Second / first singular value of the rotation fit: near 0 when the
    /// motion turned about one axis only (the rotation about it is then
    /// not measured).
    pub axis_spread: f64,
    /// Visual / gyroscope rotation about the image axes: multiply the focal
    /// length by this.
    pub focal_scale: f64,
    /// Frame intervals and RMS gyroscope rate (rad/s) behind the estimate.
    pub intervals: usize,
    pub rate_rad_s: f64,
}

struct Interval {
    t0: f64,
    t1: f64,
    /// Camera rotation from frame 0 to frame 1, rotation vector (rad), in
    /// the camera frame.
    visual: Vector3<f64>,
}

/// Collects frame-to-frame visual rotations and gyroscope samples.
pub struct TimeSync {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    camera_from_imu: Rotation3<f64>,
    previous: Option<(f64, HashMap<u64, Vector3<f64>>)>,
    intervals: VecDeque<Interval>,
    /// `(t, gyro)`, seconds and rad/s in the IMU frame.
    gyro: VecDeque<(f64, Vector3<f64>)>,
}

impl TimeSync {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, imu_from_camera: &UnitQuaternion<f64>) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            camera_from_imu: imu_from_camera.inverse().to_rotation_matrix(),
            previous: None,
            intervals: VecDeque::new(),
            gyro: VecDeque::new(),
        }
    }

    pub fn push_gyro(&mut self, t_ns: i64, gyro: Vector3<f64>) {
        let t = t_ns as f64 * 1e-9;
        self.gyro.push_back((t, gyro));
        while self
            .gyro
            .front()
            .is_some_and(|&(t0, _)| t - t0 > WINDOW_S + 2.0 * SEARCH_S + 1.0)
        {
            self.gyro.pop_front();
        }
    }

    /// A processed frame's tracks (`id, x, y` on the pinhole camera).
    pub fn push_frame(&mut self, t_ns: i64, tracks: &[(u64, f32, f32)]) {
        let t = t_ns as f64 * 1e-9;
        let bearings: HashMap<u64, Vector3<f64>> = tracks
            .iter()
            .map(|&(id, x, y)| {
                let b = Vector3::new(
                    (f64::from(x) - self.cx) / self.fx,
                    (f64::from(y) - self.cy) / self.fy,
                    1.0,
                );
                (id, b.normalize())
            })
            .collect();
        if let Some((t0, previous)) = &self.previous {
            if let Some(visual) = rotation_between(previous, &bearings) {
                self.intervals.push_back(Interval {
                    t0: *t0,
                    t1: t,
                    visual,
                });
            }
        }
        self.previous = Some((t, bearings));
        while self.intervals.front().is_some_and(|i| t - i.t0 > WINDOW_S) {
            self.intervals.pop_front();
        }
    }

    /// The next frame starts a new chain of tracks (a restarted tracker
    /// numbers its tracks from 0 again).
    pub fn break_chain(&mut self) {
        self.previous = None;
    }

    /// Forgets everything (after the frame timestamps were shifted).
    pub fn clear(&mut self) {
        self.previous = None;
        self.intervals.clear();
    }

    /// Gyroscope rotation over `[t0, t1]` (trapezoids on the samples,
    /// interpolated at the ends), in the IMU frame; `None` where the samples
    /// do not cover it.
    fn gyro_rotation(&self, t0: f64, t1: f64) -> Option<Vector3<f64>> {
        let g = &self.gyro;
        let first = g.partition_point(|s| s.0 < t0);
        if first == 0 || first >= g.len() || g.back()?.0 < t1 {
            return None;
        }
        let at = |t: f64| -> Vector3<f64> {
            let k = g.partition_point(|s| s.0 < t).clamp(1, g.len() - 1);
            let (a, b) = (g[k - 1], g[k]);
            let w = ((t - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
            a.1 * (1.0 - w) + b.1 * w
        };
        let mut sum = Vector3::zeros();
        let (mut t, mut w) = (t0, at(t0));
        for s in g.iter().skip(first) {
            if s.0 >= t1 {
                break;
            }
            sum += (w + s.1) * 0.5 * (s.0 - t);
            t = s.0;
            w = s.1;
        }
        sum += (w + at(t1)) * 0.5 * (t1 - t);
        Some(sum)
    }

    /// `Σ g vᵀ` and `Σ |v|²`, `Σ |g|²` over the intervals the gyroscope
    /// covers with the frame times shifted by `d`, and their count.
    fn moments(&self, d: f64) -> (Matrix3<f64>, f64, f64, usize) {
        let (mut m, mut vv, mut gg, mut n) = (Matrix3::zeros(), 0.0, 0.0, 0usize);
        for i in &self.intervals {
            let Some(g) = self.gyro_rotation(i.t0 + d, i.t1 + d) else {
                continue;
            };
            m += g * i.visual.transpose();
            vv += i.visual.norm_squared();
            gg += g.norm_squared();
            n += 1;
        }
        (m, vv, gg, n)
    }

    /// The residual time offset, once enough rotation has been seen.
    pub fn estimate(&self) -> Option<SyncEstimate> {
        if self.intervals.len() < 150 {
            return None;
        }
        let steps = (SEARCH_S / STEP_S).round() as i64;
        let mut scores = Vec::with_capacity(2 * steps as usize + 1);
        for k in -steps..=steps {
            let d = k as f64 * STEP_S;
            let (m, vv, gg, n) = self.moments(d);
            let score = match best_rotation(&m) {
                Some((_, fit, _)) if n >= 100 && vv > 0.0 && gg > 0.0 => fit / (vv * gg).sqrt(),
                _ => f64::NAN,
            };
            scores.push((d, score));
        }
        let (best, &(d, correlation)) = scores
            .iter()
            .enumerate()
            .filter(|(_, s)| s.1.is_finite())
            .max_by(|a, b| a.1 .1.total_cmp(&b.1 .1))?;
        // A peak on the edge of the search is not a measurement.
        if best == 0 || best + 1 == scores.len() {
            return None;
        }
        // Parabola through the peak and its neighbours.
        let (l, r) = (scores[best - 1].1, scores[best + 1].1);
        let denom = l - 2.0 * correlation + r;
        let refine = if denom < 0.0 {
            0.5 * (l - r) / denom
        } else {
            0.0
        };
        let offset = d + refine.clamp(-1.0, 1.0) * STEP_S;
        // The rotation, focal scale and motion energy at that offset.
        let (m, vv, gg, _) = self.moments(offset);
        let (camera_from_imu, _, axis_spread) = best_rotation(&m)?;
        let correlation_assumed = (self.camera_from_imu.matrix() * m).trace() / (vv * gg).sqrt();
        let (mut num, mut den, mut energy, mut n) = (0.0, 0.0, 0.0, 0usize);
        for i in &self.intervals {
            let Some(g) = self.gyro_rotation(i.t0 + offset, i.t1 + offset) else {
                continue;
            };
            let g = camera_from_imu * g;
            // Rotation about the image axes only: rotation about the optical
            // axis does not depend on the focal length.
            num += i.visual.x * g.x + i.visual.y * g.y;
            den += g.x * g.x + g.y * g.y;
            energy += g.norm_squared() / (i.t1 - i.t0).powi(2);
            n += 1;
        }
        Some(SyncEstimate {
            offset_ms: offset * 1e3,
            correlation,
            correlation_assumed,
            imu_from_camera: UnitQuaternion::from_rotation_matrix(&camera_from_imu.inverse()),
            rotation_error_deg: camera_from_imu.angle_to(&self.camera_from_imu).to_degrees(),
            axis_spread,
            focal_scale: if den > 0.0 { num / den } else { 1.0 },
            intervals: n,
            rate_rad_s: (energy / n.max(1) as f64).sqrt(),
        })
    }
}

/// The rotation `R` maximizing `Σ v · (R g)` given `m = Σ g vᵀ` (Kabsch),
/// that maximum, and the second / first singular value of `m`.
fn best_rotation(m: &Matrix3<f64>) -> Option<(Rotation3<f64>, f64, f64)> {
    let svd = m.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let s = svd.singular_values;
    // tr(R m) with m = U S Vᵀ is largest for R = V D Uᵀ: s1 + s2 + det s3.
    let det = (v_t.transpose() * u.transpose()).determinant().signum();
    let d = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, det));
    let r = Rotation3::from_matrix_unchecked(v_t.transpose() * d * u.transpose());
    let mut sorted = [s[0], s[1], s[2]];
    sorted.sort_by(|a, b| b.total_cmp(a));
    let fit = s[0] + s[1] + s[2] - (1.0 - det) * sorted[2];
    let spread = if sorted[0] > 0.0 {
        sorted[1] / sorted[0]
    } else {
        0.0
    };
    Some((r, fit, spread))
}

/// Rotation vector of the camera motion from the frame with bearings `a`
/// to the frame with bearings `b` (shared track ids), by Kabsch: points
/// move by the inverse of the camera's rotation.
fn rotation_between(
    a: &HashMap<u64, Vector3<f64>>,
    b: &HashMap<u64, Vector3<f64>>,
) -> Option<Vector3<f64>> {
    let mut h = Matrix3::zeros();
    let mut common = 0usize;
    for (id, pa) in a {
        if let Some(pb) = b.get(id) {
            h += pa * pb.transpose();
            common += 1;
        }
    }
    if common < MIN_COMMON {
        return None;
    }
    let svd = h.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut d = Matrix3::identity();
    if (v_t.transpose() * u.transpose()).determinant() < 0.0 {
        d[(2, 2)] = -1.0;
    }
    // b = R a with R = V D Uᵀ; the camera turned by Rᵀ.
    let r = v_t.transpose() * d * u.transpose();
    Some(Rotation3::from_matrix_unchecked(r.transpose()).scaled_axis())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Visual rotations made from a gyroscope signal through a known
    /// camera-IMU rotation and frame time offset; the rig assumes neither.
    #[test]
    fn recovers_offset_and_rotation() {
        let truth = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.1); // camera <- IMU
        let mut sync = TimeSync::new(300.0, 300.0, 180.0, 240.0, &UnitQuaternion::identity());
        let omega = |t: f64| {
            Vector3::new(
                0.8 * (2.1 * t).sin(),
                0.6 * (1.3 * t + 1.0).sin(),
                0.5 * (0.7 * t + 2.0).sin(),
            )
        };
        for k in 0..12 * 200 {
            let t = f64::from(k) / 200.0;
            sync.push_gyro((t * 1e9) as i64, omega(t));
        }
        // The frame clock runs 30 ms behind the IMU's.
        let (offset, h) = (0.03, 1.0 / 30.0);
        for k in 0..300 {
            let t0 = 1.0 + f64::from(k) * h;
            let r: Vector3<f64> = (0..50)
                .map(|j| omega(t0 + offset + (f64::from(j) + 0.5) / 50.0 * h) * (h / 50.0))
                .sum();
            sync.intervals.push_back(Interval {
                t0,
                t1: t0 + h,
                visual: truth * r,
            });
        }
        let e = sync.estimate().expect("an estimate");
        assert!((e.offset_ms - 30.0).abs() < 1.0, "offset {}", e.offset_ms);
        assert!(e.correlation > 0.999, "correlation {}", e.correlation);
        assert!(
            e.correlation_assumed < 0.99,
            "assumed {}",
            e.correlation_assumed
        );
        let error = e.imu_from_camera.angle_to(&truth.inverse()).to_degrees();
        assert!(error < 0.2, "rotation off by {error} deg");
        assert!((e.rotation_error_deg - truth.angle().to_degrees()).abs() < 0.2);
        assert!(
            (e.focal_scale - 1.0).abs() < 0.01,
            "focal x{}",
            e.focal_scale
        );
    }
}
