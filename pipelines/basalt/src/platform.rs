//! Moving-platform filter: keeps a vehicle's cabin out of the VIO.
//!
//! On a tram or a funicular the head rests in the cabin, so the cabin's
//! features say the head stands still, while the street outside the windows
//! says it moves. Outside the walking signature the IMU cannot tell the two
//! apart (a vehicle at constant speed feels no acceleration), and the cabin,
//! which fills most of the images, wins: the VIO halves or stops the ride
//! (`docs/lamaria_batch_ba.md`, "A ride the VIO halves").
//!
//! The filter needs neither the estimator nor ground truth. Per camera, every
//! frame, it compares each track's bearing with its bearing [`SPAN`] frames
//! earlier rotated by the integrated gyroscope: a world point at infinity, and
//! a cabin point while the vehicle drives straight, follow the rotation alone.
//! The tracks that move more than [`PARALLAX_DEG`] beyond it are fitted with
//! a common translation direction (RANSAC on their epipolar planes, with the
//! gyro rotation): the street. A frame shows a ride when the accelerometer is
//! quiet (no steps) and enough tracks share that translation; after most of
//! the last [`EVIDENCE_FRAMES`] frames do, the filter passes only the tracks
//! that have been fitted to the street's translation at least once, until the
//! evidence fades or the steps come back. Standing still, nothing moves
//! beyond the rotation, so the filter stays off.

use std::collections::{HashMap, HashSet, VecDeque};

use nalgebra::{Point2, UnitQuaternion, Vector3};

use crate::calibration::BasaltCalibration;
use crate::camera::DoubleSphereCamera;
use crate::types::{CameraId, ImuSample, TrackId, TrackObservation};

/// Frames between the two bearings compared (0.3 s at 20 Hz: the street
/// through a tram's windows leaves the image within a second).
pub const SPAN: u64 = 6;
/// Motion beyond the rotation from which a track shows translation.
pub const PARALLAX_DEG: f64 = 2.0;
/// Epipolar-plane residual (sine of the angle) of a translation inlier.
pub const EPIPOLAR_SIN: f64 = 0.015;
/// Translation inliers a camera needs for ride evidence, in count and share.
pub const MIN_INLIERS: usize = 8;
pub const MIN_INLIER_SHARE: f64 = 0.15;
/// Accelerometer-norm standard deviation (m/s^2, over [`QUIET_WINDOW_S`])
/// below which no steps are taken.
pub const QUIET_STD: f64 = 0.6;
pub const QUIET_WINDOW_S: f64 = 2.0;
/// Frames of evidence considered; a ride starts when [`EVIDENCE_ON`] of them
/// show one and ends when no more than [`EVIDENCE_OFF`] do.
pub const EVIDENCE_FRAMES: usize = 20;
pub const EVIDENCE_ON: usize = 14;
pub const EVIDENCE_OFF: usize = 4;
/// Below this many street observations in a frame the filter passes all.
pub const MIN_PASSED: usize = 10;
const RANSAC_ITERATIONS: usize = 100;

/// The thresholds above; `VISLOC_PLATFORM_PARAMS` (e.g. `span=6,min_inliers=8`)
/// overrides them for experiments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlatformParams {
    pub span: u64,
    pub parallax_deg: f64,
    pub epipolar_sin: f64,
    pub min_inliers: usize,
    pub min_inlier_share: f64,
    pub evidence_frames: usize,
    pub evidence_on: usize,
    pub evidence_off: usize,
    /// A ride ends only when the steps come back, not when the evidence fades.
    pub hold: bool,
    /// During a ride, translation inliers join the street from this many on.
    pub ride_min_inliers: usize,
    /// During a ride, the cabin stays out even below [`MIN_PASSED`].
    pub strict: bool,
}

impl Default for PlatformParams {
    fn default() -> Self {
        Self {
            span: SPAN,
            parallax_deg: PARALLAX_DEG,
            epipolar_sin: EPIPOLAR_SIN,
            min_inliers: MIN_INLIERS,
            min_inlier_share: MIN_INLIER_SHARE,
            evidence_frames: EVIDENCE_FRAMES,
            evidence_on: EVIDENCE_ON,
            evidence_off: EVIDENCE_OFF,
            hold: false,
            ride_min_inliers: usize::MAX,
            strict: false,
        }
    }
}

impl PlatformParams {
    fn from_env() -> Self {
        let mut p = Self::default();
        let Ok(spec) = std::env::var("VISLOC_PLATFORM_PARAMS") else {
            return p;
        };
        for item in spec.split(',') {
            let Some((key, value)) = item.split_once('=') else {
                continue;
            };
            let v: f64 = value.trim().parse().unwrap_or(f64::NAN);
            if !v.is_finite() {
                continue;
            }
            match key.trim() {
                "span" => p.span = v as u64,
                "parallax_deg" => p.parallax_deg = v,
                "epipolar_sin" => p.epipolar_sin = v,
                "min_inliers" => p.min_inliers = v as usize,
                "min_inlier_share" => p.min_inlier_share = v,
                "evidence_frames" => p.evidence_frames = v as usize,
                "evidence_on" => p.evidence_on = v as usize,
                "evidence_off" => p.evidence_off = v as usize,
                "hold" => p.hold = v != 0.0,
                "ride_min_inliers" => p.ride_min_inliers = v as usize,
                "strict" => p.strict = v != 0.0,
                _ => {}
            }
        }
        p
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlatformStats {
    pub frames: u64,
    pub ride_frames: u64,
    pub dropped_observations: u64,
    pub rides: u64,
}

#[derive(Debug, Clone)]
pub struct PlatformFilter {
    p: PlatformParams,
    cameras: Vec<DoubleSphereCamera>,
    imu_from_cam: Vec<UnitQuaternion<f64>>,
    /// World-from-IMU rotation integrated from the gyroscope, per frame.
    rotations: VecDeque<(u64, UnitQuaternion<f64>)>,
    rotation: UnitQuaternion<f64>,
    last_imu_ns: Option<i64>,
    accel: VecDeque<(i64, f64)>,
    /// Unit bearings of each (camera, track), per frame, newest last.
    history: HashMap<(CameraId, TrackId), VecDeque<(u64, Vector3<f64>)>>,
    street: HashSet<TrackId>,
    evidence: VecDeque<bool>,
    ride: bool,
    frame: u64,
    rng: u64,
    stats: PlatformStats,
    /// `VISLOC_PLATFORM_LOG`: one diagnostic line per frame on stderr.
    log: bool,
    last_inliers: [(usize, usize); 2],
}

impl PlatformFilter {
    pub fn new(calibration: &BasaltCalibration) -> Self {
        Self {
            p: PlatformParams::from_env(),
            cameras: calibration.cameras.clone(),
            imu_from_cam: calibration.t_imu_cam.iter().map(|t| t.rotation).collect(),
            rotations: VecDeque::new(),
            rotation: UnitQuaternion::identity(),
            last_imu_ns: None,
            accel: VecDeque::new(),
            history: HashMap::new(),
            street: HashSet::new(),
            evidence: VecDeque::new(),
            ride: false,
            frame: 0,
            rng: 0x9E37_79B9_7F4A_7C15,
            stats: PlatformStats::default(),
            log: std::env::var_os("VISLOC_PLATFORM_LOG").is_some(),
            last_inliers: [(0, 0); 2],
        }
    }

    pub const fn stats(&self) -> &PlatformStats {
        &self.stats
    }

    pub const fn riding(&self) -> bool {
        self.ride
    }

    /// Integrates this frame's IMU samples, classifies the tracks and, during
    /// a ride, removes the observations of tracks not fitted to the street.
    pub fn apply(&mut self, imu: &[ImuSample], observations: &mut Vec<TrackObservation>) {
        self.integrate(imu);
        let frame = self.frame;
        self.frame += 1;
        self.stats.frames += 1;
        self.rotations.push_back((frame, self.rotation));
        while self.rotations.len() > self.p.span as usize + 1 {
            self.rotations.pop_front();
        }
        for observation in observations.iter() {
            let Some(camera) = self.cameras.get(observation.camera_id as usize) else {
                continue;
            };
            let Some(bearing) =
                camera.unproject(&Point2::new(observation.pixel.x, observation.pixel.y))
            else {
                continue;
            };
            let entry = self
                .history
                .entry((observation.camera_id, observation.track_id))
                .or_default();
            entry.push_back((frame, bearing));
            while entry.len() > self.p.span as usize + 1 {
                entry.pop_front();
            }
        }
        self.history
            .retain(|_, entry| entry.back().is_some_and(|(seen, _)| *seen == frame));
        let live: HashSet<TrackId> = self.history.keys().map(|(_, track)| *track).collect();
        self.street.retain(|track| live.contains(track));

        let quiet = self.quiet();
        let mut evidence = false;
        if frame >= self.p.span {
            for camera in 0..self.cameras.len() {
                evidence |= self.classify(camera as CameraId, frame) && quiet;
            }
        }
        self.evidence.push_back(evidence);
        while self.evidence.len() > self.p.evidence_frames {
            self.evidence.pop_front();
        }
        let count = self.evidence.iter().filter(|e| **e).count();
        if !self.ride && quiet && count >= self.p.evidence_on {
            self.ride = true;
            self.stats.rides += 1;
        } else if self.ride && (!quiet || (!self.p.hold && count <= self.p.evidence_off)) {
            self.ride = false;
        }
        if self.log {
            let passed = observations
                .iter()
                .filter(|o| self.street.contains(&o.track_id))
                .count();
            eprintln!(
                "platform frame={frame} t={} quiet={quiet} evidence={count} ride={} obs={} street={passed} cam0={:?} cam1={:?}",
                self.last_imu_ns.unwrap_or(0),
                self.ride,
                observations.len(),
                self.last_inliers[0],
                self.last_inliers[1]
            );
        }
        if self.ride {
            let passed = observations
                .iter()
                .filter(|o| self.street.contains(&o.track_id))
                .count();
            if passed >= MIN_PASSED || self.p.strict {
                self.stats.ride_frames += 1;
                let before = observations.len();
                observations.retain(|o| self.street.contains(&o.track_id));
                self.stats.dropped_observations += (before - observations.len()) as u64;
            }
        }
    }

    fn integrate(&mut self, imu: &[ImuSample]) {
        for sample in imu {
            if let Some(last) = self.last_imu_ns {
                let dt = (sample.timestamp_ns - last) as f64 * 1e-9;
                if dt > 0.0 && dt < 0.1 {
                    self.rotation *= UnitQuaternion::from_scaled_axis(sample.gyro_rad_s * dt);
                }
            }
            if self
                .last_imu_ns
                .map_or(true, |last| sample.timestamp_ns > last)
            {
                self.last_imu_ns = Some(sample.timestamp_ns);
                self.accel
                    .push_back((sample.timestamp_ns, sample.accel_m_s2.norm()));
            }
        }
        let horizon = (QUIET_WINDOW_S * 1e9) as i64;
        if let Some(&(newest, _)) = self.accel.back() {
            while self
                .accel
                .front()
                .is_some_and(|(t, _)| newest - *t > horizon)
            {
                self.accel.pop_front();
            }
        }
    }

    fn quiet(&self) -> bool {
        if self.accel.len() < 50 {
            return false;
        }
        let n = self.accel.len() as f64;
        let mean = self.accel.iter().map(|(_, a)| a).sum::<f64>() / n;
        let var = self
            .accel
            .iter()
            .map(|(_, a)| (a - mean).powi(2))
            .sum::<f64>()
            / n;
        var.sqrt() < QUIET_STD
    }

    /// Fits the street's translation in one camera; true when it is evidence
    /// of a ride. Inliers join the street set.
    fn classify(&mut self, camera: CameraId, frame: u64) -> bool {
        let Some(&(_, then)) = self
            .rotations
            .iter()
            .find(|(f, _)| *f == frame - self.p.span)
        else {
            return false;
        };
        let imu_from_cam = self.imu_from_cam[camera as usize];
        // c(now) <- c(then): R_ic^T R_wi(now)^T R_wi(then) R_ic.
        let rotation = imu_from_cam.inverse() * self.rotation.inverse() * then * imu_from_cam;
        let parallax_cos = self.p.parallax_deg.to_radians().cos();
        let mut evaluated = 0usize;
        let mut planes: Vec<(TrackId, Vector3<f64>)> = Vec::new();
        for ((cam, track), entry) in &self.history {
            if *cam != camera {
                continue;
            }
            let Some(&(first, old)) = entry.front() else {
                continue;
            };
            if first != frame - self.p.span {
                continue;
            }
            let now = entry.back().map(|(_, b)| *b).unwrap_or(old);
            let predicted = rotation * old;
            evaluated += 1;
            let cos = predicted.dot(&now);
            if cos > parallax_cos {
                continue;
            }
            let normal = now.cross(&predicted);
            let norm = normal.norm();
            if norm > 1e-9 {
                planes.push((*track, normal / norm));
            }
        }
        if planes.len() < self.p.min_inliers.max(2) {
            return false;
        }
        let mut best: Vec<usize> = Vec::new();
        for _ in 0..RANSAC_ITERATIONS {
            let a = self.next_index(planes.len());
            let b = self.next_index(planes.len());
            if a == b {
                continue;
            }
            let t = planes[a].1.cross(&planes[b].1);
            let tn = t.norm();
            if tn < 1e-6 {
                continue;
            }
            let t = t / tn;
            let inliers: Vec<usize> = (0..planes.len())
                .filter(|&i| planes[i].1.dot(&t).abs() < self.p.epipolar_sin)
                .collect();
            if inliers.len() > best.len() {
                best = inliers;
            }
        }
        if let Some(slot) = self.last_inliers.get_mut(camera as usize) {
            *slot = (best.len(), evaluated);
        }
        let evidence = best.len() >= self.p.min_inliers
            && best.len() as f64 >= self.p.min_inlier_share * evaluated as f64;
        if evidence || (self.ride && best.len() >= self.p.ride_min_inliers) {
            for &i in &best {
                self.street.insert(planes[i].0);
            }
        }
        evidence
    }

    fn next_index(&mut self, len: usize) -> usize {
        // xorshift64*: deterministic across runs and threads.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        (self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    fn calibration() -> BasaltCalibration {
        let camera = DoubleSphereCamera::new(200.0, 200.0, 200.0, 200.0, 0.0, 0.0, 400, 400)
            .expect("camera");
        BasaltCalibration {
            t_imu_cam: vec![visloc_core::geometry::SE3::identity()],
            cameras: vec![camera],
            resolutions: vec![(400, 400)],
            calib_accel_bias: vec![0.0; 9],
            calib_gyro_bias: vec![0.0; 12],
            imu_update_rate_hz: 200.0,
            accel_noise_std: Vector3::repeat(0.01),
            gyro_noise_std: Vector3::repeat(0.01),
            accel_bias_std: Vector3::repeat(0.01),
            gyro_bias_std: Vector3::repeat(0.01),
            t_mocap_world: visloc_core::geometry::SE3::identity(),
            t_imu_marker: visloc_core::geometry::SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        }
    }

    /// Runs 60 frames: 40 cabin tracks fixed in the image, 40 street points
    /// 6-14 m ahead while the camera moves along x at `speed`. Returns the
    /// filter and the last frame's passed track ids.
    fn run(speed: f64, steps: bool) -> (PlatformFilter, Vec<TrackId>) {
        let calibration = calibration();
        let camera = calibration.cameras[0];
        let mut filter = PlatformFilter::new(&calibration);
        let street: Vec<Point3<f64>> = (0..40)
            .map(|i| {
                let i = i as f64;
                Point3::new(
                    -6.0 + 0.7 * (i % 20.0),
                    -1.5 + 0.4 * (i % 7.0),
                    6.0 + 0.2 * i,
                )
            })
            .collect();
        let mut passed = Vec::new();
        for frame in 0..60u64 {
            let t0 = frame as i64 * 50_000_000;
            let imu: Vec<ImuSample> = (0..10)
                .map(|k| {
                    let t = t0 + k * 5_000_000;
                    let bump = if steps {
                        3.0 * ((t as f64) * 1e-9 * 12.0).sin()
                    } else {
                        0.0
                    };
                    ImuSample::new(t, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81 + bump))
                })
                .collect();
            let x = speed * frame as f64 * 0.05;
            let mut observations: Vec<TrackObservation> = (0..40)
                .map(|i| TrackObservation {
                    track_id: i,
                    frame_id: frame,
                    timestamp_ns: t0,
                    camera_id: 0,
                    pixel: nalgebra::Point2::new(
                        60.0 + 7.0 * i as f64,
                        80.0 + 5.0 * (i % 9) as f64,
                    ),
                })
                .collect();
            for (i, point) in street.iter().enumerate() {
                let relative = Point3::new(point.x - x, point.y, point.z);
                if let Some(pixel) = camera.project(&relative) {
                    observations.push(TrackObservation {
                        track_id: 1000 + i as u64,
                        frame_id: frame,
                        timestamp_ns: t0,
                        camera_id: 0,
                        pixel,
                    });
                }
            }
            filter.apply(&imu, &mut observations);
            passed = observations.iter().map(|o| o.track_id).collect();
        }
        (filter, passed)
    }

    #[test]
    fn a_ride_passes_only_the_street() {
        let (filter, passed) = run(3.0, false);
        assert!(filter.riding());
        assert!(filter.stats().dropped_observations > 0);
        assert!(!passed.is_empty());
        assert!(
            passed.iter().all(|id| *id >= 1000),
            "cabin tracks passed: {passed:?}"
        );
    }

    #[test]
    fn standing_still_passes_everything() {
        let (filter, passed) = run(0.0, false);
        assert!(!filter.riding());
        assert_eq!(filter.stats().dropped_observations, 0);
        assert_eq!(passed.len(), 80);
    }

    #[test]
    fn steps_keep_the_filter_off() {
        let (filter, _) = run(3.0, true);
        assert_eq!(filter.stats().rides, 0);
        assert_eq!(filter.stats().dropped_observations, 0);
    }
}
