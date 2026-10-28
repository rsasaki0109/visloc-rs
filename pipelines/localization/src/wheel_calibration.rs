//! Where a wheeled robot's odometry is measured, from a run with a VIO.
//!
//! Wheel odometry reports a point near the middle of the drive axle, about
//! which the robot turns in place, and keeps that frame level. A robot's
//! description (`T_imu_body`, the body being e.g. `base_link`) may put the
//! body a few centimetres from that point and a degree or two off level.
//! With the camera high above the wheels, either moves the camera pose the
//! wheel odometry implies by centimetres in every turn. And the VIO, with
//! the map built from it, may be a few percent off in scale from the
//! wheels.
//!
//! [`WheelCalibration`] pairs wheel odometry poses with the VIO's IMU poses
//! and fits, in the plane,
//!
//! ```text
//! p_body(t) = s * R(a) * (p_wheel(t) + R(yaw_wheel(t)) * r) + t0
//! ```
//!
//! where `p_body` is the body origin that `T_imu_body` places from the VIO
//! pose, `r` the offset (body frame) of that origin from the wheel odometry's
//! point, and `s`, `a`, `t0` a similarity between the two trajectories.
//! Turns in place are what reveal `r`. It also measures the body frame's
//! tilt against the VIO's gravity, which on a level floor is an error of
//! `T_imu_body`. `scripts/calibrate_wheel_offset.py` fits the same offline.

use std::collections::VecDeque;

use nalgebra::{Matrix2, UnitQuaternion, Vector2, Vector3};
use visloc_core::geometry::SE3;

#[derive(Debug, Clone, PartialEq)]
pub struct WheelCalibrationConfig {
    /// A wheel pose pairs with the nearest VIO pose at most this far away.
    pub max_gap_ns: i64,
    /// The estimate is ready once the wheel odometry has turned this many
    /// degrees in total (which shows the offset) ...
    pub min_turn_deg: f64,
    /// ... and travelled this many metres (which shows the scale; turning
    /// in place alone cannot tell a larger offset from a larger scale) ...
    pub min_travel_m: f64,
    /// ... over at least this many pairs.
    pub min_pairs: usize,
    /// Level the body frame against the VIO's gravity (a level floor).
    pub level: bool,
}

impl Default for WheelCalibrationConfig {
    fn default() -> Self {
        Self {
            max_gap_ns: 20_000_000,
            min_turn_deg: 180.0,
            min_travel_m: 1.0,
            min_pairs: 200,
            level: true,
        }
    }
}

/// What [`WheelCalibration::estimate`] found.
#[derive(Debug, Clone, PartialEq)]
pub struct WheelCalibrationEstimate {
    /// Offset, in the body frame and the wheel odometry's units, of the
    /// body origin from the wheel odometry's point.
    pub offset: Vector2<f64>,
    /// The VIO's scale against the wheel odometry.
    pub scale: f64,
    /// Rotation from the levelled body frame to the given one (identity
    /// without levelling).
    pub body_from_level: UnitQuaternion<f64>,
    /// Tilt of the given body frame against the VIO's gravity, in degrees.
    pub tilt_deg: f64,
    /// RMS residual of the fit, in metres of the VIO, and without the
    /// offset.
    pub rms_m: f64,
    pub rms_without_offset_m: f64,
    /// Total turn of the wheel odometry, in degrees, and the number of pairs.
    pub turned_deg: f64,
    pub pairs: usize,
}

impl WheelCalibrationEstimate {
    /// `T_imu_body` with the body moved to the wheel odometry's point and
    /// levelled (the wheel odometry's own frame).
    pub fn imu_from_wheel(&self, imu_from_body: &SE3) -> SE3 {
        let rotation = imu_from_body.rotation * self.body_from_level;
        let offset = Vector3::new(self.offset.x, self.offset.y, 0.0);
        SE3::new(rotation, imu_from_body.translation - rotation * offset)
    }

    /// The body's pose in the odometry frame, in the VIO's scale, from a
    /// wheel odometry pose (of the wheel odometry's point, level).
    pub fn odom_from_body(&self, odom_from_wheel: &SE3) -> SE3 {
        let offset = Vector3::new(self.offset.x, self.offset.y, 0.0);
        SE3::new(
            odom_from_wheel.rotation * self.body_from_level.inverse(),
            self.scale * (odom_from_wheel.translation + odom_from_wheel.rotation * offset),
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct Pair {
    body: Vector2<f64>,
    wheel: Vector2<f64>,
    wheel_yaw: f64,
    up_in_body: Vector3<f64>,
}

/// Pairs wheel odometry with VIO poses as they arrive and fits where the
/// wheel odometry is measured.
#[derive(Debug, Clone)]
pub struct WheelCalibration {
    config: WheelCalibrationConfig,
    imu_from_body: SE3,
    /// Recent VIO poses (`world_from_imu`), oldest first.
    vio: VecDeque<(i64, SE3)>,
    /// Wheel poses waiting for a later VIO pose.
    waiting: VecDeque<(i64, SE3)>,
    pairs: Vec<Pair>,
    turned_rad: f64,
    travel_m: f64,
}

impl WheelCalibration {
    pub fn new(config: WheelCalibrationConfig, imu_from_body: SE3) -> Self {
        Self {
            config,
            imu_from_body,
            vio: VecDeque::new(),
            waiting: VecDeque::new(),
            pairs: Vec::new(),
            turned_rad: 0.0,
            travel_m: 0.0,
        }
    }

    pub fn pairs(&self) -> usize {
        self.pairs.len()
    }

    pub fn turned_deg(&self) -> f64 {
        self.turned_rad.to_degrees()
    }

    /// The wheel odometry's path length over the pairs, in its own units.
    pub fn travel_m(&self) -> f64 {
        self.travel_m
    }

    /// Whether enough turning and travel have been seen for
    /// [`Self::estimate`].
    pub fn is_ready(&self) -> bool {
        self.pairs.len() >= self.config.min_pairs.max(3)
            && self.turned_deg() >= self.config.min_turn_deg
            && self.travel_m >= self.config.min_travel_m
    }

    /// Adds a VIO pose (the IMU in the VIO's gravity-aligned world), in time
    /// order.
    pub fn add_vio(&mut self, timestamp_ns: i64, world_from_imu: &SE3) {
        if self
            .vio
            .back()
            .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            return;
        }
        self.vio.push_back((timestamp_ns, world_from_imu.clone()));
        while let Some((t, _)) = self.waiting.front() {
            if *t > timestamp_ns {
                break;
            }
            let (t, wheel) = self.waiting.pop_front().expect("front exists");
            self.pair(t, &wheel);
        }
        // Keep what the waiting wheel poses may still need.
        let keep_from = self
            .waiting
            .front()
            .map_or(timestamp_ns, |(t, _)| (*t).min(timestamp_ns))
            - self.config.max_gap_ns;
        while self.vio.len() > 2 && self.vio[1].0 < keep_from {
            self.vio.pop_front();
        }
    }

    /// Adds a wheel odometry pose (its point, level), in time order.
    pub fn add_wheel(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) {
        if self
            .vio
            .back()
            .is_some_and(|(last, _)| timestamp_ns <= *last)
        {
            self.pair(timestamp_ns, odom_from_wheel);
        } else {
            self.waiting
                .push_back((timestamp_ns, odom_from_wheel.clone()));
        }
    }

    fn pair(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) {
        let Some((t, world_from_imu)) = self
            .vio
            .iter()
            .min_by_key(|(t, _)| (t - timestamp_ns).abs())
        else {
            return;
        };
        if (t - timestamp_ns).abs() > self.config.max_gap_ns {
            return;
        }
        let body =
            world_from_imu.translation + world_from_imu.rotation * self.imu_from_body.translation;
        let world_from_body = world_from_imu.rotation * self.imu_from_body.rotation;
        let up_in_body = world_from_body.inverse() * Vector3::z();
        let wheel_yaw = yaw_of(&odom_from_wheel.rotation);
        if let Some(previous) = self.pairs.last() {
            self.turned_rad += wrap_angle(wheel_yaw - previous.wheel_yaw).abs();
            self.travel_m += (odom_from_wheel.translation.xy() - previous.wheel).norm();
        }
        self.pairs.push(Pair {
            body: body.xy(),
            wheel: odom_from_wheel.translation.xy(),
            wheel_yaw,
            up_in_body,
        });
    }

    /// The fit over every pair so far; `None` with fewer than three pairs.
    pub fn estimate(&self) -> Option<WheelCalibrationEstimate> {
        if self.pairs.len() < 3 {
            return None;
        }
        let (offset, scale, rms) = fit_offset(&self.pairs, 30);
        let (_, _, rms_without_offset) = fit_offset(&self.pairs, 0);
        let up = self
            .pairs
            .iter()
            .map(|pair| pair.up_in_body)
            .sum::<Vector3<f64>>()
            .normalize();
        let tilt_deg = up.z.clamp(-1.0, 1.0).acos().to_degrees();
        let body_from_level = if self.config.level {
            UnitQuaternion::rotation_between(&Vector3::z(), &up)
                .unwrap_or_else(UnitQuaternion::identity)
        } else {
            UnitQuaternion::identity()
        };
        Some(WheelCalibrationEstimate {
            offset,
            scale,
            body_from_level,
            tilt_deg,
            rms_m: rms,
            rms_without_offset_m: rms_without_offset,
            turned_deg: self.turned_deg(),
            pairs: self.pairs.len(),
        })
    }
}

fn yaw_of(rotation: &UnitQuaternion<f64>) -> f64 {
    let matrix = rotation.to_rotation_matrix();
    matrix[(1, 0)].atan2(matrix[(0, 0)])
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

fn rotation2(angle: f64) -> Matrix2<f64> {
    let (s, c) = angle.sin_cos();
    Matrix2::new(c, -s, s, c)
}

/// Offset `r`, scale and RMS residual of
/// `body ~ s R(a) (wheel + R(yaw) r) + t0`, alternating the similarity and
/// the (linear) offset `iterations` times; 0 fits the similarity alone.
fn fit_offset(pairs: &[Pair], iterations: usize) -> (Vector2<f64>, f64, f64) {
    let n = pairs.len() as f64;
    let mut offset = Vector2::zeros();
    let similarity = |offset: &Vector2<f64>| {
        let q: Vec<Vector2<f64>> = pairs
            .iter()
            .map(|pair| pair.wheel + rotation2(pair.wheel_yaw) * offset)
            .collect();
        let mean_q = q.iter().sum::<Vector2<f64>>() / n;
        let mean_g = pairs.iter().map(|pair| pair.body).sum::<Vector2<f64>>() / n;
        let h = q
            .iter()
            .zip(pairs)
            .map(|(q, pair)| (q - mean_q) * (pair.body - mean_g).transpose())
            .sum::<Matrix2<f64>>();
        let rotation = rotation2((h[(0, 1)] - h[(1, 0)]).atan2(h[(0, 0)] + h[(1, 1)]));
        let spread: f64 = q.iter().map(|q| (q - mean_q).norm_squared()).sum();
        let scale = if spread > 0.0 {
            q.iter()
                .zip(pairs)
                .map(|(q, pair)| (pair.body - mean_g).dot(&(rotation * (q - mean_q))))
                .sum::<f64>()
                / spread
        } else {
            1.0
        };
        let translation = mean_g - scale * rotation * mean_q;
        (rotation, scale, translation, q)
    };
    for _ in 0..iterations {
        let (rotation, scale, translation, _) = similarity(&offset);
        // body - s R wheel - t0 = s R R(yaw) r, linear in r.
        let (mut normal, mut rhs) = (Matrix2::zeros(), Vector2::zeros());
        for pair in pairs {
            let a = scale * rotation * rotation2(pair.wheel_yaw);
            let b = pair.body - scale * rotation * pair.wheel - translation;
            normal += a.transpose() * a;
            rhs += a.transpose() * b;
        }
        match normal.try_inverse() {
            Some(inverse) => offset = inverse * rhs,
            None => break,
        }
    }
    let (rotation, scale, translation, q) = similarity(&offset);
    let squared: f64 = q
        .iter()
        .zip(pairs)
        .map(|(q, pair)| (pair.body - (scale * rotation * q + translation)).norm_squared())
        .sum();
    (offset, scale, (squared / n).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn rz(angle: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
    }

    /// A drive: 2 m ahead, a half turn in place, 2 m back, a quarter turn.
    fn drive() -> Vec<(f64, f64, f64)> {
        let (mut x, mut y, mut yaw, mut poses) = (0.0, 0.0, 0.0, Vec::new());
        for (steps, speed, turn) in [
            (100, 0.02, 0.0),
            (90, 0.0, std::f64::consts::PI / 90.0),
            (100, 0.02, 0.0),
            (45, 0.0, std::f64::consts::PI / 90.0),
        ] {
            for _ in 0..steps {
                x += speed * f64::cos(yaw);
                y += speed * f64::sin(yaw);
                yaw += turn;
                poses.push((x, y, yaw));
            }
        }
        poses
    }

    /// The true `T_imu_body`: the IMU 0.9 m above the body, rotated.
    fn imu_from_body() -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(-1.55, 0.02, -1.6),
            Vector3::new(0.03, -0.89, -0.25),
        )
    }

    /// Feeds a run whose body origin is `offset` from the wheel point and
    /// whose VIO is `scale` times the wheels; returns the calibration.
    fn run(offset: Vector2<f64>, scale: f64, given: &SE3) -> WheelCalibration {
        let truth = imu_from_body();
        let world = SE3::new(rz(0.4), Vector3::new(0.3, -1.2, 0.5));
        let mut calibration =
            WheelCalibration::new(WheelCalibrationConfig::default(), given.clone());
        for (index, (x, y, yaw)) in drive().into_iter().enumerate() {
            let wheel = SE3::new(rz(yaw), Vector3::new(x, y, 0.0));
            let body_position = scale
                * (wheel.translation + wheel.rotation * Vector3::new(offset.x, offset.y, 0.0));
            let world_from_body = world.compose(&SE3::new(wheel.rotation, body_position));
            let world_from_imu = world_from_body.compose(&truth.inverse());
            let t = 1_000 * MS + index as i64 * 50 * MS;
            // The VIO runs 2 ms behind the wheels' stamps and arrives first
            // half the time.
            if index % 2 == 0 {
                calibration.add_vio(t + 2 * MS, &world_from_imu);
                calibration.add_wheel(t, &wheel);
            } else {
                calibration.add_wheel(t, &wheel);
                calibration.add_vio(t + 2 * MS, &world_from_imu);
            }
        }
        calibration
    }

    #[test]
    fn recovers_the_offset_scale_and_level() {
        let offset = Vector2::new(0.041, 0.064);
        let calibration = run(offset, 1.03, &imu_from_body());
        assert!(calibration.is_ready());
        let estimate = calibration.estimate().expect("estimate");
        assert!((estimate.offset - offset).norm() < 1e-6, "{estimate:?}");
        assert!((estimate.scale - 1.03).abs() < 1e-6);
        assert!(estimate.rms_m < 1e-6);
        assert!(estimate.rms_without_offset_m > 0.02);
        assert!(estimate.tilt_deg < 1e-6);
        assert!(
            (estimate.turned_deg - 270.0).abs() < 2.5,
            "{}",
            estimate.turned_deg
        );
        assert_eq!(estimate.pairs, drive().len());
    }

    #[test]
    fn a_tilted_extrinsic_is_levelled_and_the_body_follows_the_wheels() {
        let truth = imu_from_body();
        let tilt = UnitQuaternion::from_euler_angles(0.0, 2f64.to_radians(), 0.0);
        let given = SE3::new(truth.rotation * tilt, truth.translation);
        let offset = Vector2::new(0.013, 0.062);
        let estimate = run(offset, 1.0, &given).estimate().expect("estimate");
        assert!((estimate.tilt_deg - 2.0).abs() < 1e-6);
        // Levelled, the given extrinsic becomes the true one (no turn about z).
        let imu_from_wheel = estimate.imu_from_wheel(&given);
        assert!(imu_from_wheel.rotation.angle_to(&truth.rotation) < 1e-9);
        let expected = truth.translation - truth.rotation * Vector3::new(offset.x, offset.y, 0.0);
        assert!((imu_from_wheel.translation - expected).norm() < 1e-6);
        // The body pose from a wheel pose puts the IMU where the given
        // extrinsic says, as the VIO saw it.
        let wheel = SE3::new(rz(0.7), Vector3::new(1.0, 2.0, 0.0));
        let odom_from_body = estimate.odom_from_body(&wheel);
        let imu_from_wheel_pose = wheel.compose(&imu_from_wheel.inverse());
        let imu_from_body_pose = odom_from_body.compose(&given.inverse());
        assert!((imu_from_wheel_pose.translation - imu_from_body_pose.translation).norm() < 1e-6);
        assert!(
            imu_from_wheel_pose
                .rotation
                .angle_to(&imu_from_body_pose.rotation)
                < 1e-9
        );
    }

    #[test]
    fn not_ready_before_enough_turning() {
        let mut calibration =
            WheelCalibration::new(WheelCalibrationConfig::default(), imu_from_body());
        for index in 0..300 {
            let t = index as i64 * 50 * MS;
            let pose = SE3::new(rz(0.0), Vector3::new(index as f64 * 0.02, 0.0, 0.0));
            calibration.add_vio(t, &pose.compose(&imu_from_body().inverse()));
            calibration.add_wheel(t, &pose);
        }
        assert_eq!(calibration.pairs(), 300);
        assert!(calibration.travel_m() > 5.9);
        assert!(!calibration.is_ready());
        // Wheel poses without a VIO pose close enough are not paired.
        calibration.add_wheel(400 * 50 * MS, &SE3::identity());
        calibration.add_vio(401 * 50 * MS, &SE3::identity());
        assert_eq!(calibration.pairs(), 300);
    }
}
