//! Wheel odometry calibrated against the VIO while the robot drives, for
//! `visloc_wheel_calibration_node`.
//!
//! [`WheelOdomCalibrator`] passes the wheel odometry through until the
//! robot has turned enough for `visloc_localization::WheelCalibration` to
//! say where the wheel odometry is measured, then fixes that estimate and
//! turns every wheel pose into the body's pose at the VIO's scale: the body
//! where the robot's description puts it, levelled against the VIO's
//! gravity. Where the estimate takes over, a yaw and a translation keep the
//! output continuous, so a consumer such as `visloc_map_localize_node` sees
//! no jump. A saved estimate ([`SavedCalibration`]) applies from the start.

use nalgebra::{Quaternion, UnitQuaternion, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use visloc_core::geometry::SE3;
use visloc_localization::{WheelCalibration, WheelCalibrationConfig, WheelCalibrationEstimate};

/// A [`WheelCalibrationEstimate`] as written to `calibration_file` (JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedCalibration {
    /// Offset of the body origin from the wheel odometry's point, in the
    /// body frame (m).
    pub offset_m: [f64; 2],
    /// The VIO's scale against the wheel odometry.
    pub scale: f64,
    /// Rotation from the levelled body frame to the robot description's,
    /// as x, y, z, w.
    pub body_from_level_xyzw: [f64; 4],
    pub tilt_deg: f64,
    pub rms_m: f64,
    pub turned_deg: f64,
    pub pairs: usize,
}

impl From<&WheelCalibrationEstimate> for SavedCalibration {
    fn from(estimate: &WheelCalibrationEstimate) -> Self {
        let q = estimate.body_from_level.quaternion();
        Self {
            offset_m: [estimate.offset.x, estimate.offset.y],
            scale: estimate.scale,
            body_from_level_xyzw: [q.i, q.j, q.k, q.w],
            tilt_deg: estimate.tilt_deg,
            rms_m: estimate.rms_m,
            turned_deg: estimate.turned_deg,
            pairs: estimate.pairs,
        }
    }
}

impl SavedCalibration {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("plain numbers serialize")
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let saved: Self =
            serde_json::from_str(text).map_err(|error| format!("calibration file: {error}"))?;
        let [x, y, z, w] = saved.body_from_level_xyzw;
        let norm = (x * x + y * y + z * z + w * w).sqrt();
        if !(saved.scale > 0.0 && saved.scale.is_finite() && norm > 0.5 && norm < 2.0) {
            return Err(
                "calibration file: scale must be positive and the quaternion a rotation".into(),
            );
        }
        Ok(saved)
    }

    pub fn estimate(&self) -> WheelCalibrationEstimate {
        let [x, y, z, w] = self.body_from_level_xyzw;
        WheelCalibrationEstimate {
            offset: Vector2::new(self.offset_m[0], self.offset_m[1]),
            scale: self.scale,
            body_from_level: UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)),
            tilt_deg: self.tilt_deg,
            rms_m: self.rms_m,
            rms_without_offset_m: f64::NAN,
            turned_deg: self.turned_deg,
            pairs: self.pairs,
        }
    }
}

/// What [`WheelOdomCalibrator::wheel`] did with a wheel pose.
#[derive(Clone, Debug, PartialEq)]
pub enum WheelOutput {
    /// Not calibrated yet: the wheel pose, unchanged.
    PassThrough(SE3),
    /// The body's pose from the calibration.
    Calibrated(SE3),
    /// The first calibrated pose, with the estimate just fixed.
    Fixed(SE3, WheelCalibrationEstimate),
}

impl WheelOutput {
    pub fn pose(&self) -> &SE3 {
        match self {
            Self::PassThrough(pose) | Self::Calibrated(pose) | Self::Fixed(pose, _) => pose,
        }
    }
}

#[derive(Debug, Clone)]
enum State {
    Collecting(Box<WheelCalibration>),
    Fixed {
        estimate: WheelCalibrationEstimate,
        /// Yaw + translation keeping the output continuous.
        continuity: SE3,
    },
}

#[derive(Debug, Clone)]
pub struct WheelOdomCalibrator {
    state: State,
    /// Whether the output takes the VIO's scale.
    apply_scale: bool,
}

impl WheelOdomCalibrator {
    /// Collects pairs, `imu_from_body` being the robot description's
    /// transform from the odometry's body (e.g. `base_link`) to the VIO's IMU.
    pub fn new(config: WheelCalibrationConfig, imu_from_body: SE3) -> Self {
        Self {
            state: State::Collecting(Box::new(WheelCalibration::new(config, imu_from_body))),
            apply_scale: true,
        }
    }

    /// Whether the output takes the VIO's scale (the default) or keeps the
    /// wheels'. The VIO's scale stands for the maps' (built by the same
    /// VIO) but varies by a few percent between sessions.
    #[must_use]
    pub const fn with_scale(mut self, apply_scale: bool) -> Self {
        self.apply_scale = apply_scale;
        self
    }

    /// Applies a saved estimate from the first pose on.
    pub fn from_estimate(estimate: WheelCalibrationEstimate) -> Self {
        Self {
            state: State::Fixed {
                estimate,
                continuity: SE3::identity(),
            },
            apply_scale: true,
        }
    }

    pub fn estimate(&self) -> Option<&WheelCalibrationEstimate> {
        match &self.state {
            State::Collecting(_) => None,
            State::Fixed { estimate, .. } => Some(estimate),
        }
    }

    /// Pairs and turn so far, while collecting.
    pub fn progress(&self) -> Option<(usize, f64)> {
        match &self.state {
            State::Collecting(calibration) => Some((calibration.pairs(), calibration.turned_deg())),
            State::Fixed { .. } => None,
        }
    }

    pub fn add_vio(&mut self, timestamp_ns: i64, world_from_imu: &SE3) {
        if let State::Collecting(calibration) = &mut self.state {
            calibration.add_vio(timestamp_ns, world_from_imu);
        }
    }

    pub fn wheel(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) -> WheelOutput {
        match &mut self.state {
            State::Fixed {
                estimate,
                continuity,
            } => WheelOutput::Calibrated(continuity.compose(&body_pose(
                estimate,
                odom_from_wheel,
                self.apply_scale,
            ))),
            State::Collecting(calibration) => {
                calibration.add_wheel(timestamp_ns, odom_from_wheel);
                if !calibration.is_ready() {
                    return WheelOutput::PassThrough(odom_from_wheel.clone());
                }
                let Some(estimate) = calibration.estimate() else {
                    return WheelOutput::PassThrough(odom_from_wheel.clone());
                };
                let calibrated = body_pose(&estimate, odom_from_wheel, self.apply_scale);
                let continuity = yaw_translation_between(odom_from_wheel, &calibrated);
                let pose = continuity.compose(&calibrated);
                self.state = State::Fixed {
                    estimate: estimate.clone(),
                    continuity,
                };
                WheelOutput::Fixed(pose, estimate)
            }
        }
    }
}

/// The body's pose from a wheel pose, at the VIO's scale or the wheels'.
fn body_pose(estimate: &WheelCalibrationEstimate, odom_from_wheel: &SE3, apply_scale: bool) -> SE3 {
    if apply_scale {
        return estimate.odom_from_body(odom_from_wheel);
    }
    let unscaled = WheelCalibrationEstimate {
        scale: 1.0,
        ..estimate.clone()
    };
    unscaled.odom_from_body(odom_from_wheel)
}

/// The yaw + translation `T` with `T * to` matching `from`'s position and
/// heading (both frames gravity-aligned).
fn yaw_translation_between(from: &SE3, to: &SE3) -> SE3 {
    let yaw = |rotation: &UnitQuaternion<f64>| {
        let x = rotation * Vector3::x();
        x.y.atan2(x.x)
    };
    let rotation = UnitQuaternion::from_axis_angle(
        &Vector3::z_axis(),
        yaw(&from.rotation) - yaw(&to.rotation),
    );
    SE3::new(rotation, from.translation - rotation * to.translation)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: i64 = 1_000_000;

    fn rz(angle: f64) -> UnitQuaternion<f64> {
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle)
    }

    fn imu_from_body() -> SE3 {
        SE3::new(
            UnitQuaternion::from_euler_angles(-1.55, 0.02, -1.6),
            Vector3::new(0.03, -0.89, -0.25),
        )
    }

    /// Wheel poses driving 1 m, turning in place, then driving 1 m back,
    /// with VIO poses of a body 5 cm ahead of the wheel point and a VIO 2%
    /// long.
    fn feed(calibrator: &mut WheelOdomCalibrator) -> Vec<WheelOutput> {
        let world = SE3::new(rz(0.3), Vector3::new(1.0, -2.0, 0.4));
        let mut outputs = Vec::new();
        let mut steps = Vec::new();
        for i in 0..50 {
            steps.push((i as f64 * 0.02, 0.0, 0.0));
        }
        for i in 0..=60 {
            steps.push((1.0, 0.0, i as f64 * std::f64::consts::PI / 60.0));
        }
        for i in 1..=50 {
            steps.push((1.0 - i as f64 * 0.02, 0.0, std::f64::consts::PI));
        }
        for (index, (x, y, yaw)) in steps.into_iter().enumerate() {
            let t = index as i64 * 50 * MS;
            let wheel = SE3::new(rz(yaw), Vector3::new(x, y, 0.0));
            let body = SE3::new(
                wheel.rotation,
                1.02 * (wheel.translation + wheel.rotation * Vector3::new(0.05, 0.0, 0.0)),
            );
            calibrator.add_vio(t, &world.compose(&body).compose(&imu_from_body().inverse()));
            outputs.push(calibrator.wheel(t, &wheel));
        }
        outputs
    }

    #[test]
    fn passes_through_then_calibrates_without_a_jump() {
        let config = WheelCalibrationConfig {
            min_turn_deg: 180.0,
            min_pairs: 50,
            ..WheelCalibrationConfig::default()
        };
        let mut calibrator = WheelOdomCalibrator::new(config, imu_from_body());
        let outputs = feed(&mut calibrator);
        let fixed = outputs
            .iter()
            .position(|output| matches!(output, WheelOutput::Fixed(..)))
            .expect("calibrated");
        assert!(outputs[..fixed]
            .iter()
            .all(|o| matches!(o, WheelOutput::PassThrough(_))));
        assert!(outputs[fixed + 1..]
            .iter()
            .all(|o| matches!(o, WheelOutput::Calibrated(_))));
        let estimate = calibrator.estimate().expect("fixed").clone();
        assert!((estimate.offset - Vector2::new(0.05, 0.0)).norm() < 1e-6);
        assert!((estimate.scale - 1.02).abs() < 1e-6);
        // No jump where the estimate takes over: the step there is the
        // wheel's own step (here a turn in place: the body point swings).
        let (before, after) = (outputs[fixed - 1].pose(), outputs[fixed].pose());
        assert!((after.translation - before.translation).norm() < 0.01);
        // Driving 1 m afterwards moves the output 1.02 m (the VIO's scale).
        let last = outputs.last().unwrap().pose();
        let start = outputs[outputs.len() - 51].pose();
        assert!(((last.translation - start.translation).norm() - 1.02).abs() < 1e-6);
        // The odometry frame stays gravity-aligned.
        assert!((last.rotation * Vector3::z() - Vector3::z()).norm() < 1e-9);
    }

    #[test]
    fn a_saved_calibration_round_trips_and_applies_from_the_start() {
        let config = WheelCalibrationConfig {
            min_pairs: 50,
            ..WheelCalibrationConfig::default()
        };
        let mut calibrator = WheelOdomCalibrator::new(config, imu_from_body());
        feed(&mut calibrator);
        let saved = SavedCalibration::from(calibrator.estimate().expect("fixed"));
        let loaded = SavedCalibration::from_json(&saved.to_json()).expect("parses");
        assert_eq!(loaded, saved);
        let mut from_file = WheelOdomCalibrator::from_estimate(loaded.estimate());
        let wheel = SE3::new(rz(0.5), Vector3::new(1.0, 2.0, 0.0));
        let WheelOutput::Calibrated(pose) = from_file.wheel(0, &wheel) else {
            panic!("calibrated from the start");
        };
        let expected = calibrator.estimate().unwrap().odom_from_body(&wheel);
        assert!((pose.translation - expected.translation).norm() < 1e-9);
        // Without the scale, 1 m of wheel travel stays 1 m.
        let mut unscaled = WheelOdomCalibrator::from_estimate(loaded.estimate()).with_scale(false);
        let a = unscaled.wheel(0, &SE3::identity());
        let b = unscaled.wheel(1, &SE3::new(rz(0.0), Vector3::new(1.0, 0.0, 0.0)));
        assert!(((b.pose().translation - a.pose().translation).norm() - 1.0).abs() < 1e-9);
        assert!(SavedCalibration::from_json("{}").is_err());
        let mut bad = saved.clone();
        bad.scale = 0.0;
        assert!(SavedCalibration::from_json(&bad.to_json()).is_err());
    }
}
