//! Wheel odometry calibrated while the robot drives, for
//! `visloc_wheel_calibration_node`.
//!
//! [`WheelOdomCalibrator`] passes the wheel odometry through until
//! `visloc_localization::WheelCalibration` has an estimate of where the
//! wheel odometry is measured, from the VIO (the offset, once the robot has
//! turned enough) and from map localizations (the map's scale and level),
//! then turns every wheel pose into the body's pose: the body where the
//! robot's description puts it, levelled, at the map's scale. It
//! refits every few seconds as pairs come in. Wherever an estimate takes
//! over, a yaw and a translation keep the output continuous, so a consumer
//! such as `visloc_map_localize_node` sees no jump. A saved estimate
//! ([`SavedCalibration`]) applies from the start.

use nalgebra::{Quaternion, UnitQuaternion, Vector2, Vector3};
use serde::{Deserialize, Serialize};
use visloc_core::geometry::SE3;
use visloc_localization::{
    CalibrationSource, WheelCalibration, WheelCalibrationConfig, WheelCalibrationEstimate,
};

use crate::msgs::{Odometry, TfMessage, Transform, TransformStamped, Vector3 as RosVector3};

/// A [`WheelCalibrationEstimate`] as written to `calibration_file` (JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedCalibration {
    /// Offset of the body origin from the wheel odometry's point, in the
    /// body frame (m).
    pub offset_m: [f64; 2],
    /// The map's scale against the wheel odometry.
    pub scale: f64,
    /// Rotation from the levelled body frame to the robot description's,
    /// as x, y, z, w.
    pub body_from_level_xyzw: [f64; 4],
    pub tilt_deg: f64,
    pub rms_m: f64,
    pub turned_deg: f64,
    pub pairs: usize,
    /// Map localizations used, and where each value came from (`vio`,
    /// `map`, `prior` or `default`; empty in files from before).
    #[serde(default)]
    pub fixes: usize,
    #[serde(default)]
    pub offset_from: String,
    #[serde(default)]
    pub scale_from: String,
    #[serde(default)]
    pub level_from: String,
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
            fixes: estimate.fixes,
            offset_from: estimate.offset_from.name().into(),
            scale_from: estimate.scale_from.name().into(),
            level_from: estimate.level_from.name().into(),
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

    /// The saved values as a prior: every value from [`CalibrationSource::Prior`].
    pub fn estimate(&self) -> WheelCalibrationEstimate {
        let [x, y, z, w] = self.body_from_level_xyzw;
        WheelCalibrationEstimate {
            offset: Vector2::new(self.offset_m[0], self.offset_m[1]),
            scale: self.scale,
            body_from_level: UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z)),
            tilt_deg: self.tilt_deg,
            rms_m: self.rms_m,
            turned_deg: self.turned_deg,
            pairs: self.pairs,
            fixes: self.fixes,
            offset_from: CalibrationSource::Prior,
            scale_from: CalibrationSource::Prior,
            level_from: CalibrationSource::Prior,
            ..WheelCalibrationEstimate::identity()
        }
    }
}

/// What [`WheelOdomCalibrator::wheel`] did with a wheel pose.
#[derive(Clone, Debug, PartialEq)]
pub enum WheelOutput {
    /// No estimate yet: the wheel pose, unchanged.
    PassThrough(SE3),
    /// The body's pose from the estimate in use.
    Calibrated(SE3),
    /// The body's pose from a new estimate, which applies from here on.
    Updated(SE3, WheelCalibrationEstimate),
}

impl WheelOutput {
    pub fn pose(&self) -> &SE3 {
        match self {
            Self::PassThrough(pose) | Self::Calibrated(pose) | Self::Updated(pose, _) => pose,
        }
    }
}

/// How far the pairing has got.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationProgress {
    /// Pairs with the VIO, and the wheel odometry's turn over them.
    pub pairs: usize,
    pub turned_deg: f64,
    /// Map localizations paired, and the wheel odometry's path over them.
    pub fixes: usize,
    pub fix_travel_m: f64,
}

#[derive(Debug, Clone)]
pub struct WheelOdomCalibrator {
    calibration: WheelCalibration,
    /// The estimate in use, with the yaw + translation keeping the output
    /// continuous.
    current: Option<(WheelCalibrationEstimate, SE3)>,
    /// Whether the output takes the map's scale.
    apply_scale: bool,
    /// Refit this often (by wheel stamp) once an estimate is in use; 0
    /// keeps the first.
    refine_ns: i64,
    next_refit_ns: Option<i64>,
}

impl WheelOdomCalibrator {
    /// Collects map localizations, and VIO poses once
    /// [`Self::set_imu_from_body`] says where the VIO's IMU sits on the body.
    pub fn new(config: WheelCalibrationConfig) -> Self {
        Self {
            calibration: WheelCalibration::without_vio(config),
            current: None,
            apply_scale: true,
            refine_ns: 0,
            next_refit_ns: None,
        }
    }

    /// Applies `prior` (e.g. a saved estimate) from the first pose on, and
    /// keeps its values until the pairs show better ones.
    #[must_use]
    pub fn with_prior(mut self, prior: WheelCalibrationEstimate) -> Self {
        self.calibration = self.calibration.with_prior(prior.clone());
        self.current = Some((prior, SE3::identity()));
        self
    }

    /// Whether the output takes the map's scale, from the map
    /// localizations (the default), or keeps the wheels'.
    #[must_use]
    pub const fn with_scale(mut self, apply_scale: bool) -> Self {
        self.apply_scale = apply_scale;
        self
    }

    /// Refits every `refine_s` seconds once an estimate is in use (0, the
    /// default, keeps the first estimate, or the prior).
    #[must_use]
    pub fn with_refine_s(mut self, refine_s: f64) -> Self {
        self.refine_ns = (refine_s.max(0.0) * 1e9) as i64;
        self
    }

    /// The `T_imu_body` of the VIO's IMU, from the robot's description.
    pub fn set_imu_from_body(&mut self, imu_from_body: SE3) {
        self.calibration.set_imu_from_body(imu_from_body);
    }

    pub const fn has_imu_from_body(&self) -> bool {
        self.calibration.has_imu_from_body()
    }

    /// The estimate in use.
    pub fn estimate(&self) -> Option<&WheelCalibrationEstimate> {
        self.current.as_ref().map(|(estimate, _)| estimate)
    }

    pub fn progress(&self) -> CalibrationProgress {
        CalibrationProgress {
            pairs: self.calibration.pairs(),
            turned_deg: self.calibration.turned_deg(),
            fixes: self.calibration.fixes(),
            fix_travel_m: self.calibration.fix_travel_m(),
        }
    }

    pub fn add_vio(&mut self, timestamp_ns: i64, world_from_imu: &SE3) {
        self.calibration.add_vio(timestamp_ns, world_from_imu);
    }

    /// A localization against the map: the body's pose in the map.
    pub fn add_fix(&mut self, timestamp_ns: i64, map_from_body: &SE3) {
        self.calibration.add_fix(timestamp_ns, map_from_body);
    }

    pub fn wheel(&mut self, timestamp_ns: i64, odom_from_wheel: &SE3) -> WheelOutput {
        self.calibration.add_wheel(timestamp_ns, odom_from_wheel);
        let due = match (&self.current, self.next_refit_ns) {
            (None, _) => true,
            (Some(_), Some(next)) => timestamp_ns >= next,
            (Some(_), None) if self.refine_ns > 0 => {
                // A prior in use: the first refit one period in.
                self.next_refit_ns = Some(timestamp_ns + self.refine_ns);
                false
            }
            (Some(_), None) => false,
        };
        if due && (self.calibration.is_ready() || self.calibration.fixes_ready()) {
            if let Some(estimate) = self.calibration.estimate() {
                let previous = self.output(odom_from_wheel);
                let calibrated = body_pose(&estimate, odom_from_wheel, self.apply_scale);
                let continuity = yaw_translation_between(&previous, &calibrated);
                let pose = continuity.compose(&calibrated);
                self.current = Some((estimate.clone(), continuity));
                self.next_refit_ns = (self.refine_ns > 0).then(|| timestamp_ns + self.refine_ns);
                return WheelOutput::Updated(pose, estimate);
            }
        }
        match &self.current {
            None => WheelOutput::PassThrough(odom_from_wheel.clone()),
            Some(_) => WheelOutput::Calibrated(self.output(odom_from_wheel)),
        }
    }

    /// The output for a wheel pose with the estimate in use.
    fn output(&self, odom_from_wheel: &SE3) -> SE3 {
        match &self.current {
            None => odom_from_wheel.clone(),
            Some((estimate, continuity)) => {
                continuity.compose(&body_pose(estimate, odom_from_wheel, self.apply_scale))
            }
        }
    }
}

/// The odometry's frame -> its child frame on `/tf`, from an odometry
/// message (the calibrated wheel odometry, in place of the wheel driver's).
pub fn odom_tf(odom: &Odometry) -> TfMessage {
    let pose = &odom.pose.pose;
    TfMessage {
        transforms: vec![TransformStamped {
            header: odom.header.clone(),
            child_frame_id: odom.child_frame_id.clone(),
            transform: Transform {
                translation: RosVector3 {
                    x: pose.position.x,
                    y: pose.position.y,
                    z: pose.position.z,
                },
                rotation: pose.orientation,
            },
        }],
    }
}

/// The body's pose from a wheel pose, at the map's scale or the wheels'.
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
    /// long, and with `map`, map localizations of every 4th pose in a map
    /// 1% long, arriving after the next wheel pose.
    fn feed(calibrator: &mut WheelOdomCalibrator, map: bool) -> Vec<WheelOutput> {
        let world = SE3::new(rz(0.3), Vector3::new(1.0, -2.0, 0.4));
        let map_frame = SE3::new(rz(-0.8), Vector3::new(3.0, 1.0, 0.0));
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
        let body = |wheel: &SE3, scale: f64| {
            SE3::new(
                wheel.rotation,
                scale * (wheel.translation + wheel.rotation * Vector3::new(0.05, 0.0, 0.0)),
            )
        };
        let mut fix = None;
        for (index, (x, y, yaw)) in steps.into_iter().enumerate() {
            let t = index as i64 * 50 * MS;
            let wheel = SE3::new(rz(yaw), Vector3::new(x, y, 0.0));
            calibrator.add_vio(
                t,
                &world
                    .compose(&body(&wheel, 1.02))
                    .compose(&imu_from_body().inverse()),
            );
            outputs.push(calibrator.wheel(t, &wheel));
            if let Some((fix_t, fix)) = fix.take() {
                calibrator.add_fix(fix_t, &fix);
            }
            if map && index % 4 == 0 {
                fix = Some((t, map_frame.compose(&body(&wheel, 1.01))));
            }
        }
        outputs
    }

    fn config() -> WheelCalibrationConfig {
        WheelCalibrationConfig {
            min_turn_deg: 180.0,
            min_pairs: 50,
            min_fixes: 10,
            ..WheelCalibrationConfig::default()
        }
    }

    #[test]
    fn passes_through_then_calibrates_without_a_jump() {
        let mut calibrator = WheelOdomCalibrator::new(config());
        calibrator.set_imu_from_body(imu_from_body());
        let outputs = feed(&mut calibrator, false);
        let fixed = outputs
            .iter()
            .position(|output| matches!(output, WheelOutput::Updated(..)))
            .expect("calibrated");
        assert!(outputs[..fixed]
            .iter()
            .all(|o| matches!(o, WheelOutput::PassThrough(_))));
        // Without refitting, the first estimate stays.
        assert!(outputs[fixed + 1..]
            .iter()
            .all(|o| matches!(o, WheelOutput::Calibrated(_))));
        let estimate = calibrator.estimate().expect("fixed").clone();
        // (Drawn by the prior, zero, by a few hundredths of a mm.)
        assert!((estimate.offset - Vector2::new(0.05, 0.0)).norm() < 1e-4);
        // The VIO's scale is not the map's: without map localizations the
        // output keeps the wheels'.
        assert!((estimate.vio_scale.expect("vio") - 1.02).abs() < 1e-4);
        assert_eq!(estimate.scale_from, CalibrationSource::Default);
        // No jump where the estimate takes over: the step there is the
        // wheel's own step (here a turn in place: the body point swings).
        let (before, after) = (outputs[fixed - 1].pose(), outputs[fixed].pose());
        assert!((after.translation - before.translation).norm() < 0.01);
        // Driving 1 m afterwards moves the output 1 m.
        let last = outputs.last().unwrap().pose();
        let start = outputs[outputs.len() - 51].pose();
        assert!(((last.translation - start.translation).norm() - 1.0).abs() < 1e-9);
        // The odometry frame stays gravity-aligned.
        assert!((last.rotation * Vector3::z() - Vector3::z()).norm() < 1e-9);
    }

    #[test]
    fn map_localizations_set_the_scale_early_and_refits_stay_continuous() {
        let mut calibrator = WheelOdomCalibrator::new(config()).with_refine_s(1.0);
        calibrator.set_imu_from_body(imu_from_body());
        let outputs = feed(&mut calibrator, true);
        let updates: Vec<usize> = outputs
            .iter()
            .enumerate()
            .filter(|(_, output)| matches!(output, WheelOutput::Updated(..)))
            .map(|(index, _)| index)
            .collect();
        // Ten map localizations give the level (wheel pose 37); once they
        // span 1 m, early in the turn (wheel poses 50 to 110), the scale, and
        // the map's estimate of the offset, which with so little turn stays
        // nearer the prior, zero, than the true 5 cm; the VIO's offset comes
        // after the turn.
        // In between, the estimate is refitted every second (20 wheel poses).
        let estimates: Vec<&WheelCalibrationEstimate> = updates
            .iter()
            .map(|&index| match &outputs[index] {
                WheelOutput::Updated(_, estimate) => estimate,
                _ => unreachable!(),
            })
            .collect();
        assert!((36..45).contains(&updates[0]), "{updates:?}");
        assert_eq!(estimates[0].level_from, CalibrationSource::Map);
        assert_eq!(estimates[0].scale_from, CalibrationSource::Default);
        let scaled = estimates
            .iter()
            .position(|estimate| estimate.scale_from == CalibrationSource::Map)
            .expect("the map's scale");
        assert!(updates[scaled] < 70, "{updates:?}");
        assert_eq!(estimates[scaled].offset_from, CalibrationSource::Map);
        assert!(estimates[scaled].offset.norm() < 0.025);
        // (Without the offset, the turn's first steps bias it slightly.)
        assert!((estimates[scaled].scale - 1.01).abs() < 2e-3);
        assert!(updates.windows(2).all(|pair| pair[1] - pair[0] == 20));
        let estimate = calibrator.estimate().expect("estimate");
        assert_eq!(estimate.offset_from, CalibrationSource::Vio);
        assert_eq!(estimate.scale_from, CalibrationSource::Map);
        assert!((estimate.offset - Vector2::new(0.05, 0.0)).norm() < 1e-5);
        assert!((estimate.scale - 1.01).abs() < 1e-5);
        // Each refit continues the output: the step there is no larger
        // than the wheels' own at that point (2 cm, or a 5 cm point turning
        // 3 deg in place).
        for &update in &updates[1..] {
            let step = (outputs[update].pose().translation
                - outputs[update - 1].pose().translation)
                .norm();
            assert!(step < 0.021, "{update}: {step}");
        }
        // Until the VIO has the offset, the turn biases the map's scale (the
        // body point swings 5 cm in it); the refit after that (wheel pose
        // 114) has it. Driving the last 0.8 m moves the output 0.808 m.
        let last = outputs.last().unwrap().pose();
        let start = outputs[outputs.len() - 41].pose();
        let driven = (last.translation - start.translation).norm();
        assert!((driven - 0.808).abs() < 1e-5, "{driven}");
    }

    #[test]
    fn a_saved_calibration_round_trips_and_applies_from_the_start() {
        let mut calibrator = WheelOdomCalibrator::new(config());
        calibrator.set_imu_from_body(imu_from_body());
        feed(&mut calibrator, false);
        let saved = SavedCalibration::from(calibrator.estimate().expect("fixed"));
        assert_eq!(saved.offset_from, "vio");
        let loaded = SavedCalibration::from_json(&saved.to_json()).expect("parses");
        assert_eq!(loaded, saved);
        let mut from_file = WheelOdomCalibrator::new(config()).with_prior(loaded.estimate());
        let wheel = SE3::new(rz(0.5), Vector3::new(1.0, 2.0, 0.0));
        let WheelOutput::Calibrated(pose) = from_file.wheel(0, &wheel) else {
            panic!("calibrated from the start");
        };
        let expected = calibrator.estimate().unwrap().odom_from_body(&wheel);
        assert!((pose.translation - expected.translation).norm() < 1e-9);
        // Without refitting, the saved values stay through a whole run.
        let mut kept = WheelOdomCalibrator::new(config()).with_prior(loaded.estimate());
        kept.set_imu_from_body(imu_from_body());
        assert!(feed(&mut kept, true)
            .iter()
            .all(|output| matches!(output, WheelOutput::Calibrated(_))));
        // Without the scale, 1 m of wheel travel stays 1 m.
        let mut unscaled = WheelOdomCalibrator::new(config())
            .with_prior(loaded.estimate())
            .with_scale(false);
        let a = unscaled.wheel(0, &SE3::identity());
        let b = unscaled.wheel(1, &SE3::new(rz(0.0), Vector3::new(1.0, 0.0, 0.0)));
        assert!(((b.pose().translation - a.pose().translation).norm() - 1.0).abs() < 1e-9);
        assert!(SavedCalibration::from_json("{}").is_err());
        let mut bad = saved.clone();
        bad.scale = 0.0;
        assert!(SavedCalibration::from_json(&bad.to_json()).is_err());
        // Files from before the map localizations still load.
        let old = r#"{"offset_m": [0.04, 0.03], "scale": 1.01, "body_from_level_xyzw": [0, 0, 0, 1],
            "tilt_deg": 0.0, "rms_m": 0.01, "turned_deg": 180.0, "pairs": 600}"#;
        let old = SavedCalibration::from_json(old).expect("old file");
        assert_eq!((old.fixes, old.scale_from.as_str()), (0, ""));
    }
}
