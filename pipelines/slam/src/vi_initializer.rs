//! Visual-inertial bootstrap (stationary-window initialiser).
//!
//! Real VIO front-ends (ORB-SLAM3, VINS-Mono, OKVIS, Kimera-VIO) all
//! gate the visual-inertial state on an "initialisation" stage that
//! turns the raw IMU stream into a starting `(R_w←b, v_w, b_g, b_a)`
//! tuple before the joint estimator is allowed to consume image
//! constraints. Without this step the integrator either has to seed
//! every state from ground truth (cheating) or accumulate hundreds of
//! metres of dead-reckoning drift before the first keyframe — see the
//! IMU-only ATE baseline in `docs/progress.md` for the latter case.
//!
//! This module ships the simplest defensible flavour: detect a window
//! where the gyro and accel signals are statistically consistent with
//! a body at rest, then read out
//!
//! * the gyro bias as the gyro signal mean (`b_g = ω̄`),
//! * the body-to-world rotation as the shortest rotation that lifts
//!   the mean specific-force direction into the world "up" direction
//!   `-g_w / ‖g_w‖` (yaw is left at zero; nothing in a stationary
//!   sample stream pins yaw against the world frame),
//! * an initial world-frame velocity that is by construction zero,
//! * an accel bias that absorbs any magnitude mismatch between the
//!   measured mean specific-force magnitude and `‖g_w‖`.
//!
//! Stationary detection guards the read-out with three thresholds:
//! per-axis gyro standard deviation, per-axis accel standard
//! deviation, and the magnitude error between the mean specific-force
//! and `‖g_w‖`. The window must additionally span at least
//! `min_stationary_window_seconds` and contain at least `min_samples`
//! samples. The thresholds are conservative defaults; calibration-
//! driven datasets such as EuRoC pass them comfortably during the
//! initial holding period before take-off.
//!
//! The recovered bootstrap is returned as a
//! [`VisualInertialInitializationResult`] and consumed by callers
//! (typically followed by seeding
//! [`crate::OnlineSlamImuConfig::bias_gyro_linearisation`] /
//! [`crate::OnlineSlamImuConfig::bias_acc_linearisation`] and the
//! first-keyframe pose). The initialiser itself is appearance-free
//! and does not touch the SLAM critical path; it lives in
//! `pipelines/slam` only because it shares the IMU sample
//! conventions with [`crate::ImuPreintegrator`].

use nalgebra::{UnitQuaternion, Vector3};

/// Configuration knobs for [`VisualInertialInitializer`]. The defaults
/// match the EuRoC stationary holding period (gravity z-up, ~0.5 s of
/// quiet samples, conservative noise thresholds).
#[derive(Debug, Clone, PartialEq)]
pub struct VisualInertialInitializerConfig {
    /// Gravity in the world frame. EuRoC convention is z-up,
    /// `(0, 0, -9.81)`; KITTI convention is y-down, `(0, 9.81, 0)`.
    pub gravity_world: Vector3<f64>,
    /// Minimum duration (seconds) of the stationary window. Below
    /// this threshold the initialiser refuses to commit even when the
    /// noise statistics look clean — too short a window leaves the
    /// gyro / accel means dominated by per-sample noise.
    pub min_stationary_window_seconds: f64,
    /// Maximum acceptable per-axis standard deviation of the gyro
    /// signal in the window (rad / s). Above this threshold the
    /// window is rejected as non-stationary.
    pub max_gyro_std: f64,
    /// Maximum acceptable per-axis standard deviation of the accel
    /// signal in the window (m / s²).
    pub max_accel_std: f64,
    /// Maximum acceptable deviation of the mean accel magnitude from
    /// `‖gravity_world‖` (m / s²). Helps reject sensor-saturation
    /// windows that look stationary by std-dev alone but report a
    /// drastically wrong magnitude (e.g. mis-scaled raw counts).
    pub max_accel_magnitude_error: f64,
    /// Minimum number of IMU samples in the window. A duration check
    /// alone is too loose under sparse / irregular sample rates.
    pub min_samples: usize,
    /// Width of the **sliding detector window** (seconds). When the
    /// buffer exceeds this duration, only the most recent
    /// `detector_window_seconds` of samples (the trailing slice) are
    /// considered for every stationary-window predicate and statistic.
    ///
    /// `f64::INFINITY` (the default) keeps the historical "all-buffer"
    /// behaviour, where statistics are evaluated over every buffered
    /// sample — appropriate for the simplest standalone case where a
    /// caller buffers a single clean stationary segment up front. A
    /// finite value is the pipeline-integration setup: it lets the
    /// detector ride out a leading non-stationary segment (e.g. the
    /// pipeline started during motion and the body settled later) by
    /// re-evaluating on only the trailing window, so an early-motion
    /// buffer no longer poisons the detector forever.
    pub detector_window_seconds: f64,
}

impl Default for VisualInertialInitializerConfig {
    fn default() -> Self {
        Self {
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            min_stationary_window_seconds: 0.5,
            max_gyro_std: 0.05,
            max_accel_std: 0.5,
            max_accel_magnitude_error: 0.5,
            min_samples: 50,
            // `INFINITY` preserves the historical "evaluate on the whole
            // buffer" behaviour; existing callers see no numerical
            // change after this field lands.
            detector_window_seconds: f64::INFINITY,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImuSampleRecord {
    gyro: Vector3<f64>,
    accel: Vector3<f64>,
    dt: f64,
}

/// Stateful accumulator that ingests IMU samples one at a time and
/// reports the recovered bootstrap as soon as the buffer satisfies
/// every stationary-window predicate.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualInertialInitializer {
    config: VisualInertialInitializerConfig,
    samples: Vec<ImuSampleRecord>,
}

/// Outcome of a successful stationary-window initialisation. All
/// fields are in SI units; angles are encoded as a unit quaternion.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualInertialInitializationResult {
    /// World-frame gravity, echoed from the config so downstream
    /// callers (e.g. `OnlineSlamImuConfig`) can wire it through
    /// without re-importing the config struct.
    pub gravity_world: Vector3<f64>,
    /// Body-to-world rotation recovered from the mean specific-force
    /// direction. Yaw is set to zero (gravity does not pin yaw).
    pub initial_rotation_body_to_world: UnitQuaternion<f64>,
    /// Initial world-frame velocity. Always exactly zero — the
    /// initialiser only fires when the window is stationary.
    pub initial_velocity_world: Vector3<f64>,
    /// Recovered gyro bias = mean(gyro readings) in body frame.
    pub bias_gyro: Vector3<f64>,
    /// Recovered accel bias. When the measured specific-force
    /// magnitude exactly matches `‖gravity_world‖` this is zero;
    /// otherwise it captures the magnitude residual along the
    /// measured gravity direction.
    pub bias_acc: Vector3<f64>,
    /// Number of IMU samples consumed by the recovery.
    pub samples_consumed: usize,
    /// Duration of the stationary window (seconds).
    pub duration_seconds: f64,
    /// Per-axis gyro standard deviation observed in the window
    /// (rad / s). Surface for diagnostics / logging.
    pub gyro_std: Vector3<f64>,
    /// Per-axis accel standard deviation observed in the window
    /// (m / s²).
    pub accel_std: Vector3<f64>,
    /// Magnitude of the mean specific-force measurement (m / s²).
    pub mean_accel_magnitude: f64,
}

impl VisualInertialInitializationResult {
    /// Angle (degrees) between the body-frame "up" direction implied
    /// by the recovered rotation and the same direction implied by a
    /// caller-supplied ground-truth body→world rotation. This is the
    /// **yaw-gauge-aware** rotation residual: gravity-only stationary
    /// initialisation fundamentally cannot observe yaw, so the full
    /// quaternion residual `rotation_to(gt).angle()` unfairly penalises
    /// a yaw mismatch that the initialiser had no signal to fix. This
    /// metric isolates the roll / pitch component (the only part the
    /// gravity-only bootstrap can actually recover) by projecting the
    /// world "up" direction into body frame under both rotations and
    /// reporting the angle between the resulting body-frame vectors.
    ///
    /// Reads `gravity_world` from the result, so callers do not need
    /// to thread the gravity vector through separately. Returns `0.0`
    /// if `gravity_world` is the zero vector (degenerate config).
    pub fn gravity_alignment_residual_deg(
        &self,
        ground_truth_rotation_body_to_world: &UnitQuaternion<f64>,
    ) -> f64 {
        let g_norm = self.gravity_world.norm();
        if !g_norm.is_finite() || g_norm <= f64::EPSILON {
            return 0.0;
        }
        let world_up = -self.gravity_world / g_norm;
        let recovered_body_up = self.initial_rotation_body_to_world.inverse() * world_up;
        let gt_body_up = ground_truth_rotation_body_to_world.inverse() * world_up;
        recovered_body_up
            .dot(&gt_body_up)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees()
    }
}

/// Diagnostic reason a stationary-window check failed.
#[derive(Debug, Clone, PartialEq)]
pub enum StationaryRejectionReason {
    InsufficientSamples {
        have: usize,
        need: usize,
    },
    InsufficientDuration {
        have: f64,
        need: f64,
    },
    GyroNoiseTooHigh {
        observed: Vector3<f64>,
        limit: f64,
    },
    AccelNoiseTooHigh {
        observed: Vector3<f64>,
        limit: f64,
    },
    AccelMagnitudeMismatch {
        observed: f64,
        expected: f64,
        tolerance: f64,
    },
}

impl VisualInertialInitializer {
    /// Create a fresh initialiser with the given configuration. The
    /// returned instance owns no samples; feed it the IMU stream via
    /// [`Self::push_sample`] until [`Self::try_initialize`] reports
    /// success.
    pub fn new(config: VisualInertialInitializerConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

    /// Append one body-frame IMU sample to the accumulator. Non-
    /// positive `dt` is silently dropped to keep raw replays robust
    /// against duplicated / out-of-order timestamps (the same
    /// convention used by [`crate::ImuPreintegrator::integrate_sample`]).
    pub fn push_sample(&mut self, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        self.samples.push(ImuSampleRecord { gyro, accel, dt });
    }

    /// Number of IMU samples currently buffered.
    pub fn samples_seen(&self) -> usize {
        self.samples.len()
    }

    /// Aggregate duration (seconds) of currently buffered samples.
    pub fn buffered_duration_seconds(&self) -> f64 {
        self.samples.iter().map(|s| s.dt).sum()
    }

    /// Drop every buffered sample. The configuration is preserved.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &VisualInertialInitializerConfig {
        &self.config
    }

    /// Attempt to recover the visual-inertial bootstrap from the
    /// currently buffered samples. Returns `Ok(result)` when every
    /// stationary-window predicate passes, otherwise returns
    /// `Err(reason)` describing the first failed predicate so callers
    /// can log a diagnostic and keep collecting samples.
    ///
    /// When `config.detector_window_seconds` is finite, predicates and
    /// statistics are evaluated only on the **trailing slice** whose
    /// cumulative `dt` first reaches that bound — early non-stationary
    /// samples no longer prevent a later stationary tail from passing.
    pub fn try_initialize(
        &self,
    ) -> Result<VisualInertialInitializationResult, StationaryRejectionReason> {
        let (slice, slice_duration) = self.detector_window_slice();
        if slice.len() < self.config.min_samples {
            return Err(StationaryRejectionReason::InsufficientSamples {
                have: slice.len(),
                need: self.config.min_samples,
            });
        }
        if slice_duration < self.config.min_stationary_window_seconds {
            return Err(StationaryRejectionReason::InsufficientDuration {
                have: slice_duration,
                need: self.config.min_stationary_window_seconds,
            });
        }

        // Time-weighted mean and variance: `μ = Σ x_i · dt_i / Σ dt_i`,
        // `σ² = Σ (x_i - μ)² · dt_i / Σ dt_i`. For uniform `dt` this
        // collapses to the sample-count weighted form and existing
        // numerical outputs are preserved exactly. For irregular `dt`
        // (a real IMU stream that drops or duplicates samples) the
        // time-weighted form removes a small but real bias in the
        // statistics.
        let total_dt = slice_duration;
        let gyro_mean = slice
            .iter()
            .fold(Vector3::<f64>::zeros(), |acc, s| acc + s.gyro * s.dt)
            / total_dt;
        let accel_mean = slice
            .iter()
            .fold(Vector3::<f64>::zeros(), |acc, s| acc + s.accel * s.dt)
            / total_dt;
        let mut gyro_var = Vector3::<f64>::zeros();
        let mut accel_var = Vector3::<f64>::zeros();
        for sample in slice {
            let dg = sample.gyro - gyro_mean;
            let da = sample.accel - accel_mean;
            gyro_var += Vector3::new(dg.x * dg.x, dg.y * dg.y, dg.z * dg.z) * sample.dt;
            accel_var += Vector3::new(da.x * da.x, da.y * da.y, da.z * da.z) * sample.dt;
        }
        gyro_var /= total_dt;
        accel_var /= total_dt;
        let gyro_std = Vector3::new(gyro_var.x.sqrt(), gyro_var.y.sqrt(), gyro_var.z.sqrt());
        let accel_std = Vector3::new(accel_var.x.sqrt(), accel_var.y.sqrt(), accel_var.z.sqrt());

        if gyro_std
            .iter()
            .any(|s| !s.is_finite() || *s > self.config.max_gyro_std)
        {
            return Err(StationaryRejectionReason::GyroNoiseTooHigh {
                observed: gyro_std,
                limit: self.config.max_gyro_std,
            });
        }
        if accel_std
            .iter()
            .any(|s| !s.is_finite() || *s > self.config.max_accel_std)
        {
            return Err(StationaryRejectionReason::AccelNoiseTooHigh {
                observed: accel_std,
                limit: self.config.max_accel_std,
            });
        }

        let gravity_magnitude = self.config.gravity_world.norm();
        let mean_accel_magnitude = accel_mean.norm();
        if (mean_accel_magnitude - gravity_magnitude).abs() > self.config.max_accel_magnitude_error
        {
            return Err(StationaryRejectionReason::AccelMagnitudeMismatch {
                observed: mean_accel_magnitude,
                expected: gravity_magnitude,
                tolerance: self.config.max_accel_magnitude_error,
            });
        }
        if mean_accel_magnitude <= f64::EPSILON {
            return Err(StationaryRejectionReason::AccelMagnitudeMismatch {
                observed: mean_accel_magnitude,
                expected: gravity_magnitude,
                tolerance: self.config.max_accel_magnitude_error,
            });
        }

        // When the body is stationary the specific-force measurement
        // in body frame is `a_b ≈ -R_b←w · g_w + b_a`. We recover
        // R_w←b by aligning `a_b` (after bias cancellation) with the
        // world-frame "up" direction `-g_w`. Yaw is unobservable from
        // gravity alone, so the shortest rotation that maps `a_b` to
        // `-g_w` is the canonical choice — it leaves yaw at zero.
        let world_up = -self.config.gravity_world;
        let rotation = UnitQuaternion::rotation_between(&accel_mean, &world_up)
            .unwrap_or_else(UnitQuaternion::identity);

        // Once the direction is absorbed by the rotation, any residual
        // is in the magnitude only: `b_a = a_b - (-R_b←w · g_w)`. The
        // dot-product form below collapses to `(|a_b| - |g_w|) · û_b`
        // when `R_w←b` was estimated perfectly from the direction.
        let inverse_rotation = rotation.inverse();
        let bias_acc = accel_mean - inverse_rotation * world_up;

        Ok(VisualInertialInitializationResult {
            gravity_world: self.config.gravity_world,
            initial_rotation_body_to_world: rotation,
            initial_velocity_world: Vector3::zeros(),
            bias_gyro: gyro_mean,
            bias_acc,
            samples_consumed: slice.len(),
            duration_seconds: slice_duration,
            gyro_std,
            accel_std,
            mean_accel_magnitude,
        })
    }

    /// Return the trailing detector slice of buffered samples plus its
    /// cumulative `dt`. When `config.detector_window_seconds` is
    /// infinite (or non-positive / non-finite), this is the entire
    /// buffer. Otherwise it walks the buffer from the end backwards and
    /// keeps adding samples until the cumulative `dt` first reaches the
    /// configured window width — the slice may be slightly longer than
    /// `detector_window_seconds` by at most one sample's `dt`.
    fn detector_window_slice(&self) -> (&[ImuSampleRecord], f64) {
        let window = self.config.detector_window_seconds;
        if !window.is_finite() || window <= 0.0 {
            let dur: f64 = self.samples.iter().map(|s| s.dt).sum();
            return (&self.samples[..], dur);
        }
        let mut acc_dt = 0.0;
        let mut start_idx = self.samples.len();
        for i in (0..self.samples.len()).rev() {
            acc_dt += self.samples[i].dt;
            start_idx = i;
            if acc_dt >= window {
                break;
            }
        }
        (&self.samples[start_idx..], acc_dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    fn euroc_config() -> VisualInertialInitializerConfig {
        VisualInertialInitializerConfig::default()
    }

    #[test]
    fn default_config_uses_z_up_gravity() {
        let config = VisualInertialInitializerConfig::default();
        assert!((config.gravity_world - Vector3::new(0.0, 0.0, -9.81)).norm() < 1.0e-12);
    }

    #[test]
    fn stationary_level_window_recovers_zero_rotation_and_zero_biases() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005; // 200 Hz
        let target_accel = -Vector3::new(0.0, 0.0, -9.81); // body level → reads up
        for _ in 0..200 {
            initializer.push_sample(Vector3::zeros(), target_accel, dt);
        }
        let result = initializer
            .try_initialize()
            .expect("stationary window should recover bootstrap");
        assert_eq!(result.samples_consumed, 200);
        assert!((result.duration_seconds - 1.0).abs() < 1.0e-9);
        assert!(result.initial_velocity_world.norm() < 1.0e-12);
        assert!(result.bias_gyro.norm() < 1.0e-12);
        assert!(result.bias_acc.norm() < 1.0e-9);
        assert!(result.initial_rotation_body_to_world.angle().abs() < 1.0e-9);
        assert!(result.gyro_std.norm() < 1.0e-12);
        assert!(result.accel_std.norm() < 1.0e-12);
        assert!((result.mean_accel_magnitude - 9.81).abs() < 1.0e-9);
    }

    #[test]
    fn tilted_stationary_window_recovers_rotation() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        // Body tilted 30 degrees about x-axis. Then accel measures
        // R_b←w · (-g_w). With g_w = (0, 0, -9.81), -g_w = (0, 0, 9.81)
        // expressed in body frame is R_b←w · (0, 0, 9.81).
        let rotation_w_b = Rotation3::from_axis_angle(&Vector3::x_axis(), 0.5236); // 30 deg
        let world_up = Vector3::new(0.0, 0.0, 9.81);
        let accel_body = rotation_w_b.inverse() * world_up;
        for _ in 0..200 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        let result = initializer
            .try_initialize()
            .expect("tilted stationary should still recover bootstrap");
        // Rotating the measured accel by the recovered rotation should
        // produce the world-frame "up" direction.
        let lifted = result.initial_rotation_body_to_world * accel_body;
        assert!((lifted - world_up).norm() < 1.0e-9);
        assert!(result.bias_acc.norm() < 1.0e-9);
        // The recovered orientation should match the synthetic rotation
        // (up to yaw, which is unobservable; the synthetic rotation
        // here has no yaw component, so the match is exact).
        let recovered = Rotation3::from(result.initial_rotation_body_to_world);
        let delta = rotation_w_b * recovered.inverse();
        assert!(delta.angle().abs() < 1.0e-9);
    }

    #[test]
    fn nonzero_gyro_mean_becomes_bias() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        let gyro_bias = Vector3::new(0.01, -0.02, 0.005);
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for _ in 0..200 {
            initializer.push_sample(gyro_bias, accel_body, dt);
        }
        let result = initializer.try_initialize().expect("ok");
        assert!((result.bias_gyro - gyro_bias).norm() < 1.0e-12);
    }

    #[test]
    fn rejects_noisy_gyro() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for i in 0..200 {
            // Alternating ±1.0 rad/s on x-axis → std = 1.0, way above default 0.05.
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gyro = Vector3::new(sign, 0.0, 0.0);
            initializer.push_sample(gyro, accel_body, dt);
        }
        match initializer.try_initialize().expect_err("should reject") {
            StationaryRejectionReason::GyroNoiseTooHigh { observed, limit } => {
                assert!(observed.x > limit);
            }
            other => panic!("expected GyroNoiseTooHigh, got {other:?}"),
        }
    }

    #[test]
    fn rejects_noisy_accel() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        for i in 0..200 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let accel = Vector3::new(2.0 * sign, 0.0, 9.81);
            initializer.push_sample(Vector3::zeros(), accel, dt);
        }
        match initializer.try_initialize().expect_err("should reject") {
            StationaryRejectionReason::AccelNoiseTooHigh { observed, limit } => {
                assert!(observed.x > limit);
            }
            other => panic!("expected AccelNoiseTooHigh, got {other:?}"),
        }
    }

    #[test]
    fn rejects_accel_magnitude_mismatch() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 2.0); // way off 9.81
        for _ in 0..200 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        match initializer.try_initialize().expect_err("should reject") {
            StationaryRejectionReason::AccelMagnitudeMismatch {
                observed,
                expected,
                tolerance,
            } => {
                assert!((observed - 2.0).abs() < 1.0e-9);
                assert!((expected - 9.81).abs() < 1.0e-9);
                assert!(tolerance > 0.0);
            }
            other => panic!("expected AccelMagnitudeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_insufficient_samples() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for _ in 0..10 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        match initializer.try_initialize().expect_err("should reject") {
            StationaryRejectionReason::InsufficientSamples { have, need } => {
                assert_eq!(have, 10);
                assert_eq!(need, 50);
            }
            other => panic!("expected InsufficientSamples, got {other:?}"),
        }
    }

    #[test]
    fn rejects_insufficient_duration() {
        let config = VisualInertialInitializerConfig {
            min_samples: 10,
            min_stationary_window_seconds: 1.0,
            ..Default::default()
        };
        let mut initializer = VisualInertialInitializer::new(config);
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        // 50 samples × 5 ms = 0.25 s; below the 1 s threshold.
        for _ in 0..50 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        match initializer.try_initialize().expect_err("should reject") {
            StationaryRejectionReason::InsufficientDuration { have, need } => {
                assert!((have - 0.25).abs() < 1.0e-9);
                assert!((need - 1.0).abs() < 1.0e-9);
            }
            other => panic!("expected InsufficientDuration, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_buffered_samples() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        initializer.push_sample(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.005);
        assert_eq!(initializer.samples_seen(), 1);
        initializer.reset();
        assert_eq!(initializer.samples_seen(), 0);
        assert!((initializer.buffered_duration_seconds() - 0.0).abs() < 1.0e-12);
    }

    #[test]
    fn push_sample_drops_nonpositive_dt() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let accel = Vector3::new(0.0, 0.0, 9.81);
        initializer.push_sample(Vector3::zeros(), accel, 0.0);
        initializer.push_sample(Vector3::zeros(), accel, -0.005);
        initializer.push_sample(Vector3::zeros(), accel, f64::NAN);
        assert_eq!(initializer.samples_seen(), 0);
    }

    #[test]
    fn default_detector_window_seconds_is_infinity() {
        // Backward-compat anchor: the new sliding-window knob must
        // default to "use the entire buffer", so existing callers see
        // zero behavioural change after the field lands.
        let config = VisualInertialInitializerConfig::default();
        assert!(config.detector_window_seconds.is_infinite());
        assert!(config.detector_window_seconds > 0.0);
    }

    #[test]
    fn time_weighted_statistics_match_unweighted_for_uniform_dt() {
        // Regression sanity check: under uniform `dt` the time-weighted
        // mean / variance formula must match the historical
        // sample-count weighted output exactly. This is the property
        // that lets all 11 pre-existing tests keep passing — but it's
        // worth pinning explicitly so any future tweak to the
        // weighting formula is caught immediately.
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        for i in 0..200 {
            // A signal where the mean / std are easy to compute by
            // hand: gyro x alternates +0.001 / -0.001 (mean 0, std
            // 0.001), accel z = 9.81 + (i as f64) * 1e-6 (mean drifts
            // slightly).
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gyro = Vector3::new(sign * 0.001, 0.0, 0.0);
            let accel = Vector3::new(0.0, 0.0, 9.81 + (i as f64) * 1.0e-6);
            initializer.push_sample(gyro, accel, dt);
        }
        let result = initializer
            .try_initialize()
            .expect("clean stationary stream");
        // Sample-count weighted mean for `[+0.001, -0.001, ...]` (200
        // samples) is exactly 0; same for the time-weighted form when
        // dt is uniform.
        assert!(result.bias_gyro.norm() < 1.0e-15);
        // gyro_std on x should equal 0.001 to within fp tolerance.
        assert!((result.gyro_std.x - 0.001).abs() < 1.0e-12);
    }

    #[test]
    fn time_weighted_statistics_shift_mean_under_nonuniform_dt() {
        // With non-uniform `dt`, the time-weighted mean must weight
        // each sample by its `dt`. Push two samples: one with a tiny
        // dt at value 0, one with a much larger dt at value 1. The
        // sample-count-weighted mean would be 0.5; the time-weighted
        // mean must be much closer to 1.
        let config = VisualInertialInitializerConfig {
            min_samples: 2,
            min_stationary_window_seconds: 0.0,
            max_gyro_std: f64::INFINITY,
            max_accel_std: f64::INFINITY,
            max_accel_magnitude_error: f64::INFINITY,
            ..Default::default()
        };
        let mut initializer = VisualInertialInitializer::new(config);
        // Sample A: dt = 0.001 s, gyro.x = 0.
        initializer.push_sample(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.001);
        // Sample B: dt = 0.999 s, gyro.x = 1.0.
        initializer.push_sample(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 9.81),
            0.999,
        );
        let result = initializer.try_initialize().expect("predicates relaxed");
        // Time-weighted mean = (0 * 0.001 + 1 * 0.999) / 1.000 = 0.999.
        // Sample-count weighted would be (0 + 1) / 2 = 0.500.
        assert!(
            (result.bias_gyro.x - 0.999).abs() < 1.0e-12,
            "time-weighted mean expected 0.999, got {}",
            result.bias_gyro.x
        );
    }

    #[test]
    fn sliding_window_non_stationary_then_stationary_succeeds() {
        // Deferred test #13 from the design document: the integrator
        // starts during motion (0.5 s of noisy gyro) and the body
        // settles later (1.5 s of stationary samples). With a 1.0 s
        // sliding window the trailing slice is entirely stationary so
        // initialisation must succeed and `samples_consumed` must
        // reflect only the trailing window, not the polluted leading
        // chunk.
        let config = VisualInertialInitializerConfig {
            min_samples: 50,
            min_stationary_window_seconds: 0.5,
            detector_window_seconds: 1.0,
            ..Default::default()
        };
        let mut initializer = VisualInertialInitializer::new(config);
        let dt = 0.005; // 200 Hz
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        // Phase 1: 0.5 s of noisy gyro at ±1 rad/s on the x-axis.
        for i in 0..100 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            initializer.push_sample(Vector3::new(sign, 0.0, 0.0), accel_body, dt);
        }
        // Phase 2: 1.5 s of stationary samples.
        for _ in 0..300 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        let result = initializer
            .try_initialize()
            .expect("trailing 1.0 s window is clean");
        // Slice walks backwards from the end until cumulative `dt` ≥
        // 1.0 s. With dt = 0.005 that's 200 samples — entirely from
        // phase 2.
        assert_eq!(result.samples_consumed, 200);
        assert!((result.duration_seconds - 1.0).abs() < 1.0e-9);
        // The slice is clean, so the recovered gyro bias is exactly
        // the stationary mean (zero) and the std is zero.
        assert!(result.bias_gyro.norm() < 1.0e-12);
        assert!(result.gyro_std.norm() < 1.0e-12);
        // `samples_seen()` still reports the full buffer (400) — the
        // sliding window is an evaluation detail, not a buffer prune.
        assert_eq!(initializer.samples_seen(), 400);
    }

    #[test]
    fn sliding_window_rejects_when_recent_samples_are_noisy() {
        // The reverse of the above: 1.5 s of stationary samples
        // followed by 0.5 s of noisy gyro. With a 0.5 s sliding
        // window the trailing slice is exactly the noisy chunk so
        // initialisation must fail with `GyroNoiseTooHigh`.
        let config = VisualInertialInitializerConfig {
            min_samples: 50,
            min_stationary_window_seconds: 0.4,
            detector_window_seconds: 0.5,
            ..Default::default()
        };
        let mut initializer = VisualInertialInitializer::new(config);
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for _ in 0..300 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        for i in 0..100 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            initializer.push_sample(Vector3::new(sign, 0.0, 0.0), accel_body, dt);
        }
        match initializer
            .try_initialize()
            .expect_err("trailing slice is noisy")
        {
            StationaryRejectionReason::GyroNoiseTooHigh { observed, limit } => {
                assert!(observed.x > limit);
            }
            other => panic!("expected GyroNoiseTooHigh, got {other:?}"),
        }
    }

    #[test]
    fn sliding_window_insufficient_when_trailing_slice_too_short() {
        // The sliding-window predicates apply to the SLICE, not the
        // total buffer. A buffer with 2 s of samples but a 0.1 s
        // window where `min_stationary_window_seconds = 0.5` must be
        // rejected as `InsufficientDuration` — the buffer is large
        // but the slice is too small.
        let config = VisualInertialInitializerConfig {
            min_samples: 10,
            min_stationary_window_seconds: 0.5,
            detector_window_seconds: 0.1,
            ..Default::default()
        };
        let mut initializer = VisualInertialInitializer::new(config);
        let dt = 0.005;
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for _ in 0..400 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        match initializer
            .try_initialize()
            .expect_err("slice is shorter than required")
        {
            StationaryRejectionReason::InsufficientDuration { have, need } => {
                // Slice walks backwards until cumulative dt >= 0.1 s,
                // so `have` is `~0.1` (slightly over by one sample).
                assert!(have < need);
                assert!((need - 0.5).abs() < 1.0e-9);
                assert!(have > 0.0 && have < 0.5);
            }
            other => panic!("expected InsufficientDuration, got {other:?}"),
        }
    }

    #[test]
    fn gravity_alignment_residual_is_invariant_under_world_yaw() {
        // Test #12 from the test-strategy section of
        // `docs/vi_initialization_integration.md`. Stationary VI init
        // observes the gravity direction but not yaw, so the
        // `gravity_alignment_residual_deg` metric must report `~0` for
        // two stationary streams whose ground-truth body→world
        // rotations differ only by a 30° rotation about the world
        // "up" axis. The full quaternion residual against the
        // yaw-rotated GT is NOT zero — that is the precise mistake
        // the metric is designed to avoid. This pins yaw as a gauge
        // in the residual, not a real recovery error.
        let mut config = euroc_config();
        // Use stronger noise gates than the noise level of the
        // synthesised data so the streams pass cleanly.
        config.max_accel_std = 1.0e-6;
        let world_gravity = config.gravity_world;
        let world_up_axis = nalgebra::Unit::new_normalize(-world_gravity);

        // Tilt about world-x by 30° — perpendicular to the world up
        // (z) axis under EuRoC convention, so the tilt has zero
        // projection onto the yaw axis and the two ground-truth
        // rotations below genuinely differ only by yaw.
        let tilt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 30f64.to_radians());
        let yaw = UnitQuaternion::from_axis_angle(&world_up_axis, 30f64.to_radians());

        let gt_a = tilt;
        let gt_b = yaw * tilt;

        // Specific-force measurement in body frame for each
        // orientation: `a_b = R_b←w · (-g_w)`.
        let accel_a = gt_a.inverse() * (-world_gravity);
        let accel_b = gt_b.inverse() * (-world_gravity);

        let dt = 0.005;
        let mut initializer_a = VisualInertialInitializer::new(config.clone());
        let mut initializer_b = VisualInertialInitializer::new(config.clone());
        for _ in 0..200 {
            initializer_a.push_sample(Vector3::zeros(), accel_a, dt);
            initializer_b.push_sample(Vector3::zeros(), accel_b, dt);
        }
        let result_a = initializer_a.try_initialize().expect("stream A stationary");
        let result_b = initializer_b.try_initialize().expect("stream B stationary");

        // The yaw-gauge-aware metric must report essentially zero for
        // BOTH streams — that is the load-bearing assertion.
        let residual_a = result_a.gravity_alignment_residual_deg(&gt_a);
        let residual_b = result_b.gravity_alignment_residual_deg(&gt_b);
        assert!(residual_a < 1.0e-6, "stream A residual = {residual_a} deg");
        assert!(residual_b < 1.0e-6, "stream B residual = {residual_b} deg");

        // Cross-check: the full quaternion residual against stream B's
        // ground truth should be ~30° (the yaw mismatch the
        // initialiser cannot observe). If this assertion fails the
        // test is degenerate — i.e. the streams do not actually
        // differ in yaw — and the previous two assertions become
        // meaningless.
        let full_residual_b_deg = gt_b
            .rotation_to(&result_b.initial_rotation_body_to_world)
            .angle()
            .to_degrees();
        assert!(
            (full_residual_b_deg - 30.0).abs() < 1.0e-4,
            "stream B full residual should be ~30°, got {full_residual_b_deg}"
        );
    }

    #[test]
    fn gravity_alignment_residual_returns_zero_for_zero_gravity_config() {
        // Degenerate config guard: when `gravity_world` is the zero
        // vector the metric has no defined direction; the helper
        // returns 0.0 rather than NaN-poisoning a diagnostic log.
        let mut result_zero_g = VisualInertialInitializationResult {
            gravity_world: Vector3::zeros(),
            initial_rotation_body_to_world: UnitQuaternion::identity(),
            initial_velocity_world: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
            bias_acc: Vector3::zeros(),
            samples_consumed: 0,
            duration_seconds: 0.0,
            gyro_std: Vector3::zeros(),
            accel_std: Vector3::zeros(),
            mean_accel_magnitude: 0.0,
        };
        let any_gt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.5);
        assert_eq!(result_zero_g.gravity_alignment_residual_deg(&any_gt), 0.0);
        // Sanity: with the standard EuRoC gravity, a 90° tilt should
        // produce ~90° residual against identity.
        result_zero_g.gravity_world = Vector3::new(0.0, 0.0, -9.81);
        let tilt = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2);
        let residual_deg = result_zero_g.gravity_alignment_residual_deg(&tilt);
        assert!(
            (residual_deg - 90.0).abs() < 1.0e-9,
            "identity vs 90° tilt residual = {residual_deg} deg"
        );
    }

    #[test]
    fn accel_magnitude_mismatch_absorbed_by_bias_when_within_tolerance() {
        let mut initializer = VisualInertialInitializer::new(euroc_config());
        let dt = 0.005;
        // Measured magnitude 9.95 vs expected 9.81: 0.14 within
        // tolerance 0.5, but should leak into bias_acc.
        let accel_body = Vector3::new(0.0, 0.0, 9.95);
        for _ in 0..200 {
            initializer.push_sample(Vector3::zeros(), accel_body, dt);
        }
        let result = initializer.try_initialize().expect("ok");
        let recovered_world_accel =
            result.initial_rotation_body_to_world * (accel_body - result.bias_acc);
        let expected_world_accel = -result.gravity_world;
        assert!((recovered_world_accel - expected_world_accel).norm() < 1.0e-9);
        assert!(result.bias_acc.norm() > 0.0);
    }
}
