//! Pipeline-side glue for the stationary-window VI initialiser.
//!
//! The standalone [`crate::VisualInertialInitializer`] is appearance-free
//! and only knows how to ingest IMU samples. This module is the boundary
//! between that initialiser and [`crate::OnlineSlamPipeline`]'s running
//! IMU / local-VI-BA state: it ships the public config / event types the
//! pipeline exposes to callers, and the private `OnlineSlamViInitState`
//! that the pipeline owns and mutates atomically on a successful
//! bootstrap. See [`docs/vi_initialization_integration.md`](../../../docs/vi_initialization_integration.md)
//! for the design contract this implements (in particular the rotation-
//! direction promotion and stale-factor-gate sections, which are
//! load-bearing).

use nalgebra::Vector3;
use visloc_core::geometry::SE3;

use crate::vi_initializer::{
    StationaryRejectionReason, VisualInertialInitializationResult, VisualInertialInitializer,
    VisualInertialInitializerConfig,
};

/// Behaviour when the pipeline's auto-bootstrap stage runs out of buffer
/// space or wall-clock IMU duration without successfully recovering a
/// stationary window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViInitFallback {
    /// Keep the caller-supplied [`crate::OnlineSlamImuConfig`] seeds as
    /// the running bias / rotation. The pipeline never touches
    /// `imu_state`; the staleness gate is lifted so subsequent IMU
    /// factors flow with those original defaults.
    KeepExistingSeed,
    /// Disable the IMU stage entirely for the rest of the sequence by
    /// dropping `imu_state` and `local_vi_ba_state` and clearing
    /// `config.imu` / `config.local_vi_ba`.
    DisableImuStage,
}

/// Optional auto-bootstrap stage that runs a
/// [`crate::VisualInertialInitializer`] over the pipeline's incoming IMU
/// stream and atomically promotes the recovered `(R_w←b, b_g, b_a)` into
/// the running pre-integrator on the first keyframe.
///
/// Requires [`crate::OnlineSlamConfig::imu`] to be `Some(_)`. See
/// [`docs/vi_initialization_integration.md`](../../../docs/vi_initialization_integration.md)
/// for the full design contract, including the rotation-direction
/// conversion and stale-factor-gate semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamViInitConfig {
    /// Inner stationary-window initialiser config.
    pub initializer: VisualInertialInitializerConfig,
    /// Body-to-camera SE(3) extrinsic. When the IMU and camera share
    /// orientation, pass [`SE3::identity()`].
    pub body_to_camera: SE3,
    /// Whether to overwrite the just-registered first keyframe's
    /// `Pose.rotation` with `R_c←w = (R_w←b · R_b←c)^T` on success. The
    /// camera centre is preserved (the stationary-window flavour cannot
    /// observe absolute position).
    pub seed_first_keyframe_rotation: bool,
    /// Behaviour when [`Self::max_wait_duration_seconds`] or
    /// [`Self::max_buffered_samples`] is reached without success.
    pub on_persistent_rejection: ViInitFallback,
    /// Semantic cap: stop trying once this much wall-clock IMU duration
    /// has been buffered. `0.0` disables the duration cap.
    pub max_wait_duration_seconds: f64,
    /// Memory guard: refuse to buffer more than this many raw samples
    /// regardless of duration. `0` disables the memory cap.
    pub max_buffered_samples: usize,
    /// When `true`, `OnlineSlamPipeline::run_vi_init_step` calls the inner
    /// `try_initialize` on EVERY frame rather than only on frames that
    /// registered a new keyframe. The default (`false`) preserves the
    /// historical contract that "VI-init's recovered rotation attaches to
    /// the just-registered first keyframe", which couples promotion to
    /// the keyframe cadence. Setting `true` lets the stage promote as
    /// soon as the IMU's stationary-window detector accepts; on success
    /// without a new KF this frame, the promotion binds to the latest
    /// existing keyframe (or no keyframe if the map is empty), and the
    /// `seed_first_keyframe_rotation` rewrite is skipped when there is
    /// no candidate keyframe. Phase-19 lever — lifts the structural gate
    /// that made Phase-16's promotion-time BA trigger unreachable.
    pub try_initialize_on_every_frame: bool,
}

impl Default for OnlineSlamViInitConfig {
    fn default() -> Self {
        Self {
            initializer: VisualInertialInitializerConfig::default(),
            body_to_camera: SE3::identity(),
            seed_first_keyframe_rotation: true,
            on_persistent_rejection: ViInitFallback::KeepExistingSeed,
            max_wait_duration_seconds: 5.0,
            max_buffered_samples: 2000,
            try_initialize_on_every_frame: false,
        }
    }
}

/// Configuration-level error raised on [`crate::OnlineSlamPipeline::new`]
/// (and exposed by [`crate::OnlineSlamConfig::validate`]) for invalid
/// combinations the type system cannot rule out.
#[derive(Debug, Clone, PartialEq)]
pub enum OnlineSlamConfigError {
    /// `vi_init: Some(_)` but `imu: None`. The VI init stage needs the
    /// running IMU pre-integrator to promote its result into.
    ViInitRequiresImu,
    /// `vi_init.initializer.gravity_world != imu.gravity_world`. The two
    /// fields are deliberately duplicated (the initialiser is also used
    /// standalone) but they must agree at the pipeline boundary.
    GravityMismatch {
        imu_gravity_world: Vector3<f64>,
        vi_init_gravity_world: Vector3<f64>,
    },
    /// `vi_motion_init: Some(_)` but `imu: None`. The motion-based stage
    /// consumes IMU pre-integration factors emitted by the running
    /// pipeline; without an IMU stage they are never produced.
    MotionViInitRequiresImu,
    /// `vi_motion_init: Some(_)` but `vi_init: None`. The motion-based stage
    /// needs the static stage to select either its recovered bias seed or the
    /// explicit terminal give-up fallback.
    MotionViInitRequiresStaticViInit,
    /// Post-static-give-up motion initialization was requested, but the static
    /// fallback removes the IMU stage whose configured biases are required.
    MotionViInitAfterGiveUpRequiresKeepExistingSeed,
    /// `vi_motion_init.initializer.gravity_world != imu.gravity_world`.
    /// As with the static stage these are intentionally duplicated
    /// (the standalone module is also callable) but they must agree at
    /// the pipeline boundary.
    MotionGravityMismatch {
        imu_gravity_world: Vector3<f64>,
        motion_gravity_world: Vector3<f64>,
    },
    /// The stationary and motion-based stages were configured with different
    /// camera-to-body extrinsics. The motion solver converts the fixed visual
    /// camera trajectory into body poses, so both bootstrap stages must use
    /// the same calibrated transform.
    MotionExtrinsicMismatch,
    /// Local VI-BA and VI initialization were configured with different
    /// camera-to-body extrinsics, which would make the promoted inertial state
    /// and the subsequent joint residuals refer to different body frames.
    LocalViBaExtrinsicMismatch,
    /// Local VI-BA random-walk information weights must both be finite and
    /// strictly positive when configured.
    InvalidLocalViBaBiasRandomWalkWeights,
    /// Sparse keyframe factor lifecycle thresholds are internally invalid.
    InvalidSparseFactorGraphConfig,
}

impl std::fmt::Display for OnlineSlamConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ViInitRequiresImu => write!(
                f,
                "OnlineSlamConfig.vi_init is Some(_) but OnlineSlamConfig.imu is None: \
                 the auto-bootstrap stage requires the running IMU pre-integrator"
            ),
            Self::GravityMismatch {
                imu_gravity_world,
                vi_init_gravity_world,
            } => write!(
                f,
                "OnlineSlamConfig.imu.gravity_world ({imu_gravity_world:?}) and \
                 OnlineSlamConfig.vi_init.initializer.gravity_world \
                 ({vi_init_gravity_world:?}) must agree"
            ),
            Self::MotionViInitRequiresImu => write!(
                f,
                "OnlineSlamConfig.vi_motion_init is Some(_) but OnlineSlamConfig.imu \
                 is None: the motion-based stage consumes IMU pre-integration factors"
            ),
            Self::MotionViInitRequiresStaticViInit => write!(
                f,
                "OnlineSlamConfig.vi_motion_init is Some(_) but OnlineSlamConfig.vi_init \
                 is None: the motion-based stage refines the static seed produced by \
                 the static VI init stage"
            ),
            Self::MotionViInitAfterGiveUpRequiresKeepExistingSeed => write!(
                f,
                "motion VI init from configured biases requires \
                 OnlineSlamConfig.vi_init.on_persistent_rejection=KeepExistingSeed"
            ),
            Self::MotionGravityMismatch {
                imu_gravity_world,
                motion_gravity_world,
            } => write!(
                f,
                "OnlineSlamConfig.imu.gravity_world ({imu_gravity_world:?}) and \
                 OnlineSlamConfig.vi_motion_init.initializer.gravity_world \
                 ({motion_gravity_world:?}) must agree"
            ),
            Self::MotionExtrinsicMismatch => write!(
                f,
                "OnlineSlamConfig.vi_init.body_to_camera and \
                 OnlineSlamConfig.vi_motion_init.initializer.body_to_camera must agree"
            ),
            Self::LocalViBaExtrinsicMismatch => write!(
                f,
                "OnlineSlamConfig.vi_init.body_to_camera and \
                 OnlineSlamConfig.local_vi_ba.body_to_camera must agree"
            ),
            Self::InvalidLocalViBaBiasRandomWalkWeights => write!(
                f,
                "OnlineSlamConfig.local_vi_ba.bias_random_walk_weights must contain \
                 finite, strictly positive gyro and accelerometer weights"
            ),
            Self::InvalidSparseFactorGraphConfig => write!(
                f,
                "OnlineSlamConfig.sparse_factor_graph contains invalid window, confidence, \
                 proximity, budget, age, or damping thresholds"
            ),
        }
    }
}

impl std::error::Error for OnlineSlamConfigError {}

/// Read-only snapshot of the pipeline's VI initialisation stage.
/// Returned by [`crate::OnlineSlamPipeline::vi_initialization_status`].
#[derive(Debug, Clone, PartialEq)]
pub enum ViInitializationStatus {
    /// IMU + VI init both disabled, or only IMU is enabled (the
    /// pipeline never runs the auto-bootstrap when `vi_init: None`).
    Disabled,
    /// Buffering samples; `try_initialize` has not yet succeeded.
    Buffering {
        samples_buffered: usize,
        buffered_duration_seconds: f64,
        last_rejection: Option<StationaryRejectionReason>,
    },
    /// `try_initialize` succeeded; the carried result reflects what
    /// was promoted into the pipeline's running state.
    Initialised {
        result: VisualInertialInitializationResult,
    },
    /// Cap exceeded; the configured fallback has already been applied.
    GaveUp {
        last_reason: StationaryRejectionReason,
        fallback: ViInitFallback,
    },
}

/// State-transition event exposed on [`crate::OnlineSlamResult::vi_init`].
/// `Some` only on the frame where the auto-bootstrap stage actually
/// transitioned state; `None` otherwise. The terminal events
/// (`Succeeded` / `GaveUp`) are emitted at most once per sequence.
#[derive(Debug, Clone, PartialEq)]
pub enum ViInitializationEvent {
    /// `try_initialize` succeeded this frame. The pipeline has atomically:
    ///   1. Reset `imu_state.preintegrator` with the new bias linearisation.
    ///   2. Mirrored `imu_state.config.{bias_gyro, bias_acc}` to the new values.
    ///   3. Rewritten the just-registered keyframe's `Pose` if
    ///      `seed_first_keyframe_rotation` was true (rotation only; camera
    ///      centre preserved).
    ///   4. Seeded `local_vi_ba_state.keyframe_state[first_keyframe_id]`
    ///      with `velocity_world = 0, bias = (b_g, b_a)`.
    Succeeded {
        result: VisualInertialInitializationResult,
        first_keyframe_id: Option<u64>,
        discarded_stale_factor_count: usize,
    },
    /// `try_initialize` was attempted this frame and rejected; the
    /// buffer is preserved and the next attempt will run on the next
    /// keyframe.
    StillBuffering { reason: StationaryRejectionReason },
    /// `max_wait_duration_seconds` or `max_buffered_samples` was reached
    /// without success. The pipeline applied
    /// [`OnlineSlamViInitConfig::on_persistent_rejection`].
    GaveUp {
        last_reason: StationaryRejectionReason,
        fallback: ViInitFallback,
    },
}

/// Private. Owned only by [`crate::OnlineSlamPipeline`] so the
/// `completed` / `imu_state` / `map.keyframes` invariants cannot drift.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OnlineSlamViInitState {
    pub(crate) config: OnlineSlamViInitConfig,
    pub(crate) initializer: VisualInertialInitializer,
    pub(crate) completed: Option<VisualInertialInitializationResult>,
    pub(crate) gave_up: Option<StationaryRejectionReason>,
    pub(crate) last_rejection: Option<StationaryRejectionReason>,
    pub(crate) samples_buffered: usize,
    pub(crate) buffered_duration_seconds: f64,
    /// IMU factors that would have been staged by
    /// [`crate::OnlineSlamPipeline::stage_imu_factor_on_new_keyframe`]
    /// while the stale-factor gate was active. Counted so the
    /// `Succeeded` event can report the audit number; the underlying
    /// factor data itself is dropped.
    pub(crate) discarded_stale_factor_count: usize,
}

impl OnlineSlamViInitState {
    pub(crate) fn new(config: OnlineSlamViInitConfig) -> Self {
        let initializer = VisualInertialInitializer::new(config.initializer.clone());
        Self {
            config,
            initializer,
            completed: None,
            gave_up: None,
            last_rejection: None,
            samples_buffered: 0,
            buffered_duration_seconds: 0.0,
            discarded_stale_factor_count: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.initializer.reset();
        self.completed = None;
        self.gave_up = None;
        self.last_rejection = None;
        self.samples_buffered = 0;
        self.buffered_duration_seconds = 0.0;
        self.discarded_stale_factor_count = 0;
    }

    /// `true` while the auto-bootstrap stage is still active — neither
    /// `Succeeded` nor `GaveUp` has fired. Drives the stale-factor gate
    /// (factors staged in this state are discarded rather than exposed).
    pub(crate) fn is_active(&self) -> bool {
        self.completed.is_none() && self.gave_up.is_none()
    }

    pub(crate) fn snapshot(&self) -> ViInitializationStatus {
        if let Some(result) = &self.completed {
            ViInitializationStatus::Initialised {
                result: result.clone(),
            }
        } else if let Some(reason) = &self.gave_up {
            ViInitializationStatus::GaveUp {
                last_reason: reason.clone(),
                fallback: self.config.on_persistent_rejection,
            }
        } else {
            ViInitializationStatus::Buffering {
                samples_buffered: self.samples_buffered,
                buffered_duration_seconds: self.buffered_duration_seconds,
                last_rejection: self.last_rejection.clone(),
            }
        }
    }

    /// Returns `true` if either configured cap (duration / sample count)
    /// has been exceeded. Caps of `0.0` / `0` disable the corresponding
    /// check.
    pub(crate) fn cap_exceeded(&self) -> bool {
        let duration_cap = self.config.max_wait_duration_seconds;
        let sample_cap = self.config.max_buffered_samples;
        let duration_exceeded =
            duration_cap > 0.0 && self.buffered_duration_seconds >= duration_cap;
        let sample_exceeded = sample_cap > 0 && self.samples_buffered >= sample_cap;
        duration_exceeded || sample_exceeded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_try_initialize_on_every_frame_is_false() {
        let cfg = OnlineSlamViInitConfig::default();
        assert!(!cfg.try_initialize_on_every_frame);
    }

    #[test]
    fn config_round_trips_try_initialize_on_every_frame_override() {
        let cfg = OnlineSlamViInitConfig {
            try_initialize_on_every_frame: true,
            ..OnlineSlamViInitConfig::default()
        };
        assert!(cfg.try_initialize_on_every_frame);
    }
}
