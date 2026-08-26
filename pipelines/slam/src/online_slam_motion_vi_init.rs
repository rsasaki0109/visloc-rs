//! Pipeline-side glue for the motion-based VI initialiser.
//!
//! Companion to [`crate::online_slam_vi_init`]: where the static stage
//! recovers `(R_w←b, b_g, b_a)` from a stationary IMU window, this stage
//! holds the known-scale visual body-pose trajectory fixed while refining
//! per-keyframe velocity and a shared `(b_g, b_a)` once the body has moved
//! enough to give the IMU
//! translational excitation. It is the analogue of ORB-SLAM3's VIBA1 /
//! VIBA2 inside [`crate::OnlineSlamPipeline`].
//!
//! Lifecycle:
//!
//! 1. The pipeline normally feeds this stage AFTER static VI init succeeds and
//!    consumes that bias seed as its starting linearisation point. With
//!    `allow_after_static_give_up`, a motion-start sequence may instead begin
//!    after static initialization terminally fails, using configured IMU biases.
//! 2. On every new keyframe, the pipeline calls
//!    `OnlineSlamMotionViInitState::register_keyframe` with the
//!    keyframe id + world-frame camera centre, and banks the freshly-
//!    staged [`ImuPreintegrationFactor`] into
//!    `OnlineSlamMotionViInitState::factor_history`.
//! 3. Once both trigger gates (keyframe count + cumulative translation)
//!    fire, the pipeline calls [`MotionBasedViInitializer::try_initialize`]
//!    against the banked factors. On success the pipeline atomically:
//!    (a) mirrors refined `(velocity_world, bias_gyro, bias_acc)` into
//!    `local_vi_ba_state.keyframe_state` so the existing local-VI-BA
//!    trigger restarts from the refined linearisation point;
//!    (b) mirrors the refined biases onto `imu_state.config.{bias_*}` +
//!    resets the running pre-integrator's bias linearisation;
//!    (c) marks `vi_motion_init_state.completed = Some(...)` so the stage
//!    does not fire again until [`crate::OnlineSlamPipeline::reset_sequence_state`].

use std::collections::VecDeque;

use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::vi_motion_initializer::{
    GravityVelocityAlignment, GyroBiasAlignment, MotionBasedViInitializationResult,
    MotionBasedViInitializer, MotionBasedViInitializerConfig, MotionBasedViRejectionReason,
};

/// Pipeline-level configuration for the motion-based VI init stage.
///
/// Requires [`crate::OnlineSlamConfig::vi_init`] and
/// [`crate::OnlineSlamConfig::imu`] to both be `Some(_)`: the static stage
/// controls whether a recovered or explicit post-give-up bias seed is used,
/// and the IMU stream supplies the pre-integration factors that drive the solve.
#[derive(Debug, Clone, PartialEq)]
pub struct OnlineSlamMotionViInitConfig {
    /// Inner motion-based initialiser config (trigger thresholds + inner
    /// BA config).
    pub initializer: MotionBasedViInitializerConfig,
    /// Whether to mirror refined per-keyframe `(velocity, bias)` into the
    /// pipeline's `local_vi_ba_state.keyframe_state` table. Defaults to
    /// `true`; set to `false` if a caller wants the inner solve to be
    /// purely diagnostic (rare).
    pub mirror_into_local_vi_ba: bool,
    /// Whether to mirror refined biases onto `imu_state.config.bias_*` +
    /// reset the running pre-integrator. Defaults to `true`; the same
    /// hand-off the static stage performs.
    pub mirror_into_imu_state: bool,
    /// Permit the motion-based solver to start from the configured running IMU
    /// biases after the stationary initializer reaches `GaveUp` while keeping
    /// its existing seed. Off by default: callers must explicitly acknowledge
    /// that their sequence may begin in motion and that no stationary bias
    /// estimate is available.
    pub allow_after_static_give_up: bool,
    /// Permit a moving-start sequence to begin VIBA1 immediately from the
    /// configured running IMU biases, without waiting for the stationary
    /// initializer to succeed or reach its give-up horizon. This is an
    /// explicit opt-in because the configured biases may be inaccurate.
    pub allow_from_configured_bias_before_static: bool,
    /// Cap on the number of IMU factors banked in
    /// `OnlineSlamMotionViInitState::factor_history`. Bounds memory on
    /// long sequences where the trigger never fires. `0` disables the
    /// cap. Default `64`.
    pub max_buffered_factors: usize,
}

impl Default for OnlineSlamMotionViInitConfig {
    fn default() -> Self {
        Self {
            initializer: MotionBasedViInitializerConfig::default(),
            mirror_into_local_vi_ba: true,
            mirror_into_imu_state: true,
            allow_after_static_give_up: false,
            allow_from_configured_bias_before_static: false,
            max_buffered_factors: 64,
        }
    }
}

/// Read-only snapshot of the motion-based VI init stage. Returned by
/// [`crate::OnlineSlamPipeline::motion_vi_initialization_status`].
#[derive(Debug, Clone, PartialEq)]
pub enum MotionViInitializationStatus {
    /// The parent pipeline has no `vi_motion_init` config.
    Disabled,
    /// Static seed in hand; accumulating keyframes / translation until
    /// the trigger fires.
    Waiting {
        keyframes_observed: usize,
        cumulative_translation_meters: f64,
        buffered_factor_count: usize,
        last_rejection: Option<MotionBasedViRejectionReason>,
        /// `true` once a [`crate::BiasReleaseSchedule`] Stage A ("velocity
        /// stage") solve has fired and is cached on the inner
        /// [`MotionBasedViInitializer`] (see
        /// [`MotionBasedViInitializer::velocity_stage_result`]). Always
        /// `false` when no bias-release schedule is configured.
        velocity_stage_completed: bool,
        /// Mirrors [`MotionBasedViInitializer::last_gravity_alignment`]:
        /// the most recent gravity/velocity alignment attempt, kept even
        /// when `last_rejection` reports a later gate rejected it. `None`
        /// when `estimate_gravity` is off or no attempt has run yet.
        last_gravity_alignment: Option<GravityVelocityAlignment>,
        /// Mirrors [`MotionBasedViInitializer::last_gyro_bias_alignment`]:
        /// the most recent gyro-bias rotation-only alignment attempt, kept
        /// even when `last_rejection` reports a later gate rejected it.
        /// `None` when `estimate_gyro_bias` is off or no attempt has run
        /// yet.
        last_gyro_bias_alignment: Option<GyroBiasAlignment>,
    },
    /// VIBA1 has fired and succeeded; the result is the recovered
    /// refinement.
    Initialised {
        result: MotionBasedViInitializationResult,
    },
}

#[allow(clippy::large_enum_variant)]
/// State-transition event exposed on [`crate::OnlineSlamResult::vi_motion_init`].
/// `Some` only on the frame where the motion-based stage actually
/// changed state.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionViInitializationEvent {
    /// VIBA1 fired and succeeded this frame. The pipeline has atomically
    /// mirrored refined `(velocity, bias)` into the local-VI-BA state
    /// table and reset the IMU pre-integrator if so configured.
    ///
    /// Consumers MUST check `result.bias_released` before treating this as
    /// the terminal motion-VI-init outcome: when a
    /// [`crate::BiasReleaseSchedule`] is configured, this event may instead
    /// report a non-terminal Stage A ("velocity stage") firing
    /// (`bias_released == false`) — velocities were refined and mirrored,
    /// but biases still sit at the seed, and the stage remains active
    /// awaiting a later Stage B firing (`bias_released == true`). `true`
    /// on both the legacy single-stage path and Stage B.
    Succeeded {
        result: MotionBasedViInitializationResult,
    },
    /// Trigger evaluated this frame but rejected (gate not yet met, no
    /// usable factor, or solver failure).
    StillWaiting {
        reason: MotionBasedViRejectionReason,
    },
}

/// Private. Owned only by [`crate::OnlineSlamPipeline`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OnlineSlamMotionViInitState {
    pub(crate) config: OnlineSlamMotionViInitConfig,
    pub(crate) initializer: MotionBasedViInitializer,
    pub(crate) completed: Option<MotionBasedViInitializationResult>,
    pub(crate) last_rejection: Option<MotionBasedViRejectionReason>,
    /// Ring buffer of recently-staged IMU pre-integration factors.
    /// Capped by `config.max_buffered_factors`. The motion-based solver
    /// scans this buffer for factors that connect any pair of registered
    /// keyframes.
    pub(crate) factor_history: VecDeque<ImuPreintegrationFactor>,
}

impl OnlineSlamMotionViInitState {
    pub(crate) fn new(config: OnlineSlamMotionViInitConfig) -> Self {
        let initializer = MotionBasedViInitializer::new(config.initializer.clone());
        Self {
            config,
            initializer,
            completed: None,
            last_rejection: None,
            factor_history: VecDeque::new(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.initializer.reset();
        self.completed = None;
        self.last_rejection = None;
        self.factor_history.clear();
    }

    /// `true` while the motion-based stage is waiting for its trigger.
    /// Mirrors the convention used by [`crate::OnlineSlamViInitState::is_active`].
    pub(crate) fn is_active(&self) -> bool {
        self.completed.is_none()
    }

    /// `true` once a [`crate::BiasReleaseSchedule`] Stage A ("velocity
    /// stage") solve has fired on the inner [`MotionBasedViInitializer`],
    /// regardless of whether the stage has since gone on to reach the
    /// terminal Stage B `completed` state. Always `false` when no
    /// `bias_release` schedule is configured (the legacy path never parks
    /// in a non-terminal success, so it either has `completed.is_some()`
    /// or has not fired at all).
    ///
    /// Used by [`crate::OnlineSlamPipeline::vi_initialization_pending`] to
    /// decide whether local VI-BA may run: a Stage A firing has already
    /// replaced the placeholder-zero bias linearisation with the
    /// refined/estimated seed (see that method's doc comment), so it
    /// counts as "no longer pending" exactly like a terminal `completed`.
    pub(crate) fn velocity_stage_fired(&self) -> bool {
        self.initializer.velocity_stage_result().is_some()
    }

    pub(crate) fn snapshot(&self) -> MotionViInitializationStatus {
        if let Some(result) = &self.completed {
            MotionViInitializationStatus::Initialised {
                result: result.clone(),
            }
        } else {
            MotionViInitializationStatus::Waiting {
                keyframes_observed: self.initializer.keyframes_observed(),
                cumulative_translation_meters: self.initializer.cumulative_translation_meters(),
                buffered_factor_count: self.factor_history.len(),
                last_rejection: self.last_rejection.clone(),
                velocity_stage_completed: self.initializer.velocity_stage_result().is_some(),
                last_gravity_alignment: self.initializer.last_gravity_alignment().cloned(),
                last_gyro_bias_alignment: self.initializer.last_gyro_bias_alignment().cloned(),
            }
        }
    }

    /// Push a freshly-staged IMU factor onto the rolling history, evicting
    /// the oldest if `max_buffered_factors` would be exceeded.
    pub(crate) fn push_factor(&mut self, factor: ImuPreintegrationFactor) {
        self.factor_history.push_back(factor);
        let cap = self.config.max_buffered_factors;
        if cap > 0 {
            while self.factor_history.len() > cap {
                self.factor_history.pop_front();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};

    use crate::imu_preintegration::ImuPreintegratedDelta;

    fn test_factor(from: u64, to: u64) -> ImuPreintegrationFactor {
        ImuPreintegrationFactor {
            keyframe_id_from: from,
            keyframe_id_to: to,
            delta: ImuPreintegratedDelta::identity(),
            gravity_world: Vector3::zeros(),
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    #[test]
    fn factor_history_respects_cap() {
        let mut state = OnlineSlamMotionViInitState::new(OnlineSlamMotionViInitConfig {
            max_buffered_factors: 2,
            ..OnlineSlamMotionViInitConfig::default()
        });
        state.push_factor(test_factor(1, 2));
        state.push_factor(test_factor(2, 3));
        state.push_factor(test_factor(3, 4));
        assert_eq!(state.factor_history.len(), 2);
        // The oldest (1→2) was evicted; the remaining two are 2→3 and 3→4.
        let front = state.factor_history.front().unwrap();
        assert_eq!(front.keyframe_id_from, 2);
    }

    #[test]
    fn factor_history_cap_zero_disables_eviction() {
        let mut state = OnlineSlamMotionViInitState::new(OnlineSlamMotionViInitConfig {
            max_buffered_factors: 0,
            ..OnlineSlamMotionViInitConfig::default()
        });
        for i in 0..10 {
            state.push_factor(test_factor(i, i + 1));
        }
        assert_eq!(state.factor_history.len(), 10);
    }

    #[test]
    fn reset_clears_everything() {
        let mut state = OnlineSlamMotionViInitState::new(OnlineSlamMotionViInitConfig::default());
        state.push_factor(test_factor(1, 2));
        state
            .initializer
            .register_keyframe(1, Point3::new(0.0, 0.0, 0.0));
        state
            .initializer
            .register_keyframe(2, Point3::new(1.0, 0.0, 0.0));
        assert_eq!(state.initializer.keyframes_observed(), 2);
        assert_eq!(state.factor_history.len(), 1);
        state.reset();
        assert!(state.factor_history.is_empty());
        assert_eq!(state.initializer.keyframes_observed(), 0);
        assert!(state.completed.is_none());
        assert!(state.last_rejection.is_none());
    }

    #[test]
    fn snapshot_reports_waiting_with_buffered_counts() {
        let state = OnlineSlamMotionViInitState::new(OnlineSlamMotionViInitConfig::default());
        let snap = state.snapshot();
        match snap {
            MotionViInitializationStatus::Waiting {
                keyframes_observed,
                cumulative_translation_meters,
                buffered_factor_count,
                last_rejection,
                velocity_stage_completed,
                last_gravity_alignment,
                last_gyro_bias_alignment,
            } => {
                assert_eq!(keyframes_observed, 0);
                assert!(cumulative_translation_meters.abs() < 1e-12);
                assert_eq!(buffered_factor_count, 0);
                assert!(last_rejection.is_none());
                assert!(!velocity_stage_completed);
                assert!(last_gravity_alignment.is_none());
                assert!(last_gyro_bias_alignment.is_none());
            }
            other => panic!("expected Waiting, got {other:?}"),
        }
    }
}
