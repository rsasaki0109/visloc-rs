//! Motion-based VI initialiser — first cut (VIBA1, stereo / known-scale
//! path).
//!
//! Companion to the stationary-window [`crate::VisualInertialInitializer`].
//! Where the static seed recovers `(R_w←b, b_g, b_a)` from a stationary
//! IMU window (yaw and monocular scale unobservable, accel-bias
//! observability partial), this stage refines the per-keyframe
//! `(R_w←b, v_w, b_g, b_a)` once the body has moved enough to give the
//! IMU translational excitation. It is the analogue of ORB-SLAM3's
//! `VIBA1` step.
//!
//! Triggering. The state machine accumulates camera centres reported by
//! the pipeline via [`MotionBasedViInitializer::register_keyframe`] and
//! fires when **both** of these are satisfied (defaults from
//! [`MotionBasedViInitializerConfig::default`], drawn from the design
//! note `docs/motion_based_vi_alignment.md`):
//!
//! * `keyframes_observed >= min_keyframes` (default `10`)
//! * `cumulative_translation_meters >= min_translation_meters` (default `2.0`)
//!
//! Solve. [`MotionBasedViInitializer::try_initialize`] invokes
//! [`crate::run_inertial_only_vi_ba`] over the accumulated keyframe set,
//! which holds the vision-only poses fixed and optimises velocities plus a
//! shared short-window bias against IMU preintegration factors (landmarks are
//! not touched). Scale is fixed at `1.0`; stereo / RGB-D sequences and
//! monocular sequences with a known metric anchor are well-served.
//!
//! The optional VIBA2 outer loop handles experimental monocular scale
//! recovery. Full joint visual-inertial BA and consistent marginalization
//! remain separate follow-ups.

use std::collections::BTreeMap;

use nalgebra::{Point3, Vector3};
use visloc_core::geometry::SE3;
use visloc_core::types::VisualMap;

use crate::bundle::{BaConfig, BaResult};
use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::online_slam_vi_ba::{
    run_inertial_only_vi_ba, run_viba2_inertial_with_scale, KeyframeImuState, Viba2Config,
};
use crate::vi_initializer::VisualInertialInitializationResult;
use crate::LinearSolver;

/// Configuration for [`MotionBasedViInitializer`].
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializerConfig {
    /// Minimum number of keyframes (registered after the static seed)
    /// before the VIBA1 trigger is allowed to fire. Default `10`.
    pub min_keyframes: usize,
    /// Minimum cumulative camera-centre translation in metres before
    /// VIBA1 fires. Default `2.0`. Set to `0.0` to disable the
    /// translation gate.
    pub min_translation_meters: f64,
    /// World-frame gravity vector. Echoed onto the BA result for
    /// downstream diagnostics; the IMU factors fed into the solve carry
    /// their own gravity already.
    pub gravity_world: Vector3<f64>,
    /// Rigid transform from the tracked camera/sensor frame into the IMU body
    /// frame (EuRoC `T_BS` for cam0). The visual map stores world-to-camera
    /// poses while preintegration residuals operate on world-to-body poses.
    /// Identity preserves the co-located body/camera convention.
    pub body_to_camera: SE3,
    /// Inner LM solver config. Default mirrors
    /// [`crate::OnlineSlamLocalBaConfig::default`] (sparse linear
    /// solver, 10 LM iterations).
    pub ba_config: BaConfig,
    /// Optional VIBA2 hand-off. When `Some`, the initialiser runs an
    /// outer scale-recovery loop on top of the VIBA1 inertial-only
    /// solve. Stereo / RGB-D / known-scale paths should leave this
    /// `None` or set `recover_scale = false`; monocular paths set
    /// `recover_scale = true`. See [`crate::Viba2Config`] for the
    /// outer-loop knobs. Default `None`.
    pub viba2: Option<Viba2Config>,
    /// Post-solve sanity gate on the recovered per-keyframe
    /// `velocity_world` magnitudes. When `Some(v)`, the initialiser
    /// rejects the inner LM result if any keyframe's
    /// `||velocity_world|| > v` and parks itself in the `Waiting` state
    /// with [`MotionBasedViRejectionReason::VelocityOutOfRange`].
    /// `None` (default) preserves the legacy behaviour of accepting any
    /// LM-converged result. Indoor V1-class EuRoC sequences run safely
    /// at `Some(10.0)` (~36 km/h ceiling); set higher for outdoor
    /// driving datasets.
    pub max_velocity_magnitude_mps: Option<f64>,
    /// Optional post-solve upper bound on every recovered gyro-bias vector
    /// magnitude (rad/s). `None` preserves legacy behavior.
    pub max_gyro_bias_magnitude_rad_s: Option<f64>,
    /// Optional post-solve upper bound on every recovered accelerometer-bias
    /// vector magnitude (m/s²). `None` preserves legacy behavior.
    pub max_accel_bias_magnitude_mps2: Option<f64>,
}

impl Default for MotionBasedViInitializerConfig {
    fn default() -> Self {
        Self {
            min_keyframes: 10,
            min_translation_meters: 2.0,
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            body_to_camera: SE3::identity(),
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                max_iterations: 10,
                ..BaConfig::default()
            },
            viba2: None,
            max_velocity_magnitude_mps: None,
            max_gyro_bias_magnitude_rad_s: None,
            max_accel_bias_magnitude_mps2: None,
        }
    }
}

/// VIBA1 / VIBA2 outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializationResult {
    /// Refined per-keyframe `(velocity_world, bias_gyro, bias_acc)`.
    pub keyframe_states: BTreeMap<u64, KeyframeImuState>,
    /// Sorted keyframe ids covered by the solve.
    pub keyframe_ids: Vec<u64>,
    /// Number of IMU factors fed into the solve.
    pub imu_factors_used: usize,
    /// Scale factor recovered by the solve. `1.0` on the VIBA1-only
    /// path (no `viba2` config or `recover_scale = false`); the closed-
    /// form least-squares scale on the VIBA2 monocular path.
    pub scale: f64,
    /// Outer-loop scale history when VIBA2 ran (one entry per outer
    /// iteration, starting with `Viba2Config::initial_scale`). Empty
    /// on the VIBA1-only path.
    pub scale_history: Vec<f64>,
    /// Number of VIBA2 outer iterations executed. `0` on the VIBA1-only
    /// path.
    pub viba2_iterations_run: usize,
    /// Cumulative translation (m) since the static seed at the moment
    /// the trigger fired. Surface for logging / diagnostics.
    pub trigger_translation_meters: f64,
    /// Inner LM solve outcome (from the final VIBA2 inner solve when
    /// applicable, else the single VIBA1 solve).
    pub ba_result: BaResult,
}

/// Why a [`MotionBasedViInitializer::try_initialize`] call returned
/// without a fresh result.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionBasedViRejectionReason {
    /// Fewer than `min_keyframes` keyframes have been registered.
    InsufficientKeyframes { have: usize, need: usize },
    /// Accumulated camera-centre translation is below the trigger
    /// threshold.
    InsufficientTranslation { have: f64, need: f64 },
    /// The keyframe set has at least `min_keyframes` entries and meets
    /// the translation gate, but none of the supplied preintegration
    /// factors connect any pair of registered keyframes — the IMU
    /// stream has not yet emitted factors over this window, or the
    /// caller passed an unrelated factor slice.
    NoUsableImuFactors,
    /// One or more registered keyframes are missing their pose, or the
    /// camera lookup failed.
    MissingKeyframeData,
    /// Inner LM solver failed to converge (singular system, etc.).
    SolverFailed,
    /// Inner LM solver returned, but at least one per-keyframe
    /// `||velocity_world||` exceeded
    /// `MotionBasedViInitializerConfig::max_velocity_magnitude_mps`. The
    /// solver's bias / velocity slots are NOT promoted to `self.completed`,
    /// and its speculative map poses are discarded, so the next trigger
    /// re-runs from the same linearisation point. `kf_id` is the worst-offender keyframe;
    /// `magnitude_mps` is its recovered speed; `limit_mps` echoes the
    /// configured gate.
    VelocityOutOfRange {
        kf_id: u64,
        magnitude_mps: f64,
        limit_mps: f64,
    },
    /// A recovered gyro bias exceeded the configured physical sanity bound.
    GyroBiasOutOfRange {
        kf_id: u64,
        magnitude_rad_s: f64,
        limit_rad_s: f64,
    },
    /// A recovered accelerometer bias exceeded the configured physical sanity
    /// bound.
    AccelBiasOutOfRange {
        kf_id: u64,
        magnitude_mps2: f64,
        limit_mps2: f64,
    },
}

/// Read-only snapshot of [`MotionBasedViInitializer`]'s internal state.
#[derive(Debug, Clone, PartialEq)]
pub enum MotionBasedViInitializationStatus {
    /// Trigger has not yet fired. Reports the accumulated metrics plus
    /// the last `Err(...)` returned by [`MotionBasedViInitializer::try_initialize`]
    /// (if any) so callers can render diagnostic progress.
    Waiting {
        keyframes_observed: usize,
        cumulative_translation_meters: f64,
        last_rejection: Option<MotionBasedViRejectionReason>,
    },
    /// VIBA1 has fired and succeeded; the result is the recovered
    /// refinement.
    Initialised {
        result: MotionBasedViInitializationResult,
    },
}

/// State machine that fires a single VIBA1 inertial-only solve once the
/// keyframe count + cumulative-translation thresholds are met. See the
/// module docstring and the design note
/// `docs/motion_based_vi_alignment.md` for the full contract.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionBasedViInitializer {
    config: MotionBasedViInitializerConfig,
    keyframes: Vec<(u64, Point3<f64>)>,
    cumulative_translation: f64,
    completed: Option<MotionBasedViInitializationResult>,
    last_rejection: Option<MotionBasedViRejectionReason>,
    /// Track whether this initializer's `register_keyframe` should
    /// charge the inter-keyframe distance against the
    /// `cumulative_translation` running sum. Always `true` after the
    /// first `register_keyframe` call; the first call only seeds the
    /// chain (zero translation by definition).
    has_seed_center: bool,
}

impl MotionBasedViInitializer {
    /// Construct with the given config; carries no keyframes yet.
    pub fn new(config: MotionBasedViInitializerConfig) -> Self {
        Self {
            config,
            keyframes: Vec::new(),
            cumulative_translation: 0.0,
            completed: None,
            last_rejection: None,
            has_seed_center: false,
        }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &MotionBasedViInitializerConfig {
        &self.config
    }

    /// Drop every accumulated keyframe and the cached VIBA1 result. The
    /// configuration is preserved. Mirrors
    /// [`crate::VisualInertialInitializer::reset`].
    pub fn reset(&mut self) {
        self.keyframes.clear();
        self.cumulative_translation = 0.0;
        self.completed = None;
        self.last_rejection = None;
        self.has_seed_center = false;
    }

    /// Number of keyframes registered since construction / last reset.
    pub fn keyframes_observed(&self) -> usize {
        self.keyframes.len()
    }

    /// Cumulative camera-centre translation (m) along the registered
    /// keyframe chain.
    pub fn cumulative_translation_meters(&self) -> f64 {
        self.cumulative_translation
    }

    /// Note a freshly-registered keyframe's camera centre in world
    /// frame. The first call after construction / reset only seeds the
    /// chain; subsequent calls add the inter-keyframe distance to the
    /// running cumulative-translation total.
    ///
    /// Idempotent against re-registering the same `keyframe_id`: a
    /// second call with the same id silently overwrites the prior
    /// centre without double-counting. (This matters when a pipeline
    /// fires a "new keyframe" event and immediately corrects the pose
    /// via local-BA on the same frame.)
    pub fn register_keyframe(&mut self, keyframe_id: u64, camera_center_world: Point3<f64>) {
        if let Some(slot) = self.keyframes.iter_mut().find(|(id, _)| *id == keyframe_id) {
            slot.1 = camera_center_world;
            return;
        }
        if self.has_seed_center {
            if let Some(prev) = self.keyframes.last() {
                self.cumulative_translation += (camera_center_world - prev.1).norm();
            }
        } else {
            self.has_seed_center = true;
        }
        self.keyframes.push((keyframe_id, camera_center_world));
    }

    /// `true` once both trigger gates (keyframe count + translation)
    /// are satisfied. `try_initialize` will still return Err if no IMU
    /// factor connects any pair, but `is_ready` is the cheap pre-check
    /// the pipeline can poll on every frame.
    pub fn is_ready(&self) -> bool {
        self.completed.is_none()
            && self.keyframes.len() >= self.config.min_keyframes
            && self.cumulative_translation >= self.config.min_translation_meters
    }

    /// Returns the cached result if VIBA1 has already fired.
    pub fn result(&self) -> Option<&MotionBasedViInitializationResult> {
        self.completed.as_ref()
    }

    /// Render a read-only snapshot of the current state.
    pub fn status(&self) -> MotionBasedViInitializationStatus {
        if let Some(result) = &self.completed {
            MotionBasedViInitializationStatus::Initialised {
                result: result.clone(),
            }
        } else {
            MotionBasedViInitializationStatus::Waiting {
                keyframes_observed: self.keyframes.len(),
                cumulative_translation_meters: self.cumulative_translation,
                last_rejection: self.last_rejection.clone(),
            }
        }
    }

    /// Attempt to fire VIBA1 over the registered keyframes.
    ///
    /// On success, the inertial-only solve refines `map.keyframes[*]`'s
    /// poses in place and the cached
    /// [`MotionBasedViInitializationResult`] is returned. The
    /// initialiser is single-shot: a successful firing parks the
    /// initialiser in the `Initialised` state and subsequent calls
    /// short-circuit to the cached result. Call [`Self::reset`] to
    /// re-arm for a new sequence.
    ///
    /// `static_seed` is consumed only as a sanity seed for downstream
    /// `keyframe_states` (the inertial-only solver pulls per-keyframe
    /// initial values from `map.keyframes[*]`'s poses + the static
    /// seed's biases / velocity).
    pub fn try_initialize(
        &mut self,
        map: &mut VisualMap,
        preintegration_factors: &[ImuPreintegrationFactor],
        static_seed: &VisualInertialInitializationResult,
    ) -> Result<&MotionBasedViInitializationResult, MotionBasedViRejectionReason> {
        self.try_initialize_with_bias_seed(
            map,
            preintegration_factors,
            static_seed.bias_gyro,
            static_seed.bias_acc,
        )
    }

    /// Motion-based initialization from explicit bias linearisation values.
    /// This is the safe fallback entry point for sequences that begin in
    /// motion: the caller may retain its calibrated/configured biases after a
    /// stationary initializer gives up, without fabricating a false static
    /// gravity/bias result. The solver still requires its normal keyframe and
    /// translation excitation gates.
    pub fn try_initialize_with_bias_seed(
        &mut self,
        map: &mut VisualMap,
        preintegration_factors: &[ImuPreintegrationFactor],
        bias_gyro_seed: Vector3<f64>,
        bias_acc_seed: Vector3<f64>,
    ) -> Result<&MotionBasedViInitializationResult, MotionBasedViRejectionReason> {
        if let Some(_existing) = &self.completed {
            // SAFETY: borrow checker — re-fetch by `Option::as_ref` so
            // the `&mut self` borrow above can release before the
            // `&self` borrow below acquires.
            return Ok(self.completed.as_ref().expect("just checked"));
        }

        let observed = self.keyframes.len();
        if observed < self.config.min_keyframes {
            let err = MotionBasedViRejectionReason::InsufficientKeyframes {
                have: observed,
                need: self.config.min_keyframes,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }
        if self.cumulative_translation < self.config.min_translation_meters {
            let err = MotionBasedViRejectionReason::InsufficientTranslation {
                have: self.cumulative_translation,
                need: self.config.min_translation_meters,
            };
            self.last_rejection = Some(err.clone());
            return Err(err);
        }

        // Seed per-keyframe states for the solver. The first keyframe
        // (the gauge anchor for the inertial-only solve) inherits the
        // static seed's biases + zero velocity; subsequent keyframes
        // also start from those biases but seed velocity from the
        // inter-keyframe centre displacement and the connecting IMU
        // factor's `delta_time` when available.
        let mut initial_states: BTreeMap<u64, KeyframeImuState> = BTreeMap::new();
        let kf_ids: Vec<u64> = self.keyframes.iter().map(|(id, _)| *id).collect();
        for (idx, &kf_id) in kf_ids.iter().enumerate() {
            let velocity = if idx == 0 {
                Vector3::zeros()
            } else {
                let prev_id = kf_ids[idx - 1];
                let factor = preintegration_factors
                    .iter()
                    .find(|f| f.keyframe_id_from == prev_id && f.keyframe_id_to == kf_id);
                match factor {
                    Some(f) if f.delta.delta_time > 0.0 => {
                        let prev_center = self.keyframes[idx - 1].1;
                        let curr_center = self.keyframes[idx].1;
                        (curr_center - prev_center) / f.delta.delta_time
                    }
                    _ => Vector3::zeros(),
                }
            };
            initial_states.insert(
                kf_id,
                KeyframeImuState {
                    velocity_world: velocity,
                    bias_gyro: bias_gyro_seed,
                    bias_acc: bias_acc_seed,
                },
            );
        }

        // Validate that all registered keyframes have a pose in `map`.
        for kf_id in &kf_ids {
            let Some(kf) = map.keyframes.get(kf_id) else {
                let err = MotionBasedViRejectionReason::MissingKeyframeData;
                self.last_rejection = Some(err.clone());
                return Err(err);
            };
            if kf.frame.pose.is_none() {
                let err = MotionBasedViRejectionReason::MissingKeyframeData;
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }

        // Solve transactionally. The inertial BA helpers write refined poses
        // into their map argument before returning, but post-solve velocity
        // gates below may still reject the result. Keep those speculative
        // poses on a clone and publish them only after every gate passes.
        let mut candidate_map = map.clone();
        // The visual map stores T_cw, while the IMU factors are expressed in
        // body coordinates. EuRoC T_BS for cam0 is T_bc (camera/sensor to
        // body), hence T_bw = T_bc * T_cw. This conversion is speculative and
        // exists only on the solver clone; the tracked camera map is never
        // overwritten with body poses.
        for kf_id in &kf_ids {
            let pose = candidate_map
                .keyframes
                .get_mut(kf_id)
                .and_then(|kf| kf.frame.pose.as_mut())
                .expect("keyframe poses were validated above");
            pose.world_to_camera = self.config.body_to_camera.compose(&pose.world_to_camera);
        }

        // Dispatch on `viba2` config: when `Some`, run the VIBA2 outer
        // scale-recovery loop; otherwise run the standalone VIBA1
        // inertial-only path. Both paths return the same
        // `MotionBasedViInitializationResult` shape; the VIBA1-only
        // path leaves `scale_history` empty and `viba2_iterations_run`
        // at `0`.
        let result = if let Some(viba2_cfg) = self.config.viba2.clone() {
            let stats = run_viba2_inertial_with_scale(
                &mut candidate_map,
                &kf_ids,
                preintegration_factors,
                &initial_states,
                &viba2_cfg,
            );
            let stats = match stats {
                Some(s) => s,
                None => {
                    let any_in_window = preintegration_factors.iter().any(|f| {
                        kf_ids.contains(&f.keyframe_id_from) && kf_ids.contains(&f.keyframe_id_to)
                    });
                    let err = if any_in_window {
                        MotionBasedViRejectionReason::SolverFailed
                    } else {
                        MotionBasedViRejectionReason::NoUsableImuFactors
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            MotionBasedViInitializationResult {
                keyframe_states: stats.keyframe_states,
                keyframe_ids: stats.keyframe_ids,
                imu_factors_used: stats.imu_factor_count,
                scale: stats.scale,
                scale_history: stats.scale_history,
                viba2_iterations_run: stats.outer_iterations_run,
                trigger_translation_meters: self.cumulative_translation,
                ba_result: stats.ba_result,
            }
        } else {
            let stats = run_inertial_only_vi_ba(
                &mut candidate_map,
                &kf_ids,
                preintegration_factors,
                &initial_states,
                &self.config.ba_config,
            );
            let stats = match stats {
                Some(s) => s,
                None => {
                    let any_in_window = preintegration_factors.iter().any(|f| {
                        kf_ids.contains(&f.keyframe_id_from) && kf_ids.contains(&f.keyframe_id_to)
                    });
                    let err = if any_in_window {
                        MotionBasedViRejectionReason::SolverFailed
                    } else {
                        MotionBasedViRejectionReason::NoUsableImuFactors
                    };
                    self.last_rejection = Some(err.clone());
                    return Err(err);
                }
            };
            MotionBasedViInitializationResult {
                keyframe_states: stats.keyframe_states,
                keyframe_ids: stats.keyframe_ids,
                imu_factors_used: stats.imu_factor_count,
                scale: 1.0,
                scale_history: Vec::new(),
                viba2_iterations_run: 0,
                trigger_translation_meters: self.cumulative_translation,
                ba_result: stats.ba_result,
            }
        };

        if let Some(limit) = self.config.max_velocity_magnitude_mps {
            let mut worst: Option<(u64, f64)> = None;
            for (kf_id, state) in &result.keyframe_states {
                let mag = state.velocity_world.norm();
                if mag > limit && worst.is_none_or(|(_, m)| mag > m) {
                    worst = Some((*kf_id, mag));
                }
            }
            if let Some((kf_id, mag)) = worst {
                let err = MotionBasedViRejectionReason::VelocityOutOfRange {
                    kf_id,
                    magnitude_mps: mag,
                    limit_mps: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }
        if let Some(limit) = self.config.max_gyro_bias_magnitude_rad_s {
            let worst = result
                .keyframe_states
                .iter()
                .map(|(kf_id, state)| (*kf_id, state.bias_gyro.norm()))
                .filter(|(_, magnitude)| *magnitude > limit)
                .max_by(|left, right| left.1.total_cmp(&right.1));
            if let Some((kf_id, magnitude_rad_s)) = worst {
                let err = MotionBasedViRejectionReason::GyroBiasOutOfRange {
                    kf_id,
                    magnitude_rad_s,
                    limit_rad_s: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }
        if let Some(limit) = self.config.max_accel_bias_magnitude_mps2 {
            let worst = result
                .keyframe_states
                .iter()
                .map(|(kf_id, state)| (*kf_id, state.bias_acc.norm()))
                .filter(|(_, magnitude)| *magnitude > limit)
                .max_by(|left, right| left.1.total_cmp(&right.1));
            if let Some((kf_id, magnitude_mps2)) = worst {
                let err = MotionBasedViRejectionReason::AccelBiasOutOfRange {
                    kf_id,
                    magnitude_mps2,
                    limit_mps2: limit,
                };
                self.last_rejection = Some(err.clone());
                return Err(err);
            }
        }

        self.completed = Some(result);
        self.last_rejection = None;
        Ok(self.completed.as_ref().expect("just inserted"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::{Point3, UnitQuaternion};
    use visloc_core::geometry::{Pose, SO3};
    use visloc_core::types::{Camera, Frame, Keyframe, VisualMap};

    use crate::imu_preintegration::ImuPreintegratedDelta;

    fn pinhole_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn pose_at_world_center(c: Vector3<f64>) -> Pose {
        // World-to-camera with identity orientation: t = -R·C = -C.
        let r = UnitQuaternion::identity();
        Pose::from_world_to_camera(r, -r.transform_vector(&c))
    }

    fn synthetic_seed() -> VisualInertialInitializationResult {
        VisualInertialInitializationResult {
            gravity_world: Vector3::new(0.0, 9.81, 0.0),
            initial_rotation_body_to_world: UnitQuaternion::identity(),
            initial_velocity_world: Vector3::zeros(),
            bias_gyro: Vector3::zeros(),
            bias_acc: Vector3::zeros(),
            samples_consumed: 2000,
            duration_seconds: 1.0,
            gyro_std: Vector3::zeros(),
            accel_std: Vector3::zeros(),
            mean_accel_magnitude: 9.81,
        }
    }

    fn no_acceleration_factor(
        from_id: u64,
        to_id: u64,
        delta_t: f64,
        gravity: Vector3<f64>,
    ) -> ImuPreintegrationFactor {
        // Synthetic factor for an identity-oriented body with NO
        // proper acceleration (i.e., stationary or constant-velocity
        // in world frame). The IMU still senses gravity as specific
        // force, so the pre-integrated body-frame deltas are:
        //   ΔR = identity
        //   Δv = -g · Δt
        //   Δp = -0.5 · g · Δt²
        // With these, the residual
        //   r_v = R_b←w · (v_j - v_i - g·Δt) - Δv
        //       = (v_j - v_i - g·Δt) + g·Δt = v_j - v_i
        // is exactly satisfied for stationary AND for constant-velocity
        // motion (because the IMU integration's gravity term cancels the
        // residual's gravity term), as required for the "no proper
        // acceleration" synthetic scenarios used in these unit tests.
        let mut delta = ImuPreintegratedDelta::identity();
        delta.delta_time = delta_t;
        delta.delta_velocity = -gravity * delta_t;
        delta.delta_position = -0.5 * gravity * delta_t * delta_t;
        ImuPreintegrationFactor {
            keyframe_id_from: from_id,
            keyframe_id_to: to_id,
            delta,
            gravity_world: gravity,
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    fn build_constant_velocity_map(num_keyframes: usize) -> VisualMap {
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..num_keyframes {
            let center = Vector3::new(i as f64, 0.0, 0.0);
            let pose = pose_at_world_center(center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        map
    }

    #[test]
    fn trigger_blocked_until_keyframe_threshold_is_met() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 5,
            min_translation_meters: 0.0,
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..4 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        assert!(!init.is_ready(), "fewer keyframes than the threshold");
        let mut map = build_constant_velocity_map(4);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("threshold not met");
        match err {
            MotionBasedViRejectionReason::InsufficientKeyframes { have, need } => {
                assert_eq!(have, 4);
                assert_eq!(need, 5);
            }
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn trigger_blocked_until_translation_threshold_is_met() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 5.0,
            ..MotionBasedViInitializerConfig::default()
        });
        init.register_keyframe(1, Point3::new(0.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.0, 0.0, 0.0));
        init.register_keyframe(3, Point3::new(2.0, 0.0, 0.0));
        assert_eq!(init.keyframes_observed(), 3);
        assert!(
            (init.cumulative_translation_meters() - 2.0).abs() < 1e-12,
            "cumulative_translation tracks chain length, not endpoint distance"
        );
        assert!(!init.is_ready());
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("translation threshold not met");
        match err {
            MotionBasedViRejectionReason::InsufficientTranslation { have, need } => {
                assert!((have - 2.0).abs() < 1e-12);
                assert!((need - 5.0).abs() < 1e-12);
            }
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn re_registering_same_keyframe_does_not_double_count_translation() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig::default());
        init.register_keyframe(1, Point3::new(0.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.0, 0.0, 0.0));
        init.register_keyframe(2, Point3::new(1.5, 0.0, 0.0));
        assert_eq!(init.keyframes_observed(), 2);
        // Only the inter-frame displacement from the FIRST insertion is
        // counted — the re-registration overwrites the centre but does
        // not append to the chain.
        assert!((init.cumulative_translation_meters() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_usable_factor_is_reported_distinctly_from_solver_failure() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        assert!(init.is_ready());
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let err = init
            .try_initialize(&mut map, &[], &seed)
            .expect_err("no factors registered");
        match err {
            MotionBasedViRejectionReason::NoUsableImuFactors => {}
            other => panic!("unexpected rejection: {other:?}"),
        }
    }

    #[test]
    fn successful_solve_is_cached_and_reset_re_arms() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];

        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA1 must succeed on synthetic constant-velocity stream")
            .clone();
        assert_eq!(
            map, map_before,
            "inertial-only initialization must keep the visual trajectory fixed"
        );
        assert_eq!(result.keyframe_ids, vec![1, 2, 3]);
        assert_eq!(result.imu_factors_used, 2);
        assert!((result.scale - 1.0).abs() < 1e-12);
        assert!((result.trigger_translation_meters - 2.0).abs() < 1e-12);
        // Solver populated every registered keyframe with a state slot,
        // preserved the fixed vision-only trajectory, and terminated below
        // the initial cost.
        for kf_id in [1u64, 2, 3] {
            assert!(result.keyframe_states.contains_key(&kf_id));
        }
        assert!(result.ba_result.final_cost <= result.ba_result.initial_cost + 1e-9);
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected Initialised, got {other:?}"),
        }

        // Re-firing returns the cached result rather than re-running.
        let cached = init.try_initialize(&mut map, &factors, &seed).unwrap();
        assert_eq!(cached.keyframe_ids, result.keyframe_ids);

        // After reset, the state machine returns to Waiting and can
        // re-fire on a fresh sequence.
        init.reset();
        assert_eq!(init.keyframes_observed(), 0);
        assert!((init.cumulative_translation_meters() - 0.0).abs() < 1e-12);
        match init.status() {
            MotionBasedViInitializationStatus::Waiting {
                keyframes_observed,
                cumulative_translation_meters,
                ..
            } => {
                assert_eq!(keyframes_observed, 0);
                assert!(cumulative_translation_meters.abs() < 1e-12);
            }
            other => panic!("expected Waiting after reset, got {other:?}"),
        }
    }

    #[test]
    fn stereo_stationary_replay_is_identity() {
        // VIBA1 on three coincident keyframes (zero motion) reproduces
        // the static seed: zero velocity, zero biases. The trigger is
        // forced by relaxing `min_keyframes` and `min_translation_meters`
        // so the "no motion" case is still solvable for the no-op check.
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 0.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(0.0, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let pose = pose_at_world_center(Vector3::zeros());
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 0.1, gravity),
            no_acceleration_factor(2, 3, 0.1, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("identity replay should succeed");
        for state in result.keyframe_states.values() {
            assert!(
                state.velocity_world.norm() < 1e-6,
                "vel = {:?}",
                state.velocity_world
            );
            assert!(state.bias_gyro.norm() < 1e-6, "b_g = {:?}", state.bias_gyro);
            assert!(state.bias_acc.norm() < 1e-6, "b_a = {:?}", state.bias_acc);
        }
    }

    #[test]
    fn max_velocity_magnitude_gate_rejects_when_exceeded() {
        // Run a known-converging setup TWICE: once without the gate to
        // record the actual recovered max-velocity magnitude, then once
        // with `max_velocity_magnitude_mps = Some(max / 2.0)` so the
        // post-solve sanity gate must fire. Decoupling the threshold
        // from a hard-coded magnitude keeps the test stable against
        // inner-LM precision drift.
        let mut probe = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = probe.config().gravity_world;
        for i in 0..3 {
            probe.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut probe_map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let probe_result = probe
            .try_initialize(&mut probe_map, &factors, &seed)
            .expect("probe run must converge")
            .clone();
        let max_mag = probe_result
            .keyframe_states
            .values()
            .map(|s| s.velocity_world.norm())
            .fold(0.0_f64, f64::max);
        // The inner LM in this setup converges very near v=0 but not
        // exactly zero (small residual from the gravity term); pick the
        // gate threshold below the actual recovered magnitude so the
        // post-solve check has something to reject.
        assert!(
            max_mag > 1.0e-12,
            "probe max |v| too small to test the gate: {max_mag}"
        );

        let mut gated = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_velocity_magnitude_mps: Some(max_mag / 2.0),
            ..MotionBasedViInitializerConfig::default()
        });
        for i in 0..3 {
            gated.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut gated_map = build_constant_velocity_map(3);
        let gated_map_before = gated_map.clone();
        let err = gated
            .try_initialize(&mut gated_map, &factors, &seed)
            .expect_err("velocity gate must reject the LM result");
        assert_eq!(
            gated_map, gated_map_before,
            "a rejected speculative motion-VI solve must not write poses back"
        );
        match err {
            MotionBasedViRejectionReason::VelocityOutOfRange {
                kf_id,
                magnitude_mps,
                limit_mps,
            } => {
                assert!([1u64, 2, 3].contains(&kf_id), "kf_id = {kf_id}");
                assert!(magnitude_mps > limit_mps);
                assert!((limit_mps - max_mag / 2.0).abs() < 1.0e-12);
            }
            other => panic!("expected VelocityOutOfRange, got {other:?}"),
        }
        // The stage stays in `Waiting` after a gate rejection so the
        // caller can re-fire on a future trigger.
        match gated.status() {
            MotionBasedViInitializationStatus::Waiting { last_rejection, .. } => {
                assert!(matches!(
                    last_rejection,
                    Some(MotionBasedViRejectionReason::VelocityOutOfRange { .. })
                ));
            }
            other => panic!("expected Waiting after gate rejection, got {other:?}"),
        }
    }

    #[test]
    fn max_velocity_magnitude_gate_passes_when_under_limit() {
        // Same setup but with a generous limit; the inner result is
        // promoted to `completed` as usual.
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_velocity_magnitude_mps: Some(1.0e6),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let _ = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("generous limit must accept the LM result");
        match init.status() {
            MotionBasedViInitializationStatus::Initialised { .. } => {}
            other => panic!("expected Initialised, got {other:?}"),
        }
    }

    #[test]
    fn gyro_bias_gate_rejects_transactionally() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_gyro_bias_magnitude_rad_s: Some(0.05),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let err = init
            .try_initialize_with_bias_seed(
                &mut map,
                &factors,
                Vector3::new(0.1, 0.0, 0.0),
                Vector3::zeros(),
            )
            .expect_err("anchored gyro bias must exceed the configured gate");
        assert_eq!(map, map_before, "rejected solve must not mutate the map");
        assert!(matches!(
            err,
            MotionBasedViRejectionReason::GyroBiasOutOfRange {
                magnitude_rad_s,
                limit_rad_s,
                ..
            } if magnitude_rad_s > limit_rad_s && (limit_rad_s - 0.05).abs() < 1.0e-12
        ));
    }

    #[test]
    fn accel_bias_gate_rejects_transactionally() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            max_accel_bias_magnitude_mps2: Some(1.0),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = build_constant_velocity_map(3);
        let map_before = map.clone();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let err = init
            .try_initialize_with_bias_seed(
                &mut map,
                &factors,
                Vector3::zeros(),
                Vector3::new(2.0, 0.0, 0.0),
            )
            .expect_err("anchored accel bias must exceed the configured gate");
        assert_eq!(map, map_before, "rejected solve must not mutate the map");
        assert!(matches!(
            err,
            MotionBasedViRejectionReason::AccelBiasOutOfRange {
                magnitude_mps2,
                limit_mps2,
                ..
            } if magnitude_mps2 > limit_mps2 && (limit_mps2 - 1.0).abs() < 1.0e-12
        ));
    }

    /// Helper: SO3 type sanity (compile coverage for the test imports).
    #[test]
    fn so3_identity_is_quaternion_identity() {
        let r = SO3::identity();
        let q = r.quaternion();
        assert!((q.w - 1.0).abs() < 1e-12);
        assert!((q.i.abs() + q.j.abs() + q.k.abs()) < 1e-12);
    }

    // ============================================================
    // VIBA2 hand-off tests.
    // ============================================================

    #[test]
    fn viba2_handoff_runs_when_configured_and_reports_scale_history() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            gravity_world: Vector3::zeros(),
            viba2: Some(Viba2Config {
                initial_scale: 1.0,
                recover_scale: true,
                max_outer_iterations: 3,
                scale_tolerance: 1.0e-4,
                ..Viba2Config::default()
            }),
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let center = Vector3::new(i as f64, 0.0, 0.0);
            let pose = pose_at_world_center(center);
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA2 path should succeed");
        // VIBA2 ran at least once; scale history starts with the initial
        // scale and may have one more entry per outer iteration. The
        // exact length depends on whether the kinematic denominator was
        // identifiable, which on this zero-gravity-zero-Δp synthetic is
        // degenerate (so the outer loop bails after one inner solve and
        // freezes scale at 1.0).
        assert!(result.scale.is_finite());
        assert!(result.viba2_iterations_run >= 1);
        assert!(!result.scale_history.is_empty());
        assert!((result.scale_history[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn no_viba2_config_keeps_scale_history_empty() {
        let mut init = MotionBasedViInitializer::new(MotionBasedViInitializerConfig {
            min_keyframes: 3,
            min_translation_meters: 1.0,
            ..MotionBasedViInitializerConfig::default()
        });
        let gravity = init.config().gravity_world;
        for i in 0..3 {
            init.register_keyframe((i as u64) + 1, Point3::new(i as f64, 0.0, 0.0));
        }
        let mut map = VisualMap::new();
        let camera = pinhole_camera();
        map.cameras.insert(camera.id, camera.clone());
        for i in 0..3 {
            let pose = pose_at_world_center(Vector3::new(i as f64, 0.0, 0.0));
            let frame_id = (i as u64) + 1;
            let mut frame = Frame::new(frame_id, camera.id);
            frame.pose = Some(pose);
            let kf = Keyframe {
                frame,
                observations: Vec::new(),
            };
            map.keyframes.insert(frame_id, kf);
        }
        let seed = synthetic_seed();
        let factors = vec![
            no_acceleration_factor(1, 2, 1.0, gravity),
            no_acceleration_factor(2, 3, 1.0, gravity),
        ];
        let result = init
            .try_initialize(&mut map, &factors, &seed)
            .expect("VIBA1 path should succeed");
        assert!((result.scale - 1.0).abs() < 1e-12);
        assert!(result.scale_history.is_empty());
        assert_eq!(result.viba2_iterations_run, 0);
    }
}
