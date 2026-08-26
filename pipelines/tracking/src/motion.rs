//! Motion models (constant-pose/velocity, IMU-predictive, adaptive) and
//! the visual-odometry frontend prior plumbing.

use super::*;

pub trait MotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose>;

    fn observe(&mut self, _result: &TrackingResult) {}

    fn reset(&mut self) {}

    /// Whether this model's prediction is informative enough to seed the
    /// PnP optimizer itself. Candidate selection and pose-quality gating may
    /// still use every returned prior when this is `false`.
    ///
    /// A constant-pose prediction is deliberately excluded: on a moving
    /// camera, injecting "no motion" into PnP creates a self-reinforcing
    /// frozen-pose solution. Predictive velocity and IMU models retain the
    /// historical warm-start behavior.
    fn allows_pnp_pose_prior_warm_start(&self) -> bool {
        true
    }

    /// Apply a rigid world-frame correction to any cached world-frame
    /// state the model carries (velocities, cached poses). `correction`
    /// maps OLD world-frame points/poses to NEW world-frame points/poses:
    /// `p_new = correction.transform_point(&p_old)`. Called by
    /// [`crate::Tracker::apply_pose_correction`] after an external
    /// pose-graph optimisation (e.g. `visloc-slam`'s online loop-closure
    /// refinement) rewrites keyframe poses, so the model's own state
    /// stays consistent with the corrected map instead of predicting the
    /// next prior from stale, pre-correction poses/velocities.
    ///
    /// Default no-op — correct for stateless models (e.g.
    /// [`ConstantPoseMotionModel`]) that read `last_successful_pose`
    /// fresh from the [`Tracker`] on every call and so
    /// need no cache of their own to correct.
    fn apply_pose_correction(&mut self, _correction: &SE3) {}

    /// Apply a rigid-plus-scale (`Sim(3)`) world-frame correction —
    /// [`apply_pose_correction`](Self::apply_pose_correction)'s
    /// counterpart for a `Sim(3)` pose-graph solve (see `visloc-slam`'s
    /// online loop-closure refinement `Sim3` solver, which corrects
    /// monocular-style scale drift a rigid `SE(3)` graph cannot
    /// represent). `correction` follows the same OLD-world -> NEW-world
    /// convention as `apply_pose_correction`, generalised with a scale
    /// factor: `p_new = correction.transform_point(&p_old)`.
    ///
    /// The default projects `correction` down to its rotation+translation
    /// `SE(3)` part (dropping scale) and forwards to
    /// [`apply_pose_correction`](Self::apply_pose_correction) — correct
    /// for any cached poses, since
    /// [`crate::Tracker::apply_similarity_pose_correction`] applies the
    /// caller-visible `last_successful_pose` update itself. Models that
    /// also cache a world-frame *velocity* (a vector, not a point —
    /// translation does not apply, but the scale does: a Sim(3)
    /// correction scales a vector by `s * R`) must override this to also
    /// multiply that cached velocity by `correction.scale`.
    fn apply_similarity_correction(&mut self, correction: &Sim3) {
        let se3_part = SE3::new(correction.rotation, correction.translation);
        self.apply_pose_correction(&se3_part);
    }
}

pub trait VisualOdometryFrontend {
    type Error;

    fn estimate_relative_pose(
        &self,
        previous_frame: &Frame,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryEstimate {
    pub previous_frame_id: FrameId,
    pub current_frame_id: FrameId,
    pub previous_to_current: SE3,
    pub match_count: usize,
    pub inlier_count: usize,
    pub mean_reprojection_error: Option<f64>,
}

impl VisualOdometryEstimate {
    pub fn new(
        previous_frame_id: FrameId,
        current_frame_id: FrameId,
        previous_to_current: SE3,
    ) -> Self {
        Self {
            previous_frame_id,
            current_frame_id,
            previous_to_current,
            match_count: 0,
            inlier_count: 0,
            mean_reprojection_error: None,
        }
    }

    pub fn pose_prior_from_previous_pose(&self, previous_pose: &Pose) -> Pose {
        Pose {
            world_to_camera: self
                .previous_to_current
                .compose(&previous_pose.world_to_camera),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryPosePrior {
    pub estimate: VisualOdometryEstimate,
    pub pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisualOdometryPriorProvider<F> {
    frontend: F,
}

impl<F> VisualOdometryPriorProvider<F> {
    pub fn new(frontend: F) -> Self {
        Self { frontend }
    }

    pub fn frontend(&self) -> &F {
        &self.frontend
    }

    pub fn frontend_mut(&mut self) -> &mut F {
        &mut self.frontend
    }

    pub fn into_inner(self) -> F {
        self.frontend
    }
}

impl<F> VisualOdometryPriorProvider<F>
where
    F: VisualOdometryFrontend,
{
    pub fn predict_pose_prior(
        &self,
        previous_frame: &Frame,
        previous_pose: &Pose,
        current_frame: &Frame,
    ) -> Result<Option<VisualOdometryPosePrior>, F::Error> {
        let Some(estimate) = self
            .frontend
            .estimate_relative_pose(previous_frame, current_frame)?
        else {
            return Ok(None);
        };
        let pose = estimate.pose_prior_from_previous_pose(previous_pose);
        Ok(Some(VisualOdometryPosePrior { estimate, pose }))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoopVisualOdometryFrontend;

impl VisualOdometryFrontend for NoopVisualOdometryFrontend {
    type Error = std::convert::Infallible;

    fn estimate_relative_pose(
        &self,
        _previous_frame: &Frame,
        _current_frame: &Frame,
    ) -> Result<Option<VisualOdometryEstimate>, Self::Error> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstantPoseMotionModel;

impl MotionModel for ConstantPoseMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        last_successful_pose.cloned()
    }

    fn allows_pnp_pose_prior_warm_start(&self) -> bool {
        false
    }
}

/// Inertial pose predictor: a "loosely coupled" replacement for
/// [`ConstantPoseMotionModel`] / [`ConstantVelocityMotionModel`] that
/// integrates body-frame IMU samples forward from the previous
/// successful pose to produce the next pose prior. Mirrors the inputs of
/// [`crate::Tracker`]'s motion-model slot so it can be dropped in via
/// `Tracker::new_with_motion_model`.
///
/// Lifecycle:
///
/// 1. The caller pushes inter-frame IMU samples via
///    [`Self::push_imu_measurement`] (mutable, accumulates in
///    `pending_samples`).
/// 2. The tracker invokes [`MotionModel::predict_pose`] (read-only): the
///    model forward-Eulers `(R_bw, v_w, p_bw)` from `last_successful_pose`
///    using the pending samples + the configured gravity / biases + the
///    current `velocity_world` and returns the predicted
///    `world_to_camera` pose.
/// 3. The tracker invokes [`MotionModel::observe`]: on success the
///    pending samples are drained (the next inter-frame window starts
///    fresh). The model itself does NOT re-estimate `velocity_world`
///    from the tracker's output — that update is the caller's
///    responsibility (e.g., from a downstream local VI-BA's refined
///    velocity, or from a finite-difference of camera centres over the
///    integrated window). Without an explicit update, the predictor
///    re-uses the last set velocity, which is the same constant-velocity
///    behaviour [`ConstantVelocityMotionModel`] gives.
///
/// Coordinate conventions match the IMU pre-integrator: gyro and accel
/// are body-frame; gravity is in the world frame (KITTI y-down default:
/// `(0, 9.81, 0)`); accel includes gravity such that a stationary IMU
/// reads `R_wb^T · (−gravity_world)`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPredictiveMotionModel {
    /// Static configuration (gravity, biases). Update biases via
    /// [`Self::set_biases`] when a downstream VI-BA refines them.
    pub config: ImuPredictiveMotionModelConfig,
    /// Buffered `(gyro, accel, dt)` tuples to integrate at the next
    /// `predict_pose` call.
    pending_samples: Vec<ImuPredictivePendingSample>,
    /// World-frame velocity at the last successful pose. Used as the
    /// initial velocity of the strapdown integration. The model does
    /// NOT auto-update this on `observe`; downstream code (e.g., the
    /// local VI-BA in `OnlineSlamPipeline`) should call
    /// [`Self::set_velocity_world`] with the refined velocity after the
    /// next BA pass.
    velocity_world: nalgebra::Vector3<f64>,
    /// `true` when at least one pending sample was integrated by the
    /// most recent `predict_pose` call. Used by `observe` to decide
    /// whether to drain the buffer (a `predict_pose` call without any
    /// samples must NOT drain anything pushed *after* it).
    last_predict_consumed_samples: bool,
    /// Camera pose passed to the previous successful `observe`. Used by
    /// the carry-forward path to re-anchor the integration before
    /// committing the advanced `velocity_world`. Stays `None` until the
    /// first successful frame; ignored entirely when
    /// `config.carry_forward_velocity_world` is `false`.
    last_successful_pose: Option<Pose>,
}

/// Static parameters for [`ImuPredictiveMotionModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct ImuPredictiveMotionModelConfig {
    /// World-frame gravity vector. KITTI y-down default `(0, 9.81, 0)`;
    /// EuRoC z-up convention uses `(0, 0, -9.81)`.
    pub gravity_world: nalgebra::Vector3<f64>,
    /// Gyro bias subtracted from every sample before integration.
    pub bias_gyro: nalgebra::Vector3<f64>,
    /// Accel bias subtracted from every sample before integration.
    pub bias_acc: nalgebra::Vector3<f64>,
    /// `T_BS` body-from-sensor (= body-from-camera) rigid transform:
    /// `p_body = body_to_sensor · p_sensor`. Used by `predict_pose` to
    /// (a) convert the input camera pose to a body pose before
    /// strapdown integration, and (b) convert the integrated body pose
    /// back to a camera pose on the way out. Defaults to identity
    /// (body == camera, the assumption the original wire-up made).
    /// EuRoC's `cam0/sensor.yaml::T_BS` is exactly this transform: pass
    /// the parsed [`SE3`] verbatim. The camera-relative offset is
    /// ~0.1 m on EuRoC, so the identity default is a usable
    /// approximation but a metric-tight prediction wants the real
    /// extrinsic.
    pub body_to_sensor: SE3,
    /// When `true`, `observe` re-integrates the pending IMU
    /// samples from the previously-tracked pose to advance
    /// `velocity_world` for the next frame. Without this, the seed
    /// velocity stays frozen at the last `set_velocity_world`
    /// (i.e., last VI-BA mirror) until the next mirror fires — so on
    /// frames between mirrors, the strapdown integration restarts from
    /// the KF-time velocity rather than the velocity at the just-tracked
    /// frame. Defaults to `false` for backwards compatibility.
    pub carry_forward_velocity_world: bool,
}

impl Default for ImuPredictiveMotionModelConfig {
    fn default() -> Self {
        Self {
            gravity_world: nalgebra::Vector3::new(0.0, 9.81, 0.0),
            bias_gyro: nalgebra::Vector3::zeros(),
            bias_acc: nalgebra::Vector3::zeros(),
            body_to_sensor: SE3::identity(),
            carry_forward_velocity_world: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImuPredictivePendingSample {
    gyro: nalgebra::Vector3<f64>,
    accel: nalgebra::Vector3<f64>,
    dt: f64,
}

impl ImuPredictiveMotionModel {
    pub fn new(config: ImuPredictiveMotionModelConfig) -> Self {
        Self {
            config,
            pending_samples: Vec::new(),
            velocity_world: nalgebra::Vector3::zeros(),
            last_predict_consumed_samples: false,
            last_successful_pose: None,
        }
    }

    /// Append one body-frame `(gyro, accel)` sample with the elapsed
    /// time `dt` (seconds) since the previous sample. Non-positive
    /// `dt` is silently dropped to keep raw IMU replays robust.
    pub fn push_imu_measurement(
        &mut self,
        gyro: nalgebra::Vector3<f64>,
        accel: nalgebra::Vector3<f64>,
        dt: f64,
    ) {
        if dt <= 0.0 || dt.is_nan() {
            return;
        }
        self.pending_samples
            .push(ImuPredictivePendingSample { gyro, accel, dt });
    }

    /// Overwrite the world-frame velocity carried into the next
    /// `predict_pose`. Call this after a downstream solver (e.g., the
    /// `OnlineSlamPipeline` local VI-BA) refines the velocity at the
    /// most recent keyframe.
    pub fn set_velocity_world(&mut self, velocity_world: nalgebra::Vector3<f64>) {
        self.velocity_world = velocity_world;
    }

    /// Non-mutating finite-difference body-frame world-velocity from
    /// two successive camera poses and the elapsed time between them.
    /// Returns `None` when `dt_seconds` is not strictly positive/finite.
    /// Callers that want to write the result into `velocity_world`
    /// should use [`Self::update_velocity_from_camera_pose_difference`];
    /// callers that need to combine multiple finite-differences (e.g.
    /// the Phase-25 ThreePoseSmoother refresh policy averages two)
    /// use this directly to avoid intermediate state writes.
    pub fn body_velocity_from_camera_pose_difference(
        &self,
        prev: &Pose,
        curr: &Pose,
        dt_seconds: f64,
    ) -> Option<nalgebra::Vector3<f64>> {
        if dt_seconds <= 0.0 || !dt_seconds.is_finite() {
            return None;
        }
        let body_prev = self
            .config
            .body_to_sensor
            .compose(&prev.world_to_camera)
            .inverse()
            .translation;
        let body_curr = self
            .config
            .body_to_sensor
            .compose(&curr.world_to_camera)
            .inverse()
            .translation;
        Some((body_curr - body_prev) / dt_seconds)
    }

    /// Update `velocity_world` to a finite-difference estimate from two
    /// successive camera poses and the elapsed time between them.
    /// Internally converts each camera pose to a body pose via
    /// `body_to_sensor` so the velocity is the body's world-frame
    /// velocity (the integrator's expected initial-velocity semantics),
    /// not the camera's. Non-positive `dt_seconds` is silently dropped.
    /// This is the recommended hook for callers that do not run a
    /// downstream VI-BA (which would normally refine velocity) — without
    /// it, `velocity_world` stays at the constructor default (zero) and
    /// the position integration only picks up the quadratic accel term,
    /// systematically under-predicting motion on a moving body.
    pub fn update_velocity_from_camera_pose_difference(
        &mut self,
        prev: &Pose,
        curr: &Pose,
        dt_seconds: f64,
    ) {
        if let Some(v) = self.body_velocity_from_camera_pose_difference(prev, curr, dt_seconds) {
            self.velocity_world = v;
        }
    }

    /// Overwrite the gyro / accel bias linearisation points.
    pub fn set_biases(
        &mut self,
        bias_gyro: nalgebra::Vector3<f64>,
        bias_acc: nalgebra::Vector3<f64>,
    ) {
        self.config.bias_gyro = bias_gyro;
        self.config.bias_acc = bias_acc;
    }

    /// Read-only access to the current pending-sample count, mostly for
    /// tests / diagnostics.
    pub fn pending_samples_len(&self) -> usize {
        self.pending_samples.len()
    }

    /// Sum of `dt` over all currently-pending IMU samples. Useful for
    /// callers (e.g. [`AdaptiveImuPoseMotionModel`]) that need to know
    /// the elapsed wall-clock time between the previous successful
    /// `observe()` (which drained the buffer) and the moment this is
    /// queried — typically immediately before the next `observe()`
    /// drain. Returns `0.0` when no samples are pending.
    pub fn pending_samples_total_dt(&self) -> f64 {
        self.pending_samples.iter().map(|s| s.dt).sum()
    }

    /// Read-only access to the current world-frame velocity.
    pub fn velocity_world(&self) -> nalgebra::Vector3<f64> {
        self.velocity_world
    }
}

impl MotionModel for ImuPredictiveMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        let prev = last_successful_pose?;
        if self.pending_samples.is_empty() {
            return Some(prev.clone());
        }
        // Convert the input camera pose `T_cw = prev.world_to_camera` to
        // a body pose `T_bw = T_bs · T_cw` (body-from-world). Then take
        // its inverse to get `T_wb`, whose rotation is the body-to-world
        // orientation `R_wb` (transforms body-frame vectors to world)
        // and whose translation is the body centre in world `p_wb`.
        // With `body_to_sensor = identity` this reduces to the original
        // `r_bw = R_wc, p_bw = camera_center_world` extraction.
        let t_bw_initial = self.config.body_to_sensor.compose(&prev.world_to_camera);
        let t_wb_initial = t_bw_initial.inverse();
        let mut r_wb = t_wb_initial.rotation;
        let mut p_wb = t_wb_initial.translation;
        let mut v_w = self.velocity_world;
        for sample in &self.pending_samples {
            let gyro_unbiased = sample.gyro - self.config.bias_gyro;
            let accel_unbiased = sample.accel - self.config.bias_acc;
            let accel_world = r_wb.transform_vector(&accel_unbiased) + self.config.gravity_world;
            // Forward-Euler strapdown step. Position uses the velocity at
            // step start (mid-point integration would be more accurate but
            // matches the post-IMU-factor pre-integration semantics: tiny
            // `dt`s amortise the second-order error).
            p_wb += v_w * sample.dt + 0.5 * accel_world * sample.dt * sample.dt;
            v_w += accel_world * sample.dt;
            r_wb *= UnitQuaternion::from_scaled_axis(gyro_unbiased * sample.dt);
        }
        // Build the new body-in-world pose, invert to body-from-world,
        // and compose with `T_cb = body_to_sensor⁻¹` to recover the new
        // camera-from-world pose: `T_cw_new = T_cb · T_bw_new`.
        let t_wb_new = SE3::new(r_wb, p_wb);
        let t_bw_new = t_wb_new.inverse();
        let t_cw_new = self.config.body_to_sensor.inverse().compose(&t_bw_new);
        Some(Pose {
            world_to_camera: t_cw_new,
        })
    }

    fn observe(&mut self, result: &TrackingResult) {
        if !result.localization.success {
            return;
        }
        // Carry-forward path: when enabled and a previous successful
        // pose exists, re-run the same strapdown integration that
        // `predict_pose` performed (using the *previous* pose as anchor)
        // and commit the post-integration `v_w` as the new initial
        // velocity for the next `predict_pose` call. Without this, the
        // seed velocity stays frozen at the last `set_velocity_world`
        // (i.e. last VI-BA mirror) for every frame in the KF window,
        // so per-frame predictions silently restart from the KF-time
        // velocity instead of the velocity at the just-tracked frame.
        if self.config.carry_forward_velocity_world {
            if let Some(prev) = self.last_successful_pose.as_ref() {
                let t_bw_initial = self.config.body_to_sensor.compose(&prev.world_to_camera);
                let t_wb_initial = t_bw_initial.inverse();
                let mut r_wb = t_wb_initial.rotation;
                let mut v_w = self.velocity_world;
                for sample in &self.pending_samples {
                    let gyro_unbiased = sample.gyro - self.config.bias_gyro;
                    let accel_unbiased = sample.accel - self.config.bias_acc;
                    let accel_world =
                        r_wb.transform_vector(&accel_unbiased) + self.config.gravity_world;
                    v_w += accel_world * sample.dt;
                    r_wb *= UnitQuaternion::from_scaled_axis(gyro_unbiased * sample.dt);
                }
                self.velocity_world = v_w;
            }
            if let Some(pose) = result.localization.pose.as_ref() {
                self.last_successful_pose = Some(pose.clone());
            }
        }
        // Drain the pending window after a successful frame so the next
        // inter-frame integration starts fresh.
        self.pending_samples.clear();
        self.last_predict_consumed_samples = false;
    }

    fn reset(&mut self) {
        self.pending_samples.clear();
        self.velocity_world = nalgebra::Vector3::zeros();
        self.last_predict_consumed_samples = false;
        self.last_successful_pose = None;
    }

    fn apply_pose_correction(&mut self, correction: &SE3) {
        // `velocity_world` is a world-frame vector (a difference of
        // positions), so only the rotation component of a rigid world
        // correction applies — the translation term cancels.
        self.velocity_world = correction.rotation.transform_vector(&self.velocity_world);
        if let Some(pose) = self.last_successful_pose.as_mut() {
            pose.world_to_camera = pose.world_to_camera.compose(&correction.inverse());
        }
    }

    fn apply_similarity_correction(&mut self, correction: &Sim3) {
        let se3_part = SE3::new(correction.rotation, correction.translation);
        self.apply_pose_correction(&se3_part);
        // `velocity_world` is a vector, so the Sim(3) correction acts on
        // it as `s * R * v` (translation does not apply to a vector) —
        // the rotation was already applied by `apply_pose_correction`
        // above, so only the extra scale factor is left to fold in.
        self.velocity_world *= correction.scale;
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ConstantVelocityMotionModel {
    previous_successful_pose: Option<Pose>,
    latest_successful_pose: Option<Pose>,
}

impl ConstantVelocityMotionModel {
    pub fn new() -> Self {
        Self::default()
    }
}

impl MotionModel for ConstantVelocityMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        let (Some(previous), Some(latest)) = (
            self.previous_successful_pose.as_ref(),
            self.latest_successful_pose.as_ref(),
        ) else {
            return last_successful_pose.cloned();
        };

        let previous_center = previous.camera_center_world();
        let latest_center = latest.camera_center_world();
        let predicted_center = latest_center + (latest_center - previous_center);
        let rotation = latest.world_to_camera.rotation;
        let translation = -(rotation.transform_vector(&predicted_center.coords));
        Some(Pose::from_world_to_camera(rotation, translation))
    }

    fn observe(&mut self, result: &TrackingResult) {
        if !result.localization.success {
            return;
        }
        let Some(pose) = result.localization.pose.as_ref() else {
            return;
        };
        self.previous_successful_pose = self.latest_successful_pose.take();
        self.latest_successful_pose = Some(pose.clone());
    }

    fn reset(&mut self) {
        self.previous_successful_pose = None;
        self.latest_successful_pose = None;
    }

    fn apply_pose_correction(&mut self, correction: &SE3) {
        let inverse = correction.inverse();
        if let Some(pose) = self.previous_successful_pose.as_mut() {
            pose.world_to_camera = pose.world_to_camera.compose(&inverse);
        }
        if let Some(pose) = self.latest_successful_pose.as_mut() {
            pose.world_to_camera = pose.world_to_camera.compose(&inverse);
        }
    }
}

/// Configuration for the adaptive IMU↔ConstantPose motion model
/// ([`AdaptiveImuPoseMotionModel`]).
///
/// The Phase-23 EuRoC sweep
/// (`docs/motion_based_vi_alignment.md` §Phase-23 #2 follow-up)
/// established a clean accuracy↔survival trade-off:
///
/// - `--motion-model imu` produces tight pre-cliff trajectories
///   (V2_01 rigid ATE `0.0021 m`, similarity scale `1.000044`) but
///   the IMU's predictive aggressiveness triggers the
///   `--max-pose-jump-meters` gate at the universal cliff.
/// - `--motion-model pose` survives 25-313 % longer (MH_01 7 → 29
///   keyframes) but degrades rigid ATE 4-100× and collapses the
///   similarity scale.
///
/// The adaptive model defaults to IMU mode for accuracy and
/// transparently falls back to constant-pose after enough consecutive
/// tracking failures, then switches back to IMU once the tracker has
/// recovered. The intent is to keep IMU's tight predictions on the
/// healthy regime while the pose model carries the tracker through
/// the cliff transition.
/// How the [`AdaptiveImuPoseMotionModel`] refreshes the wrapped IMU
/// model's `velocity_world` at every Pose → IMU mode switch.
///
/// The motivation comes from the Phase-23 #4 oscillation: while the
/// adaptive wrapper sits in Pose mode, the IMU keeps absorbing raw
/// samples but never sees a successful visual `observe`, so its seed
/// `velocity_world` rapidly drifts away from the true body motion. The
/// first IMU prediction after the switch-back then mispredicts and
/// fires another failure — the wrapper oscillates Pose↔IMU.
///
/// The Phase-24 hook introduced [`Self::FiniteDifference`] to address
/// that. Phase-25 added [`Self::ZeroReset`] and
/// [`Self::ThreePoseSmoother`] as A/B alternatives after the
/// finite-difference variant produced only V1_01 wins (MH_01 / V2_01
/// were neutral-to-worse).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuVelocityRefreshPolicy {
    /// Phase-23 #4 behavior. Leave the IMU's `velocity_world`
    /// unchanged at every switch-back. The IMU continues integrating
    /// from whatever seed velocity it last computed; pose-mode-induced
    /// staleness is unmitigated.
    None,
    /// Phase-24 behavior (the post-refactor default). Recompute
    /// `velocity_world` from a finite-difference of the two most
    /// recent successful visual poses, divided by the IMU's
    /// pending-sample dt sum captured at the moment of the latest
    /// successful `observe`. Cheapest non-trivial reset; injects PnP
    /// noise from both poses into the velocity estimate.
    FiniteDifference,
    /// Phase-25 #1 (zero-reset) variant. Overwrite `velocity_world`
    /// with the zero vector at every switch-back. Cheapest possible
    /// reset; useful as a control when the cliff-region pose-mode
    /// poses are themselves PnP-noise-dominated (in which case any
    /// non-zero finite-difference seed is worse than zero).
    ZeroReset,
    /// Phase-25 #2 (smoothed finite-difference) variant. Computes two
    /// finite-difference velocities across the three most recent
    /// successful visual poses (oldest→previous and
    /// previous→latest) and averages them, then writes the result
    /// into `velocity_world`. Falls back to single
    /// finite-difference behavior when fewer than three poses are
    /// available. Aims to halve the PnP-noise variance compared with
    /// [`Self::FiniteDifference`].
    ThreePoseSmoother,
}

impl Default for ImuVelocityRefreshPolicy {
    /// Phase-25 default: three-pose smoothed finite-difference reset
    /// on switch. Empirically (Phase-25 EuRoC sweep, see
    /// `target/euroc_phase25_refresh_policy_ab/SUMMARY.md`) strictly
    /// improves on or matches [`Self::FiniteDifference`] on every
    /// 3-seq × 2-threshold case tested: identical at f=3/s=10 where
    /// the hook never fires or its result is washed out, and -25 %
    /// V2_01 / -1 % MH_01 / identical V1_01 rigid ATE at f=2/s=5
    /// compared with Phase-24's FiniteDifference default.
    fn default() -> Self {
        Self::ThreePoseSmoother
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveImuPoseMotionModelConfig {
    /// Number of consecutive failed-tracking frames (under IMU mode)
    /// that triggers a switch to constant-pose mode. Lower values
    /// react faster but oscillate more on noisy regimes.
    pub failures_to_switch_to_pose: usize,
    /// Number of consecutive successful-tracking frames (under
    /// constant-pose mode) that triggers a switch back to IMU mode.
    /// Higher values bias toward stability — the model stays in pose
    /// mode longer before re-trusting the IMU prediction.
    pub successes_to_switch_to_imu: usize,
    /// Policy for refreshing the wrapped IMU model's `velocity_world`
    /// at every Pose → IMU switch-back. See
    /// [`ImuVelocityRefreshPolicy`] for the semantics of each variant
    /// and the motivation behind the Phase-24 / Phase-25 thread.
    pub imu_velocity_refresh_policy: ImuVelocityRefreshPolicy,
}

impl Default for AdaptiveImuPoseMotionModelConfig {
    fn default() -> Self {
        Self {
            failures_to_switch_to_pose: 2,
            successes_to_switch_to_imu: 5,
            imu_velocity_refresh_policy: ImuVelocityRefreshPolicy::default(),
        }
    }
}

/// Diagnostic snapshot of which inner motion model the adaptive
/// wrapper is currently dispatching predictions through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMotionMode {
    Imu,
    Pose,
}

/// Adaptive motion model that wraps an [`ImuPredictiveMotionModel`]
/// and a [`ConstantPoseMotionModel`] and dispatches `predict_pose`
/// through whichever inner model the per-frame
/// failure / success counters select. Both inner models are kept
/// fed at all times by [`Self::observe`] so the switch is
/// instantaneous when it fires (the IMU's pending-sample buffer and
/// last-successful-pose anchor stay current even while
/// constant-pose is dispatching predictions).
///
/// The IMU sample stream is forwarded into the inner IMU model via
/// the public [`Self::imu_mut`] accessor — the [`MotionModel`] trait
/// itself does not expose a sample-feeding entry point, so callers
/// who construct an `AdaptiveImuPoseMotionModel` and want the IMU
/// integration to stay current must push samples directly into the
/// wrapped model.
#[derive(Debug, Clone)]
pub struct AdaptiveImuPoseMotionModel {
    imu: ImuPredictiveMotionModel,
    pose: ConstantPoseMotionModel,
    config: AdaptiveImuPoseMotionModelConfig,
    mode: AdaptiveMotionMode,
    consecutive_failures_under_imu: usize,
    consecutive_successes_under_pose: usize,
    switches_to_pose: u64,
    switches_to_imu: u64,
    /// Pose from the third-most-recent successful `observe()`. Used
    /// only by the
    /// [`ImuVelocityRefreshPolicy::ThreePoseSmoother`] policy to form
    /// a second finite-difference velocity that gets averaged with the
    /// most recent finite-difference. `None` until at least three
    /// successful observations have occurred since construction or the
    /// last [`Self::reset`].
    oldest_successful_pose: Option<Pose>,
    /// Pose from the second-most-recent successful `observe()`. Paired
    /// with [`Self::latest_successful_pose`] +
    /// [`Self::dt_between_latest_two_observations`] to recompute the
    /// IMU `velocity_world` from a visual finite-difference at every
    /// Pose → IMU switch event under
    /// [`ImuVelocityRefreshPolicy::FiniteDifference`] /
    /// [`ImuVelocityRefreshPolicy::ThreePoseSmoother`].
    previous_successful_pose: Option<Pose>,
    latest_successful_pose: Option<Pose>,
    /// Wall-clock seconds elapsed between
    /// [`Self::oldest_successful_pose`] and
    /// [`Self::previous_successful_pose`]. Captured as the value of
    /// [`Self::dt_between_latest_two_observations`] one shift ago.
    /// Used only by [`ImuVelocityRefreshPolicy::ThreePoseSmoother`].
    dt_between_previous_two_observations: f64,
    /// Wall-clock seconds elapsed between
    /// [`Self::previous_successful_pose`] and
    /// [`Self::latest_successful_pose`], captured as the IMU's
    /// `pending_samples_total_dt()` value at the moment of the
    /// `latest_successful_pose`'s `observe()` call (i.e. before that
    /// call drained the pending buffer).
    dt_between_latest_two_observations: f64,
    /// Cumulative number of `imu_velocity_refresh_policy` hooks that
    /// actually fired (i.e. switches at which the configured policy
    /// found enough state to write a new `velocity_world`). Smaller
    /// than or equal to [`Self::switches_to_imu`]; useful for
    /// telemetry on whether the refresh policy is engaging.
    velocity_refreshes_on_switch_to_imu: u64,
}

impl AdaptiveImuPoseMotionModel {
    pub fn new(
        imu: ImuPredictiveMotionModel,
        pose: ConstantPoseMotionModel,
        config: AdaptiveImuPoseMotionModelConfig,
    ) -> Self {
        Self {
            imu,
            pose,
            config,
            mode: AdaptiveMotionMode::Imu,
            consecutive_failures_under_imu: 0,
            consecutive_successes_under_pose: 0,
            switches_to_pose: 0,
            switches_to_imu: 0,
            oldest_successful_pose: None,
            previous_successful_pose: None,
            latest_successful_pose: None,
            dt_between_previous_two_observations: 0.0,
            dt_between_latest_two_observations: 0.0,
            velocity_refreshes_on_switch_to_imu: 0,
        }
    }

    /// Construct an adaptive model with default-config inner models.
    /// Convenience for callers that want a one-shot setup with the
    /// default IMU gravity / biases / extrinsics.
    pub fn with_defaults() -> Self {
        Self::new(
            ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default()),
            ConstantPoseMotionModel,
            AdaptiveImuPoseMotionModelConfig::default(),
        )
    }

    pub fn imu(&self) -> &ImuPredictiveMotionModel {
        &self.imu
    }

    /// Mutable access to the wrapped IMU model. Callers use this to
    /// forward raw IMU samples via
    /// [`ImuPredictiveMotionModel::push_imu_measurement`] and to
    /// mirror VI-BA-refined velocity / biases into the IMU state.
    pub fn imu_mut(&mut self) -> &mut ImuPredictiveMotionModel {
        &mut self.imu
    }

    pub fn config(&self) -> &AdaptiveImuPoseMotionModelConfig {
        &self.config
    }

    pub fn mode(&self) -> AdaptiveMotionMode {
        self.mode
    }

    /// Cumulative number of times the wrapper has switched from
    /// IMU → ConstantPose since construction (or last `reset`).
    pub fn switches_to_pose(&self) -> u64 {
        self.switches_to_pose
    }

    /// Cumulative number of times the wrapper has switched from
    /// ConstantPose → IMU since construction (or last `reset`).
    pub fn switches_to_imu(&self) -> u64 {
        self.switches_to_imu
    }

    /// Cumulative number of times the Phase-24
    /// refresh-IMU-velocity-on-switch-to-IMU hook has actually fired
    /// (i.e. the wrapper switched back to IMU AND
    /// `refresh_imu_velocity_on_switch_to_imu` was enabled AND both
    /// previous + latest visual poses were available AND
    /// `dt_between_latest_two_observations > 0`). Less than or equal
    /// to [`Self::switches_to_imu`].
    pub fn velocity_refreshes_on_switch_to_imu(&self) -> u64 {
        self.velocity_refreshes_on_switch_to_imu
    }
}

impl MotionModel for AdaptiveImuPoseMotionModel {
    fn predict_pose(
        &self,
        frame: &Frame,
        last_result: Option<&TrackingResult>,
        last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        match self.mode {
            AdaptiveMotionMode::Imu => {
                self.imu
                    .predict_pose(frame, last_result, last_successful_pose)
            }
            AdaptiveMotionMode::Pose => {
                self.pose
                    .predict_pose(frame, last_result, last_successful_pose)
            }
        }
    }

    fn allows_pnp_pose_prior_warm_start(&self) -> bool {
        self.mode == AdaptiveMotionMode::Imu
    }

    fn observe(&mut self, result: &TrackingResult) {
        // Capture the IMU's pending-sample dt sum BEFORE forwarding to
        // inner models — `imu.observe()` drains the pending buffer on
        // success, so this is the only window in which we can read the
        // wall-clock time elapsed since the previous successful
        // `observe()`. Used by the Phase-24
        // refresh-IMU-velocity-on-switch hook to compute a visual
        // finite-difference velocity at the moment of a Pose → IMU
        // switch.
        let pending_dt_before_observe = self.imu.pending_samples_total_dt();
        // Keep both inner models current regardless of which one is
        // currently dispatching predictions — when the switch fires
        // the previously-dormant model must have a coherent state.
        self.imu.observe(result);
        self.pose.observe(result);
        if result.localization.success {
            if let Some(pose) = result.localization.pose.as_ref() {
                self.oldest_successful_pose = self.previous_successful_pose.take();
                self.previous_successful_pose = self.latest_successful_pose.take();
                self.latest_successful_pose = Some(pose.clone());
                self.dt_between_previous_two_observations = self.dt_between_latest_two_observations;
                self.dt_between_latest_two_observations = pending_dt_before_observe;
            }
            match self.mode {
                AdaptiveMotionMode::Imu => {
                    self.consecutive_failures_under_imu = 0;
                }
                AdaptiveMotionMode::Pose => {
                    self.consecutive_successes_under_pose += 1;
                    if self.consecutive_successes_under_pose
                        >= self.config.successes_to_switch_to_imu
                    {
                        self.mode = AdaptiveMotionMode::Imu;
                        self.switches_to_imu += 1;
                        self.consecutive_failures_under_imu = 0;
                        self.consecutive_successes_under_pose = 0;
                        self.maybe_refresh_imu_velocity_on_switch_to_imu();
                    }
                }
            }
        } else {
            match self.mode {
                AdaptiveMotionMode::Imu => {
                    self.consecutive_failures_under_imu += 1;
                    if self.consecutive_failures_under_imu >= self.config.failures_to_switch_to_pose
                    {
                        self.mode = AdaptiveMotionMode::Pose;
                        self.switches_to_pose += 1;
                        self.consecutive_failures_under_imu = 0;
                        self.consecutive_successes_under_pose = 0;
                    }
                }
                AdaptiveMotionMode::Pose => {
                    self.consecutive_successes_under_pose = 0;
                }
            }
        }
    }

    fn reset(&mut self) {
        self.imu.reset();
        self.pose.reset();
        self.mode = AdaptiveMotionMode::Imu;
        self.consecutive_failures_under_imu = 0;
        self.consecutive_successes_under_pose = 0;
        self.switches_to_pose = 0;
        self.switches_to_imu = 0;
        self.oldest_successful_pose = None;
        self.previous_successful_pose = None;
        self.latest_successful_pose = None;
        self.dt_between_previous_two_observations = 0.0;
        self.dt_between_latest_two_observations = 0.0;
        self.velocity_refreshes_on_switch_to_imu = 0;
    }

    fn apply_pose_correction(&mut self, correction: &SE3) {
        self.imu.apply_pose_correction(correction);
        self.pose.apply_pose_correction(correction);
        let inverse = correction.inverse();
        for pose in [
            self.oldest_successful_pose.as_mut(),
            self.previous_successful_pose.as_mut(),
            self.latest_successful_pose.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            pose.world_to_camera = pose.world_to_camera.compose(&inverse);
        }
    }

    fn apply_similarity_correction(&mut self, correction: &Sim3) {
        self.imu.apply_similarity_correction(correction);
        self.pose.apply_similarity_correction(correction);
        // The finite-difference smoother poses are rigid `Pose`s; fold
        // in the rotation+translation part the same way
        // `apply_pose_correction` does above (their velocity role is
        // recomputed fresh from pose differences, so there is no cached
        // velocity vector here to additionally scale).
        let se3_part = SE3::new(correction.rotation, correction.translation);
        let inverse = se3_part.inverse();
        for pose in [
            self.oldest_successful_pose.as_mut(),
            self.previous_successful_pose.as_mut(),
            self.latest_successful_pose.as_mut(),
        ]
        .into_iter()
        .flatten()
        {
            pose.world_to_camera = pose.world_to_camera.compose(&inverse);
        }
    }
}

impl AdaptiveImuPoseMotionModel {
    /// Phase-24 / Phase-25 IMU-velocity-refresh hook. Called at the
    /// instant of every Pose → IMU mode transition from
    /// [`MotionModel::observe`]. Dispatches on
    /// [`AdaptiveImuPoseMotionModelConfig::imu_velocity_refresh_policy`]
    /// and increments [`Self::velocity_refreshes_on_switch_to_imu`]
    /// every time it actually writes a new `velocity_world` (so the
    /// counter is `0` under [`ImuVelocityRefreshPolicy::None`] and
    /// whenever the configured policy lacks enough state to compute a
    /// value). Silent no-op when the policy cannot fire — the IMU
    /// then continues with its current `velocity_world`.
    fn maybe_refresh_imu_velocity_on_switch_to_imu(&mut self) {
        match self.config.imu_velocity_refresh_policy {
            ImuVelocityRefreshPolicy::None => {}
            ImuVelocityRefreshPolicy::ZeroReset => {
                self.imu.set_velocity_world(nalgebra::Vector3::zeros());
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
            ImuVelocityRefreshPolicy::FiniteDifference => {
                let (Some(prev), Some(curr)) = (
                    self.previous_successful_pose.as_ref(),
                    self.latest_successful_pose.as_ref(),
                ) else {
                    return;
                };
                let dt = self.dt_between_latest_two_observations;
                let Some(v) = self
                    .imu
                    .body_velocity_from_camera_pose_difference(prev, curr, dt)
                else {
                    return;
                };
                self.imu.set_velocity_world(v);
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
            ImuVelocityRefreshPolicy::ThreePoseSmoother => {
                let (Some(prev), Some(curr)) = (
                    self.previous_successful_pose.as_ref(),
                    self.latest_successful_pose.as_ref(),
                ) else {
                    return;
                };
                let dt_latest = self.dt_between_latest_two_observations;
                let Some(v_latest) = self
                    .imu
                    .body_velocity_from_camera_pose_difference(prev, curr, dt_latest)
                else {
                    return;
                };
                // If the oldest pose + a valid older dt are available,
                // compute a second finite-difference and average. When
                // they're not (fewer than 3 successes), fall back to
                // single-finite-difference semantics so the policy
                // degrades gracefully into FiniteDifference rather
                // than no-op'ing.
                let v_write = match self.oldest_successful_pose.as_ref() {
                    Some(oldest) => self
                        .imu
                        .body_velocity_from_camera_pose_difference(
                            oldest,
                            prev,
                            self.dt_between_previous_two_observations,
                        )
                        .map(|v_prev| (v_prev + v_latest) * 0.5)
                        .unwrap_or(v_latest),
                    None => v_latest,
                };
                self.imu.set_velocity_world(v_write);
                self.velocity_refreshes_on_switch_to_imu += 1;
            }
        }
    }
}

#[cfg(test)]
mod imu_predictive_motion_tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::Frame;

    fn make_dummy_frame() -> Frame {
        Frame::new(1, 1)
    }

    fn rotation_angle_deg(a: &Pose, b: &Pose) -> f64 {
        let q_a = a.world_to_camera.rotation;
        let q_b = b.world_to_camera.rotation;
        q_a.rotation_to(&q_b).angle().to_degrees()
    }

    #[test]
    fn imu_predictive_motion_returns_last_pose_when_no_samples_pushed() {
        let model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 5.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted pose");
        assert_eq!(
            predicted.world_to_camera.translation,
            prev.world_to_camera.translation
        );
        assert_eq!(
            predicted.world_to_camera.rotation,
            prev.world_to_camera.rotation
        );
    }

    #[test]
    fn imu_predictive_motion_returns_none_when_no_previous_pose() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
        assert!(model
            .predict_pose(&make_dummy_frame(), None, None)
            .is_none());
    }

    #[test]
    fn imu_predictive_motion_stationary_under_gravity_holds_pose() {
        // Body stationary at world origin, gravity_world = (0, 0, -9.81)
        // (z-up). Accelerometer reads R_wb^T · (−g) = (0, 0, 9.81) when the
        // body is identity-oriented. With zero velocity / zero gyro, the
        // predicted pose must match the input exactly (no drift).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.05);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let predicted_center = predicted.camera_center_world();
        assert!(
            predicted_center.coords.norm() < 1.0e-9,
            "predicted center should stay at origin under stationary IMU, got {:?}",
            predicted_center
        );
        assert!(rotation_angle_deg(&predicted, &prev) < 1.0e-6);
    }

    #[test]
    fn imu_predictive_motion_pure_yaw_rotation_propagates_rotation() {
        // Zero gravity scene, body rotates at π/2 rad/s around world-z for
        // 1.0 s. Accel reading is zero (free fall in zero gravity).
        // Predicted pose's rotation should be a +90° yaw of the input.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let yaw_rate = std::f64::consts::FRAC_PI_2;
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::new(0.0, 0.0, yaw_rate), Vector3::zeros(), 0.01);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        // The body rotated +π/2 about world-z → R_bw is now a +90° rotation
        // about world-z → R_wc = R_bw⁻¹ is a -90° rotation about world-z.
        let expected_r_bw = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw_rate);
        let expected_r_wc = expected_r_bw.inverse();
        let actual_r_wc = predicted.world_to_camera.rotation;
        let angle_err = actual_r_wc.rotation_to(&expected_r_wc).angle().to_degrees();
        assert!(
            angle_err < 0.1,
            "yaw rotation drift too large: {angle_err} deg"
        );
    }

    #[test]
    fn imu_predictive_motion_constant_velocity_translates_position() {
        // Zero gravity, zero gyro, zero accel, but `velocity_world = (1, 0, 0)`
        // and 1 s of pending samples. Predicted position must be the input
        // position + (1, 0, 0).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.01);
        }
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        assert!(
            (center.x - 1.0).abs() < 1.0e-9,
            "predicted x should be 1.0 (start 0 + 1 m/s * 1 s), got {}",
            center.x
        );
        assert!(center.y.abs() < 1.0e-9);
        assert!(center.z.abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_observe_drains_pending_window_on_success() {
        use visloc_core::types::LocalizationSuccess;
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        for _ in 0..3 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.05);
        }
        assert_eq!(model.pending_samples_len(), 3);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_confidences: vec![None; 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let success = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&success);
        assert_eq!(model.pending_samples_len(), 0);
    }

    #[test]
    fn imu_predictive_motion_reset_clears_velocity_and_samples() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.set_velocity_world(Vector3::new(3.0, 0.0, 0.0));
        model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.05);
        model.reset();
        assert_eq!(model.pending_samples_len(), 0);
        assert_eq!(model.velocity_world(), Vector3::zeros());
    }

    #[test]
    fn imu_predictive_motion_carry_forward_default_off_leaves_velocity_frozen() {
        use visloc_core::types::LocalizationSuccess;
        // Default config: carry_forward_velocity_world = false. After
        // observe, velocity_world must equal the value set before push
        // (Phase-7 / pre-Phase-22 behaviour).
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0), 0.1);
        }
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose,
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_confidences: vec![None; 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let result = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&result);
        let v = model.velocity_world();
        assert!((v - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn imu_predictive_motion_carry_forward_on_advances_velocity_per_frame() {
        use visloc_core::types::LocalizationSuccess;
        // Zero gravity, identity body_to_sensor, accel (2,0,0) m/s² for
        // 1.0 s with initial v=(1,0,0). The body integrates to
        // v=(1+2*1.0,0,0)=(3,0,0). With carry-forward on, observe must
        // commit this back into velocity_world.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            carry_forward_velocity_world: true,
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        // Seed last_successful_pose with the body-at-origin pose by
        // running an initial observe (no pending samples → integration
        // is a no-op; effect is to populate last_successful_pose).
        let pose_zero = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization_zero = LocalizationResult::success(LocalizationSuccess {
            pose: pose_zero.clone(),
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_confidences: vec![None; 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let make_result = |pose: Pose| TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization: LocalizationResult::success(LocalizationSuccess {
                pose,
                candidate_landmark_count: 4,
                match_count: 4,
                correspondence_count: 4,
                inliers: vec![0, 1, 2, 3],
                inlier_query_indices: vec![0, 1, 2, 3],
                inlier_landmark_ids: vec![1, 2, 3, 4],
                inlier_confidences: vec![None; 4],
                inlier_reprojection_errors: vec![0.0; 4],
                mean_reprojection_error: 0.0,
                median_reprojection_error: 0.0,
                max_reprojection_error: 0.0,
            }),
            covisibility_local_map_size: None,
        };
        // First observe: populate last_successful_pose. No samples yet.
        model.observe(&TrackingResult {
            localization: localization_zero,
            ..make_result(pose_zero.clone())
        });
        // Second window: push 1.0 s of accel (2,0,0) samples, then observe.
        for _ in 0..10 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::new(2.0, 0.0, 0.0), 0.1);
        }
        // The "tracked" pose passed to this observe is irrelevant for
        // the v_w commit (the integration anchors on the *previous*
        // pose); use any plausible value.
        let pose_next =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-2.0, 0.0, 0.0));
        model.observe(&make_result(pose_next));
        let v = model.velocity_world();
        assert!(
            (v - Vector3::new(3.0, 0.0, 0.0)).norm() < 1.0e-9,
            "velocity_world should have advanced from (1,0,0) by ∫(2,0,0) dt = (2,0,0) to (3,0,0); got {v:?}",
        );
    }

    #[test]
    fn imu_predictive_motion_carry_forward_reset_clears_last_successful_pose() {
        use visloc_core::types::LocalizationSuccess;
        // After reset, a single observe with carry-forward on but no
        // prior pose should NOT touch velocity_world (no anchor to
        // integrate from). Verifies the optional gate.
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            carry_forward_velocity_world: true,
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let localization = LocalizationResult::success(LocalizationSuccess {
            pose: pose.clone(),
            candidate_landmark_count: 4,
            match_count: 4,
            correspondence_count: 4,
            inliers: vec![0, 1, 2, 3],
            inlier_query_indices: vec![0, 1, 2, 3],
            inlier_landmark_ids: vec![1, 2, 3, 4],
            inlier_confidences: vec![None; 4],
            inlier_reprojection_errors: vec![0.0; 4],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        });
        let result = TrackingResult {
            frame_id: 1,
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            successive_failures: 0,
            pose_prior: None,
            used_pose_prior: false,
            used_external_localization_prior: false,
            external_localization_prior_radius: None,
            tracking_failure_reason: None,
            map_landmark_count: 0,
            map_stats: MapProviderStats::default(),
            localization,
            covisibility_local_map_size: None,
        };
        model.observe(&result);
        model.set_velocity_world(Vector3::new(5.0, 0.0, 0.0));
        model.push_imu_measurement(Vector3::zeros(), Vector3::new(99.0, 0.0, 0.0), 0.1);
        model.reset();
        // After reset, last_successful_pose is cleared; next observe
        // with pending samples must NOT advance velocity_world (it stays
        // at the post-reset zero).
        model.push_imu_measurement(Vector3::zeros(), Vector3::new(99.0, 0.0, 0.0), 0.1);
        model.observe(&result);
        assert_eq!(model.velocity_world(), Vector3::zeros());
    }

    #[test]
    fn imu_predictive_motion_t_bs_offset_preserves_extrinsic_under_translation() {
        // Body translates at +1 m/s along world-x for 0.5 s under
        // zero-gravity, zero gyro, zero accel. The camera sits at a
        // body-frame offset of (0.1, 0, 0): `body_to_sensor.translation`
        // is the body-frame coords of the sensor origin (i.e., the
        // camera origin expressed in body coords). The body moves from
        // 0 to 0.5 m, so the body should end at world-x 0.5 m and the
        // camera should end at world-x 0.6 m. The starting camera pose
        // is set up consistent with body-at-origin, so this is a clean
        // round-trip test of the T_BS plumbing.
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            body_to_sensor: body_to_sensor.clone(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        model.set_velocity_world(Vector3::new(1.0, 0.0, 0.0));
        for _ in 0..50 {
            model.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.01);
        }
        // Starting camera pose: body at world origin with identity
        // rotation → camera centre in world = body_to_sensor.translation.
        // The world_to_camera SE3 with `R_cw = I, t_cw = -(0.1, 0, 0)`
        // places `camera_center_world = (0.1, 0, 0)`.
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        assert!(
            (center.x - 0.6).abs() < 1.0e-9,
            "predicted camera x should be 0.6 (body 0.5 + extrinsic 0.1), got {}",
            center.x
        );
        assert!(center.y.abs() < 1.0e-9);
        assert!(center.z.abs() < 1.0e-9);
        // Camera orientation should be unchanged (pure translation).
        assert!(rotation_angle_deg(&predicted, &prev) < 1.0e-6);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_from_camera_pose_diff_recovers_body_velocity() {
        // Body moved from (0,0,0) to (0.5, 0, 0) over 0.5 s → world-frame
        // body velocity should be (1, 0, 0). With `body_to_sensor =
        // identity` body==camera, so the camera-pose difference directly
        // reflects body motion.
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        let prev = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let curr =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.5, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&prev, &curr, 0.5);
        let v = model.velocity_world();
        assert!((v.x - 1.0).abs() < 1.0e-9);
        assert!(v.y.abs() < 1.0e-9);
        assert!(v.z.abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_with_lever_arm_uses_body_centre() {
        // Body translates from (0,0,0) to (0.5, 0, 0). The camera sits
        // 0.1 m ahead of body in body-x. So the camera moves from
        // (0.1, 0, 0) to (0.6, 0, 0). The velocity-update method must
        // strip the lever-arm offset and report the body's velocity
        // (1, 0, 0) instead of the camera's (which would be the same
        // here, but with a rotation the two would differ).
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig {
            body_to_sensor,
            ..ImuPredictiveMotionModelConfig::default()
        });
        // Camera centre = (0.1, 0, 0) at t=0; (0.6, 0, 0) at t=0.5.
        // World_to_camera translation = -R_cw * camera_center_world.
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let curr =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.6, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&prev, &curr, 0.5);
        let v = model.velocity_world();
        assert!((v.x - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn imu_predictive_motion_update_velocity_rejects_nonpositive_dt() {
        let mut model = ImuPredictiveMotionModel::new(ImuPredictiveMotionModelConfig::default());
        model.set_velocity_world(Vector3::new(5.0, 0.0, 0.0));
        let p = Pose::identity();
        model.update_velocity_from_camera_pose_difference(&p, &p, 0.0);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&p, &p, -1.0);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
        model.update_velocity_from_camera_pose_difference(&p, &p, f64::NAN);
        assert_eq!(model.velocity_world(), Vector3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn imu_predictive_motion_t_bs_offset_propagates_rotation_with_lever_arm() {
        // Body rotates +π/2 about world-z over 1.0 s under zero gravity,
        // zero accel, zero initial velocity. Camera sits 0.1 m ahead of
        // body in body-frame x. As the body rotates +90° about z, the
        // camera's world position should trace an arc: start (0.1, 0, 0)
        // → end (0, 0.1, 0). With `body_to_sensor = identity` the camera
        // would just rotate in place at the origin — so this test is
        // specifically validating the T_BS lever-arm contribution to the
        // predicted camera centre.
        let body_to_sensor = SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0));
        let config = ImuPredictiveMotionModelConfig {
            gravity_world: Vector3::zeros(),
            body_to_sensor: body_to_sensor.clone(),
            ..ImuPredictiveMotionModelConfig::default()
        };
        let mut model = ImuPredictiveMotionModel::new(config);
        let yaw_rate = std::f64::consts::FRAC_PI_2;
        for _ in 0..100 {
            model.push_imu_measurement(Vector3::new(0.0, 0.0, yaw_rate), Vector3::zeros(), 0.01);
        }
        // Starting camera pose corresponds to body at world origin
        // identity-oriented → camera centre = (0.1, 0, 0).
        let prev =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0));
        let predicted = model
            .predict_pose(&make_dummy_frame(), None, Some(&prev))
            .expect("predicted");
        let center = predicted.camera_center_world();
        // After +90° yaw, body's +x-axis points to world +y. Camera is
        // at body-x=0.1, so its world position is (0, 0.1, 0).
        assert!(
            center.x.abs() < 1.0e-3,
            "predicted camera x should be ≈0 after +90° yaw with lever arm, got {}",
            center.x
        );
        assert!(
            (center.y - 0.1).abs() < 1.0e-3,
            "predicted camera y should be ≈0.1 after +90° yaw, got {}",
            center.y
        );
        assert!(center.z.abs() < 1.0e-9);
    }
}
