//! Borrowed LM trial views over a `WindowProblem` and the upstream-f32 state
//! update helpers.

use super::*;

/// The dynamic part of an LM trial.  The static window topology (camera
/// models, observations, IMU links, and the square-root prior) stays borrowed
/// from the accepted [`WindowProblem`].  Only the manifold sidecars that can
/// change during a trial are materialized here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct TrialLandmarkValue {
    pub(super) direction: StereographicDirection,
    pub(super) inverse_distance: f64,
}

#[derive(Debug)]
pub(super) struct WindowTrialView<'problem, 'chart> {
    pub(super) base: &'problem WindowProblem,
    chart: &'chart DVector<f64>,
    pub(super) poses: Vec<WindowPose>,
    pub(super) states: Vec<WindowState>,
    pub(super) landmarks: Vec<TrialLandmarkValue>,
    /// Exact eliminated-landmark increments produced while constructing this
    /// eager view.  A clean accepted trial moves this vector into the
    /// one-shot LM token instead of recovering it a second time.
    pub(super) landmark_steps: Vec<Option<Vector3<f64>>>,
    /// FEJ-local point used by the carried prior.  Keeping this alongside the
    /// trial sidecars avoids rebuilding it from the accepted problem and makes
    /// the prior boundary explicit in the view contract.
    pub(super) prior_fej_delta: Option<DVector<f64>>,
}

/// Timing sink used by [`WindowTrialView::from_step`].  The clean solver's
/// normal path instantiates the no-op implementation, so the generic
/// constructor monomorphizes away the timing calls and keeps the existing
/// disabled path free of `Instant` reads and timing allocations.  The active
/// implementation is selected only by the sidecar-enabled timed hook.
pub(super) trait TrialConstructTiming {
    fn start(&self) -> Option<TimingStart>;

    fn finish(&mut self, bucket: TimingBucket, started: Option<TimingStart>);
}

#[derive(Debug, Default)]
struct NoTrialConstructTiming;

impl TrialConstructTiming for NoTrialConstructTiming {
    #[inline]
    fn start(&self) -> Option<TimingStart> {
        None
    }

    #[inline]
    fn finish(&mut self, _bucket: TimingBucket, _started: Option<TimingStart>) {}
}

pub(super) struct EnabledTrialConstructTiming<'timing> {
    pub(super) timing: &'timing mut TimingBreakdown,
}

impl TrialConstructTiming for EnabledTrialConstructTiming<'_> {
    #[inline]
    fn start(&self) -> Option<TimingStart> {
        self.timing.start()
    }

    #[inline]
    fn finish(&mut self, bucket: TimingBucket, started: Option<TimingStart>) {
        self.timing.finish(bucket, started);
    }
}

/// Add a compact solver increment at Basalt's float boundary without
/// materializing a full trial state.  The returned value is bounded by the
/// navigation/pose block width used by the eager trial view.
pub(super) fn upstream_f32_accumulated_delta(
    base: &[f64],
    increment: &[f64],
) -> Option<DVector<f64>> {
    if base.len() != increment.len() {
        return None;
    }
    let mut result = DVector::from_column_slice(base);
    for (lhs, rhs) in result.iter_mut().zip(increment) {
        *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
    }
    Some(result)
}

/// The single source of truth for an UpstreamF32 pose trial.  Both
/// `apply_step_full` (acceptance/diagnostics) and the eager trial view call
/// this exact reconstruction, including the f32 SO(3) product boundary.
pub(super) fn upstream_f32_pose_from_delta(base_pose: &SE3, delta: &[f64]) -> Option<SE3> {
    if delta.len() < POSE_DOF {
        return None;
    }
    let translation = vec3_f32(base_pose.translation)
        + Vector3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32);
    let rotation = sophus_so3_product(
        so3_exp_f32(Vector3::new(
            delta[3] as f32,
            delta[4] as f32,
            delta[5] as f32,
        )),
        quat_f32_from_f64(&base_pose.rotation),
    );
    Some(SE3::new(
        quat_f64_from_f32(&rotation),
        vec3_f64(translation),
    ))
}

/// Pose returned by native `getPoseStateWithLin().getPose()` for a retained
/// navigation state.  This boundary is also used by delayed triangulation:
/// once a state is FEJ-linearized, constructing the temporary pose applies
/// the accumulated tangent even when it is exactly zero, so the quaternion
/// crosses Sophus' packet normalization once more.  A non-linearized state
/// still exposes its raw `state_linearized.T_w_i` payload.
pub(crate) fn visual_state_pose_with_mode(state: &WindowState, scalar_mode: ScalarMode) -> SE3 {
    if scalar_mode != ScalarMode::UpstreamF32 {
        return state.nav.imu_to_world.clone();
    }
    if !state.linearized {
        return state.linearized_nav.imu_to_world.clone();
    }
    if state.linearized_delta.len() >= POSE_DOF {
        return upstream_f32_pose_from_delta(
            &state.linearized_nav.imu_to_world,
            state.linearized_delta.as_slice(),
        )
        .unwrap_or_else(|| state.nav.imu_to_world.clone());
    }
    state.nav.imu_to_world.clone()
}

/// Apply one full navigation increment using the pinned Basalt float order.
/// `trace_frame_id` is optional so callers can share the arithmetic without
/// duplicating state-update diagnostics; the acceptance path supplies it.
pub(super) fn upstream_f32_apply_nav_delta(
    nav: &mut BasaltNavState,
    delta: &[f64],
    trace_frame_id: Option<u64>,
) -> bool {
    if delta.len() < NAV_STATE_DOF {
        return false;
    }
    nav.imu_to_world.translation = vec3_f64(
        vec3_f32(nav.imu_to_world.translation)
            + Vector3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32),
    );
    let before_rotation = quat_f32_from_f64(&nav.imu_to_world.rotation);
    let omega = Vector3::new(delta[3] as f32, delta[4] as f32, delta[5] as f32);
    let delta_rotation = so3_exp_f32(omega);
    let updated_rotation = sophus_so3_product(delta_rotation, before_rotation);
    if let Some(frame_id) = trace_frame_id {
        trace_state_update_quaternion(
            frame_id,
            before_rotation,
            omega,
            delta_rotation,
            updated_rotation,
        );
    }
    nav.imu_to_world.rotation = quat_f64_from_f32(&updated_rotation);
    nav.velocity_world_m_s = vec3_f64(
        vec3_f32(nav.velocity_world_m_s)
            + Vector3::new(delta[6] as f32, delta[7] as f32, delta[8] as f32),
    );
    nav.gyro_bias_rad_s = vec3_f64(
        vec3_f32(nav.gyro_bias_rad_s)
            + Vector3::new(delta[9] as f32, delta[10] as f32, delta[11] as f32),
    );
    nav.accel_bias_m_s2 = vec3_f64(
        vec3_f32(nav.accel_bias_m_s2)
            + Vector3::new(delta[12] as f32, delta[13] as f32, delta[14] as f32),
    );
    true
}

/// Value access needed by the factor builders.  Implementations deliberately
/// expose topology through `base()` but own/borrow all mutable trial values;
/// this keeps factor order and the `LmProblem` API unchanged while allowing a
/// clean trial to avoid cloning the whole window.
pub(super) trait WindowValueView {
    fn base(&self) -> &WindowProblem;
    fn chart(&self) -> &DVector<f64>;
    fn block_nav(&self, index: usize) -> Option<BasaltNavState>;
    fn block_nav_visual_current(&self, index: usize) -> Option<BasaltNavState>;
    fn block_nav_linearized(&self, index: usize) -> Option<BasaltNavState>;
    fn landmark_parameter(&self, index: usize) -> Option<InverseDistanceLandmark>;
    fn state_is_linearized(&self, index: usize) -> Option<bool>;
    fn state_linearized_nav(&self, index: usize) -> Option<BasaltNavState>;
    fn linearized_delta(&self, kind: WindowBlockKind, index: usize) -> Option<DVector<f64>>;

    fn prior_current_point(&self, prior: &WindowPrior) -> DVector<f64>;
}

impl WindowTrialView<'_, '_> {
    pub(super) fn from_step<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
    ) -> WindowTrialView<'problem, 'chart> {
        let mut timing = NoTrialConstructTiming;
        Self::from_step_with_timing(base, state, step, chart, &mut timing)
    }

    /// Construct an eager trial view and, when timing is enabled, retain the
    /// four non-overlapping construction sub-buckets inside the existing
    /// `lm_trial_construct_step` span. The disabled branch deliberately calls
    /// the no-op constructor above so ordinary solver runs take no timestamps
    /// and allocate no timing state.
    pub(super) fn from_step_timed<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> WindowTrialView<'problem, 'chart> {
        if !timing.enabled() {
            return Self::from_step(base, state, step, chart);
        }

        let outer_started = timing.start();
        let view = {
            let mut timing_sink = EnabledTrialConstructTiming { timing };
            Self::from_step_with_timing(base, state, step, chart, &mut timing_sink)
        };
        timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
        view
    }

    fn from_step_with_timing<'problem, 'chart, T: TrialConstructTiming>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        timing: &mut T,
    ) -> WindowTrialView<'problem, 'chart> {
        let landmark_recovery_started = timing.start();
        let increments = base.landmark_steps(state, step);
        timing.finish(
            TimingBucket::LmTrialLandmarkRecovery,
            landmark_recovery_started,
        );

        Self::from_step_with_landmark_steps_timing(base, state, step, chart, increments, timing)
    }

    /// Construct a trial view from landmark increments recovered by the clean
    /// reducer.  Keeping this path separate from `from_step_with_timing`
    /// ensures a prepared trial never recomputes `base.landmark_steps`.
    pub(super) fn from_step_with_landmark_steps<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        increments: Vec<Option<Vector3<f64>>>,
    ) -> WindowTrialView<'problem, 'chart> {
        let mut timing = NoTrialConstructTiming;
        Self::from_step_with_landmark_steps_timing(
            base,
            state,
            step,
            chart,
            increments,
            &mut timing,
        )
    }

    pub(super) fn from_step_with_landmark_steps_timing<
        'problem,
        'chart,
        T: TrialConstructTiming,
    >(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        increments: Vec<Option<Vector3<f64>>>,
        timing: &mut T,
    ) -> WindowTrialView<'problem, 'chart> {
        let apply_step_started = timing.start();
        let (trial_poses, trial_states) = base.apply_step_full(state, step);
        timing.finish(TimingBucket::LmTrialApplyStepFull, apply_step_started);

        let scalar_mode = base.scalar_mode;
        let state_base_offset = base.poses.len() * POSE_DOF;

        let sidecar_started = timing.start();
        let poses = base
            .poses
            .iter()
            .enumerate()
            .zip(trial_poses)
            .map(|((index, pose), trial_pose)| {
                let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
                let mut linearized_delta = pose.linearized_delta.clone();
                accumulate_linearized_delta(&mut linearized_delta, &increment, scalar_mode);
                WindowPose {
                    frame_id: pose.frame_id,
                    timestamp_ns: pose.timestamp_ns,
                    pose: trial_pose.clone(),
                    stored_current_pose: trial_pose,
                    linearized_pose: pose.linearized_pose.clone(),
                    linearized_delta,
                    is_keyframe: pose.is_keyframe,
                }
            })
            .collect::<Vec<_>>();

        let states = base
            .states
            .iter()
            .enumerate()
            .zip(trial_states)
            .map(|((index, state_base), trial_state)| {
                let mut linearized_delta = state_base.linearized_delta.clone();
                let (linearized_nav, stored_current_nav) = if !state_base.linearized {
                    linearized_delta.fill(0.0);
                    (trial_state.clone(), state_base.stored_current_nav.clone())
                } else {
                    let increment = step
                        .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                        .into_owned();
                    accumulate_linearized_delta(&mut linearized_delta, &increment, scalar_mode);
                    (state_base.linearized_nav.clone(), trial_state.clone())
                };
                WindowState {
                    frame_id: state_base.frame_id,
                    timestamp_ns: state_base.timestamp_ns,
                    nav: trial_state,
                    stored_current_nav,
                    linearized_nav,
                    linearized_delta,
                    is_keyframe: state_base.is_keyframe,
                    is_latest: state_base.is_latest,
                    linearized: state_base.linearized,
                }
            })
            .collect::<Vec<_>>();
        timing.finish(TimingBucket::LmTrialSidecarOther, sidecar_started);

        let landmark_materialization_started = timing.start();

        // Keep observations and anchor metadata borrowed from `base`; only
        // the two numeric landmark coordinates are trial-owned.  A failed
        // increment has exactly the same semantics as
        // `WindowProblem::apply_landmark_steps`: retain the previous value.
        let mut landmarks = base
            .landmarks
            .iter()
            .map(|landmark| TrialLandmarkValue {
                direction: landmark.direction,
                inverse_distance: landmark.inverse_distance,
            })
            .collect::<Vec<_>>();
        for (index, increment) in increments.iter().enumerate() {
            if let Some(increment) = increment {
                if let Some(value) = base.trial_landmark_value(index, *increment) {
                    landmarks[index] = value;
                }
            }
        }
        timing.finish(
            TimingBucket::LmTrialLandmarkMaterialization,
            landmark_materialization_started,
        );

        let sidecar_started = timing.start();
        let mut view = WindowTrialView {
            base,
            chart,
            poses,
            states,
            landmarks,
            landmark_steps: increments,
            prior_fej_delta: None,
        };
        view.prior_fej_delta = base
            .prior
            .as_ref()
            .map(|prior| view.compute_prior_current_point(prior));
        timing.finish(TimingBucket::LmTrialSidecarOther, sidecar_started);
        view
    }

    fn compute_prior_current_point(&self, prior: &WindowPrior) -> DVector<f64> {
        let kinds = prior.kinds();
        let mut current_point = DVector::zeros(prior.fej_point.len());
        let mut prior_offset = 0;
        for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
            let kind = kinds
                .get(prior_index)
                .copied()
                .unwrap_or(WindowBlockKind::State);
            let dof = block_dof(kind);
            if let Some(current_index) = self.base.block_index_for(*frame_id, kind) {
                // `block_index_for` is absolute in the AOM layout (poses are
                // the prefix), whereas the trial sidecars store states in a
                // state-local vector.  Keep this checked conversion at the
                // concrete view boundary; a pose prefix must never be read
                // as a state delta.
                let delta = match kind {
                    WindowBlockKind::Pose => self
                        .poses
                        .get(current_index)
                        .map(|pose| pose.linearized_delta.clone()),
                    WindowBlockKind::State => current_index
                        .checked_sub(self.base.poses.len())
                        .and_then(|local_index| self.states.get(local_index))
                        .and_then(|state| {
                            (state.linearized_delta.len() >= dof)
                                .then(|| state.linearized_delta.rows(0, dof).into_owned())
                        }),
                    WindowBlockKind::StatePose => current_index
                        .checked_sub(self.base.poses.len())
                        .and_then(|local_index| self.states.get(local_index))
                        .and_then(|state| {
                            (state.linearized_delta.len() >= dof)
                                .then(|| state.linearized_delta.rows(0, dof).into_owned())
                        }),
                };
                if let Some(delta) = delta {
                    if delta.len() == dof {
                        current_point.rows_mut(prior_offset, dof).copy_from(&delta);
                    }
                }
            }
            prior_offset += dof;
        }
        current_point
    }

    fn block_pose(&self, index: usize) -> Option<SE3> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => {
                let offset = self.base.pose_offset(index);
                let vector = self.chart.rows(offset, POSE_DOF);
                let mut pose = decode_pose_with_mode(&vector, self.base.scalar_mode);
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.poses[index].pose.rotation,
                    )
                {
                    pose.rotation = self.poses[index].pose.rotation.clone();
                }
                Some(pose)
            }
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                let offset = self.base.state_offset(state_index);
                let vector = self.chart.rows(offset, NAV_STATE_DOF);
                let mut pose = decode_state_with_mode(&vector, self.base.scalar_mode).imu_to_world;
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    pose.rotation = self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(pose)
            }
        }
    }

    fn block_nav_value(&self, index: usize) -> Option<BasaltNavState> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.block_pose(index)?,
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                let offset = self.base.state_offset(state_index);
                let vector = self.chart.rows(offset, NAV_STATE_DOF);
                let mut nav = decode_state_with_mode(&vector, self.base.scalar_mode);
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    nav.imu_to_world.rotation =
                        self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(nav)
            }
        }
    }

    fn block_nav_visual_current_value(&self, index: usize) -> Option<BasaltNavState> {
        if self.base.scalar_mode == ScalarMode::UpstreamF32
            && self.base.block_kind(index) == Some(WindowBlockKind::State)
        {
            let state_index = index.checked_sub(self.base.poses.len())?;
            let state_sidecar = self.states.get(state_index)?;
            if !state_sidecar.linearized {
                return Some(state_sidecar.linearized_nav.clone());
            }
            if state_sidecar.linearized_delta.len() >= POSE_DOF {
                let mut current = state_sidecar.linearized_nav.clone();
                current.imu_to_world =
                    visual_state_pose_with_mode(state_sidecar, self.base.scalar_mode);
                return Some(current);
            }
        }
        self.block_nav_value(index)
    }
}

impl WindowValueView for WindowTrialView<'_, '_> {
    fn base(&self) -> &WindowProblem {
        self.base
    }

    fn chart(&self) -> &DVector<f64> {
        self.chart
    }

    fn block_nav(&self, index: usize) -> Option<BasaltNavState> {
        self.block_nav_value(index)
    }

    fn block_nav_visual_current(&self, index: usize) -> Option<BasaltNavState> {
        self.block_nav_visual_current_value(index)
    }

    fn block_nav_linearized(&self, index: usize) -> Option<BasaltNavState> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.poses.get(index)?.linearized_pose.clone(),
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                Some(self.states.get(state_index)?.linearized_nav.clone())
            }
        }
    }

    fn landmark_parameter(&self, index: usize) -> Option<InverseDistanceLandmark> {
        let landmark = self.base.landmarks.get(index)?;
        let anchor_frame_id = self.base.block_frame_id(landmark.anchor_state_index)?;
        let value = self.landmarks.get(index)?;
        Some(InverseDistanceLandmark {
            anchor_pose: anchor_frame_id,
            anchor_camera_id: landmark.anchor_camera_id,
            direction: value.direction,
            inverse_distance: value.inverse_distance,
        })
    }

    fn state_is_linearized(&self, index: usize) -> Option<bool> {
        self.states.get(index).map(|state| state.linearized)
    }

    fn state_linearized_nav(&self, index: usize) -> Option<BasaltNavState> {
        self.states
            .get(index)
            .map(|state| state.linearized_nav.clone())
    }

    fn linearized_delta(&self, kind: WindowBlockKind, index: usize) -> Option<DVector<f64>> {
        match kind {
            WindowBlockKind::Pose => self
                .poses
                .get(index)
                .map(|pose| pose.linearized_delta.clone()),
            WindowBlockKind::State => self
                .states
                .get(index)
                .map(|state| state.linearized_delta.clone()),
            WindowBlockKind::StatePose => self.states.get(index).and_then(|state| {
                (state.linearized_delta.len() >= POSE_DOF)
                    .then(|| state.linearized_delta.rows(0, POSE_DOF).into_owned())
            }),
        }
    }

    fn prior_current_point(&self, prior: &WindowPrior) -> DVector<f64> {
        if let (Some(base_prior), Some(delta)) = (&self.base.prior, &self.prior_fej_delta) {
            if std::ptr::eq(base_prior, prior) && delta.len() == prior.fej_point.len() {
                return delta.clone();
            }
        }
        self.compute_prior_current_point(prior)
    }
}

pub(super) fn quat_f32_from_f64(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    UnitQuaternion::new_unchecked(Quaternion::new(
        rotation.w as f32,
        rotation.i as f32,
        rotation.j as f32,
        rotation.k as f32,
    ))
}

/// Sophus `SO3<float>::exp`, including its float small-angle branch.  The
/// nalgebra constructor is mathematically equivalent but its normalization
/// and half-angle path are not the operation sequence used by the pinned
/// Basalt header.
pub(super) fn so3_exp_f32(omega: Vector3<f32>) -> UnitQuaternion<f32> {
    let theta_sq = omega.dot(&omega);
    let (imag_factor, real_factor) = if theta_sq < f32::EPSILON * f32::EPSILON {
        let theta_po4 = theta_sq * theta_sq;
        (
            0.5_f32 - (1.0_f32 / 48.0_f32) * theta_sq + (1.0_f32 / 3840.0_f32) * theta_po4,
            1.0_f32 - (1.0_f32 / 8.0_f32) * theta_sq + (1.0_f32 / 384.0_f32) * theta_po4,
        )
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5_f32 * theta;
        (
            crate::update::portable_cosf::sinf(half_theta) / theta,
            crate::update::portable_cosf::cosf(half_theta),
        )
    };
    UnitQuaternion::new_unchecked(Quaternion::new(
        real_factor,
        imag_factor * omega.x,
        imag_factor * omega.y,
        imag_factor * omega.z,
    ))
}

pub(super) fn quat_f64_from_f32(rotation: &UnitQuaternion<f32>) -> UnitQuaternion<f64> {
    UnitQuaternion::new_unchecked(Quaternion::new(
        rotation.w as f64,
        rotation.i as f64,
        rotation.j as f64,
        rotation.k as f64,
    ))
}

#[inline]
fn upstream_f32_zero_inc_pose(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    quat_f64_from_f32(&sophus_so3_product(
        so3_exp_f32(Vector3::zeros()),
        quat_f32_from_f64(rotation),
    ))
}

fn trace_state_update_quaternion(
    frame_id: u64,
    before: UnitQuaternion<f32>,
    omega: Vector3<f32>,
    delta: UnitQuaternion<f32>,
    updated: UnitQuaternion<f32>,
) {
    if diagnostic_env_snapshot().state_update_trace_frame != Some(frame_id) {
        return;
    }
    let qbits = |rotation: &UnitQuaternion<f32>| {
        format!(
            "{:08x},{:08x},{:08x},{:08x}",
            rotation.i.to_bits(),
            rotation.j.to_bits(),
            rotation.k.to_bits(),
            rotation.w.to_bits(),
        )
    };
    eprintln!(
        "STATE_UPDATE frame={} before={} omega={:08x},{:08x},{:08x} delta={} updated={}",
        frame_id,
        qbits(&before),
        omega.x.to_bits(),
        omega.y.to_bits(),
        omega.z.to_bits(),
        qbits(&delta),
        qbits(&updated),
    );
}

pub(super) fn vec3_f32(value: Vector3<f64>) -> Vector3<f32> {
    value.map(|component| component as f32)
}

pub(super) fn vec3_f64(value: Vector3<f32>) -> Vector3<f64> {
    value.map(|component| component as f64)
}

impl WindowTrialView<'_, '_> {
    fn linearize(&self) -> Result<LmLinearization, LmFailure> {
        self.base.linearize_view(self)
    }

    pub(super) fn cost(&self) -> Result<f64, LmFailure> {
        if self.chart.len() != self.base.state_dof() {
            return Err(LmFailure::LinearSolve);
        }
        if self.base.scalar_mode == ScalarMode::UpstreamF32 {
            let prior = self.base.prior.as_ref().map_or_else(
                || {
                    self.base
                        .prior_factors_view(self)
                        .first()
                        .map_or(0.0, |factor| factor.objective_cost)
                },
                |prior| self.base.marginal_prior_reduced_cost_view(self, prior),
            );
            let prior = if self.base.visual_only.is_some() {
                prior
                    + self
                        .base
                        .visual_only_objective(|index| self.block_nav(index))
            } else {
                prior
            };
            return self.base.trial_objective_f32(
                |index, visual| {
                    if visual {
                        self.block_nav_visual_current(index)
                    } else {
                        self.block_nav(index)
                    }
                },
                |index| self.landmark_parameter(index),
                prior,
            );
        }
        checked_lm_linearization_cost(&self.linearize()?)
    }
}

const fn checked_lm_linearization_cost(linearization: &LmLinearization) -> Result<f64, LmFailure> {
    if linearization.cost.is_finite() {
        Ok(linearization.cost)
    } else {
        Err(LmFailure::NonFinite)
    }
}
