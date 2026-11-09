//! `WindowProblem` trial objective, block layout / state accessors and solve
//! entry points.

use super::*;

impl WindowProblem {
    /// Shared candidate objective. Pose/landmark accessors refer to either the
    /// owned candidate or the prepared view; never to base linearization rows.
    pub(super) fn trial_objective_f32(
        &self,
        nav: impl Fn(usize, bool) -> Option<BasaltNavState> + Sync,
        parameter: impl Fn(usize) -> Option<InverseDistanceLandmark> + Sync,
        prior_cost: f64,
    ) -> Result<f64, LmFailure> {
        let ranks = self
            .trial_host_order
            .iter()
            .enumerate()
            .map(|(rank, &key)| (key, rank))
            .collect::<BTreeMap<_, _>>();
        if ranks.len() != self.trial_host_order.len() {
            return Err(LmFailure::LinearSolve);
        }
        let mut observations = BTreeMap::new();
        for (index, landmark) in self.landmarks.iter().enumerate() {
            if landmark.observations.is_empty() {
                continue;
            }
            let host = (
                block_timestamp(self, landmark.anchor_state_index).ok_or(LmFailure::LinearSolve)?
                    as u64,
                landmark.anchor_camera_id,
            );
            let rank = *ranks.get(&host).ok_or(LmFailure::LinearSolve)?;
            for observation in &landmark.observations {
                let target = (
                    block_timestamp(self, observation.state_index).ok_or(LmFailure::LinearSolve)?
                        as u64,
                    observation.camera_id,
                );
                if observations
                    .insert((rank, target, landmark.track_id), (index, observation))
                    .is_some()
                {
                    return Err(LmFailure::LinearSolve);
                }
            }
        }
        // Each observation's reprojection cost depends only on its own
        // landmark/host/target data (all read through `self`, `nav`, and
        // `parameter`, never through another observation's result), so
        // every observation's cost is computed in parallel below. The
        // items are collected from `observations` (a `BTreeMap`, so already
        // in the same deterministic `(rank, target, track_id)` key order
        // the serial loop iterated in) into a `Vec` first specifically so
        // that order survives into the results `Vec` -- `par_iter().map()`
        // preserves input order -- and the fold immediately afterward adds
        // each `Some` cost into `visual` serially, in that same order, via
        // the same `+=` the loop used. A failing item's error is still
        // surfaced at the same position in that order (via `cost?` before
        // the corresponding `visual +=` would have run), matching the
        // original loop's early-return exactly.
        let ordered_observations: Vec<_> = observations.into_iter().collect();
        let observation_costs: Vec<Result<Option<f32>, LmFailure>> = ordered_observations
            .par_iter()
            .map(|((rank, target, _), (index, observation))| {
                let landmark = &self.landmarks[*index];
                let host = self.trial_host_order[*rank];
                let from = nav(landmark.anchor_state_index, true).ok_or(LmFailure::LinearSolve)?;
                let to = nav(observation.state_index, true).ok_or(LmFailure::LinearSolve)?;
                let point = parameter(*index).ok_or(LmFailure::LinearSolve)?;
                let matrix = super::aom::upstream_trial_transform_f32(
                    &from.imu_to_world,
                    &to.imu_to_world,
                    &self.camera_to_imu(host.1).ok_or(LmFailure::LinearSolve)?,
                    &self.camera_to_imu(target.1).ok_or(LmFailure::LinearSolve)?,
                    host == *target,
                );
                let result = super::aom::upstream_trial_observation_f32(
                    self.camera_model(target.1).ok_or(LmFailure::LinearSolve)?,
                    matrix,
                    Vector3::new(
                        point.direction.xy.x as f32,
                        point.direction.xy.y as f32,
                        point.inverse_distance as f32,
                    ),
                    nalgebra::Vector2::new(observation.pixel.x as f32, observation.pixel.y as f32),
                    self.factor_config,
                );
                Ok(result.map(|(_, cost)| cost))
            })
            .collect();
        let mut visual = 0.0_f32;
        for cost in observation_costs {
            if let Some(cost) = cost? {
                visual += cost;
            }
        }
        let mut links = self.imu_links.iter().collect::<Vec<_>>();
        links.sort_by_key(|link| {
            self.states
                .get(link.from_index)
                .map(|state| state.timestamp_ns)
        });
        let (mut imu, mut bg, mut ba) = (0.0_f32, 0.0_f32, 0.0_f32);
        for link in links {
            if link.delta.delta_time == 0.0 {
                continue;
            }
            let from_state = self
                .states
                .get(link.from_index)
                .ok_or(LmFailure::LinearSolve)?;
            let to_state = self
                .states
                .get(link.to_index)
                .ok_or(LmFailure::LinearSolve)?;
            let from =
                nav(self.poses.len() + link.from_index, false).ok_or(LmFailure::LinearSolve)?;
            let to = nav(self.poses.len() + link.to_index, false).ok_or(LmFailure::LinearSolve)?;
            let raw = crate::imu::trial_preintegration_residual_f32(
                &from,
                &to,
                &link.delta,
                self.gravity_world,
            );
            let sqrt = crate::imu::sqrt_information_f32(&link.delta.covariance.map(|v| v as f32))
                .map_err(|_| LmFailure::NonFinite)?;
            imu += crate::imu::trial_imu_quadratic_stages_f32(&sqrt, &raw).2;
            let dt_ns = to_state
                .timestamp_ns
                .checked_sub(from_state.timestamp_ns)
                .ok_or(LmFailure::LinearSolve)?;
            let (gyro, accel) =
                crate::imu::trial_bias_cost_f32(&from, &to, dt_ns, self.bias_walk_noise)
                    .map_err(|_| LmFailure::LinearSolve)?;
            bg += gyro;
            ba += accel;
        }
        let cost = (visual + ((imu + bg) + ba)) + prior_cost as f32;
        if cost.is_finite() {
            Ok(cost as f64)
        } else {
            Err(LmFailure::NonFinite)
        }
    }

    pub(super) fn owned_trial_objective(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        if self.scalar_mode != ScalarMode::UpstreamF32 {
            return self.cost(state);
        }
        let prior = self.prior.as_ref().map_or_else(
            || {
                self.prior_factors(state)
                    .first()
                    .map_or(0.0, |factor| factor.objective_cost)
            },
            |prior| self.marginal_prior_reduced_cost(state, prior),
        );
        let prior = if self.visual_only.is_some() {
            prior + self.visual_only_objective(|index| self.block_nav(state, index))
        } else {
            prior
        };
        self.trial_objective_f32(
            |index, visual| {
                if visual {
                    self.block_nav_visual_current(state, index)
                } else {
                    self.block_nav(state, index)
                }
            },
            |index| {
                let landmark = self.landmarks.get(index)?;
                Some(landmark.parameter(self.block_frame_id(landmark.anchor_state_index)?))
            },
            prior,
        )
    }

    pub(super) fn layout(&self) -> WindowLayout {
        WindowLayout::new(self.poses.len(), self.states.len())
    }

    pub fn pose_offset(&self, index: usize) -> usize {
        self.layout()
            .offset_for_pose(index)
            .expect("pose index is outside the active AOM layout")
    }

    pub fn state_offset(&self, index: usize) -> usize {
        self.layout()
            .offset_for_state(index)
            .expect("state index is outside the active AOM layout")
    }

    pub fn state_dof(&self) -> usize {
        self.layout().state_dof()
    }

    pub fn initial_state(&self) -> DVector<f64> {
        flatten_blocks_with_mode(&self.poses, &self.states, self.scalar_mode)
    }

    pub(super) fn lm_trial_binding(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
    ) -> LmTrialBinding {
        LmTrialBinding::new(
            self as *const Self as usize,
            lm_token_window_generation(self),
            self.scalar_mode,
            lm_trial_vector_fingerprint(state),
            lm_trial_vector_fingerprint(step),
        )
    }

    /// Validate and expand the clean reducer's mapped landmark recovery into
    /// the window's full landmark order.  This consumes the opaque
    /// preparation before any trial-side mutation or view construction, so a
    /// stale, duplicated, or mismapped entry fails without changing state.
    pub(super) fn landmark_steps_from_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        preparation: LmTrialPreparation,
    ) -> Result<Vec<Option<Vector3<f64>>>, LmFailure> {
        let (tolerance_bits, state_fingerprint, step_fingerprint, mapped) =
            preparation.take_landmark_steps();
        if tolerance_bits != 1e-10_f64.to_bits()
            || state_fingerprint != lm_trial_vector_fingerprint(state)
            || step_fingerprint != lm_trial_vector_fingerprint(step)
            || mapped.len() > self.landmarks.len()
        {
            return Err(LmFailure::LinearSolve);
        }

        let mut steps = vec![None; self.landmarks.len()];
        let mut seen = vec![false; self.landmarks.len()];
        for (landmark_index, track_id, step) in mapped {
            let Some(landmark) = self.landmarks.get(landmark_index) else {
                return Err(LmFailure::LinearSolve);
            };
            if seen[landmark_index] || landmark.track_id != track_id {
                return Err(LmFailure::LinearSolve);
            }
            if step.is_some_and(|value| value.iter().any(|component| !component.is_finite())) {
                return Err(LmFailure::NonFinite);
            }
            seen[landmark_index] = true;
            steps[landmark_index] = step;
        }
        Ok(steps)
    }

    /// Linearize an already-solved window without running another LM pass.
    /// Marginalization uses this on the pre-shift AOM (which intentionally
    /// excludes the newest state) so its row ordering and FEJ point are
    /// derived from the same factor builder as the solver.
    pub fn linearize_snapshot(
        &self,
        state: &DVector<f64>,
    ) -> Result<Vec<WhitenedFactorRowStack>, LmFailure> {
        self.linearize(state)
            .map(|linearization| linearization.factors)
    }

    pub(super) fn block_kind(&self, index: usize) -> Option<WindowBlockKind> {
        if index < self.poses.len() {
            Some(WindowBlockKind::Pose)
        } else if index < self.poses.len() + self.states.len() {
            Some(WindowBlockKind::State)
        } else {
            None
        }
    }

    pub(crate) fn block_frame_id(&self, index: usize) -> Option<u64> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(self.poses.get(index)?.frame_id),
            WindowBlockKind::State | WindowBlockKind::StatePose => self
                .states
                .get(index - self.poses.len())
                .map(|state| state.frame_id),
        }
    }

    pub(super) fn block_pose(&self, state: &DVector<f64>, index: usize) -> Option<SE3> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => {
                let offset = self.pose_offset(index);
                let vector = state.rows(offset, POSE_DOF);
                let mut pose = decode_pose_with_mode(&vector, self.scalar_mode);
                // Upstream owns the pose as a full Sophus quaternion.  The
                // 15-column solver chart carries only the local SO(3) xyz
                // lanes, so reconstructing w from those lanes here loses the
                // last product/rounding boundary.  Keep the full quaternion
                // in the problem's pose table and use it whenever the chart
                // still names that same pose.
                if self.scalar_mode == ScalarMode::UpstreamF32
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
                let offset = self.state_offset(index - self.poses.len());
                let state_index = index - self.poses.len();
                let vector = state.rows(offset, NAV_STATE_DOF);
                let mut pose = decode_state_with_mode(&vector, self.scalar_mode).imu_to_world;
                if self.scalar_mode == ScalarMode::UpstreamF32
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

    pub(super) fn block_nav(&self, state: &DVector<f64>, index: usize) -> Option<BasaltNavState> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.block_pose(state, index)?,
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let offset = self.state_offset(index - self.poses.len());
                let state_index = index - self.poses.len();
                let vector = state.rows(offset, NAV_STATE_DOF);
                let mut nav = decode_state_with_mode(&vector, self.scalar_mode);
                if self.scalar_mode == ScalarMode::UpstreamF32
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

    /// Return the value-side pose used by Basalt's
    /// `PoseStateWithLin::getPose()` for a navigation state.  The temporary
    /// `PoseStateWithLin` made by `getPoseStateWithLin()` copies
    /// `state_linearized.T_w_i` into `pose_linearized`; for a non-linearized
    /// state `getPose()` returns that raw pose, while a linearized state uses
    /// the separately reconstructed current pose.  In particular, the zero
    /// tangent applied by the temporary constructor only initializes its
    /// unused `T_w_i_current` for the non-linearized case; it must not replace
    /// the raw `pose_linearized` quaternion returned by `getPose()`.
    pub(super) fn block_nav_visual_current(
        &self,
        state: &DVector<f64>,
        index: usize,
    ) -> Option<BasaltNavState> {
        if self.scalar_mode == ScalarMode::UpstreamF32
            && self.block_kind(index) == Some(WindowBlockKind::State)
        {
            let state_index = index.checked_sub(self.poses.len())?;
            let state_sidecar = self.states.get(state_index)?;
            if !state_sidecar.linearized {
                // Native `PoseVelBiasStateWithLin::getState()` returns
                // `state_linearized` while this flag is false, and the
                // temporary PoseStateWithLin then exposes its raw
                // `pose_linearized.T_w_i` through getPose().
                return Some(state_sidecar.linearized_nav.clone());
            }

            // `getPoseStateWithLin()` materializes a temporary
            // `PoseStateWithLin` from the full navigation state.  Its
            // constructor starts at `state_linearized.T_w_i`, then always
            // calls `PoseState::incPose(delta.head<6>(), T_w_i_current)`;
            // this call is not skipped when delta is zero.  Reconstruct the
            // value-side pose from that same frozen pose and f32 delta rather
            // than returning the stored `state_current`/compact chart.  In
            // particular, the zero SO3 increment still crosses Sophus'
            // float quaternion product/normalization boundary.
            if state_sidecar.linearized_delta.len() >= POSE_DOF {
                let base_pose = &state_sidecar.linearized_nav.imu_to_world;
                let mut current = state_sidecar.linearized_nav.clone();
                let delta = &state_sidecar.linearized_delta;
                current.imu_to_world.translation = vec3_f64(
                    vec3_f32(base_pose.translation)
                        + Vector3::from_iterator(
                            delta.rows(0, 3).iter().copied().map(|value| value as f32),
                        ),
                );
                current.imu_to_world.rotation = quat_f64_from_f32(&sophus_so3_product(
                    so3_exp_f32(Vector3::from_iterator(
                        delta.rows(3, 3).iter().copied().map(|value| value as f32),
                    )),
                    quat_f32_from_f64(&base_pose.rotation),
                ));
                return Some(current);
            }
        }
        self.block_nav(state, index)
    }

    /// Materialize the value-side pose that native
    /// `PoseStateWithLin::getPose()` exposes for the retained FEJ records.
    /// This is used only by the opt-in MargData diagnostic sidecar; factor
    /// evaluation continues through `block_nav_visual_current` above.
    pub(crate) fn diagnostic_effective_fej_poses(
        &self,
    ) -> (BTreeMap<i64, SE3>, BTreeMap<i64, SE3>) {
        let mut poses = BTreeMap::new();
        for pose in &self.poses {
            let effective = if self.scalar_mode == ScalarMode::UpstreamF32 {
                upstream_f32_pose_from_delta(
                    &pose.linearized_pose,
                    pose.linearized_delta.as_slice(),
                )
                .unwrap_or_else(|| pose.pose.clone())
            } else {
                pose.pose.clone()
            };
            poses.insert(pose.timestamp_ns, effective);
        }

        let mut states = BTreeMap::new();
        for state in &self.states {
            let effective = visual_state_pose_with_mode(state, self.scalar_mode);
            states.insert(state.timestamp_ns, effective);
        }
        (poses, states)
    }

    /// Return the pose/state used by Basalt's `getPoseLin()` for visual
    /// relative-pose Jacobians.  The value-side `T_t_h` is still evaluated
    /// from [`Self::block_nav`] at the current state; only a block marked
    /// `linearized` stays on its frozen FEJ point.
    pub(super) fn visual_block_is_linearized(&self, index: usize) -> Option<bool> {
        if index < self.poses.len() {
            // Pose-only blocks have already been retained by marginalization.
            Some(true)
        } else {
            self.states
                .get(index.checked_sub(self.poses.len())?)
                .map(|s| s.linearized)
        }
    }

    pub(super) fn block_nav_linearized(
        &self,
        _state: &DVector<f64>,
        index: usize,
    ) -> Option<BasaltNavState> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.poses.get(index)?.linearized_pose.clone(),
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.poses.len())?;
                let state = self.states.get(state_index)?;
                // `getPoseLin()` always names the frozen
                // `state_linearized.T_w_i`, including for a non-linearized
                // state whose value-side temporary receives a zero-inc
                // normalization.  The two paths intentionally differ by
                // those final quaternion bits.
                Some(state.linearized_nav.clone())
            }
        }
    }

    fn upstream_f32_trial_pose(&self, step: &DVector<f64>, index: usize) -> Option<SE3> {
        let pose = self.poses.get(index)?;
        let offset = self.pose_offset(index);
        let end = offset.checked_add(POSE_DOF)?;
        let delta = upstream_f32_accumulated_delta(
            pose.linearized_delta.as_slice(),
            step.as_slice().get(offset..end)?,
        )?;
        upstream_f32_pose_from_delta(&pose.linearized_pose, delta.as_slice())
    }

    fn upstream_f32_apply_state_trial(
        &self,
        index: usize,
        nav: &mut BasaltNavState,
        step: &DVector<f64>,
        trace_frame_id: Option<u64>,
    ) -> bool {
        let state = match self.states.get(index) {
            Some(state) => state,
            None => return false,
        };
        let offset = self.state_offset(index);
        let end = match offset.checked_add(NAV_STATE_DOF) {
            Some(end) => end,
            None => return false,
        };
        let increment = match if state.linearized && state.linearized_delta.len() == NAV_STATE_DOF {
            upstream_f32_accumulated_delta(
                state.linearized_delta.as_slice(),
                match step.as_slice().get(offset..end) {
                    Some(increment) => increment,
                    None => return false,
                },
            )
        } else {
            Some(DVector::from_column_slice(
                match step.as_slice().get(offset..end) {
                    Some(increment) => increment,
                    None => return false,
                },
            ))
        } {
            Some(increment) => increment,
            None => return false,
        };
        if state.linearized && state.linearized_delta.len() == NAV_STATE_DOF {
            *nav = state.linearized_nav.clone();
        }
        upstream_f32_apply_nav_delta(nav, increment.as_slice(), trace_frame_id)
    }

    /// Apply one solver increment while retaining the full pose objects that
    /// own the upstream quaternion payload.  `state` remains a compact
    /// 15-column chart; these vectors are the sidecar used by factor
    /// evaluation and by the trial/accept paths.
    pub(super) fn apply_step_full(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
    ) -> (Vec<SE3>, Vec<BasaltNavState>) {
        let mut updated_poses = self
            .poses
            .iter()
            .enumerate()
            .map(|(index, _)| self.block_pose(state, index).unwrap_or_default())
            .collect::<Vec<_>>();
        for (index, pose) in updated_poses.iter_mut().enumerate() {
            let offset = self.pose_offset(index);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                // Pose-only keyframes are the Rust counterpart of Basalt's
                // `PoseStateWithLin`.  Keep this reconstruction shared with
                // the eager trial view so acceptance and trial costs cross
                // the same f32 quaternion boundary.
                if let Some(trial_pose) = self.upstream_f32_trial_pose(step, index) {
                    *pose = trial_pose;
                }
            } else {
                pose.translation += Vector3::from_iterator(step.rows(offset, 3).iter().copied());
                pose.rotation = UnitQuaternion::from_scaled_axis(Vector3::from_iterator(
                    step.rows(offset + 3, 3).iter().copied(),
                )) * pose.rotation;
            }
        }

        let mut updated = (0..self.states.len())
            .map(|index| {
                self.block_nav(state, self.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        for (index, nav) in updated.iter_mut().enumerate() {
            let offset = self.state_offset(index);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                let _ = self.upstream_f32_apply_state_trial(
                    index,
                    nav,
                    step,
                    Some(self.states[index].frame_id),
                );
            } else {
                nav.imu_to_world.translation +=
                    Vector3::from_iterator(step.rows(offset, 3).iter().copied());
                nav.imu_to_world.rotation = UnitQuaternion::from_scaled_axis(
                    Vector3::from_iterator(step.rows(offset + 3, 3).iter().copied()),
                ) * nav.imu_to_world.rotation;
                nav.velocity_world_m_s +=
                    Vector3::from_iterator(step.rows(offset + 6, 3).iter().copied());
                nav.gyro_bias_rad_s +=
                    Vector3::from_iterator(step.rows(offset + 9, 3).iter().copied());
                nav.accel_bias_m_s2 +=
                    Vector3::from_iterator(step.rows(offset + 12, 3).iter().copied());
            }
        }
        (updated_poses, updated)
    }

    pub(super) fn block_index_for(&self, frame_id: u64, kind: WindowBlockKind) -> Option<usize> {
        match kind {
            WindowBlockKind::Pose => self.poses.iter().position(|pose| pose.frame_id == frame_id),
            WindowBlockKind::State | WindowBlockKind::StatePose => self
                .states
                .iter()
                .position(|state| state.frame_id == frame_id)
                .map(|index| self.poses.len() + index),
        }
    }

    pub(super) fn block_offset(&self, index: usize, kind: WindowBlockKind) -> usize {
        match kind {
            WindowBlockKind::Pose => self.pose_offset(index),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                self.state_offset(index - self.poses.len())
            }
        }
    }

    pub(super) fn camera_model(&self, camera_id: u16) -> Option<&DoubleSphereCamera> {
        if let Some(camera) = self.cameras.get(camera_id as usize) {
            Some(camera)
        } else if camera_id == 0 && self.cameras.is_empty() {
            // Preserve the original single-camera construction contract used
            // by the low-level unit tests and downstream adapters.
            Some(&self.camera)
        } else {
            None
        }
    }

    pub(super) fn camera_to_imu(&self, camera_id: u16) -> Option<SE3> {
        if let Some(transform) = self.t_imu_cam.get(camera_id as usize) {
            Some(transform.clone())
        } else if camera_id == 0 && self.t_imu_cam.is_empty() {
            Some(SE3::identity())
        } else {
            None
        }
    }

    pub fn solve(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_with_options(initial, config, true, true, &mut timing)
    }

    pub(crate) fn solve_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        self.solve_with_options(initial, config, true, true, timing)
    }

    /// Solve without retaining a duplicate final factor snapshot.
    ///
    /// The LM pass already owns the accepted-state cost and all factors needed
    /// to make that decision.  The final factor snapshot is only required by
    /// the diagnostic/retained-MargData path (`last_rows`); explicit
    /// no-MargData callers do not consume it.  Avoiding that post-solve
    /// relinearization removes a full visual/IMU factor build while leaving
    /// the solve, writeback, prior update, and window lifecycle unchanged.
    pub fn solve_without_factors(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_with_options(initial, config, false, true, &mut timing)
    }

    pub(crate) fn solve_without_factors_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        self.solve_with_options(initial, config, false, true, timing)
    }

    /// Solve without retaining factors or building the diagnostic prepass.
    ///
    /// This is reserved for the explicit no-MargData/no-trace estimator path.
    /// Any Basalt probe or compatibility environment variable selects the
    /// existing diagnostic implementation so instrumentation keeps observing
    /// the same boundaries as the retained-output path.
    pub(crate) fn solve_without_diagnostics(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_without_diagnostics_with_timing(initial, config, &mut timing)
    }

    pub(crate) fn solve_without_diagnostics_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        if diagnostic_env_active() {
            return self.solve_without_factors_with_timing(initial, config, timing);
        }
        self.solve_with_options(initial, config, false, false, timing)
    }

    /// Solve the active window for its accepted state **and** the final factor
    /// snapshot, while skipping the diagnostic prepass and per-trial
    /// diagnostic payloads.
    ///
    /// This is the online-mapper/MargData fast path.  It has the same LM math
    /// as [`Self::solve_without_diagnostics_with_timing`] (the compact f32
    /// preparation path) but, like [`Self::solve_with_timing`], retains the
    /// post-solve factor snapshot that MargData consumes.  The retained
    /// diagnostic *row counters* are recovered from that same final
    /// linearization, so `emit_marg`'s metadata is populated without the
    /// duplicate pre-solve linearization `initial_window_diagnostics` performs.
    ///
    /// Any Basalt probe or compatibility environment variable falls back to
    /// the fully diagnostic implementation so instrumentation keeps observing
    /// the historical boundaries.
    pub(crate) fn solve_lean_with_factors_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        if diagnostic_env_active() {
            return self.solve_with_timing(initial, config, timing);
        }
        self.solve_with_options(initial, config, true, false, timing)
    }

    fn solve_with_options(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        retain_factors: bool,
        retain_diagnostics: bool,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut diagnostics = if retain_diagnostics {
            initial_window_diagnostics(self, &initial)
        } else {
            minimal_window_diagnostics(self, &initial)
        };
        if initial.len() != self.state_dof() {
            let message = format!(
                "window state has {} columns, got initial vector of length {}",
                self.state_dof(),
                initial.len()
            );
            diagnostics.status = "invalid_initial_state".into();
            diagnostics.failure = Some(message.clone());
            return Err(WindowSolveError {
                message,
                diagnostics,
            });
        }
        let landmark_before = retain_diagnostics.then(|| self.landmarks.clone());
        let first = match if retain_diagnostics {
            solve_lm_with_timing(self, initial, config, true, true, timing)
        } else {
            solve_lm_with_timing(self, initial, config, false, false, timing)
        } {
            Ok(result) => result,
            Err(error) => {
                let message = format!("window LM: {error:?}");
                diagnostics.status = "first_lm_failed".into();
                diagnostics.failure = Some(message.clone());
                if retain_diagnostics {
                    diagnostics.lm.push(LmRunDiagnostics {
                        pass: "first".into(),
                        iterations: 0,
                        lambda: config.lambda_initial,
                        initial_cost: f64::NAN,
                        final_cost: f64::NAN,
                        accepted: 0,
                        rejected: 0,
                        trace: Vec::new(),
                        failure: Some(format!("{error:?}")),
                    });
                }
                return Err(WindowSolveError {
                    message,
                    diagnostics,
                });
            }
        };
        if retain_diagnostics {
            diagnostics.lm.push(lm_run_diagnostics("first", &first));
            if let Some(landmark_before) = landmark_before {
                diagnostics.landmark_writeback = self
                    .landmarks
                    .iter()
                    .zip(&landmark_before)
                    .filter(|(after, before)| {
                        after.direction != before.direction
                            || after.inverse_distance != before.inverse_distance
                    })
                    .count();
            }
        }

        // Basalt performs one canonical LM solve per active window. Landmark
        // back-substitution updates the 3-D blocks after that solve; emit a
        // fresh factor snapshot for MargData, but do not run a second LM pass
        // over the same frame (which otherwise doubles stale-prior rejects).
        let state = first.state;
        let factors = if retain_factors {
            match self.linearize(&state) {
                Ok(linearization) => {
                    // The diagnostic prepass (`initial_window_diagnostics`) is
                    // skipped on the lean-with-factors path, but `emit_marg`
                    // still reads these four row counters from the packet
                    // metadata.  Derive them from the final linearization we
                    // already hold instead of paying for a second full
                    // pre-solve linearization.
                    if !retain_diagnostics {
                        fill_diagnostic_factor_rows(self, &mut diagnostics, &linearization.factors);
                    }
                    linearization.factors
                }
                Err(error) => {
                    let message = format!("window final linearization: {error:?}");
                    diagnostics.status = "final_linearization_failed".into();
                    diagnostics.failure = Some(message.clone());
                    return Err(WindowSolveError {
                        message,
                        diagnostics,
                    });
                }
            }
        } else {
            Vec::new()
        };
        // `solve_lm` maintains this value at the accepted state. Calling
        // `self.cost(&state)` here would rebuild every factor a second time
        // without changing the returned value.
        let cost = first.cost;
        diagnostics.status = "success".into();
        Ok(WindowSolveResult {
            state,
            factors,
            cost,
            iterations: first.iterations,
            diagnostics,
        })
    }
}
