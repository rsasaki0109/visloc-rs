//! `LmProblem` implementation for `WindowProblem`: trial problems,
//! linearization and step acceptance.

use super::*;

impl WindowProblem {
    /// Build the complete candidate problem used by the LM trial objective.
    /// The compact chart alone is insufficient for Basalt's FEJ prior and
    /// quaternion state: both the full manifold objects and their accumulated
    /// linearized deltas must follow the trial. This helper is shared with
    /// the opt-in iteration detail trace so category costs are the same
    /// objective that drives the accept/reject decision.
    pub(super) fn trial_problem_with_step_owned(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
    ) -> Self {
        let increments = self.landmark_steps(state, step);
        let mut trial_problem = self.clone();
        let (trial_poses, trial_states) = self.apply_step_full(state, step);
        let scalar_mode = trial_problem.scalar_mode;
        let state_base_offset = trial_problem.poses.len() * POSE_DOF;
        for (index, (pose, trial_pose)) in
            trial_problem.poses.iter_mut().zip(trial_poses).enumerate()
        {
            pose.pose = trial_pose;
            pose.stored_current_pose = pose.pose.clone();
            let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
            accumulate_linearized_delta(&mut pose.linearized_delta, &increment, scalar_mode);
        }
        for (index, (state, trial_state)) in trial_problem
            .states
            .iter_mut()
            .zip(trial_states)
            .enumerate()
        {
            state.nav = trial_state;
            if !state.linearized {
                state.linearized_nav = state.nav.clone();
                state.linearized_delta.fill(0.0);
            } else {
                let increment = step
                    .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                    .into_owned();
                accumulate_linearized_delta(&mut state.linearized_delta, &increment, scalar_mode);
                state.stored_current_nav = state.nav.clone();
            }
        }
        trial_problem.apply_landmark_steps(&increments);
        trial_problem
    }
}

impl WindowProblem {
    pub(super) fn linearize_view<V: WindowValueView + Sync>(
        &self,
        values: &V,
    ) -> Result<LmLinearization, LmFailure> {
        let layout = self.layout();
        let mut factors = self.prior_factors_view(values);
        // Each landmark's visual factor depends only on that landmark's own
        // anchor/observation state views (`values`, shared and read-only)
        // and its own landmark index -- there is no shared mutable state
        // between landmarks (`visual_factor_view` builds and returns a fresh
        // local `WhitenedFactorRowStack`, it does not accumulate into any
        // caller-owned state). Computing every landmark's factor in
        // parallel and then pushing the `Some` results into `factors`
        // serially, in the same ascending `landmark_index` order the
        // sequential loop always used, reproduces the identical factor list
        // -- and therefore every downstream f32 rounding tree that depends
        // on factor order, e.g. the landmark-reduction accumulation --
        // bit-for-bit; only *when* each pure per-landmark factor is
        // computed changes, never its value or its position in `factors`.
        let visual_factors: Vec<Option<WhitenedFactorRowStack>> = (0..self.landmarks.len())
            .into_par_iter()
            .map(|landmark_index| self.visual_factor_view(values, landmark_index))
            .collect();
        for factor in visual_factors.into_iter().flatten() {
            factors.push(factor);
        }
        for link in self.imu_links.iter().cloned() {
            if let Some(factor) = self.imu_factor_view(values, link.clone()) {
                factors.push(factor);
            }
            if let Some(factor) = bias_walk_factor(
                values.chart(),
                link.clone(),
                self.state_dof(),
                layout.offset_for_state(link.from_index),
                layout.offset_for_state(link.to_index),
                self.bias_walk_noise,
                self.scalar_mode,
            ) {
                factors.push(factor);
            }
        }
        let prior_prefix = if self.prior.is_some() {
            1
        } else if self
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !self.states.is_empty())
        {
            1
        } else {
            0
        };
        let mut cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
            factors
                .iter()
                .skip(prior_prefix)
                .fold(0.0_f32, |sum, factor| sum + factor.objective_cost as f32) as f64
        } else {
            factors
                .iter()
                .skip(prior_prefix)
                .map(|factor| factor.objective_cost)
                .sum()
        };
        if prior_prefix != 0 {
            let prior_error = self.prior.as_ref().map_or_else(
                || factors.first().map_or(0.0, |factor| factor.objective_cost),
                |prior| self.marginal_prior_reduced_cost_view(values, prior),
            );
            cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
                (cost as f32 + prior_error as f32) as f64
            } else {
                cost + prior_error
            };
        }
        Ok(LmLinearization { factors, cost })
    }
}

impl LmProblem for WindowProblem {
    fn scalar_mode(&self) -> ScalarMode {
        self.scalar_mode
    }

    fn diagnostic_frame_id(&self) -> Option<u64> {
        self.states.last().map(|state| state.frame_id)
    }

    fn diagnostic_patch_reduced_f32(
        &self,
        iteration: usize,
        h: &mut DMatrix<f32>,
        b: &mut DVector<f32>,
    ) {
        let policy = diagnostic_env_snapshot();
        let Some(path) = policy.diagnostic_imu_hb.as_ref() else {
            return;
        };
        let Some(frame_id) = self.states.last().map(|state| state.frame_id) else {
            return;
        };
        let target_frame = policy.diagnostic_frame;
        let target_iteration = policy.diagnostic_iteration;
        if target_frame != Some(frame_id) || target_iteration != Some(iteration) {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };
        let Ok(payload) = serde_json::from_str::<JsonValue>(&text) else {
            return;
        };
        let Some(h_delta) = payload.get("h_delta_bits").and_then(JsonValue::as_array) else {
            return;
        };
        let Some(b_delta) = payload.get("b_delta_bits").and_then(JsonValue::as_array) else {
            return;
        };
        if h_delta.len() != h.nrows() * h.ncols() || b_delta.len() != b.len() {
            return;
        }
        for col in 0..h.ncols() {
            for row in 0..h.nrows() {
                let index = col * h.nrows() + row;
                let Some(bits) = h_delta[index].as_str() else {
                    return;
                };
                let Ok(bits) = u32::from_str_radix(bits, 16) else {
                    return;
                };
                h[(row, col)] += f32::from_bits(bits);
            }
        }
        for (index, value) in b_delta.iter().enumerate() {
            let Some(bits) = value.as_str() else {
                return;
            };
            let Ok(bits) = u32::from_str_radix(bits, 16) else {
                return;
            };
            b[index] += f32::from_bits(bits);
        }
    }

    fn diagnostic_lm_event(&mut self, event: LmDiagnosticEvent<'_>) {
        emit_iteration_event(self, event);
    }

    fn apply_step(&self, state: &DVector<f64>, step: &DVector<f64>) -> DVector<f64> {
        if state.len() != self.state_dof() || step.len() != self.state_dof() {
            return state + step;
        }
        let (updated_poses, updated) = self.apply_step_full(state, step);
        flatten_blocks_with_mode(
            &self
                .poses
                .iter()
                .enumerate()
                .map(|(index, _)| WindowPose {
                    frame_id: self.poses[index].frame_id,
                    timestamp_ns: self.poses[index].timestamp_ns,
                    pose: updated_poses[index].clone(),
                    stored_current_pose: updated_poses[index].clone(),
                    linearized_pose: self.poses[index].linearized_pose.clone(),
                    linearized_delta: self.poses[index].linearized_delta.clone(),
                    is_keyframe: self.poses[index].is_keyframe,
                })
                .collect::<Vec<_>>(),
            &updated
                .into_iter()
                .enumerate()
                .map(|(index, nav)| WindowState {
                    frame_id: self.states[index].frame_id,
                    timestamp_ns: self.states[index].timestamp_ns,
                    nav: nav.clone(),
                    stored_current_nav: if self.states[index].linearized {
                        nav.clone()
                    } else {
                        self.states[index].stored_current_nav.clone()
                    },
                    linearized_nav: if self.states[index].linearized {
                        self.states[index].linearized_nav.clone()
                    } else {
                        // `applyInc` on an upstream non-linearized state moves
                        // its linearization point itself, so a trial chart
                        // carries the candidate as its FEJ point.
                        nav
                    },
                    linearized_delta: self.states[index].linearized_delta.clone(),
                    is_keyframe: self.states[index].is_keyframe,
                    is_latest: self.states[index].is_latest,
                    linearized: self.states[index].linearized,
                })
                .collect::<Vec<_>>(),
            self.scalar_mode,
        )
    }

    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        if state.len() != self.state_dof() {
            return Err(LmFailure::LinearSolve);
        }
        let layout = self.layout();
        let mut factors = self.prior_factors(state);
        // As in `linearize_view`: each landmark's factor is a pure function
        // of the shared read-only state and its own index, so building them
        // in parallel and appending the results in ascending landmark order
        // yields the identical factor list (and every downstream rounding
        // tree) as the sequential loop.
        let visual_factors: Vec<Option<WhitenedFactorRowStack>> = (0..self.landmarks.len())
            .into_par_iter()
            .map(|landmark_index| {
                // The clean UpstreamF32 reducer may retain one compact Q1/R
                // payload per visual block.  Carry identity explicitly so
                // that future trial recovery is mapped by landmark rather
                // than inferred from factor order; callers without this
                // optional metadata remain on the legacy path.
                self.visual_factor(state, landmark_index).map(|factor| {
                    factor
                        .with_landmark_metadata(
                            landmark_index,
                            self.landmarks[landmark_index].track_id,
                        )
                        .with_visual_observation_ids(
                            self.landmarks[landmark_index]
                                .observations
                                .iter()
                                .map(|observation| (observation.state_index, observation.camera_id))
                                .collect(),
                        )
                })
            })
            .collect();
        factors.extend(visual_factors.into_iter().flatten());
        for link in self.imu_links.iter().cloned() {
            if let Some(factor) = self.imu_factor(state, link.clone()) {
                factors.push(factor);
            }
            if let Some(factor) = bias_walk_factor(
                state,
                link.clone(),
                self.state_dof(),
                layout.offset_for_state(link.from_index),
                layout.offset_for_state(link.to_index),
                self.bias_walk_noise,
                self.scalar_mode,
            ) {
                factors.push(factor);
            }
        }
        // `LinearizationAbsQR::linearizeProblem()` accumulates visual landmark
        // blocks first, then IMU blocks, and finally the marginal-prior error.
        // Rust's factor row stack keeps the prior row first because it must
        // occupy the AOM prefix, so do not fold its positive residual and
        // subtract it afterward: that changes the f32 rounding boundary.
        // Instead omit that row from the ordinary objective fold and append
        // Basalt's constant-free prior error at the source position.
        let prior_prefix = if self.prior.is_some() {
            1
        } else if self
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !self.states.is_empty())
        {
            1
        } else {
            0
        };
        let mut cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
            factors
                .iter()
                .skip(prior_prefix)
                .fold(0.0_f32, |sum, factor| sum + factor.objective_cost as f32) as f64
        } else {
            factors
                .iter()
                .skip(prior_prefix)
                .map(|factor| factor.objective_cost)
                .sum()
        };
        if prior_prefix != 0 {
            let prior_error = self.prior.as_ref().map_or_else(
                || factors.first().map_or(0.0, |factor| factor.objective_cost),
                |prior| self.marginal_prior_reduced_cost(state, prior),
            );
            cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
                (cost as f32 + prior_error as f32) as f64
            } else {
                cost + prior_error
            };
        }
        Ok(LmLinearization { factors, cost })
    }

    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        let linearization = self.linearize(state)?;
        // Upstream computeError evaluates the complete objective. Landmark
        // increments are recovered before trial_cost below, so there is no
        // need for the former projected-residual surrogate.
        let cost = linearization.cost;
        if cost.is_finite() {
            Ok(cost)
        } else {
            Err(LmFailure::NonFinite)
        }
    }

    fn trial_cost(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
    ) -> Result<f64, LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            self.trial_problem_with_step_owned(state, step)
                .owned_trial_objective(trial)
        } else {
            WindowTrialView::from_step(self, state, step, trial).cost()
        }
    }

    fn trial_cost_timed(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<f64, LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            // Diagnostic/probe runs retain the historical owned clone so all
            // audit boundaries continue to observe the complete candidate
            // problem and rejected steps cannot overwrite accepted sidecars.
            let trial_problem = timing.measure(TimingBucket::LmTrialConstructStep, || {
                self.trial_problem_with_step_owned(state, step)
            });
            timing.measure(TimingBucket::LmTrialCost, || {
                trial_problem.owned_trial_objective(trial)
            })
        } else {
            let trial_view = WindowTrialView::from_step_timed(self, state, step, trial, timing);
            timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())
        }
    }

    fn trial_cost_timed_with_token(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            // The clean solver is the only caller of this hook.  Keep a
            // defensive empty-token diagnostic branch so direct callers and
            // future retained paths continue to use the owned fallback.
            let trial_problem = timing.measure(TimingBucket::LmTrialConstructStep, || {
                self.trial_problem_with_step_owned(state, step)
            });
            timing
                .measure(TimingBucket::LmTrialCost, || {
                    trial_problem.owned_trial_objective(trial)
                })
                .map(|cost| (cost, LmTrialToken::default()))
        } else {
            let trial_view = WindowTrialView::from_step_timed(self, state, step, trial, timing);
            let cost = timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())?;
            Ok((
                cost,
                LmTrialToken::with_bound_landmark_steps(
                    trial_view.landmark_steps,
                    self.lm_trial_binding(state, step),
                ),
            ))
        }
    }

    fn trial_cost_timed_with_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        preparation: Option<LmTrialPreparation>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }

        // Diagnostic/probe runs intentionally retain the owned clone and the
        // historical token path.  The clean preparation is one-shot, so it
        // must be dropped before delegating rather than accidentally reused.
        if diagnostic_env_active() || self.scalar_mode != ScalarMode::UpstreamF32 {
            drop(preparation);
            return self.trial_cost_timed_with_token(state, step, trial, timing);
        }

        let Some(preparation) = preparation else {
            // Generic/legacy callers may not provide a compact payload.  Keep
            // the established eager recovery behavior in that case.
            return self.trial_cost_timed_with_token(state, step, trial, timing);
        };

        let trial_view = if timing.enabled() {
            let outer_started = timing.start();
            let recovery_started = timing.start();
            let prepared_steps = self.landmark_steps_from_preparation(state, step, preparation);
            timing.finish(TimingBucket::LmTrialLandmarkRecovery, recovery_started);
            let increments = match prepared_steps {
                Ok(increments) => increments,
                Err(error) => {
                    timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
                    return Err(error);
                }
            };
            let view = {
                let mut timing_sink = EnabledTrialConstructTiming { timing };
                WindowTrialView::from_step_with_landmark_steps_timing(
                    self,
                    state,
                    step,
                    trial,
                    increments,
                    &mut timing_sink,
                )
            };
            timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
            view
        } else {
            let increments = self.landmark_steps_from_preparation(state, step, preparation)?;
            WindowTrialView::from_step_with_landmark_steps(self, state, step, trial, increments)
        };

        let cost = timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())?;
        Ok((
            cost,
            LmTrialToken::with_bound_landmark_steps(
                trial_view.landmark_steps,
                self.lm_trial_binding(state, step),
            ),
        ))
    }

    fn accept_step(&mut self, state: &DVector<f64>, step: &DVector<f64>) -> Result<(), LmFailure> {
        let increments = self.landmark_steps(state, step);
        self.accept_step_with_landmark_steps(state, step, &increments)
    }

    fn accept_step_with_token(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        token: LmTrialToken,
    ) -> Result<(), LmFailure> {
        let expected_binding = self.lm_trial_binding(state, step);
        let Some(increments) = token
            .take_landmark_steps_for(expected_binding)
            .map_err(|_| LmFailure::LinearSolve)?
        else {
            return self.accept_step(state, step);
        };
        if increments.len() != self.landmarks.len() {
            return Err(LmFailure::LinearSolve);
        }
        self.accept_step_with_landmark_steps(state, step, &increments)
    }
}

impl WindowProblem {
    fn accept_step_with_landmark_steps(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        increments: &[Option<Vector3<f64>>],
    ) -> Result<(), LmFailure> {
        let (accepted_poses, accepted_states) = self.apply_step_full(state, step);
        let scalar_mode = self.scalar_mode;
        let state_base_offset = self.poses.len() * POSE_DOF;
        self.apply_landmark_steps(increments);
        for (index, (pose, accepted_pose)) in self.poses.iter_mut().zip(accepted_poses).enumerate()
        {
            pose.pose = accepted_pose;
            pose.stored_current_pose = pose.pose.clone();
            let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
            accumulate_linearized_delta(&mut pose.linearized_delta, &increment, scalar_mode);
        }
        for (index, (state, accepted_state)) in
            self.states.iter_mut().zip(accepted_states).enumerate()
        {
            state.nav = accepted_state;
            if !state.linearized {
                state.linearized_nav = state.nav.clone();
                state.linearized_delta.fill(0.0);
            } else {
                let increment = step
                    .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                    .into_owned();
                accumulate_linearized_delta(&mut state.linearized_delta, &increment, scalar_mode);
                state.stored_current_nav = state.nav.clone();
            }
        }
        Ok(())
    }
}
