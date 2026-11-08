//! Landmark steps and the visual / visual-only factors of `WindowProblem`.

use super::*;

impl WindowProblem {
    pub(super) fn landmark_steps(
        &self,
        state: &DVector<f64>,
        state_step: &DVector<f64>,
    ) -> Vec<Option<Vector3<f64>>> {
        let mut steps = Vec::with_capacity(self.landmarks.len());
        for landmark_index in 0..self.landmarks.len() {
            let Some(factor) = self.visual_factor(state, landmark_index) else {
                steps.push(None);
                continue;
            };
            let step = if self.scalar_mode == ScalarMode::UpstreamF32 {
                back_substitute_landmark_upstream_f32_with_track(
                    &factor,
                    state_step,
                    1e-10,
                    Some(self.landmarks[landmark_index].track_id),
                )
            } else {
                let reduced =
                    reduce_landmark_factors(std::slice::from_ref(&factor), state.len(), 1e-10);
                reduced
                    .back_substitution
                    .first()
                    .and_then(|data| back_substitute_landmark(data, state_step, 1e-10))
            };
            let Some(step) = step else {
                steps.push(None);
                continue;
            };
            if step.len() == 3 && step.iter().all(|value| value.is_finite()) {
                steps.push(Some(Vector3::new(step[0], step[1], step[2])));
            } else {
                steps.push(None);
            }
        }
        steps
    }

    pub(super) fn apply_landmark_steps(&mut self, steps: &[Option<Vector3<f64>>]) {
        let anchor_frames = self
            .landmarks
            .iter()
            .map(|landmark| self.block_frame_id(landmark.anchor_state_index))
            .collect::<Vec<_>>();
        for (index, (landmark, increment)) in self.landmarks.iter_mut().zip(steps).enumerate() {
            let Some(increment) = increment else {
                continue;
            };
            let Some(anchor_frame_id) = anchor_frames.get(index).copied().flatten() else {
                continue;
            };
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                landmark.apply_increment_f32(*increment, anchor_frame_id);
            } else {
                landmark.apply_increment(*increment, anchor_frame_id);
            }
        }
    }

    pub(super) fn trial_landmark_value(
        &self,
        index: usize,
        increment: Vector3<f64>,
    ) -> Option<TrialLandmarkValue> {
        let landmark = self.landmarks.get(index)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let mut parameter = landmark.parameter(anchor_frame_id);
        let applied = if self.scalar_mode == ScalarMode::UpstreamF32 {
            parameter.apply_increment_f32(increment)
        } else {
            parameter.apply_increment(increment)
        };
        applied.then_some(TrialLandmarkValue {
            direction: parameter.direction,
            inverse_distance: parameter.inverse_distance,
        })
    }

    pub(super) fn visual_factor(
        &self,
        state: &DVector<f64>,
        landmark_index: usize,
    ) -> Option<WhitenedFactorRowStack> {
        let landmark = self.landmarks.get(landmark_index)?;
        let anchor_nav = self.block_nav_visual_current(state, landmark.anchor_state_index)?;
        let anchor_nav_fej = self.block_nav_linearized(state, landmark.anchor_state_index)?;
        let anchor_extrinsic = self.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let parameter = landmark.parameter(anchor_frame_id);
        // The native `LandmarkBlockAbsDynamic` allocates one two-row slot for
        // every stored observation.  When a pre-marginalization AOM has
        // truncated a suffix state, its `pose_lin_vec` entry is null, but the
        // corresponding zero row remains in the packed QR storage.  Keep
        // those holes in source order instead of filtering them out: dropping
        // them changes both the Householder schedule and the subsequent Q2
        // normal system.  In-range observations retain the historical
        // camera/projection filtering semantics.
        let block_count = self.poses.len() + self.states.len();
        let entries = landmark
            .observations
            .iter()
            .filter_map(|observation| {
                if observation.state_index >= block_count {
                    return Some((*observation, None));
                }
                let camera = self.camera_model(observation.camera_id)?;
                let target_extrinsic = self.camera_to_imu(observation.camera_id)?;
                let target_nav = self.block_nav_visual_current(state, observation.state_index)?;
                let target_nav_fej = self.block_nav_linearized(state, observation.state_index)?;
                let target_frame_id = self.block_frame_id(observation.state_index)?;
                // Basalt's absolute landmark block has no pose contribution
                // only for the exact host/target `TimeCamId` identity. A
                // same-timestamp stereo observation executes computeRelPose,
                // so both pose blocks are retained and scattered below.
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.visual_block_is_linearized(landmark.anchor_state_index)?
                            || self.visual_block_is_linearized(observation.state_index)?,
                        self.factor_config,
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.factor_config,
                    )
                };
                factor.map(|factor| (*observation, Some(factor)))
            })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return None;
        }
        let rows = entries
            .iter()
            .map(|(_, factor)| factor.as_ref().map_or(2, |factor| factor.residual.len()))
            .sum::<usize>();
        let mut state_jacobian = DMatrix::zeros(rows, self.state_dof());
        let mut landmark_jacobian = DMatrix::zeros(rows, 3);
        let mut residual = DVector::zeros(rows);
        let mut objective_cost = 0.0;
        let mut row = 0;
        for (observation, factor) in entries {
            let Some(factor) = factor else {
                // This is the native null `pose_lin_vec` row.  The matrices
                // and residual were zero-initialized, so only advance the
                // row cursor; no objective contribution is produced.
                row += 2;
                continue;
            };
            let anchor_offset = self.pose_offset_for_block(landmark.anchor_state_index);
            let target_offset = self.pose_offset_for_block(observation.state_index);
            // A same-timestamp stereo pair shares one navigation block.  The
            // two relative-pose derivatives are accumulated into that block
            // by the same sequential f32 stores as every other shared block.
            // The camera-to-camera transform is mathematically independent
            // of the common IMU pose, but the independently rounded native
            // products can leave a real residual in individual lanes.
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                // The pinned `LinearizationBase<float>` accumulates every
                // visual contribution directly into its float AOM blocks.
                // The compatibility row object is widened to f64 for the
                // public API, so using nalgebra's f64 `add_assign` here would
                // defer the rounding until after host/target contributions
                // have already been summed.  Round each scalar at the same
                // accumulation boundary before it reaches ABS_QR.
                for local_row in 0..factor.residual.len() {
                    for column in 0..6 {
                        let anchor_column = anchor_offset + column;
                        let target_column = target_offset + column;
                        let anchor = state_jacobian[(row + local_row, anchor_column)] as f32
                            + factor.anchor_pose_jacobian[(local_row, column)] as f32;
                        if anchor_column == target_column {
                            // Both TimeCamIds can name the same navigation
                            // block for same-timestamp stereo. Upstream's
                            // two `+=` stores are sequential float adds;
                            // retain that order instead of overwriting the
                            // anchor contribution with an algebraic zero.
                            let combined =
                                anchor + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = combined as f64;
                        } else {
                            let target = state_jacobian[(row + local_row, target_column)] as f32
                                + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = anchor as f64;
                            state_jacobian[(row + local_row, target_column)] = target as f64;
                        }
                    }
                }
            } else {
                state_jacobian
                    .view_mut((row, anchor_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.anchor_pose_jacobian);
                state_jacobian
                    .view_mut((row, target_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.target_pose_jacobian);
            }
            landmark_jacobian
                .view_mut((row, 0), (factor.residual.len(), 3))
                .copy_from(&factor.landmark_jacobian);
            residual
                .rows_mut(row, factor.residual.len())
                .copy_from(&factor.residual);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                objective_cost = (objective_cost as f32 + factor.objective_cost as f32) as f64;
            } else {
                objective_cost += factor.objective_cost;
            }
            row += factor.residual.len();
        }
        WhitenedFactorRowStack::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Visual,
        )
    }

    /// Trial-view counterpart of [`Self::visual_factor`].  The observation
    /// list and camera topology remain borrowed from the accepted problem;
    /// only the current/FEJ poses and landmark parameter come from `values`.
    /// Keeping this as a separate builder leaves the diagnostic/audit path on
    /// the original owned `WindowProblem` implementation.
    pub(super) fn visual_factor_view<V: WindowValueView>(
        &self,
        values: &V,
        landmark_index: usize,
    ) -> Option<WhitenedFactorRowStack> {
        let base = values.base();
        let landmark = base.landmarks.get(landmark_index)?;
        let anchor_nav = values.block_nav_visual_current(landmark.anchor_state_index)?;
        let anchor_nav_fej = values.block_nav_linearized(landmark.anchor_state_index)?;
        let anchor_extrinsic = base.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = base.block_frame_id(landmark.anchor_state_index)?;
        let parameter = values.landmark_parameter(landmark_index)?;
        let valid = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index < base.poses.len() + base.states.len())
            .filter_map(|observation| {
                let camera = base.camera_model(observation.camera_id)?;
                let target_extrinsic = base.camera_to_imu(observation.camera_id)?;
                let target_nav = values.block_nav_visual_current(observation.state_index)?;
                let target_nav_fej = values.block_nav_linearized(observation.state_index)?;
                let target_frame_id = base.block_frame_id(observation.state_index)?;
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if base.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        base.visual_block_is_linearized(landmark.anchor_state_index)?
                            || base.visual_block_is_linearized(observation.state_index)?,
                        base.factor_config,
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        base.factor_config,
                    )
                };
                factor.map(|factor| (*observation, factor))
            })
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return None;
        }
        let rows = valid
            .iter()
            .map(|(_, factor)| factor.residual.len())
            .sum::<usize>();
        let mut state_jacobian = DMatrix::zeros(rows, base.state_dof());
        let mut landmark_jacobian = DMatrix::zeros(rows, 3);
        let mut residual = DVector::zeros(rows);
        let mut objective_cost = 0.0;
        let mut row = 0;
        for (observation, factor) in valid {
            let anchor_offset = base.pose_offset_for_block(landmark.anchor_state_index);
            let target_offset = base.pose_offset_for_block(observation.state_index);
            if base.scalar_mode == ScalarMode::UpstreamF32 {
                for local_row in 0..factor.residual.len() {
                    for column in 0..6 {
                        let anchor_column = anchor_offset + column;
                        let target_column = target_offset + column;
                        let anchor = state_jacobian[(row + local_row, anchor_column)] as f32
                            + factor.anchor_pose_jacobian[(local_row, column)] as f32;
                        if anchor_column == target_column {
                            let combined =
                                anchor + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = combined as f64;
                        } else {
                            let target = state_jacobian[(row + local_row, target_column)] as f32
                                + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = anchor as f64;
                            state_jacobian[(row + local_row, target_column)] = target as f64;
                        }
                    }
                }
            } else {
                state_jacobian
                    .view_mut((row, anchor_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.anchor_pose_jacobian);
                state_jacobian
                    .view_mut((row, target_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.target_pose_jacobian);
            }
            landmark_jacobian
                .view_mut((row, 0), (factor.residual.len(), 3))
                .copy_from(&factor.landmark_jacobian);
            residual
                .rows_mut(row, factor.residual.len())
                .copy_from(&factor.residual);
            if base.scalar_mode == ScalarMode::UpstreamF32 {
                objective_cost = (objective_cost as f32 + factor.objective_cost as f32) as f64;
            } else {
                objective_cost += factor.objective_cost;
            }
            row += factor.residual.len();
        }
        WhitenedFactorRowStack::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Visual,
        )
    }

    /// Capture one grouped landmark factor plus its observation-to-row mapping
    /// for the opt-in numeric audit.  The aggregate factor comes directly from
    /// `visual_factor`, preserving the normal solver path; only the audit's
    /// per-observation records are rebuilt when the detail trace is enabled.
    pub(super) fn visual_factor_with_audit(
        &self,
        state: &DVector<f64>,
        landmark_index: usize,
    ) -> Option<VisualLandmarkAudit> {
        let factor = self.visual_factor(state, landmark_index)?;
        let landmark = self.landmarks.get(landmark_index)?;
        let anchor_nav = self.block_nav_visual_current(state, landmark.anchor_state_index)?;
        let anchor_nav_fej = self.block_nav_linearized(state, landmark.anchor_state_index)?;
        let anchor_extrinsic = self.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let parameter = landmark.parameter(anchor_frame_id);
        let valid = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index < self.poses.len() + self.states.len())
            .filter_map(|observation| {
                let camera = self.camera_model(observation.camera_id)?;
                let target_extrinsic = self.camera_to_imu(observation.camera_id)?;
                let target_nav = self.block_nav_visual_current(state, observation.state_index)?;
                let target_nav_fej = self.block_nav_linearized(state, observation.state_index)?;
                let target_frame_id = self.block_frame_id(observation.state_index)?;
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.visual_block_is_linearized(landmark.anchor_state_index)?
                            || self.visual_block_is_linearized(observation.state_index)?,
                        self.factor_config,
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.factor_config,
                    )
                };
                factor.map(|factor| (*observation, factor))
            })
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return None;
        }
        let rows = valid
            .iter()
            .map(|(_, factor)| factor.residual.len())
            .sum::<usize>();
        let observations = valid
            .iter()
            .map(|(observation, factor)| VisualObservationAudit {
                observation: *observation,
                factor: factor.clone(),
            })
            .collect::<Vec<_>>();
        let invalid_rows = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index >= self.poses.len() + self.states.len())
            .count()
            * 2;
        debug_assert_eq!(factor.rows(), rows + invalid_rows);
        Some(VisualLandmarkAudit {
            track_id: landmark.track_id,
            anchor_frame_id,
            anchor_camera_id: landmark.anchor_camera_id,
            direction: landmark.direction,
            inverse_distance: landmark.inverse_distance,
            factor,
            observations,
        })
    }

    fn pose_offset_for_block(&self, index: usize) -> usize {
        match self.block_kind(index) {
            Some(WindowBlockKind::Pose) => self.pose_offset(index),
            Some(WindowBlockKind::State | WindowBlockKind::StatePose) => {
                self.state_offset(index - self.poses.len())
            }
            None => self.state_dof(),
        }
    }

    /// The visual-only rows (see [`VisualOnlyPrior`]): velocity/bias of every
    /// state towards zero, plus the scale row while no marginal prior exists.
    /// Placed right after the prior/anchor factor in every factor list.
    pub(super) fn visual_only_factor(
        &self,
        nav_of: impl Fn(usize) -> Option<BasaltNavState>,
    ) -> Option<WhitenedFactorRowStack> {
        let gauge = self.visual_only.as_ref()?;
        if self.states.is_empty() {
            return None;
        }
        let scale_index = self
            .prior
            .is_none()
            .then(|| self.block_index_for(gauge.scale_frame_id, WindowBlockKind::State))
            .flatten();
        let rows = 9 * self.states.len() + usize::from(scale_index.is_some());
        let mut jacobian = DMatrix::zeros(rows, self.state_dof());
        let mut rhs = DVector::zeros(rows);
        let w = VISUAL_ONLY_DECOUPLED_WEIGHT;
        for state_index in 0..self.states.len() {
            let offset = self.state_offset(state_index);
            let nav = nav_of(self.poses.len() + state_index)?;
            let values = nav
                .velocity_world_m_s
                .iter()
                .chain(nav.gyro_bias_rad_s.iter())
                .chain(nav.accel_bias_m_s2.iter());
            for (local, value) in values.enumerate() {
                let row = 9 * state_index + local;
                jacobian[(row, offset + 6 + local)] = w;
                rhs[row] = w * value;
            }
        }
        if let Some(index) = scale_index {
            let offset = self.block_offset(index, WindowBlockKind::State);
            let nav = nav_of(index)?;
            let pose_weight = self.initial_pose_weight.sqrt();
            let row = rows - 1;
            for axis in 0..3 {
                jacobian[(row, offset + axis)] = pose_weight * gauge.scale_direction[axis];
            }
            rhs[row] = pose_weight
                * gauge
                    .scale_direction
                    .dot(&(nav.imu_to_world.translation - gauge.scale_position));
        }
        prior_factor(jacobian, rhs)
    }

    /// Objective of [`Self::visual_only_factor`] (0 on IMU paths), added to the
    /// prior term of the f32 trial objective so the trial cost matches the
    /// linearized cost.
    pub(super) fn visual_only_objective(
        &self,
        nav_of: impl Fn(usize) -> Option<BasaltNavState>,
    ) -> f64 {
        self.visual_only_factor(nav_of)
            .map_or(0.0, |factor| factor.objective_cost)
    }
}
