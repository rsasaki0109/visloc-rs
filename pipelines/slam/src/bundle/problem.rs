//! `BundleAdjustment` problem construction and cost evaluation.

use super::*;

impl BundleAdjustment {
    pub fn new(camera: Camera) -> Self {
        Self {
            poses: BTreeMap::new(),
            landmarks: BTreeMap::new(),
            observations: Vec::new(),
            stereo_observations: Vec::new(),
            general_stereo_observations: Vec::new(),
            rig_observations: Vec::new(),
            camera,
            fixed_poses: BTreeSet::new(),
            fixed_pose_rotations: BTreeSet::new(),
            fixed_landmarks: BTreeSet::new(),
            stereo_baseline: None,
            gravity_prior: None,
            per_pose_gravity_prior: None,
            position_prior: None,
            pairwise_pose_factors: Vec::new(),
            velocities: BTreeMap::new(),
            fixed_velocities: BTreeSet::new(),
            imu_factors: Vec::new(),
            imu_body_to_camera: SE3::identity(),
            biases: BTreeMap::new(),
            fixed_biases: BTreeSet::new(),
            bias_random_walk_factors: Vec::new(),
            navigation_state_prior: None,
        }
    }

    pub fn set_navigation_state_prior(&mut self, prior: NavigationStatePrior) {
        self.navigation_state_prior = Some(prior);
    }

    /// Linearize the navigation-only portion at the current estimate.
    /// Used by the fixed-lag VI stage to Schur-marginalize a state leaving
    /// the window. Landmark-bearing problems are deliberately rejected here:
    /// the boundary prior owns only inertial-chain information, preventing
    /// retained visual observations from being counted both in the prior and
    /// again in the next window.
    pub(crate) fn linearized_navigation_system(&self) -> Option<NavigationLinearization> {
        if !self.landmarks.is_empty()
            || !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty()
        {
            return None;
        }
        let intrinsics = self.intrinsics()?;
        let mut pose_index = BTreeMap::new();
        for &id in self.poses.keys() {
            if !self.fixed_poses.contains(&id) {
                let next = pose_index.len();
                pose_index.insert(id, next);
            }
        }
        let mut velocity_index = BTreeMap::new();
        for &id in self.velocities.keys() {
            if !self.fixed_velocities.contains(&id) {
                let next = velocity_index.len();
                velocity_index.insert(id, next);
            }
        }
        let mut bias_index = BTreeMap::new();
        for &id in self.biases.keys() {
            if !self.fixed_biases.contains(&id) {
                let next = bias_index.len();
                bias_index.insert(id, next);
            }
        }
        let system = build_normal_equations(
            self,
            &intrinsics,
            &pose_index,
            &BTreeMap::new(),
            &velocity_index,
            &bias_index,
            &RobustKernel::None,
            None,
            false,
            false,
        );
        let ordered_ids = |index: &BTreeMap<u64, usize>| {
            let mut ids: Vec<(usize, u64)> = index.iter().map(|(id, slot)| (*slot, *id)).collect();
            ids.sort_unstable();
            ids.into_iter().map(|(_, id)| id).collect()
        };
        Some(NavigationLinearization {
            pose_ids: ordered_ids(&pose_index),
            velocity_ids: ordered_ids(&velocity_index),
            bias_ids: ordered_ids(&bias_index),
            information: system.h_pp.into_dense(),
            gradient: system.b_p,
        })
    }

    /// Append a relative-pose factor between two keyframes. See
    /// [`PairwisePoseFactor`] for semantics.
    pub fn add_pairwise_pose_factor(&mut self, factor: PairwisePoseFactor) {
        self.pairwise_pose_factors.push(factor);
    }

    /// Register an initial world-frame velocity for the given keyframe.
    /// Required for any keyframe referenced by an
    /// [`ImuPreintegrationFactor`] 窶・the velocity becomes a BA variable
    /// (unless also passed to [`Self::fix_velocity`]).
    pub fn add_velocity(&mut self, id: u64, velocity: Vector3<f64>) {
        self.velocities.insert(id, velocity);
    }

    /// Pin the velocity of `id` so it does not change during
    /// [`Self::optimize`]. Useful for anchoring the initial keyframe's
    /// velocity to a measured value when the IMU factor would otherwise
    /// leave it under-constrained.
    pub fn fix_velocity(&mut self, id: u64) {
        self.fixed_velocities.insert(id);
    }

    /// Append an on-manifold IMU pre-integration factor between two
    /// keyframes. Both keyframes must have a [`Self::add_velocity`]
    /// entry; otherwise the factor is silently skipped during build
    /// (the rest of the BA still runs).
    pub fn add_imu_factor(&mut self, factor: ImuPreintegrationFactor) {
        self.imu_factors.push(factor);
    }

    /// Set the calibrated camera/sensor-to-body transform used only by IMU
    /// residuals. Reprojection residuals always retain the camera pose state.
    pub const fn set_imu_body_to_camera(&mut self, body_to_camera: SE3) {
        self.imu_body_to_camera = body_to_camera;
    }

    /// Register an initial IMU bias state for the given keyframe,
    /// packing the gyro bias in the first 3 components and the
    /// accelerometer bias in the last 3. The bias becomes a BA
    /// variable (unless also passed to [`Self::fix_bias`]). Required
    /// for any keyframe that should provide the bias-correction term
    /// for its outgoing [`ImuPreintegrationFactor`]; keyframes without
    /// a registered bias use the factor's linearisation bias
    /// (no correction) and do not contribute a bias Jacobian column.
    pub fn add_bias(&mut self, id: u64, bias: Vector6<f64>) {
        self.biases.insert(id, bias);
    }

    /// Pin the bias of `id` so it does not change during
    /// [`Self::optimize`]. The bias is still used for the residual's
    /// first-order correction (so the integration's linearisation
    /// point and the BA-side bias estimate stay decoupled), but no
    /// bias Jacobian column is added to the normal equations.
    pub fn fix_bias(&mut self, id: u64) {
        self.fixed_biases.insert(id);
    }

    /// Append a bias random-walk prior. Both endpoints should have
    /// been registered via [`Self::add_bias`]; the factor pulls
    /// `bias[keyframe_id_to]` toward `bias[keyframe_id_from]` with
    /// a weight of `weight`. See [`BiasRandomWalkFactor`].
    pub fn add_bias_random_walk_factor(&mut self, factor: BiasRandomWalkFactor) {
        self.bias_random_walk_factors.push(factor);
    }

    /// Install (or replace) the gravity prior used by [`Self::optimize`]
    /// and [`Self::robust_cost`]. See [`GravityPrior`] for semantics.
    pub const fn set_gravity_prior(&mut self, prior: GravityPrior) {
        self.gravity_prior = Some(prior);
    }

    /// Install (or replace) the per-keyframe gravity prior. See
    /// [`PerPoseGravityPrior`] for semantics; this is the online-
    /// friendly companion to [`Self::set_gravity_prior`] that accepts
    /// per-keyframe `g_camera_observed` observations.
    pub fn set_per_pose_gravity_prior(&mut self, prior: PerPoseGravityPrior) {
        self.per_pose_gravity_prior = Some(prior);
    }

    /// Install (or replace) the absolute position prior. See
    /// [`PositionPrior`] for semantics.
    pub fn set_position_prior(&mut self, prior: PositionPrior) {
        self.position_prior = Some(prior);
    }

    pub fn add_pose(&mut self, id: u64, pose: Pose) {
        self.poses.insert(id, pose);
    }

    pub fn fix_pose(&mut self, id: u64) {
        self.fixed_poses.insert(id);
    }

    /// Pin only the rotation of `id` during bundle adjustment.  Translation
    /// remains a variable, which is useful for diagnostic decompositions that
    /// separate rotational from translational error.  Calling this for a
    /// fully fixed pose is harmless.
    pub fn fix_pose_rotation(&mut self, id: u64) {
        self.fixed_pose_rotations.insert(id);
    }

    pub fn add_landmark(&mut self, id: u64, xyz: Point3<f64>) {
        self.landmarks.insert(id, xyz);
    }

    pub fn fix_landmark(&mut self, id: u64) {
        self.fixed_landmarks.insert(id);
    }

    pub fn add_observation(&mut self, obs: BaObservation) {
        self.observations.push(obs);
    }

    /// Append a rectified-stereo observation. Caller must call
    /// [`Self::set_stereo_baseline`] (with the same baseline used to
    /// triangulate `landmark_id`) before [`Self::optimize`], or the optimizer
    /// returns [`BaError::MissingStereoBaseline`].
    pub fn add_stereo_observation(&mut self, obs: BaStereoObservation) {
        self.stereo_observations.push(obs);
    }

    pub fn add_general_stereo_observation(&mut self, obs: BaGeneralStereoObservation) {
        self.general_stereo_observations.push(obs);
    }

    pub fn add_rig_observation(&mut self, observation: BaRigObservation) {
        self.rig_observations.push(observation);
    }

    /// Set the rectified-stereo baseline (positive, metric, in the units of
    /// the landmark coordinates). Required when any
    /// [`Self::stereo_observations`] are present.
    pub const fn set_stereo_baseline(&mut self, baseline: f64) {
        self.stereo_baseline = Some(baseline);
    }

    /// Sum of squared reprojection residuals `ﾎ｣ ||ﾏ(K ﾂｷ T ﾂｷ X_w) 竏・u||ﾂｲ`.
    /// Observations whose camera-frame point falls behind the camera are
    /// skipped (cost is reported as if those observations are absent).
    /// Equivalent to [`Self::robust_cost`] called with [`RobustKernel::None`].
    pub fn cost(&self) -> f64 {
        self.robust_cost(&RobustKernel::None)
    }

    /// Number of visual observations that cannot currently be projected.
    ///
    /// Reprojection cost and Jacobian assembly intentionally skip points on
    /// or behind the camera plane.  Without a separate feasibility check an
    /// LM step can therefore lower its reported cost merely by moving hard
    /// observations behind a camera.  Candidate steps are only accepted when
    /// this count does not increase.
    pub(super) fn nonprojectable_observation_count(&self) -> usize {
        let mono = self.observations.iter().filter(|obs| {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || self.camera.project(&xc).is_none()
        });

        let intrinsics = self.intrinsics();
        let baseline_valid = self
            .stereo_baseline
            .is_some_and(|baseline| baseline.is_finite() && baseline > 0.0);
        let stereo = self.stereo_observations.iter().filter(|obs| {
            let (Some(intrinsics), true, Some(pose), Some(point)) = (
                intrinsics,
                baseline_valid,
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || project_pinhole(&intrinsics, &xc).is_none()
        });

        let general_stereo = self.general_stereo_observations.iter().filter(|obs| {
            let (Some(pose), Some(point), Some(left_intrinsics)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
                self.intrinsics(),
            ) else {
                return true;
            };
            general_stereo_residual_jacobians(&left_intrinsics, obs, pose, point).is_none()
        });

        let rig = self.rig_observations.iter().filter(|observation| {
            let (Some(pose), Some(point)) = (
                self.poses.get(&observation.keyframe_id),
                self.landmarks.get(&observation.landmark_id),
            ) else {
                return true;
            };
            rig_residual_jacobians(observation, pose, point).is_none()
        });

        mono.count() + stereo.count() + general_stereo.count() + rig.count()
    }

    /// Bounded read-only sample using the exact rig feasibility predicate.
    pub(super) fn nonprojectable_rig_sample(
        &self,
        limit: usize,
    ) -> Vec<(usize, u64, u64, Option<f64>)> {
        self.rig_observations
            .iter()
            .enumerate()
            .filter_map(|(index, observation)| {
                let pose = self.poses.get(&observation.keyframe_id);
                let point = self.landmarks.get(&observation.landmark_id);
                if let (Some(pose), Some(point)) = (pose, point) {
                    if rig_residual_jacobians(observation, pose, point).is_some() {
                        return None;
                    }
                }
                let depth = pose.zip(point).map(|(pose, point)| {
                    observation
                        .sensor_from_rig
                        .transform_point(&pose.transform_world_point(point))
                        .z
                });
                Some((
                    index,
                    observation.keyframe_id,
                    observation.landmark_id,
                    depth,
                ))
            })
            .take(limit)
            .collect()
    }

    /// Robust reprojection cost: `Σ ρ(||r||²)` where `ρ` is the supplied
    /// [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::cost`]. Stereo observations contribute a 3-vector residual
    /// `(u_l_pred 竏・u_l_meas, v_l_pred 竏・v_l_meas, u_r_pred 竏・u_r_meas)`
    /// where `u_r_pred = u_l_pred 竏・fx ﾂｷ b / Z` (rectified-stereo assumption,
    /// see [`BaStereoObservation`]).
    pub fn robust_cost(&self, kernel: &RobustKernel) -> f64 {
        self.robust_cost_weighted(kernel, None)
    }

    /// Additive cost decomposition at the current linearisation point.
    ///
    /// `imu_normalized_squared_residual_per_dof` is the mean whitened
    /// squared IMU residual (NIS / 9 DoF per preintegration factor). It is
    /// useful for detecting a visual/inertial consistency failure that a
    /// single aggregate BA cost would otherwise hide.
    pub fn cost_breakdown(&self, kernel: &RobustKernel) -> BaCostBreakdown {
        self.cost_breakdown_weighted(kernel, None)
    }

    /// Confidence-weighted counterpart of [`Self::cost_breakdown`]. Visual
    /// terms use the same flattened weight vector as
    /// [`Self::optimize_with_observation_weights`]; IMU, navigation and other
    /// structural terms retain their physical information matrices.
    pub fn cost_breakdown_with_observation_weights(
        &self,
        kernel: &RobustKernel,
        observation_weights: &[f64],
    ) -> Result<BaCostBreakdown, BaError> {
        self.validate_observation_weights(observation_weights)?;
        Ok(self.cost_breakdown_weighted(kernel, Some(observation_weights)))
    }

    fn cost_breakdown_weighted(
        &self,
        kernel: &RobustKernel,
        observation_weights: Option<&[f64]>,
    ) -> BaCostBreakdown {
        let total = self.robust_cost_weighted(kernel, observation_weights);

        let mut visual_problem = self.clone();
        visual_problem.gravity_prior = None;
        visual_problem.per_pose_gravity_prior = None;
        visual_problem.position_prior = None;
        visual_problem.pairwise_pose_factors.clear();
        visual_problem.bias_random_walk_factors.clear();
        visual_problem.imu_factors.clear();
        visual_problem.navigation_state_prior = None;
        let visual = visual_problem.robust_cost_weighted(kernel, observation_weights);

        let mut imu_problem = self.clone();
        clear_visual_and_structural_costs(&mut imu_problem);
        imu_problem.bias_random_walk_factors.clear();
        imu_problem.navigation_state_prior = None;
        let imu = imu_problem.robust_cost(kernel);

        let mut bias_problem = self.clone();
        clear_visual_and_structural_costs(&mut bias_problem);
        bias_problem.imu_factors.clear();
        bias_problem.navigation_state_prior = None;
        let bias_random_walk = bias_problem.robust_cost(kernel);

        let mut navigation_problem = self.clone();
        clear_visual_and_structural_costs(&mut navigation_problem);
        navigation_problem.imu_factors.clear();
        navigation_problem.bias_random_walk_factors.clear();
        let navigation_prior = navigation_problem.robust_cost(kernel);

        let other_structural = total - visual - imu - bias_random_walk - navigation_prior;
        let imu_normalized_squared_residual_per_dof =
            (!self.imu_factors.is_empty()).then_some(imu / (9.0 * self.imu_factors.len() as f64));
        let (
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        ) = self
            .imu_raw_residual_rms()
            .map_or((None, None, None), |(rotation, velocity, position)| {
                (Some(rotation), Some(velocity), Some(position))
            });
        BaCostBreakdown {
            total,
            visual,
            imu,
            bias_random_walk,
            navigation_prior,
            other_structural,
            imu_normalized_squared_residual_per_dof,
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        }
    }

    /// Unwhitened per-axis RMS of the rotation, velocity, and position IMU
    /// residual blocks at the current state, in physical units.
    pub fn imu_raw_residual_rms(&self) -> Option<(f64, f64, f64)> {
        let mut rotation_squared = 0.0;
        let mut velocity_squared = 0.0;
        let mut position_squared = 0.0;
        let mut evaluated = 0usize;
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j), Some(v_i), Some(v_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rotation, r_velocity, r_position] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bias_gyro: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let bias_accel: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(
                        &r_i,
                        &p_i,
                        v_i,
                        &r_j,
                        &p_j,
                        v_j,
                        &bias_gyro,
                        &bias_accel,
                    )
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            rotation_squared += r_rotation.norm_squared();
            velocity_squared += r_velocity.norm_squared();
            position_squared += r_position.norm_squared();
            evaluated += 1;
        }
        (evaluated > 0).then(|| {
            let denominator = 3.0 * evaluated as f64;
            (
                (rotation_squared / denominator).sqrt(),
                (velocity_squared / denominator).sqrt(),
                (position_squared / denominator).sqrt(),
            )
        })
    }

    /// Like [`Self::robust_cost`] but multiplies each reprojection
    /// observation's contribution by an external per-observation weight
    /// (the Graduated Non-Convexity Black-Rangarajan weight `w 竏・[0,1]`).
    /// `gnc_weights` is indexed monocular-observations-first
    /// (`0 .. observations.len()`), rectified stereo, then general stereo
    /// (`observations.len() .. + stereo_observations.len()`). `None`
    /// reproduces [`Self::robust_cost`] exactly. Structural and inertial
    /// terms (gravity / position priors, pairwise pose, bias random-walk,
    /// IMU) are never reweighted 窶・only outlier-prone feature
    /// reprojections are, so a wrong correspondence is the only thing GNC
    /// can switch off.
    pub(super) fn robust_cost_weighted(
        &self,
        kernel: &RobustKernel,
        gnc_weights: Option<&[f64]>,
    ) -> f64 {
        let intrinsics = match self.intrinsics() {
            Some(k) => k,
            None => return 0.0,
        };
        let mut total = 0.0;
        for (obs_idx, obs) in self.observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            // Distortion-aware projection (identical to `project_pinhole` when the
            // camera carries no distortion, so all existing callers are unchanged).
            if let Some(predicted) = self.camera.project(&xc) {
                let r = predicted - obs.xy;
                let s = r.x * r.x + r.y * r.y;
                let w = gnc_weights.map_or(1.0, |gw| gw[obs_idx]);
                total += w * kernel.cost(s);
            }
        }
        if let Some(baseline) = self.stereo_baseline {
            if baseline.is_finite() && baseline > 0.0 {
                let (fx, _fy, _cx, _cy) = intrinsics;
                let stereo_offset = self.observations.len();
                for (st_idx, obs) in self.stereo_observations.iter().enumerate() {
                    let (Some(pose), Some(point)) = (
                        self.poses.get(&obs.keyframe_id),
                        self.landmarks.get(&obs.landmark_id),
                    ) else {
                        continue;
                    };
                    let xc = pose.transform_world_point(point);
                    if xc.z <= 0.0 {
                        continue;
                    }
                    if let Some(predicted) = project_pinhole(&intrinsics, &xc) {
                        let u_r_pred = predicted.x - fx * baseline / xc.z;
                        let dx = predicted.x - obs.xy.x;
                        let dy = predicted.y - obs.xy.y;
                        let dr = u_r_pred - obs.u_right;
                        let s = dx * dx + dy * dy + dr * dr;
                        let w = gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);
                        total += w * kernel.cost(s);
                    }
                }
            }
        }
        let general_offset = self.observations.len() + self.stereo_observations.len();
        for (index, obs) in self.general_stereo_observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let Some((residual, _, _)) =
                general_stereo_residual_jacobians(&intrinsics, obs, pose, point)
            else {
                continue;
            };
            let w = gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
            total += w * kernel.cost(residual.norm_squared());
        }
        let rig_offset = general_offset + self.general_stereo_observations.len();
        for (index, observation) in self.rig_observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&observation.keyframe_id),
                self.landmarks.get(&observation.landmark_id),
            ) else {
                continue;
            };
            let Some((residual, _, _)) = rig_residual_jacobians(observation, pose, point) else {
                continue;
            };
            let weight = gnc_weights.map_or(1.0, |weights| weights[rig_offset + index]);
            total += weight * kernel.cost(residual.norm_squared());
        }
        if let Some(prior) = &self.gravity_prior {
            for pose in self.poses.values() {
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - prior.g_camera_observed;
                let s = r_vec.norm_squared();
                // Gravity prior uses an L2 (non-robust) contribution: the
                // measurement is global per pose, not a per-feature
                // outlier-prone observation, so a robust kernel here would
                // hide rather than down-weight prior窶電ata conflicts.
                total += prior.weight * s;
            }
        }
        if let Some(prior) = &self.per_pose_gravity_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - obs.g_camera_observed;
                let s = r_vec.norm_squared();
                // Same L2 (non-robust) reasoning as [`Self::gravity_prior`].
                total += prior.weight * obs.weight * s;
            }
        }
        if let Some(prior) = &self.position_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let c_world = pose.camera_center_world();
                let r_vec = c_world - obs.camera_center_world;
                // Axis-weighted L2 cost: ﾎ｣ w盞｢ ﾂｷ r盞｢ﾂｲ. Zero weight axes
                // contribute nothing, so an altitude-only prior with
                // `axis_weights = (0, w, 0)` is exact.
                let s = obs.axis_weights.x * r_vec.x * r_vec.x
                    + obs.axis_weights.y * r_vec.y * r_vec.y
                    + obs.axis_weights.z * r_vec.z * r_vec.z;
                total += s;
            }
        }
        for factor in &self.pairwise_pose_factors {
            let (Some(from), Some(to)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = factor
                .measurement
                .world_to_camera
                .inverse()
                .compose(&predicted)
                .log();
            total += factor.weight * r.norm_squared();
        }
        for factor in &self.bias_random_walk_factors {
            let (Some(b_i), Some(b_j)) = (
                self.biases.get(&factor.keyframe_id_from),
                self.biases.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let r: Vector6<f64> = b_j - b_i;
            total += factor.weight_gyro.max(0.0) * r.fixed_rows::<3>(0).norm_squared()
                + factor.weight_accel.max(0.0) * r.fixed_rows::<3>(3).norm_squared();
        }
        // IMU pre-integration factors: 9-vector residual [r_R; r_v; r_p]
        // weighted axis-wise. The factor's `residual` helper takes the
        // body-to-world rotation and world-frame body centre obtained by
        // composing the camera pose with the calibrated T_b<-c extrinsic.
        // When `self.biases` carries a bias for the factor's "from"
        // keyframe, the bias-corrected residual is used (Forster eq. 44).
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let (Some(v_i), Some(v_j)) = (
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rot, r_vel, r_pos] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let ba: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(&r_i, &p_i, v_i, &r_j, &p_j, v_j, &bg, &ba)
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            if let Some(whitener) = factor.covariance_sqrt_information() {
                let mut residual = nalgebra::SVector::<f64, 9>::zeros();
                residual.fixed_rows_mut::<3>(0).copy_from(&r_rot);
                residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
                residual.fixed_rows_mut::<3>(6).copy_from(&r_pos);
                total += (whitener * residual).norm_squared();
            } else {
                total += factor.weight_rotation * r_rot.norm_squared()
                    + factor.weight_velocity * r_vel.norm_squared()
                    + factor.weight_position * r_pos.norm_squared();
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            if let Some(delta) = navigation_prior_delta(self, prior) {
                let quadratic = delta.dot(&(&prior.information * &delta));
                total += prior.constant_cost + 2.0 * prior.gradient.dot(&delta) + quadratic;
            }
        }
        total
    }

    /// `(fx, fy, cx, cy)` of the BA camera, or `None` when its model has no
    /// projection the BA can linearise (an `Unknown` COLMAP model).
    ///
    /// Monocular residuals use the camera's full lens model through
    /// [`MonoProjection`] (radial, tangential, rational, equidistant fisheye,
    /// FOV and Double Sphere all carry analytic Jacobians), so every model with
    /// a [`Camera::project`] is admitted. The rectified / general stereo terms
    /// only implement the pinhole; [`Self::check_stereo_camera_model`] keeps
    /// them restricted to `Pinhole` / `SimplePinhole` cameras.
    pub(super) fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        match self.camera.model {
            CameraModel::Unknown(_) => None,
            _ => self.camera.intrinsics(),
        }
    }

    /// Rectified and general stereo residuals project through the plain
    /// pinhole, so they are only admitted with the two camera models the BA has
    /// always paired them with (`Pinhole` / `SimplePinhole`); any other lens
    /// would be silently linearised as a pinhole.
    pub(super) fn check_stereo_camera_model(&self) -> Result<(), BaError> {
        let has_stereo =
            !self.stereo_observations.is_empty() || !self.general_stereo_observations.is_empty();
        if has_stereo
            && !matches!(
                self.camera.model,
                CameraModel::Pinhole | CameraModel::SimplePinhole
            )
        {
            return Err(BaError::UnsupportedCameraModel);
        }
        Ok(())
    }

    /// Per-observation squared reprojection residual `s = 窶睦窶鳴ｲ` (pixelﾂｲ),
    /// evaluated at the current state and aligned to the GNC weight layout
    /// used everywhere in this file: monocular observations first
    /// (`0 .. observations.len()`), then rectified stereo, then general
    /// stereo. An observation that cannot
    /// be evaluated now (missing pose / landmark, behind the camera, or
    /// non-projectable, or 窶・for stereo 窶・no usable baseline) is reported
    /// as `f64::NAN`, so it neither sets the GNC inlier scale nor is
    /// classified as an inlier or outlier.
    pub(super) fn reprojection_squared_residuals(&self) -> Vec<f64> {
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();
        let mut out = Vec::with_capacity(n);
        let Some(intrinsics) = self.intrinsics() else {
            out.resize(n, f64::NAN);
            return out;
        };
        let mono_projection = MonoProjection::for_camera(&self.camera);
        for obs in &self.observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = if mono_projection == MonoProjection::Pinhole {
                    project_pinhole(&intrinsics, &xc)?
                } else {
                    self.camera.project(&xc)?
                };
                let r = predicted - obs.xy;
                Some(r.x * r.x + r.y * r.y)
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        let baseline = match self.stereo_baseline {
            Some(b) if b.is_finite() && b > 0.0 => Some(b),
            _ => None,
        };
        let (fx, _, _, _) = intrinsics;
        for obs in &self.stereo_observations {
            let s = baseline.and_then(|baseline| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = project_pinhole(&intrinsics, &xc)?;
                let u_r_pred = predicted.x - fx * baseline / xc.z;
                let dx = predicted.x - obs.xy.x;
                let dy = predicted.y - obs.xy.y;
                let dr = u_r_pred - obs.u_right;
                Some(dx * dx + dy * dy + dr * dr)
            });
            out.push(s.unwrap_or(f64::NAN));
        }
        for obs in &self.general_stereo_observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let (residual, _, _) =
                    general_stereo_residual_jacobians(&intrinsics, obs, pose, point)?;
                Some(residual.norm_squared())
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        for observation in &self.rig_observations {
            let squared = (|| {
                let pose = self.poses.get(&observation.keyframe_id)?;
                let point = self.landmarks.get(&observation.landmark_id)?;
                let (residual, _, _) = rig_residual_jacobians(observation, pose, point)?;
                Some(residual.norm_squared())
            })();
            out.push(squared.unwrap_or(f64::NAN));
        }
        out
    }
}
