//! `BundleAdjustment` GNC and weighted Levenberg-Marquardt backend.

use super::*;

impl BundleAdjustment {
    /// Outlier-robust bundle adjustment via Graduated Non-Convexity (GNC).
    ///
    /// A local M-estimator (`RobustKernel::Huber` / `Cauchy`) only
    /// down-weights gross reprojection errors *near* the current estimate,
    /// so a cluster of wrong correspondences that the initialisation
    /// already believes can capture the solution in a bad basin. GNC
    /// instead anneals a control parameter `ﾎｼ` from a convex surrogate
    /// (every observation trusted 窶・ordinary least squares) toward the true
    /// non-convex robust cost, recomputing the per-observation
    /// Black-Rangarajan weight `w 竏・[0,1]` at each level. Each level is a
    /// bounded weighted-LS solve reusing the same Schur-complement assembly
    /// as [`Self::optimize`] with `RobustKernel::None` (GNC supersedes the
    /// M-estimator). See [`crate::gnc`] for the surrogate math.
    ///
    /// `config` drives the inner LM solve (linear solver, ﾎｻ schedule);
    /// `config.robust_kernel` is ignored 窶・GNC sets the weights. `gnc.c` is
    /// the inlier reprojection scale **in pixels** (so `cﾂｲ` is the squared-
    /// residual band): pick it from the expected inlier reprojection error,
    /// e.g. `c 竕・3` for ~1 px noise. The returned
    /// [`BaGncResult::observation_weights`] gives the final per-observation
    /// weight (monocular, rectified stereo, then general stereo; `NaN` for un-evaluable
    /// observations); near-zero entries are the rejected outliers.
    pub fn optimize_gnc(
        &mut self,
        config: &BaConfig,
        gnc: &GncConfig,
    ) -> Result<BaGncResult, BaError> {
        let kernel_none = RobustKernel::None;
        let initial_cost = self.robust_cost(&kernel_none);
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();

        // GNC inlier scale: largest residual seeds the convex ﾎｼ竄; the same
        // residuals optionally drive the MAD auto-estimate of `c` (with the
        // configured `c` as a floor) so the pixel threshold tracks the actual
        // reprojection noise instead of a hand-set value.
        let squared_residuals = self.reprojection_squared_residuals();
        let s_max = squared_residuals
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .fold(0.0_f64, f64::max);
        let effective_gnc = match gnc.auto_scale {
            Some(k) => {
                let c = crate::gnc::estimate_scale_mad(&squared_residuals, k)
                    .map_or(gnc.c, |est| est.max(gnc.c));
                GncConfig { c, ..*gnc }
            }
            None => *gnc,
        };
        let mut inlier_scale = effective_gnc.c;
        let mut state = GncState::new(&effective_gnc, s_max);

        // Inner solve: a short weighted LM with no M-estimator (the GNC
        // weights are the only robustification) restarted at each ﾎｼ level.
        let mut inner = *config;
        inner.robust_kernel = RobustKernel::None;
        inner.max_iterations = gnc.inner_iterations.max(1);

        let mut weights = vec![1.0_f64; n];
        let mut converged = false;
        let mut outer_iterations = 0usize;
        for _ in 0..gnc.max_outer.max(1) {
            outer_iterations += 1;
            // The terminal level (ﾎｼ at its recovered extreme) reproduces the
            // true robust cost; we run it, then stop.
            let terminal_level = state.is_terminal();
            let residuals = self.reprojection_squared_residuals();
            // Adaptive inlier scale: re-derive `c` from the current residuals
            // each level (configured `c` as a floor). Level 0 reproduces the
            // one-shot estimate; later levels tighten as the surrogate
            // suppresses outliers and inlier residuals shrink.
            if gnc.auto_scale_readapt {
                if let Some(k) = gnc.auto_scale {
                    if let Some(est) = crate::gnc::estimate_scale_mad(&residuals, k) {
                        let c = est.max(gnc.c);
                        state.set_inlier_scale(c);
                        inlier_scale = c;
                    }
                }
            }
            for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
                *w = if s.is_finite() { state.weight(s) } else { 1.0 };
            }
            self.optimize_weighted(&inner, Some(&weights))?;
            if terminal_level {
                converged = true;
                break;
            }
            state.anneal();
        }

        // Final per-observation weights at the recovered estimate (NaN for
        // observations that cannot be evaluated, matching the result
        // contract and skipped by the weighted cost anyway).
        let residuals = self.reprojection_squared_residuals();
        for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
            *w = if s.is_finite() {
                state.weight(s)
            } else {
                f64::NAN
            };
        }
        let final_cost = self.robust_cost_weighted(&kernel_none, Some(&weights));

        // Inlier-only cost: hard 0/1 mask at the classification threshold,
        // so the reported cost reflects what survives outlier rejection.
        const INLIER_THRESHOLD: f64 = 0.5;
        let inlier_mask: Vec<f64> = weights
            .iter()
            .map(|&w| {
                if w.is_finite() {
                    if w >= INLIER_THRESHOLD {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            })
            .collect();
        let inlier_cost = self.robust_cost_weighted(&kernel_none, Some(&inlier_mask));

        Ok(BaGncResult {
            initial_cost,
            final_cost,
            inlier_cost,
            inlier_scale,
            observation_count: n,
            outer_iterations,
            converged,
            observation_weights: weights,
        })
    }

    /// Levenberg-Marquardt bundle adjustment with optional per-observation
    /// GNC weights folded into every reprojection contribution. `None`
    /// runs standard (optionally `RobustKernel`-IRLS) BA, bit-identical to
    /// the public [`Self::optimize`]; `Some(weights)` is the inner solve of
    /// [`Self::optimize_gnc`], where `weights` are the current
    /// Graduated-Non-Convexity surrogate weights and the cost used for the
    /// LM accept / reject test is correspondingly reweighted. `weights` is
    /// indexed monocular, rectified stereo, then general stereo (see
    /// [`Self::robust_cost_weighted`]).
    pub(super) fn optimize_weighted(
        &mut self,
        config: &BaConfig,
        gnc_weights: Option<&[f64]>,
    ) -> Result<BaResult, BaError> {
        let mut backend = BaSolveBackend::Legacy;
        self.optimize_weighted_backend(config, gnc_weights, &mut backend)
    }

    pub(super) fn optimize_weighted_backend(
        &mut self,
        config: &BaConfig,
        gnc_weights: Option<&[f64]>,
        mut backend: &mut BaSolveBackend,
    ) -> Result<BaResult, BaError> {
        let intrinsics = self.intrinsics().ok_or(BaError::UnsupportedCameraModel)?;
        self.check_stereo_camera_model()?;
        if self.poses.is_empty() {
            return Err(BaError::NoPoses);
        }
        let has_visual_observations = !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty();
        let has_imu_factors = !self.imu_factors.is_empty();
        if !has_visual_observations && !has_imu_factors {
            return Err(BaError::NoObservations);
        }
        // Visual residuals require landmarks; inertial-only solves (used by
        // the motion-based VI initialiser's VIBA1 stage) do not.
        if has_visual_observations && self.landmarks.is_empty() {
            return Err(BaError::NoLandmarks);
        }
        for obs in &self.observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
        }
        if !self.stereo_observations.is_empty() {
            match self.stereo_baseline {
                Some(b) if b.is_finite() && b > 0.0 => {}
                _ => return Err(BaError::MissingStereoBaseline),
            }
            for obs in &self.stereo_observations {
                if !self.poses.contains_key(&obs.keyframe_id) {
                    return Err(BaError::MissingPose(obs.keyframe_id));
                }
                if !self.landmarks.contains_key(&obs.landmark_id) {
                    return Err(BaError::MissingLandmark(obs.landmark_id));
                }
            }
        }
        for obs in &self.general_stereo_observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
            if obs.right_camera.intrinsics().is_none() {
                return Err(BaError::UnsupportedCameraModel);
            }
        }
        for observation in &self.rig_observations {
            if !self.poses.contains_key(&observation.keyframe_id) {
                return Err(BaError::MissingPose(observation.keyframe_id));
            }
            if !self.landmarks.contains_key(&observation.landmark_id) {
                return Err(BaError::MissingLandmark(observation.landmark_id));
            }
            if observation.camera.intrinsics().is_none() {
                return Err(BaError::UnsupportedCameraModel);
            }
        }

        // Variable layout: only non-fixed entries get a slot in the linear
        // system. Fixed poses / landmarks contribute residuals but no Hessian
        // or gradient block.
        let mut pose_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if self.fixed_poses.contains(&id) {
                continue;
            }
            let next = pose_index.len();
            pose_index.insert(id, next);
        }
        let mut landmark_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.landmarks.keys() {
            if self.fixed_landmarks.contains(&id) {
                continue;
            }
            let next = landmark_index.len();
            landmark_index.insert(id, next);
        }
        // Velocity slots: non-fixed velocities that appear on at least
        // one IMU factor. We DO NOT add a slot for every registered
        // velocity 窶・only the ones the factor touches 窶・so a stray
        // `add_velocity` without a corresponding `add_imu_factor` does
        // not introduce an unconstrained DoF that would singularise the
        // system.
        let mut velocity_index: BTreeMap<u64, usize> = BTreeMap::new();
        for factor in &self.imu_factors {
            for kf_id in [factor.keyframe_id_from, factor.keyframe_id_to] {
                if !self.velocities.contains_key(&kf_id) {
                    continue;
                }
                if self.fixed_velocities.contains(&kf_id) {
                    continue;
                }
                if velocity_index.contains_key(&kf_id) {
                    continue;
                }
                let next = velocity_index.len();
                velocity_index.insert(kf_id, next);
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                if self.velocities.contains_key(&kf_id)
                    && !self.fixed_velocities.contains(&kf_id)
                    && !velocity_index.contains_key(&kf_id)
                {
                    let next = velocity_index.len();
                    velocity_index.insert(kf_id, next);
                }
            }
        }
        // Bias slots: non-fixed biases registered on the "from" side of
        // an IMU factor OR on either side of a bias random-walk factor.
        // Same singularity guard as `velocity_index` 窶・a stray
        // `add_bias` without a matching factor does not introduce an
        // unconstrained DoF.
        let mut bias_index: BTreeMap<u64, usize> = BTreeMap::new();
        let register_bias_slot = |kf_id: u64, idx: &mut BTreeMap<u64, usize>| {
            if !self.biases.contains_key(&kf_id) {
                return;
            }
            if self.fixed_biases.contains(&kf_id) {
                return;
            }
            if idx.contains_key(&kf_id) {
                return;
            }
            let next = idx.len();
            idx.insert(kf_id, next);
        };
        for factor in &self.imu_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
        }
        for factor in &self.bias_random_walk_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
            register_bias_slot(factor.keyframe_id_to, &mut bias_index);
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                register_bias_slot(kf_id, &mut bias_index);
            }
        }
        if pose_index.is_empty()
            && landmark_index.is_empty()
            && velocity_index.is_empty()
            && bias_index.is_empty()
        {
            return Err(BaError::AllPosesFixed);
        }

        let kernel = config.robust_kernel;
        let initial_cost = self.robust_cost_weighted(&kernel, gnc_weights);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;
        let mut block_symbolic_cache = None;
        let mut rejected_system: Option<NormalEquationsBa> = None;
        let allow_retry_reuse = config.reuse_rejected_pose_diagonal
            && matches!(backend, BaSolveBackend::Legacy)
            && config.linear_solver == LinearSolver::Sparse
            && !pose_index.is_empty()
            && velocity_index.is_empty()
            && bias_index.is_empty()
            && !config.refine_intrinsics
            && !config.refine_distortion;
        let trace_phase_timing = std::env::var_os("VISLOC_BA_TRACE_PHASE_TIMING").is_some();

        static SPARSE_DEBUG_WINDOW_CLAIMED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        let sparse_debug_slot = claim_sparse_debug_window(
            std::env::var_os("VISLOC_SFM_DEBUG_BA_SPARSE_FIRST_WINDOW").is_some()
                && matches!(backend, BaSolveBackend::Legacy)
                && config.linear_solver == LinearSolver::Sparse
                && velocity_index.is_empty()
                && bias_index.is_empty()
                && !self.rig_observations.is_empty()
                && config.max_iterations > 0,
            ba_schur_debug_slot(),
            pose_index.len(),
            &SPARSE_DEBUG_WINDOW_CLAIMED,
        );

        for iteration in 0..config.max_iterations {
            let after_rejection = iterations.last().is_some_and(|step| !step.step_accepted);
            let emit_phase = |phase: &str, started: Option<std::time::Instant>| {
                if let Some(started) = started {
                    eprintln!("ba-phase-timing: iteration={} phase={} seconds={:.9} after_rejection={} variable_poses={} landmarks={}",
                        iteration, phase, started.elapsed().as_secs_f64(), after_rejection,
                        pose_index.len(), landmark_index.len());
                }
            };
            let assembly_started = trace_phase_timing.then(std::time::Instant::now);
            let adaptive_damping = match &backend {
                BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime.adaptive_damping,
                _ => false,
            };
            let prefer_pose_blocks = matches!(
                backend,
                BaSolveBackend::MatrixFree(_) | BaSolveBackend::MatrixFreeColumnScaled(_)
            ) || config.linear_solver == LinearSolver::Sparse;
            let use_landmark_qr =
                matches!(&backend, BaSolveBackend::MatrixFree(runtime) if runtime.landmark_qr);
            let reused = rejected_system.is_some();
            let mut system = rejected_system.take().or_else(|| {
                (!use_landmark_qr).then(|| {
                    build_normal_equations(
                        self,
                        &intrinsics,
                        &pose_index,
                        &landmark_index,
                        &velocity_index,
                        &bias_index,
                        &kernel,
                        gnc_weights,
                        config.parallel,
                        prefer_pose_blocks,
                    )
                })
            });
            if !reused {
                if let Some(system) = &mut system {
                    constrain_fixed_pose_rotations(&self.fixed_pose_rotations, &pose_index, system);
                }
            }
            log_process_memory("ba-after-normal-equations");
            emit_phase("normal_equations", assembly_started);

            // Build the reduced (Schur-complement) camera system. ﾎｻ is added
            // to both the pose and landmark diagonals before reduction so the
            // augmented system stays SPD when the un-damped one is rank-
            // deficient (as monocular BA generally is).
            let snapshot_started = trace_phase_timing.then(std::time::Instant::now);
            let saved_poses = self.poses.clone();
            let saved_landmarks = self.landmarks.clone();
            let saved_velocities = self.velocities.clone();
            let saved_biases = self.biases.clone();
            emit_phase("rollback_snapshot", snapshot_started);
            let cost_before = current_cost;
            // Keep the lambda actually supplied to this linear solve.  The
            // public BaIterationStats lambda intentionally retains its
            // historical post-rejection semantics.
            let solve_lambda = lambda;
            log_process_memory("ba-before-solve-step");

            // Diagnostic only: compare the same pre-step state without
            // changing the Legacy update. Strict caps bound duplicate storage.
            let qr_shadow = if matches!(&backend, BaSolveBackend::Legacy)
                && std::env::var("VISLOC_SFM_DEBUG_QR_SHARED_STATE").as_deref() == Ok("1")
                && qr_shadow_dimensions_allowed(
                    pose_index.len(),
                    self.poses.len(),
                    self.landmarks.len(),
                    self.rig_observations.len(),
                )
                && self.observations.is_empty()
                && self.stereo_observations.is_empty()
                && self.general_stereo_observations.is_empty()
                && gnc_weights.is_none()
            {
                Some(solve_rig_qr_step(
                    self,
                    config,
                    lambda,
                    MatrixFreeBaOptions::default(),
                    &landmark_index,
                ))
            } else {
                None
            };

            let solve_started = trace_phase_timing.then(std::time::Instant::now);
            // The sparse solver consumes only these undamped pose blocks.
            // Keep O(P) blocks, not a second normal system or dense Hessian.
            let original_diagonal = if allow_retry_reuse {
                system.as_ref().and_then(|system| match &system.h_pp {
                    CameraHessian::PoseDiagonal(blocks) => Some(blocks.clone()),
                    CameraHessian::Dense(_) => None,
                })
            } else {
                None
            };
            let retry_eligible = original_diagonal.is_some();
            let solve_result = match backend {
                BaSolveBackend::Legacy => solve_step_with_debug(
                    system.as_mut().expect("legacy normal system"),
                    pose_index.len(),
                    landmark_index.len(),
                    velocity_index.len(),
                    bias_index.len(),
                    lambda,
                    config.linear_solver,
                    config.parallel,
                    &mut block_symbolic_cache,
                    sparse_debug_slot.map(|slot| {
                        schur_debug_context_for_slot_with_coordinates(
                            &pose_index,
                            &landmark_index,
                            iteration,
                            slot,
                            false,
                        )
                    }),
                )
                .map(|(delta_poses, delta_landmarks)| (delta_poses, delta_landmarks, None, None)),
                BaSolveBackend::MatrixFree(runtime)
                | BaSolveBackend::MatrixFreeColumnScaled(runtime) => {
                    let column_scaled = runtime.column_scaling_iterations.is_some();
                    let scaling_result = if column_scaled {
                        match column_equilibrate_normal_system(
                            system.as_mut().expect("scaled normal system"),
                        ) {
                            Ok(mut state) => {
                                state.stats.iteration = iteration;
                                runtime
                                    .column_scaling_iterations
                                    .as_mut()
                                    .expect("column scaling diagnostics are enabled")
                                    .push(state.stats);
                                Ok(Some(state))
                            }
                            Err(error) => Err(MatrixFreeStepError {
                                diagnostic: format!("column scaling failed: {error}"),
                                pcg_iterations: None,
                                pcg_residual_norm: None,
                                pcg_target: None,
                                restart_diagnostics: None,
                            }),
                        }
                    } else {
                        Ok(None)
                    };
                    let schur_debug_context = matrix_free_schur_debug_context(
                        &pose_index,
                        &landmark_index,
                        iteration,
                        column_scaled,
                    );
                    let solve_result = match scaling_result {
                        Err(error) => Err(error),
                        Ok(_) if runtime.landmark_qr => solve_rig_qr_step(
                            self,
                            config,
                            lambda,
                            runtime.options,
                            &landmark_index,
                        ),
                        Ok(scaling_state) => solve_matrix_free_step(
                            system.as_ref().expect("Schur normal system"),
                            lambda,
                            runtime.options,
                            runtime.restart_limit,
                            runtime.restart_iterations.is_some(),
                            scaling_state.as_ref(),
                            schur_debug_context,
                            adaptive_damping,
                            runtime.cluster8,
                        ),
                    };
                    match solve_result {
                        Ok(mut outcome) => {
                            outcome.diagnostics.iteration = iteration;
                            runtime.iterations.push(outcome.diagnostics);
                            if let Some(mut restart_diagnostics) = outcome.restart_diagnostics {
                                restart_diagnostics.iteration = iteration;
                                runtime
                                    .restart_iterations
                                    .as_mut()
                                    .expect("restart diagnostics are enabled")
                                    .push(restart_diagnostics);
                            }
                            Ok((
                                outcome.delta_poses,
                                outcome.delta_landmarks,
                                outcome.adaptive_prediction,
                                outcome.quality,
                            ))
                        }
                        Err(error) => {
                            runtime.iterations.push(MatrixFreeBaIterationStats {
                                iteration,
                                pcg_iterations: error.pcg_iterations,
                                pcg_residual_norm: error.pcg_residual_norm,
                                pcg_target: error.pcg_target,
                                pcg_failure: Some(error.diagnostic.clone()),
                            });
                            if let Some(mut restart_diagnostics) = error.restart_diagnostics {
                                restart_diagnostics.iteration = iteration;
                                runtime
                                    .restart_iterations
                                    .as_mut()
                                    .expect("restart diagnostics are enabled")
                                    .push(restart_diagnostics);
                            }
                            if ba_lm_step_quality_debug_enabled() {
                                emit_matrix_free_step_quality_failure(
                                    iteration,
                                    solve_lambda,
                                    &error.diagnostic,
                                );
                            }
                            runtime.failure = Some(MatrixFreeBaError::LinearSolve {
                                iteration,
                                diagnostic: error.diagnostic,
                            });
                            Err(BaError::SingularSystem)
                        }
                    }
                }
            };
            emit_phase("linear_solve", solve_started);
            if let Some(diagonal) = original_diagonal {
                system.as_mut().expect("eligible normal system").h_pp =
                    CameraHessian::PoseDiagonal(diagonal);
            }
            if let Some(shadow) = qr_shadow {
                match (&solve_result, shadow) {
                    (Ok((poses, points, _, _)), Ok(qr)) => {
                        eprintln!(
                            "sfm-debug-qr-shared-state: iteration={iteration} lambda={solve_lambda:.17e} poses={} landmarks={} pose_difference={:.17e} point_difference={:.17e} legacy_pose_norm={:.17e} legacy_point_norm={:.17e} pcg_iterations={:?} residual={:?} target={:?}",
                            pose_index.len(), landmark_index.len(),
                            (&qr.delta_poses - poses).norm(), (&qr.delta_landmarks - points).norm(),
                            poses.norm(), points.norm(), qr.diagnostics.pcg_iterations,
                            qr.diagnostics.pcg_residual_norm, qr.diagnostics.pcg_target,
                        );
                    }
                    (_, Err(error)) => eprintln!(
                        "sfm-debug-qr-shared-state: iteration={iteration} lambda={solve_lambda:.17e} qr_failure={}",
                        error.diagnostic,
                    ),
                    (Err(error), Ok(_)) => eprintln!(
                        "sfm-debug-qr-shared-state: iteration={iteration} lambda={solve_lambda:.17e} legacy_failure={error:?}",
                    ),
                }
            }
            let (delta_poses, delta_landmarks, adaptive_prediction, quality) = match solve_result {
                Ok(d) => d,
                Err(BaError::SingularSystem) => {
                    if retry_eligible {
                        rejected_system = system.take();
                    }
                    // Treat singular system the same as a rejected LM step:
                    // bump λ and retry.
                    let next_lambda = if adaptive_damping {
                        bounded_adaptive_lambda(
                            lambda,
                            config.lambda_increase_factor,
                            config.min_lambda,
                            config.max_lambda,
                        )
                    } else {
                        (lambda * config.lambda_increase_factor).min(config.max_lambda)
                    };
                    if adaptive_damping {
                        if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                            runtime
                                .adaptive_iterations
                                .as_mut()
                                .expect("adaptive damping diagnostics are enabled")
                                .push(MatrixFreeBaAdaptiveDampingIterationStats {
                                    iteration,
                                    solve_lambda,
                                    next_lambda,
                                    predicted_undamped_squared_decrease: None,
                                    actual_cost_decrease: None,
                                    rho: None,
                                    cost_gate: None,
                                    feasibility_gate: None,
                                    nonprojectable_before: current_nonprojectable,
                                    nonprojectable_after: None,
                                    accepted: false,
                                    reason: "linear_failure".to_owned(),
                                });
                        }
                    }
                    lambda = next_lambda;
                    iterations.push(BaIterationStats {
                        iteration,
                        cost_before,
                        cost_after: cost_before,
                        max_pose_step: 0.0,
                        max_landmark_step: 0.0,
                        lambda,
                        step_accepted: false,
                    });
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
            log_process_memory("ba-after-solve-step");
            let update_started = trace_phase_timing.then(std::time::Instant::now);

            // Apply tentative update. `delta_poses` packs pose slots
            // first (`i * 6 .. i * 6 + 6`), then velocity slots
            // (`6P + v * 3 .. 6P + v * 3 + 3`), then bias slots
            // (`6P + 3V + b * 6 .. 6P + 3V + b * 6 + 6`).
            let mut max_pose_step: f64 = 0.0;
            let vel_offset_in_delta = pose_index.len() * 6;
            let bias_offset_in_delta = vel_offset_in_delta + velocity_index.len() * 3;
            for (&id, &i) in &pose_index {
                let mut xi = delta_poses.fixed_rows::<6>(i * 6).into_owned();
                if self.fixed_pose_rotations.contains(&id) {
                    xi[3] = 0.0;
                    xi[4] = 0.0;
                    xi[5] = 0.0;
                }
                let xi_vec: Vector6<f64> = xi;
                let step = xi_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let pose = self.poses.get_mut(&id).expect("pose exists");
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi_vec));
            }
            for (&id, &v) in &velocity_index {
                let dv = delta_poses
                    .fixed_rows::<3>(vel_offset_in_delta + v * 3)
                    .into_owned();
                let dv_vec: Vector3<f64> = dv;
                let step = dv_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let v_ref = self.velocities.get_mut(&id).expect("velocity exists");
                *v_ref += dv_vec;
            }
            for (&id, &b) in &bias_index {
                let db = delta_poses
                    .fixed_rows::<6>(bias_offset_in_delta + b * 6)
                    .into_owned();
                let db_vec: Vector6<f64> = db;
                let step = db_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let b_ref = self.biases.get_mut(&id).expect("bias exists");
                *b_ref += db_vec;
            }
            let mut max_landmark_step: f64 = 0.0;
            for (&id, &i) in &landmark_index {
                let dx = delta_landmarks.fixed_rows::<3>(i * 3).into_owned();
                let v: Vector3<f64> = dx;
                let step = v.norm();
                if step > max_landmark_step {
                    max_landmark_step = step;
                }
                let pt = self.landmarks.get_mut(&id).expect("landmark exists");
                *pt = Point3::from(pt.coords + v);
            }

            let nonprojectable_before = current_nonprojectable;
            let mut cost_after = self.robust_cost_weighted(&kernel, gnc_weights);
            let mut nonprojectable_after = self.nonprojectable_observation_count();
            let mut backtrack_exhausted = false;
            // Experimental bounded rescue: keep the production full-step path
            // unchanged unless explicitly enabled for pure rig sparse LM.
            if nonprojectable_after > current_nonprojectable
                && matches!(&backend, BaSolveBackend::Legacy)
                && config.linear_solver == LinearSolver::Sparse
                && config.initial_lambda.is_some()
                && velocity_index.is_empty()
                && bias_index.is_empty()
                && !self.rig_observations.is_empty()
                && self.observations.is_empty()
                && self.stereo_observations.is_empty()
                && self.general_stereo_observations.is_empty()
                && std::env::var("VISLOC_SFM_BA_FEASIBLE_BACKTRACK").as_deref() == Ok("1")
            {
                backtrack_exhausted = true;
                for alpha in [0.5, 0.25, 0.125, 0.0625] {
                    // Always start from the rollback state, never accumulate
                    // successive trial increments or copy an extra whole model.
                    for (&id, &i) in &pose_index {
                        let mut xi = delta_poses.fixed_rows::<6>(i * 6).into_owned() * alpha;
                        if self.fixed_pose_rotations.contains(&id) {
                            xi[3] = 0.0;
                            xi[4] = 0.0;
                            xi[5] = 0.0;
                        }
                        self.poses
                            .get_mut(&id)
                            .expect("pose exists")
                            .world_to_camera =
                            saved_poses[&id].world_to_camera.compose(&SE3::exp(&xi));
                    }
                    for (&id, &i) in &landmark_index {
                        let dx = delta_landmarks.fixed_rows::<3>(i * 3).into_owned() * alpha;
                        *self.landmarks.get_mut(&id).expect("landmark exists") =
                            Point3::from(saved_landmarks[&id].coords + dx);
                    }
                    let trial_cost = self.robust_cost_weighted(&kernel, gnc_weights);
                    let trial_nonprojectable = self.nonprojectable_observation_count();
                    let preserves_valid = self.rig_observations.iter().all(|obs| {
                        let old_valid = saved_poses
                            .get(&obs.keyframe_id)
                            .zip(saved_landmarks.get(&obs.landmark_id))
                            .is_some_and(|(pose, point)| {
                                rig_residual_jacobians(obs, pose, point).is_some()
                            });
                        !old_valid
                            || self
                                .poses
                                .get(&obs.keyframe_id)
                                .zip(self.landmarks.get(&obs.landmark_id))
                                .is_some_and(|(pose, point)| {
                                    rig_residual_jacobians(obs, pose, point).is_some()
                                })
                    });
                    let accepted = feasible_backtrack_accepts(
                        cost_before,
                        trial_cost,
                        current_nonprojectable,
                        trial_nonprojectable,
                        preserves_valid,
                    );
                    eprintln!("sfm-ba-feasible-backtrack: iteration={} alpha={} accepted={} cost={} nonprojectable={}", iteration, alpha, accepted, trial_cost, trial_nonprojectable);
                    cost_after = trial_cost;
                    nonprojectable_after = trial_nonprojectable;
                    if accepted {
                        backtrack_exhausted = false;
                        max_pose_step *= alpha;
                        max_landmark_step *= alpha;
                        break;
                    }
                }
                // On exhaustion force rejection and restore all saved states.
            }
            emit_phase("tentative_update_and_cost", update_started);
            let cost_accepted = match config.initial_lambda {
                None => true, // Pure GN: accept unconditionally.
                Some(_) => cost_after < cost_before,
            };
            let feasibility_gate =
                !backtrack_exhausted && nonprojectable_after <= current_nonprojectable;
            if sparse_debug_slot.is_some() && !feasibility_gate {
                for (observation, frame, landmark, depth) in self.nonprojectable_rig_sample(16) {
                    let factor = &self.rig_observations[observation];
                    let before = saved_poses.get(&frame).zip(saved_landmarks.get(&landmark));
                    let before_depth = before.map(|(pose, point)| {
                        factor
                            .sensor_from_rig
                            .transform_point(&pose.transform_world_point(point))
                            .z
                    });
                    let before_projectable = before.is_some_and(|(pose, point)| {
                        rig_residual_jacobians(factor, pose, point).is_some()
                    });
                    let point_step = saved_landmarks
                        .get(&landmark)
                        .zip(self.landmarks.get(&landmark))
                        .map(|(old, new)| (new - old).norm());
                    eprintln!("sfm-debug-ba-rig-infeasible: iteration={} observation={} frame_id={} track_id={} tentative_sensor_depth={:?} before_sensor_depth={:?} before_projectable={} point_step={:?} sample_limit=16", iteration, observation, frame, landmark, depth, before_depth, before_projectable, point_step);
                }
            }
            let mut step_accepted = cost_accepted && feasibility_gate;
            let adaptive_decision = if adaptive_damping {
                let decision = adaptive_step_decision(
                    adaptive_prediction,
                    cost_before,
                    cost_after,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_accepted,
                    nonprojectable_before == nonprojectable_after && nonprojectable_after == 0,
                );
                step_accepted &= decision.accepted;
                Some(decision)
            } else {
                None
            };

            if let Some(quality) = quality {
                emit_matrix_free_step_quality(
                    iteration,
                    solve_lambda,
                    quality,
                    cost_before,
                    cost_after,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_accepted,
                    nonprojectable_after <= nonprojectable_before,
                    step_accepted,
                );
            }

            // In particular, expose the two independent acceptance gates for
            // camera-fixed (landmark-only) solves.  A high robust cost can be
            // dominated by observations that are already down-weighted; a
            // candidate can also lower that cost while making more points
            // non-projectable, in which case the feasibility gate correctly
            // rejects it.  This line is diagnostic-only and is never emitted
            // unless both explicit BA step environment flags are set.
            if ba_step_debug_enabled() && (pose_index.is_empty() || !step_accepted) {
                eprintln!(
                    concat!(
                        "sfm-debug-ba-step-detail: poses={} landmarks={} iteration={} ",
                        "accepted={} cost_gate={} feasibility_gate={} ",
                        "nonprojectable={}->{} cost={:.9e}->{:.9e} lambda={:.3e}"
                    ),
                    pose_index.len(),
                    landmark_index.len(),
                    iteration,
                    step_accepted,
                    cost_accepted,
                    feasibility_gate,
                    nonprojectable_before,
                    nonprojectable_after,
                    cost_before,
                    cost_after,
                    lambda,
                );
            }

            if !step_accepted {
                if retry_eligible {
                    rejected_system = system.take();
                }
                self.poses = saved_poses;
                self.landmarks = saved_landmarks;
                self.velocities = saved_velocities;
                self.biases = saved_biases;
                let next_lambda = if adaptive_damping {
                    bounded_adaptive_lambda(
                        lambda,
                        config.lambda_increase_factor,
                        config.min_lambda,
                        config.max_lambda,
                    )
                } else {
                    (lambda * config.lambda_increase_factor).min(config.max_lambda)
                };
                if let Some(decision) = adaptive_decision {
                    if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                        runtime
                            .adaptive_iterations
                            .as_mut()
                            .expect("adaptive damping diagnostics are enabled")
                            .push(MatrixFreeBaAdaptiveDampingIterationStats {
                                iteration,
                                solve_lambda,
                                next_lambda,
                                predicted_undamped_squared_decrease: decision.prediction,
                                actual_cost_decrease: decision.actual_cost_decrease,
                                rho: decision.rho,
                                cost_gate: Some(decision.cost_gate),
                                feasibility_gate: Some(decision.feasibility_gate),
                                nonprojectable_before,
                                nonprojectable_after: Some(nonprojectable_after),
                                accepted: false,
                                reason: decision.reason,
                            });
                    }
                }
                lambda = next_lambda;
                iterations.push(BaIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_pose_step,
                    max_landmark_step,
                    lambda,
                    step_accepted: false,
                });
                if config.initial_lambda.is_none() {
                    break;
                }
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(BaIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_pose_step,
                max_landmark_step,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            current_nonprojectable = nonprojectable_after;
            if let Some(decision) = adaptive_decision {
                let next_lambda = adaptive_accepted_lambda(
                    lambda,
                    decision.rho.expect("accepted adaptive step has rho"),
                    config.min_lambda,
                    config.max_lambda,
                )
                .expect("accepted adaptive step has finite positive rho");
                if let BaSolveBackend::MatrixFreeColumnScaled(runtime) = &mut backend {
                    runtime
                        .adaptive_iterations
                        .as_mut()
                        .expect("adaptive damping diagnostics are enabled")
                        .push(MatrixFreeBaAdaptiveDampingIterationStats {
                            iteration,
                            solve_lambda,
                            next_lambda,
                            predicted_undamped_squared_decrease: decision.prediction,
                            actual_cost_decrease: decision.actual_cost_decrease,
                            rho: decision.rho,
                            cost_gate: Some(decision.cost_gate),
                            feasibility_gate: Some(decision.feasibility_gate),
                            nonprojectable_before,
                            nonprojectable_after: Some(nonprojectable_after),
                            accepted: true,
                            reason: decision.reason,
                        });
                }
                lambda = next_lambda;
            } else if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_pose_step < config.step_tolerance && max_landmark_step < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
            if config.relative_cost_tolerance.is_some_and(|tolerance| {
                tolerance.is_finite()
                    && tolerance >= 0.0
                    && (cost_before - cost_after) / cost_before.abs().max(f64::EPSILON) < tolerance
            }) {
                converged = true;
                break;
            }
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
}

pub(super) fn clear_visual_and_structural_costs(ba: &mut BundleAdjustment) {
    ba.observations.clear();
    ba.stereo_observations.clear();
    ba.general_stereo_observations.clear();
    ba.rig_observations.clear();
    ba.gravity_prior = None;
    ba.per_pose_gravity_prior = None;
    ba.position_prior = None;
    ba.pairwise_pose_factors.clear();
}
