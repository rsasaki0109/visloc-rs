//! `BundleAdjustment` optimizer entry points and input validation.

use super::*;

impl BundleAdjustment {
    /// Run Levenberg-Marquardt bundle adjustment with Schur-complement
    /// landmark elimination. Returns iteration trace and final cost.
    pub fn optimize(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        if !config.refine_intrinsics
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty()
        {
            return self.optimize_weighted(config, None);
        }
        // Joint pose + structure + intrinsics refinement (the COLMAP self-
        // calibration formulation) for the `[fx, fy, cx, cy(, k1, k2(, p1, p2))]`
        // layout of `Pinhole` / `OpenCv`. Other models fall back to the
        // pose/structure-only solve (which still uses their full lens model).
        if !matches!(
            self.camera.model,
            CameraModel::Pinhole | CameraModel::OpenCv
        ) || self.camera.intrinsics().is_none()
        {
            return self.optimize_weighted(config, None);
        }
        self.check_stereo_camera_model()?;
        self.optimize_joint_intrinsics(config)
    }

    /// Run pose/structure bundle adjustment with one external confidence
    /// weight per visual observation.
    ///
    /// The flattened order is monocular observations first, followed by
    /// rectified stereo and then calibrated general stereo. A weight of `1`
    /// preserves the ordinary BA contribution and `0` disables that visual
    /// residual. Intermediate values provide the confidence-weighted least
    /// squares used by learned correspondence update operators such as
    /// DROID-SLAM. The external weight multiplies the configured robust-kernel
    /// weight; inertial and structural priors are never reweighted.
    ///
    /// Joint intrinsics refinement is deliberately rejected because its
    /// separate normal-equation builder does not yet consume these weights.
    pub fn optimize_with_observation_weights(
        &mut self,
        config: &BaConfig,
        observation_weights: &[f64],
    ) -> Result<BaResult, BaError> {
        self.validate_observation_weights(observation_weights)?;
        if config.refine_intrinsics {
            return Err(BaError::ObservationWeightsWithIntrinsicsRefinement);
        }
        self.optimize_weighted(config, Some(observation_weights))
    }

    /// Run one bundle adjustment, dispatching to the matrix-free implicit-Schur
    /// PCG backend when `config.matrix_free_ba` is set (or the
    /// `VISLOC_SFM_BA_MATRIX_FREE` environment variable is present).
    ///
    /// This is the one entry point callers should use when they want the
    /// [`BaConfig::matrix_free_ba`] opt-in to apply: it returns the same
    /// [`BaResult`] shape as [`Self::optimize`] and transparently falls back to
    /// the ordinary solve when the problem is statically ineligible
    /// (intrinsics/distortion refinement, non-visual states or priors, a fully
    /// fixed pose set, or a missing gauge anchor). A numerical failure of the
    /// reduced solve is surfaced as an error rather than silently retried,
    /// matching the ordinary path's failure contract.
    ///
    /// External observation weights keep the legacy weighted objective, since
    /// the matrix-free operator does not consume them.
    pub fn optimize_honoring_matrix_free(
        &mut self,
        config: &BaConfig,
        observation_weights: Option<&[f64]>,
    ) -> Result<BaResult, BaError> {
        let matrix_free_requested =
            config.matrix_free_ba || std::env::var_os("VISLOC_SFM_BA_MATRIX_FREE").is_some();
        if matrix_free_requested && observation_weights.is_none() {
            match self.optimize_matrix_free(config, MatrixFreeBaOptions::default()) {
                Ok(result) => {
                    return Ok(BaResult {
                        initial_cost: result.initial_cost,
                        final_cost: result.final_cost,
                        iterations: result.iterations,
                        converged: result.converged,
                    })
                }
                Err(MatrixFreeBaError::Ineligible(_))
                | Err(MatrixFreeBaError::InvalidConfiguration(_)) => {
                    // Statically ineligible: fall through to the ordinary
                    // solve. The reason text is deliberately not logged here
                    // so library callers stay quiet; enable
                    // `VISLOC_SFM_TIMING`/`VISLOC_SFM_DEBUG` for the
                    // mapper-level diagnostic.
                }
                Err(MatrixFreeBaError::Ba(error)) => return Err(error),
                Err(MatrixFreeBaError::LinearSolve { .. }) => return Err(BaError::SingularSystem),
            }
        }
        match observation_weights {
            Some(weights) => self.optimize_with_observation_weights(config, weights),
            None => self.optimize(config),
        }
    }

    /// Run the opt-in matrix-free pure-visual bundle adjustment backend.
    ///
    /// The method deliberately leaves [`BaConfig`] and [`LinearSolver`]
    /// unchanged.  It assembles the same pure-visual normal equations as the
    /// ordinary optimizer, requires a `PoseDiagonal` camera Hessian, and
    /// solves its landmark-eliminated system with bounded preconditioned CG.
    /// `config.linear_solver` is ignored for this method; there is no dense
    /// fallback.  External observation weights and GNC remain on the legacy
    /// weighted entry points.
    /// Existing LM acceptance, rollback, damping, and non-projectable gates
    /// are shared with the ordinary weighted optimizer through a private
    /// backend dispatch.  A failed reduced solve is recorded and rejected by
    /// the same bounded LM retry path before any pose or landmark is modified.
    ///
    /// Gauge completeness is the caller's responsibility.  This entry point
    /// requires at least one actual fixed pose as a minimal anchor, but does
    /// not attempt to prove that every connected component (or monocular
    /// scale) is fully anchored.  Positive damping and PCG convergence are
    /// numerical facts, not a gauge proof.
    pub fn optimize_matrix_free(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<MatrixFreeBaResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        let (result, runtime) =
            self.run_matrix_free_backend(config, MatrixFreeRuntime::new(options))?;
        Ok(MatrixFreeBaResult {
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            iterations: result.iterations,
            matrix_free_iterations: runtime.iterations,
            converged: result.converged,
        })
    }

    /// Validate the same pure-visual matrix-free contract for a bounded
    /// landmark-only problem.  Native rig BA uses this only when every pose is
    /// fixed, so the reduced pose system has dimension zero; it does not
    /// expose a public solver mode or alter the ordinary matrix-free entry
    /// point.
    pub(crate) fn validate_matrix_free_landmark_only(
        &self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<(), MatrixFreeBaError> {
        if self.poses.keys().any(|id| !self.fixed_poses.contains(id)) {
            return Err(MatrixFreeBaError::Ineligible(
                "landmark-only matrix-free dispatch requires all poses fixed",
            ));
        }
        self.validate_matrix_free_entry_with_pose_requirement(config, options, false)
    }

    /// Experimental bounded QR elimination for calibrated, projectable rig rows.
    pub(crate) fn optimize_rig_qr(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<MatrixFreeBaResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        if self.rig_observations.is_empty()
            || !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || self.nonprojectable_observation_count() != 0
        {
            return Err(MatrixFreeBaError::Ineligible(
                "QR requires pure projectable rig observations",
            ));
        }
        let mut runtime = MatrixFreeRuntime::new(options);
        runtime.landmark_qr = true;
        let (result, runtime) = self.run_matrix_free_backend(config, runtime)?;
        Ok(MatrixFreeBaResult {
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            iterations: result.iterations,
            matrix_free_iterations: runtime.iterations,
            converged: result.converged,
        })
    }

    /// Experimental bounded eight-pose preconditioner for native rig BA.
    /// The exact operator, stopping criterion and LM policy are unchanged.
    pub(crate) fn optimize_matrix_free_cluster8(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<MatrixFreeBaResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        let mut runtime = MatrixFreeRuntime::new(options);
        runtime.cluster8 = true;
        let (result, runtime) = self.run_matrix_free_backend(config, runtime)?;
        Ok(MatrixFreeBaResult {
            initial_cost: result.initial_cost,
            final_cost: result.final_cost,
            iterations: result.iterations,
            matrix_free_iterations: runtime.iterations,
            converged: result.converged,
        })
    }

    /// Combine bounded cluster preconditioning with at most one existing
    /// true-residual restart, sharing the original total PCG iteration budget.
    pub(crate) fn optimize_matrix_free_cluster8_with_restart(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
        restart: MatrixFreeBaRestartOptions,
    ) -> Result<MatrixFreeBaRestartResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        if restart.max_restarts_per_solve > 1 {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "max_restarts_per_solve must be 0 or 1",
            ));
        }
        let mut runtime = MatrixFreeRuntime::with_restart(options, restart.max_restarts_per_solve);
        runtime.cluster8 = true;
        let (result, runtime) = self.run_matrix_free_backend(config, runtime)?;
        Ok(MatrixFreeBaRestartResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            restart_iterations: runtime.restart_iterations.unwrap_or_default(),
        })
    }

    /// Run matrix-free BA with explicit column equilibration and scaled LM
    /// damping.  This is a separate opt-in policy: the legacy matrix-free
    /// entry point keeps scalar `lambda * I` damping and its exact arithmetic.
    /// The returned PCG residuals are measured in the scaled coordinates.
    pub fn optimize_matrix_free_column_scaled(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaColumnScalingOptions,
    ) -> Result<MatrixFreeBaColumnScalingResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options.pcg)?;
        let (result, runtime) = self.run_matrix_free_column_scaled_backend(
            config,
            MatrixFreeRuntime::with_column_scaling(options.pcg),
        )?;
        Ok(MatrixFreeBaColumnScalingResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            scaling_iterations: runtime.column_scaling_iterations.unwrap_or_default(),
        })
    }

    /// Run column-scaled matrix-free BA with the opt-in adaptive LM damping
    /// policy.  The existing column-scaled entry point remains unchanged;
    /// this method changes only the accepted-step lambda update and records
    /// its scalar decision trace.  It requires zero initially non-projectable
    /// observations, a finite positive same-observation quadratic prediction,
    /// and the existing cost/feasibility gates for candidate acceptance.
    /// Linear failures and rejected candidates retain the configured lambda
    /// increase factor; only accepted candidates use the fixed rho rule
    /// `max(1/3, 1 - (2*rho - 1)^3)`.  PCG options and the normal-equation
    /// solve are otherwise identical.
    pub fn optimize_matrix_free_column_scaled_adaptive(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaColumnScalingOptions,
    ) -> Result<MatrixFreeBaAdaptiveDampingResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options.pcg)?;
        if self.nonprojectable_observation_count() != 0 {
            return Err(MatrixFreeBaError::Ineligible(
                "adaptive damping requires zero initial non-projectable observations",
            ));
        }
        let (result, runtime) = self.run_matrix_free_column_scaled_backend(
            config,
            MatrixFreeRuntime::with_column_scaling_adaptive(options.pcg),
        )?;
        Ok(MatrixFreeBaAdaptiveDampingResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            scaling_iterations: runtime.column_scaling_iterations.unwrap_or_default(),
            adaptive_iterations: runtime.adaptive_iterations.unwrap_or_default(),
        })
    }

    /// Run the matrix-free backend with a bounded true-residual restart policy.
    ///
    /// This is an additive diagnostic entry point.  The existing
    /// [`Self::optimize_matrix_free`] path remains the default and does not
    /// allocate restart diagnostics.  A restart is allowed at most once per
    /// reduced linear solve and never resets the total PCG iteration count.
    pub fn optimize_matrix_free_with_restart(
        &mut self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
        restart: MatrixFreeBaRestartOptions,
    ) -> Result<MatrixFreeBaRestartResult, MatrixFreeBaError> {
        self.validate_matrix_free_entry(config, options)?;
        if restart.max_restarts_per_solve > 1 {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "max_restarts_per_solve must be 0 or 1",
            ));
        }
        let (result, runtime) = self.run_matrix_free_backend(
            config,
            MatrixFreeRuntime::with_restart(options, restart.max_restarts_per_solve),
        )?;
        Ok(MatrixFreeBaRestartResult {
            ba: MatrixFreeBaResult {
                initial_cost: result.initial_cost,
                final_cost: result.final_cost,
                iterations: result.iterations,
                matrix_free_iterations: runtime.iterations,
                converged: result.converged,
            },
            restart_iterations: runtime.restart_iterations.unwrap_or_default(),
        })
    }

    pub(super) fn run_matrix_free_backend(
        &mut self,
        config: &BaConfig,
        runtime: MatrixFreeRuntime,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        self.run_matrix_free_backend_variant(config, BaSolveBackend::MatrixFree(runtime))
    }

    fn run_matrix_free_column_scaled_backend(
        &mut self,
        config: &BaConfig,
        runtime: MatrixFreeRuntime,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        self.run_matrix_free_backend_variant(
            config,
            BaSolveBackend::MatrixFreeColumnScaled(runtime),
        )
    }

    fn run_matrix_free_backend_variant(
        &mut self,
        config: &BaConfig,
        mut backend: BaSolveBackend,
    ) -> Result<(BaResult, MatrixFreeRuntime), MatrixFreeBaError> {
        let result = self
            .optimize_weighted_backend(config, None, &mut backend)
            .map_err(|error| {
                let runtime = match &backend {
                    BaSolveBackend::MatrixFree(runtime)
                    | BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime,
                    BaSolveBackend::Legacy => return MatrixFreeBaError::Ba(error),
                };
                if let Some(failure) = &runtime.failure {
                    return failure.clone();
                }
                MatrixFreeBaError::Ba(error)
            })?;
        let runtime = match backend {
            BaSolveBackend::MatrixFree(runtime)
            | BaSolveBackend::MatrixFreeColumnScaled(runtime) => runtime,
            BaSolveBackend::Legacy => {
                unreachable!("matrix-free entry installs a matrix-free backend")
            }
        };
        Ok((result, runtime))
    }

    pub(crate) fn validate_matrix_free_entry(
        &self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
    ) -> Result<(), MatrixFreeBaError> {
        self.validate_matrix_free_entry_with_pose_requirement(config, options, true)
    }

    fn validate_matrix_free_entry_with_pose_requirement(
        &self,
        config: &BaConfig,
        options: MatrixFreeBaOptions,
        require_variable_pose: bool,
    ) -> Result<(), MatrixFreeBaError> {
        if config.refine_intrinsics || config.refine_distortion {
            return Err(MatrixFreeBaError::Ineligible(
                "intrinsics/distortion refinement is not supported",
            ));
        }
        if config.max_iterations == 0 {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "max_iterations must be positive",
            ));
        }
        let initial_lambda = match config.initial_lambda {
            Some(value) if value.is_finite() && value > 0.0 => value,
            Some(_) => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "initial_lambda must be finite and positive",
                ))
            }
            None => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "matrix-free LM requires an explicit positive initial_lambda",
                ))
            }
        };
        if !config.lambda_increase_factor.is_finite()
            || config.lambda_increase_factor <= 1.0
            || !config.lambda_decrease_factor.is_finite()
            || config.lambda_decrease_factor <= 0.0
            || config.lambda_decrease_factor >= 1.0
            || !config.max_lambda.is_finite()
            || config.max_lambda < initial_lambda
            || !config.min_lambda.is_finite()
            || config.min_lambda <= 0.0
            || config.min_lambda > initial_lambda
            || !config.step_tolerance.is_finite()
            || config.step_tolerance < 0.0
            || !config.cost_tolerance.is_finite()
            || config.cost_tolerance < 0.0
            || config
                .relative_cost_tolerance
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "LM damping/tolerance values are not finite or ordered",
            ));
        }
        if options.max_pcg_iterations == 0
            || !options.pcg_relative_tolerance.is_finite()
            || options.pcg_relative_tolerance < 0.0
            || !options.pcg_absolute_tolerance.is_finite()
            || options.pcg_absolute_tolerance <= 0.0
        {
            return Err(MatrixFreeBaError::InvalidConfiguration(
                "PCG iteration and tolerance values are invalid",
            ));
        }
        match config.robust_kernel {
            RobustKernel::None => {}
            RobustKernel::Huber { delta } if delta.is_finite() && delta > 0.0 => {}
            RobustKernel::Cauchy { c } if c.is_finite() && c > 0.0 => {}
            _ => {
                return Err(MatrixFreeBaError::InvalidConfiguration(
                    "robust-kernel scale must be finite and positive",
                ))
            }
        }

        let calibration_is_finite = |camera: &Camera| {
            let Some((fx, fy, cx, cy)) = camera.intrinsics() else {
                return false;
            };
            let no_nonzero_distortion = camera
                .radial_distortion()
                .is_none_or(|(k1, k2)| k1 == 0.0 && k2 == 0.0);
            matches!(
                camera.model,
                CameraModel::Pinhole | CameraModel::SimplePinhole
            ) && no_nonzero_distortion
                && fx.is_finite()
                && fy.is_finite()
                && cx.is_finite()
                && cy.is_finite()
                && fx > 0.0
                && fy > 0.0
                && camera.params.iter().all(|value| value.is_finite())
        };
        if !calibration_is_finite(&self.camera) {
            return Err(MatrixFreeBaError::Ineligible(
                "camera calibration is unsupported or non-finite",
            ));
        }
        if !self.poses.keys().any(|id| self.fixed_poses.contains(id)) {
            return Err(MatrixFreeBaError::Ineligible(
                "at least one existing pose must be fixed as a gauge anchor",
            ));
        }
        if !self.velocities.is_empty()
            || !self.biases.is_empty()
            || !self.imu_factors.is_empty()
            || !self.fixed_velocities.is_empty()
            || !self.fixed_biases.is_empty()
            || !self.bias_random_walk_factors.is_empty()
            || self.navigation_state_prior.is_some()
            || self.gravity_prior.is_some()
            || self.per_pose_gravity_prior.is_some()
            || self.position_prior.is_some()
            || !self.pairwise_pose_factors.is_empty()
        {
            return Err(MatrixFreeBaError::Ineligible(
                "non-visual states or priors are not supported",
            ));
        }
        if self.poses.is_empty() {
            return Err(MatrixFreeBaError::Ba(BaError::NoPoses));
        }
        let has_visual_observations = !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
            || !self.rig_observations.is_empty();
        if !has_visual_observations {
            return Err(MatrixFreeBaError::Ba(BaError::NoObservations));
        }
        if self.landmarks.is_empty() {
            return Err(MatrixFreeBaError::Ba(BaError::NoLandmarks));
        }
        if require_variable_pose && !self.poses.keys().any(|id| !self.fixed_poses.contains(id)) {
            return Err(MatrixFreeBaError::Ineligible(
                "matrix-free backend requires a variable pose",
            ));
        }
        let pose_and_landmark_are_finite = self.poses.values().all(|pose| {
            pose.world_to_camera
                .matrix()
                .iter()
                .all(|value| value.is_finite())
        }) && self
            .landmarks
            .values()
            .all(|point| point.coords.iter().all(|value| value.is_finite()));
        if !pose_and_landmark_are_finite {
            return Err(MatrixFreeBaError::Ineligible(
                "pose or landmark state is non-finite",
            ));
        }
        if !self
            .robust_cost_weighted(&config.robust_kernel, None)
            .is_finite()
        {
            return Err(MatrixFreeBaError::Ineligible(
                "initial visual cost is non-finite",
            ));
        }
        if !self.observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "monocular observation has an invalid pose, landmark, or pixel",
            ));
        }
        if !self.stereo_observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && observation.u_right.is_finite()
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "stereo observation has invalid ids or pixels",
            ));
        }
        if !self.stereo_observations.is_empty()
            && !matches!(
                self.stereo_baseline,
                Some(value) if value.is_finite() && value > 0.0
            )
        {
            return Err(MatrixFreeBaError::Ba(BaError::MissingStereoBaseline));
        }
        if !self.general_stereo_observations.iter().all(|observation| {
            observation
                .xy_left
                .coords
                .iter()
                .all(|value| value.is_finite())
                && observation
                    .xy_right
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
                && calibration_is_finite(&observation.right_camera)
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
                && observation
                    .left_to_right
                    .translation
                    .iter()
                    .all(|value| value.is_finite())
                && observation
                    .left_to_right
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "general stereo calibration or observation is invalid",
            ));
        }
        if !self.rig_observations.iter().all(|observation| {
            observation.xy.coords.iter().all(|value| value.is_finite())
                && calibration_is_finite(&observation.camera)
                && self.poses.contains_key(&observation.keyframe_id)
                && self.landmarks.contains_key(&observation.landmark_id)
                && observation
                    .sensor_from_rig
                    .translation
                    .iter()
                    .all(|value| value.is_finite())
                && observation
                    .sensor_from_rig
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        }) {
            return Err(MatrixFreeBaError::Ineligible(
                "rig calibration or observation is invalid",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_observation_weights(
        &self,
        observation_weights: &[f64],
    ) -> Result<(), BaError> {
        let expected = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len()
            + self.rig_observations.len();
        if observation_weights.len() != expected {
            return Err(BaError::ObservationWeightCount {
                expected,
                actual: observation_weights.len(),
            });
        }
        if let Some((index, _)) = observation_weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
        {
            return Err(BaError::InvalidObservationWeight(index));
        }
        Ok(())
    }
}
