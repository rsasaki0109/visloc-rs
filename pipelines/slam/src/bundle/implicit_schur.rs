use super::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum ImplicitSchurError {
    DenseCameraHessian,
    EmptySystem,
    DimensionMismatch {
        expected: usize,
        actual: usize,
    },
    NonFinite(&'static str),
    InvalidLambda,
    InvalidTolerance,
    NonSpdPreconditioner(usize),
    NonPositiveCurvature,
    ResidualCheckFailed {
        iterations: usize,
        recursive_norm: f64,
        true_norm: f64,
        target: f64,
    },
    MaxIterations {
        iterations: usize,
        recursive_norm: f64,
        residual_norm: f64,
        target: f64,
    },
}

impl ImplicitSchurError {
    pub(super) const fn diagnostics(&self) -> (Option<usize>, Option<f64>, Option<f64>) {
        match *self {
            Self::ResidualCheckFailed {
                iterations,
                true_norm,
                target,
                ..
            }
            | Self::MaxIterations {
                iterations,
                residual_norm: true_norm,
                target,
                ..
            } => (Some(iterations), Some(true_norm), Some(target)),
            _ => (None, None, None),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PcgOptions {
    pub(super) max_iterations: usize,
    pub(super) relative_tolerance: f64,
    pub(super) absolute_tolerance: f64,
}

impl Default for PcgOptions {
    fn default() -> Self {
        Self {
            max_iterations: 128,
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PcgResult {
    pub(super) solution: DVector<f64>,
    pub(super) iterations: usize,
    pub(super) residual_norm: f64,
    pub(super) target: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct PcgRunDiagnostics {
    pub(super) pcg_iterations: Option<usize>,
    pub(super) true_residual_rechecks: usize,
    pub(super) failed_true_residual_rechecks: usize,
    pub(super) restarts: usize,
}

impl From<PcgRunDiagnostics> for MatrixFreeBaRestartIterationStats {
    fn from(value: PcgRunDiagnostics) -> Self {
        Self {
            iteration: 0,
            pcg_iterations: value.pcg_iterations,
            true_residual_rechecks: value.true_residual_rechecks,
            failed_true_residual_rechecks: value.failed_true_residual_rechecks,
            restarts: value.restarts,
            terminal_failure: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct PcgSolveFailure {
    pub(super) error: ImplicitSchurError,
    pub(super) diagnostics: PcgRunDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PcgRun {
    pub(super) result: PcgResult,
    pub(super) diagnostics: PcgRunDiagnostics,
}

/// A pure-visual Schur operator over the six-dimensional pose blocks.
///
/// `landmarks` and their cross blocks are borrowed from
/// `NormalEquationsBa`; only the per-landmark 3×3 inverse cache and the
/// pose-block-Jacobi inverse are retained.  A repeated pose slot in one
/// landmark is intentional: the constructor groups those blocks only for
/// the preconditioner, while `apply` evaluates every original cross block
/// so same-frame multi-sensor terms are not lost.
pub(super) struct ImplicitSchurOperator<'a> {
    diagonal: Vec<Matrix6<f64>>,
    landmarks: &'a [LandmarkBlock],
    h_ll_inverse: Vec<Option<Matrix3<f64>>>,
    preconditioner_inverse: Vec<Matrix6<f64>>,
    cluster_inverse: Option<Vec<DMatrix<f64>>>,
    rhs: DVector<f64>,
}

impl<'a> ImplicitSchurOperator<'a> {
    pub(super) fn new(
        system: &'a NormalEquationsBa,
        lambda: f64,
    ) -> Result<Self, ImplicitSchurError> {
        Self::new_with_debug(system, lambda, None)
    }

    pub(super) fn new_with_debug(
        system: &'a NormalEquationsBa,
        lambda: f64,
        debug: Option<SchurBlockDebugContext<'_>>,
    ) -> Result<Self, ImplicitSchurError> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(ImplicitSchurError::InvalidLambda);
        }
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            return Err(ImplicitSchurError::DenseCameraHessian);
        };
        if diagonal.is_empty() {
            return Err(ImplicitSchurError::EmptySystem);
        }
        let dimension = diagonal.len() * 6;
        if system.b_p.len() != dimension {
            return Err(ImplicitSchurError::DimensionMismatch {
                expected: dimension,
                actual: system.b_p.len(),
            });
        }
        if !system.b_p.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("pose gradient"));
        }
        if !diagonal
            .iter()
            .flat_map(|block| block.iter())
            .all(|value| value.is_finite())
        {
            return Err(ImplicitSchurError::NonFinite("pose Hessian"));
        }

        let mut damped_diagonal = diagonal.clone();
        if lambda > 0.0 {
            for block in &mut damped_diagonal {
                for component in 0..6 {
                    block[(component, component)] += lambda;
                }
            }
        }
        if !damped_diagonal
            .iter()
            .flat_map(|block| block.iter())
            .all(|value| value.is_finite())
        {
            return Err(ImplicitSchurError::NonFinite("damped pose Hessian"));
        }

        let mut h_ll_inverse = Vec::with_capacity(system.landmarks.len());
        let mut rhs = -&system.b_p;
        for landmark in &system.landmarks {
            if !landmark
                .h_ll
                .iter()
                .chain(landmark.b_l.iter())
                .all(|value| value.is_finite())
            {
                return Err(ImplicitSchurError::NonFinite("landmark block"));
            }
            for (pose, cross) in &landmark.cross {
                if *pose >= diagonal.len() {
                    return Err(ImplicitSchurError::DimensionMismatch {
                        expected: diagonal.len(),
                        actual: (*pose).saturating_add(1),
                    });
                }
                if !cross.iter().all(|value| value.is_finite()) {
                    return Err(ImplicitSchurError::NonFinite("landmark cross block"));
                }
            }

            let mut h_ll = landmark.h_ll;
            if lambda > 0.0 {
                for component in 0..3 {
                    h_ll[(component, component)] += lambda;
                }
            }
            if !h_ll.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("damped landmark Hessian"));
            }
            let inverse = h_ll.try_inverse();
            if let Some(inverse) = inverse {
                if !inverse.iter().all(|value| value.is_finite()) {
                    return Err(ImplicitSchurError::NonFinite("landmark inverse"));
                }
            }
            h_ll_inverse.push(inverse);
            let Some(inverse) = inverse else {
                // Match solve_step: an un-invertible landmark contributes
                // neither to the Schur system nor to back-substitution.
                continue;
            };
            for (pose, cross) in &landmark.cross {
                let update: Vector6<f64> = cross * inverse * landmark.b_l;
                for component in 0..6 {
                    rhs[pose * 6 + component] += update[component];
                }
            }
        }
        if !rhs.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("reduced right hand side"));
        }

        // Block-Jacobi preconditioner: aggregate all cross blocks that
        // touch the same pose before forming the diagonal Schur term.
        // This is essential when two sensors observe the same landmark
        // from one rig frame: the cross-sensor terms belong in the same
        // pose block, not in two independent diagonal blocks.
        let mut preconditioner = damped_diagonal.clone();
        for (landmark, inverse) in system.landmarks.iter().zip(&h_ll_inverse) {
            let Some(inverse) = inverse else {
                continue;
            };
            let mut grouped: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
            for (pose, cross) in &landmark.cross {
                *grouped.entry(*pose).or_insert_with(Matrix6x3::zeros) += cross;
            }
            for (pose, cross) in grouped {
                preconditioner[pose] -= cross * inverse * cross.transpose();
            }
        }

        if let Some(context) = debug {
            if context.pose_slot < preconditioner.len() {
                let counts = collect_schur_block_debug_counts(
                    system,
                    &h_ll_inverse,
                    lambda,
                    context.pose_slot,
                );
                let h_pp_lambda = &damped_diagonal[context.pose_slot];
                let schur_block = &preconditioner[context.pose_slot];
                emit_schur_block_debug(context, lambda, h_pp_lambda, schur_block, counts);
            }
        }

        if !preconditioner
            .iter()
            .flat_map(|block| block.iter())
            .all(|value| value.is_finite())
        {
            return Err(ImplicitSchurError::NonFinite("Schur preconditioner"));
        }

        let mut preconditioner_inverse = Vec::with_capacity(preconditioner.len());
        for (pose, block) in preconditioner.iter().enumerate() {
            let Some(cholesky) = block.cholesky() else {
                return Err(ImplicitSchurError::NonSpdPreconditioner(pose));
            };
            let inverse = cholesky.inverse();
            if !inverse.iter().all(|value| value.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("preconditioner inverse"));
            }
            preconditioner_inverse.push(inverse);
        }

        Ok(Self {
            diagonal: damped_diagonal,
            landmarks: &system.landmarks,
            h_ll_inverse,
            preconditioner_inverse,
            cluster_inverse: None,
            rhs,
        })
    }

    /// Principal Schur blocks on fixed groups of at most eight poses.
    /// O(P*K) retained scalars and O(cross_rows*K) work, K=8. In particular,
    /// never enumerate a landmark's global pose-pair clique.
    pub(super) fn with_cluster8(mut self) -> Result<Self, ImplicitSchurError> {
        const K: usize = 8;
        let poses = self.diagonal.len();
        let mut blocks: Vec<DMatrix<f64>> = self
            .diagonal
            .chunks(K)
            .map(|chunk| {
                let mut block = DMatrix::zeros(chunk.len() * 6, chunk.len() * 6);
                for (slot, diagonal) in chunk.iter().enumerate() {
                    block
                        .view_mut((slot * 6, slot * 6), (6, 6))
                        .copy_from(diagonal);
                }
                block
            })
            .collect();
        // Scratch is allocated once, reset only for touched slots. Each
        // cluster's list has at most K entries even for repeated sensors.
        let mut cross_sum = vec![Matrix6x3::zeros(); poses];
        let mut seen = vec![false; poses];
        let mut members: Vec<Vec<usize>> = (0..blocks.len()).map(|_| Vec::new()).collect();
        let mut touched = Vec::new();
        for (landmark, inverse) in self.landmarks.iter().zip(&self.h_ll_inverse) {
            let Some(inverse) = inverse else {
                continue;
            };
            for (pose, cross) in &landmark.cross {
                let cluster = *pose / K;
                if !seen[*pose] {
                    if members[cluster].is_empty() {
                        touched.push(cluster);
                    }
                    members[cluster].push(*pose);
                    seen[*pose] = true;
                }
                cross_sum[*pose] += cross;
            }
            for &cluster in &touched {
                let block = &mut blocks[cluster];
                for &a in &members[cluster] {
                    let left = cross_sum[a] * inverse;
                    for &b in &members[cluster] {
                        let update = left * cross_sum[b].transpose();
                        for r in 0..6 {
                            for c in 0..6 {
                                block[((a % K) * 6 + r, (b % K) * 6 + c)] -= update[(r, c)];
                            }
                        }
                    }
                }
                for pose in members[cluster].drain(..) {
                    seen[pose] = false;
                    cross_sum[pose].fill(0.0);
                }
            }
            touched.clear();
        }
        for (cluster, block) in blocks.iter_mut().enumerate() {
            if !block.iter().all(|v| v.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("cluster preconditioner"));
            }
            let factor = block
                .clone()
                .cholesky()
                .ok_or(ImplicitSchurError::NonSpdPreconditioner(cluster * K))?;
            *block = factor.inverse();
            if !block.iter().all(|v| v.is_finite()) {
                return Err(ImplicitSchurError::NonFinite("cluster inverse"));
            }
        }
        self.preconditioner_inverse.clear();
        self.cluster_inverse = Some(blocks);
        Ok(self)
    }

    pub(super) fn dimension(&self) -> usize {
        self.diagonal.len() * 6
    }

    pub(super) const fn rhs(&self) -> &DVector<f64> {
        &self.rhs
    }

    #[cfg(test)]
    pub(super) fn h_ll_inverse(&self, index: usize) -> Option<Matrix3<f64>> {
        self.h_ll_inverse[index]
    }

    pub(super) fn apply(&self, x: &DVector<f64>) -> Result<DVector<f64>, ImplicitSchurError> {
        if x.len() != self.dimension() {
            return Err(ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: x.len(),
            });
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("operator input"));
        }
        let mut out = DVector::zeros(self.dimension());
        for (pose, block) in self.diagonal.iter().enumerate() {
            let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
            let value = block * x_pose;
            for component in 0..6 {
                out[pose * 6 + component] = value[component];
            }
        }
        for (landmark, inverse) in self.landmarks.iter().zip(&self.h_ll_inverse) {
            let Some(inverse) = inverse else {
                continue;
            };
            let mut projected = Vector3::zeros();
            // Keep the original cross vector order.  Do not deduplicate
            // here: duplicate pose slots encode separate sensor rows.
            for (pose, cross) in &landmark.cross {
                let x_pose: Vector6<f64> = x.fixed_rows::<6>(pose * 6).into_owned();
                projected += cross.transpose() * x_pose;
            }
            let reduced = *inverse * projected;
            for (pose, cross) in &landmark.cross {
                let value: Vector6<f64> = cross * reduced;
                for component in 0..6 {
                    out[pose * 6 + component] -= value[component];
                }
            }
        }
        if !out.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("operator output"));
        }
        Ok(out)
    }

    pub(super) fn apply_preconditioner(
        &self,
        residual: &DVector<f64>,
    ) -> Result<DVector<f64>, ImplicitSchurError> {
        if residual.len() != self.dimension() {
            return Err(ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: residual.len(),
            });
        }
        if !residual.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("preconditioner input"));
        }
        let mut out = DVector::zeros(self.dimension());
        if let Some(clusters) = &self.cluster_inverse {
            let mut offset = 0;
            for inverse in clusters {
                let n = inverse.nrows();
                out.rows_mut(offset, n)
                    .copy_from(&(inverse * residual.rows(offset, n)));
                offset += n;
            }
            if !out.iter().all(|v| v.is_finite()) {
                return Err(ImplicitSchurError::NonFinite(
                    "cluster preconditioner output",
                ));
            }
            return Ok(out);
        }
        for (pose, inverse) in self.preconditioner_inverse.iter().enumerate() {
            let residual_pose: Vector6<f64> = residual.fixed_rows::<6>(pose * 6).into_owned();
            let value = inverse * residual_pose;
            for component in 0..6 {
                out[pose * 6 + component] = value[component];
            }
        }
        if !out.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("preconditioner output"));
        }
        Ok(out)
    }

    pub(super) fn solve_pcg(
        &self,
        rhs: &DVector<f64>,
        options: PcgOptions,
    ) -> Result<PcgResult, ImplicitSchurError> {
        self.solve_pcg_with_restart(rhs, options, 0)
            .map(|run| run.result)
            .map_err(|failure| failure.error)
    }

    pub(super) fn solve_pcg_with_restart(
        &self,
        rhs: &DVector<f64>,
        options: PcgOptions,
        max_restarts: usize,
    ) -> Result<PcgRun, PcgSolveFailure> {
        self.solve_pcg_with_restart_hook(rhs, options, max_restarts, |_, residual, norm| {
            Ok((residual, norm))
        })
    }

    #[cfg(test)]
    pub(super) fn solve_pcg_with_injected_recursive_residual_for_test(
        &self,
        rhs: &DVector<f64>,
        options: PcgOptions,
        max_restarts: usize,
    ) -> Result<PcgRun, PcgSolveFailure> {
        let mut inject_once = true;
        self.solve_pcg_with_restart_hook(
            rhs,
            options,
            max_restarts,
            move |check, residual, norm| {
                if check == 1 && inject_once {
                    inject_once = false;
                    // This test-only hook emulates recursive residual
                    // cancellation after the first alpha update.  The subsequent true
                    // residual check still computes b-Ax from the
                    // operator, so no true residual or norm is forged.
                    return Ok((DVector::zeros(residual.len()), 0.0));
                }
                Ok((residual, norm))
            },
        )
    }

    fn solve_pcg_with_restart_hook<F>(
        &self,
        rhs: &DVector<f64>,
        options: PcgOptions,
        max_restarts: usize,
        mut recursive_residual_adjustment: F,
    ) -> Result<PcgRun, PcgSolveFailure>
    where
        F: FnMut(usize, DVector<f64>, f64) -> Result<(DVector<f64>, f64), ImplicitSchurError>,
    {
        let mut diagnostics = PcgRunDiagnostics::default();
        let fail = |error, diagnostics| PcgSolveFailure { error, diagnostics };
        if rhs.len() != self.dimension() {
            return Err(fail(
                ImplicitSchurError::DimensionMismatch {
                    expected: self.dimension(),
                    actual: rhs.len(),
                },
                diagnostics,
            ));
        }
        if !rhs.iter().all(|value| value.is_finite()) {
            return Err(fail(
                ImplicitSchurError::NonFinite("PCG right hand side"),
                diagnostics,
            ));
        }
        if !options.relative_tolerance.is_finite()
            || options.relative_tolerance < 0.0
            || !options.absolute_tolerance.is_finite()
            || options.absolute_tolerance < 0.0
        {
            return Err(fail(ImplicitSchurError::InvalidTolerance, diagnostics));
        }
        // A converged zero-RHS or positively damped system only proves a
        // numerical linear solve.  It does not prove that the model's
        // monocular/rig gauge has been anchored; that remains a caller-
        // level invariant outside this prototype.
        let rhs_norm = rhs.norm();
        let target = options
            .absolute_tolerance
            .max(options.relative_tolerance * rhs_norm);
        if !target.is_finite() {
            return Err(fail(
                ImplicitSchurError::NonFinite("PCG target"),
                diagnostics,
            ));
        }
        let mut solution = DVector::zeros(self.dimension());
        let mut residual = rhs.clone();
        let mut residual_norm = residual.norm();
        if !residual_norm.is_finite() {
            return Err(fail(
                ImplicitSchurError::NonFinite("initial residual"),
                diagnostics,
            ));
        }
        // The solve has passed input/target validation.  Recording zero
        // explicitly makes zero-RHS and zero-budget outcomes distinct
        // from failures that occurred before a PCG iteration began.
        diagnostics.pcg_iterations = Some(0);
        if residual_norm <= target {
            diagnostics.true_residual_rechecks += 1;
            let (true_residual, true_norm) = self
                .true_residual(rhs, &solution)
                .map_err(|error| fail(error, diagnostics))?;
            if true_norm <= target {
                return Ok(PcgRun {
                    result: PcgResult {
                        solution,
                        iterations: 0,
                        residual_norm: true_norm,
                        target,
                    },
                    diagnostics,
                });
            }
            diagnostics.failed_true_residual_rechecks += 1;
            if max_restarts == 0 || options.max_iterations == 0 {
                return Err(fail(
                    ImplicitSchurError::ResidualCheckFailed {
                        iterations: 0,
                        recursive_norm: residual_norm,
                        true_norm,
                        target,
                    },
                    diagnostics,
                ));
            }
            residual = true_residual;
            residual_norm = true_norm;
            diagnostics.restarts = 1;
        }
        if options.max_iterations == 0 {
            return Err(fail(
                ImplicitSchurError::MaxIterations {
                    iterations: 0,
                    recursive_norm: residual_norm,
                    residual_norm,
                    target,
                },
                diagnostics,
            ));
        }

        let (mut direction, mut rho) = self
            .restart_direction(&residual)
            .map_err(|error| fail(error, diagnostics))?;

        for iteration in 1..=options.max_iterations {
            let applied = self
                .apply(&direction)
                .map_err(|error| fail(error, diagnostics))?;
            let curvature = direction.dot(&applied);
            if !curvature.is_finite() || curvature <= 0.0 {
                return Err(fail(ImplicitSchurError::NonPositiveCurvature, diagnostics));
            }
            let alpha = rho / curvature;
            if !alpha.is_finite() {
                return Err(fail(ImplicitSchurError::NonFinite("PCG step"), diagnostics));
            }
            solution += alpha * &direction;
            residual -= alpha * applied;
            residual_norm = residual.norm();
            (residual, residual_norm) =
                recursive_residual_adjustment(iteration, residual, residual_norm)
                    .map_err(|error| fail(error, diagnostics))?;
            diagnostics.pcg_iterations = Some(iteration);
            if !solution.iter().all(|value| value.is_finite()) || !residual_norm.is_finite() {
                return Err(fail(
                    ImplicitSchurError::NonFinite("PCG iterate"),
                    diagnostics,
                ));
            }
            if residual_norm <= target {
                diagnostics.true_residual_rechecks += 1;
                let (true_residual, true_norm) = self
                    .true_residual(rhs, &solution)
                    .map_err(|error| fail(error, diagnostics))?;
                if true_norm <= target {
                    return Ok(PcgRun {
                        result: PcgResult {
                            solution,
                            iterations: iteration,
                            residual_norm: true_norm,
                            target,
                        },
                        diagnostics,
                    });
                }
                diagnostics.failed_true_residual_rechecks += 1;
                if diagnostics.restarts < max_restarts && iteration < options.max_iterations {
                    residual = true_residual;
                    (direction, rho) = self
                        .restart_direction(&residual)
                        .map_err(|error| fail(error, diagnostics))?;
                    diagnostics.restarts += 1;
                    continue;
                }
                return Err(fail(
                    ImplicitSchurError::ResidualCheckFailed {
                        iterations: iteration,
                        recursive_norm: residual_norm,
                        true_norm,
                        target,
                    },
                    diagnostics,
                ));
            }
            if iteration == options.max_iterations {
                diagnostics.true_residual_rechecks += 1;
                let (_, true_norm) = self
                    .true_residual(rhs, &solution)
                    .map_err(|error| fail(error, diagnostics))?;
                if true_norm <= target {
                    return Ok(PcgRun {
                        result: PcgResult {
                            solution,
                            iterations: iteration,
                            residual_norm: true_norm,
                            target,
                        },
                        diagnostics,
                    });
                }
                diagnostics.failed_true_residual_rechecks += 1;
                return Err(fail(
                    ImplicitSchurError::MaxIterations {
                        iterations: iteration,
                        recursive_norm: residual_norm,
                        residual_norm: true_norm,
                        target,
                    },
                    diagnostics,
                ));
            }
            let preconditioned = self
                .apply_preconditioner(&residual)
                .map_err(|error| fail(error, diagnostics))?;
            let next_rho = residual.dot(&preconditioned);
            if !next_rho.is_finite() || next_rho <= 0.0 {
                return Err(fail(ImplicitSchurError::NonPositiveCurvature, diagnostics));
            }
            let beta = next_rho / rho;
            if !beta.is_finite() {
                return Err(fail(
                    ImplicitSchurError::NonFinite("PCG direction"),
                    diagnostics,
                ));
            }
            direction = &preconditioned + beta * direction;
            rho = next_rho;
        }
        unreachable!("the max-iteration branch returns above");
    }

    fn restart_direction(
        &self,
        residual: &DVector<f64>,
    ) -> Result<(DVector<f64>, f64), ImplicitSchurError> {
        let preconditioned = self.apply_preconditioner(residual)?;
        let rho = residual.dot(&preconditioned);
        if !rho.is_finite() || rho <= 0.0 {
            return Err(ImplicitSchurError::NonPositiveCurvature);
        }
        Ok((preconditioned, rho))
    }

    fn true_residual(
        &self,
        rhs: &DVector<f64>,
        solution: &DVector<f64>,
    ) -> Result<(DVector<f64>, f64), ImplicitSchurError> {
        let true_residual = rhs - &self.apply(solution)?;
        let true_norm = true_residual.norm();
        if !true_norm.is_finite() {
            return Err(ImplicitSchurError::NonFinite("true PCG residual"));
        }
        Ok((true_residual, true_norm))
    }

    #[cfg(test)]
    fn checked_result(
        &self,
        rhs: &DVector<f64>,
        solution: DVector<f64>,
        iterations: usize,
        recursive_norm: f64,
        target: f64,
    ) -> Result<PcgResult, ImplicitSchurError> {
        let (_, true_norm) = self.true_residual(rhs, &solution)?;
        if true_norm > target {
            return Err(ImplicitSchurError::ResidualCheckFailed {
                iterations,
                recursive_norm,
                true_norm,
                target,
            });
        }
        Ok(PcgResult {
            solution,
            iterations,
            residual_norm: true_norm,
            target,
        })
    }

    pub(super) fn complete_delta(
        &self,
        delta_pose: &DVector<f64>,
    ) -> Result<DVector<f64>, ImplicitSchurError> {
        if delta_pose.len() != self.dimension() {
            return Err(ImplicitSchurError::DimensionMismatch {
                expected: self.dimension(),
                actual: delta_pose.len(),
            });
        }
        if !delta_pose.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("pose delta"));
        }
        let mut delta_landmarks = DVector::zeros(self.landmarks.len() * 3);
        for (index, (landmark, inverse)) in
            self.landmarks.iter().zip(&self.h_ll_inverse).enumerate()
        {
            let Some(inverse) = inverse else {
                continue;
            };
            let mut accumulated = -landmark.b_l;
            for (pose, cross) in &landmark.cross {
                let delta: Vector6<f64> = delta_pose.fixed_rows::<6>(pose * 6).into_owned();
                accumulated -= cross.transpose() * delta;
            }
            let value = *inverse * accumulated;
            for component in 0..3 {
                delta_landmarks[index * 3 + component] = value[component];
            }
        }
        if !delta_landmarks.iter().all(|value| value.is_finite()) {
            return Err(ImplicitSchurError::NonFinite("landmark delta"));
        }
        Ok(delta_landmarks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn explicit_schur(system: &NormalEquationsBa, lambda: f64) -> DMatrix<f64> {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            panic!("test oracle requires pose blocks");
        };
        let dimension = diagonal.len() * 6;
        let mut schur = DMatrix::zeros(dimension, dimension);
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += lambda;
            }
            schur
                .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                .copy_from(&damped);
        }
        for landmark in &system.landmarks {
            let mut h_ll = landmark.h_ll;
            for component in 0..3 {
                h_ll[(component, component)] += lambda;
            }
            let Some(inverse) = h_ll.try_inverse() else {
                continue;
            };
            for (pose, a) in &landmark.cross {
                for (other_pose, b) in &landmark.cross {
                    let block = a * inverse * b.transpose();
                    for row in 0..6 {
                        for column in 0..6 {
                            schur[(pose * 6 + row, other_pose * 6 + column)] -=
                                block[(row, column)];
                        }
                    }
                }
            }
        }
        schur
    }

    fn synthetic_system() -> NormalEquationsBa {
        let diagonal = vec![
            Matrix6::from_diagonal(&Vector6::from_element(20.0)),
            Matrix6::from_diagonal(&Vector6::from_element(22.0)),
        ];
        let cross0 =
            Matrix6x3::from_fn(|row, column| 0.02 * (row as f64 + 1.0) * (column as f64 + 1.0));
        let cross1 =
            Matrix6x3::from_fn(|row, column| 0.015 * (row as f64 + 2.0) * (column as f64 + 1.0));
        let cross2 =
            Matrix6x3::from_fn(|row, column| 0.01 * (row as f64 + 3.0) * (column as f64 + 2.0));
        NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(diagonal),
            b_p: DVector::from_iterator(12, (0..12).map(|index| 0.03 * (index + 1) as f64)),
            landmarks: vec![LandmarkBlock {
                h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
                b_l: Vector3::new(0.2, -0.1, 0.3),
                // The first two entries deliberately share pose 0, as two
                // sensors on one rig frame would.
                cross: vec![(0, cross0), (0, cross1), (1, cross2)],
            }],
        }
    }

    fn hand_check_system() -> NormalEquationsBa {
        let mut a = Matrix6x3::zeros();
        for component in 0..3 {
            a[(component, component)] = 1.0;
        }
        NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(vec![
                Matrix6::from_diagonal(&Vector6::from_element(10.0)),
                Matrix6::from_diagonal(&Vector6::from_element(10.0)),
            ]),
            b_p: DVector::from_iterator(12, (1..=12).map(|value| value as f64)),
            landmarks: vec![LandmarkBlock {
                h_ll: Matrix3::from_diagonal(&Vector3::from_element(4.0)),
                b_l: Vector3::new(1.0, 2.0, 3.0),
                // B = 10 I, H_ll = 4 I, A = [I; 0], with two sensor
                // observations sharing pose 0 and one observation at pose 1.
                cross: vec![(0, a), (0, 2.0 * a), (1, -a)],
            }],
        }
    }

    fn full_normal_system(system: &NormalEquationsBa, lambda: f64) -> (DMatrix<f64>, DVector<f64>) {
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            panic!("test oracle requires pose blocks");
        };
        let mut normal = DMatrix::zeros(15, 15);
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += lambda;
            }
            normal
                .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
                .copy_from(&damped);
        }
        let mut h_ll = system.landmarks[0].h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        normal.fixed_view_mut::<3, 3>(12, 12).copy_from(&h_ll);
        for (pose, cross) in &system.landmarks[0].cross {
            for row in 0..6 {
                for column in 0..3 {
                    normal[(pose * 6 + row, 12 + column)] += cross[(row, column)];
                    normal[(12 + column, pose * 6 + row)] += cross[(row, column)];
                }
            }
        }
        let mut rhs = DVector::zeros(15);
        for (index, value) in system.b_p.iter().enumerate() {
            rhs[index] = -*value;
        }
        for (index, value) in system.landmarks[0].b_l.iter().enumerate() {
            rhs[12 + index] = -*value;
        }
        (normal, rhs)
    }

    #[test]
    fn implicit_operator_matches_explicit_schur_rhs_preconditioner_and_step() {
        let lambda = 0.25;
        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, lambda).unwrap();
        let explicit = explicit_schur(&system, lambda);
        let x = DVector::from_iterator(12, (0..12).map(|index| 0.04 * (index + 1) as f64));
        let explicit_product = &explicit * &x;
        let implicit_product = operator.apply(&x).unwrap();
        assert!((&explicit_product - implicit_product).norm() < 1.0e-12);

        let mut explicit_rhs = -&system.b_p;
        let h_ll_inverse = operator.h_ll_inverse(0).unwrap();
        for (pose, cross) in &system.landmarks[0].cross {
            let update: Vector6<f64> = cross * h_ll_inverse * system.landmarks[0].b_l;
            for component in 0..6 {
                explicit_rhs[pose * 6 + component] += update[component];
            }
        }
        assert_eq!(operator.rhs.as_slice(), explicit_rhs.as_slice());

        let mut direct_system = synthetic_system();
        let direct = solve_step(
            &mut direct_system,
            2,
            1,
            0,
            0,
            lambda,
            LinearSolver::Sparse,
            false,
            &mut None,
        )
        .unwrap();
        let result = operator
            .solve_pcg(operator.rhs(), PcgOptions::default())
            .unwrap();
        assert!(result.residual_norm <= result.target);
        assert!((&result.solution - &direct.0).norm() < 1.0e-9);
        let implicit_landmark_delta = operator.complete_delta(&result.solution).unwrap();
        assert!((&implicit_landmark_delta - direct.1).norm() < 1.0e-9);

        // The pose-0 preconditioner block must include the cross term between
        // the two repeated pose-0 sensor blocks, not two independent terms.
        let mut pose0_cross = system.landmarks[0].cross[0].1;
        pose0_cross += system.landmarks[0].cross[1].1;
        let mut expected = Matrix6::from_diagonal(&Vector6::from_element(20.0));
        for component in 0..6 {
            expected[(component, component)] += lambda;
        }
        expected -= pose0_cross * h_ll_inverse * pose0_cross.transpose();
        let probe = DVector::from_iterator(12, (0..12).map(|index| (index + 1) as f64));
        let applied = operator.apply_preconditioner(&probe).unwrap();
        let expected_inverse = expected.cholesky().unwrap().inverse();
        let expected_pose0 =
            expected_inverse * Vector6::from_iterator((0..6).map(|i| (i + 1) as f64));
        assert!((applied.fixed_rows::<6>(0).into_owned() - expected_pose0).norm() < 1.0e-12);
    }

    #[test]
    fn cluster8_matches_principal_schur_with_repeated_sensor_rows() {
        for lambda in [0.0, 0.5] {
            let system = synthetic_system();
            let old = ImplicitSchurOperator::new(&system, lambda).unwrap();
            let clustered = ImplicitSchurOperator::new(&system, lambda)
                .unwrap()
                .with_cluster8()
                .unwrap();
            let schur = explicit_schur(&system, lambda);
            let probe = DVector::from_fn(12, |i, _| (i + 1) as f64 / 13.0);
            assert_eq!(old.rhs(), clustered.rhs());
            assert_eq!(old.apply(&probe).unwrap(), clustered.apply(&probe).unwrap());
            let applied = clustered.apply_preconditioner(&probe).unwrap();
            assert!((&schur * &applied - &probe).norm() < 1e-11);
            let result = clustered
                .solve_pcg(clustered.rhs(), PcgOptions::default())
                .unwrap();
            assert!(result.residual_norm <= result.target);
            assert!((&schur * result.solution - clustered.rhs()).norm() < 1e-10);
        }
    }

    #[test]
    fn cluster8_storage_is_bounded_for_a_long_track_and_partial_cluster() {
        let poses = 17;
        let mut system = synthetic_system();
        system.h_pp = CameraHessian::PoseDiagonal(vec![Matrix6::identity() * 100.0; poses]);
        system.b_p = DVector::from_element(poses * 6, 1.0);
        let cross = system.landmarks[0].cross[0].1 * 0.01;
        system.landmarks[0].cross = (0..poses).flat_map(|p| [(p, cross), (p, cross)]).collect();
        let clustered = ImplicitSchurOperator::new(&system, 0.5)
            .unwrap()
            .with_cluster8()
            .unwrap();
        let blocks = clustered.cluster_inverse.as_ref().unwrap();
        assert_eq!(
            blocks.iter().map(|m| m.nrows()).collect::<Vec<_>>(),
            vec![48, 48, 6]
        );
        assert!(blocks.iter().map(|m| m.len()).sum::<usize>() <= poses * 8 * 36);
        let schur = explicit_schur(&system, 0.5);
        let probe = DVector::from_element(poses * 6, 1.0);
        let applied = clustered.apply_preconditioner(&probe).unwrap();
        let mut offset = 0;
        for inverse in blocks {
            let n = inverse.nrows();
            let principal = schur.view((offset, offset), (n, n));
            assert!((principal * applied.rows(offset, n) - probe.rows(offset, n)).norm() < 1e-10);
            offset += n;
        }
    }

    #[test]
    fn cluster8_rejects_an_indefinite_cluster_without_diagonal_fallback() {
        let mut system = synthetic_system();
        system.h_pp = CameraHessian::PoseDiagonal(vec![Matrix6::identity(); 2]);
        system.landmarks[0].h_ll = Matrix3::identity();
        let cross = Matrix6x3::from_fn(|r, c| if r == c { 0.8 } else { 0.0 });
        system.landmarks[0].cross = vec![(0, cross), (1, cross)];
        let diagonal = ImplicitSchurOperator::new(&system, 0.0).unwrap();
        assert!(matches!(
            diagonal.with_cluster8(),
            Err(ImplicitSchurError::NonSpdPreconditioner(0))
        ));
    }

    #[test]
    fn cluster8_restart_shares_budget_and_checks_true_residual() {
        let mut system = synthetic_system();
        system.h_pp = CameraHessian::PoseDiagonal(vec![Matrix6::identity() * 100.0; 17]);
        system.b_p = DVector::from_fn(102, |i, _| (i + 1) as f64);
        let cross = system.landmarks[0].cross[0].1 * 0.1;
        system.landmarks[0].cross = (0..17).map(|p| (p, cross)).collect();
        let op = ImplicitSchurOperator::new(&system, 0.5)
            .unwrap()
            .with_cluster8()
            .unwrap();
        let options = PcgOptions {
            max_iterations: 128,
            relative_tolerance: 1e-12,
            absolute_tolerance: 1e-12,
        };
        let plain = op.solve_pcg(op.rhs(), options).unwrap();
        let zero = op.solve_pcg_with_restart(op.rhs(), options, 0).unwrap();
        assert_eq!(plain, zero.result);
        let run = op
            .solve_pcg_with_injected_recursive_residual_for_test(op.rhs(), options, 1)
            .unwrap();
        assert_eq!(run.diagnostics.restarts, 1);
        assert!(run.diagnostics.pcg_iterations.unwrap() <= 128);
        assert!(run.result.residual_norm <= run.result.target);
        let capped = op
            .solve_pcg_with_injected_recursive_residual_for_test(
                op.rhs(),
                PcgOptions {
                    max_iterations: 1,
                    ..options
                },
                1,
            )
            .unwrap_err();
        assert_eq!(capped.diagnostics.pcg_iterations, Some(1));
        assert_eq!(capped.diagnostics.restarts, 0);
    }

    #[test]
    fn hand_check_fixture_matches_schur_and_full_normal_for_both_damping_values() {
        for lambda in [0.0, 0.5] {
            let system = hand_check_system();
            let operator = ImplicitSchurOperator::new(&system, lambda).unwrap();
            let schur = explicit_schur(&system, lambda);
            let expected_pose0 = if lambda == 0.0 { 7.75 } else { 8.5 };
            let expected_pose1 = if lambda == 0.0 {
                9.75
            } else {
                10.277777777777779
            };
            let expected_cross = if lambda == 0.0 {
                0.75
            } else {
                0.6666666666666666
            };
            for component in 0..3 {
                assert!((schur[(component, component)] - expected_pose0).abs() < 1.0e-12);
                assert!((schur[(6 + component, 6 + component)] - expected_pose1).abs() < 1.0e-12);
                assert!((schur[(component, 6 + component)] - expected_cross).abs() < 1.0e-12);
                assert!((schur[(6 + component, component)] - expected_cross).abs() < 1.0e-12);
            }
            for component in 3..6 {
                assert!((schur[(component, component)] - (10.0 + lambda)).abs() < 1.0e-12);
                assert!((schur[(6 + component, 6 + component)] - (10.0 + lambda)).abs() < 1.0e-12);
                assert_eq!(schur[(component, 6 + component)], 0.0);
            }

            let expected_rhs0 = if lambda == 0.0 {
                [-0.25, -0.5, -0.75]
            } else {
                [-1.0 / 3.0, -2.0 / 3.0, -1.0]
            };
            let expected_rhs1 = if lambda == 0.0 {
                [-7.25, -8.5, -9.75]
            } else {
                [-65.0 / 9.0, -76.0 / 9.0, -29.0 / 3.0]
            };
            for component in 0..3 {
                assert!((operator.rhs[component] - expected_rhs0[component]).abs() < 1.0e-12);
                assert!((operator.rhs[6 + component] - expected_rhs1[component]).abs() < 1.0e-12);
            }

            let (normal, full_rhs) = full_normal_system(&system, lambda);
            let full_delta = solve_normal_equations(&normal, &full_rhs).unwrap();
            let mut direct_system = hand_check_system();
            let direct = solve_step(
                &mut direct_system,
                2,
                1,
                0,
                0,
                lambda,
                LinearSolver::Sparse,
                false,
                &mut None,
            )
            .unwrap();
            assert!((direct.0.clone() - full_delta.rows(0, 12)).norm() < 1.0e-10);
            assert!((direct.1.clone() - full_delta.rows(12, 3)).norm() < 1.0e-10);
            let result = operator
                .solve_pcg(operator.rhs(), PcgOptions::default())
                .unwrap();
            assert!((result.solution.clone() - full_delta.rows(0, 12)).norm() < 1.0e-9);
            assert!(
                (operator.complete_delta(&result.solution).unwrap() - full_delta.rows(12, 3))
                    .norm()
                    < 1.0e-9
            );
        }
    }

    #[test]
    fn implicit_operator_matches_explicit_schur_and_direct_step_without_damping() {
        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
        let explicit = explicit_schur(&system, 0.0);
        let x = DVector::from_iterator(12, (0..12).map(|index| 0.02 * (index + 2) as f64));
        assert!((operator.apply(&x).unwrap() - explicit * &x).norm() < 1.0e-10);

        let mut direct_system = synthetic_system();
        let direct = solve_step(
            &mut direct_system,
            2,
            1,
            0,
            0,
            0.0,
            LinearSolver::Sparse,
            false,
            &mut None,
        )
        .unwrap();
        let result = operator
            .solve_pcg(operator.rhs(), PcgOptions::default())
            .unwrap();
        assert!((&result.solution - &direct.0).norm() < 1.0e-9);
        assert!((operator.complete_delta(&result.solution).unwrap() - direct.1).norm() < 1.0e-9);
    }

    #[test]
    fn rig_sensor_observations_share_one_pose_slot_and_keep_cross_terms() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.add_pose(0, Pose::identity());
        ba.add_landmark(0, Point3::new(0.3, -0.2, 4.0));
        ba.add_rig_observation(BaRigObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy: Point2::new(357.5, 215.0),
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        });
        ba.add_rig_observation(BaRigObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy: Point2::new(382.0, 215.0),
            camera,
            sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0)),
        });
        let pose_index = BTreeMap::from([(0_u64, 0_usize)]);
        let landmark_index = BTreeMap::from([(0_u64, 0_usize)]);
        let system = build_normal_equations(
            &ba,
            &ba.camera.intrinsics().unwrap(),
            &pose_index,
            &landmark_index,
            &BTreeMap::new(),
            &BTreeMap::new(),
            &RobustKernel::None,
            None,
            false,
            true,
        );
        assert_eq!(system.landmarks[0].cross.len(), 2);
        assert_eq!(
            system.landmarks[0]
                .cross
                .iter()
                .map(|(pose, _)| *pose)
                .collect::<Vec<_>>(),
            vec![0, 0]
        );
        let operator = ImplicitSchurOperator::new(&system, 0.1).unwrap();
        let x = DVector::from_element(6, 0.2);
        let expected = explicit_schur(&system, 0.1) * &x;
        let actual = operator.apply(&x).unwrap();
        // Explicit and implicit forms reassociate cross-sensor products.  The
        // rounding scale is the sum of the unreduced Bx term and the
        // eliminated E C⁻¹ Eᵀx term, since their subtraction can cancel to a
        // much smaller Schur result.  This is scale-aware rather than a
        // pixel-constant absolute threshold and follows camera/rig scaling.
        let CameraHessian::PoseDiagonal(diagonal) = &system.h_pp else {
            panic!("rig matvec test requires pose blocks");
        };
        let mut base = DVector::zeros(x.len());
        for (pose, block) in diagonal.iter().enumerate() {
            let mut damped = *block;
            for component in 0..6 {
                damped[(component, component)] += 0.1;
            }
            let value = damped * x.fixed_rows::<6>(pose * 6).into_owned();
            for component in 0..6 {
                base[pose * 6 + component] = value[component];
            }
        }
        // Sum the magnitudes of the individual E C⁻¹ Eᵀx pair products, not
        // only their potentially-cancelled aggregate.  This captures the
        // arithmetic scale of the two evaluation orders.
        let inverse = (system.landmarks[0].h_ll + 0.1 * Matrix3::<f64>::identity())
            .try_inverse()
            .unwrap();
        let mut eliminated_term_scale = 0.0;
        for (_pose, a) in &system.landmarks[0].cross {
            for (other_pose, b) in &system.landmarks[0].cross {
                let x_other: Vector6<f64> = x.fixed_rows::<6>(other_pose * 6).into_owned();
                let term: Vector6<f64> = a * inverse * b.transpose() * x_other;
                eliminated_term_scale += term.norm();
            }
        }
        let scale =
            (base.norm() + eliminated_term_scale + actual.norm() + expected.norm()).max(1.0);
        let tolerance = 64.0 * f64::EPSILON * scale;
        let difference = (&actual - &expected).norm();
        assert!(
            difference <= tolerance,
            "rig matvec diff={} scale={} tolerance={}",
            difference,
            scale,
            tolerance
        );
    }

    #[test]
    fn fixed_rotation_is_preserved_by_operator_and_back_substitution() {
        let mut system = synthetic_system();
        let pose_index = BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]);
        let fixed = BTreeSet::from([10_u64]);
        constrain_fixed_pose_rotations(&fixed, &pose_index, &mut system);
        let operator = ImplicitSchurOperator::new(&system, 0.5).unwrap();
        assert_eq!(operator.rhs[3], 0.0);
        assert_eq!(operator.rhs[4], 0.0);
        assert_eq!(operator.rhs[5], 0.0);
        let mut x = DVector::zeros(12);
        x[3] = 1.0;
        x[4] = -2.0;
        x[5] = 3.0;
        let applied = operator.apply(&x).unwrap();
        assert_eq!(applied[3], 1.5);
        assert_eq!(applied[4], -3.0);
        assert_eq!(applied[5], 4.5);
        let result = operator
            .solve_pcg(operator.rhs(), PcgOptions::default())
            .unwrap();
        assert_eq!(result.solution[3], 0.0);
        assert_eq!(result.solution[4], 0.0);
        assert_eq!(result.solution[5], 0.0);
    }

    #[test]
    fn operator_rejects_dense_dimension_and_nonfinite_inputs() {
        let mut system = synthetic_system();
        let dimension = system.b_p.len();
        system.h_pp = CameraHessian::Dense(DMatrix::identity(dimension, dimension));
        assert!(matches!(
            ImplicitSchurOperator::new(&system, 0.0),
            Err(ImplicitSchurError::DenseCameraHessian)
        ));

        let mut system = synthetic_system();
        system.b_p = DVector::zeros(1);
        assert!(matches!(
            ImplicitSchurOperator::new(&system, 0.0),
            Err(ImplicitSchurError::DimensionMismatch {
                expected: 12,
                actual: 1,
            })
        ));

        let mut system = synthetic_system();
        system.landmarks[0].cross[0].1[(0, 0)] = f64::NAN;
        assert!(matches!(
            ImplicitSchurOperator::new(&system, 0.0),
            Err(ImplicitSchurError::NonFinite("landmark cross block"))
        ));

        let mut system = synthetic_system();
        system.landmarks[0].cross[0].0 = usize::MAX;
        assert!(matches!(
            ImplicitSchurOperator::new(&system, 0.0),
            Err(ImplicitSchurError::DimensionMismatch {
                expected: 2,
                actual: usize::MAX,
            })
        ));

        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
        let mut nonfinite = DVector::zeros(operator.dimension());
        nonfinite[0] = f64::NAN;
        assert_eq!(
            operator.apply(&nonfinite),
            Err(ImplicitSchurError::NonFinite("operator input"))
        );

        let system = synthetic_system();
        assert!(matches!(
            ImplicitSchurOperator::new(&system, f64::NAN),
            Err(ImplicitSchurError::InvalidLambda)
        ));
        let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
        assert!(matches!(
            operator.solve_pcg(
                operator.rhs(),
                PcgOptions {
                    relative_tolerance: -1.0,
                    ..PcgOptions::default()
                }
            ),
            Err(ImplicitSchurError::InvalidTolerance)
        ));
        let mut huge_rhs = operator.rhs().clone();
        huge_rhs[0] = f64::MAX;
        assert!(matches!(
            operator.solve_pcg(
                &huge_rhs,
                PcgOptions {
                    relative_tolerance: 2.0,
                    ..PcgOptions::default()
                }
            ),
            Err(ImplicitSchurError::NonFinite("PCG target"))
        ));
        let invalid_restart = operator
            .solve_pcg_with_restart(
                operator.rhs(),
                PcgOptions {
                    relative_tolerance: f64::NAN,
                    ..PcgOptions::default()
                },
                1,
            )
            .unwrap_err();
        assert!(matches!(
            invalid_restart.error,
            ImplicitSchurError::InvalidTolerance
        ));
        assert_eq!(invalid_restart.diagnostics.pcg_iterations, None);
    }

    #[test]
    fn empty_pose_operator_rejects_pose_reduced_system() {
        let empty = NormalEquationsBa {
            h_pp: CameraHessian::PoseDiagonal(Vec::new()),
            b_p: DVector::zeros(0),
            landmarks: Vec::new(),
        };
        assert!(matches!(
            ImplicitSchurOperator::new(&empty, 0.0),
            Err(ImplicitSchurError::EmptySystem)
        ));
        // This only checks the prototype's empty pose-reduced input.  It is
        // not a production all-fixed-pose solver test: solve_step has a
        // separate p_count == 0 landmark-only branch.
    }

    #[test]
    fn singular_landmark_inverse_is_skipped_like_direct_solver() {
        let mut system = synthetic_system();
        system.landmarks[0].h_ll = Matrix3::zeros();
        let operator = ImplicitSchurOperator::new(&system, 0.0).unwrap();
        assert!(operator.h_ll_inverse(0).is_none());
        let expected = DVector::from_iterator(12, system.b_p.iter().map(|value| -value));
        assert_eq!(operator.rhs.as_slice(), expected.as_slice());
        let delta_landmarks = operator.complete_delta(&DVector::zeros(12)).unwrap();
        assert!(delta_landmarks.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn pcg_reports_limits_curvature_and_zero_rhs_without_gauge_claim() {
        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
        let zero = DVector::zeros(operator.dimension());
        let zero_result = operator
            .solve_pcg(
                &zero,
                PcgOptions {
                    max_iterations: 0,
                    ..PcgOptions::default()
                },
            )
            .unwrap();
        assert_eq!(zero_result.iterations, 0);
        assert_eq!(zero_result.residual_norm, 0.0);
        let zero_restart = operator
            .solve_pcg_with_restart(
                &zero,
                PcgOptions {
                    max_iterations: 0,
                    ..PcgOptions::default()
                },
                1,
            )
            .unwrap();
        assert_eq!(zero_restart.diagnostics.pcg_iterations, Some(0));
        assert_eq!(zero_restart.diagnostics.restarts, 0);

        let limited = operator.solve_pcg(
            operator.rhs(),
            PcgOptions {
                max_iterations: 0,
                ..PcgOptions::default()
            },
        );
        assert!(matches!(
            limited,
            Err(ImplicitSchurError::MaxIterations { .. })
        ));
        let limited_restart = operator
            .solve_pcg_with_restart(
                operator.rhs(),
                PcgOptions {
                    max_iterations: 0,
                    ..PcgOptions::default()
                },
                1,
            )
            .unwrap_err();
        assert!(matches!(
            limited_restart.error,
            ImplicitSchurError::MaxIterations { .. }
        ));
        assert_eq!(limited_restart.diagnostics.pcg_iterations, Some(0));

        let bad_solution = DVector::zeros(operator.dimension());
        assert!(matches!(
            operator.checked_result(operator.rhs(), bad_solution, 0, 0.0, 0.0),
            Err(ImplicitSchurError::ResidualCheckFailed { .. })
        ));

        let mut indefinite = synthetic_system();
        if let CameraHessian::PoseDiagonal(diagonal) = &mut indefinite.h_pp {
            for block in diagonal {
                *block = Matrix6::from_diagonal(&Vector6::from_element(5.0));
            }
        }
        indefinite.landmarks[0].h_ll = Matrix3::identity();
        indefinite.landmarks[0].cross = vec![
            (
                0,
                Matrix6x3::from_fn(
                    |row, column| {
                        if row == 0 && column == 0 {
                            2.0
                        } else {
                            0.0
                        }
                    },
                ),
            ),
            (
                1,
                Matrix6x3::from_fn(
                    |row, column| {
                        if row == 0 && column == 0 {
                            2.0
                        } else {
                            0.0
                        }
                    },
                ),
            ),
        ];
        let indefinite_operator = ImplicitSchurOperator::new(&indefinite, 0.0).unwrap();
        let mut negative_eigenvector = DVector::zeros(indefinite_operator.dimension());
        negative_eigenvector[0] = 1.0;
        negative_eigenvector[6] = 1.0;
        assert!(matches!(
            indefinite_operator.solve_pcg(&negative_eigenvector, PcgOptions::default()),
            Err(ImplicitSchurError::NonPositiveCurvature)
        ));
        let curvature_restart = indefinite_operator
            .solve_pcg_with_restart(&negative_eigenvector, PcgOptions::default(), 1)
            .unwrap_err();
        assert!(matches!(
            curvature_restart.error,
            ImplicitSchurError::NonPositiveCurvature
        ));
        assert_eq!(curvature_restart.diagnostics.pcg_iterations, Some(0));
        assert_eq!(curvature_restart.diagnostics.restarts, 0);
    }

    #[test]
    fn injected_recursive_residual_gap_restarts_once_without_resetting_budget() {
        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
        let options = PcgOptions {
            max_iterations: 32,
            // The test-only hook reports a deterministic zero
            // recursive residual after the first alpha update.  The
            // following honest b-Ax check forces restart.
            relative_tolerance: 1.0e-12,
            absolute_tolerance: 1.0e-12,
        };
        let no_restart = operator
            .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 0)
            .unwrap_err();
        assert!(matches!(
            no_restart.error,
            ImplicitSchurError::ResidualCheckFailed { iterations: 1, .. }
        ));
        assert_eq!(no_restart.diagnostics.pcg_iterations, Some(1));
        assert_eq!(no_restart.diagnostics.restarts, 0);

        let mut capped_options = options;
        capped_options.max_iterations = 1;
        let capped = operator
            .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), capped_options, 1)
            .unwrap_err();
        assert!(matches!(
            capped.error,
            ImplicitSchurError::ResidualCheckFailed { iterations: 1, .. }
        ));
        assert_eq!(capped.diagnostics.pcg_iterations, Some(1));
        assert_eq!(capped.diagnostics.restarts, 0);

        let run = operator
            .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 1)
            .expect("one injected recursive residual gap should be recoverable");
        assert_eq!(run.diagnostics.restarts, 1);
        assert_eq!(run.diagnostics.failed_true_residual_rechecks, 1);
        assert!(run.diagnostics.true_residual_rechecks >= 2);
        assert!(run.diagnostics.pcg_iterations.unwrap() <= 32);
        assert!(run.result.residual_norm <= run.result.target);

        let repeat = operator
            .solve_pcg_with_injected_recursive_residual_for_test(operator.rhs(), options, 1)
            .expect("the injected restart must be deterministic");
        assert_eq!(run, repeat);
    }

    #[test]
    fn repeated_pcg_runs_are_deterministic() {
        let system = synthetic_system();
        let operator = ImplicitSchurOperator::new(&system, 0.25).unwrap();
        let first = operator
            .solve_pcg(operator.rhs(), PcgOptions::default())
            .unwrap();
        let second = operator
            .solve_pcg(operator.rhs(), PcgOptions::default())
            .unwrap();
        assert_eq!(first, second);
    }
}
