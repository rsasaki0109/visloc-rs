//! Landmark nullspace projection and the f32 reduced normal system (compact back-substitution, IMU reduction).

use super::*;

#[derive(Debug, Clone)]
pub(super) struct ReducedNormalSystemF32 {
    pub(super) h: DMatrix<f32>,
    pub(super) b: DVector<f32>,
    pub(super) back_substitution: Vec<LandmarkBackSubstitution>,
    /// Optional clean-path payload extracted from the same landmark QR that
    /// produced `projected`.  Retained/diagnostic callers leave this unset so
    /// their historical f64 payload and call graph remain unchanged.
    pub(super) compact_back_substitution: Option<CompactLandmarkBackSubstitutionBatchF32>,
    /// Clean-path model-decrease rows moved from the reducer's projected
    /// visual tuples.  This is deliberately separate from Q1 recovery: the
    /// Q2 rows are consumed only by the clean model evaluator and are absent
    /// from retained/diagnostic/F64 reductions.
    pub(super) model_decrease_payload: Option<ModelDecreasePayloadF32>,
    pub(super) imu_diagnostic: Option<ImuReductionDiagnostic>,
    pub(super) diagnostic_stages: Option<DiagnosticNormalSystemF32>,
}

/// Per-landmark state retained from the clean UpstreamF32 ABS-QR walk.  The
/// first `landmark_cols` transformed rows are the only rows needed after the
/// state solve: `Q1 * J_state`, `Q1 * r`, and the upper triangular landmark
/// block `R`.  Keeping these f32 values avoids rebuilding the visual factor
/// and running Householder QR again for every LM trial.
#[derive(Debug, Clone)]
pub(crate) struct CompactLandmarkBackSubstitutionF32 {
    pub(crate) landmark_index: usize,
    pub(crate) track_id: u64,
    /// Row-major Q1 state rows, followed by Q1 residual and upper-R values.
    ///
    /// The old representation kept three independent nalgebra objects here
    /// (and therefore three heap allocations) for every visual factor.  The
    /// values are only read by the exact compact back-substitution helper, so
    /// one flat allocation is sufficient and keeps the source values/order
    /// unchanged.  `state_cols` and `landmark_cols` describe the slices.
    pub(super) storage: Vec<f32>,
    pub(super) state_cols: usize,
    pub(super) landmark_cols: usize,
    pub(crate) rank: usize,
    /// Whether the original upstream recovery preconditions hold.  Keeping
    /// rank-deficient payloads observable lets the clean caller fail closed
    /// without confusing an ineligible block with a missing mapping.
    pub(crate) eligible: bool,
}

impl CompactLandmarkBackSubstitutionF32 {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        self.storage[row * self.state_cols + column]
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols;
        self.storage[offset + row]
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols + self.landmark_cols;
        self.storage[offset + row * self.landmark_cols + column]
    }

    #[inline]
    pub(super) fn storage_len(&self) -> Option<usize> {
        checked_compact_storage_len(self.landmark_cols, self.state_cols)
    }

    #[cfg(test)]
    pub(super) const fn q1_state_shape(&self) -> (usize, usize) {
        (self.landmark_cols, self.state_cols)
    }

    #[cfg(test)]
    pub(super) const fn q1_residual_len(&self) -> usize {
        self.landmark_cols
    }

    #[cfg(test)]
    pub(super) const fn upper_r_shape(&self) -> (usize, usize) {
        (self.landmark_cols, self.landmark_cols)
    }

    #[cfg(test)]
    fn view(&self) -> CompactLandmarkBackSubstitutionViewF32<'_> {
        CompactLandmarkBackSubstitutionViewF32 {
            storage: &self.storage,
            storage_offset: 0,
            state_cols: self.state_cols,
            landmark_cols: self.landmark_cols,
            rank: self.rank,
            eligible: self.eligible,
        }
    }
}

/// Borrowed view of compact Q1/R storage.  It can address either the
/// standalone test payload or an entry in the reducer's shared arena without
/// copying any f32 values.
pub(super) struct CompactLandmarkBackSubstitutionViewF32<'a> {
    storage: &'a [f32],
    storage_offset: usize,
    state_cols: usize,
    landmark_cols: usize,
    rank: usize,
    eligible: bool,
}

impl<'a> CompactLandmarkBackSubstitutionViewF32<'a> {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        self.storage[self.storage_offset + row * self.state_cols + column]
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols;
        self.storage[self.storage_offset + offset + row]
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        let offset = self.landmark_cols * self.state_cols + self.landmark_cols;
        self.storage[self.storage_offset + offset + row * self.landmark_cols + column]
    }

    #[inline]
    const fn q1_residual_len(&self) -> usize {
        self.landmark_cols
    }
}

pub(super) trait QrModelQ1ViewF32 {
    fn q1_state_value(&self, row: usize, column: usize) -> f32;
    fn q1_residual_value(&self, row: usize) -> f32;
    fn upper_r_value(&self, row: usize, column: usize) -> f32;
    fn landmark_cols(&self) -> usize;
    fn state_cols(&self) -> usize;
    fn rank(&self) -> usize;
    fn eligible(&self) -> bool;
}

impl QrModelQ1ViewF32 for CompactLandmarkBackSubstitutionF32 {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::q1_state_value(self, row, column)
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::q1_residual_value(self, row)
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionF32::upper_r_value(self, row, column)
    }

    #[inline]
    fn landmark_cols(&self) -> usize {
        self.landmark_cols
    }

    #[inline]
    fn state_cols(&self) -> usize {
        self.state_cols
    }

    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn eligible(&self) -> bool {
        self.eligible
    }
}

impl<'a> QrModelQ1ViewF32 for CompactLandmarkBackSubstitutionViewF32<'a> {
    #[inline]
    fn q1_state_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::q1_state_value(self, row, column)
    }

    #[inline]
    fn q1_residual_value(&self, row: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::q1_residual_value(self, row)
    }

    #[inline]
    fn upper_r_value(&self, row: usize, column: usize) -> f32 {
        CompactLandmarkBackSubstitutionViewF32::upper_r_value(self, row, column)
    }

    #[inline]
    fn landmark_cols(&self) -> usize {
        self.landmark_cols
    }

    #[inline]
    fn state_cols(&self) -> usize {
        self.state_cols
    }

    #[inline]
    fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    fn eligible(&self) -> bool {
        self.eligible
    }
}

/// Metadata for one compact payload stored in a reducer-owned arena.  The
/// arena keeps all Q1/R values for a reduction in one allocation; descriptors
/// remain small and are moved into the one-shot preparation path after the
/// state solve.
#[derive(Debug, Clone)]
pub(super) struct CompactLandmarkBackSubstitutionEntryF32 {
    pub(super) landmark_index: usize,
    pub(super) track_id: u64,
    pub(super) storage_offset: usize,
    pub(super) state_cols: usize,
    pub(super) landmark_cols: usize,
    pub(super) rank: usize,
    pub(super) eligible: bool,
}

#[derive(Debug, Clone)]
pub(super) struct CompactLandmarkBackSubstitutionBatchF32 {
    pub(super) storage: Vec<f32>,
    pub(super) entries: Vec<CompactLandmarkBackSubstitutionEntryF32>,
}

/// One visual Q2 row stack retained for the clean model-decrease pass.  The
/// matrices are moved out of the reducer's projected tuple after H/b assembly;
/// no second QR or Q2 clone is needed.  Q1/R remain in the compact arena and
/// are addressed by `compact_entry_index`.
#[derive(Debug, Clone)]
struct ModelDecreaseVisualF32 {
    factor_index: usize,
    compact_entry_index: usize,
    q2_state: DMatrix<f32>,
    q2_residual: DVector<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ModelDecreasePayloadF32 {
    visual: Vec<ModelDecreaseVisualF32>,
}

/// Move the clean reducer's projected visual Q2 allocations into the model
/// sidecar after H/b assembly.  The tuple vector is consumed here, so the
/// matrices and rhs vectors are transferred without cloning.  A complete
/// mapping is required; otherwise the caller keeps the legacy evaluator.
pub(super) fn move_model_decrease_payload(
    factors: &[WhitenedFactorRowStack],
    projected: Vec<(DMatrix<f32>, DVector<f32>, usize)>,
    compact: Option<&CompactLandmarkBackSubstitutionBatchF32>,
) -> Option<ModelDecreasePayloadF32> {
    let compact = compact?;
    // `zip` would silently truncate on a malformed producer result.  The
    // model sidecar is all-or-nothing: a count mismatch must select the
    // historical full-QR evaluator instead of mixing partial Q2 rows with
    // legacy visual factors.
    if projected.len() != factors.len() {
        return None;
    }
    let mut visual = Vec::new();
    let mut compact_entry_index = 0;
    for (factor_index, (factor, (q2_state, q2_residual, _rank))) in
        factors.iter().zip(projected).enumerate()
    {
        if factor.landmark_jacobian.ncols() == 0 {
            continue;
        }
        if factor.kind != FactorKind::Visual {
            return None;
        }
        let metadata = factor.landmark_metadata?;
        let entry = compact.entries.get(compact_entry_index)?;
        if entry.landmark_index != metadata.landmark_index
            || entry.track_id != metadata.track_id
            || entry.state_cols != factor.state_jacobian.ncols()
            || entry.landmark_cols != factor.landmark_jacobian.ncols()
        {
            return None;
        }
        visual.push(ModelDecreaseVisualF32 {
            factor_index,
            compact_entry_index,
            q2_state,
            q2_residual,
        });
        compact_entry_index += 1;
    }
    (compact_entry_index == compact.entries.len()).then_some(ModelDecreasePayloadF32 { visual })
}

impl CompactLandmarkBackSubstitutionEntryF32 {
    #[inline]
    pub(super) fn storage_len(&self) -> Option<usize> {
        checked_compact_storage_len(self.landmark_cols, self.state_cols)
    }

    #[inline]
    pub(super) fn view<'a>(
        &self,
        storage: &'a [f32],
    ) -> Option<CompactLandmarkBackSubstitutionViewF32<'a>> {
        let end = self.storage_offset.checked_add(self.storage_len()?)?;
        storage.get(self.storage_offset..end)?;
        Some(CompactLandmarkBackSubstitutionViewF32 {
            storage,
            storage_offset: self.storage_offset,
            state_cols: self.state_cols,
            landmark_cols: self.landmark_cols,
            rank: self.rank,
            eligible: self.eligible,
        })
    }
}

impl CompactLandmarkBackSubstitutionBatchF32 {
    #[cfg(test)]
    fn as_slice(&self) -> &[CompactLandmarkBackSubstitutionEntryF32] {
        &self.entries
    }
}

#[derive(Debug, Clone)]
pub(super) struct DiagnosticNormalSystemF32 {
    pub(super) visual_h: DMatrix<f32>,
    pub(super) visual_b: DVector<f32>,
    pub(super) visual_imu_h: DMatrix<f32>,
    pub(super) visual_imu_b: DVector<f32>,
    pub(super) prior_h: DMatrix<f32>,
    pub(super) prior_b: DVector<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ImuReductionDiagnostic {
    pub(super) local_blocks: Vec<ImuLocalBlockDiagnostic>,
    /// Cumulative IMU DenseAccumulator snapshots after each semantic
    /// IMU+bias pair.  These are observation-only copies used to distinguish
    /// a per-link product from the later visual/prior whole-matrix adds.
    pub(super) imu_cumulative_stages: Vec<ImuCumulativeStageDiagnostic>,
    pub(super) imu_h: DMatrix<f32>,
    pub(super) imu_b: DVector<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ImuCumulativeStageDiagnostic {
    pub(super) active_offsets: Vec<usize>,
    pub(super) imu_input_diagnostic: Option<serde_json::Value>,
    pub(super) imu_h: DMatrix<f32>,
    pub(super) imu_b: DVector<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct ImuLocalBlockDiagnostic {
    pub(super) active_offsets: Vec<usize>,
    pub(super) local_jacobian: DMatrix<f32>,
    pub(super) residual: DVector<f32>,
    pub(super) local_h: DMatrix<f32>,
    pub(super) local_b: DVector<f32>,
    /// Input-boundary stages for this semantic IMU link.  This is populated
    /// only for the diagnostic path; the normal-equation reducer never reads
    /// it while assembling H/b.
    pub(super) imu_input_diagnostic: Option<serde_json::Value>,
    pub(super) local_vs_global_h_mismatches: usize,
    pub(super) local_vs_global_b_mismatches: usize,
    /// Bit mismatches after embedding the 30x30 local product into the
    /// global state-width matrix.  This includes both the active block and
    /// the expected zero padding around it, so a native Eigen packet-boundary
    /// change is visible even when the active block happens to compare equal.
    pub(super) local_vs_global_padded_h_mismatches: usize,
    pub(super) local_vs_global_padded_b_mismatches: usize,
}

impl ReducedNormalSystemF32 {
    pub(super) fn as_f64(&self) -> ReducedNormalSystem {
        ReducedNormalSystem {
            h: DMatrix::from_fn(self.h.nrows(), self.h.ncols(), |row, col| {
                self.h[(row, col)] as f64
            }),
            b: DVector::from_iterator(self.b.len(), self.b.iter().copied().map(f64::from)),
            back_substitution: self.back_substitution.clone(),
            diagnostic_stages: self.diagnostic_stages.as_ref().map(|stages| {
                DiagnosticNormalSystem {
                    visual_h: stages.visual_h.map(f64::from),
                    visual_b: stages.visual_b.map(f64::from),
                    visual_imu_h: stages.visual_imu_h.map(f64::from),
                    visual_imu_b: stages.visual_imu_b.map(f64::from),
                    prior_h: stages.prior_h.map(f64::from),
                    prior_b: stages.prior_b.map(f64::from),
                }
            }),
        }
    }

    /// Consume the clean reducer's compact payload after the state step has
    /// been solved.  Each entry is recovered exactly once with the existing
    /// f32 GEMV/triangular schedule and retained as an opaque mapped option;
    /// the concrete WindowProblem hook is responsible for validating the
    /// topology before accepting any of these values.
    pub(super) fn into_trial_preparation(
        self,
        state: &DVector<f64>,
        state_step: &DVector<f64>,
        tolerance: f64,
    ) -> Option<LmTrialPreparation> {
        let compact = self.compact_back_substitution?;
        let CompactLandmarkBackSubstitutionBatchF32 { storage, entries } = compact;
        let landmark_steps = entries
            .into_iter()
            .map(|data| {
                let landmark_index = data.landmark_index;
                let track_id = data.track_id;
                let step = back_substitute_landmark_compact_entry_f32(
                    &data, &storage, state_step, tolerance,
                )
                .and_then(|step| {
                    (step.len() == 3 && step.iter().all(|value| value.is_finite()))
                        .then(|| Vector3::new(step[0], step[1], step[2]))
                });
                LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                }
            })
            .collect();
        Some(LmTrialPreparation {
            landmark_steps,
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(state_step),
        })
    }

    /// Borrowing twin of [`Self::into_trial_preparation`].
    ///
    /// The LM damping backtracking loop reuses one reduction across every
    /// rejected lambda attempt, so the compact payload must stay owned by the
    /// caller.  This produces the same owned per-landmark preparation as the
    /// consuming version without moving (or cloning) the shared f32 arena.
    pub(super) fn trial_preparation_ref(
        &self,
        state: &DVector<f64>,
        state_step: &DVector<f64>,
        tolerance: f64,
    ) -> Option<LmTrialPreparation> {
        let compact = self.compact_back_substitution.as_ref()?;
        let landmark_steps = compact
            .entries
            .iter()
            .map(|data| {
                let landmark_index = data.landmark_index;
                let track_id = data.track_id;
                let step = back_substitute_landmark_compact_entry_f32(
                    data,
                    &compact.storage,
                    state_step,
                    tolerance,
                )
                .and_then(|step| {
                    (step.len() == 3 && step.iter().all(|value| value.is_finite()))
                        .then(|| Vector3::new(step[0], step[1], step[2]))
                });
                LmPreparedLandmarkStep {
                    landmark_index,
                    track_id,
                    step,
                }
            })
            .collect();
        Some(LmTrialPreparation {
            landmark_steps,
            tolerance_bits: tolerance.to_bits(),
            state_fingerprint: lm_trial_vector_fingerprint(state),
            step_fingerprint: lm_trial_vector_fingerprint(state_step),
        })
    }

    /// Evaluate the complete f32 model decrease from the clean reducer's
    /// retained QR sidecar.  Plain prior/IMU/bias rows remain on their
    /// existing per-factor path; only mapped visual rows use moved Q2 data.
    /// Any topology, rank, dimension, identity, or finite-value mismatch
    /// returns `None`, allowing the caller to use the historical full-QR
    /// evaluator without changing failure semantics.
    pub(super) fn model_cost_decrease_from_payload(
        &self,
        factors: &[WhitenedFactorRowStack],
        state_step: &DVector<f64>,
        tolerance: f64,
    ) -> Option<f64> {
        let compact = self.compact_back_substitution.as_ref()?;
        let payload = self.model_decrease_payload.as_ref()?;
        let step = as_f32_vector(state_step);
        let mut decrease = 0.0_f32;
        let mut deferred_prior = Vec::new();
        let mut visual_ordinal = 0;
        let mut paired_bias_index = None;
        for (factor_index, factor) in factors.iter().enumerate() {
            if paired_bias_index == Some(factor_index) {
                paired_bias_index = None;
                continue;
            }
            if factor.state_jacobian.ncols() != step.len() {
                return None;
            }
            if factor.kind == FactorKind::Imu {
                if let Some(bias) = factors.get(factor_index + 1) {
                    if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step) {
                        decrease -= dot;
                        paired_bias_index = Some(factor_index + 1);
                        continue;
                    }
                }
            }
            if let Some((j, rhs, compact_step)) = prior_model_inputs_f32(factor, state_step) {
                deferred_prior.push(prior_model_f32(&j, &rhs, &compact_step));
                continue;
            }
            if factor.landmark_jacobian.ncols() == 0 {
                let state = as_f32_matrix(&factor.state_jacobian);
                let residual = as_f32_vector(&factor.residual);
                let increment = &state * &step;
                let contribution = -increment.dot(&(0.5_f32 * &increment + &residual));
                if factor.kind == FactorKind::Prior {
                    deferred_prior.push(contribution);
                } else {
                    decrease += contribution;
                }
                continue;
            }
            if factor.kind != FactorKind::Visual {
                return None;
            }
            let visual = payload.visual.get(visual_ordinal)?;
            if visual.factor_index != factor_index {
                return None;
            }
            let metadata = factor.landmark_metadata?;
            let entry = compact.entries.get(visual.compact_entry_index)?;
            if entry.landmark_index != metadata.landmark_index
                || entry.track_id != metadata.track_id
                || entry.state_cols != step.len()
                || entry.landmark_cols != factor.landmark_jacobian.ncols()
            {
                return None;
            }
            let q1 = entry.view(&compact.storage)?;
            let dot = qr_payload_model_dot_reuse_f32(
                &q1,
                &visual.q2_state,
                &visual.q2_residual,
                state_step,
                tolerance,
            )?;
            decrease -= dot;
            visual_ordinal += 1;
        }
        for contribution in deferred_prior {
            decrease += contribution;
        }
        if visual_ordinal != payload.visual.len()
            || visual_ordinal != compact.entries.len()
            || !decrease.is_finite()
        {
            return None;
        }
        Some(decrease as f64)
    }
}

/// Append the zero landmark-damping rows that ABS_QR allocates in every
/// landmark block.  The upstream block has `obs_rows + 3` storage rows and
/// removes only its first three Q1 rows, so its Q2 contribution has
/// `obs_rows` rows (the final three are zero directions before damping).  A
/// factor stored in Rust contains only the observation rows; omitting these
/// rows changes the QR reflector sequence and the canonical row spans.
pub(super) fn augmented_landmark_rows(
    factor: &WhitenedFactorRowStack,
) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>) {
    let landmark_columns = factor.landmark_jacobian.ncols();
    let rows = factor.rows() + landmark_columns;
    let mut state_jacobian = DMatrix::zeros(rows, factor.state_jacobian.ncols());
    let mut landmark_jacobian = DMatrix::zeros(rows, landmark_columns);
    let mut residual = DVector::zeros(rows);
    state_jacobian
        .view_mut((0, 0), factor.state_jacobian.shape())
        .copy_from(&factor.state_jacobian);
    landmark_jacobian
        .view_mut((0, 0), factor.landmark_jacobian.shape())
        .copy_from(&factor.landmark_jacobian);
    residual
        .rows_mut(0, factor.rows())
        .copy_from(&factor.residual);
    (state_jacobian, landmark_jacobian, residual)
}

/// Project each landmark's rows into the left nullspace of J_l using QR.
/// This is the ABS_QR operation: the three Q1 rows are removed from the
/// augmented storage block while the zero damping directions remain in Q2.
pub fn landmark_nullspace_projection(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
) -> (DMatrix<f64>, DVector<f64>, usize) {
    // Basalt's ABS_QR implementation applies the landmark Householder
    // reflectors to the complete `[Jp | Jl | r]` row stack and retains rows
    // after the three landmark columns.  Forming `I - Jl(JlᵀJl)⁻¹Jlᵀ`
    // appears algebraically equivalent, but squares the landmark condition
    // number and is observably different on the float f4 system.  Keep the
    // QR row operation explicit so the reduced H/b follows the upstream
    // grouped-row ordering and numerical path.
    let landmark_columns = factor.landmark_jacobian.ncols();
    if landmark_columns == 0 {
        return (factor.state_jacobian.clone(), factor.residual.clone(), 0);
    }
    let (state_jacobian, landmark_jacobian, residual) = augmented_landmark_rows(factor);
    let qr = landmark_jacobian.qr();
    let r = qr.r();
    let rank = (0..r.nrows().min(r.ncols()))
        .filter(|&i| r[(i, i)].abs() > tolerance)
        .count();

    let mut transformed_state = state_jacobian;
    let mut transformed_residual =
        DMatrix::from_column_slice(residual.len(), 1, residual.as_slice());
    qr.q_tr_mul(&mut transformed_state);
    qr.q_tr_mul(&mut transformed_residual);

    if residual.len() <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
        );
    }
    let reduced_rows = residual.len() - landmark_columns;
    (
        transformed_state
            .rows(landmark_columns, reduced_rows)
            .into_owned(),
        transformed_residual
            .rows(landmark_columns, reduced_rows)
            .column(0)
            .into_owned(),
        rank,
    )
}

pub fn reduce_landmark_factors(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> ReducedNormalSystem {
    let mut h = DMatrix::zeros(state_dof, state_dof);
    let mut b = DVector::zeros(state_dof);
    let mut back = Vec::with_capacity(factors.len());
    for f in factors {
        assert_eq!(f.state_jacobian.ncols(), state_dof);
        let (j, r, rank) = landmark_nullspace_projection(f, tolerance);
        h += j.transpose() * &j;
        b += j.transpose() * &r;
        back.push(LandmarkBackSubstitution {
            state_jacobian: f.state_jacobian.clone(),
            landmark_jacobian: f.landmark_jacobian.clone(),
            residual: f.residual.clone(),
            rank,
        });
    }
    ReducedNormalSystem {
        h,
        b,
        back_substitution: back,
        diagnostic_stages: None,
    }
}

pub(super) fn as_f32_matrix(matrix: &DMatrix<f64>) -> DMatrix<f32> {
    DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)] as f32
    })
}

pub(super) fn as_f32_vector(vector: &DVector<f64>) -> DVector<f32> {
    DVector::from_iterator(
        vector.len(),
        vector.iter().copied().map(|value| value as f32),
    )
}

pub(super) fn as_f64_vector(vector: &DVector<f32>) -> DVector<f64> {
    DVector::from_iterator(vector.len(), vector.iter().copied().map(f64::from))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ImuReductionError {
    MissingBias {
        imu_index: usize,
    },
    UnexpectedBias {
        bias_index: usize,
    },
    InvalidRows {
        index: usize,
        expected: usize,
        actual: usize,
    },
    MissingOffsets {
        index: usize,
        kind: FactorKind,
    },
    MismatchedOffsets {
        imu_index: usize,
        bias_index: usize,
    },
    InvalidOffsets {
        index: usize,
        start: usize,
        end: usize,
        state_dof: usize,
    },
    UnexpectedColumns {
        index: usize,
    },
    StateWidth {
        index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLocalProduct {
        h_rows: usize,
        h_cols: usize,
        b_len: usize,
    },
    InvalidAccumulator {
        h_rows: usize,
        h_cols: usize,
        b_len: usize,
    },
    InvalidFactorShape {
        index: usize,
        state_rows: usize,
        landmark_rows: usize,
        residual_len: usize,
    },
    InvalidProjectionCount {
        expected: usize,
        actual: usize,
    },
    VisualPrefixTraceIo,
    VisualPrefixTraceInvalid {
        index: usize,
    },
    Full70OracleIo,
    Full70OracleInvalid {
        index: usize,
    },
    Full70OracleProvenanceMissing {
        key: &'static str,
    },
}

pub(super) fn validate_factor_shapes(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    for (index, factor) in factors.iter().enumerate() {
        if factor.state_jacobian.ncols() != state_dof {
            return Err(ImuReductionError::StateWidth {
                index,
                expected: state_dof,
                actual: factor.state_jacobian.ncols(),
            });
        }
        if factor.state_jacobian.nrows() != factor.landmark_jacobian.nrows()
            || factor.residual.len() != factor.state_jacobian.nrows()
        {
            return Err(ImuReductionError::InvalidFactorShape {
                index,
                state_rows: factor.state_jacobian.nrows(),
                landmark_rows: factor.landmark_jacobian.nrows(),
                residual_len: factor.residual.len(),
            });
        }
    }
    Ok(())
}

pub(super) fn validate_imu_pairing(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    for (index, factor) in factors.iter().enumerate() {
        match factor.kind {
            FactorKind::Imu => {
                if factor.landmark_jacobian.ncols() != 0 {
                    return Err(ImuReductionError::UnexpectedColumns { index });
                }
                if factor.rows() != 9 {
                    return Err(ImuReductionError::InvalidRows {
                        index,
                        expected: 9,
                        actual: factor.rows(),
                    });
                }
                if factor.state_jacobian.ncols() != state_dof {
                    return Err(ImuReductionError::StateWidth {
                        index,
                        expected: state_dof,
                        actual: factor.state_jacobian.ncols(),
                    });
                }
                let Some(offsets) = factor.imu_link_offsets else {
                    return Err(ImuReductionError::MissingOffsets {
                        index,
                        kind: factor.kind,
                    });
                };
                validate_imu_offsets(index, offsets, state_dof)?;
                let Some(bias) = factors.get(index + 1) else {
                    return Err(ImuReductionError::MissingBias { imu_index: index });
                };
                if bias.kind != FactorKind::Bias {
                    return Err(ImuReductionError::MissingBias { imu_index: index });
                }
                if bias.landmark_jacobian.ncols() != 0 {
                    return Err(ImuReductionError::UnexpectedColumns { index: index + 1 });
                }
                if bias.rows() != 6 {
                    return Err(ImuReductionError::InvalidRows {
                        index: index + 1,
                        expected: 6,
                        actual: bias.rows(),
                    });
                }
                if bias.state_jacobian.ncols() != state_dof {
                    return Err(ImuReductionError::StateWidth {
                        index: index + 1,
                        expected: state_dof,
                        actual: bias.state_jacobian.ncols(),
                    });
                }
                if bias.imu_link_offsets != Some(offsets) {
                    return Err(ImuReductionError::MismatchedOffsets {
                        imu_index: index,
                        bias_index: index + 1,
                    });
                }
                validate_pair_columns(index, &factor.state_jacobian, offsets)?;
                validate_pair_columns(index + 1, &bias.state_jacobian, offsets)?;
            }
            FactorKind::Bias => {
                if index == 0 || factors[index - 1].kind != FactorKind::Imu {
                    return Err(ImuReductionError::UnexpectedBias { bias_index: index });
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub(super) const fn validate_imu_offsets(
    index: usize,
    offsets: ImuLinkOffsets,
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    if offsets.start == offsets.end
        || offsets.start.checked_add(AOM_NAV_DOF).is_none()
        || offsets.end.checked_add(AOM_NAV_DOF).is_none()
        || offsets.start + AOM_NAV_DOF > state_dof
        || offsets.end + AOM_NAV_DOF > state_dof
    {
        return Err(ImuReductionError::InvalidOffsets {
            index,
            start: offsets.start,
            end: offsets.end,
            state_dof,
        });
    }
    Ok(())
}

fn validate_pair_columns(
    index: usize,
    jacobian: &DMatrix<f64>,
    offsets: ImuLinkOffsets,
) -> Result<(), ImuReductionError> {
    for column in 0..jacobian.ncols() {
        let in_pair = (offsets.start..offsets.start + AOM_NAV_DOF).contains(&column)
            || (offsets.end..offsets.end + AOM_NAV_DOF).contains(&column);
        if !in_pair && (0..jacobian.nrows()).any(|row| jacobian[(row, column)] != 0.0) {
            return Err(ImuReductionError::UnexpectedColumns { index });
        }
    }
    Ok(())
}

pub(super) fn reduce_landmark_factors_f32_checked(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, true, false, true,
    )
}

/// Lean LM reduction variant which omits the f64 landmark back-substitution
/// payload.  The active f32 solve computes its model decrease directly from
/// the original row stack and never consumes this payload; retaining it would
/// only recreate diagnostic/retained-path storage.  The default checked
/// reducer above deliberately preserves its historical payload for callers
/// that do need landmark recovery or diagnostic conversion.
pub(super) fn reduce_landmark_factors_f32_checked_without_back_substitution(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, false, false, true,
    )
}

/// Clean-only reducer option used by the upcoming one-shot landmark recovery
/// wiring.  It retains no legacy f64 payload, but keeps compact Q1/R data
/// alongside the projected Q2 rows produced by one QR walk.
pub(super) fn reduce_landmark_factors_f32_checked_with_compact_back_substitution(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, false, true, true,
    )
}

/// Standalone landmark recovery re-reduces one already linearized factor.  It
/// is a numeric compatibility path, not a window visual-prefix boundary, so
/// an enabled prefix sidecar must not try to classify it as an independent
/// visual event.
pub(super) fn reduce_landmark_factors_f32_checked_without_visual_prefix_trace(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    reduce_landmark_factors_f32_checked_with_options(
        factors, state_dof, tolerance, true, false, false,
    )
}

/// Test-only bridge for the active-window integration fixture.  Keeping the
/// reducer and preparation construction in this module exercises the actual
/// producer path without widening the public API or duplicating the compact
/// recovery implementation in a Window test.
#[cfg(test)]
pub(crate) fn compact_trial_preparation_for_test(
    factors: &[WhitenedFactorRowStack],
    state: &DVector<f64>,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<LmTrialPreparation> {
    reduce_landmark_factors_f32_checked_with_compact_back_substitution(
        factors,
        state.len(),
        tolerance,
    )
    .ok()?
    .into_trial_preparation(state, state_step, tolerance)
}

/// One factor's independent projection result from the parallel pre-pass in
/// [`reduce_landmark_factors_f32_checked_with_options`]. `Visual::compact`
/// carries a fresh, factor-local arena (its `storage_offset` is `0`, as if
/// it were the first and only entry) rather than an offset into the shared
/// `compact_storage` arena, because that arena's real, cumulative offsets
/// can only be assigned once factors are visited in order again.
enum ProjectedLandmarkFactor {
    Plain {
        jacobian: DMatrix<f32>,
        residual: DVector<f32>,
        rank: usize,
    },
    Visual {
        jacobian: DMatrix<f32>,
        residual: DVector<f32>,
        rank: usize,
        compact: Option<(Vec<f32>, CompactLandmarkBackSubstitutionEntryF32)>,
        invalidates_compact_mapping: bool,
    },
}

pub(super) fn reduce_landmark_factors_f32_checked_with_options(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
    retain_back_substitution: bool,
    retain_compact_back_substitution: bool,
    trace_visual_prefix: bool,
) -> Result<ReducedNormalSystemF32, ImuReductionError> {
    validate_factor_shapes(factors, state_dof)?;
    validate_imu_pairing(factors, state_dof)?;
    let mut visual_prefix_trace = if trace_visual_prefix {
        visual_prefix_trace_writer(factors, state_dof)?
    } else {
        None
    };
    // Keep the source assembly phases separate.  Basalt's absolute QR
    // linearization first reduces all landmark blocks (the visual rows) in
    // its TBB `parallel_reduce`, then adds the IMU/bias blocks through a
    // separate `DenseAccumulator`, and only then folds that accumulator into
    // the visual result.  Treating the landmark-free rows as just another
    // item in one left-to-right loop changes the f32 rounding tree even when
    // every individual factor row is already bit exact.
    let mut visual_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut visual_b = DVector::<f32>::zeros(state_dof);
    let mut imu_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut imu_b = DVector::<f32>::zeros(state_dof);
    let diagnostic_policy = crate::vio::window::diagnostic_env_snapshot();
    let diagnostic_enabled = diagnostic_policy.diagnostic_imu_rows.is_some();
    // The normal-system sidecar is independently gated so a solver frontier
    // capture can request only the four H/b boundaries without materializing
    // the larger per-IMU input records.
    let diagnostic_stages_enabled = diagnostic_enabled
        || diagnostic_policy.diagnostic_normal_stages
        || diagnostic_policy.solver_frontier_trace.is_some()
        || diagnostic_policy.full70_factor_oracle.is_some();
    let mut local_blocks = Vec::new();
    let mut imu_cumulative_stages = Vec::new();
    let mut prior_h = DMatrix::<f32>::zeros(state_dof, state_dof);
    let mut prior_b = DVector::<f32>::zeros(state_dof);
    let first_visual = factors
        .iter()
        .position(|factor| factor.landmark_jacobian.ncols() != 0);
    let last_visual = factors
        .iter()
        .rposition(|factor| factor.landmark_jacobian.ncols() != 0);
    // Project all factors once.  Keeping these row stacks materialized lets
    // the IMU9 and bias6 entries be joined into the single 15-row `J/r` that
    // upstream `ImuBlock::add_dense_H_b` multiplies, without changing the
    // public factor list or its diagnostic row topology.  The compact option
    // takes the same path, but extracts Q1/R from the already-factored
    // Householder storage instead of factoring a visual block again.
    let (projected, compact_back_substitution) = if retain_compact_back_substitution {
        let compact_capacity = factors
            .iter()
            .filter(|factor| {
                let landmark_cols = factor.landmark_jacobian.ncols();
                landmark_cols != 0 && landmark_cols <= 3
            })
            .count();
        // Capacity arithmetic must fail closed before either arena or
        // reusable QR storage is acquired.  A saturating sum can turn an
        // impossible layout into a wrapped/oversized allocation request.
        let compact_storage_capacity = checked_compact_storage_capacity(factors, state_dof);
        let compact_capacity_valid = compact_storage_capacity.is_some();
        let mut compact_entries = if compact_capacity_valid {
            Vec::with_capacity(compact_capacity)
        } else {
            Vec::new()
        };
        let mut compact_storage = match compact_storage_capacity {
            Some(capacity) => Vec::with_capacity(capacity),
            None => Vec::new(),
        };
        let mut compact_mapping_valid = compact_capacity_valid;
        let mut seen_landmark_indices = if compact_capacity_valid {
            Vec::with_capacity(compact_capacity)
        } else {
            Vec::new()
        };
        let mut projected = Vec::with_capacity(factors.len());
        // Every factor's own projection -- QR-eliminating its landmark
        // columns (the expensive Householder walk) and, for eligible visual
        // factors, extracting the compact Q1/R payload -- depends only on
        // that one factor's own data plus `tolerance`/`compact_capacity_valid`
        // (both already fixed above), never on another factor's projection.
        // So every factor's projection runs in parallel below. A worker that
        // produces a compact payload writes it into its own fresh,
        // factor-local arena (starting at offset `0`, exactly as
        // `compact_back_substitution_into` would against an empty shared
        // arena) instead of the one growing `compact_storage` the serial
        // code used to share across all factors -- which is also why the
        // reusable-workspace parameter is no longer threaded through here:
        // capacity reuse *across* factors is inherently serial, and
        // `basalt-lm-workspace-reuse`'s own contract is "capacity-only
        // reuse" (see its Cargo.toml feature doc comment), so a fresh
        // per-factor arena/workspace cannot change any computed value, only
        // how much scratch memory ends up (re)allocated.
        //
        // The sequential fold below then rebuilds `compact_storage`,
        // `compact_entries`, `seen_landmark_indices`, and
        // `compact_mapping_valid` by walking the parallel results **in the
        // original factor order**: appending each factor-local arena to
        // `compact_storage` and shifting its entry's `storage_offset` by the
        // running length reproduces, byte-for-byte, the same
        // `compact_storage` contents and the same `storage_offset` values
        // the always-serial version produced (that fold performs the exact
        // same appends in the exact same order), and the duplicate/validity
        // bookkeeping only ever reads each factor's own classification plus
        // that running state -- never a QR result from another factor. Only
        // the side-effect-free QR/projection work feeding each append now
        // happens off the critical thread; its own arithmetic is untouched
        // (`landmark_nullspace_projection_f32_with_compact_into` and
        // `landmark_nullspace_projection_f32_with_compact` are called
        // completely unmodified below, exactly as the serial code called
        // them).
        let projected_parallel: Vec<ProjectedLandmarkFactor> = factors
            .par_iter()
            .map(|factor| {
                if factor.landmark_jacobian.ncols() == 0 {
                    let (jacobian, residual, rank) =
                        landmark_nullspace_projection_f32(factor, tolerance);
                    return ProjectedLandmarkFactor::Plain {
                        jacobian,
                        residual,
                        rank,
                    };
                }
                let metadata = (factor.kind == FactorKind::Visual)
                    .then_some(factor.landmark_metadata)
                    .flatten();
                let use_direct_visual_pack = factor.kind == FactorKind::Visual
                    && compact_capacity_valid
                    && metadata.is_some()
                    && factor.landmark_jacobian.ncols() <= 3;
                let (jacobian, residual, rank, compact) = if use_direct_visual_pack {
                    let mut local_arena = Vec::new();
                    let (jacobian, residual, rank, compact) =
                        landmark_nullspace_projection_f32_with_compact_into(
                            factor,
                            tolerance,
                            metadata,
                            &mut local_arena,
                        );
                    (
                        jacobian,
                        residual,
                        rank,
                        compact.map(|entry| (local_arena, entry)),
                    )
                } else {
                    // Keep generic/non-visual and invalid visual mappings on
                    // the historical materialized constructor.  They can
                    // still contribute their ordinary Q2 rows, but they must
                    // not enter the direct visual compact allocation path.
                    let (jacobian, residual, rank, _) =
                        landmark_nullspace_projection_f32_with_compact(factor, tolerance, None);
                    (jacobian, residual, rank, None)
                };
                ProjectedLandmarkFactor::Visual {
                    jacobian,
                    residual,
                    rank,
                    compact,
                    // The factor still contributes its ordinary Q2 rows, but
                    // a missing/non-visual identity makes the compact
                    // mapping unsafe.  The caller can observe `None` and use
                    // legacy recovery without guessing from factor position.
                    invalidates_compact_mapping: factor.kind != FactorKind::Visual
                        || metadata.is_none(),
                }
            })
            .collect();
        for result in projected_parallel {
            match result {
                ProjectedLandmarkFactor::Plain {
                    jacobian,
                    residual,
                    rank,
                } => {
                    projected.push((jacobian, residual, rank));
                }
                ProjectedLandmarkFactor::Visual {
                    jacobian,
                    residual,
                    rank,
                    compact,
                    invalidates_compact_mapping,
                } => {
                    if invalidates_compact_mapping {
                        compact_mapping_valid = false;
                    }
                    if let Some((local_arena, mut entry)) = compact {
                        entry.storage_offset += compact_storage.len();
                        compact_storage.extend_from_slice(&local_arena);
                        if seen_landmark_indices.contains(&entry.landmark_index) {
                            compact_mapping_valid = false;
                        }
                        seen_landmark_indices.push(entry.landmark_index);
                        compact_entries.push(entry);
                    } else {
                        compact_mapping_valid = false;
                    }
                    projected.push((jacobian, residual, rank));
                }
            }
        }
        let compact = compact_mapping_valid.then_some(CompactLandmarkBackSubstitutionBatchF32 {
            storage: compact_storage,
            entries: compact_entries,
        });
        (projected, compact)
    } else {
        (
            factors
                .iter()
                .map(|factor| landmark_nullspace_projection_f32(factor, tolerance))
                .collect::<Vec<_>>(),
            None,
        )
    };
    let mut back = retain_back_substitution.then(|| Vec::with_capacity(factors.len()));
    if let Some(back) = back.as_mut() {
        for (factor, (_, _, rank)) in factors.iter().zip(&projected) {
            back.push(LandmarkBackSubstitution {
                state_jacobian: factor.state_jacobian.clone(),
                landmark_jacobian: factor.landmark_jacobian.clone(),
                residual: factor.residual.clone(),
                rank: *rank,
            });
        }
    }

    // Every landmark/visual factor's H/b contribution below depends only on
    // that one factor's own already-projected `(jacobian, residual)` pair
    // (computed above, serially, into `projected`); there is no shared
    // mutable state between factors here, matching upstream Basalt's own
    // `tbb::parallel_reduce` over landmark blocks. Precomputing every
    // contribution in parallel and then folding each into `visual_h` /
    // `visual_b` **serially below, in the exact same factor-index order the
    // sequential loop always visited them in** reproduces the identical
    // `+=` sequence -- and therefore the identical f32 rounding tree --
    // as calling `eigen_visual_gram_packet_tail_f32` /
    // `accumulate_transpose_vector_f32_eigen` inline did; only *when* each
    // pure contribution is computed changes, never its value, its order of
    // use, or the two audited kernels' own internal arithmetic (both are
    // called completely unmodified below, just against a scratch
    // zero-accumulator for the vector case so the contribution can be
    // captured instead of accumulated in place).
    let visual_factor_indices: Vec<usize> = factors
        .iter()
        .enumerate()
        .filter(|(_, factor)| factor.landmark_jacobian.ncols() != 0)
        .map(|(index, _)| index)
        .collect();
    let visual_contributions: Vec<(VisualGramContribution, DVector<f32>)> = visual_factor_indices
        .par_iter()
        .map(|&index| {
            let factor = &factors[index];
            let (jacobian, residual, _) = &projected[index];
            let h = if factor.kind == FactorKind::Visual {
                eigen_visual_gram_packet_tail_sparse_f32(jacobian)
            } else {
                VisualGramContribution::Dense(jacobian.transpose() * jacobian)
            };
            let mut b = DVector::<f32>::zeros(jacobian.ncols());
            accumulate_transpose_vector_f32_eigen(&mut b, jacobian, residual, false);
            (h, b)
        })
        .collect();
    let mut visual_contribution_cursor = 0usize;

    let mut factor_index = 0;
    while factor_index < factors.len() {
        let factor = &factors[factor_index];
        let (jacobian, residual, _) = &projected[factor_index];
        if factor.landmark_jacobian.ncols() != 0 {
            // Standalone landmark back-substitution callers use Generic
            // factors.  They still follow the historical visual H/b path,
            // but are not visual-prefix records; tracing them as Visual would
            // make an enabled diagnostic sidecar fail at ordinal zero before
            // the actual window visual factors are visited.
            let prefix = if factor.kind == FactorKind::Visual {
                visual_prefix_trace
                    .as_mut()
                    .map(|writer| {
                        writer.begin_visual_prefix(
                            factor_index,
                            factor,
                            jacobian,
                            residual,
                            projected[factor_index].2,
                            &visual_h,
                            &visual_b,
                        )
                    })
                    .transpose()?
            } else {
                None
            };
            let (contribution_h, contribution_b) =
                &visual_contributions[visual_contribution_cursor];
            visual_contribution_cursor += 1;
            contribution_h.add_to(&mut visual_h);
            visual_b += contribution_b;
            if let (Some(writer), Some(prefix)) = (visual_prefix_trace.as_mut(), prefix) {
                writer.finish_visual_prefix(prefix, &visual_h, &visual_b)?;
            }
            factor_index += 1;
            continue;
        }

        // Semantic tags are authoritative.  The positional fallback exists
        // only for legacy Generic factors supplied by external callers; it
        // keeps their historical prefix/suffix behavior while ensuring that a
        // Generic nine-row prior never receives the IMU packet schedule.
        let is_imu_phase = match factor.kind {
            FactorKind::Imu | FactorKind::Bias => true,
            FactorKind::Prior => false,
            FactorKind::Visual => false,
            FactorKind::Generic => last_visual.is_some_and(|end| factor_index > end),
        };
        let (phase_h, phase_b) = if is_imu_phase {
            (&mut imu_h, &mut imu_b)
        } else if factor.kind == FactorKind::Prior
            || first_visual.is_some_and(|start| factor_index < start)
        {
            (&mut prior_h, &mut prior_b)
        } else {
            // A malformed/interleaved untagged factor is safer as a prior than
            // as an accidental IMU contribution.
            (&mut prior_h, &mut prior_b)
        };

        // A stored square-root marginal prior is compact in upstream
        // `MargLinData` (one column per retained state block).  Its normal
        // product is evaluated on that compact matrix and only then embedded
        // in the active AOM.  Do not feed the zero-padded global-width view
        // through nalgebra's generic GEMM: the padding changes Eigen's
        // packet traversal and therefore the f32 reduction tree.  The
        // optional metadata is absent for synthetic/legacy Prior factors, so
        // those callers retain the historical dynamic path below.
        if factor.kind == FactorKind::Prior {
            emit_prior_factor_input_diagnostic(
                jacobian,
                residual,
                factor.prior_state_columns.as_deref(),
            );
            accumulate_prior_gram_f32_eigen(
                phase_h,
                jacobian,
                factor.prior_state_columns.as_deref(),
            );
            accumulate_prior_transpose_vector_f32_eigen(
                phase_b,
                jacobian,
                residual,
                factor.prior_state_columns.as_deref(),
            );
            factor_index += 1;
            continue;
        }

        let exact_imu_rows = factor.kind == FactorKind::Imu
            && factor.landmark_jacobian.ncols() == 0
            && jacobian.nrows() == 9;
        if exact_imu_rows {
            // Upstream ImuBlock owns one local 15-row/30-column stack:
            // [preintegration 9 | gyro-bias 3 | accel-bias 3].  Build that
            // stack explicitly from the absolute columns carried by the
            // pair.  Reducing a zero-padded global matrix changes Eigen's
            // packet tails and is not source-equivalent.
            let Some(offsets) = factor.imu_link_offsets else {
                return Err(ImuReductionError::MissingOffsets {
                    index: factor_index,
                    kind: factor.kind,
                });
            };
            let Some((bias_jacobian, bias_residual, _)) = projected.get(factor_index + 1) else {
                return Err(ImuReductionError::MissingBias {
                    imu_index: factor_index,
                });
            };
            let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
            for (local_row, source) in jacobian.row_iter().enumerate() {
                for block in 0..2 {
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    local_jacobian
                        .view_mut((local_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                        .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
                }
            }
            for (local_row, source) in bias_jacobian.row_iter().enumerate() {
                for block in 0..2 {
                    let global_offset = if block == 0 {
                        offsets.start
                    } else {
                        offsets.end
                    };
                    local_jacobian
                        .view_mut((9 + local_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                        .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
                }
            }
            let mut local_residual = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
            local_residual.rows_mut(0, 9).copy_from(residual);
            local_residual.rows_mut(9, 6).copy_from(bias_residual);
            let (local_h, local_b) = local_imu_h_b_15x30(&local_jacobian, &local_residual);
            if diagnostic_enabled {
                // The local packet is deliberately only 30 columns wide, but
                // the comparison side must retain the source AOM width.  In
                // particular, frame-0 links can start after a pose-only
                // prefix, so indexing the local product with absolute AOM
                // offsets would be invalid.  Materialize the corresponding
                // 15-row global source stack for the diagnostic only.
                let mut global_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, state_dof);
                global_jacobian.rows_mut(0, 9).copy_from(jacobian);
                global_jacobian.rows_mut(9, 6).copy_from(bias_jacobian);
                local_blocks.push(imu_local_block_diagnostic(
                    &local_jacobian,
                    &local_residual,
                    &global_jacobian,
                    offsets,
                    factor.imu_input_diagnostic.clone(),
                ));
            }
            scatter_local_imu_h_b_15x30(&mut imu_h, &mut imu_b, &local_h, &local_b, offsets)?;
            if diagnostic_enabled {
                imu_cumulative_stages.push(ImuCumulativeStageDiagnostic {
                    active_offsets: vec![offsets.start, offsets.end],
                    imu_input_diagnostic: factor.imu_input_diagnostic.clone(),
                    imu_h: imu_h.clone(),
                    imu_b: imu_b.clone(),
                });
            }
            factor_index += 2;
        } else {
            accumulate_gram_f32_eigen(phase_h, jacobian, exact_imu_rows);
            accumulate_transpose_vector_f32_eigen(phase_b, jacobian, residual, exact_imu_rows);
            factor_index += 1;
        }
    }
    // These whole-matrix adds mirror LinearizationAbsQR::get_dense_H_b:
    // visual TBB result, then IMU DenseAccumulator, then marginal prior.  The
    // stage copies below are diagnostic-only and preserve the source order
    // needed to compare the later damping/prior boundary without changing
    // this production reduction.
    let diagnostic_imu_h = diagnostic_enabled.then(|| imu_h.clone());
    let diagnostic_imu_b = diagnostic_enabled.then(|| imu_b.clone());
    let diagnostic_stages = if diagnostic_stages_enabled {
        let mut visual_imu_h = visual_h.clone();
        visual_imu_h += &imu_h;
        let mut visual_imu_b = visual_b.clone();
        visual_imu_b += &imu_b;
        Some(DiagnosticNormalSystemF32 {
            visual_h: visual_h.clone(),
            visual_b: visual_b.clone(),
            visual_imu_h,
            visual_imu_b,
            prior_h: prior_h.clone(),
            prior_b: prior_b.clone(),
        })
    } else {
        None
    };
    if let Some(writer) = visual_prefix_trace.as_mut() {
        if writer.next_visual_ordinal != writer.expected_visual_count {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: writer.next_visual_ordinal,
            });
        }
        writer.write_stage("visual_total", &visual_h, &visual_b)?;
    }
    let mut h = visual_h;
    h += imu_h;
    let mut b = visual_b;
    b += imu_b;
    if let Some(writer) = visual_prefix_trace.as_mut() {
        writer.write_stage("imu_total", &h, &b)?;
        writer.write_prior_before(&h, &b, &prior_h, &prior_b)?;
    }
    h += prior_h;
    b += prior_b;
    if let Some(writer) = visual_prefix_trace.as_mut() {
        writer.write_stage("prior_after", &h, &b)?;
        writer.write_stage("final", &h, &b)?;
    }
    // Transfer the projected visual Q2 rows into the model-decrease sidecar
    // so the lean LM loop evaluates `model_cost_decrease_from_payload`
    // instead of re-factoring every landmark on each damping attempt. The
    // transfer is all-or-nothing: any structural mismatch yields `None`, and
    // the loop then uses the full `model_cost_decrease_f32` evaluator.
    let (compact_back_substitution, model_decrease_payload) = if retain_compact_back_substitution {
        let payload =
            move_model_decrease_payload(factors, projected, compact_back_substitution.as_ref());
        (compact_back_substitution, payload)
    } else {
        drop(projected);
        (None, None)
    };
    Ok(ReducedNormalSystemF32 {
        h,
        b,
        back_substitution: back.unwrap_or_default(),
        compact_back_substitution,
        model_decrease_payload,
        imu_diagnostic: diagnostic_enabled.then(|| ImuReductionDiagnostic {
            local_blocks,
            imu_cumulative_stages,
            imu_h: diagnostic_imu_h.expect("diagnostic IMU H snapshot"),
            imu_b: diagnostic_imu_b.expect("diagnostic IMU b snapshot"),
        }),
        diagnostic_stages,
    })
}

/// Infallible compatibility entry point for the historical f32 reducer.
/// This rollback variant keeps the old reachable wrapper; malformed factors
/// retain the historical panic behavior.
pub(super) fn reduce_landmark_factors_f32(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
    tolerance: f64,
) -> ReducedNormalSystemF32 {
    reduce_landmark_factors_f32_checked(factors, state_dof, tolerance)
        .unwrap_or_else(|error| panic!("invalid f32 IMU reduction factors: {error:?}"))
}

/// Form the absolute normal system from the already reduced Q2 rows.
///
/// This is the boundary used by Basalt's `LinearizationAbsQR::get_dense_H_b`:
/// each visual landmark has already undergone its ABS-QR/null-space step and
/// the returned rows are stacked in native Q2 order.  The input is widened
/// only by the public Rust representation; the product itself is performed
/// in f32 with the pinned Eigen AVX2 GEMM/GEMV traversal (`mr=24`, `nr=4`,
/// packet width eight).  Keeping this separate from the factor reducer is
/// important: rebuilding H/b from the original factor list observes a
/// different arithmetic boundary and is not the source MargData contract.
pub(crate) fn q2_f32_normal_system(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> (DMatrix<f64>, DVector<f64>) {
    assert_eq!(jacobian.nrows(), rhs.len());
    let jacobian_f32 = as_f32_matrix(jacobian);
    let rhs_f32 = as_f32_vector(rhs);
    let h = eigen_q2_gram_f32(&jacobian_f32);
    let b = eigen_q2_transpose_gemv_f32(&jacobian_f32, &rhs_f32);
    emit_q2_abs_hb_diagnostic(&jacobian_f32, &rhs_f32, &h, &b);
    (
        DMatrix::from_fn(h.nrows(), h.ncols(), |row, column| {
            f64::from(h[(row, column)])
        }),
        DVector::from_iterator(b.len(), b.iter().copied().map(f64::from)),
    )
}
