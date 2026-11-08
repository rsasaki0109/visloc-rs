//! Square-root marginalization: projected rows, mixed-prior marginalization
//! and the upstream f32 SqrtToSqrt paths.

use super::*;

/// Converts local landmark columns to state-only square-root rows for
/// MargData and for Schur-style prior construction.
pub fn projected_rows(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
) -> Vec<WhitenedFactorRowStack> {
    factors
        .iter()
        .filter_map(|factor| {
            let (jacobian, rhs, _) = landmark_nullspace_projection(factor, tolerance);
            WhitenedFactorRowStack::new(jacobian, DMatrix::zeros(rhs.len(), 0), rhs)
                .map(|projected| projected.with_kind(factor.kind))
        })
        .collect()
}

/// Project factor rows at the scalar boundary used by the selected upstream
/// estimator.  The pinned `LinearizationAbsQR<float>` stores each landmark
/// block as f32 before its Householder tail is copied into the global Q2
/// stack; widening a later f64 projection back to f32 changes both the tail
/// bits and the subsequent marginal QR schedule.
pub fn projected_rows_with_mode(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> Vec<WhitenedFactorRowStack> {
    factors
        .iter()
        .filter_map(|factor| {
            let (jacobian, rhs) = if scalar_mode == ScalarMode::UpstreamF32 {
                let (jacobian, rhs, _) = landmark_nullspace_projection_f32(factor, tolerance);
                (
                    DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, column| {
                        jacobian[(row, column)] as f64
                    }),
                    DVector::from_iterator(rhs.len(), rhs.iter().copied().map(f64::from)),
                )
            } else {
                let (jacobian, rhs, _) = landmark_nullspace_projection(factor, tolerance);
                (jacobian, rhs)
            };
            WhitenedFactorRowStack::new(jacobian, DMatrix::zeros(rhs.len(), 0), rhs)
                .map(|projected| projected.with_kind(factor.kind))
        })
        .collect()
}

/// Build the global Q2 row stack in the order emitted by Basalt's
/// `LinearizationAbsQR::get_dense_Q2Jp_Q2r`: selected visual landmark tails,
/// then IMU/bias rows, then the existing marginal prior.  The public factor
/// list intentionally keeps the prior first for AOM prefix semantics, so the
/// marginalization boundary must restore this source order explicitly.
pub fn projected_rows_native_q2(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> Vec<WhitenedFactorRowStack> {
    let projected = projected_rows_with_mode(factors, tolerance, scalar_mode);
    let mut ordered = Vec::with_capacity(projected.len());
    for phase in 0..3 {
        for (factor, row) in factors.iter().zip(projected.iter()) {
            let is_visual = factor.landmark_jacobian.ncols() != 0;
            let is_imu = matches!(factor.kind, FactorKind::Imu | FactorKind::Bias);
            let selected = match phase {
                0 => is_visual,
                1 => is_imu,
                _ => !is_visual && !is_imu,
            };
            if selected {
                ordered.push(row.clone());
            }
        }
    }
    ordered
}

pub fn stack_rows(
    rows: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> (DMatrix<f64>, DVector<f64>) {
    let total_rows = rows.iter().map(WhitenedFactorRowStack::rows).sum();
    let mut jacobian = DMatrix::zeros(total_rows, state_dof);
    let mut rhs = DVector::zeros(total_rows);
    let mut row = 0;
    for factor in rows {
        if factor.state_jacobian.ncols() != state_dof {
            continue;
        }
        jacobian
            .view_mut((row, 0), (factor.rows(), state_dof))
            .copy_from(&factor.state_jacobian);
        rhs.rows_mut(row, factor.rows()).copy_from(&factor.residual);
        row += factor.rows();
    }
    (jacobian, rhs)
}

pub fn marginalize_prior(
    rows: &[WhitenedFactorRowStack],
    _state_dof: usize,
    states: &[WindowState],
    dropped_ids: &[u64],
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    marginalize_mixed_prior(rows, &[], states, &[], dropped_ids, &[], state)
}

/// Marginalize a mixed Basalt AOM.  `vel_bias_ids` denotes keyframe states
/// whose six pose columns survive as a pose-only block while columns 6..14
/// leave the active navigation state.  The returned prior is ordered as
/// retained pose blocks followed by retained full navigation blocks.
pub fn marginalize_mixed_prior(
    rows: &[WhitenedFactorRowStack],
    poses: &[WindowPose],
    states: &[WindowState],
    dropped_pose_ids: &[u64],
    dropped_state_ids: &[u64],
    vel_bias_ids: &[u64],
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    marginalize_mixed_prior_with_mode(
        rows,
        poses,
        states,
        dropped_pose_ids,
        dropped_state_ids,
        vel_bias_ids,
        ScalarMode::UpstreamF32,
        state,
    )
}

/// Scalar-explicit mixed prior boundary.  The compatibility estimator uses
/// [`ScalarMode::UpstreamF32`] (the pinned `SqrtKeypointVioEstimator<float>`),
/// while the opt-in extended numerical mode retains the f64 reference path.
/// Keeping the old [`marginalize_mixed_prior`] wrapper above preserves the
/// faithful-f32 default for callers that do not carry a scalar mode.
pub fn marginalize_mixed_prior_with_mode(
    rows: &[WhitenedFactorRowStack],
    poses: &[WindowPose],
    states: &[WindowState],
    dropped_pose_ids: &[u64],
    dropped_state_ids: &[u64],
    vel_bias_ids: &[u64],
    scalar_mode: ScalarMode,
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    let state_dof = poses.len() * POSE_DOF + states.len() * NAV_STATE_DOF;
    if state.len() != state_dof {
        return None;
    }
    let projected = projected_rows_native_q2(rows, 1e-10, scalar_mode);
    let (jacobian, rhs) = stack_rows(&projected, state_dof);
    let mut keep_ids = Vec::new();
    let mut block_kinds = Vec::new();
    let mut keep_columns = Vec::new();
    let mut marginal_columns = Vec::new();

    for (index, pose) in poses.iter().enumerate() {
        let columns = (index * POSE_DOF..(index + 1) * POSE_DOF).collect::<Vec<_>>();
        if dropped_pose_ids.contains(&pose.frame_id) {
            marginal_columns.extend(columns);
        } else {
            keep_ids.push(pose.frame_id);
            block_kinds.push(WindowBlockKind::Pose);
            keep_columns.extend(columns);
        }
    }

    let state_offset = poses.len() * POSE_DOF;
    for (index, nav) in states.iter().enumerate() {
        let offset = state_offset + index * NAV_STATE_DOF;
        let all_columns = (offset..offset + NAV_STATE_DOF).collect::<Vec<_>>();
        if dropped_state_ids.contains(&nav.frame_id)
            || (vel_bias_ids.contains(&nav.frame_id) && dropped_pose_ids.contains(&nav.frame_id))
        {
            marginal_columns.extend(all_columns);
        } else if vel_bias_ids.contains(&nav.frame_id) {
            keep_ids.push(nav.frame_id);
            block_kinds.push(WindowBlockKind::Pose);
            keep_columns.extend(offset..offset + POSE_DOF);
            marginal_columns.extend(offset + POSE_DOF..offset + NAV_STATE_DOF);
        } else {
            keep_ids.push(nav.frame_id);
            block_kinds.push(WindowBlockKind::State);
            keep_columns.extend(all_columns);
        }
    }
    if keep_ids.is_empty() {
        return None;
    }
    // Marginalization is performed at the solved current state, but Basalt
    // stores the resulting prior back at each block's original FEJ point.
    // `computeDelta()` followed by `marg_data.b -= marg_data.H * delta` is the
    // source operation.  Preserve both pieces explicitly instead of silently
    // re-anchoring every prior at the latest state (which makes the reduced
    // prior error lose its negative, constant-free value).
    let mut linearized_blocks = DVector::zeros(state_dof);
    for (index, pose) in poses.iter().enumerate() {
        linearized_blocks
            .rows_mut(index * POSE_DOF, POSE_DOF)
            .copy_from(&flatten_pose_with_mode(&pose.linearized_pose, scalar_mode));
    }
    for (index, nav) in states.iter().enumerate() {
        let offset = state_offset + index * NAV_STATE_DOF;
        linearized_blocks
            .rows_mut(offset, NAV_STATE_DOF)
            .copy_from(&flatten_nav_with_mode(&nav.linearized_nav, scalar_mode));
    }
    let fej_point = DVector::from_iterator(
        keep_columns.len(),
        keep_columns.iter().map(|column| linearized_blocks[*column]),
    );
    let mut keep_delta = DVector::zeros(keep_columns.len());
    let mut keep_offset = 0;
    for (_index, pose) in poses.iter().enumerate() {
        if dropped_pose_ids.contains(&pose.frame_id) {
            continue;
        }
        keep_delta
            .rows_mut(keep_offset, POSE_DOF)
            .copy_from(&pose.linearized_delta);
        keep_offset += POSE_DOF;
    }
    for nav in states {
        if dropped_state_ids.contains(&nav.frame_id)
            || (vel_bias_ids.contains(&nav.frame_id) && dropped_pose_ids.contains(&nav.frame_id))
        {
            continue;
        }
        if vel_bias_ids.contains(&nav.frame_id) {
            keep_delta
                .rows_mut(keep_offset, POSE_DOF)
                .copy_from(&nav.linearized_delta.rows(0, POSE_DOF));
            keep_offset += POSE_DOF;
        } else {
            keep_delta
                .rows_mut(keep_offset, NAV_STATE_DOF)
                .copy_from(&nav.linearized_delta);
            keep_offset += NAV_STATE_DOF;
        }
    }
    debug_assert_eq!(keep_offset, keep_columns.len());
    let prior = match scalar_mode {
        ScalarMode::UpstreamF32 => sqrt_marginalize_upstream_f32(
            &jacobian,
            &rhs,
            &keep_columns,
            &marginal_columns,
            fej_point,
        )?,
        ScalarMode::ExtendedF64 => {
            sqrt_marginalize_upstream(&jacobian, &rhs, &keep_columns, &marginal_columns, fej_point)?
        }
    };
    if scalar_mode == ScalarMode::UpstreamF32 {
        remember_diagnostic_prior_pre(&WindowPrior {
            frame_ids: keep_ids.clone(),
            block_kinds: block_kinds.clone(),
            jacobian: prior.jacobian.clone(),
            rhs: prior.rhs.clone(),
            fej_point: prior.fej_point.clone(),
        });
    }
    let adjusted_rhs = if scalar_mode == ScalarMode::UpstreamF32 {
        let jacobian = DMatrix::from_fn(
            prior.jacobian.nrows(),
            prior.jacobian.ncols(),
            |row, col| prior.jacobian[(row, col)] as f32,
        );
        let delta = DVector::from_iterator(
            keep_delta.len(),
            keep_delta.iter().map(|value| *value as f32),
        );
        let stored_rhs =
            DVector::from_iterator(prior.rhs.len(), prior.rhs.iter().map(|value| *value as f32));
        let rhs = if jacobian.nrows() == 21
            && jacobian.ncols() == 21
            && stored_rhs.len() == 21
            && delta.len() == 21
        {
            WindowProblem::prior_reanchor_rhs_col_major_21_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (27, 27) && stored_rhs.len() == 27 && delta.len() == 27 {
            WindowProblem::prior_reanchor_rhs_col_major_27_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (33, 33) && stored_rhs.len() == 33 && delta.len() == 33 {
            WindowProblem::prior_reanchor_rhs_col_major_33_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (39, 39) && stored_rhs.len() == 39 && delta.len() == 39 {
            WindowProblem::prior_reanchor_rhs_col_major_39_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (45, 45) && stored_rhs.len() == 45 && delta.len() == 45 {
            WindowProblem::prior_reanchor_rhs_col_major_45_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (51, 51) && stored_rhs.len() == 51 && delta.len() == 51 {
            WindowProblem::prior_reanchor_rhs_col_major_51_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (57, 57) && stored_rhs.len() == 57 && delta.len() == 57 {
            WindowProblem::prior_reanchor_rhs_col_major_57_f32(&jacobian, &stored_rhs, &delta)
        } else {
            stored_rhs - jacobian * delta
        };
        DVector::from_iterator(rhs.len(), rhs.iter().copied().map(f64::from))
    } else {
        prior.rhs.clone() - &prior.jacobian * keep_delta.clone()
    };

    // One-shot marginalization boundary capture for the detached native
    // Q2Jp/Q2r audit.  This serializes only already-computed f32 casts and
    // never participates in the production arithmetic.
    if scalar_mode == ScalarMode::UpstreamF32 {
        if let Some(path) = diagnostic_env_snapshot().marg_trace.as_ref() {
            static CAPTURED: OnceLock<()> = OnceLock::new();
            // Diagnostic-only event selector for auditing later marginalization
            // boundaries.  The default remains the historical one-shot capture;
            // when VISLOC_BASALT_MARG_TRACE_EVENT=N is set, count calls in this
            // process and serialize only event N.  This never participates in
            // the production arithmetic.
            static EVENT_COUNT: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let event_ordinal = EVENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let selected_event = diagnostic_env_snapshot().marg_trace_event;
            let capture_this_event = match selected_event {
                Some(wanted) => event_ordinal == wanted,
                None => CAPTURED.set(()).is_ok(),
            };
            if capture_this_event {
                let matrix_bits = |matrix: &DMatrix<f64>| {
                    json!({
                        "rows": matrix.nrows(),
                        "cols": matrix.ncols(),
                        "layout": "column_major",
                        "bits": matrix
                            .iter()
                            .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                            .collect::<Vec<_>>(),
                    })
                };
                let vector_bits = |vector: &DVector<f64>| {
                    json!(vector
                        .iter()
                        .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                        .collect::<Vec<_>>())
                };
                // Keep the Q2 residual source order visible in the detached
                // boundary capture.  This is diagnostic-only metadata: the
                // stack has already been built above, so recording the
                // producer cannot affect the production arithmetic.
                let mut q2_sources = Vec::with_capacity(rhs.len());
                let mut q2_row = 0usize;
                let projected_by_factor = projected_rows_with_mode(rows, 1e-10, scalar_mode);
                // The remaining Q2r frontier is the second IMU link (the
                // sixth factor in the pre-marginalization row list).  When
                // the opt-in IMU audit is enabled, retain exactly that
                // factor's already-materialized source payload here so raw,
                // whitener, and whitened bits can be compared without
                // rebuilding a second arithmetic path.
                let q2_imu_factor6_input =
                    if diagnostic_env_snapshot().diagnostic_imu_rows.is_some() {
                        rows.get(6)
                            .filter(|factor| matches!(factor.kind, FactorKind::Imu))
                            .and_then(|factor| factor.imu_input_diagnostic.clone())
                    } else {
                        None
                    };
                for phase in 0..3 {
                    for (factor_index, factor) in rows.iter().enumerate() {
                        let is_visual = factor.landmark_jacobian.ncols() != 0;
                        let is_imu = matches!(factor.kind, FactorKind::Imu | FactorKind::Bias);
                        let selected = match phase {
                            0 => is_visual,
                            1 => is_imu,
                            _ => !is_visual && !is_imu,
                        };
                        if !selected {
                            continue;
                        }
                        let projected_factor = &projected_by_factor[factor_index];
                        for local_row in 0..projected_factor.residual.len() {
                            q2_sources.push(json!({
                                "q2_row": q2_row,
                                "phase": phase,
                                "factor_index": factor_index,
                                "kind": format!("{:?}", factor.kind),
                                "local_row": local_row,
                                "projected_rhs_f32": format!(
                                    "{:08x}",
                                    (projected_factor.residual[local_row] as f32).to_bits()
                                ),
                                "raw_residual_f32": factor
                                    .residual
                                    .get(local_row)
                                    .map(|value| format!("{:08x}", (*value as f32).to_bits())),
                            }));
                            q2_row += 1;
                        }
                    }
                }
                let payload = json!({
                    "schema": "visloc.m7im15.rust_marg_boundary.v1",
                    "event_ordinal": event_ordinal,
                    "keep_columns": keep_columns,
                    "marginal_columns": marginal_columns,
                    "q2_jacobian_f32": matrix_bits(&jacobian),
                    "q2_rhs_f32": vector_bits(&rhs),
                    "q2_rhs_sources": q2_sources,
                    "q2_imu_factor6_input": q2_imu_factor6_input,
                    "fej_point_f32": vector_bits(&prior.fej_point),
                    "keep_delta_f32": vector_bits(&keep_delta),
                    "prior_jacobian_f32": matrix_bits(&prior.jacobian),
                    "prior_rhs_before_delta_f32": vector_bits(&prior.rhs),
                    "prior_rhs_after_delta_f32": vector_bits(&adjusted_rhs),
                });
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)
                {
                    let _ = serde_json::to_writer_pretty(&mut file, &payload);
                    let _ = file.write_all(b"\n");
                }
            }
        }
    }
    Some(WindowPrior {
        frame_ids: keep_ids,
        block_kinds,
        jacobian: prior.jacobian,
        rhs: adjusted_rhs,
        fej_point: prior.fej_point,
    })
}

/// Basalt `MargHelper::marginalizeHelperSqrtToSqrt` in the pinned upstream
/// revision.  The input rows are already square-root/QR rows (including the
/// rows retained after each visual landmark's ABS_QR); no normal matrix is
/// formed here.  Columns are permuted to `[marg | keep]`, then a
/// rank-revealing Householder walk eliminates the leaving block.  Basalt's
/// cutoff is `sqrt(numeric_limits<Scalar>::epsilon())`, not the tighter
/// tolerance used by the landmark QR.
pub(super) fn sqrt_marginalize_upstream(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    let rows = jacobian.nrows();
    let cols = jacobian.ncols();
    if rows == 0
        || rhs.len() != rows
        || keep_columns.len() + marginal_columns.len() != cols
        || fej_point.len() != keep_columns.len()
    {
        return None;
    }

    // Upstream receives sets, so every index is unique and the two sets form
    // a complete partition.  Keep the same contract at this Rust boundary;
    // silently omitting a column would otherwise change the prior dimension.
    let mut seen = vec![false; cols];
    for &column in keep_columns.iter().chain(marginal_columns.iter()) {
        if column >= cols || seen[column] {
            return None;
        }
        seen[column] = true;
    }
    if seen.iter().any(|present| !present) {
        return None;
    }

    let marg_size = marginal_columns.len();
    let mut permutation = Vec::with_capacity(cols);
    permutation.extend_from_slice(marginal_columns);
    permutation.extend_from_slice(keep_columns);
    let mut qj = DMatrix::zeros(rows, cols);
    for (new_column, &old_column) in permutation.iter().enumerate() {
        qj.column_mut(new_column)
            .copy_from(&jacobian.column(old_column));
    }
    let mut qr = rhs.clone();
    let rank_threshold = f64::EPSILON.sqrt();
    let mut total_rank = 0usize;
    let mut marginal_rank = 0usize;

    for column in 0..cols {
        if total_rank >= rows {
            break;
        }
        let row_start = total_rank;
        let norm = (row_start..rows)
            .map(|row| qj[(row, column)] * qj[(row, column)])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return None;
        }
        if norm > rank_threshold {
            // This is Eigen's makeHouseholderInPlace convention: beta is the
            // signed pivot and the reflector is H = I - tau*v*v^T.
            let x0 = qj[(row_start, column)];
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            let beta = -sign * norm;
            let v0 = x0 + sign * norm;
            let v_norm_sq = v0 * v0 + norm * norm - x0 * x0;
            if !v_norm_sq.is_finite() || v_norm_sq <= 0.0 {
                return None;
            }
            let tau = 2.0 / v_norm_sq;

            // Keep the sub-pivot entries in the current column until the RHS
            // reflector has been applied, exactly as upstream does.
            qj[(row_start, column)] = beta;
            for trailing in (column + 1)..cols {
                let mut dot = v0 * qj[(row_start, trailing)];
                for row in (row_start + 1)..rows {
                    dot += qj[(row, column)] * qj[(row, trailing)];
                }
                let scale = tau * dot;
                qj[(row_start, trailing)] -= scale * v0;
                for row in (row_start + 1)..rows {
                    qj[(row, trailing)] -= scale * qj[(row, column)];
                }
            }
            let mut dot = v0 * qr[row_start];
            for row in (row_start + 1)..rows {
                dot += qj[(row, column)] * qr[row];
            }
            let scale = tau * dot;
            qr[row_start] -= scale * v0;
            for row in (row_start + 1)..rows {
                qr[row] -= scale * qj[(row, column)];
            }
            for row in (row_start + 1)..rows {
                qj[(row, column)] = 0.0;
            }
            total_rank += 1;
        } else {
            for row in row_start..rows {
                qj[(row, column)] = 0.0;
            }
        }
        if column + 1 == marg_size {
            marginal_rank = total_rank;
        }
    }
    if marg_size == 0 {
        marginal_rank = 0;
    }
    let keep_rows = total_rank.saturating_sub(marginal_rank).max(1);
    let available_rows = rows.saturating_sub(marginal_rank);
    if available_rows == 0 || keep_rows > available_rows {
        return None;
    }
    let keep_size = keep_columns.len();
    let mut out_j = DMatrix::zeros(keep_rows, keep_size);
    for row in 0..keep_rows {
        out_j
            .row_mut(row)
            .copy_from(&qj.row(marginal_rank + row).columns(marg_size, keep_size));
    }
    let out_rhs =
        DVector::from_iterator(keep_rows, (0..keep_rows).map(|row| qr[marginal_rank + row]));
    Some(super::aom::SqrtPrior {
        jacobian: out_j,
        rhs: out_rhs,
        fej_point,
    })
}

/// Exact scalar-boundary variant of Basalt's SqrtToSqrt marginalization.
///
/// The estimator is instantiated as `MargHelper<float>` upstream.  Keeping the
/// public Rust window representation in `f64` does not make the QR boundary an
/// `f64` operation: Eigen first casts/constructs a float AOM, then
/// `makeHouseholderInPlace` computes the reflector and the rank cutoff in
/// `float`.  This helper makes that boundary explicit and returns values widened
/// back to the surrounding Rust API only after every QR operation has completed.
///
/// The ordering arguments represent upstream `std::set<int>` values and are
/// therefore sorted before the `[marg | keep]` permutation.  The reflector
/// storage and update order mirror Eigen's `makeHouseholderInPlace` plus
/// `applyHouseholderOnTheLeft`; in particular, the essential vector is stored as
/// `tail / (c0 - beta)`, rather than using a normalized `[v0 | tail]` vector.
pub(crate) fn sqrt_marginalize_upstream_f32(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    sqrt_marginalize_upstream_f32_packet(jacobian, rhs, keep_columns, marginal_columns, fej_point)
}

/// Safe f32 emulation of Eigen 5.0.1's packetized SqrtToSqrt path.
///
/// Eigen's default x86 build uses four-lane `Packet4f` operations.  The
/// explicit packet helpers above preserve the packet reduction and GEMV
/// operation order without relying on target-specific intrinsics or `unsafe`.
pub(super) fn sqrt_marginalize_upstream_f32_packet(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    let rows = jacobian.nrows();
    let cols = jacobian.ncols();
    if rows == 0
        || rhs.len() != rows
        || keep_columns.len() + marginal_columns.len() != cols
        || fej_point.len() != keep_columns.len()
    {
        return None;
    }

    // The upstream call receives std::set<int>.  Sort the Rust slices at this
    // boundary so equivalent callers cannot change the Eigen permutation by
    // passing a different input order.
    let mut keep = keep_columns.to_vec();
    let mut marg = marginal_columns.to_vec();
    keep.sort_unstable();
    marg.sort_unstable();
    let mut seen = vec![false; cols];
    for &column in keep.iter().chain(marg.iter()) {
        if column >= cols || seen[column] {
            return None;
        }
        seen[column] = true;
    }
    if seen.iter().any(|present| !present) {
        return None;
    }

    // Cast the complete stack before doing arithmetic.  This is the scalar
    // boundary of SqrtKeypointVioEstimator<float>; casting only final values
    // would retain the wrong reflector and rank decisions.
    let mut qj = vec![0.0_f32; rows * cols];
    for column in 0..cols {
        for row in 0..rows {
            qj[column * rows + row] = jacobian[(row, column)] as f32;
        }
    }
    let mut permutation = marg;
    permutation.extend_from_slice(&keep);
    let mut permuted = vec![0.0_f32; rows * cols];
    for (new_column, &old_column) in permutation.iter().enumerate() {
        for row in 0..rows {
            permuted[new_column * rows + row] = qj[old_column * rows + row];
        }
    }
    qj = permuted;
    let mut qr = rhs.iter().map(|value| *value as f32).collect::<Vec<_>>();

    let marg_size = marginal_columns.len();
    let rank_threshold = f32::EPSILON.sqrt();
    let mut total_rank = 0usize;
    let mut marginal_rank = 0usize;
    let diagnostic_policy = diagnostic_env_snapshot();
    let trace_reflectors = diagnostic_policy.sqrt_trace;
    let trace_focus_k12 = diagnostic_policy.sqrt_focus_k12;
    let trace_workspace_all = diagnostic_policy.sqrt_dump_workspace_all;
    let trace_tail_dump = diagnostic_policy.sqrt_dump_tail;
    let mut trace_qj_dump = diagnostic_policy.sqrt_dump_qj.as_ref().and_then(|path| {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .ok()
    });
    if let Some(output) = trace_qj_dump.as_mut() {
        if !upstream_f32_write_qtrace_header(output, rows, cols, qr.len(), marg_size, keep.len()) {
            trace_qj_dump = None;
        }
    }
    if trace_reflectors {
        eprintln!("trace_base {}", qj.as_ptr() as usize & 0x0f);
    }

    for column in 0..cols {
        if total_rank >= rows {
            break;
        }
        let row_start = total_rank;
        let remaining_rows = rows - row_start;
        let remaining_columns = cols - column - 1;
        let tail_len = remaining_rows.saturating_sub(1);
        let column_start = column * rows;
        let tail_start = column_start + row_start + 1;
        let c0 = qj[column_start + row_start];
        let tail_sq_norm = upstream_f32_packet8_squared_norm(&qj, tail_start, tail_len);
        if trace_tail_dump && column == 12 {
            eprint!("trace_tail_bits {} {}", column, tail_len);
            for value in &qj[tail_start..tail_start + tail_len] {
                eprint!(" {:08x}", value.to_bits());
            }
            eprintln!();
        }
        let (tau, beta) = if tail_sq_norm <= f32::MIN_POSITIVE {
            (0.0_f32, c0)
        } else {
            let mut beta = upstream_f32_householder_beta(c0, tail_sq_norm);
            if c0 >= 0.0 {
                beta = -beta;
            }
            let denominator = c0 - beta;
            upstream_f32_packet_divide_assign(&mut qj, tail_start, tail_len, denominator);
            let tau = (beta - c0) / beta;
            (tau, beta)
        };

        if trace_reflectors {
            eprintln!(
                "trace_norm_meta {} {} {} {}",
                column,
                tail_len,
                (qj.as_ptr() as usize + tail_start * std::mem::size_of::<f32>()) & 31,
                ((32 - ((qj.as_ptr() as usize + tail_start * std::mem::size_of::<f32>()) & 31))
                    & 31)
                    / std::mem::size_of::<f32>(),
            );
            eprintln!(
                "trace_reflector {} {} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {}",
                column,
                total_rank,
                c0.to_bits(),
                tail_sq_norm.to_bits(),
                beta.to_bits(),
                tau.to_bits(),
                (beta.abs() > rank_threshold) as u8,
                rank_threshold.to_bits(),
                tail_len,
            );
        }

        if beta.abs() > rank_threshold {
            qj[column_start + row_start] = beta;
            if tau != 0.0 {
                if remaining_rows == 1 {
                    let scale = 1.0_f32 - tau;
                    for trailing in (column + 1)..(column + 1 + remaining_columns) {
                        qj[trailing * rows + row_start] *= scale;
                    }
                    qr[row_start] *= scale;
                } else {
                    let essential = qj[tail_start..tail_start + tail_len].to_vec();
                    if trace_focus_k12 && column == 15 {
                        eprintln!(
                            "focus_begin {} {} {} {} {:08x} {:08x}",
                            column,
                            row_start,
                            remaining_rows,
                            remaining_columns,
                            tau.to_bits(),
                            beta.to_bits(),
                        );
                        eprint!("focus_essential");
                        for value in &essential {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!("focus_bottom {} {}", trailing_offset, trailing);
                            let base = trailing * rows + row_start + 1;
                            for offset in 0..tail_len {
                                eprint!(" {:08x}", qj[base + offset].to_bits());
                            }
                            eprintln!();
                        }
                    }
                    // ProductEvaluators materializes tau*essential before
                    // applying each scalar workspace to the outer product.
                    let scaled_essential = upstream_f32_packet_scale(&essential, tau);
                    // The pinned Eigen applyHouseholderOnTheLeft dispatches
                    // an exactly one-column block to its dynamic inner product
                    // (3d3cf3 -> 3d43e0), with four Packet8 accumulators. This
                    // is not the final output row of the multi-column GEMV.
                    let workspace = if remaining_columns == 1 {
                        vec![upstream_f32_packet8_rhs_dot(
                            &essential,
                            &qj,
                            (column + 1) * rows + row_start + 1,
                            tail_len,
                        )]
                    } else {
                        upstream_f32_packet8_gemv_workspace(
                            &essential,
                            &qj,
                            rows,
                            row_start + 1,
                            column + 1,
                            remaining_columns,
                        )
                    };
                    if trace_workspace_all {
                        eprint!(
                            "trace_workspace_all {} {} {} {}",
                            column, row_start, remaining_rows, remaining_columns,
                        );
                        for value in &workspace {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                    }
                    if trace_focus_k12 && column == 15 {
                        eprint!("focus_workspace_raw");
                        for value in &workspace {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprint!("focus_row0");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!(" {:08x}", qj[trailing * rows + row_start].to_bits());
                        }
                        eprintln!();
                        eprint!("focus_workspace_plus_row0");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            let value =
                                workspace[trailing_offset] + qj[trailing * rows + row_start];
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprint!("focus_scaled_workspace");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            let value =
                                workspace[trailing_offset] + qj[trailing * rows + row_start];
                            eprint!(" {:08x}", (tau * value).to_bits());
                        }
                        eprintln!();
                    }
                    if trace_reflectors {
                        eprint!("trace_workspace {}", column);
                        for value in workspace.iter().take(8) {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        if column == 9 {
                            eprint!("trace_workspace_full {}", column);
                            for value in &workspace {
                                eprint!(" {:08x}", value.to_bits());
                            }
                            eprintln!();
                        }
                    }
                    if trace_reflectors && column == 9 {
                        let mut workspace_after_add = workspace.clone();
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            workspace_after_add[trailing_offset] += qj[trailing * rows + row_start];
                        }
                        eprintln!(
                            "trace_stage9_workspace_add {:016x}",
                            upstream_f32_bits_hash(&workspace_after_add),
                        );
                        eprint!("trace_stage9_workspace_add_full");
                        for value in &workspace_after_add {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        let diagnostic_lane = 14usize;
                        let trailing = column + 1 + diagnostic_lane;
                        let base = trailing * rows;
                        let workspace_value = workspace_after_add[diagnostic_lane];
                        eprint!("trace_stage9_col14_pre");
                        for value in &qj[base..base + rows] {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprintln!(
                            "trace_stage9_operands {:08x} {:08x} {:08x} {:08x} {:08x}",
                            workspace_value.to_bits(),
                            essential[3].to_bits(),
                            essential[4].to_bits(),
                            scaled_essential[3].to_bits(),
                            scaled_essential[4].to_bits(),
                        );
                        eprint!("trace_stage9_outer_headhash");
                        for requested_head in 0..8 {
                            let mut candidate = qj[base..base + rows].to_vec();
                            candidate[row_start] =
                                (-tau).fused_mul_add(workspace_value, candidate[row_start]);
                            let head = requested_head.min(tail_len);
                            let packet_end = head + ((tail_len - head) / 8) * 8;
                            for offset in 0..head {
                                candidate[row_start + 1 + offset] = (-workspace_value)
                                    .fused_mul_add(
                                        scaled_essential[offset],
                                        candidate[row_start + 1 + offset],
                                    );
                            }
                            let mut offset = head;
                            while offset < packet_end {
                                let destination =
                                    upstream_f32_packet8_load(&candidate, row_start + 1 + offset);
                                let source = upstream_f32_packet8_load(&scaled_essential, offset);
                                let updated = UpstreamF32Packet8([
                                    (-workspace_value).fused_mul_add(source.0[0], destination.0[0]),
                                    (-workspace_value).fused_mul_add(source.0[1], destination.0[1]),
                                    (-workspace_value).fused_mul_add(source.0[2], destination.0[2]),
                                    (-workspace_value).fused_mul_add(source.0[3], destination.0[3]),
                                    (-workspace_value).fused_mul_add(source.0[4], destination.0[4]),
                                    (-workspace_value).fused_mul_add(source.0[5], destination.0[5]),
                                    (-workspace_value).fused_mul_add(source.0[6], destination.0[6]),
                                    (-workspace_value).fused_mul_add(source.0[7], destination.0[7]),
                                ]);
                                upstream_f32_packet8_store(
                                    &mut candidate,
                                    row_start + 1 + offset,
                                    updated,
                                );
                                offset += 8;
                            }
                            while offset < tail_len {
                                candidate[row_start + 1 + offset] = (-workspace_value)
                                    .fused_mul_add(
                                        scaled_essential[offset],
                                        candidate[row_start + 1 + offset],
                                    );
                                offset += 1;
                            }
                            eprint!(" {:016x}", upstream_f32_bits_hash(&candidate));
                        }
                        eprintln!();
                        let mut direct = qj[base..base + rows].to_vec();
                        direct[row_start] =
                            (-tau).fused_mul_add(workspace_value, direct[row_start]);
                        let direct_scale = -tau * workspace_value;
                        for offset in 0..tail_len {
                            direct[row_start + 1 + offset] = direct_scale
                                .fused_mul_add(essential[offset], direct[row_start + 1 + offset]);
                        }
                        eprintln!(
                            "trace_stage9_outer_direct {:016x}",
                            upstream_f32_bits_hash(&direct)
                        );
                    }
                    for trailing_offset in 0..remaining_columns {
                        let trailing = column + 1 + trailing_offset;
                        let trailing_tail_start = trailing * rows + row_start + 1;
                        let mut workspace_value = workspace[trailing_offset];
                        workspace_value += qj[trailing * rows + row_start];
                        qj[trailing * rows + row_start] =
                            (-tau).fused_mul_add(workspace_value, qj[trailing * rows + row_start]);
                        upstream_f32_packet8_sub_scaled(
                            &mut qj,
                            trailing_tail_start,
                            &scaled_essential,
                            workspace_value,
                        );
                    }
                    if trace_focus_k12 && column == 15 {
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!("focus_post {} {}", trailing_offset, trailing);
                            let base = trailing * rows + row_start;
                            for offset in 0..remaining_rows {
                                eprint!(" {:08x}", qj[base + offset].to_bits());
                            }
                            eprintln!();
                        }
                    }
                    if trace_reflectors && column == 9 {
                        eprintln!(
                            "trace_stage9_row0 {:016x}",
                            upstream_f32_bits_hash(
                                &(0..remaining_columns)
                                    .map(|offset| { qj[(column + 1 + offset) * rows + row_start] })
                                    .collect::<Vec<_>>(),
                            ),
                        );
                        eprint!("trace_stage9_colhash");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!(
                                " {:016x}",
                                upstream_f32_bits_hash(&qj[trailing * rows..(trailing + 1) * rows])
                            );
                        }
                        eprintln!();
                        let trailing = column + 1 + 14;
                        eprint!("trace_stage9_col14_post");
                        for value in &qj[trailing * rows..(trailing + 1) * rows] {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                    }
                    let mut workspace =
                        upstream_f32_packet8_rhs_dot(&essential, &qr, row_start + 1, tail_len);
                    workspace += qr[row_start];
                    qr[row_start] = (-tau).fused_mul_add(workspace, qr[row_start]);
                    let scaled_essential = upstream_f32_packet_scale(&essential, tau);
                    upstream_f32_packet8_sub_scaled_fma_head(
                        &mut qr,
                        row_start + 1,
                        &scaled_essential,
                        workspace,
                    );
                    if trace_reflectors {
                        eprintln!(
                            "trace_rhs {} {:016x} {:08x} {:08x}",
                            column,
                            upstream_f32_bits_hash(&qr),
                            qr[row_start].to_bits(),
                            qr[std::cmp::min(row_start + 1, qr.len() - 1)].to_bits(),
                        );
                    }
                }
            }
            total_rank += 1;
        } else {
            for row in row_start..rows {
                qj[column_start + row] = 0.0;
            }
        }

        // Eigen clears the essential vector after applying the reflector.
        for row in (row_start + 1)..rows {
            qj[column_start + row] = 0.0;
        }
        if trace_reflectors && total_rank > 0 {
            let mut hash = 1_469_598_103_934_665_603_u64;
            for value in &qj {
                for byte in value.to_bits().to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(1_099_511_628_211_u64);
                }
            }
            eprintln!(
                "trace_after {} {} {:016x} {:08x} {:08x}",
                column,
                total_rank,
                hash,
                qj[column_start + row_start].to_bits(),
                if column + 1 < cols {
                    qj[(column + 1) * rows + row_start].to_bits()
                } else {
                    0
                },
            );
        }
        if let Some(output) = trace_qj_dump.as_mut() {
            if !upstream_f32_write_qtrace_record(
                output,
                column,
                row_start,
                total_rank,
                c0,
                tail_sq_norm,
                beta,
                tau,
                &qj,
                &qr,
            ) {
                trace_qj_dump = None;
            }
        }
        if column + 1 == marg_size {
            marginal_rank = total_rank;
        }
    }
    if marg_size == 0 {
        marginal_rank = 0;
    }

    let keep_rows = total_rank.saturating_sub(marginal_rank).max(1);
    let available_rows = rows.saturating_sub(marginal_rank);
    if available_rows == 0 || keep_rows > available_rows {
        return None;
    }
    let keep_size = keep.len();
    let mut out_j = DMatrix::<f64>::zeros(keep_rows, keep_size);
    for row in 0..keep_rows {
        for column in 0..keep_size {
            out_j[(row, column)] = qj[(marg_size + column) * rows + marginal_rank + row] as f64;
        }
    }
    let out_rhs = DVector::from_iterator(
        keep_rows,
        (0..keep_rows).map(|row| qr[marginal_rank + row] as f64),
    );
    let fej_point = DVector::from_iterator(
        fej_point.len(),
        fej_point.iter().map(|value| *value as f32 as f64),
    );
    Some(super::aom::SqrtPrior {
        jacobian: out_j,
        rhs: out_rhs,
        fej_point,
    })
}
