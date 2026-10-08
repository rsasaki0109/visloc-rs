//! Opt-in solver-step and Schur-block diagnostics.

use super::*;

/// Keep solver-step diagnostics opt-in even when a caller already enables a
/// higher-level SFM trace.  This flag is intentionally read here rather than
/// threaded through [`BaConfig`], so the public/default optimizer state stays
/// byte-identical and a diagnostic cannot accidentally become a production
/// behavior switch.
pub(super) fn ba_step_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
        && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
}

/// Enable the scalar LM-step quality diagnostic only when all existing BA
/// debug gates and the dedicated quality gate are present.  The diagnostic is
/// deliberately not threaded through `BaConfig`: when this returns false the
/// matrix-free solver does not allocate its residual/backward-error scratch
/// or perform an additional normal-equation scan.
pub(super) fn ba_lm_step_quality_debug_enabled() -> bool {
    ba_step_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_LM_QUALITY").is_some()
}

/// Context for the opt-in local Schur-block diagnostic.  This is deliberately
/// private and borrowed: the normal solver does not retain a pose/landmark
/// history or any diagnostic records.
#[derive(Debug, Clone, Copy)]
pub(super) struct SchurBlockDebugContext<'a> {
    pub(super) iteration: usize,
    pub(super) pose_slot: usize,
    pub(super) frame_id: Option<u64>,
    scaled_coordinates: bool,
    pub(super) landmark_index: &'a BTreeMap<u64, usize>,
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct SchurBlockDebugCounts {
    pub(super) local_landmarks: usize,
    pub(super) valid_hll: usize,
    pub(super) singular_hll: usize,
    pub(super) cross_entries: usize,
    pub(super) same_pose_groups: usize,
    pub(super) same_pose_extra_cross_entries: usize,
    pub(super) max_elimination_norm: Option<f64>,
    pub(super) max_elimination_landmark: Option<usize>,
    pub(super) max_hll_inverse_residual: Option<f64>,
    max_elimination_cross: Option<Matrix6x3<f64>>,
    max_hll: Option<Matrix3<f64>>,
    max_hll_inverse: Option<Matrix3<f64>>,
    nonfinite_elimination: bool,
    nonfinite_hll_inverse_residual: bool,
}

/// The slot is intentionally selected through an explicit diagnostic
/// environment variable.  The existing two debug flags remain a required
/// gate, so setting only the slot cannot perturb normal BA.
pub(super) fn ba_schur_debug_slot() -> Option<usize> {
    if !ba_step_debug_enabled() {
        return None;
    }
    std::env::var("VISLOC_SFM_DEBUG_BA_SCHUR_SLOT")
        .ok()?
        .parse::<usize>()
        .ok()
}

pub(super) fn claim_sparse_debug_window(
    eligible: bool,
    slot: Option<usize>,
    pose_count: usize,
    claimed: &std::sync::atomic::AtomicBool,
) -> Option<usize> {
    let slot = slot.filter(|slot| eligible && *slot < pose_count)?;
    claimed
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .ok()?;
    Some(slot)
}

pub(super) fn matrix_free_schur_debug_context<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    scaled_coordinates: bool,
) -> Option<SchurBlockDebugContext<'a>> {
    let pose_slot = ba_schur_debug_slot()?;
    Some(schur_debug_context_for_slot_with_coordinates(
        pose_index,
        landmark_index,
        iteration,
        pose_slot,
        scaled_coordinates,
    ))
}

#[cfg(test)]
pub(super) fn schur_debug_context_for_slot<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    pose_slot: usize,
) -> SchurBlockDebugContext<'a> {
    schur_debug_context_for_slot_with_coordinates(
        pose_index,
        landmark_index,
        iteration,
        pose_slot,
        false,
    )
}

pub(super) fn schur_debug_context_for_slot_with_coordinates<'a>(
    pose_index: &'a BTreeMap<u64, usize>,
    landmark_index: &'a BTreeMap<u64, usize>,
    iteration: usize,
    pose_slot: usize,
    scaled_coordinates: bool,
) -> SchurBlockDebugContext<'a> {
    let frame_id = pose_index
        .iter()
        .find_map(|(id, slot)| (*slot == pose_slot).then_some(*id));
    SchurBlockDebugContext {
        iteration,
        pose_slot,
        frame_id,
        scaled_coordinates,
        landmark_index,
    }
}

fn finite_matrix6(matrix: &Matrix6<f64>) -> bool {
    matrix.iter().all(|value| value.is_finite())
}

fn symmetric_from_lower(matrix: &Matrix6<f64>) -> Matrix6<f64> {
    let mut symmetric = *matrix;
    for row in 0..6 {
        for column in (row + 1)..6 {
            symmetric[(row, column)] = matrix[(column, row)];
        }
    }
    symmetric
}

fn symmetric_from_upper(matrix: &Matrix6<f64>) -> Matrix6<f64> {
    let mut symmetric = *matrix;
    for row in 0..6 {
        for column in (row + 1)..6 {
            symmetric[(column, row)] = matrix[(row, column)];
        }
    }
    symmetric
}

fn matrix6_asymmetry(matrix: &Matrix6<f64>) -> f64 {
    let mut maximum = 0.0_f64;
    for row in 0..6 {
        for column in (row + 1)..6 {
            maximum = maximum.max((matrix[(row, column)] - matrix[(column, row)]).abs());
        }
    }
    maximum
}

fn matrix6_min_eigenvalue(matrix: &Matrix6<f64>) -> Option<f64> {
    if !finite_matrix6(matrix) {
        return None;
    }
    // Never let a diagnostic eigensolve run indefinitely on a pathological
    // block.  The production Cholesky result remains the acceptance decision.
    let eigenvalues =
        nalgebra::linalg::SymmetricEigen::try_new(*matrix, f64::EPSILON, 1024)?.eigenvalues;
    let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    minimum.is_finite().then_some(minimum)
}

/// Return the smallest diagonal radicand (before its square root) of a
/// lower-triangle Cholesky factorization. This is a diagnostic-only scalar
/// 6x6 routine; the production factorization remains `Matrix6::cholesky` and
/// is not replaced or symmetrized by this audit.
fn matrix6_min_lower_cholesky_pivot(matrix: &Matrix6<f64>) -> Option<f64> {
    if !finite_matrix6(matrix) {
        return None;
    }
    let mut lower = Matrix6::zeros();
    let mut minimum = f64::INFINITY;
    for row in 0..6 {
        let mut pivot = matrix[(row, row)];
        for column in 0..row {
            pivot -= lower[(row, column)] * lower[(row, column)];
        }
        if !pivot.is_finite() || pivot <= 0.0 {
            return None;
        }
        minimum = minimum.min(pivot);
        lower[(row, row)] = pivot.sqrt();
        for next_row in (row + 1)..6 {
            let mut value = matrix[(next_row, row)];
            for column in 0..row {
                value -= lower[(next_row, column)] * lower[(row, column)];
            }
            lower[(next_row, row)] = value / lower[(row, row)];
        }
    }
    minimum.is_finite().then_some(minimum)
}

fn format_matrix6(matrix: &Matrix6<f64>) -> String {
    let mut output = String::with_capacity(6 * 6 * 20 + 1);
    output.push('[');
    for row in 0..6 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..6 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

fn format_matrix6x3(matrix: &Matrix6x3<f64>) -> String {
    let mut output = String::with_capacity(6 * 3 * 20 + 1);
    output.push('[');
    for row in 0..6 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..3 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

fn format_matrix3(matrix: &Matrix3<f64>) -> String {
    let mut output = String::with_capacity(3 * 3 * 20 + 1);
    output.push('[');
    for row in 0..3 {
        if row != 0 {
            output.push(';');
        }
        for column in 0..3 {
            if column != 0 {
                output.push(',');
            }
            write!(&mut output, "{:.17e}", matrix[(row, column)])
                .expect("writing a String cannot fail");
        }
    }
    output.push(']');
    output
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SchurBlockDebugMetrics {
    pub(super) finite: bool,
    pub(super) asymmetry: f64,
    pub(super) lower_min_eigenvalue: Option<f64>,
    pub(super) upper_min_eigenvalue: Option<f64>,
    pub(super) lower_min_cholesky_radicand: Option<f64>,
    pub(super) upper_min_cholesky_radicand: Option<f64>,
}

pub(super) fn inspect_schur_block(matrix: &Matrix6<f64>) -> SchurBlockDebugMetrics {
    let finite = finite_matrix6(matrix);
    let lower = symmetric_from_lower(matrix);
    let upper = symmetric_from_upper(matrix);
    SchurBlockDebugMetrics {
        finite,
        asymmetry: matrix6_asymmetry(matrix),
        lower_min_eigenvalue: matrix6_min_eigenvalue(&lower),
        upper_min_eigenvalue: matrix6_min_eigenvalue(&upper),
        lower_min_cholesky_radicand: matrix6_min_lower_cholesky_pivot(&lower),
        upper_min_cholesky_radicand: matrix6_min_lower_cholesky_pivot(&upper),
    }
}

pub(super) fn collect_schur_block_debug_counts(
    system: &NormalEquationsBa,
    h_ll_inverse: &[Option<Matrix3<f64>>],
    lambda: f64,
    pose_slot: usize,
) -> SchurBlockDebugCounts {
    let mut counts = SchurBlockDebugCounts::default();
    for (landmark_index, (landmark, inverse)) in
        system.landmarks.iter().zip(h_ll_inverse).enumerate()
    {
        let mut same_pose_cross = Matrix6x3::zeros();
        let mut cross_entries = 0_usize;
        for (pose, cross) in &landmark.cross {
            if *pose == pose_slot {
                cross_entries += 1;
                same_pose_cross += cross;
            }
        }
        if cross_entries == 0 {
            continue;
        }
        counts.local_landmarks += 1;
        counts.cross_entries += cross_entries;
        if cross_entries > 1 {
            counts.same_pose_groups += 1;
            counts.same_pose_extra_cross_entries += cross_entries - 1;
        }
        let Some(inverse) = inverse else {
            counts.singular_hll += 1;
            continue;
        };
        counts.valid_hll += 1;
        let elimination = same_pose_cross * inverse * same_pose_cross.transpose();
        let elimination_norm = elimination.norm();
        if !elimination_norm.is_finite() {
            counts.nonfinite_elimination = true;
            continue;
        }
        let mut h_ll = landmark.h_ll;
        for component in 0..3 {
            h_ll[(component, component)] += lambda;
        }
        let inverse_residual = (h_ll * inverse - Matrix3::identity()).norm();
        if !inverse_residual.is_finite() {
            counts.nonfinite_hll_inverse_residual = true;
        }
        if counts
            .max_elimination_norm
            .is_none_or(|maximum| elimination_norm > maximum)
        {
            counts.max_elimination_norm = Some(elimination_norm);
            counts.max_elimination_landmark = Some(landmark_index);
            counts.max_hll_inverse_residual = Some(inverse_residual);
            counts.max_elimination_cross = Some(same_pose_cross);
            counts.max_hll = Some(h_ll);
            counts.max_hll_inverse = Some(*inverse);
        }
    }
    counts
}

pub(super) fn emit_schur_block_debug(
    context: SchurBlockDebugContext<'_>,
    lambda: f64,
    h_pp_lambda: &Matrix6<f64>,
    schur_block: &Matrix6<f64>,
    counts: SchurBlockDebugCounts,
) {
    let h_pp_metrics = inspect_schur_block(h_pp_lambda);
    let schur_metrics = inspect_schur_block(schur_block);
    let elimination = *h_pp_lambda - *schur_block;
    let elimination_norm = elimination.norm();
    let finite = h_pp_metrics.finite
        && schur_metrics.finite
        && elimination_norm.is_finite()
        && !counts.nonfinite_elimination
        && !counts.nonfinite_hll_inverse_residual;
    let landmark_id = counts.max_elimination_landmark.and_then(|index| {
        context
            .landmark_index
            .iter()
            .find_map(|(id, mapped)| (*mapped == index).then_some(*id))
    });
    let frame = context
        .frame_id
        .map_or_else(|| "none".to_owned(), |id| id.to_string());
    let max_landmark = counts
        .max_elimination_landmark
        .map_or_else(|| "none".to_owned(), |index| index.to_string());
    let max_landmark_id = landmark_id.map_or_else(|| "none".to_owned(), |id| id.to_string());
    let max_cross = counts
        .max_elimination_cross
        .map_or_else(|| "none".to_owned(), |cross| format_matrix6x3(&cross));
    let max_hll = counts
        .max_hll
        .map_or_else(|| "none".to_owned(), |hll| format_matrix3(&hll));
    let max_hll_inverse = counts
        .max_hll_inverse
        .map_or_else(|| "none".to_owned(), |inverse| format_matrix3(&inverse));
    // Keep the legacy diagnostic byte-for-byte stable.  The coordinate and
    // damping labels are meaningful only for the opt-in scaled system; an
    // empty annotation leaves the old `finite=... solver_...` token sequence
    // unchanged.
    let coordinate_annotation = if context.scaled_coordinates {
        " coordinate_system=scaled damping_metric=identity"
    } else {
        ""
    };
    eprintln!(
        concat!(
            "sfm-debug-ba-schur-block iteration={} slot={} frame_id={} lambda={:.17e} ",
            "finite={}{} solver_symmetric_interpretation=lower_triangle ",
            "hpp_lambda={} schur_block={} elimination_norm={:.17e} ",
            "schur_asymmetry={:.17e} lower_min_eigen={:?} upper_min_eigen={:?} ",
            "lower_min_cholesky_radicand={:?} upper_min_cholesky_radicand={:?} ",
            "local_landmarks={} valid_hll={} singular_hll={} cross_entries={} ",
            "same_pose_groups={} same_pose_extra_cross_entries={} ",
            "max_elimination_norm={:?} max_elimination_landmark={} ",
            "max_elimination_point_id={} max_hll_inverse_residual={:?} ",
            "max_elimination_cross={} max_hll={} max_hll_inverse={}"
        ),
        context.iteration,
        context.pose_slot,
        frame,
        lambda,
        finite,
        coordinate_annotation,
        format_matrix6(h_pp_lambda),
        format_matrix6(schur_block),
        elimination_norm,
        schur_metrics.asymmetry,
        schur_metrics.lower_min_eigenvalue,
        schur_metrics.upper_min_eigenvalue,
        schur_metrics.lower_min_cholesky_radicand,
        schur_metrics.upper_min_cholesky_radicand,
        counts.local_landmarks,
        counts.valid_hll,
        counts.singular_hll,
        counts.cross_entries,
        counts.same_pose_groups,
        counts.same_pose_extra_cross_entries,
        counts.max_elimination_norm,
        max_landmark,
        max_landmark_id,
        counts.max_hll_inverse_residual,
        max_cross,
        max_hll,
        max_hll_inverse,
    );
}

pub(super) fn feasible_backtrack_accepts(
    before_cost: f64,
    trial_cost: f64,
    before_nonprojectable: usize,
    trial_nonprojectable: usize,
    preserves_valid: bool,
) -> bool {
    before_cost.is_finite()
        && trial_cost.is_finite()
        && trial_cost < before_cost
        && trial_nonprojectable <= before_nonprojectable
        && preserves_valid
}
