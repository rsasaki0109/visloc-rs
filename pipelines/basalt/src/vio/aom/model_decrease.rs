//! Model cost-decrease evaluation (f64, f32 and QR-payload reuse).

use super::*;

/// Compute the model-cost decrease for a linearized row stack after the
/// eliminated landmark block is back-substituted.
///
/// Basalt's `LinearizationAbsQR::backSubstitute` returns more than the
/// reduced-camera quadratic change.  Each landmark block contributes the
/// complete transformed `[Jp | Jl | r]` stack, including its first three
/// (`Q1`) rows.  Those rows account for the landmark-only part of the model
/// decrease and must be included in LM's `relative_decrease`; using only
/// `bᵀs + 1/2 sᵀHs` omits them and changes the lambda schedule.
pub fn model_cost_decrease(
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let mut decrease = 0.0;
    for factor in factors {
        if factor.state_jacobian.ncols() != state_step.len() {
            return None;
        }

        let landmark_columns = factor.landmark_jacobian.ncols();
        if landmark_columns == 0 {
            let j_inc = &factor.state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &factor.residual));
            continue;
        }
        let (state_jacobian, landmark_jacobian, residual) = augmented_landmark_rows(factor);
        let rows = state_jacobian.nrows();
        if rows < landmark_columns {
            // There are not enough rows to recover the full landmark block;
            // landmark_steps() likewise leaves this point unchanged.
            let j_inc = &state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &residual));
            continue;
        }

        // Mirror the upstream Householder path: transform the complete row
        // stack with the same Q, then solve the leading R block for the
        // landmark increment before evaluating the full model change.
        let qr = landmark_jacobian.clone().qr();
        let r = qr.r();
        let rank = (0..r.nrows().min(r.ncols()))
            .filter(|&i| r[(i, i)].abs() > tolerance)
            .count();
        if rank < landmark_columns {
            // WindowProblem::landmark_steps deliberately leaves a
            // rank-deficient landmark unchanged.  Match that trial path in
            // the model calculation instead of turning a newly observed,
            // underconstrained point into a solver-wide failure.
            let j_inc = &state_jacobian * state_step;
            decrease -= j_inc.dot(&(0.5 * &j_inc + &residual));
            continue;
        }

        let mut transformed_state = state_jacobian;
        let mut transformed_residual = DMatrix::from_column_slice(rows, 1, residual.as_slice());
        qr.q_tr_mul(&mut transformed_state);
        qr.q_tr_mul(&mut transformed_residual);

        let mut qj_inc = &transformed_state * state_step;
        let mut rhs = transformed_residual
            .column(0)
            .rows(0, landmark_columns)
            .into_owned();
        for row in 0..landmark_columns {
            rhs[row] += qj_inc[row];
        }
        rhs = -rhs;
        let mut landmark_inc = DVector::zeros(landmark_columns);
        for row in (0..landmark_columns).rev() {
            let mut value = rhs[row];
            for column in (row + 1)..landmark_columns {
                value -= r[(row, column)] * landmark_inc[column];
            }
            let diagonal = r[(row, row)];
            if diagonal.abs() <= tolerance {
                return None;
            }
            landmark_inc[row] = value / diagonal;
        }
        let q1_landmark_inc = r * landmark_inc;
        for row in 0..landmark_columns {
            qj_inc[row] += q1_landmark_inc[row];
        }

        let qres = transformed_residual.column(0).into_owned();
        decrease -= qj_inc.dot(&(0.5 * &qj_inc + qres));
    }
    decrease.is_finite().then_some(decrease)
}

/// One factor's independent model-cost-decrease contribution from the
/// parallel pre-pass in [`model_cost_decrease_f32`], mirroring the four
/// outcomes its per-factor loop body could reach for an *unpaired* factor
/// (the Imu/Bias pairing decision is precomputed separately, since it needs
/// the adjacent factor too). `DimensionMismatch` and `Failure` both make the
/// overall function return `None`; kept distinct only for readability.
enum FactorEvaluation {
    DimensionMismatch,
    DeferredPrior(f32),
    Direct(f32),
    Failure,
}

/// Pure per-factor evaluation extracted from `model_cost_decrease_f32`'s
/// loop body, unchanged in arithmetic: every step below (the dimension
/// check, `prior_model_inputs_f32`, the zero-landmark scalar contribution,
/// and the full QR path) is copied verbatim, it just returns its outcome
/// instead of mutating a shared `decrease`/`deferred_prior`.
fn evaluate_model_decrease_factor(
    factor: &WhitenedFactorRowStack,
    step: &DVector<f32>,
    state_step: &DVector<f64>,
    threshold: f32,
) -> FactorEvaluation {
    if factor.state_jacobian.ncols() != step.len() {
        return FactorEvaluation::DimensionMismatch;
    }
    if let Some((j, rhs, compact_step)) = prior_model_inputs_f32(factor, state_step) {
        return FactorEvaluation::DeferredPrior(prior_model_f32(&j, &rhs, &compact_step));
    }
    let state = as_f32_matrix(&factor.state_jacobian);
    let residual = as_f32_vector(&factor.residual);
    let landmark_columns = factor.landmark_jacobian.ncols();
    if landmark_columns == 0 {
        let increment = state * step;
        let contribution = -increment.dot(&(0.5_f32 * &increment + &residual));
        return if factor.kind == FactorKind::Prior {
            FactorEvaluation::DeferredPrior(contribution)
        } else {
            FactorEvaluation::Direct(contribution)
        };
    }
    let landmark = as_f32_matrix(&factor.landmark_jacobian);
    let Some(qr) = LandmarkHouseholderF32::factor(&state, &landmark, &residual) else {
        return FactorEvaluation::Failure;
    };
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    if rank < landmark_columns {
        let increment = state * step;
        return FactorEvaluation::Direct(-increment.dot(&(0.5_f32 * &increment + &residual)));
    }
    let transformed_state = qr.transformed_state();
    let transformed_residual =
        DMatrix::from_column_slice(qr.rows, 1, qr.transformed_residual().as_slice());
    let r = qr.upper_r();
    let mut qj_inc = eigen_row_major_gemv_f32(&transformed_state, step);
    let mut rhs = transformed_residual
        .column(0)
        .rows(0, landmark_columns)
        .into_owned();
    for row in 0..landmark_columns {
        rhs[row] += qj_inc[row];
    }
    let mut landmark_inc = DVector::<f32>::zeros(landmark_columns);
    if landmark_columns == 3 {
        let d2 = r[(2, 2)];
        let d1 = r[(1, 1)];
        let d0 = r[(0, 0)];
        if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
            return FactorEvaluation::Failure;
        }
        let x2 = rhs[2] / d2;
        let x1 = (-r[(1, 2)]).mul_add(x2, rhs[1]) / d1;
        let row0_dot = r[(0, 2)].mul_add(x2, r[(0, 1)] * x1);
        let x0 = (rhs[0] - row0_dot) / d0;
        landmark_inc[0] = -x0;
        landmark_inc[1] = -x1;
        landmark_inc[2] = -x2;
    } else {
        rhs = -rhs;
        for row in (0..landmark_columns).rev() {
            let mut value = rhs[row];
            for column in (row + 1)..landmark_columns {
                value -= r[(row, column)] * landmark_inc[column];
            }
            let diagonal = r[(row, row)];
            if diagonal.abs() <= threshold {
                return FactorEvaluation::Failure;
            }
            landmark_inc[row] = value / diagonal;
        }
    }
    let q1_inc = r * landmark_inc;
    for row in 0..landmark_columns {
        qj_inc[row] += q1_inc[row];
    }
    let qres = transformed_residual.column(0).into_owned();
    FactorEvaluation::Direct(-eigen_visual_model_dot_f32(&qj_inc, &qres))
}

pub(super) fn model_cost_decrease_f32(
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let step = as_f32_vector(state_step);
    let threshold = tolerance as f32;

    // Every factor's Imu/Bias pairing check (needs only that factor and its
    // immediate successor) and its ordinary evaluation (needs only that one
    // factor) are pure and independent of every other factor's result and
    // of the running `decrease`/`deferred_prior`/`paired_bias_index` state
    // -- so both run in parallel below. `imu_bias_pair_model_dot_f32`
    // re-checks the same state-column/`step.len()` match internally, so a
    // factor whose own dimensions are wrong can never produce `Some` here
    // (see the comment in the serial fold). The fold afterwards makes
    // exactly the sequential decisions the original loop made -- pairing
    // precedence, immediate `None` on dimension/QR failure, deferred-prior
    // push order, `decrease +=` order -- just by reading each
    // already-computed pure result instead of recomputing it, so the f32
    // rounding tree is unchanged.
    let imu_pair_dots: Vec<Option<f32>> = factors
        .par_iter()
        .enumerate()
        .map(|(index, factor)| {
            if factor.kind != FactorKind::Imu {
                return None;
            }
            factors
                .get(index + 1)
                .and_then(|bias| imu_bias_pair_model_dot_f32(factor, bias, &step))
        })
        .collect();
    let evaluations: Vec<FactorEvaluation> = factors
        .par_iter()
        .map(|factor| evaluate_model_decrease_factor(factor, &step, state_step, threshold))
        .collect();

    let mut decrease = 0.0_f32;
    let mut deferred_prior = Vec::new();
    let mut paired_bias_index = None;
    for factor_index in 0..factors.len() {
        if paired_bias_index == Some(factor_index) {
            paired_bias_index = None;
            continue;
        }
        // A factor whose own `state_jacobian` width does not match `step`
        // cannot also produce `Some` from the pairing check above (that
        // check re-validates the same width on both the Imu and Bias
        // factor), so consulting the pairing result first cannot skip past
        // a dimension failure the original per-factor check would have
        // caught -- either this branch is not taken and `evaluations`
        // reports `DimensionMismatch` below, or the successor's own turn
        // later in this same fold reports it.
        if let Some(dot) = imu_pair_dots[factor_index] {
            decrease -= dot;
            paired_bias_index = Some(factor_index + 1);
            continue;
        }
        match evaluations[factor_index] {
            FactorEvaluation::DimensionMismatch | FactorEvaluation::Failure => return None,
            FactorEvaluation::DeferredPrior(value) => deferred_prior.push(value),
            FactorEvaluation::Direct(value) => decrease += value,
        }
    }
    // Upstream starts with the visual parallel-reduction result, applies all
    // ImuBlock terms, and only then adds the marginal-prior model change.
    for contribution in deferred_prior {
        decrease += contribution;
    }
    decrease.is_finite().then_some(decrease as f64)
}

/// Evaluate one native `ImuBlock` model term. Rust exposes the nine
/// preintegration rows and six bias-walk rows as adjacent semantic factors,
/// while upstream owns a single row-major 15x30 block and evaluates one GEMV
/// and one scalar-FMA dot over it.
pub(super) fn imu_bias_pair_model_dot_f32(
    imu: &WhitenedFactorRowStack,
    bias: &WhitenedFactorRowStack,
    state_step: &DVector<f32>,
) -> Option<f32> {
    if imu.kind != FactorKind::Imu
        || bias.kind != FactorKind::Bias
        || imu.rows() != 9
        || bias.rows() != 6
        || imu.landmark_jacobian.ncols() != 0
        || bias.landmark_jacobian.ncols() != 0
        || imu.state_jacobian.ncols() != state_step.len()
        || bias.state_jacobian.ncols() != state_step.len()
    {
        return None;
    }
    let offsets = imu.imu_link_offsets?;
    if bias.imu_link_offsets != Some(offsets)
        || offsets.start + AOM_NAV_DOF > state_step.len()
        || offsets.end + AOM_NAV_DOF > state_step.len()
        || offsets.start == offsets.end
    {
        return None;
    }

    let imu_jacobian = as_f32_matrix(&imu.state_jacobian);
    let bias_jacobian = as_f32_matrix(&bias.state_jacobian);
    let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
    for (target_row, source) in imu_jacobian.row_iter().enumerate() {
        for (block, global_offset) in [offsets.start, offsets.end].into_iter().enumerate() {
            local_jacobian
                .view_mut((target_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
        }
    }
    for (source_row, source) in bias_jacobian.row_iter().enumerate() {
        for (block, global_offset) in [offsets.start, offsets.end].into_iter().enumerate() {
            local_jacobian
                .view_mut((9 + source_row, block * AOM_NAV_DOF), (1, AOM_NAV_DOF))
                .copy_from(&source.columns(global_offset, AOM_NAV_DOF));
        }
    }
    let mut local_residual = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
    local_residual
        .rows_mut(0, 9)
        .copy_from(&as_f32_vector(&imu.residual));
    local_residual
        .rows_mut(9, 6)
        .copy_from(&as_f32_vector(&bias.residual));
    let local_step = DVector::from_iterator(
        IMU_LOCAL_COLS,
        (0..IMU_LOCAL_COLS).map(|local_column| {
            let block = local_column / AOM_NAV_DOF;
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            state_step[global_offset + local_column % AOM_NAV_DOF]
        }),
    );
    let increment = eigen_imu_15x30_gemv_f32(&local_jacobian, &local_step);
    Some(eigen_imu_15_model_dot_f32(&increment, &local_residual))
}

/// Reproduce the pinned Eigen column-major `15x30 * Vector30f` kernel used
/// inside `ImuBlock<float>::backSubstitute`. The native kernel vectorizes
/// across the 15 output rows, so every output lane accumulates columns 0..29
/// with one FMA per column; it does not horizontally reduce input packets.
#[inline]
fn eigen_imu_15x30_gemv_f32(jacobian: &DMatrix<f32>, step: &DVector<f32>) -> DVector<f32> {
    assert_eq!(jacobian.shape(), (IMU_LOCAL_ROWS, IMU_LOCAL_COLS));
    assert_eq!(step.len(), IMU_LOCAL_COLS);
    let mut result = DVector::<f32>::zeros(IMU_LOCAL_ROWS);
    for column in 0..IMU_LOCAL_COLS {
        for row in 0..IMU_LOCAL_ROWS {
            result[row] = jacobian[(row, column)].mul_add(step[column], result[row]);
        }
    }
    result
}

/// Reproduce the pinned 15-element Eigen dot in `ImuBlock::backSubstitute`.
/// Elements 0..7 are formed lane-wise and reduced with AVX Packet8 predux;
/// the seven remaining elements are then folded into that scalar with FMA.
#[inline]
fn eigen_imu_15_model_dot_f32(increment: &DVector<f32>, residual: &DVector<f32>) -> f32 {
    assert_eq!(increment.len(), IMU_LOCAL_ROWS);
    assert_eq!(residual.len(), IMU_LOCAL_ROWS);
    let lanes: [f32; 8] = std::array::from_fn(|index| {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        increment[index] * right
    });
    let mut value = eigen_predux8_f32(lanes);
    for index in 8..IMU_LOCAL_ROWS {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        value = increment[index].mul_add(right, value);
    }
    value
}

pub(super) fn prior39_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    assert_eq!(j.shape(), (39, 39));
    assert_eq!(rhs.len(), 39);
    assert_eq!(step.len(), 39);
    let mut inc = [0.0_f32; 39];
    for row in 0..39 {
        for col in 0..39 {
            inc[row] = j[(row, col)].mul_add(step[col], inc[row]);
        }
    }
    let right: [f32; 39] = std::array::from_fn(|i| 0.5_f32.mul_add(inc[i], rhs[i]));
    let products: [f32; 32] = std::array::from_fn(|i| -inc[i] * right[i]);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        products[i] + ((products[i + 16] + products[i + 24]) + products[i + 8])
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut total = (half[0] + half[2]) + (half[1] + half[3]);
    // Native 39f0dd half FMA, then 39f0e3 vfnmadd231ss.
    for i in 32..39 {
        total = (-inc[i]).mul_add(right[i], total);
    }
    total
}

/// Reproduce Eigen's AVX inner-product kernel for the 45-row marginal prior.
/// The first and fifth packets are combined with FMA, packets two through four
/// use Eigen's fixed addition tree, and rows 40..44 form the scalar FMA tail.
pub(super) fn prior45_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    assert_eq!(j.shape(), (45, 45));
    assert_eq!(rhs.len(), 45);
    assert_eq!(step.len(), 45);
    let mut inc = [0.0_f32; 45];
    for row in 0..45 {
        for col in 0..45 {
            inc[row] = j[(row, col)].mul_add(step[col], inc[row]);
        }
    }
    let right: [f32; 45] = std::array::from_fn(|i| 0.5_f32.mul_add(inc[i], rhs[i]));
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_and_fifth = (-inc[i + 32]).mul_add(right[i + 32], -inc[i] * right[i]);
        let middle = (-inc[i + 8] * right[i + 8])
            + ((-inc[i + 16] * right[i + 16]) + (-inc[i + 24] * right[i + 24]));
        first_and_fifth + middle
    });
    let mut total = eigen_predux8_f32(lanes);
    for i in 40..45 {
        total = (-inc[i]).mul_add(right[i], total);
    }
    total
}

pub(super) fn prior_model_f32(j: &DMatrix<f32>, rhs: &DVector<f32>, step: &DVector<f32>) -> f32 {
    match j.shape() {
        (39, 39) => prior39_model_f32(j, rhs, step),
        (45, 45) => prior45_model_f32(j, rhs, step),
        shape => panic!("unsupported marginal-prior model shape {shape:?}"),
    }
}

/// Compact only explicitly mapped prior columns; other shapes use the generic evaluator.
pub(super) fn prior_model_inputs_f32(
    factor: &WhitenedFactorRowStack,
    step: &DVector<f64>,
) -> Option<(DMatrix<f32>, DVector<f32>, DVector<f32>)> {
    let size = factor.rows();
    if factor.kind != FactorKind::Prior
        || !matches!(size, 39 | 45)
        || factor.landmark_jacobian.ncols() != 0
    {
        return None;
    }
    let columns = factor.prior_state_columns.as_ref()?;
    if columns.len() != size || factor.state_jacobian.ncols() != step.len() {
        return None;
    }
    for (i, &column) in columns.iter().enumerate() {
        if column >= step.len() || columns[..i].contains(&column) {
            return None;
        }
    }
    for col in 0..step.len() {
        if !columns.contains(&col) && factor.state_jacobian.column(col).iter().any(|x| *x != 0.0) {
            return None;
        }
    }
    let j = DMatrix::from_fn(size, size, |row, col| {
        factor.state_jacobian[(row, columns[col])] as f32
    });
    let rhs = as_f32_vector(&factor.residual);
    let compact_step = DVector::from_iterator(size, columns.iter().map(|&col| step[col] as f32));
    Some((j, rhs, compact_step))
}

fn diagnostic_prior39_model_input(
    factor: &WhitenedFactorRowStack,
    step: &DVector<f64>,
) -> Option<serde_json::Value> {
    let (j, rhs, compact_step) = prior_model_inputs_f32(factor, step)?;
    let columns = factor.prior_state_columns.as_ref()?;
    let candidate = prior_model_f32(&j, &rhs, &compact_step);
    Some(serde_json::json!({
        "columns": columns,
        "jacobian_column_major_bits": j.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "rhs_bits": rhs.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "step_bits": compact_step.iter().map(|x| format!("{:08x}", x.to_bits())).collect::<Vec<_>>(),
        "candidate_bits": format!("{:08x}", candidate.to_bits()),
    }))
}

/// Observation-only per-factor decomposition of the current model evaluator.
/// This never supplies LM's acceptance value or changes its factor order.
pub(crate) fn diagnostic_model_decrease_parts_f32(
    factors: &[WhitenedFactorRowStack],
    step: &DVector<f64>,
    actual: f64,
) -> Option<serde_json::Value> {
    let mut parts = Vec::with_capacity(factors.len());
    let mut original = 0.0_f32;
    let mut visual = 0.0_f32;
    let mut non_prior = 0.0_f32;
    let mut prior = 0.0_f32;
    let step_f32 = as_f32_vector(step);
    let mut paired_bias_index = None;
    for (index, factor) in factors.iter().enumerate() {
        let (value, paired_with_previous) = if paired_bias_index == Some(index) {
            paired_bias_index = None;
            (0.0_f32, true)
        } else if factor.kind == FactorKind::Imu {
            if let Some(bias) = factors.get(index + 1) {
                if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step_f32) {
                    paired_bias_index = Some(index + 1);
                    (-dot, false)
                } else {
                    (
                        model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                        false,
                    )
                }
            } else {
                (
                    model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                    false,
                )
            }
        } else {
            (
                model_cost_decrease_f32(std::slice::from_ref(factor), step, 1e-10)? as f32,
                false,
            )
        };
        if factor.kind != FactorKind::Prior {
            original += value;
        }
        if factor.kind == FactorKind::Visual {
            visual += value;
        }
        if factor.kind == FactorKind::Prior {
            prior += value;
        } else {
            non_prior += value;
        }
        parts.push(serde_json::json!({
            "factor_index": index,
            "kind": full70_factor_kind_name(factor.kind),
            "rows": factor.rows(),
            "decrease_bits": format!("{:08x}", value.to_bits()),
            "original_cumulative_bits": format!("{:08x}", original.to_bits()),
            "prior39_input": diagnostic_prior39_model_input(factor, step),
            "paired_with_previous": paired_with_previous,
        }));
    }
    // Match LinearizationAbsQR::backSubstitute: visual reduction, IMU
    // blocks, then the marginal-prior contribution.
    original += prior;
    Some(serde_json::json!({
        "parts": parts,
        "original_total_bits": format!("{:08x}", original.to_bits()),
        "direct_total_bits": format!("{:08x}", (actual as f32).to_bits()),
        "reconstruction_exact": original.to_bits() == (actual as f32).to_bits(),
        "visual_bits": format!("{:08x}", visual.to_bits()),
        "non_prior_bits": format!("{:08x}", non_prior.to_bits()),
        "prior_bits": format!("{:08x}", prior.to_bits()),
        "prior_last_candidate_bits": format!("{:08x}", (non_prior + prior).to_bits()),
    }))
}

/// Test-only reduced normal-system model-decrease candidate.
///
/// This is deliberately not used by either LM loop.  It evaluates the usual
/// reduced quadratic, `-(bᵀs + 1/2 sᵀHs)`, using the already-audited f32
/// row-major GEMV schedule.  The production model evaluator above operates
/// on the complete transformed factor rows and therefore also includes each
/// landmark block's Q1 contribution.  Keeping this candidate private makes
/// that semantic distinction executable without changing the accept/reject
/// contract.
#[cfg(test)]
pub(super) fn reduced_model_cost_decrease_f32(
    reduced: &ReducedNormalSystemF32,
    state_step: &DVector<f64>,
) -> Option<f64> {
    if reduced.h.nrows() != reduced.h.ncols()
        || reduced.h.nrows() != reduced.b.len()
        || reduced.h.nrows() != state_step.len()
    {
        return None;
    }
    let step = as_f32_vector(state_step);
    let h_step = eigen_row_major_gemv_f32(&reduced.h, &step);
    let mut linear = 0.0_f32;
    let mut quadratic = 0.0_f32;
    for index in 0..step.len() {
        linear = step[index].mul_add(reduced.b[index], linear);
        quadratic = step[index].mul_add(h_step[index], quadratic);
    }
    let decrease = -(linear + 0.5_f32 * quadratic);
    decrease.is_finite().then_some(decrease as f64)
}

/// Test-only reduced model candidate with the visual Q1 residual constant.
///
/// For a full-rank visual ABS-QR block, landmark elimination makes its Q1
/// state increment exactly `-Q1^T r`; its contribution to the model decrease
/// is therefore `+0.5 * ||Q1^T r||^2`, independent of the solved state step.
/// The reduced H/b system supplies the Q2 and non-visual terms.  This helper
/// intentionally remains disconnected from both LM loops until its arithmetic
/// and accumulation order are proven against the complete transformed-row
/// evaluator.
#[cfg(test)]
pub(super) fn reduced_model_cost_decrease_with_q1_constant_f32(
    reduced: &ReducedNormalSystemF32,
    q1_entries: &[CompactLandmarkBackSubstitutionF32],
    state_step: &DVector<f64>,
) -> Option<f64> {
    let reduced_decrease = reduced_model_cost_decrease_f32(reduced, state_step)? as f32;
    let mut q1_constant = 0.0_f32;
    for entry in q1_entries {
        if !entry.eligible || entry.rank != entry.landmark_cols {
            return None;
        }
        let mut squared_norm = 0.0_f32;
        for row in 0..entry.q1_residual_len() {
            let residual = entry.q1_residual_value(row);
            squared_norm = residual.mul_add(residual, squared_norm);
        }
        let factor_constant = 0.5_f32 * squared_norm;
        q1_constant = add_f32_exact(q1_constant, factor_constant);
    }
    let decrease = add_f32_exact(reduced_decrease, q1_constant);
    decrease.is_finite().then_some(decrease as f64)
}

/// Test-only payload containing both sides of one already-computed landmark
/// QR.  The helper below deliberately consumes these values without calling
/// [`LandmarkHouseholderF32::factor`] again: it is the primitive we would need
/// if the clean reducer retained Q2 rows for the later model-decrease pass.
#[cfg(test)]
pub(super) struct QrModelReusePayloadF32 {
    pub(super) q1: CompactLandmarkBackSubstitutionF32,
    pub(super) q2_state: DMatrix<f32>,
    pub(super) q2_residual: DVector<f32>,
}

/// Evaluate the complete transformed-row dot term from a retained Q1/Q2 QR
/// payload.  This mirrors the visual branch of `model_cost_decrease_f32`,
/// including its pinned Eigen GEMV, triangular solve, R*landmark-inc product,
/// row order, and final scalar-FMA dot, but performs no QR.
pub(super) fn qr_payload_model_dot_reuse_f32<Q1: QrModelQ1ViewF32>(
    q1: &Q1,
    q2_state: &DMatrix<f32>,
    q2_residual: &DVector<f32>,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f32> {
    let n = q1.landmark_cols();
    let state_cols = q1.state_cols();
    if !q1.eligible()
        || q1.rank() != n
        || n == 0
        || n > 3
        || state_cols != state_step.len()
        || q2_state.ncols() != state_cols
        || q2_residual.len() != q2_state.nrows()
    {
        return None;
    }

    let step = as_f32_vector(state_step);
    let q2_rows = q2_state.nrows();
    let rows = n + q2_rows;
    let transformed_state = DMatrix::from_fn(rows, state_cols, |row, column| {
        if row < n {
            q1.q1_state_value(row, column)
        } else {
            q2_state[(row - n, column)]
        }
    });
    let transformed_residual = DVector::from_iterator(
        rows,
        (0..rows).map(|row| {
            if row < n {
                q1.q1_residual_value(row)
            } else {
                q2_residual[row - n]
            }
        }),
    );
    let upper_r = DMatrix::from_fn(n, n, |row, column| q1.upper_r_value(row, column));

    let mut qj_inc = eigen_row_major_gemv_f32(&transformed_state, &step);
    let mut rhs = transformed_residual.rows(0, n).into_owned();
    for row in 0..n {
        rhs[row] += qj_inc[row];
    }
    let threshold = tolerance as f32;
    let mut landmark_inc = DVector::<f32>::zeros(n);
    if n == 3 {
        let d2 = upper_r[(2, 2)];
        let d1 = upper_r[(1, 1)];
        let d0 = upper_r[(0, 0)];
        if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
            return None;
        }
        let x2 = rhs[2] / d2;
        let x1 = (-upper_r[(1, 2)]).mul_add(x2, rhs[1]) / d1;
        let row0_dot = upper_r[(0, 2)].mul_add(x2, upper_r[(0, 1)] * x1);
        let x0 = (rhs[0] - row0_dot) / d0;
        landmark_inc[0] = -x0;
        landmark_inc[1] = -x1;
        landmark_inc[2] = -x2;
    } else {
        rhs = -rhs;
        for row in (0..n).rev() {
            let mut value = rhs[row];
            for column in (row + 1)..n {
                value -= upper_r[(row, column)] * landmark_inc[column];
            }
            let diagonal = upper_r[(row, row)];
            if diagonal.abs() <= threshold {
                return None;
            }
            landmark_inc[row] = value / diagonal;
        }
    }
    let q1_landmark_inc = &upper_r * &landmark_inc;
    for row in 0..n {
        qj_inc[row] += q1_landmark_inc[row];
    }
    let dot = eigen_visual_model_dot_f32(&qj_inc, &transformed_residual);
    dot.is_finite().then_some(dot)
}

#[cfg(test)]
fn qr_payload_model_dot_f32(
    payload: &QrModelReusePayloadF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f32> {
    qr_payload_model_dot_reuse_f32(
        &payload.q1,
        &payload.q2_state,
        &payload.q2_residual,
        state_step,
        tolerance,
    )
}

#[cfg(test)]
pub(super) fn model_cost_decrease_from_qr_payload_f32(
    payload: &QrModelReusePayloadF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    let dot = qr_payload_model_dot_f32(payload, state_step, tolerance)?;
    let mut decrease = 0.0_f32;
    decrease -= dot;
    decrease.is_finite().then_some(decrease as f64)
}

#[cfg(test)]
pub(super) enum ModelReusePayloadF32 {
    Visual(QrModelReusePayloadF32),
    Plain {
        kind: FactorKind,
        state: DMatrix<f32>,
        residual: DVector<f32>,
    },
}

#[cfg(test)]
pub(super) fn model_cost_decrease_from_payloads_f32(
    payloads: &[ModelReusePayloadF32],
    factors: &[WhitenedFactorRowStack],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<f64> {
    if payloads.len() != factors.len() {
        return None;
    }
    let step = as_f32_vector(state_step);
    let mut decrease = 0.0_f32;
    let mut deferred_prior = Vec::new();
    let mut paired_bias_index = None;
    for (index, payload) in payloads.iter().enumerate() {
        if paired_bias_index == Some(index) {
            paired_bias_index = None;
            continue;
        }
        let factor = &factors[index];
        if factor.kind == FactorKind::Imu {
            if let Some(bias) = factors.get(index + 1) {
                if let Some(dot) = imu_bias_pair_model_dot_f32(factor, bias, &step) {
                    decrease -= dot;
                    paired_bias_index = Some(index + 1);
                    continue;
                }
            }
        }
        let (dot, defer) = match payload {
            ModelReusePayloadF32::Visual(payload) => (
                qr_payload_model_dot_f32(payload, state_step, tolerance)?,
                false,
            ),
            ModelReusePayloadF32::Plain {
                kind,
                state,
                residual,
            } => {
                if state.ncols() != step.len() || state.nrows() != residual.len() {
                    return None;
                }
                let increment = state * &step;
                (
                    increment.dot(&(0.5_f32 * &increment + residual)),
                    *kind == FactorKind::Prior,
                )
            }
        };
        if defer {
            deferred_prior.push(-dot);
        } else {
            decrease -= dot;
        }
    }
    for contribution in deferred_prior {
        decrease += contribution;
    }
    decrease.is_finite().then_some(decrease as f64)
}
