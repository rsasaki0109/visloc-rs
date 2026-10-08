//! Eigen-ordered f32 kernels: local IMU blocks, gram / gemv products and packet reductions.

use super::*;

pub(super) const IMU_LOCAL_ROWS: usize = 15;
pub(super) const IMU_LOCAL_COLS: usize = AOM_NAV_DOF * 2;

/// Multiply one local IMU/bias row stack using Eigen's fixed 15x30 dynamic
/// product schedule.
///
/// `ImuBlock::add_dense_H_b` materializes a 15-row by 30-column `MatrixXf`
/// before evaluating `Jp.transpose() * Jp` and `Jp.transpose() * r`.  The
/// product is not equivalent, bit-for-bit, to reducing a zero-padded global
/// matrix: the AVX GEBP kernel dispatches output rows 0..23, 24..27 and
/// 28..29 through different packet tails.  Keep this helper independent of
/// the production reducer until the active-window metadata is available.
///
/// The matrix and vector are intentionally dynamic at the boundary to match
/// upstream's `MatrixXf`/`VectorXf` expressions.  Shape assertions make an
/// accidental visual/prior row stack fail at the call site rather than being
/// silently interpreted as an IMU packet.
#[inline]
pub(super) fn local_imu_h_b_15x30(
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
) -> (DMatrix<f32>, DVector<f32>) {
    assert_eq!(jacobian.nrows(), IMU_LOCAL_ROWS);
    assert_eq!(jacobian.ncols(), IMU_LOCAL_COLS);
    assert_eq!(residual.len(), IMU_LOCAL_ROWS);

    let mut local_h = DMatrix::<f32>::zeros(IMU_LOCAL_COLS, IMU_LOCAL_COLS);
    for row in 0..IMU_LOCAL_COLS {
        for column in 0..IMU_LOCAL_COLS {
            let value = if (24..28).contains(&row) && column < 28 {
                // The 24..27 half-packet tail has two Packet4 C/D
                // accumulators.  Reduce the first eight depths before the
                // scalar FMA tail, preserving the source association.
                let mut even = 0.0_f32;
                let mut odd = 0.0_f32;
                for depth in (0..8).step_by(2) {
                    even = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even);
                    odd = jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd);
                }
                let mut value = add_f32_exact(even, odd);
                for depth in 8..IMU_LOCAL_ROWS {
                    value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
                }
                value
            } else if row >= 28 && column < 28 {
                // The remaining-row 1x4 kernel uses SwappedTraits with
                // spk=2.  Four even and four odd accumulators are reduced
                // as (C0+C1)+(C2+C3), then paired depths 8..13 are fused
                // into the corresponding lane.  Depth 14 is the scalar
                // FMA after the parity reduction.
                let mut even = [0.0_f32; 4];
                let mut odd = [0.0_f32; 4];
                for (lane, depth) in (0..8).step_by(2).enumerate() {
                    even[lane] = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], 0.0);
                    odd[lane] =
                        jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], 0.0);
                }
                let mut even_sum = add_f32_exact(
                    add_f32_exact(even[0], even[1]),
                    add_f32_exact(even[2], even[3]),
                );
                let mut odd_sum =
                    add_f32_exact(add_f32_exact(odd[0], odd[1]), add_f32_exact(odd[2], odd[3]));
                for depth in (8..14).step_by(2) {
                    even_sum = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even_sum);
                    odd_sum =
                        jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd_sum);
                }
                let reduced = add_f32_exact(even_sum, odd_sum);
                jacobian[(14, row)].mul_add(jacobian[(14, column)], reduced)
            } else {
                // Full packet rows and the X1 columns retain direct depth
                // order, including the scalar tail.
                let mut value = 0.0_f32;
                for depth in 0..IMU_LOCAL_ROWS {
                    value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
                }
                value
            };
            local_h[(row, column)] = value;
        }
    }

    let mut local_b = DVector::<f32>::zeros(IMU_LOCAL_COLS);
    for column in 0..IMU_LOCAL_COLS {
        let mut packet_lanes = [0.0_f32; 8];
        for depth in 0..8 {
            packet_lanes[depth] = mul_f32_exact(jacobian[(depth, column)], residual[depth]);
        }

        // AVX Packet8f predux: q0=l0+l4, q1=l1+l5, q2=l2+l6,
        // q3=l3+l7; (q0+q2)+(q1+q3).
        let q0 = add_f32_exact(packet_lanes[0], packet_lanes[4]);
        let q1 = add_f32_exact(packet_lanes[1], packet_lanes[5]);
        let q2 = add_f32_exact(packet_lanes[2], packet_lanes[6]);
        let q3 = add_f32_exact(packet_lanes[3], packet_lanes[7]);
        let low = add_f32_exact(q0, q2);
        let high = add_f32_exact(q1, q3);
        let mut value = add_f32_exact(low, high);

        // Eigen's dynamic cleanup emits ordinary multiply/add for depths
        // 8..11, followed by scalar FMAs for depths 12..14.
        for depth in 8..12 {
            value = add_f32_exact(
                value,
                mul_f32_exact(jacobian[(depth, column)], residual[depth]),
            );
        }
        for depth in 12..IMU_LOCAL_ROWS {
            value = jacobian[(depth, column)].mul_add(residual[depth], value);
        }
        local_b[column] = value;
    }

    (local_h, local_b)
}

/// Add one local 30x30/30 IMU product to the global state accumulator.
///
/// The four matrix blocks and two vector blocks are visited in upstream's
/// order (`H00`, `H10`, `H01`, `H11`, `b0`, `b1`).  In particular, `H10` is
/// copied as-is into the `(end,start)` block; it must not be transposed to
/// repair the mathematical symmetry because the local packet schedule can
/// produce distinct low bits in the two cross blocks.
#[inline]
pub(super) fn scatter_local_imu_h_b_15x30(
    accumulator_h: &mut DMatrix<f32>,
    accumulator_b: &mut DVector<f32>,
    local_h: &DMatrix<f32>,
    local_b: &DVector<f32>,
    offsets: ImuLinkOffsets,
) -> Result<(), ImuReductionError> {
    if local_h.nrows() != IMU_LOCAL_COLS
        || local_h.ncols() != IMU_LOCAL_COLS
        || local_b.len() != IMU_LOCAL_COLS
    {
        return Err(ImuReductionError::InvalidLocalProduct {
            h_rows: local_h.nrows(),
            h_cols: local_h.ncols(),
            b_len: local_b.len(),
        });
    }
    if accumulator_h.nrows() != accumulator_h.ncols()
        || accumulator_b.len() != accumulator_h.nrows()
    {
        return Err(ImuReductionError::InvalidAccumulator {
            h_rows: accumulator_h.nrows(),
            h_cols: accumulator_h.ncols(),
            b_len: accumulator_b.len(),
        });
    }
    validate_imu_offsets(0, offsets, accumulator_h.nrows())?;

    let add_matrix_block = |accumulator: &mut DMatrix<f32>,
                            local_row_offset: usize,
                            global_row_offset: usize,
                            local_column_offset: usize,
                            global_column_offset: usize| {
        for row in 0..AOM_NAV_DOF {
            for column in 0..AOM_NAV_DOF {
                accumulator[(global_row_offset + row, global_column_offset + column)] +=
                    local_h[(local_row_offset + row, local_column_offset + column)];
            }
        }
    };

    // Keep these calls separate: block order is part of the f32 contract.
    add_matrix_block(accumulator_h, 0, offsets.start, 0, offsets.start);
    add_matrix_block(accumulator_h, AOM_NAV_DOF, offsets.end, 0, offsets.start);
    add_matrix_block(accumulator_h, 0, offsets.start, AOM_NAV_DOF, offsets.end);
    add_matrix_block(
        accumulator_h,
        AOM_NAV_DOF,
        offsets.end,
        AOM_NAV_DOF,
        offsets.end,
    );
    for row in 0..AOM_NAV_DOF {
        accumulator_b[offsets.start + row] += local_b[row];
    }
    for row in 0..AOM_NAV_DOF {
        accumulator_b[offsets.end + row] += local_b[AOM_NAV_DOF + row];
    }
    Ok(())
}

#[inline]
pub(super) fn add_f32_exact(left: f32, right: f32) -> f32 {
    let result = left + right;
    result
}

#[inline(never)]
pub(super) fn mul_f32_exact(left: f32, right: f32) -> f32 {
    let result = left * right;
    result
}

#[inline]
pub(super) fn eigen_predux8_f32(lanes: [f32; 8]) -> f32 {
    let q0 = add_f32_exact(lanes[0], lanes[4]);
    let q1 = add_f32_exact(lanes[1], lanes[5]);
    let q2 = add_f32_exact(lanes[2], lanes[6]);
    let q3 = add_f32_exact(lanes[3], lanes[7]);
    add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3))
}

#[inline]
fn eigen_predux4_f32(lanes: [f32; 4]) -> f32 {
    let q0 = add_f32_exact(lanes[0], lanes[2]);
    let q1 = add_f32_exact(lanes[1], lanes[3]);
    add_f32_exact(q0, q1)
}

#[inline]
fn eigen_q2_dot_fma_f32(jacobian: &DMatrix<f32>, row: usize, column: usize) -> f32 {
    let mut value = 0.0_f32;
    for depth in 0..jacobian.nrows() {
        value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
    }
    value
}

#[inline]
fn eigen_q2_dot_packet_parity_f32(jacobian: &DMatrix<f32>, row: usize, column: usize) -> f32 {
    // `lhs_process_one_packet` (and its half-packet specialization) keeps
    // two accumulators for each four-column RHS panel.  The peeled depth is
    // processed as K=0,2,... into C and K=1,3,... into D; only after the
    // packet loop does Eigen add C+D and visit the scalar remainder.
    let peeled_depth = jacobian.nrows() / 8 * 8;
    let mut even = 0.0_f32;
    let mut odd = 0.0_f32;
    for depth in (0..peeled_depth).step_by(2) {
        even = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], even);
        odd = jacobian[(depth + 1, row)].mul_add(jacobian[(depth + 1, column)], odd);
    }
    let mut value = add_f32_exact(even, odd);
    for depth in peeled_depth..jacobian.nrows() {
        value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
    }
    value
}

/// Reproduce Eigen's AVX2 `gebp` reduction for a dynamic f32 `JᵀJ`.
///
/// Eigen's packet kernel has `mr=24`, `nr=4`, and packet width eight.  The
/// 24-row and 16-row panels accumulate each depth in order; the 8-row and
/// 4-row tails use the packet kernel's doubled C/D accumulators for complete
/// four-column RHS panels.  A non-multiple-of-four RHS column remains on the
/// scalar one-column path and therefore uses the ordered FMA dot product.
#[inline]
pub(super) fn eigen_q2_gram_f32(jacobian: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = jacobian.ncols();
    let mut product = DMatrix::<f32>::zeros(columns, columns);
    if columns == 0 {
        return product;
    }

    let peeled_mc3 = columns / 24 * 24;
    let peeled_mc2 = peeled_mc3 + (columns - peeled_mc3) / 16 * 16;
    let peeled_mc1 = peeled_mc2 + (columns - peeled_mc2) / 8 * 8;
    let peeled_mc_half = peeled_mc1 + (columns - peeled_mc1) / 4 * 4;
    let packet_columns = columns / 4 * 4;

    for row in 0..peeled_mc3 {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    for row in peeled_mc3..peeled_mc2 {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    for row in peeled_mc2..peeled_mc1 {
        for column in 0..columns {
            product[(row, column)] = if column < packet_columns {
                eigen_q2_dot_packet_parity_f32(jacobian, row, column)
            } else {
                eigen_q2_dot_fma_f32(jacobian, row, column)
            };
        }
    }
    for row in peeled_mc1..peeled_mc_half {
        for column in 0..columns {
            product[(row, column)] = if column < packet_columns {
                eigen_q2_dot_packet_parity_f32(jacobian, row, column)
            } else {
                eigen_q2_dot_fma_f32(jacobian, row, column)
            };
        }
    }
    for row in peeled_mc_half..columns {
        for column in 0..columns {
            product[(row, column)] = eigen_q2_dot_fma_f32(jacobian, row, column);
        }
    }
    product
}

// Per-landmark LM product, not the stacked Q2 exporter. Pinned Eigen's
// one/half-packet four-column panels split peeled depth into C/D, merge,
// then process the remaining depth. Keep other panel kernels unchanged.
/// A visual landmark's `JᵀJ` contribution, either over the full state width
/// or restricted to the columns where the projected Jacobian is nonzero.
pub(super) enum VisualGramContribution {
    Dense(DMatrix<f32>),
    /// `columns[a]` is the full state column of compact row/column `a`.
    Sparse {
        columns: Vec<usize>,
        gram: DMatrix<f32>,
    },
}

impl VisualGramContribution {
    /// `h += contribution`. The dense form adds every entry. The sparse form
    /// adds only the support block: every other entry of the dense
    /// contribution is an exact `+0.0` (each product in its FMA chain
    /// involves a zero column), and `h + 0.0 == h`, so the result is the same.
    pub(super) fn add_to(&self, h: &mut DMatrix<f32>) {
        match self {
            Self::Dense(dense) => *h += dense,
            Self::Sparse { columns, gram } => {
                for (b, &column) in columns.iter().enumerate() {
                    for (a, &row) in columns.iter().enumerate() {
                        h[(row, column)] += gram[(a, b)];
                    }
                }
            }
        }
    }
}

/// Support-restricted counterpart of [`eigen_visual_gram_packet_tail_f32`].
///
/// A visual factor's projected Jacobian is nonzero only in the state columns
/// of the frames that observe the landmark, but the dense kernel forms the
/// full `(state x state)` product. This forms `JcᵀJc` over the nonzero
/// columns `S` with the same nalgebra/matrixmultiply product. Each gemm entry
/// is an ordered FMA chain over the depth that does not depend on the
/// entry's position, so `(JcᵀJc)[a, b] == (JᵀJ)[S[a], S[b]]` bit for bit. The
/// Eigen-parity overrides of the dense kernel are applied using the entries'
/// full-width row and column indices. With five or fewer support columns
/// nalgebra switches to a different product path, so that case (and an
/// all-dense support) uses the dense kernel unchanged.
pub(super) fn eigen_visual_gram_packet_tail_sparse_f32(
    jacobian: &DMatrix<f32>,
) -> VisualGramContribution {
    let columns = jacobian.ncols();
    let support = (0..columns)
        .filter(|&column| jacobian.column(column).iter().any(|value| *value != 0.0))
        .collect::<Vec<_>>();
    if support.len() <= 5 || support.len() == columns || jacobian.nrows() <= 5 {
        return VisualGramContribution::Dense(eigen_visual_gram_packet_tail_f32(jacobian));
    }
    let compact = DMatrix::from_fn(jacobian.nrows(), support.len(), |row, a| {
        jacobian[(row, support[a])]
    });
    let mut gram = compact.transpose() * &compact;
    let end24 = columns / 24 * 24;
    let end16 = end24 + (columns - end24) / 16 * 16;
    let end8 = end16 + (columns - end16) / 8 * 8;
    let end4 = end8 + (columns - end8) / 4 * 4;
    let parity_cols = columns / 4 * 4;
    for (a, &row) in support.iter().enumerate() {
        if !(end16..end4).contains(&row) {
            continue;
        }
        for (b, &column) in support.iter().enumerate() {
            if column < parity_cols {
                gram[(a, b)] = eigen_q2_dot_packet_parity_f32(jacobian, row, column);
            }
        }
    }
    VisualGramContribution::Sparse {
        columns: support,
        gram,
    }
}

pub(super) fn eigen_visual_gram_packet_tail_f32(jacobian: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = jacobian.ncols();
    let mut product = jacobian.transpose() * jacobian;
    let end24 = columns / 24 * 24;
    let end16 = end24 + (columns - end24) / 16 * 16;
    let end8 = end16 + (columns - end16) / 8 * 8;
    let end4 = end8 + (columns - end8) / 4 * 4;
    for row in end16..end4 {
        for col in 0..columns / 4 * 4 {
            product[(row, col)] = eigen_q2_dot_packet_parity_f32(jacobian, row, col);
        }
    }
    product
}

#[inline]
fn eigen_q2_packet_dot_fma_f32(
    jacobian: &DMatrix<f32>,
    rhs: &DVector<f32>,
    row: usize,
    peeled_depth: usize,
) -> f32 {
    let mut lanes = [0.0_f32; 8];
    for depth in (0..peeled_depth).step_by(8) {
        for lane in 0..8 {
            lanes[lane] = jacobian[(depth + lane, row)].mul_add(rhs[depth + lane], lanes[lane]);
        }
    }
    eigen_predux8_f32(lanes)
}

/// Reproduce Eigen's row-major `general_matrix_vector_product` for Q2
/// `Jᵀr`.  Eight-, four-, and two-row output groups use Packet8 accumulators
/// followed by the AVX horizontal reduction and an explicitly non-fused
/// scalar depth tail.  A final one-row group additionally uses Packet4 for
/// the remaining complete four-depth block, exactly as Eigen's `HasHalf`
/// branch does.
#[inline]
pub(super) fn eigen_q2_transpose_gemv_f32(
    jacobian: &DMatrix<f32>,
    rhs: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(jacobian.nrows(), rhs.len());
    let rows = jacobian.ncols();
    let depth = jacobian.nrows();
    let peeled_depth = depth / 8 * 8;
    let half_depth = depth / 4 * 4;
    let mut result = DVector::<f32>::zeros(rows);

    let mut output_row = 0;
    while output_row + 8 <= rows {
        for row in output_row..output_row + 8 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 8;
    }
    while output_row + 4 <= rows {
        for row in output_row..output_row + 4 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 4;
    }
    while output_row + 2 <= rows {
        for row in output_row..output_row + 2 {
            let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, row, peeled_depth);
            for depth in peeled_depth..jacobian.nrows() {
                value = add_f32_exact(value, mul_f32_exact(jacobian[(depth, row)], rhs[depth]));
            }
            result[row] = value;
        }
        output_row += 2;
    }
    while output_row < rows {
        let mut value = eigen_q2_packet_dot_fma_f32(jacobian, rhs, output_row, peeled_depth);
        if half_depth > peeled_depth {
            let mut lanes = [0.0_f32; 4];
            for depth in (peeled_depth..half_depth).step_by(4) {
                for lane in 0..4 {
                    lanes[lane] = jacobian[(depth + lane, output_row)]
                        .mul_add(rhs[depth + lane], lanes[lane]);
                }
            }
            value = add_f32_exact(value, eigen_predux4_f32(lanes));
        }
        for depth in half_depth..jacobian.nrows() {
            value = add_f32_exact(
                value,
                mul_f32_exact(jacobian[(depth, output_row)], rhs[depth]),
            );
        }
        result[output_row] = value;
        output_row += 1;
    }
    result
}

/// Accumulate `J.transpose() * J` in the source ABS_QR order.  Only an
/// explicitly tagged nine-row IMU block uses the currently pinned local
/// packet schedule. All other shapes—including a nine-row prior—use the
/// generic dynamic path. The 15-row IMU+bias product is intentionally routed
/// through that generic path until its Eigen packet tree has its own oracle.
#[inline]
pub(super) fn accumulate_gram_f32_eigen(
    accumulator: &mut DMatrix<f32>,
    jacobian: &DMatrix<f32>,
    exact_imu_rows: bool,
) {
    assert_eq!(accumulator.nrows(), accumulator.ncols());
    assert_eq!(accumulator.ncols(), jacobian.ncols());
    if !exact_imu_rows || jacobian.nrows() != 9 {
        *accumulator += jacobian.transpose() * jacobian;
        return;
    }
    for row in 0..jacobian.ncols() {
        for column in 0..jacobian.ncols() {
            let mut value = 0.0_f32;
            for depth in 0..9 {
                value = jacobian[(depth, row)].mul_add(jacobian[(depth, column)], value);
            }
            accumulator[(row, column)] += value;
        }
    }
}

/// Accumulate a stored square-root prior's compact normal product and scatter
/// it into the absolute AOM columns.  `jacobian` is the global-width view
/// carried by the factor, while `compact_state_columns` identifies the
/// columns that were present in upstream's compact `MargLinData::H`.
#[inline]
pub(super) fn accumulate_prior_gram_f32_eigen(
    accumulator: &mut DMatrix<f32>,
    jacobian: &DMatrix<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    assert_eq!(accumulator.nrows(), accumulator.ncols());
    assert_eq!(accumulator.ncols(), jacobian.ncols());
    let Some(columns) = compact_state_columns else {
        *accumulator += jacobian.transpose() * jacobian;
        return;
    };
    if columns.len() != jacobian.nrows()
        || columns.iter().any(|&column| column >= accumulator.ncols())
    {
        // Keep malformed metadata from changing the behavior of legacy
        // callers.  The active-window prior builder only attaches a complete
        // compact-to-global map.
        *accumulator += jacobian.transpose() * jacobian;
        return;
    }

    // Materialize the compact column-major MatrixXf that Eigen receives from
    // MargLinData.  This is intentionally a separate product: multiplying
    // the zero-padded AOM view changes the packet/tail traversal even though
    // all omitted entries are zero.
    let compact = DMatrix::<f32>::from_fn(jacobian.nrows(), columns.len(), |row, column| {
        jacobian[(row, columns[column])]
    });
    let compact_product = eigen_prior_compact_gram_f32(&compact);
    for local_row in 0..columns.len() {
        let global_row = columns[local_row];
        for local_column in 0..columns.len() {
            accumulator[(global_row, columns[local_column])] +=
                compact_product[(local_row, local_column)];
        }
    }
}

/// Reproduce Eigen's AVX2 `gebp` tail for the 21x21 compact prior product.
/// The main 16-row block is already equivalent to the established dynamic
/// product.  Rows 16..19 use the Packet4 half-kernel's even/odd accumulators;
/// row 20 uses the swapped 2-lane/4-column kernel, except for the final
/// scalar column (column 20), which takes the ordinary 1x1 path.
#[inline]
pub(super) fn eigen_prior_packet_tail_gram_candidate_f32(j: &DMatrix<f32>) -> DMatrix<f32> {
    let columns = j.ncols();
    let end24 = columns / 24 * 24;
    let end16 = end24 + (columns - end24) / 16 * 16;
    let end8 = end16 + (columns - end16) / 8 * 8;
    let end4 = end8 + (columns - end8) / 4 * 4;
    let depth8 = j.nrows() / 8 * 8;
    let depth2 = j.nrows() / 2 * 2;
    let mut product = j.transpose() * j;
    for row in end16..end4 {
        for col in 0..columns / 4 * 4 {
            product[(row, col)] = eigen_q2_dot_packet_parity_f32(j, row, col);
        }
    }
    for row in end4..columns {
        for col in 0..columns / 4 * 4 {
            let mut accum = [[0.0_f32; 2]; 4];
            for depth in 0..depth8 {
                let group = (depth & 7) / 2;
                let lane = depth & 1;
                accum[group][lane] = j[(depth, row)].mul_add(j[(depth, col)], accum[group][lane]);
            }
            let mut low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
            let mut high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
            for depth in (depth8..depth2).step_by(2) {
                low = j[(depth, row)].mul_add(j[(depth, col)], low);
                high = j[(depth + 1, row)].mul_add(j[(depth + 1, col)], high);
            }
            let mut value = low + high;
            for depth in depth2..j.nrows() {
                value = j[(depth, row)].mul_add(j[(depth, col)], value);
            }
            product[(row, col)] = value;
        }
    }
    product
}

pub(super) fn eigen_prior_compact_gram_f32(compact: &DMatrix<f32>) -> DMatrix<f32> {
    if matches!(
        compact.shape(),
        (33, 33) | (39, 39) | (45, 45) | (51, 51) | (57, 57)
    ) {
        return eigen_prior_packet_tail_gram_candidate_f32(compact);
    }
    if compact.shape() == (27, 27) {
        return eigen_prior27_tail_gram_f32(compact);
    }
    let mut product = compact.transpose() * compact;
    if compact.nrows() != 21 || compact.ncols() != 21 {
        return product;
    }

    for row in 16..20 {
        // The four-column panel uses the half-packet two-accumulator
        // kernel.  The final scalar RHS column is dispatched to the
        // one-column remainder kernel below.
        for column in 0..20 {
            let mut even = 0.0_f32;
            let mut odd = 0.0_f32;
            for depth in 0..16 {
                if depth & 1 == 0 {
                    even = compact[(depth, row)].mul_add(compact[(depth, column)], even);
                } else {
                    odd = compact[(depth, row)].mul_add(compact[(depth, column)], odd);
                }
            }
            let mut value = even + odd;
            for depth in 16..21 {
                value = compact[(depth, row)].mul_add(compact[(depth, column)], value);
            }
            product[(row, column)] = value;
        }
        let mut scalar = 0.0_f32;
        for depth in 0..21 {
            scalar = compact[(depth, row)].mul_add(compact[(depth, 20)], scalar);
        }
        product[(row, 20)] = scalar;
    }

    for column in 0..20 {
        let mut accum = [[0.0_f32; 2]; 4];
        for depth in 0..16 {
            let group = (depth & 7) / 2;
            let lane = depth & 1;
            accum[group][lane] =
                compact[(depth, 20)].mul_add(compact[(depth, column)], accum[group][lane]);
        }
        let low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
        let high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
        let mut merged_low = compact[(16, 20)].mul_add(compact[(16, column)], low);
        let mut merged_high = compact[(17, 20)].mul_add(compact[(17, column)], high);
        merged_low = compact[(18, 20)].mul_add(compact[(18, column)], merged_low);
        merged_high = compact[(19, 20)].mul_add(compact[(19, column)], merged_high);
        product[(20, column)] =
            compact[(20, 20)].mul_add(compact[(20, column)], merged_low + merged_high);
    }
    // Column 20 is handled by Eigen's scalar 1x1 remainder kernel, not the
    // swapped four-column panel above.
    let mut scalar = 0.0_f32;
    for depth in 0..21 {
        scalar = compact[(depth, 20)].mul_add(compact[(depth, 20)], scalar);
    }
    product[(20, 20)] = scalar;
    product
}

// Pinned prior GEMM dispatch: 2cdee0 -> 2c62f0 -> gebp 29cba0.
// The 24-row body and scalar-column remainders retain the dynamic product;
// three residual rows use the paired-depth/four-column panel reduction.
fn eigen_prior27_tail_gram_f32(compact: &DMatrix<f32>) -> DMatrix<f32> {
    assert_eq!(compact.shape(), (27, 27));
    let mut product = compact.transpose() * compact;
    for row in 24..27 {
        for column in 0..24 {
            let mut accum = [[0.0_f32; 2]; 4];
            for depth in 0..24 {
                let group = (depth & 7) / 2;
                let lane = depth & 1;
                accum[group][lane] =
                    compact[(depth, row)].mul_add(compact[(depth, column)], accum[group][lane]);
            }
            let low = (accum[0][0] + accum[1][0]) + (accum[2][0] + accum[3][0]);
            let high = (accum[0][1] + accum[1][1]) + (accum[2][1] + accum[3][1]);
            let low = compact[(24, row)].mul_add(compact[(24, column)], low);
            let high = compact[(25, row)].mul_add(compact[(25, column)], high);
            product[(row, column)] = compact[(26, row)].mul_add(compact[(26, column)], low + high);
        }
    }
    product
}

/// Emit the exact compact prior factor fed to the f32 normal reducer.  This
/// opt-in sidecar is used only to separate a Jᵀr arithmetic mismatch from a
/// frame-5 residual/current-point producer mismatch; it never participates in
/// the reducer and is disabled unless an explicit path is supplied.
#[inline]
pub(super) fn emit_prior_factor_input_diagnostic(
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    let Some(path) = crate::vio::window::diagnostic_env_snapshot()
        .diagnostic_prior_input
        .as_ref()
    else {
        return;
    };
    static COUNT: AtomicUsize = AtomicUsize::new(0);
    let compact_bits = compact_state_columns.map(|columns| {
        columns
            .iter()
            .flat_map(|&column| (0..jacobian.nrows()).map(move |row| jacobian[(row, column)]))
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
    });
    let record = json!({
        "schema": "basalt.m7im15_prior_factor_input.v1",
        "call": COUNT.fetch_add(1, Ordering::Relaxed),
        "rows": jacobian.nrows(),
        "global_cols": jacobian.ncols(),
        "columns": compact_state_columns,
        "jacobian_global_bits": jacobian
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
        "jacobian_compact_bits": compact_bits,
        "residual_bits": residual
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    });
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{}", record);
    }
}

/// Accumulate a prior `H.transpose() * r` using the same compact-column map
/// as [`accumulate_prior_gram_f32_eigen`].  The current f32 dot schedule is
/// shared with the generic transpose-vector path; compact extraction keeps
/// the source shape explicit while preserving the established RHS bits until
/// its dedicated Eigen GEMV audit is complete.
#[inline]
pub(super) fn accumulate_prior_transpose_vector_f32_eigen(
    accumulator: &mut DVector<f32>,
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    compact_state_columns: Option<&[usize]>,
) {
    assert_eq!(jacobian.nrows(), residual.len());
    assert_eq!(jacobian.ncols(), accumulator.len());
    let Some(columns) = compact_state_columns else {
        accumulate_transpose_vector_f32_eigen(accumulator, jacobian, residual, false);
        return;
    };
    if columns.len() != jacobian.nrows()
        || columns.iter().any(|&column| column >= accumulator.len())
    {
        accumulate_transpose_vector_f32_eigen(accumulator, jacobian, residual, false);
        return;
    }
    let compact = DMatrix::<f32>::from_fn(jacobian.nrows(), columns.len(), |row, column| {
        jacobian[(row, columns[column])]
    });
    // `compact.transpose() * residual` is dispatched by Eigen's row-major
    // GeneralMatrixVector kernel.  Its output rows are reduced in 8/4/2
    // groups, with Packet8/Packet4 horizontal reductions followed by the
    // scalar depth tail.  The generic column-wise helper has the same
    // mathematical result but a different f32 association for this 21x21
    // prior, so keep the compact transpose explicit here.
    let compact_transpose = compact.transpose().into_owned();
    let compact_result = eigen_prior_row_major_gemv_f32(&compact_transpose, residual);
    for (local_column, &global_column) in columns.iter().enumerate() {
        accumulator[global_column] += compact_result[local_column];
    }
}

/// Evaluate Eigen's row-major GEMV used by a compact marginal prior.
///
/// The 21x21 prior has an audited Eigen tail schedule, so retain that exact
/// path.  Later windows carry larger compact priors (for example 27x27 at
/// MH01 frame 9); those shapes use the generic dynamic Eigen GEMV instead of
/// asserting the first window's dimensions.  Keeping the dispatch here makes
/// the 21x21 parity path immutable while allowing the prior state dimension to
/// grow normally during a long run.
#[inline]
pub(super) fn eigen_prior_row_major_gemv_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    if matrix.nrows() == 21 && matrix.ncols() == 21 && vector.len() == 21 {
        return eigen_prior_row_major_gemv_21(matrix, vector);
    }
    if matrix.shape() == (27, 27) && vector.len() == 27 {
        return eigen_prior_row_major_gemv_27(matrix, vector);
    }
    if matrix.shape() == (33, 33) && vector.len() == 33 {
        return eigen_prior_row_major_gemv_33(matrix, vector);
    }
    if matrix.shape() == (39, 39) && vector.len() == 39 {
        return eigen_prior_row_major_gemv_39_f32(matrix, vector);
    }
    if matrix.shape() == (45, 45) && vector.len() == 45 {
        return eigen_prior_row_major_gemv_45_f32(matrix, vector);
    }
    if matrix.shape() == (51, 51) && vector.len() == 51 {
        return eigen_prior_row_major_gemv_51_f32(matrix, vector);
    }
    if matrix.shape() == (57, 57) && vector.len() == 57 {
        return eigen_prior_row_major_gemv_57_f32(matrix, vector);
    }
    eigen_row_major_gemv_f32(matrix, vector)
}

// Pinned 390b40 GEMV: paired outputs use sequential four-product adds
// (391eb6..391f09), then three FMA tail updates (391f1c..391f83).
pub(super) fn eigen_prior_row_major_gemv_39_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.shape(), (39, 39));
    assert_eq!(vector.len(), 39);
    DVector::from_fn(39, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for k in 0..32 {
            lanes[k % 8] = matrix[(row, k)].mul_add(vector[k], lanes[k % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        if row == 38 {
            let half: [f32; 4] = std::array::from_fn(|k| matrix[(row, 32 + k)] * vector[32 + k]);
            value += (half[0] + half[2]) + (half[1] + half[3]);
        } else {
            for k in 32..36 {
                value += matrix[(row, k)] * vector[k];
            }
        }
        for k in 36..39 {
            if row >= 36 {
                value = matrix[(row, k)].mul_add(vector[k], value);
            } else {
                value += matrix[(row, k)] * vector[k];
            }
        }
        value
    })
}

/// Pinned Eigen 45x45 prior GEMV. Output rows 0..39 use
/// the Packet8 depth body followed by five scalar multiply/add updates; the
/// four-row output tail contracts those five updates, while the final scalar
/// output row consumes depths 40..43 as one Packet4 reduction and depth 44
/// as a scalar FMA.
pub(super) fn eigen_prior_row_major_gemv_45_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.shape(), (45, 45));
    assert_eq!(vector.len(), 45);
    DVector::from_fn(45, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..40 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        if row == 44 {
            let half: [f32; 4] =
                std::array::from_fn(|k| mul_f32_exact(matrix[(row, 40 + k)], vector[40 + k]));
            value = add_f32_exact(
                value,
                add_f32_exact(
                    add_f32_exact(half[0], half[2]),
                    add_f32_exact(half[1], half[3]),
                ),
            );
            matrix[(row, 44)].mul_add(vector[44], value)
        } else if row >= 40 {
            for depth in 40..45 {
                value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
            }
            value
        } else {
            for depth in 40..45 {
                value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
            }
            value
        }
    })
}

/// Candidate for the pinned Eigen 51x51 compact-prior GEMV.  The first 48
/// depths form the Packet8 body.  Paired/scalar output tail rows 48..50
/// contract the final three depths, while complete output packets keep those
/// multiply/add operations separate.
pub(super) fn eigen_prior_row_major_gemv_51_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.shape(), (51, 51));
    assert_eq!(vector.len(), 51);
    DVector::from_fn(51, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..48 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let mut value = eigen_predux8_f32(lanes);
        for depth in 48..51 {
            value = if row >= 48 {
                matrix[(row, depth)].mul_add(vector[depth], value)
            } else {
                add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]))
            };
        }
        value
    })
}

/// Diagnostic candidate for Eigen's 57x57 compact-prior GEMV. Seven complete
/// Packet8 depth groups feed the standard horizontal reduction; the single
/// scalar depth/output remainder is fused only for Eigen's scalar output row.
pub(super) fn eigen_prior_row_major_gemv_57_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.shape(), (57, 57));
    assert_eq!(vector.len(), 57);
    DVector::from_fn(57, |row, _| {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..56 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let value = eigen_predux8_f32(lanes);
        if row == 56 {
            matrix[(row, 56)].mul_add(vector[56], value)
        } else {
            add_f32_exact(value, mul_f32_exact(matrix[(row, 56)], vector[56]))
        }
    })
}

pub(super) fn eigen_prior_row_major_gemv_33(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.shape(), (33, 33));
    assert_eq!(vector.len(), 33);
    let mut result = eigen_row_major_gemv_f32(matrix, vector);
    // Native scalar-output kernel: Packet8 redux 392070..39209e,
    // no Packet4 for depth33, then vfma231ss at392204.
    let mut lanes = [0.0_f32; 8];
    for depth in 0..32 {
        lanes[depth % 8] = matrix[(32, depth)].mul_add(vector[depth], lanes[depth % 8]);
    }
    result[32] = matrix[(32, 32)].mul_add(vector[32], eigen_predux8_f32(lanes));
    result
}

fn eigen_prior_row_major_gemv_27(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.shape(), (27, 27));
    assert_eq!(vector.len(), 27);
    let mut result = eigen_row_major_gemv_f32(matrix, vector);
    // Native 390b40: output pair 24/25 (391f1c..391f83), scalar
    // output 26 (392204..39223b) contract all three depth-tail updates.
    for row in 24..27 {
        let mut lanes = [0.0_f32; 8];
        for depth in 0..24 {
            lanes[depth % 8] = matrix[(row, depth)].mul_add(vector[depth], lanes[depth % 8]);
        }
        let q: [f32; 4] = std::array::from_fn(|k| lanes[k] + lanes[k + 4]);
        let mut value = (q[0] + q[2]) + (q[1] + q[3]);
        for depth in 24..27 {
            value = matrix[(row, depth)].mul_add(vector[depth], value);
        }
        result[row] = value;
    }
    result
}

/// Evaluate the audited Eigen row-major `21x21 * VectorXf` GEMV used by the
/// first compact marginal prior.  `GeneralMatrixVector.h` processes output
/// rows in 8/4/2 groups, but only the final scalar row receives the Packet4
/// depth tail: rows 0..19 use Packet8 depth 0..15 followed by scalar depths
/// 16..20, while row 20 uses Packet8 depth 0..15, Packet4 depth 16..19, then
/// scalar depth 20.  Keep the packet and scalar associations explicit; using
/// the landmark helper here would incorrectly apply Packet4 to every row.
#[inline]
fn eigen_prior_row_major_gemv_21(matrix: &DMatrix<f32>, vector: &DVector<f32>) -> DVector<f32> {
    assert_eq!(matrix.nrows(), 21);
    assert_eq!(matrix.ncols(), 21);
    assert_eq!(vector.len(), 21);
    let mut result = DVector::<f32>::zeros(21);

    for row in 0..20 {
        let mut lanes = [0.0_f32; 8];
        for depth in (0..16).step_by(8) {
            for lane in 0..8 {
                lanes[lane] =
                    matrix[(row, depth + lane)].mul_add(vector[depth + lane], lanes[lane]);
            }
        }
        let q0 = add_f32_exact(lanes[0], lanes[4]);
        let q1 = add_f32_exact(lanes[1], lanes[5]);
        let q2 = add_f32_exact(lanes[2], lanes[6]);
        let q3 = add_f32_exact(lanes[3], lanes[7]);
        let mut value = add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3));
        for depth in 16..21 {
            value = add_f32_exact(value, mul_f32_exact(matrix[(row, depth)], vector[depth]));
        }
        result[row] = value;
    }

    let row = 20;
    let mut lanes = [0.0_f32; 8];
    for depth in (0..16).step_by(8) {
        for lane in 0..8 {
            lanes[lane] = matrix[(row, depth + lane)].mul_add(vector[depth + lane], lanes[lane]);
        }
    }
    let q0 = add_f32_exact(lanes[0], lanes[4]);
    let q1 = add_f32_exact(lanes[1], lanes[5]);
    let q2 = add_f32_exact(lanes[2], lanes[6]);
    let q3 = add_f32_exact(lanes[3], lanes[7]);
    let mut value = add_f32_exact(add_f32_exact(q0, q2), add_f32_exact(q1, q3));

    let mut half = [0.0_f32; 4];
    for lane in 0..4 {
        half[lane] = matrix[(row, 16 + lane)].mul_add(vector[16 + lane], half[lane]);
    }
    let h0 = add_f32_exact(half[0], half[2]);
    let h1 = add_f32_exact(half[1], half[3]);
    value = add_f32_exact(value, add_f32_exact(h0, h1));
    // GCC contracts Eigen's scalar remainder at the final depth even though
    // the earlier scalar tail updates are ordinary multiply/add operations.
    value = matrix[(row, 20)].mul_add(vector[20], value);
    result[row] = value;
    result
}

/// Accumulate `J.transpose() * r` in the source ABS_QR order.
///
/// Eigen's `LandmarkBlockAbsDynamic::add_dense_H_b` uses a row-major Q2 block
/// and its packet GEMV evaluator.  For this transpose-times-vector shape the
/// pinned f32 result is the left-to-right row reduction with a fused multiply
/// add per row.  Keep the helper safe and local to the f32 reduction path;
/// f64 accumulation intentionally retains the existing nalgebra expression.
#[inline]
pub(super) fn accumulate_transpose_vector_f32_eigen(
    accumulator: &mut DVector<f32>,
    jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    exact_imu_rows: bool,
) {
    assert_eq!(jacobian.nrows(), residual.len());
    assert_eq!(jacobian.ncols(), accumulator.len());
    if exact_imu_rows && jacobian.nrows() == 9 {
        for column in 0..jacobian.ncols() {
            let lane0 = jacobian[(0, column)].mul_add(residual[0], 0.0_f32);
            let lane1 = jacobian[(1, column)].mul_add(residual[1], 0.0_f32);
            let lane2 = jacobian[(2, column)].mul_add(residual[2], 0.0_f32);
            let lane3 = jacobian[(3, column)].mul_add(residual[3], 0.0_f32);
            let lane4 = jacobian[(4, column)].mul_add(residual[4], 0.0_f32);
            let lane5 = jacobian[(5, column)].mul_add(residual[5], 0.0_f32);
            let lane6 = jacobian[(6, column)].mul_add(residual[6], 0.0_f32);
            let lane7 = jacobian[(7, column)].mul_add(residual[7], 0.0_f32);
            let q0 = lane0 + lane4;
            let q1 = lane1 + lane5;
            let q2 = lane2 + lane6;
            let q3 = lane3 + lane7;
            let packet_sum = (q0 + q2) + (q1 + q3);
            accumulator[column] += packet_sum + jacobian[(8, column)] * residual[8];
        }
        return;
    }
    for column in 0..jacobian.ncols() {
        let mut value = 0.0_f32;
        for row in 0..jacobian.nrows() {
            value = jacobian[(row, column)].mul_add(residual[row], value);
        }
        accumulator[column] += value;
    }
}

/// Evaluate Eigen's row-major dynamic MatrixXf-by-VectorXf GEMV for the
/// landmark Q1 block.  The pinned AVX2 kernel keeps one Packet8 accumulator
/// per output row across all complete eight-column packets, performs one
/// horizontal reduction, and only then visits the scalar tail.  Reducing
/// every packet immediately (the tempting scalar translation) changes the
/// association and produces different landmark increments.
#[inline]
pub(super) fn eigen_row_major_gemv_f32(
    matrix: &DMatrix<f32>,
    vector: &DVector<f32>,
) -> DVector<f32> {
    assert_eq!(matrix.ncols(), vector.len());
    let rows = matrix.nrows();
    let columns = matrix.ncols();
    let full_end = columns / 8 * 8;
    let half_end = columns / 4 * 4;
    let mut result = DVector::<f32>::zeros(rows);
    for row in 0..rows {
        let mut lanes = [0.0_f32; 8];
        let mut column = 0;
        while column < full_end {
            for lane in 0..8 {
                lanes[lane] =
                    matrix[(row, column + lane)].mul_add(vector[column + lane], lanes[lane]);
            }
            column += 8;
        }
        let q0 = lanes[0] + lanes[4];
        let q1 = lanes[1] + lanes[5];
        let q2 = lanes[2] + lanes[6];
        let q3 = lanes[3] + lanes[7];
        // Eigen's AVX2 Packet8f predux lowers the two half-packet sums as
        // `(q0 + q2) + (q1 + q3)` (the SSE movehl/movehdup tree).  Preserve
        // that association exactly; the alternative pairing changes f32
        // landmark increments.
        let mut value = (q0 + q2) + (q1 + q3);

        // On the pinned AVX2 Eigen build PacketSizeHalf/Quarter both have a
        // four-scalar lane width for f32.  Keep that packet tail explicit so
        // the helper remains faithful for state widths other than 75.
        let mut half_lanes = [0.0_f32; 4];
        while column < half_end {
            for lane in 0..4 {
                half_lanes[lane] =
                    matrix[(row, column + lane)].mul_add(vector[column + lane], half_lanes[lane]);
            }
            column += 4;
        }
        // Packet4f uses the same movehl/movehdup tree: (h0+h2)+(h1+h3).
        let h0 = half_lanes[0] + half_lanes[2];
        let h1 = half_lanes[1] + half_lanes[3];
        value += h0 + h1;
        while column < columns {
            value += matrix[(row, column)] * vector[column];
            column += 1;
        }
        result[row] = value;
    }
    result
}

/// Reproduce the scalar dot emitted by the pinned native visual
/// `backSubstitute` for `QJinc.dot(0.5 * QJinc + Qr)`. The first product is a
/// plain multiply; every remaining element is folded by one FMA. The caller
/// performs the final `l_diff -= dot` separately.
#[inline]
pub(super) fn eigen_visual_model_dot_f32(increment: &DVector<f32>, residual: &DVector<f32>) -> f32 {
    assert_eq!(increment.len(), residual.len());
    if increment.is_empty() {
        return 0.0_f32;
    }
    let first_right = 0.5_f32.mul_add(increment[0], residual[0]);
    let mut value = increment[0] * first_right;
    for index in 1..increment.len() {
        let right = 0.5_f32.mul_add(increment[index], residual[index]);
        value = increment[index].mul_add(right, value);
    }
    value
}
