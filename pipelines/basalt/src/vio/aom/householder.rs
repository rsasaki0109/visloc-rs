//! Landmark packet arithmetic and the f32 Householder landmark workspace.

use super::*;

// Eigen's pinned x86 f32 path uses eight-lane AVX packets (and four-lane
// half-packets).  Keeping this emulation local to ABS_QR makes the storage and
// arithmetic boundary explicit without target-specific intrinsics or unsafe
// code.
#[derive(Clone, Copy)]
struct LandmarkPacket4([f32; 4]);

#[inline]
fn landmark_packet_load(data: &[f32], start: usize) -> LandmarkPacket4 {
    LandmarkPacket4([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
    ])
}

#[inline]
fn landmark_packet_store(data: &mut [f32], start: usize, value: LandmarkPacket4) {
    data[start] = value.0[0];
    data[start + 1] = value.0[1];
    data[start + 2] = value.0[2];
    data[start + 3] = value.0[3];
}

#[inline]
fn landmark_packet_add(lhs: LandmarkPacket4, rhs: LandmarkPacket4) -> LandmarkPacket4 {
    LandmarkPacket4([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
    ])
}

#[inline]
fn landmark_packet_fused_sub(
    destination: LandmarkPacket4,
    workspace: LandmarkPacket4,
    scale: f32,
) -> LandmarkPacket4 {
    LandmarkPacket4([
        (-scale).mul_add(workspace.0[0], destination.0[0]),
        (-scale).mul_add(workspace.0[1], destination.0[1]),
        (-scale).mul_add(workspace.0[2], destination.0[2]),
        (-scale).mul_add(workspace.0[3], destination.0[3]),
    ])
}

#[inline]
fn landmark_packet_mul_add(
    lhs: LandmarkPacket4,
    rhs: LandmarkPacket4,
    accumulator: LandmarkPacket4,
) -> LandmarkPacket4 {
    // Eigen's SSE/AVX pmadd uses fused multiply-add when FMA is enabled.
    LandmarkPacket4([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
    ])
}

#[derive(Clone, Copy)]
struct LandmarkPacket8([f32; 8]);

#[inline]
fn landmark_packet8_load(data: &[f32], start: usize) -> LandmarkPacket8 {
    LandmarkPacket8([
        data[start],
        data[start + 1],
        data[start + 2],
        data[start + 3],
        data[start + 4],
        data[start + 5],
        data[start + 6],
        data[start + 7],
    ])
}

#[inline]
fn landmark_packet8_store(data: &mut [f32], start: usize, value: LandmarkPacket8) {
    data[start] = value.0[0];
    data[start + 1] = value.0[1];
    data[start + 2] = value.0[2];
    data[start + 3] = value.0[3];
    data[start + 4] = value.0[4];
    data[start + 5] = value.0[5];
    data[start + 6] = value.0[6];
    data[start + 7] = value.0[7];
}

#[inline]
fn landmark_packet8_add(lhs: LandmarkPacket8, rhs: LandmarkPacket8) -> LandmarkPacket8 {
    LandmarkPacket8([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
        lhs.0[4] + rhs.0[4],
        lhs.0[5] + rhs.0[5],
        lhs.0[6] + rhs.0[6],
        lhs.0[7] + rhs.0[7],
    ])
}

#[inline]
fn landmark_packet8_fused_sub(
    destination: LandmarkPacket8,
    workspace: LandmarkPacket8,
    scale: f32,
) -> LandmarkPacket8 {
    LandmarkPacket8([
        (-scale).mul_add(workspace.0[0], destination.0[0]),
        (-scale).mul_add(workspace.0[1], destination.0[1]),
        (-scale).mul_add(workspace.0[2], destination.0[2]),
        (-scale).mul_add(workspace.0[3], destination.0[3]),
        (-scale).mul_add(workspace.0[4], destination.0[4]),
        (-scale).mul_add(workspace.0[5], destination.0[5]),
        (-scale).mul_add(workspace.0[6], destination.0[6]),
        (-scale).mul_add(workspace.0[7], destination.0[7]),
    ])
}

#[inline]
fn landmark_packet8_mul_add(
    lhs: LandmarkPacket8,
    rhs: LandmarkPacket8,
    accumulator: LandmarkPacket8,
) -> LandmarkPacket8 {
    LandmarkPacket8([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
        lhs.0[4].mul_add(rhs.0[4], accumulator.0[4]),
        lhs.0[5].mul_add(rhs.0[5], accumulator.0[5]),
        lhs.0[6].mul_add(rhs.0[6], accumulator.0[6]),
        lhs.0[7].mul_add(rhs.0[7], accumulator.0[7]),
    ])
}

/// Packet workspace used by Eigen's column-major GEMV path for
/// `essential.adjoint() * bottom`.
///
/// Eigen transposes this vector-on-the-left product.  The resulting
/// `bottom.transpose()` is treated as a column-major matrix, so AVX packets
/// span eight output columns (with a four-lane half-packet tail) while the
/// essential dimension is accumulated scalar-by-scalar.  This is subtly
/// different from packetizing the dot product itself and is the boundary that
/// makes the f4 trace bitwise exact.
#[inline]
fn landmark_gemv_workspace(
    essential: &[f32],
    storage: &[f32],
    row_start: usize,
    storage_cols: usize,
) -> Vec<f32> {
    let mut result = Vec::new();
    landmark_gemv_workspace_into(essential, storage, row_start, storage_cols, &mut result);
    result
}

/// Capacity-reusing form of [`landmark_gemv_workspace`].  The evaluator
/// schedule is intentionally identical: only the destination Vec ownership
/// changes, so the clean reducer can reuse this scratch between landmark
/// factors without changing any f32 operation or traversal order.
#[inline]
fn landmark_gemv_workspace_into(
    essential: &[f32],
    storage: &[f32],
    row_start: usize,
    storage_cols: usize,
    result: &mut Vec<f32>,
) {
    debug_assert!(row_start + essential.len() <= storage.len() / storage_cols);
    result.clear();
    result.resize(storage_cols, 0.0_f32);
    let packet_end = storage_cols / 8 * 8;
    let mut column = 0;
    while column < packet_end {
        let mut packet = LandmarkPacket8([0.0; 8]);
        for (offset, &value) in essential.iter().enumerate() {
            let rhs = LandmarkPacket8([value; 8]);
            let lhs = landmark_packet8_load(storage, (row_start + offset) * storage_cols + column);
            packet = landmark_packet8_mul_add(lhs, rhs, packet);
        }
        landmark_packet8_store(result, column, packet);
        column += 8;
    }
    let half_packet_end = storage_cols / 4 * 4;
    while column < half_packet_end {
        let mut packet = LandmarkPacket4([0.0; 4]);
        for (offset, &value) in essential.iter().enumerate() {
            let rhs = LandmarkPacket4([value; 4]);
            let lhs = landmark_packet_load(storage, (row_start + offset) * storage_cols + column);
            packet = landmark_packet_mul_add(lhs, rhs, packet);
        }
        landmark_packet_store(result, column, packet);
        column += 4;
    }
    while column < storage_cols {
        let mut value = 0.0_f32;
        for (offset, &essential_value) in essential.iter().enumerate() {
            value += essential_value * storage[(row_start + offset) * storage_cols + column];
        }
        result[column] = value;
        column += 1;
    }
}

/// Scratch buffers moved into and out of one [`LandmarkHouseholderF32`] at a
/// time.  A QR result still owns its full transformed storage because Q2 rows
/// are returned to the caller, but the storage/pivot/tau vectors and the
/// Householder essential/GEMV workspaces are recycled after each projection.
/// This is deliberately capacity-only reuse; no matrix arithmetic or source
/// order is changed.
#[derive(Default)]
pub(super) struct LandmarkHouseholderWorkspace {
    pub(super) storage: Vec<f32>,
    pub(super) pivots: Vec<f32>,
    pub(super) tau: Vec<f32>,
    pub(super) essential: Vec<f32>,
    pub(super) gemv: Vec<f32>,
}

impl LandmarkHouseholderWorkspace {
    #[inline]
    fn take_zeroed(buffer: &mut Vec<f32>, len: usize) -> Vec<f32> {
        let mut value = std::mem::take(buffer);
        value.clear();
        value.resize(len, 0.0_f32);
        // `resize` does not overwrite elements when the old capacity/length
        // already covers `len`; the upstream storage starts zeroed on every
        // factor, including padding and damping rows.
        value.fill(0.0_f32);
        value
    }

    #[inline]
    fn recycle(buffer: &mut Vec<f32>, mut value: Vec<f32>) {
        value.clear();
        *buffer = value;
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct LandmarkHouseholderLayout {
    rows: usize,
    landmark_offset: usize,
    residual_offset: usize,
    storage_cols: usize,
    storage_len: usize,
}

/// Compute every derived ABS_QR dimension before moving any reusable buffer.
/// The production factors are small, but keeping this arithmetic checked makes
/// malformed or adversarial matrix metadata fail closed instead of wrapping
/// into a short allocation followed by an indexing panic.
#[inline]
pub(super) fn checked_landmark_householder_layout(
    observation_rows: usize,
    state_cols: usize,
    landmark_cols: usize,
) -> Option<LandmarkHouseholderLayout> {
    if landmark_cols == 0 || landmark_cols > 3 {
        return None;
    }
    let rows = observation_rows.checked_add(landmark_cols)?;
    let padding_cols = 4usize.checked_sub(state_cols % 4)?;
    let landmark_offset = state_cols.checked_add(padding_cols)?;
    let residual_offset = landmark_offset.checked_add(landmark_cols)?;
    let storage_cols = residual_offset.checked_add(1)?;
    let storage_len = rows.checked_mul(storage_cols)?;
    Some(LandmarkHouseholderLayout {
        rows,
        landmark_offset,
        residual_offset,
        storage_cols,
        storage_len,
    })
}

#[inline]
pub(super) fn checked_compact_storage_len(
    landmark_cols: usize,
    state_cols: usize,
) -> Option<usize> {
    let q1_state_len = landmark_cols.checked_mul(state_cols)?;
    let residual_end = q1_state_len.checked_add(landmark_cols)?;
    residual_end.checked_add(landmark_cols.checked_mul(landmark_cols)?)
}

#[inline]
pub(super) fn checked_compact_storage_capacity(
    factors: &[WhitenedFactorRowStack],
    state_cols: usize,
) -> Option<usize> {
    factors.iter().try_fold(0usize, |capacity, factor| {
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return Some(capacity);
        }
        let entry_len = checked_compact_storage_len(landmark_cols, state_cols)?;
        capacity.checked_add(entry_len)
    })
}

/// Safe f32 implementation of the three-landmark ABS_QR Householder walk.
///
/// Basalt's `LandmarkBlockAbsDynamic` owns a row-major matrix laid out as
/// `[Jp | pad | Jl | r]`.  It allocates one zero damping row per landmark
/// column, then calls `performQRHouseholder` with
/// `remainingRows = num_rows - k - 3` for `k = 0..2`; consequently the final
/// three rows never enter a reflector and remain zero.  This type intentionally
/// keeps the full row-major storage so fixture tests can check Q1, Q2,
/// pivots, and signed-zero behavior at the same boundary.
#[derive(Clone)]
pub(super) struct LandmarkHouseholderF32 {
    pub(super) rows: usize,
    pub(super) state_cols: usize,
    pub(super) landmark_cols: usize,
    pub(super) landmark_offset: usize,
    pub(super) residual_offset: usize,
    pub(super) storage: Vec<f32>,
    pub(super) pivots: Vec<f32>,
    pub(super) tau: Vec<f32>,
}

impl LandmarkHouseholderF32 {
    pub(super) fn factor(
        state: &DMatrix<f32>,
        landmark: &DMatrix<f32>,
        residual: &DVector<f32>,
    ) -> Option<Self> {
        let observation_rows = state.nrows();
        if landmark.nrows() != observation_rows || residual.len() != observation_rows {
            return None;
        }
        let landmark_cols = landmark.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        // Compute all derived dimensions before allocating any storage.  The
        // checked helper preserves the upstream non-modulo-4 padding rule
        // while making malformed/adversarial dimensions fail closed.
        let layout =
            checked_landmark_householder_layout(observation_rows, state.ncols(), landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols: state.ncols(),
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: vec![0.0; layout.storage_len],
            pivots: vec![0.0; landmark_cols],
            tau: vec![0.0; landmark_cols],
        };

        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state.ncols() {
                result.storage[destination + column] = state[(row, column)];
            }
            for column in 0..landmark_cols {
                result.storage[destination + layout.landmark_offset + column] =
                    landmark[(row, column)];
            }
            result.storage[destination + layout.residual_offset] = residual[row];
        }
        result.perform();
        Some(result)
    }

    /// Build the same row-major ABS_QR storage directly from an active
    /// whitened factor.  The ordinary `factor` constructor above remains the
    /// materialized f32 fixture/oracle path; the clean production reducer uses
    /// this variant so the three temporary f32 state/landmark/residual
    /// matrices are not allocated and copied into storage a second time.
    ///
    /// The conversion, zero padding, offsets, and `perform` call deliberately
    /// mirror `factor` above.  The small norm accumulator is the same
    /// eight-way column dot schedule used by nalgebra's dynamic `Matrix::norm`
    /// and is returned only for the compact-entry eligibility bit.
    pub(super) fn factor_from_whitened(factor: &WhitenedFactorRowStack) -> Option<(Self, f32)> {
        let observation_rows = factor.state_jacobian.nrows();
        if factor.landmark_jacobian.nrows() != observation_rows
            || factor.residual.len() != observation_rows
        {
            return None;
        }
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        let state_cols = factor.state_jacobian.ncols();
        // Keep the upstream non-modulo-4 padding rule exactly aligned with
        // `factor`: an already aligned state width still gets four zeros.
        let layout =
            checked_landmark_householder_layout(observation_rows, state_cols, landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols,
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: vec![0.0; layout.storage_len],
            pivots: vec![0.0; landmark_cols],
            tau: vec![0.0; landmark_cols],
        };

        // `DMatrix::norm()` reduces each landmark column with eight scalar
        // accumulators, then folds (0+4),(1+5),(2+6),(3+7).  Keep those
        // accumulators on the stack while the source values are converted
        // and packed, avoiding a second f32 matrix and a second conversion.
        let mut norm_accumulators = [[0.0_f32; 8]; 3];
        let mut norm_tail = [0.0_f32; 3];
        let packet_end = observation_rows / 8 * 8;
        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state_cols {
                result.storage[destination + column] = factor.state_jacobian[(row, column)] as f32;
            }
            for column in 0..landmark_cols {
                let value = factor.landmark_jacobian[(row, column)] as f32;
                result.storage[destination + layout.landmark_offset + column] = value;
                let product = value * value;
                if row < packet_end {
                    norm_accumulators[column][row % 8] += product;
                } else {
                    // nalgebra's dynamic dot product reduces the complete
                    // packet prefix first, then adds the scalar remainder.
                    // Keep that boundary explicit: folding the remainder
                    // into an accumulator lane changes the final f32 bits
                    // for rows that are not a multiple of eight.
                    norm_tail[column] += product;
                }
            }
            result.storage[destination + layout.residual_offset] = factor.residual[row] as f32;
        }
        let mut norm_squared = 0.0_f32;
        for (column, accumulators) in norm_accumulators.iter().enumerate().take(landmark_cols) {
            // nalgebra's `norm_squared` first finishes one column's local
            // dot-product accumulator and only then adds that scalar to the
            // matrix-wide result.  Keeping this local variable is important
            // for f32 rounding when there is more than one landmark column.
            let mut column_squared = 0.0_f32;
            column_squared += accumulators[0] + accumulators[4];
            column_squared += accumulators[1] + accumulators[5];
            column_squared += accumulators[2] + accumulators[6];
            column_squared += accumulators[3] + accumulators[7];
            column_squared += norm_tail[column];
            norm_squared += column_squared;
        }

        result.perform();
        Some((result, norm_squared.sqrt()))
    }

    /// Workspace-backed counterpart to [`Self::factor_from_whitened`].  The
    /// packed source conversion and norm schedule are intentionally copied
    /// exactly; only the QR-owned vectors are taken from reusable capacity and
    /// returned by the projection wrapper after Q2/compact extraction.
    fn factor_from_whitened_with_workspace(
        factor: &WhitenedFactorRowStack,
        workspace: &mut LandmarkHouseholderWorkspace,
    ) -> Option<(Self, f32)> {
        let observation_rows = factor.state_jacobian.nrows();
        if factor.landmark_jacobian.nrows() != observation_rows
            || factor.residual.len() != observation_rows
        {
            return None;
        }
        let landmark_cols = factor.landmark_jacobian.ncols();
        if landmark_cols == 0 || landmark_cols > 3 {
            return None;
        }

        let state_cols = factor.state_jacobian.ncols();
        // Resolve the complete layout before taking any reusable workspace
        // vector.  Overflow therefore cannot leave the workspace consumed or
        // turn a malformed factor into a short-buffer indexing panic.
        let layout =
            checked_landmark_householder_layout(observation_rows, state_cols, landmark_cols)?;
        let mut result = Self {
            rows: layout.rows,
            state_cols,
            landmark_cols,
            landmark_offset: layout.landmark_offset,
            residual_offset: layout.residual_offset,
            storage: LandmarkHouseholderWorkspace::take_zeroed(
                &mut workspace.storage,
                layout.storage_len,
            ),
            pivots: LandmarkHouseholderWorkspace::take_zeroed(&mut workspace.pivots, landmark_cols),
            tau: LandmarkHouseholderWorkspace::take_zeroed(&mut workspace.tau, landmark_cols),
        };

        let mut norm_accumulators = [[0.0_f32; 8]; 3];
        let mut norm_tail = [0.0_f32; 3];
        let packet_end = observation_rows / 8 * 8;
        for row in 0..observation_rows {
            let destination = row * layout.storage_cols;
            for column in 0..state_cols {
                result.storage[destination + column] = factor.state_jacobian[(row, column)] as f32;
            }
            for column in 0..landmark_cols {
                let value = factor.landmark_jacobian[(row, column)] as f32;
                result.storage[destination + layout.landmark_offset + column] = value;
                let product = value * value;
                if row < packet_end {
                    norm_accumulators[column][row % 8] += product;
                } else {
                    norm_tail[column] += product;
                }
            }
            result.storage[destination + layout.residual_offset] = factor.residual[row] as f32;
        }
        let mut norm_squared = 0.0_f32;
        for (column, accumulators) in norm_accumulators.iter().enumerate().take(landmark_cols) {
            let mut column_squared = 0.0_f32;
            column_squared += accumulators[0] + accumulators[4];
            column_squared += accumulators[1] + accumulators[5];
            column_squared += accumulators[2] + accumulators[6];
            column_squared += accumulators[3] + accumulators[7];
            column_squared += norm_tail[column];
            norm_squared += column_squared;
        }

        result.perform_with_workspace(workspace);
        Some((result, norm_squared.sqrt()))
    }

    #[inline]
    pub(super) const fn index(&self, row: usize, column: usize) -> usize {
        row * self.storage_cols() + column
    }

    #[inline]
    pub(super) const fn storage_cols(&self) -> usize {
        self.residual_offset + 1
    }

    fn perform(&mut self) {
        let mut essential = Vec::new();
        let mut workspace = Vec::new();
        self.perform_with_scratch(&mut essential, &mut workspace);
    }

    #[inline]
    fn perform_with_workspace(&mut self, workspace: &mut LandmarkHouseholderWorkspace) {
        self.perform_with_scratch(&mut workspace.essential, &mut workspace.gemv);
    }

    fn perform_with_scratch(&mut self, essential: &mut Vec<f32>, workspace: &mut Vec<f32>) {
        let storage_cols = self.storage_cols();
        let damping_rows = self.landmark_cols;
        for k in 0..self.landmark_cols {
            // The production ABS_QR path has three landmark columns and three
            // trailing damping rows.  Keep the same formula for the small
            // synthetic one/two-column rank tests as well.
            let remaining_rows = self.rows.saturating_sub(k + damping_rows);
            if remaining_rows == 0 {
                self.pivots[k] = 0.0;
                self.tau[k] = 0.0;
                continue;
            }
            let landmark_column = self.landmark_offset + k;
            let pivot_index = self.index(k, landmark_column);
            let c0 = self.storage[pivot_index];
            let tail_len = remaining_rows - 1;
            let mut tail_sq_norm = 0.0_f32;
            for offset in 0..tail_len {
                let value = self.storage[self.index(k + 1 + offset, landmark_column)];
                // Eigen's row-major column-stride VectorBlock reduction uses
                // a fused scalar multiply-add for this tail path.  Keep this
                // boundary explicit; the all61 fixture has a one-ULP beta
                // witness at k=1 (track 63).
                tail_sq_norm = value.mul_add(value, tail_sq_norm);
            }

            let (tau, beta) = if tail_sq_norm <= f32::MIN_POSITIVE {
                // Real f32 has zero imaginary component.  Do not normalize or
                // rewrite c0 here: this preserves both +0 and -0 exactly.
                essential.clear();
                essential.resize(tail_len, 0.0_f32);
                (0.0_f32, c0)
            } else {
                // Eigen's makeHouseholder adds c0^2 to the tail norm with a
                // scalar fused multiply-add before sqrt (vfmadd231ss in the
                // pinned clean build).  Keep this separate from the tail
                // reduction so both rounding boundaries are explicit.
                let beta_norm = c0.mul_add(c0, tail_sq_norm).sqrt();
                let beta = if c0 >= 0.0 { -beta_norm } else { beta_norm };
                let denominator = c0 - beta;
                essential.clear();
                essential.extend((0..tail_len).map(|offset| {
                    self.storage[self.index(k + 1 + offset, landmark_column)] / denominator
                }));
                let tau = (beta - c0) / beta;
                (tau, beta)
            };
            self.pivots[k] = beta;
            self.tau[k] = tau;

            if tau == 0.0 {
                continue;
            }

            // applyHouseholderOnTheLeft first materializes all workspace
            // entries from the old bottom and row-0 values.  Eigen
            // transposes this vector-on-the-left product and uses its
            // column-major GEMV path: each packet spans eight output columns,
            // while the essential dimension is accumulated in scalar order.
            landmark_gemv_workspace_into(essential, &self.storage, k + 1, storage_cols, workspace);
            // The row-0 update is a contiguous row-major vector operation.
            let packet_end = storage_cols / 8 * 8;
            let mut column = 0;
            while column < packet_end {
                let row_index = self.index(k, column);
                let updated_workspace = landmark_packet8_add(
                    landmark_packet8_load(workspace, column),
                    landmark_packet8_load(&self.storage, row_index),
                );
                landmark_packet8_store(workspace, column, updated_workspace);
                let updated_row = landmark_packet8_fused_sub(
                    landmark_packet8_load(&self.storage, row_index),
                    updated_workspace,
                    tau,
                );
                landmark_packet8_store(&mut self.storage, row_index, updated_row);
                column += 8;
            }
            let half_packet_end = storage_cols / 4 * 4;
            while column < half_packet_end {
                let row_index = self.index(k, column);
                let updated_workspace = landmark_packet_add(
                    landmark_packet_load(workspace, column),
                    landmark_packet_load(&self.storage, row_index),
                );
                landmark_packet_store(workspace, column, updated_workspace);
                let updated_row = landmark_packet_fused_sub(
                    landmark_packet_load(&self.storage, row_index),
                    updated_workspace,
                    tau,
                );
                landmark_packet_store(&mut self.storage, row_index, updated_row);
                column += 4;
            }
            while column < storage_cols {
                workspace[column] += self.storage[self.index(k, column)];
                let row_index = self.index(k, column);
                self.storage[row_index] =
                    (-tau).mul_add(workspace[column], self.storage[row_index]);
                column += 1;
            }
            // For a row-major destination Eigen's outer-product evaluator
            // visits rows in order and updates each contiguous row with packet
            // mul/sub operations.  Materialize tau*essential per row before
            // touching the destination, preserving the no-alias order.
            for offset in 0..tail_len {
                let row = k + 1 + offset;
                let scale = tau * essential[offset];
                let row_index = self.index(row, 0);
                let mut column = 0;
                while column < packet_end {
                    let destination = row_index + column;
                    let updated = landmark_packet8_fused_sub(
                        landmark_packet8_load(&self.storage, destination),
                        landmark_packet8_load(&workspace, column),
                        scale,
                    );
                    landmark_packet8_store(&mut self.storage, destination, updated);
                    column += 8;
                }
                while column < half_packet_end {
                    let destination = row_index + column;
                    let updated = landmark_packet_fused_sub(
                        landmark_packet_load(&self.storage, destination),
                        landmark_packet_load(&workspace, column),
                        scale,
                    );
                    landmark_packet_store(&mut self.storage, destination, updated);
                    column += 4;
                }
                while column < storage_cols {
                    let index = row_index + column;
                    self.storage[index] = (-scale).mul_add(workspace[column], self.storage[index]);
                    column += 1;
                }
            }
        }
    }

    fn recycle_into(self, workspace: &mut LandmarkHouseholderWorkspace) {
        LandmarkHouseholderWorkspace::recycle(&mut workspace.storage, self.storage);
        LandmarkHouseholderWorkspace::recycle(&mut workspace.pivots, self.pivots);
        LandmarkHouseholderWorkspace::recycle(&mut workspace.tau, self.tau);
    }

    pub(super) fn transformed_state(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.rows, self.state_cols, |row, column| {
            self.storage[self.index(row, column)]
        })
    }

    pub(super) fn transformed_residual(&self) -> DVector<f32> {
        DVector::from_iterator(
            self.rows,
            (0..self.rows).map(|row| self.storage[self.index(row, self.residual_offset)]),
        )
    }

    pub(super) fn transformed_landmark(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.rows, self.landmark_cols, |row, column| {
            self.storage[self.index(row, self.landmark_offset + column)]
        })
    }

    pub(super) fn q2_state(&self) -> DMatrix<f32> {
        let q2_rows = self.rows.saturating_sub(self.landmark_cols);
        DMatrix::from_fn(q2_rows, self.state_cols, |row, column| {
            self.storage[self.index(self.landmark_cols + row, column)]
        })
    }

    pub(super) fn q2_residual(&self) -> DVector<f32> {
        let q2_rows = self.rows.saturating_sub(self.landmark_cols);
        DVector::from_iterator(
            q2_rows,
            (0..q2_rows).map(|row| {
                self.storage[self.index(self.landmark_cols + row, self.residual_offset)]
            }),
        )
    }

    pub(super) fn upper_r(&self) -> DMatrix<f32> {
        DMatrix::from_fn(self.landmark_cols, self.landmark_cols, |row, column| {
            if column < row {
                0.0
            } else {
                self.storage[self.index(row, self.landmark_offset + column)]
            }
        })
    }

    /// Copy only the Q1 rows needed by landmark back-substitution.
    ///
    /// This is deliberately an extraction-only operation: the Householder
    /// walk has already happened in [`Self::factor`], and no arithmetic or
    /// row traversal is repeated here.  The lower triangle is materialized
    /// with the same positive zero convention as [`Self::upper_r`].
    pub(super) fn compact_back_substitution(
        &self,
        landmark_index: usize,
        track_id: u64,
        rank: usize,
        eligible: bool,
    ) -> Option<CompactLandmarkBackSubstitutionF32> {
        let mut storage = Vec::new();
        let descriptor = self.compact_back_substitution_into(
            &mut storage,
            landmark_index,
            track_id,
            rank,
            eligible,
        )?;
        debug_assert_eq!(descriptor.storage_offset, 0);
        Some(CompactLandmarkBackSubstitutionF32 {
            landmark_index,
            track_id,
            storage,
            state_cols: descriptor.state_cols,
            landmark_cols: descriptor.landmark_cols,
            rank,
            eligible,
        })
    }

    /// Extract compact Q1/R values into a caller-owned arena.  The arithmetic
    /// and source traversal are identical to [`Self::compact_back_substitution`];
    /// only ownership changes so a reducer can retain one allocation for all
    /// visual factors in the current LM iteration.
    fn compact_back_substitution_into(
        &self,
        arena: &mut Vec<f32>,
        landmark_index: usize,
        track_id: u64,
        rank: usize,
        eligible: bool,
    ) -> Option<CompactLandmarkBackSubstitutionEntryF32> {
        let rows = self.landmark_cols;
        let q1_state_len = rows.checked_mul(self.state_cols)?;
        let residual_offset = q1_state_len;
        let upper_r_offset = residual_offset.checked_add(rows)?;
        let storage_offset = arena.len();
        let storage_len = checked_compact_storage_len(rows, self.state_cols)?;
        let storage_end = storage_offset.checked_add(storage_len)?;
        arena.resize(storage_end, 0.0_f32);
        // Keep this extraction in the same row/column order as the old
        // DMatrix values.  Only the container changes; no f32 arithmetic is
        // introduced or reordered here.
        for row in 0..rows {
            for column in 0..self.state_cols {
                arena[storage_offset + row * self.state_cols + column] =
                    self.storage[self.index(row, column)];
            }
            arena[storage_offset + residual_offset + row] =
                self.storage[self.index(row, self.residual_offset)];
            for column in row..rows {
                arena[storage_offset + upper_r_offset + row * rows + column] =
                    self.storage[self.index(row, self.landmark_offset + column)];
            }
        }
        Some(CompactLandmarkBackSubstitutionEntryF32 {
            landmark_index,
            track_id,
            storage_offset,
            state_cols: self.state_cols,
            landmark_cols: rows,
            rank,
            eligible,
        })
    }
}

/// Project one landmark block and, optionally, extract its compact Q1/R
/// payload from the very same Householder storage.  The compact caller uses
/// this function so enabling the reducer option cannot introduce the second
/// QR that the old trial recovery performed.
pub(super) fn landmark_nullspace_projection_f32_with_compact(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionF32>,
) {
    let state = as_f32_matrix(&factor.state_jacobian);
    let landmark = as_f32_matrix(&factor.landmark_jacobian);
    let residual = as_f32_vector(&factor.residual);
    let landmark_columns = landmark.ncols();
    if landmark_columns == 0 || landmark_columns > 3 {
        return (
            if landmark_columns == 0 {
                state
            } else {
                DMatrix::zeros(0, factor.state_jacobian.ncols())
            },
            if landmark_columns == 0 {
                residual
            } else {
                DVector::zeros(0)
            },
            0,
            None,
        );
    }
    let norm_ok = landmark.norm() > tolerance as f32;
    let Some(qr) = LandmarkHouseholderF32::factor(&state, &landmark, &residual) else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    };
    let threshold = tolerance as f32;
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    let compact = metadata.and_then(|metadata| {
        qr.compact_back_substitution(
            metadata.landmark_index,
            metadata.track_id,
            rank,
            norm_ok && rank == landmark_columns,
        )
    });
    if qr.rows <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
            compact,
        );
    }
    (qr.q2_state(), qr.q2_residual(), rank, compact)
}

/// Arena-backed counterpart to
/// [`landmark_nullspace_projection_f32_with_compact`].  It deliberately
/// duplicates only the small projection wrapper so the Householder
/// implementation and all projected Q2 arithmetic remain shared; the
/// compact extraction writes directly into the reducer's single arena.
///
/// The QR scratch argument is reused across visual factors.  It owns no Q2
/// output: the transformed storage is recycled only after this function has
/// materialized the returned Q2 matrices and compact arena entry.
pub(super) fn landmark_nullspace_projection_f32_with_compact_into_with_workspace(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
    arena: &mut Vec<f32>,
    qr_workspace: &mut LandmarkHouseholderWorkspace,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionEntryF32>,
) {
    let landmark_columns = factor.landmark_jacobian.ncols();
    if landmark_columns == 0 {
        // Landmark-free factors are not part of the compact visual path; keep
        // the historical materialized f32 projection for their ordinary
        // state/residual contribution.
        return (
            as_f32_matrix(&factor.state_jacobian),
            as_f32_vector(&factor.residual),
            0,
            None,
        );
    }
    if landmark_columns > 3 {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    }
    // Valid visual compact factors are packed directly from the f64 source;
    // this is the only production compact projection constructor.  The
    // materialized `factor` constructor remains available above for fixtures
    // and independent parity tests.
    let Some((qr, landmark_norm)) =
        LandmarkHouseholderF32::factor_from_whitened_with_workspace(factor, qr_workspace)
    else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
            None,
        );
    };
    let norm_ok = landmark_norm > tolerance as f32;
    let threshold = tolerance as f32;
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    let compact = metadata.and_then(|metadata| {
        qr.compact_back_substitution_into(
            arena,
            metadata.landmark_index,
            metadata.track_id,
            rank,
            norm_ok && rank == landmark_columns,
        )
    });
    let result = if qr.rows <= landmark_columns {
        (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
            compact,
        )
    } else {
        (qr.q2_state(), qr.q2_residual(), rank, compact)
    };
    qr.recycle_into(qr_workspace);
    result
}

/// Compatibility wrapper for fixture/legacy callers that do not own a
/// reducer-scoped scratch workspace.  Production clean reduction calls the
/// `_with_workspace` variant above so all visual factors share capacity.
pub(super) fn landmark_nullspace_projection_f32_with_compact_into(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    metadata: Option<LandmarkFactorMetadata>,
    arena: &mut Vec<f32>,
) -> (
    DMatrix<f32>,
    DVector<f32>,
    usize,
    Option<CompactLandmarkBackSubstitutionEntryF32>,
) {
    let mut qr_workspace = LandmarkHouseholderWorkspace::default();
    landmark_nullspace_projection_f32_with_compact_into_with_workspace(
        factor,
        tolerance,
        metadata,
        arena,
        &mut qr_workspace,
    )
}

/// Factor one visual block once and retain the compact native-f32 payload
/// required after the state step is known.  This helper is intentionally
/// separate from the reducer until the clean trial-view wiring is proven.
pub(crate) fn compact_landmark_back_substitution_f32(
    factor: &WhitenedFactorRowStack,
    landmark_index: usize,
    track_id: u64,
    tolerance: f64,
) -> Option<CompactLandmarkBackSubstitutionF32> {
    landmark_nullspace_projection_f32_with_compact(
        factor,
        tolerance,
        Some(LandmarkFactorMetadata {
            landmark_index,
            track_id,
        }),
    )
    .3
}

pub(crate) fn landmark_nullspace_projection_f32(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
) -> (DMatrix<f32>, DVector<f32>, usize) {
    let state = as_f32_matrix(&factor.state_jacobian);
    let landmark = as_f32_matrix(&factor.landmark_jacobian);
    let residual = as_f32_vector(&factor.residual);
    let landmark_columns = landmark.ncols();
    if landmark_columns == 0 {
        return (state, residual, 0);
    }
    let threshold = tolerance as f32;
    let Some(qr) = LandmarkHouseholderF32::factor(&state, &landmark, &residual) else {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            0,
        );
    };
    let rank = (0..landmark_columns)
        .filter(|&index| qr.pivots[index].abs() > threshold)
        .count();
    if let Some(metadata) = factor.landmark_metadata {
        emit_landmark_projection_probe(
            metadata,
            factor,
            &state,
            &landmark,
            &residual,
            &qr,
            rank,
            // The retained legacy wrapper does not compute the compact
            // path's separate landmark norm eligibility.  Keep this probe
            // side-effect-free on the production path and report the exact
            // rank-based condition available at this boundary.
            rank == landmark_columns,
        );
    }
    if qr.rows <= landmark_columns {
        return (
            DMatrix::zeros(0, factor.state_jacobian.ncols()),
            DVector::zeros(0),
            rank,
        );
    }
    (qr.q2_state(), qr.q2_residual(), rank)
}
