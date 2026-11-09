//! Safe f32 emulation of Eigen packet kernels (loads, products, norms, GEMV)
//! and the Q-trace writer.

use super::*;

#[derive(Clone, Copy)]
struct UpstreamF32Packet4([f32; 4]);

#[derive(Clone, Copy)]
pub(super) struct UpstreamF32Packet8(pub(super) [f32; 8]);

#[inline]
pub(super) fn upstream_f32_bits_hash(values: &[f32]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1_099_511_628_211_u64);
        }
    }
    hash
}

#[inline]
fn upstream_f32_write_qtrace_u32(output: &mut fs::File, value: u32) -> bool {
    output.write_all(&value.to_le_bytes()).is_ok()
}

#[inline]
fn upstream_f32_write_qtrace_f32_bits(output: &mut fs::File, value: f32) -> bool {
    upstream_f32_write_qtrace_u32(output, value.to_bits())
}

pub(super) fn upstream_f32_write_qtrace_header(
    output: &mut fs::File,
    rows: usize,
    cols: usize,
    rhs_size: usize,
    marginal_size: usize,
    keep_size: usize,
) -> bool {
    output.write_all(b"M7IM15QTRACE1").is_ok()
        && upstream_f32_write_qtrace_u32(output, 1)
        && upstream_f32_write_qtrace_u32(output, rows as u32)
        && upstream_f32_write_qtrace_u32(output, cols as u32)
        && upstream_f32_write_qtrace_u32(output, rhs_size as u32)
        && upstream_f32_write_qtrace_u32(output, marginal_size as u32)
        && upstream_f32_write_qtrace_u32(output, keep_size as u32)
}

pub(super) fn upstream_f32_write_qtrace_record(
    output: &mut fs::File,
    column: usize,
    rank_before: usize,
    rank_after: usize,
    c0: f32,
    tail_sq_norm: f32,
    beta: f32,
    tau: f32,
    qj: &[f32],
    qr: &[f32],
) -> bool {
    if !upstream_f32_write_qtrace_u32(output, column as u32)
        || !upstream_f32_write_qtrace_u32(output, rank_before as u32)
        || !upstream_f32_write_qtrace_u32(output, rank_after as u32)
        || !upstream_f32_write_qtrace_f32_bits(output, c0)
        || !upstream_f32_write_qtrace_f32_bits(output, tail_sq_norm)
        || !upstream_f32_write_qtrace_f32_bits(output, beta)
        || !upstream_f32_write_qtrace_f32_bits(output, tau)
    {
        return false;
    }
    for &value in qj.iter().chain(qr.iter()) {
        if !upstream_f32_write_qtrace_u32(output, value.to_bits()) {
            return false;
        }
    }
    true
}

#[inline]
fn upstream_f32_packet_load(data: &[f32], index: usize) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        data[index],
        data[index + 1],
        data[index + 2],
        data[index + 3],
    ])
}

#[inline]
pub(super) fn upstream_f32_packet8_load(data: &[f32], index: usize) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        data[index],
        data[index + 1],
        data[index + 2],
        data[index + 3],
        data[index + 4],
        data[index + 5],
        data[index + 6],
        data[index + 7],
    ])
}

#[inline]
fn upstream_f32_packet_store(data: &mut [f32], index: usize, packet: UpstreamF32Packet4) {
    data[index] = packet.0[0];
    data[index + 1] = packet.0[1];
    data[index + 2] = packet.0[2];
    data[index + 3] = packet.0[3];
}

#[inline]
pub(super) fn upstream_f32_packet8_store(
    data: &mut [f32],
    index: usize,
    packet: UpstreamF32Packet8,
) {
    data[index] = packet.0[0];
    data[index + 1] = packet.0[1];
    data[index + 2] = packet.0[2];
    data[index + 3] = packet.0[3];
    data[index + 4] = packet.0[4];
    data[index + 5] = packet.0[5];
    data[index + 6] = packet.0[6];
    data[index + 7] = packet.0[7];
}

#[inline]
fn upstream_f32_packet_add(lhs: UpstreamF32Packet4, rhs: UpstreamF32Packet4) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
    ])
}

#[inline]
fn upstream_f32_packet_mul(lhs: UpstreamF32Packet4, rhs: UpstreamF32Packet4) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0] * rhs.0[0],
        lhs.0[1] * rhs.0[1],
        lhs.0[2] * rhs.0[2],
        lhs.0[3] * rhs.0[3],
    ])
}

#[inline]
fn upstream_f32_packet8_mul(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        lhs.0[0] * rhs.0[0],
        lhs.0[1] * rhs.0[1],
        lhs.0[2] * rhs.0[2],
        lhs.0[3] * rhs.0[3],
        lhs.0[4] * rhs.0[4],
        lhs.0[5] * rhs.0[5],
        lhs.0[6] * rhs.0[6],
        lhs.0[7] * rhs.0[7],
    ])
}

#[inline]
fn upstream_f32_packet_mul_scalar(packet: UpstreamF32Packet4, scalar: f32) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        packet.0[0] * scalar,
        packet.0[1] * scalar,
        packet.0[2] * scalar,
        packet.0[3] * scalar,
    ])
}

#[inline]
fn upstream_f32_packet8_mul_add(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
    accumulator: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        lhs.0[0].fused_mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].fused_mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].fused_mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].fused_mul_add(rhs.0[3], accumulator.0[3]),
        lhs.0[4].fused_mul_add(rhs.0[4], accumulator.0[4]),
        lhs.0[5].fused_mul_add(rhs.0[5], accumulator.0[5]),
        lhs.0[6].fused_mul_add(rhs.0[6], accumulator.0[6]),
        lhs.0[7].fused_mul_add(rhs.0[7], accumulator.0[7]),
    ])
}

#[inline]
pub(super) fn upstream_f32_packet8_sub_scaled(
    data: &mut [f32],
    start: usize,
    source: &[f32],
    scalar: f32,
) {
    let len = source.len();
    let negative_scalar = -scalar;
    // Eigen's AVX assignment peels each destination vector until the
    // column-tail pointer reaches its 32-byte packet boundary.  The source
    // essential vector starts at the same logical tail offset, so consume the
    // identical scalar prefix before loading Packet8 values.
    let head = ((8 - (start & 7)) & 7).min(len);
    for offset in 0..head {
        // Eigen's scalar peel is emitted as a multiply followed by a
        // subtract; this prefix is retained as the existing scalar peel.
        data[start + offset] -= scalar * source[offset];
    }
    let packet_end = head + ((len - head) / 8) * 8;
    let mut offset = head;
    while offset < packet_end {
        let destination = upstream_f32_packet8_load(data, start + offset);
        let source_packet = upstream_f32_packet8_load(source, offset);
        let updated = UpstreamF32Packet8([
            negative_scalar.fused_mul_add(source_packet.0[0], destination.0[0]),
            negative_scalar.fused_mul_add(source_packet.0[1], destination.0[1]),
            negative_scalar.fused_mul_add(source_packet.0[2], destination.0[2]),
            negative_scalar.fused_mul_add(source_packet.0[3], destination.0[3]),
            negative_scalar.fused_mul_add(source_packet.0[4], destination.0[4]),
            negative_scalar.fused_mul_add(source_packet.0[5], destination.0[5]),
            negative_scalar.fused_mul_add(source_packet.0[6], destination.0[6]),
            negative_scalar.fused_mul_add(source_packet.0[7], destination.0[7]),
        ]);
        upstream_f32_packet8_store(data, start + offset, updated);
        offset += 8;
    }
    // The pinned Eigen AVX masked remainder materializes `tau * essential`
    // before the outer product, then emits a separate multiply by the
    // workspace scalar and a subtract from the destination.  Keep the
    // packet body above fused, but prevent LLVM from contracting this
    // scalar/masked remainder back into an FMA.
    while offset < len {
        let product = upstream_f32_scalar_product(scalar, source[offset]);
        data[start + offset] -= product;
        offset += 1;
    }
}

#[inline]
pub(super) fn upstream_f32_packet8_sub_scaled_fma_head(
    data: &mut [f32],
    start: usize,
    source: &[f32],
    scalar: f32,
) {
    // The RHS vector assignment uses the same fused scalar multiply/subtract
    // as Eigen's masked packet path, including the alignment-peel head.  The
    // matrix outer update intentionally keeps `upstream_f32_packet8_sub_scaled`
    // because its scalar head is a separate multiply/subtract on this build.
    let len = source.len();
    let negative_scalar = -scalar;
    let head = ((8 - (start & 7)) & 7).min(len);
    for offset in 0..head {
        data[start + offset] = negative_scalar.fused_mul_add(source[offset], data[start + offset]);
    }
    let packet_end = head + ((len - head) / 8) * 8;
    let mut offset = head;
    while offset < packet_end {
        let destination = upstream_f32_packet8_load(data, start + offset);
        let source_packet = upstream_f32_packet8_load(source, offset);
        let updated = UpstreamF32Packet8([
            negative_scalar.fused_mul_add(source_packet.0[0], destination.0[0]),
            negative_scalar.fused_mul_add(source_packet.0[1], destination.0[1]),
            negative_scalar.fused_mul_add(source_packet.0[2], destination.0[2]),
            negative_scalar.fused_mul_add(source_packet.0[3], destination.0[3]),
            negative_scalar.fused_mul_add(source_packet.0[4], destination.0[4]),
            negative_scalar.fused_mul_add(source_packet.0[5], destination.0[5]),
            negative_scalar.fused_mul_add(source_packet.0[6], destination.0[6]),
            negative_scalar.fused_mul_add(source_packet.0[7], destination.0[7]),
        ]);
        upstream_f32_packet8_store(data, start + offset, updated);
        offset += 8;
    }
    while offset < len {
        data[start + offset] = negative_scalar.fused_mul_add(source[offset], data[start + offset]);
        offset += 1;
    }
}

#[inline]
fn upstream_f32_packet8_add(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
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
fn upstream_f32_packet_mul_add(
    lhs: UpstreamF32Packet4,
    rhs: UpstreamF32Packet4,
    accumulator: UpstreamF32Packet4,
) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0].fused_mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].fused_mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].fused_mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].fused_mul_add(rhs.0[3], accumulator.0[3]),
    ])
}

#[inline]
fn upstream_f32_packet_reduce(packet: UpstreamF32Packet4) -> f32 {
    let even = packet.0[0] + packet.0[2];
    let odd = packet.0[1] + packet.0[3];
    even + odd
}

#[inline]
fn upstream_f32_packet8_reduce(packet: UpstreamF32Packet8) -> f32 {
    // Eigen's AVX predux first adds the low and high 128-bit halves, then
    // applies the Packet4 reduction order to that result.
    let pair0 = packet.0[0] + packet.0[4];
    let pair1 = packet.0[1] + packet.0[5];
    let pair2 = packet.0[2] + packet.0[6];
    let pair3 = packet.0[3] + packet.0[7];
    let even = pair0 + pair2;
    let odd = pair1 + pair3;
    even + odd
}

fn upstream_f32_packet_squared_norm(data: &[f32], start: usize, len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }
    // Eigen's `squaredNorm()` uses the linear vectorized reduction for a
    // dynamic column segment.  Its first packet is the first 16-byte aligned
    // address, while scalar coefficients before that packet are folded after
    // the packet reduction.  Starting a packet at `start` changes both the
    // packet grouping and the scalar/packet addition order whenever the
    // Householder tail is not aligned.
    let end = start + len;
    let aligned_start = start + ((4 - (start & 3)) & 3).min(len);
    let aligned_size = ((end - aligned_start) / 4) * 4;
    let aligned_end = aligned_start + aligned_size;
    let mut result;

    if aligned_size != 0 {
        let mut packet_result = upstream_f32_packet_mul(
            upstream_f32_packet_load(data, aligned_start),
            upstream_f32_packet_load(data, aligned_start),
        );
        if aligned_size > 4 {
            let mut packet_result_1 = upstream_f32_packet_mul(
                upstream_f32_packet_load(data, aligned_start + 4),
                upstream_f32_packet_load(data, aligned_start + 4),
            );
            let aligned_end2 = aligned_start + (aligned_size / 8) * 8;
            let mut index = aligned_start + 8;
            while index < aligned_end2 {
                packet_result = upstream_f32_packet_add(
                    packet_result,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, index),
                        upstream_f32_packet_load(data, index),
                    ),
                );
                packet_result_1 = upstream_f32_packet_add(
                    packet_result_1,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, index + 4),
                        upstream_f32_packet_load(data, index + 4),
                    ),
                );
                index += 8;
            }
            packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
            if aligned_end > aligned_end2 {
                packet_result = upstream_f32_packet_add(
                    packet_result,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, aligned_end2),
                        upstream_f32_packet_load(data, aligned_end2),
                    ),
                );
            }
        }
        result = upstream_f32_packet_reduce(packet_result);
        for index in start..aligned_start {
            result += data[index] * data[index];
        }
        for index in aligned_end..end {
            result += data[index] * data[index];
        }
    } else {
        result = data[start] * data[start];
        for index in (start + 1)..end {
            result += data[index] * data[index];
        }
    }
    result
}

/// Eigen's AVX `squaredNorm()` reduction for a dynamic vector segment.
///
/// This is the same `LinearVectorizedTraversal` schedule as the pinned
/// `redux_impl` for `unaryExpr(squared_norm_functor).sum()`: keep two Packet8
/// accumulators in flight, combine packet products with the contracted
/// multiply-add sequence emitted by the pinned O3 Eigen build, merge them,
/// then reduce with AVX's low/high-half order before adding scalar
/// head/tail coefficients.
#[inline]
pub(super) fn upstream_f32_packet8_squared_norm(data: &[f32], start: usize, len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }
    let end = start + len;
    // The pinned Eigen cwiseAbs2 evaluator has no DirectAccessBit, so
    // `first_default_aligned` is the segment origin rather than the parent
    // allocation's physical address.  Keep the packet grouping relative to
    // this VectorBlock segment.
    let aligned_start = start;
    let aligned_size = ((end - aligned_start) / 8) * 8;
    let aligned_end = aligned_start + aligned_size;
    let mut result;
    if aligned_size != 0 {
        let mut packet_result = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(data, aligned_start),
            upstream_f32_packet8_load(data, aligned_start),
        );
        if aligned_size > 8 {
            let mut packet_result_1 = upstream_f32_packet8_mul(
                upstream_f32_packet8_load(data, aligned_start + 8),
                upstream_f32_packet8_load(data, aligned_start + 8),
            );
            let aligned_end2 = aligned_start + (aligned_size / 16) * 16;
            let mut index = aligned_start + 16;
            while index < aligned_end2 {
                let packet = upstream_f32_packet8_load(data, index);
                packet_result = upstream_f32_packet8_mul_add(packet, packet, packet_result);
                let packet_1 = upstream_f32_packet8_load(data, index + 8);
                packet_result_1 = upstream_f32_packet8_mul_add(packet_1, packet_1, packet_result_1);
                index += 16;
            }
            packet_result = upstream_f32_packet8_add(packet_result, packet_result_1);
            if aligned_end > aligned_end2 {
                let packet = upstream_f32_packet8_load(data, aligned_end2);
                packet_result = upstream_f32_packet8_mul_add(packet, packet, packet_result);
            }
        }
        result = upstream_f32_packet8_reduce(packet_result);
        for index in start..aligned_start {
            result = data[index].fused_mul_add(data[index], result);
        }
        // GCC's pinned Eigen build vectorizes a four-coefficient remainder as
        // `vmulps` followed by scalar `vaddss` instructions.  Any remainder
        // after that packet stays on the scalar contracted FMA path.
        let scalar_packet_end = (aligned_end + ((end - aligned_end) / 4) * 4).min(end);
        for index in aligned_end..scalar_packet_end {
            result += data[index] * data[index];
        }
        for index in scalar_packet_end..end {
            result = data[index].fused_mul_add(data[index], result);
        }
    } else {
        result = data[start] * data[start];
        let scalar_packet_end = (start + 1 + ((len - 1) / 4) * 4).min(end);
        for index in (start + 1)..scalar_packet_end {
            result += data[index] * data[index];
        }
        for index in scalar_packet_end..end {
            result = data[index].fused_mul_add(data[index], result);
        }
    }
    result
}

#[inline]
fn upstream_f32_norm_scalar_variant(data: &[f32], start: usize, len: usize) -> f32 {
    let mut result = data[start] * data[start];
    for index in (start + 1)..(start + len) {
        result += data[index] * data[index];
    }
    result
}

#[inline]
fn upstream_f32_norm_packet4_variant(
    data: &[f32],
    start: usize,
    len: usize,
    head: usize,
    two_accumulators: bool,
) -> f32 {
    let head = head.min(len);
    let aligned_start = start + head;
    let aligned_size = ((len - head) / 4) * 4;
    let aligned_end = aligned_start + aligned_size;
    if aligned_size == 0 {
        return upstream_f32_norm_scalar_variant(data, start, len);
    }
    let mut packet_result = upstream_f32_packet_mul(
        upstream_f32_packet_load(data, aligned_start),
        upstream_f32_packet_load(data, aligned_start),
    );
    if two_accumulators && aligned_size > 4 {
        let mut packet_result_1 = upstream_f32_packet_mul(
            upstream_f32_packet_load(data, aligned_start + 4),
            upstream_f32_packet_load(data, aligned_start + 4),
        );
        let aligned_end2 = aligned_start + (aligned_size / 8) * 8;
        let mut index = aligned_start + 8;
        while index < aligned_end2 {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index),
                    upstream_f32_packet_load(data, index),
                ),
            );
            packet_result_1 = upstream_f32_packet_add(
                packet_result_1,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index + 4),
                    upstream_f32_packet_load(data, index + 4),
                ),
            );
            index += 8;
        }
        packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
        if aligned_end > aligned_end2 {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, aligned_end2),
                    upstream_f32_packet_load(data, aligned_end2),
                ),
            );
        }
    } else {
        let mut index = aligned_start + 4;
        while index < aligned_end {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index),
                    upstream_f32_packet_load(data, index),
                ),
            );
            index += 4;
        }
    }
    let mut result = upstream_f32_packet_reduce(packet_result);
    for index in start..aligned_start {
        result = data[index].fused_mul_add(data[index], result);
    }
    for index in aligned_end..(start + len) {
        result = data[index].fused_mul_add(data[index], result);
    }
    result
}

#[inline]
fn upstream_f32_norm_packet8_variant(
    data: &[f32],
    start: usize,
    len: usize,
    head: usize,
    two_accumulators: bool,
) -> f32 {
    let head = head.min(len);
    let aligned_start = start + head;
    let aligned_size = ((len - head) / 8) * 8;
    let aligned_end = aligned_start + aligned_size;
    if aligned_size == 0 {
        return upstream_f32_norm_scalar_variant(data, start, len);
    }
    let mut packet_result = upstream_f32_packet8_mul(
        upstream_f32_packet8_load(data, aligned_start),
        upstream_f32_packet8_load(data, aligned_start),
    );
    if two_accumulators && aligned_size > 8 {
        let mut packet_result_1 = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(data, aligned_start + 8),
            upstream_f32_packet8_load(data, aligned_start + 8),
        );
        let aligned_end2 = aligned_start + (aligned_size / 16) * 16;
        let mut index = aligned_start + 16;
        while index < aligned_end2 {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index),
                    upstream_f32_packet8_load(data, index),
                ),
            );
            packet_result_1 = upstream_f32_packet8_add(
                packet_result_1,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index + 8),
                    upstream_f32_packet8_load(data, index + 8),
                ),
            );
            index += 16;
        }
        packet_result = upstream_f32_packet8_add(packet_result, packet_result_1);
        if aligned_end > aligned_end2 {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, aligned_end2),
                    upstream_f32_packet8_load(data, aligned_end2),
                ),
            );
        }
    } else {
        let mut index = aligned_start + 8;
        while index < aligned_end {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index),
                    upstream_f32_packet8_load(data, index),
                ),
            );
            index += 8;
        }
    }
    let mut result = upstream_f32_packet8_reduce(packet_result);
    for index in start..aligned_start {
        result += data[index] * data[index];
    }
    for index in aligned_end..(start + len) {
        result += data[index] * data[index];
    }
    result
}

fn upstream_f32_packet_dot(
    lhs: &[f32],
    lhs_start: usize,
    rhs: &[f32],
    rhs_start: usize,
    len: usize,
) -> f32 {
    if len == 0 {
        return 0.0;
    }
    let end = lhs_start + len;
    let packet_end = lhs_start + (len / 4) * 4;
    if packet_end == lhs_start {
        let mut result = lhs[lhs_start] * rhs[rhs_start];
        for offset in 1..len {
            result += lhs[lhs_start + offset] * rhs[rhs_start + offset];
        }
        return result;
    }

    let packet_count = len / 4;
    let quad_end = lhs_start + (len / 16) * 16;
    let remainder_packets = (packet_end - quad_end) / 4;
    let mut packet_result = upstream_f32_packet_mul(
        upstream_f32_packet_load(lhs, lhs_start),
        upstream_f32_packet_load(rhs, rhs_start),
    );
    if packet_count >= 2 {
        let mut packet_result_1 = upstream_f32_packet_mul(
            upstream_f32_packet_load(lhs, lhs_start + 4),
            upstream_f32_packet_load(rhs, rhs_start + 4),
        );
        if packet_count >= 3 {
            let mut packet_result_2 = upstream_f32_packet_mul(
                upstream_f32_packet_load(lhs, lhs_start + 8),
                upstream_f32_packet_load(rhs, rhs_start + 8),
            );
            if packet_count >= 4 {
                let mut packet_result_3 = upstream_f32_packet_mul(
                    upstream_f32_packet_load(lhs, lhs_start + 12),
                    upstream_f32_packet_load(rhs, rhs_start + 12),
                );
                let mut index = lhs_start + 16;
                while index < quad_end {
                    packet_result = upstream_f32_packet_add(
                        packet_result,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index),
                            upstream_f32_packet_load(rhs, rhs_start + index - lhs_start),
                        ),
                    );
                    packet_result_1 = upstream_f32_packet_add(
                        packet_result_1,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 4),
                            upstream_f32_packet_load(rhs, rhs_start + index + 4 - lhs_start),
                        ),
                    );
                    packet_result_2 = upstream_f32_packet_add(
                        packet_result_2,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 8),
                            upstream_f32_packet_load(rhs, rhs_start + index + 8 - lhs_start),
                        ),
                    );
                    packet_result_3 = upstream_f32_packet_add(
                        packet_result_3,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 12),
                            upstream_f32_packet_load(rhs, rhs_start + index + 12 - lhs_start),
                        ),
                    );
                    index += 16;
                }
                if remainder_packets >= 1 {
                    packet_result = upstream_f32_packet_add(
                        packet_result,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end - lhs_start),
                        ),
                    );
                }
                if remainder_packets >= 2 {
                    packet_result_1 = upstream_f32_packet_add(
                        packet_result_1,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end + 4),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end + 4 - lhs_start),
                        ),
                    );
                }
                if remainder_packets == 3 {
                    packet_result_2 = upstream_f32_packet_add(
                        packet_result_2,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end + 8),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end + 8 - lhs_start),
                        ),
                    );
                }
                packet_result_2 = upstream_f32_packet_add(packet_result_2, packet_result_3);
            }
            packet_result_1 = upstream_f32_packet_add(packet_result_1, packet_result_2);
        }
        packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
    }

    let mut result = upstream_f32_packet_reduce(packet_result);
    for index in packet_end..end {
        let offset = index - lhs_start;
        result += lhs[index] * rhs[rhs_start + offset];
    }
    result
}

// Eigen's vector-on-the-left matrix product is evaluated through the
// row-major GEMV kernel.  Unlike the vector inner-product path above, GEMV
// keeps one packet accumulator and advances it in four-element blocks.
#[inline]
fn upstream_f32_packet_gemv_dot(lhs: &[f32], rhs: &[f32], rhs_start: usize, len: usize) -> f32 {
    if len < 4 {
        if len == 0 {
            return 0.0;
        }
        let mut result = lhs[0] * rhs[rhs_start];
        for offset in 1..len {
            result += lhs[offset] * rhs[rhs_start + offset];
        }
        return result;
    }
    let packet_end = (len / 4) * 4;
    let mut packet_result = UpstreamF32Packet4([0.0; 4]);
    let mut offset = 0;
    while offset < packet_end {
        packet_result = upstream_f32_packet_mul_add(
            upstream_f32_packet_load(lhs, offset),
            upstream_f32_packet_load(rhs, rhs_start + offset),
            packet_result,
        );
        offset += 4;
    }
    let mut result = upstream_f32_packet_reduce(packet_result);
    for offset in packet_end..len {
        result += lhs[offset] * rhs[rhs_start + offset];
    }
    result
}

/// Emulate Eigen's AVX row-major GEMV used by
/// `essential.adjoint() * bottom`.  Eigen treats the transposed expression as
/// a row-major matrix whose rows are trailing qj columns and whose columns
/// are the Householder essential vector.  It therefore accumulates eight
/// output rows at once, broadcasting each essential packet across the rows;
/// packetizing the essential dimension instead changes the reduction order.
#[inline(never)]
pub(super) fn upstream_f32_gemv_scalar_product(lhs: f32, rhs: f32) -> f32 {
    // The pinned Eigen/GCC row-major GEMV emits the scalar remainder as a
    // separate vmulss/vaddss sequence.  Keep the product out of the caller so
    // LLVM cannot contract the source-level `sum += lhs * rhs` back into FMA.
    lhs * rhs
}

#[inline(never)]
fn upstream_f32_scalar_product(lhs: f32, rhs: f32) -> f32 {
    // Eigen's masked packet remainder materializes the product before the
    // subtract (vmulps followed by vsubps), rather than using a scalar FMA.
    // Keeping this operation out of the caller prevents LLVM contraction.
    lhs * rhs
}

#[inline(never)]
fn upstream_f32_gemv_packet4_accumulate(
    mut result: f32,
    lhs: &[f32],
    rhs: &[f32],
    lhs_start: usize,
    rhs_start: usize,
) -> f32 {
    // GCC's AVX row-major GEMV lowers a four-element scalar remainder to one
    // vmulps followed by four ordered vaddss operations into the existing
    // scalar accumulator.  Keep this packet's products separate from the
    // accumulator so it cannot become four FMAs or a tree reduction.
    let product = [
        lhs[lhs_start] * rhs[rhs_start],
        lhs[lhs_start + 1] * rhs[rhs_start + 1],
        lhs[lhs_start + 2] * rhs[rhs_start + 2],
        lhs[lhs_start + 3] * rhs[rhs_start + 3],
    ];
    result += product[0];
    result += product[1];
    result += product[2];
    result += product[3];
    result
}

#[inline]
pub(super) fn upstream_f32_packet8_gemv_workspace(
    essential: &[f32],
    qj: &[f32],
    rows: usize,
    tail_row_start: usize,
    first_trailing_column: usize,
    trailing_columns: usize,
) -> Vec<f32> {
    let mut workspace = vec![0.0_f32; trailing_columns];
    let len = essential.len();
    let full_col_end = len / 8 * 8;
    let half_col_end = full_col_end + (len - full_col_end) / 4 * 4;

    // The row-major Eigen kernel processes 8, then 4, then 2 output rows.
    // Each packet accumulator is reduced only after the complete essential
    // dimension, matching `predux(Packet8f)` and its low/high-half order.
    let mut row = 0usize;
    while row + 8 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 8];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..8 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..8 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].fused_mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 8;
    }

    while row + 4 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 4];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..4 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..4 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].fused_mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 4;
    }

    while row + 2 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 2];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..2 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..2 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].fused_mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 2;
    }

    // The final row uses Eigen's full, half, and quarter packet accumulators
    // before the scalar tail.  Packet4's reduction is the same low/high then
    // even/odd order used by `upstream_f32_packet_reduce`.
    while row < trailing_columns {
        let column = first_trailing_column + row;
        let base = column * rows + tail_row_start;
        let mut full = UpstreamF32Packet8([0.0; 8]);
        let mut offset = 0usize;
        while offset < full_col_end {
            full = upstream_f32_packet8_mul_add(
                upstream_f32_packet8_load(qj, base + offset),
                upstream_f32_packet8_load(essential, offset),
                full,
            );
            offset += 8;
        }
        let mut result = upstream_f32_packet8_reduce(full);
        if half_col_end > full_col_end {
            let mut half = UpstreamF32Packet4([0.0; 4]);
            let mut offset = full_col_end;
            while offset < half_col_end {
                half = upstream_f32_packet_mul_add(
                    upstream_f32_packet_load(qj, base + offset),
                    upstream_f32_packet_load(essential, offset),
                    half,
                );
                offset += 4;
            }
            result += upstream_f32_packet_reduce(half);
        }
        for offset in half_col_end..len {
            result = qj[base + offset].fused_mul_add(essential[offset], result);
        }
        workspace[row] = result;
        row += 1;
    }
    workspace
}

/// Eigen's runtime-vector inner product for the one-column RHS update.  The
/// AVX evaluator seeds four Packet8 accumulators, folds complete packets with
/// FMA, merges them left-associatively, and reduces the packet before the
/// scalar FMA tail.  This is distinct from the multi-column GEMV workspace.
#[inline]
pub(super) fn upstream_f32_packet8_rhs_dot(
    lhs: &[f32],
    rhs: &[f32],
    rhs_start: usize,
    len: usize,
) -> f32 {
    // `essential.adjoint() * bottom` is Eigen's dynamic inner-product
    // evaluator, not the one-accumulator GEMV used for matrix workspaces.
    // Its AVX path seeds four Packet8 accumulators, folds complete groups of
    // four packets with packet FMA, merges p2<-p3, p1<-p2, p0<-p1, then
    // preduces.  The scalar remainder is the evaluator's scalar FMA tail.
    if len < 8 {
        if len == 0 {
            return 0.0;
        }
        let mut result = lhs[0] * rhs[rhs_start];
        for offset in 1..len {
            result = lhs[offset].fused_mul_add(rhs[rhs_start + offset], result);
        }
        return result;
    }
    let packet_end = len / 8 * 8;
    let quad_end = len / 32 * 32;
    let packet_count = len / 8;
    let remainder_packets = (packet_end - quad_end) / 8;
    let mut packet0 = upstream_f32_packet8_mul(
        upstream_f32_packet8_load(lhs, 0),
        upstream_f32_packet8_load(rhs, rhs_start),
    );
    if packet_count >= 2 {
        let mut packet1 = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(lhs, 8),
            upstream_f32_packet8_load(rhs, rhs_start + 8),
        );
        if packet_count >= 3 {
            let mut packet2 = upstream_f32_packet8_mul(
                upstream_f32_packet8_load(lhs, 16),
                upstream_f32_packet8_load(rhs, rhs_start + 16),
            );
            if packet_count >= 4 {
                let mut packet3 = upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(lhs, 24),
                    upstream_f32_packet8_load(rhs, rhs_start + 24),
                );
                let mut offset = 32usize;
                while offset < quad_end {
                    packet0 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset),
                        upstream_f32_packet8_load(rhs, rhs_start + offset),
                        packet0,
                    );
                    packet1 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 8),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 8),
                        packet1,
                    );
                    packet2 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 16),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 16),
                        packet2,
                    );
                    packet3 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 24),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 24),
                        packet3,
                    );
                    offset += 32;
                }
                if remainder_packets >= 1 {
                    packet0 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end),
                        packet0,
                    );
                }
                if remainder_packets >= 2 {
                    packet1 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end + 8),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end + 8),
                        packet1,
                    );
                }
                if remainder_packets == 3 {
                    packet2 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end + 16),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end + 16),
                        packet2,
                    );
                }
                packet2 = upstream_f32_packet8_add(packet2, packet3);
            }
            packet1 = upstream_f32_packet8_add(packet1, packet2);
        }
        packet0 = upstream_f32_packet8_add(packet0, packet1);
    }
    let mut result = upstream_f32_packet8_reduce(packet0);
    for offset in packet_end..len {
        result = lhs[offset].fused_mul_add(rhs[rhs_start + offset], result);
    }
    result
}

pub(super) fn upstream_f32_packet_scale(data: &[f32], scalar: f32) -> Vec<f32> {
    let mut scaled = vec![0.0_f32; data.len()];
    let packet_end = (data.len() / 4) * 4;
    let mut index = 0;
    while index < packet_end {
        let packet = upstream_f32_packet_mul_scalar(upstream_f32_packet_load(data, index), scalar);
        upstream_f32_packet_store(&mut scaled, index, packet);
        index += 4;
    }
    for index in packet_end..data.len() {
        scaled[index] = data[index] * scalar;
    }
    scaled
}

pub(super) fn upstream_f32_packet_divide_assign(
    data: &mut [f32],
    start: usize,
    len: usize,
    denominator: f32,
) {
    let head = ((4 - (start & 3)) & 3).min(len);
    for offset in 0..head {
        data[start + offset] /= denominator;
    }
    let packet_len = ((len - head) / 4) * 4;
    let mut offset = head;
    while offset < head + packet_len {
        let packet = upstream_f32_packet_load(data, start + offset);
        let divided = UpstreamF32Packet4([
            packet.0[0] / denominator,
            packet.0[1] / denominator,
            packet.0[2] / denominator,
            packet.0[3] / denominator,
        ]);
        upstream_f32_packet_store(data, start + offset, divided);
        offset += 4;
    }
    for offset in (head + packet_len)..len {
        data[start + offset] /= denominator;
    }
}

fn upstream_f32_packet_sub_scaled(data: &mut [f32], start: usize, source: &[f32], scalar: f32) {
    let len = source.len();
    let head = ((4 - (start & 3)) & 3).min(len);
    for offset in 0..head {
        data[start + offset] -= source[offset] * scalar;
    }
    let packet_len = ((len - head) / 4) * 4;
    let mut offset = head;
    while offset < head + packet_len {
        let destination = upstream_f32_packet_load(data, start + offset);
        let source_packet = upstream_f32_packet_load(source, offset);
        let scaled = upstream_f32_packet_mul_scalar(source_packet, scalar);
        let updated = UpstreamF32Packet4([
            destination.0[0] - scaled.0[0],
            destination.0[1] - scaled.0[1],
            destination.0[2] - scaled.0[2],
            destination.0[3] - scaled.0[3],
        ]);
        upstream_f32_packet_store(data, start + offset, updated);
        offset += 4;
    }
    for offset in (head + packet_len)..len {
        data[start + offset] -= source[offset] * scalar;
    }
}

#[inline]
pub(super) fn upstream_f32_householder_beta(c0: f32, tail_sq_norm: f32) -> f32 {
    // Eigen's float makeHouseholderInPlace accumulates the pivot square into
    // the tail norm with a fused multiply-add.  Keep this as one explicit
    // operation at the scalar boundary; the sign convention is applied by
    // the caller after taking the positive square root.
    c0.fused_mul_add(c0, tail_sq_norm).sqrt()
}
