// Batched brute-force top-2 nearest neighbours over a device-resident
// descriptor bank (the GPU half of visloc-vision's `BruteForceMatcher`).
//
// Each entry names a query block and a train block of the bank; for every
// query row it finds the two smallest scores ||t||^2 - 2 q.t (= ||q - t||^2
// - ||q||^2, the CPU matcher's ranking) and the best train index. Ties go
// to the lower index, like the CPU's strict `<` scan. The host turns
// scores into distances, applies the ratio test and cross-checks with the
// swapped entry.
//
// One workgroup = 128 query rows x all train rows of one entry, in 128-row
// train tiles; a classic shared-memory GEMM with 16-wide k chunks, each
// thread holding an 8 x 8 register tile (rows 4 ty + i and 64 + 4 ty + i,
// cols 4 tx + j and 64 + 4 tx + j) so every operand read out of shared
// memory is one vec4. Every dot product still sums k = 0..dim in order.
// (The body is generated fully unrolled; see the accumulator names.)
//
// Cross-check needs the reverse top-2 as well (per train column over all
// query rows, score ||q||^2 - 2 q.t). The same dot tile gives it: each
// workgroup writes a per-column partial top-2 over its 128 rows to
// `rev_part`, and `rev_reduce` folds the row blocks. The dot products are
// bit-identical to computing the swapped entry, so the result is too.

struct MatchParams {
    // vec4s per descriptor row (dim / 4); dim is a multiple of 32.
    dim4: u32,
    num_entries: u32,
    pad0: u32,
    pad1: u32,
};

struct Entry {
    q_off: u32,
    nq: u32,
    t_off: u32,
    nt: u32,
    out_off: u32,
    // Reverse top-2: partials at rev_part[rev_part_off + block * nt + col],
    // results at out[rev_out_off + col]. rev_out_off == NONE: forward only.
    rev_part_off: u32,
    rev_out_off: u32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> p: MatchParams;
@group(0) @binding(1) var<storage, read> desc: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> norms: array<f32>;
@group(0) @binding(3) var<storage, read> entries: array<Entry>;
// Per query row: best index, best score, second score, second index.
@group(0) @binding(4) var<storage, read_write> out: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> rev_part: array<vec4<u32>>;

const BIG: f32 = 3.0e38;
const NONE: u32 = 0xFFFFFFFFu;

// k-major tiles, four consecutive rows per vec4: qs[k * 32 + row / 4][row % 4].
// After the GEMM they double as score scratch for the top-2 folds.
var<workgroup> qs: array<vec4<f32>, 512>;
var<workgroup> ts: array<vec4<f32>, 512>;
var<workgroup> red_i1: array<u32, 2048>;
var<workgroup> red_i2: array<u32, 2048>;

struct Top2 {
    s1: f32,
    i1: u32,
    s2: f32,
    i2: u32,
};

fn before(sa: f32, ia: u32, sb: f32, ib: u32) -> bool {
    return sa < sb || (sa == sb && ia < ib);
}

// Branch-free: the same result as "new best, else new second".
fn insert(t: Top2, s: f32, i: u32) -> Top2 {
    let b1 = before(s, i, t.s1, t.i1);
    let b2 = before(s, i, t.s2, t.i2);
    return Top2(
        select(t.s1, s, b1),
        select(t.i1, i, b1),
        select(select(t.s2, s, b2), t.s1, b1),
        select(select(t.i2, i, b2), t.i1, b1),
    );
}

// Fold shared-memory partial `slot` (scores in qs/ts, indices in red_i*).
fn fold(c: Top2, slot: u32) -> Top2 {
    let r = insert(c, qs[slot / 4u][slot % 4u], red_i1[slot]);
    return insert(r, ts[slot / 4u][slot % 4u], red_i2[slot]);
}

fn store(slot: u32, c: Top2) {
    qs[slot / 4u][slot % 4u] = c.s1;
    ts[slot / 4u][slot % 4u] = c.s2;
    red_i1[slot] = c.i1;
    red_i2[slot] = c.i2;
}

fn load_tile(dst_is_q: bool, row_off: u32, nrows: u32, row0: u32, kc: u32, t: u32) {
    // 128 rows x 4 vec4 (16 k) = 512 vec4, two per thread.
    for (var s = 0u; s < 2u; s = s + 1u) {
        let idx = t + 256u * s;
        let row = idx / 4u;
        let v = idx % 4u;
        var val = vec4<f32>(0.0);
        if (row0 + row < nrows) {
            val = desc[(row_off + row0 + row) * p.dim4 + kc * 4u + v];
        }
        let base = v * 4u * 32u + row / 4u;
        let c = row % 4u;
        if (dst_is_q) {
            qs[base][c] = val.x;
            qs[base + 32u][c] = val.y;
            qs[base + 64u][c] = val.z;
            qs[base + 96u][c] = val.w;
        } else {
            ts[base][c] = val.x;
            ts[base + 32u][c] = val.y;
            ts[base + 64u][c] = val.z;
            ts[base + 96u][c] = val.w;
        }
    }
}

@compute @workgroup_size(256)
fn top2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let e = entries[wid.y];
    let row0 = wid.x * 128u;
    if (row0 >= e.nq) {
        return;
    }
    let t = lid.x;
    let tx = t % 16u;
    let ty = t / 16u;
    let nchunks = p.dim4 / 4u;
    let want_rev = e.rev_out_off != NONE;
    var qn0 = BIG;
    if (row0 + 4u * ty + 0u < e.nq) {
        qn0 = norms[e.q_off + row0 + 4u * ty + 0u];
    }
    var r0 = Top2(BIG, NONE, BIG, NONE);
    var qn1 = BIG;
    if (row0 + 4u * ty + 1u < e.nq) {
        qn1 = norms[e.q_off + row0 + 4u * ty + 1u];
    }
    var r1 = Top2(BIG, NONE, BIG, NONE);
    var qn2 = BIG;
    if (row0 + 4u * ty + 2u < e.nq) {
        qn2 = norms[e.q_off + row0 + 4u * ty + 2u];
    }
    var r2 = Top2(BIG, NONE, BIG, NONE);
    var qn3 = BIG;
    if (row0 + 4u * ty + 3u < e.nq) {
        qn3 = norms[e.q_off + row0 + 4u * ty + 3u];
    }
    var r3 = Top2(BIG, NONE, BIG, NONE);
    var qn4 = BIG;
    if (row0 + 64u + 4u * ty + 0u < e.nq) {
        qn4 = norms[e.q_off + row0 + 64u + 4u * ty + 0u];
    }
    var r4 = Top2(BIG, NONE, BIG, NONE);
    var qn5 = BIG;
    if (row0 + 64u + 4u * ty + 1u < e.nq) {
        qn5 = norms[e.q_off + row0 + 64u + 4u * ty + 1u];
    }
    var r5 = Top2(BIG, NONE, BIG, NONE);
    var qn6 = BIG;
    if (row0 + 64u + 4u * ty + 2u < e.nq) {
        qn6 = norms[e.q_off + row0 + 64u + 4u * ty + 2u];
    }
    var r6 = Top2(BIG, NONE, BIG, NONE);
    var qn7 = BIG;
    if (row0 + 64u + 4u * ty + 3u < e.nq) {
        qn7 = norms[e.q_off + row0 + 64u + 4u * ty + 3u];
    }
    var r7 = Top2(BIG, NONE, BIG, NONE);

    for (var col0 = 0u; col0 < e.nt; col0 = col0 + 128u) {
        var acc0l = vec4<f32>(0.0);
        var acc0h = vec4<f32>(0.0);
        var acc1l = vec4<f32>(0.0);
        var acc1h = vec4<f32>(0.0);
        var acc2l = vec4<f32>(0.0);
        var acc2h = vec4<f32>(0.0);
        var acc3l = vec4<f32>(0.0);
        var acc3h = vec4<f32>(0.0);
        var acc4l = vec4<f32>(0.0);
        var acc4h = vec4<f32>(0.0);
        var acc5l = vec4<f32>(0.0);
        var acc5h = vec4<f32>(0.0);
        var acc6l = vec4<f32>(0.0);
        var acc6h = vec4<f32>(0.0);
        var acc7l = vec4<f32>(0.0);
        var acc7h = vec4<f32>(0.0);
        for (var kc = 0u; kc < nchunks; kc = kc + 1u) {
            load_tile(true, e.q_off, e.nq, row0, kc, t);
            load_tile(false, e.t_off, e.nt, col0, kc, t);
            workgroupBarrier();
            for (var k = 0u; k < 16u; k = k + 1u) {
                let al = qs[k * 32u + ty];
                let ah = qs[k * 32u + 16u + ty];
                let bl = ts[k * 32u + tx];
                let bh = ts[k * 32u + 16u + tx];
                acc0l = acc0l + al.x * bl;
                acc0h = acc0h + al.x * bh;
                acc1l = acc1l + al.y * bl;
                acc1h = acc1h + al.y * bh;
                acc2l = acc2l + al.z * bl;
                acc2h = acc2h + al.z * bh;
                acc3l = acc3l + al.w * bl;
                acc3h = acc3h + al.w * bh;
                acc4l = acc4l + ah.x * bl;
                acc4h = acc4h + ah.x * bh;
                acc5l = acc5l + ah.y * bl;
                acc5h = acc5h + ah.y * bh;
                acc6l = acc6l + ah.z * bl;
                acc6h = acc6h + ah.z * bh;
                acc7l = acc7l + ah.w * bl;
                acc7h = acc7h + ah.w * bh;
            }
            workgroupBarrier();
        }
        {
            let col = col0 + 4u * tx + 0u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0l.x, col);
                r1 = insert(r1, tn - 2.0 * acc1l.x, col);
                r2 = insert(r2, tn - 2.0 * acc2l.x, col);
                r3 = insert(r3, tn - 2.0 * acc3l.x, col);
                r4 = insert(r4, tn - 2.0 * acc4l.x, col);
                r5 = insert(r5, tn - 2.0 * acc5l.x, col);
                r6 = insert(r6, tn - 2.0 * acc6l.x, col);
                r7 = insert(r7, tn - 2.0 * acc7l.x, col);
            }
        }
        {
            let col = col0 + 4u * tx + 1u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0l.y, col);
                r1 = insert(r1, tn - 2.0 * acc1l.y, col);
                r2 = insert(r2, tn - 2.0 * acc2l.y, col);
                r3 = insert(r3, tn - 2.0 * acc3l.y, col);
                r4 = insert(r4, tn - 2.0 * acc4l.y, col);
                r5 = insert(r5, tn - 2.0 * acc5l.y, col);
                r6 = insert(r6, tn - 2.0 * acc6l.y, col);
                r7 = insert(r7, tn - 2.0 * acc7l.y, col);
            }
        }
        {
            let col = col0 + 4u * tx + 2u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0l.z, col);
                r1 = insert(r1, tn - 2.0 * acc1l.z, col);
                r2 = insert(r2, tn - 2.0 * acc2l.z, col);
                r3 = insert(r3, tn - 2.0 * acc3l.z, col);
                r4 = insert(r4, tn - 2.0 * acc4l.z, col);
                r5 = insert(r5, tn - 2.0 * acc5l.z, col);
                r6 = insert(r6, tn - 2.0 * acc6l.z, col);
                r7 = insert(r7, tn - 2.0 * acc7l.z, col);
            }
        }
        {
            let col = col0 + 4u * tx + 3u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0l.w, col);
                r1 = insert(r1, tn - 2.0 * acc1l.w, col);
                r2 = insert(r2, tn - 2.0 * acc2l.w, col);
                r3 = insert(r3, tn - 2.0 * acc3l.w, col);
                r4 = insert(r4, tn - 2.0 * acc4l.w, col);
                r5 = insert(r5, tn - 2.0 * acc5l.w, col);
                r6 = insert(r6, tn - 2.0 * acc6l.w, col);
                r7 = insert(r7, tn - 2.0 * acc7l.w, col);
            }
        }
        {
            let col = col0 + 64u + 4u * tx + 0u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0h.x, col);
                r1 = insert(r1, tn - 2.0 * acc1h.x, col);
                r2 = insert(r2, tn - 2.0 * acc2h.x, col);
                r3 = insert(r3, tn - 2.0 * acc3h.x, col);
                r4 = insert(r4, tn - 2.0 * acc4h.x, col);
                r5 = insert(r5, tn - 2.0 * acc5h.x, col);
                r6 = insert(r6, tn - 2.0 * acc6h.x, col);
                r7 = insert(r7, tn - 2.0 * acc7h.x, col);
            }
        }
        {
            let col = col0 + 64u + 4u * tx + 1u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0h.y, col);
                r1 = insert(r1, tn - 2.0 * acc1h.y, col);
                r2 = insert(r2, tn - 2.0 * acc2h.y, col);
                r3 = insert(r3, tn - 2.0 * acc3h.y, col);
                r4 = insert(r4, tn - 2.0 * acc4h.y, col);
                r5 = insert(r5, tn - 2.0 * acc5h.y, col);
                r6 = insert(r6, tn - 2.0 * acc6h.y, col);
                r7 = insert(r7, tn - 2.0 * acc7h.y, col);
            }
        }
        {
            let col = col0 + 64u + 4u * tx + 2u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0h.z, col);
                r1 = insert(r1, tn - 2.0 * acc1h.z, col);
                r2 = insert(r2, tn - 2.0 * acc2h.z, col);
                r3 = insert(r3, tn - 2.0 * acc3h.z, col);
                r4 = insert(r4, tn - 2.0 * acc4h.z, col);
                r5 = insert(r5, tn - 2.0 * acc5h.z, col);
                r6 = insert(r6, tn - 2.0 * acc6h.z, col);
                r7 = insert(r7, tn - 2.0 * acc7h.z, col);
            }
        }
        {
            let col = col0 + 64u + 4u * tx + 3u;
            if (col < e.nt) {
                let tn = norms[e.t_off + col];
                r0 = insert(r0, tn - 2.0 * acc0h.w, col);
                r1 = insert(r1, tn - 2.0 * acc1h.w, col);
                r2 = insert(r2, tn - 2.0 * acc2h.w, col);
                r3 = insert(r3, tn - 2.0 * acc3h.w, col);
                r4 = insert(r4, tn - 2.0 * acc4h.w, col);
                r5 = insert(r5, tn - 2.0 * acc5h.w, col);
                r6 = insert(r6, tn - 2.0 * acc6h.w, col);
                r7 = insert(r7, tn - 2.0 * acc7h.w, col);
            }
        }
        if (want_rev) {
            // Column partials over this thread's 8 rows, then over the 16
            // row-threads (ty) of each column in shared memory.
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0l.x, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1l.x, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2l.x, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3l.x, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4l.x, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5l.x, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6l.x, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7l.x, row0 + 64u + 4u * ty + 3u);
                }
                store((4u * tx + 0u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0l.y, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1l.y, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2l.y, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3l.y, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4l.y, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5l.y, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6l.y, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7l.y, row0 + 64u + 4u * ty + 3u);
                }
                store((4u * tx + 1u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0l.z, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1l.z, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2l.z, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3l.z, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4l.z, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5l.z, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6l.z, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7l.z, row0 + 64u + 4u * ty + 3u);
                }
                store((4u * tx + 2u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0l.w, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1l.w, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2l.w, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3l.w, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4l.w, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5l.w, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6l.w, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7l.w, row0 + 64u + 4u * ty + 3u);
                }
                store((4u * tx + 3u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0h.x, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1h.x, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2h.x, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3h.x, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4h.x, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5h.x, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6h.x, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7h.x, row0 + 64u + 4u * ty + 3u);
                }
                store((64u + 4u * tx + 0u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0h.y, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1h.y, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2h.y, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3h.y, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4h.y, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5h.y, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6h.y, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7h.y, row0 + 64u + 4u * ty + 3u);
                }
                store((64u + 4u * tx + 1u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0h.z, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1h.z, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2h.z, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3h.z, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4h.z, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5h.z, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6h.z, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7h.z, row0 + 64u + 4u * ty + 3u);
                }
                store((64u + 4u * tx + 2u) * 16u + ty, c);
            }
            {
                var c = Top2(BIG, NONE, BIG, NONE);
                if (row0 + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn0 - 2.0 * acc0h.w, row0 + 4u * ty + 0u);
                }
                if (row0 + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn1 - 2.0 * acc1h.w, row0 + 4u * ty + 1u);
                }
                if (row0 + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn2 - 2.0 * acc2h.w, row0 + 4u * ty + 2u);
                }
                if (row0 + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn3 - 2.0 * acc3h.w, row0 + 4u * ty + 3u);
                }
                if (row0 + 64u + 4u * ty + 0u < e.nq) {
                    c = insert(c, qn4 - 2.0 * acc4h.w, row0 + 64u + 4u * ty + 0u);
                }
                if (row0 + 64u + 4u * ty + 1u < e.nq) {
                    c = insert(c, qn5 - 2.0 * acc5h.w, row0 + 64u + 4u * ty + 1u);
                }
                if (row0 + 64u + 4u * ty + 2u < e.nq) {
                    c = insert(c, qn6 - 2.0 * acc6h.w, row0 + 64u + 4u * ty + 2u);
                }
                if (row0 + 64u + 4u * ty + 3u < e.nq) {
                    c = insert(c, qn7 - 2.0 * acc7h.w, row0 + 64u + 4u * ty + 3u);
                }
                store((64u + 4u * tx + 3u) * 16u + ty, c);
            }
        }
        workgroupBarrier();
        // Two-level fold of the 16 row-thread partials of each column:
        // 2 threads x 8 partials, then 1 thread x 2.
        let fold_base = (t % 128u) * 16u + (t / 128u) * 8u;
        if (want_rev) {
            var c = Top2(BIG, NONE, BIG, NONE);
            for (var y = 0u; y < 8u; y = y + 1u) {
                c = fold(c, fold_base + y);
            }
            store(fold_base, c);
        }
        workgroupBarrier();
        if (want_rev && t < 128u && col0 + t < e.nt) {
            var c = Top2(BIG, NONE, BIG, NONE);
            c = fold(c, t * 16u);
            c = fold(c, t * 16u + 8u);
            rev_part[e.rev_part_off + wid.x * e.nt + col0 + t] =
                vec4<u32>(c.i1, bitcast<u32>(c.s1), bitcast<u32>(c.s2), c.i2);
        }
        workgroupBarrier();
    }

    // Reduce the 16 column-threads of each row: (row, tx) partials.
    store((4u * ty + 0u) * 16u + tx, r0);
    store((4u * ty + 1u) * 16u + tx, r1);
    store((4u * ty + 2u) * 16u + tx, r2);
    store((4u * ty + 3u) * 16u + tx, r3);
    store((64u + 4u * ty + 0u) * 16u + tx, r4);
    store((64u + 4u * ty + 1u) * 16u + tx, r5);
    store((64u + 4u * ty + 2u) * 16u + tx, r6);
    store((64u + 4u * ty + 3u) * 16u + tx, r7);
    workgroupBarrier();
    if (t < 128u && row0 + t < e.nq) {
        var r = Top2(BIG, NONE, BIG, NONE);
        for (var x = 0u; x < 16u; x = x + 1u) {
            r = fold(r, t * 16u + x);
        }
        out[e.out_off + row0 + t] = vec4<u32>(r.i1, bitcast<u32>(r.s1), bitcast<u32>(r.s2), r.i2);
    }
}

// Fold the per-row-block column partials of `top2` into the reverse top-2.
@compute @workgroup_size(64)
fn rev_reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let e = entries[wid.y];
    let col = gid.x;
    if (e.rev_out_off == NONE || col >= e.nt) {
        return;
    }
    let nblk = (e.nq + 127u) / 128u;
    var c = Top2(BIG, NONE, BIG, NONE);
    for (var b = 0u; b < nblk; b = b + 1u) {
        let w = rev_part[e.rev_part_off + b * e.nt + col];
        c = insert(c, bitcast<f32>(w.y), w.x);
        c = insert(c, bitcast<f32>(w.z), w.w);
    }
    out[e.rev_out_off + col] = vec4<u32>(c.i1, bitcast<u32>(c.s1), bitcast<u32>(c.s2), c.i2);
}
