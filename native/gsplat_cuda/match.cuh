// Exhaustive SIFT matching on the GPU (visloc_gsplat_cuda_match_*), included
// at the end of gsplat_cuda.cu: per image pair, the top-2 scores in both
// directions that visloc_vision's BruteForceMatcher::
// match_descriptors_cross_checked takes its ratio-tested mutual nearest
// neighbours from. The host applies the same final filter.
//
// Descriptors are 128 bytes (SIFT stored as byte / 512, or raw bytes), so
// every dot product is an exact integer: the scores (|t|^2 - 2 q.t forward,
// |q|^2 - 2 q.t reverse, in byte units) are the CPU's exactly. The best is
// the lowest score, ties to the lower index (the CPU's strict-< first-wins
// scan), and the second is the lowest score of the other rows.

namespace {

constexpr int kMatchWords = 32;  // 128 bytes per descriptor
constexpr int kMatchTile = 128;  // queries per block, trains per tile
constexpr int kNoScore = 0x7fffffff;

// Best (score, index) and the lowest score of the others.
struct Top2 {
    int best, idx, second;
};

__device__ inline Top2 top2_empty() { return {kNoScore, kNoScore, kNoScore}; }

__device__ inline Top2 top2_merge(Top2 a, Top2 b) {
    if (b.best < a.best || (b.best == a.best && b.idx < a.idx)) return {b.best, b.idx, min(b.second, a.best)};
    return {a.best, a.idx, min(a.second, b.best)};
}

__device__ inline Top2 top2_shfl_xor(Top2 s, int m) {
    return {__shfl_xor_sync(0xffffffffu, s.best, m), __shfl_xor_sync(0xffffffffu, s.idx, m),
            __shfl_xor_sync(0xffffffffu, s.second, m)};
}

// Sum of squared bytes per descriptor.
__global__ void match_norm_kernel(uint32_t n, const uint32_t* __restrict__ words, int* __restrict__ norms) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned s = 0;
    for (int k = 0; k < kMatchWords; ++k) {
        const uint32_t w = words[size_t(i) * kMatchWords + k];
        s = __dp4a(w, w, s);
    }
    norms[i] = int(s);
}

// Load `count` descriptors from `first` into a tile transposed by word
// (tile[k][row]), zeros past the end.
__device__ inline void match_load_tile(uint32_t (*tile)[kMatchTile], const uint32_t* __restrict__ words,
                                       uint32_t first, uint32_t count) {
    // Thread: row tid % 128, words (tid / 128) * 16 .. + 16.
    const int row = threadIdx.x % kMatchTile;
    const int k0 = (threadIdx.x / kMatchTile) * 16;
    if (uint32_t(row) < count) {
        const uint4* src = reinterpret_cast<const uint4*>(words + size_t(first + row) * kMatchWords + k0);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 w = src[v];
            tile[k0 + v * 4 + 0][row] = w.x;
            tile[k0 + v * 4 + 1][row] = w.y;
            tile[k0 + v * 4 + 2][row] = w.z;
            tile[k0 + v * 4 + 3][row] = w.w;
        }
    } else {
#pragma unroll
        for (int k = 0; k < 16; ++k) tile[k0 + k][row] = 0u;
    }
}

// One pair: block b takes queries b*128 .. +128 against every train tile.
// Forward top-2 per query is complete in the block (fwd: 3 ints per query);
// reverse top-2 per train over this block's queries goes to
// partial[b * nt + t] for match_reduce_kernel. 256 threads: thread (ty, tx)
// = (tid / 16, tid % 16) computes queries ty*8 .. +8 x trains tx*8 .. +8.
__global__ void __launch_bounds__(256) match_kernel(const uint32_t* __restrict__ words, const int* __restrict__ norms,
                                                    uint32_t q_first, uint32_t nq, uint32_t t_first, uint32_t nt,
                                                    int* __restrict__ fwd, Top2* __restrict__ partial) {
    __shared__ __align__(16) uint32_t sq[kMatchWords][kMatchTile];
    __shared__ __align__(16) uint32_t st[kMatchWords][kMatchTile];
    __shared__ Top2 sr[8][kMatchTile];
    const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16, warp = tid / 32;
    const uint32_t qb = blockIdx.x * kMatchTile;
    const uint32_t q_count = min(uint32_t(kMatchTile), nq - qb);
    match_load_tile(sq, words, q_first + qb, q_count);
    int qnorm[8];
#pragma unroll
    for (int a = 0; a < 8; ++a) {
        const uint32_t q = qb + ty * 8 + a;
        qnorm[a] = q < nq ? norms[q_first + q] : 0;
    }
    Top2 f[8];
#pragma unroll
    for (int a = 0; a < 8; ++a) f[a] = top2_empty();

    for (uint32_t tb = 0; tb < nt; tb += kMatchTile) {
        const uint32_t t_count = min(uint32_t(kMatchTile), nt - tb);
        __syncthreads();  // the previous tile is no longer read
        match_load_tile(st, words, t_first + tb, t_count);
        __syncthreads();
        unsigned acc[8][8];
#pragma unroll
        for (int a = 0; a < 8; ++a)
#pragma unroll
            for (int c = 0; c < 8; ++c) acc[a][c] = 0u;
#pragma unroll 4
        for (int k = 0; k < kMatchWords; ++k) {
            const uint4 q0 = *reinterpret_cast<const uint4*>(&sq[k][ty * 8]);
            const uint4 q1 = *reinterpret_cast<const uint4*>(&sq[k][ty * 8 + 4]);
            const uint4 t0 = *reinterpret_cast<const uint4*>(&st[k][tx * 8]);
            const uint4 t1 = *reinterpret_cast<const uint4*>(&st[k][tx * 8 + 4]);
            const uint32_t qv[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            const uint32_t tv[8] = {t0.x, t0.y, t0.z, t0.w, t1.x, t1.y, t1.z, t1.w};
#pragma unroll
            for (int a = 0; a < 8; ++a)
#pragma unroll
                for (int c = 0; c < 8; ++c) acc[a][c] = __dp4a(qv[a], tv[c], acc[a][c]);
        }
        // Scores; this thread's reverse top-2 per train over its 8 queries.
        Top2 r[8];
#pragma unroll
        for (int c = 0; c < 8; ++c) r[c] = top2_empty();
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const uint32_t t = tb + tx * 8 + c;
            if (t >= nt) continue;
            const int tnorm = norms[t_first + t];
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const uint32_t q = qb + ty * 8 + a;
                if (q >= nq) continue;
                const int dot2 = 2 * int(acc[a][c]);
                f[a] = top2_merge(f[a], {tnorm - dot2, int(t), kNoScore});
                r[c] = top2_merge(r[c], {qnorm[a] - dot2, int(q), kNoScore});
            }
        }
        // Reverse: merge the two halves of the warp (ty even / odd), then
        // the 8 warps through shared memory.
#pragma unroll
        for (int c = 0; c < 8; ++c) r[c] = top2_merge(r[c], top2_shfl_xor(r[c], 16));
        if ((tid & 16) == 0) {
#pragma unroll
            for (int c = 0; c < 8; ++c) sr[warp][tx * 8 + c] = r[c];
        }
        __syncthreads();
        if (tid < kMatchTile && uint32_t(tid) < t_count) {
            Top2 s = sr[0][tid];
#pragma unroll
            for (int w = 1; w < 8; ++w) s = top2_merge(s, sr[w][tid]);
            partial[size_t(blockIdx.x) * nt + tb + tid] = s;
        }
    }
    // Forward: merge the 16 threads of each query row (lanes of one half-warp).
#pragma unroll
    for (int a = 0; a < 8; ++a)
#pragma unroll
        for (int m = 1; m < 16; m <<= 1) f[a] = top2_merge(f[a], top2_shfl_xor(f[a], m));
    if (tx == 0) {
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            const uint32_t q = qb + ty * 8 + a;
            if (q < nq) {
                fwd[size_t(q) * 3 + 0] = f[a].best;
                fwd[size_t(q) * 3 + 1] = f[a].idx;
                fwd[size_t(q) * 3 + 2] = f[a].second;
            }
        }
    }
}

// Reverse top-2 per train: merge the query blocks' partial states.
__global__ void match_reduce_kernel(uint32_t nt, uint32_t blocks, const Top2* __restrict__ partial,
                                    int* __restrict__ rev) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    Top2 s = top2_empty();
    for (uint32_t b = 0; b < blocks; ++b) s = top2_merge(s, partial[size_t(b) * nt + t]);
    rev[size_t(t) * 3 + 0] = s.best;
    rev[size_t(t) * 3 + 1] = s.idx;
    rev[size_t(t) * 3 + 2] = s.second;
}

}  // namespace

extern "C" {

// Upload the descriptors of `num_images` images for matching: counts[i]
// descriptors of 128 bytes each, image after image. Replaces the previous
// set.
GSC_EXPORT int visloc_gsplat_cuda_match_set(void* handle, int num_images, const uint32_t* counts,
                                            const uint8_t* descriptors) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (num_images < 0 || (num_images > 0 && (!counts || !descriptors))) {
        ctx->error = "match_set: counts and descriptors are required";
        return -1;
    }
    ctx->match_first.assign(size_t(num_images) + 1, 0);
    for (int i = 0; i < num_images; ++i) ctx->match_first[i + 1] = ctx->match_first[i] + counts[i];
    const size_t n = ctx->match_first.back();
    uint32_t* words;
    int* norms;
    GSC_TRY(ensure(ctx, ctx->match_words, std::max<size_t>(n, 1) * kMatchWords, &words));
    GSC_TRY(ensure(ctx, ctx->match_norms, std::max<size_t>(n, 1), &norms));
    if (n > 0) {
        GSC_CHECK(ctx, cudaMemcpy(words, descriptors, n * kMatchWords * 4, cudaMemcpyHostToDevice));
        match_norm_kernel<<<blocks_for(n), 256>>>(uint32_t(n), words, norms);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    return 0;
}

// Top-2 scores of `num_pairs` image pairs (pairs[2k], pairs[2k + 1] = query
// and train image): for pair k, `fwd` receives from its offset (the sum of
// the earlier pairs' query counts, times 3) per query descriptor (best
// score, its train index, second score), and `rev` likewise per train
// descriptor (best score, query index, second score). Scores are in byte
// units; 0x7fffffff means none.
GSC_EXPORT int visloc_gsplat_cuda_match_top2(void* handle, int num_pairs, const uint32_t* pairs, int32_t* fwd,
                                             int32_t* rev) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    const size_t num_images = ctx->match_first.empty() ? 0 : ctx->match_first.size() - 1;
    size_t fwd_total = 0, rev_total = 0, partial_max = 0;
    for (int k = 0; k < num_pairs; ++k) {
        const uint32_t i = pairs[2 * k], j = pairs[2 * k + 1];
        if (i >= num_images || j >= num_images) {
            ctx->error = "match_top2: image index out of range";
            return -1;
        }
        const size_t nq = ctx->match_first[i + 1] - ctx->match_first[i];
        const size_t nt = ctx->match_first[j + 1] - ctx->match_first[j];
        fwd_total += nq;
        rev_total += nt;
        partial_max = std::max(partial_max, ((nq + kMatchTile - 1) / kMatchTile) * nt);
    }
    int *d_fwd, *d_rev;
    Top2* partial;
    GSC_TRY(ensure(ctx, ctx->match_fwd, std::max<size_t>(fwd_total, 1) * 3, &d_fwd));
    GSC_TRY(ensure(ctx, ctx->match_rev, std::max<size_t>(rev_total, 1) * 3, &d_rev));
    GSC_TRY(ensure(ctx, ctx->match_partial, std::max<size_t>(partial_max, 1), &partial));
    const auto* words = static_cast<const uint32_t*>(ctx->match_words.ptr);
    const auto* norms = static_cast<const int*>(ctx->match_norms.ptr);
    size_t fo = 0, ro = 0;
    for (int k = 0; k < num_pairs; ++k) {
        const uint32_t i = pairs[2 * k], j = pairs[2 * k + 1];
        const uint32_t q_first = ctx->match_first[i], nq = ctx->match_first[i + 1] - q_first;
        const uint32_t t_first = ctx->match_first[j], nt = ctx->match_first[j + 1] - t_first;
        if (nq > 0 && nt > 0) {
            const uint32_t blocks = (nq + kMatchTile - 1) / kMatchTile;
            match_kernel<<<blocks, 256>>>(words, norms, q_first, nq, t_first, nt, d_fwd + fo * 3, partial);
            GSC_CHECK(ctx, cudaGetLastError());
            match_reduce_kernel<<<blocks_for(nt), 256>>>(nt, blocks, partial, d_rev + ro * 3);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        // A pair with an empty side has no matches; its rows are left as
        // they are and the host skips them.
        fo += nq;
        ro += nt;
    }
    if (fwd && fo > 0) GSC_CHECK(ctx, cudaMemcpy(fwd, d_fwd, fo * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    if (rev && ro > 0) GSC_CHECK(ctx, cudaMemcpy(rev, d_rev, ro * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    return 0;
}

}  // extern "C"
