// GPU TSDF fusion of the scene's median depth and surface-nets extraction,
// included at the end of gsplat_cuda.cu. The maths is
// visloc_gsplat_core::mesh::Tsdf's (8^3-voxel blocks allocated over the
// truncation band of every surface pixel, running means of the truncated
// SDF and colour, free-space carving, surface nets on the voxels fused from
// at least `min_weight` views), with two differences:
//
// - Every block is allocated before any view is fused, so each block sees
//   every view (Tsdf::integrate allocates as it goes: a block first touched
//   by view k misses views 0..k).
// - The blocks are fused and extracted in slabs along x of at most
//   `slab_blocks` blocks, plus one layer of overlap above the slab for the
//   cells on its top face, so the device holds one slab's voxels at a time.
//   The views' depth and colour are kept on the device when they fit in
//   `cache_mb` (16 bytes per pixel); otherwise each slab renders every view
//   again.
//
// The host gets one vertex per cell with a surface (cell origin, position,
// colour) and every grid edge with a sign change from such a cell; the
// quads around those edges are assembled from the cells on the Rust side
// (crates/gsplat-cuda/src/mesh.rs), across slabs.

namespace {

constexpr int kTsdfB = 8;  // block edge in voxels
constexpr int kTsdfBV = kTsdfB * kTsdfB * kTsdfB;
constexpr unsigned long long kTsdfEmpty = ~0ull;
constexpr int kTsdfOff = 1 << 20;  // block coordinates must lie in -2^20 .. 2^20 - 2

__host__ __device__ inline unsigned long long tsdf_key(int x, int y, int z) {
    return (static_cast<unsigned long long>(x + kTsdfOff) << 42) |
           (static_cast<unsigned long long>(y + kTsdfOff) << 21) | static_cast<unsigned long long>(z + kTsdfOff);
}
__host__ __device__ inline int tsdf_key_x(unsigned long long k) { return int((k >> 42) & 0x1FFFFF) - kTsdfOff; }
__host__ __device__ inline int tsdf_key_y(unsigned long long k) { return int((k >> 21) & 0x1FFFFF) - kTsdfOff; }
__host__ __device__ inline int tsdf_key_z(unsigned long long k) { return int(k & 0x1FFFFF) - kTsdfOff; }

__device__ inline unsigned long long tsdf_hash(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

__device__ inline int floor_div8(int a) { return a >= 0 ? a / kTsdfB : -((-a + kTsdfB - 1) / kTsdfB); }

// Insert `key` into the open-addressing set `table` (capacity mask + 1);
// counts[0] += 1 for a new key, counts[1] = 1 when the table is full.
__device__ inline void tsdf_insert(unsigned long long* table, uint32_t mask, unsigned long long key,
                                   uint32_t* counts) {
    uint32_t h = uint32_t(tsdf_hash(key)) & mask;
    for (uint32_t probe = 0; probe <= mask; ++probe) {
        const unsigned long long prev = atomicCAS(&table[h], kTsdfEmpty, key);
        if (prev == kTsdfEmpty) {
            atomicAdd(&counts[0], 1u);
            return;
        }
        if (prev == key) return;
        h = (h + 1) & mask;
    }
    counts[1] = 1;
}

// Index of `key` in a slab map (keys and their indices), -1 if absent.
__device__ inline int tsdf_find(const unsigned long long* keys, const uint32_t* vals, uint32_t mask,
                                unsigned long long key) {
    uint32_t h = uint32_t(tsdf_hash(key)) & mask;
    for (uint32_t probe = 0; probe <= mask; ++probe) {
        const unsigned long long k = keys[h];
        if (k == key) return int(vals[h]);
        if (k == kTsdfEmpty) return -1;
        h = (h + 1) & mask;
    }
    return -1;
}

struct TsdfArgs {
    float voxel, trunc, max_depth, min_weight;
    int carve, free_space;
};

// Allocation: for every second pixel in x and y with a surface (median depth
// in (0, max_depth]), the blocks its ray crosses within +-trunc of the
// surface, sampled no further apart than half a block (Tsdf::integrate).
__global__ void tsdf_alloc_kernel(GscCamera cam, TsdfArgs a, const float* __restrict__ depth_out,
                                  unsigned long long* table, uint32_t mask, uint32_t* counts) {
    const uint32_t hw = (cam.width + 1) / 2, hh = (cam.height + 1) / 2;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hw * hh) return;
    const uint32_t x = (i % hw) * 2, y = (i / hw) * 2;
    const float z = depth_out[(size_t(y) * cam.width + x) * 2];
    if (z <= 0.f || z > a.max_depth) return;
    const float pc[3] = {(float(x) + 0.5f - cam.cx) / cam.fx * z, (float(y) + 0.5f - cam.cy) / cam.fy * z, z};
    const float d[3] = {pc[0] - cam.t[0], pc[1] - cam.t[1], pc[2] - cam.t[2]};
    const float* R = cam.rot;
    float p[3], dir[3];
    for (int r = 0; r < 3; ++r) p[r] = R[r] * d[0] + R[3 + r] * d[1] + R[6 + r] * d[2];  // R^T (pc - t)
    for (int r = 0; r < 3; ++r) dir[r] = p[r] - cam.center[r];
    const float dn = sqrtf(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (int r = 0; r < 3; ++r) dir[r] /= dn;
    const float step = fminf(a.voxel * kTsdfB * 0.5f, a.trunc);
    const int n_steps = int(ceilf(2.f * a.trunc / step));
    unsigned long long last = kTsdfEmpty;
    for (int k = 0; k <= n_steps; ++k) {
        const float s = -a.trunc + float(k) * step;
        int b[3];
        for (int r = 0; r < 3; ++r) b[r] = floor_div8(int(floorf((p[r] + dir[r] * s) / a.voxel)));
        const unsigned long long key = tsdf_key(b[0], b[1], b[2]);
        if (key != last) tsdf_insert(table, mask, key, counts);
        last = key;
    }
}

// Copy the set's keys to a dense array (any order).
__global__ void tsdf_collect_kernel(const unsigned long long* __restrict__ table, uint32_t cap,
                                    unsigned long long* __restrict__ out, uint32_t* counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cap || table[i] == kTsdfEmpty) return;
    out[atomicAdd(&counts[2], 1u)] = table[i];
}

// Insert n distinct keys into a table with room for them: a slab map
// (keys[i] -> i in `vals`), or with `vals` null a block set being regrown.
__global__ void tsdf_map_kernel(const unsigned long long* __restrict__ slab, uint32_t n, unsigned long long* keys,
                                uint32_t* vals, uint32_t mask) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long key = slab[i];
    uint32_t h = uint32_t(tsdf_hash(key)) & mask;
    while (true) {
        const unsigned long long prev = atomicCAS(&keys[h], kTsdfEmpty, key);
        if (prev == kTsdfEmpty || prev == key) {
            if (vals) vals[h] = i;
            return;
        }
        h = (h + 1) & mask;
    }
}

__global__ void tsdf_reset_kernel(size_t n, float* sdf, float* w, float* col) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sdf[i] = 1.f;
    w[i] = 0.f;
    col[i * 3] = col[i * 3 + 1] = col[i * 3 + 2] = 0.f;
}

// Fuse one view into the slab's voxels (one block of threads per block, two
// voxels per thread), as Tsdf::integrate. A block whose bounding sphere lies
// outside one of the frustum's side / near planes (`planes`, unit normals in
// camera space) is skipped: none of its voxels projects into the image.
// `depth` holds the median depth every `depth_stride` floats (2 in
// depth_out, 1 in the view cache).
__global__ void __launch_bounds__(kBlock) tsdf_integrate_kernel(GscCamera cam, TsdfArgs a, float4 p0, float4 p1,
                                                                float4 p2, float4 p3, float4 p4,
                                                                const unsigned long long* __restrict__ slab,
                                                                const float* __restrict__ depth, int depth_stride,
                                                                const float* __restrict__ rgb, float* sdf, float* wt,
                                                                float* col) {
    const unsigned long long key = slab[blockIdx.x];
    const int bx = tsdf_key_x(key) * kTsdfB, by = tsdf_key_y(key) * kTsdfB, bz = tsdf_key_z(key) * kTsdfB;
    const float* R = cam.rot;
    const float half = 0.5f * kTsdfB * a.voxel;
    {
        const float c[3] = {float(bx) * a.voxel + half, float(by) * a.voxel + half, float(bz) * a.voxel + half};
        float cc[3];
        for (int r = 0; r < 3; ++r) cc[r] = R[r * 3] * c[0] + R[r * 3 + 1] * c[1] + R[r * 3 + 2] * c[2] + cam.t[r];
        const float radius = half * 1.7320508f;
        const float4 planes[5] = {p0, p1, p2, p3, p4};
        for (int k = 0; k < 5; ++k)
            if (planes[k].x * cc[0] + planes[k].y * cc[1] + planes[k].z * cc[2] < -radius) return;
    }
    const float wf = float(cam.width), hf = float(cam.height);
    for (int v = threadIdx.x; v < kTsdfBV; v += blockDim.x) {
        const int lx = v % kTsdfB, ly = (v / kTsdfB) % kTsdfB, lz = v / (kTsdfB * kTsdfB);
        const float p[3] = {(float(bx + lx) + 0.5f) * a.voxel, (float(by + ly) + 0.5f) * a.voxel,
                            (float(bz + lz) + 0.5f) * a.voxel};
        float pc[3];
        for (int r = 0; r < 3; ++r) pc[r] = R[r * 3] * p[0] + R[r * 3 + 1] * p[1] + R[r * 3 + 2] * p[2] + cam.t[r];
        if (pc[2] <= 1e-6f) continue;
        const float u = cam.fx * pc[0] / pc[2] + cam.cx;
        const float vv = cam.fy * pc[1] / pc[2] + cam.cy;
        if (u < 0.f || vv < 0.f || u >= wf || vv >= hf) continue;
        const size_t pix = size_t(vv) * cam.width + size_t(u);
        const float d = depth[pix * depth_stride];
        const size_t i = size_t(blockIdx.x) * kTsdfBV + v;
        if (d <= 0.f || d > a.max_depth) {
            // Nothing opaque (or only background) along this pixel: free space.
            if (a.carve && (d <= 0.f || pc[2] < d - a.trunc)) {
                const float w0 = wt[i], wn = w0 + 1.f;
                sdf[i] = (sdf[i] * w0 + 1.f) / wn;
                wt[i] = wn;
            }
            continue;
        }
        const float s = d - pc[2];
        if (s < -a.trunc || (!a.free_space && s > a.trunc)) continue;
        const float t = fminf(s / a.trunc, 1.f);
        const float w0 = wt[i], wn = w0 + 1.f;
        sdf[i] = (sdf[i] * w0 + t) / wn;
        for (int c = 0; c < 3; ++c) col[i * 3 + c] = (col[i * 3 + c] * w0 + rgb[pix * 3 + c]) / wn;
        wt[i] = wn;
    }
}

struct TsdfSlab {
    const unsigned long long* keys;  // slab map
    const uint32_t* vals;
    uint32_t mask;
    const float *sdf, *w, *col;
    float min_weight;
};

// The fused voxel (global voxel coordinates) and its colour (nullable), if
// its block is in the slab and it was fused from at least min_weight views.
__device__ inline bool tsdf_sample(const TsdfSlab& s, int vx, int vy, int vz, float& sdf, float* col) {
    const int bx = floor_div8(vx), by = floor_div8(vy), bz = floor_div8(vz);
    const int b = tsdf_find(s.keys, s.vals, s.mask, tsdf_key(bx, by, bz));
    if (b < 0) return false;
    const size_t i =
        size_t(b) * kTsdfBV + (vx - bx * kTsdfB) + (vy - by * kTsdfB) * kTsdfB + (vz - bz * kTsdfB) * kTsdfB * kTsdfB;
    if (!(s.w[i] >= fmaxf(s.min_weight, 1e-6f))) return false;
    sdf = s.sdf[i];
    if (col) {
        col[0] = s.col[i * 3];
        col[1] = s.col[i * 3 + 1];
        col[2] = s.col[i * 3 + 2];
    }
    return true;
}

__device__ inline bool tsdf_crosses(float a, float b) {
    return ((a < 0.f) != (b < 0.f)) && fabsf(a) < 0.999f && fabsf(b) < 0.999f;
}

// Surface nets over the cells whose origin voxel lies in one of the slab's
// own blocks (the first `n_own`): one vertex per cell with all 8 corners
// known and a crossing edge (the mean of the edge crossings, colour likewise),
// and, from each such cell, every edge leaving its origin along +x, +y, +z
// with a sign change (Tsdf::extract). counts[3] / counts[4] count vertices /
// edges; records past `cap_v` / `cap_e` are dropped (the host retries).
__global__ void tsdf_extract_kernel(TsdfSlab s, const unsigned long long* __restrict__ slab, float voxel,
                                    int32_t* __restrict__ out_cells, float* __restrict__ out_pos,
                                    uint8_t* __restrict__ out_col, int32_t* __restrict__ out_edges, uint32_t cap_v,
                                    uint32_t cap_e, uint32_t* counts) {
    const unsigned long long key = slab[blockIdx.x];
    const int bx = tsdf_key_x(key) * kTsdfB, by = tsdf_key_y(key) * kTsdfB, bz = tsdf_key_z(key) * kTsdfB;
    for (int v = threadIdx.x; v < kTsdfBV; v += blockDim.x) {
        const int o[3] = {bx + v % kTsdfB, by + (v / kTsdfB) % kTsdfB, bz + v / (kTsdfB * kTsdfB)};
        float val[8], cv[8][3];
        bool ok = true;
        for (int c = 0; c < 8 && ok; ++c)
            ok = tsdf_sample(s, o[0] + (c & 1), o[1] + ((c >> 1) & 1), o[2] + ((c >> 2) & 1), val[c], cv[c]);
        if (!ok) continue;
        // Tsdf::extract's EDGES, corner c at (c & 1, (c >> 1) & 1, (c >> 2) & 1).
        const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3},
                                  {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        float sum[3] = {0.f, 0.f, 0.f}, cs[3] = {0.f, 0.f, 0.f};
        int n = 0;
        for (int e = 0; e < 12; ++e) {
            const int ea = edges[e][0], eb = edges[e][1];
            if (!tsdf_crosses(val[ea], val[eb])) continue;
            const float t = val[ea] / (val[ea] - val[eb]);
            const float pa[3] = {float(ea & 1), float((ea >> 1) & 1), float((ea >> 2) & 1)};
            const float pb[3] = {float(eb & 1), float((eb >> 1) & 1), float((eb >> 2) & 1)};
            for (int r = 0; r < 3; ++r) {
                sum[r] += pa[r] + (pb[r] - pa[r]) * t;
                cs[r] += cv[ea][r] * (1.f - t) + cv[eb][r] * t;
            }
            ++n;
        }
        if (n == 0) continue;
        const uint32_t vi = atomicAdd(&counts[3], 1u);
        if (vi < cap_v) {
            for (int r = 0; r < 3; ++r) {
                out_cells[size_t(vi) * 3 + r] = o[r];
                out_pos[size_t(vi) * 3 + r] = (float(o[r]) + 0.5f + sum[r] / float(n)) * voxel;
                out_col[size_t(vi) * 3 + r] = uint8_t(roundf(clampf(cs[r] / float(n), 0.f, 1.f) * 255.f));
            }
        }
        // Edges from the origin: corner 1 (+x), 2 (+y), 4 (+z).
        for (int axis = 0; axis < 3; ++axis) {
            if (!tsdf_crosses(val[0], val[1 << axis])) continue;
            const uint32_t ei = atomicAdd(&counts[4], 1u);
            if (ei < cap_e) {
                int32_t* rec = out_edges + size_t(ei) * 4;
                rec[0] = o[0];
                rec[1] = o[1];
                rec[2] = o[2];
                rec[3] = axis | (val[0] > 0.f ? 4 : 0);
            }
        }
    }
}

// Replace each pixel's median depth (depth_out: median, opacity) by the
// expected depth acc / opacity where the opacity exceeds 0.5, 0 elsewhere
// (the median's "no surface").
__global__ void expected_depth_kernel(uint32_t npix, const float* __restrict__ acc, float* __restrict__ depth_out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    const float a = depth_out[i * 2 + 1];
    depth_out[i * 2] = a > 0.5f ? acc[i] / a : 0.f;
}

uint32_t next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return uint32_t(p);
}

// Unit normals (camera space) of the frustum's side and near planes: a point
// with n . p < 0 for one of them projects outside the image or behind the
// camera (Tsdf::integrate's culling planes).
void frustum_planes(const GscCamera& c, float4 out[5]) {
    const float wf = float(c.width), hf = float(c.height);
    const float raw[5][3] = {{c.fx, 0.f, c.cx}, {-c.fx, 0.f, wf - c.cx}, {0.f, c.fy, c.cy}, {0.f, -c.fy, hf - c.cy},
                             {0.f, 0.f, 1.f}};
    for (int k = 0; k < 5; ++k) {
        const float n = std::sqrt(raw[k][0] * raw[k][0] + raw[k][1] * raw[k][1] + raw[k][2] * raw[k][2]);
        out[k] = make_float4(raw[k][0] / n, raw[k][1] / n, raw[k][2] / n, 0.f);
    }
}

}  // namespace

extern "C" {

// Settings of visloc_gsplat_cuda_tsdf_mesh. Mirrored by `TsdfParams` in
// crates/gsplat-cuda/src/lib.rs.
struct GscTsdfParams {
    float voxel;       // voxel edge (world units)
    float trunc;       // truncation distance (world units)
    float max_depth;   // depths beyond are background
    float min_weight;  // voxels fused from fewer views are unknown
    uint32_t carve;    // carve free space along pixels without a surface
    uint32_t free_space;  // update voxels in front of the truncation band as free (+1)
    uint32_t planar;      // planar median depth
    int32_t sh_active_degree;
    uint32_t slab_blocks;  // blocks fused at a time (0 = 65536)
    uint32_t cache_mb;     // keep the views' depth and colour on the device up to this (0 = never)
    uint32_t expected;     // fuse the expected depth D / (1 - T) where 1 - T > 0.5 instead (planar ignored)
};

// Render the current scene's median depth from each of `num_views` cameras,
// fuse it into a TSDF on the device and extract the surface (see tsdf.cuh).
// The outputs are counts; visloc_gsplat_cuda_tsdf_take copies the surface.
GSC_EXPORT int visloc_gsplat_cuda_tsdf_mesh(void* handle, const GscTsdfParams* p, int num_views,
                                            const GscCamera* cams, uint32_t* num_vertices, uint32_t* num_edges,
                                            uint32_t* num_blocks, uint32_t* num_slabs) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!p || num_views <= 0 || !cams || !(p->voxel > 0.f) || !(p->trunc > 0.f)) {
        ctx->error = "tsdf_mesh: settings, views and a positive voxel and truncation are required";
        return -1;
    }
    ctx->tsdf_cells.clear();
    ctx->tsdf_edges.clear();
    ctx->tsdf_pos.clear();
    ctx->tsdf_col_host.clear();
    const float bg[3] = {0.f, 0.f, 0.f};
    const TsdfArgs args = {p->voxel, p->trunc, p->max_depth, p->min_weight, int(p->carve != 0),
                           int(p->free_space != 0)};
    auto render = [&](int v) -> int {
        GSC_TRY(set_camera(ctx, &cams[v], bg, p->sh_active_degree));
        float* d;
        GSC_TRY(ensure(ctx, ctx->depth_out, ctx->pixels() * 2, &d));
        if (p->expected) {
            float* acc;
            GSC_TRY(ensure(ctx, ctx->acc_depth, ctx->pixels(), &acc));
            GSC_TRY(run_forward(ctx, d, false, acc));
            expected_depth_kernel<<<blocks_for(ctx->pixels()), 256>>>(uint32_t(ctx->pixels()), acc, d);
            GSC_CHECK(ctx, cudaGetLastError());
        } else {
            GSC_TRY(run_forward(ctx, d, p->planar != 0));
        }
        return 0;
    };
    uint32_t* counts;
    GSC_TRY(ensure(ctx, ctx->tsdf_counts, 8, &counts));
    uint32_t host_counts[8] = {};
    // The views' depth and colour, kept from the allocation pass when they
    // fit (offsets in pixels).
    std::vector<size_t> view_px(num_views + 1, 0);
    for (int v = 0; v < num_views; ++v) view_px[v + 1] = view_px[v] + size_t(cams[v].width) * cams[v].height;
    const size_t cache_limit = size_t(p->cache_mb) << 20;
    const bool cache = view_px[num_views] * 16 <= cache_limit;
    float *cache_depth = nullptr, *cache_rgb = nullptr;
    if (cache) {
        GSC_TRY(ensure(ctx, ctx->tsdf_cache_depth, view_px[num_views], &cache_depth));
        GSC_TRY(ensure(ctx, ctx->tsdf_cache_rgb, view_px[num_views] * 3, &cache_rgb));
    }

    // 1. Allocation: the set of blocks over every view's truncation band.
    uint32_t cap = 1u << 20;
    unsigned long long* table;
    auto fresh_table = [&](uint32_t c) -> int {
        GSC_TRY(ensure(ctx, ctx->tsdf_table, c, &table));
        GSC_CHECK(ctx, cudaMemset(table, 0xFF, size_t(c) * sizeof(unsigned long long)));
        return 0;
    };
    GSC_TRY(fresh_table(cap));
    GSC_CHECK(ctx, cudaMemset(counts, 0, 8 * sizeof(uint32_t)));
    for (int v = 0; v < num_views; ++v) {
        GSC_TRY(render(v));
        if (cache) {
            const size_t npx = view_px[v + 1] - view_px[v];
            GSC_CHECK(ctx, cudaMemcpy2D(cache_depth + view_px[v], sizeof(float), ctx->depth_out.ptr,
                                        2 * sizeof(float), sizeof(float), npx, cudaMemcpyDeviceToDevice));
            GSC_CHECK(ctx, cudaMemcpy(cache_rgb + view_px[v] * 3, ctx->image.ptr, npx * 3 * sizeof(float),
                                      cudaMemcpyDeviceToDevice));
        }
        const uint32_t half_px = ((cams[v].width + 1) / 2) * ((cams[v].height + 1) / 2);
        while (true) {
            tsdf_alloc_kernel<<<blocks_for(half_px), 256>>>(cams[v], args,
                                                            static_cast<const float*>(ctx->depth_out.ptr), table,
                                                            cap - 1, counts);
            GSC_CHECK(ctx, cudaGetLastError());
            GSC_CHECK(ctx, cudaMemcpy(host_counts, counts, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            // Keep the set at most half full: grow (reinserting the keys) and
            // redo this view if it filled up or passed half.
            if (!host_counts[1] && host_counts[0] <= cap / 2) break;
            const uint32_t old_cap = cap;
            unsigned long long* keys;
            GSC_TRY(ensure(ctx, ctx->tsdf_keys, old_cap, &keys));
            GSC_CHECK(ctx, cudaMemset(counts + 2, 0, sizeof(uint32_t)));
            tsdf_collect_kernel<<<blocks_for(old_cap), 256>>>(table, old_cap, keys, counts);
            GSC_CHECK(ctx, cudaGetLastError());
            uint32_t n_keys = 0;
            GSC_CHECK(ctx, cudaMemcpy(&n_keys, counts + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            cap = next_pow2(uint64_t(std::max(host_counts[0], n_keys)) * 4);
            free_buf(ctx->tsdf_table);
            GSC_TRY(fresh_table(cap));
            GSC_CHECK(ctx, cudaMemset(counts, 0, 2 * sizeof(uint32_t)));
            if (n_keys > 0) {
                tsdf_map_kernel<<<blocks_for(n_keys), 256>>>(keys, n_keys, table, nullptr, cap - 1);
                GSC_CHECK(ctx, cudaGetLastError());
            }
            GSC_CHECK(ctx, cudaMemcpy(counts, &n_keys, sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }
    // All keys, sorted: by x, then y, then z (the key's bit order).
    uint32_t n_blocks = 0;
    std::vector<unsigned long long> all;
    {
        unsigned long long* keys;
        GSC_TRY(ensure(ctx, ctx->tsdf_keys, std::max(host_counts[0], 1u), &keys));
        GSC_CHECK(ctx, cudaMemset(counts + 2, 0, sizeof(uint32_t)));
        tsdf_collect_kernel<<<blocks_for(cap), 256>>>(table, cap, keys, counts);
        GSC_CHECK(ctx, cudaGetLastError());
        GSC_CHECK(ctx, cudaMemcpy(&n_blocks, counts + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        all.resize(n_blocks);
        if (n_blocks > 0)
            GSC_CHECK(ctx, cudaMemcpy(all.data(), keys, n_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        std::sort(all.begin(), all.end());
    }
    free_buf(ctx->tsdf_table);

    // 2. Slabs: consecutive x layers of at most slab_blocks blocks (a layer
    // larger than that is a slab of its own), plus the next layer up.
    const uint32_t budget = p->slab_blocks > 0 ? p->slab_blocks : 65536u;
    std::vector<size_t> layer_start;  // index of each x layer's first key
    for (size_t i = 0; i < all.size(); ++i)
        if (i == 0 || tsdf_key_x(all[i]) != tsdf_key_x(all[i - 1])) layer_start.push_back(i);
    layer_start.push_back(all.size());
    const size_t n_layers = layer_start.size() - 1;
    uint32_t slabs = 0;
    for (size_t l0 = 0; l0 < n_layers;) {
        size_t l1 = l0 + 1;  // layers l0..l1 are the slab's own
        while (l1 < n_layers && layer_start[l1 + 1] - layer_start[l0] <= budget) ++l1;
        const size_t own_begin = layer_start[l0], own_end = layer_start[l1];
        size_t end = own_end;
        // The next layer up, when it is adjacent, holds the top face's corners.
        if (l1 < n_layers && tsdf_key_x(all[layer_start[l1]]) == tsdf_key_x(all[own_end - 1]) + 1)
            end = layer_start[l1 + 1];
        const uint32_t n_own = uint32_t(own_end - own_begin), n_slab = uint32_t(end - own_begin);
        ++slabs;

        unsigned long long *slab, *map_keys;
        uint32_t* map_vals;
        GSC_TRY(ensure(ctx, ctx->tsdf_keys, n_slab, &slab));
        GSC_CHECK(ctx, cudaMemcpy(slab, all.data() + own_begin, n_slab * sizeof(unsigned long long),
                                  cudaMemcpyHostToDevice));
        const uint32_t map_cap = next_pow2(uint64_t(n_slab) * 2);
        GSC_TRY(ensure(ctx, ctx->tsdf_slab_table, map_cap, &map_keys));
        GSC_TRY(ensure(ctx, ctx->tsdf_slab_keys, map_cap, &map_vals));
        GSC_CHECK(ctx, cudaMemset(map_keys, 0xFF, size_t(map_cap) * sizeof(unsigned long long)));
        tsdf_map_kernel<<<blocks_for(n_slab), 256>>>(slab, n_slab, map_keys, map_vals, map_cap - 1);
        GSC_CHECK(ctx, cudaGetLastError());
        const size_t n_vox = size_t(n_slab) * kTsdfBV;
        float *sdf, *w, *col;
        GSC_TRY(ensure(ctx, ctx->tsdf_sdf, n_vox, &sdf));
        GSC_TRY(ensure(ctx, ctx->tsdf_w, n_vox, &w));
        GSC_TRY(ensure(ctx, ctx->tsdf_col, n_vox * 3, &col));
        tsdf_reset_kernel<<<blocks_for(n_vox), 256>>>(n_vox, sdf, w, col);
        GSC_CHECK(ctx, cudaGetLastError());

        // 3. Fuse every view into the slab.
        for (int v = 0; v < num_views; ++v) {
            const float *depth = cache_depth + view_px[v], *rgb = cache_rgb + view_px[v] * 3;
            int stride = 1;
            if (!cache) {
                GSC_TRY(render(v));
                depth = static_cast<const float*>(ctx->depth_out.ptr);
                rgb = static_cast<const float*>(ctx->image.ptr);
                stride = 2;
            }
            float4 pl[5];
            frustum_planes(cams[v], pl);
            tsdf_integrate_kernel<<<n_slab, kBlock>>>(cams[v], args, pl[0], pl[1], pl[2], pl[3], pl[4], slab, depth,
                                                      stride, rgb, sdf, w, col);
            GSC_CHECK(ctx, cudaGetLastError());
        }

        // 4. Extract the own blocks' cells; retry with room for every record.
        const TsdfSlab s = {map_keys, map_vals, map_cap - 1, sdf, w, col, p->min_weight};
        uint32_t cap_v = std::max(n_own * 64u, 1024u), cap_e = cap_v * 2;
        while (true) {
            int32_t *o_cells, *o_edges;
            float* o_pos;
            uint8_t* o_col;
            GSC_TRY(ensure(ctx, ctx->tsdf_out_cells, size_t(cap_v) * 3, &o_cells));
            GSC_TRY(ensure(ctx, ctx->tsdf_out_pos, size_t(cap_v) * 3, &o_pos));
            GSC_TRY(ensure(ctx, ctx->tsdf_out_col, size_t(cap_v) * 3, &o_col));
            GSC_TRY(ensure(ctx, ctx->tsdf_out_edges, size_t(cap_e) * 4, &o_edges));
            GSC_CHECK(ctx, cudaMemset(counts + 3, 0, 2 * sizeof(uint32_t)));
            if (n_own > 0) {
                tsdf_extract_kernel<<<n_own, kBlock>>>(s, slab, p->voxel, o_cells, o_pos, o_col, o_edges, cap_v,
                                                       cap_e, counts);
                GSC_CHECK(ctx, cudaGetLastError());
            }
            GSC_CHECK(ctx, cudaMemcpy(host_counts + 3, counts + 3, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            const uint32_t nv = host_counts[3], ne = host_counts[4];
            if (nv > cap_v || ne > cap_e) {
                cap_v = std::max(cap_v, nv);
                cap_e = std::max(cap_e, ne);
                continue;
            }
            const size_t v0 = ctx->tsdf_pos.size() / 3, e0 = ctx->tsdf_edges.size() / 4;
            ctx->tsdf_cells.resize((v0 + nv) * 3);
            ctx->tsdf_pos.resize((v0 + nv) * 3);
            ctx->tsdf_col_host.resize((v0 + nv) * 3);
            ctx->tsdf_edges.resize((e0 + ne) * 4);
            if (nv > 0) {
                GSC_CHECK(ctx, cudaMemcpy(ctx->tsdf_cells.data() + v0 * 3, o_cells, size_t(nv) * 3 * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost));
                GSC_CHECK(ctx, cudaMemcpy(ctx->tsdf_pos.data() + v0 * 3, o_pos, size_t(nv) * 3 * sizeof(float),
                                          cudaMemcpyDeviceToHost));
                GSC_CHECK(ctx, cudaMemcpy(ctx->tsdf_col_host.data() + v0 * 3, o_col, size_t(nv) * 3,
                                          cudaMemcpyDeviceToHost));
            }
            if (ne > 0)
                GSC_CHECK(ctx, cudaMemcpy(ctx->tsdf_edges.data() + e0 * 4, o_edges, size_t(ne) * 4 * sizeof(int32_t),
                                          cudaMemcpyDeviceToHost));
            break;
        }
        l0 = l1;
    }
    // The voxels are the large part; the next fusion allocates them again.
    for (Buf* b : {&ctx->tsdf_sdf, &ctx->tsdf_w, &ctx->tsdf_col, &ctx->tsdf_out_cells, &ctx->tsdf_out_pos,
                   &ctx->tsdf_out_col, &ctx->tsdf_out_edges, &ctx->tsdf_keys, &ctx->tsdf_slab_table,
                   &ctx->tsdf_slab_keys, &ctx->tsdf_cache_depth, &ctx->tsdf_cache_rgb})
        free_buf(*b);
    if (num_vertices) *num_vertices = uint32_t(ctx->tsdf_pos.size() / 3);
    if (num_edges) *num_edges = uint32_t(ctx->tsdf_edges.size() / 4);
    if (num_blocks) *num_blocks = n_blocks;
    if (num_slabs) *num_slabs = slabs;
    ctx->have_forward = false;
    return 0;
}

// Copy (all nullable) and release the last tsdf_mesh's surface: per vertex
// its cell (3 int32, the cell's origin voxel), position (3 floats) and
// colour (3 bytes); per edge with a sign change the origin voxel and
// `axis | 4 if the origin is outside` (4 int32).
GSC_EXPORT int visloc_gsplat_cuda_tsdf_take(void* handle, int32_t* cells, float* positions, uint8_t* colors,
                                            int32_t* edges) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (cells) std::copy(ctx->tsdf_cells.begin(), ctx->tsdf_cells.end(), cells);
    if (positions) std::copy(ctx->tsdf_pos.begin(), ctx->tsdf_pos.end(), positions);
    if (colors) std::copy(ctx->tsdf_col_host.begin(), ctx->tsdf_col_host.end(), colors);
    if (edges) std::copy(ctx->tsdf_edges.begin(), ctx->tsdf_edges.end(), edges);
    std::vector<int32_t>().swap(ctx->tsdf_cells);
    std::vector<int32_t>().swap(ctx->tsdf_edges);
    std::vector<float>().swap(ctx->tsdf_pos);
    std::vector<uint8_t>().swap(ctx->tsdf_col_host);
    return 0;
}

}  // extern "C"
