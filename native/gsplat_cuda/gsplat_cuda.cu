// Native CUDA 3D Gaussian Splatting rasterizer for visloc-gsplat (opt-in).
//
// Built explicitly with `scripts/build_gsplat_cuda.sh`, never by Cargo, so
// ordinary builds need no CUDA toolkit; `crates/gsplat-cuda` loads the shared
// library at runtime and checks `visloc_gsplat_cuda_abi_version`.
//
// The maths mirrors the wgpu renderer (crates/gsplat-render/src/shaders:
// compute_projected, tile_row_span, rasterize, rasterize_backward,
// project_backward), so both backends and the CPU f64 oracle
// (`visloc_gsplat_core::backward`) agree:
//   alpha = min(0.99, opacity exp(-sigma)), blended when alpha >= 1/255 and
//   while T > 1e-4; the EWA Jacobian is linearised inside 1.3x the frustum;
//   screen covariance + 0.3 I; colour = max(0, SH(dir) + 0.5).
//
// Unlike the wgpu renderer, per-gaussian state is indexed by gaussian id (no
// visible-set compaction), and tile-then-depth order comes from one 64-bit
// radix sort of (tile << 32 | depth bits), stable in gaussian order (the
// oracle's tie order).
//
// Gaussian layout (visloc_gsplat_render::packing::PackedScene):
//   transforms[i * 10 + ..] = mean xyz, quaternion wxyz (un-normalised),
//                             log-scale xyz
//   opacity[i]              = opacity logit
//   sh[i * 3 * cpc2 + ..]   = 3 DC values, then channel-major rest
// Gradients come back in the same layouts.

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <cstdio>
#include <new>
#include <string>

extern "C" {

// World-to-camera rotation (row-major) and translation, camera centre in
// world coordinates, pinhole intrinsics and image size. Mirrored by
// `GscCamera` in crates/gsplat-cuda/src/lib.rs.
struct GscCamera {
    float rot[9];
    float t[3];
    float center[3];
    float fx, fy, cx, cy;
    uint32_t width, height;
};

}  // extern "C"

namespace {

constexpr uint32_t kAbiVersion = 1;
constexpr int kTile = 16;
constexpr int kBlock = kTile * kTile;
constexpr int kBwdBatch = 128;
// du dv dA dB dC dopacity dr dg db, and the refine weight (brush / AbsGS:
// sum over pixels of |dL/du| W + |dL/dv| H).
constexpr int kNumGrad = 10;
constexpr int kNumStages = 8;
constexpr const char* kStageNames =
    "project,scan,duplicate,sort,ranges,raster,raster_bwd,project_bwd";

constexpr float kC0 = 0.2820948f;

struct Projected {
    bool ok;
    float u, v;
    float c00, c01, c11;  // conic: inverse 2D covariance
    float opacity;
    float extent_k;  // footprint is d^T conic d <= extent_k
    float depth;
    int tx0, ty0, tx1, ty1;  // inclusive tile rectangle, clamped to the image
};

// Row-major rotation of a quaternion (w, x, y, z), normalised first; the
// identity when it is (near) zero. Same as rot_from_quat in common.wgsl.
__device__ inline void quat_to_rot(float qw, float qx, float qy, float qz, float r[9]) {
    float n = qw * qw + qx * qx + qy * qy + qz * qz;
    float w = 1.f, x = 0.f, y = 0.f, z = 0.f;
    if (n > 1e-12f) {
        float inv = rsqrtf(n);
        w = qw * inv;
        x = qx * inv;
        y = qy * inv;
        z = qz * inv;
    }
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    r[0] = 1.f - 2.f * (yy + zz);
    r[1] = 2.f * (xy - wz);
    r[2] = 2.f * (xz + wy);
    r[3] = 2.f * (xy + wz);
    r[4] = 1.f - 2.f * (xx + zz);
    r[5] = 2.f * (yz - wx);
    r[6] = 2.f * (xz - wy);
    r[7] = 2.f * (yz + wx);
    r[8] = 1.f - 2.f * (xx + yy);
}

__device__ inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }

__device__ Projected compute_projected(const GscCamera& c, const float* tr, float raw_opac) {
    Projected p = {};
    p.ok = false;
    const float opacity = 1.f / (1.f + expf(-raw_opac));
    if (opacity < 1e-4f) return p;
    const float* W = c.rot;
    const float mx = tr[0], my = tr[1], mz = tr[2];
    const float px = W[0] * mx + W[1] * my + W[2] * mz + c.t[0];
    const float py = W[3] * mx + W[4] * my + W[5] * mz + c.t[1];
    const float pz = W[6] * mx + W[7] * my + W[8] * mz + c.t[2];
    if (pz <= 0.1f) return p;

    float R[9];
    quat_to_rot(tr[3], tr[4], tr[5], tr[6], R);
    const float sx = expf(tr[7]), sy = expf(tr[8]), sz = expf(tr[9]);
    const float s2[3] = {sx * sx, sy * sy, sz * sz};
    float rw[9];  // W R
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            rw[i * 3 + j] = W[i * 3] * R[j] + W[i * 3 + 1] * R[3 + j] + W[i * 3 + 2] * R[6 + j];
    // cov_cam = (W R) diag(s^2) (W R)^T
    float cov[6];  // 00 01 02 11 12 22
    {
        int n = 0;
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = i; j < 3; ++j)
                cov[n++] = rw[i * 3] * s2[0] * rw[j * 3] + rw[i * 3 + 1] * s2[1] * rw[j * 3 + 1] +
                           rw[i * 3 + 2] * s2[2] * rw[j * 3 + 2];
    }
    const float cov00 = cov[0], cov01 = cov[1], cov02 = cov[2];
    const float cov11 = cov[3], cov12 = cov[4], cov22 = cov[5];

    const float inv_z = 1.f / pz;
    const float inv_z2 = inv_z * inv_z;
    // Linearise at a point clamped to 1.3x the frustum (Inria rasterizer).
    const float lim_x = 1.3f * 0.5f * float(c.width) / c.fx;
    const float lim_y = 1.3f * 0.5f * float(c.height) / c.fy;
    const float tx = clampf(px * inv_z, -lim_x, lim_x) * pz;
    const float ty = clampf(py * inv_z, -lim_y, lim_y) * pz;
    const float j00 = c.fx * inv_z;
    const float j02 = -c.fx * tx * inv_z2;
    const float j11 = c.fy * inv_z;
    const float j12 = -c.fy * ty * inv_z2;
    const float a = j00 * j00 * cov00 + 2.f * j00 * j02 * cov02 + j02 * j02 * cov22 + 0.3f;
    const float b = j00 * j11 * cov01 + j00 * j12 * cov02 + j02 * j11 * cov12 + j02 * j12 * cov22;
    const float cc = j11 * j11 * cov11 + 2.f * j11 * j12 * cov12 + j12 * j12 * cov22 + 0.3f;
    const float det = a * cc - b * b;
    if (!(det > 0.f)) return p;
    const float mid = 0.5f * (a + cc);
    const float lambda1 = mid + sqrtf(fmaxf(mid * mid - det, 0.f));
    if (lambda1 <= 0.f) return p;
    // Only d^T C^-1 d <= k = 2 ln(255 opacity) can reach alpha >= 1/255.
    const float k = 2.f * logf(255.f * opacity);
    if (!(k > 0.f)) return p;
    const float ext_x = sqrtf(k * a);
    const float ext_y = sqrtf(k * cc);
    if (!(ext_x == ext_x) || !(ext_y == ext_y)) return p;

    const float u = px * j00 + c.cx;
    const float v = py * j11 + c.cy;
    const float w = float(c.width), h = float(c.height);
    if (u + ext_x < 0.f || v + ext_y < 0.f || u - ext_x > w - 1.f || v - ext_y > h - 1.f) return p;
    const float x0f = fmaxf(u - ext_x, 0.f);
    const float y0f = fmaxf(v - ext_y, 0.f);
    const float x1f = fminf(u + ext_x, w - 1.f);
    const float y1f = fminf(v + ext_y, h - 1.f);
    if (x1f < x0f || y1f < y0f) return p;

    const float inv_det = 1.f / det;
    p.ok = true;
    p.u = u;
    p.v = v;
    p.c00 = cc * inv_det;
    p.c01 = -b * inv_det;
    p.c11 = a * inv_det;
    p.opacity = opacity;
    p.extent_k = k;
    p.depth = pz;
    p.tx0 = int(floorf(x0f)) / kTile;
    p.ty0 = int(floorf(y0f)) / kTile;
    p.tx1 = int(floorf(x1f)) / kTile;
    p.ty1 = int(floorf(y1f)) / kTile;
    return p;
}

// Tiles of row `ty` holding a pixel the rasterizer could blend, as an
// inclusive column range (empty when x > y). See tile_row_span in
// common.wgsl for the derivation.
__device__ int2 tile_row_span(const Projected& p, int ty, uint32_t img_w, uint32_t img_h) {
    const int2 empty = make_int2(1, 0);
    const float det = p.c00 * p.c11 - p.c01 * p.c01;
    if (!(det > 0.f) || !(p.c00 > 0.f) || !(p.c11 > 0.f)) return empty;
    const float k = p.extent_k * 1.001f + 1e-3f;
    const float y0 = float(ty * kTile) + 0.5f - p.v;
    const float y1 = float(min(ty * kTile + kTile - 1, int(img_h) - 1)) + 0.5f - p.v;
    const float x_max = sqrtf(k * p.c11 / det);
    const float yr = clampf(-p.c01 * x_max / p.c11, y0, y1);
    const float yl = clampf(p.c01 * x_max / p.c11, y0, y1);
    const float dr = k * p.c00 - yr * yr * det;
    if (dr < 0.f) return empty;
    const float dl = fmaxf(k * p.c00 - yl * yl * det, 0.f);
    const float xr = (-p.c01 * yr + sqrtf(dr)) / p.c00;
    const float xl = (-p.c01 * yl - sqrtf(dl)) / p.c00;
    const float px_lo = fmaxf(ceilf(p.u + xl - 0.5f), float(p.tx0 * kTile));
    const float px_hi = fminf(floorf(p.u + xr - 0.5f), float(min(p.tx1 * kTile + kTile - 1, int(img_w) - 1)));
    if (px_hi < px_lo) return empty;
    return make_int2(int(px_lo) / kTile, int(px_hi) / kTile);
}

// Real SH basis b1..b15 (b0 = kC0) for unit direction d, Inria / brush
// normalisation.
__device__ inline void sh_basis(float x, float y, float z, float b[16]) {
    const float xx = x * x, yy = y * y, zz = z * z;
    b[0] = kC0;
    b[1] = -0.4886025f * y;
    b[2] = 0.4886025f * z;
    b[3] = -0.4886025f * x;
    b[4] = 1.0925485f * x * y;
    b[5] = -1.0925485f * y * z;
    b[6] = 0.3153916f * (2.f * zz - xx - yy);
    b[7] = -1.0925485f * x * z;
    b[8] = 0.5462742f * (xx - yy);
    b[9] = -0.5900436f * y * (3.f * xx - yy);
    b[10] = 2.8906114f * x * y * z;
    b[11] = -0.4570458f * y * (4.f * zz - xx - yy);
    b[12] = 0.3731763f * z * (2.f * zz - 3.f * xx - 3.f * yy);
    b[13] = -0.4570458f * x * (4.f * zz - xx - yy);
    b[14] = 1.4453057f * z * (xx - yy);
    b[15] = -0.5900436f * x * (xx - 3.f * yy);
}

// Unit view direction from the camera centre to the mean ((0, 0, 1) when
// they coincide) and the distance.
__device__ inline float view_dir(const GscCamera& c, const float* tr, float d[3]) {
    const float rx = tr[0] - c.center[0], ry = tr[1] - c.center[1], rz = tr[2] - c.center[2];
    const float dn = sqrtf(rx * rx + ry * ry + rz * rz);
    d[0] = 0.f;
    d[1] = 0.f;
    d[2] = 1.f;
    if (dn > 1e-6f) {
        d[0] = rx / dn;
        d[1] = ry / dn;
        d[2] = rz / dn;
    }
    return dn;
}

__global__ void project_kernel(int n, GscCamera cam, int rest_pc, int act, const float* __restrict__ transforms,
                               const float* __restrict__ opacity, const float* __restrict__ sh,
                               float2* __restrict__ xy, float4* __restrict__ conic_op, float4* __restrict__ color,
                               float* __restrict__ depth, int4* __restrict__ rect, float* __restrict__ extk,
                               uint32_t* __restrict__ tiles) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    tiles[i] = 0;
    const float* tr = transforms + size_t(i) * 10;
    const Projected p = compute_projected(cam, tr, opacity[i]);
    if (!p.ok) return;
    uint32_t count = 0;
    for (int ty = p.ty0; ty <= p.ty1; ++ty) {
        const int2 span = tile_row_span(p, ty, cam.width, cam.height);
        if (span.y >= span.x) count += uint32_t(span.y - span.x + 1);
    }
    tiles[i] = count;
    xy[i] = make_float2(p.u, p.v);
    conic_op[i] = make_float4(p.c00, p.c01, p.c11, p.opacity);
    depth[i] = p.depth;
    rect[i] = make_int4(p.tx0, p.ty0, p.tx1, p.ty1);
    extk[i] = p.extent_k;

    float d[3];
    view_dir(cam, tr, d);
    float b[16];
    sh_basis(d[0], d[1], d[2], b);
    const float* row = sh + size_t(i) * 3 * (rest_pc + 1);
    float out[3];
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        float acc = kC0 * row[ch];
        const float* rest = row + 3 + ch * rest_pc;
#pragma unroll
        for (int k = 1; k < 16; ++k)
            if (k <= act) acc += b[k] * rest[k - 1];
        out[ch] = acc + 0.5f;
    }
    color[i] = make_float4(out[0], out[1], out[2], 0.f);
}

__global__ void duplicate_kernel(int n, GscCamera cam, uint32_t tile_bw, const float2* __restrict__ xy,
                                 const float4* __restrict__ conic_op, const float* __restrict__ depth,
                                 const int4* __restrict__ rect, const float* __restrict__ extk,
                                 const uint32_t* __restrict__ tiles, const uint32_t* __restrict__ offsets,
                                 unsigned long long* __restrict__ keys, uint32_t* __restrict__ vals) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t count = tiles[i];
    if (count == 0) return;
    Projected p;
    p.ok = true;
    p.u = xy[i].x;
    p.v = xy[i].y;
    const float4 co = conic_op[i];
    p.c00 = co.x;
    p.c01 = co.y;
    p.c11 = co.z;
    p.opacity = co.w;
    p.extent_k = extk[i];
    const int4 r = rect[i];
    p.tx0 = r.x;
    p.ty0 = r.y;
    p.tx1 = r.z;
    p.ty1 = r.w;
    const unsigned long long dbits = __float_as_uint(depth[i]);
    uint32_t out = offsets[i] - count;
    const uint32_t end = offsets[i];
    for (int ty = p.ty0; ty <= p.ty1; ++ty) {
        const int2 span = tile_row_span(p, ty, cam.width, cam.height);
        for (int tx = span.x; tx <= span.y && out < end; ++tx) {
            const unsigned long long tile = uint32_t(ty) * tile_bw + uint32_t(tx);
            keys[out] = (tile << 32) | dbits;
            vals[out] = uint32_t(i);
            ++out;
        }
    }
}

__global__ void ranges_kernel(uint32_t m, const unsigned long long* __restrict__ keys, uint2* __restrict__ ranges) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const uint32_t tile = uint32_t(keys[i] >> 32);
    if (i == 0 || uint32_t(keys[i - 1] >> 32) != tile) ranges[tile].x = i;
    if (i + 1 == m || uint32_t(keys[i + 1] >> 32) != tile) ranges[tile].y = i + 1;
}

// One block per 16x16 tile, one thread per pixel, splats staged in shared
// memory 256 at a time; the block stops once every pixel is saturated.
__global__ void __launch_bounds__(kBlock) raster_fwd_kernel(
    uint32_t img_w, uint32_t img_h, float3 bg, const uint2* __restrict__ ranges, const uint32_t* __restrict__ vals,
    const float2* __restrict__ xy, const float4* __restrict__ conic_op, const float4* __restrict__ color,
    float* __restrict__ out_img, float* __restrict__ final_t, uint32_t* __restrict__ last_idx) {
    __shared__ float2 s_xy[kBlock];
    __shared__ float4 s_co[kBlock];
    __shared__ float3 s_col[kBlock];
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const uint32_t px = blockIdx.x * kTile + tid % kTile;
    const uint32_t py = blockIdx.y * kTile + tid / kTile;
    const bool in_image = px < img_w && py < img_h;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];

    float3 acc = make_float3(0.f, 0.f, 0.f);
    float trans = 1.f;
    uint32_t last = 0;
    bool done = !in_image;
    for (uint32_t start = range.x; start < range.y; start += kBlock) {
        // Also the barrier that protects the staged batch before it is reloaded.
        if (__syncthreads_count(done) == kBlock) break;
        const uint32_t cnt = min(uint32_t(kBlock), range.y - start);
        if (uint32_t(tid) < cnt) {
            const uint32_t g = vals[start + tid];
            s_xy[tid] = xy[g];
            s_co[tid] = conic_op[g];
            const float4 c = color[g];
            s_col[tid] = make_float3(c.x, c.y, c.z);
        }
        __syncthreads();
        if (!done) {
            for (uint32_t s = 0; s < cnt; ++s) {
                if (trans <= 1e-4f) break;
                const float dx = sx - s_xy[s].x;
                const float dy = sy - s_xy[s].y;
                const float4 co = s_co[s];
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma >= 0.f) {
                    const float a = fminf(0.99f, co.w * expf(-sigma));
                    if (a >= 1.f / 255.f) {
                        const float3 c = s_col[s];
                        const float w = trans * a;
                        acc.x += w * fmaxf(c.x, 0.f);
                        acc.y += w * fmaxf(c.y, 0.f);
                        acc.z += w * fmaxf(c.z, 0.f);
                        trans *= 1.f - a;
                        last = start + s + 1;
                    }
                }
            }
            done = trans <= 1e-4f;
        }
    }
    if (in_image) {
        const uint32_t pix = py * img_w + px;
        out_img[pix * 3 + 0] = acc.x + trans * bg.x;
        out_img[pix * 3 + 1] = acc.y + trans * bg.y;
        out_img[pix * 3 + 2] = acc.z + trans * bg.z;
        final_t[pix] = trans;
        last_idx[pix] = last;
    }
}

// One butterfly step: keep `lo` or `hi` by this lane's bit, add the partner's
// other half.
__device__ inline float xstep(float lo, float hi, bool h, int m) {
    return (h ? hi : lo) + __shfl_xor_sync(0xffffffffu, h ? lo : hi, m);
}

// Backward of raster_fwd_kernel for L = sum_px dot(d_image[px], C[px]):
// per (tile, splat) screen-space gradients, walking each tile's list back to
// front from its last blended entry. Each pixel recovers T before a splat
// from T after it (T_before = T_after / (1 - alpha)) and keeps the colour
// composited behind it. A splat's 256 per-pixel contributions are summed per
// warp with a transposed xor butterfly (lanes 2c end up holding component c),
// then in shared memory across warps, then added to the gaussian's record in
// screen_grads with one global float atomic per component.
__global__ void __launch_bounds__(kBlock) raster_bwd_kernel(
    uint32_t img_w, uint32_t img_h, float3 bg, const uint2* __restrict__ ranges, const uint32_t* __restrict__ vals,
    const float2* __restrict__ xy, const float4* __restrict__ conic_op, const float4* __restrict__ color,
    const float* __restrict__ final_t, const uint32_t* __restrict__ last_idx, const float* __restrict__ d_image,
    float* __restrict__ screen_grads) {
    __shared__ float2 s_xy[kBwdBatch];
    __shared__ float4 s_co[kBwdBatch];
    __shared__ float3 s_col[kBwdBatch];
    __shared__ uint32_t s_gid[kBwdBatch];
    __shared__ float s_gacc[kBwdBatch * kNumGrad];
    __shared__ uint32_t s_last;
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const uint32_t px = blockIdx.x * kTile + tid % kTile;
    const uint32_t py = blockIdx.y * kTile + tid / kTile;
    const bool in_image = px < img_w && py < img_h;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];
    const float fw = float(img_w), fh = float(img_h);

    float t_cur = 1.f;
    uint32_t last = 0;
    float3 g = make_float3(0.f, 0.f, 0.f);
    float3 behind = bg;
    if (in_image) {
        const uint32_t pix = py * img_w + px;
        t_cur = final_t[pix];
        last = last_idx[pix];
        g = make_float3(d_image[pix * 3], d_image[pix * 3 + 1], d_image[pix * 3 + 2]);
    }
    if (tid == 0) s_last = 0;
    __syncthreads();
    atomicMax(&s_last, last);
    __syncthreads();
    const uint32_t stop = min(s_last, range.y);

    for (uint32_t bend = stop; bend > range.x;) {
        const uint32_t bstart = bend - range.x > uint32_t(kBwdBatch) ? bend - kBwdBatch : range.x;
        const uint32_t cnt = bend - bstart;
        // The previous batch's flush must finish before its slots are reused.
        __syncthreads();
        if (uint32_t(tid) < cnt) {
            const uint32_t gi = vals[bstart + tid];
            s_xy[tid] = xy[gi];
            s_co[tid] = conic_op[gi];
            const float4 c = color[gi];
            s_col[tid] = make_float3(c.x, c.y, c.z);
            s_gid[tid] = gi;
        }
        for (uint32_t i = tid; i < cnt * kNumGrad; i += kBlock) s_gacc[i] = 0.f;
        __syncthreads();

        for (int s = int(cnt) - 1; s >= 0; --s) {
            const uint32_t j = bstart + uint32_t(s);
            float g_u = 0.f, g_v = 0.f, g_a = 0.f, g_b = 0.f, g_c = 0.f, g_o = 0.f, g_r = 0.f;
            float3 g_col = make_float3(0.f, 0.f, 0.f);
            bool hit = false;
            if (in_image && j < last) {
                const float4 co = s_co[s];
                const float dx = sx - s_xy[s].x;
                const float dy = sy - s_xy[s].y;
                // Same expressions as the forward, so the same splats pass.
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma >= 0.f) {
                    const float e = expf(-sigma);
                    const float raw = co.w * e;
                    const float a = fminf(0.99f, raw);
                    if (a >= 1.f / 255.f) {
                        hit = true;
                        const float t_before = t_cur / (1.f - a);
                        const float3 c = s_col[s];
                        const float3 col = make_float3(fmaxf(c.x, 0.f), fmaxf(c.y, 0.f), fmaxf(c.z, 0.f));
                        const float w = a * t_before;
                        g_col = make_float3(w * g.x, w * g.y, w * g.z);
                        const float d_alpha =
                            t_before * ((col.x - behind.x) * g.x + (col.y - behind.y) * g.y + (col.z - behind.z) * g.z);
                        behind.x = a * col.x + (1.f - a) * behind.x;
                        behind.y = a * col.y + (1.f - a) * behind.y;
                        behind.z = a * col.z + (1.f - a) * behind.z;
                        t_cur = t_before;
                        // A clamped alpha passes no gradient to opacity, conic or mean.
                        if (raw <= 0.99f) {
                            const float d_sigma = -d_alpha * a;
                            g_u = d_sigma * -(co.x * dx + co.y * dy);
                            g_v = d_sigma * -(co.y * dx + co.z * dy);
                            g_a = d_sigma * 0.5f * dx * dx;
                            g_b = d_sigma * dx * dy;
                            g_c = d_sigma * 0.5f * dy * dy;
                            g_o = d_alpha * e;
                            g_r = fabsf(g_u) * fw + fabsf(g_v) * fh;
                        }
                    }
                }
            }
            // Skip the reduction when no lane of this warp blended the splat.
            if (__any_sync(0xffffffffu, hit)) {
                const bool h4 = (lane & 16) != 0;
                const float w0 = xstep(g_u, g_col.z, h4, 16);
                const float w1 = xstep(g_v, g_r, h4, 16);
                const float w2 = xstep(g_a, 0.f, h4, 16);
                const float w3 = xstep(g_b, 0.f, h4, 16);
                const float w4 = xstep(g_c, 0.f, h4, 16);
                const float w5 = xstep(g_o, 0.f, h4, 16);
                const float w6 = xstep(g_col.x, 0.f, h4, 16);
                const float w7 = xstep(g_col.y, 0.f, h4, 16);
                const bool h3 = (lane & 8) != 0;
                const float x0 = xstep(w0, w4, h3, 8);
                const float x1 = xstep(w1, w5, h3, 8);
                const float x2 = xstep(w2, w6, h3, 8);
                const float x3 = xstep(w3, w7, h3, 8);
                const bool h2 = (lane & 4) != 0;
                const float y0 = xstep(x0, x2, h2, 4);
                const float y1 = xstep(x1, x3, h2, 4);
                float z = xstep(y0, y1, (lane & 2) != 0, 2);
                z += __shfl_xor_sync(0xffffffffu, z, 1);
                const int comp = lane >> 1;
                if ((lane & 1) == 0 && comp < kNumGrad && z != 0.f) atomicAdd(&s_gacc[s * kNumGrad + comp], z);
            }
        }
        __syncthreads();
        for (uint32_t i = tid; i < cnt * kNumGrad; i += kBlock) {
            const float v = s_gacc[i];
            if (v != 0.f) atomicAdd(&screen_grads[size_t(s_gid[i / kNumGrad]) * kNumGrad + i % kNumGrad], v);
        }
        bend = bstart;
    }
}

// Backward of the projection: chain each visible gaussian's screen-space
// gradient to its parameters (see project_backward.wgsl for the forward
// being differentiated). Matrices are row-major float[9].
__device__ inline void mat_mul(const float* a, const float* b, float* o) {
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) o[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
}
__device__ inline void mat_tmul(const float* a, const float* b, float* o) {  // a^T b
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) o[i * 3 + j] = a[i] * b[j] + a[3 + i] * b[3 + j] + a[6 + i] * b[6 + j];
}
__device__ inline void mat_mult(const float* a, const float* b, float* o) {  // a b^T
#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j)
            o[i * 3 + j] = a[i * 3] * b[j * 3] + a[i * 3 + 1] * b[j * 3 + 1] + a[i * 3 + 2] * b[j * 3 + 2];
}
__device__ inline void mat_vec(const float* a, const float* v, float* o) {
#pragma unroll
    for (int i = 0; i < 3; ++i) o[i] = a[i * 3] * v[0] + a[i * 3 + 1] * v[1] + a[i * 3 + 2] * v[2];
}
__device__ inline float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

__global__ void project_bwd_kernel(int n, GscCamera cam, int rest_pc, int act, const float* __restrict__ transforms,
                                   const float* __restrict__ opacity, const float* __restrict__ sh,
                                   const uint32_t* __restrict__ tiles, const float* __restrict__ screen_grads,
                                   float* __restrict__ g_transforms, float* __restrict__ g_opacity,
                                   float* __restrict__ g_sh) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || tiles[i] == 0) return;
    const float* sg = screen_grads + size_t(i) * kNumGrad;
    const float g_u = sg[0], g_v = sg[1], g_ca = sg[2], g_cb = sg[3], g_cc = sg[4], g_op = sg[5];
    const float g_col[3] = {sg[6], sg[7], sg[8]};
    const float* tr = transforms + size_t(i) * 10;
    const float m[3] = {tr[0], tr[1], tr[2]};
    const float q[4] = {tr[3], tr[4], tr[5], tr[6]};
    const float* W = cam.rot;

    float p[3];
    mat_vec(W, m, p);
    p[0] += cam.t[0];
    p[1] += cam.t[1];
    p[2] += cam.t[2];

    const float n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    float qn[4] = {1.f, 0.f, 0.f, 0.f};
    float inv_n = 0.f;
    if (n2 > 1e-12f) {
        inv_n = rsqrtf(n2);
        qn[0] = q[0] * inv_n;
        qn[1] = q[1] * inv_n;
        qn[2] = q[2] * inv_n;
        qn[3] = q[3] * inv_n;
    }
    float rg[9];
    quat_to_rot(qn[0], qn[1], qn[2], qn[3], rg);
    const float s[3] = {expf(tr[7]), expf(tr[8]), expf(tr[9])};
    float m3[9];  // column k of Rg scaled by s_k
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int k = 0; k < 3; ++k) m3[r * 3 + k] = rg[r * 3 + k] * s[k];
    float sigma[9], ws[9], sc[9];
    mat_mult(m3, m3, sigma);
    mat_mul(W, sigma, ws);
    mat_mult(ws, W, sc);

    // EWA Jacobian with the 1.3x frustum clamp.
    const float inv_z = 1.f / p[2];
    const float inv_z2 = inv_z * inv_z;
    const float lim_x = 1.3f * 0.5f * float(cam.width) / cam.fx;
    const float lim_y = 1.3f * 0.5f * float(cam.height) / cam.fy;
    const float rx = p[0] * inv_z;
    const float ry = p[1] * inv_z;
    const bool clamped_x = rx < -lim_x || rx > lim_x;
    const bool clamped_y = ry < -lim_y || ry > lim_y;
    const float tx = clampf(rx, -lim_x, lim_x) * p[2];
    const float ty = clampf(ry, -lim_y, lim_y) * p[2];
    const float j0[3] = {cam.fx * inv_z, 0.f, -cam.fx * tx * inv_z2};
    const float j1[3] = {0.f, cam.fy * inv_z, -cam.fy * ty * inv_z2};
    float sj0[3], sj1[3];
    mat_vec(sc, j0, sj0);
    mat_vec(sc, j1, sj1);
    const float ca = dot3(j0, sj0) + 0.3f;
    const float cb = dot3(j0, sj1);
    const float cc = dot3(j1, sj1) + 0.3f;
    const float det = ca * cc - cb * cb;
    const float qa = cc / det;
    const float qb = -cb / det;
    const float qc = ca / det;

    // Conic -> 2D covariance: dL/dM = -Q GQ Q with GQ = [gA gB/2; gB/2 gC].
    const float gq01 = 0.5f * g_cb;
    const float t00 = g_ca * qa + gq01 * qb;
    const float t01 = g_ca * qb + gq01 * qc;
    const float t10 = gq01 * qa + g_cc * qb;
    const float t11 = gq01 * qb + g_cc * qc;
    const float g_a = -(qa * t00 + qb * t10);
    const float g_b = 2.f * -(qa * t01 + qb * t11);
    const float g_c = -(qb * t01 + qc * t11);

    // 2D covariance -> Sc and J.
    float gsc[9];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int k = 0; k < 3; ++k)
            gsc[r * 3 + k] = j0[r] * j0[k] * g_a + j0[r] * j1[k] * g_b + j1[r] * j1[k] * g_c;
    float gj0[3], gj1[3];
#pragma unroll
    for (int r = 0; r < 3; ++r) {
        gj0[r] = 2.f * g_a * sj0[r] + g_b * sj1[r];
        gj1[r] = 2.f * g_c * sj1[r] + g_b * sj0[r];
    }

    // J and the screen mean -> camera-space point.
    float gp[3] = {0.f, 0.f, 0.f};
    gp[2] = gj0[0] * (-cam.fx * inv_z2) + gj1[1] * (-cam.fy * inv_z2);
    if (clamped_x) {
        gp[2] += gj0[2] * cam.fx * clampf(rx, -lim_x, lim_x) * inv_z2;
    } else {
        gp[0] += gj0[2] * (-cam.fx * inv_z2);
        gp[2] += gj0[2] * 2.f * cam.fx * p[0] * inv_z2 * inv_z;
    }
    if (clamped_y) {
        gp[2] += gj1[2] * cam.fy * clampf(ry, -lim_y, lim_y) * inv_z2;
    } else {
        gp[1] += gj1[2] * (-cam.fy * inv_z2);
        gp[2] += gj1[2] * 2.f * cam.fy * p[1] * inv_z2 * inv_z;
    }
    gp[0] += g_u * cam.fx * inv_z;
    gp[1] += g_v * cam.fy * inv_z;
    gp[2] -= (g_u * cam.fx * p[0] + g_v * cam.fy * p[1]) * inv_z2;

    // p = W m + t  ->  dL/dm = W^T gp.
    float gmean[3];
#pragma unroll
    for (int r = 0; r < 3; ++r) gmean[r] = W[r] * gp[0] + W[3 + r] * gp[1] + W[6 + r] * gp[2];

    // Sc = W Sigma W^T -> GSigma = W^T GSc W; Sigma = M3 M3^T -> GM3.
    float wtg[9], gsig[9];
    mat_tmul(W, gsc, wtg);
    mat_mul(wtg, W, gsig);
    float gsym[9];
#pragma unroll
    for (int r = 0; r < 3; ++r)
#pragma unroll
        for (int k = 0; k < 3; ++k) gsym[r * 3 + k] = gsig[r * 3 + k] + gsig[k * 3 + r];
    float gm3[9];
    mat_mul(gsym, m3, gm3);
    // M3 = Rg diag(s): column k of M3 is s_k * column k of Rg.
    float gls[3];
    float gr[9];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        gls[k] = (gm3[k] * rg[k] + gm3[3 + k] * rg[3 + k] + gm3[6 + k] * rg[6 + k]) * s[k];
#pragma unroll
        for (int r = 0; r < 3; ++r) gr[r * 3 + k] = gm3[r * 3 + k] * s[k];
    }
    // Rg(qn) -> normalised quaternion.
    const float qw = qn[0], qx = qn[1], qy = qn[2], qz = qn[3];
    const float r00 = gr[0], r01 = gr[1], r02 = gr[2];
    const float r10 = gr[3], r11 = gr[4], r12 = gr[5];
    const float r20 = gr[6], r21 = gr[7], r22 = gr[8];
    const float gqn[4] = {
        2.f * (-qz * r01 + qy * r02 + qz * r10 - qx * r12 - qy * r20 + qx * r21),
        2.f * (qy * r01 + qz * r02 + qy * r10 - 2.f * qx * r11 - qw * r12 + qz * r20 + qw * r21 - 2.f * qx * r22),
        2.f * (-2.f * qy * r00 + qx * r01 + qw * r02 + qx * r10 + qz * r12 - qw * r20 + qz * r21 - 2.f * qy * r22),
        2.f * (-2.f * qz * r00 - qw * r01 + qx * r02 + qw * r10 - 2.f * qz * r11 + qy * r12 + qx * r20 + qy * r21),
    };
    // qn = q / |q|  ->  gq = (gqn - qn (qn . gqn)) / |q|.
    const float qdot = qn[0] * gqn[0] + qn[1] * gqn[1] + qn[2] * gqn[2] + qn[3] * gqn[3];

    const float op = 1.f / (1.f + expf(-opacity[i]));
    g_opacity[i] = g_op * op * (1.f - op);

    // Colour -> SH coefficients, and -> mean through the view direction.
    float dir[3];
    const float dn = view_dir(cam, tr, dir);
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    float b[16];
    sh_basis(x, y, z, b);
    // d b_k / d dir, k = 1..15.
    const float c0 = 0.4886025f, c3 = 1.0925485f, c5 = 0.3153916f, c8 = 0.5900436f, c9 = 2.8906114f;
    const float c10 = 0.4570458f, c11 = 0.3731763f, c2b = 0.5462742f, c14 = 1.4453057f;
    const float db[16][3] = {
        {0.f, 0.f, 0.f},
        {0.f, -c0, 0.f},
        {0.f, 0.f, c0},
        {-c0, 0.f, 0.f},
        {c3 * y, c3 * x, 0.f},
        {0.f, -c3 * z, -c3 * y},
        {-2.f * c5 * x, -2.f * c5 * y, 4.f * c5 * z},
        {-c3 * z, 0.f, -c3 * x},
        {2.f * c2b * x, -2.f * c2b * y, 0.f},
        {-6.f * c8 * x * y, -c8 * (3.f * xx - 3.f * yy), 0.f},
        {c9 * y * z, c9 * x * z, c9 * x * y},
        {2.f * c10 * x * y, -c10 * (4.f * zz - xx - 3.f * yy), -8.f * c10 * y * z},
        {-6.f * c11 * x * z, -6.f * c11 * y * z, c11 * (6.f * zz - 3.f * xx - 3.f * yy)},
        {-c10 * (4.f * zz - 3.f * xx - yy), 2.f * c10 * x * y, -8.f * c10 * x * z},
        {2.f * c14 * x * z, -2.f * c14 * y * z, c14 * (xx - yy)},
        {-c8 * (3.f * xx - 3.f * yy), 6.f * c8 * x * y, 0.f},
    };
    const size_t row_len = size_t(3) * (rest_pc + 1);
    const float* row = sh + size_t(i) * row_len;
    float* grow = g_sh + size_t(i) * row_len;
    float gdir[3] = {0.f, 0.f, 0.f};
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        const float* rest = row + 3 + ch * rest_pc;
        float raw = kC0 * row[ch] + 0.5f;
        float dd[3] = {0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 1; k < 16; ++k) {
            if (k <= act) {
                const float r = rest[k - 1];
                raw += b[k] * r;
                dd[0] += db[k][0] * r;
                dd[1] += db[k][1] * r;
                dd[2] += db[k][2] * r;
            }
        }
        const float gc = raw < 0.f ? 0.f : g_col[ch];
        grow[ch] = gc * kC0;
        float* grest = grow + 3 + ch * rest_pc;
#pragma unroll
        for (int k = 1; k < 16; ++k)
            if (k <= act) grest[k - 1] = gc * b[k];
        gdir[0] += gc * dd[0];
        gdir[1] += gc * dd[1];
        gdir[2] += gc * dd[2];
    }
    if (dn > 1e-6f) {
        const float dg = dot3(dir, gdir);
#pragma unroll
        for (int r = 0; r < 3; ++r) gmean[r] += (gdir[r] - dir[r] * dg) / dn;
    }

    float* gt = g_transforms + size_t(i) * 10;
    gt[0] = gmean[0];
    gt[1] = gmean[1];
    gt[2] = gmean[2];
#pragma unroll
    for (int k = 0; k < 4; ++k) gt[3 + k] = (gqn[k] - qn[k] * qdot) * inv_n;
    gt[7] = gls[0];
    gt[8] = gls[1];
    gt[9] = gls[2];
}

// Device buffer that only grows.
struct Buf {
    void* ptr = nullptr;
    size_t cap = 0;
};

struct Ctx {
    std::string error;
    int n = 0;
    int sh_degree = 0;
    GscCamera cam = {};
    float3 bg = make_float3(0.f, 0.f, 0.f);
    int act = 0;  // evaluated rest coefficients per channel
    uint32_t tile_bw = 0, tile_bh = 0;
    uint32_t num_isect = 0;
    bool have_forward = false;
    // Scene.
    Buf transforms, opacity, sh;
    // Per gaussian, per frame.
    Buf xy, conic_op, color, depth, rect, extk, tiles, offsets;
    // Intersections.
    Buf keys, vals, keys_sorted, vals_sorted, sort_temp, scan_temp;
    // Per tile / pixel.
    Buf ranges, image, final_t, last_idx;
    // Backward.
    Buf d_image, screen_grads, g_transforms, g_opacity, g_sh;
    cudaEvent_t ev[kNumStages + 2] = {};
    float ms[kNumStages] = {};

    int rest_pc() const { return (sh_degree + 1) * (sh_degree + 1) - 1; }
    size_t row_len() const { return size_t(3) * (sh_degree + 1) * (sh_degree + 1); }
    size_t pixels() const { return size_t(cam.width) * cam.height; }
};

#define GSC_CHECK(ctx, expr)                                                               \
    do {                                                                                   \
        cudaError_t err__ = (expr);                                                        \
        if (err__ != cudaSuccess) {                                                        \
            (ctx)->error = std::string(#expr) + ": " + cudaGetErrorString(err__);          \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

template <class T>
int ensure(Ctx* ctx, Buf& b, size_t count, T** out) {
    const size_t bytes = count * sizeof(T) > 0 ? count * sizeof(T) : sizeof(T);
    if (bytes > b.cap) {
        if (b.ptr) GSC_CHECK(ctx, cudaFree(b.ptr));
        b.ptr = nullptr;
        b.cap = 0;
        // 25% headroom so a slowly growing scene does not reallocate every frame.
        const size_t want = bytes + bytes / 4;
        GSC_CHECK(ctx, cudaMalloc(&b.ptr, want));
        b.cap = want;
    }
    *out = static_cast<T*>(b.ptr);
    return 0;
}

#define GSC_TRY(expr)              \
    do {                           \
        if ((expr) != 0) return -1; \
    } while (0)

inline unsigned blocks_for(size_t n, unsigned per = 256) { return unsigned((n + per - 1) / per); }

int run_forward(Ctx* ctx, float* out_rgb) {
    const int n = ctx->n;
    const GscCamera cam = ctx->cam;
    ctx->tile_bw = (cam.width + kTile - 1) / kTile;
    ctx->tile_bh = (cam.height + kTile - 1) / kTile;
    const uint32_t num_tiles = ctx->tile_bw * ctx->tile_bh;
    const size_t pixels = ctx->pixels();

    float2* xy;
    float4 *conic_op, *color;
    float *depth, *extk;
    int4* rect;
    uint32_t *tiles, *offsets;
    GSC_TRY(ensure(ctx, ctx->xy, n, &xy));
    GSC_TRY(ensure(ctx, ctx->conic_op, n, &conic_op));
    GSC_TRY(ensure(ctx, ctx->color, n, &color));
    GSC_TRY(ensure(ctx, ctx->depth, n, &depth));
    GSC_TRY(ensure(ctx, ctx->rect, n, &rect));
    GSC_TRY(ensure(ctx, ctx->extk, n, &extk));
    GSC_TRY(ensure(ctx, ctx->tiles, n, &tiles));
    GSC_TRY(ensure(ctx, ctx->offsets, n, &offsets));
    uint2* ranges;
    float *image, *final_t;
    uint32_t* last_idx;
    GSC_TRY(ensure(ctx, ctx->ranges, num_tiles, &ranges));
    GSC_TRY(ensure(ctx, ctx->image, pixels * 3, &image));
    GSC_TRY(ensure(ctx, ctx->final_t, pixels, &final_t));
    GSC_TRY(ensure(ctx, ctx->last_idx, pixels, &last_idx));

    cudaEvent_t* ev = ctx->ev;
    GSC_CHECK(ctx, cudaEventRecord(ev[0]));
    if (n > 0) {
        project_kernel<<<blocks_for(n), 256>>>(n, cam, ctx->rest_pc(), ctx->act,
                                               static_cast<const float*>(ctx->transforms.ptr),
                                               static_cast<const float*>(ctx->opacity.ptr),
                                               static_cast<const float*>(ctx->sh.ptr), xy, conic_op, color, depth,
                                               rect, extk, tiles);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_CHECK(ctx, cudaEventRecord(ev[1]));
    uint32_t num_isect = 0;
    if (n > 0) {
        size_t temp_bytes = 0;
        GSC_CHECK(ctx, cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, tiles, offsets, n));
        uint8_t* temp;
        GSC_TRY(ensure(ctx, ctx->scan_temp, temp_bytes, &temp));
        GSC_CHECK(ctx, cub::DeviceScan::InclusiveSum(temp, temp_bytes, tiles, offsets, n));
        GSC_CHECK(ctx, cudaMemcpy(&num_isect, offsets + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
    }
    ctx->num_isect = num_isect;
    GSC_CHECK(ctx, cudaEventRecord(ev[2]));

    unsigned long long *keys, *keys_sorted;
    uint32_t *vals, *vals_sorted;
    GSC_TRY(ensure(ctx, ctx->keys, num_isect, &keys));
    GSC_TRY(ensure(ctx, ctx->vals, num_isect, &vals));
    GSC_TRY(ensure(ctx, ctx->keys_sorted, num_isect, &keys_sorted));
    GSC_TRY(ensure(ctx, ctx->vals_sorted, num_isect, &vals_sorted));
    if (num_isect > 0) {
        duplicate_kernel<<<blocks_for(n), 256>>>(n, cam, ctx->tile_bw, xy, conic_op, depth, rect, extk, tiles,
                                                 offsets, keys, vals);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_CHECK(ctx, cudaEventRecord(ev[3]));
    if (num_isect > 0) {
        int tile_bits = 1;
        while ((1u << tile_bits) <= num_tiles) ++tile_bits;
        size_t temp_bytes = 0;
        GSC_CHECK(ctx, cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys, keys_sorted, vals, vals_sorted,
                                                       int(num_isect), 0, 32 + tile_bits));
        uint8_t* temp;
        GSC_TRY(ensure(ctx, ctx->sort_temp, temp_bytes, &temp));
        GSC_CHECK(ctx, cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys, keys_sorted, vals, vals_sorted,
                                                       int(num_isect), 0, 32 + tile_bits));
    }
    GSC_CHECK(ctx, cudaEventRecord(ev[4]));
    GSC_CHECK(ctx, cudaMemsetAsync(ranges, 0, num_tiles * sizeof(uint2)));
    if (num_isect > 0) {
        ranges_kernel<<<blocks_for(num_isect), 256>>>(num_isect, keys_sorted, ranges);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_CHECK(ctx, cudaEventRecord(ev[5]));
    raster_fwd_kernel<<<dim3(ctx->tile_bw, ctx->tile_bh), kBlock>>>(cam.width, cam.height, ctx->bg, ranges,
                                                                    vals_sorted, xy, conic_op, color, image, final_t,
                                                                    last_idx);
    GSC_CHECK(ctx, cudaGetLastError());
    GSC_CHECK(ctx, cudaEventRecord(ev[6]));
    GSC_CHECK(ctx, cudaEventSynchronize(ev[6]));
    for (int s = 0; s < 6; ++s) GSC_CHECK(ctx, cudaEventElapsedTime(&ctx->ms[s], ev[s], ev[s + 1]));
    if (out_rgb) GSC_CHECK(ctx, cudaMemcpy(out_rgb, image, pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    ctx->have_forward = true;
    return 0;
}

int run_backward(Ctx* ctx, const float* d_image_host, float* g_transforms_host, float* g_opacity_host,
                 float* g_sh_host, float* screen_grads_host) {
    if (!ctx->have_forward) {
        ctx->error = "backward called before render";
        return -1;
    }
    const int n = ctx->n;
    const size_t pixels = ctx->pixels();
    float *d_image, *screen_grads, *g_transforms, *g_opacity, *g_sh;
    GSC_TRY(ensure(ctx, ctx->d_image, pixels * 3, &d_image));
    GSC_TRY(ensure(ctx, ctx->screen_grads, size_t(n) * kNumGrad, &screen_grads));
    GSC_TRY(ensure(ctx, ctx->g_transforms, size_t(n) * 10, &g_transforms));
    GSC_TRY(ensure(ctx, ctx->g_opacity, n, &g_opacity));
    GSC_TRY(ensure(ctx, ctx->g_sh, size_t(n) * ctx->row_len(), &g_sh));
    GSC_CHECK(ctx, cudaMemcpy(d_image, d_image_host, pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    GSC_CHECK(ctx, cudaMemset(screen_grads, 0, size_t(n) * kNumGrad * sizeof(float)));
    GSC_CHECK(ctx, cudaMemset(g_transforms, 0, size_t(n) * 10 * sizeof(float)));
    GSC_CHECK(ctx, cudaMemset(g_opacity, 0, size_t(n) * sizeof(float)));
    GSC_CHECK(ctx, cudaMemset(g_sh, 0, size_t(n) * ctx->row_len() * sizeof(float)));

    cudaEvent_t* ev = ctx->ev;
    GSC_CHECK(ctx, cudaEventRecord(ev[6]));
    raster_bwd_kernel<<<dim3(ctx->tile_bw, ctx->tile_bh), kBlock>>>(
        ctx->cam.width, ctx->cam.height, ctx->bg, static_cast<const uint2*>(ctx->ranges.ptr),
        static_cast<const uint32_t*>(ctx->vals_sorted.ptr), static_cast<const float2*>(ctx->xy.ptr),
        static_cast<const float4*>(ctx->conic_op.ptr), static_cast<const float4*>(ctx->color.ptr),
        static_cast<const float*>(ctx->final_t.ptr), static_cast<const uint32_t*>(ctx->last_idx.ptr), d_image,
        screen_grads);
    GSC_CHECK(ctx, cudaGetLastError());
    GSC_CHECK(ctx, cudaEventRecord(ev[7]));
    if (n > 0) {
        project_bwd_kernel<<<blocks_for(n), 256>>>(
            n, ctx->cam, ctx->rest_pc(), ctx->act, static_cast<const float*>(ctx->transforms.ptr),
            static_cast<const float*>(ctx->opacity.ptr), static_cast<const float*>(ctx->sh.ptr),
            static_cast<const uint32_t*>(ctx->tiles.ptr), screen_grads, g_transforms, g_opacity, g_sh);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_CHECK(ctx, cudaEventRecord(ev[8]));
    GSC_CHECK(ctx, cudaEventSynchronize(ev[8]));
    GSC_CHECK(ctx, cudaEventElapsedTime(&ctx->ms[6], ev[6], ev[7]));
    GSC_CHECK(ctx, cudaEventElapsedTime(&ctx->ms[7], ev[7], ev[8]));
    if (g_transforms_host)
        GSC_CHECK(ctx, cudaMemcpy(g_transforms_host, g_transforms, size_t(n) * 10 * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    if (g_opacity_host)
        GSC_CHECK(ctx, cudaMemcpy(g_opacity_host, g_opacity, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
    if (g_sh_host)
        GSC_CHECK(ctx, cudaMemcpy(g_sh_host, g_sh, size_t(n) * ctx->row_len() * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    if (screen_grads_host)
        GSC_CHECK(ctx, cudaMemcpy(screen_grads_host, screen_grads, size_t(n) * kNumGrad * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
#define GSC_EXPORT __declspec(dllexport)
#else
#define GSC_EXPORT __attribute__((visibility("default")))
#endif

GSC_EXPORT uint32_t visloc_gsplat_cuda_abi_version() { return kAbiVersion; }

GSC_EXPORT void* visloc_gsplat_cuda_create() {
    Ctx* ctx = new (std::nothrow) Ctx();
    if (!ctx) return nullptr;
    for (auto& e : ctx->ev) {
        if (cudaEventCreate(&e) != cudaSuccess) {
            ctx->error = "cudaEventCreate failed (no CUDA device?)";
            break;
        }
    }
    return ctx;
}

GSC_EXPORT void visloc_gsplat_cuda_destroy(void* handle) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return;
    Buf* bufs[] = {&ctx->transforms, &ctx->opacity,  &ctx->sh,         &ctx->xy,          &ctx->conic_op,
                   &ctx->color,      &ctx->depth,    &ctx->rect,       &ctx->extk,        &ctx->tiles,
                   &ctx->offsets,    &ctx->keys,     &ctx->vals,       &ctx->keys_sorted, &ctx->vals_sorted,
                   &ctx->sort_temp,  &ctx->scan_temp, &ctx->ranges,    &ctx->image,       &ctx->final_t,
                   &ctx->last_idx,   &ctx->d_image,  &ctx->screen_grads, &ctx->g_transforms, &ctx->g_opacity,
                   &ctx->g_sh};
    for (Buf* b : bufs)
        if (b->ptr) cudaFree(b->ptr);
    for (auto& e : ctx->ev)
        if (e) cudaEventDestroy(e);
    delete ctx;
}

GSC_EXPORT const char* visloc_gsplat_cuda_last_error(void* handle) {
    return handle ? static_cast<Ctx*>(handle)->error.c_str() : "null context";
}

GSC_EXPORT const char* visloc_gsplat_cuda_stage_names() { return kStageNames; }

// Upload a packed scene (host pointers). `sh` holds 3 * (degree + 1)^2 floats
// per gaussian.
GSC_EXPORT int visloc_gsplat_cuda_set_scene(void* handle, int n, int sh_degree, const float* transforms,
                                            const float* opacity, const float* sh) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (n < 0 || sh_degree < 0 || sh_degree > 3) {
        ctx->error = "set_scene: n must be >= 0 and the SH degree in 0..=3";
        return -1;
    }
    ctx->n = n;
    ctx->sh_degree = sh_degree;
    ctx->have_forward = false;
    float *t, *o, *s;
    GSC_TRY(ensure(ctx, ctx->transforms, size_t(n) * 10, &t));
    GSC_TRY(ensure(ctx, ctx->opacity, n, &o));
    GSC_TRY(ensure(ctx, ctx->sh, size_t(n) * ctx->row_len(), &s));
    if (n > 0) {
        GSC_CHECK(ctx, cudaMemcpy(t, transforms, size_t(n) * 10 * sizeof(float), cudaMemcpyHostToDevice));
        GSC_CHECK(ctx, cudaMemcpy(o, opacity, size_t(n) * sizeof(float), cudaMemcpyHostToDevice));
        GSC_CHECK(ctx, cudaMemcpy(s, sh, size_t(n) * ctx->row_len() * sizeof(float), cudaMemcpyHostToDevice));
    }
    return 0;
}

// Render the uploaded scene. `out_rgb` (nullable) receives width * height * 3
// floats, row-major RGB. `num_intersections` (nullable) receives the number
// of (tile, gaussian) pairs.
GSC_EXPORT int visloc_gsplat_cuda_render(void* handle, const GscCamera* cam, const float* bg, int sh_active_degree,
                                         float* out_rgb, uint32_t* num_intersections) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!cam || !bg || cam->width == 0 || cam->height == 0) {
        ctx->error = "render: camera and background are required, image size must be positive";
        return -1;
    }
    if (sh_active_degree < 0 || sh_active_degree > ctx->sh_degree) {
        ctx->error = "render: active SH degree must be in 0..=scene degree";
        return -1;
    }
    ctx->cam = *cam;
    ctx->bg = make_float3(bg[0], bg[1], bg[2]);
    ctx->act = (sh_active_degree + 1) * (sh_active_degree + 1) - 1;
    GSC_TRY(run_forward(ctx, out_rgb));
    if (num_intersections) *num_intersections = ctx->num_isect;
    return 0;
}

// Gradients of L = sum_px dot(d_image[px], C[px]) for the last render, in
// the scene layouts (all outputs nullable). Gaussians that were not visible
// get zeros. `screen_grads` receives 10 floats per gaussian: du dv dA dB dC
// dopacity dr dg db and the refine weight.
GSC_EXPORT int visloc_gsplat_cuda_backward(void* handle, const float* d_image, float* g_transforms,
                                           float* g_opacity, float* g_sh, float* screen_grads) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!d_image) {
        ctx->error = "backward: d_image is required";
        return -1;
    }
    return run_backward(ctx, d_image, g_transforms, g_opacity, g_sh, screen_grads);
}

// Milliseconds of each stage of the last render / backward (GPU time between
// CUDA events), in the order of visloc_gsplat_cuda_stage_names. Returns the
// number of stages written.
GSC_EXPORT int visloc_gsplat_cuda_stage_times(void* handle, float* ms, int cap) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !ms) return -1;
    const int count = cap < kNumStages ? cap : kNumStages;
    for (int s = 0; s < count; ++s) ms[s] = ctx->ms[s];
    return count;
}

}  // extern "C"
