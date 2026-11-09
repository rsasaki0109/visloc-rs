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
//
// Training (ABI 2) keeps the scene, its Adam moments and the ground-truth
// images on the device; one train_step call runs the wgpu trainer's step
// (loss_l1, ssim, densify_stats, adam, mean_noise shaders). brush's refine
// is split as in the wgpu trainer: the host reads the statistics and picks
// an action per gaussian, the device scatters (densify_scatter.wgsl).
//
// Semantic training (ABI 13): with labels per image (set_labels), every
// gaussian also carries a logit per class. A step blends the softmax of each
// gaussian's logits, its class probabilities, like colour, adds the
// cross-entropy of the pixel's normalised blend against its label, and
// differentiates it into the logits and, through alpha, the gaussians'
// geometry and opacity. Blending probabilities rather than logits keeps each
// gaussian's own class the one it renders. Since ABI 14 the colour kernels
// carry it (raster_fwd_kernel and raster_bwd_kernel with kLab,
// label_loss_kernel): the loss needs only the label's channel and the
// coverage, and its gradient two floats per pixel. sem_fwd_kernel and
// sem_bwd_kernel blend and differentiate every class channel, for
// render_classes and classes_backward. Since ABI 15 the labels move the
// gaussians' opacities and centres but not their shapes unless asked
// (semantic_geometry 2): through the shape they grew splats over the pixels
// of their class, large ones that showed as floaters from other views.
//
// Since ABI 16 a step can keep the Adam moments in bf16 (moments_bf16): the
// dense optimizer moves 20 bytes per element instead of 28. The moments are
// rounded stochastically, as Adam's second moment changes by 0.1% a step
// (beta2 0.999), under half a bf16 step (0.2%), and round to nearest would
// leave it where it is.

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <type_traits>
#include <vector>

extern "C" {

// World-to-camera rotation (row-major) and translation, camera centre in
// world coordinates, pinhole intrinsics and image size, and the lens in
// front of them (gsplat-core's `Lens`, see lens.cuh): lens_kind 0 is the
// undistorted pinhole, 1 a distorted pinhole, 2 an equidistant (fisheye)
// projection; lens holds the distortion [k1 k2 k3 k4 p1 p2 sx1 sy1] and
// max_theta the widest ray angle the lens is followed to (radians; past it,
// its first-order extension). Mirrored by `GscCamera` in
// crates/gsplat-cuda/src/lib.rs.
struct GscCamera {
    float rot[9];
    float t[3];
    float center[3];
    float fx, fy, cx, cy;
    uint32_t width, height;
    uint32_t lens_kind;
    float lens[8];
    float max_theta;
};

}  // extern "C"

#include "lens.cuh"

namespace {

constexpr uint32_t kAbiVersion = 17;
constexpr int kTile = 16;
constexpr int kBlock = kTile * kTile;
constexpr int kBwdBatch = 128;
// The backward pass takes kBwdPix pixels per thread, kBwdThreads per tile.
constexpr int kBwdPix = 2;
constexpr int kBwdThreads = kBlock / kBwdPix;
static_assert(kBwdPix == 2 && kBwdBatch == kBwdThreads, "32 staged splats per backward warp");
// du dv dA dB dC dopacity dr dg db, and the refine weight (brush / AbsGS:
// sum over pixels of |dL/du| W + |dL/dv| H).
constexpr int kNumGrad = 10;
// Classes of the semantic logits.
constexpr int kMaxSem = 16;

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

// compute_projected for a camera with a lens (lens_kind != kLensPinhole):
// the lens's Jacobian (lens.cuh) in place of the pinhole's.
__device__ Projected compute_projected_lens(const GscCamera& c, const float* tr, float raw_opac) {
    Projected p = {};
    p.ok = false;
    const float opacity = 1.f / (1.f + expf(-raw_opac));
    if (opacity < 1e-4f) return p;
    const float* W = c.rot;
    const float mx = tr[0], my = tr[1], mz = tr[2];
    const float px = W[0] * mx + W[1] * my + W[2] * mz + c.t[0];
    const float py = W[3] * mx + W[4] * my + W[5] * mz + c.t[1];
    const float pz = W[6] * mx + W[7] * my + W[8] * mz + c.t[2];
    // A distorted pinhole draws what lies 0.1 in front of it, sorted by z; a
    // fisheye what lies 0.1 from its centre within the wider of max_theta
    // and 90 degrees, even behind the image plane, sorted by the distance
    // (cpu_render's project_point).
    float depth = pz;
    if (c.lens_kind == kLensEquidistant) {
        depth = sqrtf(px * px + py * py + pz * pz);
        if (depth <= 0.1f || lens_ray_angle(px, py, pz) > fmaxf(c.max_theta, 1.57079633f)) return p;
    } else if (pz <= 0.1f) {
        return p;
    }

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

    // The lens at the point, past max_theta its extension from the point's
    // turn q onto the cone; a distorted pinhole linearises at q clamped to
    // 1.3x the frustum (Inria rasterizer), a fisheye at q itself.
    const float lim_x = 1.3f * 0.5f * float(c.width) / c.fx;
    const float lim_y = 1.3f * 0.5f * float(c.height) / c.fy;
    const float pt[3] = {px, py, pz};
    float m[2], J[2][3], q[3];
    lens_extended(c, pt, m, J, q);
    if (c.lens_kind == kLensDistorted) {
        const float inv_qz = 1.f / q[2];
        const float lin[3] = {clampf(q[0] * inv_qz, -lim_x, lim_x) * q[2], clampf(q[1] * inv_qz, -lim_y, lim_y) * q[2],
                              q[2]};
        float ml[2];
        lens_coords(c, lin, ml, J);
    }
    const float j0[3] = {c.fx * J[0][0], c.fx * J[0][1], c.fx * J[0][2]};
    const float j1[3] = {c.fy * J[1][0], c.fy * J[1][1], c.fy * J[1][2]};
    // j^T cov j' over the symmetric camera covariance.
    auto jcj = [&](const float* x, const float* y) {
        return x[0] * (cov00 * y[0] + cov01 * y[1] + cov02 * y[2]) +
               x[1] * (cov01 * y[0] + cov11 * y[1] + cov12 * y[2]) +
               x[2] * (cov02 * y[0] + cov12 * y[1] + cov22 * y[2]);
    };
    const float a = jcj(j0, j0) + 0.3f;
    const float b = jcj(j0, j1);
    const float cc = jcj(j1, j1) + 0.3f;
    const float u = c.fx * m[0] + c.cx;
    const float v = c.fy * m[1] + c.cy;
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
    p.depth = depth;
    p.tx0 = int(floorf(x0f)) / kTile;
    p.ty0 = int(floorf(y0f)) / kTile;
    p.tx1 = int(floorf(x1f)) / kTile;
    p.ty1 = int(floorf(y1f)) / kTile;
    return p;
}

// A gaussian's screen footprint: the undistorted pinhole's path; a camera
// with a lens goes to compute_projected_lens.
__device__ Projected compute_projected(const GscCamera& c, const float* tr, float raw_opac) {
    if (c.lens_kind != kLensPinhole) return compute_projected_lens(c, tr, raw_opac);
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

// Pixel columns [x, y] (empty when x > y) holding a pixel the rasterizer
// could blend among the pixel rows py0..=py1. See tile_row_span in
// common.wgsl for the derivation: the footprint is convex, so over a band of
// rows its x-extent is analytic, and the small slack on k only ever keeps a
// borderline pixel.
__device__ float2 footprint_columns(const Projected& p, int py0, int py1) {
    const float2 empty = make_float2(1.f, 0.f);
    const float det = p.c00 * p.c11 - p.c01 * p.c01;
    if (!(det > 0.f) || !(p.c00 > 0.f) || !(p.c11 > 0.f)) return empty;
    const float k = p.extent_k * 1.001f + 1e-3f;
    const float y0 = float(py0) + 0.5f - p.v;
    const float y1 = float(py1) + 0.5f - p.v;
    const float x_max = sqrtf(k * p.c11 / det);
    const float yr = clampf(-p.c01 * x_max / p.c11, y0, y1);
    const float yl = clampf(p.c01 * x_max / p.c11, y0, y1);
    const float dr = k * p.c00 - yr * yr * det;
    if (dr < 0.f) return empty;
    const float dl = fmaxf(k * p.c00 - yl * yl * det, 0.f);
    const float xr = (-p.c01 * yr + sqrtf(dr)) / p.c00;
    const float xl = (-p.c01 * yl - sqrtf(dl)) / p.c00;
    return make_float2(ceilf(p.u + xl - 0.5f), floorf(p.u + xr - 0.5f));
}

// Tiles of row `ty` holding a pixel the rasterizer could blend, as an
// inclusive column range (empty when x > y).
__device__ int2 tile_row_span(const Projected& p, int ty, uint32_t img_w, uint32_t img_h) {
    const float2 cols = footprint_columns(p, ty * kTile, min(ty * kTile + kTile - 1, int(img_h) - 1));
    const float px_lo = fmaxf(cols.x, float(p.tx0 * kTile));
    const float px_hi = fminf(cols.y, float(min(p.tx1 * kTile + kTile - 1, int(img_w) - 1)));
    if (px_hi < px_lo) return make_int2(1, 0);
    return make_int2(int(px_lo) / kTile, int(px_hi) / kTile);
}

// The footprint's columns in each 4-row band of tile row `ty` (empty past
// the image), shared by every tile of the row.
__device__ void band_columns(const Projected& p, int ty, uint32_t img_h, float2 bands[kTile / 4]) {
#pragma unroll
    for (int band = 0; band < kTile / 4; ++band) {
        const int py0 = ty * kTile + band * 4;
        bands[band] = py0 < int(img_h) ? footprint_columns(p, py0, min(py0 + 3, int(img_h) - 1)) : make_float2(1.f, 0.f);
    }
}

// Bit w set: the footprint reaches the 8x4 sub-tile of warp w (see
// tile_pixel) in tile column `tx` of the row whose bands are given. A
// superset of the warps that can blend the splat, so the rasterizer kernels
// skip the others without evaluating them.
__device__ uint32_t subtile_mask(const float2 bands[kTile / 4], int tx) {
    uint32_t mask = 0;
    const float x0 = float(tx * kTile);
#pragma unroll
    for (int band = 0; band < kTile / 4; ++band) {
        const float2 c = bands[band];
        if (c.x <= x0 + 7.f && c.y >= x0) mask |= 1u << (2 * band);
        if (c.x <= x0 + 15.f && c.y >= x0 + 8.f) mask |= 1u << (2 * band + 1);
    }
    return mask;
}

// Rasterizer list entries pack the gaussian id with its sub-tile mask in the
// top byte, so the sort carries the mask along.
constexpr uint32_t kIdBits = 24;
constexpr uint32_t kIdMask = (1u << kIdBits) - 1;
constexpr int kMaxGaussians = 1 << kIdBits;

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
        if (span.y < span.x) continue;
        float2 bands[kTile / 4];
        band_columns(p, ty, cam.height, bands);
        for (int tx = span.x; tx <= span.y && out < end; ++tx) {
            const unsigned long long tile = uint32_t(ty) * tile_bw + uint32_t(tx);
            keys[out] = (tile << 32) | dbits;
            vals[out] = uint32_t(i) | (subtile_mask(bands, tx) << kIdBits);
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

// Floats per staged splat of the label channel's class probabilities: the
// class count made odd, so threads staging consecutive splats write distinct
// shared-memory banks.
__host__ __device__ inline int lab_stride(int k) { return k | 1; }

// Pixel of thread `tid` within its 16x16 tile: warp w covers the 8x4
// sub-tile (w % 2, w / 2), compact so that a splat touches few warps.
__device__ inline void tile_pixel(int tid, uint32_t& lx, uint32_t& ly) {
    const int warp = tid >> 5, lane = tid & 31;
    lx = uint32_t((warp & 1) * 8 + (lane & 7));
    ly = uint32_t((warp >> 1) * 4 + (lane >> 3));
}

// The backward's sub-tile q < kBwdPix of warp w: the warp covers the 8x8
// block of sub-tiles 4 (w / 2) + w % 2 and the one below it (as tile_pixel
// numbers them, 8x4 each).
__device__ inline uint32_t bwd_subtile(int warp, int q) { return uint32_t(4 * (warp >> 1) + (warp & 1) + 2 * q); }

// One block per 16x16 tile, one thread per pixel, splats staged in shared
// memory 256 at a time; the block stops once every pixel is saturated. A
// warp skips the splats whose footprint misses its 8x4 sub-tile (the mask in
// the list entry's top byte). With kDepth, out_depth also receives per pixel
// the median depth (camera z of the splat at which the transmittance first
// drops to 0.5 or below; 0 when it never does) and the opacity 1 - T, as
// rasterize_depth.wgsl. With `planar` the median depth is where the pixel's
// ray meets that splat's plane (`plane`, plane_kernel) instead of the splat
// centre's depth (falling back to it at grazing angles); the ray is
// `rays[pixel]` through a lens (camera_rays: the depth comes out as the
// lens's, z or a fisheye's distance), else a pinhole's from `intr`, (fx, fy,
// cx, cy). With kDepth, `acc_depth` (nullable) also receives the expected depth
// sum_i T_i a_i z_i of the splat centres, the depth channel the depth-prior
// and depth-normal losses train, and `acc_normal` (nullable, 3 floats per
// pixel; needs `plane`) the normal channel sum_i T_i a_i n_i of the splats'
// plane normals, as rasterize.wgsl's geometry variant. With kLab, `acc_lab`
// receives at each pixel whose label (`labels`) is below `sem_k` the blended
// probability of that class, P_y = sum_i T_i a_i p_iy (`probs`, sem_k per
// gaussian; 0 elsewhere): all the labels' cross-entropy needs
// (label_loss_kernel), staged in dynamic shared memory (lab_smem bytes).
template <bool kDepth, bool kLab>
__global__ void __launch_bounds__(kBlock) raster_fwd_kernel(
    uint32_t img_w, uint32_t img_h, float3 bg, const uint2* __restrict__ ranges, const uint32_t* __restrict__ vals,
    const float2* __restrict__ xy, const float4* __restrict__ conic_op, const float4* __restrict__ color,
    const float* __restrict__ depth, const float4* __restrict__ plane, int planar, float4 intr,
    const float3* __restrict__ rays, float* __restrict__ out_img, float* __restrict__ final_t,
    uint32_t* __restrict__ last_idx, float* __restrict__ out_depth, float* __restrict__ acc_depth,
    float* __restrict__ acc_normal, int sem_k, const float* __restrict__ probs, const uint8_t* __restrict__ labels,
    float* __restrict__ acc_lab) {
    __shared__ float2 s_xy[kBlock];
    __shared__ float4 s_co[kBlock];
    __shared__ float3 s_col[kBlock];
    __shared__ uint32_t s_mask[kBlock];
    __shared__ float s_depth[kDepth ? kBlock : 1];
    __shared__ float4 s_plane[kDepth ? kBlock : 1];
    extern __shared__ float s_prob[];  // kLab: lab_stride(sem_k) per staged splat
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const uint32_t warp_bit = 1u << (tid >> 5);
    uint32_t lx, ly;
    tile_pixel(tid, lx, ly);
    const uint32_t px = blockIdx.x * kTile + lx;
    const uint32_t py = blockIdx.y * kTile + ly;
    const bool in_image = px < img_w && py < img_h;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];
    // The pixel's ray through a lens (for the planar depth); a pinhole's
    // comes from `intr`.
    const float3 ray = kDepth && rays && in_image ? rays[py * img_w + px] : make_float3(0.f, 0.f, 0.f);
    const int ks = lab_stride(sem_k);
    const uint32_t lab = kLab && in_image ? labels[py * img_w + px] : 0xffu;
    const uint32_t lab_col = lab < uint32_t(sem_k) ? lab : 0u;

    float3 acc = make_float3(0.f, 0.f, 0.f);
    float trans = 1.f;
    uint32_t last = 0;
    float median = 0.f;
    float zacc = 0.f;
    float3 nacc = make_float3(0.f, 0.f, 0.f);
    float pacc = 0.f;
    bool done = !in_image;
    for (uint32_t start = range.x; start < range.y; start += kBlock) {
        // Also the barrier that protects the staged batch before it is reloaded.
        if (__syncthreads_count(done) == kBlock) break;
        const uint32_t cnt = min(uint32_t(kBlock), range.y - start);
        if (uint32_t(tid) < cnt) {
            const uint32_t v = vals[start + tid];
            const uint32_t g = v & kIdMask;
            s_xy[tid] = xy[g];
            s_co[tid] = conic_op[g];
            const float4 c = color[g];
            s_col[tid] = make_float3(c.x, c.y, c.z);
            s_mask[tid] = v >> kIdBits;
            if (kDepth) {
                s_depth[tid] = depth[g];
                if (plane) s_plane[tid] = plane[g];
            }
            if (kLab)
                for (int c = 0; c < sem_k; ++c) s_prob[tid * ks + c] = probs[size_t(g) * sem_k + c];
        }
        __syncthreads();
        if (!done) {
            for (uint32_t s = 0; s < cnt; ++s) {
                if (trans <= 1e-4f) break;
                if (!(s_mask[s] & warp_bit)) continue;
                const float dx = sx - s_xy[s].x;
                const float dy = sy - s_xy[s].y;
                const float4 co = s_co[s];
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma >= 0.f) {
                    const float a = fminf(0.99f, co.w * expf(-sigma));
                    if (a >= 1.f / 255.f) {
                        const float3 c = s_col[s];
                        const float w = trans * a;
                        if (kLab) pacc += w * s_prob[s * ks + lab_col];
                        if (kDepth) {
                            zacc += w * s_depth[s];
                            if (acc_normal) {
                                const float4 pl = s_plane[s];
                                nacc.x += w * pl.x;
                                nacc.y += w * pl.y;
                                nacc.z += w * pl.z;
                            }
                        }
                        acc.x += w * fmaxf(c.x, 0.f);
                        acc.y += w * fmaxf(c.y, 0.f);
                        acc.z += w * fmaxf(c.z, 0.f);
                        const float before = trans;
                        trans *= 1.f - a;
                        last = start + s + 1;
                        if (kDepth && before > 0.5f && trans <= 0.5f) {
                            median = s_depth[s];
                            if (planar) {
                                const float4 pl = s_plane[s];
                                const float nr = rays ? pl.x * ray.x + pl.y * ray.y + pl.z * ray.z
                                                      : pl.x * (sx - intr.z) / intr.x + pl.y * (sy - intr.w) / intr.y +
                                                            pl.z;
                                const float z = nr != 0.f ? pl.w / nr : 0.f;
                                // Grazing planes put the hit far along the ray.
                                if (z > 0.f && z < 2.f * median) median = z;
                            }
                        }
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
        if (kLab) acc_lab[pix] = lab < uint32_t(sem_k) ? pacc : 0.f;
        if (kDepth) {
            out_depth[pix * 2] = median;
            out_depth[pix * 2 + 1] = 1.f - trans;
            if (acc_depth) acc_depth[pix] = zacc;
            if (acc_normal) {
                acc_normal[pix * 3 + 0] = nacc.x;
                acc_normal[pix * 3 + 1] = nacc.y;
                acc_normal[pix * 3 + 2] = nacc.z;
            }
        }
    }
}

// Label votes of the last forward: at every pixel whose label is below
// `num_classes`, each splat's blending weight there (T_i a_i, composited as
// raster_fwd_kernel does) is added to votes[g * num_classes + label]. Pixels
// labelled `num_classes` or more cast no vote.
__global__ void __launch_bounds__(kBlock) vote_kernel(uint32_t img_w, uint32_t img_h,
                                                      const uint2* __restrict__ ranges,
                                                      const uint32_t* __restrict__ vals, const float2* __restrict__ xy,
                                                      const float4* __restrict__ conic_op,
                                                      const uint8_t* __restrict__ labels, uint32_t num_classes,
                                                      float* __restrict__ votes) {
    __shared__ float2 s_xy[kBlock];
    __shared__ float4 s_co[kBlock];
    __shared__ uint32_t s_id[kBlock];
    __shared__ uint32_t s_mask[kBlock];
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const uint32_t warp_bit = 1u << (tid >> 5);
    uint32_t lx, ly;
    tile_pixel(tid, lx, ly);
    const uint32_t px = blockIdx.x * kTile + lx;
    const uint32_t py = blockIdx.y * kTile + ly;
    const bool in_image = px < img_w && py < img_h;
    const uint32_t label = in_image ? labels[py * img_w + px] : num_classes;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];

    float trans = 1.f;
    bool done = label >= num_classes;
    for (uint32_t start = range.x; start < range.y; start += kBlock) {
        // Also the barrier that protects the staged batch before it is reloaded.
        if (__syncthreads_count(done) == kBlock) break;
        const uint32_t cnt = min(uint32_t(kBlock), range.y - start);
        if (uint32_t(tid) < cnt) {
            const uint32_t v = vals[start + tid];
            const uint32_t g = v & kIdMask;
            s_xy[tid] = xy[g];
            s_co[tid] = conic_op[g];
            s_id[tid] = g;
            s_mask[tid] = v >> kIdBits;
        }
        __syncthreads();
        if (!done) {
            for (uint32_t s = 0; s < cnt; ++s) {
                if (trans <= 1e-4f) break;
                if (!(s_mask[s] & warp_bit)) continue;
                const float dx = sx - s_xy[s].x;
                const float dy = sy - s_xy[s].y;
                const float4 co = s_co[s];
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma >= 0.f) {
                    const float a = fminf(0.99f, co.w * expf(-sigma));
                    if (a >= 1.f / 255.f) {
                        atomicAdd(&votes[size_t(s_id[s]) * num_classes + label], trans * a);
                        trans *= 1.f - a;
                    }
                }
            }
            done = trans <= 1e-4f;
        }
    }
}

// The class channels of the last forward: per pixel
// S_c = sum_i T_i a_i z_ic over the splats raster_fwd_kernel blended
// (background 0), z_i the gaussian's `k` values (its class probabilities in
// training), into acc_sem (k floats per pixel), as render_classes gives them.
__global__ void __launch_bounds__(kBlock) sem_fwd_kernel(uint32_t img_w, uint32_t img_h, int k,
                                                         const uint2* __restrict__ ranges,
                                                         const uint32_t* __restrict__ vals,
                                                         const float2* __restrict__ xy,
                                                         const float4* __restrict__ conic_op,
                                                         const float* __restrict__ logits,
                                                         float* __restrict__ acc_sem) {
    __shared__ float2 s_xy[kBlock];
    __shared__ float4 s_co[kBlock];
    __shared__ uint32_t s_mask[kBlock];
    __shared__ float s_z[kBlock * kMaxSem];
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const uint32_t warp_bit = 1u << (tid >> 5);
    uint32_t lx, ly;
    tile_pixel(tid, lx, ly);
    const uint32_t px = blockIdx.x * kTile + lx;
    const uint32_t py = blockIdx.y * kTile + ly;
    const bool in_image = px < img_w && py < img_h;
    const uint32_t pix = py * img_w + px;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];

    float acc[kMaxSem];
#pragma unroll
    for (int c = 0; c < kMaxSem; ++c) acc[c] = 0.f;
    float trans = 1.f;
    bool done = !in_image;
    for (uint32_t start = range.x; start < range.y; start += kBlock) {
        // Also the barrier that protects the staged batch before it is reloaded.
        if (__syncthreads_count(done) == kBlock) break;
        const uint32_t cnt = min(uint32_t(kBlock), range.y - start);
        if (uint32_t(tid) < cnt) {
            const uint32_t v = vals[start + tid];
            const uint32_t g = v & kIdMask;
            s_xy[tid] = xy[g];
            s_co[tid] = conic_op[g];
            s_mask[tid] = v >> kIdBits;
            for (int c = 0; c < k; ++c) s_z[tid * kMaxSem + c] = logits[size_t(g) * k + c];
        }
        __syncthreads();
        if (!done) {
            for (uint32_t s = 0; s < cnt; ++s) {
                if (trans <= 1e-4f) break;
                if (!(s_mask[s] & warp_bit)) continue;
                const float dx = sx - s_xy[s].x;
                const float dy = sy - s_xy[s].y;
                const float4 co = s_co[s];
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma >= 0.f) {
                    const float a = fminf(0.99f, co.w * expf(-sigma));
                    if (a >= 1.f / 255.f) {
                        const float w = trans * a;
#pragma unroll
                        for (int c = 0; c < kMaxSem; ++c)
                            if (c < k) acc[c] += w * s_z[s * kMaxSem + c];
                        trans *= 1.f - a;
                    }
                }
            }
            done = trans <= 1e-4f;
        }
    }
    if (in_image) {
#pragma unroll
        for (int c = 0; c < kMaxSem; ++c)
            if (c < k) acc_sem[size_t(pix) * k + c] = acc[c];
    }
}

// One butterfly step: keep `lo` or `hi` by this lane's bit, add the partner's
// other half.
__device__ inline float xstep(float lo, float hi, bool h, int m) {
    return (h ? hi : lo) + __shfl_xor_sync(0xffffffffu, h ? lo : hi, m);
}

// 32 per-lane values summed over the warp by the transposed xor butterfly
// (31 shuffles): lane l returns the warp's sum of v[l].
__device__ inline float warp_sum32(float (&v)[32], int lane) {
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) {
        const bool h = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < m; ++i) v[i] = xstep(v[i], v[i + m], h, m);
    }
    return v[0];
}

// Backward of raster_fwd_kernel for L = sum_px dot(d_image[px], C[px]):
// per (tile, splat) screen-space gradients, walking each tile's list back to
// front from its last blended entry. Each pixel recovers T before a splat
// from T after it (T_before = T_after / (1 - alpha)) and keeps the colour
// composited behind it. Each thread takes two pixels (kBwdPix): warp w of the
// kBwdThreads covers the 8x8 block of the 8x4 sub-tiles bwd_subtile(w, 0)
// and (w, 1). A splat's per-lane contributions, the two pixels' summed, are
// summed per warp with a transposed xor butterfly (lanes 2c end up holding
// component c), which then add them to the gaussian's record in screen_grads
// with one global float reduction per component (fire-and-forget REDs; a
// shared-memory float atomicAdd would be a CAS loop, and the warps that hit a
// splat are few). Two pixels per thread halve the butterflies and REDs per
// pixel. Each warp stages only the splats whose footprint reaches either of
// its sub-tiles (the list entry's mask) in slots of its own, with no block
// barrier, from the last entry its own pixels blended; a sub-tile's pixels
// skip the splats that miss it. With
// kGeo >= 1 the expected-depth channel (raster_fwd_kernel's acc_depth,
// background 0) is differentiated too, for `d_depth` per pixel: its share
// reaches the opacity and conic through alpha, and the splat centre's depth
// through `g_depth` (slot 10 of the butterfly). kGeo 2 adds the normal
// channel (acc_normal; `d_normal` 3 floats per pixel) the same way, the plane
// normals' share going to `g_normal` (slots 11-13).
//
// kLab adds the labels' cross-entropy (label_loss_kernel), its gradient
// d_lab = (alpha, beta) per pixel: d/dP_c = alpha - beta [c = y] of the
// class channels P_c = sum_i T_i a_i p_ic, y the pixel's label. As the p_i
// sum to 1 the alpha part reaches no logit (the softmax's backward cancels
// a gradient shared by every class), so a splat's class-y probability gets
// -T_i a_i beta into `g_logits` (sem_k per gaussian, before the softmax's
// backward), and through alpha the channels give
// T_i (alpha (1 - B) - beta (p_iy - B_y)), B and B_y the coverage and class-y
// channel composited behind the splat: with `lab_geo` 1 to the opacity and
// the centre, 2 to the conic too, 0 to neither. The refine weight sums the
// colour's and the labels' magnitudes, as separate passes would. Per warp
// the first two labels go through the butterfly (slots 14, 15); pixels of
// any other label add their own.
template <int kGeo, bool kLab>
__global__ void __launch_bounds__(kBwdThreads) raster_bwd_kernel(
    uint32_t img_w, uint32_t img_h, float3 bg, const uint2* __restrict__ ranges, const uint32_t* __restrict__ vals,
    const float2* __restrict__ xy, const float4* __restrict__ conic_op, const float4* __restrict__ color,
    const float* __restrict__ final_t, const uint32_t* __restrict__ last_idx, const float* __restrict__ d_image,
    float* __restrict__ screen_grads, const float* __restrict__ depth, const float* __restrict__ d_depth,
    float* __restrict__ g_depth, const float4* __restrict__ plane, const float* __restrict__ d_normal,
    float* __restrict__ g_normal, int sem_k, const float* __restrict__ probs, const uint8_t* __restrict__ labels,
    const float2* __restrict__ d_lab, int lab_geo, float* __restrict__ g_logits) {
    constexpr bool kZ = kGeo >= 1;
    constexpr bool kN = kGeo >= 2;
    __shared__ float2 s_xy[kBwdBatch];
    __shared__ float4 s_co[kBwdBatch];
    __shared__ float3 s_col[kBwdBatch];
    __shared__ uint32_t s_gid[kBwdBatch];
    __shared__ uint32_t s_cm[kBwdBatch];
    __shared__ float s_z[kZ ? kBwdBatch : 1];
    __shared__ float3 s_n[kN ? kBwdBatch : 1];
    __shared__ uint32_t s_j[kBwdBatch];  // the entry's index in the sorted list
    extern __shared__ float s_prob[];    // kLab: lab_stride(sem_k) per staged splat
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const uint2 range = ranges[tile];
    const float fw = float(img_w), fh = float(img_h);
    const int ks = lab_stride(sem_k);

    // Per pixel: its sub-tile, position, transmittance and gradients, and
    // what is composited behind the current splat.
    uint32_t st[kBwdPix];
    bool in_image[kBwdPix];
    float sx[kBwdPix], sy[kBwdPix];
    float t_cur[kBwdPix];
    uint32_t last[kBwdPix];
    float3 g[kBwdPix], behind[kBwdPix];
    float gz[kBwdPix], behind_z[kBwdPix];
    float3 gn[kBwdPix], behind_n[kBwdPix];
    uint32_t lab_col[kBwdPix];
    bool lab_on[kBwdPix];
    float2 gl[kBwdPix];  // (alpha, beta)
    float behind_all[kBwdPix], behind_y[kBwdPix];
    uint32_t last_max = 0;
#pragma unroll
    for (int q = 0; q < kBwdPix; ++q) {
        st[q] = bwd_subtile(warp, q);
        const uint32_t px = blockIdx.x * kTile + (st[q] & 1u) * 8u + uint32_t(lane & 7);
        const uint32_t py = blockIdx.y * kTile + (st[q] >> 1) * 4u + uint32_t(lane >> 3);
        in_image[q] = px < img_w && py < img_h;
        sx[q] = float(px) + 0.5f;
        sy[q] = float(py) + 0.5f;
        t_cur[q] = 1.f;
        last[q] = 0;
        g[q] = make_float3(0.f, 0.f, 0.f);
        behind[q] = bg;
        gz[q] = 0.f;
        behind_z[q] = 0.f;
        gn[q] = make_float3(0.f, 0.f, 0.f);
        behind_n[q] = make_float3(0.f, 0.f, 0.f);
        uint32_t lab = 0xffu;
        gl[q] = make_float2(0.f, 0.f);
        behind_all[q] = 0.f;
        behind_y[q] = 0.f;
        if (in_image[q]) {
            const uint32_t pix = py * img_w + px;
            t_cur[q] = final_t[pix];
            last[q] = last_idx[pix];
            g[q] = make_float3(d_image[pix * 3], d_image[pix * 3 + 1], d_image[pix * 3 + 2]);
            if (kZ) gz[q] = d_depth[pix];
            if (kN) gn[q] = make_float3(d_normal[pix * 3], d_normal[pix * 3 + 1], d_normal[pix * 3 + 2]);
            if (kLab) {
                lab = labels[pix];
                gl[q] = d_lab[pix];
            }
        }
        lab_on[q] = kLab && lab < uint32_t(sem_k) && gl[q].y != 0.f;
        lab_col[q] = lab_on[q] ? lab : 0u;
        last_max = max(last_max, last[q]);
    }
    // The warp's first two labels (c0, c1; -1: none) and each pixel's slot:
    // 0, 1 for those, 2 for another label, 3 for no gradient.
    int c0 = -1, c1 = -1;
    int slot[kBwdPix];
#pragma unroll
    for (int q = 0; q < kBwdPix; ++q) slot[q] = 3;
    if (kLab) {
        // The first lane (pixel 0 before pixel 1) whose label passes `on`.
        const auto first = [&](bool on0, bool on1) {
            const unsigned m0 = __ballot_sync(0xffffffffu, on0);
            const unsigned m1 = __ballot_sync(0xffffffffu, on1);
            const int a = __shfl_sync(0xffffffffu, int(lab_col[0]), m0 ? __ffs(m0) - 1 : 0);
            const int b = __shfl_sync(0xffffffffu, int(lab_col[1]), m1 ? __ffs(m1) - 1 : 0);
            return m0 ? a : (m1 ? b : -1);
        };
        c0 = first(lab_on[0], lab_on[1]);
        if (c0 >= 0) c1 = first(lab_on[0] && int(lab_col[0]) != c0, lab_on[1] && int(lab_col[1]) != c0);
#pragma unroll
        for (int q = 0; q < kBwdPix; ++q)
            if (lab_on[q]) slot[q] = int(lab_col[q]) == c0 ? 0 : (int(lab_col[q]) == c1 ? 1 : 2);
    }
    // The warp starts at the last entry any of its pixels blended.
    uint32_t stop = last_max;
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) stop = max(stop, __shfl_xor_sync(0xffffffffu, stop, m));
    stop = min(stop, range.y);
    // Each warp stages its own splats in its 32 slots, with no block barrier:
    // lane l reads entry bend - 1 - l, and the entries whose footprint reaches
    // either of the warp's sub-tiles are staged back to front.
    const int first = warp * 32;

    for (uint32_t bend = stop; bend > range.x;) {
        const uint32_t bstart = bend - range.x > 32u ? bend - 32u : range.x;
        const uint32_t e = bend - 1u - uint32_t(lane);
        uint32_t v = 0;
        bool keep = false;
        if (uint32_t(lane) < bend - bstart) {
            v = vals[e];
            const uint32_t cm = v >> kIdBits;
            keep = ((cm >> st[0]) & 1u) || ((cm >> st[1]) & 1u);
        }
        const unsigned kept = __ballot_sync(0xffffffffu, keep);
        // Every lane must be done with the previous batch before its slots are reused.
        __syncwarp();
        if (keep) {
            const int k = first + __popc(kept & ((1u << lane) - 1u));
            const uint32_t gi = v & kIdMask;
            s_xy[k] = xy[gi];
            s_co[k] = conic_op[gi];
            const float4 c = color[gi];
            s_col[k] = make_float3(c.x, c.y, c.z);
            s_gid[k] = gi;
            s_cm[k] = v >> kIdBits;
            s_j[k] = e;
            if (kZ) s_z[k] = depth[gi];
            if (kN) {
                const float4 pl = plane[gi];
                s_n[k] = make_float3(pl.x, pl.y, pl.z);
            }
            if (kLab)
                for (int c = 0; c < sem_k; ++c) s_prob[k * ks + c] = probs[size_t(gi) * sem_k + c];
        }
        __syncwarp();

        const int end = first + __popc(kept);
        for (int s = first; s < end; ++s) {
            const uint32_t cm = s_cm[s];
            const uint32_t j = s_j[s];
            float g_u = 0.f, g_v = 0.f, g_a = 0.f, g_b = 0.f, g_c = 0.f, g_o = 0.f, g_r = 0.f;
            float3 g_col = make_float3(0.f, 0.f, 0.f);
            float g_zs = 0.f;
            float3 g_ns = make_float3(0.f, 0.f, 0.f);
            float g_l0 = 0.f, g_l1 = 0.f;
            bool hit = false;
#pragma unroll
            for (int q = 0; q < kBwdPix; ++q) {
                // Warp-uniform: the footprint misses this sub-tile.
                if (!((cm >> st[q]) & 1u)) continue;
                if (!in_image[q] || j >= last[q]) continue;
                const float4 co = s_co[s];
                const float dx = sx[q] - s_xy[s].x;
                const float dy = sy[q] - s_xy[s].y;
                // Same expressions as the forward, so the same splats pass.
                const float sigma = 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
                if (sigma < 0.f) continue;
                const float e = expf(-sigma);
                const float raw = co.w * e;
                const float a = fminf(0.99f, raw);
                if (a < 1.f / 255.f) continue;
                hit = true;
                const float t_before = t_cur[q] / (1.f - a);
                const float3 c = s_col[s];
                const float3 col = make_float3(fmaxf(c.x, 0.f), fmaxf(c.y, 0.f), fmaxf(c.z, 0.f));
                const float w = a * t_before;
                g_col.x += w * g[q].x;
                g_col.y += w * g[q].y;
                g_col.z += w * g[q].z;
                float d_alpha = t_before * ((col.x - behind[q].x) * g[q].x + (col.y - behind[q].y) * g[q].y +
                                            (col.z - behind[q].z) * g[q].z);
                if (kZ) {
                    const float z = s_z[s];
                    g_zs += w * gz[q];
                    d_alpha += t_before * (z - behind_z[q]) * gz[q];
                    behind_z[q] = a * z + (1.f - a) * behind_z[q];
                }
                if (kN) {
                    const float3 nv = s_n[s];
                    g_ns.x += w * gn[q].x;
                    g_ns.y += w * gn[q].y;
                    g_ns.z += w * gn[q].z;
                    d_alpha += t_before * ((nv.x - behind_n[q].x) * gn[q].x + (nv.y - behind_n[q].y) * gn[q].y +
                                           (nv.z - behind_n[q].z) * gn[q].z);
                    behind_n[q].x = a * nv.x + (1.f - a) * behind_n[q].x;
                    behind_n[q].y = a * nv.y + (1.f - a) * behind_n[q].y;
                    behind_n[q].z = a * nv.z + (1.f - a) * behind_n[q].z;
                }
                float d_alpha_lab = 0.f;
                if (kLab) {
                    const float pz = s_prob[s * ks + lab_col[q]];
                    const float gy = -w * gl[q].y;
                    if (slot[q] == 0)
                        g_l0 += gy;
                    else if (slot[q] == 1)
                        g_l1 += gy;
                    else if (slot[q] == 2)
                        atomicAdd(&g_logits[size_t(s_gid[s]) * sem_k + lab_col[q]], gy);
                    if (lab_geo)
                        d_alpha_lab = t_before * (gl[q].x * (1.f - behind_all[q]) - gl[q].y * (pz - behind_y[q]));
                    behind_all[q] = a + (1.f - a) * behind_all[q];
                    behind_y[q] = a * pz + (1.f - a) * behind_y[q];
                }
                behind[q].x = a * col.x + (1.f - a) * behind[q].x;
                behind[q].y = a * col.y + (1.f - a) * behind[q].y;
                behind[q].z = a * col.z + (1.f - a) * behind[q].z;
                t_cur[q] = t_before;
                // A clamped alpha passes no gradient to opacity, conic or mean.
                if (raw <= 0.99f) {
                    const float d_alpha_all = kLab ? d_alpha + d_alpha_lab : d_alpha;
                    const float d_sigma = -d_alpha_all * a;
                    // The conic takes the labels' share with lab_geo 2 only.
                    const float d_sigma_c = kLab && lab_geo < 2 ? -d_alpha * a : d_sigma;
                    const float fu = -(co.x * dx + co.y * dy), fv = -(co.y * dx + co.z * dy);
                    const float pu = d_sigma * fu, pv = d_sigma * fv;
                    g_u += pu;
                    g_v += pv;
                    g_a += d_sigma_c * 0.5f * dx * dx;
                    g_b += d_sigma_c * dx * dy;
                    g_c += d_sigma_c * 0.5f * dy * dy;
                    g_o += d_alpha_all * e;
                    g_r += kLab ? (fabsf(d_alpha) + fabsf(d_alpha_lab)) * a * (fabsf(fu) * fw + fabsf(fv) * fh)
                                : fabsf(pu) * fw + fabsf(pv) * fh;
                }
            }
            // Skip the reduction when no lane of this warp blended the splat.
            if (__any_sync(0xffffffffu, hit)) {
                const bool h4 = (lane & 16) != 0;
                const float w0 = xstep(g_u, g_col.z, h4, 16);
                const float w1 = xstep(g_v, g_r, h4, 16);
                const float w2 = xstep(g_a, g_zs, h4, 16);
                const float w3 = xstep(g_b, g_ns.x, h4, 16);
                const float w4 = xstep(g_c, g_ns.y, h4, 16);
                const float w5 = xstep(g_o, g_ns.z, h4, 16);
                const float w6 = xstep(g_col.x, g_l0, h4, 16);
                const float w7 = xstep(g_col.y, g_l1, h4, 16);
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
                if ((lane & 1) == 0 && comp < kNumGrad && z != 0.f)
                    atomicAdd(&screen_grads[size_t(s_gid[s]) * kNumGrad + comp], z);
                if (kZ && (lane & 1) == 0 && comp == kNumGrad && z != 0.f) atomicAdd(&g_depth[s_gid[s]], z);
                if (kN && (lane & 1) == 0 && comp > kNumGrad && comp <= kNumGrad + 3 && z != 0.f)
                    atomicAdd(&g_normal[size_t(s_gid[s]) * 3 + (comp - kNumGrad - 1)], z);
                if (kLab && (lane & 1) == 0 && comp >= 14 && z != 0.f)
                    atomicAdd(&g_logits[size_t(s_gid[s]) * sem_k + (comp == 14 ? c0 : c1)], z);
            }
        }
        bend = bstart;
    }
}

// Backward of sem_fwd_kernel for L = sum_px dot(d_sem[px], S[px]), walking
// each tile back to front as raster_bwd_kernel does for the colour: each
// splat's values get T_i a_i d_sem (into g_logits, k per gaussian), and with
// `geometry` its share through alpha reaches screen_grads (du dv dA dB dC
// dopacity and the refine weight). A splat's per-lane values (7 screen
// components, then the k class values) are summed per warp by warp_sum32.
__global__ void __launch_bounds__(kBlock) sem_bwd_kernel(
    uint32_t img_w, uint32_t img_h, int k, const uint2* __restrict__ ranges, const uint32_t* __restrict__ vals,
    const float2* __restrict__ xy, const float4* __restrict__ conic_op, const float* __restrict__ logits,
    const float* __restrict__ final_t, const uint32_t* __restrict__ last_idx, const float* __restrict__ d_sem,
    int geometry, float* __restrict__ screen_grads, float* __restrict__ g_logits) {
    __shared__ float2 s_xy[kBwdBatch];
    __shared__ float4 s_co[kBwdBatch];
    __shared__ uint32_t s_gid[kBwdBatch];
    __shared__ uint32_t s_cm[kBwdBatch];
    __shared__ float s_z[kBwdBatch * kMaxSem];
    __shared__ uint32_t s_last;
    const uint32_t tile = blockIdx.y * gridDim.x + blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    uint32_t lx, ly;
    tile_pixel(tid, lx, ly);
    const uint32_t px = blockIdx.x * kTile + lx;
    const uint32_t py = blockIdx.y * kTile + ly;
    const bool in_image = px < img_w && py < img_h;
    const uint32_t pix = py * img_w + px;
    const float sx = float(px) + 0.5f, sy = float(py) + 0.5f;
    const uint2 range = ranges[tile];
    const float fw = float(img_w), fh = float(img_h);

    float t_cur = 1.f;
    uint32_t last = 0;
    float gs[kMaxSem], behind[kMaxSem];
#pragma unroll
    for (int c = 0; c < kMaxSem; ++c) {
        gs[c] = in_image && c < k ? d_sem[size_t(pix) * k + c] : 0.f;
        behind[c] = 0.f;
    }
    if (in_image) {
        t_cur = final_t[pix];
        last = last_idx[pix];
    }
    if (tid == 0) s_last = 0;
    __syncthreads();
    atomicMax(&s_last, last);
    __syncthreads();
    const uint32_t stop = min(s_last, range.y);

    for (uint32_t bend = stop; bend > range.x;) {
        const uint32_t bstart = bend - range.x > uint32_t(kBwdBatch) ? bend - kBwdBatch : range.x;
        const uint32_t cnt = bend - bstart;
        // Every warp must be done with the previous batch before its slots are reused.
        __syncthreads();
        if (uint32_t(tid) < cnt) {
            const uint32_t v = vals[bstart + tid];
            const uint32_t gi = v & kIdMask;
            s_xy[tid] = xy[gi];
            s_co[tid] = conic_op[gi];
            s_gid[tid] = gi;
            s_cm[tid] = v >> kIdBits;
            for (int c = 0; c < k; ++c) s_z[tid * kMaxSem + c] = logits[size_t(gi) * k + c];
        }
        __syncthreads();

        for (int s = int(cnt) - 1; s >= 0; --s) {
            // Warp-uniform: the footprint misses this warp's sub-tile.
            if (!((s_cm[s] >> warp) & 1u)) continue;
            const uint32_t j = bstart + uint32_t(s);
            float v[32];
#pragma unroll
            for (int c = 0; c < 32; ++c) v[c] = 0.f;
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
                        const float w = a * t_before;
                        float d_alpha = 0.f;
#pragma unroll
                        for (int c = 0; c < kMaxSem; ++c) {
                            if (c < k) {
                                const float z = s_z[s * kMaxSem + c];
                                v[7 + c] = w * gs[c];
                                d_alpha += (z - behind[c]) * gs[c];
                                behind[c] = a * z + (1.f - a) * behind[c];
                            }
                        }
                        d_alpha *= t_before;
                        t_cur = t_before;
                        // A clamped alpha passes no gradient to opacity, conic or mean.
                        if (geometry && raw <= 0.99f) {
                            const float d_sigma = -d_alpha * a;
                            v[0] = d_sigma * -(co.x * dx + co.y * dy);
                            v[1] = d_sigma * -(co.y * dx + co.z * dy);
                            v[2] = d_sigma * 0.5f * dx * dx;
                            v[3] = d_sigma * dx * dy;
                            v[4] = d_sigma * 0.5f * dy * dy;
                            v[5] = d_alpha * e;
                            v[6] = fabsf(v[0]) * fw + fabsf(v[1]) * fh;
                        }
                    }
                }
            }
            // Skip the reduction when no lane of this warp blended the splat.
            if (__any_sync(0xffffffffu, hit)) {
                const float z = warp_sum32(v, lane);
                const size_t gi = s_gid[s];
                if (z != 0.f) {
                    if (lane < 6)
                        atomicAdd(&screen_grads[gi * kNumGrad + lane], z);
                    else if (lane == 6)
                        atomicAdd(&screen_grads[gi * kNumGrad + 9], z);
                    else if (lane < 7 + k)
                        atomicAdd(&g_logits[gi * k + (lane - 7)], z);
                }
            }
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

// kLens: the camera has a lens (lens_kind != kLensPinhole); the pinhole
// instantiation compiles to the code before lenses.
template <bool kLens>
__global__ void project_bwd_kernel(int n, GscCamera cam, int rest_pc, int act, const float* __restrict__ transforms,
                                   const float* __restrict__ opacity, const float* __restrict__ sh,
                                   const uint32_t* __restrict__ tiles, const float* __restrict__ screen_grads,
                                   float* __restrict__ g_transforms, float* __restrict__ g_opacity,
                                   float* __restrict__ g_sh, int sh_zeroed, uint8_t* __restrict__ touched,
                                   const float* __restrict__ g_depth, const float* __restrict__ g_normal) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* sg = screen_grads + size_t(i) * kNumGrad;
    // Not blended into any pixel (or only where the loss gradient is 0):
    // every output gradient is 0, which the caller wrote, except the SH ones
    // when `sh_zeroed` is 0: then the sparse SH Adam skips this gaussian
    // (touched = 0) and never reads its stale row.
    bool any = false;
    float gn[3] = {0.f, 0.f, 0.f};
    if (tiles[i] != 0) {
#pragma unroll
        for (int k = 0; k < kNumGrad; ++k) any |= sg[k] != 0.f;
        if (g_depth) any |= g_depth[i] != 0.f;
        if (g_normal) {
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                gn[k] = g_normal[size_t(i) * 3 + k];
                any |= gn[k] != 0.f;
            }
        }
    }
    touched[i] = any;
    if (!any) return;
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

    // EWA Jacobian with the 1.3x frustum clamp (a fisheye's at the point).
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
    constexpr bool lens = kLens;
    float j0[3] = {cam.fx * inv_z, 0.f, -cam.fx * tx * inv_z2};
    float j1[3] = {0.f, cam.fy * inv_z, -cam.fy * ty * inv_z2};
    // With a lens: d j_r[k] / d p[m] and d(u, v) / d p, from the lens (past
    // max_theta its extension) on duals in p. The clamped coordinate of a
    // distorted pinhole's linearisation point q is clamp * q_z, so it follows
    // q_z alone (zero gradient on the clamped side, as without a lens).
    float dj0[3][3], dj1[3][3], du[3], dv[3];
    if constexpr (lens) {
        const Dual3 P[3] = {dual_var(p[0], 0), dual_var(p[1], 1), dual_var(p[2], 2)};
        Dual3 m[2], J[2][3], Q[3];
        lens_extended(cam, P, m, J, Q);
        if (cam.lens_kind == kLensDistorted) {
            const float inv_qz = 1.f / Q[2].v;
            const float qx = Q[0].v * inv_qz, qy = Q[1].v * inv_qz;
            const Dual3 L[3] = {qx < -lim_x || qx > lim_x ? Q[2] * clampf(qx, -lim_x, lim_x) : Q[0],
                                qy < -lim_y || qy > lim_y ? Q[2] * clampf(qy, -lim_y, lim_y) : Q[1], Q[2]};
            Dual3 ml[2];
            lens_coords(cam, L, ml, J);
        }
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            j0[k] = cam.fx * J[0][k].v;
            j1[k] = cam.fy * J[1][k].v;
            du[k] = cam.fx * m[0].d[k];
            dv[k] = cam.fy * m[1].d[k];
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                dj0[k][q] = cam.fx * J[0][k].d[q];
                dj1[k][q] = cam.fy * J[1][k].d[q];
            }
        }
    }
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
    if constexpr (lens) {
#pragma unroll
        for (int q = 0; q < 3; ++q) {
            float acc = g_u * du[q] + g_v * dv[q];
#pragma unroll
            for (int k = 0; k < 3; ++k) acc += gj0[k] * dj0[k][q] + gj1[k] * dj1[k][q];
            gp[q] = acc;
        }
    } else {
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
    }
    // The expected-depth channel blends the centres' depth: camera z, or
    // the distance for a fisheye.
    if constexpr (lens) {
        if (g_depth && cam.lens_kind == kLensEquidistant) {
            const float d = sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
#pragma unroll
            for (int q = 0; q < 3; ++q) gp[q] += g_depth[i] * p[q] / d;
        } else if (g_depth) {
            gp[2] += g_depth[i];
        }
    } else {
        if (g_depth) gp[2] += g_depth[i];
    }

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
    // The normal channel: n = +-W Rg[:, k] for the smallest scale k, turned to
    // face the camera (plane_kernel's choice and flip, both piecewise
    // constant), so dL/dRg[:, k] = +-W^T gn.
    if (g_normal) {
        int k = 0;
        if (tr[8] < tr[7 + k]) k = 1;
        if (tr[9] < tr[7 + k]) k = 2;
        const float axis[3] = {rg[k], rg[3 + k], rg[6 + k]};
        float nc[3];
        mat_vec(W, axis, nc);
        const float sign = dot3(nc, p) > 0.f ? -1.f : 1.f;
#pragma unroll
        for (int r = 0; r < 3; ++r) gr[r * 3 + k] += sign * (W[r] * gn[0] + W[3 + r] * gn[1] + W[6 + r] * gn[2]);
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
            if (k <= act)
                grest[k - 1] = gc * b[k];
            else if (!sh_zeroed && k <= rest_pc)
                grest[k - 1] = 0.f;
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


// Per projected gaussian, its plane in camera space for the planar median
// depth: the normal is the axis of its smallest scale, turned to face the
// camera (as project_geo.wgsl), and w = n . p_cam. A ray (x, y, 1) meets
// the plane at depth w / (n . (x, y, 1)).
__global__ void plane_kernel(int n, GscCamera cam, const float* __restrict__ transforms,
                             const uint32_t* __restrict__ tiles, float4* __restrict__ plane) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || tiles[i] == 0) return;
    const float* tr = transforms + size_t(i) * 10;
    const float* W = cam.rot;
    float p[3];
    mat_vec(W, tr, p);
    p[0] += cam.t[0];
    p[1] += cam.t[1];
    p[2] += cam.t[2];
    float R[9];
    quat_to_rot(tr[3], tr[4], tr[5], tr[6], R);
    int k = 0;
    if (tr[8] < tr[7 + k]) k = 1;
    if (tr[9] < tr[7 + k]) k = 2;
    const float axis[3] = {R[k], R[3 + k], R[6 + k]};
    float nc[3];
    mat_vec(W, axis, nc);
    if (dot3(nc, p) > 0.f) {
        nc[0] = -nc[0];
        nc[1] = -nc[1];
        nc[2] = -nc[2];
    }
    plane[i] = make_float4(nc[0], nc[1], nc[2], dot3(nc, p));
}

// ---------------------------------------------------------------------------
// Training kernels: the wgpu trainer's loss_l1, ssim, densify_stats, adam,
// mean_noise and densify_scatter shaders (crates/gsplat-train/src/shaders).

__device__ inline float gt_channel(uint32_t packed, int c) { return float((packed >> (8 * c)) & 0xFFu) / 255.f; }
__device__ inline bool gt_valid(uint32_t packed) { return (packed >> 24) >= 128u; }
__device__ inline float signf(float x) { return float(x > 0.f) - float(x < 0.f); }

// Sum `v` over the block (256 threads) and add it to `*acc` once.
__device__ inline void block_add(float v, double* acc) {
    __shared__ float warp_sums[kBlock / 32];
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        float s = threadIdx.x < kBlock / 32 ? warp_sums[threadIdx.x] : 0.f;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) s += __shfl_xor_sync(0xffffffffu, s, m);
        if (threadIdx.x == 0) atomicAdd(acc, double(s));
    }
}

// d_image = weight * sign(render - gt) / (3 npix): the gradient of
// weight * mean |render - gt|. Ground truth is packed RGBA8 (R in the low
// byte); alpha below 128 masks the pixel out. acc += the unweighted mean L1.
__global__ void __launch_bounds__(kBlock) loss_l1_kernel(uint32_t npix, const uint32_t* __restrict__ gt,
                                                         const float* __restrict__ render, float weight,
                                                         float* __restrict__ d_image, double* __restrict__ acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float l = 0.f;
    if (i < npix) {
        const uint32_t packed = gt[i];
        const float scale = (gt_valid(packed) ? 1.f : 0.f) / (3.f * float(npix));
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            const float diff = render[i * 3 + c] - gt_channel(packed, c);
            d_image[i * 3 + c] = signf(diff) * weight * scale;
            l += fabsf(diff) * scale;
        }
    }
    block_add(l, acc);
}

// Depth prior: d_depth = weight * sign(D - P) / (P npix) on the pixels with a
// prior P > 0 (e.g. visloc-mvs's fused depth), D the expected depth
// (raster_fwd_kernel's acc_depth): the gradient of weight * mean |D - P| / P
// over all pixels, a relative L1 so near and far surfaces weigh alike.
// acc += the unweighted mean.
__global__ void __launch_bounds__(kBlock) depth_loss_kernel(uint32_t npix, const float* __restrict__ prior,
                                                            const float* __restrict__ acc_depth, float weight,
                                                            float* __restrict__ d_depth, double* __restrict__ acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float l = 0.f;
    if (i < npix) {
        const float p = prior[i];
        float d = 0.f;
        if (p > 0.f) {
            const float diff = acc_depth[i] - p;
            d = weight * signf(diff) / (p * float(npix));
            l = fabsf(diff) / (p * float(npix));
        }
        d_depth[i] = d;
    }
    block_add(l, acc);
}

// Each gaussian's class probabilities, the softmax of its k logits.
__global__ void softmax_kernel(int n, int k, const float* __restrict__ logits, float* __restrict__ probs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* z = logits + size_t(i) * k;
    float* p = probs + size_t(i) * k;
    float top = z[0];
    for (int c = 1; c < k; ++c) top = fmaxf(top, z[c]);
    float sum = 0.f;
    for (int c = 0; c < k; ++c) {
        p[c] = expf(z[c] - top);
        sum += p[c];
    }
    for (int c = 0; c < k; ++c) p[c] /= sum;
}

// The probabilities' gradient g (k per gaussian) through the softmax, in
// place: dL/dz_c = p_c (g_c - sum_d g_d p_d).
__global__ void softmax_bwd_kernel(int n, int k, const float* __restrict__ probs, float* __restrict__ g) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = probs + size_t(i) * k;
    float* gi = g + size_t(i) * k;
    float dot = 0.f;
    for (int c = 0; c < k; ++c) dot += gi[c] * p[c];
    for (int c = 0; c < k; ++c) gi[c] = p[c] * (gi[c] - dot);
}

// Labels: per pixel with a label y below k, P_y its class channel
// (raster_fwd_kernel's acc_lab) and A = sum_c P_c = 1 - T its coverage (the
// probabilities sum to 1), the cross-entropy -log(P_y / A) of the normalised
// blend, over the `count` such pixels. Its gradient, weight times that of the
// mean, is d/dP_c = alpha - beta [c = y] with alpha = weight / (count A) and
// beta = weight / (count P_y), into d_lab as (alpha, beta); zeros where there
// is no label or (almost) no splat. acc += the unweighted mean.
__global__ void __launch_bounds__(kBlock) label_loss_kernel(uint32_t npix, int k, const uint8_t* __restrict__ labels,
                                                            const float* __restrict__ acc_lab,
                                                            const float* __restrict__ final_t, float weight,
                                                            float inv_count, float2* __restrict__ d_lab,
                                                            double* __restrict__ acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float l = 0.f;
    if (i < npix) {
        float2 d = make_float2(0.f, 0.f);
        const float cover = 1.f - final_t[i];
        if (labels[i] < uint32_t(k) && cover > 1e-4f) {
            const float py = fmaxf(acc_lab[i], 1e-12f);
            d = make_float2(weight * inv_count / cover, weight * inv_count / py);
            l = (logf(cover) - logf(py)) * inv_count;
        }
        d_lab[i] = d;
    }
    block_add(l, acc);
}

// Depth-normal consistency (2DGS / PGSR style), normal_loss.wgsl's two passes.
// Per pixel the render gives N = sum_i w_i n_i (acc_normal), D = sum_i w_i z_i
// (acc_depth) and alpha = 1 - T. With P(q) = D_q / alpha_q r(q),
// r(q) = ((qx + 0.5 - cx) / fx, (qy + 0.5 - cy) / fy, 1) (through a lens its
// pixel ray, scaled to its depth: see NormalLossArgs::rays), the depth map's
// normal is n_d = s c / |c|, c = (P(x+1) - P(x-1)) x (P(y+1) - P(y-1)), s = +-1
// facing the camera, and L = weight / npix sum_p (1 - N_p . n_d(p)) over the
// pixels whose own and 4-neighbour opacities exceed 0.5. The gradient reaches
// N and D; alpha is held constant.
struct NormalLossArgs {
    uint32_t w, h;
    float fx, fy, cx, cy;
    float coef;  // weight / npix
    // Through a lens, r(q) is its pixel ray (camera_rays) and D the lens's
    // depth (z, or a fisheye's distance); null for a pinhole.
    const float3* rays;
};

__device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ inline float3 operator+(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator*(float s, float3 a) { return f3(s * a.x, s * a.y, s * a.z); }
__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(float3 a, float3 b) {
    return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float3 normal_ray(const NormalLossArgs& a, uint32_t x, uint32_t y) {
    if (a.rays) return a.rays[y * a.w + x];
    return f3((float(x) + 0.5f - a.cx) / a.fx, (float(y) + 0.5f - a.cy) / a.fy, 1.f);
}

// Pass 1: dL/dN into d_normal and, per pixel, dL/da and dL/db (a, b the two
// central differences) into `gab` (6 floats per pixel). acc += the
// unweighted mean of 1 - N . n_d.
__global__ void __launch_bounds__(kBlock) normal_grad_kernel(NormalLossArgs a, const float* __restrict__ acc_normal,
                                                             const float* __restrict__ acc_depth,
                                                             const float* __restrict__ final_t,
                                                             float* __restrict__ d_normal, float* __restrict__ gab,
                                                             double* __restrict__ acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t npix = a.w * a.h;
    float l = 0.f;
    if (i < npix) {
        const uint32_t x = i % a.w, y = i / a.w;
        float3 gn = f3(0.f, 0.f, 0.f), ga = gn, gb = gn;
        auto alpha = [&](uint32_t j) { return 1.f - final_t[j]; };
        auto point = [&](uint32_t px, uint32_t py) {
            const uint32_t j = py * a.w + px;
            return (acc_depth[j] / fmaxf(alpha(j), 1e-6f)) * normal_ray(a, px, py);
        };
        if (x > 0 && y > 0 && x + 1 < a.w && y + 1 < a.h && alpha(i) > 0.5f && alpha(i - 1) > 0.5f &&
            alpha(i + 1) > 0.5f && alpha(i - a.w) > 0.5f && alpha(i + a.w) > 0.5f) {
            const float3 da = point(x + 1, y) - point(x - 1, y);
            const float3 db = point(x, y + 1) - point(x, y - 1);
            const float3 c = cross(da, db);
            const float cl = sqrtf(dot(c, c));
            if (cl > 1e-12f) {
                const float3 nd = (1.f / cl) * c;
                const float s = dot(nd, point(x, y)) > 0.f ? -1.f : 1.f;
                const float3 big_n = f3(acc_normal[i * 3], acc_normal[i * 3 + 1], acc_normal[i * 3 + 2]);
                l = (1.f - s * dot(big_n, nd)) / float(npix);
                // L = coef (1 - N . s n_d)
                gn = (-a.coef * s) * nd;
                const float3 g_nd = (-a.coef * s) * big_n;
                const float3 gc = (1.f / cl) * (g_nd - dot(nd, g_nd) * nd);
                ga = cross(db, gc);
                gb = cross(gc, da);
            }
        }
        d_normal[i * 3 + 0] = gn.x;
        d_normal[i * 3 + 1] = gn.y;
        d_normal[i * 3 + 2] = gn.z;
        float* o = gab + size_t(i) * 6;
        o[0] = ga.x;
        o[1] = ga.y;
        o[2] = ga.z;
        o[3] = gb.x;
        o[4] = gb.y;
        o[5] = gb.z;
    }
    block_add(l, acc);
}

// Pass 2: gather dL/dP(q) from the four pixels whose differences use q and
// add dL/dD_q to d_depth.
__global__ void __launch_bounds__(kBlock) normal_depth_grad_kernel(NormalLossArgs a, const float* __restrict__ final_t,
                                                                   const float* __restrict__ gab,
                                                                   float* __restrict__ d_depth) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= a.w * a.h) return;
    const uint32_t x = i % a.w, y = i / a.w;
    // a(p) = P(p + x) - P(p - x): q is the +x term of p = q - x and the -x
    // term of p = q + x (likewise for b along y).
    auto ga = [&](uint32_t j) { return f3(gab[size_t(j) * 6], gab[size_t(j) * 6 + 1], gab[size_t(j) * 6 + 2]); };
    auto gb = [&](uint32_t j) { return f3(gab[size_t(j) * 6 + 3], gab[size_t(j) * 6 + 4], gab[size_t(j) * 6 + 5]); };
    float3 gp = f3(0.f, 0.f, 0.f);
    if (x > 0) gp = gp + ga(i - 1);
    if (x + 1 < a.w) gp = gp - ga(i + 1);
    if (y > 0) gp = gp + gb(i - a.w);
    if (y + 1 < a.h) gp = gp - gb(i + a.w);
    const float alpha = fmaxf(1.f - final_t[i], 1e-6f);
    d_depth[i] += dot(gp, normal_ray(a, x, y)) / alpha;
}

// D-SSIM: 11x11 gaussian window (sigma 1.5), zero padding, C1 = 0.01^2,
// C2 = 0.03^2, per channel; see ssim.wgsl for the derivation. One block per
// 16x16 tile with its 26x26 apron in shared memory.
constexpr int kSsimR = 5;
constexpr int kSsimA = kTile + 2 * kSsimR;  // 26
__constant__ float kGauss[11] = {0.0010284f, 0.0075988f, 0.0360008f, 0.1093607f, 0.2130055f, 0.2660117f,
                                 0.2130055f, 0.1093607f, 0.0360008f, 0.0075988f, 0.0010284f};

__device__ inline int apron_pixel(int tx, int ty, int ax, int ay, uint32_t w, uint32_t h) {
    const int gx = tx * kTile + ax - kSsimR;
    const int gy = ty * kTile + ay - kSsimR;
    if (gx < 0 || gy < 0 || gx >= int(w) || gy >= int(h)) return -1;
    return gy * int(w) + gx;
}

// SSIM and, per pixel and channel, A = dS/dmx, B = dS/dSxx, C = dS/dSxy.
__global__ void __launch_bounds__(kBlock) ssim_fwd_kernel(uint32_t w, uint32_t h, const float* __restrict__ render,
                                                          const uint32_t* __restrict__ gt, float* __restrict__ abc,
                                                          double* __restrict__ acc) {
    __shared__ float in0[kSsimA * kSsimA];
    __shared__ float in1[kSsimA * kSsimA];
    __shared__ float hq[kSsimA * kTile * 5];
    const int tx = blockIdx.x, ty = blockIdx.y;
    const int t = threadIdx.x;
    const int ox = t % kTile, oy = t / kTile;
    const uint32_t px = tx * kTile + ox, py = ty * kTile + oy;
    const bool inside = px < w && py < h;
    const bool valid = inside && gt_valid(gt[py * w + px]);
    const float C1 = 0.0001f, C2 = 0.0009f;
    float s_sum = 0.f;
    for (int c = 0; c < 3; ++c) {
        for (int i = t; i < kSsimA * kSsimA; i += kBlock) {
            const int p = apron_pixel(tx, ty, i % kSsimA, i / kSsimA, w, h);
            in0[i] = p >= 0 ? render[p * 3 + c] : 0.f;
            in1[i] = p >= 0 ? gt_channel(gt[p], c) : 0.f;
        }
        __syncthreads();
        for (int i = t; i < kSsimA * kTile; i += kBlock) {
            const int row = i / kTile, col = i % kTile;
            float mx = 0.f, my = 0.f, sxx = 0.f, syy = 0.f, sxy = 0.f;
#pragma unroll
            for (int k = 0; k < 11; ++k) {
                const float g = kGauss[k];
                const int j = row * kSsimA + col + k;
                const float xv = in0[j], yv = in1[j];
                mx += g * xv;
                my += g * yv;
                sxx += g * xv * xv;
                syy += g * yv * yv;
                sxy += g * xv * yv;
            }
            hq[i * 5 + 0] = mx;
            hq[i * 5 + 1] = my;
            hq[i * 5 + 2] = sxx;
            hq[i * 5 + 3] = syy;
            hq[i * 5 + 4] = sxy;
        }
        __syncthreads();
        float mx = 0.f, my = 0.f, sxx = 0.f, syy = 0.f, sxy = 0.f;
#pragma unroll
        for (int k = 0; k < 11; ++k) {
            const float g = kGauss[k];
            const int j = ((oy + k) * kTile + ox) * 5;
            mx += g * hq[j];
            my += g * hq[j + 1];
            sxx += g * hq[j + 2];
            syy += g * hq[j + 3];
            sxy += g * hq[j + 4];
        }
        if (inside) {
            const uint32_t i = py * w + px;
            float ga = 0.f, gb = 0.f, gc = 0.f;
            if (valid) {
                const float vx = sxx - mx * mx;
                const float vy = syy - my * my;
                const float cxy = sxy - mx * my;
                const float a = 2.f * mx * my + C1;
                const float b = 2.f * cxy + C2;
                const float cc = mx * mx + my * my + C1;
                const float d = vx + vy + C2;
                const float s = a * b / (cc * d);
                s_sum += s;
                ga = (2.f * my * b - 2.f * my * a) / (cc * d) - s * (2.f * mx / cc - 2.f * mx / d);
                gb = -s / d;
                gc = 2.f * a / (cc * d);
            }
            abc[i * 9 + c * 3 + 0] = ga;
            abc[i * 9 + c * 3 + 1] = gb;
            abc[i * 9 + c * 3 + 2] = gc;
        }
        // in0 / in1 / hq are reused by the next channel.
        __syncthreads();
    }
    block_add(s_sum, acc);
}

// d_image -= scale * dS/dx with dS/dx = (G*A) + 2 x (G*B) + y (G*C).
__global__ void __launch_bounds__(kBlock) ssim_bwd_kernel(uint32_t w, uint32_t h, float scale,
                                                          const float* __restrict__ render,
                                                          const uint32_t* __restrict__ gt,
                                                          const float* __restrict__ abc,
                                                          float* __restrict__ d_image) {
    __shared__ float in0[kSsimA * kSsimA];
    __shared__ float in1[kSsimA * kSsimA];
    __shared__ float in2[kSsimA * kSsimA];
    __shared__ float hq[kSsimA * kTile * 3];
    const int tx = blockIdx.x, ty = blockIdx.y;
    const int t = threadIdx.x;
    const int ox = t % kTile, oy = t / kTile;
    const uint32_t px = tx * kTile + ox, py = ty * kTile + oy;
    const bool inside = px < w && py < h;
    const bool valid = inside && gt_valid(gt[py * w + px]);
    for (int c = 0; c < 3; ++c) {
        for (int i = t; i < kSsimA * kSsimA; i += kBlock) {
            const int p = apron_pixel(tx, ty, i % kSsimA, i / kSsimA, w, h);
            in0[i] = p >= 0 ? abc[p * 9 + c * 3] : 0.f;
            in1[i] = p >= 0 ? abc[p * 9 + c * 3 + 1] : 0.f;
            in2[i] = p >= 0 ? abc[p * 9 + c * 3 + 2] : 0.f;
        }
        __syncthreads();
        for (int i = t; i < kSsimA * kTile; i += kBlock) {
            const int row = i / kTile, col = i % kTile;
            float ga = 0.f, gb = 0.f, gc = 0.f;
#pragma unroll
            for (int k = 0; k < 11; ++k) {
                const float g = kGauss[k];
                const int j = row * kSsimA + col + k;
                ga += g * in0[j];
                gb += g * in1[j];
                gc += g * in2[j];
            }
            hq[i * 3 + 0] = ga;
            hq[i * 3 + 1] = gb;
            hq[i * 3 + 2] = gc;
        }
        __syncthreads();
        float ga = 0.f, gb = 0.f, gc = 0.f;
#pragma unroll
        for (int k = 0; k < 11; ++k) {
            const float g = kGauss[k];
            const int j = ((oy + k) * kTile + ox) * 3;
            ga += g * hq[j];
            gb += g * hq[j + 1];
            gc += g * hq[j + 2];
        }
        if (valid) {
            const uint32_t i = py * w + px;
            const float xv = render[i * 3 + c];
            const float yv = gt_channel(gt[i], c);
            d_image[i * 3 + c] -= scale * (ga + 2.f * xv * gb + yv * gc);
        }
        __syncthreads();
    }
}

// brush refine statistics: per gaussian blended into at least one pixel
// this frame (a nonzero opacity or colour gradient), the max of its refine
// weight and the number of such frames.
__global__ void stats_kernel(int n, const uint32_t* __restrict__ tiles, const float* __restrict__ screen_grads,
                             float* __restrict__ weight_max, float* __restrict__ seen) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || tiles[i] == 0) return;
    const float* sg = screen_grads + size_t(i) * kNumGrad;
    if (sg[5] == 0.f && sg[6] == 0.f && sg[7] == 0.f && sg[8] == 0.f) return;
    weight_max[i] = fmaxf(weight_max[i], sg[9]);
    seen[i] += 1.f;
}

// PCG-style integer hash, uniform and normal samples (mean_noise.wgsl,
// densify_scatter.wgsl).
__device__ inline uint32_t hash_u32(uint32_t x) {
    const uint32_t v = x * 747796405u + 2891336453u;
    const uint32_t w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}
__device__ inline float uniform01(uint32_t key) { return (float(hash_u32(key) >> 8u) + 0.5f) / 16777216.f; }
__device__ inline float normal_sample(uint32_t key) {
    const float u1 = uniform01(key * 2u + 1u);
    const float u2 = uniform01(key * 2u + 2u);
    return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}

// Adam moments as float, or as bf16 (uint16_t, a float's top 16 bits). A
// bf16 store rounds up with the probability of the dropped bits (`r`, 16
// random bits), so small updates still move the moment on average.
__device__ inline float load_moment(const float* m, size_t e) { return m[e]; }
__device__ inline float load_moment(const uint16_t* m, size_t e) { return __uint_as_float(uint32_t(m[e]) << 16); }
__device__ inline void store_moment(float* m, size_t e, float v, uint32_t) { m[e] = v; }
__device__ inline void store_moment(uint16_t* m, size_t e, float v, uint32_t r) {
    m[e] = uint16_t((__float_as_uint(v) + (r & 0xffffu)) >> 16);
}

// Adam over one parameter buffer of `stride`-float records: element
// k = e % stride uses lr_a for k < split_a, lr_b for k < split_b, else lr_c.
// brush's auxiliary losses ride along: aux_kind 1 adds
// aux_coef * v * exp(p) to the log-scale elements (k >= 7), aux_kind 2 adds
// aux_coef * v to every element (an opacity logit), with
// v = 1e-3 + (1 if the gaussian was projected this frame). With
// `sparse_touched`, gaussians whose flag is 0 are left as they are. `seed`
// varies the bf16 moments' rounding from step to step.
template <class M>
__global__ void adam_kernel(size_t n_elem, uint32_t stride, uint32_t split_a, uint32_t split_b, float lr_a,
                            float lr_b, float lr_c, float beta1, float beta2, float eps, float bc1, float bc2,
                            int aux_kind, float aux_coef, const uint8_t* __restrict__ sparse_touched,
                            const uint32_t* __restrict__ tiles,
                            float* __restrict__ params, const float* __restrict__ grads, M* __restrict__ m1,
                            M* __restrict__ m2, uint32_t seed) {
    const size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;
    // Sparse: gaussians the frame did not blend keep their values and moments.
    if (sparse_touched && !sparse_touched[e / stride]) return;
    const uint32_t k = uint32_t(e % stride);
    const float lr = k < split_a ? lr_a : (k < split_b ? lr_b : lr_c);
    const float p = params[e];
    float g = grads[e];
    if (aux_kind != 0) {
        const float v = 1e-3f + (tiles[e / stride] > 0 ? 1.f : 0.f);
        if (aux_kind == 1) {
            if (k >= 7) g += aux_coef * v * expf(p);
        } else {
            g += aux_coef * v;
        }
    }
    const float m = beta1 * load_moment(m1, e) + (1.f - beta1) * g;
    const float vv = beta2 * load_moment(m2, e) + (1.f - beta2) * g * g;
    const uint32_t r = std::is_same<M, float>::value ? 0u : hash_u32(uint32_t(e) ^ seed);
    store_moment(m1, e, m, r);
    store_moment(m2, e, vv, r >> 16);
    params[e] = p - lr * (m / bc1) / (sqrtf(vv / bc2) + eps);
}

// The moments converted when a step asks for the other precision (bf16
// rounded to nearest).
__global__ void moments_to_bf16_kernel(size_t n, const float* __restrict__ src, uint16_t* __restrict__ dst) {
    const size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e < n) dst[e] = uint16_t((__float_as_uint(src[e]) + 0x8000u) >> 16);
}
__global__ void moments_to_float_kernel(size_t n, const uint16_t* __restrict__ src, float* __restrict__ dst) {
    const size_t e = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e < n) dst[e] = __uint_as_float(uint32_t(src[e]) << 16);
}

// Rotate v by the normalised quaternion (w, x, y, z).
__device__ inline void quat_rotate(const float* q, const float* v, float* out) {
    const float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    float w = 1.f, x = 0.f, y = 0.f, z = 0.f;
    if (n > 1e-12f) {
        w = q[0] / n;
        x = q[1] / n;
        y = q[2] / n;
        z = q[3] / n;
    }
    // t = 2 u x v; out = v + w t + u x t
    const float t0 = 2.f * (y * v[2] - z * v[1]);
    const float t1 = 2.f * (z * v[0] - x * v[2]);
    const float t2 = 2.f * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * t0 + (y * t2 - z * t1);
    out[1] = v[1] + w * t1 + (z * t0 - x * t2);
    out[2] = v[2] + w * t2 + (x * t1 - y * t0);
}

// brush mean noise: nearly transparent gaussians wander,
// mean += clamp(R (N(0, 1) * s) * w, +-max_noise),
// w = clamp((1 - opacity)^100, 0, 1) * scale.
__global__ void noise_kernel(int n, uint32_t seed, float scale, float max_noise, float* __restrict__ transforms,
                             const float* __restrict__ opacity) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = 1.f / (1.f + expf(-opacity[i]));
    const float w = fminf(fmaxf(powf(1.f - a, 100.f), 0.f), 1.f) * scale;
    if (w <= 0.f) return;
    float* t = transforms + size_t(i) * 10;
    const uint32_t key = (seed ^ (uint32_t(i) * 3u)) * 3u;
    const float v[3] = {normal_sample(key) * expf(t[7]), normal_sample(key + 1u) * expf(t[8]),
                        normal_sample(key + 2u) * expf(t[9])};
    float d[3];
    quat_rotate(t + 3, v, d);
#pragma unroll
    for (int k = 0; k < 3; ++k) t[k] += clampf(d[k] * w, -max_noise, max_noise);
}

__global__ void action_counts_kernel(int n, const uint32_t* __restrict__ actions, uint32_t* __restrict__ counts) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t a = actions[i];
    counts[i] = a == 0 ? 0u : (a >= 2 ? 2u : 1u);
}

// Copy one parameter group's rows and Adam moments to the new buffers per
// action: 0 prune, 1 keep, 2 clone (keep + a fresh copy), 4 brush split
// (the parent shifted by -o and a fresh copy shifted by +o, o = R (N(0, 0.5)
// * s), both with scales / sqrt 2 and opacity 1 - sqrt(1 - a)). Row 0 keeps
// the parent's moments, row 1 starts at zero. kind: 0 transforms,
// 1 opacity, 2 SH. The moments are float or bf16 (M, as adam_kernel's).
template <class M>
__global__ void scatter_kernel(int n, uint32_t stride, int kind, uint32_t seed, const uint32_t* __restrict__ actions,
                               const uint32_t* __restrict__ cum, const float* __restrict__ src_p,
                               const M* __restrict__ src_m1, const M* __restrict__ src_m2,
                               float* __restrict__ dst_p, M* __restrict__ dst_m1, M* __restrict__ dst_m2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t action = actions[i];
    if (action == 0) return;
    const uint32_t count = action >= 2 ? 2u : 1u;
    const size_t out0 = cum[i] - count;
    const size_t src = size_t(i) * stride;

    if (kind == 0 && action == 4) {
        const float* p = src_p + src;
        const uint32_t key = (seed ^ (uint32_t(i) * 6u)) * 3u;
        const float v[3] = {normal_sample(key) * 0.5f * expf(p[7]), normal_sample(key + 1u) * 0.5f * expf(p[8]),
                            normal_sample(key + 2u) * 0.5f * expf(p[9])};
        float o[3];
        quat_rotate(p + 3, v, o);
        const float shrink = logf(sqrtf(2.f));
        for (uint32_t c = 0; c < 2; ++c) {
            const size_t d = (out0 + c) * stride;
            const float sgn = c == 1 ? 1.f : -1.f;
            for (int k = 0; k < 3; ++k) dst_p[d + k] = p[k] + sgn * o[k];
            for (int k = 3; k < 7; ++k) dst_p[d + k] = p[k];
            for (int k = 7; k < 10; ++k) dst_p[d + k] = p[k] - shrink;
            for (int k = 0; k < 10; ++k) {
                dst_m1[d + k] = c == 1 ? M(0) : src_m1[src + k];
                dst_m2[d + k] = c == 1 ? M(0) : src_m2[src + k];
            }
        }
        return;
    }
    if (kind == 1 && action == 4) {
        const float a = 1.f / (1.f + expf(-src_p[src]));
        const float na = clampf(1.f - sqrtf(fmaxf(1.f - a, 0.f)), 1e-7f, 1.f - 1e-7f);
        const float logit = logf(na / (1.f - na));
        for (uint32_t c = 0; c < 2; ++c) {
            dst_p[out0 + c] = logit;
            dst_m1[out0 + c] = c == 1 ? M(0) : src_m1[src];
            dst_m2[out0 + c] = c == 1 ? M(0) : src_m2[src];
        }
        return;
    }
    for (uint32_t c = 0; c < count; ++c) {
        const size_t d = (out0 + c) * stride;
        for (uint32_t k = 0; k < stride; ++k) {
            dst_p[d + k] = src_p[src + k];
            dst_m1[d + k] = c == 1 ? M(0) : src_m1[src + k];
            dst_m2[d + k] = c == 1 ? M(0) : src_m2[src + k];
        }
    }
}

// ---------------------------------------------------------------------------
// Host side.

// Device buffer that only grows.
struct Buf {
    void* ptr = nullptr;
    size_t cap = 0;
};

enum Stage {
    kProject,
    kScan,
    kDuplicate,
    kSort,
    kRanges,
    kRaster,
    kLoss,
    kRasterBwd,
    kProjectBwd,
    kOptimizer,
    kSemantic,
    kSemanticBwd,
    kNumStages,
};
constexpr const char* kStageNames =
    "project,scan,duplicate,sort,ranges,raster,loss,raster_bwd,project_bwd,optimizer,semantic,semantic_bwd";

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
    // Scene and its Adam moments, per group (transforms, opacity, SH). The
    // moments are bf16 (uint16_t) when `m_bf16`, else float; the semantic
    // logits' moments stay float.
    Buf params[3], m1[3], m2[3];
    bool m_bf16 = false;
    uint32_t adam_steps = 0;  // seeds the bf16 moments' rounding
    // Per gaussian, per frame.
    Buf xy, conic_op, color, depth, rect, extk, tiles, offsets;
    // Intersections.
    Buf keys, vals, keys_sorted, vals_sorted, sort_temp, scan_temp;
    // Per tile / pixel.
    Buf ranges, image, final_t, last_idx, depth_out, plane;
    // Backward.
    Buf d_image, screen_grads, grads[3], touched;
    // Training: ground truth (packed RGBA8 per pixel, image-major), SSIM
    // partials, loss sums (L1, SSIM), refine statistics, refine scratch.
    Buf gt, abc, loss_acc, weight_max, seen, actions, counts, cum;
    // Depth prior: per image and pixel (0 = none), the expected depth of the
    // last render, its gradient, and per gaussian the centre-depth gradient.
    Buf depth_prior, acc_depth, d_depth, g_depth;
    // Depth-normal loss: the normal channel of the last render, its gradient,
    // the loss's per-pixel scratch, and per gaussian the normal gradient.
    Buf acc_normal, d_normal, gab, g_normal;
    bool have_normal = false;  // the last render has the depth and normal channels
    // Per pixel the lens's ray (lens_pixel_ray) for the planar depth and the
    // depth-normal loss through a lens, and the camera it was made for.
    Buf rays;
    GscCamera rays_cam = {};
    bool rays_valid = false;
    // GPU TSDF (tsdf.cuh): the block table, the current slab's blocks and
    // voxels, the extraction output, and on the host the surface of the last
    // fusion until visloc_gsplat_cuda_tsdf_take.
    Buf tsdf_table, tsdf_keys, tsdf_slab_table, tsdf_slab_keys, tsdf_sdf, tsdf_w, tsdf_col, tsdf_counts,
        tsdf_out_cells, tsdf_out_pos, tsdf_out_col, tsdf_out_edges, tsdf_cache_depth, tsdf_cache_rgb, tsdf_override;
    std::vector<int32_t> tsdf_cells, tsdf_edges;
    std::vector<float> tsdf_pos;
    std::vector<uint8_t> tsdf_col_host;
    // Matching (match.cuh): the descriptors (128 bytes each) and their
    // squared norms, each image's first row, and the top-2 outputs.
    Buf match_words, match_norms, match_fwd, match_rev, match_partial;
    std::vector<uint32_t> match_first;
    // Label votes (vote_kernel): the current view's labels, and per gaussian
    // and class the summed blending weight, for `vote_n` gaussians and
    // `vote_k` classes (0: none yet).
    Buf labels, votes;
    int vote_n = 0, vote_k = 0;
    // Semantic training: `sem_k` classes (0: none), per gaussian their
    // logits, Adam moments, probabilities and gradient; per uploaded image its
    // labels (`sem_images` of them) and labelled pixels; per pixel the last
    // render's class channels and their gradient (render_classes), and a
    // step's label channel and its (alpha, beta) gradient (label_loss_kernel).
    Buf sem, sem_m1, sem_m2, sem_prob, sem_grad, sem_labels, acc_sem, d_sem, acc_lab, d_lab;
    int sem_k = 0, sem_images = 0;
    std::vector<uint32_t> sem_counts;
    int num_priors = 0;
    int num_images = 0;
    uint32_t gt_w = 0, gt_h = 0;
    uint32_t loss_steps = 0;
    cudaEvent_t ev_start[kNumStages] = {}, ev_end[kNumStages] = {};
    bool ran[kNumStages] = {};
    float ms[kNumStages] = {};

    int rest_pc() const { return (sh_degree + 1) * (sh_degree + 1) - 1; }
    size_t row_len() const { return size_t(3) * (sh_degree + 1) * (sh_degree + 1); }
    size_t stride(int group) const { return group == 0 ? 10 : (group == 1 ? 1 : row_len()); }
    size_t pixels() const { return size_t(cam.width) * cam.height; }
    float* p(int group) const { return static_cast<float*>(params[group].ptr); }
};

#define GSC_CHECK(ctx, expr)                                                      \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            (ctx)->error = std::string(#expr) + ": " + cudaGetErrorString(err__); \
            return -1;                                                            \
        }                                                                         \
    } while (0)

#define GSC_TRY(expr)               \
    do {                            \
        if ((expr) != 0) return -1; \
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

// A buffer of exactly `count` zeros (contents not preserved).
template <class T>
int ensure_zero(Ctx* ctx, Buf& b, size_t count, T** out) {
    GSC_TRY(ensure(ctx, b, count, out));
    if (count > 0) GSC_CHECK(ctx, cudaMemsetAsync(*out, 0, count * sizeof(T)));
    return 0;
}

void free_buf(Buf& b) {
    if (b.ptr) cudaFree(b.ptr);
    b.ptr = nullptr;
    b.cap = 0;
}

inline unsigned blocks_for(size_t n, unsigned per = 256) { return unsigned((n + per - 1) / per); }

// `count` zeroed Adam moments for group k, in the context's precision.
int zero_moments(Ctx* ctx, int k, size_t count) {
    if (ctx->m_bf16) {
        uint16_t *a, *b;
        GSC_TRY(ensure_zero(ctx, ctx->m1[k], count, &a));
        GSC_TRY(ensure_zero(ctx, ctx->m2[k], count, &b));
    } else {
        float *a, *b;
        GSC_TRY(ensure_zero(ctx, ctx->m1[k], count, &a));
        GSC_TRY(ensure_zero(ctx, ctx->m2[k], count, &b));
    }
    return 0;
}

// Convert every group's moments to bf16 or back to float.
int convert_moments(Ctx* ctx, bool to_bf16) {
    if (ctx->m_bf16 == to_bf16) return 0;
    for (int k = 0; k < 3; ++k) {
        const size_t count = size_t(ctx->n) * ctx->stride(k);
        for (Buf* m : {&ctx->m1[k], &ctx->m2[k]}) {
            Buf out;
            if (to_bf16) {
                uint16_t* dst;
                GSC_TRY(ensure(ctx, out, count, &dst));
                if (count > 0)
                    moments_to_bf16_kernel<<<blocks_for(count), 256>>>(count, static_cast<const float*>(m->ptr), dst);
            } else {
                float* dst;
                GSC_TRY(ensure(ctx, out, count, &dst));
                if (count > 0)
                    moments_to_float_kernel<<<blocks_for(count), 256>>>(count, static_cast<const uint16_t*>(m->ptr),
                                                                        dst);
            }
            GSC_CHECK(ctx, cudaGetLastError());
            GSC_CHECK(ctx, cudaDeviceSynchronize());
            free_buf(*m);
            *m = out;
        }
    }
    ctx->m_bf16 = to_bf16;
    return 0;
}

int stage_begin(Ctx* ctx, Stage s) {
    GSC_CHECK(ctx, cudaEventRecord(ctx->ev_start[s]));
    ctx->ran[s] = true;
    return 0;
}
int stage_end(Ctx* ctx, Stage s) {
    GSC_CHECK(ctx, cudaEventRecord(ctx->ev_end[s]));
    return 0;
}
void clear_stages(Ctx* ctx) {
    for (int s = 0; s < kNumStages; ++s) {
        ctx->ran[s] = false;
        ctx->ms[s] = 0.f;
    }
}
// Wait for every recorded stage (the enum's order is not the run's: the
// semantic stages come between others) and read each one's time.
int collect_stages(Ctx* ctx) {
    for (int s = 0; s < kNumStages; ++s)
        if (ctx->ran[s]) GSC_CHECK(ctx, cudaEventSynchronize(ctx->ev_end[s]));
    for (int s = 0; s < kNumStages; ++s)
        if (ctx->ran[s]) GSC_CHECK(ctx, cudaEventElapsedTime(&ctx->ms[s], ctx->ev_start[s], ctx->ev_end[s]));
    return 0;
}

__global__ void ray_lut_kernel(GscCamera cam, float3* __restrict__ rays) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cam.width * cam.height) return;
    const uint32_t x = i % cam.width, y = i / cam.width;
    rays[i] = lens_pixel_ray(cam, float(x) + 0.5f, float(y) + 0.5f);
}

// Whether two cameras share their intrinsics, size and lens (their pixel
// rays).
bool same_rays(const GscCamera& a, const GscCamera& b) {
    if (a.width != b.width || a.height != b.height || a.fx != b.fx || a.fy != b.fy || a.cx != b.cx ||
        a.cy != b.cy || a.lens_kind != b.lens_kind)
        return false;
    for (int i = 0; i < 8; ++i)
        if (a.lens[i] != b.lens[i]) return false;
    return true;
}

// ctx->cam's pixel rays through its lens, made on the device and kept while
// the camera's intrinsics stay (training cycles through views of a few
// cameras); null for an undistorted pinhole, whose kernels compute its rays.
int camera_rays(Ctx* ctx, const float3** out) {
    *out = nullptr;
    if (ctx->cam.lens_kind == kLensPinhole) return 0;
    float3* rays;
    const void* held = ctx->rays.ptr;
    GSC_TRY(ensure(ctx, ctx->rays, ctx->pixels(), &rays));
    if (ctx->rays.ptr != held) ctx->rays_valid = false;
    if (!ctx->rays_valid || !same_rays(ctx->rays_cam, ctx->cam)) {
        ray_lut_kernel<<<blocks_for(uint32_t(ctx->pixels())), 256>>>(ctx->cam, rays);
        GSC_CHECK(ctx, cudaGetLastError());
        ctx->rays_cam = ctx->cam;
        ctx->rays_valid = true;
    }
    *out = rays;
    return 0;
}

// Render the uploaded scene from ctx->cam; with `out_depth` (device, 2 floats
// per pixel) also the median depth and opacity, at the splat planes when
// `planar`, and (nullable) the expected-depth and normal channels. With
// `labels` (device, a label per pixel) also the label channel into acc_lab,
// from the class probabilities in sem_prob.
int run_forward(Ctx* ctx, float* out_depth = nullptr, bool planar = false, float* acc_depth = nullptr,
                float* acc_normal = nullptr, const uint8_t* labels = nullptr) {
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

    GSC_TRY(stage_begin(ctx, kProject));
    if (n > 0) {
        project_kernel<<<blocks_for(n), 256>>>(n, cam, ctx->rest_pc(), ctx->act, ctx->p(0), ctx->p(1), ctx->p(2), xy,
                                               conic_op, color, depth, rect, extk, tiles);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_TRY(stage_end(ctx, kProject));
    GSC_TRY(stage_begin(ctx, kScan));
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
    GSC_TRY(stage_end(ctx, kScan));

    unsigned long long *keys, *keys_sorted;
    uint32_t *vals, *vals_sorted;
    GSC_TRY(ensure(ctx, ctx->keys, num_isect, &keys));
    GSC_TRY(ensure(ctx, ctx->vals, num_isect, &vals));
    GSC_TRY(ensure(ctx, ctx->keys_sorted, num_isect, &keys_sorted));
    GSC_TRY(ensure(ctx, ctx->vals_sorted, num_isect, &vals_sorted));
    GSC_TRY(stage_begin(ctx, kDuplicate));
    if (num_isect > 0) {
        duplicate_kernel<<<blocks_for(n), 256>>>(n, cam, ctx->tile_bw, xy, conic_op, depth, rect, extk, tiles,
                                                 offsets, keys, vals);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_TRY(stage_end(ctx, kDuplicate));
    GSC_TRY(stage_begin(ctx, kSort));
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
    GSC_TRY(stage_end(ctx, kSort));
    GSC_TRY(stage_begin(ctx, kRanges));
    GSC_CHECK(ctx, cudaMemsetAsync(ranges, 0, num_tiles * sizeof(uint2)));
    if (num_isect > 0) {
        ranges_kernel<<<blocks_for(num_isect), 256>>>(num_isect, keys_sorted, ranges);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_TRY(stage_end(ctx, kRanges));
    GSC_TRY(stage_begin(ctx, kRaster));
    const dim3 grid(ctx->tile_bw, ctx->tile_bh);
    const float4 intr = make_float4(cam.fx, cam.fy, cam.cx, cam.cy);
    float4* plane = nullptr;
    const float3* rays = nullptr;
    if (out_depth && (planar || acc_normal) && n > 0) {
        GSC_TRY(ensure(ctx, ctx->plane, n, &plane));
        plane_kernel<<<blocks_for(n), 256>>>(n, cam, ctx->p(0), tiles, plane);
        GSC_CHECK(ctx, cudaGetLastError());
        if (planar) GSC_TRY(camera_rays(ctx, &rays));
    }
    float* acc_lab = nullptr;
    if (labels) GSC_TRY(ensure(ctx, ctx->acc_lab, pixels, &acc_lab));
    const auto* probs = static_cast<const float*>(ctx->sem_prob.ptr);
    const size_t smem = labels ? size_t(kBlock) * lab_stride(ctx->sem_k) * sizeof(float) : 0;
    const auto raster = [&](auto depth_c, auto lab_c) {
        constexpr bool kDepth = decltype(depth_c)::value;
        raster_fwd_kernel<kDepth, decltype(lab_c)::value><<<grid, kBlock, smem>>>(
            cam.width, cam.height, ctx->bg, ranges, vals_sorted, xy, conic_op, color, depth, kDepth ? plane : nullptr,
            kDepth && planar && plane, intr, kDepth ? rays : nullptr, image, final_t, last_idx, out_depth, acc_depth,
            acc_normal, ctx->sem_k, probs, labels, acc_lab);
    };
    if (out_depth)
        labels ? raster(std::true_type{}, std::true_type{}) : raster(std::true_type{}, std::false_type{});
    else
        labels ? raster(std::false_type{}, std::true_type{}) : raster(std::false_type{}, std::false_type{});
    GSC_CHECK(ctx, cudaGetLastError());
    GSC_TRY(stage_end(ctx, kRaster));
    ctx->have_forward = true;
    ctx->have_normal = out_depth && acc_depth && acc_normal;
    return 0;
}

// Backward of the last forward for the d_image already on the device, into
// screen_grads and the per-group gradient buffers (zeroed first; with
// `zero_sh` false the SH gradients are written for every projected gaussian
// instead, and left stale for the others). `geo`: 1 also differentiates the
// expected-depth channel (d_depth), 2 the depth and normal channels (d_depth,
// d_normal; the render must have them). With `sem.channels` the class
// channels' d_sem (render_classes) is differentiated too (sem_bwd_kernel,
// after the colour; `rgb` false leaves the colour out), with `sem.labels_ce`
// the labels' cross-entropy, d_lab (raster_bwd_kernel with kLab; the forward
// rendered the label channel of `sem.labels`). Either reaches the logits
// through the softmax (softmax_bwd_kernel, into sem_grad).
struct SemBackward {
    bool channels = false;
    bool labels_ce = false;
    const uint8_t* labels = nullptr;  // labels_ce: the pixels' labels
    // Alpha's share reaches the gaussians: 0 not, 1 their opacities and
    // centres (labels_ce only; the class channels take 1 as 2), 2 their
    // shapes too.
    int geometry = 2;
};
int run_backward(Ctx* ctx, bool zero_sh, int geo = 0, SemBackward sem = {}, bool rgb = true) {
    if (!ctx->have_forward) {
        ctx->error = "backward called before render";
        return -1;
    }
    if (geo >= 2 && !ctx->have_normal) {
        ctx->error = "backward: the last render has no normal channel";
        return -1;
    }
    const int n = ctx->n;
    float* screen_grads;
    GSC_TRY(ensure_zero(ctx, ctx->screen_grads, size_t(n) * kNumGrad, &screen_grads));
    float* g[3];
    for (int k = 0; k < 2; ++k) GSC_TRY(ensure_zero(ctx, ctx->grads[k], size_t(n) * ctx->stride(k), &g[k]));
    uint8_t* touched;
    GSC_TRY(ensure(ctx, ctx->touched, n, &touched));
    if (zero_sh)
        GSC_TRY(ensure_zero(ctx, ctx->grads[2], size_t(n) * ctx->stride(2), &g[2]));
    else
        GSC_TRY(ensure(ctx, ctx->grads[2], size_t(n) * ctx->stride(2), &g[2]));

    float* g_depth = nullptr;
    if (geo >= 1) GSC_TRY(ensure_zero(ctx, ctx->g_depth, n, &g_depth));
    float* g_normal = nullptr;
    if (geo >= 2) GSC_TRY(ensure_zero(ctx, ctx->g_normal, size_t(n) * 3, &g_normal));
    float* g_logits = nullptr;
    if (sem.channels || sem.labels_ce)
        GSC_TRY(ensure_zero(ctx, ctx->sem_grad, size_t(n) * ctx->sem_k, &g_logits));

    GSC_TRY(stage_begin(ctx, kRasterBwd));
    const dim3 grid(ctx->tile_bw, ctx->tile_bh);
    const auto* ranges = static_cast<const uint2*>(ctx->ranges.ptr);
    const auto* vals = static_cast<const uint32_t*>(ctx->vals_sorted.ptr);
    const auto* xy = static_cast<const float2*>(ctx->xy.ptr);
    const auto* conic_op = static_cast<const float4*>(ctx->conic_op.ptr);
    const auto* color = static_cast<const float4*>(ctx->color.ptr);
    const auto* final_t = static_cast<const float*>(ctx->final_t.ptr);
    const auto* last_idx = static_cast<const uint32_t*>(ctx->last_idx.ptr);
    const auto* d_image = static_cast<const float*>(ctx->d_image.ptr);
    const auto* depth = static_cast<const float*>(ctx->depth.ptr);
    const auto* d_depth = static_cast<const float*>(ctx->d_depth.ptr);
    const auto* probs = static_cast<const float*>(ctx->sem_prob.ptr);
    const size_t smem = sem.labels_ce ? size_t(kBwdBatch) * lab_stride(ctx->sem_k) * sizeof(float) : 0;
    const auto raster = [&](auto geo_c, auto lab_c) {
        constexpr int kGeo = decltype(geo_c)::value;
        raster_bwd_kernel<kGeo, decltype(lab_c)::value><<<grid, kBwdThreads, smem>>>(
            ctx->cam.width, ctx->cam.height, ctx->bg, ranges, vals, xy, conic_op, color, final_t, last_idx, d_image,
            screen_grads, kGeo >= 1 ? depth : nullptr, kGeo >= 1 ? d_depth : nullptr, g_depth,
            kGeo >= 2 ? static_cast<const float4*>(ctx->plane.ptr) : nullptr,
            kGeo >= 2 ? static_cast<const float*>(ctx->d_normal.ptr) : nullptr, g_normal, ctx->sem_k, probs,
            sem.labels, static_cast<const float2*>(ctx->d_lab.ptr), sem.geometry, g_logits);
    };
    const auto with_labels = [&](auto geo_c) {
        sem.labels_ce ? raster(geo_c, std::true_type{}) : raster(geo_c, std::false_type{});
    };
    if (!rgb) {
        // Nothing from the colour.
    } else if (geo >= 2)
        with_labels(std::integral_constant<int, 2>{});
    else if (geo == 1)
        with_labels(std::integral_constant<int, 1>{});
    else
        with_labels(std::integral_constant<int, 0>{});
    GSC_CHECK(ctx, cudaGetLastError());
    GSC_TRY(stage_end(ctx, kRasterBwd));
    if (g_logits) {
        GSC_TRY(stage_begin(ctx, kSemanticBwd));
        if (sem.channels) {
            sem_bwd_kernel<<<grid, kBlock>>>(ctx->cam.width, ctx->cam.height, ctx->sem_k, ranges, vals, xy, conic_op,
                                             probs, final_t, last_idx, static_cast<const float*>(ctx->d_sem.ptr),
                                             sem.geometry != 0 ? 1 : 0, screen_grads, g_logits);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        if (n > 0) {
            softmax_bwd_kernel<<<blocks_for(n), 256>>>(n, ctx->sem_k, probs, g_logits);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        GSC_TRY(stage_end(ctx, kSemanticBwd));
    }
    GSC_TRY(stage_begin(ctx, kProjectBwd));
    if (n > 0) {
        const auto* tiles = static_cast<const uint32_t*>(ctx->tiles.ptr);
        if (ctx->cam.lens_kind != kLensPinhole)
            project_bwd_kernel<true><<<blocks_for(n), 256>>>(n, ctx->cam, ctx->rest_pc(), ctx->act, ctx->p(0),
                                                             ctx->p(1), ctx->p(2), tiles, screen_grads, g[0], g[1],
                                                             g[2], zero_sh ? 1 : 0, touched, g_depth, g_normal);
        else
            project_bwd_kernel<false><<<blocks_for(n), 256>>>(n, ctx->cam, ctx->rest_pc(), ctx->act, ctx->p(0),
                                                              ctx->p(1), ctx->p(2), tiles, screen_grads, g[0], g[1],
                                                              g[2], zero_sh ? 1 : 0, touched, g_depth, g_normal);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_TRY(stage_end(ctx, kProjectBwd));
    return 0;
}

int set_camera(Ctx* ctx, const GscCamera* cam, const float* bg, int sh_active_degree) {
    if (!cam || !bg || cam->width == 0 || cam->height == 0) {
        ctx->error = "camera and background are required, image size must be positive";
        return -1;
    }
    if (sh_active_degree < 0 || sh_active_degree > ctx->sh_degree) {
        ctx->error = "active SH degree must be in 0..=scene degree";
        return -1;
    }
    ctx->cam = *cam;
    ctx->bg = make_float3(bg[0], bg[1], bg[2]);
    ctx->act = (sh_active_degree + 1) * (sh_active_degree + 1) - 1;
    return 0;
}

constexpr int kNumLossSums = 5;  // L1, SSIM, depth prior, depth-normal, labels

// The depth-normal loss of the last render (which has the depth and normal
// channels): its gradient into d_normal, added to d_depth (written by the
// caller), and its unweighted mean added to `acc`.
int run_normal_loss(Ctx* ctx, float weight, double* acc) {
    const uint32_t npix = uint32_t(ctx->pixels());
    float *d_normal, *gab;
    GSC_TRY(ensure(ctx, ctx->d_normal, size_t(npix) * 3, &d_normal));
    GSC_TRY(ensure(ctx, ctx->gab, size_t(npix) * 6, &gab));
    const GscCamera& cam = ctx->cam;
    const float3* rays;
    GSC_TRY(camera_rays(ctx, &rays));
    const NormalLossArgs a = {cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, weight / float(npix), rays};
    const auto* final_t = static_cast<const float*>(ctx->final_t.ptr);
    normal_grad_kernel<<<blocks_for(npix), kBlock>>>(a, static_cast<const float*>(ctx->acc_normal.ptr),
                                                     static_cast<const float*>(ctx->acc_depth.ptr), final_t,
                                                     d_normal, gab, acc);
    GSC_CHECK(ctx, cudaGetLastError());
    normal_depth_grad_kernel<<<blocks_for(npix), kBlock>>>(a, final_t, gab, static_cast<float*>(ctx->d_depth.ptr));
    GSC_CHECK(ctx, cudaGetLastError());
    return 0;
}

// The labels of uploaded image `image` on the device.
const uint8_t* image_labels(const Ctx* ctx, uint32_t image) {
    return static_cast<const uint8_t*>(ctx->sem_labels.ptr) + size_t(image) * ctx->pixels();
}

// The labels' cross-entropy of the last render (which has the label channel
// of uploaded image `image`, see run_forward) times `weight`
// (label_loss_kernel): its gradient into d_lab, the unweighted mean added to
// `acc`.
int run_label_loss(Ctx* ctx, uint32_t image, float weight, double* acc) {
    const uint32_t npix = uint32_t(ctx->pixels());
    float2* d_lab;
    GSC_TRY(ensure(ctx, ctx->d_lab, npix, &d_lab));
    const uint32_t count = ctx->sem_counts[image];
    label_loss_kernel<<<blocks_for(npix), kBlock>>>(npix, ctx->sem_k, image_labels(ctx, image),
                                                    static_cast<const float*>(ctx->acc_lab.ptr),
                                                    static_cast<const float*>(ctx->final_t.ptr), weight,
                                                    count > 0 ? 1.f / float(count) : 0.f, d_lab, acc);
    GSC_CHECK(ctx, cudaGetLastError());
    return 0;
}

// (1 - w) L1 + w (1 - SSIM) of the last render against uploaded image
// `image`: its gradient into d_image, and the unweighted mean L1 and the
// SSIM sum added to loss_acc. With `depth_weight` the depth prior and with
// `normal_weight` the depth-normal loss too (their gradients into d_depth and
// d_normal; the render must have the channels), and with `label_weight` the
// labels' cross-entropy (run_label_loss; the render has the label channel).
int run_loss(Ctx* ctx, uint32_t image, float ssim_weight, float depth_weight = 0.f, float normal_weight = 0.f,
             float label_weight = 0.f) {
    if (!ctx->have_forward) {
        ctx->error = "loss: no render to compare";
        return -1;
    }
    if (image >= uint32_t(ctx->num_images) || ctx->cam.width != ctx->gt_w || ctx->cam.height != ctx->gt_h) {
        ctx->error = "loss: image index or size does not match the uploaded images";
        return -1;
    }
    const uint32_t npix = uint32_t(ctx->pixels());
    const uint32_t* gt = static_cast<const uint32_t*>(ctx->gt.ptr) + size_t(image) * npix;
    const float* render = static_cast<const float*>(ctx->image.ptr);
    float* d_image;
    double* acc;
    GSC_TRY(ensure(ctx, ctx->d_image, size_t(npix) * 3, &d_image));
    // One more slot for visloc_gsplat_cuda_normal_loss.
    GSC_TRY(ensure(ctx, ctx->loss_acc, kNumLossSums + 1, &acc));
    if (ctx->loss_steps == 0) GSC_CHECK(ctx, cudaMemsetAsync(acc, 0, kNumLossSums * sizeof(double)));
    GSC_TRY(stage_begin(ctx, kLoss));
    loss_l1_kernel<<<blocks_for(npix), kBlock>>>(npix, gt, render, 1.f - ssim_weight, d_image, acc);
    GSC_CHECK(ctx, cudaGetLastError());
    if (ssim_weight > 0.f) {
        float* abc;
        GSC_TRY(ensure(ctx, ctx->abc, size_t(npix) * 9, &abc));
        const dim3 grid(ctx->tile_bw, ctx->tile_bh);
        ssim_fwd_kernel<<<grid, kBlock>>>(ctx->cam.width, ctx->cam.height, render, gt, abc, acc + 1);
        GSC_CHECK(ctx, cudaGetLastError());
        ssim_bwd_kernel<<<grid, kBlock>>>(ctx->cam.width, ctx->cam.height, ssim_weight / (3.f * float(npix)), render,
                                          gt, abc, d_image);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    if (depth_weight > 0.f) {
        float* d_depth;
        GSC_TRY(ensure(ctx, ctx->d_depth, npix, &d_depth));
        const float* prior = static_cast<const float*>(ctx->depth_prior.ptr) + size_t(image) * npix;
        depth_loss_kernel<<<blocks_for(npix), kBlock>>>(npix, prior, static_cast<const float*>(ctx->acc_depth.ptr),
                                                        depth_weight, d_depth, acc + 2);
        GSC_CHECK(ctx, cudaGetLastError());
    } else if (normal_weight > 0.f) {
        float* d_depth;
        GSC_TRY(ensure_zero(ctx, ctx->d_depth, npix, &d_depth));
    }
    if (normal_weight > 0.f) {
        if (!ctx->have_normal) {
            ctx->error = "loss: the depth-normal loss needs a render with the normal channel";
            return -1;
        }
        GSC_TRY(run_normal_loss(ctx, normal_weight, acc + 3));
    }
    if (label_weight > 0.f) GSC_TRY(run_label_loss(ctx, image, label_weight, acc + 4));
    GSC_TRY(stage_end(ctx, kLoss));
    ctx->loss_steps++;
    return 0;
}

// Zero logits and Adam moments for the current scene's gaussians.
int reset_logits(Ctx* ctx) {
    const size_t count = size_t(ctx->n) * ctx->sem_k;
    float *z, *a, *b;
    GSC_TRY(ensure_zero(ctx, ctx->sem, count, &z));
    GSC_TRY(ensure_zero(ctx, ctx->sem_m1, count, &a));
    GSC_TRY(ensure_zero(ctx, ctx->sem_m2, count, &b));
    return 0;
}

// The gaussians' class probabilities, the softmax of their logits
// (softmax_kernel, into sem_prob).
int run_softmax(Ctx* ctx) {
    float* probs;
    GSC_TRY(ensure(ctx, ctx->sem_prob, size_t(ctx->n) * ctx->sem_k, &probs));
    if (ctx->n > 0) {
        softmax_kernel<<<blocks_for(ctx->n), 256>>>(ctx->n, ctx->sem_k, static_cast<const float*>(ctx->sem.ptr),
                                                    probs);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    return 0;
}

// The gaussians' class probabilities and their channels for the last
// forward, at every pixel, into acc_sem (sem_fwd_kernel).
int run_sem_forward(Ctx* ctx) {
    if (!ctx->have_forward || ctx->sem_k == 0) {
        ctx->error = "semantic forward: needs a render and logits";
        return -1;
    }
    float* acc_sem;
    GSC_TRY(ensure(ctx, ctx->acc_sem, ctx->pixels() * ctx->sem_k, &acc_sem));
    GSC_TRY(run_softmax(ctx));
    if (ctx->n > 0) {
        const dim3 grid(ctx->tile_bw, ctx->tile_bh);
        sem_fwd_kernel<<<grid, kBlock>>>(ctx->cam.width, ctx->cam.height, ctx->sem_k,
                                         static_cast<const uint2*>(ctx->ranges.ptr),
                                         static_cast<const uint32_t*>(ctx->vals_sorted.ptr),
                                         static_cast<const float2*>(ctx->xy.ptr),
                                         static_cast<const float4*>(ctx->conic_op.ptr),
                                         static_cast<const float*>(ctx->sem_prob.ptr), acc_sem);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    return 0;
}

}  // namespace

extern "C" {

// One optimisation step (see visloc_gsplat_cuda_train_step). Mirrored by
// `GscStepParams` in crates/gsplat-cuda/src/lib.rs.
struct GscStepParams {
    GscCamera cam;
    uint32_t image;  // index into the uploaded ground-truth images
    int32_t sh_active_degree;
    float bg[3];
    float ssim_weight;  // loss = (1 - w) L1 + w (1 - SSIM)
    float lr[6];        // mean, quaternion, log-scale, opacity logit, SH DC, SH rest
    float beta1, beta2, eps;
    float bc1, bc2;            // Adam bias corrections 1 - beta^t
    float scale_aux_coef;      // brush aux loss on log-scales (0 = off)
    float opacity_aux_coef;    // brush aux loss on opacity logits (0 = off)
    uint32_t accumulate_stats; // brush refine statistics
    uint32_t noise_seed;
    float noise_scale;  // lr_mean * mean noise weight (0 = off)
    float noise_max;
    uint32_t sparse_sh_adam;   // SH Adam only on the gaussians blended this frame
    float depth_weight;        // depth-prior loss (set_depth_priors; 0 = off)
    float normal_weight;       // depth-normal loss (0 = off)
    float semantic_weight;     // labels' cross-entropy (set_labels; 0 = off)
    float lr_semantic;         // learning rate of the logits
    uint32_t semantic_geometry; // the labels also move: 0 only the logits, 1 opacities and centres, 2 shapes too
    uint32_t moments_bf16;      // keep the Adam moments in bf16 (converted when this changes)
};

#if defined(_WIN32)
#define GSC_EXPORT __declspec(dllexport)
#else
#define GSC_EXPORT __attribute__((visibility("default")))
#endif

GSC_EXPORT uint32_t visloc_gsplat_cuda_abi_version() { return kAbiVersion; }

GSC_EXPORT void* visloc_gsplat_cuda_create() {
    Ctx* ctx = new (std::nothrow) Ctx();
    if (!ctx) return nullptr;
    for (int s = 0; s < kNumStages; ++s) {
        if (cudaEventCreate(&ctx->ev_start[s]) != cudaSuccess || cudaEventCreate(&ctx->ev_end[s]) != cudaSuccess) {
            ctx->error = "cudaEventCreate failed (no CUDA device?)";
            break;
        }
    }
    return ctx;
}

GSC_EXPORT void visloc_gsplat_cuda_destroy(void* handle) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return;
    for (int k = 0; k < 3; ++k) {
        free_buf(ctx->params[k]);
        free_buf(ctx->m1[k]);
        free_buf(ctx->m2[k]);
        free_buf(ctx->grads[k]);
    }
    Buf* bufs[] = {&ctx->xy,          &ctx->conic_op,  &ctx->color,    &ctx->depth,      &ctx->rect,
                   &ctx->extk,        &ctx->tiles,     &ctx->offsets,  &ctx->keys,       &ctx->vals,
                   &ctx->keys_sorted, &ctx->vals_sorted, &ctx->sort_temp, &ctx->scan_temp, &ctx->ranges,
                   &ctx->image,       &ctx->final_t,   &ctx->last_idx, &ctx->depth_out, &ctx->plane, &ctx->d_image,    &ctx->screen_grads, &ctx->touched, &ctx->depth_prior, &ctx->acc_depth, &ctx->d_depth, &ctx->g_depth,
                   &ctx->acc_normal,  &ctx->d_normal,  &ctx->gab,      &ctx->g_normal,
                   &ctx->tsdf_table,  &ctx->tsdf_keys, &ctx->tsdf_slab_table, &ctx->tsdf_slab_keys,
                   &ctx->tsdf_sdf,    &ctx->tsdf_w,    &ctx->tsdf_col, &ctx->tsdf_counts,
                   &ctx->match_words, &ctx->match_norms, &ctx->match_fwd, &ctx->match_rev, &ctx->match_partial,
                   &ctx->tsdf_out_cells, &ctx->tsdf_out_pos, &ctx->tsdf_out_col, &ctx->tsdf_out_edges, &ctx->tsdf_cache_depth, &ctx->tsdf_cache_rgb, &ctx->tsdf_override,
                   &ctx->gt,          &ctx->abc,       &ctx->loss_acc, &ctx->weight_max, &ctx->seen,
                   &ctx->actions,     &ctx->counts,    &ctx->cum,      &ctx->labels,     &ctx->votes,
                   &ctx->sem,         &ctx->sem_m1,    &ctx->sem_m2,   &ctx->sem_grad,   &ctx->sem_labels,
                   &ctx->acc_sem,     &ctx->d_sem, &ctx->sem_prob, &ctx->acc_lab, &ctx->d_lab};
    for (Buf* b : bufs) free_buf(*b);
    for (int s = 0; s < kNumStages; ++s) {
        if (ctx->ev_start[s]) cudaEventDestroy(ctx->ev_start[s]);
        if (ctx->ev_end[s]) cudaEventDestroy(ctx->ev_end[s]);
    }
    delete ctx;
}

GSC_EXPORT const char* visloc_gsplat_cuda_last_error(void* handle) {
    return handle ? static_cast<Ctx*>(handle)->error.c_str() : "null context";
}

GSC_EXPORT const char* visloc_gsplat_cuda_stage_names() { return kStageNames; }

// Upload a packed scene (host pointers), replacing the current one; the Adam
// moments and refine statistics start at zero. `sh` holds
// 3 * (degree + 1)^2 floats per gaussian.
GSC_EXPORT int visloc_gsplat_cuda_set_scene(void* handle, int n, int sh_degree, const float* transforms,
                                            const float* opacity, const float* sh) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (n < 0 || n > kMaxGaussians || sh_degree < 0 || sh_degree > 3) {
        ctx->error = "set_scene: n must be in 0..=2^24 and the SH degree in 0..=3";
        return -1;
    }
    ctx->n = n;
    ctx->sh_degree = sh_degree;
    ctx->have_forward = false;
    ctx->vote_k = 0;
    const float* src[3] = {transforms, opacity, sh};
    for (int k = 0; k < 3; ++k) {
        const size_t count = size_t(n) * ctx->stride(k);
        float* p;
        GSC_TRY(ensure(ctx, ctx->params[k], count, &p));
        GSC_TRY(zero_moments(ctx, k, count));
        if (count > 0) GSC_CHECK(ctx, cudaMemcpy(p, src[k], count * sizeof(float), cudaMemcpyHostToDevice));
    }
    float *w, *s;
    GSC_TRY(ensure_zero(ctx, ctx->weight_max, n, &w));
    GSC_TRY(ensure_zero(ctx, ctx->seen, n, &s));
    if (ctx->sem_k > 0) GSC_TRY(reset_logits(ctx));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    return 0;
}

GSC_EXPORT int visloc_gsplat_cuda_num_gaussians(void* handle) {
    return handle ? static_cast<Ctx*>(handle)->n : -1;
}

// Download the current scene (all outputs nullable), in the upload layouts.
GSC_EXPORT int visloc_gsplat_cuda_get_scene(void* handle, float* transforms, float* opacity, float* sh) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    float* dst[3] = {transforms, opacity, sh};
    for (int k = 0; k < 3; ++k) {
        const size_t count = size_t(ctx->n) * ctx->stride(k);
        if (dst[k] && count > 0)
            GSC_CHECK(ctx, cudaMemcpy(dst[k], ctx->params[k].ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return 0;
}

// Render the current scene. `out_rgb` (nullable) receives width * height * 3
// floats, row-major RGB. `num_intersections` (nullable) receives the number
// of (tile, gaussian) pairs.
GSC_EXPORT int visloc_gsplat_cuda_render(void* handle, const GscCamera* cam, const float* bg, int sh_active_degree,
                                         float* out_rgb, uint32_t* num_intersections) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    GSC_TRY(set_camera(ctx, cam, bg, sh_active_degree));
    clear_stages(ctx);
    GSC_TRY(run_forward(ctx));
    GSC_TRY(collect_stages(ctx));
    if (out_rgb)
        GSC_CHECK(ctx, cudaMemcpy(out_rgb, ctx->image.ptr, ctx->pixels() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    if (num_intersections) *num_intersections = ctx->num_isect;
    return 0;
}

// visloc_gsplat_cuda_render, plus per pixel the median depth (camera z of the
// splat at which the transmittance first drops to 0.5 or below; 0 when it
// never does) and the opacity 1 - T: `out_depth` (nullable) receives
// width * height * 2 floats. With `planar` the depth is where the pixel's ray
// meets that splat's plane (normal: its smallest axis) instead of its centre.
// `out_expected` (nullable) receives the expected depth sum_i T_i a_i z_i per
// pixel, the channel the depth-prior loss trains. With `out_normal`
// (nullable) the render also has the normal channel sum_i T_i a_i n_i (3
// floats per pixel; n_i the splat's plane normal, facing the camera), the
// depth-normal loss's input.
GSC_EXPORT int visloc_gsplat_cuda_render_depth(void* handle, const GscCamera* cam, const float* bg,
                                               int sh_active_degree, int planar, float* out_rgb, float* out_depth,
                                               float* out_expected, float* out_normal) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    GSC_TRY(set_camera(ctx, cam, bg, sh_active_degree));
    float* d;
    GSC_TRY(ensure(ctx, ctx->depth_out, ctx->pixels() * 2, &d));
    float* expected;
    GSC_TRY(ensure(ctx, ctx->acc_depth, ctx->pixels(), &expected));
    float* normal = nullptr;
    if (out_normal) GSC_TRY(ensure(ctx, ctx->acc_normal, ctx->pixels() * 3, &normal));
    clear_stages(ctx);
    GSC_TRY(run_forward(ctx, d, planar != 0, expected, normal));
    GSC_TRY(collect_stages(ctx));
    if (out_rgb)
        GSC_CHECK(ctx, cudaMemcpy(out_rgb, ctx->image.ptr, ctx->pixels() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    if (out_depth) GSC_CHECK(ctx, cudaMemcpy(out_depth, d, ctx->pixels() * 2 * sizeof(float), cudaMemcpyDeviceToHost));
    if (out_expected)
        GSC_CHECK(ctx, cudaMemcpy(out_expected, expected, ctx->pixels() * sizeof(float), cudaMemcpyDeviceToHost));
    if (out_normal)
        GSC_CHECK(ctx, cudaMemcpy(out_normal, normal, ctx->pixels() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Render the current scene from `cam` and add its label votes (vote_kernel)
// to the totals: `labels` holds width * height bytes, each pixel's class
// (`num_classes`, at most 255, or more: no vote). The totals start at zero
// with `reset`, and whenever the scene or the class count changes.
GSC_EXPORT int visloc_gsplat_cuda_vote_labels(void* handle, const GscCamera* cam, const uint8_t* labels,
                                              int num_classes, int reset) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (num_classes < 1 || num_classes > 255) {
        ctx->error = "vote_labels: num_classes must be in 1..=255";
        return -1;
    }
    const float bg[3] = {0.f, 0.f, 0.f};
    GSC_TRY(set_camera(ctx, cam, bg, 0));
    float* votes;
    if (reset || ctx->vote_k != num_classes || ctx->vote_n != ctx->n) {
        GSC_TRY(ensure_zero(ctx, ctx->votes, size_t(ctx->n) * num_classes, &votes));
        ctx->vote_n = ctx->n;
        ctx->vote_k = num_classes;
    } else {
        GSC_TRY(ensure(ctx, ctx->votes, size_t(ctx->n) * num_classes, &votes));
    }
    uint8_t* lab;
    GSC_TRY(ensure(ctx, ctx->labels, ctx->pixels(), &lab));
    GSC_CHECK(ctx, cudaMemcpy(lab, labels, ctx->pixels(), cudaMemcpyHostToDevice));
    clear_stages(ctx);
    GSC_TRY(run_forward(ctx));
    if (ctx->n > 0) {
        const dim3 grid(ctx->tile_bw, ctx->tile_bh);
        vote_kernel<<<grid, kBlock>>>(ctx->cam.width, ctx->cam.height, static_cast<const uint2*>(ctx->ranges.ptr),
                                      static_cast<const uint32_t*>(ctx->vals_sorted.ptr),
                                      static_cast<const float2*>(ctx->xy.ptr),
                                      static_cast<const float4*>(ctx->conic_op.ptr), lab, uint32_t(num_classes),
                                      votes);
        GSC_CHECK(ctx, cudaGetLastError());
    }
    GSC_TRY(collect_stages(ctx));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    return 0;
}

// The vote totals of visloc_gsplat_cuda_vote_labels: n * num_classes floats,
// gaussian-major, into `out`.
GSC_EXPORT int visloc_gsplat_cuda_get_votes(void* handle, int num_classes, float* out) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (ctx->vote_k != num_classes || ctx->vote_n != ctx->n) {
        ctx->error = "get_votes: no votes for this scene and class count";
        return -1;
    }
    const size_t count = size_t(ctx->n) * num_classes;
    if (count > 0) GSC_CHECK(ctx, cudaMemcpy(out, ctx->votes.ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Use `num_classes` (1..=16) semantic logits per gaussian: upload them
// (n * num_classes floats, gaussian-major; null: zeros), with zero Adam
// moments. set_scene zeroes them again, and refines carry them along.
GSC_EXPORT int visloc_gsplat_cuda_set_logits(void* handle, int num_classes, const float* logits) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (num_classes < 1 || num_classes > kMaxSem) {
        ctx->error = "set_logits: num_classes must be in 1..=16";
        return -1;
    }
    if (ctx->sem_k != num_classes) ctx->sem_images = 0;
    ctx->sem_k = num_classes;
    GSC_TRY(reset_logits(ctx));
    const size_t count = size_t(ctx->n) * num_classes;
    if (logits && count > 0) GSC_CHECK(ctx, cudaMemcpy(ctx->sem.ptr, logits, count * sizeof(float), cudaMemcpyHostToDevice));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    return 0;
}

// The semantic logits: n * num_classes floats, gaussian-major, into `out`.
GSC_EXPORT int visloc_gsplat_cuda_get_logits(void* handle, int num_classes, float* out) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (ctx->sem_k == 0 || num_classes != ctx->sem_k) {
        ctx->error = "get_logits: no logits of this class count";
        return -1;
    }
    const size_t count = size_t(ctx->n) * num_classes;
    if (count > 0) GSC_CHECK(ctx, cudaMemcpy(out, ctx->sem.ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Upload a label image per uploaded image (same order and size), one byte
// per pixel: the class, or num_classes or more for none, image-major. The
// logits (num_classes per gaussian) start at zero unless set_logits gave
// this many. A step with semantic_weight > 0 adds semantic_weight times the
// mean cross-entropy over the image's labelled pixels.
GSC_EXPORT int visloc_gsplat_cuda_set_labels(void* handle, int num_classes, int count, uint32_t width,
                                             uint32_t height, const uint8_t* labels) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (num_classes < 1 || num_classes > kMaxSem) {
        ctx->error = "set_labels: num_classes must be in 1..=16";
        return -1;
    }
    if (count != ctx->num_images || width != ctx->gt_w || height != ctx->gt_h || (count > 0 && !labels)) {
        ctx->error = "set_labels: one label image per uploaded image, of its size";
        return -1;
    }
    if (ctx->sem_k != num_classes) {
        ctx->sem_k = num_classes;
        GSC_TRY(reset_logits(ctx));
    }
    const size_t per = size_t(width) * height;
    uint8_t* d;
    GSC_TRY(ensure(ctx, ctx->sem_labels, size_t(count) * per, &d));
    if (count > 0) GSC_CHECK(ctx, cudaMemcpy(d, labels, size_t(count) * per, cudaMemcpyHostToDevice));
    ctx->sem_counts.assign(size_t(count), 0);
    for (size_t i = 0; i < size_t(count) * per; ++i)
        if (labels[i] < num_classes) ctx->sem_counts[i / per]++;
    ctx->sem_images = count;
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    return 0;
}

// Render the current scene from `cam` and its class channels (every pixel),
// as training blends them: `out` (nullable) receives width * height *
// num_classes floats, P_c = sum_i T_i a_i p_ic per pixel, p_i the softmax of
// gaussian i's logits.
GSC_EXPORT int visloc_gsplat_cuda_render_classes(void* handle, const GscCamera* cam, float* out) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (ctx->sem_k == 0) {
        ctx->error = "render_classes: no logits (set_logits or set_labels)";
        return -1;
    }
    const float bg[3] = {0.f, 0.f, 0.f};
    GSC_TRY(set_camera(ctx, cam, bg, 0));
    clear_stages(ctx);
    GSC_TRY(run_forward(ctx));
    GSC_TRY(run_sem_forward(ctx));
    GSC_TRY(collect_stages(ctx));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    if (out)
        GSC_CHECK(ctx, cudaMemcpy(out, ctx->acc_sem.ptr, ctx->pixels() * ctx->sem_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

// Gradients of L = sum_px dot(d_sem[px], P[px]) (num_classes floats per
// pixel) for the last render_classes, as visloc_gsplat_cuda_backward's for
// the colour (all outputs nullable): the transforms and opacity, the logits
// (n * num_classes, through the softmax) and the 10 screen-space floats per
// gaussian (the colour ones zero).
GSC_EXPORT int visloc_gsplat_cuda_classes_backward(void* handle, const float* d_sem, float* g_transforms,
                                                   float* g_opacity, float* g_logits, float* screen_grads) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!d_sem || ctx->sem_k == 0) {
        ctx->error = "classes_backward: d_sem and logits are required";
        return -1;
    }
    const size_t pixels = ctx->pixels();
    float* d;
    GSC_TRY(ensure(ctx, ctx->d_sem, pixels * ctx->sem_k, &d));
    GSC_CHECK(ctx, cudaMemcpy(d, d_sem, pixels * ctx->sem_k * sizeof(float), cudaMemcpyHostToDevice));
    for (int s = kLoss; s < kNumStages; ++s) {
        ctx->ran[s] = false;
        ctx->ms[s] = 0.f;
    }
    SemBackward sem;
    sem.channels = true;
    GSC_TRY(run_backward(ctx, true, 0, sem, false));
    GSC_TRY(collect_stages(ctx));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    float* dst[2] = {g_transforms, g_opacity};
    for (int k = 0; k < 2; ++k) {
        const size_t count = size_t(ctx->n) * ctx->stride(k);
        if (dst[k] && count > 0)
            GSC_CHECK(ctx, cudaMemcpy(dst[k], ctx->grads[k].ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (g_logits && ctx->n > 0)
        GSC_CHECK(ctx, cudaMemcpy(g_logits, ctx->sem_grad.ptr, size_t(ctx->n) * ctx->sem_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    if (screen_grads && ctx->n > 0)
        GSC_CHECK(ctx, cudaMemcpy(screen_grads, ctx->screen_grads.ptr, size_t(ctx->n) * kNumGrad * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

// The labels' cross-entropy as a training step computes it, for uploaded
// image `image` seen from `cam` (that image's size): `loss` (nullable)
// receives the unweighted mean over its labelled pixels, and the outputs
// (nullable, as classes_backward's) the gradients of weight times it, the
// logits' through the softmax. `geometry` as train_step's
// semantic_geometry: 0 only the logits get one, 1 also the opacities and
// centres, 2 the shapes too (classes_backward's).
GSC_EXPORT int visloc_gsplat_cuda_label_backward(void* handle, const GscCamera* cam, uint32_t image, float weight,
                                                 int geometry, double* loss, float* g_transforms, float* g_opacity,
                                                 float* g_logits, float* screen_grads) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!cam || ctx->sem_k == 0 || ctx->sem_images != ctx->num_images || image >= uint32_t(ctx->num_images) ||
        cam->width != ctx->gt_w || cam->height != ctx->gt_h) {
        ctx->error = "label_backward: needs labels (set_labels) and a camera of their size";
        return -1;
    }
    const float bg[3] = {0.f, 0.f, 0.f};
    GSC_TRY(set_camera(ctx, cam, bg, 0));
    clear_stages(ctx);
    GSC_TRY(run_softmax(ctx));
    SemBackward sem;
    sem.labels_ce = true;
    sem.labels = image_labels(ctx, image);
    sem.geometry = std::clamp(geometry, 0, 2);
    GSC_TRY(run_forward(ctx, nullptr, false, nullptr, nullptr, sem.labels));
    // The slot after the step's sums, so take_loss is undisturbed.
    double* acc;
    GSC_TRY(ensure(ctx, ctx->loss_acc, kNumLossSums + 1, &acc));
    double* sum = acc + kNumLossSums;
    GSC_CHECK(ctx, cudaMemsetAsync(sum, 0, sizeof(double)));
    GSC_TRY(run_label_loss(ctx, image, weight, sum));
    // No colour gradient.
    float* d_image;
    GSC_TRY(ensure_zero(ctx, ctx->d_image, ctx->pixels() * 3, &d_image));
    GSC_TRY(run_backward(ctx, true, 0, sem));
    GSC_TRY(collect_stages(ctx));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    if (loss) GSC_CHECK(ctx, cudaMemcpy(loss, sum, sizeof(double), cudaMemcpyDeviceToHost));
    float* dst[2] = {g_transforms, g_opacity};
    for (int k = 0; k < 2; ++k) {
        const size_t count = size_t(ctx->n) * ctx->stride(k);
        if (dst[k] && count > 0)
            GSC_CHECK(ctx, cudaMemcpy(dst[k], ctx->grads[k].ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (g_logits && ctx->n > 0)
        GSC_CHECK(ctx, cudaMemcpy(g_logits, ctx->sem_grad.ptr, size_t(ctx->n) * ctx->sem_k * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    if (screen_grads && ctx->n > 0)
        GSC_CHECK(ctx, cudaMemcpy(screen_grads, ctx->screen_grads.ptr, size_t(ctx->n) * kNumGrad * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

// The depth-normal loss (weight * mean (1 - N . n_d), see normal_grad_kernel)
// of the last render, which must have come from render_depth with the normal
// channel. `loss` (nullable) receives its unweighted value, `d_depth` and
// `d_normal` (nullable; 1 and 3 floats per pixel) its gradient with respect
// to the expected-depth and normal channels.
GSC_EXPORT int visloc_gsplat_cuda_normal_loss(void* handle, float weight, double* loss, float* d_depth,
                                              float* d_normal) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!ctx->have_forward || !ctx->have_normal) {
        ctx->error = "normal_loss: render the normal channel first (render_depth with out_normal)";
        return -1;
    }
    const size_t pixels = ctx->pixels();
    float* dd;
    GSC_TRY(ensure_zero(ctx, ctx->d_depth, pixels, &dd));
    // The slot after the step's sums, so take_loss is undisturbed.
    double* acc;
    GSC_TRY(ensure(ctx, ctx->loss_acc, kNumLossSums + 1, &acc));
    double* sum = acc + kNumLossSums;
    GSC_CHECK(ctx, cudaMemsetAsync(sum, 0, sizeof(double)));
    GSC_TRY(run_normal_loss(ctx, weight, sum));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    if (loss) GSC_CHECK(ctx, cudaMemcpy(loss, sum, sizeof(double), cudaMemcpyDeviceToHost));
    if (d_depth) GSC_CHECK(ctx, cudaMemcpy(d_depth, dd, pixels * sizeof(float), cudaMemcpyDeviceToHost));
    if (d_normal)
        GSC_CHECK(ctx, cudaMemcpy(d_normal, ctx->d_normal.ptr, pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Gradients of L = sum_px dot(d_image[px], C[px]) for the last render, in
// the scene layouts (all outputs nullable). Gaussians that were not visible
// get zeros. `screen_grads` receives 10 floats per gaussian: du dv dA dB dC
// dopacity dr dg db and the refine weight. With `d_depth` (nullable, one
// float per pixel) L also has sum_px d_depth[px] D[px], D the expected depth,
// and with `d_normal` (nullable, 3 floats per pixel; needs `d_depth` and a
// render with the normal channel) sum_px dot(d_normal[px], N[px]).
GSC_EXPORT int visloc_gsplat_cuda_backward(void* handle, const float* d_image, const float* d_depth,
                                           const float* d_normal, float* g_transforms, float* g_opacity, float* g_sh,
                                           float* screen_grads) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!d_image || (d_normal && !d_depth)) {
        ctx->error = "backward: d_image is required, and d_normal needs d_depth";
        return -1;
    }
    const size_t pixels = ctx->pixels();
    float* d;
    GSC_TRY(ensure(ctx, ctx->d_image, pixels * 3, &d));
    GSC_CHECK(ctx, cudaMemcpy(d, d_image, pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    if (d_depth) {
        float* dd;
        GSC_TRY(ensure(ctx, ctx->d_depth, pixels, &dd));
        GSC_CHECK(ctx, cudaMemcpy(dd, d_depth, pixels * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (d_normal) {
        float* dn;
        GSC_TRY(ensure(ctx, ctx->d_normal, pixels * 3, &dn));
        GSC_CHECK(ctx, cudaMemcpy(dn, d_normal, pixels * 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
    for (int s = kLoss; s < kNumStages; ++s) {
        ctx->ran[s] = false;
        ctx->ms[s] = 0.f;
    }
    GSC_TRY(run_backward(ctx, true, d_normal ? 2 : (d_depth ? 1 : 0)));
    GSC_TRY(collect_stages(ctx));
    float* dst[3] = {g_transforms, g_opacity, g_sh};
    for (int k = 0; k < 3; ++k) {
        const size_t count = size_t(ctx->n) * ctx->stride(k);
        if (dst[k] && count > 0)
            GSC_CHECK(ctx, cudaMemcpy(dst[k], ctx->grads[k].ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (screen_grads && ctx->n > 0)
        GSC_CHECK(ctx, cudaMemcpy(screen_grads, ctx->screen_grads.ptr, size_t(ctx->n) * kNumGrad * sizeof(float),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

// Upload the training images: `count` images of width x height, one packed
// RGBA8 u32 per pixel (R in the low byte; alpha below 128 leaves the pixel
// out of the loss), image-major.
GSC_EXPORT int visloc_gsplat_cuda_set_images(void* handle, int count, uint32_t width, uint32_t height,
                                             const uint32_t* rgba) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (count < 0 || (count > 0 && (!rgba || width == 0 || height == 0))) {
        ctx->error = "set_images: bad image set";
        return -1;
    }
    const size_t total = size_t(count) * width * height;
    uint32_t* gt;
    GSC_TRY(ensure(ctx, ctx->gt, total, &gt));
    if (total > 0) GSC_CHECK(ctx, cudaMemcpy(gt, rgba, total * sizeof(uint32_t), cudaMemcpyHostToDevice));
    ctx->num_images = count;
    ctx->gt_w = width;
    ctx->gt_h = height;
    return 0;
}

// Upload a depth prior per uploaded image (same order and size): camera-z
// depths, 0 where there is none, image-major. A step with depth_weight > 0
// adds depth_weight * mean |D - P| / P over the pixels with a prior P, D the
// render's expected depth.
GSC_EXPORT int visloc_gsplat_cuda_set_depth_priors(void* handle, int count, uint32_t width, uint32_t height,
                                                   const float* depths) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (count != ctx->num_images || width != ctx->gt_w || height != ctx->gt_h || (count > 0 && !depths)) {
        ctx->error = "set_depth_priors: one prior per uploaded image, of its size";
        return -1;
    }
    const size_t total = size_t(count) * width * height;
    float* d;
    GSC_TRY(ensure(ctx, ctx->depth_prior, total, &d));
    if (total > 0) GSC_CHECK(ctx, cudaMemcpy(d, depths, total * sizeof(float), cudaMemcpyHostToDevice));
    ctx->num_priors = count;
    return 0;
}

// One optimisation step on uploaded image `p->image` (the wgpu trainer's
// step): render, (1 - w) L1 + w D-SSIM and its image gradient, backward,
// refine statistics, Adam on every group with brush's aux losses, and mean
// noise. With a semantic weight also the labels' cross-entropy and Adam on
// the logits.
GSC_EXPORT int visloc_gsplat_cuda_train_step(void* handle, const GscStepParams* p) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!p) {
        ctx->error = "train_step: params are required";
        return -1;
    }
    GSC_TRY(set_camera(ctx, &p->cam, p->bg, p->sh_active_degree));
    const bool depth_on = p->depth_weight > 0.f;
    const bool normal_on = p->normal_weight > 0.f;
    const bool sem_on = p->semantic_weight > 0.f;
    if (depth_on && (ctx->num_priors != ctx->num_images || p->cam.width != ctx->gt_w || p->cam.height != ctx->gt_h)) {
        ctx->error = "train_step: a depth weight needs a depth prior per uploaded image";
        return -1;
    }
    if (sem_on && (ctx->sem_k == 0 || ctx->sem_images != ctx->num_images || p->cam.width != ctx->gt_w ||
                   p->cam.height != ctx->gt_h)) {
        ctx->error = "train_step: a semantic weight needs labels per uploaded image (set_labels)";
        return -1;
    }
    clear_stages(ctx);
    SemBackward sem;
    if (sem_on) {
        sem.labels_ce = true;
        sem.labels = image_labels(ctx, p->image);
        sem.geometry = int(std::min(p->semantic_geometry, 2u));
        GSC_TRY(stage_begin(ctx, kSemantic));
        GSC_TRY(run_softmax(ctx));
        GSC_TRY(stage_end(ctx, kSemantic));
    }
    if (depth_on || normal_on) {
        float *depth_out, *acc_depth, *acc_normal = nullptr;
        GSC_TRY(ensure(ctx, ctx->depth_out, ctx->pixels() * 2, &depth_out));
        GSC_TRY(ensure(ctx, ctx->acc_depth, ctx->pixels(), &acc_depth));
        if (normal_on) GSC_TRY(ensure(ctx, ctx->acc_normal, ctx->pixels() * 3, &acc_normal));
        GSC_TRY(run_forward(ctx, depth_out, false, acc_depth, acc_normal, sem.labels));
    } else {
        GSC_TRY(run_forward(ctx, nullptr, false, nullptr, nullptr, sem.labels));
    }
    GSC_TRY(run_loss(ctx, p->image, p->ssim_weight, depth_on ? p->depth_weight : 0.f,
                     normal_on ? p->normal_weight : 0.f, sem_on ? p->semantic_weight : 0.f));
    GSC_TRY(run_backward(ctx, p->sparse_sh_adam == 0, normal_on ? 2 : (depth_on ? 1 : 0), sem));

    const int n = ctx->n;
    const uint32_t* tiles = static_cast<const uint32_t*>(ctx->tiles.ptr);
    GSC_TRY(stage_begin(ctx, kOptimizer));
    if (n > 0) {
        if (p->accumulate_stats) {
            float *w, *s;
            GSC_TRY(ensure(ctx, ctx->weight_max, n, &w));
            GSC_TRY(ensure(ctx, ctx->seen, n, &s));
            stats_kernel<<<blocks_for(n), 256>>>(n, tiles, static_cast<const float*>(ctx->screen_grads.ptr), w, s);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        // (stride, split_a, split_b, lr_a, lr_b, lr_c, aux kind, aux coef) per group.
        const uint32_t rl = uint32_t(ctx->row_len());
        const struct {
            uint32_t stride, split_a, split_b;
            float lr_a, lr_b, lr_c;
            int aux_kind;
            float aux_coef;
            int sparse;
        } groups[3] = {
            {10, 3, 7, p->lr[0], p->lr[1], p->lr[2], p->scale_aux_coef > 0.f ? 1 : 0, p->scale_aux_coef, 0},
            {1, 1, 1, p->lr[3], p->lr[3], p->lr[3], p->opacity_aux_coef > 0.f ? 2 : 0, p->opacity_aux_coef, 0},
            {rl, 3, 3, p->lr[4], p->lr[5], p->lr[5], 0, 0.f, p->sparse_sh_adam ? 1 : 0},
        };
        GSC_TRY(convert_moments(ctx, p->moments_bf16 != 0));
        const uint32_t seed = ctx->adam_steps++ * 0x9e3779b9u;
        for (int k = 0; k < 3; ++k) {
            const size_t n_elem = size_t(n) * groups[k].stride;
            const uint8_t* sparse = groups[k].sparse ? static_cast<const uint8_t*>(ctx->touched.ptr) : nullptr;
            const float* grads = static_cast<const float*>(ctx->grads[k].ptr);
            if (ctx->m_bf16)
                adam_kernel<<<blocks_for(n_elem), 256>>>(
                    n_elem, groups[k].stride, groups[k].split_a, groups[k].split_b, groups[k].lr_a, groups[k].lr_b,
                    groups[k].lr_c, p->beta1, p->beta2, p->eps, p->bc1, p->bc2, groups[k].aux_kind,
                    groups[k].aux_coef, sparse, tiles, ctx->p(k), grads, static_cast<uint16_t*>(ctx->m1[k].ptr),
                    static_cast<uint16_t*>(ctx->m2[k].ptr), seed);
            else
                adam_kernel<<<blocks_for(n_elem), 256>>>(
                    n_elem, groups[k].stride, groups[k].split_a, groups[k].split_b, groups[k].lr_a, groups[k].lr_b,
                    groups[k].lr_c, p->beta1, p->beta2, p->eps, p->bc1, p->bc2, groups[k].aux_kind,
                    groups[k].aux_coef, sparse, tiles, ctx->p(k), grads, static_cast<float*>(ctx->m1[k].ptr),
                    static_cast<float*>(ctx->m2[k].ptr), seed);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        if (sem_on) {
            const uint32_t k = uint32_t(ctx->sem_k);
            const size_t n_elem = size_t(n) * k;
            adam_kernel<<<blocks_for(n_elem), 256>>>(
                n_elem, k, k, k, p->lr_semantic, p->lr_semantic, p->lr_semantic, p->beta1, p->beta2, p->eps, p->bc1,
                p->bc2, 0, 0.f, nullptr, tiles, static_cast<float*>(ctx->sem.ptr),
                static_cast<const float*>(ctx->sem_grad.ptr), static_cast<float*>(ctx->sem_m1.ptr),
                static_cast<float*>(ctx->sem_m2.ptr), 0u);
            GSC_CHECK(ctx, cudaGetLastError());
        }
        if (p->noise_scale > 0.f) {
            noise_kernel<<<blocks_for(n), 256>>>(n, p->noise_seed, p->noise_scale, p->noise_max, ctx->p(0),
                                                 ctx->p(1));
            GSC_CHECK(ctx, cudaGetLastError());
        }
    }
    GSC_TRY(stage_end(ctx, kOptimizer));
    // The parameters moved: a backward needs a fresh render.
    ctx->have_forward = false;
    GSC_TRY(collect_stages(ctx));
    return 0;
}

// The training loss of the last render against uploaded image `image`
// (what train_step computes): `d_image` (nullable) receives its gradient,
// width * height * 3 floats. The loss sums go to take_loss as a step's do.
GSC_EXPORT int visloc_gsplat_cuda_loss_grad(void* handle, uint32_t image, float ssim_weight, float* d_image) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    GSC_TRY(run_loss(ctx, image, ssim_weight));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    if (d_image)
        GSC_CHECK(ctx, cudaMemcpy(d_image, ctx->d_image.ptr, ctx->pixels() * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

// Sums of the mean L1, of the SSIM (over pixels and channels), of the mean
// depth-prior loss, of the mean depth-normal loss and of the labels' mean
// cross-entropy (all unweighted) since the last call, and the number of
// steps; resets them.
GSC_EXPORT int visloc_gsplat_cuda_take_loss(void* handle, double* l1_sum, double* ssim_sum, double* depth_sum,
                                            double* normal_sum, double* semantic_sum, uint32_t* steps) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    double acc[kNumLossSums] = {};
    if (ctx->loss_steps > 0) GSC_CHECK(ctx, cudaMemcpy(acc, ctx->loss_acc.ptr, sizeof(acc), cudaMemcpyDeviceToHost));
    if (l1_sum) *l1_sum = acc[0];
    if (ssim_sum) *ssim_sum = acc[1];
    if (depth_sum) *depth_sum = acc[2];
    if (normal_sum) *normal_sum = acc[3];
    if (semantic_sum) *semantic_sum = acc[4];
    if (steps) *steps = ctx->loss_steps;
    ctx->loss_steps = 0;
    return 0;
}

// brush refine statistics per gaussian since the last refine (nullable
// outputs): the max refine weight and the number of frames it was blended.
GSC_EXPORT int visloc_gsplat_cuda_get_refine_stats(void* handle, float* weight_max, float* seen) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    const size_t bytes = size_t(ctx->n) * sizeof(float);
    if (weight_max && bytes > 0)
        GSC_CHECK(ctx, cudaMemcpy(weight_max, ctx->weight_max.ptr, bytes, cudaMemcpyDeviceToHost));
    if (seen && bytes > 0) GSC_CHECK(ctx, cudaMemcpy(seen, ctx->seen.ptr, bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// Apply one action per gaussian (0 prune, 1 keep, 2 clone, 4 brush split;
// see scatter_kernel) to the scene and its Adam moments on the device, then
// zero the refine statistics. `n_after` (nullable) receives the new count.
GSC_EXPORT int visloc_gsplat_cuda_apply_actions(void* handle, const uint32_t* actions, uint32_t seed, int* n_after) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    const int n = ctx->n;
    if (n > 0 && !actions) {
        ctx->error = "apply_actions: actions are required";
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (actions[i] == 3 || actions[i] > 4) {
            ctx->error = "apply_actions: only actions 0 (prune), 1 (keep), 2 (clone) and 4 (brush split) are supported";
            return -1;
        }
    }
    uint32_t total = 0;
    if (n > 0) {
        uint32_t *d_actions, *counts, *cum;
        GSC_TRY(ensure(ctx, ctx->actions, n, &d_actions));
        GSC_TRY(ensure(ctx, ctx->counts, n, &counts));
        GSC_TRY(ensure(ctx, ctx->cum, n, &cum));
        GSC_CHECK(ctx, cudaMemcpy(d_actions, actions, size_t(n) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        action_counts_kernel<<<blocks_for(n), 256>>>(n, d_actions, counts);
        GSC_CHECK(ctx, cudaGetLastError());
        size_t temp_bytes = 0;
        GSC_CHECK(ctx, cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, counts, cum, n));
        uint8_t* temp;
        GSC_TRY(ensure(ctx, ctx->scan_temp, temp_bytes, &temp));
        GSC_CHECK(ctx, cub::DeviceScan::InclusiveSum(temp, temp_bytes, counts, cum, n));
        GSC_CHECK(ctx, cudaMemcpy(&total, cum + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        if (total > uint32_t(kMaxGaussians)) {
            ctx->error = "apply_actions: more than 2^24 gaussians";
            return -1;
        }
        for (int k = 0; k < 3; ++k) {
            const size_t stride = ctx->stride(k);
            Buf dp, dm1, dm2;
            float* p;
            GSC_TRY(ensure(ctx, dp, size_t(total) * stride, &p));
            if (ctx->m_bf16) {
                uint16_t *a, *b;
                GSC_TRY(ensure(ctx, dm1, size_t(total) * stride, &a));
                GSC_TRY(ensure(ctx, dm2, size_t(total) * stride, &b));
                scatter_kernel<<<blocks_for(n), 256>>>(n, uint32_t(stride), k, seed, d_actions, cum, ctx->p(k),
                                                       static_cast<const uint16_t*>(ctx->m1[k].ptr),
                                                       static_cast<const uint16_t*>(ctx->m2[k].ptr), p, a, b);
            } else {
                float *a, *b;
                GSC_TRY(ensure(ctx, dm1, size_t(total) * stride, &a));
                GSC_TRY(ensure(ctx, dm2, size_t(total) * stride, &b));
                scatter_kernel<<<blocks_for(n), 256>>>(n, uint32_t(stride), k, seed, d_actions, cum, ctx->p(k),
                                                       static_cast<const float*>(ctx->m1[k].ptr),
                                                       static_cast<const float*>(ctx->m2[k].ptr), p, a, b);
            }
            GSC_CHECK(ctx, cudaGetLastError());
            GSC_CHECK(ctx, cudaDeviceSynchronize());
            free_buf(ctx->params[k]);
            free_buf(ctx->m1[k]);
            free_buf(ctx->m2[k]);
            ctx->params[k] = dp;
            ctx->m1[k] = dm1;
            ctx->m2[k] = dm2;
        }
        if (ctx->sem_k > 0) {
            // The logits ride along as plain copies (a split's two halves keep
            // the parent's).
            const uint32_t stride = uint32_t(ctx->sem_k);
            Buf dp, dm1, dm2;
            float *p, *a, *b;
            GSC_TRY(ensure(ctx, dp, size_t(total) * stride, &p));
            GSC_TRY(ensure(ctx, dm1, size_t(total) * stride, &a));
            GSC_TRY(ensure(ctx, dm2, size_t(total) * stride, &b));
            scatter_kernel<<<blocks_for(n), 256>>>(n, stride, 3, seed, d_actions, cum,
                                                   static_cast<const float*>(ctx->sem.ptr),
                                                   static_cast<const float*>(ctx->sem_m1.ptr),
                                                   static_cast<const float*>(ctx->sem_m2.ptr), p, a, b);
            GSC_CHECK(ctx, cudaGetLastError());
            GSC_CHECK(ctx, cudaDeviceSynchronize());
            free_buf(ctx->sem);
            free_buf(ctx->sem_m1);
            free_buf(ctx->sem_m2);
            ctx->sem = dp;
            ctx->sem_m1 = dm1;
            ctx->sem_m2 = dm2;
        }
    }
    ctx->n = int(total);
    ctx->have_forward = false;
    ctx->vote_k = 0;
    float *w, *s;
    GSC_TRY(ensure_zero(ctx, ctx->weight_max, total, &w));
    GSC_TRY(ensure_zero(ctx, ctx->seen, total, &s));
    GSC_CHECK(ctx, cudaDeviceSynchronize());
    if (n_after) *n_after = int(total);
    return 0;
}

// Milliseconds of each stage of the last render / backward / train step (GPU
// time between CUDA events; 0 for stages that did not run), in the order of
// visloc_gsplat_cuda_stage_names. Returns the number of stages written.
GSC_EXPORT int visloc_gsplat_cuda_stage_times(void* handle, float* ms, int cap) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !ms) return -1;
    const int count = cap < kNumStages ? cap : kNumStages;
    for (int s = 0; s < count; ++s) ms[s] = ctx->ms[s];
    return count;
}

}  // extern "C"

// GPU TSDF fusion and surface-nets extraction (visloc_gsplat_cuda_tsdf_*).
#include "tsdf.cuh"

// Exhaustive SIFT matching (visloc_gsplat_cuda_match_*).
#include "match.cuh"
