// PatchMatch MVS on CUDA for visloc-mvs (opt-in): the wgpu shaders of
// crates/mvs/src/shaders/patch_match.wgsl as CUDA kernels, one thread per
// pixel. Built explicitly with `scripts/build_mvs_cuda.sh`, never by Cargo;
// `crates/mvs` (feature `cuda`) loads the shared library at runtime and checks
// `visloc_mvs_cuda_abi_version`.
//
// The kernels follow the shader line by line, and so the CPU path
// (`patch_match.rs`): the same sampling regions, view selection, refinement
// and random streams. Two differences, both towards the CPU path:
//
// - The sources' grey levels are f32 and sampled bilinearly in software,
//   where the shader samples a half-float texture with the texture unit's
//   fixed-point weights.
// - Every WGSL variable and private array starts at zero, as WGSL defines
//   them; here they are zeroed explicitly.
//
// The host side mirrors `GpuPatchMatch::run`: `begin` uploads a pass's inputs
// and runs `init`; `classify`, `set_classes` and `find_anchors` handle
// deformable patches (the nearest strong pixels come from the CPU); one
// `propagate` call per (iteration, colour); `read` returns the planes and
// costs.

#include <cuda_runtime.h>

#include <cstdint>
#include <new>
#include <string>

extern "C" {

// Mirrored by `pass_setup::Params` in crates/mvs/src/pass_setup.rs.
struct MvsParams {
    uint32_t width, height, num_sources, color, iteration, seed, view_id, geometric;
    float fx, fy, cx, cy, depth_min, depth_max, tau, tau1, beta;
    uint32_t n1, n2, init_top_k;
    float geom_lambda, geom_delta, depth_perturbation, normal_perturbation;
    int32_t window_radius, window_step;
    float sigma_spatial, sigma_color;
    uint32_t view_selection;
    float inv_texture_width, inv_texture_height;
    uint32_t planar_prior;
    float prior_inv_alpha, prior_gamma, prior_depth_weight, prior_angle_weight, prior_depth_sigma, prior_angle_sigma;
    uint32_t restore;
    float restore_margin;
    uint32_t deformable;
    int32_t sweep, peak_radius;
    float max_peak_cost, strong_cost, peak_spread;
    uint32_t sectors;
    float ransac_threshold;
    uint32_t min_inliers;
    float center_weight, baseline;
};

// Mirrored by `pass_setup::SourceCam`: reference-to-source rotation rows
// with the translation in w, fx fy cx cy, (width, height, unused, offset of
// the earlier depth map).
struct MvsSource {
    float r0[4], r1[4], r2[4], k[4];
    uint32_t dims[4];
};

// A pass's inputs (host pointers), see visloc_mvs_cuda_begin.
struct MvsInputs {
    uint32_t width, height, num_sources, texture_width, texture_height;
    const MvsSource* sources;  // num_sources
    const float* layers;       // max(1, num_sources) layers of texture_width x texture_height
    const float* image;        // the reference grey levels, width x height
    uint32_t prior_len;
    const float* priors;        // the sources' earlier depth maps (nullable when prior_len is 0)
    const float* init_planes;   // (normal, depth) per pixel, depth 0 for a random plane
    const float* prior_planes;  // (n, rho) per pixel for the planar prior pass (nullable)
    uint32_t pre_costs;         // 1: the detail restorer's baseline costs are needed
    uint32_t classes;           // 1: per-pixel classes are needed (deformable or sorting)
    uint32_t deformable;        // 1: nearest strong pixels and anchors are needed
};

}  // extern "C"

namespace {

constexpr uint32_t kAbiVersion = 1;

constexpr float MAX_COST = 2.0f;
constexpr uint32_t MAX_SAMPLES = 64u;
constexpr float MIN_WEIGHT_SUM = 1e-6f;
constexpr uint32_t MAX_SOURCES = 32u;
constexpr uint32_t NO_VIEW = 0xffffffffu;
constexpr uint32_t VIEW_SAMPLES = 15u;
constexpr uint32_t SAMPLING_STREAM = 0x80000000u;
constexpr uint32_t MAX_ANCHORS = 8u;
constexpr uint32_t ANCHOR_SAMPLES = 9u;
constexpr uint32_t MAX_CURVE = 61u;
constexpr uint32_t MAX_SECTORS = 32u;
constexpr uint32_t BORDER = 6u;
constexpr uint32_t WEAK = 0u;
constexpr uint32_t STRONG = 1u;
constexpr uint32_t UNKNOWN = 2u;

// ---- Small vector maths (WGSL's vec2 / vec3 / vec4) ----

__device__ inline float3 f3(float x, float y, float z) { return make_float3(x, y, z); }
__device__ inline float3 operator+(float3 a, float3 b) { return f3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ inline float3 operator-(float3 a, float3 b) { return f3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ inline float3 operator-(float3 a) { return f3(-a.x, -a.y, -a.z); }
__device__ inline float3 operator*(float s, float3 a) { return f3(s * a.x, s * a.y, s * a.z); }
__device__ inline float3 operator*(float3 a, float s) { return f3(s * a.x, s * a.y, s * a.z); }
__device__ inline float3 operator/(float3 a, float s) { return f3(a.x / s, a.y / s, a.z / s); }
__device__ inline float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ inline float3 cross(float3 a, float3 b) {
    return f3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ inline float length(float3 a) { return sqrtf(dot(a, a)); }
__device__ inline float3 xyz(float4 v) { return f3(v.x, v.y, v.z); }
__device__ inline float4 f4(float3 v, float w) { return make_float4(v.x, v.y, v.z, w); }
__device__ inline float2 f2(float x, float y) { return make_float2(x, y); }
__device__ inline float2 operator-(float2 a, float2 b) { return f2(a.x - b.x, a.y - b.y); }
__device__ inline float length(float2 a) { return sqrtf(a.x * a.x + a.y * a.y); }
__device__ inline float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
__device__ inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ inline int clampi(int x, int lo, int hi) { return min(max(x, lo), hi); }
__device__ inline bool is_zero3(float4 v) { return v.x == 0.f && v.y == 0.f && v.z == 0.f; }

// Device buffers of a pass.
struct Bufs {
    const MvsSource* sources;
    const float* images;
    const float* layers;
    uint32_t texture_width, texture_height;
    const float* priors;
    float4* planes;
    float* costs;
    uint32_t* best_views;
    const float4* init_planes;
    const float4* prior_planes;
    float* pre_costs;
    uint32_t* classes;
    const uint32_t* nearest;
    uint32_t* anchors;
};

__device__ inline uint32_t hash(uint32_t v) {
    uint32_t x = v;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// One pixel's work: the shader's `var<private>` state and its functions.
struct Pixel {
    const MvsParams& params;
    const Bufs& b;
    float patch_values[64] = {};
    float patch_weights[64] = {};
    uint32_t patch_n = 0;
    float patch_center = 0.f;
    float patch_sum_w = 0.f;
    float patch_mean = 0.f;
    float patch_var = 0.f;
    uint32_t rng_state = 0;
    uint32_t top_k_views = 0;
    uint32_t anchor_count = 0;
    float2 anchor_offsets[72] = {};
    float anchor_values[72] = {};
    float anchor_centers[8] = {};
    float anchor_means[8] = {};
    float anchor_vars[8] = {};
    uint32_t draw_state = 0;

    __device__ Pixel(const MvsParams& p, const Bufs& bufs) : params(p), b(bufs) {}

    __device__ void rng_init(uint32_t x, uint32_t y, uint32_t pass_id) {
        uint32_t h = hash(params.seed ^ 0x9e3779b9u);
        h = hash(h ^ params.view_id);
        h = hash(h ^ x);
        h = hash(h ^ y);
        h = hash(h ^ pass_id);
        rng_state = h;
    }

    __device__ float uniform01() {
        rng_state = hash(rng_state + 0x9e3779b9u);
        return float(rng_state >> 8u) * (1.0f / 16777216.0f);
    }

    __device__ float ref_gray(int x, int y) const {
        const int w = int(params.width), h = int(params.height);
        const int cx = clampi(x, 0, w - 1), cy = clampi(y, 0, h - 1);
        return b.images[uint32_t(cy) * params.width + uint32_t(cx)];
    }

    __device__ float3 pixel_ray(uint32_t x, uint32_t y) const {
        return f3((float(x) + 0.5f - params.cx) / params.fx, (float(y) + 0.5f - params.cy) / params.fy, 1.0f);
    }

    __device__ __noinline__ void load_patch(uint32_t x, uint32_t y) {
        const float center = ref_gray(int(x), int(y));
        patch_center = center;
        const float inv_color = 1.0f / (2.0f * params.sigma_color * params.sigma_color);
        const float inv_spatial = 1.0f / (2.0f * params.sigma_spatial * params.sigma_spatial);
        uint32_t n = 0;
        float sum_w = 0.f, sum = 0.f, sum_sq = 0.f;
        const int r = params.window_radius;
        const int step = max(params.window_step, 1);
        for (int dy = -r; dy <= r; dy += step) {
            for (int dx = -r; dx <= r; dx += step) {
                if (n >= MAX_SAMPLES) break;
                const float value = ref_gray(int(x) + dx, int(y) + dy) - center;
                const float weight = expf(-value * value * inv_color - float(dx * dx + dy * dy) * inv_spatial);
                patch_values[n] = value;
                patch_weights[n] = weight;
                sum_w += weight;
                sum += weight * value;
                sum_sq += weight * value * value;
                n += 1u;
            }
        }
        patch_n = n;
        patch_sum_w = sum_w;
        if (sum_w < MIN_WEIGHT_SUM) {
            patch_mean = 0.f;
            patch_var = 0.f;
            return;
        }
        patch_mean = sum / sum_w;
        patch_var = sum_sq / sum_w - patch_mean * patch_mean;
    }

    __device__ void uniform_patch() {
        float sum = 0.f, sum_sq = 0.f;
        for (uint32_t k = 0; k < patch_n; k++) {
            patch_weights[k] = 1.0f;
            sum += patch_values[k];
            sum_sq += patch_values[k] * patch_values[k];
        }
        const float n = float(patch_n);
        patch_sum_w = n;
        patch_mean = sum / n;
        patch_var = sum_sq / n - patch_mean * patch_mean;
    }

    __device__ static float plane_depth(float4 plane, float3 ray) {
        const float denom = dot(xyz(plane), ray);
        if (denom >= -1e-6f) return -1.0f;
        const float d = plane.w / denom;
        if (!(d > 0.0f) || d > 3.0e38f) return -1.0f;
        return d;
    }

    // Source j's grey level at pixel coordinates (u, v) (pixel centres at
    // +0.5), clamped to its image, bilinearly: the texture unit's lookup
    // (linear filter, clamp to edge) in f32.
    __device__ float source_gray(uint32_t j, const MvsSource& s, float u, float v) const {
        const float x = clampf(u, 0.5f, float(s.dims[0]) - 0.5f) - 0.5f;
        const float y = clampf(v, 0.5f, float(s.dims[1]) - 0.5f) - 0.5f;
        const int tw = int(b.texture_width), th = int(b.texture_height);
        const float fx0 = floorf(x), fy0 = floorf(y);
        const float ax = x - fx0, ay = y - fy0;
        const int x0 = clampi(int(fx0), 0, tw - 1), y0 = clampi(int(fy0), 0, th - 1);
        const int x1 = min(x0 + 1, tw - 1), y1 = min(y0 + 1, th - 1);
        const float* layer = b.layers + size_t(j) * size_t(tw) * size_t(th);
        const float v00 = __ldg(layer + size_t(y0) * tw + x0), v01 = __ldg(layer + size_t(y0) * tw + x1);
        const float v10 = __ldg(layer + size_t(y1) * tw + x0), v11 = __ldg(layer + size_t(y1) * tw + x1);
        const float top = v00 + (v01 - v00) * ax;
        const float bottom = v10 + (v11 - v10) * ax;
        return top + (bottom - top) * ay;
    }

    __device__ static float2 warp_sample(const MvsSource& s, float3 a, float3 bx, float3 by, float dx, float dy) {
        const float3 p = a + dx * bx + dy * by;
        if (p.z <= 0.0f) return f2(-1.0f, -1.0f);
        return f2(s.k[0] * p.x / p.z + s.k[2], s.k[1] * p.y / p.z + s.k[3]);
    }

    __device__ __noinline__ float photometric_cost(uint32_t j, float3 ray, float4 plane) const {
        if (fabsf(plane.w) < 1e-12f) return MAX_COST;
        const MvsSource& s = b.sources[j];
        const float3 t = f3(s.r0[3], s.r1[3], s.r2[3]);
        const float inv_rho = 1.0f / plane.w;
        const float3 n = xyz(plane);
        const float3 m0 = f3(s.r0[0], s.r0[1], s.r0[2]) + t.x * n * inv_rho;
        const float3 m1 = f3(s.r1[0], s.r1[1], s.r1[2]) + t.y * n * inv_rho;
        const float3 m2 = f3(s.r2[0], s.r2[1], s.r2[2]) + t.z * n * inv_rho;
        const float3 a = f3(dot(m0, ray), dot(m1, ray), dot(m2, ray));
        const float3 bx = f3(m0.x, m1.x, m2.x) / params.fx;
        const float3 by = f3(m0.y, m1.y, m2.y) / params.fy;
        if (a.z <= 0.0f) return MAX_COST;
        const float cu = s.k[0] * a.x / a.z + s.k[2];
        const float cv = s.k[1] * a.y / a.z + s.k[3];
        if (cu < 0.0f || cv < 0.0f || cu >= float(s.dims[0]) || cv >= float(s.dims[1])) return MAX_COST;
        float center = MAX_COST;
        if (patch_var >= 1e-5f) {
            float s_w = 0.f, s_ww = 0.f, s_rw = 0.f;
            bool ok = true;
            const int r = params.window_radius;
            const int step = max(params.window_step, 1);
            uint32_t k = 0;
            for (int dy = -r; dy <= r; dy += step) {
                for (int dx = -r; dx <= r; dx += step) {
                    if (k >= patch_n) break;
                    const float3 p = a + float(dx) * bx + float(dy) * by;
                    if (p.z <= 0.0f) {
                        ok = false;
                        break;
                    }
                    const float u = s.k[0] * p.x / p.z + s.k[2];
                    const float v = s.k[1] * p.y / p.z + s.k[3];
                    const float value = source_gray(j, s, u, v) - patch_center;
                    const float weight = patch_weights[k];
                    s_w += weight * value;
                    s_ww += weight * value * value;
                    s_rw += weight * patch_values[k] * value;
                    k += 1u;
                }
                if (!ok) break;
            }
            const float mean = s_w / patch_sum_w;
            const float var_s = s_ww / patch_sum_w - mean * mean;
            if (ok && var_s >= 1e-5f) {
                const float cov = s_rw / patch_sum_w - patch_mean * mean;
                const float ncc = cov / sqrtf(patch_var * var_s);
                center = clampf(1.0f - ncc, 0.0f, MAX_COST);
            }
        }
        if (anchor_count == 0u) return center;
        float sum = 0.f;
        for (uint32_t q = 0; q < anchor_count; q++) {
            float cost = MAX_COST;
            bool ok = anchor_vars[q] >= 1e-5f;
            float s1 = 0.f, s2 = 0.f, sr = 0.f;
            for (uint32_t m = 0; m < ANCHOR_SAMPLES; m++) {
                if (!ok) break;
                const float2 o = anchor_offsets[q * ANCHOR_SAMPLES + m];
                const float2 uv = warp_sample(s, a, bx, by, o.x, o.y);
                if (uv.x < 0.0f || uv.y < 0.0f || uv.x >= float(s.dims[0]) || uv.y >= float(s.dims[1])) {
                    ok = false;
                    break;
                }
                const float value = source_gray(j, s, uv.x, uv.y) - anchor_centers[q];
                s1 += value;
                s2 += value * value;
                sr += anchor_values[q * ANCHOR_SAMPLES + m] * value;
            }
            if (ok) {
                const float nn = float(ANCHOR_SAMPLES);
                const float mean = s1 / nn;
                const float var_s = s2 / nn - mean * mean;
                if (var_s >= 1e-5f) {
                    const float cov = sr / nn - anchor_means[q] * mean;
                    cost = clampf(1.0f - cov / sqrtf(anchor_vars[q] * var_s), 0.0f, MAX_COST);
                }
            }
            sum += cost;
        }
        return params.center_weight * center + (1.0f - params.center_weight) * fminf(sum / float(anchor_count), MAX_COST);
    }

    __device__ __noinline__ float geometric_cost(uint32_t j, uint32_t x, uint32_t y, float3 ray, float4 plane) const {
        if (params.geometric == 0u) return 0.0f;
        const float delta = params.geom_delta;
        const float lambda = params.geom_lambda;
        const float depth = plane_depth(plane, ray);
        if (depth <= 0.0f) return lambda * delta;
        const MvsSource& s = b.sources[j];
        const float3 t = f3(s.r0[3], s.r1[3], s.r2[3]);
        const float3 point = ray * depth;
        const float3 p = f3(dot(f3(s.r0[0], s.r0[1], s.r0[2]), point), dot(f3(s.r1[0], s.r1[1], s.r1[2]), point),
                            dot(f3(s.r2[0], s.r2[1], s.r2[2]), point)) +
                         t;
        if (p.z <= 0.0f) return lambda * delta;
        const float u = s.k[0] * p.x / p.z + s.k[2];
        const float v = s.k[1] * p.y / p.z + s.k[3];
        if (u < 0.0f || v < 0.0f || u >= float(s.dims[0]) || v >= float(s.dims[1])) return lambda * delta;
        const float source_depth = b.priors[s.dims[3] + uint32_t(v) * s.dims[0] + uint32_t(u)];
        if (source_depth <= 0.0f) return lambda * delta;
        const float3 back = f3((u - s.k[2]) / s.k[0] * source_depth, (v - s.k[3]) / s.k[1] * source_depth, source_depth);
        const float3 d = back - t;
        const float3 q = f3(s.r0[0] * d.x + s.r1[0] * d.y + s.r2[0] * d.z, s.r0[1] * d.x + s.r1[1] * d.y + s.r2[1] * d.z,
                            s.r0[2] * d.x + s.r1[2] * d.y + s.r2[2] * d.z);
        if (q.z <= 0.0f) return lambda * delta;
        const float du = params.fx * q.x / q.z + params.cx - (float(x) + 0.5f);
        const float dv = params.fy * q.y / q.z + params.cy - (float(y) + 0.5f);
        return lambda * fminf(sqrtf(du * du + dv * dv), delta);
    }

    __device__ float3 random_normal(float3 ray) {
        const float z = 2.0f * uniform01() - 1.0f;
        const float phi = 6.2831855f * uniform01();
        const float s = sqrtf(fmaxf(1.0f - z * z, 0.0f));
        const float3 n = f3(s * cosf(phi), s * sinf(phi), z);
        if (dot(n, ray) > 0.0f) return -n;
        return n;
    }

    __device__ float3 perturb_normal(float3 n, float3 ray, float amount) {
        // The three draws in order (x, y, z), as WGSL evaluates the
        // constructor's arguments.
        const float px = n.x + (uniform01() - 0.5f) * amount;
        const float py = n.y + (uniform01() - 0.5f) * amount;
        const float pz = n.z + (uniform01() - 0.5f) * amount;
        const float3 p = f3(px, py, pz);
        const float3 q = p / sqrtf(dot(p, p));
        if (dot(q, ray) >= 0.0f) return n;
        return q;
    }

    __device__ static float4 plane_from_depth(float depth, float3 n, float3 ray) { return f4(n, depth * dot(n, ray)); }

    __device__ __noinline__ float top_k_cost(uint32_t x, uint32_t y, float3 ray, float4 plane) {
        const uint32_t n = params.num_sources;
        top_k_views = 0u;
        if (n == 0u) return MAX_COST;
        const uint32_t k = min(max(params.init_top_k, 1u), min(n, 8u));
        float best[8];
        for (uint32_t i = 0; i < 8u; i++) best[i] = 3.0e38f;
        float view_costs[32] = {};
        for (uint32_t j = 0; j < n; j++) {
            float c = photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
            view_costs[j] = c;
            for (uint32_t i = 0; i < k; i++) {
                if (c < best[i]) {
                    const float tmp = best[i];
                    best[i] = c;
                    c = tmp;
                }
            }
        }
        float sum = 0.f;
        for (uint32_t i = 0; i < k; i++) sum += best[i];
        for (uint32_t j = 0; j < n; j++)
            if (view_costs[j] <= best[k - 1u]) top_k_views |= 1u << j;
        return sum / float(k);
    }

    __device__ float view_cost(uint32_t j, uint32_t x, uint32_t y, float3 ray, float4 plane) const {
        return photometric_cost(j, ray, plane) + geometric_cost(j, x, y, ray, plane);
    }

    // ---- deformable patches ----

    __device__ __noinline__ uint32_t classify_curve(const float* curve, uint32_t len) const {
        const int center = int(len / 2u);
        int lowest = -1;
        float low = 3.0e38f;
        uint32_t peaks = 0;
        for (uint32_t i = 2u; i + 2u < len; i++) {
            const float c = curve[i];
            if (curve[i - 1u] > c && curve[i + 1u] > c) {
                peaks += 1u;
                if (c < low) {
                    low = c;
                    lowest = int(i);
                }
            }
        }
        if (lowest < 0) return WEAK;
        if (abs(lowest - center) > params.peak_radius || low > params.max_peak_cost) return WEAK;
        if (peaks == 1u) return low <= params.strong_cost ? STRONG : WEAK;
        float excess = 0.f;
        for (uint32_t i = 2u; i + 2u < len; i++) {
            const float c = curve[i];
            if (int(i) != lowest && curve[i - 1u] > c && curve[i + 1u] > c) excess += (c - low) * (c - low);
        }
        if (sqrtf(excess) / float(peaks - 1u) > params.peak_spread) return STRONG;
        return WEAK;
    }

    __device__ uint32_t draws_below(uint32_t n) {
        draw_state = draw_state ^ (draw_state << 13u);
        draw_state = draw_state ^ (draw_state >> 17u);
        draw_state = draw_state ^ (draw_state << 5u);
        return draw_state % max(n, 1u);
    }

    __device__ static bool in_triangle(float2 a, float2 bb, float2 c, float2 p) {
        if (length(a - bb) <= 2.0f || length(bb - c) <= 2.0f || length(c - a) <= 2.0f) return false;
        const float area = (bb.x - a.x) * (c.y - a.y) - (bb.y - a.y) * (c.x - a.x);
        if (fabsf(area) < 1e-6f) return false;
        const float d1 = (bb.x - a.x) * (p.y - a.y) - (bb.y - a.y) * (p.x - a.x);
        const float d2 = (c.x - bb.x) * (p.y - bb.y) - (c.y - bb.y) * (p.x - bb.x);
        const float d3 = (a.x - c.x) * (p.y - c.y) - (a.y - c.y) * (p.x - c.x);
        const bool has_neg = d1 < 0.0f || d2 < 0.0f || d3 < 0.0f;
        const bool has_pos = d1 > 0.0f || d2 > 0.0f || d3 > 0.0f;
        return !(has_neg && has_pos);
    }

    __device__ static float4 plane_through(float3 a, float3 bb, float3 c) {
        const float3 n = cross(a - c, bb - c);
        const float norm = length(n);
        if (norm < 1e-12f) return make_float4(0.f, 0.f, 0.f, 0.f);
        const float3 u = n / norm;
        return f4(u, -dot(u, a));
    }

    __device__ float4 anchor_point(uint32_t q) const {
        const float3 ray = pixel_ray(q % params.width, q / params.width);
        const float d = plane_depth(b.planes[q], ray);
        if (d <= 0.0f) return make_float4(0.f, 0.f, 0.f, 0.f);
        return f4(ray * d, 1.0f);
    }

    __device__ float2 pixel_xy(uint32_t q) const { return f2(float(q % params.width), float(q / params.width)); }

    __device__ __noinline__ void load_anchor_patches(uint32_t x, uint32_t y) {
        const uint32_t i = y * params.width + x;
        anchor_count = 0u;
        for (uint32_t k = 0; k < MAX_ANCHORS; k++) {
            const uint32_t a = b.anchors[i * MAX_ANCHORS + k];
            if (a == NO_VIEW) break;
            const int ax = int(a % params.width);
            const int ay = int(a / params.width);
            const float center = ref_gray(ax, ay);
            float s1 = 0.f, s2 = 0.f;
            uint32_t m = 0;
            for (int dy = -5; dy <= 5; dy += 5) {
                for (int dx = -5; dx <= 5; dx += 5) {
                    const int sx = clampi(ax + dx, 0, int(params.width) - 1);
                    const int sy = clampi(ay + dy, 0, int(params.height) - 1);
                    const float value = ref_gray(sx, sy) - center;
                    anchor_offsets[k * ANCHOR_SAMPLES + m] = f2(float(sx - int(x)), float(sy - int(y)));
                    anchor_values[k * ANCHOR_SAMPLES + m] = value;
                    s1 += value;
                    s2 += value * value;
                    m += 1u;
                }
            }
            const float nn = float(ANCHOR_SAMPLES);
            anchor_centers[k] = center;
            anchor_means[k] = s1 / nn;
            anchor_vars[k] = s2 / nn - (s1 / nn) * (s1 / nn);
            anchor_count = k + 1u;
        }
    }

    __device__ __noinline__ float4 fitted_plane(uint32_t x, uint32_t y, float3 ray) {
        const uint32_t i = y * params.width + x;
        uint32_t pixels[8] = {};
        float3 points[8];
        for (int k = 0; k < 8; k++) points[k] = f3(0.f, 0.f, 0.f);
        uint32_t n = 0;
        for (uint32_t k = 0; k < MAX_ANCHORS; k++) {
            const uint32_t a = b.anchors[i * MAX_ANCHORS + k];
            if (a == NO_VIEW) break;
            const float4 p = anchor_point(a);
            if (p.w != 0.0f) {
                pixels[n] = a;
                points[n] = xyz(p);
                n += 1u;
            }
        }
        if (n < 3u) return make_float4(0.f, 0.f, 0.f, 0.f);
        const float2 center = f2(float(x), float(y));
        float4 best = make_float4(0.f, 0.f, 0.f, 0.f);
        float best_score = 3.0e38f;
        bool found = false;
        for (uint32_t it = 0; it < 50u; it++) {
            const uint32_t ia = draws_below(n);
            const uint32_t ib = draws_below(n);
            const uint32_t ic = draws_below(n);
            if (ia == ib || ib == ic || ia == ic) continue;
            if (!in_triangle(pixel_xy(pixels[ia]), pixel_xy(pixels[ib]), pixel_xy(pixels[ic]), center)) continue;
            const float4 plane = plane_through(points[ia], points[ib], points[ic]);
            if (is_zero3(plane)) continue;
            float score = 0.f;
            for (uint32_t k = 0; k < n; k++)
                if (k != ia && k != ib && k != ic) score += fabsf(dot(xyz(plane), points[k]) + plane.w);
            if (!found || score < best_score) {
                found = true;
                best_score = score;
                best = plane;
                if (score == 0.0f) break;
            }
        }
        if (!found) return make_float4(0.f, 0.f, 0.f, 0.f);
        float4 plane = f4(xyz(best), -best.w);
        if (dot(xyz(plane), ray) > 0.0f) plane = make_float4(-plane.x, -plane.y, -plane.z, -plane.w);
        return plane;
    }

    __device__ float hypothesis_score(float cost, float4 plane, float3 ray, float4 prior_plane, float prior_depth) const {
        if (prior_depth <= 0.0f) return -cost;
        const float depth = plane_depth(plane, ray);
        if (depth <= 0.0f) return -3.0e38f;
        const float dd = depth - prior_depth;
        const float angle = acosf(clampf(dot(xyz(plane), xyz(prior_plane)), -1.0f, 1.0f));
        return expf(-cost * cost * params.prior_inv_alpha) *
               (params.prior_gamma +
                expf(-dd * dd * params.prior_depth_weight) * expf(-angle * angle * params.prior_angle_weight));
    }
};

__device__ inline int2 region_offset(uint32_t region, uint32_t k) {
    int2 up;
    if (region < 4u) {
        const int level = int((k + 1u) / 2u);
        int dx = level;
        if (k % 2u == 1u) dx = -level;
        up = make_int2(dx, -(level + 1));
    } else {
        up = make_int2(0, -(2 * int(k) + 3));
    }
    switch (region % 4u) {
        case 0u: return up;
        case 1u: return make_int2(up.x, -up.y);
        case 2u: return make_int2(up.y, up.x);
        default: return make_int2(-up.y, up.x);
    }
}

__global__ void __launch_bounds__(128) init_kernel(MvsParams params, Bufs b) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;
    Pixel px(params, b);
    const uint32_t i = y * params.width + x;
    const float3 ray = px.pixel_ray(x, y);
    float4 plane;
    const float4 prior = b.init_planes[i];
    if (prior.w > 0.0f) {
        plane = Pixel::plane_from_depth(prior.w, xyz(prior), ray);
    } else {
        px.rng_init(x, y, 0xffffffffu);
        const float depth = params.depth_min + px.uniform01() * (params.depth_max - params.depth_min);
        plane = Pixel::plane_from_depth(depth, px.random_normal(ray), ray);
    }
    px.load_patch(x, y);
    b.planes[i] = plane;
    b.costs[i] = px.top_k_cost(x, y, ray, plane);
    b.best_views[i] = params.view_selection == 1u ? px.top_k_views : NO_VIEW;
    if (params.restore == 1u) {
        b.pre_costs[i] = b.costs[i];
    } else if (params.restore == 2u) {
        b.pre_costs[i] = px.top_k_cost(x, y, ray, make_float4(0.f, 0.f, 0.f, 1.f));
    }
}

__global__ void __launch_bounds__(128) classify_kernel(MvsParams params, Bufs b) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t w = params.width, h = params.height;
    if (x >= w || y >= h) return;
    const uint32_t i = y * w + x;
    if (params.num_sources == 0u || x < BORDER || y < BORDER || x + BORDER >= w || y + BORDER >= h) {
        b.classes[i] = UNKNOWN;
        return;
    }
    Pixel px(params, b);
    const float3 ray = px.pixel_ray(x, y);
    const float4 plane = b.planes[i];
    const float depth = Pixel::plane_depth(plane, ray);
    if (depth <= 0.0f) {
        b.classes[i] = UNKNOWN;
        return;
    }
    px.load_patch(x, y);
    const uint32_t n = params.num_sources;
    const uint32_t k = min(max(params.init_top_k, 1u), min(n, 8u));
    float center_costs[32] = {};
    for (uint32_t j = 0; j < n; j++) center_costs[j] = px.view_cost(j, x, y, ray, plane);
    uint32_t views[8] = {};
    uint32_t taken = 0;
    for (uint32_t q = 0; q < k; q++) {
        uint32_t best = NO_VIEW;
        for (uint32_t j = 0; j < n; j++) {
            if (((taken >> j) & 1u) == 1u) continue;
            if (best == NO_VIEW || center_costs[j] < center_costs[best]) best = j;
        }
        views[q] = best;
        taken |= 1u << best;
    }
    px.uniform_patch();
    const float disparity = params.fx * params.baseline / depth;
    const uint32_t len = min(uint32_t(2 * params.sweep + 1), MAX_CURVE);
    float curve[61] = {};
    for (uint32_t m = 0; m < len; m++) {
        const float step = disparity + float(int(m) - params.sweep);
        const float d = params.fx * params.baseline / step;
        if (step <= 0.0f || !(d >= params.depth_min && d <= params.depth_max)) {
            curve[m] = MAX_COST;
        } else {
            const float4 p = Pixel::plane_from_depth(d, xyz(plane), ray);
            float sum = 0.f;
            for (uint32_t q = 0; q < k; q++) sum += px.view_cost(views[q], x, y, ray, p);
            curve[m] = fminf(sum / float(k), MAX_COST);
        }
    }
    b.classes[i] = px.classify_curve(curve, len);
}

__global__ void __launch_bounds__(128) find_anchors_kernel(MvsParams params, Bufs b) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t w = params.width, h = params.height;
    if (x >= w || y >= h) return;
    const uint32_t i = y * w + x;
    for (uint32_t k = 0; k < MAX_ANCHORS; k++) b.anchors[i * MAX_ANCHORS + k] = NO_VIEW;
    if (b.classes[i] != WEAK) return;
    Pixel px(params, b);
    const uint32_t sectors = min(max(params.sectors, 1u), MAX_SECTORS);
    const float half = 3.14159265f / float(sectors);
    const float cos_half = cosf(half);
    uint32_t found[32] = {};
    uint32_t count = 0;
    for (uint32_t sector = 0; sector < sectors; sector++) {
        const float angle = 2.0f * half * float(sector);
        const float2 dir = f2(cosf(angle), sinf(angle));
        float radius = 2.0f;
        while (true) {
            const float qx = floorf(float(x) + 0.5f + dir.x * radius);
            const float qy = floorf(float(y) + 0.5f + dir.y * radius);
            if (qx < 0.0f || qy < 0.0f || qx >= float(w) || qy >= float(h)) break;
            const uint32_t ux = uint32_t(qx), uy = uint32_t(qy);
            bool taken = false;
            if (ux >= BORDER && uy >= BORDER && ux + BORDER < w && uy + BORDER < h) {
                const uint32_t q = uy * w + ux;
                uint32_t a = NO_VIEW;
                if (b.classes[q] == STRONG) {
                    a = q;
                } else {
                    a = b.nearest[q];
                }
                if (a != NO_VIEW) {
                    const float2 off = px.pixel_xy(a) - f2(float(x), float(y));
                    const float norm = length(off);
                    if (norm > 0.0f && dot(off, dir) / norm > cos_half) {
                        found[count] = a;
                        count += 1u;
                        taken = true;
                    }
                }
            }
            if (taken) break;
            radius = fminf(2.0f * radius, radius + 25.0f);
        }
    }
    if (count < 4u) return;
    const float4 center_point = px.anchor_point(i);
    if (center_point.w == 0.0f) return;
    uint32_t pixels[32] = {};
    float3 points[32];
    for (int k = 0; k < 32; k++) points[k] = f3(0.f, 0.f, 0.f);
    uint32_t n = 0;
    for (uint32_t k = 0; k < count; k++) {
        const float4 p = px.anchor_point(found[k]);
        if (p.w != 0.0f) {
            pixels[n] = found[k];
            points[n] = xyz(p);
            n += 1u;
        }
    }
    if (n < 3u) return;
    const float2 center = f2(float(x), float(y));
    px.draw_state = (params.seed ^ (i * 0x9e3779b9u)) | 1u;
    uint32_t best_inliers = 0;
    float best_center = 3.0e38f;
    float4 best_plane = make_float4(0.f, 0.f, 0.f, 0.f);
    uint3 best_sample = make_uint3(0u, 0u, 0u);
    for (uint32_t it = 0; it < 50u; it++) {
        const uint32_t ia = px.draws_below(n);
        const uint32_t ib = px.draws_below(n);
        const uint32_t ic = px.draws_below(n);
        if (ia == ib || ib == ic || ia == ic) continue;
        if (!Pixel::in_triangle(px.pixel_xy(pixels[ia]), px.pixel_xy(pixels[ib]), px.pixel_xy(pixels[ic]), center))
            continue;
        const float4 plane = Pixel::plane_through(points[ia], points[ib], points[ic]);
        if (is_zero3(plane)) continue;
        uint32_t inliers = 0;
        for (uint32_t k = 0; k < n; k++)
            if (fabsf(dot(xyz(plane), points[k]) + plane.w) < params.ransac_threshold) inliers += 1u;
        if (inliers < params.min_inliers) continue;
        const float cd = fabsf(dot(xyz(plane), xyz(center_point)) + plane.w);
        if (best_inliers == 0u || inliers > best_inliers || (inliers == best_inliers && cd < best_center)) {
            best_inliers = inliers;
            best_center = cd;
            best_plane = plane;
            best_sample = make_uint3(ia, ib, ic);
        }
    }
    if (best_inliers == 0u) return;
    uint32_t order[32] = {};
    float keys[32] = {};
    uint32_t m = 0;
    for (uint32_t k = 0; k < n; k++) {
        const float d = fabsf(dot(xyz(best_plane), points[k]) + best_plane.w);
        if (d < params.ransac_threshold) {
            float key = d;
            if (k == best_sample.x || k == best_sample.y || k == best_sample.z) key = -1.0f;
            uint32_t at = m;
            while (at > 0u && keys[at - 1u] > key) {
                keys[at] = keys[at - 1u];
                order[at] = order[at - 1u];
                at -= 1u;
            }
            keys[at] = key;
            order[at] = k;
            m += 1u;
        }
    }
    for (uint32_t k = 0; k < min(m, MAX_ANCHORS); k++) b.anchors[i * MAX_ANCHORS + k] = pixels[order[k]];
    b.classes[i] = 3u;
}

__global__ void __launch_bounds__(128) propagate_kernel(MvsParams params, Bufs b) {
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t x = 2u * (blockIdx.x * blockDim.x + threadIdx.x) + ((y + params.color) % 2u);
    if (x >= params.width || y >= params.height) return;
    const uint32_t w = params.width, h = params.height;
    const uint32_t n = params.num_sources;
    const uint32_t i = y * w + x;
    if (n == 0u) {
        b.costs[i] = MAX_COST;
        return;
    }
    Pixel px(params, b);
    const float3 ray = px.pixel_ray(x, y);
    px.load_patch(x, y);
    px.anchor_count = 0u;
    if (params.deformable == 1u && b.anchors[i * MAX_ANCHORS] != NO_VIEW) px.load_anchor_patches(x, y);
    const bool weak = px.anchor_count > 0u;
    float4 prior_plane = make_float4(0.f, 0.f, 0.f, 0.f);
    float prior_depth = -1.0f;
    if (params.planar_prior == 1u) {
        prior_plane = b.prior_planes[i];
        if (!is_zero3(prior_plane)) prior_depth = Pixel::plane_depth(prior_plane, ray);
    }

    // Adaptive checkerboard sampling.
    float4 cand[9];
    bool valid[9];
    for (int k = 0; k < 9; k++) {
        cand[k] = make_float4(0.f, 0.f, 0.f, 0.f);
        valid[k] = false;
    }
    for (uint32_t region = 0; region < 8u; region++) {
        if (weak) {
            valid[region] = region < px.anchor_count;
            if (valid[region]) cand[region] = b.planes[b.anchors[i * MAX_ANCHORS + region]];
            continue;
        }
        const uint32_t count = region >= 4u ? 11u : 7u;
        float best_cost = 3.0e38f;
        uint32_t best_index = NO_VIEW;
        for (uint32_t k = 0; k < count; k++) {
            const int2 o = region_offset(region, k);
            const int sx = int(x) + o.x, sy = int(y) + o.y;
            if (sx < 0 || sy < 0 || sx >= int(w) || sy >= int(h)) continue;
            const uint32_t j = uint32_t(sy) * w + uint32_t(sx);
            if (best_index == NO_VIEW || b.costs[j] < best_cost) {
                best_cost = b.costs[j];
                best_index = j;
            }
        }
        valid[region] = best_index != NO_VIEW;
        if (valid[region]) cand[region] = b.planes[best_index];
    }
    const float4 current = b.planes[i];
    cand[8] = current;
    valid[8] = true;

    // Multi-hypothesis joint view selection, accumulated view by view.
    const uint32_t previous = b.best_views[i];
    const bool sampled = params.view_selection == 1u;
    float weights[32] = {};
    float agg_w[9] = {};
    float agg_u[9] = {};
    float weight_sum = 0.f;
    uint32_t new_best = NO_VIEW;
    float new_best_weight = 0.f;
    float probs[32] = {};
    const float two_beta_sq = 2.0f * params.beta * params.beta;
    for (uint32_t j = 0; j < n; j++) {
        float row_cost[8] = {};
        uint32_t good = 0, bad = 0;
        float confidence = 0.f;
        for (uint32_t r = 0; r < 8u; r++) {
            if (!valid[r]) continue;
            const float m = px.photometric_cost(j, ray, cand[r]);
            if (!sampled) row_cost[r] = m + px.geometric_cost(j, x, y, ray, cand[r]);
            if (m < params.tau) {
                good += 1u;
                confidence += expf(-m * m / two_beta_sq);
            } else if (m > params.tau1) {
                bad += 1u;
            }
        }
        if (sampled) {
            if (good > params.n1 && bad < params.n2) {
                probs[j] = confidence / float(good);
            } else if (bad < params.n2) {
                probs[j] = expf(-params.tau * params.tau / 0.32f);
            }
            continue;
        }
        const bool selected = good > params.n1 && bad < params.n2;
        const bool is_previous = previous == j;
        float weight = 0.f;
        if (selected) {
            weight = confidence / float(good);
            if (is_previous) weight = 2.0f * weight;
        } else if (is_previous) {
            weight = 0.2f;
        }
        weights[j] = weight;
        weight_sum += weight;
        if (weight > new_best_weight) {
            new_best_weight = weight;
            new_best = j;
        }
        for (uint32_t r = 0; r < 8u; r++) {
            if (valid[r]) {
                agg_w[r] += weight * row_cost[r];
                agg_u[r] += row_cost[r];
            }
        }
        if (weight > 0.0f) agg_w[8] += weight * px.view_cost(j, x, y, ray, current);
    }
    uint32_t sampled_views = 0;
    if (sampled) {
        float prior[32] = {};
        for (uint32_t d = 0; d < 4u; d++) {
            int2 o = make_int2(0, -1);
            switch (d) {
                case 1u: o = make_int2(0, 1); break;
                case 2u: o = make_int2(-1, 0); break;
                case 3u: o = make_int2(1, 0); break;
                default: break;
            }
            const int sx = int(x) + o.x, sy = int(y) + o.y;
            if (sx < 0 || sy < 0 || sx >= int(w) || sy >= int(h)) continue;
            const uint32_t bits = b.best_views[uint32_t(sy) * w + uint32_t(sx)];
            for (uint32_t j = 0; j < n; j++) {
                if (((bits >> j) & 1u) == 1u) {
                    prior[j] += 0.9f;
                } else {
                    prior[j] += 0.1f;
                }
            }
        }
        float cdf[32] = {};
        float total = 0.f;
        for (uint32_t j = 0; j < n; j++) {
            total += probs[j] * prior[j];
            cdf[j] = total;
        }
        if (total > 0.0f) {
            px.rng_init(x, y, SAMPLING_STREAM | (params.iteration * 2u + params.color));
            for (uint32_t sample = 0; sample < VIEW_SAMPLES; sample++) {
                const float u = px.uniform01() * total;
                for (uint32_t j = 0; j < n; j++) {
                    if (cdf[j] > u) {
                        weights[j] += 1.0f;
                        sampled_views |= 1u << j;
                        break;
                    }
                }
            }
        }
        for (uint32_t j = 0; j < n; j++) weight_sum += weights[j];
        for (uint32_t j = 0; j < n; j++) {
            if (weight_sum > 0.0f && weights[j] <= 0.0f) continue;
            for (uint32_t r = 0; r < 8u; r++) {
                if (valid[r]) {
                    const float c = px.view_cost(j, x, y, ray, cand[r]);
                    agg_w[r] += weights[j] * c;
                    agg_u[r] += c;
                }
            }
            if (weight_sum > 0.0f) agg_w[8] += weights[j] * px.view_cost(j, x, y, ray, current);
        }
    }
    const bool uniform_weights = weight_sum <= 0.0f;
    if (uniform_weights) {
        weight_sum = float(n);
        new_best = 0u;
        for (uint32_t j = 0; j < n; j++) agg_u[8] += px.view_cost(j, x, y, ray, current);
    }
    float4 best_plane = current;
    float best_cost = uniform_weights ? agg_u[8] / weight_sum : agg_w[8] / weight_sum;
    const float current_cost = best_cost;
    float best_score = px.hypothesis_score(best_cost, best_plane, ray, prior_plane, prior_depth);
    bool adopted = false;
    for (uint32_t r = 0; r < 8u; r++) {
        if (!valid[r]) continue;
        const float c = uniform_weights ? agg_u[r] / weight_sum : agg_w[r] / weight_sum;
        const float s = px.hypothesis_score(c, cand[r], ray, prior_plane, prior_depth);
        if (s > best_score && Pixel::plane_depth(cand[r], ray) > 0.0f) {
            best_cost = c;
            best_score = s;
            best_plane = cand[r];
            adopted = true;
        }
    }

    // Refinement; with a prior, the random depth and normal are drawn
    // around it.
    px.rng_init(x, y, params.iteration * 2u + params.color);
    float depth_now = Pixel::plane_depth(best_plane, ray);
    if (depth_now <= 0.0f) depth_now = 0.5f * (params.depth_min + params.depth_max);
    const float3 normal_now = xyz(best_plane);
    float depth_random;
    float3 normal_random;
    if (prior_depth > 0.0f) {
        depth_random = prior_depth + (px.uniform01() - 0.5f) * 6.0f * params.prior_depth_sigma;
        normal_random = px.perturb_normal(xyz(prior_plane), ray, params.prior_angle_sigma);
    } else {
        depth_random = params.depth_min + px.uniform01() * (params.depth_max - params.depth_min);
        normal_random = px.random_normal(ray);
    }
    const float depth_perturbed = depth_now * (1.0f + (px.uniform01() - 0.5f) * params.depth_perturbation);
    const float3 normal_perturbed = px.perturb_normal(normal_now, ray, params.normal_perturbation);
    if (weak) {
        px.draw_state = px.rng_state | 1u;
        const float4 plane = px.fitted_plane(x, y, ray);
        const float depth = Pixel::plane_depth(plane, ray);
        if (!is_zero3(plane) && depth >= params.depth_min && depth <= params.depth_max) {
            float sum = 0.f;
            for (uint32_t j = 0; j < n; j++) {
                if (uniform_weights) {
                    sum += px.view_cost(j, x, y, ray, plane);
                } else if (weights[j] > 0.0f) {
                    sum += weights[j] * px.view_cost(j, x, y, ray, plane);
                }
            }
            const float c = sum / weight_sum;
            const float s = px.hypothesis_score(c, plane, ray, prior_plane, prior_depth);
            if (s > best_score) {
                best_cost = c;
                best_score = s;
                best_plane = plane;
            }
        }
    }
    for (uint32_t hyp = 0; hyp < 6u; hyp++) {
        float depth = depth_now;
        float3 normal = normal_now;
        switch (hyp) {
            case 0u: depth = depth_random; break;
            case 1u: normal = normal_random; break;
            case 2u:
                depth = depth_random;
                normal = normal_random;
                break;
            case 3u: depth = depth_perturbed; break;
            case 4u: normal = normal_perturbed; break;
            default:
                depth = depth_perturbed;
                normal = normal_perturbed;
                break;
        }
        if (!(depth >= params.depth_min && depth <= params.depth_max)) continue;
        const float4 plane = Pixel::plane_from_depth(depth, normal, ray);
        float sum = 0.f;
        for (uint32_t j = 0; j < n; j++) {
            if (uniform_weights) {
                sum += px.view_cost(j, x, y, ray, plane);
            } else if (weights[j] > 0.0f) {
                sum += weights[j] * px.view_cost(j, x, y, ray, plane);
            }
        }
        const float c = sum / weight_sum;
        const float s = px.hypothesis_score(c, plane, ray, prior_plane, prior_depth);
        if (s > best_score) {
            best_cost = c;
            best_score = s;
            best_plane = plane;
        }
    }
    if (weak) {
        if (params.restore != 0u && best_cost >= current_cost - params.restore_margin) best_plane = current;
        px.anchor_count = 0u;
        float sum = 0.f;
        for (uint32_t j = 0; j < n; j++) {
            if (uniform_weights) {
                sum += px.view_cost(j, x, y, ray, best_plane);
            } else if (weights[j] > 0.0f) {
                sum += weights[j] * px.view_cost(j, x, y, ray, best_plane);
            }
        }
        best_cost = sum / weight_sum;
    } else if (params.restore != 0u && best_cost >= b.pre_costs[i] - params.restore_margin) {
        best_plane = current;
        best_cost = current_cost;
    }
    b.planes[i] = best_plane;
    b.costs[i] = best_cost;
    if (!sampled) {
        b.best_views[i] = new_best;
    } else if (adopted) {
        b.best_views[i] = sampled_views;
    }
}

// ---- Host side ----

struct Buf {
    void* ptr = nullptr;
    size_t cap = 0;
};

struct Ctx {
    std::string error;
    uint32_t w = 0, h = 0;
    Bufs b = {};
    bool begun = false;
    Buf sources, images, layers, priors, planes, costs, best_views, init_planes, prior_planes, pre_costs, classes,
        nearest, anchors;
};

#define MVS_CHECK(ctx, expr)                                                      \
    do {                                                                          \
        cudaError_t err__ = (expr);                                               \
        if (err__ != cudaSuccess) {                                               \
            (ctx)->error = std::string(#expr) + ": " + cudaGetErrorString(err__); \
            return -1;                                                            \
        }                                                                         \
    } while (0)

// A zeroed device buffer of at least `bytes` (16 at least), reused when
// large enough.
int zeroed(Ctx* ctx, Buf& b, size_t bytes, void** out) {
    const size_t want = bytes < 16 ? 16 : bytes;
    if (want > b.cap) {
        if (b.ptr) MVS_CHECK(ctx, cudaFree(b.ptr));
        b.ptr = nullptr;
        b.cap = 0;
        MVS_CHECK(ctx, cudaMalloc(&b.ptr, want));
        b.cap = want;
    }
    MVS_CHECK(ctx, cudaMemset(b.ptr, 0, want));
    *out = b.ptr;
    return 0;
}

int upload(Ctx* ctx, Buf& b, const void* host, size_t bytes, void** out) {
    if (zeroed(ctx, b, bytes, out) != 0) return -1;
    if (host && bytes > 0) MVS_CHECK(ctx, cudaMemcpy(*out, host, bytes, cudaMemcpyHostToDevice));
    return 0;
}

void free_buf(Buf& b) {
    if (b.ptr) cudaFree(b.ptr);
    b.ptr = nullptr;
    b.cap = 0;
}

dim3 pixel_grid(uint32_t w, uint32_t h) { return dim3((w + 15) / 16, (h + 7) / 8); }

int finish(Ctx* ctx) {
    MVS_CHECK(ctx, cudaGetLastError());
    MVS_CHECK(ctx, cudaDeviceSynchronize());
    return 0;
}

}  // namespace

extern "C" {

#if defined(_WIN32)
#define MVS_EXPORT __declspec(dllexport)
#else
#define MVS_EXPORT __attribute__((visibility("default")))
#endif

MVS_EXPORT uint32_t visloc_mvs_cuda_abi_version() { return kAbiVersion; }

MVS_EXPORT void* visloc_mvs_cuda_create() {
    Ctx* ctx = new (std::nothrow) Ctx();
    if (!ctx) return nullptr;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) ctx->error = "no CUDA device";
    return ctx;
}

MVS_EXPORT void visloc_mvs_cuda_destroy(void* handle) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return;
    for (Buf* b : {&ctx->sources, &ctx->images, &ctx->layers, &ctx->priors, &ctx->planes, &ctx->costs,
                   &ctx->best_views, &ctx->init_planes, &ctx->prior_planes, &ctx->pre_costs, &ctx->classes,
                   &ctx->nearest, &ctx->anchors})
        free_buf(*b);
    delete ctx;
}

MVS_EXPORT const char* visloc_mvs_cuda_last_error(void* handle) {
    return handle ? static_cast<Ctx*>(handle)->error.c_str() : "null context";
}

// Upload a pass's inputs and run `init` with `params`.
MVS_EXPORT int visloc_mvs_cuda_begin(void* handle, const MvsInputs* in, const MvsParams* params) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx) return -1;
    if (!in || !params || in->width == 0 || in->height == 0 || !in->layers || !in->image || !in->init_planes ||
        in->num_sources > MAX_SOURCES || (in->num_sources > 0 && !in->sources) ||
        (in->prior_len > 0 && !in->priors)) {
        ctx->error = "begin: a complete pass is required";
        return -1;
    }
    const size_t npix = size_t(in->width) * in->height;
    const size_t layers = size_t(in->num_sources > 0 ? in->num_sources : 1) * in->texture_width * in->texture_height;
    ctx->w = in->width;
    ctx->h = in->height;
    Bufs& b = ctx->b;
    void* p;
    if (upload(ctx, ctx->sources, in->sources, size_t(in->num_sources) * sizeof(MvsSource), &p) != 0) return -1;
    b.sources = static_cast<const MvsSource*>(p);
    if (upload(ctx, ctx->images, in->image, npix * sizeof(float), &p) != 0) return -1;
    b.images = static_cast<const float*>(p);
    if (upload(ctx, ctx->layers, in->layers, layers * sizeof(float), &p) != 0) return -1;
    b.layers = static_cast<const float*>(p);
    b.texture_width = in->texture_width;
    b.texture_height = in->texture_height;
    if (upload(ctx, ctx->priors, in->priors, size_t(in->prior_len) * sizeof(float), &p) != 0) return -1;
    b.priors = static_cast<const float*>(p);
    if (zeroed(ctx, ctx->planes, npix * sizeof(float4), &p) != 0) return -1;
    b.planes = static_cast<float4*>(p);
    if (zeroed(ctx, ctx->costs, npix * sizeof(float), &p) != 0) return -1;
    b.costs = static_cast<float*>(p);
    if (zeroed(ctx, ctx->best_views, npix * sizeof(uint32_t), &p) != 0) return -1;
    b.best_views = static_cast<uint32_t*>(p);
    if (upload(ctx, ctx->init_planes, in->init_planes, npix * sizeof(float4), &p) != 0) return -1;
    b.init_planes = static_cast<const float4*>(p);
    if (upload(ctx, ctx->prior_planes, in->prior_planes, in->prior_planes ? npix * sizeof(float4) : 0, &p) != 0)
        return -1;
    b.prior_planes = static_cast<const float4*>(p);
    if (zeroed(ctx, ctx->pre_costs, in->pre_costs ? npix * sizeof(float) : 0, &p) != 0) return -1;
    b.pre_costs = static_cast<float*>(p);
    if (zeroed(ctx, ctx->classes, in->classes ? npix * sizeof(uint32_t) : 0, &p) != 0) return -1;
    b.classes = static_cast<uint32_t*>(p);
    if (zeroed(ctx, ctx->nearest, in->deformable ? npix * sizeof(uint32_t) : 0, &p) != 0) return -1;
    b.nearest = static_cast<const uint32_t*>(p);
    if (zeroed(ctx, ctx->anchors, in->deformable ? npix * MAX_ANCHORS * sizeof(uint32_t) : 0, &p) != 0) return -1;
    b.anchors = static_cast<uint32_t*>(p);
    ctx->begun = true;
    init_kernel<<<pixel_grid(ctx->w, ctx->h), dim3(16, 8)>>>(*params, b);
    return finish(ctx);
}

// `classify` with `params` on the current planes; the classes (0 weak, 1
// strong, 2 unknown) go to `out` (w x h).
MVS_EXPORT int visloc_mvs_cuda_classify(void* handle, const MvsParams* params, uint32_t* out) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !params || !ctx->begun) return -1;
    classify_kernel<<<pixel_grid(ctx->w, ctx->h), dim3(16, 8)>>>(*params, ctx->b);
    if (finish(ctx) != 0) return -1;
    if (out)
        MVS_CHECK(ctx, cudaMemcpy(out, ctx->b.classes, size_t(ctx->w) * ctx->h * sizeof(uint32_t),
                                  cudaMemcpyDeviceToHost));
    return 0;
}

MVS_EXPORT int visloc_mvs_cuda_set_classes(void* handle, const uint32_t* classes) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !classes || !ctx->begun) return -1;
    MVS_CHECK(ctx, cudaMemcpy(ctx->b.classes, classes, size_t(ctx->w) * ctx->h * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));
    return 0;
}

// Each weak pixel's anchors, from its nearest strong pixels (`nearest`,
// w x h, NO_VIEW for none).
MVS_EXPORT int visloc_mvs_cuda_find_anchors(void* handle, const MvsParams* params, const uint32_t* nearest) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !params || !nearest || !ctx->begun) return -1;
    MVS_CHECK(ctx, cudaMemcpy(const_cast<uint32_t*>(ctx->b.nearest), nearest,
                              size_t(ctx->w) * ctx->h * sizeof(uint32_t), cudaMemcpyHostToDevice));
    find_anchors_kernel<<<pixel_grid(ctx->w, ctx->h), dim3(16, 8)>>>(*params, ctx->b);
    return finish(ctx);
}

// One propagation of the pixels of colour params->color.
MVS_EXPORT int visloc_mvs_cuda_propagate(void* handle, const MvsParams* params) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !params || !ctx->begun) return -1;
    const uint32_t half = (ctx->w + 1) / 2;
    propagate_kernel<<<dim3((half + 15) / 16, (ctx->h + 7) / 8), dim3(16, 8)>>>(*params, ctx->b);
    return finish(ctx);
}

// The planes ((n, rho), 4 floats per pixel) and costs (nullable outputs).
MVS_EXPORT int visloc_mvs_cuda_read(void* handle, float* planes, float* costs) {
    Ctx* ctx = static_cast<Ctx*>(handle);
    if (!ctx || !ctx->begun) return -1;
    const size_t npix = size_t(ctx->w) * ctx->h;
    if (planes) MVS_CHECK(ctx, cudaMemcpy(planes, ctx->b.planes, npix * sizeof(float4), cudaMemcpyDeviceToHost));
    if (costs) MVS_CHECK(ctx, cudaMemcpy(costs, ctx->b.costs, npix * sizeof(float), cudaMemcpyDeviceToHost));
    return 0;
}

}  // extern "C"
