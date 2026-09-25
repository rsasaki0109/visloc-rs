// On-device adaptive density control (the Inria clone / split / prune rule of
// crate::densify, without the host round-trip).
//
//   densify_classify: per gaussian, an action and its output row count
//       0 prune (0 rows), 1 keep (1), 2 clone (keep + a copy, 2), 3 split
//       (two children drawn from the parent, 2); the counts are then prefix
//       summed on the device to give each gaussian its output offset.
//   densify_scatter: one dispatch per parameter group (transforms | opacity |
//       SH), copying rows and Adam moments to the new buffers. Kept rows keep
//       their moments, new rows start at zero; split children get a mean
//       sampled from the parent (hashed normals) and scales / 1.6. An opacity
//       reset clamps every opacity logit and zeroes that group's moments.
//   Action 4 (brush refine): the parent is replaced by itself shifted by -o
//       and a new copy shifted by +o, o = R (N(0, 0.5) * s); both get scales /
//       sqrt(2) and opacity 1 - sqrt(1 - a). Row 0 keeps the parent's
//       moments, row 1 starts at zero.

struct ScatterUniforms {
    n: u32,
    stride: u32,
    // 0 transforms (split resamples means / shrinks scales), 1 opacity, 2 SH.
    kind: u32,
    // Opacity group only: clamp logits to reset_cap and zero the moments.
    reset: u32,
    reset_cap: f32,
    seed: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<uniform> su: ScatterUniforms;
@group(0) @binding(1) var<storage, read> s_actions: array<u32>;
// Inclusive prefix sum of the per-gaussian output counts.
@group(0) @binding(2) var<storage, read> cum: array<u32>;
@group(0) @binding(3) var<storage, read> src_p: array<f32>;
@group(0) @binding(4) var<storage, read> src_m1: array<f32>;
@group(0) @binding(5) var<storage, read> src_m2: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst_p: array<f32>;
@group(0) @binding(7) var<storage, read_write> dst_m1: array<f32>;
@group(0) @binding(8) var<storage, read_write> dst_m2: array<f32>;

fn hash_u32(x: u32) -> u32 {
    // PCG-style integer hash.
    var v = x * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

fn uniform01(key: u32) -> f32 {
    return (f32(hash_u32(key) >> 8u) + 0.5) / 16777216.0;
}

// Standard normal from two hashed uniforms (Box-Muller).
fn normal(key: u32) -> f32 {
    let u1 = uniform01(key * 2u + 1u);
    let u2 = uniform01(key * 2u + 2u);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

// Rotate v by the (normalised) quaternion (w, x, y, z).
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let n = length(q);
    var qq = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    if (n > 1e-12) {
        qq = q / n;
    }
    let u = qq.yzw;
    let t = 2.0 * cross(u, v);
    return v + qq.x * t + cross(u, t);
}

@compute @workgroup_size(256)
fn densify_scatter(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid3.x + gid3.y * nwg.x * 256u;
    if (i >= su.n) {
        return;
    }
    let action = s_actions[i];
    if (action == 0u) {
        return;
    }
    var count = 1u;
    if (action >= 2u) {
        count = 2u;
    }
    let out0 = cum[i] - count;
    let st = su.stride;
    let src = i * st;

    if (su.kind == 0u && action == 4u) {
        let q = vec4<f32>(src_p[src + 3u], src_p[src + 4u], src_p[src + 5u], src_p[src + 6u]);
        let s = exp(vec3<f32>(src_p[src + 7u], src_p[src + 8u], src_p[src + 9u]));
        let mean = vec3<f32>(src_p[src], src_p[src + 1u], src_p[src + 2u]);
        let key = (su.seed ^ (i * 6u)) * 3u;
        let o = quat_rotate(q, vec3<f32>(normal(key), normal(key + 1u), normal(key + 2u)) * 0.5 * s);
        let shrink = log(sqrt(2.0));
        for (var c = 0u; c < 2u; c = c + 1u) {
            let m = select(mean - o, mean + o, c == 1u);
            let d = (out0 + c) * st;
            dst_p[d + 0u] = m.x;
            dst_p[d + 1u] = m.y;
            dst_p[d + 2u] = m.z;
            for (var k = 3u; k < 7u; k = k + 1u) {
                dst_p[d + k] = src_p[src + k];
            }
            for (var k = 7u; k < 10u; k = k + 1u) {
                dst_p[d + k] = src_p[src + k] - shrink;
            }
            for (var k = 0u; k < 10u; k = k + 1u) {
                dst_m1[d + k] = select(src_m1[src + k], 0.0, c == 1u);
                dst_m2[d + k] = select(src_m2[src + k], 0.0, c == 1u);
            }
        }
        return;
    }
    if (su.kind == 1u && action == 4u) {
        let a = 1.0 / (1.0 + exp(-src_p[src]));
        let na = clamp(1.0 - sqrt(max(1.0 - a, 0.0)), 1e-7, 1.0 - 1e-7);
        let logit = log(na / (1.0 - na));
        for (var c = 0u; c < 2u; c = c + 1u) {
            let d = out0 + c;
            dst_p[d] = logit;
            dst_m1[d] = select(src_m1[src], 0.0, c == 1u);
            dst_m2[d] = select(src_m2[src], 0.0, c == 1u);
        }
        return;
    }

    if (su.kind == 0u && action == 3u) {
        // Split: two children N(mean, R diag(s)^2 R^T), scales / 1.6.
        let q = vec4<f32>(src_p[src + 3u], src_p[src + 4u], src_p[src + 5u], src_p[src + 6u]);
        let s = exp(vec3<f32>(src_p[src + 7u], src_p[src + 8u], src_p[src + 9u]));
        let mean = vec3<f32>(src_p[src], src_p[src + 1u], src_p[src + 2u]);
        let shrink = log(1.6);
        for (var c = 0u; c < 2u; c = c + 1u) {
            let key = (su.seed ^ (i * 6u + c * 3u)) * 3u;
            let local = vec3<f32>(normal(key), normal(key + 1u), normal(key + 2u)) * s;
            let m = mean + quat_rotate(q, local);
            let d = (out0 + c) * st;
            dst_p[d + 0u] = m.x;
            dst_p[d + 1u] = m.y;
            dst_p[d + 2u] = m.z;
            for (var k = 3u; k < 7u; k = k + 1u) {
                dst_p[d + k] = src_p[src + k];
            }
            for (var k = 7u; k < 10u; k = k + 1u) {
                dst_p[d + k] = src_p[src + k] - shrink;
            }
            for (var k = 0u; k < 10u; k = k + 1u) {
                dst_m1[d + k] = 0.0;
                dst_m2[d + k] = 0.0;
            }
        }
        return;
    }

    for (var c = 0u; c < count; c = c + 1u) {
        let d = (out0 + c) * st;
        // Row 0 of keep/clone is the original (moments kept); the clone copy
        // and both split children (non-transform groups) start fresh.
        let fresh = c == 1u || action == 3u;
        for (var k = 0u; k < st; k = k + 1u) {
            var v = src_p[src + k];
            var m1 = src_m1[src + k];
            var m2 = src_m2[src + k];
            if (fresh) {
                m1 = 0.0;
                m2 = 0.0;
            }
            if (su.reset == 1u) {
                v = min(v, su.reset_cap);
                m1 = 0.0;
                m2 = 0.0;
            }
            dst_p[d + k] = v;
            dst_m1[d + k] = m1;
            dst_m2[d + k] = m2;
        }
    }
}
