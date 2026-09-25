// brush-style mean noise (3DGS-MCMC flavour): nearly transparent gaussians
// wander so they can re-land where the scene needs them.
//
//   mean += clamp(R (N(0, 1) * s) * w, -max_noise, max_noise),
//   w = clamp((1 - opacity)^100, 0, 1) * lr_mean * noise_weight.

struct NoiseUniforms {
    n: u32,
    seed: u32,
    // lr_mean * noise_weight
    scale: f32,
    max_noise: f32,
};

@group(0) @binding(0) var<uniform> nu: NoiseUniforms;
@group(0) @binding(1) var<storage, read_write> n_transforms: array<f32>;
@group(0) @binding(2) var<storage, read> n_opacity: array<f32>;

fn hash_u32(x: u32) -> u32 {
    var v = x * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

fn uniform01(key: u32) -> f32 {
    return (f32(hash_u32(key) >> 8u) + 0.5) / 16777216.0;
}

fn normal(key: u32) -> f32 {
    let u1 = uniform01(key * 2u + 1u);
    let u2 = uniform01(key * 2u + 2u);
    return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

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
fn mean_noise(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid3.x + gid3.y * nwg.x * 256u;
    if (i >= nu.n) {
        return;
    }
    let a = 1.0 / (1.0 + exp(-n_opacity[i]));
    let w = clamp(pow(1.0 - a, 100.0), 0.0, 1.0) * nu.scale;
    if (w <= 0.0) {
        return;
    }
    let b = i * 10u;
    let q = vec4<f32>(n_transforms[b + 3u], n_transforms[b + 4u], n_transforms[b + 5u], n_transforms[b + 6u]);
    let s = exp(vec3<f32>(n_transforms[b + 7u], n_transforms[b + 8u], n_transforms[b + 9u]));
    let key = (nu.seed ^ (i * 3u)) * 3u;
    let d = quat_rotate(q, vec3<f32>(normal(key), normal(key + 1u), normal(key + 2u)) * s) * w;
    let m = clamp(d, vec3<f32>(-nu.max_noise), vec3<f32>(nu.max_noise));
    n_transforms[b + 0u] = n_transforms[b + 0u] + m.x;
    n_transforms[b + 1u] = n_transforms[b + 1u] + m.y;
    n_transforms[b + 2u] = n_transforms[b + 2u] + m.z;
}
