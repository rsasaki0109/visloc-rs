// Accumulate each visible gaussian's view-space position gradient norm for
// densification: |dL/d mean2d| in NDC units (pixels * (W/2, H/2)), summed over
// the frames it was visible in, plus that frame count.

struct StatsUniforms {
    num_visible: u32,
    half_w: f32,
    half_h: f32,
    // 0: Inria (sum of |view-space xy grad| in NDC units);
    // 1: brush (max of the refine weight, screen_grads[9]).
    mode: u32,
};

@group(0) @binding(0) var<uniform> su: StatsUniforms;
@group(0) @binding(1) var<storage, read> global_from_compact: array<u32>;
// 10 floats per visible gaussian: du, dv, ... (see rasterize_backward).
@group(0) @binding(2) var<storage, read> screen_grads: array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_accum: array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_count: array<f32>;

@compute @workgroup_size(256)
fn densify_stats(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let c = gid3.x;
    if (c >= su.num_visible) {
        return;
    }
    let g = global_from_compact[c];
    let du = screen_grads[c * 10u] * su.half_w;
    let dv = screen_grads[c * 10u + 1u] * su.half_h;
    if (su.mode == 1u) {
        grad_accum[g] = max(grad_accum[g], screen_grads[c * 10u + 9u]);
    } else {
        grad_accum[g] = grad_accum[g] + sqrt(du * du + dv * dv);
    }
    grad_count[g] = grad_count[g] + 1.0;
}
