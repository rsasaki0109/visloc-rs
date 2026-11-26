// Depth prior on the geometry channels (as gsplat-cuda's depth_loss_kernel).
//
// With D_p = sum_i T_i a_i z_i the renderer's depth channel (out_geo.w) and
// P_p > 0 a prior depth (e.g. a LiDAR scan projected into the view; 0 where
// there is none), the loss is the relative L1
//   L = weight / npix * sum_{p : P_p > 0} |D_p - P_p| / P_p
// (near and far surfaces weigh alike), so dL/dD_p = coef sign(D_p - P_p) / P_p
// with coef = weight / npix. It is written into d_geo.w; with `accumulate` it
// is added to what the depth-normal loss wrote there, else d_geo is set to
// (0, 0, 0, dL/dD).

struct PriorUniforms {
    npix: u32,
    // weight / npix
    coef: f32,
    accumulate: u32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> pu: PriorUniforms;
@group(0) @binding(1) var<storage, read> out_geo: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prior: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_geo: array<vec4<f32>>;

@compute @workgroup_size(256)
fn depth_prior(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid3.x + gid3.y * nwg.x * 256u;
    if (i >= pu.npix) {
        return;
    }
    let p = prior[i];
    var g = 0.0;
    if (p > 0.0) {
        let diff = out_geo[i].w - p;
        g = pu.coef * sign(diff) / p;
    }
    if (pu.accumulate != 0u) {
        d_geo[i].w = d_geo[i].w + g;
    } else {
        d_geo[i] = vec4<f32>(0.0, 0.0, 0.0, g);
    }
}
