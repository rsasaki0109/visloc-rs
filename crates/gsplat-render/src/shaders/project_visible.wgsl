// Pass 2: after the depth sort, re-project each visible gaussian and compute its
// view-dependent SH colour, writing the compact `projected_splats` array.
//
// One invocation per *visible* gaussian (`compact` index). `global_from_compact`
// maps the sorted compact index to the original gaussian id.

@group(0) @binding(0) var<uniform> u: ProjectUniforms;
@group(0) @binding(1) var<storage, read> transforms: array<f32>;
@group(0) @binding(2) var<storage, read> opacity_in: array<f32>;
@group(0) @binding(3) var<storage, read> sh_in: array<f32>;
@group(0) @binding(4) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(5) var<storage, read_write> projected_splats: array<f32>;
@group(0) @binding(6) var<storage, read_write> compact_from_global: array<u32>;
// Per-gaussian tile counts (indexed by global id) and the compact-order gather
// that the prefix scan consumes.
@group(0) @binding(7) var<storage, read> intersect_counts: array<u32>;
@group(0) @binding(8) var<storage, read_write> counts_sorted: array<u32>;

// SH rest coefficient k (after DC) of channel ch, or 0 past the degree.
fn sh_rest_coef(sh_base: u32, rest_pc: u32, ch: u32, k: u32) -> f32 {
    if (k >= rest_pc) {
        return 0.0;
    }
    return sh_in[sh_base + 3u + ch * rest_pc + k];
}

// View-dependent colour (raw, + 0.5) up to degree 3; same basis as
// `eval_sh_basis`.
fn sh_color(sh_base: u32, rest_pc: u32, act: u32, d: vec3<f32>) -> vec3<f32> {
    // rest_pc: stored rest coefficients per channel (stride); act: evaluated.
    let x = d.x;
    let y = d.y;
    let z = d.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let c0 = 0.4886025;
    let c3 = 1.0925485;
    let c5 = 0.3153916;
    let c8 = 0.3731762;
    let c9 = 2.8906113;
    let c10 = 1.843772;
    let c11 = 0.5900436;
    let b1 = -c0 * y;
    let b2 = c0 * z;
    let b3 = -c0 * x;
    let b4 = c3 * x * y;
    let b5 = -c3 * y * z;
    let b6 = c5 * (2.0 * zz - xx - yy);
    let b7 = -c3 * x * z;
    let b8 = c3 * (xx - yy);
    let b9 = -c8 * y * (3.0 * xx - yy);
    let b10 = c9 * x * y * z;
    let b11 = -c10 * y * (4.0 * zz - xx - yy);
    let b12 = c11 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    let b13 = -c10 * x * (4.0 * zz - xx - yy);
    let b14 = c9 * z * (xx - yy);
    let b15 = -c8 * x * (xx - 3.0 * yy);
    var out: vec3<f32>;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        var acc = 0.2820948 * sh_in[sh_base + ch];
        if (act >= 3u) {
            acc = acc + b1 * sh_rest_coef(sh_base, rest_pc, ch, 0u)
                + b2 * sh_rest_coef(sh_base, rest_pc, ch, 1u)
                + b3 * sh_rest_coef(sh_base, rest_pc, ch, 2u);
        }
        if (act >= 8u) {
            acc = acc + b4 * sh_rest_coef(sh_base, rest_pc, ch, 3u)
                + b5 * sh_rest_coef(sh_base, rest_pc, ch, 4u)
                + b6 * sh_rest_coef(sh_base, rest_pc, ch, 5u)
                + b7 * sh_rest_coef(sh_base, rest_pc, ch, 6u)
                + b8 * sh_rest_coef(sh_base, rest_pc, ch, 7u);
        }
        if (act >= 15u) {
            acc = acc + b9 * sh_rest_coef(sh_base, rest_pc, ch, 8u)
                + b10 * sh_rest_coef(sh_base, rest_pc, ch, 9u)
                + b11 * sh_rest_coef(sh_base, rest_pc, ch, 10u)
                + b12 * sh_rest_coef(sh_base, rest_pc, ch, 11u)
                + b13 * sh_rest_coef(sh_base, rest_pc, ch, 12u)
                + b14 * sh_rest_coef(sh_base, rest_pc, ch, 13u)
                + b15 * sh_rest_coef(sh_base, rest_pc, ch, 14u);
        }
        out[ch] = acc + 0.5;
    }
    return out;
}

@compute @workgroup_size(256)
fn project_visible(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let compact = gid3.x;
    if (compact >= u.num_visible) {
        return;
    }
    let gid = global_from_compact[compact];
    compact_from_global[gid] = compact;
    // Gather the tile count into compact order for the prefix scan.
    counts_sorted[compact] = intersect_counts[gid];

    let base = gid * 10u;
    let mean = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let ls = vec3<f32>(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]);
    let p = compute_projected(
        u, mean,
        transforms[base + 3u], transforms[base + 4u],
        transforms[base + 5u], transforms[base + 6u],
        ls, opacity_in[gid],
    );

    // View-dependent colour: direction is world-space mean - camera centre.
    let dir_raw = mean - u.camera_center.xyz;
    var dir = vec3<f32>(0.0, 0.0, 1.0);
    let dn = length(dir_raw);
    if (dn > 1e-6) {
        dir = dir_raw / dn;
    }
    let cpc = u.sh_degree + 1u;
    let cpc2 = cpc * cpc;
    let sh_base = gid * 3u * cpc2;
    // Written out term by term: copying the coefficients into a
    // runtime-indexed local array made the compiler spill it.
    let act = (u.sh_active_degree + 1u) * (u.sh_active_degree + 1u) - 1u;
    let color = sh_color(sh_base, cpc2 - 1u, act, dir);

    let out = compact * PROJECTED_STRIDE;
    projected_splats[out + 0u] = select(0.0, p.proj_u, p.ok);
    projected_splats[out + 1u] = select(0.0, p.proj_v, p.ok);
    projected_splats[out + 2u] = p.conic.x;
    projected_splats[out + 3u] = p.conic.y;
    projected_splats[out + 4u] = p.conic.z;
    // A non-ok gaussian gets alpha 0 so it never contributes.
    projected_splats[out + 5u] = select(0.0, p.opacity, p.ok);
    projected_splats[out + 6u] = color.r;
    projected_splats[out + 7u] = color.g;
    projected_splats[out + 8u] = color.b;
}
