// Depth-normal consistency (2DGS / PGSR style) on the geometry channels.
//
// Per pixel p the renderer gives N_p = sum_i w_i n_i (alpha-weighted
// camera-space normals) and D_p = sum_i w_i z_i (alpha-weighted depth), and
// alpha_p = 1 - T_p. With the surface point P(q) = (D_q / alpha_q) r(q),
// r(q) = ((qx + 0.5 - cx) / fx, (qy + 0.5 - cy) / fy, 1) (through a lens its
// pixel ray from `rays`, and D the lens's depth: z, or a fisheye's distance),
// the normal of the depth map is n_d = s c / |c|,
// c = (P(x+1) - P(x-1)) x (P(y+1) - P(y-1)),
// with s = +-1 so that it faces the camera. The loss is
//   L = weight / npix * sum_p (1 - N_p . n_d(p))
// over pixels whose own and 4-neighbour opacities exceed 0.5. Gradients go to
// both N (rotating the splats onto the depth surface) and D (flattening the
// depth onto the splats' normals); alpha is treated as a constant.
//
// Pass `normal_grad` writes dL/dN into d_geo.xyz and, per pixel, dL/da and
// dL/db (a, b the two central differences) into `gab`; pass `depth_grad`
// gathers dL/dP(q) from the four pixels whose differences use q and writes
// dL/dD_q into d_geo.w.

struct NormalUniforms {
    w: u32,
    h: u32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    // weight / npix
    coef: f32,
    // 1: r(q) is rays[q] (the camera has a lens).
    lensed: u32,
};

@group(0) @binding(0) var<uniform> nu: NormalUniforms;
@group(0) @binding(1) var<storage, read> out_geo: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> final_t: array<f32>;
@group(0) @binding(3) var<storage, read_write> d_geo: array<vec4<f32>>;
// Two vec4 per pixel: (dL/da, 0), (dL/db, 0).
@group(0) @binding(4) var<storage, read_write> gab: array<vec4<f32>>;
// Per pixel, its ray through the lens (xyz); unread for a pinhole.
@group(0) @binding(5) var<storage, read> rays: array<vec4<f32>>;

fn ray(x: u32, y: u32) -> vec3<f32> {
    if (nu.lensed != 0u) {
        return rays[y * nu.w + x].xyz;
    }
    return vec3<f32>((f32(x) + 0.5 - nu.cx) / nu.fx, (f32(y) + 0.5 - nu.cy) / nu.fy, 1.0);
}

fn alpha_at(x: u32, y: u32) -> f32 {
    return 1.0 - final_t[y * nu.w + x];
}

fn point(x: u32, y: u32) -> vec3<f32> {
    let i = y * nu.w + x;
    return out_geo[i].w / max(1.0 - final_t[i], 1e-6) * ray(x, y);
}

fn pixel_id(gid3: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid3.x + gid3.y * nwg.x * 256u;
}

@compute @workgroup_size(256)
fn normal_grad(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = pixel_id(gid3, nwg);
    if (i >= nu.w * nu.h) {
        return;
    }
    let x = i % nu.w;
    let y = i / nu.w;
    var gn = vec3<f32>(0.0);
    var ga = vec3<f32>(0.0);
    var gb = vec3<f32>(0.0);
    let interior = x > 0u && y > 0u && x + 1u < nu.w && y + 1u < nu.h;
    if (interior && alpha_at(x, y) > 0.5 && alpha_at(x - 1u, y) > 0.5 && alpha_at(x + 1u, y) > 0.5
        && alpha_at(x, y - 1u) > 0.5 && alpha_at(x, y + 1u) > 0.5) {
        let a = point(x + 1u, y) - point(x - 1u, y);
        let b = point(x, y + 1u) - point(x, y - 1u);
        let c = cross(a, b);
        let cl = length(c);
        if (cl > 1e-12) {
            let n = c / cl;
            let s = select(1.0, -1.0, dot(n, point(x, y)) > 0.0);
            let big_n = out_geo[i].xyz;
            // L = coef (1 - N . s n)
            gn = -nu.coef * s * n;
            let g_n = -nu.coef * s * big_n;
            let gc = (g_n - n * dot(n, g_n)) / cl;
            ga = cross(b, gc);
            gb = cross(gc, a);
        }
    }
    d_geo[i] = vec4<f32>(gn, 0.0);
    gab[2u * i] = vec4<f32>(ga, 0.0);
    gab[2u * i + 1u] = vec4<f32>(gb, 0.0);
}

@compute @workgroup_size(256)
fn depth_grad(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = pixel_id(gid3, nwg);
    if (i >= nu.w * nu.h) {
        return;
    }
    let x = i % nu.w;
    let y = i / nu.w;
    // a(p) = P(p + x) - P(p - x): q is the +x term of p = q - x and the -x
    // term of p = q + x (likewise for b along y).
    var gp = vec3<f32>(0.0);
    if (x > 0u) {
        gp = gp + gab[2u * (i - 1u)].xyz;
    }
    if (x + 1u < nu.w) {
        gp = gp - gab[2u * (i + 1u)].xyz;
    }
    if (y > 0u) {
        gp = gp + gab[2u * (i - nu.w) + 1u].xyz;
    }
    if (y + 1u < nu.h) {
        gp = gp - gab[2u * (i + nu.w) + 1u].xyz;
    }
    let alpha = max(alpha_at(x, y), 1e-6);
    var g = d_geo[i];
    g.w = dot(gp, ray(x, y)) / alpha;
    d_geo[i] = g;
}
