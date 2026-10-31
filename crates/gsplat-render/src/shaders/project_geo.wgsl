// Geometry channels, per visible gaussian (compact order): the camera-space
// normal -- the rotation axis of the smallest scale, flipped to face the
// camera -- and the camera-space depth (the distance for a fisheye), as 4
// floats in `geo_splats`.

@group(0) @binding(0) var<uniform> u: ProjectUniforms;
@group(0) @binding(1) var<storage, read> transforms: array<f32>;
@group(0) @binding(2) var<storage, read> global_from_compact: array<u32>;
@group(0) @binding(3) var<storage, read_write> geo_splats: array<vec4<f32>>;

@compute @workgroup_size(256)
fn project_geo(@builtin(global_invocation_id) gid3: vec3<u32>) {
    let compact = gid3.x;
    if (compact >= u.num_visible) {
        return;
    }
    let base = global_from_compact[compact] * 10u;
    let m = vec3<f32>(transforms[base], transforms[base + 1u], transforms[base + 2u]);
    let rg = rot_from_quat(transforms[base + 3u], transforms[base + 4u], transforms[base + 5u], transforms[base + 6u]);
    let ls = vec3<f32>(transforms[base + 7u], transforms[base + 8u], transforms[base + 9u]);
    var axis = rg[0];
    var smin = ls.x;
    if (ls.y < smin) {
        axis = rg[1];
        smin = ls.y;
    }
    if (ls.z < smin) {
        axis = rg[2];
    }
    let w = view_rot(u);
    let p = w * m + u.view_t.xyz;
    var n = w * axis;
    if (dot(n, p) > 0.0) {
        n = -n;
    }
    // A fisheye's depth is the distance (compute_projected's).
    var depth = p.z;
    if (u.lens_kind == LENS_EQUIDISTANT) {
        depth = length(p);
    }
    geo_splats[compact] = vec4<f32>(n, depth);
}
