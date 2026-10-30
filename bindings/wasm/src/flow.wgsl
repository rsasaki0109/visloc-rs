// Basalt's frame-to-frame frontend on the GPU, for one camera: the image
// pyramid, the KLT tracking of the previous frame's points (Pattern51
// patches, inverse-compositional SE(2) search from the coarsest level,
// forward-backward check) and the grid FAST replenishment. It follows
// pipelines/basalt/src/{pyramid,patch,update,stream,fast}.rs step by step;
// the arithmetic is plain f32 (the CPU code reproduces Eigen's rounding
// order bit for bit, which a GPU cannot), so results agree to rounding.
//
// Pixels are the CPU pyramid's u16 values (8-bit grey << 8, or the
// undistorted image), stored as u32 with both frames' pyramids in one
// buffer.

struct Params {
    // Per level: width, height, offset within a pyramid, unused.
    levels: array<vec4<u32>, 8>,
    level_count: u32,
    old_base: u32,
    new_base: u32,
    track_count: u32,
    max_iterations: u32,
    fb_squared: f32,
    cell: u32,
    cells_x: u32,
    cells_y: u32,
    x_start: u32,
    y_start: u32,
    threshold: u32,
    min_threshold: u32,
    edge: f32,
    _pad0: u32,
    _pad1: u32,
}

// A track's affine patch transform: linear part row-major, translation.
struct Track {
    linear: vec4<f32>,
    position: vec2<f32>,
    _pad: vec2<f32>,
}

// The tracked transform and 0, or a failure code (see `failure` in
// gpu_flow.rs).
struct TrackResult {
    linear: vec4<f32>,
    position: vec2<f32>,
    status: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pyr: array<u32>;
@group(0) @binding(2) var<storage, read> image: array<u32>;
@group(0) @binding(3) var<storage, read> tracks: array<Track>;
@group(0) @binding(4) var<storage, read_write> results: array<TrackResult>;
@group(0) @binding(5) var<storage, read_write> occupied: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> corners: array<vec2<i32>>;
@group(1) @binding(0) var<uniform> down_level: vec4<u32>;

const SAMPLES: u32 = 52u;
const EPS: f32 = 1.1920929e-7;
const MAX_INCREMENT: f32 = 1e6;
const SMALL_ANGLE: f32 = 1e-5;

const SOURCE_PATCH_INVALID: u32 = 1u;
const TARGET_NO_VALID_SAMPLES: u32 = 2u;
const TARGET_INSUFFICIENT_OVERLAP: u32 = 3u;
const INCREMENT_NON_FINITE: u32 = 4u;
const INCREMENT_TOO_LARGE: u32 = 5u;
const TARGET_OUT_OF_BOUNDS: u32 = 6u;
const BACKWARD: u32 = 16u;
const FB_SQUARED: u32 = 32u;

// ------------------------------------------------------------- pyramid

// Level 0 of the new frame from the image's u16 pixels, two to a word.
@compute @workgroup_size(256)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let level = params.levels[0];
    let i = gid.x;
    if i >= level.x * level.y {
        return;
    }
    pyr[params.new_base + level.z + i] = (image[i >> 1u] >> ((i & 1u) * 16u)) & 0xffffu;
}

fn abs_diff(a: u32, b: u32) -> u32 {
    return select(a - b, b - a, a < b);
}

// OpenCV's BORDER_REFLECT_101 index, as `border101` in pyramid.rs.
fn border101(x: u32, n: u32) -> u32 {
    return n - 1u - abs_diff(n - 1u, x);
}

// Level `down_level.x` of the new frame: the 5x5 binomial filter of the
// level above, decimated by two, rounded as the CPU's integer passes.
@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
    let l = down_level.x;
    let dst = params.levels[l];
    let src = params.levels[l - 1u];
    if gid.x >= dst.x || gid.y >= dst.y {
        return;
    }
    let weights = array<i32, 5>(1, 4, 6, 4, 1);
    let y2 = 2u * gid.y;
    let x2 = 2u * gid.x;
    let rows = array<u32, 5>(abs_diff(y2, 2u), abs_diff(y2, 1u), y2, border101(y2 + 1u, src.y), border101(y2 + 2u, src.y));
    let cols = array<u32, 5>(abs_diff(x2, 2u), abs_diff(x2, 1u), x2, border101(x2 + 1u, src.x), border101(x2 + 2u, src.x));
    let base = params.new_base + src.z;
    var acc = 0;
    for (var i = 0u; i < 5u; i++) {
        var row = 0;
        for (var j = 0u; j < 5u; j++) {
            row += weights[j] * i32(pyr[base + rows[i] * src.x + cols[j]]);
        }
        acc += weights[i] * row;
    }
    pyr[params.new_base + dst.z + gid.y * dst.x + gid.x] = u32((acc + 128) >> 8u);
}

// ------------------------------------------------------------- sampling

fn pixel(base: u32, l: u32, x: u32, y: u32) -> f32 {
    let level = params.levels[l];
    return f32(pyr[base + level.z + y * level.x + x]);
}

// Basalt's `InBounds(p, border)`: four neighbours plus `border` pixels.
fn in_bounds(l: u32, p: vec2<f32>, border: f32) -> bool {
    let level = params.levels[l];
    return border <= p.x && p.x < f32(level.x) - border - 1.0
        && border <= p.y && p.y < f32(level.y) - border - 1.0;
}

fn bilinear(base: u32, l: u32, ix: u32, iy: u32, d: vec2<f32>) -> f32 {
    let p00 = pixel(base, l, ix, iy);
    let p01 = pixel(base, l, ix, iy + 1u);
    let p10 = pixel(base, l, ix + 1u, iy);
    let p11 = pixel(base, l, ix + 1u, iy + 1u);
    let ddx = 1.0 - d.x;
    let ddy = 1.0 - d.y;
    return ddx * ddy * p00 + ddx * d.y * p01 + d.x * ddy * p10 + d.x * d.y * p11;
}

fn interp(base: u32, l: u32, p: vec2<f32>) -> f32 {
    let ix = u32(p.x);
    let iy = u32(p.y);
    return bilinear(base, l, ix, iy, p - vec2<f32>(f32(ix), f32(iy)));
}

// Value and central-difference gradient, each interpolated bilinearly.
fn interp_grad(base: u32, l: u32, p: vec2<f32>) -> vec3<f32> {
    let ix = u32(p.x);
    let iy = u32(p.y);
    let d = p - vec2<f32>(f32(ix), f32(iy));
    let value = bilinear(base, l, ix, iy, d);
    let gx = 0.5 * (bilinear(base, l, ix + 1u, iy, d) - bilinear(base, l, ix - 1u, iy, d));
    let gy = 0.5 * (bilinear(base, l, ix, iy + 1u, d) - bilinear(base, l, ix, iy - 1u, d));
    return vec3<f32>(value, gx, gy);
}

// Pattern51's sample offsets: rows of 4, 6, 8, 8, 8, 8, 6 and 4 samples
// from y = 3.5 down to -3.5, each row centred on x = 0.
fn pattern(i: u32) -> vec2<f32> {
    var n = 4u;
    var start = 0u;
    var y = 3.5;
    if i < 4u {
        n = 4u; start = 0u; y = 3.5;
    } else if i < 10u {
        n = 6u; start = 4u; y = 2.5;
    } else if i < 18u {
        n = 8u; start = 10u; y = 1.5;
    } else if i < 26u {
        n = 8u; start = 18u; y = 0.5;
    } else if i < 34u {
        n = 8u; start = 26u; y = -0.5;
    } else if i < 42u {
        n = 8u; start = 34u; y = -1.5;
    } else if i < 48u {
        n = 6u; start = 42u; y = -2.5;
    } else {
        n = 4u; start = 48u; y = -3.5;
    }
    return vec2<f32>(f32(i - start) - 0.5 * (f32(n) - 1.0), y);
}

fn finite(v: f32) -> bool {
    // The exponent bits, not comparisons: shader compilers may assume
    // floats are never NaN and fold those away.
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

// ------------------------------------------------------------- KLT

var<workgroup> red_a: array<vec4<f32>, 64>;
var<workgroup> red_b: array<vec4<f32>, 64>;
// The search transform at the current level: a00, a01, a10, a11, tx, ty.
var<workgroup> xf: array<f32, 6>;
var<workgroup> hinv: array<f32, 9>;
var<workgroup> status: u32;

// Workgroup sums of `a` and `b` (every thread gets them).
fn reduce2(t: u32, a: vec4<f32>, b: vec4<f32>) -> array<vec4<f32>, 2> {
    red_a[t] = a;
    red_b[t] = b;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if t < s {
            red_a[t] += red_a[t + s];
            red_b[t] += red_b[t + s];
        }
        workgroupBarrier();
    }
    let out = array<vec4<f32>, 2>(red_a[0], red_b[0]);
    workgroupBarrier();
    return out;
}

// Thread 0 sets `status` to `code` if it is still 0. Branch on `status`
// only through workgroupUniformLoad: barriers must stay in uniform control
// flow, which browsers' WGSL compilers enforce.
fn fail(t: u32, failed: bool, code: u32) {
    if t == 0u && failed && status == 0u {
        status = code;
    }
}

// Right-composes SE2::exp(inc) onto `xf` (thread 0), as
// AffineCompact2f::right_compose_se2.
fn compose(inc: vec3<f32>) {
    let theta = inc.z;
    var s = sin(theta);
    var c = cos(theta);
    let norm = sqrt(s * s + c * c);
    s /= norm;
    c /= norm;
    var sot: f32;
    var omc: f32;
    if abs(theta) < SMALL_ANGLE {
        let t2 = theta * theta;
        sot = 1.0 - t2 / 6.0;
        omc = 0.5 * theta - theta * t2 / 24.0;
    } else {
        sot = s / theta;
        omc = (1.0 - c) / theta;
    }
    let ux = sot * inc.x - omc * inc.y;
    let uy = omc * inc.x + sot * inc.y;
    let a00 = xf[0];
    let a01 = xf[1];
    let a10 = xf[2];
    let a11 = xf[3];
    xf[0] = a00 * c + a01 * s;
    xf[1] = a01 * c - a00 * s;
    xf[2] = a10 * c + a11 * s;
    xf[3] = a11 * c - a10 * s;
    xf[4] = a00 * ux + a01 * uy + xf[4];
    xf[5] = a10 * ux + a11 * uy + xf[5];
}

// One KLT direction, as track_direction_from_seed_impl: the patch around
// `source` in the pyramid at `src`, searched for in the pyramid at `dst`
// from the transform in `xf` (identity linear part, full-resolution seed
// translation). Leaves the found transform in `xf` (full resolution) and
// returns 0 or a failure code.
fn direction(t: u32, src: u32, dst: u32, source: vec2<f32>) -> u32 {
    let offset = pattern(min(t, SAMPLES - 1u));
    let sampler = t < SAMPLES;
    for (var lv = i32(params.level_count) - 1; lv >= 0; lv--) {
        let l = u32(lv);
        let scale = f32(1u << l);
        // The patch and its inverse-compositional Jacobian.
        let sp = source / scale + offset;
        let valid = sampler && in_bounds(l, sp, 2.0);
        var data = -1.0;
        var jac = vec3<f32>(0.0);
        if valid {
            let g = interp_grad(src, l, sp);
            data = g.x;
            jac = vec3<f32>(g.y, g.z, g.z * offset.x - g.y * offset.y);
        }
        let r1 = reduce2(t, vec4<f32>(select(0.0, data, valid), select(0.0, 1.0, valid), jac.x, jac.y), vec4<f32>(jac.z, 0.0, 0.0, 0.0));
        let sum = r1[0].x;
        let count = r1[0].y;
        let grad_sum = vec3<f32>(r1[0].z, r1[0].w, r1[1].x);
        fail(t, !(count > 0.0 && sum > EPS && finite(sum)), SOURCE_PATCH_INVALID);
        let patch_status = workgroupUniformLoad(&status);
        if patch_status != 0u {
            return patch_status;
        }
        let mean_inv = count / sum;
        if valid {
            jac = (jac - grad_sum * data / sum) * mean_inv;
            data *= mean_inv;
        } else {
            jac = vec3<f32>(0.0);
        }
        let r2 = reduce2(t, vec4<f32>(jac.x * jac.x, jac.x * jac.y, jac.x * jac.z, jac.y * jac.y), vec4<f32>(jac.y * jac.z, jac.z * jac.z, 0.0, 0.0));
        if t == 0u {
            let a = r2[0].x;
            let b = r2[0].y;
            let c = r2[0].z;
            let d = r2[0].w;
            let e = r2[1].x;
            let f = r2[1].y;
            let c00 = d * f - e * e;
            let c01 = c * e - b * f;
            let c02 = b * e - c * d;
            let c11 = a * f - c * c;
            let c12 = b * c - a * e;
            let c22 = a * d - b * b;
            let det = a * c00 + b * c01 + c * c02;
            let ok = det != 0.0 && finite(det) && sum / count > EPS;
            let inv = 1.0 / det;
            hinv[0] = c00 * inv;
            hinv[1] = c01 * inv;
            hinv[2] = c02 * inv;
            hinv[3] = c01 * inv;
            hinv[4] = c11 * inv;
            hinv[5] = c12 * inv;
            hinv[6] = c02 * inv;
            hinv[7] = c12 * inv;
            hinv[8] = c22 * inv;
            var all_finite = ok;
            for (var k = 0u; k < 9u; k++) {
                all_finite = all_finite && finite(hinv[k]);
            }
            if !all_finite {
                status = SOURCE_PATCH_INVALID;
            }
            // This level's search starts from the finer level's result.
            xf[4] /= scale;
            xf[5] /= scale;
        }
        let inverse_status = workgroupUniformLoad(&status);
        if inverse_status != 0u {
            return inverse_status;
        }
        let hj = vec3<f32>(
            hinv[0] * jac.x + hinv[1] * jac.y + hinv[2] * jac.z,
            hinv[3] * jac.x + hinv[4] * jac.y + hinv[5] * jac.z,
            hinv[6] * jac.x + hinv[7] * jac.y + hinv[8] * jac.z,
        );
        let has_data = data >= 0.0;
        for (var it = 0u; it < params.max_iterations; it++) {
            let p = vec2<f32>(xf[0] * offset.x + xf[1] * offset.y + xf[4], xf[2] * offset.x + xf[3] * offset.y + xf[5]);
            let target_valid = sampler && in_bounds(l, p, 2.0);
            var value = 0.0;
            if target_valid {
                value = interp(dst, l, p);
            }
            let r3 = reduce2(t, vec4<f32>(value, select(0.0, 1.0, target_valid), 0.0, 0.0), vec4<f32>(0.0));
            let tsum = r3[0].x;
            let tcount = r3[0].y;
            fail(t, !(tcount > 0.0 && tsum > EPS && finite(tsum)), TARGET_NO_VALID_SAMPLES);
            let target_status = workgroupUniformLoad(&status);
            if target_status != 0u {
                return target_status;
            }
            let both = target_valid && has_data;
            var residual = 0.0;
            if both {
                residual = tcount * value / tsum - data;
            }
            let r4 = reduce2(t, vec4<f32>(hj * residual, select(0.0, 1.0, both)), vec4<f32>(0.0));
            let inc = -r4[0].xyz;
            if t == 0u {
                if r4[0].w <= f32(SAMPLES / 2u) {
                    status = TARGET_INSUFFICIENT_OVERLAP;
                } else if !(finite(inc.x) && finite(inc.y) && finite(inc.z)) {
                    status = INCREMENT_NON_FINITE;
                } else if max(abs(inc.x), max(abs(inc.y), abs(inc.z))) >= MAX_INCREMENT {
                    status = INCREMENT_TOO_LARGE;
                } else {
                    compose(inc);
                    if !in_bounds(l, vec2<f32>(xf[4], xf[5]), 2.0) {
                        status = TARGET_OUT_OF_BOUNDS;
                    }
                }
            }
            let step_status = workgroupUniformLoad(&status);
            if step_status != 0u {
                return step_status;
            }
        }
        if t == 0u {
            xf[4] *= scale;
            xf[5] *= scale;
        }
        workgroupBarrier();
    }
    return 0u;
}

// One workgroup per track: forward from the old frame to the new, back
// again from the result, and the forward-backward check.
@compute @workgroup_size(64)
fn klt(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let i = wg.x + wg.y * 65535u;
    if i >= params.track_count {
        return;
    }
    let track = tracks[i];
    let old = track.position;
    if t == 0u {
        status = 0u;
        xf = array<f32, 6>(1.0, 0.0, 0.0, 1.0, old.x, old.y);
    }
    workgroupBarrier();
    let forward = direction(t, params.old_base, params.new_base, old);
    if forward != 0u {
        if t == 0u {
            results[i].status = forward;
        }
        return;
    }
    // The forward result composes the track's accumulated linear part.
    let b = track.linear;
    let found = vec4<f32>(
        b.x * xf[0] + b.y * xf[2],
        b.x * xf[1] + b.y * xf[3],
        b.z * xf[0] + b.w * xf[2],
        b.z * xf[1] + b.w * xf[3],
    );
    let position = vec2<f32>(xf[4], xf[5]);
    workgroupBarrier();
    if t == 0u {
        xf = array<f32, 6>(1.0, 0.0, 0.0, 1.0, position.x, position.y);
    }
    workgroupBarrier();
    let backward = direction(t, params.new_base, params.old_base, position);
    if t == 0u {
        var code = 0u;
        if backward != 0u {
            code = BACKWARD + backward;
        } else {
            let d = old - vec2<f32>(xf[4], xf[5]);
            if dot(d, d) >= params.fb_squared {
                code = FB_SQUARED;
            }
        }
        results[i] = TrackResult(found, position, code, 0u);
    }
}

// Marks the grid cells that hold a tracked point.
@compute @workgroup_size(64)
fn occupancy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.track_count || results[i].status != 0u {
        return;
    }
    let p = results[i].position;
    let x_end = f32(params.x_start + params.cells_x * params.cell);
    let y_end = f32(params.y_start + params.cells_y * params.cell);
    if p.x < f32(params.x_start) || p.y < f32(params.y_start) || p.x >= x_end || p.y >= y_end {
        return;
    }
    let cx = (u32(p.x) - params.x_start) / params.cell;
    let cy = (u32(p.y) - params.y_start) / params.cell;
    atomicStore(&occupied[cx * params.cells_y + cy], 1u);
}

// ------------------------------------------------------------- FAST

const MAX_CELL: u32 = 60u;
var<workgroup> scores: array<i32, 3600>;
var<workgroup> best: atomic<u32>;
var<workgroup> best_plain: u32;
var<workgroup> skip: u32;

fn grey_at(x: i32, y: i32) -> i32 {
    let level = params.levels[0];
    return i32(pyr[params.new_base + level.z + u32(y) * level.x + u32(x)] >> 8u);
}

// FAST-9's score at (x, y), or 0: fast_score_without_nms in fast.rs.
fn fast_score(x: i32, y: i32, threshold: i32) -> i32 {
    let level = params.levels[0];
    if x < 3 || y < 3 || x + 3 >= i32(level.x) || y + 3 >= i32(level.y) {
        return 0;
    }
    var circle = array<vec2<i32>, 16>(
        vec2<i32>(0, -3), vec2<i32>(1, -3), vec2<i32>(2, -2), vec2<i32>(3, -1),
        vec2<i32>(3, 0), vec2<i32>(3, 1), vec2<i32>(2, 2), vec2<i32>(1, 3),
        vec2<i32>(0, 3), vec2<i32>(-1, 3), vec2<i32>(-2, 2), vec2<i32>(-3, 1),
        vec2<i32>(-3, 0), vec2<i32>(-3, -1), vec2<i32>(-2, -2), vec2<i32>(-1, -3),
    );
    let center = grey_at(x, y);
    // A run of 9 holds two neighbouring compass pixels (0, 4, 8, 12).
    var compass_bright = 0u;
    var compass_dark = 0u;
    for (var k = 0u; k < 4u; k++) {
        let o = circle[k * 4u];
        let d = center - grey_at(x + o.x, y + o.y);
        if d < -threshold {
            compass_bright |= 1u << k;
        }
        if d > threshold {
            compass_dark |= 1u << k;
        }
    }
    let pair_bright = (compass_bright & ((compass_bright >> 1u) | (compass_bright << 3u)) & 0xfu) != 0u;
    let pair_dark = (compass_dark & ((compass_dark >> 1u) | (compass_dark << 3u)) & 0xfu) != 0u;
    if !pair_bright && !pair_dark {
        return 0;
    }
    var diffs: array<i32, 24>;
    var bright = 0u;
    var dark = 0u;
    for (var k = 0u; k < 16u; k++) {
        let o = circle[k];
        let d = center - grey_at(x + o.x, y + o.y);
        diffs[k] = d;
        if d < -threshold {
            bright |= 1u << k;
        }
        if d > threshold {
            dark |= 1u << k;
        }
    }
    if !has_run9(bright) && !has_run9(dark) {
        return 0;
    }
    for (var k = 0u; k < 8u; k++) {
        diffs[16u + k] = diffs[k];
    }
    var score = -32768;
    for (var start = 0u; start < 16u; start++) {
        var lo = 32767;
        var hi = -32768;
        for (var k = 0u; k < 9u; k++) {
            lo = min(lo, diffs[start + k]);
            hi = max(hi, diffs[start + k]);
        }
        score = max(score, max(lo - 1, -hi - 1));
    }
    return score;
}

fn has_run9(mask: u32) -> bool {
    let doubled = mask | (mask << 16u);
    var starts = doubled;
    for (var s = 1u; s < 9u; s++) {
        starts &= doubled >> s;
    }
    return (starts & 0xffffu) != 0u;
}

// One workgroup per grid cell: if no tracked point is in it, the strongest
// FAST corner (after 3x3 non-maximum suppression, away from the image
// edge), halving the threshold until one is found, as detect_cell. Ties go
// to the first pixel in raster order.
@compute @workgroup_size(256)
fn fast(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let index = wg.x;
    if index >= params.cells_x * params.cells_y {
        return;
    }
    let cell = params.cell;
    if t == 0u {
        corners[index] = vec2<i32>(-1, -1);
        skip = atomicLoad(&occupied[index]);
    }
    if workgroupUniformLoad(&skip) != 0u || cell <= 6u || cell > MAX_CELL {
        return;
    }
    let x0 = i32(params.x_start + (index / params.cells_y) * cell);
    let y0 = i32(params.y_start + (index % params.cells_y) * cell);
    let inner = cell - 6u;
    let level = params.levels[0];
    var threshold = params.threshold;
    loop {
        for (var k = t; k < cell * cell; k += 256u) {
            scores[k] = 0;
        }
        if t == 0u {
            atomicStore(&best, 0u);
        }
        workgroupBarrier();
        for (var k = t; k < inner * inner; k += 256u) {
            let lx = 3u + k % inner;
            let ly = 3u + k / inner;
            scores[ly * cell + lx] = fast_score(x0 + i32(lx), y0 + i32(ly), i32(threshold));
        }
        workgroupBarrier();
        for (var k = t; k < inner * inner; k += 256u) {
            let lx = 3u + k % inner;
            let ly = 3u + k / inner;
            let score = scores[ly * cell + lx];
            if score <= 0 {
                continue;
            }
            var is_max = true;
            for (var dy = -1; dy <= 1; dy++) {
                for (var dx = -1; dx <= 1; dx++) {
                    if (dx != 0 || dy != 0) && scores[u32(i32(ly) + dy) * cell + u32(i32(lx) + dx)] >= score {
                        is_max = false;
                    }
                }
            }
            let x = f32(x0 + i32(lx));
            let y = f32(y0 + i32(ly));
            let edge = params.edge;
            let inside = edge <= x && x < f32(level.x) - edge - 1.0 && edge <= y && y < f32(level.y) - edge - 1.0;
            if is_max && inside {
                let local = ly * cell + lx;
                atomicMax(&best, (u32(score) << 16u) | (0xffffu - local));
            }
        }
        workgroupBarrier();
        if t == 0u {
            best_plain = atomicLoad(&best);
        }
        let found = workgroupUniformLoad(&best_plain);
        if found != 0u {
            if t == 0u {
                let local = 0xffffu - (found & 0xffffu);
                corners[index] = vec2<i32>(x0 + i32(local % cell), y0 + i32(local / cell));
            }
            break;
        }
        if threshold == params.min_threshold {
            break;
        }
        threshold /= 2u;
        if threshold < params.min_threshold {
            break;
        }
    }
}
