// DoG extrema + orientation assignment (ports of the legacy
// `detect_extremum` / `assign_orientations` isotropic path).

struct DetectParams {
    w: u32,
    h: u32,
    octave: u32,
    cand_cap: u32,
    contrast: f32,
    edge: f32,
    // 1.5 * sigma_base (orientation window base)
    hist_base: f32,
    // 2^(1 / intervals)
    k: f32,
    kp_cap: u32,
    max_orientations: u32,
    subpixel: u32,
    num_dog_levels: u32,
};

struct Candidate {
    x: u32,
    y: u32,
    level: u32,
    value: f32,
    dx: f32,
    dy: f32,
    ds: f32,
};

// Oriented keypoint: integer locus (the host maps it to the original frame
// in f64), orientation and |DoG| contrast.
struct OrientedKp {
    x: u32,
    y: u32,
    // octave << 8 | level
    oct_level: u32,
    bin: u32,
    orientation: f32,
    contrast: f32,
    dx_bits: u32,
    dy_bits: u32,
    ds_bits: u32,
};

@group(0) @binding(0) var<uniform> p: DetectParams;
@group(0) @binding(1) var<storage, read> dogs: array<f32>;
@group(0) @binding(2) var<storage, read_write> cands: array<Candidate>;
// counters[octave] = candidates of that octave, counters[16] = keypoints.
@group(0) @binding(3) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> kps: array<OrientedKp>;
@group(0) @binding(5) var<storage, read_write> args: array<u32>;

const KP_COUNTER: u32 = 16u;

fn dog_at(level: u32, x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(p.w) - 1));
    let yi = u32(clamp(y, 0, i32(p.h) - 1));
    return dogs[level * p.w * p.h + yi * p.w + xi];
}

// Unclamped access for extrema interior loops. The caller has already
// confined x/y and level such that every 3x3x3 neighbor is in bounds.
fn dog_raw(level: u32, x: i32, y: i32) -> f32 {
    return dogs[level * p.w * p.h + u32(y) * p.w + u32(x)];
}

// Test one DoG level at an interior pixel. Returns after the first
// non-extremum so the caller can continue with the next level.
fn detect_level(level: u32, x: i32, y: i32) {
    let v = dog_raw(level, x, y);
    if (abs(v) < p.contrast) {
        return;
    }
    var is_min = true;
    var is_max = true;
    // Check the same-level axis neighbors first, then the vertical neighbor,
    // then diagonals. Visiting the center DoG level first makes the common
    // reject case exit sooner than the original z-major nested loops.
    let offsets = array<vec2<i32>, 9>(
        vec2<i32>(1, 0),
        vec2<i32>(-1, 0),
        vec2<i32>(0, 1),
        vec2<i32>(0, -1),
        vec2<i32>(0, 0),
        vec2<i32>(1, 1),
        vec2<i32>(1, -1),
        vec2<i32>(-1, 1),
        vec2<i32>(-1, -1),
    );
    for (var dzi = 0u; dzi < 3u; dzi = dzi + 1u) {
        let dz = (dzi + 1u) % 3u;
        for (var i = 0u; i < 9u; i = i + 1u) {
            let dx = offsets[i].x;
            let dy = offsets[i].y;
            if (dz == 1u && dx == 0 && dy == 0) {
                continue;
            }
            let n = dog_raw(level + dz - 1u, x + dx, y + dy);
            if (n >= v) {
                is_max = false;
            }
            if (n <= v) {
                is_min = false;
            }
            // A candidate with both a larger and a smaller neighbor can
            // no longer be an extremum; skip the rest of its neighborhood.
            if (!is_min && !is_max) {
                return;
            }
        }
    }
    if (!(is_min || is_max)) {
        return;
    }
    let dxx = dog_raw(level, x + 1, y) + dog_raw(level, x - 1, y) - 2.0 * v;
    let dyy = dog_raw(level, x, y + 1) + dog_raw(level, x, y - 1) - 2.0 * v;
    let dxy = (dog_raw(level, x + 1, y + 1) - dog_raw(level, x + 1, y - 1)
        - dog_raw(level, x - 1, y + 1) + dog_raw(level, x - 1, y - 1)) / 4.0;
    let tr = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    let e1 = p.edge + 1.0;
    if (det <= 0.0 || tr * tr * p.edge > det * e1 * e1) {
        return;
    }
    if (p.subpixel == 0u) {
        let slot = atomicAdd(&counters[p.octave], 1u);
        if (slot < p.cand_cap) {
            cands[slot] = Candidate(u32(x), u32(y), level, v, 0.0, 0.0, 0.0);
        }
        return;
    }

    var cx: i32 = x;
    var cy: i32 = y;
    var clevel: u32 = level;
    var dv = v;
    var gx = 0.0;
    var gy = 0.0;
    var gs = 0.0;
    var hxx = dxx;
    var hyy = dyy;
    var hxy = dxy;
    var ox = 0.0;
    var oy = 0.0;
    var os = 0.0;
    for (var iteration = 0u; iteration < 5u; iteration = iteration + 1u) {
        // Each Newton fit is relative to the current integer sample.
        dv = dog_raw(clevel, cx, cy);
        gx = 0.5 * (dog_raw(clevel, cx + 1, cy) - dog_raw(clevel, cx - 1, cy));
        gy = 0.5 * (dog_raw(clevel, cx, cy + 1) - dog_raw(clevel, cx, cy - 1));
        gs = 0.5 * (dog_raw(clevel + 1u, cx, cy) - dog_raw(clevel - 1u, cx, cy));
        hxx = dog_raw(clevel, cx + 1, cy) + dog_raw(clevel, cx - 1, cy) - 2.0 * dv;
        hyy = dog_raw(clevel, cx, cy + 1) + dog_raw(clevel, cx, cy - 1) - 2.0 * dv;
        let hss = dog_raw(clevel + 1u, cx, cy) + dog_raw(clevel - 1u, cx, cy) - 2.0 * dv;
        hxy = (dog_raw(clevel, cx + 1, cy + 1) - dog_raw(clevel, cx + 1, cy - 1)
            - dog_raw(clevel, cx - 1, cy + 1) + dog_raw(clevel, cx - 1, cy - 1)) / 4.0;
        let hxs = (dog_raw(clevel + 1u, cx + 1, cy) - dog_raw(clevel + 1u, cx - 1, cy)
            - dog_raw(clevel - 1u, cx + 1, cy) + dog_raw(clevel - 1u, cx - 1, cy)) / 4.0;
        let hys = (dog_raw(clevel + 1u, cx, cy + 1) - dog_raw(clevel + 1u, cx, cy - 1)
            - dog_raw(clevel - 1u, cx, cy + 1) + dog_raw(clevel - 1u, cx, cy - 1)) / 4.0;

        // Symmetric adjugate / Cramer's rule: H * offset = -gradient.
        let cxx = hyy * hss - hys * hys;
        let cxy = hxs * hys - hxy * hss;
        let cxs = hxy * hys - hxs * hyy;
        let cyy = hxx * hss - hxs * hxs;
        let cys = hxy * hxs - hxx * hys;
        let css = hxx * hyy - hxy * hxy;
        let det3 = hxx * cxx + hxy * cxy + hxs * cxs;
        if (abs(det3) <= 1e-6) {
            // Preserve the original valid detection even if a retry is singular.
            cx = x;
            cy = y;
            clevel = level;
            dv = v;
            hxx = dxx;
            hyy = dyy;
            hxy = dxy;
            ox = 0.0;
            oy = 0.0;
            os = 0.0;
            break;
        }
        ox = -(cxx * gx + cxy * gy + cxs * gs) / det3;
        oy = -(cxy * gx + cyy * gy + cys * gs) / det3;
        os = -(cxs * gx + cys * gy + css * gs) / det3;
        if (abs(ox) <= 0.5 && abs(oy) <= 0.5) {
            os = clamp(os, -1.0, 1.0);
            break;
        }
        if (iteration < 4u) {
            cx = clamp(cx + i32(round(ox)), 1, i32(p.w) - 2);
            cy = clamp(cy + i32(round(oy)), 1, i32(p.h) - 2);
            let new_level = clamp(i32(clevel) + i32(round(os)), 1, i32(p.num_dog_levels) - 2);
            clevel = u32(new_level);
        }
    }
    if (abs(ox) > 0.5 || abs(oy) > 0.5) {
        return;
    }
    let final_tr = hxx + hyy;
    let final_det = hxx * hyy - hxy * hxy;
    if (final_det <= 0.0 || final_tr * final_tr * p.edge > final_det * e1 * e1) {
        return;
    }
    let refined = dv + 0.5 * (gx * ox + gy * oy + gs * os);
    if (abs(refined) < p.contrast) {
        return;
    }
    let slot = atomicAdd(&counters[p.octave], 1u);
    if (slot < p.cand_cap) {
        cands[slot] = Candidate(u32(cx), u32(cy), clevel, refined, ox, oy, os);
    }
}

// One thread per interior pixel; walk the overlapping DoG levels in z so
// their neighborhoods can share cache lines.
@compute @workgroup_size(16, 16)
fn extrema(@builtin(global_invocation_id) g: vec3<u32>) {
    let x = i32(g.x);
    let y = i32(g.y);
    if (x < 1 || y < 1 || x >= i32(p.w) - 1 || y >= i32(p.h) - 1) {
        return;
    }
    for (var level = 1u; level < p.num_dog_levels - 1u; level = level + 1u) {
        detect_level(level, x, y);
    }
}

// Indirect dispatch arguments for the orientation pass of this octave.
@compute @workgroup_size(1)
fn orientation_args() {
    let n = min(atomicLoad(&counters[p.octave]), p.cand_cap);
    let x = min(n, 65535u);
    let y = select(1u, (n + 65534u) / 65535u, n > 65535u);
    args[p.octave * 3u + 0u] = x;
    args[p.octave * 3u + 1u] = y;
    args[p.octave * 3u + 2u] = 1u;
}

const OT: u32 = 64u;
var<workgroup> s_bin: array<u32, 64>;
var<workgroup> s_val: array<f32, 64>;
var<workgroup> hist: array<f32, 36>;

// One workgroup per candidate. The legacy histogram reads the DoG gradient
// at row `y` for `gx` and column `x + wx` for `gy`, independent of `wy`
// (only the gaussian weight depends on it), so the window collapses to one
// row of samples weighted by the column sum of the 2D gaussian.
@compute @workgroup_size(64)
fn orientation(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ci = wid.y * 65535u + wid.x;
    let n = min(atomicLoad(&counters[p.octave]), p.cand_cap);
    if (ci >= n) {
        return;
    }
    let t = lid.x;
    let c = cands[ci];
    let x = i32(c.x);
    let y = i32(c.y);
    let hist_sigma = p.hist_base * pow(p.k, f32(c.level) + c.ds) / f32(1u << p.octave);
    let radius = i32(ceil(hist_sigma * 3.0));
    let inv2s2 = 1.0 / (2.0 * hist_sigma * hist_sigma);
    var col_sum = 0.0;
    for (var wy = -radius; wy <= radius; wy = wy + 1) {
        col_sum = col_sum + exp(-f32(wy * wy) * inv2s2);
    }
    if (t < 36u) {
        hist[t] = 0.0;
    }
    let side = u32(2 * radius + 1);
    for (var base = 0u; base < side; base = base + OT) {
        let s = base + t;
        var bin = 36u;
        var val = 0.0;
        if (s < side) {
            let wx = i32(s) - radius;
            let gx = 0.5 * (dog_at(c.level, x + wx + 1, y) - dog_at(c.level, x + wx - 1, y));
            let gy = 0.5 * (dog_at(c.level, x + wx, y + 1) - dog_at(c.level, x + wx, y - 1));
            let m = sqrt(gx * gx + gy * gy);
            if (m > 1.1920929e-7) {
                var deg = degrees(atan2(gy, gx));
                if (deg < 0.0) {
                    deg = deg + 360.0;
                }
                bin = min(u32(deg / 10.0), 35u);
                val = m * exp(-f32(wx * wx) * inv2s2) * col_sum;
            }
        }
        s_bin[t] = bin;
        s_val[t] = val;
        workgroupBarrier();
        if (t < 36u) {
            var acc = hist[t];
            for (var j = 0u; j < OT; j = j + 1u) {
                if (s_bin[j] == t) {
                    acc = acc + s_val[j];
                }
            }
            hist[t] = acc;
        }
        workgroupBarrier();
    }
    if (t != 0u) {
        return;
    }
    var max_hist = 0.0;
    for (var b = 0u; b < 36u; b = b + 1u) {
        max_hist = max(max_hist, hist[b]);
    }
    if (max_hist <= 0.0) {
        return;
    }
    for (var b = 0u; b < 36u; b = b + 1u) {
        let hb = hist[b];
        if (hb < 0.8 * max_hist) {
            continue;
        }
        if (p.max_orientations > 0u) {
            // Keep the strongest peaks (ties by lower bin), as the host's
            // stable sort + truncate does.
            var rank = 0u;
            for (var o = 0u; o < 36u; o = o + 1u) {
                let ho = hist[o];
                if (ho >= 0.8 * max_hist && (ho > hb || (ho == hb && o < b))) {
                    rank = rank + 1u;
                }
            }
            if (rank >= p.max_orientations) {
                continue;
            }
        }
        let prev = hist[(b + 35u) % 36u];
        let next = hist[(b + 1u) % 36u];
        let denom = 2.0 * (hb * 2.0 - prev - next);
        var delta = 0.0;
        if (abs(denom) > 1.1920929e-7) {
            delta = clamp((prev - next) / denom, -0.5, 0.5);
        }
        var deg = (f32(b) + delta) * 10.0;
        deg = deg - 360.0 * floor(deg / 360.0);
        let slot = atomicAdd(&counters[KP_COUNTER], 1u);
        if (slot < p.kp_cap) {
            kps[slot] = OrientedKp(c.x, c.y, (p.octave << 8u) | c.level, b, radians(deg), abs(c.value),
                bitcast<u32>(c.dx), bitcast<u32>(c.dy), bitcast<u32>(c.ds));
        }
    }
}
