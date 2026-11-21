// 128-D descriptors (port of the legacy isotropic `describe_raw` +
// `normalize_sift_descriptor`): gradients are bilinear central differences
// on the source image over a (2 half + 1)^2 grid of integer offsets, rotated
// into the keypoint frame, weighted by a gaussian of width 2 * cell and
// trilinearly binned into 4 x 4 cells x 8 orientations.
//
// One workgroup per keypoint. Each 256-sample tile is evaluated once (one
// sample per thread) into shared memory, then gathered by 16 cells x 16
// lanes that keep their cell's 8 orientation bins in registers: no atomics
// and a fixed summation order (deterministic output).

struct DescParams {
    num: u32,
    img_w: u32,
    img_h: u32,
    // 0 = L2 (clip 0.2, renormalize), 1 = L1-root
    normalization: u32,
    magnification: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Kp {
    x: f32,
    y: f32,
    sigma: f32,
    orientation: f32,
};

@group(0) @binding(0) var<uniform> p: DescParams;
@group(0) @binding(1) var<storage, read> image: array<f32>;
@group(0) @binding(2) var<storage, read> kps: array<Kp>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

const TAU: f32 = 6.283185307179586;
const WG: u32 = 256u;

var<workgroup> s_val: array<f32, 256>;
var<workgroup> s_cx: array<f32, 256>;
var<workgroup> s_cy: array<f32, 256>;
var<workgroup> s_ob: array<f32, 256>;
var<workgroup> red: array<f32, 2048>;
var<workgroup> desc: array<f32, 128>;
var<workgroup> norm: f32;

fn pix(x: i32, y: i32) -> f32 {
    let xi = u32(clamp(x, 0, i32(p.img_w) - 1));
    let yi = u32(clamp(y, 0, i32(p.img_h) - 1));
    return image[yi * p.img_w + xi];
}

fn bilinear(x: f32, y: f32) -> f32 {
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let ix = i32(x0);
    let iy = i32(y0);
    return pix(ix, iy) * (1.0 - fx) * (1.0 - fy)
        + pix(ix + 1, iy) * fx * (1.0 - fy)
        + pix(ix, iy + 1) * (1.0 - fx) * fy
        + pix(ix + 1, iy + 1) * fx * fy;
}

@compute @workgroup_size(256)
fn describe(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let ki = wid.y * 65535u + wid.x;
    if (ki >= p.num) {
        return;
    }
    let t = lid.x;
    let kp = kps[ki];
    let sigma = max(kp.sigma, 1e-6);
    let cell = max(max(p.magnification, 1e-6) * sigma, 3.0);
    let half = i32(cell * 2.0);
    let side = u32(2 * half + 1);
    let total = side * side;
    let c = cos(kp.orientation);
    let s = sin(kp.orientation);
    let inv_w = 1.0 / (2.0 * (cell * 2.0) * (cell * 2.0));

    let my_cell = t / 16u;
    let lane = t % 16u;
    let ci = f32(my_cell % 4u);
    let ri = f32(my_cell / 4u);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    let b0 = vec4<f32>(0.0, 1.0, 2.0, 3.0);
    let b1 = vec4<f32>(4.0, 5.0, 6.0, 7.0);

    for (var base = 0u; base < total; base = base + WG) {
        let si = base + t;
        var val = 0.0;
        var cx = -10.0;
        var cy = -10.0;
        var ob = 0.0;
        if (si < total) {
            let dx = f32(i32(si % side) - half);
            let dy = f32(i32(si / side) - half);
            let rx = dx * c - dy * s;
            let ry = dx * s + dy * c;
            let px = kp.x + rx;
            let py = kp.y + ry;
            let gx = 0.5 * (bilinear(px + 1.0, py) - bilinear(px - 1.0, py));
            let gy = 0.5 * (bilinear(px, py + 1.0) - bilinear(px, py - 1.0));
            let m = sqrt(gx * gx + gy * gy);
            if (m > 1.1920929e-7) {
                let theta = atan2(gy, gx) - kp.orientation;
                let wrapped = theta - TAU * floor(theta / TAU);
                ob = (wrapped / TAU) * 8.0;
                ob = ob - 8.0 * floor(ob / 8.0);
                val = m * exp(-(rx * rx + ry * ry) * inv_w);
                // Bins in the keypoint frame (dx, dy), as the CPU `describe_raw`.
                cx = dx / cell + 1.5;
                cy = dy / cell + 1.5;
            }
        }
        s_val[t] = val;
        s_cx[t] = cx;
        s_cy[t] = cy;
        s_ob[t] = ob;
        workgroupBarrier();
        for (var j = lane; j < WG; j = j + 16u) {
            let wxy = max(0.0, 1.0 - abs(s_cx[j] - ci)) * max(0.0, 1.0 - abs(s_cy[j] - ri));
            if (wxy > 0.0) {
                let v = s_val[j] * wxy;
                let o = vec4<f32>(s_ob[j]);
                var d0 = abs(o - b0);
                var d1 = abs(o - b1);
                d0 = min(d0, vec4<f32>(8.0) - d0);
                d1 = min(d1, vec4<f32>(8.0) - d1);
                acc0 = acc0 + v * max(vec4<f32>(0.0), vec4<f32>(1.0) - d0);
                acc1 = acc1 + v * max(vec4<f32>(0.0), vec4<f32>(1.0) - d1);
            }
        }
        workgroupBarrier();
    }

    for (var b = 0u; b < 4u; b = b + 1u) {
        red[t * 8u + b] = acc0[b];
        red[t * 8u + 4u + b] = acc1[b];
    }
    workgroupBarrier();
    if (t < 128u) {
        let cell_id = t / 8u;
        let o = t % 8u;
        var sum = 0.0;
        for (var l = 0u; l < 16u; l = l + 1u) {
            sum = sum + red[(cell_id * 16u + l) * 8u + o];
        }
        desc[t] = sum;
    }
    workgroupBarrier();
    if (t == 0u) {
        var n2 = 0.0;
        for (var i = 0u; i < 128u; i = i + 1u) {
            let v = desc[i];
            n2 = select(n2 + v * v, n2 + abs(v), p.normalization == 1u);
        }
        norm = n2;
    }
    workgroupBarrier();
    if (p.normalization == 1u) {
        // L1-root: v = sqrt(v / l1), then L2 (unit length already up to
        // rounding; renormalize as the CPU path does).
        let l1 = norm;
        var v = desc[min(t, 127u)];
        if (l1 > 1e-30) {
            v = sqrt(max(v / l1, 0.0));
        }
        workgroupBarrier();
        if (t < 128u) {
            desc[t] = v;
        }
    } else {
        let l2 = sqrt(norm);
        var v = desc[min(t, 127u)];
        if (l2 > 1e-30) {
            v = v / l2;
        }
        v = min(v, 0.2);
        workgroupBarrier();
        if (t < 128u) {
            desc[t] = v;
        }
    }
    workgroupBarrier();
    if (t == 0u) {
        var n2 = 0.0;
        for (var i = 0u; i < 128u; i = i + 1u) {
            n2 = n2 + desc[i] * desc[i];
        }
        norm = sqrt(n2);
    }
    workgroupBarrier();
    if (t < 128u) {
        var v = desc[t];
        if (norm > 1e-30) {
            v = v / norm;
        }
        out[ki * 128u + t] = v;
    }
}
