// D-SSIM loss term: adds lambda * d(1 - mean SSIM)/d render to d_image
// (the L1 kernel has already written (1 - lambda) * its own gradient).
//
// SSIM per channel with an 11x11 gaussian window (sigma 1.5), zero padding
// (as F.conv2d(padding=5) in the Inria code), C1 = 0.01^2, C2 = 0.03^2:
//   S = (2 mx my + C1)(2 sxy + C2) / ((mx^2 + my^2 + C1)(sx^2 + sy^2 + C2))
// with mx = G*x, Sxx = G*(x^2), Sxy = G*(xy), sx^2 = Sxx - mx^2,
// sxy = Sxy - mx my. With A = dS/dmx, B = dS/dSxx, C = dS/dSxy at each
// window centre q, the gradient of sum_q S_q w.r.t. pixel p is
//   (G*A)_p + 2 x_p (G*B)_p + y_p (G*C)_p
// (G is symmetric, so correlation = convolution).
//
// Two tiled kernels, one workgroup per 16x16 output tile: each stages its
// (16 + 2 * 5)^2 input apron in shared memory and does both separable blur
// passes there, one channel at a time (keeps shared memory under 16 KiB).
//   ssim_fwd: render + ground truth -> SSIM sum and the A, B, C maps
//   ssim_bwd: A, B, C maps -> d_image += -lambda/(3N) * dS/dx
// A masked-out pixel (ground-truth alpha below 128) neither centres a
// window (its A, B, C are 0 and its SSIM is not summed) nor receives a
// gradient.

struct SsimUniforms {
    width: u32,
    height: u32,
    // lambda / (3 * npix): d(lambda * (1 - mean SSIM)) / dS_q per channel.
    scale: f32,
    pad0: u32,
};

@group(0) @binding(0) var<uniform> su: SsimUniforms;
@group(0) @binding(1) var<storage, read> render: array<f32>;
@group(0) @binding(2) var<storage, read> gt: array<u32>;
// 9 floats per pixel: per channel (A, B, C).
@group(0) @binding(4) var<storage, read_write> abc: array<f32>;
@group(0) @binding(6) var<storage, read_write> d_image: array<f32>;
// Sum of SSIM over pixels and channels, fixed point (1e-3 units).
@group(0) @binding(7) var<storage, read_write> ssim_acc: atomic<u32>;

const RADIUS: i32 = 5;
const T: u32 = 16u;
const A: u32 = 26u; // T + 2 * RADIUS
const C1: f32 = 0.0001;
const C2: f32 = 0.0009;

// Apron of two inputs (x, y) and the horizontal pass of up to 5 quantities.
var<workgroup> in0: array<f32, 676>; // A * A
var<workgroup> in1: array<f32, 676>;
var<workgroup> in2: array<f32, 676>;
var<workgroup> hq: array<f32, 2080>; // A rows * T cols * 5
var<workgroup> ssim_part: atomic<u32>;

fn gauss_w(k: i32) -> f32 {
    // Normalised 1D gaussian, sigma 1.5, taps -5..5.
    let w = array<f32, 11>(
        0.0010284, 0.0075988, 0.0360008, 0.1093607, 0.2130055, 0.2660117,
        0.2130055, 0.1093607, 0.0360008, 0.0075988, 0.0010284,
    );
    return w[k + RADIUS];
}

fn gt_channel(i: u32, c: u32) -> f32 {
    return f32((gt[i] >> (8u * c)) & 0xFFu) / 255.0;
}

fn gt_valid(i: u32) -> bool {
    return (gt[i] >> 24u) >= 128u;
}

// Global pixel of apron cell (ax, ay) of tile (tx, ty), or -1 when outside.
fn apron_pixel(tx: u32, ty: u32, ax: u32, ay: u32) -> i32 {
    let gx = i32(tx * T + ax) - RADIUS;
    let gy = i32(ty * T + ay) - RADIUS;
    if (gx < 0 || gy < 0 || gx >= i32(su.width) || gy >= i32(su.height)) {
        return -1;
    }
    return gy * i32(su.width) + gx;
}

@compute @workgroup_size(256)
fn ssim_fwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tiles_x = (su.width + T - 1u) / T;
    let tx = wid.x % tiles_x;
    let ty = wid.x / tiles_x;
    let t = lid.x;
    let ox = t % T;
    let oy = t / T;
    let px = tx * T + ox;
    let py = ty * T + oy;
    let inside = px < su.width && py < su.height;
    if (t == 0u) {
        atomicStore(&ssim_part, 0u);
    }
    var s_sum = 0.0;
    for (var c = 0u; c < 3u; c = c + 1u) {
        // Stage x and y (zero outside the image).
        for (var i = t; i < A * A; i = i + 256u) {
            let p = apron_pixel(tx, ty, i % A, i / A);
            var xv = 0.0;
            var yv = 0.0;
            if (p >= 0) {
                xv = render[u32(p) * 3u + c];
                yv = gt_channel(u32(p), c);
            }
            in0[i] = xv;
            in1[i] = yv;
        }
        workgroupBarrier();
        // Horizontal pass for every apron row, the tile's 16 columns.
        for (var i = t; i < A * T; i = i + 256u) {
            let row = i / T;
            let col = i % T;
            var m_x = 0.0;
            var m_y = 0.0;
            var s_xx = 0.0;
            var s_yy = 0.0;
            var s_xy = 0.0;
            for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
                let w = gauss_w(k);
                let j = row * A + u32(i32(col) + RADIUS + k);
                let xv = in0[j];
                let yv = in1[j];
                m_x = m_x + w * xv;
                m_y = m_y + w * yv;
                s_xx = s_xx + w * xv * xv;
                s_yy = s_yy + w * yv * yv;
                s_xy = s_xy + w * xv * yv;
            }
            hq[i * 5u + 0u] = m_x;
            hq[i * 5u + 1u] = m_y;
            hq[i * 5u + 2u] = s_xx;
            hq[i * 5u + 3u] = s_yy;
            hq[i * 5u + 4u] = s_xy;
        }
        workgroupBarrier();
        // Vertical pass for this thread's pixel -> SSIM and its partials.
        var mx = 0.0;
        var my = 0.0;
        var sxx = 0.0;
        var syy = 0.0;
        var sxy = 0.0;
        for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
            let w = gauss_w(k);
            let j = (u32(i32(oy) + RADIUS + k) * T + ox) * 5u;
            mx = mx + w * hq[j];
            my = my + w * hq[j + 1u];
            sxx = sxx + w * hq[j + 2u];
            syy = syy + w * hq[j + 3u];
            sxy = sxy + w * hq[j + 4u];
        }
        if (inside && gt_valid(py * su.width + px)) {
            let vx = sxx - mx * mx;
            let vy = syy - my * my;
            let cxy = sxy - mx * my;
            let a = 2.0 * mx * my + C1;
            let b = 2.0 * cxy + C2;
            let cc = mx * mx + my * my + C1;
            let d = vx + vy + C2;
            let s = a * b / (cc * d);
            s_sum = s_sum + s;
            let i = py * su.width + px;
            abc[i * 9u + c * 3u + 0u] =
                (2.0 * my * b - 2.0 * my * a) / (cc * d) - s * (2.0 * mx / cc - 2.0 * mx / d);
            abc[i * 9u + c * 3u + 1u] = -s / d;
            abc[i * 9u + c * 3u + 2u] = 2.0 * a / (cc * d);
        } else if (inside) {
            let i = py * su.width + px;
            abc[i * 9u + c * 3u + 0u] = 0.0;
            abc[i * 9u + c * 3u + 1u] = 0.0;
            abc[i * 9u + c * 3u + 2u] = 0.0;
        }
        // in0/in1/hq are reused by the next channel.
        workgroupBarrier();
    }
    atomicAdd(&ssim_part, u32(max(s_sum, 0.0) * 1e3 + 0.5));
    workgroupBarrier();
    if (t == 0u) {
        atomicAdd(&ssim_acc, atomicLoad(&ssim_part));
    }
}

@compute @workgroup_size(256)
fn ssim_bwd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tiles_x = (su.width + T - 1u) / T;
    let tx = wid.x % tiles_x;
    let ty = wid.x / tiles_x;
    let t = lid.x;
    let ox = t % T;
    let oy = t / T;
    let px = tx * T + ox;
    let py = ty * T + oy;
    let inside = px < su.width && py < su.height;
    for (var c = 0u; c < 3u; c = c + 1u) {
        // Stage A, B, C of this channel (zero outside the image).
        for (var i = t; i < A * A; i = i + 256u) {
            let p = apron_pixel(tx, ty, i % A, i / A);
            var va = 0.0;
            var vb = 0.0;
            var vc = 0.0;
            if (p >= 0) {
                let j = u32(p) * 9u + c * 3u;
                va = abc[j];
                vb = abc[j + 1u];
                vc = abc[j + 2u];
            }
            in0[i] = va;
            in1[i] = vb;
            in2[i] = vc;
        }
        workgroupBarrier();
        for (var i = t; i < A * T; i = i + 256u) {
            let row = i / T;
            let col = i % T;
            var ga = 0.0;
            var gb = 0.0;
            var gc = 0.0;
            for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
                let w = gauss_w(k);
                let j = row * A + u32(i32(col) + RADIUS + k);
                ga = ga + w * in0[j];
                gb = gb + w * in1[j];
                gc = gc + w * in2[j];
            }
            hq[i * 5u + 0u] = ga;
            hq[i * 5u + 1u] = gb;
            hq[i * 5u + 2u] = gc;
        }
        workgroupBarrier();
        var ga = 0.0;
        var gb = 0.0;
        var gc = 0.0;
        for (var k = -RADIUS; k <= RADIUS; k = k + 1) {
            let w = gauss_w(k);
            let j = (u32(i32(oy) + RADIUS + k) * T + ox) * 5u;
            ga = ga + w * hq[j];
            gb = gb + w * hq[j + 1u];
            gc = gc + w * hq[j + 2u];
        }
        if (inside && gt_valid(py * su.width + px)) {
            let i = py * su.width + px;
            let xv = render[i * 3u + c];
            let yv = gt_channel(i, c);
            let ds_dx = ga + 2.0 * xv * gb + yv * gc;
            d_image[i * 3u + c] = d_image[i * 3u + c] - su.scale * ds_dx;
        }
        workgroupBarrier();
    }
}
