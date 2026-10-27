// Scale-space image kernels (f32 ports of visloc-vision's legacy SIFT
// `double_up`, separable `blur`, `halve` and DoG). All layers of an octave
// live in one buffer at `level * w * h`; coordinates are clamped at the
// borders exactly like `Layer::get`.

struct ImageParams {
    w: u32,
    h: u32,
    src_off: u32,
    dst_off: u32,
    radius: u32,
    wt_off: u32,
    src_w: u32,
    src_h: u32,
    // Only `blur_v_dog` reads these: the DoG plane offset in binding 4 and
    // the previous Gaussian level offset in the destination buffer.
    dog_off: u32,
    prev_off: u32,
};

@group(0) @binding(0) var<uniform> p: ImageParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> dog_out: array<f32>;

fn src_at(x: i32, y: i32, w: u32, h: u32) -> f32 {
    let xi = u32(clamp(x, 0, i32(w) - 1));
    let yi = u32(clamp(y, 0, i32(h) - 1));
    return src[p.src_off + yi * w + xi];
}

// Nearest-neighbour 2x upsampling at half-sample offsets (`double_up`).
//
// NOTE: this indexes the source with `sx = (x+2)/2` (integer division),
// which puts source pixel `i` at doubled indices `{2i-2, 2i-1}` instead of
// the `{2i, 2i+1}` the keypoint-coordinate formula (`x_orig = x_doubled *
// upsample`) assumes. That is a ~0.7-original-pixel systematic bias,
// confirmed against COLMAP's own keypoints
// (`docs/euroc_gpu_sfm_vs_colmap.md`, 2026-09-28). Kept as-is for the
// legacy byte-identical contract; `upsample2x_aligned` below is the fix.
@compute @workgroup_size(16, 16)
fn upsample2x(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.w || g.y >= p.h) {
        return;
    }
    let sx = min((g.x + 2u) / 2u, p.src_w - 1u);
    let sy = min((g.y + 2u) / 2u, p.src_h - 1u);
    dst[p.dst_off + g.y * p.w + g.x] = src[p.src_off + sy * p.src_w + sx];
}

// Align-corners bilinear 2x upsampling: `dst(2i) = src(i)`,
// `dst(2i+1) = lerp(src(i), src(i+1), 0.5)`, matching VLFeat/COLMAP's
// `copy_and_upsample_rows` and the CPU `double_up_vlfeat` this mirrors.
// Unbiased: doubled index `x` truly corresponds to original coordinate
// `x/2`, consistent with the keypoint-coordinate formula.
@compute @workgroup_size(16, 16)
fn upsample2x_aligned(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.w || g.y >= p.h) {
        return;
    }
    let sx0 = min(g.x / 2u, p.src_w - 1u);
    let sx1 = min(sx0 + 1u, p.src_w - 1u);
    let sy0 = min(g.y / 2u, p.src_h - 1u);
    let sy1 = min(sy0 + 1u, p.src_h - 1u);
    let fx = select(0.0, 0.5, (g.x % 2u) == 1u);
    let fy = select(0.0, 0.5, (g.y % 2u) == 1u);
    let p00 = src[p.src_off + sy0 * p.src_w + sx0];
    let p10 = src[p.src_off + sy0 * p.src_w + sx1];
    let p01 = src[p.src_off + sy1 * p.src_w + sx0];
    let p11 = src[p.src_off + sy1 * p.src_w + sx1];
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bot = p01 * (1.0 - fx) + p11 * fx;
    dst[p.dst_off + g.y * p.w + g.x] = top * (1.0 - fy) + bot * fy;
}

@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) g: vec3<u32>) {
    // Four adjacent columns per invocation share taps and weight loads.
    let x0 = i32(4u * g.x);
    if (x0 >= i32(p.w) || g.y >= p.h) {
        return;
    }
    let r = i32(p.radius);
    let w = i32(p.w);
    let row = p.src_off + g.y * p.w;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    if (x0 >= r && x0 + 3 < w - r) {
        let base = row + u32(x0 - r);
        for (var t = 0u; t <= 2u * u32(r); t = t + 1u) {
            let wv = weights[p.wt_off + t];
            acc0 = acc0 + wv * src[base + t];
            acc1 = acc1 + wv * src[base + 1u + t];
            acc2 = acc2 + wv * src[base + 2u + t];
            acc3 = acc3 + wv * src[base + 3u + t];
        }
    } else {
        for (var k = -r; k <= r; k = k + 1) {
            let wv = weights[p.wt_off + u32(k + r)];
            acc0 = acc0 + wv * src[row + u32(clamp(x0 + k, 0, w - 1))];
            if (x0 + 1 < w) {
                acc1 = acc1 + wv * src[row + u32(clamp(x0 + 1 + k, 0, w - 1))];
            }
            if (x0 + 2 < w) {
                acc2 = acc2 + wv * src[row + u32(clamp(x0 + 2 + k, 0, w - 1))];
            }
            if (x0 + 3 < w) {
                acc3 = acc3 + wv * src[row + u32(clamp(x0 + 3 + k, 0, w - 1))];
            }
        }
    }
    let out = p.dst_off + g.y * p.w + u32(x0);
    dst[out] = acc0;
    if (x0 + 1 < w) {
        dst[out + 1u] = acc1;
    }
    if (x0 + 2 < w) {
        dst[out + 2u] = acc2;
    }
    if (x0 + 3 < w) {
        dst[out + 3u] = acc3;
    }
}

@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) g: vec3<u32>) {
    // Four adjacent rows per invocation share taps and weight loads.
    let y0 = i32(4u * g.y);
    if (g.x >= p.w || y0 >= i32(p.h)) {
        return;
    }
    let r = i32(p.radius);
    let h = i32(p.h);
    let w = p.w;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    if (y0 >= r && y0 + 3 < h - r) {
        let base = p.src_off + u32(y0 - r) * w + g.x;
        for (var t = 0u; t <= 2u * u32(r); t = t + 1u) {
            let wv = weights[p.wt_off + t];
            acc0 = acc0 + wv * src[base + t * w];
            acc1 = acc1 + wv * src[base + (t + 1u) * w];
            acc2 = acc2 + wv * src[base + (t + 2u) * w];
            acc3 = acc3 + wv * src[base + (t + 3u) * w];
        }
    } else {
        for (var k = -r; k <= r; k = k + 1) {
            let wv = weights[p.wt_off + u32(k + r)];
            acc0 = acc0 + wv * src[p.src_off + u32(clamp(y0 + k, 0, h - 1)) * w + g.x];
            if (y0 + 1 < h) {
                acc1 = acc1 + wv * src[p.src_off + u32(clamp(y0 + 1 + k, 0, h - 1)) * w + g.x];
            }
            if (y0 + 2 < h) {
                acc2 = acc2 + wv * src[p.src_off + u32(clamp(y0 + 2 + k, 0, h - 1)) * w + g.x];
            }
            if (y0 + 3 < h) {
                acc3 = acc3 + wv * src[p.src_off + u32(clamp(y0 + 3 + k, 0, h - 1)) * w + g.x];
            }
        }
    }
    let out = p.dst_off + u32(y0) * p.w + g.x;
    dst[out] = acc0;
    if (y0 + 1 < h) {
        dst[out + p.w] = acc1;
    }
    if (y0 + 2 < h) {
        dst[out + 2u * p.w] = acc2;
    }
    if (y0 + 3 < h) {
        dst[out + 3u * p.w] = acc3;
    }
}

// Vertical blur fused with the DoG subtraction: writes Gaussian level j and
// DoG level j-1 in one dispatch, reading Gaussian level j-1 already in `dst`.
// Four adjacent rows per invocation like `blur_v`.
@compute @workgroup_size(16, 16)
fn blur_v_dog(@builtin(global_invocation_id) g: vec3<u32>) {
    let y0 = i32(4u * g.y);
    if (g.x >= p.w || y0 >= i32(p.h)) {
        return;
    }
    let r = i32(p.radius);
    let h = i32(p.h);
    let w = p.w;
    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    if (y0 >= r && y0 + 3 < h - r) {
        let base = p.src_off + u32(y0 - r) * w + g.x;
        for (var t = 0u; t <= 2u * u32(r); t = t + 1u) {
            let wv = weights[p.wt_off + t];
            acc0 = acc0 + wv * src[base + t * w];
            acc1 = acc1 + wv * src[base + (t + 1u) * w];
            acc2 = acc2 + wv * src[base + (t + 2u) * w];
            acc3 = acc3 + wv * src[base + (t + 3u) * w];
        }
    } else {
        for (var k = -r; k <= r; k = k + 1) {
            let wv = weights[p.wt_off + u32(k + r)];
            acc0 = acc0 + wv * src[p.src_off + u32(clamp(y0 + k, 0, h - 1)) * w + g.x];
            if (y0 + 1 < h) {
                acc1 = acc1 + wv * src[p.src_off + u32(clamp(y0 + 1 + k, 0, h - 1)) * w + g.x];
            }
            if (y0 + 2 < h) {
                acc2 = acc2 + wv * src[p.src_off + u32(clamp(y0 + 2 + k, 0, h - 1)) * w + g.x];
            }
            if (y0 + 3 < h) {
                acc3 = acc3 + wv * src[p.src_off + u32(clamp(y0 + 3 + k, 0, h - 1)) * w + g.x];
            }
        }
    }
    let row = u32(y0) * p.w + g.x;
    dst[p.dst_off + row] = acc0;
    dog_out[p.dog_off + row] = acc0 - dst[p.prev_off + row];
    if (y0 + 1 < h) {
        dst[p.dst_off + row + p.w] = acc1;
        dog_out[p.dog_off + row + p.w] = acc1 - dst[p.prev_off + row + p.w];
    }
    if (y0 + 2 < h) {
        dst[p.dst_off + row + 2u * p.w] = acc2;
        dog_out[p.dog_off + row + 2u * p.w] = acc2 - dst[p.prev_off + row + 2u * p.w];
    }
    if (y0 + 3 < h) {
        dst[p.dst_off + row + 3u * p.w] = acc3;
        dog_out[p.dog_off + row + 3u * p.w] = acc3 - dst[p.prev_off + row + 3u * p.w];
    }
}

// Decimate by two (`halve`): dst(x, y) = src(2x, 2y).
@compute @workgroup_size(16, 16)
fn halve(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.w || g.y >= p.h) {
        return;
    }
    dst[p.dst_off + g.y * p.w + g.x] = src_at(i32(2u * g.x), i32(2u * g.y), p.src_w, p.src_h);
}

// DoG level z = gaussian(z + 1) - gaussian(z).
@compute @workgroup_size(16, 16)
fn dog(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.w || g.y >= p.h) {
        return;
    }
    let n = p.w * p.h;
    let i = g.y * p.w + g.x;
    dst[g.z * n + i] = src[(g.z + 1u) * n + i] - src[g.z * n + i];
}
