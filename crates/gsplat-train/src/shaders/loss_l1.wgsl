// L1 photometric loss gradient: d_image = weight * sign(render - gt) /
// (3 * npix), the gradient of weight * mean |render - gt| over all pixels and
// channels (the SSIM term, if any, is added afterwards). Also accumulates the
// unweighted mean L1 (fixed point, 1e-6 units) for logging. Each workgroup
// sums its pixels' shares before the fixed-point rounding: one pixel's share
// is ~1e-8 at 2.8 MP, which rounds to 0 on its own. A ground-truth alpha
// below 128 marks a masked-out pixel: no gradient, no loss.

struct LossUniforms {
    npix: u32,
    weight: f32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> lu: LossUniforms;
// Rendered image: row-major RGB f32.
@group(0) @binding(1) var<storage, read> render: array<f32>;
// Ground truth: one packed RGBA8 u32 per pixel (R in the low byte; alpha
// 0 = masked out).
@group(0) @binding(2) var<storage, read> gt: array<u32>;
@group(0) @binding(3) var<storage, read_write> d_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> loss_acc: atomic<u32>;

var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn loss_l1(
    @builtin(global_invocation_id) gid3: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let i = gid3.x + gid3.y * nwg.x * 256u;
    var l = 0.0;
    if (i < lu.npix) {
        let packed = gt[i];
        let g = vec3<f32>(
            f32(packed & 0xFFu),
            f32((packed >> 8u) & 0xFFu),
            f32((packed >> 16u) & 0xFFu),
        ) / 255.0;
        let r = vec3<f32>(render[i * 3u], render[i * 3u + 1u], render[i * 3u + 2u]);
        let diff = r - g;
        let valid = select(0.0, 1.0, (packed >> 24u) >= 128u);
        let scale = valid / (3.0 * f32(lu.npix));
        let d = sign(diff) * (lu.weight * scale);
        d_image[i * 3u] = d.x;
        d_image[i * 3u + 1u] = d.y;
        d_image[i * 3u + 2u] = d.z;
        l = (abs(diff.x) + abs(diff.y) + abs(diff.z)) * scale;
    }
    partial[lid] = l;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            partial[lid] = partial[lid] + partial[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        atomicAdd(&loss_acc, u32(partial[0] * 1e6 + 0.5));
    }
}
