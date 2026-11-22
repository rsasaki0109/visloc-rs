//! Device-side D-SSIM loss term (tiled, see `shaders/ssim.wgsl`), shared by the
//! trainer and the GPU-vs-CPU gradient test.

/// The two tiled SSIM kernels plus their scratch buffers for one image size.
pub struct SsimKernels {
    width: u32,
    height: u32,
    fwd: wgpu::ComputePipeline,
    bwd: wgpu::ComputePipeline,
    uniforms: wgpu::Buffer,
    abc: wgpu::Buffer,
    /// Sum of SSIM over pixels and channels (fixed point, 1e-3 units).
    pub acc: wgpu::Buffer,
}

/// Bind groups for one (render, ground truth, d_image) triple.
pub struct SsimBinds {
    fwd: wgpu::BindGroup,
    bwd: wgpu::BindGroup,
}

fn storage(dev: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn bind_at(
    dev: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|(binding, b)| wgpu::BindGroupEntry {
            binding: *binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ssim"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    })
}

impl SsimKernels {
    pub fn new(dev: &wgpu::Device, width: u32, height: u32) -> Self {
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ssim"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ssim.wgsl").into()),
        });
        let mk = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let npix = width as u64 * height as u64;
        Self {
            width,
            height,
            fwd: mk("ssim_fwd"),
            bwd: mk("ssim_bwd"),
            uniforms: dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("ssim_uniforms"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            abc: storage(dev, "ssim_abc", npix * 9 * 4),
            acc: storage(dev, "ssim_acc", 4),
        }
    }

    /// Set the loss weight `lambda` of the `lambda * (1 - mean SSIM)` term.
    pub fn set_weight(&self, queue: &wgpu::Queue, lambda: f32) {
        let scale = lambda / (3.0 * self.width as f32 * self.height as f32);
        let words: [u32; 4] = [self.width, self.height, scale.to_bits(), 0];
        queue.write_buffer(&self.uniforms, 0, bytemuck::cast_slice(&words));
    }

    /// Bind a render (RGB f32), a ground truth (packed RGBA8) and the
    /// `dL/dC` buffer the gradient is added to.
    pub fn bind(
        &self,
        dev: &wgpu::Device,
        render: &wgpu::Buffer,
        gt: &wgpu::Buffer,
        d_image: &wgpu::Buffer,
    ) -> SsimBinds {
        let u = &self.uniforms;
        SsimBinds {
            fwd: bind_at(
                dev,
                &self.fwd,
                &[(0, u), (1, render), (2, gt), (4, &self.abc), (7, &self.acc)],
            ),
            bwd: bind_at(
                dev,
                &self.bwd,
                &[(0, u), (1, render), (2, gt), (4, &self.abc), (6, d_image)],
            ),
        }
    }

    /// Record the two passes (wgpu orders dispatches within a pass).
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>, binds: &SsimBinds) {
        let tiles = self.width.div_ceil(16) * self.height.div_ceil(16);
        for (p, b) in [(&self.fwd, &binds.fwd), (&self.bwd, &binds.bwd)] {
            pass.set_pipeline(p);
            pass.set_bind_group(0, b, &[]);
            let x = tiles.clamp(1, 65535);
            pass.dispatch_workgroups(x, tiles.div_ceil(x), 1);
        }
    }
}

/// Normalised 1D gaussian, sigma 1.5, taps -5..5 (`ssim.wgsl`'s window).
const GAUSS: [f32; 11] = [
    0.0010284, 0.0075988, 0.0360008, 0.1093607, 0.2130055, 0.2660117, 0.2130055, 0.1093607,
    0.0360008, 0.0075988, 0.0010284,
];

/// One output row of the horizontal 11-tap blur of `src` (zero padding),
/// per tap along the row so it vectorizes. `pad` is scratch of `w + 10`.
fn blur_row_h(src: &[f32], pad: &mut [f32], out: &mut [f32]) {
    let w = src.len();
    pad[..5].fill(0.0);
    pad[5..5 + w].copy_from_slice(src);
    pad[5 + w..].fill(0.0);
    out.fill(0.0);
    for (t, &g) in GAUSS.iter().enumerate() {
        for (o, v) in out.iter_mut().zip(&pad[t..t + w]) {
            *o += g * v;
        }
    }
}

/// The D-SSIM term on the host, for CPU adapters (lavapipe), where the
/// tiled kernels' barriers make it ~50 ms a step at 600 x 440. Same
/// definition as `ssim.wgsl`. Keeps its scratch between calls.
///
/// Rows run in parallel in four passes over row-major blocks (per image
/// row, every map's row): the horizontal blur of x, y, x^2, y^2, xy per
/// channel; the vertical blur and dS/dmx, dS/dSxx, dS/dSxy per window
/// centre; the horizontal blur of those; the vertical blur and the
/// gradient.
pub struct SsimHost {
    w: usize,
    h: usize,
    /// Horizontal blurs, 15 maps per row (5 per channel).
    hm: Vec<f32>,
    /// dS/dmx, dS/dSxx, dS/dSxy, 9 maps per row (3 per channel).
    abc: Vec<f32>,
    /// Their horizontal blurs, 9 maps per row.
    hg: Vec<f32>,
}

impl SsimHost {
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            hm: vec![0.0; w * h * 15],
            abc: vec![0.0; w * h * 9],
            hg: vec![0.0; w * h * 9],
        }
    }

    /// Add what the kernels add to `d_image` (RGB per pixel):
    /// `-lambda / (3 N) * dSSIM/dx`. `render` is RGB f32, `gt` packed RGBA8
    /// (alpha below 128 = masked out).
    pub fn add_grad(&mut self, render: &[f32], gt: &[u32], lambda: f32, d_image: &mut [f32]) {
        use rayon::prelude::*;
        const C1: f32 = 0.0001;
        const C2: f32 = 0.0009;
        let (w, h) = (self.w, self.h);
        let n = w * h;
        let scale = lambda / (3.0 * n as f32);
        let gtc = |p: usize, c: usize| ((gt[p] >> (8 * c)) & 0xFF) as f32 / 255.0;
        let valid = |p: usize| (gt[p] >> 24) >= 128;
        // 1. Horizontal blur of x, y, x^2, y^2, xy.
        self.hm
            .par_chunks_mut(15 * w)
            .enumerate()
            .for_each(|(y, rows)| {
                let mut pad = vec![0.0f32; w + 10];
                let mut src = vec![0.0f32; w];
                for c in 0..3 {
                    for q in 0..5 {
                        for (x, v) in src.iter_mut().enumerate() {
                            let p = y * w + x;
                            let (a, b) = (render[p * 3 + c], gtc(p, c));
                            *v = [a, b, a * a, b * b, a * b][q];
                        }
                        let m = c * 5 + q;
                        blur_row_h(&src, &mut pad, &mut rows[m * w..(m + 1) * w]);
                    }
                }
            });
        // 2. Vertical blur, then the SSIM partials at valid centres.
        let hm = &self.hm;
        self.abc
            .par_chunks_mut(9 * w)
            .enumerate()
            .for_each(|(y, rows)| {
                let mut m = vec![0.0f32; 15 * w];
                for (t, &g) in GAUSS.iter().enumerate() {
                    let yy = y as isize + t as isize - 5;
                    if yy < 0 || yy >= h as isize {
                        continue;
                    }
                    let src = &hm[yy as usize * 15 * w..(yy as usize + 1) * 15 * w];
                    for (o, v) in m.iter_mut().zip(src) {
                        *o += g * v;
                    }
                }
                for c in 0..3 {
                    let q = |k: usize, x: usize| m[(c * 5 + k) * w + x];
                    for x in 0..w {
                        let (ra, rb, rc) = if valid(y * w + x) {
                            let (mx, my) = (q(0, x), q(1, x));
                            let a = 2.0 * mx * my + C1;
                            let b = 2.0 * (q(4, x) - mx * my) + C2;
                            let cc = mx * mx + my * my + C1;
                            let d = (q(2, x) - mx * mx) + (q(3, x) - my * my) + C2;
                            let s = a * b / (cc * d);
                            (
                                (2.0 * my * b - 2.0 * my * a) / (cc * d)
                                    - s * (2.0 * mx / cc - 2.0 * mx / d),
                                -s / d,
                                2.0 * a / (cc * d),
                            )
                        } else {
                            (0.0, 0.0, 0.0)
                        };
                        rows[(c * 3) * w + x] = ra;
                        rows[(c * 3 + 1) * w + x] = rb;
                        rows[(c * 3 + 2) * w + x] = rc;
                    }
                }
            });
        // 3. Horizontal blur of the partials.
        let abc = &self.abc;
        self.hg
            .par_chunks_mut(9 * w)
            .enumerate()
            .for_each(|(y, rows)| {
                let mut pad = vec![0.0f32; w + 10];
                for m in 0..9 {
                    let src = &abc[(y * 9 + m) * w..(y * 9 + m + 1) * w];
                    blur_row_h(src, &mut pad, &mut rows[m * w..(m + 1) * w]);
                }
            });
        // 4. Vertical blur and the gradient.
        let hg = &self.hg;
        d_image
            .par_chunks_mut(3 * w)
            .enumerate()
            .for_each(|(y, out)| {
                let mut g = vec![0.0f32; 9 * w];
                for (t, &gw) in GAUSS.iter().enumerate() {
                    let yy = y as isize + t as isize - 5;
                    if yy < 0 || yy >= h as isize {
                        continue;
                    }
                    let src = &hg[yy as usize * 9 * w..(yy as usize + 1) * 9 * w];
                    for (o, v) in g.iter_mut().zip(src) {
                        *o += gw * v;
                    }
                }
                for x in 0..w {
                    let p = y * w + x;
                    if !valid(p) {
                        continue;
                    }
                    for c in 0..3 {
                        let ds = g[(c * 3) * w + x]
                            + 2.0 * render[p * 3 + c] * g[(c * 3 + 1) * w + x]
                            + gtc(p, c) * g[(c * 3 + 2) * w + x];
                        out[x * 3 + c] += -scale * ds;
                    }
                }
            });
    }
}

/// `loss_l1.wgsl` on the host (CPU adapters): writes `d_image` =
/// `weight * sign(render - gt) / (3 N)` per pixel and channel (0 at masked
/// pixels, alpha below 128) with the kernel's operations in its order, and
/// returns the step's mean-L1 log in its fixed point (1e-6 units), summed
/// per 256-pixel workgroup with the kernel's tree reduction.
pub fn l1_grad_host(render: &[f32], gt: &[u32], weight: f32, d_image: &mut [f32]) -> u64 {
    use rayon::prelude::*;
    let npix = gt.len();
    let wgsl_sign = |v: f32| {
        if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else {
            0.0
        }
    };
    d_image
        .par_chunks_mut(256 * 3)
        .enumerate()
        .map(|(wg, d)| {
            let mut partial = [0.0f32; 256];
            for (lid, (dp, l)) in d.chunks_mut(3).zip(partial.iter_mut()).enumerate() {
                let i = wg * 256 + lid;
                let packed = gt[i];
                let g = [
                    (packed & 0xFF) as f32 / 255.0,
                    ((packed >> 8) & 0xFF) as f32 / 255.0,
                    ((packed >> 16) & 0xFF) as f32 / 255.0,
                ];
                let diff = [0, 1, 2].map(|c| render[i * 3 + c] - g[c]);
                let valid = if (packed >> 24) >= 128 { 1.0f32 } else { 0.0 };
                let scale = valid / (3.0 * npix as f32);
                let ws = weight * scale;
                for c in 0..3 {
                    dp[c] = wgsl_sign(diff[c]) * ws;
                }
                *l = (diff[0].abs() + diff[1].abs() + diff[2].abs()) * scale;
            }
            let mut stride = 128;
            while stride > 0 {
                for lid in 0..stride {
                    partial[lid] += partial[lid + stride];
                }
                stride >>= 1;
            }
            (partial[0] * 1e6 + 0.5) as u32 as u64
        })
        .sum()
}

/// [`SsimHost::add_grad`] into a zero image (tests).
pub fn ssim_grad_host(render: &[f32], gt: &[u32], w: usize, h: usize, lambda: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h * 3];
    SsimHost::new(w, h).add_grad(render, gt, lambda, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::ssim_with_grad;

    #[test]
    fn gpu_ssim_gradient_matches_cpu() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let (w, h) = (37u32, 29u32);
        let n = (w * h) as usize;
        let mut s = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let render: Vec<[f32; 3]> = (0..n).map(|_| [rnd(), rnd(), rnd()]).collect();
        // Quantise the ground truth to 8 bits, as the device stores it.
        let gt8: Vec<[u8; 3]> = (0..n)
            .map(|_| {
                [
                    (rnd() * 255.0) as u8,
                    (rnd() * 255.0) as u8,
                    (rnd() * 255.0) as u8,
                ]
            })
            .collect();
        let packed: Vec<u32> = gt8
            .iter()
            .map(|p| p[0] as u32 | (p[1] as u32) << 8 | (p[2] as u32) << 16 | 255 << 24)
            .collect();
        let dev = &ctx.device;
        let queue = &ctx.queue;
        let rb = storage(dev, "render", n as u64 * 12);
        queue.write_buffer(&rb, 0, bytemuck::cast_slice(&render));
        let gb = storage(dev, "gt", n as u64 * 4);
        queue.write_buffer(&gb, 0, bytemuck::cast_slice(&packed));
        let db = storage(dev, "d_image", n as u64 * 12);
        let k = SsimKernels::new(dev, w, h);
        let lambda = 0.2;
        k.set_weight(queue, lambda);
        let binds = k.bind(dev, &rb, &gb, &db);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            k.encode(&mut pass, &binds);
        }
        queue.submit(Some(enc.finish()));
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n as u64 * 12,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&db, 0, &staging, 0, n as u64 * 12);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        dev.poll(wgpu::PollType::wait_indefinitely()).ok();
        let gpu: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();

        let r64: Vec<[f64; 3]> = render.iter().map(|p| p.map(|v| v as f64)).collect();
        let g64: Vec<[f64; 3]> = gt8.iter().map(|p| p.map(|v| v as f64 / 255.0)).collect();
        let (_, grad) = ssim_with_grad(&r64, &g64, w as usize, h as usize, true);
        let grad = grad.unwrap();
        // d_image started at zero, so the kernel wrote -lambda * dSSIM/dx.
        let scale = grad.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max) * lambda as f64;
        let mut worst = 0.0f64;
        for (p, gp) in grad.iter().enumerate() {
            for c in 0..3 {
                let want = -(lambda as f64) * gp[c];
                let got = gpu[p * 3 + c] as f64;
                worst = worst.max((want - got).abs() / scale);
            }
        }
        eprintln!("ssim grad worst error {worst:.2e} of max");
        assert!(worst < 1e-3, "worst relative error {worst}");

        // The host version (CPU adapters) against the same reference.
        let rf: Vec<f32> = render.iter().flatten().copied().collect();
        let host = ssim_grad_host(&rf, &packed, w as usize, h as usize, lambda);
        let mut worst_host = 0.0f64;
        for (p, gp) in grad.iter().enumerate() {
            for c in 0..3 {
                let want = -(lambda as f64) * gp[c];
                worst_host = worst_host.max((want - host[p * 3 + c] as f64).abs() / scale);
            }
        }
        eprintln!("host ssim grad worst error {worst_host:.2e} of max");
        assert!(worst_host < 1e-3, "host: worst relative error {worst_host}");
    }

    #[test]
    fn host_l1_matches_kernel_definition() {
        // 300 pixels (two workgroups): a masked pixel, a zero difference,
        // and differences of both signs.
        let n = 300;
        let gt: Vec<u32> = (0..n as u32)
            .map(|i| {
                (i % 256)
                    | ((i * 7) % 256) << 8
                    | ((i * 13) % 256) << 16
                    | if i == 5 { 0 } else { 255 << 24 }
            })
            .collect();
        let mut render: Vec<f32> = (0..n * 3).map(|k| (k % 17) as f32 / 16.0).collect();
        render[3 * 9] = 9.0 / 255.0; // pixel 9, red: diff 0
        let mut d = vec![1.0f32; n * 3];
        let fixed = l1_grad_host(&render, &gt, 0.8, &mut d);
        let mut sum = 0.0f64;
        for i in 0..n {
            for c in 0..3 {
                let g = ((gt[i] >> (8 * c)) & 0xFF) as f32 / 255.0;
                let diff = render[i * 3 + c] - g;
                let valid = i != 5;
                let want = if !valid || diff == 0.0 {
                    0.0
                } else {
                    diff.signum() * 0.8 / (3.0 * n as f32)
                };
                assert_eq!(d[i * 3 + c], want, "pixel {i} channel {c}");
                if valid {
                    sum += diff.abs() as f64 / (3.0 * n as f64);
                }
            }
        }
        assert!((fixed as f64 * 1e-6 - sum).abs() < 2e-6, "{fixed} vs {sum}");
    }

    /// Sum over valid window centres and channels of the SSIM map (11x11
    /// gaussian, sigma 1.5, zero padding: the kernel's definition).
    fn masked_ssim_sum(x: &[[f64; 3]], y: &[[f64; 3]], valid: &[bool], w: usize, h: usize) -> f64 {
        let g: Vec<f64> = (-5..=5i32)
            .map(|k| (-(k * k) as f64 / (2.0 * 1.5 * 1.5)).exp())
            .collect();
        let gs: f64 = g.iter().sum();
        let (c1, c2) = (1e-4, 9e-4);
        let mut total = 0.0;
        for (q, _) in valid.iter().enumerate().filter(|(_, &v)| v) {
            let (qx, qy) = ((q % w) as i32, (q / w) as i32);
            for c in 0..3 {
                let (mut mx, mut my, mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
                for dy in -5..=5i32 {
                    for dx in -5..=5i32 {
                        let (px, py) = (qx + dx, qy + dy);
                        if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                            continue;
                        }
                        let wt = g[(dx + 5) as usize] * g[(dy + 5) as usize] / (gs * gs);
                        let p = py as usize * w + px as usize;
                        let (a, b) = (x[p][c], y[p][c]);
                        mx += wt * a;
                        my += wt * b;
                        sxx += wt * a * a;
                        syy += wt * b * b;
                        sxy += wt * a * b;
                    }
                }
                let (vx, vy, cxy) = (sxx - mx * mx, syy - my * my, sxy - mx * my);
                total += (2.0 * mx * my + c1) * (2.0 * cxy + c2)
                    / ((mx * mx + my * my + c1) * (vx + vy + c2));
            }
        }
        total
    }

    #[test]
    fn masked_pixels_get_no_loss_gradient() {
        let Some(ctx) = visloc_gsplat_render::try_context() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let (w, h) = (37u32, 29u32);
        let n = (w * h) as usize;
        let mut s = 0x0F1E_2D3C_4B5A_6978u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        let render: Vec<[f32; 3]> = (0..n).map(|_| [rnd(), rnd(), rnd()]).collect();
        let gt8: Vec<[u8; 3]> = (0..n)
            .map(|_| {
                [
                    (rnd() * 255.0) as u8,
                    (rnd() * 255.0) as u8,
                    (rnd() * 255.0) as u8,
                ]
            })
            .collect();
        // A masked-out block, like a blacked-out car.
        let valid: Vec<bool> = (0..n)
            .map(|i| !((8..20).contains(&(i % w as usize)) && (6..15).contains(&(i / w as usize))))
            .collect();
        let packed: Vec<u32> = gt8
            .iter()
            .zip(&valid)
            .map(|(p, &v)| {
                p[0] as u32
                    | (p[1] as u32) << 8
                    | (p[2] as u32) << 16
                    | (if v { 255 } else { 0 }) << 24
            })
            .collect();
        let dev = &ctx.device;
        let queue = &ctx.queue;
        let rb = storage(dev, "render", n as u64 * 12);
        queue.write_buffer(&rb, 0, bytemuck::cast_slice(&render));
        let gb = storage(dev, "gt", n as u64 * 4);
        queue.write_buffer(&gb, 0, bytemuck::cast_slice(&packed));
        let db = storage(dev, "d_image", n as u64 * 12);
        let k = SsimKernels::new(dev, w, h);
        let lambda = 0.2;
        k.set_weight(queue, lambda);
        let binds = k.bind(dev, &rb, &gb, &db);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            k.encode(&mut pass, &binds);
        }
        queue.submit(Some(enc.finish()));
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: n as u64 * 12,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&db, 0, &staging, 0, n as u64 * 12);
        queue.submit(Some(enc.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        dev.poll(wgpu::PollType::wait_indefinitely()).ok();
        let gpu: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();

        for (p, &v) in valid.iter().enumerate() {
            if !v {
                assert_eq!(&gpu[p * 3..p * 3 + 3], &[0.0; 3], "masked pixel {p}");
            }
        }
        // Valid pixels, inside and at the edge of the mask's windows:
        // -lambda/(3N) * d(sum of valid-centre SSIM)/dx by central differences.
        let x: Vec<[f64; 3]> = render.iter().map(|p| p.map(|v| v as f64)).collect();
        let y: Vec<[f64; 3]> = gt8.iter().map(|p| p.map(|v| v as f64 / 255.0)).collect();
        let (wu, hu) = (w as usize, h as usize);
        let scale = lambda as f64 / (3.0 * n as f64);
        let mut worst = 0.0f64;
        let mut max_grad = 0.0f64;
        for &(px, py) in &[
            (7usize, 10usize),
            (21, 10),
            (14, 16),
            (3, 3),
            (30, 25),
            (14, 5),
        ] {
            let p = py * wu + px;
            assert!(valid[p]);
            for c in 0..3 {
                let eps = 1e-4;
                let (mut xp, mut xm) = (x.clone(), x.clone());
                xp[p][c] += eps;
                xm[p][c] -= eps;
                let fd = (masked_ssim_sum(&xp, &y, &valid, wu, hu)
                    - masked_ssim_sum(&xm, &y, &valid, wu, hu))
                    / (2.0 * eps);
                let want = -scale * fd;
                max_grad = max_grad.max(want.abs());
                worst = worst.max((want - gpu[p * 3 + c] as f64).abs());
            }
        }
        eprintln!("masked ssim grad worst abs error {worst:.2e} (max {max_grad:.2e})");
        assert!(
            worst < 1e-3 * max_grad.max(1e-12) * 10.0,
            "worst {worst} of {max_grad}"
        );
    }
}
