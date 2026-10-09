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
            pass.dispatch_workgroups(tiles, 1, 1);
        }
    }
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
