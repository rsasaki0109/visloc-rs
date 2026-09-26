//! GPU (wgpu) forward renderer for 3D Gaussian Splatting — stage 1 of the
//! visloc-rs 3DGS effort.
//!
//! This crate is the GPU counterpart of `visloc-gsplat-core`'s CPU reference
//! rasterizer. It consumes the same [`visloc_gsplat_core::gaussian::Scene`] and
//! [`visloc_gsplat_core::camera::CameraView`] and produces the same linear-RGB
//! [`visloc_gsplat_core::cpu_render::Image`], so the two can be compared
//! directly (see the `gpu_matches_cpu_reference` test).
//!
//! Scope (stage 1): the whole forward pass runs on the GPU
//! (`project_forward` → `project_visible` → `map_gaussians` → `tile_offsets` →
//! `rasterize`). The two per-frame sorts currently run on the host; a
//! subgroup-free device-side radix sort is a planned follow-up. All shaders use
//! only baseline WebGPU features (no subgroups, only `u32` atomics).
//!
//! # Feature flags
//!
//! The GPU path is behind the **`gpu`** feature (off by default). wgpu 30 pulls
//! `naga`, whose `indexmap` dependency needs edition2024 (Rust >= 1.85), so
//! enabling `gpu` requires a current toolchain. The pure host-side packing and
//! camera math always compile, so the default build keeps the workspace MSRV.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "gpu")] {
//! use visloc_gsplat_core::splat;
//! use visloc_gsplat_render::{GpuContext, Renderer};
//!
//! let scene = splat::load_splat("scene.splat")?;
//! let ctx = GpuContext::new()?;
//! let mut renderer = Renderer::new(ctx, &scene, 640, 480)?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod packing;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod renderer;
#[cfg(feature = "gpu")]
pub mod scan;
#[cfg(feature = "gpu")]
pub mod shaders;
#[cfg(feature = "gpu")]
pub mod sort;
#[cfg(feature = "gpu")]
pub mod uniforms;

pub use packing::{PackedScene, TRANSFORM_FLOATS};

#[cfg(feature = "gpu")]
pub use gpu::{try_context, GpuContext, GpuError};
#[cfg(feature = "gpu")]
pub use renderer::{DeviceParams, GpuScene, ParamGrads, Renderer, SCREEN_GRAD_FLOATS};
#[cfg(feature = "gpu")]
pub use scan::PrefixScanner;
#[cfg(feature = "gpu")]
pub use sort::{RadixParams, RadixSorter, SortBuffers};
#[cfg(feature = "gpu")]
pub use uniforms::{tile_bounds, ProjectUniforms, RasterUniforms, TILE_SIZE, TILE_WIDTH};

#[cfg(all(test, not(feature = "gpu")))]
mod tests {
    // The host-side packing is pure and testable without a GPU or wgpu.
    use super::*;
    use nalgebra::{Quaternion, Vector3};
    use visloc_gsplat_core::gaussian::{Gaussian, Scene};

    fn solid_gaussian(mean: Vector3<f32>, scale: f32, color: [f32; 3]) -> Gaussian {
        Gaussian {
            mean,
            scale_log: Vector3::new(scale.ln(), scale.ln(), scale.ln()),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 10.0,
            sh_dc: [
                (color[0] - 0.5) / visloc_gsplat_core::sh::SH_C0,
                (color[1] - 0.5) / visloc_gsplat_core::sh::SH_C0,
                (color[2] - 0.5) / visloc_gsplat_core::sh::SH_C0,
            ],
            sh_rest: Vec::new(),
            sh_degree: 0,
        }
    }

    #[test]
    fn packing_matches_cpu_conventions() {
        let scene = Scene::new(
            vec![solid_gaussian(
                Vector3::new(0.0, 0.0, 5.0),
                0.2,
                [1.0, 0.0, 0.0],
            )],
            0,
        );
        let p = PackedScene::from_scene(&scene);
        assert_eq!(p.num_gaussians, 1);
        assert_eq!(p.sh_coeffs_per_channel, 1);
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use nalgebra::{Matrix3, Quaternion, Vector3};
    use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
    use visloc_gsplat_core::cpu_render;
    use visloc_gsplat_core::gaussian::{Gaussian, Scene};

    use crate::{try_context, RadixSorter, Renderer};

    /// Blocking read of `count` `u32`s from `buffer`.
    fn read_back_u32(ctx: &crate::GpuContext, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
        ctx.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let _ = rx.recv();
        let data = slice.get_mapped_range().expect("map");
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    fn front_camera() -> CameraView {
        CameraView::new(
            Matrix3::identity(),
            Vector3::zeros(),
            PinholeCamera::new(64, 64, 50.0, 50.0, 32.0, 32.0),
        )
    }

    fn solid_gaussian(mean: Vector3<f32>, scale: f32, color: [f32; 3]) -> Gaussian {
        Gaussian {
            mean,
            scale_log: Vector3::new(scale.ln(), scale.ln(), scale.ln()),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 10.0,
            sh_dc: [
                (color[0] - 0.5) / visloc_gsplat_core::sh::SH_C0,
                (color[1] - 0.5) / visloc_gsplat_core::sh::SH_C0,
                (color[2] - 0.5) / visloc_gsplat_core::sh::SH_C0,
            ],
            sh_rest: Vec::new(),
            sh_degree: 0,
        }
    }

    #[test]
    fn context_only_creates() {
        let Some(ctx) = try_context() else {
            return;
        };
        assert!(!ctx.adapter_info.name.is_empty());
    }

    #[test]
    fn gpu_single_gaussian_matches_cpu() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping gpu_single_gaussian_matches_cpu: no GPU adapter");
            return;
        };
        let scene = Scene::new(
            vec![solid_gaussian(
                Vector3::new(0.0, 0.0, 5.0),
                0.3,
                [0.9, 0.1, 0.1],
            )],
            0,
        );
        let view = front_camera();
        let bg = [0.0, 0.0, 0.0];
        let cpu = cpu_render::render(&scene, &view, bg);
        let mut renderer = Renderer::new(ctx, &scene, 64, 64).expect("renderer");
        let gpu = renderer.render(&view, bg);
        let mut max_err = 0.0f32;
        for (a, b) in cpu.rgb.iter().zip(gpu.rgb.iter()) {
            for c in 0..3 {
                max_err = max_err.max((a[c] - b[c]).abs());
            }
        }
        assert!(max_err < 0.02, "single gaussian max abs error {max_err}");
    }

    #[test]
    fn gpu_radix_sort_is_correct_and_stable() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping gpu_radix_sort_is_correct_and_stable: no GPU adapter");
            return;
        };
        // Keys with many duplicates so stability is observable: values are the
        // original indices and must be increasing within a key. Cover the full
        // 32-bit sort plus reduced-bit sorts with an even (2) and odd (3) pass
        // count, since an odd count leaves the result in `pairs[1]`.
        //
        // The 1.2M case has 293 blocks (16 * 293 = 4688 digit-major entries),
        // so `radix_scan` spans more than one of its 4096-entry chunks.
        let max_n = 1_200_000usize;
        let sorter = RadixSorter::new(&ctx.device, max_n);
        let pairs = sorter.allocate(&ctx.device, "test");
        for (n, modulus, key_bits) in [
            (5000usize, 17u32, 32u32),
            (5000, 17, 5),
            (5000, 4000, 12),
            (max_n, 1 << 20, 20),
        ] {
            let mut state = 0x1234_5678u32;
            let mut keys = Vec::with_capacity(n);
            for _ in 0..n {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                keys.push(state % modulus);
            }
            let values: Vec<u32> = (0..n as u32).collect();

            ctx.queue
                .write_buffer(&pairs[0].keys, 0, bytemuck::cast_slice(&keys));
            ctx.queue
                .write_buffer(&pairs[0].values, 0, bytemuck::cast_slice(&values));
            sorter.prepare(&ctx.queue, n, key_bits);
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sort"),
                });
            let out = sorter.encode(&ctx.device, &mut encoder, &pairs, n, key_bits);
            ctx.queue.submit(Some(encoder.finish()));

            let gpu_keys: Vec<u32> = read_back_u32(&ctx, &pairs[out].keys, n);
            let gpu_values: Vec<u32> = read_back_u32(&ctx, &pairs[out].values, n);

            // Reference: stable host sort.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by_key(|&i| keys[i]);
            let expect_keys: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
            let expect_values: Vec<u32> = order.iter().map(|&i| values[i]).collect();

            assert_eq!(
                gpu_keys, expect_keys,
                "{key_bits}-bit: keys differ from host sort"
            );
            assert_eq!(
                gpu_values, expect_values,
                "{key_bits}-bit: sort is not stable"
            );
        }
    }

    #[test]
    fn gpu_prefix_scan_matches_host_across_many_blocks() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping gpu_prefix_scan_matches_host_across_many_blocks: no GPU adapter");
            return;
        };
        // Many 2048-element blocks, so the block-sum carry path is exercised
        // (a single-block scan never touches it).
        let n = 300_000usize;
        let mut state = 0x9e37_79b9u32;
        let input: Vec<u32> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state % 50
            })
            .collect();
        let mk = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let (inb, outb) = (mk("scan_in"), mk("scan_out"));
        ctx.queue
            .write_buffer(&inb, 0, bytemuck::cast_slice(&input));
        let scanner = crate::scan::PrefixScanner::new(&ctx.device, n);
        scanner.scan(&ctx.device, &ctx.queue, &inb, &outb, n);
        let gpu = read_back_u32(&ctx, &outb, n);
        let mut acc = 0u32;
        let expect: Vec<u32> = input
            .iter()
            .map(|&v| {
                acc += v;
                acc
            })
            .collect();
        let first_bad = gpu.iter().zip(&expect).position(|(a, b)| a != b);
        assert_eq!(first_bad, None, "inclusive scan differs from host");
    }

    #[test]
    fn gpu_grows_isect_buffers_past_initial_capacity() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping gpu_grows_isect_buffers_past_initial_capacity: no GPU adapter");
            return;
        };
        // 40 faint gaussians covering all 16 tiles of a 64x64 frame: 640
        // intersections against an initial capacity of 100, so the renderer
        // must grow its isect buffers (and re-point the map/offsets/raster
        // bind groups) mid-frame. Before on-demand growth the excess was
        // truncated and the far gaussians silently dropped.
        let count = 40;
        let gaussians: Vec<Gaussian> = (0..count)
            .map(|i| {
                let t = i as f32 / count as f32;
                let mut g = solid_gaussian(
                    Vector3::new(0.0, 0.0, 4.0 + 4.0 * t),
                    30.0,
                    [t, 1.0 - t, 0.5],
                );
                g.opacity_logit = -3.0; // ~5% each, so every layer shows
                g
            })
            .collect();
        let scene = Scene::new(gaussians, 0);
        let view = front_camera();
        let bg = [0.1, 0.2, 0.3];
        let cpu = cpu_render::render(&scene, &view, bg);
        let mut renderer = Renderer::with_initial_isect_capacity(ctx, &scene, 64, 64, Some(100))
            .expect("renderer");
        // Twice: the first frame grows, the second runs on the grown buffers.
        for frame in 0..2 {
            let gpu = renderer.render(&view, bg);
            let mut max_err = 0.0f32;
            for (a, b) in cpu.rgb.iter().zip(gpu.rgb.iter()) {
                for c in 0..3 {
                    max_err = max_err.max((a[c] - b[c]).abs());
                }
            }
            assert!(
                max_err < 0.02,
                "frame {frame}: max abs error {max_err} vs CPU"
            );
        }
    }

    /// Deterministic pseudo-random scene: `n` anisotropic, rotated,
    /// semi-transparent splats with degree-`degree` SH in front of the camera.
    fn random_scene(n: usize, seed: u32, degree: u32) -> Scene {
        let mut st = seed;
        let mut rnd = move || {
            st = st.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (st >> 8) as f32 / (1u32 << 24) as f32
        };
        let gaussians = (0..n)
            .map(|_| {
                let z = 3.0 + 4.0 * rnd();
                Gaussian {
                    mean: Vector3::new((rnd() - 0.5) * z * 0.9, (rnd() - 0.5) * z * 0.7, z),
                    scale_log: Vector3::new(
                        (0.05 + 0.3 * rnd()).ln(),
                        (0.05 + 0.3 * rnd()).ln(),
                        (0.05 + 0.3 * rnd()).ln(),
                    ),
                    rotation: Quaternion::new(0.5 + rnd(), rnd() - 0.5, rnd() - 0.5, rnd() - 0.5),
                    opacity_logit: (rnd() - 0.5) * 3.0,
                    sh_dc: [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0],
                    sh_rest: (0..3 * visloc_gsplat_core::gaussian::sh_rest_coeffs_per_channel(
                        degree,
                    ))
                        .map(|_| (rnd() - 0.5) * 0.2)
                        .collect(),
                    sh_degree: degree,
                }
            })
            .collect();
        Scene::new(gaussians, degree)
    }

    /// GPU screen-space gradients (rasterize_backward + grad_reduce) against
    /// the f64 CPU reference, per gaussian and component.
    fn check_screen_grads(scene: &Scene, w: u32, h: u32) {
        let Some(ctx) = try_context() else {
            eprintln!("skipping screen-grad check: no GPU adapter");
            return;
        };
        if !ctx.features.contains(wgpu::Features::SUBGROUP) {
            eprintln!("skipping screen-grad check: no SUBGROUP support");
            return;
        }
        let view = CameraView::new(
            nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner(),
            Vector3::new(0.1, -0.05, 0.2),
            PinholeCamera::new(
                w,
                h,
                w as f32 * 1.1,
                w as f32 * 1.1,
                w as f32 * 0.5,
                h as f32 * 0.5,
            ),
        );
        let bg = [0.1f32, 0.2, 0.3];
        let d_image: Vec<[f32; 3]> = (0..(w * h) as usize)
            .map(|i| {
                let f = i as f32;
                [
                    (f * 0.37).sin(),
                    (f * 0.91).cos(),
                    (f * 0.13).sin() * 0.5 + 0.2,
                ]
            })
            .collect();
        let d64: Vec<[f64; 3]> = d_image.iter().map(|p| p.map(|x| x as f64)).collect();
        let cpu = visloc_gsplat_core::backward::render_backward_screen(
            scene,
            &view,
            bg.map(|x| x as f64),
            &d64,
        );
        let mut renderer = Renderer::new(ctx, scene, w, h).expect("renderer");
        let _ = renderer.render(&view, bg);
        let gpu = renderer.backward_screen(&d_image).expect("backward");

        // Scale tolerances per component by its largest magnitude, since
        // f32 vs f64 and alpha-gate ties at footprint edges add absolute noise.
        let mut worst = 0.0f64;
        for k in 0..9 {
            let scale = cpu.iter().map(|g| g[k].abs()).fold(0.0, f64::max).max(1e-9);
            for (i, (c, g)) in cpu.iter().zip(&gpu).enumerate() {
                let err = (c[k] - g[k] as f64).abs() / scale;
                worst = worst.max(err);
                assert!(
                    err < 2e-3,
                    "gaussian {i} component {k}: cpu {:.6e} gpu {:.6e} (rel to max {err:.2e})",
                    c[k],
                    g[k]
                );
            }
        }
        eprintln!("screen grads: worst error {worst:.2e} of each component's max");
    }

    /// GPU parameter gradients (full backward) against the f64 CPU reference.
    fn check_param_grads(scene: &Scene, w: u32, h: u32) {
        check_param_grads_active(scene, w, h, None);
    }

    /// With `active = Some(a)` the GPU evaluates SH only up to degree `a`:
    /// compare against the CPU oracle on the scene with the higher bands
    /// zeroed, whose gradients for those bands must come back as 0.
    fn check_param_grads_active(scene: &Scene, w: u32, h: u32, active: Option<u32>) {
        let Some(ctx) = try_context() else {
            eprintln!("skipping param-grad check: no GPU adapter");
            return;
        };
        if !ctx.features.contains(wgpu::Features::SUBGROUP) {
            eprintln!("skipping param-grad check: no SUBGROUP support");
            return;
        }
        let view = CameraView::new(
            nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner(),
            Vector3::new(0.1, -0.05, 0.2),
            PinholeCamera::new(
                w,
                h,
                w as f32 * 1.1,
                w as f32 * 1.1,
                w as f32 * 0.5,
                h as f32 * 0.5,
            ),
        );
        let bg = [0.1f32, 0.2, 0.3];
        let d_image: Vec<[f32; 3]> = (0..(w * h) as usize)
            .map(|i| {
                let f = i as f32;
                [
                    (f * 0.37).sin(),
                    (f * 0.91).cos(),
                    (f * 0.13).sin() * 0.5 + 0.2,
                ]
            })
            .collect();
        let d64: Vec<[f64; 3]> = d_image.iter().map(|p| p.map(|x| x as f64)).collect();
        let rest_full = visloc_gsplat_core::gaussian::sh_rest_coeffs_per_channel(scene.sh_degree);
        let act = active.map_or(rest_full, |a| {
            visloc_gsplat_core::gaussian::sh_rest_coeffs_per_channel(a.min(scene.sh_degree))
        });
        let mut reference = scene.clone();
        for g in reference.gaussians.iter_mut() {
            for ch in 0..3 {
                for k in act..rest_full {
                    g.sh_rest[ch * rest_full + k] = 0.0;
                }
            }
        }
        let mut cpu = visloc_gsplat_core::backward::render_backward(
            &reference,
            &view,
            bg.map(|x| x as f64),
            &d64,
        );
        for g in cpu.sh_rest.iter_mut() {
            for ch in 0..3 {
                for k in act..rest_full {
                    g[ch * rest_full + k] = 0.0;
                }
            }
        }
        let mut renderer = Renderer::new(ctx, scene, w, h).expect("renderer");
        renderer.set_active_sh_degree(active);
        let _ = renderer.render(&view, bg);
        let gpu = renderer.backward(&d_image).expect("backward");

        let n = scene.len();
        let cpc2 = ((scene.sh_degree + 1) * (scene.sh_degree + 1)) as usize;
        let rest_pc = cpc2 - 1;
        // (name, cpu values, gpu values) per parameter group, gaussian-major.
        let mut groups: Vec<(&str, Vec<f64>, Vec<f64>)> = Vec::new();
        let gsh = &gpu.sh;
        let gt = |i: usize, k: usize| gpu.transforms[i * 10 + k] as f64;
        groups.push((
            "mean",
            (0..n).flat_map(|i| cpu.mean[i]).collect(),
            (0..n).flat_map(|i| (0..3).map(move |k| gt(i, k))).collect(),
        ));
        groups.push((
            "rotation",
            (0..n).flat_map(|i| cpu.rotation[i]).collect(),
            (0..n).flat_map(|i| (3..7).map(move |k| gt(i, k))).collect(),
        ));
        groups.push((
            "scale_log",
            (0..n).flat_map(|i| cpu.scale_log[i]).collect(),
            (0..n)
                .flat_map(|i| (7..10).map(move |k| gt(i, k)))
                .collect(),
        ));
        groups.push((
            "opacity_logit",
            cpu.opacity_logit.clone(),
            gpu.opacity.iter().map(|&x| x as f64).collect(),
        ));
        groups.push((
            "sh_dc",
            (0..n).flat_map(|i| cpu.sh_dc[i]).collect(),
            (0..n)
                .flat_map(|i| (0..3).map(move |c| gsh[i * 3 * cpc2 + c] as f64))
                .collect(),
        ));
        groups.push((
            "sh_rest",
            (0..n).flat_map(|i| cpu.sh_rest[i].clone()).collect(),
            (0..n)
                .flat_map(|i| {
                    let gs = gsh;
                    (0..3 * rest_pc).map(move |k| gs[i * 3 * cpc2 + 3 + k] as f64)
                })
                .collect(),
        ));
        for (name, c, g) in &groups {
            assert_eq!(c.len(), g.len(), "{name}: length");
            let scale = c.iter().map(|x| x.abs()).fold(0.0, f64::max).max(1e-9);
            let worst = c
                .iter()
                .zip(g)
                .map(|(a, b)| (a - b).abs() / scale)
                .fold(0.0, f64::max);
            eprintln!("{name:<14} worst error {worst:.2e} of max {scale:.3e}");
            for (k, (a, b)) in c.iter().zip(g).enumerate() {
                assert!(
                    (a - b).abs() / scale < 3e-3,
                    "{name}[{k}]: cpu {a:.6e} gpu {b:.6e}"
                );
            }
        }
    }

    #[test]
    fn gpu_param_grads_match_cpu_small() {
        let scene = random_scene(3, 7, 2);
        check_param_grads(&scene, 40, 32);
    }

    #[test]
    fn gpu_param_grads_match_cpu_many_tiles() {
        let scene = random_scene(64, 11, 3);
        check_param_grads(&scene, 96, 80);
    }

    #[test]
    fn gpu_param_grads_respect_active_sh_degree() {
        let scene = random_scene(24, 5, 3);
        check_param_grads_active(&scene, 64, 48, Some(1));
        check_param_grads_active(&scene, 64, 48, Some(0));
    }

    #[test]
    fn gpu_screen_grads_match_cpu_small() {
        let scene = random_scene(3, 7, 2);
        check_screen_grads(&scene, 40, 32);
    }

    #[test]
    fn gpu_screen_grads_match_cpu_many_tiles() {
        let scene = random_scene(64, 11, 3);
        check_screen_grads(&scene, 96, 80);
    }

    #[test]
    fn gpu_matches_cpu_reference() {
        let Some(ctx) = try_context() else {
            eprintln!("skipping gpu_matches_cpu_reference: no GPU adapter");
            return;
        };
        let gaussians = vec![
            solid_gaussian(Vector3::new(0.0, 0.0, 5.0), 0.3, [0.9, 0.1, 0.1]),
            solid_gaussian(Vector3::new(0.5, -0.3, 5.5), 0.4, [0.1, 0.8, 0.2]),
            solid_gaussian(Vector3::new(-0.4, 0.4, 6.0), 0.5, [0.2, 0.3, 0.9]),
        ];
        let scene = Scene::new(gaussians, 0);
        let view = front_camera();
        let bg = [0.05, 0.06, 0.07];

        let cpu = cpu_render::render(&scene, &view, bg);
        let mut renderer = Renderer::new(ctx, &scene, 64, 64).expect("renderer");
        let gpu = renderer.render(&view, bg);

        let mut max_err = 0.0f32;
        let mut sum_err = 0.0f64;
        for (a, b) in cpu.rgb.iter().zip(gpu.rgb.iter()) {
            for c in 0..3 {
                let e = (a[c] - b[c]).abs();
                max_err = max_err.max(e);
                sum_err += e as f64;
            }
        }
        let mean_err = sum_err / (cpu.rgb.len() * 3) as f64;
        assert!(mean_err < 0.002, "mean abs error {mean_err} too high");
        assert!(max_err < 0.05, "max abs error {max_err} too high");
    }
}
