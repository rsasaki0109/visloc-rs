//! The CUDA training step: its loss gradient against the CPU L1 and SSIM
//! (`visloc_gsplat_train::metrics::ssim_with_grad`), the refine scatter, and
//! a short fit. Skipped unless `VISLOC_GSPLAT_CUDA_LIB` points at the
//! library built by `scripts/build_gsplat_cuda.sh`.

use nalgebra::{Quaternion, Rotation3, Vector3};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::{sh_rest_coeffs_per_channel, Gaussian, Scene};
use visloc_gsplat_cuda::{CudaGsplat, GscCamera, StepParams};
use visloc_gsplat_render::packing::PackedScene;
use visloc_gsplat_train::config::pack_rgba8;
use visloc_gsplat_train::metrics::ssim_with_grad;

fn gsplat() -> Option<CudaGsplat> {
    match CudaGsplat::from_env() {
        None => {
            eprintln!("skipping: {} is not set", visloc_gsplat_cuda::LIBRARY_ENV);
            None
        }
        Some(r) => Some(r.expect("load the CUDA library")),
    }
}

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
                sh_rest: (0..3 * sh_rest_coeffs_per_channel(degree))
                    .map(|_| (rnd() - 0.5) * 0.2)
                    .collect(),
                sh_degree: degree,
            }
        })
        .collect();
    Scene::new(gaussians, degree)
}

fn view(w: u32, h: u32, yaw: f32) -> CameraView {
    CameraView::new(
        Rotation3::from_euler_angles(0.05, yaw, 0.03).into_inner(),
        Vector3::new(0.1, -0.05, 0.2),
        PinholeCamera::new(
            w,
            h,
            w as f32 * 1.1,
            w as f32 * 1.1,
            w as f32 * 0.5,
            h as f32 * 0.5,
        ),
    )
}

/// A smooth target image in [0, 1], with a masked-out block when `mask`.
fn target(w: u32, h: u32, mask: bool) -> (Vec<[f32; 3]>, Vec<bool>) {
    let mut rgb = Vec::new();
    let mut valid = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            rgb.push([
                0.5 + 0.4 * (fx * 7.0).sin(),
                0.5 + 0.4 * (fy * 5.0 + fx).cos(),
                0.3 + 0.5 * fx * fy,
            ]);
            valid.push(!(mask && x > w / 2 && y < h / 3));
        }
    }
    (rgb, valid)
}

#[test]
fn loss_gradient_matches_cpu_l1_and_ssim() {
    let Some(mut gs) = gsplat() else { return };
    let (w, h) = (70u32, 52u32);
    let scene = random_scene(40, 9, 1);
    let v = view(w, h, -0.08);
    for (mask, lambda) in [(false, 0.2f32), (true, 0.2), (false, 0.0)] {
        let (gt, valid) = target(w, h, mask);
        let packed = pack_rgba8(&gt, mask.then_some(valid.as_slice()));
        // The ground truth as the device sees it (8-bit).
        let gt8: Vec<[f64; 3]> = packed
            .iter()
            .map(|p| [0, 8, 16].map(|s| ((p >> s) & 0xFF) as f64 / 255.0))
            .collect();
        gs.set_scene(&PackedScene::from_scene(&scene))
            .expect("upload");
        gs.set_images(w, h, &packed).expect("images");
        let frame = gs.render(&v, [0.0; 3], 1).expect("render");
        let d = gs.loss_grad(0, lambda).expect("loss");
        gs.take_loss().expect("reset");

        let r: Vec<[f64; 3]> = frame
            .rgb
            .chunks_exact(3)
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
        let npix = (w * h) as f64;
        // L1 part: (1 - lambda) sign(r - g) / (3 npix) on valid pixels.
        let mut cpu: Vec<[f64; 3]> = r
            .iter()
            .zip(&gt8)
            .enumerate()
            .map(|(i, (rp, gp))| {
                let ok = !mask || valid[i];
                [0, 1, 2].map(|c| {
                    let diff = rp[c] - gp[c];
                    if ok && diff != 0.0 {
                        (1.0 - lambda as f64) * diff.signum() / (3.0 * npix)
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        if lambda > 0.0 && !mask {
            // d(lambda (1 - SSIM)) = -lambda dSSIM (the CPU SSIM has no mask).
            let (_, g) = ssim_with_grad(&r, &gt8, w as usize, h as usize, true);
            for (c, s) in cpu.iter_mut().zip(g.expect("gradient")) {
                for k in 0..3 {
                    c[k] -= lambda as f64 * s[k];
                }
            }
        }
        let scale = cpu
            .iter()
            .flatten()
            .map(|x| x.abs())
            .fold(0.0, f64::max)
            .max(1e-12);
        let mut worst = 0.0f64;
        for (i, c) in cpu.iter().enumerate() {
            if mask && lambda > 0.0 {
                // Masked SSIM has no CPU reference: check only that masked
                // pixels get no gradient.
                if !valid[i] {
                    assert!(d[i * 3..i * 3 + 3].iter().all(|x| *x == 0.0));
                }
                continue;
            }
            for k in 0..3 {
                worst = worst.max((c[k] - d[i * 3 + k] as f64).abs() / scale);
            }
        }
        eprintln!("mask {mask} lambda {lambda}: worst error {worst:.2e} of max {scale:.3e}");
        assert!(worst < 1e-3, "loss gradient off by {worst:.2e} of its max");
    }
}

#[test]
fn refine_actions_scatter_rows() {
    let Some(mut gs) = gsplat() else { return };
    let scene = random_scene(5, 3, 2);
    let packed = PackedScene::from_scene(&scene);
    gs.set_scene(&packed).expect("upload");
    // prune, keep, brush split, clone, keep
    let n_after = gs.apply_actions(&[0, 1, 4, 2, 1], 7).expect("apply");
    assert_eq!(n_after, 6);
    let out = gs.scene().expect("download");
    assert_eq!(out.num_gaussians, 6);
    let row = 3 * packed.sh_coeffs_per_channel;
    let t = |s: &PackedScene, i: usize| s.transforms[i * 10..i * 10 + 10].to_vec();
    let sh = |s: &PackedScene, i: usize| s.sh[i * row..(i + 1) * row].to_vec();
    // Output rows: [1], [2a, 2b], [3, 3 copy], [4].
    assert_eq!(t(&out, 0), t(&packed, 1));
    assert_eq!(out.opacity[0], packed.opacity[1]);
    assert_eq!(t(&out, 3), t(&packed, 3));
    assert_eq!(t(&out, 4), t(&packed, 3));
    assert_eq!(sh(&out, 4), sh(&packed, 3));
    assert_eq!(t(&out, 5), t(&packed, 4));
    // brush split: means symmetric about the parent, quaternion kept,
    // scales / sqrt 2, opacity 1 - sqrt(1 - a), SH copied.
    let (p, a, b) = (t(&packed, 2), t(&out, 1), t(&out, 2));
    for k in 0..3 {
        assert!((a[k] + b[k] - 2.0 * p[k]).abs() < 1e-5, "mean {k}");
    }
    assert!(
        (0..3).any(|k| (a[k] - p[k]).abs() > 1e-6),
        "split moved the mean"
    );
    assert_eq!(&a[3..7], &p[3..7]);
    for k in 7..10 {
        assert!((a[k] - (p[k] - 2f32.sqrt().ln())).abs() < 1e-6);
        assert_eq!(a[k], b[k]);
    }
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let want = 1.0 - (1.0 - sig(packed.opacity[2])).sqrt();
    assert!((sig(out.opacity[1]) - want).abs() < 1e-5);
    assert_eq!(out.opacity[1], out.opacity[2]);
    assert_eq!(sh(&out, 1), sh(&packed, 2));
    assert_eq!(sh(&out, 2), sh(&packed, 2));
    // Refine statistics restart at zero for the new population.
    let stats = gs.refine_stats().expect("stats");
    assert_eq!(stats.seen, vec![0.0; 6]);
}

#[test]
fn training_steps_fit_a_rendered_target() {
    let Some(mut gs) = gsplat() else { return };
    let (w, h) = (96u32, 80u32);
    let truth = random_scene(60, 21, 1);
    let views: Vec<CameraView> = (0..4)
        .map(|k| view(w, h, -0.12 + 0.08 * k as f32))
        .collect();
    // Targets rendered from the true scene.
    gs.set_scene(&PackedScene::from_scene(&truth))
        .expect("upload");
    let mut images = Vec::new();
    for v in &views {
        let f = gs.render(v, [0.0; 3], 1).expect("render");
        let rgb: Vec<[f32; 3]> = f.rgb.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect();
        images.extend(pack_rgba8(&rgb, None));
    }
    gs.set_images(w, h, &images).expect("images");
    // Start from the truth with perturbed colours and opacities.
    let mut init = truth.clone();
    for (i, g) in init.gaussians.iter_mut().enumerate() {
        let f = i as f32;
        g.sh_dc = [(f * 0.7).sin(), (f * 1.3).cos(), (f * 0.4).sin()];
        g.opacity_logit = 0.0;
    }
    gs.set_scene(&PackedScene::from_scene(&init))
        .expect("upload");
    let mut mean_loss = Vec::new();
    for step in 0..300u32 {
        let t = (step + 1) as f32;
        let p = StepParams {
            camera: GscCamera::from_view(&views[step as usize % views.len()]),
            image: step % views.len() as u32,
            sh_active_degree: 1,
            background: [0.0; 3],
            ssim_weight: 0.2,
            lr: [0.0, 0.0, 0.0, 5e-2, 2.5e-2, 2.5e-3],
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-15,
            bc1: 1.0 - 0.9f32.powf(t),
            bc2: 1.0 - 0.999f32.powf(t),
            scale_aux_coef: 0.0,
            opacity_aux_coef: 0.0,
            accumulate_stats: 1,
            noise_seed: step,
            noise_scale: 0.0,
            noise_max: 0.0,
        };
        gs.train_step(&p).expect("step");
        if (step + 1) % 50 == 0 {
            let s = gs.take_loss().expect("loss");
            mean_loss.push(s.l1 / s.steps as f64);
        }
    }
    eprintln!("mean L1 every 50 steps: {mean_loss:.4?}");
    assert!(
        mean_loss.last().unwrap() < &(0.25 * mean_loss[0]),
        "L1 fell from {:.4} to only {:.4}",
        mean_loss[0],
        mean_loss.last().unwrap()
    );
    let stats = gs.refine_stats().expect("stats");
    assert!(stats.seen.iter().any(|&s| s > 0.0), "no refine statistics");
}
