//! The CUDA rasterizer against the CPU f64 oracle
//! (`visloc_gsplat_core::backward`), with the same scenes and tolerances as
//! the wgpu renderer's tests in `visloc-gsplat-render`. Skipped unless
//! `VISLOC_GSPLAT_CUDA_LIB` points at the library built by
//! `scripts/build_gsplat_cuda.sh`.

use nalgebra::{Quaternion, Vector3};
use visloc_gsplat_core::backward::{render_backward, render_backward_screen, render_f64};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::{sh_rest_coeffs_per_channel, Gaussian, Scene};
use visloc_gsplat_cuda::{CudaGsplat, SCREEN_GRAD_FLOATS};
use visloc_gsplat_render::packing::PackedScene;

fn rasterizer() -> Option<CudaGsplat> {
    match CudaGsplat::from_env() {
        None => {
            eprintln!("skipping: {} is not set", visloc_gsplat_cuda::LIBRARY_ENV);
            None
        }
        Some(r) => Some(r.expect("load the CUDA rasterizer")),
    }
}

/// Deterministic pseudo-random scene: `n` anisotropic, rotated,
/// semi-transparent splats with degree-`degree` SH in front of the camera
/// (the generator of the wgpu renderer's tests). `opacity_shift` moves every
/// opacity logit, so dense scenes can stay unsaturated.
fn random_scene(n: usize, seed: u32, degree: u32, opacity_shift: f32) -> Scene {
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
                opacity_logit: (rnd() - 0.5) * 3.0 + opacity_shift,
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

fn view(w: u32, h: u32) -> CameraView {
    CameraView::new(
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
    )
}

fn d_image(w: u32, h: u32) -> Vec<[f32; 3]> {
    (0..(w * h) as usize)
        .map(|i| {
            let f = i as f32;
            [
                (f * 0.37).sin(),
                (f * 0.91).cos(),
                (f * 0.13).sin() * 0.5 + 0.2,
            ]
        })
        .collect()
}

const BG: [f32; 3] = [0.1, 0.2, 0.3];

fn check_forward(scene: &Scene, w: u32, h: u32) {
    let Some(mut cuda) = rasterizer() else { return };
    let view = view(w, h);
    let cpu = render_f64(scene, &view, BG.map(f64::from));
    cuda.set_scene(&PackedScene::from_scene(scene))
        .expect("upload");
    // Twice: the first frame grows the buffers, the second reuses them.
    for frame in 0..2 {
        let out = cuda.render(&view, BG, scene.sh_degree).expect("render");
        let mut max_err = 0.0f64;
        for (c, g) in cpu.iter().zip(out.rgb.chunks_exact(3)) {
            for (a, b) in c.iter().zip(g) {
                max_err = max_err.max((a - *b as f64).abs());
            }
        }
        eprintln!(
            "frame {frame}: max abs error {max_err:.2e}, {} intersections",
            out.num_intersections
        );
        assert!(
            max_err < 2e-3,
            "frame {frame}: max abs error {max_err} vs CPU"
        );
    }
}

/// Screen-space gradients per gaussian and component, tolerances scaled by
/// each component's largest magnitude (f32 vs f64 and alpha-gate ties at
/// footprint edges add absolute noise).
fn check_screen_grads(scene: &Scene, w: u32, h: u32) {
    let Some(mut cuda) = rasterizer() else { return };
    let view = view(w, h);
    let d = d_image(w, h);
    let d64: Vec<[f64; 3]> = d.iter().map(|p| p.map(f64::from)).collect();
    let cpu = render_backward_screen(scene, &view, BG.map(f64::from), &d64);
    cuda.set_scene(&PackedScene::from_scene(scene))
        .expect("upload");
    cuda.render(&view, BG, scene.sh_degree).expect("render");
    let flat: Vec<f32> = d.iter().flatten().copied().collect();
    let gpu = cuda.backward(&flat).expect("backward").screen;
    let mut worst = 0.0f64;
    for k in 0..9 {
        let scale = cpu.iter().map(|g| g[k].abs()).fold(0.0, f64::max).max(1e-9);
        for (i, c) in cpu.iter().enumerate() {
            let g = gpu[i * SCREEN_GRAD_FLOATS + k] as f64;
            let err = (c[k] - g).abs() / scale;
            worst = worst.max(err);
            assert!(
                err < 2e-3,
                "gaussian {i} component {k}: cpu {:.6e} cuda {g:.6e} (rel to max {err:.2e})",
                c[k]
            );
        }
    }
    eprintln!("screen grads: worst error {worst:.2e} of each component's max");
}

/// Parameter gradients against the oracle. With `active = Some(a)` SH is
/// evaluated only up to degree `a`: the oracle runs on the scene with the
/// higher bands zeroed, and those bands' gradients must come back as 0.
fn check_param_grads(scene: &Scene, w: u32, h: u32, active: Option<u32>) {
    let Some(mut cuda) = rasterizer() else { return };
    let view = view(w, h);
    let d = d_image(w, h);
    let d64: Vec<[f64; 3]> = d.iter().map(|p| p.map(f64::from)).collect();
    let rest_full = sh_rest_coeffs_per_channel(scene.sh_degree);
    let active_degree = active.map_or(scene.sh_degree, |a| a.min(scene.sh_degree));
    let act = sh_rest_coeffs_per_channel(active_degree);
    let mut reference = scene.clone();
    for g in reference.gaussians.iter_mut() {
        for ch in 0..3 {
            for k in act..rest_full {
                g.sh_rest[ch * rest_full + k] = 0.0;
            }
        }
    }
    let mut cpu = render_backward(&reference, &view, BG.map(f64::from), &d64);
    for g in cpu.sh_rest.iter_mut() {
        for ch in 0..3 {
            for k in act..rest_full {
                g[ch * rest_full + k] = 0.0;
            }
        }
    }
    cuda.set_scene(&PackedScene::from_scene(scene))
        .expect("upload");
    cuda.render(&view, BG, active_degree).expect("render");
    let flat: Vec<f32> = d.iter().flatten().copied().collect();
    let gpu = cuda.backward(&flat).expect("backward");

    let n = scene.len();
    let cpc2 = ((scene.sh_degree + 1) * (scene.sh_degree + 1)) as usize;
    let rest_pc = cpc2 - 1;
    let gt = |i: usize, k: usize| gpu.transforms[i * 10 + k] as f64;
    let gsh = &gpu.sh;
    let groups: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
        (
            "mean",
            (0..n).flat_map(|i| cpu.mean[i]).collect(),
            (0..n).flat_map(|i| (0..3).map(move |k| gt(i, k))).collect(),
        ),
        (
            "rotation",
            (0..n).flat_map(|i| cpu.rotation[i]).collect(),
            (0..n).flat_map(|i| (3..7).map(move |k| gt(i, k))).collect(),
        ),
        (
            "scale_log",
            (0..n).flat_map(|i| cpu.scale_log[i]).collect(),
            (0..n)
                .flat_map(|i| (7..10).map(move |k| gt(i, k)))
                .collect(),
        ),
        (
            "opacity_logit",
            cpu.opacity_logit.clone(),
            gpu.opacity.iter().map(|&x| x as f64).collect(),
        ),
        (
            "sh_dc",
            (0..n).flat_map(|i| cpu.sh_dc[i]).collect(),
            (0..n)
                .flat_map(|i| (0..3).map(move |c| gsh[i * 3 * cpc2 + c] as f64))
                .collect(),
        ),
        (
            "sh_rest",
            (0..n).flat_map(|i| cpu.sh_rest[i].clone()).collect(),
            (0..n)
                .flat_map(|i| (0..3 * rest_pc).map(move |k| gsh[i * 3 * cpc2 + 3 + k] as f64))
                .collect(),
        ),
    ];
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
                "{name}[{k}]: cpu {a:.6e} cuda {b:.6e}"
            );
        }
    }
}

#[test]
fn forward_matches_cpu() {
    check_forward(&random_scene(64, 11, 3, 0.0), 96, 80);
}

#[test]
fn forward_matches_cpu_dense_tiles() {
    // Hundreds of faint splats per tile: several forward batches of 256 and
    // backward batches of 128 per tile, without saturating.
    check_forward(&random_scene(2500, 3, 1, -3.0), 64, 48);
}

#[test]
fn screen_grads_match_cpu_small() {
    check_screen_grads(&random_scene(3, 7, 2, 0.0), 40, 32);
}

#[test]
fn screen_grads_match_cpu_many_tiles() {
    check_screen_grads(&random_scene(64, 11, 3, 0.0), 96, 80);
}

#[test]
fn screen_grads_match_cpu_dense_tiles() {
    check_screen_grads(&random_scene(2500, 3, 1, -3.0), 64, 48);
}

#[test]
fn param_grads_match_cpu_small() {
    check_param_grads(&random_scene(3, 7, 2, 0.0), 40, 32, None);
}

#[test]
fn param_grads_match_cpu_many_tiles() {
    check_param_grads(&random_scene(64, 11, 3, 0.0), 96, 80, None);
}

#[test]
fn param_grads_respect_active_sh_degree() {
    let scene = random_scene(24, 5, 3, 0.0);
    check_param_grads(&scene, 64, 48, Some(1));
    check_param_grads(&scene, 64, 48, Some(0));
}

#[test]
fn param_grads_match_cpu_dense_tiles() {
    check_param_grads(&random_scene(2500, 3, 1, -3.0), 64, 48, None);
}

/// A flat, round gaussian facing the camera at depth `z` with opacity `a`.
fn disc(z: f32, a: f32, colour: [f32; 3]) -> Gaussian {
    let sh0 = visloc_gsplat_core::sh::SH_C0;
    Gaussian {
        mean: Vector3::new(0.0, 0.0, z),
        scale_log: Vector3::new(0.5f32.ln(), 0.5f32.ln(), 0.01f32.ln()),
        rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
        opacity_logit: (a / (1.0 - a)).ln(),
        sh_dc: colour.map(|c| (c - 0.5) / sh0),
        sh_rest: Vec::new(),
        sh_degree: 0,
    }
}

#[test]
fn median_depth_is_where_transmittance_halves() {
    let Some(mut cuda) = rasterizer() else { return };
    let (w, h) = (64u32, 48u32);
    let view = CameraView::new(
        nalgebra::Matrix3::identity(),
        Vector3::zeros(),
        // The discs project onto the centre of pixel (31, 23).
        PinholeCamera::new(w, h, 60.0, 60.0, 31.5, 23.5),
    );
    let centre = (23 * w + 31) as usize;
    // Only a faint front disc: T = 0.6 > 0.5 at the centre, so no median.
    let front = Scene::new(vec![disc(3.0, 0.4, [0.9, 0.1, 0.1])], 0);
    cuda.set_scene(&PackedScene::from_scene(&front))
        .expect("upload");
    let f = cuda.render_depth(&view, BG, 0, false).expect("render");
    assert_eq!(f.depth[centre], 0.0);
    assert!(
        (f.opacity[centre] - 0.4).abs() < 1e-3,
        "{}",
        f.opacity[centre]
    );
    // An opaque disc behind it takes T below 0.5: the median is its depth.
    let both = Scene::new(
        vec![
            disc(3.0, 0.4, [0.9, 0.1, 0.1]),
            disc(5.0, 0.9, [0.1, 0.9, 0.1]),
        ],
        0,
    );
    cuda.set_scene(&PackedScene::from_scene(&both))
        .expect("upload");
    let f = cuda.render_depth(&view, BG, 0, false).expect("render");
    assert!((f.depth[centre] - 5.0).abs() < 1e-4, "{}", f.depth[centre]);
    assert!((f.opacity[centre] - (1.0 - 0.6 * 0.1)).abs() < 1e-3);
    // Corners see neither disc.
    assert_eq!(f.depth[0], 0.0);
    // The colour is the plain render's.
    let plain = cuda.render(&view, BG, 0).expect("render");
    assert_eq!(f.rgb, plain.rgb);
}

#[test]
fn planar_depth_meets_the_splat_plane() {
    let Some(mut cuda) = rasterizer() else { return };
    let (w, h) = (64u32, 48u32);
    let view = CameraView::new(
        nalgebra::Matrix3::identity(),
        Vector3::zeros(),
        PinholeCamera::new(w, h, 60.0, 60.0, 31.5, 23.5),
    );
    // An opaque disc at depth 5, turned 30 degrees about y.
    let half = 15f32.to_radians();
    let mut tilted = disc(5.0, 0.9, [0.5, 0.5, 0.5]);
    tilted.rotation = Quaternion::new(half.cos(), 0.0, half.sin(), 0.0);
    cuda.set_scene(&PackedScene::from_scene(&Scene::new(vec![tilted], 0)))
        .expect("upload");
    let centre_depth = cuda.render_depth(&view, BG, 0, false).expect("render");
    let planar = cuda.render_depth(&view, BG, 0, true).expect("render");
    // Pixel (36, 23): its ray is (5 / 60, 0, 1). The disc's plane has normal
    // (sin 30, 0, cos 30) through (0, 0, 5).
    let pix = (23 * w + 36) as usize;
    let x = 5.0f32 / 60.0;
    let (s, c) = (30f32.to_radians().sin(), 30f32.to_radians().cos());
    let want = 5.0 * c / (x * s + c);
    assert!((centre_depth.depth[pix] - 5.0).abs() < 1e-4);
    assert!(
        (planar.depth[pix] - want).abs() < 1e-3,
        "planar {} want {want}",
        planar.depth[pix]
    );
    // At the centre pixel both agree.
    let centre = (23 * w + 31) as usize;
    assert!((planar.depth[centre] - 5.0).abs() < 1e-4);
    assert_eq!(planar.rgb, centre_depth.rgb);
}
