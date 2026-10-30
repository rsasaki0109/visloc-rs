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

/// The expected-depth channel's gradients (the depth-prior loss's path).
/// Finite differences are useless here: a pixel crossing the alpha gate
/// moves D by z / 255, centimetres at a few metres. Instead each splat's red
/// is set to its view depth / 10 on a black background, so the red channel
/// is exactly D / 10: a red gradient g and a depth gradient g / 10 must give
/// the same opacity, conic and 2D-mean gradients, and the mean gradients
/// must differ by exactly W^T e_z times the red colour gradient / 10.
#[test]
fn depth_channel_gradients_match_the_colour_channel() {
    let Some(mut cuda) = rasterizer() else { return };
    let (w, h) = (64u32, 48u32);
    let view = view(w, h);
    let mut scene = random_scene(40, 5, 0, 0.0);
    let s = 10.0f32;
    let sh0 = visloc_gsplat_core::sh::SH_C0;
    for g in scene.gaussians.iter_mut() {
        let z = view.world_to_camera_point(g.mean).z;
        g.sh_dc = [(z / s - 0.5) / sh0, (0.3 - 0.5) / sh0, (0.6 - 0.5) / sh0];
    }
    let npix = (w * h) as usize;
    let g: Vec<f32> = (0..npix).map(|i| (i as f32 * 0.31).sin()).collect();
    cuda.set_scene(&PackedScene::from_scene(&scene))
        .expect("upload");
    let f = cuda
        .render_depth(&view, [0.0; 3], 0, false)
        .expect("render");
    let red_err = f
        .expected
        .iter()
        .zip(f.rgb.chunks_exact(3))
        .map(|(d, c)| (d / s - c[0]).abs())
        .fold(0.0f32, f32::max);
    assert!(red_err < 1e-5, "red is not D / s: {red_err}");
    let mut d_red = vec![0.0f32; npix * 3];
    for (k, v) in g.iter().enumerate() {
        d_red[k * 3] = *v;
    }
    let by_red = cuda.backward(&d_red).expect("backward");
    let d_depth: Vec<f32> = g.iter().map(|v| v / s).collect();
    let by_depth = cuda
        .backward_with_depth(&vec![0.0; npix * 3], Some(&d_depth))
        .expect("backward");

    let scale = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
    // Screen gradients du dv dA dB dC dopacity.
    let screen_scale = scale(&by_red.screen);
    let mut worst = 0.0f32;
    for i in 0..scene.len() {
        for c in 0..6 {
            let (a, b) = (by_red.screen[i * 10 + c], by_depth.screen[i * 10 + c]);
            worst = worst.max((a - b).abs() / screen_scale);
        }
    }
    // Mean: the depth path adds W^T e_z * (red colour gradient / s).
    let wrow = view.rotation.row(2);
    let mean_scale = scale(&by_depth.transforms);
    for i in 0..scene.len() {
        let extra = by_red.screen[i * 10 + 6] / s;
        for k in 0..3 {
            let want = by_red.transforms[i * 10 + k] + wrow[k] * extra;
            worst = worst.max((by_depth.transforms[i * 10 + k] - want).abs() / mean_scale);
        }
        for k in 3..10 {
            let (a, b) = (
                by_red.transforms[i * 10 + k],
                by_depth.transforms[i * 10 + k],
            );
            worst = worst.max((a - b).abs() / mean_scale);
        }
        worst = worst.max((by_red.opacity[i] - by_depth.opacity[i]).abs() / scale(&by_red.opacity));
    }
    eprintln!("depth channel against the red channel: worst error {worst:.2e}");
    assert!(worst < 1e-4, "depth channel off by {worst:.2e}");
}

/// The camera-space normal the CUDA side blends for a splat of rotation `q`
/// and log-scales `ls` at camera-space centre `p`: the axis of the smallest
/// scale (ties to the lower axis), turned to face the camera when `flip` is
/// `None`, else multiplied by the given sign. Returns the normal and its sign.
fn splat_normal(
    view: &CameraView,
    q: Quaternion<f64>,
    ls: Vector3<f32>,
    p: Vector3<f32>,
    flip: Option<f64>,
) -> (Vector3<f64>, f64) {
    let r = nalgebra::UnitQuaternion::from_quaternion(q).to_rotation_matrix();
    let mut k = 0;
    if ls.y < ls[k] {
        k = 1;
    }
    if ls.z < ls[k] {
        k = 2;
    }
    let n = view.rotation.cast::<f64>() * r.matrix().column(k).into_owned();
    let sign = flip.unwrap_or(if n.dot(&p.cast::<f64>()) > 0.0 {
        -1.0
    } else {
        1.0
    });
    (n * sign, sign)
}

/// The normal channel's gradients (the depth-normal loss's path), by the
/// depth test's trick: each splat's colour is set to its normal + 1 on black,
/// so the colour is N + A (A = 1 - T), and a colour-1 render gives A. A colour
/// gradient g then gives, through alpha, the normal channel's opacity, conic,
/// 2D-mean, mean and scale gradients plus the colour-1 render's. The
/// quaternion gradients also differ by J_n^T g_n, the rotation of the normal
/// (central differences of the smooth n(q) on the CPU), with g_n the colour
/// gradient.
#[test]
fn normal_channel_gradients_match_the_colour_channel() {
    let Some(mut cuda) = rasterizer() else { return };
    let (w, h) = (64u32, 48u32);
    let view = view(w, h);
    let base = random_scene(40, 7, 0, 0.0);
    let sh0 = visloc_gsplat_core::sh::SH_C0;
    let quat = |g: &Gaussian| g.rotation.cast::<f64>();
    let mut signs = Vec::new();
    let mut coloured = base.clone();
    for g in coloured.gaussians.iter_mut() {
        let p = view.world_to_camera_point(g.mean);
        let (n, sign) = splat_normal(&view, quat(g), g.scale_log, p, None);
        signs.push(sign);
        g.sh_dc = [0, 1, 2].map(|c| (n[c] as f32 + 1.0 - 0.5) / sh0);
    }
    let mut ones = base.clone();
    for g in ones.gaussians.iter_mut() {
        g.sh_dc = [(1.0 - 0.5) / sh0; 3];
    }
    let npix = (w * h) as usize;
    let d: Vec<f32> = (0..npix * 3).map(|i| (i as f32 * 0.29).sin()).collect();

    cuda.set_scene(&PackedScene::from_scene(&coloured))
        .expect("upload");
    let f = cuda.render_geometry(&view, [0.0; 3], 0).expect("render");
    let fwd_err = (0..npix * 3)
        .map(|i| (f.rgb[i] - (f.normal[i] + f.opacity[i / 3])).abs())
        .fold(0.0f32, f32::max);
    assert!(fwd_err < 1e-5, "colour is not N + A: {fwd_err}");
    let by_colour = cuda.backward(&d).expect("backward");
    let by_normal = cuda
        .backward_with_geometry(&vec![0.0; npix * 3], Some(&vec![0.0; npix]), Some(&d))
        .expect("backward");
    cuda.set_scene(&PackedScene::from_scene(&ones))
        .expect("upload");
    cuda.render(&view, [0.0; 3], 0).expect("render");
    let by_one = cuda.backward(&d).expect("backward");

    let scale = |v: &[f32]| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-9);
    let screen_scale = scale(&by_normal.screen);
    let tr_scale = scale(&by_normal.transforms);
    let op_scale = scale(&by_normal.opacity);
    let mut worst = 0.0f32;
    let mut rotation_part = 0.0f32;
    for (i, g) in base.gaussians.iter().enumerate() {
        for c in 0..6 {
            let want = by_colour.screen[i * 10 + c] - by_one.screen[i * 10 + c];
            worst = worst.max((by_normal.screen[i * 10 + c] - want).abs() / screen_scale);
        }
        let want = by_colour.opacity[i] - by_one.opacity[i];
        worst = worst.max((by_normal.opacity[i] - want).abs() / op_scale);
        // dL/dn per splat: the colour gradient of the n + 1 render.
        let g_n = Vector3::from_fn(|c, _| f64::from(by_colour.screen[i * 10 + 6 + c]));
        let p = view.world_to_camera_point(g.mean);
        let q = quat(g);
        let hq = 1e-6;
        for k in 0..10 {
            let mut want = by_colour.transforms[i * 10 + k] - by_one.transforms[i * 10 + k];
            if (3..7).contains(&k) {
                let mut dq = [0.0f64; 4];
                dq[k - 3] = hq;
                let step = Quaternion::new(dq[0], dq[1], dq[2], dq[3]);
                let (np, _) = splat_normal(&view, q + step, g.scale_log, p, Some(signs[i]));
                let (nm, _) = splat_normal(&view, q - step, g.scale_log, p, Some(signs[i]));
                let extra = ((np - nm).dot(&g_n) / (2.0 * hq)) as f32;
                rotation_part = rotation_part.max(extra.abs() / tr_scale);
                want += extra;
            }
            worst = worst.max((by_normal.transforms[i * 10 + k] - want).abs() / tr_scale);
        }
    }
    eprintln!(
        "normal channel against the colour channel: worst error {worst:.2e} \
         (rotation of the normal up to {rotation_part:.2e} of the largest gradient)"
    );
    assert!(
        rotation_part > 1e-2,
        "the test does not exercise the normal's rotation"
    );
    assert!(worst < 2e-4, "normal channel off by {worst:.2e}");
}

/// The depth-normal loss L(N, D) on the CPU in f64, alpha held fixed (see
/// normal_grad_kernel): weight / npix * sum over the valid pixels of
/// 1 - s N . n_d.
fn normal_loss_cpu(cam: &PinholeCamera, weight: f64, n: &[f64], d: &[f64], alpha: &[f64]) -> f64 {
    let (w, h) = (cam.width as usize, cam.height as usize);
    let ray = |x: usize, y: usize| {
        Vector3::new(
            (x as f64 + 0.5 - f64::from(cam.cx)) / f64::from(cam.fx),
            (y as f64 + 0.5 - f64::from(cam.cy)) / f64::from(cam.fy),
            1.0,
        )
    };
    let point = |x: usize, y: usize| {
        let i = y * w + x;
        ray(x, y) * (d[i] / alpha[i].max(1e-6))
    };
    let mut sum = 0.0;
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = y * w + x;
            if [i, i - 1, i + 1, i - w, i + w]
                .iter()
                .any(|&j| alpha[j] <= 0.5)
            {
                continue;
            }
            let c = (point(x + 1, y) - point(x - 1, y)).cross(&(point(x, y + 1) - point(x, y - 1)));
            let cl = c.norm();
            if cl <= 1e-12 {
                continue;
            }
            let nd = c / cl;
            let s = if nd.dot(&point(x, y)) > 0.0 {
                -1.0
            } else {
                1.0
            };
            let big_n = Vector3::new(n[i * 3], n[i * 3 + 1], n[i * 3 + 2]);
            sum += 1.0 - s * big_n.dot(&nd);
        }
    }
    weight * sum / (w * h) as f64
}

/// The depth-normal loss's value and its gradients with respect to the normal
/// and expected-depth channels, against central differences of
/// [`normal_loss_cpu`] on the rendered channels.
#[test]
fn depth_normal_loss_matches_cpu() {
    let Some(mut cuda) = rasterizer() else { return };
    let (w, h) = (40u32, 32u32);
    let view = view(w, h);
    let scene = random_scene(400, 11, 0, 3.0);
    cuda.set_scene(&PackedScene::from_scene(&scene))
        .expect("upload");
    let f = cuda.render_geometry(&view, BG, 0).expect("render");
    let weight = 0.7f32;
    let gpu = cuda.normal_loss(weight).expect("normal loss");
    let npix = (w * h) as usize;
    let n: Vec<f64> = f.normal.iter().map(|&v| f64::from(v)).collect();
    let d: Vec<f64> = f.expected.iter().map(|&v| f64::from(v)).collect();
    let alpha: Vec<f64> = f.opacity.iter().map(|&v| f64::from(v)).collect();
    let weight = f64::from(weight);
    let cpu = normal_loss_cpu(&view.camera, weight, &n, &d, &alpha);
    assert!(
        (gpu.loss * weight - cpu).abs() < 1e-5 * cpu.abs().max(1e-3),
        "loss: gpu {} cpu {cpu}",
        gpu.loss * weight
    );
    let valid = gpu
        .d_normal
        .chunks_exact(3)
        .filter(|g| g.iter().any(|&v| v != 0.0))
        .count();
    assert!(
        valid > npix / 4,
        "only {valid} of {npix} pixels in the loss"
    );

    let fd = |v: &[f64], i: usize, eval: &dyn Fn(&[f64]) -> f64| {
        let hh = 1e-6 * v[i].abs().max(1.0);
        let mut plus = v.to_vec();
        plus[i] += hh;
        let mut minus = v.to_vec();
        minus[i] -= hh;
        (eval(&plus) - eval(&minus)) / (2.0 * hh)
    };
    let (mut worst_n, mut scale_n) = (0.0f64, 0.0f64);
    for i in 0..npix * 3 {
        let want = fd(&n, i, &|v| {
            normal_loss_cpu(&view.camera, weight, v, &d, &alpha)
        });
        scale_n = scale_n.max(want.abs());
        worst_n = worst_n.max((f64::from(gpu.d_normal[i]) - want).abs());
    }
    let (mut worst_d, mut scale_d) = (0.0f64, 0.0f64);
    for i in 0..npix {
        let want = fd(&d, i, &|v| {
            normal_loss_cpu(&view.camera, weight, &n, v, &alpha)
        });
        scale_d = scale_d.max(want.abs());
        worst_d = worst_d.max((f64::from(gpu.d_depth[i]) - want).abs());
    }
    let (rel_n, rel_d) = (worst_n / scale_n, worst_d / scale_d);
    eprintln!(
        "depth-normal loss: d_normal off by {rel_n:.2e}, d_depth by {rel_d:.2e} of their largest values"
    );
    assert!(rel_n < 1e-4, "d_normal off by {rel_n:.2e}");
    assert!(rel_d < 1e-3, "d_depth off by {rel_d:.2e}");
}
