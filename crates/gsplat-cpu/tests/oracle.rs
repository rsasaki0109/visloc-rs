//! The CPU trainer's image, gradients and Gauss-Newton matrices against the
//! f64 oracle of `visloc_gsplat_core::backward` and finite differences.

use nalgebra::{Quaternion, Vector3};
use visloc_gsplat_core::backward::{render_backward, render_f64};
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};
use visloc_gsplat_core::gaussian::{Gaussian, Scene};
use visloc_gsplat_cpu::model::{Model, NSH};
use visloc_gsplat_cpu::train::{render, tri, view_gauss_newton};

fn scene(degree: u32) -> Scene {
    let rest = 3 * (((degree + 1) * (degree + 1)) as usize - 1);
    let mk = |m: [f32; 3], s: [f32; 3], q: [f32; 4], o: f32, dc: [f32; 3], seed: f32| Gaussian {
        mean: Vector3::new(m[0], m[1], m[2]),
        scale_log: Vector3::new(s[0].ln(), s[1].ln(), s[2].ln()),
        rotation: Quaternion::new(q[0], q[1], q[2], q[3]),
        opacity_logit: o,
        sh_dc: dc,
        sh_rest: (0..rest)
            .map(|k| 0.05 * ((k as f32 + seed) * 1.7).sin())
            .collect(),
        sh_degree: degree,
    };
    Scene::new(
        vec![
            mk(
                [0.0, 0.0, 4.0],
                [0.4, 0.2, 0.3],
                [0.9, 0.2, -0.3, 0.1],
                0.2,
                [0.8, -0.3, 0.1],
                0.0,
            ),
            mk(
                [0.3, -0.2, 5.0],
                [0.3, 0.5, 0.2],
                [0.7, -0.1, 0.4, 0.5],
                -0.4,
                [-0.2, 0.6, 0.3],
                1.0,
            ),
            mk(
                [-0.4, 0.3, 6.0],
                [0.6, 0.3, 0.4],
                [1.1, 0.3, 0.2, -0.2],
                0.5,
                [0.1, 0.2, -0.5],
                2.0,
            ),
        ],
        degree,
    )
}

fn view() -> CameraView {
    let r = nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner();
    CameraView::new(
        r,
        Vector3::new(0.1, -0.05, 0.2),
        PinholeCamera::new(40, 32, 45.0, 44.0, 20.0, 16.0),
    )
}

fn target(n: usize) -> Vec<f32> {
    (0..n * 3)
        .map(|i| 0.5 + 0.3 * ((i as f32) * 0.37).sin())
        .collect()
}

#[test]
fn forward_matches_oracle() {
    let s = scene(2);
    let mut m = Model::from_scene(&s);
    m.active_degree = 2;
    let v = view();
    let r = render(&m, &v, [0.1, 0.2, 0.3]);
    let o = render_f64(&s, &v, [0.1, 0.2, 0.3]);
    let max = r
        .raster
        .image
        .chunks(3)
        .zip(&o)
        .flat_map(|(a, b)| (0..3).map(move |c| (a[c] as f64 - b[c]).abs()))
        .fold(0.0f64, f64::max);
    assert!(max < 2e-4, "max abs diff {max}");
}

#[test]
fn gradients_match_oracle() {
    // Degree 0: the colour does not depend on the view direction, which the
    // CPU trainer holds fixed.
    let s = scene(0);
    let m = Model::from_scene(&s);
    let v = view();
    let n = (v.camera.width * v.camera.height) as usize;
    let gt = target(n);
    let (_, _, g, _, gcol) = view_gauss_newton(&m, &v, &gt, [0.0; 3]);
    // The oracle's gradient of sum dot(d, C) with d = dL/dC of the MSE.
    let r = render_f64(&s, &v, [0.0; 3]);
    let d: Vec<[f64; 3]> = r
        .iter()
        .enumerate()
        .map(|(p, c)| {
            std::array::from_fn(|ch| 2.0 * (c[ch] - gt[p * 3 + ch] as f64) / (3 * n) as f64)
        })
        .collect();
    let o = render_backward(&s, &v, [0.0; 3], &d);
    for i in 0..s.len() {
        let want: Vec<f64> = o.mean[i]
            .iter()
            .chain(&o.scale_log[i])
            .chain(&o.rotation[i])
            .chain(std::iter::once(&o.opacity_logit[i]))
            .copied()
            .collect();
        let scale = want.iter().fold(1e-9f64, |a, b| a.max(b.abs()));
        for k in 0..11 {
            let err = (g[i][k] as f64 - want[k]).abs() / scale;
            assert!(
                err < 1e-3,
                "gaussian {i} param {k}: {} vs {}",
                g[i][k],
                want[k]
            );
        }
        for ch in 0..3 {
            let got = gcol[i * 3 * NSH + ch * NSH] as f64;
            let want = o.sh_dc[i][ch];
            assert!(
                (got - want).abs() < 1e-3 * want.abs().max(1e-6) + 1e-7,
                "dc {i} {ch}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn gauss_newton_matches_finite_differences() {
    // J of the whole image w.r.t. each geometry parameter by central
    // differences of the f64 oracle; H = 2/(3N) J^T J.
    let s = scene(0);
    let m = Model::from_scene(&s);
    let v = view();
    let n = (v.camera.width * v.camera.height) as usize;
    let gt = target(n);
    let (_, h, _, hcol, _) = view_gauss_newton(&m, &v, &gt, [0.0; 3]);
    let params = |g: &Gaussian| -> [f64; 11] {
        [
            g.mean.x as f64,
            g.mean.y as f64,
            g.mean.z as f64,
            g.scale_log.x as f64,
            g.scale_log.y as f64,
            g.scale_log.z as f64,
            g.rotation.w as f64,
            g.rotation.i as f64,
            g.rotation.j as f64,
            g.rotation.k as f64,
            g.opacity_logit as f64,
        ]
    };
    let set = |g: &mut Gaussian, p: [f64; 11]| {
        g.mean = Vector3::new(p[0] as f32, p[1] as f32, p[2] as f32);
        g.scale_log = Vector3::new(p[3] as f32, p[4] as f32, p[5] as f32);
        g.rotation = Quaternion::new(p[6] as f32, p[7] as f32, p[8] as f32, p[9] as f32);
        g.opacity_logit = p[10] as f32;
    };
    for i in 0..s.len() {
        let mut jac = vec![vec![0.0f64; n * 3]; 11];
        for k in 0..11 {
            // The alpha >= 1/255 edge is a jump, which adds ~1/eps to a
            // difference's square: with a fourth-order stencil, 3e-2 passes
            // and smaller steps overshoot the analytic matrix more the
            // smaller they are (1e-2: 2.5%, 3e-3: 6%), as that term does.
            let eps: f64 = std::env::var("FD_EPS")
                .ok()
                .and_then(|x| x.parse().ok())
                .unwrap_or(3e-2);
            let mut sp = s.clone();
            let mut p = params(&sp.gaussians[i]);
            p[k] += eps;
            set(&mut sp.gaussians[i], p);
            let a = render_f64(&sp, &v, [0.0; 3]);
            p[k] -= 2.0 * eps;
            set(&mut sp.gaussians[i], p);
            let b = render_f64(&sp, &v, [0.0; 3]);
            p[k] += 3.0 * eps;
            set(&mut sp.gaussians[i], p);
            let a2 = render_f64(&sp, &v, [0.0; 3]);
            p[k] -= 4.0 * eps;
            set(&mut sp.gaussians[i], p);
            let b2 = render_f64(&sp, &v, [0.0; 3]);
            // Fourth-order central difference.
            for px in 0..n {
                for ch in 0..3 {
                    jac[k][px * 3 + ch] =
                        (8.0 * (a[px][ch] - b[px][ch]) - (a2[px][ch] - b2[px][ch])) / (12.0 * eps);
                }
            }
        }
        let w = 2.0 / (3 * n) as f64;
        let mut scale = 1e-12f64;
        let mut fd = [[0.0f64; 11]; 11];
        for a in 0..11 {
            for b in 0..11 {
                fd[a][b] = w * jac[a].iter().zip(&jac[b]).map(|(x, y)| x * y).sum::<f64>();
                scale = scale.max(fd[a][b].abs());
            }
        }
        for a in 0..11 {
            for b in a..11 {
                let got = h[i][tri(a, b)] as f64;
                let err = (got - fd[a][b]).abs() / scale;
                // Finite differences of a gated renderer: a loose bound.
                assert!(
                    err < 2e-2,
                    "gaussian {i} H[{a}][{b}]: {got} vs {}",
                    fd[a][b]
                );
            }
        }
        // Colour: d C / d dc = alpha T * SH_C0.
        let mut fdc = 0.0;
        let eps = 1e-3;
        let mut sp = s.clone();
        sp.gaussians[i].sh_dc[0] += eps as f32;
        let a = render_f64(&sp, &v, [0.0; 3]);
        sp.gaussians[i].sh_dc[0] -= 2.0 * eps as f32;
        let b = render_f64(&sp, &v, [0.0; 3]);
        for px in 0..n {
            let jj = (a[px][0] - b[px][0]) / (2.0 * eps);
            fdc += 2.0 / (3 * n) as f64 * jj * jj;
        }
        let got = hcol[i][0] as f64;
        assert!(
            (got - fdc).abs() < 2e-2 * fdc.abs() + 1e-9,
            "colour H {i}: {got} vs {fdc}"
        );
    }
}

#[test]
fn screen_vjp_matches_the_dual_jacobian() {
    use visloc_gsplat_cpu::project::{jacobian, screen_vjp};
    let s = scene(0);
    let m = Model::from_scene(&s);
    // A wide view too, so some splats sit past the 1.3x frustum clamp.
    let mut views = vec![view()];
    let mut wide = view();
    wide.camera = PinholeCamera::new(40, 32, 8.0, 8.0, 20.0, 16.0);
    views.push(wide);
    let mut moved = view();
    moved.translation = Vector3::new(1.6, -0.9, 0.2);
    views.push(moved);
    // Narrow and off to the side: centres past the clamp, footprints in view.
    let mut narrow = view();
    narrow.camera = PinholeCamera::new(40, 32, 200.0, 200.0, 20.0, 16.0);
    narrow.translation = Vector3::new(0.9, 0.0, 0.2);
    views.push(narrow);
    let mut clamped = 0;
    for v in &views {
        for i in 0..s.len() {
            let Some(j) = jacobian(&m, i, v) else {
                continue;
            };
            let g = [0.3, -0.7, 1.1, -0.4, 0.9, 0.5];
            let got = screen_vjp(&m, i, v, &g).expect("vjp");
            let pc = v.world_to_camera_point(s.gaussians[i].mean);
            if (pc.x / pc.z).abs() > 1.3 * 0.5 * 40.0 / v.camera.fx {
                clamped += 1;
            }
            for k in 0..11 {
                let want: f32 = (0..6).map(|q| g[q] * j[q][k]).sum();
                let scale = want.abs().max(1e-3);
                assert!(
                    (got[k] - want).abs() / scale < 1e-3,
                    "gaussian {i} param {k}: {} vs {want}",
                    got[k]
                );
            }
        }
    }
    assert!(clamped > 0, "no splat past the frustum clamp");
}
