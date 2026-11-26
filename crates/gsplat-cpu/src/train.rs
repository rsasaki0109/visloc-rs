//! The training loop: Adam (the Inria recipe, one view per step) and a
//! per-gaussian local Gauss-Newton step after 3DGS² (Lan et al., arXiv
//! 2501.13975): each iteration renders one primary view at full resolution
//! and its `K` nearest training views at half resolution, sums every
//! gaussian's gradient and Gauss-Newton matrix over them, and moves every
//! gaussian by its own damped Newton step at once (Jacobi), step size 1.
//!
//! Both optimizers share the projection, rasterizer and loss, so a
//! comparison between them measures the optimizer alone.

use std::time::Instant;

use rayon::prelude::*;
use visloc_gsplat_core::camera::{CameraView, PinholeCamera};

use crate::model::{Model, NCOL, NGEO, NSH};
use crate::project::{jacobian, project_all, Proj};
use crate::raster::{backward, forward, tri6, Frame, Raster, B_COL, B_GRAD, B_H, B_HC, NB};

/// Upper triangle of an 11x11 symmetric matrix.
pub const NTRI: usize = NGEO * (NGEO + 1) / 2;

#[inline(always)]
fn tri11(i: usize, j: usize) -> usize {
    i * NGEO - i * (i.wrapping_sub(1)) / 2 + (j - i)
}

/// A training view with its image at full and half resolution.
pub struct TrainView {
    pub view: CameraView,
    pub gt: Vec<f32>,
    pub half_view: CameraView,
    pub half_gt: Vec<f32>,
}

impl TrainView {
    pub fn new(view: CameraView, rgb: &[[f32; 3]]) -> Self {
        let (w, h) = (view.camera.width as usize, view.camera.height as usize);
        let gt: Vec<f32> = rgb.iter().flatten().copied().collect();
        let (hw, hh) = (w / 2, h / 2);
        let mut half_gt = vec![0.0f32; hw * hh * 3];
        for y in 0..hh {
            for x in 0..hw {
                for ch in 0..3 {
                    let s = |xx: usize, yy: usize| gt[(yy * w + xx) * 3 + ch];
                    half_gt[(y * hw + x) * 3 + ch] =
                        0.25 * (s(2 * x, 2 * y) + s(2 * x + 1, 2 * y) + s(2 * x, 2 * y + 1) + s(2 * x + 1, 2 * y + 1));
                }
            }
        }
        let c = view.camera;
        let half_cam = PinholeCamera {
            width: hw as u32,
            height: hh as u32,
            fx: c.fx * hw as f32 / w as f32,
            fy: c.fy * hh as f32 / h as f32,
            cx: c.cx * hw as f32 / w as f32,
            cy: c.cy * hh as f32 / h as f32,
            ..c
        };
        Self {
            view,
            gt,
            half_view: CameraView { camera: half_cam, ..view },
            half_gt,
        }
    }
}

/// A rendered view, kept for its backward pass.
pub struct Rendered {
    pub proj: Vec<Proj>,
    pub frame: Frame,
    pub raster: Raster,
}

pub fn render(model: &Model, view: &CameraView, bg: [f32; 3]) -> Rendered {
    let proj = project_all(model, view);
    let frame = Frame::new(&proj, view.camera.width, view.camera.height, bg);
    let raster = forward(&frame);
    Rendered { proj, frame, raster }
}

/// Which loss the image term is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loss {
    /// Mean squared error over pixels and channels.
    L2,
    /// Mean absolute error (Adam only: no Gauss-Newton matrix).
    L1,
}

/// The loss and `dL/dC` per pixel channel.
pub fn image_loss(image: &[f32], gt: &[f32], loss: Loss) -> (f64, Vec<f32>) {
    let n = image.len() as f32;
    let mut d = vec![0.0f32; image.len()];
    let total: f64 = d
        .par_chunks_mut(4096)
        .zip(image.par_chunks(4096).zip(gt.par_chunks(4096)))
        .map(|(dd, (im, g))| {
            let mut s = 0.0f64;
            for ((o, a), b) in dd.iter_mut().zip(im).zip(g) {
                let r = a - b;
                match loss {
                    Loss::L2 => {
                        s += (r * r) as f64;
                        *o = 2.0 * r / n;
                    }
                    Loss::L1 => {
                        s += r.abs() as f64;
                        *o = r.signum() / n;
                    }
                }
            }
            s
        })
        .sum();
    (total / n as f64, d)
}

/// Per-gaussian sums of one Newton iteration.
struct Accum {
    h: Vec<[f32; NTRI]>,
    g: Vec<[f32; NGEO]>,
    hcol: Vec<[f32; NSH]>,
    gcol: Vec<f32>,
}

impl Accum {
    fn new(n: usize) -> Self {
        Self {
            h: vec![[0.0; NTRI]; n],
            g: vec![[0.0; NGEO]; n],
            hcol: vec![[0.0; NSH]; n],
            gcol: vec![0.0; n * NCOL],
        }
    }
    fn clear(&mut self) {
        self.h.par_iter_mut().for_each(|x| *x = [0.0; NTRI]);
        self.g.par_iter_mut().for_each(|x| *x = [0.0; NGEO]);
        self.hcol.par_iter_mut().for_each(|x| *x = [0.0; NSH]);
        self.gcol.par_iter_mut().for_each(|x| *x = 0.0);
    }
}

/// Adam moments for every parameter.
struct AdamState {
    m_geo: Vec<[f32; NGEO]>,
    v_geo: Vec<[f32; NGEO]>,
    m_col: Vec<f32>,
    v_col: Vec<f32>,
    t: i32,
}

#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub lr_mean: f32,
    pub lr_mean_final: f32,
    pub lr_scale: f32,
    pub lr_quat: f32,
    pub lr_opacity: f32,
    pub lr_dc: f32,
    pub lr_rest: f32,
    /// Steps per SH degree increase.
    pub sh_interval: usize,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr_mean: 1.6e-4,
            lr_mean_final: 1.6e-6,
            lr_scale: 5e-3,
            lr_quat: 1e-3,
            lr_opacity: 5e-2,
            lr_dc: 2.5e-3,
            lr_rest: 2.5e-3 / 20.0,
            sh_interval: 1000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NewtonConfig {
    /// Secondary views per iteration (3DGS²: 3).
    pub neighbours: usize,
    /// Weight of each secondary view's loss.
    pub neighbour_weight: f32,
    /// Marquardt damping: the diagonal is multiplied by `1 + lambda`.
    pub lambda: f64,
    /// Iterations per SH degree increase.
    pub sh_interval: usize,
    /// Largest log-scale change per iteration.
    pub max_dlog_scale: f32,
    /// Largest mean step, in units of the gaussian's largest scale.
    pub max_dmean: f32,
    pub max_dlogit: f32,
    pub max_dquat: f32,
    pub max_dcol: f32,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            neighbours: 3,
            neighbour_weight: 1.0,
            lambda: 0.1,
            sh_interval: 100,
            max_dlog_scale: 0.5,
            max_dmean: 1.0,
            max_dlogit: 2.0,
            max_dquat: 0.5,
            max_dcol: 1.0,
        }
    }
}

pub struct Trainer {
    pub model: Model,
    pub views: Vec<TrainView>,
    pub bg: [f32; 3],
    pub extent: f32,
    neighbours: Vec<Vec<usize>>,
    adam: Option<AdamState>,
    acc: Option<Accum>,
    pub loss: Loss,
    /// See [`backward`]'s `overlap`.
    pub overlap: f32,
    /// Seconds spent per stage: project+bin, forward, loss, backward, lift, update.
    pub timing: [f64; 6],
}

impl Trainer {
    pub fn new(model: Model, views: Vec<TrainView>, bg: [f32; 3]) -> Self {
        let centres: Vec<_> = views.iter().map(|v| v.view.camera_center()).collect();
        let mean = centres.iter().fold(nalgebra::Vector3::zeros(), |a, c| a + c) / centres.len() as f32;
        let extent = 1.1 * centres.iter().map(|c| (c - mean).norm()).fold(0.0f32, f32::max);
        let dirs: Vec<_> = views.iter().map(|v| v.view.rotation.row(2).transpose()).collect();
        let neighbours = (0..views.len())
            .map(|i| {
                let mut d: Vec<(f32, usize)> = (0..views.len())
                    .filter(|&j| j != i)
                    .map(|j| {
                        let dist = (centres[i] - centres[j]).norm() / extent;
                        let ang = 1.0 - dirs[i].dot(&dirs[j]);
                        (dist + ang, j)
                    })
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                d.into_iter().take(8).map(|x| x.1).collect()
            })
            .collect();
        Self {
            model,
            views,
            bg,
            extent,
            neighbours,
            adam: None,
            acc: None,
            loss: Loss::L2,
            overlap: 1.0,
            timing: [0.0; 6],
        }
    }

    /// Render, take the loss and run the backward pass of one view.
    fn view_grads(&mut self, view: &CameraView, gt: &[f32], weight: f32, gauss_newton: bool) -> (f64, Rendered, Vec<[f32; NB]>) {
        let t0 = Instant::now();
        let proj = project_all(&self.model, view);
        let frame = Frame::new(&proj, view.camera.width, view.camera.height, self.bg);
        let t1 = Instant::now();
        let raster = forward(&frame);
        let t2 = Instant::now();
        let (loss, mut d) = image_loss(&raster.image, gt, self.loss);
        if weight != 1.0 {
            d.par_iter_mut().for_each(|x| *x *= weight);
        }
        // Gauss-Newton weight of a mean squared error: 2 / (3 N) per pixel.
        let h_px = weight * 2.0 / (gt.len() as f32);
        let t3 = Instant::now();
        let out = backward(&frame, &raster, &d, h_px, gauss_newton, self.overlap);
        let t4 = Instant::now();
        self.timing[0] += (t1 - t0).as_secs_f64();
        self.timing[1] += (t2 - t1).as_secs_f64();
        self.timing[2] += (t3 - t2).as_secs_f64();
        self.timing[3] += (t4 - t3).as_secs_f64();
        (loss, Rendered { proj, frame, raster }, out)
    }

    /// One Adam step on view `vi`.
    pub fn adam_step(&mut self, vi: usize, step: usize, total: usize, cfg: &AdamConfig) -> f64 {
        let n = self.model.len();
        self.model.active_degree = ((step / cfg.sh_interval) as u32).min(3);
        let (view, gt) = {
            let v = &self.views[vi];
            (v.view, std::mem::take(&mut self.views[vi].gt))
        };
        let (loss, r, out) = self.view_grads(&view, &gt, 1.0, false);
        self.views[vi].gt = gt;
        let t0 = Instant::now();
        // Per-gaussian gradients.
        let mut g_geo = vec![[0.0f32; NGEO]; n];
        let mut g_col = vec![0.0f32; n * NCOL];
        let mut idx = vec![u32::MAX; n];
        for (c, p) in r.proj.iter().enumerate() {
            idx[p.id as usize] = c as u32;
        }
        let model = &self.model;
        let view = &view;
        g_geo
            .par_iter_mut()
            .zip(g_col.par_chunks_mut(NCOL))
            .zip(idx.par_iter())
            .for_each(|((gg, gc), &c)| {
                if c == u32::MAX {
                    return;
                }
                let p = &r.proj[c as usize];
                let o = &out[c as usize];
                if o[B_GRAD..B_GRAD + 6].iter().any(|&x| x != 0.0) {
                    if let Some(j) = jacobian(model, p.id as usize, view) {
                        for q in 0..6 {
                            let gs = o[B_GRAD + q];
                            for (k, x) in gg.iter_mut().enumerate() {
                                *x += gs * j[q][k];
                            }
                        }
                    }
                }
                for ch in 0..3 {
                    if p.clamped[ch] {
                        continue;
                    }
                    let g = o[B_COL + ch];
                    for k in 0..NSH {
                        gc[ch * NSH + k] = g * p.basis[k];
                    }
                }
            });
        let t1 = Instant::now();
        let st = self.adam.get_or_insert_with(|| AdamState {
            m_geo: vec![[0.0; NGEO]; n],
            v_geo: vec![[0.0; NGEO]; n],
            m_col: vec![0.0; n * NCOL],
            v_col: vec![0.0; n * NCOL],
            t: 0,
        });
        st.t += 1;
        let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-15f32);
        let bc1 = 1.0 - b1.powi(st.t);
        let bc2 = 1.0 - b2.powi(st.t);
        let frac = step as f32 / total.max(1) as f32;
        let lr_mean = self.extent * cfg.lr_mean * (cfg.lr_mean_final / cfg.lr_mean).powf(frac);
        let mut lr_geo = [0.0f32; NGEO];
        lr_geo[0..3].fill(lr_mean);
        lr_geo[3..6].fill(cfg.lr_scale);
        lr_geo[6..10].fill(cfg.lr_quat);
        lr_geo[10] = cfg.lr_opacity;
        self.model
            .geo
            .par_iter_mut()
            .zip(st.m_geo.par_iter_mut().zip(st.v_geo.par_iter_mut()))
            .zip(g_geo.par_iter())
            .for_each(|((p, (m, v)), g)| {
                for k in 0..NGEO {
                    m[k] = b1 * m[k] + (1.0 - b1) * g[k];
                    v[k] = b2 * v[k] + (1.0 - b2) * g[k] * g[k];
                    p[k] -= lr_geo[k] * (m[k] / bc1) / ((v[k] / bc2).sqrt() + eps);
                }
            });
        let (lr_dc, lr_rest) = (cfg.lr_dc, cfg.lr_rest);
        self.model
            .col
            .par_chunks_mut(NCOL)
            .zip(st.m_col.par_chunks_mut(NCOL).zip(st.v_col.par_chunks_mut(NCOL)))
            .zip(g_col.par_chunks(NCOL))
            .for_each(|((p, (m, v)), g)| {
                for i in 0..NCOL {
                    let lr = if i % NSH == 0 { lr_dc } else { lr_rest };
                    m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                    p[i] -= lr * (m[i] / bc1) / ((v[i] / bc2).sqrt() + eps);
                }
            });
        let t2 = Instant::now();
        self.timing[4] += (t1 - t0).as_secs_f64();
        self.timing[5] += (t2 - t1).as_secs_f64();
        loss
    }

    /// Add one view's gradients and Gauss-Newton matrices into `acc`.
    fn lift(acc: &mut Accum, model: &Model, view: &CameraView, r: &Rendered, out: &[[f32; NB]], n: usize) {
        let mut idx = vec![u32::MAX; n];
        for (c, p) in r.proj.iter().enumerate() {
            idx[p.id as usize] = c as u32;
        }
        acc.h
            .par_iter_mut()
            .zip(acc.g.par_iter_mut())
            .zip(acc.hcol.par_iter_mut().zip(acc.gcol.par_chunks_mut(NCOL)))
            .zip(idx.par_iter())
            .for_each(|(((h, g), (hc, gc)), &c)| {
                if c == u32::MAX {
                    return;
                }
                let p = &r.proj[c as usize];
                let o = &out[c as usize];
                if o[B_GRAD..B_GRAD + 6].iter().all(|&x| x == 0.0) && o[B_HC] == 0.0 {
                    return;
                }
                let Some(jm) = jacobian(model, p.id as usize, view) else {
                    return;
                };
                let j = &jm;
                // Screen Gauss-Newton matrix (symmetric 6x6).
                let mut hs = [[0.0f32; 6]; 6];
                for a in 0..6 {
                    for b in a..6 {
                        let v = o[B_H + tri6(a, b)];
                        hs[a][b] = v;
                        hs[b][a] = v;
                    }
                }
                // T = Hs J (6 x 11), then J^T T.
                let mut t = [[0.0f32; NGEO]; 6];
                for a in 0..6 {
                    for b in 0..6 {
                        let hv = hs[a][b];
                        if hv != 0.0 {
                            for k in 0..NGEO {
                                t[a][k] += hv * j[b][k];
                            }
                        }
                    }
                }
                for k0 in 0..NGEO {
                    for k1 in k0..NGEO {
                        let mut s = 0.0f32;
                        for a in 0..6 {
                            s += j[a][k0] * t[a][k1];
                        }
                        h[tri11(k0, k1)] += s;
                    }
                }
                for a in 0..6 {
                    let gs = o[B_GRAD + a];
                    for k in 0..NGEO {
                        g[k] += gs * j[a][k];
                    }
                }
                let hcw = o[B_HC];
                for k in 0..NSH {
                    hc[k] += hcw * p.basis[k] * p.basis[k];
                }
                for ch in 0..3 {
                    if p.clamped[ch] {
                        continue;
                    }
                    let gv = o[B_COL + ch];
                    for k in 0..NSH {
                        gc[ch * NSH + k] += gv * p.basis[k];
                    }
                }
            });
    }

    /// One local Gauss-Newton iteration with primary view `vi`.
    pub fn newton_step(&mut self, vi: usize, it: usize, cfg: &NewtonConfig) -> f64 {
        let n = self.model.len();
        self.model.active_degree = ((it / cfg.sh_interval.max(1)) as u32).min(3);
        let mut acc = self.acc.take().unwrap_or_else(|| Accum::new(n));
        if acc.g.len() != n {
            acc = Accum::new(n);
        }
        acc.clear();
        let mut primary_loss = 0.0;
        let nb: Vec<usize> = self.neighbours[vi].iter().take(cfg.neighbours).copied().collect();
        for (k, v) in std::iter::once(vi).chain(nb).enumerate() {
            let (view, gt, w) = if k == 0 {
                (self.views[v].view, std::mem::take(&mut self.views[v].gt), 1.0)
            } else {
                (
                    self.views[v].half_view,
                    std::mem::take(&mut self.views[v].half_gt),
                    cfg.neighbour_weight,
                )
            };
            let (loss, r, out) = self.view_grads(&view, &gt, w, true);
            if k == 0 {
                primary_loss = loss;
                self.views[v].gt = gt;
            } else {
                self.views[v].half_gt = gt;
            }
            let t0 = Instant::now();
            Self::lift(&mut acc, &self.model, &view, &r, &out, n);
            self.timing[4] += t0.elapsed().as_secs_f64();
        }
        let t0 = Instant::now();
        let lambda = cfg.lambda;
        self.model
            .geo
            .par_iter_mut()
            .zip(self.model.col.par_chunks_mut(NCOL))
            .zip(acc.h.par_iter().zip(acc.g.par_iter()))
            .zip(acc.hcol.par_iter().zip(acc.gcol.par_chunks(NCOL)))
            .for_each(|(((p, col), (h, g)), (hc, gc))| {
                solve_geometry(p, h, g, lambda, cfg);
                for ch in 0..3 {
                    for k in 0..NSH {
                        let hh = hc[k] as f64;
                        if hh <= 0.0 {
                            continue;
                        }
                        let step = -(gc[ch * NSH + k] as f64) / (hh * (1.0 + lambda) + 1e-12);
                        col[ch * NSH + k] += (step as f32).clamp(-cfg.max_dcol, cfg.max_dcol);
                    }
                }
            });
        self.timing[5] += t0.elapsed().as_secs_f64();
        self.acc = Some(acc);
        primary_loss
    }
}

/// Damped Newton step of one gaussian's geometry, scaled down to the trust
/// limits, applied in place.
fn solve_geometry(p: &mut [f32; NGEO], h: &[f32; NTRI], g: &[f32; NGEO], lambda: f64, cfg: &NewtonConfig) {
    let mut a = [[0.0f64; NGEO]; NGEO];
    let mut maxd = 0.0f64;
    for i in 0..NGEO {
        for j in i..NGEO {
            let v = h[tri11(i, j)] as f64;
            a[i][j] = v;
            a[j][i] = v;
        }
        maxd = maxd.max(a[i][i]);
    }
    if maxd <= 0.0 || !maxd.is_finite() {
        return;
    }
    let floor = 1e-6 * maxd;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = row[i] * (1.0 + lambda) + floor;
    }
    // Cholesky in place (lower).
    for j in 0..NGEO {
        let mut d = a[j][j];
        for k in 0..j {
            d -= a[j][k] * a[j][k];
        }
        if d <= 0.0 || !d.is_finite() {
            return;
        }
        let d = d.sqrt();
        a[j][j] = d;
        for i in j + 1..NGEO {
            let mut s = a[i][j];
            for k in 0..j {
                s -= a[i][k] * a[j][k];
            }
            a[i][j] = s / d;
        }
    }
    let mut x = [0.0f64; NGEO];
    for i in 0..NGEO {
        let mut s = -(g[i] as f64);
        for k in 0..i {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    for i in (0..NGEO).rev() {
        let mut s = x[i];
        for k in i + 1..NGEO {
            s -= a[k][i] * x[k];
        }
        x[i] = s / a[i][i];
    }
    // Trust limits: shrink the whole step to fit the tightest.
    let scale_max = p[3].max(p[4]).max(p[5]).exp() as f64;
    let dm = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    let qn = ((p[6] * p[6] + p[7] * p[7] + p[8] * p[8] + p[9] * p[9]) as f64).sqrt().max(1e-12);
    let dq = (x[6] * x[6] + x[7] * x[7] + x[8] * x[8] + x[9] * x[9]).sqrt() / qn;
    let dls = x[3].abs().max(x[4].abs()).max(x[5].abs());
    let mut s = 1.0f64;
    let mut limit = |v: f64, lim: f64| {
        if v > lim {
            s = s.min(lim / v);
        }
    };
    limit(dm, cfg.max_dmean as f64 * scale_max);
    limit(dq, cfg.max_dquat as f64);
    limit(dls, cfg.max_dlog_scale as f64);
    limit(x[10].abs(), cfg.max_dlogit as f64);
    if !s.is_finite() {
        return;
    }
    for i in 0..NGEO {
        p[i] += (s * x[i]) as f32;
    }
    let qn = (p[6] * p[6] + p[7] * p[7] + p[8] * p[8] + p[9] * p[9]).sqrt();
    if qn > 1e-12 {
        for v in &mut p[6..10] {
            *v /= qn;
        }
    }
}

/// One view's mean-squared-error gradient and per-gaussian Gauss-Newton
/// matrices (upper triangles), and the colour gradient and diagonal
/// Gauss-Newton weights (for tests and diagnostics).
#[allow(clippy::type_complexity)]
pub fn view_gauss_newton(
    model: &Model,
    view: &CameraView,
    gt: &[f32],
    bg: [f32; 3],
) -> (f64, Vec<[f32; NTRI]>, Vec<[f32; NGEO]>, Vec<[f32; NSH]>, Vec<f32>) {
    let n = model.len();
    let r = render(model, view, bg);
    let (loss, d) = image_loss(&r.raster.image, gt, Loss::L2);
    let out = backward(&r.frame, &r.raster, &d, 2.0 / gt.len() as f32, true, 0.0);
    let mut acc = Accum::new(n);
    Trainer::lift(&mut acc, model, view, &r, &out, n);
    (loss, acc.h, acc.g, acc.hcol, acc.gcol)
}

/// Index of `(i, j)`, `i <= j`, in [`view_gauss_newton`]'s upper triangles.
pub fn tri(i: usize, j: usize) -> usize {
    let (i, j) = if i <= j { (i, j) } else { (j, i) };
    tri11(i, j)
}
