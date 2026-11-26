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
use visloc_gsplat_train::config::pack_rgba8;
use visloc_gsplat_train::densify::{self, DensifyConfig, DensifyReport, Group, Population};
use visloc_gsplat_train::loss::SsimHost;

use crate::model::{Model, NCOL, NGEO, NSH};
use crate::project::{jacobian, project_all, screen_vjp_geo, Proj};
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
    /// `gt` as RGBA8 (for the SSIM term).
    pub gt_packed: Vec<u32>,
    pub half_view: CameraView,
    pub half_gt: Vec<f32>,
}

impl TrainView {
    /// The image box-averaged over `f x f` blocks, and the camera of that
    /// resolution (its pixel `j` is the full image's pixels `f j .. f j + f`).
    pub fn downsampled(&self, f: usize) -> (CameraView, Vec<f32>, Vec<u32>) {
        let c = self.view.camera;
        let (w, h) = (c.width as usize, c.height as usize);
        let (w2, h2) = (w / f, h / f);
        let mut gt = vec![0.0f32; w2 * h2 * 3];
        let inv = 1.0 / (f * f) as f32;
        gt.par_chunks_mut(w2 * 3).enumerate().for_each(|(j, row)| {
            for i in 0..w2 {
                for ch in 0..3 {
                    let mut s = 0.0f32;
                    for dy in 0..f {
                        for dx in 0..f {
                            s += self.gt[((f * j + dy) * w + f * i + dx) * 3 + ch];
                        }
                    }
                    row[i * 3 + ch] = s * inv;
                }
            }
        });
        let rgb: Vec<[f32; 3]> = gt.chunks(3).map(|x| [x[0], x[1], x[2]]).collect();
        let cam = PinholeCamera {
            width: w2 as u32,
            height: h2 as u32,
            fx: c.fx / f as f32,
            fy: c.fy / f as f32,
            cx: c.cx / f as f32,
            cy: c.cy / f as f32,
            ..c
        };
        (
            CameraView {
                camera: cam,
                ..self.view
            },
            gt,
            pack_rgba8(&rgb, None),
        )
    }

    /// The pixels `(2 i + ox, 2 j + oy)` as an image of their own, and a
    /// camera whose pixel centres are those pixels' centres.
    pub fn strided(&self, ox: usize, oy: usize) -> (CameraView, Vec<f32>, Vec<u32>) {
        let c = self.view.camera;
        let (w, h) = (c.width as usize, c.height as usize);
        let (w2, h2) = (w / 2, h / 2);
        let mut gt = vec![0.0f32; w2 * h2 * 3];
        let mut packed = vec![0u32; w2 * h2];
        for j in 0..h2 {
            for i in 0..w2 {
                let src = (2 * j + oy) * w + 2 * i + ox;
                gt[(j * w2 + i) * 3..(j * w2 + i) * 3 + 3]
                    .copy_from_slice(&self.gt[src * 3..src * 3 + 3]);
                packed[j * w2 + i] = self.gt_packed[src];
            }
        }
        // Pixel i's centre i + 0.5 is the full image's 2 i + ox + 0.5.
        let cam = PinholeCamera {
            width: w2 as u32,
            height: h2 as u32,
            fx: 0.5 * c.fx,
            fy: 0.5 * c.fy,
            cx: 0.5 * c.cx - 0.5 * ox as f32 + 0.25,
            cy: 0.5 * c.cy - 0.5 * oy as f32 + 0.25,
            ..c
        };
        (
            CameraView {
                camera: cam,
                ..self.view
            },
            gt,
            packed,
        )
    }

    pub fn new(view: CameraView, rgb: &[[f32; 3]]) -> Self {
        let (w, h) = (view.camera.width as usize, view.camera.height as usize);
        let gt: Vec<f32> = rgb.iter().flatten().copied().collect();
        let (hw, hh) = (w / 2, h / 2);
        let mut half_gt = vec![0.0f32; hw * hh * 3];
        for y in 0..hh {
            for x in 0..hw {
                for ch in 0..3 {
                    let s = |xx: usize, yy: usize| gt[(yy * w + xx) * 3 + ch];
                    half_gt[(y * hw + x) * 3 + ch] = 0.25
                        * (s(2 * x, 2 * y)
                            + s(2 * x + 1, 2 * y)
                            + s(2 * x, 2 * y + 1)
                            + s(2 * x + 1, 2 * y + 1));
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
            gt_packed: pack_rgba8(rgb, None),
            half_view: CameraView {
                camera: half_cam,
                ..view
            },
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
    Rendered {
        proj,
        frame,
        raster,
    }
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
    /// Multiply every sum by `decay` (0 clears them).
    fn decay(&mut self, decay: f32) {
        self.h
            .par_iter_mut()
            .for_each(|x| x.iter_mut().for_each(|v| *v *= decay));
        self.g
            .par_iter_mut()
            .for_each(|x| x.iter_mut().for_each(|v| *v *= decay));
        self.hcol
            .par_iter_mut()
            .for_each(|x| x.iter_mut().for_each(|v| *v *= decay));
        self.gcol.par_iter_mut().for_each(|v| *v *= decay);
    }
}

/// Adam moments for every parameter.
struct AdamState {
    m_geo: Vec<[f32; NGEO]>,
    v_geo: Vec<[f32; NGEO]>,
    m_col: Vec<f32>,
    v_col: Vec<f32>,
    t: i32,
    /// With deferred updates: the step each gaussian was last brought up to.
    last: Vec<i32>,
    /// The latest step's learning rates (for catching up).
    lr_geo: [f32; NGEO],
    lr_dc: f32,
    lr_rest: f32,
}

const B1: f32 = 0.9;
const B2: f32 = 0.999;

/// Moments below 1e-30 are flushed to zero: a gaussian that gets no
/// gradient for a few hundred steps (e.g. culled after an opacity reset)
/// would otherwise decay into subnormals, which x86 computes on a slow path
/// (the GPU flushes them).
#[inline(always)]
fn ftz(x: f32) -> f32 {
    if x.abs() < 1e-30 {
        0.0
    } else {
        x
    }
}

/// `d` Adam steps with a zero gradient, in closed form (GS-Scale's deferred
/// update, Lee et al. 2025): the moments decay by `B1^d` and `B2^d`, and the
/// parameter moves by `lr m / sqrt(v) * sum_k (B1 / sqrt(B2))^k` with the
/// bias corrections and learning rates of the current step and `eps`
/// dropped (they change little over the skipped steps).
#[inline(always)]
fn catch_up(
    p: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: impl Fn(usize) -> f32,
    d: i32,
    bc1: f32,
    bc2: f32,
) {
    if d <= 0 {
        return;
    }
    let r = B1 / B2.sqrt();
    let sum = r * (1.0 - r.powi(d)) / (1.0 - r);
    let k = bc2.sqrt() / bc1 * sum;
    let (f1, f2) = (B1.powi(d), B2.powi(d));
    for i in 0..p.len() {
        if v[i] > 0.0 {
            p[i] -= lr(i) * k * m[i] / v[i].sqrt();
        }
        m[i] = ftz(m[i] * f1);
        v[i] = ftz(v[i] * f2);
    }
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
    /// Which parameters the step moves: mean, scale+rotation, opacity, colour.
    pub groups: [bool; 4],
    /// Step size (1 = the full Newton step).
    pub step: f32,
    /// The sums of earlier iterations are kept, multiplied by this each
    /// iteration, so a step solves the sum of the recent views' quadratic
    /// models rather than the current view's alone (0: 3DGS²'s step).
    pub decay: f32,
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
            groups: [true; 4],
            step: 1.0,
            decay: 0.0,
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
    /// Update only the gaussians projected this step, catching the others up
    /// in closed form when they next are (see [`Trainer::sync`]).
    pub sparse_adam: bool,
    /// D-SSIM weight with the L1 loss (Inria: 0.2); the L1 term gets
    /// `1 - ssim_weight`.
    pub ssim_weight: f32,
    ssim: Option<SsimHost>,
    ssim_size: (usize, usize),
    /// 1: every pixel every step; 2: one pixel of every 2x2 block a step.
    pub stride: usize,
    /// Train on the images box-averaged by this factor (a resolution
    /// schedule sets it per step; 1 = full resolution).
    pub res_factor: usize,
    /// Inria's densification statistic: per gaussian, the summed NDC norm
    /// of its screen-mean gradient, and the views it was projected in.
    grad_accum: Vec<f32>,
    grad_count: Vec<f32>,
    /// Tile intersections and projected splats summed over the rendered views.
    pub isects: u64,
    pub splats: u64,
    /// Seconds spent per stage: project+bin, forward, loss, backward, lift, update.
    pub timing: [f64; 6],
}

impl Trainer {
    pub fn new(model: Model, views: Vec<TrainView>, bg: [f32; 3]) -> Self {
        let centres: Vec<_> = views.iter().map(|v| v.view.camera_center()).collect();
        let mean = centres
            .iter()
            .fold(nalgebra::Vector3::zeros(), |a, c| a + c)
            / centres.len() as f32;
        let extent = 1.1
            * centres
                .iter()
                .map(|c| (c - mean).norm())
                .fold(0.0f32, f32::max);
        let dirs: Vec<_> = views
            .iter()
            .map(|v| v.view.rotation.row(2).transpose())
            .collect();
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
            ssim_weight: 0.0,
            sparse_adam: true,
            ssim: None,
            ssim_size: (0, 0),
            stride: 1,
            res_factor: 1,
            grad_accum: Vec::new(),
            grad_count: Vec::new(),
            timing: [0.0; 6],
            isects: 0,
            splats: 0,
        }
    }

    /// Render, take the loss and run the backward pass of one view.
    fn view_grads(
        &mut self,
        view: &CameraView,
        gt: &[f32],
        gt_packed: Option<&[u32]>,
        weight: f32,
        gauss_newton: bool,
    ) -> (f64, Rendered, Vec<[f32; NB]>) {
        let t0 = Instant::now();
        let proj = project_all(&self.model, view);
        let frame = Frame::new(&proj, view.camera.width, view.camera.height, self.bg);
        let t1 = Instant::now();
        self.isects += frame.sorted.len() as u64;
        self.splats += proj.len() as u64;
        let raster = forward(&frame);
        let t2 = Instant::now();
        let (loss, mut d) = image_loss(&raster.image, gt, self.loss);
        if let (Some(packed), true) = (gt_packed, self.ssim_weight > 0.0 && self.loss == Loss::L1) {
            let lam = self.ssim_weight;
            d.par_iter_mut().for_each(|x| *x *= 1.0 - lam);
            let (w, h) = (view.camera.width as usize, view.camera.height as usize);
            if self.ssim_size != (w, h) {
                self.ssim = Some(SsimHost::new(w, h));
                self.ssim_size = (w, h);
            }
            let ssim = self.ssim.as_mut().expect("just made");
            ssim.add_grad(&raster.image, packed, lam, &mut d);
        }
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
        (
            loss,
            Rendered {
                proj,
                frame,
                raster,
            },
            out,
        )
    }

    /// One Adam step on view `vi`.
    pub fn adam_step(&mut self, vi: usize, step: usize, total: usize, cfg: &AdamConfig) -> f64 {
        let n = self.model.len();
        self.model.active_degree = ((step / cfg.sh_interval) as u32).min(3);
        let (loss, r, out, view) = if self.res_factor > 1 {
            let (view, gt, packed) = self.views[vi].downsampled(self.res_factor);
            let (loss, r, out) = self.view_grads(&view, &gt, Some(&packed), 1.0, false);
            (loss, r, out, view)
        } else if self.stride == 2 {
            // One pixel of every 2x2 block, the offset cycling through the
            // four (Turbo-GS / PennyGS), with Turbo-GS's wider low-pass:
            // 0.3 + 0.5 (p - 1) = 0.8 px² of the full image, 0.2 of this one.
            let (ox, oy) = [(0, 0), (1, 1), (1, 0), (0, 1)][step % 4];
            let (view, gt, packed) = self.views[vi].strided(ox, oy);
            let full_blur = self.model.blur;
            self.model.blur = 0.2;
            let (loss, r, out) = self.view_grads(&view, &gt, Some(&packed), 1.0, false);
            self.model.blur = full_blur;
            (loss, r, out, view)
        } else {
            let view = self.views[vi].view;
            let gt = std::mem::take(&mut self.views[vi].gt);
            let packed = std::mem::take(&mut self.views[vi].gt_packed);
            let (loss, r, out) = self.view_grads(&view, &gt, Some(&packed), 1.0, false);
            self.views[vi].gt = gt;
            self.views[vi].gt_packed = packed;
            (loss, r, out, view)
        };
        // Densification statistics (every projected splat counts a view).
        if self.grad_accum.len() != n {
            self.grad_accum = vec![0.0; n];
            self.grad_count = vec![0.0; n];
        }
        let (hw, hh) = (
            0.5 * view.camera.width as f32,
            0.5 * view.camera.height as f32,
        );
        for (p, o) in r.proj.iter().zip(&out) {
            let (du, dv) = (o[B_GRAD] * hw, o[B_GRAD + 1] * hh);
            self.grad_accum[p.id as usize] += (du * du + dv * dv).sqrt();
            self.grad_count[p.id as usize] += 1.0;
        }
        let t0 = Instant::now();
        let mut idx = vec![u32::MAX; n];
        for (c, p) in r.proj.iter().enumerate() {
            idx[p.id as usize] = c as u32;
        }
        let t1 = Instant::now();
        let st = self.adam.get_or_insert_with(|| AdamState {
            m_geo: vec![[0.0; NGEO]; n],
            v_geo: vec![[0.0; NGEO]; n],
            m_col: vec![0.0; n * NCOL],
            v_col: vec![0.0; n * NCOL],
            t: 0,
            last: vec![0; n],
            lr_geo: [0.0; NGEO],
            lr_dc: 0.0,
            lr_rest: 0.0,
        });
        st.t += 1;
        let (b1, b2, eps) = (B1, B2, 1e-15f32);
        let bc1 = 1.0 - b1.powi(st.t);
        let bc2 = 1.0 - b2.powi(st.t);
        let frac = step as f32 / total.max(1) as f32;
        let lr_mean = self.extent * cfg.lr_mean * (cfg.lr_mean_final / cfg.lr_mean).powf(frac);
        let mut lr_geo = [0.0f32; NGEO];
        lr_geo[0..3].fill(lr_mean);
        lr_geo[3..6].fill(cfg.lr_scale);
        lr_geo[6..10].fill(cfg.lr_quat);
        lr_geo[10] = cfg.lr_opacity;
        let (lr_dc, lr_rest) = (cfg.lr_dc, cfg.lr_rest);
        st.lr_geo = lr_geo;
        st.lr_dc = lr_dc;
        st.lr_rest = lr_rest;
        let t_now = st.t;
        let sparse = self.sparse_adam;
        let blur = if self.stride == 2 {
            0.2
        } else {
            self.model.blur
        };
        let view = &view;
        // One pass per gaussian: its gradient (zero if it was not
        // projected) and the dense Adam update, without per-step gradient
        // arrays.
        self.model
            .geo
            .par_iter_mut()
            .zip(st.m_geo.par_iter_mut().zip(st.v_geo.par_iter_mut()))
            .zip(
                self.model.col.par_chunks_mut(NCOL).zip(
                    st.m_col
                        .par_chunks_mut(NCOL)
                        .zip(st.v_col.par_chunks_mut(NCOL)),
                ),
            )
            .zip(idx.par_iter().zip(st.last.par_iter_mut()))
            .for_each(|(((p, (m, v)), (pc, (mc, vc))), (&c, last))| {
                if sparse {
                    if c == u32::MAX {
                        return;
                    }
                    let d = t_now - 1 - *last;
                    catch_up(p, m, v, |k| lr_geo[k], d, bc1, bc2);
                    catch_up(
                        pc,
                        mc,
                        vc,
                        |i| if i % NSH == 0 { lr_dc } else { lr_rest },
                        d,
                        bc1,
                        bc2,
                    );
                    *last = t_now;
                }
                let mut g = [0.0f32; NGEO];
                let mut gc = [0.0f32; NCOL];
                if c != u32::MAX {
                    let pr = &r.proj[c as usize];
                    let o = &out[c as usize];
                    if o[B_GRAD..B_GRAD + 6].iter().any(|&x| x != 0.0) {
                        let gs = [o[0], o[1], o[2], o[3], o[4], o[5]];
                        if let Some(vj) = screen_vjp_geo(p, view, blur, &gs) {
                            g = vj;
                        }
                    }
                    for ch in 0..3 {
                        if pr.clamped[ch] {
                            continue;
                        }
                        let gv = o[B_COL + ch];
                        for k in 0..NSH {
                            gc[ch * NSH + k] = gv * pr.basis[k];
                        }
                    }
                }
                for k in 0..NGEO {
                    m[k] = ftz(b1 * m[k] + (1.0 - b1) * g[k]);
                    v[k] = ftz(b2 * v[k] + (1.0 - b2) * g[k] * g[k]);
                    p[k] -= lr_geo[k] * (m[k] / bc1) / ((v[k] / bc2).sqrt() + eps);
                }
                for ch in 0..3 {
                    for k in 0..NSH {
                        let i = ch * NSH + k;
                        let lr = if k == 0 { lr_dc } else { lr_rest };
                        mc[i] = ftz(b1 * mc[i] + (1.0 - b1) * gc[i]);
                        vc[i] = ftz(b2 * vc[i] + (1.0 - b2) * gc[i] * gc[i]);
                        pc[i] -= lr * (mc[i] / bc1) / ((vc[i] / bc2).sqrt() + eps);
                    }
                }
            });
        let t2 = Instant::now();
        self.timing[4] += (t1 - t0).as_secs_f64();
        self.timing[5] += (t2 - t1).as_secs_f64();
        loss
    }

    /// Bring every gaussian deferred by the sparse Adam up to the current
    /// step (before rendering for evaluation, saving or densifying).
    pub fn sync(&mut self) {
        let Some(st) = self.adam.as_mut() else { return };
        let t = st.t;
        let bc1 = 1.0 - B1.powi(t);
        let bc2 = 1.0 - B2.powi(t);
        let (lr_geo, lr_dc, lr_rest) = (st.lr_geo, st.lr_dc, st.lr_rest);
        self.model
            .geo
            .par_iter_mut()
            .zip(st.m_geo.par_iter_mut().zip(st.v_geo.par_iter_mut()))
            .zip(
                self.model.col.par_chunks_mut(NCOL).zip(
                    st.m_col
                        .par_chunks_mut(NCOL)
                        .zip(st.v_col.par_chunks_mut(NCOL)),
                ),
            )
            .zip(st.last.par_iter_mut())
            .for_each(|(((p, (m, v)), (pc, (mc, vc))), last)| {
                let d = t - *last;
                catch_up(p, m, v, |k| lr_geo[k], d, bc1, bc2);
                catch_up(
                    pc,
                    mc,
                    vc,
                    |i| if i % NSH == 0 { lr_dc } else { lr_rest },
                    d,
                    bc1,
                    bc2,
                );
                *last = t;
            });
    }

    /// Inria's adaptive density control (`visloc_gsplat_train::densify`, the
    /// same rule `gsplat_train` runs): clone / split / prune with `grow`,
    /// and cap every opacity at 0.01 with `reset`. Adam moments are kept for
    /// kept gaussians and zeroed for new ones; the statistics restart.
    pub fn densify(
        &mut self,
        cfg: &DensifyConfig,
        grow: bool,
        reset: bool,
        step: usize,
        seed: u64,
    ) -> Option<DensifyReport> {
        self.sync();
        let n = self.model.len();
        let st = self.adam.take();
        let zero_geo = vec![[0.0f32; NGEO]; n];
        let zero_col = vec![0.0f32; n * NCOL];
        let (m_geo, v_geo, m_col, v_col) = match &st {
            Some(a) => (&a.m_geo, &a.v_geo, &a.m_col, &a.v_col),
            None => (&zero_geo, &zero_geo, &zero_col, &zero_col),
        };
        // Population transforms: mean | quaternion | log-scale.
        let tr = |g: &[[f32; NGEO]]| -> Vec<f32> {
            g.iter()
                .flat_map(|x| [x[0], x[1], x[2], x[6], x[7], x[8], x[9], x[3], x[4], x[5]])
                .collect()
        };
        let op = |g: &[[f32; NGEO]]| -> Vec<f32> { g.iter().map(|x| x[10]).collect() };
        let mut pop = Population {
            transforms: Group {
                stride: 10,
                values: tr(&self.model.geo),
                m1: tr(m_geo),
                m2: tr(v_geo),
            },
            opacity: Group {
                stride: 1,
                values: op(&self.model.geo),
                m1: op(m_geo),
                m2: op(v_geo),
            },
            sh: Group {
                stride: NCOL,
                values: self.model.col.clone(),
                m1: m_col.clone(),
                m2: v_col.clone(),
            },
        };
        let mut report = None;
        // At the cap gsplat_train neither grows nor prunes (its classify
        // pass prunes only when it may grow); so here.
        let grow = grow && n < cfg.max_gaussians;
        if grow {
            if self.grad_accum.len() != n {
                self.grad_accum = vec![0.0; n];
                self.grad_count = vec![0.0; n];
            }
            let prune_large = step > cfg.opacity_reset_interval;
            let (p, rep) = densify::densify(
                &pop,
                &self.grad_accum,
                &self.grad_count,
                self.extent,
                cfg,
                prune_large,
                seed,
            );
            pop = p;
            report = Some(rep);
        }
        if reset {
            densify::reset_opacity(&mut pop, 0.01);
        }
        let n2 = pop.len();
        let back = |t: &[f32], o: &[f32]| -> Vec<[f32; NGEO]> {
            (0..n2)
                .map(|i| {
                    let r = &t[i * 10..i * 10 + 10];
                    [
                        r[0], r[1], r[2], r[7], r[8], r[9], r[3], r[4], r[5], r[6], o[i],
                    ]
                })
                .collect()
        };
        self.model.geo = back(&pop.transforms.values, &pop.opacity.values);
        self.model.col = pop.sh.values;
        if let Some(mut a) = st {
            a.m_geo = back(&pop.transforms.m1, &pop.opacity.m1);
            a.v_geo = back(&pop.transforms.m2, &pop.opacity.m2);
            a.m_col = pop.sh.m1;
            a.v_col = pop.sh.m2;
            a.last = vec![a.t; n2];
            self.adam = Some(a);
        }
        if grow {
            self.grad_accum = vec![0.0; n2];
            self.grad_count = vec![0.0; n2];
        }
        self.acc = None;
        report
    }

    /// Add one view's gradients and Gauss-Newton matrices into `acc`.
    fn lift(
        acc: &mut Accum,
        model: &Model,
        view: &CameraView,
        r: &Rendered,
        out: &[[f32; NB]],
        n: usize,
    ) {
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
        acc.decay(cfg.decay);
        let mut primary_loss = 0.0;
        let nb: Vec<usize> = self.neighbours[vi]
            .iter()
            .take(cfg.neighbours)
            .copied()
            .collect();
        for (k, v) in std::iter::once(vi).chain(nb).enumerate() {
            let (view, gt, w) = if k == 0 {
                (
                    self.views[v].view,
                    std::mem::take(&mut self.views[v].gt),
                    1.0,
                )
            } else {
                (
                    self.views[v].half_view,
                    std::mem::take(&mut self.views[v].half_gt),
                    cfg.neighbour_weight,
                )
            };
            let (loss, r, out) = self.view_grads(&view, &gt, None, w, true);
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
            .zip(acc.h.par_iter().zip(acc.g.par_iter_mut()))
            .zip(acc.hcol.par_iter().zip(acc.gcol.par_chunks_mut(NCOL)))
            .for_each(|(((p, col), (h, g)), (hc, gc))| {
                let before = *p;
                solve_geometry(p, h, g, lambda, cfg);
                if cfg.decay > 0.0 {
                    // Keep the summed model's gradient at the new point:
                    // g += H dx (the quaternion renormalisation aside).
                    let mut dx = [0.0f32; NGEO];
                    for i in 0..NGEO {
                        dx[i] = p[i] - before[i];
                    }
                    for i in 0..NGEO {
                        let mut s = 0.0f32;
                        for j in 0..NGEO {
                            s += h[tri11(i.min(j), i.max(j))] * dx[j];
                        }
                        g[i] += s;
                    }
                }
                if !cfg.groups[3] {
                    return;
                }
                for ch in 0..3 {
                    for k in 0..NSH {
                        let hh = hc[k] as f64;
                        if hh <= 0.0 {
                            continue;
                        }
                        let step = -(gc[ch * NSH + k] as f64) / (hh * (1.0 + lambda) + 1e-12);
                        let d = cfg.step * (step as f32).clamp(-cfg.max_dcol, cfg.max_dcol);
                        col[ch * NSH + k] += d;
                        if cfg.decay > 0.0 {
                            gc[ch * NSH + k] += hc[k] * d;
                        }
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
fn solve_geometry(
    p: &mut [f32; NGEO],
    h: &[f32; NTRI],
    g: &[f32; NGEO],
    lambda: f64,
    cfg: &NewtonConfig,
) {
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
    let mut g = *g;
    for (i, row) in a.iter_mut().enumerate() {
        row[i] = row[i] * (1.0 + lambda) + floor;
    }
    let group = |i: usize| match i {
        0..=2 => 0,
        3..=9 => 1,
        _ => 2,
    };
    for i in 0..NGEO {
        if !cfg.groups[group(i)] {
            for j in 0..NGEO {
                a[i][j] = 0.0;
                a[j][i] = 0.0;
            }
            a[i][i] = 1.0;
            g[i] = 0.0;
        }
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
    let qn = ((p[6] * p[6] + p[7] * p[7] + p[8] * p[8] + p[9] * p[9]) as f64)
        .sqrt()
        .max(1e-12);
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
    let s = s * cfg.step as f64;
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
) -> (
    f64,
    Vec<[f32; NTRI]>,
    Vec<[f32; NGEO]>,
    Vec<[f32; NSH]>,
    Vec<f32>,
) {
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
