//! Projection of every gaussian into one pinhole view, with the Jacobian of
//! its screen quantities (`u, v`, conic `A B C`, opacity) with respect to
//! its 11 geometry parameters (mean 3, log-scale 3, quaternion 4, opacity
//! logit 1), by forward-mode dual numbers in `f32`.
//!
//! The maths and gates are `visloc_gsplat_core::backward::project`'s (the
//! f64 oracle of the wgpu and CUDA renderers) for a pinhole lens: the EWA
//! Jacobian linearised at a point clamped to 1.3x the frustum, a 0.3 px²
//! blur, the footprint extent at `alpha = 1/255`. The view direction of the
//! SH colour is held fixed (no colour-to-mean derivative), in both
//! optimizers.

use std::ops::{Add, Div, Mul, Neg, Sub};

use rayon::prelude::*;
use visloc_gsplat_core::camera::CameraView;
use visloc_gsplat_core::sh::{SH_BASIS_COEFFS, SH_C0};

use crate::model::{Model, NGEO, NSH};

/// Forward-mode dual over the 11 geometry parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct D {
    pub v: f32,
    pub d: [f32; NGEO],
}

impl D {
    #[inline(always)]
    fn c(v: f32) -> Self {
        Self { v, d: [0.0; NGEO] }
    }
    #[inline(always)]
    fn var(v: f32, i: usize) -> Self {
        let mut d = [0.0; NGEO];
        d[i] = 1.0;
        Self { v, d }
    }
    #[inline(always)]
    fn map(self, v: f32, dv: f32) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= dv;
        }
        Self { v, d }
    }
    #[inline(always)]
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        let r = self.v.sqrt();
        self.map(r, 0.5 / r)
    }
    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> Self {
        if self.v < lo {
            D::c(lo)
        } else if self.v > hi {
            D::c(hi)
        } else {
            self
        }
    }
}

impl Add for D {
    type Output = D;
    #[inline(always)]
    fn add(self, o: D) -> D {
        let mut d = self.d;
        for (a, b) in d.iter_mut().zip(o.d) {
            *a += b;
        }
        D { v: self.v + o.v, d }
    }
}
impl Sub for D {
    type Output = D;
    #[inline(always)]
    fn sub(self, o: D) -> D {
        let mut d = self.d;
        for (a, b) in d.iter_mut().zip(o.d) {
            *a -= b;
        }
        D { v: self.v - o.v, d }
    }
}
impl Neg for D {
    type Output = D;
    #[inline(always)]
    fn neg(self) -> D {
        self.map(-self.v, -1.0)
    }
}
impl Mul for D {
    type Output = D;
    #[inline(always)]
    fn mul(self, o: D) -> D {
        let mut d = [0.0; NGEO];
        for (i, x) in d.iter_mut().enumerate() {
            *x = self.d[i] * o.v + self.v * o.d[i];
        }
        D { v: self.v * o.v, d }
    }
}
impl Div for D {
    type Output = D;
    #[inline(always)]
    fn div(self, o: D) -> D {
        let inv = 1.0 / o.v;
        let mut d = [0.0; NGEO];
        for (i, x) in d.iter_mut().enumerate() {
            *x = (self.d[i] * o.v - self.v * o.d[i]) * inv * inv;
        }
        D { v: self.v * inv, d }
    }
}
impl Mul<f32> for D {
    type Output = D;
    #[inline(always)]
    fn mul(self, s: f32) -> D {
        self.map(self.v * s, s)
    }
}
impl Add<f32> for D {
    type Output = D;
    #[inline(always)]
    fn add(self, s: f32) -> D {
        D {
            v: self.v + s,
            d: self.d,
        }
    }
}

/// The scalar the projection is written over: `f32` for values, [`D`] for
/// values and the Jacobian.
pub(crate) trait S:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
    + Mul<f32, Output = Self> + Add<f32, Output = Self>
{
    fn c(v: f32) -> Self;
    fn var(v: f32, i: usize) -> Self;
    fn v(self) -> f32;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn clamp(self, lo: f32, hi: f32) -> Self;
}

impl S for f32 {
    #[inline(always)]
    fn c(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn var(v: f32, _: usize) -> Self {
        v
    }
    #[inline(always)]
    fn v(self) -> f32 {
        self
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> Self {
        f32::clamp(self, lo, hi)
    }
}

impl S for D {
    #[inline(always)]
    fn c(v: f32) -> Self {
        D::c(v)
    }
    #[inline(always)]
    fn var(v: f32, i: usize) -> Self {
        D::var(v, i)
    }
    #[inline(always)]
    fn v(self) -> f32 {
        self.v
    }
    #[inline(always)]
    fn exp(self) -> Self {
        D::exp(self)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        D::sqrt(self)
    }
    #[inline(always)]
    fn clamp(self, lo: f32, hi: f32) -> Self {
        D::clamp(self, lo, hi)
    }
}

/// One visible gaussian in one view.
#[derive(Clone, Debug)]
pub struct Proj {
    /// Gaussian index in the model.
    pub id: u32,
    pub depth: f32,
    /// `u, v, A, B, C, opacity, r, g, b` (colour after the `max(0)` clamp).
    pub s: [f32; 9],
    /// SH basis of the view direction (zero past the active degree).
    pub basis: [f32; NSH],
    /// Channels clamped by `max(0)` (no colour gradient).
    pub clamped: [bool; 3],
    pub ext: [f32; 2],
}

/// SH basis (degree 0..=3) of a unit direction, in the order of
/// `visloc_gsplat_core::sh`; entries past `degree` are zero.
pub fn sh_basis(degree: u32, x: f32, y: f32, z: f32) -> [f32; NSH] {
    let c = |i: usize| SH_BASIS_COEFFS[i];
    let mut b = [0.0f32; NSH];
    b[0] = SH_C0;
    if degree >= 1 {
        b[1] = -c(0) * y;
        b[2] = c(1) * z;
        b[3] = -c(2) * x;
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b[4] = c(3) * x * y;
        b[5] = -c(4) * y * z;
        b[6] = c(5) * (2.0 * zz - xx - yy);
        b[7] = -c(6) * x * z;
        b[8] = c(7) * (xx - yy);
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b[9] = -c(8) * y * (3.0 * xx - yy);
        b[10] = c(9) * x * y * z;
        b[11] = -c(10) * y * (4.0 * zz - xx - yy);
        b[12] = c(11) * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
        b[13] = -c(12) * x * (4.0 * zz - xx - yy);
        b[14] = c(13) * z * (xx - yy);
        b[15] = -c(14) * x * (xx - 3.0 * yy);
    }
    b
}

/// The screen quantities `u, v, A, B, C, opacity` of gaussian `id` over
/// scalar `T`, with the depth and the footprint extent, or `None` if it is
/// culled (behind the near plane, too transparent, degenerate, off screen).
#[inline(always)]
fn screen<T: S>(model: &Model, id: usize, view: &CameraView) -> Option<([T; 6], f32, [f32; 2])> {
    let g = &model.geo[id];
    let cam = &view.camera;
    let (w, h) = (cam.width as f32, cam.height as f32);
    let (fx, fy, cx, cy) = (cam.fx, cam.fy, cam.cx, cam.cy);

    let logit = g[10];
    let opac_v = 1.0 / (1.0 + (-logit).exp());
    if opac_v < 1.0 / 255.0 {
        return None;
    }
    let r = view.rotation;
    let t = view.translation;
    let pz = r[(2, 0)] * g[0] + r[(2, 1)] * g[1] + r[(2, 2)] * g[2] + t[2];
    if pz <= 0.1 {
        return None;
    }
    // A cheap frustum test with a bound on the footprint (3 sigma of the
    // largest scale, plus the 0.3 px blur).
    {
        let px = r[(0, 0)] * g[0] + r[(0, 1)] * g[1] + r[(0, 2)] * g[2] + t[0];
        let py = r[(1, 0)] * g[0] + r[(1, 1)] * g[1] + r[(1, 2)] * g[2] + t[1];
        let smax = g[3].max(g[4]).max(g[5]).exp();
        let rad = 3.5 * smax / pz * fx.max(fy) + 2.0;
        let u = px / pz * fx + cx;
        let v = py / pz * fy + cy;
        if u + rad < -0.5 * w || v + rad < -0.5 * h || u - rad > 1.5 * w || v - rad > 1.5 * h {
            return None;
        }
    }

    let mean = [T::var(g[0], 0), T::var(g[1], 1), T::var(g[2], 2)];
    let log_s = [T::var(g[3], 3), T::var(g[4], 4), T::var(g[5], 5)];
    let q = [
        T::var(g[6], 6),
        T::var(g[7], 7),
        T::var(g[8], 8),
        T::var(g[9], 9),
    ];
    let one = T::c(1.0);
    let opacity = one / (one + (-T::var(logit, 10)).exp());

    let p_cam: [T; 3] = std::array::from_fn(|i| {
        mean[0] * r[(i, 0)] + mean[1] * r[(i, 1)] + mean[2] * r[(i, 2)] + t[i]
    });
    let depth = p_cam[2].v();

    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2.v().is_nan() || n2.v() <= f32::EPSILON * f32::EPSILON {
        return None;
    }
    let n = n2.sqrt();
    let (qw, qx, qy, qz) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let rg = [
        [
            one - (qy * qy + qz * qz) * 2.0,
            (qx * qy - qw * qz) * 2.0,
            (qx * qz + qw * qy) * 2.0,
        ],
        [
            (qx * qy + qw * qz) * 2.0,
            one - (qx * qx + qz * qz) * 2.0,
            (qy * qz - qw * qx) * 2.0,
        ],
        [
            (qx * qz - qw * qy) * 2.0,
            (qy * qz + qw * qx) * 2.0,
            one - (qx * qx + qy * qy) * 2.0,
        ],
    ];
    let s2 = log_s.map(|l| (l * 2.0).exp());
    // M = W Rg (camera-from-local); cov_cam = M diag(s2) M^T.
    let m: [[T; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|k| rg[0][k] * r[(i, 0)] + rg[1][k] * r[(i, 1)] + rg[2][k] * r[(i, 2)])
    });

    let z = p_cam[2];
    let inv_z = one / z;
    let inv_z2 = inv_z * inv_z;
    let lim_x = 1.3 * 0.5 * w / fx;
    let lim_y = 1.3 * 0.5 * h / fy;
    let tx = (p_cam[0] * inv_z).clamp(-lim_x, lim_x) * z;
    let ty = (p_cam[1] * inv_z).clamp(-lim_y, lim_y) * z;
    let j00 = inv_z * fx;
    let j02 = -(tx * inv_z2) * fx;
    let j11 = inv_z * fy;
    let j12 = -(ty * inv_z2) * fy;
    let u = p_cam[0] * inv_z * fx + cx;
    let v = p_cam[1] * inv_z * fy + cy;
    // Rows of J M: the 2D covariance is sum_k s2_k (JM)_ak (JM)_bk.
    let jm0: [T; 3] = std::array::from_fn(|k| j00 * m[0][k] + j02 * m[2][k]);
    let jm1: [T; 3] = std::array::from_fn(|k| j11 * m[1][k] + j12 * m[2][k]);
    let mut a = T::c(0.3);
    let mut b = T::c(0.0);
    let mut c = T::c(0.3);
    for k in 0..3 {
        a = a + jm0[k] * jm0[k] * s2[k];
        b = b + jm0[k] * jm1[k] * s2[k];
        c = c + jm1[k] * jm1[k] * s2[k];
    }
    let det = a * c - b * b;
    if det.v().is_nan() || det.v() <= 0.0 {
        return None;
    }
    let conic = [c / det, -b / det, a / det];

    let kk = 2.0 * (255.0 * opac_v).ln();
    if !kk.is_finite() || kk <= 0.0 {
        return None;
    }
    let ext_x = (kk * a.v()).sqrt();
    let ext_y = (kk * c.v()).sqrt();
    let (uv, vv) = (u.v(), v.v());
    if uv + ext_x < 0.0 || vv + ext_y < 0.0 || uv - ext_x > w - 1.0 || vv - ext_y > h - 1.0 {
        return None;
    }
    Some(([u, v, conic[0], conic[1], conic[2], opacity], depth, [ext_x, ext_y]))
}

/// Project gaussian `id` (values only), or `None` if it is culled.
pub fn project_one(model: &Model, id: usize, view: &CameraView) -> Option<Proj> {
    let (sc, depth, ext) = screen::<f32>(model, id, view)?;
    let g = &model.geo[id];
    // Colour along the (fixed) view direction.
    let cc = view.camera_center();
    let d = [g[0] - cc.x, g[1] - cc.y, g[2] - cc.z];
    let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let dir = if dn > f32::EPSILON {
        [d[0] / dn, d[1] / dn, d[2] / dn]
    } else {
        [0.0, 0.0, 1.0]
    };
    let basis = sh_basis(model.active_degree, dir[0], dir[1], dir[2]);
    let col = model.color(id);
    let mut rgb = [0.0f32; 3];
    let mut clamped = [false; 3];
    for ch in 0..3 {
        let mut acc = 0.5;
        for (k, bk) in basis.iter().enumerate() {
            acc += bk * col[ch * NSH + k];
        }
        if acc < 0.0 {
            clamped[ch] = true;
        } else {
            rgb[ch] = acc;
        }
    }
    Some(Proj {
        id: id as u32,
        depth,
        s: [sc[0], sc[1], sc[2], sc[3], sc[4], sc[5], rgb[0], rgb[1], rgb[2]],
        basis,
        clamped,
        ext,
    })
}

/// `d (u, v, A, B, C, opacity) / d geometry` of gaussian `id` (row-major
/// 6 x 11), by forward-mode duals; `None` if it is culled.
pub fn jacobian(model: &Model, id: usize, view: &CameraView) -> Option<[[f32; NGEO]; 6]> {
    let (sc, _, _) = screen::<D>(model, id, view)?;
    Some(sc.map(|x| x.d))
}

/// Every visible gaussian of `model` in `view`, sorted front to back (by
/// depth, ties by id).
pub fn project_all(model: &Model, view: &CameraView) -> Vec<Proj> {
    let mut out: Vec<Proj> = (0..model.len())
        .into_par_iter()
        .filter_map(|i| project_one(model, i, view))
        .collect();
    out.par_sort_unstable_by(|a, b| {
        a.depth
            .total_cmp(&b.depth)
            .then(a.id.cmp(&b.id))
    });
    out
}

/// `J^T g` for `g = dL / d(u, v, A, B, C, opacity)` of gaussian `id`: the
/// gradient of its 11 geometry parameters, by a hand-written reverse pass
/// (what [`jacobian`] gives, at a fraction of the cost); `None` if culled.
pub fn screen_vjp(model: &Model, id: usize, view: &CameraView, gs: &[f32; 6]) -> Option<[f32; NGEO]> {
    screen_vjp_geo(&model.geo[id], view, gs)
}

/// [`screen_vjp`] of the geometry parameters `g` themselves.
pub fn screen_vjp_geo(g: &[f32; NGEO], view: &CameraView, gs: &[f32; 6]) -> Option<[f32; NGEO]> {
    // Forward, keeping what the reverse pass needs; the culls are
    // `screen`'s, which the caller already passed.
    let cam = &view.camera;
    let (w, h) = (cam.width as f32, cam.height as f32);
    let (fx, fy) = (cam.fx, cam.fy);
    let r = view.rotation;
    let t = view.translation;
    let o = 1.0 / (1.0 + (-g[10]).exp());
    let p: [f32; 3] = std::array::from_fn(|i| r[(i, 0)] * g[0] + r[(i, 1)] * g[1] + r[(i, 2)] * g[2] + t[i]);
    let qr = [g[6], g[7], g[8], g[9]];
    let n2 = qr.iter().map(|x| x * x).sum::<f32>();
    if n2.is_nan() || n2 <= f32::EPSILON * f32::EPSILON || p[2] <= 0.1 {
        return None;
    }
    let n = n2.sqrt();
    let (qw, qx, qy, qz) = (qr[0] / n, qr[1] / n, qr[2] / n, qr[3] / n);
    let rg = [
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ];
    let s2 = [(2.0 * g[3]).exp(), (2.0 * g[4]).exp(), (2.0 * g[5]).exp()];
    let m: [[f32; 3]; 3] =
        std::array::from_fn(|i| std::array::from_fn(|k| rg[0][k] * r[(i, 0)] + rg[1][k] * r[(i, 1)] + rg[2][k] * r[(i, 2)]));
    let z = p[2];
    let iz = 1.0 / z;
    let iz2 = iz * iz;
    let lim_x = 1.3 * 0.5 * w / fx;
    let lim_y = 1.3 * 0.5 * h / fy;
    let xr = p[0] * iz;
    let yr = p[1] * iz;
    let x_free = (-lim_x..=lim_x).contains(&xr);
    let y_free = (-lim_y..=lim_y).contains(&yr);
    let tx = xr.clamp(-lim_x, lim_x) * z;
    let ty = yr.clamp(-lim_y, lim_y) * z;
    let j00 = fx * iz;
    let j02 = -fx * tx * iz2;
    let j11 = fy * iz;
    let j12 = -fy * ty * iz2;
    let jm0: [f32; 3] = std::array::from_fn(|k| j00 * m[0][k] + j02 * m[2][k]);
    let jm1: [f32; 3] = std::array::from_fn(|k| j11 * m[1][k] + j12 * m[2][k]);
    let mut a = 0.3f32;
    let mut b = 0.0f32;
    let mut c = 0.3f32;
    for k in 0..3 {
        a += s2[k] * jm0[k] * jm0[k];
        b += s2[k] * jm0[k] * jm1[k];
        c += s2[k] * jm1[k] * jm1[k];
    }
    let det = a * c - b * b;
    if det.is_nan() || det <= 0.0 {
        return None;
    }
    let id_ = 1.0 / det;

    // Reverse.
    let [gu, gv, g_a, g_b, g_c, go] = *gs;
    let mut out = [0.0f32; NGEO];
    out[10] = go * o * (1.0 - o);
    let gi = g_a * c - g_b * b + g_c * a;
    let gdet = -gi * id_ * id_;
    let ga = g_c * id_ + gdet * c;
    let gb = -g_b * id_ - 2.0 * gdet * b;
    let gc = g_a * id_ + gdet * a;
    let mut g_jm0 = [0.0f32; 3];
    let mut g_jm1 = [0.0f32; 3];
    for k in 0..3 {
        let gs2 = ga * jm0[k] * jm0[k] + gb * jm0[k] * jm1[k] + gc * jm1[k] * jm1[k];
        out[3 + k] = gs2 * 2.0 * s2[k];
        g_jm0[k] = s2[k] * (2.0 * ga * jm0[k] + gb * jm1[k]);
        g_jm1[k] = s2[k] * (2.0 * gc * jm1[k] + gb * jm0[k]);
    }
    let mut g_j00 = 0.0;
    let mut g_j02 = 0.0;
    let mut g_j11 = 0.0;
    let mut g_j12 = 0.0;
    let mut gm = [[0.0f32; 3]; 3];
    for k in 0..3 {
        g_j00 += g_jm0[k] * m[0][k];
        g_j02 += g_jm0[k] * m[2][k];
        g_j11 += g_jm1[k] * m[1][k];
        g_j12 += g_jm1[k] * m[2][k];
        gm[0][k] = g_jm0[k] * j00;
        gm[1][k] = g_jm1[k] * j11;
        gm[2][k] = g_jm0[k] * j02 + g_jm1[k] * j12;
    }
    // Camera-space mean.
    let mut gp = [0.0f32; 3];
    gp[0] += gu * fx * iz;
    gp[1] += gv * fy * iz;
    gp[2] += -(gu * fx * p[0] + gv * fy * p[1]) * iz2;
    gp[2] += -(g_j00 * fx + g_j11 * fy) * iz2;
    if x_free {
        gp[0] += -g_j02 * fx * iz2;
        gp[2] += g_j02 * 2.0 * fx * p[0] * iz2 * iz;
    } else {
        gp[2] += g_j02 * fx * xr.clamp(-lim_x, lim_x) * iz2;
    }
    if y_free {
        gp[1] += -g_j12 * fy * iz2;
        gp[2] += g_j12 * 2.0 * fy * p[1] * iz2 * iz;
    } else {
        gp[2] += g_j12 * fy * yr.clamp(-lim_y, lim_y) * iz2;
    }
    for j in 0..3 {
        out[j] = r[(0, j)] * gp[0] + r[(1, j)] * gp[1] + r[(2, j)] * gp[2];
    }
    // Rotation: gRg = R^T gM, then through the normalised quaternion.
    let grg: [[f32; 3]; 3] =
        std::array::from_fn(|l| std::array::from_fn(|k| r[(0, l)] * gm[0][k] + r[(1, l)] * gm[1][k] + r[(2, l)] * gm[2][k]));
    let gq = [
        2.0 * (-qz * grg[0][1] + qy * grg[0][2] + qz * grg[1][0] - qx * grg[1][2] - qy * grg[2][0] + qx * grg[2][1]),
        2.0 * (qy * grg[0][1] + qz * grg[0][2] + qy * grg[1][0] - 2.0 * qx * grg[1][1] - qw * grg[1][2] + qz * grg[2][0]
            + qw * grg[2][1]
            - 2.0 * qx * grg[2][2]),
        2.0 * (-2.0 * qy * grg[0][0] + qx * grg[0][1] + qw * grg[0][2] + qx * grg[1][0] + qz * grg[1][2] - qw * grg[2][0]
            + qz * grg[2][1]
            - 2.0 * qy * grg[2][2]),
        2.0 * (-2.0 * qz * grg[0][0] - qw * grg[0][1] + qx * grg[0][2] + qw * grg[1][0] - 2.0 * qz * grg[1][1]
            + qy * grg[1][2]
            + qx * grg[2][0]
            + qy * grg[2][1]),
    ];
    let qn = [qw, qx, qy, qz];
    let dot = gq.iter().zip(&qn).map(|(a, b)| a * b).sum::<f32>();
    for i in 0..4 {
        out[6 + i] = (gq[i] - qn[i] * dot) / n;
    }
    Some(out)
}
