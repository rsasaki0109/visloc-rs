//! CPU reference gradients of the forward rasterizer (the oracle for the GPU
//! backward kernels).
//!
//! [`render_f64`] is the forward pass of [`crate::cpu_render::render`] in `f64`
//! with the same gates (opacity, near plane, footprint extent, `alpha >= 1/255`,
//! `alpha <= 0.99`, `T > 1e-4`). [`render_backward`] returns the gradient of
//! `L = sum_px dot(d_image[px], C[px])` with respect to every gaussian
//! parameter:
//!
//! - Compositing is differentiated analytically per pixel, back to front, from
//!   the recorded list of contributing splats (`dC/dalpha_i = T_i (c_i - S_i)`
//!   with `S_i` the colour composited behind splat `i`).
//! - Projection (mean, log-scale, quaternion, opacity logit -> screen mean,
//!   conic, opacity, colour) is differentiated with forward-mode dual numbers,
//!   so the chain through the EWA Jacobian, the frustum clamp, quaternion
//!   normalisation and the SH view direction is exact rather than hand-derived.
//! - SH coefficients enter the colour linearly: their gradient is the basis.
//!
//! Everything is `f64` so central finite differences can check it tightly.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::camera::CameraView;
use crate::gaussian::{Gaussian, Scene};
use crate::lens::{extended_lens_coords, lens_coords, ray_angle, LensScalar, Projection};
use crate::sh::{SH_BASIS_COEFFS, SH_C0};

/// Number of non-SH parameters per gaussian differentiated with duals:
/// mean (3), log-scale (3), quaternion (4, un-normalised), opacity logit (1).
const NP: usize = 11;

/// Forward-mode dual number over the [`NP`] projection parameters.
#[derive(Clone, Copy, Debug)]
struct D {
    v: f64,
    d: [f64; NP],
}

impl D {
    fn c(v: f64) -> Self {
        Self { v, d: [0.0; NP] }
    }
    fn var(v: f64, i: usize) -> Self {
        let mut d = [0.0; NP];
        d[i] = 1.0;
        Self { v, d }
    }
    fn map(self, v: f64, dv: f64) -> Self {
        let mut d = self.d;
        for x in &mut d {
            *x *= dv;
        }
        Self { v, d }
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }
    fn sqrt(self) -> Self {
        let r = self.v.sqrt();
        self.map(r, 0.5 / r)
    }
    /// `clamp` whose gradient is zero on the clamped side (as in the kernels).
    fn clamp(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            D::c(lo)
        } else if self.v > hi {
            D::c(hi)
        } else {
            self
        }
    }
}

impl LensScalar for D {
    fn from_f64(v: f64) -> Self {
        D::c(v)
    }
    fn value(self) -> f64 {
        self.v
    }
    fn sqrt(self) -> Self {
        D::sqrt(self)
    }
    fn atan2(self, x: Self) -> Self {
        let inv = 1.0 / (x.v * x.v + self.v * self.v);
        let mut d = [0.0; NP];
        for (i, out) in d.iter_mut().enumerate() {
            *out = (x.v * self.d[i] - self.v * x.d[i]) * inv;
        }
        D {
            v: self.v.atan2(x.v),
            d,
        }
    }
}

impl Add for D {
    type Output = D;
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
    fn sub(self, o: D) -> D {
        self + (-o)
    }
}
impl Neg for D {
    type Output = D;
    fn neg(self) -> D {
        self.map(-self.v, -1.0)
    }
}
impl Mul for D {
    type Output = D;
    fn mul(self, o: D) -> D {
        let mut d = [0.0; NP];
        for (i, x) in d.iter_mut().enumerate() {
            *x = self.d[i] * o.v + self.v * o.d[i];
        }
        D { v: self.v * o.v, d }
    }
}
impl Div for D {
    type Output = D;
    fn div(self, o: D) -> D {
        let inv = 1.0 / o.v;
        let mut d = [0.0; NP];
        for (i, x) in d.iter_mut().enumerate() {
            *x = (self.d[i] * o.v - self.v * o.d[i]) * inv * inv;
        }
        D { v: self.v * inv, d }
    }
}
impl Mul<f64> for D {
    type Output = D;
    fn mul(self, s: f64) -> D {
        self.map(self.v * s, s)
    }
}
impl Add<f64> for D {
    type Output = D;
    fn add(self, s: f64) -> D {
        D {
            v: self.v + s,
            d: self.d,
        }
    }
}

/// SH basis (degree 0..=3) over duals; same order and signs as
/// [`crate::sh::eval_sh_basis`].
fn sh_basis(degree: u32, x: D, y: D, z: D) -> Vec<D> {
    let c = |i: usize| SH_BASIS_COEFFS[i] as f64;
    let mut out = vec![D::c(SH_C0 as f64)];
    if degree >= 1 {
        out.push(y * -c(0));
        out.push(z * c(1));
        out.push(x * -c(2));
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        out.push(x * y * c(3));
        out.push(y * z * -c(4));
        out.push((zz * 2.0 - xx - yy) * c(5));
        out.push(x * z * -c(6));
        out.push((xx - yy) * c(7));
    }
    if degree >= 3 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        out.push(y * (xx * 3.0 - yy) * -c(8));
        out.push(x * y * z * c(9));
        out.push(y * (zz * 4.0 - xx - yy) * -c(10));
        out.push(z * (zz * 2.0 - xx * 3.0 - yy * 3.0) * c(11));
        out.push(x * (zz * 4.0 - xx - yy) * -c(12));
        out.push(z * (xx - yy) * c(13));
        out.push(x * (xx - yy * 3.0) * -c(14));
    }
    out
}

/// A gaussian projected to the screen, with derivatives w.r.t. its [`NP`]
/// projection parameters.
struct Proj {
    /// Index into `scene.gaussians`.
    index: usize,
    depth: f64,
    u: D,
    v: D,
    /// Conic (inverse 2D covariance) `[A, B, C]`: sigma = 0.5 (A dx^2 + C dy^2) + B dx dy.
    conic: [D; 3],
    opacity: D,
    /// Colour after the `max(0)` clamp.
    color: [D; 3],
    /// Whether each channel was clamped by `max(0)` (zero gradient).
    color_clamped: [bool; 3],
    /// SH basis values (for the coefficient gradients).
    basis: Vec<f64>,
    ext_x: f64,
    ext_y: f64,
}

fn project(index: usize, g: &Gaussian, view: &CameraView, sh_degree: u32) -> Option<Proj> {
    let w = view.camera.width as f64;
    let h = view.camera.height as f64;
    let fx = view.camera.fx as f64;
    let fy = view.camera.fy as f64;
    let cx = view.camera.cx as f64;
    let cy = view.camera.cy as f64;

    let mean = [
        D::var(g.mean.x as f64, 0),
        D::var(g.mean.y as f64, 1),
        D::var(g.mean.z as f64, 2),
    ];
    let log_s = [
        D::var(g.scale_log.x as f64, 3),
        D::var(g.scale_log.y as f64, 4),
        D::var(g.scale_log.z as f64, 5),
    ];
    let q = [
        D::var(g.rotation.w as f64, 6),
        D::var(g.rotation.i as f64, 7),
        D::var(g.rotation.j as f64, 8),
        D::var(g.rotation.k as f64, 9),
    ];
    let logit = D::var(g.opacity_logit as f64, 10);

    // opacity = sigmoid(logit)
    let one = D::c(1.0);
    let opacity = one / (one + (-logit).exp());
    if opacity.v < 1e-4 {
        return None;
    }

    // Camera-space mean.
    let r = view.rotation;
    let rv = |i: usize, j: usize| r[(i, j)] as f64;
    let t = view.translation;
    let p_cam: [D; 3] = std::array::from_fn(|i| {
        mean[0] * rv(i, 0) + mean[1] * rv(i, 1) + mean[2] * rv(i, 2) + t[i] as f64
    });
    // The same gates and sort depth as `cpu_render::project_point`.
    let lens = view.camera.lens;
    let p_val = p_cam.map(|x| x.v);
    let depth = if lens.projection == Projection::Equidistant && !lens.is_pinhole() {
        let distance = (p_val[0] * p_val[0] + p_val[1] * p_val[1] + p_val[2] * p_val[2]).sqrt();
        let widest = (lens.max_theta as f64).max(std::f64::consts::FRAC_PI_2);
        if distance <= 0.1 || ray_angle(p_val) > widest {
            return None;
        }
        distance
    } else {
        if p_val[2] <= 0.1 {
            return None;
        }
        p_val[2]
    };

    // Rotation from the normalised quaternion (w, x, y, z).
    let n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if n2.v.is_nan() || n2.v <= (f32::EPSILON as f64).powi(2) {
        return None;
    }
    let n = n2.sqrt();
    let (qw, qx, qy, qz) = (q[0] / n, q[1] / n, q[2] / n, q[3] / n);
    let two = 2.0;
    let rg = [
        [
            one - (qy * qy + qz * qz) * two,
            (qx * qy - qw * qz) * two,
            (qx * qz + qw * qy) * two,
        ],
        [
            (qx * qy + qw * qz) * two,
            one - (qx * qx + qz * qz) * two,
            (qy * qz - qw * qx) * two,
        ],
        [
            (qx * qz - qw * qy) * two,
            (qy * qz + qw * qx) * two,
            one - (qx * qx + qy * qy) * two,
        ],
    ];
    let s2 = log_s.map(|l| (l * 2.0).exp());
    // Sigma = Rg diag(s2) Rg^T.
    let sigma: [[D; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            rg[i][0] * rg[j][0] * s2[0] + rg[i][1] * rg[j][1] * s2[1] + rg[i][2] * rg[j][2] * s2[2]
        })
    });
    // Camera-space covariance W Sigma W^T.
    let ws: [[D; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            sigma[0][j] * rv(i, 0) + sigma[1][j] * rv(i, 1) + sigma[2][j] * rv(i, 2)
        })
    });
    let cov_cam: [[D; 3]; 3] = std::array::from_fn(|i| {
        std::array::from_fn(|j| ws[i][0] * rv(j, 0) + ws[i][1] * rv(j, 1) + ws[i][2] * rv(j, 2))
    });

    // EWA Jacobian, linearised at a point clamped to 1.3x the frustum (a
    // pinhole's), or at the point itself (a fisheye's), past max_theta at
    // the point's turn onto the cone; the screen mean.
    let z = p_cam[2];
    let inv_z = one / z;
    let inv_z2 = inv_z * inv_z;
    let lim_x = 1.3 * 0.5 * w / fx;
    let lim_y = 1.3 * 0.5 * h / fy;
    let tx = (p_cam[0] * inv_z).clamp(-lim_x, lim_x) * z;
    let ty = (p_cam[1] * inv_z).clamp(-lim_y, lim_y) * z;
    let (j0, j1, u, v) = if lens.is_pinhole() {
        (
            [inv_z * fx, D::c(0.0), -(tx * inv_z2) * fx],
            [D::c(0.0), inv_z * fy, -(ty * inv_z2) * fy],
            p_cam[0] * inv_z * fx + cx,
            p_cam[1] * inv_z * fy + cy,
        )
    } else {
        let ([mu, mv], rows, q) = extended_lens_coords(&lens, p_cam);
        let rows = match lens.projection {
            Projection::Pinhole => {
                let inv_qz = one / q[2];
                let clamped = |c: D, lim: f64| (c * inv_qz).clamp(-lim, lim) * q[2];
                lens_coords(&lens, [clamped(q[0], lim_x), clamped(q[1], lim_y), q[2]]).1
            }
            Projection::Equidistant => rows,
        };
        (
            rows[0].map(|x| x * fx),
            rows[1].map(|x| x * fy),
            mu * fx + cx,
            mv * fy + cy,
        )
    };
    let jcj = |a: &[D; 3], b: &[D; 3]| {
        let mut acc = D::c(0.0);
        for i in 0..3 {
            for k in 0..3 {
                acc = acc + a[i] * cov_cam[i][k] * b[k];
            }
        }
        acc
    };
    let a = jcj(&j0, &j0) + 0.3;
    let b = jcj(&j0, &j1);
    let c = jcj(&j1, &j1) + 0.3;
    let det = a * c - b * b;
    if det.v.is_nan() || det.v <= 0.0 {
        return None;
    }
    let mid = 0.5 * (a.v + c.v);
    let lambda2 = mid - (mid * mid - det.v).max(0.0).sqrt();
    if lambda2 <= 0.0 {
        return None;
    }
    let conic = [c / det, -b / det, a / det];

    // Footprint extent (culling only, not differentiated).
    let k = 2.0 * (255.0 * opacity.v).ln();
    if !k.is_finite() || k <= 0.0 {
        return None;
    }
    let ext_x = (k * a.v).sqrt();
    let ext_y = (k * c.v).sqrt();

    if u.v + ext_x < 0.0 || v.v + ext_y < 0.0 || u.v - ext_x > w - 1.0 || v.v - ext_y > h - 1.0 {
        return None;
    }

    // View-dependent colour; the direction depends on the mean.
    let cc = view.camera_center();
    let dir_raw = [
        mean[0] + -(cc.x as f64),
        mean[1] + -(cc.y as f64),
        mean[2] + -(cc.z as f64),
    ];
    let dn = (dir_raw[0] * dir_raw[0] + dir_raw[1] * dir_raw[1] + dir_raw[2] * dir_raw[2]).sqrt();
    let dir = if dn.v > f32::EPSILON as f64 {
        dir_raw.map(|x| x / dn)
    } else {
        [D::c(0.0), D::c(0.0), D::c(1.0)]
    };
    let basis = sh_basis(sh_degree, dir[0], dir[1], dir[2]);
    let per_ch = basis.len() - 1;
    let mut color = [D::c(0.0); 3];
    let mut color_clamped = [false; 3];
    for ch in 0..3 {
        let mut acc = basis[0] * g.sh_dc[ch] as f64;
        for kk in 0..per_ch {
            if let Some(&coef) = g.sh_rest.get(ch * per_ch + kk) {
                acc = acc + basis[kk + 1] * coef as f64;
            }
        }
        let raw = acc + 0.5;
        if raw.v < 0.0 {
            color[ch] = D::c(0.0);
            color_clamped[ch] = true;
        } else {
            color[ch] = raw;
        }
    }

    Some(Proj {
        index,
        depth,
        u,
        v,
        conic,
        opacity,
        color,
        color_clamped,
        basis: basis.iter().map(|b| b.v).collect(),
        ext_x,
        ext_y,
    })
}

fn project_all(scene: &Scene, view: &CameraView) -> Vec<Proj> {
    let mut projected: Vec<Proj> = scene
        .gaussians
        .iter()
        .enumerate()
        .filter_map(|(i, g)| project(i, g, view, scene.sh_degree))
        .collect();
    projected.sort_by(|a, b| {
        a.depth
            .partial_cmp(&b.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    projected
}

/// One splat's contribution to one pixel, recorded by the forward walk.
struct Hit {
    /// Index into the projected list.
    p: usize,
    alpha: f64,
    /// Transmittance in front of this splat.
    t: f64,
    /// `opacity * exp(-sigma)` exceeded 0.99 (alpha clamped: no gradient).
    clamped: bool,
    exp_neg_sigma: f64,
    dx: f64,
    dy: f64,
}

/// Composite one pixel front to back, recording each contributing splat.
fn walk_pixel(projected: &[Proj], px: u32, py: u32) -> (Vec<Hit>, f64) {
    let mut hits = Vec::new();
    let mut t = 1.0f64;
    for (i, p) in projected.iter().enumerate() {
        // Same pixel rect as the CPU renderer.
        let x0 = (p.u.v - p.ext_x).floor().max(0.0);
        let y0 = (p.v.v - p.ext_y).floor().max(0.0);
        let x1 = (p.u.v + p.ext_x).ceil();
        let y1 = (p.v.v + p.ext_y).ceil();
        let (fx, fy) = (px as f64, py as f64);
        if fx < x0 || fx > x1 || fy < y0 || fy > y1 {
            continue;
        }
        let dx = fx + 0.5 - p.u.v;
        let dy = fy + 0.5 - p.v.v;
        let [ca, cb, cc] = [p.conic[0].v, p.conic[1].v, p.conic[2].v];
        let power = -0.5 * (ca * dx * dx + cc * dy * dy) - cb * dx * dy;
        if power > 0.0 {
            continue;
        }
        let e = power.exp();
        let raw = p.opacity.v * e;
        let alpha = raw.min(0.99);
        if alpha < 1.0 / 255.0 {
            continue;
        }
        if t <= 1e-4 {
            continue;
        }
        hits.push(Hit {
            p: i,
            alpha,
            t,
            clamped: raw > 0.99,
            exp_neg_sigma: e,
            dx,
            dy,
        });
        t *= 1.0 - alpha;
    }
    (hits, t)
}

/// Forward render in `f64` (same result as [`crate::cpu_render::render`] up to
/// float precision). Row-major RGB.
pub fn render_f64(scene: &Scene, view: &CameraView, bg: [f64; 3]) -> Vec<[f64; 3]> {
    let projected = project_all(scene, view);
    let (w, h) = (view.camera.width, view.camera.height);
    let mut out = Vec::with_capacity((w * h) as usize);
    for py in 0..h {
        for px in 0..w {
            let (hits, t_final) = walk_pixel(&projected, px, py);
            let mut c = [0.0f64; 3];
            for hit in &hits {
                let p = &projected[hit.p];
                for (ch, cv) in c.iter_mut().enumerate() {
                    *cv += hit.t * hit.alpha * p.color[ch].v;
                }
            }
            for (cv, b) in c.iter_mut().zip(bg) {
                *cv += t_final * b;
            }
            out.push(c);
        }
    }
    out
}

/// Gradients of `L = sum_px dot(d_image[px], C[px])` per gaussian parameter,
/// indexed like `scene.gaussians`.
#[derive(Debug, Clone)]
pub struct Grads {
    pub mean: Vec<[f64; 3]>,
    pub scale_log: Vec<[f64; 3]>,
    /// Gradient w.r.t. the stored (un-normalised) quaternion `(w, x, y, z)`.
    pub rotation: Vec<[f64; 4]>,
    pub opacity_logit: Vec<f64>,
    pub sh_dc: Vec<[f64; 3]>,
    /// Channel-major like [`Gaussian::sh_rest`].
    pub sh_rest: Vec<Vec<f64>>,
}

impl Grads {
    fn zeros(scene: &Scene) -> Self {
        let n = scene.len();
        Self {
            mean: vec![[0.0; 3]; n],
            scale_log: vec![[0.0; 3]; n],
            rotation: vec![[0.0; 4]; n],
            opacity_logit: vec![0.0; n],
            sh_dc: vec![[0.0; 3]; n],
            sh_rest: scene
                .gaussians
                .iter()
                .map(|g| vec![0.0; g.sh_rest.len()])
                .collect(),
        }
    }
}

/// Screen-space gradient of one projected splat, summed over pixels.
#[derive(Default, Clone, Copy)]
struct ScreenGrad {
    u: f64,
    v: f64,
    conic: [f64; 3],
    opacity: f64,
    color: [f64; 3],
}

/// Per projected splat, the pixel-summed gradient of its screen parameters.
fn screen_grads(
    projected: &[Proj],
    view: &CameraView,
    bg: [f64; 3],
    d_image: &[[f64; 3]],
) -> Vec<ScreenGrad> {
    let (w, h) = (view.camera.width, view.camera.height);
    assert_eq!(d_image.len(), (w * h) as usize, "d_image size");
    let mut screen = vec![ScreenGrad::default(); projected.len()];

    for py in 0..h {
        for px in 0..w {
            let g = d_image[(py * w + px) as usize];
            let (hits, _t_final) = walk_pixel(projected, px, py);
            // Colour composited behind the current splat (background first).
            let mut behind = bg;
            for hit in hits.iter().rev() {
                let p = &projected[hit.p];
                let sg = &mut screen[hit.p];
                let col = [p.color[0].v, p.color[1].v, p.color[2].v];
                for (dc, gc) in sg.color.iter_mut().zip(g) {
                    *dc += hit.alpha * hit.t * gc;
                }
                let d_alpha = hit.t
                    * ((col[0] - behind[0]) * g[0]
                        + (col[1] - behind[1]) * g[1]
                        + (col[2] - behind[2]) * g[2]);
                for ch in 0..3 {
                    behind[ch] = hit.alpha * col[ch] + (1.0 - hit.alpha) * behind[ch];
                }
                if hit.clamped {
                    continue;
                }
                // alpha = opacity * exp(-sigma)
                sg.opacity += d_alpha * hit.exp_neg_sigma;
                let d_sigma = -d_alpha * hit.alpha;
                let [ca, cb, cc] = [p.conic[0].v, p.conic[1].v, p.conic[2].v];
                // sigma = 0.5 (A dx^2 + C dy^2) + B dx dy, d = pixel - mean
                sg.conic[0] += d_sigma * 0.5 * hit.dx * hit.dx;
                sg.conic[1] += d_sigma * hit.dx * hit.dy;
                sg.conic[2] += d_sigma * 0.5 * hit.dy * hit.dy;
                sg.u += d_sigma * -(ca * hit.dx + cb * hit.dy);
                sg.v += d_sigma * -(cb * hit.dx + cc * hit.dy);
            }
        }
    }

    screen
}

/// Screen-space gradients per gaussian (scene order; zeros when culled):
/// `[du, dv, dA, dB, dC, dopacity, dr, dg, db]` for the projected mean, conic
/// `[A, B, C]`, opacity value and clamped colour. The GPU backward's
/// `backward_screen` returns the same records.
pub fn render_backward_screen(
    scene: &Scene,
    view: &CameraView,
    bg: [f64; 3],
    d_image: &[[f64; 3]],
) -> Vec<[f64; 9]> {
    let projected = project_all(scene, view);
    let screen = screen_grads(&projected, view, bg, d_image);
    let mut out = vec![[0.0; 9]; scene.len()];
    for (p, sg) in projected.iter().zip(&screen) {
        out[p.index] = [
            sg.u,
            sg.v,
            sg.conic[0],
            sg.conic[1],
            sg.conic[2],
            sg.opacity,
            sg.color[0],
            sg.color[1],
            sg.color[2],
        ];
    }
    out
}

/// Reference backward pass: see the module docs.
pub fn render_backward(
    scene: &Scene,
    view: &CameraView,
    bg: [f64; 3],
    d_image: &[[f64; 3]],
) -> Grads {
    let projected = project_all(scene, view);
    let screen = screen_grads(&projected, view, bg, d_image);
    let mut grads = Grads::zeros(scene);
    for (p, sg) in projected.iter().zip(&screen) {
        let mut dp = [0.0f64; NP];
        let mut add = |d: &D, s: f64| {
            for (acc, x) in dp.iter_mut().zip(d.d) {
                *acc += s * x;
            }
        };
        add(&p.u, sg.u);
        add(&p.v, sg.v);
        for k in 0..3 {
            add(&p.conic[k], sg.conic[k]);
            add(&p.color[k], sg.color[k]);
        }
        add(&p.opacity, sg.opacity);

        let i = p.index;
        grads.mean[i] = [dp[0], dp[1], dp[2]];
        grads.scale_log[i] = [dp[3], dp[4], dp[5]];
        grads.rotation[i] = [dp[6], dp[7], dp[8], dp[9]];
        grads.opacity_logit[i] = dp[10];

        let per_ch = p.basis.len() - 1;
        for ch in 0..3 {
            if p.color_clamped[ch] {
                continue;
            }
            grads.sh_dc[i][ch] = sg.color[ch] * p.basis[0];
            for k in 0..per_ch {
                if let Some(slot) = grads.sh_rest[i].get_mut(ch * per_ch + k) {
                    *slot = sg.color[ch] * p.basis[k + 1];
                }
            }
        }
    }
    grads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::PinholeCamera;
    use nalgebra::{Quaternion, Vector3};

    fn test_scene() -> Scene {
        // A few overlapping, anisotropic, rotated, semi-transparent splats with
        // degree-2 SH, well inside the frame and away from the gates.
        let mk = |m: [f32; 3], s: [f32; 3], q: [f32; 4], o: f32, dc: [f32; 3], seed: f32| {
            let sh_rest = (0..24)
                .map(|k| 0.05 * ((k as f32 + seed) * 1.7).sin())
                .collect();
            Gaussian {
                mean: Vector3::new(m[0], m[1], m[2]),
                scale_log: Vector3::new(s[0].ln(), s[1].ln(), s[2].ln()),
                rotation: Quaternion::new(q[0], q[1], q[2], q[3]),
                opacity_logit: o,
                sh_dc: dc,
                sh_rest,
                sh_degree: 2,
            }
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
            2,
        )
    }

    fn test_view() -> CameraView {
        let r = nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.03).into_inner();
        CameraView::new(
            r,
            Vector3::new(0.1, -0.05, 0.2),
            PinholeCamera::new(40, 32, 45.0, 44.0, 20.0, 16.0),
        )
    }

    /// The pinhole view, then the same scene through lenses: a distorted
    /// pinhole, a fisheye seeing it ~50° off the axis, ETH3D's thin-prism
    /// rig lens, a wide fisheye seeing it past 90° (behind the image plane),
    /// and narrow ones (a fisheye and a distorted pinhole) seeing it past
    /// their `max_theta`, through the lens's extension.
    fn test_views() -> Vec<(&'static str, CameraView)> {
        use crate::lens::{Lens, Projection};
        let turned = |angle: f64, camera: PinholeCamera| {
            let r = nalgebra::Rotation3::from_axis_angle(&Vector3::y_axis(), angle as f32)
                * nalgebra::Rotation3::from_euler_angles(0.05, -0.08, 0.03);
            CameraView::new(r.into_inner(), Vector3::new(0.1, -0.05, 0.2), camera)
        };
        let base = test_view();
        let radtan = Lens::new(
            Projection::Pinhole,
            [-0.15, 0.03, 0.0, 0.0, 0.002, -0.001, 0.0, 0.0],
        );
        let kb = Lens::new(
            Projection::Equidistant,
            [0.03, -0.01, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let thin_prism = Lens::new(
            Projection::Equidistant,
            [
                -0.0654481,
                -0.0781379,
                0.115547,
                -0.0710756,
                -0.00131596,
                0.00171383,
                -0.0103304,
                0.00700661,
            ],
        );
        let equidistant = Lens::new(Projection::Equidistant, [0.0; 8]);
        vec![
            ("pinhole", base),
            (
                "radtan",
                CameraView {
                    camera: base.camera.with_lens(radtan),
                    ..base
                },
            ),
            (
                "fisheye 50°",
                turned(
                    0.9,
                    PinholeCamera::new(40, 32, 14.0, 14.5, 20.0, 16.0).with_lens(kb),
                ),
            ),
            (
                "thin prism",
                turned(
                    0.3,
                    PinholeCamera::new(40, 32, 30.0, 30.0, 21.0, 15.0).with_lens(thin_prism),
                ),
            ),
            (
                "fisheye past 90°",
                turned(
                    1.75,
                    PinholeCamera::new(40, 32, 8.0, 8.0, 20.0, 16.0).with_lens(equidistant),
                ),
            ),
            (
                "fisheye past its cone",
                turned(
                    0.35,
                    PinholeCamera::new(40, 32, 150.0, 150.0, 20.0, 16.0).with_lens(kb),
                ),
            ),
            (
                "radtan past its cone",
                turned(
                    0.36,
                    PinholeCamera::new(40, 32, 150.0, 150.0, 20.0, 16.0).with_lens(radtan),
                ),
            ),
        ]
    }

    fn weights(n: usize) -> Vec<[f64; 3]> {
        (0..n)
            .map(|i| {
                let f = i as f64;
                [
                    (f * 0.37).sin(),
                    (f * 0.91).cos(),
                    (f * 0.13).sin() * 0.5 + 0.2,
                ]
            })
            .collect()
    }

    fn loss(scene: &Scene, view: &CameraView, bg: [f64; 3], w: &[[f64; 3]]) -> f64 {
        render_f64(scene, view, bg)
            .iter()
            .zip(w)
            .map(|(c, g)| c[0] * g[0] + c[1] * g[1] + c[2] * g[2])
            .sum()
    }

    /// Central differences of the loss w.r.t. one parameter (set via `f`) at
    /// several step sizes. The renderer has hard gates (alpha >= 1/255, the
    /// footprint rect, T > 1e-4); a step can push a pixel across one and make
    /// that single difference jump, but not every step size at once.
    fn fd(
        scene: &Scene,
        view: &CameraView,
        bg: [f64; 3],
        w: &[[f64; 3]],
        f: impl Fn(&mut Gaussian, f32),
    ) -> Vec<f64> {
        [1e-3f32, 3e-4, 1e-4, 3e-5]
            .iter()
            .map(|&eps| {
                let mut plus = scene.clone();
                let mut minus = scene.clone();
                f(&mut plus.gaussians[0], eps);
                f(&mut minus.gaussians[0], -eps);
                (loss(&plus, view, bg, w) - loss(&minus, view, bg, w)) / (2.0 * eps as f64)
            })
            .collect()
    }

    #[test]
    fn f64_forward_matches_cpu_render() {
        let scene = test_scene();
        let bg = [0.1f32, 0.2, 0.3];
        for (name, view) in test_views() {
            let a = crate::cpu_render::render(&scene, &view, bg);
            let b = render_f64(&scene, &view, bg.map(|x| x as f64));
            let max = a
                .rgb
                .iter()
                .zip(&b)
                .flat_map(|(x, y)| (0..3).map(move |c| (x[c] as f64 - y[c]).abs()))
                .fold(0.0, f64::max);
            assert!(max < 1e-4, "{name}: f32 vs f64 forward differ by {max}");
        }
    }

    #[test]
    fn gradients_match_finite_differences() {
        for (name, view) in test_views() {
            check_gradients(name, &view);
        }
    }

    fn check_gradients(view_name: &str, view: &CameraView) {
        let view = *view;
        let bg = [0.1, 0.2, 0.3];
        let n = (view.camera.width * view.camera.height) as usize;
        let w = weights(n);
        // Check every gaussian by rotating it into slot 0.
        for which in 0..3 {
            let mut scene = test_scene();
            scene.gaussians.swap(0, which);
            let grads = render_backward(&scene, &view, bg, &w);
            let mut checks: Vec<(String, f64, Vec<f64>)> = Vec::new();
            for axis in 0..3 {
                checks.push((
                    format!("mean[{axis}]"),
                    grads.mean[0][axis],
                    fd(&scene, &view, bg, &w, |g, e| g.mean[axis] += e),
                ));
                checks.push((
                    format!("scale_log[{axis}]"),
                    grads.scale_log[0][axis],
                    fd(&scene, &view, bg, &w, |g, e| g.scale_log[axis] += e),
                ));
            }
            for k in 0..4 {
                checks.push((
                    format!("rotation[{k}]"),
                    grads.rotation[0][k],
                    fd(&scene, &view, bg, &w, |g, e| match k {
                        0 => g.rotation.w += e,
                        1 => g.rotation.i += e,
                        2 => g.rotation.j += e,
                        _ => g.rotation.k += e,
                    }),
                ));
            }
            checks.push((
                "opacity_logit".into(),
                grads.opacity_logit[0],
                fd(&scene, &view, bg, &w, |g, e| g.opacity_logit += e),
            ));
            for ch in 0..3 {
                checks.push((
                    format!("sh_dc[{ch}]"),
                    grads.sh_dc[0][ch],
                    fd(&scene, &view, bg, &w, |g, e| g.sh_dc[ch] += e),
                ));
            }
            for k in [0usize, 3, 7, 11, 16, 23] {
                checks.push((
                    format!("sh_rest[{k}]"),
                    grads.sh_rest[0][k],
                    fd(&scene, &view, bg, &w, |g, e| g.sh_rest[k] += e),
                ));
            }
            for (name, analytic, numeric) in checks {
                let agrees =
                    |n: &f64| (analytic - n).abs() <= 2e-3 * n.abs().max(analytic.abs()).max(1e-2);
                assert!(
                    numeric.iter().any(agrees),
                    "{view_name}: gaussian {which} {name}: analytic {analytic:.6e} vs FD {numeric:?}"
                );
            }
        }
    }

    #[test]
    fn gradients_are_nonzero() {
        // Guard against a vacuous FD match (e.g. everything culled).
        let scene = test_scene();
        for (name, view) in test_views() {
            let n = (view.camera.width * view.camera.height) as usize;
            let grads = render_backward(&scene, &view, [0.1, 0.2, 0.3], &weights(n));
            for i in 0..3 {
                assert!(
                    grads.mean[i].iter().any(|x| x.abs() > 1e-6),
                    "{name}: mean grad {i}"
                );
                assert!(
                    grads.opacity_logit[i].abs() > 1e-6,
                    "{name}: opacity grad {i}"
                );
            }
        }
    }

    /// The lens views draw the scene where intended: ~50° and past 90° off
    /// the axis, the latter behind the image plane, and past the narrow
    /// lenses' cones.
    #[test]
    fn lens_views_see_the_scene_off_axis() {
        let scene = test_scene();
        let views = test_views();
        let angle = |view: &CameraView| {
            let p = view.world_to_camera_point(scene.gaussians[0].mean);
            ray_angle([p.x, p.y, p.z].map(f64::from)).to_degrees()
        };
        assert!((40.0..60.0).contains(&angle(&views[2].1)));
        let past = &views[4].1;
        assert!(angle(past) > 90.0 && past.world_to_camera_point(scene.gaussians[0].mean).z < 0.0);
        assert!(!project_all(&scene, past).is_empty());
        // Every centre lies past the narrow lenses' max_theta, yet all of
        // them are drawn (gradients_are_nonzero checks they reach pixels).
        for (name, view) in &views[5..] {
            for g in &scene.gaussians {
                let p = view.world_to_camera_point(g.mean);
                let theta = ray_angle([p.x, p.y, p.z].map(f64::from));
                assert!(
                    theta > view.camera.lens.max_theta as f64 + 0.005,
                    "{name}: {theta}"
                );
            }
            assert_eq!(project_all(&scene, view).len(), 3, "{name}");
        }
    }
}
