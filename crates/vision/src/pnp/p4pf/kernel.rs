// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Vendored verbatim from OpenMVG-rs (openmvg-geometry/src/pose/p4pf/kernel.rs,
// https://github.com/openMVG/openmvg-rs, MIT OR Apache-2.0). Changes: the
// `if ... && let` chain rewritten as nested `if`s for edition 2021, the
// `unstable-test-support` module dropped, and the reference-test pointer below
// refers to the upstream test, which is not vendored.

//! Absolute pose with unknown focal length from four 3D-2D correspondences (P4Pf).
//!
//! ```text
//! 4 points (X_i, uv_i) -> center and scale both sets -> DLT null basis B (12 x 4)
//!   -> camera rows p1, p2, p3 = B [x; 1]; metric constraints p1.p2 = p1.p3 = p2.p3 = 0
//!   -> 3 quadrics in x (E3Q3) -> hidden-variable pencil in a rational Cayley chart
//!   -> 8 x 8 companion, Hessenberg + Francis QR, Newton-polished roots
//!   -> seeds (R, t, f) from near-real roots (+ a planar seed for near-coplanar points)
//!   -> Gauss-Newton polish on the 8 reprojection residuals -> gates, sort, dedup
//! ```
//!
//! The arithmetic is bit-exact with the project's Julia reference implementation:
//! upstream's `tests/p4pf_reference.rs` checks the solver and its branch cases against exported reference
//! outputs. The derivations live in that reference.
//!
//! * Front end: world points centered and scaled by their mean radius, image points by theirs;
//!   repeated world points declined; the DLT system (12 unknowns, 8 equations) by Householder QR,
//!   its 12 x 4 null basis `B` and a singular-value ratio estimate (collinear points declined);
//!   the coplanarity ratio of the 4 points (no decline).
//! * Seeds: near-real roots seed (R, t, f) (polar factor, translation normal equations); for
//!   (near-)coplanar points also a planar seed (plane frame, 4-point homography, focal length and
//!   pose from it). Each seed gets a Gauss-Newton polish and the gates: f > 0, maximum reprojection
//!   at most `5e-2` of the mean image radius, points in front, perspective.
//!
//! # Contract
//!
//! [`p4pf`]`(x, uv, out, &mut status)`: world points `x[i]`, principal-point-centered image points
//! `uv[i]` (any units); poses with `uv_i ~ f (R x_i + t)[0..2] / (R x_i + t)[2]`, `R` row-major,
//! `reproj` = maximum reprojection error / mean image radius. Returns the number of poses
//! (at most [`MAX_POSES`]), sorted by `reproj`. `status`: [`FRONT_END`] the front end declined
//! (non-finite, repeated or collinear points), [`LANE`] the lane answered, [`LANE_DECLINED`] the
//! pencil itself failed (no pose). No allocation, no state.
//!
//! Floating point: Rust never contracts `a * b + c`; the reference's `muladd` and `fma` sites
//! (evalpoly, hypot) are `f64::mul_add`, so results are bit-identical across targets.

// Index loops mirror the reference one to one (bit-exactness); the negated float comparisons fix
// the branch a NaN takes.
#![allow(
    clippy::needless_range_loop,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::nonminimal_bool
)]

use core::ops::Add;
use core::ops::Div;
use core::ops::Mul;
use core::ops::Sub;

/// A pose: `r` row-major rotation, `t` translation, `f` focal length, `reproj` max reprojection error / mean image
/// radius.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    /// Row-major rotation, `X_camera = R X_world + t`.
    pub r: [f64; 9],
    /// Translation.
    pub t: [f64; 3],
    /// Focal length, in the units of the image points.
    pub f: f64,
    /// Maximum reprojection error over the four points, divided by the mean image radius.
    pub reproj: f64,
}

/// Status: the front end declined the input (non-finite, repeated, or collinear points).
pub const FRONT_END: i32 = 0;
/// Status: the pencil lane answered.
pub const LANE: i32 = 1;
/// Status: the pencil itself failed, so no pose is returned.
pub const LANE_DECLINED: i32 = 2;

// -------------------------------------------- Julia scalar semantics
// Base.max on Float64 (NaN-propagating, -0.0 < 0.0); f64::max ignores NaN
fn jmax(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    if x == y {
        return if x.is_sign_negative() { y } else { x };
    }
    if x > y {
        x
    } else {
        y
    }
}

// Base.Math._hypot (the hardware-fma branch)
fn jhypot(x: f64, y: f64) -> f64 {
    let (mut ax, mut ay) = (x.abs(), y.abs());
    if ax.is_infinite() || ay.is_infinite() {
        return f64::INFINITY;
    }
    if ay > ax {
        core::mem::swap(&mut ax, &mut ay);
    }
    if ay <= ax * (f64::EPSILON / 2.0).sqrt() {
        return ax;
    }
    let mut scale = f64::EPSILON * f64::MIN_POSITIVE.sqrt();
    if ax > (f64::MAX / 2.0).sqrt() {
        ax *= scale;
        ay *= scale;
        scale = 1.0 / scale;
    } else if ay < f64::MIN_POSITIVE.sqrt() {
        ax /= scale;
        ay /= scale;
    } else {
        scale = 1.0;
    }
    let mut h = ax.mul_add(ax, ay * ay).sqrt();
    let (hs, axs) = (h * h, ax * ax);
    h -= ((-ay).mul_add(ay, hs - axs) + h.mul_add(h, -hs) - ax.mul_add(ax, -axs)) / (2.0 * h);
    h * scale
}

// LinearAlgebra.generic_norm2 (n < 32: no BLAS), x in Julia iteration order
fn jnorm(x: &[f64]) -> f64 {
    let n = x.len();
    let mut m = x[0].abs();
    for i in 1..n {
        m = jmax(m, x[i].abs());
    }
    if m == 0.0 || m.is_infinite() {
        return m;
    }
    if (n as f64 * m * m).is_finite() && m * m != 0.0 {
        let mut s = x[0] * x[0];
        for i in 1..n {
            s += x[i] * x[i];
        }
        return s.sqrt();
    }
    let mut s = (x[0].abs() / m) * (x[0].abs() / m);
    for i in 1..n {
        let y = x[i].abs() / m;
        s += y * y;
    }
    m * s.sqrt()
}

// Base Complex{Float64}: * / + - as in complex.jl (no fused operations), / = Baudin-Smith with Julia's scaling
#[derive(Clone, Copy, Debug, Default)]
struct Cx {
    re: f64,
    im: f64,
}
impl Add for Cx {
    type Output = Cx;
    fn add(self, b: Cx) -> Cx {
        Cx {
            re: self.re + b.re,
            im: self.im + b.im,
        }
    }
}
impl Sub for Cx {
    type Output = Cx;
    fn sub(self, b: Cx) -> Cx {
        Cx {
            re: self.re - b.re,
            im: self.im - b.im,
        }
    }
}
impl Mul for Cx {
    type Output = Cx;
    fn mul(self, b: Cx) -> Cx {
        Cx {
            re: self.re * b.re - self.im * b.im,
            im: self.re * b.im + self.im * b.re,
        }
    }
}
fn jsign(x: f64) -> f64 {
    if x < 0.0 {
        -1.0
    } else if x > 0.0 {
        1.0
    } else {
        x
    }
}
fn cdiv2(a: f64, b: f64, c: f64, d: f64, r: f64, t: f64) -> f64 {
    if r != 0.0 {
        let br = b * r;
        return if br != 0.0 {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        };
    }
    (a + d * (b / c)) * t
}
fn cdiv1(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    let r = d / c;
    let t = 1.0 / (c + d * r);
    (cdiv2(a, b, c, d, r, t), cdiv2(b, -a, c, d, r, t))
}
fn cdiv0(a: f64, b: f64, c: f64, d: f64) -> (f64, f64) {
    if d.abs() <= c.abs() {
        cdiv1(a, b, c, d)
    } else {
        let (p, q) = cdiv1(b, a, d, c);
        (p, -q)
    }
}
impl Div for Cx {
    type Output = Cx;
    fn div(self, w: Cx) -> Cx {
        let (mut a, mut b, mut c, mut d) = (self.re, self.im, w.re, w.im);
        let (absa, absb) = (a.abs(), b.abs());
        let ab = if absa >= absb { absa } else { absb };
        let (absc, absd) = (c.abs(), d.abs());
        let cd = if absc >= absd { absc } else { absd };
        if c.is_infinite() || d.is_infinite() {
            if a.is_finite() && b.is_finite() {
                return Cx {
                    re: 0.0 * jsign(a) * jsign(c),
                    im: -0.0 * jsign(b) * jsign(d),
                };
            }
            return Cx {
                re: f64::NAN,
                im: f64::NAN,
            };
        }
        let halfov = 0.5 * f64::MAX;
        let twoune = f64::MIN_POSITIVE * 2.0 / f64::EPSILON;
        if ab >= halfov || ab <= twoune || cd >= halfov || cd <= twoune {
            let bs = 2.0 / (f64::EPSILON * f64::EPSILON);
            let mut s = 1.0;
            if ab >= halfov {
                a *= 0.5;
                b *= 0.5;
                s *= 2.0;
            } else if ab <= twoune {
                a *= bs;
                b *= bs;
                s /= bs;
            }
            if cd >= halfov {
                c *= 0.5;
                d *= 0.5;
                s *= 0.5;
            } else if cd <= twoune {
                c *= bs;
                d *= bs;
                s *= bs;
            }
            let (p, q) = cdiv0(a, b, c, d);
            return Cx {
                re: p * s,
                im: q * s,
            };
        }
        let (p, q) = cdiv0(a, b, c, d);
        Cx { re: p, im: q }
    }
}

type P5 = [f64; 5]; // ascending coefficients in z, degree <= 4

/// The scalar of the lane: real eigenvalues stay in real arithmetic, complex ones in Julia's complex arithmetic.
trait Sc:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    fn scale(self, x: f64) -> Self; // x * self
    fn addr(self, x: f64) -> Self; // self + x
    fn abs2(self) -> f64;
    fn mag(self) -> f64;
    fn is_zero(self) -> bool;
    fn finite(self) -> bool;
    fn evalpoly(self, p: &P5) -> Self;
    // _qeval's componentwise weight (real: |c_j m_j|, complex: |c_j| |m_j|)
    fn weight(q: &Q10, x: Self, y: Self, z: Self) -> f64;
    fn to_cx(self) -> Cx;
}
impl Sc for f64 {
    fn scale(self, x: f64) -> f64 {
        x * self
    }
    fn addr(self, x: f64) -> f64 {
        self + x
    }
    fn abs2(self) -> f64 {
        self * self
    }
    fn mag(self) -> f64 {
        self.abs()
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    // evalpoly on a real z: Horner with muladd
    fn evalpoly(self, p: &P5) -> f64 {
        let mut e = p[4];
        for i in (0..4).rev() {
            e = self.mul_add(e, p[i]);
        }
        e
    }
    fn weight(q: &Q10, x: f64, y: f64, z: f64) -> f64 {
        q[0].abs() * (x * x)
            + (q[1] * x * y).abs()
            + (q[2] * x * z).abs()
            + q[3].abs() * (y * y)
            + (q[4] * y * z).abs()
            + q[5].abs() * (z * z)
            + (q[6] * x).abs()
            + (q[7] * y).abs()
            + (q[8] * z).abs()
            + q[9].abs()
    }
    fn to_cx(self) -> Cx {
        Cx { re: self, im: 0.0 }
    }
}
impl Sc for Cx {
    fn scale(self, x: f64) -> Cx {
        Cx {
            re: x * self.re,
            im: x * self.im,
        }
    }
    fn addr(self, x: f64) -> Cx {
        Cx {
            re: self.re + x,
            im: self.im,
        }
    }
    fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    fn mag(self) -> f64 {
        jhypot(self.re, self.im)
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    // the Goertzel form of Base.Math.evalpoly on a complex z
    fn evalpoly(self, p: &P5) -> Cx {
        let (x, y) = (self.re, self.im);
        let (r, s) = (x + x, x.mul_add(x, y * y));
        let (mut a, mut b) = (p[4], p[3]);
        for i in (0..3).rev() {
            let ai = a;
            a = r.mul_add(ai, b);
            b = (-s).mul_add(ai, p[i]);
        }
        Cx {
            re: self.re.mul_add(a, b),
            im: self.im * a,
        } // muladd(a0, z, b)
    }
    fn weight(q: &Q10, x: Cx, y: Cx, z: Cx) -> f64 {
        let (a, b, c) = (x.mag(), y.mag(), z.mag());
        q[0].abs() * (a * a)
            + q[1].abs() * a * b
            + q[2].abs() * a * c
            + q[3].abs() * (b * b)
            + q[4].abs() * b * c
            + q[5].abs() * (c * c)
            + q[6].abs() * a
            + q[7].abs() * b
            + q[8].abs() * c
            + q[9].abs()
    }
    fn to_cx(self) -> Cx {
        self
    }
}

// -------------------------------------------- refine.jl: small_solve!, house_nullspace!, tri_sv_ratio
// J \ b by partial pivoting in place (n <= 16); false on an exactly zero pivot
fn small_solve<const N: usize, const M: usize>(
    j: &mut [[f64; N]; N],
    b: &mut [[f64; M]; N],
) -> bool {
    for k in 0..N {
        let (mut p, mut mx) = (k, j[k][k].abs());
        for i in k + 1..N {
            let a = j[i][k].abs();
            if a > mx {
                p = i;
                mx = a;
            }
        }
        if mx == 0.0 {
            return false;
        }
        if p != k {
            for c in k..N {
                let s = j[k][c];
                j[k][c] = j[p][c];
                j[p][c] = s;
            }
            b.swap(k, p);
        }
        for i in k + 1..N {
            let l = j[i][k] / j[k][k];
            if l == 0.0 {
                continue;
            }
            for c in k + 1..N {
                j[i][c] -= l * j[k][c];
            }
            for c in 0..M {
                b[i][c] -= l * b[k][c];
            }
        }
    }
    for c in 0..M {
        for k in (0..N).rev() {
            let mut s = b[k][c];
            for jj in k + 1..N {
                s -= j[k][jj] * b[jj][c];
            }
            b[k][c] = s / j[k][k];
        }
    }
    true
}

// tri_sv_gate(w, rd, tol): a value on the same side of tol as S_n / S_1 (U = min / max |R_kk| when <= tol,
// L = 1 / (|R|_F |R^-1|_F) when > tol, else tri_sv_ratio)
fn tri_sv_gate<const M: usize, const K: usize>(w: &[[f64; K]; M], rd: &[f64; K], tol: f64) -> f64 {
    if rd.contains(&0.0) {
        return 0.0;
    }
    let (mut dmin, mut dmax) = (rd[0].abs(), rd[0].abs());
    for &r in &rd[1..] {
        let a = r.abs();
        dmin = if a < dmin { a } else { dmin };
        dmax = if a > dmax { a } else { dmax };
    }
    let u = dmin / dmax;
    if u <= tol {
        return u;
    }
    let (mut fr, mut fi, mut x) = (0.0, 0.0, [0.0; K]);
    for j in 0..K {
        // column j of R^-1 into x[0..=j]; |R|_F^2 by columns too
        fr += rd[j] * rd[j];
        for row in &w[..j] {
            fr += row[j] * row[j];
        }
        x[j] = 1.0 / rd[j];
        fi += x[j] * x[j];
        for k in (0..j).rev() {
            let mut a = 0.0;
            for i in k + 1..=j {
                a += w[k][i] * x[i];
            }
            x[k] = -a / rd[k];
            fi += x[k] * x[k];
        }
    }
    let l = 1.0 / (fr * fi).sqrt();
    if l > tol {
        l
    } else {
        tri_sv_ratio(w, rd)
    }
}

// S_n / S_1 of the upper triangular R (diagonal rd, R_kj = w[k][j] above it)
fn tri_sv_ratio<const M: usize, const K: usize>(w: &[[f64; K]; M], rd: &[f64; K]) -> f64 {
    if rd.contains(&0.0) {
        return 0.0;
    }
    let (mut u, mut v) = ([0.0; K], [0.0; K]);
    u[0] = 1.0;
    let mut s1 = 0.0;
    for _ in 0..16 {
        let mut s = 0.0;
        for k in 0..K {
            let mut a = 0.0;
            a += rd[k] * u[k];
            for j in k + 1..K {
                a += w[k][j] * u[j];
            }
            v[k] = a;
            s += a * a;
        }
        s1 = s.sqrt();
        s = 0.0;
        for j in 0..K {
            let mut a = 0.0;
            for k in 0..j {
                a += w[k][j] * v[k];
            }
            a += rd[j] * v[j];
            u[j] = a;
            s += a * a;
        }
        let r = 1.0 / s.sqrt();
        for j in 0..K {
            u[j] *= r;
        }
    }
    u = [0.1; K];
    u[K - 1] = 1.0;
    let mut sn = 0.0;
    let ss = |u: &[f64; K]| {
        let mut s = u[0] * u[0];
        for j in 1..K {
            s += u[j] * u[j];
        }
        s
    };
    for _ in 0..4 {
        let r = 1.0 / ss(&u).sqrt();
        for j in 0..K {
            u[j] *= r;
        }
        for j in 0..K {
            let mut a = u[j];
            for k in 0..j {
                a -= w[k][j] * v[k];
            }
            v[j] = a / rd[j];
        }
        for k in (0..K).rev() {
            let mut a = v[k];
            for j in k + 1..K {
                a -= w[k][j] * u[j];
            }
            u[k] = a / rd[k];
        }
        sn = 1.0 / ss(&u).sqrt().sqrt();
    }
    if u.iter().all(|x| x.is_finite()) {
        sn / s1
    } else {
        0.0
    }
}

// Householder QR of w (M x K, K <= M) in place, column pivoting optional: the reflectors in w, R's diagonal returned;
// None where house_nullspace! returns early (only NaN columns left to pivot: NaN never pivots; w as transformed so far)
fn house_qr<const M: usize, const K: usize>(
    w: &mut [[f64; K]; M],
    pivot: bool,
) -> Option<[f64; K]> {
    let mut rd = [0.0; K];
    for c in 0..K {
        let (mut jm, mut nm) = (c, -1.0);
        for j in c..=(if pivot { K - 1 } else { c }) {
            let mut sj = 0.0;
            for i in c..M {
                sj += w[i][j] * w[i][j];
            }
            if sj > nm {
                jm = j;
                nm = sj;
            }
        }
        if nm < 0.0 {
            return None;
        }
        if jm != c {
            for i in 0..M {
                w[i].swap(c, jm);
            }
        }
        let alpha = -nm.sqrt().copysign(w[c][c]);
        rd[c] = alpha;
        w[c][c] -= alpha;
        let mut vn = 0.0;
        for i in c..M {
            vn += w[i][c] * w[i][c];
        }
        vn = vn.sqrt();
        if !(vn > 0.0) {
            continue;
        }
        for i in c..M {
            w[i][c] /= vn;
        }
        for j in c + 1..K {
            let mut s = 0.0;
            for i in c..M {
                s += w[i][c] * w[i][j];
            }
            s *= 2.0;
            for i in c..M {
                w[i][j] -= s * w[i][c];
            }
        }
    }
    Some(rd)
}

// house_nullspace!(W; pivot, tol): (N, ratio) = the null basis Q e_(K+1..M) (L = M - K) and tri_sv_ratio(R), or
// tri_sv_gate(R, tol) for Some(tol); N all NaN and ratio 0 when only NaN columns are left to pivot (a failed rank test)
fn house_nullspace<const M: usize, const K: usize, const L: usize>(
    w: &mut [[f64; K]; M],
    pivot: bool,
    tol: Option<f64>,
) -> ([[f64; L]; M], f64) {
    let rd = match house_qr(w, pivot) {
        Some(rd) => rd,
        None => return ([[f64::NAN; L]; M], 0.0),
    };
    let n = null_basis(w);
    let ratio = match tol {
        Some(tol) => tri_sv_gate(w, &rd, tol),
        None => tri_sv_ratio(w, &rd),
    };
    (n, ratio)
}

// house_nullspace!'s null basis Q e_(K+1..M) of the M x K QR in w (L = M - K)
fn null_basis<const M: usize, const K: usize, const L: usize>(w: &[[f64; K]; M]) -> [[f64; L]; M] {
    let mut n = [[0.0; L]; M];
    for l in 0..L {
        n[K + l][l] = 1.0;
    }
    for c in (0..K).rev() {
        for l in 0..L {
            let mut s = 0.0;
            for i in c..M {
                s += w[i][c] * n[i][l];
            }
            s *= 2.0;
            for i in c..M {
                n[i][l] -= s * w[i][c];
            }
        }
    }
    n
}

// -------------------------------------------- pencil.jl: hessenberg_gauss!, hqr_eigvals!(fast = true, tol = 1e-12)
// (1-based, as NR)
const NE: usize = 8;
type Mat8 = [[f64; NE + 1]; NE + 1];

fn hessenberg_gauss(a: &mut Mat8) {
    let n = NE;
    for m in 2..n {
        let (mut x, mut i) = (0.0f64, m);
        for j in m..=n {
            if a[j][m - 1].abs() > x.abs() {
                x = a[j][m - 1];
                i = j;
            }
        }
        if i != m {
            for j in m - 1..=n {
                let s = a[i][j];
                a[i][j] = a[m][j];
                a[m][j] = s;
            }
            for j in 1..=n {
                a[j].swap(i, m);
            }
        }
        if x != 0.0 {
            for i2 in m + 1..=n {
                let mut y = a[i2][m - 1];
                if y != 0.0 {
                    y /= x;
                    a[i2][m - 1] = y;
                    for j in m..=n {
                        a[i2][j] -= y * a[m][j];
                    }
                    for j in 1..=n {
                        a[j][m] += y * a[j][i2];
                    }
                }
            }
        }
    }
    for j in 1..=n {
        for i in j + 2..=n {
            a[i][j] = 0.0;
        }
    }
}

fn nrsign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

#[allow(unused_assignments)] // NR keeps every scalar function-scoped; kept as is for the line-by-line mirror
fn hqr_eigvals(a: &mut Mat8, wr: &mut [f64; NE + 1], wi: &mut [f64; NE + 1]) -> bool {
    let n = NE as isize;
    let mut anorm = 0.0;
    for i in 1..=NE {
        for j in (if i > 1 { i - 1 } else { 1 })..=NE {
            anorm += a[i][j].abs();
        }
    }
    macro_rules! at {
        ($i:expr, $j:expr) => {
            a[($i) as usize][($j) as usize]
        };
    }
    let mut nn: isize = n;
    let mut t = 0.0;
    let (mut x, mut y, mut z, mut w, mut p, mut q, mut r, mut s) =
        (0.0f64, 0.0f64, 0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    while nn >= 1 {
        let mut its = 0;
        loop {
            let mut l = nn;
            while l >= 2 {
                s = at!(l - 1, l - 1).abs() + at!(l, l).abs();
                if s == 0.0 {
                    s = anorm;
                }
                if at!(l, l - 1).abs() + s == s || at!(l, l - 1).abs() <= 1e-12 * s {
                    at!(l, l - 1) = 0.0;
                    break;
                }
                l -= 1;
            }
            x = at!(nn, nn);
            if l == nn {
                wr[nn as usize] = x + t;
                wi[nn as usize] = 0.0;
                nn -= 1;
            } else {
                y = at!(nn - 1, nn - 1);
                w = at!(nn, nn - 1) * at!(nn - 1, nn);
                if l == nn - 1 {
                    p = 0.5 * (y - x);
                    q = p * p + w;
                    z = q.abs().sqrt();
                    x += t;
                    let (k1, k2) = ((nn - 1) as usize, nn as usize);
                    if q >= 0.0 {
                        z = p + nrsign(z, p);
                        wr[k1] = x + z;
                        wr[k2] = x + z;
                        if z != 0.0 {
                            wr[k2] = x - w / z;
                        }
                        wi[k1] = 0.0;
                        wi[k2] = 0.0;
                    } else {
                        wr[k1] = x + p;
                        wr[k2] = x + p;
                        wi[k1] = -z;
                        wi[k2] = z;
                    }
                    nn -= 2;
                } else {
                    if its == 60 {
                        return false;
                    }
                    if its == 10 || its == 20 || its == 40 {
                        t += x;
                        for i in 1..=nn {
                            at!(i, i) -= x;
                        }
                        s = at!(nn, nn - 1).abs() + at!(nn - 1, nn - 2).abs();
                        x = 0.75 * s;
                        y = x;
                        w = -0.4375 * s * s;
                    }
                    its += 1;
                    let mut m = nn - 2;
                    while m >= l {
                        z = at!(m, m);
                        r = x - z;
                        s = y - z;
                        p = (r * s - w) / at!(m + 1, m) + at!(m, m + 1);
                        q = at!(m + 1, m + 1) - z - r - s;
                        r = at!(m + 2, m + 1);
                        s = 1.0 / (p.abs() + q.abs() + r.abs());
                        p *= s;
                        q *= s;
                        r *= s;
                        if m == l {
                            break;
                        }
                        let u = at!(m, m - 1).abs() * (q.abs() + r.abs());
                        let v =
                            p.abs() * (at!(m - 1, m - 1).abs() + z.abs() + at!(m + 1, m + 1).abs());
                        if u + v == v {
                            break;
                        }
                        m -= 1;
                    }
                    for i in m + 2..=nn {
                        at!(i, i - 2) = 0.0;
                        if i != m + 2 {
                            at!(i, i - 3) = 0.0;
                        }
                    }
                    let mut k = m;
                    while k < nn {
                        if k != m {
                            p = at!(k, k - 1);
                            q = at!(k + 1, k - 1);
                            r = 0.0;
                            if k != nn - 1 {
                                r = at!(k + 2, k - 1);
                            }
                            if 1e-280 < p * p + q * q + r * r && p * p + q * q + r * r < 1e280 {
                                x = 1.0;
                            } else {
                                x = p.abs() + q.abs() + r.abs();
                                if x != 0.0 {
                                    let ix = 1.0 / x;
                                    p *= ix;
                                    q *= ix;
                                    r *= ix;
                                }
                            }
                        }
                        s = nrsign((p * p + q * q + r * r).sqrt(), p);
                        if s != 0.0 {
                            if k == m {
                                if l != m {
                                    at!(k, k - 1) = -at!(k, k - 1);
                                }
                            } else {
                                at!(k, k - 1) = -s * x;
                            }
                            p += s;
                            let (is, ip) = (1.0 / s, 1.0 / p);
                            x = p * is;
                            y = q * is;
                            z = r * is;
                            q *= ip;
                            r *= ip;
                            for j in k..=nn {
                                p = at!(k, j) + q * at!(k + 1, j);
                                if k != nn - 1 {
                                    p += r * at!(k + 2, j);
                                    at!(k + 2, j) -= p * z;
                                }
                                at!(k + 1, j) -= p * y;
                                at!(k, j) -= p * x;
                            }
                            let mmin = if nn < k + 3 { nn } else { k + 3 };
                            for i in l..=mmin {
                                p = x * at!(i, k) + y * at!(i, k + 1);
                                if k != nn - 1 {
                                    p += z * at!(i, k + 2);
                                    at!(i, k + 2) -= p * r;
                                }
                                at!(i, k + 1) -= p * q;
                                at!(i, k) -= p;
                            }
                        }
                        k += 1;
                    }
                }
            }
            if !(l < nn - 1) {
                break;
            }
        }
    }
    true
}

// -------------------------------------------- geometry.jl: polar3, cayley_left, plane_frame, homography4, focal_pose_h
type V9 = [f64; 9];

fn polar3(h: &V9) -> V9 {
    let mut x = *h;
    for _ in 0..30 {
        let [x11, x12, x13, x21, x22, x23, x31, x32, x33] = x;
        let c = [
            x22 * x33 - x23 * x32,
            x23 * x31 - x21 * x33,
            x21 * x32 - x22 * x31,
            x13 * x32 - x12 * x33,
            x11 * x33 - x13 * x31,
            x12 * x31 - x11 * x32,
            x12 * x23 - x13 * x22,
            x13 * x21 - x11 * x23,
            x11 * x22 - x12 * x21,
        ];
        let d = x11 * c[0] + x12 * c[1] + x13 * c[2];
        if d == 0.0 {
            break;
        }
        let (mut nx, mut nc) = (0.0, 0.0);
        for i in 0..9 {
            nx += x[i] * x[i];
            nc += c[i] * c[i];
        }
        let g = ((nc / nx).sqrt() / d.abs()).sqrt();
        let (a, b) = (0.5 * g, 0.5 / (g * d));
        let xn: V9 = core::array::from_fn(|i| a * x[i] + b * c[i]);
        let mut dn = 0.0;
        for i in 0..9 {
            dn += (xn[i] - x[i]).abs();
        }
        x = xn;
        if dn < 1e-15 {
            break;
        }
    }
    x
}

fn cayley_left(s: [f64; 3], r: &V9) -> V9 {
    let [s1, s2, s3] = s;
    let n = s1 * s1 + s2 * s2 + s3 * s3;
    let q = 1.0 / (1.0 + n);
    let c = [
        q * (1.0 - n + 2.0 * s1 * s1),
        q * (2.0 * s1 * s2 - 2.0 * s3),
        q * (2.0 * s1 * s3 + 2.0 * s2),
        q * (2.0 * s1 * s2 + 2.0 * s3),
        q * (1.0 - n + 2.0 * s2 * s2),
        q * (2.0 * s2 * s3 - 2.0 * s1),
        q * (2.0 * s1 * s3 - 2.0 * s2),
        q * (2.0 * s2 * s3 + 2.0 * s1),
        q * (1.0 - n + 2.0 * s3 * s3),
    ];
    core::array::from_fn(|k| {
        let (i, j) = (k / 3, k % 3);
        c[3 * i] * r[j] + c[3 * i + 1] * r[3 + j] + c[3 * i + 2] * r[6 + j]
    })
}

// plane_frame(X): orthonormal (e1, e2, n), n the normal of the largest-area triangle of the 4 points
fn plane_frame(x: &[[f64; 3]; 4]) -> ([f64; 3], [f64; 3], [f64; 3]) {
    let sub = |a: &[f64; 3], b: &[f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let (mut bi, mut bj, mut bm, mut bn) = (0, 1, 2, -1.0);
    for i in 0..4 {
        for j in i + 1..4 {
            for m in j + 1..4 {
                let c = cross3(&sub(&x[j], &x[i]), &sub(&x[m], &x[i]));
                let nc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                if nc > bn {
                    (bi, bj, bm, bn) = (i, j, m, nc);
                }
            }
        }
    }
    let d = sub(&x[bj], &x[bi]);
    let c = cross3(&d, &sub(&x[bm], &x[bi]));
    let (sn, sd) = (bn.sqrt(), (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
    let n = [c[0] / sn, c[1] / sn, c[2] / sn];
    let e1 = [d[0] / sd, d[1] / sd, d[2] / sd];
    (e1, cross3(&n, &e1), n)
}

// homography4(ab, uv): (H row-major, the rank ratio) with uv ~ H [a; b; 1] from 4 points
fn homography4(ab: &[[f64; 2]; 4], uv: &[[f64; 2]; 4]) -> (V9, f64) {
    let mut w = [[0.0; 8]; 9]; // columns = the 8 equations
    for i in 0..4 {
        let ([a, b], [u, v]) = (ab[i], uv[i]);
        let (c, d) = (2 * i, 2 * i + 1);
        (w[0][c], w[1][c], w[2][c]) = (a, b, 1.0);
        (w[3][d], w[4][d], w[5][d]) = (a, b, 1.0);
        (w[6][c], w[7][c], w[8][c]) = (-u * a, -u * b, -u);
        (w[6][d], w[7][d], w[8][d]) = (-v * a, -v * b, -v);
    }
    let (n, ratio): ([[f64; 1]; 9], f64) = house_nullspace(&mut w, false, None);
    (core::array::from_fn(|k| n[k][0]), ratio)
}

// focal_pose_h(H): the plane-to-image pose (Rp, t) and f, or None; omega = 1 / f^2 by least squares on the 2 metric
// constraints
fn focal_pose_h(h: &V9) -> Option<(V9, [f64; 3], f64)> {
    let [h11, h12, h13, h21, h22, h23, h31, h32, h33] = *h;
    let (a1, b1) = (h11 * h12 + h21 * h22, h31 * h32);
    let (a2, b2) = (
        (h11 * h11 + h21 * h21) - (h12 * h12 + h22 * h22),
        h31 * h31 - h32 * h32,
    );
    let om = -(a1 * b1 + a2 * b2) / (a1 * a1 + a2 * a2);
    if !(om > 0.0 && om.is_finite()) {
        return None;
    }
    let w = om.sqrt(); // 1 / f
    let (g11, g21, g12, g22) = (h11 * w, h21 * w, h12 * w, h22 * w);
    let lam = (1.0
        / ((g11 * g11 + g21 * g21 + h31 * h31).sqrt()
            * (g12 * g12 + g22 * g22 + h32 * h32).sqrt())
        .sqrt())
    .copysign(h33);
    let (r1, r2) = (
        [lam * g11, lam * g21, lam * h31],
        [lam * g12, lam * g22, lam * h32],
    );
    let c = cross3(&r1, &r2);
    let sc = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let n = [c[0] / sc, c[1] / sc, c[2] / sc];
    let rp = polar3(&[r1[0], r2[0], n[0], r1[1], r2[1], n[1], r1[2], r2[2], n[2]]);
    let t = [lam * (h13 * w), lam * (h23 * w), lam * h33];
    if !(rp.iter().all(|v| v.is_finite()) && t.iter().all(|v| v.is_finite())) {
        return None;
    }
    Some((rp, t, 1.0 / w))
}

// -------------------------------------------- e3q3_kernel.jl 6b: the complex E3Q3 lane
type Q10 = [f64; 10];

// ROTATIONS[1] = Float64.(cayley_rotation([1/3, -1/4, 1/5])) (exact rational, rounded)
const R1: [[f64; 3]; 3] = [
    [
        0.8310826276035707,
        -0.4669260700389105,
        -0.30212863355458913,
    ],
    [0.19226367589837492, 0.7509727626459144, -0.6317235065232318],
    [0.5218585488670177, 0.4669260700389105, 0.7138933394369421],
];
const HIDDEN_COLS: [usize; 7] = [6, 2, 7, 4, 9, 8, 5]; // x, xz, y, yz, 1, z, z^2 (0-based MONOMIALS)

// dot_quadric(p, oa, sa, ob, sb): a(x) . b(x) of two 3-row blocks linear in [x; 1], p 0-based
fn dot_quadric(p: &[f64; 48], oa: usize, sa: usize, ob: usize, sb: usize) -> Q10 {
    let m = |c: usize, d: usize| {
        p[oa + sa * (c - 1)] * p[ob + sb * (d - 1)]
            + p[oa + 1 + sa * (c - 1)] * p[ob + 1 + sb * (d - 1)]
            + p[oa + 2 + sa * (c - 1)] * p[ob + 2 + sb * (d - 1)]
    };
    [
        m(1, 1),
        m(1, 2) + m(2, 1),
        m(1, 3) + m(3, 1),
        m(2, 2),
        m(2, 3) + m(3, 2),
        m(3, 3),
        m(1, 4) + m(4, 1),
        m(2, 4) + m(4, 2),
        m(3, 4) + m(4, 3),
        m(4, 4),
    ]
}

// _quadric_linear(q, W) with W = R1' (col(j) = row j of R1), then _unit
fn quadric_linear_unit(q: &Q10) -> Q10 {
    let s3 = [
        [q[0], q[1] / 2.0, q[2] / 2.0],
        [q[1] / 2.0, q[3], q[4] / 2.0],
        [q[2] / 2.0, q[4] / 2.0, q[5]],
    ];
    let sw = |i: usize, j: usize| s3[i][0] * R1[j][0] + s3[i][1] * R1[j][1] + s3[i][2] * R1[j][2];
    let s = |i: usize, j: usize| R1[i][0] * sw(0, j) + R1[i][1] * sw(1, j) + R1[i][2] * sw(2, j);
    let l = |j: usize| q[6] * R1[j][0] + q[7] * R1[j][1] + q[8] * R1[j][2];
    let o = [
        s(0, 0),
        2.0 * s(0, 1),
        2.0 * s(0, 2),
        s(1, 1),
        2.0 * s(1, 2),
        s(2, 2),
        l(0),
        l(1),
        l(2),
        q[9],
    ];
    let mut ss = o[0] * o[0];
    for k in 1..10 {
        ss += o[k] * o[k];
    }
    let n = ss.sqrt();
    o.map(|v| v / n)
}

// _pm (product truncated to degree 4)
fn pm(p: &P5, q: &P5) -> P5 {
    core::array::from_fn(|k| {
        let mut s = p[0] * q[k];
        for i in 1..=k {
            s += p[i] * q[k - i];
        }
        s
    })
}

// rows[i][j]: row i, column j (x, y, 1) of M(z), column degrees (2, 2, 4)
type Rows = [[P5; 3]; 3];

// hidden_rows(S), S = -Q \ rhs (3 x 7 used)
fn hidden_rows(s: &[[f64; 10]; 3]) -> Rows {
    let a: [P5; 3] = core::array::from_fn(|i| [s[i][0], s[i][1], 0.0, 0.0, 0.0]);
    let b: [P5; 3] = core::array::from_fn(|i| [s[i][2], s[i][3], 0.0, 0.0, 0.0]);
    let c: [P5; 3] = core::array::from_fn(|i| [s[i][4], s[i][5], s[i][6], 0.0, 0.0]);
    let op = |f: &dyn Fn(usize) -> f64| -> P5 { core::array::from_fn(f) };
    let (t1, t2) = (pm(&a[1], &b[1]), pm(&a[2], &b[0]));
    let a1 = op(&|k| t1[k] - t2[k] + c[1][k]);
    let (t1, t2, t3, t4) = (
        pm(&a[1], &b[0]),
        pm(&b[1], &b[1]),
        pm(&a[0], &b[1]),
        pm(&b[0], &b[2]),
    );
    let b1 = op(&|k| t1[k] + t2[k] - t3[k] - t4[k] - c[0][k]);
    let d = op(&|k| b[1][k] - a[0][k]);
    let (t1, t2, t3) = (pm(&a[1], &c[0]), pm(&d, &c[1]), pm(&b[0], &c[2]));
    let c1 = op(&|k| t1[k] + t2[k] - t3[k]);
    let d = op(&|k| b[2][k] - a[1][k]);
    let (t1, t2, t3) = (pm(&a[2], &a[0]), pm(&d, &a[1]), pm(&b[1], &a[2]));
    let a2 = op(&|k| t1[k] + t2[k] - t3[k] + c[2][k]);
    let (t1, t2) = (pm(&a[2], &b[0]), pm(&a[1], &b[1]));
    let b2 = op(&|k| t1[k] - t2[k] - c[1][k]);
    let (t1, t2, t3) = (pm(&a[2], &c[0]), pm(&d, &c[1]), pm(&b[1], &c[2]));
    let c2 = op(&|k| t1[k] + t2[k] - t3[k]);
    let (t1, t2) = (pm(&a1, &a[1]), pm(&b1, &a[2]));
    let a3 = op(&|k| t1[k] + t2[k]);
    let (t1, t2) = (pm(&a1, &b[1]), pm(&b1, &b[2]));
    let b3 = op(&|k| t1[k] + t2[k] + c1[k]);
    let (t1, t2) = (pm(&a1, &c[1]), pm(&b1, &c[2]));
    let c3 = op(&|k| t1[k] + t2[k]);
    let dd = a1[2] * b2[2] - a2[2] * b1[2];
    let al = (a3[3] * b2[2] - a2[2] * b3[3]) / dd;
    let ga = (a1[2] * b3[3] - a3[3] * b1[2]) / dd;
    let zsh = |p: &P5, k: usize| if k == 0 { 0.0 } else { p[k - 1] };
    let r3 = |x3: &P5, x1: &P5, x2: &P5| -> P5 {
        core::array::from_fn(|k| x3[k] - al * zsh(x1, k) - ga * zsh(x2, k))
    };
    [
        [a1, b1, c1],
        [a2, b2, c2],
        [r3(&a3, &a1, &a2), r3(&b3, &b1, &b2), r3(&c3, &c1, &c2)],
    ]
}

// _hidden_eigs(cs): the rows of M(z) and the 8 eigenvalues of the companion; None when the pencil does not answer
#[allow(clippy::type_complexity)]
fn hidden_eigs(cs: &[Q10; 3]) -> Option<(Rows, [f64; NE + 1], [f64; NE + 1])> {
    let c: [Q10; 3] = core::array::from_fn(|k| quadric_linear_unit(&cs[k]));
    let mut q: [[f64; 3]; 3] = core::array::from_fn(|k| [c[k][0], c[k][1], c[k][3]]);
    let mut x: [[f64; 10]; 3] = core::array::from_fn(|k| {
        core::array::from_fn(|j| {
            if j < 7 {
                -c[k][HIDDEN_COLS[j]]
            } else {
                (k == j - 7) as i32 as f64
            }
        })
    });
    let colsum = |m: &[[f64; 3]; 3], j: usize| m[0][j].abs() + m[1][j].abs() + m[2][j].abs();
    let n1 = jmax(jmax(colsum(&q, 0), colsum(&q, 1)), colsum(&q, 2));
    if !small_solve(&mut q, &mut x) {
        return None;
    }
    let xs = |j: usize| x[0][j].abs() + x[1][j].abs() + x[2][j].abs();
    if !(n1 * jmax(jmax(xs(7), xs(8)), xs(9)) < 1e8) {
        return None;
    }
    let rows = hidden_rows(&x);
    let mut l: [[f64; 3]; 3] =
        core::array::from_fn(|i| core::array::from_fn(|j| rows[i][j][if j == 2 { 4 } else { 2 }]));
    let mut kk: [[f64; 8]; 3] = core::array::from_fn(|i| {
        let r = &rows[i];
        [
            r[0][0], r[0][1], r[1][0], r[1][1], r[2][0], r[2][1], r[2][2], r[2][3],
        ]
    });
    if !small_solve(&mut l, &mut kk) || !kk.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let mut t: Mat8 = [[0.0; NE + 1]; NE + 1];
    t[1][2] = 1.0;
    t[3][4] = 1.0;
    t[5][6] = 1.0;
    t[6][7] = 1.0;
    t[7][8] = 1.0;
    for j in 1..=8 {
        t[2][j] = -kk[0][j - 1];
        t[4][j] = -kk[1][j - 1];
        t[8][j] = -kk[2][j - 1];
    }
    hessenberg_gauss(&mut t);
    let (mut wr, mut wi) = ([0.0; NE + 1], [0.0; NE + 1]);
    if !hqr_eigvals(&mut t, &mut wr, &mut wi) {
        return None;
    }
    Some((rows, wr, wi))
}

fn cross3<T: Sc>(a: &[T; 3], b: &[T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn sum_abs2<T: Sc>(v: &[T; 3]) -> f64 {
    v[0].abs2() + v[1].abs2() + v[2].abs2()
}

// _pencil_root(W, r, z): (x, y) from the largest cross product of the rows of M(z), mapped back by W = R1'
fn pencil_root<T: Sc>(rw: &Rows, z: T) -> Option<[T; 3]> {
    let m: [[T; 3]; 3] = core::array::from_fn(|i| core::array::from_fn(|j| z.evalpoly(&rw[i][j])));
    let cand = [
        cross3(&m[0], &m[1]),
        cross3(&m[0], &m[2]),
        cross3(&m[1], &m[2]),
    ];
    let (mut best, mut fb) = (0, sum_abs2(&cand[0]));
    for k in 1..3 {
        let fk = sum_abs2(&cand[k]);
        if (!fb.is_nan() && fk.is_nan()) || fb < fk {
            // isless(fb, fk)
            best = k;
            fb = fk;
        }
    }
    let v = &cand[best];
    if !(v[2].mag() > 1e-10 * sum_abs2(v).sqrt()) {
        return None;
    }
    let u = [v[0] / v[2], v[1] / v[2], z];
    Some(core::array::from_fn(|i| {
        u[0].scale(R1[0][i]) + u[1].scale(R1[1][i]) + u[2].scale(R1[2][i])
    }))
}

// _qeval(q, X, Y, Z): value, componentwise weight, gradient
fn qeval<T: Sc>(q: &Q10, x: T, y: T, z: T) -> (T, f64, [T; 3]) {
    let f = (x * x).scale(q[0])
        + x.scale(q[1]) * y
        + x.scale(q[2]) * z
        + (y * y).scale(q[3])
        + y.scale(q[4]) * z
        + (z * z).scale(q[5])
        + x.scale(q[6])
        + y.scale(q[7])
        + z.scale(q[8]);
    let f = f.addr(q[9]);
    let g = [
        (x.scale(2.0 * q[0]) + y.scale(q[1]) + z.scale(q[2])).addr(q[6]),
        (x.scale(q[1]) + y.scale(2.0 * q[3]) + z.scale(q[4])).addr(q[7]),
        (x.scale(q[2]) + y.scale(q[4]) + z.scale(2.0 * q[5])).addr(q[8]),
    ];
    (f, T::weight(q, x, y, z), g)
}

fn bwd<T: Sc>(cs: &[Q10; 3], x: &[T; 3]) -> f64 {
    let e = |k: usize| {
        let (f, w, _) = qeval(&cs[k], x[0], x[1], x[2]);
        if f.is_zero() {
            0.0
        } else {
            f.mag() / w
        }
    };
    jmax(jmax(e(0), e(1)), e(2))
}

// _qnewton(cs, x, iters): Newton by Cramer's rule while the backward error drops; returns the backward error
fn qnewton<T: Sc>(cs: &[Q10; 3], x: &mut [T; 3], iters: usize) -> f64 {
    let mut e = bwd(cs, x);
    for _ in 0..iters {
        if e <= 4.0 * f64::EPSILON {
            break;
        }
        let (f1, _, g1) = qeval(&cs[0], x[0], x[1], x[2]);
        let (f2, _, g2) = qeval(&cs[1], x[0], x[1], x[2]);
        let (f3, _, g3) = qeval(&cs[2], x[0], x[1], x[2]);
        let (c23, c31, c12) = (cross3(&g2, &g3), cross3(&g3, &g1), cross3(&g1, &g2));
        let d = g1[0] * c23[0] + g1[1] * c23[1] + g1[2] * c23[2];
        if d.is_zero() {
            break;
        }
        let y: [T; 3] =
            core::array::from_fn(|i| x[i] - (c23[i] * f1 + c31[i] * f2 + c12[i] * f3) / d);
        if !y.iter().all(|v| v.finite()) {
            break;
        }
        let ey = bwd(cs, &y);
        if !(ey < e) {
            break;
        }
        *x = y;
        e = ey;
    }
    e
}

type Root = [Cx; 3];

// e3q3_complex_seeds(cs; keep = _near_real): the near-real roots into roots, returns (n, their number); n = the
// number of roots the pencil gave (0: the pencil failed or every root is at infinity). A root that does not polish is
// kept (the caller gates every seed); the distinct / pair-split checks only feed the kernel's `why`: not ported.
fn e3q3_complex_seeds(cs: &[Q10; 3], roots: &mut [Root; 8]) -> (usize, usize) {
    let Some((rw, wr, wi)) = hidden_eigs(cs) else {
        return (0, 0);
    };
    let (mut n, mut nk) = (0, 0);
    for k in 1..=8 {
        if wi[k] < 0.0 {
            continue;
        }
        let y: Root = if wi[k] == 0.0 {
            let Some(mut x) = pencil_root(&rw, wr[k]) else {
                continue;
            };
            qnewton(cs, &mut x, 6);
            x.map(|v| v.to_cx())
        } else {
            let Some(mut x) = pencil_root(
                &rw,
                Cx {
                    re: wr[k],
                    im: wi[k],
                },
            ) else {
                continue;
            };
            qnewton(cs, &mut x, 6);
            x
        };
        let pair = [
            y,
            y.map(|v| Cx {
                re: v.re,
                im: -v.im,
            }),
        ];
        for x in &pair[..if wi[k] > 0.0 { 2 } else { 1 }] {
            // y, then its conjugate; _near_real: |Im x| <= NEAR_REAL (1 + |Re x|), max over coordinates
            n += 1;
            let mi = jmax(jmax(x[0].im.abs(), x[1].im.abs()), x[2].im.abs());
            let mr = jmax(jmax(x[0].re.abs(), x[1].re.abs()), x[2].re.abs());
            if mi <= 0.1 * (1.0 + mr) && nk < 8 {
                // hqr pairs: at most 8 roots
                roots[nk] = *x;
                nk += 1;
            }
        }
    }
    (n, nk)
}

// -------------------------------------------- p4pf_e3q3.jl section 6: front end, recover, polish, solve
const REPEATED: f64 = 1e-8; // normalized world points closer than this: declined
const PLANAR: f64 = 1e-2; // the planar seed runs when the coplanarity ratio rc is at most this

struct Fe {
    p: [f64; 48], // vec(B) column-major, B the 12 x 4 null basis of the DLT system
    xn: [[f64; 3]; 4],
    uvn: [[f64; 2]; 4],
    c: [f64; 3],
    s: f64,
    k: f64,
    rc: f64,
}

fn frontend(x: &[[f64; 3]; 4], uv: &[[f64; 2]; 4]) -> Option<Fe> {
    if !x
        .iter()
        .flatten()
        .chain(uv.iter().flatten())
        .all(|v| v.is_finite())
    {
        return None;
    }
    let c: [f64; 3] = core::array::from_fn(|j| (((x[0][j] + x[1][j]) + x[2][j]) + x[3][j]) / 4.0);
    let (mut s, mut k, mut mx) = (0.0, 0.0, 0.0);
    for i in 0..4 {
        let d = [x[i][0] - c[0], x[i][1] - c[1], x[i][2] - c[2]];
        let (nd, nu, nx) = (jnorm(&d), jnorm(&uv[i]), jnorm(&x[i]));
        s = if i == 0 { nd } else { s + nd };
        k = if i == 0 { nu } else { k + nu };
        mx = if i == 0 { nx } else { jmax(mx, nx) };
    }
    s /= 4.0;
    k /= 4.0;
    if !(s > 1e-12 * mx && k > 0.0) {
        return None;
    }
    let xn: [[f64; 3]; 4] =
        core::array::from_fn(|i| core::array::from_fn(|j| (x[i][j] - c[j]) / s));
    let uvn: [[f64; 2]; 4] = core::array::from_fn(|i| core::array::from_fn(|j| uv[i][j] / k));
    for i in 1..4 {
        // repeated world points
        for j in 0..i {
            let (dx, dy, dz) = (
                xn[i][0] - xn[j][0],
                xn[i][1] - xn[j][1],
                xn[i][2] - xn[j][2],
            );
            if !(dx * dx + dy * dy + dz * dz > REPEATED * REPEATED) {
                return None;
            }
        }
    }
    // columns: the homogeneous points; the 8 DLT equations, point by point
    let (mut w, mut w12) = ([[0.0; 4]; 4], [[0.0; 8]; 12]);
    for i in 0..4 {
        let (u, v) = (uvn[i][0], uvn[i][1]);
        for j in 0..4 {
            let xj = if j < 3 { xn[i][j] } else { 1.0 };
            w[j][i] = xj;
            w12[j][2 * i] = xj;
            w12[8 + j][2 * i] = -u * xj;
            w12[4 + j][2 * i + 1] = xj;
            w12[8 + j][2 * i + 1] = -v * xj;
        }
    }
    let (_, rc): ([[f64; 0]; 4], f64) = house_nullspace(&mut w, true, Some(PLANAR)); // gates only
    let (b, ra): ([[f64; 4]; 12], f64) = house_nullspace(&mut w12, false, Some(1e-10));
    if !(ra > 1e-10) {
        return None; // collinear points
    }
    let p: [f64; 48] = core::array::from_fn(|m| b[m % 12][m / 12]);
    Some(Fe {
        p,
        xn,
        uvn,
        c,
        s,
        k,
        rc,
    })
}

fn det9(r: &V9) -> f64 {
    r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6])
        + r[2] * (r[3] * r[7] - r[4] * r[6])
}
fn orth_err(r: &V9) -> f64 {
    let e = [
        (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - 1.0).abs(),
        (r[3] * r[3] + r[4] * r[4] + r[5] * r[5] - 1.0).abs(),
        (r[6] * r[6] + r[7] * r[7] + r[8] * r[8] - 1.0).abs(),
        (r[0] * r[3] + r[1] * r[4] + r[2] * r[5]).abs(),
        (r[0] * r[6] + r[1] * r[7] + r[2] * r[8]).abs(),
        (r[3] * r[6] + r[4] * r[7] + r[5] * r[8]).abs(),
    ];
    let mut m = e[0];
    for i in 1..6 {
        m = jmax(m, e[i]);
    }
    m
}
fn rot(r: &V9, x: &[f64; 3], i: usize) -> f64 {
    r[3 * i] * x[0] + r[3 * i + 1] * x[1] + r[3 * i + 2] * x[2]
}

// recover(fe, x): normalized-unit (R, t, f) from a real chart point; None where Julia returns nothing, and where it
// would take the SVD fallback (polar3 not a rotation within 1e-12; simplified, see README)
fn recover(fe: &Fe, x: &[f64; 3]) -> Option<(V9, [f64; 3], f64)> {
    let p = &fe.p;
    let row = |o: usize, s: usize, j: usize| {
        p[o + j] * x[0] + p[o + j + s] * x[1] + p[o + j + 2 * s] * x[2] + p[o + j + 3 * s]
    };
    let a: [f64; 4] = core::array::from_fn(|j| row(0, 12, j));
    let b: [f64; 4] = core::array::from_fn(|j| row(4, 12, j));
    let c: [f64; 4] = core::array::from_fn(|j| row(8, 12, j));
    let sg = if det9(&[a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]]) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let w = sg / (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
    let fx = w.abs() * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let fy = w.abs() * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    let f = (fx + fy) / 2.0;
    let h = [
        w * a[0] / fx,
        w * a[1] / fx,
        w * a[2] / fx,
        w * b[0] / fy,
        w * b[1] / fy,
        w * b[2] / fy,
        w * c[0],
        w * c[1],
        w * c[2],
    ];
    if !(h.iter().all(|v| v.is_finite()) && f.is_finite() && f > 0.0) {
        return None;
    }
    let r = polar3(&h);
    if !(orth_err(&r) <= 1e-12 && det9(&r) > 0.0) {
        return None;
    }
    let (mut s1, mut s2, mut s11, mut b1, mut b2, mut b3) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..4 {
        let xi = &fe.xn[i];
        let (y1, y2, y3) = (rot(&r, xi, 0), rot(&r, xi, 1), rot(&r, xi, 2));
        let (w1, w2) = (fe.uvn[i][0] / f, fe.uvn[i][1] / f);
        let (r1, r2) = (w1 * y3 - y1, w2 * y3 - y2);
        s1 += w1;
        s2 += w2;
        s11 += w1 * w1 + w2 * w2;
        b1 += r1;
        b2 += r2;
        b3 -= w1 * r1 + w2 * r2;
    }
    let mut g = [[4.0, 0.0, -s1], [0.0, 4.0, -s2], [-s1, -s2, s11]];
    let mut tb = [[b1], [b2], [b3]];
    if !small_solve(&mut g, &mut tb) {
        return None;
    }
    Some((r, [tb[0][0], tb[1][0], tb[2][0]], f))
}

// _reprojection!(r, J, R, t, f, Xn, uvn): the 8 residuals (+ Jacobian in (w, t, f) when j is Some), |r|^2
fn reprojection(
    res: &mut [f64; 8],
    mut jac: Option<&mut [[f64; 7]; 8]>,
    r: &V9,
    t: &[f64; 3],
    f: f64,
    fe: &Fe,
) -> f64 {
    let mut e = 0.0;
    for i in 0..4 {
        let x = &fe.xn[i];
        let (a1, a2, a3) = (rot(r, x, 0), rot(r, x, 1), rot(r, x, 2));
        let y3 = a3 + t[2];
        let (m1, m2, s) = ((a1 + t[0]) / y3, (a2 + t[1]) / y3, f / y3);
        res[2 * i] = f * m1 - fe.uvn[i][0];
        res[2 * i + 1] = f * m2 - fe.uvn[i][1];
        e += res[2 * i] * res[2 * i] + res[2 * i + 1] * res[2 * i + 1];
        if let Some(j) = jac.as_deref_mut() {
            j[2 * i] = [
                -s * m1 * a2,
                s * (a3 + m1 * a1),
                -s * a2,
                s,
                0.0,
                -s * m1,
                m1,
            ];
            j[2 * i + 1] = [
                -s * (a3 + m2 * a2),
                s * m2 * a1,
                s * a1,
                0.0,
                s,
                -s * m2,
                m2,
            ];
        }
    }
    e
}

const STEP_TOL: f64 = 1e-10; // polish stops after an accepted step below it (rad, |dt| / |t|_1, |df| / |f|)

// polish: Gauss-Newton on the 8 residuals (7 dof), decreasing steps only; returns max reprojection error
fn polish(r: &mut V9, t: &mut [f64; 3], f: &mut f64, fe: &Fe, iters: usize) -> f64 {
    let (mut res, mut res2, mut jac) = ([0.0; 8], [0.0; 8], [[0.0; 7]; 8]);
    let mut e = reprojection(&mut res, Some(&mut jac), r, t, *f, fe);
    for _ in 0..iters {
        let (mut h, mut g) = ([[0.0; 7]; 7], [[0.0; 1]; 7]);
        for a in 0..7 {
            let mut ga = 0.0;
            for k in 0..8 {
                ga += jac[k][a] * res[k];
            }
            g[a][0] = ga;
            for b in a..7 {
                let mut hh = 0.0;
                for k in 0..8 {
                    hh += jac[k][a] * jac[k][b];
                }
                h[a][b] = hh;
                h[b][a] = hh;
            }
        }
        if !small_solve(&mut h, &mut g) || !g.iter().all(|v| v[0].is_finite()) {
            break;
        }
        let r2 = cayley_left([-g[0][0] / 2.0, -g[1][0] / 2.0, -g[2][0] / 2.0], r);
        let t2 = [t[0] - g[3][0], t[1] - g[4][0], t[2] - g[5][0]];
        let f2 = *f - g[6][0];
        let e2 = reprojection(&mut res2, None, &r2, &t2, f2, fe);
        if !(e2 < e) {
            break;
        }
        *r = r2;
        *t = t2;
        *f = f2;
        e = e2;
        if jmax(jmax(g[0][0].abs(), g[1][0].abs()), g[2][0].abs()) <= STEP_TOL
            && jmax(jmax(g[3][0].abs(), g[4][0].abs()), g[5][0].abs())
                <= STEP_TOL * ((t[0].abs() + t[1].abs()) + t[2].abs())
            && g[6][0].abs() <= STEP_TOL * f.abs()
        {
            res = res2; // a step below STEP_TOL: stop, res2 = the residuals here
            break;
        }
        reprojection(&mut res, Some(&mut jac), r, t, *f, fe);
    }
    let mut m = jhypot(res[0], res[1]);
    for i in 1..4 {
        m = jmax(m, jhypot(res[2 * i], res[2 * i + 1]));
    }
    m
}

const REPROJ_TOL: f64 = 5e-2;
const POLISH_ITERS: usize = 10;
const SEED_FRONT: usize = 3; // seeds with 2+ points behind the camera are not polished
const PERSPECTIVE: f64 = 1e-2; // depth spread (max - min) / mean >= PERSPECTIVE x reproj (near-affine gate)

/// Output capacity: up to 8 lane candidates + the planar seed.
pub const MAX_POSES: usize = 9;

// _seed!: one seed (R, t, f) in normalized units -> seed screen, polish, gates; a candidate into found
fn seed(
    fe: &Fe,
    (mut r, mut t, mut f): (V9, [f64; 3], f64),
    found: &mut [Pose; MAX_POSES],
    nf: &mut usize,
) {
    // seed screen: at most 1 point behind the camera
    let mut nfront = 0;
    for xi in &fe.xn {
        nfront += usize::from(rot(&r, xi, 2) + t[2] > 0.0);
    }
    if nfront < SEED_FRONT {
        return;
    }
    let reproj = polish(&mut r, &mut t, &mut f, fe, POLISH_ITERS);
    if !(f > 0.0 && reproj <= REPROJ_TOL) {
        return;
    }
    let y: [f64; 4] = core::array::from_fn(|i| rot(&r, &fe.xn[i], 2) + t[2]);
    if !y.iter().all(|&yi| yi > 0.0) {
        return; // 4 points in front
    }
    // all > 0: plain comparisons = Julia's max / min
    let (ymax, ymin) = y[1..].iter().fold((y[0], y[0]), |(a, b), &yi| {
        (if yi > a { yi } else { a }, if yi < b { yi } else { b })
    });
    if !(ymax - ymin >= PERSPECTIVE * reproj * ((((y[0] + y[1]) + y[2]) + y[3]) / 4.0)) {
        return;
    }
    found[*nf] = Pose {
        r,
        t: core::array::from_fn(|i| fe.s * t[i] - rot(&r, &fe.c, i)),
        f: f * fe.k,
        reproj,
    };
    *nf += 1;
}

// planar_seed(fe): plane_frame, the plane coordinates, homography4 (rank test > 1e-10), focal_pose_h; R = Rp [e1 e2 n]'
fn planar_seed(fe: &Fe) -> Option<(V9, [f64; 3], f64)> {
    let (e1, e2, n) = plane_frame(&fe.xn);
    let ab: [[f64; 2]; 4] = core::array::from_fn(|i| {
        let x = &fe.xn[i];
        [
            e1[0] * x[0] + e1[1] * x[1] + e1[2] * x[2],
            e2[0] * x[0] + e2[1] * x[1] + e2[2] * x[2],
        ]
    });
    let (h, ra) = homography4(&ab, &fe.uvn);
    if !(ra > 1e-10) {
        return None;
    }
    let (rp, t, f) = focal_pose_h(&h)?;
    let r = core::array::from_fn(|k| {
        let (a, b) = (k / 3, k % 3);
        rp[3 * a] * e1[b] + rp[3 * a + 1] * e2[b] + rp[3 * a + 2] * n[b]
    });
    Some((r, t, f))
}

/// P4Pf: 4 world points `x`, 4 principal-point-centered image points `uv` (any units) -> poses in `out` (sorted by
/// `reproj`), returns their number; `status` = [`FRONT_END`] / [`LANE`] / [`LANE_DECLINED`].
pub fn p4pf(
    x: &[[f64; 3]; 4],
    uv: &[[f64; 2]; 4],
    out: &mut [Pose; MAX_POSES],
    status: &mut i32,
) -> usize {
    let Some(fe) = frontend(x, uv) else {
        *status = FRONT_END;
        return 0;
    };
    let cs = [
        dot_quadric(&fe.p, 0, 12, 4, 12),
        dot_quadric(&fe.p, 0, 12, 8, 12),
        dot_quadric(&fe.p, 4, 12, 8, 12),
    ];
    let mut roots = [[Cx::default(); 3]; 8];
    let (n8, nr) = e3q3_complex_seeds(&cs, &mut roots);
    if n8 == 0 {
        *status = LANE_DECLINED;
        return 0;
    }
    *status = LANE;
    let mut found = [Pose::default(); MAX_POSES];
    let mut nf = 0;
    for root in &roots[..nr] {
        // _candidates!: recover, then _seed!
        if let Some(c0) = recover(&fe, &root.map(|v| v.re)) {
            seed(&fe, c0, &mut found, &mut nf);
        }
    }
    if fe.rc <= PLANAR {
        if let Some(c0) = planar_seed(&fe) {
            seed(&fe, c0, &mut found, &mut nf);
        }
    }
    for i in 1..nf {
        // stable sort by reproj (Julia's default sort is stable)
        let mut j = i;
        while j > 0 && found[j].reproj < found[j - 1].reproj {
            found.swap(j, j - 1);
            j -= 1;
        }
    }
    let mut n = 0;
    for q in &found[..nf] {
        // dedup, keeping the lowest-residual copy; norm(c.R - q.R) iterates column-major
        let dup = out[..n].iter().any(|c| {
            let d: [f64; 9] =
                core::array::from_fn(|k| c.r[3 * (k % 3) + k / 3] - q.r[3 * (k % 3) + k / 3]);
            jnorm(&d) + (c.f - q.f).abs() / q.f <= 1e-6
        });
        if !dup {
            out[n] = *q;
            n += 1;
        }
    }
    n
}
