//! Lens models for the rasterizers: how a camera-space point maps to the
//! normalized image coordinates in front of the focal lengths, and the 2×3
//! Jacobian the EWA splatting linearises with.
//!
//! A [`Lens`] is a projection, the pinhole's `(x, y) / z` or the equidistant
//! `θ (x, y) / r` (`r = |(x, y)|`, `θ = atan2(r, z)`), followed by COLMAP's
//! distortion of the projected coordinates `(u, v)` ([`distort`]). COLMAP's
//! `SIMPLE_RADIAL`, `RADIAL` and `OPENCV` (pinhole) and `SIMPLE_RADIAL_FISHEYE`,
//! `RADIAL_FISHEYE`, `OPENCV_FISHEYE` and `THIN_PRISM_FISHEYE` (equidistant)
//! are all of this form. The pixel is `(fx u' + cx, fy v' + cy)`.
//!
//! The math is generic over [`LensScalar`]: `f32` in the renderer, `f64` and
//! dual numbers in the gradient oracle ([`crate::backward`]), whose duals then
//! carry the Jacobian's own derivatives.

use std::f64::consts::{FRAC_PI_2, PI};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// How rays map to the image plane before the distortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Projection {
    /// `(x, y) / z`.
    #[default]
    Pinhole,
    /// `θ (x, y) / r` with `r = |(x, y)|` and `θ = atan2(r, z)`: the
    /// Kannala-Brandt / OpenCV fisheye base, which images rays past 90° too.
    Equidistant,
}

/// Distortion coefficients per lens.
pub const LENS_COEFFS: usize = 8;

/// A lens: its [`Projection`], the distortion coefficients
/// `[k1, k2, k3, k4, p1, p2, sx1, sy1]` (see [`distort`]) and the widest ray
/// angle from the optical axis it is followed to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lens {
    pub projection: Projection,
    pub coeffs: [f32; LENS_COEFFS],
    /// The lens is trusted out to this angle from the optical axis
    /// (radians): past the edge of the image, or where the distortion
    /// polynomial folds back ([`Lens::fit_max_theta`]). Gaussians centred
    /// farther out are drawn through its first-order extension there
    /// ([`extended_lens_coords`]).
    pub max_theta: f32,
}

impl Default for Lens {
    fn default() -> Self {
        Self::PINHOLE
    }
}

impl Lens {
    /// The undistorted pinhole.
    pub const PINHOLE: Self = Self {
        projection: Projection::Pinhole,
        coeffs: [0.0; LENS_COEFFS],
        max_theta: FRAC_PI_2 as f32,
    };

    /// A lens of `projection` with `coeffs`, drawing every ray the
    /// projection images until [`Lens::fit_max_theta`] narrows it.
    pub fn new(projection: Projection, coeffs: [f32; LENS_COEFFS]) -> Self {
        Self {
            projection,
            coeffs,
            max_theta: match projection {
                Projection::Pinhole => FRAC_PI_2 as f32,
                Projection::Equidistant => PI as f32,
            },
        }
    }

    /// The undistorted pinhole, which the renderers draw on their original
    /// path.
    pub fn is_pinhole(&self) -> bool {
        self.projection == Projection::Pinhole && self.coeffs == [0.0; LENS_COEFFS]
    }

    /// The GPU renderers' code for this lens (their `lens_kind`): 0 the
    /// undistorted pinhole, 1 a distorted pinhole, 2 an equidistant
    /// projection.
    pub fn kind(&self) -> u32 {
        match self.projection {
            _ if self.is_pinhole() => 0,
            Projection::Pinhole => 1,
            Projection::Equidistant => 2,
        }
    }

    /// Sets [`Lens::max_theta`] for a `width x height` image with these
    /// intrinsics: along each of 64 directions around the axis, the angle at
    /// which a ray leaves the image grown by a tenth of its size on every
    /// side (so splats just outside still reach the border pixels), but
    /// never past the first angle where the distortion stops moving the
    /// point outwards, in any direction. Returns the lens.
    pub fn fit_max_theta(
        mut self,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let (fx, fy, cx, cy) = (fx as f64, fy as f64, cx as f64, cy as f64);
        let (w, h) = (width as f64, height as f64);
        let (margin_x, margin_y) = (0.1 * w, 0.1 * h);
        let limit = match self.projection {
            Projection::Pinhole => FRAC_PI_2 - 1e-3,
            Projection::Equidistant => PI - 1e-3,
        };
        let step = 1e-3;
        let mut exit_max = 0.0f64;
        let mut fold_min = limit;
        for k in 0..64 {
            let phi = 2.0 * PI * k as f64 / 64.0;
            let (sp, cp) = phi.sin_cos();
            let mut last_radius = 0.0f64;
            let mut theta = step;
            let mut exit = limit;
            while theta < limit {
                let (st, ct) = theta.sin_cos();
                let p = [st * cp, st * sp, ct];
                let ([u, v], _) = lens_coords(&self, p);
                let (px, py) = (fx * u + cx, fy * v + cy);
                let radius = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
                if !radius.is_finite() || radius <= last_radius {
                    fold_min = fold_min.min(theta - step);
                    exit = exit.min(theta - step);
                    break;
                }
                last_radius = radius;
                if px < -margin_x || py < -margin_y || px > w + margin_x || py > h + margin_y {
                    exit = theta;
                    break;
                }
                theta += step;
            }
            exit_max = exit_max.max(exit);
        }
        self.max_theta = exit_max.min(fold_min) as f32;
        self
    }
}

/// The arithmetic the lens math runs on: `f32` in the renderer, `f64` and
/// dual numbers in the gradient oracle.
pub trait LensScalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn from_f64(v: f64) -> Self;
    fn value(self) -> f64;
    fn sqrt(self) -> Self;
    /// `self.atan2(x)`: the angle of `(x, self)`.
    fn atan2(self, x: Self) -> Self;
}

impl LensScalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn value(self) -> f64 {
        self as f64
    }
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    fn atan2(self, x: Self) -> Self {
        f32::atan2(self, x)
    }
}

impl LensScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn value(self) -> f64 {
        self
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
}

/// Below this `(r / z)²` the equidistant scale `θ / r` and its lateral
/// derivative come from their series in `r / z` (exact to ~1e-9 there):
/// the closed forms cancel catastrophically near the axis, worst in `f32`.
const SERIES_S2: f64 = 1e-2;

/// The pinhole's `(x, y) / z` and its Jacobian.
fn pinhole_coords<T: LensScalar>(p: [T; 3]) -> ([T; 2], [[T; 3]; 2]) {
    let [x, y, z] = p;
    let one = T::from_f64(1.0);
    let zero = T::from_f64(0.0);
    let inv_z = one / z;
    let inv_z2 = inv_z * inv_z;
    (
        [x * inv_z, y * inv_z],
        [[inv_z, zero, -(x * inv_z2)], [zero, inv_z, -(y * inv_z2)]],
    )
}

/// The equidistant `θ (x, y) / r` and its Jacobian.
fn equidistant_coords<T: LensScalar>(p: [T; 3]) -> ([T; 2], [[T; 3]; 2]) {
    let [x, y, z] = p;
    let c = T::from_f64;
    let r2 = x * x + y * y;
    let rho2 = r2 + z * z;
    // scale = θ / r; lateral = (∂scale/∂r) / r, so ∂scale/∂x = x lateral.
    let (scale, lateral) = if z.value() > 0.0 && r2.value() < SERIES_S2 * z.value() * z.value() {
        // atan(s) / s and (1/(1+s²) − atan(s)/s) / s² in s = r / z.
        let inv_z = c(1.0) / z;
        let s2 = r2 * inv_z * inv_z;
        let scale =
            inv_z * (c(1.0) + s2 * (c(-1.0 / 3.0) + s2 * (c(1.0 / 5.0) + s2 * c(-1.0 / 7.0))));
        let lateral = inv_z
            * inv_z
            * inv_z
            * (c(-2.0 / 3.0) + s2 * (c(4.0 / 5.0) + s2 * (c(-6.0 / 7.0) + s2 * c(8.0 / 9.0))));
        (scale, lateral)
    } else {
        let r = r2.sqrt();
        let scale = r.atan2(z) / r;
        (scale, (z / rho2 - scale) / r2)
    };
    let ds_dz = -(c(1.0) / rho2);
    let xy = x * y * lateral;
    (
        [scale * x, scale * y],
        [
            [scale + x * x * lateral, xy, x * ds_dz],
            [xy, scale + y * y * lateral, y * ds_dz],
        ],
    )
}

/// COLMAP's distortion of projected coordinates `(u, v)` (`ρ² = u² + v²`)
/// and its 2×2 Jacobian:
/// `u' = u + u (k1 ρ² + k2 ρ⁴ + k3 ρ⁶ + k4 ρ⁸) + 2 p1 u v + p2 (ρ² + 2 u²) + sx1 ρ²`,
/// `v' = v + v (…) + 2 p2 u v + p1 (ρ² + 2 v²) + sy1 ρ²`.
pub fn distort<T: LensScalar>(n: [T; 2], coeffs: &[f32; LENS_COEFFS]) -> ([T; 2], [[T; 2]; 2]) {
    let c = |i: usize| T::from_f64(coeffs[i] as f64);
    let (k1, k2, k3, k4, p1, p2, sx1, sy1) = (c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7));
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let six = T::from_f64(6.0);
    let [u, v] = n;
    let t2 = u * u + v * v;
    let radial = t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)));
    // d(radial)/d(ρ²)
    let g = k1 + t2 * (two * k2 + t2 * (T::from_f64(3.0) * k3 + t2 * T::from_f64(4.0) * k4));
    let uu = u * u;
    let vv = v * v;
    let uv = u * v;
    let du = u * radial + two * p1 * uv + p2 * (t2 + two * uu) + sx1 * t2;
    let dv = v * radial + two * p2 * uv + p1 * (t2 + two * vv) + sy1 * t2;
    let cross = two * uv * g + two * p1 * u + two * p2 * v;
    (
        [u + du, v + dv],
        [
            [
                one + radial + two * uu * g + two * p1 * v + six * p2 * u + two * sx1 * u,
                cross + two * sx1 * v,
            ],
            [
                cross + two * sy1 * u,
                one + radial + two * vv * g + two * p2 * u + six * p1 * v + two * sy1 * v,
            ],
        ],
    )
}

/// The distorted normalized coordinates of camera-space point `p` (before
/// the focal lengths) and their 2×3 Jacobian, for `p` within
/// [`Lens::max_theta`] (and `z > 0` for a pinhole).
pub fn lens_coords<T: LensScalar>(lens: &Lens, p: [T; 3]) -> ([T; 2], [[T; 3]; 2]) {
    let (n, nj) = match lens.projection {
        Projection::Pinhole => pinhole_coords(p),
        Projection::Equidistant => equidistant_coords(p),
    };
    if lens.coeffs == [0.0; LENS_COEFFS] {
        return (n, nj);
    }
    let (m, dj) = distort(n, &lens.coeffs);
    let row =
        |r: usize| -> [T; 3] { std::array::from_fn(|k| dj[r][0] * nj[0][k] + dj[r][1] * nj[1][k]) };
    (m, [row(0), row(1)])
}

/// [`lens_coords`] for any camera-space point `p`: within
/// [`Lens::max_theta`] the lens itself; past it, the lens's first-order
/// extension at `q`, `p` turned towards the optical axis onto the
/// `max_theta` cone (same distance and azimuth): `m(q) + J(q) (p − q)`.
/// Returns the coordinates, the Jacobian at `q`, and `q`.
///
/// Out there the distortion is not calibrated (and may fold), but a
/// gaussian centred there can still reach into the image, as one off an
/// undistorted pinhole's frustum does; culling it instead would hide it
/// from the views it covers. The extension is continuous at the cone and
/// moves a point by `sin(θ − max_theta)` times the lens's angular rate
/// there, outwards up to `θ = max_theta + 90°`.
pub fn extended_lens_coords<T: LensScalar>(
    lens: &Lens,
    p: [T; 3],
) -> ([T; 2], [[T; 3]; 2], [T; 3]) {
    if ray_angle(p.map(T::value)) <= lens.max_theta as f64 {
        let (m, jac) = lens_coords(lens, p);
        return (m, jac, p);
    }
    let [x, y, z] = p;
    let r2 = x * x + y * y;
    let rho = (r2 + z * z).sqrt();
    let (sin, cos) = (lens.max_theta as f64).sin_cos();
    let k = rho * T::from_f64(sin) / r2.sqrt();
    let q = [x * k, y * k, rho * T::from_f64(cos)];
    let (mq, jac) = lens_coords(lens, q);
    let d = [x - q[0], y - q[1], z - q[2]];
    let m = std::array::from_fn(|r| mq[r] + jac[r][0] * d[0] + jac[r][1] * d[1] + jac[r][2] * d[2]);
    (m, jac, q)
}

/// Angle between the optical axis and the ray to camera-space point `p`.
pub fn ray_angle(p: [f64; 3]) -> f64 {
    (p[0] * p[0] + p[1] * p[1]).sqrt().atan2(p[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Camera 0 of ETH3D's delivery_area rig in this crate's coefficient
    /// order.
    fn thin_prism() -> Lens {
        Lens::new(
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
        )
    }

    fn lenses() -> Vec<Lens> {
        vec![
            Lens::PINHOLE,
            Lens::new(
                Projection::Pinhole,
                [-0.2, 0.05, 0.0, 0.0, 0.001, -0.002, 0.0, 0.0],
            ),
            Lens::new(Projection::Equidistant, [0.0; LENS_COEFFS]),
            Lens::new(
                Projection::Equidistant,
                [0.02, -0.01, 0.003, -0.0005, 0.0, 0.0, 0.0, 0.0],
            ),
            thin_prism(),
        ]
    }

    #[test]
    fn jacobian_matches_central_differences() {
        let points: [[f64; 3]; 6] = [
            [0.3, -0.2, 2.0],
            [-0.9, 0.6, 1.5],
            [0.05, 0.02, 3.0],
            [1e-4, -2e-4, 1.0],
            [0.0, 0.0, 2.5],
            [0.03, 0.04, 1.0],
        ];
        for lens in lenses() {
            for p in points {
                let (_, jac) = lens_coords(&lens, p);
                for axis in 0..3 {
                    let h = 1e-6;
                    let (mut plus, mut minus) = (p, p);
                    plus[axis] += h;
                    minus[axis] -= h;
                    let (a, _) = lens_coords(&lens, plus);
                    let (b, _) = lens_coords(&lens, minus);
                    for row in 0..2 {
                        let numeric = (a[row] - b[row]) / (2.0 * h);
                        assert!(
                            (jac[row][axis] - numeric).abs() < 1e-6 * (1.0 + numeric.abs()),
                            "{lens:?} at {p:?}: d{row}/d{axis} {} vs {numeric}",
                            jac[row][axis]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn equidistant_series_meets_the_closed_form() {
        // Just inside and outside the series branch, values and Jacobians
        // agree to f64 precision.
        let lens = Lens::new(Projection::Equidistant, [0.0; LENS_COEFFS]);
        let z = 2.0;
        let r = (SERIES_S2).sqrt() * z;
        let (a, ja) = lens_coords(&lens, [r * (1.0 - 1e-9), 0.0, z]);
        let (b, jb) = lens_coords(&lens, [r * (1.0 + 1e-9), 0.0, z]);
        assert!((a[0] - b[0]).abs() < 1e-9);
        for row in 0..2 {
            for k in 0..3 {
                assert!((ja[row][k] - jb[row][k]).abs() < 1e-8, "{row} {k}");
            }
        }
    }

    #[test]
    fn thin_prism_matches_visloc_core() {
        use nalgebra::Point3;
        use visloc_core::types::{Camera, CameraModel};
        let lens = thin_prism();
        let k = lens.coeffs.map(|c| c as f64);
        let camera = Camera {
            id: 1,
            model: CameraModel::ThinPrismFisheye,
            width: 752,
            height: 480,
            params: vec![
                538.731, 538.615, 413.003, 244.558, k[0], k[1], k[4], k[5], k[2], k[3], k[6], k[7],
            ],
        };
        for p in [[0.4, -0.3, 1.1], [0.0, 0.0, 2.0], [-0.8, 0.5, 1.3]] {
            let ([u, v], _) = lens_coords(&lens, p);
            let expected = camera.project(&Point3::new(p[0], p[1], p[2])).unwrap();
            assert!((538.731 * u + 413.003 - expected.x).abs() < 1e-3);
            assert!((538.615 * v + 244.558 - expected.y).abs() < 1e-3);
        }
    }

    #[test]
    fn extension_meets_the_lens_and_keeps_moving_outwards() {
        // ETH3D's narrow rig lens (camera 3), whose polynomial blows up
        // past the image, fitted to its 752x480 image.
        let lens = Lens::new(
            Projection::Equidistant,
            [
                0.578522,
                -0.483915,
                -0.433404,
                1.53835,
                -0.00535528,
                0.0063216,
                -0.00547212,
                0.00645311,
            ],
        )
        .fit_max_theta(711.926, 710.904, 410.892, 250.748, 752, 480);
        let edge = lens.max_theta as f64;
        let at = |theta: f64, phi: f64| {
            let p = [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]
            .map(|c| 3.0 * c);
            extended_lens_coords(&lens, p)
        };
        for phi in [0.3, 2.0, 4.4] {
            // Continuous at the cone, in the coordinates and the Jacobian.
            let (a, ja, _) = at(edge - 1e-7, phi);
            let (b, jb, q) = at(edge + 1e-7, phi);
            assert!((ray_angle(q) - edge).abs() < 1e-9);
            for r in 0..2 {
                assert!((a[r] - b[r]).abs() < 1e-5, "{a:?} vs {b:?}");
                for k in 0..3 {
                    assert!((ja[r][k] - jb[r][k]).abs() < 1e-4);
                }
            }
            // Then outwards, and bounded, until 90° past the cone.
            let mut last = (a[0] * a[0] + a[1] * a[1]).sqrt();
            let mut theta = edge + 0.05;
            while theta < edge + 1.5 {
                let (m, _, _) = at(theta, phi);
                let radius = (m[0] * m[0] + m[1] * m[1]).sqrt();
                assert!(
                    radius > last && radius < 4.0,
                    "{theta}: {radius} after {last}"
                );
                last = radius;
                theta += 0.05;
            }
        }
    }

    #[test]
    fn max_theta_stops_at_the_image_edge() {
        // An undistorted equidistant lens of f = 300 on a 600 px square
        // image: the corners are at r = 300 √2 px, θ = √2 rad; with the 10%
        // margin the widest exit is just past that.
        let lens = Lens::new(Projection::Equidistant, [0.0; LENS_COEFFS])
            .fit_max_theta(300.0, 300.0, 300.0, 300.0, 600, 600);
        let corner = 2f64.sqrt() * 1.2;
        assert!(
            (lens.max_theta as f64 - corner).abs() < 0.02,
            "{}",
            lens.max_theta
        );
        // A barrel polynomial that folds at θ = 1 caps it there.
        let folding = Lens::new(
            Projection::Equidistant,
            [-1.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        .fit_max_theta(300.0, 300.0, 300.0, 300.0, 600, 600);
        assert!(
            (folding.max_theta - 1.0).abs() < 0.01,
            "{}",
            folding.max_theta
        );
    }
}
