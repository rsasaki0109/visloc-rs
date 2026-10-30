//! Basalt's Double Sphere camera model.
//!
//! The equations here follow the Double Sphere model used by Basalt at the
//! pinned upstream revision in [`crate::provenance`]. A camera-frame point is
//! projected into EuRoC pixel coordinates; an unprojection returns a unit ray
//! in that same camera frame. Image bounds are not imposed by `project`, so a
//! valid point outside the sensor rectangle can still be used by an optimizer.

use crate::fma::FusedMulAdd;
use nalgebra::{Point2, Point3, Vector3};
use thiserror::Error;

const EPS: f64 = 1e-12;

/// Parameters and image dimensions for a Double Sphere camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoubleSphereCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub xi: f64,
    pub alpha: f64,
    pub width: u32,
    pub height: u32,
}

impl DoubleSphereCamera {
    /// Constructs and validates a Double Sphere calibration.
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        xi: f64,
        alpha: f64,
        width: u32,
        height: u32,
    ) -> Result<Self, CameraModelError> {
        let values = [fx, fy, cx, cy, xi, alpha];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CameraModelError::NonFiniteParameter);
        }
        if fx <= 0.0 || fy <= 0.0 {
            return Err(CameraModelError::NonPositiveFocalLength { fx, fy });
        }
        if !(0.0..=1.0).contains(&alpha) {
            return Err(CameraModelError::AlphaOutOfRange { alpha });
        }
        if width == 0 || height == 0 {
            return Err(CameraModelError::InvalidResolution { width, height });
        }
        Ok(Self {
            fx,
            fy,
            cx,
            cy,
            xi,
            alpha,
            width,
            height,
        })
    }

    /// Projects a camera-frame point into pixel coordinates.
    ///
    /// The point must have a finite, non-zero norm and a positive Double
    /// Sphere denominator. The result is not clipped to image bounds.
    pub fn project(&self, point: &Point3<f64>) -> Option<Point2<f64>> {
        if !point.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let d1 = point.coords.norm();
        if d1 <= EPS {
            return None;
        }

        let zeta = self.xi * d1 + point.z;
        let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
        let denominator = self.alpha * d2 + (1.0 - self.alpha) * zeta;
        if !denominator.is_finite() || denominator <= EPS {
            return None;
        }

        let pixel = Point2::new(
            self.fx * point.x / denominator + self.cx,
            self.fy * point.y / denominator + self.cy,
        );
        pixel
            .coords
            .iter()
            .all(|value| value.is_finite())
            .then_some(pixel)
    }

    /// Float32-owned projection used by the upstream compatibility core.
    /// Parameters are cast at the API boundary and every intermediate below
    /// is an f32, matching `GenericCamera<float>` rather than merely casting
    /// the final pixel produced by the f64 implementation.
    pub fn project_f32(&self, point: &Point3<f32>) -> Option<Point2<f32>> {
        if !point.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let fx = self.fx as f32;
        let fy = self.fy as f32;
        let cx = self.cx as f32;
        let cy = self.cy as f32;
        let xi = self.xi as f32;
        let alpha = self.alpha as f32;
        let d1 = point.coords.norm();
        if d1 <= f32::EPSILON {
            return None;
        }
        let zeta = xi * d1 + point.z;
        let d2 = (point.x * point.x + point.y * point.y + zeta * zeta).sqrt();
        let denominator = alpha * d2 + (1.0_f32 - alpha) * zeta;
        if !denominator.is_finite() || denominator <= f32::EPSILON {
            return None;
        }
        let pixel = Point2::new(
            fx * point.x / denominator + cx,
            fy * point.y / denominator + cy,
        );
        pixel
            .coords
            .iter()
            .all(|value| value.is_finite())
            .then_some(pixel)
    }

    /// Unprojects a pixel into a unit camera-frame ray.
    pub fn unproject(&self, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        let ray = self.unproject_raw(pixel)?;
        let norm = ray.norm();
        if !norm.is_finite() || norm <= EPS {
            return None;
        }
        Some(ray / norm)
    }

    /// Unprojects a pixel into Basalt's raw camera-frame vector.  Upstream's
    /// DLT triangulator consumes this non-unit vector directly; the public
    /// [`Self::unproject`] method continues to expose the normalized ray used
    /// by the factor/geometry APIs.
    pub fn unproject_raw(&self, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return None;
        }

        let mx = (pixel.x - self.cx) / self.fx;
        let my = (pixel.y - self.cy) / self.fy;
        let r_sq = mx * mx + my * my;
        let sqrt_argument = 1.0 - (2.0 * self.alpha - 1.0) * r_sq;
        if !sqrt_argument.is_finite() || sqrt_argument < 0.0 {
            return None;
        }

        let mz_denominator = self.alpha * sqrt_argument.sqrt() + (1.0 - self.alpha);
        if mz_denominator.abs() <= EPS {
            return None;
        }
        let mz = (1.0 - self.alpha * self.alpha * r_sq) / mz_denominator;

        let k_radicand = mz * mz + (1.0 - self.xi * self.xi) * r_sq;
        if !k_radicand.is_finite() || k_radicand < 0.0 {
            return None;
        }
        let k_denominator = mz * mz + r_sq;
        if k_denominator <= EPS {
            return None;
        }
        let k = (mz * self.xi + k_radicand.sqrt()) / k_denominator;
        // The inverse first recovers the sphere point `k * [mx, my, mz]`;
        // subtracting xi on z returns the original camera-frame ray.
        let ray = Vector3::new(k * mx, k * my, k * mz - self.xi);
        if !ray.iter().all(|value| value.is_finite()) || ray.norm() <= EPS {
            return None;
        }
        Some(ray)
    }

    /// Float32-owned raw double-sphere unprojection.  This is deliberately a
    /// separate path so observation bearings and triangulation retain the
    /// upstream storage/rounding boundary all the way through the formulas.
    pub fn unproject_raw_f32(&self, pixel: &Point2<f32>) -> Option<Vector3<f32>> {
        if !pixel.coords.iter().all(|value| value.is_finite()) {
            return None;
        }
        let fx = self.fx as f32;
        let fy = self.fy as f32;
        let cx = self.cx as f32;
        let cy = self.cy as f32;
        let xi = self.xi as f32;
        let alpha = self.alpha as f32;
        let mx = (pixel.x - cx) / fx;
        let my = (pixel.y - cy) / fy;
        // Eigen's scalar packet path contracts the final square/addition
        // (`mx * mx + my * my`) into one FMA. Keep this reduction explicit;
        // it is the first expression-order boundary that can differ from a
        // generic f32 translation.
        let r_sq = mx.fused_mul_add(mx, my * my);
        // Keep the scalar temporaries and expression order of
        // DoubleSphereCamera<float>::unproject. The pinned native build
        // contracts the scalar reductions with FMA; algebraically equivalent
        // rewrites can differ by one or more f32 ulps here.
        let xi2_2 = alpha * alpha;
        let xi1_2 = xi * xi;
        let sqrt2 = (-(2.0_f32 * alpha - 1.0_f32))
            .fused_mul_add(r_sq, 1.0_f32)
            .sqrt();
        if !sqrt2.is_finite() {
            return None;
        }
        // The header writes `alpha * sqrt2 + Scalar(1) - alpha`; retain its
        // left-associative additions rather than evaluating `(1 - alpha)`
        // first, while preserving the native multiply-add contraction.
        let norm2 = alpha.fused_mul_add(sqrt2, 1.0_f32) - alpha;
        if !norm2.is_finite() || norm2.abs() <= f32::EPSILON {
            return None;
        }
        // GCC's native upstream build contracts the numerator subtraction
        // into `1 - alpha^2 * r_sq` and the subsequent norm/sqrt reductions
        // into the same scalar FMA order.
        let mz = (-xi2_2).fused_mul_add(r_sq, 1.0_f32) / norm2;
        let norm1 = mz.fused_mul_add(mz, r_sq);
        if !norm1.is_finite() || norm1 <= f32::EPSILON {
            return None;
        }
        let sqrt1 = mz.fused_mul_add(mz, (1.0_f32 - xi1_2) * r_sq).sqrt();
        if !sqrt1.is_finite() {
            return None;
        }
        let k = mz.fused_mul_add(xi, sqrt1) / norm1;
        let ray = Vector3::new(k * mx, k * my, k.fused_mul_add(mz, -xi));
        if !ray.iter().all(|value| value.is_finite()) || ray.norm() <= f32::EPSILON {
            return None;
        }
        Some(ray)
    }

    pub fn unproject_f32(&self, pixel: &Point2<f32>) -> Option<Vector3<f32>> {
        let ray = self.unproject_raw_f32(pixel)?;
        let norm = ray.norm();
        if !norm.is_finite() || norm <= f32::EPSILON {
            return None;
        }
        Some(ray / norm)
    }

    /// Returns whether a pixel lies inside the calibrated image rectangle.
    pub fn contains_pixel(&self, pixel: &Point2<f64>) -> bool {
        pixel.x >= 0.0
            && pixel.x < self.width as f64
            && pixel.y >= 0.0
            && pixel.y < self.height as f64
    }
}

/// Validation failures for Double Sphere parameters.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum CameraModelError {
    #[error("Double Sphere parameters contain a non-finite value")]
    NonFiniteParameter,
    #[error("focal lengths must be positive, got fx={fx}, fy={fy}")]
    NonPositiveFocalLength { fx: f64, fy: f64 },
    #[error("Double Sphere alpha must lie in [0, 1], got {alpha}")]
    AlphaOutOfRange { alpha: f64 },
    #[error("image resolution must be non-zero, got {width}x{height}")]
    InvalidResolution { width: u32, height: u32 },
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use super::*;

    fn camera() -> DoubleSphereCamera {
        DoubleSphereCamera::new(
            458.654, 457.296, 367.215, 248.375, 0.662073, 0.779792, 752, 480,
        )
        .unwrap()
    }

    #[test]
    fn projection_matches_double_sphere_golden_value() {
        let pixel = camera().project(&Point3::new(0.1, -0.05, 1.0)).unwrap();
        assert!((pixel.x - 394.693_793_482_414_3).abs() < 1e-9, "{pixel:?}");
        assert!((pixel.y - 234.676_283_381_008_18).abs() < 1e-9, "{pixel:?}");
    }

    #[test]
    fn projection_and_unprojection_preserve_ray_direction() {
        let point = Point3::new(0.1, -0.05, 1.0);
        let pixel = camera().project(&point).unwrap();
        let ray = camera().unproject(&pixel).unwrap();
        let expected = point.coords.normalize();
        assert!(
            (ray - expected).norm() < 1e-10,
            "ray={ray:?}, expected={expected:?}"
        );
    }

    #[test]
    fn raw_f32_unprojection_preserves_pinned_expression_bits() {
        let camera = DoubleSphereCamera::new(
            349.7560023050409,
            348.72454229977037,
            365.89440762590149,
            249.32995565708704,
            -0.2409573942178872,
            0.566996899163044,
            752,
            480,
        )
        .unwrap();
        let ray = camera
            .unproject_raw_f32(&Point2::new(97.0_f32, 258.0_f32))
            .unwrap();
        assert_eq!(ray.x.to_bits(), 0xbf0c_f651);
        assert_eq!(ray.y.to_bits(), 0x3c91_df79);
        assert_eq!(ray.z.to_bits(), 0x3f55_a58a);
    }

    #[test]
    fn m7_track2_raw_f32_unprojection_matches_pinned_native_bits() {
        let camera0 = DoubleSphereCamera::new(
            349.7560023050409,
            348.72454229977037,
            365.89440762590149,
            249.32995565708704,
            -0.2409573942178872,
            0.566996899163044,
            752,
            480,
        )
        .unwrap();
        let camera1 = DoubleSphereCamera::new(
            361.6713883800533,
            360.5856493689301,
            379.40818394080869,
            255.9772968522045,
            -0.21300835384809328,
            0.5767008625037023,
            752,
            480,
        )
        .unwrap();
        let ray0 = camera0
            .unproject_raw_f32(&Point2::new(46.0_f32, 118.0_f32))
            .unwrap();
        let ray1 = camera1
            .unproject_raw_f32(&Point2::new(52.372352600097656, 132.4171142578125))
            .unwrap();
        assert_eq!(
            [ray0.x.to_bits(), ray0.y.to_bits(), ray0.z.to_bits()],
            [0xbf21_4708, 0xbe84_d05b, 0x3f3b_6451]
        );
        assert_eq!(
            [ray1.x.to_bits(), ray1.y.to_bits(), ray1.z.to_bits()],
            [0xbf24_c3d6, 0xbe79_c13a, 0x3f39_b6ef]
        );
    }

    #[test]
    fn invalid_parameters_and_degenerate_points_are_rejected() {
        assert!(matches!(
            DoubleSphereCamera::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 1, 1),
            Err(CameraModelError::NonPositiveFocalLength { .. })
        ));
        assert!(camera().project(&Point3::origin()).is_none());
        assert!(camera().unproject(&Point2::new(f64::NAN, 0.0)).is_none());
    }
}
