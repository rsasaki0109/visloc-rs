//! Lens distortion models.
//!
//! Today this is just the OpenCV / Bouguet plumb-bob "radial-tangential"
//! model used by EuRoC MAV cam0 / cam1 and by every COLMAP `OPENCV`
//! camera. The forward (distort) map has the closed-form expression
//!
//! ```text
//! x_d = x · (1 + k₁·r² + k₂·r⁴) + 2·p₁·x·y + p₂·(r² + 2·x²)
//! y_d = y · (1 + k₁·r² + k₂·r⁴) + p₁·(r² + 2·y²) + 2·p₂·x·y
//! ```
//!
//! where `(x, y)` are *undistorted* normalized image coordinates and
//! `r² = x² + y²`. The inverse (undistort) has no closed form; the
//! standard fix is a fixed-point iteration that converges in well
//! under ten iterations for typical lens distortion magnitudes (EuRoC
//! cam0 sees |k₁| ≈ 0.28, |p_*| ≈ 1.9·10⁻⁴).
//!
//! The helpers operate on **normalized** coordinates so the caller is
//! free to attach them to either a `visloc_core::Camera` (via
//! `Camera::normalize_pixel` + manual projection back) or any other
//! pinhole intrinsics. A `undistort_pixel` convenience wrapper that
//! takes the camera intrinsics and applies normalize → undistort →
//! project is provided for the most common case.

use nalgebra::Point2;
use visloc_core::types::Camera;

/// OpenCV-style radial-tangential distortion coefficients
/// `(k₁, k₂, p₁, p₂)`. The `k₃` / higher-order radial term is
/// intentionally not modelled — EuRoC and most automotive sensors
/// publish only the 4-coefficient form, and adding `k₃` later is an
/// additive change to this struct, not a redesign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialTangential {
    /// First radial coefficient (`k₁`).
    pub k1: f64,
    /// Second radial coefficient (`k₂`).
    pub k2: f64,
    /// First tangential coefficient (`p₁`).
    pub p1: f64,
    /// Second tangential coefficient (`p₂`).
    pub p2: f64,
}

impl RadialTangential {
    /// Zero distortion — useful as the identity element for tests and
    /// as the default when a calibration only ships pinhole intrinsics.
    pub const IDENTITY: Self = Self {
        k1: 0.0,
        k2: 0.0,
        p1: 0.0,
        p2: 0.0,
    };

    /// Construct from EuRoC's `distortion_coefficients` vector. Accepts
    /// the 4-coefficient `(k1, k2, p1, p2)` form (the EuRoC default).
    /// Returns `None` if the input has a different length so the caller
    /// can branch on the calibration shape explicitly.
    pub fn from_euroc_coefficients(coeffs: &[f64]) -> Option<Self> {
        match coeffs {
            [k1, k2, p1, p2] => Some(Self {
                k1: *k1,
                k2: *k2,
                p1: *p1,
                p2: *p2,
            }),
            _ => None,
        }
    }

    /// Return `true` if every coefficient is exactly zero — the
    /// "no distortion" identity. Callers can fast-path the
    /// `undistort_*` helpers using this check.
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0
    }

    /// Apply the **forward** distortion model. Input is undistorted
    /// normalized coordinates `(x, y)` (i.e. the rays a pinhole camera
    /// would project); output is the distorted normalized coordinates
    /// the real sensor records. Closed-form, single-pass.
    pub fn distort_normalized(&self, undistorted: Point2<f64>) -> Point2<f64> {
        let x = undistorted.x;
        let y = undistorted.y;
        let r2 = x * x + y * y;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
        let dx_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        Point2::new(x * radial + dx_tan, y * radial + dy_tan)
    }

    /// Invert the distortion model. Input is the distorted normalized
    /// coordinates the sensor records; output is the undistorted
    /// normalized coordinates that match a pinhole projection.
    ///
    /// Uses fixed-point iteration with a fixed budget of 20 steps —
    /// well above what is needed for any real lens (EuRoC's
    /// `MH_01_easy` cam0 converges to `<1e-12` residual in ~6 steps).
    /// `is_identity()` is a fast-path that skips the loop entirely
    /// for zero-distortion calibrations.
    pub fn undistort_normalized(&self, distorted: Point2<f64>) -> Point2<f64> {
        if self.is_identity() {
            return distorted;
        }
        let xd = distorted.x;
        let yd = distorted.y;
        let mut x = xd;
        let mut y = yd;
        for _ in 0..20 {
            let r2 = x * x + y * y;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            let dx_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
            let dy_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
            let next_x = (xd - dx_tan) / radial;
            let next_y = (yd - dy_tan) / radial;
            // Tight enough that a converged residual sits well below
            // any sub-pixel feature precision; the loop normally exits
            // well before the 20-step ceiling.
            if (next_x - x).abs() + (next_y - y).abs() < 1.0e-12 {
                return Point2::new(next_x, next_y);
            }
            x = next_x;
            y = next_y;
        }
        Point2::new(x, y)
    }

    /// Convenience wrapper that takes a pixel observed by the real
    /// distorted sensor and returns the pixel a pinhole camera with
    /// the same intrinsics would have observed. The same `Camera` is
    /// used for both directions of the transformation, so this is the
    /// drop-in path for a pipeline that wants the rest of its geometry
    /// (back-projection, projection, PnP) to behave as pure pinhole.
    pub fn undistort_pixel(
        &self,
        camera: &Camera,
        distorted_pixel: Point2<f64>,
    ) -> Option<Point2<f64>> {
        let (fx, fy, cx, cy) = camera.intrinsics()?;
        let normalized_distorted =
            Point2::new((distorted_pixel.x - cx) / fx, (distorted_pixel.y - cy) / fy);
        let normalized_undistorted = self.undistort_normalized(normalized_distorted);
        Some(Point2::new(
            fx * normalized_undistorted.x + cx,
            fy * normalized_undistorted.y + cy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_distortion_is_a_no_op() {
        let id = RadialTangential::IDENTITY;
        assert!(id.is_identity());
        let p = Point2::new(0.123, -0.456);
        assert_eq!(id.distort_normalized(p), p);
        assert_eq!(id.undistort_normalized(p), p);
    }

    #[test]
    fn from_euroc_coefficients_parses_four_element_vector() {
        // EuRoC MH_01_easy cam0 published coefficients.
        let coeffs = vec![-0.28340811, 0.07395907, 0.00019359, 0.0000176187114];
        let model = RadialTangential::from_euroc_coefficients(&coeffs).expect("four-coeff form");
        assert!((model.k1 + 0.28340811).abs() < 1.0e-12);
        assert!((model.k2 - 0.07395907).abs() < 1.0e-12);
        assert!((model.p1 - 0.00019359).abs() < 1.0e-12);
        assert!((model.p2 - 0.0000176187114).abs() < 1.0e-12);
        assert!(!model.is_identity());
    }

    #[test]
    fn from_euroc_coefficients_rejects_other_lengths() {
        assert!(RadialTangential::from_euroc_coefficients(&[]).is_none());
        assert!(RadialTangential::from_euroc_coefficients(&[0.1]).is_none());
        assert!(RadialTangential::from_euroc_coefficients(&[0.1, 0.2, 0.3]).is_none());
        assert!(RadialTangential::from_euroc_coefficients(&[0.1, 0.2, 0.3, 0.4, 0.5]).is_none());
    }

    #[test]
    fn distort_then_undistort_returns_input() {
        let model = RadialTangential {
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 0.0000176187114,
        };
        // Spread points across normalized image space — EuRoC cam0 has
        // FOV ~80°, so |x|, |y| up to ~0.6 is realistic at the edges.
        for &(x, y) in &[
            (0.0, 0.0),
            (0.1, 0.1),
            (-0.2, 0.3),
            (0.5, -0.4),
            (-0.6, -0.6),
            (0.6, 0.6),
        ] {
            let p = Point2::new(x, y);
            let distorted = model.distort_normalized(p);
            let recovered = model.undistort_normalized(distorted);
            let residual = (recovered.coords - p.coords).norm();
            assert!(
                residual < 1.0e-10,
                "round-trip failed for ({x}, {y}): residual = {residual}, recovered = {recovered:?}",
            );
        }
    }

    #[test]
    fn undistort_then_distort_returns_input() {
        // Reverse direction round-trip: feed a "raw" sensor pixel,
        // recover the pinhole equivalent, push it back through the
        // forward model. Pins both halves of the model agree.
        let model = RadialTangential {
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 0.0000176187114,
        };
        for &(x, y) in &[
            (0.0, 0.0),
            (0.15, -0.25),
            (-0.45, 0.35),
            (0.5, 0.5),
            (-0.55, -0.55),
        ] {
            let distorted = Point2::new(x, y);
            let undistorted = model.undistort_normalized(distorted);
            let redistorted = model.distort_normalized(undistorted);
            let residual = (redistorted.coords - distorted.coords).norm();
            assert!(
                residual < 1.0e-10,
                "reverse round-trip failed for ({x}, {y}): residual = {residual}",
            );
        }
    }

    #[test]
    fn euroc_cam0_edge_pixel_shifts_meaningfully() {
        // Sanity bound the magnitude of the correction. EuRoC cam0
        // intrinsics are roughly fu=458.654, fv=457.296, cu=367.215,
        // cv=248.375 on a 752×480 sensor. A corner near the image
        // edge should shift by several pixels under k1 ≈ -0.28; the
        // centre should barely move. This guards against a silent
        // bug where `undistort_pixel` accidentally collapsed to the
        // identity (e.g. by passing the wrong coefficient set).
        let camera = Camera::pinhole(0, 752, 480, 458.654, 457.296, 367.215, 248.375);
        let model = RadialTangential {
            k1: -0.28340811,
            k2: 0.07395907,
            p1: 0.00019359,
            p2: 0.0000176187114,
        };
        let centre = Point2::new(367.215, 248.375);
        let centre_undist = model
            .undistort_pixel(&camera, centre)
            .expect("intrinsics ok");
        let centre_shift = (centre_undist.coords - centre.coords).norm();
        assert!(
            centre_shift < 0.01,
            "principal-point shift should be ~0, got {centre_shift}",
        );

        // Corner pixel (top-left). |k1| ≈ 0.28 over a normalized
        // radius near 0.7 produces a several-pixel shift.
        let corner = Point2::new(10.0, 10.0);
        let corner_undist = model
            .undistort_pixel(&camera, corner)
            .expect("intrinsics ok");
        let corner_shift = (corner_undist.coords - corner.coords).norm();
        assert!(
            corner_shift > 5.0,
            "corner pixel should shift > 5 px under EuRoC distortion, got {corner_shift}",
        );
        // But the shift must remain within the sensor footprint — a
        // numerical implosion would have produced an absurd value.
        assert!(
            corner_shift < 200.0,
            "corner shift unreasonably large: {corner_shift}"
        );
    }

    #[test]
    fn undistort_pixel_returns_none_for_unknown_camera_model() {
        // `intrinsics()` returns `None` for `CameraModel::Unknown` —
        // confirm the convenience wrapper propagates that instead of
        // panicking or returning a meaningless value.
        let camera = Camera {
            id: 0,
            model: visloc_core::types::CameraModel::Unknown("FISHEYE_XYZ".to_string()),
            width: 752,
            height: 480,
            params: vec![458.654, 457.296, 367.215, 248.375],
        };
        let model = RadialTangential::IDENTITY;
        assert!(model
            .undistort_pixel(&camera, Point2::new(100.0, 100.0))
            .is_none());
    }
}
