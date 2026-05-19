//! Rectified-stereo triangulation primitives.
//!
//! For a rectified stereo pair the right image has been warped so it
//! shares the left's intrinsics and rectification matches the rows. A
//! 3D point `X` in the left-camera frame projects to:
//!
//! - left:  `u_l = fx · X / Z + cx`,  `v_l = fy · Y / Z + cy`
//! - right: `u_r = u_l − fx · b / Z`, `v_r = v_l`
//!
//! where `b > 0` is the baseline (camera centers separation along the
//! left camera's `+x` axis). The disparity `d = u_l − u_r = fx · b / Z`
//! recovers the depth `Z = fx · b / d`, which combined with the left
//! pixel's normalized coordinates yields `X` and `Y`.

use nalgebra::Point3;
use visloc_core::types::Camera;

/// Triangulate a 3D point in the left camera's frame from a rectified
/// stereo pixel pair. `camera` carries the (shared) pinhole intrinsics
/// and `baseline` is the positive distance between the camera centers.
///
/// Returns `None` when the inputs do not represent a valid stereo
/// observation:
/// - non-finite baseline or non-positive baseline,
/// - unsupported camera model (no `fx, fy, cx, cy`),
/// - non-positive disparity (`u_l ≤ u_r`),
/// - disparity below `min_disparity_px` (set the threshold to clamp
///   numerically unstable far-field triangulations).
pub fn triangulate_stereo_pixel(
    camera: &Camera,
    baseline: f64,
    left_xy: (f64, f64),
    right_xy: (f64, f64),
    min_disparity_px: f64,
) -> Option<Point3<f64>> {
    if !baseline.is_finite() || baseline <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    let (u_l, v_l) = left_xy;
    let (u_r, _v_r) = right_xy;
    let disparity = u_l - u_r;
    if !disparity.is_finite() || disparity <= min_disparity_px.max(0.0) {
        return None;
    }
    let z = fx * baseline / disparity;
    let x = (u_l - cx) * z / fx;
    let y = (v_l - cy) * z / fy;
    Some(Point3::new(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::pinhole(1, 1241, 376, 718.856, 718.856, 607.193, 185.216)
    }

    /// Project a 3D point in the left frame to its rectified left/right
    /// pixel pair using a known baseline. Used to set up round-trip tests.
    fn project_stereo(
        camera: &Camera,
        baseline: f64,
        point: &Point3<f64>,
    ) -> ((f64, f64), (f64, f64)) {
        let (fx, fy, cx, cy) = camera.intrinsics().unwrap();
        let u_l = fx * point.x / point.z + cx;
        let v_l = fy * point.y / point.z + cy;
        let u_r = u_l - fx * baseline / point.z;
        ((u_l, v_l), (u_r, v_l))
    }

    #[test]
    fn round_trips_known_point() {
        let camera = camera();
        let baseline = 0.537150888; // KITTI 00 P1.
        let point = Point3::new(0.5, 0.2, 12.0);
        let (left, right) = project_stereo(&camera, baseline, &point);
        let recovered = triangulate_stereo_pixel(&camera, baseline, left, right, 0.5)
            .expect("valid stereo input");
        let err = (recovered - point).norm();
        assert!(err < 1e-9, "round-trip error {err} on {point:?}");
    }

    #[test]
    fn rejects_zero_disparity() {
        let camera = camera();
        let result = triangulate_stereo_pixel(&camera, 0.5, (320.0, 200.0), (320.0, 200.0), 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn rejects_negative_disparity() {
        let camera = camera();
        // u_r > u_l would imply the right camera saw the point further to
        // the right than the left camera, which is geometrically impossible
        // for points in front of a left-camera-centered rectified rig.
        let result = triangulate_stereo_pixel(&camera, 0.5, (320.0, 200.0), (340.0, 200.0), 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn rejects_disparity_below_threshold() {
        let camera = camera();
        // Disparity = 0.3 < 0.5 threshold.
        let result = triangulate_stereo_pixel(&camera, 0.5, (320.0, 200.0), (319.7, 200.0), 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn rejects_non_positive_baseline() {
        let camera = camera();
        let result = triangulate_stereo_pixel(&camera, 0.0, (320.0, 200.0), (300.0, 200.0), 0.5);
        assert!(result.is_none());
    }

    #[test]
    fn round_trip_works_for_off_axis_point() {
        let camera = camera();
        let baseline = 0.537150888;
        let point = Point3::new(-1.5, -0.8, 25.0);
        let (left, right) = project_stereo(&camera, baseline, &point);
        let recovered = triangulate_stereo_pixel(&camera, baseline, left, right, 0.0).unwrap();
        assert!((recovered - point).norm() < 1e-9);
    }
}
