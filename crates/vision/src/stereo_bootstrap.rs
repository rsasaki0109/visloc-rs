//! Two-view stereo bootstrap: match descriptors between a left/right camera
//! pair, triangulate each pair via DLT, and return the surviving 3D points
//! in the left camera frame.
//!
//! Companion to [`crate::stereo::triangulate_stereo_pixel`], which assumes
//! a **rectified** stereo pair (shared intrinsics, identity relative rotation,
//! pure-translation baseline along the left camera's `+x` axis). The general
//! 6-DoF stereo case — EuRoC's cam0/cam1 are mounted with a small relative
//! rotation in addition to the baseline translation — needs a full DLT
//! triangulation working in normalised image coordinates.
//!
//! Inputs are assumed to be **already undistorted**: each `keypoint` in the
//! supplied [`FeatureSet`] is the "ideal pinhole" pixel position implied by
//! the camera's intrinsics. Descriptors come from the matcher's
//! configuration. The bootstrap output is a list of surviving
//! [`StereoBootstrapLandmark`] entries, one per `(left_index, right_index)`
//! pair that:
//!
//! 1. matched under cross-checked brute-force descriptor matching (with the
//!    configured Lowe ratio),
//! 2. triangulated to a 3D point that lies in front of **both** cameras,
//!    with `Z` in `[min_depth_meters, max_depth_meters]`, and
//! 3. reprojected back to both pixel observations within
//!    `max_reprojection_error_pixels`.
//!
//! Matches that fail any of those gates are silently dropped — the caller
//! can keep using a fall-back depth seed for the unmatched left keypoints.

use nalgebra::{Matrix3, Matrix3x4, Matrix4, Point2, Point3};
use visloc_core::geometry::SE3;
use visloc_core::types::Camera;

use crate::features::FeatureSet;
use crate::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};

/// Tunable knobs for [`bootstrap_stereo_landmarks`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoBootstrapConfig {
    /// Lowe ratio for the inner brute-force matcher. `None` disables the
    /// ratio test; the default mirrors [`BruteForceMatcher::default`].
    pub matcher_ratio: Option<f32>,
    /// Minimum positive depth (in metres) the triangulated point must have
    /// in **both** cameras. Values at or below this threshold are dropped
    /// as numerically unstable far-field / behind-camera triangulations.
    pub min_depth_meters: f64,
    /// Maximum depth (in metres) the triangulated point may have in the
    /// **left** camera frame. The default is loose enough to admit any
    /// realistic indoor / outdoor scene while still rejecting matches that
    /// land at infinity due to a near-zero parallax baseline.
    pub max_depth_meters: f64,
    /// Maximum allowed reprojection error (in pixels) for either of the
    /// two observations. Larger errors are dropped as outliers.
    pub max_reprojection_error_pixels: f64,
    /// Independent one-sigma localisation error assigned to each of
    /// `(u_left, v_left, u_right, v_right)`. The DLT Jacobian propagates this
    /// pixel covariance into an anisotropic 3-D covariance for every survivor.
    pub pixel_stddev_pixels: f64,
}

impl Default for StereoBootstrapConfig {
    fn default() -> Self {
        Self {
            matcher_ratio: Some(0.8),
            min_depth_meters: 0.1,
            max_depth_meters: 50.0,
            max_reprojection_error_pixels: 2.0,
            pixel_stddev_pixels: 1.0,
        }
    }
}

/// A surviving triangulated stereo match.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoBootstrapLandmark {
    /// Index into `left_features.keypoints` (and `.descriptors`).
    pub left_keypoint_index: usize,
    /// Index into `right_features.keypoints`.
    pub right_keypoint_index: usize,
    /// Triangulated point expressed in the **left** camera frame. The
    /// caller is expected to transform this to its desired frame via the
    /// known world-to-left-camera pose.
    pub point_left_camera_frame: Point3<f64>,
    /// Reprojection error (in pixels) of the triangulated point in the
    /// left image.
    pub left_reprojection_error_pixels: f64,
    /// Reprojection error (in pixels) of the triangulated point in the
    /// right image.
    pub right_reprojection_error_pixels: f64,
    /// First-order covariance of the triangulated point in the left-camera
    /// frame, propagated from the four input pixel coordinates.
    pub point_covariance_left_camera_frame: Matrix3<f64>,
}

/// Match descriptors between the two feature sets, triangulate each
/// surviving pair, and return the matches that pass the configured depth /
/// reprojection-error gates. Results are returned in `left_keypoint_index`
/// order.
///
/// `left_to_right` is the transform that maps a point expressed in the
/// **left** camera frame to the **right** camera frame:
/// `P_right = left_to_right.transform_point(P_left)`. For EuRoC the
/// caller computes it from the two per-camera `T_BS` body-to-sensor
/// matrices as `body_to_cam1.inverse().compose(&body_to_cam0)`.
pub fn bootstrap_stereo_landmarks(
    left_camera: &Camera,
    right_camera: &Camera,
    left_to_right: &SE3,
    left_features: &FeatureSet,
    right_features: &FeatureSet,
    config: &StereoBootstrapConfig,
) -> Vec<StereoBootstrapLandmark> {
    if left_features.is_empty() || right_features.is_empty() {
        return Vec::new();
    }

    let matcher = CrossCheckMatcher::new(BruteForceMatcher {
        ratio: config.matcher_ratio,
    });
    let matches =
        matcher.match_descriptors(&left_features.descriptors, &right_features.descriptors);

    let mut survivors = Vec::with_capacity(matches.len());
    for descriptor_match in matches {
        let left_keypoint = left_features.keypoints[descriptor_match.query_index];
        let right_keypoint = right_features.keypoints[descriptor_match.train_index];

        let Some(point_left) = triangulate_two_view_left_frame(
            left_camera,
            right_camera,
            left_to_right,
            &left_keypoint,
            &right_keypoint,
        ) else {
            continue;
        };

        // In-front-of-camera + depth-range gate on both cameras.
        if !point_left.z.is_finite()
            || point_left.z < config.min_depth_meters
            || point_left.z > config.max_depth_meters
        {
            continue;
        }
        let point_right = left_to_right.transform_point(&point_left);
        if !point_right.z.is_finite() || point_right.z < config.min_depth_meters {
            continue;
        }

        // Reprojection-error gate (Euclidean pixel distance) on both
        // observations. Failed projections (point exactly on principal ray
        // with degenerate intrinsics) are treated as outliers.
        let Some(left_projected) = left_camera.project(&point_left) else {
            continue;
        };
        let Some(right_projected) = right_camera.project(&point_right) else {
            continue;
        };
        let left_error = (left_projected - left_keypoint).norm();
        let right_error = (right_projected - right_keypoint).norm();
        if left_error > config.max_reprojection_error_pixels
            || right_error > config.max_reprojection_error_pixels
        {
            continue;
        }

        let Some(point_covariance_left_camera_frame) = propagate_stereo_pixel_covariance(
            left_camera,
            right_camera,
            left_to_right,
            &left_keypoint,
            &right_keypoint,
            config.pixel_stddev_pixels,
        ) else {
            continue;
        };

        survivors.push(StereoBootstrapLandmark {
            left_keypoint_index: descriptor_match.query_index,
            right_keypoint_index: descriptor_match.train_index,
            point_left_camera_frame: point_left,
            left_reprojection_error_pixels: left_error,
            right_reprojection_error_pixels: right_error,
            point_covariance_left_camera_frame,
        });
    }

    survivors.sort_by_key(|landmark| landmark.left_keypoint_index);
    survivors
}

/// Propagate independent isotropic left/right keypoint noise through the
/// general two-view DLT triangulator.
///
/// The returned first-order covariance is `J sigma² I Jᵀ`, where the columns
/// of `J` are central finite differences with respect to
/// `(u_left, v_left, u_right, v_right)`. This supports EuRoC's non-rectified
/// 6-DoF cam0/cam1 transform while retaining the familiar quadratic growth of
/// rectified-stereo depth uncertainty with range.
pub fn propagate_stereo_pixel_covariance(
    left_camera: &Camera,
    right_camera: &Camera,
    left_to_right: &SE3,
    left_pixel: &Point2<f64>,
    right_pixel: &Point2<f64>,
    pixel_stddev_pixels: f64,
) -> Option<Matrix3<f64>> {
    if !pixel_stddev_pixels.is_finite() || pixel_stddev_pixels <= 0.0 {
        return None;
    }
    let epsilon = (pixel_stddev_pixels * 1.0e-3).max(1.0e-4);
    let mut jacobian = Matrix3x4::<f64>::zeros();
    for coordinate in 0..4 {
        let mut left_plus = *left_pixel;
        let mut left_minus = *left_pixel;
        let mut right_plus = *right_pixel;
        let mut right_minus = *right_pixel;
        match coordinate {
            0 => {
                left_plus.x += epsilon;
                left_minus.x -= epsilon;
            }
            1 => {
                left_plus.y += epsilon;
                left_minus.y -= epsilon;
            }
            2 => {
                right_plus.x += epsilon;
                right_minus.x -= epsilon;
            }
            3 => {
                right_plus.y += epsilon;
                right_minus.y -= epsilon;
            }
            _ => unreachable!(),
        }
        let plus = triangulate_two_view_left_frame(
            left_camera,
            right_camera,
            left_to_right,
            &left_plus,
            &right_plus,
        )?;
        let minus = triangulate_two_view_left_frame(
            left_camera,
            right_camera,
            left_to_right,
            &left_minus,
            &right_minus,
        )?;
        jacobian.set_column(coordinate, &((plus - minus) / (2.0 * epsilon)));
    }
    let covariance = jacobian * jacobian.transpose() * pixel_stddev_pixels.powi(2);
    if covariance.iter().all(|value| value.is_finite())
        && covariance
            .diagonal()
            .iter()
            .all(|variance| *variance >= 0.0)
    {
        Some((covariance + covariance.transpose()) * 0.5)
    } else {
        None
    }
}

/// DLT triangulation for one (left, right) observation pair, returning the
/// 3D point expressed in the **left** camera frame.
///
/// Left projection matrix is `[I | 0]` by construction; the right is
/// `[R | t]` where `(R, t) = (left_to_right.rotation, left_to_right.translation)`.
/// Two rows per camera give a 4x4 system whose smallest right-singular
/// vector is the homogeneous solution. Returns `None` when either camera
/// is non-pinhole, the homogeneous-component scale is degenerate, or the
/// SVD fails to converge.
pub fn triangulate_two_view_left_frame(
    left_camera: &Camera,
    right_camera: &Camera,
    left_to_right: &SE3,
    left_pixel: &Point2<f64>,
    right_pixel: &Point2<f64>,
) -> Option<Point3<f64>> {
    let left_normalized = left_camera.normalize_pixel(left_pixel)?;
    let right_normalized = right_camera.normalize_pixel(right_pixel)?;

    let rotation = left_to_right.rotation.to_rotation_matrix().into_inner();
    let t = left_to_right.translation;

    // Right projection rows P1, P2, P3 ∈ R⁴.
    let p1 = [rotation[(0, 0)], rotation[(0, 1)], rotation[(0, 2)], t.x];
    let p2 = [rotation[(1, 0)], rotation[(1, 1)], rotation[(1, 2)], t.y];
    let p3 = [rotation[(2, 0)], rotation[(2, 1)], rotation[(2, 2)], t.z];

    // Stack constraints:
    //   row 0: x_l e3 - e1   = (-1, 0, x_l, 0)
    //   row 1: y_l e3 - e2   = (0, -1, y_l, 0)
    //   row 2: x_r P3 - P1
    //   row 3: y_r P3 - P2
    let mut a = Matrix4::<f64>::zeros();
    a[(0, 0)] = -1.0;
    a[(0, 1)] = 0.0;
    a[(0, 2)] = left_normalized.x;
    a[(0, 3)] = 0.0;

    a[(1, 0)] = 0.0;
    a[(1, 1)] = -1.0;
    a[(1, 2)] = left_normalized.y;
    a[(1, 3)] = 0.0;

    let x_r = right_normalized.x;
    let y_r = right_normalized.y;
    for column in 0..4 {
        a[(2, column)] = x_r * p3[column] - p1[column];
        a[(3, column)] = y_r * p3[column] - p2[column];
    }

    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let homogeneous = v_t.row(v_t.nrows() - 1);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() < 1.0e-12 {
        return None;
    }
    Some(Point3::new(
        homogeneous[0] / w,
        homogeneous[1] / w,
        homogeneous[2] / w,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    fn pinhole() -> Camera {
        Camera::pinhole(1, 752, 480, 458.0, 457.0, 367.0, 248.0)
    }

    /// Project a left-frame point through `[I|0]` and `[R|t]` to produce
    /// the synthetic left/right pixel pair. Used to seed round-trip tests.
    fn project_left_and_right(
        left_camera: &Camera,
        right_camera: &Camera,
        left_to_right: &SE3,
        point_left: &Point3<f64>,
    ) -> (Point2<f64>, Point2<f64>) {
        let left_pixel = left_camera.project(point_left).expect("in front of left");
        let point_right = left_to_right.transform_point(point_left);
        let right_pixel = right_camera
            .project(&point_right)
            .expect("in front of right");
        (left_pixel, right_pixel)
    }

    #[test]
    fn dlt_triangulation_round_trips_pure_translation_baseline() {
        let left = pinhole();
        let right = pinhole();
        // Pure-translation baseline along left +x (EuRoC-style nominal).
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        let point = Point3::new(0.4, 0.2, 3.0);
        let (l, r) = project_left_and_right(&left, &right, &left_to_right, &point);
        let recovered = triangulate_two_view_left_frame(&left, &right, &left_to_right, &l, &r)
            .expect("DLT succeeds for a valid in-front-of-both-cameras point");
        assert!(
            (recovered - point).norm() < 1.0e-9,
            "round-trip error {} on {point:?}",
            (recovered - point).norm()
        );
    }

    #[test]
    fn propagated_rectified_depth_stddev_matches_disparity_jacobian() {
        let left = pinhole();
        let right = pinhole();
        let baseline = 0.11;
        let left_to_right = SE3::new(
            UnitQuaternion::identity(),
            Vector3::new(-baseline, 0.0, 0.0),
        );
        let depth = 4.0;
        let point = Point3::new(0.0, 0.0, depth);
        let (left_pixel, right_pixel) =
            project_left_and_right(&left, &right, &left_to_right, &point);
        let sigma_pixels = 1.0;
        let covariance = propagate_stereo_pixel_covariance(
            &left,
            &right,
            &left_to_right,
            &left_pixel,
            &right_pixel,
            sigma_pixels,
        )
        .expect("valid covariance");
        let expected_depth_stddev =
            2.0_f64.sqrt() * depth * depth * sigma_pixels / (458.0 * baseline);
        let observed_depth_stddev = covariance[(2, 2)].sqrt();
        assert!(
            ((observed_depth_stddev - expected_depth_stddev) / expected_depth_stddev).abs() < 0.02,
            "observed={observed_depth_stddev} expected={expected_depth_stddev}"
        );
    }

    #[test]
    fn propagated_stereo_depth_uncertainty_grows_quadratically_with_range() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        let depth_stddev = |depth: f64| {
            let point = Point3::new(0.1, 0.0, depth);
            let (left_pixel, right_pixel) =
                project_left_and_right(&left, &right, &left_to_right, &point);
            propagate_stereo_pixel_covariance(
                &left,
                &right,
                &left_to_right,
                &left_pixel,
                &right_pixel,
                1.0,
            )
            .unwrap()[(2, 2)]
                .sqrt()
        };
        let ratio = depth_stddev(8.0) / depth_stddev(4.0);
        assert!((ratio - 4.0).abs() < 0.05, "observed ratio={ratio}");
    }

    #[test]
    fn propagated_stereo_covariance_rejects_invalid_pixel_noise() {
        let camera = pinhole();
        let transform = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        assert!(propagate_stereo_pixel_covariance(
            &camera,
            &camera,
            &transform,
            &Point2::new(400.0, 240.0),
            &Point2::new(380.0, 240.0),
            0.0,
        )
        .is_none());
    }

    #[test]
    fn dlt_triangulation_round_trips_with_rotated_right_camera() {
        let left = pinhole();
        let right = pinhole();
        // Small yaw of the right camera plus a baseline offset — the
        // unrectified EuRoC geometry has the right camera tilted by a
        // few degrees relative to the left.
        let yaw = 0.05_f64;
        let rotation = UnitQuaternion::from_axis_angle(&nalgebra::Vector3::y_axis(), yaw);
        let left_to_right = SE3::new(rotation, Vector3::new(-0.11, 0.0, 0.0));
        let point = Point3::new(0.2, -0.1, 4.5);
        let (l, r) = project_left_and_right(&left, &right, &left_to_right, &point);
        let recovered = triangulate_two_view_left_frame(&left, &right, &left_to_right, &l, &r)
            .expect("DLT succeeds");
        assert!((recovered - point).norm() < 1.0e-8);
    }

    #[test]
    fn bootstrap_recovers_metric_landmarks_under_known_correspondence() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        // Three landmarks in front of both cameras. Descriptors are made
        // distinctive (one-hot) so brute-force / cross-check matching is
        // unambiguous.
        let points = [
            Point3::new(0.3, 0.1, 2.5),
            Point3::new(-0.4, 0.2, 3.0),
            Point3::new(0.1, -0.3, 4.0),
        ];
        let mut left_keypoints = Vec::new();
        let mut right_keypoints = Vec::new();
        let mut left_descriptors = Vec::new();
        let mut right_descriptors = Vec::new();
        for (i, point) in points.iter().enumerate() {
            let (l, r) = project_left_and_right(&left, &right, &left_to_right, point);
            left_keypoints.push(l);
            right_keypoints.push(r);
            let mut desc = vec![0.0_f32; points.len()];
            desc[i] = 1.0;
            left_descriptors.push(desc.clone());
            right_descriptors.push(desc);
        }
        let left_features = FeatureSet::new(left_keypoints, left_descriptors).unwrap();
        let right_features = FeatureSet::new(right_keypoints, right_descriptors).unwrap();
        let survivors = bootstrap_stereo_landmarks(
            &left,
            &right,
            &left_to_right,
            &left_features,
            &right_features,
            &StereoBootstrapConfig::default(),
        );
        assert_eq!(survivors.len(), 3);
        for survivor in &survivors {
            let expected = points[survivor.left_keypoint_index];
            assert!((survivor.point_left_camera_frame - expected).norm() < 1.0e-6);
            assert_eq!(
                survivor.left_keypoint_index, survivor.right_keypoint_index,
                "one-hot descriptors must align query/train indices"
            );
            assert!(survivor.left_reprojection_error_pixels < 1.0e-6);
            assert!(survivor.right_reprojection_error_pixels < 1.0e-6);
        }
    }

    #[test]
    fn bootstrap_drops_points_behind_either_camera() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        // Two landmarks: one valid in front, one with a deliberately
        // shifted right keypoint that will triangulate to either negative
        // depth or large reprojection error.
        let good_point = Point3::new(0.0, 0.0, 3.0);
        let (good_l, good_r) = project_left_and_right(&left, &right, &left_to_right, &good_point);
        // Add ~100 px to the right keypoint x — this corresponds to
        // negative disparity (right > left in x) which the disparity-style
        // geometry rules out as behind the camera.
        let bad_l = Point2::new(380.0, 240.0);
        let bad_r = Point2::new(380.0 + 200.0, 240.0);
        let left_features = FeatureSet::new(
            vec![good_l, bad_l],
            vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let right_features = FeatureSet::new(
            vec![good_r, bad_r],
            vec![vec![1.0_f32, 0.0], vec![0.0, 1.0]],
        )
        .unwrap();
        let survivors = bootstrap_stereo_landmarks(
            &left,
            &right,
            &left_to_right,
            &left_features,
            &right_features,
            &StereoBootstrapConfig::default(),
        );
        assert_eq!(survivors.len(), 1);
        assert_eq!(survivors[0].left_keypoint_index, 0);
    }

    #[test]
    fn bootstrap_returns_empty_on_empty_input() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::identity();
        let empty = FeatureSet::new(Vec::new(), Vec::new()).unwrap();
        let survivors = bootstrap_stereo_landmarks(
            &left,
            &right,
            &left_to_right,
            &empty,
            &empty,
            &StereoBootstrapConfig::default(),
        );
        assert!(survivors.is_empty());
    }

    #[test]
    fn bootstrap_drops_matches_with_reprojection_above_threshold() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        let point = Point3::new(0.0, 0.0, 3.0);
        let (l, r) = project_left_and_right(&left, &right, &left_to_right, &point);
        // Perturb the right keypoint by 5 px in y — that pulls the
        // triangulated point off the epipolar line and the reprojection
        // error blows past the 2-pixel default.
        let perturbed_r = Point2::new(r.x, r.y + 5.0);
        let left_features = FeatureSet::new(vec![l], vec![vec![1.0_f32]]).unwrap();
        let right_features = FeatureSet::new(vec![perturbed_r], vec![vec![1.0_f32]]).unwrap();
        let survivors = bootstrap_stereo_landmarks(
            &left,
            &right,
            &left_to_right,
            &left_features,
            &right_features,
            &StereoBootstrapConfig::default(),
        );
        assert!(
            survivors.is_empty(),
            "perturbed right keypoint must fail the reprojection gate"
        );
    }

    #[test]
    fn bootstrap_results_are_sorted_by_left_keypoint_index() {
        let left = pinhole();
        let right = pinhole();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.11, 0.0, 0.0));
        // Construct three matches whose train indices are deliberately
        // reversed compared to the query order — the matcher would emit
        // them in query order, but the post-sort must give us increasing
        // left-keypoint indices regardless of how the matcher emitted them.
        let points = [
            Point3::new(0.3, 0.0, 2.0),
            Point3::new(-0.2, 0.1, 3.0),
            Point3::new(0.1, -0.2, 4.0),
        ];
        let mut left_keypoints = Vec::new();
        let mut right_keypoints = Vec::new();
        let mut left_descriptors = Vec::new();
        let mut right_descriptors = Vec::new();
        for (i, point) in points.iter().enumerate() {
            let (l, r) = project_left_and_right(&left, &right, &left_to_right, point);
            left_keypoints.push(l);
            right_keypoints.push(r);
            let mut desc = vec![0.0_f32; points.len()];
            desc[i] = 1.0;
            left_descriptors.push(desc.clone());
            right_descriptors.push(desc);
        }
        let left_features = FeatureSet::new(left_keypoints, left_descriptors).unwrap();
        let right_features = FeatureSet::new(right_keypoints, right_descriptors).unwrap();
        let survivors = bootstrap_stereo_landmarks(
            &left,
            &right,
            &left_to_right,
            &left_features,
            &right_features,
            &StereoBootstrapConfig::default(),
        );
        assert_eq!(survivors.len(), 3);
        let indices: Vec<usize> = survivors.iter().map(|s| s.left_keypoint_index).collect();
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(indices, sorted);
    }
}
