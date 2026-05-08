//! Classical two-view geometry building blocks.
//!
//! The module exposes a small, testable pipeline:
//! 1. Normalize pixel correspondences with camera intrinsics.
//! 2. Estimate the essential matrix with the 8-point algorithm in a
//!    Sampson-distance scored RANSAC loop (`EssentialRansac`).
//! 3. Decompose the essential matrix into the four (R, t) candidates and pick
//!    the one with the most correspondences in front of both cameras
//!    (`recover_relative_pose`).
//! 4. Compose the above as a `RelativePoseEstimator`, optionally applying a
//!    caller-supplied translation scale.
//!
//! These components are intentionally independent of `Frame`, `FrameId`, or
//! the tracking pipeline so they can be used as the geometric core of any
//! `VisualOdometryFrontend` implementation. The
//! `EssentialMatrixVisualOdometryFrontend` in the top-level `visloc-rs` crate
//! wires this module into the existing tracking layer.

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::geometry::SE3;
use visloc_core::types::Camera;

/// One pixel-space correspondence between a previous and a current frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoViewCorrespondence {
    pub previous_xy: Point2<f64>,
    pub current_xy: Point2<f64>,
}

impl TwoViewCorrespondence {
    pub fn new(previous_xy: Point2<f64>, current_xy: Point2<f64>) -> Self {
        Self {
            previous_xy,
            current_xy,
        }
    }
}

/// Output of the essential-matrix RANSAC loop.
#[derive(Debug, Clone, PartialEq)]
pub struct EssentialRansacReport {
    pub essential: Matrix3<f64>,
    pub inliers: Vec<usize>,
    pub mean_sampson_error: f64,
}

/// Output of relative-pose recovery: rotation, unit translation direction,
/// applied scale, the implied `SE3` previous-to-current transform, inlier
/// indices, and Sampson diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePose {
    pub previous_to_current: SE3,
    pub translation_unit: Vector3<f64>,
    pub translation_scale: f64,
    pub inliers: Vec<usize>,
    pub mean_sampson_error: f64,
}

/// Estimator for an essential matrix from pixel correspondences plus
/// intrinsics. Implementors return `None` when the input is degenerate.
pub trait EssentialMatrixEstimator {
    fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<Matrix3<f64>>;
    fn minimum_correspondences(&self) -> usize;
}

/// Hartley-normalized 8-point essential-matrix estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EightPointEssentialMatrixEstimator {
    pub min_correspondences: usize,
}

impl Default for EightPointEssentialMatrixEstimator {
    fn default() -> Self {
        Self {
            min_correspondences: 8,
        }
    }
}

impl EssentialMatrixEstimator for EightPointEssentialMatrixEstimator {
    fn minimum_correspondences(&self) -> usize {
        self.min_correspondences
    }

    fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<Matrix3<f64>> {
        if correspondences.len() < self.min_correspondences {
            return None;
        }

        let normalized = normalize_pairs(correspondences, camera)?;
        let (previous_normalization, previous_points) =
            hartley_normalization(normalized.iter().map(|(p, _)| *p))?;
        let (current_normalization, current_points) =
            hartley_normalization(normalized.iter().map(|(_, c)| *c))?;

        let mut a = DMatrix::<f64>::zeros(normalized.len(), 9);
        for row in 0..normalized.len() {
            let (x, y) = (previous_points[row].x, previous_points[row].y);
            let (xp, yp) = (current_points[row].x, current_points[row].y);
            a[(row, 0)] = xp * x;
            a[(row, 1)] = xp * y;
            a[(row, 2)] = xp;
            a[(row, 3)] = yp * x;
            a[(row, 4)] = yp * y;
            a[(row, 5)] = yp;
            a[(row, 6)] = x;
            a[(row, 7)] = y;
            a[(row, 8)] = 1.0;
        }

        // The 8-point linear system A * f = 0 has 9 unknowns. When there are
        // fewer than 9 rows the thin SVD that nalgebra computes for non-square
        // inputs drops the last right singular vector — the very direction we
        // want. Multiply by A^T A so the SVD always operates on a 9x9 matrix.
        let ata = a.transpose() * a;
        let svd = ata.svd(true, true);
        let v_t = svd.v_t?;
        let last = v_t.row(v_t.nrows() - 1);
        let mut essential_normalized = Matrix3::new(
            last[0], last[1], last[2], last[3], last[4], last[5], last[6], last[7], last[8],
        );

        // Project E_normalized onto the essential manifold.
        let essential_norm_svd = essential_normalized.svd(true, true);
        let u_n = essential_norm_svd.u?;
        let v_t_n = essential_norm_svd.v_t?;
        let s_n =
            (essential_norm_svd.singular_values[0] + essential_norm_svd.singular_values[1]) * 0.5;
        let constrained_n = Matrix3::from_diagonal(&Vector3::new(s_n, s_n, 0.0));
        essential_normalized = u_n * constrained_n * v_t_n;

        let essential_calibrated: Matrix3<f64> =
            current_normalization.transpose() * essential_normalized * previous_normalization;
        Some(essential_calibrated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialRansacConfig {
    pub iterations: usize,
    /// Sampson distance threshold in normalized image-plane units. A 1-pixel
    /// reprojection error at focal length `f` corresponds to a normalized
    /// distance of `1.0 / f`.
    pub sampson_threshold: f64,
    pub seed: u64,
}

impl Default for EssentialRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 256,
            sampson_threshold: 5.0e-3,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialRansac<E = EightPointEssentialMatrixEstimator> {
    pub estimator: E,
    pub config: EssentialRansacConfig,
}

impl Default for EssentialRansac {
    fn default() -> Self {
        Self {
            estimator: EightPointEssentialMatrixEstimator::default(),
            config: EssentialRansacConfig::default(),
        }
    }
}

impl<E> EssentialRansac<E>
where
    E: EssentialMatrixEstimator,
{
    pub fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<EssentialRansacReport> {
        let sample_size = self.estimator.minimum_correspondences();
        if correspondences.len() < sample_size {
            return None;
        }

        let mut rng = SmallRng::seed_from_u64(self.config.seed);
        let mut indices: Vec<usize> = (0..correspondences.len()).collect();
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_essential: Option<Matrix3<f64>> = None;
        let threshold_sq = self.config.sampson_threshold * self.config.sampson_threshold;

        for _ in 0..self.config.iterations {
            indices.shuffle(&mut rng);
            let sample: Vec<TwoViewCorrespondence> = indices[..sample_size]
                .iter()
                .map(|&i| correspondences[i])
                .collect();

            let Some(candidate) = self.estimator.estimate(&sample, camera) else {
                continue;
            };

            let inliers = score_inliers(&candidate, correspondences, camera, threshold_sq);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_essential = Some(candidate);
            }
        }

        let essential = best_essential?;
        if best_inliers.len() < sample_size {
            return None;
        }

        let inlier_correspondences: Vec<TwoViewCorrespondence> =
            best_inliers.iter().map(|&i| correspondences[i]).collect();
        let refined = self
            .estimator
            .estimate(&inlier_correspondences, camera)
            .unwrap_or(essential);
        let final_inliers = score_inliers(&refined, correspondences, camera, threshold_sq);
        let final_inliers = if final_inliers.len() >= best_inliers.len() {
            final_inliers
        } else {
            best_inliers
        };
        let mean = mean_sampson_error(&refined, correspondences, camera, &final_inliers);

        Some(EssentialRansacReport {
            essential: refined,
            inliers: final_inliers,
            mean_sampson_error: mean,
        })
    }
}

/// Composes essential-matrix RANSAC with relative-pose recovery, applying a
/// caller-controlled translation scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePoseEstimator<E = EightPointEssentialMatrixEstimator> {
    pub ransac: EssentialRansac<E>,
    /// Default translation scale applied when no per-call scale is supplied.
    /// Stays at 1.0 unless the caller knows the metric scale (e.g., from a
    /// GNSS displacement, the previous frame's translation, or a configured
    /// default).
    pub default_translation_scale: f64,
}

impl Default for RelativePoseEstimator {
    fn default() -> Self {
        Self {
            ransac: EssentialRansac::default(),
            default_translation_scale: 1.0,
        }
    }
}

impl<E> RelativePoseEstimator<E>
where
    E: EssentialMatrixEstimator,
{
    pub fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<RelativePose> {
        self.estimate_with_scale(correspondences, camera, self.default_translation_scale)
    }

    pub fn estimate_with_scale(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        translation_scale: f64,
    ) -> Option<RelativePose> {
        let report = self.ransac.estimate(correspondences, camera)?;
        let (rotation, translation_unit) =
            recover_relative_pose(&report.essential, correspondences, camera, &report.inliers)?;
        let se3 = SE3::new(rotation, translation_unit * translation_scale);
        Some(RelativePose {
            previous_to_current: se3,
            translation_unit,
            translation_scale,
            inliers: report.inliers,
            mean_sampson_error: report.mean_sampson_error,
        })
    }
}

/// Decompose an essential matrix into the (R, t_unit) pair that puts the most
/// inlier correspondences in front of both cameras.
pub fn recover_relative_pose(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
) -> Option<(UnitQuaternion<f64>, Vector3<f64>)> {
    let svd = essential.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let mut r1 = u * w * v_t;
    let mut r2 = u * w.transpose() * v_t;
    if r1.determinant() < 0.0 {
        r1 = -r1;
    }
    if r2.determinant() < 0.0 {
        r2 = -r2;
    }
    let t_unit = u.column(2).into_owned();

    let candidates = [(r1, t_unit), (r1, -t_unit), (r2, t_unit), (r2, -t_unit)];
    let mut best: Option<(Matrix3<f64>, Vector3<f64>)> = None;
    let mut best_score: i64 = -1;
    for (rotation, translation) in candidates {
        let score = cheirality_score(&rotation, &translation, correspondences, camera, inliers);
        if score > best_score {
            best_score = score;
            best = Some((rotation, translation));
        }
    }

    let (rotation, translation) = best?;
    if best_score <= 0 {
        return None;
    }
    let rotation = UnitQuaternion::from_matrix(&rotation);
    Some((rotation, translation))
}

fn cheirality_score(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
) -> i64 {
    let p_prev = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p_curr = Matrix3x4::zeros();
    p_curr.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
    p_curr.fixed_view_mut::<3, 1>(0, 3).copy_from(translation);

    let mut score: i64 = 0;
    for &index in inliers {
        let correspondence = &correspondences[index];
        let Some(prev) = camera.normalize_pixel(&correspondence.previous_xy) else {
            continue;
        };
        let Some(curr) = camera.normalize_pixel(&correspondence.current_xy) else {
            continue;
        };

        let mut a = DMatrix::<f64>::zeros(4, 4);
        for column in 0..4 {
            a[(0, column)] = prev.x * p_prev[(2, column)] - p_prev[(0, column)];
            a[(1, column)] = prev.y * p_prev[(2, column)] - p_prev[(1, column)];
            a[(2, column)] = curr.x * p_curr[(2, column)] - p_curr[(0, column)];
            a[(3, column)] = curr.y * p_curr[(2, column)] - p_curr[(1, column)];
        }
        let svd = a.svd(true, true);
        let Some(v_t) = svd.v_t else {
            continue;
        };
        let solution = v_t.row(v_t.nrows() - 1);
        let w = solution[3];
        if w.abs() < 1e-12 {
            continue;
        }
        let world = Vector3::new(solution[0] / w, solution[1] / w, solution[2] / w);
        let camera_curr = rotation * world + translation;
        if world.z > 0.0 && camera_curr.z > 0.0 {
            score += 1;
        }
    }
    score
}

/// Hartley normalization for a set of 2D points: translate so the centroid is
/// at the origin and scale so the average distance to the origin is sqrt(2).
/// Returns the 3x3 transform `T` (so `T * [x, y, 1]` gives the normalized
/// point) and the normalized points themselves.
fn hartley_normalization<I>(points: I) -> Option<(Matrix3<f64>, Vec<Point2<f64>>)>
where
    I: IntoIterator<Item = Point2<f64>>,
{
    let collected: Vec<Point2<f64>> = points.into_iter().collect();
    if collected.is_empty() {
        return None;
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for point in &collected {
        mean_x += point.x;
        mean_y += point.y;
    }
    let count = collected.len() as f64;
    mean_x /= count;
    mean_y /= count;

    let mut mean_distance = 0.0;
    for point in &collected {
        let dx = point.x - mean_x;
        let dy = point.y - mean_y;
        mean_distance += (dx * dx + dy * dy).sqrt();
    }
    mean_distance /= count;
    if mean_distance < 1.0e-12 {
        return None;
    }
    let scale = std::f64::consts::SQRT_2 / mean_distance;
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * mean_x,
        0.0,
        scale,
        -scale * mean_y,
        0.0,
        0.0,
        1.0,
    );

    let normalized = collected
        .into_iter()
        .map(|point| Point2::new(scale * (point.x - mean_x), scale * (point.y - mean_y)))
        .collect();
    Some((transform, normalized))
}

fn normalize_pairs(
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<Vec<(Point2<f64>, Point2<f64>)>> {
    correspondences
        .iter()
        .map(|correspondence| {
            Some((
                camera.normalize_pixel(&correspondence.previous_xy)?,
                camera.normalize_pixel(&correspondence.current_xy)?,
            ))
        })
        .collect()
}

fn score_inliers(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    threshold_sq: f64,
) -> Vec<usize> {
    let mut inliers = Vec::with_capacity(correspondences.len());
    for (index, correspondence) in correspondences.iter().enumerate() {
        let Some(distance_sq) = sampson_distance_squared(essential, correspondence, camera) else {
            continue;
        };
        if distance_sq <= threshold_sq {
            inliers.push(index);
        }
    }
    inliers
}

fn mean_sampson_error(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
) -> f64 {
    if inliers.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for &index in inliers {
        if let Some(distance_sq) =
            sampson_distance_squared(essential, &correspondences[index], camera)
        {
            total += distance_sq.sqrt();
            count += 1.0;
        }
    }
    if count > 0.0 {
        total / count
    } else {
        f64::INFINITY
    }
}

fn sampson_distance_squared(
    essential: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
    camera: &Camera,
) -> Option<f64> {
    let prev = camera.normalize_pixel(&correspondence.previous_xy)?;
    let curr = camera.normalize_pixel(&correspondence.current_xy)?;
    let prev_h = Vector3::new(prev.x, prev.y, 1.0);
    let curr_h = Vector3::new(curr.x, curr.y, 1.0);
    let e_prev = essential * prev_h;
    let et_curr = essential.transpose() * curr_h;
    let numerator = curr_h.dot(&e_prev).powi(2);
    let denominator = e_prev.x.powi(2) + e_prev.y.powi(2) + et_curr.x.powi(2) + et_curr.y.powi(2);
    if denominator < 1e-18 {
        return None;
    }
    Some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion};
    use visloc_core::geometry::Pose;

    fn synthetic_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project(pose: &Pose, camera: &Camera, point: &Point3<f64>) -> Point2<f64> {
        camera
            .project(&pose.transform_world_point(point))
            .expect("synthetic point must project in front of the camera")
    }

    fn synthetic_world_points() -> Vec<Point3<f64>> {
        vec![
            Point3::new(-1.0, -1.0, 5.0),
            Point3::new(1.0, -1.0, 5.0),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 5.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.7, 0.4, 4.5),
            Point3::new(0.6, 0.8, 5.5),
            Point3::new(-0.3, -0.6, 4.8),
            Point3::new(0.2, 0.2, 6.5),
        ]
    }

    fn correspondences(
        previous_pose: &Pose,
        current_pose: &Pose,
        camera: &Camera,
        points: &[Point3<f64>],
    ) -> Vec<TwoViewCorrespondence> {
        points
            .iter()
            .map(|point| TwoViewCorrespondence {
                previous_xy: project(previous_pose, camera, point),
                current_xy: project(current_pose, camera, point),
            })
            .collect()
    }

    #[test]
    fn essential_ransac_recovers_pure_translation() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());

        let estimator = RelativePoseEstimator::default();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, 0.3)
            .expect("relative pose must be recovered");

        let translation = pose.previous_to_current.translation;
        assert!(
            (translation - Vector3::new(-0.3, 0.0, 0.0)).norm() < 5.0e-3,
            "translation drifted: {translation:?}"
        );
        let rotation = pose.previous_to_current.rotation.angle();
        assert!(
            rotation < 5.0e-3,
            "rotation should be near zero: {rotation}"
        );
        assert!(pose.inliers.len() >= 8);
        assert!(pose.mean_sampson_error < 5.0e-3);
    }

    #[test]
    fn essential_ransac_recovers_translation_with_yaw() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.05);
        let current_world_to_camera = SE3::new(yaw, Vector3::new(-0.2, 0.0, -0.05));
        let current = Pose {
            world_to_camera: current_world_to_camera.clone(),
        };
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());

        let estimator = RelativePoseEstimator::default();
        let scale = current_world_to_camera.translation.norm();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, scale)
            .expect("relative pose must be recovered");

        assert!(pose.inliers.len() >= 8);
        let translation_error =
            (pose.previous_to_current.translation - current_world_to_camera.translation).norm();
        assert!(
            translation_error < 5.0e-3,
            "translation drifted: error={translation_error}"
        );
        let rotation_error = pose
            .previous_to_current
            .rotation
            .rotation_to(&current_world_to_camera.rotation)
            .angle()
            .abs();
        assert!(
            rotation_error < 5.0e-3,
            "rotation drifted: error_rad={rotation_error}"
        );
    }

    #[test]
    fn essential_ransac_recovers_pure_translation_with_eight_points() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.30, 0.0, 0.0));
        let points = [
            Point3::new(-1.0, -1.0, 4.5),
            Point3::new(1.0, -1.0, 4.6),
            Point3::new(-1.0, 1.0, 5.5),
            Point3::new(1.0, 1.0, 5.4),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.6, 0.4, 4.8),
            Point3::new(0.4, 0.7, 5.2),
        ];
        let correspondences = correspondences(&previous, &current, &camera, &points);

        let estimator = RelativePoseEstimator::default();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, 0.30)
            .expect("relative pose must be recovered");
        let translation = pose.previous_to_current.translation;
        assert!(
            (translation - Vector3::new(-0.30, 0.0, 0.0)).norm() < 5.0e-3,
            "translation drifted: {translation:?}"
        );
        assert!(pose.inliers.len() >= 8);
    }

    #[test]
    fn essential_ransac_returns_none_for_too_few_points() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0));
        let mut points = synthetic_world_points();
        points.truncate(6);
        let correspondences = correspondences(&previous, &current, &camera, &points);

        let estimator = RelativePoseEstimator::default();
        assert!(estimator.estimate(&correspondences, &camera).is_none());
    }
}
