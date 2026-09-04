//! Non-central absolute pose for calibrated multi-camera rigs.
//!
//! A generalized camera observation is a ray with a known origin in the rig
//! frame.  Given world point `X`, rig pose `(R, t)`, ray direction `f`, and ray
//! origin `c`, the absolute-pose constraint is
//!
//! ```text
//! f × (R X + t - c) = 0.
//! ```
//!
//! Unlike selecting the best central-camera PnP, every hypothesis below pools
//! observations from distinct sensor origins and estimates one `world -> rig`
//! pose.  The linear initializer is the generalized-camera analogue of DLT;
//! RANSAC scores the true per-sensor pixel residual and the final
//! Gauss-Newton pass refines the shared rig pose.  The formulation follows the
//! non-central absolute-pose model used by OpenGV's GP3P/GPnP implementations.

use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::Camera;

use crate::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert};
use crate::ransac::{PnPRansac, RobustPoseEstimator};

/// One calibrated sensor rigidly attached to a generalized camera rig.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSensor {
    pub camera: Camera,
    /// Fixed transform `T_sensor<-rig`.
    pub sensor_from_rig: SE3,
}

/// Validated sensor geometry used by generalized absolute pose.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedCameraRig {
    sensors: Vec<RigSensor>,
    origins_rig: Vec<Point3<f64>>,
}

impl GeneralizedCameraRig {
    /// Construct a rig. Returns `None` for an empty rig, invalid intrinsics, or
    /// non-finite extrinsics.
    pub fn new(sensors: Vec<RigSensor>) -> Option<Self> {
        if sensors.is_empty() {
            return None;
        }
        let mut origins_rig = Vec::with_capacity(sensors.len());
        for sensor in &sensors {
            let intrinsics = sensor.camera.intrinsics()?;
            if ![intrinsics.0, intrinsics.1, intrinsics.2, intrinsics.3]
                .iter()
                .all(|value| value.is_finite())
                || intrinsics.0 <= 0.0
                || intrinsics.1 <= 0.0
                || !sensor
                    .sensor_from_rig
                    .translation
                    .iter()
                    .all(|value| value.is_finite())
            {
                return None;
            }
            let rig_from_sensor = sensor.sensor_from_rig.inverse();
            let origin = Point3::from(rig_from_sensor.translation);
            if !origin.coords.iter().all(|value| value.is_finite()) {
                return None;
            }
            origins_rig.push(origin);
        }
        Some(Self {
            sensors,
            origins_rig,
        })
    }

    pub fn sensors(&self) -> &[RigSensor] {
        &self.sensors
    }

    pub fn sensor(&self, index: usize) -> Option<&RigSensor> {
        self.sensors.get(index)
    }

    pub fn sensor_origin_rig(&self, index: usize) -> Option<&Point3<f64>> {
        self.origins_rig.get(index)
    }

    fn ray_rig(
        &self,
        correspondence: &GeneralizedCorrespondence2D3D,
    ) -> Option<(Point3<f64>, Vector3<f64>)> {
        let sensor = self.sensor(correspondence.sensor_index)?;
        let normalized = sensor.camera.normalize_pixel(&correspondence.point2d)?;
        let bearing_sensor = Vector3::new(normalized.x, normalized.y, 1.0).try_normalize(1e-15)?;
        let bearing_rig = sensor
            .sensor_from_rig
            .rotation
            .inverse_transform_vector(&bearing_sensor)
            .try_normalize(1e-15)?;
        Some((
            *self.sensor_origin_rig(correspondence.sensor_index)?,
            bearing_rig,
        ))
    }

    fn project(
        &self,
        pose: &Pose,
        correspondence: &GeneralizedCorrespondence2D3D,
    ) -> Option<Point2<f64>> {
        let sensor = self.sensor(correspondence.sensor_index)?;
        let point_rig = pose.transform_world_point(&correspondence.point3d);
        let point_sensor = sensor.sensor_from_rig.transform_point(&point_rig);
        sensor.camera.project(&point_sensor)
    }
}

/// One world-point to rig-sensor pixel correspondence.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedCorrespondence2D3D {
    pub sensor_index: usize,
    pub point2d: Point2<f64>,
    pub point3d: Point3<f64>,
    pub confidence: Option<f32>,
}

/// Linear non-central absolute-pose initializer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedDltPoseEstimator {
    /// Minimum ray-origin span required by a solve sample. A zero-origin
    /// central-camera sample is deliberately rejected instead of silently
    /// degenerating to ordinary PnP.
    pub minimum_origin_span: f64,
    /// Maximum singular-value anisotropy of the recovered scaled rotation.
    pub maximum_rotation_anisotropy: f64,
}

impl Default for GeneralizedDltPoseEstimator {
    fn default() -> Self {
        Self {
            minimum_origin_span: 1.0e-9,
            maximum_rotation_anisotropy: 0.35,
        }
    }
}

impl GeneralizedDltPoseEstimator {
    pub const MINIMUM_CORRESPONDENCES: usize = 6;

    /// Estimate one shared `world -> rig` pose. Every accepted solve contains
    /// at least two distinct ray origins.
    pub fn estimate_pose(
        &self,
        rig: &GeneralizedCameraRig,
        correspondences: &[GeneralizedCorrespondence2D3D],
    ) -> Option<Pose> {
        if correspondences.len() < Self::MINIMUM_CORRESPONDENCES
            || !self.minimum_origin_span.is_finite()
            || self.minimum_origin_span < 0.0
            || !self.maximum_rotation_anisotropy.is_finite()
            || self.maximum_rotation_anisotropy < 0.0
        {
            return None;
        }

        let rays = correspondences
            .iter()
            .map(|correspondence| rig.ray_rig(correspondence))
            .collect::<Option<Vec<_>>>()?;
        let origin_span = rays
            .iter()
            .enumerate()
            .flat_map(|(left, (origin_left, _))| {
                rays[(left + 1)..]
                    .iter()
                    .map(move |(origin_right, _)| (origin_left - origin_right).norm())
            })
            .fold(0.0_f64, f64::max);
        if origin_span <= self.minimum_origin_span {
            return None;
        }

        let count = correspondences.len() as f64;
        let centroid = correspondences
            .iter()
            .fold(Vector3::zeros(), |sum, correspondence| {
                sum + correspondence.point3d.coords
            })
            / count;
        let mean_distance = correspondences
            .iter()
            .map(|correspondence| (correspondence.point3d.coords - centroid).norm())
            .sum::<f64>()
            / count;
        if !mean_distance.is_finite() || mean_distance <= 1.0e-12 {
            return None;
        }
        let world_scale = 3.0_f64.sqrt() / mean_distance;

        // Three cross-product rows are retained per ray. Each 3-row block has
        // rank two, while SVD least squares handles that intentional
        // redundancy without choosing a numerically fragile row pair.
        let mut design = DMatrix::<f64>::zeros(correspondences.len() * 3, 12);
        let mut rhs = DVector::<f64>::zeros(correspondences.len() * 3);
        for (index, (correspondence, (origin, bearing))) in
            correspondences.iter().zip(&rays).enumerate()
        {
            let point = world_scale * (correspondence.point3d.coords - centroid);
            let cross = skew(bearing);
            let cross_origin = cross * origin.coords;
            for equation in 0..3 {
                let row = index * 3 + equation;
                for output_axis in 0..3 {
                    for world_axis in 0..3 {
                        design[(row, output_axis * 3 + world_axis)] =
                            cross[(equation, output_axis)] * point[world_axis];
                    }
                    design[(row, 9 + output_axis)] = cross[(equation, output_axis)];
                }
                rhs[row] = cross_origin[equation];
            }
        }

        let svd = design.svd(true, true);
        let solution = svd.solve(&rhs, 1.0e-12).ok()?;
        if solution.len() != 12 || !solution.iter().all(|value| value.is_finite()) {
            return None;
        }
        let scaled_rotation = Matrix3::new(
            solution[0],
            solution[1],
            solution[2],
            solution[3],
            solution[4],
            solution[5],
            solution[6],
            solution[7],
            solution[8],
        );
        let rotation_svd = scaled_rotation.svd(true, true);
        let singular = rotation_svd.singular_values;
        let rotation_scale = singular.iter().sum::<f64>() / 3.0;
        if !rotation_scale.is_finite() || rotation_scale <= 1.0e-12 {
            return None;
        }
        let anisotropy = (singular.max() - singular.min()) / rotation_scale;
        if anisotropy > self.maximum_rotation_anisotropy {
            return None;
        }
        let u = rotation_svd.u?;
        let v_t = rotation_svd.v_t?;
        let mut sign = Matrix3::identity();
        sign[(2, 2)] = (u * v_t).determinant().signum();
        if sign[(2, 2)] == 0.0 {
            return None;
        }
        let rotation_matrix = u * sign * v_t;
        let translation_normalized = Vector3::new(solution[9], solution[10], solution[11]);
        let translation = translation_normalized - rotation_matrix * centroid;
        let pose =
            Pose::from_world_to_camera(UnitQuaternion::from_matrix(&rotation_matrix), translation);

        // Reject the reflected/behind-camera branch and gross linear failures.
        let positive = correspondences
            .iter()
            .filter(|correspondence| rig.project(&pose, correspondence).is_some())
            .count();
        (positive * 2 >= correspondences.len()).then_some(pose)
    }
}

/// Shared-rig nonlinear pose refiner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedGaussNewtonPoseRefiner {
    pub iterations: usize,
    pub damping: f64,
    pub finite_difference_epsilon: f64,
    pub minimum_error_reduction: f64,
}

impl Default for GeneralizedGaussNewtonPoseRefiner {
    fn default() -> Self {
        Self {
            iterations: 10,
            damping: 1.0e-6,
            finite_difference_epsilon: 1.0e-6,
            minimum_error_reduction: 1.0e-10,
        }
    }
}

impl GeneralizedGaussNewtonPoseRefiner {
    pub fn refine_pose(
        &self,
        rig: &GeneralizedCameraRig,
        initial_pose: &Pose,
        correspondences: &[GeneralizedCorrespondence2D3D],
    ) -> Option<Pose> {
        if correspondences.is_empty()
            || self.iterations == 0
            || !self.damping.is_finite()
            || self.damping < 0.0
            || !self.finite_difference_epsilon.is_finite()
            || self.finite_difference_epsilon <= 0.0
        {
            return None;
        }
        let mut pose = initial_pose.clone();
        let mut best_error = mean_squared_error(rig, &pose, correspondences)?;
        for _ in 0..self.iterations {
            let residual = residuals(rig, &pose, correspondences)?;
            let mut jacobian = DMatrix::<f64>::zeros(residual.len(), 6);
            for parameter in 0..6 {
                let mut delta = DVector::<f64>::zeros(6);
                delta[parameter] = self.finite_difference_epsilon;
                let perturbed_pose = perturb_pose(&pose, &delta);
                let perturbed = residuals(rig, &perturbed_pose, correspondences)?;
                jacobian.set_column(
                    parameter,
                    &((perturbed - &residual) / self.finite_difference_epsilon),
                );
            }
            let transpose = jacobian.transpose();
            let mut hessian = &transpose * &jacobian;
            for diagonal in 0..6 {
                hessian[(diagonal, diagonal)] += self.damping;
            }
            let gradient = transpose * residual;
            let step = hessian.lu().solve(&(-gradient))?;
            if !step.iter().all(|value| value.is_finite()) {
                break;
            }
            let candidate = perturb_pose(&pose, &step);
            let Some(candidate_error) = mean_squared_error(rig, &candidate, correspondences) else {
                break;
            };
            if candidate_error + self.minimum_error_reduction >= best_error {
                break;
            }
            pose = candidate;
            best_error = candidate_error;
        }
        Some(pose)
    }
}

/// Generalized-camera RANSAC result.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneralizedRansacReport {
    pub pose: Pose,
    pub inliers: Vec<usize>,
    pub inlier_reprojection_errors: Vec<f64>,
    pub mean_reprojection_error: f64,
    pub median_reprojection_error: f64,
    pub max_reprojection_error: f64,
    pub refinement_applied: bool,
}

/// Deterministic pixel-space RANSAC for one generalized rig frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneralizedPnPRansac {
    pub pose_estimator: GeneralizedDltPoseEstimator,
    pub pose_refiner: Option<GeneralizedGaussNewtonPoseRefiner>,
    pub iterations: usize,
    pub reprojection_threshold: f64,
    pub confidence: Option<f64>,
    pub seed: u64,
}

impl Default for GeneralizedPnPRansac {
    fn default() -> Self {
        Self {
            pose_estimator: GeneralizedDltPoseEstimator::default(),
            pose_refiner: Some(GeneralizedGaussNewtonPoseRefiner::default()),
            iterations: 256,
            reprojection_threshold: 4.0,
            confidence: Some(0.999),
            seed: 7,
        }
    }
}

impl GeneralizedPnPRansac {
    pub fn estimate(
        &self,
        rig: &GeneralizedCameraRig,
        correspondences: &[GeneralizedCorrespondence2D3D],
    ) -> Option<GeneralizedRansacReport> {
        self.estimate_with_pose_prior(rig, correspondences, None)
    }

    pub fn estimate_with_pose_prior(
        &self,
        rig: &GeneralizedCameraRig,
        correspondences: &[GeneralizedCorrespondence2D3D],
        pose_prior: Option<&Pose>,
    ) -> Option<GeneralizedRansacReport> {
        let sample_size = GeneralizedDltPoseEstimator::MINIMUM_CORRESPONDENCES;
        if correspondences.len() < sample_size
            || self.iterations == 0
            || !self.reprojection_threshold.is_finite()
            || self.reprojection_threshold <= 0.0
        {
            return None;
        }

        let mut best_pose = pose_prior.cloned();
        let mut best_score = pose_prior
            .map(|pose| score_pose(rig, pose, correspondences, self.reprojection_threshold))
            .unwrap_or_else(GeneralizedScore::empty);
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let mut indices = (0..correspondences.len()).collect::<Vec<_>>();
        let mut required_iterations = self.iterations;
        for iteration in 0..self.iterations {
            indices.shuffle(&mut rng);
            let sample = indices
                .iter()
                .take(sample_size)
                .map(|index| correspondences[*index].clone())
                .collect::<Vec<_>>();
            let Some(pose) = self.pose_estimator.estimate_pose(rig, &sample) else {
                continue;
            };
            let score = score_pose(rig, &pose, correspondences, self.reprojection_threshold);
            if score.is_better_than(&best_score) {
                best_pose = Some(pose);
                best_score = score;
                if let Some(confidence) = self.confidence {
                    let inlier_ratio =
                        best_score.inliers.len() as f64 / correspondences.len() as f64;
                    if inlier_ratio >= 1.0 {
                        required_iterations = iteration + 1;
                    } else if inlier_ratio > 0.0 && confidence > 0.0 && confidence < 1.0 {
                        let denominator = (1.0 - inlier_ratio.powi(sample_size as i32)).ln();
                        if denominator < -1.0e-12 {
                            required_iterations = required_iterations
                                .min(((1.0 - confidence).ln() / denominator).ceil().max(1.0)
                                    as usize);
                        }
                    }
                }
            }
            if iteration + 1 >= required_iterations {
                break;
            }
        }

        // A small physical rig baseline makes the inhomogeneous generalized
        // DLT poorly conditioned under pixel noise. Generate additional
        // minimal P3P hypotheses independently inside each central sensor,
        // transform them to the rig frame, then score them against *all*
        // sensors. Sensor choice affects hypothesis generation only; the
        // accepted body pose and nonlinear refinement remain fully pooled.
        for sensor_index in 0..rig.sensors().len() {
            let sensor_correspondences = correspondences
                .iter()
                .filter(|correspondence| correspondence.sensor_index == sensor_index)
                .map(|correspondence| Correspondence2D3D {
                    point2d: correspondence.point2d,
                    point3d: correspondence.point3d,
                    confidence: correspondence.confidence,
                })
                .collect::<Vec<_>>();
            if sensor_correspondences.len() < 4 {
                continue;
            }
            let central = PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: self.iterations.max(32),
                reprojection_threshold: self.reprojection_threshold,
                seed: self.seed.wrapping_add(sensor_index as u64),
                early_stop_min_iterations: 16,
                early_stop_inlier_ratio: Some(0.9),
                confidence: self.confidence,
            };
            let Some(report) =
                central.estimate(&sensor_correspondences, &rig.sensors()[sensor_index].camera)
            else {
                continue;
            };
            let world_to_rig = rig.sensors()[sensor_index]
                .sensor_from_rig
                .inverse()
                .compose(&report.pose.world_to_camera);
            let pose = Pose {
                world_to_camera: world_to_rig,
            };
            let score = score_pose(rig, &pose, correspondences, self.reprojection_threshold);
            if score.is_better_than(&best_score) {
                best_pose = Some(pose);
                best_score = score;
            }
        }

        if best_score.inliers.len() < sample_size {
            return None;
        }
        let inliers = best_score
            .inliers
            .iter()
            .map(|index| correspondences[*index].clone())
            .collect::<Vec<_>>();
        let refit_pose = self
            .pose_estimator
            .estimate_pose(rig, &inliers)
            .filter(|pose| {
                let score = score_pose(rig, pose, correspondences, self.reprojection_threshold);
                !best_score.is_better_than(&score)
            });
        let mut pose = refit_pose.or(best_pose)?;
        let refit_score = score_pose(rig, &pose, correspondences, self.reprojection_threshold);
        let mut refinement_applied = false;
        if let Some(refiner) = self.pose_refiner {
            if let Some(candidate) = refiner.refine_pose(rig, &pose, &inliers) {
                let candidate_score = score_pose(
                    rig,
                    &candidate,
                    correspondences,
                    self.reprojection_threshold,
                );
                if candidate_score.is_better_than(&refit_score)
                    || (candidate_score.inliers.len() == refit_score.inliers.len()
                        && candidate_score.mean_error <= refit_score.mean_error + 1.0e-12)
                {
                    pose = candidate;
                    refinement_applied = true;
                }
            }
        }
        let score = score_pose(rig, &pose, correspondences, self.reprojection_threshold);
        Some(GeneralizedRansacReport {
            pose,
            inliers: score.inliers,
            inlier_reprojection_errors: score.errors,
            mean_reprojection_error: score.mean_error,
            median_reprojection_error: score.median_error,
            max_reprojection_error: score.max_error,
            refinement_applied,
        })
    }
}

#[derive(Debug, Clone)]
struct GeneralizedScore {
    inliers: Vec<usize>,
    errors: Vec<f64>,
    mean_error: f64,
    median_error: f64,
    max_error: f64,
}

impl GeneralizedScore {
    fn empty() -> Self {
        Self {
            inliers: Vec::new(),
            errors: Vec::new(),
            mean_error: f64::INFINITY,
            median_error: f64::INFINITY,
            max_error: f64::INFINITY,
        }
    }

    fn is_better_than(&self, other: &Self) -> bool {
        self.inliers.len() > other.inliers.len()
            || (self.inliers.len() == other.inliers.len() && self.mean_error < other.mean_error)
    }
}

fn score_pose(
    rig: &GeneralizedCameraRig,
    pose: &Pose,
    correspondences: &[GeneralizedCorrespondence2D3D],
    threshold: f64,
) -> GeneralizedScore {
    let mut indexed_errors = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            let projected = rig.project(pose, correspondence)?;
            let error = (projected - correspondence.point2d).norm();
            (error.is_finite() && error <= threshold).then_some((index, error))
        })
        .collect::<Vec<_>>();
    if indexed_errors.is_empty() {
        return GeneralizedScore::empty();
    }
    let inliers = indexed_errors.iter().map(|(index, _)| *index).collect();
    let errors = indexed_errors
        .iter()
        .map(|(_, error)| *error)
        .collect::<Vec<_>>();
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    indexed_errors.sort_by(|left, right| left.1.total_cmp(&right.1));
    let median_error = indexed_errors[indexed_errors.len() / 2].1;
    let max_error = indexed_errors.last().map(|entry| entry.1).unwrap_or(0.0);
    GeneralizedScore {
        inliers,
        errors,
        mean_error,
        median_error,
        max_error,
    }
}

fn mean_squared_error(
    rig: &GeneralizedCameraRig,
    pose: &Pose,
    correspondences: &[GeneralizedCorrespondence2D3D],
) -> Option<f64> {
    let residual = residuals(rig, pose, correspondences)?;
    Some(residual.norm_squared() / correspondences.len() as f64)
}

fn residuals(
    rig: &GeneralizedCameraRig,
    pose: &Pose,
    correspondences: &[GeneralizedCorrespondence2D3D],
) -> Option<DVector<f64>> {
    let mut residual = DVector::zeros(correspondences.len() * 2);
    for (index, correspondence) in correspondences.iter().enumerate() {
        let projected = rig.project(pose, correspondence)?;
        residual[index * 2] = projected.x - correspondence.point2d.x;
        residual[index * 2 + 1] = projected.y - correspondence.point2d.y;
    }
    Some(residual)
}

fn perturb_pose(pose: &Pose, delta: &DVector<f64>) -> Pose {
    let rotation_delta =
        UnitQuaternion::from_scaled_axis(Vector3::new(delta[0], delta[1], delta[2]));
    let translation_delta = Vector3::new(delta[3], delta[4], delta[5]);
    Pose::from_world_to_camera(
        rotation_delta * pose.world_to_camera.rotation,
        pose.world_to_camera.translation + translation_delta,
    )
}

fn skew(vector: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rig() -> GeneralizedCameraRig {
        let camera_left = Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5);
        let camera_right = Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5);
        GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: camera_left,
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: camera_right,
                // Right optical centre is +0.20 m on the rig x axis.
                sensor_from_rig: SE3::new(
                    UnitQuaternion::from_euler_angles(0.002, -0.004, 0.003),
                    Vector3::new(-0.20, 0.001, -0.002),
                ),
            },
        ])
        .unwrap()
    }

    fn truth_pose() -> Pose {
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.06, -0.12, 0.04),
            Vector3::new(0.35, -0.18, 0.42),
        )
    }

    fn synthetic_correspondences(
        rig: &GeneralizedCameraRig,
        pose: &Pose,
    ) -> Vec<GeneralizedCorrespondence2D3D> {
        let mut correspondences = Vec::new();
        for index in 0..24 {
            let point = Point3::new(
                -1.6 + 0.31 * (index % 7) as f64,
                -0.9 + 0.28 * (index % 5) as f64,
                4.0 + 0.22 * index as f64 + 0.13 * (index % 3) as f64,
            );
            let sensor_index = index % 2;
            let sensor = rig.sensor(sensor_index).unwrap();
            let point_rig = pose.transform_world_point(&point);
            let point_sensor = sensor.sensor_from_rig.transform_point(&point_rig);
            let point2d = sensor.camera.project(&point_sensor).unwrap();
            correspondences.push(GeneralizedCorrespondence2D3D {
                sensor_index,
                point2d,
                point3d: point,
                confidence: None,
            });
        }
        correspondences
    }

    fn pose_errors(estimate: &Pose, truth: &Pose) -> (f64, f64) {
        let rotation =
            (estimate.world_to_camera.rotation * truth.world_to_camera.rotation.inverse()).angle();
        let translation =
            (estimate.world_to_camera.translation - truth.world_to_camera.translation).norm();
        (rotation, translation)
    }

    #[test]
    fn generalized_dlt_recovers_shared_rig_pose() {
        let rig = test_rig();
        let truth = truth_pose();
        let correspondences = synthetic_correspondences(&rig, &truth);
        let estimate = GeneralizedDltPoseEstimator::default()
            .estimate_pose(&rig, &correspondences)
            .unwrap();
        let (rotation_error, translation_error) = pose_errors(&estimate, &truth);
        assert!(rotation_error < 1.0e-8, "rotation error {rotation_error}");
        assert!(
            translation_error < 1.0e-8,
            "translation error {translation_error}"
        );
    }

    #[test]
    fn rejects_single_origin_instead_of_falling_back_to_central_pnp() {
        let rig = test_rig();
        let truth = truth_pose();
        let mut correspondences = synthetic_correspondences(&rig, &truth);
        correspondences.retain(|correspondence| correspondence.sensor_index == 0);
        assert!(GeneralizedDltPoseEstimator::default()
            .estimate_pose(&rig, &correspondences)
            .is_none());
    }

    #[test]
    fn generalized_ransac_rejects_pixel_outliers_and_refines_one_pose() {
        let rig = test_rig();
        let truth = truth_pose();
        let mut correspondences = synthetic_correspondences(&rig, &truth);
        for index in [1usize, 6, 11, 16, 21] {
            correspondences[index].point2d.x += 90.0 + index as f64;
            correspondences[index].point2d.y -= 55.0;
        }
        let report = GeneralizedPnPRansac {
            iterations: 512,
            reprojection_threshold: 1.0,
            ..GeneralizedPnPRansac::default()
        }
        .estimate(&rig, &correspondences)
        .unwrap();
        let (rotation_error, translation_error) = pose_errors(&report.pose, &truth);
        assert_eq!(report.inliers.len(), 19);
        assert!(rotation_error < 1.0e-7, "rotation error {rotation_error}");
        assert!(
            translation_error < 1.0e-7,
            "translation error {translation_error}"
        );
        assert!(report.mean_reprojection_error < 1.0e-5);
    }

    #[test]
    fn nonlinear_refinement_reduces_joint_sensor_error() {
        let rig = test_rig();
        let truth = truth_pose();
        let correspondences = synthetic_correspondences(&rig, &truth);
        let initial = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.008, 0.006))
                * truth.world_to_camera.rotation,
            truth.world_to_camera.translation + Vector3::new(0.04, -0.03, 0.02),
        );
        let before = mean_squared_error(&rig, &initial, &correspondences).unwrap();
        let refined = GeneralizedGaussNewtonPoseRefiner::default()
            .refine_pose(&rig, &initial, &correspondences)
            .unwrap();
        let after = mean_squared_error(&rig, &refined, &correspondences).unwrap();
        assert!(after < before * 1.0e-6, "before={before} after={after}");
        let (rotation_error, translation_error) = pose_errors(&refined, &truth);
        assert!(rotation_error < 1.0e-6);
        assert!(translation_error < 1.0e-6);
    }
}
