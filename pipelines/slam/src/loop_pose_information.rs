//! Covariance-aware information estimation for PnP pose-graph edges.

use crate::central_difference_projection_jacobian;
use crate::loop_closure::LoopClosureConstraint;
use nalgebra::{Matrix3, Matrix6, Point3, Vector6};
use std::fmt;
use visloc_core::geometry::SE3;
use visloc_core::types::{Camera, Frame, VisualMap};

/// Controls the covariance-aware PnP loop-edge information estimator.
///
/// `max_information_eigenvalue` is an explicit back-end strength cap: the
/// reprojection Hessian determines anisotropy and observability, while this
/// scalar prevents its pixel-domain units from silently overpowering the
/// pose graph's unit-weight odometry edges. It is reported/configured
/// separately rather than folded into correspondence count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopPoseInformationConfig {
    pub pixel_sigma_px: f64,
    pub max_reprojection_error_px: f64,
    pub min_correspondences: usize,
    pub min_landmark_observations: usize,
    pub max_landmark_condition_number: f64,
    pub max_pose_condition_number: f64,
    pub max_information_eigenvalue: f64,
    /// Additional scale applied only when the matrix is inserted as a loop
    /// edge. Sequential PnP edges deliberately ignore it, so this changes the
    /// loop/odometry balance without destroying measured anisotropy.
    pub loop_edge_scale: f64,
    pub finite_difference_step: f64,
}

/// Numerical evidence emitted for each PnP loop whose covariance-aware pose
/// information passed all gates. Eigenvalues are recorded before the explicit
/// graph-strength cap; `applied_spectral_scale` is `1.0` when no cap was needed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopPoseInformationDiagnostic {
    pub pnp_inlier_count: usize,
    pub used_correspondence_count: usize,
    /// Used correspondences whose 3-D uncertainty came from calibrated stereo
    /// metadata rather than the multi-view left-image Hessian fallback.
    pub stereo_covariance_correspondence_count: usize,
    pub raw_min_eigenvalue: f64,
    pub raw_max_eigenvalue: f64,
    pub raw_condition_number: f64,
    pub applied_spectral_scale: f64,
}

/// Why covariance-aware pose information could not be formed.
///
/// Keeping these cases distinct is important for real-sequence evaluation:
/// adding more matches cannot repair an invalid configuration or a
/// full-rank-but-ill-conditioned camera/landmark arrangement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopPoseInformationFailure {
    InvalidConfiguration,
    MissingFromKeyframe,
    MissingFromPose,
    InsufficientUsableCorrespondences,
    PoseRankDeficient,
    PoseIllConditioned,
    UnsupportedSolver,
}

/// Additive failure histogram suitable for per-frame stats and full-run
/// EuRoC summaries.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LoopPoseInformationFailureCounts {
    pub invalid_configuration: usize,
    pub missing_from_keyframe: usize,
    pub missing_from_pose: usize,
    pub insufficient_usable_correspondences: usize,
    pub pose_rank_deficient: usize,
    pub pose_ill_conditioned: usize,
    pub unsupported_solver: usize,
}

impl LoopPoseInformationFailureCounts {
    pub fn record(&mut self, failure: LoopPoseInformationFailure) {
        match failure {
            LoopPoseInformationFailure::InvalidConfiguration => self.invalid_configuration += 1,
            LoopPoseInformationFailure::MissingFromKeyframe => self.missing_from_keyframe += 1,
            LoopPoseInformationFailure::MissingFromPose => self.missing_from_pose += 1,
            LoopPoseInformationFailure::InsufficientUsableCorrespondences => {
                self.insufficient_usable_correspondences += 1;
            }
            LoopPoseInformationFailure::PoseRankDeficient => self.pose_rank_deficient += 1,
            LoopPoseInformationFailure::PoseIllConditioned => self.pose_ill_conditioned += 1,
            LoopPoseInformationFailure::UnsupportedSolver => self.unsupported_solver += 1,
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.invalid_configuration += other.invalid_configuration;
        self.missing_from_keyframe += other.missing_from_keyframe;
        self.missing_from_pose += other.missing_from_pose;
        self.insufficient_usable_correspondences += other.insufficient_usable_correspondences;
        self.pose_rank_deficient += other.pose_rank_deficient;
        self.pose_ill_conditioned += other.pose_ill_conditioned;
        self.unsupported_solver += other.unsupported_solver;
    }
}

impl fmt::Display for LoopPoseInformationFailureCounts {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid_config:{},missing_keyframe:{},missing_pose:{},insufficient_correspondences:{},rank_deficient:{},ill_conditioned:{},unsupported_solver:{}",
            self.invalid_configuration,
            self.missing_from_keyframe,
            self.missing_from_pose,
            self.insufficient_usable_correspondences,
            self.pose_rank_deficient,
            self.pose_ill_conditioned,
            self.unsupported_solver,
        )
    }
}

impl Default for LoopPoseInformationConfig {
    fn default() -> Self {
        Self {
            pixel_sigma_px: 1.0,
            max_reprojection_error_px: 4.0,
            min_correspondences: 8,
            min_landmark_observations: 2,
            max_landmark_condition_number: 1.0e8,
            max_pose_condition_number: 1.0e6,
            max_information_eigenvalue: 1.0,
            loop_edge_scale: 1.0,
            finite_difference_step: 1.0e-5,
        }
    }
}

/// Estimate the verified PnP loop measurement's 6×6 information in the pose
/// graph's translation-first right-tangent convention.
///
/// For every final PnP inlier, the landmark covariance is recovered from the
/// multi-view reprojection Hessian of its existing map observations. That
/// covariance is propagated through the loop reprojection and added to the
/// query pixel covariance before accumulating `J_poseᵀ S⁻¹ J_pose`. Both the
/// landmark and pose Hessians must be positive full-rank and pass explicit
/// condition-number gates; no ridge is added. The final spectral cap keeps the
/// pixel-domain Hessian commensurate with the graph's unit-weight odometry
/// edges while preserving its measured anisotropy.
pub(crate) fn estimate_loop_pose_information(
    map: &VisualMap,
    query_frame: &Frame,
    camera: &Camera,
    constraint: &LoopClosureConstraint,
    pnp_inliers: &[(usize, u64)],
    config: LoopPoseInformationConfig,
) -> Result<(Matrix6<f64>, LoopPoseInformationDiagnostic), LoopPoseInformationFailure> {
    let valid_config = config.pixel_sigma_px.is_finite()
        && config.pixel_sigma_px > 0.0
        && config.max_reprojection_error_px.is_finite()
        && config.max_reprojection_error_px > 0.0
        && config.min_correspondences >= 3
        && config.min_landmark_observations >= 2
        && config.max_landmark_condition_number.is_finite()
        && config.max_landmark_condition_number >= 1.0
        && config.max_pose_condition_number.is_finite()
        && config.max_pose_condition_number >= 1.0
        && config.max_information_eigenvalue.is_finite()
        && config.max_information_eigenvalue > 0.0
        && config.loop_edge_scale.is_finite()
        && config.loop_edge_scale > 0.0
        && config.finite_difference_step.is_finite()
        && config.finite_difference_step > 0.0;
    if !valid_config {
        return Err(LoopPoseInformationFailure::InvalidConfiguration);
    }
    let from_pose = map
        .keyframes
        .get(&constraint.from_keyframe_id)
        .ok_or(LoopPoseInformationFailure::MissingFromKeyframe)?
        .frame
        .pose
        .as_ref()
        .ok_or(LoopPoseInformationFailure::MissingFromPose)?;
    let sigma2 = config.pixel_sigma_px * config.pixel_sigma_px;
    let eps = config.finite_difference_step;
    let mut omega = Matrix6::<f64>::zeros();
    let mut used = 0usize;
    let mut used_stereo_covariance = 0usize;

    for &(query_index, landmark_id) in pnp_inliers {
        let Some(&query_xy) = query_frame.keypoints.get(query_index) else {
            continue;
        };
        let Some(landmark) = map.landmarks.get(&landmark_id) else {
            continue;
        };
        let has_stereo_covariance = map.landmark_position_covariances.contains_key(&landmark_id);

        // Prefer the calibrated stereo seed covariance: it retains both
        // left/right measurements and their anisotropic depth uncertainty.
        // Landmarks without such metadata fall back to the historical
        // multi-view left-image Hessian. The query observation is excluded in
        // that fallback to avoid counting its residual twice.
        let covariance_world =
            if let Some(covariance) = map.landmark_position_covariances.get(&landmark_id) {
                let eigenvalues = covariance.symmetric_eigen().eigenvalues;
                let minimum = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
                let maximum = eigenvalues.iter().copied().fold(0.0, f64::max);
                if !minimum.is_finite()
                    || !maximum.is_finite()
                    || minimum <= 0.0
                    || maximum / minimum > config.max_landmark_condition_number
                {
                    continue;
                }
                *covariance
            } else {
                let mut landmark_hessian = Matrix3::<f64>::zeros();
                let mut landmark_observation_count = 0usize;
                for observation in &landmark.observations {
                    if observation.frame_id == query_frame.id {
                        continue;
                    }
                    let Some(observing_pose) = map
                        .keyframes
                        .get(&observation.frame_id)
                        .and_then(|keyframe| keyframe.frame.pose.as_ref())
                    else {
                        continue;
                    };
                    let base_camera = observing_pose.transform_world_point(&landmark.position);
                    let Some(predicted) = camera.project(&base_camera) else {
                        continue;
                    };
                    if (predicted - observation.xy).norm() > config.max_reprojection_error_px {
                        continue;
                    }
                    let center = landmark.position.coords;
                    if let Some(jacobian) =
                        central_difference_projection_jacobian(&center, eps, |position| {
                            let world_point = Point3::from(*position);
                            camera.project(&observing_pose.transform_world_point(&world_point))
                        })
                    {
                        landmark_hessian += jacobian.transpose() * jacobian / sigma2;
                        landmark_observation_count += 1;
                    }
                }
                if landmark_observation_count < config.min_landmark_observations {
                    continue;
                }
                let landmark_eigenvalues = landmark_hessian.symmetric_eigen().eigenvalues;
                let landmark_min = landmark_eigenvalues
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                let landmark_max = landmark_eigenvalues.iter().copied().fold(0.0, f64::max);
                if !landmark_min.is_finite()
                    || !landmark_max.is_finite()
                    || landmark_min <= 0.0
                    || landmark_max / landmark_min > config.max_landmark_condition_number
                {
                    continue;
                }
                let Some(covariance) = landmark_hessian.try_inverse() else {
                    continue;
                };
                covariance
            };

        let older_point = from_pose.transform_world_point(&landmark.position);
        let current_point = constraint.relative_pose.transform_point(&older_point);
        let Some(predicted_query) = camera.project(&current_point) else {
            continue;
        };
        if (predicted_query - query_xy).norm() > config.max_reprojection_error_px {
            continue;
        }

        let Some(pose_jacobian) =
            central_difference_projection_jacobian(&Vector6::zeros(), eps, |delta| {
                let pose = constraint.relative_pose.compose(&SE3::exp(delta));
                camera.project(&pose.transform_point(&older_point))
            })
        else {
            continue;
        };

        // Pixel sensitivity to the uncertain point in the older camera frame.
        let older_center = older_point.coords;
        let Some(point_jacobian) =
            central_difference_projection_jacobian(&older_center, eps, |point| {
                camera.project(
                    &constraint
                        .relative_pose
                        .transform_point(&Point3::from(*point)),
                )
            })
        else {
            continue;
        };
        let rotation_world_to_older = from_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .matrix()
            .clone_owned();
        let covariance_older =
            rotation_world_to_older * covariance_world * rotation_world_to_older.transpose();
        let residual_covariance = nalgebra::SMatrix::<f64, 2, 2>::identity() * sigma2
            + point_jacobian * covariance_older * point_jacobian.transpose();
        let Some(residual_precision) = residual_covariance.try_inverse() else {
            continue;
        };
        omega += pose_jacobian.transpose() * residual_precision * pose_jacobian;
        used += 1;
        if has_stereo_covariance {
            used_stereo_covariance += 1;
        }
    }

    if used < config.min_correspondences {
        return Err(LoopPoseInformationFailure::InsufficientUsableCorrespondences);
    }
    omega = (omega + omega.transpose()) * 0.5;
    let pose_eigenvalues = omega.symmetric_eigen().eigenvalues;
    let pose_min = pose_eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let pose_max = pose_eigenvalues.iter().copied().fold(0.0, f64::max);
    if !pose_min.is_finite() || !pose_max.is_finite() || pose_min <= 0.0 {
        return Err(LoopPoseInformationFailure::PoseRankDeficient);
    }
    if pose_max / pose_min > config.max_pose_condition_number {
        return Err(LoopPoseInformationFailure::PoseIllConditioned);
    }
    let applied_spectral_scale = if pose_max > config.max_information_eigenvalue {
        config.max_information_eigenvalue / pose_max
    } else {
        1.0
    };
    omega *= applied_spectral_scale;
    Ok((
        omega,
        LoopPoseInformationDiagnostic {
            pnp_inlier_count: pnp_inliers.len(),
            used_correspondence_count: used,
            stereo_covariance_correspondence_count: used_stereo_covariance,
            raw_min_eigenvalue: pose_min,
            raw_max_eigenvalue: pose_max,
            raw_condition_number: pose_max / pose_min,
            applied_spectral_scale,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, Vector3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::{Keyframe, Landmark, Observation};

    fn covariance_information_scene(
        observations_per_landmark: usize,
    ) -> (
        VisualMap,
        Frame,
        Camera,
        LoopClosureConstraint,
        Vec<(usize, u64)>,
    ) {
        let camera = Camera::pinhole(1, 640, 480, 460.0, 455.0, 320.0, 240.0);
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        let observing_poses = [
            Pose::identity(),
            Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::identity(),
                Vector3::new(-0.35, 0.02, 0.0),
            ),
            Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.03, 0.0),
                Vector3::new(0.25, -0.08, 0.02),
            ),
        ];
        for (index, pose) in observing_poses.iter().enumerate() {
            let mut frame = Frame::new(index as u64 + 1, camera.id);
            frame.pose = Some(pose.clone());
            map.keyframes.insert(
                frame.id,
                Keyframe {
                    frame,
                    observations: Vec::new(),
                },
            );
        }
        let points = [
            Point3::new(-1.2, -0.7, 4.0),
            Point3::new(-0.5, 0.9, 4.5),
            Point3::new(0.4, -0.8, 5.0),
            Point3::new(1.1, 0.6, 5.5),
            Point3::new(-1.0, 0.2, 6.0),
            Point3::new(0.2, 1.1, 6.5),
            Point3::new(1.3, -0.1, 7.0),
            Point3::new(-0.3, -1.0, 7.5),
            Point3::new(0.8, 0.8, 8.0),
            Point3::new(-1.4, 1.0, 5.8),
            Point3::new(1.5, -0.9, 6.8),
            Point3::new(0.0, 0.0, 4.8),
        ];
        let mut query = Frame::new(100, camera.id);
        let mut pairs = Vec::new();
        for (point_index, point) in points.into_iter().enumerate() {
            let landmark_id = point_index as u64 + 10;
            let mut landmark = Landmark::new(landmark_id, point);
            for (view_index, pose) in observing_poses
                .iter()
                .take(observations_per_landmark.min(observing_poses.len()))
                .enumerate()
            {
                let xy = camera.project(&pose.transform_world_point(&point)).unwrap();
                landmark.observations.push(Observation {
                    frame_id: view_index as u64 + 1,
                    landmark_id,
                    keypoint_index: point_index,
                    xy,
                });
            }
            map.landmarks.insert(landmark_id, landmark);
            query.keypoints.push(camera.project(&point).unwrap());
            pairs.push((point_index, landmark_id));
        }
        let constraint = LoopClosureConstraint {
            from_keyframe_id: 1,
            to_keyframe_id: query.id,
            relative_pose: SE3::identity(),
            inlier_count: points.len(),
            inlier_ratio: 1.0,
            mean_sampson_error: 0.0,
            score: points.len() as f64,
        };
        (map, query, camera, constraint, pairs)
    }

    #[test]
    fn covariance_loop_information_is_full_rank_anisotropic_and_spectrally_capped() {
        let (map, query, camera, constraint, pairs) = covariance_information_scene(3);
        let (information, diagnostic) = estimate_loop_pose_information(
            &map,
            &query,
            &camera,
            &constraint,
            &pairs,
            LoopPoseInformationConfig::default(),
        )
        .expect("well-spread multi-view geometry should yield information");
        let eigenvalues = information.symmetric_eigen().eigenvalues;
        let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let max = eigenvalues.iter().copied().fold(0.0, f64::max);
        assert!(min > 0.0, "eigenvalues={eigenvalues:?}");
        assert!(max <= 1.0 + 1.0e-9, "spectral cap violated: {max}");
        assert!(max / min > 2.0, "expected anisotropy: {eigenvalues:?}");
        assert_eq!(diagnostic.pnp_inlier_count, pairs.len());
        assert_eq!(diagnostic.used_correspondence_count, pairs.len());
        assert_eq!(diagnostic.stereo_covariance_correspondence_count, 0);
        assert!(diagnostic.raw_condition_number > 2.0);
        assert!(diagnostic.applied_spectral_scale < 1.0);
    }

    #[test]
    fn covariance_loop_information_rejects_single_view_landmark_rays() {
        let (map, query, camera, constraint, pairs) = covariance_information_scene(1);
        assert_eq!(
            estimate_loop_pose_information(
                &map,
                &query,
                &camera,
                &constraint,
                &pairs,
                LoopPoseInformationConfig::default(),
            )
            .unwrap_err(),
            LoopPoseInformationFailure::InsufficientUsableCorrespondences
        );
    }

    #[test]
    fn calibrated_stereo_covariance_makes_single_view_landmarks_usable() {
        let (mut map, query, camera, constraint, pairs) = covariance_information_scene(1);
        let covariance = Matrix3::from_diagonal(&Vector3::new(0.0025, 0.0025, 0.04));
        for &(_, landmark_id) in &pairs {
            map.landmark_position_covariances
                .insert(landmark_id, covariance);
        }
        let (_, diagnostic) = estimate_loop_pose_information(
            &map,
            &query,
            &camera,
            &constraint,
            &pairs,
            LoopPoseInformationConfig::default(),
        )
        .expect("stereo seed covariance observes the depth direction");
        assert_eq!(diagnostic.used_correspondence_count, pairs.len());
        assert_eq!(
            diagnostic.stereo_covariance_correspondence_count,
            pairs.len()
        );
    }

    #[test]
    fn reports_invalid_config_and_ill_conditioned_pose_separately() {
        let (map, query, camera, constraint, pairs) = covariance_information_scene(3);
        let invalid = LoopPoseInformationConfig {
            finite_difference_step: 0.0,
            ..LoopPoseInformationConfig::default()
        };
        assert_eq!(
            estimate_loop_pose_information(&map, &query, &camera, &constraint, &pairs, invalid)
                .unwrap_err(),
            LoopPoseInformationFailure::InvalidConfiguration
        );

        let ill_conditioned = LoopPoseInformationConfig {
            max_pose_condition_number: 1.0,
            ..LoopPoseInformationConfig::default()
        };
        assert_eq!(
            estimate_loop_pose_information(
                &map,
                &query,
                &camera,
                &constraint,
                &pairs,
                ill_conditioned,
            )
            .unwrap_err(),
            LoopPoseInformationFailure::PoseIllConditioned
        );
    }

    #[test]
    fn failure_counts_merge_and_render_stably() {
        let mut counts = LoopPoseInformationFailureCounts::default();
        counts.record(LoopPoseInformationFailure::InsufficientUsableCorrespondences);
        counts.record(LoopPoseInformationFailure::PoseIllConditioned);
        let mut other = LoopPoseInformationFailureCounts::default();
        other.record(LoopPoseInformationFailure::PoseIllConditioned);
        counts.merge(other);

        assert_eq!(counts.insufficient_usable_correspondences, 1);
        assert_eq!(counts.pose_ill_conditioned, 2);
        assert_eq!(
            counts.to_string(),
            "invalid_config:0,missing_keyframe:0,missing_pose:0,insufficient_correspondences:1,rank_deficient:0,ill_conditioned:2,unsupported_solver:0"
        );
    }
}
