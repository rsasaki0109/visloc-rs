use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::geometry::{reproject, Pose};
use visloc_core::types::{
    Camera, PoseEstimationFailureDiagnostics, PoseEstimationFailureReason, PoseEstimatorDiagnostics,
};

use crate::pnp::{Correspondence2D3D, DltPnP, GaussNewtonPoseRefiner, PoseEstimator, PoseRefiner};

#[derive(Debug, Clone, PartialEq)]
pub struct RansacReport {
    pub pose: Pose,
    pub inliers: Vec<usize>,
    pub inlier_reprojection_errors: Vec<f64>,
    pub mean_reprojection_error: f64,
    pub median_reprojection_error: f64,
    pub max_reprojection_error: f64,
    pub diagnostics: PoseEstimatorDiagnostics,
}

#[derive(Debug, Clone, PartialEq)]
struct ReprojectionScore {
    inliers: Vec<usize>,
    inlier_errors: Vec<f64>,
    mean_error: f64,
    median_error: f64,
    max_error: f64,
}

pub trait RobustPoseEstimator {
    fn estimate(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<RansacReport>;

    fn failure_diagnostics(
        &self,
        _correspondences: &[Correspondence2D3D],
        _camera: &Camera,
    ) -> Option<PoseEstimationFailureDiagnostics> {
        None
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PnPRansac<P = DltPnP, R = GaussNewtonPoseRefiner> {
    pub pose_estimator: P,
    pub pose_refiner: Option<R>,
    pub iterations: usize,
    pub reprojection_threshold: f64,
    pub seed: u64,
}

impl Default for PnPRansac {
    fn default() -> Self {
        Self {
            pose_estimator: DltPnP::default(),
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: 128,
            reprojection_threshold: 4.0,
            seed: 7,
        }
    }
}

impl<P, R> RobustPoseEstimator for PnPRansac<P, R>
where
    P: PoseEstimator,
    R: PoseRefiner,
{
    fn failure_diagnostics(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<PoseEstimationFailureDiagnostics> {
        let sample_size = self.pose_estimator.minimum_correspondences();
        if correspondences.len() < sample_size {
            return Some(PoseEstimationFailureDiagnostics {
                reason: PoseEstimationFailureReason::InsufficientCorrespondences,
                correspondence_count: correspondences.len(),
                minimum_correspondence_count: Some(sample_size),
                ransac_iterations: Some(self.iterations),
                ransac_reprojection_threshold: Some(self.reprojection_threshold),
                best_inlier_count: None,
            });
        }

        let search = self.search_best_pose(correspondences, camera, sample_size);
        Some(PoseEstimationFailureDiagnostics {
            reason: if search.best_pose.is_some() {
                PoseEstimationFailureReason::EstimatorRejected
            } else {
                PoseEstimationFailureReason::NoValidPose
            },
            correspondence_count: correspondences.len(),
            minimum_correspondence_count: Some(sample_size),
            ransac_iterations: Some(self.iterations),
            ransac_reprojection_threshold: Some(self.reprojection_threshold),
            best_inlier_count: Some(search.best_inliers.len()),
        })
    }

    fn estimate(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
    ) -> Option<RansacReport> {
        let sample_size = self.pose_estimator.minimum_correspondences();
        if correspondences.len() < sample_size {
            return None;
        }

        let search = self.search_best_pose(correspondences, camera, sample_size);

        if search.best_inliers.len() >= sample_size {
            let refined = search
                .best_inliers
                .iter()
                .map(|index| correspondences[*index].clone())
                .collect::<Vec<_>>();
            if let Some(mut refined_pose) = self.pose_estimator.estimate_pose(&refined, camera) {
                let pre_refinement_score = score_pose(
                    camera,
                    &refined_pose,
                    correspondences,
                    self.reprojection_threshold,
                );
                let mut refinement_applied = false;
                if let Some(pose_refiner) = &self.pose_refiner {
                    if let Some(nonlinear_pose) =
                        pose_refiner.refine_pose(&refined_pose, &refined, camera)
                    {
                        refined_pose = nonlinear_pose;
                        refinement_applied = true;
                    }
                }
                let score = score_pose(
                    camera,
                    &refined_pose,
                    correspondences,
                    self.reprojection_threshold,
                );
                return Some(RansacReport {
                    pose: refined_pose,
                    inliers: score.inliers,
                    inlier_reprojection_errors: score.inlier_errors,
                    mean_reprojection_error: score.mean_error,
                    median_reprojection_error: score.median_error,
                    max_reprojection_error: score.max_error,
                    diagnostics: PoseEstimatorDiagnostics {
                        refinement_applied,
                        pre_refinement_mean_reprojection_error: Some(
                            pre_refinement_score.mean_error,
                        ),
                        post_refinement_mean_reprojection_error: Some(score.mean_error),
                        refinement_error_delta: Some(
                            pre_refinement_score.mean_error - score.mean_error,
                        ),
                    },
                });
            }
        }

        search.best_pose.map(|pose| {
            let score = score_pose(camera, &pose, correspondences, self.reprojection_threshold);
            RansacReport {
                pose,
                inliers: score.inliers,
                inlier_reprojection_errors: score.inlier_errors,
                mean_reprojection_error: score.mean_error,
                median_reprojection_error: score.median_error,
                max_reprojection_error: score.max_error,
                diagnostics: PoseEstimatorDiagnostics {
                    refinement_applied: false,
                    pre_refinement_mean_reprojection_error: None,
                    post_refinement_mean_reprojection_error: Some(score.mean_error),
                    refinement_error_delta: None,
                },
            }
        })
    }
}

impl<P, R> PnPRansac<P, R>
where
    P: PoseEstimator,
{
    fn search_best_pose(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        sample_size: usize,
    ) -> RansacSearchResult {
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let mut indices = (0..correspondences.len()).collect::<Vec<_>>();
        let mut best_pose = None;
        let mut best_inliers = Vec::new();
        let mut best_error = f64::INFINITY;

        for _ in 0..self.iterations {
            indices.shuffle(&mut rng);
            let sample = indices
                .iter()
                .take(sample_size)
                .map(|index| correspondences[*index].clone())
                .collect::<Vec<_>>();

            let Some(pose) = self.pose_estimator.estimate_pose(&sample, camera) else {
                continue;
            };

            let score = score_pose(camera, &pose, correspondences, self.reprojection_threshold);
            if score.inliers.len() > best_inliers.len()
                || (score.inliers.len() == best_inliers.len() && score.mean_error < best_error)
            {
                best_pose = Some(pose);
                best_inliers = score.inliers;
                best_error = score.mean_error;
            }
        }

        RansacSearchResult {
            best_pose,
            best_inliers,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RansacSearchResult {
    best_pose: Option<Pose>,
    best_inliers: Vec<usize>,
}

fn score_pose(
    camera: &Camera,
    pose: &Pose,
    correspondences: &[Correspondence2D3D],
    threshold: f64,
) -> ReprojectionScore {
    let mut inliers = Vec::new();
    let mut inlier_errors = Vec::new();

    for (index, correspondence) in correspondences.iter().enumerate() {
        let Some(projected) = reproject(camera, pose, &correspondence.point3d) else {
            continue;
        };
        let error = (projected - correspondence.point2d).norm();
        if error <= threshold {
            inliers.push(index);
            inlier_errors.push(error);
        }
    }

    let (mean_error, median_error, max_error) = reprojection_stats(&inlier_errors);
    ReprojectionScore {
        inliers,
        inlier_errors,
        mean_error,
        median_error,
        max_error,
    }
}

fn reprojection_stats(errors: &[f64]) -> (f64, f64, f64) {
    if errors.is_empty() {
        return (f64::INFINITY, f64::INFINITY, f64::INFINITY);
    }

    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let mut sorted = errors.to_vec();
    sorted.sort_by(f64::total_cmp);
    let median = if sorted.len() % 2 == 0 {
        let upper = sorted.len() / 2;
        (sorted[upper - 1] + sorted[upper]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    let max = *sorted.last().expect("errors is non-empty");
    (mean, median, max)
}

#[cfg(test)]
mod tests {
    use super::{PnPRansac, RobustPoseEstimator};
    use crate::pnp::{Correspondence2D3D, PoseEstimator};
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::Pose;
    use visloc_core::types::Camera;

    #[derive(Debug, Clone, Copy)]
    struct IdentityPnP {
        min_correspondences: usize,
    }

    impl PoseEstimator for IdentityPnP {
        fn estimate_pose(
            &self,
            _correspondences: &[Correspondence2D3D],
            _camera: &Camera,
        ) -> Option<Pose> {
            Some(Pose::identity())
        }

        fn minimum_correspondences(&self) -> usize {
            self.min_correspondences
        }
    }

    #[test]
    fn ransac_accepts_custom_minimal_pose_estimator() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let points = vec![
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
        ];
        let correspondences = points
            .into_iter()
            .map(|point3d| Correspondence2D3D {
                point2d: camera
                    .project(&pose.transform_world_point(&point3d))
                    .unwrap(),
                point3d,
            })
            .collect::<Vec<_>>();

        let report = PnPRansac {
            pose_estimator: IdentityPnP {
                min_correspondences: 3,
            },
            pose_refiner: None::<crate::pnp::GaussNewtonPoseRefiner>,
            iterations: 8,
            reprojection_threshold: 1.0e-6,
            seed: 3,
        }
        .estimate(&correspondences, &camera)
        .unwrap();

        assert_eq!(report.inliers.len(), 3);
        assert_eq!(report.inlier_reprojection_errors.len(), 3);
        assert!(report.max_reprojection_error < 1.0e-9);
        assert!(!report.diagnostics.refinement_applied);
        assert_eq!(report.pose, Pose::identity());
    }
}
