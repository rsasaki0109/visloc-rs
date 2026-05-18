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

    /// Confidence-aware variant: when `weights[i]` is high, correspondence
    /// `i` is preferred during RANSAC sampling (PROSAC-style). Default
    /// implementation falls back to the unweighted `estimate` so existing
    /// implementors don't need to change.
    fn estimate_with_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        weights: &[f32],
    ) -> Option<RansacReport> {
        let _ = weights;
        self.estimate(correspondences, camera)
    }

    /// Prior-aware variant: when `pose_prior` is `Some`, the estimator
    /// MAY seed its search with the prior pose as an initial hypothesis
    /// (so a successful prior short-circuits RANSAC and a failed prior
    /// is still beaten by the standard random search). Default
    /// implementation ignores the prior and delegates to the weighted /
    /// unweighted path so existing implementors don't need to change.
    fn estimate_with_pose_prior_and_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        pose_prior: Option<&Pose>,
        weights: Option<&[f32]>,
    ) -> Option<RansacReport> {
        let _ = pose_prior;
        match weights {
            Some(w) => self.estimate_with_weights(correspondences, camera, w),
            None => self.estimate(correspondences, camera),
        }
    }

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
    /// Stop the RANSAC search after this many trials when
    /// `early_stop_inlier_ratio` is also satisfied. Set the ratio to `None`
    /// to force the full iteration budget.
    pub early_stop_min_iterations: usize,
    /// Optional consensus-ratio threshold for early termination. This keeps
    /// high-consensus PnP pairs from spending the full budget scoring
    /// near-identical hypotheses.
    pub early_stop_inlier_ratio: Option<f64>,
}

impl Default for PnPRansac {
    fn default() -> Self {
        Self {
            pose_estimator: DltPnP::default(),
            pose_refiner: Some(GaussNewtonPoseRefiner::default()),
            iterations: 128,
            reprojection_threshold: 4.0,
            seed: 7,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
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

        let search = self.search_best_pose(correspondences, camera, sample_size, None, None);
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
        self.estimate_with_optional_inputs(correspondences, camera, None, None)
    }

    /// PROSAC-style PnP RANSAC: sort correspondences by descending weight
    /// (e.g. matcher confidence) and expand the sampling subset linearly
    /// from `sample_size` to `n` over the iteration budget. High-confidence
    /// correspondences anchor early iterations. Falls back to the uniform
    /// shuffle when `weights` is the wrong length, all-zero, or
    /// non-finite.
    fn estimate_with_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        weights: &[f32],
    ) -> Option<RansacReport> {
        if weights.len() != correspondences.len() {
            return self.estimate(correspondences, camera);
        }
        self.estimate_with_optional_inputs(correspondences, camera, Some(weights), None)
    }

    fn estimate_with_pose_prior_and_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        pose_prior: Option<&Pose>,
        weights: Option<&[f32]>,
    ) -> Option<RansacReport> {
        let valid_weights = weights.and_then(|w| (w.len() == correspondences.len()).then_some(w));
        self.estimate_with_optional_inputs(correspondences, camera, valid_weights, pose_prior)
    }
}

impl<P, R> PnPRansac<P, R>
where
    P: PoseEstimator,
    R: PoseRefiner,
{
    fn estimate_with_optional_inputs(
        &self,
        correspondences: &[Correspondence2D3D],
        camera: &Camera,
        weights: Option<&[f32]>,
        pose_prior: Option<&Pose>,
    ) -> Option<RansacReport> {
        let sample_size = self.pose_estimator.minimum_correspondences();
        if correspondences.len() < sample_size {
            return None;
        }

        let search =
            self.search_best_pose(correspondences, camera, sample_size, weights, pose_prior);

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
                        let nonlinear_score = score_pose(
                            camera,
                            &nonlinear_pose,
                            correspondences,
                            self.reprojection_threshold,
                        );
                        if is_refinement_better(&nonlinear_score, &pre_refinement_score) {
                            refined_pose = nonlinear_pose;
                            refinement_applied = true;
                        }
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
        weights: Option<&[f32]>,
        pose_prior: Option<&Pose>,
    ) -> RansacSearchResult {
        let mut rng = SmallRng::seed_from_u64(self.seed);
        let n = correspondences.len();
        // PROSAC ordering: sort indices by descending weight when weights
        // are usable; otherwise fall back to natural order + uniform
        // shuffle (the original behaviour).
        let weighted =
            weights.filter(|w| w.len() == n && w.iter().any(|&v| v.is_finite() && v > 0.0));
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        if let Some(w) = weighted {
            sorted_indices
                .sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut best_pose = None;
        let mut best_inliers = Vec::new();
        let mut best_error = f64::INFINITY;

        // Warm-start with the pose prior, if supplied. Random samples
        // must beat its inlier count to win, so a well-aligned prior
        // short-circuits RANSAC on hard scenes while a misaligned prior
        // gracefully degrades to the standard random search.
        if let Some(prior) = pose_prior {
            let score = score_pose(camera, prior, correspondences, self.reprojection_threshold);
            if !score.inliers.is_empty() {
                best_pose = Some(prior.clone());
                best_inliers = score.inliers;
                best_error = score.mean_error;
            }
        }
        let total_iters = self.iterations.max(1);
        let expansion_iters = if weighted.is_some() && self.early_stop_inlier_ratio.is_some() {
            self.early_stop_min_iterations.max(1).min(total_iters)
        } else {
            total_iters
        };
        let expansion_denom = expansion_iters.saturating_sub(1).max(1);

        for iteration in 0..self.iterations {
            // PROSAC shrinking sample-set: m_k expands linearly from
            // `sample_size` to `n`. When an early-stop guard is active,
            // expand over that guard window instead of the whole budget so
            // a larger fail-safe iteration count does not delay good samples.
            // When weights are absent this collapses to `m_k = n` and the
            // shuffle samples uniformly across all correspondences.
            let m_k = if weighted.is_some() {
                let progress = (iteration.min(expansion_denom)) as f64 / expansion_denom as f64;
                let m = sample_size as f64 + (n - sample_size) as f64 * progress;
                (m.ceil() as usize).clamp(sample_size, n)
            } else {
                n
            };
            let mut subset: Vec<usize> = sorted_indices[..m_k].to_vec();
            subset.shuffle(&mut rng);
            let sample: Vec<Correspondence2D3D> = subset
                .iter()
                .take(sample_size)
                .map(|index| correspondences[*index].clone())
                .collect();

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

            if should_stop_pnp_search_early(
                iteration + 1,
                best_pose.is_some(),
                best_inliers.len(),
                n,
                self.early_stop_min_iterations,
                self.early_stop_inlier_ratio,
            ) {
                break;
            }
        }

        RansacSearchResult {
            best_pose,
            best_inliers,
        }
    }
}

fn should_stop_pnp_search_early(
    completed_iterations: usize,
    has_pose: bool,
    best_inliers: usize,
    total_correspondences: usize,
    min_iterations: usize,
    inlier_ratio: Option<f64>,
) -> bool {
    let Some(inlier_ratio) = inlier_ratio else {
        return false;
    };
    if !has_pose
        || completed_iterations < min_iterations.max(1)
        || total_correspondences == 0
        || !inlier_ratio.is_finite()
        || inlier_ratio <= 0.0
    {
        return false;
    }
    (best_inliers as f64) >= inlier_ratio.min(1.0) * total_correspondences as f64
}

fn is_refinement_better(candidate: &ReprojectionScore, baseline: &ReprojectionScore) -> bool {
    let candidate_inliers = candidate.inliers.len();
    let baseline_inliers = baseline.inliers.len();
    candidate_inliers > baseline_inliers
        || (candidate_inliers == baseline_inliers
            && candidate.mean_error <= baseline.mean_error + 1.0e-9)
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
    use super::{
        is_refinement_better, should_stop_pnp_search_early, PnPRansac, ReprojectionScore,
        RobustPoseEstimator,
    };
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

    fn reprojection_score(inlier_count: usize, mean_error: f64) -> ReprojectionScore {
        ReprojectionScore {
            inliers: (0..inlier_count).collect(),
            inlier_errors: vec![mean_error; inlier_count],
            mean_error,
            median_error: mean_error,
            max_error: mean_error,
        }
    }

    #[test]
    fn refinement_guard_rejects_worse_reprojection_consensus() {
        let baseline = reprojection_score(12, 0.8);
        let worse_error = reprojection_score(12, 0.9);
        let fewer_inliers = reprojection_score(11, 0.1);
        let more_inliers = reprojection_score(13, 1.2);
        let same_or_better = reprojection_score(12, 0.8);

        assert!(!is_refinement_better(&worse_error, &baseline));
        assert!(!is_refinement_better(&fewer_inliers, &baseline));
        assert!(is_refinement_better(&more_inliers, &baseline));
        assert!(is_refinement_better(&same_or_better, &baseline));
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
                confidence: None,
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
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
        }
        .estimate(&correspondences, &camera)
        .unwrap();

        assert_eq!(report.inliers.len(), 3);
        assert_eq!(report.inlier_reprojection_errors.len(), 3);
        assert!(report.max_reprojection_error < 1.0e-9);
        assert!(!report.diagnostics.refinement_applied);
        assert_eq!(report.pose, Pose::identity());
    }

    #[test]
    fn weighted_pnp_ransac_recovers_pose_with_outlier_heavy_input() {
        use crate::pnp::DltPnP;
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        // 12 inliers from a real geometry + 24 random outliers with
        // mismatched 2D pixels. Inliers carry confidence 0.9, outliers 0.05.
        let inlier_points = [
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 4.7),
            Point3::new(0.0, 0.0, 5.2),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.5, 0.25, 5.5),
            Point3::new(-0.7, 0.4, 4.5),
            Point3::new(0.6, 0.8, 5.5),
            Point3::new(-0.3, -0.6, 4.8),
            Point3::new(0.2, 0.2, 6.5),
            Point3::new(-0.2, 0.7, 5.7),
        ];
        let mut correspondences: Vec<Correspondence2D3D> = inlier_points
            .iter()
            .map(|p| Correspondence2D3D {
                point2d: camera.project(&pose.transform_world_point(p)).unwrap(),
                point3d: *p,
                confidence: None,
            })
            .collect();
        let n_inliers = correspondences.len();
        let mut weights: Vec<f32> = vec![0.9; n_inliers];

        // Outlier rays: random 3D points with off-image 2D coords.
        let outlier_seeds = [
            (0.4_f64, 0.3, 7.0, 100.0_f64, 50.0_f64),
            (-0.6, 0.5, 8.0, 580.0, 60.0),
            (0.1, -0.8, 5.0, 50.0, 420.0),
            (0.7, -0.3, 6.0, 600.0, 410.0),
            (-0.4, -0.4, 9.0, 320.0, 60.0),
            (0.5, 0.5, 5.5, 40.0, 240.0),
            (-0.7, 0.0, 7.5, 600.0, 240.0),
            (0.0, -0.6, 4.5, 320.0, 420.0),
            (0.2, 0.7, 6.5, 540.0, 100.0),
            (-0.5, -0.6, 5.0, 80.0, 160.0),
            (0.3, 0.4, 5.8, 340.0, 320.0),
            (-0.2, 0.6, 6.2, 200.0, 80.0),
            (0.6, -0.5, 4.5, 460.0, 360.0),
            (0.4, 0.2, 5.0, 260.0, 200.0),
            (-0.8, 0.3, 7.0, 540.0, 220.0),
            (0.1, 0.5, 5.5, 60.0, 300.0),
            (0.5, 0.6, 6.0, 480.0, 100.0),
            (-0.3, -0.7, 4.8, 380.0, 380.0),
            (0.7, 0.1, 5.3, 60.0, 80.0),
            (-0.6, -0.4, 6.5, 580.0, 280.0),
            (0.0, 0.4, 5.6, 200.0, 240.0),
            (-0.4, 0.0, 5.2, 420.0, 280.0),
            (0.2, -0.5, 5.0, 120.0, 360.0),
            (0.1, 0.3, 6.0, 540.0, 60.0),
        ];
        for (px, py, pz, cx, cy) in outlier_seeds {
            correspondences.push(Correspondence2D3D {
                point2d: nalgebra::Point2::new(cx, cy),
                point3d: Point3::new(px, py, pz),
                confidence: None,
            });
            weights.push(0.05);
        }
        assert_eq!(weights.len(), correspondences.len());

        let ransac = PnPRansac {
            pose_estimator: DltPnP::default(),
            pose_refiner: None::<crate::pnp::GaussNewtonPoseRefiner>,
            iterations: 64,
            reprojection_threshold: 2.0,
            seed: 11,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
        };

        let weighted = ransac
            .estimate_with_weights(&correspondences, &camera, &weights)
            .expect("weighted PnP RANSAC must recover the pose");

        let recovered_inliers: usize = weighted.inliers.iter().filter(|&&i| i < n_inliers).count();
        assert!(
            recovered_inliers >= n_inliers - 1,
            "weighted PnP RANSAC should recover the geometric inliers, got {}/{}",
            recovered_inliers,
            n_inliers
        );
    }

    #[test]
    fn weighted_pnp_ransac_falls_back_to_uniform_when_weight_length_mismatches() {
        use crate::pnp::DltPnP;
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::identity();
        let points = [
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 4.7),
            Point3::new(0.0, 0.0, 5.2),
            Point3::new(0.5, -0.25, 6.0),
        ];
        let correspondences: Vec<Correspondence2D3D> = points
            .iter()
            .map(|p| Correspondence2D3D {
                point2d: camera.project(&pose.transform_world_point(p)).unwrap(),
                point3d: *p,
                confidence: None,
            })
            .collect();
        let bad_weights = vec![1.0_f32; correspondences.len() + 3]; // wrong length
        let ransac = PnPRansac {
            pose_estimator: DltPnP::default(),
            pose_refiner: None::<crate::pnp::GaussNewtonPoseRefiner>,
            iterations: 16,
            reprojection_threshold: 1.0e-3,
            seed: 7,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
        };
        let unweighted = ransac.estimate(&correspondences, &camera).unwrap();
        let fallback = ransac
            .estimate_with_weights(&correspondences, &camera, &bad_weights)
            .unwrap();
        assert_eq!(unweighted.inliers, fallback.inliers);
    }

    #[test]
    fn pnp_early_stop_requires_pose_min_iterations_and_ratio() {
        assert!(!should_stop_pnp_search_early(100, true, 90, 100, 100, None));
        assert!(!should_stop_pnp_search_early(
            99,
            true,
            90,
            100,
            100,
            Some(0.85)
        ));
        assert!(!should_stop_pnp_search_early(
            100,
            false,
            90,
            100,
            100,
            Some(0.85)
        ));
        assert!(!should_stop_pnp_search_early(
            100,
            true,
            84,
            100,
            100,
            Some(0.85)
        ));
        assert!(should_stop_pnp_search_early(
            100,
            true,
            85,
            100,
            100,
            Some(0.85)
        ));
    }

    #[test]
    fn pnp_ransac_pose_prior_warm_start_recovers_when_random_search_fails() {
        use crate::pnp::DltPnP;
        // Scene: 7 correspondences perfectly consistent with the
        // identity pose, plus 5 outliers (3D points whose pixels are
        // scrambled). With iterations=1, the unweighted random search
        // has a ~0.76 % chance of stumbling on a 6-of-7 inlier sample
        // and at seed=1234 it does not — so the no-prior estimate
        // returns `None`. The warm-start path, in contrast, scores the
        // identity prior against the correspondences before the loop,
        // seeds `best_inliers` with the 7 geometric inliers, and
        // refines via DLT on those inliers, returning a valid report.
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::identity();
        let inlier_points = [
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 4.7),
            Point3::new(0.0, 0.0, 5.2),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.5, 0.25, 5.5),
        ];
        let mut correspondences: Vec<Correspondence2D3D> = inlier_points
            .iter()
            .map(|p| Correspondence2D3D {
                point2d: camera.project(&pose.transform_world_point(p)).unwrap(),
                point3d: *p,
                confidence: None,
            })
            .collect();
        let n_inliers = correspondences.len();
        let outlier_seeds = [
            (0.4_f64, 0.3, 7.0, 50.0_f64, 50.0_f64),
            (-0.6, 0.5, 8.0, 590.0, 50.0),
            (0.1, -0.8, 5.0, 50.0, 430.0),
            (0.7, -0.3, 6.0, 600.0, 410.0),
            (-0.4, -0.4, 9.0, 320.0, 50.0),
        ];
        for (px, py, pz, cx, cy) in outlier_seeds {
            correspondences.push(Correspondence2D3D {
                point2d: nalgebra::Point2::new(cx, cy),
                point3d: Point3::new(px, py, pz),
                confidence: None,
            });
        }

        let ransac = PnPRansac {
            pose_estimator: DltPnP::default(),
            pose_refiner: None::<crate::pnp::GaussNewtonPoseRefiner>,
            iterations: 1,
            reprojection_threshold: 2.0,
            seed: 1234,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
        };

        // No prior: 1 random iteration is overwhelmingly likely to
        // sample an outlier, and at this fixed seed it does.
        assert!(
            ransac.estimate(&correspondences, &camera).is_none(),
            "control: without a prior, 1-iteration RANSAC should fail at this seed"
        );

        // Identity prior: pre-iteration scoring nails the 7 inliers
        // immediately, the iteration can't beat that, and DLT-on-inliers
        // recovers the identity pose.
        let prior = Pose::identity();
        let warm = ransac
            .estimate_with_pose_prior_and_weights(&correspondences, &camera, Some(&prior), None)
            .expect("warm-started PnP must recover a pose");
        let warm_inliers: usize = warm.inliers.iter().filter(|&&i| i < n_inliers).count();
        assert_eq!(
            warm_inliers, n_inliers,
            "warm-start PnP RANSAC should converge to all 7 geometric inliers"
        );
    }

    #[test]
    fn pnp_ransac_pose_prior_with_zero_inliers_falls_back_to_random_search() {
        use crate::pnp::DltPnP;
        // Same scene, but the prior is grossly wrong (1000 m off): all
        // pixels reproject far outside the threshold, so the prior
        // contributes zero inliers and the RANSAC iteration budget
        // takes over. The result should match the no-prior estimate.
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose = Pose::identity();
        let points = [
            Point3::new(-1.0, -1.0, 4.0),
            Point3::new(1.0, -1.0, 4.5),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 4.7),
            Point3::new(0.0, 0.0, 5.2),
            Point3::new(0.5, -0.25, 6.0),
        ];
        let correspondences: Vec<Correspondence2D3D> = points
            .iter()
            .map(|p| Correspondence2D3D {
                point2d: camera.project(&pose.transform_world_point(p)).unwrap(),
                point3d: *p,
                confidence: None,
            })
            .collect();

        let ransac = PnPRansac {
            pose_estimator: DltPnP::default(),
            pose_refiner: None::<crate::pnp::GaussNewtonPoseRefiner>,
            iterations: 32,
            reprojection_threshold: 1.0e-3,
            seed: 99,
            early_stop_min_iterations: 0,
            early_stop_inlier_ratio: None,
        };

        let bad_prior = Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(1000.0, 1000.0, 1000.0),
        );
        let no_prior = ransac.estimate(&correspondences, &camera).unwrap();
        let with_bad_prior = ransac
            .estimate_with_pose_prior_and_weights(&correspondences, &camera, Some(&bad_prior), None)
            .unwrap();
        assert_eq!(no_prior.inliers, with_bad_prior.inliers);
    }
}
