//! Typed loop constraints between independently reconstructed local submaps.
//!
//! Rotation-only evidence and scale-bearing `Sim3` evidence are intentionally
//! different types. A caller cannot accidentally turn a homography-dominant or
//! low-parallax loop into a scale edge merely by filling a default scale field.

use nalgebra::{Matrix3, Point3, UnitQuaternion, Vector3};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use visloc_core::geometry::Sim3;
use visloc_tracking::umeyama_similarity_transform;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationConstraintGeometry {
    Essential,
    HomographyDominant,
    PureRotation,
}

/// A loop that constrains orientation but makes no translation/scale claim.
#[derive(Debug, Clone, PartialEq)]
pub struct RotationOnlyConstraint {
    pub source_submap_id: u64,
    pub target_submap_id: u64,
    /// Rotation from the source local frame into the target local frame.
    pub target_from_source_rotation: UnitQuaternion<f64>,
    pub inlier_count: usize,
    pub spatial_coverage: f64,
    pub geometry: RotationConstraintGeometry,
}

/// One independently reconstructed same-point relation across two submaps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubmapPointMatch {
    pub source_landmark_id: u64,
    pub target_landmark_id: u64,
    pub source_point: Point3<f64>,
    pub target_point: Point3<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmapSim3AlignmentConfig {
    pub min_correspondences: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    /// Inlier distance divided by the target point-cloud median pair distance.
    pub max_inlier_residual_ratio: f64,
    pub max_mean_residual_ratio: f64,
    pub max_rotation_disagreement_deg: f64,
    /// Median absolute leave-one-out log-scale deviation from the final fit.
    pub max_leave_one_out_log_scale_mad: f64,
    /// Reject nearly collinear point sets using σ₂/σ₁ of centred coordinates.
    pub min_second_to_first_singular_ratio: f64,
    pub min_scale: f64,
    pub max_scale: f64,
    pub ransac_iterations: usize,
    pub random_seed: u64,
}

impl Default for SubmapSim3AlignmentConfig {
    fn default() -> Self {
        Self {
            min_correspondences: 12,
            min_inliers: 10,
            min_inlier_ratio: 0.6,
            max_inlier_residual_ratio: 0.03,
            max_mean_residual_ratio: 0.015,
            max_rotation_disagreement_deg: 10.0,
            max_leave_one_out_log_scale_mad: 0.03,
            min_second_to_first_singular_ratio: 0.01,
            min_scale: 1.0e-3,
            max_scale: 1.0e3,
            ransac_iterations: 512,
            random_seed: 0,
        }
    }
}

/// A full similarity measurement, admitted only from independent 3D geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmapSim3Constraint {
    pub source_submap_id: u64,
    pub target_submap_id: u64,
    /// `target ≈ target_from_source(source)`.
    pub target_from_source: Sim3,
    pub correspondence_count: usize,
    pub inlier_match_indices: Vec<usize>,
    pub inlier_ratio: f64,
    pub mean_residual_ratio: f64,
    pub rotation_disagreement_deg: f64,
    pub leave_one_out_log_scale_mad: f64,
    pub target_scene_scale: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum VerifiedSubmapConstraint {
    RotationOnly(RotationOnlyConstraint),
    Sim3(SubmapSim3Constraint),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmapSim3RejectionReason {
    TooFewCorrespondences,
    NonFinitePoint,
    NonUniqueCorrespondences,
    DegenerateSourceGeometry,
    DegenerateTargetGeometry,
    InvalidTargetSceneScale,
    NoRobustFit,
    ScaleOutOfBounds,
    TooFewInliers,
    LowInlierRatio,
    HighMeanResidual,
    RotationInconsistent,
    UnstableLeaveOneOutScale,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmapSim3Rejection {
    pub reason: SubmapSim3RejectionReason,
    pub correspondence_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_residual_ratio: Option<f64>,
    pub rotation_disagreement_deg: Option<f64>,
    pub leave_one_out_log_scale_mad: Option<f64>,
}

impl SubmapSim3Rejection {
    fn new(reason: SubmapSim3RejectionReason, correspondence_count: usize) -> Self {
        Self {
            reason,
            correspondence_count,
            inlier_count: 0,
            inlier_ratio: 0.0,
            mean_residual_ratio: None,
            rotation_disagreement_deg: None,
            leave_one_out_log_scale_mad: None,
        }
    }
}

/// Robustly align independently reconstructed source points into the target
/// submap and cross-check the result against an independently verified E
/// rotation. No live trajectory pose or depth participates in this estimate.
pub fn estimate_submap_sim3_constraint(
    source_submap_id: u64,
    target_submap_id: u64,
    matches: &[SubmapPointMatch],
    essential_target_from_source_rotation: &UnitQuaternion<f64>,
    config: &SubmapSim3AlignmentConfig,
) -> Result<SubmapSim3Constraint, SubmapSim3Rejection> {
    let count = matches.len();
    if count < config.min_correspondences.max(3) {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::TooFewCorrespondences,
            count,
        ));
    }
    if matches.iter().any(|point_match| {
        !point_match
            .source_point
            .coords
            .iter()
            .all(|value| value.is_finite())
            || !point_match
                .target_point
                .coords
                .iter()
                .all(|value| value.is_finite())
    }) {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::NonFinitePoint,
            count,
        ));
    }
    let unique_source = matches
        .iter()
        .map(|point_match| point_match.source_landmark_id)
        .collect::<std::collections::HashSet<_>>();
    let unique_target = matches
        .iter()
        .map(|point_match| point_match.target_landmark_id)
        .collect::<std::collections::HashSet<_>>();
    if unique_source.len() != count || unique_target.len() != count {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::NonUniqueCorrespondences,
            count,
        ));
    }
    let source_points = matches
        .iter()
        .map(|point_match| point_match.source_point)
        .collect::<Vec<_>>();
    let target_points = matches
        .iter()
        .map(|point_match| point_match.target_point)
        .collect::<Vec<_>>();
    if point_cloud_second_to_first_ratio(&source_points) < config.min_second_to_first_singular_ratio
    {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::DegenerateSourceGeometry,
            count,
        ));
    }
    if point_cloud_second_to_first_ratio(&target_points) < config.min_second_to_first_singular_ratio
    {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::DegenerateTargetGeometry,
            count,
        ));
    }
    let target_scene_scale = median_pairwise_distance(&target_points);
    if !target_scene_scale.is_finite() || target_scene_scale <= 1.0e-12 {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::InvalidTargetSceneScale,
            count,
        ));
    }
    let threshold = (target_scene_scale * config.max_inlier_residual_ratio).max(1.0e-9);
    let mut rng = StdRng::seed_from_u64(config.random_seed);
    let mut best_inliers = Vec::new();
    let mut best_mean = f64::INFINITY;
    for _ in 0..config.ransac_iterations.max(1) {
        let Some(indices) = sample_three(count, &mut rng) else {
            break;
        };
        let source = indices.map(|index| matches[index].source_point);
        let target = indices.map(|index| matches[index].target_point);
        let Some(fit) = umeyama_similarity_transform(&source, &target, true) else {
            continue;
        };
        if !valid_scale(fit.scale, config) {
            continue;
        }
        let residuals = matches
            .iter()
            .map(|point_match| {
                (fit.apply(&point_match.source_point) - point_match.target_point).norm()
            })
            .collect::<Vec<_>>();
        let inliers = residuals
            .iter()
            .enumerate()
            .filter_map(|(index, residual)| (*residual <= threshold).then_some(index))
            .collect::<Vec<_>>();
        let mean = if inliers.is_empty() {
            f64::INFINITY
        } else {
            inliers.iter().map(|index| residuals[*index]).sum::<f64>() / inliers.len() as f64
        };
        if inliers.len() > best_inliers.len()
            || (inliers.len() == best_inliers.len() && mean < best_mean)
        {
            best_inliers = inliers;
            best_mean = mean;
        }
    }
    if best_inliers.is_empty() {
        return Err(SubmapSim3Rejection::new(
            SubmapSim3RejectionReason::NoRobustFit,
            count,
        ));
    }
    let inlier_ratio = best_inliers.len() as f64 / count as f64;
    let mut rejection = SubmapSim3Rejection::new(SubmapSim3RejectionReason::TooFewInliers, count);
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = inlier_ratio;
    if best_inliers.len() < config.min_inliers.max(3) {
        return Err(rejection);
    }
    if inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }

    let first_inlier_source = best_inliers
        .iter()
        .map(|index| matches[*index].source_point)
        .collect::<Vec<_>>();
    let first_inlier_target = best_inliers
        .iter()
        .map(|index| matches[*index].target_point)
        .collect::<Vec<_>>();
    let Some(first_refit) =
        umeyama_similarity_transform(&first_inlier_source, &first_inlier_target, true)
    else {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    };
    if !valid_scale(first_refit.scale, config) {
        rejection.reason = SubmapSim3RejectionReason::ScaleOutOfBounds;
        return Err(rejection);
    }
    // Reclassify against the consensus refit rather than reporting the stale
    // minimal-hypothesis inlier set. This also allows genuine points omitted
    // by the winning sample's noisier fit to re-enter exactly once.
    best_inliers = matches
        .iter()
        .enumerate()
        .filter_map(|(index, point_match)| {
            ((first_refit.apply(&point_match.source_point) - point_match.target_point).norm()
                <= threshold)
                .then_some(index)
        })
        .collect();
    rejection.inlier_count = best_inliers.len();
    rejection.inlier_ratio = best_inliers.len() as f64 / count as f64;
    if best_inliers.len() < config.min_inliers.max(3) {
        rejection.reason = SubmapSim3RejectionReason::TooFewInliers;
        return Err(rejection);
    }
    if rejection.inlier_ratio < config.min_inlier_ratio {
        rejection.reason = SubmapSim3RejectionReason::LowInlierRatio;
        return Err(rejection);
    }
    let inlier_source = best_inliers
        .iter()
        .map(|index| matches[*index].source_point)
        .collect::<Vec<_>>();
    let inlier_target = best_inliers
        .iter()
        .map(|index| matches[*index].target_point)
        .collect::<Vec<_>>();
    let Some(fit) = umeyama_similarity_transform(&inlier_source, &inlier_target, true) else {
        rejection.reason = SubmapSim3RejectionReason::NoRobustFit;
        return Err(rejection);
    };
    if !valid_scale(fit.scale, config) {
        rejection.reason = SubmapSim3RejectionReason::ScaleOutOfBounds;
        return Err(rejection);
    }
    let mean_residual_ratio = inlier_source
        .iter()
        .zip(&inlier_target)
        .map(|(source, target)| (fit.apply(source) - target).norm())
        .sum::<f64>()
        / inlier_source.len() as f64
        / target_scene_scale;
    rejection.mean_residual_ratio = Some(mean_residual_ratio);
    if !mean_residual_ratio.is_finite() || mean_residual_ratio > config.max_mean_residual_ratio {
        rejection.reason = SubmapSim3RejectionReason::HighMeanResidual;
        return Err(rejection);
    }

    let fit_rotation = UnitQuaternion::from_rotation_matrix(&fit.rotation);
    let rotation_disagreement_deg = fit_rotation
        .rotation_to(essential_target_from_source_rotation)
        .angle()
        .to_degrees();
    rejection.rotation_disagreement_deg = Some(rotation_disagreement_deg);
    if !rotation_disagreement_deg.is_finite()
        || rotation_disagreement_deg > config.max_rotation_disagreement_deg
    {
        rejection.reason = SubmapSim3RejectionReason::RotationInconsistent;
        return Err(rejection);
    }

    let leave_one_out_log_scale_mad =
        leave_one_out_log_scale_mad(&inlier_source, &inlier_target, fit.scale)
            .unwrap_or(f64::INFINITY);
    rejection.leave_one_out_log_scale_mad = Some(leave_one_out_log_scale_mad);
    if !leave_one_out_log_scale_mad.is_finite()
        || leave_one_out_log_scale_mad > config.max_leave_one_out_log_scale_mad
    {
        rejection.reason = SubmapSim3RejectionReason::UnstableLeaveOneOutScale;
        return Err(rejection);
    }

    let final_inlier_ratio = best_inliers.len() as f64 / count as f64;
    Ok(SubmapSim3Constraint {
        source_submap_id,
        target_submap_id,
        target_from_source: Sim3::new(fit_rotation, fit.translation, fit.scale),
        correspondence_count: count,
        inlier_match_indices: best_inliers,
        inlier_ratio: final_inlier_ratio,
        mean_residual_ratio,
        rotation_disagreement_deg,
        leave_one_out_log_scale_mad,
        target_scene_scale,
    })
}

fn valid_scale(scale: f64, config: &SubmapSim3AlignmentConfig) -> bool {
    scale.is_finite() && scale >= config.min_scale && scale <= config.max_scale
}

fn sample_three(count: usize, rng: &mut StdRng) -> Option<[usize; 3]> {
    if count < 3 {
        return None;
    }
    for _ in 0..64 {
        let sample = [
            rng.gen_range(0..count),
            rng.gen_range(0..count),
            rng.gen_range(0..count),
        ];
        if sample[0] != sample[1] && sample[0] != sample[2] && sample[1] != sample[2] {
            return Some(sample);
        }
    }
    None
}

fn point_cloud_second_to_first_ratio(points: &[Point3<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let centre = points
        .iter()
        .fold(Vector3::zeros(), |sum, point| sum + point.coords)
        / points.len() as f64;
    let covariance = points.iter().fold(Matrix3::zeros(), |sum, point| {
        let delta = point.coords - centre;
        sum + delta * delta.transpose()
    });
    let mut singular = covariance
        .svd(false, false)
        .singular_values
        .as_slice()
        .to_vec();
    singular.sort_by(|left, right| right.total_cmp(left));
    if singular[0] <= f64::EPSILON {
        0.0
    } else {
        singular[1] / singular[0]
    }
}

fn median_pairwise_distance(points: &[Point3<f64>]) -> f64 {
    let mut distances = Vec::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let distance = (points[i] - points[j]).norm();
            if distance.is_finite() && distance > 0.0 {
                distances.push(distance);
            }
        }
    }
    median(distances).unwrap_or(0.0)
}

fn leave_one_out_log_scale_mad(
    source: &[Point3<f64>],
    target: &[Point3<f64>],
    reference_scale: f64,
) -> Option<f64> {
    if source.len() != target.len() || source.len() < 4 || reference_scale <= 0.0 {
        return None;
    }
    let mut deviations = Vec::with_capacity(source.len());
    for omitted in 0..source.len() {
        let kept_source = source
            .iter()
            .enumerate()
            .filter_map(|(index, point)| (index != omitted).then_some(*point))
            .collect::<Vec<_>>();
        let kept_target = target
            .iter()
            .enumerate()
            .filter_map(|(index, point)| (index != omitted).then_some(*point))
            .collect::<Vec<_>>();
        let fit = umeyama_similarity_transform(&kept_source, &kept_target, true)?;
        if !fit.scale.is_finite() || fit.scale <= 0.0 {
            return None;
        }
        deviations.push((fit.scale / reference_scale).ln().abs());
    }
    median(deviations)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<SubmapPointMatch>, Sim3) {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.08, -0.12, 0.21),
            Vector3::new(1.2, -0.4, 0.8),
            3.5,
        );
        let mut matches = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                let source =
                    Point3::new(x as f64 * 0.4, y as f64 * 0.3, ((x + y) % 3) as f64 * 0.2);
                let target = truth.transform_point(&source);
                let id = matches.len() as u64;
                matches.push(SubmapPointMatch {
                    source_landmark_id: id,
                    target_landmark_id: id + 100,
                    source_point: source,
                    target_point: target,
                });
            }
        }
        // Four deterministic outliers exercise the robust fit and index report.
        for index in 0..4 {
            matches[index].target_point += Vector3::new(4.0 + index as f64, -3.0, 2.0);
        }
        (matches, truth)
    }

    #[test]
    fn robust_independent_geometry_recovers_scale_rotation_and_outliers() {
        let (matches, truth) = fixture();
        let result = estimate_submap_sim3_constraint(
            7,
            9,
            &matches,
            &truth.rotation,
            &SubmapSim3AlignmentConfig::default(),
        )
        .expect("well-conditioned independent point sets should align");

        assert_eq!(result.source_submap_id, 7);
        assert_eq!(result.target_submap_id, 9);
        assert_eq!(result.inlier_match_indices.len(), matches.len() - 4);
        assert!((result.target_from_source.scale - truth.scale).abs() < 1.0e-9);
        assert!(result.rotation_disagreement_deg < 1.0e-7);
        assert!(result.mean_residual_ratio < 1.0e-10);
        assert!(result.leave_one_out_log_scale_mad < 1.0e-10);
        assert!(result.inlier_match_indices.iter().all(|index| *index >= 4));
    }

    #[test]
    fn correct_3d_fit_cannot_bypass_independent_rotation_consensus() {
        let (matches, truth) = fixture();
        let wrong_rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, 1.2) * truth.rotation;
        let error = estimate_submap_sim3_constraint(
            7,
            9,
            &matches,
            &wrong_rotation,
            &SubmapSim3AlignmentConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            error.reason,
            SubmapSim3RejectionReason::RotationInconsistent
        );
        assert!(error.rotation_disagreement_deg.unwrap() > 60.0);
    }

    #[test]
    fn collinear_geometry_is_rejected_before_ransac() {
        let matches = (0..20)
            .map(|index| {
                let source = Point3::new(index as f64, 0.0, 0.0);
                SubmapPointMatch {
                    source_landmark_id: index,
                    target_landmark_id: index,
                    source_point: source,
                    target_point: Point3::new(2.0 * source.x + 1.0, 0.0, 0.0),
                }
            })
            .collect::<Vec<_>>();
        let error = estimate_submap_sim3_constraint(
            1,
            2,
            &matches,
            &UnitQuaternion::identity(),
            &SubmapSim3AlignmentConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            error.reason,
            SubmapSim3RejectionReason::DegenerateSourceGeometry
        );
    }

    #[test]
    fn repeated_landmark_identity_cannot_inflate_consensus() {
        let (mut matches, truth) = fixture();
        matches[1].source_landmark_id = matches[0].source_landmark_id;
        let error = estimate_submap_sim3_constraint(
            1,
            2,
            &matches,
            &truth.rotation,
            &SubmapSim3AlignmentConfig::default(),
        )
        .unwrap_err();
        assert_eq!(
            error.reason,
            SubmapSim3RejectionReason::NonUniqueCorrespondences
        );
    }

    #[test]
    fn constraint_enum_preserves_rotation_only_observability() {
        let constraint = VerifiedSubmapConstraint::RotationOnly(RotationOnlyConstraint {
            source_submap_id: 1,
            target_submap_id: 2,
            target_from_source_rotation: UnitQuaternion::identity(),
            inlier_count: 80,
            spatial_coverage: 0.6,
            geometry: RotationConstraintGeometry::PureRotation,
        });
        assert!(matches!(
            constraint,
            VerifiedSubmapConstraint::RotationOnly(RotationOnlyConstraint {
                geometry: RotationConstraintGeometry::PureRotation,
                ..
            })
        ));
    }
}
