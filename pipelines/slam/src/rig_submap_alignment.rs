//! Bounded camera-centre alignment between independently gauged rig submaps.
//!
//! A rig frame is represented once in each local submap by its world-to-camera
//! [`Pose`].  The adapter turns the common rig-frame camera centres into the
//! [`SubmapPointMatch`] representation consumed by the landmark-only Sim(3)
//! verifier.  It also derives the local-gauge rotation from the two camera
//! attitudes and uses a bounded medoid consensus before invoking that
//! verifier.
//!
//! The input may contain a large trajectory, but all quadratic work is over
//! the retained boundary only.  Common rig-frame ids are sorted in ascending
//! order and the earliest `max_boundary_frames` are retained.  This makes the
//! adapter suitable for a moving seam whose newest window is still being
//! registered without allowing the cost of a frame Cartesian product to grow
//! with the full trajectories.
//!
//! Callers conventionally pass the newer submap window as `source_frames` and
//! the previous/Atlas-adjacent window as `target_frames`.  In that orientation,
//! the earliest common global rig-frame ids are the seam boundary by
//! construction.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use nalgebra::{Point3, UnitQuaternion};
use visloc_core::geometry::Pose;

use crate::{
    estimate_submap_sim3_constraint, SubmapPointMatch, SubmapSim3AlignmentConfig,
    SubmapSim3Constraint, SubmapSim3Rejection,
};

/// Configuration for the bounded rig-frame seam adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentConfig {
    /// Maximum number of common rig frames retained for the seam.  The
    /// retained ids are always the earliest ids in ascending order.
    pub max_boundary_frames: usize,
    /// Angular radius of the independent local-gauge rotation consensus.
    pub rotation_consensus_deg: f64,
    /// Minimum fraction of retained rig-frame rotations in that consensus.
    pub min_rotation_consensus_ratio: f64,
    /// Existing robust 3D Sim(3) alignment and residual gates.
    pub alignment: SubmapSim3AlignmentConfig,
}

impl Default for RigSubmapAlignmentConfig {
    fn default() -> Self {
        Self {
            max_boundary_frames: 64,
            rotation_consensus_deg: 5.0,
            min_rotation_consensus_ratio: 0.6,
            alignment: SubmapSim3AlignmentConfig::default(),
        }
    }
}

/// Why a rig-submap camera-centre seam was not admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapAlignmentRejectionReason {
    /// The bounded seam or rotation-consensus settings are not finite or are
    /// outside their fail-closed domains.
    InvalidConfiguration,
    /// A source or target trajectory contains the same global rig-frame id
    /// more than once.
    DuplicateFrameId,
    /// A source or target pose (including its rotation or translation) is not
    /// finite.
    NonFinitePose,
    /// A finite pose produced a non-finite camera centre.
    NonFiniteCameraCenter,
    /// The retained common boundary is smaller than the Sim(3) minimum.
    InsufficientCommonFrames,
    /// The independent camera-attitude rotations do not form a large enough
    /// consensus cluster.
    LowRotationConsensus,
    /// The existing robust Sim(3) estimator rejected the camera-centre
    /// matches.  The detailed rejection is retained in the error.
    Sim3Rejected,
}

/// Side of a rig-frame input that supplied a malformed record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigSubmapInputSide {
    Source,
    Target,
}

/// Deterministic and auditable counts from a rig-submap alignment attempt.
///
/// Counts are populated on both success and rejection.  In particular,
/// `common_frame_count` is the full unique intersection while
/// `retained_frame_count` is the bounded seam actually passed to the Sim(3)
/// estimator.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentDiagnostics {
    pub source_frame_count: usize,
    pub target_frame_count: usize,
    pub source_unique_frame_count: usize,
    pub target_unique_frame_count: usize,
    pub common_frame_count: usize,
    pub retained_frame_count: usize,
    pub retained_frame_ids: Vec<u64>,
    pub rotation_candidate_count: usize,
    pub rotation_consensus_count: usize,
    pub rotation_consensus_ratio: f64,
    pub rotation_consensus_max_disagreement_deg: f64,
    pub rotation_consensus_threshold_deg: f64,
    pub rotation_medoid_frame_id: Option<u64>,
    pub camera_centre_match_count: usize,
    pub sim3_inlier_count: Option<usize>,
    pub sim3_inlier_ratio: Option<f64>,
}

impl RigSubmapAlignmentDiagnostics {
    fn from_counts(
        source_frame_count: usize,
        target_frame_count: usize,
        source_unique_frame_count: usize,
        target_unique_frame_count: usize,
        common_frame_count: usize,
        retained_frame_ids: Vec<u64>,
        rotation_consensus_threshold_deg: f64,
    ) -> Self {
        Self {
            source_frame_count,
            target_frame_count,
            source_unique_frame_count,
            target_unique_frame_count,
            common_frame_count,
            retained_frame_count: retained_frame_ids.len(),
            retained_frame_ids,
            rotation_candidate_count: 0,
            rotation_consensus_count: 0,
            rotation_consensus_ratio: 0.0,
            rotation_consensus_max_disagreement_deg: 0.0,
            rotation_consensus_threshold_deg,
            rotation_medoid_frame_id: None,
            camera_centre_match_count: 0,
            sim3_inlier_count: None,
            sim3_inlier_ratio: None,
        }
    }
}

/// Typed rejection carrying the partial seam diagnostics and, when the
/// wrapper reached the robust estimator, its detailed rejection.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentRejection {
    pub reason: RigSubmapAlignmentRejectionReason,
    pub diagnostics: Box<RigSubmapAlignmentDiagnostics>,
    pub offending_side: Option<RigSubmapInputSide>,
    pub offending_frame_id: Option<u64>,
    pub sim3_rejection: Option<Box<SubmapSim3Rejection>>,
}

/// Error alias using the conventional `Error` name for callers that prefer
/// `Result<_, RigSubmapAlignmentError>`.
pub type RigSubmapAlignmentError = RigSubmapAlignmentRejection;

impl fmt::Display for RigSubmapAlignmentRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason {
            RigSubmapAlignmentRejectionReason::InvalidConfiguration => write!(
                f,
                "rig-submap alignment has an invalid boundary or rotation-consensus configuration"
            ),
            RigSubmapAlignmentRejectionReason::DuplicateFrameId => {
                write!(f, "rig-submap seam contains a duplicate frame id")
            }
            RigSubmapAlignmentRejectionReason::NonFinitePose => {
                write!(f, "rig-submap seam contains a non-finite pose")
            }
            RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter => {
                write!(f, "rig-submap seam contains a non-finite camera centre")
            }
            RigSubmapAlignmentRejectionReason::InsufficientCommonFrames => write!(
                f,
                "rig-submap seam has {} retained common frames",
                self.diagnostics.retained_frame_count
            ),
            RigSubmapAlignmentRejectionReason::LowRotationConsensus => write!(
                f,
                "rig-submap rotation consensus ratio {} is below the configured minimum",
                self.diagnostics.rotation_consensus_ratio,
            ),
            RigSubmapAlignmentRejectionReason::Sim3Rejected => {
                write!(f, "rig-submap Sim(3) alignment was rejected")
            }
        }
    }
}

impl Error for RigSubmapAlignmentRejection {}

/// A verified camera-centre seam constraint plus the bounded evidence used to
/// admit it.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSubmapAlignmentResult {
    pub constraint: SubmapSim3Constraint,
    pub diagnostics: RigSubmapAlignmentDiagnostics,
}

/// Estimate a `target local <- source local` Sim(3) from corresponding global
/// rig-frame poses in two independently reconstructed submap gauges.
///
/// Callers should pass the newer submap window as `source_frames` and the
/// previous/Atlas-adjacent window as `target_frames`; the earliest common ids
/// then identify the new-window seam boundary.
///
/// The source and target slices are indexed by global rig-frame id.  Duplicate
/// ids are rejected.  Their unique intersection is sorted by id and truncated
/// to `config.max_boundary_frames` before any rotation consensus or Sim(3)
/// work.  For each retained frame, the camera-centre pair is represented by a
/// [`SubmapPointMatch`] whose source and target ids are that global rig-frame
/// id.  The independent rotation candidate is exactly
/// `q_target_cw.inverse() * q_source_cw`.
pub fn estimate_rig_submap_sim3_constraint(
    source_submap_id: u64,
    target_submap_id: u64,
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
    config: &RigSubmapAlignmentConfig,
) -> Result<RigSubmapAlignmentResult, RigSubmapAlignmentRejection> {
    if !valid_config(config) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::InvalidConfiguration,
            diagnostics: Box::new(RigSubmapAlignmentDiagnostics::from_counts(
                source_frames.len(),
                target_frames.len(),
                0,
                0,
                0,
                Vec::new(),
                config.rotation_consensus_deg,
            )),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
        });
    }

    let source_unique = unique_frame_ids(source_frames);
    let target_unique = unique_frame_ids(target_frames);
    let base_diagnostics = RigSubmapAlignmentDiagnostics::from_counts(
        source_frames.len(),
        target_frames.len(),
        source_unique.len(),
        target_unique.len(),
        0,
        Vec::new(),
        config.rotation_consensus_deg,
    );

    if source_unique.len() != source_frames.len() {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::DuplicateFrameId,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(RigSubmapInputSide::Source),
            offending_frame_id: duplicate_frame_id(source_frames),
            sim3_rejection: None,
        });
    }
    if target_unique.len() != target_frames.len() {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::DuplicateFrameId,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(RigSubmapInputSide::Target),
            offending_frame_id: duplicate_frame_id(target_frames),
            sim3_rejection: None,
        });
    }

    if let Some((side, frame_id)) = first_nonfinite_pose(source_frames, target_frames) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::NonFinitePose,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(side),
            offending_frame_id: Some(frame_id),
            sim3_rejection: None,
        });
    }

    if let Some((side, frame_id)) = first_nonfinite_camera_center(source_frames, target_frames) {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
            diagnostics: Box::new(base_diagnostics),
            offending_side: Some(side),
            offending_frame_id: Some(frame_id),
            sim3_rejection: None,
        });
    }

    let common_frame_ids = source_unique
        .keys()
        .filter(|frame_id| target_unique.contains_key(frame_id))
        .copied()
        .collect::<Vec<_>>();
    let retained_frame_ids = common_frame_ids
        .iter()
        .take(config.max_boundary_frames)
        .copied()
        .collect::<Vec<_>>();
    let mut diagnostics = RigSubmapAlignmentDiagnostics::from_counts(
        source_frames.len(),
        target_frames.len(),
        source_unique.len(),
        target_unique.len(),
        common_frame_ids.len(),
        retained_frame_ids.clone(),
        config.rotation_consensus_deg,
    );

    let required_common_frames = config.alignment.min_correspondences.max(3);
    if retained_frame_ids.len() < required_common_frames {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::InsufficientCommonFrames,
            diagnostics: Box::new(diagnostics),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
        });
    }

    let mut matches = Vec::with_capacity(retained_frame_ids.len());
    let mut rotations = Vec::with_capacity(retained_frame_ids.len());
    for frame_id in &retained_frame_ids {
        let source_pose = source_unique
            .get(frame_id)
            .expect("retained frame id came from source map");
        let target_pose = target_unique
            .get(frame_id)
            .expect("retained frame id came from target map");
        let source_center = source_pose.camera_center_world();
        let target_center = target_pose.camera_center_world();
        if !point_is_finite(&source_center) {
            diagnostics.camera_centre_match_count = matches.len();
            return Err(RigSubmapAlignmentRejection {
                reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
                diagnostics: Box::new(diagnostics),
                offending_side: Some(RigSubmapInputSide::Source),
                offending_frame_id: Some(*frame_id),
                sim3_rejection: None,
            });
        }
        if !point_is_finite(&target_center) {
            diagnostics.camera_centre_match_count = matches.len();
            return Err(RigSubmapAlignmentRejection {
                reason: RigSubmapAlignmentRejectionReason::NonFiniteCameraCenter,
                diagnostics: Box::new(diagnostics),
                offending_side: Some(RigSubmapInputSide::Target),
                offending_frame_id: Some(*frame_id),
                sim3_rejection: None,
            });
        }
        matches.push(SubmapPointMatch {
            source_landmark_id: *frame_id,
            target_landmark_id: *frame_id,
            source_point: source_center,
            target_point: target_center,
        });
        rotations.push(IndependentRotationCandidate {
            frame_id: *frame_id,
            rotation: target_pose.world_to_camera.rotation.inverse()
                * source_pose.world_to_camera.rotation,
        });
    }
    diagnostics.camera_centre_match_count = matches.len();
    diagnostics.rotation_candidate_count = rotations.len();

    let consensus = rotation_consensus(&rotations, config.rotation_consensus_deg);
    diagnostics.rotation_consensus_count = consensus.indices.len();
    diagnostics.rotation_consensus_ratio =
        consensus.indices.len() as f64 / rotations.len().max(1) as f64;
    diagnostics.rotation_consensus_max_disagreement_deg = consensus.max_disagreement_deg;
    diagnostics.rotation_medoid_frame_id = Some(consensus.medoid_frame_id);
    if !diagnostics.rotation_consensus_ratio.is_finite()
        || diagnostics.rotation_consensus_ratio < config.min_rotation_consensus_ratio
    {
        return Err(RigSubmapAlignmentRejection {
            reason: RigSubmapAlignmentRejectionReason::LowRotationConsensus,
            diagnostics: Box::new(diagnostics),
            offending_side: None,
            offending_frame_id: None,
            sim3_rejection: None,
        });
    }

    let constraint = match estimate_submap_sim3_constraint(
        source_submap_id,
        target_submap_id,
        &matches,
        &consensus.rotation,
        &config.alignment,
    ) {
        Ok(constraint) => constraint,
        Err(sim3_rejection) => {
            diagnostics.sim3_inlier_count = Some(sim3_rejection.inlier_count);
            diagnostics.sim3_inlier_ratio = Some(sim3_rejection.inlier_ratio);
            return Err(RigSubmapAlignmentRejection {
                reason: RigSubmapAlignmentRejectionReason::Sim3Rejected,
                diagnostics: Box::new(diagnostics),
                offending_side: None,
                offending_frame_id: None,
                sim3_rejection: Some(Box::new(sim3_rejection)),
            });
        }
    };
    diagnostics.sim3_inlier_count = Some(constraint.inlier_match_indices.len());
    diagnostics.sim3_inlier_ratio = Some(constraint.inlier_ratio);
    Ok(RigSubmapAlignmentResult {
        constraint,
        diagnostics,
    })
}

/// Alias emphasizing that the returned Sim(3) is built from rig camera
/// centres.  It has the same bounded implementation and result type as
/// [`estimate_rig_submap_sim3_constraint`].
pub fn estimate_rig_submap_camera_centre_constraint(
    source_submap_id: u64,
    target_submap_id: u64,
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
    config: &RigSubmapAlignmentConfig,
) -> Result<RigSubmapAlignmentResult, RigSubmapAlignmentRejection> {
    estimate_rig_submap_sim3_constraint(
        source_submap_id,
        target_submap_id,
        source_frames,
        target_frames,
        config,
    )
}

#[derive(Debug, Clone, Copy)]
struct IndependentRotationCandidate {
    frame_id: u64,
    rotation: UnitQuaternion<f64>,
}

#[derive(Debug, Clone)]
struct RotationConsensus {
    rotation: UnitQuaternion<f64>,
    medoid_frame_id: u64,
    indices: Vec<usize>,
    max_disagreement_deg: f64,
}

/// Select the largest angular cluster and use its deterministic medoid as the
/// independent rotation.  The pairwise matrix is computed once, so this is
/// O(B²) in both rotations and angular evaluations.
fn rotation_consensus(
    candidates: &[IndependentRotationCandidate],
    radius_deg: f64,
) -> RotationConsensus {
    debug_assert!(!candidates.is_empty());
    let count = candidates.len();
    let radius = radius_deg.max(0.0).to_radians();
    let mut distances = vec![vec![0.0; count]; count];
    for i in 0..count {
        for j in (i + 1)..count {
            let distance = candidates[i]
                .rotation
                .rotation_to(&candidates[j].rotation)
                .angle();
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    let mut medoid_index = 0;
    let mut medoid_support = 0;
    let mut medoid_distance_sum = f64::INFINITY;
    let mut medoid_cluster = Vec::new();
    for candidate_index in 0..count {
        let cluster = (0..count)
            .filter(|&other_index| distances[candidate_index][other_index] <= radius)
            .collect::<Vec<_>>();
        let support = cluster.len();
        let distance_sum = cluster
            .iter()
            .map(|&other_index| distances[candidate_index][other_index])
            .sum::<f64>();
        let frame_id = candidates[candidate_index].frame_id;
        let current_frame_id = candidates[medoid_index].frame_id;
        let better = support > medoid_support
            || (support == medoid_support
                && (distance_sum < medoid_distance_sum
                    || (distance_sum == medoid_distance_sum && frame_id < current_frame_id)));
        if better {
            medoid_index = candidate_index;
            medoid_support = support;
            medoid_distance_sum = distance_sum;
            medoid_cluster = cluster;
        }
    }
    if medoid_cluster.is_empty() {
        medoid_cluster.push(medoid_index);
    }
    // The input is sorted by frame id before this function is called.  Keep
    // the cluster in that order so the inlier evidence is reproducible.
    medoid_cluster.sort_unstable();
    let max_disagreement_deg = medoid_cluster
        .iter()
        .map(|&index| distances[medoid_index][index])
        .fold(0.0_f64, f64::max)
        .to_degrees();
    RotationConsensus {
        rotation: candidates[medoid_index].rotation,
        medoid_frame_id: candidates[medoid_index].frame_id,
        indices: medoid_cluster,
        max_disagreement_deg,
    }
}

fn unique_frame_ids(frames: &[(u64, Pose)]) -> BTreeMap<u64, &Pose> {
    let mut unique = BTreeMap::new();
    for (frame_id, pose) in frames {
        unique.entry(*frame_id).or_insert(pose);
    }
    unique
}

fn valid_config(config: &RigSubmapAlignmentConfig) -> bool {
    config.max_boundary_frames > 0
        && config.rotation_consensus_deg.is_finite()
        && config.rotation_consensus_deg > 0.0
        && config.rotation_consensus_deg <= 180.0
        && config.min_rotation_consensus_ratio.is_finite()
        && config.min_rotation_consensus_ratio > 0.0
        && config.min_rotation_consensus_ratio <= 1.0
}

fn duplicate_frame_id(frames: &[(u64, Pose)]) -> Option<u64> {
    let mut seen = BTreeSet::new();
    frames
        .iter()
        .find_map(|(frame_id, _)| (!seen.insert(*frame_id)).then_some(*frame_id))
}

fn first_nonfinite_pose(
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
) -> Option<(RigSubmapInputSide, u64)> {
    source_frames
        .iter()
        .find_map(|(frame_id, pose)| {
            (!pose_is_finite(pose)).then_some((RigSubmapInputSide::Source, *frame_id))
        })
        .or_else(|| {
            target_frames.iter().find_map(|(frame_id, pose)| {
                (!pose_is_finite(pose)).then_some((RigSubmapInputSide::Target, *frame_id))
            })
        })
}

fn first_nonfinite_camera_center(
    source_frames: &[(u64, Pose)],
    target_frames: &[(u64, Pose)],
) -> Option<(RigSubmapInputSide, u64)> {
    source_frames
        .iter()
        .find_map(|(frame_id, pose)| {
            (!point_is_finite(&pose.camera_center_world()))
                .then_some((RigSubmapInputSide::Source, *frame_id))
        })
        .or_else(|| {
            target_frames.iter().find_map(|(frame_id, pose)| {
                (!point_is_finite(&pose.camera_center_world()))
                    .then_some((RigSubmapInputSide::Target, *frame_id))
            })
        })
}

fn pose_is_finite(pose: &Pose) -> bool {
    pose.world_to_camera
        .rotation
        .coords
        .iter()
        .all(|value| value.is_finite())
        && pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
}

fn point_is_finite(point: &Point3<f64>) -> bool {
    point.coords.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::Sim3;

    type RigFrames = Vec<(u64, Pose)>;

    fn pose_from_center(rotation: UnitQuaternion<f64>, center: Point3<f64>) -> Pose {
        Pose::from_world_to_camera(rotation, -rotation.transform_vector(&center.coords))
    }

    fn fixture(count: usize) -> (RigFrames, RigFrames, Sim3) {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.09, -0.13, 0.2),
            Vector3::new(1.3, -0.7, 0.5),
            2.4,
        );
        let mut source = Vec::with_capacity(count);
        let mut target = Vec::with_capacity(count);
        for index in 0..count {
            let frame_id = index as u64;
            let centre = Point3::new(
                index as f64 * 0.17,
                (index as f64 * 0.31).sin() * 0.7,
                (index as f64 * 0.21).cos() * 0.5,
            );
            let source_rotation = UnitQuaternion::from_euler_angles(
                index as f64 * 0.01,
                index as f64 * -0.007,
                index as f64 * 0.013,
            );
            let target_center = truth.transform_point(&centre);
            let target_rotation = source_rotation * truth.rotation.inverse();
            source.push((frame_id, pose_from_center(source_rotation, centre)));
            target.push((frame_id, pose_from_center(target_rotation, target_center)));
        }
        // Deliberately scramble input order: the adapter's result must still
        // use the ascending frame-id seam boundary.
        source.reverse();
        target.rotate_left((count / 3).min(count));
        (source, target, truth)
    }

    fn permissive_config() -> RigSubmapAlignmentConfig {
        RigSubmapAlignmentConfig {
            alignment: SubmapSim3AlignmentConfig {
                min_correspondences: 3,
                min_inliers: 3,
                min_inlier_ratio: 0.5,
                max_inlier_residual_ratio: 0.1,
                max_mean_residual_ratio: 0.1,
                max_rotation_disagreement_deg: 10.0,
                max_leave_one_out_log_scale_mad: 0.1,
                min_second_to_first_singular_ratio: 0.0,
                ransac_iterations: 512,
                ..SubmapSim3AlignmentConfig::default()
            },
            ..RigSubmapAlignmentConfig::default()
        }
    }

    #[test]
    fn known_sim3_is_recovered_from_rig_camera_centres() {
        let (source, target, truth) = fixture(24);
        let result =
            estimate_rig_submap_sim3_constraint(11, 17, &source, &target, &permissive_config())
                .expect("known Sim3 should pass");
        assert_eq!(result.constraint.source_submap_id, 11);
        assert_eq!(result.constraint.target_submap_id, 17);
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-9);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-9
        );
        assert!(
            result
                .constraint
                .target_from_source
                .rotation
                .rotation_to(&truth.rotation)
                .angle()
                < 1.0e-9
        );
        assert_eq!(
            result.diagnostics.retained_frame_ids,
            (0..24).collect::<Vec<_>>()
        );
    }

    #[test]
    fn centre_and_rotation_outliers_are_rejected_by_the_two_consensus_stages() {
        let (source, mut target, truth) = fixture(30);
        for index in [2_usize, 5, 8] {
            target[index].1 = pose_from_center(
                UnitQuaternion::from_euler_angles(1.2, -0.4, 0.7),
                Point3::new(40.0 + index as f64, -30.0, 20.0),
            );
        }
        let result =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &permissive_config())
                .expect("minority outliers should be robustly rejected");
        assert!(result.diagnostics.rotation_consensus_ratio > 0.8);
        assert!(result.constraint.inlier_ratio > 0.8);
        assert!((result.constraint.target_from_source.scale - truth.scale).abs() < 1.0e-6);
        assert!(
            (result.constraint.target_from_source.translation - truth.translation).norm() < 1.0e-6
        );
    }

    #[test]
    fn duplicate_ids_are_rejected_before_intersection() {
        let (mut source, target, _) = fixture(12);
        source.push(
            source
                .iter()
                .find(|(frame_id, _)| *frame_id == 0)
                .unwrap()
                .clone(),
        );
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &permissive_config())
                .unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::DuplicateFrameId
        );
        assert_eq!(error.offending_side, Some(RigSubmapInputSide::Source));
        assert_eq!(error.offending_frame_id, Some(0));
    }

    #[test]
    fn earliest_boundary_cap_is_sorted_and_deterministic() {
        let (source, target, _) = fixture(20);
        let mut config = permissive_config();
        config.max_boundary_frames = 5;
        let first = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("five-frame seam should pass");
        let second = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
            .expect("same input should be deterministic");
        assert_eq!(first, second);
        assert_eq!(first.diagnostics.retained_frame_ids, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn low_rotation_consensus_is_typed() {
        let (source, mut target, _) = fixture(12);
        for (index, (_, target_pose)) in target.iter_mut().enumerate() {
            let centre = target_pose.camera_center_world();
            let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, (index as f64 + 1.0) * 0.4);
            *target_pose = pose_from_center(rotation, centre);
        }
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 0.9;
        let error =
            estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config).unwrap_err();
        assert_eq!(
            error.reason,
            RigSubmapAlignmentRejectionReason::LowRotationConsensus
        );
    }

    #[test]
    fn invalid_boundary_and_rotation_configurations_fail_closed_at_entry() {
        let (source, target, _) = fixture(3);
        let mut invalid_configs = Vec::new();

        let mut config = permissive_config();
        config.max_boundary_frames = 0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = 0.0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = 180.1;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.rotation_consensus_deg = f64::NAN;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 0.0;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = 1.1;
        invalid_configs.push(config);
        let mut config = permissive_config();
        config.min_rotation_consensus_ratio = f64::NAN;
        invalid_configs.push(config);

        for config in invalid_configs {
            let error = estimate_rig_submap_sim3_constraint(0, 1, &source, &target, &config)
                .expect_err("invalid configuration must reject before alignment");
            assert_eq!(
                error.reason,
                RigSubmapAlignmentRejectionReason::InvalidConfiguration
            );
            assert!(error.to_string().contains("invalid"));
        }
    }
}
