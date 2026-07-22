//! Auditable overlap evidence between independently reconstructed submaps.
//!
//! Shared `(source frame, keypoint)` identities vote for one-to-one landmark
//! matches. Essential-matrix rotations remain separate evidence: they are
//! converted from camera-to-camera coordinates into the two local submap gauges
//! and consensus-filtered before R2's 3D alignment sees them.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use nalgebra::UnitQuaternion;

use crate::{LocalSubmap, SubmapPointMatch};

/// Essential-verified relative rotation for one ordered image pair.
#[derive(Debug, Clone, PartialEq)]
pub struct PairRotationEvidence {
    pub image_i: u64,
    pub image_j: u64,
    /// Rotation from camera `i` coordinates into camera `j` coordinates.
    pub image_j_from_i: UnitQuaternion<f64>,
    pub inlier_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmapOverlapConfig {
    /// Shared observations required before two local landmarks can correspond.
    pub min_shared_observations_per_landmark: usize,
    /// Minimum agreeing camera-pair rotation candidates.
    pub min_rotation_consensus_candidates: usize,
    /// Angular radius of the essential-rotation consensus cluster.
    pub max_rotation_disagreement_deg: f64,
}

impl Default for SubmapOverlapConfig {
    fn default() -> Self {
        Self {
            min_shared_observations_per_landmark: 1,
            min_rotation_consensus_candidates: 1,
            max_rotation_disagreement_deg: 5.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmapOverlapEvidence {
    pub point_matches: Vec<SubmapPointMatch>,
    /// Essential-derived rotation from source-local into target-local gauge.
    pub target_from_source_rotation: UnitQuaternion<f64>,
    pub rotation_candidate_count: usize,
    pub rotation_consensus_count: usize,
    pub rotation_consensus_inlier_support: usize,
    pub max_rotation_disagreement_deg: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmapOverlapError {
    NoRotationCandidates,
    InsufficientRotationConsensus { required: usize, found: usize },
}

impl fmt::Display for SubmapOverlapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRotationCandidates => write!(
                f,
                "no essential pair connects a registered source frame to a registered target frame"
            ),
            Self::InsufficientRotationConsensus { required, found } => write!(
                f,
                "essential rotation consensus has {found} candidates; {required} required"
            ),
        }
    }
}

impl Error for SubmapOverlapError {}

pub fn collect_submap_overlap_evidence(
    source: &LocalSubmap,
    target: &LocalSubmap,
    pair_rotations: &[PairRotationEvidence],
    config: &SubmapOverlapConfig,
) -> Result<SubmapOverlapEvidence, SubmapOverlapError> {
    let point_matches = shared_landmark_point_matches(
        source,
        target,
        config.min_shared_observations_per_landmark.max(1),
    );
    let candidates = gauge_rotation_candidates(source, target, pair_rotations);
    if candidates.is_empty() {
        return Err(SubmapOverlapError::NoRotationCandidates);
    }
    let max_angle = config.max_rotation_disagreement_deg.max(0.0).to_radians();
    let (winner, consensus) = strongest_rotation_consensus(&candidates, max_angle);
    let required = config.min_rotation_consensus_candidates.max(1);
    if consensus.len() < required {
        return Err(SubmapOverlapError::InsufficientRotationConsensus {
            required,
            found: consensus.len(),
        });
    }
    let target_from_source_rotation = candidates[winner].0;
    let rotation_consensus_inlier_support =
        consensus.iter().map(|&index| candidates[index].1).sum();
    let max_rotation_disagreement_deg = consensus
        .iter()
        .map(|&index| {
            target_from_source_rotation
                .rotation_to(&candidates[index].0)
                .angle()
                .to_degrees()
        })
        .fold(0.0_f64, f64::max);
    Ok(SubmapOverlapEvidence {
        point_matches,
        target_from_source_rotation,
        rotation_candidate_count: candidates.len(),
        rotation_consensus_count: consensus.len(),
        rotation_consensus_inlier_support,
        max_rotation_disagreement_deg,
    })
}

/// Build deterministic mutual-best one-to-one 3D matches from shared image
/// observations. Ambiguous observation identities do not vote.
pub fn shared_landmark_point_matches(
    source: &LocalSubmap,
    target: &LocalSubmap,
    min_shared_observations: usize,
) -> Vec<SubmapPointMatch> {
    let mut target_observation = BTreeMap::<(u64, usize), Option<u64>>::new();
    for landmark in &target.landmarks {
        for observation in &landmark.observations {
            let key = (observation.source_frame_id, observation.keypoint_index);
            target_observation
                .entry(key)
                .and_modify(|entry| *entry = None)
                .or_insert(Some(landmark.local_landmark_id));
        }
    }

    let mut votes = BTreeMap::<(u64, u64), usize>::new();
    for landmark in &source.landmarks {
        for observation in &landmark.observations {
            let key = (observation.source_frame_id, observation.keypoint_index);
            if let Some(Some(target_id)) = target_observation.get(&key) {
                *votes
                    .entry((landmark.local_landmark_id, *target_id))
                    .or_default() += 1;
            }
        }
    }

    let threshold = min_shared_observations.max(1);
    let mut best_by_source = BTreeMap::<u64, (u64, usize)>::new();
    let mut best_by_target = BTreeMap::<u64, (u64, usize)>::new();
    for (&(source_id, target_id), &count) in &votes {
        if count < threshold {
            continue;
        }
        update_best(&mut best_by_source, source_id, target_id, count);
        update_best(&mut best_by_target, target_id, source_id, count);
    }
    let source_points = source
        .landmarks
        .iter()
        .map(|landmark| (landmark.local_landmark_id, landmark.position))
        .collect::<BTreeMap<_, _>>();
    let target_points = target
        .landmarks
        .iter()
        .map(|landmark| (landmark.local_landmark_id, landmark.position))
        .collect::<BTreeMap<_, _>>();
    best_by_source
        .into_iter()
        .filter_map(|(source_id, (target_id, _))| {
            let (mutual_source, _) = best_by_target.get(&target_id)?;
            if *mutual_source != source_id {
                return None;
            }
            Some(SubmapPointMatch {
                source_landmark_id: source_id,
                target_landmark_id: target_id,
                source_point: *source_points.get(&source_id)?,
                target_point: *target_points.get(&target_id)?,
            })
        })
        .collect()
}

/// Corresponding camera centres for frames registered independently in both
/// submaps. They stay separate from landmark matches so trajectory consistency
/// can be audited without silently changing R2's landmark weighting.
pub fn shared_camera_center_matches(
    source: &LocalSubmap,
    target: &LocalSubmap,
) -> Vec<SubmapPointMatch> {
    let target_centres = target
        .frames
        .iter()
        .map(|frame| (frame.source_frame_id, frame.pose.camera_center_world()))
        .collect::<BTreeMap<_, _>>();
    source
        .frames
        .iter()
        .filter_map(|frame| {
            let target_point = target_centres.get(&frame.source_frame_id)?;
            Some(SubmapPointMatch {
                source_landmark_id: frame.source_frame_id,
                target_landmark_id: frame.source_frame_id,
                source_point: frame.pose.camera_center_world(),
                target_point: *target_point,
            })
        })
        .collect()
}

fn update_best(best: &mut BTreeMap<u64, (u64, usize)>, key: u64, candidate: u64, count: usize) {
    let replace = best.get(&key).is_none_or(|&(current, current_count)| {
        count > current_count || (count == current_count && candidate < current)
    });
    if replace {
        best.insert(key, (candidate, count));
    }
}

fn gauge_rotation_candidates(
    source: &LocalSubmap,
    target: &LocalSubmap,
    evidence: &[PairRotationEvidence],
) -> Vec<(UnitQuaternion<f64>, usize)> {
    let source_frames = source
        .frames
        .iter()
        .map(|frame| (frame.source_frame_id, &frame.pose))
        .collect::<BTreeMap<_, _>>();
    let target_frames = target
        .frames
        .iter()
        .map(|frame| (frame.source_frame_id, &frame.pose))
        .collect::<BTreeMap<_, _>>();
    let mut candidates = Vec::new();
    for pair in evidence {
        if pair.image_i == pair.image_j {
            continue;
        }
        if let (Some(source_i), Some(target_j)) = (
            source_frames.get(&pair.image_i),
            target_frames.get(&pair.image_j),
        ) {
            candidates.push((
                target_j.world_to_camera.rotation.inverse()
                    * pair.image_j_from_i
                    * source_i.world_to_camera.rotation,
                pair.inlier_count,
            ));
        }
        if let (Some(source_j), Some(target_i)) = (
            source_frames.get(&pair.image_j),
            target_frames.get(&pair.image_i),
        ) {
            candidates.push((
                target_i.world_to_camera.rotation.inverse()
                    * pair.image_j_from_i.inverse()
                    * source_j.world_to_camera.rotation,
                pair.inlier_count,
            ));
        }
    }
    candidates
}

fn strongest_rotation_consensus(
    candidates: &[(UnitQuaternion<f64>, usize)],
    max_angle: f64,
) -> (usize, Vec<usize>) {
    let mut best_seed = 0;
    let mut best_inliers = Vec::new();
    let mut best_support = 0;
    let mut best_residual = f64::INFINITY;
    for (seed, (rotation, _)) in candidates.iter().enumerate() {
        let inliers = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, (candidate, _))| {
                (rotation.rotation_to(candidate).angle() <= max_angle).then_some(index)
            })
            .collect::<Vec<_>>();
        let support = inliers.iter().map(|&index| candidates[index].1).sum();
        let residual = inliers
            .iter()
            .map(|&index| {
                rotation.rotation_to(&candidates[index].0).angle()
                    * candidates[index].1.max(1) as f64
            })
            .sum::<f64>();
        if support > best_support
            || (support == best_support && residual < best_residual)
            || (support == best_support && residual == best_residual && seed < best_seed)
        {
            best_seed = seed;
            best_inliers = inliers;
            best_support = support;
            best_residual = residual;
        }
    }
    (best_seed, best_inliers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, Vector3};
    use visloc_core::geometry::{Pose, Sim3};
    use visloc_core::types::Camera;

    use crate::{
        LocalSubmapFrame, LocalSubmapLandmark, LocalSubmapObservation, LocalSubmapQuality,
        TrackBuildStats,
    };

    fn submap(frames: Vec<LocalSubmapFrame>, landmarks: Vec<LocalSubmapLandmark>) -> LocalSubmap {
        LocalSubmap {
            camera: Camera::pinhole(0, 64, 48, 50.0, 50.0, 32.0, 24.0),
            source_frame_ids: frames.iter().map(|frame| frame.source_frame_id).collect(),
            frames,
            landmarks,
            quality: LocalSubmapQuality {
                requested_images: 0,
                registered_images: 0,
                registration_fraction: 0.0,
                landmarks: 0,
                observations: 0,
                median_track_length: 0.0,
                median_max_parallax_deg: 0.0,
                camera_center_diameter: 0.0,
                mean_reprojection_px: 0.0,
            },
            track_build_stats: TrackBuildStats::default(),
            ba_result: None,
        }
    }

    fn landmark(id: u64, point: Point3<f64>, frame: u64, keypoint: usize) -> LocalSubmapLandmark {
        LocalSubmapLandmark {
            local_landmark_id: id,
            position: point,
            observations: vec![LocalSubmapObservation {
                local_frame_index: 0,
                source_frame_id: frame,
                keypoint_index: keypoint,
                pixel: Point2::new(keypoint as f64, 1.0),
            }],
        }
    }

    #[test]
    fn recovers_shared_points_and_essential_rotation_in_local_gauges() {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(2.0, -1.0, 0.5),
            1.7,
        );
        let source_camera_rotation = UnitQuaternion::from_euler_angles(-0.2, 0.1, 0.05);
        let target_camera_rotation = UnitQuaternion::from_euler_angles(0.15, 0.08, -0.12);
        let source = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 100,
                pose: Pose::from_world_to_camera(source_camera_rotation, Vector3::zeros()),
            }],
            (0..12)
                .map(|index| {
                    landmark(
                        index,
                        Point3::new(index as f64, (index % 3) as f64, 0.2 * index as f64),
                        150,
                        index as usize,
                    )
                })
                .collect(),
        );
        let target = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 200,
                pose: Pose::from_world_to_camera(target_camera_rotation, Vector3::zeros()),
            }],
            source
                .landmarks
                .iter()
                .map(|source_landmark| {
                    landmark(
                        source_landmark.local_landmark_id + 100,
                        truth.transform_point(&source_landmark.position),
                        150,
                        source_landmark.local_landmark_id as usize,
                    )
                })
                .collect(),
        );
        let camera_j_from_i =
            target_camera_rotation * truth.rotation * source_camera_rotation.inverse();
        let overlap = collect_submap_overlap_evidence(
            &source,
            &target,
            &[PairRotationEvidence {
                image_i: 100,
                image_j: 200,
                image_j_from_i: camera_j_from_i,
                inlier_count: 80,
            }],
            &SubmapOverlapConfig::default(),
        )
        .unwrap();
        assert_eq!(overlap.point_matches.len(), 12);
        assert!(
            overlap
                .target_from_source_rotation
                .rotation_to(&truth.rotation)
                .angle()
                < 1e-12
        );
        assert_eq!(overlap.rotation_consensus_inlier_support, 80);
    }

    #[test]
    fn mutual_best_filter_keeps_landmark_matches_one_to_one() {
        let source = submap(
            Vec::new(),
            vec![
                landmark(0, Point3::origin(), 10, 0),
                landmark(1, Point3::new(1.0, 0.0, 0.0), 10, 1),
            ],
        );
        let mut target_landmark = landmark(10, Point3::origin(), 10, 0);
        target_landmark.observations.push(LocalSubmapObservation {
            local_frame_index: 0,
            source_frame_id: 10,
            keypoint_index: 1,
            pixel: Point2::origin(),
        });
        let target = submap(Vec::new(), vec![target_landmark]);
        let matches = shared_landmark_point_matches(&source, &target, 1);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source_landmark_id, 0);
    }

    #[test]
    fn rejects_when_rotation_consensus_requirement_is_not_met() {
        let source = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 1,
                pose: Pose::identity(),
            }],
            Vec::new(),
        );
        let target = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 2,
                pose: Pose::identity(),
            }],
            Vec::new(),
        );
        let error = collect_submap_overlap_evidence(
            &source,
            &target,
            &[PairRotationEvidence {
                image_i: 1,
                image_j: 2,
                image_j_from_i: UnitQuaternion::identity(),
                inlier_count: 20,
            }],
            &SubmapOverlapConfig {
                min_rotation_consensus_candidates: 2,
                ..SubmapOverlapConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            SubmapOverlapError::InsufficientRotationConsensus {
                required: 2,
                found: 1
            }
        );
    }

    #[test]
    fn shared_camera_centres_keep_source_frame_identity() {
        let source = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 7,
                pose: Pose::identity(),
            }],
            Vec::new(),
        );
        let target = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 7,
                pose: Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-1.0, 0.0, 0.0),
                ),
            }],
            Vec::new(),
        );
        let matches = shared_camera_center_matches(&source, &target);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].source_landmark_id, 7);
        assert_eq!(matches[0].source_point, Point3::origin());
        assert_eq!(matches[0].target_point, Point3::new(1.0, 0.0, 0.0));
    }
}
