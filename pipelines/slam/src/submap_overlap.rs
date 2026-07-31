//! Auditable overlap evidence between independently reconstructed submaps.
//!
//! Shared `(source frame, keypoint)` identities vote for one-to-one landmark
//! matches. Essential-matrix rotations remain separate evidence: they are
//! converted from camera-to-camera coordinates into the two local submap gauges
//! and consensus-filtered before R2's 3D alignment sees them.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use nalgebra::{Point3, UnitQuaternion};

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

/// Which side of a seam [`seam_step_shape_diagnostic`] identifies as the
/// more internally inconsistent one -- purely evidential (both the merge
/// remediation and the trigger decision itself only need
/// [`SeamStepShapeDiagnostic::disagreement_ratio`]; the merge always widens
/// both windows into one regardless of which side "caused" it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SeamDriftSide {
    Source,
    Target,
}

/// Cross-submap internal-consistency comparison for one seam
/// (`LOWINLIERRATIO_DIAGNOSIS.md` Probe 2, formalized). `source_change_factor`
/// / `target_change_factor` are each that side's own
/// [`crate::local_submap::windowed_camera_center_drift_ratio`] computed over
/// *only* the frames the two submaps share -- how much that side's own step
/// size, self-compared across the shared span, varies. `disagreement_ratio`
/// is the ratio between the two: close to `1.0` means both sides agree on
/// the shared span's motion shape (whether or not the underlying real
/// motion is itself fast or slow, uniform or accelerating -- genuine motion
/// is common to both independent reconstructions of the same frames and
/// cancels out of this comparison), far from `1.0` means one side's own
/// reconstruction disagrees with an otherwise-independent, otherwise-equally
/// -valid account of the same real motion, i.e. an internal defect specific
/// to that one side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SeamStepShapeDiagnostic {
    pub shared_frames: usize,
    pub source_change_factor: f64,
    pub target_change_factor: f64,
    pub disagreement_ratio: f64,
    pub defective_side: SeamDriftSide,
}

/// Compute [`SeamStepShapeDiagnostic`] for one seam, or `None` if there are
/// too few shared frames (fewer than `2 * window_count`, the minimum needed
/// for each side to form `window_count` non-empty windows) to say anything.
pub(crate) fn seam_step_shape_diagnostic(
    source: &LocalSubmap,
    target: &LocalSubmap,
    window_count: usize,
) -> Option<SeamStepShapeDiagnostic> {
    let camera_matches = shared_camera_center_matches(source, target);
    // `shared_camera_center_matches` carries the shared `source_frame_id` on
    // both id fields (see that function's doc); `BTreeMap` sorts by frame id
    // for free, giving temporal order directly.
    let by_frame: BTreeMap<u64, (Point3<f64>, Point3<f64>)> = camera_matches
        .iter()
        .map(|point_match| {
            (
                point_match.source_landmark_id,
                (point_match.source_point, point_match.target_point),
            )
        })
        .collect();
    let frames: Vec<u64> = by_frame.keys().copied().collect();
    if frames.len() < 2 * window_count.max(1) {
        return None;
    }
    let source_steps: Vec<f64> = frames
        .windows(2)
        .map(|pair| {
            let (a, _) = by_frame[&pair[0]];
            let (b, _) = by_frame[&pair[1]];
            (b - a).norm()
        })
        .collect();
    let target_steps: Vec<f64> = frames
        .windows(2)
        .map(|pair| {
            let (_, a) = by_frame[&pair[0]];
            let (_, b) = by_frame[&pair[1]];
            (b - a).norm()
        })
        .collect();
    let source_change_factor =
        crate::local_submap::windowed_camera_center_drift_ratio(&source_steps, window_count);
    let target_change_factor =
        crate::local_submap::windowed_camera_center_drift_ratio(&target_steps, window_count);
    if source_change_factor <= 1.0e-9 && target_change_factor <= 1.0e-9 {
        return None;
    }
    let (defective_side, max_cf, min_cf) = if source_change_factor >= target_change_factor {
        (
            SeamDriftSide::Source,
            source_change_factor,
            target_change_factor,
        )
    } else {
        (
            SeamDriftSide::Target,
            target_change_factor,
            source_change_factor,
        )
    };
    let disagreement_ratio = if min_cf > 1.0e-9 {
        max_cf / min_cf
    } else if max_cf > 1.0e-9 {
        f64::INFINITY
    } else {
        1.0
    };
    Some(SeamStepShapeDiagnostic {
        shared_frames: frames.len(),
        source_change_factor,
        target_change_factor,
        disagreement_ratio,
        defective_side,
    })
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
                camera_center_step_median: 0.0,
                camera_center_step_max: 0.0,
                seed_pair_final_distance: 1.0,
                camera_center_window_drift_ratio: 0.0,
                mean_reprojection_px: 0.0,
                leave_one_out_attempts: 0,
                leave_one_out_supported: 0,
                leave_one_out_support_fraction: 0.0,
                median_leave_one_out_reprojection_px: 0.0,
            },
            track_build_stats: TrackBuildStats::default(),
            ba_result: None,
            seed_source_frame_i: 0,
            seed_source_frame_j: 0,
            seed_match_count: 0,
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

    /// `n + 1` frames sharing ids `0..=n`, with camera centres along `x` at
    /// the cumulative sum of `steps` (so `steps.len() == n`), scaled by
    /// `gauge_scale` -- an independent reconstruction's absolute gauge is
    /// arbitrary, so tests exercise that this cancels out of the comparison.
    fn stepped_submap(steps: &[f64], gauge_scale: f64) -> LocalSubmap {
        let mut x = 0.0_f64;
        let mut frames = vec![LocalSubmapFrame {
            local_frame_index: 0,
            source_frame_id: 0,
            pose: Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
        }];
        for (index, step) in steps.iter().enumerate() {
            x += step * gauge_scale;
            frames.push(LocalSubmapFrame {
                local_frame_index: index + 1,
                source_frame_id: (index + 1) as u64,
                pose: Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(x, 0.0, 0.0),
                ),
            });
        }
        submap(frames, Vec::new())
    }

    #[test]
    fn seam_step_shape_diagnostic_stays_quiet_when_both_sides_agree_on_shape() {
        // Both sides see the same *relative* shape (steps 0.1 for the first
        // half, 1.0 for the second -- a 10x internal change, mimicking a
        // genuinely fast-accelerating shared span) but at unrelated absolute
        // gauges (source: raw; target: 3x). A real, common-mode motion trend
        // must cancel out of the comparison regardless of the two
        // independent reconstructions' arbitrary monocular scales.
        let mut steps = vec![0.1_f64; 6];
        steps.extend(vec![1.0_f64; 6]);
        let source = stepped_submap(&steps, 1.0);
        let target = stepped_submap(&steps, 3.0);
        let diagnostic = seam_step_shape_diagnostic(&source, &target, 2)
            .expect("12 shared steps is enough for a 2-window split");
        assert_eq!(diagnostic.shared_frames, 13);
        assert!((diagnostic.source_change_factor - 10.0).abs() < 1.0e-9);
        assert!((diagnostic.target_change_factor - 10.0).abs() < 1.0e-9);
        assert!(
            (diagnostic.disagreement_ratio - 1.0).abs() < 1.0e-9,
            "matching shapes at different gauges must not disagree, got {}",
            diagnostic.disagreement_ratio
        );
    }

    #[test]
    fn seam_step_shape_diagnostic_fires_when_one_side_diverges() {
        // Source is flat (no internal drift); target shows the same 10x
        // within-window growth as the diagnosed submap 9/13 defect. Only
        // target disagrees with the (shared, real) motion the flat source
        // faithfully reports.
        let flat_steps = vec![0.5_f64; 12];
        let mut drifting_steps = vec![0.1_f64; 6];
        drifting_steps.extend(vec![1.0_f64; 6]);
        let source = stepped_submap(&flat_steps, 1.0);
        let target = stepped_submap(&drifting_steps, 1.0);
        let diagnostic = seam_step_shape_diagnostic(&source, &target, 2).unwrap();
        assert!((diagnostic.source_change_factor - 1.0).abs() < 1.0e-9);
        assert!((diagnostic.target_change_factor - 10.0).abs() < 1.0e-9);
        assert!((diagnostic.disagreement_ratio - 10.0).abs() < 1.0e-9);
        assert_eq!(diagnostic.defective_side, SeamDriftSide::Target);
    }

    #[test]
    fn seam_step_shape_diagnostic_none_when_too_few_shared_frames() {
        // A single shared step (2 frames) is fewer than the 2 windows
        // requested -- `windowed_camera_center_drift_ratio` reports `0.0`
        // (nothing to compare) on both sides, so the diagnostic itself is
        // `None` rather than a meaningless `1.0`.
        let steps = vec![0.1_f64];
        let source = stepped_submap(&steps, 1.0);
        let target = stepped_submap(&steps, 1.0);
        assert_eq!(seam_step_shape_diagnostic(&source, &target, 2), None);
    }

    #[test]
    fn seam_step_shape_diagnostic_none_when_no_shared_frames() {
        let source = stepped_submap(&[0.1; 6], 1.0);
        let target = submap(
            vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: 900,
                pose: Pose::identity(),
            }],
            Vec::new(),
        );
        assert_eq!(seam_step_shape_diagnostic(&source, &target, 2), None);
    }
}
