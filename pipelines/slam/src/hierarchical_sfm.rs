//! End-to-end assembly of an ordered sequence into a typed submap hierarchy.
//!
//! This module composes the S2 boundaries without weakening any of them:
//! adaptive overlapping partitioning, independent local reconstruction, shared
//! observation + essential-rotation overlap evidence, R2 Sim(3) verification,
//! and the R3 sparse global submap graph.

use std::error::Error;
use std::fmt;

use rayon::prelude::*;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;

use crate::{
    collect_submap_overlap_evidence, estimate_submap_sim3_constraint, partition_ordered_submaps,
    refine_submap_sim3_from_camera_centres, remap_pairs_to_submap, shared_camera_center_matches,
    AdaptiveSubmapPartitionConfig, AdaptiveSubmapPartitionHints, CameraCentreScaleRefinementConfig,
    CameraCentreScaleRefinementRejection, HierarchicalSeamBaConfig, HierarchicalSeamBaError,
    HierarchicalSeamBaResult, HierarchicalSeamLandmarkLink, HierarchicalSubmapGraph,
    HierarchicalSubmapGraphError, HierarchicalSubmapOptimizationResult, LocalSubmap,
    LocalSubmapBuildError, LocalSubmapBuilder, LocalSubmapConfig, PairRotationEvidence,
    PairwiseMatches, Sim3PoseGraphConfig, SubmapOverlapConfig, SubmapOverlapError,
    SubmapPartitionError, SubmapSim3AlignmentConfig, SubmapSim3Rejection, SubmapWindow,
    VerifiedSubmapConstraint,
};

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalSfmConfig {
    pub partition: AdaptiveSubmapPartitionConfig,
    pub local_submap: LocalSubmapConfig,
    pub overlap: SubmapOverlapConfig,
    pub alignment: SubmapSim3AlignmentConfig,
    pub pose_graph: Sim3PoseGraphConfig,
    pub camera_centre_refinement: Option<CameraCentreScaleRefinementConfig>,
    pub seam_bundle_adjustment: Option<HierarchicalSeamBaConfig>,
    /// Maximum independent local reconstructions evaluated concurrently.
    pub max_parallel_local_builds: usize,
}

impl Default for HierarchicalSfmConfig {
    fn default() -> Self {
        Self {
            partition: AdaptiveSubmapPartitionConfig::default(),
            local_submap: LocalSubmapConfig::default(),
            overlap: SubmapOverlapConfig::default(),
            alignment: SubmapSim3AlignmentConfig::default(),
            pose_graph: Sim3PoseGraphConfig::default(),
            camera_centre_refinement: None,
            seam_bundle_adjustment: None,
            max_parallel_local_builds: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalSfmSeam {
    pub source_submap_id: u64,
    pub target_submap_id: u64,
    pub shared_point_matches: usize,
    pub sim3_inliers: usize,
    pub sim3_inlier_ratio: f64,
    pub mean_residual_ratio: f64,
    pub essential_rotation_candidates: usize,
    pub essential_rotation_consensus: usize,
    pub essential_rotation_support: usize,
    pub essential_rotation_max_disagreement_deg: f64,
    pub shared_camera_centres: usize,
    pub camera_sim3_inliers: Option<usize>,
    pub camera_sim3_inlier_ratio: Option<f64>,
    pub camera_mean_residual_ratio: Option<f64>,
    pub camera_landmark_log_scale_disagreement: Option<f64>,
    pub camera_landmark_rotation_disagreement_deg: Option<f64>,
    pub camera_refinement_applied: bool,
    pub camera_refinement_rejection: Option<CameraCentreScaleRefinementRejection>,
    pub camera_refinement_abs_log_scale_change: Option<f64>,
    pub camera_refinement_mean_residual_ratio: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalSfmAtlas {
    pub hierarchy: HierarchicalSubmapGraph,
    pub seams: Vec<HierarchicalSfmSeam>,
    /// `None` for a single-submap sequence, which has no global gauge variables.
    pub optimization: Option<HierarchicalSubmapOptimizationResult>,
    pub seam_bundle_adjustment: Option<HierarchicalSeamBaResult>,
}

#[derive(Debug, Clone)]
pub struct HierarchicalSfmResult {
    pub windows: Vec<SubmapWindow>,
    pub atlas: HierarchicalSfmAtlas,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HierarchicalSfmError {
    SourceFrameCountMismatch {
        ids: usize,
        features: usize,
    },
    Partition(SubmapPartitionError),
    LocalBuild {
        submap_id: u64,
        image_start: usize,
        image_end: usize,
        error: LocalSubmapBuildError,
    },
    Overlap {
        source_submap_id: u64,
        target_submap_id: u64,
        error: SubmapOverlapError,
    },
    Alignment {
        source_submap_id: u64,
        target_submap_id: u64,
        rejection: SubmapSim3Rejection,
    },
    Hierarchy(HierarchicalSubmapGraphError),
    ParallelBuild(String),
    SeamBundleAdjustment(HierarchicalSeamBaError),
    NoSubmaps,
}

impl fmt::Display for HierarchicalSfmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceFrameCountMismatch { ids, features } => {
                write!(f, "source frame id count {ids} != feature count {features}")
            }
            Self::Partition(error) => write!(f, "submap partition failed: {error}"),
            Self::LocalBuild {
                submap_id,
                image_start,
                image_end,
                error,
            } => write!(
                f,
                "submap {submap_id} build failed for images {image_start}..{image_end}: {error}"
            ),
            Self::Overlap {
                source_submap_id,
                target_submap_id,
                error,
            } => write!(
                f,
                "submap seam {source_submap_id}->{target_submap_id} overlap failed: {error}"
            ),
            Self::Alignment {
                source_submap_id,
                target_submap_id,
                rejection,
            } => write!(
                f,
                "submap seam {source_submap_id}->{target_submap_id} rejected: {:?}",
                rejection.reason
            ),
            Self::Hierarchy(error) => write!(f, "hierarchical graph failed: {error}"),
            Self::ParallelBuild(error) => write!(f, "local submap worker pool failed: {error}"),
            Self::SeamBundleAdjustment(error) => {
                write!(f, "seam bundle adjustment failed: {error}")
            }
            Self::NoSubmaps => write!(f, "hierarchical SfM received no submaps"),
        }
    }
}

impl Error for HierarchicalSfmError {}

impl From<SubmapPartitionError> for HierarchicalSfmError {
    fn from(value: SubmapPartitionError) -> Self {
        Self::Partition(value)
    }
}

impl From<HierarchicalSubmapGraphError> for HierarchicalSfmError {
    fn from(value: HierarchicalSubmapGraphError) -> Self {
        Self::Hierarchy(value)
    }
}

impl From<HierarchicalSeamBaError> for HierarchicalSfmError {
    fn from(value: HierarchicalSeamBaError) -> Self {
        Self::SeamBundleAdjustment(value)
    }
}

/// Build and link an ordered image sequence using only verified image evidence.
pub fn hierarchical_sfm(
    camera: &Camera,
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    pair_rotations: &[PairRotationEvidence],
    partition_hints: &AdaptiveSubmapPartitionHints,
    config: &HierarchicalSfmConfig,
) -> Result<HierarchicalSfmResult, HierarchicalSfmError> {
    if source_frame_ids.len() != features.len() {
        return Err(HierarchicalSfmError::SourceFrameCountMismatch {
            ids: source_frame_ids.len(),
            features: features.len(),
        });
    }
    let windows =
        partition_ordered_submaps(features.len(), pairwise, &config.partition, partition_hints)?;
    if windows.is_empty() {
        return Err(HierarchicalSfmError::NoSubmaps);
    }
    let builder = LocalSubmapBuilder::new(config.local_submap.clone());
    let build_one = |(submap_id, window): (usize, &SubmapWindow)| {
        let range = window.image_range.clone();
        let local_pairs = remap_pairs_to_submap(pairwise, range.clone());
        builder
            .build(
                camera,
                &source_frame_ids[range.clone()],
                &features[range.clone()],
                &local_pairs,
            )
            .map_err(|error| HierarchicalSfmError::LocalBuild {
                submap_id: submap_id as u64,
                image_start: range.start,
                image_end: range.end,
                error,
            })
    };
    let worker_count = config.max_parallel_local_builds.max(1).min(windows.len());
    let submaps = if worker_count == 1 {
        windows
            .iter()
            .enumerate()
            .map(build_one)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .build()
            .map_err(|error| HierarchicalSfmError::ParallelBuild(error.to_string()))?;
        pool.install(|| {
            windows
                .par_iter()
                .enumerate()
                .map(build_one)
                .collect::<Result<Vec<_>, _>>()
        })?
    };
    let atlas = optimize_independent_submaps(submaps, pair_rotations, config)?;
    Ok(HierarchicalSfmResult { windows, atlas })
}

/// Link already reconstructed independent submaps. This is also the testable
/// transaction boundary for parallel/local builders: no node transform is
/// committed until every adjacent R2 seam is verified and the R3 solve passes.
pub fn optimize_independent_submaps(
    submaps: Vec<LocalSubmap>,
    pair_rotations: &[PairRotationEvidence],
    config: &HierarchicalSfmConfig,
) -> Result<HierarchicalSfmAtlas, HierarchicalSfmError> {
    if submaps.is_empty() {
        return Err(HierarchicalSfmError::NoSubmaps);
    }
    if submaps.len() == 1 {
        let root = submaps.into_iter().next().expect("length checked above");
        return Ok(HierarchicalSfmAtlas {
            hierarchy: HierarchicalSubmapGraph::new(0, root),
            seams: Vec::new(),
            optimization: None,
            seam_bundle_adjustment: None,
        });
    }

    let mut constraints = Vec::with_capacity(submaps.len() - 1);
    let mut seam_links = Vec::new();
    let mut seams = Vec::with_capacity(submaps.len() - 1);
    for index in 0..submaps.len() - 1 {
        let source_id = index as u64;
        let target_id = source_id + 1;
        let overlap = collect_submap_overlap_evidence(
            &submaps[index],
            &submaps[index + 1],
            pair_rotations,
            &config.overlap,
        )
        .map_err(|error| HierarchicalSfmError::Overlap {
            source_submap_id: source_id,
            target_submap_id: target_id,
            error,
        })?;
        let landmark_constraint = estimate_submap_sim3_constraint(
            source_id,
            target_id,
            &overlap.point_matches,
            &overlap.target_from_source_rotation,
            &config.alignment,
        )
        .map_err(|rejection| HierarchicalSfmError::Alignment {
            source_submap_id: source_id,
            target_submap_id: target_id,
            rejection,
        })?;
        let camera_matches = shared_camera_center_matches(&submaps[index], &submaps[index + 1]);
        let camera_constraint = estimate_submap_sim3_constraint(
            source_id,
            target_id,
            &camera_matches,
            &overlap.target_from_source_rotation,
            &config.alignment,
        )
        .ok();
        let camera_refinement = config.camera_centre_refinement.as_ref().map(|refinement| {
            refine_submap_sim3_from_camera_centres(
                &landmark_constraint,
                &overlap.point_matches,
                &camera_matches,
                &config.alignment,
                refinement,
            )
        });
        let (constraint, refinement_rejection, refinement_scale_change, refinement_residual) =
            match camera_refinement {
                Some(Ok(refined)) => (
                    refined.constraint,
                    None,
                    Some(refined.abs_log_scale_change),
                    Some(refined.mean_camera_residual_ratio),
                ),
                Some(Err(rejection)) => (landmark_constraint.clone(), Some(rejection), None, None),
                None => (landmark_constraint.clone(), None, None, None),
            };
        seams.push(HierarchicalSfmSeam {
            source_submap_id: source_id,
            target_submap_id: target_id,
            shared_point_matches: overlap.point_matches.len(),
            sim3_inliers: constraint.inlier_match_indices.len(),
            sim3_inlier_ratio: constraint.inlier_ratio,
            mean_residual_ratio: constraint.mean_residual_ratio,
            essential_rotation_candidates: overlap.rotation_candidate_count,
            essential_rotation_consensus: overlap.rotation_consensus_count,
            essential_rotation_support: overlap.rotation_consensus_inlier_support,
            essential_rotation_max_disagreement_deg: overlap.max_rotation_disagreement_deg,
            shared_camera_centres: camera_matches.len(),
            camera_sim3_inliers: camera_constraint
                .as_ref()
                .map(|candidate| candidate.inlier_match_indices.len()),
            camera_sim3_inlier_ratio: camera_constraint
                .as_ref()
                .map(|candidate| candidate.inlier_ratio),
            camera_mean_residual_ratio: camera_constraint
                .as_ref()
                .map(|candidate| candidate.mean_residual_ratio),
            camera_landmark_log_scale_disagreement: camera_constraint.as_ref().map(|candidate| {
                (candidate.target_from_source.scale / landmark_constraint.target_from_source.scale)
                    .ln()
                    .abs()
            }),
            camera_landmark_rotation_disagreement_deg: camera_constraint.as_ref().map(
                |candidate| {
                    candidate
                        .target_from_source
                        .rotation
                        .rotation_to(&landmark_constraint.target_from_source.rotation)
                        .angle()
                        .to_degrees()
                },
            ),
            camera_refinement_applied: refinement_scale_change.is_some(),
            camera_refinement_rejection: refinement_rejection,
            camera_refinement_abs_log_scale_change: refinement_scale_change,
            camera_refinement_mean_residual_ratio: refinement_residual,
        });
        for &match_index in &constraint.inlier_match_indices {
            let point_match = &overlap.point_matches[match_index];
            seam_links.push(HierarchicalSeamLandmarkLink {
                source_submap_id: source_id,
                target_submap_id: target_id,
                source_landmark_id: point_match.source_landmark_id,
                target_landmark_id: point_match.target_landmark_id,
            });
        }
        constraints.push(constraint);
    }

    let mut submaps = submaps.into_iter();
    let root = submaps.next().expect("non-empty checked above");
    let mut hierarchy = HierarchicalSubmapGraph::new(0, root);
    for (offset, submap) in submaps.enumerate() {
        hierarchy.insert_independent(offset as u64 + 1, submap)?;
    }
    for constraint in constraints {
        hierarchy.add_constraint(VerifiedSubmapConstraint::Sim3(constraint))?;
    }
    let optimization = hierarchy.optimize(&config.pose_graph)?;
    let seam_bundle_adjustment = config
        .seam_bundle_adjustment
        .as_ref()
        .map(|ba_config| {
            crate::hierarchical_seam_ba::refine_hierarchical_seams(
                &mut hierarchy,
                &seam_links,
                ba_config,
            )
        })
        .transpose()?;
    Ok(HierarchicalSfmAtlas {
        hierarchy,
        seams,
        optimization: Some(optimization),
        seam_bundle_adjustment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::{Pose, Sim3};

    use crate::{
        LocalSubmapFrame, LocalSubmapLandmark, LocalSubmapObservation, LocalSubmapQuality,
        TrackBuildStats,
    };

    fn local_submap(
        frame_id: u64,
        rotation: UnitQuaternion<f64>,
        landmarks: Vec<LocalSubmapLandmark>,
    ) -> LocalSubmap {
        LocalSubmap {
            camera: Camera::pinhole(0, 64, 48, 50.0, 50.0, 32.0, 24.0),
            source_frame_ids: vec![frame_id],
            frames: vec![LocalSubmapFrame {
                local_frame_index: 0,
                source_frame_id: frame_id,
                pose: Pose::from_world_to_camera(rotation, Vector3::zeros()),
            }],
            landmarks,
            quality: LocalSubmapQuality {
                requested_images: 1,
                registered_images: 1,
                registration_fraction: 1.0,
                landmarks: 15,
                observations: 15,
                median_track_length: 1.0,
                median_max_parallax_deg: 5.0,
                camera_center_diameter: 0.0,
                mean_reprojection_px: 0.0,
            },
            track_build_stats: TrackBuildStats::default(),
            ba_result: None,
        }
    }

    fn point_landmark(id: u64, point: Point3<f64>, shared_frame: u64) -> LocalSubmapLandmark {
        LocalSubmapLandmark {
            local_landmark_id: id,
            position: point,
            observations: vec![LocalSubmapObservation {
                local_frame_index: 0,
                source_frame_id: shared_frame,
                keypoint_index: id as usize % 100,
                pixel: Point2::new(id as f64, 0.0),
            }],
        }
    }

    #[test]
    fn links_independent_submaps_through_r2_and_r3_transactionally() {
        let truth = Sim3::new(
            UnitQuaternion::from_euler_angles(0.08, -0.12, 0.21),
            Vector3::new(1.2, -0.4, 0.8),
            2.5,
        );
        let source_rotation = UnitQuaternion::from_euler_angles(-0.1, 0.05, 0.2);
        let target_rotation = UnitQuaternion::from_euler_angles(0.2, -0.08, -0.1);
        let source_points = (0..15)
            .map(|index| {
                let x = (index % 5) as f64 * 0.4;
                let y = (index / 5) as f64 * 0.35;
                let z = ((index * 7) % 4) as f64 * 0.2;
                point_landmark(index, Point3::new(x, y, z), 50)
            })
            .collect::<Vec<_>>();
        let target_points = source_points
            .iter()
            .map(|landmark| {
                let mut transformed = point_landmark(
                    landmark.local_landmark_id,
                    truth.transform_point(&landmark.position),
                    50,
                );
                transformed.local_landmark_id += 100;
                transformed.observations[0].keypoint_index = landmark.local_landmark_id as usize;
                transformed
            })
            .collect();
        let camera_j_from_i = target_rotation * truth.rotation * source_rotation.inverse();
        let atlas = optimize_independent_submaps(
            vec![
                local_submap(10, source_rotation, source_points),
                local_submap(90, target_rotation, target_points),
            ],
            &[PairRotationEvidence {
                image_i: 10,
                image_j: 90,
                image_j_from_i: camera_j_from_i,
                inlier_count: 100,
            }],
            &HierarchicalSfmConfig::default(),
        )
        .unwrap();
        assert_eq!(atlas.seams.len(), 1);
        assert_eq!(atlas.seams[0].sim3_inliers, 15);
        let recovered = &atlas.hierarchy.node(1).unwrap().local_from_atlas;
        let disagreement = recovered
            .as_ref()
            .unwrap()
            .compose(&truth.inverse())
            .log()
            .norm();
        assert!(disagreement < 1e-8, "Sim3 disagreement {disagreement}");
    }

    #[test]
    fn refuses_to_link_without_independent_rotation_evidence() {
        let empty = local_submap(0, UnitQuaternion::identity(), Vec::new());
        let error = optimize_independent_submaps(
            vec![empty.clone(), empty],
            &[],
            &HierarchicalSfmConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            HierarchicalSfmError::Overlap {
                error: SubmapOverlapError::NoRotationCandidates,
                ..
            }
        ));
    }

    #[test]
    fn single_submap_needs_no_global_constraint() {
        let only = local_submap(0, UnitQuaternion::identity(), Vec::new());
        let atlas =
            optimize_independent_submaps(vec![only], &[], &HierarchicalSfmConfig::default())
                .unwrap();
        assert_eq!(atlas.hierarchy.nodes().count(), 1);
        assert!(atlas.optimization.is_none());
        assert!(atlas.seams.is_empty());
    }
}
