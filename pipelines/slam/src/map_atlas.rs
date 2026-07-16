//! Independent visual-map ownership and verified metric SE(3) map welding.
//!
//! A [`MapAtlas`] keeps every submap in its own local world frame. Geometry is
//! transformed only when a verified cross-submap edge aligns that submap to an
//! already-aligned target. [`MapAtlas::materialize_aligned`] then creates a
//! collision-free flat [`VisualMap`] for existing consumers; the source maps
//! remain untouched.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

use nalgebra::{Matrix3, Point3, UnitQuaternion};
use visloc_core::geometry::{Pose, SE3};
use visloc_core::types::{
    Camera, CameraId, FrameId, Keyframe, Landmark, LandmarkDescriptorStore, LandmarkId,
    Observation, QueryImage, StereoObservation, VisualMap,
};
use visloc_localization::{CandidateSelector, LocalizationPipeline};
use visloc_vision::matching::Matcher;
use visloc_vision::ransac::RobustPoseEstimator;

use crate::online_slam::{relocalization_descriptor_cosine, relocalization_mean_descriptor};

pub type SubmapId = u64;

#[derive(Debug, Clone, PartialEq)]
pub struct AtlasSubmap {
    pub id: SubmapId,
    pub map: VisualMap,
    /// `T_atlas<-local`; `None` means this submap is still independent.
    pub atlas_from_local: Option<SE3>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubmapMergeEvidence {
    pub source_submap_id: SubmapId,
    pub target_submap_id: SubmapId,
    /// Metric rigid transform `T_target<-source` recovered from cross-submap
    /// geometry. Scale is reported separately and must pass the metric gate.
    pub target_from_source: SE3,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error_px: f64,
    pub estimated_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubmapMergeQuality {
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub mean_reprojection_error_px: f64,
    pub estimated_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubmapMergeVerificationConfig {
    pub min_inliers: usize,
    pub min_inlier_ratio: f64,
    pub max_mean_reprojection_error_px: f64,
    pub max_metric_scale_error: f64,
    /// When both maps are already aligned, bound disagreement with the
    /// existing Atlas transform rather than silently changing its gauge.
    pub max_existing_translation_disagreement_m: f64,
    pub max_existing_rotation_disagreement_rad: f64,
}

impl Default for SubmapMergeVerificationConfig {
    fn default() -> Self {
        Self {
            min_inliers: 30,
            min_inlier_ratio: 0.3,
            max_mean_reprojection_error_px: 3.0,
            max_metric_scale_error: 0.1,
            max_existing_translation_disagreement_m: 0.5,
            max_existing_rotation_disagreement_rad: 5.0_f64.to_radians(),
        }
    }
}

/// Appearance retrieval and independent metric-scale checks applied before a
/// cross-submap PnP bridge is admitted to the Atlas.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSubmapAlignmentConfig {
    /// Maximum aligned target submaps on which to run descriptor matching and
    /// PnP. Each submap is represented in retrieval by its best keyframe.
    pub max_target_submaps: usize,
    /// Minimum cosine similarity between mean frame descriptors.
    pub min_appearance_similarity: f32,
    /// Minimum candidate-keyframe landmark descriptors required for PnP.
    pub min_candidate_landmarks: usize,
    /// Bound the quadratic pair-distance scale estimator.
    pub max_scale_points: usize,
    /// Ignore 3D pairs with a shorter baseline in either submap.
    pub min_scale_pair_distance_m: f64,
    /// Minimum number of pair-distance ratios supporting the scale estimate.
    pub min_scale_pair_count: usize,
    /// Relative distance from the median scale counted as scale consensus.
    pub scale_inlier_relative_tolerance: f64,
    /// Minimum fraction of distance ratios inside the scale consensus band.
    pub min_scale_inlier_ratio: f64,
    /// Maximum median absolute deviation of the inlier scale ratios.
    pub max_scale_mad: f64,
    /// Maximum transformed 3D distance for retaining a verified landmark
    /// equivalence used during map welding.
    pub max_landmark_match_residual_m: f64,
    /// Minimum retained source/target landmark equivalences required before
    /// the bridge may weld map structure.
    pub min_landmark_matches_for_welding: usize,
    /// Optional projection window for a causal Atlas-frame pose prior. The
    /// prior only harvests correspondences; PnP and all merge gates still run.
    pub projection_search_radius_px: Option<f64>,
    /// Retry descriptor-global localization when projection-guided PnP fails.
    pub appearance_fallback_on_projection_failure: bool,
    pub merge_verification: SubmapMergeVerificationConfig,
}

impl Default for CrossSubmapAlignmentConfig {
    fn default() -> Self {
        Self {
            max_target_submaps: 5,
            min_appearance_similarity: 0.2,
            min_candidate_landmarks: 30,
            max_scale_points: 64,
            min_scale_pair_distance_m: 0.05,
            min_scale_pair_count: 15,
            scale_inlier_relative_tolerance: 0.1,
            min_scale_inlier_ratio: 0.6,
            max_scale_mad: 0.05,
            max_landmark_match_residual_m: 0.25,
            min_landmark_matches_for_welding: 8,
            projection_search_radius_px: Some(30.0),
            appearance_fallback_on_projection_failure: true,
            merge_verification: SubmapMergeVerificationConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossSubmapScaleEstimate {
    pub estimated_scale: f64,
    pub pair_count: usize,
    pub inlier_count: usize,
    pub inlier_ratio: f64,
    pub median_absolute_deviation: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossSubmapLandmarkMatch {
    pub source_landmark_id: LandmarkId,
    pub target_landmark_id: LandmarkId,
    pub transformed_distance_m: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrossSubmapCandidateFailureReason {
    TooFewCandidateLandmarks { actual: usize, minimum: usize },
    LocalizationRejected,
    TooFewMetricPointCorrespondences { actual: usize, minimum: usize },
    TooFewScalePairs { actual: usize, minimum: usize },
    LowScaleInlierRatio { actual: f64, minimum: f64 },
    HighScaleMedianAbsoluteDeviation { actual: f64, maximum: f64 },
    RigidAlignmentFailed,
    TooFewRigidInliers { actual: usize, minimum: usize },
    TooFewWeldableLandmarkMatches { actual: usize, minimum: usize },
    MergeRejected(MapAtlasError),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossSubmapCandidateDiagnostic {
    pub target_submap_id: SubmapId,
    pub target_frame_id: FrameId,
    pub appearance_similarity: f32,
    pub candidate_landmark_count: usize,
    pub used_projection_prior: bool,
    pub localization_correspondence_count: usize,
    pub localization_inlier_count: usize,
    pub localization_inlier_ratio: f64,
    pub mean_reprojection_error_px: Option<f64>,
    pub scale_estimate: Option<CrossSubmapScaleEstimate>,
    pub failure_reason: Option<CrossSubmapCandidateFailureReason>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrossSubmapAlignmentResult {
    pub source_submap_id: SubmapId,
    pub source_frame_id: FrameId,
    pub ranked_candidate_count: usize,
    pub best_appearance_similarity: Option<f32>,
    pub diagnostics: Vec<CrossSubmapCandidateDiagnostic>,
    pub verified_merge: Option<VerifiedSubmapMerge>,
}

/// Aggregate result for a broader recovery pass over multiple retained source
/// keyframes. Individual alignment diagnostics remain available so callers
/// can audit which boundary/current view supplied the verified bridge.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSubmapWindowAlignmentResult {
    pub source_submap_id: SubmapId,
    pub requested_source_frame_count: usize,
    pub attempted_source_frame_ids: Vec<FrameId>,
    pub alignments: Vec<CrossSubmapAlignmentResult>,
    pub verified_merge: Option<VerifiedSubmapMerge>,
}

/// Auditable result of a DROID-style boundary factor: descriptors are matched
/// directly between the last tracked frame of the old submap and the stereo
/// seed of the new submap, then verified in metric 3D before Atlas admission.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSubmapBoundaryFactorResult {
    pub source_submap_id: SubmapId,
    pub source_frame_id: FrameId,
    pub target_submap_id: SubmapId,
    pub target_frame_id: FrameId,
    pub descriptor_match_count: usize,
    pub spatial_mutual_match_count: usize,
    pub metric_correspondence_count: usize,
    pub projection_refined_correspondence_count: usize,
    pub projection_refinement_iterations: usize,
    pub used_transform_prior: bool,
    pub rigid_inlier_count: usize,
    pub rigid_inlier_ratio: f64,
    pub mean_target_reprojection_error_px: Option<f64>,
    pub scale_estimate: Option<CrossSubmapScaleEstimate>,
    pub failure_reason: Option<CrossSubmapCandidateFailureReason>,
    pub verified_merge: Option<VerifiedSubmapMerge>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedSubmapMerge {
    pub evidence: SubmapMergeEvidence,
    pub atlas_from_source: SE3,
    /// Geometrically verified same-point relations. Empty for pose-only
    /// bridges supplied through [`MapAtlas::verify_and_align`].
    pub landmark_matches: Vec<CrossSubmapLandmarkMatch>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubmapIdRemap {
    pub camera_ids: BTreeMap<CameraId, CameraId>,
    pub frame_ids: BTreeMap<FrameId, FrameId>,
    pub landmark_ids: BTreeMap<LandmarkId, LandmarkId>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedAtlas {
    pub map: VisualMap,
    pub id_remaps: BTreeMap<SubmapId, SubmapIdRemap>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MapAtlasError {
    MissingSubmap(SubmapId),
    MissingKeyframe {
        submap_id: SubmapId,
        frame_id: FrameId,
    },
    MissingKeyframePose {
        submap_id: SubmapId,
        frame_id: FrameId,
    },
    MissingLandmark {
        submap_id: SubmapId,
        landmark_id: LandmarkId,
    },
    NonFiniteLandmarkMatchDistance,
    TargetSubmapUnaligned(SubmapId),
    SameSubmapMerge(SubmapId),
    InsufficientInliers {
        actual: usize,
        minimum: usize,
    },
    InlierRatioBelowMinimum {
        actual: f64,
        minimum: f64,
    },
    ReprojectionErrorAboveMaximum {
        actual: f64,
        maximum: f64,
    },
    MetricScaleInconsistent {
        actual: f64,
        maximum_error: f64,
    },
    ExistingAlignmentInconsistent {
        translation_disagreement_m: f64,
        rotation_disagreement_rad: f64,
    },
    NonFiniteTransform,
    IdSpaceExhausted,
}

impl fmt::Display for MapAtlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSubmap(id) => write!(f, "missing submap {id}"),
            Self::MissingKeyframe {
                submap_id,
                frame_id,
            } => write!(f, "submap {submap_id} is missing keyframe {frame_id}"),
            Self::MissingKeyframePose {
                submap_id,
                frame_id,
            } => write!(f, "submap {submap_id} keyframe {frame_id} has no pose"),
            Self::MissingLandmark {
                submap_id,
                landmark_id,
            } => write!(f, "submap {submap_id} is missing landmark {landmark_id}"),
            Self::NonFiniteLandmarkMatchDistance => {
                write!(f, "landmark match distance is non-finite or negative")
            }
            Self::TargetSubmapUnaligned(id) => write!(f, "target submap {id} is not aligned"),
            Self::SameSubmapMerge(id) => write!(f, "cannot merge submap {id} with itself"),
            Self::InsufficientInliers { actual, minimum } => {
                write!(f, "merge has {actual} inliers; need at least {minimum}")
            }
            Self::InlierRatioBelowMinimum { actual, minimum } => {
                write!(f, "merge inlier ratio {actual} is below {minimum}")
            }
            Self::ReprojectionErrorAboveMaximum { actual, maximum } => {
                write!(f, "merge reprojection error {actual} px exceeds {maximum} px")
            }
            Self::MetricScaleInconsistent {
                actual,
                maximum_error,
            } => write!(
                f,
                "merge metric scale {actual} differs from one by more than {maximum_error}"
            ),
            Self::ExistingAlignmentInconsistent {
                translation_disagreement_m,
                rotation_disagreement_rad,
            } => write!(
                f,
                "merge disagrees with Atlas by {translation_disagreement_m} m / {rotation_disagreement_rad} rad"
            ),
            Self::NonFiniteTransform => write!(f, "merge transform contains non-finite values"),
            Self::IdSpaceExhausted => write!(f, "materialized Atlas id space exhausted"),
        }
    }
}

impl Error for MapAtlasError {}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MapAtlas {
    submaps: BTreeMap<SubmapId, AtlasSubmap>,
    verified_merges: Vec<VerifiedSubmapMerge>,
    next_submap_id: SubmapId,
    root_submap_id: Option<SubmapId>,
}

impl MapAtlas {
    pub fn new(root_map: VisualMap) -> Self {
        let mut atlas = Self::default();
        let root_id = atlas.insert_independent(root_map);
        atlas.root_submap_id = Some(root_id);
        atlas.submaps.get_mut(&root_id).unwrap().atlas_from_local = Some(SE3::identity());
        atlas
    }

    /// Insert an independent local map without transforming or combining its
    /// frame/landmark id space with any existing map.
    pub fn insert_independent(&mut self, map: VisualMap) -> SubmapId {
        self.try_insert_independent(map)
            .expect("MapAtlas submap id space exhausted")
    }

    /// Fallible counterpart to [`Self::insert_independent`] for long-running
    /// services that must surface identifier exhaustion instead of panicking.
    pub fn try_insert_independent(&mut self, map: VisualMap) -> Result<SubmapId, MapAtlasError> {
        let id = self.next_submap_id;
        self.next_submap_id = self
            .next_submap_id
            .checked_add(1)
            .ok_or(MapAtlasError::IdSpaceExhausted)?;
        self.submaps.insert(
            id,
            AtlasSubmap {
                id,
                map,
                atlas_from_local: None,
            },
        );
        Ok(id)
    }

    pub fn root_submap_id(&self) -> Option<SubmapId> {
        self.root_submap_id
    }

    pub fn active_submaps(&self) -> impl Iterator<Item = &AtlasSubmap> {
        self.submaps.values()
    }

    pub fn submap(&self, id: SubmapId) -> Option<&AtlasSubmap> {
        self.submaps.get(&id)
    }

    pub fn submap_mut(&mut self, id: SubmapId) -> Option<&mut AtlasSubmap> {
        self.submaps.get_mut(&id)
    }

    /// Refresh an owned submap from its active online pipeline while retaining
    /// the submap's local gauge and any verified Atlas alignment.
    pub fn replace_submap_map(
        &mut self,
        id: SubmapId,
        map: VisualMap,
    ) -> Result<VisualMap, MapAtlasError> {
        let submap = self
            .submaps
            .get_mut(&id)
            .ok_or(MapAtlasError::MissingSubmap(id))?;
        Ok(std::mem::replace(&mut submap.map, map))
    }

    pub fn aligned_submap_count(&self) -> usize {
        self.submaps
            .values()
            .filter(|submap| submap.atlas_from_local.is_some())
            .count()
    }

    pub fn independent_submap_count(&self) -> usize {
        self.submaps
            .len()
            .saturating_sub(self.aligned_submap_count())
    }

    pub fn verified_merges(&self) -> &[VerifiedSubmapMerge] {
        &self.verified_merges
    }

    /// Deterministic source window for Atlas-level broader recovery. The
    /// current view is tried first, followed by the earliest boundary views,
    /// then recent retained keyframes. This keeps the cheap causal attempt
    /// first while preserving overlap that may have disappeared after a cliff.
    pub fn source_keyframe_recovery_window(
        &self,
        source_submap_id: SubmapId,
        current_frame_id: FrameId,
        maximum: usize,
    ) -> Result<Vec<FrameId>, MapAtlasError> {
        let source = self
            .submaps
            .get(&source_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(source_submap_id))?;
        if !source.map.keyframes.contains_key(&current_frame_id) {
            return Err(MapAtlasError::MissingKeyframe {
                submap_id: source_submap_id,
                frame_id: current_frame_id,
            });
        }
        let maximum = maximum.max(1);
        let mut sorted = source.map.keyframes.keys().copied().collect::<Vec<_>>();
        sorted.sort_unstable();
        let mut window = Vec::with_capacity(maximum.min(sorted.len()));
        let mut seen = HashSet::new();
        let mut push = |frame_id: FrameId| {
            if window.len() < maximum && seen.insert(frame_id) {
                window.push(frame_id);
            }
        };
        push(current_frame_id);
        for frame_id in sorted.iter().take(2).copied() {
            push(frame_id);
        }
        for frame_id in sorted.iter().rev().copied() {
            push(frame_id);
        }
        Ok(window)
    }

    /// Retry cross-submap alignment over retained current/boundary keyframes.
    /// The map-frame prior derives projection windows only; every candidate
    /// still passes PnP, metric scale, 3D residual, and merge verification.
    #[allow(clippy::too_many_arguments)]
    pub fn align_submap_keyframe_window<M, S, E>(
        &mut self,
        source_submap_id: SubmapId,
        source_frame_ids: &[FrameId],
        source_camera: &Camera,
        atlas_from_source_prior: Option<&SE3>,
        localizer: &LocalizationPipeline<M, S, E>,
        config: &CrossSubmapAlignmentConfig,
    ) -> Result<CrossSubmapWindowAlignmentResult, MapAtlasError>
    where
        M: Matcher + Clone,
        S: CandidateSelector + Clone,
        E: RobustPoseEstimator + Clone,
    {
        if !self.submaps.contains_key(&source_submap_id) {
            return Err(MapAtlasError::MissingSubmap(source_submap_id));
        }
        let mut attempted_source_frame_ids = Vec::new();
        let mut alignments = Vec::new();
        let mut verified_merge = None;
        let mut seen = HashSet::new();
        for source_frame_id in source_frame_ids.iter().copied() {
            if !seen.insert(source_frame_id) {
                continue;
            }
            let atlas_pose_prior = if let Some(atlas_from_source_prior) = atlas_from_source_prior {
                let source = self
                    .submaps
                    .get(&source_submap_id)
                    .ok_or(MapAtlasError::MissingSubmap(source_submap_id))?;
                let keyframe = source.map.keyframes.get(&source_frame_id).ok_or(
                    MapAtlasError::MissingKeyframe {
                        submap_id: source_submap_id,
                        frame_id: source_frame_id,
                    },
                )?;
                let pose =
                    keyframe
                        .frame
                        .pose
                        .as_ref()
                        .ok_or(MapAtlasError::MissingKeyframePose {
                            submap_id: source_submap_id,
                            frame_id: source_frame_id,
                        })?;
                Some(Pose {
                    world_to_camera: pose
                        .world_to_camera
                        .compose(&atlas_from_source_prior.inverse()),
                })
            } else {
                None
            };
            attempted_source_frame_ids.push(source_frame_id);
            let alignment = self.align_submap_keyframe(
                source_submap_id,
                source_frame_id,
                source_camera,
                atlas_pose_prior.as_ref(),
                localizer,
                config,
            )?;
            if let Some(verified) = alignment.verified_merge.as_ref() {
                verified_merge = Some(verified.clone());
                alignments.push(alignment);
                break;
            }
            alignments.push(alignment);
        }
        Ok(CrossSubmapWindowAlignmentResult {
            source_submap_id,
            requested_source_frame_count: source_frame_ids.len(),
            attempted_source_frame_ids,
            alignments,
            verified_merge,
        })
    }

    /// Align two adjacent submaps from the explicit frame-to-frame factor at
    /// their tracking boundary. Unlike map-wide relocalization, this retains
    /// the actual last-good view and its landmark observations across the
    /// reset. Descriptor matches are only proposals: deterministic metric
    /// 3D RANSAC, scale consensus, target-view reprojection, and the normal
    /// Atlas merge gates remain mandatory.
    pub fn align_adjacent_boundary_factor<M: Matcher>(
        &mut self,
        source_submap_id: SubmapId,
        source_frame_id: FrameId,
        target_submap_id: SubmapId,
        target_frame_id: FrameId,
        target_from_source_prior: Option<&SE3>,
        matcher: &M,
        config: &CrossSubmapAlignmentConfig,
    ) -> Result<CrossSubmapBoundaryFactorResult, MapAtlasError> {
        let (source_observed, target_observed, target_pose, target_camera) =
            {
                let source = self
                    .submaps
                    .get(&source_submap_id)
                    .ok_or(MapAtlasError::MissingSubmap(source_submap_id))?;
                let target = self
                    .submaps
                    .get(&target_submap_id)
                    .ok_or(MapAtlasError::MissingSubmap(target_submap_id))?;
                let source_keyframe = source.map.keyframes.get(&source_frame_id).ok_or(
                    MapAtlasError::MissingKeyframe {
                        submap_id: source_submap_id,
                        frame_id: source_frame_id,
                    },
                )?;
                let target_keyframe = target.map.keyframes.get(&target_frame_id).ok_or(
                    MapAtlasError::MissingKeyframe {
                        submap_id: target_submap_id,
                        frame_id: target_frame_id,
                    },
                )?;
                let target_pose = target_keyframe.frame.pose.clone().ok_or(
                    MapAtlasError::MissingKeyframePose {
                        submap_id: target_submap_id,
                        frame_id: target_frame_id,
                    },
                )?;
                let target_camera = target
                    .map
                    .cameras
                    .get(&target_keyframe.frame.camera_id)
                    .cloned();
                let source_observed = source_keyframe
                    .observations
                    .iter()
                    .filter_map(|observation| {
                        let landmark = source.map.landmarks.get(&observation.landmark_id)?;
                        let descriptor = source_keyframe
                            .frame
                            .descriptors
                            .get(observation.keypoint_index)?;
                        Some((
                            observation.landmark_id,
                            landmark.position,
                            observation.xy,
                            descriptor.clone(),
                        ))
                    })
                    .collect::<Vec<_>>();
                let target_observed = target_keyframe
                    .observations
                    .iter()
                    .filter_map(|observation| {
                        let landmark = target.map.landmarks.get(&observation.landmark_id)?;
                        let descriptor = target_keyframe
                            .frame
                            .descriptors
                            .get(observation.keypoint_index)?;
                        Some((
                            observation.landmark_id,
                            landmark.position,
                            observation.xy,
                            descriptor.clone(),
                        ))
                    })
                    .collect::<Vec<_>>();
                (source_observed, target_observed, target_pose, target_camera)
            };

        let source_descriptors = source_observed
            .iter()
            .map(|(_, _, _, descriptor)| descriptor.clone())
            .collect::<Vec<_>>();
        let target_descriptors = target_observed
            .iter()
            .map(|(_, _, _, descriptor)| descriptor.clone())
            .collect::<Vec<_>>();
        let descriptor_matches =
            matcher.match_descriptors(&source_descriptors, &target_descriptors);
        let mut result = CrossSubmapBoundaryFactorResult {
            source_submap_id,
            source_frame_id,
            target_submap_id,
            target_frame_id,
            descriptor_match_count: descriptor_matches.len(),
            spatial_mutual_match_count: 0,
            metric_correspondence_count: 0,
            projection_refined_correspondence_count: 0,
            projection_refinement_iterations: 0,
            used_transform_prior: target_from_source_prior.is_some(),
            rigid_inlier_count: 0,
            rigid_inlier_ratio: 0.0,
            mean_target_reprojection_error_px: None,
            scale_estimate: None,
            failure_reason: None,
            verified_merge: None,
        };
        let boundary_radius_px = config
            .projection_search_radius_px
            .map(|radius| 3.0 * radius)
            .unwrap_or(f64::INFINITY);
        let mut candidate_pairs = Vec::new();
        for (source_index, (_, _, source_xy, _)) in source_observed.iter().enumerate() {
            let Some((target_index, distance)) = target_observed
                .iter()
                .enumerate()
                .map(|(index, (_, _, target_xy, _))| (index, (*source_xy - *target_xy).norm()))
                .min_by(|left, right| left.1.total_cmp(&right.1))
            else {
                continue;
            };
            if distance > boundary_radius_px {
                continue;
            }
            let reverse_source_index = source_observed
                .iter()
                .enumerate()
                .min_by(|(_, (_, _, left_xy, _)), (_, (_, _, right_xy, _))| {
                    (*left_xy - target_observed[target_index].2)
                        .norm()
                        .total_cmp(&(*right_xy - target_observed[target_index].2).norm())
                })
                .map(|(index, _)| index);
            if reverse_source_index == Some(source_index) {
                candidate_pairs.push((source_index, target_index));
            }
        }
        result.spatial_mutual_match_count = candidate_pairs.len();
        for descriptor_match in descriptor_matches {
            let pair = (descriptor_match.query_index, descriptor_match.train_index);
            let Some((_, _, source_xy, _)) = source_observed.get(pair.0) else {
                continue;
            };
            let Some((_, _, target_xy, _)) = target_observed.get(pair.1) else {
                continue;
            };
            if (*source_xy - *target_xy).norm() <= boundary_radius_px
                && !candidate_pairs.contains(&pair)
            {
                candidate_pairs.push(pair);
            }
        }

        let mut used_source_landmarks = HashSet::new();
        let mut used_target_landmarks = HashSet::new();
        let mut metric = Vec::new();
        for (source_index, target_index) in candidate_pairs {
            let (source_landmark_id, source_point, _, _) = source_observed[source_index];
            let (target_landmark_id, target_point, target_xy, _) = &target_observed[target_index];
            if used_source_landmarks.insert(source_landmark_id)
                && used_target_landmarks.insert(*target_landmark_id)
            {
                metric.push((
                    source_landmark_id,
                    *target_landmark_id,
                    source_point,
                    *target_point,
                    *target_xy,
                ));
            }
        }
        result.metric_correspondence_count = metric.len();
        if metric.len() < 3 {
            result.failure_reason = Some(
                CrossSubmapCandidateFailureReason::TooFewMetricPointCorrespondences {
                    actual: metric.len(),
                    minimum: 3,
                },
            );
            return Ok(result);
        }

        let point_pairs = metric
            .iter()
            .map(|(_, _, source, target, _)| (*source, *target))
            .collect::<Vec<_>>();
        let Some((mut target_from_source, mut rigid_inlier_indices)) =
            estimate_boundary_rigid_transform_ransac(
                &point_pairs,
                config.max_landmark_match_residual_m,
                256,
            )
        else {
            result.failure_reason = Some(CrossSubmapCandidateFailureReason::RigidAlignmentFailed);
            return Ok(result);
        };

        // Recurrent geometric update: use the first metric seed to reproject
        // every observed source landmark into the target boundary view,
        // rebuild one-to-one correspondences, and solve again. This mirrors
        // DROID's target-update -> BA loop without weakening Atlas admission.
        if let Some(camera) = target_camera.as_ref() {
            let refinement_radius_px = config.projection_search_radius_px.unwrap_or(15.0).min(15.0);
            let mut projection_transform = target_from_source_prior
                .cloned()
                .unwrap_or_else(|| target_from_source.clone());
            for _ in 0..2 {
                let mut proposals = Vec::new();
                for (source_index, (_, source_point, _, _)) in source_observed.iter().enumerate() {
                    let target_point = projection_transform.transform_point(source_point);
                    let camera_point = target_pose.transform_world_point(&target_point);
                    let Some(projected) = camera.project(&camera_point) else {
                        continue;
                    };
                    for (target_index, (_, _, target_xy, _)) in target_observed.iter().enumerate() {
                        let error = (projected - *target_xy).norm();
                        if error.is_finite() && error <= refinement_radius_px {
                            proposals.push((error, source_index, target_index));
                        }
                    }
                }
                proposals.sort_by(|left, right| {
                    left.0
                        .total_cmp(&right.0)
                        .then_with(|| left.1.cmp(&right.1))
                        .then_with(|| left.2.cmp(&right.2))
                });
                let mut used_sources = HashSet::new();
                let mut used_targets = HashSet::new();
                let mut refined_metric = Vec::new();
                for (_, source_index, target_index) in proposals {
                    if !used_sources.insert(source_index) || !used_targets.insert(target_index) {
                        continue;
                    }
                    let (source_landmark_id, source_point, _, _) = source_observed[source_index];
                    let (target_landmark_id, target_point, target_xy, _) =
                        &target_observed[target_index];
                    refined_metric.push((
                        source_landmark_id,
                        *target_landmark_id,
                        source_point,
                        *target_point,
                        *target_xy,
                    ));
                }
                result.projection_refined_correspondence_count = result
                    .projection_refined_correspondence_count
                    .max(refined_metric.len());
                if refined_metric.len() < 3 {
                    break;
                }
                let refined_points = refined_metric
                    .iter()
                    .map(|(_, _, source, target, _)| (*source, *target))
                    .collect::<Vec<_>>();
                let Some((refined_transform, refined_inliers)) =
                    estimate_boundary_rigid_transform_ransac(
                        &refined_points,
                        config.max_landmark_match_residual_m,
                        256,
                    )
                else {
                    break;
                };
                result.projection_refinement_iterations += 1;
                if refined_inliers.len() <= rigid_inlier_indices.len() {
                    break;
                }
                metric = refined_metric;
                target_from_source = refined_transform;
                rigid_inlier_indices = refined_inliers;
                projection_transform = target_from_source.clone();
            }
        }
        result.rigid_inlier_count = rigid_inlier_indices.len();
        result.rigid_inlier_ratio = rigid_inlier_indices.len() as f64 / metric.len() as f64;
        if rigid_inlier_indices.len() < config.merge_verification.min_inliers {
            result.failure_reason = Some(CrossSubmapCandidateFailureReason::TooFewRigidInliers {
                actual: rigid_inlier_indices.len(),
                minimum: config.merge_verification.min_inliers,
            });
            return Ok(result);
        }

        let scale_points = rigid_inlier_indices
            .iter()
            .map(|index| {
                let (_, _, source, target, _) = metric[*index];
                (source, target)
            })
            .collect::<Vec<_>>();
        let scale_estimate = match estimate_cross_submap_scale(&scale_points, config) {
            Ok(estimate) => estimate,
            Err(reason) => {
                result.failure_reason = Some(reason);
                return Ok(result);
            }
        };
        result.scale_estimate = Some(scale_estimate);

        let reprojection_errors = target_camera
            .as_ref()
            .map(|camera| {
                rigid_inlier_indices
                    .iter()
                    .filter_map(|index| {
                        let (_, _, source_point, _, target_xy) = metric[*index];
                        let target_point = target_from_source.transform_point(&source_point);
                        let camera_point = target_pose.transform_world_point(&target_point);
                        camera
                            .project(&camera_point)
                            .map(|projected| (projected - target_xy).norm())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let mean_reprojection_error_px = (!reprojection_errors.is_empty())
            .then(|| reprojection_errors.iter().sum::<f64>() / reprojection_errors.len() as f64);
        result.mean_target_reprojection_error_px = mean_reprojection_error_px;

        let landmark_matches = rigid_inlier_indices
            .iter()
            .map(|index| {
                let (source_landmark_id, target_landmark_id, source, target, _) = metric[*index];
                CrossSubmapLandmarkMatch {
                    source_landmark_id,
                    target_landmark_id,
                    transformed_distance_m: (target_from_source.transform_point(&source) - target)
                        .norm(),
                }
            })
            .collect::<Vec<_>>();
        if landmark_matches.len() < config.min_landmark_matches_for_welding {
            result.failure_reason = Some(
                CrossSubmapCandidateFailureReason::TooFewWeldableLandmarkMatches {
                    actual: landmark_matches.len(),
                    minimum: config.min_landmark_matches_for_welding,
                },
            );
            return Ok(result);
        }
        let evidence = SubmapMergeEvidence {
            source_submap_id,
            target_submap_id,
            target_from_source,
            inlier_count: rigid_inlier_indices.len(),
            inlier_ratio: result.rigid_inlier_ratio,
            mean_reprojection_error_px: mean_reprojection_error_px.unwrap_or(f64::INFINITY),
            estimated_scale: scale_estimate.estimated_scale,
        };
        match self.verify_and_align_with_landmark_matches(
            evidence,
            landmark_matches,
            &config.merge_verification,
        ) {
            Ok(verified) => result.verified_merge = Some(verified),
            Err(error) => {
                result.failure_reason =
                    Some(CrossSubmapCandidateFailureReason::MergeRejected(error));
            }
        }
        Ok(result)
    }

    /// Retrieve aligned target keyframes by global descriptor, relocalize one
    /// source keyframe against each candidate's 3D landmarks, independently
    /// verify metric scale from matched 3D-to-3D distances, and admit the
    /// first bridge that passes [`Self::verify_and_align`]. Ground truth is
    /// never consulted.
    pub fn align_submap_keyframe<M, S, E>(
        &mut self,
        source_submap_id: SubmapId,
        source_frame_id: FrameId,
        source_camera: &Camera,
        atlas_pose_prior: Option<&Pose>,
        localizer: &LocalizationPipeline<M, S, E>,
        config: &CrossSubmapAlignmentConfig,
    ) -> Result<CrossSubmapAlignmentResult, MapAtlasError>
    where
        M: Matcher + Clone,
        S: CandidateSelector + Clone,
        E: RobustPoseEstimator + Clone,
    {
        let source = self
            .submaps
            .get(&source_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(source_submap_id))?;
        let source_keyframe = source
            .map
            .keyframes
            .get(&source_frame_id)
            .ok_or(MapAtlasError::MissingKeyframe {
                submap_id: source_submap_id,
                frame_id: source_frame_id,
            })?
            .clone();
        let source_pose = source_keyframe
            .frame
            .pose
            .as_ref()
            .ok_or(MapAtlasError::MissingKeyframePose {
                submap_id: source_submap_id,
                frame_id: source_frame_id,
            })?
            .clone();
        let Some(source_descriptor) =
            relocalization_mean_descriptor(&source_keyframe.frame.descriptors)
        else {
            return Ok(CrossSubmapAlignmentResult {
                source_submap_id,
                source_frame_id,
                ranked_candidate_count: 0,
                best_appearance_similarity: None,
                diagnostics: Vec::new(),
                verified_merge: None,
            });
        };

        let mut source_points_by_query = HashMap::new();
        for observation in &source_keyframe.observations {
            if let Some(landmark) = source.map.landmarks.get(&observation.landmark_id) {
                source_points_by_query
                    .entry(observation.keypoint_index)
                    .or_insert((observation.landmark_id, landmark.position));
            }
        }

        let mut best_keyframe_by_target_submap = BTreeMap::new();
        for (target_submap_id, target) in &self.submaps {
            if *target_submap_id == source_submap_id || target.atlas_from_local.is_none() {
                continue;
            }
            for (target_frame_id, target_keyframe) in &target.map.keyframes {
                let Some(target_descriptor) =
                    relocalization_mean_descriptor(&target_keyframe.frame.descriptors)
                else {
                    continue;
                };
                let similarity =
                    relocalization_descriptor_cosine(&source_descriptor, &target_descriptor);
                if similarity >= config.min_appearance_similarity {
                    let entry = best_keyframe_by_target_submap
                        .entry(*target_submap_id)
                        .or_insert((*target_frame_id, similarity));
                    if similarity > entry.1 || (similarity == entry.1 && *target_frame_id < entry.0)
                    {
                        *entry = (*target_frame_id, similarity);
                    }
                }
            }
        }
        let mut ranked = best_keyframe_by_target_submap
            .into_iter()
            .map(|(submap_id, (frame_id, similarity))| (submap_id, frame_id, similarity))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| left.1.cmp(&right.1))
        });
        let ranked_candidate_count = ranked.len();
        let best_appearance_similarity = ranked.first().map(|candidate| candidate.2);
        let query = QueryImage::from_frame(&source_keyframe.frame, source_camera.clone());
        let mut diagnostics = Vec::new();

        for (target_submap_id, target_frame_id, appearance_similarity) in
            ranked.into_iter().take(config.max_target_submaps.max(1))
        {
            let (candidate_store, localization, used_projection_prior, target_points) = {
                let target = self
                    .submaps
                    .get(&target_submap_id)
                    .ok_or(MapAtlasError::MissingSubmap(target_submap_id))?;
                let _target_keyframe = target.map.keyframes.get(&target_frame_id).ok_or(
                    MapAtlasError::MissingKeyframe {
                        submap_id: target_submap_id,
                        frame_id: target_frame_id,
                    },
                )?;
                // Retrieval is keyframe-scoped, but PnP consumes the complete
                // target submap. Local mapping may retain only a small subset
                // of seed landmarks on any one keyframe; restricting the
                // bridge store to that subset creates an artificial cliff.
                let mut landmark_ids = target.map.landmarks.keys().copied().collect::<Vec<_>>();
                landmark_ids.sort_unstable();
                landmark_ids.dedup();
                let mut store = LandmarkDescriptorStore::new();
                for landmark_id in landmark_ids {
                    if let Some(descriptor) = target
                        .map
                        .landmarks
                        .get(&landmark_id)
                        .and_then(|landmark| landmark.descriptor.as_ref())
                    {
                        store.insert(landmark_id, descriptor.clone());
                    }
                }
                if store.len() < config.min_candidate_landmarks {
                    diagnostics.push(CrossSubmapCandidateDiagnostic {
                        target_submap_id,
                        target_frame_id,
                        appearance_similarity,
                        candidate_landmark_count: store.len(),
                        used_projection_prior: false,
                        localization_correspondence_count: 0,
                        localization_inlier_count: 0,
                        localization_inlier_ratio: 0.0,
                        mean_reprojection_error_px: None,
                        scale_estimate: None,
                        failure_reason: Some(
                            CrossSubmapCandidateFailureReason::TooFewCandidateLandmarks {
                                actual: store.len(),
                                minimum: config.min_candidate_landmarks,
                            },
                        ),
                    });
                    continue;
                }
                let projection_result = atlas_pose_prior
                    .zip(config.projection_search_radius_px)
                    .and_then(|(atlas_pose_prior, search_radius_px)| {
                        let atlas_from_target = target.atlas_from_local.as_ref()?;
                        let target_pose_prior = Pose {
                            world_to_camera: atlas_pose_prior
                                .world_to_camera
                                .compose(atlas_from_target),
                        };
                        Some(
                            localizer.localize_with_projection_window_and_descriptor_store(
                                &query,
                                &target.map,
                                &store,
                                localizer.candidate_selector.clone(),
                                &target_pose_prior,
                                search_radius_px,
                            ),
                        )
                    });
                let (localization, used_projection_prior) = match projection_result {
                    Some(result)
                        if result.success || !config.appearance_fallback_on_projection_failure =>
                    {
                        (result, true)
                    }
                    _ => (
                        localizer.localize_with_descriptor_store(&query, &target.map, &store),
                        false,
                    ),
                };
                let target_points = target
                    .map
                    .landmarks
                    .iter()
                    .map(|(id, landmark)| (*id, landmark.position))
                    .collect::<HashMap<_, _>>();
                (store, localization, used_projection_prior, target_points)
            };

            let mut diagnostic = CrossSubmapCandidateDiagnostic {
                target_submap_id,
                target_frame_id,
                appearance_similarity,
                candidate_landmark_count: candidate_store.len(),
                used_projection_prior,
                localization_correspondence_count: localization.correspondence_count,
                localization_inlier_count: localization.inlier_count,
                localization_inlier_ratio: localization.inlier_ratio,
                mean_reprojection_error_px: localization.reprojection_error,
                scale_estimate: None,
                failure_reason: None,
            };
            let Some(target_pose) = localization
                .success
                .then(|| localization.pose.as_ref())
                .flatten()
            else {
                diagnostic.failure_reason =
                    Some(CrossSubmapCandidateFailureReason::LocalizationRejected);
                diagnostics.push(diagnostic);
                continue;
            };

            let mut used_source_landmarks = HashSet::new();
            let mut used_target_landmarks = HashSet::new();
            let mut metric_landmark_points = Vec::new();
            for (query_index, target_landmark_id) in localization
                .inlier_query_indices
                .iter()
                .copied()
                .zip(localization.inlier_landmark_ids.iter().copied())
            {
                let Some(&(source_landmark_id, source_point)) =
                    source_points_by_query.get(&query_index)
                else {
                    continue;
                };
                let Some(&target_point) = target_points.get(&target_landmark_id) else {
                    continue;
                };
                if used_source_landmarks.insert(source_landmark_id)
                    && used_target_landmarks.insert(target_landmark_id)
                {
                    metric_landmark_points.push((
                        source_landmark_id,
                        target_landmark_id,
                        source_point,
                        target_point,
                    ));
                }
            }
            if metric_landmark_points.len() < 3 {
                diagnostic.failure_reason = Some(
                    CrossSubmapCandidateFailureReason::TooFewMetricPointCorrespondences {
                        actual: metric_landmark_points.len(),
                        minimum: 3,
                    },
                );
                diagnostics.push(diagnostic);
                continue;
            }
            let metric_points = metric_landmark_points
                .iter()
                .map(|(_, _, source_point, target_point)| (*source_point, *target_point))
                .collect::<Vec<_>>();
            let scale_estimate = match estimate_cross_submap_scale(&metric_points, config) {
                Ok(estimate) => estimate,
                Err(reason) => {
                    diagnostic.failure_reason = Some(reason);
                    diagnostics.push(diagnostic);
                    continue;
                }
            };
            diagnostic.scale_estimate = Some(scale_estimate);

            let target_from_source = target_pose
                .world_to_camera
                .inverse()
                .compose(&source_pose.world_to_camera);
            let evidence = SubmapMergeEvidence {
                source_submap_id,
                target_submap_id,
                target_from_source: target_from_source.clone(),
                inlier_count: localization.inlier_count,
                inlier_ratio: localization.inlier_ratio,
                mean_reprojection_error_px: localization
                    .reprojection_error
                    .unwrap_or(f64::INFINITY),
                estimated_scale: scale_estimate.estimated_scale,
            };
            if let Err(error) = validate_merge_evidence(&evidence, &config.merge_verification) {
                diagnostic.failure_reason =
                    Some(CrossSubmapCandidateFailureReason::MergeRejected(error));
                diagnostics.push(diagnostic);
                continue;
            }
            let landmark_matches = metric_landmark_points
                .iter()
                .filter_map(
                    |(source_landmark_id, target_landmark_id, source_point, target_point)| {
                        let residual = (target_from_source.transform_point(source_point)
                            - target_point)
                            .norm();
                        (residual.is_finite() && residual <= config.max_landmark_match_residual_m)
                            .then_some(CrossSubmapLandmarkMatch {
                                source_landmark_id: *source_landmark_id,
                                target_landmark_id: *target_landmark_id,
                                transformed_distance_m: residual,
                            })
                    },
                )
                .collect::<Vec<_>>();
            if landmark_matches.len() < config.min_landmark_matches_for_welding {
                diagnostic.failure_reason = Some(
                    CrossSubmapCandidateFailureReason::TooFewWeldableLandmarkMatches {
                        actual: landmark_matches.len(),
                        minimum: config.min_landmark_matches_for_welding,
                    },
                );
                diagnostics.push(diagnostic);
                continue;
            }
            match self.verify_and_align_with_landmark_matches(
                evidence,
                landmark_matches,
                &config.merge_verification,
            ) {
                Ok(verified_merge) => {
                    diagnostics.push(diagnostic);
                    return Ok(CrossSubmapAlignmentResult {
                        source_submap_id,
                        source_frame_id,
                        ranked_candidate_count,
                        best_appearance_similarity,
                        diagnostics,
                        verified_merge: Some(verified_merge),
                    });
                }
                Err(error) => {
                    diagnostic.failure_reason =
                        Some(CrossSubmapCandidateFailureReason::MergeRejected(error));
                    diagnostics.push(diagnostic);
                }
            }
        }

        Ok(CrossSubmapAlignmentResult {
            source_submap_id,
            source_frame_id,
            ranked_candidate_count,
            best_appearance_similarity,
            diagnostics,
            verified_merge: None,
        })
    }

    /// Convert a verified cross-submap keyframe pose into the map-frame
    /// transform consumed by [`Self::verify_and_align`].
    ///
    /// `source_camera_from_target_camera` uses the same `a -> b` convention as
    /// [`crate::LoopClosureConstraint::relative_pose`]. If target keyframe `a`
    /// has `T_ca<-Wt`, source keyframe `b` has `T_cb<-Ws`, and the bridge is
    /// `z = T_cb<-ca`, then:
    /// `T_Wt<-Ws = T_ca<-Wt^-1 * z^-1 * T_cb<-Ws`.
    #[allow(clippy::too_many_arguments)]
    pub fn evidence_from_keyframe_bridge(
        &self,
        source_submap_id: SubmapId,
        source_frame_id: FrameId,
        target_submap_id: SubmapId,
        target_frame_id: FrameId,
        source_camera_from_target_camera: &SE3,
        quality: SubmapMergeQuality,
    ) -> Result<SubmapMergeEvidence, MapAtlasError> {
        let source = self
            .submaps
            .get(&source_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(source_submap_id))?;
        let target = self
            .submaps
            .get(&target_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(target_submap_id))?;
        let source_keyframe =
            source
                .map
                .keyframes
                .get(&source_frame_id)
                .ok_or(MapAtlasError::MissingKeyframe {
                    submap_id: source_submap_id,
                    frame_id: source_frame_id,
                })?;
        let target_keyframe =
            target
                .map
                .keyframes
                .get(&target_frame_id)
                .ok_or(MapAtlasError::MissingKeyframe {
                    submap_id: target_submap_id,
                    frame_id: target_frame_id,
                })?;
        let source_pose =
            source_keyframe
                .frame
                .pose
                .as_ref()
                .ok_or(MapAtlasError::MissingKeyframePose {
                    submap_id: source_submap_id,
                    frame_id: source_frame_id,
                })?;
        let target_pose =
            target_keyframe
                .frame
                .pose
                .as_ref()
                .ok_or(MapAtlasError::MissingKeyframePose {
                    submap_id: target_submap_id,
                    frame_id: target_frame_id,
                })?;
        let target_from_source = target_pose
            .world_to_camera
            .inverse()
            .compose(&source_camera_from_target_camera.inverse())
            .compose(&source_pose.world_to_camera);
        Ok(SubmapMergeEvidence {
            source_submap_id,
            target_submap_id,
            target_from_source,
            inlier_count: quality.inlier_count,
            inlier_ratio: quality.inlier_ratio,
            mean_reprojection_error_px: quality.mean_reprojection_error_px,
            estimated_scale: quality.estimated_scale,
        })
    }

    /// Verify a metric SE(3) edge and align `source` to an already-aligned
    /// `target`. A rejected edge leaves both submaps and Atlas transforms
    /// unchanged.
    pub fn verify_and_align(
        &mut self,
        evidence: SubmapMergeEvidence,
        config: &SubmapMergeVerificationConfig,
    ) -> Result<VerifiedSubmapMerge, MapAtlasError> {
        self.verify_and_align_with_landmark_matches(evidence, Vec::new(), config)
    }

    /// [`Self::verify_and_align`] with verified same-landmark relations that
    /// will be welded to one output id by [`Self::materialize_aligned`].
    pub fn verify_and_align_with_landmark_matches(
        &mut self,
        evidence: SubmapMergeEvidence,
        landmark_matches: Vec<CrossSubmapLandmarkMatch>,
        config: &SubmapMergeVerificationConfig,
    ) -> Result<VerifiedSubmapMerge, MapAtlasError> {
        if evidence.source_submap_id == evidence.target_submap_id {
            return Err(MapAtlasError::SameSubmapMerge(evidence.source_submap_id));
        }
        let source = self
            .submaps
            .get(&evidence.source_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(evidence.source_submap_id))?;
        let target = self
            .submaps
            .get(&evidence.target_submap_id)
            .ok_or(MapAtlasError::MissingSubmap(evidence.target_submap_id))?;
        for landmark_match in &landmark_matches {
            if !source
                .map
                .landmarks
                .contains_key(&landmark_match.source_landmark_id)
            {
                return Err(MapAtlasError::MissingLandmark {
                    submap_id: evidence.source_submap_id,
                    landmark_id: landmark_match.source_landmark_id,
                });
            }
            if !target
                .map
                .landmarks
                .contains_key(&landmark_match.target_landmark_id)
            {
                return Err(MapAtlasError::MissingLandmark {
                    submap_id: evidence.target_submap_id,
                    landmark_id: landmark_match.target_landmark_id,
                });
            }
            if !landmark_match.transformed_distance_m.is_finite()
                || landmark_match.transformed_distance_m < 0.0
            {
                return Err(MapAtlasError::NonFiniteLandmarkMatchDistance);
            }
        }
        let atlas_from_target =
            target
                .atlas_from_local
                .clone()
                .ok_or(MapAtlasError::TargetSubmapUnaligned(
                    evidence.target_submap_id,
                ))?;
        validate_merge_evidence(&evidence, config)?;

        let atlas_from_source = atlas_from_target.compose(&evidence.target_from_source);
        if let Some(existing) = source.atlas_from_local.as_ref() {
            let disagreement = atlas_from_source.compose(&existing.inverse());
            let translation_disagreement_m = disagreement.translation.norm();
            let rotation_disagreement_rad = disagreement.rotation.angle();
            if translation_disagreement_m > config.max_existing_translation_disagreement_m
                || rotation_disagreement_rad > config.max_existing_rotation_disagreement_rad
            {
                return Err(MapAtlasError::ExistingAlignmentInconsistent {
                    translation_disagreement_m,
                    rotation_disagreement_rad,
                });
            }
        }

        self.submaps
            .get_mut(&evidence.source_submap_id)
            .unwrap()
            .atlas_from_local = Some(atlas_from_source.clone());
        let verified = VerifiedSubmapMerge {
            evidence,
            atlas_from_source,
            landmark_matches,
        };
        self.verified_merges.push(verified.clone());
        Ok(verified)
    }

    /// Materialize all currently aligned submaps into one collision-free map.
    /// Independent/unverified submaps are intentionally omitted so unrelated
    /// coordinate frames can never leak into a flat-map consumer.
    pub fn materialize_aligned(&self) -> Result<MaterializedAtlas, MapAtlasError> {
        let mut output = VisualMap::new();
        let mut id_remaps = BTreeMap::new();
        let mut next_camera_id = 0u64;
        let mut next_frame_id = 0u64;
        let mut next_landmark_id = 0u64;
        let landmark_output_ids = aligned_landmark_output_ids(self, &mut next_landmark_id)?;

        for (&submap_id, submap) in &self.submaps {
            let Some(atlas_from_local) = submap.atlas_from_local.as_ref() else {
                continue;
            };
            let mut remap = SubmapIdRemap::default();

            let mut camera_ids = submap.map.cameras.keys().copied().collect::<Vec<_>>();
            camera_ids.sort_unstable();
            for old_id in camera_ids {
                let new_id = allocate_id(&mut next_camera_id)?;
                remap.camera_ids.insert(old_id, new_id);
                let mut camera = submap.map.cameras[&old_id].clone();
                camera.id = new_id;
                output.cameras.insert(new_id, camera);
            }

            let mut frame_ids = submap.map.keyframes.keys().copied().collect::<Vec<_>>();
            frame_ids.sort_unstable();
            for old_id in frame_ids {
                remap
                    .frame_ids
                    .insert(old_id, allocate_id(&mut next_frame_id)?);
            }
            let mut landmark_ids = submap.map.landmarks.keys().copied().collect::<Vec<_>>();
            landmark_ids.sort_unstable();
            for old_id in landmark_ids {
                let output_id = *landmark_output_ids.get(&(submap_id, old_id)).ok_or(
                    MapAtlasError::MissingLandmark {
                        submap_id,
                        landmark_id: old_id,
                    },
                )?;
                remap.landmark_ids.insert(old_id, output_id);
            }

            append_transformed_submap(&mut output, &submap.map, atlas_from_local, &remap);
            id_remaps.insert(submap_id, remap);
        }

        Ok(MaterializedAtlas {
            map: output,
            id_remaps,
        })
    }
}

fn validate_merge_evidence(
    evidence: &SubmapMergeEvidence,
    config: &SubmapMergeVerificationConfig,
) -> Result<(), MapAtlasError> {
    let finite_transform = evidence
        .target_from_source
        .translation
        .iter()
        .chain(evidence.target_from_source.rotation.coords.iter())
        .all(|value| value.is_finite());
    if !finite_transform {
        return Err(MapAtlasError::NonFiniteTransform);
    }
    if evidence.inlier_count < config.min_inliers {
        return Err(MapAtlasError::InsufficientInliers {
            actual: evidence.inlier_count,
            minimum: config.min_inliers,
        });
    }
    if !evidence.inlier_ratio.is_finite() || evidence.inlier_ratio < config.min_inlier_ratio {
        return Err(MapAtlasError::InlierRatioBelowMinimum {
            actual: evidence.inlier_ratio,
            minimum: config.min_inlier_ratio,
        });
    }
    if !evidence.mean_reprojection_error_px.is_finite()
        || evidence.mean_reprojection_error_px > config.max_mean_reprojection_error_px
    {
        return Err(MapAtlasError::ReprojectionErrorAboveMaximum {
            actual: evidence.mean_reprojection_error_px,
            maximum: config.max_mean_reprojection_error_px,
        });
    }
    if !evidence.estimated_scale.is_finite()
        || (evidence.estimated_scale - 1.0).abs() > config.max_metric_scale_error
    {
        return Err(MapAtlasError::MetricScaleInconsistent {
            actual: evidence.estimated_scale,
            maximum_error: config.max_metric_scale_error,
        });
    }
    Ok(())
}

fn estimate_boundary_rigid_transform_ransac(
    points: &[(Point3<f64>, Point3<f64>)],
    inlier_threshold_m: f64,
    iterations: usize,
) -> Option<(SE3, Vec<usize>)> {
    if points.len() < 3 || !inlier_threshold_m.is_finite() || inlier_threshold_m <= 0.0 {
        return None;
    }
    let fit = |indices: &[usize]| {
        let source = indices
            .iter()
            .map(|index| points[*index].0)
            .collect::<Vec<_>>();
        let target = indices
            .iter()
            .map(|index| points[*index].1)
            .collect::<Vec<_>>();
        let transform = visloc_tracking::umeyama_similarity_transform(&source, &target, false)?;
        Some(SE3::new(
            UnitQuaternion::from_rotation_matrix(&transform.rotation),
            transform.translation,
        ))
    };
    let score = |transform: &SE3| {
        let mut inliers = Vec::new();
        let mut residual_sum = 0.0;
        for (index, (source, target)) in points.iter().enumerate() {
            let residual = (transform.transform_point(source) - target).norm();
            if residual.is_finite() && residual <= inlier_threshold_m {
                inliers.push(index);
                residual_sum += residual;
            }
        }
        let mean = if inliers.is_empty() {
            f64::INFINITY
        } else {
            residual_sum / inliers.len() as f64
        };
        (inliers, mean)
    };

    let all = (0..points.len()).collect::<Vec<_>>();
    let mut best = fit(&all).map(|transform| {
        let (inliers, mean) = score(&transform);
        (transform, inliers, mean)
    });
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ points.len() as u64;
    for _ in 0..iterations.max(1) {
        let mut sample = [0usize; 3];
        for slot in 0..3 {
            let mut attempts = 0;
            loop {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let index = (state as usize) % points.len();
                if !sample[..slot].contains(&index) {
                    sample[slot] = index;
                    break;
                }
                attempts += 1;
                if attempts > points.len() * 2 {
                    break;
                }
            }
        }
        let Some(transform) = fit(&sample) else {
            continue;
        };
        let (inliers, mean) = score(&transform);
        let replace = best.as_ref().is_none_or(|(_, best_inliers, best_mean)| {
            inliers.len() > best_inliers.len()
                || (inliers.len() == best_inliers.len() && mean < *best_mean)
        });
        if replace {
            best = Some((transform, inliers, mean));
        }
    }
    let (_, best_inliers, _) = best?;
    if best_inliers.len() < 3 {
        return None;
    }
    let refined = fit(&best_inliers)?;
    let (refined_inliers, _) = score(&refined);
    (refined_inliers.len() >= 3).then_some((refined, refined_inliers))
}

fn estimate_cross_submap_scale(
    metric_points: &[(Point3<f64>, Point3<f64>)],
    config: &CrossSubmapAlignmentConfig,
) -> Result<CrossSubmapScaleEstimate, CrossSubmapCandidateFailureReason> {
    let point_count = metric_points.len().min(config.max_scale_points.max(2));
    let mut ratios = Vec::new();
    for left in 0..point_count {
        for right in (left + 1)..point_count {
            let source_distance = (metric_points[left].0 - metric_points[right].0).norm();
            let target_distance = (metric_points[left].1 - metric_points[right].1).norm();
            if !source_distance.is_finite()
                || !target_distance.is_finite()
                || source_distance < config.min_scale_pair_distance_m
                || target_distance < config.min_scale_pair_distance_m
            {
                continue;
            }
            let ratio = target_distance / source_distance;
            if ratio.is_finite() && ratio > 0.0 {
                ratios.push(ratio);
            }
        }
    }
    if ratios.len() < config.min_scale_pair_count {
        return Err(CrossSubmapCandidateFailureReason::TooFewScalePairs {
            actual: ratios.len(),
            minimum: config.min_scale_pair_count,
        });
    }
    ratios.sort_by(f64::total_cmp);
    let initial_median = median_sorted(&ratios);
    let tolerance = config.scale_inlier_relative_tolerance.max(0.0) * initial_median;
    let mut inlier_ratios = ratios
        .iter()
        .copied()
        .filter(|ratio| (*ratio - initial_median).abs() <= tolerance)
        .collect::<Vec<_>>();
    let inlier_ratio = inlier_ratios.len() as f64 / ratios.len() as f64;
    if inlier_ratios.is_empty() || inlier_ratio < config.min_scale_inlier_ratio {
        return Err(CrossSubmapCandidateFailureReason::LowScaleInlierRatio {
            actual: inlier_ratio,
            minimum: config.min_scale_inlier_ratio,
        });
    }
    inlier_ratios.sort_by(f64::total_cmp);
    let estimated_scale = median_sorted(&inlier_ratios);
    let mut deviations = inlier_ratios
        .iter()
        .map(|ratio| (*ratio - estimated_scale).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let median_absolute_deviation = median_sorted(&deviations);
    if median_absolute_deviation > config.max_scale_mad {
        return Err(
            CrossSubmapCandidateFailureReason::HighScaleMedianAbsoluteDeviation {
                actual: median_absolute_deviation,
                maximum: config.max_scale_mad,
            },
        );
    }
    Ok(CrossSubmapScaleEstimate {
        estimated_scale,
        pair_count: ratios.len(),
        inlier_count: inlier_ratios.len(),
        inlier_ratio,
        median_absolute_deviation,
    })
}

fn median_sorted(values: &[f64]) -> f64 {
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn allocate_id(next: &mut u64) -> Result<u64, MapAtlasError> {
    let id = *next;
    *next = next.checked_add(1).ok_or(MapAtlasError::IdSpaceExhausted)?;
    Ok(id)
}

type AtlasLandmarkKey = (SubmapId, LandmarkId);

fn aligned_landmark_output_ids(
    atlas: &MapAtlas,
    next_output_id: &mut u64,
) -> Result<BTreeMap<AtlasLandmarkKey, LandmarkId>, MapAtlasError> {
    let mut parent = BTreeMap::new();
    for (&submap_id, submap) in &atlas.submaps {
        if submap.atlas_from_local.is_none() {
            continue;
        }
        for &landmark_id in submap.map.landmarks.keys() {
            parent.insert((submap_id, landmark_id), (submap_id, landmark_id));
        }
    }
    for merge in &atlas.verified_merges {
        if atlas
            .submaps
            .get(&merge.evidence.source_submap_id)
            .and_then(|submap| submap.atlas_from_local.as_ref())
            .is_none()
            || atlas
                .submaps
                .get(&merge.evidence.target_submap_id)
                .and_then(|submap| submap.atlas_from_local.as_ref())
                .is_none()
        {
            continue;
        }
        for landmark_match in &merge.landmark_matches {
            let source_key = (
                merge.evidence.source_submap_id,
                landmark_match.source_landmark_id,
            );
            let target_key = (
                merge.evidence.target_submap_id,
                landmark_match.target_landmark_id,
            );
            if parent.contains_key(&source_key) && parent.contains_key(&target_key) {
                union_landmark_keys(&mut parent, source_key, target_key);
            }
        }
    }

    let keys = parent.keys().copied().collect::<Vec<_>>();
    let mut component_ids = BTreeMap::new();
    let mut output_ids = BTreeMap::new();
    for key in keys {
        let root = find_landmark_root(&parent, key);
        let output_id = match component_ids.get(&root) {
            Some(id) => *id,
            None => {
                let id = allocate_id(next_output_id)?;
                component_ids.insert(root, id);
                id
            }
        };
        output_ids.insert(key, output_id);
    }
    Ok(output_ids)
}

fn find_landmark_root(
    parent: &BTreeMap<AtlasLandmarkKey, AtlasLandmarkKey>,
    mut key: AtlasLandmarkKey,
) -> AtlasLandmarkKey {
    while let Some(&next) = parent.get(&key) {
        if next == key {
            break;
        }
        key = next;
    }
    key
}

fn union_landmark_keys(
    parent: &mut BTreeMap<AtlasLandmarkKey, AtlasLandmarkKey>,
    left: AtlasLandmarkKey,
    right: AtlasLandmarkKey,
) {
    let left_root = find_landmark_root(parent, left);
    let right_root = find_landmark_root(parent, right);
    if left_root == right_root {
        return;
    }
    let (root, child) = if left_root <= right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    parent.insert(child, root);
}

fn fuse_landmark_estimates(
    existing_position: Point3<f64>,
    existing_covariance: Option<&Matrix3<f64>>,
    source_position: Point3<f64>,
    source_covariance: Option<&Matrix3<f64>>,
    existing_observation_count: usize,
    source_observation_count: usize,
) -> (Point3<f64>, Option<Matrix3<f64>>) {
    match (existing_covariance, source_covariance) {
        (Some(existing_covariance), Some(source_covariance)) => {
            if let (Some(existing_information), Some(source_information)) = (
                existing_covariance.try_inverse(),
                source_covariance.try_inverse(),
            ) {
                let information = existing_information + source_information;
                if let Some(covariance) = information.try_inverse() {
                    let position = covariance
                        * (existing_information * existing_position.coords
                            + source_information * source_position.coords);
                    return (Point3::from(position), Some(covariance));
                }
            }
        }
        (Some(existing_covariance), None) => {
            return (existing_position, Some(*existing_covariance));
        }
        (None, Some(source_covariance)) => {
            return (source_position, Some(*source_covariance));
        }
        (None, None) => {}
    }

    let existing_weight = existing_observation_count.max(1) as f64;
    let source_weight = source_observation_count.max(1) as f64;
    let position = (existing_position.coords * existing_weight
        + source_position.coords * source_weight)
        / (existing_weight + source_weight);
    (
        Point3::from(position),
        existing_covariance
            .cloned()
            .or_else(|| source_covariance.cloned()),
    )
}

fn append_transformed_submap(
    output: &mut VisualMap,
    source: &VisualMap,
    atlas_from_local: &SE3,
    remap: &SubmapIdRemap,
) {
    let local_from_atlas = atlas_from_local.inverse();
    let rotation = atlas_from_local.rotation.to_rotation_matrix().into_inner();

    for (&old_id, &new_id) in &remap.landmark_ids {
        let source_landmark = &source.landmarks[&old_id];
        let transformed_position = atlas_from_local.transform_point(&source_landmark.position);
        let remapped_observations = source_landmark
            .observations
            .iter()
            .filter_map(|observation| remap_observation(observation, remap))
            .collect::<Vec<_>>();
        let transformed_covariance = source
            .landmark_position_covariances
            .get(&old_id)
            .map(|covariance| rotation * covariance * rotation.transpose());

        if output.landmarks.contains_key(&new_id) {
            let existing_covariance = output.landmark_position_covariances.get(&new_id).cloned();
            let existing = output.landmarks.get_mut(&new_id).unwrap();
            let existing_observation_count = existing.observations.len().max(1);
            let source_observation_count = remapped_observations.len().max(1);
            let (fused_position, fused_covariance) = fuse_landmark_estimates(
                existing.position,
                existing_covariance.as_ref(),
                transformed_position,
                transformed_covariance.as_ref(),
                existing_observation_count,
                source_observation_count,
            );
            existing.position = fused_position;
            if existing.descriptor.is_none() {
                existing.descriptor = source_landmark.descriptor.clone();
            }
            let mut known_observations = existing
                .observations
                .iter()
                .map(|observation| {
                    (
                        observation.frame_id,
                        observation.landmark_id,
                        observation.keypoint_index,
                    )
                })
                .collect::<HashSet<_>>();
            for observation in remapped_observations {
                if known_observations.insert((
                    observation.frame_id,
                    observation.landmark_id,
                    observation.keypoint_index,
                )) {
                    existing.observations.push(observation);
                }
            }
            if let Some(covariance) = fused_covariance {
                output
                    .landmark_position_covariances
                    .insert(new_id, covariance);
            }
        } else {
            let mut landmark = Landmark::new(new_id, transformed_position);
            landmark.descriptor = source_landmark.descriptor.clone();
            landmark.observations = remapped_observations;
            output.landmarks.insert(new_id, landmark);
            if let Some(covariance) = transformed_covariance {
                output
                    .landmark_position_covariances
                    .insert(new_id, covariance);
            }
        }
    }

    for (&old_id, &new_id) in &remap.frame_ids {
        let source_keyframe = &source.keyframes[&old_id];
        let mut frame = source_keyframe.frame.clone();
        frame.id = new_id;
        frame.camera_id = remap.camera_ids[&frame.camera_id];
        frame.pose = frame.pose.map(|pose| Pose {
            world_to_camera: pose.world_to_camera.compose(&local_from_atlas),
        });
        let mut observations = Vec::with_capacity(source_keyframe.observations.len());
        for source_observation in &source_keyframe.observations {
            if let Some(target_observation) = remap_observation(source_observation, remap) {
                if let Some(confidence) = source.observation_confidence(source_observation) {
                    output.set_observation_confidence(&target_observation, confidence);
                }
                observations.push(target_observation);
            }
        }
        output.keyframes.insert(
            new_id,
            Keyframe {
                frame,
                observations,
            },
        );
    }

    for stereo in &source.stereo_observations {
        let (Some(&frame_id), Some(&landmark_id), Some(&right_camera_id)) = (
            remap.frame_ids.get(&stereo.frame_id),
            remap.landmark_ids.get(&stereo.landmark_id),
            remap.camera_ids.get(&stereo.right_camera_id),
        ) else {
            continue;
        };
        output.stereo_observations.push(StereoObservation {
            frame_id,
            landmark_id,
            right_camera_id,
            xy_right: stereo.xy_right,
            left_to_right: stereo.left_to_right.clone(),
        });
    }
}

fn remap_observation(observation: &Observation, remap: &SubmapIdRemap) -> Option<Observation> {
    Some(Observation {
        frame_id: *remap.frame_ids.get(&observation.frame_id)?,
        landmark_id: *remap.landmark_ids.get(&observation.landmark_id)?,
        keypoint_index: observation.keypoint_index,
        xy: observation.xy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::types::{Camera, Frame};
    use visloc_vision::matching::BruteForceMatcher;

    fn one_point_map(position: Point3<f64>, covariance: Matrix3<f64>) -> VisualMap {
        let mut map = VisualMap::new();
        map.cameras
            .insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
        map.cameras
            .insert(2, Camera::pinhole(2, 640, 480, 500.0, 500.0, 320.0, 240.0));
        let observation = Observation {
            frame_id: 0,
            landmark_id: 0,
            keypoint_index: 0,
            xy: Point2::new(320.0, 240.0),
        };
        let mut frame = Frame::new(0, 1);
        frame.pose = Some(Pose::identity());
        frame.keypoints.push(observation.xy);
        let mut landmark = Landmark::new(0, position);
        landmark.observations.push(observation.clone());
        map.landmarks.insert(0, landmark);
        map.landmark_position_covariances.insert(0, covariance);
        map.keyframes.insert(
            0,
            Keyframe {
                frame,
                observations: vec![observation.clone()],
            },
        );
        map.set_observation_confidence(&observation, 0.75);
        map.stereo_observations.push(StereoObservation {
            frame_id: 0,
            landmark_id: 0,
            right_camera_id: 2,
            xy_right: Point2::new(300.0, 240.0),
            left_to_right: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.1, 0.0, 0.0)),
        });
        map
    }

    fn strong_evidence(source: SubmapId, target: SubmapId, transform: SE3) -> SubmapMergeEvidence {
        SubmapMergeEvidence {
            source_submap_id: source,
            target_submap_id: target,
            target_from_source: transform,
            inlier_count: 80,
            inlier_ratio: 0.8,
            mean_reprojection_error_px: 0.5,
            estimated_scale: 1.01,
        }
    }

    fn synthetic_revisit_map(frame_id: FrameId, camera: &Camera, world_scale: f64) -> VisualMap {
        let points = [
            Point3::new(-1.0, -0.8, 4.0),
            Point3::new(0.9, -0.7, 4.4),
            Point3::new(-0.8, 0.9, 4.8),
            Point3::new(1.1, 0.8, 5.2),
            Point3::new(-0.4, -0.2, 5.7),
            Point3::new(0.5, -0.3, 6.1),
            Point3::new(-0.6, 0.4, 6.6),
            Point3::new(0.7, 0.5, 7.0),
            Point3::new(-1.2, 0.1, 7.4),
            Point3::new(1.3, -0.1, 7.8),
            Point3::new(-0.2, 1.1, 8.2),
            Point3::new(0.3, -1.0, 8.6),
        ];
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        let mut frame = Frame::new(frame_id, camera.id);
        frame.pose = Some(Pose::identity());
        let mut observations = Vec::new();
        for (index, target_point) in points.into_iter().enumerate() {
            let source_point = Point3::from(target_point.coords * world_scale);
            let xy = camera.project(&source_point).unwrap();
            let mut descriptor = vec![0.0_f32; 12];
            descriptor[index] = 1.0;
            frame.keypoints.push(xy);
            frame.descriptors.push(descriptor.clone());
            let observation = Observation {
                frame_id,
                landmark_id: index as u64,
                keypoint_index: index,
                xy,
            };
            let mut landmark = Landmark::new(index as u64, source_point);
            landmark.descriptor = Some(descriptor);
            landmark.observations.push(observation.clone());
            map.landmarks.insert(landmark.id, landmark);
            observations.push(observation);
        }
        map.keyframes.insert(
            frame_id,
            Keyframe {
                frame,
                observations,
            },
        );
        map
    }

    fn synthetic_alignment_config() -> CrossSubmapAlignmentConfig {
        CrossSubmapAlignmentConfig {
            max_target_submaps: 2,
            min_candidate_landmarks: 8,
            min_scale_pair_count: 10,
            merge_verification: SubmapMergeVerificationConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_reprojection_error_px: 2.0,
                max_metric_scale_error: 0.1,
                ..SubmapMergeVerificationConfig::default()
            },
            ..CrossSubmapAlignmentConfig::default()
        }
    }

    #[test]
    fn independent_maps_remain_separate_until_verified_then_materialize_without_id_collisions() {
        let covariance = Matrix3::from_diagonal(&Vector3::new(1.0, 2.0, 3.0));
        let mut atlas = MapAtlas::new(one_point_map(Point3::new(1.0, 0.0, 5.0), covariance));
        let source =
            atlas.insert_independent(one_point_map(Point3::new(2.0, 0.0, 5.0), covariance));

        let before = atlas.materialize_aligned().unwrap();
        assert_eq!(before.map.keyframes.len(), 1);
        assert_eq!(before.map.landmarks.len(), 1);

        let target_from_source = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, std::f64::consts::FRAC_PI_2),
            Vector3::new(10.0, 0.0, 0.0),
        );
        atlas
            .verify_and_align(
                strong_evidence(source, atlas.root_submap_id().unwrap(), target_from_source),
                &SubmapMergeVerificationConfig::default(),
            )
            .unwrap();
        let merged = atlas.materialize_aligned().unwrap();

        assert_eq!(merged.map.cameras.len(), 4);
        assert_eq!(merged.map.keyframes.len(), 2);
        assert_eq!(merged.map.landmarks.len(), 2);
        assert_eq!(merged.map.stereo_observations.len(), 2);
        assert_eq!(merged.map.observation_confidence_count(), 2);
        assert!(merged.map.validate().is_valid());
        let root_remap = &merged.id_remaps[&0];
        let source_remap = &merged.id_remaps[&source];
        assert_ne!(root_remap.frame_ids[&0], source_remap.frame_ids[&0]);
        assert_ne!(root_remap.landmark_ids[&0], source_remap.landmark_ids[&0]);
        let source_landmark = &merged.map.landmarks[&source_remap.landmark_ids[&0]];
        assert!((source_landmark.position - Point3::new(10.0, 2.0, 5.0)).norm() < 1.0e-12);
        let source_covariance =
            &merged.map.landmark_position_covariances[&source_remap.landmark_ids[&0]];
        let expected_covariance = Matrix3::from_diagonal(&Vector3::new(2.0, 1.0, 3.0));
        assert!((source_covariance - expected_covariance).norm() < 1.0e-12);
        let source_pose = merged.map.keyframes[&source_remap.frame_ids[&0]]
            .frame
            .pose
            .as_ref()
            .unwrap();
        let expected_pose = Pose {
            world_to_camera: atlas
                .submap(source)
                .unwrap()
                .atlas_from_local
                .as_ref()
                .unwrap()
                .inverse(),
        };
        assert!(
            (source_pose.camera_center_world() - expected_pose.camera_center_world()).norm()
                < 1.0e-12
        );
    }

    #[test]
    fn appearance_pnp_and_metric_scale_align_an_independent_submap() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut atlas = MapAtlas::new(synthetic_revisit_map(0, &camera, 1.0));
        let source = atlas.insert_independent(synthetic_revisit_map(10, &camera, 1.0));
        let localizer = LocalizationPipeline::default();

        let result = atlas
            .align_submap_keyframe(
                source,
                10,
                &camera,
                None,
                &localizer,
                &synthetic_alignment_config(),
            )
            .unwrap();

        let verified = result.verified_merge.unwrap();
        assert_eq!(verified.evidence.source_submap_id, source);
        assert_eq!(verified.evidence.target_submap_id, 0);
        assert!(verified.evidence.inlier_count >= 8);
        assert!((verified.evidence.estimated_scale - 1.0).abs() < 1.0e-9);
        assert!(verified.atlas_from_source.translation.norm() < 1.0e-6);
        assert!(verified.atlas_from_source.rotation.angle() < 1.0e-6);
        assert!(verified.landmark_matches.len() >= 8);
        assert_eq!(atlas.aligned_submap_count(), 2);
        assert_eq!(result.diagnostics.len(), 1);
        assert!(result.diagnostics[0].failure_reason.is_none());
        let welded_count = verified.landmark_matches.len();
        let materialized = atlas.materialize_aligned().unwrap();
        assert_eq!(materialized.map.landmarks.len(), 24 - welded_count);
        assert_eq!(materialized.map.keyframes.len(), 2);
        for landmark_match in &verified.landmark_matches {
            assert_eq!(
                materialized.id_remaps[&source].landmark_ids[&landmark_match.source_landmark_id],
                materialized.id_remaps[&0].landmark_ids[&landmark_match.target_landmark_id],
            );
        }
    }

    #[test]
    fn adjacent_boundary_factor_recovers_metric_bridge() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut atlas = MapAtlas::new(synthetic_revisit_map(0, &camera, 1.0));
        let source = atlas.insert_independent(synthetic_revisit_map(10, &camera, 1.0));

        let result = atlas
            .align_adjacent_boundary_factor(
                source,
                10,
                0,
                0,
                None,
                &BruteForceMatcher::default(),
                &synthetic_alignment_config(),
            )
            .unwrap();

        assert_eq!(result.descriptor_match_count, 12);
        assert_eq!(result.metric_correspondence_count, 12);
        assert_eq!(result.rigid_inlier_count, 12);
        assert!((result.rigid_inlier_ratio - 1.0).abs() < 1.0e-12);
        assert!(result.mean_target_reprojection_error_px.unwrap() < 1.0e-9);
        assert!(result.failure_reason.is_none());
        assert!(result.verified_merge.is_some());
        assert_eq!(atlas.aligned_submap_count(), 2);
    }

    #[test]
    fn broader_source_window_recovers_with_retained_boundary_keyframe() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut atlas = MapAtlas::new(synthetic_revisit_map(0, &camera, 1.0));
        let source = atlas.insert_independent(synthetic_revisit_map(10, &camera, 1.0));
        let mut current = atlas.submap(source).unwrap().map.keyframes[&10].clone();
        current.frame.id = 20;
        current.frame.descriptors.clear();
        for observation in &mut current.observations {
            observation.frame_id = 20;
        }
        atlas
            .submap_mut(source)
            .unwrap()
            .map
            .keyframes
            .insert(20, current);
        let window = atlas
            .source_keyframe_recovery_window(source, 20, 4)
            .unwrap();
        assert_eq!(window, vec![20, 10]);

        let result = atlas
            .align_submap_keyframe_window(
                source,
                &window,
                &camera,
                Some(&SE3::identity()),
                &LocalizationPipeline::default(),
                &synthetic_alignment_config(),
            )
            .unwrap();

        assert_eq!(result.attempted_source_frame_ids, vec![20, 10]);
        assert_eq!(result.alignments.len(), 2);
        assert!(result.alignments[0].verified_merge.is_none());
        assert_eq!(result.alignments[0].ranked_candidate_count, 0);
        assert!(result.verified_merge.is_some());
        assert!(result.alignments[1].verified_merge.is_some());
        assert!(atlas.submap(source).unwrap().atlas_from_local.is_some());
    }

    #[test]
    fn pnp_consistent_but_scale_inconsistent_submap_is_not_aligned() {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut atlas = MapAtlas::new(synthetic_revisit_map(0, &camera, 1.0));
        // Scaling every source point about the camera leaves all pixels
        // unchanged, so PnP alone cannot detect this metric inconsistency.
        let source = atlas.insert_independent(synthetic_revisit_map(10, &camera, 1.5));
        let localizer = LocalizationPipeline::default();

        let result = atlas
            .align_submap_keyframe(
                source,
                10,
                &camera,
                None,
                &localizer,
                &synthetic_alignment_config(),
            )
            .unwrap();

        assert!(result.verified_merge.is_none());
        assert!(atlas.submap(source).unwrap().atlas_from_local.is_none());
        let diagnostic = result.diagnostics.first().unwrap();
        let scale = diagnostic.scale_estimate.unwrap();
        assert!((scale.estimated_scale - (2.0 / 3.0)).abs() < 1.0e-9);
        assert!(matches!(
            diagnostic.failure_reason,
            Some(CrossSubmapCandidateFailureReason::MergeRejected(
                MapAtlasError::MetricScaleInconsistent { .. }
            ))
        ));
    }

    #[test]
    fn verified_landmark_match_welds_geometry_observations_stereo_confidence_and_covariance() {
        let covariance = Matrix3::from_diagonal(&Vector3::new(2.0, 4.0, 6.0));
        let target_map = one_point_map(Point3::new(0.0, 0.0, 5.0), covariance);
        let source_map = one_point_map(Point3::new(0.0, 0.0, 5.0), covariance);
        let mut atlas = MapAtlas::new(target_map);
        let source = atlas.insert_independent(source_map);
        atlas
            .verify_and_align_with_landmark_matches(
                strong_evidence(source, 0, SE3::identity()),
                vec![CrossSubmapLandmarkMatch {
                    source_landmark_id: 0,
                    target_landmark_id: 0,
                    transformed_distance_m: 0.0,
                }],
                &SubmapMergeVerificationConfig::default(),
            )
            .unwrap();

        let materialized = atlas.materialize_aligned().unwrap();

        assert_eq!(materialized.map.landmarks.len(), 1);
        assert_eq!(materialized.map.keyframes.len(), 2);
        assert_eq!(materialized.map.stereo_observations.len(), 2);
        assert_eq!(materialized.map.observation_confidence_count(), 2);
        let landmark = materialized.map.landmarks.values().next().unwrap();
        assert_eq!(landmark.observations.len(), 2);
        let fused_covariance = materialized
            .map
            .landmark_position_covariances
            .values()
            .next()
            .unwrap();
        assert!((fused_covariance - covariance * 0.5).norm() < 1.0e-12);
        assert_eq!(
            materialized.id_remaps[&0].landmark_ids[&0],
            materialized.id_remaps[&source].landmark_ids[&0],
        );
    }

    #[test]
    fn scale_inconsistent_merge_is_rejected_without_aligning_source() {
        let mut atlas = MapAtlas::new(one_point_map(Point3::origin(), Matrix3::identity()));
        let source = atlas.insert_independent(one_point_map(Point3::origin(), Matrix3::identity()));
        let mut evidence = strong_evidence(source, 0, SE3::identity());
        evidence.estimated_scale = 1.5;

        assert!(matches!(
            atlas.verify_and_align(evidence, &SubmapMergeVerificationConfig::default()),
            Err(MapAtlasError::MetricScaleInconsistent { .. })
        ));
        assert!(atlas.submap(source).unwrap().atlas_from_local.is_none());
        assert_eq!(atlas.verified_merges().len(), 0);
    }

    #[test]
    fn verified_edges_compose_through_an_aligned_submap_chain() {
        let map = || one_point_map(Point3::origin(), Matrix3::identity());
        let mut atlas = MapAtlas::new(map());
        let middle = atlas.insert_independent(map());
        let source = atlas.insert_independent(map());
        atlas
            .verify_and_align(
                strong_evidence(
                    middle,
                    0,
                    SE3::new(UnitQuaternion::identity(), Vector3::new(10.0, 0.0, 0.0)),
                ),
                &SubmapMergeVerificationConfig::default(),
            )
            .unwrap();
        atlas
            .verify_and_align(
                strong_evidence(
                    source,
                    middle,
                    SE3::new(UnitQuaternion::identity(), Vector3::new(5.0, 0.0, 0.0)),
                ),
                &SubmapMergeVerificationConfig::default(),
            )
            .unwrap();

        let atlas_from_source = atlas
            .submap(source)
            .unwrap()
            .atlas_from_local
            .as_ref()
            .unwrap();
        assert!((atlas_from_source.translation - Vector3::new(15.0, 0.0, 0.0)).norm() < 1.0e-12);
        assert_eq!(atlas.aligned_submap_count(), 3);
        assert_eq!(atlas.independent_submap_count(), 0);
    }

    #[test]
    fn keyframe_bridge_recovers_the_target_from_source_map_transform() {
        let mut target_map = one_point_map(Point3::origin(), Matrix3::identity());
        target_map.keyframes.get_mut(&0).unwrap().frame.pose = Some(Pose::identity());
        let mut source_map = one_point_map(Point3::origin(), Matrix3::identity());
        // Source world is translated +10 m into target world. The same
        // physical camera at target-world origin therefore has T_c<-Ws = +10.
        source_map.keyframes.get_mut(&0).unwrap().frame.pose = Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(10.0, 0.0, 0.0),
        ));
        let mut atlas = MapAtlas::new(target_map);
        let source = atlas.insert_independent(source_map);
        let evidence = atlas
            .evidence_from_keyframe_bridge(
                source,
                0,
                0,
                0,
                &SE3::identity(),
                SubmapMergeQuality {
                    inlier_count: 80,
                    inlier_ratio: 0.8,
                    mean_reprojection_error_px: 0.5,
                    estimated_scale: 1.0,
                },
            )
            .unwrap();

        assert!(
            (evidence.target_from_source.translation - Vector3::new(10.0, 0.0, 0.0)).norm()
                < 1.0e-12
        );
        atlas
            .verify_and_align(evidence, &SubmapMergeVerificationConfig::default())
            .unwrap();
        assert_eq!(atlas.aligned_submap_count(), 2);
    }

    #[test]
    fn fallible_insert_rejects_submap_id_exhaustion_without_overwrite() {
        let mut atlas = MapAtlas::default();
        atlas.next_submap_id = u64::MAX;

        let result = atlas.try_insert_independent(VisualMap::new());

        assert_eq!(result, Err(MapAtlasError::IdSpaceExhausted));
        assert_eq!(atlas.active_submaps().count(), 0);
        assert_eq!(atlas.next_submap_id, u64::MAX);
    }
}
