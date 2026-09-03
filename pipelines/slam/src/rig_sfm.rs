//! Incremental structure-from-motion for a calibrated, synchronized camera rig.
//!
//! Each input frame owns one or more images with fixed `sensor <- rig`
//! extrinsics.  A frame is registered from all of its 2D-3D observations at
//! once with generalized PnP; image poses are derived from that single body
//! pose.  This preserves the physical stereo baseline during initialization
//! and avoids treating synchronized sensors as unrelated monocular cameras.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};

use nalgebra::{Matrix3, Point2, Point3, SMatrix, UnitQuaternion, Vector2, Vector3};
use thiserror::Error;
use visloc_core::geometry::{Pose, SE3};
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{
    GeneralizedCameraRig, GeneralizedCorrespondence2D3D, GeneralizedPnPRansac,
};
use visloc_vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};

use crate::bundle::{BaConfig, BaRigObservation, BundleAdjustment};
use crate::incremental_sfm::{
    build_tracks_confidence_ordered, build_tracks_confidence_ordered_with_trusted_prefix,
    build_tracks_detailed, build_tracks_incremental_correspondence,
    build_tracks_incremental_correspondence_in_order, PairwiseMatches, SfmTrack, TrackBuildOutput,
    TrackBuildStats,
};
use crate::{LinearSolver, RobustKernel};

/// One image and its calibrated sensor slot within a synchronized rig frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RigFrameImage {
    pub image_index: usize,
    pub sensor_index: usize,
}

/// Images captured at one timestamp and governed by one `world -> rig` pose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigFrame {
    pub images: Vec<RigFrameImage>,
}

/// Deterministic feature-track construction policy for rig reconstruction.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RigTrackBuilder {
    /// Historical transitive closure; rejects a whole component if one image
    /// contributes two observations.
    #[default]
    LegacyUnionFind,
    /// Reject only the conflicting edge, in physical endpoint order.
    ConflictPreserving,
    /// Reject only conflicting edges while preserving the verified input
    /// stream. This permits a checksummed trusted prefix followed by
    /// lower-priority component bridges.
    StreamOrderConflictPreserving,
    /// Reject only the conflicting edge, processing stronger verified image
    /// pairs before weaker pairs.
    PairConfidence,
    /// Process a frozen pair prefix in confidence order, then process all
    /// remaining pairs in confidence order without letting them displace
    /// trusted observations.
    TrustedPrefixPairConfidence(usize),
    /// Prefer correspondences that close through the same feature in one or
    /// more third images. Common neighbours are intersected from the sparse
    /// bounded view graph; no image Cartesian product is materialized.
    SparseCycle,
    /// Preserve metric stereo ownership first, then use sparse third-view
    /// cycle support and pair confidence for temporal extension.
    MetricSparseCycle,
    /// Preserve metric stereo anchors, then favor short temporal edges before
    /// cycle support so low-texture bridge chains are not consumed by stronger
    /// but more distant pair conflicts.
    MetricTemporalCycle,
    /// Emit only exact four-edge cycles between two synchronized stereo
    /// frames: stereo at each timestamp and temporal support on both sensors.
    /// Each result is an independent four-observation track; no transitive
    /// union-chain extension is performed.
    MetricTemporalQuadrilateral,
    /// Metric-first sparse cycles, retaining only tracks that contain a
    /// calibrated multi-sensor observation at one synchronized frame.
    MetricAnchoredCycle,
}

/// Conservative controls for generalized-rig incremental reconstruction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigSfmConfig {
    pub min_track_length: usize,
    pub min_pnp_inliers: usize,
    pub min_triangulation_angle_deg: f64,
    /// Admit a triangulated track when a majority of its currently registered
    /// observations agree, pruning only the inconsistent registered
    /// observations. Disabled by default until the quality A/B is promoted.
    pub robust_triangulation_pruning: bool,
    pub triangulation_min_inlier_fraction: f64,
    pub max_reprojection_error_px: f64,
    pub pnp_max_iterations: usize,
    pub ransac_seed: u64,
    /// Minimum number of distinct rig sensors contributing 2D-3D
    /// correspondences to a frame registration. One is geometrically valid
    /// because the calibrated sensor-to-rig transform is fixed.
    pub min_pnp_sensors: usize,
    /// Maximum rig-frame distance for a bounded stereo-to-temporal PnP
    /// fallback. Zero disables it. Each verified pair is consumed at most
    /// once, keeping work proportional to the sparse pair graph.
    pub direct_stereo_pnp_max_frame_gap: usize,
    /// Optional sensor-count gate used only when temporary direct-stereo 3-D
    /// points contribute to a PnP attempt. `None` preserves `min_pnp_sensors`.
    pub direct_stereo_min_pnp_sensors: Option<usize>,
    /// Minimum source-stereo ray angle used only by the direct PnP fallback.
    /// A separate gate allows long-depth rig points without weakening normal
    /// multi-view structure triangulation.
    pub direct_stereo_min_triangulation_angle_deg: f64,
    /// Maximum frame distance for an essential-rotation/constant-motion
    /// bridge. Zero disables it. Translation direction and magnitude always
    /// come from the preceding registered rig motion, never from the
    /// forward-motion-degenerate essential translation.
    pub motion_bridge_max_frame_gap: usize,
    pub motion_bridge_min_inliers: usize,
    pub motion_bridge_max_rotation_deviation_deg: f64,
    /// Split the verified-pair stream into a structure-building prefix and a
    /// deferred registration suffix. Suffix matches never merge or move the
    /// established structure: after BA and fixed-pose structure refinement,
    /// they may only register missing frames against existing landmarks and
    /// append PnP-inlier observations. This preserves the accepted base map
    /// while allowing a lower-priority retrieval pass to fill coverage gaps.
    pub deferred_registration_pair_prefix: Option<usize>,
    /// Optional start of a second, disjoint deferred tail used only for
    /// post-registration retriangulation. When absent, the registration tail
    /// is reused exactly as before. When present, pairs between the two
    /// prefixes remain available to registration but are excluded from the
    /// retriangulation builder.
    pub deferred_retriangulation_pair_prefix: Option<usize>,
    /// Optional PnP controls for the bounded deferred suffix. `None` inherits
    /// the corresponding primary-mapper setting. A single calibrated sensor
    /// is geometrically sufficient once metric 3D landmarks are fixed.
    pub deferred_registration_min_pnp_sensors: Option<usize>,
    pub deferred_registration_min_pnp_inliers: Option<usize>,
    pub deferred_registration_pnp_max_iterations: Option<usize>,
    /// Maximum bounded run of missing temporal frames that may be filled by
    /// SE(3) interpolation between registered endpoints. Zero disables it;
    /// every filled frame must also have a verified deferred-pair adjacency.
    pub deferred_registration_max_interpolation_gap: usize,
    /// Build an independent track graph from the deferred pair tail only
    /// after registration, then triangulate previously unowned tracks against
    /// the frozen poses. This mirrors COLMAP's post-registration
    /// retriangulation ordering without exposing the added tracks to PnP.
    pub retriangulate_deferred_tracks_after_registration: bool,
    /// Build only the deferred post-registration graph with metric-first,
    /// temporal-first sparse-cycle conflict handling. The registration graph
    /// and its track builder remain frozen.
    pub deferred_retriangulation_metric_temporal_cycle_tracks: bool,
    /// Build deferred tracks only from exact stereo-temporal quadrilaterals.
    /// This is stricter than temporal sparse-cycle ordering and remains
    /// independent of the reconstructed poses.
    pub deferred_retriangulation_metric_temporal_quadrilateral_tracks: bool,
    /// Inclusive temporal baseline accepted by the exact quadrilateral
    /// builder. Defaults to adjacent frames; wider ranges remain bounded by
    /// the supplied sparse verified-pair tail.
    pub deferred_retriangulation_quadrilateral_min_frame_gap: usize,
    pub deferred_retriangulation_quadrilateral_max_frame_gap: usize,
    /// Require a deferred track to contain synchronized multi-sensor support
    /// in at least this many distinct rig frames before post-registration
    /// triangulation. Zero preserves the historical deferred-track behavior.
    pub deferred_retriangulation_min_metric_frames: usize,
    pub track_builder: RigTrackBuilder,
    /// Recover at most one metric, cycle-supported 3-D hypothesis from each
    /// union-find component discarded for a duplicate-image conflict. The
    /// recovery uses only already-registered poses and verified edges.
    pub recover_metric_conflict_tracks: bool,
    pub conflict_recovery_max_hypotheses: usize,
    pub conflict_recovery_max_reprojection_error_px: f64,
    pub conflict_recovery_max_mean_reprojection_px: f64,
    /// COLMAP-style bounded correspondence-graph completion after all
    /// registrations. It never merges positioned tracks or mutates the PnP
    /// observation index; a fixed-pose structure pass consumes additions.
    pub complete_tracks_after_registration: bool,
    pub track_completion_max_passes: usize,
    pub track_completion_max_reprojection_error_px: f64,
    /// Replace a registered frame whose centre makes an implausible one-frame
    /// detour between two registered temporal neighbours. The test uses only
    /// reconstructed poses, runs in O(frames * passes), and is disabled by
    /// default because the absolute threshold is dataset-scale dependent.
    pub repair_isolated_pose_outliers: bool,
    pub isolated_pose_max_midpoint_error_m: f64,
    pub isolated_pose_min_detour_ratio: f64,
    pub isolated_pose_repair_max_passes: usize,
    /// Repair a short contiguous segment bracketed by large, mutually
    /// cancelling translation jumps. Internal segment motion is preserved;
    /// one world-frame offset is removed from every pose in the segment.
    pub repair_paired_pose_jumps: bool,
    pub paired_pose_jump_absolute_step_m: f64,
    pub paired_pose_jump_min_step_ratio: f64,
    pub paired_pose_jump_max_frame_span: usize,
    pub paired_pose_jump_max_closure_ratio: f64,
    /// Permit all tracks for triangulation/PnP, but let only tracks with a
    /// synchronized multi-sensor observation contribute pose-moving BA rows.
    pub ba_metric_tracks_only: bool,
    /// Keep an active rig pose fixed in BA when fewer than this many usable
    /// image observations constrain it inside the current window. Zero keeps
    /// the historical behavior. This prevents an already registered pose
    /// from becoming numerically free after landmark/reprojection filtering.
    pub final_ba_min_pose_observations: usize,
    pub final_bundle_adjustment: bool,
    pub ba_config: BaConfig,
    pub local_ba_every: usize,
    pub local_ba_window_size: usize,
    pub local_ba_iterations: usize,
    pub final_ba_passes: usize,
    pub final_ba_window_size: usize,
    pub final_ba_fix_window_ends: bool,
    /// Bounded COLMAP-style final refinement cycles. Each cycle removes
    /// registered observations outside `max_reprojection_error_px`, then
    /// reruns the existing windowed BA and fixed-pose structure refinement.
    /// Zero preserves the historical single-BA behavior.
    pub final_filter_refinement_passes: usize,
    /// Fixed-pose, per-landmark Gauss-Newton passes after registration.  This
    /// is linear in retained observations and never forms a global BA matrix.
    pub structure_refinement_iterations: usize,
}

impl Default for RigSfmConfig {
    fn default() -> Self {
        Self {
            min_track_length: 2,
            min_pnp_inliers: 8,
            min_triangulation_angle_deg: 1.0,
            robust_triangulation_pruning: false,
            triangulation_min_inlier_fraction: 0.5,
            max_reprojection_error_px: 4.0,
            pnp_max_iterations: 512,
            ransac_seed: 7,
            min_pnp_sensors: 2,
            direct_stereo_pnp_max_frame_gap: 0,
            direct_stereo_min_pnp_sensors: None,
            direct_stereo_min_triangulation_angle_deg: 1.0,
            motion_bridge_max_frame_gap: 0,
            motion_bridge_min_inliers: 12,
            motion_bridge_max_rotation_deviation_deg: 5.0,
            deferred_registration_pair_prefix: None,
            deferred_retriangulation_pair_prefix: None,
            deferred_registration_min_pnp_sensors: None,
            deferred_registration_min_pnp_inliers: None,
            deferred_registration_pnp_max_iterations: None,
            deferred_registration_max_interpolation_gap: 0,
            retriangulate_deferred_tracks_after_registration: false,
            deferred_retriangulation_metric_temporal_cycle_tracks: false,
            deferred_retriangulation_metric_temporal_quadrilateral_tracks: false,
            deferred_retriangulation_quadrilateral_min_frame_gap: 1,
            deferred_retriangulation_quadrilateral_max_frame_gap: 1,
            deferred_retriangulation_min_metric_frames: 0,
            track_builder: RigTrackBuilder::default(),
            recover_metric_conflict_tracks: false,
            conflict_recovery_max_hypotheses: 8,
            conflict_recovery_max_reprojection_error_px: 2.0,
            conflict_recovery_max_mean_reprojection_px: 1.0,
            complete_tracks_after_registration: false,
            track_completion_max_passes: 2,
            track_completion_max_reprojection_error_px: 1.0,
            repair_isolated_pose_outliers: false,
            isolated_pose_max_midpoint_error_m: 0.25,
            isolated_pose_min_detour_ratio: 8.0,
            isolated_pose_repair_max_passes: 1,
            repair_paired_pose_jumps: false,
            paired_pose_jump_absolute_step_m: 0.25,
            paired_pose_jump_min_step_ratio: 8.0,
            paired_pose_jump_max_frame_span: 16,
            paired_pose_jump_max_closure_ratio: 0.1,
            ba_metric_tracks_only: false,
            final_ba_min_pose_observations: 0,
            final_bundle_adjustment: true,
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::Huber { delta: 6.0 },
                parallel: true,
                ..BaConfig::default()
            },
            local_ba_every: 10,
            local_ba_window_size: 40,
            local_ba_iterations: 8,
            final_ba_passes: 2,
            final_ba_window_size: 60,
            final_ba_fix_window_ends: true,
            final_filter_refinement_passes: 0,
            structure_refinement_iterations: 5,
        }
    }
}

/// Output poses retain both the physical frame state and derived image states.
#[derive(Debug, Clone, PartialEq)]
pub struct RigSfmResult {
    pub frame_poses: Vec<Option<Pose>>,
    pub image_poses: Vec<Option<Pose>>,
    pub tracks: Vec<SfmTrack>,
    pub registered_frames: usize,
    pub registered_images: usize,
    pub mean_reprojection_error_px: f64,
    pub seed_frame_index: usize,
    pub track_build_stats: TrackBuildStats,
    pub work: RigSfmWorkStats,
    pub bundle_adjustment: Option<RigBaStats>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigBaStats {
    pub observations: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Counters used to prove that frontier growth stays proportional to sparse
/// observation support rather than rescanning the frame/track Cartesian
/// product.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RigSfmWorkStats {
    pub triangulation_attempts: usize,
    pub robust_triangulation_tracks: usize,
    pub robust_triangulation_pruned_observations: usize,
    pub robust_triangulation_majority_rejections: usize,
    pub correspondence_cache_insertions: usize,
    pub pnp_attempts: usize,
    pub pnp_insufficient_sensor_attempts: usize,
    pub pnp_estimation_failures: usize,
    pub pnp_inlier_rejections: usize,
    pub pnp_registrations: usize,
    pub direct_bridge_pair_visits: usize,
    pub direct_bridge_correspondence_insertions: usize,
    pub direct_bridge_registrations: usize,
    pub motion_bridge_pair_visits: usize,
    pub motion_bridge_estimation_failures: usize,
    pub motion_bridge_rotation_rejections: usize,
    pub motion_bridge_registrations: usize,
    pub deferred_pair_visits: usize,
    pub deferred_correspondence_insertions: usize,
    pub deferred_pnp_attempts: usize,
    pub deferred_pnp_estimation_failures: usize,
    pub deferred_pnp_inlier_rejections: usize,
    pub deferred_registrations: usize,
    pub deferred_interpolation_registrations: usize,
    pub deferred_observations_attached: usize,
    pub deferred_retriangulated_tracks: usize,
    pub deferred_retriangulated_observations: usize,
    pub unregistered_zero_support_frames: usize,
    pub unregistered_below_pnp_support_frames: usize,
    pub unregistered_eligible_pnp_frames: usize,
    pub unregistered_below_sensor_frames: usize,
    pub max_unregistered_support: usize,
    pub local_ba_runs: usize,
    pub ba_retriangulated_tracks: usize,
    pub ba_requeued_frames: usize,
    pub structure_refined_tracks: usize,
    pub geometry_recovered_tracks: usize,
    pub geometry_recovered_observations: usize,
    pub track_completion_passes: usize,
    pub track_completion_pair_visits: usize,
    pub track_completion_observations: usize,
    pub track_completion_reprojection_rejections: usize,
    pub final_filter_refinement_passes: usize,
    pub final_filter_refinement_pruned_observations: usize,
    pub isolated_pose_repair_passes: usize,
    pub isolated_pose_repairs: usize,
    pub paired_pose_jump_repairs: usize,
    pub paired_pose_jump_repaired_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum RigSfmError {
    #[error("rig reconstruction requires at least two calibrated sensors")]
    TooFewSensors,
    #[error("rig reconstruction contains no frames")]
    NoFrames,
    #[error(
        "minimum PnP sensor count {requested} is outside the calibrated range 1..={sensor_count}"
    )]
    InvalidMinPnpSensors {
        requested: usize,
        sensor_count: usize,
    },
    #[error("direct-stereo minimum PnP sensor count {requested} is outside 1..={sensor_count}")]
    InvalidDirectStereoMinPnpSensors {
        requested: usize,
        sensor_count: usize,
    },
    #[error("direct stereo triangulation angle must be finite and positive, found {0}")]
    InvalidDirectStereoTriangulationAngle(f64),
    #[error("triangulation inlier fraction must be finite and in [0.5, 1], found {0}")]
    InvalidTriangulationInlierFraction(f64),
    #[error("conflict recovery {name} must be finite and positive, found {value}")]
    InvalidConflictRecoveryGate { name: &'static str, value: f64 },
    #[error("motion bridge requires at least 8 essential inliers, found {0}")]
    InvalidMotionBridgeMinInliers(usize),
    #[error("motion bridge rotation deviation must be finite and positive, found {0}")]
    InvalidMotionBridgeRotationDeviation(f64),
    #[error("deferred registration pair prefix {prefix} exceeds {pair_count} verified pairs")]
    DeferredPairPrefix { prefix: usize, pair_count: usize },
    #[error(
        "deferred retriangulation pair prefix {prefix} is outside {registration_prefix}..={pair_count}"
    )]
    DeferredRetriangulationPairPrefix {
        prefix: usize,
        registration_prefix: usize,
        pair_count: usize,
    },
    #[error("deferred quadrilateral frame gap range is invalid: {min}..={max}")]
    InvalidDeferredQuadrilateralFrameGap { min: usize, max: usize },
    #[error("deferred minimum PnP sensor count {requested} is outside 1..={sensor_count}")]
    InvalidDeferredMinPnpSensors {
        requested: usize,
        sensor_count: usize,
    },
    #[error("deferred PnP requires at least one RANSAC iteration")]
    InvalidDeferredPnpIterations,
    #[error("deferred PnP requires at least 6 inliers, found {0}")]
    InvalidDeferredMinPnpInliers(usize),
    #[error("track completion requires at least one pass")]
    InvalidTrackCompletionPasses,
    #[error("track completion reprojection error must be finite and positive, found {0}")]
    InvalidTrackCompletionReprojectionError(f64),
    #[error("isolated-pose repair requires at least one pass")]
    InvalidIsolatedPoseRepairPasses,
    #[error("isolated-pose midpoint error must be finite and positive, found {0}")]
    InvalidIsolatedPoseMidpointError(f64),
    #[error("isolated-pose detour ratio must be finite and greater than one, found {0}")]
    InvalidIsolatedPoseDetourRatio(f64),
    #[error("paired-pose jump step must be finite and positive, found {0}")]
    InvalidPairedPoseJumpStep(f64),
    #[error("paired-pose jump step ratio must be finite and greater than one, found {0}")]
    InvalidPairedPoseJumpStepRatio(f64),
    #[error("paired-pose jump maximum frame span must be positive")]
    InvalidPairedPoseJumpFrameSpan,
    #[error("paired-pose jump closure ratio must be finite and in (0, 1), found {0}")]
    InvalidPairedPoseJumpClosureRatio(f64),
    #[error("frame {frame} contains no images")]
    EmptyFrame { frame: usize },
    #[error("image {image} is outside the feature range 0..{feature_count}")]
    ImageIndex { image: usize, feature_count: usize },
    #[error("sensor {sensor} is outside the calibrated range 0..{sensor_count}")]
    SensorIndex { sensor: usize, sensor_count: usize },
    #[error("image {image} occurs in more than one frame")]
    DuplicateImage { image: usize },
    #[error("sensor {sensor} occurs twice in frame {frame}")]
    DuplicateSensor { frame: usize, sensor: usize },
    #[error("image {image} is not assigned to a rig frame")]
    UnassignedImage { image: usize },
    #[error("invalid pair ({image_i}, {image_j}) for {feature_count} images")]
    PairImageIndex {
        image_i: usize,
        image_j: usize,
        feature_count: usize,
    },
    #[error("pair ({image_i}, {image_j}) references an invalid keypoint")]
    PairKeypointIndex { image_i: usize, image_j: usize },
    #[error("no frame has enough multi-sensor tracks for metric initialization")]
    NoMetricSeed,
    #[error("metric seed frame {frame} triangulated fewer than {required} landmarks")]
    InsufficientSeedStructure {
        frame: usize,
        required: usize,
        triangulated: usize,
    },
    #[error("rig bundle adjustment failed: {0}")]
    BundleAdjustment(String),
    #[error("fixed frame rotation count {actual} does not match the frame count {expected}")]
    InvalidFixedFrameRotationCount { expected: usize, actual: usize },
    #[error("rig refinement frame-pose count {actual} does not match the frame count {expected}")]
    InvalidResultFramePoseCount { expected: usize, actual: usize },
    #[error(
        "rig refinement image-pose count {actual} does not match the feature count {expected}"
    )]
    InvalidResultImagePoseCount { expected: usize, actual: usize },
    #[error("fixed frame rotation {frame} contains non-finite values")]
    NonFiniteFixedFrameRotation { frame: usize },
    #[error("fixed frame rotation {frame} is invalid")]
    InvalidFixedFrameRotation { frame: usize },
    #[error("rig refinement result contains non-finite state")]
    NonFiniteResultState,
    #[error("rig refinement result has invalid track observation {track} ({image}, {keypoint})")]
    InvalidResultTrackObservation {
        track: usize,
        image: usize,
        keypoint: usize,
    },
    #[error("rig refinement result has non-finite track state at index {track}")]
    NonFiniteResultTrack { track: usize },
    #[error("rig refinement seed frame {frame} is not registered")]
    InvalidResultSeedFrame { frame: usize },
    #[error("rig refinement requires at least one registered frame")]
    NoRegisteredFrames,
    #[error("rig refinement bundle adjustment had no usable observations")]
    NoBundleAdjustmentObservations,
}

#[derive(Debug, Clone)]
struct WorkingTrack {
    observations: Vec<(usize, usize)>,
    position: Option<Point3<f64>>,
    metric_anchored: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TemporalTrackSupportCount {
    tracks: usize,
    observations: usize,
    metric_tracks: usize,
}

fn temporal_track_span_class(span_frames: usize) -> &'static str {
    match span_frames {
        0 => "same-frame",
        1..=7 => "1-7",
        8..=15 => "8-15",
        16..=31 => "16-31",
        32..=127 => "32-127",
        _ => "128+",
    }
}

fn temporal_track_support<I>(
    tracks: I,
    image_assignment: &[(usize, usize)],
    bin_frames: usize,
) -> BTreeMap<(usize, &'static str), TemporalTrackSupportCount>
where
    I: IntoIterator<Item = Vec<usize>>,
{
    let mut counts = BTreeMap::new();
    for images in tracks {
        if images.is_empty() {
            continue;
        }
        let mut assignments = images
            .iter()
            .filter_map(|&image| image_assignment.get(image).copied())
            .collect::<Vec<_>>();
        if assignments.is_empty() {
            continue;
        }
        assignments.sort_unstable();
        let anchor_frame = assignments[assignments.len() / 2].0;
        let span_frames = assignments.last().unwrap().0 - assignments[0].0;
        let metric = assignments
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0 && pair[0].1 != pair[1].1);
        let count = counts
            .entry((
                anchor_frame / bin_frames,
                temporal_track_span_class(span_frames),
            ))
            .or_insert_with(TemporalTrackSupportCount::default);
        count.tracks += 1;
        count.observations += assignments.len();
        count.metric_tracks += usize::from(metric);
    }
    counts
}

fn log_temporal_track_support<I>(
    stage: &str,
    tracks: I,
    image_assignment: &[(usize, usize)],
    bin_frames: usize,
) where
    I: IntoIterator<Item = Vec<usize>>,
{
    for ((frame_bin, span_class), count) in
        temporal_track_support(tracks, image_assignment, bin_frames)
    {
        eprintln!(
            "rig-temporal-support: stage={stage} frame_start={} frame_end_exclusive={} span_class={span_class} tracks={} observations={} metric_tracks={}",
            frame_bin * bin_frames,
            (frame_bin + 1) * bin_frames,
            count.tracks,
            count.observations,
            count.metric_tracks,
        );
    }
}

fn temporal_support_bin_frames_from_env() -> Option<usize> {
    let value = std::env::var("VISLOC_SFM_TEMPORAL_SUPPORT_BIN_FRAMES").ok()?;
    match value.parse::<usize>() {
        Ok(value) if value > 0 => Some(value),
        _ => {
            eprintln!(
                "rig-temporal-support: disabled invalid VISLOC_SFM_TEMPORAL_SUPPORT_BIN_FRAMES={value:?}"
            );
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CachedRigCorrespondence {
    sensor_index: usize,
    point2d: Point2<f64>,
    track_index: Option<usize>,
    direct_point3d: Option<Point3<f64>>,
}

/// Reconstruct synchronized rig frames with one generalized body pose per
/// timestamp.  The first pose fixes the world gauge; the calibrated inter-
/// sensor baseline fixes metric scale.
pub fn incremental_rig_sfm(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
) -> Result<RigSfmResult, RigSfmError> {
    if rig.sensors().len() < 2 {
        return Err(RigSfmError::TooFewSensors);
    }
    if frames.is_empty() {
        return Err(RigSfmError::NoFrames);
    }
    validate_inputs(rig, frames, features, pairwise, config)?;

    let image_assignment = image_assignment(frames, features.len());
    let temporal_support_bin_frames = temporal_support_bin_frames_from_env();
    let registration_prefix = config
        .deferred_registration_pair_prefix
        .unwrap_or(pairwise.len());
    let mapping_pairwise = &pairwise[..registration_prefix];
    let deferred_pairwise = config
        .deferred_retriangulation_pair_prefix
        .map_or(&pairwise[registration_prefix..], |prefix| {
            &pairwise[registration_prefix..prefix]
        });
    let retriangulation_pairwise = config
        .deferred_retriangulation_pair_prefix
        .map_or(deferred_pairwise, |prefix| &pairwise[prefix..]);
    let track_output =
        build_rig_track_output(features, mapping_pairwise, &image_assignment, config);
    let track_build_stats = track_output.stats;
    let conflicting_components = track_output.conflicting_components;
    let raw_tracks = track_output.tracks;
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "constructed",
            raw_tracks
                .iter()
                .map(|track| track.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
        log_temporal_track_support(
            "conflicting",
            conflicting_components
                .iter()
                .map(|track| track.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
    }
    let mut tracks = raw_tracks
        .into_iter()
        .map(|observations| {
            let metric_anchored = track_is_metric_anchored(&observations, &image_assignment);
            WorkingTrack {
                observations,
                position: None,
                metric_anchored,
            }
        })
        .collect::<Vec<_>>();
    let mut image_tracks = image_track_index(features.len(), &tracks);
    for index in &mut image_tracks {
        index.sort_unstable_by_key(|&(keypoint, _)| keypoint);
    }
    let metric_supports = metric_frame_supports(frames.len(), &tracks, &image_assignment);
    let seed_candidates = metric_seed_candidates(&metric_supports);
    if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
        let metric_tracks = tracks.iter().filter(|track| track.metric_anchored).count();
        let max_frame_support = metric_supports.iter().copied().max().unwrap_or(0);
        eprintln!(
            "rig-sfm-debug: tracks={} metric_anchored_tracks={} metric_seed_candidates={} max_frame_support={max_frame_support}",
            tracks.len(),
            metric_tracks,
            seed_candidates.len(),
        );
    }
    let mut image_poses = vec![None; features.len()];
    let required_seed_landmarks = config.min_pnp_inliers.max(6);
    let mut total_seed_attempts = 0usize;
    let mut total_seed_robust_tracks = 0usize;
    let mut total_seed_pruned_observations = 0usize;
    let mut total_seed_majority_rejections = 0usize;
    let mut best_failed_seed = (
        seed_candidates
            .first()
            .copied()
            .ok_or(RigSfmError::NoMetricSeed)?,
        0usize,
    );
    let mut accepted_seed = None;
    for seed_frame_index in seed_candidates {
        let tracks_before_seed = config.robust_triangulation_pruning.then(|| tracks.clone());
        for track in &mut tracks {
            track.position = None;
        }
        image_poses.fill(None);
        install_image_poses(
            rig,
            &frames[seed_frame_index],
            &Pose::identity(),
            &mut image_poses,
        );
        let seed_frontier = frames[seed_frame_index]
            .images
            .iter()
            .flat_map(|image| {
                image_tracks[image.image_index]
                    .iter()
                    .map(|(_, track)| *track)
            })
            .collect::<HashSet<_>>();
        let triangulation = triangulate_frontier(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
            seed_frontier,
        );
        total_seed_attempts += triangulation.attempts;
        total_seed_robust_tracks += triangulation.robust_tracks;
        total_seed_pruned_observations += triangulation.pruned_observations;
        total_seed_majority_rejections += triangulation.majority_rejections;
        let seed_landmarks = triangulation.landmarks.len();
        if seed_landmarks > best_failed_seed.1 {
            best_failed_seed = (seed_frame_index, seed_landmarks);
        }
        if seed_landmarks >= required_seed_landmarks {
            accepted_seed = Some((seed_frame_index, triangulation.landmarks));
            break;
        }
        if let Some(tracks_before_seed) = tracks_before_seed {
            tracks = tracks_before_seed;
        }
    }
    let Some((seed_frame_index, seed_landmarks)) = accepted_seed else {
        return Err(RigSfmError::InsufficientSeedStructure {
            frame: best_failed_seed.0,
            required: required_seed_landmarks,
            triangulated: best_failed_seed.1,
        });
    };
    let seed_triangulation = TriangulationUpdate {
        landmarks: seed_landmarks,
        attempts: total_seed_attempts,
        robust_tracks: total_seed_robust_tracks,
        pruned_observations: total_seed_pruned_observations,
        majority_rejections: total_seed_majority_rejections,
    };
    let mut frame_poses = vec![None; frames.len()];
    frame_poses[seed_frame_index] = Some(Pose::identity());
    let mut registration_order = vec![seed_frame_index];

    let pnp = GeneralizedPnPRansac {
        iterations: config.pnp_max_iterations,
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };

    // Each landmark enters each observing frame's cache exactly once. Heap
    // versions make stale support counts cheap to discard, avoiding the
    // all-unregistered-frame rescan that becomes quadratic at 10k scale.
    let mut frame_correspondences: Vec<Vec<CachedRigCorrespondence>> =
        vec![Vec::new(); frames.len()];
    let mut frame_versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_heap = BinaryHeap::new();
    let mut work = RigSfmWorkStats {
        triangulation_attempts: seed_triangulation.attempts,
        robust_triangulation_tracks: seed_triangulation.robust_tracks,
        robust_triangulation_pruned_observations: seed_triangulation.pruned_observations,
        robust_triangulation_majority_rejections: seed_triangulation.majority_rejections,
        ..RigSfmWorkStats::default()
    };
    work.correspondence_cache_insertions += append_landmark_correspondences(
        &seed_triangulation.landmarks,
        &tracks,
        features,
        &image_assignment,
        &mut frame_correspondences,
        &mut frame_versions,
        &mut candidate_heap,
    );
    // A deferred pair is forbidden from changing the established track
    // structure, but it is still valid input to the opt-in direct bridge:
    // direct points are temporary source-stereo triangulations consumed by
    // robust PnP and never enter union-find or BA. This lets low-support local
    // edges cross a registration gap without letting them corrupt the map.
    let stereo_links = (config.direct_stereo_pnp_max_frame_gap > 0)
        .then(|| build_verified_stereo_links(pairwise, &image_assignment));
    let direct_pair_adjacency = (config.direct_stereo_pnp_max_frame_gap > 0)
        .then(|| build_frame_pair_adjacency(pairwise, &image_assignment, frames.len()));
    let motion_pair_adjacency = (config.motion_bridge_max_frame_gap > 0)
        .then(|| build_frame_pair_adjacency(mapping_pairwise, &image_assignment, frames.len()));
    let mut direct_pairs_visited = vec![false; pairwise.len()];
    let mut direct_target_observations = vec![HashSet::new(); frames.len()];
    let mut direct_source_queue = registration_order.clone();
    let mut motion_pairs_visited = vec![false; mapping_pairwise.len()];
    let mut motion_source_queue = registration_order.clone();
    let mut motion_pending = Vec::new();
    loop {
        while let Some((support, Reverse(frame), version)) = candidate_heap.pop() {
            if frame_poses[frame].is_some()
                || frame_versions[frame] != version
                || frame_correspondences[frame].len() != support
                || attempted_versions[frame] == Some(version)
                || support < config.min_pnp_inliers.max(6)
            {
                continue;
            }
            attempted_versions[frame] = Some(version);
            work.pnp_attempts += 1;
            let used_direct_bridge = frame_correspondences[frame]
                .iter()
                .any(|cached| cached.direct_point3d.is_some());
            let correspondences = frame_correspondences[frame]
                .iter()
                .filter_map(|cached| {
                    let point3d = cached
                        .direct_point3d
                        .or_else(|| cached.track_index.and_then(|track| tracks[track].position))?;
                    Some(GeneralizedCorrespondence2D3D {
                        sensor_index: cached.sensor_index,
                        point2d: cached.point2d,
                        point3d,
                        confidence: None,
                    })
                })
                .collect::<Vec<_>>();
            let distinct_sensors = correspondences
                .iter()
                .map(|correspondence| correspondence.sensor_index)
                .collect::<HashSet<_>>()
                .len();
            let required_sensors = if used_direct_bridge {
                config
                    .direct_stereo_min_pnp_sensors
                    .unwrap_or(config.min_pnp_sensors)
            } else {
                config.min_pnp_sensors
            };
            if distinct_sensors < required_sensors {
                work.pnp_insufficient_sensor_attempts += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=insufficient-sensors required={}",
                    correspondences.len(),
                        required_sensors,
                );
                }
                continue;
            }
            let Some(report) = pnp.estimate(rig, &correspondences) else {
                work.pnp_estimation_failures += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=estimation-failed",
                    correspondences.len(),
                );
                }
                continue;
            };
            if report.inliers.len() < config.min_pnp_inliers.max(6) {
                work.pnp_inlier_rejections += 1;
                if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
                    eprintln!(
                    "rig-sfm-debug: frame={frame} support={} sensors={distinct_sensors} pnp=inlier-rejected inliers={} required={}",
                    correspondences.len(),
                    report.inliers.len(),
                    config.min_pnp_inliers.max(6),
                );
                }
                continue;
            }
            work.pnp_registrations += 1;
            work.direct_bridge_registrations += usize::from(used_direct_bridge);
            frame_poses[frame] = Some(report.pose);
            registration_order.push(frame);
            direct_source_queue.push(frame);
            motion_source_queue.push(frame);
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame].as_ref().unwrap(),
                &mut image_poses,
            );
            let frontier = frames[frame]
                .images
                .iter()
                .flat_map(|image| {
                    image_tracks[image.image_index]
                        .iter()
                        .map(|(_, track)| *track)
                })
                .collect::<HashSet<_>>();
            let triangulation = triangulate_frontier(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
                frontier,
            );
            work.triangulation_attempts += triangulation.attempts;
            work.robust_triangulation_tracks += triangulation.robust_tracks;
            work.robust_triangulation_pruned_observations += triangulation.pruned_observations;
            work.robust_triangulation_majority_rejections += triangulation.majority_rejections;
            work.correspondence_cache_insertions += append_landmark_correspondences(
                &triangulation.landmarks,
                &tracks,
                features,
                &image_assignment,
                &mut frame_correspondences,
                &mut frame_versions,
                &mut candidate_heap,
            );
            if config.local_ba_every > 0
                && config.local_ba_window_size >= 2
                && registration_order.len() % config.local_ba_every == 0
            {
                let start = registration_order
                    .len()
                    .saturating_sub(config.local_ba_window_size);
                let active_frames = registration_order[start..]
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>();
                let anchor = registration_order[start];
                let local_ba_config = BaConfig {
                    max_iterations: config.local_ba_iterations,
                    ..config.ba_config
                };
                if run_rig_bundle_adjustment(
                    rig,
                    features,
                    &image_assignment,
                    config,
                    &active_frames,
                    anchor,
                    &local_ba_config,
                    0,
                    &[],
                    false,
                    &mut frame_poses,
                    &mut image_poses,
                    &mut tracks,
                )?
                .is_some()
                {
                    work.local_ba_runs += 1;
                    if config.ba_metric_tracks_only {
                        let affected_tracks = active_frames
                            .iter()
                            .flat_map(|frame| frames[*frame].images.iter())
                            .flat_map(|image| {
                                image_tracks[image.image_index]
                                    .iter()
                                    .map(|(_, track)| *track)
                            })
                            .collect::<HashSet<_>>();
                        let refreshed = retriangulate_unanchored_tracks(
                            rig,
                            features,
                            &image_assignment,
                            &image_poses,
                            config,
                            &mut tracks,
                            &affected_tracks,
                        );
                        work.ba_retriangulated_tracks += refreshed;
                        work.ba_requeued_frames += requeue_connected_frames(
                            &affected_tracks,
                            &tracks,
                            &image_assignment,
                            &frame_poses,
                            &frame_correspondences,
                            &mut frame_versions,
                            &mut candidate_heap,
                        );
                    }
                }
            }
        }
        if let Some(stereo_links) = stereo_links.as_ref() {
            let direct_pair_adjacency = direct_pair_adjacency
                .as_ref()
                .expect("direct pair adjacency accompanies stereo links");
            let direct = append_direct_stereo_pnp_correspondences(
                rig,
                frames,
                features,
                pairwise,
                &image_assignment,
                &image_poses,
                config,
                stereo_links,
                direct_pair_adjacency,
                &mut direct_source_queue,
                &mut direct_pairs_visited,
                &mut direct_target_observations,
                &mut frame_correspondences,
                &mut frame_versions,
                &mut candidate_heap,
            );
            work.direct_bridge_pair_visits += direct.pair_visits;
            work.direct_bridge_correspondence_insertions += direct.insertions;
            work.correspondence_cache_insertions += direct.insertions;
            if direct.insertions > 0 {
                continue;
            }
        }
        if let Some(adjacency) = motion_pair_adjacency.as_ref() {
            let update = collect_motion_bridge_candidates(
                rig,
                frames,
                features,
                mapping_pairwise,
                &image_assignment,
                &frame_poses,
                config,
                adjacency,
                &mut motion_source_queue,
                &mut motion_pairs_visited,
            );
            work.motion_bridge_pair_visits += update.pair_visits;
            work.motion_bridge_estimation_failures += update.estimation_failures;
            work.motion_bridge_rotation_rejections += update.rotation_rejections;
            motion_pending.extend(update.candidates);
            motion_pending.retain(|candidate| frame_poses[candidate.frame].is_none());
            motion_pending.sort_unstable_by(|left, right| {
                left.inliers
                    .cmp(&right.inliers)
                    .then_with(|| right.frame_gap.cmp(&left.frame_gap))
                    .then_with(|| right.frame.cmp(&left.frame))
            });
            if let Some(candidate) = motion_pending.pop() {
                let frame = candidate.frame;
                frame_poses[frame] = Some(candidate.pose);
                registration_order.push(frame);
                direct_source_queue.push(frame);
                motion_source_queue.push(frame);
                work.motion_bridge_registrations += 1;
                install_image_poses(
                    rig,
                    &frames[frame],
                    frame_poses[frame].as_ref().unwrap(),
                    &mut image_poses,
                );
                let frontier = frames[frame]
                    .images
                    .iter()
                    .flat_map(|image| {
                        image_tracks[image.image_index]
                            .iter()
                            .map(|(_, track)| *track)
                    })
                    .collect::<HashSet<_>>();
                let triangulation = triangulate_frontier(
                    rig,
                    features,
                    &image_assignment,
                    &image_poses,
                    config,
                    &mut tracks,
                    frontier,
                );
                work.triangulation_attempts += triangulation.attempts;
                work.robust_triangulation_tracks += triangulation.robust_tracks;
                work.robust_triangulation_pruned_observations += triangulation.pruned_observations;
                work.robust_triangulation_majority_rejections += triangulation.majority_rejections;
                work.correspondence_cache_insertions += append_landmark_correspondences(
                    &triangulation.landmarks,
                    &tracks,
                    features,
                    &image_assignment,
                    &mut frame_correspondences,
                    &mut frame_versions,
                    &mut candidate_heap,
                );
                continue;
            }
        }
        break;
    }

    let required_pnp_support = config.min_pnp_inliers.max(6);
    for (frame, pose) in frame_poses.iter().enumerate() {
        if pose.is_some() {
            continue;
        }
        let support = frame_correspondences[frame].len();
        work.max_unregistered_support = work.max_unregistered_support.max(support);
        if support == 0 {
            work.unregistered_zero_support_frames += 1;
        } else if support < required_pnp_support {
            work.unregistered_below_pnp_support_frames += 1;
        } else {
            work.unregistered_eligible_pnp_frames += 1;
        }
        let sensors = frame_correspondences[frame]
            .iter()
            .map(|correspondence| correspondence.sensor_index)
            .collect::<HashSet<_>>()
            .len();
        if sensors < config.min_pnp_sensors {
            work.unregistered_below_sensor_frames += 1;
        }
    }

    if config.recover_metric_conflict_tracks && !conflicting_components.is_empty() {
        let recovered = recover_metric_conflict_tracks(
            rig,
            features,
            mapping_pairwise,
            &conflicting_components,
            &image_assignment,
            &image_poses,
            config,
        );
        work.geometry_recovered_tracks = recovered.len();
        work.geometry_recovered_observations =
            recovered.iter().map(|track| track.observations.len()).sum();
        tracks.extend(recovered);
    }

    // Correct a short, mutually-cancelling pose jump before final BA can
    // deliberately freeze the affected poses through
    // `final_ba_min_pose_observations`.  This must run before the isolated
    // one-frame detector: repairing the pair first turns a two-frame gauge
    // discontinuity into ordinary local motion, and keeps the two detectors'
    // decisions deterministic.
    if config.repair_paired_pose_jumps {
        let repair = repair_paired_pose_jumps(
            rig,
            frames,
            seed_frame_index,
            config.paired_pose_jump_absolute_step_m,
            config.paired_pose_jump_min_step_ratio,
            config.paired_pose_jump_max_frame_span,
            config.paired_pose_jump_max_closure_ratio,
            &mut frame_poses,
            &mut image_poses,
        );
        work.paired_pose_jump_repairs += repair.repairs;
        work.paired_pose_jump_repaired_frames += repair.repaired_frames;
    }

    // Correct an unmistakable low-support PnP detour before final BA can
    // deliberately freeze that pose through `final_ba_min_pose_observations`.
    // A second pass after deferred registration handles poses added later.
    if config.repair_isolated_pose_outliers {
        let repair = repair_isolated_pose_outliers(
            rig,
            frames,
            seed_frame_index,
            config.isolated_pose_max_midpoint_error_m,
            config.isolated_pose_min_detour_ratio,
            config.isolated_pose_repair_max_passes,
            &mut frame_poses,
            &mut image_poses,
        );
        work.isolated_pose_repair_passes += repair.passes;
        work.isolated_pose_repairs += repair.repairs;
    }

    let mut bundle_adjustment = if config.final_bundle_adjustment && config.final_ba_passes > 0 {
        run_windowed_final_ba(
            rig,
            features,
            &image_assignment,
            config,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
        )?
    } else {
        None
    };

    work.structure_refined_tracks = refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &image_poses,
        config,
        &mut tracks,
    );

    if !deferred_pairwise.is_empty() {
        let deferred_stereo_links = build_relevant_verified_stereo_links(
            mapping_pairwise,
            deferred_pairwise,
            &image_assignment,
        );
        let deferred = register_deferred_frames(
            rig,
            frames,
            features,
            deferred_pairwise,
            &image_assignment,
            config,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
            &mut image_tracks,
            &deferred_stereo_links,
        );
        work.deferred_pair_visits = deferred.pair_visits;
        work.deferred_correspondence_insertions = deferred.correspondence_insertions;
        work.deferred_pnp_attempts = deferred.pnp_attempts;
        work.deferred_pnp_estimation_failures = deferred.pnp_estimation_failures;
        work.deferred_pnp_inlier_rejections = deferred.pnp_inlier_rejections;
        work.deferred_registrations = deferred.registrations;
        work.deferred_interpolation_registrations = deferred.interpolation_registrations;
        work.deferred_observations_attached = deferred.observations_attached;
    }

    if config.repair_isolated_pose_outliers {
        let repair = repair_isolated_pose_outliers(
            rig,
            frames,
            seed_frame_index,
            config.isolated_pose_max_midpoint_error_m,
            config.isolated_pose_min_detour_ratio,
            config.isolated_pose_repair_max_passes,
            &mut frame_poses,
            &mut image_poses,
        );
        work.isolated_pose_repair_passes += repair.passes;
        work.isolated_pose_repairs += repair.repairs;
        if repair.repairs > 0 {
            work.structure_refined_tracks += refine_rig_structure(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
            );
        }
    }

    if config.retriangulate_deferred_tracks_after_registration
        && !retriangulation_pairwise.is_empty()
    {
        let recovered = append_retriangulated_deferred_tracks(
            rig,
            features,
            retriangulation_pairwise,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
        );
        work.deferred_retriangulated_tracks = recovered.tracks;
        work.deferred_retriangulated_observations = recovered.observations;
    }

    if config.complete_tracks_after_registration {
        let completion = complete_positioned_tracks(
            rig,
            features,
            pairwise,
            &image_assignment,
            &image_poses,
            config.track_completion_max_passes,
            config.track_completion_max_reprojection_error_px,
            &mut tracks,
        );
        work.track_completion_passes = completion.passes;
        work.track_completion_pair_visits = completion.pair_visits;
        work.track_completion_observations = completion.observations;
        work.track_completion_reprojection_rejections = completion.reprojection_rejections;
        if completion.observations > 0 {
            work.structure_refined_tracks += refine_rig_structure(
                rig,
                features,
                &image_assignment,
                &image_poses,
                config,
                &mut tracks,
            );
        }
    }
    for _ in 0..config.final_filter_refinement_passes {
        let pruned = filter_positioned_track_observations(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config.max_reprojection_error_px,
            &mut tracks,
        );
        if pruned == 0 {
            break;
        }
        work.final_filter_refinement_passes += 1;
        work.final_filter_refinement_pruned_observations += pruned;
        if config.final_bundle_adjustment && config.final_ba_passes > 0 {
            let refinement = run_windowed_final_ba(
                rig,
                features,
                &image_assignment,
                config,
                &mut frame_poses,
                &mut image_poses,
                &mut tracks,
            )?;
            merge_rig_ba_stats(&mut bundle_adjustment, refinement);
        }
        work.structure_refined_tracks += refine_rig_structure(
            rig,
            features,
            &image_assignment,
            &image_poses,
            config,
            &mut tracks,
        );
    }
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "positioned",
            tracks
                .iter()
                .filter(|track| track.position.is_some())
                .map(|track| track.observations.iter().map(|&(image, _)| image).collect()),
            &image_assignment,
            bin_frames,
        );
    }
    let sfm_tracks = tracks
        .into_iter()
        .filter_map(|track| {
            let position = track.position?;
            let observations = track
                .observations
                .into_iter()
                .filter(|(image, keypoint)| {
                    let Some(pose) = image_poses[*image].as_ref() else {
                        return false;
                    };
                    let sensor = &rig.sensors()[image_assignment[*image].1];
                    sensor
                        .camera
                        .project(&pose.transform_world_point(&position))
                        .is_some_and(|projected| {
                            (projected - features[*image].keypoints[*keypoint]).norm()
                                <= config.max_reprojection_error_px
                        })
                })
                .map(|(image, keypoint)| (image, keypoint, features[image].keypoints[keypoint]))
                .collect::<Vec<_>>();
            (observations.len() >= 2).then_some(SfmTrack {
                position,
                observations,
            })
        })
        .collect::<Vec<_>>();
    if let Some(bin_frames) = temporal_support_bin_frames {
        log_temporal_track_support(
            "published",
            sfm_tracks.iter().map(|track| {
                track
                    .observations
                    .iter()
                    .map(|&(image, _, _)| image)
                    .collect()
            }),
            &image_assignment,
            bin_frames,
        );
    }
    let (error_sum, error_count) =
        reprojection_error(rig, &image_assignment, &image_poses, &sfm_tracks);

    Ok(RigSfmResult {
        registered_frames: frame_poses.iter().filter(|pose| pose.is_some()).count(),
        registered_images: image_poses.iter().filter(|pose| pose.is_some()).count(),
        mean_reprojection_error_px: if error_count == 0 {
            0.0
        } else {
            error_sum / error_count as f64
        },
        frame_poses,
        image_poses,
        tracks: sfm_tracks,
        seed_frame_index,
        track_build_stats,
        work,
        bundle_adjustment,
    })
}

/// Refine an existing rig reconstruction while holding the supplied frame
/// rotations fixed.
///
/// `desired_rotations` is indexed by rig frame (not image).  A `Some` entry
/// replaces the world-to-rig/world-to-camera rotation of a registered frame;
/// its camera centre is retained when that replacement is installed.  `None`
/// entries and unregistered frames retain their existing rotations.  The
/// subsequent sparse rig BA includes every registered frame and fixes every
/// active rotation, while translations and landmarks remain free.  A final
/// fixed-pose structure refinement updates the landmarks without changing the
/// frame poses.
///
/// The operation is transactional: validation failures, solver failures, and
/// non-finite candidate states restore `result` byte-for-byte (as represented
/// by its clone) and return an error.
pub fn refine_rig_sfm_with_fixed_frame_rotations(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<(), RigSfmError> {
    let result_before = result.clone();
    match refine_rig_sfm_with_fixed_frame_rotations_impl(
        rig,
        frames,
        features,
        desired_rotations,
        config,
        result,
    ) {
        Ok(()) => Ok(()),
        Err(error) => {
            *result = result_before;
            Err(error)
        }
    }
}

fn refine_rig_sfm_with_fixed_frame_rotations_impl(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<(), RigSfmError> {
    validate_fixed_frame_rotation_refinement_inputs(
        rig,
        frames,
        features,
        desired_rotations,
        config,
        result,
    )?;

    let image_assignment = image_assignment(frames, features.len());
    let active_frames = result
        .frame_poses
        .iter()
        .enumerate()
        .filter_map(|(frame, pose)| pose.as_ref().map(|_| frame))
        .collect::<HashSet<_>>();
    if active_frames.is_empty() {
        return Err(RigSfmError::NoRegisteredFrames);
    }
    if !active_frames.contains(&result.seed_frame_index) {
        return Err(RigSfmError::InvalidResultSeedFrame {
            frame: result.seed_frame_index,
        });
    }

    let mut frame_poses = result.frame_poses.clone();
    let mut image_poses = result.image_poses.clone();
    for (frame, desired) in desired_rotations.iter().enumerate() {
        let (Some(frame_pose), Some(desired)) = (frame_poses[frame].as_mut(), desired.as_ref())
        else {
            continue;
        };
        let centre = frame_pose.camera_center_world();
        frame_pose.world_to_camera.rotation = *desired;
        frame_pose.world_to_camera.translation = -desired.transform_vector(&centre.coords);
    }
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);

    let mut tracks = working_tracks_from_result(result, &image_assignment)?;
    let ba_stats = run_rig_bundle_adjustment(
        rig,
        features,
        &image_assignment,
        config,
        &active_frames,
        result.seed_frame_index,
        &config.ba_config,
        config.final_ba_min_pose_observations,
        &[],
        true,
        &mut frame_poses,
        &mut image_poses,
        &mut tracks,
    )?
    .ok_or(RigSfmError::NoBundleAdjustmentObservations)?;
    if !ba_stats.initial_cost.is_finite() || !ba_stats.final_cost.is_finite() {
        return Err(RigSfmError::NonFiniteResultState);
    }

    // `run_rig_bundle_adjustment` derives image poses after its write-back;
    // keep this explicit here because the following fixed-pose structure pass
    // and the public track conversion both consume the derived image states.
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);
    let mut registered_image_poses = image_poses.clone();
    for (image, &(frame, _)) in image_assignment.iter().enumerate() {
        if frame_poses[frame].is_none() {
            registered_image_poses[image] = None;
        }
    }
    let structure_refined = refine_rig_structure(
        rig,
        features,
        &image_assignment,
        &registered_image_poses,
        config,
        &mut tracks,
    );
    regenerate_registered_image_poses(rig, frames, &frame_poses, &mut image_poses);

    if !working_track_state_is_finite(&tracks) {
        return Err(RigSfmError::NonFiniteResultState);
    }

    let sfm_tracks = public_tracks_from_working_tracks(
        rig,
        features,
        &image_assignment,
        &frame_poses,
        &image_poses,
        config,
        &tracks,
    );
    let (error_sum, error_count) = reprojection_error_registered(
        rig,
        &image_assignment,
        &frame_poses,
        &image_poses,
        &sfm_tracks,
    );
    let mean_reprojection_error_px = if error_count == 0 {
        0.0
    } else {
        error_sum / error_count as f64
    };
    if !mean_reprojection_error_px.is_finite()
        || !refinement_state_is_finite(&frame_poses, &image_poses, &sfm_tracks, Some(&ba_stats))
    {
        return Err(RigSfmError::NonFiniteResultState);
    }

    result.frame_poses = frame_poses;
    result.image_poses = image_poses;
    result.tracks = sfm_tracks;
    result.registered_frames = result
        .frame_poses
        .iter()
        .filter(|pose| pose.is_some())
        .count();
    result.registered_images = result
        .frame_poses
        .iter()
        .enumerate()
        .filter(|(frame, pose)| {
            pose.is_some()
                && frames[*frame]
                    .images
                    .iter()
                    .all(|image| result.image_poses[image.image_index].is_some())
        })
        .map(|frame| frames[frame.0].images.len())
        .sum();
    result.mean_reprojection_error_px = mean_reprojection_error_px;
    result.bundle_adjustment = Some(ba_stats);
    result.work.structure_refined_tracks = result
        .work
        .structure_refined_tracks
        .saturating_add(structure_refined);
    Ok(())
}

fn validate_fixed_frame_rotation_refinement_inputs(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    desired_rotations: &[Option<UnitQuaternion<f64>>],
    config: &RigSfmConfig,
    result: &RigSfmResult,
) -> Result<(), RigSfmError> {
    if rig.sensors().len() < 2 {
        return Err(RigSfmError::TooFewSensors);
    }
    if frames.is_empty() {
        return Err(RigSfmError::NoFrames);
    }
    if desired_rotations.len() != frames.len() {
        return Err(RigSfmError::InvalidFixedFrameRotationCount {
            expected: frames.len(),
            actual: desired_rotations.len(),
        });
    }
    if result.frame_poses.len() != frames.len() {
        return Err(RigSfmError::InvalidResultFramePoseCount {
            expected: frames.len(),
            actual: result.frame_poses.len(),
        });
    }
    if result.image_poses.len() != features.len() {
        return Err(RigSfmError::InvalidResultImagePoseCount {
            expected: features.len(),
            actual: result.image_poses.len(),
        });
    }
    // Reuse the mapper's complete rig/frame/feature validation.  There are no
    // pairwise inputs in this refinement API, so an empty pair stream is
    // sufficient here.  The mapper-only deferred prefix is deliberately
    // disabled for this validation copy: with no pair stream to partition, a
    // perfectly valid mapper config would otherwise fail merely because its
    // prefix is nonzero.
    let validation_config = RigSfmConfig {
        deferred_registration_pair_prefix: None,
        ..*config
    };
    validate_inputs(rig, frames, features, &[], &validation_config)?;

    if features.iter().any(|feature_set| {
        feature_set
            .keypoints
            .iter()
            .any(|point| !point.coords.iter().all(|value| value.is_finite()))
    }) {
        return Err(RigSfmError::NonFiniteResultState);
    }

    for (frame, desired) in desired_rotations.iter().enumerate() {
        let Some(desired) = desired else {
            continue;
        };
        let norm_squared = desired.coords.norm_squared();
        if !desired.coords.iter().all(|value| value.is_finite()) || !norm_squared.is_finite() {
            return Err(RigSfmError::NonFiniteFixedFrameRotation { frame });
        }
        if norm_squared <= f64::EPSILON {
            return Err(RigSfmError::InvalidFixedFrameRotation { frame });
        }
    }
    if !result.mean_reprojection_error_px.is_finite()
        || result
            .bundle_adjustment
            .as_ref()
            .is_some_and(|stats| !stats.initial_cost.is_finite() || !stats.final_cost.is_finite())
        || result.frame_poses.iter().any(|pose| {
            pose.as_ref()
                .is_some_and(|pose| !pose_state_is_finite(pose))
        })
        || result.image_poses.iter().any(|pose| {
            pose.as_ref()
                .is_some_and(|pose| !pose_state_is_finite(pose))
        })
    {
        return Err(RigSfmError::NonFiniteResultState);
    }
    for (track, sfm_track) in result.tracks.iter().enumerate() {
        if !sfm_track
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(RigSfmError::NonFiniteResultTrack { track });
        }
        for &(image, keypoint, pixel) in &sfm_track.observations {
            if image >= features.len() || keypoint >= features[image].keypoints.len() {
                return Err(RigSfmError::InvalidResultTrackObservation {
                    track,
                    image,
                    keypoint,
                });
            }
            if !pixel.coords.iter().all(|value| value.is_finite()) {
                return Err(RigSfmError::NonFiniteResultTrack { track });
            }
        }
    }
    Ok(())
}

fn working_tracks_from_result(
    result: &RigSfmResult,
    image_assignment: &[(usize, usize)],
) -> Result<Vec<WorkingTrack>, RigSfmError> {
    result
        .tracks
        .iter()
        .enumerate()
        .map(|(track, sfm_track)| {
            let observations = sfm_track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect::<Vec<_>>();
            if observations
                .iter()
                .any(|&(image, _)| image >= image_assignment.len())
            {
                return Err(RigSfmError::InvalidResultTrackObservation {
                    track,
                    image: observations
                        .iter()
                        .find(|&&(image, _)| image >= image_assignment.len())
                        .map(|&(image, _)| image)
                        .unwrap_or(usize::MAX),
                    keypoint: usize::MAX,
                });
            }
            Ok(WorkingTrack {
                metric_anchored: track_is_metric_anchored(&observations, image_assignment),
                observations,
                position: Some(sfm_track.position),
            })
        })
        .collect()
}

fn regenerate_registered_image_poses(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    frame_poses: &[Option<Pose>],
    image_poses: &mut [Option<Pose>],
) {
    for (frame, frame_pose) in frame_poses.iter().enumerate() {
        let Some(frame_pose) = frame_pose.as_ref() else {
            continue;
        };
        install_image_poses(rig, &frames[frame], frame_pose, image_poses);
    }
}

fn public_tracks_from_working_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &[WorkingTrack],
) -> Vec<SfmTrack> {
    tracks
        .iter()
        .filter_map(|track| {
            let position = track.position?;
            let observations = track
                .observations
                .iter()
                .filter(|&&(image, keypoint)| {
                    let frame = image_assignment[image].0;
                    if frame_poses[frame].is_none() || image_poses[image].is_none() {
                        return false;
                    }
                    let sensor = &rig.sensors()[image_assignment[image].1];
                    sensor
                        .camera
                        .project(
                            &image_poses[image]
                                .as_ref()
                                .unwrap()
                                .transform_world_point(&position),
                        )
                        .is_some_and(|projected| {
                            (projected - features[image].keypoints[keypoint]).norm()
                                <= config.max_reprojection_error_px
                        })
                })
                .map(|&(image, keypoint)| (image, keypoint, features[image].keypoints[keypoint]))
                .collect::<Vec<_>>();
            (observations.len() >= 2).then_some(SfmTrack {
                position,
                observations,
            })
        })
        .collect()
}

fn reprojection_error_registered(
    rig: &GeneralizedCameraRig,
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for track in tracks {
        for (image, _, pixel) in &track.observations {
            let frame = image_assignment[*image].0;
            if frame_poses[frame].is_none() {
                continue;
            }
            let Some(pose) = image_poses[*image].as_ref() else {
                continue;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            if let Some(projected) = sensor
                .camera
                .project(&pose.transform_world_point(&track.position))
            {
                let error = (projected - pixel).norm();
                if error.is_finite() {
                    sum += error;
                    count += 1;
                }
            }
        }
    }
    (sum, count)
}

fn refinement_state_is_finite(
    frame_poses: &[Option<Pose>],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
    ba_stats: Option<&RigBaStats>,
) -> bool {
    frame_poses
        .iter()
        .all(|pose| pose.as_ref().is_none_or(pose_state_is_finite))
        && image_poses
            .iter()
            .all(|pose| pose.as_ref().is_none_or(pose_state_is_finite))
        && tracks.iter().all(|track| {
            track.position.coords.iter().all(|value| value.is_finite())
                && track
                    .observations
                    .iter()
                    .all(|(_, _, pixel)| pixel.coords.iter().all(|value| value.is_finite()))
        })
        && ba_stats
            .is_none_or(|stats| stats.initial_cost.is_finite() && stats.final_cost.is_finite())
}

fn pose_state_is_finite(pose: &Pose) -> bool {
    let rotation_norm_squared = pose.world_to_camera.rotation.coords.norm_squared();
    pose.world_to_camera
        .rotation
        .coords
        .iter()
        .all(|value| value.is_finite())
        && rotation_norm_squared.is_finite()
        && rotation_norm_squared > f64::EPSILON
        && pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
}

fn working_track_state_is_finite(tracks: &[WorkingTrack]) -> bool {
    tracks.iter().all(|track| {
        track
            .position
            .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
    })
}

fn build_rig_track_output(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
) -> TrackBuildOutput {
    match config.track_builder {
        RigTrackBuilder::LegacyUnionFind => {
            build_tracks_detailed(features.len(), pairwise, config.min_track_length.max(2))
        }
        RigTrackBuilder::ConflictPreserving => build_tracks_incremental_correspondence(
            features,
            pairwise,
            config.min_track_length.max(2),
        ),
        RigTrackBuilder::StreamOrderConflictPreserving => {
            build_tracks_incremental_correspondence_in_order(
                features,
                pairwise,
                config.min_track_length.max(2),
            )
        }
        RigTrackBuilder::PairConfidence => build_tracks_confidence_ordered(
            features.len(),
            pairwise,
            config.min_track_length.max(2),
        ),
        RigTrackBuilder::TrustedPrefixPairConfidence(pair_count) => {
            build_tracks_confidence_ordered_with_trusted_prefix(
                features.len(),
                pairwise,
                config.min_track_length.max(2),
                pair_count,
            )
        }
        RigTrackBuilder::SparseCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            None,
            false,
        ),
        RigTrackBuilder::MetricSparseCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            Some(image_assignment),
            false,
        ),
        RigTrackBuilder::MetricTemporalCycle => build_tracks_sparse_cycle(
            features,
            pairwise,
            config.min_track_length.max(2),
            Some(image_assignment),
            true,
        ),
        RigTrackBuilder::MetricTemporalQuadrilateral => {
            build_metric_temporal_quadrilaterals_in_frame_gap(
                features,
                pairwise,
                image_assignment,
                config.min_track_length.max(2),
                config.deferred_retriangulation_quadrilateral_min_frame_gap,
                config.deferred_retriangulation_quadrilateral_max_frame_gap,
            )
        }
        RigTrackBuilder::MetricAnchoredCycle => {
            let output = build_tracks_sparse_cycle(
                features,
                pairwise,
                config.min_track_length.max(2),
                Some(image_assignment),
                false,
            );
            retain_metric_anchored_tracks(output, image_assignment)
        }
    }
}

fn retain_metric_anchored_tracks(
    mut output: TrackBuildOutput,
    image_assignment: &[(usize, usize)],
) -> TrackBuildOutput {
    output
        .tracks
        .retain(|track| track_is_metric_anchored(track, image_assignment));
    output.stats.retained_tracks = output.tracks.len();
    output.stats.retained_observations = output.tracks.iter().map(Vec::len).sum();
    output
}

fn track_is_metric_anchored(track: &[(usize, usize)], image_assignment: &[(usize, usize)]) -> bool {
    let mut frame_sensor = HashMap::<usize, usize>::new();
    track.iter().any(|&(image, _)| {
        let (frame, sensor) = image_assignment[image];
        frame_sensor
            .insert(frame, sensor)
            .is_some_and(|previous| previous != sensor)
    })
}

/// Recover a single conservative 3-D track from each legacy conflict
/// component. Verified edges propose anchors, but already-registered rig poses
/// decide admission. Requiring a graph cycle and a synchronized multi-sensor
/// observation prevents an accidental two-view corridor match from creating a
/// scale-free landmark.
fn recover_metric_conflict_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    conflicting_components: &[Vec<(usize, usize)>],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
) -> Vec<WorkingTrack> {
    type Observation = (usize, usize);
    if config.conflict_recovery_max_hypotheses == 0 {
        return Vec::new();
    }
    let mut component_of = HashMap::new();
    for (component_id, component) in conflicting_components.iter().enumerate() {
        for &observation in component {
            component_of.insert(observation, component_id);
        }
    }
    let mut edges_by_component =
        vec![Vec::<(Observation, Observation)>::new(); conflicting_components.len()];
    for pair in pairwise {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let left = (pair.image_i, keypoint_i);
            let right = (pair.image_j, keypoint_j);
            let Some(&component_id) = component_of.get(&left) else {
                continue;
            };
            if component_of.get(&right) != Some(&component_id) {
                continue;
            }
            edges_by_component[component_id].push(if left <= right {
                (left, right)
            } else {
                (right, left)
            });
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
        edges.dedup();
    }

    let ray = |observation: Observation| {
        let pose = image_poses.get(observation.0)?.as_ref()?;
        let sensor_index = image_assignment.get(observation.0)?.1;
        let sensor = rig.sensors().get(sensor_index)?;
        let pixel = *features.get(observation.0)?.keypoints.get(observation.1)?;
        let normalized = sensor.camera.normalize_pixel(&pixel)?;
        let direction = pose
            .camera_to_world()
            .rotation
            .transform_vector(&Vector3::new(normalized.x, normalized.y, 1.0).normalize())
            .normalize();
        Some((pose.camera_center_world(), direction))
    };
    let reprojection_error = |observation: Observation, point: &Point3<f64>| {
        let pose = image_poses.get(observation.0)?.as_ref()?;
        let sensor_index = image_assignment.get(observation.0)?.1;
        let sensor = rig.sensors().get(sensor_index)?;
        let point_camera = pose.transform_world_point(point);
        if point_camera.z <= 0.0 {
            return None;
        }
        let projected = sensor.camera.project(&point_camera)?;
        let pixel = features.get(observation.0)?.keypoints.get(observation.1)?;
        Some((projected - pixel).norm())
    };

    let mut recovered = Vec::new();
    for (component, edges) in conflicting_components.iter().zip(&edges_by_component) {
        let mut adjacency = HashMap::<Observation, Vec<Observation>>::new();
        let mut anchors = Vec::new();
        for &(left, right) in edges {
            adjacency.entry(left).or_default().push(right);
            adjacency.entry(right).or_default().push(left);
            let (Some((left_center, left_ray)), Some((right_center, right_ray))) =
                (ray(left), ray(right))
            else {
                continue;
            };
            let angle = left_ray.dot(&right_ray).clamp(-1.0, 1.0).abs().acos();
            if angle.is_finite() {
                anchors.push((
                    angle,
                    left,
                    right,
                    left_center,
                    left_ray,
                    right_center,
                    right_ray,
                ));
            }
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
        anchors.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut best: Option<(WorkingTrack, f64)> = None;
        for &(_angle, anchor_left, anchor_right, left_center, left_ray, right_center, right_ray) in
            anchors.iter().take(config.conflict_recovery_max_hypotheses)
        {
            let Some(point) =
                closest_ray_midpoint(&left_center, &left_ray, &right_center, &right_ray)
            else {
                continue;
            };
            let mut valid_errors = HashMap::<Observation, f64>::new();
            for &observation in component {
                let Some(error) = reprojection_error(observation, &point) else {
                    continue;
                };
                if error <= config.conflict_recovery_max_reprojection_error_px {
                    valid_errors.insert(observation, error);
                }
            }
            if !valid_errors.contains_key(&anchor_left) || !valid_errors.contains_key(&anchor_right)
            {
                continue;
            }
            let mut reachable = HashSet::from([anchor_left]);
            let mut pending = vec![anchor_left];
            while let Some(observation) = pending.pop() {
                for &neighbour in adjacency.get(&observation).into_iter().flatten() {
                    if valid_errors.contains_key(&neighbour) && reachable.insert(neighbour) {
                        pending.push(neighbour);
                    }
                }
            }
            let mut best_by_image = HashMap::<usize, (usize, f64)>::new();
            for observation in reachable {
                let error = valid_errors[&observation];
                let entry = best_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, error));
                if error < entry.1 || (error == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, error);
                }
            }
            let mut selected = best_by_image
                .iter()
                .map(|(&image, &(keypoint, _))| (image, keypoint))
                .collect::<Vec<_>>();
            selected.sort_unstable();
            if selected.len() < 3 || !track_is_metric_anchored(&selected, image_assignment) {
                continue;
            }
            let selected_set = selected.iter().copied().collect::<HashSet<_>>();
            let cycle_edges = edges
                .iter()
                .filter(|(left, right)| selected_set.contains(left) && selected_set.contains(right))
                .count();
            if cycle_edges < selected.len() {
                continue;
            }
            let mean_error = selected
                .iter()
                .map(|observation| valid_errors[observation])
                .sum::<f64>()
                / selected.len() as f64;
            if mean_error > config.conflict_recovery_max_mean_reprojection_px {
                continue;
            }
            let candidate = WorkingTrack {
                observations: selected,
                position: Some(point),
                metric_anchored: true,
            };
            let replace = best.as_ref().is_none_or(|(current, current_error)| {
                candidate.observations.len() > current.observations.len()
                    || (candidate.observations.len() == current.observations.len()
                        && mean_error < *current_error)
            });
            if replace {
                best = Some((candidate, mean_error));
            }
        }
        if let Some((track, _)) = best {
            recovered.push(track);
        }
    }
    recovered
}

#[derive(Debug)]
struct SparsePairLookup {
    image_i: usize,
    image_j: usize,
    i_to_j: Vec<u32>,
    j_to_i: Vec<u32>,
}

const SPARSE_PAIR_AMBIGUOUS: u32 = u32::MAX - 1;

impl SparsePairLookup {
    fn counterpart(&self, image: usize, keypoint: usize) -> Option<usize> {
        let value = if image == self.image_i {
            *self.i_to_j.get(keypoint)?
        } else if image == self.image_j {
            *self.j_to_i.get(keypoint)?
        } else {
            return None;
        };
        (value < SPARSE_PAIR_AMBIGUOUS).then_some(value as usize)
    }
}

fn unique_pair_counterpart(
    lookups: &[SparsePairLookup],
    pair_indices: &BTreeMap<(usize, usize), Vec<usize>>,
    image: usize,
    keypoint: usize,
    other_image: usize,
) -> Option<usize> {
    let key = if image < other_image {
        (image, other_image)
    } else {
        (other_image, image)
    };
    let mut counterpart = None;
    for &pair_index in pair_indices.get(&key)? {
        let lookup = &lookups[pair_index];
        let Some(candidate) = lookup.counterpart(image, keypoint) else {
            continue;
        };
        // A one-way minimum is not enough: reject many-to-one descriptor
        // ambiguities unless the reverse lookup selects the same endpoint.
        if lookup.counterpart(other_image, candidate) != Some(keypoint) {
            continue;
        }
        match counterpart {
            None => counterpart = Some(candidate),
            Some(previous) if previous == candidate => {}
            Some(_) => return None,
        }
    }
    counterpart
}

fn build_metric_temporal_quadrilaterals(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_metric_temporal_quadrilaterals_in_frame_gap(
        features,
        pairwise,
        image_assignment,
        min_track_length,
        1,
        1,
    )
}

fn build_metric_temporal_quadrilaterals_in_frame_gap(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_track_length: usize,
    min_frame_gap: usize,
    max_frame_gap: usize,
) -> TrackBuildOutput {
    let mut lookups = Vec::with_capacity(pairwise.len());
    let mut pair_indices = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let mut i_to_j = vec![u32::MAX; features[pair.image_i].len()];
        let mut j_to_i = vec![u32::MAX; features[pair.image_j].len()];
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if keypoint_i >= i_to_j.len()
                || keypoint_j >= j_to_i.len()
                || keypoint_i >= SPARSE_PAIR_AMBIGUOUS as usize
                || keypoint_j >= SPARSE_PAIR_AMBIGUOUS as usize
            {
                continue;
            }
            let keypoint_j = keypoint_j as u32;
            let keypoint_i = keypoint_i as u32;
            let forward = &mut i_to_j[keypoint_i as usize];
            *forward = match *forward {
                u32::MAX => keypoint_j,
                previous if previous == keypoint_j => previous,
                _ => SPARSE_PAIR_AMBIGUOUS,
            };
            let reverse = &mut j_to_i[keypoint_j as usize];
            *reverse = match *reverse {
                u32::MAX => keypoint_i,
                previous if previous == keypoint_i => previous,
                _ => SPARSE_PAIR_AMBIGUOUS,
            };
        }
        lookups.push(SparsePairLookup {
            image_i: pair.image_i,
            image_j: pair.image_j,
            i_to_j,
            j_to_i,
        });
        let key = if pair.image_i < pair.image_j {
            (pair.image_i, pair.image_j)
        } else {
            (pair.image_j, pair.image_i)
        };
        pair_indices.entry(key).or_default().push(pair_index);
    }

    let frame_count = image_assignment
        .iter()
        .map(|&(frame, _)| frame)
        .max()
        .map_or(0, |maximum| maximum + 1);
    let mut frame_images = vec![BTreeMap::<usize, Option<usize>>::new(); frame_count];
    for (image, &(frame, sensor)) in image_assignment.iter().enumerate() {
        frame_images[frame]
            .entry(sensor)
            .and_modify(|slot| *slot = None)
            .or_insert(Some(image));
    }

    let mut quadrilaterals = BTreeSet::<Vec<(usize, usize)>>::new();
    for pair in pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        let frame_gap = frame_i.abs_diff(frame_j);
        if !(min_frame_gap..=max_frame_gap).contains(&frame_gap) || sensor_i != sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if unique_pair_counterpart(
                &lookups,
                &pair_indices,
                pair.image_i,
                keypoint_i,
                pair.image_j,
            ) != Some(keypoint_j)
            {
                continue;
            }
            for (&other_sensor, &other_image_i) in &frame_images[frame_i] {
                if other_sensor == sensor_i {
                    continue;
                }
                let (Some(other_image_i), Some(other_image_j)) = (
                    other_image_i,
                    frame_images[frame_j].get(&other_sensor).copied().flatten(),
                ) else {
                    continue;
                };
                let Some(other_keypoint_i) = unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    pair.image_i,
                    keypoint_i,
                    other_image_i,
                ) else {
                    continue;
                };
                let Some(other_keypoint_j) = unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    pair.image_j,
                    keypoint_j,
                    other_image_j,
                ) else {
                    continue;
                };
                if unique_pair_counterpart(
                    &lookups,
                    &pair_indices,
                    other_image_i,
                    other_keypoint_i,
                    other_image_j,
                ) != Some(other_keypoint_j)
                {
                    continue;
                }
                let mut observations = vec![
                    (pair.image_i, keypoint_i),
                    (pair.image_j, keypoint_j),
                    (other_image_i, other_keypoint_i),
                    (other_image_j, other_keypoint_j),
                ];
                observations.sort_unstable();
                quadrilaterals.insert(observations);
            }
        }
    }

    let connected_components = quadrilaterals.len();
    let mut owned_observations = HashSet::new();
    let mut rejected_conflicts = 0usize;
    let mut tracks = Vec::new();
    for observations in quadrilaterals {
        if observations.len() < min_track_length {
            continue;
        }
        if observations
            .iter()
            .any(|observation| owned_observations.contains(observation))
        {
            rejected_conflicts += 1;
            continue;
        }
        owned_observations.extend(observations.iter().copied());
        tracks.push(observations);
    }
    TrackBuildOutput {
        stats: TrackBuildStats {
            input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
            connected_components,
            conflicting_components: rejected_conflicts,
            conflicting_observations: rejected_conflicts.saturating_mul(4),
            retained_tracks: tracks.len(),
            retained_observations: tracks.iter().map(Vec::len).sum(),
            ..TrackBuildStats::default()
        },
        tracks,
        conflicting_components: Vec::new(),
    }
}

/// Return deterministic, observation-disjoint stereo-temporal
/// quadrilaterals for diagnostics and independent correspondence screening.
/// The returned tracks contain exactly four `(image, keypoint)` observations.
pub fn metric_temporal_quadrilateral_tracks(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> Vec<Vec<(usize, usize)>> {
    build_metric_temporal_quadrilaterals(features, pairwise, image_assignment, 2).tracks
}

/// Return the same deterministic, observation-disjoint exact quadrilaterals
/// for an explicit temporal-baseline range. Invalid or reversed ranges yield
/// no tracks. This is intended for bounded long-range correspondence audits.
pub fn metric_temporal_quadrilateral_tracks_in_frame_gap(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    min_frame_gap: usize,
    max_frame_gap: usize,
) -> Vec<Vec<(usize, usize)>> {
    if min_frame_gap == 0 || min_frame_gap > max_frame_gap {
        return Vec::new();
    }
    build_metric_temporal_quadrilaterals_in_frame_gap(
        features,
        pairwise,
        image_assignment,
        2,
        min_frame_gap,
        max_frame_gap,
    )
    .tracks
}

#[derive(Debug, Clone, Copy)]
struct SparseCycleCandidate {
    metric_stereo: bool,
    frame_gap: u32,
    cycles: u16,
    pair_support: u32,
    pair_index: u32,
    match_index: u32,
}

fn build_tracks_sparse_cycle(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    image_assignment: Option<&[(usize, usize)]>,
    temporal_first: bool,
) -> TrackBuildOutput {
    let mut lookups = Vec::with_capacity(pairwise.len());
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); features.len()];
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let mut i_to_j = vec![u32::MAX; features[pair.image_i].len()];
        let mut j_to_i = vec![u32::MAX; features[pair.image_j].len()];
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if keypoint_i >= i_to_j.len()
                || keypoint_j >= j_to_i.len()
                || keypoint_i > u32::MAX as usize
                || keypoint_j > u32::MAX as usize
            {
                continue;
            }
            i_to_j[keypoint_i] = i_to_j[keypoint_i].min(keypoint_j as u32);
            j_to_i[keypoint_j] = j_to_i[keypoint_j].min(keypoint_i as u32);
        }
        lookups.push(SparsePairLookup {
            image_i: pair.image_i,
            image_j: pair.image_j,
            i_to_j,
            j_to_i,
        });
        adjacency[pair.image_i].push((pair.image_j, pair_index));
        adjacency[pair.image_j].push((pair.image_i, pair_index));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
    }

    let cycle_count =
        |image_i: usize, keypoint_i: usize, image_j: usize, keypoint_j: usize| -> u16 {
            let left = &adjacency[image_i];
            let right = &adjacency[image_j];
            let mut left_index = 0usize;
            let mut right_index = 0usize;
            let mut cycles = 0u16;
            while left_index < left.len() && right_index < right.len() {
                match left[left_index].0.cmp(&right[right_index].0) {
                    std::cmp::Ordering::Less => left_index += 1,
                    std::cmp::Ordering::Greater => right_index += 1,
                    std::cmp::Ordering::Equal => {
                        let neighbour = left[left_index].0;
                        let left_end = left[left_index..]
                            .iter()
                            .position(|entry| entry.0 != neighbour)
                            .map_or(left.len(), |offset| left_index + offset);
                        let right_end = right[right_index..]
                            .iter()
                            .position(|entry| entry.0 != neighbour)
                            .map_or(right.len(), |offset| right_index + offset);
                        if neighbour != image_i
                            && neighbour != image_j
                            && left[left_index..left_end].iter().any(|&(_, left_pair)| {
                                let Some(left_keypoint) =
                                    lookups[left_pair].counterpart(image_i, keypoint_i)
                                else {
                                    return false;
                                };
                                right[right_index..right_end]
                                    .iter()
                                    .any(|&(_, right_pair)| {
                                        lookups[right_pair].counterpart(image_j, keypoint_j)
                                            == Some(left_keypoint)
                                    })
                            })
                        {
                            cycles = cycles.saturating_add(1);
                        }
                        left_index = left_end;
                        right_index = right_end;
                    }
                }
            }
            cycles
        };

    let mut candidates = Vec::with_capacity(pairwise.iter().map(|pair| pair.matches.len()).sum());
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for (match_index, &(keypoint_i, keypoint_j)) in pair.matches.iter().enumerate() {
            candidates.push(SparseCycleCandidate {
                metric_stereo: image_assignment.is_some_and(|assignment| {
                    let (frame_i, sensor_i) = assignment[pair.image_i];
                    let (frame_j, sensor_j) = assignment[pair.image_j];
                    frame_i == frame_j && sensor_i != sensor_j
                }),
                frame_gap: image_assignment.map_or(u32::MAX, |assignment| {
                    assignment[pair.image_i]
                        .0
                        .abs_diff(assignment[pair.image_j].0)
                        .min(u32::MAX as usize) as u32
                }),
                cycles: cycle_count(pair.image_i, keypoint_i, pair.image_j, keypoint_j),
                pair_support: pair.matches.len().min(u32::MAX as usize) as u32,
                pair_index: pair_index as u32,
                match_index: match_index as u32,
            });
        }
    }
    candidates.sort_unstable_by(|left, right| {
        let left_pair = &pairwise[left.pair_index as usize];
        let right_pair = &pairwise[right.pair_index as usize];
        let (left_keypoint_i, left_keypoint_j) = left_pair.matches[left.match_index as usize];
        let (right_keypoint_i, right_keypoint_j) = right_pair.matches[right.match_index as usize];
        right
            .metric_stereo
            .cmp(&left.metric_stereo)
            .then_with(|| {
                if temporal_first {
                    left.frame_gap.cmp(&right.frame_gap)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .then_with(|| right.cycles.cmp(&left.cycles))
            .then_with(|| right.pair_support.cmp(&left.pair_support))
            .then_with(|| left_pair.image_i.cmp(&right_pair.image_i))
            .then_with(|| left_pair.image_j.cmp(&right_pair.image_j))
            .then_with(|| left_keypoint_i.cmp(&right_keypoint_i))
            .then_with(|| left_keypoint_j.cmp(&right_keypoint_j))
    });

    let mut node_id = HashMap::<(usize, usize), usize>::new();
    let mut nodes = Vec::<(usize, usize)>::new();
    let mut parent = Vec::<usize>::new();
    let mut component_size = Vec::<usize>::new();
    let mut component_images = Vec::<HashSet<usize>>::new();
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let pair = &pairwise[candidate.pair_index as usize];
        let (keypoint_i, keypoint_j) = pair.matches[candidate.match_index as usize];
        let mut node = |observation: (usize, usize)| {
            *node_id.entry(observation).or_insert_with(|| {
                let id = nodes.len();
                nodes.push(observation);
                parent.push(id);
                component_size.push(1);
                component_images.push(HashSet::from([observation.0]));
                id
            })
        };
        let left = node((pair.image_i, keypoint_i));
        let right = node((pair.image_j, keypoint_j));
        let left_root = sparse_find(&mut parent, left);
        let right_root = sparse_find(&mut parent, right);
        if left_root == right_root {
            continue;
        }
        if component_images[left_root]
            .iter()
            .any(|image| component_images[right_root].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        let (root, child) = if component_size[left_root] > component_size[right_root]
            || (component_size[left_root] == component_size[right_root] && left_root < right_root)
        {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups = HashMap::<usize, Vec<(usize, usize)>>::new();
    for (node_index, &observation) in nodes.iter().enumerate() {
        let root = sparse_find(&mut parent, node_index);
        groups.entry(root).or_default().push(observation);
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        conflicting_components: rejected_conflicts,
        conflicting_observations: rejected_conflicts.saturating_mul(2),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for mut observations in groups.into_values() {
        if observations.len() < min_track_length {
            continue;
        }
        observations.sort_unstable();
        stats.retained_observations += observations.len();
        tracks.push(observations);
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

fn sparse_find(parent: &mut [usize], mut node: usize) -> usize {
    let mut root = node;
    while parent[root] != root {
        root = parent[root];
    }
    while parent[node] != node {
        let next = parent[node];
        parent[node] = root;
        node = next;
    }
    root
}

fn filter_positioned_track_observations(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    max_reprojection_error_px: f64,
    tracks: &mut [WorkingTrack],
) -> usize {
    let mut pruned = 0usize;
    for track in tracks {
        let Some(position) = track.position else {
            continue;
        };
        let mut registered = 0usize;
        let mut inliers = 0usize;
        for &(image, keypoint) in &track.observations {
            let Some(pose) = image_poses[image].as_ref() else {
                continue;
            };
            registered += 1;
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&position);
            if point_camera.z > 0.0
                && sensor
                    .camera
                    .project(&point_camera)
                    .is_some_and(|projected| {
                        (projected - features[image].keypoints[keypoint]).norm()
                            <= max_reprojection_error_px
                    })
            {
                inliers += 1;
            }
        }
        if inliers < 2 {
            track.position = None;
            pruned += registered;
            continue;
        }
        track.observations.retain(|&(image, keypoint)| {
            let Some(pose) = image_poses[image].as_ref() else {
                return true;
            };
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&position);
            point_camera.z > 0.0
                && sensor
                    .camera
                    .project(&point_camera)
                    .is_some_and(|projected| {
                        (projected - features[image].keypoints[keypoint]).norm()
                            <= max_reprojection_error_px
                    })
        });
        pruned += registered - inliers;
        track.metric_anchored = track_is_metric_anchored(&track.observations, image_assignment);
    }
    pruned
}

fn merge_rig_ba_stats(target: &mut Option<RigBaStats>, source: Option<RigBaStats>) {
    let Some(source) = source else {
        return;
    };
    let target = target.get_or_insert(RigBaStats {
        observations: 0,
        initial_cost: 0.0,
        final_cost: 0.0,
        iterations: 0,
        converged: true,
    });
    target.observations += source.observations;
    target.initial_cost += source.initial_cost;
    target.final_cost += source.final_cost;
    target.iterations += source.iterations;
    target.converged &= source.converged;
}

fn refine_rig_structure(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
) -> usize {
    if config.structure_refinement_iterations == 0 {
        return 0;
    }
    tracks
        .iter_mut()
        .filter_map(|track| {
            let mut point = track.position?;
            let initial = point;
            let mut accepted = false;
            for _ in 0..config.structure_refinement_iterations {
                let mut hessian = Matrix3::zeros();
                let mut gradient = Vector3::zeros();
                let mut observations = 0usize;
                let Some(current_cost) = rig_point_cost(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    track,
                    &point,
                    config.max_reprojection_error_px,
                ) else {
                    break;
                };
                let step = 1.0e-6 * (1.0 + point.coords.norm());
                for &(image, keypoint) in &track.observations {
                    let Some(pose) = image_poses[image].as_ref() else {
                        continue;
                    };
                    let sensor = &rig.sensors()[image_assignment[image].1];
                    let measured = features[image].keypoints[keypoint];
                    let Some(projected) =
                        sensor.camera.project(&pose.transform_world_point(&point))
                    else {
                        continue;
                    };
                    let residual = projected - measured;
                    let norm = residual.norm();
                    if !norm.is_finite() || norm > 2.0 * config.max_reprojection_error_px {
                        continue;
                    }
                    let mut jacobian = SMatrix::<f64, 2, 3>::zeros();
                    let mut valid = true;
                    for axis in 0..3 {
                        let mut plus = point;
                        let mut minus = point;
                        plus[axis] += step;
                        minus[axis] -= step;
                        let projections = sensor
                            .camera
                            .project(&pose.transform_world_point(&plus))
                            .zip(sensor.camera.project(&pose.transform_world_point(&minus)));
                        let Some((plus_pixel, minus_pixel)) = projections else {
                            valid = false;
                            break;
                        };
                        jacobian.set_column(axis, &((plus_pixel - minus_pixel) / (2.0 * step)));
                    }
                    if !valid || !jacobian.iter().all(|value| value.is_finite()) {
                        continue;
                    }
                    let weight = huber_weight(norm, config.max_reprojection_error_px);
                    hessian += jacobian.transpose() * jacobian * weight;
                    gradient +=
                        jacobian.transpose() * Vector2::new(residual.x, residual.y) * weight;
                    observations += 1;
                }
                if observations < 2 {
                    break;
                }
                let damping = 1.0e-8_f64.max(1.0e-6 * hessian.diagonal().amax());
                hessian += Matrix3::identity() * damping;
                let Some(delta) = hessian.lu().solve(&(-gradient)) else {
                    break;
                };
                if !delta.iter().all(|value| value.is_finite()) {
                    break;
                }
                let candidate = point + delta;
                let Some(candidate_cost) = rig_point_cost(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    track,
                    &candidate,
                    config.max_reprojection_error_px,
                ) else {
                    break;
                };
                if candidate_cost + 1.0e-12 >= current_cost {
                    break;
                }
                point = candidate;
                accepted = true;
                if delta.norm() <= 1.0e-8 * (1.0 + point.coords.norm()) {
                    break;
                }
            }
            if accepted && (point - initial).norm() > 0.0 {
                track.position = Some(point);
                Some(())
            } else {
                None
            }
        })
        .count()
}

fn rig_point_cost(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    track: &WorkingTrack,
    point: &Point3<f64>,
    huber_delta: f64,
) -> Option<f64> {
    let mut cost = 0.0;
    let mut count = 0usize;
    for &(image, keypoint) in &track.observations {
        let Some(pose) = image_poses[image].as_ref() else {
            continue;
        };
        let sensor = &rig.sensors()[image_assignment[image].1];
        let Some(projected) = sensor.camera.project(&pose.transform_world_point(point)) else {
            continue;
        };
        let norm = (projected - features[image].keypoints[keypoint]).norm();
        if !norm.is_finite() || norm > 2.0 * huber_delta {
            continue;
        }
        cost += if norm <= huber_delta {
            0.5 * norm * norm
        } else {
            huber_delta * (norm - 0.5 * huber_delta)
        };
        count += 1;
    }
    (count >= 2).then_some(cost)
}

fn huber_weight(norm: f64, delta: f64) -> f64 {
    if norm <= delta || norm <= f64::EPSILON {
        1.0
    } else {
        delta / norm
    }
}

#[allow(clippy::too_many_arguments)]
fn run_windowed_final_ba(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
) -> Result<Option<RigBaStats>, RigSfmError> {
    let window = config.final_ba_window_size.max(2);
    let stride = (window / 2).max(1);
    let mut starts = (0..frame_poses.len()).step_by(stride).collect::<Vec<_>>();
    if let Some(tail_start) = frame_poses.len().checked_sub(window) {
        starts.push(tail_start);
    }
    starts.sort_unstable();
    starts.dedup();
    let ba_config = BaConfig {
        max_iterations: config.local_ba_iterations,
        ..config.ba_config
    };
    let mut aggregate: Option<RigBaStats> = None;
    for pass in 0..config.final_ba_passes {
        starts.sort_unstable();
        if pass % 2 == 1 {
            starts.reverse();
        }
        for &start in &starts {
            let end = (start + window).min(frame_poses.len());
            let active_frames = (start..end)
                .filter(|frame| frame_poses[*frame].is_some())
                .collect::<HashSet<_>>();
            if active_frames.len() < 2 {
                continue;
            }
            let anchor = if pass % 2 == 0 {
                *active_frames.iter().min().unwrap()
            } else {
                *active_frames.iter().max().unwrap()
            };
            let other_end = if pass % 2 == 0 {
                *active_frames.iter().max().unwrap()
            } else {
                *active_frames.iter().min().unwrap()
            };
            let extra_fixed = config
                .final_ba_fix_window_ends
                .then_some(other_end)
                .into_iter()
                .collect::<Vec<_>>();
            if let Some(stats) = run_rig_bundle_adjustment(
                rig,
                features,
                image_assignment,
                config,
                &active_frames,
                anchor,
                &ba_config,
                config.final_ba_min_pose_observations,
                &extra_fixed,
                false,
                frame_poses,
                image_poses,
                tracks,
            )? {
                let current = aggregate.get_or_insert(RigBaStats {
                    observations: 0,
                    initial_cost: 0.0,
                    final_cost: 0.0,
                    iterations: 0,
                    converged: true,
                });
                current.observations += stats.observations;
                current.initial_cost += stats.initial_cost;
                current.final_cost += stats.final_cost;
                current.iterations += stats.iterations;
                current.converged &= stats.converged;
            }
        }
    }
    Ok(aggregate)
}

#[allow(clippy::too_many_arguments)]
fn run_rig_bundle_adjustment(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    active_frames: &HashSet<usize>,
    anchor_frame_index: usize,
    ba_config: &BaConfig,
    min_pose_observations: usize,
    extra_fixed_frames: &[usize],
    fix_active_rotations: bool,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
) -> Result<Option<RigBaStats>, RigSfmError> {
    let mut problem = BundleAdjustment::new(rig.sensors()[0].camera.clone());
    for (frame, pose) in frame_poses.iter().enumerate() {
        if !active_frames.contains(&frame) {
            continue;
        }
        let Some(pose) = pose else {
            continue;
        };
        problem.add_pose(frame as u64, pose.clone());
    }
    problem.fix_pose(anchor_frame_index as u64);
    for &frame in extra_fixed_frames {
        problem.fix_pose(frame as u64);
    }
    if fix_active_rotations {
        for &frame in active_frames {
            if frame != anchor_frame_index && frame_poses[frame].is_some() {
                problem.fix_pose_rotation(frame as u64);
            }
        }
    }

    let mut visual_observations = 0usize;
    let mut observations_per_frame = vec![0usize; frame_poses.len()];
    for (track_index, track) in tracks.iter().enumerate() {
        if config.ba_metric_tracks_only && !track.metric_anchored {
            continue;
        }
        let Some(position) = track.position else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, keypoint) in &track.observations {
            let Some(pose) = image_poses[image].as_ref() else {
                continue;
            };
            let (frame, sensor_index) = image_assignment[image];
            if !active_frames.contains(&frame) {
                continue;
            }
            let sensor = &rig.sensors()[sensor_index];
            let pixel = features[image].keypoints[keypoint];
            let usable = sensor
                .camera
                .project(&pose.transform_world_point(&position))
                .is_some_and(|projected| {
                    (projected - pixel).norm() <= 2.0 * config.max_reprojection_error_px
                });
            if usable {
                observations_per_frame[frame] += 1;
                observations.push(BaRigObservation {
                    keyframe_id: frame as u64,
                    landmark_id: track_index as u64,
                    xy: pixel,
                    camera: sensor.camera.clone(),
                    sensor_from_rig: sensor.sensor_from_rig.clone(),
                });
            }
        }
        if observations.len() < 2 {
            continue;
        }
        problem.add_landmark(track_index as u64, position);
        visual_observations += observations.len();
        for observation in observations {
            problem.add_rig_observation(observation);
        }
    }
    if visual_observations == 0 {
        return Ok(None);
    }
    if min_pose_observations > 0 {
        for &frame in active_frames {
            if observations_per_frame[frame] < min_pose_observations {
                problem.fix_pose(frame as u64);
            }
        }
    }
    let result = problem
        .optimize(ba_config)
        .map_err(|error| RigSfmError::BundleAdjustment(error.to_string()))?;
    for (frame, pose) in frame_poses.iter_mut().enumerate() {
        let Some(reference_pose) = problem.poses.get(&(frame as u64)) else {
            continue;
        };
        *pose = Some(reference_pose.clone());
    }
    for (track_index, track) in tracks.iter_mut().enumerate() {
        if let Some(position) = problem.landmarks.get(&(track_index as u64)) {
            track.position = Some(*position);
        }
    }
    for (image, pose) in image_poses.iter_mut().enumerate() {
        let (frame, sensor_index) = image_assignment[image];
        let Some(frame_pose) = frame_poses[frame].as_ref() else {
            continue;
        };
        *pose = Some(Pose {
            world_to_camera: rig.sensors()[sensor_index]
                .sensor_from_rig
                .compose(&frame_pose.world_to_camera),
        });
    }
    Ok(Some(RigBaStats {
        observations: visual_observations,
        initial_cost: result.initial_cost,
        final_cost: result.final_cost,
        iterations: result.iterations.len(),
        converged: result.converged,
    }))
}

fn validate_inputs(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &RigSfmConfig,
) -> Result<(), RigSfmError> {
    if !(1..=rig.sensors().len()).contains(&config.min_pnp_sensors) {
        return Err(RigSfmError::InvalidMinPnpSensors {
            requested: config.min_pnp_sensors,
            sensor_count: rig.sensors().len(),
        });
    }
    if let Some(requested) = config.direct_stereo_min_pnp_sensors {
        if !(1..=rig.sensors().len()).contains(&requested) {
            return Err(RigSfmError::InvalidDirectStereoMinPnpSensors {
                requested,
                sensor_count: rig.sensors().len(),
            });
        }
    }
    if !config.direct_stereo_min_triangulation_angle_deg.is_finite()
        || config.direct_stereo_min_triangulation_angle_deg <= 0.0
    {
        return Err(RigSfmError::InvalidDirectStereoTriangulationAngle(
            config.direct_stereo_min_triangulation_angle_deg,
        ));
    }
    if !config.triangulation_min_inlier_fraction.is_finite()
        || !(0.5..=1.0).contains(&config.triangulation_min_inlier_fraction)
    {
        return Err(RigSfmError::InvalidTriangulationInlierFraction(
            config.triangulation_min_inlier_fraction,
        ));
    }
    for (name, value) in [
        (
            "maximum reprojection error",
            config.conflict_recovery_max_reprojection_error_px,
        ),
        (
            "maximum mean reprojection error",
            config.conflict_recovery_max_mean_reprojection_px,
        ),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(RigSfmError::InvalidConflictRecoveryGate { name, value });
        }
    }
    if config.motion_bridge_min_inliers < 8 {
        return Err(RigSfmError::InvalidMotionBridgeMinInliers(
            config.motion_bridge_min_inliers,
        ));
    }
    if !config.motion_bridge_max_rotation_deviation_deg.is_finite()
        || config.motion_bridge_max_rotation_deviation_deg <= 0.0
    {
        return Err(RigSfmError::InvalidMotionBridgeRotationDeviation(
            config.motion_bridge_max_rotation_deviation_deg,
        ));
    }
    if let Some(prefix) = config.deferred_registration_pair_prefix {
        if prefix > pairwise.len() {
            return Err(RigSfmError::DeferredPairPrefix {
                prefix,
                pair_count: pairwise.len(),
            });
        }
    }
    if let Some(prefix) = config.deferred_retriangulation_pair_prefix {
        let registration_prefix = config
            .deferred_registration_pair_prefix
            .unwrap_or(pairwise.len());
        if prefix < registration_prefix || prefix > pairwise.len() {
            return Err(RigSfmError::DeferredRetriangulationPairPrefix {
                prefix,
                registration_prefix,
                pair_count: pairwise.len(),
            });
        }
    }
    if config.deferred_retriangulation_quadrilateral_min_frame_gap == 0
        || config.deferred_retriangulation_quadrilateral_min_frame_gap
            > config.deferred_retriangulation_quadrilateral_max_frame_gap
    {
        return Err(RigSfmError::InvalidDeferredQuadrilateralFrameGap {
            min: config.deferred_retriangulation_quadrilateral_min_frame_gap,
            max: config.deferred_retriangulation_quadrilateral_max_frame_gap,
        });
    }
    if let Some(requested) = config.deferred_registration_min_pnp_sensors {
        if !(1..=rig.sensors().len()).contains(&requested) {
            return Err(RigSfmError::InvalidDeferredMinPnpSensors {
                requested,
                sensor_count: rig.sensors().len(),
            });
        }
    }
    if config.deferred_registration_pnp_max_iterations == Some(0) {
        return Err(RigSfmError::InvalidDeferredPnpIterations);
    }
    if config
        .deferred_registration_min_pnp_inliers
        .is_some_and(|inliers| inliers < 6)
    {
        return Err(RigSfmError::InvalidDeferredMinPnpInliers(
            config
                .deferred_registration_min_pnp_inliers
                .expect("checked as present"),
        ));
    }
    if config.complete_tracks_after_registration {
        if config.track_completion_max_passes == 0 {
            return Err(RigSfmError::InvalidTrackCompletionPasses);
        }
        if !config
            .track_completion_max_reprojection_error_px
            .is_finite()
            || config.track_completion_max_reprojection_error_px <= 0.0
        {
            return Err(RigSfmError::InvalidTrackCompletionReprojectionError(
                config.track_completion_max_reprojection_error_px,
            ));
        }
    }
    if config.repair_isolated_pose_outliers {
        if config.isolated_pose_repair_max_passes == 0 {
            return Err(RigSfmError::InvalidIsolatedPoseRepairPasses);
        }
        if !config.isolated_pose_max_midpoint_error_m.is_finite()
            || config.isolated_pose_max_midpoint_error_m <= 0.0
        {
            return Err(RigSfmError::InvalidIsolatedPoseMidpointError(
                config.isolated_pose_max_midpoint_error_m,
            ));
        }
        if !config.isolated_pose_min_detour_ratio.is_finite()
            || config.isolated_pose_min_detour_ratio <= 1.0
        {
            return Err(RigSfmError::InvalidIsolatedPoseDetourRatio(
                config.isolated_pose_min_detour_ratio,
            ));
        }
    }
    if config.repair_paired_pose_jumps {
        if !config.paired_pose_jump_absolute_step_m.is_finite()
            || config.paired_pose_jump_absolute_step_m <= 0.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpStep(
                config.paired_pose_jump_absolute_step_m,
            ));
        }
        if !config.paired_pose_jump_min_step_ratio.is_finite()
            || config.paired_pose_jump_min_step_ratio <= 1.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpStepRatio(
                config.paired_pose_jump_min_step_ratio,
            ));
        }
        if config.paired_pose_jump_max_frame_span == 0 {
            return Err(RigSfmError::InvalidPairedPoseJumpFrameSpan);
        }
        if !config.paired_pose_jump_max_closure_ratio.is_finite()
            || config.paired_pose_jump_max_closure_ratio <= 0.0
            || config.paired_pose_jump_max_closure_ratio >= 1.0
        {
            return Err(RigSfmError::InvalidPairedPoseJumpClosureRatio(
                config.paired_pose_jump_max_closure_ratio,
            ));
        }
    }
    let mut assigned = vec![false; features.len()];
    for (frame_index, frame) in frames.iter().enumerate() {
        if frame.images.is_empty() {
            return Err(RigSfmError::EmptyFrame { frame: frame_index });
        }
        let mut sensors = HashSet::new();
        for image in &frame.images {
            if image.image_index >= features.len() {
                return Err(RigSfmError::ImageIndex {
                    image: image.image_index,
                    feature_count: features.len(),
                });
            }
            if image.sensor_index >= rig.sensors().len() {
                return Err(RigSfmError::SensorIndex {
                    sensor: image.sensor_index,
                    sensor_count: rig.sensors().len(),
                });
            }
            if assigned[image.image_index] {
                return Err(RigSfmError::DuplicateImage {
                    image: image.image_index,
                });
            }
            assigned[image.image_index] = true;
            if !sensors.insert(image.sensor_index) {
                return Err(RigSfmError::DuplicateSensor {
                    frame: frame_index,
                    sensor: image.sensor_index,
                });
            }
        }
    }
    if let Some(image) = assigned.iter().position(|assigned| !assigned) {
        return Err(RigSfmError::UnassignedImage { image });
    }
    for pair in pairwise {
        if pair.image_i >= features.len() || pair.image_j >= features.len() {
            return Err(RigSfmError::PairImageIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
                feature_count: features.len(),
            });
        }
        if pair.matches.iter().any(|(left, right)| {
            *left >= features[pair.image_i].len() || *right >= features[pair.image_j].len()
        }) {
            return Err(RigSfmError::PairKeypointIndex {
                image_i: pair.image_i,
                image_j: pair.image_j,
            });
        }
    }
    Ok(())
}

fn image_assignment(frames: &[RigFrame], image_count: usize) -> Vec<(usize, usize)> {
    let mut assignment = vec![(usize::MAX, usize::MAX); image_count];
    for (frame_index, frame) in frames.iter().enumerate() {
        for image in &frame.images {
            assignment[image.image_index] = (frame_index, image.sensor_index);
        }
    }
    assignment
}

fn image_track_index(image_count: usize, tracks: &[WorkingTrack]) -> Vec<Vec<(usize, usize)>> {
    let mut by_image = vec![Vec::new(); image_count];
    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in &track.observations {
            by_image[image].push((keypoint, track_index));
        }
    }
    by_image
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct TrackCompletionUpdate {
    passes: usize,
    pair_visits: usize,
    observations: usize,
    reprojection_rejections: usize,
}

/// Complete positioned tracks through the verified sparse correspondence
/// graph without merging landmarks or changing the PnP observation index.
/// Dense per-keypoint owner slots make lookup O(1), so the bound is
/// O(keypoints + observations + passes * verified matches), never O(images²).
#[allow(clippy::too_many_arguments)]
fn complete_positioned_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    max_passes: usize,
    max_reprojection_error_px: f64,
    tracks: &mut [WorkingTrack],
) -> TrackCompletionUpdate {
    const CONFLICTED_OWNER: usize = usize::MAX;

    // This index is deliberately private to completion. In particular, it is
    // not the mapper's `image_tracks` cache used for subsequent PnP attempts.
    let mut positioned_by_image = features
        .iter()
        .map(|feature| vec![None; feature.len()])
        .collect::<Vec<Vec<Option<usize>>>>();
    let mut unpositioned_by_image = features
        .iter()
        .map(|feature| vec![None; feature.len()])
        .collect::<Vec<Vec<Option<usize>>>>();
    for (track_index, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in &track.observations {
            if track.position.is_some() {
                let owner = &mut positioned_by_image[image][keypoint];
                match owner {
                    Some(previous) if *previous != track_index => *owner = Some(CONFLICTED_OWNER),
                    None => *owner = Some(track_index),
                    _ => {}
                }
            } else {
                unpositioned_by_image[image][keypoint].get_or_insert(track_index);
            }
        }
    }
    let mut update = TrackCompletionUpdate::default();
    for _ in 0..max_passes {
        update.passes += 1;
        let before = update.observations;
        for pair in pairwise {
            update.pair_visits += 1;
            if image_poses[pair.image_i].is_none() || image_poses[pair.image_j].is_none() {
                continue;
            }
            for &(keypoint_i, keypoint_j) in &pair.matches {
                let owner_i = positioned_by_image[pair.image_i][keypoint_i];
                let owner_j = positioned_by_image[pair.image_j][keypoint_j];
                let (source_track, target_image, target_keypoint) = match (owner_i, owner_j) {
                    (Some(track), None) if track != CONFLICTED_OWNER => {
                        (track, pair.image_j, keypoint_j)
                    }
                    (None, Some(track)) if track != CONFLICTED_OWNER => {
                        (track, pair.image_i, keypoint_i)
                    }
                    // Never merge two positioned tracks, and never propagate
                    // from an observation with ambiguous positioned ownership.
                    _ => continue,
                };
                if tracks[source_track]
                    .observations
                    .iter()
                    .any(|&(image, _)| image == target_image)
                {
                    continue;
                }
                let Some(position) = tracks[source_track].position else {
                    continue;
                };
                let pose = image_poses[target_image]
                    .as_ref()
                    .expect("registered target was checked above");
                let sensor = &rig.sensors()[image_assignment[target_image].1];
                let Some(projected) = sensor
                    .camera
                    .project(&pose.transform_world_point(&position))
                else {
                    update.reprojection_rejections += 1;
                    continue;
                };
                let error = (projected - features[target_image].keypoints[target_keypoint]).norm();
                if !error.is_finite() || error > max_reprojection_error_px {
                    update.reprojection_rejections += 1;
                    continue;
                }

                // If the observation belonged to an unpositioned raw track,
                // detach it there before attaching it to the positioned track.
                // No external/PnP index is updated.
                if let Some(old_track) = unpositioned_by_image[target_image][target_keypoint].take()
                {
                    tracks[old_track]
                        .observations
                        .retain(|&observation| observation != (target_image, target_keypoint));
                }
                tracks[source_track]
                    .observations
                    .push((target_image, target_keypoint));
                positioned_by_image[target_image][target_keypoint] = Some(source_track);
                update.observations += 1;
            }
        }
        if update.observations == before {
            break;
        }
    }
    update
}

fn metric_frame_supports(
    frame_count: usize,
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
) -> Vec<usize> {
    let mut supports = vec![0usize; frame_count];
    for track in tracks {
        let mut frame_sensors = track
            .observations
            .iter()
            .map(|(image, _)| image_assignment[*image])
            .collect::<Vec<_>>();
        frame_sensors.sort_unstable();
        let mut offset = 0usize;
        while offset < frame_sensors.len() {
            let frame = frame_sensors[offset].0;
            let first_sensor = frame_sensors[offset].1;
            offset += 1;
            let mut multiple_sensors = false;
            while offset < frame_sensors.len() && frame_sensors[offset].0 == frame {
                multiple_sensors |= frame_sensors[offset].1 != first_sensor;
                offset += 1;
            }
            supports[frame] += usize::from(multiple_sensors);
        }
    }
    supports
}

fn metric_seed_candidates(supports: &[usize]) -> Vec<usize> {
    let mut candidates = supports
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, supported)| *supported >= 6)
        .collect::<Vec<_>>();
    candidates
        .sort_unstable_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    candidates.into_iter().map(|(frame, _)| frame).collect()
}

fn install_image_poses(
    rig: &GeneralizedCameraRig,
    frame: &RigFrame,
    frame_pose: &Pose,
    image_poses: &mut [Option<Pose>],
) {
    for image in &frame.images {
        let sensor = &rig.sensors()[image.sensor_index];
        image_poses[image.image_index] = Some(Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&frame_pose.world_to_camera),
        });
    }
}

fn build_verified_stereo_links(
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> HashMap<(usize, usize), (usize, usize)> {
    let mut links = HashMap::new();
    for pair in pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        if frame_i != frame_j || sensor_i == sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            links
                .entry((pair.image_i, keypoint_i))
                .or_insert((pair.image_j, keypoint_j));
            links
                .entry((pair.image_j, keypoint_j))
                .or_insert((pair.image_i, keypoint_i));
        }
    }
    links
}

fn build_relevant_verified_stereo_links(
    base_pairwise: &[PairwiseMatches],
    deferred_pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
) -> HashMap<(usize, usize), (usize, usize)> {
    let referenced = deferred_pairwise
        .iter()
        .flat_map(|pair| {
            pair.matches.iter().flat_map(|&(keypoint_i, keypoint_j)| {
                [(pair.image_i, keypoint_i), (pair.image_j, keypoint_j)]
            })
        })
        .collect::<HashSet<_>>();
    let mut links = HashMap::new();
    for pair in base_pairwise {
        let (frame_i, sensor_i) = image_assignment[pair.image_i];
        let (frame_j, sensor_j) = image_assignment[pair.image_j];
        if frame_i != frame_j || sensor_i == sensor_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            if referenced.contains(&(pair.image_i, keypoint_i)) {
                links
                    .entry((pair.image_i, keypoint_i))
                    .or_insert((pair.image_j, keypoint_j));
            }
            if referenced.contains(&(pair.image_j, keypoint_j)) {
                links
                    .entry((pair.image_j, keypoint_j))
                    .or_insert((pair.image_i, keypoint_i));
            }
        }
    }
    links
}

fn build_frame_pair_adjacency(
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    frame_count: usize,
) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); frame_count];
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let frame_i = image_assignment[pair.image_i].0;
        let frame_j = image_assignment[pair.image_j].0;
        if frame_i == frame_j {
            continue;
        }
        adjacency[frame_i].push(pair_index);
        adjacency[frame_j].push(pair_index);
    }
    adjacency
}

#[derive(Debug, Clone, Copy)]
struct DeferredRigCorrespondence {
    sensor_index: usize,
    point2d: Point2<f64>,
    point3d: Point3<f64>,
    track_index: Option<usize>,
    image: usize,
    keypoint: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DeferredRigSource {
    frame: usize,
    image: usize,
    keypoint: usize,
    track_index: Option<usize>,
}

type DeferredTargetSources = HashMap<(usize, usize), Vec<DeferredRigSource>>;

#[derive(Debug, Default)]
struct DeferredRegistrationUpdate {
    pair_visits: usize,
    correspondence_insertions: usize,
    pnp_attempts: usize,
    pnp_estimation_failures: usize,
    pnp_inlier_rejections: usize,
    registrations: usize,
    interpolation_registrations: usize,
    observations_attached: usize,
}

fn observation_track(index: &[(usize, usize)], keypoint: usize) -> Option<usize> {
    index
        .binary_search_by_key(&keypoint, |&(candidate, _)| candidate)
        .ok()
        .map(|position| index[position].1)
}

fn insert_observation_track(index: &mut Vec<(usize, usize)>, keypoint: usize, track: usize) {
    match index.binary_search_by_key(&keypoint, |&(candidate, _)| candidate) {
        Ok(position) => index[position].1 = track,
        Err(position) => index.insert(position, (keypoint, track)),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_deferred_correspondences(
    target_frame: usize,
    target_sources: &DeferredTargetSources,
    source_frame_filter: Option<usize>,
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &[WorkingTrack],
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
) -> Vec<DeferredRigCorrespondence> {
    let mut entries = target_sources
        .iter()
        .filter_map(|(&(image, keypoint), sources)| {
            sources
                .iter()
                .filter(|source| source_frame_filter.is_none_or(|frame| source.frame == frame))
                .min_by_key(|source| {
                    (
                        target_frame.abs_diff(source.frame),
                        source.track_index.is_none(),
                        source.frame,
                        source.image,
                        source.keypoint,
                    )
                })
                .copied()
                .map(|source| (image, keypoint, source))
        })
        .collect::<Vec<_>>();
    entries.sort_unstable();
    let mut used_image_tracks = HashSet::new();
    entries
        .into_iter()
        .filter(|&(image, _, source)| {
            source
                .track_index
                .is_none_or(|track| used_image_tracks.insert((image, track)))
        })
        .filter_map(|(image, keypoint, source)| {
            let point3d = source
                .track_index
                .and_then(|track| tracks[track].position)
                .or_else(|| {
                    triangulate_verified_stereo_observation(
                        rig,
                        features,
                        image_assignment,
                        image_poses,
                        config,
                        stereo_links,
                        source.image,
                        source.keypoint,
                    )
                })?;
            Some(DeferredRigCorrespondence {
                sensor_index: image_assignment[image].1,
                point2d: features[image].keypoints[keypoint],
                point3d,
                track_index: source.track_index,
                image,
                keypoint,
            })
        })
        .collect()
}

fn interpolate_deferred_pose_gaps(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    adjacency: &[Vec<usize>],
    max_gap: usize,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> usize {
    if max_gap == 0 {
        return 0;
    }
    let mut filled = 0usize;
    let mut cursor = 0usize;
    while cursor < frame_poses.len() {
        if frame_poses[cursor].is_some() {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < frame_poses.len() && frame_poses[cursor].is_none() {
            cursor += 1;
        }
        let end = cursor;
        let gap = end - start;
        if start == 0
            || end == frame_poses.len()
            || gap > max_gap
            || (start..end).any(|frame| adjacency[frame].is_empty())
        {
            continue;
        }
        let left = frame_poses[start - 1]
            .as_ref()
            .expect("gap begins after a registered frame")
            .camera_to_world();
        let right = frame_poses[end]
            .as_ref()
            .expect("bounded gap ends at a registered frame")
            .camera_to_world();
        for (offset, frame) in (start..end).enumerate() {
            let fraction = (offset + 1) as f64 / (gap + 1) as f64;
            let camera_to_world = SE3::new(
                left.rotation.slerp(&right.rotation, fraction),
                left.translation * (1.0 - fraction) + right.translation * fraction,
            );
            frame_poses[frame] = Some(Pose {
                world_to_camera: camera_to_world.inverse(),
            });
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("interpolated pose was just installed"),
                image_poses,
            );
            filled += 1;
        }
    }
    filled
}

#[derive(Debug, Default, PartialEq, Eq)]
struct IsolatedPoseRepairUpdate {
    passes: usize,
    repairs: usize,
}

/// Repair only a one-frame translation detour with registered temporal
/// neighbours. Candidates are collected from an immutable pass snapshot and
/// installed together, so ordering cannot affect the result. The rotation is
/// replaced by the midpoint SLERP as well, preserving a coherent SE(3) pose.
#[allow(clippy::too_many_arguments)]
fn repair_isolated_pose_outliers(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    seed_frame_index: usize,
    max_midpoint_error_m: f64,
    min_detour_ratio: f64,
    max_passes: usize,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> IsolatedPoseRepairUpdate {
    let mut update = IsolatedPoseRepairUpdate::default();
    for _ in 0..max_passes {
        let mut candidates = Vec::new();
        for frame in 1..frame_poses.len().saturating_sub(1) {
            if frame == seed_frame_index {
                continue;
            }
            let (Some(left), Some(current), Some(right)) = (
                frame_poses[frame - 1].as_ref(),
                frame_poses[frame].as_ref(),
                frame_poses[frame + 1].as_ref(),
            ) else {
                continue;
            };
            let left = left.camera_to_world();
            let current = current.camera_to_world();
            let right = right.camera_to_world();
            let midpoint = (left.translation + right.translation) * 0.5;
            let midpoint_error = (current.translation - midpoint).norm();
            let endpoint_chord = (right.translation - left.translation).norm();
            let detour = (current.translation - left.translation).norm()
                + (right.translation - current.translation).norm();
            // One millimetre avoids an unstable ratio for a stationary rig;
            // the independent midpoint-error gate still controls scale.
            if midpoint_error <= max_midpoint_error_m
                || detour <= min_detour_ratio * endpoint_chord.max(1.0e-3)
            {
                continue;
            }
            candidates.push((
                frame,
                Pose {
                    world_to_camera: SE3::new(left.rotation.slerp(&right.rotation, 0.5), midpoint)
                        .inverse(),
                },
            ));
        }
        if candidates.is_empty() {
            break;
        }
        update.passes += 1;
        update.repairs += candidates.len();
        for (frame, pose) in candidates {
            frame_poses[frame] = Some(pose);
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("repaired pose was just installed"),
                image_poses,
            );
        }
    }
    update
}

#[derive(Debug, Clone, Copy)]
struct PairedPoseJumpCandidate {
    first_jump: usize,
    second_jump: usize,
    closure_ratio: f64,
    offset: Vector3<f64>,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct PairedPoseJumpRepairUpdate {
    repairs: usize,
    repaired_frames: usize,
}

/// Repair short, mutually cancelling translation jumps in one registered
/// model.  If `v_i = C_(i+1) - C_i` and `v_j = C_(j+1) - C_j` are large but
/// nearly opposite, the observations are consistent with a world-frame
/// offset applied to the contiguous segment `i+1..=j`.  The symmetric offset
/// estimate `(v_i - v_j) / 2` is removed from every pose in that segment,
/// preserving its internal motion and each pose's rotation.
///
/// Candidate generation examines at most `max_frame_span` successors for each
/// frame.  Missing poses terminate a candidate interval, so model boundaries
/// are never crossed.  Candidate selection is deterministic and temporal: for
/// each first jump, the eligible second jump with the smallest closure ratio
/// (then the smallest frame index) wins, and overlapping intervals are
/// discarded.  Thus the
/// repair work and temporary state are bounded by the sparse `F * span`
/// envelope rather than an image/track Cartesian product.
#[allow(clippy::too_many_arguments)]
fn repair_paired_pose_jumps(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    seed_frame_index: usize,
    absolute_step_m: f64,
    min_step_ratio: f64,
    max_frame_span: usize,
    max_closure_ratio: f64,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
) -> PairedPoseJumpRepairUpdate {
    let frame_count = frame_poses.len().min(frames.len());
    if frame_count < 3 || max_frame_span == 0 {
        return PairedPoseJumpRepairUpdate::default();
    }

    // Build the local-velocity vector once.  An absent pose leaves a hole,
    // which is also the model-boundary marker used by candidate generation.
    let mut velocities = vec![None; frame_count.saturating_sub(1)];
    let mut step_lengths = Vec::new();
    for frame in 0..frame_count.saturating_sub(1) {
        let (Some(left), Some(right)) =
            (frame_poses[frame].as_ref(), frame_poses[frame + 1].as_ref())
        else {
            continue;
        };
        let left = left.camera_center_world();
        let right = right.camera_center_world();
        let velocity = right - left;
        let length = velocity.norm();
        if !velocity.iter().all(|value| value.is_finite()) || !length.is_finite() {
            continue;
        }
        velocities[frame] = Some(velocity);
        step_lengths.push(length);
    }
    if step_lengths.is_empty() {
        return PairedPoseJumpRepairUpdate::default();
    }
    step_lengths.sort_unstable_by(f64::total_cmp);
    let midpoint = step_lengths.len() / 2;
    let global_median_step = if step_lengths.len() % 2 == 0 {
        (step_lengths[midpoint - 1] + step_lengths[midpoint]) * 0.5
    } else {
        step_lengths[midpoint]
    };
    let required_step = absolute_step_m.max(min_step_ratio * global_median_step);
    if !global_median_step.is_finite() || !required_step.is_finite() || required_step <= 0.0 {
        return PairedPoseJumpRepairUpdate::default();
    }

    // Prefix counts make the all-registered interval test O(1), keeping the
    // total detector cost linear in the configured span.
    let mut missing_prefix = vec![0usize; frame_count + 1];
    for frame in 0..frame_count {
        missing_prefix[frame + 1] =
            missing_prefix[frame] + usize::from(frame_poses[frame].is_none());
    }
    let interval_is_registered =
        |start: usize, end: usize| missing_prefix[end + 1] == missing_prefix[start];

    let mut candidates = Vec::new();
    for first_jump in 0..velocities.len() {
        let Some(first_velocity) = velocities[first_jump] else {
            continue;
        };
        let first_length = first_velocity.norm();
        if first_length < required_step {
            continue;
        }
        let last_jump = first_jump
            .saturating_add(max_frame_span)
            .min(velocities.len().saturating_sub(1));
        let mut best = None;
        for (second_jump, second_velocity) in velocities
            .iter()
            .enumerate()
            .take(last_jump + 1)
            .skip(first_jump + 1)
        {
            // A missing pose can only make later intervals invalid until the
            // right endpoint; avoid repeatedly scanning the same gap.
            if second_velocity.is_none() || !interval_is_registered(first_jump, second_jump + 1) {
                continue;
            }
            if first_jump < seed_frame_index && seed_frame_index <= second_jump {
                // The seed is a gauge anchor.  It may be an outer endpoint,
                // but it must not be shifted as part of the repaired segment.
                continue;
            }
            let second_velocity =
                second_velocity.expect("the candidate's second velocity was checked above");
            let second_length = second_velocity.norm();
            if second_length < required_step {
                continue;
            }
            let denominator = first_length.max(second_length);
            if denominator <= 0.0 || !denominator.is_finite() {
                continue;
            }
            let closure_ratio = (first_velocity + second_velocity).norm() / denominator;
            if !closure_ratio.is_finite() || closure_ratio > max_closure_ratio {
                continue;
            }
            let candidate = PairedPoseJumpCandidate {
                first_jump,
                second_jump,
                closure_ratio,
                offset: (first_velocity - second_velocity) * 0.5,
            };
            let replace = best.is_none_or(|current: PairedPoseJumpCandidate| {
                closure_ratio.total_cmp(&current.closure_ratio).is_lt()
                    || (closure_ratio.total_cmp(&current.closure_ratio).is_eq()
                        && second_jump < current.second_jump)
            });
            if replace {
                best = Some(candidate);
            }
        }
        if let Some(candidate) = best {
            candidates.push(candidate);
        }
    }

    // `candidates` has one entry per first jump and is therefore already in
    // temporal order.  Reserve the full candidate interval, including its
    // endpoints, so adjacent repairs cannot share an untrusted boundary.
    let mut occupied = vec![false; frame_count];
    let mut selected = Vec::new();
    for candidate in candidates {
        let interval_end = candidate.second_jump + 1;
        if occupied[candidate.first_jump..=interval_end]
            .iter()
            .any(|occupied| *occupied)
        {
            continue;
        }
        occupied[candidate.first_jump..=interval_end].fill(true);
        selected.push(candidate);
    }

    // Construct every replacement from the immutable pass snapshot before
    // writing any pose.  If an unexpected non-finite state appears, abort the
    // whole repair transaction rather than exposing a partially shifted map.
    let mut pending = Vec::new();
    for candidate in &selected {
        for (frame, current) in frame_poses
            .iter()
            .enumerate()
            .take(candidate.second_jump + 1)
            .skip(candidate.first_jump + 1)
        {
            let Some(current) = current.as_ref() else {
                return PairedPoseJumpRepairUpdate::default();
            };
            let current_center = current.camera_center_world();
            let repaired_center = current_center - candidate.offset;
            if !repaired_center.coords.iter().all(|value| value.is_finite()) {
                return PairedPoseJumpRepairUpdate::default();
            }
            let rotation = current.world_to_camera.rotation;
            let translation = -rotation.transform_vector(&repaired_center.coords);
            pending.push((frame, Pose::from_world_to_camera(rotation, translation)));
        }
    }
    let update = PairedPoseJumpRepairUpdate {
        repairs: selected.len(),
        repaired_frames: pending.len(),
    };
    for (frame, pose) in pending {
        frame_poses[frame] = Some(pose);
        install_image_poses(
            rig,
            &frames[frame],
            frame_poses[frame]
                .as_ref()
                .expect("repaired pose was just installed"),
            image_poses,
        );
    }
    update
}

/// Register missing frames from a lower-priority verified-pair suffix without
/// allowing that suffix to alter the accepted base structure. Every suffix
/// pair is visited at most once, when either endpoint first becomes
/// registered, and only PnP-inlier observations are attached to base tracks.
#[allow(clippy::too_many_arguments)]
fn register_deferred_frames(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    config: &RigSfmConfig,
    frame_poses: &mut [Option<Pose>],
    image_poses: &mut [Option<Pose>],
    tracks: &mut [WorkingTrack],
    image_tracks: &mut [Vec<(usize, usize)>],
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
) -> DeferredRegistrationUpdate {
    let adjacency = build_frame_pair_adjacency(pairwise, image_assignment, frames.len());
    let mut pair_visited = vec![false; pairwise.len()];
    let mut source_queue = frame_poses
        .iter()
        .enumerate()
        .filter_map(|(frame, pose)| pose.as_ref().map(|_| frame))
        .collect::<Vec<_>>();
    let mut target_candidates = vec![DeferredTargetSources::new(); frames.len()];
    let mut versions = vec![0usize; frames.len()];
    let mut attempted_versions = vec![None; frames.len()];
    let mut candidate_support = vec![0usize; frames.len()];
    let mut candidate_sensors = vec![0usize; frames.len()];
    let mut candidate_direct = vec![0usize; frames.len()];
    let mut heap = BinaryHeap::new();
    let mut update = DeferredRegistrationUpdate::default();
    let required_inliers = config
        .deferred_registration_min_pnp_inliers
        .unwrap_or(config.min_pnp_inliers)
        .max(6);
    let required_sensors = config
        .deferred_registration_min_pnp_sensors
        .unwrap_or(config.min_pnp_sensors);
    let pnp = GeneralizedPnPRansac {
        iterations: config
            .deferred_registration_pnp_max_iterations
            .unwrap_or(config.pnp_max_iterations),
        reprojection_threshold: config.max_reprojection_error_px,
        seed: config.ransac_seed,
        ..GeneralizedPnPRansac::default()
    };

    loop {
        let mut changed_frames = HashSet::new();
        for source_frame in std::mem::take(&mut source_queue) {
            if frame_poses[source_frame].is_none() {
                continue;
            }
            for &pair_index in &adjacency[source_frame] {
                if pair_visited[pair_index] {
                    continue;
                }
                let pair = &pairwise[pair_index];
                let frame_i = image_assignment[pair.image_i].0;
                let frame_j = image_assignment[pair.image_j].0;
                let (target_frame, source_is_i) = if frame_i == source_frame {
                    (frame_j, true)
                } else {
                    (frame_i, false)
                };
                if target_frame == source_frame || frame_poses[target_frame].is_some() {
                    pair_visited[pair_index] = true;
                    continue;
                }
                pair_visited[pair_index] = true;
                update.pair_visits += 1;
                let mut inserted = 0usize;
                for &(keypoint_i, keypoint_j) in &pair.matches {
                    let (source_image, source_keypoint, target_image, target_keypoint) =
                        if source_is_i {
                            (pair.image_i, keypoint_i, pair.image_j, keypoint_j)
                        } else {
                            (pair.image_j, keypoint_j, pair.image_i, keypoint_i)
                        };
                    let track_index =
                        observation_track(&image_tracks[source_image], source_keypoint)
                            .filter(|&track| tracks[track].position.is_some());
                    if track_index.is_none()
                        && !stereo_links.contains_key(&(source_image, source_keypoint))
                    {
                        continue;
                    }
                    let occupied_by_positioned_track =
                        observation_track(&image_tracks[target_image], target_keypoint)
                            .is_some_and(|existing| tracks[existing].position.is_some());
                    if occupied_by_positioned_track
                        || track_index.is_some_and(|track_index| {
                            tracks[track_index]
                                .observations
                                .iter()
                                .any(|&(image, _)| image == target_image)
                        })
                    {
                        continue;
                    }
                    let source = DeferredRigSource {
                        frame: source_frame,
                        image: source_image,
                        keypoint: source_keypoint,
                        track_index,
                    };
                    target_candidates[target_frame]
                        .entry((target_image, target_keypoint))
                        .or_default();
                    let sources = target_candidates[target_frame]
                        .get_mut(&(target_image, target_keypoint))
                        .expect("candidate entry was just installed");
                    if !sources.contains(&source) {
                        sources.push(source);
                        inserted += 1;
                    }
                }
                if inserted > 0 {
                    update.correspondence_insertions += inserted;
                    changed_frames.insert(target_frame);
                }
            }
        }
        for frame in changed_frames {
            versions[frame] += 1;
            heap.push((
                target_candidates[frame].len(),
                Reverse(frame),
                versions[frame],
            ));
        }

        let mut registered_any = false;
        while let Some((_, Reverse(frame), version)) = heap.pop() {
            if frame_poses[frame].is_some()
                || versions[frame] != version
                || attempted_versions[frame] == Some(version)
            {
                continue;
            }
            let candidates = collect_deferred_correspondences(
                frame,
                &target_candidates[frame],
                None,
                rig,
                features,
                image_assignment,
                image_poses,
                config,
                tracks,
                stereo_links,
            );
            candidate_support[frame] = candidates.len();
            candidate_sensors[frame] = candidates
                .iter()
                .map(|candidate| candidate.sensor_index)
                .collect::<HashSet<_>>()
                .len();
            candidate_direct[frame] = candidates
                .iter()
                .filter(|candidate| candidate.track_index.is_none())
                .count();
            if candidates.len() < required_inliers || candidate_sensors[frame] < required_sensors {
                continue;
            }
            attempted_versions[frame] = Some(version);
            let mut source_support = HashMap::<usize, usize>::new();
            for sources in target_candidates[frame].values() {
                for source in sources {
                    *source_support.entry(source.frame).or_default() += 1;
                }
            }
            let mut source_frames = source_support.into_iter().collect::<Vec<_>>();
            source_frames.sort_unstable_by_key(|&(source, support)| {
                (Reverse(support), frame.abs_diff(source), source)
            });
            let mut hypotheses = vec![None];
            hypotheses.extend(
                source_frames
                    .into_iter()
                    .take(31)
                    .map(|(source, _)| Some(source)),
            );
            let mut accepted = None;
            for source_filter in hypotheses {
                let hypothesis = if source_filter.is_none() {
                    candidates.clone()
                } else {
                    collect_deferred_correspondences(
                        frame,
                        &target_candidates[frame],
                        source_filter,
                        rig,
                        features,
                        image_assignment,
                        image_poses,
                        config,
                        tracks,
                        stereo_links,
                    )
                };
                let sensors = hypothesis
                    .iter()
                    .map(|candidate| candidate.sensor_index)
                    .collect::<HashSet<_>>()
                    .len();
                if hypothesis.len() < required_inliers || sensors < required_sensors {
                    continue;
                }
                update.pnp_attempts += 1;
                let correspondences = hypothesis
                    .iter()
                    .map(|candidate| GeneralizedCorrespondence2D3D {
                        sensor_index: candidate.sensor_index,
                        point2d: candidate.point2d,
                        point3d: candidate.point3d,
                        confidence: None,
                    })
                    .collect::<Vec<_>>();
                let Some(report) = pnp.estimate(rig, &correspondences) else {
                    update.pnp_estimation_failures += 1;
                    continue;
                };
                if report.inliers.len() < required_inliers {
                    update.pnp_inlier_rejections += 1;
                    continue;
                }
                accepted = Some((report, hypothesis));
                break;
            }
            let Some((report, candidates)) = accepted else {
                continue;
            };
            frame_poses[frame] = Some(report.pose);
            install_image_poses(
                rig,
                &frames[frame],
                frame_poses[frame]
                    .as_ref()
                    .expect("pose was just installed"),
                image_poses,
            );
            for &inlier in &report.inliers {
                let candidate = candidates[inlier];
                let Some(track_index) = candidate.track_index else {
                    continue;
                };
                let existing =
                    observation_track(&image_tracks[candidate.image], candidate.keypoint);
                if existing.is_some_and(|track| tracks[track].position.is_some())
                    || tracks[track_index]
                        .observations
                        .iter()
                        .any(|&(image, _)| image == candidate.image)
                {
                    continue;
                }
                if let Some(existing) = existing {
                    tracks[existing].observations.retain(|&(image, keypoint)| {
                        image != candidate.image || keypoint != candidate.keypoint
                    });
                }
                tracks[track_index]
                    .observations
                    .push((candidate.image, candidate.keypoint));
                insert_observation_track(
                    &mut image_tracks[candidate.image],
                    candidate.keypoint,
                    track_index,
                );
                update.observations_attached += 1;
            }
            update.registrations += 1;
            source_queue.push(frame);
            registered_any = true;
        }
        if !registered_any {
            break;
        }
    }
    update.interpolation_registrations = interpolate_deferred_pose_gaps(
        rig,
        frames,
        &adjacency,
        config.deferred_registration_max_interpolation_gap,
        frame_poses,
        image_poses,
    );
    if std::env::var_os("VISLOC_DEFERRED_DEBUG").is_some() {
        for frame in 0..frames.len() {
            if frame_poses[frame].is_none() && !adjacency[frame].is_empty() {
                eprintln!(
                    "rig-deferred-debug: frame={frame} adjacent_pairs={} raw_targets={} usable={} direct={} sensors={} required_inliers={} required_sensors={}",
                    adjacency[frame].len(),
                    target_candidates[frame].len(),
                    candidate_support[frame],
                    candidate_direct[frame],
                    candidate_sensors[frame],
                    required_inliers,
                    required_sensors,
                );
            }
        }
    }
    update
}

#[derive(Debug, Default)]
struct DirectBridgeUpdate {
    pair_visits: usize,
    insertions: usize,
}

#[derive(Debug, Clone)]
struct MotionBridgeCandidate {
    frame: usize,
    pose: Pose,
    inliers: usize,
    frame_gap: usize,
}

#[derive(Debug, Default)]
struct MotionBridgeUpdate {
    candidates: Vec<MotionBridgeCandidate>,
    pair_visits: usize,
    estimation_failures: usize,
    rotation_rejections: usize,
}

#[allow(clippy::too_many_arguments)]
fn collect_motion_bridge_candidates(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    pair_adjacency: &[Vec<usize>],
    source_queue: &mut Vec<usize>,
    pair_visited: &mut [bool],
) -> MotionBridgeUpdate {
    let estimator = RelativePoseEstimator::default();
    let mut update = MotionBridgeUpdate::default();
    let mut best_by_frame = HashMap::<usize, MotionBridgeCandidate>::new();
    for source_frame in std::mem::take(source_queue) {
        let Some(source_frame_pose) = frame_poses[source_frame].as_ref() else {
            continue;
        };
        for &pair_index in &pair_adjacency[source_frame] {
            if pair_visited[pair_index] {
                continue;
            }
            let pair = &pairwise[pair_index];
            let frame_i = image_assignment[pair.image_i].0;
            let frame_j = image_assignment[pair.image_j].0;
            let (target_frame, source_is_i) = if frame_i == source_frame {
                (frame_j, true)
            } else {
                (frame_i, false)
            };
            if frame_poses[target_frame].is_some() {
                pair_visited[pair_index] = true;
                continue;
            }
            let frame_gap = source_frame.abs_diff(target_frame);
            if frame_gap == 0 || frame_gap > config.motion_bridge_max_frame_gap {
                pair_visited[pair_index] = true;
                continue;
            }
            let prior_frame = (1..=config.motion_bridge_max_frame_gap).find_map(|prior_gap| {
                let candidate = if target_frame > source_frame {
                    source_frame.checked_sub(prior_gap)
                } else {
                    source_frame.checked_add(prior_gap)
                }?;
                (candidate < frames.len() && frame_poses[candidate].is_some())
                    .then_some((candidate, prior_gap))
            });
            let Some((prior_frame, prior_gap)) = prior_frame else {
                pair_visited[pair_index] = true;
                continue;
            };
            let prior_pose = frame_poses[prior_frame]
                .as_ref()
                .expect("selected prior frame is registered");
            pair_visited[pair_index] = true;
            update.pair_visits += 1;
            let (source_image, target_image) = if source_is_i {
                (pair.image_i, pair.image_j)
            } else {
                (pair.image_j, pair.image_i)
            };
            let correspondences = pair
                .matches
                .iter()
                .map(|&(keypoint_i, keypoint_j)| {
                    let (source_keypoint, target_keypoint) = if source_is_i {
                        (keypoint_i, keypoint_j)
                    } else {
                        (keypoint_j, keypoint_i)
                    };
                    TwoViewCorrespondence::new(
                        features[source_image].keypoints[source_keypoint],
                        features[target_image].keypoints[target_keypoint],
                    )
                })
                .collect::<Vec<_>>();
            let source_sensor = &rig.sensors()[image_assignment[source_image].1];
            let target_sensor = &rig.sensors()[image_assignment[target_image].1];
            let Some(relative) = estimator.estimate_with_cameras(
                &correspondences,
                &source_sensor.camera,
                &target_sensor.camera,
            ) else {
                update.estimation_failures += 1;
                continue;
            };
            if relative.inliers.len() < config.motion_bridge_min_inliers {
                update.estimation_failures += 1;
                continue;
            }
            let observed_delta = source_frame_pose
                .world_to_camera
                .compose(&prior_pose.world_to_camera.inverse());
            let motion_ratio = frame_gap as f64 / prior_gap as f64;
            let predicted_delta = SE3::new(
                nalgebra::UnitQuaternion::from_scaled_axis(
                    observed_delta.rotation.scaled_axis() * motion_ratio,
                ),
                observed_delta.translation * motion_ratio,
            );
            let to_rig_rotation = |sensor_rotation| {
                target_sensor.sensor_from_rig.rotation.inverse()
                    * sensor_rotation
                    * source_sensor.sensor_from_rig.rotation
            };
            let mut rotations = vec![to_rig_rotation(relative.previous_to_current.rotation)];
            if let Some((alternate, _)) = relative.alternate {
                rotations.push(to_rig_rotation(alternate));
            }
            rotations.sort_by(|left, right| {
                left.angle_to(&predicted_delta.rotation)
                    .total_cmp(&right.angle_to(&predicted_delta.rotation))
            });
            let rotation = rotations[0];
            let rotation_deviation_deg = rotation.angle_to(&predicted_delta.rotation).to_degrees();
            if rotation_deviation_deg > config.motion_bridge_max_rotation_deviation_deg {
                update.rotation_rejections += 1;
                continue;
            }
            let bridge_delta = SE3::new(rotation, predicted_delta.translation);
            let candidate = MotionBridgeCandidate {
                frame: target_frame,
                pose: Pose {
                    world_to_camera: bridge_delta.compose(&source_frame_pose.world_to_camera),
                },
                inliers: relative.inliers.len(),
                frame_gap,
            };
            let replace = best_by_frame
                .get(&target_frame)
                .is_none_or(|current| candidate.inliers > current.inliers);
            if replace {
                best_by_frame.insert(target_frame, candidate);
            }
        }
    }
    update.candidates = best_by_frame.into_values().collect();
    update
}

#[allow(clippy::too_many_arguments)]
fn append_direct_stereo_pnp_correspondences(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
    pair_adjacency: &[Vec<usize>],
    source_queue: &mut Vec<usize>,
    pair_visited: &mut [bool],
    target_observations: &mut [HashSet<(usize, usize)>],
    frame_correspondences: &mut [Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> DirectBridgeUpdate {
    let mut update = DirectBridgeUpdate::default();
    let mut changed_frames = HashSet::new();
    let queued = std::mem::take(source_queue);
    for source_frame in queued {
        if image_poses[frames[source_frame].images[0].image_index].is_none() {
            continue;
        }
        for &pair_index in &pair_adjacency[source_frame] {
            if pair_visited[pair_index] {
                continue;
            }
            let pair = &pairwise[pair_index];
            let frame_i = image_assignment[pair.image_i].0;
            let frame_j = image_assignment[pair.image_j].0;
            let target_frame = if frame_i == source_frame {
                frame_j
            } else {
                frame_i
            };
            if image_poses[frames[target_frame].images[0].image_index].is_some() {
                pair_visited[pair_index] = true;
                continue;
            }
            if source_frame.abs_diff(target_frame) > config.direct_stereo_pnp_max_frame_gap {
                pair_visited[pair_index] = true;
                continue;
            }
            pair_visited[pair_index] = true;
            update.pair_visits += 1;
            let source_is_i = frame_i == source_frame;
            for &(keypoint_i, keypoint_j) in &pair.matches {
                let (source_image, source_keypoint, target_image, target_keypoint) = if source_is_i
                {
                    (pair.image_i, keypoint_i, pair.image_j, keypoint_j)
                } else {
                    (pair.image_j, keypoint_j, pair.image_i, keypoint_i)
                };
                if target_observations[target_frame].contains(&(target_image, target_keypoint)) {
                    continue;
                }
                let Some(point3d) = triangulate_verified_stereo_observation(
                    rig,
                    features,
                    image_assignment,
                    image_poses,
                    config,
                    stereo_links,
                    source_image,
                    source_keypoint,
                ) else {
                    continue;
                };
                target_observations[target_frame].insert((target_image, target_keypoint));
                frame_correspondences[target_frame].push(CachedRigCorrespondence {
                    sensor_index: image_assignment[target_image].1,
                    point2d: features[target_image].keypoints[target_keypoint],
                    track_index: None,
                    direct_point3d: Some(point3d),
                });
                update.insertions += 1;
                changed_frames.insert(target_frame);
            }
        }
    }
    for frame in changed_frames {
        frame_versions[frame] += 1;
        heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    update
}

#[allow(clippy::too_many_arguments)]
fn triangulate_verified_stereo_observation(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    stereo_links: &HashMap<(usize, usize), (usize, usize)>,
    image: usize,
    keypoint: usize,
) -> Option<Point3<f64>> {
    let &(other_image, other_keypoint) = stereo_links.get(&(image, keypoint))?;
    let pose = image_poses[image].as_ref()?;
    let other_pose = image_poses[other_image].as_ref()?;
    let sensor = &rig.sensors()[image_assignment[image].1];
    let other_sensor = &rig.sensors()[image_assignment[other_image].1];
    let bearing = sensor
        .camera
        .normalize_pixel(&features[image].keypoints[keypoint])?;
    let other_bearing = other_sensor
        .camera
        .normalize_pixel(&features[other_image].keypoints[other_keypoint])?;
    let direction = pose
        .camera_to_world()
        .rotation
        .transform_vector(&Vector3::new(bearing.x, bearing.y, 1.0).normalize())
        .normalize();
    let other_direction = other_pose
        .camera_to_world()
        .rotation
        .transform_vector(&Vector3::new(other_bearing.x, other_bearing.y, 1.0).normalize())
        .normalize();
    let angle_deg = direction
        .dot(&other_direction)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    if angle_deg < config.direct_stereo_min_triangulation_angle_deg {
        return None;
    }
    let point = closest_ray_midpoint(
        &pose.camera_center_world(),
        &direction,
        &other_pose.camera_center_world(),
        &other_direction,
    )?;
    for (check_image, check_keypoint) in [(image, keypoint), (other_image, other_keypoint)] {
        let check_pose = image_poses[check_image].as_ref()?;
        let check_sensor = &rig.sensors()[image_assignment[check_image].1];
        let point_camera = check_pose.transform_world_point(&point);
        if point_camera.z <= 0.0 {
            return None;
        }
        let projected = check_sensor.camera.project(&point_camera)?;
        if (projected - features[check_image].keypoints[check_keypoint]).norm()
            > config.max_reprojection_error_px
        {
            return None;
        }
    }
    Some(point)
}

fn append_landmark_correspondences(
    landmark_indices: &[usize],
    tracks: &[WorkingTrack],
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    frame_correspondences: &mut [Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> usize {
    let mut changed_frames = HashSet::new();
    let mut insertions = 0;
    for &track_index in landmark_indices {
        let track = &tracks[track_index];
        if track.position.is_none() {
            continue;
        }
        for &(image, keypoint) in &track.observations {
            let (frame, sensor) = image_assignment[image];
            frame_correspondences[frame].push(CachedRigCorrespondence {
                sensor_index: sensor,
                point2d: features[image].keypoints[keypoint],
                track_index: Some(track_index),
                direct_point3d: None,
            });
            insertions += 1;
            changed_frames.insert(frame);
        }
    }
    for frame in changed_frames {
        frame_versions[frame] += 1;
        heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    insertions
}

#[allow(clippy::too_many_arguments)]
fn requeue_connected_frames(
    track_indices: &HashSet<usize>,
    tracks: &[WorkingTrack],
    image_assignment: &[(usize, usize)],
    frame_poses: &[Option<Pose>],
    frame_correspondences: &[Vec<CachedRigCorrespondence>],
    frame_versions: &mut [usize],
    heap: &mut BinaryHeap<(usize, Reverse<usize>, usize)>,
) -> usize {
    let mut frames = track_indices
        .iter()
        .flat_map(|track| tracks[*track].observations.iter())
        .map(|(image, _)| image_assignment[*image].0)
        .filter(|frame| frame_poses[*frame].is_none())
        .collect::<Vec<_>>();
    frames.sort_unstable();
    frames.dedup();
    for &frame in &frames {
        frame_versions[frame] += 1;
        heap.push((
            frame_correspondences[frame].len(),
            Reverse(frame),
            frame_versions[frame],
        ));
    }
    frames.len()
}

#[allow(clippy::too_many_arguments)]
fn retriangulate_unanchored_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
    track_indices: &HashSet<usize>,
) -> usize {
    let mut track_indices = track_indices.iter().copied().collect::<Vec<_>>();
    track_indices.sort_unstable();
    let mut refreshed = 0usize;
    for track_index in track_indices {
        let track = &mut tracks[track_index];
        let Some(current) = track.position else {
            continue;
        };
        if track.metric_anchored {
            continue;
        }
        let rays = track
            .observations
            .iter()
            .filter_map(|&(image, keypoint)| {
                let pose = image_poses[image].as_ref()?;
                let sensor = &rig.sensors()[image_assignment[image].1];
                let normalized = sensor
                    .camera
                    .normalize_pixel(&features[image].keypoints[keypoint])?;
                let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
                let direction_world = pose
                    .camera_to_world()
                    .rotation
                    .transform_vector(&bearing_camera)
                    .normalize();
                Some((image, keypoint, pose.camera_center_world(), direction_world))
            })
            .collect::<Vec<_>>();
        if rays.len() < 2 {
            continue;
        }

        // Bound pair generation independently of track length. Long tracks
        // are sampled deterministically, so this diagnostic recovery path
        // cannot reintroduce quadratic state or work at 10k scale.
        let rays = bounded_evenly_spaced_indices(rays.len(), 16)
            .into_iter()
            .map(|index| rays[index])
            .collect::<Vec<_>>();

        // A long conflict-preserving track can contain a bad transitive edge.
        // Evaluate a small, deterministic set of the widest-baseline ray pairs
        // instead of trusting one pair or materializing any global dense state.
        let mut pairs = Vec::new();
        for left in 0..rays.len() {
            for right in (left + 1)..rays.len() {
                let angle = rays[left].3.dot(&rays[right].3).clamp(-1.0, 1.0).acos();
                if angle.to_degrees() >= config.min_triangulation_angle_deg {
                    pairs.push((angle, left, right));
                }
            }
        }
        pairs.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });
        pairs.truncate(16);

        let current_score = rig_point_support(
            rig,
            features,
            image_assignment,
            image_poses,
            track,
            &current,
            config.max_reprojection_error_px,
        );
        let mut best = None;
        for (_, left, right) in pairs {
            let Some(candidate) =
                closest_ray_midpoint(&rays[left].2, &rays[left].3, &rays[right].2, &rays[right].3)
            else {
                continue;
            };
            let score = rig_point_support(
                rig,
                features,
                image_assignment,
                image_poses,
                track,
                &candidate,
                config.max_reprojection_error_px,
            );
            if score.0 < 2 {
                continue;
            }
            if best
                .as_ref()
                .is_none_or(|(_, best_score): &(Point3<f64>, (usize, f64))| score > *best_score)
            {
                best = Some((candidate, score));
            }
        }
        let Some((candidate, candidate_score)) = best else {
            continue;
        };
        if candidate_score > current_score {
            track.position = Some(candidate);
            refreshed += 1;
        }
    }
    refreshed
}

fn rig_point_support(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    track: &WorkingTrack,
    point: &Point3<f64>,
    max_reprojection_error_px: f64,
) -> (usize, f64) {
    let mut inliers = 0usize;
    let mut negative_squared_error = 0.0;
    for &(image, keypoint) in &track.observations {
        let Some(pose) = image_poses[image].as_ref() else {
            continue;
        };
        let sensor = &rig.sensors()[image_assignment[image].1];
        let point_camera = pose.transform_world_point(point);
        if point_camera.z <= 0.0 {
            continue;
        }
        let Some(projected) = sensor.camera.project(&point_camera) else {
            continue;
        };
        let squared_error = (projected - features[image].keypoints[keypoint]).norm_squared();
        if squared_error <= max_reprojection_error_px * max_reprojection_error_px {
            inliers += 1;
            negative_squared_error -= squared_error;
        }
    }
    (inliers, negative_squared_error)
}

#[derive(Debug, Default)]
struct TriangulationUpdate {
    landmarks: Vec<usize>,
    attempts: usize,
    robust_tracks: usize,
    pruned_observations: usize,
    majority_rejections: usize,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct DeferredRetriangulationUpdate {
    tracks: usize,
    observations: usize,
}

fn append_retriangulated_deferred_tracks(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    deferred_pairwise: &[PairwiseMatches],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut Vec<WorkingTrack>,
) -> DeferredRetriangulationUpdate {
    let owned = tracks
        .iter()
        .flat_map(|track| track.observations.iter().copied())
        .collect::<HashSet<_>>();
    let deferred_config = RigSfmConfig {
        track_builder: if config.deferred_retriangulation_metric_temporal_quadrilateral_tracks {
            RigTrackBuilder::MetricTemporalQuadrilateral
        } else if config.deferred_retriangulation_metric_temporal_cycle_tracks {
            RigTrackBuilder::MetricTemporalCycle
        } else {
            config.track_builder
        },
        ..*config
    };
    let candidates = build_rig_track_output(
        features,
        deferred_pairwise,
        image_assignment,
        &deferred_config,
    )
    .tracks;
    let first_new_track = tracks.len();
    tracks.extend(
        candidates
            .into_iter()
            .filter(|observations| {
                observations.len() >= config.min_track_length
                    && (config.deferred_retriangulation_min_metric_frames == 0
                        || metric_anchor_frame_count(observations, image_assignment)
                            >= config.deferred_retriangulation_min_metric_frames)
                    && observations
                        .iter()
                        .all(|observation| !owned.contains(observation))
            })
            .map(|observations| WorkingTrack {
                metric_anchored: track_is_metric_anchored(&observations, image_assignment),
                observations,
                position: None,
            }),
    );
    let frontier = (first_new_track..tracks.len()).collect::<HashSet<_>>();
    let _ = triangulate_frontier(
        rig,
        features,
        image_assignment,
        image_poses,
        config,
        tracks,
        frontier,
    );
    retain_positioned_deferred_tracks(tracks, first_new_track)
}

fn metric_anchor_frame_count(
    observations: &[(usize, usize)],
    image_assignment: &[(usize, usize)],
) -> usize {
    let mut first_sensor = HashMap::<usize, usize>::new();
    let mut metric_frames = HashSet::new();
    for &(image, _) in observations {
        let (frame, sensor) = image_assignment[image];
        if first_sensor
            .insert(frame, sensor)
            .is_some_and(|previous| previous != sensor)
        {
            metric_frames.insert(frame);
        }
    }
    metric_frames.len()
}

fn retain_positioned_deferred_tracks(
    tracks: &mut Vec<WorkingTrack>,
    first_new_track: usize,
) -> DeferredRetriangulationUpdate {
    let mut update = DeferredRetriangulationUpdate::default();
    let mut index = 0usize;
    tracks.retain(|track| {
        let keep = index < first_new_track || track.position.is_some();
        if index >= first_new_track && keep {
            update.tracks += 1;
            update.observations += track.observations.len();
        }
        index += 1;
        keep
    });
    update
}

fn triangulate_frontier(
    rig: &GeneralizedCameraRig,
    features: &[FeatureSet],
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    config: &RigSfmConfig,
    tracks: &mut [WorkingTrack],
    frontier: HashSet<usize>,
) -> TriangulationUpdate {
    let mut update = TriangulationUpdate::default();
    // Hash iteration order is process-randomized.  The order in which newly
    // triangulated landmarks enter each frame cache changes the indexed
    // RANSAC samples, so canonicalize it before any numeric work.
    let mut frontier = frontier.into_iter().collect::<Vec<_>>();
    frontier.sort_unstable();
    for track_index in frontier {
        if tracks[track_index].position.is_some() {
            continue;
        }
        update.attempts += 1;
        let rays = tracks[track_index]
            .observations
            .iter()
            .filter_map(|(image, keypoint)| {
                let pose = image_poses[*image].as_ref()?;
                let sensor = &rig.sensors()[image_assignment[*image].1];
                let normalized = sensor
                    .camera
                    .normalize_pixel(&features[*image].keypoints[*keypoint])?;
                let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
                let direction_world = pose
                    .camera_to_world()
                    .rotation
                    .transform_vector(&bearing_camera)
                    .normalize();
                Some((
                    *image,
                    *keypoint,
                    pose.camera_center_world(),
                    direction_world,
                ))
            })
            .collect::<Vec<_>>();
        // A long track may contain thousands of observations. Searching its
        // full Cartesian ray-pair set is quadratic and can dominate a large
        // reconstruction. Keep the input scan linear, then choose a stable,
        // bounded subset for the widest-baseline search.
        let candidate_rays = bounded_evenly_spaced_indices(rays.len(), 1_024);
        let mut best = None;
        for left_offset in 0..candidate_rays.len() {
            for right_offset in (left_offset + 1)..candidate_rays.len() {
                let left = candidate_rays[left_offset];
                let right = candidate_rays[right_offset];
                let cosine = rays[left].3.dot(&rays[right].3).clamp(-1.0, 1.0);
                let angle = cosine.acos();
                if best.is_none_or(|(_, _, best_angle)| angle > best_angle) {
                    best = Some((left, right, angle));
                }
            }
        }
        let Some((left, right, angle)) = best else {
            continue;
        };
        if angle.to_degrees() < config.min_triangulation_angle_deg {
            continue;
        }
        let Some(point) =
            closest_ray_midpoint(&rays[left].2, &rays[left].3, &rays[right].2, &rays[right].3)
        else {
            continue;
        };
        let observation_is_inlier = |&(image, keypoint): &(usize, usize)| {
            let pose = image_poses[image].as_ref()?;
            let sensor = &rig.sensors()[image_assignment[image].1];
            let point_camera = pose.transform_world_point(&point);
            Some(
                point_camera.z > 0.0
                    && sensor.camera.project(&point_camera).is_some_and(|pixel| {
                        (pixel - features[image].keypoints[keypoint]).norm()
                            <= config.max_reprojection_error_px
                    }),
            )
        };
        if config.robust_triangulation_pruning {
            let registered = tracks[track_index]
                .observations
                .iter()
                .filter(|observation| observation_is_inlier(observation).is_some())
                .count();
            let inliers = tracks[track_index]
                .observations
                .iter()
                .filter(|observation| observation_is_inlier(observation) == Some(true))
                .count();
            let required = ((registered as f64 * config.triangulation_min_inlier_fraction).ceil()
                as usize)
                .max(2);
            if inliers < required {
                update.majority_rejections += 1;
                continue;
            }
            let observations_before = tracks[track_index].observations.len();
            tracks[track_index]
                .observations
                .retain(|observation| observation_is_inlier(observation).unwrap_or(true));
            update.pruned_observations +=
                observations_before - tracks[track_index].observations.len();
            tracks[track_index].metric_anchored =
                track_is_metric_anchored(&tracks[track_index].observations, image_assignment);
            tracks[track_index].position = Some(point);
            update.robust_tracks += 1;
            update.landmarks.push(track_index);
        } else if tracks[track_index]
            .observations
            .iter()
            .all(|(image, keypoint)| {
                let Some(pose) = image_poses[*image].as_ref() else {
                    return true;
                };
                let sensor = &rig.sensors()[image_assignment[*image].1];
                let point_camera = pose.transform_world_point(&point);
                point_camera.z > 0.0
                    && sensor.camera.project(&point_camera).is_some_and(|pixel| {
                        (pixel - features[*image].keypoints[*keypoint]).norm()
                            <= config.max_reprojection_error_px
                    })
            })
        {
            tracks[track_index].position = Some(point);
            update.landmarks.push(track_index);
        }
    }
    update
}

fn bounded_evenly_spaced_indices(len: usize, max_samples: usize) -> Vec<usize> {
    if len == 0 || max_samples == 0 {
        return Vec::new();
    }
    if len <= max_samples {
        return (0..len).collect();
    }
    if max_samples == 1 {
        return vec![0];
    }
    (0..max_samples)
        .map(|sample| sample * (len - 1) / (max_samples - 1))
        .collect()
}

fn closest_ray_midpoint(
    origin_left: &Point3<f64>,
    direction_left: &Vector3<f64>,
    origin_right: &Point3<f64>,
    direction_right: &Vector3<f64>,
) -> Option<Point3<f64>> {
    let offset = origin_left - origin_right;
    let cosine = direction_left.dot(direction_right);
    let denominator = 1.0 - cosine * cosine;
    if denominator <= 1.0e-12 {
        return None;
    }
    let left_projection = direction_left.dot(&offset);
    let right_projection = direction_right.dot(&offset);
    let left_depth = (cosine * right_projection - left_projection) / denominator;
    let right_depth = (right_projection - cosine * left_projection) / denominator;
    if left_depth <= 0.0 || right_depth <= 0.0 {
        return None;
    }
    let point_left = origin_left + left_depth * direction_left;
    let point_right = origin_right + right_depth * direction_right;
    Some(Point3::from((point_left.coords + point_right.coords) * 0.5))
}

fn reprojection_error(
    rig: &GeneralizedCameraRig,
    image_assignment: &[(usize, usize)],
    image_poses: &[Option<Pose>],
    tracks: &[SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0;
    for track in tracks {
        for (image, _, pixel) in &track.observations {
            let Some(pose) = image_poses[*image].as_ref() else {
                continue;
            };
            let sensor = &rig.sensors()[image_assignment[*image].1];
            if let Some(projected) = sensor
                .camera
                .project(&pose.transform_world_point(&track.position))
            {
                sum += (projected - pixel).norm();
                count += 1;
            }
        }
    }
    (sum, count)
}

#[cfg(test)]
mod tests {
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_core::types::Camera;
    use visloc_vision::pnp::RigSensor;

    use super::*;

    #[test]
    fn metric_seed_support_counts_each_multisensor_track_once_per_frame() {
        let assignment = vec![(2, 1), (0, 0), (2, 0), (1, 0), (0, 1), (1, 0), (2, 2)];
        let track = |observations| WorkingTrack {
            observations,
            position: None,
            metric_anchored: false,
        };
        let tracks = vec![
            track(vec![(0, 0), (1, 0), (2, 0), (3, 0)]),
            track(vec![(4, 0), (1, 1)]),
            track(vec![(3, 1), (5, 0)]),
            track(vec![(6, 0), (2, 1), (0, 1)]),
        ];

        assert_eq!(metric_frame_supports(3, &tracks, &assignment), [1, 0, 2]);
        assert_eq!(metric_seed_candidates(&[6, 8, 8, 5]), [1, 2, 0]);
    }

    #[test]
    fn temporal_track_support_bins_tracks_by_anchor_and_span_without_pair_state() {
        let assignment = vec![
            (0, 0),
            (0, 1),
            (7, 0),
            (8, 0),
            (15, 0),
            (16, 0),
            (127, 0),
            (128, 0),
        ];
        let counts = temporal_track_support(
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![0, 5],
                vec![0, 6],
                vec![0, 7],
            ],
            &assignment,
            8,
        );

        assert_eq!(
            counts.get(&(0, "same-frame")),
            Some(&TemporalTrackSupportCount {
                tracks: 1,
                observations: 2,
                metric_tracks: 1,
            })
        );
        for (frame_bin, span_class) in [
            (0, "1-7"),
            (1, "8-15"),
            (2, "16-31"),
            (15, "32-127"),
            (16, "128+"),
        ] {
            assert_eq!(
                counts.get(&(frame_bin, span_class)),
                Some(&TemporalTrackSupportCount {
                    tracks: 1,
                    observations: 2,
                    metric_tracks: 0,
                })
            );
        }
    }

    #[test]
    fn robust_triangulation_prunes_one_registered_outlier_instead_of_dropping_track() {
        let camera = Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0);
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: camera.clone(),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: camera.clone(),
                sensor_from_rig: SE3::identity(),
            },
        ])
        .unwrap();
        let features = vec![
            FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(300.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(280.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(300.0, 242.0)], vec![Vec::new()]).unwrap(),
        ];
        let assignment = vec![(0, 0), (1, 0), (2, 0), (1, 1)];
        let image_poses = [0.0, 1.0, 2.0, 1.0]
            .into_iter()
            .map(|centre_x| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-centre_x, 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>();
        let track = || WorkingTrack {
            observations: vec![(0, 0), (1, 0), (2, 0), (3, 0)],
            position: None,
            metric_anchored: true,
        };
        let config = RigSfmConfig {
            max_reprojection_error_px: 1.0,
            robust_triangulation_pruning: true,
            triangulation_min_inlier_fraction: 0.5,
            ..RigSfmConfig::default()
        };
        let mut robust_tracks = vec![track()];
        let robust = triangulate_frontier(
            &rig,
            &features,
            &assignment,
            &image_poses,
            &config,
            &mut robust_tracks,
            HashSet::from([0]),
        );
        assert_eq!(robust.landmarks, [0]);
        assert_eq!(robust.robust_tracks, 1);
        assert_eq!(robust.pruned_observations, 1);
        assert_eq!(robust_tracks[0].observations.len(), 3);

        let mut strict_tracks = vec![track()];
        let strict = triangulate_frontier(
            &rig,
            &features,
            &assignment,
            &image_poses,
            &RigSfmConfig {
                max_reprojection_error_px: 1.0,
                ..RigSfmConfig::default()
            },
            &mut strict_tracks,
            HashSet::from([0]),
        );
        assert!(strict.landmarks.is_empty());
    }

    #[test]
    fn final_filter_prunes_bad_registered_observations_but_keeps_unregistered_support() {
        let camera = Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0);
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: camera.clone(),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera,
                sensor_from_rig: SE3::identity(),
            },
        ])
        .unwrap();
        let features = vec![
            FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(300.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(280.0, 240.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(300.0, 242.0)], vec![Vec::new()]).unwrap(),
            FeatureSet::new(vec![Point2::new(0.0, 0.0)], vec![Vec::new()]).unwrap(),
        ];
        let assignment = vec![(0, 0), (1, 0), (2, 0), (1, 1), (3, 0)];
        let mut image_poses = [0.0, 1.0, 2.0, 1.0]
            .into_iter()
            .map(|centre_x| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-centre_x, 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>();
        image_poses.push(None);
        let point = Point3::new(0.0, 0.0, 5.0);
        let mut tracks = vec![
            WorkingTrack {
                observations: vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
                position: Some(point),
                metric_anchored: true,
            },
            WorkingTrack {
                observations: vec![(0, 0), (3, 0)],
                position: Some(point),
                metric_anchored: true,
            },
        ];

        let pruned = filter_positioned_track_observations(
            &rig,
            &features,
            &assignment,
            &image_poses,
            1.0,
            &mut tracks,
        );

        assert_eq!(pruned, 3);
        assert_eq!(tracks[0].observations, [(0, 0), (1, 0), (2, 0), (4, 0)]);
        assert!(!tracks[0].metric_anchored);
        assert_eq!(tracks[0].position, Some(point));
        assert_eq!(tracks[1].position, None);
    }

    #[test]
    fn bounded_ray_sampling_is_stable_and_never_exceeds_the_cap() {
        assert!(bounded_evenly_spaced_indices(8, 0).is_empty());
        assert_eq!(bounded_evenly_spaced_indices(0, 8), Vec::<usize>::new());
        assert_eq!(bounded_evenly_spaced_indices(4, 8), [0, 1, 2, 3]);
        assert_eq!(bounded_evenly_spaced_indices(100, 1), [0]);

        let sampled = bounded_evenly_spaced_indices(10_000, 16);
        assert_eq!(sampled.len(), 16);
        assert_eq!(sampled.first(), Some(&0));
        assert_eq!(sampled.last(), Some(&9_999));
        assert!(sampled.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn deferred_retriangulation_prunes_only_failed_new_tracks() {
        let track = |observation, position| WorkingTrack {
            observations: vec![observation],
            position,
            metric_anchored: false,
        };
        let point = Point3::new(0.0, 0.0, 5.0);
        let mut tracks = vec![
            track((0, 0), None),
            track((1, 0), Some(point)),
            track((2, 0), None),
            track((3, 0), Some(point)),
        ];

        let update = retain_positioned_deferred_tracks(&mut tracks, 2);

        assert_eq!(update.tracks, 1);
        assert_eq!(update.observations, 1);
        assert_eq!(tracks.len(), 3);
        assert_eq!(tracks[0].observations, [(0, 0)]);
        assert_eq!(tracks[1].observations, [(1, 0)]);
        assert_eq!(tracks[2].observations, [(3, 0)]);
    }

    #[test]
    fn metric_anchor_frame_count_requires_distinct_sensors_in_distinct_frames() {
        let assignment = vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 0)];
        assert_eq!(
            metric_anchor_frame_count(
                &[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)],
                &assignment,
            ),
            2
        );
        assert_eq!(
            metric_anchor_frame_count(&[(0, 0), (2, 0), (3, 0)], &assignment),
            1
        );
    }

    #[test]
    fn bounded_track_completion_is_transitive_without_merging_positioned_tracks() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0)),
            },
        ])
        .unwrap();
        let features = (0..4)
            .map(|_| FeatureSet::new(vec![Point2::new(320.0, 240.0)], vec![vec![0.0]]).unwrap())
            .collect::<Vec<_>>();
        let assignment = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
        let poses = vec![Some(Pose::identity()); 4];
        // The first edge is deliberately visited before image 1 acquires an
        // owner, forcing the continuation to the second bounded pass.
        let pairwise = vec![
            PairwiseMatches::new(1, 2, vec![(0, 0)]),
            PairwiseMatches::new(0, 1, vec![(0, 0)]),
            PairwiseMatches::new(0, 3, vec![(0, 0)]),
        ];
        let mut tracks = vec![
            WorkingTrack {
                observations: vec![(0, 0)],
                position: Some(Point3::new(0.0, 0.0, 5.0)),
                metric_anchored: true,
            },
            WorkingTrack {
                observations: vec![(3, 0)],
                position: Some(Point3::new(0.0, 0.0, 5.0)),
                metric_anchored: true,
            },
            WorkingTrack {
                observations: vec![(1, 0)],
                position: None,
                metric_anchored: false,
            },
        ];

        let update = complete_positioned_tracks(
            &rig,
            &features,
            &pairwise,
            &assignment,
            &poses,
            2,
            1.0,
            &mut tracks,
        );

        assert_eq!(update.passes, 2);
        assert_eq!(update.pair_visits, 6);
        assert_eq!(update.observations, 2);
        assert_eq!(tracks[0].observations, [(0, 0), (1, 0), (2, 0)]);
        assert_eq!(tracks[1].observations, [(3, 0)]);
        assert!(tracks[2].observations.is_empty());
    }

    #[test]
    fn conflict_preserving_track_builder_rejects_only_the_bad_edge() {
        let features = (0..3)
            .map(|_| {
                FeatureSet::new(
                    vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                    vec![vec![0.0], vec![1.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1), (0, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        ];
        let assignment = vec![(0, 0), (1, 0), (2, 0)];
        let legacy =
            build_rig_track_output(&features, &pairwise, &assignment, &RigSfmConfig::default());
        let preserved = build_rig_track_output(
            &features,
            &pairwise,
            &assignment,
            &RigSfmConfig {
                track_builder: RigTrackBuilder::ConflictPreserving,
                ..RigSfmConfig::default()
            },
        );

        assert_eq!(legacy.stats.retained_tracks, 0);
        assert_eq!(legacy.stats.conflicting_components, 1);
        assert_eq!(preserved.stats.retained_tracks, 2);
        assert_eq!(preserved.stats.retained_observations, 6);
        assert_eq!(preserved.stats.conflicting_components, 1);
    }

    #[test]
    fn stream_order_builder_protects_a_trusted_prefix_from_a_late_bridge() {
        let features = (0..3)
            .map(|_| {
                FeatureSet::new(
                    vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                    vec![vec![0.0], vec![1.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pairwise = vec![
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 1, vec![(0, 1), (1, 0)]),
        ];
        let assignment = vec![(0, 0), (1, 0), (2, 0)];
        let output = build_rig_track_output(
            &features,
            &pairwise,
            &assignment,
            &RigSfmConfig {
                track_builder: RigTrackBuilder::StreamOrderConflictPreserving,
                ..RigSfmConfig::default()
            },
        );

        assert_eq!(output.stats.conflicting_components, 2);
        assert_eq!(output.tracks.len(), 2);
        assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
        assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
    }

    #[test]
    fn trusted_pair_prefix_is_admitted_before_a_stronger_late_bridge() {
        let features = (0..3)
            .map(|_| {
                FeatureSet::new(
                    vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                    vec![vec![0.0], vec![1.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pairwise = vec![
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 1, vec![(0, 1), (1, 0), (0, 1)]),
        ];
        let assignment = vec![(0, 0), (1, 0), (2, 0)];
        let output = build_rig_track_output(
            &features,
            &pairwise,
            &assignment,
            &RigSfmConfig {
                track_builder: RigTrackBuilder::TrustedPrefixPairConfidence(2),
                ..RigSfmConfig::default()
            },
        );

        assert_eq!(output.tracks.len(), 2);
        assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
        assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
    }

    #[test]
    fn sparse_cycle_orders_third_view_supported_edges_before_a_bad_edge() {
        let features = (0..3)
            .map(|_| {
                FeatureSet::new(
                    vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                    vec![vec![0.0], vec![1.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1), (1, 1)]),
            PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        ];

        let output = build_tracks_sparse_cycle(&features, &pairwise, 2, None, false);

        assert_eq!(output.tracks.len(), 2);
        assert_eq!(output.stats.retained_observations, 6);
        assert_eq!(output.stats.conflicting_components, 1);
        assert!(output.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));
        assert!(output.tracks.contains(&vec![(0, 1), (1, 1), (2, 1)]));
    }

    #[test]
    fn metric_temporal_quadrilateral_requires_all_four_unique_edges() {
        let features = (0..4)
            .map(|_| {
                FeatureSet::new(
                    vec![
                        Point2::new(0.0, 0.0),
                        Point2::new(1.0, 0.0),
                        Point2::new(2.0, 0.0),
                    ],
                    vec![vec![0.0], vec![1.0], vec![2.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        // Images 0/1 are the two sensors at frame 0; 2/3 are frame 1.
        let assignment = vec![(0, 0), (0, 1), (1, 0), (1, 1)];
        let identity = vec![(0, 0), (1, 1)];
        let complete = vec![
            PairwiseMatches::new(0, 1, identity.clone()),
            PairwiseMatches::new(2, 3, identity.clone()),
            PairwiseMatches::new(0, 2, identity.clone()),
            PairwiseMatches::new(1, 3, identity.clone()),
        ];

        let output = build_metric_temporal_quadrilaterals(&features, &complete, &assignment, 2);
        assert_eq!(output.tracks.len(), 2);
        assert_eq!(output.stats.retained_observations, 8);
        assert!(output
            .tracks
            .contains(&vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
        assert!(output
            .tracks
            .contains(&vec![(0, 1), (1, 1), (2, 1), (3, 1)]));

        let wide_assignment = vec![(0, 0), (0, 1), (32, 0), (32, 1)];
        assert!(
            build_metric_temporal_quadrilaterals(&features, &complete, &wide_assignment, 2,)
                .tracks
                .is_empty()
        );
        assert_eq!(
            metric_temporal_quadrilateral_tracks_in_frame_gap(
                &features,
                &complete,
                &wide_assignment,
                32,
                128,
            )
            .len(),
            2
        );

        let missing_temporal_sensor = &complete[..3];
        assert!(build_metric_temporal_quadrilaterals(
            &features,
            missing_temporal_sensor,
            &assignment,
            2,
        )
        .tracks
        .is_empty());

        let mut ambiguous = complete;
        ambiguous[0].matches.push((0, 2));
        let ambiguous_output =
            build_metric_temporal_quadrilaterals(&features, &ambiguous, &assignment, 2);
        assert_eq!(
            ambiguous_output.tracks,
            vec![vec![(0, 1), (1, 1), (2, 1), (3, 1)]]
        );
    }

    #[test]
    fn metric_temporal_quadrilateral_assigns_each_observation_once() {
        let features = (0..6)
            .map(|_| FeatureSet::new(vec![Point2::origin()], vec![vec![0.0]]).unwrap())
            .collect::<Vec<_>>();
        let assignment = vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0)]),
            PairwiseMatches::new(3, 4, vec![(0, 0)]),
            PairwiseMatches::new(0, 3, vec![(0, 0)]),
            PairwiseMatches::new(1, 4, vec![(0, 0)]),
            PairwiseMatches::new(0, 2, vec![(0, 0)]),
            PairwiseMatches::new(3, 5, vec![(0, 0)]),
            PairwiseMatches::new(2, 5, vec![(0, 0)]),
        ];

        let output = build_metric_temporal_quadrilaterals(&features, &pairwise, &assignment, 2);
        assert_eq!(output.tracks, vec![vec![(0, 0), (1, 0), (3, 0), (4, 0)]]);
        assert_eq!(output.stats.connected_components, 2);
        assert_eq!(output.stats.conflicting_components, 1);
    }

    #[test]
    fn deferred_interpolation_requires_verified_support_and_a_bounded_gap() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let frames = (0..4)
            .map(|frame| RigFrame {
                images: vec![
                    RigFrameImage {
                        image_index: frame * 2,
                        sensor_index: 0,
                    },
                    RigFrameImage {
                        image_index: frame * 2 + 1,
                        sensor_index: 1,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let endpoints = || {
            vec![
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::zeros(),
                )),
                None,
                None,
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-3.0, 0.0, 0.0),
                )),
            ]
        };
        let verified = vec![vec![], vec![0], vec![1], vec![]];

        let mut too_short = endpoints();
        let mut image_poses = vec![None; 8];
        assert_eq!(
            interpolate_deferred_pose_gaps(
                &rig,
                &frames,
                &verified,
                1,
                &mut too_short,
                &mut image_poses,
            ),
            0
        );
        assert!(too_short[1].is_none() && too_short[2].is_none());

        let mut unsupported = endpoints();
        let mut image_poses = vec![None; 8];
        let missing_support = vec![vec![], vec![0], vec![], vec![]];
        assert_eq!(
            interpolate_deferred_pose_gaps(
                &rig,
                &frames,
                &missing_support,
                2,
                &mut unsupported,
                &mut image_poses,
            ),
            0
        );

        let mut interpolated = endpoints();
        let mut image_poses = vec![None; 8];
        assert_eq!(
            interpolate_deferred_pose_gaps(
                &rig,
                &frames,
                &verified,
                2,
                &mut interpolated,
                &mut image_poses,
            ),
            2
        );
        assert!((interpolated[1].as_ref().unwrap().camera_center_world().x - 1.0).abs() < 1.0e-12);
        assert!((interpolated[2].as_ref().unwrap().camera_center_world().x - 2.0).abs() < 1.0e-12);
        assert!(image_poses[2..6].iter().all(Option::is_some));
    }

    #[test]
    fn isolated_pose_repair_replaces_only_a_large_one_frame_detour() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let frames = (0..5)
            .map(|frame| RigFrame {
                images: vec![
                    RigFrameImage {
                        image_index: frame * 2,
                        sensor_index: 0,
                    },
                    RigFrameImage {
                        image_index: frame * 2 + 1,
                        sensor_index: 1,
                    },
                ],
            })
            .collect::<Vec<_>>();
        let mut frame_poses = [0.0, 1.0, 11.0, 3.0, 4.0]
            .into_iter()
            .map(|centre| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-centre, 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>();
        let mut image_poses = vec![None; 10];

        let update = repair_isolated_pose_outliers(
            &rig,
            &frames,
            0,
            0.25,
            8.0,
            2,
            &mut frame_poses,
            &mut image_poses,
        );

        assert_eq!(
            update,
            IsolatedPoseRepairUpdate {
                passes: 1,
                repairs: 1,
            }
        );
        assert!((frame_poses[2].as_ref().unwrap().camera_center_world().x - 2.0).abs() < 1.0e-12);
        assert!((frame_poses[1].as_ref().unwrap().camera_center_world().x - 1.0).abs() < 1.0e-12);
        assert!((frame_poses[3].as_ref().unwrap().camera_center_world().x - 3.0).abs() < 1.0e-12);
        assert!(image_poses[4].is_some() && image_poses[5].is_some());
    }

    #[test]
    fn paired_pose_jump_repair_shifts_a_contiguous_offset_and_respects_boundaries() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let frames = (0..7)
            .map(|frame| RigFrame {
                images: (0..2)
                    .map(|sensor| RigFrameImage {
                        image_index: frame * 2 + sensor,
                        sensor_index: sensor,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let centres = [0.0, 0.1, 10.2, 10.3, 0.4, 0.5, 0.6];
        let make_poses = || {
            centres
                .into_iter()
                .map(|centre| {
                    Some(Pose::from_world_to_camera(
                        UnitQuaternion::identity(),
                        Vector3::new(-centre, 0.0, 0.0),
                    ))
                })
                .collect::<Vec<_>>()
        };
        let mut frame_poses = make_poses();
        let mut image_poses = vec![None; frames.len() * 2];
        let update = repair_paired_pose_jumps(
            &rig,
            &frames,
            0,
            0.25,
            8.0,
            4,
            0.1,
            &mut frame_poses,
            &mut image_poses,
        );
        assert_eq!(
            update,
            PairedPoseJumpRepairUpdate {
                repairs: 1,
                repaired_frames: 2,
            }
        );
        for (actual, expected) in frame_poses.iter().zip([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) {
            assert!((actual.as_ref().unwrap().camera_center_world().x - expected).abs() < 1.0e-12);
        }
        assert!(image_poses[4..8].iter().all(Option::is_some));

        // A normal fast trajectory has no jump above the robust global-step
        // floor, while a missing frame prevents a candidate from crossing the
        // model boundary. Both cases leave every pose untouched.
        let normal_centres = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let mut normal = normal_centres
            .into_iter()
            .map(|centre| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(-centre, 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>();
        let normal_before = normal.clone();
        let mut normal_images = vec![None; normal.len() * 2];
        assert_eq!(
            repair_paired_pose_jumps(
                &rig,
                &frames[..normal.len()],
                0,
                0.25,
                8.0,
                4,
                0.1,
                &mut normal,
                &mut normal_images,
            ),
            PairedPoseJumpRepairUpdate::default()
        );
        assert_eq!(normal, normal_before);

        let mut missing = make_poses();
        missing[3] = None;
        let missing_before = missing.clone();
        let mut missing_images = vec![None; frames.len() * 2];
        assert_eq!(
            repair_paired_pose_jumps(
                &rig,
                &frames,
                0,
                0.25,
                8.0,
                4,
                0.1,
                &mut missing,
                &mut missing_images,
            ),
            PairedPoseJumpRepairUpdate::default()
        );
        assert_eq!(missing, missing_before);

        let mut seed_internal = make_poses();
        let seed_before = seed_internal.clone();
        let mut seed_images = vec![None; frames.len() * 2];
        assert_eq!(
            repair_paired_pose_jumps(
                &rig,
                &frames,
                2,
                0.25,
                8.0,
                4,
                0.1,
                &mut seed_internal,
                &mut seed_images,
            ),
            PairedPoseJumpRepairUpdate::default()
        );
        assert_eq!(seed_internal, seed_before);

        // Candidate selection and application are deterministic across runs.
        let mut repeat = make_poses();
        let mut repeat_images = vec![None; frames.len() * 2];
        let repeat_update = repair_paired_pose_jumps(
            &rig,
            &frames,
            0,
            0.25,
            8.0,
            4,
            0.1,
            &mut repeat,
            &mut repeat_images,
        );
        assert_eq!(repeat, frame_poses);
        assert_eq!(repeat_update, update);
    }

    #[test]
    fn reconstructs_metric_synchronized_rig_frames() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let world_points = (0..24)
            .map(|index| {
                Point3::new(
                    (index % 6) as f64 * 0.28 - 0.7,
                    (index / 6) as f64 * 0.24 - 0.35,
                    4.0 + (index % 5) as f64 * 0.17,
                )
            })
            .collect::<Vec<_>>();
        let truth = (0..8)
            .map(|frame| {
                Pose::from_world_to_camera(
                    UnitQuaternion::from_euler_angles(
                        0.002 * frame as f64,
                        -0.01 * frame as f64,
                        0.003 * frame as f64,
                    ),
                    Vector3::new(-0.08 * frame as f64, 0.01 * frame as f64, 0.0),
                )
            })
            .collect::<Vec<_>>();
        let mut frames = Vec::new();
        let mut features = Vec::new();
        for (frame_index, frame_pose) in truth.iter().enumerate() {
            let mut frame_images = Vec::new();
            for sensor_index in 0..2 {
                let image_index = features.len();
                let image_pose = rig.sensors()[sensor_index]
                    .sensor_from_rig
                    .compose(&frame_pose.world_to_camera);
                let keypoints = world_points
                    .iter()
                    .map(|point| {
                        rig.sensors()[sensor_index]
                            .camera
                            .project(&image_pose.transform_point(point))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                features
                    .push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
                frame_images.push(RigFrameImage {
                    image_index,
                    sensor_index,
                });
            }
            assert_eq!(frame_images[0].image_index, frame_index * 2);
            frames.push(RigFrame {
                images: frame_images,
            });
        }
        let identity_matches = (0..world_points.len())
            .map(|index| (index, index))
            .collect::<Vec<_>>();
        let mut pairwise = Vec::new();
        for frame in 0..8 {
            pairwise.push(PairwiseMatches::new(
                frame * 2,
                frame * 2 + 1,
                identity_matches.clone(),
            ));
            if frame + 1 < 8 {
                pairwise.push(PairwiseMatches::new(
                    frame * 2,
                    (frame + 1) * 2,
                    identity_matches.clone(),
                ));
                pairwise.push(PairwiseMatches::new(
                    frame * 2 + 1,
                    (frame + 1) * 2 + 1,
                    identity_matches.clone(),
                ));
            }
        }

        let result = incremental_rig_sfm(
            &rig,
            &frames,
            &features,
            &pairwise,
            &RigSfmConfig::default(),
        )
        .unwrap();
        let recovery_pairs = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0)]),
            PairwiseMatches::new(1, 2, vec![(0, 0)]),
            PairwiseMatches::new(0, 2, vec![(0, 0), (1, 0)]),
        ];
        let recovered = recover_metric_conflict_tracks(
            &rig,
            &features,
            &recovery_pairs,
            &[vec![(0, 0), (0, 1), (1, 0), (2, 0)]],
            &image_assignment(&frames, features.len()),
            &result.image_poses,
            &RigSfmConfig {
                recover_metric_conflict_tracks: true,
                conflict_recovery_max_reprojection_error_px: 0.1,
                conflict_recovery_max_mean_reprojection_px: 0.05,
                ..RigSfmConfig::default()
            },
        );
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].observations, vec![(0, 0), (1, 0), (2, 0)]);
        assert!(recovered[0].metric_anchored);
        assert!((recovered[0].position.unwrap() - world_points[0]).norm() < 1.0e-4);
        assert_eq!(result.registered_frames, 8);
        assert_eq!(result.registered_images, 16);
        assert_eq!(result.tracks.len(), world_points.len());
        assert!(result.mean_reprojection_error_px < 1.0e-4);
        let total_observations = result
            .tracks
            .iter()
            .map(|track| track.observations.len())
            .sum::<usize>();
        assert_eq!(
            result.work.correspondence_cache_insertions,
            total_observations
        );
        assert!(result.work.triangulation_attempts <= total_observations);
        assert!(result.work.pnp_attempts < frames.len());
        assert_eq!(
            result.work.pnp_attempts,
            result.work.pnp_insufficient_sensor_attempts
                + result.work.pnp_estimation_failures
                + result.work.pnp_inlier_rejections
                + result.work.pnp_registrations
        );
        assert_eq!(
            frames.len() - result.registered_frames,
            result.work.unregistered_zero_support_frames
                + result.work.unregistered_below_pnp_support_frames
                + result.work.unregistered_eligible_pnp_frames
        );
        for (estimated, expected) in result.frame_poses.iter().zip(&truth) {
            let estimated = estimated.as_ref().unwrap();
            assert!(
                (estimated.camera_center_world() - expected.camera_center_world()).norm() < 1.0e-4
            );
            assert!(
                estimated
                    .world_to_camera
                    .rotation
                    .angle_to(&expected.world_to_camera.rotation)
                    < 1.0e-4
            );
        }
        let baseline = (result.image_poses[0]
            .as_ref()
            .unwrap()
            .camera_center_world()
            - result.image_poses[1]
                .as_ref()
                .unwrap()
                .camera_center_world())
        .norm();
        assert!((baseline - 0.20).abs() < 1.0e-9);

        // Keep the last frame's stereo pair in the base graph, but defer its
        // two temporal links. The base mapper must remain byte-for-byte
        // equivalent for already registered poses; the suffix may only fill
        // the isolated final frame from established landmarks.
        let mut base_pairs = pairwise
            .iter()
            .filter(|pair| !matches!((pair.image_i, pair.image_j), (12, 14) | (13, 15)))
            .cloned()
            .collect::<Vec<_>>();
        let deferred_prefix = base_pairs.len();
        base_pairs.extend(
            pairwise
                .iter()
                .filter(|pair| matches!((pair.image_i, pair.image_j), (12, 14) | (13, 15)))
                .cloned(),
        );
        let no_final_ba = RigSfmConfig {
            final_bundle_adjustment: false,
            structure_refinement_iterations: 0,
            ..RigSfmConfig::default()
        };
        let base = incremental_rig_sfm(
            &rig,
            &frames,
            &features,
            &base_pairs[..deferred_prefix],
            &no_final_ba,
        )
        .unwrap();
        assert_eq!(base.registered_frames, 7);
        let completed = incremental_rig_sfm(
            &rig,
            &frames,
            &features,
            &base_pairs,
            &RigSfmConfig {
                deferred_registration_pair_prefix: Some(deferred_prefix),
                ..no_final_ba
            },
        )
        .unwrap();
        assert_eq!(completed.registered_frames, 8);
        assert_eq!(completed.work.deferred_pair_visits, 2);
        assert_eq!(completed.work.deferred_pnp_attempts, 1);
        assert_eq!(completed.work.deferred_registrations, 1);
        assert_eq!(
            completed.work.deferred_observations_attached,
            world_points.len() * 2
        );
        for frame in 0..7 {
            assert_eq!(completed.frame_poses[frame], base.frame_poses[frame]);
        }

        // A suffix source feature need not belong to a retained base track:
        // synchronized source stereo can supply a short-lived 3D point for
        // PnP without adding that point to the accepted reconstruction.
        let direct_suffix = vec![
            PairwiseMatches::new(12, 14, identity_matches.clone()),
            PairwiseMatches::new(13, 15, identity_matches.clone()),
        ];
        let source_stereo = vec![PairwiseMatches::new(12, 13, identity_matches.clone())];
        let assignment = image_assignment(&frames, features.len());
        let direct_links =
            build_relevant_verified_stereo_links(&source_stereo, &direct_suffix, &assignment);
        let mut direct_frame_poses = truth
            .iter()
            .enumerate()
            .map(|(frame, pose)| (frame < 7).then_some(pose.clone()))
            .collect::<Vec<_>>();
        let mut direct_image_poses = vec![None; features.len()];
        for frame in 0..7 {
            install_image_poses(&rig, &frames[frame], &truth[frame], &mut direct_image_poses);
        }
        let mut no_tracks = Vec::new();
        let mut no_image_tracks = vec![Vec::new(); features.len()];
        let direct = register_deferred_frames(
            &rig,
            &frames,
            &features,
            &direct_suffix,
            &assignment,
            &no_final_ba,
            &mut direct_frame_poses,
            &mut direct_image_poses,
            &mut no_tracks,
            &mut no_image_tracks,
            &direct_links,
        );
        assert_eq!(direct.registrations, 1);
        assert_eq!(direct.observations_attached, 0);
        assert!(
            (direct_frame_poses[7]
                .as_ref()
                .unwrap()
                .camera_center_world()
                - truth[7].camera_center_world())
            .norm()
                < 1.0e-4
        );

        let links = build_verified_stereo_links(&pairwise, &assignment);
        let adjacency = build_frame_pair_adjacency(&pairwise, &assignment, frames.len());
        let mut image_poses = vec![None; features.len()];
        install_image_poses(&rig, &frames[0], &Pose::identity(), &mut image_poses);
        let mut source_queue = vec![0];
        let mut pair_visited = vec![false; pairwise.len()];
        let mut target_observations = vec![HashSet::new(); frames.len()];
        let mut cached = vec![Vec::new(); frames.len()];
        let mut versions = vec![0; frames.len()];
        let mut heap = BinaryHeap::new();
        let direct_config = RigSfmConfig {
            direct_stereo_pnp_max_frame_gap: 1,
            ..RigSfmConfig::default()
        };
        let direct = append_direct_stereo_pnp_correspondences(
            &rig,
            &frames,
            &features,
            &pairwise,
            &assignment,
            &image_poses,
            &direct_config,
            &links,
            &adjacency,
            &mut source_queue,
            &mut pair_visited,
            &mut target_observations,
            &mut cached,
            &mut versions,
            &mut heap,
        );
        assert_eq!(direct.pair_visits, 2);
        assert_eq!(direct.insertions, world_points.len() * 2);
        let correspondences = cached[1]
            .iter()
            .map(|cached| GeneralizedCorrespondence2D3D {
                sensor_index: cached.sensor_index,
                point2d: cached.point2d,
                point3d: cached.direct_point3d.unwrap(),
                confidence: None,
            })
            .collect::<Vec<_>>();
        let bridged = GeneralizedPnPRansac {
            iterations: direct_config.pnp_max_iterations,
            reprojection_threshold: direct_config.max_reprojection_error_px,
            seed: direct_config.ransac_seed,
            ..GeneralizedPnPRansac::default()
        }
        .estimate(&rig, &correspondences)
        .unwrap();
        assert!(bridged.inliers.len() >= world_points.len());
        assert!(
            (bridged.pose.camera_center_world() - truth[1].camera_center_world()).norm() < 1.0e-4
        );

        let mut motion_frame_poses = vec![None; frames.len()];
        motion_frame_poses[0] = Some(truth[0].clone());
        motion_frame_poses[1] = Some(truth[1].clone());
        let motion_config = RigSfmConfig {
            motion_bridge_max_frame_gap: 1,
            motion_bridge_min_inliers: 12,
            motion_bridge_max_rotation_deviation_deg: 5.0,
            ..RigSfmConfig::default()
        };
        let mut motion_queue = vec![1];
        let mut motion_visited = vec![false; pairwise.len()];
        let motion = collect_motion_bridge_candidates(
            &rig,
            &frames,
            &features,
            &pairwise,
            &assignment,
            &motion_frame_poses,
            &motion_config,
            &adjacency,
            &mut motion_queue,
            &mut motion_visited,
        );
        let predicted = motion
            .candidates
            .iter()
            .find(|candidate| candidate.frame == 2)
            .unwrap();
        assert!(predicted.inliers >= world_points.len());
        assert!(
            (predicted.pose.camera_center_world() - truth[2].camera_center_world()).norm() < 0.01
        );
        assert!(
            predicted
                .pose
                .world_to_camera
                .rotation
                .angle_to(&truth[2].world_to_camera.rotation)
                .to_degrees()
                < 0.1
        );
    }

    #[test]
    fn fixed_active_rotations_refine_rig_translation_and_landmarks() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let world_points = (0..24)
            .map(|index| {
                Point3::new(
                    (index % 6) as f64 * 0.28 - 0.7,
                    (index / 6) as f64 * 0.24 - 0.35,
                    4.0 + (index % 5) as f64 * 0.17,
                )
            })
            .collect::<Vec<_>>();
        let truth = vec![
            Pose::identity(),
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
                Vector3::new(-0.45, 0.03, 0.02),
            ),
        ];
        let frames = (0..2)
            .map(|frame| RigFrame {
                images: (0..2)
                    .map(|sensor| RigFrameImage {
                        image_index: frame * 2 + sensor,
                        sensor_index: sensor,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let mut features = Vec::new();
        for (frame, frame_pose) in truth.iter().enumerate() {
            for sensor in 0..2 {
                let image_pose = rig.sensors()[sensor]
                    .sensor_from_rig
                    .compose(&frame_pose.world_to_camera);
                let keypoints = world_points
                    .iter()
                    .map(|point| {
                        rig.sensors()[sensor]
                            .camera
                            .project(&image_pose.transform_point(point))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                features
                    .push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
                assert_eq!(features.len(), frame * 2 + sensor + 1);
            }
        }
        let image_assignment = image_assignment(&frames, features.len());
        let mut frame_poses = vec![
            Some(truth[0].clone()),
            Some(Pose::from_world_to_camera(
                truth[1].world_to_camera.rotation,
                truth[1].world_to_camera.translation + Vector3::new(0.04, -0.025, 0.02),
            )),
        ];
        let rotations_before = frame_poses
            .iter()
            .map(|pose| pose.as_ref().unwrap().world_to_camera.rotation)
            .collect::<Vec<_>>();
        let mut image_poses = vec![None; features.len()];
        for (frame, pose) in frame_poses.iter().enumerate() {
            install_image_poses(
                &rig,
                &frames[frame],
                pose.as_ref().unwrap(),
                &mut image_poses,
            );
        }
        let mut tracks = world_points
            .iter()
            .enumerate()
            .map(|(track, point)| WorkingTrack {
                observations: (0..features.len()).map(|image| (image, track)).collect(),
                position: Some(*point + Vector3::new(0.015, -0.01, 0.025)),
                metric_anchored: true,
            })
            .collect::<Vec<_>>();
        let public_tracks = |tracks: &[WorkingTrack]| {
            tracks
                .iter()
                .map(|track| SfmTrack {
                    position: track.position.unwrap(),
                    observations: track
                        .observations
                        .iter()
                        .map(|&(image, keypoint)| {
                            (image, keypoint, features[image].keypoints[keypoint])
                        })
                        .collect(),
                })
                .collect::<Vec<_>>()
        };
        let before_tracks = public_tracks(&tracks);
        let (before_sum, before_count) =
            reprojection_error(&rig, &image_assignment, &image_poses, &before_tracks);
        let before_reprojection = before_sum / before_count as f64;
        let before_center_error = (frame_poses[1].as_ref().unwrap().camera_center_world()
            - truth[1].camera_center_world())
        .norm();
        let before_landmark_error = tracks
            .iter()
            .zip(&world_points)
            .map(|(track, point)| (track.position.unwrap() - point).norm())
            .sum::<f64>()
            / world_points.len() as f64;
        let active_frames = HashSet::from([0, 1]);
        let config = RigSfmConfig {
            final_bundle_adjustment: false,
            structure_refinement_iterations: 0,
            ..RigSfmConfig::default()
        };
        let ba_config = BaConfig {
            max_iterations: 40,
            linear_solver: LinearSolver::Sparse,
            robust_kernel: RobustKernel::None,
            parallel: false,
            ..BaConfig::default()
        };
        let stats = run_rig_bundle_adjustment(
            &rig,
            &features,
            &image_assignment,
            &config,
            &active_frames,
            0,
            &ba_config,
            0,
            &[],
            true,
            &mut frame_poses,
            &mut image_poses,
            &mut tracks,
        )
        .unwrap();
        assert!(stats.is_some());
        for (before, after) in rotations_before.iter().zip(&frame_poses) {
            assert!(before.angle_to(&after.as_ref().unwrap().world_to_camera.rotation) < 1.0e-10);
        }
        let after_tracks = public_tracks(&tracks);
        let (after_sum, after_count) =
            reprojection_error(&rig, &image_assignment, &image_poses, &after_tracks);
        let after_reprojection = after_sum / after_count as f64;
        let after_center_error = (frame_poses[1].as_ref().unwrap().camera_center_world()
            - truth[1].camera_center_world())
        .norm();
        let after_landmark_error = tracks
            .iter()
            .zip(&world_points)
            .map(|(track, point)| (track.position.unwrap() - point).norm())
            .sum::<f64>()
            / world_points.len() as f64;
        assert!(after_reprojection < before_reprojection);
        assert!(after_center_error < before_center_error);
        assert!(after_landmark_error < before_landmark_error);
    }

    #[test]
    fn public_fixed_frame_rotation_refinement_updates_result_transactionally() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();
        let world_points = (0..24)
            .map(|index| {
                Point3::new(
                    (index % 6) as f64 * 0.28 - 0.7,
                    (index / 6) as f64 * 0.24 - 0.35,
                    4.0 + (index % 5) as f64 * 0.17,
                )
            })
            .collect::<Vec<_>>();
        let truth = [
            Pose::identity(),
            Pose::from_world_to_camera(
                UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
                Vector3::new(-0.45, 0.03, 0.02),
            ),
        ];
        let frames = (0..2)
            .map(|frame| RigFrame {
                images: (0..2)
                    .map(|sensor| RigFrameImage {
                        image_index: frame * 2 + sensor,
                        sensor_index: sensor,
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let mut features = Vec::new();
        for frame_pose in &truth {
            for sensor in 0..2 {
                let image_pose = rig.sensors()[sensor]
                    .sensor_from_rig
                    .compose(&frame_pose.world_to_camera);
                let keypoints = world_points
                    .iter()
                    .map(|point| {
                        rig.sensors()[sensor]
                            .camera
                            .project(&image_pose.transform_point(point))
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                features
                    .push(FeatureSet::new(keypoints, vec![vec![0.0]; world_points.len()]).unwrap());
            }
        }
        let image_assignment = image_assignment(&frames, features.len());
        let mut frame_poses = vec![Some(truth[0].clone()), Some(truth[1].clone())];
        frame_poses[1] = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.02, -0.03, 0.04),
            truth[1].world_to_camera.translation + Vector3::new(0.04, -0.025, 0.02),
        ));
        let mut image_poses = vec![None; features.len()];
        for (frame, pose) in frame_poses.iter().enumerate() {
            install_image_poses(
                &rig,
                &frames[frame],
                pose.as_ref().unwrap(),
                &mut image_poses,
            );
        }
        let tracks = world_points
            .iter()
            .enumerate()
            .map(|(track, point)| SfmTrack {
                position: *point + Vector3::new(0.015, -0.01, 0.025),
                observations: (0..features.len())
                    .map(|image| (image, track, features[image].keypoints[track]))
                    .collect(),
            })
            .collect::<Vec<_>>();
        let (before_sum, before_count) =
            reprojection_error(&rig, &image_assignment, &image_poses, &tracks);
        let before_mean = before_sum / before_count as f64;
        let before_center = frame_poses[1].as_ref().unwrap().camera_center_world();
        let before_center_error = (before_center - truth[1].camera_center_world()).norm();
        let mut result = RigSfmResult {
            frame_poses,
            image_poses,
            tracks,
            registered_frames: 2,
            registered_images: 4,
            mean_reprojection_error_px: before_mean,
            seed_frame_index: 0,
            track_build_stats: TrackBuildStats::default(),
            work: RigSfmWorkStats::default(),
            bundle_adjustment: None,
        };
        let config = RigSfmConfig {
            final_bundle_adjustment: false,
            structure_refinement_iterations: 5,
            deferred_registration_pair_prefix: Some(1),
            ba_config: BaConfig {
                max_iterations: 40,
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::None,
                parallel: false,
                ..BaConfig::default()
            },
            ..RigSfmConfig::default()
        };
        let target_rotations = vec![None, Some(truth[1].world_to_camera.rotation)];
        refine_rig_sfm_with_fixed_frame_rotations(
            &rig,
            &frames,
            &features,
            &target_rotations,
            &config,
            &mut result,
        )
        .unwrap();
        assert!(
            result.frame_poses[1]
                .as_ref()
                .unwrap()
                .world_to_camera
                .rotation
                .angle_to(&truth[1].world_to_camera.rotation)
                < 1.0e-10
        );
        let after_center_error = (result.frame_poses[1]
            .as_ref()
            .unwrap()
            .camera_center_world()
            - truth[1].camera_center_world())
        .norm();
        assert!(after_center_error < before_center_error);
        assert!(result.mean_reprojection_error_px.is_finite());
        assert!(result.bundle_adjustment.is_some());
        assert!(!result.tracks.is_empty());
    }

    #[test]
    fn rejects_pnp_sensor_count_outside_the_calibrated_rig() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 640, 480, 300.0, 300.0, 320.0, 240.0),
                sensor_from_rig: SE3::new(
                    UnitQuaternion::identity(),
                    Vector3::new(-0.20, 0.0, 0.0),
                ),
            },
        ])
        .unwrap();

        for requested in [0, 3] {
            let error = validate_inputs(
                &rig,
                &[],
                &[],
                &[],
                &RigSfmConfig {
                    min_pnp_sensors: requested,
                    ..RigSfmConfig::default()
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                RigSfmError::InvalidMinPnpSensors {
                    requested,
                    sensor_count: 2,
                }
            );
        }
        for requested in [0, 3] {
            let error = validate_inputs(
                &rig,
                &[],
                &[],
                &[],
                &RigSfmConfig {
                    direct_stereo_min_pnp_sensors: Some(requested),
                    ..RigSfmConfig::default()
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                RigSfmError::InvalidDirectStereoMinPnpSensors {
                    requested,
                    sensor_count: 2,
                }
            );
        }
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                direct_stereo_min_triangulation_angle_deg: 0.0,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            RigSfmError::InvalidDirectStereoTriangulationAngle(0.0)
        );

        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                complete_tracks_after_registration: true,
                track_completion_max_passes: 0,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, RigSfmError::InvalidTrackCompletionPasses);
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                complete_tracks_after_registration: true,
                track_completion_max_reprojection_error_px: f64::NAN,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RigSfmError::InvalidTrackCompletionReprojectionError(value) if value.is_nan()
        ));

        for requested in [0.0, f64::NAN] {
            let error = validate_inputs(
                &rig,
                &[],
                &[],
                &[],
                &RigSfmConfig {
                    repair_paired_pose_jumps: true,
                    paired_pose_jump_absolute_step_m: requested,
                    ..RigSfmConfig::default()
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                RigSfmError::InvalidPairedPoseJumpStep(value)
                    if value == requested || (value.is_nan() && requested.is_nan())
            ));
        }
        for requested in [1.0, f64::NAN] {
            let error = validate_inputs(
                &rig,
                &[],
                &[],
                &[],
                &RigSfmConfig {
                    repair_paired_pose_jumps: true,
                    paired_pose_jump_min_step_ratio: requested,
                    ..RigSfmConfig::default()
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                RigSfmError::InvalidPairedPoseJumpStepRatio(value)
                    if value == requested || (value.is_nan() && requested.is_nan())
            ));
        }
        let error = validate_inputs(
            &rig,
            &[],
            &[],
            &[],
            &RigSfmConfig {
                repair_paired_pose_jumps: true,
                paired_pose_jump_max_frame_span: 0,
                ..RigSfmConfig::default()
            },
        )
        .unwrap_err();
        assert_eq!(error, RigSfmError::InvalidPairedPoseJumpFrameSpan);
        for requested in [0.0, 1.0, f64::NAN] {
            let error = validate_inputs(
                &rig,
                &[],
                &[],
                &[],
                &RigSfmConfig {
                    repair_paired_pose_jumps: true,
                    paired_pose_jump_max_closure_ratio: requested,
                    ..RigSfmConfig::default()
                },
            )
            .unwrap_err();
            assert!(matches!(
                error,
                RigSfmError::InvalidPairedPoseJumpClosureRatio(value)
                    if value == requested || (value.is_nan() && requested.is_nan())
            ));
        }
    }
}
