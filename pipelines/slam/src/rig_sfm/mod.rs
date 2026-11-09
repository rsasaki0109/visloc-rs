//! Incremental structure-from-motion for a calibrated, synchronized camera rig.
//!
//! Each input frame owns one or more images with fixed `sensor <- rig`
//! extrinsics.  A frame is registered from all of its 2D-3D observations at
//! once with generalized PnP; image poses are derived from that single body
//! pose.  This preserves the physical stereo baseline during initialization
//! and avoids treating synchronized sensors as unrelated monocular cameras.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::time::Instant;

/// Coarse stage/progress timing for `incremental_rig_sfm`, opt-in via
/// `VISLOC_RIG_SFM_TIMING=1`. Prints wall-clock elapsed time and problem-size
/// counters at track construction, every local-BA checkpoint (registration
/// count, cumulative triangulation/PnP attempts), conflict recovery, pose
/// repair passes, and final BA -- enough to localize which stage a stalled
/// run is spending time in without the very verbose per-attempt
/// `VISLOC_SFM_DEBUG` stream.
fn rig_timing_enabled() -> bool {
    std::env::var_os("VISLOC_RIG_SFM_TIMING").is_some()
}

use nalgebra::{Matrix3, Point2, Point3, SMatrix, UnitQuaternion, Vector2, Vector3};
use thiserror::Error;
use visloc_core::geometry::{Pose, SE3};
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{
    GeneralizedCameraRig, GeneralizedCorrespondence2D3D, GeneralizedPnPRansac,
};
use visloc_vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};

use crate::bundle::{
    BaConfig, BaRigObservation, BundleAdjustment, MatrixFreeBaOptions, MatrixFreeBaRestartOptions,
};
use crate::incremental_sfm::{
    build_tracks_confidence_ordered, build_tracks_confidence_ordered_with_trusted_prefix,
    build_tracks_detailed, build_tracks_incremental_correspondence,
    build_tracks_incremental_correspondence_in_order, PairwiseMatches, SfmTrack, TrackBuildOutput,
    TrackBuildStats,
};
use crate::rig_correspondence::{
    build_rig_correspondence_csr_from_features, RigCorrespondenceCsr, RigObservationId,
};
use crate::{LinearSolver, RobustKernel};

mod bundle_adjustment;
mod deferred;
mod diagnostics;
mod driver;
mod dynamic;
mod pose_repair;
mod refinement;
mod setup;
mod tracks;
mod triangulation;

use bundle_adjustment::*;
use deferred::*;
use diagnostics::*;
pub use driver::*;
use dynamic::*;
use pose_repair::*;
pub use refinement::*;
use setup::*;
pub use tracks::*;
use triangulation::*;

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

/// Bundle-adjustment backend used by the native calibrated-rig mapper.
///
/// `Legacy` is deliberately the default so existing callers retain their
/// configured [`BaConfig`] (including its linear-solver choice).  The
/// matrix-free entry point is an explicit, strict opt-in and never falls back
/// to the legacy solve when it rejects an input or fails numerically.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RigBaBackend {
    #[default]
    Legacy,
    MatrixFreeStrict,
    /// Experimental fixed-size (at most eight poses) Schur cluster preconditioner.
    MatrixFreeCluster8,
    /// Cluster8 with one true-residual restart inside the same PCG budget.
    MatrixFreeCluster8Restart1,
    /// Experimental compact landmark QR with bounded pose-block Jacobi PCG.
    MatrixFreeQr,
    /// Choose the direct backend before solving at <=64 variable poses;
    /// larger systems use compact QR, never a failure-triggered fallback.
    BoundedDirect64Qr,
}

const fn select_bounded_rig_backend(
    requested: RigBaBackend,
    variable_poses: usize,
) -> RigBaBackend {
    match requested {
        RigBaBackend::BoundedDirect64Qr if variable_poses <= 64 => RigBaBackend::Legacy,
        RigBaBackend::BoundedDirect64Qr => RigBaBackend::MatrixFreeQr,
        other => other,
    }
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
    /// Restrict metric initialization to one caller-selected rig frame.
    /// `None` preserves support-descending automatic selection. This is useful
    /// for graph A/B tests where added edges must not silently change the
    /// initialization frame.
    pub seed_frame: Option<usize>,
    /// Cap how many support-descending seed candidates the initial-frame
    /// search tries before giving up with `InsufficientSeedStructure`. `0`
    /// means unbounded (search every candidate, matching the historical
    /// behavior). On a heavily conflict-fragmented graph (many aliased,
    /// near-duplicate frames that all yield a plausible-looking but
    /// under-landmarked two-view seed) the candidate list can run to
    /// thousands of entries; each failed attempt is now cheap (only the
    /// touched tracks/images are reset, not the whole track/pose arrays), but
    /// this cap still bounds worst-case wall time and gives a faster, more
    /// legible `InsufficientSeedStructure` report. Candidates are tried in
    /// descending support order, so a nonzero cap never skips a
    /// stronger-looking candidate in favor of a weaker one.
    pub max_seed_attempts: usize,
    /// Skip triangulation entirely for any track whose raw observation count
    /// exceeds this many entries (`0` means unbounded, matching historical
    /// behavior). A genuine 3-D landmark cannot physically be seen in
    /// thousands of a rig's images; a track that large is a union-find
    /// conflict blob produced by heavy visual aliasing (e.g. a long
    /// repetitive corridor) that happened not to collide two observations
    /// from the *same* image, so it was never split into
    /// `conflicting_components`. Left unbounded, `triangulate_frontier`
    /// re-scans that track's full observation list (building per-observation
    /// bearing rays) every single time *any* of the thousands of images
    /// touching it gets registered, before ever reaching the (already
    /// bounded) widest-baseline ray-pair search -- an O(registrations x
    /// track size) blowup. The check is a cheap `O(1)` length read before
    /// that scan, so setting this is unconditionally safe; it defaults to
    /// `0` (off) purely so existing byte-identical-output tests are
    /// unaffected by a cap they never needed.
    pub max_track_observations_for_triangulation: usize,
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
    pub ba_backend: RigBaBackend,
    /// Fix the lowest frame ID in each BA observation component lacking a
    /// fixed pose. Removes its rigid gauge, not every possible degeneracy.
    pub ba_anchor_disconnected_components: bool,
    /// Retain eligible outside observations of active tracks with fixed poses.
    pub ba_fixed_boundary_observations: bool,
    pub ba_config: BaConfig,
    pub local_ba_every: usize,
    pub local_ba_window_size: usize,
    /// Select bounded neighbors sharing usable landmarks instead of recent frames.
    pub local_ba_covisibility: bool,
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
    /// Build and consume a compact rig-local correspondence CSR during
    /// registration.  When enabled, tracks are created/continued only as
    /// their incident correspondence rows become active, matching COLMAP's
    /// registration-time point-map semantics.  The historical all-pairs
    /// track builder remains byte-for-byte unchanged when this is `false`.
    pub dynamic_correspondence_tracking: bool,
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
            seed_frame: None,
            max_seed_attempts: 0,
            max_track_observations_for_triangulation: 0,
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
            ba_backend: RigBaBackend::Legacy,
            ba_anchor_disconnected_components: false,
            ba_fixed_boundary_observations: false,
            ba_config: BaConfig {
                linear_solver: LinearSolver::Sparse,
                robust_kernel: RobustKernel::Huber { delta: 6.0 },
                parallel: true,
                ..BaConfig::default()
            },
            local_ba_every: 10,
            local_ba_window_size: 40,
            local_ba_covisibility: false,
            local_ba_iterations: 8,
            final_ba_passes: 2,
            final_ba_window_size: 60,
            final_ba_fix_window_ends: true,
            final_filter_refinement_passes: 0,
            structure_refinement_iterations: 5,
            dynamic_correspondence_tracking: false,
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
    /// Number of flattened CSR observation rows activated after a frame
    /// registration in dynamic correspondence mode.
    pub dynamic_activated_rows: usize,
    /// Number of directed CSR neighbour entries visited by those activated
    /// rows.
    pub dynamic_activated_edges: usize,
    /// Number of dynamic two-observation track creations.
    pub dynamic_track_creates: usize,
    /// Number of dynamic track continuations into a previously unowned
    /// observation.
    pub dynamic_track_continues: usize,
    /// Edges rejected because both endpoints already belong to different
    /// owners and the conservative fragment-merge gate did not admit them.
    pub dynamic_owner_conflicts: usize,
    /// Edges rejected because continuation would put a second observation
    /// from one image into a track.
    pub dynamic_same_image_conflicts: usize,
    /// Positioned-owner continuations rejected by the current pose/point
    /// reprojection gate.  PnP cache outliers never become owners.
    pub dynamic_geometry_rejections: usize,
    /// Number of compact entries in the flattened observation inverse lookup.
    /// This is a bounded O(observations) allocation, not a pairwise cache.
    pub dynamic_observation_lookup_entries: usize,
    /// Unique observation entries inserted into the dynamic PnP graph cache.
    pub dynamic_pnp_graph_insertions: usize,
    /// Number of retained legacy-union tracks inspected for dynamic
    /// bootstrap.  This is a diagnostic only; the tracks themselves are not
    /// imported into the registration-time owner map.
    pub dynamic_bootstrap_legacy_tracks: usize,
    /// Number of legacy-membership seed frames admitted by the historical
    /// metric support gate.
    pub dynamic_bootstrap_candidates: usize,
    /// Metric support (one canonical stereo pair per retained legacy track)
    /// of the selected bootstrap frame.
    pub dynamic_bootstrap_seed_support: usize,
    /// Number of selected bootstrap stereo pairs installed as owners.
    pub dynamic_bootstrap_seed_pairs: usize,
    /// Number of independently triangulated selected bootstrap landmarks.
    pub dynamic_bootstrap_seed_landmarks: usize,
    /// Number of times the direct CSR same-frame stereo fallback was used.
    pub dynamic_bootstrap_direct_fallbacks: usize,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum RigSfmError {
    #[error("rig reconstruction requires at least two calibrated sensors")]
    TooFewSensors,
    #[error("rig reconstruction contains no frames")]
    NoFrames,
    #[error("requested metric seed frame {frame} is outside 0..{frame_count}")]
    InvalidSeedFrame { frame: usize, frame_count: usize },
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
    #[error("dynamic correspondence tracking cannot build its CSR: {0}")]
    DynamicCorrespondenceBuild(String),
    #[error("dynamic correspondence tracking cannot be combined with {0}")]
    DynamicIncompatible(&'static str),
    #[error("dynamic correspondence tracking cannot represent {kind} index {index} in a u32 slot")]
    DynamicIndexOverflow { kind: &'static str, index: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct WorkingTrack {
    observations: Vec<(usize, usize)>,
    position: Option<Point3<f64>>,
    metric_anchored: bool,
}

#[cfg(test)]
mod tests;
