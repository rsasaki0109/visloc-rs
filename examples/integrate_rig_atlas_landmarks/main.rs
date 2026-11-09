//! Build a physical, landmark-bearing COLMAP model from a stitched rig atlas.
//!
//! This is deliberately an opt-in, bounded first integration step.  Source
//! windows are read one at a time; local COLMAP POINT3D_ID values are never
//! used as global identities.  Observations are keyed by the rig-manifest
//! image name (and its deterministic global image id) plus the original
//! keypoint index, then re-triangulated with the final atlas camera poses.
//!
//! The trajectory-only stitcher remains unchanged.  Pass one atlas component
//! directory (the directory containing one `images.txt`) per invocation.  A
//! root containing multiple component gauges is rejected rather than
//! flattening independent gauges into one model. Poses remain fixed by default.
//! Optional unsupported-frame recovery uses leave-target-out landmarks and
//! calibrated generalized PnP. A separate `--joint-rig-ba` opt-in runs a
//! bounded forward sweep of calibrated rig bundle adjustment; its observation
//! filter can explicitly repeat that fixed two-sweep experiment.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{DMatrix, Point2, Point3, Quaternion, UnitQuaternion, Vector3};
use visloc_rs::io::colmap::parse_cameras_txt;
use visloc_rs::slam::{BaConfig, BaRigObservation, BundleAdjustment, LinearSolver, RobustKernel};
use visloc_rs::vision::pnp::{
    GeneralizedCameraRig, GeneralizedCorrespondence2D3D, GeneralizedPnPRansac, RigSensor,
};
use visloc_rs::{Camera, CameraModel, Pose, SE3};

mod boundary_repair;
mod cli;
mod cross_boundary_pnp;
mod diagnostics;
mod input;
mod joint_ba;
mod joint_ba_filter;
mod metrics;
mod output;
mod recovery;
mod tracks;
mod util;

use boundary_repair::*;
use cli::*;
use cross_boundary_pnp::*;
use diagnostics::*;
use input::*;
use joint_ba::*;
use joint_ba_filter::*;
use metrics::*;
use output::*;
use recovery::*;
use tracks::*;
use util::*;

const USAGE: &str = "usage: integrate_rig_atlas_landmarks\n    --rig-manifest PATH --nodes-tsv PATH --atlas-dir PATH --out-dir PATH\n    [--recover-zero-support-frames] [--joint-rig-ba [--pre-ba-out-dir PATH]]\n    [--joint-rig-ba-filter-observations [--joint-rig-ba-preserve-optimized-points]]\n    [--joint-rig-ba-filter-sweeps 1|2 [--post-pass1-out-dir PATH]]\n    [--diagnose-cross-boundary-pnp --diagnostic-left-max-frame FRAME]\n    [--repair-cross-boundary --repair-left-max-frame FRAME]";
const XY_TOLERANCE_PX: f64 = 1.0e-9;
const INTRINSIC_TOLERANCE: f64 = 1.0e-8;
const MIN_TRACK_OBSERVATIONS: usize = 2;
const MAX_DLT_OBSERVATIONS: usize = 64;
const MAX_MEAN_REPROJECTION_PX: f64 = 2.0;
const MAX_REPROJECTION_PX: f64 = 4.0;
const MIN_HOMOGENEOUS_SCALE: f64 = 1.0e-12;
const MIN_BASELINE_M: f64 = 1.0e-9;
const MIN_RAY_CROSS_NORM: f64 = 1.0e-8;
const MAX_RIG_CENTER_DISAGREEMENT_M: f64 = 1.0e-4;
const MAX_RIG_ROTATION_DISAGREEMENT_DEG: f64 = 1.0e-3;
const RECOVERY_MIN_PNP_INLIERS: usize = 6;
const RECOVERY_PNP_ITERATIONS: usize = 4096;
const RECOVERY_PNP_REPROJECTION_PX: f64 = 4.0;
const RECOVERY_PNP_SEED: u64 = 7;
const MAX_RECOVERY_TRACKS_PER_FRAME: usize = 512;
const MAX_RECOVERY_CORRESPONDENCES: usize = 4096;
// The cross-boundary PnP path is diagnostic-only. These caps keep a bad
// boundary selection from turning the report into an all-pairs experiment.
const DIAGNOSTIC_MAX_CROSS_TRACKS: usize = 4096;
const DIAGNOSTIC_MAX_CROSS_OBSERVATIONS: usize = 262_144;
const DIAGNOSTIC_MAX_CANDIDATE_FRAMES: usize = 16_384;
const DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME: usize = 4096;
const JOINT_BA_WINDOW_LENGTH: usize = 60;
const JOINT_BA_WINDOW_STRIDE: usize = 30;
const JOINT_BA_MAX_LANDMARKS: usize = 16_384;
const JOINT_BA_MAX_OBSERVATIONS: usize = 262_144;
const JOINT_BA_MAX_REFERENCED_FRAMES: usize = 1_024;
const JOINT_BA_MAX_FREE_FRAMES: usize = 60;
const JOINT_BA_MAX_ITERATIONS: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Args {
    rig_manifest: PathBuf,
    nodes_tsv: PathBuf,
    atlas_dir: PathBuf,
    out_dir: PathBuf,
    pre_ba_out_dir: Option<PathBuf>,
    post_pass1_out_dir: Option<PathBuf>,
    recover_zero_support_frames: bool,
    joint_rig_ba: bool,
    joint_rig_ba_filter_observations: bool,
    joint_rig_ba_preserve_optimized_points: bool,
    joint_rig_ba_filter_sweeps: u8,
    diagnose_cross_boundary_pnp: bool,
    diagnostic_left_max_frame: Option<u64>,
    repair_cross_boundary: bool,
    repair_left_max_frame: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct NodeSpec {
    node_id: u64,
    window_start: u64,
    images_txt: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
struct SensorCalibration {
    camera_id: u64,
    width: u32,
    height: u32,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: SE3,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImageAssignment {
    frame_id: u64,
    sensor_index: usize,
    global_image_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
struct RigManifest {
    sensors: BTreeMap<usize, SensorCalibration>,
    assignments: BTreeMap<String, ImageAssignment>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceKeypoint {
    xy: Point2<f64>,
    point3d_id: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceImage {
    local_id: u64,
    camera_id: u64,
    name: String,
    keypoints: Vec<SourceKeypoint>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LocalObservation {
    image_id: u64,
    keypoint_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct SourcePoint {
    id: u64,
    position: Point3<f64>,
    observations: Vec<LocalObservation>,
}

#[derive(Debug, Clone, PartialEq)]
struct SourceWindow {
    images: Vec<SourceImage>,
    points: Vec<SourcePoint>,
}

#[derive(Debug, Clone, PartialEq)]
struct AtlasImage {
    global_image_id: u64,
    frame_id: u64,
    sensor_index: usize,
    name: String,
    camera_id: u64,
    pose: Pose,
}

#[derive(Debug, Clone, PartialEq)]
struct GlobalImage {
    atlas: AtlasImage,
    keypoints: Vec<Point2<f64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ObservationKey {
    global_image_id: u64,
    keypoint_index: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct ObservationState {
    xy: Point2<f64>,
    owner_track: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct GlobalTrack {
    observations: Vec<ObservationKey>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct IngestStats {
    source_nodes: usize,
    source_points: usize,
    source_observations: usize,
    accepted_candidates: usize,
    extended_candidates: usize,
    duplicate_observations: usize,
    rejected_candidates: usize,
    rejected_missing_atlas_images: usize,
    filtered_out_of_component_observations: usize,
    rejected_short_candidates: usize,
    rejected_xy: usize,
    rejected_track_conflict: usize,
    rejected_same_image: usize,
    rejected_invalid_source: usize,
    rejected_triangulation: usize,
    triangulation_sampled_tracks: usize,
    triangulation_sampled_observations: usize,
    output_landmarks: usize,
    output_observations: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct TrackStore {
    observations: BTreeMap<ObservationKey, ObservationState>,
    tracks: Vec<GlobalTrack>,
}

#[derive(Debug, Clone, PartialEq)]
struct LandmarkOutput {
    track_id: usize,
    position: Point3<f64>,
    observations: Vec<ObservationKey>,
    errors: Vec<f64>,
    rms_error: f64,
    mean_error: f64,
    max_error: f64,
    dlt_sample_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SupportSummary {
    atlas_image_count: usize,
    supported_image_count: usize,
    zero_support_image_count: usize,
    atlas_frame_count: usize,
    supported_frame_count: usize,
    zero_support_frame_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RecoverySummary {
    frames_considered: usize,
    frames_attempted: usize,
    frames_recovered: usize,
    candidate_tracks: usize,
    anchor_tracks: usize,
    anchor_rejections: usize,
    pnp_correspondences: usize,
    pnp_inliers: usize,
    pnp_rejections: usize,
    accepted_landmarks: usize,
    accepted_observations: usize,
    candidate_target_landmarks: usize,
    candidate_target_observations: usize,
    accepted_target_landmarks: usize,
    accepted_target_observations: usize,
    track_rejections: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct JointRigBaSummary {
    windows_considered: usize,
    windows_accepted: usize,
    windows_skipped: usize,
    selected_landmarks: usize,
    selected_observations: usize,
    max_referenced_frames: usize,
    max_free_frames: usize,
    max_iterations: usize,
    converged_windows: usize,
    max_solver_initial_cost: f64,
    min_solver_final_cost: f64,
    final_cost: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct JointRigBaFilteringSummary {
    windows_considered: usize,
    windows_accepted: usize,
    windows_skipped: usize,
    selected_landmarks: usize,
    selected_observations: usize,
    max_referenced_frames: usize,
    max_free_frames: usize,
    max_iterations: usize,
    converged_windows: usize,
    removed_observations: usize,
    removed_tracks: usize,
    retriangulated_tracks: usize,
    removed_reason_counts: BTreeMap<FilterObservationReason, usize>,
    /// These are selected-landmark events accumulated over windows, not
    /// unique global point counts.
    raw_preserved_tracks: usize,
    dlt_attempted_tracks: usize,
    raw_fallback_reason_counts: BTreeMap<RawPointFallbackReason, usize>,
    full_pre_ba_cost: f64,
    full_post_ba_cost: f64,
    retained_pre_ba_cost: f64,
    retained_post_filter_cost: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RawPointFallbackReason {
    ObservationKeysChanged,
    TooFewObservations,
    NonFinite,
    NoObservableParallax,
    ParallaxCheckFailed,
    NonPositiveDepth,
    ProjectionFailure,
    MeanOverMax,
    MaxOverMax,
}

impl std::fmt::Display for RawPointFallbackReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::ObservationKeysChanged => "observations-changed",
            Self::TooFewObservations => "too-few-observations",
            Self::NonFinite => "nonfinite",
            Self::NoObservableParallax => "no-observable-parallax",
            Self::ParallaxCheckFailed => "parallax-check-failed",
            Self::NonPositiveDepth => "nonpositive-depth",
            Self::ProjectionFailure => "projection-failure",
            Self::MeanOverMax => "mean-over-max",
            Self::MaxOverMax => "max-over-max",
        };
        f.write_str(label)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FilterObservationReason {
    NonFinite,
    BehindCamera,
    ProjectionFailure,
    ReprojectionOverMax,
    TrackBelowMinimum,
    RetriangulationFailed,
}

impl std::fmt::Display for FilterObservationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::NonFinite => "nonfinite",
            Self::BehindCamera => "behind-camera",
            Self::ProjectionFailure => "projection-failure",
            Self::ReprojectionOverMax => "reprojection-over-max",
            Self::TrackBelowMinimum => "track-below-minimum",
            Self::RetriangulationFailed => "retriangulation-failed",
        };
        f.write_str(label)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeRejection {
    DuplicateCandidateObservation,
    SameImageDifferentKeypoint,
    ExistingTrackSameImageConflict,
}

impl std::fmt::Display for MergeRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateCandidateObservation => write!(f, "duplicate candidate observation"),
            Self::SameImageDifferentKeypoint => write!(f, "same image has different keypoints"),
            Self::ExistingTrackSameImageConflict => {
                write!(f, "existing track already owns another keypoint in image")
            }
        }
    }
}

fn main() {
    let args = parse_args(std::env::args()).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    if let Err(error) = run(&args) {
        eprintln!("integrate_rig_atlas_landmarks: {error}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    let manifest = parse_rig_manifest(&args.rig_manifest)?;
    let nodes = parse_node_specs(&args.nodes_tsv)?;
    validate_write_destinations(args, &nodes)?;
    let atlas_images = parse_atlas_images(&args.atlas_dir, &manifest)?;
    let mut cameras = BTreeMap::<u64, Camera>::new();
    let mut global_images = atlas_images
        .values()
        .cloned()
        .map(|atlas| {
            (
                atlas.global_image_id,
                GlobalImage {
                    atlas,
                    keypoints: Vec::new(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut image_name_to_id = BTreeMap::new();
    for image in global_images.values() {
        image_name_to_id.insert(image.atlas.name.clone(), image.atlas.global_image_id);
    }

    let mut store = TrackStore::default();
    let mut stats = IngestStats::default();
    for node in &nodes {
        let source = read_source_window(node, &manifest, &mut cameras)?;
        stats.source_nodes += 1;
        stats.source_points += source.points.len();
        stats.source_observations += source
            .points
            .iter()
            .map(|point| point.observations.len())
            .sum::<usize>();
        update_global_images(&mut global_images, &source, &image_name_to_id, &mut stats)?;
        ingest_source_points(
            &source,
            &image_name_to_id,
            &global_images,
            &mut store,
            &mut stats,
        )?;
    }
    validate_global_images(&global_images)?;
    validate_output_cameras(&manifest, &global_images, &cameras)?;

    let mut landmarks = Vec::new();
    for (track_id, track) in store.tracks.iter_mut().enumerate() {
        track.observations.sort_unstable();
        track.observations.dedup();
        if track.observations.len() < MIN_TRACK_OBSERVATIONS {
            stats.rejected_short_candidates += 1;
            continue;
        }
        match triangulate_track(
            track_id,
            track,
            &store.observations,
            &global_images,
            &cameras,
        ) {
            Ok(landmark) => {
                if landmark.dlt_sample_count < landmark.observations.len() {
                    stats.triangulation_sampled_tracks += 1;
                    stats.triangulation_sampled_observations +=
                        landmark.observations.len() - landmark.dlt_sample_count;
                }
                landmarks.push(landmark)
            }
            Err(_) => stats.rejected_triangulation += 1,
        }
    }
    if args.diagnose_cross_boundary_pnp {
        let boundary = args
            .diagnostic_left_max_frame
            .expect("diagnostic boundary validated by parse_args");
        diagnose_cross_boundary_pnp(
            &manifest,
            boundary,
            &store,
            &global_images,
            &cameras,
            &landmarks,
        )?;
        // This mode is intentionally non-mutating and must not materialize a
        // second large COLMAP model merely to print the diagnostic evidence.
        return Ok(());
    }
    if args.recover_zero_support_frames {
        let RecoveryResult {
            pose_overrides,
            landmarks: recovered_landmarks,
            summary,
        } = recover_zero_support_frames(&manifest, &store, &global_images, &cameras, &landmarks)?;
        apply_pose_overrides(&mut global_images, &pose_overrides)?;
        validate_output_cameras(&manifest, &global_images, &cameras)?;
        landmarks.extend(recovered_landmarks);
        println!(
            "recovery frames_considered={} frames_attempted={} frames_recovered={} candidate_tracks={} anchor_tracks={} anchor_rejections={} pnp_correspondences={} pnp_inliers={} pnp_rejections={} candidate_target_landmarks={} candidate_target_observations={} accepted_landmarks={} accepted_observations={} accepted_target_landmarks={} accepted_target_observations={} track_rejections={}",
            summary.frames_considered,
            summary.frames_attempted,
            summary.frames_recovered,
            summary.candidate_tracks,
            summary.anchor_tracks,
            summary.anchor_rejections,
            summary.pnp_correspondences,
            summary.pnp_inliers,
            summary.pnp_rejections,
            summary.candidate_target_landmarks,
            summary.candidate_target_observations,
            summary.accepted_landmarks,
            summary.accepted_observations,
            summary.accepted_target_landmarks,
            summary.accepted_target_observations,
            summary.track_rejections,
        );
    }
    if args.repair_cross_boundary {
        let boundary = args
            .repair_left_max_frame
            .expect("repair boundary validated by parse_args");
        let summary = repair_cross_boundary(
            &manifest,
            boundary,
            &store,
            &mut global_images,
            &cameras,
            &mut landmarks,
        )?;
        validate_output_cameras(&manifest, &global_images, &cameras)?;
        println!(
            "boundary_repair status={} candidates={} attempted={} accepted_frame={} added_tracks={} added_observations={} removed_tracks={} removed_observations={} supported_component_count_before={} supported_component_count_after={}",
            summary.status,
            summary.candidates,
            summary.attempted,
            summary
                .accepted_frame
                .map_or_else(|| "none".to_owned(), |frame| frame.to_string()),
            summary.added_tracks,
            summary.added_observations,
            summary.removed_tracks,
            summary.removed_observations,
            summary.baseline_component_count,
            summary.candidate_component_count,
        );
    }
    if args.joint_rig_ba {
        if let Some(pre_ba_out_dir) = args.pre_ba_out_dir.as_deref() {
            require_nonempty_landmarks(&landmarks, landmark_observation_count(&landmarks))?;
            stats.output_landmarks = landmarks.len();
            stats.output_observations = landmark_observation_count(&landmarks);
            let support = write_model_canonical(
                pre_ba_out_dir,
                &global_images,
                &cameras,
                &landmarks,
                &stats,
            )?;
            println!(
                "pre_ba_checkpoint out_dir={} landmarks={} observations={} supported_images={} supported_frames={}",
                pre_ba_out_dir.display(),
                stats.output_landmarks,
                stats.output_observations,
                support.supported_image_count,
                support.supported_frame_count,
            );
        }
        if args.joint_rig_ba_filter_observations {
            if args.joint_rig_ba_filter_sweeps == 2 {
                let pass1 = run_joint_rig_ba_filtering_with_policy_and_pass(
                    &manifest,
                    &mut store,
                    &mut global_images,
                    &cameras,
                    &mut landmarks,
                    false,
                    Some(1),
                )?;
                validate_output_cameras(&manifest, &global_images, &cameras)?;
                print_joint_rig_ba_filter_summary(&pass1, false, Some(1));
                print_joint_rig_ba_filter_pass_support(1, &global_images, &landmarks)?;
                let post_pass1_out_dir = args
                    .post_pass1_out_dir
                    .as_deref()
                    .expect("sweep 2 validates --post-pass1-out-dir");
                stats.output_landmarks = landmarks.len();
                stats.output_observations = landmark_observation_count(&landmarks);
                require_nonempty_landmarks(&landmarks, stats.output_observations)?;
                let support = write_model_canonical(
                    post_pass1_out_dir,
                    &global_images,
                    &cameras,
                    &landmarks,
                    &stats,
                )?;
                println!(
                    "post_pass1_checkpoint out_dir={} landmarks={} observations={} supported_images={} supported_frames={}",
                    post_pass1_out_dir.display(),
                    stats.output_landmarks,
                    stats.output_observations,
                    support.supported_image_count,
                    support.supported_frame_count,
                );
                let pass2 = run_joint_rig_ba_filtering_with_policy_and_pass(
                    &manifest,
                    &mut store,
                    &mut global_images,
                    &cameras,
                    &mut landmarks,
                    false,
                    Some(2),
                )?;
                validate_output_cameras(&manifest, &global_images, &cameras)?;
                print_joint_rig_ba_filter_summary(&pass2, false, Some(2));
                print_joint_rig_ba_filter_pass_support(2, &global_images, &landmarks)?;
            } else {
                let summary = run_joint_rig_ba_filtering_with_policy(
                    &manifest,
                    &mut store,
                    &mut global_images,
                    &cameras,
                    &mut landmarks,
                    args.joint_rig_ba_preserve_optimized_points,
                )?;
                validate_output_cameras(&manifest, &global_images, &cameras)?;
                print_joint_rig_ba_filter_summary(
                    &summary,
                    args.joint_rig_ba_preserve_optimized_points,
                    None,
                );
            }
        } else {
            let summary = run_joint_rig_ba(
                &manifest,
                &store,
                &mut global_images,
                &cameras,
                &mut landmarks,
            )?;
            validate_output_cameras(&manifest, &global_images, &cameras)?;
            println!(
                "joint_rig_ba windows_considered={} windows_accepted={} windows_skipped={} selected_landmarks={} selected_observations={} max_referenced_frames={} max_free_frames={} max_iterations={} converged_windows={} max_solver_initial_cost={:.9} min_solver_final_cost={:.9} final_cost={:.9}",
                summary.windows_considered,
                summary.windows_accepted,
                summary.windows_skipped,
                summary.selected_landmarks,
                summary.selected_observations,
                summary.max_referenced_frames,
                summary.max_free_frames,
                summary.max_iterations,
                summary.converged_windows,
                summary.max_solver_initial_cost,
                summary.min_solver_final_cost,
                summary.final_cost,
            );
        }
    }
    landmarks.sort_by_key(|landmark| {
        (
            landmark
                .observations
                .first()
                .copied()
                .unwrap_or(ObservationKey {
                    global_image_id: u64::MAX,
                    keypoint_index: usize::MAX,
                }),
            landmark.track_id,
        )
    });
    stats.output_landmarks = landmarks.len();
    stats.output_observations = landmarks
        .iter()
        .map(|landmark| landmark.observations.len())
        .sum();
    require_nonempty_landmarks(&landmarks, stats.output_observations)?;
    let support = write_model(&args.out_dir, &global_images, &cameras, &landmarks, &stats)?;

    let errors = landmarks
        .iter()
        .flat_map(|landmark| landmark.errors.iter().copied())
        .collect::<Vec<_>>();
    let mean = if errors.is_empty() {
        0.0
    } else {
        errors.iter().sum::<f64>() / errors.len() as f64
    };
    let rms = if errors.is_empty() {
        0.0
    } else {
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt()
    };
    let p95 = percentile(&errors, 0.95).unwrap_or(0.0);
    let max = errors.iter().copied().fold(0.0, f64::max);
    println!(
        "nodes={} atlas_poses={} supported_images={} zero_support_images={} atlas_frames={} supported_frames={} zero_support_frames={} tracks={} landmarks={} observations={} mean_px={mean:.9} rms_px={rms:.9} p95_px={p95:.9} max_px={max:.9} sampled_tracks={} sampled_observations={} rejected_triangulation={} rejected_collision={}",
        stats.source_nodes,
        global_images.len(),
        support.supported_image_count,
        support.zero_support_image_count,
        support.atlas_frame_count,
        support.supported_frame_count,
        support.zero_support_frame_count,
        store.tracks.len(),
        stats.output_landmarks,
        stats.output_observations,
        stats.triangulation_sampled_tracks,
        stats.triangulation_sampled_observations,
        stats.rejected_triangulation,
        stats.rejected_candidates,
    );
    Ok(())
}

#[cfg(test)]
mod tests;
