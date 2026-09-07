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

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = args.into_iter();
    let _program = values.next();
    let mut rig_manifest = None;
    let mut nodes_tsv = None;
    let mut atlas_dir = None;
    let mut out_dir = None;
    let mut pre_ba_out_dir = None;
    let mut post_pass1_out_dir = None;
    let mut recover_zero_support_frames = false;
    let mut joint_rig_ba = false;
    let mut joint_rig_ba_filter_observations = false;
    let mut joint_rig_ba_preserve_optimized_points = false;
    let mut joint_rig_ba_filter_sweeps = 1u8;
    let mut filter_sweeps_explicit = false;
    let mut diagnose_cross_boundary_pnp = false;
    let mut diagnostic_left_max_frame = None;
    let mut repair_cross_boundary = false;
    let mut repair_left_max_frame = None;
    while let Some(flag) = values.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.to_owned());
        }
        if flag == "--recover-zero-support-frames" {
            if recover_zero_support_frames {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            recover_zero_support_frames = true;
            continue;
        }
        if flag == "--joint-rig-ba" {
            if joint_rig_ba {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba = true;
            continue;
        }
        if flag == "--joint-rig-ba-filter-observations" {
            if joint_rig_ba_filter_observations {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba_filter_observations = true;
            continue;
        }
        if flag == "--joint-rig-ba-preserve-optimized-points" {
            if joint_rig_ba_preserve_optimized_points {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            joint_rig_ba_preserve_optimized_points = true;
            continue;
        }
        if flag == "--pre-ba-out-dir" {
            if pre_ba_out_dir.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
            }
            pre_ba_out_dir = Some(PathBuf::from(value));
            continue;
        }
        if flag == "--post-pass1-out-dir" {
            if post_pass1_out_dir.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
            }
            post_pass1_out_dir = Some(PathBuf::from(value));
            continue;
        }
        if flag == "--joint-rig-ba-filter-sweeps" {
            if filter_sweeps_explicit {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires 1 or 2\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires 1 or 2, got {value:?}\n{USAGE}"));
            }
            joint_rig_ba_filter_sweeps = value
                .parse::<u8>()
                .map_err(|error| format!("{flag} requires 1 or 2: {error}\n{USAGE}"))?;
            if !matches!(joint_rig_ba_filter_sweeps, 1 | 2) {
                return Err(format!("{flag} accepts only 1 or 2\n{USAGE}"));
            }
            filter_sweeps_explicit = true;
            continue;
        }
        if flag == "--diagnose-cross-boundary-pnp" {
            if diagnose_cross_boundary_pnp {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            diagnose_cross_boundary_pnp = true;
            continue;
        }
        if flag == "--repair-cross-boundary" {
            if repair_cross_boundary {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            repair_cross_boundary = true;
            continue;
        }
        if flag == "--diagnostic-left-max-frame" {
            if diagnostic_left_max_frame.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires FRAME\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires non-negative FRAME, got {value:?}"));
            }
            let frame = value
                .parse::<u64>()
                .map_err(|error| format!("{flag} requires an integer FRAME: {error}\n{USAGE}"))?;
            diagnostic_left_max_frame = Some(frame);
            continue;
        }
        if flag == "--repair-left-max-frame" {
            if repair_left_max_frame.is_some() {
                return Err(format!("duplicate argument {flag}\n{USAGE}"));
            }
            let value = values
                .next()
                .ok_or_else(|| format!("{flag} requires FRAME\n{USAGE}"))?;
            if value.starts_with('-') {
                return Err(format!("{flag} requires non-negative FRAME, got {value:?}"));
            }
            let frame = value
                .parse::<u64>()
                .map_err(|error| format!("{flag} requires an integer FRAME: {error}\n{USAGE}"))?;
            repair_left_max_frame = Some(frame);
            continue;
        }
        let slot = match flag.as_str() {
            "--rig-manifest" => &mut rig_manifest,
            "--nodes-tsv" => &mut nodes_tsv,
            "--atlas-dir" => &mut atlas_dir,
            "--out-dir" => &mut out_dir,
            other => return Err(format!("unknown argument {other:?}\n{USAGE}")),
        };
        if slot.is_some() {
            return Err(format!("duplicate argument {flag}\n{USAGE}"));
        }
        let value = values
            .next()
            .ok_or_else(|| format!("{flag} requires PATH\n{USAGE}"))?;
        if value.starts_with('-') {
            return Err(format!("{flag} requires PATH, got {value:?}\n{USAGE}"));
        }
        *slot = Some(PathBuf::from(value));
    }
    if diagnose_cross_boundary_pnp && diagnostic_left_max_frame.is_none() {
        return Err(format!(
            "--diagnose-cross-boundary-pnp requires --diagnostic-left-max-frame FRAME\n{USAGE}"
        ));
    }
    if !diagnose_cross_boundary_pnp && diagnostic_left_max_frame.is_some() {
        return Err(format!(
            "--diagnostic-left-max-frame requires --diagnose-cross-boundary-pnp\n{USAGE}"
        ));
    }
    if repair_cross_boundary && repair_left_max_frame.is_none() {
        return Err(format!(
            "--repair-cross-boundary requires --repair-left-max-frame FRAME\n{USAGE}"
        ));
    }
    if !repair_cross_boundary && repair_left_max_frame.is_some() {
        return Err(format!(
            "--repair-left-max-frame requires --repair-cross-boundary\n{USAGE}"
        ));
    }
    if diagnose_cross_boundary_pnp && (recover_zero_support_frames || joint_rig_ba) {
        return Err(format!(
            "--diagnose-cross-boundary-pnp is diagnostic-only and cannot be combined with recovery or BA\n{USAGE}"
        ));
    }
    if diagnose_cross_boundary_pnp && repair_cross_boundary {
        return Err(format!(
            "--diagnose-cross-boundary-pnp and --repair-cross-boundary are mutually exclusive\n{USAGE}"
        ));
    }
    if pre_ba_out_dir.is_some() && !joint_rig_ba {
        return Err(format!("--pre-ba-out-dir requires --joint-rig-ba\n{USAGE}"));
    }
    if post_pass1_out_dir.is_some() && joint_rig_ba_filter_sweeps != 2 {
        return Err(format!(
            "--post-pass1-out-dir requires --joint-rig-ba-filter-sweeps 2\n{USAGE}"
        ));
    }
    if filter_sweeps_explicit && !joint_rig_ba_filter_observations {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps requires --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_observations && !joint_rig_ba {
        return Err(format!(
            "--joint-rig-ba-filter-observations requires --joint-rig-ba\n{USAGE}"
        ));
    }
    if joint_rig_ba_preserve_optimized_points && !(joint_rig_ba && joint_rig_ba_filter_observations)
    {
        return Err(format!(
            "--joint-rig-ba-preserve-optimized-points requires --joint-rig-ba and --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && post_pass1_out_dir.is_none() {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 requires --post-pass1-out-dir PATH\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && !joint_rig_ba_filter_observations {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 requires --joint-rig-ba-filter-observations\n{USAGE}"
        ));
    }
    if joint_rig_ba_filter_sweeps == 2 && joint_rig_ba_preserve_optimized_points {
        return Err(format!(
            "--joint-rig-ba-filter-sweeps 2 cannot be combined with --joint-rig-ba-preserve-optimized-points\n{USAGE}"
        ));
    }
    Ok(Args {
        rig_manifest: rig_manifest.ok_or_else(|| format!("--rig-manifest is required\n{USAGE}"))?,
        nodes_tsv: nodes_tsv.ok_or_else(|| format!("--nodes-tsv is required\n{USAGE}"))?,
        atlas_dir: atlas_dir.ok_or_else(|| format!("--atlas-dir is required\n{USAGE}"))?,
        out_dir: out_dir.ok_or_else(|| format!("--out-dir is required\n{USAGE}"))?,
        pre_ba_out_dir,
        post_pass1_out_dir,
        recover_zero_support_frames,
        joint_rig_ba,
        joint_rig_ba_filter_observations,
        joint_rig_ba_preserve_optimized_points,
        joint_rig_ba_filter_sweeps,
        diagnose_cross_boundary_pnp,
        diagnostic_left_max_frame,
        repair_cross_boundary,
        repair_left_max_frame,
    })
}

fn resolved_comparison_path(path: &Path) -> Result<PathBuf, String> {
    let absolute = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|error| format!("resolve current directory: {error}"))?
            .join(path)
    };
    let mut probe = absolute;
    let mut missing = Vec::new();
    let base = loop {
        match fs::canonicalize(&probe) {
            Ok(path) => break path,
            Err(error) => {
                let Some(name) = probe.file_name() else {
                    return Err(format!("cannot resolve path {:?}: {error}", path));
                };
                missing.push(name.to_os_string());
                if !probe.pop() {
                    return Err(format!("cannot resolve path {:?}: {error}", path));
                }
            }
        }
    };
    let mut resolved = base;
    for component in missing.iter().rev() {
        if component == "." {
            continue;
        }
        if component == ".." {
            resolved.pop();
        } else {
            resolved.push(component);
        }
    }
    Ok(resolved)
}

fn paths_overlap(first: &Path, second: &Path) -> Result<bool, String> {
    let first = resolved_comparison_path(first)?;
    let second = resolved_comparison_path(second)?;
    Ok(first == second || first.starts_with(&second) || second.starts_with(&first))
}

fn validate_write_destinations(args: &Args, nodes: &[NodeSpec]) -> Result<(), String> {
    let mut destinations = vec![("--out-dir", args.out_dir.as_path())];
    if let Some(path) = args.pre_ba_out_dir.as_deref() {
        destinations.push(("--pre-ba-out-dir", path));
    }
    if let Some(path) = args.post_pass1_out_dir.as_deref() {
        destinations.push(("--post-pass1-out-dir", path));
    }
    let mut inputs = vec![
        ("--rig-manifest", args.rig_manifest.as_path()),
        ("--nodes-tsv", args.nodes_tsv.as_path()),
        ("--atlas-dir", args.atlas_dir.as_path()),
    ];
    inputs.extend(
        nodes
            .iter()
            .map(|node| ("node images.txt", node.images_txt.as_path())),
    );
    for (index, (destination_name, destination)) in destinations.iter().enumerate() {
        for (other_name, other) in destinations.iter().skip(index + 1) {
            if paths_overlap(destination, other)? {
                return Err(format!(
                    "{destination_name} {:?} overlaps {other_name} {:?}; refusing ambiguous output destinations",
                    destination, other
                ));
            }
        }
        for (input_name, input) in &inputs {
            if paths_overlap(destination, input)? {
                return Err(format!(
                    "{destination_name} {:?} overlaps input {input_name} {:?}; refusing to write input data",
                    destination, input
                ));
            }
        }
    }
    if let Some(path) = args.pre_ba_out_dir.as_deref() {
        if path.exists() {
            if !path.is_dir() {
                return Err(format!(
                    "--pre-ba-out-dir {:?} exists but is not a directory",
                    path
                ));
            }
            let mut entries = fs::read_dir(path)
                .map_err(|error| format!("read --pre-ba-out-dir {}: {error}", path.display()))?;
            if entries
                .next()
                .transpose()
                .map_err(|error| format!("inspect --pre-ba-out-dir {}: {error}", path.display()))?
                .is_some()
            {
                return Err(format!(
                    "--pre-ba-out-dir {:?} must be new or empty; refusing to overwrite an existing checkpoint",
                    path
                ));
            }
        }
    }
    if let Some(path) = args.post_pass1_out_dir.as_deref() {
        if path.exists() {
            if !path.is_dir() {
                return Err(format!(
                    "--post-pass1-out-dir {:?} exists but is not a directory",
                    path
                ));
            }
            let mut entries = fs::read_dir(path).map_err(|error| {
                format!("read --post-pass1-out-dir {}: {error}", path.display())
            })?;
            if entries
                .next()
                .transpose()
                .map_err(|error| {
                    format!("inspect --post-pass1-out-dir {}: {error}", path.display())
                })?
                .is_some()
            {
                return Err(format!(
                    "--post-pass1-out-dir {:?} must be new or empty; refusing to overwrite an existing checkpoint",
                    path
                ));
            }
        }
    }
    Ok(())
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

fn parse_node_specs(path: &Path) -> Result<Vec<NodeSpec>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read node manifest {}: {error}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    let mut nodes = Vec::new();
    let mut ids = BTreeSet::new();
    let mut starts_paths = BTreeSet::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "nodes {}:{} requires node_id<TAB>window_start<TAB>images.txt",
                path.display(),
                line
            ));
        }
        let node_id = parse_u64(fields[0], "node id", path, line)?;
        let window_start = parse_u64(fields[1], "window start", path, line)?;
        let images_txt = resolve_path(base, fields[2]);
        if !ids.insert(node_id) {
            return Err(format!(
                "nodes {}:{} duplicates node id {node_id}",
                path.display(),
                line
            ));
        }
        if !starts_paths.insert((window_start, images_txt.clone())) {
            return Err(format!(
                "nodes {}:{} duplicates ({window_start}, {})",
                path.display(),
                line,
                images_txt.display()
            ));
        }
        if !images_txt.is_file() {
            return Err(format!(
                "nodes {}:{} source images.txt does not exist: {}",
                path.display(),
                line,
                images_txt.display()
            ));
        }
        nodes.push(NodeSpec {
            node_id,
            window_start,
            images_txt,
        });
    }
    if nodes.is_empty() {
        return Err(format!("nodes {} contains no rows", path.display()));
    }
    nodes.sort_by_key(|node| (node.window_start, node.node_id));
    Ok(nodes)
}

fn parse_rig_manifest(path: &Path) -> Result<RigManifest, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read rig manifest {}: {error}", path.display()))?;
    let mut sensors = BTreeMap::new();
    let mut frame_rows = Vec::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        match fields.first().copied() {
            Some("S") => {
                if fields.len() != 16 {
                    return Err(format!(
                        "rig manifest {}:{} sensor row requires 16 fields",
                        path.display(),
                        line
                    ));
                }
                let index = parse_usize(fields[1], "sensor index", path, line)?;
                let camera_id = parse_u64(fields[2], "camera id", path, line)?;
                let width = parse_u32(fields[3], "sensor width", path, line)?;
                let height = parse_u32(fields[4], "sensor height", path, line)?;
                if width == 0 || height == 0 {
                    return Err(format!(
                        "rig manifest {}:{} has zero sensor size",
                        path.display(),
                        line
                    ));
                }
                let values = (5..16)
                    .map(|index| parse_f64(fields[index], "sensor value", path, line))
                    .collect::<Result<Vec<_>, _>>()?;
                let quaternion = Quaternion::new(values[4], values[5], values[6], values[7]);
                let norm = quaternion.norm();
                if !norm.is_finite() || norm <= MIN_HOMOGENEOUS_SCALE {
                    return Err(format!(
                        "rig manifest {}:{} has invalid sensor quaternion",
                        path.display(),
                        line
                    ));
                }
                if sensors
                    .insert(
                        index,
                        SensorCalibration {
                            camera_id,
                            width,
                            height,
                            fx: values[0],
                            fy: values[1],
                            cx: values[2],
                            cy: values[3],
                            sensor_from_rig: SE3::new(
                                UnitQuaternion::new_normalize(quaternion),
                                Vector3::new(values[8], values[9], values[10]),
                            ),
                        },
                    )
                    .is_some()
                {
                    return Err(format!(
                        "rig manifest {}:{} duplicates sensor {index}",
                        path.display(),
                        line
                    ));
                }
            }
            Some("F") => {
                if fields.len() != 4 {
                    return Err(format!(
                        "rig manifest {}:{} frame row requires 4 fields",
                        path.display(),
                        line
                    ));
                }
                let frame_id = parse_u64(fields[1], "frame id", path, line)?;
                let name = fields[2].to_owned();
                if name.is_empty() {
                    return Err(format!(
                        "rig manifest {}:{} has empty image name",
                        path.display(),
                        line
                    ));
                }
                let sensor_index = parse_usize(fields[3], "sensor index", path, line)?;
                frame_rows.push((frame_id, name, sensor_index, line));
            }
            Some(kind) => {
                return Err(format!(
                    "rig manifest {}:{} unknown row kind {kind:?}",
                    path.display(),
                    line
                ));
            }
            None => unreachable!(),
        }
    }
    if sensors.is_empty() || frame_rows.is_empty() {
        return Err(format!(
            "rig manifest {} has no sensors or frame rows",
            path.display()
        ));
    }
    for (expected, actual) in sensors.keys().copied().enumerate() {
        if expected != actual {
            return Err(format!(
                "rig manifest {} sensor indices must be contiguous from zero",
                path.display()
            ));
        }
    }
    let mut assignments = BTreeMap::new();
    let mut frame_sensors = BTreeSet::new();
    for (frame_id, name, sensor_index, line) in frame_rows {
        if !sensors.contains_key(&sensor_index) {
            return Err(format!(
                "rig manifest {}:{} references unknown sensor {sensor_index}",
                path.display(),
                line
            ));
        }
        if !frame_sensors.insert((frame_id, sensor_index)) {
            return Err(format!(
                "rig manifest {}:{} duplicates frame/sensor ({frame_id}, {sensor_index})",
                path.display(),
                line
            ));
        }
        let global_image_id = assignments.len() as u64 + 1;
        if assignments
            .insert(
                name,
                ImageAssignment {
                    frame_id,
                    sensor_index,
                    global_image_id,
                },
            )
            .is_some()
        {
            return Err(format!(
                "rig manifest {}:{} duplicates image name",
                path.display(),
                line
            ));
        }
    }
    Ok(RigManifest {
        sensors,
        assignments,
    })
}

fn parse_atlas_images(
    atlas_dir: &Path,
    manifest: &RigManifest,
) -> Result<BTreeMap<String, AtlasImage>, String> {
    let mut paths = Vec::new();
    let direct = atlas_dir.join("images.txt");
    if direct.is_file() {
        paths.push(direct);
    } else {
        let entries = fs::read_dir(atlas_dir)
            .map_err(|error| format!("read atlas directory {}: {error}", atlas_dir.display()))?;
        let mut component_paths = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| format!("read atlas entry: {error}"))?;
            let path = entry.path();
            if entry
                .file_type()
                .map_err(|error| format!("stat atlas entry: {error}"))?
                .is_dir()
                && entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("component-")
            {
                let image_path = path.join("images.txt");
                if image_path.is_file() {
                    component_paths.push(image_path);
                }
            }
        }
        component_paths.sort();
        match component_paths.as_slice() {
            [] => {}
            [only] => paths.push(only.clone()),
            _ => {
                return Err(format!(
                    "atlas {} contains {} independent component images.txt files; pass one component directory per invocation",
                    atlas_dir.display(),
                    component_paths.len()
                ));
            }
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err(format!(
            "atlas {} contains no images.txt",
            atlas_dir.display()
        ));
    }
    let mut images = BTreeMap::new();
    for path in paths {
        for image in parse_atlas_images_file(&path)? {
            let assignment = manifest.assignments.get(&image.name).ok_or_else(|| {
                format!("atlas image {:?} is absent from rig manifest", image.name)
            })?;
            let sensor = manifest
                .sensors
                .get(&assignment.sensor_index)
                .expect("manifest sensor index was validated");
            if image.camera_id != sensor.camera_id {
                return Err(format!(
                    "atlas image {:?} camera {} disagrees with manifest camera {}",
                    image.name, image.camera_id, sensor.camera_id
                ));
            }
            let bound = AtlasImage {
                global_image_id: assignment.global_image_id,
                frame_id: assignment.frame_id,
                sensor_index: assignment.sensor_index,
                name: image.name.clone(),
                camera_id: image.camera_id,
                pose: image.pose,
            };
            if images.insert(image.name.clone(), bound).is_some() {
                return Err(format!(
                    "atlas contains duplicate image name {:?}",
                    image.name
                ));
            }
        }
    }
    Ok(images)
}

#[derive(Debug, Clone, PartialEq)]
struct ParsedPoseImage {
    camera_id: u64,
    name: String,
    pose: Pose,
}

fn parse_atlas_images_file(path: &Path) -> Result<Vec<ParsedPoseImage>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read atlas images {}: {error}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let mut parsed = Vec::new();
    let mut index = 0;
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    while let Some((line_number, header)) = next_data_line(&lines, &mut index) {
        let fields = header.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 10 {
            return Err(format!(
                "atlas images {}:{} pose row requires at least 10 fields",
                path.display(),
                line_number
            ));
        }
        let local_id = parse_u64(fields[0], "image id", path, line_number)?;
        let quaternion = parse_quaternion(&fields[1..5], path, line_number)?;
        let translation = parse_vector(&fields[5..8], path, line_number)?;
        let camera_id = parse_u64(fields[8], "camera id", path, line_number)?;
        let name = fields[9].to_owned();
        if !ids.insert(local_id) || !names.insert(name.clone()) {
            return Err(format!(
                "atlas images {}:{} duplicates image id or name",
                path.display(),
                line_number
            ));
        }
        let points_line = next_required_line(&lines, &mut index).ok_or_else(|| {
            format!(
                "atlas images {}:{} missing POINTS2D row",
                path.display(),
                line_number
            )
        })?;
        // The atlas produced by the trajectory stitcher intentionally has no
        // POINTS2D.  Still parse non-empty rows so malformed input fails closed.
        parse_points2d(points_line.1, path, points_line.0)?;
        parsed.push(ParsedPoseImage {
            camera_id,
            name,
            pose: Pose::from_world_to_camera(quaternion, translation),
        });
    }
    if parsed.is_empty() {
        return Err(format!(
            "atlas images {} contains no images",
            path.display()
        ));
    }
    Ok(parsed)
}

fn read_source_window(
    node: &NodeSpec,
    manifest: &RigManifest,
    cameras: &mut BTreeMap<u64, Camera>,
) -> Result<SourceWindow, String> {
    let model_dir = node
        .images_txt
        .parent()
        .ok_or_else(|| format!("node {} images path has no parent", node.node_id))?;
    let cameras_path = model_dir.join("cameras.txt");
    let points_path = model_dir.join("points3D.txt");
    let camera_text = fs::read_to_string(&cameras_path).map_err(|error| {
        format!(
            "node {} read {}: {error}",
            node.node_id,
            cameras_path.display()
        )
    })?;
    let source_cameras = parse_cameras_txt(&camera_text)
        .map_err(|error| format!("node {} parse cameras: {error}", node.node_id))?;
    let mut local_cameras = BTreeMap::new();
    for camera in source_cameras {
        if camera.width == 0
            || camera.height == 0
            || camera.params.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "node {} has invalid camera {}",
                node.node_id, camera.id
            ));
        }
        if local_cameras.insert(camera.id, camera.clone()).is_some() {
            return Err(format!(
                "node {} duplicates camera {}",
                node.node_id, camera.id
            ));
        }
        if let Some(previous) = cameras.insert(camera.id, camera.clone()) {
            if previous != camera {
                return Err(format!(
                    "camera {} has incompatible definitions across source windows",
                    camera.id
                ));
            }
        }
    }
    let images = parse_source_images(&node.images_txt)?;
    for image in &images {
        let assignment = manifest.assignments.get(&image.name).ok_or_else(|| {
            format!(
                "node {} image {:?} is absent from rig manifest",
                node.node_id, image.name
            )
        })?;
        let sensor = manifest
            .sensors
            .get(&assignment.sensor_index)
            .expect("manifest sensor index was validated");
        let camera = local_cameras.get(&image.camera_id).ok_or_else(|| {
            format!(
                "node {} image {:?} references missing camera {}",
                node.node_id, image.name, image.camera_id
            )
        })?;
        if image.camera_id != sensor.camera_id
            || camera.width != sensor.width
            || camera.height != sensor.height
            || !camera_matches_manifest(camera, sensor)
        {
            return Err(format!(
                "node {} image {:?} calibration disagrees with manifest sensor {}",
                node.node_id, image.name, assignment.sensor_index
            ));
        }
    }
    let points_text = fs::read_to_string(&points_path).map_err(|error| {
        format!(
            "node {} read {}: {error}",
            node.node_id,
            points_path.display()
        )
    })?;
    let points = parse_source_points(&points_text, &node.images_txt)?;
    validate_source_bidirectional(&images, &points, &node.images_txt)?;
    Ok(SourceWindow { images, points })
}

fn parse_source_images(path: &Path) -> Result<Vec<SourceImage>, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("read source images {}: {error}", path.display()))?;
    let lines = contents.lines().collect::<Vec<_>>();
    let mut index = 0;
    let mut images = Vec::new();
    let mut ids = BTreeSet::new();
    let mut names = BTreeSet::new();
    while let Some((line_number, header)) = next_data_line(&lines, &mut index) {
        let fields = header.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 10 {
            return Err(format!(
                "source images {}:{} pose row requires at least 10 fields",
                path.display(),
                line_number
            ));
        }
        let local_id = parse_u64(fields[0], "image id", path, line_number)?;
        let _quaternion = parse_quaternion(&fields[1..5], path, line_number)?;
        let _translation = parse_vector(&fields[5..8], path, line_number)?;
        let camera_id = parse_u64(fields[8], "camera id", path, line_number)?;
        let name = fields[9].to_owned();
        if !ids.insert(local_id) {
            return Err(format!(
                "source images {}:{} duplicate image id",
                path.display(),
                line_number
            ));
        }
        if !names.insert(name.clone()) {
            return Err(format!(
                "source images {}:{} duplicate image name",
                path.display(),
                line_number
            ));
        }
        let points_line = next_required_line(&lines, &mut index).ok_or_else(|| {
            format!(
                "source images {}:{} missing POINTS2D row",
                path.display(),
                line_number
            )
        })?;
        let points = parse_points2d(points_line.1, path, points_line.0)?;
        images.push(SourceImage {
            local_id,
            camera_id,
            name,
            keypoints: points,
        });
    }
    if images.is_empty() {
        return Err(format!(
            "source images {} contains no images",
            path.display()
        ));
    }
    Ok(images)
}

fn parse_source_points(contents: &str, images_path: &Path) -> Result<Vec<SourcePoint>, String> {
    let mut points = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, raw) in contents.lines().enumerate() {
        let line_number = line_index + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let fields = text.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 8 || (fields.len() - 8) % 2 != 0 {
            return Err(format!(
                "source points3D {}:{} has malformed fields",
                images_path.display(),
                line_number
            ));
        }
        let id = parse_u64(fields[0], "POINT3D_ID", images_path, line_number)?;
        if !ids.insert(id) {
            return Err(format!(
                "source points3D {}:{} duplicate point {id}",
                images_path.display(),
                line_number
            ));
        }
        let x = parse_f64(fields[1], "point x", images_path, line_number)?;
        let y = parse_f64(fields[2], "point y", images_path, line_number)?;
        let z = parse_f64(fields[3], "point z", images_path, line_number)?;
        let error = parse_f64(fields[7], "point error", images_path, line_number)?;
        if !error.is_finite() {
            return Err(format!(
                "source points3D {}:{} has non-finite error",
                images_path.display(),
                line_number
            ));
        }
        let mut observations = Vec::new();
        for pair in fields[8..].chunks_exact(2) {
            let image_id = parse_u64(pair[0], "track image id", images_path, line_number)?;
            let keypoint_index =
                parse_usize(pair[1], "track keypoint index", images_path, line_number)?;
            observations.push(LocalObservation {
                image_id,
                keypoint_index,
            });
        }
        observations.sort_unstable();
        if observations.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(format!(
                "source points3D {}:{} repeats a track observation",
                images_path.display(),
                line_number
            ));
        }
        if observations
            .windows(2)
            .any(|pair| pair[0].image_id == pair[1].image_id)
        {
            return Err(format!(
                "source points3D {}:{} uses two keypoints from one image",
                images_path.display(),
                line_number
            ));
        }
        points.push(SourcePoint {
            id,
            position: Point3::new(x, y, z),
            observations,
        });
    }
    points.sort_by_key(|point| point.id);
    Ok(points)
}

fn validate_source_bidirectional(
    images: &[SourceImage],
    points: &[SourcePoint],
    path: &Path,
) -> Result<(), String> {
    let by_id = images
        .iter()
        .map(|image| (image.local_id, image))
        .collect::<BTreeMap<_, _>>();
    let mut references = BTreeMap::<LocalObservation, u64>::new();
    for point in points {
        if !point.position.coords.iter().all(|value| value.is_finite()) {
            return Err(format!(
                "source points3D {} has non-finite point",
                path.display()
            ));
        }
        for observation in &point.observations {
            let image = by_id.get(&observation.image_id).ok_or_else(|| {
                format!(
                    "source points3D {} references unknown image {}",
                    path.display(),
                    observation.image_id
                )
            })?;
            let keypoint = image
                .keypoints
                .get(observation.keypoint_index)
                .ok_or_else(|| {
                    format!(
                        "source points3D {} references image {} keypoint {} out of range",
                        path.display(),
                        observation.image_id,
                        observation.keypoint_index
                    )
                })?;
            if keypoint.point3d_id != Some(point.id) {
                return Err(format!(
                    "source points3D {} has non-bidirectional ref image {} keypoint {}",
                    path.display(),
                    observation.image_id,
                    observation.keypoint_index
                ));
            }
            if references.insert(observation.clone(), point.id).is_some() {
                return Err(format!(
                    "source points3D {} repeats image {} keypoint {}",
                    path.display(),
                    observation.image_id,
                    observation.keypoint_index
                ));
            }
        }
    }
    for image in images {
        for (keypoint_index, keypoint) in image.keypoints.iter().enumerate() {
            if let Some(point_id) = keypoint.point3d_id {
                if references.get(&LocalObservation {
                    image_id: image.local_id,
                    keypoint_index,
                }) != Some(&point_id)
                {
                    return Err(format!(
                        "source images {} has POINTS2D ref without matching track",
                        path.display()
                    ));
                }
            }
        }
    }
    Ok(())
}

fn update_global_images(
    global_images: &mut BTreeMap<u64, GlobalImage>,
    source: &SourceWindow,
    image_name_to_id: &BTreeMap<String, u64>,
    stats: &mut IngestStats,
) -> Result<(), String> {
    for image in &source.images {
        let Some(global_image_id) = image_name_to_id.get(&image.name).copied() else {
            stats.rejected_invalid_source += 1;
            continue;
        };
        let Some(global) = global_images.get_mut(&global_image_id) else {
            return Err(format!("manifest/atlas lost image {:?}", image.name));
        };
        if global.atlas.camera_id != image.camera_id {
            return Err(format!(
                "image {:?} changes camera id across source windows",
                image.name
            ));
        }
        let keypoints = image
            .keypoints
            .iter()
            .map(|keypoint| keypoint.xy)
            .collect::<Vec<_>>();
        if global.keypoints.is_empty() {
            global.keypoints = keypoints;
        } else if global.keypoints.len() != keypoints.len()
            || global
                .keypoints
                .iter()
                .zip(keypoints.iter())
                .any(|(left, right)| !points_close(left, right))
        {
            return Err(format!(
                "image {:?} has inconsistent ordered keypoint coordinates across windows",
                image.name
            ));
        }
    }
    Ok(())
}

fn validate_global_images(global_images: &BTreeMap<u64, GlobalImage>) -> Result<(), String> {
    for image in global_images.values() {
        if image.keypoints.is_empty() {
            return Err(format!(
                "atlas image {:?} has no source keypoint array",
                image.atlas.name
            ));
        }
    }
    Ok(())
}

fn ingest_source_points(
    source: &SourceWindow,
    image_name_to_id: &BTreeMap<String, u64>,
    global_images: &BTreeMap<u64, GlobalImage>,
    store: &mut TrackStore,
    stats: &mut IngestStats,
) -> Result<(), String> {
    let local_images = source
        .images
        .iter()
        .map(|image| (image.local_id, image))
        .collect::<BTreeMap<_, _>>();
    for point in &source.points {
        let mut candidate = Vec::with_capacity(point.observations.len());
        let mut filtered_out = 0;
        for observation in &point.observations {
            let image = local_images.get(&observation.image_id).ok_or_else(|| {
                format!(
                    "validated source point {} references missing image",
                    point.id
                )
            })?;
            let Some(global_image_id) = image_name_to_id.get(&image.name).copied() else {
                stats.rejected_missing_atlas_images += 1;
                filtered_out += 1;
                continue;
            };
            let global = global_images
                .get(&global_image_id)
                .expect("global image id came from the atlas manifest");
            if observation.keypoint_index >= global.keypoints.len() {
                return Err(format!(
                    "source point {} keypoint {} exceeds canonical image {:?}",
                    point.id, observation.keypoint_index, image.name
                ));
            }
            candidate.push(ObservationKey {
                global_image_id,
                keypoint_index: observation.keypoint_index,
            });
        }
        stats.filtered_out_of_component_observations += filtered_out;
        if candidate.len() < MIN_TRACK_OBSERVATIONS {
            stats.rejected_short_candidates += 1;
            continue;
        }
        candidate.sort_unstable();
        match merge_candidate(store, &candidate, global_images) {
            Ok(MergeOutcome::New) => stats.accepted_candidates += 1,
            Ok(MergeOutcome::Extended { duplicate_count }) => {
                stats.extended_candidates += 1;
                stats.duplicate_observations += duplicate_count;
            }
            Err(rejection) => {
                stats.rejected_candidates += 1;
                match rejection {
                    MergeRejection::SameImageDifferentKeypoint
                    | MergeRejection::ExistingTrackSameImageConflict => {
                        stats.rejected_same_image += 1;
                        if matches!(rejection, MergeRejection::ExistingTrackSameImageConflict) {
                            stats.rejected_track_conflict += 1;
                        }
                    }
                    MergeRejection::DuplicateCandidateObservation => stats.rejected_xy += 1,
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeOutcome {
    New,
    Extended { duplicate_count: usize },
}

fn merge_candidate(
    store: &mut TrackStore,
    candidate: &[ObservationKey],
    global_images: &BTreeMap<u64, GlobalImage>,
) -> Result<MergeOutcome, MergeRejection> {
    if candidate.is_empty() {
        return Err(MergeRejection::DuplicateCandidateObservation);
    }
    let mut image_keypoints = BTreeMap::<u64, usize>::new();
    let mut owners = BTreeSet::new();
    for key in candidate {
        if let Some(previous) = image_keypoints.insert(key.global_image_id, key.keypoint_index) {
            if previous != key.keypoint_index {
                return Err(MergeRejection::SameImageDifferentKeypoint);
            }
            return Err(MergeRejection::DuplicateCandidateObservation);
        }
        if let Some(state) = store.observations.get(key) {
            owners.insert(state.owner_track);
        }
        let image = global_images
            .get(&key.global_image_id)
            .expect("candidate was built from atlas images");
        if key.keypoint_index >= image.keypoints.len() {
            return Err(MergeRejection::DuplicateCandidateObservation);
        }
    }
    let owner = owners.iter().next().copied();
    let mut combined = BTreeSet::new();
    for key in candidate {
        combined.insert(*key);
    }
    for owner_id in &owners {
        for key in &store.tracks[*owner_id].observations {
            combined.insert(*key);
        }
    }
    let mut combined_image_keypoints = BTreeMap::<u64, usize>::new();
    for key in &combined {
        if let Some(previous) =
            combined_image_keypoints.insert(key.global_image_id, key.keypoint_index)
        {
            if previous != key.keypoint_index {
                return Err(MergeRejection::ExistingTrackSameImageConflict);
            }
        }
    }
    let target = match owner {
        Some(owner) => owner,
        None => {
            let track_id = store.tracks.len();
            store.tracks.push(GlobalTrack {
                observations: Vec::new(),
            });
            track_id
        }
    };
    let duplicate_count = candidate
        .iter()
        .filter(|key| store.observations.contains_key(key))
        .count();
    if owners.len() > 1 {
        for owner_id in owners
            .iter()
            .copied()
            .filter(|owner_id| *owner_id != target)
        {
            store.tracks[owner_id].observations.clear();
        }
    }
    let combined = combined.into_iter().collect::<Vec<_>>();
    store.tracks[target].observations = combined.clone();
    for key in &combined {
        let image = global_images
            .get(&key.global_image_id)
            .expect("candidate was built from atlas images");
        let xy = image.keypoints[key.keypoint_index];
        match store.observations.get_mut(key) {
            Some(state) => {
                state.owner_track = target;
                state.xy = xy;
            }
            None => {
                store.observations.insert(
                    *key,
                    ObservationState {
                        xy,
                        owner_track: target,
                    },
                );
            }
        }
    }
    if owner.is_some() {
        Ok(MergeOutcome::Extended { duplicate_count })
    } else {
        Ok(MergeOutcome::New)
    }
}

fn triangulate_track(
    track_id: usize,
    track: &GlobalTrack,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<LandmarkOutput, String> {
    triangulate_track_with_pose_overrides(
        track_id,
        track,
        observations,
        images,
        cameras,
        &BTreeMap::new(),
    )
}

fn triangulate_track_with_pose_overrides(
    track_id: usize,
    track: &GlobalTrack,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkOutput, String> {
    if track.observations.len() < MIN_TRACK_OBSERVATIONS {
        return Err("track has too few observations".to_owned());
    }
    let sample = bounded_sample(&track.observations, MAX_DLT_OBSERVATIONS);
    if !has_observable_parallax(&sample, observations, images, cameras, pose_overrides)? {
        return Err("track has no observable camera baseline/parallax".to_owned());
    }
    let mut matrix = DMatrix::<f64>::zeros(sample.len() * 2, 4);
    for (index, key) in sample.iter().enumerate() {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let normalized = camera
            .normalize_pixel(&state.xy)
            .ok_or_else(|| "camera model cannot normalize pixel".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let pose_matrix = pose.matrix();
        let row_x = index * 2;
        let row_y = row_x + 1;
        for column in 0..4 {
            matrix[(row_x, column)] =
                normalized.x * pose_matrix[(2, column)] - pose_matrix[(0, column)];
            matrix[(row_y, column)] =
                normalized.y * pose_matrix[(2, column)] - pose_matrix[(1, column)];
        }
    }
    let svd = matrix.svd(false, true);
    let v_t = svd
        .v_t
        .ok_or_else(|| "triangulation SVD has no V^T".to_owned())?;
    let homogeneous = v_t.row(v_t.nrows() - 1);
    let w = homogeneous[3];
    if !w.is_finite() || w.abs() <= MIN_HOMOGENEOUS_SCALE {
        return Err("triangulation is degenerate".to_owned());
    }
    let position = Point3::new(homogeneous[0] / w, homogeneous[1] / w, homogeneous[2] / w);
    if !position.coords.iter().all(|value| value.is_finite()) {
        return Err("triangulated position is non-finite".to_owned());
    }
    let mut errors = Vec::with_capacity(track.observations.len());
    for key in &track.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
            return Err("triangulated point is behind a camera".to_owned());
        }
        let projected = camera
            .project(&point_camera)
            .ok_or_else(|| "projection failed".to_owned())?;
        let error = (projected - state.xy).norm();
        if !error.is_finite() {
            return Err("reprojection is non-finite".to_owned());
        }
        errors.push(error);
    }
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms_error =
        (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    let max_error = errors.iter().copied().fold(0.0, f64::max);
    if mean_error > MAX_MEAN_REPROJECTION_PX || max_error > MAX_REPROJECTION_PX {
        return Err(format!(
            "reprojection gate failed mean={mean_error} max={max_error}"
        ));
    }
    Ok(LandmarkOutput {
        track_id,
        position,
        observations: track.observations.clone(),
        errors,
        rms_error,
        mean_error,
        max_error,
        dlt_sample_count: sample.len(),
    })
}

#[derive(Debug, Clone, PartialEq)]
struct DiagnosticCrossBoundaryTrack {
    track_id: usize,
    left_keys: Vec<ObservationKey>,
    right_keys: Vec<ObservationKey>,
}

#[derive(Debug, Clone, PartialEq)]
enum DiagnosticDltStatus {
    Short {
        observations: usize,
    },
    Pass {
        observations: usize,
        mean_error: f64,
        max_error: f64,
    },
    Rejected {
        observations: usize,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
struct DiagnosticDltResult {
    status: DiagnosticDltStatus,
    landmark: Option<LandmarkOutput>,
}

#[derive(Debug, Clone)]
struct DiagnosticCandidateObservation {
    track_id: usize,
    key: ObservationKey,
    correspondence: GeneralizedCorrespondence2D3D,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct DiagnosticCandidateAudit {
    frame_id: u64,
    full_cross_success_track_ids: Vec<usize>,
    retriangulated_failure_track_ids: Vec<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CrossBoundaryDiagnosticSummary {
    boundary_frame: u64,
    cross_tracks: usize,
    left_pass: usize,
    left_short: usize,
    left_rejected: usize,
    right_pass: usize,
    right_short: usize,
    right_rejected: usize,
    both_sides_two_observations: usize,
    both_sides_dlt_pass: usize,
    candidate_frames: usize,
    pnp_reports: usize,
    distinct_anchor_frames: usize,
    target_support_attempts: usize,
    target_support_successes: usize,
    full_cross_support_attempts: usize,
    full_cross_support_successes: usize,
    existing_support_checks: usize,
    fixed_xyz_gate_failures: usize,
    retriangulated_existing_support_failures: usize,
    cap_skips: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiagnosticConnectivityReport {
    supported_images: BTreeSet<u64>,
    supported_frames: BTreeSet<u64>,
    supported_component_sizes: Vec<usize>,
    removed_tracks: usize,
    removed_observations: usize,
    added_tracks: usize,
    added_observations: usize,
}

#[derive(Debug, Clone)]
struct DiagnosticDsu {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DiagnosticDsu {
    fn new(count: usize) -> Self {
        Self {
            parent: (0..count).collect(),
            size: vec![1; count],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            let root = self.find(self.parent[index]);
            self.parent[index] = root;
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        if self.size[left_root] < self.size[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }
        self.parent[right_root] = left_root;
        self.size[left_root] += self.size[right_root];
    }
}

fn diagnostic_dlt_result(
    track_id: usize,
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> DiagnosticDltResult {
    if keys.len() < MIN_TRACK_OBSERVATIONS {
        return DiagnosticDltResult {
            status: DiagnosticDltStatus::Short {
                observations: keys.len(),
            },
            landmark: None,
        };
    }
    let track = GlobalTrack {
        observations: keys.to_vec(),
    };
    match triangulate_track(track_id, &track, observations, images, cameras) {
        Ok(landmark) => DiagnosticDltResult {
            status: DiagnosticDltStatus::Pass {
                observations: keys.len(),
                mean_error: landmark.mean_error,
                max_error: landmark.max_error,
            },
            landmark: Some(landmark),
        },
        Err(reason) => DiagnosticDltResult {
            status: DiagnosticDltStatus::Rejected {
                observations: keys.len(),
                reason,
            },
            landmark: None,
        },
    }
}

fn diagnostic_dlt_status_label(status: &DiagnosticDltStatus) -> String {
    match status {
        DiagnosticDltStatus::Short { observations } => format!("short({observations})"),
        DiagnosticDltStatus::Pass {
            observations,
            mean_error,
            max_error,
        } => format!("pass(obs={observations},mean={mean_error:.6},max={max_error:.6})"),
        DiagnosticDltStatus::Rejected {
            observations,
            reason,
        } => format!("reject(obs={observations},reason={reason})"),
    }
}

fn sort_diagnostic_keys(
    keys: &mut [ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<(), String> {
    for key in keys.iter().copied() {
        if !images.contains_key(&key.global_image_id) {
            return Err("diagnostic track references unknown image".to_owned());
        }
    }
    // Keep the same ObservationKey ordering used by the production
    // triangulator. In particular, bounded endpoint sampling must not change
    // merely because this report groups keys into left/right frame sets.
    keys.sort_unstable();
    Ok(())
}

fn diagnostic_frame_range(
    keys: &[ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<String, String> {
    let mut frames = keys.iter().map(|key| {
        images
            .get(&key.global_image_id)
            .map(|image| image.atlas.frame_id)
            .ok_or_else(|| "diagnostic track references unknown image".to_owned())
    });
    let Some(first) = frames.next() else {
        return Ok("none".to_owned());
    };
    let first = first?;
    let (minimum, maximum) = frames.try_fold((first, first), |(minimum, maximum), frame| {
        let frame = frame?;
        Ok::<_, String>((minimum.min(frame), maximum.max(frame)))
    })?;
    Ok(format!("{minimum}..{maximum}"))
}

fn collect_diagnostic_cross_tracks(
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    left_max_frame: u64,
) -> Result<Vec<DiagnosticCrossBoundaryTrack>, String> {
    let mut result = Vec::new();
    let mut cross_observations = 0usize;
    for (track_id, track) in store.tracks.iter().enumerate() {
        let mut keys = track.observations.clone();
        sort_diagnostic_keys(&mut keys, images)?;
        let mut left_keys = Vec::new();
        let mut right_keys = Vec::new();
        for key in keys {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "diagnostic track references unknown image".to_owned())?;
            if image.atlas.frame_id <= left_max_frame {
                left_keys.push(key);
            } else {
                right_keys.push(key);
            }
        }
        if !left_keys.is_empty() && !right_keys.is_empty() {
            cross_observations = cross_observations
                .checked_add(left_keys.len() + right_keys.len())
                .ok_or_else(|| "cross-boundary diagnostic observation count overflow".to_owned())?;
            if cross_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            result.push(DiagnosticCrossBoundaryTrack {
                track_id,
                left_keys,
                right_keys,
            });
            if result.len() > DIAGNOSTIC_MAX_CROSS_TRACKS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} tracks",
                    DIAGNOSTIC_MAX_CROSS_TRACKS
                ));
            }
        }
    }
    Ok(result)
}

fn diagnostic_distinct_track_ids(
    observations: &[DiagnosticCandidateObservation],
) -> BTreeSet<usize> {
    observations
        .iter()
        .map(|observation| observation.track_id)
        .collect()
}

fn diagnostic_inlier_track_ids(
    inliers: &[usize],
    observations: &[DiagnosticCandidateObservation],
) -> BTreeSet<usize> {
    inliers
        .iter()
        .filter_map(|index| observations.get(*index))
        .map(|observation| observation.track_id)
        .collect()
}

fn diagnostic_pose_is_finite(pose: &Pose) -> bool {
    pose.world_to_camera
        .translation
        .iter()
        .all(|value| value.is_finite())
        && pose
            .world_to_camera
            .rotation
            .quaternion()
            .coords
            .iter()
            .all(|value| value.is_finite())
}

fn validate_diagnostic_rig_pose(
    manifest: &RigManifest,
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
    rig_pose: &Pose,
) -> Result<usize, String> {
    if !diagnostic_pose_is_finite(rig_pose) {
        return Err("generalized PnP returned a non-finite rig pose".to_owned());
    }
    let mut sensor_count = 0;
    for image in images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
    {
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "diagnostic image references unknown sensor".to_owned())?;
        let sensor_pose = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        if !sensor_pose
            .translation
            .iter()
            .all(|value| value.is_finite())
            || !sensor_pose
                .rotation
                .quaternion()
                .coords
                .iter()
                .all(|value| value.is_finite())
        {
            return Err("generalized PnP sensor pose is non-finite".to_owned());
        }
        let round_trip = sensor.sensor_from_rig.inverse().compose(&sensor_pose);
        let translation_error =
            (round_trip.translation - rig_pose.world_to_camera.translation).norm();
        let rotation_error = round_trip
            .rotation
            .rotation_to(&rig_pose.world_to_camera.rotation)
            .angle();
        if !translation_error.is_finite()
            || !rotation_error.is_finite()
            || translation_error > 1.0e-8
            || rotation_error > 1.0e-8
        {
            return Err(format!(
                "generalized PnP fixed-extrinsic round trip failed for frame {frame_id}"
            ));
        }
        sensor_count += 1;
    }
    if sensor_count == 0 {
        return Err(format!("diagnostic frame {frame_id} has no atlas images"));
    }
    Ok(sensor_count)
}

fn diagnostic_landmark_frame_index(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    candidate_frames: &BTreeSet<u64>,
) -> Result<BTreeMap<u64, Vec<usize>>, String> {
    let mut indexed = BTreeMap::<u64, BTreeSet<usize>>::new();
    for (landmark_id, landmark) in landmarks.iter().enumerate() {
        let mut seen_frames = BTreeSet::new();
        for key in &landmark.observations {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "baseline landmark references unknown image".to_owned())?;
            if candidate_frames.contains(&image.atlas.frame_id)
                && seen_frames.insert(image.atlas.frame_id)
            {
                indexed
                    .entry(image.atlas.frame_id)
                    .or_default()
                    .insert(landmark_id);
            }
        }
    }
    Ok(indexed
        .into_iter()
        .map(|(frame, ids)| (frame, ids.into_iter().collect()))
        .collect())
}

fn diagnostic_track_with_keys(
    left_keys: &[ObservationKey],
    right_keys: &[ObservationKey],
) -> GlobalTrack {
    let mut keys = left_keys.to_vec();
    keys.extend_from_slice(right_keys);
    keys.sort_unstable();
    keys.dedup();
    GlobalTrack { observations: keys }
}

fn diagnostic_frame_ids(images: &BTreeMap<u64, GlobalImage>) -> Vec<u64> {
    images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn diagnostic_image_ids(images: &BTreeMap<u64, GlobalImage>) -> BTreeSet<u64> {
    images.keys().copied().collect()
}

fn diagnostic_add_track_to_dsu(
    keys: &[ObservationKey],
    images: &BTreeMap<u64, GlobalImage>,
    frame_indices: &BTreeMap<u64, usize>,
    dsu: &mut DiagnosticDsu,
    supported_images: &mut BTreeSet<u64>,
    supported_frames: &mut BTreeSet<u64>,
) -> Result<(), String> {
    let mut track_frames = BTreeSet::new();
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "connectivity track references unknown image".to_owned())?;
        let frame_id = image.atlas.frame_id;
        let frame_index = *frame_indices
            .get(&frame_id)
            .ok_or_else(|| "connectivity frame index is missing".to_owned())?;
        supported_images.insert(key.global_image_id);
        supported_frames.insert(frame_id);
        track_frames.insert(frame_index);
    }
    let mut indices = track_frames.into_iter();
    if let Some(first) = indices.next() {
        for index in indices {
            dsu.union(first, index);
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupportConnectivity {
    supported_images: BTreeSet<u64>,
    supported_frames: BTreeSet<u64>,
    frame_components: BTreeMap<u64, usize>,
    component_count: usize,
}

fn support_connectivity_for_landmarks(
    baseline_landmarks: &[LandmarkOutput],
    replacements: &BTreeMap<usize, LandmarkOutput>,
    removed_track_ids: &BTreeSet<usize>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<SupportConnectivity, String> {
    let frame_ids = diagnostic_frame_ids(images);
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported_images = BTreeSet::new();
    let mut supported_frames = BTreeSet::new();
    for (index, baseline) in baseline_landmarks.iter().enumerate() {
        if removed_track_ids.contains(&baseline.track_id) {
            continue;
        }
        let keys = replacements
            .get(&index)
            .map(|landmark| landmark.observations.as_slice())
            .unwrap_or(&baseline.observations);
        diagnostic_add_track_to_dsu(
            keys,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }
    let mut root_to_component = BTreeMap::<usize, usize>::new();
    let mut frame_components = BTreeMap::new();
    for frame_id in &supported_frames {
        let frame_index = *frame_indices
            .get(frame_id)
            .ok_or_else(|| "supported connectivity frame index is missing".to_owned())?;
        let root = dsu.find(frame_index);
        let next = root_to_component.len();
        let component = *root_to_component.entry(root).or_insert(next);
        frame_components.insert(*frame_id, component);
    }
    Ok(SupportConnectivity {
        supported_images,
        supported_frames,
        component_count: root_to_component.len(),
        frame_components,
    })
}

fn connectivity_not_split(baseline: &SupportConnectivity, candidate: &SupportConnectivity) -> bool {
    if baseline.supported_images != candidate.supported_images
        || baseline.supported_frames != candidate.supported_frames
        || candidate.component_count > baseline.component_count
    {
        return false;
    }
    let mut component_map = BTreeMap::<usize, usize>::new();
    for frame_id in &baseline.supported_frames {
        let Some(&baseline_component) = baseline.frame_components.get(frame_id) else {
            return false;
        };
        let Some(&candidate_component) = candidate.frame_components.get(frame_id) else {
            return false;
        };
        if let Some(previous) = component_map.insert(baseline_component, candidate_component) {
            if previous != candidate_component {
                return false;
            }
        }
    }
    true
}

fn diagnostic_connectivity_report(
    baseline_landmarks: &[LandmarkOutput],
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    removed_track_ids: &BTreeSet<usize>,
    added_track_ids: &BTreeSet<usize>,
) -> Result<DiagnosticConnectivityReport, String> {
    let frame_ids = diagnostic_frame_ids(images);
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let actually_removed = removed_track_ids
        .difference(added_track_ids)
        .copied()
        .collect::<BTreeSet<_>>();

    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported_images = BTreeSet::new();
    let mut supported_frames = BTreeSet::new();
    let mut removed_observations = 0;
    let mut removed_seen = BTreeSet::new();
    let mut added_existing_seen = BTreeSet::new();
    for landmark in baseline_landmarks {
        if actually_removed.contains(&landmark.track_id) {
            removed_observations += landmark.observations.len();
            removed_seen.insert(landmark.track_id);
            continue;
        }
        if added_track_ids.contains(&landmark.track_id) {
            added_existing_seen.insert(landmark.track_id);
        }
        diagnostic_add_track_to_dsu(
            &landmark.observations,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }

    if let Some(track_id) = actually_removed.difference(&removed_seen).next() {
        return Err(format!(
            "connectivity removal references non-baseline track {track_id}"
        ));
    }

    let mut added_observations = 0;
    let mut added_new = 0;
    for track_id in added_track_ids.difference(&added_existing_seen) {
        let track = store
            .tracks
            .get(*track_id)
            .ok_or_else(|| format!("connectivity addition references unknown track {track_id}"))?;
        added_new += 1;
        added_observations += track.observations.len();
        diagnostic_add_track_to_dsu(
            &track.observations,
            images,
            &frame_indices,
            &mut dsu,
            &mut supported_images,
            &mut supported_frames,
        )?;
    }

    let mut component_counts = BTreeMap::<usize, usize>::new();
    for frame_id in &supported_frames {
        let frame_index = *frame_indices
            .get(frame_id)
            .ok_or_else(|| "supported connectivity frame index is missing".to_owned())?;
        let root = dsu.find(frame_index);
        *component_counts.entry(root).or_default() += 1;
    }
    let mut supported_component_sizes = component_counts.into_values().collect::<Vec<_>>();
    supported_component_sizes.sort_unstable();
    Ok(DiagnosticConnectivityReport {
        supported_images,
        supported_frames,
        supported_component_sizes,
        removed_tracks: actually_removed.len(),
        removed_observations,
        added_tracks: added_new,
        added_observations,
    })
}

fn diagnostic_format_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn diagnostic_format_track_ids(ids: &[usize]) -> String {
    ids.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn diagnostic_emit_connectivity(
    baseline_landmarks: &[LandmarkOutput],
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    audits: &[DiagnosticCandidateAudit],
) -> Result<(), String> {
    let all_images = diagnostic_image_ids(images);
    let all_frames = diagnostic_frame_ids(images)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let empty = BTreeSet::new();
    let baseline =
        diagnostic_connectivity_report(baseline_landmarks, store, images, &empty, &empty)?;
    let baseline_unsupported_images = all_images
        .difference(&baseline.supported_images)
        .copied()
        .collect::<BTreeSet<_>>();
    let baseline_unsupported_frames = all_frames
        .difference(&baseline.supported_frames)
        .copied()
        .collect::<BTreeSet<_>>();
    println!(
        "cross_boundary_connectivity_baseline supported_images={} unsupported_images={} supported_frames={} unsupported_frames={} supported_component_count={} supported_component_sizes={}",
        baseline.supported_images.len(),
        baseline_unsupported_images.len(),
        baseline.supported_frames.len(),
        baseline_unsupported_frames.len(),
        baseline.supported_component_sizes.len(),
        diagnostic_format_usize_list(&baseline.supported_component_sizes),
    );

    for audit in audits {
        let removed = audit
            .retriangulated_failure_track_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let added = audit
            .full_cross_success_track_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let candidate =
            diagnostic_connectivity_report(baseline_landmarks, store, images, &removed, &added)?;
        let candidate_unsupported_images = all_images
            .difference(&candidate.supported_images)
            .copied()
            .collect::<BTreeSet<_>>();
        let candidate_unsupported_frames = all_frames
            .difference(&candidate.supported_frames)
            .copied()
            .collect::<BTreeSet<_>>();
        let existing_unsupported_images = baseline_unsupported_images
            .intersection(&candidate_unsupported_images)
            .count();
        let existing_unsupported_frames = baseline_unsupported_frames
            .intersection(&candidate_unsupported_frames)
            .count();
        let recovered_existing_images = baseline_unsupported_images
            .difference(&candidate_unsupported_images)
            .count();
        let recovered_existing_frames = baseline_unsupported_frames
            .difference(&candidate_unsupported_frames)
            .count();
        let new_unsupported_images = candidate_unsupported_images
            .difference(&baseline_unsupported_images)
            .count();
        let new_unsupported_frames = candidate_unsupported_frames
            .difference(&baseline_unsupported_frames)
            .count();
        println!(
            "cross_boundary_connectivity frame={} full_cross_success_tracks={} full_cross_success_ids={} retriangulated_failure_tracks={} retriangulated_failure_ids={} removed_tracks={} removed_observations={} added_tracks={} added_observations={} supported_images={} unsupported_images={} supported_frames={} unsupported_frames={} existing_unsupported_images={} existing_unsupported_frames={} recovered_existing_images={} recovered_existing_frames={} new_unsupported_images={} new_unsupported_frames={} supported_component_count={} supported_component_sizes={}",
            audit.frame_id,
            audit.full_cross_success_track_ids.len(),
            diagnostic_format_track_ids(&audit.full_cross_success_track_ids),
            audit.retriangulated_failure_track_ids.len(),
            diagnostic_format_track_ids(&audit.retriangulated_failure_track_ids),
            candidate.removed_tracks,
            candidate.removed_observations,
            candidate.added_tracks,
            candidate.added_observations,
            candidate.supported_images.len(),
            candidate_unsupported_images.len(),
            candidate.supported_frames.len(),
            candidate_unsupported_frames.len(),
            existing_unsupported_images,
            existing_unsupported_frames,
            recovered_existing_images,
            recovered_existing_frames,
            new_unsupported_images,
            new_unsupported_frames,
            candidate.supported_component_sizes.len(),
            diagnostic_format_usize_list(&candidate.supported_component_sizes),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BoundaryRepairSummary {
    status: String,
    candidates: usize,
    attempted: usize,
    accepted_frame: Option<u64>,
    added_tracks: usize,
    added_observations: usize,
    removed_tracks: usize,
    removed_observations: usize,
    baseline_component_count: usize,
    candidate_component_count: usize,
}

fn boundary_repair_candidate_is_accepted(
    baseline: &DiagnosticConnectivityReport,
    candidate: &DiagnosticConnectivityReport,
    added_tracks: usize,
) -> bool {
    added_tracks > 0
        && candidate.supported_component_sizes.len() < baseline.supported_component_sizes.len()
        && baseline
            .supported_images
            .is_subset(&candidate.supported_images)
        && baseline
            .supported_frames
            .is_subset(&candidate.supported_frames)
}

fn landmark_output_gate_valid(landmark: &LandmarkOutput) -> bool {
    landmark.observations.len() >= MIN_TRACK_OBSERVATIONS
        && landmark
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        && landmark.errors.len() == landmark.observations.len()
        && landmark.errors.iter().all(|error| error.is_finite())
        && landmark.mean_error.is_finite()
        && landmark.rms_error.is_finite()
        && landmark.max_error.is_finite()
        && landmark.mean_error <= MAX_MEAN_REPROJECTION_PX
        && landmark.max_error <= MAX_REPROJECTION_PX
}

/// Try one deterministic, transactional cross-boundary repair.
///
/// This is intentionally separate from `diagnose_cross_boundary_pnp`: the
/// diagnostic path never mutates a model, while this opt-in path stages one
/// candidate and commits only the first candidate that improves supported
/// frame connectivity without losing any existing support.  Existing
/// landmark observations, rather than the source TrackStore, are authoritative
/// for affected-track re-triangulation.
fn repair_cross_boundary(
    manifest: &RigManifest,
    left_max_frame: u64,
    store: &TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<BoundaryRepairSummary, String> {
    let baseline_connectivity = diagnostic_connectivity_report(
        landmarks,
        store,
        images,
        &BTreeSet::new(),
        &BTreeSet::new(),
    )?;
    let cross_tracks = collect_diagnostic_cross_tracks(store, images, left_max_frame)?;
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let mut cross_by_id = BTreeMap::<usize, DiagnosticCrossBoundaryTrack>::new();
    let mut candidates_by_frame = BTreeMap::<u64, Vec<DiagnosticCandidateObservation>>::new();
    let mut candidate_observations = 0usize;
    for cross in cross_tracks {
        let left = diagnostic_dlt_result(
            cross.track_id,
            &cross.left_keys,
            &store.observations,
            images,
            cameras,
        );
        let Some(anchor) = left.landmark else {
            continue;
        };
        cross_by_id.insert(cross.track_id, cross.clone());
        for key in &cross.right_keys {
            candidate_observations = candidate_observations
                .checked_add(1)
                .ok_or_else(|| "boundary repair observation count overflow".to_owned())?;
            if candidate_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "boundary repair exceeds {} candidate observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "boundary repair references unknown image".to_owned())?;
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "boundary repair references unknown observation".to_owned())?;
            let frame_candidates = candidates_by_frame.entry(image.atlas.frame_id).or_default();
            frame_candidates.push(DiagnosticCandidateObservation {
                track_id: cross.track_id,
                key: *key,
                correspondence: GeneralizedCorrespondence2D3D {
                    sensor_index: image.atlas.sensor_index,
                    point2d: observation.xy,
                    point3d: anchor.position,
                    confidence: None,
                },
            });
            if candidates_by_frame.len() > DIAGNOSTIC_MAX_CANDIDATE_FRAMES {
                return Err(format!(
                    "boundary repair exceeds {} candidate frames",
                    DIAGNOSTIC_MAX_CANDIDATE_FRAMES
                ));
            }
        }
    }
    for observations in candidates_by_frame.values_mut() {
        observations.sort_by_key(|observation| (observation.track_id, observation.key));
    }

    let mut summary = BoundaryRepairSummary {
        status: "no-accepted-candidate".to_owned(),
        candidates: candidates_by_frame.len(),
        attempted: 0,
        accepted_frame: None,
        added_tracks: 0,
        added_observations: 0,
        removed_tracks: 0,
        removed_observations: 0,
        baseline_component_count: baseline_connectivity.supported_component_sizes.len(),
        candidate_component_count: baseline_connectivity.supported_component_sizes.len(),
    };

    for (frame_id, frame_candidates) in candidates_by_frame {
        summary.attempted += 1;
        let distinct_candidates = diagnostic_distinct_track_ids(&frame_candidates);
        if frame_candidates.len() > DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=correspondence-cap correspondences={} distinct_tracks={}",
                frame_candidates.len(),
                distinct_candidates.len(),
            );
            continue;
        }
        if distinct_candidates.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=insufficient-distinct-candidates correspondences={} distinct_tracks={} required={}",
                frame_candidates.len(),
                distinct_candidates.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        let correspondences = frame_candidates
            .iter()
            .map(|observation| observation.correspondence.clone())
            .collect::<Vec<_>>();
        let Some(report) = pnp.estimate(&rig, &correspondences) else {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=pnp-failed correspondences={} distinct_tracks={}",
                correspondences.len(),
                distinct_candidates.len(),
            );
            continue;
        };
        let distinct_inliers = diagnostic_inlier_track_ids(&report.inliers, &frame_candidates);
        if distinct_inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=insufficient-distinct-inliers pnp_inliers={} distinct_inliers={} required={}",
                report.inliers.len(),
                distinct_inliers.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        // Only candidate ids are retained here.  Scanning the baseline once
        // avoids a second model-sized track-id map for every candidate.
        let baseline_candidate_track_ids = landmarks
            .iter()
            .filter(|landmark| distinct_inliers.contains(&landmark.track_id))
            .map(|landmark| landmark.track_id)
            .collect::<BTreeSet<_>>();
        let mut pose_overrides = BTreeMap::new();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut pose_overrides,
        )?;
        validate_diagnostic_rig_pose(manifest, frame_id, images, &report.pose)?;

        let mut added_landmarks = Vec::new();
        let mut added_ids = BTreeSet::new();
        let mut added_keys = BTreeSet::new();
        let mut full_cross_failure_count = 0usize;
        let mut full_cross_failure_observations = 0usize;
        for track_id in &distinct_inliers {
            let cross = cross_by_id
                .get(track_id)
                .ok_or_else(|| format!("boundary repair lost cross track {track_id}"))?;
            let full_track = diagnostic_track_with_keys(&cross.left_keys, &cross.right_keys);
            let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &full_track,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            ) else {
                full_cross_failure_count += 1;
                full_cross_failure_observations += full_track.observations.len();
                continue;
            };
            if baseline_candidate_track_ids.contains(track_id) || !added_ids.insert(*track_id) {
                continue;
            }
            if landmark
                .observations
                .iter()
                .any(|key| !added_keys.insert(*key))
            {
                println!(
                    "boundary_repair_candidate frame={frame_id} status=rejected reason=added-observation-conflict",
                );
                added_landmarks.clear();
                added_ids.clear();
                break;
            }
            added_landmarks.push(landmark);
        }
        if added_landmarks.is_empty() {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=no-full-cross-addition full_cross_failures={} full_cross_failure_observations={}",
                full_cross_failure_count,
                full_cross_failure_observations,
            );
            continue;
        }

        let mut affected = BTreeMap::<usize, LandmarkOutput>::new();
        let mut removed_ids = BTreeSet::new();
        let mut removed_observations = 0usize;
        let mut affected_count = 0usize;
        let mut affected_observations = 0usize;
        let mut affected_cap_exceeded = false;
        for (index, landmark) in landmarks.iter().enumerate() {
            let touches_candidate = landmark.observations.iter().any(|key| {
                images
                    .get(&key.global_image_id)
                    .is_some_and(|image| image.atlas.frame_id == frame_id)
            });
            if !touches_candidate {
                continue;
            }
            affected_count += 1;
            affected_observations = affected_observations
                .checked_add(landmark.observations.len())
                .ok_or_else(|| "boundary repair affected observation count overflow".to_owned())?;
            if affected_count > JOINT_BA_MAX_LANDMARKS
                || affected_observations > JOINT_BA_MAX_OBSERVATIONS
            {
                println!(
                    "boundary_repair_candidate frame={frame_id} status=rejected reason=affected-track-cap affected_landmarks={} affected_observations={} landmark_cap={} observation_cap={}",
                    affected_count,
                    affected_observations,
                    JOINT_BA_MAX_LANDMARKS,
                    JOINT_BA_MAX_OBSERVATIONS,
                );
                affected.clear();
                removed_ids.clear();
                affected_cap_exceeded = true;
                break;
            }
            let original_track = GlobalTrack {
                observations: landmark.observations.clone(),
            };
            match triangulate_track_with_pose_overrides(
                landmark.track_id,
                &original_track,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            ) {
                Ok(updated) => {
                    affected.insert(index, updated);
                }
                Err(_) => {
                    removed_ids.insert(landmark.track_id);
                    removed_observations += landmark.observations.len();
                }
            }
        }
        if affected_cap_exceeded {
            continue;
        }
        let retained_key_conflict = landmarks.iter().any(|landmark| {
            !removed_ids.contains(&landmark.track_id)
                && landmark
                    .observations
                    .iter()
                    .any(|key| added_keys.contains(key))
        });
        if retained_key_conflict {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=retained-observation-conflict",
            );
            continue;
        }

        let candidate_connectivity =
            diagnostic_connectivity_report(landmarks, store, images, &removed_ids, &added_ids)?;
        if !boundary_repair_candidate_is_accepted(
            &baseline_connectivity,
            &candidate_connectivity,
            added_landmarks.len(),
        ) {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=support-or-connectivity-gate added_tracks={} removed_tracks={} baseline_components={} candidate_components={} baseline_supported_frames={} candidate_supported_frames={}",
                added_landmarks.len(),
                removed_ids.len(),
                baseline_connectivity.supported_component_sizes.len(),
                candidate_connectivity.supported_component_sizes.len(),
                baseline_connectivity.supported_frames.len(),
                candidate_connectivity.supported_frames.len(),
            );
            continue;
        }

        let mut all_remaining_valid = true;
        for (index, landmark) in landmarks.iter().enumerate() {
            if removed_ids.contains(&landmark.track_id) {
                continue;
            }
            let candidate = affected.get(&index).unwrap_or(landmark);
            if !landmark_output_gate_valid(candidate) {
                all_remaining_valid = false;
                break;
            }
            let metrics = evaluate_landmark_metrics(
                candidate,
                &store.observations,
                images,
                cameras,
                &pose_overrides,
            );
            if metrics.is_err()
                || metrics.as_ref().is_ok_and(|metrics| {
                    metrics.mean_error > MAX_MEAN_REPROJECTION_PX
                        || metrics.max_error > MAX_REPROJECTION_PX
                })
            {
                all_remaining_valid = false;
                break;
            }
        }
        if all_remaining_valid {
            for landmark in &added_landmarks {
                if !landmark_output_gate_valid(landmark) {
                    all_remaining_valid = false;
                    break;
                }
            }
        }
        if !all_remaining_valid {
            println!(
                "boundary_repair_candidate frame={frame_id} status=rejected reason=remaining-landmark-gate removed_tracks={} removed_observations={}",
                removed_ids.len(),
                removed_observations,
            );
            continue;
        }

        let accepted_added_tracks = added_landmarks.len();
        let accepted_added_observations = added_landmarks
            .iter()
            .map(|landmark| landmark.observations.len())
            .sum::<usize>();
        apply_pose_overrides(images, &pose_overrides)?;
        for (index, replacement) in affected {
            landmarks[index] = replacement;
        }
        landmarks.retain(|landmark| !removed_ids.contains(&landmark.track_id));
        landmarks.extend(added_landmarks);
        summary.status = "accepted".to_owned();
        summary.accepted_frame = Some(frame_id);
        summary.added_tracks = accepted_added_tracks;
        summary.added_observations = accepted_added_observations;
        summary.removed_tracks = removed_ids.len();
        summary.removed_observations = removed_observations;
        summary.candidate_component_count = candidate_connectivity.supported_component_sizes.len();
        println!(
            "boundary_repair_candidate frame={frame_id} status=accepted pnp_inliers={} distinct_inliers={} pnp_mean_px={:.6} pnp_max_px={:.6} added_tracks={} added_observations={} removed_tracks={} removed_observations={} supported_component_sizes_before={} supported_component_sizes_after={}",
            report.inliers.len(),
            distinct_inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            summary.added_tracks,
            summary.added_observations,
            summary.removed_tracks,
            summary.removed_observations,
            diagnostic_format_usize_list(&baseline_connectivity.supported_component_sizes),
            diagnostic_format_usize_list(&candidate_connectivity.supported_component_sizes),
        );
        break;
    }
    Ok(summary)
}

#[cfg(test)]
fn diagnostic_component_sizes_from_frame_tracks(
    frame_ids: &[u64],
    track_frames: &[Vec<u64>],
) -> (Vec<usize>, BTreeSet<u64>) {
    let frame_indices = frame_ids
        .iter()
        .enumerate()
        .map(|(index, frame_id)| (*frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let mut dsu = DiagnosticDsu::new(frame_ids.len());
    let mut supported = BTreeSet::new();
    for track in track_frames {
        let mut indices = track
            .iter()
            .filter_map(|frame_id| {
                supported.insert(*frame_id);
                frame_indices.get(frame_id).copied()
            })
            .collect::<BTreeSet<_>>()
            .into_iter();
        if let Some(first) = indices.next() {
            for index in indices {
                dsu.union(first, index);
            }
        }
    }
    let mut counts = BTreeMap::<usize, usize>::new();
    for frame_id in &supported {
        if let Some(index) = frame_indices.get(frame_id) {
            *counts.entry(dsu.find(*index)).or_default() += 1;
        }
    }
    let mut sizes = counts.into_values().collect::<Vec<_>>();
    sizes.sort_unstable();
    (sizes, supported)
}

/// Run the strict, non-mutating cross-boundary generalized-PnP experiment.
///
/// Only tracks whose left-side DLT passes are used as 3D anchors. Each right
/// frame is a separate candidate; there is no pose propagation or whole-gauge
/// transform. The returned pose is used only for diagnostic re-triangulation.
fn diagnose_cross_boundary_pnp(
    manifest: &RigManifest,
    left_max_frame: u64,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    baseline_landmarks: &[LandmarkOutput],
) -> Result<CrossBoundaryDiagnosticSummary, String> {
    let cross_tracks = collect_diagnostic_cross_tracks(store, images, left_max_frame)?;
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let mut summary = CrossBoundaryDiagnosticSummary {
        boundary_frame: left_max_frame,
        cross_tracks: cross_tracks.len(),
        ..CrossBoundaryDiagnosticSummary::default()
    };
    let mut evaluated = Vec::with_capacity(cross_tracks.len());
    for cross in cross_tracks {
        let left = diagnostic_dlt_result(
            cross.track_id,
            &cross.left_keys,
            &store.observations,
            images,
            cameras,
        );
        let right = diagnostic_dlt_result(
            cross.track_id,
            &cross.right_keys,
            &store.observations,
            images,
            cameras,
        );
        match &left.status {
            DiagnosticDltStatus::Pass { .. } => summary.left_pass += 1,
            DiagnosticDltStatus::Short { .. } => summary.left_short += 1,
            DiagnosticDltStatus::Rejected { .. } => summary.left_rejected += 1,
        }
        match &right.status {
            DiagnosticDltStatus::Pass { .. } => summary.right_pass += 1,
            DiagnosticDltStatus::Short { .. } => summary.right_short += 1,
            DiagnosticDltStatus::Rejected { .. } => summary.right_rejected += 1,
        }
        if cross.left_keys.len() >= MIN_TRACK_OBSERVATIONS
            && cross.right_keys.len() >= MIN_TRACK_OBSERVATIONS
        {
            summary.both_sides_two_observations += 1;
        }
        if matches!(&left.status, DiagnosticDltStatus::Pass { .. })
            && matches!(&right.status, DiagnosticDltStatus::Pass { .. })
        {
            summary.both_sides_dlt_pass += 1;
        }
        let left_range = diagnostic_frame_range(&cross.left_keys, images)?;
        let right_range = diagnostic_frame_range(&cross.right_keys, images)?;
        println!(
            "cross_boundary_track track_id={} left_obs={} left_frames={} right_obs={} right_frames={} left_dlt={} right_dlt={}",
            cross.track_id,
            cross.left_keys.len(),
            left_range,
            cross.right_keys.len(),
            right_range,
            diagnostic_dlt_status_label(&left.status),
            diagnostic_dlt_status_label(&right.status),
        );
        evaluated.push((cross, left, right));
    }

    let mut candidates_by_frame = BTreeMap::<u64, Vec<DiagnosticCandidateObservation>>::new();
    let mut candidate_observations = 0usize;
    for (cross, left, _) in &evaluated {
        let Some(anchor) = left.landmark.as_ref() else {
            continue;
        };
        for key in &cross.right_keys {
            candidate_observations = candidate_observations
                .checked_add(1)
                .ok_or_else(|| "diagnostic candidate observation count overflow".to_owned())?;
            if candidate_observations > DIAGNOSTIC_MAX_CROSS_OBSERVATIONS {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} candidate observations",
                    DIAGNOSTIC_MAX_CROSS_OBSERVATIONS
                ));
            }
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                "diagnostic right observation references unknown image".to_owned()
            })?;
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "diagnostic right observation is missing".to_owned())?;
            let frame_candidates = candidates_by_frame.entry(image.atlas.frame_id).or_default();
            frame_candidates.push(DiagnosticCandidateObservation {
                track_id: cross.track_id,
                key: *key,
                correspondence: GeneralizedCorrespondence2D3D {
                    sensor_index: image.atlas.sensor_index,
                    point2d: observation.xy,
                    point3d: anchor.position,
                    confidence: None,
                },
            });
            if candidates_by_frame.len() > DIAGNOSTIC_MAX_CANDIDATE_FRAMES {
                return Err(format!(
                    "cross-boundary diagnostic exceeds {} candidate frames",
                    DIAGNOSTIC_MAX_CANDIDATE_FRAMES
                ));
            }
        }
    }
    for observations in candidates_by_frame.values_mut() {
        observations.sort_by_key(|observation| (observation.track_id, observation.key));
    }
    summary.candidate_frames = candidates_by_frame.len();
    let candidate_frames = candidates_by_frame.keys().copied().collect::<BTreeSet<_>>();
    let baseline_by_frame =
        diagnostic_landmark_frame_index(baseline_landmarks, images, &candidate_frames)?;
    let cross_by_id = evaluated
        .iter()
        .map(|(cross, _, _)| (cross.track_id, cross))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_audits = Vec::new();

    for (frame_id, frame_candidates) in candidates_by_frame {
        let distinct_candidates = diagnostic_distinct_track_ids(&frame_candidates);
        if frame_candidates.len() > DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME {
            summary.cap_skips += 1;
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=cap-exceeded correspondences={} distinct_tracks={} cap={}",
                frame_candidates.len(),
                distinct_candidates.len(),
                DIAGNOSTIC_MAX_CORRESPONDENCES_PER_FRAME,
            );
            continue;
        }
        if distinct_candidates.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=insufficient-distinct-candidates correspondences={} distinct_tracks={} required={}",
                frame_candidates.len(),
                distinct_candidates.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        let correspondences = frame_candidates
            .iter()
            .map(|observation| observation.correspondence.clone())
            .collect::<Vec<_>>();
        let Some(report) = pnp.estimate(&rig, &correspondences) else {
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=pnp-failed correspondences={} distinct_tracks={}",
                correspondences.len(),
                distinct_candidates.len(),
            );
            continue;
        };
        summary.pnp_reports += 1;
        let distinct_inliers = diagnostic_inlier_track_ids(&report.inliers, &frame_candidates);
        if distinct_inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            println!(
                "cross_boundary_pnp_frame frame={frame_id} status=insufficient-distinct-inliers correspondences={} candidate_tracks={} inliers={} distinct_inliers={} required={}",
                correspondences.len(),
                distinct_candidates.len(),
                report.inliers.len(),
                distinct_inliers.len(),
                RECOVERY_MIN_PNP_INLIERS,
            );
            continue;
        }
        let mut candidate_pose_overrides = BTreeMap::new();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut candidate_pose_overrides,
        )?;
        let sensor_count = validate_diagnostic_rig_pose(manifest, frame_id, images, &report.pose)?;
        summary.distinct_anchor_frames += 1;

        let mut target_support_tracks = 0;
        let mut full_cross_support_tracks = 0;
        let mut target_support_observations = 0;
        let mut full_cross_support_observations = 0;
        let mut candidate_audit = DiagnosticCandidateAudit {
            frame_id,
            ..DiagnosticCandidateAudit::default()
        };
        // Only RANSAC-distinct inlier tracks are eligible for the support
        // experiment. Outlier candidate tracks remain visible in the
        // candidate_tracks field but cannot be presented as recovered support.
        for track_id in &distinct_inliers {
            let Some(cross) = cross_by_id.get(track_id) else {
                return Err(format!("missing diagnostic cross track {track_id}"));
            };
            let target_keys = frame_candidates
                .iter()
                .filter(|observation| observation.track_id == *track_id)
                .map(|observation| observation.key)
                .collect::<Vec<_>>();
            let target_track = diagnostic_track_with_keys(&cross.left_keys, &target_keys);
            if let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &target_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                target_support_tracks += 1;
                target_support_observations += landmark.observations.len();
            }
            let full_track = diagnostic_track_with_keys(&cross.left_keys, &cross.right_keys);
            if let Ok(landmark) = triangulate_track_with_pose_overrides(
                *track_id,
                &full_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                full_cross_support_tracks += 1;
                full_cross_support_observations += landmark.observations.len();
                candidate_audit.full_cross_success_track_ids.push(*track_id);
            }
        }
        summary.target_support_attempts += distinct_inliers.len();
        summary.target_support_successes += target_support_tracks;
        summary.full_cross_support_attempts += distinct_inliers.len();
        summary.full_cross_support_successes += full_cross_support_tracks;

        let mut fixed_xyz_gate_failures = 0;
        let mut retriangulated_existing_support_failures = 0;
        let affected_landmarks = baseline_by_frame
            .get(&frame_id)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for landmark_id in affected_landmarks {
            let landmark = baseline_landmarks
                .get(*landmark_id)
                .ok_or_else(|| "diagnostic baseline landmark index is invalid".to_owned())?;
            let fixed_xyz_failed = match evaluate_landmark_metrics(
                landmark,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) {
                Ok(metrics) => {
                    metrics.mean_error > MAX_MEAN_REPROJECTION_PX
                        || metrics.max_error > MAX_REPROJECTION_PX
                }
                Err(_) => true,
            };
            if fixed_xyz_failed {
                fixed_xyz_gate_failures += 1;
            }
            let retriangulated_failed = store
                .tracks
                .get(landmark.track_id)
                .map(|track| {
                    triangulate_track_with_pose_overrides(
                        landmark.track_id,
                        track,
                        &store.observations,
                        images,
                        cameras,
                        &candidate_pose_overrides,
                    )
                    .is_err()
                })
                .unwrap_or(true);
            if retriangulated_failed {
                retriangulated_existing_support_failures += 1;
                candidate_audit
                    .retriangulated_failure_track_ids
                    .push(landmark.track_id);
            }
        }
        summary.fixed_xyz_gate_failures += fixed_xyz_gate_failures;
        summary.existing_support_checks += affected_landmarks.len();
        summary.retriangulated_existing_support_failures +=
            retriangulated_existing_support_failures;
        println!(
            "cross_boundary_pnp_frame frame={frame_id} status=anchor-success correspondences={} candidate_tracks={} inliers={} distinct_inliers={} mean_px={:.6} max_px={:.6} sensors={} target_support_attempts={} target_support_successes={} target_support_observations={} full_cross_support_attempts={} full_cross_support_successes={} full_cross_support_observations={} affected_landmarks={} fixed_xyz_gate_failures={} retriangulated_existing_support_failures={}",
            correspondences.len(),
            distinct_candidates.len(),
            report.inliers.len(),
            distinct_inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            sensor_count,
            distinct_inliers.len(),
            target_support_tracks,
            target_support_observations,
            distinct_inliers.len(),
            full_cross_support_tracks,
            full_cross_support_observations,
            affected_landmarks.len(),
            fixed_xyz_gate_failures,
            retriangulated_existing_support_failures,
        );
        candidate_audits.push(candidate_audit);
    }
    diagnostic_emit_connectivity(baseline_landmarks, store, images, &candidate_audits)?;
    println!(
        "cross_boundary_pnp_summary boundary_frame={} cross_tracks={} left_pass={} left_short={} left_rejected={} right_pass={} right_short={} right_rejected={} both_sides_two_observations={} both_sides_dlt_pass={} candidate_frames={} pnp_reports={} distinct_anchor_frames={} target_support_attempts={} target_support_successes={} full_cross_support_attempts={} full_cross_support_successes={} existing_support_checks={} fixed_xyz_gate_failures={} retriangulated_existing_support_failures={} cap_skips={} default_model_unchanged=true",
        summary.boundary_frame,
        summary.cross_tracks,
        summary.left_pass,
        summary.left_short,
        summary.left_rejected,
        summary.right_pass,
        summary.right_short,
        summary.right_rejected,
        summary.both_sides_two_observations,
        summary.both_sides_dlt_pass,
        summary.candidate_frames,
        summary.pnp_reports,
        summary.distinct_anchor_frames,
        summary.target_support_attempts,
        summary.target_support_successes,
        summary.full_cross_support_attempts,
        summary.full_cross_support_successes,
        summary.existing_support_checks,
        summary.fixed_xyz_gate_failures,
        summary.retriangulated_existing_support_failures,
        summary.cap_skips,
    );
    Ok(summary)
}

#[derive(Debug, Clone)]
struct RecoveryResult {
    pose_overrides: BTreeMap<u64, Pose>,
    landmarks: Vec<LandmarkOutput>,
    summary: RecoverySummary,
}

#[derive(Debug, Clone)]
struct RecoveryAnchor {
    track_id: usize,
    correspondence_indices: Vec<usize>,
}

/// Recover initially unsupported rig frames from rejected tracks without
/// letting one unsupported frame bootstrap another.  Anchor points are
/// triangulated only from the frame set supported by the fixed-pose model;
/// the target frame is then estimated with the existing deterministic
/// generalized-PnP implementation and fixed sensor extrinsics.  Everything
/// for one frame is staged and committed only after the six-inlier and
/// six-observation gates pass.
fn recover_zero_support_frames(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    baseline_landmarks: &[LandmarkOutput],
) -> Result<RecoveryResult, String> {
    let initial_supported_frames = baseline_landmarks
        .iter()
        .flat_map(|landmark| landmark.observations.iter())
        .filter_map(|key| images.get(&key.global_image_id))
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>();
    let atlas_frames = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<BTreeSet<_>>();
    let zero_support_frames = atlas_frames
        .difference(&initial_supported_frames)
        .copied()
        .collect::<Vec<_>>();
    let mut summary = RecoverySummary {
        frames_considered: zero_support_frames.len(),
        ..RecoverySummary::default()
    };
    let rig = build_generalized_rig(manifest, cameras)?;
    let pnp = GeneralizedPnPRansac {
        iterations: RECOVERY_PNP_ITERATIONS,
        reprojection_threshold: RECOVERY_PNP_REPROJECTION_PX,
        seed: RECOVERY_PNP_SEED,
        ..GeneralizedPnPRansac::default()
    };
    let baseline_keys = baseline_landmarks
        .iter()
        .flat_map(|landmark| landmark.observations.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut used_track_ids = baseline_landmarks
        .iter()
        .map(|landmark| landmark.track_id)
        .collect::<BTreeSet<_>>();
    let mut used_observations = baseline_keys;
    let mut working_pose_overrides = BTreeMap::<u64, Pose>::new();
    let mut recovered_landmarks = Vec::new();

    for frame_id in zero_support_frames {
        let track_ids = recovery_candidate_track_ids(frame_id, images, store);
        let candidate_track_count = track_ids.len();
        summary.candidate_tracks += candidate_track_count;
        if track_ids.is_empty() {
            continue;
        }
        summary.frames_attempted += 1;

        let mut anchors = Vec::new();
        let mut correspondences = Vec::new();
        for track_id in track_ids {
            let track = &store.tracks[track_id];
            let target_keys = track
                .observations
                .iter()
                .copied()
                .filter(|key| {
                    images
                        .get(&key.global_image_id)
                        .is_some_and(|image| image.atlas.frame_id == frame_id)
                })
                .collect::<Vec<_>>();
            let anchor_keys = track
                .observations
                .iter()
                .copied()
                .filter(|key| {
                    images.get(&key.global_image_id).is_some_and(|image| {
                        initial_supported_frames.contains(&image.atlas.frame_id)
                    })
                })
                .collect::<Vec<_>>();
            if target_keys.is_empty() || anchor_keys.len() < MIN_TRACK_OBSERVATIONS {
                summary.anchor_rejections += 1;
                continue;
            }
            let anchor_track = GlobalTrack {
                observations: anchor_keys,
            };
            let Ok(anchor_landmark) = triangulate_track(
                track_id,
                &anchor_track,
                &store.observations,
                images,
                cameras,
            ) else {
                summary.anchor_rejections += 1;
                continue;
            };
            summary.anchor_tracks += 1;
            if correspondences.len() + target_keys.len() > MAX_RECOVERY_CORRESPONDENCES {
                break;
            }
            let first_correspondence = correspondences.len();
            for key in &target_keys {
                let image = images
                    .get(&key.global_image_id)
                    .ok_or_else(|| "recovery track references unknown image".to_owned())?;
                let observation = store
                    .observations
                    .get(key)
                    .ok_or_else(|| "recovery track references unknown observation".to_owned())?;
                correspondences.push(GeneralizedCorrespondence2D3D {
                    sensor_index: image.atlas.sensor_index,
                    point2d: observation.xy,
                    point3d: anchor_landmark.position,
                    confidence: None,
                });
            }
            anchors.push(RecoveryAnchor {
                track_id,
                correspondence_indices: (first_correspondence..correspondences.len()).collect(),
            });
        }
        summary.pnp_correspondences += correspondences.len();
        if correspondences.len() < RECOVERY_MIN_PNP_INLIERS {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} candidates={} anchors={} correspondences={} status=insufficient-correspondences",
                candidate_track_count,
                anchors.len(),
                correspondences.len(),
            );
            continue;
        }
        let Some(report) = pnp.estimate(&rig, &correspondences) else {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} anchors={} correspondences={} status=pnp-failed",
                anchors.len(),
                correspondences.len(),
            );
            continue;
        };
        summary.pnp_inliers += report.inliers.len();
        if report.inliers.len() < RECOVERY_MIN_PNP_INLIERS {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} anchors={} correspondences={} pnp_inliers={} status=insufficient-inliers",
                anchors.len(),
                correspondences.len(),
                report.inliers.len(),
            );
            continue;
        }
        let inliers = report.inliers.iter().copied().collect::<BTreeSet<_>>();
        let mut candidate_pose_overrides = working_pose_overrides.clone();
        compose_recovered_frame_poses(
            manifest,
            frame_id,
            &report.pose,
            images,
            &mut candidate_pose_overrides,
        )?;
        if !existing_landmarks_non_regressed(
            baseline_landmarks,
            &store.observations,
            images,
            cameras,
            &candidate_pose_overrides,
        )? {
            summary.pnp_rejections += 1;
            println!(
                "recovery frame={frame_id} pnp_inliers={} status=existing-model-regression",
                report.inliers.len(),
            );
            continue;
        }

        let anchor_count = anchors.len();
        let mut frame_landmarks = Vec::new();
        let mut frame_observations = BTreeSet::new();
        for anchor in anchors {
            if !anchor
                .correspondence_indices
                .iter()
                .any(|index| inliers.contains(index))
            {
                summary.track_rejections += 1;
                continue;
            }
            let filtered_keys = recovery_track_keys(
                &store.tracks[anchor.track_id],
                frame_id,
                &initial_supported_frames,
                images,
            );
            if filtered_keys.len() < MIN_TRACK_OBSERVATIONS {
                summary.track_rejections += 1;
                continue;
            }
            if filtered_keys
                .iter()
                .any(|key| used_observations.contains(key) || frame_observations.contains(key))
            {
                summary.track_rejections += 1;
                continue;
            }
            let filtered_track = GlobalTrack {
                observations: filtered_keys,
            };
            let Ok(landmark) = triangulate_track_with_pose_overrides(
                anchor.track_id,
                &filtered_track,
                &store.observations,
                images,
                cameras,
                &candidate_pose_overrides,
            ) else {
                summary.track_rejections += 1;
                continue;
            };
            if landmark.observations.len() < MIN_TRACK_OBSERVATIONS
                || used_track_ids.contains(&landmark.track_id)
            {
                summary.track_rejections += 1;
                continue;
            }
            frame_observations.extend(landmark.observations.iter().copied());
            frame_landmarks.push(landmark);
        }
        let accepted_observations = frame_landmarks
            .iter()
            .map(|landmark| landmark.observations.len())
            .sum::<usize>();
        let accepted_target_landmarks = frame_landmarks
            .iter()
            .filter(|landmark| {
                landmark.observations.iter().any(|key| {
                    images
                        .get(&key.global_image_id)
                        .is_some_and(|image| image.atlas.frame_id == frame_id)
                })
            })
            .count();
        let accepted_target_observations = frame_landmarks
            .iter()
            .flat_map(|landmark| landmark.observations.iter())
            .filter(|key| {
                images
                    .get(&key.global_image_id)
                    .is_some_and(|image| image.atlas.frame_id == frame_id)
            })
            .count();
        summary.candidate_target_landmarks += accepted_target_landmarks;
        summary.candidate_target_observations += accepted_target_observations;
        if frame_landmarks.is_empty()
            || accepted_target_landmarks < RECOVERY_MIN_PNP_INLIERS
            || accepted_target_observations < RECOVERY_MIN_PNP_INLIERS
        {
            summary.track_rejections += frame_landmarks.len();
            println!(
                "recovery frame={frame_id} pnp_inliers={} accepted_landmarks={} accepted_observations={} target_landmarks={} target_observations={} status=track-gate-rejected",
                report.inliers.len(),
                frame_landmarks.len(),
                accepted_observations,
                accepted_target_landmarks,
                accepted_target_observations,
            );
            continue;
        }
        for landmark in &frame_landmarks {
            used_track_ids.insert(landmark.track_id);
            used_observations.extend(landmark.observations.iter().copied());
        }
        summary.frames_recovered += 1;
        summary.accepted_landmarks += frame_landmarks.len();
        summary.accepted_observations += accepted_observations;
        summary.accepted_target_landmarks += accepted_target_landmarks;
        summary.accepted_target_observations += accepted_target_observations;
        println!(
            "recovery frame={frame_id} anchors={} correspondences={} pnp_inliers={} pnp_mean_px={:.6} pnp_max_px={:.6} accepted_landmarks={} accepted_observations={} target_landmarks={} target_observations={} status=accepted",
            anchor_count,
            correspondences.len(),
            report.inliers.len(),
            report.mean_reprojection_error,
            report.max_reprojection_error,
            frame_landmarks.len(),
            accepted_observations,
            accepted_target_landmarks,
            accepted_target_observations,
        );
        working_pose_overrides = candidate_pose_overrides;
        recovered_landmarks.extend(frame_landmarks);
    }

    Ok(RecoveryResult {
        pose_overrides: working_pose_overrides,
        landmarks: recovered_landmarks,
        summary,
    })
}

fn recovery_candidate_track_ids(
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
    store: &TrackStore,
) -> Vec<usize> {
    let image_ids = images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
        .map(|image| image.atlas.global_image_id)
        .collect::<Vec<_>>();
    let mut track_ids = BTreeSet::new();
    for global_image_id in image_ids {
        let first = ObservationKey {
            global_image_id,
            keypoint_index: 0,
        };
        let last = ObservationKey {
            global_image_id,
            keypoint_index: usize::MAX,
        };
        track_ids.extend(
            store
                .observations
                .range(first..=last)
                .map(|(_, state)| state.owner_track),
        );
        if track_ids.len() >= MAX_RECOVERY_TRACKS_PER_FRAME {
            break;
        }
    }
    track_ids
        .into_iter()
        .take(MAX_RECOVERY_TRACKS_PER_FRAME)
        .filter(|track_id| {
            store
                .tracks
                .get(*track_id)
                .is_some_and(|track| !track.observations.is_empty())
        })
        .collect()
}

fn build_generalized_rig(
    manifest: &RigManifest,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<GeneralizedCameraRig, String> {
    let mut sensors = Vec::with_capacity(manifest.sensors.len());
    for (index, calibration) in &manifest.sensors {
        if *index != sensors.len() {
            return Err("rig sensor indices must be contiguous from zero".to_owned());
        }
        let camera = cameras
            .get(&calibration.camera_id)
            .ok_or_else(|| format!("recovery camera {} is unavailable", calibration.camera_id))?;
        sensors.push(RigSensor {
            camera: camera.clone(),
            sensor_from_rig: calibration.sensor_from_rig.clone(),
        });
    }
    GeneralizedCameraRig::new(sensors)
        .ok_or_else(|| "recovery rig has invalid intrinsics or extrinsics".to_owned())
}

fn compose_recovered_frame_poses(
    manifest: &RigManifest,
    frame_id: u64,
    rig_pose: &Pose,
    images: &BTreeMap<u64, GlobalImage>,
    pose_overrides: &mut BTreeMap<u64, Pose>,
) -> Result<(), String> {
    let mut changed = 0;
    for image in images.values() {
        if image.atlas.frame_id != frame_id {
            continue;
        }
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "recovery image references unknown sensor".to_owned())?;
        let world_to_sensor = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
        pose_overrides.insert(
            image.atlas.global_image_id,
            Pose {
                world_to_camera: world_to_sensor,
            },
        );
        changed += 1;
    }
    if changed == 0 {
        return Err(format!("recovery frame {frame_id} has no atlas images"));
    }
    Ok(())
}

fn apply_pose_overrides(
    images: &mut BTreeMap<u64, GlobalImage>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(), String> {
    for (global_image_id, pose) in pose_overrides {
        let image = images
            .get_mut(global_image_id)
            .ok_or_else(|| "recovery pose references unknown image".to_owned())?;
        image.atlas.pose = pose.clone();
    }
    Ok(())
}

fn recovery_track_keys(
    track: &GlobalTrack,
    target_frame: u64,
    initially_supported_frames: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Vec<ObservationKey> {
    track
        .observations
        .iter()
        .copied()
        .filter(|key| {
            images.get(&key.global_image_id).is_some_and(|image| {
                image.atlas.frame_id == target_frame
                    || initially_supported_frames.contains(&image.atlas.frame_id)
            })
        })
        .collect()
}

fn existing_landmarks_non_regressed(
    landmarks: &[LandmarkOutput],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<bool, String> {
    for landmark in landmarks {
        let (mean, max) =
            evaluate_landmark_position(landmark, observations, images, cameras, pose_overrides)?;
        if mean > landmark.mean_error + 1.0e-9 || max > landmark.max_error + 1.0e-9 {
            return Ok(false);
        }
    }
    Ok(true)
}

fn evaluate_landmark_position(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(f64, f64), String> {
    let metrics =
        evaluate_landmark_metrics(landmark, observations, images, cameras, pose_overrides)?;
    Ok((metrics.mean_error, metrics.max_error))
}

#[derive(Debug, Clone, PartialEq)]
struct LandmarkMetrics {
    errors: Vec<f64>,
    mean_error: f64,
    rms_error: f64,
    max_error: f64,
}

fn evaluate_landmark_metrics(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkMetrics, String> {
    let mut errors = Vec::with_capacity(landmark.observations.len());
    for key in &landmark.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&landmark.position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
            return Err("existing landmark became invalid".to_owned());
        }
        let projected = camera
            .project(&point_camera)
            .ok_or_else(|| "existing landmark projection failed".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("existing landmark reprojection became non-finite".to_owned());
        }
        errors.push(error);
    }
    if errors.is_empty() {
        return Err("landmark has no observations".to_owned());
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    Ok(LandmarkMetrics {
        max_error: errors.iter().copied().fold(0.0, f64::max),
        errors,
        mean_error: mean,
        rms_error: rms,
    })
}

// The filter mode must inspect depth failures after the all-observation cost
// gate.  Camera::project intentionally rejects non-positive depth, so this
// raw-cost helper evaluates the same projection equations without applying
// the depth gate.  The depth gate remains mandatory during candidate
// filtering/retriangulation and in the final validator.
fn project_allowing_nonpositive_depth(
    camera: &Camera,
    point_camera: &Point3<f64>,
) -> Option<Point2<f64>> {
    if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z == 0.0 {
        return None;
    }
    if point_camera.z > 0.0 {
        return camera.project(point_camera);
    }
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    if ![fx, fy, cx, cy].iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut x = point_camera.x / point_camera.z;
    let mut y = point_camera.y / point_camera.z;
    if let Some((k1, k2)) = camera.radial_distortion() {
        if !k1.is_finite() || !k2.is_finite() {
            return None;
        }
        let r2 = x * x + y * y;
        let d = 1.0 + k1 * r2 + k2 * r2 * r2;
        x *= d;
        y *= d;
    }
    let projected = Point2::new(fx * x + cx, fy * y + cy);
    projected
        .coords
        .iter()
        .all(|value| value.is_finite())
        .then_some(projected)
}

fn evaluate_landmark_metrics_allowing_nonpositive_depth(
    landmark: &LandmarkOutput,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkMetrics, String> {
    let mut errors = Vec::with_capacity(landmark.observations.len());
    for key in &landmark.observations {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let pose = pose_for_image(image, pose_overrides);
        let point_camera = pose.transform_world_point(&landmark.position);
        let projected = project_allowing_nonpositive_depth(camera, &point_camera)
            .ok_or_else(|| "landmark projection became non-finite".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("landmark reprojection became non-finite".to_owned());
        }
        errors.push(error);
    }
    if errors.is_empty() {
        return Err("landmark has no observations".to_owned());
    }
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let rms = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    Ok(LandmarkMetrics {
        max_error: errors.iter().copied().fold(0.0, f64::max),
        errors,
        mean_error: mean,
        rms_error: rms,
    })
}

fn pose_for_image<'a>(image: &'a GlobalImage, pose_overrides: &'a BTreeMap<u64, Pose>) -> &'a Pose {
    pose_overrides
        .get(&image.atlas.global_image_id)
        .unwrap_or(&image.atlas.pose)
}

#[derive(Debug, Clone, PartialEq)]
struct JointRigBaWindowUpdate {
    pose_overrides: BTreeMap<u64, Pose>,
    landmark_updates: BTreeMap<usize, LandmarkOutput>,
    selected_landmarks: usize,
    selected_observations: usize,
    referenced_frames: usize,
    free_frames: usize,
    initial_cost: f64,
    final_cost: f64,
    solver_initial_cost: f64,
    solver_final_cost: f64,
    iterations: usize,
    converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct JointRigBaFilteringCandidate {
    pose_overrides: BTreeMap<u64, Pose>,
    landmark_updates: BTreeMap<usize, LandmarkOutput>,
    removed_observations: BTreeMap<ObservationKey, FilterObservationReason>,
    removed_track_ids: BTreeSet<usize>,
    selected_landmarks: usize,
    selected_observations: usize,
    referenced_frames: usize,
    free_frames: usize,
    full_pre_ba_cost: f64,
    full_post_ba_cost: f64,
    retained_pre_ba_cost: f64,
    retained_post_filter_cost: f64,
    retriangulated_tracks: usize,
    raw_preserved_tracks: usize,
    dlt_attempted_tracks: usize,
    raw_fallback_reason_counts: BTreeMap<RawPointFallbackReason, usize>,
    iterations: usize,
    converged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum JointRigBaSkip {
    LandmarkCap { count: usize },
    ObservationCap { count: usize },
    ReferencedFrameCap { count: usize },
    NoFreeFrames,
    Invalid(String),
}

impl std::fmt::Display for JointRigBaSkip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LandmarkCap { count } => write!(f, "landmark cap exceeded ({count})"),
            Self::ObservationCap { count } => write!(f, "observation cap exceeded ({count})"),
            Self::ReferencedFrameCap { count } => {
                write!(f, "referenced-frame cap exceeded ({count})")
            }
            Self::NoFreeFrames => write!(f, "window has no free frames"),
            Self::Invalid(reason) => write!(f, "invalid BA candidate: {reason}"),
        }
    }
}

fn derive_rig_pose_for_frame(
    manifest: &RigManifest,
    frame_id: u64,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<Pose, String> {
    let mut pose: Option<Pose> = None;
    for image in images
        .values()
        .filter(|image| image.atlas.frame_id == frame_id)
    {
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| format!("frame {frame_id} references unknown sensor"))?;
        let candidate = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&image.atlas.pose.world_to_camera),
        };
        if let Some(previous) = &pose {
            let center_error =
                (previous.camera_center_world() - candidate.camera_center_world()).norm();
            let rotation_error = previous
                .world_to_camera
                .rotation
                .rotation_to(&candidate.world_to_camera.rotation)
                .angle()
                .to_degrees();
            if !center_error.is_finite()
                || !rotation_error.is_finite()
                || center_error > MAX_RIG_CENTER_DISAGREEMENT_M
                || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
            {
                return Err(format!(
                    "frame {frame_id} rig pose disagreement: centre={center_error:.9}m rotation={rotation_error:.9}deg"
                ));
            }
        } else {
            pose = Some(candidate);
        }
    }
    pose.ok_or_else(|| format!("frame {frame_id} has no atlas images"))
}

fn derive_rig_poses_for_frames(
    manifest: &RigManifest,
    frame_ids: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<BTreeMap<u64, Pose>, String> {
    frame_ids
        .iter()
        .map(|frame_id| {
            derive_rig_pose_for_frame(manifest, *frame_id, images).map(|pose| (*frame_id, pose))
        })
        .collect()
}

fn selected_landmarks_for_frames(
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    images: &BTreeMap<u64, GlobalImage>,
) -> Result<Vec<usize>, String> {
    let mut selected = Vec::new();
    for (index, landmark) in landmarks.iter().enumerate() {
        let touches_active = landmark.observations.iter().any(|key| {
            images
                .get(&key.global_image_id)
                .is_some_and(|image| active_frames.contains(&image.atlas.frame_id))
        });
        if touches_active {
            selected.push(index);
        }
    }
    // This is intentionally a definition, not a heuristic: all tracks with
    // an active observation are selected, so no active pose is later changed
    // while silently leaving an unselected landmark unsupported.
    Ok(selected)
}

fn validate_joint_ba_update(
    update: &JointRigBaWindowUpdate,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    selected: &[usize],
) -> Result<(), JointRigBaSkip> {
    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    for (index, landmark) in landmarks.iter().enumerate() {
        let touches_active = landmark.observations.iter().any(|key| {
            images.get(&key.global_image_id).is_some_and(|image| {
                update
                    .pose_overrides
                    .contains_key(&image.atlas.global_image_id)
            })
        });
        if touches_active != selected_set.contains(&index) {
            return Err(JointRigBaSkip::Invalid(format!(
                "selection invariant failed for landmark index {index}"
            )));
        }
    }
    if update.landmark_updates.len() != selected.len() {
        return Err(JointRigBaSkip::Invalid(format!(
            "candidate landmark count {} differs from selected count {}",
            update.landmark_updates.len(),
            selected.len()
        )));
    }
    let mut initial_cost = 0.0;
    let mut final_cost = 0.0;
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let old_metrics =
            evaluate_landmark_metrics(old, &store.observations, images, cameras, &BTreeMap::new())
                .map_err(JointRigBaSkip::Invalid)?;
        initial_cost += old_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
        let candidate = update.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("candidate omitted selected landmark {index}"))
        })?;
        if candidate.track_id != old.track_id || candidate.observations != old.observations {
            return Err(JointRigBaSkip::Invalid(format!(
                "candidate changed identity or observations for landmark {index}"
            )));
        }
        let candidate_metrics = evaluate_landmark_metrics(
            candidate,
            &store.observations,
            images,
            cameras,
            &update.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        final_cost += candidate_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
    }
    if !update.initial_cost.is_finite()
        || !update.final_cost.is_finite()
        || !initial_cost.is_finite()
        || !final_cost.is_finite()
        || final_cost > initial_cost + 1.0e-8 * initial_cost.abs().max(1.0)
    {
        return Err(JointRigBaSkip::Invalid(format!(
            "validated cost increased from {initial_cost:.12} to {final_cost:.12}"
        )));
    }
    if (update.initial_cost - initial_cost).abs() > 1.0e-8 * initial_cost.abs().max(1.0)
        || (update.final_cost - final_cost).abs() > 1.0e-8 * final_cost.abs().max(1.0)
    {
        return Err(JointRigBaSkip::Invalid(
            "candidate cost diagnostics disagree with independent reprojection".to_owned(),
        ));
    }
    for (index, candidate) in &update.landmark_updates {
        if !selected_set.contains(index) {
            return Err(JointRigBaSkip::Invalid(format!(
                "candidate updated unselected landmark {index}"
            )));
        }
        let metrics = evaluate_landmark_metrics(
            candidate,
            &store.observations,
            images,
            cameras,
            &update.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        if candidate.observations.len() < MIN_TRACK_OBSERVATIONS
            || metrics.mean_error > MAX_MEAN_REPROJECTION_PX
            || metrics.max_error > MAX_REPROJECTION_PX
        {
            return Err(JointRigBaSkip::Invalid(format!(
                "landmark {index} failed reprojection gate mean={:.6} max={:.6} observations={}",
                metrics.mean_error,
                metrics.max_error,
                candidate.observations.len()
            )));
        }
    }
    Ok(())
}

fn build_joint_rig_ba_window_raw(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    build_joint_rig_ba_window_raw_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        &selected,
    )
}

fn build_joint_rig_ba_window_raw_with_selected(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    if selected.len() > JOINT_BA_MAX_LANDMARKS {
        return Err(JointRigBaSkip::LandmarkCap {
            count: selected.len(),
        });
    }
    let selected_observations = selected.iter().try_fold(0usize, |count, index| {
        count.checked_add(landmarks[*index].observations.len())
    });
    let selected_observations =
        selected_observations.ok_or(JointRigBaSkip::ObservationCap { count: usize::MAX })?;
    if selected_observations > JOINT_BA_MAX_OBSERVATIONS {
        return Err(JointRigBaSkip::ObservationCap {
            count: selected_observations,
        });
    }
    let mut referenced_frames = BTreeSet::new();
    for index in selected {
        for key in &landmarks[*index].observations {
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown image".to_owned())
            })?;
            referenced_frames.insert(image.atlas.frame_id);
        }
    }
    if referenced_frames.len() > JOINT_BA_MAX_REFERENCED_FRAMES {
        return Err(JointRigBaSkip::ReferencedFrameCap {
            count: referenced_frames.len(),
        });
    }
    let free_frames = active_frames
        .intersection(&referenced_frames)
        .copied()
        .collect::<BTreeSet<_>>();
    if free_frames.is_empty() {
        return Err(JointRigBaSkip::NoFreeFrames);
    }
    if free_frames.len() > JOINT_BA_MAX_FREE_FRAMES {
        return Err(JointRigBaSkip::Invalid(format!(
            "free-frame cap exceeded ({})",
            free_frames.len()
        )));
    }
    let rig_poses = derive_rig_poses_for_frames(manifest, &referenced_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    let camera = cameras
        .values()
        .next()
        .cloned()
        .ok_or_else(|| JointRigBaSkip::Invalid("no cameras available".to_owned()))?;
    let mut ba = BundleAdjustment::new(camera);
    for (frame_id, pose) in &rig_poses {
        ba.add_pose(*frame_id, pose.clone());
        if !free_frames.contains(frame_id) {
            ba.fix_pose(*frame_id);
        }
    }
    if !rig_poses
        .keys()
        .any(|frame_id| !free_frames.contains(frame_id))
    {
        let anchor = *referenced_frames
            .first()
            .ok_or(JointRigBaSkip::NoFreeFrames)?;
        ba.fix_pose(anchor);
    }
    for index in selected {
        let landmark = &landmarks[*index];
        let id = u64::try_from(landmark.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        if !landmark
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(JointRigBaSkip::Invalid(format!(
                "landmark {} has non-finite position",
                landmark.track_id
            )));
        }
        ba.add_landmark(id, landmark.position);
    }
    for index in selected {
        let landmark = &landmarks[*index];
        let landmark_id = u64::try_from(landmark.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        for key in &landmark.observations {
            let image = images.get(&key.global_image_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown image".to_owned())
            })?;
            let observation = store.observations.get(key).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown observation".to_owned())
            })?;
            let sensor = manifest
                .sensors
                .get(&image.atlas.sensor_index)
                .ok_or_else(|| {
                    JointRigBaSkip::Invalid("landmark references unknown sensor".to_owned())
                })?;
            let camera = cameras.get(&image.atlas.camera_id).ok_or_else(|| {
                JointRigBaSkip::Invalid("landmark references unknown camera".to_owned())
            })?;
            if !observation.xy.coords.iter().all(|value| value.is_finite()) {
                return Err(JointRigBaSkip::Invalid(
                    "non-finite BA observation".to_owned(),
                ));
            }
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: image.atlas.frame_id,
                landmark_id,
                xy: observation.xy,
                camera: camera.clone(),
                sensor_from_rig: sensor.sensor_from_rig.clone(),
            });
        }
    }
    let baseline_cost = ba.cost();
    if !baseline_cost.is_finite() {
        return Err(JointRigBaSkip::Invalid(
            "initial BA cost is non-finite".to_owned(),
        ));
    }
    let result = ba
        .optimize(&BaConfig {
            max_iterations: JOINT_BA_MAX_ITERATIONS,
            initial_lambda: Some(1.0e-4),
            linear_solver: LinearSolver::Sparse,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            refine_distortion: false,
            parallel: false,
            ..BaConfig::default()
        })
        .map_err(|error| JointRigBaSkip::Invalid(format!("bundle adjustment failed: {error}")))?;
    let solver_initial_cost = result.initial_cost;
    let solver_final_cost = result.final_cost;
    if !solver_initial_cost.is_finite() || !solver_final_cost.is_finite() {
        return Err(JointRigBaSkip::Invalid(
            "BA costs are non-finite".to_owned(),
        ));
    }
    let mut pose_overrides = BTreeMap::new();
    for image in images.values() {
        if !free_frames.contains(&image.atlas.frame_id) {
            continue;
        }
        let rig_pose = ba
            .poses
            .get(&image.atlas.frame_id)
            .ok_or_else(|| JointRigBaSkip::Invalid("BA omitted referenced rig pose".to_owned()))?;
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| {
                JointRigBaSkip::Invalid("BA image references unknown sensor".to_owned())
            })?;
        let pose = Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
        };
        if !pose
            .world_to_camera
            .translation
            .iter()
            .all(|value| value.is_finite())
            || !pose
                .world_to_camera
                .rotation
                .coords
                .iter()
                .all(|value| value.is_finite())
        {
            return Err(JointRigBaSkip::Invalid(
                "BA produced non-finite pose".to_owned(),
            ));
        }
        pose_overrides.insert(image.atlas.global_image_id, pose);
    }
    let mut landmark_updates = BTreeMap::new();
    for index in selected {
        let old = &landmarks[*index];
        let id = u64::try_from(old.track_id)
            .map_err(|_| JointRigBaSkip::Invalid("track id exceeds u64".to_owned()))?;
        let position = *ba
            .landmarks
            .get(&id)
            .ok_or_else(|| JointRigBaSkip::Invalid("BA omitted selected landmark".to_owned()))?;
        let mut candidate = old.clone();
        candidate.position = position;
        let metrics = evaluate_landmark_metrics_allowing_nonpositive_depth(
            &candidate,
            &store.observations,
            images,
            cameras,
            &pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        candidate.errors = metrics.errors;
        candidate.mean_error = metrics.mean_error;
        candidate.rms_error = metrics.rms_error;
        candidate.max_error = metrics.max_error;
        landmark_updates.insert(*index, candidate);
    }
    let mut initial_cost = 0.0;
    let mut final_cost = 0.0;
    for index in selected {
        let old_metrics = evaluate_landmark_metrics(
            &landmarks[*index],
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        let new_metrics = evaluate_landmark_metrics_allowing_nonpositive_depth(
            landmark_updates
                .get(index)
                .ok_or_else(|| JointRigBaSkip::Invalid("BA update omitted landmark".to_owned()))?,
            &store.observations,
            images,
            cameras,
            &pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        initial_cost += old_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
        final_cost += new_metrics
            .errors
            .iter()
            .map(|error| error * error)
            .sum::<f64>();
    }
    let update = JointRigBaWindowUpdate {
        pose_overrides,
        landmark_updates,
        selected_landmarks: selected.len(),
        selected_observations,
        referenced_frames: referenced_frames.len(),
        free_frames: free_frames.len(),
        initial_cost,
        final_cost,
        solver_initial_cost,
        solver_final_cost,
        iterations: result.iterations.len(),
        converged: result.converged,
    };
    Ok(update)
}

fn build_joint_rig_ba_window(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
) -> Result<JointRigBaWindowUpdate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    let update =
        build_joint_rig_ba_window_raw(manifest, store, images, cameras, landmarks, active_frames)?;
    if let Err(reason) =
        validate_joint_ba_update(&update, store, images, cameras, landmarks, &selected)
    {
        println!(
            "joint_rig_ba_candidate status=rejected selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} solver_initial_cost={:.9} solver_final_cost={:.9} iterations={} converged={} reason={reason}",
            update.selected_landmarks,
            update.selected_observations,
            update.referenced_frames,
            update.free_frames,
            update.solver_initial_cost,
            update.solver_final_cost,
            update.iterations,
            update.converged,
        );
        return Err(reason);
    }
    Ok(update)
}

fn classify_filter_observation(
    key: ObservationKey,
    position: &Point3<f64>,
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<f64, FilterObservationReason> {
    if !position.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    let image = images
        .get(&key.global_image_id)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let camera = cameras
        .get(&image.atlas.camera_id)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let observation = observations
        .get(&key)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    if !observation.xy.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    let point_camera = pose_for_image(image, pose_overrides).transform_world_point(position);
    if !point_camera.coords.iter().all(|value| value.is_finite()) {
        return Err(FilterObservationReason::NonFinite);
    }
    if point_camera.z <= 0.0 {
        return Err(FilterObservationReason::BehindCamera);
    }
    let projected = project_allowing_nonpositive_depth(camera, &point_camera)
        .ok_or(FilterObservationReason::ProjectionFailure)?;
    let error = (projected - observation.xy).norm();
    if !error.is_finite() {
        return Err(FilterObservationReason::NonFinite);
    }
    if error > MAX_REPROJECTION_PX {
        return Err(FilterObservationReason::ReprojectionOverMax);
    }
    Ok(error)
}

/// Keep a raw BA point only when filtering left its complete original key set
/// intact and the point passes the same bounded geometric gates as DLT.  The
/// caller falls back to the historical DLT path for every error here; this
/// helper never changes observation ownership or filtering decisions.
fn retain_raw_ba_point_if_valid(
    candidate: &LandmarkOutput,
    expected_observations: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<LandmarkOutput, RawPointFallbackReason> {
    if candidate.observations != expected_observations {
        return Err(RawPointFallbackReason::ObservationKeysChanged);
    }
    if candidate.observations.len() < MIN_TRACK_OBSERVATIONS {
        return Err(RawPointFallbackReason::TooFewObservations);
    }
    if !candidate
        .position
        .coords
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(RawPointFallbackReason::NonFinite);
    }
    let sample = bounded_sample(&candidate.observations, MAX_DLT_OBSERVATIONS);
    match has_observable_parallax(&sample, observations, images, cameras, pose_overrides) {
        Ok(true) => {}
        Ok(false) => return Err(RawPointFallbackReason::NoObservableParallax),
        Err(_) => return Err(RawPointFallbackReason::ParallaxCheckFailed),
    }

    let mut errors = Vec::with_capacity(candidate.observations.len());
    for key in &candidate.observations {
        let Some(image) = images.get(&key.global_image_id) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let Some(camera) = cameras.get(&image.atlas.camera_id) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let Some(observation) = observations.get(key) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        if !observation.xy.coords.iter().all(|value| value.is_finite()) {
            return Err(RawPointFallbackReason::NonFinite);
        }
        let point_camera =
            pose_for_image(image, pose_overrides).transform_world_point(&candidate.position);
        if !point_camera.coords.iter().all(|value| value.is_finite()) {
            return Err(RawPointFallbackReason::NonFinite);
        }
        if point_camera.z <= 0.0 {
            return Err(RawPointFallbackReason::NonPositiveDepth);
        }
        let Some(projected) = camera.project(&point_camera) else {
            return Err(RawPointFallbackReason::ProjectionFailure);
        };
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err(RawPointFallbackReason::NonFinite);
        }
        errors.push(error);
    }
    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    let squared_sum = errors.iter().map(|error| error * error).sum::<f64>();
    let rms_error = (squared_sum / errors.len() as f64).sqrt();
    let max_error = errors.iter().copied().fold(0.0, f64::max);
    if !mean_error.is_finite() || !rms_error.is_finite() || !max_error.is_finite() {
        return Err(RawPointFallbackReason::NonFinite);
    }
    if mean_error > MAX_MEAN_REPROJECTION_PX {
        return Err(RawPointFallbackReason::MeanOverMax);
    }
    if max_error > MAX_REPROJECTION_PX {
        return Err(RawPointFallbackReason::MaxOverMax);
    }
    let mut retained = candidate.clone();
    retained.errors = errors;
    retained.mean_error = mean_error;
    retained.rms_error = rms_error;
    retained.max_error = max_error;
    Ok(retained)
}

fn cost_for_landmark_keys(
    position: &Point3<f64>,
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<f64, String> {
    let mut cost = 0.0;
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "landmark references unknown image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "landmark references unknown camera".to_owned())?;
        let observation = observations
            .get(key)
            .ok_or_else(|| "landmark references unknown observation".to_owned())?;
        let point_camera = pose_for_image(image, pose_overrides).transform_world_point(position);
        let projected = project_allowing_nonpositive_depth(camera, &point_camera)
            .ok_or_else(|| "landmark cost became non-finite".to_owned())?;
        let error = (projected - observation.xy).norm();
        if !error.is_finite() {
            return Err("landmark cost became non-finite".to_owned());
        }
        cost += error * error;
    }
    if !cost.is_finite() {
        return Err("landmark cost became non-finite".to_owned());
    }
    Ok(cost)
}

fn cost_non_increasing(before: f64, after: f64) -> bool {
    before.is_finite() && after.is_finite() && after <= before + 1.0e-8 * before.abs().max(1.0)
}

fn validate_filter_costs(
    full_pre_ba_cost: f64,
    full_post_ba_cost: f64,
    retained_pre_ba_cost: f64,
    retained_post_filter_cost: f64,
) -> Result<(), String> {
    if !cost_non_increasing(full_pre_ba_cost, full_post_ba_cost) {
        return Err(format!(
            "full selected cost increased before filtering from {full_pre_ba_cost:.12} to {full_post_ba_cost:.12}"
        ));
    }
    if !cost_non_increasing(retained_pre_ba_cost, retained_post_filter_cost) {
        return Err(format!(
            "retained cost increased after filtering from {retained_pre_ba_cost:.12} to {retained_post_filter_cost:.12}"
        ));
    }
    Ok(())
}

fn joint_ba_window_counts(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    active_frames: &BTreeSet<u64>,
) -> Result<(usize, usize, usize, usize), String> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)?;
    joint_ba_window_counts_with_selected(landmarks, images, active_frames, &selected)
}

fn joint_ba_window_counts_with_selected(
    landmarks: &[LandmarkOutput],
    images: &BTreeMap<u64, GlobalImage>,
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
) -> Result<(usize, usize, usize, usize), String> {
    let selected_observations = selected
        .iter()
        .map(|index| landmarks[*index].observations.len())
        .sum::<usize>();
    let mut referenced_frames = BTreeSet::new();
    for index in selected {
        for key in &landmarks[*index].observations {
            let image = images
                .get(&key.global_image_id)
                .ok_or_else(|| "landmark references unknown image".to_owned())?;
            referenced_frames.insert(image.atlas.frame_id);
        }
    }
    let free_frames = active_frames.intersection(&referenced_frames).count();
    Ok((
        selected.len(),
        selected_observations,
        referenced_frames.len(),
        free_frames,
    ))
}

#[cfg(test)]
fn build_joint_rig_ba_filter_candidate(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    baseline_connectivity: &SupportConnectivity,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    build_joint_rig_ba_filter_candidate_with_policy(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        baseline_connectivity,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn build_joint_rig_ba_filter_candidate_with_policy(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    baseline_connectivity: &SupportConnectivity,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    let selected = selected_landmarks_for_frames(landmarks, active_frames, images)
        .map_err(JointRigBaSkip::Invalid)?;
    build_joint_rig_ba_filter_candidate_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        &selected,
        baseline_connectivity,
        preserve_optimized_points,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_joint_rig_ba_filter_candidate_with_selected(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    active_frames: &BTreeSet<u64>,
    selected: &[usize],
    baseline_connectivity: &SupportConnectivity,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringCandidate, JointRigBaSkip> {
    let raw = build_joint_rig_ba_window_raw_with_selected(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        active_frames,
        selected,
    )?;
    if raw.landmark_updates.len() != selected.len() {
        return Err(JointRigBaSkip::Invalid(
            "raw BA candidate omitted a selected landmark".to_owned(),
        ));
    }

    let mut full_pre_ba_cost = 0.0;
    let mut full_post_ba_cost = 0.0;
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let candidate = raw.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("raw BA omitted selected landmark {index}"))
        })?;
        if candidate.track_id != old.track_id || candidate.observations != old.observations {
            return Err(JointRigBaSkip::Invalid(format!(
                "raw BA changed identity or observations for landmark {index}"
            )));
        }
        full_pre_ba_cost += cost_for_landmark_keys(
            &old.position,
            &old.observations,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        full_post_ba_cost += cost_for_landmark_keys(
            &candidate.position,
            &candidate.observations,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
    }
    let mut landmark_updates = BTreeMap::new();
    let mut removed_observations = BTreeMap::new();
    let mut removed_track_ids = BTreeSet::new();
    let mut retained_pre_ba_cost = 0.0;
    let mut retained_post_filter_cost = 0.0;
    let mut retriangulated_tracks = 0;
    let mut raw_preserved_tracks = 0;
    let mut dlt_attempted_tracks = 0;
    let mut raw_fallback_reason_counts = BTreeMap::new();
    for index in selected {
        let old = landmarks.get(*index).ok_or_else(|| {
            JointRigBaSkip::Invalid("selected landmark index is invalid".to_owned())
        })?;
        let candidate = raw.landmark_updates.get(index).ok_or_else(|| {
            JointRigBaSkip::Invalid(format!("raw BA omitted selected landmark {index}"))
        })?;
        let raw_gate_result = preserve_optimized_points.then(|| {
            retain_raw_ba_point_if_valid(
                candidate,
                &old.observations,
                &store.observations,
                images,
                cameras,
                &raw.pose_overrides,
            )
        });
        let mut retained = Vec::with_capacity(candidate.observations.len());
        for key in &candidate.observations {
            match classify_filter_observation(
                *key,
                &candidate.position,
                &store.observations,
                images,
                cameras,
                &raw.pose_overrides,
            ) {
                Ok(_) => retained.push(*key),
                Err(reason) => {
                    removed_observations.insert(*key, reason);
                }
            }
        }
        if retained.len() < MIN_TRACK_OBSERVATIONS {
            if preserve_optimized_points {
                if let Some(Err(reason)) = raw_gate_result {
                    *raw_fallback_reason_counts.entry(reason).or_default() += 1;
                }
            }
            removed_track_ids.insert(old.track_id);
            for key in &old.observations {
                removed_observations
                    .entry(*key)
                    .or_insert(FilterObservationReason::TrackBelowMinimum);
            }
            continue;
        }

        if preserve_optimized_points {
            let raw_result = raw_gate_result
                .expect("raw gate result is present when optimized points are enabled");
            match (retained == old.observations, raw_result) {
                (true, Ok(raw_landmark)) => {
                    retained_pre_ba_cost += cost_for_landmark_keys(
                        &old.position,
                        &retained,
                        &store.observations,
                        images,
                        cameras,
                        &BTreeMap::new(),
                    )
                    .map_err(JointRigBaSkip::Invalid)?;
                    retained_post_filter_cost += cost_for_landmark_keys(
                        &raw_landmark.position,
                        &retained,
                        &store.observations,
                        images,
                        cameras,
                        &raw.pose_overrides,
                    )
                    .map_err(JointRigBaSkip::Invalid)?;
                    raw_preserved_tracks += 1;
                    landmark_updates.insert(*index, raw_landmark);
                    continue;
                }
                (false, Ok(_)) => {
                    *raw_fallback_reason_counts
                        .entry(RawPointFallbackReason::ObservationKeysChanged)
                        .or_default() += 1;
                }
                (_, Err(reason)) => {
                    *raw_fallback_reason_counts.entry(reason).or_default() += 1;
                }
            }
        }

        dlt_attempted_tracks += 1;
        let refined_track = GlobalTrack {
            observations: retained.clone(),
        };
        let refined = match triangulate_track_with_pose_overrides(
            old.track_id,
            &refined_track,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        ) {
            Ok(landmark) => landmark,
            Err(_) => {
                removed_track_ids.insert(old.track_id);
                for key in &old.observations {
                    removed_observations
                        .entry(*key)
                        .or_insert(FilterObservationReason::RetriangulationFailed);
                }
                continue;
            }
        };
        retained_pre_ba_cost += cost_for_landmark_keys(
            &old.position,
            &retained,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )
        .map_err(JointRigBaSkip::Invalid)?;
        retained_post_filter_cost += cost_for_landmark_keys(
            &refined.position,
            &retained,
            &store.observations,
            images,
            cameras,
            &raw.pose_overrides,
        )
        .map_err(JointRigBaSkip::Invalid)?;
        retriangulated_tracks += 1;
        landmark_updates.insert(*index, refined);
    }
    validate_filter_costs(
        full_pre_ba_cost,
        full_post_ba_cost,
        retained_pre_ba_cost,
        retained_post_filter_cost,
    )
    .map_err(JointRigBaSkip::Invalid)?;

    let candidate_connectivity = support_connectivity_for_landmarks(
        landmarks,
        &landmark_updates,
        &removed_track_ids,
        images,
    )
    .map_err(JointRigBaSkip::Invalid)?;
    if !connectivity_not_split(baseline_connectivity, &candidate_connectivity) {
        return Err(JointRigBaSkip::Invalid(
            "filter candidate loses supported image/frame support or splits connectivity"
                .to_owned(),
        ));
    }
    validate_candidate_pose_overrides(manifest, images, &raw.pose_overrides)
        .map_err(JointRigBaSkip::Invalid)?;
    Ok(JointRigBaFilteringCandidate {
        pose_overrides: raw.pose_overrides,
        landmark_updates,
        removed_observations,
        removed_track_ids,
        selected_landmarks: raw.selected_landmarks,
        selected_observations: raw.selected_observations,
        referenced_frames: raw.referenced_frames,
        free_frames: raw.free_frames,
        full_pre_ba_cost,
        full_post_ba_cost,
        retained_pre_ba_cost,
        retained_post_filter_cost,
        retriangulated_tracks,
        raw_preserved_tracks,
        dlt_attempted_tracks,
        raw_fallback_reason_counts,
        iterations: raw.iterations,
        converged: raw.converged,
    })
}

fn validate_candidate_pose_overrides(
    manifest: &RigManifest,
    images: &BTreeMap<u64, GlobalImage>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<(), String> {
    let mut frame_poses = BTreeMap::<u64, Pose>::new();
    for (image_id, pose) in pose_overrides {
        let image = images
            .get(image_id)
            .ok_or_else(|| "BA pose override references unknown image".to_owned())?;
        let sensor = manifest
            .sensors
            .get(&image.atlas.sensor_index)
            .ok_or_else(|| "BA pose override references unknown sensor".to_owned())?;
        if !diagnostic_pose_is_finite(pose) {
            return Err("BA pose override is non-finite".to_owned());
        }
        let rig_pose = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&pose.world_to_camera),
        };
        if let Some(previous) = frame_poses.insert(image.atlas.frame_id, rig_pose.clone()) {
            let center_error =
                (previous.camera_center_world() - rig_pose.camera_center_world()).norm();
            let rotation_error = previous
                .world_to_camera
                .rotation
                .rotation_to(&rig_pose.world_to_camera.rotation)
                .angle()
                .to_degrees();
            if !center_error.is_finite()
                || !rotation_error.is_finite()
                || center_error > MAX_RIG_CENTER_DISAGREEMENT_M
                || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
            {
                return Err(format!(
                    "BA pose overrides disagree for frame {}",
                    image.atlas.frame_id
                ));
            }
        }
    }
    Ok(())
}

fn filter_reason_counts(
    removed_observations: &BTreeMap<ObservationKey, FilterObservationReason>,
) -> BTreeMap<FilterObservationReason, usize> {
    let mut counts = BTreeMap::new();
    for reason in removed_observations.values() {
        *counts.entry(*reason).or_default() += 1;
    }
    counts
}

fn format_filter_reason_counts(counts: &BTreeMap<FilterObservationReason, usize>) -> String {
    counts
        .iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_raw_fallback_reason_counts(counts: &BTreeMap<RawPointFallbackReason, usize>) -> String {
    if counts.is_empty() {
        return "none".to_owned();
    }
    counts
        .iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn validate_filter_candidate_for_apply(
    candidate: &JointRigBaFilteringCandidate,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    for (image_id, pose) in &candidate.pose_overrides {
        if !images.contains_key(image_id) {
            return Err(format!(
                "filter candidate pose override references unknown image {image_id}"
            ));
        }
        if !diagnostic_pose_is_finite(pose) {
            return Err(format!(
                "filter candidate pose override for image {image_id} is non-finite"
            ));
        }
    }

    let mut replacement_tracks = BTreeSet::new();
    let mut retained_keys = BTreeSet::new();
    let mut expected_removed_keys = BTreeSet::new();
    for (index, replacement) in &candidate.landmark_updates {
        let old = landmarks
            .get(*index)
            .ok_or_else(|| format!("filter candidate landmark index {index} is invalid"))?;
        if old.track_id != replacement.track_id {
            return Err(format!(
                "filter candidate changed track identity at landmark index {index}"
            ));
        }
        if replacement.observations.len() < MIN_TRACK_OBSERVATIONS {
            return Err(format!(
                "filter candidate replacement track {} has too few observations",
                replacement.track_id
            ));
        }
        if !replacement
            .position
            .coords
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "filter candidate replacement track {} has a non-finite position",
                replacement.track_id
            ));
        }
        if !replacement_tracks.insert(replacement.track_id) {
            return Err(format!(
                "filter candidate updates track {} more than once",
                replacement.track_id
            ));
        }
        let source_track = store.tracks.get(replacement.track_id).ok_or_else(|| {
            format!(
                "filter candidate replacement references unknown track {}",
                replacement.track_id
            )
        })?;
        let old_keys = old.observations.iter().copied().collect::<BTreeSet<_>>();
        let source_keys = source_track
            .observations
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if source_keys != old_keys {
            return Err(format!(
                "filter candidate source track {} disagrees with landmark observations",
                replacement.track_id
            ));
        }
        let mut replacement_keys = BTreeSet::new();
        for key in &replacement.observations {
            if !replacement_keys.insert(*key) {
                return Err(format!(
                    "filter candidate replacement track {} has duplicate observation {:?}",
                    replacement.track_id, key
                ));
            }
            if !old_keys.contains(key) {
                return Err(format!(
                    "filter candidate replacement track {} adds observation {:?}",
                    replacement.track_id, key
                ));
            }
            if !images.contains_key(&key.global_image_id) {
                return Err(format!(
                    "filter candidate replacement references unknown image {}",
                    key.global_image_id
                ));
            }
            let state = store.observations.get(key).ok_or_else(|| {
                format!("filter candidate replacement references unknown observation {key:?}")
            })?;
            if state.owner_track != replacement.track_id {
                return Err(format!(
                    "filter candidate replacement observation {:?} belongs to track {}",
                    key, state.owner_track
                ));
            }
            retained_keys.insert(*key);
        }
        expected_removed_keys.extend(old_keys.difference(&replacement_keys).copied());
        if candidate.removed_track_ids.contains(&replacement.track_id) {
            return Err(format!(
                "filter candidate both replaces and removes track {}",
                replacement.track_id
            ));
        }
    }

    for track_id in &candidate.removed_track_ids {
        let track = store
            .tracks
            .get(*track_id)
            .ok_or_else(|| format!("filter candidate references unknown track {track_id}"))?;
        if replacement_tracks.contains(track_id) {
            return Err(format!(
                "filter candidate both replaces and removes track {track_id}"
            ));
        }
        for key in &track.observations {
            if !images.contains_key(&key.global_image_id) {
                return Err(format!(
                    "source track {track_id} references unknown image {}",
                    key.global_image_id
                ));
            }
            let state = store.observations.get(key).ok_or_else(|| {
                format!("source track {track_id} references unknown observation {key:?}")
            })?;
            if state.owner_track != *track_id {
                return Err(format!(
                    "source track {track_id} observation {:?} has owner {}",
                    key, state.owner_track
                ));
            }
            expected_removed_keys.insert(*key);
        }
    }

    let actual_removed_keys = candidate
        .removed_observations
        .keys()
        .copied()
        .collect::<BTreeSet<_>>();
    if actual_removed_keys != expected_removed_keys {
        return Err(format!(
            "filter candidate removal ledger disagrees: expected {} keys, got {}",
            expected_removed_keys.len(),
            actual_removed_keys.len()
        ));
    }
    for key in candidate.removed_observations.keys() {
        if !images.contains_key(&key.global_image_id) {
            return Err(format!(
                "filter candidate removes observation {:?} from an unknown image",
                key
            ));
        }
        let state = store
            .observations
            .get(key)
            .ok_or_else(|| format!("filter candidate removes unknown observation {key:?}"))?;
        store.tracks.get(state.owner_track).ok_or_else(|| {
            format!(
                "filter candidate observation {:?} has unknown owner track {}",
                key, state.owner_track
            )
        })?;
        if !candidate.removed_track_ids.contains(&state.owner_track)
            && !replacement_tracks.contains(&state.owner_track)
        {
            return Err(format!(
                "filter candidate removes observation {:?} from an untouched track {}",
                key, state.owner_track
            ));
        }
        if retained_keys.contains(key) {
            return Err(format!(
                "filter candidate both removes and retains observation {key:?}"
            ));
        }
    }

    let mut required_landmark_tracks = replacement_tracks.clone();
    required_landmark_tracks.extend(candidate.removed_track_ids.iter().copied());
    let mut found_landmark_tracks = BTreeSet::new();
    for landmark in landmarks {
        if required_landmark_tracks.contains(&landmark.track_id) {
            found_landmark_tracks.insert(landmark.track_id);
        }
    }
    for track_id in &candidate.removed_track_ids {
        if !found_landmark_tracks.contains(track_id) {
            return Err(format!(
                "filter candidate removes track {track_id} without a landmark"
            ));
        }
    }

    Ok(())
}

fn apply_joint_rig_ba_filter_candidate(
    candidate: &JointRigBaFilteringCandidate,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<(), String> {
    // All fallible validation happens before the first mutation.  The actual
    // apply phase below only indexes entries proven to exist here, so a
    // rejected candidate cannot leave a partially updated model behind.
    validate_filter_candidate_for_apply(candidate, store, images, landmarks)?;
    apply_pose_overrides(images, &candidate.pose_overrides)?;
    for (index, replacement) in &candidate.landmark_updates {
        let landmark = landmarks
            .get_mut(*index)
            .expect("filter candidate landmark index was prevalidated");
        *landmark = replacement.clone();
    }
    for key in candidate.removed_observations.keys() {
        store.observations.remove(key);
    }
    for track_id in &candidate.removed_track_ids {
        let track = store
            .tracks
            .get_mut(*track_id)
            .expect("filter candidate removed track was prevalidated");
        track.observations.clear();
    }
    for replacement in candidate.landmark_updates.values() {
        let track_id = replacement.track_id;
        store
            .tracks
            .get_mut(track_id)
            .expect("replacement track was prevalidated")
            .observations = replacement.observations.clone();
    }
    landmarks.retain(|landmark| !candidate.removed_track_ids.contains(&landmark.track_id));
    Ok(())
}

fn validate_filtered_model(
    baseline_connectivity: &SupportConnectivity,
    store: &TrackStore,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    let current_connectivity =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    if !connectivity_not_split(baseline_connectivity, &current_connectivity) {
        return Err("final filtered model loses support or splits connectivity".to_owned());
    }
    for landmark in landmarks {
        if landmark.observations.len() < MIN_TRACK_OBSERVATIONS {
            return Err(format!(
                "final landmark {} has too few observations",
                landmark.track_id
            ));
        }
        let metrics = evaluate_landmark_metrics(
            landmark,
            &store.observations,
            images,
            cameras,
            &BTreeMap::new(),
        )?;
        if metrics.mean_error > MAX_MEAN_REPROJECTION_PX || metrics.max_error > MAX_REPROJECTION_PX
        {
            return Err(format!(
                "final landmark {} failed reprojection gate",
                landmark.track_id
            ));
        }
        let track = store
            .tracks
            .get(landmark.track_id)
            .ok_or_else(|| format!("final landmark {} has no source track", landmark.track_id))?;
        if track.observations != landmark.observations {
            return Err(format!(
                "final landmark {} and source track observations disagree",
                landmark.track_id
            ));
        }
        for key in &landmark.observations {
            let observation = store
                .observations
                .get(key)
                .ok_or_else(|| "final landmark references removed observation".to_owned())?;
            if observation.owner_track != landmark.track_id {
                return Err("final landmark observation owner disagrees".to_owned());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn run_joint_rig_ba_filtering(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
) -> Result<JointRigBaFilteringSummary, String> {
    run_joint_rig_ba_filtering_with_policy(manifest, store, images, cameras, landmarks, false)
}

fn print_joint_rig_ba_filter_log(pass: Option<usize>, line: std::fmt::Arguments<'_>) {
    if let Some(pass) = pass {
        println!("pass={pass} cost_scope=overlapping_window_events {line}");
    } else {
        println!("{line}");
    }
}

fn print_joint_rig_ba_filter_summary(
    summary: &JointRigBaFilteringSummary,
    preserve_optimized_points: bool,
    pass: Option<usize>,
) {
    if preserve_optimized_points {
        print_joint_rig_ba_filter_log(pass, format_args!(
            "joint_rig_ba_filter_observations windows_considered={} windows_accepted={} windows_skipped={} selected_landmarks={} selected_observations={} max_referenced_frames={} max_free_frames={} max_iterations={} converged_windows={} removed_observations={} removed_tracks={} retriangulated_tracks={} raw_preserved_tracks={} dlt_attempted_tracks={} raw_fallback_reasons={} raw_counts_are_window_events=true full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9}",
            summary.windows_considered,
            summary.windows_accepted,
            summary.windows_skipped,
            summary.selected_landmarks,
            summary.selected_observations,
            summary.max_referenced_frames,
            summary.max_free_frames,
            summary.max_iterations,
            summary.converged_windows,
            summary.removed_observations,
            summary.removed_tracks,
            summary.retriangulated_tracks,
            summary.raw_preserved_tracks,
            summary.dlt_attempted_tracks,
            format_raw_fallback_reason_counts(&summary.raw_fallback_reason_counts),
            summary.full_pre_ba_cost,
            summary.full_post_ba_cost,
            summary.retained_pre_ba_cost,
            summary.retained_post_filter_cost,
        ));
    } else {
        print_joint_rig_ba_filter_log(pass, format_args!(
            "joint_rig_ba_filter_observations windows_considered={} windows_accepted={} windows_skipped={} selected_landmarks={} selected_observations={} max_referenced_frames={} max_free_frames={} max_iterations={} converged_windows={} removed_observations={} removed_tracks={} retriangulated_tracks={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9}",
            summary.windows_considered,
            summary.windows_accepted,
            summary.windows_skipped,
            summary.selected_landmarks,
            summary.selected_observations,
            summary.max_referenced_frames,
            summary.max_free_frames,
            summary.max_iterations,
            summary.converged_windows,
            summary.removed_observations,
            summary.removed_tracks,
            summary.retriangulated_tracks,
            summary.full_pre_ba_cost,
            summary.full_post_ba_cost,
            summary.retained_pre_ba_cost,
            summary.retained_post_filter_cost,
        ));
    }
}

fn print_joint_rig_ba_filter_pass_support(
    pass: usize,
    images: &BTreeMap<u64, GlobalImage>,
    landmarks: &[LandmarkOutput],
) -> Result<(), String> {
    let support =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    println!(
        "pass={pass} cost_scope=overlapping_window_events support_images={} support_frames={} components={}",
        support.supported_images.len(),
        support.supported_frames.len(),
        support.component_count
    );
    Ok(())
}

fn run_joint_rig_ba_filtering_with_policy(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
    preserve_optimized_points: bool,
) -> Result<JointRigBaFilteringSummary, String> {
    run_joint_rig_ba_filtering_with_policy_and_pass(
        manifest,
        store,
        images,
        cameras,
        landmarks,
        preserve_optimized_points,
        None,
    )
}

fn run_joint_rig_ba_filtering_with_policy_and_pass(
    manifest: &RigManifest,
    store: &mut TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut Vec<LandmarkOutput>,
    preserve_optimized_points: bool,
    pass: Option<usize>,
) -> Result<JointRigBaFilteringSummary, String> {
    let baseline_connectivity =
        support_connectivity_for_landmarks(landmarks, &BTreeMap::new(), &BTreeSet::new(), images)?;
    let mut frame_ids = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<Vec<_>>();
    frame_ids.sort_unstable();
    frame_ids.dedup();
    let mut summary = JointRigBaFilteringSummary::default();
    for start in (0..frame_ids.len()).step_by(JOINT_BA_WINDOW_STRIDE) {
        let end = (start + JOINT_BA_WINDOW_LENGTH).min(frame_ids.len());
        let active_frames = frame_ids[start..end]
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        summary.windows_considered += 1;
        let selected_result = selected_landmarks_for_frames(landmarks, &active_frames, images);
        let window_counts = match selected_result.as_ref() {
            Ok(selected) => {
                joint_ba_window_counts_with_selected(landmarks, images, &active_frames, selected)
            }
            Err(error) => Err(error.clone()),
        };
        if let Ok((selected_count, observation_count, referenced_count, free_count)) =
            window_counts.as_ref()
        {
            summary.selected_landmarks += *selected_count;
            summary.selected_observations += *observation_count;
            summary.max_referenced_frames = summary.max_referenced_frames.max(*referenced_count);
            summary.max_free_frames = summary.max_free_frames.max(*free_count);
        }
        let candidate_result = match selected_result {
            Ok(selected) => build_joint_rig_ba_filter_candidate_with_selected(
                manifest,
                store,
                images,
                cameras,
                landmarks,
                &active_frames,
                &selected,
                &baseline_connectivity,
                preserve_optimized_points,
            ),
            Err(reason) => Err(JointRigBaSkip::Invalid(reason)),
        };
        match candidate_result {
            Ok(candidate) => {
                let reason_counts = filter_reason_counts(&candidate.removed_observations);
                apply_joint_rig_ba_filter_candidate(&candidate, store, images, landmarks)?;
                summary.windows_accepted += 1;
                summary.max_iterations = summary.max_iterations.max(candidate.iterations);
                if candidate.converged {
                    summary.converged_windows += 1;
                }
                summary.removed_observations += candidate.removed_observations.len();
                summary.removed_tracks += candidate.removed_track_ids.len();
                summary.retriangulated_tracks += candidate.retriangulated_tracks;
                summary.raw_preserved_tracks += candidate.raw_preserved_tracks;
                summary.dlt_attempted_tracks += candidate.dlt_attempted_tracks;
                summary.full_pre_ba_cost += candidate.full_pre_ba_cost;
                summary.full_post_ba_cost += candidate.full_post_ba_cost;
                summary.retained_pre_ba_cost += candidate.retained_pre_ba_cost;
                summary.retained_post_filter_cost += candidate.retained_post_filter_cost;
                for (reason, count) in &reason_counts {
                    *summary.removed_reason_counts.entry(*reason).or_default() += *count;
                }
                for (reason, count) in &candidate.raw_fallback_reason_counts {
                    *summary
                        .raw_fallback_reason_counts
                        .entry(*reason)
                        .or_default() += *count;
                }
                if preserve_optimized_points {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} removed_observations={} removed_tracks={} retriangulated_tracks={} raw_preserved_tracks={} dlt_attempted_tracks={} raw_fallback_reasons={} raw_counts_are_window_events=true reasons={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9} iterations={} converged={}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        candidate.selected_landmarks,
                        candidate.selected_observations,
                        candidate.referenced_frames,
                        candidate.free_frames,
                        candidate.removed_observations.len(),
                        candidate.removed_track_ids.len(),
                        candidate.retriangulated_tracks,
                        candidate.raw_preserved_tracks,
                        candidate.dlt_attempted_tracks,
                        format_raw_fallback_reason_counts(&candidate.raw_fallback_reason_counts),
                        format_filter_reason_counts(&reason_counts),
                        candidate.full_pre_ba_cost,
                        candidate.full_post_ba_cost,
                        candidate.retained_pre_ba_cost,
                        candidate.retained_post_filter_cost,
                        candidate.iterations,
                        candidate.converged,
                    ));
                } else {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} removed_observations={} removed_tracks={} retriangulated_tracks={} reasons={} full_pre_ba_cost={:.9} full_post_ba_cost={:.9} retained_pre_ba_cost={:.9} retained_post_filter_cost={:.9} iterations={} converged={}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        candidate.selected_landmarks,
                        candidate.selected_observations,
                        candidate.referenced_frames,
                        candidate.free_frames,
                        candidate.removed_observations.len(),
                        candidate.removed_track_ids.len(),
                        candidate.retriangulated_tracks,
                        format_filter_reason_counts(&reason_counts),
                        candidate.full_pre_ba_cost,
                        candidate.full_post_ba_cost,
                        candidate.retained_pre_ba_cost,
                        candidate.retained_post_filter_cost,
                        candidate.iterations,
                        candidate.converged,
                    ));
                }
            }
            Err(reason) => {
                summary.windows_skipped += 1;
                let (selected_count, observation_count, referenced_count, free_count) =
                    window_counts.unwrap_or_default();
                if preserve_optimized_points {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} raw_preserved_tracks=0 dlt_attempted_tracks=0 raw_fallback_reasons=none raw_counts_are_window_events=true reason={reason}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        selected_count,
                        observation_count,
                        referenced_count,
                        free_count,
                    ));
                } else {
                    print_joint_rig_ba_filter_log(pass, format_args!(
                        "joint_rig_ba_filter_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} reason={reason}",
                        active_frames.first().copied().unwrap_or_default(),
                        active_frames.last().copied().unwrap_or_default(),
                        active_frames.len(),
                        selected_count,
                        observation_count,
                        referenced_count,
                        free_count,
                    ));
                }
            }
        }
    }
    validate_filtered_model(&baseline_connectivity, store, images, cameras, landmarks)?;
    Ok(summary)
}

fn run_joint_rig_ba(
    manifest: &RigManifest,
    store: &TrackStore,
    images: &mut BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &mut [LandmarkOutput],
) -> Result<JointRigBaSummary, String> {
    let mut frame_ids = images
        .values()
        .map(|image| image.atlas.frame_id)
        .collect::<Vec<_>>();
    frame_ids.sort_unstable();
    frame_ids.dedup();
    let mut summary = JointRigBaSummary::default();
    for start in (0..frame_ids.len()).step_by(JOINT_BA_WINDOW_STRIDE) {
        let end = (start + JOINT_BA_WINDOW_LENGTH).min(frame_ids.len());
        let active_frames = frame_ids[start..end]
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        assert!(active_frames.len() <= JOINT_BA_MAX_FREE_FRAMES);
        summary.windows_considered += 1;
        let window_counts = joint_ba_window_counts(landmarks, images, &active_frames);
        if let Ok((selected_count, observation_count, referenced_count, free_count)) =
            window_counts.as_ref()
        {
            summary.selected_landmarks += *selected_count;
            summary.selected_observations += *observation_count;
            summary.max_referenced_frames = summary.max_referenced_frames.max(*referenced_count);
            summary.max_free_frames = summary.max_free_frames.max(*free_count);
        }
        match build_joint_rig_ba_window(manifest, store, images, cameras, landmarks, &active_frames)
        {
            Ok(update) => {
                apply_pose_overrides(images, &update.pose_overrides)?;
                for (index, landmark) in update.landmark_updates {
                    landmarks[index] = landmark;
                }
                summary.windows_accepted += 1;
                summary.max_iterations = summary.max_iterations.max(update.iterations);
                if update.converged {
                    summary.converged_windows += 1;
                }
                if summary.windows_accepted == 1 {
                    summary.max_solver_initial_cost = update.solver_initial_cost;
                    summary.min_solver_final_cost = update.solver_final_cost;
                } else {
                    summary.max_solver_initial_cost = summary
                        .max_solver_initial_cost
                        .max(update.solver_initial_cost);
                    summary.min_solver_final_cost =
                        summary.min_solver_final_cost.min(update.solver_final_cost);
                }
                summary.final_cost = update.final_cost;
                println!(
                    "joint_rig_ba_window start_frame={} end_frame={} active_frames={} status=accepted selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} solver_initial_cost={:.9} solver_final_cost={:.9} validated_initial_cost={:.9} validated_final_cost={:.9} iterations={} converged={}",
                    active_frames.first().copied().unwrap_or_default(),
                    active_frames.last().copied().unwrap_or_default(),
                    active_frames.len(),
                    update.selected_landmarks,
                    update.selected_observations,
                    update.referenced_frames,
                    update.free_frames,
                    update.solver_initial_cost,
                    update.solver_final_cost,
                    update.initial_cost,
                    update.final_cost,
                    update.iterations,
                    update.converged,
                );
            }
            Err(reason) => {
                summary.windows_skipped += 1;
                let (selected_count, observation_count, referenced_count, free_count) =
                    window_counts.unwrap_or_default();
                println!(
                    "joint_rig_ba_window start_frame={} end_frame={} active_frames={} status=skipped selected_landmarks={} selected_observations={} referenced_frames={} free_frames={} reason={reason}",
                    active_frames.first().copied().unwrap_or_default(),
                    active_frames.last().copied().unwrap_or_default(),
                    active_frames.len(),
                    selected_count,
                    observation_count,
                    referenced_count,
                    free_count,
                );
            }
        }
    }
    Ok(summary)
}

fn has_observable_parallax(
    keys: &[ObservationKey],
    observations: &BTreeMap<ObservationKey, ObservationState>,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    pose_overrides: &BTreeMap<u64, Pose>,
) -> Result<bool, String> {
    let mut rays = Vec::with_capacity(keys.len());
    for key in keys {
        let image = images
            .get(&key.global_image_id)
            .ok_or_else(|| "track references unknown atlas image".to_owned())?;
        let camera = cameras
            .get(&image.atlas.camera_id)
            .ok_or_else(|| "track references unknown camera".to_owned())?;
        let state = observations
            .get(key)
            .ok_or_else(|| "track references unknown observation".to_owned())?;
        let normalized = camera
            .normalize_pixel(&state.xy)
            .ok_or_else(|| "camera model cannot normalize pixel".to_owned())?;
        let bearing_camera = Vector3::new(normalized.x, normalized.y, 1.0).normalize();
        let pose = pose_for_image(image, pose_overrides);
        let bearing_world = pose
            .camera_to_world()
            .rotation
            .transform_vector(&bearing_camera);
        rays.push((pose.camera_center_world(), bearing_world));
    }
    for (index, (center_a, ray_a)) in rays.iter().enumerate() {
        for (center_b, ray_b) in rays.iter().skip(index + 1) {
            let baseline = (center_a - center_b).norm();
            let ray_cross = ray_a.cross(ray_b).norm();
            if baseline > MIN_BASELINE_M && ray_cross > MIN_RAY_CROSS_NORM {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn landmark_observation_count(landmarks: &[LandmarkOutput]) -> usize {
    landmarks
        .iter()
        .map(|landmark| landmark.observations.len())
        .sum()
}

fn canonical_landmark_indices(landmarks: &[LandmarkOutput]) -> Vec<usize> {
    let mut indices = (0..landmarks.len()).collect::<Vec<_>>();
    indices.sort_by_key(|index| {
        (
            landmarks[*index]
                .observations
                .first()
                .copied()
                .unwrap_or(ObservationKey {
                    global_image_id: u64::MAX,
                    keypoint_index: usize::MAX,
                }),
            landmarks[*index].track_id,
        )
    });
    indices
}

fn write_model(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
) -> Result<SupportSummary, String> {
    write_model_ordered(out_dir, images, cameras, landmarks, stats, false)
}

fn write_model_canonical(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
) -> Result<SupportSummary, String> {
    write_model_ordered(out_dir, images, cameras, landmarks, stats, true)
}

fn write_model_ordered(
    out_dir: &Path,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
    landmarks: &[LandmarkOutput],
    stats: &IngestStats,
    canonical_order: bool,
) -> Result<SupportSummary, String> {
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("create output {}: {error}", out_dir.display()))?;
    let indices = if canonical_order {
        canonical_landmark_indices(landmarks)
    } else {
        (0..landmarks.len()).collect::<Vec<_>>()
    };
    let ordered_landmarks = indices
        .iter()
        .map(|index| &landmarks[*index])
        .collect::<Vec<_>>();
    let used_cameras = images
        .values()
        .map(|image| image.atlas.camera_id)
        .collect::<BTreeSet<_>>();
    let mut cameras_text = String::from("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n");
    for camera_id in used_cameras {
        let camera = cameras
            .get(&camera_id)
            .ok_or_else(|| format!("missing output camera {camera_id}"))?;
        let model = camera_model_name(&camera.model)?;
        write!(
            cameras_text,
            "{} {} {} {}",
            camera.id, model, camera.width, camera.height
        )
        .unwrap();
        for parameter in &camera.params {
            write!(cameras_text, " {}", format_f64(*parameter)).unwrap();
        }
        cameras_text.push('\n');
    }

    let mut point_ids = BTreeMap::<ObservationKey, u64>::new();
    let mut points_text =
        String::from("# POINT3D_ID X Y Z R G B ERROR TRACK[] as IMAGE_ID POINT2D_IDX\n");
    for (index, landmark) in ordered_landmarks.iter().enumerate() {
        let point_id = index as u64 + 1;
        for key in &landmark.observations {
            if point_ids.insert(*key, point_id).is_some() {
                return Err(format!(
                    "observation {:?} belongs to multiple output landmarks",
                    key
                ));
            }
        }
        writeln!(
            points_text,
            "{} {} {} {} 255 255 255 {} {}",
            point_id,
            format_f64(landmark.position.x),
            format_f64(landmark.position.y),
            format_f64(landmark.position.z),
            format_f64(landmark.mean_error),
            landmark
                .observations
                .iter()
                .map(|key| format!("{} {}", key.global_image_id, key.keypoint_index))
                .collect::<Vec<_>>()
                .join(" ")
        )
        .unwrap();
    }

    let mut images_text = String::from(
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as X Y POINT3D_ID\n",
    );
    let mut image_support = BTreeMap::<u64, usize>::new();
    for key in point_ids.keys() {
        *image_support.entry(key.global_image_id).or_default() += 1;
    }
    let mut frame_support = BTreeMap::<u64, (usize, usize)>::new();
    for image in images.values() {
        let entry = frame_support.entry(image.atlas.frame_id).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += image_support
            .get(&image.atlas.global_image_id)
            .copied()
            .unwrap_or(0);
        validate_image_name(&image.atlas.name)?;
        let quaternion = image.atlas.pose.world_to_camera.rotation.quaternion();
        let translation = image.atlas.pose.world_to_camera.translation;
        writeln!(
            images_text,
            "{} {} {} {} {} {} {} {} {} {}",
            image.atlas.global_image_id,
            format_f64(quaternion.w),
            format_f64(quaternion.i),
            format_f64(quaternion.j),
            format_f64(quaternion.k),
            format_f64(translation.x),
            format_f64(translation.y),
            format_f64(translation.z),
            image.atlas.camera_id,
            image.atlas.name
        )
        .unwrap();
        let mut tokens = Vec::with_capacity(image.keypoints.len());
        for (keypoint_index, xy) in image.keypoints.iter().enumerate() {
            let key = ObservationKey {
                global_image_id: image.atlas.global_image_id,
                keypoint_index,
            };
            let point_id = point_ids.get(&key).copied().unwrap_or(0);
            let point_text = if point_id == 0 {
                "-1".to_owned()
            } else {
                point_id.to_string()
            };
            tokens.push(format!(
                "{} {} {}",
                format_f64(xy.x),
                format_f64(xy.y),
                point_text
            ));
        }
        images_text.push_str(&tokens.join(" "));
        images_text.push('\n');
    }
    fs::write(out_dir.join("cameras.txt"), cameras_text)
        .map_err(|error| format!("write cameras.txt: {error}"))?;
    fs::write(out_dir.join("images.txt"), images_text)
        .map_err(|error| format!("write images.txt: {error}"))?;
    fs::write(out_dir.join("points3D.txt"), points_text)
        .map_err(|error| format!("write points3D.txt: {error}"))?;
    let mut image_support_text =
        String::from("# GLOBAL_IMAGE_ID FRAME_ID SENSOR_INDEX NAME RETAINED_OBSERVATIONS\n");
    let mut supported_image_count = 0;
    for image in images.values() {
        let support = image_support
            .get(&image.atlas.global_image_id)
            .copied()
            .unwrap_or(0);
        if support > 0 {
            supported_image_count += 1;
        }
        writeln!(
            image_support_text,
            "{} {} {} {} {}",
            image.atlas.global_image_id,
            image.atlas.frame_id,
            image.atlas.sensor_index,
            image.atlas.name,
            support
        )
        .unwrap();
    }
    let mut frame_support_text = String::from("# FRAME_ID IMAGE_COUNT RETAINED_OBSERVATIONS\n");
    let mut supported_frame_count = 0;
    for (frame_id, (image_count, support)) in &frame_support {
        if *support > 0 {
            supported_frame_count += 1;
        }
        writeln!(
            frame_support_text,
            "{} {} {}",
            frame_id, image_count, support
        )
        .unwrap();
    }
    fs::write(out_dir.join("image_support.tsv"), image_support_text)
        .map_err(|error| format!("write image_support.tsv: {error}"))?;
    fs::write(out_dir.join("frame_support.tsv"), frame_support_text)
        .map_err(|error| format!("write frame_support.tsv: {error}"))?;
    let support_summary = SupportSummary {
        atlas_image_count: images.len(),
        supported_image_count,
        zero_support_image_count: images.len() - supported_image_count,
        atlas_frame_count: frame_support.len(),
        supported_frame_count,
        zero_support_frame_count: frame_support.len() - supported_frame_count,
    };
    let summary = format!(
        "source_nodes\t{}\nsource_points\t{}\nsource_observations\t{}\naccepted_candidates\t{}\nextended_candidates\t{}\nduplicate_observations\t{}\nrejected_candidates\t{}\nrejected_missing_atlas_images\t{}\nfiltered_out_of_component_observations\t{}\nrejected_short_candidates\t{}\nrejected_same_image\t{}\nrejected_track_conflict\t{}\nrejected_triangulation\t{}\ntriangulation_sampled_tracks\t{}\ntriangulation_sampled_observations\t{}\natlas_poses\t{}\nsupported_images\t{}\nzero_support_images\t{}\natlas_frames\t{}\nsupported_frames\t{}\nzero_support_frames\t{}\noutput_landmarks\t{}\noutput_observations\t{}\n",
        stats.source_nodes,
        stats.source_points,
        stats.source_observations,
        stats.accepted_candidates,
        stats.extended_candidates,
        stats.duplicate_observations,
        stats.rejected_candidates,
        stats.rejected_missing_atlas_images,
        stats.filtered_out_of_component_observations,
        stats.rejected_short_candidates,
        stats.rejected_same_image,
        stats.rejected_track_conflict,
        stats.rejected_triangulation,
        stats.triangulation_sampled_tracks,
        stats.triangulation_sampled_observations,
        support_summary.atlas_image_count,
        support_summary.supported_image_count,
        support_summary.zero_support_image_count,
        support_summary.atlas_frame_count,
        support_summary.supported_frame_count,
        support_summary.zero_support_frame_count,
        stats.output_landmarks,
        stats.output_observations,
    );
    fs::write(out_dir.join("integration_summary.tsv"), summary)
        .map_err(|error| format!("write integration_summary.tsv: {error}"))?;
    Ok(support_summary)
}

fn require_nonempty_landmarks(
    landmarks: &[LandmarkOutput],
    observation_count: usize,
) -> Result<(), String> {
    if landmarks.is_empty() || observation_count == 0 {
        return Err(
            "no landmarks survived conservative ownership, parallax, and reprojection gates; refusing to publish a pose-only model"
                .to_owned(),
        );
    }
    Ok(())
}

fn validate_output_cameras(
    manifest: &RigManifest,
    images: &BTreeMap<u64, GlobalImage>,
    cameras: &BTreeMap<u64, Camera>,
) -> Result<(), String> {
    let mut frame_poses = BTreeMap::<u64, Pose>::new();
    for image in images.values() {
        let assignment = manifest
            .assignments
            .get(&image.atlas.name)
            .expect("atlas images were bound to manifest");
        let sensor = manifest
            .sensors
            .get(&assignment.sensor_index)
            .expect("manifest sensor index was validated");
        if image.atlas.frame_id != assignment.frame_id
            || image.atlas.sensor_index != assignment.sensor_index
        {
            return Err(format!(
                "atlas image {:?} frame/sensor assignment disagrees with manifest",
                image.atlas.name
            ));
        }
        let camera = cameras.get(&image.atlas.camera_id).ok_or_else(|| {
            format!(
                "atlas image {:?} has no camera definition",
                image.atlas.name
            )
        })?;
        if camera.id != sensor.camera_id
            || camera.width != sensor.width
            || camera.height != sensor.height
            || !camera_matches_manifest(camera, sensor)
        {
            return Err(format!(
                "atlas image {:?} camera calibration is not manifest-compatible",
                image.atlas.name
            ));
        }
        if !sensor
            .sensor_from_rig
            .translation
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(format!(
                "sensor {} has non-finite extrinsic",
                assignment.sensor_index
            ));
        }
        camera_model_name(&camera.model)?;
        let rig_pose = Pose {
            world_to_camera: sensor
                .sensor_from_rig
                .inverse()
                .compose(&image.atlas.pose.world_to_camera),
        };
        if let Some(previous) = frame_poses.insert(image.atlas.frame_id, rig_pose.clone()) {
            let center_error =
                (previous.camera_center_world() - rig_pose.camera_center_world()).norm();
            let rotation_error = previous
                .world_to_camera
                .rotation
                .rotation_to(&rig_pose.world_to_camera.rotation)
                .angle()
                .to_degrees();
            if !center_error.is_finite()
                || !rotation_error.is_finite()
                || center_error > MAX_RIG_CENTER_DISAGREEMENT_M
                || rotation_error > MAX_RIG_ROTATION_DISAGREEMENT_DEG
            {
                return Err(format!(
                    "atlas frame {} sensor poses disagree: centre={center_error:.9}m rotation={rotation_error:.9}deg",
                    image.atlas.frame_id
                ));
            }
        }
    }
    Ok(())
}

fn camera_matches_manifest(camera: &Camera, sensor: &SensorCalibration) -> bool {
    if !matches!(camera.model, CameraModel::Pinhole) || camera.params.len() != 4 {
        return false;
    }
    let expected = [sensor.fx, sensor.fy, sensor.cx, sensor.cy];
    camera.params[..4]
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= INTRINSIC_TOLERANCE)
}

fn bounded_sample(keys: &[ObservationKey], max_count: usize) -> Vec<ObservationKey> {
    if keys.len() <= max_count {
        return keys.to_vec();
    }
    let mut result = Vec::with_capacity(max_count);
    for index in 0..max_count {
        let source_index = index * (keys.len() - 1) / (max_count - 1);
        if result.last().copied() != Some(keys[source_index]) {
            result.push(keys[source_index]);
        }
    }
    result
}

fn points_close(left: &Point2<f64>, right: &Point2<f64>) -> bool {
    (left - right).norm() <= XY_TOLERANCE_PX
        && left.coords.iter().all(|value| value.is_finite())
        && right.coords.iter().all(|value| value.is_finite())
}

fn parse_points2d(
    line: &str,
    path: &Path,
    line_number: usize,
) -> Result<Vec<SourceKeypoint>, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() % 3 != 0 {
        return Err(format!(
            "COLMAP {}:{} POINTS2D is not triples",
            path.display(),
            line_number
        ));
    }
    let mut points = Vec::with_capacity(fields.len() / 3);
    for triple in fields.chunks_exact(3) {
        let x = parse_f64(triple[0], "POINTS2D x", path, line_number)?;
        let y = parse_f64(triple[1], "POINTS2D y", path, line_number)?;
        let point_id = triple[2].parse::<i64>().map_err(|error| {
            format!(
                "COLMAP {}:{} invalid POINT3D_ID: {error}",
                path.display(),
                line_number
            )
        })?;
        if point_id < -1 {
            return Err(format!(
                "COLMAP {}:{} POINT3D_ID is below -1",
                path.display(),
                line_number
            ));
        }
        points.push(SourceKeypoint {
            xy: Point2::new(x, y),
            point3d_id: (point_id >= 0).then_some(point_id as u64),
        });
    }
    Ok(points)
}

fn next_data_line<'a>(lines: &'a [&'a str], index: &mut usize) -> Option<(usize, &'a str)> {
    while *index < lines.len() {
        let line_number = *index + 1;
        let line = lines[*index].trim();
        *index += 1;
        if !line.is_empty() && !line.starts_with('#') {
            return Some((line_number, line));
        }
    }
    None
}

fn next_required_line<'a>(lines: &'a [&'a str], index: &mut usize) -> Option<(usize, &'a str)> {
    if *index >= lines.len() {
        return None;
    }
    let line_number = *index + 1;
    let line = lines[*index].trim();
    *index += 1;
    if line.starts_with('#') {
        None
    } else {
        Some((line_number, line))
    }
}

fn parse_quaternion(
    fields: &[&str],
    path: &Path,
    line: usize,
) -> Result<UnitQuaternion<f64>, String> {
    if fields.len() != 4 {
        return Err(format!(
            "COLMAP {}:{} quaternion requires four values",
            path.display(),
            line
        ));
    }
    let values = fields
        .iter()
        .map(|field| parse_f64(field, "quaternion", path, line))
        .collect::<Result<Vec<_>, _>>()?;
    let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
    let norm = quaternion.norm();
    if !norm.is_finite() || norm <= MIN_HOMOGENEOUS_SCALE {
        return Err(format!(
            "COLMAP {}:{} quaternion is invalid",
            path.display(),
            line
        ));
    }
    Ok(UnitQuaternion::new_normalize(quaternion))
}

fn parse_vector(fields: &[&str], path: &Path, line: usize) -> Result<Vector3<f64>, String> {
    if fields.len() != 3 {
        return Err(format!(
            "COLMAP {}:{} translation requires three values",
            path.display(),
            line
        ));
    }
    Ok(Vector3::new(
        parse_f64(fields[0], "translation", path, line)?,
        parse_f64(fields[1], "translation", path, line)?,
        parse_f64(fields[2], "translation", path, line)?,
    ))
}

fn parse_u64(value: &str, label: &str, path: &Path, line: usize) -> Result<u64, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_u32(value: &str, label: &str, path: &Path, line: usize) -> Result<u32, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_usize(value: &str, label: &str, path: &Path, line: usize) -> Result<usize, String> {
    value.parse().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })
}

fn parse_f64(value: &str, label: &str, path: &Path, line: usize) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|error| {
        format!(
            "{}:{} invalid {label} {value:?}: {error}",
            path.display(),
            line
        )
    })?;
    if !parsed.is_finite() {
        return Err(format!("{}:{} non-finite {label}", path.display(), line));
    }
    Ok(parsed)
}

fn resolve_path(base: &Path, value: &str) -> PathBuf {
    let path = PathBuf::from(value);
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn camera_model_name(model: &CameraModel) -> Result<&'static str, String> {
    match model {
        CameraModel::Pinhole => Ok("PINHOLE"),
        CameraModel::SimplePinhole => Ok("SIMPLE_PINHOLE"),
        CameraModel::SimpleRadial => Ok("SIMPLE_RADIAL"),
        CameraModel::Radial => Ok("RADIAL"),
        CameraModel::OpenCv => Ok("OPENCV"),
        CameraModel::Unknown(name) => Err(format!("unsupported camera model {name:?}")),
    }
}

fn validate_image_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte == 0)
    {
        return Err(format!(
            "image name is not representable in COLMAP text: {name:?}"
        ));
    }
    Ok(())
}

fn format_f64(value: f64) -> String {
    let formatted = format!("{value:.12}");
    formatted
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned()
}

fn percentile(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted.get(index.min(sorted.len() - 1)).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_parse_args(extra: &[&str]) -> Result<Args, String> {
        let mut args = vec![
            "example",
            "--rig-manifest",
            "r",
            "--nodes-tsv",
            "n",
            "--atlas-dir",
            "a",
            "--out-dir",
            "o",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        args.extend(extra.iter().map(|value| (*value).to_owned()));
        parse_args(args)
    }

    fn test_image(id: u64, name: &str, keypoints: Vec<Point2<f64>>) -> GlobalImage {
        GlobalImage {
            atlas: AtlasImage {
                global_image_id: id,
                frame_id: id,
                sensor_index: 0,
                name: name.to_owned(),
                camera_id: 1,
                pose: Pose::identity(),
            },
            keypoints,
        }
    }

    fn test_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 100.0, 100.0, 320.0, 240.0)
    }

    fn recovery_fixture(
        add_bad_second_sensor_observation: bool,
    ) -> (
        RigManifest,
        TrackStore,
        BTreeMap<u64, GlobalImage>,
        BTreeMap<u64, Camera>,
        Vec<LandmarkOutput>,
        Pose,
    ) {
        let camera0 = Camera::pinhole(1, 640, 480, 100.0, 101.0, 320.0, 240.0);
        let camera1 = Camera::pinhole(2, 640, 480, 102.0, 99.0, 321.0, 239.0);
        let sensor0 = SensorCalibration {
            camera_id: 1,
            width: 640,
            height: 480,
            fx: 100.0,
            fy: 101.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::identity(),
        };
        let sensor1 = SensorCalibration {
            camera_id: 2,
            width: 640,
            height: 480,
            fx: 102.0,
            fy: 99.0,
            cx: 321.0,
            cy: 239.0,
            sensor_from_rig: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.005, -0.01, 0.008)),
                Vector3::new(-0.35, 0.01, -0.02),
            ),
        };
        let manifest = RigManifest {
            sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
            assignments: BTreeMap::new(),
        };
        let truth_rig_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.04, -0.03, 0.02)),
            Vector3::new(0.25, -0.15, 0.35),
        );
        let bad_rig_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(-0.1, 0.08, -0.04)),
            Vector3::new(-0.8, 0.4, 0.2),
        );
        let sensor_pose = |sensor: &SensorCalibration, rig_pose: &Pose| Pose {
            world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
        };
        let mut images = BTreeMap::new();
        let mut insert_image = |global_image_id, frame_id, sensor_index, camera_id, pose| {
            images.insert(
                global_image_id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id,
                        frame_id,
                        sensor_index,
                        name: format!("{global_image_id}.png"),
                        camera_id,
                        pose,
                    },
                    keypoints: vec![Point2::new(0.0, 0.0); 7],
                },
            );
        };
        insert_image(1, 10, 0, 1, sensor_pose(&sensor0, &Pose::identity()));
        insert_image(2, 10, 1, 2, sensor_pose(&sensor1, &Pose::identity()));
        insert_image(3, 20, 0, 1, sensor_pose(&sensor0, &bad_rig_pose));
        insert_image(4, 20, 1, 2, sensor_pose(&sensor1, &bad_rig_pose));

        let mut states = BTreeMap::new();
        let mut tracks = Vec::new();
        let base_point = Point3::new(0.1, -0.2, 4.6);
        let base_xy0 = camera0
            .project(&images[&1].atlas.pose.transform_world_point(&base_point))
            .unwrap();
        let base_xy1 = camera1
            .project(&images[&2].atlas.pose.transform_world_point(&base_point))
            .unwrap();
        images.get_mut(&1).unwrap().keypoints[6] = base_xy0;
        images.get_mut(&2).unwrap().keypoints[6] = base_xy1;
        let base_keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 6,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 6,
            },
        ];
        for (key, xy) in [(base_keys[0], base_xy0), (base_keys[1], base_xy1)] {
            states.insert(key, ObservationState { xy, owner_track: 0 });
        }
        tracks.push(GlobalTrack {
            observations: base_keys.clone(),
        });
        let baseline_landmark = LandmarkOutput {
            track_id: 0,
            position: base_point,
            observations: base_keys,
            errors: vec![0.0, 0.0],
            rms_error: 0.0,
            mean_error: 0.0,
            max_error: 0.0,
            dlt_sample_count: 2,
        };

        let points = [
            Point3::new(-1.2, -0.6, 4.5),
            Point3::new(-0.7, 0.4, 5.1),
            Point3::new(-0.1, -0.3, 5.8),
            Point3::new(0.4, 0.5, 4.8),
            Point3::new(0.9, -0.4, 5.5),
            Point3::new(1.3, 0.2, 6.2),
        ];
        let target_pose0 = sensor_pose(&sensor0, &truth_rig_pose);
        let target_pose1 = sensor_pose(&sensor1, &truth_rig_pose);
        let point_count = points.len();
        for (index, point) in points.into_iter().enumerate() {
            let xy0 = camera0
                .project(&target_pose0.transform_world_point(&point))
                .unwrap();
            let xy1 = camera1
                .project(&target_pose1.transform_world_point(&point))
                .unwrap();
            let anchor_xy0 = camera0
                .project(&images[&1].atlas.pose.transform_world_point(&point))
                .unwrap();
            let anchor_xy1 = camera1
                .project(&images[&2].atlas.pose.transform_world_point(&point))
                .unwrap();
            images.get_mut(&1).unwrap().keypoints[index] = anchor_xy0;
            images.get_mut(&2).unwrap().keypoints[index] = anchor_xy1;
            images.get_mut(&4).unwrap().keypoints[index] = xy1;
            if add_bad_second_sensor_observation && index == point_count - 1 {
                images.get_mut(&3).unwrap().keypoints[index] = Point2::new(xy0.x + 120.0, xy0.y);
            }
            let key1 = ObservationKey {
                global_image_id: 1,
                keypoint_index: index,
            };
            let key2 = ObservationKey {
                global_image_id: 2,
                keypoint_index: index,
            };
            let key4 = ObservationKey {
                global_image_id: 4,
                keypoint_index: index,
            };
            let mut keys = vec![key1, key2, key4];
            states.insert(
                key1,
                ObservationState {
                    xy: images[&1].keypoints[index],
                    owner_track: index + 1,
                },
            );
            states.insert(
                key2,
                ObservationState {
                    xy: images[&2].keypoints[index],
                    owner_track: index + 1,
                },
            );
            states.insert(
                key4,
                ObservationState {
                    xy: xy1,
                    owner_track: index + 1,
                },
            );
            if add_bad_second_sensor_observation && index == points.len() - 1 {
                let key3 = ObservationKey {
                    global_image_id: 3,
                    keypoint_index: index,
                };
                keys.insert(2, key3);
                states.insert(
                    key3,
                    ObservationState {
                        xy: images[&3].keypoints[index],
                        owner_track: index + 1,
                    },
                );
            }
            keys.sort_unstable();
            tracks.push(GlobalTrack { observations: keys });
        }
        (
            manifest,
            TrackStore {
                observations: states,
                tracks,
            },
            images,
            BTreeMap::from([(1, camera0), (2, camera1)]),
            vec![baseline_landmark],
            truth_rig_pose,
        )
    }

    fn boundary_repair_fixture() -> (
        RigManifest,
        TrackStore,
        BTreeMap<u64, GlobalImage>,
        BTreeMap<u64, Camera>,
        Vec<LandmarkOutput>,
    ) {
        let camera0 = Camera::pinhole(1, 640, 480, 100.0, 101.0, 320.0, 240.0);
        let camera1 = Camera::pinhole(2, 640, 480, 102.0, 99.0, 321.0, 239.0);
        let sensor0 = SensorCalibration {
            camera_id: 1,
            width: 640,
            height: 480,
            fx: 100.0,
            fy: 101.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::identity(),
        };
        let sensor1 = SensorCalibration {
            camera_id: 2,
            width: 640,
            height: 480,
            fx: 102.0,
            fy: 99.0,
            cx: 321.0,
            cy: 239.0,
            sensor_from_rig: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.005, -0.01, 0.008)),
                Vector3::new(-0.35, 0.01, -0.02),
            ),
        };
        let manifest = RigManifest {
            sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
            assignments: BTreeMap::new(),
        };
        let rig_pose = Pose::identity();
        let mut images = BTreeMap::new();
        for (image_id, frame_id, sensor_index, sensor) in [
            (1, 10, 0, &sensor0),
            (2, 10, 1, &sensor1),
            (3, 30, 0, &sensor0),
            (4, 30, 1, &sensor1),
        ] {
            images.insert(
                image_id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id: image_id,
                        frame_id,
                        sensor_index,
                        name: format!("{frame_id}-{sensor_index}.png"),
                        camera_id: sensor.camera_id,
                        pose: Pose {
                            world_to_camera: sensor
                                .sensor_from_rig
                                .compose(&rig_pose.world_to_camera),
                        },
                    },
                    keypoints: vec![Point2::new(0.0, 0.0); 7],
                },
            );
        }
        let cameras = BTreeMap::from([(1, camera0.clone()), (2, camera1.clone())]);
        let mut observations = BTreeMap::new();
        let mut tracks = Vec::new();
        let mut landmarks = Vec::new();
        let mut add_track = |track_id: usize, point: Point3<f64>, keys: Vec<ObservationKey>| {
            for key in &keys {
                let image = images.get_mut(&key.global_image_id).unwrap();
                let camera = cameras.get(&image.atlas.camera_id).unwrap();
                let xy = camera
                    .project(&image.atlas.pose.transform_world_point(&point))
                    .unwrap();
                image.keypoints[key.keypoint_index] = xy;
                observations.insert(
                    *key,
                    ObservationState {
                        xy,
                        owner_track: track_id,
                    },
                );
            }
            tracks.push(GlobalTrack {
                observations: keys.clone(),
            });
            (point, keys)
        };
        let base10 = add_track(
            0,
            Point3::new(0.1, -0.2, 4.6),
            vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 6,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 6,
                },
            ],
        );
        let base30 = add_track(
            1,
            Point3::new(-0.2, 0.3, 5.2),
            vec![
                ObservationKey {
                    global_image_id: 3,
                    keypoint_index: 6,
                },
                ObservationKey {
                    global_image_id: 4,
                    keypoint_index: 6,
                },
            ],
        );
        landmarks.push(LandmarkOutput {
            track_id: 0,
            position: base10.0,
            observations: base10.1,
            errors: vec![0.0, 0.0],
            rms_error: 0.0,
            mean_error: 0.0,
            max_error: 0.0,
            dlt_sample_count: 2,
        });
        landmarks.push(LandmarkOutput {
            track_id: 1,
            position: base30.0,
            observations: base30.1,
            errors: vec![0.0, 0.0],
            rms_error: 0.0,
            mean_error: 0.0,
            max_error: 0.0,
            dlt_sample_count: 2,
        });
        let points = [
            Point3::new(-1.2, -0.6, 4.5),
            Point3::new(-0.7, 0.4, 5.1),
            Point3::new(-0.1, -0.3, 5.8),
            Point3::new(0.4, 0.5, 4.8),
            Point3::new(0.9, -0.4, 5.5),
            Point3::new(1.3, 0.2, 6.2),
        ];
        for (index, point) in points.into_iter().enumerate() {
            let keys = vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: index,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: index,
                },
                ObservationKey {
                    global_image_id: 3,
                    keypoint_index: index,
                },
            ];
            add_track(index + 2, point, keys);
        }
        (
            manifest,
            TrackStore {
                observations,
                tracks,
            },
            images,
            cameras,
            landmarks,
        )
    }

    fn joint_ba_fixture() -> (
        RigManifest,
        TrackStore,
        BTreeMap<u64, GlobalImage>,
        BTreeMap<u64, Camera>,
        Vec<LandmarkOutput>,
        BTreeSet<u64>,
    ) {
        let camera0 = Camera::pinhole(1, 640, 480, 180.0, 181.0, 320.0, 240.0);
        let camera1 = Camera::pinhole(2, 640, 480, 179.0, 180.0, 321.0, 239.0);
        let sensor0 = SensorCalibration {
            camera_id: 1,
            width: 640,
            height: 480,
            fx: 180.0,
            fy: 181.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::identity(),
        };
        let sensor1 = SensorCalibration {
            camera_id: 2,
            width: 640,
            height: 480,
            fx: 179.0,
            fy: 180.0,
            cx: 321.0,
            cy: 239.0,
            sensor_from_rig: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.015)),
                Vector3::new(-0.35, 0.02, -0.01),
            ),
        };
        let manifest = RigManifest {
            sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
            assignments: BTreeMap::new(),
        };
        let truth_poses = [
            (
                100,
                Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
            ),
            (
                140,
                Pose::from_world_to_camera(
                    UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.015, 0.02)),
                    Vector3::new(-0.35, 0.02, 0.03),
                ),
            ),
            (
                200,
                Pose::from_world_to_camera(
                    UnitQuaternion::from_scaled_axis(Vector3::new(-0.02, 0.01, -0.01)),
                    Vector3::new(-0.75, -0.03, 0.02),
                ),
            ),
        ];
        let mut images = BTreeMap::new();
        let mut image_for = BTreeMap::new();
        let mut global_image_id = 1;
        for (frame_id, truth_pose) in &truth_poses {
            let rig_pose = if *frame_id == 140 {
                Pose::from_world_to_camera(
                    truth_pose.world_to_camera.rotation
                        * UnitQuaternion::from_scaled_axis(Vector3::new(0.006, -0.004, 0.003)),
                    truth_pose.world_to_camera.translation + Vector3::new(0.04, -0.02, 0.015),
                )
            } else {
                truth_pose.clone()
            };
            for (sensor_index, sensor) in [(0usize, &sensor0), (1usize, &sensor1)] {
                let image_id = global_image_id;
                global_image_id += 1;
                let pose = Pose {
                    world_to_camera: sensor.sensor_from_rig.compose(&rig_pose.world_to_camera),
                };
                images.insert(
                    image_id,
                    GlobalImage {
                        atlas: AtlasImage {
                            global_image_id: image_id,
                            frame_id: *frame_id,
                            sensor_index,
                            name: format!("{frame_id}-{sensor_index}.png"),
                            camera_id: sensor.camera_id,
                            pose,
                        },
                        keypoints: vec![Point2::new(0.0, 0.0); 12],
                    },
                );
                image_for.insert((*frame_id, sensor_index), image_id);
            }
        }
        let points = (0..12)
            .map(|index| {
                Point3::new(
                    -0.9 + index as f64 * 0.16,
                    -0.45 + (index % 4) as f64 * 0.22,
                    4.0 + (index % 5) as f64 * 0.25,
                )
            })
            .collect::<Vec<_>>();
        let mut observations = BTreeMap::new();
        let mut tracks = Vec::new();
        let mut landmarks = Vec::new();
        for (point_index, point) in points.iter().enumerate() {
            let mut keys = Vec::new();
            for (frame_id, truth_pose) in &truth_poses {
                for (sensor_index, sensor, camera) in
                    [(0usize, &sensor0, &camera0), (1usize, &sensor1, &camera1)]
                {
                    let image_id = image_for[&(*frame_id, sensor_index)];
                    let xy = camera
                        .project(
                            &sensor
                                .sensor_from_rig
                                .transform_point(&truth_pose.transform_world_point(point)),
                        )
                        .unwrap();
                    images.get_mut(&image_id).unwrap().keypoints[point_index] = xy;
                    let key = ObservationKey {
                        global_image_id: image_id,
                        keypoint_index: point_index,
                    };
                    observations.insert(
                        key,
                        ObservationState {
                            xy,
                            owner_track: point_index,
                        },
                    );
                    keys.push(key);
                }
            }
            keys.sort_unstable();
            tracks.push(GlobalTrack {
                observations: keys.clone(),
            });
            landmarks.push(LandmarkOutput {
                track_id: point_index,
                position: Point3::from(point.coords + Vector3::new(0.015, -0.012, 0.025)),
                observations: keys,
                errors: vec![0.0; 6],
                rms_error: 0.0,
                mean_error: 0.0,
                max_error: 0.0,
                dlt_sample_count: 6,
            });
        }
        (
            manifest,
            TrackStore {
                observations,
                tracks,
            },
            images,
            BTreeMap::from([(1, camera0), (2, camera1)]),
            landmarks,
            BTreeSet::from([100, 140, 200]),
        )
    }

    #[test]
    fn args_require_all_paths_and_reject_unknown() {
        let error = parse_args(["example".to_owned(), "--rig-manifest".to_owned()]).unwrap_err();
        assert!(error.contains("requires PATH"));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--bad".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("unknown argument"));

        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
        ])
        .unwrap();
        assert!(!args.recover_zero_support_frames);
        assert!(!args.joint_rig_ba);
        assert!(!args.joint_rig_ba_filter_observations);
        assert!(!args.joint_rig_ba_preserve_optimized_points);
        assert_eq!(args.pre_ba_out_dir, None);
        assert_eq!(args.post_pass1_out_dir, None);
        assert_eq!(args.joint_rig_ba_filter_sweeps, 1);
        assert!(!args.diagnose_cross_boundary_pnp);
        assert_eq!(args.diagnostic_left_max_frame, None);
        assert!(!args.repair_cross_boundary);
        assert_eq!(args.repair_left_max_frame, None);
        let args = parse_args([
            "example".to_owned(),
            "--recover-zero-support-frames".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
        ])
        .unwrap();
        assert!(args.recover_zero_support_frames);
        assert!(!args.joint_rig_ba);
        let duplicate_error = parse_args([
            "example".to_owned(),
            "--recover-zero-support-frames".to_owned(),
            "--recover-zero-support-frames".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
        ])
        .unwrap_err();
        assert!(duplicate_error.contains("duplicate argument"));
        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba".to_owned(),
        ])
        .unwrap();
        assert!(args.joint_rig_ba);
        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba".to_owned(),
            "--joint-rig-ba-filter-observations".to_owned(),
        ])
        .unwrap();
        assert!(args.joint_rig_ba_filter_observations);
        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba".to_owned(),
            "--joint-rig-ba-filter-observations".to_owned(),
            "--joint-rig-ba-preserve-optimized-points".to_owned(),
        ])
        .unwrap();
        assert!(args.joint_rig_ba_preserve_optimized_points);
        assert_eq!(args.joint_rig_ba_filter_sweeps, 1);
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba-filter-observations".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --joint-rig-ba"));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba-preserve-optimized-points".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --joint-rig-ba and --joint-rig-ba-filter-observations"));

        let args = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-observations",
            "--joint-rig-ba-filter-sweeps",
            "2",
            "--post-pass1-out-dir",
            "pass1",
        ])
        .unwrap();
        assert_eq!(args.joint_rig_ba_filter_sweeps, 2);
        assert_eq!(args.post_pass1_out_dir, Some(PathBuf::from("pass1")));
        for invalid in ["0", "3", "255"] {
            let error = minimal_parse_args(&[
                "--joint-rig-ba",
                "--joint-rig-ba-filter-observations",
                "--joint-rig-ba-filter-sweeps",
                invalid,
                "--post-pass1-out-dir",
                "pass1",
            ])
            .unwrap_err();
            assert!(error.contains("accepts only 1 or 2"), "{invalid}: {error}");
        }
        let error = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-observations",
            "--joint-rig-ba-filter-sweeps",
            "2",
        ])
        .unwrap_err();
        assert!(error.contains("post-pass1-out-dir"));
        let error = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-sweeps",
            "2",
            "--post-pass1-out-dir",
            "pass1",
        ])
        .unwrap_err();
        assert!(error.contains("filter-observations"));
        let error = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-observations",
            "--joint-rig-ba-filter-sweeps",
            "2",
            "--post-pass1-out-dir",
            "pass1",
            "--joint-rig-ba-filter-sweeps",
            "1",
        ])
        .unwrap_err();
        assert!(error.contains("duplicate argument"));
        let error = minimal_parse_args(&[
            "--joint-rig-ba",
            "--joint-rig-ba-filter-observations",
            "--joint-rig-ba-filter-sweeps",
            "2",
            "--post-pass1-out-dir",
            "pass1",
            "--joint-rig-ba-preserve-optimized-points",
        ])
        .unwrap_err();
        assert!(error.contains("cannot be combined"));
        let error = minimal_parse_args(&["--post-pass1-out-dir", "pass1"]).unwrap_err();
        assert!(error.contains("filter-sweeps 2"));
        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba".to_owned(),
            "--pre-ba-out-dir".to_owned(),
            "checkpoint".to_owned(),
        ])
        .unwrap();
        assert_eq!(args.pre_ba_out_dir, Some(PathBuf::from("checkpoint")));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--pre-ba-out-dir".to_owned(),
            "checkpoint".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --joint-rig-ba"));
        let duplicate_error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--joint-rig-ba".to_owned(),
            "--joint-rig-ba".to_owned(),
        ])
        .unwrap_err();
        assert!(duplicate_error.contains("duplicate argument"));

        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--diagnose-cross-boundary-pnp".to_owned(),
            "--diagnostic-left-max-frame".to_owned(),
            "1999".to_owned(),
        ])
        .unwrap();
        assert!(args.diagnose_cross_boundary_pnp);
        assert_eq!(args.diagnostic_left_max_frame, Some(1999));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--diagnose-cross-boundary-pnp".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("diagnostic-left-max-frame"));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--diagnostic-left-max-frame".to_owned(),
            "1999".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --diagnose-cross-boundary-pnp"));

        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--repair-cross-boundary".to_owned(),
            "--repair-left-max-frame".to_owned(),
            "1999".to_owned(),
            "--recover-zero-support-frames".to_owned(),
        ])
        .unwrap();
        assert!(args.repair_cross_boundary);
        assert_eq!(args.repair_left_max_frame, Some(1999));
        assert!(args.recover_zero_support_frames);
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--repair-cross-boundary".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("repair-left-max-frame"));
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--repair-left-max-frame".to_owned(),
            "1999".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --repair-cross-boundary"));
        let args = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--repair-cross-boundary".to_owned(),
            "--repair-left-max-frame".to_owned(),
            "1999".to_owned(),
            "--joint-rig-ba".to_owned(),
        ])
        .unwrap();
        assert!(args.repair_cross_boundary);
        assert!(args.joint_rig_ba);
        let error = parse_args([
            "example".to_owned(),
            "--rig-manifest".to_owned(),
            "r".to_owned(),
            "--nodes-tsv".to_owned(),
            "n".to_owned(),
            "--atlas-dir".to_owned(),
            "a".to_owned(),
            "--out-dir".to_owned(),
            "o".to_owned(),
            "--repair-cross-boundary".to_owned(),
            "--repair-left-max-frame".to_owned(),
            "1999".to_owned(),
            "--diagnose-cross-boundary-pnp".to_owned(),
            "--diagnostic-left-max-frame".to_owned(),
            "1999".to_owned(),
        ])
        .unwrap_err();
        assert!(error.contains("mutually exclusive"));
    }

    #[test]
    fn pre_ba_order_view_is_canonical_without_mutating_ba_order() {
        let (_manifest, _store, _images, _cameras, mut landmarks, _active) = joint_ba_fixture();
        landmarks.reverse();
        let before = landmarks.clone();
        let indices = canonical_landmark_indices(&landmarks);
        assert_eq!(indices.len(), landmarks.len());
        assert_eq!(indices, (0..landmarks.len()).rev().collect::<Vec<_>>());
        assert_eq!(landmarks, before);
    }

    #[test]
    fn canonical_pass1_checkpoint_writer_does_not_mutate_filter_state() {
        let (_manifest, _store, images, cameras, mut landmarks, _active) = joint_ba_fixture();
        landmarks.reverse();
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_canonical_checkpoint_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        let stats = IngestStats {
            output_landmarks: landmarks.len(),
            output_observations: landmark_observation_count(&landmarks),
            ..IngestStats::default()
        };
        write_model_canonical(&root, &images, &cameras, &landmarks, &stats).unwrap();
        assert_eq!(images, before_images);
        assert_eq!(landmarks, before_landmarks);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn output_and_checkpoint_ancestor_collision_is_rejected() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_checkpoint_paths_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let args = Args {
            rig_manifest: root.join("manifest.tsv"),
            nodes_tsv: root.join("nodes.tsv"),
            atlas_dir: root.join("atlas"),
            out_dir: root.join("out"),
            pre_ba_out_dir: Some(root.join("out/checkpoint")),
            post_pass1_out_dir: None,
            recover_zero_support_frames: false,
            joint_rig_ba: true,
            joint_rig_ba_filter_observations: false,
            joint_rig_ba_preserve_optimized_points: false,
            joint_rig_ba_filter_sweeps: 1,
            diagnose_cross_boundary_pnp: false,
            diagnostic_left_max_frame: None,
            repair_cross_boundary: false,
            repair_left_max_frame: None,
        };
        let nodes = vec![NodeSpec {
            node_id: 1,
            window_start: 0,
            images_txt: root.join("source/images.txt"),
        }];
        let error = validate_write_destinations(&args, &nodes).unwrap_err();
        assert!(error.contains("ambiguous output destinations"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn post_pass1_checkpoint_requires_empty_nonoverlapping_destination() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_post_pass1_paths_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("source")).unwrap();
        fs::write(root.join("source/images.txt"), "").unwrap();
        let nodes = vec![NodeSpec {
            node_id: 1,
            window_start: 0,
            images_txt: root.join("source/images.txt"),
        }];
        let mut args = Args {
            rig_manifest: root.join("manifest.tsv"),
            nodes_tsv: root.join("nodes.tsv"),
            atlas_dir: root.join("atlas"),
            out_dir: root.join("out"),
            pre_ba_out_dir: None,
            post_pass1_out_dir: Some(root.join("post")),
            recover_zero_support_frames: false,
            joint_rig_ba: true,
            joint_rig_ba_filter_observations: true,
            joint_rig_ba_preserve_optimized_points: false,
            joint_rig_ba_filter_sweeps: 2,
            diagnose_cross_boundary_pnp: false,
            diagnostic_left_max_frame: None,
            repair_cross_boundary: false,
            repair_left_max_frame: None,
        };
        validate_write_destinations(&args, &nodes).unwrap();
        fs::create_dir_all(args.post_pass1_out_dir.as_ref().unwrap()).unwrap();
        fs::write(
            args.post_pass1_out_dir.as_ref().unwrap().join("stale"),
            "stale",
        )
        .unwrap();
        let error = validate_write_destinations(&args, &nodes).unwrap_err();
        assert!(error.contains("post-pass1-out-dir") && error.contains("new or empty"));
        args.post_pass1_out_dir = Some(root.join("out/post"));
        let error = validate_write_destinations(&args, &nodes).unwrap_err();
        assert!(error.contains("ambiguous output destinations"));
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn path_comparison_resolves_symlink_before_parent_dir() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_checkpoint_symlink_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("real")).unwrap();
        symlink(root.join("real"), root.join("link")).unwrap();
        assert!(paths_overlap(&root.join("link/../real"), &root.join("real")).unwrap());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn filtered_joint_ba_candidate_updates_transactionally_and_preserves_rig() {
        let (manifest, mut store, mut images, cameras, mut landmarks, active) = joint_ba_fixture();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let candidate = build_joint_rig_ba_filter_candidate(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
        )
        .unwrap();
        assert!(cost_non_increasing(
            candidate.full_pre_ba_cost,
            candidate.full_post_ba_cost
        ));
        assert!(cost_non_increasing(
            candidate.retained_pre_ba_cost,
            candidate.retained_post_filter_cost
        ));
        apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
            .unwrap();
        validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
        for image in images.values() {
            let sensor = &manifest.sensors[&image.atlas.sensor_index];
            let rig_pose =
                derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
            let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
            assert!(
                (expected.translation - image.atlas.pose.world_to_camera.translation).norm()
                    < 1.0e-8
            );
        }
    }

    #[test]
    fn filtered_apply_removes_observations_and_track_without_resurrection() {
        let (_manifest, mut store, mut images, cameras, mut landmarks, _active) =
            joint_ba_fixture();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let removed = landmarks[0].clone();
        let removed_observations = removed
            .observations
            .iter()
            .copied()
            .map(|key| (key, FilterObservationReason::TrackBelowMinimum))
            .collect();
        let candidate = JointRigBaFilteringCandidate {
            pose_overrides: BTreeMap::new(),
            landmark_updates: BTreeMap::new(),
            removed_observations,
            removed_track_ids: BTreeSet::from([removed.track_id]),
            selected_landmarks: 1,
            selected_observations: removed.observations.len(),
            referenced_frames: 3,
            free_frames: 1,
            full_pre_ba_cost: 0.0,
            full_post_ba_cost: 0.0,
            retained_pre_ba_cost: 0.0,
            retained_post_filter_cost: 0.0,
            retriangulated_tracks: 0,
            raw_preserved_tracks: 0,
            dlt_attempted_tracks: 0,
            raw_fallback_reason_counts: BTreeMap::new(),
            iterations: 0,
            converged: true,
        };
        apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
            .unwrap();
        assert!(store.tracks[removed.track_id].observations.is_empty());
        for key in removed.observations {
            assert!(!store.observations.contains_key(&key));
        }
        assert!(!landmarks
            .iter()
            .any(|landmark| landmark.track_id == removed.track_id));
        validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
    }

    #[test]
    fn filtered_apply_removes_a_partial_observation_and_updates_source_track() {
        let (_manifest, mut store, mut images, cameras, mut landmarks, _active) =
            joint_ba_fixture();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let old = landmarks[0].clone();
        let removed_key = old.observations[0];
        let replacement = triangulate_track(
            old.track_id,
            &GlobalTrack {
                observations: old.observations[1..].to_vec(),
            },
            &store.observations,
            &images,
            &cameras,
        )
        .unwrap();
        let candidate = JointRigBaFilteringCandidate {
            pose_overrides: BTreeMap::new(),
            landmark_updates: BTreeMap::from([(0, replacement.clone())]),
            removed_observations: BTreeMap::from([(
                removed_key,
                FilterObservationReason::ReprojectionOverMax,
            )]),
            removed_track_ids: BTreeSet::new(),
            selected_landmarks: 1,
            selected_observations: old.observations.len(),
            referenced_frames: 3,
            free_frames: 1,
            full_pre_ba_cost: 0.0,
            full_post_ba_cost: 0.0,
            retained_pre_ba_cost: 0.0,
            retained_post_filter_cost: 0.0,
            retriangulated_tracks: 1,
            raw_preserved_tracks: 0,
            dlt_attempted_tracks: 1,
            raw_fallback_reason_counts: BTreeMap::new(),
            iterations: 0,
            converged: true,
        };
        apply_joint_rig_ba_filter_candidate(&candidate, &mut store, &mut images, &mut landmarks)
            .unwrap();
        assert!(!store.observations.contains_key(&removed_key));
        assert_eq!(landmarks[0].observations, replacement.observations);
        assert_eq!(store.tracks[0].observations, replacement.observations);
        validate_filtered_model(&baseline, &store, &images, &cameras, &landmarks).unwrap();
    }

    #[test]
    fn filtered_apply_rejects_invalid_candidate_without_partial_mutation() {
        let (_manifest, mut store, mut images, _cameras, mut landmarks, _active) =
            joint_ba_fixture();
        let replacement = landmarks[0].clone();
        let candidate = JointRigBaFilteringCandidate {
            pose_overrides: BTreeMap::new(),
            landmark_updates: BTreeMap::from([
                (0, replacement.clone()),
                (landmarks.len() + 1, replacement),
            ]),
            removed_observations: BTreeMap::new(),
            removed_track_ids: BTreeSet::new(),
            selected_landmarks: 2,
            selected_observations: 0,
            referenced_frames: 0,
            free_frames: 0,
            full_pre_ba_cost: 0.0,
            full_post_ba_cost: 0.0,
            retained_pre_ba_cost: 0.0,
            retained_post_filter_cost: 0.0,
            retriangulated_tracks: 0,
            raw_preserved_tracks: 0,
            dlt_attempted_tracks: 0,
            raw_fallback_reason_counts: BTreeMap::new(),
            iterations: 0,
            converged: false,
        };
        let before_store = store.clone();
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let error = apply_joint_rig_ba_filter_candidate(
            &candidate,
            &mut store,
            &mut images,
            &mut landmarks,
        )
        .unwrap_err();
        assert!(error.contains("landmark index"));
        assert_eq!(store, before_store);
        assert_eq!(images, before_images);
        assert_eq!(landmarks, before_landmarks);
    }

    #[test]
    fn filtered_joint_ba_run_validates_final_model_and_reports_counts() {
        let (manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
        let summary = run_joint_rig_ba_filtering(
            &manifest,
            &mut store,
            &mut images,
            &cameras,
            &mut landmarks,
        )
        .unwrap();
        assert_eq!(summary.windows_considered, 1);
        assert_eq!(summary.windows_accepted, 1);
        assert_eq!(summary.windows_skipped, 0);
        assert!(summary.retriangulated_tracks > 0);
    }

    #[test]
    fn two_filter_sweeps_are_deterministic_and_preserve_support() {
        let (manifest, mut store, mut images, cameras, mut landmarks, _active) = joint_ba_fixture();
        let first = run_joint_rig_ba_filtering_with_policy_and_pass(
            &manifest,
            &mut store,
            &mut images,
            &cameras,
            &mut landmarks,
            false,
            Some(1),
        )
        .unwrap();
        let second = run_joint_rig_ba_filtering_with_policy_and_pass(
            &manifest,
            &mut store,
            &mut images,
            &cameras,
            &mut landmarks,
            false,
            Some(2),
        )
        .unwrap();
        assert!(first.windows_accepted > 0);
        assert!(second.windows_accepted > 0);
        for image in images.values() {
            let sensor = &manifest.sensors[&image.atlas.sensor_index];
            let rig_pose =
                derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
            let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
            assert!(
                (expected.translation - image.atlas.pose.world_to_camera.translation).norm()
                    < 1.0e-8
            );
            let rotation_error =
                expected.rotation.inverse() * image.atlas.pose.world_to_camera.rotation;
            assert!(rotation_error.angle() < 1.0e-8);
        }
        let support = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        assert_eq!(support.supported_frames.len(), 3);

        let (
            manifest_again,
            mut store_again,
            mut images_again,
            cameras_again,
            mut landmarks_again,
            _active,
        ) = joint_ba_fixture();
        let first_again = run_joint_rig_ba_filtering_with_policy_and_pass(
            &manifest_again,
            &mut store_again,
            &mut images_again,
            &cameras_again,
            &mut landmarks_again,
            false,
            Some(1),
        )
        .unwrap();
        let second_again = run_joint_rig_ba_filtering_with_policy_and_pass(
            &manifest_again,
            &mut store_again,
            &mut images_again,
            &cameras_again,
            &mut landmarks_again,
            false,
            Some(2),
        )
        .unwrap();
        assert_eq!(first, first_again);
        assert_eq!(second, second_again);
        assert_eq!(store, store_again);
        assert_eq!(images, images_again);
        assert_eq!(landmarks, landmarks_again);
    }

    #[test]
    fn filtered_joint_ba_candidate_is_deterministic_and_rollback_is_sparse() {
        let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let first = build_joint_rig_ba_filter_candidate(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
        )
        .unwrap();
        let second = build_joint_rig_ba_filter_candidate(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
        )
        .unwrap();
        assert_eq!(first, second);

        let mut impossible_baseline = baseline.clone();
        impossible_baseline.component_count = 0;
        let error = build_joint_rig_ba_filter_candidate(
            &manifest,
            &store,
            &images,
            &cameras,
            &landmarks,
            &active,
            &impossible_baseline,
        )
        .unwrap_err();
        assert!(error.to_string().contains("connectivity"));
    }

    #[test]
    fn optimized_point_policy_is_flag_dependent_and_deterministic() {
        let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let disabled = build_joint_rig_ba_filter_candidate(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline,
        )
        .unwrap();
        assert_eq!(disabled.raw_preserved_tracks, 0);
        assert_eq!(disabled.raw_fallback_reason_counts, BTreeMap::new());

        let enabled = build_joint_rig_ba_filter_candidate_with_policy(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
        )
        .unwrap();
        let enabled_again = build_joint_rig_ba_filter_candidate_with_policy(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
        )
        .unwrap();
        let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
        let enabled_shared = build_joint_rig_ba_filter_candidate_with_selected(
            &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, true,
        )
        .unwrap();
        assert_eq!(enabled, enabled_again);
        assert_eq!(enabled, enabled_shared);
        assert!(enabled.raw_preserved_tracks > 0);
        assert!(enabled.dlt_attempted_tracks < enabled.selected_landmarks);
    }

    #[test]
    fn filtered_observation_depth_and_cost_gates_are_explicit() {
        let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
        let landmark = &landmarks[0];
        let key = landmark.observations[0];
        let image = &images[&key.global_image_id];
        let behind = image
            .atlas
            .pose
            .camera_to_world()
            .transform_point(&Point3::new(0.0, 0.0, -1.0));
        let error = classify_filter_observation(
            key,
            &behind,
            &store.observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert_eq!(error, FilterObservationReason::BehindCamera);
        assert!(cost_non_increasing(10.0, 10.0 + 1.0e-9));
        assert!(!cost_non_increasing(10.0, 10.1));
    }

    #[test]
    fn optimized_raw_point_is_retained_only_with_complete_valid_keys_and_metrics() {
        let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
        let track = &store.tracks[landmarks[0].track_id];
        let dlt = triangulate_track(
            landmarks[0].track_id,
            track,
            &store.observations,
            &images,
            &cameras,
        )
        .unwrap();
        let mut raw = dlt.clone();
        raw.position += Vector3::new(1.0e-4, -1.0e-4, 2.0e-4);
        let retained = retain_raw_ba_point_if_valid(
            &raw,
            &raw.observations,
            &store.observations,
            &images,
            &cameras,
            &BTreeMap::new(),
        )
        .unwrap();
        assert_eq!(retained.position, raw.position);
        assert_ne!(retained.position, dlt.position);
        assert_eq!(retained.observations, raw.observations);
        assert!(retained.mean_error <= MAX_MEAN_REPROJECTION_PX);
        assert!(retained.max_error <= MAX_REPROJECTION_PX);
        assert_eq!(retained.errors.len(), retained.observations.len());

        let changed_keys = raw.observations[..raw.observations.len() - 1].to_vec();
        assert_eq!(
            retain_raw_ba_point_if_valid(
                &raw,
                &changed_keys,
                &store.observations,
                &images,
                &cameras,
                &BTreeMap::new(),
            )
            .unwrap_err(),
            RawPointFallbackReason::ObservationKeysChanged
        );
    }

    #[test]
    fn optimized_raw_point_rejects_parallax_depth_and_mean_gates() {
        let (_manifest, store, images, cameras, landmarks, _active) = joint_ba_fixture();
        let raw = triangulate_track(
            landmarks[0].track_id,
            &store.tracks[landmarks[0].track_id],
            &store.observations,
            &images,
            &cameras,
        )
        .unwrap();

        let duplicate_key = raw.observations[0];
        let mut no_parallax = raw.clone();
        no_parallax.observations = vec![duplicate_key, duplicate_key];
        assert_eq!(
            retain_raw_ba_point_if_valid(
                &no_parallax,
                &no_parallax.observations,
                &store.observations,
                &images,
                &cameras,
                &BTreeMap::new(),
            )
            .unwrap_err(),
            RawPointFallbackReason::NoObservableParallax
        );

        let mut behind = raw.clone();
        let first_image = &images[&raw.observations[0].global_image_id];
        behind.position = first_image
            .atlas
            .pose
            .camera_to_world()
            .transform_point(&Point3::new(0.0, 0.0, -1.0));
        assert_eq!(
            retain_raw_ba_point_if_valid(
                &behind,
                &behind.observations,
                &store.observations,
                &images,
                &cameras,
                &BTreeMap::new(),
            )
            .unwrap_err(),
            RawPointFallbackReason::NonPositiveDepth
        );

        let mut shifted_observations = store.observations.clone();
        for key in &raw.observations {
            let image = &images[&key.global_image_id];
            let camera = &cameras[&image.atlas.camera_id];
            let projected = camera
                .project(&image.atlas.pose.transform_world_point(&raw.position))
                .unwrap();
            shifted_observations.get_mut(key).unwrap().xy =
                Point2::new(projected.x + 2.5, projected.y);
        }
        assert_eq!(
            retain_raw_ba_point_if_valid(
                &raw,
                &raw.observations,
                &shifted_observations,
                &images,
                &cameras,
                &BTreeMap::new(),
            )
            .unwrap_err(),
            RawPointFallbackReason::MeanOverMax
        );
    }

    #[test]
    fn filtered_full_cost_gate_rejects_when_only_retained_cost_improves() {
        let error = validate_filter_costs(10.0, 11.0, 5.0, 4.0).unwrap_err();
        assert!(error.contains("full selected cost increased"));
        let error = validate_filter_costs(10.0, 9.0, 5.0, 6.0).unwrap_err();
        assert!(error.contains("retained cost increased"));
    }

    #[test]
    fn filtered_partial_observation_removal_rejects_support_loss() {
        let (_manifest, _store, images, _cameras, landmarks, _active) = joint_ba_fixture();
        let single_track = vec![landmarks[0].clone()];
        let baseline = support_connectivity_for_landmarks(
            &single_track,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let replacement = LandmarkOutput {
            observations: single_track[0].observations[..2].to_vec(),
            ..single_track[0].clone()
        };
        let candidate = support_connectivity_for_landmarks(
            &single_track,
            &BTreeMap::from([(0, replacement)]),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        assert!(!connectivity_not_split(&baseline, &candidate));
    }

    #[test]
    fn filtered_connectivity_rejects_a_split_and_accepts_a_merge() {
        let baseline = SupportConnectivity {
            supported_images: BTreeSet::new(),
            supported_frames: BTreeSet::from([1, 2, 3]),
            frame_components: BTreeMap::from([(1, 0), (2, 0), (3, 0)]),
            component_count: 1,
        };
        let split = SupportConnectivity {
            supported_images: BTreeSet::new(),
            supported_frames: BTreeSet::from([1, 2, 3]),
            frame_components: BTreeMap::from([(1, 0), (2, 1), (3, 1)]),
            component_count: 2,
        };
        assert!(!connectivity_not_split(&baseline, &split));
        let merged = SupportConnectivity {
            supported_images: BTreeSet::new(),
            supported_frames: BTreeSet::from([1, 2, 3]),
            frame_components: BTreeMap::from([(1, 0), (2, 0), (3, 0)]),
            component_count: 1,
        };
        assert!(connectivity_not_split(&baseline, &merged));
    }

    #[test]
    fn joint_ba_selection_is_deterministic_and_respects_window_caps() {
        let (_manifest, store, images, _cameras, landmarks, active) = joint_ba_fixture();
        let first = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
        let second = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.0, landmarks.len());
        assert_eq!(first.1, landmarks.len() * 6);
        assert_eq!(first.2, 3);
        assert_eq!(first.3, 3);
        assert!(store
            .tracks
            .iter()
            .all(|track| track.observations.len() == 6));
        assert!(first.2 <= JOINT_BA_MAX_REFERENCED_FRAMES);
        assert!(first.3 <= JOINT_BA_MAX_FREE_FRAMES);
    }

    #[test]
    fn filter_window_reuses_selection_and_reselects_after_compaction() {
        let (manifest, store, images, cameras, mut landmarks, active) = joint_ba_fixture();
        let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
        let counts_from_wrapper = joint_ba_window_counts(&landmarks, &images, &active).unwrap();
        let counts_from_shared =
            joint_ba_window_counts_with_selected(&landmarks, &images, &active, &selected).unwrap();
        assert_eq!(counts_from_wrapper, counts_from_shared);

        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let wrapper_candidate = build_joint_rig_ba_filter_candidate_with_policy(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, false,
        )
        .unwrap();
        let shared_candidate = build_joint_rig_ba_filter_candidate_with_selected(
            &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, false,
        )
        .unwrap();
        assert_eq!(wrapper_candidate, shared_candidate);

        let old_len = landmarks.len();
        landmarks.remove(0);
        let selected_after_compaction =
            selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
        assert_eq!(selected_after_compaction.len(), old_len - 1);
        assert_eq!(
            selected_after_compaction,
            (0..landmarks.len()).collect::<Vec<_>>()
        );
        let compacted_counts = joint_ba_window_counts_with_selected(
            &landmarks,
            &images,
            &active,
            &selected_after_compaction,
        )
        .unwrap();
        assert_eq!(compacted_counts.0, old_len - 1);
    }

    #[test]
    fn joint_ba_synthetic_update_is_transactional_and_preserves_rig_extrinsics() {
        let (manifest, store, mut images, cameras, mut landmarks, active) = joint_ba_fixture();
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let update =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap();
        assert!(update.final_cost <= update.initial_cost + 1.0e-8);
        assert!(update.solver_final_cost <= update.solver_initial_cost + 1.0e-8);
        assert!(update.iterations <= JOINT_BA_MAX_ITERATIONS);
        assert_eq!(update.landmark_updates.len(), landmarks.len());
        apply_pose_overrides(&mut images, &update.pose_overrides).unwrap();
        for (index, candidate) in update.landmark_updates {
            landmarks[index] = candidate;
        }
        assert_ne!(images, before_images);
        assert_ne!(landmarks, before_landmarks);
        for image in images.values() {
            let sensor = &manifest.sensors[&image.atlas.sensor_index];
            let rig_pose =
                derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
            let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
            assert!(
                (expected.translation - image.atlas.pose.world_to_camera.translation).norm()
                    < 1.0e-8
            );
            assert!(
                (expected.rotation * image.atlas.pose.world_to_camera.rotation.inverse()).angle()
                    < 1.0e-8
            );
        }
    }

    #[test]
    fn joint_ba_external_anchors_are_fixed_and_all_track_observations_are_used() {
        let (manifest, store, mut images, cameras, landmarks, _active) = joint_ba_fixture();
        let active = BTreeSet::from([140]);
        let outside_before = images
            .iter()
            .filter(|(_, image)| image.atlas.frame_id != 140)
            .map(|(id, image)| (*id, image.atlas.pose.clone()))
            .collect::<BTreeMap<_, _>>();
        let update =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap();
        assert_eq!(update.selected_observations, landmarks.len() * 6);
        assert_eq!(update.referenced_frames, 3);
        assert_eq!(update.free_frames, 1);
        assert!(update
            .pose_overrides
            .keys()
            .all(|image_id| { images[image_id].atlas.frame_id == 140 }));
        apply_pose_overrides(&mut images, &update.pose_overrides).unwrap();
        for (image_id, pose) in outside_before {
            assert_eq!(images[&image_id].atlas.pose, pose);
        }
    }

    #[test]
    fn joint_ba_repeated_input_is_bitwise_deterministic_and_validator_rejects_mutation() {
        let (manifest, store, images, cameras, landmarks, active) = joint_ba_fixture();
        let first =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap();
        let second =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap();
        assert_eq!(first, second);
        let mut malformed = first.clone();
        let index = *malformed.landmark_updates.keys().next().unwrap();
        malformed
            .landmark_updates
            .get_mut(&index)
            .unwrap()
            .observations
            .pop();
        let error = validate_joint_ba_update(
            &malformed,
            &store,
            &images,
            &cameras,
            &landmarks,
            &selected_landmarks_for_frames(&landmarks, &active, &images).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("identity or observations"));
        let mut nan_cost = first.clone();
        nan_cost.initial_cost = f64::NAN;
        let error = validate_joint_ba_update(
            &nan_cost,
            &store,
            &images,
            &cameras,
            &landmarks,
            &selected_landmarks_for_frames(&landmarks, &active, &images).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("validated cost"));
    }

    #[test]
    fn joint_ba_rejects_cap_without_truncating_observations() {
        let (manifest, store, images, cameras, mut landmarks, active) = joint_ba_fixture();
        landmarks.extend(
            (12..=JOINT_BA_MAX_LANDMARKS).map(|track_id| LandmarkOutput {
                track_id,
                position: Point3::new(0.0, 0.0, 4.0),
                observations: vec![ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                }],
                errors: vec![0.0],
                rms_error: 0.0,
                mean_error: 0.0,
                max_error: 0.0,
                dlt_sample_count: 1,
            }),
        );
        let error =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap_err();
        assert!(matches!(error, JointRigBaSkip::LandmarkCap { .. }));
        assert_eq!(landmarks.len(), JOINT_BA_MAX_LANDMARKS + 1);
        let selected = selected_landmarks_for_frames(&landmarks, &active, &images).unwrap();
        let baseline = support_connectivity_for_landmarks(
            &landmarks,
            &BTreeMap::new(),
            &BTreeSet::new(),
            &images,
        )
        .unwrap();
        let wrapper_error = build_joint_rig_ba_filter_candidate_with_policy(
            &manifest, &store, &images, &cameras, &landmarks, &active, &baseline, true,
        )
        .unwrap_err();
        let shared_error = build_joint_rig_ba_filter_candidate_with_selected(
            &manifest, &store, &images, &cameras, &landmarks, &active, &selected, &baseline, true,
        )
        .unwrap_err();
        assert_eq!(wrapper_error, shared_error);
    }

    #[test]
    fn joint_ba_rejects_malformed_candidate_without_mutating_inputs() {
        let (manifest, mut store, images, cameras, landmarks, active) = joint_ba_fixture();
        let before_store = store.clone();
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let key = *store.observations.keys().next().unwrap();
        store.observations.get_mut(&key).unwrap().xy.x = f64::NAN;
        let error =
            build_joint_rig_ba_window(&manifest, &store, &images, &cameras, &landmarks, &active)
                .unwrap_err();
        assert!(matches!(error, JointRigBaSkip::Invalid(_)));
        assert_eq!(images, before_images);
        assert_eq!(landmarks, before_landmarks);
        assert_ne!(store, before_store);
    }

    #[test]
    fn merge_ignores_local_point_ids_and_remaps_by_global_image_key() {
        let mut images = BTreeMap::new();
        images.insert(1, test_image(1, "a", vec![Point2::new(1.0, 2.0)]));
        images.insert(2, test_image(2, "b", vec![Point2::new(3.0, 4.0)]));
        let mut store = TrackStore::default();
        let first = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &first, &images),
            Ok(MergeOutcome::New)
        );
        let second = first.clone();
        assert_eq!(
            merge_candidate(&mut store, &second, &images),
            Ok(MergeOutcome::Extended { duplicate_count: 2 })
        );
        assert_eq!(store.tracks.len(), 1);
        assert_eq!(store.observations.len(), 2);
    }

    #[test]
    fn repeated_image_with_changed_xy_is_rejected_before_track_merge() {
        let mut images = BTreeMap::new();
        images.insert(1, test_image(1, "shared", vec![Point2::new(10.0, 20.0)]));
        let mut names = BTreeMap::new();
        names.insert("shared".to_owned(), 1);
        let first = SourceWindow {
            images: vec![SourceImage {
                local_id: 7,
                camera_id: 1,
                name: "shared".to_owned(),
                keypoints: vec![SourceKeypoint {
                    xy: Point2::new(10.0, 20.0),
                    point3d_id: None,
                }],
            }],
            points: Vec::new(),
        };
        let second = SourceWindow {
            images: vec![SourceImage {
                local_id: 99,
                camera_id: 1,
                name: "shared".to_owned(),
                keypoints: vec![SourceKeypoint {
                    xy: Point2::new(10.25, 20.0),
                    point3d_id: None,
                }],
            }],
            points: Vec::new(),
        };
        let mut stats = IngestStats::default();
        update_global_images(&mut images, &first, &names, &mut stats).unwrap();
        let error = update_global_images(&mut images, &second, &names, &mut stats).unwrap_err();
        assert!(error.contains("inconsistent ordered keypoint coordinates"));
    }

    #[test]
    fn same_image_different_keypoint_is_rejected_transactionally() {
        let mut images = BTreeMap::new();
        images.insert(
            1,
            test_image(1, "a", vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)]),
        );
        let mut store = TrackStore::default();
        let candidate = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 1,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &candidate, &images),
            Err(MergeRejection::SameImageDifferentKeypoint)
        );
        assert!(store.tracks.is_empty());
        assert!(store.observations.is_empty());
    }

    #[test]
    fn conflict_free_crossing_two_existing_tracks_is_unioned_transactionally() {
        let mut images = BTreeMap::new();
        for id in 1..=4 {
            images.insert(
                id,
                test_image(id, &id.to_string(), vec![Point2::new(id as f64, 0.0)]),
            );
        }
        let mut store = TrackStore::default();
        let a = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let b = vec![
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 4,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &a, &images),
            Ok(MergeOutcome::New)
        );
        assert_eq!(
            merge_candidate(&mut store, &b, &images),
            Ok(MergeOutcome::New)
        );
        let crossing = vec![a[0], b[0]];
        assert_eq!(
            merge_candidate(&mut store, &crossing, &images),
            Ok(MergeOutcome::Extended { duplicate_count: 2 })
        );
        assert_eq!(store.tracks.len(), 2);
        assert_eq!(store.observations.len(), 4);
        assert!(store.tracks[1].observations.is_empty());
        assert_eq!(store.tracks[0].observations.len(), 4);
    }

    #[test]
    fn conflicting_existing_tracks_are_rejected_without_partial_union() {
        let mut images = BTreeMap::new();
        for id in 1..=4 {
            images.insert(
                id,
                test_image(
                    id,
                    &id.to_string(),
                    vec![
                        Point2::new(id as f64, 0.0),
                        Point2::new(id as f64 + 0.5, 0.0),
                    ],
                ),
            );
        }
        let mut store = TrackStore::default();
        let a = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let b = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 1,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
        ];
        assert_eq!(
            merge_candidate(&mut store, &a, &images),
            Ok(MergeOutcome::New)
        );
        assert_eq!(
            merge_candidate(&mut store, &b, &images),
            Ok(MergeOutcome::New)
        );
        let bridge = vec![a[1], b[1]];
        assert_eq!(
            merge_candidate(&mut store, &bridge, &images),
            Err(MergeRejection::ExistingTrackSameImageConflict)
        );
        assert_eq!(store.tracks.len(), 2);
        assert_eq!(store.tracks[0].observations.len(), 2);
        assert_eq!(store.tracks[1].observations.len(), 2);
    }

    #[test]
    fn dlt_retriangulates_final_pose_and_reports_actual_error() {
        let camera = test_camera();
        let point = Point3::new(0.2, -0.1, 4.0);
        let pose_a = Pose::identity();
        let pose_b =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-1.0, 0.0, 0.0));
        let mut images = BTreeMap::new();
        images.insert(
            1,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 1,
                    frame_id: 1,
                    sensor_index: 0,
                    name: "a".to_owned(),
                    camera_id: 1,
                    pose: pose_a.clone(),
                },
                keypoints: vec![],
            },
        );
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 2,
                    sensor_index: 0,
                    name: "b".to_owned(),
                    camera_id: 1,
                    pose: pose_b.clone(),
                },
                keypoints: vec![],
            },
        );
        let pose_c =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, -0.75, 0.0));
        images.insert(
            3,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 3,
                    frame_id: 3,
                    sensor_index: 0,
                    name: "c".to_owned(),
                    camera_id: 1,
                    pose: pose_c.clone(),
                },
                keypoints: vec![],
            },
        );
        let xy_a = camera
            .project(&pose_a.transform_world_point(&point))
            .unwrap();
        let xy_b = camera
            .project(&pose_b.transform_world_point(&point))
            .unwrap();
        let xy_c = camera
            .project(&pose_c.transform_world_point(&point))
            .unwrap();
        let keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 3,
                keypoint_index: 0,
            },
        ];
        let mut observations = BTreeMap::new();
        observations.insert(
            keys[0],
            ObservationState {
                xy: xy_a,
                owner_track: 0,
            },
        );
        observations.insert(
            keys[1],
            ObservationState {
                xy: Point2::new(xy_b.x + 0.25, xy_b.y),
                owner_track: 0,
            },
        );
        observations.insert(
            keys[2],
            ObservationState {
                xy: xy_c,
                owner_track: 0,
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let track = GlobalTrack { observations: keys };
        let landmark = triangulate_track(0, &track, &observations, &images, &cameras).unwrap();
        assert!(landmark.rms_error > 0.0);
        assert!(landmark.max_error < MAX_REPROJECTION_PX);
    }

    #[test]
    fn coincident_camera_centres_are_rejected_even_with_finite_zero_error() {
        let camera = test_camera();
        let point = Point3::new(0.0, 0.0, 4.0);
        let pose = Pose::identity();
        let mut images = BTreeMap::new();
        for id in 1..=2 {
            images.insert(
                id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id: id,
                        frame_id: id,
                        sensor_index: 0,
                        name: id.to_string(),
                        camera_id: 1,
                        pose: pose.clone(),
                    },
                    keypoints: vec![],
                },
            );
        }
        let xy = camera.project(&pose.transform_world_point(&point)).unwrap();
        let keys = vec![
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 0,
            },
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 0,
            },
        ];
        let observations = keys
            .iter()
            .map(|key| (*key, ObservationState { xy, owner_track: 0 }))
            .collect::<BTreeMap<_, _>>();
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let error = triangulate_track(
            0,
            &GlobalTrack { observations: keys },
            &observations,
            &images,
            &cameras,
        )
        .unwrap_err();
        assert!(error.contains("no observable camera baseline"));
    }

    #[test]
    fn mismatched_sensor_poses_for_one_rig_frame_are_rejected() {
        let sensor = |camera_id| SensorCalibration {
            camera_id,
            width: 640,
            height: 480,
            fx: 100.0,
            fy: 100.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::identity(),
        };
        let mut sensors = BTreeMap::new();
        sensors.insert(0, sensor(1));
        sensors.insert(1, sensor(1));
        let mut assignments = BTreeMap::new();
        assignments.insert(
            "a".to_owned(),
            ImageAssignment {
                frame_id: 7,
                sensor_index: 0,
                global_image_id: 1,
            },
        );
        assignments.insert(
            "b".to_owned(),
            ImageAssignment {
                frame_id: 7,
                sensor_index: 1,
                global_image_id: 2,
            },
        );
        let manifest = RigManifest {
            sensors,
            assignments,
        };
        let mut images = BTreeMap::new();
        let mut first = test_image(1, "a", vec![Point2::new(320.0, 240.0)]);
        first.atlas.frame_id = 7;
        images.insert(1, first);
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 7,
                    sensor_index: 1,
                    name: "b".to_owned(),
                    camera_id: 1,
                    pose: Pose::from_world_to_camera(
                        UnitQuaternion::identity(),
                        Vector3::new(0.01, 0.0, 0.0),
                    ),
                },
                keypoints: vec![Point2::new(320.0, 240.0)],
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, test_camera());
        let error = validate_output_cameras(&manifest, &images, &cameras).unwrap_err();
        assert!(error.contains("sensor poses disagree"));
    }

    #[test]
    fn bounded_sample_is_sorted_endpoint_inclusive_and_deterministic() {
        let keys = (0..100)
            .map(|index| ObservationKey {
                global_image_id: index,
                keypoint_index: 0,
            })
            .collect::<Vec<_>>();
        let sample = bounded_sample(&keys, 7);
        assert_eq!(sample.len(), 7);
        assert_eq!(sample.first(), keys.first());
        assert_eq!(sample.last(), keys.last());
        assert_eq!(sample, bounded_sample(&keys, 7));
    }

    #[test]
    fn recovery_anchor_filter_never_uses_another_unsupported_frame() {
        let mut images = BTreeMap::new();
        let mut supported = test_image(1, "supported", vec![Point2::new(1.0, 1.0)]);
        supported.atlas.frame_id = 10;
        images.insert(1, supported);
        let mut target = test_image(2, "target", vec![Point2::new(2.0, 2.0)]);
        target.atlas.frame_id = 20;
        images.insert(2, target);
        let mut other_unsupported = test_image(3, "other", vec![Point2::new(3.0, 3.0)]);
        other_unsupported.atlas.frame_id = 30;
        images.insert(3, other_unsupported);
        let track = GlobalTrack {
            observations: vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 3,
                    keypoint_index: 0,
                },
            ],
        };
        let supported_frames = BTreeSet::from([10]);
        let keys = recovery_track_keys(&track, 20, &supported_frames, &images);
        assert_eq!(
            keys,
            vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 0,
                },
            ]
        );
    }

    #[test]
    fn recovery_synthetic_end_to_end_accepts_six_target_landmarks() {
        let (manifest, store, images, cameras, baseline, truth_rig_pose) = recovery_fixture(false);
        let result =
            recover_zero_support_frames(&manifest, &store, &images, &cameras, &baseline).unwrap();
        assert_eq!(result.summary.frames_recovered, 1);
        assert_eq!(result.summary.pnp_inliers, 6);
        assert_eq!(result.summary.candidate_target_landmarks, 6);
        assert_eq!(result.summary.accepted_target_landmarks, 6);
        assert_eq!(result.landmarks.len(), 6);
        assert_eq!(result.pose_overrides.len(), 2);
        for (global_image_id, sensor) in &manifest.sensors {
            let image_id = *global_image_id as u64 + 3;
            let expected = sensor
                .sensor_from_rig
                .compose(&truth_rig_pose.world_to_camera);
            let actual = &result.pose_overrides[&image_id].world_to_camera;
            assert!((actual.translation - expected.translation).norm() < 1.0e-8);
            assert!((actual.rotation * expected.rotation.inverse()).angle() < 1.0e-8);
        }
    }

    #[test]
    fn recovery_transactionally_rejects_pose_when_only_five_target_tracks_survive() {
        let (manifest, store, images, cameras, baseline, _) = recovery_fixture(true);
        let result =
            recover_zero_support_frames(&manifest, &store, &images, &cameras, &baseline).unwrap();
        assert!(result.summary.pnp_inliers >= RECOVERY_MIN_PNP_INLIERS);
        assert_eq!(result.summary.candidate_target_landmarks, 5);
        assert_eq!(result.summary.accepted_target_landmarks, 0);
        assert!(result.pose_overrides.is_empty());
        assert!(result.landmarks.is_empty());
    }

    #[test]
    fn recovered_frame_pose_recomposes_both_fixed_sensor_extrinsics() {
        let sensor0 = SensorCalibration {
            camera_id: 1,
            width: 640,
            height: 480,
            fx: 100.0,
            fy: 101.0,
            cx: 320.0,
            cy: 240.0,
            sensor_from_rig: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03)),
                Vector3::new(0.1, -0.02, 0.04),
            ),
        };
        let sensor1 = SensorCalibration {
            camera_id: 2,
            width: 640,
            height: 480,
            fx: 102.0,
            fy: 99.0,
            cx: 321.0,
            cy: 239.0,
            sensor_from_rig: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(-0.03, 0.01, 0.02)),
                Vector3::new(-0.2, 0.03, 0.01),
            ),
        };
        let manifest = RigManifest {
            sensors: BTreeMap::from([(0, sensor0.clone()), (1, sensor1.clone())]),
            assignments: BTreeMap::new(),
        };
        let mut images = BTreeMap::new();
        for (global_image_id, (sensor_index, camera_id)) in [(1, (0, 1)), (2, (1, 2))] {
            images.insert(
                global_image_id,
                GlobalImage {
                    atlas: AtlasImage {
                        global_image_id,
                        frame_id: 7,
                        sensor_index,
                        name: global_image_id.to_string(),
                        camera_id,
                        pose: Pose::identity(),
                    },
                    keypoints: vec![],
                },
            );
        }
        let rig_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.1, 0.05)),
            Vector3::new(1.0, -0.4, 0.7),
        );
        let mut overrides = BTreeMap::new();
        compose_recovered_frame_poses(&manifest, 7, &rig_pose, &images, &mut overrides).unwrap();
        assert_eq!(overrides.len(), 2);
        for (sensor_index, calibration) in [(0, sensor0), (1, sensor1)] {
            let expected = calibration
                .sensor_from_rig
                .compose(&rig_pose.world_to_camera);
            let actual = &overrides[&(sensor_index as u64 + 1)].world_to_camera;
            assert!((actual.translation - expected.translation).norm() < 1.0e-12);
            assert!((actual.rotation * expected.rotation.inverse()).angle() < 1.0e-12);
        }
        assert!(images
            .values()
            .all(|image| image.atlas.pose == Pose::identity()));
    }

    #[test]
    fn multiple_atlas_components_are_rejected_as_independent_gauges() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_landmark_components_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("component-000")).unwrap();
        fs::create_dir_all(root.join("component-001")).unwrap();
        fs::write(root.join("component-000/images.txt"), "").unwrap();
        fs::write(root.join("component-001/images.txt"), "").unwrap();
        let manifest = RigManifest {
            sensors: BTreeMap::new(),
            assignments: BTreeMap::new(),
        };
        let error = parse_atlas_images(&root, &manifest).unwrap_err();
        assert!(error.contains("independent component images.txt"));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn empty_landmark_set_cannot_be_published_as_pose_only_model() {
        let error = require_nonempty_landmarks(&[], 0).unwrap_err();
        assert!(error.contains("refusing to publish a pose-only model"));
    }

    #[test]
    fn actual_error_and_bidirectional_track_are_written() {
        let root = std::env::temp_dir().join(format!(
            "visloc_integrate_landmark_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let camera = test_camera();
        let pose = Pose::identity();
        let xy = camera.project(&Point3::new(0.0, 0.0, 4.0)).unwrap();
        let mut images = BTreeMap::new();
        images.insert(
            1,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 1,
                    frame_id: 1,
                    sensor_index: 0,
                    name: "a.png".to_owned(),
                    camera_id: 1,
                    pose,
                },
                keypoints: vec![xy],
            },
        );
        images.insert(
            2,
            GlobalImage {
                atlas: AtlasImage {
                    global_image_id: 2,
                    frame_id: 2,
                    sensor_index: 0,
                    name: "b.png".to_owned(),
                    camera_id: 1,
                    pose: Pose::from_world_to_camera(
                        UnitQuaternion::identity(),
                        Vector3::new(-0.5, 0.0, 0.0),
                    ),
                },
                keypoints: vec![Point2::new(320.0, 240.0)],
            },
        );
        let mut cameras = BTreeMap::new();
        cameras.insert(1, camera);
        let landmark = LandmarkOutput {
            track_id: 0,
            position: Point3::new(0.0, 0.0, 4.0),
            observations: vec![
                ObservationKey {
                    global_image_id: 1,
                    keypoint_index: 0,
                },
                ObservationKey {
                    global_image_id: 2,
                    keypoint_index: 0,
                },
            ],
            errors: vec![0.1, 0.3],
            rms_error: (0.05_f64).sqrt(),
            mean_error: 0.2,
            max_error: 0.3,
            dlt_sample_count: 2,
        };
        let mut stats = IngestStats::default();
        stats.output_landmarks = 1;
        stats.output_observations = 2;
        write_model(&root, &images, &cameras, &[landmark], &stats).unwrap();
        let points = fs::read_to_string(root.join("points3D.txt")).unwrap();
        let image = fs::read_to_string(root.join("images.txt")).unwrap();
        assert!(points.contains(" 0.2 1 0 2 0"));
        assert!(image.lines().any(|line| line.contains("320 240 1")));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn diagnostic_cross_boundary_partition_is_deterministic() {
        let mut images = BTreeMap::new();
        images.insert(1, test_image(1, "a", vec![Point2::new(1.0, 1.0)]));
        images.insert(2, test_image(2, "b", vec![Point2::new(2.0, 2.0)]));
        images.insert(3, test_image(3, "c", vec![Point2::new(3.0, 3.0)]));
        let store = TrackStore {
            observations: BTreeMap::new(),
            tracks: vec![GlobalTrack {
                observations: vec![
                    ObservationKey {
                        global_image_id: 3,
                        keypoint_index: 0,
                    },
                    ObservationKey {
                        global_image_id: 1,
                        keypoint_index: 0,
                    },
                    ObservationKey {
                        global_image_id: 2,
                        keypoint_index: 0,
                    },
                ],
            }],
        };
        let first = collect_diagnostic_cross_tracks(&store, &images, 1).unwrap();
        let second = collect_diagnostic_cross_tracks(&store, &images, 1).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].left_keys.len(), 1);
        assert_eq!(first[0].right_keys.len(), 2);
        assert_eq!(first[0].left_keys[0].global_image_id, 1);
        assert_eq!(
            first[0]
                .right_keys
                .iter()
                .map(|key| key.global_image_id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn diagnostic_inliers_count_distinct_tracks_not_observations() {
        let make = |track_id, image_id| DiagnosticCandidateObservation {
            track_id,
            key: ObservationKey {
                global_image_id: image_id,
                keypoint_index: 0,
            },
            correspondence: GeneralizedCorrespondence2D3D {
                sensor_index: 0,
                point2d: Point2::new(0.0, 0.0),
                point3d: Point3::new(0.0, 0.0, 1.0),
                confidence: None,
            },
        };
        let candidates = vec![make(4, 1), make(4, 2), make(8, 3), make(9, 4)];
        assert_eq!(diagnostic_distinct_track_ids(&candidates).len(), 3);
        assert_eq!(
            diagnostic_inlier_track_ids(&[0, 1, 2], &candidates).len(),
            2
        );
        assert_eq!(
            diagnostic_inlier_track_ids(&[0, 2, 3], &candidates).len(),
            3
        );
    }

    #[test]
    fn diagnostic_pose_round_trip_preserves_fixed_sensor_extrinsics() {
        let (manifest, _store, images, cameras, _baseline, _) = recovery_fixture(false);
        let rig_pose = Pose::from_world_to_camera(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.03, -0.02, 0.01)),
            Vector3::new(0.2, -0.1, 0.4),
        );
        assert_eq!(
            validate_diagnostic_rig_pose(&manifest, 10, &images, &rig_pose).unwrap(),
            2
        );
        let rig = build_generalized_rig(&manifest, &cameras).unwrap();
        assert_eq!(rig.sensors().len(), 2);
    }

    #[test]
    fn diagnostic_target_track_deduplicates_shared_observations() {
        let left = vec![ObservationKey {
            global_image_id: 1,
            keypoint_index: 3,
        }];
        let right = vec![
            ObservationKey {
                global_image_id: 2,
                keypoint_index: 3,
            },
            ObservationKey {
                global_image_id: 1,
                keypoint_index: 3,
            },
        ];
        let track = diagnostic_track_with_keys(&left, &right);
        assert_eq!(track.observations.len(), 2);
        assert_eq!(track.observations[0], left[0]);
        assert_eq!(track.observations[1], right[0]);
    }

    #[test]
    fn diagnostic_all_supported_frames_can_remain_disconnected() {
        let (sizes, supported) =
            diagnostic_component_sizes_from_frame_tracks(&[0, 1, 2, 3], &[vec![0, 1], vec![2, 3]]);
        assert_eq!(supported, BTreeSet::from([0, 1, 2, 3]));
        assert_eq!(sizes, vec![2, 2]);
    }

    #[test]
    fn boundary_repair_acceptance_requires_strict_connectivity_and_support_subset() {
        let baseline = DiagnosticConnectivityReport {
            supported_images: BTreeSet::from([1, 2]),
            supported_frames: BTreeSet::from([10, 30]),
            supported_component_sizes: vec![1, 1],
            removed_tracks: 0,
            removed_observations: 0,
            added_tracks: 0,
            added_observations: 0,
        };
        let mut candidate = baseline.clone();
        candidate.supported_images.insert(3);
        candidate.supported_frames.insert(40);
        candidate.supported_component_sizes = vec![3];
        assert!(boundary_repair_candidate_is_accepted(
            &baseline, &candidate, 1
        ));
        candidate.supported_frames.remove(&10);
        assert!(!boundary_repair_candidate_is_accepted(
            &baseline, &candidate, 1
        ));
        candidate.supported_frames.insert(10);
        candidate.supported_component_sizes = vec![1, 2];
        assert!(!boundary_repair_candidate_is_accepted(
            &baseline, &candidate, 1
        ));
        assert!(!boundary_repair_candidate_is_accepted(
            &baseline, &candidate, 0
        ));
    }

    #[test]
    fn boundary_repair_accepts_first_candidate_transactionally_and_preserves_extrinsics() {
        let (manifest, store, mut images, cameras, mut landmarks) = boundary_repair_fixture();
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let first =
            repair_cross_boundary(&manifest, 10, &store, &mut images, &cameras, &mut landmarks)
                .unwrap();
        assert_eq!(first.status, "accepted");
        assert_eq!(first.accepted_frame, Some(30));
        assert_eq!(first.added_tracks, 6);
        assert_eq!(first.removed_tracks, 0);
        assert_eq!(first.baseline_component_count, 2);
        assert_eq!(first.candidate_component_count, 1);
        assert_ne!(images, before_images);
        assert_ne!(landmarks, before_landmarks);
        assert_eq!(landmarks.len(), 8);
        assert_eq!(
            validate_diagnostic_rig_pose(&manifest, 30, &images, &Pose::identity()),
            Ok(2)
        );
        for image in images.values() {
            let sensor = &manifest.sensors[&image.atlas.sensor_index];
            let rig_pose =
                derive_rig_pose_for_frame(&manifest, image.atlas.frame_id, &images).unwrap();
            let expected = sensor.sensor_from_rig.compose(&rig_pose.world_to_camera);
            assert!(
                (expected.translation - image.atlas.pose.world_to_camera.translation).norm()
                    < 1.0e-8
            );
            assert!(
                (expected.rotation * image.atlas.pose.world_to_camera.rotation.inverse()).angle()
                    < 1.0e-8
            );
        }

        let (manifest_again, store_again, mut images_again, cameras_again, mut landmarks_again) =
            boundary_repair_fixture();
        let second = repair_cross_boundary(
            &manifest_again,
            10,
            &store_again,
            &mut images_again,
            &cameras_again,
            &mut landmarks_again,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(images, images_again);
        assert_eq!(landmarks, landmarks_again);
    }

    #[test]
    fn boundary_repair_rejection_is_transactional_when_candidate_is_underconstrained() {
        let (manifest, mut store, mut images, cameras, mut landmarks) = boundary_repair_fixture();
        store.tracks.truncate(7);
        let before_images = images.clone();
        let before_landmarks = landmarks.clone();
        let summary =
            repair_cross_boundary(&manifest, 10, &store, &mut images, &cameras, &mut landmarks)
                .unwrap();
        assert_eq!(summary.status, "no-accepted-candidate");
        assert_eq!(summary.accepted_frame, None);
        assert_eq!(images, before_images);
        assert_eq!(landmarks, before_landmarks);
    }
}
