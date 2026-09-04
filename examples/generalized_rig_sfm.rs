//! Replay a verified-pair snapshot through the generalized-rig mapper.
//!
//! The manifest is deliberately dependency-free and explicit:
//!
//! ```text
//! # S index camera_id width height fx fy cx cy qw qx qy qz tx ty tz
//! S 0 1 848 800 284.98 286.10 425.24 398.46 1 0 0 0 0 0 0
//! S 1 2 848 800 284.81 285.97 427.66 397.12 0.99999 ... -0.0639 ...
//! # F frame_id image_name sensor_index
//! F 0 cam1_000000.png 0
//! F 0 cam2_000001.png 1
//! ```
//!
//! Sensor poses are `T_sensor<-rig`. Feature files are resolved from each
//! image stem plus `--feature-suffix`; descriptors are intentionally not kept
//! because track construction and mapping consume only keypoints and verified
//! feature indices.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use nalgebra::{Matrix3, Point2, Point3, Quaternion, UnitQuaternion, Vector3};
use visloc_rs::slam::global_sfm::average_positions_with_independent_edge_scales;
use visloc_rs::slam::incremental_sfm::preview_pair_confidence_conflicts;
use visloc_rs::verified_pair_snapshot;
use visloc_rs::{
    build_rig_correspondence, incremental_rig_sfm,
    metric_temporal_quadrilateral_tracks_in_frame_gap, umeyama_similarity_transform,
    write_colmap_reconstruction_for_3dgs_with_cameras, Camera, FeatureSet, GeneralizedCameraRig,
    GlobalSfmEdge, LinearSolver, PairwiseMatches, Pose, PoseGraph, PoseGraphEdge,
    PoseGraphEdgeKind, RigFrame, RigFrameImage, RigSensor, RigSfmConfig, RigSfmError, RigSfmResult,
    RigTrackBuilder, RobustKernel, SE3,
};

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    features_dir: PathBuf,
    append_features_dirs: Vec<PathBuf>,
    deferred_overlay_snapshot: Option<PathBuf>,
    deferred_overlay_min_temporal_frame_gap: usize,
    deferred_overlay_max_frame_gap: Option<usize>,
    deferred_overlay_max_matches_per_pair: Option<usize>,
    feature_suffix: String,
    snapshot: PathBuf,
    out_colmap: PathBuf,
    max_models: usize,
    min_model_frames: usize,
    min_pnp_inliers: usize,
    min_pnp_sensors: usize,
    direct_stereo_pnp_max_frame_gap: usize,
    direct_stereo_min_pnp_sensors: Option<usize>,
    direct_stereo_min_triangulation_angle_deg: f64,
    motion_bridge_max_frame_gap: usize,
    motion_bridge_min_inliers: usize,
    motion_bridge_max_rotation_deviation_deg: f64,
    deferred_registration_pair_prefix: Option<usize>,
    deferred_retriangulation_pair_prefix: Option<usize>,
    deferred_registration_min_pnp_sensors: Option<usize>,
    deferred_registration_min_pnp_inliers: Option<usize>,
    deferred_registration_pnp_max_iterations: Option<usize>,
    deferred_registration_max_interpolation_gap: usize,
    retriangulate_deferred_tracks_after_registration: bool,
    deferred_retriangulation_metric_temporal_cycle_tracks: bool,
    deferred_retriangulation_metric_temporal_quadrilateral_tracks: bool,
    deferred_retriangulation_quadrilateral_min_frame_gap: usize,
    deferred_retriangulation_quadrilateral_max_frame_gap: usize,
    deferred_retriangulation_min_metric_frames: usize,
    export_deferred_quadrilaterals_tsv: Option<PathBuf>,
    deferred_quadrilateral_whitelist_tsv: Option<PathBuf>,
    structure_min_pair_matches: Option<usize>,
    deferred_long_pair_min_frame_gap: Option<usize>,
    structure_long_min_e_inliers: Option<u64>,
    long_pair_pose_prior_images: Vec<PathBuf>,
    structure_long_max_rotation_disagreement_deg: Option<f64>,
    final_rig_rotation_average_max_disagreement_deg: Option<f64>,
    final_rig_rotation_average_max_update_deg: f64,
    final_rig_rotation_average_weight_cap: usize,
    final_rig_translation_average: bool,
    final_rig_translation_average_min_frame_gap: usize,
    final_rig_translation_average_min_matches: usize,
    final_rig_translation_average_max_direction_error_deg: f64,
    final_rig_translation_average_max_update_m: f64,
    final_rig_translation_average_iterations: usize,
    robust_triangulation_pruning: bool,
    triangulation_min_inlier_fraction: f64,
    max_reprojection_error_px: f64,
    pnp_max_iterations: usize,
    final_bundle_adjustment: bool,
    max_matches_per_pair: Option<usize>,
    max_track_frame_gap: Option<usize>,
    local_ba_every: usize,
    local_ba_window_size: usize,
    local_ba_iterations: usize,
    final_ba_passes: usize,
    final_ba_window_size: usize,
    final_ba_fix_window_ends: bool,
    final_filter_refinement_passes: usize,
    ransac_seed: u64,
    track_builder: RigTrackBuilder,
    recover_metric_conflict_tracks: bool,
    conflict_recovery_max_hypotheses: usize,
    conflict_recovery_max_reprojection_error_px: f64,
    conflict_recovery_max_mean_reprojection_px: f64,
    complete_tracks_after_registration: bool,
    track_completion_max_passes: usize,
    track_completion_max_reprojection_error_px: f64,
    repair_isolated_pose_outliers: bool,
    isolated_pose_max_midpoint_error_m: f64,
    isolated_pose_min_detour_ratio: f64,
    isolated_pose_repair_max_passes: usize,
    repair_paired_pose_jumps: bool,
    paired_pose_jump_absolute_step_m: f64,
    paired_pose_jump_min_step_ratio: f64,
    paired_pose_jump_max_frame_span: usize,
    paired_pose_jump_max_closure_ratio: f64,
    ba_metric_tracks_only: bool,
    final_ba_min_pose_observations: usize,
    ba_huber_delta: f64,
    structure_refinement_iterations: usize,
    preview_rig_correspondence_csr: bool,
    preview_pair_confidence_conflicts: bool,
    dynamic_correspondence_tracking: bool,
    gr6p_seed_candidate_cap: usize,
    gr6p_seed_correspondence_cap: usize,
    gr6p_seed_max_iterations: usize,
    gr6p_seed_min_iterations: usize,
    gr6p_seed_min_inliers: usize,
    gr6p_seed_angular_threshold_deg: f64,
    gr6p_seed_min_positive_depth_fraction: f64,
    gr6p_seed_min_baseline_m: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut values = std::env::args().skip(1);
    let mut manifest = None;
    let mut features_dir = None;
    let mut append_features_dirs = Vec::new();
    let mut deferred_overlay_snapshot = None;
    let mut deferred_overlay_min_temporal_frame_gap = 1usize;
    let mut deferred_overlay_max_frame_gap = None;
    let mut deferred_overlay_max_matches_per_pair = None;
    let mut feature_suffix = "_features.txt".to_owned();
    let mut snapshot = None;
    let mut out_colmap = None;
    let mut max_models = 1usize;
    let mut min_model_frames = 10usize;
    let mut min_pnp_inliers = 8usize;
    let mut max_reprojection_error_px = 4.0;
    let mut pnp_max_iterations = 512usize;
    let defaults = RigSfmConfig::default();
    let mut min_pnp_sensors = defaults.min_pnp_sensors;
    let mut direct_stereo_pnp_max_frame_gap = defaults.direct_stereo_pnp_max_frame_gap;
    let mut direct_stereo_min_pnp_sensors = defaults.direct_stereo_min_pnp_sensors;
    let mut direct_stereo_min_triangulation_angle_deg =
        defaults.direct_stereo_min_triangulation_angle_deg;
    let mut motion_bridge_max_frame_gap = defaults.motion_bridge_max_frame_gap;
    let mut motion_bridge_min_inliers = defaults.motion_bridge_min_inliers;
    let mut motion_bridge_max_rotation_deviation_deg =
        defaults.motion_bridge_max_rotation_deviation_deg;
    let mut deferred_registration_pair_prefix = defaults.deferred_registration_pair_prefix;
    let deferred_retriangulation_pair_prefix = None;
    let mut deferred_registration_min_pnp_sensors = defaults.deferred_registration_min_pnp_sensors;
    let mut deferred_registration_min_pnp_inliers = defaults.deferred_registration_min_pnp_inliers;
    let mut deferred_registration_pnp_max_iterations =
        defaults.deferred_registration_pnp_max_iterations;
    let mut deferred_registration_max_interpolation_gap =
        defaults.deferred_registration_max_interpolation_gap;
    let mut retriangulate_deferred_tracks_after_registration =
        defaults.retriangulate_deferred_tracks_after_registration;
    let mut deferred_retriangulation_metric_temporal_cycle_tracks =
        defaults.deferred_retriangulation_metric_temporal_cycle_tracks;
    let mut deferred_retriangulation_metric_temporal_quadrilateral_tracks =
        defaults.deferred_retriangulation_metric_temporal_quadrilateral_tracks;
    let mut deferred_retriangulation_quadrilateral_min_frame_gap =
        defaults.deferred_retriangulation_quadrilateral_min_frame_gap;
    let mut deferred_retriangulation_quadrilateral_max_frame_gap =
        defaults.deferred_retriangulation_quadrilateral_max_frame_gap;
    let mut deferred_retriangulation_min_metric_frames =
        defaults.deferred_retriangulation_min_metric_frames;
    let mut export_deferred_quadrilaterals_tsv = None;
    let mut deferred_quadrilateral_whitelist_tsv = None;
    let mut structure_min_pair_matches = None;
    let mut deferred_long_pair_min_frame_gap = None;
    let mut structure_long_min_e_inliers = None;
    let mut long_pair_pose_prior_images = Vec::new();
    let mut structure_long_max_rotation_disagreement_deg = None;
    let mut final_rig_rotation_average_max_disagreement_deg = None;
    let mut final_rig_rotation_average_max_update_deg = 1.0f64;
    let mut final_rig_rotation_average_weight_cap = 128usize;
    let mut final_rig_translation_average = false;
    let mut final_rig_translation_average_min_frame_gap = 8usize;
    let mut final_rig_translation_average_min_matches = 30usize;
    let mut final_rig_translation_average_max_direction_error_deg = 60.0f64;
    let mut final_rig_translation_average_max_update_m = 2.0f64;
    let mut final_rig_translation_average_iterations = 8usize;
    let mut robust_triangulation_pruning = defaults.robust_triangulation_pruning;
    let mut triangulation_min_inlier_fraction = defaults.triangulation_min_inlier_fraction;
    let mut final_bundle_adjustment = defaults.final_bundle_adjustment;
    let mut max_matches_per_pair = None;
    let mut max_track_frame_gap = None;
    let mut local_ba_every = defaults.local_ba_every;
    let mut local_ba_window_size = defaults.local_ba_window_size;
    let mut local_ba_iterations = defaults.local_ba_iterations;
    let mut final_ba_passes = defaults.final_ba_passes;
    let mut final_ba_window_size = defaults.final_ba_window_size;
    let mut final_ba_fix_window_ends = defaults.final_ba_fix_window_ends;
    let mut final_filter_refinement_passes = defaults.final_filter_refinement_passes;
    let mut ransac_seed = defaults.ransac_seed;
    let mut track_builder = defaults.track_builder;
    let mut recover_metric_conflict_tracks = defaults.recover_metric_conflict_tracks;
    let mut conflict_recovery_max_hypotheses = defaults.conflict_recovery_max_hypotheses;
    let mut conflict_recovery_max_reprojection_error_px =
        defaults.conflict_recovery_max_reprojection_error_px;
    let mut conflict_recovery_max_mean_reprojection_px =
        defaults.conflict_recovery_max_mean_reprojection_px;
    let mut complete_tracks_after_registration = defaults.complete_tracks_after_registration;
    let mut track_completion_max_passes = defaults.track_completion_max_passes;
    let mut track_completion_max_reprojection_error_px =
        defaults.track_completion_max_reprojection_error_px;
    let mut repair_isolated_pose_outliers = defaults.repair_isolated_pose_outliers;
    let mut isolated_pose_max_midpoint_error_m = defaults.isolated_pose_max_midpoint_error_m;
    let mut isolated_pose_min_detour_ratio = defaults.isolated_pose_min_detour_ratio;
    let mut isolated_pose_repair_max_passes = defaults.isolated_pose_repair_max_passes;
    let mut repair_paired_pose_jumps = defaults.repair_paired_pose_jumps;
    let mut paired_pose_jump_absolute_step_m = defaults.paired_pose_jump_absolute_step_m;
    let mut paired_pose_jump_min_step_ratio = defaults.paired_pose_jump_min_step_ratio;
    let mut paired_pose_jump_max_frame_span = defaults.paired_pose_jump_max_frame_span;
    let mut paired_pose_jump_max_closure_ratio = defaults.paired_pose_jump_max_closure_ratio;
    let mut ba_metric_tracks_only = defaults.ba_metric_tracks_only;
    let mut final_ba_min_pose_observations = defaults.final_ba_min_pose_observations;
    let mut ba_huber_delta = 6.0;
    let mut structure_refinement_iterations = defaults.structure_refinement_iterations;
    let mut preview_rig_correspondence_csr = false;
    let mut preview_pair_confidence_conflicts = false;
    let mut dynamic_correspondence_tracking = false;
    let mut gr6p_seed_candidate_cap = defaults.gr6p_seed_candidate_cap;
    let mut gr6p_seed_correspondence_cap = defaults.gr6p_seed_correspondence_cap;
    let mut gr6p_seed_max_iterations = defaults.gr6p_seed_max_iterations;
    let mut gr6p_seed_min_iterations = defaults.gr6p_seed_min_iterations;
    let mut gr6p_seed_min_inliers = defaults.gr6p_seed_min_inliers;
    let mut gr6p_seed_angular_threshold_deg = defaults.gr6p_seed_angular_threshold_deg;
    let mut gr6p_seed_min_positive_depth_fraction = defaults.gr6p_seed_min_positive_depth_fraction;
    let mut gr6p_seed_min_baseline_m = defaults.gr6p_seed_min_baseline_m;
    while let Some(flag) = values.next() {
        let mut value = || {
            values
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value()?)),
            "--features-dir" => features_dir = Some(PathBuf::from(value()?)),
            "--append-features-dir" => append_features_dirs.push(PathBuf::from(value()?)),
            "--deferred-overlay-snapshot" => {
                deferred_overlay_snapshot = Some(PathBuf::from(value()?))
            }
            "--deferred-overlay-max-frame-gap" => {
                deferred_overlay_max_frame_gap =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-overlay-min-temporal-frame-gap" => {
                deferred_overlay_min_temporal_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--deferred-overlay-max-matches-per-pair" => {
                let parsed = value()?.parse().map_err(|error| format!("{error}"))?;
                deferred_overlay_max_matches_per_pair = (parsed > 0).then_some(parsed);
            }
            "--feature-suffix" => feature_suffix = value()?,
            "--snapshot" => snapshot = Some(PathBuf::from(value()?)),
            "--preview-rig-correspondence-csr" => preview_rig_correspondence_csr = true,
            "--preview-pair-confidence-conflicts" => preview_pair_confidence_conflicts = true,
            "--dynamic-correspondence-tracking" => dynamic_correspondence_tracking = true,
            "--gr6p-seed-candidate-cap" => {
                gr6p_seed_candidate_cap = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-correspondence-cap" => {
                gr6p_seed_correspondence_cap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-max-iterations" => {
                gr6p_seed_max_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-min-iterations" => {
                gr6p_seed_min_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-min-inliers" => {
                gr6p_seed_min_inliers = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-angular-threshold-deg" => {
                gr6p_seed_angular_threshold_deg =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-min-positive-depth-fraction" => {
                gr6p_seed_min_positive_depth_fraction =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--gr6p-seed-min-baseline-m" => {
                gr6p_seed_min_baseline_m = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--out-colmap" => out_colmap = Some(PathBuf::from(value()?)),
            "--max-models" => max_models = value()?.parse().map_err(|error| format!("{error}"))?,
            "--min-model-frames" => {
                min_model_frames = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--min-pnp-inliers" => {
                min_pnp_inliers = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--min-pnp-sensors" => {
                min_pnp_sensors = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--direct-stereo-pnp-max-frame-gap" => {
                direct_stereo_pnp_max_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--direct-stereo-min-pnp-sensors" => {
                direct_stereo_min_pnp_sensors =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--direct-stereo-min-triangulation-angle-deg" => {
                direct_stereo_min_triangulation_angle_deg =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--motion-bridge-max-frame-gap" => {
                motion_bridge_max_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--motion-bridge-min-inliers" => {
                motion_bridge_min_inliers = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--motion-bridge-max-rotation-deviation-deg" => {
                motion_bridge_max_rotation_deviation_deg =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--deferred-registration-pair-prefix" => {
                deferred_registration_pair_prefix =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-registration-min-pnp-sensors" => {
                deferred_registration_min_pnp_sensors =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-registration-min-pnp-inliers" => {
                deferred_registration_min_pnp_inliers =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-registration-pnp-iterations" => {
                deferred_registration_pnp_max_iterations =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-registration-max-interpolation-gap" => {
                deferred_registration_max_interpolation_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--retriangulate-deferred-tracks-after-registration" => {
                retriangulate_deferred_tracks_after_registration = true
            }
            "--deferred-retriangulation-metric-temporal-cycle-tracks" => {
                deferred_retriangulation_metric_temporal_cycle_tracks = true
            }
            "--deferred-retriangulation-metric-temporal-quadrilateral-tracks" => {
                deferred_retriangulation_metric_temporal_quadrilateral_tracks = true
            }
            "--deferred-retriangulation-quadrilateral-min-frame-gap" => {
                deferred_retriangulation_quadrilateral_min_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--deferred-retriangulation-quadrilateral-max-frame-gap" => {
                deferred_retriangulation_quadrilateral_max_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--deferred-retriangulation-min-metric-frames" => {
                deferred_retriangulation_min_metric_frames =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--export-deferred-quadrilaterals-tsv" => {
                export_deferred_quadrilaterals_tsv = Some(PathBuf::from(value()?))
            }
            "--deferred-quadrilateral-whitelist-tsv" => {
                deferred_quadrilateral_whitelist_tsv = Some(PathBuf::from(value()?))
            }
            "--structure-min-pair-matches" => {
                structure_min_pair_matches =
                    Some(value()?.parse().map_err(|error| format!("{error}"))?)
            }
            "--deferred-long-pair-min-frame-gap" => {
                let parsed = value()?.parse().map_err(|error| format!("{error}"))?;
                deferred_long_pair_min_frame_gap = Some(parsed);
            }
            "--structure-long-min-e-inliers" => {
                let parsed = value()?.parse().map_err(|error| format!("{error}"))?;
                structure_long_min_e_inliers = Some(parsed);
            }
            "--long-pair-pose-prior-images" => {
                long_pair_pose_prior_images.push(PathBuf::from(value()?));
            }
            "--structure-long-max-rotation-disagreement-deg" => {
                let parsed: f64 = value()?.parse().map_err(|error| format!("{error}"))?;
                structure_long_max_rotation_disagreement_deg = Some(parsed);
            }
            "--final-rig-rotation-average-max-disagreement-deg" => {
                let parsed: f64 = value()?.parse().map_err(|error| format!("{error}"))?;
                final_rig_rotation_average_max_disagreement_deg = Some(parsed);
            }
            "--final-rig-rotation-average-max-update-deg" => {
                final_rig_rotation_average_max_update_deg =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-rotation-average-weight-cap" => {
                final_rig_rotation_average_weight_cap =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-translation-average" => final_rig_translation_average = true,
            "--no-final-rig-translation-average" => final_rig_translation_average = false,
            "--final-rig-translation-average-min-frame-gap" => {
                final_rig_translation_average_min_frame_gap =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-translation-average-min-matches" => {
                final_rig_translation_average_min_matches =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-translation-average-max-direction-error-deg" => {
                final_rig_translation_average_max_direction_error_deg =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-translation-average-max-update-m" => {
                final_rig_translation_average_max_update_m =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--final-rig-translation-average-iterations" => {
                final_rig_translation_average_iterations =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--robust-triangulation-pruning" => robust_triangulation_pruning = true,
            "--no-robust-triangulation-pruning" => robust_triangulation_pruning = false,
            "--triangulation-min-inlier-fraction" => {
                triangulation_min_inlier_fraction =
                    value()?.parse().map_err(|error| format!("{error}"))?;
            }
            "--max-reprojection-error-px" => {
                max_reprojection_error_px = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--pnp-max-iterations" => {
                pnp_max_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--no-final-ba" => final_bundle_adjustment = false,
            "--final-ba" => final_bundle_adjustment = true,
            "--max-matches-per-pair" => {
                let parsed: usize = value()?.parse().map_err(|error| format!("{error}"))?;
                max_matches_per_pair = (parsed > 0).then_some(parsed);
            }
            "--max-track-frame-gap" => {
                let parsed: usize = value()?.parse().map_err(|error| format!("{error}"))?;
                max_track_frame_gap = (parsed > 0).then_some(parsed);
            }
            "--local-ba-every" => {
                local_ba_every = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--local-ba-window" => {
                local_ba_window_size = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--local-ba-iterations" => {
                local_ba_iterations = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-passes" => {
                final_ba_passes = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-window" => {
                final_ba_window_size = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--final-ba-single-anchor" => final_ba_fix_window_ends = false,
            "--final-ba-fix-window-ends" => final_ba_fix_window_ends = true,
            "--final-filter-refinement-passes" => {
                final_filter_refinement_passes =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--ransac-seed" => {
                ransac_seed = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--conflict-preserving-tracks" => track_builder = RigTrackBuilder::ConflictPreserving,
            "--stream-order-conflict-preserving-tracks" => {
                track_builder = RigTrackBuilder::StreamOrderConflictPreserving
            }
            "--pair-confidence-tracks" => track_builder = RigTrackBuilder::PairConfidence,
            "--trusted-pair-prefix" => {
                let pair_count = value()?.parse().map_err(|error| format!("{error}"))?;
                track_builder = RigTrackBuilder::TrustedPrefixPairConfidence(pair_count);
            }
            "--sparse-cycle-tracks" => track_builder = RigTrackBuilder::SparseCycle,
            "--metric-sparse-cycle-tracks" => track_builder = RigTrackBuilder::MetricSparseCycle,
            "--metric-temporal-cycle-tracks" => {
                track_builder = RigTrackBuilder::MetricTemporalCycle
            }
            "--metric-anchored-cycle-tracks" => {
                track_builder = RigTrackBuilder::MetricAnchoredCycle
            }
            "--legacy-union-find-tracks" => track_builder = RigTrackBuilder::LegacyUnionFind,
            "--recover-metric-conflict-tracks" => recover_metric_conflict_tracks = true,
            "--no-recover-metric-conflict-tracks" => recover_metric_conflict_tracks = false,
            "--conflict-recovery-max-hypotheses" => {
                conflict_recovery_max_hypotheses =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--conflict-recovery-max-reprojection-error-px" => {
                conflict_recovery_max_reprojection_error_px =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--conflict-recovery-max-mean-reprojection-px" => {
                conflict_recovery_max_mean_reprojection_px =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--complete-tracks-after-registration" => complete_tracks_after_registration = true,
            "--no-complete-tracks-after-registration" => complete_tracks_after_registration = false,
            "--track-completion-max-passes" => {
                track_completion_max_passes =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--track-completion-max-reprojection-error-px" => {
                track_completion_max_reprojection_error_px =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--repair-isolated-pose-outliers" => repair_isolated_pose_outliers = true,
            "--no-repair-isolated-pose-outliers" => repair_isolated_pose_outliers = false,
            "--isolated-pose-max-midpoint-error-m" => {
                isolated_pose_max_midpoint_error_m =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--isolated-pose-min-detour-ratio" => {
                isolated_pose_min_detour_ratio =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--isolated-pose-repair-max-passes" => {
                isolated_pose_repair_max_passes =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--repair-paired-pose-jumps" => repair_paired_pose_jumps = true,
            "--no-repair-paired-pose-jumps" => repair_paired_pose_jumps = false,
            "--paired-pose-jump-absolute-step-m" => {
                paired_pose_jump_absolute_step_m =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--paired-pose-jump-min-step-ratio" => {
                paired_pose_jump_min_step_ratio =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--paired-pose-jump-max-frame-span" => {
                paired_pose_jump_max_frame_span =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--paired-pose-jump-max-closure-ratio" => {
                paired_pose_jump_max_closure_ratio =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--ba-metric-tracks-only" => ba_metric_tracks_only = true,
            "--ba-all-tracks" => ba_metric_tracks_only = false,
            "--final-ba-min-pose-observations" => {
                final_ba_min_pose_observations =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--ba-huber-delta" => {
                ba_huber_delta = value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--structure-refinement-iterations" => {
                structure_refinement_iterations =
                    value()?.parse().map_err(|error| format!("{error}"))?
            }
            "--help" | "-h" => {
                return Err(concat!(
                    "usage: generalized_rig_sfm --manifest FILE --features-dir DIR ",
                    "[--append-features-dir DIR ...] ",
                    "[--deferred-overlay-snapshot FILE ",
                    "--deferred-overlay-min-temporal-frame-gap COUNT ",
                    "--deferred-overlay-max-frame-gap COUNT ",
                    "--deferred-overlay-max-matches-per-pair COUNT] ",
                    "--snapshot FILE [--out-colmap DIR] [--feature-suffix _features.txt] ",
                    "[--max-models 1] [--min-model-frames 10] ",
                    "[--min-pnp-inliers 8] [--min-pnp-sensors 2] ",
                    "[--direct-stereo-pnp-max-frame-gap 0] ",
                    "[--direct-stereo-min-pnp-sensors COUNT] ",
                    "[--direct-stereo-min-triangulation-angle-deg 1] ",
                    "[--motion-bridge-max-frame-gap 0] ",
                    "[--motion-bridge-min-inliers 12] ",
                    "[--motion-bridge-max-rotation-deviation-deg 5] ",
                    "[--deferred-registration-pair-prefix COUNT] ",
                    "[--deferred-registration-min-pnp-sensors COUNT] ",
                    "[--deferred-registration-min-pnp-inliers COUNT] ",
                    "[--deferred-registration-pnp-iterations COUNT] ",
                    "[--deferred-registration-max-interpolation-gap COUNT] ",
                    "[--retriangulate-deferred-tracks-after-registration] ",
                    "[--deferred-retriangulation-metric-temporal-cycle-tracks] ",
                    "[--deferred-retriangulation-metric-temporal-quadrilateral-tracks] ",
                    "[--deferred-retriangulation-quadrilateral-min-frame-gap COUNT] ",
                    "[--deferred-retriangulation-quadrilateral-max-frame-gap COUNT] ",
                    "[--deferred-retriangulation-min-metric-frames COUNT] ",
                    "[--export-deferred-quadrilaterals-tsv FILE] ",
                    "[--deferred-quadrilateral-whitelist-tsv FILE] ",
                    "[--structure-min-pair-matches COUNT] ",
                    "[--deferred-long-pair-min-frame-gap COUNT ",
                    "--structure-long-min-e-inliers COUNT] ",
                    "[--long-pair-pose-prior-images IMAGES_TXT ... ",
                    "--structure-long-max-rotation-disagreement-deg DEG] ",
                    "[--final-rig-rotation-average-max-disagreement-deg DEG] ",
                    "[--final-rig-rotation-average-max-update-deg DEG] ",
                    "[--final-rig-rotation-average-weight-cap COUNT] ",
                    "[--final-rig-translation-average] ",
                    "[--final-rig-translation-average-min-frame-gap 8] ",
                    "[--final-rig-translation-average-min-matches 30] ",
                    "[--final-rig-translation-average-max-direction-error-deg 60] ",
                    "[--final-rig-translation-average-max-update-m 2] ",
                    "[--final-rig-translation-average-iterations 8] ",
                    "[--robust-triangulation-pruning] ",
                    "[--triangulation-min-inlier-fraction 0.5] ",
                    "[--max-reprojection-error-px 4] ",
                    "[--pnp-max-iterations 512] [--max-matches-per-pair 0] ",
                    "[--preview-rig-correspondence-csr|--preview-pair-confidence-conflicts] ",
                    "[--dynamic-correspondence-tracking] ",
                    "[--gr6p-seed-candidate-cap 0] ",
                    "[--gr6p-seed-correspondence-cap 512] ",
                    "[--gr6p-seed-max-iterations 64] ",
                    "[--gr6p-seed-min-iterations 16] ",
                    "[--gr6p-seed-min-inliers 12] ",
                    "[--gr6p-seed-angular-threshold-deg 0.5] ",
                    "[--gr6p-seed-min-positive-depth-fraction 0.8] ",
                    "[--gr6p-seed-min-baseline-m 1e-4] ",
                    "[--max-track-frame-gap 0] ",
                    "[--local-ba-every 10] [--local-ba-window 40] ",
                    "[--local-ba-iterations 8] [--ba-huber-delta 6] ",
                    "[--structure-refinement-iterations 5] ",
                    "[--metric-anchored-cycle-tracks|--metric-temporal-cycle-tracks|",
                    "--metric-sparse-cycle-tracks|",
                    "--sparse-cycle-tracks|",
                    "--pair-confidence-tracks|",
                    "--trusted-pair-prefix COUNT|",
                    "--stream-order-conflict-preserving-tracks|",
                    "--conflict-preserving-tracks|",
                    "--legacy-union-find-tracks] ",
                    "[--recover-metric-conflict-tracks] ",
                    "[--conflict-recovery-max-hypotheses 8] ",
                    "[--conflict-recovery-max-reprojection-error-px 2] ",
                    "[--conflict-recovery-max-mean-reprojection-px 1] ",
                    "[--complete-tracks-after-registration] ",
                    "[--track-completion-max-passes 2] ",
                    "[--track-completion-max-reprojection-error-px 1] ",
                    "[--repair-isolated-pose-outliers] ",
                    "[--isolated-pose-max-midpoint-error-m 0.25] ",
                    "[--isolated-pose-min-detour-ratio 8] ",
                    "[--isolated-pose-repair-max-passes 1] ",
                    "[--repair-paired-pose-jumps] ",
                    "[--paired-pose-jump-absolute-step-m 0.25] ",
                    "[--paired-pose-jump-min-step-ratio 8] ",
                    "[--paired-pose-jump-max-frame-span 16] ",
                    "[--paired-pose-jump-max-closure-ratio 0.1] ",
                    "[--ba-metric-tracks-only|--ba-all-tracks] ",
                    "[--final-ba-min-pose-observations 0] ",
                    "[--ransac-seed 7] [--final-ba|--no-final-ba] ",
                    "[--final-ba-passes 2] [--final-ba-window 60]",
                    " [--final-ba-fix-window-ends|--final-ba-single-anchor] ",
                    "[--final-filter-refinement-passes 0]"
                )
                .into());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if feature_suffix.is_empty() {
        return Err("--feature-suffix must not be empty".into());
    }
    if deferred_overlay_snapshot.is_some() != deferred_overlay_max_frame_gap.is_some() {
        return Err(
            "--deferred-overlay-snapshot and --deferred-overlay-max-frame-gap must be provided together"
                .into(),
        );
    }
    if deferred_overlay_snapshot.is_some() && append_features_dirs.len() != 1 {
        return Err(
            "--deferred-overlay-snapshot requires exactly one --append-features-dir".into(),
        );
    }
    if deferred_overlay_max_matches_per_pair.is_some() && deferred_overlay_snapshot.is_none() {
        return Err(
            "--deferred-overlay-max-matches-per-pair requires --deferred-overlay-snapshot".into(),
        );
    }
    if deferred_retriangulation_min_metric_frames > 0
        && !retriangulate_deferred_tracks_after_registration
    {
        return Err("--deferred-retriangulation-min-metric-frames requires --retriangulate-deferred-tracks-after-registration".into());
    }
    if deferred_retriangulation_metric_temporal_cycle_tracks
        && !retriangulate_deferred_tracks_after_registration
    {
        return Err("--deferred-retriangulation-metric-temporal-cycle-tracks requires --retriangulate-deferred-tracks-after-registration".into());
    }
    if deferred_retriangulation_metric_temporal_quadrilateral_tracks
        && !retriangulate_deferred_tracks_after_registration
    {
        return Err("--deferred-retriangulation-metric-temporal-quadrilateral-tracks requires --retriangulate-deferred-tracks-after-registration".into());
    }
    if deferred_retriangulation_metric_temporal_cycle_tracks
        && deferred_retriangulation_metric_temporal_quadrilateral_tracks
    {
        return Err("the deferred temporal-cycle and temporal-quadrilateral track builders are mutually exclusive".into());
    }
    if deferred_retriangulation_quadrilateral_min_frame_gap == 0
        || deferred_retriangulation_quadrilateral_min_frame_gap
            > deferred_retriangulation_quadrilateral_max_frame_gap
    {
        return Err("deferred quadrilateral frame-gap range is invalid".into());
    }
    if !deferred_retriangulation_metric_temporal_quadrilateral_tracks
        && (deferred_retriangulation_quadrilateral_min_frame_gap != 1
            || deferred_retriangulation_quadrilateral_max_frame_gap != 1)
    {
        return Err("deferred quadrilateral frame-gap flags require --deferred-retriangulation-metric-temporal-quadrilateral-tracks".into());
    }
    if deferred_quadrilateral_whitelist_tsv.is_some()
        && (!deferred_retriangulation_metric_temporal_quadrilateral_tracks
            || deferred_registration_pair_prefix.is_none())
    {
        return Err("--deferred-quadrilateral-whitelist-tsv requires --deferred-retriangulation-metric-temporal-quadrilateral-tracks and --deferred-registration-pair-prefix".into());
    }
    if deferred_overlay_max_frame_gap
        .is_some_and(|maximum| deferred_overlay_min_temporal_frame_gap > maximum)
    {
        return Err("--deferred-overlay-min-temporal-frame-gap cannot exceed the maximum".into());
    }
    if deferred_long_pair_min_frame_gap.is_some() != structure_long_min_e_inliers.is_some() {
        return Err("--deferred-long-pair-min-frame-gap and --structure-long-min-e-inliers must be provided together".into());
    }
    if deferred_long_pair_min_frame_gap == Some(0) {
        return Err("--deferred-long-pair-min-frame-gap must be positive".into());
    }
    if structure_long_min_e_inliers == Some(0) {
        return Err("--structure-long-min-e-inliers must be positive".into());
    }
    if structure_min_pair_matches == Some(0) {
        return Err("--structure-min-pair-matches must be positive".into());
    }
    if (deferred_long_pair_min_frame_gap.is_some() || structure_min_pair_matches.is_some())
        && deferred_registration_pair_prefix.is_some()
    {
        return Err("automatic structure/deferred partition cannot be combined with --deferred-registration-pair-prefix".into());
    }
    if structure_long_max_rotation_disagreement_deg.is_some()
        == long_pair_pose_prior_images.is_empty()
    {
        return Err(
            "--long-pair-pose-prior-images and --structure-long-max-rotation-disagreement-deg must be provided together"
                .into(),
        );
    }
    if structure_long_max_rotation_disagreement_deg.is_some()
        && (deferred_long_pair_min_frame_gap.is_none() || structure_long_min_e_inliers.is_none())
    {
        return Err(
            "rotation-gated long-pair priors require --deferred-long-pair-min-frame-gap and --structure-long-min-e-inliers"
                .into(),
        );
    }
    if let Some(limit) = structure_long_max_rotation_disagreement_deg {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(
                "--structure-long-max-rotation-disagreement-deg must be finite and positive".into(),
            );
        }
    }
    if let Some(limit) = final_rig_rotation_average_max_disagreement_deg {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(
                "--final-rig-rotation-average-max-disagreement-deg must be finite and positive"
                    .into(),
            );
        }
    }
    if !final_rig_rotation_average_max_update_deg.is_finite()
        || final_rig_rotation_average_max_update_deg <= 0.0
    {
        return Err(
            "--final-rig-rotation-average-max-update-deg must be finite and positive".into(),
        );
    }
    if final_rig_rotation_average_weight_cap == 0 {
        return Err("--final-rig-rotation-average-weight-cap must be positive".into());
    }
    if final_rig_translation_average {
        if final_rig_translation_average_min_frame_gap == 0 {
            return Err("--final-rig-translation-average-min-frame-gap must be positive".into());
        }
        if final_rig_translation_average_min_matches < 8 {
            return Err("--final-rig-translation-average-min-matches must be at least 8".into());
        }
        if !final_rig_translation_average_max_direction_error_deg.is_finite()
            || final_rig_translation_average_max_direction_error_deg <= 0.0
            || final_rig_translation_average_max_direction_error_deg >= 90.0
        {
            return Err(
                "--final-rig-translation-average-max-direction-error-deg must be finite and in (0, 90)"
                    .into(),
            );
        }
        if !final_rig_translation_average_max_update_m.is_finite()
            || final_rig_translation_average_max_update_m <= 0.0
        {
            return Err(
                "--final-rig-translation-average-max-update-m must be finite and positive".into(),
            );
        }
        if final_rig_translation_average_iterations == 0 {
            return Err("--final-rig-translation-average-iterations must be positive".into());
        }
    }
    if max_models == 0 {
        return Err("--max-models must be at least 1".into());
    }
    if min_model_frames < 2 {
        return Err("--min-model-frames must be at least 2".into());
    }
    if !triangulation_min_inlier_fraction.is_finite()
        || !(0.5..=1.0).contains(&triangulation_min_inlier_fraction)
    {
        return Err("--triangulation-min-inlier-fraction must be finite and in [0.5, 1]".into());
    }
    let out_colmap = out_colmap.or_else(|| {
        (preview_rig_correspondence_csr || preview_pair_confidence_conflicts).then(PathBuf::new)
    });
    Ok(Args {
        manifest: manifest.ok_or("--manifest is required")?,
        features_dir: features_dir.ok_or("--features-dir is required")?,
        append_features_dirs,
        deferred_overlay_snapshot,
        deferred_overlay_min_temporal_frame_gap,
        deferred_overlay_max_frame_gap,
        deferred_overlay_max_matches_per_pair,
        feature_suffix,
        snapshot: snapshot.ok_or("--snapshot is required")?,
        out_colmap: out_colmap.ok_or("--out-colmap is required")?,
        max_models,
        min_model_frames,
        min_pnp_inliers,
        min_pnp_sensors,
        direct_stereo_pnp_max_frame_gap,
        direct_stereo_min_pnp_sensors,
        direct_stereo_min_triangulation_angle_deg,
        motion_bridge_max_frame_gap,
        motion_bridge_min_inliers,
        motion_bridge_max_rotation_deviation_deg,
        deferred_registration_pair_prefix,
        deferred_retriangulation_pair_prefix,
        deferred_registration_min_pnp_sensors,
        deferred_registration_min_pnp_inliers,
        deferred_registration_pnp_max_iterations,
        deferred_registration_max_interpolation_gap,
        retriangulate_deferred_tracks_after_registration,
        deferred_retriangulation_metric_temporal_cycle_tracks,
        deferred_retriangulation_metric_temporal_quadrilateral_tracks,
        deferred_retriangulation_quadrilateral_min_frame_gap,
        deferred_retriangulation_quadrilateral_max_frame_gap,
        deferred_retriangulation_min_metric_frames,
        export_deferred_quadrilaterals_tsv,
        deferred_quadrilateral_whitelist_tsv,
        structure_min_pair_matches,
        deferred_long_pair_min_frame_gap,
        structure_long_min_e_inliers,
        long_pair_pose_prior_images,
        structure_long_max_rotation_disagreement_deg,
        final_rig_rotation_average_max_disagreement_deg,
        final_rig_rotation_average_max_update_deg,
        final_rig_rotation_average_weight_cap,
        final_rig_translation_average,
        final_rig_translation_average_min_frame_gap,
        final_rig_translation_average_min_matches,
        final_rig_translation_average_max_direction_error_deg,
        final_rig_translation_average_max_update_m,
        final_rig_translation_average_iterations,
        robust_triangulation_pruning,
        triangulation_min_inlier_fraction,
        max_reprojection_error_px,
        pnp_max_iterations,
        final_bundle_adjustment,
        max_matches_per_pair,
        max_track_frame_gap,
        local_ba_every,
        local_ba_window_size,
        local_ba_iterations,
        final_ba_passes,
        final_ba_window_size,
        final_ba_fix_window_ends,
        final_filter_refinement_passes,
        ransac_seed,
        track_builder,
        recover_metric_conflict_tracks,
        conflict_recovery_max_hypotheses,
        conflict_recovery_max_reprojection_error_px,
        conflict_recovery_max_mean_reprojection_px,
        complete_tracks_after_registration,
        track_completion_max_passes,
        track_completion_max_reprojection_error_px,
        repair_isolated_pose_outliers,
        isolated_pose_max_midpoint_error_m,
        isolated_pose_min_detour_ratio,
        isolated_pose_repair_max_passes,
        repair_paired_pose_jumps,
        paired_pose_jump_absolute_step_m,
        paired_pose_jump_min_step_ratio,
        paired_pose_jump_max_frame_span,
        paired_pose_jump_max_closure_ratio,
        ba_metric_tracks_only,
        final_ba_min_pose_observations,
        ba_huber_delta,
        structure_refinement_iterations,
        preview_rig_correspondence_csr,
        preview_pair_confidence_conflicts,
        dynamic_correspondence_tracking,
        gr6p_seed_candidate_cap,
        gr6p_seed_correspondence_cap,
        gr6p_seed_max_iterations,
        gr6p_seed_min_iterations,
        gr6p_seed_min_inliers,
        gr6p_seed_angular_threshold_deg,
        gr6p_seed_min_positive_depth_fraction,
        gr6p_seed_min_baseline_m,
    })
}

#[derive(Debug)]
struct ParsedManifest {
    rig: GeneralizedCameraRig,
    frame_rows: Vec<(u64, String, usize)>,
}

fn token<'a>(tokens: &'a [&str], index: usize, line: usize) -> Result<&'a str, String> {
    tokens
        .get(index)
        .copied()
        .ok_or_else(|| format!("manifest line {line}: missing field {index}"))
}

fn number<T: std::str::FromStr>(tokens: &[&str], index: usize, line: usize) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    token(tokens, index, line)?
        .parse()
        .map_err(|error| format!("manifest line {line} field {index}: {error}"))
}

fn parse_manifest(path: &Path) -> Result<ParsedManifest, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read manifest {}: {error}", path.display()))?;
    let mut sensors = BTreeMap::new();
    let mut frame_rows = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let line = zero_line + 1;
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let tokens = text.split_whitespace().collect::<Vec<_>>();
        match token(&tokens, 0, line)? {
            "S" => {
                if tokens.len() != 16 {
                    return Err(format!(
                        "manifest line {line}: sensor row requires 16 fields, found {}",
                        tokens.len()
                    ));
                }
                let index: usize = number(&tokens, 1, line)?;
                let camera = Camera::pinhole(
                    number(&tokens, 2, line)?,
                    number(&tokens, 3, line)?,
                    number(&tokens, 4, line)?,
                    number(&tokens, 5, line)?,
                    number(&tokens, 6, line)?,
                    number(&tokens, 7, line)?,
                    number(&tokens, 8, line)?,
                );
                let quaternion: Quaternion<f64> = Quaternion::new(
                    number(&tokens, 9, line)?,
                    number(&tokens, 10, line)?,
                    number(&tokens, 11, line)?,
                    number(&tokens, 12, line)?,
                );
                let norm = quaternion.norm();
                if !norm.is_finite() || norm <= 1.0e-12 {
                    return Err(format!("manifest line {line}: invalid sensor quaternion"));
                }
                let sensor = RigSensor {
                    camera,
                    sensor_from_rig: SE3::new(
                        UnitQuaternion::new_normalize(quaternion),
                        Vector3::new(
                            number(&tokens, 13, line)?,
                            number(&tokens, 14, line)?,
                            number(&tokens, 15, line)?,
                        ),
                    ),
                };
                if sensors.insert(index, sensor).is_some() {
                    return Err(format!("manifest line {line}: duplicate sensor {index}"));
                }
            }
            "F" => {
                if tokens.len() != 4 {
                    return Err(format!(
                        "manifest line {line}: frame row requires 4 fields, found {}",
                        tokens.len()
                    ));
                }
                frame_rows.push((
                    number(&tokens, 1, line)?,
                    token(&tokens, 2, line)?.to_owned(),
                    number(&tokens, 3, line)?,
                ));
            }
            kind => return Err(format!("manifest line {line}: unknown row kind {kind}")),
        }
    }
    if sensors.len() < 2 || frame_rows.is_empty() {
        return Err("manifest requires at least two sensors and one frame row".into());
    }
    if sensors.keys().copied().ne(0..sensors.len()) {
        return Err("manifest sensor indices must be contiguous from zero".into());
    }
    let rig = GeneralizedCameraRig::new(sensors.into_values().collect())
        .ok_or("manifest contains invalid rig calibration")?;
    Ok(ParsedManifest { rig, frame_rows })
}

fn feature_path(directory: &Path, image_name: &str, suffix: &str) -> Result<PathBuf, String> {
    let stem = Path::new(image_name)
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("image name has no UTF-8 stem: {image_name:?}"))?;
    Ok(directory.join(format!("{stem}{suffix}")))
}

fn read_keypoints(path: &Path) -> Result<FeatureSet, String> {
    let contents = std::fs::read_to_string(path)
        .map_err(|error| format!("read feature file {}: {error}", path.display()))?;
    let mut keypoints = Vec::new();
    for (zero_line, raw) in contents.lines().enumerate() {
        let text = raw.trim();
        if text.is_empty() || text.starts_with('#') {
            continue;
        }
        let mut fields = text.split_whitespace();
        let x = fields
            .next()
            .ok_or_else(|| format!("{}:{} missing x", path.display(), zero_line + 1))?
            .parse::<f64>()
            .map_err(|error| format!("{}:{} invalid x: {error}", path.display(), zero_line + 1))?;
        let y = fields
            .next()
            .ok_or_else(|| format!("{}:{} missing y", path.display(), zero_line + 1))?
            .parse::<f64>()
            .map_err(|error| format!("{}:{} invalid y: {error}", path.display(), zero_line + 1))?;
        keypoints.push(Point2::new(x, y));
    }
    let descriptors = vec![Vec::new(); keypoints.len()];
    FeatureSet::new(keypoints, descriptors).map_err(|error| error.to_string())
}

fn read_combined_keypoints(
    primary_directory: &Path,
    append_directories: &[PathBuf],
    image_name: &str,
    suffix: &str,
) -> Result<FeatureSet, String> {
    let primary_path = feature_path(primary_directory, image_name, suffix)?;
    let mut combined = read_keypoints(&primary_path)?;
    for directory in append_directories {
        let path = feature_path(directory, image_name, suffix)?;
        let mut appended = read_keypoints(&path)?;
        combined.keypoints.append(&mut appended.keypoints);
        combined.descriptors.append(&mut appended.descriptors);
    }
    combined.validate().map_err(|error| error.to_string())?;
    Ok(combined)
}

fn build_frames(
    rows: &[(u64, String, usize)],
    image_names: &[String],
) -> Result<Vec<RigFrame>, String> {
    let image_indices = image_names
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect::<HashMap<_, _>>();
    let mut grouped: BTreeMap<u64, Vec<RigFrameImage>> = BTreeMap::new();
    for (frame_id, image_name, sensor_index) in rows {
        let image_index = image_indices
            .get(image_name.as_str())
            .copied()
            .ok_or_else(|| format!("manifest image {image_name:?} is absent from snapshot"))?;
        grouped.entry(*frame_id).or_default().push(RigFrameImage {
            image_index,
            sensor_index: *sensor_index,
        });
    }
    let assigned = grouped.values().map(Vec::len).sum::<usize>();
    if assigned != image_names.len() {
        return Err(format!(
            "manifest assigns {assigned} images but snapshot contains {}",
            image_names.len()
        ));
    }
    Ok(grouped
        .into_values()
        .map(|images| RigFrame { images })
        .collect())
}

fn convert_pairs(
    snapshot: &verified_pair_snapshot::Snapshot,
) -> Result<Vec<PairwiseMatches>, String> {
    snapshot
        .pairs
        .iter()
        .map(|pair| {
            let image_i = usize::try_from(pair.image_i)
                .map_err(|_| "snapshot image_i does not fit usize".to_owned())?;
            let image_j = usize::try_from(pair.image_j)
                .map_err(|_| "snapshot image_j does not fit usize".to_owned())?;
            let matches = pair
                .matches
                .iter()
                .map(|&(left, right)| {
                    Ok((
                        usize::try_from(left)
                            .map_err(|_| "snapshot left keypoint does not fit usize".to_owned())?,
                        usize::try_from(right)
                            .map_err(|_| "snapshot right keypoint does not fit usize".to_owned())?,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(PairwiseMatches::new(image_i, image_j, matches))
        })
        .collect()
}

fn verifier_overlay_compatible(base: &str, overlay: &str) -> Result<bool, String> {
    let parse = |config: &str| -> Result<(usize, HashMap<String, String>), String> {
        let mut fields = HashMap::new();
        for field in config.split(';') {
            let (key, value) = field
                .split_once('=')
                .ok_or_else(|| "verifier configuration contains a malformed field".to_owned())?;
            if fields.insert(key.to_owned(), value.to_owned()).is_some() {
                return Err(format!("verifier configuration repeats field {key:?}"));
            }
        }
        let minimum = fields
            .remove("min_matches")
            .ok_or_else(|| "verifier configuration omits min_matches".to_owned())?
            .parse::<usize>()
            .map_err(|error| format!("invalid verifier min_matches: {error}"))?;
        Ok((minimum, fields))
    };
    let (base_minimum, base_fields) = parse(base)?;
    let (overlay_minimum, overlay_fields) = parse(overlay)?;
    Ok(base_fields == overlay_fields && overlay_minimum >= base_minimum)
}

fn shift_overlay_pair(
    pair: &mut verified_pair_snapshot::PairRecord,
    base_counts: &[u64],
) -> Result<(), String> {
    let image_i = usize::try_from(pair.image_i)
        .map_err(|_| "overlay image_i does not fit usize".to_owned())?;
    let image_j = usize::try_from(pair.image_j)
        .map_err(|_| "overlay image_j does not fit usize".to_owned())?;
    let left_offset = *base_counts
        .get(image_i)
        .ok_or_else(|| "overlay image_i is outside base feature counts".to_owned())?;
    let right_offset = *base_counts
        .get(image_j)
        .ok_or_else(|| "overlay image_j is outside base feature counts".to_owned())?;
    let shift = |matched: &mut (u64, u64)| -> Result<(), String> {
        matched.0 = matched
            .0
            .checked_add(left_offset)
            .ok_or_else(|| "overlay left keypoint overflows u64".to_owned())?;
        matched.1 = matched
            .1
            .checked_add(right_offset)
            .ok_or_else(|| "overlay right keypoint overflows u64".to_owned())?;
        Ok(())
    };
    for matched in &mut pair.raw_matches {
        shift(matched)?;
    }
    for matched in &mut pair.matches {
        shift(matched)?;
    }
    if let Some(matches) = pair.essential_matches.as_mut() {
        for matched in matches {
            shift(matched)?;
        }
    }
    Ok(())
}

fn append_deferred_overlay(
    base: &mut verified_pair_snapshot::Snapshot,
    mut overlay: verified_pair_snapshot::Snapshot,
    frames: &[RigFrame],
    min_temporal_frame_gap: usize,
    max_frame_gap: usize,
) -> Result<(usize, usize), String> {
    if base.image_names != overlay.image_names
        || base.image_manifest_hash != overlay.image_manifest_hash
    {
        return Err("base and deferred overlay image manifests differ".into());
    }
    if base.width != overlay.width
        || base.height != overlay.height
        || base.intrinsics_bits != overlay.intrinsics_bits
    {
        return Err("base and deferred overlay camera envelopes differ".into());
    }
    if !verifier_overlay_compatible(&base.verifier_config, &overlay.verifier_config)? {
        return Err(
            "deferred overlay verifier must differ only by an equal or stricter min_matches".into(),
        );
    }
    if base.feature_counts.len() != overlay.feature_counts.len() {
        return Err("base and deferred overlay feature-count vectors differ in length".into());
    }
    let mut image_frames = vec![None; base.image_names.len()];
    for (frame, rig_frame) in frames.iter().enumerate() {
        for image in &rig_frame.images {
            let slot = image_frames
                .get_mut(image.image_index)
                .ok_or_else(|| "rig image is outside deferred overlay manifest".to_owned())?;
            if slot.replace(frame).is_some() {
                return Err("rig assigns an image to multiple frames".into());
            }
        }
    }
    if image_frames.iter().any(Option::is_none) {
        return Err("rig omits an image from the deferred overlay manifest".into());
    }
    let base_counts = base.feature_counts.clone();
    let mut retained = Vec::new();
    let mut matches = 0usize;
    for mut pair in overlay.pairs.drain(..) {
        let image_i = usize::try_from(pair.image_i)
            .map_err(|_| "overlay image_i does not fit usize".to_owned())?;
        let image_j = usize::try_from(pair.image_j)
            .map_err(|_| "overlay image_j does not fit usize".to_owned())?;
        let frame_i = image_frames
            .get(image_i)
            .copied()
            .flatten()
            .ok_or_else(|| "overlay image_i is outside rig frames".to_owned())?;
        let frame_j = image_frames
            .get(image_j)
            .copied()
            .flatten()
            .ok_or_else(|| "overlay image_j is outside rig frames".to_owned())?;
        let frame_gap = frame_i.abs_diff(frame_j);
        if frame_gap > max_frame_gap || (frame_gap != 0 && frame_gap < min_temporal_frame_gap) {
            continue;
        }
        shift_overlay_pair(&mut pair, &base_counts)?;
        matches = matches
            .checked_add(pair.matches.len())
            .ok_or_else(|| "overlay match count overflows usize".to_owned())?;
        retained.push(pair);
    }
    let pair_count = retained.len();
    base.pairs.extend(retained);
    for (base_count, overlay_count) in base.feature_counts.iter_mut().zip(&overlay.feature_counts) {
        *base_count = base_count
            .checked_add(*overlay_count)
            .ok_or_else(|| "combined feature count overflows u64".to_owned())?;
    }
    base.accepted_match_count = base
        .accepted_match_count
        .checked_add(matches as u64)
        .ok_or_else(|| "combined accepted match count overflows u64".to_owned())?;
    Ok((pair_count, matches))
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let rest = line.strip_prefix("VmHWM:")?;
        rest.split_whitespace().next()?.parse().ok()
    })
}

#[allow(clippy::too_many_arguments)]
fn export_deferred_quadrilaterals(
    path: &Path,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairs: &[PairwiseMatches],
    image_names: &[String],
    deferred_pair_prefix: usize,
    min_frame_gap: usize,
    max_frame_gap: usize,
) -> Result<usize, Box<dyn Error>> {
    if deferred_pair_prefix > pairs.len() {
        return Err(format!(
            "deferred pair prefix {deferred_pair_prefix} exceeds {} pairs",
            pairs.len()
        )
        .into());
    }
    let mut assignment = vec![(usize::MAX, usize::MAX); features.len()];
    for (frame, rig_frame) in frames.iter().enumerate() {
        for image in &rig_frame.images {
            assignment[image.image_index] = (frame, image.sensor_index);
        }
    }
    let tracks = metric_temporal_quadrilateral_tracks_in_frame_gap(
        features,
        &pairs[deferred_pair_prefix..],
        &assignment,
        min_frame_gap.max(1),
        max_frame_gap,
    );
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let mut output = BufWriter::new(File::create(path)?);
    write!(output, "track")?;
    for observation in 0..4 {
        write!(
            output,
            "\timage_{observation}\tkeypoint_{observation}\tframe_{observation}\tsensor_{observation}\tname_{observation}\tx_{observation}\ty_{observation}"
        )?;
    }
    writeln!(output)?;
    for (track_index, track) in tracks.iter().enumerate() {
        if track.len() != 4 {
            return Err(format!(
                "quadrilateral track {track_index} has {} observations",
                track.len()
            )
            .into());
        }
        write!(output, "{track_index}")?;
        for &(image, keypoint) in track {
            let point = features
                .get(image)
                .and_then(|feature| feature.keypoints.get(keypoint))
                .ok_or_else(|| {
                    format!("quadrilateral observation ({image}, {keypoint}) is invalid")
                })?;
            let (frame, sensor) = assignment[image];
            write!(
                output,
                "\t{image}\t{keypoint}\t{frame}\t{sensor}\t{}\t{:.9}\t{:.9}",
                image_names[image], point.x, point.y
            )?;
        }
        writeln!(output)?;
    }
    output.flush()?;
    Ok(tracks.len())
}

type ObservationEdge = ((usize, usize), (usize, usize));

fn canonical_observation_edge(left: (usize, usize), right: (usize, usize)) -> ObservationEdge {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn read_quadrilateral_whitelist(path: &Path) -> Result<HashSet<ObservationEdge>, Box<dyn Error>> {
    let input = std::fs::read_to_string(path)?;
    let mut lines = input.lines();
    let header = lines.next().ok_or("quadrilateral whitelist is empty")?;
    let columns = header.split('\t').collect::<Vec<_>>();
    let required = (0..4)
        .flat_map(|index| [format!("image_{index}"), format!("keypoint_{index}")])
        .collect::<Vec<_>>();
    let positions = required
        .iter()
        .map(|name| {
            columns
                .iter()
                .position(|column| *column == name)
                .ok_or_else(|| format!("quadrilateral whitelist is missing column {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut edges = HashSet::new();
    for (line_index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let values = line.split('\t').collect::<Vec<_>>();
        let observations = (0..4)
            .map(|index| {
                let image = values
                    .get(positions[index * 2])
                    .ok_or_else(|| format!("whitelist line {} is truncated", line_index + 2))?
                    .parse::<usize>()
                    .map_err(|error| {
                        format!(
                            "invalid image on whitelist line {}: {error}",
                            line_index + 2
                        )
                    })?;
                let keypoint = values
                    .get(positions[index * 2 + 1])
                    .ok_or_else(|| format!("whitelist line {} is truncated", line_index + 2))?
                    .parse::<usize>()
                    .map_err(|error| {
                        format!(
                            "invalid keypoint on whitelist line {}: {error}",
                            line_index + 2
                        )
                    })?;
                Ok::<_, String>((image, keypoint))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if observations.iter().collect::<HashSet<_>>().len() != 4 {
            return Err(format!("whitelist line {} repeats an observation", line_index + 2).into());
        }
        for left in 0..4 {
            for right in (left + 1)..4 {
                edges.insert(canonical_observation_edge(
                    observations[left],
                    observations[right],
                ));
            }
        }
    }
    Ok(edges)
}

fn filter_deferred_pairs_by_quadrilateral_whitelist(
    pairs: &mut [PairwiseMatches],
    prefix: usize,
    edges: &HashSet<ObservationEdge>,
) -> Result<(usize, usize), String> {
    if prefix > pairs.len() {
        return Err(format!(
            "deferred pair prefix {prefix} exceeds {} pairs",
            pairs.len()
        ));
    }
    let before = pairs[prefix..]
        .iter()
        .map(|pair| pair.matches.len())
        .sum::<usize>();
    for pair in &mut pairs[prefix..] {
        pair.matches.retain(|&(left, right)| {
            edges.contains(&canonical_observation_edge(
                (pair.image_i, left),
                (pair.image_j, right),
            ))
        });
    }
    let after = pairs[prefix..]
        .iter()
        .map(|pair| pair.matches.len())
        .sum::<usize>();
    Ok((before, after))
}

fn mapper_config(args: &Args) -> RigSfmConfig {
    RigSfmConfig {
        min_pnp_inliers: args.min_pnp_inliers,
        min_pnp_sensors: args.min_pnp_sensors,
        robust_triangulation_pruning: args.robust_triangulation_pruning,
        triangulation_min_inlier_fraction: args.triangulation_min_inlier_fraction,
        direct_stereo_pnp_max_frame_gap: args.direct_stereo_pnp_max_frame_gap,
        direct_stereo_min_pnp_sensors: args.direct_stereo_min_pnp_sensors,
        direct_stereo_min_triangulation_angle_deg: args.direct_stereo_min_triangulation_angle_deg,
        motion_bridge_max_frame_gap: args.motion_bridge_max_frame_gap,
        motion_bridge_min_inliers: args.motion_bridge_min_inliers,
        motion_bridge_max_rotation_deviation_deg: args.motion_bridge_max_rotation_deviation_deg,
        deferred_registration_pair_prefix: args.deferred_registration_pair_prefix,
        deferred_retriangulation_pair_prefix: args.deferred_retriangulation_pair_prefix,
        deferred_registration_min_pnp_sensors: args.deferred_registration_min_pnp_sensors,
        deferred_registration_min_pnp_inliers: args.deferred_registration_min_pnp_inliers,
        deferred_registration_pnp_max_iterations: args.deferred_registration_pnp_max_iterations,
        deferred_registration_max_interpolation_gap: args
            .deferred_registration_max_interpolation_gap,
        retriangulate_deferred_tracks_after_registration: args
            .retriangulate_deferred_tracks_after_registration,
        deferred_retriangulation_metric_temporal_cycle_tracks: args
            .deferred_retriangulation_metric_temporal_cycle_tracks,
        deferred_retriangulation_metric_temporal_quadrilateral_tracks: args
            .deferred_retriangulation_metric_temporal_quadrilateral_tracks,
        deferred_retriangulation_quadrilateral_min_frame_gap: args
            .deferred_retriangulation_quadrilateral_min_frame_gap,
        deferred_retriangulation_quadrilateral_max_frame_gap: args
            .deferred_retriangulation_quadrilateral_max_frame_gap,
        deferred_retriangulation_min_metric_frames: args.deferred_retriangulation_min_metric_frames,
        max_reprojection_error_px: args.max_reprojection_error_px,
        pnp_max_iterations: args.pnp_max_iterations,
        final_bundle_adjustment: args.final_bundle_adjustment,
        local_ba_every: args.local_ba_every,
        local_ba_window_size: args.local_ba_window_size,
        local_ba_iterations: args.local_ba_iterations,
        final_ba_passes: args.final_ba_passes,
        final_ba_window_size: args.final_ba_window_size,
        final_ba_fix_window_ends: args.final_ba_fix_window_ends,
        final_filter_refinement_passes: args.final_filter_refinement_passes,
        ransac_seed: args.ransac_seed,
        track_builder: args.track_builder,
        recover_metric_conflict_tracks: args.recover_metric_conflict_tracks,
        conflict_recovery_max_hypotheses: args.conflict_recovery_max_hypotheses,
        conflict_recovery_max_reprojection_error_px: args
            .conflict_recovery_max_reprojection_error_px,
        conflict_recovery_max_mean_reprojection_px: args.conflict_recovery_max_mean_reprojection_px,
        complete_tracks_after_registration: args.complete_tracks_after_registration,
        track_completion_max_passes: args.track_completion_max_passes,
        track_completion_max_reprojection_error_px: args.track_completion_max_reprojection_error_px,
        repair_isolated_pose_outliers: args.repair_isolated_pose_outliers,
        isolated_pose_max_midpoint_error_m: args.isolated_pose_max_midpoint_error_m,
        isolated_pose_min_detour_ratio: args.isolated_pose_min_detour_ratio,
        isolated_pose_repair_max_passes: args.isolated_pose_repair_max_passes,
        repair_paired_pose_jumps: args.repair_paired_pose_jumps,
        paired_pose_jump_absolute_step_m: args.paired_pose_jump_absolute_step_m,
        paired_pose_jump_min_step_ratio: args.paired_pose_jump_min_step_ratio,
        paired_pose_jump_max_frame_span: args.paired_pose_jump_max_frame_span,
        paired_pose_jump_max_closure_ratio: args.paired_pose_jump_max_closure_ratio,
        ba_metric_tracks_only: args.ba_metric_tracks_only,
        final_ba_min_pose_observations: args.final_ba_min_pose_observations,
        structure_refinement_iterations: args.structure_refinement_iterations,
        dynamic_correspondence_tracking: args.dynamic_correspondence_tracking,
        gr6p_seed_candidate_cap: args.gr6p_seed_candidate_cap,
        gr6p_seed_correspondence_cap: args.gr6p_seed_correspondence_cap,
        gr6p_seed_max_iterations: args.gr6p_seed_max_iterations,
        gr6p_seed_min_iterations: args.gr6p_seed_min_iterations,
        gr6p_seed_min_inliers: args.gr6p_seed_min_inliers,
        gr6p_seed_angular_threshold_deg: args.gr6p_seed_angular_threshold_deg,
        gr6p_seed_min_positive_depth_fraction: args.gr6p_seed_min_positive_depth_fraction,
        gr6p_seed_min_baseline_m: args.gr6p_seed_min_baseline_m,
        ba_config: visloc_rs::BaConfig {
            robust_kernel: RobustKernel::Huber {
                delta: args.ba_huber_delta,
            },
            ..RigSfmConfig::default().ba_config
        },
        ..RigSfmConfig::default()
    }
}

struct RemainingModelInput {
    frames: Vec<RigFrame>,
    features: Vec<FeatureSet>,
    pairs: Vec<PairwiseMatches>,
    /// The verifier metadata must follow the compacted pair stream exactly:
    /// image indices are local to this model, while all rotation/evidence
    /// fields remain byte-for-byte copies of the snapshot records.
    snapshot_pairs: Vec<verified_pair_snapshot::PairRecord>,
    local_to_global_frames: Vec<usize>,
    local_to_global_images: Vec<usize>,
    base_pair_count: usize,
    registration_pair_count: usize,
}

fn remaining_model_input(
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairs: &[PairwiseMatches],
    snapshot_pairs: &[verified_pair_snapshot::PairRecord],
    remaining_frames: &[bool],
    base_pair_prefix: Option<usize>,
    retriangulation_pair_prefix: Option<usize>,
) -> RemainingModelInput {
    let mut local_to_global_frames = Vec::new();
    let mut local_to_global_images = Vec::new();
    let mut global_to_local_images = vec![None; features.len()];
    let mut local_frames = Vec::new();
    for (global_frame, frame) in frames.iter().enumerate() {
        if !remaining_frames[global_frame] {
            continue;
        }
        local_to_global_frames.push(global_frame);
        let mut images = Vec::with_capacity(frame.images.len());
        for image in &frame.images {
            let local_image = local_to_global_images.len();
            global_to_local_images[image.image_index] = Some(local_image);
            local_to_global_images.push(image.image_index);
            images.push(RigFrameImage {
                image_index: local_image,
                sensor_index: image.sensor_index,
            });
        }
        local_frames.push(RigFrame { images });
    }
    let local_features = local_to_global_images
        .iter()
        .map(|&global_image| features[global_image].clone())
        .collect::<Vec<_>>();
    let mut base_pair_count = 0usize;
    let mut registration_pair_count = 0usize;
    debug_assert_eq!(
        pairs.len(),
        snapshot_pairs.len(),
        "mapper pairs and snapshot metadata must retain the same stream"
    );
    let mut local_snapshot_pairs = Vec::new();
    let local_pairs = pairs
        .iter()
        .enumerate()
        .filter_map(|(pair_index, pair)| {
            let image_i = global_to_local_images
                .get(pair.image_i)
                .copied()
                .flatten()?;
            let image_j = global_to_local_images
                .get(pair.image_j)
                .copied()
                .flatten()?;
            let snapshot_pair = snapshot_pairs.get(pair_index)?;
            base_pair_count +=
                usize::from(base_pair_prefix.is_some_and(|prefix| pair_index < prefix));
            registration_pair_count +=
                usize::from(retriangulation_pair_prefix.is_some_and(|prefix| pair_index < prefix));
            let mut local = pair.clone();
            local.image_i = image_i;
            local.image_j = image_j;
            let mut local_snapshot = snapshot_pair.clone();
            local_snapshot.image_i = image_i as u64;
            local_snapshot.image_j = image_j as u64;
            local_snapshot_pairs.push(local_snapshot);
            Some(local)
        })
        .collect::<Vec<_>>();
    RemainingModelInput {
        frames: local_frames,
        features: local_features,
        pairs: local_pairs,
        snapshot_pairs: local_snapshot_pairs,
        local_to_global_frames,
        local_to_global_images,
        base_pair_count,
        registration_pair_count,
    }
}

fn export_rig_model(
    output: &Path,
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    global_image_names: &[String],
    local_to_global_images: &[usize],
    result: &RigSfmResult,
) -> Result<(usize, usize), Box<dyn Error>> {
    let registered = result
        .image_poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.as_ref().map(|pose| (image, pose.clone())))
        .collect::<Vec<_>>();
    let remap = registered
        .iter()
        .enumerate()
        .map(|(output, (input, _))| (*input, output))
        .collect::<HashMap<_, _>>();
    let image_sensors = frames
        .iter()
        .flat_map(|frame| &frame.images)
        .map(|image| (image.image_index, image.sensor_index))
        .collect::<HashMap<_, _>>();
    let cameras = registered
        .iter()
        .map(|(image, _)| rig.sensors()[image_sensors[image]].camera.clone())
        .collect::<Vec<_>>();
    let poses = registered
        .iter()
        .map(|(_, pose)| pose.clone())
        .collect::<Vec<_>>();
    let names = registered
        .iter()
        .map(|(image, _)| global_image_names[local_to_global_images[*image]].clone())
        .collect::<Vec<_>>();
    let output_features = registered
        .iter()
        .map(|(image, _)| features[*image].clone())
        .collect::<Vec<_>>();
    let landmarks = result
        .tracks
        .iter()
        .map(|track| {
            let observations = track
                .observations
                .iter()
                .filter_map(|&(image, keypoint, pixel)| {
                    remap.get(&image).map(|output| (*output, keypoint, pixel))
                })
                .collect::<Vec<_>>();
            (track.position, observations)
        })
        .collect::<Vec<_>>();
    let export = write_colmap_reconstruction_for_3dgs_with_cameras(
        output,
        &cameras,
        &poses,
        &output_features,
        &landmarks,
        |index| names[index].clone(),
    )?;
    Ok((export.landmark_count, export.observation_count))
}

#[allow(clippy::too_many_arguments)]
fn map_remaining_models(
    args: &Args,
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    pairs: &[PairwiseMatches],
    snapshot_pairs: &[verified_pair_snapshot::PairRecord],
    image_names: &[String],
    mapping_started: Instant,
) -> Result<(), Box<dyn Error>> {
    let mut remaining_frames = vec![true; frames.len()];
    let mut total_registered_frames = 0usize;
    let mut total_registered_images = 0usize;
    let mut total_tracks = 0usize;
    let mut total_observations = 0usize;
    let mut weighted_reprojection = 0.0;
    let mut manifest = String::from(
        "rank\tsupplied_frames\tsupplied_images\tverified_pairs\tregistered_frames\tregistered_images\ttracks\tobservations\tmean_reprojection_px\tseed_global_frame\tpnp_attempts\tpnp_insufficient_sensors\tpnp_estimation_failures\tpnp_inlier_rejections\tpnp_registrations\tdirect_bridge_pair_visits\tdirect_bridge_correspondences\tdirect_bridge_registrations\tmotion_bridge_pair_visits\tmotion_bridge_estimation_failures\tmotion_bridge_rotation_rejections\tmotion_bridge_registrations\tdeferred_pair_visits\tdeferred_correspondences\tdeferred_pnp_attempts\tdeferred_pnp_estimation_failures\tdeferred_pnp_inlier_rejections\tdeferred_registrations\tdeferred_interpolation_registrations\tdeferred_observations_attached\tunregistered_zero_support\tunregistered_below_pnp_support\tunregistered_eligible_pnp\tunregistered_below_sensors\tmax_unregistered_support\tpaired_pose_jump_repairs\tpaired_pose_jump_repaired_frames\tmodel_dir\n",
    );
    let mut retrieval_components = String::from("# retrieval-component-manifest-v1\n");
    let mut emitted_models = 0usize;
    for rank in 0..args.max_models {
        let input = remaining_model_input(
            frames,
            features,
            pairs,
            snapshot_pairs,
            &remaining_frames,
            args.deferred_registration_pair_prefix,
            args.deferred_retriangulation_pair_prefix,
        );
        if input.frames.len() < args.min_model_frames {
            break;
        }
        let supplied_frames = input.frames.len();
        let supplied_images = input.features.len();
        let verified_pairs = input.pairs.len();
        let mut config = mapper_config(args);
        if args.deferred_registration_pair_prefix.is_some() {
            config.deferred_registration_pair_prefix = Some(input.base_pair_count);
        }
        if args.deferred_retriangulation_pair_prefix.is_some() {
            config.deferred_retriangulation_pair_prefix = Some(input.registration_pair_count);
        }
        if rank > 0 {
            // `remaining_model_input` compacts away frames claimed by earlier
            // models, so its local frame indices are not a continuous time
            // axis across component boundaries. Disable every operation that
            // interprets adjacent local indices as adjacent timestamps.
            config.deferred_registration_max_interpolation_gap = 0;
            config.repair_isolated_pose_outliers = false;
            config.repair_paired_pose_jumps = false;
        }
        let mut result = match incremental_rig_sfm(
            rig,
            &input.frames,
            &input.features,
            &input.pairs,
            &config,
        ) {
            Ok(result) => result,
            Err(
                error @ (RigSfmError::NoMetricSeed | RigSfmError::InsufficientSeedStructure { .. }),
            ) if rank > 0 => {
                eprintln!(
                        "rig-component-stop: rank={rank} supplied_frames={supplied_frames} verified_pairs={verified_pairs} reason={error}"
                    );
                break;
            }
            Err(error) => return Err(error.into()),
        };
        if let Some(max_disagreement_deg) = args.final_rig_rotation_average_max_disagreement_deg {
            let rotation_stats = apply_final_rig_rotation_average(
                rig,
                &input.frames,
                &input.features,
                &input.snapshot_pairs,
                max_disagreement_deg,
                args.final_rig_rotation_average_max_update_deg,
                args.final_rig_rotation_average_weight_cap,
                &config,
                &mut result,
            )?;
            eprintln!(
                "rig-component rotation-average: rank={rank} max_disagreement_deg={max_disagreement_deg:.6} max_update_deg={:.6} weight_cap={} candidate_pairs={} accepted_pairs={} deduped_pairs={} registered_components={} averaged_components={} failed_components={} unsafe_update_components={} averaged_frames={} disagreement_rejections={} max_proposed_update_deg={:.6} max_applied_update_deg={:.6} support_rollback={}",
                args.final_rig_rotation_average_max_update_deg,
                args.final_rig_rotation_average_weight_cap,
                rotation_stats.candidate_pairs,
                rotation_stats.accepted_pairs,
                rotation_stats.deduped_pairs,
                rotation_stats.registered_components,
                rotation_stats.averaged_components,
                rotation_stats.failed_components,
                rotation_stats.unsafe_update_components,
                rotation_stats.averaged_frames,
                rotation_stats.disagreement_rejections,
                rotation_stats.max_proposed_update_deg,
                rotation_stats.max_rotation_update_deg,
                rotation_stats.support_rollback,
            );
        }
        if args.final_rig_translation_average && rank == 0 {
            let translation_stats = apply_final_rig_translation_average(
                rig,
                &input.frames,
                &input.features,
                &input.snapshot_pairs,
                args.final_rig_translation_average_min_frame_gap,
                args.final_rig_translation_average_min_matches,
                args.final_rig_translation_average_max_direction_error_deg,
                args.final_rig_translation_average_max_update_m,
                args.final_rig_translation_average_iterations,
                &config,
                &mut result,
            )?;
            eprintln!(
                "rig-component translation-average: rank={rank} candidate_pairs={} accepted_pairs={} degenerate_pairs={} direction_rejections={} deduped_edges={} backbone_edges={} solved_frames={} max_proposed_update_m={:.9} candidate_tracks={} candidate_observations={} candidate_mean_reprojection_px={:.9} support_regressed={} reprojection_regressed={} unsafe_update={} support_rollback={}",
                translation_stats.candidate_pairs,
                translation_stats.accepted_pairs,
                translation_stats.degenerate_pairs,
                translation_stats.direction_rejections,
                translation_stats.deduped_edges,
                translation_stats.backbone_edges,
                translation_stats.solved_frames,
                translation_stats.max_proposed_update_m,
                translation_stats.candidate_tracks,
                translation_stats.candidate_observations,
                translation_stats.candidate_mean_reprojection_px,
                translation_stats.support_regressed,
                translation_stats.reprojection_regressed,
                translation_stats.unsafe_update,
                translation_stats.support_rollback,
            );
        }
        if result.registered_frames == 0 {
            break;
        }
        for (local_frame, pose) in result.frame_poses.iter().enumerate() {
            if pose.is_some() {
                remaining_frames[input.local_to_global_frames[local_frame]] = false;
            }
        }
        if result.registered_frames < args.min_model_frames {
            continue;
        }
        for (local_frame, pose) in result.frame_poses.iter().enumerate() {
            if pose.is_none() {
                continue;
            }
            for image in &input.frames[local_frame].images {
                let global_image = input.local_to_global_images[image.image_index];
                retrieval_components.push_str(&format!("C {rank} {}\n", image_names[global_image]));
            }
        }
        let model_dir = args.out_colmap.join(format!("component-{rank:03}"));
        let (tracks, observations) = export_rig_model(
            &model_dir,
            rig,
            &input.frames,
            &input.features,
            image_names,
            &input.local_to_global_images,
            &result,
        )?;
        let seed_global_frame = input.local_to_global_frames[result.seed_frame_index];
        total_registered_frames += result.registered_frames;
        total_registered_images += result.registered_images;
        total_tracks += tracks;
        total_observations += observations;
        weighted_reprojection += result.mean_reprojection_error_px * observations as f64;
        emitted_models += 1;
        manifest.push_str(&format!(
            "{rank}\t{supplied_frames}\t{supplied_images}\t{verified_pairs}\t{}\t{}\t{tracks}\t{observations}\t{:.9}\t{seed_global_frame}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tcomponent-{rank:03}\n",
            result.registered_frames,
            result.registered_images,
            result.mean_reprojection_error_px,
            result.work.pnp_attempts,
            result.work.pnp_insufficient_sensor_attempts,
            result.work.pnp_estimation_failures,
            result.work.pnp_inlier_rejections,
            result.work.pnp_registrations,
            result.work.direct_bridge_pair_visits,
            result.work.direct_bridge_correspondence_insertions,
            result.work.direct_bridge_registrations,
            result.work.motion_bridge_pair_visits,
            result.work.motion_bridge_estimation_failures,
            result.work.motion_bridge_rotation_rejections,
            result.work.motion_bridge_registrations,
            result.work.deferred_pair_visits,
            result.work.deferred_correspondence_insertions,
            result.work.deferred_pnp_attempts,
            result.work.deferred_pnp_estimation_failures,
            result.work.deferred_pnp_inlier_rejections,
            result.work.deferred_registrations,
            result.work.deferred_interpolation_registrations,
            result.work.deferred_observations_attached,
            result.work.unregistered_zero_support_frames,
            result.work.unregistered_below_pnp_support_frames,
            result.work.unregistered_eligible_pnp_frames,
            result.work.unregistered_below_sensor_frames,
            result.work.max_unregistered_support,
            result.work.paired_pose_jump_repairs,
            result.work.paired_pose_jump_repaired_frames,
        ));
        eprintln!(
            "rig-component: rank={rank} supplied_frames={supplied_frames} pairs={verified_pairs} registered_frames={} registered_images={} tracks={tracks} observations={observations} mean_reprojection_px={:.9} seed_global_frame={seed_global_frame} triangulation_attempts={} robust_triangulation_tracks={} robust_triangulation_pruned_observations={} robust_triangulation_majority_rejections={} pnp_attempts={} pnp_insufficient_sensors={} pnp_estimation_failures={} pnp_inlier_rejections={} pnp_registrations={} dynamic_activated_rows={} dynamic_activated_edges={} dynamic_track_creates={} dynamic_track_continues={} dynamic_owner_conflicts={} dynamic_same_image_conflicts={} dynamic_geometry_rejections={} dynamic_observation_lookup_entries={} dynamic_pnp_graph_insertions={} dynamic_bootstrap_legacy_tracks={} dynamic_bootstrap_candidates={} dynamic_bootstrap_seed_support={} dynamic_bootstrap_seed_pairs={} dynamic_bootstrap_seed_landmarks={} dynamic_bootstrap_direct_fallbacks={} direct_bridge_pair_visits={} direct_bridge_correspondences={} direct_bridge_registrations={} motion_bridge_pair_visits={} motion_bridge_estimation_failures={} motion_bridge_rotation_rejections={} motion_bridge_registrations={} deferred_pair_visits={} deferred_correspondences={} deferred_pnp_attempts={} deferred_pnp_estimation_failures={} deferred_pnp_inlier_rejections={} deferred_registrations={} deferred_interpolation_registrations={} deferred_observations_attached={} deferred_retriangulated_tracks={} deferred_retriangulated_observations={} unregistered_zero_support={} unregistered_below_pnp_support={} unregistered_eligible_pnp={} unregistered_below_sensors={} max_unregistered_support={} geometry_recovered_tracks={} geometry_recovered_observations={} track_completion_passes={} track_completion_pair_visits={} track_completion_observations={} track_completion_reprojection_rejections={} isolated_pose_repair_passes={} isolated_pose_repairs={} paired_pose_jump_repairs={} paired_pose_jump_repaired_frames={} VmHWM={} KiB",
            result.registered_frames,
            result.registered_images,
            result.mean_reprojection_error_px,
            result.work.triangulation_attempts,
            result.work.robust_triangulation_tracks,
            result.work.robust_triangulation_pruned_observations,
            result.work.robust_triangulation_majority_rejections,
            result.work.pnp_attempts,
            result.work.pnp_insufficient_sensor_attempts,
            result.work.pnp_estimation_failures,
            result.work.pnp_inlier_rejections,
            result.work.pnp_registrations,
            result.work.dynamic_activated_rows,
            result.work.dynamic_activated_edges,
            result.work.dynamic_track_creates,
            result.work.dynamic_track_continues,
            result.work.dynamic_owner_conflicts,
            result.work.dynamic_same_image_conflicts,
            result.work.dynamic_geometry_rejections,
            result.work.dynamic_observation_lookup_entries,
            result.work.dynamic_pnp_graph_insertions,
            result.work.dynamic_bootstrap_legacy_tracks,
            result.work.dynamic_bootstrap_candidates,
            result.work.dynamic_bootstrap_seed_support,
            result.work.dynamic_bootstrap_seed_pairs,
            result.work.dynamic_bootstrap_seed_landmarks,
            result.work.dynamic_bootstrap_direct_fallbacks,
            result.work.direct_bridge_pair_visits,
            result.work.direct_bridge_correspondence_insertions,
            result.work.direct_bridge_registrations,
            result.work.motion_bridge_pair_visits,
            result.work.motion_bridge_estimation_failures,
            result.work.motion_bridge_rotation_rejections,
            result.work.motion_bridge_registrations,
            result.work.deferred_pair_visits,
            result.work.deferred_correspondence_insertions,
            result.work.deferred_pnp_attempts,
            result.work.deferred_pnp_estimation_failures,
            result.work.deferred_pnp_inlier_rejections,
            result.work.deferred_registrations,
            result.work.deferred_interpolation_registrations,
            result.work.deferred_observations_attached,
            result.work.deferred_retriangulated_tracks,
            result.work.deferred_retriangulated_observations,
            result.work.unregistered_zero_support_frames,
            result.work.unregistered_below_pnp_support_frames,
            result.work.unregistered_eligible_pnp_frames,
            result.work.unregistered_below_sensor_frames,
            result.work.max_unregistered_support,
            result.work.geometry_recovered_tracks,
            result.work.geometry_recovered_observations,
            result.work.track_completion_passes,
            result.work.track_completion_pair_visits,
            result.work.track_completion_observations,
            result.work.track_completion_reprojection_rejections,
            result.work.isolated_pose_repair_passes,
            result.work.isolated_pose_repairs,
            result.work.paired_pose_jump_repairs,
            result.work.paired_pose_jump_repaired_frames,
            peak_rss_kib().unwrap_or(0),
        );
    }
    let mean_reprojection = weighted_reprojection / total_observations.max(1) as f64;
    manifest.push_str(&format!(
        "# independent_gauges=true models={emitted_models} registered_frames={total_registered_frames} registered_images={total_registered_images} tracks={total_tracks} observations={total_observations} weighted_mean_reprojection_px={mean_reprojection:.9}\n",
    ));
    std::fs::create_dir_all(&args.out_colmap)?;
    std::fs::write(args.out_colmap.join("components.tsv"), manifest)?;
    std::fs::write(
        args.out_colmap.join("retrieval-components.txt"),
        retrieval_components,
    )?;
    println!(
        "rig-multimodel result: models={emitted_models} registered_frames={total_registered_frames}/{} registered_images={total_registered_images}/{} tracks={total_tracks} observations={total_observations} mean_reprojection_px={mean_reprojection:.9} mapper_seconds={:.6} VmHWM_KiB={} out={}",
        frames.len(),
        features.len(),
        mapping_started.elapsed().as_secs_f64(),
        peak_rss_kib().unwrap_or(0),
        args.out_colmap.display(),
    );
    Ok(())
}

#[derive(Debug, Clone)]
struct ColmapRotationPrior {
    model_index: usize,
    world_to_camera_rotation: UnitQuaternion<f64>,
}

fn read_colmap_rotation_priors(
    paths: &[PathBuf],
    image_names: &[String],
) -> Result<HashMap<String, ColmapRotationPrior>, String> {
    let known_names = image_names
        .iter()
        .map(String::as_str)
        .collect::<std::collections::HashSet<_>>();
    let mut priors = HashMap::new();
    for (model_index, path) in paths.iter().enumerate() {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("read COLMAP pose prior {}: {error}", path.display()))?;
        let mut matched = 0usize;
        for (line_index, line) in text.lines().enumerate() {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            // COLMAP's second line for every image starts with a floating
            // point x coordinate, while image rows start with an integer id.
            if fields.len() < 10 || fields[0].parse::<u64>().is_err() {
                continue;
            }
            let image_name = fields[9..].join(" ");
            if !known_names.contains(image_name.as_str()) {
                continue;
            }
            let values = (1..8)
                .map(|index| {
                    fields[index].parse::<f64>().map_err(|error| {
                        format!(
                            "{}:{} invalid COLMAP pose value {}: {error}",
                            path.display(),
                            line_index + 1,
                            fields[index]
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "{}:{} COLMAP pose contains a non-finite value",
                    path.display(),
                    line_index + 1
                ));
            }
            let quaternion = Quaternion::new(values[0], values[1], values[2], values[3]);
            let norm = quaternion.norm();
            if norm <= 1.0e-12 || !norm.is_finite() {
                return Err(format!(
                    "{}:{} COLMAP pose has a zero-length quaternion",
                    path.display(),
                    line_index + 1
                ));
            }
            let prior = ColmapRotationPrior {
                model_index,
                world_to_camera_rotation: UnitQuaternion::new_normalize(quaternion),
            };
            if priors.insert(image_name.clone(), prior).is_some() {
                return Err(format!(
                    "COLMAP pose priors contain duplicate image {image_name:?}"
                ));
            }
            matched += 1;
        }
        if matched == 0 {
            return Err(format!(
                "COLMAP pose prior {} contains no images from the snapshot",
                path.display()
            ));
        }
    }
    if priors.is_empty() {
        return Err("COLMAP pose prior list is empty".into());
    }
    Ok(priors)
}

fn rotation_from_snapshot_bits(bits: [u64; 9]) -> Option<UnitQuaternion<f64>> {
    let matrix = Matrix3::from_column_slice(&bits.map(f64::from_bits));
    if !matrix.iter().all(|value| value.is_finite())
        || (matrix.transpose() * matrix - Matrix3::identity()).norm() > 1.0e-3
        || (matrix.determinant() - 1.0).abs() > 1.0e-3
    {
        return None;
    }
    Some(UnitQuaternion::from_matrix(&matrix))
}

fn pair_rotation_disagreement_deg(
    pair: &verified_pair_snapshot::PairRecord,
    image_names: &[String],
    priors: &HashMap<String, ColmapRotationPrior>,
) -> Option<f64> {
    let image_i = usize::try_from(pair.image_i).ok()?;
    let image_j = usize::try_from(pair.image_j).ok()?;
    let name_i = image_names.get(image_i)?;
    let name_j = image_names.get(image_j)?;
    let prior_i = priors.get(name_i)?;
    let prior_j = priors.get(name_j)?;
    if prior_i.model_index != prior_j.model_index {
        return None;
    }
    let observed = rotation_from_snapshot_bits(pair.relative_rotation_bits?);
    let predicted = prior_j.world_to_camera_rotation * prior_i.world_to_camera_rotation.inverse();
    Some(observed?.angle_to(&predicted).to_degrees())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WeakLongPartitionStats {
    base_pairs: usize,
    base_matches: usize,
    deferred_pairs: usize,
    deferred_matches: usize,
    weak_support_pairs: usize,
    weak_support_matches: usize,
    pose_checked_pairs: usize,
    pose_rejected_pairs: usize,
}

fn should_defer_weak_long_pair(
    frame_i: usize,
    frame_j: usize,
    e_inliers: u64,
    min_frame_gap: usize,
    min_e_inliers: u64,
) -> bool {
    frame_i.abs_diff(frame_j) >= min_frame_gap && e_inliers < min_e_inliers
}

fn partition_weak_long_pairs(
    snapshot: &mut verified_pair_snapshot::Snapshot,
    frames: &[RigFrame],
    min_pair_matches: Option<usize>,
    long_pair_gate: Option<(usize, u64)>,
    pose_priors: Option<&HashMap<String, ColmapRotationPrior>>,
    max_rotation_disagreement_deg: Option<f64>,
) -> Result<WeakLongPartitionStats, String> {
    let mut image_frames = vec![None; snapshot.image_names.len()];
    for (frame, rig_frame) in frames.iter().enumerate() {
        for image in &rig_frame.images {
            let slot = image_frames
                .get_mut(image.image_index)
                .ok_or_else(|| format!("rig image {} is outside snapshot", image.image_index))?;
            if slot.replace(frame).is_some() {
                return Err(format!(
                    "snapshot image {} belongs to two frames",
                    image.image_index
                ));
            }
        }
    }
    if image_frames.iter().any(Option::is_none) {
        return Err("not every snapshot image belongs to a rig frame".into());
    }
    let image_frames = image_frames
        .into_iter()
        .map(Option::unwrap)
        .collect::<Vec<_>>();
    let mut base = Vec::with_capacity(snapshot.pairs.len());
    let mut deferred = Vec::new();
    let mut base_matches = 0usize;
    let mut deferred_matches = 0usize;
    let mut weak_support_pairs = 0usize;
    let mut weak_support_matches = 0usize;
    let mut pose_checked_pairs = 0usize;
    let mut pose_rejected_pairs = 0usize;
    for pair in std::mem::take(&mut snapshot.pairs) {
        let image_i = usize::try_from(pair.image_i)
            .map_err(|_| "pair image_i does not fit usize".to_owned())?;
        let image_j = usize::try_from(pair.image_j)
            .map_err(|_| "pair image_j does not fit usize".to_owned())?;
        let frame_i = *image_frames
            .get(image_i)
            .ok_or_else(|| format!("pair image_i {image_i} is outside snapshot"))?;
        let frame_j = *image_frames
            .get(image_j)
            .ok_or_else(|| format!("pair image_j {image_j} is outside snapshot"))?;
        let weak_support = min_pair_matches.is_some_and(|minimum| pair.matches.len() < minimum);
        if weak_support {
            weak_support_pairs += 1;
            weak_support_matches += pair.matches.len();
        }
        let weak_long = long_pair_gate.is_some_and(|(min_frame_gap, min_e_inliers)| {
            should_defer_weak_long_pair(
                frame_i,
                frame_j,
                pair.e_inlier_count,
                min_frame_gap,
                min_e_inliers,
            )
        });
        let pose_rejected = if !weak_support
            && !weak_long
            && long_pair_gate
                .is_some_and(|(min_frame_gap, _)| frame_i.abs_diff(frame_j) >= min_frame_gap)
        {
            match (pose_priors, max_rotation_disagreement_deg) {
                (Some(priors), Some(limit)) => {
                    pose_checked_pairs += 1;
                    let rejected =
                        pair_rotation_disagreement_deg(&pair, &snapshot.image_names, priors)
                            .is_none_or(|disagreement| disagreement > limit);
                    if rejected {
                        pose_rejected_pairs += 1;
                    }
                    rejected
                }
                _ => false,
            }
        } else {
            false
        };
        if weak_support || weak_long || pose_rejected {
            deferred_matches += pair.matches.len();
            deferred.push(pair);
        } else {
            base_matches += pair.matches.len();
            base.push(pair);
        }
    }
    let stats = WeakLongPartitionStats {
        base_pairs: base.len(),
        base_matches,
        deferred_pairs: deferred.len(),
        deferred_matches,
        weak_support_pairs,
        weak_support_matches,
        pose_checked_pairs,
        pose_rejected_pairs,
    };
    base.extend(deferred);
    snapshot.pairs = base;
    Ok(stats)
}

/// One accepted snapshot rotation after conversion to the rig-frame edge
/// convention.  `from_frame < to_frame` is always true; this makes both
/// component discovery and duplicate selection independent of input pair
/// direction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FrameRotationConstraint {
    from_frame: usize,
    to_frame: usize,
    rotation: UnitQuaternion<f64>,
    capped_e_inliers: u64,
    source_image_i: usize,
    source_image_j: usize,
}

impl FrameRotationConstraint {
    fn source_tie_key(self) -> (usize, usize, usize, usize) {
        (
            self.source_image_i.min(self.source_image_j),
            self.source_image_i.max(self.source_image_j),
            self.source_image_i,
            self.source_image_j,
        )
    }

    fn weight(self) -> f64 {
        self.capped_e_inliers as f64
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RigRotationAverageStats {
    /// Number of records that survived essential-support, valid-rotation,
    /// registration, and current-pose disagreement gates, before frame-pair
    /// deduplication.
    candidate_pairs: usize,
    /// Number of candidates before deduplication (kept for concise logging).
    accepted_pairs: usize,
    /// Number of unique unordered frame pairs after deterministic selection.
    deduped_pairs: usize,
    registered_components: usize,
    averaged_components: usize,
    failed_components: usize,
    /// Components whose chordal solution would move at least one node farther
    /// than the configured per-component safety gate.
    unsafe_update_components: usize,
    averaged_frames: usize,
    disagreement_rejections: usize,
    /// Largest node update proposed by any successfully solved component,
    /// including components rejected by `max_update_deg`.
    max_proposed_update_deg: f64,
    max_rotation_update_deg: f64,
    /// Set when the fixed-rotation refinement was rolled back because map
    /// support or registered-frame coverage regressed.
    support_rollback: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct RigTranslationAverageStats {
    candidate_pairs: usize,
    accepted_pairs: usize,
    degenerate_pairs: usize,
    direction_rejections: usize,
    deduped_edges: usize,
    backbone_edges: usize,
    solved_frames: usize,
    max_proposed_update_m: f64,
    candidate_tracks: usize,
    candidate_observations: usize,
    candidate_mean_reprojection_px: f64,
    support_regressed: bool,
    reprojection_regressed: bool,
    support_rollback: bool,
    unsafe_update: bool,
}

#[derive(Debug, Clone)]
struct RigTranslationEdge {
    edge: GlobalSfmEdge,
    nullspace_ratio: f64,
}

fn fixed_rotation_translation_direction(
    camera_i: &Camera,
    camera_j: &Camera,
    rotation_i_to_j: &UnitQuaternion<f64>,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    matches: &[(u64, u64)],
) -> Option<(Vector3<f64>, f64)> {
    if matches.len() < 8 {
        return None;
    }
    let rotation = rotation_i_to_j.to_rotation_matrix();
    let rotation = rotation.matrix();
    let stride = (matches.len() / 64).max(1);
    let mut normal = Matrix3::zeros();
    let mut used = 0usize;
    for &(keypoint_i, keypoint_j) in matches.iter().step_by(stride) {
        let (Ok(keypoint_i), Ok(keypoint_j)) =
            (usize::try_from(keypoint_i), usize::try_from(keypoint_j))
        else {
            continue;
        };
        let (Some(pixel_i), Some(pixel_j)) = (
            features_i.keypoints.get(keypoint_i),
            features_j.keypoints.get(keypoint_j),
        ) else {
            continue;
        };
        let (Some(point_i), Some(point_j)) = (
            camera_i.normalize_pixel(pixel_i),
            camera_j.normalize_pixel(pixel_j),
        ) else {
            continue;
        };
        let bearing_i = Vector3::new(point_i.x, point_i.y, 1.0);
        let bearing_j = Vector3::new(point_j.x, point_j.y, 1.0);
        let row = bearing_j.cross(&(rotation * bearing_i));
        if !row.iter().all(|value| value.is_finite()) {
            continue;
        }
        normal += row * row.transpose();
        used += 1;
    }
    if used < 8 {
        return None;
    }
    let eigen = normal.symmetric_eigen();
    let mut order = [0usize, 1, 2];
    order.sort_unstable_by(|left, right| {
        eigen.eigenvalues[*left].total_cmp(&eigen.eigenvalues[*right])
    });
    let smallest = eigen.eigenvalues[order[0]];
    let second = eigen.eigenvalues[order[1]];
    if !smallest.is_finite() || !second.is_finite() || second <= 1.0e-12 {
        return None;
    }
    let nullspace_ratio = smallest.max(0.0) / second;
    // A noisy planar/forward-motion pair has no isolated epipolar nullspace.
    // Keep this strict because a wrong direction can bend an entire component.
    if !nullspace_ratio.is_finite() || nullspace_ratio > 0.1 {
        return None;
    }
    let translation = eigen.eigenvectors.column(order[0]).into_owned();
    let translation = translation.try_normalize(1.0e-12)?;
    let centre_direction = -(rotation.transpose() * translation).try_normalize(1.0e-12)?;
    Some((centre_direction, nullspace_ratio))
}

#[allow(clippy::too_many_arguments)]
fn apply_final_rig_translation_average(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    snapshot_pairs: &[verified_pair_snapshot::PairRecord],
    min_frame_gap: usize,
    min_matches: usize,
    max_direction_error_deg: f64,
    max_update_m: f64,
    iterations: usize,
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<RigTranslationAverageStats, Box<dyn Error>> {
    let before = result.clone();
    let mut stats = RigTranslationAverageStats::default();
    let mut assignment = vec![None; features.len()];
    for (frame, rig_frame) in frames.iter().enumerate() {
        for image in &rig_frame.images {
            if let Some(slot) = assignment.get_mut(image.image_index) {
                *slot = Some((frame, image.sensor_index));
            }
        }
    }
    let reference_sensor = rig
        .sensors()
        .iter()
        .enumerate()
        .min_by(|left, right| {
            left.1
                .sensor_from_rig
                .translation
                .norm_squared()
                .total_cmp(&right.1.sensor_from_rig.translation.norm_squared())
        })
        .map(|(sensor, _)| sensor)
        .ok_or("rig has no reference sensor")?;
    let reference = &rig.sensors()[reference_sensor];
    if reference.sensor_from_rig.translation.norm() > 1.0e-6 {
        return Err("translation averaging requires one sensor at the rig origin".into());
    }

    let mut deduped = BTreeMap::<(usize, usize), RigTranslationEdge>::new();
    for pair in snapshot_pairs {
        let (Ok(image_i), Ok(image_j)) =
            (usize::try_from(pair.image_i), usize::try_from(pair.image_j))
        else {
            continue;
        };
        let (Some((frame_i, sensor_i)), Some((frame_j, sensor_j))) = (
            assignment.get(image_i).copied().flatten(),
            assignment.get(image_j).copied().flatten(),
        ) else {
            continue;
        };
        if frame_i == frame_j
            || sensor_i != reference_sensor
            || sensor_j != reference_sensor
            || frame_i.abs_diff(frame_j) < min_frame_gap
            || pair.matches.len() < min_matches
            || pair.e_inlier_count < min_matches as u64
        {
            continue;
        }
        let (Some(pose_i), Some(pose_j)) = (
            result.frame_poses.get(frame_i).and_then(Option::as_ref),
            result.frame_poses.get(frame_j).and_then(Option::as_ref),
        ) else {
            continue;
        };
        stats.candidate_pairs += 1;
        let rotation_i_to_j =
            pose_j.world_to_camera.rotation * pose_i.world_to_camera.rotation.inverse();
        let Some((sensor_direction, nullspace_ratio)) = fixed_rotation_translation_direction(
            &reference.camera,
            &reference.camera,
            &rotation_i_to_j,
            &features[image_i],
            &features[image_j],
            &pair.matches,
        ) else {
            stats.degenerate_pairs += 1;
            continue;
        };
        let mut rig_direction = reference
            .sensor_from_rig
            .rotation
            .inverse_transform_vector(&sensor_direction);
        let current_world = pose_j.camera_center_world() - pose_i.camera_center_world();
        let Some(current_rig_i) = pose_i
            .world_to_camera
            .rotation
            .transform_vector(&current_world)
            .try_normalize(1.0e-12)
        else {
            stats.degenerate_pairs += 1;
            continue;
        };
        if rig_direction.dot(&current_rig_i) < 0.0 {
            rig_direction = -rig_direction;
        }
        let direction_error_deg = rig_direction
            .dot(&current_rig_i)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if !direction_error_deg.is_finite() || direction_error_deg > max_direction_error_deg {
            stats.direction_rejections += 1;
            continue;
        }
        let (from_frame, to_frame, direction, rotation) = if frame_i < frame_j {
            (frame_i, frame_j, rig_direction, rotation_i_to_j)
        } else {
            (
                frame_j,
                frame_i,
                -rotation_i_to_j.transform_vector(&rig_direction),
                rotation_i_to_j.inverse(),
            )
        };
        let candidate = RigTranslationEdge {
            edge: GlobalSfmEdge {
                image_i: from_frame,
                image_j: to_frame,
                rotation_ij: rotation,
                direction_ij: direction,
                weight: pair.e_inlier_count.min(128) as f64,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
            nullspace_ratio,
        };
        let key = (from_frame, to_frame);
        let replace = deduped.get(&key).is_none_or(|previous| {
            candidate.edge.weight > previous.edge.weight
                || (candidate.edge.weight == previous.edge.weight
                    && candidate.nullspace_ratio < previous.nullspace_ratio)
        });
        if replace {
            deduped.insert(key, candidate);
        }
        stats.accepted_pairs += 1;
    }
    let mut edges = deduped
        .into_values()
        .map(|candidate| candidate.edge)
        .collect::<Vec<_>>();
    stats.deduped_edges = edges.len();
    // High-confidence epipolar edges need not touch every registered frame.
    // Add a unit-weight direction backbone between consecutive registered
    // poses. Independent per-edge scales keep current step magnitudes free,
    // while verified edges retain much larger support weights.
    let registered = result
        .frame_poses
        .iter()
        .enumerate()
        .filter_map(|(frame, pose)| pose.as_ref().map(|_| frame))
        .collect::<Vec<_>>();
    let existing = edges
        .iter()
        .map(|edge| {
            (
                edge.image_i.min(edge.image_j),
                edge.image_i.max(edge.image_j),
            )
        })
        .collect::<HashSet<_>>();
    for frames in registered.windows(2) {
        let from = frames[0];
        let to = frames[1];
        if existing.contains(&(from, to)) {
            continue;
        }
        let from_pose = result.frame_poses[from]
            .as_ref()
            .expect("registered backbone frame");
        let to_pose = result.frame_poses[to]
            .as_ref()
            .expect("registered backbone frame");
        let world_delta = to_pose.camera_center_world() - from_pose.camera_center_world();
        let Some(direction) = from_pose
            .world_to_camera
            .rotation
            .transform_vector(&world_delta)
            .try_normalize(1.0e-12)
        else {
            continue;
        };
        edges.push(GlobalSfmEdge {
            image_i: from,
            image_j: to,
            rotation_ij: to_pose.world_to_camera.rotation
                * from_pose.world_to_camera.rotation.inverse(),
            direction_ij: direction,
            weight: 1.0,
            inlier_sample: Vec::new(),
            rotation_alt: None,
            direction_alt: None,
        });
        stats.backbone_edges += 1;
    }
    let rotations = result
        .frame_poses
        .iter()
        .map(|pose| pose.as_ref().map(|pose| pose.world_to_camera.rotation))
        .collect::<Vec<_>>();
    let weights = edges.iter().map(|edge| edge.weight).collect::<Vec<_>>();
    let Some(proposed) = average_positions_with_independent_edge_scales(
        &edges,
        &weights,
        &rotations,
        result.seed_frame_index,
        iterations,
    ) else {
        return Ok(stats);
    };
    let mut source = Vec::new();
    let mut destination = Vec::new();
    for (frame, centre) in proposed.iter().enumerate() {
        let (Some(centre), Some(current)) = (
            centre.as_ref(),
            result.frame_poses.get(frame).and_then(Option::as_ref),
        ) else {
            continue;
        };
        source.push(*centre);
        destination.push(current.camera_center_world());
    }
    stats.solved_frames = source.len();
    if source.len() * 10 < result.registered_frames * 9 {
        return Ok(stats);
    }
    let Some(alignment) = umeyama_similarity_transform(&source, &destination, true) else {
        return Ok(stats);
    };
    let transform = |point: &Point3<f64>| {
        Point3::from(alignment.scale * (alignment.rotation * point.coords) + alignment.translation)
    };
    for (frame, proposed) in proposed.iter().enumerate() {
        let (Some(proposed), Some(current)) = (
            proposed.as_ref(),
            result.frame_poses.get(frame).and_then(Option::as_ref),
        ) else {
            continue;
        };
        stats.max_proposed_update_m = stats
            .max_proposed_update_m
            .max((transform(proposed) - current.camera_center_world()).norm());
    }
    if !stats.max_proposed_update_m.is_finite() || stats.max_proposed_update_m > max_update_m {
        stats.unsafe_update = true;
        return Ok(stats);
    }
    for (frame, proposed) in proposed.iter().enumerate() {
        let Some(current) = result.frame_poses.get_mut(frame).and_then(Option::as_mut) else {
            continue;
        };
        let Some(proposed) = proposed.as_ref() else {
            continue;
        };
        let centre = transform(proposed);
        current.world_to_camera.translation = -current
            .world_to_camera
            .rotation
            .transform_vector(&centre.coords);
        for image in &frames[frame].images {
            let sensor = &rig.sensors()[image.sensor_index];
            result.image_poses[image.image_index] = Some(Pose {
                world_to_camera: sensor.sensor_from_rig.compose(&current.world_to_camera),
            });
        }
    }
    let fixed_rotations = result
        .frame_poses
        .iter()
        .map(|pose| pose.as_ref().map(|pose| pose.world_to_camera.rotation))
        .collect::<Vec<_>>();
    let refinement_failed = visloc_rs::refine_rig_sfm_with_fixed_frame_rotations(
        rig,
        frames,
        features,
        &fixed_rotations,
        config,
        result,
    )
    .is_err();
    stats.candidate_tracks = result.tracks.len();
    stats.candidate_observations = result
        .tracks
        .iter()
        .map(|track| track.observations.len())
        .sum();
    stats.candidate_mean_reprojection_px = result.mean_reprojection_error_px;
    stats.support_regressed = rotation_average_support_regressed(&before, result);
    stats.reprojection_regressed =
        result.mean_reprojection_error_px > before.mean_reprojection_error_px + 1.0e-9;
    if refinement_failed || stats.support_regressed || stats.reprojection_regressed {
        *result = before;
        stats.support_rollback = true;
    }
    Ok(stats)
}

/// Keep a rotation-average refinement only when it retains at least this
/// fraction of both tracks and observations.  This is deliberately a strict,
/// GT-free support guard: a wrong orientation can make the fixed-pose BA
/// discard most tracks while still returning a finite result.
const FINAL_RIG_ROTATION_AVERAGE_MIN_SUPPORT_FRACTION: f64 = 0.90;

fn result_observation_count(result: &RigSfmResult) -> usize {
    result
        .tracks
        .iter()
        .map(|track| track.observations.len())
        .sum()
}

fn rotation_average_support_regressed(before: &RigSfmResult, after: &RigSfmResult) -> bool {
    let minimum = FINAL_RIG_ROTATION_AVERAGE_MIN_SUPPORT_FRACTION;
    (after.tracks.len() as f64) < minimum * before.tracks.len() as f64
        || (result_observation_count(after) as f64)
            < minimum * result_observation_count(before) as f64
        || after.registered_frames != before.registered_frames
        || after.registered_images != before.registered_images
}

/// Convert the snapshot's image-indexed metadata to frame-level rig rotation
/// constraints.  Snapshot matrices are serialized in nalgebra's column-major
/// order.  A pair's sensor-to-sensor rotation is converted with
/// `R_sj_rig⁻¹ · R_sj_si · R_si_rig`, then inverted when the source record's
/// frame order is opposite the canonical `(min_frame, max_frame)` order.
fn build_frame_rotation_constraints(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    snapshot_pairs: &[verified_pair_snapshot::PairRecord],
    frame_poses: &[Option<Pose>],
    max_disagreement_deg: f64,
    weight_cap: usize,
) -> (Vec<FrameRotationConstraint>, RigRotationAverageStats) {
    let image_count = frames
        .iter()
        .flat_map(|frame| frame.images.iter().map(|image| image.image_index))
        .max()
        .map_or(0, |image| image.saturating_add(1));
    let mut image_assignment = vec![None; image_count];
    for (frame, rig_frame) in frames.iter().enumerate() {
        for image in &rig_frame.images {
            let Some(slot) = image_assignment.get_mut(image.image_index) else {
                continue;
            };
            if slot.is_none() {
                *slot = Some((frame, image.sensor_index));
            }
        }
    }
    let cap = u64::try_from(weight_cap).unwrap_or(u64::MAX).max(1);
    let mut stats = RigRotationAverageStats::default();
    let mut deduped = BTreeMap::<(usize, usize), FrameRotationConstraint>::new();
    for pair in snapshot_pairs {
        // The snapshot's `calibrated` bit describes the verifier's camera
        // configuration, not whether an essential relative rotation was
        // retained.  Some replay snapshots carry valid E rotations in records
        // whose configuration bit is uncalibrated; use the actual E evidence
        // instead of dropping those constraints.
        if pair.e_inlier_count == 0 {
            continue;
        }
        let (Some(image_i), Some(image_j)) = (
            usize::try_from(pair.image_i).ok(),
            usize::try_from(pair.image_j).ok(),
        ) else {
            continue;
        };
        let (Some((frame_i, sensor_i)), Some((frame_j, sensor_j))) = (
            image_assignment.get(image_i).copied().flatten(),
            image_assignment.get(image_j).copied().flatten(),
        ) else {
            continue;
        };
        if frame_i == frame_j {
            continue;
        }
        let (Some(sensor_i), Some(sensor_j)) =
            (rig.sensors().get(sensor_i), rig.sensors().get(sensor_j))
        else {
            continue;
        };
        let Some(sensor_rotation) = pair
            .relative_rotation_bits
            .and_then(rotation_from_snapshot_bits)
        else {
            continue;
        };
        let rig_rotation = sensor_j.sensor_from_rig.rotation.inverse()
            * sensor_rotation
            * sensor_i.sensor_from_rig.rotation;
        let (from_frame, to_frame, rotation) = if frame_i < frame_j {
            (frame_i, frame_j, rig_rotation)
        } else {
            (frame_j, frame_i, rig_rotation.inverse())
        };
        let (Some(from_pose), Some(to_pose)) = (
            frame_poses.get(from_frame).and_then(Option::as_ref),
            frame_poses.get(to_frame).and_then(Option::as_ref),
        ) else {
            continue;
        };
        let current =
            to_pose.world_to_camera.rotation * from_pose.world_to_camera.rotation.inverse();
        let disagreement_deg = rotation.angle_to(&current).to_degrees();
        if !disagreement_deg.is_finite() || disagreement_deg > max_disagreement_deg {
            stats.disagreement_rejections += 1;
            continue;
        }
        let candidate = FrameRotationConstraint {
            from_frame,
            to_frame,
            rotation,
            capped_e_inliers: pair.e_inlier_count.min(cap),
            source_image_i: image_i,
            source_image_j: image_j,
        };
        stats.candidate_pairs += 1;
        stats.accepted_pairs += 1;
        let key = (from_frame, to_frame);
        let replace = deduped.get(&key).is_none_or(|previous| {
            candidate.capped_e_inliers > previous.capped_e_inliers
                || (candidate.capped_e_inliers == previous.capped_e_inliers
                    && candidate.source_tie_key() < previous.source_tie_key())
        });
        if replace {
            deduped.insert(key, candidate);
        }
    }
    stats.deduped_pairs = deduped.len();
    (deduped.into_values().collect(), stats)
}

/// Solve each connected registered frame component with the sparse chordal
/// rotation initializer.  The graph receives the current frame poses, so its
/// initializer keeps every camera centre unchanged while replacing only the
/// orientation.  Unconstrained and unregistered frames retain their current
/// rotation (or `None`).
fn desired_rotations_from_constraints(
    result: &RigSfmResult,
    constraints: &[FrameRotationConstraint],
    max_update_deg: f64,
    mut stats: RigRotationAverageStats,
) -> (Vec<Option<UnitQuaternion<f64>>>, RigRotationAverageStats) {
    let mut desired = result
        .frame_poses
        .iter()
        .map(|pose| pose.as_ref().map(|pose| pose.world_to_camera.rotation))
        .collect::<Vec<_>>();
    let mut adjacency = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for (constraint_index, constraint) in constraints.iter().enumerate() {
        adjacency
            .entry(constraint.from_frame)
            .or_default()
            .push((constraint.to_frame, constraint_index));
        adjacency
            .entry(constraint.to_frame)
            .or_default()
            .push((constraint.from_frame, constraint_index));
    }
    let mut visited = HashSet::new();
    for &start in adjacency.keys() {
        if !visited.insert(start) {
            continue;
        }
        let mut component = vec![start];
        let mut pending = vec![start];
        let mut component_edges = HashSet::new();
        while let Some(frame) = pending.pop() {
            for &(neighbor, edge_index) in adjacency.get(&frame).into_iter().flatten() {
                component_edges.insert(edge_index);
                if visited.insert(neighbor) {
                    component.push(neighbor);
                    pending.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        stats.registered_components += 1;
        if component.len() < 2 {
            continue;
        }
        let mut graph = PoseGraph::new();
        let mut all_registered = true;
        for &frame in &component {
            let Some(pose) = result.frame_poses.get(frame).and_then(Option::as_ref) else {
                all_registered = false;
                break;
            };
            graph.add_pose(frame as u64, pose.clone());
        }
        if !all_registered {
            stats.failed_components += 1;
            continue;
        }
        graph.anchor(start as u64);
        let mut component_edges = component_edges.into_iter().collect::<Vec<_>>();
        component_edges.sort_unstable();
        for constraint_index in component_edges {
            let constraint = &constraints[constraint_index];
            // Keep this construction local until the public PoseGraph helper
            // lands in the shared API.  A zero-translation edge contributes
            // exactly its rotation and scalar weight to chordal initialization;
            // the fixed-rotation rig refinement below never consumes its
            // translation.
            graph.edges.push(PoseGraphEdge {
                from: constraint.from_frame as u64,
                to: constraint.to_frame as u64,
                measurement: SE3::new(constraint.rotation, Vector3::zeros()),
                kind: PoseGraphEdgeKind::LoopClosure,
                weight: constraint.weight(),
                information: None,
            });
        }
        let Ok(_) = graph.initialize_rotations_chordal(LinearSolver::Sparse) else {
            stats.failed_components += 1;
            continue;
        };
        let proposed_update_deg = component
            .iter()
            .filter_map(|&frame| {
                let before = result.frame_poses.get(frame)?.as_ref()?;
                let after = graph.poses.get(&(frame as u64))?;
                Some(
                    before
                        .world_to_camera
                        .rotation
                        .angle_to(&after.world_to_camera.rotation)
                        .to_degrees(),
                )
            })
            .fold(0.0, f64::max);
        stats.max_proposed_update_deg = stats.max_proposed_update_deg.max(proposed_update_deg);
        if !proposed_update_deg.is_finite() || proposed_update_deg > max_update_deg {
            stats.unsafe_update_components += 1;
            continue;
        }
        stats.averaged_components += 1;
        for &frame in &component {
            let Some(averaged) = graph.poses.get(&(frame as u64)) else {
                continue;
            };
            let Some(before) = result.frame_poses[frame].as_ref() else {
                continue;
            };
            stats.max_rotation_update_deg = stats.max_rotation_update_deg.max(
                before
                    .world_to_camera
                    .rotation
                    .angle_to(&averaged.world_to_camera.rotation)
                    .to_degrees(),
            );
            desired[frame] = Some(averaged.world_to_camera.rotation);
            stats.averaged_frames += 1;
        }
    }
    (desired, stats)
}

/// Apply the opt-in rotation average and then let the rig mapper's fixed-frame
/// rotation refinement update translations/landmarks while preserving those
/// averaged orientations.
#[allow(clippy::too_many_arguments)]
fn apply_final_rig_rotation_average(
    rig: &GeneralizedCameraRig,
    frames: &[RigFrame],
    features: &[FeatureSet],
    snapshot_pairs: &[verified_pair_snapshot::PairRecord],
    max_disagreement_deg: f64,
    max_update_deg: f64,
    weight_cap: usize,
    config: &RigSfmConfig,
    result: &mut RigSfmResult,
) -> Result<RigRotationAverageStats, Box<dyn Error>> {
    let (constraints, stats) = build_frame_rotation_constraints(
        rig,
        frames,
        snapshot_pairs,
        &result.frame_poses,
        max_disagreement_deg,
        weight_cap,
    );
    let (desired_rotations, mut stats) =
        desired_rotations_from_constraints(result, &constraints, max_update_deg, stats);
    if stats.averaged_components > 0 {
        let result_before = result.clone();
        if let Err(error) = visloc_rs::refine_rig_sfm_with_fixed_frame_rotations(
            rig,
            frames,
            features,
            &desired_rotations,
            config,
            result,
        ) {
            *result = result_before;
            return Err(error.into());
        }
        if rotation_average_support_regressed(&result_before, result) {
            *result = result_before;
            stats.support_rollback = true;
        }
    }
    Ok(stats)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = parse_args().map_err(std::io::Error::other)?;
    let started = Instant::now();
    let manifest = parse_manifest(&args.manifest).map_err(std::io::Error::other)?;
    let mut snapshot = if let Some(limit) = args.max_matches_per_pair {
        verified_pair_snapshot::read_mapper_compact_capped(&args.snapshot, limit)
    } else {
        verified_pair_snapshot::read_mapper_compact(&args.snapshot)
    }
    .map_err(std::io::Error::other)?;
    let frames =
        build_frames(&manifest.frame_rows, &snapshot.image_names).map_err(std::io::Error::other)?;
    if let (Some(overlay_path), Some(max_frame_gap)) = (
        args.deferred_overlay_snapshot.as_ref(),
        args.deferred_overlay_max_frame_gap,
    ) {
        let overlay_limit = args
            .deferred_overlay_max_matches_per_pair
            .or(args.max_matches_per_pair);
        let overlay = if let Some(limit) = overlay_limit {
            verified_pair_snapshot::read_mapper_compact_capped(overlay_path, limit)
        } else {
            verified_pair_snapshot::read_mapper_compact(overlay_path)
        }
        .map_err(std::io::Error::other)?;
        let (overlay_pairs, overlay_matches) = append_deferred_overlay(
            &mut snapshot,
            overlay,
            &frames,
            args.deferred_overlay_min_temporal_frame_gap,
            max_frame_gap,
        )
        .map_err(std::io::Error::other)?;
        eprintln!(
            "rig-replay deferred overlay: min_temporal_frame_gap={} max_frame_gap={max_frame_gap} pairs={overlay_pairs} matches={overlay_matches} total_pairs={} total_matches={} total_features={}",
            args.deferred_overlay_min_temporal_frame_gap,
            snapshot.pairs.len(),
            snapshot.pairs.iter().map(|pair| pair.matches.len()).sum::<usize>(),
            snapshot.feature_counts.iter().sum::<u64>(),
        );
    }
    let pose_priors = if args.long_pair_pose_prior_images.is_empty() {
        None
    } else {
        Some(
            read_colmap_rotation_priors(&args.long_pair_pose_prior_images, &snapshot.image_names)
                .map_err(std::io::Error::other)?,
        )
    };
    let long_pair_gate = args
        .deferred_long_pair_min_frame_gap
        .zip(args.structure_long_min_e_inliers);
    if args.structure_min_pair_matches.is_some() || long_pair_gate.is_some() {
        let partition = partition_weak_long_pairs(
            &mut snapshot,
            &frames,
            args.structure_min_pair_matches,
            long_pair_gate,
            pose_priors.as_ref(),
            args.structure_long_max_rotation_disagreement_deg,
        )
        .map_err(std::io::Error::other)?;
        args.deferred_registration_pair_prefix = Some(partition.base_pairs);
        eprintln!(
            "rig-replay structure/deferred partition: min_pair_matches={:?} long_pair_gate={:?} pose_prior_files={} max_rotation_disagreement_deg={:?} base_pairs={} base_matches={} deferred_pairs={} deferred_matches={} weak_support_pairs={} weak_support_matches={} pose_checked_pairs={} pose_rejected_pairs={}",
            args.structure_min_pair_matches,
            long_pair_gate,
            args.long_pair_pose_prior_images.len(),
            args.structure_long_max_rotation_disagreement_deg,
            partition.base_pairs,
            partition.base_matches,
            partition.deferred_pairs,
            partition.deferred_matches,
            partition.weak_support_pairs,
            partition.weak_support_matches,
            partition.pose_checked_pairs,
            partition.pose_rejected_pairs,
        );
    }
    let features = snapshot
        .image_names
        .iter()
        .map(|name| {
            read_combined_keypoints(
                &args.features_dir,
                &args.append_features_dirs,
                name,
                &args.feature_suffix,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(std::io::Error::other)?;
    for (index, (&expected, feature)) in snapshot.feature_counts.iter().zip(&features).enumerate() {
        if expected != feature.len() as u64 {
            return Err(format!(
                "feature count mismatch at image {index}: snapshot={expected}, file={}",
                feature.len()
            )
            .into());
        }
    }
    if let Some(max_gap) = args.max_track_frame_gap {
        let assignment = frames
            .iter()
            .enumerate()
            .flat_map(|(frame, rig_frame)| {
                rig_frame
                    .images
                    .iter()
                    .map(move |image| (image.image_index, frame))
            })
            .collect::<HashMap<_, _>>();
        let before_pairs = snapshot.pairs.len();
        let before_matches = snapshot
            .pairs
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>();
        snapshot.pairs.retain(|pair| {
            let (Ok(image_i), Ok(image_j)) =
                (usize::try_from(pair.image_i), usize::try_from(pair.image_j))
            else {
                return false;
            };
            assignment
                .get(&image_i)
                .zip(assignment.get(&image_j))
                .is_some_and(|(&frame_i, &frame_j)| frame_i.abs_diff(frame_j) <= max_gap)
        });
        eprintln!(
            "rig-replay track-gap filter: max_frame_gap={max_gap} pairs={before_pairs}->{} matches={before_matches}->{}",
            snapshot.pairs.len(),
            snapshot
                .pairs
                .iter()
                .map(|pair| pair.matches.len())
                .sum::<usize>(),
        );
    }
    let mut pairs = convert_pairs(&snapshot).map_err(std::io::Error::other)?;
    if let Some(path) = args.deferred_quadrilateral_whitelist_tsv.as_ref() {
        let prefix = args.deferred_registration_pair_prefix.ok_or(
            "--deferred-quadrilateral-whitelist-tsv requires --deferred-registration-pair-prefix",
        )?;
        let edges = read_quadrilateral_whitelist(path)?;
        let retriangulation_prefix = pairs.len();
        let mut filtered = pairs[prefix..].to_vec();
        let (before, after) =
            filter_deferred_pairs_by_quadrilateral_whitelist(&mut filtered, 0, &edges)
                .map_err(std::io::Error::other)?;
        let duplicate_snapshot_pairs = snapshot.pairs[prefix..].to_vec();
        pairs.append(&mut filtered);
        snapshot.pairs.extend(duplicate_snapshot_pairs);
        args.deferred_retriangulation_pair_prefix = Some(retriangulation_prefix);
        eprintln!(
            "rig-replay photometric quadrilateral whitelist: edges={} retriangulation_prefix={} deferred_matches={before}->{after} source={}",
            edges.len(),
            retriangulation_prefix,
            path.display(),
        );
    }
    eprintln!(
        "rig-replay input: frames={} images={} pairs={} matches={} keypoints={} VmHWM={} KiB",
        frames.len(),
        features.len(),
        pairs.len(),
        pairs.iter().map(|pair| pair.matches.len()).sum::<usize>(),
        features.iter().map(FeatureSet::len).sum::<usize>(),
        peak_rss_kib().unwrap_or(0),
    );
    if args.preview_pair_confidence_conflicts {
        let pair_count = args
            .deferred_registration_pair_prefix
            .unwrap_or(pairs.len());
        let mapping_pairs = pairs.get(..pair_count).ok_or_else(|| {
            format!(
                "--deferred-registration-pair-prefix {pair_count} exceeds {} verified pairs",
                pairs.len()
            )
        })?;
        let trusted_prefix = match args.track_builder {
            RigTrackBuilder::TrustedPrefixPairConfidence(pair_count) => pair_count,
            _ => 0,
        };
        let preview_started = Instant::now();
        let stats =
            preview_pair_confidence_conflicts(features.len(), mapping_pairs, trusted_prefix);
        eprintln!(
            "rig-pair-confidence-conflicts: correspondences={} nodes={} accepted_edges={} rejected_edges={} final_components={} conflict_regions={} involved_components={} involved_observations={} max_region_components={} max_region_observations={} max_overlapping_images_per_rejected_edge={} histogram={:?} build_seconds={:.6} VmHWM={} KiB",
            stats.correspondences,
            stats.nodes,
            stats.accepted_edges,
            stats.rejected_edges,
            stats.final_components,
            stats.conflict_regions,
            stats.involved_components,
            stats.involved_observations,
            stats.max_region_components,
            stats.max_region_observations,
            stats.max_overlapping_images_per_rejected_edge,
            stats.region_component_count_histogram,
            preview_started.elapsed().as_secs_f64(),
            peak_rss_kib().unwrap_or(0),
        );
        return Ok(());
    }
    if args.preview_rig_correspondence_csr {
        let pair_count = args
            .deferred_registration_pair_prefix
            .unwrap_or(pairs.len());
        let mapping_pairs = pairs.get(..pair_count).ok_or_else(|| {
            format!(
                "--deferred-registration-pair-prefix {pair_count} exceeds {} verified pairs",
                pairs.len()
            )
        })?;
        let feature_counts = features.iter().map(FeatureSet::len).collect::<Vec<_>>();
        let started = Instant::now();
        let graph = build_rig_correspondence(&feature_counts, mapping_pairs)?;
        eprintln!(
            "rig-correspondence-csr: images={} pairs={} observations={} undirected_edges={} directed_edges={} duplicate_drops={} max_row_degree={} persistent_bytes={} digest={:016x} build_seconds={:.6} VmHWM={} KiB",
            features.len(),
            mapping_pairs.len(),
            graph.stats.total_observations,
            graph.stats.undirected_unique_edges,
            graph.stats.directed_unique_edges,
            graph.stats.duplicate_drops,
            graph.stats.max_row_degree,
            graph.stats.estimated_persistent_bytes,
            graph.stats.digest,
            started.elapsed().as_secs_f64(),
            peak_rss_kib().unwrap_or(0),
        );
        return Ok(());
    }
    if let Some(path) = args.export_deferred_quadrilaterals_tsv.as_ref() {
        let pair_prefix = args.deferred_registration_pair_prefix.ok_or(
            "--export-deferred-quadrilaterals-tsv requires --deferred-registration-pair-prefix",
        )?;
        let tracks = export_deferred_quadrilaterals(
            path,
            &frames,
            &features,
            &pairs,
            &snapshot.image_names,
            pair_prefix,
            args.deferred_overlay_min_temporal_frame_gap,
            args.deferred_overlay_max_frame_gap.unwrap_or(1),
        )?;
        eprintln!(
            "rig-replay exported deferred quadrilaterals: tracks={tracks} observations={} out={}",
            tracks.saturating_mul(4),
            path.display()
        );
        return Ok(());
    }
    let mapping_started = Instant::now();
    if args.max_models > 1 {
        return map_remaining_models(
            &args,
            &manifest.rig,
            &frames,
            &features,
            &pairs,
            &snapshot.pairs,
            &snapshot.image_names,
            mapping_started,
        );
    }
    let config = mapper_config(&args);
    let mut result = incremental_rig_sfm(&manifest.rig, &frames, &features, &pairs, &config)?;
    if let Some(max_disagreement_deg) = args.final_rig_rotation_average_max_disagreement_deg {
        let rotation_stats = apply_final_rig_rotation_average(
            &manifest.rig,
            &frames,
            &features,
            &snapshot.pairs,
            max_disagreement_deg,
            args.final_rig_rotation_average_max_update_deg,
            args.final_rig_rotation_average_weight_cap,
            &config,
            &mut result,
        )?;
        eprintln!(
            "rig-replay rotation-average: max_disagreement_deg={max_disagreement_deg:.6} max_update_deg={:.6} weight_cap={} candidate_pairs={} accepted_pairs={} deduped_pairs={} registered_components={} averaged_components={} failed_components={} unsafe_update_components={} averaged_frames={} disagreement_rejections={} max_proposed_update_deg={:.6} max_applied_update_deg={:.6} support_rollback={}",
            args.final_rig_rotation_average_max_update_deg,
            args.final_rig_rotation_average_weight_cap,
            rotation_stats.candidate_pairs,
            rotation_stats.accepted_pairs,
            rotation_stats.deduped_pairs,
            rotation_stats.registered_components,
            rotation_stats.averaged_components,
            rotation_stats.failed_components,
            rotation_stats.unsafe_update_components,
            rotation_stats.averaged_frames,
            rotation_stats.disagreement_rejections,
            rotation_stats.max_proposed_update_deg,
            rotation_stats.max_rotation_update_deg,
            rotation_stats.support_rollback,
        );
    }
    if args.final_rig_translation_average {
        let translation_stats = apply_final_rig_translation_average(
            &manifest.rig,
            &frames,
            &features,
            &snapshot.pairs,
            args.final_rig_translation_average_min_frame_gap,
            args.final_rig_translation_average_min_matches,
            args.final_rig_translation_average_max_direction_error_deg,
            args.final_rig_translation_average_max_update_m,
            args.final_rig_translation_average_iterations,
            &config,
            &mut result,
        )?;
        eprintln!(
            "rig-replay translation-average: candidate_pairs={} accepted_pairs={} degenerate_pairs={} direction_rejections={} deduped_edges={} backbone_edges={} solved_frames={} max_proposed_update_m={:.9} candidate_tracks={} candidate_observations={} candidate_mean_reprojection_px={:.9} support_regressed={} reprojection_regressed={} unsafe_update={} support_rollback={}",
            translation_stats.candidate_pairs,
            translation_stats.accepted_pairs,
            translation_stats.degenerate_pairs,
            translation_stats.direction_rejections,
            translation_stats.deduped_edges,
            translation_stats.backbone_edges,
            translation_stats.solved_frames,
            translation_stats.max_proposed_update_m,
            translation_stats.candidate_tracks,
            translation_stats.candidate_observations,
            translation_stats.candidate_mean_reprojection_px,
            translation_stats.support_regressed,
            translation_stats.reprojection_regressed,
            translation_stats.unsafe_update,
            translation_stats.support_rollback,
        );
    }
    let mapper_seconds = mapping_started.elapsed().as_secs_f64();
    if let Some(ba) = result.bundle_adjustment {
        eprintln!(
            "rig-replay BA: observations={} cost={:.9}->{:.9} iterations={} converged={}",
            ba.observations, ba.initial_cost, ba.final_cost, ba.iterations, ba.converged,
        );
    }

    let registered = result
        .image_poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.as_ref().map(|pose| (image, pose.clone())))
        .collect::<Vec<_>>();
    let remap = registered
        .iter()
        .enumerate()
        .map(|(output, (input, _))| (*input, output))
        .collect::<HashMap<_, _>>();
    let cameras = registered
        .iter()
        .map(|(image, _)| {
            let sensor_index = frames
                .iter()
                .flat_map(|frame| &frame.images)
                .find(|row| row.image_index == *image)
                .expect("validated image assignment")
                .sensor_index;
            manifest.rig.sensors()[sensor_index].camera.clone()
        })
        .collect::<Vec<_>>();
    let poses = registered
        .iter()
        .map(|(_, pose)| pose.clone())
        .collect::<Vec<_>>();
    let names = registered
        .iter()
        .map(|(image, _)| snapshot.image_names[*image].clone())
        .collect::<Vec<_>>();
    let output_features = registered
        .iter()
        .map(|(image, _)| features[*image].clone())
        .collect::<Vec<_>>();
    let landmarks = result
        .tracks
        .iter()
        .map(|track| {
            let observations = track
                .observations
                .iter()
                .filter_map(|&(image, keypoint, pixel)| {
                    remap.get(&image).map(|output| (*output, keypoint, pixel))
                })
                .collect::<Vec<_>>();
            (track.position, observations)
        })
        .collect::<Vec<_>>();
    let export = write_colmap_reconstruction_for_3dgs_with_cameras(
        &args.out_colmap,
        &cameras,
        &poses,
        &output_features,
        &landmarks,
        |index| names[index].clone(),
    )?;
    println!(
        concat!(
            "rig-replay result: registered_frames={}/{} registered_images={}/{} ",
            "tracks={} observations={} mean_reprojection_px={:.9} seed_frame={} ",
            "mapper_seconds={:.6} total_seconds={:.6} VmHWM_KiB={} ",
            "track_components={} conflicting_track_edges={} retained_track_observations={} ",
            "triangulation_attempts={} robust_triangulation_tracks={} robust_triangulation_pruned_observations={} robust_triangulation_majority_rejections={} cache_insertions={} pnp_attempts={} pnp_insufficient_sensors={} pnp_estimation_failures={} pnp_inlier_rejections={} pnp_registrations={} dynamic_activated_rows={} dynamic_activated_edges={} dynamic_track_creates={} dynamic_track_continues={} dynamic_owner_conflicts={} dynamic_same_image_conflicts={} dynamic_geometry_rejections={} dynamic_observation_lookup_entries={} dynamic_pnp_graph_insertions={} dynamic_bootstrap_legacy_tracks={} dynamic_bootstrap_candidates={} dynamic_bootstrap_seed_support={} dynamic_bootstrap_seed_pairs={} dynamic_bootstrap_seed_landmarks={} dynamic_bootstrap_direct_fallbacks={} direct_bridge_pair_visits={} direct_bridge_correspondences={} direct_bridge_registrations={} motion_bridge_pair_visits={} motion_bridge_estimation_failures={} motion_bridge_rotation_rejections={} motion_bridge_registrations={} deferred_pair_visits={} deferred_correspondences={} deferred_pnp_attempts={} deferred_pnp_estimation_failures={} deferred_pnp_inlier_rejections={} deferred_registrations={} deferred_interpolation_registrations={} deferred_observations_attached={} deferred_retriangulated_tracks={} deferred_retriangulated_observations={} unregistered_zero_support={} unregistered_below_pnp_support={} unregistered_eligible_pnp={} unregistered_below_sensors={} max_unregistered_support={} local_ba_runs={} ba_retriangulated_tracks={} ba_requeued_frames={} structure_refined_tracks={} geometry_recovered_tracks={} geometry_recovered_observations={} track_completion_passes={} track_completion_pair_visits={} track_completion_observations={} track_completion_reprojection_rejections={} final_filter_refinement_passes={} final_filter_refinement_pruned_observations={} isolated_pose_repair_passes={} isolated_pose_repairs={} paired_pose_jump_repairs={} paired_pose_jump_repaired_frames={} out={}"
        ),
        result.registered_frames,
        frames.len(),
        result.registered_images,
        features.len(),
        export.landmark_count,
        export.observation_count,
        result.mean_reprojection_error_px,
        result.seed_frame_index,
        mapper_seconds,
        started.elapsed().as_secs_f64(),
        peak_rss_kib().unwrap_or(0),
        result.track_build_stats.connected_components,
        result.track_build_stats.conflicting_components,
        result.track_build_stats.retained_observations,
        result.work.triangulation_attempts,
        result.work.robust_triangulation_tracks,
        result.work.robust_triangulation_pruned_observations,
        result.work.robust_triangulation_majority_rejections,
        result.work.correspondence_cache_insertions,
        result.work.pnp_attempts,
        result.work.pnp_insufficient_sensor_attempts,
        result.work.pnp_estimation_failures,
        result.work.pnp_inlier_rejections,
        result.work.pnp_registrations,
        result.work.dynamic_activated_rows,
        result.work.dynamic_activated_edges,
        result.work.dynamic_track_creates,
        result.work.dynamic_track_continues,
        result.work.dynamic_owner_conflicts,
        result.work.dynamic_same_image_conflicts,
        result.work.dynamic_geometry_rejections,
        result.work.dynamic_observation_lookup_entries,
        result.work.dynamic_pnp_graph_insertions,
        result.work.dynamic_bootstrap_legacy_tracks,
        result.work.dynamic_bootstrap_candidates,
        result.work.dynamic_bootstrap_seed_support,
        result.work.dynamic_bootstrap_seed_pairs,
        result.work.dynamic_bootstrap_seed_landmarks,
        result.work.dynamic_bootstrap_direct_fallbacks,
        result.work.direct_bridge_pair_visits,
        result.work.direct_bridge_correspondence_insertions,
        result.work.direct_bridge_registrations,
        result.work.motion_bridge_pair_visits,
        result.work.motion_bridge_estimation_failures,
        result.work.motion_bridge_rotation_rejections,
        result.work.motion_bridge_registrations,
        result.work.deferred_pair_visits,
        result.work.deferred_correspondence_insertions,
        result.work.deferred_pnp_attempts,
        result.work.deferred_pnp_estimation_failures,
        result.work.deferred_pnp_inlier_rejections,
        result.work.deferred_registrations,
        result.work.deferred_interpolation_registrations,
        result.work.deferred_observations_attached,
        result.work.deferred_retriangulated_tracks,
        result.work.deferred_retriangulated_observations,
        result.work.unregistered_zero_support_frames,
        result.work.unregistered_below_pnp_support_frames,
        result.work.unregistered_eligible_pnp_frames,
        result.work.unregistered_below_sensor_frames,
        result.work.max_unregistered_support,
        result.work.local_ba_runs,
        result.work.ba_retriangulated_tracks,
        result.work.ba_requeued_frames,
        result.work.structure_refined_tracks,
        result.work.geometry_recovered_tracks,
        result.work.geometry_recovered_observations,
        result.work.track_completion_passes,
        result.work.track_completion_pair_visits,
        result.work.track_completion_observations,
        result.work.track_completion_reprojection_rejections,
        result.work.final_filter_refinement_passes,
        result.work.final_filter_refinement_pruned_observations,
        result.work.isolated_pose_repair_passes,
        result.work.isolated_pose_repairs,
        result.work.paired_pose_jump_repairs,
        result.work.paired_pose_jump_repaired_frames,
        args.out_colmap.display(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use visloc_rs::SfmTrack;

    fn snapshot_pair(
        image_i: u64,
        image_j: u64,
        e_inlier_count: u64,
        relative_rotation: UnitQuaternion<f64>,
    ) -> verified_pair_snapshot::PairRecord {
        let matrix = relative_rotation.to_rotation_matrix();
        verified_pair_snapshot::PairRecord {
            image_i,
            image_j,
            raw_match_count: 0,
            raw_matches: Vec::new(),
            accepted_inlier_indices: Vec::new(),
            essential_inlier_indices: Vec::new(),
            matches: Vec::new(),
            essential_matches: None,
            config: 3,
            calibrated: true,
            e_inlier_count,
            f_inlier_count: 0,
            h_inlier_count: 0,
            essential_matrix_bits: None,
            fundamental_matrix_bits: None,
            homography_matrix_bits: None,
            relative_rotation_bits: Some(std::array::from_fn(|index| {
                matrix.matrix().as_slice()[index].to_bits()
            })),
            relative_translation_bits: None,
        }
    }

    #[test]
    fn parses_manifest_and_preserves_frame_grouping() {
        let root = std::env::temp_dir().join(format!(
            "visloc-generalized-rig-manifest-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("rig.txt");
        std::fs::write(
            &path,
            concat!(
                "S 0 1 848 800 285 286 425 398 1 0 0 0 0 0 0\n",
                "S 1 2 848 800 284 286 428 397 1 0 0 0 -0.2 0 0\n",
                "F 10 left.png 0\n",
                "F 10 right.png 1\n",
            ),
        )
        .unwrap();
        let parsed = parse_manifest(&path).unwrap();
        assert_eq!(parsed.rig.sensors().len(), 2);
        let frames = build_frames(
            &parsed.frame_rows,
            &["left.png".to_owned(), "right.png".to_owned()],
        )
        .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].images.len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fixed_rotation_translation_direction_recovers_camera_centre_motion() {
        let camera = Camera::pinhole(1, 640, 480, 400.0, 400.0, 320.0, 240.0);
        let centre_j = Vector3::new(1.0, 0.0, 0.0);
        let points = (0..16)
            .map(|index| {
                Point3::new(
                    -1.4 + 0.21 * index as f64,
                    -0.8 + 0.13 * ((index * 7) % 11) as f64,
                    4.0 + 0.37 * ((index * 5) % 13) as f64,
                )
            })
            .collect::<Vec<_>>();
        let pixels_i = points
            .iter()
            .map(|point| {
                Point2::new(
                    400.0 * point.x / point.z + 320.0,
                    400.0 * point.y / point.z + 240.0,
                )
            })
            .collect::<Vec<_>>();
        let pixels_j = points
            .iter()
            .map(|point| {
                let point = point.coords - centre_j;
                Point2::new(
                    400.0 * point.x / point.z + 320.0,
                    400.0 * point.y / point.z + 240.0,
                )
            })
            .collect::<Vec<_>>();
        let features_i = FeatureSet::new(pixels_i, vec![Vec::new(); points.len()]).unwrap();
        let features_j = FeatureSet::new(pixels_j, vec![Vec::new(); points.len()]).unwrap();
        let matches = (0..points.len())
            .map(|index| (index as u64, index as u64))
            .collect::<Vec<_>>();

        let (direction, nullspace_ratio) = fixed_rotation_translation_direction(
            &camera,
            &camera,
            &UnitQuaternion::identity(),
            &features_i,
            &features_j,
            &matches,
        )
        .expect("well-conditioned translation direction");

        assert!(direction.dot(&Vector3::x()).abs() > 0.999_999);
        assert!(nullspace_ratio < 1.0e-10);
    }

    #[test]
    fn weak_long_partition_keeps_local_and_strong_geometry_in_structure() {
        assert!(!should_defer_weak_long_pair(10, 42, 20, 128, 128));
        assert!(should_defer_weak_long_pair(10, 138, 127, 128, 128));
        assert!(!should_defer_weak_long_pair(10, 138, 128, 128, 128));
        assert!(!should_defer_weak_long_pair(10, 138, 200, 128, 128));
    }

    #[test]
    fn rotation_prior_gate_matches_colmap_world_to_camera_composition() {
        let root = std::env::temp_dir().join(format!(
            "visloc-generalized-rig-rotation-prior-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("images.txt");
        let rotation_j = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.1);
        std::fs::write(
            &path,
            format!(
                "1 1 0 0 0 0 0 0 1 a.png\n2 {} {} {} {} 0 0 0 1 b.png\n3 {} {} {} {} 0 0 0 1 c.png\n",
                rotation_j.w,
                rotation_j.i,
                rotation_j.j,
                rotation_j.k,
                rotation_j.w,
                rotation_j.i,
                rotation_j.j,
                rotation_j.k,
            ),
        )
        .unwrap();
        let image_names = vec!["a.png".into(), "b.png".into(), "c.png".into()];
        let priors = read_colmap_rotation_priors(&[path.clone()], &image_names).unwrap();
        let good = snapshot_pair(0, 1, 10, rotation_j);
        assert!(pair_rotation_disagreement_deg(&good, &image_names, &priors).unwrap() < 1.0e-9);

        // A pair from b to c should have identity relative rotation because
        // both prior images use the same world-to-camera rotation.  Feeding
        // a ten-degree observation must therefore be rejected by a five-
        // degree structure gate.
        let bad = snapshot_pair(
            1,
            2,
            10,
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.1),
        );
        let mut snapshot = verified_pair_snapshot::Snapshot {
            schema_version: verified_pair_snapshot::SCHEMA_VERSION,
            image_names: image_names.clone(),
            image_manifest_hash: 0,
            feature_manifest_hash: 0,
            feature_counts: vec![0; image_names.len()],
            width: 1,
            height: 1,
            intrinsics_bits: [0; 4],
            effective_config_hash: 0,
            effective_config: String::new(),
            verifier_config_hash: 0,
            verifier_config: String::new(),
            pair_order_hash: 0,
            unordered_edge_hash: 0,
            accepted_match_count: 0,
            pairs: vec![good, bad],
        };
        let frames = (0..5)
            .map(|frame| RigFrame {
                images: match frame {
                    0 => vec![RigFrameImage {
                        image_index: 0,
                        sensor_index: 0,
                    }],
                    2 => vec![RigFrameImage {
                        image_index: 1,
                        sensor_index: 0,
                    }],
                    4 => vec![RigFrameImage {
                        image_index: 2,
                        sensor_index: 0,
                    }],
                    _ => Vec::new(),
                },
            })
            .collect::<Vec<_>>();
        let stats = partition_weak_long_pairs(
            &mut snapshot,
            &frames,
            None,
            Some((2, 1)),
            Some(&priors),
            Some(5.0),
        )
        .unwrap();
        assert_eq!(stats.base_pairs, 1);
        assert_eq!(stats.deferred_pairs, 1);
        assert_eq!(stats.pose_checked_pairs, 2);
        assert_eq!(stats.pose_rejected_pairs, 1);
        assert_eq!(snapshot.pairs[0].image_i, 0);
        assert_eq!(snapshot.pairs[1].image_i, 1);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn weak_support_pairs_are_deferred_without_reordering_strong_pairs() {
        let mut weak = snapshot_pair(0, 1, 8, UnitQuaternion::identity());
        weak.matches = vec![(0, 0); 8];
        let mut strong = snapshot_pair(1, 2, 30, UnitQuaternion::identity());
        strong.matches = vec![(0, 0); 30];
        let mut snapshot = verified_pair_snapshot::Snapshot {
            schema_version: verified_pair_snapshot::SCHEMA_VERSION,
            image_names: vec!["a.png".into(), "b.png".into(), "c.png".into()],
            image_manifest_hash: 0,
            feature_manifest_hash: 0,
            feature_counts: vec![30; 3],
            width: 1,
            height: 1,
            intrinsics_bits: [0; 4],
            effective_config_hash: 0,
            effective_config: String::new(),
            verifier_config_hash: 0,
            verifier_config: String::new(),
            pair_order_hash: 0,
            unordered_edge_hash: 0,
            accepted_match_count: 38,
            pairs: vec![weak, strong],
        };
        let frames = (0..3)
            .map(|frame| RigFrame {
                images: vec![RigFrameImage {
                    image_index: frame,
                    sensor_index: 0,
                }],
            })
            .collect::<Vec<_>>();
        let stats =
            partition_weak_long_pairs(&mut snapshot, &frames, Some(30), None, None, None).unwrap();
        assert_eq!(stats.base_pairs, 1);
        assert_eq!(stats.deferred_pairs, 1);
        assert_eq!(stats.weak_support_pairs, 1);
        assert_eq!(stats.weak_support_matches, 8);
        assert_eq!(snapshot.pairs[0].image_i, 1);
        assert_eq!(snapshot.pairs[1].image_i, 0);
    }

    #[test]
    fn frame_rotation_constraint_uses_sensor_to_rig_convention() {
        let sensor_i_rotation = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.23);
        let sensor_j_rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.31);
        let rig_rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.47);
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 32, 24, 20.0, 20.0, 16.0, 12.0),
                sensor_from_rig: SE3::new(sensor_i_rotation, Vector3::zeros()),
            },
            RigSensor {
                camera: Camera::pinhole(2, 32, 24, 20.0, 20.0, 16.0, 12.0),
                sensor_from_rig: SE3::new(sensor_j_rotation, Vector3::zeros()),
            },
        ])
        .unwrap();
        let frames = vec![
            RigFrame {
                images: vec![RigFrameImage {
                    image_index: 0,
                    sensor_index: 0,
                }],
            },
            RigFrame {
                images: vec![RigFrameImage {
                    image_index: 1,
                    sensor_index: 1,
                }],
            },
        ];
        let sensor_rotation = sensor_j_rotation * rig_rotation * sensor_i_rotation.inverse();
        let pair = snapshot_pair(0, 1, 32, sensor_rotation);
        let frame_poses = vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::zeros(),
            )),
            Some(Pose::from_world_to_camera(rig_rotation, Vector3::zeros())),
        ];
        let (constraints, stats) =
            build_frame_rotation_constraints(&rig, &frames, &[pair], &frame_poses, 0.1, 128);
        assert_eq!(constraints.len(), 1);
        assert_eq!(stats.candidate_pairs, 1);
        assert_eq!(constraints[0].from_frame, 0);
        assert_eq!(constraints[0].to_frame, 1);
        assert!(constraints[0].rotation.angle_to(&rig_rotation) < 1.0e-10);
    }

    #[test]
    fn frame_rotation_constraints_filter_and_dedupe_by_capped_e_support() {
        let rig = GeneralizedCameraRig::new(vec![
            RigSensor {
                camera: Camera::pinhole(1, 32, 24, 20.0, 20.0, 16.0, 12.0),
                sensor_from_rig: SE3::identity(),
            },
            RigSensor {
                camera: Camera::pinhole(2, 32, 24, 20.0, 20.0, 16.0, 12.0),
                sensor_from_rig: SE3::identity(),
            },
        ])
        .unwrap();
        let frames = vec![
            RigFrame {
                images: vec![
                    RigFrameImage {
                        image_index: 0,
                        sensor_index: 0,
                    },
                    RigFrameImage {
                        image_index: 1,
                        sensor_index: 1,
                    },
                ],
            },
            RigFrame {
                images: vec![
                    RigFrameImage {
                        image_index: 2,
                        sensor_index: 0,
                    },
                    RigFrameImage {
                        image_index: 3,
                        sensor_index: 1,
                    },
                ],
            },
        ];
        let frame_poses = vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::zeros(),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::zeros(),
            )),
        ];
        let mut uncalibrated = snapshot_pair(0, 2, 70, UnitQuaternion::identity());
        uncalibrated.calibrated = false;
        let mut missing_rotation = snapshot_pair(0, 2, 600, UnitQuaternion::identity());
        missing_rotation.relative_rotation_bits = None;
        let missing_e_support = snapshot_pair(0, 2, 0, UnitQuaternion::identity());
        let rejected_by_gate = snapshot_pair(
            0,
            2,
            300,
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 20.0_f64.to_radians()),
        );
        let weaker = snapshot_pair(0, 2, 20, UnitQuaternion::identity());
        let stronger_after_cap = snapshot_pair(1, 3, 200, UnitQuaternion::identity());
        let same_frame = snapshot_pair(0, 1, 999, UnitQuaternion::identity());
        let (constraints, stats) = build_frame_rotation_constraints(
            &rig,
            &frames,
            &[
                uncalibrated,
                missing_rotation,
                missing_e_support,
                rejected_by_gate,
                weaker,
                stronger_after_cap,
                same_frame,
            ],
            &frame_poses,
            5.0,
            128,
        );
        assert_eq!(stats.disagreement_rejections, 1);
        assert_eq!(stats.candidate_pairs, 3);
        assert_eq!(stats.accepted_pairs, 3);
        assert_eq!(stats.deduped_pairs, 1);
        assert_eq!(constraints[0].capped_e_inliers, 128);
        assert_eq!(
            (constraints[0].source_image_i, constraints[0].source_image_j),
            (1, 3),
            "an uncalibrated record with stored E rotation is admitted, but the stronger capped support still wins"
        );
    }

    #[test]
    fn cumulative_chain_rotation_update_is_rejected_as_one_component() {
        let identity_pose =
            || Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let result = RigSfmResult {
            frame_poses: vec![
                Some(identity_pose()),
                Some(identity_pose()),
                Some(identity_pose()),
            ],
            image_poses: Vec::new(),
            tracks: Vec::new(),
            registered_frames: 3,
            registered_images: 3,
            mean_reprojection_error_px: 0.0,
            seed_frame_index: 0,
            track_build_stats: Default::default(),
            work: Default::default(),
            bundle_adjustment: None,
        };
        let five_degrees = 5.0_f64.to_radians();
        let constraints = vec![
            FrameRotationConstraint {
                from_frame: 0,
                to_frame: 1,
                rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), five_degrees),
                capped_e_inliers: 10,
                source_image_i: 0,
                source_image_j: 1,
            },
            FrameRotationConstraint {
                from_frame: 1,
                to_frame: 2,
                rotation: UnitQuaternion::from_axis_angle(&Vector3::z_axis(), five_degrees),
                capped_e_inliers: 10,
                source_image_i: 1,
                source_image_j: 2,
            },
        ];
        let (desired, stats) = desired_rotations_from_constraints(
            &result,
            &constraints,
            1.0,
            RigRotationAverageStats::default(),
        );
        assert_eq!(stats.registered_components, 1);
        assert_eq!(stats.averaged_components, 0);
        assert_eq!(stats.unsafe_update_components, 1);
        assert!(stats.max_proposed_update_deg > 1.0);
        assert!(desired
            .iter()
            .all(|rotation| { rotation.is_some_and(|rotation| rotation.angle() < 1.0e-10) }));
    }

    #[test]
    fn rotation_average_support_guard_covers_tracks_observations_and_registration() {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let track = SfmTrack {
            position: Point3::origin(),
            observations: vec![(0, 0, Point2::origin()), (1, 0, Point2::origin())],
        };
        let before = RigSfmResult {
            frame_poses: vec![Some(pose.clone()), Some(pose.clone())],
            image_poses: vec![Some(pose.clone()), Some(pose)],
            tracks: vec![track; 10],
            registered_frames: 2,
            registered_images: 2,
            mean_reprojection_error_px: 0.0,
            seed_frame_index: 0,
            track_build_stats: Default::default(),
            work: Default::default(),
            bundle_adjustment: None,
        };
        let mut track_loss = before.clone();
        track_loss.tracks.truncate(8);
        assert!(rotation_average_support_regressed(&before, &track_loss));

        let mut observation_loss = before.clone();
        for track in observation_loss.tracks.iter_mut().take(2) {
            track.observations.truncate(1);
        }
        assert!(!rotation_average_support_regressed(
            &before,
            &observation_loss
        ));
        observation_loss.tracks[0].observations.clear();
        assert!(rotation_average_support_regressed(
            &before,
            &observation_loss
        ));

        let mut registered_loss = before.clone();
        registered_loss.registered_frames = 1;
        assert!(rotation_average_support_regressed(
            &before,
            &registered_loss
        ));
        registered_loss.registered_frames = before.registered_frames;
        registered_loss.registered_images = 1;
        assert!(rotation_average_support_regressed(
            &before,
            &registered_loss
        ));
    }

    #[test]
    fn remaining_model_input_remaps_snapshot_constraint_image_indices() {
        let frames = (0..3)
            .map(|frame| RigFrame {
                images: vec![RigFrameImage {
                    image_index: frame,
                    sensor_index: 0,
                }],
            })
            .collect::<Vec<_>>();
        let features = (0..3)
            .map(|_| FeatureSet::new(vec![Point2::origin()], vec![Vec::new()]).unwrap())
            .collect::<Vec<_>>();
        let pairs = vec![PairwiseMatches::new(1, 2, Vec::new()); 3];
        let snapshot_pairs = vec![snapshot_pair(1, 2, 16, UnitQuaternion::identity()); 3];
        let input = remaining_model_input(
            &frames,
            &features,
            &pairs,
            &snapshot_pairs,
            &[false, true, true],
            Some(1),
            Some(2),
        );
        assert_eq!(input.local_to_global_frames, vec![1, 2]);
        assert_eq!(input.local_to_global_images, vec![1, 2]);
        assert_eq!(input.pairs[0].image_i, 0);
        assert_eq!(input.pairs[0].image_j, 1);
        assert_eq!(input.snapshot_pairs[0].image_i, 0);
        assert_eq!(input.snapshot_pairs[0].image_j, 1);
        assert_eq!(input.base_pair_count, 1);
        assert_eq!(input.registration_pair_count, 2);
    }

    #[test]
    fn deferred_quadrilateral_whitelist_filters_only_the_deferred_suffix() {
        let mut pairs = vec![
            PairwiseMatches::new(0, 1, vec![(0, 1), (2, 3)]),
            PairwiseMatches::new(2, 3, vec![(4, 5), (6, 7)]),
        ];
        let edges = HashSet::from([canonical_observation_edge((2, 4), (3, 5))]);
        let counts =
            filter_deferred_pairs_by_quadrilateral_whitelist(&mut pairs, 1, &edges).unwrap();
        assert_eq!(counts, (2, 1));
        assert_eq!(pairs[0].matches, vec![(0, 1), (2, 3)]);
        assert_eq!(pairs[1].matches, vec![(4, 5)]);
    }
}
