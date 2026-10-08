//! Persistent match-worker plan parsing, validation and execution.

use super::*;

pub(super) const PERSISTENT_MATCH_WORKER_PLAN_MAGIC: &str = "visloc_match_worker_plan_v1";
pub(super) const PERSISTENT_MATCH_WORKER_PLAN_MAGIC_V2: &str = "visloc_match_worker_plan_v2";

#[derive(Debug, Clone)]
pub(super) struct PersistentMatchWorkerShard {
    pub(super) id: usize,
    candidate_path: PathBuf,
    snapshot_path: PathBuf,
    candidate_sha256: String,
}

#[derive(Debug, Clone)]
pub(super) struct PersistentMatchWorkerPlan {
    pub(super) root: PathBuf,
    pub(super) image_names: Vec<String>,
    pub(super) pair_count: usize,
    candidate_index_sha256: String,
    feature_manifest_sha256: String,
    pub(super) candidate_source_sha256: Option<String>,
    pub(super) image_manifest_sha256: Option<String>,
    pub(super) shards: Vec<PersistentMatchWorkerShard>,
}

/// Reject plan paths that could escape the plan directory.  The external
/// runner writes plans at the artifact root, so both candidate and snapshot
/// paths are intentionally simple POSIX-style relative paths.
fn persistent_plan_relative_path(raw: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(raw);
    if raw.is_empty()
        || raw.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
    {
        return Err(format!(
            "persistent match worker {label} must be a simple relative path: {raw:?}"
        ));
    }
    Ok(path)
}

/// Parse the dependency-free, versioned plan consumed by the persistent match
/// worker.  The candidate files remain the source of truth for pair metadata;
/// this plan only binds image order, total coverage, and each input/output
/// path.  Python performs the stronger SHA-256/index validation before launch.
pub(super) fn parse_persistent_match_worker_plan(
    path: &Path,
) -> Result<PersistentMatchWorkerPlan, String> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!(
            "cannot read persistent match worker plan {}: {error}",
            path.display()
        )
    })?;
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    let mut cursor = 0usize;
    let next = |cursor: &mut usize, label: &str| -> Result<&str, String> {
        let line = lines.get(*cursor).copied().ok_or_else(|| {
            format!(
                "persistent match worker plan {} is truncated while reading {label}",
                path.display()
            )
        })?;
        *cursor += 1;
        Ok(line)
    };
    let schema = next(&mut cursor, "header")?;
    let is_v2 = match schema {
        PERSISTENT_MATCH_WORKER_PLAN_MAGIC => false,
        PERSISTENT_MATCH_WORKER_PLAN_MAGIC_V2 => true,
        _ => {
            return Err(format!(
                "persistent match worker plan {} has unsupported header (expected {PERSISTENT_MATCH_WORKER_PLAN_MAGIC} or {PERSISTENT_MATCH_WORKER_PLAN_MAGIC_V2})",
                path.display()
            ));
        }
    };
    let parse_count = |line: &str, kind: &str| -> Result<usize, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 2 || fields[0] != kind {
            return Err(format!(
                "persistent match worker plan {} requires `{kind} N`",
                path.display()
            ));
        }
        fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} {kind} count is not numeric: {error}",
                path.display()
            )
        })
    };
    let image_count = parse_count(next(&mut cursor, "image count")?, "images")?;
    if image_count < 2 {
        return Err(format!(
            "persistent match worker plan {} needs at least two images",
            path.display()
        ));
    }
    let mut image_names = Vec::with_capacity(image_count);
    for expected_index in 0..image_count {
        let line = next(&mut cursor, "image entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 3 || fields[0] != "image" {
            return Err(format!(
                "persistent match worker plan {} image entry must be image INDEX NAME",
                path.display()
            ));
        }
        let index = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} image index is not numeric: {error}",
                path.display()
            )
        })?;
        if index != expected_index || fields[2].is_empty() {
            return Err(format!(
                "persistent match worker plan {} image entry {expected_index} is not ordered",
                path.display()
            ));
        }
        image_names.push(fields[2].to_owned());
    }
    let parse_hash = |line: &str, kind: &str| -> Result<String, String> {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let value = fields.get(1).copied().unwrap_or_default();
        if fields.len() != 2
            || fields[0] != kind
            || value.len() != 64
            || !value.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(format!(
                "persistent match worker plan {} requires `{kind} SHA256`",
                path.display()
            ));
        }
        Ok(value.to_ascii_lowercase())
    };
    let candidate_source_sha256 = if is_v2 {
        Some(parse_hash(
            next(&mut cursor, "candidate source hash")?,
            "candidate_source_sha256",
        )?)
    } else {
        None
    };
    let image_manifest_sha256 = if is_v2 {
        Some(parse_hash(
            next(&mut cursor, "image manifest hash")?,
            "image_manifest_sha256",
        )?)
    } else {
        None
    };
    if let Some(expected_image_manifest_sha256) = image_manifest_sha256.as_deref() {
        let actual_image_manifest_sha256 = candidate_image_manifest_sha256(&image_names);
        if expected_image_manifest_sha256 != actual_image_manifest_sha256 {
            return Err(format!(
                "persistent match worker plan {} image manifest hash differs from image order",
                path.display()
            ));
        }
    }
    let candidate_index_sha256 = parse_hash(
        next(&mut cursor, "candidate index hash")?,
        "candidate_index_sha256",
    )?;
    let feature_manifest_sha256 = parse_hash(
        next(&mut cursor, "feature manifest hash")?,
        "feature_manifest_sha256",
    )?;
    let pair_count = parse_count(next(&mut cursor, "pair count")?, "pairs")?;
    if pair_count == 0 {
        return Err(format!(
            "persistent match worker plan {} must contain at least one pair",
            path.display()
        ));
    }
    let shard_count = parse_count(next(&mut cursor, "shard count")?, "shards")?;
    if shard_count == 0 {
        return Err(format!(
            "persistent match worker plan {} must contain at least one shard",
            path.display()
        ));
    }
    let mut shards = Vec::with_capacity(shard_count);
    let mut all_paths = HashSet::with_capacity(shard_count * 2);
    let mut previous_id = None;
    for _shard_index in 0..shard_count {
        let line = next(&mut cursor, "shard entry")?;
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 5 || fields[0] != "shard" {
            return Err(format!(
                "persistent match worker plan {} shard entry must be shard ID CANDIDATE SNAPSHOT CANDIDATE_SHA256",
                path.display()
            ));
        }
        let id = fields[1].parse::<usize>().map_err(|error| {
            format!(
                "persistent match worker plan {} shard id is not numeric: {error}",
                path.display()
            )
        })?;
        if previous_id.is_some_and(|previous| id <= previous) {
            return Err(format!(
                "persistent match worker plan {} shard IDs must be strictly increasing",
                path.display()
            ));
        }
        previous_id = Some(id);
        let candidate_path =
            persistent_plan_relative_path(fields[2], &format!("shard {id} candidate path"))?;
        let snapshot_path =
            persistent_plan_relative_path(fields[3], &format!("shard {id} snapshot path"))?;
        let candidate_sha256 = parse_hash(
            &format!("candidate_sha256 {}", fields[4]),
            "candidate_sha256",
        )?;
        if candidate_path == snapshot_path {
            return Err(format!(
                "persistent match worker plan {} shard {id} reuses one path for candidate and snapshot",
                path.display()
            ));
        }
        if !all_paths.insert(candidate_path.clone()) {
            return Err(format!(
                "persistent match worker plan {} repeats candidate or snapshot path {}",
                path.display(),
                candidate_path.display()
            ));
        }
        if !all_paths.insert(snapshot_path.clone()) {
            return Err(format!(
                "persistent match worker plan {} repeats candidate or snapshot path {}",
                path.display(),
                snapshot_path.display()
            ));
        }
        shards.push(PersistentMatchWorkerShard {
            id,
            candidate_path,
            snapshot_path,
            candidate_sha256,
        });
    }
    if cursor != lines.len() {
        return Err(format!(
            "persistent match worker plan {} has unexpected trailing data",
            path.display()
        ));
    }
    let root = path
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    Ok(PersistentMatchWorkerPlan {
        root,
        image_names,
        pair_count,
        candidate_index_sha256,
        feature_manifest_sha256,
        candidate_source_sha256,
        image_manifest_sha256,
        shards,
    })
}

/// Dispatch a persistent-worker candidate shard by its versioned envelope.
/// Ordinary ``--candidate-manifest`` input continues to use the legacy v1
/// image-name-bound parser; only the plan-driven worker accepts compact v2.
fn parse_persistent_candidate_manifest(
    path: &Path,
    image_names: &[String],
    plan: &PersistentMatchWorkerPlan,
    expected_candidate_sha256: &str,
) -> Result<(Vec<(usize, usize)>, BTreeMap<String, String>), String> {
    let actual_candidate_sha256 = candidate_file_sha256(path)?;
    if actual_candidate_sha256 != expected_candidate_sha256.to_ascii_lowercase() {
        return Err(format!(
            "candidate shard {path:?} SHA-256 differs from persistent plan"
        ));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read candidate manifest {path:?}: {error}"))?;
    let header = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('#'))
        .ok_or_else(|| format!("candidate manifest {path:?} is empty"))?;
    match header {
        CANDIDATE_MANIFEST_MAGIC => parse_candidate_manifest_with_metadata(path, image_names),
        CANDIDATE_SHARD_MAGIC_V2 => Ok((
            parse_candidate_shard_v2_bound(
                path,
                image_names,
                plan.candidate_source_sha256.as_deref(),
                plan.image_manifest_sha256.as_deref(),
                plan.image_manifest_sha256.as_deref().ok_or_else(|| {
                    format!("candidate shard {path:?} requires a v2 persistent plan image hash")
                })?,
            )?,
            BTreeMap::new(),
        )),
        _ => Err(format!(
            "candidate manifest {path:?} has unsupported header"
        )),
    }
}

/// Keep the plan-driven worker as a narrow, reproducible matching path.  The
/// worker exits before mapper construction, but accepting mapper/diagnostic
/// switches here would make a typo look like a successful persistent A/B.
pub(super) fn validate_persistent_match_worker_args(args: &Args) -> Result<(), String> {
    if args.persistent_match_worker_plan.is_none() {
        return Ok(());
    }
    if args.feature_extractor != FeatureExtractorKind::Files {
        return Err(
            "--persistent-match-worker-plan currently requires --feature-extractor files".into(),
        );
    }
    if args.input_colmap_calibration.is_none() {
        return Err(
            "--persistent-match-worker-plan requires --input-colmap-calibration for the frozen per-image camera contract".into(),
        );
    }
    if args.mapper != MapperKind::Incremental || args.colmap_style {
        return Err(
            "--persistent-match-worker-plan currently requires the plain incremental mapper (remove --mapper global|hybrid and --colmap-style)".into(),
        );
    }
    if args.matcher != MatcherKind::Nn {
        return Err("--persistent-match-worker-plan currently requires --matcher nn".into());
    }
    if args.verification_mode != VerificationMode::Full {
        return Err(
            "--persistent-match-worker-plan currently requires --verification-mode full".into(),
        );
    }
    if args.persistent_match_worker_plan.is_some()
        && (args.candidate_manifest.is_some()
            || args.export_candidate_manifest.is_some()
            || args.exhaustive
            || args.pair_stem_window.is_some()
            || args.local_stem_window.is_some()
            || args.rig_local_grouping
            || args.retrieval_min_frame_gap.is_some()
            || args.candidate_budget.is_some()
            || args.pair_source != PairSource::Vlad
            || args.retrieval_topk != 12
            || args.vocab_size != 64
            || args.vocab_tree_branching != 10
            || args.vocab_tree_depth != 3
            || args.vocab_tree_num_images != 100)
    {
        return Err(
            "--persistent-match-worker-plan owns the candidate shard schedule; remove candidate generation/filter flags".into(),
        );
    }
    if args.export_verified_pairs_snapshot.is_some()
        || args.export_verified_pairs_only
        || args.import_verified_pairs_file.is_some()
        || args.import_verified_pairs_snapshot.is_some()
        || args.snapshot_keypoints_only
        || args.snapshot_coordinate_override_dir.is_some()
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with snapshot/raw-pair import or export modes".into(),
        );
    }
    if args.import_matches_file.is_some() || args.import_matches_supplement_file.is_some() {
        return Err(
            "--persistent-match-worker-plan cannot be combined with raw match imports".into(),
        );
    }
    if args.export_features_dir.is_some()
        || args.export_features_only
        || args.sift_stream_export
        || args.sift_stream_resume
        || args.incremental_correspondence_triangulation
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with feature export or alternate incremental modes".into(),
        );
    }
    if args.multiple_models
        || args.min_e_f_inlier_ratio.is_some()
        || args.calibrated_prefer_essential
        || args.refine_uncalibrated_f_to_essential
        || args.strict_uncalibrated_f_to_essential
        || args.calibrated_essential_primary
        || args.force_essential_matches
        || args.force_essential_uncalibrated_only
        || args.prefer_essential_inliers
        || args.prefer_essential_free_endpoints
        || !args.prefer_essential_stems.is_empty()
        || args.prefer_essential_stem_clique
        || !args.prefer_essential_pairs.is_empty()
        || args.require_essential_selected_edges
        || !args.require_essential_stems.is_empty()
        || args.require_essential_min_e_inliers != 0
        || args.essential_edge_weight_boost != 1.0
    {
        return Err(
            "--persistent-match-worker-plan currently supports only the frozen NN/full verifier settings (optionally with guided matching)".into(),
        );
    }
    if args.sift_append_descriptor_magnification.is_some()
        || args.sift_extra_matches_append_only
        || args.canonical_feature_order
        || args.orientation_locus_canonicalization
        || args.union_traversal_order != UnionTraversalOrder::Original
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with alternate descriptor, feature-order, or union-traversal modes".into(),
        );
    }
    if args.rescue_bridging
        || args.rescue_cross_check
        || args.rescue_match_ratio != 0.95
        || args.rescue_min_matches != 15
        || args.rescue_max_candidates != 200
        || args.sequence_relative_pose_fallback
        || args.sequence_fallback_after_post
        || args.sequence_constant_velocity_scale
        || args.sequence_relaxed_constant_velocity_scale
        || args.sequence_fallback_carry_scale
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with rematch, rescue, or sequence-fallback modes".into(),
        );
    }
    if !args.rematch_stems.is_empty()
        || args.rematch_ratio != 0.9
        || !args.rematch_cross_check
        || args.rematch_guided
        || args.rematch_free_vs_priors
        || args.rematch_prefer_min_e_inliers != 0
        || !args.rematch_prefer_strong_stems.is_empty()
        || args.rematch_tracks_use_essential
        || args.rematch_min_chirality_margin != 0.0
        || args.rematch_prior_anchor
        || args.rematch_min_e_f_inlier_ratio.is_some()
        || args.rematch_calibrated_prefer_essential
        || args.rematch_prior_ray_guided
        || args.rematch_prior_ray_min_rays != 2
        || args.rematch_prior_ray_min_e_inliers != 25
        || args.rematch_anchor_min_e_inliers != 25
        || args.rematch_prefer_strong_min_e != 50
        || args.rematch_verification_mode.is_some()
        || args.rematch_pose_guided_after_global
        || args.rematch_pose_guided_gt.is_some()
        || args.rematch_max_gt_bearing_deg != 0.0
        || args.rematch_gt_bearing_path.is_some()
        || args.rematch_guided_max_error_px.is_some()
        || args.rematch_guided_lowe_ratio.is_some()
        || args.rematch_require_calibrated
        || args.rematch_max_mean_sampson != 0.0
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with rematch configuration".into(),
        );
    }
    if args.diagnose_ba_oracle_poses_file.is_some()
        || args.diagnose_fixed_rotation_ba.is_some()
        || args.diagnose_model_score_file.is_some()
        || args.initial_poses_file.is_some()
        || args.diagnose_colmap_track_membership.is_some()
        || !args.diagnose_pairs.is_empty()
        || args.diagnose_pairs_csv.is_some()
        || !args.diagnose_pair_stems.is_empty()
        || args.diagnose_bearing_gt.is_some()
        || !args.diagnose_bearing_stems.is_empty()
        || args.gt_chirality_oracle
        || args.gt_chirality_oracle_path.is_some()
        || args.rematch_gt_bearing_path.is_some()
        || args.rematch_pose_guided_gt.is_some()
    {
        return Err(
            "--persistent-match-worker-plan cannot consume GT/oracle or matching diagnostic inputs"
                .into(),
        );
    }
    if args.track_source != TrackSource::UnionFind
        || args.confidence_ordered_tracks
        || args.geometric_confidence_tracks
        || args.stable_track_order
        || args.cycle_supported_tracks
        || args.geometry_guided_conflict_recovery
        || args.pose_guided_track_splitting
        || args.pose_guided_track_merging
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with alternate mapper track/recovery modes".into(),
        );
    }
    if args.final_min_track_length.is_some()
        || args.seed_pair.is_some()
        || args.seed_trials != 12
        || !args.final_ba
        || args.refine_intrinsics
        || args.refine_distortion
        || args.refine_tangential_distortion
        || args.shared_focal
        || args.final_iterative_global_refinement
        || args.global_ba_max_refinements.is_some()
        || args.post_refinement_registration
        || args.structureless_registration
        || args.pnp_max_iterations != 128
        || args.ba_max_iterations.is_some()
        || args.ba_huber_delta.is_some()
        || args.ba_linear_solver.is_some()
        || args.periodic_ba_min_registered_images != 0
        || args.final_ba_polish_iterations != 0
        || args.geometry_weighted_ba
        || args.freeze_ill_conditioned_landmarks
        || args.landmark_ba_warm_start_iterations != 0
        || args.landmark_ba_warm_start_min_registered_images != 0
        || args.filter_images
        || args.min_pnp_inliers != 12
        || args.max_mapper_matches_per_pair.is_some()
        || args.max_reproj != 4.0
        || args.next_image_policy != NextImagePolicy::Auto
        || args.chirality_harden
        || args.rotation_seed_trials != 1
        || args.refine_global_translations
        || args.global_independent_edge_scales
        || args.multi_hypothesis_edges
        || args.min_edge_inliers != 15
        || args.min_edge_parallax_deg != 2.0
        || args.weight_by_chirality_margin
        || args.hybrid_filter_priors
        || args.hybrid_drop_inconsistent_priors
        || !args.hybrid_drop_prior_stems.is_empty()
        || args.verify_registration_two_view
        || args.hybrid_rotation_priors_only
        || args.joint_global_positioning
        || args.calibrated_view_edges_only
        || args.pose_guided_track_splitting
        || args.pose_guided_track_splitting_graph_support
        || args.pose_guided_track_splitting_bridge_cuts
        || args.pose_guided_split_max_reproj.is_some()
        || args.pose_guided_track_splitting_iterations.is_some()
        || args.pose_guided_track_merging
        || args.pose_guided_merge_max_reproj.is_some()
        || args.matrix_free_ba
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with mapper/refinement options"
                .into(),
        );
    }
    if args.lightglue_model.is_some()
        || args.onnx_backend != "auto"
        || args.lightglue_max_keypoints != 0
    {
        return Err(
            "--persistent-match-worker-plan cannot be combined with learned-matcher options".into(),
        );
    }
    Ok(())
}

pub(super) const STREAM_DESCRIPTOR_CACHE_ROWS: usize = 65_536;
const PERSISTENT_MATCH_MAX_PAIRS_PER_SHARD: usize = 32;

/// Run a versioned, plan-driven match worker without reloading the feature
/// bank between candidate shards.  Candidate manifests are preflighted before
/// the first snapshot is published so a duplicate pair or metadata mismatch
/// cannot leave a prefix that looks complete.  Verification itself remains
/// the existing `verify_pairs` implementation and each shard's temporary
/// result is dropped before the next shard starts.
pub(super) fn run_persistent_match_worker(
    plan: &PersistentMatchWorkerPlan,
    features: &mut [FeatureSet],
    image_names: &[String],
    camera: &Camera,
    matcher: &PairMatcher,
    args: &Args,
    feature_validation: &SnapshotFeatureValidation,
    stream_sources: Option<(&[PathBuf], &[SnapshotFeatureFileFingerprint])>,
) -> Result<(), Box<dyn std::error::Error>> {
    if plan.image_names != image_names {
        return Err("persistent match worker plan image order differs from loaded features".into());
    }
    let mut seen_pairs = HashSet::with_capacity(plan.pair_count);
    let mut expected_metadata: Option<BTreeMap<String, String>> = None;
    let mut total_pairs = 0usize;
    for shard in &plan.shards {
        let candidate_path = plan.root.join(&shard.candidate_path);
        let (candidates, metadata) = parse_persistent_candidate_manifest(
            &candidate_path,
            image_names,
            plan,
            &shard.candidate_sha256,
        )
        .map_err(std::io::Error::other)?;
        if candidates.len() > PERSISTENT_MATCH_MAX_PAIRS_PER_SHARD {
            return Err(format!(
                "persistent match worker shard {} has {} candidate pairs; maximum is {}",
                shard.id,
                candidates.len(),
                PERSISTENT_MATCH_MAX_PAIRS_PER_SHARD
            )
            .into());
        }
        if let Some(expected) = expected_metadata.as_ref() {
            if expected != &metadata {
                return Err(format!(
                    "persistent match worker candidate shard {} metadata differs from shard 0",
                    shard.id
                )
                .into());
            }
        } else {
            expected_metadata = Some(metadata);
        }
        for &pair in &candidates {
            if !seen_pairs.insert(pair) {
                return Err(format!(
                    "persistent match worker candidate shards overlap at pair ({},{})",
                    pair.0, pair.1
                )
                .into());
            }
        }
        total_pairs = total_pairs
            .checked_add(candidates.len())
            .ok_or("persistent match worker candidate pair count overflow")?;
        // This pass only validates coverage/metadata.  Do not retain any
        // candidate vectors while the feature bank is resident; each shard is
        // parsed again immediately before verification below.
        drop(candidates);
    }
    if total_pairs != plan.pair_count {
        return Err(format!(
            "persistent match worker plan declares {} pairs but candidate shards contain {}",
            plan.pair_count, total_pairs
        )
        .into());
    }

    let mut stdout = std::io::stdout().lock();
    writeln!(
        stdout,
        "persistent-match-plan candidate_index_sha256={} feature_manifest_sha256={}",
        plan.candidate_index_sha256, plan.feature_manifest_sha256,
    )?;
    stdout.flush()?;
    let mut stream_cache = BTreeSet::<(u64, usize)>::new();
    let mut stream_stamps = vec![None; features.len()];
    let mut stream_clock = 0u64;
    let mut stream_resident_rows = 0usize;
    let mut stream_peak_resident_rows = 0usize;
    let mut stream_peak_resident_images = 0usize;
    let shared_writer = if args.shared_snapshot_envelope {
        let first = plan
            .shards
            .first()
            .ok_or("shared snapshot worker requires a shard")?;
        let path = plan.root.join(&first.snapshot_path);
        let directory = path.parent().ok_or("shared snapshot directory missing")?;
        if plan
            .shards
            .iter()
            .any(|shard| plan.root.join(&shard.snapshot_path).parent() != Some(directory))
        {
            return Err("shared snapshot worker requires one output directory".into());
        }
        let envelope = snapshot_for_export(
            image_names,
            features,
            camera,
            &[],
            &HashMap::new(),
            args,
            Some(feature_validation),
        )?;
        Some(verified_pair_snapshot::SharedSnapshotWriter::new(
            directory, &envelope,
        )?)
    } else {
        None
    };
    for shard in &plan.shards {
        let candidate_path = plan.root.join(&shard.candidate_path);
        let (candidates, _candidate_metadata) = parse_persistent_candidate_manifest(
            &candidate_path,
            image_names,
            plan,
            &shard.candidate_sha256,
        )
        .map_err(std::io::Error::other)?;
        let started = std::time::Instant::now();
        let incident_images = candidate_incident_images(&candidates);
        if let Some((paths, fingerprints)) = stream_sources {
            let missing = incident_images
                .iter()
                .copied()
                .filter(|&image| stream_stamps[image].is_none())
                .collect::<Vec<_>>();
            hydrate_match_images(features, paths, fingerprints, &missing)
                .map_err(std::io::Error::other)?;
            for &image in &incident_images {
                if let Some(old_stamp) = stream_stamps[image] {
                    stream_cache.remove(&(old_stamp, image));
                } else {
                    stream_resident_rows += features[image].descriptors.len();
                }
                stream_clock = stream_clock.wrapping_add(1);
                stream_stamps[image] = Some(stream_clock);
                stream_cache.insert((stream_clock, image));
            }
            stream_peak_resident_rows = stream_peak_resident_rows.max(stream_resident_rows);
            stream_peak_resident_images = stream_peak_resident_images.max(stream_cache.len());
        }
        let (mut pairwise, _stats, metadata) = verify_pairs(
            features,
            camera,
            &candidates,
            args.match_ratio,
            args.min_matches,
            args.verification_mode,
            matcher,
            true,
            args.guided_matching,
            args.multiple_models,
            args.min_e_f_inlier_ratio,
            args.calibrated_prefer_essential,
            args.refine_uncalibrated_f_to_essential,
            args.strict_uncalibrated_f_to_essential,
            args.calibrated_essential_primary,
            args.force_essential_matches,
            args.force_essential_min_ef_ratio,
            args.force_essential_min_e_inliers,
            args.force_essential_uncalibrated_only,
            None,
            None,
            None,
            None,
            args.colmap_guided_matching,
        );
        let edge_hash_before = unordered_pairwise_edge_hash(&pairwise);
        apply_union_traversal_order_with_features(
            &mut pairwise,
            args.union_traversal_order,
            features,
        );
        let edge_hash_after = unordered_pairwise_edge_hash(&pairwise);
        if edge_hash_before != edge_hash_after {
            return Err(format!(
                "persistent match worker shard {} changed the verified edge multiset: before={edge_hash_before:016x} after={edge_hash_after:016x}",
                shard.id
            )
            .into());
        }
        let accepted = pairwise
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>();
        let ordered_hash = ordered_pairwise_edge_hash(&pairwise);
        let unordered_hash = edge_hash_after;
        let snapshot_path = plan.root.join(&shard.snapshot_path);
        if let Some(writer) = &shared_writer {
            let records = pairwise
                .iter()
                .map(|pair| {
                    let key = (
                        pair.image_i.min(pair.image_j),
                        pair.image_i.max(pair.image_j),
                    );
                    snapshot_pair_record(
                        pair,
                        metadata
                            .get(&key)
                            .filter(|metadata| snapshot_metadata_matches_pair(pair, metadata)),
                    )
                })
                .collect::<Vec<_>>();
            writer.write_chunk(
                &snapshot_path,
                verified_pair_snapshot::SharedPairChunk {
                    pair_order_hash: ordered_hash,
                    unordered_edge_hash: unordered_hash,
                    accepted_match_count: accepted as u64,
                    pairs: &records,
                },
            )?;
        } else {
            write_verified_pair_snapshot_atomic(
                &snapshot_path,
                image_names,
                features,
                camera,
                &pairwise,
                &metadata,
                args,
                feature_validation,
            )?;
        }
        let elapsed = started.elapsed().as_secs_f64();
        writeln!(
            stdout,
            "persistent-match-complete shard_id={} candidate_path={} snapshot_path={} candidate_sha256={} candidate_pairs={} pairs={} accepted={} ordered_edge_fnv1a64={ordered_hash:016x} unordered_edge_fnv1a64={unordered_hash:016x} elapsed_s={elapsed:.9}",
            shard.id,
            shard.candidate_path.display(),
            shard.snapshot_path.display(),
            shard.candidate_sha256,
            candidates.len(),
            pairwise.len(),
            accepted,
        )?;
        stdout.flush()?;
        // `pairwise`, metadata, and the candidate vector are all shard-local;
        // dropping them at this boundary keeps the worker's result buffers
        // bounded even when the feature bank is large.
        drop(_candidate_metadata);
        drop(candidates);
        drop(metadata);
        drop(pairwise);
        while stream_resident_rows > STREAM_DESCRIPTOR_CACHE_ROWS {
            let Some((stamp, image)) = stream_cache.pop_first() else {
                break;
            };
            if stream_stamps[image] != Some(stamp) {
                continue;
            }
            stream_resident_rows =
                stream_resident_rows.saturating_sub(features[image].descriptors.len());
            features[image].descriptors = Vec::new();
            stream_stamps[image] = None;
        }
        trim_process_allocator();
    }
    if let Some(writer) = &shared_writer {
        writer.finish()?;
    }
    if stream_sources.is_some() {
        writeln!(
            stdout,
            "persistent-match-stream-state cache_rows_cap={} peak_resident_rows={} peak_resident_images={} final_resident_rows={} final_resident_images={}",
            STREAM_DESCRIPTOR_CACHE_ROWS,
            stream_peak_resident_rows,
            stream_peak_resident_images,
            stream_resident_rows,
            stream_cache.len(),
        )?;
        stdout.flush()?;
    }
    Ok(())
}
