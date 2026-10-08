//! Command-line option enums, effective-config snapshot and argument parsing into `Args`.

use super::*;

/// A COLMAP-export landmark: world position + `(image, keypoint, pixel)` track.
pub(super) type ExportLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

/// The M1/M1.1 two-view verification A/B switch — see the file header and
/// `verify_pairs`'s doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VerificationMode {
    /// Essential-matrix-only RANSAC, legacy fixed `5e-3`-normalized Sampson
    /// threshold. The M1 "OFF" path, byte-identical to pre-M1 behaviour.
    Legacy,
    /// Essential-matrix-only RANSAC (same single-model estimator as
    /// `Legacy`), but with the per-camera pixel-derived Sampson threshold
    /// (`TwoViewGeometryOptions::for_camera`) instead of the fixed default.
    /// No fundamental/homography models, no `ConfigurationType`
    /// classification, no watermark detection. The M1.1 ablation mode.
    ThresholdOnly,
    /// Full COLMAP-style `TwoViewGeometryVerifier` (E/F/H + classification).
    /// The M1 "ON" path, byte-identical to pre-M1.1 `--colmap-verification`.
    Full,
}

impl std::str::FromStr for VerificationMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "legacy" => Ok(Self::Legacy),
            "threshold-only" => Ok(Self::ThresholdOnly),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "unknown --verification-mode {other:?} (expected legacy|threshold-only|full)"
            )),
        }
    }
}

/// The M3 pair-generation A/B switch (`docs/colmap_port_plan.md`'s "M3
/// results"): which candidate-pair source feeds two-view verification.
/// [`PairSource::Vlad`] (default) is the pre-M3 flat-VLAD top-K path,
/// unchanged (`candidate_pairs_vlad`, formerly this file's only
/// `candidate_pairs`). [`PairSource::VocabTree`] routes through
/// `visloc_rs::vision::vocab_tree`'s hierarchical-k-means +
/// TF-IDF/Hamming-embedding retrieval instead (COLMAP's
/// `VocabTreePairGenerator`-equivalent, `src/colmap/controllers/pairing.h`)
/// — see `candidate_pairs_vocab_tree`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PairSource {
    /// Flat-VLAD top-K cosine retrieval (pre-M3 behaviour, unchanged).
    Vlad,
    /// VLAD top-K pairs retained only when retrieval is mutual.
    VladMutual,
    /// Union of a bounded numeric-stem local schedule and flat-VLAD
    /// retrieval.  Local edges are retained first when a budget is applied.
    VladUnion,
    /// Rig-aware temporal-pyramid offsets, same-timestamp cross-camera edges,
    /// and a VLAD fill pass under an optional candidate budget.
    TemporalPyramid,
    /// Hierarchical-k-means vocab-tree retrieval (M3).
    VocabTree,
    /// COLMAP's `TransitivePairGenerator` port: propose pairs through the
    /// *verified-match* graph — images that share a matched partner but have
    /// no direct pair yet get proposed (`pairing.cc`). Runs a vocab-tree
    /// base pass, then expands transitively for
    /// [`TRANSITIVE_ROUNDS`] rounds.
    Transitive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RetrievalBackend {
    Exact,
    Lsh,
}

impl std::str::FromStr for RetrievalBackend {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "exact" => Ok(Self::Exact),
            "lsh" => Ok(Self::Lsh),
            other => Err(format!(
                "unknown --retrieval-backend {other:?} (expected exact|lsh)"
            )),
        }
    }
}

impl std::str::FromStr for PairSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vlad" => Ok(Self::Vlad),
            "vlad-mutual" => Ok(Self::VladMutual),
            "vlad-union" => Ok(Self::VladUnion),
            "temporal-pyramid" => Ok(Self::TemporalPyramid),
            "vocab-tree" => Ok(Self::VocabTree),
            "transitive" => Ok(Self::Transitive),
            other => Err(format!(
                "unknown --pair-source {other:?} (expected vlad|vlad-mutual|vlad-union|temporal-pyramid|vocab-tree|transitive)"
            )),
        }
    }
}

/// The M6 pair-*matching* A/B switch (`docs/colmap_port_plan.md`'s "M6
/// results"): which algorithm turns two images' descriptor sets into
/// candidate correspondences, **before** two-view geometric verification
/// ([`VerificationMode`]) ever runs. Orthogonal to [`VerificationMode`] and
/// [`PairSource`] — this only changes how a *given* candidate pair's raw
/// matches are produced, not which pairs are proposed or how they're
/// classified afterwards.
///
/// [`MatcherKind::Nn`] (default) is the pre-M6 nearest-neighbour + Lowe-ratio
/// path (`BruteForceMatcher`/`CrossCheckMatcher`), unchanged.
/// [`MatcherKind::LightGlue`] routes through
/// [`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`] — a
/// learned, *joint* matcher that attends over both images' descriptors
/// together (as opposed to NN+ratio's independent per-descriptor nearest-
/// neighbour search) — motivated directly by M5's diagnosis
/// (`docs/colmap_port_plan.md`'s "M5 results"): ETH3D `courtyard`'s
/// cross-component bridge pairs carry real but very sparse correspondence
/// signal that a per-descriptor ratio test cannot safely extract from a
/// repeated-texture scene (M5's own "naive rescue" experiment showed a
/// *classifier-passing* false-bridge failure mode from over-relaxing the
/// NN+ratio matcher — the concrete evidence that the matcher itself, not
/// just its threshold, needed to change). Requires the `onnx-inference`
/// feature; `--matcher lightglue` without it is a hard runtime error (see
/// `parse_args`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MatcherKind {
    /// Nearest-neighbour + Lowe-ratio test, optionally bidirectional
    /// cross-checked. Pre-M6 behaviour, unchanged.
    Nn,
    /// LightGlue (SuperPoint variant), run in-process via ONNX Runtime.
    LightGlue,
}

impl std::str::FromStr for MatcherKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "nn" => Ok(Self::Nn),
            "lightglue" => Ok(Self::LightGlue),
            other => Err(format!(
                "unknown --matcher {other:?} (expected nn|lightglue)"
            )),
        }
    }
}

/// Diagnostic ordering applied to the verified pair/match stream immediately
/// before the mapper consumes it.  The default preserves the historical
/// traversal exactly; the reverse variants only reorder existing entries and
/// never add, remove, or rewrite a correspondence index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum UnionTraversalOrder {
    #[default]
    Original,
    ReversePairs,
    ReverseMatches,
    ReverseBoth,
    /// Sort pair and correspondence traversal by a stable hash of the
    /// physical endpoint coordinates.  The seed makes independent replay
    /// orders possible without changing matching or verification.
    PhysicalHash(u64),
    /// Descending counterpart of [`Self::PhysicalHash`].
    PhysicalHashReverse(u64),
}

impl UnionTraversalOrder {
    pub(super) fn as_string(self) -> String {
        match self {
            Self::Original => "original".to_string(),
            Self::ReversePairs => "reverse-pairs".to_string(),
            Self::ReverseMatches => "reverse-matches".to_string(),
            Self::ReverseBoth => "reverse-both".to_string(),
            Self::PhysicalHash(seed) => format!("physical-hash:{seed}"),
            Self::PhysicalHashReverse(seed) => format!("physical-hash-reverse:{seed}"),
        }
    }
}

impl std::str::FromStr for UnionTraversalOrder {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "original" => Ok(Self::Original),
            "reverse-pairs" => Ok(Self::ReversePairs),
            "reverse-matches" => Ok(Self::ReverseMatches),
            "reverse-both" => Ok(Self::ReverseBoth),
            other => {
                let (kind, raw_seed) = if let Some(seed) = other.strip_prefix("physical-hash:") {
                    ("physical-hash", seed)
                } else if let Some(seed) = other.strip_prefix("physical-hash-reverse:") {
                    ("physical-hash-reverse", seed)
                } else {
                    return Err(format!(
                        "unknown --union-traversal-order {other:?} (expected original|reverse-pairs|reverse-matches|reverse-both|physical-hash:SEED|physical-hash-reverse:SEED)"
                    ));
                };
                let seed = if let Some(hex) = raw_seed
                    .strip_prefix("0x")
                    .or_else(|| raw_seed.strip_prefix("0X"))
                {
                    u64::from_str_radix(hex, 16)
                } else {
                    raw_seed.parse::<u64>()
                }
                .map_err(|_| {
                    format!(
                        "invalid {kind} seed {raw_seed:?}; use an unsigned decimal or 0x-prefixed hexadecimal integer"
                    )
                })?;
                Ok(if kind == "physical-hash" {
                    Self::PhysicalHash(seed)
                } else {
                    Self::PhysicalHashReverse(seed)
                })
            }
        }
    }
}

/// Serialize the complete parsed command line in declaration order.  Keeping
/// this as `Debug` output is intentional: unlike a map-based representation it
/// has a fixed field order, and it automatically includes newly added flags
/// once they are added to [`Args`].  The snapshot is diagnostic only and never
/// participates in reconstruction decisions.
pub(super) fn effective_config_snapshot(args: &Args) -> String {
    let orientation_cap = if args.sift_vlfeat_compatible_detector {
        if args.sift_max_orientations == 0 {
            "2 (compatible-mode default)".to_owned()
        } else {
            args.sift_max_orientations.min(4).to_string()
        }
    } else if args.sift_max_orientations == 0 {
        "unlimited (legacy mode)".to_owned()
    } else {
        args.sift_max_orientations.to_string()
    };
    let descriptor_magnification = if args.sift_vlfeat_compatible_descriptor {
        "3.0 (compatible-mode fixed)".to_owned()
    } else {
        args.sift_descriptor_magnification.to_string()
    };
    format!(
        "raw={args:?};effective_sift_orientation_cap={orientation_cap};\
         effective_sift_descriptor_magnification={descriptor_magnification}"
    )
}

/// Stable, dependency-free hash for the effective command-line snapshot.
/// `DefaultHasher` is deliberately avoided because its implementation is not
/// a public cross-version serialization contract.  FNV-1a is sufficient here:
/// this is a reproducibility label, not a cryptographic integrity check.
fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3u64);
    }
    hash
}

pub(super) fn effective_config_hash(snapshot: &str) -> u64 {
    fnv1a64_bytes(snapshot.as_bytes())
}

/// Validate the mutually-exclusive diagnostic output modes before loading
/// features, and optionally validate image indices once the image count is
/// known. Keeping this separate from `parse_args` makes the CLI contract
/// testable without synthesizing process arguments.
pub(super) fn validate_diagnose_options(
    pairs_csv: Option<&Path>,
    pair_stems: &[String],
    pairs: &[(usize, usize)],
    image_count: Option<usize>,
) -> Result<(), String> {
    if pairs_csv.is_none() && !pair_stems.is_empty() {
        return Err("--diagnose-pair-stems requires --diagnose-pairs-csv PATH".into());
    }
    if pairs_csv.is_some() && !pairs.is_empty() {
        return Err("--diagnose-pairs-csv and --diagnose-pair are mutually exclusive".into());
    }
    if let Some(n) = image_count {
        for &(i, j) in pairs {
            if i == j {
                return Err(format!(
                    "--diagnose-pair requires distinct image indices, got {i},{j}"
                ));
            }
            if i >= n || j >= n {
                return Err(format!(
                    "--diagnose-pair index out of range: {i},{j} for {n} images"
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn validate_diagnose_stems(
    image_names: &[String],
    stems: &[String],
) -> Result<(), String> {
    for stem in stems {
        if !image_names.iter().any(|name| image_stem(name) == stem) {
            return Err(format!(
                "--diagnose-pair-stems stem {stem:?} does not match a loaded image"
            ));
        }
    }
    Ok(())
}

fn parse_seed_pair(raw: &str) -> Result<(usize, usize), String> {
    let (left, right) = raw
        .split_once(',')
        .ok_or_else(|| format!("--seed-pair expects I,J, got {raw:?}"))?;
    let i: usize = left
        .trim()
        .parse()
        .map_err(|e| format!("--seed-pair invalid first index in {raw:?}: {e}"))?;
    let j: usize = right
        .trim()
        .parse()
        .map_err(|e| format!("--seed-pair invalid second index in {raw:?}: {e}"))?;
    if i == j {
        return Err(format!(
            "--seed-pair requires two distinct images, got {i},{j}"
        ));
    }
    Ok((i.min(j), i.max(j)))
}

pub(super) fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

pub(super) fn parse_args_from<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut features_dir = None;
    let mut feature_suffix = String::from("_features.txt");
    let mut image_suffix = String::from(".png");
    let mut out_colmap = None;
    let mut input_colmap_calibration: Option<PathBuf> = None;
    let (mut width, mut height) = (None, None);
    let (mut fx, mut fy, mut cx, mut cy) = (None, None, None, None);
    let mut vocab_size = 64usize;
    let mut retrieval_topk = 12usize;
    let mut exhaustive = false;
    let mut match_ratio = 0.8f32;
    let mut min_matches = 30usize;
    let mut min_pnp_inliers = 12usize;
    let mut max_mapper_matches_per_pair: Option<usize> = None;
    let mut max_reproj = 4.0f64;
    // The demo's no-flag workflow uses the robust two-stage policy; the public
    // library default remains CorrespondenceCount for API/snapshot identity.
    let mut next_image_policy = NextImagePolicy::Auto;
    let mut next_image_policy_explicit = false;
    let mut final_ba = true;
    let mut final_min_track_length: Option<usize> = None;
    let mut seed_trials = 12usize;
    let mut seed_attempts = 0usize;
    let mut seed_pair: Option<(usize, usize)> = None;
    let mut component_model_min_images: Option<usize> = None;
    let mut component_model_max_count = 16usize;
    let mut pair_stem_window: Option<u64> = None;
    let mut local_stem_window: Option<u64> = None;
    let mut rig_local_grouping = false;
    let mut rig_frame_manifest: Option<PathBuf> = None;
    let mut retrieval_component_manifest: Option<PathBuf> = None;
    let mut retrieval_min_frame_gap: Option<u64> = None;
    let mut temporal_pyramid_max_offset = 32u64;
    let mut candidate_budget: Option<usize> = None;
    let mut refine_intrinsics = false;
    let mut refine_distortion = false;
    let mut refine_tangential_distortion = false;
    let mut shared_focal = false;
    let mut colmap_style = false;
    let mut final_iterative_global_refinement = false;
    let mut global_ba_max_refinements: Option<usize> = None;
    let mut post_refinement_registration = false;
    let mut structureless_registration = false;
    let mut guided_matching = false;
    let mut colmap_guided_matching = false;
    let mut multiple_models = false;
    let mut min_e_f_inlier_ratio: Option<f64> = None;
    let mut calibrated_prefer_essential = false;
    let mut refine_uncalibrated_f_to_essential = false;
    let mut strict_uncalibrated_f_to_essential = false;
    let mut calibrated_essential_primary = false;
    let mut prefer_essential_inliers = false;
    let mut prefer_essential_free_endpoints = false;
    let mut prefer_essential_stems: Vec<String> = Vec::new();
    let mut prefer_essential_stem_clique = false;
    let mut prefer_essential_pairs: Vec<(usize, usize)> = Vec::new();
    let mut require_essential_selected_edges = false;
    let mut require_essential_stems: Vec<String> = Vec::new();
    let mut require_essential_min_e_inliers = 0usize;
    let mut rematch_stems: Vec<String> = Vec::new();
    let mut rematch_ratio = 0.9f32;
    let mut rematch_cross_check = true;
    let mut rematch_guided = false;
    let mut rematch_free_vs_priors = false;
    let mut rematch_prefer_min_e_inliers = 0usize;
    let mut rematch_prefer_strong_stems: Vec<String> = Vec::new();
    let mut rematch_prefer_strong_min_e = 50usize;
    let mut rematch_tracks_use_essential = false;
    let mut rematch_min_chirality_margin = 0.0f64;
    let mut rematch_prior_anchor = false;
    let mut rematch_anchor_min_e_inliers = 25usize;
    let mut rematch_min_e_f_inlier_ratio: Option<f64> = None;
    let mut rematch_calibrated_prefer_essential = false;
    let mut rematch_prior_ray_guided = false;
    let mut rematch_prior_ray_min_rays = 2usize;
    let mut rematch_prior_ray_min_e_inliers = 25usize;
    let mut rematch_verification_mode: Option<VerificationMode> = None;
    let mut rematch_pose_guided_after_global = false;
    let mut rematch_pose_guided_gt: Option<PathBuf> = None;
    let mut essential_edge_weight_boost = 1.0f64;
    let mut force_essential_matches = false;
    let mut force_essential_min_ef_ratio = 0.7f64;
    let mut force_essential_min_e_inliers = 0usize;
    let mut force_essential_uncalibrated_only = false;
    let mut repnp_free_from_priors = false;
    let mut repnp_free_min_corrs = 0usize;
    let mut repnp_seed_free_as_priors = false;
    let mut repair_prior_edges = false;
    let mut repair_free_edges_from_solved = false;
    let mut repair_free_edges_only_flipped = false;
    let mut repair_free_edges_stems: Vec<String> = Vec::new();
    let mut drop_free_edges_antipodal = false;
    let mut prior_guided_free_chirality = false;
    let mut metric_prior_chirality_edges = false;
    let mut metric_prior_chirality_min_rays = 3usize;
    let mut diagnose_bearing_gt: Option<PathBuf> = None;
    let mut diagnose_bearing_stems: Vec<String> = Vec::new();
    let mut gt_chirality_oracle = false;
    let mut gt_chirality_oracle_path: Option<PathBuf> = None;
    let mut rematch_max_gt_bearing_deg = 0.0f64;
    let mut rematch_gt_bearing_path: Option<PathBuf> = None;
    let mut rematch_guided_max_error_px: Option<f64> = None;
    let mut rematch_guided_lowe_ratio: Option<f64> = None;
    let mut rematch_require_calibrated = false;
    let mut rematch_max_mean_sampson = 0.0f64;
    let mut metric_prior_scale = false;
    let mut sequence_relative_pose_fallback = false;
    let mut sequence_fallback_after_post = false;
    let mut sequence_constant_velocity_scale = false;
    let mut sequence_relaxed_constant_velocity_scale = false;
    let mut sequence_fallback_carry_scale = false;
    let mut pnp_max_iterations = 128usize;
    let mut ba_max_iterations: Option<usize> = None;
    let mut ba_huber_delta: Option<f64> = None;
    let mut ba_linear_solver: Option<LinearSolver> = None;
    let mut matrix_free_ba = false;
    let mut periodic_ba_min_registered_images = 0usize;
    let mut final_ba_polish_iterations = 0usize;
    let mut geometry_weighted_ba = false;
    let mut freeze_ill_conditioned_landmarks = false;
    let mut landmark_ba_warm_start_iterations = 0usize;
    let mut landmark_ba_warm_start_min_registered_images = 0usize;
    let mut filter_images = false;
    let mut verification_mode = VerificationMode::Legacy;
    let mut track_source = TrackSource::UnionFind;
    let mut confidence_ordered_tracks = false;
    let mut geometric_confidence_tracks = false;
    let mut stable_track_order = false;
    let mut cycle_supported_tracks = false;
    let mut canonical_feature_order = false;
    let mut union_traversal_order = UnionTraversalOrder::Original;
    let mut geometry_guided_conflict_recovery = false;
    let mut pair_source = PairSource::Vlad;
    let mut vocab_tree_branching = 10usize;
    let mut vocab_tree_depth = 3usize;
    let mut vocab_tree_num_images = 100usize;
    let mut rescue_bridging = false;
    let mut rescue_match_ratio = 0.95f32;
    let mut rescue_min_matches = 15usize;
    let mut rescue_max_candidates = 200usize;
    let mut rescue_cross_check = false;
    let mut gpu_match = false;
    let mut gpu_sift = false;
    let mut gpu_ba = false;
    let mut diagnose_pairs: Vec<(usize, usize)> = Vec::new();
    let mut diagnose_pairs_csv: Option<PathBuf> = None;
    let mut diagnose_pair_stems: Vec<String> = Vec::new();
    let mut matcher = MatcherKind::Nn;
    let mut lightglue_model: Option<PathBuf> = None;
    let mut onnx_backend = String::from("auto");
    let mut lightglue_max_keypoints = 0usize;
    let mut import_matches_file: Option<PathBuf> = None;
    let mut import_matches_supplement_file: Option<PathBuf> = None;
    let mut export_features_dir: Option<PathBuf> = None;
    let mut export_features_only = false;
    let mut sift_stream_export = false;
    let mut sift_stream_resume = false;
    let mut import_verified_pairs_file: Option<PathBuf> = None;
    let mut export_verified_pairs_snapshot: Option<PathBuf> = None;
    let mut shared_snapshot_envelope = false;
    let mut import_verified_pairs_snapshot: Option<PathBuf> = None;
    let mut snapshot_keypoints_only = false;
    let mut export_verified_pairs_only = false;
    let mut persistent_match_worker_plan: Option<PathBuf> = None;
    let mut stream_match_features = false;
    let mut candidate_manifest: Option<PathBuf> = None;
    let mut export_candidate_manifest: Option<PathBuf> = None;
    let mut stream_candidate_features = false;
    let mut retrieval_backend = RetrievalBackend::Exact;
    let mut ann_tables = 8usize;
    let mut ann_bits = 0usize;
    let mut ann_probes = 6usize;
    let mut snapshot_coordinate_override_dir: Option<PathBuf> = None;
    let mut diagnose_ba_oracle_poses_file: Option<PathBuf> = None;
    let mut diagnose_fixed_rotation_ba: Option<String> = None;
    let mut diagnose_model_score_file: Option<PathBuf> = None;
    let mut initial_poses_file: Option<PathBuf> = None;
    let mut diagnose_colmap_track_membership: Option<PathBuf> = None;
    let mut pose_guided_track_splitting = false;
    let mut pose_guided_track_splitting_graph_support = false;
    let mut pose_guided_track_splitting_bridge_cuts = false;
    let mut pose_guided_split_max_reproj: Option<f64> = None;
    let mut pose_guided_track_splitting_iterations: Option<usize> = None;
    let mut pose_guided_track_merging = false;
    let mut pose_guided_merge_max_reproj: Option<f64> = None;
    let mut feature_extractor = FeatureExtractorKind::Files;
    let mut mapper = MapperKind::Incremental;
    let mut chirality_harden = false;
    let mut rotation_seed_trials = 1usize;
    let mut refine_global_translations = false;
    let mut global_independent_edge_scales = false;
    let mut multi_hypothesis_edges = false;
    let mut min_edge_inliers = 15usize;
    let mut min_edge_parallax_deg = 2.0f64;
    let mut weight_by_chirality_margin = false;
    let mut hybrid_filter_priors = false;
    let mut hybrid_prior_min_obs = 50usize;
    let mut hybrid_prior_max_reproj = 0.45f64;
    let mut hybrid_drop_prior_stems: Vec<String> = Vec::new();
    let mut hybrid_drop_inconsistent_priors = false;
    let mut verify_registration_two_view = false;
    let mut hybrid_rotation_priors_only = false;
    let mut joint_global_positioning = false;
    let mut calibrated_view_edges_only = false;
    let mut images_dir: Option<PathBuf> = None;
    let mut sift_max_keypoints = 2048usize;
    let mut sift_affine = false;
    let mut sift_detector = String::from("dog");
    let mut sift_multi_anisotropy = false;
    let mut sift_dsp = false;
    let mut sift_dsp_num_scales = 15usize;
    let mut sift_l1_root = false;
    let mut sift_max_orientations = 0usize;
    let mut sift_standard_orientations = false;
    let mut sift_prefer_larger_scale = false;
    let mut sift_full_pyramid = false;
    let mut sift_contrast_threshold = 0.02f64;
    let mut sift_descriptor_magnification = 8.0f64;
    let mut sift_descriptor_magnification_explicit = false;
    let mut sift_scale_adaptive_gradients = false;
    let mut sift_vlfeat_compatible_descriptor = false;
    let mut sift_vlfeat_compatible_detector = false;
    let mut sift_vlfeat_bilinear_orientations = false;
    let mut sift_vlfeat_compatible_output_order = false;
    let mut sift_colmap_compatible_grayscale = false;
    let mut sift_split_colmap_detector_grayscale = false;
    let mut sift_append_descriptor_magnification: Option<f64> = None;
    let mut sift_extra_keypoints_stems: Vec<String> = Vec::new();
    let mut sift_extra_keypoints = 0usize;
    let mut sift_extra_contrast_threshold: Option<f64> = None;
    let mut sift_extra_matches_append_only = false;
    let mut incremental_correspondence_triangulation = false;
    let mut orientation_locus_canonicalization = false;

    let mut a: Vec<String> = args.into_iter().collect();
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--features-dir" => features_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--hybrid-filter-priors" => hybrid_filter_priors = true,
            "--hybrid-prior-min-obs" => {
                hybrid_prior_min_obs = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--hybrid-prior-max-reproj" => {
                hybrid_prior_max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--hybrid-drop-prior-stems" => {
                hybrid_drop_prior_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();
            }
            "--hybrid-drop-inconsistent-priors" => hybrid_drop_inconsistent_priors = true,
            "--verify-registration-two-view" => verify_registration_two_view = true,
            "--hybrid-rotation-priors-only" => hybrid_rotation_priors_only = true,
            "--joint-global-positioning" => joint_global_positioning = true,
            "--calibrated-view-edges-only" => calibrated_view_edges_only = true,
            "--images-dir" => images_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--feature-extractor" => {
                feature_extractor = match a.remove(i + 1).as_str() {
                    "files" => FeatureExtractorKind::Files,
                    "sift" => FeatureExtractorKind::Sift,
                    other => {
                        return Err(format!(
                            "--feature-extractor must be files|sift, got {other}"
                        ))
                    }
                };
            }
            "--sift-max-keypoints" => {
                sift_max_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-affine" => sift_affine = true,
            "--sift-multi-anisotropy" => sift_multi_anisotropy = true,
            "--sift-dsp" => sift_dsp = true,
            "--sift-dsp-num-scales" => {
                sift_dsp_num_scales = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-l1-root" => sift_l1_root = true,
            "--sift-max-orientations" => {
                sift_max_orientations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-standard-orientations" => sift_standard_orientations = true,
            "--sift-prefer-larger-scale" => sift_prefer_larger_scale = true,
            "--sift-full-pyramid" => sift_full_pyramid = true,
            "--sift-contrast-threshold" => {
                sift_contrast_threshold = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-descriptor-magnification" => {
                sift_descriptor_magnification_explicit = true;
                sift_descriptor_magnification =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-scale-adaptive-gradients" => sift_scale_adaptive_gradients = true,
            "--sift-vlfeat-compatible-descriptor" => sift_vlfeat_compatible_descriptor = true,
            "--sift-vlfeat-compatible-detector" => sift_vlfeat_compatible_detector = true,
            "--sift-vlfeat-bilinear-orientations" => sift_vlfeat_bilinear_orientations = true,
            "--sift-vlfeat-compatible-output-order" => sift_vlfeat_compatible_output_order = true,
            "--sift-colmap-compatible-grayscale" => sift_colmap_compatible_grayscale = true,
            "--sift-split-colmap-detector-grayscale" => sift_split_colmap_detector_grayscale = true,
            "--sift-append-descriptor-magnification" => {
                let magnification: f64 = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if !magnification.is_finite() || magnification <= 0.0 {
                    return Err(format!(
                        "--sift-append-descriptor-magnification must be finite and > 0, got {magnification}"
                    ));
                }
                sift_append_descriptor_magnification = Some(magnification);
            }
            "--sift-extra-keypoints-stems" => {
                sift_extra_keypoints_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--sift-extra-keypoints" => {
                sift_extra_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--sift-extra-contrast-threshold" => {
                let threshold: f64 = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if !threshold.is_finite() || threshold < 0.0 {
                    return Err(format!(
                        "--sift-extra-contrast-threshold must be finite and >= 0, got {threshold}"
                    ));
                }
                sift_extra_contrast_threshold = Some(threshold);
            }
            "--sift-extra-matches-append-only" => sift_extra_matches_append_only = true,
            "--incremental-correspondence-triangulation" => {
                incremental_correspondence_triangulation = true
            }
            "--orientation-locus-canonicalization" => orientation_locus_canonicalization = true,
            "--sift-detector" => sift_detector = a.remove(i + 1),
            "--feature-suffix" => feature_suffix = a.remove(i + 1),
            "--image-suffix" => image_suffix = a.remove(i + 1),
            "--out-colmap" => out_colmap = Some(PathBuf::from(a.remove(i + 1))),
            "--input-colmap-calibration" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--input-colmap-calibration requires MODEL_DIR")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--input-colmap-calibration requires a non-empty MODEL_DIR".into());
                }
                a.remove(i + 1);
                input_colmap_calibration = Some(PathBuf::from(raw));
            }
            "--width" => width = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--height" => height = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--fx" => fx = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--fy" => fy = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--cx" => cx = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--cy" => cy = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?),
            "--vocab-size" => vocab_size = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--retrieval-topk" => {
                retrieval_topk = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--exhaustive" => exhaustive = true,
            "--pair-stem-window" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--pair-stem-window requires a positive integer N")?
                    .clone();
                let window: u64 = raw.parse().map_err(|error| {
                    format!("--pair-stem-window must be a positive integer: {error}")
                })?;
                if window == 0 {
                    return Err("--pair-stem-window must be at least 1".into());
                }
                a.remove(i + 1);
                pair_stem_window = Some(window);
            }
            "--local-stem-window" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--local-stem-window requires a positive integer N")?
                    .clone();
                let window: u64 = raw.parse().map_err(|error| {
                    format!("--local-stem-window must be a positive integer: {error}")
                })?;
                if window == 0 {
                    return Err("--local-stem-window must be at least 1".into());
                }
                a.remove(i + 1);
                local_stem_window = Some(window);
            }
            "--rig-local-grouping" => rig_local_grouping = true,
            "--rig-frame-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--rig-frame-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--rig-frame-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                rig_frame_manifest = Some(PathBuf::from(raw));
            }
            "--retrieval-component-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--retrieval-component-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--retrieval-component-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                retrieval_component_manifest = Some(PathBuf::from(raw));
            }
            "--retrieval-min-frame-gap" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--retrieval-min-frame-gap requires a positive integer")?
                    .clone();
                let gap: u64 = raw.parse().map_err(|error| {
                    format!("--retrieval-min-frame-gap must be a positive integer: {error}")
                })?;
                if gap == 0 {
                    return Err("--retrieval-min-frame-gap must be at least 1".into());
                }
                a.remove(i + 1);
                retrieval_min_frame_gap = Some(gap);
            }
            "--temporal-pyramid-max-offset" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--temporal-pyramid-max-offset requires a positive integer")?
                    .clone();
                let offset: u64 = raw.parse().map_err(|error| {
                    format!("--temporal-pyramid-max-offset must be a positive integer: {error}")
                })?;
                if offset == 0 {
                    return Err("--temporal-pyramid-max-offset must be at least 1".into());
                }
                a.remove(i + 1);
                temporal_pyramid_max_offset = offset;
            }
            "--candidate-budget" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--candidate-budget requires a positive integer")?
                    .clone();
                let budget: usize = raw.parse().map_err(|error| {
                    format!("--candidate-budget must be a positive integer: {error}")
                })?;
                if budget == 0 {
                    return Err("--candidate-budget must be at least 1".into());
                }
                a.remove(i + 1);
                candidate_budget = Some(budget);
            }
            "--match-ratio" => match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-matches" => min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-pnp-inliers" => {
                min_pnp_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--max-mapper-matches-per-pair" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--max-mapper-matches-per-pair requires a positive integer")?
                    .clone();
                let limit: usize = raw.parse().map_err(|error| {
                    format!("--max-mapper-matches-per-pair must be a positive integer: {error}")
                })?;
                if limit == 0 {
                    return Err("--max-mapper-matches-per-pair must be at least 1".into());
                }
                a.remove(i + 1);
                max_mapper_matches_per_pair = Some(limit);
            }
            "--max-reproj" => max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--next-image-policy" => {
                let value = a
                    .get(i + 1)
                    .ok_or("--next-image-policy requires auto, count, or visibility")?
                    .clone();
                next_image_policy_explicit = true;
                next_image_policy = match value.as_str() {
                    "auto" => NextImagePolicy::Auto,
                    "count" => NextImagePolicy::CorrespondenceCount,
                    "visibility" => NextImagePolicy::VisibilityPyramid,
                    other => {
                        return Err(format!(
                            "--next-image-policy must be auto, count, or visibility, got {other}"
                        ));
                    }
                };
                a.remove(i + 1);
            }
            "--no-final-ba" => final_ba = false,
            "--final-min-track-length" => {
                let value: usize = a
                    .get(i + 1)
                    .ok_or("--final-min-track-length requires 3")?
                    .parse()
                    .map_err(|error| {
                        format!("--final-min-track-length must be an integer: {error}")
                    })?;
                if value != 3 {
                    return Err(format!(
                        "--final-min-track-length currently supports only 3, got {value}"
                    ));
                }
                a.remove(i + 1);
                final_min_track_length = Some(value);
            }
            "--seed-trials" => seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--seed-attempts" => {
                seed_attempts = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--seed-pair" => seed_pair = Some(parse_seed_pair(&a.remove(i + 1))?),
            "--component-model-min-images" => {
                component_model_min_images = Some(
                    a.remove(i + 1)
                        .parse()
                        .map_err(|e| format!("--component-model-min-images: {e}"))?,
                )
            }
            "--component-model-max-count" => {
                component_model_max_count = a
                    .remove(i + 1)
                    .parse()
                    .map_err(|e| format!("--component-model-max-count: {e}"))?
            }
            "--refine-intrinsics" => refine_intrinsics = true,
            "--refine-distortion" => refine_distortion = true,
            "--refine-tangential-distortion" => refine_tangential_distortion = true,
            "--shared-focal" => shared_focal = true,
            "--colmap-style" => colmap_style = true,
            "--final-iterative-refinement" => final_iterative_global_refinement = true,
            "--global-ba-max-refinements" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--global-ba-max-refinements requires a non-negative integer")?
                    .clone();
                let refinements: usize = raw.parse().map_err(|error| {
                    format!("--global-ba-max-refinements must be a non-negative integer: {error}")
                })?;
                a.remove(i + 1);
                global_ba_max_refinements = Some(refinements);
            }
            "--post-refinement-registration" => post_refinement_registration = true,
            "--structureless-registration" => structureless_registration = true,
            "--sequence-relative-pose-fallback" => sequence_relative_pose_fallback = true,
            "--sequence-fallback-after-post" => sequence_fallback_after_post = true,
            "--sequence-constant-velocity-scale" => sequence_constant_velocity_scale = true,
            "--sequence-relaxed-constant-velocity-scale" => {
                sequence_relaxed_constant_velocity_scale = true
            }
            "--sequence-fallback-carry-scale" => sequence_fallback_carry_scale = true,
            "--guided-matching" => guided_matching = true,
            "--colmap-guided-matching" => colmap_guided_matching = true,
            "--multiple-models" => multiple_models = true,
            "--min-e-f-inlier-ratio" => {
                min_e_f_inlier_ratio = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--calibrated-prefer-essential" => calibrated_prefer_essential = true,
            "--refine-uncalibrated-f-to-essential" => refine_uncalibrated_f_to_essential = true,
            "--strict-uncalibrated-f-to-essential" => strict_uncalibrated_f_to_essential = true,
            "--calibrated-essential-primary" => calibrated_essential_primary = true,
            "--prefer-essential-inliers" => prefer_essential_inliers = true,
            "--prefer-essential-free-endpoints" => prefer_essential_free_endpoints = true,
            "--prefer-essential-stems" => {
                prefer_essential_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--prefer-essential-stem-clique" => prefer_essential_stem_clique = true,
            "--prefer-essential-pairs" => {
                let raw = a.remove(i + 1);
                prefer_essential_pairs = raw
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        let (l, r) = s.split_once('-').ok_or_else(|| {
                            format!("--prefer-essential-pairs expects I-J, got {s:?}")
                        })?;
                        Ok::<_, String>((
                            l.parse().map_err(|e| format!("{e}"))?,
                            r.parse().map_err(|e| format!("{e}"))?,
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--require-essential-selected-edges" => require_essential_selected_edges = true,
            "--require-essential-stems" => {
                require_essential_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--require-essential-min-e-inliers" => {
                require_essential_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-stems" => {
                rematch_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--rematch-ratio" => {
                rematch_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-no-cross-check" => rematch_cross_check = false,
            "--rematch-guided" => rematch_guided = true,
            "--rematch-free-vs-priors" => rematch_free_vs_priors = true,
            "--rematch-prefer-min-e-inliers" => {
                rematch_prefer_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prefer-strong-stems" => {
                rematch_prefer_strong_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--rematch-prefer-strong-min-e" => {
                rematch_prefer_strong_min_e = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-tracks-use-essential" => rematch_tracks_use_essential = true,
            "--rematch-min-chirality-margin" => {
                rematch_min_chirality_margin =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prior-anchor" => rematch_prior_anchor = true,
            "--rematch-anchor-min-e-inliers" => {
                rematch_anchor_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-min-e-f-inlier-ratio" => {
                rematch_min_e_f_inlier_ratio =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-calibrated-prefer-essential" => rematch_calibrated_prefer_essential = true,
            "--rematch-prior-ray-guided" => rematch_prior_ray_guided = true,
            "--rematch-prior-ray-min-rays" => {
                rematch_prior_ray_min_rays = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-prior-ray-min-e-inliers" => {
                rematch_prior_ray_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-verification-mode" => {
                rematch_verification_mode = Some(a.remove(i + 1).parse().map_err(|e: String| e)?)
            }
            "--rematch-pose-guided-after-global" => rematch_pose_guided_after_global = true,
            "--rematch-pose-guided-gt" => {
                rematch_pose_guided_gt = Some(PathBuf::from(a.remove(i + 1)));
                rematch_pose_guided_after_global = true;
            }
            "--essential-edge-weight-boost" => {
                essential_edge_weight_boost = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-matches" => force_essential_matches = true,
            "--force-essential-min-ef-ratio" => {
                force_essential_min_ef_ratio =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-min-e-inliers" => {
                force_essential_min_e_inliers =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--force-essential-uncalibrated-only" => force_essential_uncalibrated_only = true,
            "--repnp-free-from-priors" => repnp_free_from_priors = true,
            "--repnp-free-min-corrs" => {
                repnp_free_min_corrs = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--repnp-seed-free-as-priors" => repnp_seed_free_as_priors = true,
            "--repair-prior-edges" => repair_prior_edges = true,
            "--repair-free-edges-from-solved" => repair_free_edges_from_solved = true,
            "--repair-free-edges-only-flipped" => repair_free_edges_only_flipped = true,
            "--repair-free-edges-stems" => {
                repair_free_edges_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--drop-free-edges-antipodal" => drop_free_edges_antipodal = true,
            "--prior-guided-free-chirality" => prior_guided_free_chirality = true,
            "--metric-prior-chirality-edges" => metric_prior_chirality_edges = true,
            "--metric-prior-chirality-min-rays" => {
                metric_prior_chirality_min_rays =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--diagnose-bearing-gt" => {
                diagnose_bearing_gt = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--diagnose-bearing-stems" => {
                diagnose_bearing_stems = a
                    .remove(i + 1)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
            "--gt-chirality-oracle" => gt_chirality_oracle = true,
            "--gt-chirality-oracle-path" => {
                gt_chirality_oracle_path = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--rematch-max-gt-bearing-deg" => {
                rematch_max_gt_bearing_deg = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rematch-gt-bearing-path" => {
                rematch_gt_bearing_path = Some(PathBuf::from(a.remove(i + 1)));
            }
            "--rematch-guided-max-error-px" => {
                rematch_guided_max_error_px =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-guided-lowe-ratio" => {
                rematch_guided_lowe_ratio =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--rematch-require-calibrated" => rematch_require_calibrated = true,
            "--rematch-max-mean-sampson" => {
                rematch_max_mean_sampson = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--metric-prior-scale" => metric_prior_scale = true,
            "--pnp-max-iterations" => {
                pnp_max_iterations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--ba-max-iterations" => {
                let iterations: usize = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
                if iterations == 0 {
                    return Err("--ba-max-iterations must be at least 1".into());
                }
                ba_max_iterations = Some(iterations);
            }
            "--ba-huber-delta" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--ba-huber-delta requires a positive finite pixel value")?
                    .clone();
                let delta: f64 = raw
                    .parse()
                    .map_err(|error| format!("--ba-huber-delta must be numeric: {error}"))?;
                if !delta.is_finite() || delta <= 0.0 {
                    return Err("--ba-huber-delta must be a positive finite pixel value".into());
                }
                a.remove(i + 1);
                ba_huber_delta = Some(delta);
            }
            "--ba-linear-solver" => {
                let solver = a
                    .get(i + 1)
                    .ok_or("--ba-linear-solver requires dense or sparse")?
                    .as_str();
                let parsed = match solver {
                    "dense" => LinearSolver::Dense,
                    "sparse" => LinearSolver::Sparse,
                    other => {
                        return Err(format!(
                            "--ba-linear-solver must be dense or sparse, got {other:?}"
                        ))
                    }
                };
                a.remove(i + 1);
                ba_linear_solver = Some(parsed);
            }
            "--matrix-free-ba" => matrix_free_ba = true,
            "--periodic-ba-min-registered-images" => {
                periodic_ba_min_registered_images =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--final-ba-polish-iterations" => {
                final_ba_polish_iterations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--geometry-weighted-ba" => geometry_weighted_ba = true,
            "--freeze-ill-conditioned-landmarks" => freeze_ill_conditioned_landmarks = true,
            "--landmark-ba-warm-start-iterations" => {
                landmark_ba_warm_start_iterations =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--landmark-ba-warm-start-min-registered-images" => {
                landmark_ba_warm_start_min_registered_images =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?;
            }
            "--mapper" => {
                mapper = match a.remove(i + 1).as_str() {
                    "incremental" => MapperKind::Incremental,
                    "global" => MapperKind::Global,
                    "hybrid" => MapperKind::Hybrid,
                    other => {
                        return Err(format!(
                            "--mapper must be incremental|global|hybrid, got {other}"
                        ))
                    }
                };
            }
            "--chirality-harden" => chirality_harden = true,
            "--rotation-seed-trials" => {
                rotation_seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--refine-global-translations" => refine_global_translations = true,
            "--global-independent-edge-scales" => global_independent_edge_scales = true,
            "--multi-hypothesis-edges" => multi_hypothesis_edges = true,
            "--min-edge-inliers" => {
                min_edge_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--min-edge-parallax-deg" => {
                min_edge_parallax_deg = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--weight-by-chirality-margin" => weight_by_chirality_margin = true,
            "--filter-images" => filter_images = true,
            "--colmap-verification" => verification_mode = VerificationMode::Full,
            "--verification-mode" => {
                verification_mode = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--track-source" => track_source = parse_track_source(&a.remove(i + 1))?,
            "--confidence-ordered-tracks" => confidence_ordered_tracks = true,
            "--geometric-confidence-tracks" => geometric_confidence_tracks = true,
            "--stable-track-order" => stable_track_order = true,
            "--cycle-supported-tracks" => cycle_supported_tracks = true,
            "--canonical-feature-order" => canonical_feature_order = true,
            "--union-traversal-order" => {
                union_traversal_order = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--geometry-guided-conflict-recovery" => geometry_guided_conflict_recovery = true,
            "--pose-guided-track-splitting" => pose_guided_track_splitting = true,
            "--pose-guided-track-splitting-graph-support" => {
                pose_guided_track_splitting_graph_support = true
            }
            "--pose-guided-track-splitting-bridge-cuts" => {
                pose_guided_track_splitting_bridge_cuts = true
            }
            "--pose-guided-split-max-reproj" => {
                pose_guided_split_max_reproj =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pose-guided-track-splitting-iterations" => {
                pose_guided_track_splitting_iterations =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pose-guided-track-merging" => pose_guided_track_merging = true,
            "--pose-guided-merge-max-reproj" => {
                pose_guided_merge_max_reproj =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--pair-source" => pair_source = a.remove(i + 1).parse().map_err(|e: String| e)?,
            "--vocab-tree-branching" => {
                vocab_tree_branching = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--vocab-tree-depth" => {
                vocab_tree_depth = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--vocab-tree-num-images" => {
                vocab_tree_num_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-bridging" => rescue_bridging = true,
            "--rescue-match-ratio" => {
                rescue_match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-min-matches" => {
                rescue_min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-max-candidates" => {
                rescue_max_candidates = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--rescue-cross-check" => rescue_cross_check = true,
            "--gpu-match" => {
                if !cfg!(feature = "gpu") {
                    return Err("--gpu-match needs a build with --features gpu".into());
                }
                gpu_match = true
            }
            "--gpu-sift" => {
                if !cfg!(feature = "gpu") {
                    return Err("--gpu-sift needs a build with --features gpu".into());
                }
                gpu_sift = true
            }
            "--gpu-ba" => {
                if !cfg!(feature = "gpu") {
                    return Err("--gpu-ba needs a build with --features gpu".into());
                }
                gpu_ba = true
            }
            "--diagnose-pair" => {
                let raw = a.remove(i + 1);
                let (lhs, rhs) = raw
                    .split_once(',')
                    .ok_or_else(|| format!("--diagnose-pair expects I,J, got {raw:?}"))?;
                let i_idx: usize = lhs.trim().parse().map_err(|e| format!("{e}"))?;
                let j_idx: usize = rhs.trim().parse().map_err(|e| format!("{e}"))?;
                diagnose_pairs.push((i_idx, j_idx));
            }
            "--diagnose-pairs-csv" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-pairs-csv requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--diagnose-pairs-csv requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                diagnose_pairs_csv = Some(PathBuf::from(raw));
            }
            "--diagnose-pair-stems" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-pair-stems requires STEM[,STEM…]")?
                    .clone();
                a.remove(i + 1);
                diagnose_pair_stems = parse_diagnose_stems(&raw)?;
            }
            "--matcher" => matcher = a.remove(i + 1).parse().map_err(|e: String| e)?,
            "--lightglue-model" => lightglue_model = Some(PathBuf::from(a.remove(i + 1))),
            "--onnx-backend" => onnx_backend = a.remove(i + 1),
            "--lightglue-max-keypoints" => {
                lightglue_max_keypoints = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--import-matches-file" => import_matches_file = Some(PathBuf::from(a.remove(i + 1))),
            "--import-matches-supplement-file" => {
                import_matches_supplement_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--export-features-dir" => export_features_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--export-features-only" => export_features_only = true,
            "--sift-stream-export" => sift_stream_export = true,
            "--sift-stream-resume" => sift_stream_resume = true,
            "--import-verified-pairs-file" => {
                import_verified_pairs_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--export-verified-pairs-snapshot" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--export-verified-pairs-snapshot requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--export-verified-pairs-snapshot requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                export_verified_pairs_snapshot = Some(PathBuf::from(raw));
            }
            "--import-verified-pairs-snapshot" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--import-verified-pairs-snapshot requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--import-verified-pairs-snapshot requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                import_verified_pairs_snapshot = Some(PathBuf::from(raw));
            }
            "--shared-snapshot-envelope" => shared_snapshot_envelope = true,
            "--snapshot-keypoints-only" => snapshot_keypoints_only = true,
            "--export-verified-pairs-only" => export_verified_pairs_only = true,
            "--persistent-match-worker-plan" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--persistent-match-worker-plan requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--persistent-match-worker-plan requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                persistent_match_worker_plan = Some(PathBuf::from(raw));
            }
            "--stream-match-features" => stream_match_features = true,
            "--candidate-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--candidate-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--candidate-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                candidate_manifest = Some(PathBuf::from(raw));
            }
            "--export-candidate-manifest" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--export-candidate-manifest requires PATH")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--export-candidate-manifest requires a non-empty PATH".into());
                }
                a.remove(i + 1);
                export_candidate_manifest = Some(PathBuf::from(raw));
            }
            "--stream-candidate-features" => stream_candidate_features = true,
            "--retrieval-backend" => {
                retrieval_backend = a.remove(i + 1).parse()?;
            }
            "--ann-tables" => {
                ann_tables = a
                    .remove(i + 1)
                    .parse()
                    .map_err(|error| format!("{error}"))?
            }
            "--ann-bits" => {
                ann_bits = a
                    .remove(i + 1)
                    .parse()
                    .map_err(|error| format!("{error}"))?
            }
            "--ann-probes" => {
                ann_probes = a
                    .remove(i + 1)
                    .parse()
                    .map_err(|error| format!("{error}"))?
            }
            "--snapshot-coordinate-override-dir" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--snapshot-coordinate-override-dir requires DIR")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--snapshot-coordinate-override-dir requires a non-empty DIR".into(),
                    );
                }
                a.remove(i + 1);
                snapshot_coordinate_override_dir = Some(PathBuf::from(raw));
            }
            "--diagnose-ba-oracle-poses" => {
                diagnose_ba_oracle_poses_file = Some(PathBuf::from(a.remove(i + 1)))
            }
            "--diagnose-fixed-rotation-ba" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-fixed-rotation-ba requires current or MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--diagnose-fixed-rotation-ba requires a non-empty source".into());
                }
                a.remove(i + 1);
                diagnose_fixed_rotation_ba = Some(raw);
            }
            "--diagnose-model-score" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-model-score requires MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--diagnose-model-score requires a non-empty MODEL/images.txt".into(),
                    );
                }
                a.remove(i + 1);
                diagnose_model_score_file = Some(PathBuf::from(raw));
            }
            "--initial-poses" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--initial-poses requires MODEL/images.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err("--initial-poses requires a non-empty MODEL/images.txt".into());
                }
                a.remove(i + 1);
                initial_poses_file = Some(PathBuf::from(raw));
            }
            "--diagnose-colmap-track-membership" => {
                let raw = a
                    .get(i + 1)
                    .ok_or("--diagnose-colmap-track-membership requires MODEL/points3D.txt")?
                    .clone();
                if raw.trim().is_empty() {
                    return Err(
                        "--diagnose-colmap-track-membership requires a non-empty points3D.txt path"
                            .into(),
                    );
                }
                a.remove(i + 1);
                diagnose_colmap_track_membership = Some(PathBuf::from(raw));
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let camera = if input_colmap_calibration.is_some() {
        if width.is_some()
            || height.is_some()
            || fx.is_some()
            || fy.is_some()
            || cx.is_some()
            || cy.is_some()
        {
            return Err(
                "--input-colmap-calibration cannot be combined with --width/--height/--fx/--fy/--cx/--cy"
                    .into(),
            );
        }
        // The real reference camera is loaded after feature/image names are
        // known.  This finite placeholder keeps the parsed Args total and is
        // replaced before any matching, verification, or mapping work.
        Camera::pinhole(0, 1, 1, 1.0, 1.0, 0.0, 0.0)
    } else {
        let width = width.ok_or("--width is required")?;
        let height = height.ok_or("--height is required")?;
        Camera::pinhole(
            0,
            width,
            height,
            fx.ok_or("--fx is required")?,
            fy.ok_or("--fy is required")?,
            cx.ok_or("--cx is required")?,
            cy.ok_or("--cy is required")?,
        )
    };

    if !sift_descriptor_magnification.is_finite() || sift_descriptor_magnification <= 0.0 {
        return Err(format!(
            "--sift-descriptor-magnification must be finite and > 0, got {sift_descriptor_magnification}"
        ));
    }
    if colmap_guided_matching && !guided_matching {
        return Err(
            "--colmap-guided-matching requires --guided-matching so the guided pass is explicit"
                .into(),
        );
    }
    if sift_vlfeat_compatible_descriptor && sift_scale_adaptive_gradients {
        return Err(
            "--sift-vlfeat-compatible-descriptor already selects the complete scale-adaptive descriptor; remove --sift-scale-adaptive-gradients".into(),
        );
    }
    if sift_dsp && !sift_vlfeat_compatible_descriptor {
        return Err(
            "--sift-dsp requires --sift-vlfeat-compatible-descriptor so domain-size pooling uses the corrected VLFeat/COLMAP descriptor".into(),
        );
    }
    if sift_dsp && sift_dsp_num_scales == 0 {
        return Err("--sift-dsp requires at least one domain-size sample".into());
    }
    if sift_vlfeat_compatible_descriptor && sift_affine {
        return Err(
            "--sift-vlfeat-compatible-descriptor currently requires isotropic keypoints; remove --sift-affine".into(),
        );
    }
    if sift_vlfeat_compatible_detector && sift_affine {
        return Err(
            "--sift-vlfeat-compatible-detector currently requires isotropic keypoints; remove --sift-affine".into(),
        );
    }
    if sift_vlfeat_compatible_detector && sift_standard_orientations {
        return Err(
            "--sift-vlfeat-compatible-detector already selects VLFeat orientation peaks; remove --sift-standard-orientations".into(),
        );
    }
    if sift_vlfeat_bilinear_orientations && !sift_vlfeat_compatible_detector {
        return Err(
            "--sift-vlfeat-bilinear-orientations requires --sift-vlfeat-compatible-detector".into(),
        );
    }
    if sift_vlfeat_compatible_output_order && !sift_vlfeat_compatible_detector {
        return Err(
            "--sift-vlfeat-compatible-output-order requires --sift-vlfeat-compatible-detector"
                .into(),
        );
    }
    if sift_vlfeat_compatible_descriptor && sift_append_descriptor_magnification.is_some() {
        return Err(
            "--sift-vlfeat-compatible-descriptor cannot be combined with --sift-append-descriptor-magnification".into(),
        );
    }
    if sift_vlfeat_compatible_descriptor
        && sift_descriptor_magnification_explicit
        && (sift_descriptor_magnification - 3.0).abs() > 1e-12
    {
        return Err(
            "--sift-vlfeat-compatible-descriptor fixes descriptor magnification at 3.0; omit --sift-descriptor-magnification or use 3.0".into(),
        );
    }
    if sift_split_colmap_detector_grayscale
        && (!sift_vlfeat_compatible_detector || !sift_vlfeat_compatible_descriptor)
    {
        return Err(
            "--sift-split-colmap-detector-grayscale requires --sift-vlfeat-compatible-detector and --sift-vlfeat-compatible-descriptor".into(),
        );
    }
    if sift_split_colmap_detector_grayscale && sift_colmap_compatible_grayscale {
        return Err(
            "--sift-split-colmap-detector-grayscale cannot be combined with --sift-colmap-compatible-grayscale".into(),
        );
    }
    if initial_poses_file.is_some() && mapper != MapperKind::Incremental {
        return Err("--initial-poses currently requires --mapper incremental".into());
    }
    if global_independent_edge_scales && mapper != MapperKind::Global {
        return Err("--global-independent-edge-scales currently requires --mapper global".into());
    }
    if initial_poses_file.is_some() && seed_pair.is_some() {
        return Err("--initial-poses cannot be combined with --seed-pair".into());
    }
    if component_model_min_images.is_some() && seed_pair.is_some() {
        return Err("--component-model-min-images cannot be combined with --seed-pair".into());
    }
    if component_model_min_images.is_some() && initial_poses_file.is_some() {
        return Err("--component-model-min-images cannot be combined with --initial-poses".into());
    }
    if component_model_min_images == Some(0) {
        return Err("--component-model-min-images must be positive".into());
    }
    if component_model_max_count == 0 {
        return Err("--component-model-max-count must be positive".into());
    }
    if component_model_min_images.is_some() && mapper != MapperKind::Incremental {
        return Err("--component-model-min-images requires --mapper incremental".into());
    }
    if component_model_min_images.is_some()
        && (sequence_relative_pose_fallback
            || diagnose_colmap_track_membership.is_some()
            || diagnose_ba_oracle_poses_file.is_some()
            || diagnose_fixed_rotation_ba.is_some())
    {
        return Err(
            "--component-model-min-images cannot be combined with sequence fallback, oracle track membership, or BA diagnostic probes"
                .into(),
        );
    }
    if shared_focal && !(refine_intrinsics || refine_distortion) {
        return Err("--shared-focal requires --refine-intrinsics or --refine-distortion".into());
    }
    if refine_tangential_distortion && !refine_distortion {
        return Err("--refine-tangential-distortion requires --refine-distortion".into());
    }
    if input_colmap_calibration.is_some() && (refine_intrinsics || refine_distortion) {
        return Err(
            "--input-colmap-calibration currently keeps per-image PINHOLE intrinsics fixed; remove --refine-intrinsics/--refine-distortion"
                .into(),
        );
    }
    if sift_stream_export && feature_extractor != FeatureExtractorKind::Sift {
        return Err("--sift-stream-export requires --feature-extractor sift".into());
    }
    if sift_stream_export && export_features_dir.is_none() {
        return Err("--sift-stream-export requires --export-features-dir DIR".into());
    }
    if sift_stream_export && !export_features_only {
        return Err(
            "--sift-stream-export requires --export-features-only so extracted banks are not retained in memory"
                .into(),
        );
    }
    if sift_stream_resume && !sift_stream_export {
        return Err("--sift-stream-resume requires --sift-stream-export".into());
    }
    if export_verified_pairs_only && export_verified_pairs_snapshot.is_none() {
        return Err(
            "--export-verified-pairs-only requires --export-verified-pairs-snapshot PATH".into(),
        );
    }
    if export_verified_pairs_only && import_verified_pairs_snapshot.is_some() {
        return Err(
            "--export-verified-pairs-only cannot be combined with --import-verified-pairs-snapshot"
                .into(),
        );
    }
    if candidate_manifest.is_some() && export_candidate_manifest.is_some() {
        return Err(
            "--candidate-manifest and --export-candidate-manifest are mutually exclusive".into(),
        );
    }
    if local_stem_window.is_some() && pair_source != PairSource::VladUnion {
        return Err("--local-stem-window requires --pair-source vlad-union".into());
    }
    if rig_local_grouping && pair_source != PairSource::VladUnion {
        return Err("--rig-local-grouping requires --pair-source vlad-union".into());
    }
    if rig_frame_manifest.is_some() && pair_source != PairSource::TemporalPyramid {
        return Err("--rig-frame-manifest requires --pair-source temporal-pyramid".into());
    }
    if retrieval_component_manifest.is_some() && pair_source != PairSource::TemporalPyramid {
        return Err(
            "--retrieval-component-manifest requires --pair-source temporal-pyramid".into(),
        );
    }
    if retrieval_min_frame_gap.is_some() && pair_source != PairSource::TemporalPyramid {
        return Err("--retrieval-min-frame-gap requires --pair-source temporal-pyramid".into());
    }
    if pair_source == PairSource::TemporalPyramid
        && (local_stem_window.is_some() || rig_local_grouping)
    {
        return Err(
            "--pair-source temporal-pyramid owns rig grouping; do not combine it with --local-stem-window or --rig-local-grouping".into(),
        );
    }
    if pair_source == PairSource::VladUnion && local_stem_window.is_none() {
        return Err("--pair-source vlad-union requires --local-stem-window N".into());
    }
    if pair_source == PairSource::VladUnion && exhaustive {
        return Err("--pair-source vlad-union cannot be combined with --exhaustive".into());
    }
    if pair_source == PairSource::TemporalPyramid && exhaustive {
        return Err("--pair-source temporal-pyramid cannot be combined with --exhaustive".into());
    }
    if candidate_budget.is_some()
        && !matches!(
            pair_source,
            PairSource::VladUnion | PairSource::TemporalPyramid
        )
    {
        return Err(
            "--candidate-budget currently requires --pair-source vlad-union or temporal-pyramid"
                .into(),
        );
    }
    if pair_source == PairSource::VladUnion && pair_stem_window.is_some() {
        return Err(
            "--pair-source vlad-union uses --local-stem-window; do not combine it with --pair-stem-window".into(),
        );
    }
    if pair_source == PairSource::TemporalPyramid && pair_stem_window.is_some() {
        return Err(
            "--pair-source temporal-pyramid uses rig timestamps; do not combine it with --pair-stem-window".into(),
        );
    }
    if candidate_manifest.is_some()
        && (exhaustive
            || local_stem_window.is_some()
            || rig_local_grouping
            || rig_frame_manifest.is_some()
            || retrieval_component_manifest.is_some()
            || retrieval_min_frame_gap.is_some()
            || candidate_budget.is_some()
            || pair_stem_window.is_some()
            || pair_source == PairSource::Transitive)
    {
        return Err(
            "--candidate-manifest cannot be combined with generated candidate filters or transitive expansion".into(),
        );
    }
    if export_candidate_manifest.is_some()
        && (import_matches_file.is_some()
            || import_matches_supplement_file.is_some()
            || import_verified_pairs_file.is_some()
            || import_verified_pairs_snapshot.is_some())
    {
        return Err(
            "--export-candidate-manifest cannot be combined with imported pair streams".into(),
        );
    }
    if stream_candidate_features
        && (feature_extractor != FeatureExtractorKind::Files
            || export_candidate_manifest.is_none()
            || input_colmap_calibration.is_none()
            || pair_source != PairSource::TemporalPyramid)
    {
        return Err(
            "--stream-candidate-features currently requires --feature-extractor files, --input-colmap-calibration, --pair-source temporal-pyramid, and --export-candidate-manifest"
                .into(),
        );
    }
    if retrieval_backend != RetrievalBackend::Exact && !stream_candidate_features {
        return Err(
            "--retrieval-backend lsh currently requires --stream-candidate-features".into(),
        );
    }
    if ann_tables == 0
        || ann_bits > 63
        || ann_probes > 63
        || (ann_bits != 0 && ann_probes > ann_bits)
    {
        return Err(
            "ANN settings require --ann-tables >= 1, --ann-bits auto (0) or 1..=63, and --ann-probes <= the effective bit count"
                .into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && mapper != MapperKind::Incremental {
        return Err(
            "--diagnose-colmap-track-membership currently requires --mapper incremental".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some()
        && feature_extractor != FeatureExtractorKind::Files
    {
        return Err(
            "--diagnose-colmap-track-membership currently requires --feature-extractor files"
                .into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && initial_poses_file.is_some() {
        return Err(
            "--diagnose-colmap-track-membership cannot be combined with --initial-poses".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some() && colmap_style {
        return Err(
            "--diagnose-colmap-track-membership uses the plain incremental schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if diagnose_colmap_track_membership.is_some()
        && (incremental_correspondence_triangulation
            || confidence_ordered_tracks
            || geometric_confidence_tracks
            || stable_track_order
            || cycle_supported_tracks
            || canonical_feature_order)
    {
        return Err(
            "--diagnose-colmap-track-membership cannot be combined with an alternate track strategy/order".into(),
        );
    }
    if pose_guided_track_splitting && mapper != MapperKind::Incremental {
        return Err("--pose-guided-track-splitting currently requires --mapper incremental".into());
    }
    if pose_guided_track_splitting_graph_support && !pose_guided_track_splitting {
        return Err(
            "--pose-guided-track-splitting-graph-support requires --pose-guided-track-splitting"
                .into(),
        );
    }
    if pose_guided_track_splitting_bridge_cuts && !pose_guided_track_splitting {
        return Err(
            "--pose-guided-track-splitting-bridge-cuts requires --pose-guided-track-splitting"
                .into(),
        );
    }
    if pose_guided_split_max_reproj.is_some() && !pose_guided_track_splitting {
        return Err("--pose-guided-split-max-reproj requires --pose-guided-track-splitting".into());
    }
    if pose_guided_track_merging && !pose_guided_track_splitting {
        return Err("--pose-guided-track-merging requires --pose-guided-track-splitting".into());
    }
    if pose_guided_merge_max_reproj.is_some() && !pose_guided_track_merging {
        return Err("--pose-guided-merge-max-reproj requires --pose-guided-track-merging".into());
    }
    if let Some(value) = pose_guided_merge_max_reproj {
        if !value.is_finite() || value <= 0.0 {
            return Err("--pose-guided-merge-max-reproj must be finite and positive".into());
        }
    }
    if let Some(value) = pose_guided_split_max_reproj {
        if !value.is_finite() || value <= 0.0 {
            return Err("--pose-guided-split-max-reproj must be finite and positive".into());
        }
    }
    if final_min_track_length.is_some() && !final_ba {
        return Err("--final-min-track-length requires final bundle adjustment".into());
    }
    if let Some(iterations) = pose_guided_track_splitting_iterations {
        if !(1..=8).contains(&iterations) {
            return Err("--pose-guided-track-splitting-iterations must be between 1 and 8".into());
        }
        if !pose_guided_track_splitting {
            return Err(
                "--pose-guided-track-splitting-iterations requires --pose-guided-track-splitting"
                    .into(),
            );
        }
    }
    if pose_guided_track_splitting && colmap_style {
        return Err(
            "--pose-guided-track-splitting uses the plain incremental schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if pose_guided_track_splitting && diagnose_colmap_track_membership.is_some() {
        return Err(
            "--pose-guided-track-splitting cannot be combined with imported COLMAP track membership".into(),
        );
    }
    if pose_guided_track_splitting
        && (track_source != TrackSource::UnionFind
            || incremental_correspondence_triangulation
            || confidence_ordered_tracks
            || geometric_confidence_tracks
            || stable_track_order
            || cycle_supported_tracks
            || canonical_feature_order)
    {
        return Err(
            "--pose-guided-track-splitting requires the legacy union-find track builder without another track strategy flag".into(),
        );
    }
    if incremental_correspondence_triangulation && mapper != MapperKind::Incremental {
        return Err(
            "--incremental-correspondence-triangulation currently requires --mapper incremental"
                .into(),
        );
    }
    if incremental_correspondence_triangulation && colmap_style {
        return Err(
            "--incremental-correspondence-triangulation keeps the plain growth schedule and cannot be combined with --colmap-style".into(),
        );
    }
    if sequence_relative_pose_fallback && mapper != MapperKind::Incremental {
        return Err(
            "--sequence-relative-pose-fallback currently requires --mapper incremental".into(),
        );
    }
    if sequence_relative_pose_fallback && colmap_style {
        return Err(
            "--sequence-relative-pose-fallback currently requires the plain incremental schedule; remove --colmap-style".into(),
        );
    }
    if sequence_relative_pose_fallback && initial_poses_file.is_some() {
        return Err(
            "--sequence-relative-pose-fallback cannot be combined with --initial-poses".into(),
        );
    }
    if sequence_fallback_after_post && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-fallback-after-post requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_fallback_after_post && !post_refinement_registration {
        return Err(
            "--sequence-fallback-after-post requires --post-refinement-registration".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_fallback_after_post {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-fallback-after-post".into(),
        );
    }
    if sequence_fallback_carry_scale && !sequence_relaxed_constant_velocity_scale {
        return Err(
            "--sequence-fallback-carry-scale requires --sequence-relaxed-constant-velocity-scale"
                .into(),
        );
    }
    if sequence_constant_velocity_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-constant-velocity-scale requires --sequence-relative-pose-fallback".into(),
        );
    }
    if sequence_relaxed_constant_velocity_scale && !sequence_relative_pose_fallback {
        return Err(
            "--sequence-relaxed-constant-velocity-scale requires --sequence-relative-pose-fallback"
                .into(),
        );
    }
    if sequence_constant_velocity_scale && sequence_relaxed_constant_velocity_scale {
        return Err(
            "--sequence-constant-velocity-scale and --sequence-relaxed-constant-velocity-scale are mutually exclusive".into(),
        );
    }
    if import_verified_pairs_file.is_some() && import_verified_pairs_snapshot.is_some() {
        return Err(
            "--import-verified-pairs-file and --import-verified-pairs-snapshot are mutually exclusive"
                .into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some() && import_verified_pairs_snapshot.is_none() {
        return Err(
            "--snapshot-coordinate-override-dir requires --import-verified-pairs-snapshot".into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some()
        && feature_extractor != FeatureExtractorKind::Files
    {
        return Err(
            "--snapshot-coordinate-override-dir currently requires --feature-extractor files"
                .into(),
        );
    }
    if snapshot_coordinate_override_dir.is_some() && export_verified_pairs_snapshot.is_some() {
        return Err(
            "--snapshot-coordinate-override-dir cannot be combined with --export-verified-pairs-snapshot"
                .into(),
        );
    }
    if snapshot_keypoints_only {
        if import_verified_pairs_snapshot.is_none() {
            return Err(
                "--snapshot-keypoints-only requires --import-verified-pairs-snapshot PATH".into(),
            );
        }
        if feature_extractor != FeatureExtractorKind::Files {
            return Err(
                "--snapshot-keypoints-only currently requires --feature-extractor files".into(),
            );
        }
        if mapper != MapperKind::Incremental || colmap_style {
            return Err(
                "--snapshot-keypoints-only currently requires the plain incremental mapper (remove --mapper global|hybrid and --colmap-style)".into(),
            );
        }
        if snapshot_coordinate_override_dir.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --snapshot-coordinate-override-dir".into(),
            );
        }
        if export_features_dir.is_some() || export_features_only {
            return Err("--snapshot-keypoints-only cannot be combined with feature export".into());
        }
        if export_verified_pairs_snapshot.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --export-verified-pairs-snapshot".into(),
            );
        }
        if canonical_feature_order || orientation_locus_canonicalization {
            return Err(
                "--snapshot-keypoints-only cannot be combined with feature-order or orientation-locus canonicalization".into(),
            );
        }
        if diagnose_model_score_file.is_some() {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --diagnose-model-score".into(),
            );
        }
        if stable_track_order {
            return Err(
                "--snapshot-keypoints-only cannot be combined with --stable-track-order (descriptor tie-breaks require descriptor payloads)".into(),
            );
        }
    }
    if stream_match_features {
        if persistent_match_worker_plan.is_none() {
            return Err(
                "--stream-match-features requires --persistent-match-worker-plan PLAN".into(),
            );
        }
        if feature_extractor != FeatureExtractorKind::Files {
            return Err(
                "--stream-match-features currently requires --feature-extractor files".into(),
            );
        }
        if snapshot_keypoints_only {
            return Err(
                "--stream-match-features and --snapshot-keypoints-only are mutually exclusive"
                    .into(),
            );
        }
    }
    if import_verified_pairs_snapshot.is_some() {
        if import_matches_file.is_some() || import_matches_supplement_file.is_some() {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with raw match imports".into(),
            );
        }
        if pair_stem_window.is_some()
            || local_stem_window.is_some()
            || rig_local_grouping
            || rig_frame_manifest.is_some()
            || candidate_budget.is_some()
            || candidate_manifest.is_some()
            || export_candidate_manifest.is_some()
            || pair_source == PairSource::Transitive
        {
            return Err(
                "--import-verified-pairs-snapshot cannot generate, filter, or transitively expand pairs"
                    .into(),
            );
        }
        if !rematch_stems.is_empty()
            || rematch_free_vs_priors
            || rescue_bridging
            || orientation_locus_canonicalization
            || canonical_feature_order
            || union_traversal_order != UnionTraversalOrder::Original
        {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with pair-stream rematching, reordering, or canonicalization flags"
                    .into(),
            );
        }
        if !diagnose_pairs.is_empty() || diagnose_pairs_csv.is_some() {
            return Err(
                "--import-verified-pairs-snapshot cannot be combined with matching diagnostics"
                    .into(),
            );
        }
    }
    if import_verified_pairs_snapshot.is_some() && !next_image_policy_explicit {
        // Snapshot v1 records the historical Count-default replay semantics.
        // Keep an explicit user policy authoritative, but do not silently
        // change model bytes merely because the demo default is Auto.
        next_image_policy = NextImagePolicy::CorrespondenceCount;
    }
    validate_diagnose_options(
        diagnose_pairs_csv.as_deref(),
        &diagnose_pair_stems,
        &diagnose_pairs,
        None,
    )?;

    let parsed = Args {
        gpu_match,
        gpu_sift,
        gpu_ba,
        feature_extractor,
        features_dir: features_dir.unwrap_or_default(),
        hybrid_filter_priors,
        hybrid_prior_min_obs,
        hybrid_prior_max_reproj,
        hybrid_drop_prior_stems,
        hybrid_drop_inconsistent_priors,
        verify_registration_two_view,
        hybrid_rotation_priors_only,
        joint_global_positioning,
        calibrated_view_edges_only,
        images_dir,
        feature_suffix,
        image_suffix,
        out_colmap: out_colmap.ok_or("--out-colmap is required")?,
        input_colmap_calibration,
        camera,
        vocab_size,
        retrieval_topk,
        exhaustive,
        pair_stem_window,
        local_stem_window,
        rig_local_grouping,
        rig_frame_manifest,
        retrieval_component_manifest,
        retrieval_min_frame_gap,
        temporal_pyramid_max_offset,
        candidate_budget,
        match_ratio,
        min_matches,
        min_pnp_inliers,
        max_mapper_matches_per_pair,
        max_reproj,
        next_image_policy,
        final_ba,
        final_min_track_length,
        seed_trials,
        seed_attempts,
        seed_pair,
        component_model_min_images,
        component_model_max_count,
        refine_intrinsics,
        refine_distortion,
        refine_tangential_distortion,
        shared_focal,
        colmap_style,
        final_iterative_global_refinement,
        global_ba_max_refinements,
        post_refinement_registration,
        structureless_registration,
        pnp_max_iterations,
        ba_max_iterations,
        ba_huber_delta,
        ba_linear_solver,
        matrix_free_ba,
        periodic_ba_min_registered_images,
        final_ba_polish_iterations,
        geometry_weighted_ba,
        freeze_ill_conditioned_landmarks,
        landmark_ba_warm_start_iterations,
        landmark_ba_warm_start_min_registered_images,
        mapper,
        chirality_harden,
        rotation_seed_trials,
        refine_global_translations,
        global_independent_edge_scales,
        multi_hypothesis_edges,
        min_edge_inliers,
        min_edge_parallax_deg,
        weight_by_chirality_margin,
        filter_images,
        verification_mode,
        guided_matching,
        colmap_guided_matching,
        multiple_models,
        min_e_f_inlier_ratio,
        calibrated_prefer_essential,
        refine_uncalibrated_f_to_essential,
        strict_uncalibrated_f_to_essential,
        calibrated_essential_primary,
        prefer_essential_inliers,
        prefer_essential_free_endpoints,
        prefer_essential_stems,
        prefer_essential_stem_clique,
        prefer_essential_pairs,
        require_essential_selected_edges,
        require_essential_stems,
        require_essential_min_e_inliers,
        rematch_stems,
        rematch_ratio,
        rematch_cross_check,
        rematch_guided,
        rematch_free_vs_priors,
        rematch_prefer_min_e_inliers,
        rematch_prefer_strong_stems,
        rematch_prefer_strong_min_e,
        rematch_tracks_use_essential,
        rematch_min_chirality_margin,
        rematch_prior_anchor,
        rematch_anchor_min_e_inliers,
        rematch_min_e_f_inlier_ratio,
        rematch_calibrated_prefer_essential,
        rematch_prior_ray_guided,
        rematch_prior_ray_min_rays,
        rematch_prior_ray_min_e_inliers,
        rematch_verification_mode,
        rematch_pose_guided_after_global,
        rematch_pose_guided_gt,
        essential_edge_weight_boost,
        force_essential_matches,
        force_essential_min_ef_ratio,
        force_essential_min_e_inliers,
        force_essential_uncalibrated_only,
        repnp_free_from_priors,
        repnp_free_min_corrs,
        repnp_seed_free_as_priors,
        repair_prior_edges,
        repair_free_edges_from_solved,
        repair_free_edges_only_flipped,
        repair_free_edges_stems,
        drop_free_edges_antipodal,
        prior_guided_free_chirality,
        metric_prior_chirality_edges,
        metric_prior_chirality_min_rays,
        diagnose_bearing_gt,
        diagnose_bearing_stems,
        gt_chirality_oracle,
        gt_chirality_oracle_path,
        rematch_max_gt_bearing_deg,
        rematch_gt_bearing_path,
        rematch_guided_max_error_px,
        rematch_guided_lowe_ratio,
        rematch_require_calibrated,
        rematch_max_mean_sampson,
        metric_prior_scale,
        sequence_relative_pose_fallback,
        sequence_fallback_after_post,
        sequence_constant_velocity_scale,
        sequence_relaxed_constant_velocity_scale,
        sequence_fallback_carry_scale,
        track_source,
        confidence_ordered_tracks,
        geometric_confidence_tracks,
        stable_track_order,
        cycle_supported_tracks,
        canonical_feature_order,
        union_traversal_order,
        geometry_guided_conflict_recovery,
        pose_guided_track_splitting,
        pose_guided_track_splitting_graph_support,
        pose_guided_track_splitting_bridge_cuts,
        pose_guided_split_max_reproj,
        pose_guided_track_splitting_iterations,
        pose_guided_track_merging,
        pose_guided_merge_max_reproj,
        pair_source,
        candidate_manifest,
        export_candidate_manifest,
        stream_candidate_features,
        retrieval_backend,
        ann_tables,
        ann_bits,
        ann_probes,
        vocab_tree_branching,
        vocab_tree_depth,
        vocab_tree_num_images,
        rescue_bridging,
        rescue_match_ratio,
        rescue_min_matches,
        rescue_max_candidates,
        rescue_cross_check,
        diagnose_pairs,
        diagnose_pairs_csv,
        diagnose_pair_stems,
        matcher,
        lightglue_model,
        onnx_backend,
        lightglue_max_keypoints,
        sift_max_keypoints,
        sift_affine,
        sift_detector,
        sift_multi_anisotropy,
        sift_dsp,
        sift_dsp_num_scales,
        sift_l1_root,
        sift_max_orientations,
        sift_standard_orientations,
        sift_prefer_larger_scale,
        sift_full_pyramid,
        sift_contrast_threshold,
        sift_descriptor_magnification,
        sift_scale_adaptive_gradients,
        sift_vlfeat_compatible_descriptor,
        sift_vlfeat_compatible_detector,
        sift_vlfeat_bilinear_orientations,
        sift_vlfeat_compatible_output_order,
        sift_colmap_compatible_grayscale,
        sift_split_colmap_detector_grayscale,
        sift_append_descriptor_magnification,
        sift_extra_keypoints_stems,
        sift_extra_keypoints,
        sift_extra_contrast_threshold,
        sift_extra_matches_append_only,
        incremental_correspondence_triangulation,
        orientation_locus_canonicalization,
        import_matches_file,
        import_matches_supplement_file,
        export_features_dir,
        export_features_only,
        sift_stream_export,
        sift_stream_resume,
        import_verified_pairs_file,
        export_verified_pairs_snapshot,
        shared_snapshot_envelope,
        import_verified_pairs_snapshot,
        snapshot_keypoints_only,
        export_verified_pairs_only,
        persistent_match_worker_plan,
        stream_match_features,
        snapshot_coordinate_override_dir,
        diagnose_ba_oracle_poses_file,
        diagnose_fixed_rotation_ba,
        diagnose_model_score_file,
        initial_poses_file,
        diagnose_colmap_track_membership,
    };
    validate_persistent_match_worker_args(&parsed)?;
    Ok(parsed)
}
