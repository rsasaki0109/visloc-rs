//! Pair matching, two-view verification, rematching and rescue bridging.

use super::*;

/// Per-`ConfigurationType` pair counts from the COLMAP-style verifier, for
/// the M1 acceptance experiment's pair-rejection report (how many VLAD
/// candidate pairs got reclassified away from a naive essential-matrix
/// accept). Unused (stays all-zero) when `--colmap-verification` is off.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct VerificationStats {
    pub(super) calibrated: usize,
    pub(super) uncalibrated: usize,
    pub(super) planar: usize,
    pub(super) panoramic: usize,
    pub(super) planar_or_panoramic: usize,
    pub(super) watermark: usize,
    pub(super) degenerate: usize,
    pub(super) multiple: usize,
    /// Pairs whose primary match set was swapped to E inliers.
    pub(super) force_essential_swaps: usize,
    /// F-winning uncalibrated pairs whose accepted matches were replaced by
    /// the guarded calibrated F→E refinement.
    pub(super) uncalibrated_f_to_essential_refinements: usize,
    /// F-winning uncalibrated pairs rejected by the opt-in strict strategy
    /// instead of falling back to their F inliers.
    pub(super) strict_uncalibrated_f_to_essential_exclusions: usize,
    /// Accepted F inliers removed by the strict strategy.
    pub(super) strict_uncalibrated_f_to_essential_excluded_inliers: usize,
    /// F-winning pairs promoted to a direct calibrated-essential track model.
    pub(super) calibrated_essential_primary_promotions: usize,
}

impl VerificationStats {
    pub(super) const fn record(&mut self, config: ConfigurationType) {
        match config {
            ConfigurationType::Calibrated => self.calibrated += 1,
            ConfigurationType::Uncalibrated => self.uncalibrated += 1,
            ConfigurationType::Planar => self.planar += 1,
            ConfigurationType::Panoramic => self.panoramic += 1,
            ConfigurationType::PlanarOrPanoramic => self.planar_or_panoramic += 1,
            ConfigurationType::Watermark => self.watermark += 1,
            ConfigurationType::Degenerate => self.degenerate += 1,
            ConfigurationType::Multiple => self.multiple += 1,
            ConfigurationType::Undefined => {}
        }
    }

    pub(super) const fn merge(&mut self, other: &VerificationStats) {
        self.calibrated += other.calibrated;
        self.uncalibrated += other.uncalibrated;
        self.planar += other.planar;
        self.panoramic += other.panoramic;
        self.planar_or_panoramic += other.planar_or_panoramic;
        self.watermark += other.watermark;
        self.degenerate += other.degenerate;
        self.multiple += other.multiple;
        self.force_essential_swaps += other.force_essential_swaps;
        self.uncalibrated_f_to_essential_refinements +=
            other.uncalibrated_f_to_essential_refinements;
        self.strict_uncalibrated_f_to_essential_exclusions +=
            other.strict_uncalibrated_f_to_essential_exclusions;
        self.strict_uncalibrated_f_to_essential_excluded_inliers +=
            other.strict_uncalibrated_f_to_essential_excluded_inliers;
        self.calibrated_essential_primary_promotions +=
            other.calibrated_essential_primary_promotions;
    }

    pub(super) const fn total(&self) -> usize {
        self.calibrated
            + self.uncalibrated
            + self.planar
            + self.panoramic
            + self.planar_or_panoramic
            + self.watermark
            + self.degenerate
            + self.multiple
    }
}

/// The M6 pair-matching backend, dispatched on [`MatcherKind`]. Holds the
/// loaded LightGlue ONNX session (cheap to `Clone`: it wraps an
/// `Arc<Mutex<ort::session::Session>>`, same as
/// [`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`]'s
/// own doc comment explains) so it can be shared across the rayon-parallel
/// per-pair closures in [`verify_pairs`] and [`rescue_bridging`] without
/// re-loading the model.
pub(super) enum PairMatcher {
    /// Pre-M6 nearest-neighbour + Lowe-ratio matcher.
    Nn,
    /// NN matcher that preserves the matches obtained from each image's
    /// primary feature prefix before appending non-conflicting extra matches.
    NnAppendOnly { primary_keypoint_counts: Vec<usize> },
    /// NN matcher with an alternate descriptor bank for the same keypoint
    /// indices. Alternate matches are append-only and cannot replace the
    /// primary result.
    NnDescriptorEnsemble {
        primary_keypoint_counts: Option<Vec<usize>>,
        alternate_descriptors: Vec<Option<Vec<Vec<f32>>>>,
    },
    /// LightGlue (SuperPoint variant), in-process via ONNX Runtime.
    /// `max_keypoints` truncates each side to a score-sorted prefix (`0` = all).
    #[cfg(feature = "onnx-inference")]
    LightGlue {
        matcher: LightGlueOnnxMatcher,
        max_keypoints: usize,
    },
}

impl PairMatcher {
    /// Raw descriptor matches for one candidate pair `(features_i,
    /// features_j)`. `ratio`/`cross_check` are [`MatcherKind::Nn`]-only
    /// knobs (Lowe ratio test / bidirectional mutual-NN confirmation); they
    /// are silently ignored under [`MatcherKind::LightGlue`], which has no
    /// equivalent parameters of its own — LightGlue's matching decision is
    /// the learned assignment-matrix + `filter_threshold` cut baked into the
    /// exported ONNX graph (see `scripts/export_lightglue_onnx.py`), not a
    /// per-descriptor ratio the caller can tune. This is a deliberate M6
    /// design choice (see the file header and `docs/colmap_port_plan.md`'s
    /// "M6 results"): LightGlue *replaces* the NN+ratio matcher rather than
    /// taking its knobs as a compatibility shim.
    pub(super) fn match_pair(
        &self,
        ratio: f32,
        cross_check: bool,
        image_i: usize,
        image_j: usize,
        features_i: &FeatureSet,
        features_j: &FeatureSet,
    ) -> Vec<DescriptorMatch> {
        match self {
            PairMatcher::Nn => nn_matches(
                ratio,
                cross_check,
                &features_i.descriptors,
                &features_j.descriptors,
            ),
            PairMatcher::NnAppendOnly {
                primary_keypoint_counts,
            } => {
                let primary_i = primary_keypoint_counts
                    .get(image_i)
                    .copied()
                    .unwrap_or(features_i.descriptors.len());
                let primary_j = primary_keypoint_counts
                    .get(image_j)
                    .copied()
                    .unwrap_or(features_j.descriptors.len());
                append_only_nn_matches(
                    ratio,
                    cross_check,
                    features_i,
                    features_j,
                    primary_i,
                    primary_j,
                )
            }
            PairMatcher::NnDescriptorEnsemble {
                primary_keypoint_counts,
                alternate_descriptors,
            } => {
                let baseline = if let Some(counts) = primary_keypoint_counts {
                    let primary_i = counts
                        .get(image_i)
                        .copied()
                        .unwrap_or(features_i.descriptors.len());
                    let primary_j = counts
                        .get(image_j)
                        .copied()
                        .unwrap_or(features_j.descriptors.len());
                    append_only_nn_matches(
                        ratio,
                        cross_check,
                        features_i,
                        features_j,
                        primary_i,
                        primary_j,
                    )
                } else {
                    nn_matches(
                        ratio,
                        cross_check,
                        &features_i.descriptors,
                        &features_j.descriptors,
                    )
                };
                let Some(alternate_i) = alternate_descriptors
                    .get(image_i)
                    .and_then(|descriptors| descriptors.as_ref())
                else {
                    return baseline;
                };
                let Some(alternate_j) = alternate_descriptors
                    .get(image_j)
                    .and_then(|descriptors| descriptors.as_ref())
                else {
                    return baseline;
                };
                assert_eq!(
                    alternate_i.len(),
                    features_i.descriptors.len(),
                    "alternate descriptor indices must match image-i keypoints"
                );
                assert_eq!(
                    alternate_j.len(),
                    features_j.descriptors.len(),
                    "alternate descriptor indices must match image-j keypoints"
                );
                let alternate = nn_matches(ratio, cross_check, alternate_i, alternate_j);
                append_nonconflicting_matches(baseline, alternate)
            }
            #[cfg(feature = "onnx-inference")]
            PairMatcher::LightGlue {
                matcher,
                max_keypoints,
            } => {
                let (kp_i, desc_i) = truncate_features(features_i, *max_keypoints);
                let (kp_j, desc_j) = truncate_features(features_j, *max_keypoints);
                match matcher.match_features(kp_i, desc_i, kp_j, desc_j) {
                    Ok(matches) => matches
                        .into_iter()
                        .map(|m| DescriptorMatch {
                            query_index: m.query_index,
                            train_index: m.train_index,
                            // LightGlue's assignment matrix has no notion of an
                            // L2 descriptor "distance" the way NN+ratio does —
                            // its own `score` (the assignment-matrix confidence)
                            // is carried in `confidence` instead, which is what
                            // every downstream consumer here actually reads.
                            // `distance = 1.0 - score` keeps this field
                            // orderable (lower = better) for any generic caller
                            // that still sorts on it, without claiming a false
                            // Euclidean-distance semantics.
                            distance: 1.0 - m.score,
                            second_best_distance: None,
                            ratio: None,
                            confidence: Some(m.score),
                        })
                        .collect(),
                    Err(error) => {
                        eprintln!("lightglue match error (treated as zero matches for this pair): {error}");
                        Vec::new()
                    }
                }
            }
        }
    }
}

/// Run the legacy NN+ratio matcher on descriptor slices, preserving the
/// exact cross-check and tie-breaking behavior used when append-only mode is
/// disabled.
/// GPU matcher shared by every [`verify_pairs`] call (`--gpu-match`).
#[cfg(feature = "gpu")]
pub(super) static GPU_NN: std::sync::OnceLock<(
    visloc_sift_gpu::GpuContext,
    visloc_sift_gpu::GpuMatcher,
)> = std::sync::OnceLock::new();

/// GPU SIFT extractor shared by every image (`--gpu-sift`); images still
/// decode in parallel, extraction serialises on the one device.
#[cfg(feature = "gpu")]
pub(super) static GPU_SIFT: std::sync::OnceLock<std::sync::Mutex<visloc_sift_gpu::SiftGpu>> =
    std::sync::OnceLock::new();

/// `extract_sift`, on the GPU when `--gpu-sift` is on and supports `config`.
#[cfg(feature = "image-io")]
pub(super) fn extract_sift_maybe_gpu(
    image: &visloc_rs::vision::features::sift::GrayImage<'_>,
    config: &visloc_rs::vision::features::sift::SiftConfig,
) -> Result<
    (
        Vec<visloc_rs::vision::features::sift::SiftKeypoint>,
        Vec<Vec<f32>>,
    ),
    Box<dyn std::error::Error>,
> {
    #[cfg(feature = "gpu")]
    if let Some(gpu) = GPU_SIFT.get() {
        if visloc_sift_gpu::SiftGpu::supports(config) {
            let mut gpu = gpu.lock().map_err(|_| "gpu sift mutex poisoned")?;
            return Ok(gpu.extract(image, config)?);
        }
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!("gpu-sift: configuration unsupported on the GPU; using the CPU extractor")
        });
    }
    Ok(visloc_rs::vision::features::sift::extract_sift(
        image, config,
    )?)
}

/// Batched GPU equivalent of [`nn_matches`] over `pairs`, when `--gpu-match`
/// is on and the descriptors fit the GPU bank. `None` = use the CPU path.
#[cfg(feature = "gpu")]
fn gpu_nn_bank(features: &[FeatureSet]) -> Option<visloc_sift_gpu::FeatureBank> {
    let (ctx, _) = GPU_NN.get()?;
    let sets: Vec<&[Vec<f32>]> = features.iter().map(|f| f.descriptors.as_slice()).collect();
    match visloc_sift_gpu::FeatureBank::upload(ctx, &sets) {
        Ok(bank) => Some(bank),
        Err(e) => {
            eprintln!("gpu-match: falling back to the CPU matcher ({e})");
            None
        }
    }
}

fn nn_matches(
    ratio: f32,
    cross_check: bool,
    query: &[Vec<f32>],
    train: &[Vec<f32>],
) -> Vec<DescriptorMatch> {
    if cross_check {
        BruteForceMatcher { ratio: Some(ratio) }.match_descriptors_cross_checked(query, train)
    } else {
        BruteForceMatcher { ratio: Some(ratio) }.match_descriptors(query, train)
    }
}

/// Preserve the primary-prefix NN matches exactly, then append only matches
/// from the full descriptor set that involve at least one extra descriptor
/// and do not reuse a primary query or train endpoint. The full matcher is
/// still used to rank extra candidates, but it can never replace a primary
/// match whose Lowe decision changed after extras were appended.
pub(super) fn append_only_nn_matches(
    ratio: f32,
    cross_check: bool,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    primary_i: usize,
    primary_j: usize,
) -> Vec<DescriptorMatch> {
    let primary_i = primary_i.min(features_i.descriptors.len());
    let primary_j = primary_j.min(features_j.descriptors.len());
    if primary_i == features_i.descriptors.len() && primary_j == features_j.descriptors.len() {
        return nn_matches(
            ratio,
            cross_check,
            &features_i.descriptors,
            &features_j.descriptors,
        );
    }
    let primary = nn_matches(
        ratio,
        cross_check,
        &features_i.descriptors[..primary_i],
        &features_j.descriptors[..primary_j],
    );
    let full = nn_matches(
        ratio,
        cross_check,
        &features_i.descriptors,
        &features_j.descriptors,
    );

    let mut used_queries = HashSet::new();
    let mut used_trains = HashSet::new();
    let mut seen_pairs = HashSet::new();
    for m in &primary {
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        seen_pairs.insert((m.query_index, m.train_index));
    }

    let mut out = primary;
    for m in full {
        // A match entirely inside the primary prefix is represented by the
        // preserved baseline result, even if the full set picked another
        // primary-to-primary neighbour after extras were added.
        if m.query_index < primary_i && m.train_index < primary_j {
            continue;
        }
        if used_queries.contains(&m.query_index) || used_trains.contains(&m.train_index) {
            continue;
        }
        if !seen_pairs.insert((m.query_index, m.train_index)) {
            continue;
        }
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        out.push(m);
    }
    out
}

/// Append alternate-bank matches without reusing either endpoint claimed by
/// the baseline. Both banks share the original keypoint-index space, so this
/// only augments correspondences and never creates duplicate track nodes.
fn append_nonconflicting_matches(
    baseline: Vec<DescriptorMatch>,
    alternate: Vec<DescriptorMatch>,
) -> Vec<DescriptorMatch> {
    let mut used_queries = HashSet::new();
    let mut used_trains = HashSet::new();
    for m in &baseline {
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
    }
    let mut out = baseline;
    for m in alternate {
        if used_queries.contains(&m.query_index) || used_trains.contains(&m.train_index) {
            continue;
        }
        used_queries.insert(m.query_index);
        used_trains.insert(m.train_index);
        out.push(m);
    }
    out
}

/// Score-sorted prefix for LightGlue (external SuperPoint dumps are already
/// descending by score). `0` or oversized caps keep the full set.
#[cfg(feature = "onnx-inference")]
fn truncate_features(features: &FeatureSet, max_keypoints: usize) -> (&[Point2<f64>], &[Vec<f32>]) {
    let n = features.keypoints.len();
    let take = if max_keypoints == 0 || max_keypoints >= n {
        n
    } else {
        max_keypoints
    };
    (&features.keypoints[..take], &features.descriptors[..take])
}

/// Build the [`PairMatcher`] `--matcher` selects. Fails fast (before any
/// pair is processed) if `--matcher lightglue` is requested without either
/// the `onnx-inference` feature compiled in or a `--lightglue-model` path.
pub(super) fn build_matcher(
    args: &Args,
    primary_keypoint_counts: &[usize],
    alternate_descriptors: Vec<Option<Vec<Vec<f32>>>>,
) -> Result<PairMatcher, Box<dyn std::error::Error>> {
    if args.sift_append_descriptor_magnification.is_some() {
        if matches!(args.matcher, MatcherKind::LightGlue) {
            return Err(
                "--sift-append-descriptor-magnification requires --matcher nn (LightGlue has no NN descriptor bank)".into(),
            );
        }
        if alternate_descriptors.len() != primary_keypoint_counts.len()
            || alternate_descriptors.iter().any(Option::is_none)
        {
            return Err(
                "--sift-append-descriptor-magnification requires --feature-extractor sift".into(),
            );
        }
        return Ok(PairMatcher::NnDescriptorEnsemble {
            primary_keypoint_counts: args
                .sift_extra_matches_append_only
                .then(|| primary_keypoint_counts.to_vec()),
            alternate_descriptors,
        });
    }
    match args.matcher {
        MatcherKind::Nn if args.sift_extra_matches_append_only => Ok(PairMatcher::NnAppendOnly {
            primary_keypoint_counts: primary_keypoint_counts.to_vec(),
        }),
        MatcherKind::Nn => Ok(PairMatcher::Nn),
        MatcherKind::LightGlue => {
            if args.sift_extra_matches_append_only {
                return Err(
                    "--sift-extra-matches-append-only requires --matcher nn (LightGlue has no NN prefix)".into(),
                );
            }
            #[cfg(feature = "onnx-inference")]
            {
                let path = args
                    .lightglue_model
                    .as_ref()
                    .ok_or("--matcher lightglue requires --lightglue-model PATH")?;
                let backend = match args.onnx_backend.as_str() {
                    "auto" => OnnxBackend::CudaThenCpu,
                    "cuda" => OnnxBackend::Cuda,
                    "cpu" => OnnxBackend::Cpu,
                    other => {
                        return Err(format!(
                            "unknown --onnx-backend {other:?} (expected auto|cuda|cpu)"
                        )
                        .into())
                    }
                };
                eprintln!(
                    "loading LightGlue ONNX from {path:?} (backend={:?})…",
                    backend
                );
                let matcher = LightGlueOnnxMatcher::load_from_path_with_backend(path, backend)
                    .map_err(|error| {
                        format!("failed to load LightGlue ONNX model {path:?}: {error}")
                    })?;
                eprintln!("LightGlue ONNX loaded");
                Ok(PairMatcher::LightGlue {
                    matcher,
                    max_keypoints: args.lightglue_max_keypoints,
                })
            }
            #[cfg(not(feature = "onnx-inference"))]
            {
                Err(
                    "--matcher lightglue requires rebuilding with --features onnx-inference \
                     (see docs/colmap_port_plan.md's M6 results)"
                        .into(),
                )
            }
        }
    }
}

/// Match and geometrically verify each candidate pair into `PairwiseMatches`.
/// Candidate pairs are independent, so the (descriptor-matching dominated) loop
/// is run across cores with rayon.
///
/// `mode` is the M1/M1.1 A/B switch:
/// - [`VerificationMode::Legacy`] (default) reproduces the exact pre-M1
///   essential-matrix-only path byte-for-byte (same estimator, same fixed
///   `5e-3` threshold, same call, same acceptance test) — the "flag off means
///   unchanged behaviour" guarantee `docs/colmap_port_plan.md` asks for.
/// - [`VerificationMode::ThresholdOnly`] runs the *same* single-model
///   essential-matrix-only estimator, but with the per-camera pixel-derived
///   Sampson threshold — isolates the "tighter threshold" half of the M1
///   confound from the "E/F/H classification" half (M1.1).
/// - [`VerificationMode::Full`] goes through [`TwoViewGeometryVerifier`]
///   instead: only `DEGENERATE` and `WATERMARK` pairs are dropped rather than
///   handed to `incremental_sfm` — COLMAP's own admission gate
///   (`database_cache.cc`'s `UseInlierMatchesCheck`) keeps everything else,
///   including `PANORAMIC` (pure rotation — no baseline to triangulate from)
///   and unresolved `PLANAR_OR_PANORAMIC` (M2.1 parity fix; see
///   `docs/colmap_port_plan.md`'s "M2.1 results" — previously this demo
///   dropped both, stricter than real COLMAP). `CALIBRATED` / `UNCALIBRATED`
///   / `PLANAR` / `PANORAMIC` / `PLANAR_OR_PANORAMIC` / `MULTIPLE` pairs all
///   keep their winning model's own inliers (which need not be the essential
///   matrix's); a `PANORAMIC`/`PLANAR_OR_PANORAMIC` pair's correspondences
///   can still help track connectivity and BA even though the pair itself
///   can never become a seed (`incremental_sfm`'s parallax gate at growth
///   time excludes near-zero-baseline pairs independently of this
///   classification, mirroring how COLMAP's own init-pair search
///   recomputes and gates on triangulation angle rather than consulting the
///   stored `ConfigurationType`).
///
/// Re-match pairs incident to `stems` at a looser Lowe ratio; keep the new
/// geometry when it exposes more essential inliers (hub densification without
/// changing the global keypoint set).
#[allow(clippy::too_many_arguments)]
pub(super) fn rematch_stem_pairs(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut [PairwiseMatches],
    camera: &Camera,
    stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    force_essential_min_ef_ratio: f64,
    force_essential_min_e_inliers: usize,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
) -> usize {
    let want: HashSet<&str> = stems.iter().map(String::as_str).collect();
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let targets: Vec<(usize, usize)> = pairwise
        .iter()
        .filter(|p| want.contains(stem_of(p.image_i)) || want.contains(stem_of(p.image_j)))
        .map(|p| (p.image_i, p.image_j))
        .collect();
    if targets.is_empty() || mode != VerificationMode::Full {
        return 0;
    }
    let (fresh, _, _) = verify_pairs(
        features,
        camera,
        &targets,
        rematch_ratio,
        min_matches,
        mode,
        matcher,
        rematch_cross_check,
        guided_matching,
        multiple_models,
        min_e_f_inlier_ratio,
        calibrated_prefer_essential,
        false, // F→E refinement is only enabled on the main configured pass
        false, // strict F→E exclusion is only enabled on the main configured pass
        false, // calibrated-essential promotion is only enabled on the main pass
        false, // do not force-swap primary matches on rematch
        force_essential_min_ef_ratio,
        force_essential_min_e_inliers,
        false,
        guided_max_error_px,
        guided_lowe_ratio,
        None,
        None,
        false,
    );
    let mut improved = 0usize;
    for new in fresh {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        if let Some(old) = pairwise
            .iter_mut()
            .find(|p| (p.image_i.min(p.image_j), p.image_i.max(p.image_j)) == key)
        {
            let old_e = old.essential_matches.as_ref().map_or(0, |e| e.len());
            let new_e = new.essential_matches.as_ref().map_or(0, |e| e.len());
            if new_e > old_e || (old_e == 0 && new_e >= min_matches) {
                let name = |idx: usize| {
                    Path::new(&image_names[idx])
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or(image_names[idx].as_str())
                };
                eprintln!(
                    "rematch: improved {}-{} E {} -> {} (inliers {} -> {})",
                    name(new.image_i),
                    name(new.image_j),
                    old_e,
                    new_e,
                    old.matches.len(),
                    new.matches.len()
                );
                *old = new;
                improved += 1;
            }
        }
    }
    improved
}

/// Single-pair pose-guided verify (COLMAP FindGuidedMatches under known E).
fn verify_pose_guided_pair(
    features: &[FeatureSet],
    camera: &Camera,
    i: usize,
    j: usize,
    pose_i: &Pose,
    pose_j: &Pose,
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    matcher: &PairMatcher,
    guided_max_error_px: f64,
    guided_lowe_ratio: f64,
    calibrated_prefer_essential: bool,
) -> Option<PairwiseMatches> {
    let pose_e = essential_from_absolute_poses(pose_i, pose_j)?;
    let dm0 = matcher.match_pair(
        rematch_ratio,
        rematch_cross_check,
        i,
        j,
        &features[i],
        &features[j],
    );
    let extra = guided_epipolar_matches(
        camera,
        &features[i],
        &features[j],
        &dm0,
        &[],
        guided_max_error_px,
        Some(pose_e),
        guided_lowe_ratio,
    );
    let mut dm = dm0;
    dm.extend(extra);
    if dm.len() < min_matches {
        return None;
    }
    let corrs: Vec<TwoViewCorrespondence> = dm
        .iter()
        .map(|m| {
            TwoViewCorrespondence::new(
                features[i].keypoints[m.query_index],
                features[j].keypoints[m.train_index],
            )
        })
        .collect();
    let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
    opts.calibrated_prefer_essential = calibrated_prefer_essential;
    let verifier = TwoViewGeometryVerifier::new(opts);
    let report = verifier.classify(&corrs, camera);
    let keep = matches!(
        report.config,
        ConfigurationType::Calibrated
            | ConfigurationType::Uncalibrated
            | ConfigurationType::Planar
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
            | ConfigurationType::Multiple
    );
    if !keep || report.inliers.len() < min_matches {
        return None;
    }
    let essential_matches = if report.essential_inliers.len() >= min_matches {
        Some(
            report
                .essential_inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let matches: Vec<(usize, usize)> = report
        .inliers
        .iter()
        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
        .collect();
    Some(PairwiseMatches {
        image_i: i,
        image_j: j,
        matches,
        two_view_config: Some(report.config),
        essential_matches,
        essential_matrix: report.essential,
    })
}

/// Post-incremental: rematch free hub stems only against cameras that have
/// pose priors — so densification targets prior↔hub bridges (the courtyard
/// unlock), not free–free pairs like `0297–0298`.
#[allow(clippy::too_many_arguments)]
pub(super) fn rematch_free_against_priors(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut Vec<PairwiseMatches>,
    camera: &Camera,
    pose_priors: &[Option<Pose>],
    free_stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    _force_essential_min_ef_ratio: f64,
    _force_essential_min_e_inliers: usize,
    tracks_use_essential: bool,
    min_chirality_margin: f64,
    require_prior_anchor: bool,
    anchor_min_e_inliers: usize,
    gt_by_stem: Option<&HashMap<String, Pose>>,
    max_gt_bearing_deg: f64,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
    require_calibrated: bool,
    max_mean_sampson: f64,
    prior_ray_guided: bool,
    prior_ray_min_rays: usize,
    prior_ray_min_e_inliers: usize,
    rematch_verification_mode: Option<VerificationMode>,
    pair_stem_window: Option<u64>,
) -> (usize, Vec<((usize, usize), usize)>) {
    let rematch_mode = rematch_verification_mode.unwrap_or(mode);
    match rematch_mode {
        VerificationMode::Full | VerificationMode::ThresholdOnly => {}
        VerificationMode::Legacy => return (0, Vec::new()),
    }
    let prior_idx: HashSet<usize> = pose_priors
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if prior_idx.is_empty() {
        return (0, Vec::new());
    }
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let free_want: HashSet<&str> = free_stems.iter().map(String::as_str).collect();
    let free_idx: HashSet<usize> = (0..features.len())
        .filter(|&i| {
            if prior_idx.contains(&i) {
                return false;
            }
            if free_want.is_empty() {
                true
            } else {
                free_want.contains(stem_of(i))
            }
        })
        .collect();
    if free_idx.is_empty() {
        return (0, Vec::new());
    }
    // All free×prior pairs (exhaustive within the bipartite cut) — includes
    // pairs the main pass never admitted, which is the point for bridges.
    let mut targets: Vec<(usize, usize)> = Vec::new();
    for &f in &free_idx {
        for &p in &prior_idx {
            targets.push((f.min(p), f.max(p)));
        }
    }
    targets.sort_unstable();
    targets.dedup();
    if let Some(window) = pair_stem_window {
        let stem_values = numeric_stem_values(image_names)
            .expect("pair stem window was validated before rematch-free-vs-priors");
        targets.retain(|&pair| {
            pair_within_stem_window(pair, &stem_values, window)
                .expect("rematch pair indices are loaded image indices")
        });
    }
    eprintln!(
        "rematch-free-vs-priors: {} free × {} prior → {} candidate pairs (ratio={:.2}, guided={})",
        free_idx.len(),
        prior_idx.len(),
        targets.len(),
        rematch_ratio,
        guided_matching
    );
    let guided_max_px = guided_max_error_px.unwrap_or(2.0);
    let guided_lowe = guided_lowe_ratio.unwrap_or(0.8);
    let fresh = if prior_ray_guided {
        let free_poses = estimate_free_poses_from_prior_rays(
            pairwise,
            features,
            camera,
            pose_priors,
            prior_ray_min_rays,
            prior_ray_min_e_inliers,
        );
        eprintln!(
            "rematch prior-ray-guided: {} free pose(s) from incremental rays (min_rays={}, min_e={})",
            free_poses.len(),
            prior_ray_min_rays,
            prior_ray_min_e_inliers
        );
        let pose_at = |idx: usize| -> Option<Pose> {
            pose_priors
                .get(idx)
                .and_then(|p| p.as_ref())
                .cloned()
                .or_else(|| free_poses.get(&idx).cloned())
        };
        let mut out = Vec::new();
        let mut std_targets: Vec<(usize, usize)> = Vec::new();
        let mut guided_attempts = 0usize;
        for &(i, j) in &targets {
            let (Some(pi), Some(pj)) = (pose_at(i), pose_at(j)) else {
                std_targets.push((i, j));
                continue;
            };
            guided_attempts += 1;
            if let Some(pm) = verify_pose_guided_pair(
                features,
                camera,
                i,
                j,
                &pi,
                &pj,
                rematch_ratio,
                rematch_cross_check,
                min_matches,
                matcher,
                guided_max_px,
                guided_lowe,
                calibrated_prefer_essential,
            ) {
                out.push(pm);
            } else {
                std_targets.push((i, j));
            }
        }
        eprintln!(
            "rematch prior-ray-guided: pose-guided {}/{} pair(s) ok; {} fallback to standard verify",
            out.len(),
            guided_attempts,
            std_targets.len()
        );
        if !std_targets.is_empty() {
            let (more, _, _) = verify_pairs(
                features,
                camera,
                &std_targets,
                rematch_ratio,
                min_matches,
                rematch_mode,
                matcher,
                rematch_cross_check,
                guided_matching,
                multiple_models,
                min_e_f_inlier_ratio,
                calibrated_prefer_essential,
                false,
                false,
                false,
                false,
                0.0,
                0,
                false,
                guided_max_error_px,
                guided_lowe_ratio,
                None,
                None,
                false,
            );
            out.extend(more);
        }
        out
    } else {
        verify_pairs(
            features,
            camera,
            &targets,
            rematch_ratio,
            min_matches,
            rematch_mode,
            matcher,
            rematch_cross_check,
            guided_matching,
            multiple_models,
            min_e_f_inlier_ratio,
            calibrated_prefer_essential,
            false,
            false,
            false,
            false,
            0.0,
            0,
            false,
            guided_max_error_px,
            guided_lowe_ratio,
            None,
            None,
            false,
        )
        .0
    };
    let mut changed = 0usize;
    let mut gained_e_pairs: Vec<((usize, usize), usize)> = Vec::new();
    let mut rejected = 0usize;
    let mut existing: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, p) in pairwise.iter().enumerate() {
        existing.insert((p.image_i.min(p.image_j), p.image_i.max(p.image_j)), idx);
    }
    for new in fresh {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        let new_e = new.essential_matches.as_ref().map_or(0, |e| e.len());
        let name = |idx: usize| stem_of(idx);
        let mut new = new;
        if tracks_use_essential {
            if let Some(ess) = new.essential_matches.clone() {
                if ess.len() >= min_matches {
                    new.matches = ess;
                }
            }
        }
        let (prior_cam, free_cam) = if prior_idx.contains(&new.image_i) {
            (new.image_i, new.image_j)
        } else {
            (new.image_j, new.image_i)
        };
        if !rematch_essential_admission_ok(
            &new,
            prior_cam,
            free_cam,
            features,
            camera,
            pose_priors,
            pairwise,
            min_chirality_margin,
            require_prior_anchor,
            anchor_min_e_inliers,
        ) {
            rejected += 1;
            continue;
        }
        if require_calibrated
            && !matches!(
                new.two_view_config,
                Some(visloc_vision::two_view::ConfigurationType::Calibrated)
            )
        {
            eprintln!(
                "rematch-free-vs-priors: reject {}-{} config={:?} (require Calibrated)",
                name(prior_cam),
                name(free_cam),
                new.two_view_config
            );
            rejected += 1;
            continue;
        }
        if max_mean_sampson > 0.0 {
            if let Some(ms) = pair_essential_mean_sampson_error(&new, features, camera) {
                if ms > max_mean_sampson {
                    eprintln!(
                        "rematch-free-vs-priors: Sampson reject {}-{} mean={:.5} > {:.5}",
                        name(prior_cam),
                        name(free_cam),
                        ms,
                        max_mean_sampson
                    );
                    rejected += 1;
                    continue;
                }
            }
        }
        if max_gt_bearing_deg > 0.0 {
            if let Some(gt) = gt_by_stem {
                let stem_of = |idx: usize| -> Option<&str> {
                    Path::new(&image_names[idx])
                        .file_stem()
                        .and_then(|s| s.to_str())
                };
                if let (Some(ps), Some(fs)) = (stem_of(prior_cam), stem_of(free_cam)) {
                    if let (Some(gt_p), Some(gt_f)) = (gt.get(ps), gt.get(fs)) {
                        if let Some(err) = prior_free_essential_gt_bearing_error_deg(
                            &new, prior_cam, free_cam, features, camera, gt_p, gt_f,
                        ) {
                            if err > max_gt_bearing_deg {
                                eprintln!(
                                    "rematch-free-vs-priors: GT-bearing reject {}-{} err={:.1}° > {:.1}°",
                                    ps, fs, err, max_gt_bearing_deg
                                );
                                rejected += 1;
                                continue;
                            }
                        }
                    }
                }
            }
        }
        if let Some(&idx) = existing.get(&key) {
            let old = &pairwise[idx];
            let old_e = old.essential_matches.as_ref().map_or(0, |e| e.len());
            // Prior↔hub unlock needs *essential* support; F-only densification
            // (E=0) poisons tracks without a calibrated bridge.
            if new_e > old_e {
                eprintln!(
                    "rematch-free-vs-priors: improved {}-{} E {} -> {} config={:?} (inliers {} -> {}, tracks_e={})",
                    name(new.image_i),
                    name(new.image_j),
                    old_e,
                    new_e,
                    new.two_view_config,
                    old.matches.len(),
                    new.matches.len(),
                    tracks_use_essential
                );
                pairwise[idx] = new;
                gained_e_pairs.push((key, new_e));
                changed += 1;
            }
        } else if new_e >= min_matches {
            eprintln!(
                "rematch-free-vs-priors: new bridge {}-{} E={} config={:?} inliers={} tracks_e={}",
                name(new.image_i),
                name(new.image_j),
                new_e,
                new.two_view_config,
                new.matches.len(),
                tracks_use_essential
            );
            existing.insert(key, pairwise.len());
            pairwise.push(new);
            gained_e_pairs.push((key, new_e));
            changed += 1;
        }
    }
    if rejected > 0 {
        eprintln!(
            "rematch-free-vs-priors: rejected {rejected} E-gain(s) (margin>={min_chirality_margin:.2}, prior_anchor={require_prior_anchor})"
        );
    }
    (changed, gained_e_pairs)
}

/// Post-global free↔prior rematch seeded by absolute-pose essentials.
/// For each free×prior pair where both cameras registered, expand NN matches
/// under the pose-derived E Sampson gate, re-verify, and keep pairs whose
/// essential inlier count rises (same accept gate as pre-global rematch).
pub(super) fn rematch_pose_guided_free_vs_priors(
    features: &[FeatureSet],
    image_names: &[String],
    pairwise: &mut Vec<PairwiseMatches>,
    camera: &Camera,
    poses: &[Option<Pose>],
    pose_priors: &[Option<Pose>],
    // When set, replace guidance poses by stem (GT oracle); reconstruction
    // poses stay untouched.
    guidance_poses_by_stem: Option<&HashMap<String, Pose>>,
    free_stems: &[String],
    rematch_ratio: f32,
    rematch_cross_check: bool,
    min_matches: usize,
    matcher: &PairMatcher,
    tracks_use_essential: bool,
    pair_stem_window: Option<u64>,
) -> (usize, Vec<((usize, usize), usize)>) {
    let prior_idx: HashSet<usize> = pose_priors
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if prior_idx.is_empty() {
        return (0, Vec::new());
    }
    let stem_of = |idx: usize| -> &str {
        Path::new(&image_names[idx])
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(image_names[idx].as_str())
    };
    let guide_pose = |idx: usize| -> Option<Pose> {
        if let Some(by_stem) = guidance_poses_by_stem {
            if let Some(p) = by_stem.get(stem_of(idx)) {
                return Some(p.clone());
            }
        }
        poses.get(idx).and_then(|p| p.clone())
    };
    let free_want: HashSet<&str> = free_stems.iter().map(String::as_str).collect();
    let free_idx: HashSet<usize> = (0..features.len())
        .filter(|&i| {
            if prior_idx.contains(&i) {
                return false;
            }
            if guide_pose(i).is_none() {
                return false;
            }
            if free_want.is_empty() {
                true
            } else {
                free_want.contains(stem_of(i))
            }
        })
        .collect();
    if free_idx.is_empty() {
        return (0, Vec::new());
    }
    let mut targets: Vec<(usize, usize)> = Vec::new();
    for &f in &free_idx {
        for &p in &prior_idx {
            if guide_pose(p).is_none() {
                continue;
            }
            targets.push((f.min(p), f.max(p)));
        }
    }
    targets.sort_unstable();
    targets.dedup();
    if let Some(window) = pair_stem_window {
        let stem_values = numeric_stem_values(image_names)
            .expect("pair stem window was validated before pose-guided rematch");
        targets.retain(|&pair| {
            pair_within_stem_window(pair, &stem_values, window)
                .expect("rematch pair indices are loaded image indices")
        });
    }
    eprintln!(
        "rematch-pose-guided: {} free × {} prior → {} candidate pairs (ratio={:.2}, gt_guide={})",
        free_idx.len(),
        prior_idx.len(),
        targets.len(),
        rematch_ratio,
        guidance_poses_by_stem.is_some()
    );

    let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
    opts.calibrated_prefer_essential = true;
    let verifier = TwoViewGeometryVerifier::new(opts);

    let results: Vec<Option<PairwiseMatches>> = targets
        .par_iter()
        .map(|&(i, j)| {
            let (Some(pi), Some(pj)) = (guide_pose(i), guide_pose(j)) else {
                return None;
            };
            let pose_e = essential_from_absolute_poses(&pi, &pj)?;
            let dm0 = matcher.match_pair(
                rematch_ratio,
                rematch_cross_check,
                i,
                j,
                &features[i],
                &features[j],
            );
            let extra = guided_epipolar_matches(
                camera,
                &features[i],
                &features[j],
                &dm0,
                &[],
                2.0,
                Some(pose_e),
                0.8,
            );
            let mut dm = dm0;
            dm.extend(extra);
            if dm.len() < min_matches {
                return None;
            }
            let corrs: Vec<TwoViewCorrespondence> = dm
                .iter()
                .map(|m| {
                    TwoViewCorrespondence::new(
                        features[i].keypoints[m.query_index],
                        features[j].keypoints[m.train_index],
                    )
                })
                .collect();
            let report = verifier.classify(&corrs, camera);
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < min_matches {
                return None;
            }
            let essential_matches = if report.essential_inliers.len() >= min_matches {
                Some(
                    report
                        .essential_inliers
                        .iter()
                        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            let matches: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            Some(PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: Some(report.config),
                essential_matches,
                essential_matrix: report.essential,
            })
        })
        .collect();

    let mut changed = 0usize;
    let mut gained_e_pairs: Vec<((usize, usize), usize)> = Vec::new();
    let mut existing: HashMap<(usize, usize), usize> = HashMap::new();
    for (idx, p) in pairwise.iter().enumerate() {
        existing.insert((p.image_i.min(p.image_j), p.image_i.max(p.image_j)), idx);
    }
    for new in results.into_iter().flatten() {
        let key = (new.image_i.min(new.image_j), new.image_i.max(new.image_j));
        let new_e = new.essential_matches.as_ref().map_or(0, |e| e.len());
        let mut new = new;
        if tracks_use_essential {
            if let Some(ess) = new.essential_matches.clone() {
                if ess.len() >= min_matches {
                    new.matches = ess;
                }
            }
        }
        if let Some(&idx) = existing.get(&key) {
            let old = &pairwise[idx];
            let old_e = old.essential_matches.as_ref().map_or(0, |e| e.len());
            if new_e > old_e {
                eprintln!(
                    "rematch-pose-guided: improved {}-{} E {} -> {} (inliers {} -> {})",
                    stem_of(new.image_i),
                    stem_of(new.image_j),
                    old_e,
                    new_e,
                    old.matches.len(),
                    new.matches.len()
                );
                pairwise[idx] = new;
                gained_e_pairs.push((key, new_e));
                changed += 1;
            }
        } else if new_e >= min_matches {
            eprintln!(
                "rematch-pose-guided: new bridge {}-{} E={} inliers={}",
                stem_of(new.image_i),
                stem_of(new.image_j),
                new_e,
                new.matches.len()
            );
            existing.insert(key, pairwise.len());
            pairwise.push(new);
            gained_e_pairs.push((key, new_e));
            changed += 1;
        }
    }
    (changed, gained_e_pairs)
}

pub(super) fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    cross_check: bool,
    guided_matching: bool,
    multiple_models: bool,
    min_e_f_inlier_ratio: Option<f64>,
    calibrated_prefer_essential: bool,
    refine_uncalibrated_f_to_essential: bool,
    strict_uncalibrated_f_to_essential: bool,
    calibrated_essential_primary: bool,
    force_essential_matches: bool,
    force_essential_min_ef_ratio: f64,
    force_essential_min_e_inliers: usize,
    force_essential_uncalibrated_only: bool,
    guided_max_error_px: Option<f64>,
    guided_lowe_ratio: Option<f64>,
    imported_matches: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    imported_matches_supplement: Option<&HashMap<(usize, usize), Vec<(usize, usize)>>>,
    colmap_guided_matching: bool,
) -> (
    Vec<PairwiseMatches>,
    VerificationStats,
    HashMap<(usize, usize), SnapshotPairMetadata>,
) {
    let verifier = (mode == VerificationMode::Full).then(|| {
        let mut opts = TwoViewGeometryOptions::for_camera(camera, 4.0);
        opts.multiple_models = multiple_models;
        if let Some(r) = min_e_f_inlier_ratio {
            opts.min_e_f_inlier_ratio = r;
        }
        opts.calibrated_prefer_essential = calibrated_prefer_essential;
        TwoViewGeometryVerifier::new(opts)
    });
    // Same single-model essential-only estimator as the legacy path, just
    // with `for_camera`'s per-camera pixel-derived Sampson threshold swapped
    // in for the fixed `5e-3` default — everything else (iterations, seed,
    // translation scale) stays at `EssentialRansacConfig`/`RelativePoseEstimator`
    // defaults, matching the legacy path field-for-field.
    let threshold_only_estimator = (mode == VerificationMode::ThresholdOnly).then(|| {
        let sampson_threshold =
            TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
        RelativePoseEstimator {
            ransac: EssentialRansac {
                estimator: EightPointEssentialMatrixEstimator::default(),
                config: EssentialRansacConfig {
                    sampson_threshold,
                    ..EssentialRansacConfig::default()
                },
            },
            default_translation_scale: 1.0,
            ..RelativePoseEstimator::default()
        }
    });

    let verify_started = std::time::Instant::now();
    #[allow(unused_mut)] // only written with `--features gpu`
    let mut gpu_match_seconds = 0.0f64;
    #[cfg(feature = "onnx-inference")]
    let sequential = matches!(matcher, PairMatcher::LightGlue { .. });
    #[cfg(not(feature = "onnx-inference"))]
    let sequential = false;
    let force_e_swaps = AtomicUsize::new(0);
    let f_to_e_refinements = AtomicUsize::new(0);
    let strict_f_to_e_exclusions = AtomicUsize::new(0);
    let strict_f_to_e_excluded_inliers = AtomicUsize::new(0);
    let calibrated_essential_primary_promotions = AtomicUsize::new(0);
    let dump_match_stats = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_MATCH_STATS").is_some();
    let dump_pair_outcomes = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_PAIR_OUTCOMES").is_some();
    let dump_match_indices = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_MATCH_INDICES").is_some();
    let dump_guided_matches = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_GUIDED_MATCHES").is_some();
    // Full-graph, GT-independent E quality probe.  This is intentionally
    // environment-gated: it adds bounded triangulation work and must never
    // alter the ordinary verification or mapper path.  When a refined F is
    // available, the same row also contains a diagnostics-only `Kᵀ F K` → E
    // comparison (`f2e_*` fields); it is never fed back into verification.
    let dump_essential_quality =
        std::env::var_os("VISLOC_SFM_DEBUG_DUMP_ESSENTIAL_QUALITY").is_some();
    let dump_f2e_diagnostics = std::env::var_os("VISLOC_SFM_DEBUG_DUMP_F2E_DIAGNOSTICS").is_some();
    let verify_one = |&(i, j): &(usize, usize), precomputed: Option<Vec<DescriptorMatch>>| {
        let dm: Vec<DescriptorMatch> = if let Some(imp) = imported_matches {
            let key = (i.min(j), i.max(j));
            let Some(raw) = imp.get(&key) else {
                return (None, None, None);
            };
            let flip = i > j;
            raw.iter()
                .map(|&(a, b)| {
                    let (qi, tj) = if flip { (b, a) } else { (a, b) };
                    DescriptorMatch {
                        query_index: qi,
                        train_index: tj,
                        distance: 0.0,
                        second_best_distance: None,
                        ratio: None,
                        confidence: None,
                    }
                })
                .collect()
        } else if let Some(supp) = imported_matches_supplement {
            let key = (i.min(j), i.max(j));
            if let Some(raw) = supp.get(&key) {
                let flip = i > j;
                raw.iter()
                    .map(|&(a, b)| {
                        let (qi, tj) = if flip { (b, a) } else { (a, b) };
                        DescriptorMatch {
                            query_index: qi,
                            train_index: tj,
                            distance: 0.0,
                            second_best_distance: None,
                            ratio: None,
                            confidence: None,
                        }
                    })
                    .collect()
            } else {
                matcher.match_pair(match_ratio, cross_check, i, j, &features[i], &features[j])
            }
        } else if let Some(dm) = precomputed {
            dm
        } else {
            matcher.match_pair(match_ratio, cross_check, i, j, &features[i], &features[j])
        };
        if dump_match_stats {
            eprintln!("sfm-debug-raw: {} {} matches={}", i, j, dm.len());
        }
        if dm.len() < min_matches {
            if dump_pair_outcomes {
                eprintln!(
                        "sfm-debug-outcome: {i} {j} raw={} config=TOO_FEW accepted=0 e=0 f=0 h=0 reason=too_few_raw",
                        dm.len(),
                    );
            }
            return (None, None, None);
        }
        let corrs: Vec<TwoViewCorrespondence> = dm
            .iter()
            .map(|m| {
                TwoViewCorrespondence::new(
                    features[i].keypoints[m.query_index],
                    features[j].keypoints[m.train_index],
                )
            })
            .collect();

        if let Some(verifier) = &verifier {
            let report = verifier.classify(&corrs, camera);
            // M2.1: mirror COLMAP's real gate (`database_cache.cc`'s
            // `UseInlierMatchesCheck`), which is `num_matches >=
            // min_num_matches && (!ignore_watermarks || config !=
            // WATERMARK)` — i.e. every non-`DEGENERATE`, non-`WATERMARK`
            // configuration contributes its inlier matches, including
            // `PLANAR_OR_PANORAMIC`/`PANORAMIC` (homography-only, no
            // triangulatable baseline). `DEGENERATE` needs no explicit
            // arm here because [`TwoViewGeometryVerifier`] already
            // returns an empty inlier list for it (`degenerate_report()`
            // in `colmap_verification.rs`), the same reason COLMAP's own
            // degenerate branch never populates `inlier_matches`.
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < min_matches {
                if dump_pair_outcomes {
                    let reason = if !keep {
                        "configuration_rejected"
                    } else {
                        "inliers_below_min"
                    };
                    eprintln!(
                            "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason={reason}",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                        );
                }
                if dump_essential_quality {
                    let quality = essential_pair_quality(&report, &corrs, camera);
                    let f2e_quality = fundamental_to_essential_quality(&report, &corrs, camera);
                    eprintln!(
                            "sfm-debug-essential-quality: {i} {j} raw={} config={} accepted={} e={} f={} h={}{}{}",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                            format_essential_pair_quality(quality),
                            format_fundamental_to_essential_quality(f2e_quality),
                        );
                }
                return (None, Some(report.config), None);
            }
            // Guided matching (COLMAP FindGuidedMatches): expand the
            // match set under the verified epipolar geometry, then
            // re-verify so config/inliers describe the final set.
            let (dm, report, report_corrs) = if guided_matching {
                let original_report = report.clone();
                // Prefer E inliers as the epipolar seed when available —
                // F-seeded guided matching densifies Uncalibrated façades
                // without raising calibrated bridges (courtyard prior↔hub).
                let seed_idx: &[usize] = if report.essential_inliers.len() >= 8 {
                    &report.essential_inliers
                } else {
                    &report.inliers
                };
                let inlier_corrs: Vec<TwoViewCorrespondence> = seed_idx
                    .iter()
                    .filter_map(|&idx| corrs.get(idx).copied())
                    .collect();
                let guided_max_error = guided_max_error_px.unwrap_or(2.0);
                let guided_ratio = guided_lowe_ratio.unwrap_or(0.8);
                let extra = if colmap_guided_matching {
                    colmap_guided_matches(
                        camera,
                        &features[i],
                        &features[j],
                        &dm,
                        &report,
                        guided_max_error,
                        guided_ratio,
                        cross_check,
                    )
                } else {
                    guided_epipolar_matches(
                        camera,
                        &features[i],
                        &features[j],
                        &dm,
                        &inlier_corrs,
                        guided_max_error,
                        None,
                        guided_ratio,
                    )
                };
                if dump_guided_matches {
                    eprintln!(
                        "sfm-debug-guided: {i} {j} model={} base={} extra={} ratio={:.3} max_error_px={:.3} cross_check={}",
                        colmap_guided_geometry_name(colmap_guided_geometry(&report)),
                        dm.len(),
                        extra.len(),
                        guided_ratio,
                        guided_max_error,
                        cross_check,
                    );
                    for descriptor_match in &extra {
                        eprintln!(
                            "sfm-debug-guided-match: {i} {j} query={} train={} distance={:.9e}",
                            descriptor_match.query_index,
                            descriptor_match.train_index,
                            descriptor_match.distance,
                        );
                    }
                }
                if extra.is_empty() {
                    (dm, report, corrs)
                } else {
                    let mut expanded = dm.clone();
                    expanded.extend(extra);
                    let new_corrs: Vec<TwoViewCorrespondence> = expanded
                        .iter()
                        .map(|m| {
                            TwoViewCorrespondence::new(
                                features[i].keypoints[m.query_index],
                                features[j].keypoints[m.train_index],
                            )
                        })
                        .collect();
                    let new_report = verifier.classify(&new_corrs, camera);
                    if new_report.inliers.len() >= min_matches {
                        if colmap_guided_matching {
                            // The compatibility mode is append-only by
                            // contract: a new model is allowed to add
                            // verified inliers, but it must not make a
                            // previously accepted baseline correspondence
                            // disappear merely because model selection changed
                            // after expansion.
                            let mut preserved_report = new_report;
                            let mut inliers = preserved_report.inliers.clone();
                            for &index in &original_report.inliers {
                                if !inliers.contains(&index) {
                                    inliers.push(index);
                                }
                            }
                            inliers.sort_unstable();
                            preserved_report.inliers = inliers;
                            (expanded, preserved_report, new_corrs)
                        } else {
                            (expanded, new_report, new_corrs)
                        }
                    } else {
                        (dm, report, corrs)
                    }
                }
            } else {
                (dm, report, corrs)
            };
            if dump_f2e_diagnostics && report.config == ConfigurationType::Uncalibrated {
                if let Some(diagnostics) =
                    f_to_e_candidate_diagnostics(&report, &report_corrs, camera)
                {
                    eprintln!(
                            "sfm-debug-f2e-candidate: {i} {j} s1={:.9e} s2={:.9e} s3={:.9e} projection_distortion={:.6} s1_s2_mismatch={:.6} s3_s2={:.6} f_inliers={} ef_inliers={} ef_overlap_on_f={:.6} f_norm_residual={:.6e} ef_norm_residual_on_f={:.6e} ef_to_f_residual_ratio={:.6} cheirality_ratio={:.6} cheirality_margin={:.6} ef_angle_p25_deg={:.6} stable_refits={} pose_rotation_spread_deg={:.6} pose_translation_spread_deg={:.6}",
                            diagnostics.calibrated_s1,
                            diagnostics.calibrated_s2,
                            diagnostics.calibrated_s3,
                            diagnostics.projection_distortion,
                            diagnostics.s1_s2_mismatch,
                            diagnostics.s3_s2_ratio,
                            diagnostics.f_inliers,
                            diagnostics.ef_inliers,
                            diagnostics.ef_overlap_on_f,
                            diagnostics.f_normalized_residual,
                            diagnostics.ef_normalized_residual_on_f,
                            diagnostics.ef_to_f_residual_ratio,
                            diagnostics.cheirality_ratio,
                            diagnostics.cheirality_margin,
                            diagnostics.ef_angle_p25_deg,
                            diagnostics.stable_refits,
                            diagnostics.pose_rotation_spread_deg,
                            diagnostics.pose_translation_spread_deg,
                        );
                } else {
                    eprintln!("sfm-debug-f2e-candidate: {i} {j} invalid=1");
                }
            }
            if dump_essential_quality {
                let quality = essential_pair_quality(&report, &report_corrs, camera);
                let f2e_quality = fundamental_to_essential_quality(&report, &report_corrs, camera);
                eprintln!(
                        "sfm-debug-essential-quality: {i} {j} raw={} config={} accepted={} e={} f={} h={}{}{}",
                        dm.len(),
                        configuration_name(report.config),
                        report.inliers.len(),
                        report.e_inlier_count,
                        report.f_inlier_count,
                        report.h_inlier_count,
                        format_essential_pair_quality(quality),
                        format_fundamental_to_essential_quality(f2e_quality),
                    );
            }
            let direct_essential_primary = if calibrated_essential_primary {
                select_calibrated_essential_primary(&report, &report_corrs, camera, min_matches)
            } else {
                None
            };
            if let Some(selection) = &direct_essential_primary {
                calibrated_essential_primary_promotions.fetch_add(1, Ordering::Relaxed);
                if dump_pair_outcomes {
                    eprintln!(
                            "sfm-debug-calibrated-essential-primary: {i} {j} f_inliers={} initial_e_inliers={} rescored_e_inliers={} cheirality={}/{} mean_sampson={:.6e}",
                            report.f_inlier_count,
                            selection.initial_inlier_count,
                            selection.inlier_indices.len(),
                            selection.quality.best_cheirality,
                            selection.inlier_indices.len(),
                            selection.quality.mean_sampson,
                        );
                }
            }
            let f_to_e_refinement = if direct_essential_primary.is_none()
                && (refine_uncalibrated_f_to_essential || strict_uncalibrated_f_to_essential)
            {
                refine_uncalibrated_f_winner(&report, &report_corrs, camera, min_matches)
            } else {
                None
            };
            if let Some(refinement) = &f_to_e_refinement {
                f_to_e_refinements.fetch_add(1, Ordering::Relaxed);
                if dump_pair_outcomes {
                    let cheirality_ratio = refinement.quality.best_cheirality as f64
                        / refinement.inlier_indices.len() as f64;
                    let second_over_best = if refinement.quality.best_cheirality > 0 {
                        refinement.quality.second_cheirality as f64
                            / refinement.quality.best_cheirality as f64
                    } else {
                        f64::NAN
                    };
                    eprintln!(
                            "sfm-debug-f2e-refinement: {i} {j} f_inliers={} ef_inliers={} cheirality={}/{} cheirality_ratio={:.6} second_over_best={:.6}",
                            refinement.f_inlier_count,
                            refinement.inlier_indices.len(),
                            refinement.quality.best_cheirality,
                            refinement.inlier_indices.len(),
                            cheirality_ratio,
                            second_over_best,
                        );
                }
            }
            if should_exclude_strict_uncalibrated_f_winner(
                strict_uncalibrated_f_to_essential,
                camera,
                &report,
                f_to_e_refinement.as_ref(),
            ) && direct_essential_primary.is_none()
            {
                strict_f_to_e_exclusions.fetch_add(1, Ordering::Relaxed);
                strict_f_to_e_excluded_inliers.fetch_add(report.inliers.len(), Ordering::Relaxed);
                if dump_pair_outcomes {
                    eprintln!(
                            "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason=strict_f2e_gate_rejected",
                            dm.len(),
                            configuration_name(report.config),
                            report.inliers.len(),
                            report.e_inlier_count,
                            report.f_inlier_count,
                            report.h_inlier_count,
                        );
                }
                // There is no rotation-only PairwiseMatches representation;
                // omit this edge so its uncalibrated F observations cannot
                // enter translation initialization or track construction.
                return (None, Some(report.config), None);
            }
            let essential_matches = {
                let ef_ratio = if report.f_inlier_count > 0 {
                    report.e_inlier_count as f64 / report.f_inlier_count as f64
                } else if report.e_inlier_count > 0 {
                    f64::INFINITY
                } else {
                    0.0
                };
                // Always require the strong-E gate before exposing E inliers
                // to prefer-essential edge construction (weak E poisons bearings).
                let strong = report.e_inlier_count >= force_essential_min_e_inliers
                    && ef_ratio >= force_essential_min_ef_ratio
                    && report.essential_inliers.len() >= min_matches;
                if strong {
                    Some(
                        report
                            .essential_inliers
                            .iter()
                            .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                            .collect::<Vec<_>>(),
                    )
                } else {
                    None
                }
            };
            let refined_matches = f_to_e_refinement.as_ref().map(|refinement| {
                refinement
                    .inlier_indices
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect::<Vec<_>>()
            });
            let direct_matches = direct_essential_primary.as_ref().map(|selection| {
                selection
                    .inlier_indices
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect::<Vec<_>>()
            });
            let winning: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let uncalibrated_ok = !force_essential_uncalibrated_only
                || matches!(report.config, ConfigurationType::Uncalibrated);
            let matches: Vec<(usize, usize)> = if let Some(primary) = &direct_matches {
                primary.clone()
            } else if let Some(refined) = &refined_matches {
                refined.clone()
            } else {
                match (
                    force_essential_matches && uncalibrated_ok && essential_matches.is_some(),
                    essential_matches.as_ref(),
                ) {
                    (true, Some(ess)) => {
                        force_e_swaps.fetch_add(1, Ordering::Relaxed);
                        ess.clone()
                    }
                    _ => winning,
                }
            };
            let essential_matrix = direct_essential_primary
                .as_ref()
                .map(|selection| selection.essential)
                .or_else(|| {
                    f_to_e_refinement
                        .as_ref()
                        .map(|refinement| refinement.essential)
                })
                .or(report.essential);
            let output_config = if direct_essential_primary.is_some() {
                ConfigurationType::Calibrated
            } else {
                report.config
            };
            if dump_pair_outcomes {
                eprintln!(
                        "sfm-debug-outcome: {i} {j} raw={} config={} accepted={} e={} f={} h={} reason=accepted",
                        dm.len(),
                        configuration_name(output_config),
                        matches.len(),
                        report.e_inlier_count,
                        report.f_inlier_count,
                        report.h_inlier_count,
                    );
                // Keep the accepted-set dump self-contained for an
                // order-only replay.  The verified E is diagnostics
                // output only; ordinary reconstruction never serializes
                // or consumes this line.
                let e_values = essential_matrix.as_ref().map_or_else(
                    || "0 0 0 0 0 0 0 0 0".to_string(),
                    |matrix| {
                        matrix
                            .as_slice()
                            .iter()
                            .map(|value| format!("{value:.17e}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    },
                );
                eprintln!("sfm-debug-essential-matrix: {i} {j} values={e_values}");
            }
            if dump_match_indices {
                for &(query_index, train_index) in &matches {
                    eprintln!("sfm-debug-match: {i} {j} query={query_index} train={train_index}");
                }
            }
            let pair = PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: Some(output_config),
                essential_matches: direct_matches.or(refined_matches).or(essential_matches),
                essential_matrix,
            };
            let mut metadata = snapshot_metadata_from_report(&dm, &report);
            metadata.accepted_inlier_indices =
                snapshot_indices_for_matches(&metadata.raw_matches, &pair.matches)
                    .unwrap_or_default();
            metadata.essential_inlier_indices = pair
                .essential_matches
                .as_ref()
                .and_then(|matches| snapshot_indices_for_matches(&metadata.raw_matches, matches))
                .unwrap_or_default();
            (Some(pair), Some(output_config), Some(metadata))
        } else {
            let estimator = match &threshold_only_estimator {
                Some(e) => *e,
                None => RelativePoseEstimator::default(),
            };
            let Some(rel) = estimator.estimate(&corrs, camera) else {
                return (None, None, None);
            };
            if rel.inliers.len() < min_matches {
                return (None, None, None);
            }
            let matches: Vec<(usize, usize)> = rel
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let pair = PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            };
            let metadata = SnapshotPairMetadata {
                raw_match_count: dm.len(),
                raw_matches: dm.iter().map(|m| (m.query_index, m.train_index)).collect(),
                accepted_inlier_indices: rel.inliers.clone(),
                essential_inlier_indices: Vec::new(),
                e_inlier_count: rel.inliers.len(),
                relative_pose: Some((
                    rel.previous_to_current
                        .rotation
                        .to_rotation_matrix()
                        .into_inner(),
                    rel.previous_to_current.translation,
                )),
                ..SnapshotPairMetadata::default()
            };
            (Some(pair), None, Some(metadata))
        }
    };

    // LightGlue holds a single Mutex'd ORT session. Rayon + ORT's internal
    // thread pool deadlocks on this machine; keep LightGlue sequential with
    // progress logs. NN matching stays parallel.
    let results: Vec<(
        Option<PairwiseMatches>,
        Option<ConfigurationType>,
        Option<SnapshotPairMetadata>,
    )> = if sequential {
        let total = candidates.len();
        candidates
            .iter()
            .enumerate()
            .map(|(k, pair)| {
                if k % 25 == 0 || k + 1 == total {
                    eprintln!("lightglue verify: {} / {} pairs", k + 1, total);
                }
                verify_one(pair, None)
            })
            .collect()
    } else {
        // `--gpu-match`: one batched GPU pass per chunk of plain-NN pairs
        // (imported/supplement matches keep their own source), then the
        // usual parallel verification.
        #[cfg(feature = "gpu")]
        let gpu = (matches!(matcher, PairMatcher::Nn)
            && imported_matches.is_none()
            && imported_matches_supplement.is_none())
        .then(|| gpu_nn_bank(features))
        .flatten();
        #[cfg(not(feature = "gpu"))]
        let gpu: Option<()> = None;
        match gpu {
            #[cfg(feature = "gpu")]
            Some(bank) => {
                let (ctx, gm) = GPU_NN.get().expect("gpu bank implies GPU_NN");
                let verify_chunk =
                    |chunk: &[(usize, usize)], dms: Vec<Vec<DescriptorMatch>>| -> Vec<_> {
                        chunk
                            .par_iter()
                            .zip(dms.into_par_iter())
                            .map(|(pair, dm)| verify_one(pair, Some(dm)))
                            .collect()
                    };
                // Software pipeline: the GPU matches chunk k while the CPU
                // verifies chunk k - 1.
                let mut out = Vec::with_capacity(candidates.len());
                let mut pending: Option<(&[(usize, usize)], Vec<Vec<DescriptorMatch>>)> = None;
                for chunk in candidates.chunks(512) {
                    let ((dms, seconds), verified) = rayon::join(
                        || {
                            let started = std::time::Instant::now();
                            let dms =
                                gm.match_pairs(ctx, &bank, chunk, Some(match_ratio), cross_check);
                            (dms, started.elapsed().as_secs_f64())
                        },
                        || pending.take().map(|(c, d)| verify_chunk(c, d)),
                    );
                    gpu_match_seconds += seconds;
                    out.extend(verified.into_iter().flatten());
                    pending = Some((chunk, dms));
                }
                if let Some((c, d)) = pending {
                    out.extend(verify_chunk(c, d));
                }
                out
            }
            _ => candidates.par_iter().map(|p| verify_one(p, None)).collect(),
        }
    };
    eprintln!(
        "verify-pairs: {} candidates in {:.2}s (gpu match {:.2}s)",
        candidates.len(),
        verify_started.elapsed().as_secs_f64(),
        gpu_match_seconds
    );

    let mut stats = VerificationStats::default();
    let mut pairwise = Vec::with_capacity(results.len());
    let mut metadata_by_pair = HashMap::new();
    for (pair, config, metadata) in results {
        if let Some(config) = config {
            stats.record(config);
        }
        if let Some(pair) = pair {
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            if let Some(metadata) = metadata {
                metadata_by_pair.insert(key, metadata);
            }
            pairwise.push(pair);
        }
    }
    stats.force_essential_swaps = force_e_swaps.load(Ordering::Relaxed);
    stats.uncalibrated_f_to_essential_refinements = f_to_e_refinements.load(Ordering::Relaxed);
    stats.strict_uncalibrated_f_to_essential_exclusions =
        strict_f_to_e_exclusions.load(Ordering::Relaxed);
    stats.strict_uncalibrated_f_to_essential_excluded_inliers =
        strict_f_to_e_excluded_inliers.load(Ordering::Relaxed);
    stats.calibrated_essential_primary_promotions =
        calibrated_essential_primary_promotions.load(Ordering::Relaxed);
    (pairwise, stats, metadata_by_pair)
}

/// One rescue-pass candidate's outcome, kept for reporting regardless of
/// whether it was admitted — `main`'s acceptance report (M5,
/// `docs/colmap_port_plan.md`) needs both "which bridges were found" and,
/// in the honest-negative case, "how close did the closest attempt get".
#[derive(Debug, Clone, Copy)]
struct RescueAttempt {
    pair: (usize, usize),
    raw_matches: usize,
    config: ConfigurationType,
    inliers: usize,
}

/// M5 (`docs/colmap_port_plan.md`): opt-in rescue-bridging pass, run after
/// the initial [`verify_pairs`] call. Detects whether the resulting
/// verified-pair graph (`pairwise`) is disconnected
/// (`visloc_rs::vision::two_view::connected_components`); if so, proposes
/// cross-component candidate pairs ranked by a fresh VLAD global-descriptor
/// similarity and budget-capped (`generate_bridge_candidates`), rematches
/// each with the relaxed `--rescue-*` profile, and re-verifies with the same
/// [`TwoViewGeometryVerifier`] / keep-list [`verify_pairs`] itself uses under
/// `--verification-mode full` — a looser matcher only ever *proposes* a
/// bridge here, never *admits* one unverified (the M1.1 lesson).
///
/// Returns the admitted bridge pairs, already in `PairwiseMatches` form and
/// ready to append to the caller's verified-pair list (every attempt's
/// [`RescueAttempt`] outcome — admitted or not — is reported via `println!`
/// as it's produced, per this milestone's acceptance-report requirement).
pub(super) fn rescue_bridging(
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    args: &Args,
    matcher: &PairMatcher,
) -> Result<Vec<PairwiseMatches>, String> {
    let n = features.len();
    let edges: Vec<(usize, usize)> = pairwise.iter().map(|p| (p.image_i, p.image_j)).collect();
    let components = connected_components(n, &edges);
    println!(
        "rescue-bridging: view graph has {} connected component(s) (sizes {})",
        components.len(),
        components
            .iter()
            .map(|c| c.len().to_string())
            .collect::<Vec<_>>()
            .join("+"),
    );
    if components.len() <= 1 {
        println!("rescue-bridging: graph is already connected, nothing to bridge");
        return Ok(Vec::new());
    }

    // Retrieval score for ranking cross-component candidates: a fresh VLAD
    // vocabulary/global descriptor per image, independent of whichever
    // `--pair-source` built the *initial* graph (so this still works under
    // `--pair-source vocab-tree`). Falls back to a uniform (unranked) score
    // if the vocabulary cannot be built — the candidate generator itself
    // still enforces "cross-component only, budget-capped" either way.
    let sample = sampled_training_descriptors(features);
    let globals: Option<Vec<Vec<f32>>> =
        Vocabulary::build(&sample, args.vocab_size, 10, 0).map(|vocab| {
            features
                .iter()
                .map(|f| vlad(&f.descriptors, &vocab))
                .collect()
        });
    let similarity = |i: usize, j: usize| -> f32 {
        match &globals {
            Some(g) => cosine_similarity(&g[i], &g[j]),
            None => 0.0,
        }
    };

    let all_candidates = generate_bridge_candidates(
        &components,
        similarity,
        &BridgeCandidateOptions {
            max_candidates: args.rescue_max_candidates,
        },
    );
    let candidates =
        filter_pairs_by_stem_window(all_candidates, image_names, args.pair_stem_window)?;
    println!(
        "rescue-bridging: {} cross-component candidate pair(s) proposed (ratio={}, cross_check={}, min_matches={})",
        candidates.len(),
        args.rescue_match_ratio,
        args.rescue_cross_check,
        args.rescue_min_matches,
    );

    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));

    let results: Vec<(Option<PairwiseMatches>, RescueAttempt)> = candidates
        .par_iter()
        .map(|&(i, j)| {
            let dm = matcher.match_pair(
                args.rescue_match_ratio,
                args.rescue_cross_check,
                i,
                j,
                &features[i],
                &features[j],
            );
            let raw_matches = dm.len();
            if raw_matches < args.rescue_min_matches {
                return (
                    None,
                    RescueAttempt {
                        pair: (i, j),
                        raw_matches,
                        config: ConfigurationType::Degenerate,
                        inliers: 0,
                    },
                );
            }

            let corrs: Vec<TwoViewCorrespondence> = dm
                .iter()
                .map(|m| {
                    TwoViewCorrespondence::new(
                        features[i].keypoints[m.query_index],
                        features[j].keypoints[m.train_index],
                    )
                })
                .collect();
            let report = verifier.classify(&corrs, camera);
            let attempt = RescueAttempt {
                pair: (i, j),
                raw_matches,
                config: report.config,
                inliers: report.inliers.len(),
            };
            // Same keep-list `verify_pairs`'s `full` mode uses (M2.1): every
            // non-DEGENERATE, non-WATERMARK configuration is admissible.
            let keep = matches!(
                report.config,
                ConfigurationType::Calibrated
                    | ConfigurationType::Uncalibrated
                    | ConfigurationType::Planar
                    | ConfigurationType::Panoramic
                    | ConfigurationType::PlanarOrPanoramic
                    | ConfigurationType::Multiple
            );
            if !keep || report.inliers.len() < args.rescue_min_matches {
                return (None, attempt);
            }
            let matches: Vec<(usize, usize)> = report
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            let essential_matches = if report.essential_inliers.len() >= args.rescue_min_matches {
                Some(
                    report
                        .essential_inliers
                        .iter()
                        .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                        .collect(),
                )
            } else {
                None
            };
            (
                Some(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                    two_view_config: Some(report.config),
                    essential_matches,
                    essential_matrix: report.essential,
                }),
                attempt,
            )
        })
        .collect();

    let mut admitted = Vec::new();
    let mut attempts = Vec::with_capacity(results.len());
    for (pair, attempt) in results {
        if let Some(pair) = &pair {
            println!(
                "rescue-bridging: BRIDGE admitted ({}, {}) raw_matches={} inliers={} config={:?}",
                attempt.pair.0,
                attempt.pair.1,
                attempt.raw_matches,
                attempt.inliers,
                attempt.config,
            );
            admitted.push(pair.clone());
        }
        attempts.push(attempt);
    }

    if let Some(best) = attempts.iter().max_by_key(|a| a.inliers) {
        println!(
            "rescue-bridging: best cross-component attempt ({}, {}) raw_matches={} inliers={} config={:?}",
            best.pair.0, best.pair.1, best.raw_matches, best.inliers, best.config,
        );
    }
    println!(
        "rescue-bridging: {} bridge pair(s) admitted out of {} attempted",
        admitted.len(),
        candidates.len(),
    );

    if !admitted.is_empty() {
        let mut all_edges = edges;
        all_edges.extend(admitted.iter().map(|p| (p.image_i, p.image_j)));
        let components_after = connected_components(n, &all_edges);
        println!(
            "rescue-bridging: view graph now has {} connected component(s) after admission",
            components_after.len(),
        );
    }

    Ok(admitted)
}
