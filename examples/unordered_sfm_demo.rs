//! Incremental structure-from-motion from an **unordered** image set — the
//! COLMAP-style SfM pillar of visloc-rs.
//!
//! Unlike the stereo-VO SfM path (`--sfm-colmap-out` on
//! `stereo_vo_external_deep_files`), which needs an *ordered* video with
//! frame→frame matches, this demo takes a directory of per-image deep features
//! with **no temporal order**, builds its own view graph, and grows one
//! reconstruction:
//!
//! 1. **View graph.** A VLAD vocabulary over all descriptors gives each image a
//!    global descriptor; the top-K most similar images per image become
//!    candidate pairs (or `--exhaustive` for all pairs).
//! 2. **Verified matches.** Each candidate pair is matched (cross-checked
//!    brute-force + Lowe ratio) and geometrically verified per
//!    `--verification-mode` (default `legacy`):
//!    - `legacy`: essential-matrix-only RANSAC, COLMAP's legacy fixed
//!      `5e-3`-normalized Sampson threshold (`RelativePoseEstimator`,
//!      unchanged since before M1).
//!    - `threshold-only`: the same single-model essential-matrix-only RANSAC
//!      as `legacy`, but with the per-camera pixel-derived Sampson threshold
//!      (`TwoViewGeometryOptions::for_camera`'s ≈4px-equivalent bound) instead
//!      of the fixed `5e-3` default — isolates the threshold half of the M1
//!      confound (see `docs/colmap_port_plan.md`'s "M1.1 results").
//!    - `full`: COLMAP-style multi-model (essential / fundamental /
//!      homography) verification with `ConfigurationType` classification
//!      (`visloc_rs::vision::two_view::colmap_verification`, ported from
//!      `src/colmap/estimators/two_view_geometry.cc`): pairs classified
//!      `DEGENERATE`, `WATERMARK`, `PANORAMIC` (no triangulatable baseline),
//!      or unresolved `PLANAR_OR_PANORAMIC` are dropped entirely instead of
//!      being fed to `incremental_sfm`, and the pair's own model inliers (not
//!      necessarily the essential matrix's) become its `PairwiseMatches`.
//!
//!    A per-`ConfigurationType` count is printed under `full`, so all three
//!    modes can be A/B'd on the same view graph — this is the M1/M1.1
//!    acceptance experiments' switch (see `docs/colmap_port_plan.md`). The
//!    legacy `--colmap-verification` boolean flag still works as a shorthand
//!    for `--verification-mode full`.
//! 3. **Incremental SfM.** [`visloc_rs::slam::incremental_sfm`] seeds from the
//!    strongest pair, registers images by PnP, triangulates tracks, and bundle-
//!    adjusts. Its first internal step — building feature tracks out of the
//!    verified pairs above — is itself an M2 A/B switch: `--track-source
//!    union-find` (default) is the original ad hoc union-find, `--track-source
//!    graph` routes through COLMAP's persistent `CorrespondenceGraph`
//!    (`visloc_rs::vision::two_view::correspondence_graph`, ported from
//!    `src/colmap/scene/correspondence_graph.{h,cc}`) instead. Both are proven
//!    to produce byte-identical tracks (see `pipelines/slam/src/
//!    incremental_sfm.rs`'s `graph_tracks_match_union_find_tracks_*` tests),
//!    so this flag is the M2 acceptance experiment's switch, not a behaviour
//!    change — see `docs/colmap_port_plan.md`'s "M2 results".
//! 4. **Export.** The registered poses + merged multi-view tracks are written as
//!    a COLMAP text model (`cameras.txt` / `images.txt` / `points3D.txt`),
//!    ready for 3DGS / NeRF training.
//!
//! Feature-file format is the same `X Y SCORE D0 D1 …` per keypoint used by
//! `read_external_deep_features_txt` (export SuperPoint with the repo's helper
//! scripts). The image set is every file in `--features-dir` ending with
//! `--feature-suffix`, sorted lexically; each image's COLMAP name is that file
//! with the suffix replaced by `--image-suffix`.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --example unordered_sfm_demo -- \
//!     --features-dir /tmp/sp_photos \
//!     --feature-suffix _features.txt --image-suffix .png \
//!     --width 752 --height 480 --fx 458.6 --fy 457.3 --cx 367.2 --cy 248.4 \
//!     --retrieval-topk 12 --min-matches 30 \
//!     --out-colmap /tmp/photos_sfm_colmap
//! ```
//!
//! Add `--verification-mode threshold-only` or `--verification-mode full`
//! (or the legacy `--colmap-verification` boolean, equivalent to `full`) to
//! swap in the COLMAP-style two-view verification paths described above
//! instead of the default legacy essential-matrix-only path; see
//! `verify_pairs`'s doc comment and `docs/colmap_port_plan.md`'s M1/M1.1
//! sections.

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3};
use rayon::prelude::*;
use visloc_rs::vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_rs::vision::two_view::{
    ConfigurationType, EightPointEssentialMatrixEstimator, EssentialRansac, EssentialRansacConfig,
    RelativePoseEstimator, TwoViewCorrespondence, TwoViewGeometryOptions, TwoViewGeometryVerifier,
};
use visloc_rs::{
    incremental_sfm, read_external_deep_features_txt, write_colmap_reconstruction_for_3dgs,
    BaConfig, BruteForceMatcher, Camera, CrossCheckMatcher, FeatureSet, IncrementalSfmConfig,
    Matcher, PairwiseMatches, Pose, TrackSource,
};

/// A COLMAP-export landmark: world position + `(image, keypoint, pixel)` track.
type ExportLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

/// The M1/M1.1 two-view verification A/B switch — see the file header and
/// `verify_pairs`'s doc comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerificationMode {
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

struct Args {
    features_dir: PathBuf,
    feature_suffix: String,
    image_suffix: String,
    out_colmap: PathBuf,
    camera: Camera,
    vocab_size: usize,
    retrieval_topk: usize,
    exhaustive: bool,
    match_ratio: f32,
    min_matches: usize,
    min_pnp_inliers: usize,
    max_reproj: f64,
    final_ba: bool,
    seed_trials: usize,
    refine_intrinsics: bool,
    refine_distortion: bool,
    colmap_style: bool,
    filter_images: bool,
    verification_mode: VerificationMode,
    /// M2 A/B switch: which algorithm builds feature tracks from the verified
    /// pairs (`docs/colmap_port_plan.md`'s M2 milestone) — the legacy ad hoc
    /// union-find (default) or COLMAP's persistent `CorrespondenceGraph`.
    track_source: TrackSource,
}

/// Parse `--track-source`'s value into the M2 [`TrackSource`] A/B switch.
/// `TrackSource` lives in `visloc-slam` and has no `FromStr` of its own (it's
/// a plain engine config knob, not a CLI type), so this demo owns the string
/// mapping.
fn parse_track_source(s: &str) -> Result<TrackSource, String> {
    match s {
        "union-find" => Ok(TrackSource::UnionFind),
        "graph" => Ok(TrackSource::CorrespondenceGraph),
        other => Err(format!(
            "unknown --track-source {other:?} (expected union-find|graph)"
        )),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut features_dir = None;
    let mut feature_suffix = String::from("_features.txt");
    let mut image_suffix = String::from(".png");
    let mut out_colmap = None;
    let (mut width, mut height) = (None, None);
    let (mut fx, mut fy, mut cx, mut cy) = (None, None, None, None);
    let mut vocab_size = 64usize;
    let mut retrieval_topk = 12usize;
    let mut exhaustive = false;
    let mut match_ratio = 0.8f32;
    let mut min_matches = 30usize;
    let mut min_pnp_inliers = 12usize;
    let mut max_reproj = 4.0f64;
    let mut final_ba = true;
    let mut seed_trials = 12usize;
    let mut refine_intrinsics = false;
    let mut refine_distortion = false;
    let mut colmap_style = false;
    let mut filter_images = false;
    let mut verification_mode = VerificationMode::Legacy;
    let mut track_source = TrackSource::UnionFind;

    let mut a: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--features-dir" => features_dir = Some(PathBuf::from(a.remove(i + 1))),
            "--feature-suffix" => feature_suffix = a.remove(i + 1),
            "--image-suffix" => image_suffix = a.remove(i + 1),
            "--out-colmap" => out_colmap = Some(PathBuf::from(a.remove(i + 1))),
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
            "--match-ratio" => match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-matches" => min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-pnp-inliers" => {
                min_pnp_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--max-reproj" => max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--no-final-ba" => final_ba = false,
            "--seed-trials" => seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--refine-intrinsics" => refine_intrinsics = true,
            "--refine-distortion" => refine_distortion = true,
            "--colmap-style" => colmap_style = true,
            "--filter-images" => filter_images = true,
            "--colmap-verification" => verification_mode = VerificationMode::Full,
            "--verification-mode" => {
                verification_mode = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--track-source" => track_source = parse_track_source(&a.remove(i + 1))?,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    let width = width.ok_or("--width is required")?;
    let height = height.ok_or("--height is required")?;
    let camera = Camera::pinhole(
        0,
        width,
        height,
        fx.ok_or("--fx is required")?,
        fy.ok_or("--fy is required")?,
        cx.ok_or("--cx is required")?,
        cy.ok_or("--cy is required")?,
    );

    Ok(Args {
        features_dir: features_dir.ok_or("--features-dir is required")?,
        feature_suffix,
        image_suffix,
        out_colmap: out_colmap.ok_or("--out-colmap is required")?,
        camera,
        vocab_size,
        retrieval_topk,
        exhaustive,
        match_ratio,
        min_matches,
        min_pnp_inliers,
        max_reproj,
        final_ba,
        seed_trials,
        refine_intrinsics,
        refine_distortion,
        colmap_style,
        filter_images,
        verification_mode,
        track_source,
    })
}

fn image_name_for(feat_filename: &str, feat_suffix: &str, image_suffix: &str) -> String {
    match feat_filename.strip_suffix(feat_suffix) {
        Some(stem) => format!("{stem}{image_suffix}"),
        None => feat_filename.to_string(),
    }
}

/// Read every `*<feature_suffix>` file in `dir`, sorted lexically, returning the
/// per-image feature sets and their COLMAP image names.
fn load_images(
    dir: &Path,
    feature_suffix: &str,
    image_suffix: &str,
) -> Result<(Vec<FeatureSet>, Vec<String>), Box<dyn std::error::Error>> {
    let mut files: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(feature_suffix))
        .collect();
    files.sort();
    let mut features = Vec::new();
    let mut names = Vec::new();
    for f in &files {
        features.push(read_external_deep_features_txt(dir.join(f))?.into_feature_set()?);
        names.push(image_name_for(f, feature_suffix, image_suffix));
    }
    Ok((features, names))
}

/// Candidate image pairs `(i, j)` with `i < j` from VLAD retrieval (or all
/// pairs when `exhaustive`).
fn candidate_pairs(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    let n = features.len();
    if exhaustive || n <= topk + 1 {
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                pairs.push((i, j));
            }
        }
        return pairs;
    }

    // Build the vocabulary from a bounded, deterministic descriptor sample —
    // k-means over *every* descriptor (262 k for 128×2048-kpt images) is the
    // pipeline's bottleneck and unnecessary: a VLAD vocabulary only needs a
    // representative sample. Stride the full descriptor list down to ~VOCAB_SAMPLE.
    const VOCAB_SAMPLE: usize = 40_000;
    let all_desc: Vec<&[f32]> = features
        .iter()
        .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
        .collect();
    let stride = (all_desc.len() / VOCAB_SAMPLE).max(1);
    let sample: Vec<&[f32]> = all_desc.iter().step_by(stride).copied().collect();
    let Some(vocab) = Vocabulary::build(&sample, vocab_size, 10, 0) else {
        // Fall back to exhaustive if the vocabulary cannot be built.
        return candidate_pairs(features, vocab_size, topk, true);
    };
    let globals: Vec<Vec<f32>> = features
        .iter()
        .map(|f| vlad(&f.descriptors, &vocab))
        .collect();

    let mut set: std::collections::BTreeSet<(usize, usize)> = std::collections::BTreeSet::new();
    for i in 0..n {
        let mut sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, cosine_similarity(&globals[i], &globals[j])))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for &(j, _) in sims.iter().take(topk) {
            set.insert((i.min(j), i.max(j)));
        }
    }
    set.into_iter().collect()
}

/// Per-`ConfigurationType` pair counts from the COLMAP-style verifier, for
/// the M1 acceptance experiment's pair-rejection report (how many VLAD
/// candidate pairs got reclassified away from a naive essential-matrix
/// accept). Unused (stays all-zero) when `--colmap-verification` is off.
#[derive(Debug, Default, Clone, Copy)]
struct VerificationStats {
    calibrated: usize,
    uncalibrated: usize,
    planar: usize,
    panoramic: usize,
    planar_or_panoramic: usize,
    watermark: usize,
    degenerate: usize,
    multiple: usize,
}

impl VerificationStats {
    fn record(&mut self, config: ConfigurationType) {
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

    fn total(&self) -> usize {
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
///   instead: `DEGENERATE`, `WATERMARK`, `PANORAMIC` (pure rotation — no
///   baseline to seed or triangulate from), and unresolved
///   `PLANAR_OR_PANORAMIC` pairs are dropped entirely rather than handed to
///   `incremental_sfm`; `CALIBRATED` / `UNCALIBRATED` / `PLANAR` / `MULTIPLE`
///   pairs keep their winning model's own inliers (which need not be the
///   essential matrix's).
fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
    mode: VerificationMode,
) -> (Vec<PairwiseMatches>, VerificationStats) {
    let verifier = (mode == VerificationMode::Full)
        .then(|| TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0)));
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
        }
    });

    let results: Vec<(Option<PairwiseMatches>, Option<ConfigurationType>)> = candidates
        .par_iter()
        .map(|&(i, j)| {
            let matcher = CrossCheckMatcher::new(BruteForceMatcher {
                ratio: Some(match_ratio),
            });
            let dm = matcher.match_descriptors(&features[i].descriptors, &features[j].descriptors);
            if dm.len() < min_matches {
                return (None, None);
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
                let keep = matches!(
                    report.config,
                    ConfigurationType::Calibrated
                        | ConfigurationType::Uncalibrated
                        | ConfigurationType::Planar
                        | ConfigurationType::Multiple
                );
                if !keep || report.inliers.len() < min_matches {
                    return (None, Some(report.config));
                }
                let matches: Vec<(usize, usize)> = report
                    .inliers
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect();
                (
                    Some(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                    }),
                    Some(report.config),
                )
            } else {
                let estimator = match &threshold_only_estimator {
                    Some(e) => *e,
                    None => RelativePoseEstimator::default(),
                };
                let Some(rel) = estimator.estimate(&corrs, camera) else {
                    return (None, None);
                };
                if rel.inliers.len() < min_matches {
                    return (None, None);
                }
                let matches: Vec<(usize, usize)> = rel
                    .inliers
                    .iter()
                    .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                    .collect();
                (
                    Some(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                    }),
                    None,
                )
            }
        })
        .collect();

    let mut stats = VerificationStats::default();
    let mut pairwise = Vec::with_capacity(results.len());
    for (pair, config) in results {
        if let Some(config) = config {
            stats.record(config);
        }
        if let Some(pair) = pair {
            pairwise.push(pair);
        }
    }
    (pairwise, stats)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\nsee the file header for usage.");
            std::process::exit(2);
        }
    };

    let (features, image_names) =
        load_images(&args.features_dir, &args.feature_suffix, &args.image_suffix)?;
    if features.len() < 2 {
        return Err(format!("need ≥2 images, found {}", features.len()).into());
    }
    let total_kp: usize = features.iter().map(|f| f.keypoints.len()).sum();
    println!(
        "loaded {} images, {} keypoints total, camera {}x{}",
        features.len(),
        total_kp,
        args.camera.width,
        args.camera.height,
    );

    let candidates = candidate_pairs(
        &features,
        args.vocab_size,
        args.retrieval_topk,
        args.exhaustive,
    );
    println!(
        "view graph: {} candidate pairs ({})",
        candidates.len(),
        if args.exhaustive {
            "exhaustive"
        } else {
            "VLAD top-k"
        },
    );

    let (pairwise, verification_stats) = verify_pairs(
        &features,
        &args.camera,
        &candidates,
        args.match_ratio,
        args.min_matches,
        args.verification_mode,
    );
    let verified_matches: usize = pairwise.iter().map(|p| p.matches.len()).sum();
    println!(
        "verified {} / {} pairs, {} inlier correspondences",
        pairwise.len(),
        candidates.len(),
        verified_matches,
    );
    if args.verification_mode == VerificationMode::Full {
        println!(
            "colmap-style verification: {} pairs classified (CALIBRATED={} UNCALIBRATED={} \
             PLANAR={} PANORAMIC={} PLANAR_OR_PANORAMIC={} WATERMARK={} DEGENERATE={} MULTIPLE={})",
            verification_stats.total(),
            verification_stats.calibrated,
            verification_stats.uncalibrated,
            verification_stats.planar,
            verification_stats.panoramic,
            verification_stats.planar_or_panoramic,
            verification_stats.watermark,
            verification_stats.degenerate,
            verification_stats.multiple,
        );
    }
    if pairwise.is_empty() {
        return Err("no pair survived geometric verification — lower --min-matches?".into());
    }

    let config = IncrementalSfmConfig {
        min_seed_matches: args.min_matches,
        min_pnp_inliers: args.min_pnp_inliers,
        max_reprojection_error_px: args.max_reproj,
        final_global_ba: args.final_ba,
        seed_trials: args.seed_trials,
        // Distortion self-calibration runs inside the joint intrinsics BA, so it
        // implies intrinsics refinement; the (k1, k2) flag rides on `ba_config`.
        refine_intrinsics: args.refine_intrinsics || args.refine_distortion,
        ba_config: BaConfig {
            refine_distortion: args.refine_distortion,
            ..IncrementalSfmConfig::default().ba_config
        },
        colmap_style_mapper: args.colmap_style,
        filter_images: args.filter_images,
        track_source: args.track_source,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&args.camera, &features, &pairwise, &config)?;
    println!(
        "reconstruction ({}): {} / {} images registered, {} tracks, mean reproj {:.3} px",
        match args.track_source {
            TrackSource::UnionFind => "track-source=union-find",
            TrackSource::CorrespondenceGraph => "track-source=graph",
        },
        result.registered_images,
        features.len(),
        result.tracks.len(),
        result.mean_reprojection_px,
    );

    // When intrinsics were refined, export with the refined camera (and report the
    // before→after pull — on observable, wide-parallax capture this is where focal
    // length is recoverable, unlike low-parallax forward video).
    let export_camera = result.refined_camera.clone().unwrap_or(args.camera.clone());
    if let (Some(i0), Some(i1)) = (args.camera.intrinsics(), export_camera.intrinsics()) {
        if result.refined_camera.is_some() {
            println!(
                "refined intrinsics: fx {:.2}->{:.2}  fy {:.2}->{:.2}  cx {:.2}->{:.2}  cy {:.2}->{:.2}",
                i0.0, i1.0, i0.1, i1.1, i0.2, i1.2, i0.3, i1.3,
            );
            if let Some((k1, k2)) = export_camera.radial_distortion() {
                let (k1_0, k2_0) = args.camera.radial_distortion().unwrap_or((0.0, 0.0));
                println!("refined distortion: k1 {k1_0:.5}->{k1:.5}  k2 {k2_0:.5}->{k2:.5}");
            }
        }
    }

    // Compact to registered images (the COLMAP writer expects a dense pose list)
    // and remap each track observation's image index.
    let registered: Vec<usize> = (0..features.len())
        .filter(|&i| result.poses[i].is_some())
        .collect();
    let remap: HashMap<usize, usize> = registered
        .iter()
        .enumerate()
        .map(|(new_idx, &old)| (old, new_idx))
        .collect();
    let poses_out: Vec<Pose> = registered
        .iter()
        .map(|&i| result.poses[i].clone().unwrap())
        .collect();
    let features_out: Vec<FeatureSet> = registered.iter().map(|&i| features[i].clone()).collect();
    let names_out: Vec<String> = registered.iter().map(|&i| image_names[i].clone()).collect();
    let landmarks_out: Vec<ExportLandmark> = result
        .tracks
        .iter()
        .map(|t| {
            let obs = t
                .observations
                .iter()
                .filter_map(|&(img, kp, px)| remap.get(&img).map(|&ni| (ni, kp, px)))
                .collect();
            (t.position, obs)
        })
        .collect();

    let summary = write_colmap_reconstruction_for_3dgs(
        &args.out_colmap,
        &export_camera,
        &poses_out,
        &features_out,
        &landmarks_out,
        |k| names_out[k].clone(),
    )?;
    println!(
        "wrote COLMAP model to {} ({} images, {} points, {} observations)",
        args.out_colmap.display(),
        summary.frame_count,
        summary.landmark_count,
        summary.observation_count,
    );
    Ok(())
}
