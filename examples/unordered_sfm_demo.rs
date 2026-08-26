//! Incremental structure-from-motion from an **unordered** image set — the
//! COLMAP-style SfM pillar of visloc-rs.
//!
//! Unlike the stereo-VO SfM path (`--sfm-colmap-out` on
//! `stereo_vo_external_deep_files`), which needs an *ordered* video with
//! frame→frame matches, this demo takes a directory of per-image deep features
//! with **no temporal order**, builds its own view graph, and grows one
//! reconstruction:
//!
//! 1. **View graph.** `--pair-source` (default `vlad`) selects how candidate
//!    pairs are proposed:
//!    - `vlad`: a VLAD vocabulary over all descriptors gives each image a
//!      global descriptor; the top-K most similar images per image become
//!      candidate pairs.
//!    - `vocab-tree`: `visloc_rs::vision::vocab_tree`'s hierarchical-k-means
//!      vocabulary + TF-IDF/Hamming-embedding inverted-file retrieval
//!      (COLMAP's `VocabTreePairGenerator`-equivalent, M3 in
//!      `docs/colmap_port_plan.md`) — `--vocab-tree-branching`/
//!      `--vocab-tree-depth` size the tree, `--vocab-tree-num-images` is the
//!      top-N retrieved per query image (COLMAP default 100).
//!
//!    `--exhaustive` overrides either source with all pairs.
//! 2. **Verified matches.** Each candidate pair is matched — `--matcher`
//!    (default `nn`) selects the algorithm:
//!    - `nn`: cross-checked brute-force nearest-neighbour + Lowe ratio
//!      (pre-M6 behaviour, unchanged).
//!    - `lightglue` (M6, `docs/colmap_port_plan.md`): the learned LightGlue
//!      matcher (SuperPoint variant), run in-process via ONNX Runtime
//!      (`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`,
//!      `--lightglue-model PATH`, `onnx-inference` feature required). Unlike
//!      NN+ratio's independent per-descriptor search, LightGlue attends over
//!      *both* images' descriptors jointly — the lever M5's diagnosis
//!      motivated: ETH3D `courtyard`'s cross-component bridge pairs carry
//!      real but sparse correspondence signal that a per-descriptor ratio
//!      test cannot safely extract on a repeated-texture scene (M5's "naive
//!      rescue" experiment found classifier-passing *false* bridges from
//!      over-relaxing the ratio test — evidence the matcher itself, not just
//!      its threshold, was the bottleneck). `--matcher lightglue` replaces
//!      the matching step in *both* the main pass below and the M5
//!      rescue-bridging pass (step 4); `--rescue-match-ratio`/
//!      `--rescue-cross-check` are `nn`-only knobs, ignored under
//!      `lightglue` (see `PairMatcher::match_pair`'s doc comment). One ONNX
//!      graph is exported per camera resolution
//!      (`scripts/export_lightglue_onnx.py --width --height`); re-export for
//!      a different scene's intrinsics.
//!
//!    Every matched pair is then geometrically verified per
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
//!      `src/colmap/estimators/two_view_geometry.cc`): only `DEGENERATE` and
//!      `WATERMARK` pairs are dropped before `incremental_sfm` ever sees
//!      them, matching COLMAP's real admission gate
//!      (`database_cache.cc`'s `UseInlierMatchesCheck`, M2.1 — see
//!      `docs/colmap_port_plan.md`). `PANORAMIC` (pure rotation, no
//!      triangulatable baseline) and unresolved `PLANAR_OR_PANORAMIC` pairs
//!      *do* contribute their homography inliers to `PairwiseMatches`, same
//!      as `PLANAR`; they just never become a *seed* pair, because
//!      `pipelines/slam/src/incremental_sfm.rs`'s own parallax gate
//!      (`place_seed_pair`) independently rejects near-zero-baseline pairs at
//!      growth time — the same "recompute and gate on triangulation angle,
//!      don't consult the stored classification" design COLMAP's own
//!      `IncrementalMapperImpl::EstimateInitialTwoViewGeometry` uses. Every
//!      other configuration (`CALIBRATED`/`UNCALIBRATED`/`PLANAR`/`MULTIPLE`)
//!      keeps its winning model's own inliers (which need not be the
//!      essential matrix's).
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
//! 4. **Rescue-bridging (opt-in, `--rescue-bridging`, M5 in
//!    `docs/colmap_port_plan.md`).** Runs after the initial verification pass
//!    above. Detects whether the verified-pair graph is disconnected
//!    (`visloc_rs::vision::two_view::connected_components`) — the diagnosed
//!    ETH3D `courtyard` failure mode (images 0-24 vs 25-37 never verify a
//!    single pair against each other at any pair budget M3/M4 tried). If so,
//!    it proposes cross-component candidate pairs, ranked by a fresh VLAD
//!    global-descriptor similarity and budget-capped
//!    (`generate_bridge_candidates`), rematches each with a deliberately
//!    relaxed profile (`--rescue-match-ratio`, default a looser Lowe ratio
//!    than `--match-ratio`, and mutual-NN instead of strict cross-check
//!    unless `--rescue-cross-check` is set), and re-verifies every candidate
//!    with the *same* full [`TwoViewGeometryVerifier`] every other pair goes
//!    through — a relaxed matcher only ever *proposes* a bridge, the
//!    classifier still decides what's *admitted* (the M1.1 lesson: loose
//!    thresholds are only safe when a real classifier gates the result).
//!    Admitted pairs are appended to the same `PairwiseMatches` list that
//!    feeds `incremental_sfm`, so a successful bridge participates in track
//!    building exactly like any other verified pair.
//! 5. **Export.** The registered poses + merged multi-view tracks are written as
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

use std::collections::{HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3};
use rayon::prelude::*;
#[cfg(feature = "onnx-inference")]
use visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher;
use visloc_rs::vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_rs::vision::two_view::{
    connected_components, generate_bridge_candidates, BridgeCandidateOptions, ConfigurationType,
    EightPointEssentialMatrixEstimator, EssentialMatrixEstimator, EssentialRansac,
    EssentialRansacConfig, RelativePoseEstimator, TwoViewCorrespondence, TwoViewGeometryOptions,
    TwoViewGeometryVerifier,
};
use visloc_rs::vision::vocab_tree::{
    generate_pairs, HkmBuildOptions, VocabTree, VocabTreeOptions, VocabTreePairGeneratorOptions,
};
use visloc_rs::{
    incremental_sfm, read_external_deep_features_txt, reconstruct_global_sfm,
    write_colmap_reconstruction_for_3dgs, BaConfig, BruteForceMatcher, Camera, CrossCheckMatcher,
    DescriptorMatch, FeatureSet, GlobalReconstructionTuning, IncrementalSfmConfig, Matcher,
    PairwiseMatches, Pose, TrackSource,
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
enum PairSource {
    /// Flat-VLAD top-K cosine retrieval (pre-M3 behaviour, unchanged).
    Vlad,
    /// Hierarchical-k-means vocab-tree retrieval (M3).
    VocabTree,
    /// COLMAP's `TransitivePairGenerator` port: propose pairs through the
    /// *verified-match* graph — images that share a matched partner but have
    /// no direct pair yet get proposed (`pairing.cc`). Runs a vocab-tree
    /// base pass, then expands transitively for
    /// [`TRANSITIVE_ROUNDS`] rounds.
    Transitive,
}

impl std::str::FromStr for PairSource {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vlad" => Ok(Self::Vlad),
            "vocab-tree" => Ok(Self::VocabTree),
            "transitive" => Ok(Self::Transitive),
            other => Err(format!(
                "unknown --pair-source {other:?} (expected vlad|vocab-tree|transitive)"
            )),
        }
    }
}

/// COLMAP-style guided matching (`FeaturePairsMatching`'s
/// `FindGuidedMatches`): given the pair's verified essential geometry,
/// rematch descriptors that the initial NN+ratio pass missed under an
/// epipolar constraint. For every not-yet-matched query descriptor the best
/// unused train descriptor is accepted only when **both** the Lowe ratio
/// (`0.9`, looser than the main pass) and the squared Sampson distance
/// (`guided_max_error_px`) pass — pure geometric admission without a ratio
/// gate is what produced M5's false-bridge failure, so this stays
/// deliberately conservative. Conflicts (two queries claiming one train)
/// resolve to the smaller descriptor distance, greedy by distance order.
fn guided_epipolar_matches(
    camera: &Camera,
    features_i: &FeatureSet,
    features_j: &FeatureSet,
    initial: &[DescriptorMatch],
    inlier_corrs: &[TwoViewCorrespondence],
    max_error_px: f64,
) -> Vec<DescriptorMatch> {
    // Essential matrix from the verified inliers (normalized eight-point).
    let Some(essential) = EssentialMatrixEstimator::estimate(
        &EightPointEssentialMatrixEstimator::default(),
        inlier_corrs,
        camera,
    ) else {
        return Vec::new();
    };
    let (fx, fy, _, _) = camera.intrinsics().unwrap_or((1.0, 1.0, 0.0, 0.0));
    let focal = 0.5 * (fx + fy);
    let max_sq_norm = (max_error_px / focal).powi(2);

    let normalize_all = |keypoints: &[Point2<f64>]| -> Vec<Option<[f64; 3]>> {
        keypoints
            .iter()
            .map(|p| camera.normalize_pixel(p).map(|n| [n.x, n.y, 1.0]))
            .collect()
    };
    let norm_i = normalize_all(&features_i.keypoints);
    let norm_j = normalize_all(&features_j.keypoints);
    let sampson_sq = |ni: &[f64; 3], nj: &[f64; 3]| -> Option<f64> {
        let e_ni = essential * nalgebra::Vector3::new(ni[0], ni[1], ni[2]);
        let et_nj = essential.transpose() * nalgebra::Vector3::new(nj[0], nj[1], nj[2]);
        let numerator = nalgebra::Vector3::new(nj[0], nj[1], nj[2])
            .dot(&e_ni)
            .powi(2);
        let denominator = e_ni.x * e_ni.x + e_ni.y * e_ni.y + et_nj.x * et_nj.x + et_nj.y * et_nj.y;
        if denominator < 1e-18 {
            None
        } else {
            Some(numerator / denominator)
        }
    };

    let mut used_query = vec![false; features_i.descriptors.len()];
    let mut used_train = vec![false; features_j.descriptors.len()];
    for m in initial {
        used_query[m.query_index] = true;
        used_train[m.train_index] = true;
    }

    // Descriptor-distance matrix over the full pair (one GEMM), rows =
    // queries, cols = trains.
    let n_q = features_i.descriptors.len();
    let n_t = features_j.descriptors.len();
    if n_q == 0 || n_t == 0 || features_i.descriptors[0].is_empty() {
        return Vec::new();
    }
    let dim = features_i.descriptors[0].len();
    let q = nalgebra::DMatrix::from_fn(n_q, dim, |a, k| features_i.descriptors[a][k] as f64);
    let t = nalgebra::DMatrix::from_fn(n_t, dim, |b, k| features_j.descriptors[b][k] as f64);
    let dist = &q * &t.transpose();

    struct Candidate {
        query: usize,
        train: usize,
        distance: f32,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for qi in 0..n_q {
        if used_query[qi] || norm_i[qi].is_none() {
            continue;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut second: f64 = f64::INFINITY;
        for tj in 0..n_t {
            if used_train[tj] {
                continue;
            }
            let d = ((dist[(qi, tj)]).max(0.0)).sqrt();
            if d < second {
                if d < best.map_or(f64::INFINITY, |(_, bd)| bd) {
                    second = best.map_or(f64::INFINITY, |(_, bd)| bd);
                    best = Some((tj, d));
                } else {
                    second = d;
                }
            }
        }
        let Some((tj, d)) = best else { continue };
        if d <= 0.0 || d >= second {
            continue;
        }
        if d / second > 0.8 {
            continue;
        }
        let Some(nj) = norm_j[tj] else { continue };
        let Some(sq) = sampson_sq(&norm_i[qi].unwrap(), &nj) else {
            continue;
        };
        if sq <= max_sq_norm {
            candidates.push(Candidate {
                query: qi,
                train: tj,
                distance: d as f32,
            });
        }
    }
    candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    let mut taken_train = used_train;
    let mut out = Vec::new();
    for c in candidates {
        if taken_train[c.train] {
            continue;
        }
        taken_train[c.train] = true;
        out.push(DescriptorMatch {
            query_index: c.query,
            train_index: c.train,
            distance: c.distance,
            second_best_distance: None,
            ratio: None,
            confidence: None,
        });
    }
    out
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
enum MatcherKind {
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

struct Args {
    /// `Files` (default): read precomputed `X Y SCORE D…` feature files from
    /// `features_dir`. `Sift`: run the pure-Rust SIFT frontend in-process on
    /// every image in `images_dir` (requires `--images-dir`; ignores
    /// `--features-dir`). See `visloc_vision::features::sift`.
    feature_extractor: FeatureExtractorKind,
    features_dir: PathBuf,
    images_dir: Option<PathBuf>,
    feature_suffix: String,
    image_suffix: String,
    sift_max_keypoints: usize,
    /// Enable SIFT affine shape adaptation (descriptor-side Baumberg).
    sift_affine: bool,
    /// Interest-point operator: `dog` (default) or `hessian-laplace`.
    sift_detector: String,
    /// Multi-anisotropy detection proposals (requires `--sift-affine`).
    sift_multi_anisotropy: bool,
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
    structureless_registration: bool,
    /// Raise above 128 to opt into the COLMAP-style confidence-based
    /// adaptive PnP RANSAC budget for large correspondence sets.
    pnp_max_iterations: usize,
    filter_images: bool,
    verification_mode: VerificationMode,
    /// COLMAP-style guided matching: after a pair verifies, rematch
    /// descriptors missed by the initial NN+ratio pass under the verified
    /// epipolar geometry, then re-verify. Off by default (byte-identical
    /// legacy behaviour when off).
    guided_matching: bool,
    /// `Incremental` (default): the existing grow-from-seed mapper.
    /// `Global`: GLOMAP-style — per-pair essential relative poses, rotation +
    /// position averaging, track triangulation, one joint BA
    /// (`visloc_slam::global_sfm::reconstruct_global_sfm`).
    mapper: MapperKind,
    /// Global mapper only: harden essential cheirality (min tri-angle,
    /// ambiguity rejection). Default off = byte-identical legacy edges.
    chirality_harden: bool,
    /// Global mapper only: try this many high-degree rotation seeds and keep
    /// the best. `1` = legacy single-seed.
    rotation_seed_trials: usize,
    /// Global mapper: re-estimate edge translations under consensus rotations.
    refine_global_translations: bool,
    /// M2 A/B switch: which algorithm builds feature tracks from the verified
    /// pairs (`docs/colmap_port_plan.md`'s M2 milestone) — the legacy ad hoc
    /// union-find (default) or COLMAP's persistent `CorrespondenceGraph`.
    track_source: TrackSource,
    /// M3 A/B switch: which candidate-pair source feeds verification — flat
    /// VLAD top-K (default) or the hierarchical vocab-tree
    /// (`docs/colmap_port_plan.md`'s M3 milestone).
    pair_source: PairSource,
    /// Vocab-tree hierarchical-k-means branching factor (M3; ignored under
    /// `--pair-source vlad`). See `vocab_tree::hkm::HkmBuildOptions`.
    vocab_tree_branching: usize,
    /// Vocab-tree hierarchical-k-means depth (M3; ignored under
    /// `--pair-source vlad`).
    vocab_tree_depth: usize,
    /// Vocab-tree pair generator's `num_images` (top-N retrieved per query
    /// image before dedup) — COLMAP default 100
    /// (`VocabTreePairingOptions::num_images`). Ignored under
    /// `--pair-source vlad`.
    vocab_tree_num_images: usize,
    /// M5 (`docs/colmap_port_plan.md`): run the opt-in rescue-bridging pass
    /// after initial verification (see the file header's step 4).
    rescue_bridging: bool,
    /// Rescue pass's relaxed Lowe ratio (looser than `--match-ratio`) — the
    /// M5 "matching relaxation" lever.
    rescue_match_ratio: f32,
    /// Rescue pass's minimum raw-match / verified-inlier floor. Deliberately
    /// independent of `--min-matches`: rescue candidates are, by
    /// construction, the pairs the main pass already couldn't reach, so this
    /// is the floor the M5 brief's "cheapest lever first" default should use
    /// (COLMAP's own `min_num_inliers` default, 15) rather than inheriting
    /// whatever (possibly stricter) floor the main pass used.
    rescue_min_matches: usize,
    /// Maximum number of cross-component candidate pairs the rescue pass will
    /// attempt (budget cap, `BridgeCandidateOptions::max_candidates`).
    rescue_max_candidates: usize,
    /// Whether the rescue pass's relaxed matcher also applies strict
    /// bidirectional cross-check (default `false`: mutual-NN + ratio only,
    /// per the M5 brief's "mutual-NN with Lowe ratio *instead of* ... strict
    /// cross-check").
    rescue_cross_check: bool,
    /// M5 diagnosis tool (`--diagnose-pair I,J`, repeatable): dump raw match
    /// counts and verification outcomes for specific `(i, j)` image-index
    /// pairs across a battery of matching profiles, then exit without
    /// running the reconstruction. Used to inspect the exact bridge
    /// candidates the M5 brief asks for (e.g. the boundary pair) by hand.
    diagnose_pairs: Vec<(usize, usize)>,
    /// M6 (`docs/colmap_port_plan.md`): which algorithm produces raw
    /// descriptor matches for a candidate pair, before two-view verification
    /// — `nn` (default, pre-M6 NN+ratio behaviour) or `lightglue` (learned
    /// joint matcher, `onnx-inference`-gated). See [`MatcherKind`].
    matcher: MatcherKind,
    /// Path to the exported LightGlue ONNX graph (`--matcher lightglue`
    /// only; see `scripts/export_lightglue_onnx.py`). One graph per camera
    /// resolution — re-export for a different `--width`/`--height`. Only
    /// read from [`build_matcher`]'s `onnx-inference`-gated branch — the
    /// `#[allow(dead_code)]` covers the default (feature-off) build, where
    /// `--matcher lightglue` is rejected before this field would be read.
    #[cfg_attr(not(feature = "onnx-inference"), allow(dead_code))]
    lightglue_model: Option<PathBuf>,
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
    let mut structureless_registration = false;
    let mut guided_matching = false;
    let mut pnp_max_iterations = 128usize;
    let mut filter_images = false;
    let mut verification_mode = VerificationMode::Legacy;
    let mut track_source = TrackSource::UnionFind;
    let mut pair_source = PairSource::Vlad;
    let mut vocab_tree_branching = 10usize;
    let mut vocab_tree_depth = 3usize;
    let mut vocab_tree_num_images = 100usize;
    let mut rescue_bridging = false;
    let mut rescue_match_ratio = 0.95f32;
    let mut rescue_min_matches = 15usize;
    let mut rescue_max_candidates = 200usize;
    let mut rescue_cross_check = false;
    let mut diagnose_pairs: Vec<(usize, usize)> = Vec::new();
    let mut matcher = MatcherKind::Nn;
    let mut lightglue_model: Option<PathBuf> = None;
    let mut feature_extractor = FeatureExtractorKind::Files;
    let mut mapper = MapperKind::Incremental;
    let mut chirality_harden = false;
    let mut rotation_seed_trials = 1usize;
    let mut refine_global_translations = false;
    let mut images_dir: Option<PathBuf> = None;
    let mut sift_max_keypoints = 2048usize;
    let mut sift_affine = false;
    let mut sift_detector = String::from("dog");
    let mut sift_multi_anisotropy = false;

    let mut a: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < a.len() {
        match a[i].as_str() {
            "--features-dir" => features_dir = Some(PathBuf::from(a.remove(i + 1))),
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
            "--sift-detector" => sift_detector = a.remove(i + 1),
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
            "--structureless-registration" => structureless_registration = true,
            "--guided-matching" => guided_matching = true,
            "--pnp-max-iterations" => {
                pnp_max_iterations = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--mapper" => {
                mapper = match a.remove(i + 1).as_str() {
                    "incremental" => MapperKind::Incremental,
                    "global" => MapperKind::Global,
                    other => {
                        return Err(format!("--mapper must be incremental|global, got {other}"))
                    }
                };
            }
            "--chirality-harden" => chirality_harden = true,
            "--rotation-seed-trials" => {
                rotation_seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--refine-global-translations" => refine_global_translations = true,
            "--filter-images" => filter_images = true,
            "--colmap-verification" => verification_mode = VerificationMode::Full,
            "--verification-mode" => {
                verification_mode = a.remove(i + 1).parse().map_err(|e: String| e)?
            }
            "--track-source" => track_source = parse_track_source(&a.remove(i + 1))?,
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
            "--diagnose-pair" => {
                let raw = a.remove(i + 1);
                let (lhs, rhs) = raw
                    .split_once(',')
                    .ok_or_else(|| format!("--diagnose-pair expects I,J, got {raw:?}"))?;
                let i_idx: usize = lhs.trim().parse().map_err(|e| format!("{e}"))?;
                let j_idx: usize = rhs.trim().parse().map_err(|e| format!("{e}"))?;
                diagnose_pairs.push((i_idx, j_idx));
            }
            "--matcher" => matcher = a.remove(i + 1).parse().map_err(|e: String| e)?,
            "--lightglue-model" => lightglue_model = Some(PathBuf::from(a.remove(i + 1))),
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
        feature_extractor,
        features_dir: features_dir.unwrap_or_default(),
        images_dir,
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
        structureless_registration,
        pnp_max_iterations,
        mapper,
        chirality_harden,
        rotation_seed_trials,
        refine_global_translations,
        filter_images,
        verification_mode,
        guided_matching,
        track_source,
        pair_source,
        vocab_tree_branching,
        vocab_tree_depth,
        vocab_tree_num_images,
        rescue_bridging,
        rescue_match_ratio,
        rescue_min_matches,
        rescue_max_candidates,
        rescue_cross_check,
        diagnose_pairs,
        matcher,
        lightglue_model,
        sift_max_keypoints,
        sift_affine,
        sift_detector,
        sift_multi_anisotropy,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MapperKind {
    Incremental,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FeatureExtractorKind {
    Files,
    Sift,
}

/// Extract SIFT features for one image path.
#[cfg(feature = "image-io")]
fn extract_sift_for_image(
    path: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
) -> Result<FeatureSet, Box<dyn std::error::Error>> {
    use visloc_rs::vision::features::sift::{extract_sift, GrayImage, SiftConfig, SiftDetector};
    let grayscale = visloc_io::images::read_common_image(path)?;
    let image = GrayImage::new(grayscale.width(), grayscale.height(), grayscale.pixels())?;
    let detector = match detector {
        "dog" => SiftDetector::Dog,
        "hessian-laplace" | "hessian" => SiftDetector::HessianLaplace,
        other => {
            return Err(format!("unknown --sift-detector {other} (dog|hessian-laplace)").into())
        }
    };
    let config = SiftConfig {
        max_keypoints,
        affine,
        detector,
        multi_anisotropy: multi_anisotropy && affine,
        ..SiftConfig::default()
    };
    let (keypoints, descriptors) = extract_sift(&image, &config)?;
    Ok(FeatureSet::new(
        keypoints.iter().map(|k| Point2::new(k.x, k.y)).collect(),
        descriptors,
    )?)
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(not(feature = "image-io"))]
fn load_images_with_sift(
    _dir: &Path,
    _max_keypoints: usize,
    _affine: bool,
    _detector: &str,
    _multi_anisotropy: bool,
) -> Result<(Vec<FeatureSet>, Vec<String>), Box<dyn std::error::Error>> {
    Err("--feature-extractor sift requires building with --features image-io".into())
}

/// In-process SIFT over every common-format image in `dir`, sorted lexically.
#[cfg(feature = "image-io")]
fn load_images_with_sift(
    dir: &Path,
    max_keypoints: usize,
    affine: bool,
    detector: &str,
    multi_anisotropy: bool,
) -> Result<(Vec<FeatureSet>, Vec<String>), Box<dyn std::error::Error>> {
    const IMAGE_SUFFIXES: [&str; 5] = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"];
    let mut files: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| {
            IMAGE_SUFFIXES
                .iter()
                .any(|suffix| n.to_lowercase().ends_with(suffix))
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(format!("no images found under {dir:?}").into());
    }
    let mut features = Vec::new();
    let mut names = Vec::new();
    for f in &files {
        features.push(extract_sift_for_image(
            &dir.join(f),
            max_keypoints,
            affine,
            detector,
            multi_anisotropy,
        )?);
        names.push(f.to_string());
    }
    Ok((features, names))
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

/// Bounded, deterministic descriptor sample for training a retrieval
/// vocabulary — k-means over *every* descriptor (262 k for 128×2048-kpt
/// images) is the pipeline's bottleneck and unnecessary for either VLAD or
/// the vocab-tree: both only need a representative sample. Strides the full
/// descriptor list down to ~`VOCAB_SAMPLE`. Shared by
/// [`candidate_pairs_vlad`] and [`candidate_pairs_vocab_tree`] (M3).
fn sampled_training_descriptors(features: &[FeatureSet]) -> Vec<&[f32]> {
    const VOCAB_SAMPLE: usize = 40_000;
    let all_desc: Vec<&[f32]> = features
        .iter()
        .flat_map(|f| f.descriptors.iter().map(|d| d.as_slice()))
        .collect();
    let stride = (all_desc.len() / VOCAB_SAMPLE).max(1);
    all_desc.iter().step_by(stride).copied().collect()
}

/// All `(i, j)` pairs with `i < j` — the exhaustive fallback shared by both
/// pair sources.
fn all_pairs(n: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            pairs.push((i, j));
        }
    }
    pairs
}

/// Candidate image pairs `(i, j)` with `i < j` from flat-VLAD top-K cosine
/// retrieval (or all pairs when `exhaustive`) — the pre-M3 pair source,
/// unchanged.
fn candidate_pairs_vlad(
    features: &[FeatureSet],
    vocab_size: usize,
    topk: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    let n = features.len();
    if exhaustive || n <= topk + 1 {
        return all_pairs(n);
    }

    let sample = sampled_training_descriptors(features);
    let Some(vocab) = Vocabulary::build(&sample, vocab_size, 10, 0) else {
        // Fall back to exhaustive if the vocabulary cannot be built.
        return all_pairs(n);
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

/// Candidate image pairs `(i, j)` with `i < j` from the M3 hierarchical
/// vocab-tree (`visloc_rs::vision::vocab_tree`, COLMAP's
/// `VocabTreePairGenerator`-equivalent, `docs/colmap_port_plan.md`'s M3
/// milestone), or all pairs when `exhaustive`.
///
/// Trains the hierarchical vocabulary on the same bounded descriptor sample
/// [`candidate_pairs_vlad`] uses, indexes every image's *full* descriptor
/// set (unsampled — retrieval quality for images the tree has never seen
/// depends on it having every one of their features, unlike the shared
/// training sample which only needs to be representative), then queries each
/// image against the finalized tree with its own descriptors, keeping the
/// top `vocab_tree_num_images` other images per query
/// ([`generate_pairs`]/[`VocabTreePairGeneratorOptions`]).
fn candidate_pairs_vocab_tree(
    features: &[FeatureSet],
    branching_factor: usize,
    depth: usize,
    num_images: usize,
    exhaustive: bool,
) -> Vec<(usize, usize)> {
    let n = features.len();
    if exhaustive {
        return all_pairs(n);
    }

    let sample = sampled_training_descriptors(features);
    let hkm_options = HkmBuildOptions {
        branching_factor,
        depth,
        ..HkmBuildOptions::default()
    };
    let vocab_tree_options = VocabTreeOptions::default();
    let Some(mut tree) = VocabTree::build(&sample, &hkm_options, &vocab_tree_options) else {
        // Fall back to exhaustive if the vocabulary cannot be built (mirrors
        // candidate_pairs_vlad's own degenerate-input fallback).
        return all_pairs(n);
    };
    for (i, f) in features.iter().enumerate() {
        tree.add_image(i, &f.descriptors);
    }
    tree.finalize();
    println!(
        "vocab-tree: {} leaf words (requested {}^{}={}), {} images indexed",
        tree.num_words(),
        branching_factor,
        depth,
        branching_factor.pow(depth as u32),
        tree.num_images(),
    );

    let image_descriptors: Vec<Vec<Vec<f32>>> =
        features.iter().map(|f| f.descriptors.clone()).collect();
    generate_pairs(
        &tree,
        &image_descriptors,
        &VocabTreePairGeneratorOptions { num_images },
    )
}

/// How many transitive-expansion rounds
/// ([`PairSource::Transitive`], COLMAP's `TransitivePairGenerator`) run
/// after the vocab-tree base pass. Two rounds cover the common
/// "bridge image chains a-b-c and b-d-e" real-scene topology; each round
/// only proposes pairs not proposed before, so cost is bounded by the
/// verified-graph neighbourhood size.
const TRANSITIVE_ROUNDS: usize = 2;

/// Candidate image pairs `(i, j)` with `i < j` — dispatches on
/// [`PairSource`] (`docs/colmap_port_plan.md`'s M3 A/B switch); `exhaustive`
/// overrides either source, matching pre-M3 behaviour.
/// [`PairSource::Transitive`] returns its *base* pass here (vocab-tree);
/// the transitive expansion happens in [`expand_transitive`] after those
/// base pairs are verified, mirroring COLMAP's generator running against
/// an existing match table.
fn candidate_pairs(features: &[FeatureSet], args: &Args) -> Vec<(usize, usize)> {
    match args.pair_source {
        PairSource::Vlad => candidate_pairs_vlad(
            features,
            args.vocab_size,
            args.retrieval_topk,
            args.exhaustive,
        ),
        PairSource::VocabTree | PairSource::Transitive => candidate_pairs_vocab_tree(
            features,
            args.vocab_tree_branching,
            args.vocab_tree_depth,
            args.vocab_tree_num_images,
            args.exhaustive,
        ),
    }
}

/// One round of COLMAP's `TransitivePairGenerator` (`src/colmap/pairing.cc`):
/// from the verified-match adjacency, propose every `(i, k)` with `i < k`
/// that shares a common matched partner `j` but has no direct pair yet.
fn expand_transitive(
    pairwise: &[PairwiseMatches],
    already_proposed: &HashSet<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in pairwise {
        if p.matches.is_empty() {
            continue;
        }
        neighbors.entry(p.image_i).or_default().insert(p.image_j);
        neighbors.entry(p.image_j).or_default().insert(p.image_i);
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut seen = already_proposed.clone();
    for (&i, ni) in &neighbors {
        for &j in ni {
            // Partners of partners.
            let Some(nj) = neighbors.get(&j) else {
                continue;
            };
            for &k in nj {
                if k == i {
                    continue;
                }
                let key = if i < k { (i, k) } else { (k, i) };
                if seen.insert(key) {
                    out.push(key);
                }
            }
        }
    }
    out.sort_unstable();
    out
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

    fn merge(&mut self, other: &VerificationStats) {
        self.calibrated += other.calibrated;
        self.uncalibrated += other.uncalibrated;
        self.planar += other.planar;
        self.panoramic += other.panoramic;
        self.planar_or_panoramic += other.planar_or_panoramic;
        self.watermark += other.watermark;
        self.degenerate += other.degenerate;
        self.multiple += other.multiple;
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

/// The M6 pair-matching backend, dispatched on [`MatcherKind`]. Holds the
/// loaded LightGlue ONNX session (cheap to `Clone`: it wraps an
/// `Arc<Mutex<ort::session::Session>>`, same as
/// [`visloc_rs::vision::features::lightglue_onnx::LightGlueOnnxMatcher`]'s
/// own doc comment explains) so it can be shared across the rayon-parallel
/// per-pair closures in [`verify_pairs`] and [`rescue_bridging`] without
/// re-loading the model.
enum PairMatcher {
    /// Pre-M6 nearest-neighbour + Lowe-ratio matcher.
    Nn,
    /// LightGlue (SuperPoint variant), in-process via ONNX Runtime.
    #[cfg(feature = "onnx-inference")]
    LightGlue(LightGlueOnnxMatcher),
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
    fn match_pair(
        &self,
        ratio: f32,
        cross_check: bool,
        features_i: &FeatureSet,
        features_j: &FeatureSet,
    ) -> Vec<DescriptorMatch> {
        match self {
            PairMatcher::Nn => {
                if cross_check {
                    CrossCheckMatcher::new(BruteForceMatcher { ratio: Some(ratio) })
                        .match_descriptors(&features_i.descriptors, &features_j.descriptors)
                } else {
                    BruteForceMatcher { ratio: Some(ratio) }
                        .match_descriptors(&features_i.descriptors, &features_j.descriptors)
                }
            }
            #[cfg(feature = "onnx-inference")]
            PairMatcher::LightGlue(matcher) => {
                match matcher.match_features(
                    &features_i.keypoints,
                    &features_i.descriptors,
                    &features_j.keypoints,
                    &features_j.descriptors,
                ) {
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

/// Build the [`PairMatcher`] `--matcher` selects. Fails fast (before any
/// pair is processed) if `--matcher lightglue` is requested without either
/// the `onnx-inference` feature compiled in or a `--lightglue-model` path.
fn build_matcher(args: &Args) -> Result<PairMatcher, Box<dyn std::error::Error>> {
    match args.matcher {
        MatcherKind::Nn => Ok(PairMatcher::Nn),
        MatcherKind::LightGlue => {
            #[cfg(feature = "onnx-inference")]
            {
                let path = args
                    .lightglue_model
                    .as_ref()
                    .ok_or("--matcher lightglue requires --lightglue-model PATH")?;
                let matcher = LightGlueOnnxMatcher::load_from_path(path).map_err(|error| {
                    format!("failed to load LightGlue ONNX model {path:?}: {error}")
                })?;
                Ok(PairMatcher::LightGlue(matcher))
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
#[allow(clippy::too_many_arguments)]
fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
    mode: VerificationMode,
    matcher: &PairMatcher,
    guided_matching: bool,
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
            ..RelativePoseEstimator::default()
        }
    });

    let results: Vec<(Option<PairwiseMatches>, Option<ConfigurationType>)> = candidates
        .par_iter()
        .map(|&(i, j)| {
            // `cross_check = true`: the main pass has always used bidirectional
            // mutual-NN confirmation (pre-M6 behaviour, unchanged for
            // `MatcherKind::Nn`); ignored entirely under `MatcherKind::LightGlue`
            // (see `PairMatcher::match_pair`'s doc comment).
            let dm = matcher.match_pair(match_ratio, true, &features[i], &features[j]);
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
                    return (None, Some(report.config));
                }
                // Guided matching (COLMAP FindGuidedMatches): expand the
                // match set under the verified epipolar geometry, then
                // re-verify so config/inliers describe the final set.
                let (dm, report) = if guided_matching {
                    let inlier_corrs: Vec<TwoViewCorrespondence> = report
                        .inliers
                        .iter()
                        .filter_map(|&idx| corrs.get(idx).copied())
                        .collect();
                    let extra = guided_epipolar_matches(
                        camera,
                        &features[i],
                        &features[j],
                        &dm,
                        &inlier_corrs,
                        2.0,
                    );
                    if extra.is_empty() {
                        (dm, report)
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
                            (expanded, new_report)
                        } else {
                            (dm, report)
                        }
                    }
                } else {
                    (dm, report)
                };
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
fn rescue_bridging(
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    args: &Args,
    matcher: &PairMatcher,
) -> Vec<PairwiseMatches> {
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
        return Vec::new();
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

    let candidates = generate_bridge_candidates(
        &components,
        similarity,
        &BridgeCandidateOptions {
            max_candidates: args.rescue_max_candidates,
        },
    );
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
            (
                Some(PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
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
        let mut all_edges = edges.clone();
        all_edges.extend(admitted.iter().map(|p| (p.image_i, p.image_j)));
        let components_after = connected_components(n, &all_edges);
        println!(
            "rescue-bridging: view graph now has {} connected component(s) after admission",
            components_after.len(),
        );
    }

    admitted
}

/// M5 diagnosis tool (`--diagnose-pair I,J`): dump raw match counts and
/// [`TwoViewGeometryVerifier`] outcomes for one specific `(i, j)` image pair
/// across a fixed battery of matching profiles — the strict main-pass
/// profile (ratio 0.8, cross-check), a mid profile (0.9, cross-check), and
/// the two rescue-pass extremes (0.95 with and without cross-check). Answers
/// the M5 brief's "for a handful of such pairs, dump the current match
/// counts and verification outcomes, and look at why they fail" directly,
/// on demand, for any specific candidate (e.g. the exact temporally-adjacent
/// boundary pair across a diagnosed component split).
fn diagnose_pair(features: &[FeatureSet], camera: &Camera, i: usize, j: usize) {
    println!("=== diagnose-pair ({i}, {j}) ===");
    let verifier = TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(camera, 4.0));
    for &(ratio, cross_check) in &[(0.8f32, true), (0.9, true), (0.95, true), (0.95, false)] {
        let dm = if cross_check {
            CrossCheckMatcher::new(BruteForceMatcher { ratio: Some(ratio) })
                .match_descriptors(&features[i].descriptors, &features[j].descriptors)
        } else {
            BruteForceMatcher { ratio: Some(ratio) }
                .match_descriptors(&features[i].descriptors, &features[j].descriptors)
        };
        if dm.len() < 8 {
            println!(
                "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} (too few to classify)",
                dm.len()
            );
            continue;
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
        println!(
            "  ratio={ratio:.2} cross_check={cross_check:<5} raw_matches={:<5} config={:?} inliers={}",
            dm.len(),
            report.config,
            report.inliers.len(),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\nsee the file header for usage.");
            std::process::exit(2);
        }
    };

    if args.feature_extractor == FeatureExtractorKind::Files
        && args.features_dir.as_os_str().is_empty()
    {
        return Err(
            "--features-dir is required (or use --feature-extractor sift with --images-dir)".into(),
        );
    }
    let (features, image_names) = match args.feature_extractor {
        FeatureExtractorKind::Files => {
            load_images(&args.features_dir, &args.feature_suffix, &args.image_suffix)?
        }
        FeatureExtractorKind::Sift => {
            let dir = args.images_dir.clone().unwrap_or_else(|| {
                eprintln!("error: --feature-extractor sift requires --images-dir");
                std::process::exit(2);
            });
            load_images_with_sift(
                &dir,
                args.sift_max_keypoints,
                args.sift_affine,
                &args.sift_detector,
                args.sift_multi_anisotropy,
            )?
        }
    };
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

    if !args.diagnose_pairs.is_empty() {
        for &(i, j) in &args.diagnose_pairs {
            diagnose_pair(&features, &args.camera, i, j);
        }
        return Ok(());
    }

    // M6 (`docs/colmap_port_plan.md`): built once, up front, so a bad
    // `--matcher lightglue` invocation (missing feature / missing model
    // path) fails immediately rather than after the (potentially expensive)
    // candidate-pair generation step below.
    let pair_matcher = build_matcher(&args)?;
    println!(
        "pair matcher: {}",
        match args.matcher {
            MatcherKind::Nn => "nn (NN + Lowe ratio)",
            MatcherKind::LightGlue => "lightglue (learned joint matcher, ONNX)",
        },
    );

    let candidates = candidate_pairs(&features, &args);
    println!(
        "view graph: {} candidate pairs ({})",
        candidates.len(),
        if args.exhaustive {
            "exhaustive"
        } else {
            match args.pair_source {
                PairSource::Vlad => "VLAD top-k",
                PairSource::VocabTree => "vocab-tree",
                PairSource::Transitive => "transitive (vocab-tree base)",
            }
        },
    );

    let (mut pairwise, mut verification_stats) = verify_pairs(
        &features,
        &args.camera,
        &candidates,
        args.match_ratio,
        args.min_matches,
        args.verification_mode,
        &pair_matcher,
        args.guided_matching,
    );
    if args.pair_source == PairSource::Transitive {
        let mut all_proposed: HashSet<(usize, usize)> = candidates.iter().copied().collect();
        for _ in 0..TRANSITIVE_ROUNDS {
            let extension = expand_transitive(&pairwise, &all_proposed);
            if extension.is_empty() {
                break;
            }
            println!("transitive expansion: {} new pairs", extension.len());
            extension.iter().for_each(|p| {
                all_proposed.insert(*p);
            });
            let (more, stats) = verify_pairs(
                &features,
                &args.camera,
                &extension,
                args.match_ratio,
                args.min_matches,
                args.verification_mode,
                &pair_matcher,
                args.guided_matching,
            );
            verification_stats.merge(&stats);
            pairwise.extend(more);
        }
    }
    let verified_matches: usize = pairwise.iter().map(|p| p.matches.len()).sum();
    println!(
        "verified {} / {} pairs, {} inlier correspondences",
        pairwise.len(),
        candidates.len(),
        verified_matches,
    );
    // M4 diagnosis probe (docs/colmap_port_plan.md): dump the raw verified-pair
    // image-index graph so the connected-component structure can be inspected
    // directly (temporary, env-gated; not part of the milestone's shipped
    // behaviour).
    if std::env::var_os("VISLOC_SFM_DEBUG_DUMP_PAIRS").is_some() {
        for p in &pairwise {
            eprintln!(
                "sfm-debug-pairs: {} {} matches={}",
                p.image_i,
                p.image_j,
                p.matches.len()
            );
        }
    }
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

    // M5 (`docs/colmap_port_plan.md`): opt-in rescue-bridging pass. Runs
    // after the standard verification above, strictly additive — admitted
    // bridge pairs are appended to `pairwise`, the same list `incremental_sfm`
    // consumes below, so a successful bridge participates in track building
    // exactly like any other verified pair.
    if args.rescue_bridging {
        let bridges = rescue_bridging(&features, &args.camera, &pairwise, &args, &pair_matcher);
        pairwise.extend(bridges);
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
        structureless_registration: args.structureless_registration,
        pnp_max_iterations: args.pnp_max_iterations,
        filter_images: args.filter_images,
        track_source: args.track_source,
        ..IncrementalSfmConfig::default()
    };
    if args.mapper == MapperKind::Global {
        let tuning = GlobalReconstructionTuning {
            min_pair_matches: args.min_matches,
            chirality_harden_edges: args.chirality_harden,
            rotation_seed_trials: args.rotation_seed_trials,
            refine_translations_with_global_rotations: args.refine_global_translations,
            ..GlobalReconstructionTuning::default()
        };
        let (poses, tracks, mean_reproj) =
            reconstruct_global_sfm(&args.camera, &features, &pairwise, &tuning, &config)?;
        let registered = poses.iter().filter(|p| p.is_some()).count();
        println!(
            "reconstruction: mapper=global: {} / {} images registered, {} tracks, mean reproj {:.3} px",
            registered,
            features.len(),
            tracks.len(),
            mean_reproj
        );
        // Compact the pose list to only registered images so the shared
        // COLMAP export path applies unchanged.
        let mut remap = HashMap::new();
        for (image, pose) in poses.iter().enumerate() {
            if pose.is_some() {
                remap.insert(image, remap.len());
            }
        }
        let poses_out: Vec<Pose> = poses
            .iter()
            .enumerate()
            .filter(|(i, p)| p.is_some() && remap.contains_key(i))
            .map(|(_, p)| p.clone().unwrap())
            .collect();
        let features_out: Vec<FeatureSet> = (0..features.len())
            .filter(|i| remap.contains_key(i))
            .map(|i| features[i].clone())
            .collect();
        let names_out: Vec<String> = (0..features.len())
            .filter(|i| remap.contains_key(i))
            .map(|i| image_names[i].clone())
            .collect();
        let landmarks_out: Vec<ExportLandmark> = tracks
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
            &args.camera,
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
        return Ok(());
    }
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
