//! Incremental structure-from-motion from an **ordered** image sequence — the
//! sequential-video counterpart to [`unordered_sfm_demo`].
//!
//! [`unordered_sfm_demo`] takes a directory of per-image deep features with **no
//! temporal order**, builds its view graph with a VLAD retrieval pass, and grows
//! one reconstruction. This demo instead assumes the images form an **ordered
//! video**: the feature files sort lexically into capture order, so the view
//! graph is the cheap, reliable thing COLMAP's `sequential_matcher` does — match
//! each frame `i` to its temporal neighbours `i+1 … i+window`, with optional
//! wider fixed-baseline skip offsets. The reusable generator can also merge
//! appearance and transitive hints; this demo's first slice supplies temporal
//! and skip policies only. The rest of the pipeline is identical to the
//! unordered demo:
//!
//! 1. **View graph.** For each frame `i`, the candidate pairs are
//!    `(i, i+1), (i, i+2), …, (i, i+window)`.
//! 2. **Verified matches.** Each candidate pair is matched (cross-checked
//!    brute-force + Lowe ratio) and geometrically verified by an
//!    essential-matrix RANSAC; the inliers become `PairwiseMatches`.
//! 3. **Incremental SfM.** [`visloc_rs::slam::incremental_sfm`] seeds from the
//!    strongest pair, registers images by PnP, triangulates tracks, and bundle-
//!    adjusts.
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
//! cargo run --release --example sequential_sfm_demo -- \
//!     --features-dir /tmp/sp_seq \
//!     --feature-suffix _features.txt --image-suffix .png \
//!     --width 752 --height 480 --fx 458.6 --fy 457.3 --cx 367.2 --cy 248.4 \
//!     --window 5 --skip-offsets 8,12 --skip-stride 2 --min-matches 30 \
//!     --out-colmap /tmp/seq_sfm_colmap
//! ```
//!
//! Flags of note. `--colmap-style` runs COLMAP's `IncrementalMapper` BA schedule
//! (per-registration local BA + growth-triggered iterative global refinement +
//! registration retries) instead of the simple "global BA every N + final BA"
//! path; on a 300-frame EuRoC MH_03 monocular subset this lifts accuracy from
//! 2.13 to 1.64 cm and registration from 272 to 299 / 300 (see
//! `docs/sfm_vs_colmap_benchmark.md`). `--retriangulate` (simple path only)
//! re-triangulates tracks after the final BA — a structure-density lever,
//! ATE-neutral, off by default.
//! `--post-refinement-registration` enables one bounded completion sweep after
//! the final iterative refinement: each still-missing image gets one fresh PnP
//! attempt against the filtered/re-triangulated structure, followed by another
//! final refinement only if at least one image registers. It is experimental and
//! off by default.
//! `--geometry-conflict-recovery` revisits union-find components discarded for
//! duplicate-image observations only after the normal posed model is refined.
//! It requires three-view reprojection plus a verified-edge cycle and rolls the
//! added tracks and BA back if the original clean tracks' residual worsens. It is
//! experimental and off by default.
//! `--structureless-registration` runs bounded passes after ordinary
//! post-refinement PnP. A missing image needs at least three registered verified
//! neighbours whose independently recovered rotations agree; their camera-
//! centre direction lines recover translation scale in the existing model.
//! Passes repeat until a pass registers nothing (budget
//! `--structureless-max-rounds`, default 4), so an island can chain inward
//! through a bridge whose index is higher than the images it unlocks.
//! It is experimental and off by default.
//! `--pose-guided-track-augmentation` keeps the separately matched
//! `--pose-graph-offsets` out of initial union-find construction, then attaches
//! only previously unowned keypoints that reproject from a trusted landmark
//! within 2 px. A guarded joint BA commits only if clean observations, new
//! observations, and bounded pose correction all pass. It is experimental and
//! off by default.
//! `--wide-hypothesis` instead runs a completely independent reconstruction
//! from the base pairs plus those wide pairs. It replaces the trusted arm only
//! when registration is not lower, at least 90% of landmarks survive, valid
//! observations grow by at least 25%, and mean reprojection strictly improves.
//! `--parallel-ba` sets [`visloc_rs::BaConfig::parallel`] on every bundle
//! adjustment this binary runs (periodic, final, and — under `--hierarchical`
//! — each local submap's and the seam BA's): a rayon-parallelized assembly /
//! Schur-reduction / back-substitution path proven bit-identical to the serial
//! one. Off by default.
//! `--submap-loop-closure` enables descriptor-derived long-range submap loops
//! and a second Sim(3) pose-graph solve after seam BA. Candidate frame pairs
//! use `--match-ratio`; `--submap-loop-min-matches` defaults to 30.
//! Accepted loops then run a loop-landmark-welded second BA by default; use
//! `--no-submap-loop-ba` to retain the loop-PGO result without that pass.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, SMatrix, UnitQuaternion};
use rayon::prelude::*;
use visloc_rs::slam::{
    hierarchical_sfm, partition_ordered_submaps, AdaptiveSubmapPartitionHints,
    CameraCentreScaleRefinementConfig, HierarchicalSeamBaConfig, HierarchicalSfmConfig,
    HierarchicalSfmResult, PairRotationEvidence,
};
use visloc_rs::vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_rs::vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};
use visloc_rs::{
    generate_ordered_pairs, incremental_sfm, preview_track_build_stats,
    read_external_deep_features_txt, relative_world_to_camera,
    write_colmap_reconstruction_for_3dgs, BaConfig, BaObservation, BruteForceMatcher,
    BundleAdjustment, Camera, CrossCheckMatcher, FeatureSet, IncrementalSfmConfig, LinearSolver,
    Matcher, NextImagePolicy, OrderedPairGeneratorConfig, OrderedPairHints, OrderedPairSource,
    PairwiseMatches, Pose, PoseGraph, PoseGraphEdgeKind, PoseGraphSe3Config, RobustKernel,
    TrackBuildStats, TrackSource,
};

/// A COLMAP-export landmark: world position + `(image, keypoint, pixel)` track.
type ExportLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

#[derive(Debug, Clone)]
struct VerifiedPair {
    pairwise: PairwiseMatches,
    image_j_from_i: UnitQuaternion<f64>,
}

struct Args {
    features_dir: PathBuf,
    feature_suffix: String,
    image_suffix: String,
    out_colmap: PathBuf,
    camera: Camera,
    window: usize,
    skip_offsets: Vec<usize>,
    skip_stride: usize,
    pose_graph_offsets: Vec<usize>,
    pose_graph_stride: usize,
    pose_guided_track_augmentation: bool,
    wide_hypothesis: bool,
    match_ratio: f32,
    min_matches: usize,
    min_pnp_inliers: usize,
    max_reproj: f64,
    final_ba: bool,
    seed_trials: usize,
    retriangulate: bool,
    colmap_style: bool,
    min_tri_angle: f64,
    refine_intrinsics: bool,
    next_image_policy: NextImagePolicy,
    post_refinement_registration: bool,
    structureless_registration: bool,
    structureless_max_rounds: usize,
    geometry_guided_conflict_recovery: bool,
    track_source: TrackSource,
    hierarchical: bool,
    submap_min_images: usize,
    submap_target_images: usize,
    submap_max_images: usize,
    submap_overlap_images: usize,
    submap_boundary_search_radius: usize,
    submap_min_shared_observations: usize,
    submap_build_threads: usize,
    submap_camera_scale_refinement: bool,
    submap_constraint_band: usize,
    submap_loop_closure: bool,
    submap_loop_min_matches: usize,
    submap_loop_top_k: usize,
    submap_loop_min_similarity: Option<f32>,
    submap_loop_ba: bool,
    submap_seam_ba: bool,
    submap_seam_ba_iterations: Option<usize>,
    submap_seam_ba_rounds: Option<usize>,
    submap_seam_ba_filter_px: Option<f64>,
    parallel_ba: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut features_dir = None;
    let mut feature_suffix = String::from("_features.txt");
    let mut image_suffix = String::from(".png");
    let mut out_colmap = None;
    let (mut width, mut height) = (None, None);
    let (mut fx, mut fy, mut cx, mut cy) = (None, None, None, None);
    let mut window = 5usize;
    let mut skip_offsets = Vec::new();
    let mut skip_stride = 1usize;
    let mut pose_graph_offsets = Vec::new();
    let mut pose_graph_stride = 1usize;
    let mut pose_guided_track_augmentation = false;
    let mut wide_hypothesis = false;
    let mut match_ratio = 0.8f32;
    let mut min_matches = 30usize;
    let mut min_pnp_inliers = 12usize;
    let mut max_reproj = 4.0f64;
    let mut final_ba = true;
    let mut seed_trials = 12usize;
    let mut retriangulate = false;
    let mut colmap_style = false;
    let mut min_tri_angle = 2.0f64;
    let mut refine_intrinsics = false;
    let mut next_image_policy = NextImagePolicy::VisibilityPyramid;
    let mut post_refinement_registration = false;
    let mut structureless_registration = false;
    let mut structureless_max_rounds = 4usize;
    let mut geometry_guided_conflict_recovery = false;
    let mut track_source = TrackSource::UnionFind;
    let mut hierarchical = false;
    let mut submap_min_images = 24usize;
    let mut submap_target_images = 64usize;
    let mut submap_max_images = 96usize;
    let mut submap_overlap_images = 16usize;
    let mut submap_boundary_search_radius = 16usize;
    let mut submap_min_shared_observations = 2usize;
    let mut submap_build_threads = 2usize;
    let mut submap_camera_scale_refinement = false;
    let mut submap_constraint_band = 4usize;
    let mut submap_loop_closure = false;
    let mut submap_loop_min_matches = 30usize;
    let mut submap_loop_top_k = 8usize;
    let mut submap_loop_min_similarity: Option<f32> = None;
    let mut submap_loop_ba: Option<bool> = None;
    let mut submap_seam_ba = false;
    let mut submap_seam_ba_iterations = None;
    let mut submap_seam_ba_rounds: Option<usize> = None;
    let mut submap_seam_ba_filter_px: Option<f64> = None;
    let mut parallel_ba = false;

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
            "--window" => window = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--skip-offsets" => {
                let value = a.remove(i + 1);
                skip_offsets = value
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(|part| {
                        part.parse::<usize>()
                            .map_err(|e| format!("invalid --skip-offsets entry {part:?}: {e}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--skip-stride" => skip_stride = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--pose-graph-offsets" => {
                let value = a.remove(i + 1);
                pose_graph_offsets = value
                    .split(',')
                    .filter(|part| !part.is_empty())
                    .map(|part| {
                        part.parse::<usize>().map_err(|e| {
                            format!("invalid --pose-graph-offsets entry {part:?}: {e}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--pose-graph-stride" => {
                pose_graph_stride = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--pose-guided-track-augmentation" => pose_guided_track_augmentation = true,
            "--wide-hypothesis" => wide_hypothesis = true,
            "--match-ratio" => match_ratio = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-matches" => min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--min-pnp-inliers" => {
                min_pnp_inliers = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--max-reproj" => max_reproj = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--no-final-ba" => final_ba = false,
            "--retriangulate" => retriangulate = true,
            "--colmap-style" => colmap_style = true,
            "--min-tri-angle" => {
                min_tri_angle = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--refine-intrinsics" => refine_intrinsics = true,
            "--next-image-policy" => {
                next_image_policy = match a.remove(i + 1).as_str() {
                    "visibility" => NextImagePolicy::VisibilityPyramid,
                    "count" => NextImagePolicy::CorrespondenceCount,
                    value => {
                        return Err(format!(
                            "--next-image-policy must be visibility or count, got {value}"
                        ))
                    }
                }
            }
            "--seed-trials" => seed_trials = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?,
            "--post-refinement-registration" => post_refinement_registration = true,
            "--structureless-registration" => structureless_registration = true,
            "--structureless-max-rounds" => {
                structureless_max_rounds = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--geometry-conflict-recovery" => geometry_guided_conflict_recovery = true,
            "--track-source" => {
                track_source = match a.remove(i + 1).as_str() {
                    "union-find" => TrackSource::UnionFind,
                    "graph" => TrackSource::CorrespondenceGraph,
                    value => {
                        return Err(format!(
                            "--track-source must be union-find or graph, got {value}"
                        ))
                    }
                }
            }
            "--hierarchical" => hierarchical = true,
            "--submap-min-images" => {
                submap_min_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-target-images" => {
                submap_target_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-max-images" => {
                submap_max_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-overlap-images" => {
                submap_overlap_images = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-boundary-search-radius" => {
                submap_boundary_search_radius =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-min-shared-observations" => {
                submap_min_shared_observations =
                    a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-build-threads" => {
                submap_build_threads = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-camera-scale-refinement" => submap_camera_scale_refinement = true,
            "--submap-constraint-band" => {
                submap_constraint_band = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-loop-closure" => submap_loop_closure = true,
            "--submap-loop-min-matches" => {
                submap_loop_min_matches = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-loop-top-k" => {
                submap_loop_top_k = a.remove(i + 1).parse().map_err(|e| format!("{e}"))?
            }
            "--submap-loop-min-similarity" => {
                submap_loop_min_similarity =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--submap-loop-ba" => submap_loop_ba = Some(true),
            "--no-submap-loop-ba" => submap_loop_ba = Some(false),
            "--submap-seam-ba" => submap_seam_ba = true,
            "--submap-seam-ba-iterations" => {
                submap_seam_ba_iterations =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--submap-seam-ba-rounds" => {
                submap_seam_ba_rounds = Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--submap-seam-ba-filter-px" => {
                submap_seam_ba_filter_px =
                    Some(a.remove(i + 1).parse().map_err(|e| format!("{e}"))?)
            }
            "--parallel-ba" => parallel_ba = true,
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }

    if window == 0 {
        return Err("--window must be ≥1".into());
    }
    if skip_stride == 0 {
        return Err("--skip-stride must be ≥1".into());
    }
    if pose_graph_stride == 0 {
        return Err("--pose-graph-stride must be ≥1".into());
    }
    if submap_seam_ba_rounds == Some(0) {
        return Err("--submap-seam-ba-rounds must be ≥1".into());
    }
    if submap_loop_min_matches == 0 {
        return Err("--submap-loop-min-matches must be ≥1".into());
    }
    if submap_loop_top_k == 0 {
        return Err("--submap-loop-top-k must be ≥1".into());
    }
    if submap_loop_min_similarity
        .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        return Err("--submap-loop-min-similarity must be finite and within [-1, 1]".into());
    }
    if submap_seam_ba_filter_px.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err("--submap-seam-ba-filter-px must be finite and >0".into());
    }
    if pose_guided_track_augmentation && wide_hypothesis {
        return Err(
            "--pose-guided-track-augmentation and --wide-hypothesis are mutually exclusive".into(),
        );
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
        window,
        skip_offsets,
        skip_stride,
        pose_graph_offsets,
        pose_graph_stride,
        pose_guided_track_augmentation,
        wide_hypothesis,
        match_ratio,
        min_matches,
        min_pnp_inliers,
        max_reproj,
        final_ba,
        seed_trials,
        retriangulate,
        colmap_style,
        min_tri_angle,
        refine_intrinsics,
        next_image_policy,
        post_refinement_registration,
        structureless_registration,
        structureless_max_rounds,
        geometry_guided_conflict_recovery,
        track_source,
        hierarchical,
        submap_min_images,
        submap_target_images,
        submap_max_images,
        submap_overlap_images,
        submap_boundary_search_radius,
        submap_min_shared_observations,
        submap_build_threads,
        submap_camera_scale_refinement,
        submap_constraint_band,
        submap_loop_closure,
        submap_loop_min_matches,
        submap_loop_top_k,
        submap_loop_min_similarity,
        submap_loop_ba: submap_loop_ba.unwrap_or(submap_loop_closure),
        submap_seam_ba,
        submap_seam_ba_iterations,
        submap_seam_ba_rounds,
        submap_seam_ba_filter_px,
        parallel_ba,
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

/// Match and geometrically verify each candidate pair, retaining both the
/// inlier correspondences and the independently estimated essential rotation.
/// Candidate pairs are independent, so the (descriptor-matching dominated) loop
/// is run across cores with rayon.
fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
) -> Vec<VerifiedPair> {
    candidates
        .par_iter()
        .filter_map(|&(i, j)| {
            let matcher = CrossCheckMatcher::new(BruteForceMatcher {
                ratio: Some(match_ratio),
            });
            let estimator = RelativePoseEstimator::default();
            let dm = matcher.match_descriptors(&features[i].descriptors, &features[j].descriptors);
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
            let rel = estimator.estimate(&corrs, camera)?;
            if rel.inliers.len() < min_matches {
                return None;
            }
            let matches: Vec<(usize, usize)> = rel
                .inliers
                .iter()
                .map(|&idx| (dm[idx].query_index, dm[idx].train_index))
                .collect();
            Some(VerifiedPair {
                pairwise: PairwiseMatches {
                    image_i: i,
                    image_j: j,
                    matches,
                },
                image_j_from_i: rel.previous_to_current.rotation,
            })
        })
        .collect()
}

fn fixed_offset_pairs(
    image_count: usize,
    offsets: &[usize],
    source_stride: usize,
) -> Vec<(usize, usize)> {
    let mut pairs = std::collections::BTreeSet::new();
    for image_i in (0..image_count).step_by(source_stride.max(1)) {
        for &offset in offsets {
            if offset == 0 {
                continue;
            }
            if let Some(image_j) = image_i.checked_add(offset).filter(|&j| j < image_count) {
                pairs.insert((image_i, image_j));
            }
        }
    }
    pairs.into_iter().collect()
}

fn observation_coverage(
    features: &[FeatureSet],
    pair: &PairwiseMatches,
    camera: &Camera,
) -> (f64, f64) {
    let coverage = |image: usize, first: bool| {
        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &(a, b) in &pair.matches {
            let keypoint = if first { a } else { b };
            let Some(pixel) = features[image].keypoints.get(keypoint) else {
                continue;
            };
            min_x = min_x.min(pixel.x);
            min_y = min_y.min(pixel.y);
            max_x = max_x.max(pixel.x);
            max_y = max_y.max(pixel.y);
        }
        if !min_x.is_finite() {
            return 0.0;
        }
        ((max_x - min_x).max(0.0) * (max_y - min_y).max(0.0))
            / (camera.width as f64 * camera.height as f64)
    };
    (coverage(pair.image_i, true), coverage(pair.image_j, false))
}

fn reconstruction_reprojection(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[visloc_rs::SfmTrack],
) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0usize;
    for track in tracks {
        for &(image, _, pixel) in &track.observations {
            let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                continue;
            };
            let Some(predicted) = camera.project(&pose.transform_world_point(&track.position))
            else {
                continue;
            };
            sum += (predicted - pixel).norm();
            count += 1;
        }
    }
    if count == 0 {
        (f64::NAN, 0)
    } else {
        (sum / count as f64, count)
    }
}

#[derive(Debug, Clone)]
struct TrackObservationProposal {
    track: usize,
    image: usize,
    keypoint: usize,
    pixel: Point2<f64>,
    error_px: f64,
}

#[derive(Debug, Clone)]
struct TrackMergeProposal {
    track_a: usize,
    track_b: usize,
    max_cross_error_px: f64,
}

#[derive(Default)]
struct ObservationDisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl ObservationDisjointSet {
    fn push(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.rank.push(0);
        id
    }

    fn find(&mut self, node: usize) -> usize {
        if self.parent[node] != node {
            self.parent[node] = self.find(self.parent[node]);
        }
        self.parent[node]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut root_a = self.find(a);
        let mut root_b = self.find(b);
        if root_a == root_b {
            return;
        }
        if self.rank[root_a] < self.rank[root_b] {
            std::mem::swap(&mut root_a, &mut root_b);
        }
        self.parent[root_b] = root_a;
        if self.rank[root_a] == self.rank[root_b] {
            self.rank[root_a] += 1;
        }
    }
}

/// Triangulate new multi-view tracks from wide matches whose endpoints are both
/// absent from the trusted reconstructed structure. Duplicate-image components,
/// two-view-only components, low parallax, failed cheirality, and any observation
/// over the reprojection gate are rejected before BA.
fn pose_guided_new_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    existing_tracks: &[visloc_rs::SfmTrack],
    wide_pairs: &[PairwiseMatches],
    min_views: usize,
    min_angle_deg: f64,
    max_error_px: f64,
) -> Vec<visloc_rs::SfmTrack> {
    let owned = existing_tracks
        .iter()
        .flat_map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
        })
        .collect::<HashSet<_>>();
    let mut node_ids = HashMap::<(usize, usize), usize>::new();
    let mut nodes = Vec::<(usize, usize)>::new();
    let mut sets = ObservationDisjointSet::default();
    for pair in wide_pairs {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let a = (pair.image_i, keypoint_i);
            let b = (pair.image_j, keypoint_j);
            if owned.contains(&a) || owned.contains(&b) {
                continue;
            }
            let id_a = *node_ids.entry(a).or_insert_with(|| {
                nodes.push(a);
                sets.push()
            });
            let id_b = *node_ids.entry(b).or_insert_with(|| {
                nodes.push(b);
                sets.push()
            });
            sets.union(id_a, id_b);
        }
    }
    let mut components = HashMap::<usize, Vec<(usize, usize)>>::new();
    for (node_id, &observation) in nodes.iter().enumerate() {
        components
            .entry(sets.find(node_id))
            .or_default()
            .push(observation);
    }

    let mut new_tracks = Vec::new();
    for mut observations in components.into_values() {
        observations.sort_unstable();
        observations.dedup();
        if observations.len() < min_views
            || observations
                .iter()
                .map(|&(image, _)| image)
                .collect::<HashSet<_>>()
                .len()
                != observations.len()
        {
            continue;
        }
        let mut best_pair = None;
        for a in 0..observations.len() {
            for b in a + 1..observations.len() {
                let (image_a, keypoint_a) = observations[a];
                let (image_b, keypoint_b) = observations[b];
                let (Some(normalized_a), Some(normalized_b)) = (
                    camera.normalize_pixel(&features[image_a].keypoints[keypoint_a]),
                    camera.normalize_pixel(&features[image_b].keypoints[keypoint_b]),
                ) else {
                    continue;
                };
                let ray_a_camera =
                    nalgebra::Vector3::new(normalized_a.x, normalized_a.y, 1.0).normalize();
                let ray_b_camera =
                    nalgebra::Vector3::new(normalized_b.x, normalized_b.y, 1.0).normalize();
                let ray_a_world =
                    poses[image_a].as_ref().unwrap().camera_to_world().rotation * ray_a_camera;
                let ray_b_world =
                    poses[image_b].as_ref().unwrap().camera_to_world().rotation * ray_b_camera;
                let angle = ray_a_world.dot(&ray_b_world).clamp(-1.0, 1.0).acos();
                if best_pair.as_ref().is_none_or(|(best, _, _)| angle > *best) {
                    best_pair = Some((angle, a, b));
                }
            }
        }
        let Some((angle, index_a, index_b)) = best_pair else {
            continue;
        };
        if angle.to_degrees() < min_angle_deg {
            continue;
        }
        let (image_a, keypoint_a) = observations[index_a];
        let (image_b, keypoint_b) = observations[index_b];
        let relative = relative_world_to_camera(
            poses[image_a].as_ref().unwrap(),
            poses[image_b].as_ref().unwrap(),
        );
        let Some(point_a) = triangulate_two_view_left_frame(
            camera,
            camera,
            &relative,
            &features[image_a].keypoints[keypoint_a],
            &features[image_b].keypoints[keypoint_b],
        ) else {
            continue;
        };
        let point_world = poses[image_a]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .transform_point(&point_a);
        let mut track_observations = Vec::with_capacity(observations.len());
        let mut valid = true;
        for &(image, keypoint) in &observations {
            let pixel = features[image].keypoints[keypoint];
            let Some(projected) = camera.project(
                &poses[image]
                    .as_ref()
                    .unwrap()
                    .transform_world_point(&point_world),
            ) else {
                valid = false;
                break;
            };
            if (projected - pixel).norm() > max_error_px {
                valid = false;
                break;
            }
            track_observations.push((image, keypoint, pixel));
        }
        if valid {
            new_tracks.push(visloc_rs::SfmTrack {
                position: point_world,
                observations: track_observations,
            });
        }
    }
    new_tracks
}

/// Find disjoint landmark fragments connected by a verified wide match whose
/// two current 3D estimates both explain every observation of the other
/// fragment. Each track participates in at most one merge in this conservative
/// first slice; ambiguous chains remain separate.
fn pose_guided_track_merge_proposals(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[visloc_rs::SfmTrack],
    wide_pairs: &[PairwiseMatches],
    max_error_px: f64,
) -> Vec<TrackMergeProposal> {
    let mut owner = HashMap::new();
    let mut track_images = vec![HashSet::new(); tracks.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, keypoint, _) in &track.observations {
            owner.insert((image, keypoint), track_id);
            track_images[track_id].insert(image);
        }
    }
    let cross_error = |point: &Point3<f64>, track: &visloc_rs::SfmTrack| {
        let mut worst = 0.0f64;
        for &(image, _, pixel) in &track.observations {
            let pose = poses.get(image).and_then(Option::as_ref)?;
            let projected = camera.project(&pose.transform_world_point(point))?;
            let error = (projected - pixel).norm();
            if !error.is_finite() {
                return None;
            }
            worst = worst.max(error);
        }
        Some(worst)
    };

    let mut best_by_pair = HashMap::<(usize, usize), f64>::new();
    for pair in wide_pairs {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (Some(&track_i), Some(&track_j)) = (
                owner.get(&(pair.image_i, keypoint_i)),
                owner.get(&(pair.image_j, keypoint_j)),
            ) else {
                continue;
            };
            if track_i == track_j || !track_images[track_i].is_disjoint(&track_images[track_j]) {
                continue;
            }
            let (track_a, track_b) = if track_i < track_j {
                (track_i, track_j)
            } else {
                (track_j, track_i)
            };
            let Some(error_ab) = cross_error(&tracks[track_a].position, &tracks[track_b]) else {
                continue;
            };
            let Some(error_ba) = cross_error(&tracks[track_b].position, &tracks[track_a]) else {
                continue;
            };
            let error = error_ab.max(error_ba);
            if error <= max_error_px {
                best_by_pair
                    .entry((track_a, track_b))
                    .and_modify(|best| *best = best.min(error))
                    .or_insert(error);
            }
        }
    }
    let mut candidates = best_by_pair
        .into_iter()
        .map(
            |((track_a, track_b), max_cross_error_px)| TrackMergeProposal {
                track_a,
                track_b,
                max_cross_error_px,
            },
        )
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        a.max_cross_error_px
            .total_cmp(&b.max_cross_error_px)
            .then_with(|| a.track_a.cmp(&b.track_a))
            .then_with(|| a.track_b.cmp(&b.track_b))
    });
    let mut used_tracks = HashSet::new();
    candidates
        .into_iter()
        .filter(|proposal| {
            if used_tracks.contains(&proposal.track_a) || used_tracks.contains(&proposal.track_b) {
                false
            } else {
                used_tracks.insert(proposal.track_a);
                used_tracks.insert(proposal.track_b);
                true
            }
        })
        .collect()
}

/// Associate a wide-pair endpoint only when the other endpoint already owns a
/// trusted landmark and that landmark reprojects onto the unowned endpoint.
/// Greedy lowest-error selection enforces one owner per keypoint and one
/// observation per track/image, so this stage cannot merge conflicting track
/// components or silently replace a trusted observation.
fn pose_guided_observation_proposals(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    tracks: &[visloc_rs::SfmTrack],
    wide_pairs: &[PairwiseMatches],
    max_error_px: f64,
) -> Vec<TrackObservationProposal> {
    let mut owner = HashMap::new();
    let mut track_images = vec![HashSet::new(); tracks.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, keypoint, _) in &track.observations {
            owner.insert((image, keypoint), track_id);
            track_images[track_id].insert(image);
        }
    }

    let mut candidates = Vec::new();
    for pair in wide_pairs {
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let endpoints = [
                ((pair.image_i, keypoint_i), (pair.image_j, keypoint_j)),
                ((pair.image_j, keypoint_j), (pair.image_i, keypoint_i)),
            ];
            for (source, target) in endpoints {
                let Some(&track_id) = owner.get(&source) else {
                    continue;
                };
                if owner.contains_key(&target) || track_images[track_id].contains(&target.0) {
                    continue;
                }
                let Some(pixel) = features
                    .get(target.0)
                    .and_then(|set| set.keypoints.get(target.1))
                    .copied()
                else {
                    continue;
                };
                let Some(pose) = poses.get(target.0).and_then(Option::as_ref) else {
                    continue;
                };
                let Some(projected) =
                    camera.project(&pose.transform_world_point(&tracks[track_id].position))
                else {
                    continue;
                };
                let error_px = (projected - pixel).norm();
                if error_px.is_finite() && error_px <= max_error_px {
                    candidates.push(TrackObservationProposal {
                        track: track_id,
                        image: target.0,
                        keypoint: target.1,
                        pixel,
                        error_px,
                    });
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        a.error_px
            .total_cmp(&b.error_px)
            .then_with(|| a.track.cmp(&b.track))
            .then_with(|| a.image.cmp(&b.image))
            .then_with(|| a.keypoint.cmp(&b.keypoint))
    });

    let mut used_keypoints = owner.keys().copied().collect::<HashSet<_>>();
    let mut used_track_images = tracks
        .iter()
        .map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, _, _)| image)
                .collect::<HashSet<_>>()
        })
        .collect::<Vec<_>>();
    let mut accepted = Vec::new();
    for proposal in candidates {
        if used_keypoints.contains(&(proposal.image, proposal.keypoint))
            || used_track_images[proposal.track].contains(&proposal.image)
        {
            continue;
        }
        used_keypoints.insert((proposal.image, proposal.keypoint));
        used_track_images[proposal.track].insert(proposal.image);
        accepted.push(proposal);
    }
    accepted
}

/// Merge only cross-reprojection-compatible landmark fragments, add
/// pose-consistent wide observations, then run a transactional joint BA.
fn augment_tracks_with_pose_guided_edges(
    camera: &Camera,
    features: &[FeatureSet],
    wide_pairs: &[PairwiseMatches],
    min_observations: usize,
    parallel_ba: bool,
    result: &mut visloc_rs::IncrementalSfmResult,
) -> Result<(), Box<dyn std::error::Error>> {
    if result.registered_images != result.poses.len() || wide_pairs.is_empty() {
        println!(
            "pose-guided track augmentation skipped (registered={} total={} edges={})",
            result.registered_images,
            result.poses.len(),
            wide_pairs.len()
        );
        return Ok(());
    }
    let original_poses = result.poses.clone();
    let original_tracks = result.tracks.clone();
    let (clean_before, clean_observations_before) =
        reconstruction_reprojection(camera, &original_poses, &original_tracks);
    let merges = pose_guided_track_merge_proposals(
        camera,
        &original_poses,
        &original_tracks,
        wide_pairs,
        2.0,
    );
    let mut dropped_into = HashMap::new();
    for merge in &merges {
        dropped_into.insert(merge.track_b, merge.track_a);
    }
    let mut candidate_tracks = Vec::with_capacity(original_tracks.len() - merges.len());
    let mut origin_to_candidate = vec![usize::MAX; original_tracks.len()];
    for (track_id, track) in original_tracks.iter().enumerate() {
        if dropped_into.contains_key(&track_id) {
            continue;
        }
        let candidate_id = candidate_tracks.len();
        origin_to_candidate[track_id] = candidate_id;
        let mut candidate = track.clone();
        if let Some(merge) = merges.iter().find(|merge| merge.track_a == track_id) {
            let dropped = &original_tracks[merge.track_b];
            let weight_a = candidate.observations.len() as f64;
            let weight_b = dropped.observations.len() as f64;
            candidate.position = Point3::from(
                (candidate.position.coords * weight_a + dropped.position.coords * weight_b)
                    / (weight_a + weight_b),
            );
            candidate
                .observations
                .extend_from_slice(&dropped.observations);
            origin_to_candidate[merge.track_b] = candidate_id;
        }
        candidate_tracks.push(candidate);
    }
    let new_tracks = pose_guided_new_tracks(
        camera,
        features,
        &original_poses,
        &candidate_tracks,
        wide_pairs,
        3,
        2.0,
        2.0,
    );
    let new_track_observations = new_tracks
        .iter()
        .map(|track| track.observations.len())
        .sum::<usize>();
    candidate_tracks.extend(new_tracks);
    let proposals = pose_guided_observation_proposals(
        camera,
        features,
        &original_poses,
        &candidate_tracks,
        wide_pairs,
        2.0,
    );
    if proposals.len() + merges.len() + new_track_observations < min_observations {
        println!(
            "pose-guided track augmentation skipped: {} additions + {} merges + {} new-track observations below minimum {}",
            proposals.len(),
            merges.len(),
            new_track_observations,
            min_observations
        );
        return Ok(());
    }

    for proposal in &proposals {
        candidate_tracks[proposal.track].observations.push((
            proposal.image,
            proposal.keypoint,
            proposal.pixel,
        ));
    }
    let mut ba = BundleAdjustment::new(camera.clone());
    for (image, pose) in original_poses.iter().enumerate() {
        ba.add_pose(image as u64, pose.clone().unwrap());
    }
    let anchor = 0usize;
    ba.fix_pose(anchor as u64);
    let anchor_center = original_poses[anchor]
        .as_ref()
        .unwrap()
        .camera_center_world();
    let scale_anchor = original_poses
        .iter()
        .enumerate()
        .filter(|(image, _)| *image != anchor)
        .max_by(|(_, a), (_, b)| {
            let da = (a.as_ref().unwrap().camera_center_world() - anchor_center).norm_squared();
            let db = (b.as_ref().unwrap().camera_center_world() - anchor_center).norm_squared();
            da.total_cmp(&db)
        })
        .map(|(image, _)| image)
        .unwrap_or(anchor);
    ba.fix_pose(scale_anchor as u64);
    for (track_id, track) in candidate_tracks.iter().enumerate() {
        ba.add_landmark(track_id as u64, track.position);
        for &(image, _, pixel) in &track.observations {
            ba.add_observation(BaObservation {
                keyframe_id: image as u64,
                landmark_id: track_id as u64,
                xy: pixel,
            });
        }
    }
    let ba_result = ba.optimize(&BaConfig {
        max_iterations: 30,
        robust_kernel: RobustKernel::Huber { delta: 2.0 },
        parallel: parallel_ba,
        ..BaConfig::default()
    })?;
    let candidate_poses = (0..original_poses.len())
        .map(|image| ba.poses.get(&(image as u64)).cloned())
        .collect::<Vec<_>>();
    for (track_id, track) in candidate_tracks.iter_mut().enumerate() {
        track.position = ba.landmarks[&(track_id as u64)];
    }

    let mut clean_after_tracks = original_tracks.clone();
    for (original_id, clean) in clean_after_tracks.iter_mut().enumerate() {
        clean.position = candidate_tracks[origin_to_candidate[original_id]].position;
    }
    let (clean_after, clean_observations_after) =
        reconstruction_reprojection(camera, &candidate_poses, &clean_after_tracks);
    let mut added_error_sum = 0.0;
    let mut added_valid = 0usize;
    for proposal in &proposals {
        let pose = candidate_poses[proposal.image].as_ref().unwrap();
        if let Some(projected) =
            camera.project(&pose.transform_world_point(&candidate_tracks[proposal.track].position))
        {
            added_error_sum += (projected - proposal.pixel).norm();
            added_valid += 1;
        }
    }
    let added_mean = added_error_sum / added_valid.max(1) as f64;
    let centers = original_poses
        .iter()
        .map(|pose| pose.as_ref().unwrap().camera_center_world())
        .collect::<Vec<_>>();
    let diameter = centers
        .iter()
        .flat_map(|a| centers.iter().map(move |b| (a - b).norm()))
        .fold(0.0f64, f64::max);
    let max_correction = candidate_poses
        .iter()
        .zip(&original_poses)
        .map(|(candidate, original)| {
            (candidate.as_ref().unwrap().camera_center_world()
                - original.as_ref().unwrap().camera_center_world())
            .norm()
        })
        .fold(0.0f64, f64::max);
    let (total_after, total_observations_after) =
        reconstruction_reprojection(camera, &candidate_poses, &candidate_tracks);
    let accepted = ba_result.final_cost.is_finite()
        && clean_before.is_finite()
        && clean_after.is_finite()
        && added_mean.is_finite()
        && total_after.is_finite()
        && clean_observations_after == clean_observations_before
        && added_valid == proposals.len()
        && total_observations_after
            == clean_observations_before + new_track_observations + proposals.len()
        && clean_after <= clean_before * 1.001 + 1e-12
        && added_mean <= 2.0
        && max_correction <= 0.10 * diameter.max(1e-12);
    println!(
        "pose-guided track augmentation: added={} merged={} new-tracks={}/{} ba={:.3}->{:.3} clean={:.6}->{:.6} \
         added={:.6}/{} total={:.6}/{} max-correction={:.6}/{:.6} accepted={}",
        proposals.len(),
        merges.len(),
        candidate_tracks.len() + merges.len() - original_tracks.len(),
        new_track_observations,
        ba_result.initial_cost,
        ba_result.final_cost,
        clean_before,
        clean_after,
        added_mean,
        added_valid,
        total_after,
        total_observations_after,
        max_correction,
        diameter,
        accepted,
    );
    if accepted {
        result.poses = candidate_poses;
        result.tracks = candidate_tracks;
        result.mean_reprojection_px = total_after;
    }
    Ok(())
}

/// Refine a complete reconstruction with wide-baseline edges without merging
/// their correspondences into the trusted feature tracks. The proposed SE(3)
/// correction is committed only when a fixed-camera landmark solve preserves
/// every valid observation and does not worsen mean reprojection by more than
/// 0.1%; otherwise poses and structure remain byte-identical.
fn refine_with_pose_only_edges(
    camera: &Camera,
    features: &[FeatureSet],
    wide_pairs: &[PairwiseMatches],
    min_matches: usize,
    parallel_ba: bool,
    result: &mut visloc_rs::IncrementalSfmResult,
) -> Result<(), Box<dyn std::error::Error>> {
    if result.registered_images != result.poses.len() || wide_pairs.is_empty() {
        println!(
            "pose-only wide refinement skipped (registered={} total={} edges={})",
            result.registered_images,
            result.poses.len(),
            wide_pairs.len()
        );
        return Ok(());
    }
    let original_poses = result.poses.clone();
    let original_tracks = result.tracks.clone();
    let (mean_before, observations_before) =
        reconstruction_reprojection(camera, &original_poses, &original_tracks);

    let mut graph = PoseGraph::new();
    for (image, pose) in original_poses.iter().enumerate() {
        graph.add_pose(image as u64, pose.clone().unwrap());
    }
    graph.anchor(0);
    let sequential_information = SMatrix::<f64, 6, 6>::identity() * 10.0;
    for image in 0..original_poses.len() - 1 {
        let measurement = relative_world_to_camera(
            original_poses[image].as_ref().unwrap(),
            original_poses[image + 1].as_ref().unwrap(),
        );
        graph.add_edge_with_information(
            image as u64,
            (image + 1) as u64,
            measurement,
            PoseGraphEdgeKind::Sequential,
            sequential_information,
        );
    }

    let estimator = RelativePoseEstimator::default();
    let mut admitted_edges = 0usize;
    for pair in wide_pairs {
        let (coverage_i, coverage_j) = observation_coverage(features, pair, camera);
        if pair.matches.len() < min_matches || coverage_i < 0.02 || coverage_j < 0.02 {
            continue;
        }
        let correspondences: Vec<_> = pair
            .matches
            .iter()
            .map(|&(a, b)| {
                TwoViewCorrespondence::new(
                    features[pair.image_i].keypoints[a],
                    features[pair.image_j].keypoints[b],
                )
            })
            .collect();
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < min_matches {
            continue;
        }
        let current = relative_world_to_camera(
            original_poses[pair.image_i].as_ref().unwrap(),
            original_poses[pair.image_j].as_ref().unwrap(),
        );
        let rotation_error = (current.rotation.inverse() * relative.previous_to_current.rotation)
            .angle()
            .to_degrees();
        let Some(current_direction) = current.translation.try_normalize(1e-12) else {
            continue;
        };
        let Some(measured_direction) = relative
            .previous_to_current
            .translation
            .try_normalize(1e-12)
        else {
            continue;
        };
        if rotation_error > 15.0 || current_direction.dot(&measured_direction) <= 0.0 {
            continue;
        }
        let mut measurement = relative.previous_to_current;
        measurement.translation = measured_direction * current.translation.norm();
        let edge_weight =
            (relative.inliers.len() as f64 / min_matches.max(1) as f64).clamp(1.0, 10.0);
        let mut information = SMatrix::<f64, 6, 6>::zeros();
        for axis in 0..3 {
            information[(axis, axis)] = edge_weight;
            information[(axis + 3, axis + 3)] = edge_weight * 10.0;
        }
        graph.add_edge_with_information(
            pair.image_i as u64,
            pair.image_j as u64,
            measurement,
            PoseGraphEdgeKind::LoopClosure,
            information,
        );
        admitted_edges += 1;
    }
    if admitted_edges == 0 {
        println!("pose-only wide refinement rejected every edge before optimization");
        return Ok(());
    }

    let pgo = graph.optimize_se3_iterative(&PoseGraphSe3Config {
        max_iterations: 30,
        robust_kernel: RobustKernel::Cauchy { c: 1.0 },
        initial_lambda: Some(1e-4),
        linear_solver: LinearSolver::Sparse,
        ..PoseGraphSe3Config::default()
    })?;
    let candidate_poses: Vec<Option<Pose>> = (0..original_poses.len())
        .map(|image| graph.poses.get(&(image as u64)).cloned())
        .collect();
    let original_centers: Vec<_> = original_poses
        .iter()
        .map(|pose| pose.as_ref().unwrap().camera_center_world())
        .collect();
    let diameter = original_centers
        .iter()
        .flat_map(|a| original_centers.iter().map(move |b| (a - b).norm()))
        .fold(0.0f64, f64::max);
    let max_correction = candidate_poses
        .iter()
        .zip(&original_poses)
        .map(|(candidate, original)| {
            (candidate.as_ref().unwrap().camera_center_world()
                - original.as_ref().unwrap().camera_center_world())
            .norm()
        })
        .fold(0.0f64, f64::max);
    if !diameter.is_finite() || max_correction > 0.25 * diameter.max(1e-12) {
        println!(
            "pose-only wide refinement rolled back: correction {:.6} exceeds 25% diameter {:.6}",
            max_correction, diameter
        );
        return Ok(());
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for (image, pose) in candidate_poses.iter().enumerate() {
        ba.add_pose(image as u64, pose.clone().unwrap());
        ba.fix_pose(image as u64);
    }
    for (track_id, track) in original_tracks.iter().enumerate() {
        ba.add_landmark(track_id as u64, track.position);
        for &(image, _, pixel) in &track.observations {
            ba.add_observation(BaObservation {
                keyframe_id: image as u64,
                landmark_id: track_id as u64,
                xy: pixel,
            });
        }
    }
    let point_ba = ba.optimize(&BaConfig {
        max_iterations: 20,
        robust_kernel: RobustKernel::Huber { delta: 2.0 },
        parallel: parallel_ba,
        ..BaConfig::default()
    })?;
    let mut candidate_tracks = original_tracks.clone();
    for (track_id, track) in candidate_tracks.iter_mut().enumerate() {
        track.position = ba.landmarks[&(track_id as u64)];
    }
    let (mean_after, observations_after) =
        reconstruction_reprojection(camera, &candidate_poses, &candidate_tracks);
    let accepted = pgo.final_cost.is_finite()
        && point_ba.final_cost.is_finite()
        && mean_before.is_finite()
        && mean_after.is_finite()
        && observations_after == observations_before
        && mean_after <= mean_before * 1.001 + 1e-12;
    println!(
        "pose-only wide refinement: edges={} pgo={:.6}->{:.6} max-correction={:.6}/{:.6} \
         point-ba={:.3}->{:.3} reproj={:.6}->{:.6} obs={}->{} accepted={}",
        admitted_edges,
        pgo.initial_cost,
        pgo.final_cost,
        max_correction,
        diameter,
        point_ba.initial_cost,
        point_ba.final_cost,
        mean_before,
        mean_after,
        observations_before,
        observations_after,
        accepted,
    );
    if accepted {
        result.poses = candidate_poses;
        result.tracks = candidate_tracks;
        result.mean_reprojection_px = mean_after;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ReconstructionQuality {
    registered: usize,
    points: usize,
    observations: usize,
    mean_reprojection_px: f64,
}

fn reconstruction_quality(result: &visloc_rs::IncrementalSfmResult) -> ReconstructionQuality {
    ReconstructionQuality {
        registered: result.registered_images,
        points: result.tracks.len(),
        observations: result
            .tracks
            .iter()
            .map(|track| track.observations.len())
            .sum(),
        mean_reprojection_px: result.mean_reprojection_px,
    }
}

/// GT-free selector for an independently grown wide-graph reconstruction.
/// The deliberately conjunctive gate captures the measured MH05 signature
/// (nearly all landmarks retained, many more observations, lower residual)
/// while rejecting the MH03 topology-collapse signature.
fn accept_wide_hypothesis(
    trusted: ReconstructionQuality,
    candidate: ReconstructionQuality,
) -> bool {
    trusted.registered > 0
        && trusted.points > 0
        && trusted.observations > 0
        && candidate.mean_reprojection_px.is_finite()
        && trusted.mean_reprojection_px.is_finite()
        && candidate.registered >= trusted.registered
        && candidate.points * 10 >= trusted.points * 9
        && candidate.observations * 4 >= trusted.observations * 5
        && candidate.mean_reprojection_px < trusted.mean_reprojection_px
}

/// Cheap topology preflight learned from the development/held-out separation:
/// MH05's useful wide graph retains 85.6% of tracks and 90.5% of observations,
/// whereas MH03's harmful graph collapses to 77.6% / 81.9%. The thresholds sit
/// between those measured regimes and are evaluated before any candidate seed,
/// triangulation, or BA work.
fn accept_wide_track_preflight(trusted: TrackBuildStats, candidate: TrackBuildStats) -> bool {
    candidate.input_correspondences > trusted.input_correspondences
        && candidate.retained_tracks * 100 >= trusted.retained_tracks * 82
        && candidate.retained_observations * 100 >= trusted.retained_observations * 85
}

fn merge_pairwise_graphs(
    trusted: &[PairwiseMatches],
    wide: &[PairwiseMatches],
) -> Vec<PairwiseMatches> {
    let mut merged = HashMap::<(usize, usize), Vec<(usize, usize)>>::new();
    for pair in trusted.iter().chain(wide) {
        merged
            .entry((pair.image_i, pair.image_j))
            .or_default()
            .extend_from_slice(&pair.matches);
    }
    let mut graph = merged
        .into_iter()
        .map(|((image_i, image_j), mut matches)| {
            matches.sort_unstable();
            matches.dedup();
            PairwiseMatches {
                image_i,
                image_j,
                matches,
            }
        })
        .collect::<Vec<_>>();
    graph.sort_by_key(|pair| (pair.image_i, pair.image_j));
    graph
}

fn export_hierarchical_result(
    args: &Args,
    features: &[FeatureSet],
    image_names: &[String],
    result: &HierarchicalSfmResult,
) -> Result<(), Box<dyn std::error::Error>> {
    // Export every overlap image exactly once, from the earliest submap that
    // registered it, so COLMAP image/keypoint identity stays unambiguous.
    let mut owner = BTreeMap::<u64, u64>::new();
    let mut poses_by_frame = BTreeMap::<u64, Pose>::new();
    for node in result.atlas.hierarchy.nodes() {
        let local_from_atlas = node
            .local_from_atlas
            .as_ref()
            .expect("successful hierarchy leaves every node aligned");
        let atlas_from_local = local_from_atlas.inverse();
        for frame in &node.submap.frames {
            if owner.contains_key(&frame.source_frame_id) {
                continue;
            }
            let rotation = frame.pose.world_to_camera.rotation * local_from_atlas.rotation;
            let centre_atlas = atlas_from_local.transform_point(&frame.pose.camera_center_world());
            let translation = -(rotation * centre_atlas.coords);
            owner.insert(frame.source_frame_id, node.id);
            poses_by_frame.insert(
                frame.source_frame_id,
                Pose::from_world_to_camera(rotation, translation),
            );
        }
    }

    let registered = poses_by_frame.keys().copied().collect::<Vec<_>>();
    let remap = registered
        .iter()
        .enumerate()
        .map(|(index, &frame_id)| (frame_id, index))
        .collect::<BTreeMap<_, _>>();
    let poses_out = registered
        .iter()
        .map(|frame_id| poses_by_frame[frame_id].clone())
        .collect::<Vec<_>>();
    let features_out = registered
        .iter()
        .map(|&frame_id| features[frame_id as usize].clone())
        .collect::<Vec<_>>();
    let names_out = registered
        .iter()
        .map(|&frame_id| image_names[frame_id as usize].clone())
        .collect::<Vec<_>>();

    let mut landmarks_out = Vec::<ExportLandmark>::new();
    let mut welded_members = BTreeSet::<(u64, u64)>::new();
    if let Some(seam_ba) = &result.atlas.seam_bundle_adjustment {
        for group in &seam_ba.welded_landmark_groups {
            let mut atlas_position = None;
            let mut observations = BTreeMap::<(u64, usize), Point2<f64>>::new();
            for &(submap_id, landmark_id) in group {
                welded_members.insert((submap_id, landmark_id));
                let node = result
                    .atlas
                    .hierarchy
                    .node(submap_id)
                    .expect("weld group references retained submap");
                let landmark = node
                    .submap
                    .landmarks
                    .iter()
                    .find(|landmark| landmark.local_landmark_id == landmark_id)
                    .expect("weld group references retained landmark");
                atlas_position.get_or_insert_with(|| {
                    node.atlas_from_local()
                        .expect("successful hierarchy leaves every node aligned")
                        .transform_point(&landmark.position)
                });
                for observation in &landmark.observations {
                    observations
                        .entry((observation.source_frame_id, observation.keypoint_index))
                        .or_insert(observation.pixel);
                }
            }
            let observations = observations
                .into_iter()
                .filter_map(|((frame_id, keypoint_index), pixel)| {
                    remap
                        .get(&frame_id)
                        .map(|&image_index| (image_index, keypoint_index, pixel))
                })
                .collect::<Vec<_>>();
            if observations.len() >= 2 {
                landmarks_out.push((
                    atlas_position.expect("weld group is non-empty"),
                    observations,
                ));
            }
        }
    }

    // Unwelded local points retain only observations owned by their submap; no
    // unverified cross-seam keypoint can enter two different COLMAP tracks.
    for node in result.atlas.hierarchy.nodes() {
        let atlas_from_local = node
            .atlas_from_local()
            .expect("successful hierarchy leaves every node aligned");
        for landmark in &node.submap.landmarks {
            if welded_members.contains(&(node.id, landmark.local_landmark_id)) {
                continue;
            }
            let observations = landmark
                .observations
                .iter()
                .filter_map(|observation| {
                    if owner.get(&observation.source_frame_id) != Some(&node.id) {
                        return None;
                    }
                    remap.get(&observation.source_frame_id).map(|&image_index| {
                        (image_index, observation.keypoint_index, observation.pixel)
                    })
                })
                .collect::<Vec<_>>();
            if observations.len() >= 2 {
                landmarks_out.push((
                    atlas_from_local.transform_point(&landmark.position),
                    observations,
                ));
            }
        }
    }

    let summary = write_colmap_reconstruction_for_3dgs(
        &args.out_colmap,
        &args.camera,
        &poses_out,
        &features_out,
        &landmarks_out,
        |index| names_out[index].clone(),
    )?;
    let mut seam_csv = String::from(
        "source_submap,target_submap,shared_matches,sim3_inliers,sim3_inlier_ratio,mean_residual_ratio,rotation_candidates,rotation_consensus,rotation_support,rotation_max_disagreement_deg,shared_camera_centres,camera_sim3_inliers,camera_sim3_inlier_ratio,camera_mean_residual_ratio,camera_landmark_log_scale_disagreement,camera_landmark_rotation_disagreement_deg,camera_refinement_applied,camera_refinement_rejection,camera_refinement_abs_log_scale_change,camera_refinement_mean_residual_ratio\n",
    );
    for seam in &result.atlas.seams {
        let optional = |value: Option<f64>| value.map(|v| format!("{v:.9}")).unwrap_or_default();
        seam_csv.push_str(&format!(
            "{},{},{},{},{:.6},{:.9},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
            seam.source_submap_id,
            seam.target_submap_id,
            seam.shared_point_matches,
            seam.sim3_inliers,
            seam.sim3_inlier_ratio,
            seam.mean_residual_ratio,
            seam.essential_rotation_candidates,
            seam.essential_rotation_consensus,
            seam.essential_rotation_support,
            seam.essential_rotation_max_disagreement_deg,
            seam.shared_camera_centres,
            seam.camera_sim3_inliers
                .map(|value| value.to_string())
                .unwrap_or_default(),
            optional(seam.camera_sim3_inlier_ratio),
            optional(seam.camera_mean_residual_ratio),
            optional(seam.camera_landmark_log_scale_disagreement),
            optional(seam.camera_landmark_rotation_disagreement_deg),
            seam.camera_refinement_applied,
            seam.camera_refinement_rejection
                .map(|value| format!("{value:?}"))
                .unwrap_or_default(),
            optional(seam.camera_refinement_abs_log_scale_change),
            optional(seam.camera_refinement_mean_residual_ratio),
        ));
    }
    std::fs::write(args.out_colmap.join("hierarchical_seams.csv"), seam_csv)?;
    println!(
        "hierarchical reconstruction: {} submaps, {} seams, {} / {} unique images registered",
        result.atlas.hierarchy.nodes().count(),
        result.atlas.seams.len(),
        poses_out.len(),
        features.len()
    );
    if let Some(ba) = &result.atlas.seam_bundle_adjustment {
        println!(
            "  seam BA: poses={} (fixed={}) landmarks={} observations={} weld_groups={} cost={:.3}->{:.3} iterations={}",
            ba.pose_count,
            ba.fixed_pose_count,
            ba.landmark_count,
            ba.observation_count,
            ba.welded_landmark_groups.len(),
            ba.initial_cost,
            ba.final_cost,
            ba.iterations,
        );
    }
    for (index, window) in result.windows.iter().enumerate() {
        println!(
            "  submap {index}: images {:?}, seam_support={}",
            window.image_range, window.outgoing_seam_support
        );
    }
    println!(
        "wrote hierarchical COLMAP model to {} ({} images, {} points, {} observations)",
        args.out_colmap.display(),
        summary.frame_count,
        summary.landmark_count,
        summary.observation_count,
    );
    Ok(())
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

    let pair_config = OrderedPairGeneratorConfig {
        min_temporal_window: args.window,
        max_temporal_window: args.window,
        skip_offsets: args.skip_offsets.clone(),
        skip_source_stride: args.skip_stride,
    };
    let generated =
        generate_ordered_pairs(features.len(), &pair_config, &OrderedPairHints::default());
    let temporal_pairs = generated
        .iter()
        .filter(|pair| pair.sources.contains(&OrderedPairSource::Temporal))
        .count();
    let skip_pairs = generated
        .iter()
        .filter(|pair| pair.sources.contains(&OrderedPairSource::Skip))
        .count();
    let candidates: Vec<_> = generated
        .iter()
        .map(|pair| (pair.image_i, pair.image_j))
        .collect();
    println!(
        "view graph: {} candidate pairs (temporal={} skip={} window={} skip_offsets={:?} skip_stride={})",
        candidates.len(),
        temporal_pairs,
        skip_pairs,
        args.window,
        args.skip_offsets,
        args.skip_stride,
    );

    let t_match = std::time::Instant::now();
    let verified_pairs = verify_pairs(
        &features,
        &args.camera,
        &candidates,
        args.match_ratio,
        args.min_matches,
    );
    let pairwise = verified_pairs
        .iter()
        .map(|verified| verified.pairwise.clone())
        .collect::<Vec<_>>();
    let pair_rotations = verified_pairs
        .iter()
        .map(|verified| PairRotationEvidence {
            image_i: verified.pairwise.image_i as u64,
            image_j: verified.pairwise.image_j as u64,
            image_j_from_i: verified.image_j_from_i,
            inlier_count: verified.pairwise.matches.len(),
        })
        .collect::<Vec<_>>();
    let verified_matches: usize = pairwise.iter().map(|p| p.matches.len()).sum();
    println!(
        "verified {} / {} pairs, {} inlier correspondences [match {:.1}s]",
        pairwise.len(),
        candidates.len(),
        verified_matches,
        t_match.elapsed().as_secs_f64(),
    );
    if pairwise.is_empty() {
        return Err("no pair survived geometric verification — lower --min-matches?".into());
    }
    let pose_graph_candidates = fixed_offset_pairs(
        features.len(),
        &args.pose_graph_offsets,
        args.pose_graph_stride,
    );
    let t_pose_graph_match = std::time::Instant::now();
    let pose_graph_verified = verify_pairs(
        &features,
        &args.camera,
        &pose_graph_candidates,
        args.match_ratio,
        args.min_matches,
    );
    let pose_graph_pairwise = pose_graph_verified
        .iter()
        .map(|verified| verified.pairwise.clone())
        .collect::<Vec<_>>();
    if !pose_graph_candidates.is_empty() {
        println!(
            "pose-only graph: verified {} / {} candidates at offsets {:?} stride={} [match {:.1}s]",
            pose_graph_pairwise.len(),
            pose_graph_candidates.len(),
            args.pose_graph_offsets,
            args.pose_graph_stride,
            t_pose_graph_match.elapsed().as_secs_f64(),
        );
    }

    let mut config = IncrementalSfmConfig {
        min_seed_matches: args.min_matches,
        min_pnp_inliers: args.min_pnp_inliers,
        max_reprojection_error_px: args.max_reproj,
        final_global_ba: args.final_ba,
        seed_trials: args.seed_trials,
        retriangulate: args.retriangulate,
        colmap_style_mapper: args.colmap_style,
        refine_intrinsics: args.refine_intrinsics,
        next_image_policy: args.next_image_policy,
        post_refinement_registration: args.post_refinement_registration,
        structureless_registration: args.structureless_registration,
        structureless_max_rounds: args.structureless_max_rounds,
        geometry_guided_conflict_recovery: args.geometry_guided_conflict_recovery,
        track_source: args.track_source,
        min_triangulation_angle_deg: args.min_tri_angle,
        // NB: the low-parallax multi-view exemption (`low_parallax_min_observations`)
        // is deliberately left off. On this forward-flying trajectory it is a trap —
        // even tracks seen by ≥6 views stay near-zero-parallax (the camera barely
        // sideways-translates), so keeping them injects depth-ambiguous points that
        // corrupt the poses: measured 10 k tracks but ATE blows up to 16 cm vs the
        // strict 2° gate's 1.64 cm. The exemption is for sideways/orbiting capture,
        // not forward video. See `docs/sfm_vs_colmap_benchmark.md`.
        ..IncrementalSfmConfig::default()
    };
    // `--parallel-ba` flag-gates `BaConfig::parallel` (see `bundle.rs`'s "Parallelism"
    // section): a rayon-parallel assembly / Schur-reduction / back-substitution path
    // proven bit-identical to the serial path. Default off, so every existing
    // invocation is unchanged. This is the single place that needs to be set: every
    // BA this binary runs — periodic, final, and (via `hierarchical_config.local_submap.sfm
    // = config.clone()` below) each hierarchical local submap's — reads `config.ba_config`.
    config.ba_config.parallel = args.parallel_ba;
    if args.hierarchical {
        if args.refine_intrinsics {
            return Err(
                "--hierarchical currently requires fixed shared intrinsics; omit --refine-intrinsics"
                    .into(),
            );
        }
        if args.wide_hypothesis || args.pose_guided_track_augmentation {
            return Err(
                "--hierarchical does not yet compose the trusted/wide transaction; omit --wide-hypothesis and --pose-guided-track-augmentation"
                    .into(),
            );
        }
        let mut hierarchical_config = HierarchicalSfmConfig::default();
        hierarchical_config.partition.min_images = args.submap_min_images;
        hierarchical_config.partition.target_images = args.submap_target_images;
        hierarchical_config.partition.max_images = args.submap_max_images;
        hierarchical_config.partition.overlap_images = args.submap_overlap_images;
        hierarchical_config.partition.boundary_search_radius = args.submap_boundary_search_radius;
        // `min_post_widen_overlap_images` is deliberately left at
        // `HierarchicalSfmConfig::default()`'s small constant (see
        // `AdaptiveSubmapPartitionConfig`'s doc comment) rather than mirrored
        // from `--submap-overlap-images`: tying it to the full configured
        // overlap chains a costly cascade of full incremental-SfM rebuilds
        // (measured far worse than linear in image count) for a marginal
        // safety gain over the cheap single-absorption default.
        hierarchical_config
            .overlap
            .min_shared_observations_per_landmark = args.submap_min_shared_observations;
        hierarchical_config.max_parallel_local_builds = args.submap_build_threads;
        hierarchical_config.submap_constraint_band = args.submap_constraint_band;
        hierarchical_config.submap_loop_closure = args.submap_loop_closure;
        hierarchical_config.submap_loop_min_matches = args.submap_loop_min_matches;
        hierarchical_config.submap_loop_top_k = args.submap_loop_top_k;
        hierarchical_config.submap_loop_min_similarity = args.submap_loop_min_similarity;
        hierarchical_config.submap_loop_bundle_adjustment = args.submap_loop_ba;
        // Loop discovery intentionally uses the identical descriptor matcher
        // and Lowe ratio as the sequential-pair path above.
        hierarchical_config.submap_loop_match_ratio = args.match_ratio;
        if args.submap_camera_scale_refinement {
            hierarchical_config.camera_centre_refinement =
                Some(CameraCentreScaleRefinementConfig::default());
        }
        if args.submap_seam_ba {
            let mut seam_ba_config = HierarchicalSeamBaConfig::default();
            if let Some(max_iterations) = args.submap_seam_ba_iterations {
                seam_ba_config.ba.max_iterations = max_iterations;
            }
            // Preserve the frozen CLI behavior unless iterative filtering is
            // explicitly requested. The library config defaults to three
            // COLMAP-style rounds, while this example historically ran one
            // unfiltered solve.
            seam_ba_config.max_rounds = args.submap_seam_ba_rounds.unwrap_or(1);
            if let Some(max_reprojection_px) = args.submap_seam_ba_filter_px {
                seam_ba_config.max_reprojection_px = max_reprojection_px;
            }
            seam_ba_config.ba.parallel = args.parallel_ba;
            hierarchical_config.seam_bundle_adjustment = Some(seam_ba_config);
        }
        hierarchical_config.local_submap.sfm = config.clone();
        let source_frame_ids = (0..features.len())
            .map(|index| index as u64)
            .collect::<Vec<_>>();
        let partition_hints = AdaptiveSubmapPartitionHints::default();
        let planned_windows = partition_ordered_submaps(
            features.len(),
            &pairwise,
            &hierarchical_config.partition,
            &partition_hints,
        )?;
        println!("hierarchical plan: {} submaps", planned_windows.len());
        for (index, window) in planned_windows.iter().enumerate() {
            println!(
                "  planned submap {index}: images {:?}, seam_support={}",
                window.image_range, window.outgoing_seam_support
            );
        }
        let started = std::time::Instant::now();
        let result = hierarchical_sfm(
            &args.camera,
            &source_frame_ids,
            &features,
            &pairwise,
            &pair_rotations,
            &partition_hints,
            &hierarchical_config,
        )?;
        println!(
            "hierarchical SfM completed in {:.1}s",
            started.elapsed().as_secs_f64()
        );
        export_hierarchical_result(&args, &features, &image_names, &result)?;
        return Ok(());
    }
    let t_sfm = std::time::Instant::now();
    let mut result = incremental_sfm(&args.camera, &features, &pairwise, &config)?;
    if args.wide_hypothesis && !pose_graph_pairwise.is_empty() {
        let wide_graph = merge_pairwise_graphs(&pairwise, &pose_graph_pairwise);
        let trusted_quality = reconstruction_quality(&result);
        let candidate_track_stats = preview_track_build_stats(&features, &wide_graph, &config);
        let preflight_accepted =
            accept_wide_track_preflight(result.track_build_stats, candidate_track_stats);
        println!(
            "wide hypothesis preflight: tracks={}->{} ({:.3}) observations={}->{} ({:.3}) \
             conflicts={}->{} accepted={}",
            result.track_build_stats.retained_tracks,
            candidate_track_stats.retained_tracks,
            candidate_track_stats.retained_tracks as f64
                / result.track_build_stats.retained_tracks.max(1) as f64,
            result.track_build_stats.retained_observations,
            candidate_track_stats.retained_observations,
            candidate_track_stats.retained_observations as f64
                / result.track_build_stats.retained_observations.max(1) as f64,
            result.track_build_stats.conflicting_components,
            candidate_track_stats.conflicting_components,
            preflight_accepted,
        );
        if preflight_accepted {
            match incremental_sfm(&args.camera, &features, &wide_graph, &config) {
                Ok(candidate) => {
                    let candidate_quality = reconstruction_quality(&candidate);
                    let accepted = accept_wide_hypothesis(trusted_quality, candidate_quality);
                    println!(
                        "wide hypothesis: pairs={} registered={}->{} points={}->{} ({:.3}) \
                     observations={}->{} ({:.3}) reproj={:.6}->{:.6} accepted={}",
                        wide_graph.len(),
                        trusted_quality.registered,
                        candidate_quality.registered,
                        trusted_quality.points,
                        candidate_quality.points,
                        candidate_quality.points as f64 / trusted_quality.points.max(1) as f64,
                        trusted_quality.observations,
                        candidate_quality.observations,
                        candidate_quality.observations as f64
                            / trusted_quality.observations.max(1) as f64,
                        trusted_quality.mean_reprojection_px,
                        candidate_quality.mean_reprojection_px,
                        accepted,
                    );
                    if accepted {
                        result = candidate;
                    }
                }
                Err(error) => println!("wide hypothesis rolled back after mapper error: {error}"),
            }
        }
    }
    let result_camera = result
        .refined_camera
        .clone()
        .unwrap_or_else(|| args.camera.clone());
    if !args.wide_hypothesis
        && args.pose_guided_track_augmentation
        && !pose_graph_pairwise.is_empty()
    {
        augment_tracks_with_pose_guided_edges(
            &result_camera,
            &features,
            &pose_graph_pairwise,
            args.min_matches,
            args.parallel_ba,
            &mut result,
        )?;
    } else if !args.wide_hypothesis && !pose_graph_pairwise.is_empty() {
        refine_with_pose_only_edges(
            &result_camera,
            &features,
            &pose_graph_pairwise,
            args.min_matches,
            args.parallel_ba,
            &mut result,
        )?;
    }
    println!(
        "track build: source={:?}, input={}, components={}, conflicts={} ({} obs), retained={} tracks / {} obs",
        args.track_source,
        result.track_build_stats.input_correspondences,
        result.track_build_stats.connected_components,
        result.track_build_stats.conflicting_components,
        result.track_build_stats.conflicting_observations,
        result.track_build_stats.retained_tracks,
        result.track_build_stats.retained_observations,
    );
    println!(
        "reconstruction: {} / {} images registered (post-refinement +{}, structure-less +{}, geometry-recovered {} tracks / {} obs, pose-ba={}), {} tracks, mean reproj {:.3} px [sfm {:.1}s]",
        result.registered_images,
        features.len(),
        result.post_refinement_registered_images,
        result.structureless_registered_images,
        result.geometry_recovered_tracks,
        result.geometry_recovered_observations,
        result.geometry_recovery_pose_ba_applied,
        result.tracks.len(),
        result.mean_reprojection_px,
        t_sfm.elapsed().as_secs_f64(),
    );

    // If intrinsics were refined, the poses/tracks are expressed against the
    // refined camera — export with it, not the input calibration.
    let export_camera = result
        .refined_camera
        .clone()
        .unwrap_or_else(|| args.camera.clone());
    if let Some(refined) = &result.refined_camera {
        if let (Some((fx0, fy0, cx0, cy0)), Some((fx, fy, cx, cy))) =
            (args.camera.intrinsics(), refined.intrinsics())
        {
            println!(
                "refined intrinsics: fx {fx0:.2}->{fx:.2}  fy {fy0:.2}->{fy:.2}  \
                 cx {cx0:.2}->{cx:.2}  cy {cy0:.2}->{cy:.2}",
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pose_only_offset_pairs_are_bounded_sorted_and_strided() {
        assert_eq!(
            fixed_offset_pairs(8, &[3, 0, 3], 2),
            vec![(0, 3), (2, 5), (4, 7)]
        );
        assert!(fixed_offset_pairs(1, &[2], 1).is_empty());
    }

    #[test]
    fn pose_only_coverage_rejects_clustered_and_accepts_spread_matches() {
        let camera = Camera::pinhole(0, 100, 100, 80.0, 80.0, 50.0, 50.0);
        let features = vec![
            FeatureSet {
                keypoints: vec![
                    Point2::new(10.0, 10.0),
                    Point2::new(90.0, 10.0),
                    Point2::new(10.0, 90.0),
                ],
                descriptors: vec![vec![0.0]; 3],
            },
            FeatureSet {
                keypoints: vec![
                    Point2::new(20.0, 20.0),
                    Point2::new(80.0, 20.0),
                    Point2::new(20.0, 80.0),
                ],
                descriptors: vec![vec![0.0]; 3],
            },
        ];
        let spread = PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0), (1, 1), (2, 2)],
        };
        let clustered = PairwiseMatches {
            matches: vec![(0, 0)],
            ..spread.clone()
        };
        let (spread_i, spread_j) = observation_coverage(&features, &spread, &camera);
        let (clustered_i, clustered_j) = observation_coverage(&features, &clustered, &camera);
        assert!(spread_i > 0.02 && spread_j > 0.02);
        assert_eq!((clustered_i, clustered_j), (0.0, 0.0));
    }

    #[test]
    fn pose_guided_augmentation_adds_only_low_error_unowned_observation() {
        let camera = Camera::pinhole(0, 100, 100, 80.0, 80.0, 50.0, 50.0);
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(50.0, 50.0)],
                descriptors: vec![vec![0.0]],
            },
            FeatureSet {
                // kp0 is the correct projection, kp1 competes for the same
                // track/image, and kp2 is already owned by another track.
                keypoints: vec![
                    Point2::new(34.0, 50.0),
                    Point2::new(35.5, 50.0),
                    Point2::new(66.0, 50.0),
                ],
                descriptors: vec![vec![0.0]; 3],
            },
        ];
        let poses = vec![
            Some(Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::identity(),
                nalgebra::Vector3::zeros(),
            )),
            Some(Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::identity(),
                nalgebra::Vector3::new(-1.0, 0.0, 0.0),
            )),
        ];
        let tracks = vec![
            visloc_rs::SfmTrack {
                position: Point3::new(0.0, 0.0, 5.0),
                observations: vec![(0, 0, Point2::new(50.0, 50.0))],
            },
            visloc_rs::SfmTrack {
                position: Point3::new(2.0, 0.0, 5.0),
                observations: vec![(1, 2, Point2::new(66.0, 50.0))],
            },
        ];
        let pairs = vec![PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 1), (0, 0), (0, 2)],
        }];
        let proposals =
            pose_guided_observation_proposals(&camera, &features, &poses, &tracks, &pairs, 2.0);
        assert_eq!(proposals.len(), 1);
        assert_eq!(
            (
                proposals[0].track,
                proposals[0].image,
                proposals[0].keypoint
            ),
            (0, 1, 0)
        );
        assert!(proposals[0].error_px < 1e-12);
    }

    #[test]
    fn pose_guided_merge_requires_disjoint_cross_reprojecting_fragments() {
        let camera = Camera::pinhole(0, 100, 100, 80.0, 80.0, 50.0, 50.0);
        let poses = vec![
            Some(Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::identity(),
                nalgebra::Vector3::zeros(),
            )),
            Some(Pose::from_world_to_camera(
                nalgebra::UnitQuaternion::identity(),
                nalgebra::Vector3::new(-1.0, 0.0, 0.0),
            )),
        ];
        let tracks = vec![
            visloc_rs::SfmTrack {
                position: Point3::new(0.0, 0.0, 5.0),
                observations: vec![(0, 0, Point2::new(50.0, 50.0))],
            },
            visloc_rs::SfmTrack {
                position: Point3::new(0.0, 0.0, 5.0),
                observations: vec![(1, 0, Point2::new(34.0, 50.0))],
            },
        ];
        let pairs = vec![PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
        }];
        let merges = pose_guided_track_merge_proposals(&camera, &poses, &tracks, &pairs, 2.0);
        assert_eq!(merges.len(), 1);
        assert_eq!((merges[0].track_a, merges[0].track_b), (0, 1));

        let mut overlapping = tracks;
        overlapping[0]
            .observations
            .push((1, 1, Point2::new(34.0, 50.0)));
        assert!(
            pose_guided_track_merge_proposals(&camera, &poses, &overlapping, &pairs, 2.0)
                .is_empty()
        );
    }

    #[test]
    fn pose_guided_new_track_requires_three_valid_wide_views() {
        let camera = Camera::pinhole(0, 100, 100, 80.0, 80.0, 50.0, 50.0);
        let features = vec![50.0, 34.0, 18.0]
            .into_iter()
            .map(|x| FeatureSet {
                keypoints: vec![Point2::new(x, 50.0)],
                descriptors: vec![vec![0.0]],
            })
            .collect::<Vec<_>>();
        let poses = (0..3)
            .map(|image| {
                Some(Pose::from_world_to_camera(
                    nalgebra::UnitQuaternion::identity(),
                    nalgebra::Vector3::new(-(image as f64), 0.0, 0.0),
                ))
            })
            .collect::<Vec<_>>();
        let pairs = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
            },
        ];
        let tracks = pose_guided_new_tracks(&camera, &features, &poses, &[], &pairs, 3, 2.0, 2.0);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].observations.len(), 3);
        assert!((tracks[0].position - Point3::new(0.0, 0.0, 5.0)).norm() < 1e-9);
        assert!(pose_guided_new_tracks(
            &camera,
            &features[..2],
            &poses[..2],
            &[],
            &pairs[..1],
            3,
            2.0,
            2.0,
        )
        .is_empty());
    }

    #[test]
    fn wide_hypothesis_selector_accepts_mh05_signature_and_rejects_mh03_collapse() {
        let mh05_trusted = ReconstructionQuality {
            registered: 300,
            points: 6641,
            observations: 63532,
            mean_reprojection_px: 0.644,
        };
        let mh05_wide = ReconstructionQuality {
            registered: 300,
            points: 6513,
            observations: 101178,
            mean_reprojection_px: 0.572,
        };
        assert!(accept_wide_hypothesis(mh05_trusted, mh05_wide));

        let mh03_trusted = ReconstructionQuality {
            registered: 300,
            points: 8254,
            observations: 131742,
            mean_reprojection_px: 0.570,
        };
        let mh03_wide = ReconstructionQuality {
            registered: 300,
            points: 2602,
            observations: 95629,
            mean_reprojection_px: 0.708,
        };
        assert!(!accept_wide_hypothesis(mh03_trusted, mh03_wide));
    }

    #[test]
    fn wide_track_preflight_accepts_mh05_and_early_rejects_mh03() {
        let mh05_base = TrackBuildStats {
            input_correspondences: 1_544_485,
            retained_tracks: 18_335,
            retained_observations: 185_870,
            ..TrackBuildStats::default()
        };
        let mh05_wide = TrackBuildStats {
            input_correspondences: 1_702_992,
            retained_tracks: 15_689,
            retained_observations: 168_205,
            ..TrackBuildStats::default()
        };
        assert!(accept_wide_track_preflight(mh05_base, mh05_wide));

        let mh03_base = TrackBuildStats {
            input_correspondences: 2_033_967,
            retained_tracks: 24_038,
            retained_observations: 248_633,
            ..TrackBuildStats::default()
        };
        let mh03_wide = TrackBuildStats {
            input_correspondences: 2_280_221,
            retained_tracks: 18_648,
            retained_observations: 203_722,
            ..TrackBuildStats::default()
        };
        assert!(!accept_wide_track_preflight(mh03_base, mh03_wide));
    }

    #[test]
    fn merged_pairwise_graph_deduplicates_overlapping_edges_and_matches() {
        let base = vec![PairwiseMatches {
            image_i: 0,
            image_j: 2,
            matches: vec![(1, 3), (2, 4)],
        }];
        let wide = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 2,
                matches: vec![(1, 3), (5, 6)],
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 3,
                matches: vec![(7, 8)],
            },
        ];
        let merged = merge_pairwise_graphs(&base, &wide);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].matches, vec![(1, 3), (2, 4), (5, 6)]);
        assert_eq!((merged[1].image_i, merged[1].image_j), (1, 3));
    }
}
