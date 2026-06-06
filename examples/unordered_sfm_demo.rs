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
//! cargo run --release --example unordered_sfm_demo -- \
//!     --features-dir /tmp/sp_photos \
//!     --feature-suffix _features.txt --image-suffix .png \
//!     --width 752 --height 480 --fx 458.6 --fy 457.3 --cx 367.2 --cy 248.4 \
//!     --retrieval-topk 12 --min-matches 30 \
//!     --out-colmap /tmp/photos_sfm_colmap
//! ```

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3};
use rayon::prelude::*;
use visloc_rs::vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_rs::vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};
use visloc_rs::{
    incremental_sfm, read_external_deep_features_txt, write_colmap_reconstruction_for_3dgs,
    BruteForceMatcher, Camera, CrossCheckMatcher, FeatureSet, IncrementalSfmConfig, Matcher,
    PairwiseMatches, Pose,
};

/// A COLMAP-export landmark: world position + `(image, keypoint, pixel)` track.
type ExportLandmark = (Point3<f64>, Vec<(usize, usize, Point2<f64>)>);

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

/// Match and geometrically verify each candidate pair into `PairwiseMatches`.
/// Candidate pairs are independent, so the (descriptor-matching dominated) loop
/// is run across cores with rayon.
fn verify_pairs(
    features: &[FeatureSet],
    camera: &Camera,
    candidates: &[(usize, usize)],
    match_ratio: f32,
    min_matches: usize,
) -> Vec<PairwiseMatches> {
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
            Some(PairwiseMatches {
                image_i: i,
                image_j: j,
                matches,
            })
        })
        .collect()
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

    let pairwise = verify_pairs(
        &features,
        &args.camera,
        &candidates,
        args.match_ratio,
        args.min_matches,
    );
    let verified_matches: usize = pairwise.iter().map(|p| p.matches.len()).sum();
    println!(
        "verified {} / {} pairs, {} inlier correspondences",
        pairwise.len(),
        candidates.len(),
        verified_matches,
    );
    if pairwise.is_empty() {
        return Err("no pair survived geometric verification — lower --min-matches?".into());
    }

    let config = IncrementalSfmConfig {
        min_seed_matches: args.min_matches,
        min_pnp_inliers: args.min_pnp_inliers,
        max_reprojection_error_px: args.max_reproj,
        final_global_ba: args.final_ba,
        seed_trials: args.seed_trials,
        ..IncrementalSfmConfig::default()
    };
    let result = incremental_sfm(&args.camera, &features, &pairwise, &config)?;
    println!(
        "reconstruction: {} / {} images registered, {} tracks, mean reproj {:.3} px",
        result.registered_images,
        features.len(),
        result.tracks.len(),
        result.mean_reprojection_px,
    );

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
    Ok(())
}
