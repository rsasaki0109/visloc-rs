//! Camera relocalization benchmark on the REAL Microsoft 7-Scenes dataset.
//!
//! This exercises the `visloc-rs` map-reuse / localization stack end-to-end on a
//! standard relocalization benchmark — the literal "GPS-denied localization =
//! prior map + PnP" capability the README advertises — and reports the metric
//! the literature uses (median translation / rotation error and the fraction of
//! queries localized within 5 cm / 5 deg).
//!
//! Pipeline (no synthetic ingredients — real images, real depth, real poses):
//!
//!   1. **Build a metric 3D map from the train sequences.** For every sampled
//!      train keyframe we extract features, read the registered depth (mm), and
//!      back-project each keypoint to a 3D point in the camera frame, then lift it
//!      to the world frame through the frame's ground-truth camera-to-world pose.
//!      Each lifted point becomes a descriptor-tagged landmark.
//!   2. **Localize every test frame against that prior map.** Features from the
//!      query image are matched to the map's landmark descriptors and the camera
//!      pose is recovered with PnP + RANSAC (`LocalizationPipeline`).
//!   3. **Score against ground truth.** Translation error is the distance between
//!      the estimated and ground-truth camera centers; rotation error is the
//!      angle between the estimated and ground-truth orientations.
//!
//! 7-Scenes "chess" default split: train = seq 1,2,4,6 ; test = seq 3,5. The
//! depth files are the dataset's provided (Kinect) depth used directly with the
//! standard RGB intrinsic (f = 585, principal point = (320, 240), 640x480); this
//! is the common pragmatic choice and is reported as such.
//!
//! Features: classical corners by default (a weak baseline that essentially
//! fails to relocalize across 7-Scenes' textureless, repetitive surfaces), or
//! pre-exported SuperPoint descriptors via `--sp-features-dir`. Export them with
//! `scripts/export_superpoint_7scenes.py` (needs torch + lightglue):
//!
//!   python3 scripts/export_superpoint_7scenes.py --dataset /path/to/7scenes/chess \
//!       --seqs 1,2,3,4,5,6 --stride 10 --out-dir /tmp/sp7_chess
//!
//! With `--retrieve-topk K` (default 10) the query is matched only against the
//! K appearance-nearest train keyframes (hierarchical localization), which both
//! raises recall and cuts the brute-force cost. By default each retrieved
//! keyframe is matched and PnP'd SEPARATELY and the pose with the most inliers
//! is kept — every keyframe's depth-lifted points form one internally-consistent
//! 3D set, so this avoids mixing 3D from keyframes whose depth registers slightly
//! differently. That mixing is the dominant ACCURACY limiter: on chess this
//! per-keyframe PnP lifts the 5cm/5deg fraction from ~9% to ~42% AND doubles
//! recall versus pooling all keyframes' correspondences into one PnP
//! (`--accumulate-corrs`), which in turn beats merging descriptor clouds
//! (`--merged-submap`). Recall/precision still trades on `--retrieve-topk` /
//! `--ratio` / `--min-inliers`.
//!
//! To benchmark a learned matcher (LightGlue) against the in-Rust BruteForce
//! baseline under identical pose estimation, pass `--correspondences-dir`: the
//! matcher is run out-of-process (`scripts/hloc_lightglue_7scenes.py`) and emits
//! per-query 2D-3D correspondences; this mode skips the map build and only
//! exercises the Rust PnP+RANSAC. With `--grouped-corrs` the files carry a
//! leading keyframe column ("KF x y X Y Z conf") and each keyframe group is
//! PnP'd separately (per-keyframe PnP, as above). LightGlue + per-keyframe PnP
//! is the strongest combination measured here on chess (~99% localized, median
//! ~3.3cm / ~1.7deg, ~72% within 5cm/5deg) — the learned matcher and the
//! per-keyframe-consistent 3D compound; neither alone gets close.
//!
//! Run (after extracting chess.zip and its inner seq-*.zip):
//!   cargo run --release --features image-io --example relocalization_7scenes_demo -- \
//!       --dataset /path/to/7scenes/chess --sp-features-dir /tmp/sp7_chess \
//!       --train-stride 10 --test-stride 10 --retrieve-topk 15 --ratio 0.9 --reproj 6

use std::error::Error;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use visloc_rs::prelude::*;

struct Args {
    dataset: PathBuf,
    train_seqs: Vec<u32>,
    test_seqs: Vec<u32>,
    train_stride: usize,
    test_stride: usize,
    max_features: usize,
    min_depth: f64,
    max_depth: f64,
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    width: u32,
    height: u32,
    frames_per_seq: usize,
    sp_features_dir: Option<PathBuf>,
    min_inliers: usize,
    retrieve_topk: usize,
    ratio: f32,
    reproj: f64,
    merged_submap: bool,
    accumulate_corrs: bool,
    correspondences_dir: Option<PathBuf>,
    grouped_corrs: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            dataset: PathBuf::new(),
            train_seqs: vec![1, 2, 4, 6],
            test_seqs: vec![3, 5],
            train_stride: 20,
            test_stride: 50,
            max_features: 800,
            min_depth: 0.3,
            max_depth: 4.0,
            fx: 585.0,
            fy: 585.0,
            cx: 320.0,
            cy: 240.0,
            width: 640,
            height: 480,
            frames_per_seq: 1000,
            sp_features_dir: None,
            min_inliers: 12,
            retrieve_topk: 10,
            ratio: 0.8,
            reproj: 4.0,
            merged_submap: false,
            accumulate_corrs: false,
            correspondences_dir: None,
            grouped_corrs: false,
        }
    }
}

/// L2-normalized mean of a descriptor set — a crude global image descriptor for
/// appearance-based keyframe retrieval (bag-of-features centroid).
fn normalized_mean(descriptors: &[Vec<f32>]) -> Vec<f32> {
    let Some(first) = descriptors.first() else {
        return Vec::new();
    };
    let dim = first.len();
    let mut acc = vec![0f32; dim];
    for d in descriptors {
        for (a, v) in acc.iter_mut().zip(d) {
            *a += v;
        }
    }
    let n = descriptors.len() as f32;
    let mut norm = 0f32;
    for a in acc.iter_mut() {
        *a /= n;
        norm += *a * *a;
    }
    let norm = norm.sqrt();
    if norm > 0.0 {
        for a in acc.iter_mut() {
            *a /= norm;
        }
    }
    acc
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One train keyframe's retrieval signature plus its own landmark cloud. Keeping
/// each keyframe's landmarks separate (rather than only their global-map ids) is
/// what lets us match the query against ONE keyframe at a time: when many
/// keyframes are merged into a single descriptor cloud, the same physical point
/// — independently depth-lifted by each keyframe into a near-duplicate
/// descriptor — makes the Lowe ratio test reject the match (1st and 2nd nearest
/// are duplicates), which silently caps recall.
struct KeyframeIndex {
    global: Vec<f32>,
    landmark_ids: Vec<u64>,
    positions: Vec<Point3<f64>>,
    descriptors: Vec<Vec<f32>>,
}

fn parse_seqs(s: &str) -> Vec<u32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<u32>().ok())
        .collect()
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("missing value for {flag}"));
        match flag.as_str() {
            "--dataset" => args.dataset = PathBuf::from(val()?),
            "--train-seqs" => args.train_seqs = parse_seqs(&val()?),
            "--test-seqs" => args.test_seqs = parse_seqs(&val()?),
            "--train-stride" => args.train_stride = val()?.parse()?,
            "--test-stride" => args.test_stride = val()?.parse()?,
            "--max-features" => args.max_features = val()?.parse()?,
            "--min-depth" => args.min_depth = val()?.parse()?,
            "--max-depth" => args.max_depth = val()?.parse()?,
            "--focal" => {
                let f: f64 = val()?.parse()?;
                args.fx = f;
                args.fy = f;
            }
            "--frames-per-seq" => args.frames_per_seq = val()?.parse()?,
            "--sp-features-dir" => args.sp_features_dir = Some(PathBuf::from(val()?)),
            "--min-inliers" => args.min_inliers = val()?.parse()?,
            "--retrieve-topk" => args.retrieve_topk = val()?.parse()?,
            "--ratio" => args.ratio = val()?.parse()?,
            "--reproj" => args.reproj = val()?.parse()?,
            "--merged-submap" => args.merged_submap = true,
            "--accumulate-corrs" => args.accumulate_corrs = true,
            "--correspondences-dir" => args.correspondences_dir = Some(PathBuf::from(val()?)),
            "--grouped-corrs" => args.grouped_corrs = true,
            other => return Err(format!("unknown flag: {other}").into()),
        }
    }
    if args.dataset.as_os_str().is_empty() {
        return Err("missing --dataset <path to 7scenes/chess>".into());
    }
    Ok(args)
}

/// Parse a 7-Scenes `frame-NNNNNN.pose.txt`: a 4x4 camera-to-world matrix.
/// Returns (R_cw, t_cw) where world = R_cw * camera + t_cw.
fn read_pose(path: &Path) -> Result<(Matrix3<f64>, Vector3<f64>), Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let v: Vec<f64> = text
        .split_whitespace()
        .filter_map(|t| t.parse::<f64>().ok())
        .collect();
    if v.len() < 12 {
        return Err(format!("pose file {} has {} values", path.display(), v.len()).into());
    }
    let r = Matrix3::new(v[0], v[1], v[2], v[4], v[5], v[6], v[8], v[9], v[10]);
    let t = Vector3::new(v[3], v[7], v[11]);
    Ok((r, t))
}

/// Parse a per-query correspondence file: lines "x y X Y Z [confidence]" where
/// (x,y) is a query pixel and (X,Y,Z) the matched world point. Produced by an
/// external matcher (e.g. LightGlue) so only Rust pose estimation is exercised.
fn read_correspondences(path: &Path) -> Result<Vec<Correspondence2D3D>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut out = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if t.len() < 5 {
            continue;
        }
        out.push(Correspondence2D3D {
            point2d: Point2::new(t[0], t[1]),
            point3d: Point3::new(t[2], t[3], t[4]),
            confidence: t.get(5).map(|&c| c as f32),
        });
    }
    Ok(out)
}

/// Parse a per-query GROUPED correspondence file: lines "KF x y X Y Z [conf]"
/// where KF is the source keyframe index. Returns the correspondences grouped by
/// keyframe so each group can be PnP'd on its own internally-consistent 3D set.
fn read_grouped_correspondences(
    path: &Path,
) -> Result<Vec<Vec<Correspondence2D3D>>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut groups: std::collections::HashMap<i64, Vec<Correspondence2D3D>> =
        std::collections::HashMap::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let t: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse::<f64>().ok())
            .collect();
        if t.len() < 6 {
            continue;
        }
        groups
            .entry(t[0] as i64)
            .or_default()
            .push(Correspondence2D3D {
                point2d: Point2::new(t[1], t[2]),
                point3d: Point3::new(t[3], t[4], t[5]),
                confidence: t.get(6).map(|&c| c as f32),
            });
    }
    Ok(groups.into_values().collect())
}

/// Per-keyframe PnP: estimate a pose from each keyframe's correspondence group on
/// its own, and keep the pose with the most inliers (>= `min_inliers`).
fn per_keyframe_best_pose(
    groups: &[Vec<Correspondence2D3D>],
    estimator: &PnPRansac,
    camera: &Camera,
    min_inliers: usize,
) -> Option<(Pose, usize)> {
    let mut best: Option<(Pose, usize)> = None;
    for corrs in groups {
        if let Some(r) = estimator.estimate(corrs, camera) {
            if r.inliers.len() >= min_inliers
                && best.as_ref().map_or(true, |(_, n)| r.inliers.len() > *n)
            {
                best = Some((r.pose, r.inliers.len()));
            }
        }
    }
    best
}

fn frame_base(dataset: &Path, seq: u32, idx: usize) -> PathBuf {
    dataset
        .join(format!("seq-{seq:02}"))
        .join(format!("frame-{idx:06}"))
}

/// Keypoints + descriptors for a frame. With `--sp-features-dir` set, loads
/// pre-exported SuperPoint features (`seq-SS_frame-NNNNNN.txt`, "X Y SCORE D...");
/// otherwise extracts classical corner features from the color image. Returns
/// `None` if the frame's source file is absent.
fn frame_features(
    args: &Args,
    extractor: &CornerFeatureExtractor,
    seq: u32,
    idx: usize,
) -> Result<Option<(Vec<Point2<f64>>, Vec<Vec<f32>>)>, Box<dyn Error>> {
    if let Some(dir) = &args.sp_features_dir {
        let path = dir.join(format!("seq-{seq:02}_frame-{idx:06}.txt"));
        if !path.exists() {
            return Ok(None);
        }
        let set = read_external_deep_features_txt(&path)?;
        // SuperPoint files are score-ordered; keep the strongest `max_features`.
        let mut keypoints = set.keypoints();
        let mut descriptors = set.descriptors();
        keypoints.truncate(args.max_features);
        descriptors.truncate(args.max_features);
        Ok(Some((keypoints, descriptors)))
    } else {
        let color = frame_base(&args.dataset, seq, idx).with_extension("color.png");
        if !color.exists() {
            return Ok(None);
        }
        let gray = read_common_image(&color)?;
        let feats = extractor.extract(&gray)?;
        Ok(Some((feats.keypoints, feats.descriptors)))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let camera = Camera::pinhole(
        0,
        args.width,
        args.height,
        args.fx,
        args.fy,
        args.cx,
        args.cy,
    );
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: args.max_features,
        ..CornerFeatureConfig::default()
    });

    let feature_mode = match &args.sp_features_dir {
        Some(dir) => format!("superpoint ({})", dir.display()),
        None => "classical-corner".to_string(),
    };
    println!(
        "dataset={} train_seqs={:?} test_seqs={:?} train_stride={} test_stride={} features={} min_inliers={} retrieve_topk={}",
        args.dataset.display(),
        args.train_seqs,
        args.test_seqs,
        args.train_stride,
        args.test_stride,
        feature_mode,
        args.min_inliers,
        args.retrieve_topk,
    );

    // ---- 1. Build the metric map from train keyframes. ----
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut store = LandmarkDescriptorStore::new();
    let mut next_id: u64 = 0;
    let mut train_keyframes = 0usize;
    let mut keyframe_index: Vec<KeyframeIndex> = Vec::new();

    // In --correspondences-dir mode the 2D-3D matches are supplied externally, so
    // the in-process map / descriptor matching is not needed at all.
    if args.correspondences_dir.is_none() {
        for &seq in &args.train_seqs {
            for idx in (0..args.frames_per_seq).step_by(args.train_stride.max(1)) {
                let base = frame_base(&args.dataset, seq, idx);
                let Some((keypoints, descriptors)) = frame_features(&args, &extractor, seq, idx)?
                else {
                    continue;
                };
                let depth_path = base.with_extension("depth.png");
                if !depth_path.exists() {
                    continue;
                }
                let depth = image::open(&depth_path)?.into_luma16();
                let (r_cw, t_cw) = read_pose(&base.with_extension("pose.txt"))?;
                train_keyframes += 1;

                let mut kf_ids = Vec::new();
                let mut kf_descs = Vec::new();
                let mut kf_positions = Vec::new();
                for (kp, desc) in keypoints.iter().zip(descriptors.iter()) {
                    let u = kp.x.round().clamp(0.0, (args.width - 1) as f64) as u32;
                    let v = kp.y.round().clamp(0.0, (args.height - 1) as f64) as u32;
                    let raw = depth.get_pixel(u, v)[0];
                    if raw == 0 || raw == 65535 {
                        continue;
                    }
                    let d = raw as f64 / 1000.0;
                    if d < args.min_depth || d > args.max_depth {
                        continue;
                    }
                    // Back-project pixel to the camera frame, then lift to world.
                    let p_cam = Point3::new(
                        (kp.x - args.cx) / args.fx * d,
                        (kp.y - args.cy) / args.fy * d,
                        d,
                    );
                    let p_world = Point3::from(r_cw * p_cam.coords + t_cw);
                    let mut lm = Landmark::new(next_id, p_world);
                    lm.descriptor = Some(desc.clone());
                    map.landmarks.insert(next_id, lm);
                    store.insert(next_id, desc.clone());
                    kf_ids.push(next_id);
                    kf_descs.push(desc.clone());
                    kf_positions.push(p_world);
                    next_id += 1;
                }
                if !kf_ids.is_empty() {
                    keyframe_index.push(KeyframeIndex {
                        global: normalized_mean(&kf_descs),
                        landmark_ids: kf_ids,
                        positions: kf_positions,
                        descriptors: kf_descs,
                    });
                }
            }
        }
    }
    println!(
        "map built: {train_keyframes} train keyframes -> {} landmarks",
        map.landmarks.len()
    );

    let provider = InMemoryMapProvider::with_descriptor_store(map, store);
    let mut pipeline = LocalizationPipeline::default();
    pipeline.config.min_inliers = args.min_inliers;
    pipeline.config.ratio = Some(args.ratio);
    pipeline.config.reprojection_threshold = args.reproj;
    pipeline.matcher = BruteForceMatcher {
        ratio: Some(args.ratio),
    };

    // ---- 2./3. Localize each test frame and score against ground truth. ----
    // Queries are independent: build the job list, then localize in parallel.
    let jobs: Vec<(u32, usize)> = args
        .test_seqs
        .iter()
        .flat_map(|&seq| {
            (0..args.frames_per_seq)
                .step_by(args.test_stride.max(1))
                .map(move |idx| (seq, idx))
        })
        .collect();

    // Per-query outcome: None = no source frame (skip); Some((localized, trans, rot, inliers)).
    let outcomes: Vec<Option<(bool, f64, f64, usize)>> = jobs
        .par_iter()
        .map(|&(seq, idx)| {
            let base = frame_base(&args.dataset, seq, idx);
            let (r_cw_gt, t_cw_gt) = read_pose(&base.with_extension("pose.txt")).ok()?;
            let estimator = PnPRansac {
                reprojection_threshold: args.reproj,
                iterations: 256,
                ..PnPRansac::default()
            };

            // Localize. Returns the estimated world->camera pose and inlier count.
            let localized_pose: Option<(Pose, usize)> = if let Some(dir) = &args.correspondences_dir
            {
                // Externally-matched 2D-3D correspondences (e.g. LightGlue), one
                // file per query: "x y X Y Z [conf]". Only the Rust PnP+RANSAC is
                // exercised, so this measures a learned matcher against the
                // in-pipeline BruteForce baseline under identical pose estimation.
                let path = dir.join(format!("seq-{seq:02}_frame-{idx:06}.corr.txt"));
                if !path.exists() {
                    return None;
                }
                if args.grouped_corrs {
                    // "KF x y X Y Z conf": PnP each keyframe group, keep best.
                    let groups = read_grouped_correspondences(&path).ok()?;
                    per_keyframe_best_pose(&groups, &estimator, &camera, args.min_inliers)
                } else {
                    let corrs = read_correspondences(&path).ok()?;
                    estimator
                        .estimate(&corrs, &camera)
                        .filter(|r| r.inliers.len() >= args.min_inliers)
                        .map(|r| (r.pose, r.inliers.len()))
                }
            } else if let Some((keypoints, descriptors)) =
                frame_features(&args, &extractor, seq, idx).ok()?
            {
                if args.retrieve_topk > 0 && !keyframe_index.is_empty() {
                    // Appearance-based retrieval: top-K most similar train keyframes.
                    let qg = normalized_mean(&descriptors);
                    let mut scored: Vec<(f32, usize)> = keyframe_index
                        .iter()
                        .enumerate()
                        .map(|(i, kf)| (dot(&qg, &kf.global), i))
                        .collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let topk = scored.iter().take(args.retrieve_topk).map(|&(_, i)| i);

                    if args.merged_submap {
                        // Merge the retrieved keyframes into one cloud, then one
                        // match + PnP via the pipeline (suffers the duplicate-
                        // descriptor ratio-test pathology described on KeyframeIndex).
                        let mut ids: Vec<u64> = Vec::new();
                        for i in topk {
                            ids.extend_from_slice(&keyframe_index[i].landmark_ids);
                        }
                        let submap = InMemoryMapProvider::from_provider_landmarks(&provider, ids);
                        let query = QueryImage {
                            camera: camera.clone(),
                            keypoints,
                            descriptors,
                        };
                        let r = pipeline.localize_with_provider(&query, &submap);
                        r.pose.filter(|_| r.success).map(|p| (p, r.inlier_count))
                    } else {
                        let matcher = BruteForceMatcher {
                            ratio: Some(args.ratio),
                        };
                        if args.accumulate_corrs {
                            // Match each keyframe separately, keep the single best
                            // match per query keypoint across keyframes, then ONE PnP
                            // over the pooled correspondences. Pooling mixes 3D points
                            // from keyframes whose depth registers slightly
                            // differently — an internally-inconsistent patchwork that
                            // caps pose accuracy (kept as an ablation knob).
                            let mut best: std::collections::HashMap<
                                usize,
                                (f32, Point3<f64>, Option<f32>),
                            > = std::collections::HashMap::new();
                            for i in topk {
                                let kf = &keyframe_index[i];
                                for m in matcher.match_descriptors(&descriptors, &kf.descriptors) {
                                    let Some(&point3d) = kf.positions.get(m.train_index) else {
                                        continue;
                                    };
                                    best.entry(m.query_index)
                                        .and_modify(|e| {
                                            if m.distance < e.0 {
                                                *e = (m.distance, point3d, m.confidence);
                                            }
                                        })
                                        .or_insert((m.distance, point3d, m.confidence));
                                }
                            }
                            let corrs: Vec<Correspondence2D3D> = best
                                .into_iter()
                                .filter_map(|(qi, (_, point3d, confidence))| {
                                    keypoints.get(qi).map(|&point2d| Correspondence2D3D {
                                        point2d,
                                        point3d,
                                        confidence,
                                    })
                                })
                                .collect();
                            estimator
                                .estimate(&corrs, &camera)
                                .filter(|r| r.inliers.len() >= args.min_inliers)
                                .map(|r| (r.pose, r.inliers.len()))
                        } else {
                            // Per-keyframe PnP (the default): match and PnP against
                            // EACH retrieved keyframe on its own — each keyframe's
                            // depth-lifted points form ONE internally-consistent 3D
                            // set — and keep the pose with the most inliers. This
                            // keeps top-K recall while recovering single-keyframe
                            // accuracy, and is the dominant accuracy lever (mixing
                            // keyframes' independently-registered depth is what hurt).
                            let mut best_pose: Option<(Pose, usize)> = None;
                            for i in topk {
                                let kf = &keyframe_index[i];
                                let corrs: Vec<Correspondence2D3D> = matcher
                                    .match_descriptors(&descriptors, &kf.descriptors)
                                    .iter()
                                    .filter_map(|m| {
                                        match (
                                            keypoints.get(m.query_index),
                                            kf.positions.get(m.train_index),
                                        ) {
                                            (Some(&point2d), Some(&point3d)) => {
                                                Some(Correspondence2D3D {
                                                    point2d,
                                                    point3d,
                                                    confidence: m.confidence,
                                                })
                                            }
                                            _ => None,
                                        }
                                    })
                                    .collect();
                                if let Some(r) = estimator.estimate(&corrs, &camera) {
                                    if r.inliers.len() >= args.min_inliers
                                        && best_pose
                                            .as_ref()
                                            .map_or(true, |(_, n)| r.inliers.len() > *n)
                                    {
                                        best_pose = Some((r.pose, r.inliers.len()));
                                    }
                                }
                            }
                            best_pose
                        }
                    }
                } else {
                    // No retrieval: match against the whole global map.
                    let query = QueryImage {
                        camera: camera.clone(),
                        keypoints,
                        descriptors,
                    };
                    let r = pipeline.localize_with_provider(&query, &provider);
                    r.pose.filter(|_| r.success).map(|p| (p, r.inlier_count))
                }
            } else {
                return None;
            };

            let Some((pose, inliers)) = localized_pose else {
                return Some((false, 0.0, 0.0, 0));
            };

            let trans_err = (pose.camera_center_world() - Point3::from(t_cw_gt)).norm();
            // Ground-truth world->camera rotation is the transpose of camera->world.
            let gt_wc = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
                r_cw_gt.transpose(),
            ));
            let rot_err = pose.world_to_camera.rotation.angle_to(&gt_wc).to_degrees();
            Some((true, trans_err, rot_err, inliers))
        })
        .collect();

    let mut total = 0usize;
    let mut localized = 0usize;
    let mut trans_errors: Vec<f64> = Vec::new();
    let mut rot_errors: Vec<f64> = Vec::new();
    let mut within_5_5 = 0usize;
    let mut inlier_sum = 0usize;
    for outcome in outcomes.into_iter().flatten() {
        total += 1;
        let (ok, trans_err, rot_err, inliers) = outcome;
        if !ok {
            continue;
        }
        localized += 1;
        inlier_sum += inliers;
        if trans_err <= 0.05 && rot_err <= 5.0 {
            within_5_5 += 1;
        }
        trans_errors.push(trans_err);
        rot_errors.push(rot_err);
    }

    let median = |v: &mut Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    println!("--- relocalization results ---");
    println!("test frames: {total}");
    println!(
        "localized (PnP success): {localized} ({:.1}%)",
        100.0 * localized as f64 / total.max(1) as f64
    );
    if localized > 0 {
        println!(
            "mean inliers / localized: {:.1}",
            inlier_sum as f64 / localized as f64
        );
        println!(
            "median translation error: {:.3} m",
            median(&mut trans_errors)
        );
        println!(
            "median rotation error:    {:.2} deg",
            median(&mut rot_errors)
        );
        println!(
            "within 5cm/5deg: {within_5_5} ({:.1}% of all test frames)",
            100.0 * within_5_5 as f64 / total.max(1) as f64
        );
    }

    Ok(())
}
