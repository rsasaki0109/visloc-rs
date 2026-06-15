//! Multi-session **lifelong** mapping on the REAL Microsoft 7-Scenes dataset,
//! driven by learned visual-place-recognition retrieval.
//!
//! The existing `multi_session_kitti_merge_demo` merges *synthetic* pose graphs
//! over a hand-supplied bridge edge — no images, no features, no retrieval. This
//! demo exercises the genuine lifelong-SLAM loop the README advertises: a map
//! built once from an initial session is **grown across later sessions by
//! relocalizing each new traversal against the map so far and folding its
//! observations in**, with no ground-truth poses for the later sessions. The
//! only thing that lets a later keyframe attach to the map is appearance-based
//! retrieval surfacing a relevant prior keyframe — so retrieval quality is the
//! binding constraint, and a learned global descriptor (EigenPlaces) is measured
//! against the bag-of-features baseline (`normalized_mean`) on exactly that.
//!
//! Pipeline (real images, real depth; GT poses used ONLY to seed session 0 and
//! to score, never to merge later sessions):
//!
//!   1. **Bootstrap** the metric map from the first session's keyframes using
//!      their ground-truth camera-to-world poses (the "surveyed" deployment map):
//!      back-project each keypoint through the registered depth and lift to world.
//!   2. **Integrate each later session** keyframe-by-keyframe. For every keyframe
//!      we relocalize against the map *so far* (learned-retrieval top-K -> per-
//!      keyframe PnP, the dominant accuracy lever from `relocalization_7scenes_demo`).
//!      If it localizes (>= min-inliers), we lift its own keypoints through its
//!      depth using the ESTIMATED pose and append them as a new keyframe — the map
//!      grows. If it does not, it is dropped (no GT fallback). The map can bridge
//!      within a session: a keyframe can attach to an earlier same-session keyframe
//!      that already merged.
//!   3. **Evaluate** the held-out test sessions against (a) the bootstrap-only map
//!      and (b) the full lifelong map, reporting localization rate, median
//!      translation/rotation error, and the 5 cm / 5 deg fraction. A broader
//!      lifelong map should localize more test frames; the merge step's own
//!      relocalization error (vs GT) confirms the grown map stays metric.
//!
//! A/B the retrieval front-end by running twice: with `--global-descriptor-dir`
//! (learned EigenPlaces globals from `scripts/export_vpr_globals_7scenes.py`) vs
//! without (bag-of-features `normalized_mean`). The headline is how many later-
//! session keyframes each manages to merge, and the resulting test recall.
//!
//! 7-Scenes "chess" default split: sessions = seq 1,2,4,6 (seq 1 bootstraps),
//! test = seq 3,5. Run (after exporting SuperPoint + EigenPlaces globals):
//!   cargo run --release --features image-io --example multi_session_lifelong_demo -- \
//!       --dataset /path/to/7scenes/chess --sp-features-dir /tmp/sp_7scenes_chess \
//!       --global-descriptor-dir /tmp/vpr_7scenes_chess \
//!       --sessions 1,2,4,6 --test-seqs 3,5 --session-stride 20 --test-stride 20 \
//!       --retrieve-topk 15 --ratio 0.9 --reproj 6

use std::error::Error;
use std::path::{Path, PathBuf};

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use rayon::prelude::*;
use visloc_rs::prelude::*;

struct Args {
    dataset: PathBuf,
    sessions: Vec<u32>,
    test_seqs: Vec<u32>,
    session_stride: usize,
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
    global_descriptor_dir: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            dataset: PathBuf::new(),
            sessions: vec![1, 2, 4, 6],
            test_seqs: vec![3, 5],
            session_stride: 20,
            test_stride: 20,
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
            retrieve_topk: 15,
            ratio: 0.9,
            reproj: 6.0,
            global_descriptor_dir: None,
        }
    }
}

/// L2-normalized mean of a descriptor set — a crude global image descriptor for
/// appearance-based keyframe retrieval (bag-of-features centroid), the baseline
/// the learned EigenPlaces global is measured against.
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

/// Load a precomputed learned global descriptor for one frame from
/// `<dir>/seq-SS_frame-NNNNNN.txt`, written by `export_vpr_globals_7scenes.py`.
/// `None` (dir unset / file absent) makes the caller fall back to `normalized_mean`.
fn frame_global(dir: Option<&PathBuf>, seq: u32, idx: usize) -> Option<Vec<f32>> {
    let path = dir?.join(format!("seq-{seq:02}_frame-{idx:06}.txt"));
    let text = std::fs::read_to_string(path).ok()?;
    let v: Vec<f32> = text
        .split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect();
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// A frame's 2D keypoints paired with their per-keypoint descriptors.
type Features = (Vec<Point2<f64>>, Vec<Vec<f32>>);

/// One keyframe's retrieval signature plus its own internally-consistent landmark
/// cloud (world points + descriptors). Per-keyframe PnP matches the query against
/// ONE keyframe at a time, so each keyframe's depth-lifted points stay a single
/// consistent 3D set (mixing keyframes' independent depth registration is what
/// caps accuracy — see `relocalization_7scenes_demo`).
struct Keyframe {
    global: Vec<f32>,
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
            "--sessions" => args.sessions = parse_seqs(&val()?),
            "--test-seqs" => args.test_seqs = parse_seqs(&val()?),
            "--session-stride" => args.session_stride = val()?.parse()?,
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
            "--global-descriptor-dir" => args.global_descriptor_dir = Some(PathBuf::from(val()?)),
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

fn frame_base(dataset: &Path, seq: u32, idx: usize) -> PathBuf {
    dataset
        .join(format!("seq-{seq:02}"))
        .join(format!("frame-{idx:06}"))
}

/// Keypoints + descriptors for a frame. With `--sp-features-dir` set, loads
/// pre-exported SuperPoint features; otherwise classical corners. `None` if the
/// frame's source file is absent.
fn frame_features(
    args: &Args,
    extractor: &CornerFeatureExtractor,
    seq: u32,
    idx: usize,
) -> Result<Option<Features>, Box<dyn Error>> {
    if let Some(dir) = &args.sp_features_dir {
        let path = dir.join(format!("seq-{seq:02}_frame-{idx:06}.txt"));
        if !path.exists() {
            return Ok(None);
        }
        let set = read_external_deep_features_txt(&path)?;
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

/// Depth-lift a frame's keypoints to world points using a known camera-to-world
/// pose (R_cw, t_cw): back-project each pixel through its registered depth, then
/// transform to the world frame. Drops keypoints with invalid/out-of-range depth.
/// Returns the surviving (world-point, descriptor) pairs.
#[allow(clippy::too_many_arguments)]
fn lift_keyframe(
    keypoints: &[Point2<f64>],
    descriptors: &[Vec<f32>],
    depth: &image::ImageBuffer<image::Luma<u16>, Vec<u16>>,
    r_cw: &Matrix3<f64>,
    t_cw: &Vector3<f64>,
    args: &Args,
) -> (Vec<Point3<f64>>, Vec<Vec<f32>>) {
    let mut positions = Vec::new();
    let mut descs = Vec::new();
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
        let p_cam = Point3::new(
            (kp.x - args.cx) / args.fx * d,
            (kp.y - args.cy) / args.fy * d,
            d,
        );
        positions.push(Point3::from(r_cw * p_cam.coords + t_cw));
        descs.push(desc.clone());
    }
    (positions, descs)
}

/// Relocalize one query against the lifelong map via learned-retrieval top-K then
/// per-keyframe PnP, keeping the pose with the most inliers. This is the SINGLE
/// localization primitive used both to merge later-session keyframes and to score
/// test frames, so the merge and the evaluation are measured on identical terms.
#[allow(clippy::too_many_arguments)]
fn relocalize(
    map: &[Keyframe],
    keypoints: &[Point2<f64>],
    descriptors: &[Vec<f32>],
    query_global: &[f32],
    estimator: &PnPRansac,
    matcher: &BruteForceMatcher,
    camera: &Camera,
    retrieve_topk: usize,
    min_inliers: usize,
) -> Option<(Pose, usize)> {
    if map.is_empty() {
        return None;
    }
    let mut scored: Vec<(f32, usize)> = map
        .iter()
        .enumerate()
        .map(|(i, kf)| (dot(query_global, &kf.global), i))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    let mut best: Option<(Pose, usize)> = None;
    for &(_, i) in scored.iter().take(retrieve_topk) {
        let kf = &map[i];
        let corrs: Vec<Correspondence2D3D> = matcher
            .match_descriptors(descriptors, &kf.descriptors)
            .iter()
            .filter_map(|m| {
                match (
                    keypoints.get(m.query_index),
                    kf.positions.get(m.train_index),
                ) {
                    (Some(&point2d), Some(&point3d)) => Some(Correspondence2D3D {
                        point2d,
                        point3d,
                        confidence: m.confidence,
                    }),
                    _ => None,
                }
            })
            .collect();
        if let Some(r) = estimator.estimate(&corrs, camera) {
            if r.inliers.len() >= min_inliers
                && best.as_ref().is_none_or(|(_, n)| r.inliers.len() > *n)
            {
                best = Some((r.pose, r.inliers.len()));
            }
        }
    }
    best
}

/// Camera center + world-frame error of an estimated pose against GT (R_cw, t_cw).
fn pose_error(pose: &Pose, r_cw_gt: &Matrix3<f64>, t_cw_gt: &Vector3<f64>) -> (f64, f64) {
    let trans_err = (pose.camera_center_world() - Point3::from(*t_cw_gt)).norm();
    let gt_wc = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        r_cw_gt.transpose(),
    ));
    let rot_err = pose.world_to_camera.rotation.angle_to(&gt_wc).to_degrees();
    (trans_err, rot_err)
}

/// Evaluate the test sessions against the current map and print the metrics.
fn evaluate(label: &str, map: &[Keyframe], args: &Args, extractor: &CornerFeatureExtractor) {
    let jobs: Vec<(u32, usize)> = args
        .test_seqs
        .iter()
        .flat_map(|&seq| {
            (0..args.frames_per_seq)
                .step_by(args.test_stride.max(1))
                .map(move |idx| (seq, idx))
        })
        .collect();

    let outcomes: Vec<Option<(bool, f64, f64)>> = jobs
        .par_iter()
        .map(|&(seq, idx)| {
            let base = frame_base(&args.dataset, seq, idx);
            let (r_cw_gt, t_cw_gt) = read_pose(&base.with_extension("pose.txt")).ok()?;
            let (keypoints, descriptors) = frame_features(args, extractor, seq, idx).ok()??;
            let qg = frame_global(args.global_descriptor_dir.as_ref(), seq, idx)
                .unwrap_or_else(|| normalized_mean(&descriptors));
            let estimator = PnPRansac {
                reprojection_threshold: args.reproj,
                iterations: 256,
                ..PnPRansac::default()
            };
            let matcher = BruteForceMatcher {
                ratio: Some(args.ratio),
            };
            match relocalize(
                map,
                &keypoints,
                &descriptors,
                &qg,
                &estimator,
                &matcher,
                &Camera::pinhole(
                    0,
                    args.width,
                    args.height,
                    args.fx,
                    args.fy,
                    args.cx,
                    args.cy,
                ),
                args.retrieve_topk,
                args.min_inliers,
            ) {
                Some((pose, _)) => {
                    let (te, re) = pose_error(&pose, &r_cw_gt, &t_cw_gt);
                    Some((true, te, re))
                }
                None => Some((false, 0.0, 0.0)),
            }
        })
        .collect();

    let mut total = 0usize;
    let mut localized = 0usize;
    let mut within_5_5 = 0usize;
    let mut trans_errors: Vec<f64> = Vec::new();
    let mut rot_errors: Vec<f64> = Vec::new();
    for outcome in outcomes.into_iter().flatten() {
        total += 1;
        let (ok, te, re) = outcome;
        if !ok {
            continue;
        }
        localized += 1;
        if te <= 0.05 && re <= 5.0 {
            within_5_5 += 1;
        }
        trans_errors.push(te);
        rot_errors.push(re);
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    println!(
        "--- test eval [{label}] : {} keyframes in map ---",
        map.len()
    );
    println!(
        "  test frames: {total}  localized: {localized} ({:.1}%)",
        100.0 * localized as f64 / total.max(1) as f64
    );
    println!(
        "  median trans: {:.3} m  median rot: {:.2} deg  within 5cm/5deg: {within_5_5} ({:.1}%)",
        median(&mut trans_errors),
        median(&mut rot_errors),
        100.0 * within_5_5 as f64 / total.max(1) as f64
    );
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
    let estimator = PnPRansac {
        reprojection_threshold: args.reproj,
        iterations: 256,
        ..PnPRansac::default()
    };
    let matcher = BruteForceMatcher {
        ratio: Some(args.ratio),
    };
    let retrieval = if args.global_descriptor_dir.is_some() {
        "learned EigenPlaces"
    } else {
        "bag-of-features (normalized_mean)"
    };
    let feature_mode = match &args.sp_features_dir {
        Some(dir) => format!("superpoint ({})", dir.display()),
        None => "classical-corner".to_string(),
    };
    println!(
        "dataset={} sessions={:?} test_seqs={:?} session_stride={} test_stride={}",
        args.dataset.display(),
        args.sessions,
        args.test_seqs,
        args.session_stride,
        args.test_stride,
    );
    println!(
        "retrieval={retrieval}  features={feature_mode}  retrieve_topk={} min_inliers={}",
        args.retrieve_topk, args.min_inliers
    );

    let Some((&bootstrap_seq, later_sessions)) = args.sessions.split_first() else {
        return Err("need at least one session in --sessions".into());
    };

    // ---- 1. Bootstrap the map from the first session using GT poses. ----
    let mut map: Vec<Keyframe> = Vec::new();
    for idx in (0..args.frames_per_seq).step_by(args.session_stride.max(1)) {
        let base = frame_base(&args.dataset, bootstrap_seq, idx);
        let Some((keypoints, descriptors)) = frame_features(&args, &extractor, bootstrap_seq, idx)?
        else {
            continue;
        };
        let depth_path = base.with_extension("depth.png");
        if !depth_path.exists() {
            continue;
        }
        let depth = image::open(&depth_path)?.into_luma16();
        let (r_cw, t_cw) = read_pose(&base.with_extension("pose.txt"))?;
        let (positions, descs) =
            lift_keyframe(&keypoints, &descriptors, &depth, &r_cw, &t_cw, &args);
        if positions.is_empty() {
            continue;
        }
        let global = frame_global(args.global_descriptor_dir.as_ref(), bootstrap_seq, idx)
            .unwrap_or_else(|| normalized_mean(&descs));
        map.push(Keyframe {
            global,
            positions,
            descriptors: descs,
        });
    }
    println!(
        "\nbootstrap session seq-{bootstrap_seq:02} (GT poses): {} keyframes",
        map.len()
    );
    let bootstrap_size = map.len();
    evaluate("bootstrap only", &map, &args, &extractor);

    // ---- 2. Integrate each later session by relocalization (no GT used). ----
    for &seq in later_sessions {
        let mut attempted = 0usize;
        let mut merged = 0usize;
        let mut merge_trans: Vec<f64> = Vec::new();
        let mut merge_rot: Vec<f64> = Vec::new();
        for idx in (0..args.frames_per_seq).step_by(args.session_stride.max(1)) {
            let base = frame_base(&args.dataset, seq, idx);
            let Some((keypoints, descriptors)) = frame_features(&args, &extractor, seq, idx)?
            else {
                continue;
            };
            let depth_path = base.with_extension("depth.png");
            if !depth_path.exists() {
                continue;
            }
            attempted += 1;
            let qg = frame_global(args.global_descriptor_dir.as_ref(), seq, idx)
                .unwrap_or_else(|| normalized_mean(&descriptors));
            let Some((pose, _inliers)) = relocalize(
                &map,
                &keypoints,
                &descriptors,
                &qg,
                &estimator,
                &matcher,
                &camera,
                args.retrieve_topk,
                args.min_inliers,
            ) else {
                continue;
            };
            // Record how accurate this relocalization is (GT used only to score).
            if let Ok((r_cw_gt, t_cw_gt)) = read_pose(&base.with_extension("pose.txt")) {
                let (te, re) = pose_error(&pose, &r_cw_gt, &t_cw_gt);
                merge_trans.push(te);
                merge_rot.push(re);
            }
            // Lift this keyframe through the ESTIMATED pose and grow the map.
            let r_cw = pose
                .world_to_camera
                .rotation
                .inverse()
                .to_rotation_matrix()
                .into_inner();
            let t_cw = pose.camera_center_world().coords;
            let depth = image::open(&depth_path)?.into_luma16();
            let (positions, descs) =
                lift_keyframe(&keypoints, &descriptors, &depth, &r_cw, &t_cw, &args);
            if positions.is_empty() {
                continue;
            }
            map.push(Keyframe {
                global: qg,
                positions,
                descriptors: descs,
            });
            merged += 1;
        }
        let median = |v: &mut Vec<f64>| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        println!(
            "\nintegrate session seq-{seq:02}: merged {merged}/{attempted} keyframes ({:.1}%); \
             merge reloc median {:.3} m / {:.2} deg; map now {} keyframes",
            100.0 * merged as f64 / attempted.max(1) as f64,
            median(&mut merge_trans),
            median(&mut merge_rot),
            map.len()
        );
    }

    // ---- 3. Evaluate the held-out test sessions against the full lifelong map. ----
    println!(
        "\nlifelong map grew {bootstrap_size} -> {} keyframes across {} sessions",
        map.len(),
        args.sessions.len()
    );
    evaluate("full lifelong map", &map, &args, &extractor);

    Ok(())
}
