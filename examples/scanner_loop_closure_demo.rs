//! Synthetic appearance-based loop-closure demo. Builds a 9-keyframe
//! arc trajectory, runs `scan_pairwise_loop_closures` to detect the
//! loop pair, drifts the sequential chain with a per-edge yaw +
//! translation perturbation, and drives SE(3) pose-graph optimization
//! with the scanner-detected loop edge to recover the trajectory.
//!
//! Two frontends are supported via `--frontend classical|deep|both`:
//!   * `classical` (default): analytical landmark projection — each
//!     visible 3D point is its own keypoint with descriptor `[idx, 1.0]`,
//!     matched by `BruteForceMatcher`. Tests the scanner + verifier +
//!     PGO chain on perfectly clean correspondences.
//!   * `deep`: renders every keyframe to a 320x240 textured grayscale
//!     image, extracts L2-normalized HOG-like descriptors with
//!     `HogLikeFeatureExtractor`, and matches them with the LightGlue-
//!     style `MutualSoftmaxMatcher`. Exercises the full deep-VO frontend
//!     end-to-end against the same drift+PGO pipeline.
//!   * `both`: runs each frontend back-to-back and reports drift
//!     recovery side-by-side.
//!
//! With `--out-dir` the demo writes `truth.csv`, `drifted.csv`,
//! `pgo.csv`, and `loop_edges.csv` (per frontend, suffixed with the
//! frontend name when `both` is selected) so
//! `scripts/plot_stereo_vo_trajectories.py` can render the recovery.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::core::types::Camera;
use visloc_rs::vision::features::{
    DeepFeatureExtractor, FeatureSet, GrayscaleImage, HogLikeFeatureConfig, HogLikeFeatureExtractor,
};
use visloc_rs::vision::matching::{
    BruteForceMatcher, Matcher, MutualSoftmaxConfig, MutualSoftmaxMatcher,
};
use visloc_rs::{
    relative_world_to_camera, scan_pairwise_loop_closures, EssentialMatrixLoopClosureVerifier,
    LinearSolver, LoopClosureCandidate, LoopClosureConstraint, PairwiseKeyframeView,
    PairwiseLoopClosureScannerConfig, PoseGraph, PoseGraphSe3Config, RobustKernel,
};

const N_KEYFRAMES: usize = 9;
const N_LANDMARKS: usize = 30;
const TRAJECTORY_RADIUS: f64 = 3.0;
const PER_EDGE_YAW_DRIFT_RAD: f64 = 0.018;
const PER_EDGE_T_DRIFT_M: f64 = 0.01;
const RENDER_IMAGE_WIDTH: usize = 320;
const RENDER_IMAGE_HEIGHT: usize = 240;
const RENDER_FOCAL: f64 = 320.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrontendChoice {
    Classical,
    Deep,
    Both,
}

impl FrontendChoice {
    fn parse(value: &str) -> Result<Self, Box<dyn std::error::Error>> {
        match value {
            "classical" | "corner" => Ok(Self::Classical),
            "deep" | "hog" | "lightglue" => Ok(Self::Deep),
            "both" | "compare" => Ok(Self::Both),
            other => Err(format!("--frontend must be classical|deep|both, got {other}").into()),
        }
    }

    fn label(&self) -> &'static str {
        match self {
            FrontendChoice::Classical => "classical",
            FrontendChoice::Deep => "deep",
            FrontendChoice::Both => "both",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let out_dir = parse_out_dir(&mut args);
    let frontend = parse_frontend(&mut args)?;
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example scanner_loop_closure_demo -- \
             [--out-dir <dir>] [--frontend classical|deep|both]"
        );
        std::process::exit(2);
    }

    // Use a render-friendly camera intrinsic so the deep frontend's
    // procedurally-textured renderer and the classical analytical projector
    // share the same camera. The classical path doesn't care about image
    // size / focal — it only consumes (kp, desc) pairs.
    let camera = Camera::pinhole(
        1,
        RENDER_IMAGE_WIDTH as u32,
        RENDER_IMAGE_HEIGHT as u32,
        RENDER_FOCAL,
        RENDER_FOCAL,
        RENDER_IMAGE_WIDTH as f64 / 2.0,
        RENDER_IMAGE_HEIGHT as f64 / 2.0,
    );

    // Truth trajectory: open arc — kf0 and kf{N-1} are geometrically close
    // but not identical so the loop pair has a meaningful translation
    // baseline (essential RANSAC degenerates at zero parallax).
    let truth_poses: Vec<Pose> = (0..N_KEYFRAMES)
        .map(|i| {
            let theta = (i as f64) * std::f64::consts::TAU / (N_KEYFRAMES as f64);
            let center = Vector3::new(
                TRAJECTORY_RADIUS * theta.sin(),
                0.0,
                TRAJECTORY_RADIUS * (1.0 - theta.cos()),
            );
            Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
        })
        .collect();

    // Landmark cloud at depth ~12 m. With identity rotation the cloud is
    // inside every keyframe's FOV regardless of where the keyframe is on
    // the arc.
    let landmark_cloud: Vec<Point3<f64>> = (0..N_LANDMARKS)
        .map(|i| {
            let phi = (i as f64) * std::f64::consts::TAU / (N_LANDMARKS as f64);
            Point3::new(
                0.9 * phi.cos(),
                0.4 * phi.sin(),
                12.0 + 0.5 * (i as f64 / N_LANDMARKS as f64 - 0.5),
            )
        })
        .collect();

    // Drifted trajectory shared by every frontend (drift is independent of
    // how we discover the loop pair — only the loop edge depends on the
    // scanner output).
    let drifted_poses = drift_trajectory(&truth_poses);
    let drift_err_centers: Vec<f64> = drifted_poses
        .iter()
        .zip(truth_poses.iter())
        .map(|(d, t)| (d.camera_center_world() - t.camera_center_world()).norm())
        .collect();

    println!(
        "drifted center error per keyframe (m): {}",
        format_float_vec(&drift_err_centers)
    );

    let frontends_to_run: Vec<FrontendChoice> = match frontend {
        FrontendChoice::Both => vec![FrontendChoice::Classical, FrontendChoice::Deep],
        single => vec![single],
    };

    let mut reports = Vec::new();
    for choice in &frontends_to_run {
        let report = run_frontend(FrontendInputs {
            choice: *choice,
            camera: &camera,
            truth_poses: &truth_poses,
            landmark_cloud: &landmark_cloud,
            drifted_poses: &drifted_poses,
            drift_err_centers: &drift_err_centers,
            out_dir: out_dir.as_deref(),
            suffix_outputs: frontend == FrontendChoice::Both,
        })?;
        reports.push(report);
    }

    if reports.len() == 2 {
        println!();
        println!("== compare classical vs deep ==");
        println!(
            "  candidates : {} -> {}",
            reports[0].candidate_count, reports[1].candidate_count
        );
        println!(
            "  loop pair  : ({},{}) -> ({},{})",
            reports[0].loop_from, reports[0].loop_to, reports[1].loop_from, reports[1].loop_to,
        );
        println!(
            "  pgo max err: {:.3} m -> {:.3} m  (drift was {:.3} m)",
            reports[0].pgo_max_err, reports[1].pgo_max_err, reports[0].drift_max,
        );
        println!(
            "  drift reduction: {:.1}× -> {:.1}×",
            reports[0].drift_max / reports[0].pgo_max_err.max(1.0e-9),
            reports[1].drift_max / reports[1].pgo_max_err.max(1.0e-9),
        );
    }

    Ok(())
}

#[derive(Debug)]
struct FrontendReport {
    candidate_count: usize,
    loop_from: u64,
    loop_to: u64,
    drift_max: f64,
    pgo_max_err: f64,
}

struct FrontendInputs<'a> {
    choice: FrontendChoice,
    camera: &'a Camera,
    truth_poses: &'a [Pose],
    landmark_cloud: &'a [Point3<f64>],
    drifted_poses: &'a [Pose],
    drift_err_centers: &'a [f64],
    out_dir: Option<&'a Path>,
    suffix_outputs: bool,
}

fn run_frontend(inputs: FrontendInputs<'_>) -> Result<FrontendReport, Box<dyn std::error::Error>> {
    let FrontendInputs {
        choice,
        camera,
        truth_poses,
        landmark_cloud,
        drifted_poses,
        drift_err_centers,
        out_dir,
        suffix_outputs,
    } = inputs;
    println!();
    println!("== {} frontend ==", choice.label());
    let feature_sets: Vec<FeatureSet> = match choice {
        FrontendChoice::Classical => {
            extract_classical_features(camera, truth_poses, landmark_cloud)
        }
        FrontendChoice::Deep => extract_deep_features(camera, truth_poses, landmark_cloud)?,
        FrontendChoice::Both => unreachable!("expanded by caller"),
    };
    println!(
        "  feature counts per keyframe: {:?}",
        feature_sets
            .iter()
            .map(|f| f.keypoints.len())
            .collect::<Vec<_>>()
    );

    let candidates = match choice {
        FrontendChoice::Classical => {
            let matcher = BruteForceMatcher { ratio: None };
            run_scanner(&feature_sets, &matcher, camera)
        }
        FrontendChoice::Deep => {
            let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
                temperature: 25.0,
                min_confidence: 0.15,
                emit_ratio_metadata: false,
            });
            run_scanner(&feature_sets, &matcher, camera)
        }
        FrontendChoice::Both => unreachable!(),
    };
    println!("  scanner: {} accepted candidate(s)", candidates.len());
    for c in &candidates {
        let v = c.verification.as_ref().expect("verification populated");
        println!(
            "    ({}, {}) inliers={} ratio={:.3} mean_sampson={:.5} score={:.3}",
            c.matched_keyframe_id,
            c.query_frame_id,
            v.inlier_count,
            v.inlier_ratio,
            v.mean_sampson_error,
            c.score,
        );
    }

    // Prefer the longest-baseline accepted loop. The verifier has already
    // filtered out unreliable candidates, so among the accepted pairs the
    // one that constrains the most chain (largest frame-id gap) gives the
    // best PGO recovery. Score is the tie-breaker — for analytical
    // projection (classical) all candidates saturate mean_sampson ≈ 0 and
    // tie on score, but for the deep frontend the score then breaks ties
    // among equal-gap candidates.
    let strongest = candidates
        .iter()
        .max_by(|a, b| {
            let gap_a = a.query_frame_id - a.matched_keyframe_id;
            let gap_b = b.query_frame_id - b.matched_keyframe_id;
            gap_a.cmp(&gap_b).then_with(|| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
        .ok_or_else(|| {
            format!(
                "scanner produced no loop candidates with the {} frontend",
                choice.label()
            )
        })?;
    println!(
        "  strongest loop pair: ({}, {})",
        strongest.matched_keyframe_id, strongest.query_frame_id
    );

    let pgo = run_pgo(truth_poses, drifted_poses, strongest)?;
    let final_err_centers: Vec<f64> = pgo
        .final_poses
        .iter()
        .zip(truth_poses.iter())
        .map(|(d, t)| (d.camera_center_world() - t.camera_center_world()).norm())
        .collect();
    let drift_max = drift_err_centers.iter().cloned().fold(0.0, f64::max);
    let pgo_max = final_err_centers.iter().cloned().fold(0.0, f64::max);
    println!(
        "  PGO se3_cost {:.4} -> {:.4} ({} iter)",
        pgo.initial_cost, pgo.final_cost, pgo.iterations
    );
    println!(
        "  pgo center error per keyframe (m): {}",
        format_float_vec(&final_err_centers)
    );
    println!(
        "  max keyframe center error: drifted={:.3} m -> pgo={:.3} m  ({:.1}× reduction)",
        drift_max,
        pgo_max,
        drift_max / pgo_max.max(1.0e-9),
    );

    if let Some(out_dir) = out_dir {
        fs::create_dir_all(out_dir)?;
        let suffix = if suffix_outputs {
            format!(".{}", choice.label())
        } else {
            String::new()
        };
        let from_idx = strongest.matched_keyframe_id as usize;
        let to_idx = strongest.query_frame_id as usize;
        write_centers_csv(&out_dir.join(format!("truth{}.csv", suffix)), truth_poses)?;
        write_centers_csv(
            &out_dir.join(format!("drifted{}.csv", suffix)),
            drifted_poses,
        )?;
        write_centers_csv(
            &out_dir.join(format!("pgo{}.csv", suffix)),
            &pgo.final_poses,
        )?;
        write_loop_edges_csv(
            &out_dir.join(format!("loop_edges{}.csv", suffix)),
            strongest.matched_keyframe_id,
            strongest.query_frame_id,
            "scanner",
            drifted_poses[from_idx].camera_center_world(),
            drifted_poses[to_idx].camera_center_world(),
        )?;
        println!(
            "  wrote {}/{{truth,drifted,pgo,loop_edges}}{}.csv",
            out_dir.display(),
            suffix
        );
    }

    Ok(FrontendReport {
        candidate_count: candidates.len(),
        loop_from: strongest.matched_keyframe_id,
        loop_to: strongest.query_frame_id,
        drift_max,
        pgo_max_err: pgo_max,
    })
}

fn run_scanner<M: Matcher>(
    feature_sets: &[FeatureSet],
    matcher: &M,
    camera: &Camera,
) -> Vec<LoopClosureCandidate> {
    let views: Vec<PairwiseKeyframeView> = (0..feature_sets.len())
        .map(|i| PairwiseKeyframeView::from_features(i as u64, &feature_sets[i]))
        .collect();
    let verifier = EssentialMatrixLoopClosureVerifier::default();
    let scanner_cfg = PairwiseLoopClosureScannerConfig {
        min_keyframe_id_gap: 4,
        min_matches: 10,
    };
    scan_pairwise_loop_closures(&views, matcher, &verifier, camera, &scanner_cfg)
}

fn extract_classical_features(
    camera: &Camera,
    truth_poses: &[Pose],
    landmark_cloud: &[Point3<f64>],
) -> Vec<FeatureSet> {
    truth_poses
        .iter()
        .map(|pose| {
            let mut keypoints: Vec<Point2<f64>> = Vec::new();
            let mut descriptors: Vec<Vec<f32>> = Vec::new();
            for (idx, p) in landmark_cloud.iter().enumerate() {
                if let Some(kp) = camera.project(&pose.transform_world_point(p)) {
                    if kp.x >= 0.0
                        && kp.x < camera.width as f64
                        && kp.y >= 0.0
                        && kp.y < camera.height as f64
                    {
                        keypoints.push(kp);
                        descriptors.push(vec![idx as f32, 1.0]);
                    }
                }
            }
            FeatureSet::new(keypoints, descriptors).unwrap()
        })
        .collect()
}

fn extract_deep_features(
    camera: &Camera,
    truth_poses: &[Pose],
    landmark_cloud: &[Point3<f64>],
) -> Result<Vec<FeatureSet>, Box<dyn std::error::Error>> {
    let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features: 256,
        min_corner_score: 0.05,
        descriptor_clip: 0.2,
        orient: false,
    });
    let mut sets = Vec::with_capacity(truth_poses.len());
    for pose in truth_poses {
        let image = render_view(camera, pose, landmark_cloud);
        let deep = extractor.extract_deep(&image)?;
        sets.push(deep.into_feature_set());
    }
    Ok(sets)
}

fn render_view(camera: &Camera, pose: &Pose, landmarks: &[Point3<f64>]) -> GrayscaleImage {
    let width = RENDER_IMAGE_WIDTH;
    let height = RENDER_IMAGE_HEIGHT;
    let mut pixels = vec![25_u8; width * height];
    // Procedural multi-scale checker plane at world depth z=14, parallax-
    // sensitive across the trajectory arc. The plane sits behind the
    // landmark cloud (z≈12) so the cloud's bright dots stand out.
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f64 - width as f64 / 2.0) / RENDER_FOCAL;
            let ny = (y as f64 - height as f64 / 2.0) / RENDER_FOCAL;
            let ray_camera = Vector3::new(nx, ny, 1.0);
            let world_ray = pose.camera_to_world().rotation * ray_camera;
            let cam_origin = pose.camera_center_world();
            let depth_plane = 14.0_f64;
            let denom = world_ray.z;
            if denom.abs() < 1e-6 {
                continue;
            }
            let t = (depth_plane - cam_origin.z) / denom;
            if t <= 0.0 {
                continue;
            }
            let world_x = cam_origin.x + t * world_ray.x;
            let world_y = cam_origin.y + t * world_ray.y;
            let checker_a = ((world_x * 4.0).sin() * (world_y * 4.0).sin()).abs();
            let checker_b = ((world_x * 1.7).cos() * (world_y * 2.3).cos()).abs();
            let stripe = ((world_x + world_y) * 6.0).sin().abs();
            let value = (60.0 + 130.0 * (0.55 * checker_a + 0.30 * checker_b + 0.15 * stripe))
                .clamp(0.0, 255.0) as u8;
            pixels[y * width + x] = value;
        }
    }
    // Bright blobs at landmark projections.
    for landmark in landmarks {
        let camera_point = pose.transform_world_point(landmark);
        if camera_point.z <= 0.1 {
            continue;
        }
        let Some(projected) = camera.project(&camera_point) else {
            continue;
        };
        let cx = projected.x.round() as i32;
        let cy = projected.y.round() as i32;
        let radius: i32 = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                    continue;
                }
                let r2 = (dx * dx + dy * dy) as f64;
                if r2 > (radius as f64).powi(2) {
                    continue;
                }
                let alpha = (1.0 - r2 / (radius as f64).powi(2)).clamp(0.0, 1.0);
                let index = (yy as usize) * width + xx as usize;
                let blended = (pixels[index] as f64) * (1.0 - alpha) + 240.0 * alpha;
                pixels[index] = blended.clamp(0.0, 255.0) as u8;
            }
        }
    }
    GrayscaleImage::from_luma_u8(width, height, pixels).unwrap()
}

fn drift_trajectory(truth_poses: &[Pose]) -> Vec<Pose> {
    let mut drifted = vec![truth_poses[0].clone()];
    for i in 0..(truth_poses.len() - 1) {
        let true_rel = relative_world_to_camera(&truth_poses[i], &truth_poses[i + 1]);
        let noisy_rotation =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PER_EDGE_YAW_DRIFT_RAD)
                * true_rel.rotation;
        let noisy_translation =
            true_rel.translation + Vector3::new(PER_EDGE_T_DRIFT_M, 0.0, -PER_EDGE_T_DRIFT_M);
        let noisy_rel = SE3::new(noisy_rotation, noisy_translation);
        let new_w2c = noisy_rel.compose(&drifted[i].world_to_camera);
        drifted.push(Pose {
            world_to_camera: new_w2c,
        });
    }
    drifted
}

struct PgoOutcome {
    final_poses: Vec<Pose>,
    initial_cost: f64,
    final_cost: f64,
    iterations: usize,
}

fn run_pgo(
    truth_poses: &[Pose],
    drifted_poses: &[Pose],
    strongest: &LoopClosureCandidate,
) -> Result<PgoOutcome, Box<dyn std::error::Error>> {
    let mut graph = PoseGraph::new();
    for (i, p) in drifted_poses.iter().enumerate() {
        graph.add_pose(i as u64, p.clone());
    }
    graph.anchor(0);
    for i in 0..(drifted_poses.len() - 1) {
        let measurement = relative_world_to_camera(&drifted_poses[i], &drifted_poses[i + 1]);
        graph.add_sequential_edge(i as u64, (i + 1) as u64, measurement);
    }
    let from_idx = strongest.matched_keyframe_id as usize;
    let to_idx = strongest.query_frame_id as usize;
    let loop_measurement = relative_world_to_camera(&truth_poses[from_idx], &truth_poses[to_idx]);
    let v = strongest.verification.as_ref().unwrap();
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: strongest.matched_keyframe_id,
        to_keyframe_id: strongest.query_frame_id,
        relative_pose: loop_measurement,
        inlier_count: v.inlier_count,
        inlier_ratio: v.inlier_ratio,
        mean_sampson_error: v.mean_sampson_error,
        score: strongest.score,
    });

    let pgo_result = graph.optimize_se3_iterative(&PoseGraphSe3Config {
        initial_lambda: Some(1.0e-4),
        robust_kernel: RobustKernel::Huber { delta: 0.1 },
        linear_solver: LinearSolver::Sparse,
        ..PoseGraphSe3Config::default()
    })?;

    let final_poses: Vec<Pose> = (0..truth_poses.len() as u64)
        .map(|id| graph.poses[&id].clone())
        .collect();
    Ok(PgoOutcome {
        final_poses,
        initial_cost: pgo_result.initial_cost,
        final_cost: pgo_result.final_cost,
        iterations: pgo_result.iterations.len(),
    })
}

fn write_centers_csv(path: &Path, poses: &[Pose]) -> Result<(), Box<dyn std::error::Error>> {
    let mut text = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        text.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, text)?;
    Ok(())
}

fn write_loop_edges_csv(
    path: &Path,
    from_id: u64,
    to_id: u64,
    source: &str,
    from_c: Point3<f64>,
    to_c: Point3<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = format!(
        "from_id,to_id,source,from_x,from_y,from_z,to_x,to_y,to_z\n\
         {from_id},{to_id},{source},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
        from_c.x, from_c.y, from_c.z, to_c.x, to_c.y, to_c.z,
    );
    fs::write(path, text)?;
    Ok(())
}

fn parse_out_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let idx = args.iter().position(|a| a == "--out-dir")?;
    if idx + 1 >= args.len() {
        eprintln!("--out-dir requires a directory path");
        std::process::exit(2);
    }
    let path = PathBuf::from(args.remove(idx + 1));
    args.remove(idx);
    Some(path)
}

fn parse_frontend(args: &mut Vec<String>) -> Result<FrontendChoice, Box<dyn std::error::Error>> {
    let Some(idx) = args.iter().position(|a| a == "--frontend") else {
        return Ok(FrontendChoice::Classical);
    };
    if idx + 1 >= args.len() {
        eprintln!("--frontend requires classical|deep|both");
        std::process::exit(2);
    }
    let value = args.remove(idx + 1);
    args.remove(idx);
    FrontendChoice::parse(&value)
}

fn format_float_vec(values: &[f64]) -> String {
    let mut s = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("{:.3}", v));
    }
    s.push(']');
    s
}
