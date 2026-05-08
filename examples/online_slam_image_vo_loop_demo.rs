//! End-to-end real-image visual odometry + loop closure demo.
//!
//! Loads a KITTI-format grayscale image sequence (e.g., `<KITTI>/sequences/00/
//! image_0`) plus its calibration (`calib.txt`), runs feature extraction +
//! matching + essential-matrix RANSAC between consecutive frames to integrate
//! a monocular VO trajectory, attempts a single loop closure between the last
//! frame and the first frame via the same essential-matrix pipeline, and
//! finally runs `PoseGraph::optimize_se3_iterative` (Levenberg-Marquardt +
//! Cholesky) on the resulting graph.
//!
//! All inputs are read from disk — keypoints, descriptors, and relative poses
//! are derived from real pixel data, not synthesized. Monocular essential-
//! matrix VO is scale-ambiguous; this demo uses a fixed unit translation
//! scale per pair, so the resulting trajectory is in arbitrary units. The
//! loop closure constraint, computed from the same essential-matrix path
//! between the start and end frames, is what pulls the chain back together.
//!
//! Usage:
//!
//! ```sh
//! cargo run --release --features image-io \
//!     --example online_slam_image_vo_loop_demo -- \
//!     --image-dir /path/to/KITTI_odometry/sequences/00/image_0 \
//!     --calib    /path/to/KITTI_odometry/sequences/00/calib.txt \
//!     --max-frames 200 \
//!     --frame-stride 4 \
//!     --out-dir target/kitti_image_vo_loop_demo
//! ```
//!
//! The example writes `vo.csv` and `corrected.csv` (id, x, y, z) to the
//! output directory; `scripts/build_kitti_loop_asset.py` can render those
//! into the README asset.

#[cfg(not(feature = "image-io"))]
fn main() {
    eprintln!(
        "this example requires the `image-io` feature; rebuild with \
         `cargo run --release --features image-io --example online_slam_image_vo_loop_demo`"
    );
    std::process::exit(2);
}

#[cfg(feature = "image-io")]
use std::env;
#[cfg(feature = "image-io")]
use std::fs;
#[cfg(feature = "image-io")]
use std::path::{Path, PathBuf};

#[cfg(feature = "image-io")]
use nalgebra::Point2;
#[cfg(feature = "image-io")]
use visloc_rs::core::geometry::{Pose, SE3};
#[cfg(feature = "image-io")]
use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::{
    CornerFeatureConfig, CornerFeatureExtractor, FeatureExtractor, FeatureSet, GrayscaleImage,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::matching::{BruteForceMatcher, CrossCheckMatcher, Matcher};
#[cfg(feature = "image-io")]
use visloc_rs::vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};
#[cfg(feature = "image-io")]
use visloc_rs::{LoopClosureConstraint, PoseGraph, PoseGraphSe3Config};

#[cfg(feature = "image-io")]
#[derive(Debug)]
struct CliArgs {
    image_dir: PathBuf,
    calib: PathBuf,
    out_dir: PathBuf,
    projection_label: String,
    max_frames: usize,
    frame_stride: usize,
    min_inliers: usize,
}

#[cfg(feature = "image-io")]
fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut image_dir: Option<PathBuf> = None;
    let mut calib: Option<PathBuf> = None;
    let mut out_dir: PathBuf = PathBuf::from("target/kitti_image_vo_loop_demo");
    let mut projection_label = String::from("P0");
    let mut max_frames: usize = 200;
    let mut frame_stride: usize = 4;
    let mut min_inliers: usize = 24;

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image-dir" => {
                image_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--calib" => {
                calib = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--projection-label" => {
                projection_label = args.remove(i + 1);
                args.remove(i);
            }
            "--max-frames" => {
                max_frames = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--frame-stride" => {
                frame_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--min-inliers" => {
                min_inliers = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let image_dir = image_dir.ok_or("--image-dir <path> is required")?;
    let calib = calib.ok_or("--calib <path/to/calib.txt> is required")?;
    Ok(CliArgs {
        image_dir,
        calib,
        out_dir,
        projection_label,
        max_frames,
        frame_stride,
        min_inliers,
    })
}

#[cfg(feature = "image-io")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    println!(
        "image_dir={} calib={} projection={} stride={} max_frames={}",
        args.image_dir.display(),
        args.calib.display(),
        args.projection_label,
        args.frame_stride,
        args.max_frames,
    );

    let sequence =
        read_kitti_image_sequence_dir(&args.image_dir, &args.calib, &args.projection_label, 1)?;
    let camera = sequence.camera.clone();
    let frames: Vec<&GrayscaleImage> = sequence
        .frames
        .iter()
        .step_by(args.frame_stride.max(1))
        .take(args.max_frames)
        .map(|frame| &frame.image)
        .collect();
    let n = frames.len();
    if n < 4 {
        return Err(format!("need at least 4 frames, got {n}").into());
    }
    println!(
        "loaded n_frames={n} (camera {}x{} fx,fy=({:.1},{:.1}))",
        camera.width,
        camera.height,
        camera.params.first().copied().unwrap_or(0.0),
        camera.params.get(1).copied().unwrap_or(0.0),
    );

    // Extract features for every kept frame once. CornerFeatureExtractor uses
    // simple intensity-difference patches as descriptors which are enough for
    // short-baseline matching across a sequence.
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: 600,
        min_score: 0.04,
        descriptor_radius: 3,
    });
    let mut feature_sets: Vec<FeatureSet> = Vec::with_capacity(n);
    for image in &frames {
        let fs = extractor.extract(image)?;
        feature_sets.push(fs);
    }
    let feature_counts: Vec<usize> = feature_sets.iter().map(|f| f.len()).collect();
    println!(
        "feature counts: min={} median={} max={}",
        feature_counts.iter().copied().min().unwrap_or(0),
        median(&feature_counts),
        feature_counts.iter().copied().max().unwrap_or(0),
    );

    let matcher = CrossCheckMatcher::new(BruteForceMatcher { ratio: Some(0.85) });
    let estimator = RelativePoseEstimator::default();

    // Sequential VO: for each consecutive pair, build correspondences from
    // matched descriptors and run essential-matrix RANSAC.
    let mut sequential_edges: Vec<SE3> = Vec::with_capacity(n - 1);
    let mut sequential_inliers: Vec<usize> = Vec::with_capacity(n - 1);
    let mut sequential_correspondences: Vec<usize> = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let correspondences =
            build_correspondences(&feature_sets[i], &feature_sets[i + 1], &matcher);
        sequential_correspondences.push(correspondences.len());
        let edge = estimator
            .estimate(&correspondences, &camera)
            .ok_or_else(|| {
                format!(
                    "essential-matrix VO failed between frame {} and {}: only {} correspondences",
                    i,
                    i + 1,
                    correspondences.len(),
                )
            })?;
        sequential_inliers.push(edge.inliers.len());
        sequential_edges.push(edge.previous_to_current);
    }
    println!(
        "sequential edges: count={} mean_correspondences={:.1} mean_inliers={:.1}",
        sequential_edges.len(),
        mean(&sequential_correspondences),
        mean(&sequential_inliers),
    );

    // Integrate the sequential edges into a VO trajectory. The world frame is
    // anchored at the first camera (T_0 = identity).
    let mut vo_poses: Vec<Pose> = vec![Pose::identity()];
    for edge in &sequential_edges {
        let last = vo_poses.last().unwrap();
        let new_world_to_camera = edge.compose(&last.world_to_camera);
        vo_poses.push(Pose {
            world_to_camera: new_world_to_camera,
        });
    }

    // Loop detection: try matching the final frame against the first one. If
    // we get enough RANSAC inliers, accept the recovered relative pose as a
    // metric-consistent loop edge (same scale convention as the sequential
    // edges, so the chain pulls back to where it started).
    let loop_correspondences =
        build_correspondences(&feature_sets[0], &feature_sets[n - 1], &matcher);
    println!(
        "loop_pair frame 0 ↔ frame {} correspondences={}",
        n - 1,
        loop_correspondences.len(),
    );
    let loop_edge = estimator.estimate(&loop_correspondences, &camera);
    let loop_constraint = match &loop_edge {
        Some(edge) if edge.inliers.len() >= args.min_inliers => {
            println!(
                "loop_pair_verified inliers={} mean_sampson={:.4}",
                edge.inliers.len(),
                edge.mean_sampson_error,
            );
            Some(LoopClosureConstraint {
                from_keyframe_id: 0,
                to_keyframe_id: (n - 1) as u64,
                relative_pose: edge.previous_to_current.clone(),
                inlier_count: edge.inliers.len(),
                inlier_ratio: edge.inliers.len() as f64 / loop_correspondences.len().max(1) as f64,
                mean_sampson_error: edge.mean_sampson_error,
                score: edge.inliers.len() as f64,
            })
        }
        Some(edge) => {
            println!(
                "loop_pair_rejected inliers={} (< min_inliers={}); pose graph runs without loop edge",
                edge.inliers.len(),
                args.min_inliers,
            );
            None
        }
        None => {
            println!("loop_pair_rejected essential RANSAC did not converge");
            None
        }
    };

    let mut graph = PoseGraph::new();
    for (id, pose) in vo_poses.iter().enumerate() {
        graph.add_pose(id as u64, pose.clone());
    }
    graph.anchor(0);
    for (i, edge) in sequential_edges.iter().enumerate() {
        graph.add_sequential_edge(i as u64, (i + 1) as u64, edge.clone());
    }
    if let Some(ref constraint) = loop_constraint {
        graph.add_loop_closure_constraint(constraint);
    }

    println!(
        "pose_graph nodes={} edges={} se3_cost_before={:.4}",
        graph.poses.len(),
        graph.edges.len(),
        graph.se3_cost(),
    );

    let result = if loop_constraint.is_some() {
        Some(graph.optimize_se3_iterative(&PoseGraphSe3Config {
            initial_lambda: Some(1.0e-3),
            max_iterations: 50,
            ..PoseGraphSe3Config::default()
        })?)
    } else {
        None
    };

    if let Some(result) = result.as_ref() {
        println!(
            "optimization initial_cost={:.4} final_cost={:.4} iterations={} converged={}",
            result.initial_cost,
            result.final_cost,
            result.iterations.len(),
            result.converged,
        );
        for stats in result.iterations.iter().take(8) {
            println!(
                "  iter={} cost_before={:.4} cost_after={:.4} max_step={:.4} lambda={:.2e} accepted={}",
                stats.iteration,
                stats.cost_before,
                stats.cost_after,
                stats.max_step_norm,
                stats.lambda,
                stats.step_accepted,
            );
        }
    }

    let corrected: Vec<Pose> = (0..n as u64).map(|id| graph.poses[&id].clone()).collect();
    fs::create_dir_all(&args.out_dir)?;
    write_xyz_csv(&args.out_dir.join("vo.csv"), &vo_poses)?;
    write_xyz_csv(&args.out_dir.join("corrected.csv"), &corrected)?;
    println!("trajectories written to {}", args.out_dir.display());

    let last_vo = vo_poses[n - 1].camera_center_world();
    let last_corrected = corrected[n - 1].camera_center_world();
    let endpoint_distance = (last_corrected - last_vo).norm();
    println!(
        "endpoint vo=[{:.3}, {:.3}, {:.3}] corrected=[{:.3}, {:.3}, {:.3}] vo_to_corrected={:.3}",
        last_vo.x,
        last_vo.y,
        last_vo.z,
        last_corrected.x,
        last_corrected.y,
        last_corrected.z,
        endpoint_distance,
    );

    Ok(())
}

#[cfg(feature = "image-io")]
fn build_correspondences(
    a: &FeatureSet,
    b: &FeatureSet,
    matcher: &CrossCheckMatcher<BruteForceMatcher>,
) -> Vec<TwoViewCorrespondence> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let matches = matcher.match_descriptors(&a.descriptors, &b.descriptors);
    let mut out = Vec::with_capacity(matches.len());
    for m in matches {
        let (Some(prev_xy), Some(curr_xy)) = (
            a.keypoints.get(m.query_index),
            b.keypoints.get(m.train_index),
        ) else {
            continue;
        };
        out.push(TwoViewCorrespondence {
            previous_xy: Point2::new(prev_xy.x, prev_xy.y),
            current_xy: Point2::new(curr_xy.x, curr_xy.y),
        });
    }
    out
}

#[cfg(feature = "image-io")]
fn write_xyz_csv(path: &Path, poses: &[Pose]) -> std::io::Result<()> {
    let mut s = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        s.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, s)
}

#[cfg(feature = "image-io")]
fn mean(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

#[cfg(feature = "image-io")]
fn median(values: &[usize]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort();
    sorted[sorted.len() / 2]
}
