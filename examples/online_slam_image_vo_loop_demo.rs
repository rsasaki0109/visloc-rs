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
use std::collections::HashMap;
#[cfg(feature = "image-io")]
use visloc_rs::core::geometry::{Pose, SE3};
#[cfg(feature = "image-io")]
use visloc_rs::io::kitti::read_kitti_image_sequence_dir;
#[cfg(feature = "image-io")]
use visloc_rs::vision::features::{
    CornerFeatureConfig, CornerFeatureExtractor, FeatureExtractor, FeatureSet, GrayscaleImage,
};
#[cfg(feature = "image-io")]
use visloc_rs::vision::matching::{BruteForceMatcher, Matcher};
#[cfg(feature = "image-io")]
use visloc_rs::vision::two_view::{RelativePoseEstimator, TwoViewCorrespondence};
#[cfg(feature = "image-io")]
use visloc_rs::{
    BaConfig, BaObservation, BundleAdjustment, LinearSolver, LoopClosureConstraint, PoseGraph,
    PoseGraphSe3Config, RobustKernel,
};

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
    // simple intensity-difference patches as descriptors. For real KITTI
    // imagery we need many more candidate corners and a larger descriptor
    // patch than the synthetic-fixture defaults to keep matching alive across
    // the per-pair vehicle motion.
    let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
        max_features: 1500,
        min_score: 0.02,
        descriptor_radius: 5,
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

    // Drop cross-check on real data — it cuts matches roughly in half, which
    // patch descriptors can't afford. Rely on the essential-matrix RANSAC to
    // reject the extra outliers a pure brute-force matcher leaves in.
    let matcher = BruteForceMatcher { ratio: Some(0.9) };
    let estimator = RelativePoseEstimator::default();

    // Sequential VO: for each consecutive pair, build correspondences from
    // matched descriptors and run essential-matrix RANSAC. Also remember
    // the (kp_a, kp_b) index pairs per consecutive pair so we can link
    // them into multi-frame feature tracks for the BA step below.
    let mut sequential_edges: Vec<SE3> = Vec::with_capacity(n - 1);
    let mut sequential_inliers: Vec<usize> = Vec::with_capacity(n - 1);
    let mut sequential_correspondences: Vec<usize> = Vec::with_capacity(n - 1);
    let mut per_pair_inlier_matches: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let (correspondences, index_pairs) =
            build_correspondences_with_indices(&feature_sets[i], &feature_sets[i + 1], &matcher);
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
        // Keep only RANSAC-inlier matches for tracking — outliers would
        // pollute the tracks and force BA to reject them anyway.
        let inlier_pairs: Vec<(usize, usize)> =
            edge.inliers.iter().map(|&idx| index_pairs[idx]).collect();
        per_pair_inlier_matches.push(inlier_pairs);
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

    // For monocular VO the per-pair translation scale is arbitrary (unit by
    // default). The loop edge expresses the same arbitrary-scale convention,
    // so the chain's translations must be rescaled to satisfy it. The
    // translation-only Gauss-Newton step is exact for that linear residual
    // system, converges in one solve, and (because rotations look reasonable
    // out of the essential-matrix RANSAC) preserves the trajectory shape.
    if loop_constraint.is_some() {
        // Use sparse Cholesky so the optimizer scales to the thousand-keyframe
        // KITTI run. The pose graph is block-banded (≤ 4 dense `3×3` (or
        // `6×6`) blocks per edge), so CSC + `CscCholesky` is much faster and
        // uses orders of magnitude less memory than the dense path.
        let step = graph.optimize_translations_once_with(LinearSolver::Sparse)?;
        println!(
            "translation_pgo cost_before={:.4} cost_after={:.4} mean_correction={:.4} max_correction={:.4}",
            step.cost_before,
            step.cost_after,
            step.mean_translation_correction,
            step.max_translation_correction,
        );
        // Follow up with a few SE(3) GN iterations so rotations are also
        // pulled into agreement now that translations are consistent.
        let result = graph.optimize_se3_iterative(&PoseGraphSe3Config {
            initial_lambda: Some(1.0e-4),
            max_iterations: 20,
            linear_solver: LinearSolver::Sparse,
            ..PoseGraphSe3Config::default()
        })?;
        println!(
            "se3_refine initial_cost={:.4} final_cost={:.4} iterations={} converged={}",
            result.initial_cost,
            result.final_cost,
            result.iterations.len(),
            result.converged,
        );
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

    // -----------------------------------------------------------------
    // Bundle adjustment pass on real-image feature tracks.
    // -----------------------------------------------------------------
    // Link the per-pair RANSAC-inlier matches into multi-frame feature
    // tracks. Each match `(kp_a, kp_b)` between frames `i, i+1` either
    // extends an existing track that ended at `(i, kp_a)` or starts a
    // new track. Tracks observed by ≥ MIN_TRACK_LEN frames are
    // triangulated by DLT using the post-PGO poses and become BA
    // landmarks. Each track's per-frame keypoints become BA observations.
    const MIN_TRACK_LEN: usize = 3;
    let mut tracks: Vec<Vec<(usize, Point2<f64>)>> = Vec::new();
    let mut endpoint_to_track: HashMap<(usize, usize), usize> = HashMap::new();
    for (pair_idx, pair_matches) in per_pair_inlier_matches.iter().enumerate() {
        let frame_a = pair_idx;
        let frame_b = pair_idx + 1;
        for &(kp_a, kp_b) in pair_matches {
            let xy_b = feature_sets[frame_b].keypoints[kp_b];
            let p_b = Point2::new(xy_b.x as f64, xy_b.y as f64);
            if let Some(&track_id) = endpoint_to_track.get(&(frame_a, kp_a)) {
                tracks[track_id].push((frame_b, p_b));
                endpoint_to_track.remove(&(frame_a, kp_a));
                endpoint_to_track.insert((frame_b, kp_b), track_id);
            } else {
                let xy_a = feature_sets[frame_a].keypoints[kp_a];
                let p_a = Point2::new(xy_a.x as f64, xy_a.y as f64);
                let track_id = tracks.len();
                tracks.push(vec![(frame_a, p_a), (frame_b, p_b)]);
                endpoint_to_track.insert((frame_b, kp_b), track_id);
            }
        }
    }
    let total_tracks = tracks.len();
    tracks.retain(|t| t.len() >= MIN_TRACK_LEN);
    let track_lengths: Vec<usize> = tracks.iter().map(|t| t.len()).collect();
    let mean_track_len = if track_lengths.is_empty() {
        0.0
    } else {
        track_lengths.iter().copied().sum::<usize>() as f64 / track_lengths.len() as f64
    };
    let max_track_len = track_lengths.iter().copied().max().unwrap_or(0);
    println!(
        "ba_tracks total={} long={} (≥{} frames) mean_len={:.1} max_len={}",
        total_tracks,
        tracks.len(),
        MIN_TRACK_LEN,
        mean_track_len,
        max_track_len,
    );

    // Triangulate every long track via 2-row DLT and reject those with
    // negative depth or large reprojection residual on any view.
    let mut ba = BundleAdjustment::new(camera.clone());
    for id in 0..(n as u64) {
        ba.add_pose(id, corrected[id as usize].clone());
    }
    // Fix the first keyframe (gauge anchor) plus, when a loop closure was
    // verified, the last keyframe too. Without this, BA has no incentive to
    // honor the loop-closure constraint that lives only in the pose graph
    // and would drift the chain back toward an arbitrary-scale per-pair
    // optimum.
    ba.fix_pose(0);
    if loop_constraint.is_some() {
        ba.fix_pose((n - 1) as u64);
    }
    let mut accepted = 0usize;
    let mut rejected_depth = 0usize;
    let mut rejected_reproj = 0usize;
    for (track_id, observations) in tracks.iter().enumerate() {
        let Some(point) = triangulate_track_dlt(observations, &corrected, &camera) else {
            rejected_depth += 1;
            continue;
        };
        // Sanity filter: reject only landmarks that are visibly broken
        // — every-view depth ≤ 0 (DLT failed, point behind camera) or
        // every-view reprojection error > 32 px (gross mismatch). The
        // remaining residuals are absorbed by the Huber kernel below.
        let mut max_reproj: f64 = 0.0;
        let mut behind_camera = false;
        for &(frame_id, xy) in observations {
            let xc = corrected[frame_id].transform_world_point(&point);
            if xc.z <= 0.0 {
                behind_camera = true;
                break;
            }
            if let Some(predicted) = camera.project(&xc) {
                let err = (predicted - xy).norm();
                if err > max_reproj {
                    max_reproj = err;
                }
            }
        }
        if behind_camera {
            rejected_depth += 1;
            continue;
        }
        if max_reproj > 32.0 {
            rejected_reproj += 1;
            continue;
        }
        ba.add_landmark(track_id as u64, point);
        for &(frame_id, xy) in observations {
            ba.add_observation(BaObservation {
                keyframe_id: frame_id as u64,
                landmark_id: track_id as u64,
                xy,
            });
        }
        accepted += 1;
    }
    println!(
        "ba_triangulation accepted={} rejected_depth={} rejected_reproj={}",
        accepted, rejected_depth, rejected_reproj,
    );
    if accepted == 0 {
        println!("ba_skipped no_landmarks");
    } else {
        // Sparse Cholesky on the Schur-reduced camera system + Huber kernel
        // (δ=4 px) so any remaining matching outliers get down-weighted.
        // Huber δ=4 px clips obvious matching outliers; δ matches the
        // typical KITTI inlier residual after PGO so good observations
        // stay in the quadratic region. Sparse Cholesky on the Schur-
        // reduced camera system scales the solve to thousands of
        // landmarks with hundreds of keyframes.
        let ba_config = BaConfig {
            linear_solver: LinearSolver::Sparse,
            robust_kernel: RobustKernel::Huber { delta: 4.0 },
            initial_lambda: Some(1.0e-2),
            max_iterations: 50,
            ..BaConfig::default()
        };
        match ba.optimize(&ba_config) {
            Ok(result) => {
                println!(
                    "bundle_adjustment landmarks={} observations={} initial_cost={:.3} final_cost={:.3} iterations={} converged={}",
                    accepted,
                    ba.observations.len(),
                    result.initial_cost,
                    result.final_cost,
                    result.iterations.len(),
                    result.converged,
                );
                let ba_poses: Vec<Pose> = (0..n as u64).map(|id| ba.poses[&id].clone()).collect();
                write_xyz_csv(&args.out_dir.join("ba.csv"), &ba_poses)?;
                let last_ba = ba_poses[n - 1].camera_center_world();
                let ba_endpoint_shift = (last_ba - last_corrected).norm();
                println!(
                    "endpoint ba=[{:.3}, {:.3}, {:.3}] corrected_to_ba_shift={:.3}",
                    last_ba.x, last_ba.y, last_ba.z, ba_endpoint_shift,
                );
            }
            Err(error) => println!("bundle_adjustment error={error}"),
        }
    }

    Ok(())
}

#[cfg(feature = "image-io")]
fn triangulate_track_dlt(
    observations: &[(usize, Point2<f64>)],
    poses: &[Pose],
    camera: &visloc_rs::core::types::Camera,
) -> Option<nalgebra::Point3<f64>> {
    use nalgebra::DMatrix;
    let n = observations.len();
    if n < 2 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(n * 2, 4);
    for (i, &(frame_id, xy)) in observations.iter().enumerate() {
        let pose = &poses[frame_id];
        let normalized = camera.normalize_pixel(&xy)?;
        let matrix = pose.world_to_camera.matrix();
        let row0 = matrix.row(0);
        let row1 = matrix.row(1);
        let row2 = matrix.row(2);
        for col in 0..4 {
            a[(i * 2, col)] = normalized.x * row2[col] - row0[col];
            a[(i * 2 + 1, col)] = normalized.y * row2[col] - row1[col];
        }
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let h = v_t.row(v_t.nrows() - 1);
    let w = h[3];
    if w.abs() < 1.0e-9 {
        return None;
    }
    Some(nalgebra::Point3::new(h[0] / w, h[1] / w, h[2] / w))
}

#[cfg(feature = "image-io")]
fn build_correspondences(
    a: &FeatureSet,
    b: &FeatureSet,
    matcher: &BruteForceMatcher,
) -> Vec<TwoViewCorrespondence> {
    build_correspondences_with_indices(a, b, matcher).0
}

/// Same as [`build_correspondences`] but also returns the matched
/// `(query_index, train_index)` pairs (in the same order as the returned
/// `TwoViewCorrespondence` slice) so the demo can link multi-frame feature
/// tracks for bundle adjustment.
#[cfg(feature = "image-io")]
fn build_correspondences_with_indices(
    a: &FeatureSet,
    b: &FeatureSet,
    matcher: &BruteForceMatcher,
) -> (Vec<TwoViewCorrespondence>, Vec<(usize, usize)>) {
    if a.is_empty() || b.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let matches = matcher.match_descriptors(&a.descriptors, &b.descriptors);
    let mut correspondences = Vec::with_capacity(matches.len());
    let mut index_pairs = Vec::with_capacity(matches.len());
    for m in matches {
        let (Some(prev_xy), Some(curr_xy)) = (
            a.keypoints.get(m.query_index),
            b.keypoints.get(m.train_index),
        ) else {
            continue;
        };
        correspondences.push(TwoViewCorrespondence {
            previous_xy: Point2::new(prev_xy.x, prev_xy.y),
            current_xy: Point2::new(curr_xy.x, curr_xy.y),
        });
        index_pairs.push((m.query_index, m.train_index));
    }
    (correspondences, index_pairs)
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
