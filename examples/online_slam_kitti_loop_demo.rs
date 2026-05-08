//! KITTI loop-closure pose-graph demo on real public-data ground-truth poses.
//!
//! Loads a KITTI odometry ground-truth pose file (e.g.,
//! `<dataset>/poses/00.txt`), subsamples it to a manageable keyframe set,
//! fabricates a realistic odometry drift by perturbing each sequential edge's
//! yaw, and adds a single truth-relative loop-closure constraint between the
//! first and last keyframes (KITTI 00 returns close to its starting pose).
//! The full SE(3) Levenberg-Marquardt + Cholesky solver
//! (`PoseGraph::optimize_se3_iterative`) is then run on the resulting graph
//! and the truth / drifted / corrected trajectories are written as CSV files
//! for downstream visualization (`scripts/build_kitti_loop_asset.py`).
//!
//! Usage:
//!
//! ```sh
//! cargo run --example online_slam_kitti_loop_demo -- \
//!     --kitti-poses /path/to/KITTI_odometry/poses/00.txt \
//!     --keyframe-stride 30 \
//!     --yaw-drift-deg-per-edge 0.45 \
//!     --out-dir target/kitti_loop_demo
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::{Pose, SE3};
use visloc_rs::tracking::PoseTrajectory;
use visloc_rs::{relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphSe3Config};

#[derive(Debug)]
struct CliArgs {
    kitti_poses: PathBuf,
    keyframe_stride: usize,
    yaw_drift_per_edge_rad: f64,
    out_dir: PathBuf,
    max_keyframes: usize,
}

fn parse_args() -> Result<CliArgs, Box<dyn std::error::Error>> {
    let mut kitti_poses: Option<PathBuf> = None;
    let mut keyframe_stride: usize = 30;
    let mut yaw_drift_deg: f64 = 0.45;
    let mut out_dir: PathBuf = PathBuf::from("target/kitti_loop_demo");
    let mut max_keyframes: usize = 200;

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kitti-poses" => {
                kitti_poses = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            "--keyframe-stride" => {
                keyframe_stride = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--yaw-drift-deg-per-edge" => {
                yaw_drift_deg = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            "--out-dir" => {
                out_dir = PathBuf::from(args.remove(i + 1));
                args.remove(i);
            }
            "--max-keyframes" => {
                max_keyframes = args.remove(i + 1).parse()?;
                args.remove(i);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let kitti_poses = kitti_poses
        .ok_or("--kitti-poses <path/to/poses/SS.txt> is required (KITTI odometry GT pose file)")?;
    Ok(CliArgs {
        kitti_poses,
        keyframe_stride,
        yaw_drift_per_edge_rad: yaw_drift_deg.to_radians(),
        out_dir,
        max_keyframes,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let trajectory = PoseTrajectory::read_kitti_poses(&args.kitti_poses)?;
    let samples = trajectory.samples();
    if samples.is_empty() {
        return Err("KITTI pose file is empty".into());
    }
    println!(
        "kitti_poses={} total_samples={} stride={} max_keyframes={} yaw_drift_per_edge={:.4} rad",
        args.kitti_poses.display(),
        samples.len(),
        args.keyframe_stride,
        args.max_keyframes,
        args.yaw_drift_per_edge_rad,
    );

    // Subsample with stride; cap at max_keyframes.
    let mut keyframes: Vec<Pose> = samples
        .iter()
        .step_by(args.keyframe_stride.max(1))
        .map(|s| s.pose.clone())
        .collect();
    if keyframes.len() > args.max_keyframes {
        keyframes.truncate(args.max_keyframes);
    }
    let n = keyframes.len();
    if n < 4 {
        return Err(format!("need at least 4 keyframes after subsampling, got {n}").into());
    }
    println!("keyframe_count={n}");

    // Truth sequential edges.
    let mut truth_edges: Vec<SE3> = Vec::with_capacity(n - 1);
    for w in keyframes.windows(2) {
        truth_edges.push(relative_world_to_camera(&w[0], &w[1]));
    }

    // Inject yaw drift on each sequential edge to simulate odometry error.
    let yaw_drift_rot =
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), args.yaw_drift_per_edge_rad);
    let noisy_edges: Vec<SE3> = truth_edges
        .iter()
        .map(|edge| SE3::new(yaw_drift_rot * edge.rotation, edge.translation))
        .collect();

    // Initial drifted node estimates: integrate noisy edges from the anchor.
    let mut drifted: Vec<Pose> = vec![keyframes[0].clone()];
    for edge in &noisy_edges {
        let last = drifted.last().unwrap();
        let new_world_to_camera = edge.compose(&last.world_to_camera);
        drifted.push(Pose {
            world_to_camera: new_world_to_camera,
        });
    }

    // Loop closure: KITTI 00 returns close to origin, so add a single
    // truth-relative constraint between the first and last keyframes.
    let loop_edge = relative_world_to_camera(&keyframes[0], &keyframes[n - 1]);

    // Build the pose graph from the drifted state.
    let mut graph = PoseGraph::new();
    for (id, pose) in drifted.iter().enumerate() {
        graph.add_pose(id as u64, pose.clone());
    }
    graph.anchor(0);
    for (i, edge) in noisy_edges.iter().enumerate() {
        graph.add_sequential_edge(i as u64, (i + 1) as u64, edge.clone());
    }
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 0,
        to_keyframe_id: (n - 1) as u64,
        relative_pose: loop_edge,
        inlier_count: 100,
        inlier_ratio: 1.0,
        mean_sampson_error: 0.0,
        score: 100.0,
    });

    println!("se3_cost_before_optimization={:.6}", graph.se3_cost());

    let result = graph.optimize_se3_iterative(&PoseGraphSe3Config {
        initial_lambda: Some(1.0e-3),
        max_iterations: 50,
        ..PoseGraphSe3Config::default()
    })?;

    println!(
        "result anchor={} edges={} variables={} initial_cost={:.4} final_cost={:.4} iterations={} converged={}",
        result.anchor_id,
        result.edge_count,
        result.variable_count,
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

    let corrected: Vec<Pose> = (0..n as u64).map(|id| graph.poses[&id].clone()).collect();

    fs::create_dir_all(&args.out_dir)?;
    write_xz_csv(&args.out_dir.join("truth.csv"), &keyframes)?;
    write_xz_csv(&args.out_dir.join("drifted.csv"), &drifted)?;
    write_xz_csv(&args.out_dir.join("corrected.csv"), &corrected)?;

    let truth_last = keyframes[n - 1].camera_center_world();
    let drifted_last = drifted[n - 1].camera_center_world();
    let corrected_last = corrected[n - 1].camera_center_world();
    println!(
        "trajectory_endpoint_drift truth=[{:.3}, {:.3}, {:.3}] drifted_err={:.3} corrected_err={:.3}",
        truth_last.x,
        truth_last.y,
        truth_last.z,
        (drifted_last - truth_last).norm(),
        (corrected_last - truth_last).norm(),
    );
    println!("trajectories written to {}", args.out_dir.display());
    Ok(())
}

fn write_xz_csv(path: &Path, poses: &[Pose]) -> std::io::Result<()> {
    let mut s = String::from("id,x,y,z\n");
    for (i, pose) in poses.iter().enumerate() {
        let c = pose.camera_center_world();
        s.push_str(&format!("{i},{:.6},{:.6},{:.6}\n", c.x, c.y, c.z));
    }
    fs::write(path, s)
}
