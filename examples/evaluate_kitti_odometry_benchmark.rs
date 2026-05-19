use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::{
    KittiOdometryBenchmarkConfig, Pose, PoseTrajectory, TrackingEvent, TrackingState,
    TrajectorySample, SE3,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    let lengths = parse_lengths(&mut args)
        .unwrap_or_else(|| KittiOdometryBenchmarkConfig::default().segment_lengths_m);
    let start_frame_step = parse_usize_flag(&mut args, "--start-step").unwrap_or(1);
    let config = KittiOdometryBenchmarkConfig {
        segment_lengths_m: lengths,
        start_frame_step,
    };

    let (estimated, reference) = match args.as_slice() {
        [] => synthetic_fixture(),
        [estimated_path, reference_path] => {
            let estimated = PoseTrajectory::read_kitti_poses(estimated_path)?;
            let reference = PoseTrajectory::read_kitti_poses(reference_path)?;
            (estimated, reference)
        }
        _ => {
            eprintln!(
                "usage: cargo run --example evaluate_kitti_odometry_benchmark -- \
                 [--out-dir <dir>] [--lengths 100,200,...,800] [--start-step <n>] \
                 [estimated_poses.txt reference_poses.txt]"
            );
            std::process::exit(2);
        }
    };

    let summary = estimated.kitti_odometry_benchmark_against(&reference, &config);
    println!(
        "loaded trajectories: estimated={} reference={} matched={} lengths={:?}",
        summary.estimated_pose_count,
        summary.reference_pose_count,
        summary.matched_pose_count,
        config.segment_lengths_m
    );
    println!(
        "KITTI odometry windows={} t_rel={:?}% r_rel={:?} deg/m",
        summary.segment_count,
        summary.mean_translational_error_percent,
        summary.mean_rotational_error_deg_per_m
    );
    println!(
        "KITTI odometry worst window: max_t_rel={:?}% max_r_rel={:?} deg/m",
        summary.max_translational_error_percent(),
        summary.max_rotational_error_deg_per_m()
    );
    println!("segment_errors_csv:\n{}", summary.segment_errors_csv());
    println!("summary_json:\n{}", summary.to_json());

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let segments_path = output_dir.join("kitti_odometry_segments.csv");
        let summary_path = output_dir.join("kitti_odometry_summary.json");
        summary.write_segment_errors_csv(&segments_path)?;
        summary.write_json(&summary_path)?;
        println!(
            "wrote KITTI odometry exports: segments={} summary={}",
            segments_path.display(),
            summary_path.display()
        );
    }

    Ok(())
}

fn synthetic_fixture() -> (PoseTrajectory, PoseTrajectory) {
    let mut estimated = PoseTrajectory::new();
    let mut reference = PoseTrajectory::new();
    for frame_id in 0..=12_u64 {
        let x_ref = frame_id as f64 * 10.0;
        let x_est = x_ref * 1.01;
        reference.push_sample(sample(frame_id, x_ref));
        estimated.push_sample(sample(frame_id, x_est));
    }
    (estimated, reference)
}

fn sample(frame_id: u64, x: f64) -> TrajectorySample {
    let camera_to_world = SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0));
    TrajectorySample {
        frame_id,
        pose: Pose {
            world_to_camera: camera_to_world.inverse(),
        },
        state: TrackingState::Tracking,
        event: TrackingEvent::Tracked,
        inlier_count: 0,
        inlier_ratio: 0.0,
        reprojection_error: None,
    }
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let output_flag_index = args.iter().position(|arg| arg == "--out-dir")?;
    if output_flag_index + 1 >= args.len() {
        eprintln!("--out-dir requires a directory path");
        std::process::exit(2);
    }
    let output_dir = PathBuf::from(args.remove(output_flag_index + 1));
    args.remove(output_flag_index);
    Some(output_dir)
}

fn parse_lengths(args: &mut Vec<String>) -> Option<Vec<f64>> {
    let flag_index = args.iter().position(|arg| arg == "--lengths")?;
    if flag_index + 1 >= args.len() {
        eprintln!("--lengths requires a comma-separated list");
        std::process::exit(2);
    }
    let value = args.remove(flag_index + 1);
    args.remove(flag_index);
    let lengths = value
        .split(',')
        .filter(|field| !field.trim().is_empty())
        .map(|field| {
            field.trim().parse::<f64>().unwrap_or_else(|error| {
                eprintln!("invalid --lengths entry {field:?}: {error}");
                std::process::exit(2);
            })
        })
        .collect::<Vec<_>>();
    if lengths.is_empty() {
        eprintln!("--lengths must include at least one value");
        std::process::exit(2);
    }
    Some(lengths)
}

fn parse_usize_flag(args: &mut Vec<String>, flag: &str) -> Option<usize> {
    let flag_index = args.iter().position(|arg| arg == flag)?;
    if flag_index + 1 >= args.len() {
        eprintln!("{flag} requires an integer value");
        std::process::exit(2);
    }
    let value = args.remove(flag_index + 1);
    args.remove(flag_index);
    Some(value.parse::<usize>().unwrap_or_else(|error| {
        eprintln!("invalid value for {flag}: {error}");
        std::process::exit(2);
    }))
}
