use std::env;
use std::fs;
use std::path::PathBuf;

use visloc_rs::core::types::Frame;
use visloc_rs::io::colmap::ColmapMapProvider;
use visloc_rs::io::query_features::read_query_features_txt;
use visloc_rs::{
    write_tracking_results_csv, write_tracking_results_html_report, DescriptorProvider,
    LocalizationPipeline, MapProvider, PoseTrajectory, Tracker, TrackingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    let example_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples");
    let (map_dir, descriptor_path, camera_id, query_feature_paths) = match args.as_slice() {
        [] => {
            let default_query = example_dir.join("data").join("query_features.txt");
            (
                example_dir.join("data").join("colmap_text"),
                example_dir.join("data").join("landmark_descriptors.txt"),
                1_u64,
                vec![
                    default_query.clone(),
                    default_query.clone(),
                    default_query.clone(),
                ],
            )
        }
        [map_dir, descriptor_path, camera_id, query_feature_paths @ ..]
            if !query_feature_paths.is_empty() =>
        {
            (
                PathBuf::from(map_dir),
                PathBuf::from(descriptor_path),
                camera_id.parse()?,
                query_feature_paths.iter().map(PathBuf::from).collect(),
            )
        }
        _ => {
            eprintln!(
                "usage: cargo run --example localize_sequence_from_files -- [--out-dir <dir>] <colmap_text_dir> <landmark_descriptors.txt> <camera_id> <query_features.txt> [query_features_2.txt ...]"
            );
            std::process::exit(2);
        }
    };

    let provider = ColmapMapProvider::from_text_model_dir_with_descriptors_validated(
        map_dir,
        descriptor_path,
    )?;
    let map = provider.visual_map();
    if !map.cameras.contains_key(&camera_id) {
        return Err(format!("camera id {camera_id} not found in map").into());
    }

    let mut frames = Vec::new();
    for (index, query_feature_path) in query_feature_paths.iter().enumerate() {
        let features = read_query_features_txt(query_feature_path)?;
        frames.push(Frame {
            id: index as u64,
            camera_id,
            keypoints: features.keypoints,
            descriptors: features.descriptors,
            pose: None,
        });
    }

    let descriptor_count = provider
        .landmark_descriptor_store()
        .map(|store| store.len())
        .unwrap_or(0);
    println!(
        "loaded map: cameras={} keyframes={} landmarks={} descriptors={} frames={}",
        map.cameras.len(),
        map.keyframes.len(),
        map.landmarks.len(),
        descriptor_count,
        frames.len()
    );

    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    let results = tracker.track_frames_with_provider(&frames, &provider);
    for result in &results {
        println!(
            "frame={} success={} event={:?} inliers={} ratio={:.3} reprojection_error={:?}",
            result.frame_id,
            result.localization.success,
            result.event,
            result.localization.inlier_count,
            result.localization.inlier_ratio,
            result.localization.reprojection_error,
        );
    }

    let trajectory = PoseTrajectory::from_tracking_results(&results);
    println!(
        "trajectory poses={} path_length={:.6} mean_reprojection_error={:?}",
        trajectory.len(),
        trajectory.total_path_length(),
        trajectory.mean_reprojection_error(),
    );
    println!("trajectory_csv:\n{}", trajectory.to_csv());
    println!("trajectory_kitti_poses:\n{}", trajectory.to_kitti_poses());
    println!("trajectory_tum_poses:\n{}", trajectory.to_tum_poses());
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let csv_path = output_dir.join("trajectory.csv");
        let kitti_path = output_dir.join("poses.txt");
        let tum_path = output_dir.join("trajectory_tum.txt");
        let summary_path = output_dir.join("summary.json");
        let tracking_csv_path = output_dir.join("tracking.csv");
        let trajectory_report_path = output_dir.join("trajectory_report.html");
        let tracking_report_path = output_dir.join("tracking_report.html");
        trajectory.write_csv(&csv_path)?;
        trajectory.write_kitti_poses(&kitti_path)?;
        trajectory.write_tum_poses(&tum_path)?;
        trajectory.write_summary_json(&summary_path)?;
        write_tracking_results_csv(&results, &tracking_csv_path)?;
        trajectory.write_html_report(&trajectory_report_path)?;
        write_tracking_results_html_report(&results, &tracking_report_path)?;
        println!(
            "wrote trajectory exports: csv={} kitti={} tum={} summary={} tracking_csv={} trajectory_report={} tracking_report={}",
            csv_path.display(),
            kitti_path.display(),
            tum_path.display(),
            summary_path.display(),
            tracking_csv_path.display(),
            trajectory_report_path.display(),
            tracking_report_path.display()
        );
    }

    Ok(())
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
