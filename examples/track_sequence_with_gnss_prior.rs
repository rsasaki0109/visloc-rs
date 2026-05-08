use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    write_tracking_results_csv, write_tracking_results_html_report, FramePriorSource,
    FrameTimestampIndex, GnssMeasurement, InMemoryMapProvider, LocalizationPipeline,
    MeasurementBuffer, PoseTrajectory, PriorConfig, TimeDelta, Timed, Timestamp, Tracker,
    TrackingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!("usage: cargo run --example track_sequence_with_gnss_prior -- [--out-dir <dir>]");
        std::process::exit(2);
    }

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let near_points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    let mut frame_a = Frame::new(100, camera.id);
    let mut frame_b = Frame::new(101, camera.id);
    for (index, point) in near_points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be in front of the camera");
        frame_a.keypoints.push(keypoint);
        frame_a.descriptors.push(descriptor.clone());
        frame_b.keypoints.push(keypoint);
        frame_b.descriptors.push(descriptor);
    }

    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let provider = InMemoryMapProvider::new(map);
    let frame_timestamps = FrameTimestampIndex::from_timed_frames([
        Timed::new(Timestamp::from_nanoseconds(0), frame_a.clone()),
        Timed::new(Timestamp::from_nanoseconds(100_000_000), frame_b.clone()),
    ]);
    let gnss_measurements = MeasurementBuffer::from_measurements([
        GnssMeasurement::new(Timestamp::from_nanoseconds(5_000_000), Point3::origin())
            .with_accuracy(Some(4.0), None),
        GnssMeasurement::new(
            Timestamp::from_nanoseconds(105_000_000),
            Point3::new(0.1, 0.0, 0.0),
        )
        .with_accuracy(Some(4.0), None),
    ]);
    let prior_source = FramePriorSource::new(
        frame_timestamps,
        gnss_measurements,
        TimeDelta::from_nanoseconds(20_000_000),
    )
    .with_prior_config(PriorConfig {
        default_radius: 50.0,
        min_radius: 2.0,
        confidence_multiplier: 2.0,
    });

    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    let mut results = Vec::new();
    for frame in [&frame_a, &frame_b] {
        let prior = prior_source
            .localization_prior_for_frame(frame)
            .expect("dummy GNSS prior must be available");
        let result =
            tracker.track_frame_with_localization_prior_submap_provider(frame, &provider, &prior);

        println!(
            "frame={} gnss_radius={:?} map_landmarks={} candidates={} success={} inliers={} event={:?}",
            result.frame_id,
            prior.radius,
            result.map_landmark_count,
            result.localization.candidate_landmark_count,
            result.localization.success,
            result.localization.inlier_count,
            result.event,
        );
        results.push(result);
    }

    let stats = tracker.stats();
    println!(
        "stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        stats.frame_count,
        stats.success_rate(),
        stats.external_localization_prior_usage_rate(),
        stats.external_localization_prior_used_count,
    );

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let tracking_csv_path = output_dir.join("tracking.csv");
        let tracking_summary_path = output_dir.join("tracking_summary.json");
        let tracking_report_path = output_dir.join("tracking_report.html");
        let trajectory_report_path = output_dir.join("trajectory_report.html");
        let trajectory = PoseTrajectory::from_tracking_results(&results);
        write_tracking_results_csv(&results, &tracking_csv_path)?;
        tracker.stats().write_json(&tracking_summary_path)?;
        write_tracking_results_html_report(&results, &tracking_report_path)?;
        trajectory.write_html_report(&trajectory_report_path)?;
        println!(
            "wrote gnss tracking exports: csv={} summary={} tracking_report={} trajectory_report={}",
            tracking_csv_path.display(),
            tracking_summary_path.display(),
            tracking_report_path.display(),
            trajectory_report_path.display()
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
