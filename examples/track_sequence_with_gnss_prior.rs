use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    write_tracking_results_csv, write_tracking_results_html_report, FramePriorSource,
    FrameTimestampIndex, GnssMeasurement, InMemoryMapProvider, LocalizationPipeline,
    MeasurementBuffer, PoseTrajectory, PriorConfig, TimeDelta, Timed, Timestamp, Tracker,
    TrackingConfig, TrackingStats,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!("usage: cargo run --example track_sequence_with_gnss_prior -- [--out-dir <dir>]");
        std::process::exit(2);
    }

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
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

    let mut descriptors = Vec::new();
    for (index, point) in near_points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        descriptors.push(descriptor);
    }

    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let poses = [
        (
            100,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
            Point3::new(0.0, 0.0, 0.0),
        ),
        (
            101,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.45, 0.0, 0.0)),
            Point3::new(0.45, 0.0, 0.0),
        ),
        (
            102,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.9, 0.0, -0.1)),
            Point3::new(0.9, 0.0, 0.1),
        ),
    ];
    let frames = poses
        .iter()
        .map(|(frame_id, pose, _center)| {
            frame_from_projected_landmarks(*frame_id, &camera, pose, &near_points, &descriptors)
        })
        .collect::<Vec<_>>();

    let provider = InMemoryMapProvider::new(map);
    let frame_timestamps =
        FrameTimestampIndex::from_timed_frames(frames.iter().enumerate().map(|(index, frame)| {
            Timed::new(
                Timestamp::from_nanoseconds(index as i128 * 100_000_000),
                frame.clone(),
            )
        }));
    let gnss_measurements = MeasurementBuffer::from_measurements(poses.iter().enumerate().map(
        |(index, (_frame_id, _pose, center))| {
            GnssMeasurement::new(
                Timestamp::from_nanoseconds(index as i128 * 100_000_000 + 5_000_000),
                *center,
            )
            .with_accuracy(Some(4.0), None)
        },
    ));
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
    for frame in &frames {
        let prior = prior_source
            .localization_prior_for_frame(frame)
            .expect("dummy GNSS prior must be available");
        let result =
            tracker.track_frame_with_localization_prior_submap_provider(frame, &provider, &prior);

        println!(
            "frame={} gnss_radius={:?} map_landmarks={} candidates={} success={} inliers={} event={:?} center={}",
            result.frame_id,
            prior.radius,
            result.map_landmark_count,
            result.localization.candidate_landmark_count,
            result.localization.success,
            result.localization.inlier_count,
            result.event,
            format_estimated_center(&result.localization.pose),
        );
        results.push(result);
    }

    let stats = tracker.stats();
    let trajectory = PoseTrajectory::from_tracking_results(&results);
    println!(
        "stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={} trajectory_poses={} path_length={:.3}",
        stats.frame_count,
        stats.success_rate(),
        stats.external_localization_prior_usage_rate(),
        stats.external_localization_prior_used_count,
        trajectory.len(),
        trajectory.total_path_length(),
    );

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let tracking_csv_path = output_dir.join("tracking.csv");
        let tracking_summary_path = output_dir.join("tracking_summary.json");
        let tracking_report_path = output_dir.join("tracking_report.html");
        let trajectory_csv_path = output_dir.join("trajectory.csv");
        let trajectory_summary_path = output_dir.join("trajectory_summary.json");
        let trajectory_report_path = output_dir.join("trajectory_report.html");
        let demo_index_path = output_dir.join("index.html");
        write_tracking_results_csv(&results, &tracking_csv_path)?;
        tracker.stats().write_json(&tracking_summary_path)?;
        write_tracking_results_html_report(&results, &tracking_report_path)?;
        trajectory.write_csv(&trajectory_csv_path)?;
        trajectory.write_summary_json(&trajectory_summary_path)?;
        trajectory.write_html_report(&trajectory_report_path)?;
        write_demo_index_html(&demo_index_path, tracker.stats(), &trajectory)?;
        println!(
            "wrote gnss tracking exports: index={} tracking_csv={} tracking_summary={} tracking_report={} trajectory_csv={} trajectory_summary={} trajectory_report={}",
            demo_index_path.display(),
            tracking_csv_path.display(),
            tracking_summary_path.display(),
            tracking_report_path.display(),
            trajectory_csv_path.display(),
            trajectory_summary_path.display(),
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

fn write_demo_index_html(
    path: impl AsRef<Path>,
    stats: &TrackingStats,
    trajectory: &PoseTrajectory,
) -> std::io::Result<()> {
    fs::write(path, demo_index_html(stats, trajectory))
}

fn demo_index_html(stats: &TrackingStats, trajectory: &PoseTrajectory) -> String {
    format!(
        "<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>visloc-rs GNSS-prior localization demo</title>
<style>
body{{margin:0;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#182026}}
main{{max-width:960px;margin:0 auto;padding:28px}}
h1{{font-size:26px;margin:0 0 8px}}
.sub{{margin:0 0 22px;color:#52616b;line-height:1.5}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:18px 0}}
.metric,.link{{background:white;border:1px solid #dde3ea;border-radius:8px;padding:14px}}
.label{{display:block;font-size:12px;color:#65727e}}
.value{{display:block;font-size:23px;font-weight:700;margin-top:4px}}
.links{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:18px}}
a{{color:#185abc;text-decoration:none;font-weight:700}}
.detail{{display:block;color:#52616b;font-size:13px;margin-top:6px;line-height:1.4}}
</style>
</head>
<body>
<main>
<h1>visloc-rs GNSS-prior localization demo</h1>
<p class=\"sub\">Moving-camera sequence localization with a reusable sparse visual map. GNSS measurements create an external localization prior that narrows the map before PnP localization, then successful poses are exported as a camera trajectory.</p>
<section class=\"grid\">
<div class=\"metric\"><span class=\"label\">Frames</span><span class=\"value\">{}</span></div>
<div class=\"metric\"><span class=\"label\">Success rate</span><span class=\"value\">{:.1}%</span></div>
<div class=\"metric\"><span class=\"label\">External prior</span><span class=\"value\">{} ({:.1}%)</span></div>
<div class=\"metric\"><span class=\"label\">Trajectory poses</span><span class=\"value\">{}</span></div>
<div class=\"metric\"><span class=\"label\">Path length</span><span class=\"value\">{:.3} m</span></div>
</section>
<section class=\"links\">
<div class=\"link\"><a href=\"tracking_report.html\">Tracking report</a><span class=\"detail\">Frame states, inliers, failures, and external-prior usage.</span></div>
<div class=\"link\"><a href=\"trajectory_report.html\">Trajectory report</a><span class=\"detail\">Top-down camera-center path estimated from localized frames.</span></div>
<div class=\"link\"><a href=\"tracking.csv\">tracking.csv</a><span class=\"detail\">Frame-by-frame localization diagnostics.</span></div>
<div class=\"link\"><a href=\"trajectory.csv\">trajectory.csv</a><span class=\"detail\">Estimated camera centers and poses for plotting or regression checks.</span></div>
<div class=\"link\"><a href=\"tracking_summary.json\">tracking_summary.json</a><span class=\"detail\">Aggregate tracking success and prior-use metrics.</span></div>
<div class=\"link\"><a href=\"trajectory_summary.json\">trajectory_summary.json</a><span class=\"detail\">Pose count, path length, bounds, and reprojection summary.</span></div>
</section>
</main>
</body>
</html>
",
        stats.frame_count,
        stats.success_rate() * 100.0,
        stats.external_localization_prior_used_count,
        stats.external_localization_prior_usage_rate() * 100.0,
        trajectory.len(),
        trajectory.total_path_length(),
    )
}

fn frame_from_projected_landmarks(
    frame_id: u64,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
    descriptors: &[Vec<f32>],
) -> Frame {
    let mut frame = Frame::new(frame_id, camera.id);
    for (point, descriptor) in points.iter().zip(descriptors) {
        let keypoint: Point2<f64> = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be visible in the moving camera");
        frame.keypoints.push(keypoint);
        frame.descriptors.push(descriptor.clone());
    }
    frame
}

fn format_estimated_center(pose: &Option<Pose>) -> String {
    if let Some(pose) = pose {
        let center = pose.camera_center_world();
        format!("[{:.3}, {:.3}, {:.3}]", center.x, center.y, center.z)
    } else {
        "none".to_string()
    }
}
