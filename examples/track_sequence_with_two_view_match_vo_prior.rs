use std::collections::HashMap;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    read_two_view_matches_txt, InMemoryMapProvider, LocalizationPipeline, LocalizationPrior,
    Tracker, TrackingConfig, TrackingStats, TwoViewMatchVisualOdometryConfig,
    TwoViewMatchVisualOdometryFrontend, VisualOdometryPriorProvider,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example track_sequence_with_two_view_match_vo_prior -- [--out-dir <dir>]"
        );
        std::process::exit(2);
    }

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    // Map points are placed roughly at the same distance so the median two-view
    // pixel flow stays close to the true translation flow scaled by 1/Z. Small
    // depth jitter avoids a coplanar configuration that would make DLT PnP
    // rank-deficient.
    let map_points = [
        Point3::new(-1.0, -1.0, 5.0),
        Point3::new(1.0, -1.0, 5.1),
        Point3::new(-1.0, 1.0, 4.9),
        Point3::new(1.0, 1.0, 5.0),
        Point3::new(0.0, 0.0, 5.05),
        Point3::new(0.5, -0.25, 4.95),
    ];
    let frame_specs: [(u64, Pose); 3] = [
        (
            100,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0)),
        ),
        (
            101,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.45, 0.0, 0.0)),
        ),
        (
            102,
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.9, 0.0, -0.1)),
        ),
    ];

    let (map, descriptors) = build_map(&camera, &map_points);
    let frames = frame_specs
        .iter()
        .map(|(frame_id, pose)| {
            frame_from_projected_landmarks(*frame_id, &camera, pose, &map_points, &descriptors)
        })
        .collect::<Vec<_>>();
    let provider = InMemoryMapProvider::new(map);

    let inputs_dir = output_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("target").join("visloc_two_view_match_vo_demo"))
        .join("inputs");
    fs::create_dir_all(&inputs_dir)?;

    let frame_lookup: HashMap<u64, &Frame> = frames.iter().map(|frame| (frame.id, frame)).collect();
    let pair_specs = [(100u64, 101u64), (101u64, 102u64)];
    let mut frontend = TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
        min_matches: 4,
        min_inliers: 4,
        max_residual_pixels: 5.0,
        pixel_translation_scale: 0.01,
        forward_translation: 0.0,
    });
    let mut input_paths = Vec::new();
    for (previous_id, current_id) in pair_specs {
        let previous_frame = frame_lookup[&previous_id];
        let current_frame = frame_lookup[&current_id];
        let path = inputs_dir.join(format!("matches_{previous_id}_to_{current_id}.txt"));
        write_two_view_match_file(&path, previous_frame, current_frame)?;
        let matches = read_two_view_matches_txt(&path)?;
        frontend.insert_matches(previous_id, current_id, matches);
        input_paths.push(path);
    }

    let vo_prior_provider = VisualOdometryPriorProvider::new(frontend);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());
    let mut previous_frame: Option<&Frame> = None;
    let mut previous_pose: Option<Pose> = None;
    let mut diagnostics: Vec<FrameDiagnostics> = Vec::new();

    for frame in &frames {
        let vo_prior =
            previous_frame
                .zip(previous_pose.as_ref())
                .and_then(|(prev_frame, prev_pose)| {
                    vo_prior_provider
                        .predict_pose_prior(prev_frame, prev_pose, frame)
                        .expect("two-view match frontend cannot fail")
                });
        let (result, diag) = if let Some(vo_prior) = vo_prior {
            let prior = LocalizationPrior::from_pose(vo_prior.pose.clone(), 8.0);
            let result = tracker
                .track_frame_with_localization_prior_submap_provider(frame, &provider, &prior);
            let diag = FrameDiagnostics {
                frame_id: result.frame_id,
                used_vo_prior: true,
                match_count: vo_prior.estimate.match_count,
                inlier_count: vo_prior.estimate.inlier_count,
                mean_flow_residual_px: vo_prior.estimate.mean_reprojection_error,
                candidate_landmark_count: result.localization.candidate_landmark_count,
                localization_succeeded: result.localization.success,
                used_external_localization_prior: result.used_external_localization_prior,
                estimated_camera_center: result
                    .localization
                    .pose
                    .as_ref()
                    .map(Pose::camera_center_world),
            };
            println!(
                "frame={} vo_prior=true matches={} vo_inliers={} mean_flow_residual_px={} candidates={} success={} tracking_external_prior={} center={}",
                diag.frame_id,
                diag.match_count,
                diag.inlier_count,
                format_optional_residual(diag.mean_flow_residual_px),
                diag.candidate_landmark_count,
                diag.localization_succeeded,
                diag.used_external_localization_prior,
                format_optional_center(diag.estimated_camera_center),
            );
            (result, diag)
        } else {
            let result = tracker.track_frame_with_provider(frame, &provider);
            let diag = FrameDiagnostics {
                frame_id: result.frame_id,
                used_vo_prior: false,
                match_count: 0,
                inlier_count: 0,
                mean_flow_residual_px: None,
                candidate_landmark_count: result.localization.candidate_landmark_count,
                localization_succeeded: result.localization.success,
                used_external_localization_prior: result.used_external_localization_prior,
                estimated_camera_center: result
                    .localization
                    .pose
                    .as_ref()
                    .map(Pose::camera_center_world),
            };
            println!(
                "frame={} vo_prior=false candidates={} success={} tracking_external_prior={} center={}",
                diag.frame_id,
                diag.candidate_landmark_count,
                diag.localization_succeeded,
                diag.used_external_localization_prior,
                format_optional_center(diag.estimated_camera_center),
            );
            (result, diag)
        };

        previous_frame = Some(frame);
        previous_pose = result.localization.pose.clone();
        diagnostics.push(diag);
    }

    let stats = tracker.stats();
    println!(
        "stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        stats.frame_count,
        stats.success_rate(),
        stats.external_localization_prior_usage_rate(),
        stats.external_localization_prior_used_count,
    );

    if let Some(output_dir) = output_dir.as_deref() {
        fs::create_dir_all(output_dir)?;
        let report_path = output_dir.join("two_view_match_vo_report.txt");
        write_demo_report(&report_path, &input_paths, &diagnostics, stats)?;
        println!("wrote two-view match VO report: {}", report_path.display());
    }

    Ok(())
}

#[derive(Debug)]
struct FrameDiagnostics {
    frame_id: u64,
    used_vo_prior: bool,
    match_count: usize,
    inlier_count: usize,
    mean_flow_residual_px: Option<f64>,
    candidate_landmark_count: usize,
    localization_succeeded: bool,
    used_external_localization_prior: bool,
    estimated_camera_center: Option<Point3<f64>>,
}

fn build_map(camera: &Camera, points: &[Point3<f64>]) -> (VisualMap, Vec<Vec<f32>>) {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut descriptors = Vec::new();
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        descriptors.push(descriptor);
    }
    (map, descriptors)
}

fn frame_from_projected_landmarks(
    frame_id: u64,
    camera: &Camera,
    pose: &Pose,
    points: &[Point3<f64>],
    descriptors: &[Vec<f32>],
) -> Frame {
    let mut frame = Frame::new(frame_id, camera.id);
    for (point, descriptor) in points.iter().zip(descriptors.iter()) {
        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be in front of the camera");
        frame.keypoints.push(keypoint);
        frame.descriptors.push(descriptor.clone());
    }
    frame
}

fn write_two_view_match_file(
    path: &Path,
    previous_frame: &Frame,
    current_frame: &Frame,
) -> Result<(), Box<dyn std::error::Error>> {
    let count = previous_frame
        .keypoints
        .len()
        .min(current_frame.keypoints.len());
    let mut contents = String::from("# PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y\n");
    for index in 0..count {
        let previous_xy = previous_frame.keypoints[index];
        let current_xy = current_frame.keypoints[index];
        writeln!(
            contents,
            "{index} {index} {:.4} {:.4} {:.4} {:.4}",
            previous_xy.x, previous_xy.y, current_xy.x, current_xy.y
        )?;
    }
    fs::write(path, contents)?;
    Ok(())
}

fn write_demo_report(
    path: &Path,
    input_paths: &[PathBuf],
    diagnostics: &[FrameDiagnostics],
    stats: &TrackingStats,
) -> std::io::Result<()> {
    let mut report = String::new();
    let _ = writeln!(report, "two-view match VO demo report");
    let _ = writeln!(report, "input match files:");
    for input_path in input_paths {
        let _ = writeln!(report, "  - {}", input_path.display());
    }
    let _ = writeln!(report, "frames:");
    for diag in diagnostics {
        let _ = writeln!(
            report,
            "  frame={} vo_prior={} matches={} inliers={} mean_flow_residual_px={} candidates={} success={} tracking_external_prior={} center={}",
            diag.frame_id,
            diag.used_vo_prior,
            diag.match_count,
            diag.inlier_count,
            format_optional_residual(diag.mean_flow_residual_px),
            diag.candidate_landmark_count,
            diag.localization_succeeded,
            diag.used_external_localization_prior,
            format_optional_center(diag.estimated_camera_center),
        );
    }
    let _ = writeln!(
        report,
        "stats frames={} success_rate={:.3} external_prior_rate={:.3} external_prior_count={}",
        stats.frame_count,
        stats.success_rate(),
        stats.external_localization_prior_usage_rate(),
        stats.external_localization_prior_used_count,
    );
    fs::write(path, report)
}

fn format_optional_residual(value: Option<f64>) -> String {
    value
        .map(|residual| format!("{residual:.3}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_optional_center(center: Option<Point3<f64>>) -> String {
    center
        .map(|point| format!("[{:.3}, {:.3}, {:.3}]", point.x, point.y, point.z))
        .unwrap_or_else(|| "n/a".to_string())
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
