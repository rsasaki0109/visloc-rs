use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    write_online_slam_results_html_report, LocalMappingPipeline, LocalizationPipeline,
    LoopClosureConfig, OnlineSlamConfig, OnlineSlamPipeline, Tracker, TrackingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example online_slam_loop_candidate_dummy -- [--out-dir <dir>]"
        );
        std::process::exit(2);
    }

    let (map, first_frame) = map_and_frame(10, 1, Vector3::new(0.0, 0.0, 0.0));
    let (_, outbound_frame) = map_and_frame(20, 1, Vector3::new(1.4, 0.0, 0.4));
    let (_, return_frame) = map_and_frame(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 15,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            ..OnlineSlamConfig::default()
        },
    );

    let first = slam.process_frame(&first_frame, []);
    println!(
        "frame={} tracking={} keyframes={} loop_candidates={}",
        first.tracking.frame_id,
        first.tracking_succeeded(),
        first.map_keyframe_count,
        first.loop_closure_candidates.len()
    );

    let outbound = slam.process_frame(&outbound_frame, []);
    println!(
        "frame={} tracking={} keyframes={} loop_candidates={}",
        outbound.tracking.frame_id,
        outbound.tracking_succeeded(),
        outbound.map_keyframe_count,
        outbound.loop_closure_candidates.len()
    );

    let returned = slam.process_frame(&return_frame, []);
    println!(
        "frame={} tracking={} keyframes={} loop_candidates={}",
        returned.tracking.frame_id,
        returned.tracking_succeeded(),
        returned.map_keyframe_count,
        returned.loop_closure_candidates.len()
    );

    for candidate in &returned.loop_closure_candidates {
        println!(
            "loop_candidate query={} matched_keyframe={} shared_landmarks={} ratio={:.2} score={:.2} verified={}",
            candidate.query_frame_id,
            candidate.matched_keyframe_id,
            candidate.shared_landmark_count,
            candidate.shared_landmark_ratio,
            candidate.score,
            candidate.geometrically_verified
        );
    }

    let results = vec![first, outbound, returned];
    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let report_path = output_dir.join("loop_report.html");
        write_online_slam_results_html_report(&results, &report_path)?;
        println!("wrote online SLAM loop report: {}", report_path.display());
    }

    Ok(())
}

fn map_and_frame(
    frame_id: u64,
    camera_id: u64,
    camera_center_world: Vector3<f64>,
) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center_world);
    let points = [
        Point3::new(-1.0, -1.0, 5.0),
        Point3::new(1.0, -1.0, 5.0),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.0),
        Point3::new(0.0, -0.5, 6.0),
        Point3::new(0.5, 0.75, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame = Frame::new(frame_id, camera_id);

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 1.0];
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark_id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }

    (map, frame)
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
