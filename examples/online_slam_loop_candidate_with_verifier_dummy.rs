use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    loop_closure_constraints_from_candidates, verify_loop_closure_candidates,
    write_online_slam_results_html_report, EssentialMatrixLoopClosureVerifier,
    LocalMappingPipeline, LocalizationPipeline, LoopClosureConfig, LoopClosureConstraint,
    LoopClosureVerifierConfig, OnlineSlamConfig, OnlineSlamPipeline, Tracker, TrackingConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example online_slam_loop_candidate_with_verifier_dummy -- [--out-dir <dir>]"
        );
        std::process::exit(2);
    }

    // Build a 12-landmark synthetic map so the essential-matrix verifier has
    // enough correspondences to actually run.
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let landmark_points = synthetic_world_points();
    let map = build_map(&camera, &landmark_points);

    let first_frame = frame_from_pose(10, &camera, Vector3::new(0.0, 0.0, 0.0), &landmark_points);
    let outbound_frame =
        frame_from_pose(20, &camera, Vector3::new(1.4, 0.0, 0.4), &landmark_points);
    let return_frame = frame_from_pose(30, &camera, Vector3::new(0.2, 0.0, 0.1), &landmark_points);

    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 15,
                min_shared_landmarks: 8,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
        },
    );

    // Frame 30 sits at world center (0.2, 0, 0.1) while keyframe 10 is at the
    // origin. The two-view geometry recovers translation up to scale; use the
    // truth magnitude here so the constraint's translation is metric.
    let translation_scale = 0.2_f64.hypot(0.1);
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: translation_scale,
        },
        ..Default::default()
    };

    let mut results = Vec::new();
    let mut constraints: Vec<LoopClosureConstraint> = Vec::new();
    for frame in [&first_frame, &outbound_frame, &return_frame] {
        let mut result = slam.process_frame(frame, []);
        if !result.loop_closure_candidates.is_empty() {
            verify_loop_closure_candidates(
                &mut result.loop_closure_candidates,
                frame,
                &result.tracking,
                slam.map(),
                &camera,
                &verifier,
            );
        }
        let frame_constraints =
            loop_closure_constraints_from_candidates(&result.loop_closure_candidates);
        println!(
            "frame={} tracking={} keyframes={} loop_candidates={} loop_constraints={}",
            result.tracking.frame_id,
            result.tracking_succeeded(),
            result.map_keyframe_count,
            result.loop_closure_candidates.len(),
            frame_constraints.len(),
        );
        for candidate in &result.loop_closure_candidates {
            print_candidate(candidate);
        }
        for constraint in &frame_constraints {
            print_constraint(constraint);
        }
        constraints.extend(frame_constraints);
        results.push(result);
    }

    println!("total_loop_constraints={}", constraints.len());

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let report_path = output_dir.join("loop_report_with_verifier.html");
        write_online_slam_results_html_report(&results, &report_path)?;
        println!("wrote online SLAM loop report: {}", report_path.display());
    }

    Ok(())
}

fn synthetic_world_points() -> Vec<Point3<f64>> {
    vec![
        Point3::new(-1.0, -1.0, 5.0),
        Point3::new(1.0, -1.0, 5.1),
        Point3::new(-1.0, 1.0, 4.9),
        Point3::new(1.0, 1.0, 5.0),
        Point3::new(0.0, 0.0, 5.05),
        Point3::new(0.5, -0.25, 4.95),
        Point3::new(-0.6, 0.4, 4.8),
        Point3::new(0.4, 0.7, 5.2),
        Point3::new(-0.3, -0.6, 4.85),
        Point3::new(0.7, -0.5, 5.3),
        Point3::new(0.0, 0.5, 5.4),
        Point3::new(-0.7, -0.2, 4.7),
    ]
}

fn build_map(camera: &Camera, points: &[Point3<f64>]) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 1.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor);
        map.landmarks.insert(landmark.id, landmark);
    }
    map
}

fn frame_from_pose(
    frame_id: u64,
    camera: &Camera,
    camera_center_world: Vector3<f64>,
    points: &[Point3<f64>],
) -> Frame {
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center_world);
    let mut frame = Frame::new(frame_id, camera.id);
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 1.0];
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }
    frame
}

fn print_candidate(candidate: &visloc_rs::LoopClosureCandidate) {
    let verification = candidate.verification.as_ref();
    let verified = verification
        .map(|v| v.verified.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let inliers = verification
        .map(|v| v.inlier_count.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let inlier_ratio = verification
        .map(|v| format!("{:.3}", v.inlier_ratio))
        .unwrap_or_else(|| "n/a".to_string());
    let mean_sampson = verification
        .map(|v| {
            if v.mean_sampson_error.is_finite() {
                format!("{:.4}", v.mean_sampson_error)
            } else {
                "inf".to_string()
            }
        })
        .unwrap_or_else(|| "n/a".to_string());
    let verifier_score = verification
        .map(|v| format!("{:.3}", v.score))
        .unwrap_or_else(|| "n/a".to_string());
    let failure = verification
        .and_then(|v| v.failure_reason.as_ref())
        .map(|reason| format!("{:?}", reason))
        .unwrap_or_else(|| "-".to_string());
    println!(
        "loop_candidate query={} matched_keyframe={} shared={} shared_ratio={:.3} score={:.3} verified={} verifier_inliers={} verifier_inlier_ratio={} mean_sampson={} verifier_score={} failure={}",
        candidate.query_frame_id,
        candidate.matched_keyframe_id,
        candidate.shared_landmark_count,
        candidate.shared_landmark_ratio,
        candidate.score,
        verified,
        inliers,
        inlier_ratio,
        mean_sampson,
        verifier_score,
        failure,
    );
}

fn print_constraint(constraint: &LoopClosureConstraint) {
    let translation = constraint.relative_pose.translation;
    let rotation_axis_angle = constraint.relative_pose.rotation.scaled_axis();
    println!(
        "loop_constraint from_keyframe={} to_keyframe={} inliers={} inlier_ratio={:.3} mean_sampson={:.4} score={:.3} relative_translation=[{:.3}, {:.3}, {:.3}] relative_rotation_axis_angle=[{:.3}, {:.3}, {:.3}]",
        constraint.from_keyframe_id,
        constraint.to_keyframe_id,
        constraint.inlier_count,
        constraint.inlier_ratio,
        constraint.mean_sampson_error,
        constraint.score,
        translation.x,
        translation.y,
        translation.z,
        rotation_axis_angle.x,
        rotation_axis_angle.y,
        rotation_axis_angle.z,
    );
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
