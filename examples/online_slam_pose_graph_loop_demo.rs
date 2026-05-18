use std::env;
use std::fs;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    loop_closure_constraints_from_candidates, relative_world_to_camera,
    verify_loop_closure_candidates, write_online_slam_results_html_report,
    EssentialMatrixLoopClosureVerifier, LocalMappingPipeline, LocalizationPipeline,
    LoopClosureConfig, LoopClosureConstraint, LoopClosureVerifierConfig, OnlineSlamConfig,
    OnlineSlamPipeline, OnlineSlamResult, PoseGraph, PoseGraphSe3Config, Tracker, TrackingConfig,
};

const KEYFRAME_IDS: [u64; 6] = [10, 20, 30, 40, 50, 60];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let output_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example online_slam_pose_graph_loop_demo -- [--out-dir <dir>]"
        );
        std::process::exit(2);
    }

    // A six-frame synthetic loop. Keyframe 10 is at the world origin and
    // keyframe 60 sits close to it again, so the shared-landmark detector
    // should propose a loop candidate that the verifier can confirm. The
    // outbound keyframes spread far enough that map-based localization can
    // still triangulate against the same 14 landmarks.
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let landmark_points = synthetic_world_points();
    let map = build_map(&camera, &landmark_points);

    let camera_centers: [Vector3<f64>; KEYFRAME_IDS.len()] = [
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.6, 0.0, 0.2),
        Vector3::new(1.1, 0.0, 0.5),
        Vector3::new(0.9, 0.0, 0.9),
        Vector3::new(0.4, 0.0, 0.6),
        Vector3::new(0.1, 0.0, 0.05),
    ];
    let frames: Vec<Frame> = KEYFRAME_IDS
        .iter()
        .zip(camera_centers.iter())
        .map(|(frame_id, center)| frame_from_pose(*frame_id, &camera, *center, &landmark_points))
        .collect();

    // Require a frame-id gap of at least 50 so only the keyframe-10 ↔
    // keyframe-60 pair qualifies as a loop closure. The verifier scale is
    // calibrated for that specific pair; allowing intermediate pairs through
    // would break the per-pair scale assumption and bias the pose graph.
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 50,
                min_shared_landmarks: 10,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            ..OnlineSlamConfig::default()
        },
    );

    // Keyframe 60 sits at world (0.1, 0, 0.05); use the truth magnitude as the
    // verifier's translation scale so the recovered relative pose is metric.
    let loop_translation_scale = 0.1_f64.hypot(0.05);
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.5,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: loop_translation_scale,
        },
        ..Default::default()
    };

    let mut results: Vec<OnlineSlamResult> = Vec::new();
    let mut all_constraints: Vec<LoopClosureConstraint> = Vec::new();
    for frame in &frames {
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
            let verification = candidate.verification.as_ref();
            let verified = verification
                .map(|v| v.verified.to_string())
                .unwrap_or_else(|| "n/a".to_string());
            println!(
                "  candidate query={} matched_keyframe={} shared={} ratio={:.3} verified={}",
                candidate.query_frame_id,
                candidate.matched_keyframe_id,
                candidate.shared_landmark_count,
                candidate.shared_landmark_ratio,
                verified,
            );
        }
        for constraint in &frame_constraints {
            let translation = constraint.relative_pose.translation;
            println!(
                "  constraint from={} to={} relative_t=[{:.3}, {:.3}, {:.3}] inliers={} score={:.3}",
                constraint.from_keyframe_id,
                constraint.to_keyframe_id,
                translation.x,
                translation.y,
                translation.z,
                constraint.inlier_count,
                constraint.score,
            );
        }
        all_constraints.extend(frame_constraints);
        results.push(result);
    }

    println!("total_loop_constraints={}", all_constraints.len());

    // Build a pose graph over the tracked keyframes.
    let mut graph = PoseGraph::new();
    for (frame_id, result) in KEYFRAME_IDS.iter().zip(results.iter()) {
        if let Some(pose) = result.tracking.localization.pose.clone() {
            graph.add_pose(*frame_id, pose);
        }
    }
    for window in KEYFRAME_IDS.windows(2) {
        let from = window[0];
        let to = window[1];
        let (Some(from_pose), Some(to_pose)) = (
            graph.poses.get(&from).cloned(),
            graph.poses.get(&to).cloned(),
        ) else {
            continue;
        };
        graph.add_sequential_edge(from, to, relative_world_to_camera(&from_pose, &to_pose));
    }
    for constraint in &all_constraints {
        graph.add_loop_closure_constraint(constraint);
    }
    graph.anchor(KEYFRAME_IDS[0]);

    println!(
        "pose_graph nodes={} sequential_edges={} loop_constraints={} translation_cost={:.6}",
        graph.poses.len(),
        graph.edges.len() - all_constraints.len(),
        all_constraints.len(),
        graph.translation_cost(),
    );

    // Inject a deliberate drift on the most recent keyframe and watch the
    // single Gauss-Newton step pull it back along the loop.
    let drift = Vector3::new(0.06, 0.03, -0.05);
    if let Some(last_id) = KEYFRAME_IDS.last() {
        if let Some(pose) = graph.poses.get_mut(last_id) {
            let drifted_center = pose.camera_center_world() + drift;
            pose.world_to_camera.translation =
                -(pose.world_to_camera.rotation.to_rotation_matrix() * drifted_center.coords);
        }
    }
    let drifted_center = graph
        .poses
        .get(&KEYFRAME_IDS[5])
        .map(|pose| pose.camera_center_world());
    if let Some(center) = drifted_center {
        println!(
            "drifted_keyframe id={} center=[{:.3}, {:.3}, {:.3}] (truth=[{:.3}, {:.3}, {:.3}])",
            KEYFRAME_IDS[5],
            center.x,
            center.y,
            center.z,
            camera_centers[5].x,
            camera_centers[5].y,
            camera_centers[5].z,
        );
    }

    match graph.optimize_translations_once() {
        Ok(step) => {
            println!(
                "pose_graph_step anchor={} edges={} variables={} cost_before={:.6} cost_after={:.6} mean_correction={:.6} max_correction={:.6}",
                step.anchor_id,
                step.edge_count,
                step.variable_count,
                step.cost_before,
                step.cost_after,
                step.mean_translation_correction,
                step.max_translation_correction,
            );
            for (frame_id, truth) in KEYFRAME_IDS.iter().zip(camera_centers.iter()) {
                let center = graph
                    .poses
                    .get(frame_id)
                    .map(|pose| pose.camera_center_world());
                if let Some(center) = center {
                    let truth_point = Point3::from(*truth);
                    let error = (center - truth_point).norm();
                    println!(
                        "  post_optim keyframe={} center=[{:.3}, {:.3}, {:.3}] truth=[{:.3}, {:.3}, {:.3}] err={:.4}",
                        frame_id,
                        center.x,
                        center.y,
                        center.z,
                        truth.x,
                        truth.y,
                        truth.z,
                        error,
                    );
                }
            }
        }
        Err(error) => println!("pose_graph_step error={error}"),
    }

    // Demonstrate rotation-aware correction: translation-only GN already pulled
    // KF60 back to truth above, so now we inject a combined translation +
    // rotation drift onto KF60 and run the full SE(3) Gauss-Newton solver.
    let rotation_drift = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.18);
    let translation_drift = Vector3::new(0.04, 0.0, -0.03);
    if let Some(last_id) = KEYFRAME_IDS.last() {
        if let Some(pose) = graph.poses.get_mut(last_id) {
            let truth_center = pose.camera_center_world().coords + translation_drift;
            let new_rotation = rotation_drift * pose.world_to_camera.rotation;
            let new_translation = -(new_rotation.transform_vector(&truth_center));
            pose.world_to_camera.rotation = new_rotation;
            pose.world_to_camera.translation = new_translation;
        }
    }
    println!(
        "se3_drift_applied keyframe={} se3_cost_before={:.6}",
        KEYFRAME_IDS[5],
        graph.se3_cost(),
    );

    match graph.optimize_se3_iterative(&PoseGraphSe3Config::default()) {
        Ok(result) => {
            println!(
                "pose_graph_se3 anchor={} edges={} variables={} initial_cost={:.6} final_cost={:.6} iterations={} converged={}",
                result.anchor_id,
                result.edge_count,
                result.variable_count,
                result.initial_cost,
                result.final_cost,
                result.iterations.len(),
                result.converged,
            );
            for stats in &result.iterations {
                println!(
                    "  iter={} cost_before={:.6} cost_after={:.6} max_step={:.6}",
                    stats.iteration, stats.cost_before, stats.cost_after, stats.max_step_norm,
                );
            }
            for (frame_id, truth) in KEYFRAME_IDS.iter().zip(camera_centers.iter()) {
                let pose = graph.poses.get(frame_id);
                if let Some(pose) = pose {
                    let center = pose.camera_center_world();
                    let truth_point = Point3::from(*truth);
                    let translation_err = (center - truth_point).norm();
                    // All truth keyframes are at identity rotation, so the residual
                    // angle is just the rotation magnitude of the current quaternion.
                    let rotation_err = pose.world_to_camera.rotation.angle();
                    println!(
                        "  post_se3 keyframe={} center=[{:.3}, {:.3}, {:.3}] t_err={:.4} rot_err={:.4}",
                        frame_id,
                        center.x,
                        center.y,
                        center.z,
                        translation_err,
                        rotation_err,
                    );
                }
            }
        }
        Err(error) => println!("pose_graph_se3 error={error}"),
    }

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir)?;
        let report_path = output_dir.join("loop_demo_report.html");
        write_online_slam_results_html_report(&results, &report_path)?;
        println!(
            "wrote pose-graph loop demo report: {}",
            report_path.display()
        );
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
        Point3::new(0.3, 0.3, 5.1),
        Point3::new(-0.4, 0.6, 4.95),
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
