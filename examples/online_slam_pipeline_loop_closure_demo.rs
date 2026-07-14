//! Online loop-closure + pose-graph refinement INSIDE `OnlineSlamPipeline`.
//!
//! Mirrors the synthetic 6-keyframe orbit from
//! `online_slam_pose_graph_loop_demo.rs` but drives the new
//! `OnlineSlamConfig::pose_graph_refinement` API: the pipeline owns the
//! verifier, the running `PoseGraph`, and the `optimize_se3_iterative`
//! trigger — the caller only inspects
//! `OnlineSlamResult::pose_graph_refinement` for per-frame stats.
//!
//! Compared with the manual flow demo, this is the path real online SLAM
//! integrations should use: no per-call boilerplate to wire verifier /
//! constraint extraction / graph construction / PGO trigger.

use std::env;
use std::path::PathBuf;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    LocalMappingPipeline, LocalizationPipeline, LoopClosureConfig, LoopClosureVerifierConfig,
    LoopRefinementSolver, LoopRefinementVerifier, OnlineSlamConfig,
    OnlineSlamLoopClosureRefinementConfig, OnlineSlamPipeline, PoseGraphSe3Config, Tracker,
    TrackingConfig,
};

const KEYFRAME_IDS: [u64; 6] = [10, 20, 30, 40, 50, 60];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    let _out_dir = parse_output_dir(&mut args);
    if !args.is_empty() {
        eprintln!(
            "usage: cargo run --example online_slam_pipeline_loop_closure_demo -- [--out-dir <dir>]"
        );
        std::process::exit(2);
    }

    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let landmark_points = synthetic_world_points();
    let map = build_map(&camera, &landmark_points);

    // Six-keyframe orbit: KF#10 at origin, KF#20–50 outbound, KF#60 returns
    // near the origin so the shared-landmark gate fires on the final frame.
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

    // The same `default_translation_scale` calibration that the manual
    // demo uses for the KF#10 ↔ KF#60 pair.
    let loop_translation_scale = 0.1_f64.hypot(0.05);

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
            pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                camera: camera.clone(),
                verifier_config: LoopClosureVerifierConfig {
                    min_inliers: 8,
                    min_inlier_ratio: 0.5,
                    max_mean_sampson_error: 5.0e-3,
                    default_translation_scale: loop_translation_scale,
                },
                verifier: LoopRefinementVerifier::EssentialMatrix,
                pose_graph_config: PoseGraphSe3Config::default(),
                fixed_loop_edge_weight: None,
                loop_pose_information: None,
                gnc: None,
                pcm: None,
                covariance_gate: None,
                pcm_batch_rescreen: false,
                marginalization_window: None,
                marginalization_sparsify: false,
                trigger_every_new_constraints: 1,
                appearance_candidates: None,
                propagate_corrections: false,
                solver: LoopRefinementSolver::Se3,
            }),
            ..OnlineSlamConfig::default()
        },
    );

    for frame in &frames {
        let result = slam.process_frame(frame, []);
        let stats_summary = match result.pose_graph_refinement.as_ref() {
            Some(stats) => format!(
                "candidates={} accepted={} pgo_fired={} keyframes_updated={}",
                stats.verified_candidate_count,
                stats.accepted_count,
                stats.pose_graph_result.is_some(),
                stats.keyframes_updated,
            ),
            None => "stage_skipped".to_string(),
        };
        println!(
            "frame={} tracking={} keyframes={} loop_candidates={} {}",
            frame.id,
            result.tracking_succeeded(),
            result.map_keyframe_count,
            result.loop_closure_candidates.len(),
            stats_summary,
        );
    }

    let state = slam
        .pose_graph_state
        .as_ref()
        .expect("pose-graph state present when refinement is enabled");
    println!(
        "final pose_graph nodes={} sequential_edges={} loop_edges={} verified_constraints={} pgo_triggers={}",
        state.graph.poses.len(),
        state
            .graph
            .edges
            .iter()
            .filter(|e| matches!(e.kind, visloc_rs::PoseGraphEdgeKind::Sequential))
            .count(),
        state
            .graph
            .edges
            .iter()
            .filter(|e| matches!(e.kind, visloc_rs::PoseGraphEdgeKind::LoopClosure))
            .count(),
        state.verified_constraints.len(),
        state.trigger_count,
    );
    Ok(())
}

fn parse_output_dir(args: &mut Vec<String>) -> Option<PathBuf> {
    let mut out_dir: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out-dir" => {
                if i + 1 >= args.len() {
                    eprintln!("--out-dir requires a path argument");
                    std::process::exit(2);
                }
                out_dir = Some(PathBuf::from(args.remove(i + 1)));
                args.remove(i);
            }
            _ => i += 1,
        }
    }
    out_dir
}

fn synthetic_world_points() -> Vec<Point3<f64>> {
    vec![
        Point3::new(-1.5, -1.0, 5.0),
        Point3::new(1.5, -1.0, 5.1),
        Point3::new(-1.5, 1.0, 4.9),
        Point3::new(1.5, 1.0, 5.0),
        Point3::new(0.0, 0.0, 5.05),
        Point3::new(0.6, -0.4, 4.95),
        Point3::new(-0.7, 0.5, 4.8),
        Point3::new(0.5, 0.8, 5.2),
        Point3::new(-0.4, -0.7, 4.85),
        Point3::new(0.8, -0.6, 5.3),
        Point3::new(0.0, 0.6, 5.4),
        Point3::new(-0.8, -0.3, 4.7),
        Point3::new(0.3, 0.3, 5.15),
        Point3::new(-0.2, -0.4, 5.25),
    ]
}

fn build_map(camera: &Camera, points: &[Point3<f64>]) -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    for (index, p) in points.iter().enumerate() {
        let landmark_id = (index + 1) as u64;
        let descriptor: Vec<f32> = (0..16)
            .map(|d| ((index as f32) * 0.17 + (d as f32) * 0.11).cos())
            .collect();
        let mut landmark = Landmark::new(landmark_id, *p);
        landmark.descriptor = Some(descriptor);
        map.landmarks.insert(landmark_id, landmark);
    }
    map
}

fn frame_from_pose(
    frame_id: u64,
    camera: &Camera,
    center: Vector3<f64>,
    points: &[Point3<f64>],
) -> Frame {
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -center);
    let mut frame = Frame::new(frame_id, camera.id);
    for (index, p) in points.iter().enumerate() {
        let landmark_id = (index + 1) as u64;
        let projected = camera.project(&pose.transform_world_point(p)).unwrap();
        let descriptor: Vec<f32> = (0..16)
            .map(|d| ((index as f32) * 0.17 + (d as f32) * 0.11).cos())
            .collect();
        frame.keypoints.push(projected);
        frame.descriptors.push(descriptor);
        let _ = landmark_id;
    }
    frame
}
