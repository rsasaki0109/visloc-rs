use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::LocalMappingPipeline;
use visloc_slam::{
    correspondences_2d3d_for_loop_candidate, loop_closure_constraints_from_candidates,
    online_slam_results_to_html_report, relative_world_to_camera, verify_loop_closure_candidates,
    verify_loop_closure_candidates_pnp, EssentialMatrixLoopClosureVerifier, LoopClosureConfig,
    LoopClosureConstraint, LoopClosureVerifierConfig, OnlineSlamConfig, OnlineSlamPipeline,
    PnPLoopClosureVerifier, PnPLoopClosureVerifierConfig, PoseGraph, PoseGraphError,
    PoseGraphSe3Config, RobustKernel,
};
use visloc_tracking::{Tracker, TrackingConfig};

fn map_and_frame(frame_id: u64, camera_id: u64) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
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

fn slam_pipeline(
    map: VisualMap,
    apply_map_updates: bool,
) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
    OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
        },
    )
}

#[test]
fn online_slam_tracks_and_applies_keyframe_update() {
    let (map, frame) = map_and_frame(10, 1);
    let mut slam = slam_pipeline(map, true);

    let result = slam.process_frame(&frame, []);

    assert!(result.tracking_succeeded());
    assert!(result.map_was_updated());
    assert_eq!(result.map_keyframe_count, 1);
    assert_eq!(result.map_landmark_count, 6);
    assert!(result.mapping.as_ref().unwrap().keyframe_decision.selected);
    assert!(!result.has_loop_closure_candidate());
    assert_eq!(slam.map().keyframes.len(), 1);
    assert!(slam.map().validate().is_valid());
}

#[test]
fn online_slam_can_return_staged_update_without_applying_it() {
    let (map, frame) = map_and_frame(10, 1);
    let mut slam = slam_pipeline(map, false);

    let result = slam.process_frame(&frame, []);

    assert!(result.tracking_succeeded());
    assert!(!result.map_was_updated());
    assert_eq!(result.map_keyframe_count, 0);
    assert_eq!(slam.map().keyframes.len(), 0);
    assert_eq!(result.mapping.unwrap().staged_update.keyframes.len(), 1);
}

#[test]
fn online_slam_skips_mapping_when_tracking_fails() {
    let (map, mut frame) = map_and_frame(10, 1);
    frame.camera_id = 999;
    let mut slam = slam_pipeline(map, true);

    let result = slam.process_frame(&frame, []);

    assert!(!result.tracking_succeeded());
    assert!(result.mapping.is_none());
    assert!(!result.map_was_updated());
    assert_eq!(slam.map().keyframes.len(), 0);
}

#[test]
fn online_slam_reset_clears_sequence_state_but_keeps_map() {
    let (map, frame) = map_and_frame(10, 1);
    let mut slam = slam_pipeline(map, true);

    let result = slam.process_frame(&frame, []);
    assert!(result.tracking_succeeded());
    assert_eq!(slam.map().keyframes.len(), 1);

    slam.reset_sequence_state();

    assert_eq!(slam.map().keyframes.len(), 1);
    assert_eq!(slam.tracker.stats().frame_count, 0);
}

#[test]
fn online_slam_reports_loop_closure_candidate_against_older_keyframe() {
    let (map, first_frame) = map_and_frame(10, 1);
    let (_, second_frame) = map_and_frame(30, 1);
    let mut slam = slam_pipeline(map, true);

    let first = slam.process_frame(&first_frame, []);
    assert!(first.tracking_succeeded());
    assert!(!first.has_loop_closure_candidate());

    let second = slam.process_frame(&second_frame, []);

    assert!(second.tracking_succeeded());
    assert!(second.has_loop_closure_candidate());
    assert_eq!(second.loop_closure_candidates.len(), 1);
    let candidate = &second.loop_closure_candidates[0];
    assert_eq!(candidate.query_frame_id, 30);
    assert_eq!(candidate.matched_keyframe_id, 10);
    assert_eq!(candidate.shared_landmark_count, 6);
    assert_eq!(candidate.query_inlier_count, 6);
    assert_eq!(candidate.keyframe_observation_count, 6);
    assert!((candidate.shared_landmark_ratio - 1.0).abs() < 1.0e-9);
    assert!(candidate.geometrically_verified);
}

#[test]
fn online_slam_html_report_renders_loop_candidate_edge() {
    let (map, first_frame) = map_and_frame(10, 1);
    let (_, second_frame) = map_and_frame(30, 1);
    let mut slam = slam_pipeline(map, true);

    let first = slam.process_frame(&first_frame, []);
    let second = slam.process_frame(&second_frame, []);

    let html = online_slam_results_to_html_report(&[first, second]);

    assert!(html.contains("online SLAM loop report"));
    assert!(html.contains("Loop Closure Candidates"));
    assert!(html.contains("loop candidate edge"));
    assert!(html.contains("<td>30</td><td>10</td><td>6</td>"));
}

fn map_and_frame_with_extra_landmarks(
    frame_id: u64,
    camera_id: u64,
    camera_center: Vector3<f64>,
) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
    let points = [
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

fn slam_pipeline_for_verifier(
    map: VisualMap,
) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
    OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 8,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
        },
    )
}

#[test]
fn essential_matrix_loop_closure_verifier_marks_consistent_candidate_as_verified() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert!(second.tracking_succeeded());
    assert_eq!(second.loop_closure_candidates.len(), 1);

    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    let candidate = &second.loop_closure_candidates[0];
    let verification = candidate
        .verification
        .as_ref()
        .expect("verifier was run and must produce output");
    assert!(verification.verified);
    assert!(candidate.geometrically_verified);
    assert_eq!(verification.correspondence_count, 12);
    assert_eq!(verification.inlier_count, 12);
    assert!((verification.inlier_ratio - 1.0).abs() < 1.0e-9);
    assert!(verification.mean_sampson_error < 5.0e-3);
    assert!(verification.failure_reason.is_none());
}

#[test]
fn essential_matrix_loop_closure_verifier_rejects_loose_threshold_candidate() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    // Demand more inliers than the pair of frames can ever produce so the
    // verifier rejects the candidate up front through the
    // `InsufficientCorrespondences` failure path.
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 64,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    let candidate = &second.loop_closure_candidates[0];
    let verification = candidate
        .verification
        .as_ref()
        .expect("verifier was run and must produce output");
    assert!(!verification.verified);
    assert!(!candidate.geometrically_verified);
    assert_eq!(
        verification.failure_reason,
        Some(visloc_slam::LoopClosureVerificationFailureReason::InsufficientCorrespondences)
    );
}

#[test]
fn loop_closure_constraint_from_verified_candidate_carries_relative_pose() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    let constraints = loop_closure_constraints_from_candidates(&second.loop_closure_candidates);
    assert_eq!(constraints.len(), 1);
    let constraint = &constraints[0];
    assert_eq!(constraint.from_keyframe_id, 10);
    assert_eq!(constraint.to_keyframe_id, 30);
    assert_eq!(constraint.inlier_count, 12);
    assert!((constraint.inlier_ratio - 1.0).abs() < 1.0e-9);
    assert!(constraint.mean_sampson_error < 5.0e-3);
    let from_candidate =
        LoopClosureConstraint::from_verified_candidate(&second.loop_closure_candidates[0]).unwrap();
    assert_eq!(from_candidate, *constraint);
}

#[test]
fn loop_closure_constraint_skips_unverified_candidate() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    // Demand more inliers than available; the verifier rejects up front.
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 64,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    assert!(loop_closure_constraints_from_candidates(&second.loop_closure_candidates).is_empty());
    assert!(
        LoopClosureConstraint::from_verified_candidate(&second.loop_closure_candidates[0])
            .is_none()
    );
}

#[test]
fn online_slam_respects_loop_closure_frame_gap() {
    let (map, first_frame) = map_and_frame(10, 1);
    let (_, second_frame) = map_and_frame(12, 1);
    let mut slam = slam_pipeline(map, true);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let second = slam.process_frame(&second_frame, []);

    assert!(second.tracking_succeeded());
    assert!(!second.has_loop_closure_candidate());
}

fn pose_at(camera_center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center)
}

#[test]
fn pose_graph_translation_gauss_newton_pulls_drifted_loop_back_to_anchor() {
    // Three keyframes laid out as a small loop. Truth camera centers:
    //   10 -> (0.0, 0.0, 0.0)  (anchor)
    //   20 -> (1.4, 0.0, 0.4)
    //   30 -> (0.2, 0.0, 0.1)
    // Sequential edges 10->20 and 20->30 carry truth measurements; the loop
    // closure 10->30 is also at truth. The graph node for 30 is initialized
    // with drift, and the single Gauss-Newton step must pull it back.
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(1.4, 0.0, 0.4));
    let truth_30 = pose_at(Vector3::new(0.2, 0.0, 0.1));

    let edge_10_20 = relative_world_to_camera(&truth_10, &truth_20);
    let edge_20_30 = relative_world_to_camera(&truth_20, &truth_30);
    let edge_10_30 = relative_world_to_camera(&truth_10, &truth_30);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    // Initialize node 30 with a deliberate translation drift.
    let drifted_30 = pose_at(Vector3::new(0.45, 0.10, 0.30));
    graph.add_pose(30, drifted_30);
    graph.anchor(10);

    graph.add_sequential_edge(10, 20, edge_10_20);
    graph.add_sequential_edge(20, 30, edge_20_30);
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 30,
        relative_pose: edge_10_30,
        inlier_count: 12,
        inlier_ratio: 1.0,
        mean_sampson_error: 1.0e-4,
        score: 100.0,
    });

    let cost_before = graph.translation_cost();
    assert!(
        cost_before > 0.05,
        "expected nontrivial drift cost, got {cost_before}"
    );

    let step = graph
        .optimize_translations_once()
        .expect("solve must succeed");
    assert_eq!(step.anchor_id, 10);
    assert_eq!(step.edge_count, 3);
    assert_eq!(step.variable_count, 2);
    assert!(step.cost_after < 1.0e-9);
    assert!(step.cost_after < step.cost_before);

    let center_30 = graph.poses[&30].camera_center_world();
    assert!(
        (center_30 - Point3::new(0.2, 0.0, 0.1)).norm() < 1.0e-9,
        "node 30 should snap back to truth: {center_30:?}"
    );
    let center_20 = graph.poses[&20].camera_center_world();
    assert!((center_20 - Point3::new(1.4, 0.0, 0.4)).norm() < 1.0e-9);
}

#[test]
fn pose_graph_optimize_returns_no_anchor_error_when_unset() {
    let mut graph = PoseGraph::new();
    graph.add_pose(1, pose_at(Vector3::zeros()));
    graph.add_pose(2, pose_at(Vector3::new(1.0, 0.0, 0.0)));
    graph.add_sequential_edge(
        1,
        2,
        relative_world_to_camera(
            &pose_at(Vector3::zeros()),
            &pose_at(Vector3::new(1.0, 0.0, 0.0)),
        ),
    );

    assert_eq!(
        graph.optimize_translations_once(),
        Err(PoseGraphError::NoAnchor)
    );
}

#[test]
fn pose_graph_optimize_returns_no_edges_error_when_empty() {
    let mut graph = PoseGraph::new();
    graph.add_pose(1, pose_at(Vector3::zeros()));
    graph.anchor(1);

    assert_eq!(
        graph.optimize_translations_once(),
        Err(PoseGraphError::NoEdges)
    );
}

#[test]
fn pose_graph_optimize_returns_no_variables_error_when_only_anchor_present() {
    let mut graph = PoseGraph::new();
    graph.add_pose(1, pose_at(Vector3::zeros()));
    graph.anchor(1);
    graph.add_sequential_edge(
        1,
        1,
        relative_world_to_camera(&pose_at(Vector3::zeros()), &pose_at(Vector3::zeros())),
    );

    assert_eq!(
        graph.optimize_translations_once(),
        Err(PoseGraphError::NoVariables)
    );
}

fn pose_with_yaw(camera_center: Vector3<f64>, yaw_rad: f64) -> Pose {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_rad);
    let translation = -(rotation.transform_vector(&camera_center));
    Pose::from_world_to_camera(rotation, translation)
}

#[test]
fn pose_graph_se3_gauss_newton_corrects_rotation_and_translation_drift() {
    // Three keyframes with non-trivial rotations. Truth poses come from a small
    // 2D loop traversal where each keyframe yaws progressively. Sequential and
    // loop-closure edges encode the truth measurements; node 30 is initialized
    // with both translation drift AND rotation drift, and the iterative SE3 GN
    // solver must pull it back.
    let truth_10 = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let truth_20 = pose_with_yaw(Vector3::new(1.5, 0.0, 0.5), 0.4);
    let truth_30 = pose_with_yaw(Vector3::new(0.3, 0.0, 0.2), 0.8);

    let edge_10_20 = relative_world_to_camera(&truth_10, &truth_20);
    let edge_20_30 = relative_world_to_camera(&truth_20, &truth_30);
    let edge_10_30 = relative_world_to_camera(&truth_10, &truth_30);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    let drifted_30 = pose_with_yaw(Vector3::new(0.6, 0.05, 0.4), 0.55);
    graph.add_pose(30, drifted_30);
    graph.anchor(10);

    graph.add_sequential_edge(10, 20, edge_10_20);
    graph.add_sequential_edge(20, 30, edge_20_30);
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 30,
        relative_pose: edge_10_30,
        inlier_count: 12,
        inlier_ratio: 1.0,
        mean_sampson_error: 1.0e-4,
        score: 100.0,
    });

    let cost_before = graph.se3_cost();
    assert!(
        cost_before > 1.0e-3,
        "expected nontrivial drift cost, got {cost_before}"
    );

    let result = graph
        .optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("solve must succeed");
    assert_eq!(result.anchor_id, 10);
    assert_eq!(result.edge_count, 3);
    assert_eq!(result.variable_count, 2);
    assert!(result.converged, "GN should converge: {:?}", result);
    assert!(
        result.final_cost < 1.0e-12,
        "final cost too large: {}",
        result.final_cost
    );
    assert!(result.final_cost < result.initial_cost);

    let pose_30 = &graph.poses[&30];
    let center_30 = pose_30.camera_center_world();
    assert!(
        (center_30 - Point3::new(0.3, 0.0, 0.2)).norm() < 1.0e-9,
        "node 30 center should snap to truth: {center_30:?}"
    );
    let r_truth = truth_30
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let r_now = pose_30
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    assert!(
        (r_truth - r_now).norm() < 1.0e-9,
        "node 30 rotation should snap to truth"
    );
}

#[test]
fn pose_graph_se3_converges_immediately_without_drift() {
    let truth_10 = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let truth_20 = pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.3);
    let edge = relative_world_to_camera(&truth_10, &truth_20);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10);
    graph.add_pose(20, truth_20);
    graph.anchor(10);
    graph.add_sequential_edge(10, 20, edge);

    let result = graph
        .optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("solve must succeed");
    assert!(result.converged);
    assert!(result.initial_cost < 1.0e-15);
    assert!(result.final_cost < 1.0e-15);
}

#[test]
fn pose_graph_se3_returns_no_anchor_error_when_unset() {
    let mut graph = PoseGraph::new();
    graph.add_pose(1, pose_at(Vector3::zeros()));
    graph.add_pose(2, pose_at(Vector3::new(1.0, 0.0, 0.0)));
    graph.add_sequential_edge(
        1,
        2,
        relative_world_to_camera(
            &pose_at(Vector3::zeros()),
            &pose_at(Vector3::new(1.0, 0.0, 0.0)),
        ),
    );

    assert_eq!(
        graph.optimize_se3_iterative(&PoseGraphSe3Config::default()),
        Err(PoseGraphError::NoAnchor)
    );
}

fn build_three_node_loop(third_drift: Vector3<f64>) -> PoseGraph {
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(0.5, 0.0, 0.5));

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    let drifted_30 = pose_at(Vector3::new(0.5, 0.0, 0.5) + third_drift);
    graph.add_pose(30, drifted_30);
    graph.anchor(10);
    graph.add_sequential_edge(10, 20, relative_world_to_camera(&truth_10, &truth_20));
    graph.add_sequential_edge(20, 30, relative_world_to_camera(&truth_20, &truth_30));
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 30,
        relative_pose: relative_world_to_camera(&truth_10, &truth_30),
        inlier_count: 12,
        inlier_ratio: 1.0,
        mean_sampson_error: 1.0e-4,
        score: 100.0,
    });
    graph
}

#[test]
fn pose_graph_se3_huber_kernel_rejects_outlier_loop_constraint() {
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(0.5, 0.0, 0.5));

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    graph.add_pose(30, truth_30.clone());
    graph.anchor(10);
    graph.add_sequential_edge(10, 20, relative_world_to_camera(&truth_10, &truth_20));
    graph.add_sequential_edge(20, 30, relative_world_to_camera(&truth_20, &truth_30));

    // Truth-consistent loop constraint with high weight.
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 30,
        relative_pose: relative_world_to_camera(&truth_10, &truth_30),
        inlier_count: 24,
        inlier_ratio: 1.0,
        mean_sampson_error: 1.0e-4,
        score: 240.0,
    });
    // Outlier loop constraint with a wildly wrong relative pose.
    let bogus_pose = pose_at(Vector3::new(5.0, 0.0, 5.0));
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 20,
        to_keyframe_id: 30,
        relative_pose: relative_world_to_camera(&truth_20, &bogus_pose),
        inlier_count: 8,
        inlier_ratio: 0.6,
        mean_sampson_error: 1.0e-2,
        score: 30.0,
    });

    let mut graph_no_kernel = graph.clone();
    let _ = graph_no_kernel
        .optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("pure GN must succeed");
    let center_30_no_kernel = graph_no_kernel.poses[&30].camera_center_world();
    // Without a robust kernel, the outlier drags KF30 away from the truth.
    let unbounded_drift = (center_30_no_kernel - Point3::new(0.5, 0.0, 0.5)).norm();
    assert!(
        unbounded_drift > 0.1,
        "outlier should measurably bias KF30 without robust kernel; got drift {unbounded_drift}"
    );

    let mut graph_huber = graph.clone();
    let result = graph_huber
        .optimize_se3_iterative(&PoseGraphSe3Config {
            robust_kernel: RobustKernel::Huber { delta: 0.05 },
            initial_lambda: Some(1e-4),
            max_iterations: 50,
            ..PoseGraphSe3Config::default()
        })
        .expect("LM + Huber must succeed");
    assert!(result.iterations.iter().any(|s| s.lambda > 0.0));
    let center_30_huber = graph_huber.poses[&30].camera_center_world();
    let huber_drift = (center_30_huber - Point3::new(0.5, 0.0, 0.5)).norm();
    assert!(
        huber_drift < unbounded_drift * 0.5,
        "Huber kernel should suppress outlier; huber_drift={huber_drift} unbounded={unbounded_drift}"
    );
}

#[test]
fn pose_graph_se3_lm_records_lambda_trajectory() {
    let mut graph = build_three_node_loop(Vector3::new(0.4, 0.05, -0.3));
    let result = graph
        .optimize_se3_iterative(&PoseGraphSe3Config {
            initial_lambda: Some(1e-2),
            max_iterations: 30,
            ..PoseGraphSe3Config::default()
        })
        .expect("LM must succeed");
    assert!(result.converged);
    assert!(result.final_cost < 1.0e-9);
    // LM must have run with positive λ for at least one iteration.
    assert!(result.iterations.iter().any(|s| s.lambda > 0.0));
    // After at least one accepted step, λ should have been adjusted (decreased
    // on accept, increased on reject) so the trajectory is nonconstant.
    let unique_lambdas: std::collections::BTreeSet<u64> = result
        .iterations
        .iter()
        .map(|s| s.lambda.to_bits())
        .collect();
    assert!(
        unique_lambdas.len() >= 2,
        "λ should change across iterations"
    );
}

#[test]
fn pose_graph_se3_robust_kernel_cost_matches_se3_cost_when_none() {
    let graph = build_three_node_loop(Vector3::new(0.1, 0.0, 0.0));
    let kernel_none = graph.robust_se3_cost(&RobustKernel::None);
    let plain = graph.se3_cost();
    assert!((kernel_none - plain).abs() < 1.0e-12);
}

#[test]
fn robust_kernel_huber_below_threshold_matches_quadratic() {
    let kernel = RobustKernel::Huber { delta: 1.0 };
    // s = 0.25 < δ² = 1.0 → ρ(s) = s, weight = 1.
    assert!((kernel.cost(0.25) - 0.25).abs() < 1.0e-12);
    assert!((kernel.weight(0.25) - 1.0).abs() < 1.0e-12);
    // s = 4.0 > δ² = 1.0 → ρ(s) = 2δ√s − δ² = 4 − 1 = 3, weight = δ/√s = 0.5.
    assert!((kernel.cost(4.0) - 3.0).abs() < 1.0e-12);
    assert!((kernel.weight(4.0) - 0.5).abs() < 1.0e-12);
}

#[test]
fn robust_kernel_cauchy_saturates_at_high_residual() {
    let kernel = RobustKernel::Cauchy { c: 1.0 };
    // weight strictly decreases as residual grows.
    let w_small = kernel.weight(0.0);
    let w_med = kernel.weight(1.0);
    let w_large = kernel.weight(100.0);
    assert!((w_small - 1.0).abs() < 1.0e-12);
    assert!(w_med < w_small);
    assert!(w_large < w_med);
    assert!(w_large > 0.0);
}

#[test]
fn pnp_loop_closure_verifier_marks_consistent_candidate_as_verified() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert!(second.tracking_succeeded());
    assert_eq!(second.loop_closure_candidates.len(), 1);

    let verifier: PnPLoopClosureVerifier = PnPLoopClosureVerifier {
        config: PnPLoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_reprojection_error_px: 4.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates_pnp(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    let candidate = &second.loop_closure_candidates[0];
    let verification = candidate
        .verification
        .as_ref()
        .expect("verifier output must be populated");
    assert!(verification.verified, "verification: {:?}", verification);
    assert!(candidate.geometrically_verified);
    assert!(verification.inlier_count >= 8);
    assert!(verification.mean_reprojection_error_px.is_some());
    assert!(verification.relative_pose.is_some());
    // Compare against the truth relative SE3 (10 → 30) computed from the same
    // poses the fixture generated.
    let truth_first = pose_at(Vector3::zeros());
    let truth_second = pose_at(Vector3::new(0.2, 0.0, 0.1));
    let expected = relative_world_to_camera(&truth_first, &truth_second);
    let relative = verification.relative_pose.as_ref().unwrap();
    let translation_err = (relative.translation - expected.translation).norm();
    let rotation_err = relative.rotation.rotation_to(&expected.rotation).angle();
    assert!(
        translation_err < 0.02,
        "PnP relative translation should match truth; got {:?} expected {:?} err={translation_err}",
        relative.translation,
        expected.translation
    );
    assert!(
        rotation_err < 0.02,
        "PnP relative rotation should match truth; rot_err={rotation_err}"
    );
}

#[test]
fn pnp_loop_closure_verifier_rejects_when_too_few_correspondences() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    // Demand more inliers than the test fixture can produce.
    let verifier: PnPLoopClosureVerifier = PnPLoopClosureVerifier {
        config: PnPLoopClosureVerifierConfig {
            min_inliers: 64,
            min_inlier_ratio: 0.6,
            max_mean_reprojection_error_px: 4.0,
        },
        ..Default::default()
    };
    verify_loop_closure_candidates_pnp(
        &mut second.loop_closure_candidates,
        &second_frame,
        &second.tracking,
        slam.map(),
        &camera,
        &verifier,
    );

    let candidate = &second.loop_closure_candidates[0];
    let verification = candidate
        .verification
        .as_ref()
        .expect("verifier output must be populated");
    assert!(!verification.verified);
    assert!(!candidate.geometrically_verified);
    assert_eq!(
        verification.failure_reason,
        Some(visloc_slam::LoopClosureVerificationFailureReason::InsufficientCorrespondences)
    );
    assert!(verification.relative_pose.is_none());
}

#[test]
fn correspondences_2d3d_only_includes_landmarks_observed_by_keyframe() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let mut slam = slam_pipeline_for_verifier(map);

    let _ = slam.process_frame(&first_frame, []);
    let second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);
    let candidate = &second.loop_closure_candidates[0];
    let keyframe = slam
        .map()
        .keyframes
        .get(&candidate.matched_keyframe_id)
        .unwrap();

    let correspondences = correspondences_2d3d_for_loop_candidate(
        &second_frame,
        &second.tracking.localization.inlier_query_indices,
        &second.tracking.localization.inlier_landmark_ids,
        keyframe,
        slam.map(),
    );
    assert_eq!(correspondences.len(), 12);
    // Each correspondence must have its 2D point inside the second frame's
    // keypoint list and its 3D point matching a map landmark.
    for c in &correspondences {
        assert!(second_frame.keypoints.contains(&c.point2d));
        assert!(slam
            .map()
            .landmarks
            .values()
            .any(|landmark| landmark.position == c.point3d));
    }
}
