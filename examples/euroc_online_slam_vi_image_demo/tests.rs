use super::*;

#[test]
fn superpoint_onnx_backend_arg_parses_strict_cuda_without_fallback() {
    assert_eq!(
        SuperPointOnnxBackendArg::parse("cuda"),
        Ok(SuperPointOnnxBackendArg::Cuda)
    );
    assert!(SuperPointOnnxBackendArg::parse("gpu").is_err());
}

fn fake_cam0() -> EurocCameraCalibration {
    EurocCameraCalibration {
        t_body_sensor: Matrix4::identity(),
        rate_hz: 20.0,
        resolution: (752, 480),
        camera_model: "pinhole".to_string(),
        intrinsics: [458.0, 457.0, 367.0, 248.0],
        distortion_model: "radial-tangential".to_string(),
        distortion_coefficients: vec![0.0, 0.0, 0.0, 0.0],
    }
}

#[test]
fn se3_from_t_bs_identity_round_trip() {
    let se3 = se3_from_t_bs(&Matrix4::identity());
    assert!((se3.rotation.angle()).abs() < 1.0e-12);
    assert!(se3.translation.norm() < 1.0e-12);
}

#[test]
fn back_project_pixel_to_world_at_identity_camera_returns_camera_frame_point() {
    // World-to-camera = identity → camera centre at origin, looking
    // along +Z. The principal-point pixel back-projects to a world
    // point at (0, 0, depth).
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
    let world = back_project_pixel_to_world(&camera, &pose, &principal_pixel, 5.0).expect("valid");
    assert!((world - Point3::new(0.0, 0.0, 5.0)).norm() < 1.0e-9);
}

#[test]
fn back_project_then_project_round_trip_identity_camera() {
    // Back-projecting an arbitrary pixel and re-projecting through the
    // same camera must return the same pixel within floating-point
    // tolerance — the projection / unprojection are exact inverses
    // when the camera is identity-posed and the assumed depth is
    // honoured.
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let pixel = Point2::new(200.0, 150.0);
    let world = back_project_pixel_to_world(&camera, &pose, &pixel, 7.5).expect("valid");
    let p_cam = pose.transform_world_point(&world);
    let projected = camera.project(&p_cam).expect("in front of camera");
    assert!((projected.x - pixel.x).abs() < 1.0e-9);
    assert!((projected.y - pixel.y).abs() < 1.0e-9);
}

#[test]
fn bootstrap_map_seeds_one_landmark_per_keypoint() {
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let features = FeatureSet::new(
        vec![Point2::new(300.0, 200.0), Point2::new(450.0, 250.0)],
        vec![vec![0.1, 0.2, 0.3], vec![-0.1, 0.0, 0.5]],
    )
    .expect("valid feature set");
    let map = bootstrap_map_from_first_frame(&camera, &pose, &features, 4.0, &[], &[], false);
    assert_eq!(map.landmarks.len(), 2);
    // Each seeded landmark must carry the source corner descriptor.
    for landmark in map.landmarks.values() {
        assert!(landmark.descriptor.is_some());
        assert_eq!(landmark.descriptor.as_ref().unwrap().len(), 3);
    }
}

#[test]
fn frame_from_features_preserves_keypoint_and_descriptor_order() {
    let features = FeatureSet::new(
        vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .expect("valid feature set");
    let frame = frame_from_features(42, 1, &features);
    assert_eq!(frame.id, 42);
    assert_eq!(frame.camera_id, 1);
    assert_eq!(frame.keypoints, features.keypoints);
    assert_eq!(frame.descriptors, features.descriptors);
}

#[test]
fn atlas_boundary_snapshot_preserves_unpromoted_tracking_overlap() {
    let mut map = VisualMap::new();
    map.landmarks
        .insert(7, Landmark::new(7, Point3::new(0.0, 0.0, 4.0)));
    let mut frame = Frame::new(42, 1);
    frame.keypoints.push(Point2::new(320.0, 240.0));
    frame.descriptors.push(vec![1.0, 0.0]);
    frame.pose = Some(Pose::from_world_to_camera(
        UnitQuaternion::identity(),
        Vector3::zeros(),
    ));
    let snapshot = Keyframe {
        frame,
        observations: vec![Observation {
            frame_id: 42,
            landmark_id: 7,
            keypoint_index: 0,
            xy: Point2::new(320.0, 240.0),
        }],
    };

    assert!(insert_atlas_boundary_keyframe_snapshot(&mut map, &snapshot));
    assert!(map.keyframes.contains_key(&42));
    assert!(map.landmarks[&7]
        .observations
        .iter()
        .any(|observation| observation.frame_id == 42));
    assert!(!insert_atlas_boundary_keyframe_snapshot(
        &mut map, &snapshot
    ));
}

#[test]
fn normalized_mean_descriptor_averages_and_l2_normalizes() {
    let mean = normalized_mean_descriptor(&[vec![2.0, 0.0], vec![0.0, 2.0]])
        .expect("non-empty descriptor set");

    assert!((mean[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
    assert!((mean[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
    assert_eq!(normalized_mean_descriptor(&[]), None);
}

#[test]
fn relocalization_attempt_reject_reason_reports_first_failed_gate() {
    let config = RelocalizationAttemptGateConfig {
        min_inliers: 20,
        min_inlier_ratio: 0.5,
        max_reprojection_error: Some(8.0),
        max_translation_per_frame_from_last_success_meters: None,
        min_inlier_depth_median_ratio_to_last_success: None,
        max_inlier_depth_median_ratio_to_last_success: None,
    };
    let stats = OnlineSlamRelocalizationStats {
        attempted: true,
        localization_success: true,
        inlier_count: 30,
        inlier_ratio: 0.8,
        mean_reprojection_error: Some(9.0),
        ..Default::default()
    };

    let gates = relocalization_attempt_gate_status(&stats, config);

    assert!(!gates.reprojection_pass);
    assert_eq!(
        relocalization_attempt_reject_reason(&stats, gates),
        "max_reprojection_error"
    );
}

#[test]
fn relocalization_attempt_reject_reason_reports_confirmation_waiting() {
    let config = RelocalizationAttemptGateConfig {
        min_inliers: 20,
        min_inlier_ratio: 0.5,
        max_reprojection_error: Some(8.0),
        max_translation_per_frame_from_last_success_meters: None,
        min_inlier_depth_median_ratio_to_last_success: None,
        max_inlier_depth_median_ratio_to_last_success: None,
    };
    let stats = OnlineSlamRelocalizationStats {
        attempted: true,
        localization_success: true,
        inlier_count: 30,
        inlier_ratio: 0.8,
        mean_reprojection_error: Some(2.0),
        passed_acceptance_gates: true,
        confirmation_count: 1,
        confirmation_required_count: 2,
        ..Default::default()
    };

    let gates = relocalization_attempt_gate_status(&stats, config);

    assert_eq!(
        relocalization_attempt_reject_reason(&stats, gates),
        "confirmation_waiting"
    );
}

#[test]
fn undistort_feature_keypoints_is_noop_under_identity_distortion() {
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let features = FeatureSet::new(
        vec![Point2::new(100.0, 80.0), Point2::new(450.0, 320.0)],
        vec![vec![0.1, 0.2], vec![0.3, 0.4]],
    )
    .expect("valid feature set");
    let undistorted = undistort_feature_keypoints(&RadialTangential::IDENTITY, &camera, &features);
    assert_eq!(undistorted.keypoints, features.keypoints);
    assert_eq!(undistorted.descriptors, features.descriptors);
}

#[test]
fn bootstrap_map_overrides_landmark_position_when_stereo_point_provided() {
    // When `stereo_world_points[i]` is `Some(...)`, that 3D point must
    // win over the fixed-depth back-projection for keypoint `i`. The
    // other keypoint must still fall back to the depth-based seeding.
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
    let other_pixel = Point2::new(450.0, 250.0);
    let features = FeatureSet::new(
        vec![principal_pixel, other_pixel],
        vec![vec![0.1, 0.2], vec![-0.1, 0.0]],
    )
    .expect("valid feature set");
    // Override the first keypoint with an explicit world point.
    let override_point = Point3::new(7.0, -1.5, 9.0);
    let stereo_overrides = vec![Some(override_point), None];
    let covariance = nalgebra::Matrix3::from_diagonal_element(0.25);
    let stereo_covariances = vec![Some(covariance), None];
    let map = bootstrap_map_from_first_frame(
        &camera,
        &pose,
        &features,
        4.0,
        &stereo_overrides,
        &stereo_covariances,
        false,
    );
    assert_eq!(map.landmarks.len(), 2);
    // Landmark id = index + 1.
    let first = map.landmarks.get(&1).expect("override landmark seeded");
    assert!((first.position - override_point).norm() < 1.0e-9);
    assert_eq!(map.landmark_position_covariances.get(&1), Some(&covariance));
    let second = map.landmarks.get(&2).expect("fallback landmark seeded");
    // Fallback path: back-project the other pixel at depth 4.0 m and
    // confirm the seeded landmark sits there.
    let expected_fallback = back_project_pixel_to_world(&camera, &pose, &other_pixel, 4.0).unwrap();
    assert!((second.position - expected_fallback).norm() < 1.0e-9);
}

#[test]
fn bootstrap_map_strict_stereo_drops_keypoints_without_triangulated_depth() {
    // Phase-23 #2: with `strict_stereo = true`, every keypoint that
    // has no corresponding stereo-triangulated 3D point is dropped
    // from the map instead of falling back to the fixed-depth
    // back-projection. On a 2-keypoint frame with one stereo
    // override and one without, the strict map must hold exactly
    // one landmark — the overridden one.
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let principal_pixel = Point2::new(camera.params[2], camera.params[3]);
    let other_pixel = Point2::new(450.0, 250.0);
    let features = FeatureSet::new(
        vec![principal_pixel, other_pixel],
        vec![vec![0.1, 0.2], vec![-0.1, 0.0]],
    )
    .expect("valid feature set");
    let override_point = Point3::new(7.0, -1.5, 9.0);
    let stereo_overrides = vec![Some(override_point), None];
    let stereo_covariances = vec![Some(nalgebra::Matrix3::identity()), None];
    let strict_map = bootstrap_map_from_first_frame(
        &camera,
        &pose,
        &features,
        4.0,
        &stereo_overrides,
        &stereo_covariances,
        true,
    );
    assert_eq!(strict_map.landmarks.len(), 1);
    let kept = strict_map
        .landmarks
        .get(&1)
        .expect("strict mode keeps the overridden landmark");
    assert!((kept.position - override_point).norm() < 1.0e-9);
    assert!(strict_map.landmarks.get(&2).is_none());
}

#[test]
fn undistort_feature_keypoints_shifts_edge_pixels_under_euroc_distortion() {
    let camera = camera_from_cam0(&fake_cam0(), 1);
    // EuRoC MH_01_easy cam0 published coefficients.
    let distortion = RadialTangential {
        k1: -0.28340811,
        k2: 0.07395907,
        p1: 0.00019359,
        p2: 0.0000176187114,
    };
    let principal = Point2::new(camera.params[2], camera.params[3]);
    let edge = Point2::new(10.0, 10.0);
    let features = FeatureSet::new(vec![principal, edge], vec![vec![0.0], vec![1.0]])
        .expect("valid feature set");
    let undistorted = undistort_feature_keypoints(&distortion, &camera, &features);
    assert_eq!(undistorted.keypoints.len(), 2);
    assert_eq!(undistorted.descriptors, features.descriptors);
    // Principal-point pixel barely moves; corner pixel shifts by
    // several pixels under the EuRoC distortion magnitude.
    let principal_shift = (undistorted.keypoints[0].coords - principal.coords).norm();
    let edge_shift = (undistorted.keypoints[1].coords - edge.coords).norm();
    assert!(
        principal_shift < 0.01,
        "principal shift = {principal_shift}"
    );
    assert!(edge_shift > 5.0, "edge shift = {edge_shift}");
}

#[test]
fn stereo_segment_restart_tracking_result_marks_all_stereo_matches_inliers() {
    let camera = camera_from_cam0(&fake_cam0(), 1);
    let pose = Pose::identity();
    let features = FeatureSet::new(
        vec![Point2::new(100.0, 120.0), Point2::new(200.0, 220.0)],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    )
    .expect("valid feature set");
    let matches = vec![
        StereoBootstrapLandmark {
            left_keypoint_index: 0,
            right_keypoint_index: 0,
            point_left_camera_frame: Point3::new(0.1, 0.2, 2.0),
            left_reprojection_error_pixels: 0.5,
            right_reprojection_error_pixels: 0.5,
            point_covariance_left_camera_frame: nalgebra::Matrix3::identity(),
        },
        StereoBootstrapLandmark {
            left_keypoint_index: 1,
            right_keypoint_index: 1,
            point_left_camera_frame: Point3::new(-0.2, 0.1, 3.0),
            left_reprojection_error_pixels: 0.4,
            right_reprojection_error_pixels: 0.4,
            point_covariance_left_camera_frame: nalgebra::Matrix3::identity(),
        },
    ];
    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera);
    let mut next_id = 100_u64;
    let landmark_ids = append_stereo_bootstrap_landmarks_to_map(
        &mut map,
        &pose,
        &features,
        &matches,
        &mut next_id,
    );
    let stats = map_provider_stats_from_map(&map);
    let tracking =
        build_stereo_segment_restart_tracking_result(42, pose, &matches, &landmark_ids, stats);
    assert!(tracking.localization.success);
    assert_eq!(tracking.localization.inlier_count, 2);
    assert_eq!(tracking.localization.inlier_landmark_ids, landmark_ids);
}
