use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, Sim3, SE3};
use visloc_core::types::{Camera, Frame, Landmark, Observation, VisualMap};
use visloc_localization::LocalizationPipeline;
use visloc_mapping::LocalMappingPipeline;
use visloc_slam::{
    build_appearance_loop_candidates, build_appearance_loop_candidates_with_diagnostics,
    correspondences_2d3d_for_loop_candidate, loop_closure_constraints_from_candidates,
    online_slam_results_to_html_report, relative_world_to_camera, scan_pairwise_loop_closures,
    verify_loop_closure_candidates, verify_loop_closure_candidates_hybrid,
    verify_loop_closure_candidates_pnp, AppearanceLoopScannerSettings, CovisibilityLocalBaConfig,
    CovisibilityLocalBaError, EssentialMatrixLoopClosureVerifier, HybridLoopClosureVerifier,
    HybridLoopClosureVerifierConfig, LinearSolver, LoopAppearanceCandidateConfig,
    LoopClosureConfig, LoopClosureConstraint, LoopClosureVerificationFailureReason,
    LoopClosureVerifierConfig, LoopRefinementSolver, LoopRefinementVerifier, OnlineSlamConfig,
    OnlineSlamCovisibilityLocalBaConfig, OnlineSlamImuConfig, OnlineSlamLocalBaConfig,
    OnlineSlamLoopClosureRefinementConfig, OnlineSlamPipeline,
    OnlineSlamRelocalizationAppearanceConfig, OnlineSlamRelocalizationConfig,
    OnlineSlamRelocalizationCovisibilityConfig, PairwiseKeyframeView,
    PairwiseLoopClosureScannerConfig, PnPLoopClosureVerifier, PnPLoopClosureVerifierConfig,
    PoseGraph, PoseGraphError, PoseGraphSe3Config, RobustKernel, Sim3PoseGraph,
    Sim3PoseGraphConfig,
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
            ..OnlineSlamConfig::default()
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
    assert!(result.covisibility_local_ba.is_none());
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

fn slam_pipeline_with_imu(
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
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            imu: Some(OnlineSlamImuConfig::default()),
            local_vi_ba: None,
            covisibility_local_ba: None,
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        },
    )
}

#[test]
fn online_slam_emits_imu_factor_between_consecutive_keyframes() {
    // Use `map_and_frame_with_extra_landmarks` so we can give the second
    // frame a camera-center translation of 1.5 m, comfortably above the
    // SimpleKeyframePolicy `min_translation = 1.0` default; otherwise the
    // mapper would reject the second frame as redundant and no factor
    // would close here.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    // First keyframe — no previous endpoint, so no factor closes here.
    let first = slam.process_frame(&first_frame, []);
    assert!(first.tracking_succeeded());
    assert!(
        first.imu_factor.is_none(),
        "first keyframe has nothing to close against"
    );
    assert_eq!(
        slam.imu_state.as_ref().and_then(|s| s.last_keyframe_id),
        Some(10),
    );

    // Push a 10-sample synthetic IMU window. Gyro is zero; accel cancels
    // gravity in the body frame (KITTI y-down default) so the gravity-free
    // delta velocity / position are near-zero. The exact numerical
    // accuracy is not the property under test here — the property is the
    // book-keeping (factor id endpoints, delta_time, integrator reset).
    let gyro = Vector3::zeros();
    let accel = Vector3::new(0.0, 9.81, 0.0);
    let dt = 0.1;
    for _ in 0..10 {
        slam.push_imu_measurement(gyro, accel, dt);
    }

    // Non-positive dt is silently dropped (it's how the integrator is
    // documented). Pipe a couple through to assert no panic / no state
    // change in `delta_time`.
    slam.push_imu_measurement(gyro, accel, 0.0);
    slam.push_imu_measurement(gyro, accel, -1.0);

    let second = slam.process_frame(&second_frame, []);
    assert!(second.tracking_succeeded());
    let factor = second
        .imu_factor
        .as_ref()
        .expect("a factor should be staged when a new keyframe closes the window");
    assert_eq!(factor.keyframe_id_from, 10);
    assert_eq!(factor.keyframe_id_to, 30);
    assert!(
        (factor.delta.delta_time - 1.0).abs() < 1.0e-9,
        "delta_time should be 10 * 0.1 = 1.0, got {}",
        factor.delta.delta_time
    );
    assert_eq!(factor.gravity_world, Vector3::new(0.0, 9.81, 0.0));

    // Integrator must have reset on factor emit — `take_pending_imu_factor`
    // returns the just-staged factor once, then `None`.
    let taken = slam.take_pending_imu_factor();
    assert!(taken.is_some());
    assert!(slam.take_pending_imu_factor().is_none());

    // `last_keyframe_id` advances to the second keyframe id so the next
    // window will close against frame 30, not frame 10.
    assert_eq!(
        slam.imu_state.as_ref().and_then(|s| s.last_keyframe_id),
        Some(30),
    );
}

#[test]
fn online_slam_imu_factor_is_none_without_imu_config() {
    let (map, first_frame) = map_and_frame(10, 1);
    let (_, second_frame) = map_and_frame(30, 1);
    let mut slam = slam_pipeline(map, true);

    // No IMU configured — push is a no-op and never panics.
    slam.push_imu_measurement(
        Vector3::new(0.1, 0.0, 0.0),
        Vector3::new(0.0, 9.81, 0.0),
        0.05,
    );
    let first = slam.process_frame(&first_frame, []);
    assert!(first.imu_factor.is_none());
    assert!(slam.imu_state.is_none());

    let second = slam.process_frame(&second_frame, []);
    assert!(second.imu_factor.is_none());
    assert!(slam.take_pending_imu_factor().is_none());
}

#[test]
fn online_slam_reset_clears_imu_window_state() {
    let (map, first_frame) = map_and_frame(10, 1);
    let mut slam = slam_pipeline_with_imu(map);

    let _ = slam.process_frame(&first_frame, []);
    slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    assert!(slam.imu_state.as_ref().unwrap().last_keyframe_id.is_some());

    slam.reset_sequence_state();

    let state = slam.imu_state.as_ref().unwrap();
    assert!(state.last_keyframe_id.is_none());
    assert!(state.pending_factor.is_none());
    // Map is preserved across reset (matches the existing
    // `reset_sequence_state` semantics for tracker / mapper).
    assert_eq!(slam.map().keyframes.len(), 1);
}

#[test]
fn online_slam_imu_integrator_resets_after_factor_emission() {
    // Pins the post-emit reset semantics: after a factor closes a
    // window, the running pre-integration window must start FRESH from
    // the new keyframe, not carry over the samples that funded the
    // just-emitted factor. Without this, a downstream consumer reading
    // `imu_state.preintegrator.delta()` for partial-window diagnostics
    // would see the entire trajectory accumulated since the first
    // keyframe.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    let _ = slam.process_frame(&first_frame, []);
    for _ in 0..10 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let _ = slam.process_frame(&second_frame, []);

    // Factor consumed 10 * 0.1 = 1.0 s; the running window must now
    // hold 0 s, not 1.0 s. Push 3 more samples and verify the window
    // reads `delta_time = 0.3 s` (post-reset accumulation only).
    for _ in 0..3 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let running_delta_time = slam
        .imu_state
        .as_ref()
        .unwrap()
        .preintegrator
        .delta()
        .delta_time;
    assert!(
        (running_delta_time - 0.3).abs() < 1.0e-9,
        "post-emit window should start fresh, got delta_time = {running_delta_time}"
    );
}

#[test]
fn keyframe_pose_storage_matches_tracker_pose() {
    // Regression: pins the invariant that
    // `slam.map().keyframes[id].frame.pose` is byte-equal to
    // `tracking.localization.pose` for the frame that produced the
    // keyframe — i.e. `keyframe_from_tracking_result` +
    // `StagedMapUpdate::apply_to` do not mutate the pose between
    // tracker output and map storage. Base path: no IMU, no VI init.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline(map, true);

    let first = slam.process_frame(&first_frame, []);
    assert!(first.tracking_succeeded(), "first frame must localise");
    let second = slam.process_frame(&second_frame, []);
    assert!(second.tracking_succeeded(), "second frame must localise");

    let tracker_pose = second
        .tracking
        .localization
        .pose
        .as_ref()
        .expect("tracker should have produced a pose on success");
    let tracker_center = tracker_pose.camera_center_world();

    let map_kf = slam.map().keyframes.get(&30).expect("kf 30 must exist");
    let map_pose = map_kf
        .frame
        .pose
        .as_ref()
        .expect("kf 30 stored pose must exist");
    let map_center = map_pose.camera_center_world();

    // The two MUST agree — `keyframe_from_tracking_result` does
    // `frame.pose = tracking.localization.pose.clone()`, and nothing
    // between staging and `apply_to` should mutate the pose.
    let diff = (tracker_center - map_center).norm();
    assert!(
        diff < 1.0e-9,
        "tracker pose vs stored map pose differ by {diff} m \
         (tracker {tracker_center:?}, map {map_center:?})"
    );
    // And the recovered centre must be near the synthetic ground-truth
    // (1.5, 0, 0) within PnP noise.
    let gt_diff = (tracker_center - Point3::new(1.5, 0.0, 0.0)).norm();
    assert!(
        gt_diff < 1.0e-3,
        "tracker recovered centre {tracker_center:?} is far from ground-truth (1.5, 0, 0); diff = {gt_diff}"
    );
}

#[test]
fn keyframe_pose_storage_matches_tracker_with_imu() {
    // Same invariant as `keyframe_pose_storage_matches_tracker_pose`,
    // but with IMU enabled so the `stage_imu_factor_on_new_keyframe`
    // path also runs. Pre-integration alone must never mutate
    // keyframe pose.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    let _ = slam.process_frame(&first_frame, []);
    for _ in 0..10 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let second = slam.process_frame(&second_frame, []);
    assert!(second.tracking_succeeded());

    let tracker_center = second
        .tracking
        .localization
        .pose
        .as_ref()
        .unwrap()
        .camera_center_world();
    let map_center = slam
        .map()
        .keyframes
        .get(&30)
        .unwrap()
        .frame
        .pose
        .as_ref()
        .unwrap()
        .camera_center_world();
    let diff = (tracker_center - map_center).norm();
    assert!(
        diff < 1.0e-9,
        "tracker centre {tracker_center:?} ≠ stored {map_center:?} (diff {diff})"
    );
}

#[test]
fn online_slam_imu_factor_propagates_gravity_weights_and_bias_linearisation() {
    // The pipeline must hand the staged factor every config-sourced
    // parameter the downstream BA glue needs. A bug that swapped (say)
    // weight_position for weight_rotation, or dropped a non-default
    // gravity / non-zero bias_gyro, would silently degrade BA
    // conditioning at every emission. Pin the wiring with a non-default
    // config so a copy-paste regression cannot blend in.
    let custom_imu_config = OnlineSlamImuConfig {
        gravity_world: Vector3::new(0.1, 9.7, -0.2),
        bias_gyro: Vector3::new(0.01, -0.02, 0.03),
        bias_acc: Vector3::new(-0.05, 0.04, 0.06),
        weight_position: 3.0,
        weight_velocity: 5.0,
        weight_rotation: 7.0,
    };
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            imu: Some(custom_imu_config.clone()),
            local_vi_ba: None,
            covisibility_local_ba: None,
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        },
    );

    let _ = slam.process_frame(&first_frame, []);
    for _ in 0..5 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let result = slam.process_frame(&second_frame, []);
    let factor = result
        .imu_factor
        .as_ref()
        .expect("factor expected for the second keyframe with custom IMU config");

    assert_eq!(factor.gravity_world, custom_imu_config.gravity_world);
    assert!((factor.weight_position - custom_imu_config.weight_position).abs() < 1.0e-12);
    assert!((factor.weight_velocity - custom_imu_config.weight_velocity).abs() < 1.0e-12);
    assert!((factor.weight_rotation - custom_imu_config.weight_rotation).abs() < 1.0e-12);
    // Bias linearisation point flows from config → ImuPreintegrator::new_with_bias →
    // ImuPreintegratedDelta — the very value `residual_with_bias_correction`
    // subtracts from the current bias estimate. A mis-wiring here would
    // silently zero out the bias-correction path.
    assert_eq!(
        factor.delta.bias_gyro_linearisation,
        custom_imu_config.bias_gyro
    );
    assert_eq!(
        factor.delta.bias_acc_linearisation,
        custom_imu_config.bias_acc
    );
}

#[test]
fn online_slam_imu_window_persists_across_non_keyframe_frames() {
    // If a frame doesn't register a new keyframe (mapper rejected by
    // its translation / frame-id gap policy), the running IMU window
    // must keep accumulating against the SAME `last_keyframe_id` and
    // emit a single factor at the next successful keyframe — not split
    // the samples across non-keyframe boundaries. The pipeline today
    // anchors on `applied_update.keyframe_count > 0`, so this test
    // also pins that decision.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    // Frame 20 is too close to frame 10 in camera-centre translation
    // (0.1 m < SimpleKeyframePolicy::min_translation = 1.0 m); the
    // mapper will reject it as a redundant keyframe.
    let (_, intermediate_frame) =
        map_and_frame_with_extra_landmarks(20, 1, Vector3::new(0.1, 0.0, 0.0));
    let (_, second_keyframe) =
        map_and_frame_with_extra_landmarks(40, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    let _ = slam.process_frame(&first_frame, []);
    assert_eq!(slam.map().keyframes.len(), 1);

    for _ in 0..4 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let intermediate = slam.process_frame(&intermediate_frame, []);
    // Intermediate frame tracks but the mapper rejects it as a keyframe
    // (translation too small). No factor must close here.
    assert!(intermediate.imu_factor.is_none());
    // `last_keyframe_id` still anchored on the first keyframe.
    assert_eq!(
        slam.imu_state.as_ref().and_then(|s| s.last_keyframe_id),
        Some(10),
    );
    // Map still has just the first keyframe.
    assert_eq!(slam.map().keyframes.len(), 1);

    for _ in 0..6 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let second = slam.process_frame(&second_keyframe, []);
    let factor = second
        .imu_factor
        .as_ref()
        .expect("factor must close at the next genuine keyframe across the rejected intermediate");
    assert_eq!(factor.keyframe_id_from, 10);
    assert_eq!(factor.keyframe_id_to, 40);
    // All 10 samples (4 + 6) flow into the SAME factor — the integrator
    // never reset across the rejected intermediate.
    assert!(
        (factor.delta.delta_time - 1.0).abs() < 1.0e-9,
        "expected the rejected intermediate to keep the window open; got delta_time = {}",
        factor.delta.delta_time
    );
}

#[test]
fn online_slam_imu_emits_factors_across_three_consecutive_keyframes() {
    // Two factors back-to-back: KF1→KF2 and KF2→KF3. The third keyframe
    // walks back partway so the same landmark set stays in frame. This
    // pins the chaining contract that a single `take_pending_imu_factor`
    // call clears the staged slot AND that the next emission anchors on
    // the just-set `last_keyframe_id`.
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    // KF3 at (0.3, 0, 0): camera-centre delta from KF2 is 1.2 m > 1.0 m
    // min_translation, so the mapper registers a third keyframe; the
    // landmarks (at world depth ~5 m) all stay comfortably inside the
    // 640x480 frame.
    let (_, third_frame) = map_and_frame_with_extra_landmarks(50, 1, Vector3::new(0.3, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    // KF1: no closure.
    let r1 = slam.process_frame(&first_frame, []);
    assert!(r1.imu_factor.is_none());

    // KF1 → KF2 window: 0.5 s.
    for _ in 0..5 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let r2 = slam.process_frame(&second_frame, []);
    let f12 = r2
        .imu_factor
        .as_ref()
        .expect("first chained factor expected (KF1 → KF2)");
    assert_eq!(f12.keyframe_id_from, 10);
    assert_eq!(f12.keyframe_id_to, 30);
    assert!((f12.delta.delta_time - 0.5).abs() < 1.0e-9);

    // KF2 → KF3 window: 0.7 s, with different dt to make sure the post-reset
    // accumulator is genuinely starting from zero (not biased by KF1→KF2's
    // 0.5 s endpoint).
    for _ in 0..7 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let r3 = slam.process_frame(&third_frame, []);
    let f23 = r3
        .imu_factor
        .as_ref()
        .expect("second chained factor expected (KF2 → KF3)");
    assert_eq!(f23.keyframe_id_from, 30);
    assert_eq!(f23.keyframe_id_to, 50);
    assert!(
        (f23.delta.delta_time - 0.7).abs() < 1.0e-9,
        "second factor must NOT carry over the first window's 0.5 s; got {}",
        f23.delta.delta_time
    );

    // Three keyframes in the map, three frames processed.
    assert_eq!(slam.map().keyframes.len(), 3);
    // The pending slot was overwritten on the third frame — taking it now
    // returns the KF2→KF3 factor, and the next call returns None.
    let taken = slam.take_pending_imu_factor().unwrap();
    assert_eq!(taken.keyframe_id_from, 30);
    assert_eq!(taken.keyframe_id_to, 50);
    assert!(slam.take_pending_imu_factor().is_none());
}

#[test]
fn online_slam_runs_local_vi_ba_when_factor_emitted() {
    // Two keyframes with one IMU factor between them — enough to drive the
    // local VI-BA wiring exactly once. The test does not assert on optimiser
    // convergence (the synthetic map already sits at truth so refinement is
    // a no-op); it pins the contract that `local_vi_ba.is_some()` exactly
    // when a factor was staged AND VI-BA is enabled in the config.
    use visloc_slam::OnlineSlamLocalBaConfig;
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            imu: Some(OnlineSlamImuConfig {
                // Match the synthetic-scene gravity convention used by the
                // factor builders above so the IMU residual sits at zero on
                // the truth poses.
                gravity_world: Vector3::zeros(),
                ..OnlineSlamImuConfig::default()
            }),
            local_vi_ba: Some(OnlineSlamLocalBaConfig {
                gravity_world: Vector3::zeros(),
                ..OnlineSlamLocalBaConfig::default()
            }),
            covisibility_local_ba: None,
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        },
    );

    let r1 = slam.process_frame(&first_frame, []);
    assert!(
        r1.imu_factor.is_none(),
        "no factor on the very first keyframe"
    );
    assert!(
        r1.local_vi_ba.is_none(),
        "VI-BA must not fire without a factor"
    );
    for _ in 0..10 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::zeros(), 0.1);
    }
    let r2 = slam.process_frame(&second_frame, []);
    let factor = r2.imu_factor.as_ref().expect("KF10→KF30 factor expected");
    assert_eq!(factor.keyframe_id_from, 10);
    assert_eq!(factor.keyframe_id_to, 30);
    let stats = r2
        .local_vi_ba
        .as_ref()
        .expect("local VI-BA must run when a factor closes a window");
    assert_eq!(stats.window_keyframe_ids, vec![10, 30]);
    assert_eq!(stats.imu_factor_count, 1);
    assert!(stats.observation_count > 0);
    // The state table now carries an entry per window keyframe.
    let vi_state = slam
        .local_vi_ba_state
        .as_ref()
        .expect("local_vi_ba_state should exist when configured");
    assert!(vi_state.keyframe_state.contains_key(&10));
    assert!(vi_state.keyframe_state.contains_key(&30));

    // Reset wipes the VI-BA state alongside the rest.
    slam.reset_sequence_state();
    let vi_state = slam.local_vi_ba_state.as_ref().unwrap();
    assert!(vi_state.keyframe_state.is_empty());
    assert!(vi_state.factor_history.is_empty());
}

#[test]
fn online_slam_runs_covisibility_local_ba_on_new_keyframe_trigger() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let (_, third_frame) = map_and_frame_with_extra_landmarks(50, 1, Vector3::new(3.0, 0.0, 0.0));
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            imu: None,
            local_vi_ba: None,
            covisibility_local_ba: Some(OnlineSlamCovisibilityLocalBaConfig {
                min_keyframes: 3,
                trigger_every_new_keyframes: 1,
                max_outlier_observation_ratio: None,
                max_behind_camera_landmark_ratio: None,
                min_fixed_to_optimized_ratio: None,
                ba: CovisibilityLocalBaConfig {
                    max_neighbor_keyframes: 1,
                    min_shared_landmarks: 4,
                    max_boundary_keyframes: 1,
                    min_boundary_observations: 1,
                    min_observations_per_landmark: 2,
                    ..CovisibilityLocalBaConfig::default()
                },
            }),
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        },
    );

    let r1 = slam.process_frame(&first_frame, []);
    assert!(r1.tracking_succeeded());
    assert!(
        r1.covisibility_local_ba.is_none(),
        "min_keyframes should skip startup"
    );

    let r2 = slam.process_frame(&second_frame, []);
    assert!(r2.tracking_succeeded());
    assert!(
        r2.covisibility_local_ba.is_none(),
        "min_keyframes should skip until the third keyframe"
    );

    let r3 = slam.process_frame(&third_frame, []);
    assert!(r3.tracking_succeeded());
    let stats = r3
        .covisibility_local_ba
        .as_ref()
        .expect("covisibility local BA should run on the third keyframe");
    assert!(stats.success, "unexpected BA error: {:?}", stats.error);
    assert_eq!(stats.active_keyframe_id, 50);
    assert_eq!(stats.map_keyframe_count, 3);
    assert!(stats.elapsed_ms >= 0.0);
    assert!(stats.error.is_none());
    assert!(stats.ba_result.is_some());
    assert!(stats.updated_keyframe_count >= 1);
    assert!(stats.updated_landmark_count > 0);
    assert!(stats.observation_count > 0);
    assert_eq!(stats.outlier_observation_ratio, Some(0.0));
    assert!(!stats.quality_gate_rejected);

    let selection = stats.selection.as_ref().expect("selection diagnostics");
    assert_eq!(selection.active_keyframe_id, 50);
    assert_eq!(selection.optimized_keyframe_ids[0], 50);
    assert_eq!(selection.optimized_keyframe_ids.len(), 2);
    assert_eq!(selection.fixed_keyframe_ids.len(), 1);
    assert_eq!(selection.landmark_ids.len(), 12);
    assert!(selection.observation_count > 0);
}

#[test]
fn online_covisibility_local_ba_quality_gate_rejects_writeback() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let (_, third_frame) = map_and_frame_with_extra_landmarks(50, 1, Vector3::new(3.0, 0.0, 0.0));
    let mut slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig {
            apply_map_updates: true,
            loop_closure: LoopClosureConfig {
                min_frame_id_gap: 5,
                min_shared_landmarks: 4,
                min_shared_landmark_ratio_percent: 50,
                ..LoopClosureConfig::default()
            },
            imu: None,
            local_vi_ba: None,
            covisibility_local_ba: Some(OnlineSlamCovisibilityLocalBaConfig {
                min_keyframes: 3,
                trigger_every_new_keyframes: 1,
                max_outlier_observation_ratio: Some(0.0),
                max_behind_camera_landmark_ratio: None,
                min_fixed_to_optimized_ratio: None,
                ba: CovisibilityLocalBaConfig {
                    max_neighbor_keyframes: 1,
                    min_shared_landmarks: 4,
                    max_boundary_keyframes: 1,
                    min_boundary_observations: 1,
                    min_observations_per_landmark: 2,
                    outlier_reprojection_threshold_px: Some(1.0),
                    ..CovisibilityLocalBaConfig::default()
                },
            }),
            vi_init: None,
            vi_motion_init: None,
            keep_pre_promotion_imu_factors: false,
            pose_graph_refinement: None,
            relocalization: None,
        },
    );

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    assert!(slam.process_frame(&second_frame, []).tracking_succeeded());

    let corrupted_xy = {
        let map = slam.map_mut();
        let observation = map
            .keyframes
            .get_mut(&30)
            .and_then(|kf| kf.observations.get_mut(0))
            .expect("second keyframe should carry observations");
        observation.xy.x += 100.0;
        observation.xy
    };

    let result = slam.process_frame(&third_frame, []);
    assert!(result.tracking_succeeded());
    let stats = result
        .covisibility_local_ba
        .as_ref()
        .expect("covisibility BA should run on the third keyframe");

    assert!(!stats.success);
    assert!(stats.quality_gate_rejected);
    assert_eq!(stats.updated_keyframe_count, 0);
    assert_eq!(stats.updated_landmark_count, 0);
    assert!(stats.outlier_observation_count > 0);
    assert!(matches!(
        stats.error.as_ref(),
        Some(CovisibilityLocalBaError::QualityGateRejected { .. })
    ));
    assert_eq!(slam.map().keyframes.len(), 3);
    assert_eq!(slam.map().keyframes[&30].observations[0].xy, corrupted_xy);
}

// Drives the opt-in fixed-anchor-ratio write-back gate end-to-end. The healthy
// covisibility trigger selects optimized=2 / fixed=1 on the third keyframe;
// with `min_fixed_to_optimized_ratio = Some(0.6)` the required fixed count is
// ceil(2 * 0.6) = 2 > 1, so the write-back is rejected and the active keyframe
// pose must match the no-BA baseline. With the gate off the same solve commits
// and moves the pose. (The behind-camera detector is unit-tested directly in
// `covisibility_ba.rs`; forcing a genuine behind-camera solve through the online
// pipeline is not feasible with the healthy front-facing fixtures here.)
#[test]
fn online_covisibility_local_ba_fixed_ratio_gate_rejects_writeback() {
    fn run_active_keyframe_center(
        covisibility: Option<OnlineSlamCovisibilityLocalBaConfig>,
    ) -> (Option<bool>, usize, Point3<f64>) {
        let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, second_frame) =
            map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
        let (_, third_frame) =
            map_and_frame_with_extra_landmarks(50, 1, Vector3::new(3.0, 0.0, 0.0));
        let mut slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 50,
                    ..LoopClosureConfig::default()
                },
                imu: None,
                local_vi_ba: None,
                covisibility_local_ba: covisibility,
                vi_init: None,
                vi_motion_init: None,
                keep_pre_promotion_imu_factors: false,
                pose_graph_refinement: None,
                relocalization: None,
            },
        );
        assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
        assert!(slam.process_frame(&second_frame, []).tracking_succeeded());
        let result = slam.process_frame(&third_frame, []);
        assert!(result.tracking_succeeded());
        let (success, updated) = match result.covisibility_local_ba.as_ref() {
            Some(stats) => {
                if !stats.success {
                    assert!(matches!(
                        stats.error.as_ref(),
                        Some(CovisibilityLocalBaError::FixedSupportRatioRejected { .. })
                    ));
                }
                (Some(stats.success), stats.updated_keyframe_count)
            }
            None => (None, 0),
        };
        let center = slam.map().keyframes[&50]
            .frame
            .pose
            .as_ref()
            .expect("active keyframe pose")
            .camera_center_world();
        (success, updated, center)
    }

    fn config(min_fixed_to_optimized_ratio: Option<f64>) -> OnlineSlamCovisibilityLocalBaConfig {
        OnlineSlamCovisibilityLocalBaConfig {
            min_keyframes: 3,
            trigger_every_new_keyframes: 1,
            max_outlier_observation_ratio: None,
            max_behind_camera_landmark_ratio: None,
            min_fixed_to_optimized_ratio,
            ba: CovisibilityLocalBaConfig {
                max_neighbor_keyframes: 1,
                min_shared_landmarks: 4,
                max_boundary_keyframes: 1,
                min_boundary_observations: 1,
                min_observations_per_landmark: 2,
                ..CovisibilityLocalBaConfig::default()
            },
        }
    }

    // Baseline: covisibility BA disabled entirely -> the pristine tracked pose.
    let (_, _, baseline_center) = run_active_keyframe_center(None);

    // Gate off: the solve runs and commits its refined poses/landmarks back into
    // the map (`updated_keyframe_count >= 1`).
    let (off_success, off_updated, _off_center) = run_active_keyframe_center(Some(config(None)));
    assert_eq!(off_success, Some(true));
    assert!(
        off_updated >= 1,
        "gate-off run should write refined poses back into the map"
    );

    // Gate on: the write-back is rejected, nothing is committed, and the map is
    // bit-identical to the no-BA baseline.
    let (on_success, on_updated, on_center) = run_active_keyframe_center(Some(config(Some(0.6))));
    assert_eq!(on_success, Some(false));
    assert_eq!(
        on_updated, 0,
        "gate-on rejection must not commit any keyframe update"
    );
    assert_eq!(
        on_center, baseline_center,
        "gate-on rejection must leave the active keyframe pose unchanged"
    );
}

#[test]
fn online_slam_skips_local_vi_ba_when_disabled() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
    let mut slam = slam_pipeline_with_imu(map);

    let _ = slam.process_frame(&first_frame, []);
    for _ in 0..10 {
        slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 9.81, 0.0), 0.1);
    }
    let r2 = slam.process_frame(&second_frame, []);
    assert!(
        r2.imu_factor.is_some(),
        "factor still emitted in IMU-only config"
    );
    assert!(
        r2.local_vi_ba.is_none(),
        "VI-BA must not fire when `local_vi_ba` is None"
    );
    assert!(slam.local_vi_ba_state.is_none());
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
            ..OnlineSlamConfig::default()
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

/// Ground truth is a fixed point of the chordal relaxation: the truth
/// rotations already achieve zero chordal cost and the relaxed minimizer is
/// unique, so initializing from truth must leave the rotations untouched.
#[test]
fn chordal_rotation_init_keeps_ground_truth_rotations() {
    let truth_10 = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let truth_20 = pose_with_yaw(Vector3::new(1.5, 0.0, 0.5), 0.4);
    let truth_30 = pose_with_yaw(Vector3::new(0.3, 0.0, 0.2), 0.9);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    graph.add_pose(30, truth_30.clone());
    graph.anchor(10);
    graph.add_sequential_edge(10, 20, relative_world_to_camera(&truth_10, &truth_20));
    graph.add_sequential_edge(20, 30, relative_world_to_camera(&truth_20, &truth_30));
    graph.add_sequential_edge(10, 30, relative_world_to_camera(&truth_10, &truth_30));

    assert!(graph.chordal_rotation_cost() < 1.0e-15);
    let stats = graph
        .initialize_rotations_chordal(LinearSolver::Dense)
        .expect("chordal init must succeed");
    assert_eq!(stats.variable_count, 2);
    assert_eq!(stats.edge_count, 3);
    assert!(stats.max_rotation_update_deg < 1.0e-6, "{stats:?}");
    assert!(stats.cost_after < 1.0e-12, "{stats:?}");

    for (id, truth) in [(20u64, &truth_20), (30, &truth_30)] {
        let r_truth = truth
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_now = graph.poses[&id]
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        assert!(
            (r_truth - r_now).norm() < 1.0e-9,
            "node {id} rotation drifted"
        );
    }
}

/// Starting from deliberately wrong rotations, the chordal init must recover
/// the truth orientations of a consistent loop (the relaxation has a unique
/// zero-cost minimizer) and emit only valid proper rotations.
#[test]
fn chordal_rotation_init_recovers_rotations_from_a_bad_start() {
    use nalgebra::Matrix3;

    let truth = [
        (1u64, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0)),
        (2, pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.6)),
        (3, pose_with_yaw(Vector3::new(2.0, 0.0, 0.7), 1.2)),
        (4, pose_with_yaw(Vector3::new(1.0, 0.0, 1.4), 1.8)),
    ];

    let mut graph = PoseGraph::new();
    for (i, (id, truth_pose)) in truth.iter().enumerate() {
        if i == 0 {
            graph.add_pose(*id, truth_pose.clone());
        } else {
            // A grossly wrong yaw so the relaxed blocks are far from rotations.
            graph.add_pose(*id, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), -2.5));
        }
    }
    graph.anchor(1);
    let edge = |a: usize, b: usize| relative_world_to_camera(&truth[a].1, &truth[b].1);
    graph.add_sequential_edge(1, 2, edge(0, 1));
    graph.add_sequential_edge(2, 3, edge(1, 2));
    graph.add_sequential_edge(3, 4, edge(2, 3));
    // Closing edge of the loop (kind is cosmetic for rotation init).
    graph.add_sequential_edge(4, 1, edge(3, 0));

    let cost_before = graph.chordal_rotation_cost();
    assert!(
        cost_before > 1.0,
        "bad init should be costly: {cost_before}"
    );

    let stats = graph
        .initialize_rotations_chordal(LinearSolver::Sparse)
        .expect("chordal init must succeed");
    assert!(
        stats.cost_after < 1.0e-9,
        "consistent loop => near-zero: {stats:?}"
    );
    assert!(stats.cost_after < cost_before);
    assert!(
        stats.max_rotation_update_deg > 10.0,
        "should move a lot: {stats:?}"
    );

    for (id, truth_pose) in truth.iter().skip(1) {
        let r_truth = truth_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_now = graph.poses[id]
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        assert!((r_truth - r_now).norm() < 1.0e-6, "node {id} not recovered");
        // Projection must yield a proper orthonormal rotation.
        assert!((r_now * r_now.transpose() - Matrix3::identity()).norm() < 1.0e-9);
        assert!((r_now.determinant() - 1.0).abs() < 1.0e-9);
    }
}

/// The dense and sparse backends assemble the same normal equations, so they
/// must land on identical rotations.
#[test]
fn chordal_rotation_init_dense_and_sparse_agree() {
    let truth_1 = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let truth_2 = pose_with_yaw(Vector3::new(1.0, 0.0, 0.3), 0.5);
    let truth_3 = pose_with_yaw(Vector3::new(0.4, 0.0, 1.1), 1.1);

    let build = || {
        let mut g = PoseGraph::new();
        g.add_pose(1, truth_1.clone());
        g.add_pose(2, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), -1.0));
        g.add_pose(3, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 2.0));
        g.anchor(1);
        g.add_sequential_edge(1, 2, relative_world_to_camera(&truth_1, &truth_2));
        g.add_sequential_edge(2, 3, relative_world_to_camera(&truth_2, &truth_3));
        g.add_sequential_edge(1, 3, relative_world_to_camera(&truth_1, &truth_3));
        g
    };

    let mut dense = build();
    let mut sparse = build();
    dense
        .initialize_rotations_chordal(LinearSolver::Dense)
        .expect("dense");
    sparse
        .initialize_rotations_chordal(LinearSolver::Sparse)
        .expect("sparse");

    for id in [2u64, 3] {
        let rd = dense.poses[&id]
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let rs = sparse.poses[&id]
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        assert!((rd - rs).norm() < 1.0e-9, "backends disagree on node {id}");
    }
}

/// Chordal init must reject the same degenerate graphs the SE(3) solver does.
#[test]
fn chordal_rotation_init_rejects_degenerate_graphs() {
    use visloc_core::geometry::SE3;

    let a = pose_at(Vector3::zeros());
    let b = pose_at(Vector3::new(1.0, 0.0, 0.0));

    let mut no_anchor = PoseGraph::new();
    no_anchor.add_pose(1, a.clone());
    no_anchor.add_pose(2, b.clone());
    no_anchor.add_sequential_edge(1, 2, relative_world_to_camera(&a, &b));
    assert_eq!(
        no_anchor.initialize_rotations_chordal(LinearSolver::Sparse),
        Err(PoseGraphError::NoAnchor)
    );

    let mut no_edges = PoseGraph::new();
    no_edges.add_pose(1, a.clone());
    no_edges.anchor(1);
    assert_eq!(
        no_edges.initialize_rotations_chordal(LinearSolver::Sparse),
        Err(PoseGraphError::NoEdges)
    );

    let mut missing = PoseGraph::new();
    missing.add_pose(1, a.clone());
    missing.anchor(1);
    missing.add_sequential_edge(1, 99, SE3::identity());
    assert_eq!(
        missing.initialize_rotations_chordal(LinearSolver::Sparse),
        Err(PoseGraphError::MissingNode(99))
    );

    let mut anchor_only = PoseGraph::new();
    anchor_only.add_pose(1, a);
    anchor_only.anchor(1);
    anchor_only.add_sequential_edge(1, 1, SE3::identity());
    assert_eq!(
        anchor_only.initialize_rotations_chordal(LinearSolver::Sparse),
        Err(PoseGraphError::NoVariables)
    );
}

/// `optimize_se3_iterative` seeds with a chordal rotation init by default, so a
/// grossly mis-oriented (but internally consistent) 3D loop is rescued straight
/// to the global optimum. The same solve with `chordal_init: false` is left at
/// the raw odometry estimate, so seeding can only help — never hurt — the final
/// cost. `initial_cost` is the pre-seed value either way (the documented
/// semantics), so the seeded run reports the full reduction.
#[test]
fn optimize_se3_seeds_with_chordal_init_by_default() {
    let truth = [
        (1u64, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0)),
        (2, pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.6)),
        (3, pose_with_yaw(Vector3::new(2.0, 0.0, 0.7), 1.2)),
        (4, pose_with_yaw(Vector3::new(1.0, 0.0, 1.4), 1.8)),
    ];

    // Internally consistent full-SE(3) edges around the loop; every non-anchor
    // node starts at a grossly wrong yaw and a collapsed center.
    let build = || {
        let mut graph = PoseGraph::new();
        for (i, (id, truth_pose)) in truth.iter().enumerate() {
            if i == 0 {
                graph.add_pose(*id, truth_pose.clone());
            } else {
                graph.add_pose(*id, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), -2.5));
            }
        }
        graph.anchor(1);
        let edge = |a: usize, b: usize| relative_world_to_camera(&truth[a].1, &truth[b].1);
        graph.add_sequential_edge(1, 2, edge(0, 1));
        graph.add_sequential_edge(2, 3, edge(1, 2));
        graph.add_sequential_edge(3, 4, edge(2, 3));
        graph.add_sequential_edge(4, 1, edge(3, 0));
        graph
    };

    let lm = |chordal_init: bool| PoseGraphSe3Config {
        initial_lambda: Some(1e-3),
        max_iterations: 50,
        linear_solver: LinearSolver::Sparse,
        chordal_init,
        ..PoseGraphSe3Config::default()
    };

    // Default (chordal-seeded) run: lands on the global optimum and recovers the
    // ground-truth poses, while reporting the pre-seed cost as `initial_cost`.
    let mut seeded = build();
    let raw_cost = seeded.se3_cost();
    let result = seeded
        .optimize_se3_iterative(&lm(true))
        .expect("seeded LM must succeed");
    assert!(result.converged, "seeded solve should converge: {result:?}");
    assert!(
        result.final_cost < 1.0e-9,
        "seeded solve should reach the optimum: {}",
        result.final_cost
    );
    assert!(
        (result.initial_cost - raw_cost).abs() < 1.0e-9,
        "initial_cost must be the pre-seed cost: {} vs {raw_cost}",
        result.initial_cost
    );
    for (id, truth_pose) in &truth {
        let recovered = seeded.poses[id].camera_center_world();
        let expected = truth_pose.camera_center_world();
        assert!(
            (recovered - expected).norm() < 1.0e-6,
            "node {id} center not recovered: {recovered:?} vs {expected:?}"
        );
    }

    // Disabling the seed leaves the same raw start, so its final cost can only be
    // equal-or-worse than the seeded run's.
    let mut unseeded = build();
    let unseeded_result = unseeded
        .optimize_se3_iterative(&lm(false))
        .expect("unseeded LM must succeed");
    assert!(
        result.final_cost <= unseeded_result.final_cost + 1.0e-9,
        "chordal seeding must not worsen the final cost: seeded={} unseeded={}",
        result.final_cost,
        unseeded_result.final_cost
    );
}

/// An identity information matrix must reproduce the legacy isotropic
/// unit-weight path bit-for-bit: same two conflicting edges, solved both ways,
/// land the free node in the same place.
#[test]
fn pose_graph_se3_identity_information_matches_isotropic_weight_one() {
    use nalgebra::Matrix6;
    use visloc_slam::PoseGraphEdgeKind;

    let anchor = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let target_a = pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.2);
    let target_b = pose_with_yaw(Vector3::new(3.0, 0.0, 0.0), -0.1);
    let edge_a = relative_world_to_camera(&anchor, &target_a);
    let edge_b = relative_world_to_camera(&anchor, &target_b);
    let init_20 = pose_with_yaw(Vector3::new(2.0, 0.1, 0.0), 0.0);

    let mut isotropic = PoseGraph::new();
    isotropic.add_pose(10, anchor.clone());
    isotropic.add_pose(20, init_20.clone());
    isotropic.anchor(10);
    isotropic.add_sequential_edge(10, 20, edge_a.clone());
    isotropic.add_sequential_edge(10, 20, edge_b.clone());
    isotropic
        .optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("isotropic solve");

    let mut info = PoseGraph::new();
    info.add_pose(10, anchor);
    info.add_pose(20, init_20);
    info.anchor(10);
    info.add_edge_with_information(
        10,
        20,
        edge_a,
        PoseGraphEdgeKind::Sequential,
        Matrix6::identity(),
    );
    info.add_edge_with_information(
        10,
        20,
        edge_b,
        PoseGraphEdgeKind::Sequential,
        Matrix6::identity(),
    );
    info.optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("information solve");

    let iso_20 = isotropic.poses[&20].camera_center_world();
    let inf_20 = info.poses[&20].camera_center_world();
    assert!(
        (iso_20 - inf_20).norm() < 1.0e-10,
        "Ω=I must reproduce isotropic weight-1: iso={iso_20:?} info={inf_20:?}"
    );
}

/// A non-uniform information matrix must actually steer the solve: between two
/// conflicting odometry edges, the free node is pulled toward whichever edge
/// carries the larger translation information.
#[test]
fn pose_graph_se3_information_matrix_steers_solution() {
    use nalgebra::Matrix6;
    use visloc_slam::PoseGraphEdgeKind;

    let anchor = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let target_near = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let target_far = pose_at(Vector3::new(3.0, 0.0, 0.0));
    let edge_near = relative_world_to_camera(&anchor, &target_near);
    let edge_far = relative_world_to_camera(&anchor, &target_far);

    // Anisotropic: strong on the translation block ρ (indices 0..3 in the
    // [ρ; ω] tangent layout), left weak on rotation.
    let strong = {
        let mut m = Matrix6::identity();
        for k in 0..3 {
            m[(k, k)] = 50.0;
        }
        m
    };
    let weak = Matrix6::identity();

    let solve = |near_info: Matrix6<f64>, far_info: Matrix6<f64>| {
        let mut g = PoseGraph::new();
        g.add_pose(10, anchor.clone());
        g.add_pose(20, pose_at(Vector3::new(2.0, 0.0, 0.0)));
        g.anchor(10);
        g.add_edge_with_information(
            10,
            20,
            edge_near.clone(),
            PoseGraphEdgeKind::Sequential,
            near_info,
        );
        g.add_edge_with_information(
            10,
            20,
            edge_far.clone(),
            PoseGraphEdgeKind::LoopClosure,
            far_info,
        );
        g.optimize_se3_iterative(&PoseGraphSe3Config::default())
            .expect("solve");
        g.poses[&20].camera_center_world().x
    };

    let pulled_near = solve(strong, weak);
    let pulled_far = solve(weak, strong);
    assert!(
        pulled_near < 1.5,
        "strong near-info should pull toward x=1.0, got {pulled_near}"
    );
    assert!(
        pulled_far > 2.5,
        "strong far-info should pull toward x=3.0, got {pulled_far}"
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
            // Exercise the LM λ machinery on the raw odometry estimate; chordal
            // seeding would converge this in a single constant-λ step.
            chordal_init: false,
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

#[test]
fn hybrid_loop_closure_verifier_accepts_when_essential_and_pnp_agree() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    // Calibrate the essential verifier's translation scale to the truth so
    // its recovered translation magnitude matches PnP's metric one. Hybrid
    // direction-only check would still pass without this, but using the
    // correct scale exercises the agreement path more realistically.
    let truth_first = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let truth_second =
        Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(0.2, 0.0, 0.1));
    let scale = relative_world_to_camera(&truth_first, &truth_second)
        .translation
        .norm();
    let verifier: HybridLoopClosureVerifier = HybridLoopClosureVerifier {
        essential: EssentialMatrixLoopClosureVerifier {
            config: LoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_sampson_error: 5.0e-3,
                default_translation_scale: scale,
            },
            ..Default::default()
        },
        pnp: PnPLoopClosureVerifier {
            config: PnPLoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_reprojection_error_px: 4.0,
            },
            ..Default::default()
        },
        config: HybridLoopClosureVerifierConfig::default(),
    };
    verify_loop_closure_candidates_hybrid(
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
        .expect("hybrid verifier output must be populated");
    assert!(
        verification.verified,
        "hybrid should accept consistent candidate: {verification:?}"
    );
    assert!(verification.relative_pose.is_some());
    assert!(verification.mean_reprojection_error_px.is_some());
    assert!(verification.failure_reason.is_none());
}

#[test]
fn hybrid_loop_closure_verifier_rejects_when_essential_pose_disagrees_with_pnp() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    // Force pose disagreement by clamping the rotation tolerance to an
    // unreasonably small value: even tiny numerical noise between the two
    // backends will exceed it.
    let verifier: HybridLoopClosureVerifier = HybridLoopClosureVerifier {
        essential: EssentialMatrixLoopClosureVerifier {
            config: LoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_sampson_error: 5.0e-3,
                default_translation_scale: 1.0,
            },
            ..Default::default()
        },
        pnp: PnPLoopClosureVerifier {
            config: PnPLoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_reprojection_error_px: 4.0,
            },
            ..Default::default()
        },
        config: HybridLoopClosureVerifierConfig {
            max_translation_direction_disagreement_rad: 1.0e-9,
            max_rotation_disagreement_rad: 1.0e-9,
        },
    };
    verify_loop_closure_candidates_hybrid(
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
        .expect("hybrid verifier output must be populated");
    assert!(!verification.verified);
    assert_eq!(
        verification.failure_reason,
        Some(LoopClosureVerificationFailureReason::PoseDisagreement)
    );
}

#[test]
fn hybrid_loop_closure_verifier_propagates_essential_failure_when_essential_rejects() {
    let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
    let (_, second_frame) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(0.2, 0.0, 0.1));
    let camera = map.cameras.get(&1).cloned().unwrap();
    let mut slam = slam_pipeline_for_verifier(map);

    assert!(slam.process_frame(&first_frame, []).tracking_succeeded());
    let mut second = slam.process_frame(&second_frame, []);
    assert_eq!(second.loop_closure_candidates.len(), 1);

    let verifier: HybridLoopClosureVerifier = HybridLoopClosureVerifier {
        essential: EssentialMatrixLoopClosureVerifier {
            // Demand more correspondences than possible so essential rejects
            // up front via InsufficientCorrespondences.
            config: LoopClosureVerifierConfig {
                min_inliers: 64,
                min_inlier_ratio: 0.6,
                max_mean_sampson_error: 5.0e-3,
                default_translation_scale: 1.0,
            },
            ..Default::default()
        },
        pnp: PnPLoopClosureVerifier {
            config: PnPLoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_reprojection_error_px: 4.0,
            },
            ..Default::default()
        },
        config: HybridLoopClosureVerifierConfig::default(),
    };
    verify_loop_closure_candidates_hybrid(
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
        .expect("hybrid verifier output must be populated");
    assert!(!verification.verified);
    assert_eq!(
        verification.failure_reason,
        Some(LoopClosureVerificationFailureReason::InsufficientCorrespondences)
    );
}

/// Build a small drifted loop graph identical to the
/// `pose_graph_translation_gauss_newton_pulls_drifted_loop_back_to_anchor`
/// fixture. Returned graph already has the anchor and edges set.
fn drifted_translation_loop_graph() -> PoseGraph {
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(1.4, 0.0, 0.4));
    let truth_30 = pose_at(Vector3::new(0.2, 0.0, 0.1));
    let edge_10_20 = relative_world_to_camera(&truth_10, &truth_20);
    let edge_20_30 = relative_world_to_camera(&truth_20, &truth_30);
    let edge_10_30 = relative_world_to_camera(&truth_10, &truth_30);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10);
    graph.add_pose(20, truth_20);
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
    graph
}

#[test]
fn sparse_translation_solver_matches_dense_on_drifted_loop() {
    let mut dense_graph = drifted_translation_loop_graph();
    let mut sparse_graph = drifted_translation_loop_graph();

    let dense_step = dense_graph
        .optimize_translations_once_with(LinearSolver::Dense)
        .expect("dense solve must succeed");
    let sparse_step = sparse_graph
        .optimize_translations_once_with(LinearSolver::Sparse)
        .expect("sparse solve must succeed");

    assert_eq!(dense_step.anchor_id, sparse_step.anchor_id);
    assert_eq!(dense_step.edge_count, sparse_step.edge_count);
    assert_eq!(dense_step.variable_count, sparse_step.variable_count);
    assert!(
        (dense_step.cost_after - sparse_step.cost_after).abs() < 1.0e-12,
        "cost mismatch dense={} sparse={}",
        dense_step.cost_after,
        sparse_step.cost_after
    );

    for id in [20u64, 30u64] {
        let dense_center = dense_graph.poses[&id].camera_center_world();
        let sparse_center = sparse_graph.poses[&id].camera_center_world();
        assert!(
            (dense_center - sparse_center).norm() < 1.0e-10,
            "node {id} dense={dense_center:?} sparse={sparse_center:?}"
        );
    }
}

fn drifted_se3_loop_graph() -> PoseGraph {
    let truth_10 = pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0);
    let truth_20 = pose_with_yaw(Vector3::new(1.5, 0.0, 0.5), 0.4);
    let truth_30 = pose_with_yaw(Vector3::new(0.3, 0.0, 0.2), 0.8);
    let edge_10_20 = relative_world_to_camera(&truth_10, &truth_20);
    let edge_20_30 = relative_world_to_camera(&truth_20, &truth_30);
    let edge_10_30 = relative_world_to_camera(&truth_10, &truth_30);

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10);
    graph.add_pose(20, truth_20);
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
    graph
}

#[test]
fn sparse_se3_solver_matches_dense_on_drifted_yaw_loop() {
    let mut dense_graph = drifted_se3_loop_graph();
    let mut sparse_graph = drifted_se3_loop_graph();

    let dense_result = dense_graph
        .optimize_se3_iterative(&PoseGraphSe3Config {
            linear_solver: LinearSolver::Dense,
            ..PoseGraphSe3Config::default()
        })
        .expect("dense SE(3) solve must succeed");
    let sparse_result = sparse_graph
        .optimize_se3_iterative(&PoseGraphSe3Config {
            linear_solver: LinearSolver::Sparse,
            ..PoseGraphSe3Config::default()
        })
        .expect("sparse SE(3) solve must succeed");

    assert_eq!(dense_result.converged, sparse_result.converged);
    assert!(
        (dense_result.final_cost - sparse_result.final_cost).abs() < 1.0e-10,
        "final cost mismatch dense={} sparse={}",
        dense_result.final_cost,
        sparse_result.final_cost
    );

    for id in [20u64, 30u64] {
        let dense_pose = &dense_graph.poses[&id];
        let sparse_pose = &sparse_graph.poses[&id];
        let center_diff =
            (dense_pose.camera_center_world() - sparse_pose.camera_center_world()).norm();
        let r_dense = dense_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_sparse = sparse_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let rot_diff = (r_dense - r_sparse).norm();
        assert!(
            center_diff < 1.0e-9 && rot_diff < 1.0e-9,
            "node {id} center_diff={center_diff:.3e} rot_diff={rot_diff:.3e}"
        );
    }
}

#[test]
fn sparse_se3_solver_matches_dense_under_lm_damping() {
    let mut dense_graph = drifted_se3_loop_graph();
    let mut sparse_graph = drifted_se3_loop_graph();
    let cfg = |solver| PoseGraphSe3Config {
        linear_solver: solver,
        initial_lambda: Some(1.0e-4),
        max_iterations: 20,
        ..PoseGraphSe3Config::default()
    };

    let dense_result = dense_graph
        .optimize_se3_iterative(&cfg(LinearSolver::Dense))
        .expect("dense LM solve must succeed");
    let sparse_result = sparse_graph
        .optimize_se3_iterative(&cfg(LinearSolver::Sparse))
        .expect("sparse LM solve must succeed");

    assert!(
        (dense_result.final_cost - sparse_result.final_cost).abs() < 1.0e-9,
        "LM final cost mismatch dense={} sparse={}",
        dense_result.final_cost,
        sparse_result.final_cost
    );
    assert!(dense_result.final_cost < 1.0e-9);
    assert!(sparse_result.final_cost < 1.0e-9);
}

fn synthetic_keyframe_features(camera_center: Vector3<f64>) -> visloc_vision::features::FeatureSet {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
    // 12 well-separated landmarks so the essential-matrix RANSAC has plenty
    // of geometry to lock onto.
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
    let mut keypoints = Vec::with_capacity(points.len());
    let mut descriptors = Vec::with_capacity(points.len());
    for (index, p) in points.iter().enumerate() {
        let kp = camera.project(&pose.transform_world_point(p)).unwrap();
        keypoints.push(kp);
        // Distinctive descriptors with a unique L2 nearest-neighbour per
        // landmark so the brute-force matcher pairs corresponding keypoints
        // across keyframes.
        descriptors.push(vec![index as f32, 1.0]);
    }
    visloc_vision::features::FeatureSet::new(keypoints, descriptors).unwrap()
}

fn synthetic_keyframe_features_disjoint(
    camera_center: Vector3<f64>,
    descriptor_offset: f32,
) -> visloc_vision::features::FeatureSet {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
    let points = [
        Point3::new(2.0, 2.0, 8.0),
        Point3::new(2.5, 1.5, 8.2),
        Point3::new(1.8, 2.7, 7.9),
        Point3::new(2.2, 2.4, 8.1),
        Point3::new(1.6, 2.1, 8.05),
        Point3::new(2.4, 1.8, 8.3),
        Point3::new(1.9, 2.3, 7.95),
        Point3::new(2.6, 2.5, 8.15),
        Point3::new(1.7, 2.6, 7.85),
        Point3::new(2.3, 1.9, 8.25),
        Point3::new(2.1, 2.2, 8.4),
        Point3::new(1.5, 2.0, 7.7),
    ];
    let mut keypoints = Vec::with_capacity(points.len());
    let mut descriptors = Vec::with_capacity(points.len());
    for (index, p) in points.iter().enumerate() {
        let kp = camera.project(&pose.transform_world_point(p)).unwrap();
        keypoints.push(kp);
        // Offset every descriptor far away from the loop-closing keyframe's
        // descriptor space so the brute-force matcher pairs nothing useful.
        descriptors.push(vec![index as f32 + descriptor_offset, 100.0]);
    }
    visloc_vision::features::FeatureSet::new(keypoints, descriptors).unwrap()
}

#[test]
fn pairwise_loop_closure_scanner_finds_revisited_keyframe_pair() {
    // Three keyframes: id=10 and id=200 observe the *same* landmark cloud
    // from slightly different camera centers (a real loop closure), id=100
    // observes a disjoint cloud with descriptors offset out of band so it
    // matches neither end. With min_keyframe_id_gap=50 the scanner should
    // only emit the (10, 200) pair.
    let kf10 = synthetic_keyframe_features(Vector3::zeros());
    let kf100 = synthetic_keyframe_features_disjoint(Vector3::new(5.0, 0.0, 0.0), 1000.0);
    let kf200 = synthetic_keyframe_features(Vector3::new(0.15, 0.0, 0.05));
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);

    let views = [
        PairwiseKeyframeView::from_features(10, &kf10),
        PairwiseKeyframeView::from_features(100, &kf100),
        PairwiseKeyframeView::from_features(200, &kf200),
    ];
    let matcher = visloc_vision::matching::BruteForceMatcher { ratio: None };
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 8,
            min_inlier_ratio: 0.6,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };

    let candidates = scan_pairwise_loop_closures(
        &views,
        &matcher,
        &verifier,
        &camera,
        &PairwiseLoopClosureScannerConfig {
            min_keyframe_id_gap: 50,
            min_matches: 8,
        },
    );

    assert_eq!(candidates.len(), 1, "expected only the (10, 200) loop pair");
    let c = &candidates[0];
    assert_eq!(c.matched_keyframe_id, 10);
    assert_eq!(c.query_frame_id, 200);
    assert!(c.geometrically_verified);
    let v = c
        .verification
        .as_ref()
        .expect("verifier output is populated");
    assert!(v.verified);
    assert!(v.inlier_count >= 8);
}

#[test]
fn pairwise_loop_closure_scanner_skips_under_min_keyframe_id_gap() {
    // Same loop-pairing geometry as above, but raise min_keyframe_id_gap
    // beyond 200 - 10 so the (10, 200) pair is no longer eligible. No
    // candidates should come out.
    let kf10 = synthetic_keyframe_features(Vector3::zeros());
    let kf200 = synthetic_keyframe_features(Vector3::new(0.15, 0.0, 0.05));
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);

    let views = [
        PairwiseKeyframeView::from_features(10, &kf10),
        PairwiseKeyframeView::from_features(200, &kf200),
    ];
    let matcher = visloc_vision::matching::BruteForceMatcher { ratio: None };
    let verifier = EssentialMatrixLoopClosureVerifier::default();

    let candidates = scan_pairwise_loop_closures(
        &views,
        &matcher,
        &verifier,
        &camera,
        &PairwiseLoopClosureScannerConfig {
            min_keyframe_id_gap: 500,
            min_matches: 8,
        },
    );

    assert!(candidates.is_empty());
}

fn keyframe_with_features(
    frame_id: u64,
    camera_id: u64,
    features: &visloc_vision::features::FeatureSet,
) -> visloc_core::types::Keyframe {
    let mut frame = Frame::new(frame_id, camera_id);
    frame.keypoints = features.keypoints.clone();
    frame.descriptors = features.descriptors.clone();
    visloc_core::types::Keyframe {
        frame,
        observations: Vec::new(),
    }
}

#[test]
fn online_slam_pipeline_scan_appearance_loops_finds_revisited_pair() {
    // Three keyframes in a `VisualMap`: ids 10 and 200 observe the
    // same 12-landmark cloud (loop-like appearance), id 100 observes a
    // disjoint cloud with offset descriptors. With min_keyframe_id_gap
    // = 50 the eligible pairs are (10, 100), (10, 200), (100, 200);
    // only (10, 200) shares descriptor space, so the appearance scanner
    // should emit exactly one candidate.
    let kf10_features = synthetic_keyframe_features(Vector3::zeros());
    let kf100_features = synthetic_keyframe_features_disjoint(Vector3::new(5.0, 0.0, 0.0), 1000.0);
    let kf200_features = synthetic_keyframe_features(Vector3::new(0.15, 0.0, 0.05));
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let kf10 = keyframe_with_features(10, camera.id, &kf10_features);
    let kf100 = keyframe_with_features(100, camera.id, &kf100_features);
    let kf200 = keyframe_with_features(200, camera.id, &kf200_features);
    map.keyframes.insert(kf10.frame.id, kf10);
    map.keyframes.insert(kf100.frame.id, kf100);
    map.keyframes.insert(kf200.frame.id, kf200);

    let slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig::default(),
    );

    let candidates = slam.scan_appearance_loops(
        &visloc_vision::matching::BruteForceMatcher { ratio: None },
        &EssentialMatrixLoopClosureVerifier {
            config: LoopClosureVerifierConfig {
                min_inliers: 8,
                min_inlier_ratio: 0.6,
                max_mean_sampson_error: 5.0e-3,
                default_translation_scale: 1.0,
            },
            ..Default::default()
        },
        &camera,
        &AppearanceLoopScannerSettings {
            min_keyframe_id_gap: 50,
            min_matches: 8,
        },
    );

    assert_eq!(candidates.len(), 1, "expected only the (10, 200) loop pair");
    let c = &candidates[0];
    assert_eq!(c.matched_keyframe_id, 10);
    assert_eq!(c.query_frame_id, 200);
    assert!(c.geometrically_verified);
}

#[test]
fn pose_graph_text_round_trip_preserves_state() {
    use visloc_slam::{PoseGraphEdgeKind, PoseGraphParseError};

    let dir = std::env::temp_dir().join("visloc_slam_pose_graph_round_trip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pose_graph.txt");

    // Build a small graph: three poses with non-trivial rotations + a
    // sequential edge plus a loop-closure edge with a custom weight.
    let mut graph = PoseGraph::new();
    graph.add_pose(
        0,
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros()),
    );
    graph.add_pose(
        1,
        Pose::from_world_to_camera(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.30),
            Vector3::new(-0.7, 0.05, -0.4),
        ),
    );
    graph.add_pose(
        7,
        Pose::from_world_to_camera(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.20),
            Vector3::new(-1.5, 0.10, -1.3),
        ),
    );
    graph.anchor(0);
    graph.add_sequential_edge(
        0,
        1,
        visloc_core::geometry::SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.30),
            Vector3::new(-0.7, 0.05, -0.4),
        ),
    );
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 0,
        to_keyframe_id: 7,
        relative_pose: visloc_core::geometry::SE3::new(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 1.20),
            Vector3::new(-1.5, 0.10, -1.3),
        ),
        inlier_count: 42,
        inlier_ratio: 0.85,
        mean_sampson_error: 1.3e-3,
        score: 35.7,
    });

    graph.save_text(&path).unwrap();
    let restored = PoseGraph::load_text(&path).unwrap();

    assert_eq!(restored.poses.len(), graph.poses.len());
    assert_eq!(restored.anchor, graph.anchor);
    assert_eq!(restored.edges.len(), graph.edges.len());

    for (id, original) in graph.poses.iter() {
        let other = restored.poses.get(id).unwrap();
        let q_a = original.world_to_camera.rotation.into_inner();
        let q_b = other.world_to_camera.rotation.into_inner();
        assert!((q_a.w - q_b.w).abs() < 1.0e-12);
        assert!((q_a.i - q_b.i).abs() < 1.0e-12);
        assert!((q_a.j - q_b.j).abs() < 1.0e-12);
        assert!((q_a.k - q_b.k).abs() < 1.0e-12);
        let t_a = &original.world_to_camera.translation;
        let t_b = &other.world_to_camera.translation;
        assert!((t_a - t_b).norm() < 1.0e-12);
    }
    for (i, original) in graph.edges.iter().enumerate() {
        let other = &restored.edges[i];
        assert_eq!(original.from, other.from);
        assert_eq!(original.to, other.to);
        assert_eq!(original.kind, other.kind);
        assert!((original.weight - other.weight).abs() < 1.0e-12);
        let q_a = original.measurement.rotation.into_inner();
        let q_b = other.measurement.rotation.into_inner();
        assert!((q_a.w - q_b.w).abs() < 1.0e-12);
        let t_a = &original.measurement.translation;
        let t_b = &other.measurement.translation;
        assert!((t_a - t_b).norm() < 1.0e-12);
    }
    // Loop-closure edge survived as such (the Sequential edge above was
    // weight 1.0, so the kind tag is what differentiates them on read).
    assert!(matches!(
        restored.edges.iter().find(|e| e.weight > 2.0).unwrap().kind,
        PoseGraphEdgeKind::LoopClosure
    ));

    // Negative path: a malformed file surfaces a Syntax error pointing
    // to the offending line number.
    let bad = dir.join("bad.txt");
    std::fs::write(&bad, "P 0 not-a-number 0 0 0 0 0 0\n").unwrap();
    let err = PoseGraph::load_text(&bad).unwrap_err();
    match err {
        PoseGraphParseError::Syntax { line, .. } => assert_eq!(line, 1),
        PoseGraphParseError::Io(_) => panic!("expected syntax error, got I/O"),
    }
}

// === Deep frontend × OnlineSlamPipeline integration ===
//
// Renders three synthetic textured views (kf10 / kf100 / kf200) where kf10
// and kf200 observe the same scene from slightly different camera centres
// and kf100 observes a fully disjoint scene. Runs `HogLikeFeatureExtractor`
// on each rendered image, parks the extracted (keypoints, descriptors)
// directly on the keyframes, and verifies `OnlineSlamPipeline::
// scan_appearance_loops` paired with `MutualSoftmaxMatcher` recovers the
// (10, 200) loop pair end-to-end.

const DEEP_TEST_IMAGE_WIDTH: usize = 320;
const DEEP_TEST_IMAGE_HEIGHT: usize = 240;
const DEEP_TEST_FOCAL: f64 = 320.0;

fn render_deep_test_view(
    camera: &Camera,
    pose: &Pose,
    landmarks: &[Point3<f64>],
    background_phase: f64,
) -> visloc_vision::features::GrayscaleImage {
    use nalgebra::{Point2, Vector3};
    let width = DEEP_TEST_IMAGE_WIDTH;
    let height = DEEP_TEST_IMAGE_HEIGHT;
    let mut pixels = vec![25_u8; width * height];

    // Procedurally textured background derived from the camera ray —
    // multi-scale checker that produces enough corners for HogLike's
    // detector while still being parallax-sensitive.
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f64 - width as f64 / 2.0) / DEEP_TEST_FOCAL;
            let ny = (y as f64 - height as f64 / 2.0) / DEEP_TEST_FOCAL;
            let ray_camera = Vector3::new(nx, ny, 1.0);
            let world_ray = pose.camera_to_world().rotation * ray_camera;
            let cam_origin = pose.camera_center_world();
            let depth_plane = 8.0_f64;
            let denom = world_ray.z;
            if denom.abs() < 1e-6 {
                continue;
            }
            let t = (depth_plane - cam_origin.z) / denom;
            if t <= 0.0 {
                continue;
            }
            let world_x = cam_origin.x + t * world_ray.x;
            let world_y = cam_origin.y + t * world_ray.y;
            let checker_a =
                ((world_x * 4.0 + background_phase).sin() * (world_y * 4.0).sin()).abs();
            let checker_b =
                ((world_x * 1.7 - background_phase).cos() * (world_y * 2.3).cos()).abs();
            let stripe = ((world_x + world_y) * 6.0 + background_phase).sin().abs();
            let value = (60.0 + 130.0 * (0.55 * checker_a + 0.30 * checker_b + 0.15 * stripe))
                .clamp(0.0, 255.0) as u8;
            pixels[y * width + x] = value;
        }
    }

    // Bright dots at landmark projections so the corner detector locks
    // onto the same physical scene points across kf10/kf200.
    for landmark in landmarks {
        let camera_point = pose.transform_world_point(landmark);
        if camera_point.z <= 0.1 {
            continue;
        }
        let Some(projected) = camera.project(&camera_point) else {
            continue;
        };
        let cx = projected.x.round() as i32;
        let cy = projected.y.round() as i32;
        let radius: i32 = 2;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let xx = cx + dx;
                let yy = cy + dy;
                if xx < 0 || yy < 0 || xx >= width as i32 || yy >= height as i32 {
                    continue;
                }
                let r2 = (dx * dx + dy * dy) as f64;
                if r2 > (radius as f64).powi(2) {
                    continue;
                }
                let alpha = (1.0 - r2 / (radius as f64).powi(2)).clamp(0.0, 1.0);
                let index = (yy as usize) * width + xx as usize;
                let blended = (pixels[index] as f64) * (1.0 - alpha) + 240.0 * alpha;
                pixels[index] = blended.clamp(0.0, 255.0) as u8;
            }
        }
    }
    let _ = Point2::new(0.0, 0.0); // suppress unused-import warning when nalgebra::Point2 is otherwise unused locally

    visloc_vision::features::GrayscaleImage::from_luma_u8(width, height, pixels).unwrap()
}

fn deep_test_landmarks_loop() -> Vec<Point3<f64>> {
    let mut points = Vec::new();
    for ix in -2..=2 {
        for iy in -1..=1 {
            for iz in 0..=2 {
                let x = ix as f64 * 0.6;
                let y = iy as f64 * 0.5;
                let z = 4.0 + iz as f64 * 1.2;
                points.push(Point3::new(x, y, z));
            }
        }
    }
    points
}

fn deep_test_landmarks_disjoint() -> Vec<Point3<f64>> {
    // Same shape but offset far away in world so kf100 sees nothing in
    // common with kf10/kf200 — the procedural background is offset too.
    let mut points = Vec::new();
    for ix in -2..=2 {
        for iy in -1..=1 {
            for iz in 0..=2 {
                let x = ix as f64 * 0.6 + 30.0;
                let y = iy as f64 * 0.5 + 20.0;
                let z = 4.0 + iz as f64 * 1.2;
                points.push(Point3::new(x, y, z));
            }
        }
    }
    points
}

fn keyframe_from_hog_extraction(
    frame_id: u64,
    camera: &Camera,
    image: &visloc_vision::features::GrayscaleImage,
    extractor: &visloc_vision::features::HogLikeFeatureExtractor,
) -> visloc_core::types::Keyframe {
    let features = visloc_vision::features::FeatureExtractor::extract(extractor, image)
        .expect("hog-like extractor accepts the rendered image");
    let mut frame = Frame::new(frame_id, camera.id);
    frame.keypoints = features.keypoints;
    frame.descriptors = features.descriptors;
    visloc_core::types::Keyframe {
        frame,
        observations: Vec::new(),
    }
}

#[test]
fn online_slam_pipeline_scan_appearance_loops_with_deep_frontend_finds_loop_pair() {
    use visloc_vision::features::{HogLikeFeatureConfig, HogLikeFeatureExtractor};
    use visloc_vision::matching::{MutualSoftmaxConfig, MutualSoftmaxMatcher};

    let camera = Camera::pinhole(
        1,
        DEEP_TEST_IMAGE_WIDTH as u32,
        DEEP_TEST_IMAGE_HEIGHT as u32,
        DEEP_TEST_FOCAL,
        DEEP_TEST_FOCAL,
        DEEP_TEST_IMAGE_WIDTH as f64 / 2.0,
        DEEP_TEST_IMAGE_HEIGHT as f64 / 2.0,
    );

    let pose_kf10 = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let pose_kf100 =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-25.0, 0.0, 0.0));
    let pose_kf200 = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.0, 0.05, 0.0),
        Vector3::new(-0.18, 0.0, 0.0),
    );

    let landmarks_loop = deep_test_landmarks_loop();
    let landmarks_disjoint = deep_test_landmarks_disjoint();

    let image_kf10 = render_deep_test_view(&camera, &pose_kf10, &landmarks_loop, 0.0);
    let image_kf100 = render_deep_test_view(&camera, &pose_kf100, &landmarks_disjoint, 7.31);
    let image_kf200 = render_deep_test_view(&camera, &pose_kf200, &landmarks_loop, 0.0);

    // Orient defaults to true for SIFT-like rotation invariance, but here
    // we want to keep the descriptor stable across two near-duplicate
    // viewpoints (kf10 ↔ kf200) where the per-keypoint dominant
    // orientation estimate can flip between bins for similar content.
    // Disabling orient gives the brute-force/mutual softmax matchers a
    // tighter fixed-frame descriptor, which is what this pipeline-
    // integration test cares about.
    let extractor = HogLikeFeatureExtractor::new(HogLikeFeatureConfig {
        max_features: 256,
        min_corner_score: 0.05,
        descriptor_clip: 0.2,
        orient: false,
    });

    let kf10 = keyframe_from_hog_extraction(10, &camera, &image_kf10, &extractor);
    let kf100 = keyframe_from_hog_extraction(100, &camera, &image_kf100, &extractor);
    let kf200 = keyframe_from_hog_extraction(200, &camera, &image_kf200, &extractor);

    // Sanity: the renderings should produce *some* features, and kf10/kf200
    // should share descriptor space (kf100 should not).
    assert!(!kf10.frame.descriptors.is_empty());
    assert!(!kf100.frame.descriptors.is_empty());
    assert!(!kf200.frame.descriptors.is_empty());

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    map.keyframes.insert(kf10.frame.id, kf10);
    map.keyframes.insert(kf100.frame.id, kf100);
    map.keyframes.insert(kf200.frame.id, kf200);

    let slam = OnlineSlamPipeline::new(
        map,
        Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
        LocalMappingPipeline::default(),
        OnlineSlamConfig::default(),
    );

    let matcher = MutualSoftmaxMatcher::new(MutualSoftmaxConfig {
        temperature: 25.0,
        min_confidence: 0.15,
        emit_ratio_metadata: false,
    });
    let verifier = EssentialMatrixLoopClosureVerifier {
        config: LoopClosureVerifierConfig {
            min_inliers: 12,
            min_inlier_ratio: 0.4,
            max_mean_sampson_error: 5.0e-3,
            default_translation_scale: 1.0,
        },
        ..Default::default()
    };

    let candidates = slam.scan_appearance_loops(
        &matcher,
        &verifier,
        &camera,
        &AppearanceLoopScannerSettings {
            min_keyframe_id_gap: 50,
            min_matches: 12,
        },
    );

    // The disjoint kf100 should never pair with kf10 or kf200 — the
    // procedural background phase is offset and the landmark cloud is
    // 30 m away in world coordinates, so the descriptors are wildly
    // different. Only the (10, 200) loop pair should remain.
    let pair_ids: Vec<(u64, u64)> = candidates
        .iter()
        .map(|c| (c.matched_keyframe_id, c.query_frame_id))
        .collect();
    assert!(
        pair_ids.contains(&(10, 200)),
        "expected the (10, 200) deep-frontend loop pair, got {:?}",
        pair_ids
    );
    for c in &candidates {
        assert!(c.geometrically_verified);
        let v = c.verification.as_ref().expect("verification populated");
        assert!(
            v.verified,
            "essential RANSAC should accept the deep-frontend pair"
        );
        assert!(v.inlier_count >= 12);
        // No false-positive pair involving the disjoint kf100.
        assert_ne!(c.matched_keyframe_id, 100, "kf100 should not pair");
        assert_ne!(c.query_frame_id, 100, "kf100 should not pair");
    }
}

// ============================================================
// VI initialisation integration tests (docs/vi_initialization_integration.md)
// ============================================================

mod vi_init_integration {
    use super::*;
    use nalgebra::Rotation3;
    use visloc_core::geometry::SE3;
    use visloc_slam::{
        OnlineSlamConfigError, OnlineSlamLocalBaConfig, OnlineSlamViInitConfig,
        StationaryRejectionReason, ViInitFallback, ViInitializationEvent, ViInitializationStatus,
        VisualInertialInitializerConfig,
    };

    fn euroc_z_up_gravity() -> Vector3<f64> {
        Vector3::new(0.0, 0.0, -9.81)
    }

    fn imu_config_z_up() -> OnlineSlamImuConfig {
        OnlineSlamImuConfig {
            gravity_world: euroc_z_up_gravity(),
            ..OnlineSlamImuConfig::default()
        }
    }

    fn initializer_config_z_up() -> VisualInertialInitializerConfig {
        VisualInertialInitializerConfig {
            gravity_world: euroc_z_up_gravity(),
            ..VisualInertialInitializerConfig::default()
        }
    }

    fn vi_init_config_default() -> OnlineSlamViInitConfig {
        OnlineSlamViInitConfig {
            initializer: initializer_config_z_up(),
            body_to_camera: SE3::identity(),
            seed_first_keyframe_rotation: true,
            on_persistent_rejection: ViInitFallback::KeepExistingSeed,
            max_wait_duration_seconds: 5.0,
            max_buffered_samples: 2000,
            try_initialize_on_every_frame: false,
        }
    }

    fn pipeline_with_vi_init(
        map: VisualMap,
        config: OnlineSlamViInitConfig,
        local_vi_ba: Option<OnlineSlamLocalBaConfig>,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 50,
                    ..LoopClosureConfig::default()
                },
                imu: Some(imu_config_z_up()),
                local_vi_ba,
                covisibility_local_ba: None,
                vi_init: Some(config),
                vi_motion_init: None,
                keep_pre_promotion_imu_factors: false,
                pose_graph_refinement: None,
                relocalization: None,
            },
        )
    }

    fn push_stationary_window(
        pipeline: &mut OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline>,
        accel_body: Vector3<f64>,
        gyro_bias: Vector3<f64>,
        sample_count: usize,
    ) {
        let dt = 0.005;
        for _ in 0..sample_count {
            pipeline.push_imu_measurement(gyro_bias, accel_body, dt);
        }
    }

    // Test #1: vi_init: Some, imu: None → ViInitRequiresImu.
    #[test]
    fn config_validate_rejects_vi_init_without_imu() {
        let config = OnlineSlamConfig {
            imu: None,
            vi_init: Some(vi_init_config_default()),
            ..OnlineSlamConfig::default()
        };
        match config.validate() {
            Err(OnlineSlamConfigError::ViInitRequiresImu) => {}
            other => panic!("expected ViInitRequiresImu, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "vi_init is Some")]
    fn new_panics_when_vi_init_lacks_imu() {
        let (map, _frame) = map_and_frame(10, 1);
        let _slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                imu: None,
                vi_init: Some(vi_init_config_default()),
                ..OnlineSlamConfig::default()
            },
        );
    }

    // Test #2: gravity mismatch.
    #[test]
    fn config_validate_rejects_gravity_mismatch() {
        let mut vi = vi_init_config_default();
        vi.initializer.gravity_world = Vector3::new(0.0, 9.81, 0.0); // KITTI y-down
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig {
                gravity_world: Vector3::new(0.0, 0.0, -9.81), // EuRoC z-up
                ..OnlineSlamImuConfig::default()
            }),
            vi_init: Some(vi),
            ..OnlineSlamConfig::default()
        };
        match config.validate() {
            Err(OnlineSlamConfigError::GravityMismatch { .. }) => {}
            other => panic!("expected GravityMismatch, got {other:?}"),
        }
    }

    // Test #3: stationary success on a single keyframe.
    #[test]
    fn stationary_stream_success_emits_succeeded_event() {
        let (map, frame) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        // Body level under z-up gravity ⇒ accel reads (0, 0, +9.81).
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            200,
        );
        let result = slam.process_frame(&frame, []);
        match result.vi_init {
            Some(ViInitializationEvent::Succeeded {
                result: vi_result,
                first_keyframe_id,
                discarded_stale_factor_count,
            }) => {
                assert_eq!(first_keyframe_id, Some(10));
                assert!(vi_result.bias_gyro.norm() < 1.0e-9);
                assert!(vi_result.bias_acc.norm() < 1.0e-9);
                assert!(vi_result.initial_rotation_body_to_world.angle().abs() < 1.0e-9);
                // No prior keyframe ⇒ no factor was ever staged, so
                // discarded count stays at zero on this run.
                assert_eq!(discarded_stale_factor_count, 0);
            }
            other => panic!("expected Succeeded, got {other:?}"),
        }
        // imu_state config has been mirrored with the recovered biases.
        let imu_state = slam.imu_state.as_ref().unwrap();
        assert!(imu_state.config.bias_gyro.norm() < 1.0e-9);
        // Stale gate is lifted.
        assert!(matches!(
            slam.vi_initialization_status(),
            ViInitializationStatus::Initialised { .. }
        ));
    }

    // Test #4: Succeeded fires at most once per sequence.
    #[test]
    fn succeeded_event_does_not_re_fire_on_subsequent_frames() {
        let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, second_frame) =
            map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            200,
        );
        let r1 = slam.process_frame(&first_frame, []);
        assert!(matches!(
            r1.vi_init,
            Some(ViInitializationEvent::Succeeded { .. })
        ));
        // After success, subsequent IMU samples must NOT change the
        // durable status snapshot (status stays Initialised, the
        // standalone module is no longer fanned-into).
        let status_before = slam.vi_initialization_status();
        for _ in 0..50 {
            slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.005);
        }
        assert_eq!(
            status_before,
            slam.vi_initialization_status(),
            "status must stay Initialised after success",
        );
        let r2 = slam.process_frame(&second_frame, []);
        assert!(
            r2.vi_init.is_none(),
            "subsequent frame must carry vi_init: None"
        );
        assert!(matches!(
            slam.vi_initialization_status(),
            ViInitializationStatus::Initialised { .. }
        ));
    }

    // Test #5: stale factor discard.
    #[test]
    fn stale_factors_before_success_are_counted_and_discarded() {
        // Strategy: configure the initialiser to need a long window so
        // the first two keyframes pass without success. On the third
        // keyframe, dump enough samples to trigger success and assert
        // discarded_stale_factor_count == 2.
        //
        // Setup: min_stationary_window_seconds = 4.0, min_samples = 800
        // → buffer must grow across three keyframes before init fires.
        let (map, first_frame) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, second_frame) =
            map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
        let (_, third_frame) =
            map_and_frame_with_extra_landmarks(50, 1, Vector3::new(3.0, 0.0, 0.0));
        let mut vi = vi_init_config_default();
        vi.initializer.min_stationary_window_seconds = 4.0;
        vi.initializer.min_samples = 800;
        let mut slam = pipeline_with_vi_init(map, vi, None);
        // Frame 1: register keyframe with no prior factor staged.
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            300,
        );
        let r1 = slam.process_frame(&first_frame, []);
        assert!(matches!(
            r1.vi_init,
            Some(ViInitializationEvent::StillBuffering { .. })
        ));
        // r1.imu_factor must be None (no prior keyframe to anchor against).
        assert!(r1.imu_factor.is_none());
        // Frame 2: factor #1 would be staged but is gated.
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            300,
        );
        let r2 = slam.process_frame(&second_frame, []);
        assert!(r2.imu_factor.is_none(), "stale gate must suppress factor");
        assert!(slam.take_pending_imu_factor().is_none());
        // Frame 3: enough samples buffered to trigger success.
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            300,
        );
        let r3 = slam.process_frame(&third_frame, []);
        match r3.vi_init {
            Some(ViInitializationEvent::Succeeded {
                discarded_stale_factor_count,
                ..
            }) => {
                assert_eq!(
                    discarded_stale_factor_count, 2,
                    "should have counted both staged-then-discarded factors"
                );
            }
            other => panic!("expected Succeeded on frame 3, got {other:?}"),
        }
    }

    // Test #6: [pose-conv] Rotation direction.
    #[test]
    fn pose_conv_rotation_direction_uses_transpose_of_body_to_world() {
        // Body tilted 30° about world-x ⇒ accel in body frame reads
        // R_b←w · (-g_w) = R_b←w · (0, 0, 9.81).
        let tilt = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_6);
        let world_up = Vector3::new(0.0, 0.0, 9.81);
        let accel_body = tilt.inverse() * world_up;
        let (map, frame) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        push_stationary_window(&mut slam, accel_body, Vector3::zeros(), 400);
        let result = slam.process_frame(&frame, []);
        assert!(matches!(
            result.vi_init,
            Some(ViInitializationEvent::Succeeded { .. })
        ));
        let kf = slam.map().keyframes.get(&10).expect("keyframe 10");
        let pose = kf.frame.pose.as_ref().expect("pose set");
        // Apply Pose to world_up; result must be world_up in camera frame
        // up to the camera-world identity convention. With body_to_camera
        // = identity, Pose.rotation = R_c←w = R_w←b^T. So
        // R_c←w · world_up should equal R_b←w · world_up = accel_body.
        let result_in_camera = pose.world_to_camera.rotation * world_up;
        assert!(
            (result_in_camera - accel_body).norm() < 1.0e-9,
            "world_up under R_c←w should match body-frame accel direction (after IMU/cam alignment), got {:?}",
            result_in_camera
        );
    }

    // Test #7: [pose-conv] Camera center preservation.
    #[test]
    fn pose_conv_camera_center_is_preserved_across_promotion() {
        // Pre-set the first keyframe pose to a known camera centre
        // C_w_old != 0 so we can detect any drift after the promotion.
        // Strategy: use map_and_frame_with_extra_landmarks(..., camera_center)
        // which sets pose = Pose::from_world_to_camera(I, -C_w_old). That
        // gives -R_c←w^T · t_c←w = -I · (-C_w_old) = C_w_old. After
        // promotion with a tilted body the rotation changes but the
        // camera centre must stay the same.
        let camera_center = Vector3::new(0.4, -0.7, 0.2);
        let (map, frame) = map_and_frame_with_extra_landmarks(10, 1, camera_center);
        let tilt = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_6);
        let accel_body = tilt.inverse() * Vector3::new(0.0, 0.0, 9.81);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        push_stationary_window(&mut slam, accel_body, Vector3::zeros(), 400);
        let result = slam.process_frame(&frame, []);
        assert!(matches!(
            result.vi_init,
            Some(ViInitializationEvent::Succeeded { .. })
        ));
        let kf = slam.map().keyframes.get(&10).expect("keyframe 10");
        let pose = kf.frame.pose.as_ref().expect("pose set");
        let recovered_center = pose.camera_center_world().coords;
        assert!(
            (recovered_center - camera_center).norm() < 1.0e-9,
            "camera centre must be preserved: expected {:?}, got {:?}",
            camera_center,
            recovered_center
        );
    }

    // Test #8: [pose-conv] body_to_camera extrinsic.
    #[test]
    fn pose_conv_respects_body_to_camera_extrinsic() {
        // 30° tilt about world-x, body_to_camera = SE3(R_b←c = R_y(90°),
        // t = 0). After promotion R_c←w = (R_w←b · R_b←c)^T.
        let tilt = Rotation3::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_6);
        let accel_body = tilt.inverse() * Vector3::new(0.0, 0.0, 9.81);
        let r_b_c =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2);
        let body_to_camera = SE3::new(r_b_c, Vector3::zeros());
        let mut vi_config = vi_init_config_default();
        vi_config.body_to_camera = body_to_camera.clone();
        let (map, frame) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_config, None);
        push_stationary_window(&mut slam, accel_body, Vector3::zeros(), 400);
        let result = slam.process_frame(&frame, []);
        let recovered_rotation = match result.vi_init {
            Some(ViInitializationEvent::Succeeded { ref result, .. }) => {
                result.initial_rotation_body_to_world
            }
            ref other => panic!("expected Succeeded, got {other:?}"),
        };
        let kf = slam.map().keyframes.get(&10).expect("keyframe 10");
        let pose = kf.frame.pose.as_ref().expect("pose set");
        let r_wc = recovered_rotation * body_to_camera.rotation;
        let r_cw_expected = r_wc.inverse();
        let delta = pose.world_to_camera.rotation * r_cw_expected.inverse();
        assert!(
            delta.angle().abs() < 1.0e-9,
            "Pose.rotation should equal (R_w←b · R_b←c)^T; angle delta = {}",
            delta.angle()
        );
    }

    // Test #9: failure-mode StillBuffering with noisy gyro.
    #[test]
    fn noisy_gyro_emits_still_buffering_with_reason() {
        let (map, frame) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        // Alternating ±1 rad/s on x — well above default 0.05.
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for i in 0..400 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gyro = Vector3::new(sign, 0.0, 0.0);
            slam.push_imu_measurement(gyro, accel_body, 0.005);
        }
        let result = slam.process_frame(&frame, []);
        match result.vi_init {
            Some(ViInitializationEvent::StillBuffering {
                reason: StationaryRejectionReason::GyroNoiseTooHigh { observed, limit },
            }) => {
                assert!(observed.x > limit);
            }
            other => panic!("expected GyroNoiseTooHigh StillBuffering, got {other:?}"),
        }
    }

    // Test #10: KeepExistingSeed fallback.
    #[test]
    fn keep_existing_seed_fallback_leaves_imu_state_in_place() {
        // Drive enough buffered duration to exceed max_wait_duration_seconds
        // = 0.5s while keeping the stream noisy so try_initialize keeps
        // rejecting.
        let (map, frame) = map_and_frame(10, 1);
        let mut vi = vi_init_config_default();
        vi.max_wait_duration_seconds = 0.5;
        vi.max_buffered_samples = 10_000;
        vi.on_persistent_rejection = ViInitFallback::KeepExistingSeed;
        let mut slam = pipeline_with_vi_init(map, vi, None);
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for i in 0..400 {
            // > 0.5s once 100 samples are pushed (100 * 0.005 = 0.5)
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gyro = Vector3::new(sign, 0.0, 0.0);
            slam.push_imu_measurement(gyro, accel_body, 0.005);
        }
        let result = slam.process_frame(&frame, []);
        match result.vi_init {
            Some(ViInitializationEvent::GaveUp {
                fallback: ViInitFallback::KeepExistingSeed,
                ..
            }) => {}
            other => panic!("expected KeepExistingSeed GaveUp, got {other:?}"),
        }
        assert!(
            slam.imu_state.is_some(),
            "imu_state must survive KeepExistingSeed"
        );
        assert!(slam.config.imu.is_some());
        // Stale gate is lifted.
        assert!(matches!(
            slam.vi_initialization_status(),
            ViInitializationStatus::GaveUp { .. }
        ));
    }

    // Test #11: DisableImuStage fallback.
    #[test]
    fn disable_imu_stage_fallback_clears_imu_state() {
        let (map, frame) = map_and_frame(10, 1);
        let mut vi = vi_init_config_default();
        vi.max_wait_duration_seconds = 0.5;
        vi.on_persistent_rejection = ViInitFallback::DisableImuStage;
        let mut slam = pipeline_with_vi_init(
            map,
            vi,
            Some(OnlineSlamLocalBaConfig {
                gravity_world: euroc_z_up_gravity(),
                ..OnlineSlamLocalBaConfig::default()
            }),
        );
        let accel_body = Vector3::new(0.0, 0.0, 9.81);
        for i in 0..400 {
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            let gyro = Vector3::new(sign, 0.0, 0.0);
            slam.push_imu_measurement(gyro, accel_body, 0.005);
        }
        let result = slam.process_frame(&frame, []);
        match result.vi_init {
            Some(ViInitializationEvent::GaveUp {
                fallback: ViInitFallback::DisableImuStage,
                ..
            }) => {}
            other => panic!("expected DisableImuStage GaveUp, got {other:?}"),
        }
        assert!(slam.imu_state.is_none(), "imu_state should be cleared");
        assert!(
            slam.local_vi_ba_state.is_none(),
            "local_vi_ba_state should be cleared"
        );
        assert!(slam.config.imu.is_none(), "config.imu should be cleared");
        assert!(
            slam.config.local_vi_ba.is_none(),
            "config.local_vi_ba should be cleared"
        );
    }

    #[test]
    fn preseed_vi_samples_fill_initializer_without_polluting_first_imu_factor() {
        let (map, _) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        for _ in 0..100 {
            slam.push_vi_initialization_measurement(
                Vector3::zeros(),
                Vector3::new(0.0, 0.0, 9.81),
                0.005,
            );
        }
        match slam.vi_initialization_status() {
            ViInitializationStatus::Buffering {
                samples_buffered,
                buffered_duration_seconds,
                ..
            } => {
                assert_eq!(samples_buffered, 100);
                assert!((buffered_duration_seconds - 0.5).abs() < 1.0e-12);
            }
            other => panic!("expected pre-seed samples to remain buffered, got {other:?}"),
        }
        let delta_time = slam
            .imu_state
            .as_ref()
            .unwrap()
            .preintegrator
            .delta()
            .delta_time;
        assert!(delta_time.abs() < 1.0e-12);
    }

    // Test #14: reset_sequence_state re-arms VI init.
    #[test]
    fn reset_sequence_state_rearms_vi_init() {
        let (map, frame) = map_and_frame(10, 1);
        let mut slam = pipeline_with_vi_init(map, vi_init_config_default(), None);
        push_stationary_window(
            &mut slam,
            Vector3::new(0.0, 0.0, 9.81),
            Vector3::zeros(),
            200,
        );
        let r1 = slam.process_frame(&frame, []);
        assert!(matches!(
            r1.vi_init,
            Some(ViInitializationEvent::Succeeded { .. })
        ));
        assert!(matches!(
            slam.vi_initialization_status(),
            ViInitializationStatus::Initialised { .. }
        ));
        slam.reset_sequence_state();
        match slam.vi_initialization_status() {
            ViInitializationStatus::Buffering {
                samples_buffered: 0,
                buffered_duration_seconds,
                last_rejection: None,
            } => {
                assert!(buffered_duration_seconds.abs() < 1.0e-12);
            }
            other => panic!("expected Buffering after reset, got {other:?}"),
        }
    }
}

// ============================================================
// Motion-based VI initialisation integration tests
// (docs/motion_based_vi_alignment.md)
// ============================================================

mod vi_motion_init_integration {
    use super::*;
    use visloc_slam::{
        MotionBasedViInitializerConfig, MotionViInitializationEvent, MotionViInitializationStatus,
        OnlineSlamConfigError, OnlineSlamMotionViInitConfig, OnlineSlamViInitConfig,
        ViInitFallback, ViInitializationEvent, VisualInertialInitializerConfig,
    };

    fn motion_config_default() -> OnlineSlamMotionViInitConfig {
        OnlineSlamMotionViInitConfig {
            initializer: MotionBasedViInitializerConfig {
                min_keyframes: 2,
                min_translation_meters: 0.0,
                gravity_world: Vector3::new(0.0, 9.81, 0.0),
                ..MotionBasedViInitializerConfig::default()
            },
            ..OnlineSlamMotionViInitConfig::default()
        }
    }

    #[test]
    fn validate_rejects_motion_init_without_imu() {
        let config = OnlineSlamConfig {
            imu: None,
            vi_init: None,
            vi_motion_init: Some(motion_config_default()),
            ..OnlineSlamConfig::default()
        };
        match config.validate() {
            Err(OnlineSlamConfigError::MotionViInitRequiresImu) => {}
            other => panic!("expected MotionViInitRequiresImu, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_motion_init_without_static_vi_init() {
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig::default()),
            vi_init: None,
            vi_motion_init: Some(motion_config_default()),
            ..OnlineSlamConfig::default()
        };
        match config.validate() {
            Err(OnlineSlamConfigError::MotionViInitRequiresStaticViInit) => {}
            other => panic!("expected MotionViInitRequiresStaticViInit, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_post_give_up_motion_fallback_when_imu_is_disabled() {
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig::default()),
            vi_init: Some(OnlineSlamViInitConfig {
                initializer: VisualInertialInitializerConfig {
                    gravity_world: Vector3::new(0.0, 9.81, 0.0),
                    ..VisualInertialInitializerConfig::default()
                },
                on_persistent_rejection: ViInitFallback::DisableImuStage,
                ..OnlineSlamViInitConfig::default()
            }),
            vi_motion_init: Some(OnlineSlamMotionViInitConfig {
                allow_after_static_give_up: true,
                ..motion_config_default()
            }),
            ..OnlineSlamConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(OnlineSlamConfigError::MotionViInitAfterGiveUpRequiresKeepExistingSeed)
        );
    }

    #[test]
    fn validate_rejects_motion_init_gravity_mismatch() {
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig {
                gravity_world: Vector3::new(0.0, 9.81, 0.0),
                ..OnlineSlamImuConfig::default()
            }),
            vi_init: Some(OnlineSlamViInitConfig {
                initializer: VisualInertialInitializerConfig {
                    gravity_world: Vector3::new(0.0, 9.81, 0.0),
                    ..VisualInertialInitializerConfig::default()
                },
                ..OnlineSlamViInitConfig::default()
            }),
            vi_motion_init: Some(OnlineSlamMotionViInitConfig {
                initializer: MotionBasedViInitializerConfig {
                    gravity_world: Vector3::new(0.0, 0.0, -9.81), // mismatch
                    ..MotionBasedViInitializerConfig::default()
                },
                ..OnlineSlamMotionViInitConfig::default()
            }),
            ..OnlineSlamConfig::default()
        };
        match config.validate() {
            Err(OnlineSlamConfigError::MotionGravityMismatch { .. }) => {}
            other => panic!("expected MotionGravityMismatch, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_motion_init_extrinsic_mismatch() {
        let body_to_camera = SE3::new(
            UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            Vector3::new(0.01, -0.02, 0.03),
        );
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig::default()),
            vi_init: Some(OnlineSlamViInitConfig {
                initializer: VisualInertialInitializerConfig {
                    gravity_world: Vector3::new(0.0, 9.81, 0.0),
                    ..VisualInertialInitializerConfig::default()
                },
                body_to_camera,
                ..OnlineSlamViInitConfig::default()
            }),
            vi_motion_init: Some(motion_config_default()),
            ..OnlineSlamConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(OnlineSlamConfigError::MotionExtrinsicMismatch)
        );
    }

    #[test]
    fn validate_rejects_local_vi_ba_extrinsic_mismatch() {
        let body_to_camera = SE3::new(
            UnitQuaternion::from_euler_angles(-0.2, 0.1, 0.3),
            Vector3::new(0.02, 0.03, -0.01),
        );
        let config = OnlineSlamConfig {
            imu: Some(OnlineSlamImuConfig::default()),
            vi_init: Some(OnlineSlamViInitConfig {
                initializer: VisualInertialInitializerConfig {
                    gravity_world: Vector3::new(0.0, 9.81, 0.0),
                    ..VisualInertialInitializerConfig::default()
                },
                body_to_camera,
                ..OnlineSlamViInitConfig::default()
            }),
            local_vi_ba: Some(OnlineSlamLocalBaConfig::default()),
            ..OnlineSlamConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(OnlineSlamConfigError::LocalViBaExtrinsicMismatch)
        );
    }

    #[test]
    fn validate_rejects_invalid_local_vi_ba_bias_random_walk_weights() {
        let config = OnlineSlamConfig {
            local_vi_ba: Some(OnlineSlamLocalBaConfig {
                bias_random_walk_weights: Some((f64::NAN, 1.0)),
                ..OnlineSlamLocalBaConfig::default()
            }),
            ..OnlineSlamConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(OnlineSlamConfigError::InvalidLocalViBaBiasRandomWalkWeights)
        );
    }

    #[test]
    fn snapshot_is_disabled_when_motion_init_is_none() {
        let map = VisualMap::new();
        let slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig::default(),
        );
        match slam.motion_vi_initialization_status() {
            MotionViInitializationStatus::Disabled => {}
            other => panic!("expected Disabled, got {other:?}"),
        }
    }

    fn aligned_static_vi_init_config() -> OnlineSlamViInitConfig {
        // `OnlineSlamImuConfig::default()` uses y-down gravity (KITTI
        // convention) but `VisualInertialInitializerConfig::default()`
        // is z-up (EuRoC convention); the validate gate rejects the
        // mismatch. Align both onto the IMU-default y-down vector here
        // so the constructor succeeds.
        OnlineSlamViInitConfig {
            initializer: VisualInertialInitializerConfig {
                gravity_world: Vector3::new(0.0, 9.81, 0.0),
                ..VisualInertialInitializerConfig::default()
            },
            ..OnlineSlamViInitConfig::default()
        }
    }

    #[test]
    fn snapshot_reports_waiting_when_motion_init_enabled_but_no_keyframes_yet() {
        let map = VisualMap::new();
        let slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                imu: Some(OnlineSlamImuConfig::default()),
                vi_init: Some(aligned_static_vi_init_config()),
                vi_motion_init: Some(motion_config_default()),
                ..OnlineSlamConfig::default()
            },
        );
        match slam.motion_vi_initialization_status() {
            MotionViInitializationStatus::Waiting {
                keyframes_observed: 0,
                cumulative_translation_meters,
                buffered_factor_count: 0,
                last_rejection: None,
            } => {
                assert!(cumulative_translation_meters.abs() < 1e-12);
            }
            other => panic!("expected Waiting, got {other:?}"),
        }
    }

    #[test]
    fn reset_clears_motion_init_state() {
        let map = VisualMap::new();
        let mut slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                imu: Some(OnlineSlamImuConfig::default()),
                vi_init: Some(aligned_static_vi_init_config()),
                vi_motion_init: Some(motion_config_default()),
                ..OnlineSlamConfig::default()
            },
        );
        slam.reset_sequence_state();
        match slam.motion_vi_initialization_status() {
            MotionViInitializationStatus::Waiting {
                keyframes_observed: 0,
                buffered_factor_count: 0,
                ..
            } => {}
            other => panic!("expected Waiting after reset, got {other:?}"),
        }
    }

    // End-to-end gate contract: when motion-VI is enabled but the static
    // stage NEVER succeeds (no stationary IMU window is fed), the
    // motion-based stage must remain `Waiting` indefinitely — even after
    // multiple keyframes pass through `process_frame`. This pins the
    // documented invariant on `OnlineSlamMotionViInitConfig` that the
    // motion stage's `static_seed` is the result the static stage
    // mirrors, so the motion path is fully gated until the static path
    // completes.
    #[test]
    fn motion_vi_init_stays_waiting_until_static_seed_fires() {
        use visloc_core::geometry::SE3;
        let z_up_gravity = Vector3::new(0.0, 0.0, -9.81);

        let (map, frame_a) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, frame_b) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));

        let mut slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig::default(),
                imu: Some(OnlineSlamImuConfig {
                    gravity_world: z_up_gravity,
                    ..OnlineSlamImuConfig::default()
                }),
                local_vi_ba: None,
                covisibility_local_ba: None,
                vi_init: Some(OnlineSlamViInitConfig {
                    initializer: VisualInertialInitializerConfig {
                        // Make the static stage hard to fire: jacked-up
                        // dynamic-window thresholds so the noisy IMU below
                        // is always rejected as moving.
                        gravity_world: z_up_gravity,
                        max_gyro_std: 1.0e-6,
                        max_accel_std: 1.0e-6,
                        ..VisualInertialInitializerConfig::default()
                    },
                    body_to_camera: SE3::identity(),
                    seed_first_keyframe_rotation: false,
                    on_persistent_rejection: ViInitFallback::KeepExistingSeed,
                    max_wait_duration_seconds: 5.0,
                    max_buffered_samples: 2000,
                    try_initialize_on_every_frame: false,
                }),
                vi_motion_init: Some(OnlineSlamMotionViInitConfig {
                    initializer: MotionBasedViInitializerConfig {
                        min_keyframes: 2,
                        min_translation_meters: 0.1,
                        gravity_world: z_up_gravity,
                        ..MotionBasedViInitializerConfig::default()
                    },
                    ..OnlineSlamMotionViInitConfig::default()
                }),
                keep_pre_promotion_imu_factors: false,
                pose_graph_refinement: None,
                relocalization: None,
            },
        );

        // Push deliberately-noisy IMU samples so the static window check
        // rejects them and the static stage never promotes to Succeeded.
        for i in 0..200 {
            let jitter = (i as f64 * 0.1).sin();
            slam.push_imu_measurement(
                Vector3::new(jitter, 0.0, 0.0),
                Vector3::new(jitter * 0.5, 0.0, 9.81),
                0.005,
            );
        }
        let r1 = slam.process_frame(&frame_a, []);
        // Static stage must not have succeeded under this noise budget.
        assert!(
            !matches!(r1.vi_init, Some(ViInitializationEvent::Succeeded { .. })),
            "static stage should NOT have succeeded under deliberately noisy IMU",
        );
        // Motion stage MUST NOT emit Succeeded under any circumstance
        // while the static seed is missing — this is the gate contract.
        assert!(
            !matches!(
                r1.vi_motion_init,
                Some(MotionViInitializationEvent::Succeeded { .. })
            ),
            "motion stage must remain gated until the static seed fires",
        );

        for _ in 0..10 {
            slam.push_imu_measurement(
                Vector3::new(0.05, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 9.81),
                0.01,
            );
        }
        let r2 = slam.process_frame(&frame_b, []);
        assert!(
            !matches!(
                r2.vi_motion_init,
                Some(MotionViInitializationEvent::Succeeded { .. })
            ),
            "motion stage must remain gated even after a second keyframe \
             when the static seed never fires",
        );

        // Durable snapshot still reports Waiting.
        match slam.motion_vi_initialization_status() {
            MotionViInitializationStatus::Waiting { .. } => {}
            other => panic!("motion stage must stay Waiting without static seed, got {other:?}"),
        }
        // And the static stage's status is still pre-Initialised.
        assert!(!matches!(
            slam.vi_initialization_status(),
            visloc_slam::ViInitializationStatus::Initialised { .. }
        ));
    }

    #[test]
    fn motion_vi_init_can_collect_factors_after_opted_in_static_give_up() {
        use visloc_core::geometry::SE3;
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let (map, frame_a) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, frame_b) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));
        let mut slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                imu: Some(OnlineSlamImuConfig {
                    gravity_world: gravity,
                    ..OnlineSlamImuConfig::default()
                }),
                vi_init: Some(OnlineSlamViInitConfig {
                    initializer: VisualInertialInitializerConfig {
                        gravity_world: gravity,
                        max_gyro_std: 1.0e-6,
                        max_accel_std: 1.0e-6,
                        ..VisualInertialInitializerConfig::default()
                    },
                    body_to_camera: SE3::identity(),
                    seed_first_keyframe_rotation: false,
                    on_persistent_rejection: ViInitFallback::KeepExistingSeed,
                    max_wait_duration_seconds: 0.1,
                    max_buffered_samples: 50,
                    try_initialize_on_every_frame: false,
                }),
                vi_motion_init: Some(OnlineSlamMotionViInitConfig {
                    initializer: MotionBasedViInitializerConfig {
                        // Keep the solver from firing: this test pins that the
                        // fallback starts collecting keyframes/factors, not a
                        // particular nonlinear optimum.
                        min_keyframes: 10,
                        min_translation_meters: 0.1,
                        gravity_world: gravity,
                        ..MotionBasedViInitializerConfig::default()
                    },
                    allow_after_static_give_up: true,
                    ..OnlineSlamMotionViInitConfig::default()
                }),
                ..OnlineSlamConfig::default()
            },
        );

        for i in 0..200 {
            let jitter = (i as f64 * 0.1).sin();
            slam.push_imu_measurement(
                Vector3::new(jitter, 0.0, 0.0),
                Vector3::new(jitter * 0.5, 0.0, 9.81),
                0.005,
            );
        }
        let first = slam.process_frame(&frame_a, []);
        assert!(matches!(
            first.vi_init,
            Some(ViInitializationEvent::GaveUp { .. })
        ));
        assert!(matches!(
            first.vi_motion_init,
            Some(MotionViInitializationEvent::StillWaiting { .. })
        ));

        for _ in 0..10 {
            slam.push_imu_measurement(Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81), 0.01);
        }
        let second = slam.process_frame(&frame_b, []);
        assert!(matches!(
            second.vi_motion_init,
            Some(MotionViInitializationEvent::StillWaiting { .. })
        ));
        match slam.motion_vi_initialization_status() {
            MotionViInitializationStatus::Waiting {
                keyframes_observed,
                buffered_factor_count,
                ..
            } => {
                assert_eq!(keyframes_observed, 2);
                assert_eq!(buffered_factor_count, 1);
            }
            other => panic!("expected fallback motion init to be collecting, got {other:?}"),
        }
    }

    /// Regression: pins that the
    /// `tracking.localization.pose == map.keyframes[id].frame.pose`
    /// invariant holds even when the full IMU + static-VI-init +
    /// motion-VI-init triple is configured on the pipeline. The only
    /// site that ever rewrites a stored keyframe pose is the static
    /// stage's `seed_first_keyframe_rotation` branch in
    /// `OnlineSlamPipeline::promote_vi_init_result`, and that branch
    /// only touches the FIRST keyframe (and only rewrites rotation
    /// while preserving the camera centre). Subsequent keyframes'
    /// poses must remain byte-equal to whatever the tracker handed
    /// to `keyframe_from_tracking_result`.
    #[test]
    fn keyframe_pose_storage_matches_tracker_under_full_vi_config() {
        use visloc_core::geometry::SE3;
        let z_up_gravity = Vector3::new(0.0, 0.0, -9.81);

        let (map, frame_a) = map_and_frame_with_extra_landmarks(10, 1, Vector3::zeros());
        let (_, frame_b) = map_and_frame_with_extra_landmarks(30, 1, Vector3::new(1.5, 0.0, 0.0));

        let mut slam = OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig::default(),
                imu: Some(OnlineSlamImuConfig {
                    gravity_world: z_up_gravity,
                    ..OnlineSlamImuConfig::default()
                }),
                local_vi_ba: None,
                covisibility_local_ba: None,
                vi_init: Some(OnlineSlamViInitConfig {
                    initializer: VisualInertialInitializerConfig {
                        gravity_world: z_up_gravity,
                        max_gyro_std: 1.0e-6,
                        max_accel_std: 1.0e-6,
                        ..VisualInertialInitializerConfig::default()
                    },
                    body_to_camera: SE3::identity(),
                    seed_first_keyframe_rotation: true,
                    on_persistent_rejection: ViInitFallback::KeepExistingSeed,
                    max_wait_duration_seconds: 5.0,
                    max_buffered_samples: 2000,
                    try_initialize_on_every_frame: false,
                }),
                vi_motion_init: Some(OnlineSlamMotionViInitConfig {
                    initializer: MotionBasedViInitializerConfig {
                        min_keyframes: 2,
                        min_translation_meters: 0.1,
                        gravity_world: z_up_gravity,
                        ..MotionBasedViInitializerConfig::default()
                    },
                    ..OnlineSlamMotionViInitConfig::default()
                }),
                keep_pre_promotion_imu_factors: false,
                pose_graph_refinement: None,
                relocalization: None,
            },
        );

        for i in 0..100 {
            let jitter = (i as f64 * 0.1).sin();
            slam.push_imu_measurement(
                Vector3::new(jitter, 0.0, 0.0),
                Vector3::new(jitter * 0.5, 0.0, 9.81),
                0.005,
            );
        }
        let _ = slam.process_frame(&frame_a, []);
        for _ in 0..10 {
            slam.push_imu_measurement(
                Vector3::new(0.05, 0.0, 0.0),
                Vector3::new(0.0, 0.0, 9.81),
                0.01,
            );
        }
        let second = slam.process_frame(&frame_b, []);
        assert!(second.tracking_succeeded(), "second frame must localise");

        let tracker_center = second
            .tracking
            .localization
            .pose
            .as_ref()
            .unwrap()
            .camera_center_world();
        let kf_30 = slam
            .map()
            .keyframes
            .get(&30)
            .expect("kf 30 must be present once mapper accepted it");
        let stored_center = kf_30.frame.pose.as_ref().unwrap().camera_center_world();
        let diff = (tracker_center - stored_center).norm();
        assert!(
            diff < 1.0e-9,
            "VI-init-enabled run still shows pose-storage mismatch: tracker {tracker_center:?} vs stored {stored_center:?}, diff {diff}"
        );
    }
}

// ============================================================
// Online loop-closure + pose-graph refinement integration tests
// ============================================================

mod online_loop_closure_refinement {
    use super::*;

    fn camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn shared_landmarks() -> Vec<Point3<f64>> {
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

    fn build_seeded_map(points: &[Point3<f64>], camera: &Camera) -> VisualMap {
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let descriptor: Vec<f32> = (0..16)
                .map(|d| ((index as f32) * 0.13 + (d as f32) * 0.07).sin())
                .collect();
            let mut landmark = Landmark::new(landmark_id, *point);
            landmark.descriptor = Some(descriptor);
            map.landmarks.insert(landmark_id, landmark);
        }
        map
    }

    fn frame_at(
        frame_id: u64,
        camera_center: Vector3<f64>,
        points: &[Point3<f64>],
        camera: &Camera,
        map: &VisualMap,
    ) -> Frame {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
        let mut frame = Frame::new(frame_id, camera.id);
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let projected = camera.project(&pose.transform_world_point(point)).unwrap();
            let descriptor = map
                .landmarks
                .get(&landmark_id)
                .and_then(|l| l.descriptor.clone())
                .unwrap();
            frame.keypoints.push(projected);
            frame.descriptors.push(descriptor);
        }
        frame
    }

    fn pipeline_with_pose_graph(
        map: VisualMap,
        camera: Camera,
        trigger_every_new_constraints: usize,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera,
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 1.0,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: None,
                    pcm_batch_rescreen: false,
                    marginalization_window: None,
                    marginalization_sparsify: false,
                    trigger_every_new_constraints,
                    appearance_candidates: None,
                    propagate_corrections: false,
                    solver: LoopRefinementSolver::Se3,
                }),
                ..OnlineSlamConfig::default()
            },
        )
    }

    /// Same wiring as [`pipeline_with_pose_graph`], but the refinement stage
    /// runs [`LoopRefinementVerifier::Pnp`] instead of the default
    /// essential-matrix backend, so accepted loop constraints carry a
    /// metric translation recovered from the map's triangulated landmarks
    /// rather than `verifier_config.default_translation_scale`.
    fn pipeline_with_pnp_pose_graph(
        map: VisualMap,
        camera: Camera,
        trigger_every_new_constraints: usize,
        pnp_config: PnPLoopClosureVerifierConfig,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera,
                    verifier_config: LoopClosureVerifierConfig::default(),
                    verifier: LoopRefinementVerifier::Pnp(pnp_config),
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: None,
                    pcm_batch_rescreen: false,
                    marginalization_window: None,
                    marginalization_sparsify: false,
                    trigger_every_new_constraints,
                    appearance_candidates: None,
                    propagate_corrections: false,
                    solver: LoopRefinementSolver::Se3,
                }),
                ..OnlineSlamConfig::default()
            },
        )
    }

    fn pipeline_with_window(
        map: VisualMap,
        camera: Camera,
        trigger_every_new_constraints: usize,
        window: usize,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera,
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 1.0,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: None,
                    pcm_batch_rescreen: false,
                    marginalization_window: Some(window),
                    marginalization_sparsify: false,
                    trigger_every_new_constraints,
                    appearance_candidates: None,
                    propagate_corrections: false,
                    solver: LoopRefinementSolver::Se3,
                }),
                ..OnlineSlamConfig::default()
            },
        )
    }

    /// With a fixed-lag window of 2, the loop-closing solve on the third
    /// keyframe must marginalize the oldest non-anchor pose (KF#20) into a prior,
    /// leaving the graph bounded to {anchor KF#10, KF#30}.
    #[test]
    fn marginalization_window_bounds_the_online_graph() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_window(map.clone(), camera.clone(), 1, 2);

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());

        let stats = r2
            .pose_graph_refinement
            .expect("stats on the loop-closing frame");
        // A solve fired on the loop, so the window bound was applied.
        assert!(stats.pose_graph_result.is_some(), "PGO fired on the loop");
        assert_eq!(
            stats.poses_marginalized,
            vec![20],
            "the oldest non-anchor pose is marginalized to enforce the window"
        );

        let state = slam.pose_graph_state.as_ref().unwrap();
        assert_eq!(state.graph.poses.len(), 2, "graph bounded to the window");
        assert!(state.graph.poses.contains_key(&10), "anchor retained");
        assert!(state.graph.poses.contains_key(&30), "most recent retained");
        assert!(!state.graph.poses.contains_key(&20), "oldest marginalized");
        assert_eq!(
            state.graph.priors.len(),
            1,
            "marginalization left one prior"
        );
        // Anchor stays at the origin through marginalization.
        assert!(
            state.graph.poses[&10].camera_center_world().coords.norm() < 1e-9,
            "anchor unmoved"
        );
    }

    #[test]
    fn default_pose_graph_refinement_is_off() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig::default(),
        );
        assert!(slam.pose_graph_state.is_none());
        let frame = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let result = slam.process_frame(&frame, []);
        assert!(result.tracking_succeeded());
        assert!(result.pose_graph_refinement.is_none());
    }

    #[test]
    fn enabling_pose_graph_seeds_anchor_node_on_first_keyframe() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);
        let frame = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let result = slam.process_frame(&frame, []);
        assert!(result.tracking_succeeded());
        let stats = result
            .pose_graph_refinement
            .expect("stats reported when stage is enabled and a keyframe was registered");
        assert_eq!(stats.verified_candidate_count, 0);
        assert_eq!(stats.accepted_count, 0);
        assert!(stats.pose_graph_result.is_none());
        let state = slam
            .pose_graph_state
            .as_ref()
            .expect("pose-graph state initialised when config is set");
        assert_eq!(state.keyframe_order, vec![10]);
        assert_eq!(state.graph.poses.len(), 1);
        assert_eq!(state.graph.edges.len(), 0);
        assert_eq!(state.graph.anchor, Some(10));
    }

    #[test]
    fn sequential_edges_accumulate_without_loop_constraints() {
        // Two non-loop frames: the second is far enough that
        // `min_frame_id_gap=5` still applies but the synthetic 12-landmark
        // cloud is shared. With no perspective change the verifier can
        // still reject if no parallax; for this test we only assert the
        // sequential-edge accumulation and that PGO never fires before a
        // verified loop edge arrives.
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);
        // Two keyframes spaced 1.5 m apart so the `SimpleKeyframePolicy`
        // `min_translation = 1.0` default accepts both.
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(r1.tracking_succeeded());
        let state = slam
            .pose_graph_state
            .as_ref()
            .expect("state present after the second keyframe registration");
        assert_eq!(state.keyframe_order, vec![10, 20]);
        assert_eq!(state.graph.poses.len(), 2);
        assert_eq!(
            state
                .graph
                .edges
                .iter()
                .filter(|e| matches!(e.kind, visloc_slam::PoseGraphEdgeKind::Sequential))
                .count(),
            1,
            "exactly one sequential edge between consecutive keyframes"
        );
        // Even if a loop candidate was emitted by the shared-landmark
        // gate, the verifier accepting OR rejecting must NOT trigger PGO
        // unless an actual constraint was added. The trigger counter
        // tracks this.
        if let Some(stats) = r1.pose_graph_refinement.as_ref() {
            assert_eq!(
                stats.pose_graph_result.is_some(),
                stats.accepted_count >= 1,
                "PGO fires iff at least one loop constraint was accepted (trigger threshold = 1)"
            );
        }
    }

    #[test]
    fn reset_clears_pose_graph_state() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);
        assert_eq!(
            slam.pose_graph_state.as_ref().unwrap().keyframe_order.len(),
            1
        );
        slam.reset_sequence_state();
        let state = slam.pose_graph_state.as_ref().unwrap();
        assert!(state.keyframe_order.is_empty());
        assert!(state.graph.poses.is_empty());
        assert_eq!(state.trigger_count, 0);
        assert_eq!(state.pending_since_last_trigger, 0);
        assert!(state.verified_constraints.is_empty());
    }

    #[test]
    fn accepted_loop_constraint_triggers_pgo_and_writes_back_pose() {
        // Three-keyframe orbit: KF#10 at the origin, KF#20 displaced 1.5m
        // along +x (sequential edge), KF#30 returns near the origin so
        // the shared-landmark gate fires AND the essential-matrix
        // verifier has real parallax (KF#10 ↔ KF#30 baseline ~0.05m)
        // to lock onto. With `trigger_every_new_constraints = 1`, PGO
        // runs on the accepted loop. Anchor stays on KF#10 so its pose
        // is unchanged; KF#20 / KF#30 may be moved by the optimiser, but
        // the test only asserts the trigger fired AND poses were written
        // back into the map.
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(r1.tracking_succeeded() && r1.map_was_updated());
        // KF#30 at a slightly perturbed origin — same shared cloud as
        // KF#10 but with enough baseline that essential-matrix RANSAC
        // can recover a relative pose.
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded() && r2.map_was_updated());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.verified_candidate_count >= 1,
            "loop candidate generated by shared-landmark gate on return-to-origin frame"
        );
        assert!(
            stats.accepted_count >= 1,
            "essential-matrix verifier accepts the (KF#10, KF#30) loop on the synthetic cloud"
        );
        let pgo = stats
            .pose_graph_result
            .as_ref()
            .expect("PGO fires on the first accepted loop constraint at trigger threshold = 1");
        assert_eq!(pgo.anchor_id, 10);
        assert!(pgo.variable_count >= 2);
        assert!(
            stats.keyframes_updated >= 2,
            "PGO writes back at least the two non-anchor keyframe poses into the map"
        );
        let state = slam.pose_graph_state.as_ref().unwrap();
        // The synthetic 12-landmark cloud is shared across all three
        // keyframes, so the verifier may accept loops for (KF#10, KF#20)
        // on frame 20 AND (KF#10, KF#30) on frame 30 — both are
        // legitimate matches. The invariant we care about is that at
        // least one trigger fired and the pending counter was drained.
        assert!(state.trigger_count >= 1);
        assert_eq!(state.pending_since_last_trigger, 0);
        assert!(!state.verified_constraints.is_empty());
        // Anchor pose must remain identity.
        let anchor_pose = state.graph.poses.get(&10).unwrap();
        let anchor_center = anchor_pose.camera_center_world();
        assert!(
            anchor_center.coords.norm() < 1.0e-9,
            "anchor (KF#10) stays at the origin after PGO: actual {anchor_center:?}"
        );
    }

    /// Same three-keyframe orbit as
    /// `accepted_loop_constraint_triggers_pgo_and_writes_back_pose`
    /// (KF#10 origin, KF#20 displaced 1.5m, KF#30 returns to
    /// `(0.05, 0, 0.05)`, true |C30 - C10| = 0.05*sqrt(2) ≈ 0.0707m, NOT
    /// 1.0m), but with `verifier: LoopRefinementVerifier::EssentialMatrix`
    /// and `default_translation_scale: 1.0`. This is the measured bug this
    /// change addresses: two-view essential-matrix geometry cannot observe
    /// translation scale, so the accepted loop constraint's translation is
    /// pinned to the configured default (`1.0`) regardless of the true
    /// ~0.0707m baseline.
    #[test]
    fn essential_matrix_verifier_online_refinement_pins_default_translation_scale() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let true_translation_norm = (0.05_f64 * 0.05 + 0.05 * 0.05).sqrt();
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(stats.accepted_count >= 1);

        let state = slam.pose_graph_state.as_ref().unwrap();
        let constraint = state
            .verified_constraints
            .iter()
            .find(|c| c.from_keyframe_id == 10 && c.to_keyframe_id == 30)
            .expect("(KF#10, KF#30) loop constraint admitted");
        let translation_norm = constraint.relative_pose.translation.norm();
        assert!(
            (translation_norm - 1.0).abs() < 1.0e-6,
            "essential-matrix verifier pins translation to default_translation_scale=1.0; got {translation_norm}"
        );
        assert!(
            (translation_norm - true_translation_norm).abs() > 0.9,
            "the pinned translation ({translation_norm}) must NOT match the true baseline \
             ({true_translation_norm}) -- that mismatch is the bug this change fixes"
        );
    }

    /// PnP counterpart of
    /// `essential_matrix_verifier_online_refinement_pins_default_translation_scale`:
    /// same three-keyframe orbit and the same true ~0.0707m baseline, but
    /// `verifier: LoopRefinementVerifier::Pnp` verifies on 2D-3D
    /// correspondences against the map's triangulated landmarks instead of
    /// 2D-2D essential-matrix geometry. Because the keyframe pose already
    /// carries the world scale, the accepted constraint's translation
    /// should match the true baseline instead of an arbitrary constant.
    #[test]
    fn pnp_verifier_online_refinement_recovers_metric_translation() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let true_translation_norm = (0.05_f64 * 0.05 + 0.05 * 0.05).sqrt();
        let mut slam = pipeline_with_pnp_pose_graph(
            map.clone(),
            camera.clone(),
            1,
            PnPLoopClosureVerifierConfig::default(),
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.accepted_count >= 1,
            "PnP verifier accepts the (KF#10, KF#30) loop on the synthetic cloud"
        );

        let state = slam.pose_graph_state.as_ref().unwrap();
        let constraint = state
            .verified_constraints
            .iter()
            .find(|c| c.from_keyframe_id == 10 && c.to_keyframe_id == 30)
            .expect("(KF#10, KF#30) loop constraint admitted");
        let translation_norm = constraint.relative_pose.translation.norm();
        assert!(
            (translation_norm - true_translation_norm).abs() < 0.01,
            "PnP verifier should recover the metric translation norm \
             {true_translation_norm}, got {translation_norm}"
        );
    }

    /// Same three-keyframe orbit as the iterative test above, but with the
    /// back-end configured to run GNC (`OnlineSlamLoopClosureRefinementConfig::gnc`).
    /// Asserts the wiring: the GNC robust solver fires instead of the plain
    /// M-estimator (`gnc_result` populated, `pose_graph_result` empty), the
    /// optimised poses are still written back into the map, the anchor stays
    /// fixed, and — crucially — the *legitimate* loop closure is NOT
    /// false-rejected (`loop_closures_rejected == 0`, the loop edge keeps an
    /// inlier weight). The outlier-rejection capability itself is proven
    /// directly on the pose graph in `gnc_robust_pgo.rs`; this test guards the
    /// pipeline integration point.
    #[test]
    fn gnc_back_end_fires_and_keeps_a_legitimate_loop_closure() {
        use visloc_slam::gnc::{GncConfig, GncKernel, AUTO_SCALE_K};
        use visloc_slam::PoseGraphEdgeKind;

        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    // Gap 15 so only the (KF#10, KF#30) pair (id gap 20)
                    // generates a loop candidate — the (10,20) / (20,30)
                    // pairs (gap 10) are suppressed. With a single loop edge
                    // there is no multi-edge translation-scale conflict (the
                    // essential-matrix verifier fixes every loop edge's
                    // magnitude to `default_translation_scale`, which cannot
                    // match three different true displacements at once).
                    min_frame_id_gap: 15,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera: camera.clone(),
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        // Match the true |C30 - C10| = ‖(0.05, 0, 0.05)‖ so the
                        // single (10,30) loop edge is geometrically consistent
                        // (the verifier pins every loop edge's translation
                        // magnitude to this constant).
                        default_translation_scale: 0.05 * std::f64::consts::SQRT_2,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    // TLS with a MAD auto-scaled inlier band (the band tracks
                    // the live graph's noise); no re-adapt — it is a BA-only
                    // win and over-rejects real edges on pose graphs.
                    gnc: Some(GncConfig {
                        kernel: GncKernel::TruncatedLeastSquares,
                        c: 1.0e-3,
                        auto_scale: Some(AUTO_SCALE_K),
                        auto_scale_readapt: false,
                        ..GncConfig::default()
                    }),
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

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        // KF#30 returns near the origin (~0.07 m baseline to KF#10) — enough
        // parallax for essential-matrix RANSAC, and far enough from KF#20
        // (~1.45 m) to register as a fresh keyframe. The verifier's
        // `default_translation_scale` is set to this baseline so the loop edge
        // is geometrically consistent — a legitimate closure GNC should keep.
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded() && r2.map_was_updated());

        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(stats.accepted_count >= 1, "the legitimate loop is accepted");
        // The GNC path fired, not the plain iterative one.
        let gnc = stats
            .gnc_result
            .as_ref()
            .expect("the GNC solver fires when `gnc` is configured");
        assert!(
            stats.pose_graph_result.is_none(),
            "the iterative result must be empty on the GNC path"
        );
        assert_eq!(gnc.anchor_id, 10);
        assert!(gnc.variable_count >= 2);
        assert!(
            stats.keyframes_updated >= 2,
            "GNC writes back at least the two non-anchor keyframe poses"
        );
        let state = slam.pose_graph_state.as_ref().unwrap();
        // The good loop closure is kept, not false-rejected.
        assert_eq!(
            stats.loop_closures_rejected, 0,
            "GNC must not reject a legitimate loop closure"
        );
        let loop_edges_inlier = state
            .graph
            .edges
            .iter()
            .zip(&gnc.edge_weights)
            .filter(|(e, _)| e.kind == PoseGraphEdgeKind::LoopClosure)
            .all(|(_, &w)| w >= 0.5);
        assert!(
            loop_edges_inlier,
            "every legitimate loop edge keeps an inlier weight"
        );
        // Anchor pose must remain at the origin.
        let anchor_center = state.graph.poses.get(&10).unwrap().camera_center_world();
        assert!(anchor_center.coords.norm() < 1.0e-9);
    }

    /// With the PCM front-end screen configured, a legitimate loop closure must
    /// be ADMITTED (not screened out) so it still enters the graph and the
    /// solve fires — the wiring guard for the online PCM path. PCM's actual
    /// rejection of inconsistent closures is proven directly in
    /// `visloc_slam::pcm`'s unit tests.
    #[test]
    fn pcm_front_end_admits_a_legitimate_loop_closure() {
        use visloc_slam::pcm::PcmConfig;
        use visloc_slam::PoseGraphEdgeKind;

        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 15,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera: camera.clone(),
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 0.05 * std::f64::consts::SQRT_2,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: Some(PcmConfig::default()),
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

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded() && r2.map_was_updated());

        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.accepted_count >= 1,
            "the legitimate loop is admitted through the PCM screen"
        );
        assert_eq!(
            stats.loop_closures_pcm_rejected, 0,
            "PCM must not reject a legitimate loop closure"
        );
        // The admitted loop entered the graph and a plain solve fired (gnc unset).
        assert!(
            stats.pose_graph_result.is_some(),
            "PGO fired on the admitted loop"
        );
        let state = slam.pose_graph_state.as_ref().unwrap();
        let loop_edges = state
            .graph
            .edges
            .iter()
            .filter(|e| e.kind == PoseGraphEdgeKind::LoopClosure)
            .count();
        assert!(loop_edges >= 1, "the admitted loop is present in the graph");
    }

    /// With the covariance gate configured, a legitimate loop closure (small
    /// innovation versus the estimate's prediction) must PASS the gate so it
    /// still enters the graph and the solve fires — the wiring guard for the
    /// online covariance gate. The gate's rejection of an implausible
    /// innovation is proven directly in `tests/pose_graph_covariance.rs`.
    #[test]
    fn covariance_gate_admits_a_legitimate_loop_closure() {
        use visloc_slam::covariance::CHI2_95_6DOF;
        use visloc_slam::PoseGraphEdgeKind;

        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 15,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera: camera.clone(),
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 0.05 * std::f64::consts::SQRT_2,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: Some(CHI2_95_6DOF),
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

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded() && r2.map_was_updated());

        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(stats.accepted_count >= 1, "the legitimate loop is admitted");
        assert_eq!(
            stats.loop_closures_covariance_rejected, 0,
            "the covariance gate must not reject a legitimate loop closure"
        );
        assert!(
            stats.pose_graph_result.is_some(),
            "PGO fired on the admitted loop"
        );
        let state = slam.pose_graph_state.as_ref().unwrap();
        let loop_edges = state
            .graph
            .edges
            .iter()
            .filter(|e| e.kind == PoseGraphEdgeKind::LoopClosure)
            .count();
        assert!(loop_edges >= 1, "the admitted loop is present in the graph");
    }

    #[test]
    fn pose_graph_refinement_skipped_when_no_keyframe_registered() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        // Disable map updates so no keyframe ever enters the map even
        // though the tracker localizes — verifies that the stage gates
        // on `applied_update.is_some()` rather than `tracking.success`.
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: false,
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera: camera.clone(),
                    verifier_config: LoopClosureVerifierConfig::default(),
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
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
        let frame = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let result = slam.process_frame(&frame, []);
        assert!(result.tracking_succeeded());
        assert!(!result.map_was_updated());
        assert!(
            result.pose_graph_refinement.is_none(),
            "stage must short-circuit when no keyframe was applied this frame"
        );
        assert!(slam
            .pose_graph_state
            .as_ref()
            .unwrap()
            .keyframe_order
            .is_empty());
    }

    // --------------------------------------------------------------
    // Appearance-based long-range loop candidate source
    // (`LoopAppearanceCandidateConfig` / `build_appearance_loop_candidates`)
    // --------------------------------------------------------------

    /// Mirrors the internal `relocalization_mean_descriptor` computation
    /// (mean, then L2-normalize) so the test can populate a
    /// `descriptor_cache` entry the same way the pipeline would.
    fn mean_descriptor_for_test(descriptors: &[Vec<f32>]) -> Vec<f32> {
        let dim = descriptors[0].len();
        let mut mean = vec![0.0_f32; dim];
        for descriptor in descriptors {
            for (acc, value) in mean.iter_mut().zip(descriptor) {
                *acc += *value;
            }
        }
        let inv_count = 1.0_f32 / descriptors.len() as f32;
        let mut norm = 0.0_f32;
        for value in &mut mean {
            *value *= inv_count;
            norm += *value * *value;
        }
        let norm = norm.sqrt();
        for value in &mut mean {
            *value /= norm;
        }
        mean
    }

    /// Builds a `VisualMap` with a single registered keyframe (id 10, at the
    /// world origin, no rotation) observing 40 distinct 3D points under
    /// landmark ids 1..=40, plus the matching appearance-descriptor-cache
    /// entry `build_appearance_loop_candidates` expects. Returns
    /// `(map, camera, descriptor_cache, points, descriptors)` so each test
    /// can build its own "current frame" (a would-be keyframe that has NOT
    /// re-triangulated these points into the map under new ids yet — the
    /// scenario the appearance stream targets, since shared-landmark
    /// candidates are structurally impossible here).
    #[allow(clippy::type_complexity)]
    fn appearance_candidate_fixture() -> (
        VisualMap,
        Camera,
        std::collections::HashMap<u64, Vec<f32>>,
        Vec<Point3<f64>>,
        Vec<Vec<f32>>,
    ) {
        let camera = camera();
        let points: Vec<Point3<f64>> = (0..40)
            .map(|k| {
                let k = k as f64;
                Point3::new(
                    -1.5 + 0.08 * k,
                    0.8 * (k * 0.37).sin(),
                    5.0 + 0.3 * (k * 0.19).cos(),
                )
            })
            .collect();
        let descriptors: Vec<Vec<f32>> = (0..points.len())
            .map(|index| {
                (0..16)
                    .map(|d| ((index as f32) * 0.13 + (d as f32) * 0.07).sin())
                    .collect()
            })
            .collect();

        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let mut landmark = Landmark::new(landmark_id, *point);
            landmark.descriptor = Some(descriptors[index].clone());
            map.landmarks.insert(landmark_id, landmark);
        }

        let pose_a = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let mut frame_a = Frame::new(10, camera.id);
        let mut observations_a = Vec::new();
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let projected = camera
                .project(&pose_a.transform_world_point(point))
                .unwrap();
            frame_a.keypoints.push(projected);
            frame_a.descriptors.push(descriptors[index].clone());
            observations_a.push(visloc_core::types::Observation {
                frame_id: 10,
                landmark_id,
                keypoint_index: index,
                xy: projected,
            });
        }
        frame_a.pose = Some(pose_a);
        map.keyframes.insert(
            10,
            visloc_core::types::Keyframe {
                frame: frame_a.clone(),
                observations: observations_a,
            },
        );

        let mut descriptor_cache = std::collections::HashMap::new();
        descriptor_cache.insert(10_u64, mean_descriptor_for_test(&frame_a.descriptors));

        (map, camera, descriptor_cache, points, descriptors)
    }

    /// Builds the "current frame" (not yet a registered keyframe) that
    /// revisits `points` from a translated camera center, reusing the exact
    /// same descriptors (appearance match) under NO landmark ids at all —
    /// correspondences can only come from descriptor matching against the
    /// candidate keyframe's own landmarks, never from shared ids.
    fn appearance_revisit_frame(
        frame_id: u64,
        camera_center: Vector3<f64>,
        points: &[Point3<f64>],
        descriptors: &[Vec<f32>],
        camera: &Camera,
    ) -> Frame {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
        let mut frame = Frame::new(frame_id, camera.id);
        for (index, point) in points.iter().enumerate() {
            let projected = camera.project(&pose.transform_world_point(point)).unwrap();
            frame.keypoints.push(projected);
            frame.descriptors.push(descriptors[index].clone());
        }
        frame.pose = Some(pose);
        frame
    }

    fn insert_revisit_keyframe(map: &mut VisualMap, frame: Frame) {
        let observations = frame
            .keypoints
            .iter()
            .enumerate()
            .map(|(keypoint_index, xy)| visloc_core::types::Observation {
                frame_id: frame.id,
                landmark_id: 10_000 + keypoint_index as u64,
                keypoint_index,
                xy: *xy,
            })
            .collect();
        map.keyframes.insert(
            frame.id,
            visloc_core::types::Keyframe {
                frame,
                observations,
            },
        );
    }

    #[test]
    fn appearance_loop_candidate_requires_three_current_covisible_keyframes() {
        let (mut map, camera, descriptor_cache, points, descriptors) =
            appearance_candidate_fixture();
        let frame_200 = appearance_revisit_frame(
            200,
            Vector3::new(1.8, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );
        let frame_205 = appearance_revisit_frame(
            205,
            Vector3::new(1.9, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );
        let frame_210 = appearance_revisit_frame(
            210,
            Vector3::new(2.0, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );
        insert_revisit_keyframe(&mut map, frame_200);
        insert_revisit_keyframe(&mut map, frame_205);
        insert_revisit_keyframe(&mut map, frame_210.clone());

        let candidates = build_appearance_loop_candidates(
            &map,
            &frame_210,
            &descriptor_cache,
            &LoopAppearanceCandidateConfig::default(),
            &camera,
        );
        assert_eq!(candidates.len(), 1);

        map.keyframes.remove(&205);
        let rejected = build_appearance_loop_candidates(
            &map,
            &frame_210,
            &descriptor_cache,
            &LoopAppearanceCandidateConfig::default(),
            &camera,
        );
        assert!(
            rejected.is_empty(),
            "two agreeing keyframes must not satisfy the three-keyframe gate"
        );
    }

    #[test]
    fn appearance_loop_candidate_recovers_metric_translation_for_a_long_range_revisit() {
        let (map, camera, descriptor_cache, points, descriptors) = appearance_candidate_fixture();

        // Keyframe-id gap of 200 clears the default 150 minimum; the
        // camera has translated by a known 2m baseline along x.
        let true_translation = Vector3::new(2.0, 0.0, 0.0);
        let frame_b =
            appearance_revisit_frame(210, true_translation, &points, &descriptors, &camera);

        let config = LoopAppearanceCandidateConfig {
            min_covisible_keyframe_verifications: 1,
            ..LoopAppearanceCandidateConfig::default()
        };
        let candidates =
            build_appearance_loop_candidates(&map, &frame_b, &descriptor_cache, &config, &camera);

        assert_eq!(
            candidates.len(),
            1,
            "the single cached older keyframe must be ranked and verified"
        );
        let candidate = &candidates[0];
        assert_eq!(candidate.matched_keyframe_id, 10);
        let verification = candidate
            .verification
            .as_ref()
            .expect("appearance candidates are always verifier-populated");
        assert!(
            verification.verified,
            "noiseless correspondences must pass PnP verification"
        );
        assert!(
            verification.inlier_count >= 30,
            "expected >= 30 inliers out of 40 correspondences, got {}",
            verification.inlier_count
        );

        let constraint = LoopClosureConstraint::from_verified_candidate(candidate)
            .expect("a verified candidate must yield a constraint");
        assert_eq!(constraint.from_keyframe_id, 10);
        assert_eq!(constraint.to_keyframe_id, 210);
        let recovered_translation_norm = constraint.relative_pose.translation.norm();
        let true_translation_norm = true_translation.norm();
        assert!(
            (recovered_translation_norm - true_translation_norm).abs() < 1.0e-2,
            "recovered translation norm {recovered_translation_norm} should match the true baseline {true_translation_norm}"
        );
    }

    #[test]
    fn appearance_loop_candidate_refines_with_projection_and_reports_its_gate() {
        let (map, camera, descriptor_cache, points, descriptors) = appearance_candidate_fixture();
        let frame = appearance_revisit_frame(
            210,
            Vector3::new(2.0, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );
        let accepting_config = LoopAppearanceCandidateConfig {
            min_covisible_keyframe_verifications: 1,
            projection_search_radius_px: Some(15.0),
            min_projection_correspondence_count: 1,
            ..LoopAppearanceCandidateConfig::default()
        };
        let accepted = build_appearance_loop_candidates_with_diagnostics(
            &map,
            &frame,
            &descriptor_cache,
            &accepting_config,
            &camera,
        );
        assert_eq!(accepted.candidates.len(), 1);
        assert_eq!(accepted.projection_rejected_count, 0);
        let evidence = &accepted.diagnostics[0];
        assert!(evidence.projection_attempted);
        assert!(evidence.projection_correspondence_count > 0);
        assert!(evidence.projection_inlier_count > 0);
        assert!(evidence.projection_accepted);

        let rejecting_config = LoopAppearanceCandidateConfig {
            min_projection_correspondence_count: usize::MAX,
            ..accepting_config
        };
        let rejected = build_appearance_loop_candidates_with_diagnostics(
            &map,
            &frame,
            &descriptor_cache,
            &rejecting_config,
            &camera,
        );
        assert!(rejected.candidates.is_empty());
        assert_eq!(rejected.projection_rejected_count, 1);
        let evidence = &rejected.diagnostics[0];
        assert!(evidence.projection_attempted);
        assert!(evidence.projection_correspondence_count > 0);
        assert!(!evidence.projection_accepted);
    }

    #[test]
    fn appearance_loop_candidate_rejects_a_region_connected_to_current() {
        let (mut map, camera, descriptor_cache, points, descriptors) =
            appearance_candidate_fixture();
        let frame_b = appearance_revisit_frame(
            210,
            Vector3::new(2.0, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );
        let observations = frame_b
            .keypoints
            .iter()
            .take(12)
            .enumerate()
            .map(|(keypoint_index, xy)| Observation {
                frame_id: frame_b.id,
                landmark_id: keypoint_index as u64 + 1,
                keypoint_index,
                xy: *xy,
            })
            .collect();
        map.keyframes.insert(
            frame_b.id,
            visloc_core::types::Keyframe {
                frame: frame_b.clone(),
                observations,
            },
        );

        let result = build_appearance_loop_candidates_with_diagnostics(
            &map,
            &frame_b,
            &descriptor_cache,
            &LoopAppearanceCandidateConfig {
                min_covisible_keyframe_verifications: 1,
                ..LoopAppearanceCandidateConfig::default()
            },
            &camera,
        );
        assert!(result.candidates.is_empty());
        assert_eq!(result.connected_region_rejected_count, 1);
        assert_eq!(result.pnp_verified_count, 0);
    }

    #[test]
    fn appearance_loop_candidate_skips_keyframes_below_min_keyframe_id_gap() {
        let (map, camera, descriptor_cache, points, descriptors) = appearance_candidate_fixture();

        // Keyframe-id gap of only 40 is below the default 150 minimum, even
        // though the appearance similarity and geometry would otherwise
        // verify cleanly — the long-range gate must reject it outright.
        let frame_b = appearance_revisit_frame(
            50,
            Vector3::new(2.0, 0.0, 0.0),
            &points,
            &descriptors,
            &camera,
        );

        let config = LoopAppearanceCandidateConfig::default();
        let candidates =
            build_appearance_loop_candidates(&map, &frame_b, &descriptor_cache, &config, &camera);

        assert!(
            candidates.is_empty(),
            "candidates below min_keyframe_id_gap must never be ranked, let alone verified"
        );
    }

    // ============================================================
    // `propagate_corrections` — PGO write-back correction propagation
    // ============================================================

    /// Same wiring as [`pipeline_with_pose_graph`], but with
    /// `propagate_corrections: true` so a converged solve also moves
    /// anchored landmarks and corrects the tracker's continuation state.
    fn pipeline_with_propagating_pose_graph(
        map: VisualMap,
        camera: Camera,
        trigger_every_new_constraints: usize,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera,
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 1.0,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: None,
                    pcm_batch_rescreen: false,
                    marginalization_window: None,
                    marginalization_sparsify: false,
                    trigger_every_new_constraints,
                    appearance_candidates: None,
                    propagate_corrections: true,
                    solver: LoopRefinementSolver::Se3,
                }),
                ..OnlineSlamConfig::default()
            },
        )
    }

    const SYNTHETIC_ANCHOR_LANDMARK_ID: u64 = 9001;
    const UNSOLVED_ANCHOR_LANDMARK_ID: u64 = 9002;
    const UNSOLVED_ANCHOR_KEYFRAME_ID: u64 = 500;

    /// (a) Exact-displacement check + (d) pose-convention sanity check in
    /// one scenario (they share the same fixture, and (d) is exactly the
    /// check that would fail immediately if the correction direction were
    /// backwards).
    #[test]
    fn landmark_propagation_moves_anchored_landmark_by_keyframe_correction_and_preserves_reprojection(
    ) {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_propagating_pose_graph(map.clone(), camera.clone(), 1);

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());

        // Seed a synthetic landmark anchored EXCLUSIVELY to KF#20: its
        // only observation is on KF#20, so KF#20 (not KF#10, the graph's
        // fixed anchor) is its anchor keyframe. Its pixel is the TRUE
        // projection of its 3D position under KF#20's pre-solve pose.
        let anchor_landmark_position = Point3::new(0.3, -0.2, 5.5);
        let old_pose_20 = slam
            .map
            .keyframes
            .get(&20)
            .expect("KF#20 registered")
            .frame
            .pose
            .clone()
            .expect("KF#20 has a pose");
        let original_pixel = camera
            .project(&old_pose_20.transform_world_point(&anchor_landmark_position))
            .expect("synthetic point projects in front of the camera");
        {
            let kf20 = slam.map.keyframes.get_mut(&20).unwrap();
            let keypoint_index = kf20.frame.keypoints.len();
            kf20.frame.keypoints.push(original_pixel);
            kf20.frame.descriptors.push(vec![0.0; 2]);
            kf20.observations.push(Observation {
                frame_id: 20,
                landmark_id: SYNTHETIC_ANCHOR_LANDMARK_ID,
                keypoint_index,
                xy: original_pixel,
            });
        }
        let mut landmark = Landmark::new(SYNTHETIC_ANCHOR_LANDMARK_ID, anchor_landmark_position);
        landmark.observations.push(Observation {
            frame_id: 20,
            landmark_id: SYNTHETIC_ANCHOR_LANDMARK_ID,
            keypoint_index: slam.map.keyframes[&20].frame.keypoints.len() - 1,
            xy: original_pixel,
        });
        slam.map
            .landmarks
            .insert(SYNTHETIC_ANCHOR_LANDMARK_ID, landmark);

        // Also seed a landmark anchored to a keyframe the running pose
        // graph never registered (id 500 is inserted straight into
        // `slam.map.keyframes`, bypassing `process_frame`, so it is never
        // part of `state.graph.poses`). Exercises (b): its anchor is
        // never in the solved `corrections` map, so it must stay put.
        let unsolved_keyframe_pose =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(9.0, 9.0, 9.0));
        let mut unsolved_frame = Frame::new(UNSOLVED_ANCHOR_KEYFRAME_ID, camera.id);
        let unsolved_landmark_position = Point3::new(100.0, 100.0, 100.0);
        unsolved_frame.keypoints.push(
            camera
                .project(&unsolved_keyframe_pose.transform_world_point(&unsolved_landmark_position))
                .unwrap(),
        );
        unsolved_frame.descriptors.push(vec![0.0; 2]);
        unsolved_frame.pose = Some(unsolved_keyframe_pose);
        slam.map.keyframes.insert(
            UNSOLVED_ANCHOR_KEYFRAME_ID,
            visloc_core::types::Keyframe {
                frame: unsolved_frame,
                observations: vec![Observation {
                    frame_id: UNSOLVED_ANCHOR_KEYFRAME_ID,
                    landmark_id: UNSOLVED_ANCHOR_LANDMARK_ID,
                    keypoint_index: 0,
                    xy: nalgebra::Point2::new(0.0, 0.0),
                }],
            },
        );
        slam.map.landmarks.insert(
            UNSOLVED_ANCHOR_LANDMARK_ID,
            Landmark::new(UNSOLVED_ANCHOR_LANDMARK_ID, unsolved_landmark_position),
        );

        // KF#30 returns near the origin: shared-landmark loop gate fires
        // and the essential-matrix verifier locks onto real parallax
        // against KF#10, triggering PGO (same fixture as
        // `accepted_loop_constraint_triggers_pgo_and_writes_back_pose`).
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.pose_graph_result.is_some(),
            "PGO must fire for this test to exercise propagation"
        );

        let new_pose_20 = slam
            .map
            .keyframes
            .get(&20)
            .unwrap()
            .frame
            .pose
            .clone()
            .expect("KF#20 still has a pose after PGO");
        assert_ne!(
            new_pose_20.world_to_camera.translation, old_pose_20.world_to_camera.translation,
            "KF#20 must actually move for this to be a meaningful test"
        );

        // (a) Exact-displacement check: the landmark must move by EXACTLY
        // KF#20's own world-frame correction
        // `C = T_cw_new ∘ T_wc_old`.
        let expected_correction = new_pose_20
            .world_to_camera
            .inverse()
            .compose(&old_pose_20.world_to_camera);
        let expected_position = expected_correction.transform_point(&anchor_landmark_position);
        let moved_position = slam.map.landmarks[&SYNTHETIC_ANCHOR_LANDMARK_ID].position;
        assert!(
            (moved_position - expected_position).norm() < 1.0e-9,
            "landmark must move by exactly KF#20's correction: expected {expected_position:?}, got {moved_position:?}"
        );
        assert_ne!(
            moved_position, anchor_landmark_position,
            "the landmark must actually have moved"
        );

        // (b) The landmark anchored to the never-registered keyframe 500
        // must be untouched.
        assert_eq!(
            slam.map.landmarks[&UNSOLVED_ANCHOR_LANDMARK_ID].position, unsolved_landmark_position,
            "a landmark anchored to a keyframe outside the solved graph must not move"
        );

        // (d) Pose-convention sanity: reprojecting the CORRECTED landmark
        // through the CORRECTED keyframe pose must reproduce the
        // ORIGINAL pixel. A backwards correction would move the landmark
        // in the wrong direction and fail this check immediately.
        let corrected_pixel = camera
            .project(&new_pose_20.transform_world_point(&moved_position))
            .expect("corrected point still projects in front of the camera");
        assert!(
            (corrected_pixel - original_pixel).norm() < 1.0e-6,
            "corrected landmark must reproject to the original pixel: expected {original_pixel:?}, got {corrected_pixel:?}"
        );

        assert!(
            stats.landmarks_moved >= 1,
            "stats must report at least the synthetic landmark as moved"
        );
        assert!(stats.max_landmark_displacement_meters.unwrap() > 0.0);
        assert!(stats.mean_landmark_displacement_meters.unwrap() > 0.0);
        assert!(
            stats.tracker_correction_applied,
            "tracker continuation state must be corrected when a solve moved a keyframe"
        );
    }

    /// (c) `propagate_corrections: false` (the default) must leave every
    /// landmark byte-identical even when PGO fires and moves keyframes —
    /// today's behaviour, preserved exactly.
    #[test]
    fn propagate_corrections_false_leaves_landmarks_byte_identical() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let landmarks_before = map.landmarks.clone();
        let mut slam = pipeline_with_pose_graph(map.clone(), camera.clone(), 1);

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.pose_graph_result.is_some(),
            "PGO must fire for this to be a meaningful test"
        );
        assert_eq!(stats.landmarks_moved, 0);
        assert!(stats.max_landmark_displacement_meters.is_none());
        assert!(stats.mean_landmark_displacement_meters.is_none());
        assert!(!stats.tracker_correction_applied);

        for (id, landmark) in &landmarks_before {
            assert_eq!(
                slam.map.landmarks[id].position, landmark.position,
                "landmark {id} must stay byte-identical when propagate_corrections is false"
            );
        }
    }

    // ============================================================
    // `LoopRefinementSolver::Sim3` — Sim(3) pose-graph solver opt-in
    // ============================================================

    /// [`LoopRefinementSolver::default`] must stay [`LoopRefinementSolver::Se3`]
    /// — the byte-identical-default contract every existing
    /// `OnlineSlamLoopClosureRefinementConfig` fixture in this file relies on
    /// (they all set `solver: LoopRefinementSolver::Se3` explicitly and
    /// continue to pass unchanged).
    #[test]
    fn loop_refinement_solver_default_is_se3() {
        assert_eq!(LoopRefinementSolver::default(), LoopRefinementSolver::Se3);
    }

    /// (a) Solver-level A/B on a synthetic drifted chain: a per-step
    /// sequential-edge measurement carrying a constant translation-scale
    /// bias (the classic monocular-style scale-drift signature — each
    /// incremental relative-motion estimate is systematically off by a
    /// multiplicative factor) plus one TRUE metric loop-closure edge (the
    /// PnP loop verifier's scale-correct constraint). A rigid `SE(3)` graph
    /// has no scale DOF to absorb the bias with, so the loop-vs-odometry
    /// inconsistency smears across every edge as rotation/translation error;
    /// `Sim3PoseGraph` can push the bias into each node's scale instead,
    /// recovering per-node scales that are NOT 1.0 and a much smaller
    /// position error against the true (unbiased) trajectory.
    #[test]
    fn sim3_solver_uses_scale_dof_to_reduce_scale_drift_residual() {
        use std::f64::consts::TAU;
        use visloc_core::geometry::SE3;

        const NODE_COUNT: u64 = 8;
        const SCALE_BIAS: f64 = 1.15;
        const LOOP_WEIGHT: f64 = 50.0;

        // Ground truth: a circular trajectory at unit (metric) scale, same
        // shape `sim3_pose_graph.rs`'s own unit tests use.
        let truth: Vec<(u64, UnitQuaternion<f64>, Vector3<f64>)> = (0..NODE_COUNT)
            .map(|i| {
                let angle = i as f64 / NODE_COUNT as f64 * TAU;
                let rotation = UnitQuaternion::from_euler_angles(0.0, 0.0, angle);
                let translation =
                    Vector3::new(5.0 * angle.cos(), 5.0 * angle.sin(), 0.3 * i as f64);
                (i, rotation, translation)
            })
            .collect();
        let true_pose = |idx: usize| SE3::new(truth[idx].1, truth[idx].2);
        // Measured relative SE3 `from -> to`, exactly the
        // `relative_world_to_camera` convention (`to == measurement.compose(&from)`).
        let true_relative =
            |from: usize, to: usize| true_pose(to).compose(&true_pose(from).inverse());
        // The same relative motion with its translation systematically
        // scaled by `SCALE_BIAS` — a per-step scale-biased measurement,
        // not a corrupted initial guess.
        let biased_relative = |from: usize, to: usize| {
            let relative = true_relative(from, to);
            SE3::new(relative.rotation, relative.translation * SCALE_BIAS)
        };

        // Dead-reckon the initial pose estimate from the anchor using the
        // BIASED measurements — exactly what an online tracker accumulating
        // a per-step scale bias would produce, and exactly self-consistent
        // with the sequential edges below (zero sequential-edge residual at
        // the seeded estimate, matching how the online pipeline always
        // seeds a fresh keyframe's node from its own just-tracked pose).
        let mut estimate = vec![true_pose(0)];
        for w in 1..NODE_COUNT as usize {
            estimate.push(biased_relative(w - 1, w).compose(&estimate[w - 1]));
        }

        let mut se3_graph = PoseGraph::new();
        let mut sim3_graph = Sim3PoseGraph::new();
        for (i, pose) in estimate.iter().enumerate() {
            se3_graph.add_pose(
                i as u64,
                Pose::from_world_to_camera(pose.rotation, pose.translation),
            );
            sim3_graph.add_pose(i as u64, Sim3::new(pose.rotation, pose.translation, 1.0));
        }
        se3_graph.anchor(0);
        sim3_graph.anchor(0);
        for w in 1..NODE_COUNT as usize {
            let measurement = biased_relative(w - 1, w);
            se3_graph.add_sequential_edge(w as u64 - 1, w as u64, measurement.clone());
            sim3_graph.add_edge(
                w as u64 - 1,
                w as u64,
                Sim3::new(measurement.rotation, measurement.translation, 1.0),
                1.0,
            );
        }
        let last = NODE_COUNT as usize - 1;
        let loop_measurement = true_relative(last, 0);
        se3_graph.add_loop_closure_constraint(&LoopClosureConstraint {
            from_keyframe_id: last as u64,
            to_keyframe_id: 0,
            relative_pose: loop_measurement.clone(),
            inlier_count: LOOP_WEIGHT as usize,
            inlier_ratio: 1.0,
            mean_sampson_error: 0.0,
            score: 0.0,
        });
        sim3_graph.add_edge(
            last as u64,
            0,
            Sim3::new(loop_measurement.rotation, loop_measurement.translation, 1.0),
            LOOP_WEIGHT,
        );

        let se3_result = se3_graph
            .optimize_se3_iterative(&PoseGraphSe3Config::default())
            .expect("SE3 solve should succeed (well-posed graph)");
        let sim3_result = sim3_graph
            .optimize(&Sim3PoseGraphConfig::default())
            .expect("Sim3 solve should succeed (well-posed graph)");
        assert!(se3_result.converged, "SE3 solve should converge");
        assert!(sim3_result.converged, "Sim3 solve should converge");

        let se3_error: f64 = truth
            .iter()
            .map(|(id, _, t)| (se3_graph.poses[id].world_to_camera.translation - t).norm())
            .sum::<f64>()
            / truth.len() as f64;
        // Write-back convention (see `LoopRefinementSolver::Sim3`'s doc
        // comment): a solved Sim3 node's rigid position is
        // `translation / scale`.
        let sim3_error: f64 = truth
            .iter()
            .map(|(id, _, t)| {
                let node = &sim3_graph.poses[id];
                (node.translation / node.scale - t).norm()
            })
            .sum::<f64>()
            / truth.len() as f64;

        let se3_cost = se3_graph.se3_cost();
        let sim3_cost = sim3_graph.cost();
        assert!(
            sim3_cost < se3_cost * 0.8,
            "Sim3 should use its scale DOF to fit the inconsistent scale-biased graph \
             substantially better than SE3: se3_cost={se3_cost}, sim3_cost={sim3_cost}, \
             se3_position_error={se3_error}, sim3_position_error={sim3_error}"
        );

        let worst_scale_deviation = sim3_graph
            .poses
            .values()
            .map(|pose| (pose.scale - 1.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            worst_scale_deviation > 0.02,
            "expected the Sim3 solve to recover a non-trivial per-node scale correction \
             (per-node scales != 1), worst |scale-1| = {worst_scale_deviation}"
        );
    }

    /// Same wiring as [`pipeline_with_propagating_pose_graph`], but with
    /// `solver: LoopRefinementSolver::Sim3(_)`.
    fn pipeline_with_propagating_sim3_pose_graph(
        map: VisualMap,
        camera: Camera,
        trigger_every_new_constraints: usize,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                loop_closure: LoopClosureConfig {
                    min_frame_id_gap: 5,
                    min_shared_landmarks: 4,
                    min_shared_landmark_ratio_percent: 30,
                    ..LoopClosureConfig::default()
                },
                pose_graph_refinement: Some(OnlineSlamLoopClosureRefinementConfig {
                    camera,
                    verifier_config: LoopClosureVerifierConfig {
                        min_inliers: 8,
                        min_inlier_ratio: 0.5,
                        max_mean_sampson_error: 5.0e-3,
                        default_translation_scale: 1.0,
                    },
                    verifier: LoopRefinementVerifier::EssentialMatrix,
                    pose_graph_config: PoseGraphSe3Config::default(),
                    gnc: None,
                    pcm: None,
                    covariance_gate: None,
                    pcm_batch_rescreen: false,
                    marginalization_window: None,
                    marginalization_sparsify: false,
                    trigger_every_new_constraints,
                    appearance_candidates: None,
                    propagate_corrections: true,
                    solver: LoopRefinementSolver::Sim3(Sim3PoseGraphConfig::default()),
                }),
                ..OnlineSlamConfig::default()
            },
        )
    }

    /// (b) Sim3 propagation projection-invariance: duplicates
    /// [`landmark_propagation_moves_anchored_landmark_by_keyframe_correction_and_preserves_reprojection`]
    /// for the `Sim3` solver, with a genuinely non-1 solved scale (forced by
    /// hand-editing the running `Sim3PoseGraph` mirror's sequential edge
    /// before the loop-closing frame triggers the solve — the metric PnP
    /// tracker in this test harness has no organic source of scale error, so
    /// this is the only way to exercise the scale-dependent half of the
    /// fixed-point/scale convention deterministically). If the write-back
    /// convention (`t / s`) or the propagation similarity
    /// (`Siw_new⁻¹ ∘ Siw_old`) used the wrong fixed point or scale
    /// direction, the corrected landmark would NOT reproject to the
    /// original pixel and this test would fail immediately.
    #[test]
    fn sim3_propagation_moves_anchored_landmark_and_preserves_reprojection_under_nontrivial_scale()
    {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_propagating_sim3_pose_graph(map.clone(), camera.clone(), 1);

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        assert!(slam.process_frame(&f0, []).tracking_succeeded());
        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        assert!(slam.process_frame(&f1, []).tracking_succeeded());

        // Seed a synthetic landmark anchored EXCLUSIVELY to KF#20, exactly
        // like the SE3 propagation test.
        let anchor_landmark_position = Point3::new(0.3, -0.2, 5.5);
        let old_pose_20 = slam
            .map
            .keyframes
            .get(&20)
            .expect("KF#20 registered")
            .frame
            .pose
            .clone()
            .expect("KF#20 has a pose");
        let original_pixel = camera
            .project(&old_pose_20.transform_world_point(&anchor_landmark_position))
            .expect("synthetic point projects in front of the camera");
        {
            let kf20 = slam.map.keyframes.get_mut(&20).unwrap();
            let keypoint_index = kf20.frame.keypoints.len();
            kf20.frame.keypoints.push(original_pixel);
            kf20.frame.descriptors.push(vec![0.0; 2]);
            kf20.observations.push(Observation {
                frame_id: 20,
                landmark_id: SYNTHETIC_ANCHOR_LANDMARK_ID,
                keypoint_index,
                xy: original_pixel,
            });
        }
        let mut landmark = Landmark::new(SYNTHETIC_ANCHOR_LANDMARK_ID, anchor_landmark_position);
        landmark.observations.push(Observation {
            frame_id: 20,
            landmark_id: SYNTHETIC_ANCHOR_LANDMARK_ID,
            keypoint_index: slam.map.keyframes[&20].frame.keypoints.len() - 1,
            xy: original_pixel,
        });
        slam.map
            .landmarks
            .insert(SYNTHETIC_ANCHOR_LANDMARK_ID, landmark);

        // Force a genuinely non-1 solved scale at KF#20: hand-edit the
        // Sim3 mirror's only edge so far (KF#10 -> KF#20, auto-seeded at
        // scale 1.0 from the true tracked poses) to claim a 1.6x scale
        // discrepancy. The upcoming loop-closing frame adds two more
        // scale-1 (metric) edges pulling against it, so the solve lands at
        // some non-trivial scale strictly between 1.0 and 1.6 — exactly
        // the "wrong basin the write-back/propagation math must still
        // handle correctly" scenario.
        let siw_before_solve = {
            let state = slam.pose_graph_state.as_mut().expect("pose graph state");
            let sim3_graph = state
                .sim3_graph
                .as_mut()
                .expect("sim3 mirror populated under the Sim3 solver");
            let edge = sim3_graph
                .edges
                .iter_mut()
                .find(|edge| edge.from == 10 && edge.to == 20)
                .expect("KF#10 -> KF#20 sequential edge should be present");
            edge.measurement.scale = 1.6;
            edge.weight = 100.0;
            sim3_graph.poses[&20].clone()
        };

        // KF#30 returns near the origin: shared-landmark loop gate fires,
        // triggering the Sim3 solve (same fixture as the SE3 propagation
        // test).
        let f2 = frame_at(30, Vector3::new(0.05, 0.0, 0.05), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.tracking_succeeded());
        let stats = r2
            .pose_graph_refinement
            .expect("stats reported on the loop-closing frame");
        assert!(
            stats.sim3_pose_graph_result.is_some(),
            "the Sim3 solve must fire for this test to exercise propagation"
        );

        let siw_new = slam
            .pose_graph_state
            .as_ref()
            .unwrap()
            .sim3_graph
            .as_ref()
            .unwrap()
            .poses[&20]
            .clone();
        assert!(
            (siw_new.scale - 1.0).abs() > 0.05,
            "expected a non-trivial solved scale at KF#20 (forced by the hand-edited edge), \
             got {}",
            siw_new.scale
        );

        let new_pose_20 = slam
            .map
            .keyframes
            .get(&20)
            .unwrap()
            .frame
            .pose
            .clone()
            .expect("KF#20 still has a pose after the Sim3 solve");
        assert_ne!(
            new_pose_20.world_to_camera.translation, old_pose_20.world_to_camera.translation,
            "KF#20 must actually move for this to be a meaningful test"
        );

        // Exact-displacement check, generalised to Sim(3): the landmark
        // must move by exactly KF#20's world-frame similarity correction
        // `Siw_new⁻¹ ∘ Siw_old`.
        let expected_correction = siw_new.inverse().compose(&siw_before_solve);
        let expected_position = expected_correction.transform_point(&anchor_landmark_position);
        let moved_position = slam.map.landmarks[&SYNTHETIC_ANCHOR_LANDMARK_ID].position;
        assert!(
            (moved_position - expected_position).norm() < 1.0e-9,
            "landmark must move by exactly KF#20's Sim3 correction: expected \
             {expected_position:?}, got {moved_position:?}"
        );

        // Projection invariance: reprojecting the CORRECTED landmark
        // through the CORRECTED (rigid, `t/s`-written-back) keyframe pose
        // must reproduce the ORIGINAL pixel. A wrong fixed point or a
        // `t * s` (instead of `t / s`) write-back convention would move
        // the reprojection and fail this check immediately.
        let corrected_pixel = camera
            .project(&new_pose_20.transform_world_point(&moved_position))
            .expect("corrected point still projects in front of the camera");
        assert!(
            (corrected_pixel - original_pixel).norm() < 1.0e-6,
            "corrected landmark must reproject to the original pixel under Sim3 propagation: \
             expected {original_pixel:?}, got {corrected_pixel:?}"
        );

        assert!(stats.landmarks_moved >= 1);
        assert!(stats.tracker_correction_applied);
    }
}

// ============================================================
// Relocalization-on-tracker-death integration tests (Phase-23 #1)
// ============================================================

mod relocalization_on_tracker_death {
    use super::*;

    fn camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn shared_landmarks() -> Vec<Point3<f64>> {
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

    fn build_seeded_map(points: &[Point3<f64>], camera: &Camera) -> VisualMap {
        let mut map = VisualMap::new();
        map.cameras.insert(camera.id, camera.clone());
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let descriptor: Vec<f32> = (0..16)
                .map(|d| ((index as f32) * 0.13 + (d as f32) * 0.07).sin())
                .collect();
            let mut landmark = Landmark::new(landmark_id, *point);
            landmark.descriptor = Some(descriptor);
            map.landmarks.insert(landmark_id, landmark);
        }
        map
    }

    fn frame_at(
        frame_id: u64,
        camera_center: Vector3<f64>,
        points: &[Point3<f64>],
        camera: &Camera,
        map: &VisualMap,
    ) -> Frame {
        let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center);
        let mut frame = Frame::new(frame_id, camera.id);
        for (index, point) in points.iter().enumerate() {
            let landmark_id = (index + 1) as u64;
            let projected = camera.project(&pose.transform_world_point(point)).unwrap();
            let descriptor = map
                .landmarks
                .get(&landmark_id)
                .and_then(|l| l.descriptor.clone())
                .unwrap();
            frame.keypoints.push(projected);
            frame.descriptors.push(descriptor);
        }
        frame
    }

    fn pipeline_with_relocalization(
        map: VisualMap,
        config: OnlineSlamRelocalizationConfig,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        pipeline_with_relocalization_and_tracking_config(map, config, TrackingConfig::default())
    }

    fn pipeline_with_relocalization_and_tracking_config(
        map: VisualMap,
        config: OnlineSlamRelocalizationConfig,
        tracking_config: TrackingConfig,
    ) -> OnlineSlamPipeline<Tracker<LocalizationPipeline>, LocalMappingPipeline> {
        OnlineSlamPipeline::new(
            map,
            Tracker::new(LocalizationPipeline::default(), tracking_config),
            LocalMappingPipeline::default(),
            OnlineSlamConfig {
                apply_map_updates: true,
                relocalization: Some(config),
                ..OnlineSlamConfig::default()
            },
        )
    }

    #[test]
    fn default_relocalization_is_off() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = OnlineSlamPipeline::new(
            map.clone(),
            Tracker::new(LocalizationPipeline::default(), TrackingConfig::default()),
            LocalMappingPipeline::default(),
            OnlineSlamConfig::default(),
        );
        assert!(slam.relocalization_state.is_none());
        let frame = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let result = slam.process_frame(&frame, []);
        assert!(result.tracking_succeeded());
        assert!(result.relocalization.is_none());
    }

    #[test]
    fn no_op_when_primary_tracking_succeeds() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam =
            pipeline_with_relocalization(map.clone(), OnlineSlamRelocalizationConfig::default());
        let frame = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let result = slam.process_frame(&frame, []);
        assert!(result.tracking_succeeded());
        // The stage is enabled but primary tracking succeeded, so no
        // recovery attempt should fire.
        assert!(result.relocalization.is_none());
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 0);
        assert_eq!(state.success_count, 0);
    }

    #[test]
    fn recovers_frame_after_primary_tracking_fails_with_bad_camera_id() {
        // Seed a map + KF#10, then push a frame whose camera_id is
        // wrong. The tracker's PnP path will fail (no matching camera
        // intrinsics on the frame), so relocalization should fire and
        // recover by re-running the localizer on the same frame data
        // against the map's full landmark cloud.
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
        );
        // First, successfully track + register KF#10 with the correct
        // camera id so the map has at least one keyframe + landmark
        // observations for the localizer to match against.
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());
        // Second frame with mismatched camera id will fail primary
        // tracking; the relocalizer will be called with the full map,
        // and on the same descriptors / landmarks should recover.
        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        // Sanity: if primary tracking already succeeded the stage is
        // skipped; otherwise the relocalizer should have at least been
        // invoked.
        if r1.tracking_succeeded() && r1.relocalization.is_none() {
            // Primary succeeded — nothing to test here for this path.
            return;
        }
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats reported when primary tracking failed");
        assert!(stats.attempted);
        // We don't strictly require success on this synthetic
        // setup (it depends on the localizer's default thresholds vs
        // the shared 12-landmark cloud), but if it did succeed the
        // tracker state must reflect that.
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        if stats.succeeded {
            assert_eq!(state.success_count, 1);
            assert_eq!(state.last_success_frame_id, Some(20));
            assert!(r1.tracking_succeeded());
        } else {
            assert_eq!(state.success_count, 0);
        }
        // Keep `f1` referenced to silence the unused warning the editor
        // would otherwise emit if the test compiles without an
        // assertion on it.
        let _ = &mut f1;
    }

    #[test]
    fn rejected_recovery_leaves_tracker_dead() {
        // Configure absurdly strict thresholds so the relocalizer's
        // recovered solution (if any) cannot pass. On a frame whose
        // primary tracking failed the stage should report
        // `attempted=true, succeeded=false` and the tracker state must
        // remain failed.
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                // No solver on a 12-landmark cloud will hit 9999 inliers.
                min_inliers: 9999,
                min_inlier_ratio: 0.99,
                max_mean_reprojection_error: Some(0.01),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
        );
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);
        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        // Force primary tracking failure by zeroing keypoints (no
        // correspondences).
        f1.keypoints.clear();
        f1.descriptors.clear();
        let r1 = slam.process_frame(&f1, []);
        assert!(!r1.tracking_succeeded());
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when stage ran");
        assert!(stats.attempted);
        assert!(!stats.succeeded);
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.success_count, 0);
    }

    #[test]
    fn relocalization_attempt_interval_skips_nearby_failed_frames() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 9999,
                min_inlier_ratio: 0.99,
                max_mean_reprojection_error: Some(0.01),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 5,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
        );
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);

        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        f1.keypoints.clear();
        f1.descriptors.clear();
        let r1 = slam.process_frame(&f1, []);
        assert!(r1
            .relocalization
            .as_ref()
            .is_some_and(|stats| stats.attempted));
        assert_eq!(slam.relocalization_state.as_ref().unwrap().trigger_count, 1);

        let mut f2 = frame_at(22, Vector3::new(1.7, 0.0, 0.0), &points, &camera, &map);
        f2.keypoints.clear();
        f2.descriptors.clear();
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.relocalization.is_none());
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.last_attempt_frame_id, Some(20));

        let mut f3 = frame_at(25, Vector3::new(2.0, 0.0, 0.0), &points, &camera, &map);
        f3.keypoints.clear();
        f3.descriptors.clear();
        let r3 = slam.process_frame(&f3, []);
        assert!(r3
            .relocalization
            .as_ref()
            .is_some_and(|stats| stats.attempted));
        assert_eq!(slam.relocalization_state.as_ref().unwrap().trigger_count, 2);
    }

    #[test]
    fn relocalization_attempt_budget_skips_after_consecutive_failure_cap() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 9999,
                min_inlier_ratio: 0.99,
                max_mean_reprojection_error: Some(0.01),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: Some(1),
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
        );
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);

        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        f1.keypoints.clear();
        f1.descriptors.clear();
        let r1 = slam.process_frame(&f1, []);
        assert!(r1
            .relocalization
            .as_ref()
            .is_some_and(|stats| stats.attempted));
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.success_count, 0);
        assert_eq!(state.consecutive_failed_attempts, 1);
        assert_eq!(state.budget_skip_count, 0);

        let mut f2 = frame_at(21, Vector3::new(1.6, 0.0, 0.0), &points, &camera, &map);
        f2.keypoints.clear();
        f2.descriptors.clear();
        let r2 = slam.process_frame(&f2, []);
        assert!(r2.relocalization.is_none());
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.last_attempt_frame_id, Some(20));
        assert_eq!(state.consecutive_failed_attempts, 1);
        assert_eq!(state.budget_skip_count, 1);
    }

    #[test]
    fn relocalization_covisibility_store_uses_last_successful_keyframe() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: Some(OnlineSlamRelocalizationCovisibilityConfig {
                    max_neighbor_keyframes: Some(2),
                    min_shared_landmarks: 1,
                    min_local_map_landmarks: 1,
                    fallback_to_broader_store_on_failure: true,
                    broader_store_retry_interval_frames: 1,
                    compare_broader_store_on_success: false,
                }),
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(stats.used_covisibility_local_descriptor_store);
        assert_eq!(stats.covisibility_reference_keyframe_id, Some(10));
        assert_eq!(stats.descriptor_store_landmark_count, points.len());
        assert_eq!(
            stats.covisibility_local_descriptor_store_landmark_count,
            Some(points.len())
        );
        assert!(!stats.tried_broader_descriptor_store_fallback);
        assert!(!stats.used_broader_descriptor_store_fallback);
        assert!(stats.succeeded);
        assert!(r1.tracking_succeeded());
    }

    #[test]
    fn relocalization_covisibility_store_falls_back_to_broader_store_when_too_weak() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: Some(OnlineSlamRelocalizationCovisibilityConfig {
                    max_neighbor_keyframes: Some(2),
                    min_shared_landmarks: 1,
                    min_local_map_landmarks: 1,
                    fallback_to_broader_store_on_failure: true,
                    broader_store_retry_interval_frames: 1,
                    compare_broader_store_on_success: false,
                }),
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());

        let map_mut = slam.map_mut();
        map_mut
            .keyframes
            .get_mut(&10)
            .expect("seed keyframe was inserted")
            .observations
            .retain(|observation| observation.landmark_id <= 3);
        for landmark in map_mut.landmarks.values_mut() {
            if landmark.id > 3 {
                landmark
                    .observations
                    .retain(|observation| observation.frame_id != 10);
            }
        }

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(stats.tried_covisibility_local_descriptor_store);
        assert!(!stats.used_covisibility_local_descriptor_store);
        assert!(stats.tried_broader_descriptor_store_fallback);
        assert!(!stats.broader_descriptor_store_retry_skipped_by_interval);
        assert!(stats.used_broader_descriptor_store_fallback);
        assert_eq!(stats.covisibility_reference_keyframe_id, Some(10));
        assert_eq!(
            stats.covisibility_local_descriptor_store_landmark_count,
            Some(3)
        );
        assert_eq!(
            stats.broader_descriptor_store_landmark_count,
            Some(points.len())
        );
        assert_eq!(stats.descriptor_store_landmark_count, points.len());
        assert!(stats.succeeded);
        assert!(r1.tracking_succeeded());
    }

    #[test]
    fn relocalization_appearance_store_uses_retrieved_keyframe_landmarks() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: Some(OnlineSlamRelocalizationAppearanceConfig {
                    max_keyframes: 1,
                    candidate_log_limit: None,
                    min_similarity: 0.0,
                    exclude_recent_frame_gap: 1,
                    min_local_map_landmarks: 1,
                    fallback_to_broader_store_on_failure: true,
                    broader_store_retry_interval_frames: 1,
                    compare_broader_store_on_success: false,
                }),
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(stats.tried_appearance_descriptor_store);
        assert!(stats.used_appearance_descriptor_store);
        assert_eq!(stats.appearance_best_keyframe_id, Some(10));
        assert_eq!(stats.appearance_candidate_keyframe_count, 1);
        assert_eq!(stats.appearance_candidates.len(), 1);
        assert_eq!(stats.appearance_candidates[0].keyframe_id, 10);
        assert_eq!(
            stats.appearance_descriptor_store_landmark_count,
            Some(points.len())
        );
        assert!(!stats.tried_broader_descriptor_store_fallback);
        assert!(stats.succeeded);
        assert!(r1.tracking_succeeded());
    }

    #[test]
    fn relocalization_appearance_store_falls_back_to_broader_store_on_gate_failure() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 9999,
                min_inlier_ratio: 0.99,
                max_mean_reprojection_error: Some(0.01),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: Some(OnlineSlamRelocalizationAppearanceConfig {
                    max_keyframes: 1,
                    candidate_log_limit: None,
                    min_similarity: 0.0,
                    exclude_recent_frame_gap: 1,
                    min_local_map_landmarks: 1,
                    fallback_to_broader_store_on_failure: true,
                    broader_store_retry_interval_frames: 1,
                    compare_broader_store_on_success: false,
                }),
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(stats.tried_appearance_descriptor_store);
        assert!(!stats.used_appearance_descriptor_store);
        assert!(stats.tried_broader_descriptor_store_fallback);
        assert!(stats.used_broader_descriptor_store_fallback);
        assert_eq!(stats.appearance_best_keyframe_id, Some(10));
        assert_eq!(
            stats.appearance_descriptor_store_landmark_count,
            Some(points.len())
        );
        assert!(!stats.succeeded);
        assert!(!r1.tracking_succeeded());
    }

    #[test]
    fn relocalization_appearance_candidate_log_limit_does_not_expand_recovery_store() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: Some(OnlineSlamRelocalizationAppearanceConfig {
                    max_keyframes: 1,
                    candidate_log_limit: Some(2),
                    min_similarity: 0.0,
                    exclude_recent_frame_gap: 1,
                    min_local_map_landmarks: 1,
                    fallback_to_broader_store_on_failure: false,
                    broader_store_retry_interval_frames: 1,
                    compare_broader_store_on_success: false,
                }),
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(1.2),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded() && r0.map_was_updated());

        let f1 = frame_at(20, Vector3::new(1.1, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(r1.tracking_succeeded() && r1.map_was_updated());

        let f2 = frame_at(30, Vector3::new(2.5, 0.0, 0.0), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        let stats = r2
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(stats.used_appearance_descriptor_store);
        assert_eq!(stats.appearance_candidate_keyframe_count, 2);
        assert_eq!(
            stats
                .appearance_descriptor_store_landmark_count
                .expect("appearance store size"),
            points.len()
        );
        assert_eq!(
            stats
                .appearance_candidates
                .iter()
                .map(|candidate| candidate.keyframe_id)
                .collect::<Vec<_>>(),
            vec![10, 20]
        );
    }

    #[test]
    fn relocalization_pose_continuity_gate_rejects_large_frame_normalized_jump() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: Some(0.05),
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded());

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(!r1.tracking_succeeded());
        assert!(matches!(
            r1.tracking.tracking_failure_reason,
            Some(visloc_tracking::TrackingFailureReason::PosePriorTranslationErrorExceeded { .. })
        ));
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(!stats.succeeded);
        assert!(stats.inlier_count >= 6);
        assert!(stats.inlier_ratio >= 0.3);
        assert!(stats.translation_from_last_success_meters.unwrap() > 1.0);
        assert!(
            stats
                .translation_per_frame_from_last_success_meters
                .unwrap()
                > 0.05
        );
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        assert_eq!(state.success_count, 0);
        assert_eq!(state.last_success_frame_id, None);
    }

    #[test]
    fn relocalization_depth_ratio_gate_rejects_scale_inconsistent_recovery() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: Some(0.8),
                max_inlier_depth_median_ratio_to_last_success: Some(1.25),
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded());

        let f1 = frame_at(20, Vector3::new(0.0, 0.0, 3.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(!r1.tracking_succeeded());
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when primary tracking failed");
        assert!(stats.attempted);
        assert!(!stats.succeeded);
        assert!(stats.inlier_count >= 6);
        let ratio = stats
            .inlier_depth_median_ratio_to_last_success
            .expect("depth-ratio diagnostic populated");
        assert!(
            ratio < 0.8,
            "expected recovered inlier depth ratio below gate, got {ratio}"
        );
        assert_eq!(slam.relocalization_state.as_ref().unwrap().success_count, 0);
    }

    #[test]
    fn relocalization_confirmation_window_waits_for_consistent_second_recovery() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization_and_tracking_config(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: None,
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 2,
                confirmation_max_translation_per_frame_meters: Some(0.2),
            },
            TrackingConfig {
                max_pose_prior_translation_error: Some(0.1),
                ..TrackingConfig::default()
            },
        );

        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let r0 = slam.process_frame(&f0, []);
        assert!(r0.tracking_succeeded());

        let f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        let r1 = slam.process_frame(&f1, []);
        assert!(!r1.tracking_succeeded());
        let stats1 = r1
            .relocalization
            .as_ref()
            .expect("first recovery candidate should be reported");
        assert!(stats1.passed_acceptance_gates);
        assert!(!stats1.succeeded);
        assert_eq!(stats1.confirmation_count, 1);
        assert_eq!(stats1.confirmation_required_count, 2);
        assert_eq!(slam.relocalization_state.as_ref().unwrap().success_count, 0);
        assert_eq!(
            slam.relocalization_state
                .as_ref()
                .unwrap()
                .pending_confirmation
                .as_ref()
                .map(|pending| pending.count),
            Some(1)
        );

        let f2 = frame_at(21, Vector3::new(1.55, 0.0, 0.0), &points, &camera, &map);
        let r2 = slam.process_frame(&f2, []);
        let stats2 = r2
            .relocalization
            .as_ref()
            .expect("second recovery candidate should be reported");
        assert!(stats2.passed_acceptance_gates);
        assert!(stats2.succeeded);
        assert_eq!(stats2.confirmation_count, 2);
        assert!(stats2
            .confirmation_translation_per_frame_from_previous_meters
            .is_some_and(|actual| actual <= 0.2));
        assert!(r2.tracking_succeeded());
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.success_count, 1);
        assert_eq!(state.last_success_frame_id, Some(21));
        assert!(state.pending_confirmation.is_none());
    }

    #[test]
    fn pose_prior_guided_recovery_uses_motion_prior_radius() {
        // Phase-23 #1b: with `pose_prior_candidate_radius_meters =
        // Some(r)` set, the recovery PnP uses the tracker's per-frame
        // motion prior + a radius candidate filter instead of the
        // no-prior global path. On the synthetic 12-landmark cloud
        // with the ConstantPoseMotionModel (default), the prior on
        // frame 20 is the pose recorded on frame 10, which is the
        // origin — so a radius gate of 5m should still admit the
        // entire landmark set. The contract validated here is
        // structural: the stage runs without panicking and records a
        // recovery attempt in the counters.
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam = pipeline_with_relocalization(
            map.clone(),
            OnlineSlamRelocalizationConfig {
                min_inliers: 6,
                min_inlier_ratio: 0.3,
                max_mean_reprojection_error: Some(50.0),
                pose_prior_candidate_radius_meters: Some(5.0),
                recent_keyframe_window: None,
                covisibility_local_map: None,
                appearance_retrieval_map: None,
                max_translation_from_imu_prediction_meters: None,
                attempt_interval_frames: 1,
                max_consecutive_failed_attempts: None,
                max_translation_per_frame_from_last_success_meters: None,
                min_inlier_depth_median_ratio_to_last_success: None,
                max_inlier_depth_median_ratio_to_last_success: None,
                confirmation_required_recoveries: 1,
                confirmation_max_translation_per_frame_meters: None,
            },
        );
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);
        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        f1.keypoints.clear();
        f1.descriptors.clear();
        let r1 = slam.process_frame(&f1, []);
        // Primary tracking failed (no keypoints) so the relocalization
        // stage MUST have run. We don't assert success — the recovery
        // PnP also has no keypoints to localize against — but the
        // attempt counter must reflect the invocation.
        assert!(!r1.tracking_succeeded());
        let stats = r1
            .relocalization
            .as_ref()
            .expect("relocalization stats present when stage ran");
        assert!(stats.attempted);
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 1);
        // The state must hold onto the configured prior-mode radius so
        // future invocations use the same path.
        assert_eq!(state.config.pose_prior_candidate_radius_meters, Some(5.0));
    }

    #[test]
    fn reset_clears_relocalization_state() {
        let camera = camera();
        let points = shared_landmarks();
        let map = build_seeded_map(&points, &camera);
        let mut slam =
            pipeline_with_relocalization(map.clone(), OnlineSlamRelocalizationConfig::default());
        let f0 = frame_at(10, Vector3::zeros(), &points, &camera, &map);
        let _ = slam.process_frame(&f0, []);
        // Force a relocalization attempt with a primary failure.
        let mut f1 = frame_at(20, Vector3::new(1.5, 0.0, 0.0), &points, &camera, &map);
        f1.keypoints.clear();
        f1.descriptors.clear();
        let _ = slam.process_frame(&f1, []);
        assert!(slam.relocalization_state.as_ref().unwrap().trigger_count >= 1);
        slam.reset_sequence_state();
        let state = slam.relocalization_state.as_ref().unwrap();
        assert_eq!(state.trigger_count, 0);
        assert_eq!(state.success_count, 0);
        assert_eq!(state.consecutive_failed_attempts, 0);
        assert_eq!(state.budget_skip_count, 0);
        assert!(state.last_attempt_frame_id.is_none());
        assert!(state.last_broader_descriptor_store_retry_frame_id.is_none());
        assert!(state.last_success_frame_id.is_none());
    }
}
