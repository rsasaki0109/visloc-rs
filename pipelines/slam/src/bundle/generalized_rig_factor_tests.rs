use nalgebra::{Point3, UnitQuaternion, Vector3};

use super::*;

#[test]
fn feasible_backtrack_requires_finite_decrease_and_preserved_observations() {
    assert!(feasible_backtrack_accepts(10.0, 9.0, 0, 0, true));
    for cost in [10.0, 11.0, f64::NAN, f64::INFINITY] {
        assert!(!feasible_backtrack_accepts(10.0, cost, 0, 0, true));
    }
    assert!(!feasible_backtrack_accepts(f64::INFINITY, 9.0, 0, 0, true));
    assert!(!feasible_backtrack_accepts(10.0, 9.0, 0, 1, true));
    assert!(!feasible_backtrack_accepts(10.0, 9.0, 1, 1, false));
}

// Run this fixture in separate test processes with the experimental env
// flag absent and set to 1; never mutate process environment in a test.
#[test]
fn feasible_backtrack_production_rescue_and_exhaustion() {
    let enabled = std::env::var("VISLOC_SFM_BA_FEASIBLE_BACKTRACK").as_deref() == Ok("1");
    for target_x in [30.0, 1000.0] {
        for joint in [false, true] {
            let camera = Camera::pinhole(1, 848, 800, 1.0, 1.0, 0.0, 0.0);
            let mut ba = BundleAdjustment::new(camera.clone());
            ba.add_pose(0, Pose::identity());
            if joint {
                ba.fix_pose_rotation(0);
            } else {
                ba.fix_pose(0);
            }
            ba.add_landmark(0, Point3::new(1.0, 0.0, 0.1));
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: 0,
                landmark_id: 0,
                xy: Point2::new(target_x, 0.0),
                camera,
                sensor_from_rig: SE3::identity(),
            });
            let before = ba.clone();
            let result = ba
                .optimize(&BaConfig {
                    max_iterations: 1,
                    initial_lambda: Some(0.01),
                    linear_solver: LinearSolver::Sparse,
                    ..BaConfig::default()
                })
                .unwrap();
            assert_eq!(result.iterations.len(), 1);
            assert_eq!(
                ba.poses[&0].world_to_camera.rotation, before.poses[&0].world_to_camera.rotation,
                "fixed rotation moved"
            );
            if !joint {
                assert_eq!(ba.poses, before.poses, "fixed pose moved");
            }
            assert_eq!(ba.nonprojectable_observation_count(), 0);
            if enabled && target_x == 30.0 {
                assert!(result.iterations[0].step_accepted);
                assert!(result.final_cost < result.initial_cost);
                let actual_step = (ba.landmarks[&0] - before.landmarks[&0]).norm();
                assert!(actual_step > 0.0);
                assert!((result.iterations[0].max_landmark_step - actual_step).abs() < 1e-12);
                if joint {
                    let pose_step = (ba.poses[&0].world_to_camera.translation
                        - before.poses[&0].world_to_camera.translation)
                        .norm();
                    assert!(pose_step > 0.0);
                    assert!((result.iterations[0].max_pose_step - pose_step).abs() < 1e-12);
                }
            } else {
                assert!(!result.iterations[0].step_accepted);
                assert_eq!(ba, before, "rejection must restore the complete problem");
                assert_eq!(result.final_cost, result.initial_cost);
            }
        }
    }
}

#[test]
fn infeasible_rig_sample_matches_predicate_and_is_bounded() {
    let camera = Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5);
    let mut ba = BundleAdjustment::new(camera.clone());
    ba.add_pose(0, Pose::identity());
    for id in 0..20 {
        ba.add_landmark(id, Point3::new(0.0, 0.0, if id == 0 { 3.0 } else { -1.0 }));
        ba.add_rig_observation(BaRigObservation {
            keyframe_id: 0,
            landmark_id: id,
            xy: Point2::new(425.5, 398.5),
            camera: camera.clone(),
            sensor_from_rig: SE3::identity(),
        });
    }
    assert_eq!(ba.nonprojectable_observation_count(), 19);
    let sample = ba.nonprojectable_rig_sample(16);
    assert_eq!(sample.len(), 16);
    assert_eq!(sample[0], (1, 0, 1, Some(-1.0)));
    assert_eq!(sample[15], (16, 0, 16, Some(-1.0)));
    assert!(ba.nonprojectable_rig_sample(0).is_empty());
    assert_eq!(ba.nonprojectable_rig_sample(100).len(), 19);
    assert_eq!(ba.nonprojectable_observation_count(), 19);
}

#[test]
fn arbitrary_sensor_factors_refine_one_shared_body_pose() {
    let camera_left = Camera::pinhole(1, 848, 800, 285.0, 286.0, 425.5, 398.5);
    let camera_right = Camera::pinhole(2, 848, 800, 284.8, 286.1, 428.0, 397.5);
    let sensors = [
        (camera_left.clone(), SE3::identity()),
        (
            camera_right,
            SE3::new(UnitQuaternion::identity(), Vector3::new(-0.20, 0.0, 0.0)),
        ),
    ];
    let truth = [
        Pose::identity(),
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.01, -0.04, 0.02),
            Vector3::new(-0.35, 0.03, 0.01),
        ),
    ];
    let mut problem = BundleAdjustment::new(camera_left);
    problem.add_pose(0, truth[0].clone());
    problem.add_pose(
        1,
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.03, -0.01, -0.01),
            Vector3::new(-0.27, -0.02, 0.04),
        ),
    );
    problem.fix_pose(0);
    for landmark in 0..24u64 {
        let point = Point3::new(
            (landmark % 6) as f64 * 0.25 - 0.6,
            (landmark / 6) as f64 * 0.22 - 0.3,
            4.0 + (landmark % 5) as f64 * 0.15,
        );
        problem.add_landmark(
            landmark,
            Point3::from(point.coords + Vector3::new(0.02, -0.01, 0.03)),
        );
        for (frame, frame_pose) in truth.iter().enumerate() {
            for (camera, sensor_from_rig) in &sensors {
                let point_rig = frame_pose.transform_world_point(&point);
                let pixel = camera
                    .project(&sensor_from_rig.transform_point(&point_rig))
                    .unwrap();
                problem.add_rig_observation(BaRigObservation {
                    keyframe_id: frame as u64,
                    landmark_id: landmark,
                    xy: pixel,
                    camera: camera.clone(),
                    sensor_from_rig: sensor_from_rig.clone(),
                });
            }
        }
    }
    let initial_cost = problem.cost();
    let initial_center_error =
        (problem.poses[&1].camera_center_world() - truth[1].camera_center_world()).norm();
    let result = problem
        .optimize(&BaConfig {
            max_iterations: 30,
            ..BaConfig::default()
        })
        .unwrap();
    let final_center_error =
        (problem.poses[&1].camera_center_world() - truth[1].camera_center_world()).norm();
    assert!(result.final_cost < initial_cost * 1.0e-6);
    assert!(final_center_error < initial_center_error * 1.0e-3);
}
