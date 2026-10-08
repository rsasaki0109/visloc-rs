use super::*;
use nalgebra::UnitQuaternion;

/// Deterministic `[0, 1)` pseudo-random value (GLSL-style sine hash) 窶・
/// avoids pulling in a `rand` dependency just to scatter synthetic
/// points/poses reproducibly.
fn pseudo_rand(seed: u64) -> f64 {
    let x = (seed as f64 + 1.0) * 12.9898;
    let y = x.sin() * 43758.5453;
    y - y.floor()
}

/// Build a synthetic multi-camera, multi-landmark monocular BA problem:
/// `num_cameras` cameras translated along a horizontal baseline (plus a
/// small yaw each) all observe every one of `num_landmarks` landmarks
/// scattered in front of them, so every camera/landmark pair is a valid,
/// positive-depth observation. The first two poses are fixed (anchor +
/// scale, per the module doc's gauge-fixing rule); every other pose and
/// every landmark is then nudged away from the ground truth that
/// generated the observations by a small deterministic offset, so LM has
/// a real (if easy, well-conditioned) problem to converge on rather than
/// starting already at the optimum.
fn build_synthetic_ba(num_cameras: usize, num_landmarks: usize) -> BundleAdjustment {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut ba = BundleAdjustment::new(camera.clone());

    let center = (num_cameras as f64 - 1.0) / 2.0;
    let gt_poses: Vec<Pose> = (0..num_cameras)
        .map(|i| {
            let baseline = (i as f64 - center) * 0.4;
            let yaw = (i as f64 - center) * 0.02;
            let rotation = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
            let translation = Vector3::new(-baseline, 0.0, 0.0);
            Pose::from_world_to_camera(rotation, translation)
        })
        .collect();
    let gt_points: Vec<Point3<f64>> = (0..num_landmarks)
        .map(|j| {
            let x = (pseudo_rand(j as u64 * 3) - 0.5) * 6.0;
            let y = (pseudo_rand(j as u64 * 3 + 1) - 0.5) * 6.0;
            let z = 8.0 + pseudo_rand(j as u64 * 3 + 2) * 4.0;
            Point3::new(x, y, z)
        })
        .collect();

    for (i, pose) in gt_poses.iter().enumerate() {
        ba.add_pose(i as u64, pose.clone());
    }
    for (j, point) in gt_points.iter().enumerate() {
        ba.add_landmark(1_000_000 + j as u64, *point);
    }
    for (i, pose) in gt_poses.iter().enumerate() {
        for (j, point) in gt_points.iter().enumerate() {
            let xc = pose.transform_world_point(point);
            let xy = camera
                .project(&xc)
                .expect("synthetic landmarks stay in front of every camera");
            ba.add_observation(BaObservation {
                keyframe_id: i as u64,
                landmark_id: 1_000_000 + j as u64,
                xy,
            });
        }
    }

    ba.fix_pose(0);
    ba.fix_pose(1);

    // Nudge every non-fixed pose off ground truth.
    for i in 2..num_cameras {
        let dxi = Vector6::new(
            (pseudo_rand(i as u64 * 7) - 0.5) * 0.02,
            (pseudo_rand(i as u64 * 7 + 1) - 0.5) * 0.02,
            (pseudo_rand(i as u64 * 7 + 2) - 0.5) * 0.02,
            (pseudo_rand(i as u64 * 7 + 3) - 0.5) * 0.01,
            (pseudo_rand(i as u64 * 7 + 4) - 0.5) * 0.01,
            (pseudo_rand(i as u64 * 7 + 5) - 0.5) * 0.01,
        );
        let pose = ba.poses.get_mut(&(i as u64)).expect("pose was just added");
        pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&dxi));
    }
    // Nudge every landmark off ground truth.
    for j in 0..num_landmarks {
        let d = Vector3::new(
            (pseudo_rand(j as u64 * 11) - 0.5) * 0.05,
            (pseudo_rand(j as u64 * 11 + 1) - 0.5) * 0.05,
            (pseudo_rand(j as u64 * 11 + 2) - 0.5) * 0.05,
        );
        let point = ba
            .landmarks
            .get_mut(&(1_000_000 + j as u64))
            .expect("landmark was just added");
        *point = Point3::from(point.coords + d);
    }

    ba
}

/// `num_landmarks` clears `PARALLEL_MIN_LANDMARKS` and `num_cameras *
/// num_landmarks` clears `PARALLEL_MIN_OBSERVATIONS`, so every parallel
/// path in the module (assembly, Schur reduction, back-substitution) is
/// actually dispatched by these tests instead of falling through the
/// work gate.
const TEST_CAMERAS: usize = 6;
const TEST_LANDMARKS: usize = 2_500;

#[test]
fn parallel_config_defaults_to_off() {
    assert!(!BaConfig::default().parallel);
}

#[test]
fn serial_and_parallel_converge_to_the_same_result() {
    let mut ba_serial = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
    let mut ba_parallel = ba_serial.clone();

    let serial_config = BaConfig {
        max_iterations: 8,
        parallel: false,
        ..BaConfig::default()
    };
    let parallel_config = BaConfig {
        parallel: true,
        ..serial_config
    };

    let result_serial = ba_serial
        .optimize(&serial_config)
        .expect("serial BA should solve the synthetic problem");
    let result_parallel = ba_parallel
        .optimize(&parallel_config)
        .expect("parallel BA should solve the synthetic problem");

    assert!(result_serial.converged, "serial run should converge");
    assert!(result_parallel.converged, "parallel run should converge");

    // The parallel assembly / Schur-reduction / back-substitution paths
    // change only *how* the normal equations are computed, never the
    // summation order (see the module's "Parallelism" section), so the
    // two runs must land on bit-identical states -- a far tighter check
    // than the "~1e-9 relative" bar a reassociating design would need.
    assert_eq!(
        result_serial.final_cost, result_parallel.final_cost,
        "final cost must match exactly"
    );
    assert_eq!(
        result_serial.iterations.len(),
        result_parallel.iterations.len(),
        "iteration count must match exactly"
    );
    assert_eq!(
        ba_serial.poses, ba_parallel.poses,
        "poses must match exactly"
    );
    assert_eq!(
        ba_serial.landmarks, ba_parallel.landmarks,
        "landmarks must match exactly"
    );
}

#[test]
fn parallel_sparse_ba_keeps_pose_block_system_and_matches_serial() {
    let mut ba_serial = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
    let mut ba_parallel = ba_serial.clone();
    let serial_config = BaConfig {
        max_iterations: 8,
        linear_solver: LinearSolver::Sparse,
        parallel: false,
        ..BaConfig::default()
    };
    let parallel_config = BaConfig {
        parallel: true,
        ..serial_config
    };

    let result_serial = ba_serial
        .optimize(&serial_config)
        .expect("serial sparse BA should solve the synthetic problem");
    let result_parallel = ba_parallel
        .optimize(&parallel_config)
        .expect("parallel sparse BA should solve the synthetic problem");

    assert_eq!(result_serial.final_cost, result_parallel.final_cost);
    assert_eq!(result_serial.iterations, result_parallel.iterations);
    assert_eq!(ba_serial.poses, ba_parallel.poses);
    assert_eq!(ba_serial.landmarks, ba_parallel.landmarks);
}

#[test]
fn parallel_path_is_deterministic_across_runs() {
    let ba = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
    let mut ba_run_a = ba.clone();
    let mut ba_run_b = ba;

    let config = BaConfig {
        max_iterations: 8,
        parallel: true,
        ..BaConfig::default()
    };

    let result_a = ba_run_a
        .optimize(&config)
        .expect("parallel BA should solve the synthetic problem");
    let result_b = ba_run_b
        .optimize(&config)
        .expect("parallel BA should solve the synthetic problem");

    assert_eq!(
        result_a.final_cost, result_b.final_cost,
        "repeated parallel runs must produce bitwise-identical cost"
    );
    assert_eq!(
        ba_run_a.poses, ba_run_b.poses,
        "repeated parallel runs must produce bitwise-identical poses"
    );
    assert_eq!(
        ba_run_a.landmarks, ba_run_b.landmarks,
        "repeated parallel runs must produce bitwise-identical landmarks"
    );
}
