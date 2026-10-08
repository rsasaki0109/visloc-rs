use super::*;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn rejected_pose_diagonal_reuse_preserves_iteration_trace_and_state() {
    let original = make_problem();
    let config = BaConfig {
        max_iterations: 80,
        step_tolerance: 0.0,
        cost_tolerance: 0.0,
        linear_solver: LinearSolver::Sparse,
        ..BaConfig::default()
    };
    let mut control = original.clone();
    let mut candidate = original.clone();
    let expected = control.optimize(&config).unwrap();
    let actual = candidate
        .optimize(&BaConfig {
            reuse_rejected_pose_diagonal: true,
            ..config
        })
        .unwrap();
    assert!(expected.iterations.iter().any(|step| step.step_accepted));
    assert!(
        expected.iterations.windows(3).any(|steps| {
            !steps[0].step_accepted && steps[1].step_accepted && !steps[2].step_accepted
        }),
        "fixture must exercise rejection, acceptance and renewed rejection: {:?}",
        expected
            .iterations
            .iter()
            .map(|s| s.step_accepted)
            .collect::<Vec<_>>()
    );
    assert!(expected
        .iterations
        .windows(2)
        .any(|steps| !steps[0].step_accepted && !steps[1].step_accepted));
    assert_eq!(expected, actual);
    assert_eq!(control, candidate);
    // Dense dispatch remains outside the reuse policy.
    let dense = BaConfig {
        linear_solver: LinearSolver::Dense,
        ..config
    };
    let mut control = original.clone();
    let mut candidate = original;
    assert_eq!(
        control.optimize(&dense).unwrap(),
        candidate
            .optimize(&BaConfig {
                reuse_rejected_pose_diagonal: true,
                ..dense
            })
            .unwrap()
    );
    assert_eq!(control, candidate);
}

fn make_problem() -> BundleAdjustment {
    let camera = Camera::pinhole(7, 640, 480, 420.0, 418.0, 320.0, 240.0);
    let mut problem = BundleAdjustment::new(camera.clone());
    let truth_pose0 = Pose::identity();
    let truth_pose1 = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.01, -0.02, 0.015),
        Vector3::new(-0.18, 0.015, 0.02),
    );
    let truth_pose2 = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(-0.015, 0.025, -0.01),
        Vector3::new(-0.34, -0.01, 0.04),
    );
    problem.poses.insert(0, truth_pose0.clone());
    problem.poses.insert(
        1,
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.012, -0.018, 0.016),
            Vector3::new(-0.205, 0.022, 0.026),
        ),
    );
    problem.poses.insert(
        2,
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(-0.013, 0.023, -0.012),
            Vector3::new(-0.365, -0.004, 0.045),
        ),
    );
    problem.fixed_poses.insert(0);
    let points = [
        Point3::new(-0.8, -0.45, 3.8),
        Point3::new(-0.35, 0.55, 4.2),
        Point3::new(0.05, -0.25, 4.6),
        Point3::new(0.45, 0.35, 5.0),
        Point3::new(0.85, -0.5, 5.4),
        Point3::new(-0.65, 0.25, 5.8),
        Point3::new(0.25, 0.7, 6.2),
        Point3::new(0.7, 0.15, 6.8),
    ];
    for (landmark_id, truth_point) in points.into_iter().enumerate() {
        problem.landmarks.insert(
            landmark_id as u64,
            Point3::from(truth_point.coords + Vector3::new(0.006, -0.004, 0.008)),
        );
        for (keyframe_id, pose) in [(0, &truth_pose0), (1, &truth_pose1), (2, &truth_pose2)] {
            let xy = camera
                .project(&pose.transform_world_point(&truth_point))
                .expect("synthetic point must project");
            problem.observations.push(BaObservation {
                keyframe_id,
                landmark_id: landmark_id as u64,
                xy,
            });
        }
    }
    problem.fixed_landmarks.insert(0);
    problem
}

fn matrix_free_config() -> BaConfig {
    BaConfig {
        max_iterations: 4,
        linear_solver: LinearSolver::Sparse,
        ..BaConfig::default()
    }
}

#[test]
fn matrix_free_mono_matches_sparse_cost_and_keeps_anchor_fixed() {
    let problem = make_problem();
    let mut direct = problem.clone();
    let mut matrix_free = problem.clone();
    let config = matrix_free_config();
    let direct_result = direct.optimize(&config).unwrap();
    let matrix_result = matrix_free
        .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
        .unwrap();
    assert!(matrix_result.final_cost <= matrix_result.initial_cost);
    assert!(direct_result.final_cost <= direct_result.initial_cost);
    assert!((direct_result.final_cost - matrix_result.final_cost).abs() < 1.0e-6);
    assert_eq!(matrix_free.poses[&0], problem.poses[&0]);
    for id in 1..=2 {
        assert!(matrix_free.poses[&id] != problem.poses[&id]);
        assert!(
            (direct.poses[&id].world_to_camera.matrix()
                - matrix_free.poses[&id].world_to_camera.matrix())
            .norm()
                < 1.0e-5
        );
    }
    assert_eq!(matrix_free.landmarks[&0], problem.landmarks[&0]);
    for id in 1..8 {
        assert!((direct.landmarks[&id].coords - matrix_free.landmarks[&id].coords).norm() < 1.0e-5);
    }
    assert!(matrix_result
        .matrix_free_iterations
        .iter()
        .all(|iteration| iteration.pcg_failure.is_none()));

    let mut dense_config = config;
    dense_config.linear_solver = LinearSolver::Dense;
    let mut dense_dispatch = problem.clone();
    let dense_dispatch_result = dense_dispatch
        .optimize_matrix_free(&dense_config, MatrixFreeBaOptions::default())
        .unwrap();
    assert_eq!(matrix_result, dense_dispatch_result);

    let mut repeat = problem;
    let repeat_result = repeat
        .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
        .unwrap();
    assert_eq!(matrix_result, repeat_result);
    assert_eq!(matrix_free, repeat);
}

#[test]
fn honoring_dispatch_selects_the_matrix_free_backend_and_defaults_to_legacy() {
    let problem = make_problem();
    let config = matrix_free_config();

    // Off (default): identical to `optimize`.
    let mut legacy = problem.clone();
    let mut dispatch_off = problem.clone();
    let legacy_result = legacy.optimize(&config).unwrap();
    let off_result = dispatch_off
        .optimize_honoring_matrix_free(&config, None)
        .unwrap();
    assert_eq!(legacy_result, off_result);
    assert_eq!(legacy, dispatch_off);

    // On: identical to the explicit matrix-free entry point.
    let mut explicit = problem.clone();
    let mut dispatch_on = problem.clone();
    let explicit_result = explicit
        .optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default())
        .unwrap();
    let on_result = dispatch_on
        .optimize_honoring_matrix_free(
            &BaConfig {
                matrix_free_ba: true,
                ..matrix_free_config()
            },
            None,
        )
        .unwrap();
    assert_eq!(explicit_result.initial_cost, on_result.initial_cost);
    assert_eq!(explicit_result.final_cost, on_result.final_cost);
    assert_eq!(explicit_result.iterations, on_result.iterations);
    assert_eq!(explicit, dispatch_on);

    // Observation weights keep the weighted objective (no matrix-free).
    let weights = vec![1.0; explicit.observations.len()];
    let mut weighted = problem.clone();
    let mut weighted_reference = problem.clone();
    let expected = weighted_reference
        .optimize_with_observation_weights(&config, &weights)
        .unwrap();
    let actual = weighted
        .optimize_honoring_matrix_free(
            &BaConfig {
                matrix_free_ba: true,
                ..config
            },
            Some(&weights),
        )
        .unwrap();
    assert_eq!(expected, actual);
    assert_eq!(weighted_reference, weighted);
}

#[test]
fn cluster8_api_preserves_anchor_observations_and_repeatability() {
    let original = make_problem();
    let mut a = original.clone();
    let mut b = original.clone();
    let options = MatrixFreeBaOptions::default();
    let config = matrix_free_config();
    let result = a.optimize_matrix_free_cluster8(&config, options).unwrap();
    let repeat = b.optimize_matrix_free_cluster8(&config, options).unwrap();
    assert_eq!(result, repeat);
    assert_eq!(a, b);
    assert_eq!(a.poses[&0], original.poses[&0]);
    assert_eq!(a.observations, original.observations);
    assert!(result.final_cost < result.initial_cost);
    let mut invalid = original.clone();
    let mut bad = config;
    bad.refine_intrinsics = true;
    assert!(invalid
        .optimize_matrix_free_cluster8(&bad, options)
        .is_err());
    assert_eq!(invalid, original);
}

#[test]
fn cluster8_restart_api_zero_parity_and_invalid_limit_rollback() {
    let original = make_problem();
    let config = matrix_free_config();
    let options = MatrixFreeBaOptions::default();
    let mut a = original.clone();
    let mut b = original.clone();
    let plain = a.optimize_matrix_free_cluster8(&config, options).unwrap();
    let zero = b
        .optimize_matrix_free_cluster8_with_restart(
            &config,
            options,
            MatrixFreeBaRestartOptions {
                max_restarts_per_solve: 0,
            },
        )
        .unwrap();
    assert_eq!(plain, zero.ba);
    assert_eq!(a, b);
    let mut invalid = original.clone();
    assert!(invalid
        .optimize_matrix_free_cluster8_with_restart(
            &config,
            options,
            MatrixFreeBaRestartOptions {
                max_restarts_per_solve: 2
            }
        )
        .is_err());
    assert_eq!(invalid, original);
}

#[test]
fn matrix_free_supports_rectified_stereo_and_rig_visual_factors() {
    let mut stereo = make_problem();
    stereo.fixed_landmarks.clear();
    let camera = stereo.camera.clone();
    let baseline = 0.12;
    stereo.stereo_baseline = Some(baseline);
    stereo.stereo_observations = stereo
        .observations
        .drain(..)
        .map(|observation| {
            let pose = &stereo.poses[&observation.keyframe_id];
            let point = &stereo.landmarks[&observation.landmark_id];
            let xc = pose.transform_world_point(point);
            BaStereoObservation {
                keyframe_id: observation.keyframe_id,
                landmark_id: observation.landmark_id,
                xy: observation.xy,
                u_right: observation.xy.x - camera.params[0] * baseline / xc.z,
            }
        })
        .collect();
    let mut stereo_direct = stereo.clone();
    let stereo_direct_result = stereo_direct.optimize(&matrix_free_config()).unwrap();
    let stereo_result = stereo
        .optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default())
        .unwrap();
    assert!(stereo_result.final_cost.is_finite());
    assert!(stereo_result.final_cost < stereo_result.initial_cost);
    assert!((stereo_direct_result.final_cost - stereo_result.final_cost).abs() < 1.0e-6);

    let mut rig = make_problem();
    rig.fixed_landmarks.clear();
    let camera = rig.camera.clone();
    let sensor1 = SE3::new(
        UnitQuaternion::from_euler_angles(0.02, -0.01, 0.03),
        Vector3::new(0.22, -0.015, 0.01),
    );
    let observations = std::mem::take(&mut rig.observations);
    for observation in observations {
        let pose = &rig.poses[&observation.keyframe_id].world_to_camera;
        let point = &rig.landmarks[&observation.landmark_id];
        for sensor_from_rig in [SE3::identity(), sensor1.clone()] {
            let sensor_pose = sensor_from_rig.compose(pose);
            let xy = camera
                .project(&sensor_pose.transform_point(point))
                .expect("rig synthetic point must project");
            rig.rig_observations.push(BaRigObservation {
                keyframe_id: observation.keyframe_id,
                landmark_id: observation.landmark_id,
                xy,
                camera: camera.clone(),
                sensor_from_rig,
            });
        }
    }
    let rig_before_extrinsics: Vec<SE3> = rig
        .rig_observations
        .iter()
        .map(|observation| observation.sensor_from_rig.clone())
        .collect();
    rig.poses.get_mut(&1).unwrap().world_to_camera.translation.x += 0.03;
    rig.poses.get_mut(&2).unwrap().world_to_camera.translation.y -= 0.02;
    for id in 1..8 {
        rig.landmarks.get_mut(&id).unwrap().coords.z += 0.015;
    }
    let mut rig_scaled = rig.clone();
    rig_scaled.fixed_landmarks.insert(0);
    // Keep the gauge anchor pose fixed through `fixed_poses`, while
    // constraining the rotation of a genuinely variable pose so the
    // scaled path exercises the identity rotation rows.
    rig_scaled.fixed_pose_rotations.insert(1);
    let anchor_rig_pose = rig_scaled.poses[&0].clone();
    let fixed_rig_rotation = rig_scaled.poses[&1].world_to_camera.rotation;
    let fixed_rig_landmark = rig_scaled.landmarks[&0];
    let scaled_extrinsics: Vec<SE3> = rig_scaled
        .rig_observations
        .iter()
        .map(|observation| observation.sensor_from_rig.clone())
        .collect();
    let mut rig_direct = rig.clone();
    let rig_direct_result = rig_direct.optimize(&matrix_free_config()).unwrap();
    let rig_result = rig
        .optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default())
        .unwrap();
    assert!(rig_result.final_cost.is_finite());
    assert!(rig_result.final_cost < rig_result.initial_cost);
    assert!((rig_direct_result.final_cost - rig_result.final_cost).abs() < 1.0e-6);
    for id in 1..=2 {
        assert!(
            (rig_direct.poses[&id].world_to_camera.matrix()
                - rig.poses[&id].world_to_camera.matrix())
            .norm()
                < 1.0e-5
        );
    }
    for id in 0..8 {
        assert!((rig_direct.landmarks[&id].coords - rig.landmarks[&id].coords).norm() < 1.0e-5);
    }
    assert_eq!(
        rig.rig_observations
            .iter()
            .map(|observation| observation.sensor_from_rig.clone())
            .collect::<Vec<_>>(),
        rig_before_extrinsics
    );

    let scaled_result = rig_scaled
        .optimize_matrix_free_column_scaled(
            &matrix_free_config(),
            MatrixFreeBaColumnScalingOptions::default(),
        )
        .unwrap();
    assert!(scaled_result.ba.final_cost.is_finite());
    assert!(scaled_result.ba.final_cost <= scaled_result.ba.initial_cost);
    assert!(scaled_result
        .ba
        .iterations
        .iter()
        .any(|iteration| iteration.step_accepted));
    assert_eq!(rig_scaled.poses[&0], anchor_rig_pose);
    assert_eq!(
        rig_scaled.poses[&1].world_to_camera.rotation,
        fixed_rig_rotation
    );
    assert_eq!(rig_scaled.landmarks[&0], fixed_rig_landmark);
    assert_eq!(
        rig_scaled
            .rig_observations
            .iter()
            .map(|observation| observation.sensor_from_rig.clone())
            .collect::<Vec<_>>(),
        scaled_extrinsics
    );
    assert!(!scaled_result.scaling_iterations.is_empty());

    let mut rig_adaptive = rig.clone();
    rig_adaptive.fixed_landmarks.insert(0);
    rig_adaptive.fixed_pose_rotations.insert(1);
    let adaptive_anchor_pose = rig_adaptive.poses[&0].clone();
    let adaptive_fixed_rotation = rig_adaptive.poses[&1].world_to_camera.rotation;
    let adaptive_fixed_landmark = rig_adaptive.landmarks[&0];
    let adaptive_extrinsics: Vec<SE3> = rig_adaptive
        .rig_observations
        .iter()
        .map(|observation| observation.sensor_from_rig.clone())
        .collect();
    let adaptive_result = rig_adaptive
        .optimize_matrix_free_column_scaled_adaptive(
            &matrix_free_config(),
            MatrixFreeBaColumnScalingOptions::default(),
        )
        .unwrap();
    assert!(adaptive_result.ba.final_cost.is_finite());
    assert!(adaptive_result
        .ba
        .iterations
        .iter()
        .any(|iteration| iteration.step_accepted));
    assert_eq!(rig_adaptive.poses[&0], adaptive_anchor_pose);
    assert_eq!(
        rig_adaptive.poses[&1].world_to_camera.rotation,
        adaptive_fixed_rotation
    );
    assert_eq!(rig_adaptive.landmarks[&0], adaptive_fixed_landmark);
    assert_eq!(
        rig_adaptive
            .rig_observations
            .iter()
            .map(|observation| observation.sensor_from_rig.clone())
            .collect::<Vec<_>>(),
        adaptive_extrinsics
    );
}

#[test]
fn rig_qr_adapter_matches_existing_weighted_assembly_and_full_step() {
    let mut rig = make_problem();
    rig.fixed_landmarks.clear();
    rig.fixed_landmarks.insert(0);
    rig.fixed_pose_rotations.insert(1);
    let observations = std::mem::take(&mut rig.observations);
    for (index, obs) in observations.into_iter().enumerate() {
        for extrinsic in [
            SE3::identity(),
            SE3::new(
                UnitQuaternion::from_euler_angles(0.02, -0.03, 0.01),
                Vector3::new(0.2, 0.01, -0.02),
            ),
        ] {
            let sensor_pose = extrinsic.compose(&rig.poses[&obs.keyframe_id].world_to_camera);
            let mut xy = rig
                .camera
                .project(&sensor_pose.transform_point(&rig.landmarks[&obs.landmark_id]))
                .unwrap();
            xy.x += if index % 5 == 0 { 20.0 } else { 0.2 };
            xy.y -= 0.3;
            rig.rig_observations.push(BaRigObservation {
                keyframe_id: obs.keyframe_id,
                landmark_id: obs.landmark_id,
                xy,
                camera: rig.camera.clone(),
                sensor_from_rig: extrinsic,
            });
        }
    }
    let original = rig.clone();
    for kernel in [RobustKernel::None, RobustKernel::Huber { delta: 6.0 }] {
        let config = BaConfig {
            robust_kernel: kernel,
            ..matrix_free_config()
        };
        for lambda in [0.5, 100.0] {
            let qr = RigQrLinearization::new(&rig, &config, lambda).unwrap();
            assert_eq!(qr.observation_rows, 2 * rig.rig_observations.len());
            let landmarks: BTreeMap<_, _> = rig
                .landmarks
                .keys()
                .filter(|id| !rig.fixed_landmarks.contains(id))
                .enumerate()
                .map(|(slot, id)| (*id, slot))
                .collect();
            let build = || {
                let mut s = build_normal_equations(
                    &rig,
                    &rig.camera.intrinsics().unwrap(),
                    &qr.pose_index,
                    &landmarks,
                    &BTreeMap::new(),
                    &BTreeMap::new(),
                    &config.robust_kernel,
                    None,
                    false,
                    true,
                );
                constrain_fixed_pose_rotations(&rig.fixed_pose_rotations, &qr.pose_index, &mut s);
                s
            };
            let system = build();
            let schur = implicit_schur::ImplicitSchurOperator::new(&system, lambda).unwrap();
            let rhs = DVector::from_vec(qr.rhs().unwrap());
            assert!((&rhs - schur.rhs()).norm() < 1e-9 * (1.0 + schur.rhs().norm()));
            let n = qr.pose_index.len() * 6;
            let mut matrix = DMatrix::zeros(n, n);
            for col in 0..n {
                let mut x = DVector::zeros(n);
                x[col] = 1.0;
                let actual = DVector::from_vec(qr.apply(x.as_slice()).unwrap());
                let expected = schur.apply(&x).unwrap();
                assert!((&actual - &expected).norm() < 1e-9 * (1.0 + expected.norm()));
                matrix.column_mut(col).copy_from(&actual);
            }
            let dx = matrix.cholesky().unwrap().solve(&rhs);
            assert_eq!(qr.preconditioner.len(), qr.pose_index.len());
            let probe = DVector::from_fn(n, |i, _| (i as f64 + 0.25).sin());
            let expected = schur.apply_preconditioner(&probe).unwrap();
            assert!(
                (qr.precondition(&probe).unwrap() - &expected).norm()
                    < 1e-9 * (1.0 + expected.norm())
            );
            assert!(qr.precondition(&DVector::zeros(n + 1)).is_err());
            let options = implicit_schur::PcgOptions::default();
            let iterative = qr.solve(options).unwrap();
            let true_residual =
                &rhs - DVector::from_vec(qr.apply(iterative.solution.as_slice()).unwrap());
            assert!(true_residual.norm() <= iterative.target);
            assert!((&iterative.solution - &dx).norm() < 1e-8 * (1.0 + dx.norm()));
            assert_eq!(iterative, qr.solve(options).unwrap());
            assert!(qr
                .solve(implicit_schur::PcgOptions {
                    max_iterations: 0,
                    ..options
                })
                .is_err());
            let mut direct_system = build();
            let direct = solve_step(
                &mut direct_system,
                qr.pose_index.len(),
                landmarks.len(),
                0,
                0,
                lambda,
                LinearSolver::Sparse,
                false,
                &mut None,
            )
            .unwrap();
            assert!((&dx - &direct.0).norm() < 1e-8 * (1.0 + direct.0.norm()));
            let mut scratch = Vec::new();
            for (id, block) in &qr.blocks {
                let point = block.back_substitute(dx.as_slice(), &mut scratch).unwrap();
                if let Some(slot) = landmarks.get(id) {
                    assert!(
                        (Vector3::from(point.unwrap()) - direct.1.fixed_rows::<3>(slot * 3)).norm()
                            < 1e-8 * (1.0 + direct.1.norm())
                    );
                } else {
                    assert!(point.is_none());
                }
            }
            assert!(qr.apply(&[f64::NAN],).is_err());
            // Global validation must also cover unused/fixed-rotation
            // coordinates after per-track full-vector scans are removed.
            let mut invalid = vec![0.0; n];
            invalid[3] = f64::NAN;
            assert!(qr.apply(&invalid).is_err());
            invalid[3] = f64::MAX;
            assert!(qr.apply(&invalid).is_err());
        }
    }
    assert_eq!(rig, original);
    let mut nonlinear_config = matrix_free_config();
    nonlinear_config.initial_lambda = Some(100.0);
    nonlinear_config.max_iterations = 3;
    let mut a = rig.clone();
    let mut b = rig.clone();
    // A landmark absent from every observation must retain its XYZ.
    a.landmarks.insert(u64::MAX, Point3::new(1.0, 2.0, 3.0));
    b.landmarks = a.landmarks.clone();
    let mut native_entry = a.clone();
    let native_result = native_entry
        .optimize_rig_qr(&nonlinear_config, MatrixFreeBaOptions::default())
        .unwrap();
    let run = |problem: &mut BundleAdjustment| {
        let mut runtime = MatrixFreeRuntime::new(MatrixFreeBaOptions::default());
        runtime.landmark_qr = true;
        problem
            .run_matrix_free_backend(&nonlinear_config, runtime)
            .unwrap()
    };
    let (result, runtime) = run(&mut a);
    let (repeat, repeated_runtime) = run(&mut b);
    assert!(result.final_cost < result.initial_cost);
    assert_eq!(result.final_cost, repeat.final_cost);
    assert_eq!(a, b);
    assert_eq!(a, native_entry);
    assert_eq!(result.final_cost, native_result.final_cost);
    assert_eq!(runtime.iterations, repeated_runtime.iterations);
    assert!(runtime.iterations.iter().any(|s| s.pcg_failure.is_none()));
    assert_eq!(a.poses[&0], rig.poses[&0]);
    assert_eq!(
        a.poses[&1].world_to_camera.rotation,
        rig.poses[&1].world_to_camera.rotation
    );
    assert_eq!(a.landmarks[&0], rig.landmarks[&0]);
    assert_eq!(a.landmarks[&u64::MAX], Point3::new(1.0, 2.0, 3.0));
    assert_eq!(a.rig_observations, rig.rig_observations);
    let mut failed = rig.clone();
    let mut runtime = MatrixFreeRuntime::new(MatrixFreeBaOptions {
        max_pcg_iterations: 1,
        pcg_relative_tolerance: 0.0,
        pcg_absolute_tolerance: 1e-30,
    });
    runtime.landmark_qr = true;
    let (rejected, runtime) = failed
        .run_matrix_free_backend(&nonlinear_config, runtime)
        .unwrap();
    assert_eq!(failed, rig);
    assert_eq!(runtime.iterations.len(), 3);
    assert!(runtime
        .iterations
        .iter()
        .all(|s| s.pcg_failure.is_some() && s.pcg_iterations == Some(1)));
    assert!(rejected
        .iterations
        .windows(2)
        .all(|s| s[0].lambda < s[1].lambda));
    let config = matrix_free_config();
    let mut mono = make_problem();
    let before = mono.clone();
    assert!(mono
        .optimize_rig_qr(&config, MatrixFreeBaOptions::default())
        .is_err());
    assert_eq!(mono, before);
    assert!(RigQrLinearization::new(&rig, &config, 0.0).is_err());
    rig.rig_observations[0].landmark_id = u64::MAX;
    assert!(RigQrLinearization::new(&rig, &config, 0.5).is_err());
}

#[test]
fn matrix_free_rejects_unsupported_input_without_mutation() {
    let mut no_anchor = make_problem();
    no_anchor.fixed_poses.clear();
    let before = no_anchor.clone();
    assert!(matches!(
        no_anchor.optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default()),
        Err(MatrixFreeBaError::Ineligible(_))
    ));
    assert_eq!(no_anchor, before);

    let mut with_prior = make_problem();
    with_prior.position_prior = Some(PositionPrior::default());
    assert!(matches!(
        with_prior.optimize_matrix_free(&matrix_free_config(), MatrixFreeBaOptions::default()),
        Err(MatrixFreeBaError::Ineligible(_))
    ));

    let mut bad_config = make_problem();
    let mut config = matrix_free_config();
    config.initial_lambda = None;
    assert!(matches!(
        bad_config.optimize_matrix_free(&config, MatrixFreeBaOptions::default()),
        Err(MatrixFreeBaError::InvalidConfiguration(_))
    ));
}

#[test]
fn landmark_only_validation_rejects_variable_pose() {
    let problem = make_problem();
    let error = problem
        .validate_matrix_free_landmark_only(&matrix_free_config(), MatrixFreeBaOptions::default())
        .unwrap_err();
    assert!(matches!(
        error,
        MatrixFreeBaError::Ineligible(
            "landmark-only matrix-free dispatch requires all poses fixed"
        )
    ));
}

#[test]
fn matrix_free_preflight_rejects_nonfinite_and_unsupported_cases_deterministically() {
    let mut cases: Vec<(&str, BundleAdjustment, BaConfig)> = Vec::new();
    for (label, initial_lambda) in [
        ("none-lambda", None),
        ("zero-lambda", Some(0.0)),
        ("nan-lambda", Some(f64::NAN)),
    ] {
        let mut config = matrix_free_config();
        config.initial_lambda = initial_lambda;
        cases.push((label, make_problem(), config));
    }
    let mut reversed = matrix_free_config();
    reversed.min_lambda = 2.0;
    reversed.max_lambda = 1.0;
    cases.push(("reversed-lambda", make_problem(), reversed));

    let mut invalid_kernel = matrix_free_config();
    invalid_kernel.robust_kernel = RobustKernel::Huber { delta: f64::NAN };
    cases.push(("invalid-kernel", make_problem(), invalid_kernel));

    let mut nonfinite_pose = make_problem();
    nonfinite_pose
        .poses
        .get_mut(&1)
        .unwrap()
        .world_to_camera
        .translation
        .x = f64::NAN;
    cases.push(("nonfinite-pose", nonfinite_pose, matrix_free_config()));

    let mut nonfinite_landmark = make_problem();
    nonfinite_landmark.landmarks.get_mut(&1).unwrap().coords.y = f64::NAN;
    cases.push((
        "nonfinite-landmark",
        nonfinite_landmark,
        matrix_free_config(),
    ));

    let mut distorted = make_problem();
    distorted.camera = Camera::pinhole_radial(7, 640, 480, 420.0, 418.0, 320.0, 240.0, 0.01, 0.0);
    cases.push(("nonzero-distortion", distorted, matrix_free_config()));

    let mut fake_anchor = make_problem();
    fake_anchor.fixed_poses.insert(99);
    fake_anchor.fixed_poses.remove(&0);
    cases.push(("unknown-anchor", fake_anchor, matrix_free_config()));

    let mut all_fixed = make_problem();
    all_fixed.fixed_poses.insert(1);
    all_fixed.fixed_poses.insert(2);
    cases.push(("all-fixed", all_fixed, matrix_free_config()));

    let mut unsupported_state = make_problem();
    unsupported_state.velocities.insert(1, Vector3::zeros());
    cases.push(("velocity-state", unsupported_state, matrix_free_config()));

    for (label, mut problem, config) in cases {
        let pose_snapshot = format!("{:?}", problem.poses);
        let landmark_snapshot = format!("{:?}", problem.landmarks);
        let result = problem.optimize_matrix_free(&config, MatrixFreeBaOptions::default());
        assert!(
            matches!(
                result,
                Err(MatrixFreeBaError::InvalidConfiguration(_))
                    | Err(MatrixFreeBaError::Ineligible(_))
            ),
            "{label}: unexpected result {result:?}"
        );
        assert_eq!(
            format!("{:?}", problem.poses),
            pose_snapshot,
            "{label} poses changed"
        );
        assert_eq!(
            format!("{:?}", problem.landmarks),
            landmark_snapshot,
            "{label} landmarks changed"
        );
    }
}

#[test]
fn matrix_free_failed_pcg_rolls_back_and_reports_attempt() {
    let mut problem = make_problem();
    let before = problem.clone();
    let mut config = matrix_free_config();
    config.max_iterations = 3;
    let options = MatrixFreeBaOptions {
        max_pcg_iterations: 1,
        pcg_relative_tolerance: 0.0,
        pcg_absolute_tolerance: 1.0e-30,
    };
    let result = problem.optimize_matrix_free(&config, options).unwrap();
    assert_eq!(problem, before);
    assert_eq!(result.matrix_free_iterations.len(), 3);
    assert!(result.matrix_free_iterations.iter().all(|iteration| {
        iteration.pcg_failure.is_some()
            && iteration.pcg_iterations == Some(1)
            && iteration.pcg_residual_norm.is_some()
    }));
    assert_eq!(result.iterations.len(), 3);
    assert!(result.iterations[0].lambda < result.iterations[1].lambda);
    assert!(result.iterations[1].lambda < result.iterations[2].lambda);
}

#[test]
fn matrix_free_column_scaled_is_deterministic_and_rolls_back_failed_steps() {
    let problem = make_problem();
    let config = matrix_free_config();
    let mut first = problem.clone();
    let mut second = problem;
    let first_result = first
        .optimize_matrix_free_column_scaled(&config, MatrixFreeBaColumnScalingOptions::default())
        .unwrap();
    let second_result = second
        .optimize_matrix_free_column_scaled(&config, MatrixFreeBaColumnScalingOptions::default())
        .unwrap();
    assert_eq!(first_result, second_result);
    assert_eq!(first, second);
    assert!(!first_result.scaling_iterations.is_empty());

    let mut failed = make_problem();
    let before = failed.clone();
    let mut failure_config = matrix_free_config();
    failure_config.max_iterations = 3;
    let failure_result = failed
        .optimize_matrix_free_column_scaled(
            &failure_config,
            MatrixFreeBaColumnScalingOptions {
                pcg: MatrixFreeBaOptions {
                    max_pcg_iterations: 1,
                    pcg_relative_tolerance: 0.0,
                    pcg_absolute_tolerance: 1.0e-30,
                },
            },
        )
        .unwrap();
    assert_eq!(failed, before);
    assert_eq!(failure_result.ba.iterations.len(), 3);
    assert!(failure_result
        .ba
        .matrix_free_iterations
        .iter()
        .all(|iteration| iteration.pcg_failure.is_some()));
}

#[test]
fn adaptive_damping_is_deterministic_and_records_candidate_gates() {
    let problem = make_problem();
    let config = matrix_free_config();
    let mut first = problem.clone();
    let mut second = problem;
    let first_result = first
        .optimize_matrix_free_column_scaled_adaptive(
            &config,
            MatrixFreeBaColumnScalingOptions::default(),
        )
        .unwrap();
    let second_result = second
        .optimize_matrix_free_column_scaled_adaptive(
            &config,
            MatrixFreeBaColumnScalingOptions::default(),
        )
        .unwrap();
    assert_eq!(first_result, second_result);
    assert_eq!(first, second);
    assert!(!first_result.adaptive_iterations.is_empty());
    assert!(first_result
        .adaptive_iterations
        .iter()
        .all(|stats| stats.next_lambda.is_finite()
            && stats.next_lambda >= config.min_lambda
            && stats.next_lambda <= config.max_lambda
            && stats.cost_gate.is_some()
            && stats.feasibility_gate.is_some()
            && stats.nonprojectable_after.is_some()));
    assert!(first_result
        .adaptive_iterations
        .iter()
        .any(|stats| stats.accepted));
}

#[test]
fn adaptive_damping_linear_failure_rolls_back_and_records_none_candidate_metrics() {
    let mut problem = make_problem();
    let before = problem.clone();
    let mut config = matrix_free_config();
    config.max_iterations = 3;
    let result = problem
        .optimize_matrix_free_column_scaled_adaptive(
            &config,
            MatrixFreeBaColumnScalingOptions {
                pcg: MatrixFreeBaOptions {
                    max_pcg_iterations: 1,
                    pcg_relative_tolerance: 0.0,
                    pcg_absolute_tolerance: 1.0e-30,
                },
            },
        )
        .unwrap();
    assert_eq!(problem, before);
    assert_eq!(result.adaptive_iterations.len(), config.max_iterations);
    assert!(result.adaptive_iterations.iter().all(|stats| stats
        .predicted_undamped_squared_decrease
        .is_none()
        && stats.actual_cost_decrease.is_none()
        && stats.rho.is_none()
        && stats.cost_gate.is_none()
        && stats.feasibility_gate.is_none()
        && stats.nonprojectable_after.is_none()
        && !stats.accepted
        && stats.reason == "linear_failure"));
    assert!(result.adaptive_iterations.iter().all(|stats| {
        let expected = (stats.solve_lambda * config.lambda_increase_factor).min(config.max_lambda);
        stats.next_lambda == expected
    }));
    assert!(result
        .adaptive_iterations
        .windows(2)
        .all(|pair| pair[1].solve_lambda == pair[0].next_lambda));
    assert!(result
        .ba
        .matrix_free_iterations
        .iter()
        .all(|stats| stats.pcg_failure.is_some()));
}

#[test]
fn adaptive_damping_rejects_nonpositive_rho_without_panic() {
    let decision =
        adaptive_step_decision(Some(Ok(f64::MAX)), f64::from_bits(1), 0.0, 0, 0, true, true);
    assert_eq!(decision.actual_cost_decrease, Some(f64::from_bits(1)));
    assert_eq!(decision.rho, Some(0.0));
    assert!(!decision.accepted);
    assert_eq!(decision.reason, "candidate_rejected_rho_nonpositive");
    assert_eq!(
        adaptive_accepted_lambda(1.0, 0.0, 1.0e-6, 1.0e6),
        Err("adaptive rho is non-finite or non-positive")
    );
}

#[test]
fn adaptive_damping_policy_clamps_and_distinguishes_prediction_failure() {
    let high_rho = adaptive_accepted_lambda(1.0, 2.0, 1.0e-6, 1.0e6).unwrap();
    assert_eq!(high_rho, 1.0 / 3.0);
    let moderate_rho = adaptive_accepted_lambda(1.0, 0.25, 1.0e-6, 1.0e6).unwrap();
    assert!((moderate_rho - 1.125).abs() < 1.0e-15);
    let bounded = adaptive_accepted_lambda(1.0e6, 2.0, 1.0e-6, 10.0).unwrap();
    assert_eq!(bounded, 10.0);
    assert_eq!(
        adaptive_accepted_lambda(1.0e-12, 1.0e-300, 1.0e-6, 1.0e6).unwrap(),
        1.0e-6
    );
    assert_eq!(
        adaptive_accepted_lambda(1.0, f64::MAX, 1.0e-6, 1.0e6).unwrap(),
        1.0 / 3.0
    );

    let cost_rejected = adaptive_step_decision(Some(Ok(1.0)), 2.0, 3.0, 0, 0, false, true);
    assert!(!cost_rejected.accepted);
    assert_eq!(cost_rejected.reason, "candidate_rejected_cost_gate");
    let feasibility_rejected = adaptive_step_decision(Some(Ok(1.0)), 2.0, 1.0, 0, 1, true, false);
    assert!(!feasibility_rejected.accepted);
    assert_eq!(
        feasibility_rejected.reason,
        "candidate_rejected_feasibility_gate"
    );
    let nonpositive_prediction = adaptive_step_decision(Some(Ok(0.0)), 2.0, 1.0, 0, 0, true, true);
    assert!(!nonpositive_prediction.accepted);
    assert_eq!(nonpositive_prediction.rho, None);
    assert_eq!(
        nonpositive_prediction.reason,
        "candidate_rejected_rho:prediction_nonpositive_or_nonfinite"
    );
    let infinite_prediction =
        adaptive_step_decision(Some(Ok(f64::INFINITY)), 2.0, 1.0, 0, 0, true, true);
    assert!(!infinite_prediction.accepted);
    assert_eq!(
        infinite_prediction.reason,
        "candidate_rejected_prediction:adaptive prediction is non-finite"
    );
    let overflowed_cost_difference =
        adaptive_step_decision(Some(Ok(1.0)), f64::MAX, -f64::MAX, 0, 0, true, true);
    assert!(!overflowed_cost_difference.accepted);
    assert_eq!(
        overflowed_cost_difference.reason,
        "candidate_rejected_rho:actual_cost_decrease_nonfinite"
    );

    let rejected = adaptive_step_decision(
        Some(Err("adaptive prediction is non-finite")),
        2.0,
        1.0,
        0,
        0,
        true,
        true,
    );
    assert!(!rejected.accepted);
    assert_eq!(rejected.prediction, None);
    assert_eq!(rejected.rho, None);
    assert_eq!(
        rejected.reason,
        "candidate_rejected_prediction:adaptive prediction is non-finite"
    );

    let invalid = adaptive_step_decision(Some(Ok(f64::NAN)), 2.0, 1.0, 0, 0, true, true);
    assert!(!invalid.accepted);
    assert_eq!(invalid.prediction, None);
    assert_eq!(invalid.rho, None);
}

#[test]
fn adaptive_damping_rejects_initial_nonprojectable_state_without_mutation() {
    let mut problem = make_problem();
    problem.landmarks.get_mut(&1).unwrap().coords.z = -1.0;
    let before = problem.clone();
    let result = problem.optimize_matrix_free_column_scaled_adaptive(
        &matrix_free_config(),
        MatrixFreeBaColumnScalingOptions::default(),
    );
    assert!(matches!(
        result,
        Err(MatrixFreeBaError::Ineligible(
            "adaptive damping requires zero initial non-projectable observations"
        ))
    ));
    assert_eq!(problem, before);
}

#[test]
fn matrix_free_restart_zero_matches_default_result_and_state() {
    let problem = make_problem();
    let config = matrix_free_config();
    let mut default_path = problem.clone();
    let mut restart_path = problem;
    let default_result = default_path
        .optimize_matrix_free(&config, MatrixFreeBaOptions::default())
        .unwrap();
    let restart_result = restart_path
        .optimize_matrix_free_with_restart(
            &config,
            MatrixFreeBaOptions::default(),
            MatrixFreeBaRestartOptions::default(),
        )
        .unwrap();
    assert_eq!(default_result, restart_result.ba);
    assert_eq!(default_path, restart_path);
    assert_eq!(
        restart_result.restart_iterations.len(),
        restart_result.ba.matrix_free_iterations.len()
    );
    assert!(restart_result
        .restart_iterations
        .iter()
        .all(|stats| stats.restarts <= 1 && stats.terminal_failure.is_none()));
}

#[test]
fn matrix_free_restart_limit_rejects_without_mutation() {
    let mut problem = make_problem();
    let before = problem.clone();
    let result = problem.optimize_matrix_free_with_restart(
        &matrix_free_config(),
        MatrixFreeBaOptions::default(),
        MatrixFreeBaRestartOptions {
            max_restarts_per_solve: 2,
        },
    );
    assert!(matches!(
        result,
        Err(MatrixFreeBaError::InvalidConfiguration(_))
    ));
    assert_eq!(problem, before);
}

#[test]
fn matrix_free_restart_failure_records_bounded_stats_and_rolls_back() {
    let mut problem = make_problem();
    let before = problem.clone();
    let mut config = matrix_free_config();
    config.max_iterations = 3;
    let result = problem
        .optimize_matrix_free_with_restart(
            &config,
            MatrixFreeBaOptions {
                max_pcg_iterations: 1,
                pcg_relative_tolerance: 0.0,
                pcg_absolute_tolerance: 1.0e-30,
            },
            MatrixFreeBaRestartOptions {
                max_restarts_per_solve: 1,
            },
        )
        .unwrap();
    assert_eq!(problem, before);
    assert_eq!(result.restart_iterations.len(), config.max_iterations);
    assert!(result.restart_iterations.iter().all(|stats| {
        stats.pcg_iterations == Some(1)
            && stats.true_residual_rechecks >= 1
            && stats.failed_true_residual_rechecks >= 1
            && stats.restarts == 0
            && stats.terminal_failure.is_some()
    }));
}
