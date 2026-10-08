use super::*;
use nalgebra::UnitQuaternion;

fn audit_camera() -> Camera {
    Camera::pinhole(1, 1600, 1066, 879.4, 879.4, 803.4, 532.6)
}

fn audit_pose() -> Pose {
    Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.08, -0.11, 0.17),
        Vector3::new(0.24, -0.13, 0.31),
    )
}

fn assert_case_is_accurate(label: &str, case: BaVisualJacobianCase) {
    eprintln!(
        "ba-jacobian-test: case={label} residual={:.3e} depth={:.3e} pose=(abs {:.3e},rel {:.3e}; trans {:.3e}/{:.3e}; rot {:.3e}/{:.3e}) landmark=(abs {:.3e},rel {:.3e}) intrinsics=(abs {:.3e},rel {:.3e})",
        case.residual_norm,
        case.depth,
        case.pose_max_abs,
        case.pose_relative,
        case.pose_translation_max_abs,
        case.pose_translation_relative,
        case.pose_rotation_max_abs,
        case.pose_rotation_relative,
        case.landmark_max_abs,
        case.landmark_relative,
        case.intrinsics_max_abs,
        case.intrinsics_relative,
    );
    assert!(
        case.pose_max_abs < 1.0e-5 && case.pose_relative < 1.0e-6,
        "{label} pose Jacobian mismatch: {:?}",
        case
    );
    assert!(
        case.landmark_max_abs < 1.0e-5 && case.landmark_relative < 1.0e-6,
        "{label} landmark Jacobian mismatch: {:?}",
        case
    );
    assert!(
        case.intrinsics_max_abs < 1.0e-6 && case.intrinsics_relative < 1.0e-8,
        "{label} intrinsics Jacobian mismatch: {:?}",
        case
    );
}

#[test]
fn analytic_visual_jacobians_match_finite_differences_across_regimes() {
    let camera = audit_camera();
    let pose = audit_pose();
    let normal_point = Point3::new(0.45, -0.35, 4.8);
    let normal_measurement = camera
        .project(&pose.transform_world_point(&normal_point))
        .unwrap();
    assert_case_is_accurate(
        "normal",
        audit_visual_jacobian_case(&camera, &pose, &normal_point, &normal_measurement, 1.0e-6)
            .unwrap(),
    );

    let far_point = Point3::new(15.0, -8.0, 10_000.0);
    let far_measurement = camera
        .project(&pose.transform_world_point(&far_point))
        .unwrap();
    assert_case_is_accurate(
        "far-depth",
        audit_visual_jacobian_case(&camera, &pose, &far_point, &far_measurement, 1.0e-6).unwrap(),
    );

    // A very small camera baseline relative to depth is the low-parallax
    // regime that made the captured 27-camera point block ill-conditioned.
    // The per-observation Jacobian itself remains well-defined, so this
    // case checks that no special-case branch changes its numerical value.
    let low_parallax_point = Point3::new(-0.15, 0.12, 100.0);
    let low_parallax_measurement = camera
        .project(&pose.transform_world_point(&low_parallax_point))
        .unwrap();
    assert_case_is_accurate(
        "low-parallax",
        audit_visual_jacobian_case(
            &camera,
            &pose,
            &low_parallax_point,
            &low_parallax_measurement,
            1.0e-6,
        )
        .unwrap(),
    );

    let high_residual_measurement = normal_measurement + Vector2::new(80.0, -55.0);
    assert_case_is_accurate(
        "high-residual",
        audit_visual_jacobian_case(
            &camera,
            &pose,
            &normal_point,
            &high_residual_measurement,
            1.0e-6,
        )
        .unwrap(),
    );
}

#[test]
fn bundle_audit_reports_low_parallax_and_high_residual_buckets() {
    let camera = audit_camera();
    let pose0 = Pose::identity();
    let pose1 =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.01, 0.0, 0.0));
    let point = Point3::new(0.2, -0.1, 100.0);
    let mut ba = BundleAdjustment::new(camera.clone());
    ba.add_pose(0, pose0.clone());
    ba.add_pose(1, pose1);
    ba.add_landmark(0, point);
    let exact0 = camera
        .project(&pose0.transform_world_point(&point))
        .unwrap();
    let exact1 = camera
        .project(&ba.poses[&1].transform_world_point(&point))
        .unwrap();
    ba.add_observation(BaObservation {
        keyframe_id: 0,
        landmark_id: 0,
        xy: exact0,
    });
    ba.add_observation(BaObservation {
        keyframe_id: 1,
        landmark_id: 0,
        xy: exact1 + Vector2::new(25.0, 0.0),
    });
    let report = audit_bundle_visual_jacobians(&ba, 16);
    assert_eq!(report.observations_seen, 2);
    assert_eq!(report.samples_audited, 2);
    assert_eq!(report.low_parallax.samples, 1);
    assert_eq!(report.high_residual.samples, 1);
    assert_eq!(report.invalid_samples, 0);
}

#[test]
fn huber_weight_is_the_derivative_of_the_squared_residual_cost() {
    let kernel = RobustKernel::Huber { delta: 3.0 };
    for squared_residual in [1.0, 4.0, 16.0, 100.0] {
        let epsilon = 1.0e-6 * squared_residual;
        let numerical = (kernel.cost(squared_residual + epsilon)
            - kernel.cost(squared_residual - epsilon))
            / (2.0 * epsilon);
        let analytic = kernel.weight(squared_residual);
        assert!(
            (analytic - numerical).abs() < 1.0e-8,
            "s={squared_residual}: rho'={analytic}, finite difference={numerical}"
        );
    }
}

#[test]
fn schur_rhs_and_back_substitution_match_the_full_normal_system() {
    let mut h_pp = DMatrix::<f64>::zeros(6, 6);
    for i in 0..6 {
        h_pp[(i, i)] = 10.0 + i as f64;
    }
    let mut h_ll = Matrix3::<f64>::zeros();
    h_ll[(0, 0)] = 4.0;
    h_ll[(1, 1)] = 5.0;
    h_ll[(2, 2)] = 6.0;
    let cross = Matrix6x3::<f64>::from_fn(|r, c| 0.03 * (r as f64 + 1.0) * (c as f64 + 2.0));
    let b_p = DVector::from_iterator(6, (0..6).map(|i| 0.2 * (i as f64 + 1.0)));
    let b_l = Vector3::new(-0.4, 0.3, 0.2);
    let mut system = NormalEquationsBa {
        h_pp: CameraHessian::Dense(h_pp.clone()),
        b_p: b_p.clone(),
        landmarks: vec![LandmarkBlock {
            h_ll,
            b_l,
            cross: vec![(0, cross)],
        }],
    };

    let (delta_p, delta_l) = solve_step(
        &mut system,
        1,
        1,
        0,
        0,
        0.0,
        LinearSolver::Dense,
        false,
        &mut None,
    )
    .unwrap();

    let mut sparse_system = NormalEquationsBa {
        h_pp: CameraHessian::Dense(h_pp.clone()),
        b_p: b_p.clone(),
        landmarks: vec![LandmarkBlock {
            h_ll,
            b_l,
            cross: vec![(0, cross)],
        }],
    };
    let (sparse_delta_p, sparse_delta_l) = solve_step(
        &mut sparse_system,
        1,
        1,
        0,
        0,
        0.0,
        LinearSolver::Sparse,
        false,
        &mut None,
    )
    .unwrap();

    let mut block_system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![h_pp.fixed_view::<6, 6>(0, 0).into_owned()]),
        b_p: b_p.clone(),
        landmarks: vec![LandmarkBlock {
            h_ll,
            b_l,
            cross: vec![(0, cross)],
        }],
    };
    let (block_delta_p, block_delta_l) = solve_step(
        &mut block_system,
        1,
        1,
        0,
        0,
        0.0,
        LinearSolver::Sparse,
        false,
        &mut None,
    )
    .unwrap();

    let mut full_h = DMatrix::<f64>::zeros(9, 9);
    full_h.view_mut((0, 0), (6, 6)).copy_from(&h_pp);
    for r in 0..6 {
        for c in 0..3 {
            full_h[(r, 6 + c)] = cross[(r, c)];
            full_h[(6 + c, r)] = cross[(r, c)];
        }
    }
    full_h.view_mut((6, 6), (3, 3)).copy_from(&h_ll);
    let mut full_rhs = DVector::<f64>::zeros(9);
    for i in 0..6 {
        full_rhs[i] = -b_p[i];
    }
    for i in 0..3 {
        full_rhs[6 + i] = -b_l[i];
    }
    let full_delta = solve_normal_equations(&full_h, &full_rhs).unwrap();
    assert!((delta_p - full_delta.rows(0, 6)).norm() < 1.0e-10);
    assert!((delta_l - full_delta.rows(6, 3)).norm() < 1.0e-10);
    assert_eq!(block_delta_p.as_slice(), sparse_delta_p.as_slice());
    assert_eq!(block_delta_l.as_slice(), sparse_delta_l.as_slice());
    assert!((&sparse_delta_p - full_delta.rows(0, 6)).norm() < 1.0e-10);
    assert!((&sparse_delta_l - full_delta.rows(6, 3)).norm() < 1.0e-10);
}

#[test]
fn pose_block_sparse_path_matches_dense_sparse_path_bitwise() {
    let diagonal = vec![
        Matrix6::from_diagonal(&Vector6::new(20.0, 21.0, 22.0, 23.0, 24.0, 25.0)),
        Matrix6::from_diagonal(&Vector6::new(26.0, 27.0, 28.0, 29.0, 30.0, 31.0)),
    ];
    let mut dense = DMatrix::zeros(12, 12);
    for (pose, block) in diagonal.iter().enumerate() {
        dense
            .fixed_view_mut::<6, 6>(pose * 6, pose * 6)
            .copy_from(block);
    }
    let cross0 = Matrix6x3::from_fn(|r, c| 0.01 * (r + c + 1) as f64);
    let cross1 = Matrix6x3::from_fn(|r, c| 0.015 * (2 * r + c + 1) as f64);
    let landmark = LandmarkBlock {
        h_ll: Matrix3::from_diagonal(&Vector3::new(8.0, 9.0, 10.0)),
        b_l: Vector3::new(0.2, -0.1, 0.3),
        cross: vec![(0, cross0), (1, cross1)],
    };
    let gradient = DVector::from_iterator(12, (0..12).map(|i| 0.03 * (i + 1) as f64));
    let mut dense_system = NormalEquationsBa {
        h_pp: CameraHessian::Dense(dense),
        b_p: gradient.clone(),
        landmarks: vec![landmark.clone()],
    };
    let mut block_system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(diagonal),
        b_p: gradient,
        landmarks: vec![landmark],
    };

    let dense_delta = solve_step(
        &mut dense_system,
        2,
        1,
        0,
        0,
        0.25,
        LinearSolver::Sparse,
        false,
        &mut None,
    )
    .unwrap();
    let mut symbolic_cache = None;
    let block_delta = solve_step(
        &mut block_system,
        2,
        1,
        0,
        0,
        0.25,
        LinearSolver::Sparse,
        false,
        &mut symbolic_cache,
    )
    .unwrap();

    assert_eq!(block_delta.0.as_slice(), dense_delta.0.as_slice());
    assert_eq!(block_delta.1.as_slice(), dense_delta.1.as_slice());
    assert!(symbolic_cache.is_some());
}
