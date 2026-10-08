use super::*;

fn synthetic_system() -> NormalEquationsBa {
    let h_pp = Matrix6::from_diagonal(&Vector6::from_row_slice(&[
        4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
    ]));
    let h_ll = Matrix3::from_diagonal(&Vector3::from_row_slice(&[4.0, 9.0, 16.0]));
    let cross_a = Matrix6x3::from_fn(|row, column| {
        if row == column {
            0.25
        } else if row == column + 3 {
            -0.125
        } else {
            0.0
        }
    });
    let cross_b = Matrix6x3::from_fn(|row, column| if row == column { -0.0625 } else { 0.0 });
    NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![h_pp]),
        b_p: DVector::from_row_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]),
        landmarks: vec![LandmarkBlock {
            h_ll,
            b_l: Vector3::new(0.5, -0.75, 1.25),
            // Two entries for the same pose model two sensors observing
            // one landmark.  The transform must touch both entries.
            cross: vec![(0, cross_a), (0, cross_b)],
        }],
    }
}

fn full_normal(system: &NormalEquationsBa) -> (DMatrix<f64>, DVector<f64>) {
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        panic!("synthetic system must use pose blocks");
    };
    let pose_count = pose_blocks.len();
    let landmark_count = system.landmarks.len();
    let dimension = pose_count * 6 + landmark_count * 3;
    let mut h = DMatrix::zeros(dimension, dimension);
    let mut b = DVector::zeros(dimension);
    for (pose, block) in pose_blocks.iter().enumerate() {
        for row in 0..6 {
            b[pose * 6 + row] = system.b_p[pose * 6 + row];
            for column in 0..6 {
                h[(pose * 6 + row, pose * 6 + column)] = block[(row, column)];
            }
        }
    }
    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        let offset = pose_count * 6 + landmark_index * 3;
        for row in 0..3 {
            b[offset + row] = landmark.b_l[row];
            for column in 0..3 {
                h[(offset + row, offset + column)] = landmark.h_ll[(row, column)];
            }
        }
        for (pose, cross) in &landmark.cross {
            for row in 0..6 {
                for column in 0..3 {
                    h[(pose * 6 + row, offset + column)] += cross[(row, column)];
                    h[(offset + column, pose * 6 + row)] += cross[(row, column)];
                }
            }
        }
    }
    (h, b)
}

fn diagonal_damping(system: &NormalEquationsBa) -> DVector<f64> {
    let (h, _) = full_normal(system);
    DVector::from_iterator(
        h.nrows(),
        h.diagonal()
            .iter()
            .map(|value| value.clamp(COLUMN_SCALING_MIN_DIAGONAL, COLUMN_SCALING_MAX_DIAGONAL)),
    )
}

#[test]
fn scaled_system_matches_h_plus_lambda_diagonal_damping() {
    let mut original = synthetic_system();
    let (full_h, full_b) = full_normal(&original);
    let damping_diagonal = diagonal_damping(&original);
    let lambda = 0.25;

    let state = column_equilibrate_normal_system(&mut original).unwrap();
    assert_eq!(state.stats.minimum_diagonal, 4.0);
    assert_eq!(state.stats.maximum_diagonal, 49.0);
    assert_eq!(state.stats.clamped_to_minimum, 0);
    assert_eq!(state.stats.clamped_to_maximum, 0);

    let (scaled_h, scaled_b) = full_normal(&original);
    let scaled_solution: DVector<f64> = (scaled_h + lambda * DMatrix::<f64>::identity(9, 9))
        .lu()
        .solve(&(-scaled_b))
        .expect("scaled synthetic system should solve");
    let mut scaled_pose = DVector::from_iterator(6, scaled_solution.rows(0, 6).iter().copied());
    let mut scaled_landmarks =
        DVector::from_iterator(3, scaled_solution.rows(6, 3).iter().copied());
    state
        .unscale_deltas(&mut scaled_pose, &mut scaled_landmarks)
        .unwrap();
    let mut physical_solution = DVector::zeros(9);
    physical_solution.rows_mut(0, 6).copy_from(&scaled_pose);
    physical_solution
        .rows_mut(6, 3)
        .copy_from(&scaled_landmarks);

    let mut damped_h = full_h;
    for index in 0..9 {
        damped_h[(index, index)] += lambda * damping_diagonal[index];
    }
    let expected = damped_h
        .lu()
        .solve(&(-full_b))
        .expect("physical synthetic system should solve");
    assert!((physical_solution - expected).norm() < 1.0e-12);
}

#[test]
fn matrix_free_scaled_step_matches_the_diagonally_damped_full_system() {
    let mut scaled = synthetic_system();
    let (full_h, full_b) = full_normal(&scaled);
    let damping_diagonal = diagonal_damping(&scaled);
    let state = column_equilibrate_normal_system(&mut scaled).unwrap();
    let lambda = 0.25;

    // Exercise the same Schur/PCG/back-substitution path used by the
    // production opt-in entry point.  The assertion below is against the
    // independently assembled full system, rather than against another
    // call to the scaled operator.
    let outcome = match solve_matrix_free_step(
        &scaled,
        lambda,
        MatrixFreeBaOptions {
            max_pcg_iterations: 128,
            pcg_relative_tolerance: 1.0e-10,
            pcg_absolute_tolerance: 1.0e-12,
        },
        0,
        false,
        Some(&state),
        None,
        false,
        false,
    ) {
        Ok(outcome) => outcome,
        Err(_) => panic!("scaled synthetic matrix-free solve should succeed"),
    };
    assert!(outcome.diagnostics.pcg_failure.is_none());

    let mut damped_h = full_h;
    for index in 0..damped_h.nrows() {
        damped_h[(index, index)] += lambda * damping_diagonal[index];
    }
    let expected = damped_h
        .lu()
        .solve(&(-full_b))
        .expect("physical synthetic system should solve");
    let mut actual = DVector::zeros(9);
    actual.rows_mut(0, 6).copy_from(&outcome.delta_poses);
    actual.rows_mut(6, 3).copy_from(&outcome.delta_landmarks);
    assert!((actual - expected).norm() < 1.0e-8);
}

#[test]
fn scaling_preserves_fixed_rotation_identity_and_rejects_bad_diagonals() {
    let mut fixed = synthetic_system();
    let pose_index = BTreeMap::from([(7_u64, 0_usize)]);
    let fixed_rotations = BTreeSet::from([7_u64]);
    constrain_fixed_pose_rotations(&fixed_rotations, &pose_index, &mut fixed);
    let state = column_equilibrate_normal_system(&mut fixed).unwrap();
    assert_eq!(state.pose_transforms[0][3], 1.0);
    assert_eq!(state.pose_transforms[0][4], 1.0);
    assert_eq!(state.pose_transforms[0][5], 1.0);
    let CameraHessian::PoseDiagonal(blocks) = fixed.h_pp else {
        panic!("fixed synthetic system must use pose blocks");
    };
    for component in 3..6 {
        assert_eq!(blocks[0][(component, component)], 1.0);
    }

    let mut clamped = synthetic_system();
    if let CameraHessian::PoseDiagonal(blocks) = &mut clamped.h_pp {
        blocks[0][(0, 0)] = 0.0;
        blocks[0][(1, 1)] = 1.0e40;
    }
    let clamped_state = column_equilibrate_normal_system(&mut clamped).unwrap();
    assert_eq!(clamped_state.stats.minimum_diagonal, 1.0e-6);
    assert_eq!(clamped_state.stats.maximum_diagonal, 1.0e32);
    assert!(clamped_state.stats.clamped_to_minimum >= 1);
    assert!(clamped_state.stats.clamped_to_maximum >= 1);

    let mut negative = synthetic_system();
    if let CameraHessian::PoseDiagonal(blocks) = &mut negative.h_pp {
        blocks[0][(0, 0)] = -1.0;
    }
    assert!(column_equilibrate_normal_system(&mut negative)
        .unwrap_err()
        .contains("negative"));

    let mut nonfinite = synthetic_system();
    nonfinite.landmarks[0].h_ll[(1, 1)] = f64::NAN;
    assert!(column_equilibrate_normal_system(&mut nonfinite)
        .unwrap_err()
        .contains("non-finite"));

    let mut nonfinite_offdiag = synthetic_system();
    if let CameraHessian::PoseDiagonal(blocks) = &mut nonfinite_offdiag.h_pp {
        blocks[0][(0, 1)] = f64::NAN;
    }
    assert!(column_equilibrate_normal_system(&mut nonfinite_offdiag)
        .unwrap_err()
        .contains("pose Hessian"));

    let mut nonfinite_pose_gradient = synthetic_system();
    nonfinite_pose_gradient.b_p[0] = f64::NAN;
    assert!(
        column_equilibrate_normal_system(&mut nonfinite_pose_gradient)
            .unwrap_err()
            .contains("pose gradient")
    );

    let mut nonfinite_cross = synthetic_system();
    nonfinite_cross.landmarks[0].cross[0].1[(0, 0)] = f64::NAN;
    assert!(column_equilibrate_normal_system(&mut nonfinite_cross)
        .unwrap_err()
        .contains("landmark cross"));

    let state = column_equilibrate_normal_system(&mut synthetic_system()).unwrap();
    let mut wrong_pose = DVector::zeros(1);
    let mut valid_landmarks = DVector::zeros(3);
    assert!(state
        .unscale_deltas(&mut wrong_pose, &mut valid_landmarks)
        .unwrap_err()
        .contains("dimensions"));
    let mut nonfinite_pose = DVector::from_element(6, f64::NAN);
    let mut valid_landmarks = DVector::zeros(3);
    assert!(state
        .unscale_deltas(&mut nonfinite_pose, &mut valid_landmarks)
        .unwrap_err()
        .contains("non-finite"));
}
