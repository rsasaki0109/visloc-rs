use super::*;

fn synthetic_system() -> NormalEquationsBa {
    let h_pp = Matrix6::from_diagonal(&Vector6::from_row_slice(&[
        4.0, 9.0, 16.0, 25.0, 36.0, 49.0,
    ]));
    let h_ll = Matrix3::from_diagonal(&Vector3::new(4.0, 9.0, 16.0));
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
            // Deliberate same-pose duplicate: the quality denominator
            // must use |cross_a + cross_b|, not |cross_a|+|cross_b|.
            cross: vec![(0, cross_a), (0, cross_b)],
        }],
    }
}

fn full_normal(system: &NormalEquationsBa) -> (DMatrix<f64>, DVector<f64>) {
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        panic!("quality test requires pose blocks");
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

fn assert_close(actual: f64, expected: f64) {
    let scale = actual.abs().max(expected.abs()).max(1.0);
    assert!((actual - expected).abs() <= 1.0e-11 * scale);
}

fn dense_quality_metrics(
    undamped: &DMatrix<f64>,
    damped: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
) -> (f64, f64, f64) {
    let residual = damped * delta + rhs;
    let prediction = -2.0 * rhs.dot(delta) - delta.dot(&(undamped * delta));
    let denominator = damped.norm() * delta.norm() + rhs.norm();
    let mut componentwise: f64 = 0.0;
    for row in 0..damped.nrows() {
        let row_denominator = (0..damped.ncols())
            .map(|column| damped[(row, column)].abs() * delta[column].abs())
            .sum::<f64>()
            + rhs[row].abs();
        let ratio = if row_denominator == 0.0 {
            assert_eq!(residual[row], 0.0);
            0.0
        } else {
            residual[row].abs() / row_denominator
        };
        componentwise = componentwise.max(ratio);
    }
    (prediction, residual.norm() / denominator, componentwise)
}

#[test]
fn quality_matches_explicit_full_normal_prediction_and_eta() {
    let system = synthetic_system();
    let before = full_normal(&system);
    let delta_pose = DVector::from_row_slice(&[0.2, -0.3, 0.4, -0.5, 0.6, -0.7]);
    let delta_landmark = DVector::from_row_slice(&[0.8, -0.9, 1.0]);
    let lambda = 0.5;
    let quality = quality_coordinate_metrics(
        &system,
        lambda,
        &delta_pose,
        &delta_landmark,
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap();

    let (h, b) = full_normal(&system);
    let mut damped = h.clone();
    for index in 0..damped.nrows() {
        damped[(index, index)] += lambda;
    }
    let mut delta = DVector::zeros(9);
    delta.rows_mut(0, 6).copy_from(&delta_pose);
    delta.rows_mut(6, 3).copy_from(&delta_landmark);
    let (expected_prediction, expected_normwise, expected_componentwise) =
        dense_quality_metrics(&h, &damped, &b, &delta);

    assert_close(
        quality.predicted_undamped_squared_decrease,
        expected_prediction,
    );
    assert_close(quality.normwise_backward_error, expected_normwise);
    assert_close(quality.componentwise_backward_error, expected_componentwise);
    assert_eq!(before, full_normal(&system));
}

#[test]
fn quality_streams_multiple_landmarks_against_full_normal_oracle() {
    let mut system = synthetic_system();
    system.landmarks.push(LandmarkBlock {
        h_ll: Matrix3::from_diagonal(&Vector3::new(7.0, 8.0, 9.0)),
        b_l: Vector3::new(-0.2, 0.4, -0.6),
        cross: vec![(
            0,
            Matrix6x3::from_fn(|row, column| if row == column + 1 { 0.15 } else { 0.0 }),
        )],
    });
    let delta_pose = DVector::from_row_slice(&[0.2, -0.3, 0.4, -0.5, 0.6, -0.7]);
    let delta_landmarks = DVector::from_row_slice(&[0.8, -0.9, 1.0, -1.1, 1.2, -1.3]);
    let lambda = 0.5;
    let quality = quality_coordinate_metrics(
        &system,
        lambda,
        &delta_pose,
        &delta_landmarks,
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap();
    let (h, b) = full_normal(&system);
    let mut delta = DVector::zeros(12);
    delta.rows_mut(0, 6).copy_from(&delta_pose);
    delta.rows_mut(6, 6).copy_from(&delta_landmarks);
    let mut damped = h.clone();
    for index in 0..damped.nrows() {
        damped[(index, index)] += lambda;
    }
    let (expected_prediction, expected_normwise, expected_componentwise) =
        dense_quality_metrics(&h, &damped, &b, &delta);
    assert_close(
        quality.predicted_undamped_squared_decrease,
        expected_prediction,
    );
    assert_close(quality.normwise_backward_error, expected_normwise);
    assert_close(quality.componentwise_backward_error, expected_componentwise);
}

#[test]
fn adaptive_prediction_matches_full_normal_with_rig_crosses_and_fixed_rotation() {
    let mut system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![
            Matrix6::from_diagonal(&Vector6::from_row_slice(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0])),
            Matrix6::from_diagonal(&Vector6::from_row_slice(&[
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ])),
        ]),
        b_p: DVector::from_row_slice(&[
            1.0, -2.0, 3.0, -4.0, 5.0, -6.0, -1.5, 2.5, -3.5, 0.0, 0.0, 0.0,
        ]),
        landmarks: vec![
            LandmarkBlock {
                h_ll: Matrix3::from_diagonal(&Vector3::new(3.0, 4.0, 5.0)),
                b_l: Vector3::new(0.25, -0.5, 0.75),
                cross: vec![
                    (
                        0,
                        Matrix6x3::from_fn(|row, column| {
                            if row == column {
                                0.2
                            } else if row == column + 3 {
                                -0.1
                            } else {
                                0.0
                            }
                        }),
                    ),
                    (
                        0,
                        Matrix6x3::from_fn(|row, column| if row == column { -0.05 } else { 0.0 }),
                    ),
                    (
                        1,
                        Matrix6x3::from_fn(
                            |row, column| {
                                if row == column + 3 {
                                    0.08
                                } else {
                                    0.0
                                }
                            },
                        ),
                    ),
                ],
            },
            LandmarkBlock {
                h_ll: Matrix3::from_diagonal(&Vector3::new(6.0, 7.0, 8.0)),
                b_l: Vector3::new(-0.4, 0.6, -0.8),
                cross: vec![
                    (
                        1,
                        Matrix6x3::from_fn(|row, column| if row == column { -0.12 } else { 0.0 }),
                    ),
                    (
                        0,
                        Matrix6x3::from_fn(
                            |row, column| {
                                if row == column + 3 {
                                    0.07
                                } else {
                                    0.0
                                }
                            },
                        ),
                    ),
                ],
            },
        ],
    };
    constrain_fixed_pose_rotations(
        &BTreeSet::from([20_u64]),
        &BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]),
        &mut system,
    );
    let before = full_normal(&system);
    let (full_h, full_b) = full_normal(&system);
    let lambda = 0.5;
    let damped = &full_h + lambda * DMatrix::<f64>::identity(18, 18);
    let solved = damped
        .lu()
        .solve(&(-full_b.clone()))
        .expect("multi-pose prediction fixture should solve");
    let perturbation = DVector::from_row_slice(&[
        0.003, -0.002, 0.001, -0.0015, 0.0025, -0.001, 0.0012, -0.0018, 0.0009, 0.0, 0.0, 0.0,
        0.0017, -0.0011, 0.0008, -0.0014, 0.0006, -0.0009,
    ]);
    let delta = solved + perturbation;
    let delta_poses = delta.rows(0, 12).into_owned();
    let delta_landmarks = delta.rows(12, 6).into_owned();
    let expected_prediction = -2.0 * full_b.dot(&delta) - delta.dot(&(&full_h * &delta));
    let prediction = matrix_free_undamped_prediction(&system, &delta_poses, &delta_landmarks)
        .expect("adaptive prediction fixture should be finite");
    let quality = quality_coordinate_metrics(
        &system,
        lambda,
        &delta_poses,
        &delta_landmarks,
        MatrixFreeQualityCoordinate::Current,
    )
    .expect("quality fixture should be finite");
    assert_close(prediction, expected_prediction);
    assert_close(
        quality.predicted_undamped_squared_decrease,
        expected_prediction,
    );
    assert_eq!(before, full_normal(&system));
}

#[test]
fn componentwise_eta_is_invariant_under_positive_column_scaling() {
    let original = synthetic_system();
    let mut scaled = synthetic_system();
    let state = column_equilibrate_normal_system(&mut scaled).unwrap();
    let lambda = 0.5;
    let (scaled_h, scaled_b) = full_normal(&scaled);
    let mut scaled_damped = scaled_h.clone();
    for index in 0..scaled_damped.nrows() {
        scaled_damped[(index, index)] += lambda;
    }
    let scaled_solution = scaled_damped
        .clone()
        .lu()
        .solve(&(-scaled_b.clone()))
        .expect("scaled synthetic system should solve");
    // Use a solved step plus a small, non-collinear perturbation.  This
    // keeps the residual nonzero without making eta a saturated 1.0
    // artifact of an arbitrary hand-written delta.
    let scaled_delta = scaled_solution
        + DVector::from_row_slice(&[
            3.0e-3, -2.0e-3, 1.5e-3, -1.0e-3, 2.5e-3, -1.25e-3, 1.75e-3, -2.25e-3, 0.875e-3,
        ]);
    let scaled_pose = scaled_delta.rows(0, 6).into_owned();
    let scaled_landmark = scaled_delta.rows(6, 3).into_owned();

    let mut physical_transform = DVector::zeros(9);
    for component in 0..6 {
        physical_transform[component] = state.pose_transforms[0][component];
    }
    for component in 0..3 {
        physical_transform[6 + component] = state.landmark_transforms[0][component];
    }
    let mut inverse_transform = DVector::zeros(9);
    let mut physical_delta = DVector::zeros(9);
    for index in 0..9 {
        inverse_transform[index] = physical_transform[index].recip();
        physical_delta[index] = physical_transform[index] * scaled_delta[index];
    }
    let mut physical_h = DMatrix::zeros(9, 9);
    let mut physical_damped = DMatrix::zeros(9, 9);
    let mut physical_b = DVector::zeros(9);
    for row in 0..9 {
        physical_b[row] = inverse_transform[row] * scaled_b[row];
        for column in 0..9 {
            physical_h[(row, column)] =
                inverse_transform[row] * scaled_h[(row, column)] * inverse_transform[column];
            physical_damped[(row, column)] =
                inverse_transform[row] * scaled_damped[(row, column)] * inverse_transform[column];
        }
    }

    let (scaled_expected_prediction, scaled_expected_normwise, scaled_expected_eta) =
        dense_quality_metrics(&scaled_h, &scaled_damped, &scaled_b, &scaled_delta);
    let (physical_expected_prediction, physical_expected_normwise, physical_expected_eta) =
        dense_quality_metrics(&physical_h, &physical_damped, &physical_b, &physical_delta);
    let scaled_quality = quality_coordinate_metrics(
        &scaled,
        lambda,
        &scaled_pose,
        &scaled_landmark,
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap();
    let physical_equivalent = quality_coordinate_metrics(
        &scaled,
        lambda,
        &scaled_pose,
        &scaled_landmark,
        MatrixFreeQualityCoordinate::PhysicalEquivalent(&state),
    )
    .unwrap();

    assert_close(
        scaled_quality.predicted_undamped_squared_decrease,
        scaled_expected_prediction,
    );
    assert_close(
        scaled_quality.normwise_backward_error,
        scaled_expected_normwise,
    );
    assert_close(
        scaled_quality.componentwise_backward_error,
        scaled_expected_eta,
    );
    assert_close(
        physical_equivalent.predicted_undamped_squared_decrease,
        physical_expected_prediction,
    );
    assert_close(
        physical_equivalent.normwise_backward_error,
        physical_expected_normwise,
    );
    assert_close(
        physical_equivalent.componentwise_backward_error,
        physical_expected_eta,
    );
    assert_close(scaled_expected_eta, physical_expected_eta);
    assert!(scaled_expected_eta > 0.0 && scaled_expected_eta < 1.0);
    assert!(physical_expected_eta > 0.0 && physical_expected_eta < 1.0);

    // Undamped prediction is invariant under the positive diagonal
    // coordinate transform, even though lambda I is represented as
    // lambda*diag(d) after returning to physical coordinates.
    let (original_h, original_b) = full_normal(&original);
    let original_quality = quality_coordinate_metrics(
        &original,
        lambda,
        &physical_delta.rows(0, 6).into_owned(),
        &physical_delta.rows(6, 3).into_owned(),
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap();
    let original_expected_prediction = -2.0 * original_b.dot(&physical_delta)
        - physical_delta.dot(&(&original_h * &physical_delta));
    assert_close(
        original_quality.predicted_undamped_squared_decrease,
        original_expected_prediction,
    );
    assert_close(
        original_quality.predicted_undamped_squared_decrease,
        physical_expected_prediction,
    );
}

#[test]
fn quality_allows_fixed_zero_rows_and_reports_invalid_inputs() {
    let mut fixed = synthetic_system();
    constrain_fixed_pose_rotations(
        &BTreeSet::from([7_u64]),
        &BTreeMap::from([(7_u64, 0_usize)]),
        &mut fixed,
    );
    let zero_rotation = DVector::from_row_slice(&[0.2, -0.3, 0.4, 0.0, 0.0, 0.0]);
    let landmark = DVector::from_row_slice(&[0.8, -0.9, 1.0]);
    let fixed_quality = quality_coordinate_metrics(
        &fixed,
        0.5,
        &zero_rotation,
        &landmark,
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap();
    assert!(fixed_quality.componentwise_backward_error.is_finite());

    let mut nonfinite = synthetic_system();
    nonfinite.b_p[0] = f64::NAN;
    assert!(quality_coordinate_metrics(
        &nonfinite,
        0.5,
        &zero_rotation,
        &landmark,
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap_err()
    .contains("non-finite"));

    let zero_system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![Matrix6::zeros()]),
        b_p: DVector::zeros(6),
        landmarks: Vec::new(),
    };
    let zero_delta = DVector::zeros(6);
    assert!(quality_coordinate_metrics(
        &zero_system,
        0.0,
        &zero_delta,
        &DVector::zeros(0),
        MatrixFreeQualityCoordinate::Current,
    )
    .unwrap_err()
    .contains("denominator is zero"));
}

#[test]
fn rho_is_undefined_when_nonprojectable_set_changes() {
    let (actual, rho, reason) = matrix_free_quality_actual_and_rho(100.0, 90.0, Some(20.0), 0, 1);
    assert_eq!(actual, Some(10.0));
    assert_eq!(rho, None);
    assert_eq!(reason, "nonprojectable_count_nonzero");

    let (actual, rho, reason) = matrix_free_quality_actual_and_rho(100.0, 90.0, Some(20.0), 0, 0);
    assert_eq!(actual, Some(10.0));
    assert_eq!(rho, Some(0.5));
    assert_eq!(reason, "defined");

    let (actual, rho, reason) =
        matrix_free_quality_actual_and_rho(f64::MAX, -f64::MAX, Some(1.0), 0, 0);
    assert_eq!(actual, None);
    assert_eq!(rho, None);
    assert_eq!(reason, "actual_cost_decrease_nonfinite");

    let (actual, rho, reason) =
        matrix_free_quality_actual_and_rho(1.0, 0.0, Some(f64::from_bits(1)), 0, 0);
    assert_eq!(actual, Some(1.0));
    assert_eq!(rho, None);
    assert_eq!(reason, "rho_nonfinite");
}
