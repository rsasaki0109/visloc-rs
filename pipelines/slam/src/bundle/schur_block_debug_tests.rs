use super::*;

#[test]
fn sparse_debug_claim_is_once_and_ineligible_calls_do_not_consume_it() {
    use std::sync::atomic::AtomicBool;
    let claimed = AtomicBool::new(false);
    assert_eq!(claim_sparse_debug_window(false, Some(0), 2, &claimed), None);
    assert_eq!(claim_sparse_debug_window(true, None, 2, &claimed), None);
    assert_eq!(claim_sparse_debug_window(true, Some(2), 2, &claimed), None);
    assert_eq!(
        claim_sparse_debug_window(true, Some(1), 2, &claimed),
        Some(1)
    );
    assert_eq!(claim_sparse_debug_window(true, Some(0), 2, &claimed), None);
    let concurrent = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let handles = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    claim_sparse_debug_window(true, Some(0), 1, &concurrent).is_some() as usize
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .sum::<usize>(),
            1
        );
    });
}

#[test]
fn local_block_metrics_distinguish_spd_non_spd_nonfinite_and_asymmetry() {
    let spd = Matrix6::from_diagonal(&Vector6::from_element(2.0));
    let spd_metrics = inspect_schur_block(&spd);
    assert!(spd_metrics.finite);
    assert_eq!(spd_metrics.asymmetry, 0.0);
    assert_eq!(spd_metrics.lower_min_eigenvalue, Some(2.0));
    assert_eq!(spd_metrics.upper_min_eigenvalue, Some(2.0));
    assert_eq!(spd_metrics.lower_min_cholesky_radicand, Some(2.0));
    assert_eq!(spd_metrics.upper_min_cholesky_radicand, Some(2.0));

    let mut non_spd = spd;
    non_spd[(0, 0)] = -1.0;
    let non_spd_metrics = inspect_schur_block(&non_spd);
    assert!(non_spd_metrics.finite);
    assert!(non_spd_metrics.lower_min_eigenvalue.unwrap() < 0.0);
    assert_eq!(non_spd_metrics.lower_min_cholesky_radicand, None);

    let mut nonfinite = spd;
    nonfinite[(0, 0)] = f64::NAN;
    let nonfinite_metrics = inspect_schur_block(&nonfinite);
    assert!(!nonfinite_metrics.finite);
    assert_eq!(nonfinite_metrics.lower_min_eigenvalue, None);
    assert_eq!(nonfinite_metrics.upper_min_cholesky_radicand, None);

    let mut asymmetric = spd;
    asymmetric[(0, 1)] = 3.0;
    asymmetric[(1, 0)] = 2.0;
    let asymmetric_metrics = inspect_schur_block(&asymmetric);
    assert_eq!(asymmetric_metrics.asymmetry, 1.0);
    assert!(asymmetric_metrics.lower_min_eigenvalue.is_some());
    assert!(asymmetric_metrics.upper_min_eigenvalue.is_some());
    assert_ne!(
        asymmetric_metrics.lower_min_eigenvalue,
        asymmetric_metrics.upper_min_eigenvalue
    );
}

#[test]
fn debug_context_maps_variable_slot_after_fixed_pose_and_counts_rig_crosses() {
    let pose_index = BTreeMap::from([(10_u64, 0_usize), (20_u64, 1_usize)]);
    let landmark_index = BTreeMap::from([(42_u64, 0_usize)]);
    // Use the pure mapping helper.  The environment-gated wrapper is not
    // exercised here so tests remain independent under the parallel test
    // runner.
    let context = schur_debug_context_for_slot(&pose_index, &landmark_index, 7, 1);
    assert_eq!(context.iteration, 7);
    assert_eq!(context.pose_slot, 1);
    assert_eq!(context.frame_id, Some(20));

    let cross = Matrix6x3::from_fn(|row, column| if row == column { 1.0 } else { 0.0 });
    let system = NormalEquationsBa {
        h_pp: CameraHessian::PoseDiagonal(vec![Matrix6::identity(); 2]),
        b_p: DVector::from_fn(12, |row, _| (row as f64 + 1.0) * 0.01),
        landmarks: vec![LandmarkBlock {
            h_ll: Matrix3::identity(),
            b_l: Vector3::new(0.02, -0.03, 0.04),
            cross: vec![(1, cross), (1, cross)],
        }],
    };
    let h_ll_inverse = (system.landmarks[0].h_ll + 0.25 * Matrix3::<f64>::identity())
        .try_inverse()
        .unwrap();
    let counts = collect_schur_block_debug_counts(&system, &[Some(h_ll_inverse)], 0.25, 1);
    assert_eq!(counts.local_landmarks, 1);
    assert_eq!(counts.valid_hll, 1);
    assert_eq!(counts.singular_hll, 0);
    assert_eq!(counts.cross_entries, 2);
    assert_eq!(counts.same_pose_groups, 1);
    assert_eq!(counts.same_pose_extra_cross_entries, 1);
    assert_eq!(counts.max_elimination_landmark, Some(0));
    assert!(counts.max_hll_inverse_residual.unwrap() < 1.0e-12);
    assert_eq!(context.landmark_index.get(&42), Some(&0));
    let diagonal = vec![10.0 * Matrix6::identity(); 2];
    let plain = solve_step_pose_blocks(&system, diagonal.clone(), 2, 1, 0.25, &mut None)
        .expect("positive definite control");
    let diagnosed =
        solve_step_pose_blocks_with_debug(&system, diagonal, 2, 1, 0.25, &mut None, Some(context))
            .expect("positive definite diagnostic");
    assert_eq!(plain, diagnosed, "diagnostics must not modify the solve");
    assert!(plain.0.norm() > 0.0);
    assert!(plain.1.norm() > 0.0);
    let elimination = (2.0 * cross) * h_ll_inverse * (2.0 * cross).transpose();
    assert!((counts.max_elimination_norm.unwrap() - elimination.norm()).abs() < 1.0e-12);
}
