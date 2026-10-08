use super::*;
use crate::imu_preintegration::ImuPreintegrator;
use nalgebra::UnitQuaternion;

fn make_problem() -> BundleAdjustment {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut ba = BundleAdjustment::new(camera);
    let pose_i = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.12, -0.08, 0.2),
        Vector3::new(0.3, -0.2, 0.1),
    );
    let pose_j = Pose::from_world_to_camera(
        UnitQuaternion::from_euler_angles(0.17, -0.03, 0.27),
        Vector3::new(0.42, -0.15, 0.18),
    );
    ba.add_pose(10, pose_i);
    ba.add_pose(20, pose_j);
    ba.add_velocity(10, Vector3::new(0.4, -0.1, 0.2));
    ba.add_velocity(20, Vector3::new(0.35, 0.05, 0.1));
    ba.add_bias(10, Vector6::new(0.002, -0.001, 0.003, 0.02, -0.01, 0.03));
    ba.add_bias(20, Vector6::zeros());
    ba.set_imu_body_to_camera(SE3::new(
        UnitQuaternion::from_euler_angles(-0.04, 0.03, -0.02),
        Vector3::new(-0.02, -0.06, 0.01),
    ));
    let mut preintegrator = ImuPreintegrator::new();
    for _ in 0..20 {
        preintegrator.integrate_sample(
            Vector3::new(0.03, -0.02, 0.04),
            Vector3::new(0.2, -0.1, 9.7),
            0.01,
        );
    }
    ba.add_imu_factor(ImuPreintegrationFactor {
        keyframe_id_from: 10,
        keyframe_id_to: 20,
        delta: preintegrator.delta(),
        gravity_world: Vector3::new(0.0, 0.0, -9.81),
        weight_position: 1.3,
        weight_velocity: 0.8,
        weight_rotation: 1.1,
    });
    ba
}

fn perturb(problem: &mut BundleAdjustment, coordinate: usize, step: f64) {
    match coordinate {
        0..=11 => {
            let pose_slot = coordinate / 6;
            let component = coordinate % 6;
            let id = [10_u64, 20][pose_slot];
            let mut xi = Vector6::zeros();
            xi[component] = step;
            let pose = problem.poses.get_mut(&id).unwrap();
            pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
        }
        12..=17 => {
            let velocity_slot = (coordinate - 12) / 3;
            let component = (coordinate - 12) % 3;
            let id = [10_u64, 20][velocity_slot];
            problem.velocities.get_mut(&id).unwrap()[component] += step;
        }
        18..=29 => {
            let bias_slot = (coordinate - 18) / 6;
            let component = (coordinate - 18) % 6;
            let id = [10_u64, 20][bias_slot];
            problem.biases.get_mut(&id).unwrap()[component] += step;
        }
        _ => unreachable!(),
    }
}

#[test]
fn analytic_imu_gradient_matches_central_difference_with_extrinsic() {
    let problem = make_problem();
    let linearized = problem.linearized_navigation_system().unwrap();
    assert_eq!(linearized.information.shape(), (30, 30));
    let epsilon = 1.0e-6;
    for coordinate in 0..30 {
        let mut plus = problem.clone();
        let mut minus = problem.clone();
        perturb(&mut plus, coordinate, epsilon);
        perturb(&mut minus, coordinate, -epsilon);
        let numerical = (plus.robust_cost(&RobustKernel::None)
            - minus.robust_cost(&RobustKernel::None))
            / (4.0 * epsilon);
        let analytic = linearized.gradient[coordinate];
        let tolerance = 2.0e-4 * analytic.abs().max(numerical.abs()).max(1.0);
        assert!(
            (analytic - numerical).abs() <= tolerance,
            "coordinate {coordinate}: analytic={analytic} numerical={numerical} tolerance={tolerance}"
        );
    }
}

/// `build_sqrt_factor_rows` must reproduce the full structured Hessian/gradient that
/// `build_normal_equations` scatters: `jac盞ﾂｷjac == H` and `jac盞ﾂｷresid == b` over the
/// pose+velocity+bias+landmark layout. This proves the emit path is exactly the dense
/// path in square-root form, so a Basalt step-A QR marginalization can consume it without
/// changing the dense numerics.
#[test]
fn sqrt_factor_rows_match_dense_normal_equations() {
    let mut ba = make_problem();
    // Two landmarks observed by the two keyframes (mono).
    ba.add_landmark(100, Point3::new(1.2, -0.4, 3.0));
    ba.add_landmark(101, Point3::new(-0.3, 0.9, 2.6));
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 100,
        xy: Point2::new(333.0, 239.0),
    });
    ba.add_observation(BaObservation {
        keyframe_id: 20,
        landmark_id: 100,
        xy: Point2::new(340.0, 238.0),
    });
    ba.add_observation(BaObservation {
        keyframe_id: 10,
        landmark_id: 101,
        xy: Point2::new(311.0, 246.0),
    });
    ba.add_observation(BaObservation {
        keyframe_id: 20,
        landmark_id: 101,
        xy: Point2::new(318.0, 244.0),
    });

    let pose_index: BTreeMap<u64, usize> = BTreeMap::from([(10, 0), (20, 1)]);
    let velocity_index: BTreeMap<u64, usize> = BTreeMap::from([(10, 0), (20, 1)]);
    let bias_index: BTreeMap<u64, usize> = BTreeMap::from([(10, 0), (20, 1)]);
    let landmark_index: BTreeMap<u64, usize> = BTreeMap::from([(100, 0), (101, 1)]);
    let intrinsics = (500.0_f64, 500.0_f64, 320.0_f64, 240.0_f64);

    let stack = build_sqrt_factor_rows(
        &ba,
        &intrinsics,
        &pose_index,
        &landmark_index,
        &velocity_index,
        &bias_index,
        &RobustKernel::None,
        None,
    )
    .expect("stack with observations and IMU");
    let (h_sqrt, b_sqrt) = stack.to_normal_equations();

    let dense = build_normal_equations(
        &ba,
        &intrinsics,
        &pose_index,
        &landmark_index,
        &velocity_index,
        &bias_index,
        &RobustKernel::None,
        None,
        false,
        false,
    );
    // `prefer_pose_blocks = false` above (and the non-empty velocity/bias
    // indices regardless) guarantees the dense representation.
    let h_pp_dense = match &dense.h_pp {
        CameraHessian::Dense(matrix) => matrix,
        CameraHessian::PoseDiagonal(_) => {
            panic!("expected a dense camera Hessian for this test")
        }
    };

    // Assemble the full structured Hessian / gradient from the dense NormalEquationsBa:
    //   H = [ h_pp        cross^T ]
    //       [ cross       h_ll    ]
    // ordering columns as pose(12), vel(6), bias(12), landmark(6).
    let p = 2usize;
    let v = 2usize;
    let b = 2usize;
    let l = 2usize;
    let pose_dim = p * 6;
    let vel_offset = pose_dim;
    let bias_offset = vel_offset + v * 3;
    let lm_offset = bias_offset + b * 6;
    let total = lm_offset + l * 3;
    let mut h_full = DMatrix::<f64>::zeros(total, total);
    let mut b_full = DVector::<f64>::zeros(total);

    // h_pp / b_p (pose+vel+bias).
    h_full
        .view_mut((0, 0), (lm_offset, lm_offset))
        .copy_from(h_pp_dense);
    b_full
        .view_mut((0, 0), (lm_offset, 1))
        .copy_from(&dense.b_p);

    // Landmark block (block-diagonal h_ll) and cross (h_pl).
    for (l_idx, lm) in dense.landmarks.iter().enumerate() {
        for a in 0..3 {
            for c in 0..3 {
                h_full[(lm_offset + l_idx * 3 + a, lm_offset + l_idx * 3 + c)] += lm.h_ll[(a, c)];
            }
            b_full[lm_offset + l_idx * 3 + a] += lm.b_l[a];
        }
        for (p_idx, cross) in &lm.cross {
            for a in 0..6 {
                for c in 0..3 {
                    h_full[(p_idx * 6 + a, lm_offset + l_idx * 3 + c)] += cross[(a, c)];
                    h_full[(lm_offset + l_idx * 3 + c, p_idx * 6 + a)] += cross[(a, c)];
                }
            }
        }
    }

    let max_h = (h_sqrt - h_full.clone())
        .iter()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(max_h < 1e-8, "sqrt-jac H mismatch: {max_h}");
    let max_b = (b_sqrt - b_full.clone())
        .iter()
        .fold(0.0_f64, |acc, x| acc.max(x.abs()));
    assert!(max_b < 1e-8, "sqrt-jac b mismatch: {max_b}");
}
