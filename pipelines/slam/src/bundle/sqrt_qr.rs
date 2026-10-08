//! Square-root factor rows and the rig QR / PCG step.

use super::*;

/// A square-root (QR-friendly) factor stack of a [`BundleAdjustment`] over the *structural*
/// (pre-Schur) state layout: pose + velocity + bias + landmark columns in the same order as
/// `build_normal_equations`. Each row is a whitened Jacobian row and the matching whitened
/// residual. `jacᵀ·jac` equals the full structured Hessian and `jacᵀ·resid` the full structured
/// gradient, so a caller can marginalize any column subset with
/// [`crate::marginalize_sqrt`] (Basalt Step A: never forming `JᵀJ`, never inverting a dense
/// block).
#[derive(Debug, Clone)]
pub struct SqrtStack {
    /// Stacked whitened Jacobian rows (rows = factors, cols = `total_dof`).
    pub jac: DMatrix<f64>,
    /// Stacked whitened residuals (one per row).
    pub resid: DVector<f64>,
    /// Total pose DoF (6·P).
    pub pose_dof: usize,
    /// Start column of the velocity block.
    pub vel_offset: usize,
    /// Start column of the bias block.
    pub bias_offset: usize,
    /// Start column of the landmark block.
    pub landmark_offset: usize,
    /// Total state DoF (number of `jac` columns).
    pub total_dof: usize,
}

impl SqrtStack {
    /// Normal equations implied by the stack: `(jacᵀ·jac, jacᵀ·resid)`.
    pub fn to_normal_equations(&self) -> (DMatrix<f64>, DVector<f64>) {
        let h = self.jac.transpose() * &self.jac;
        let b = self.jac.transpose() * &self.resid;
        (h, b)
    }
}

/// Build the square-root factor stack of a [`BundleAdjustment`] over its structural state
/// layout (pose + velocity + bias + landmark columns, in `build_normal_equations` order).
///
/// This mirrors the linearization inside `build_normal_equations`, reusing the exact Jacobian
/// and whitening computations so that `jacᵀ·jac == build_normal_equations(...).h_pp` (respecting
/// the landmark block) and `jacᵀ·resid == b_p`. It is intended as the input to a Basalt-step-A
/// square-root marginalization: unlike the dense `h_pp`/`b_p`, the stack keeps each factor's
/// whitened row so a leaving block can be dropped by QR without forming `JᵀJ` or inverting a
/// block.
///
/// Returns `None` when no factor contributes a Jacobian row (empty problem).
#[allow(clippy::too_many_arguments)]
pub fn build_sqrt_factor_rows(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    velocity_index: &BTreeMap<u64, usize>,
    bias_index: &BTreeMap<u64, usize>,
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
) -> Option<SqrtStack> {
    let p_count = pose_index.len();
    let l_count = landmark_index.len();
    let v_count = velocity_index.len();
    let b_count = bias_index.len();
    let pose_dim = p_count * 6;
    let vel_offset = pose_dim;
    let bias_offset = pose_dim + v_count * 3;
    let landmark_offset = pose_dim + v_count * 3 + b_count * 6;
    let total_dof = landmark_offset + l_count * 3;

    // First pass: compute the number of Jacobian rows to size the matrices exactly.
    let mut n_rows = 0usize;
    for obs in &ba.observations {
        if pose_index.contains_key(&obs.keyframe_id)
            || landmark_index.contains_key(&obs.landmark_id)
        {
            n_rows += 2;
        }
    }
    for _ in &ba.stereo_observations {
        if ba.stereo_baseline.is_some() {
            n_rows += 3;
        }
    }
    for obs in &ba.general_stereo_observations {
        if general_stereo_residual_jacobians(
            intrinsics,
            obs,
            &ba.poses[&obs.keyframe_id],
            &ba.landmarks[&obs.landmark_id],
        )
        .is_some()
        {
            n_rows += 4;
        }
    }
    for factor in &ba.imu_factors {
        if ba.poses.contains_key(&factor.keyframe_id_from)
            && ba.poses.contains_key(&factor.keyframe_id_to)
            && ba.velocities.contains_key(&factor.keyframe_id_from)
            && ba.velocities.contains_key(&factor.keyframe_id_to)
        {
            n_rows += 9;
        }
    }
    if n_rows == 0 {
        return None;
    }

    let mut jac = DMatrix::zeros(n_rows, total_dof);
    let mut resid = DVector::zeros(n_rows);
    let mut row = 0usize;

    let stereo_offset = ba.observations.len();
    let general_offset = ba.observations.len() + ba.stereo_observations.len();

    let mono_projection = MonoProjection::for_camera(&ba.camera);
    for (obs_idx, obs) in ba.observations.iter().enumerate() {
        let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
            continue;
        };
        let Some(point) = ba.landmarks.get(&obs.landmark_id) else {
            continue;
        };
        let r_mat = pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let xc = pose.transform_world_point(point);
        if xc.z <= 0.0 {
            continue;
        }
        let Some((predicted, j_pi)) = mono_project_with_jacobian(mono_projection, intrinsics, &xc)
        else {
            continue;
        };
        let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);
        let xw_skew = skew(&point.coords);
        let mut dx_dxi = nalgebra::Matrix3x6::<f64>::zeros();
        dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
        dx_dxi
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-r_mat * xw_skew));
        let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
        let j_lm: Matrix2x3<f64> = j_pi * r_mat;
        let s = residual.x * residual.x + residual.y * residual.y;
        let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[obs_idx]);
        let sqrt_w = w.max(0.0).sqrt();

        let i_pose = pose_index.get(&obs.keyframe_id).copied();
        let i_lm = landmark_index.get(&obs.landmark_id).copied();
        if let (Some(p), Some(l)) = (i_pose, i_lm) {
            for r in 0..2 {
                let mut jr = DVector::<f64>::zeros(total_dof);
                for c in 0..6 {
                    jr[p * 6 + c] = sqrt_w * j_pose[(r, c)];
                }
                for c in 0..3 {
                    jr[landmark_offset + l * 3 + c] = sqrt_w * j_lm[(r, c)];
                }
                jac.row_mut(row).copy_from(&jr.transpose());
                resid[row] = sqrt_w * residual[r];
                row += 1;
            }
        }
    }

    if let Some(baseline) = ba
        .stereo_baseline
        .filter(|b| b.is_finite() && *b > 0.0)
        .filter(|_| !ba.stereo_observations.is_empty())
    {
        let (fx, fy, _, _) = *intrinsics;
        for (st_idx, obs) in ba.stereo_observations.iter().enumerate() {
            let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
                continue;
            };
            let Some(point) = ba.landmarks.get(&obs.landmark_id) else {
                continue;
            };
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            let Some(predicted) = project_pinhole(intrinsics, &xc) else {
                continue;
            };
            let u_r_pred = predicted.x - fx * baseline / xc.z;
            let residual = Vector3::new(
                predicted.x - obs.xy.x,
                predicted.y - obs.xy.y,
                u_r_pred - obs.u_right,
            );
            let z_inv = 1.0 / xc.z;
            let z_inv2 = z_inv * z_inv;
            let mut j_pi = Matrix3::<f64>::zeros();
            j_pi[(0, 0)] = fx * z_inv;
            j_pi[(0, 2)] = -fx * xc.x * z_inv2;
            j_pi[(1, 1)] = fy * z_inv;
            j_pi[(1, 2)] = -fy * xc.y * z_inv2;
            j_pi[(2, 0)] = fx * z_inv;
            j_pi[(2, 2)] = -fx * (xc.x - baseline) * z_inv2;
            let xw_skew = skew(&point.coords);
            let mut dx_dxi = Matrix3x6::<f64>::zeros();
            dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
            dx_dxi
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r_mat * xw_skew));
            let j_pose: Matrix3x6<f64> = j_pi * dx_dxi;
            let j_lm: Matrix3<f64> = j_pi * r_mat;
            let s = residual.norm_squared();
            let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);
            let sqrt_w = w.max(0.0).sqrt();
            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();
            if let (Some(p), Some(l)) = (i_pose, i_lm) {
                for r in 0..3 {
                    let mut jr = DVector::<f64>::zeros(total_dof);
                    for c in 0..6 {
                        jr[p * 6 + c] = sqrt_w * j_pose[(r, c)];
                    }
                    for c in 0..3 {
                        jr[landmark_offset + l * 3 + c] = sqrt_w * j_lm[(r, c)];
                    }
                    jac.row_mut(row).copy_from(&jr.transpose());
                    resid[row] = sqrt_w * residual[r];
                    row += 1;
                }
            }
        }
    }

    for (index, obs) in ba.general_stereo_observations.iter().enumerate() {
        let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
            continue;
        };
        let Some(point) = ba.landmarks.get(&obs.landmark_id) else {
            continue;
        };
        let Some((residual, j_pose, j_lm)) =
            general_stereo_residual_jacobians(intrinsics, obs, pose, point)
        else {
            continue;
        };
        let s = residual.norm_squared();
        let w =
            kernel.weight(s) * gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
        let sqrt_w = w.max(0.0).sqrt();
        let i_pose = pose_index.get(&obs.keyframe_id).copied();
        let i_lm = landmark_index.get(&obs.landmark_id).copied();
        if let (Some(p), Some(l)) = (i_pose, i_lm) {
            let n = j_pose.nrows();
            for r in 0..n {
                let mut jr = DVector::<f64>::zeros(total_dof);
                for c in 0..6 {
                    jr[p * 6 + c] = sqrt_w * j_pose[(r, c)];
                }
                for c in 0..3 {
                    jr[landmark_offset + l * 3 + c] = sqrt_w * j_lm[(r, c)];
                }
                // Each general-stereo observation contributes `n` rows (2 or 4).
                jac.row_mut(row).copy_from(&jr.transpose());
                resid[row] = sqrt_w * residual[r];
                row += 1;
            }
        }
    }

    // IMU factors: 9 whitened rows each over [pose_i | pose_j | vel_i | vel_j | bias_i].
    for factor in &ba.imu_factors {
        let (Some(pose_i), Some(pose_j)) = (
            ba.poses.get(&factor.keyframe_id_from),
            ba.poses.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let (Some(v_i_w), Some(v_j_w)) = (
            ba.velocities.get(&factor.keyframe_id_from),
            ba.velocities.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let body_i = ba.imu_body_to_camera.compose(&pose_i.world_to_camera);
        let body_j = ba.imu_body_to_camera.compose(&pose_j.world_to_camera);
        let r_wc_i = body_i.rotation.to_rotation_matrix().into_inner();
        let r_wc_j = body_j.rotation.to_rotation_matrix().into_inner();
        let c_i: Vector3<f64> = body_i.inverse().translation;
        let c_j: Vector3<f64> = body_j.inverse().translation;
        let dt = factor.delta.delta_time;
        let g = factor.gravity_world;

        let r_i_so3 = SO3::from_quaternion(body_i.rotation.inverse());
        let r_j_so3 = SO3::from_quaternion(body_j.rotation.inverse());
        let bias_for_factor = ba.biases.get(&factor.keyframe_id_from);
        let [r_rot, r_vel, r_pos] = if let Some(bias) = bias_for_factor {
            let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
            let ba_acc: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
            factor.residual_with_bias_correction(
                &r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w, &bg, &ba_acc,
            )
        } else {
            factor.residual(&r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w)
        };

        let whitener = factor.covariance_sqrt_information().unwrap_or_else(|| {
            let mut diagonal = nalgebra::SVector::<f64, 9>::zeros();
            diagonal
                .fixed_rows_mut::<3>(0)
                .fill(factor.weight_rotation.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(3)
                .fill(factor.weight_velocity.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(6)
                .fill(factor.weight_position.max(0.0).sqrt());
            crate::imu_preintegration::Matrix9::from_diagonal(&diagonal)
        });

        let q_diff = v_j_w - v_i_w - g * dt;
        let q_pos_i = c_j - v_i_w * dt - 0.5 * g * dt * dt;
        let jr_inv = right_jacobian_inverse_so3(&r_rot);
        let jr_inv_rwc_j = jr_inv * r_wc_j;

        let mut j_pose_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        j_pose_i
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&jr_inv_rwc_j);
        j_pose_i
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-r_wc_i * skew(&q_diff)));
        j_pose_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_wc_i);
        j_pose_i
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-r_wc_i * skew(&q_pos_i)));

        let mut j_pose_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        j_pose_j
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-jr_inv_rwc_j));
        j_pose_j.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-r_wc_i));
        j_pose_j
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(r_wc_i * skew(&c_j)));

        let mut j_vel_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_i.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_wc_i));
        j_vel_i
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-dt * r_wc_i));
        let mut j_vel_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_wc_i);

        let mut r_stack = nalgebra::SVector::<f64, 9>::zeros();
        r_stack.fixed_rows_mut::<3>(0).copy_from(&r_rot);
        r_stack.fixed_rows_mut::<3>(3).copy_from(&r_vel);
        r_stack.fixed_rows_mut::<3>(6).copy_from(&r_pos);

        let i_bias = bias_index.get(&factor.keyframe_id_from).copied();
        let j_bias_block = if i_bias.is_some() {
            let neg_r_rot_mat: Matrix3<f64> = nalgebra::Rotation3::from_scaled_axis(-r_rot).into();
            let lhs_rot = -jr_inv * neg_r_rot_mat;
            let mut j_bias = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
            j_bias
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(lhs_rot * factor.delta.j_rotation_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-factor.delta.j_velocity_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(-factor.delta.j_velocity_ba));
            j_bias
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(-factor.delta.j_position_bg));
            j_bias
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-factor.delta.j_position_ba));
            Some(j_bias)
        } else {
            None
        };

        let j_pose_i = whitener * j_pose_i;
        let j_pose_j = whitener * j_pose_j;
        let j_vel_i = whitener * j_vel_i;
        let j_vel_j = whitener * j_vel_j;
        let j_bias_block = j_bias_block.map(|jb| whitener * jb);
        let r_stack = whitener * r_stack;

        let i_pose = pose_index.get(&factor.keyframe_id_from).copied();
        let j_pose = pose_index.get(&factor.keyframe_id_to).copied();
        let i_vel = velocity_index.get(&factor.keyframe_id_from).copied();
        let j_vel = velocity_index.get(&factor.keyframe_id_to).copied();

        for r in 0..9 {
            let mut jr = DVector::<f64>::zeros(total_dof);
            if let Some(p) = i_pose {
                for c in 0..6 {
                    jr[p * 6 + c] += j_pose_i[(r, c)];
                }
            }
            if let Some(p) = j_pose {
                for c in 0..6 {
                    jr[p * 6 + c] += j_pose_j[(r, c)];
                }
            }
            if let Some(v) = i_vel {
                for c in 0..3 {
                    jr[vel_offset + v * 3 + c] += j_vel_i[(r, c)];
                }
            }
            if let Some(v) = j_vel {
                for c in 0..3 {
                    jr[vel_offset + v * 3 + c] += j_vel_j[(r, c)];
                }
            }
            if let (Some(b), Some(jb)) = (i_bias, &j_bias_block) {
                for c in 0..6 {
                    jr[bias_offset + b * 6 + c] += jb[(r, c)];
                }
            }
            jac.row_mut(row).copy_from(&jr.transpose());
            resid[row] = r_stack[r];
            row += 1;
        }
    }

    Some(SqrtStack {
        jac,
        resid,
        pose_dof: pose_dim,
        vel_offset,
        bias_offset,
        landmark_offset,
        total_dof,
    })
}

pub(super) fn pinhole_projection_jacobian(
    intrinsics: &(f64, f64, f64, f64),
    point: &Point3<f64>,
) -> Option<Matrix2x3<f64>> {
    if point.z <= 0.0 {
        return None;
    }
    let (fx, fy, _, _) = *intrinsics;
    let z_inv = point.z.recip();
    let z_inv2 = z_inv * z_inv;
    Some(Matrix2x3::new(
        fx * z_inv,
        0.0,
        -fx * point.x * z_inv2,
        0.0,
        fy * z_inv,
        -fy * point.y * z_inv2,
    ))
}

pub(super) fn general_stereo_residual_jacobians(
    left_intrinsics: &(f64, f64, f64, f64),
    observation: &BaGeneralStereoObservation,
    pose: &Pose,
    point_world: &Point3<f64>,
) -> Option<(Vector4<f64>, Matrix4x6<f64>, Matrix4x3<f64>)> {
    let right_intrinsics = observation.right_camera.intrinsics()?;
    let point_left = pose.transform_world_point(point_world);
    let point_right = observation.left_to_right.transform_point(&point_left);
    let predicted_left = project_pinhole(left_intrinsics, &point_left)?;
    let predicted_right = project_pinhole(&right_intrinsics, &point_right)?;
    let residual = Vector4::new(
        predicted_left.x - observation.xy_left.x,
        predicted_left.y - observation.xy_left.y,
        predicted_right.x - observation.xy_right.x,
        predicted_right.y - observation.xy_right.y,
    );

    let j_left_projection = pinhole_projection_jacobian(left_intrinsics, &point_left)?;
    let j_right_projection = pinhole_projection_jacobian(&right_intrinsics, &point_right)?;
    let rotation_world_to_left = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation_left_to_right = observation
        .left_to_right
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut d_left_d_pose = Matrix3x6::<f64>::zeros();
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_world_to_left);
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_world_to_left * skew(&point_world.coords)));

    let j_left_pose = j_left_projection * d_left_d_pose;
    let j_right_pose = j_right_projection * rotation_left_to_right * d_left_d_pose;
    let j_left_landmark = j_left_projection * rotation_world_to_left;
    let j_right_landmark = j_right_projection * rotation_left_to_right * rotation_world_to_left;
    let mut j_pose = Matrix4x6::<f64>::zeros();
    j_pose.fixed_rows_mut::<2>(0).copy_from(&j_left_pose);
    j_pose.fixed_rows_mut::<2>(2).copy_from(&j_right_pose);
    let mut j_landmark = Matrix4x3::<f64>::zeros();
    j_landmark
        .fixed_rows_mut::<2>(0)
        .copy_from(&j_left_landmark);
    j_landmark
        .fixed_rows_mut::<2>(2)
        .copy_from(&j_right_landmark);
    Some((residual, j_pose, j_landmark))
}

pub(super) fn qr_shadow_dimensions_allowed(
    variable_poses: usize,
    poses: usize,
    landmarks: usize,
    rows: usize,
) -> bool {
    variable_poses <= 60 && poses <= 64 && landmarks <= 1024 && (1..=20000).contains(&rows)
}

#[allow(clippy::result_large_err)] // Same diagnostic payload as the existing linear-step boundary.
pub(super) fn solve_rig_qr_step(
    ba: &BundleAdjustment,
    config: &BaConfig,
    lambda: f64,
    options: MatrixFreeBaOptions,
    landmark_index: &BTreeMap<u64, usize>,
) -> Result<MatrixFreeStepOutcome, MatrixFreeStepError> {
    let failure = |diagnostic: String| MatrixFreeStepError {
        diagnostic,
        pcg_iterations: None,
        pcg_residual_norm: None,
        pcg_target: None,
        restart_diagnostics: None,
    };
    let qr = RigQrLinearization::new(ba, config, lambda).map_err(failure)?;
    let pcg = qr
        .solve(implicit_schur::PcgOptions {
            max_iterations: options.max_pcg_iterations,
            relative_tolerance: options.pcg_relative_tolerance,
            absolute_tolerance: options.pcg_absolute_tolerance,
        })
        .map_err(|error| {
            let (pcg_iterations, pcg_residual_norm, pcg_target) = error.diagnostics();
            MatrixFreeStepError {
                diagnostic: format!("{error:?}"),
                pcg_iterations,
                pcg_residual_norm,
                pcg_target,
                restart_diagnostics: None,
            }
        })?;
    // Unobserved variable landmarks have only damping and hence a zero step.
    // The caller's complete layout is authoritative, not the observed-track list.
    let mut delta_landmarks = DVector::zeros(3 * landmark_index.len());
    let mut scratch = Vec::new();
    for (id, block) in &qr.blocks {
        if let Some(delta) = block
            .back_substitute(pcg.solution.as_slice(), &mut scratch)
            .map_err(|e| failure(e.into()))?
        {
            let slot = landmark_index
                .get(id)
                .ok_or_else(|| failure("QR landmark layout mismatch".into()))?;
            delta_landmarks
                .fixed_rows_mut::<3>(slot * 3)
                .copy_from(&Vector3::from(delta));
        }
    }
    Ok(MatrixFreeStepOutcome {
        delta_poses: pcg.solution,
        delta_landmarks,
        diagnostics: MatrixFreeBaIterationStats {
            iteration: 0,
            pcg_iterations: Some(pcg.iterations),
            pcg_residual_norm: Some(pcg.residual_norm),
            pcg_target: Some(pcg.target),
            pcg_failure: None,
        },
        restart_diagnostics: None,
        adaptive_prediction: None,
        quality: None,
    })
}

/// Bounded rig QR adapter: reuses the actual rig Jacobians and robust kernel.
pub(super) struct RigQrLinearization {
    #[cfg(test)]
    pub(super) pose_index: BTreeMap<u64, usize>,
    pub(super) blocks: Vec<(u64, crate::landmark_qr::ReducedLandmark)>,
    pose_diagonal: Vec<f64>,
    #[cfg(test)]
    pub(super) observation_rows: usize,
    pub(super) preconditioner: Vec<Matrix6<f64>>,
}

impl RigQrLinearization {
    pub(super) fn new(
        ba: &BundleAdjustment,
        config: &BaConfig,
        lambda: f64,
    ) -> Result<Self, String> {
        ba.validate_matrix_free_entry(config, MatrixFreeBaOptions::default())
            .map_err(|e| e.to_string())?;
        if ba.rig_observations.is_empty()
            || !ba.observations.is_empty()
            || !ba.stereo_observations.is_empty()
            || !ba.general_stereo_observations.is_empty()
            || !lambda.is_finite()
            || lambda <= 0.0
        {
            return Err("QR adapter requires pure rig observations and positive damping".into());
        }
        let pose_index: BTreeMap<_, _> = ba
            .poses
            .keys()
            .filter(|id| !ba.fixed_poses.contains(id))
            .enumerate()
            .map(|(slot, id)| (*id, slot))
            .collect();
        let mut pose_diagonal = vec![lambda; pose_index.len() * 6];
        for id in &ba.fixed_pose_rotations {
            if let Some(slot) = pose_index.get(id) {
                pose_diagonal[slot * 6 + 3..slot * 6 + 6].fill(1.0 + lambda);
            }
        }
        let mut rows: BTreeMap<u64, Vec<crate::landmark_qr::WeightedRow>> = BTreeMap::new();
        for obs in &ba.rig_observations {
            let pose = ba.poses.get(&obs.keyframe_id).ok_or("missing rig pose")?;
            let point = ba
                .landmarks
                .get(&obs.landmark_id)
                .ok_or("missing rig landmark")?;
            let (residual, jp, jl) =
                rig_residual_jacobians(obs, pose, point).ok_or("nonprojectable rig row")?;
            let weight = config.robust_kernel.weight(residual.norm_squared());
            if !weight.is_finite() || weight < 0.0 {
                return Err("invalid rig weight".into());
            }
            let scale = weight.sqrt();
            for r in 0..2 {
                rows.entry(obs.landmark_id)
                    .or_default()
                    .push(crate::landmark_qr::WeightedRow {
                        pose: pose_index.get(&obs.keyframe_id).copied(),
                        pose_jacobian: std::array::from_fn(|c| {
                            if c >= 3 && ba.fixed_pose_rotations.contains(&obs.keyframe_id) {
                                0.0
                            } else {
                                scale * jp[(r, c)]
                            }
                        }),
                        landmark_jacobian: std::array::from_fn(|c| scale * jl[(r, c)]),
                        residual: scale * residual[r],
                    });
            }
        }
        #[cfg(test)]
        let observation_rows = rows.values().map(Vec::len).sum();
        let mut preconditioner: Vec<_> = pose_diagonal
            .chunks_exact(6)
            .map(|d| Matrix6::from_diagonal(&Vector6::from_column_slice(d)))
            .collect();
        let blocks = rows
            .into_iter()
            .map(|(id, rows)| {
                crate::landmark_qr::ReducedLandmark::new_accumulating_diagonal(
                    rows,
                    pose_index.len(),
                    !ba.fixed_landmarks.contains(&id),
                    lambda,
                    &mut preconditioner,
                )
                .map(|block| (id, block))
                .map_err(str::to_owned)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for block in &mut preconditioner {
            if !block.iter().all(|v| v.is_finite()) {
                return Err("nonfinite QR preconditioner".into());
            }
            *block = block
                .cholesky()
                .ok_or("non-SPD QR preconditioner")?
                .inverse();
            if !block.iter().all(|v| v.is_finite()) {
                return Err("nonfinite QR preconditioner inverse".into());
            }
        }
        Ok(Self {
            #[cfg(test)]
            pose_index,
            blocks,
            pose_diagonal,
            #[cfg(test)]
            observation_rows,
            preconditioner,
        })
    }

    pub(super) fn apply(&self, x: &[f64]) -> Result<Vec<f64>, &'static str> {
        if x.len() != self.pose_diagonal.len() || !x.iter().all(|v| v.is_finite()) {
            return Err("invalid QR pose vector");
        }
        let mut out: Vec<_> = x
            .iter()
            .zip(&self.pose_diagonal)
            .map(|(x, d)| x * d)
            .collect();
        let mut scratch = Vec::new();
        for (_, block) in &self.blocks {
            block.normal_add(x, &mut out, &mut scratch)?;
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err("nonfinite QR normal action");
        }
        Ok(out)
    }

    pub(super) fn rhs(&self) -> Result<Vec<f64>, &'static str> {
        let mut out = vec![0.0; self.pose_diagonal.len()];
        let mut scratch = Vec::new();
        for (_, block) in &self.blocks {
            block.rhs_add(&mut out, &mut scratch)?;
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err("nonfinite QR right hand side");
        }
        Ok(out)
    }

    pub(super) fn precondition(
        &self,
        residual: &DVector<f64>,
    ) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError> {
        if residual.len() != self.pose_diagonal.len() {
            return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
                expected: self.pose_diagonal.len(),
                actual: residual.len(),
            });
        }
        let mut out = DVector::zeros(residual.len());
        for (slot, inverse) in self.preconditioner.iter().enumerate() {
            out.fixed_rows_mut::<6>(slot * 6)
                .copy_from(&(inverse * residual.fixed_rows::<6>(slot * 6)));
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "QR preconditioner action",
            ));
        }
        Ok(out)
    }

    pub(super) fn solve(
        &self,
        options: implicit_schur::PcgOptions,
    ) -> Result<implicit_schur::PcgResult, implicit_schur::ImplicitSchurError> {
        let rhs = DVector::from_vec(
            self.rhs()
                .map_err(implicit_schur::ImplicitSchurError::NonFinite)?,
        );
        solve_qr_pcg(
            &rhs,
            rhs.len(),
            options,
            |x| {
                self.apply(x.as_slice())
                    .map(DVector::from_vec)
                    .map_err(implicit_schur::ImplicitSchurError::NonFinite)
            },
            |r| self.precondition(r),
        )
    }
}

fn checked_qr_pcg_result<Apply>(
    rhs: &DVector<f64>,
    solution: DVector<f64>,
    iterations: usize,
    recursive_norm: f64,
    target: f64,
    apply: &mut Apply,
) -> Result<implicit_schur::PcgResult, implicit_schur::ImplicitSchurError>
where
    Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
{
    let applied = apply(&solution)?;
    let true_residual = rhs - &applied;
    let true_norm = true_residual.norm();
    if !true_norm.is_finite() {
        return Err(implicit_schur::ImplicitSchurError::NonFinite(
            "true PCG residual",
        ));
    }
    if true_norm > target {
        return Err(implicit_schur::ImplicitSchurError::ResidualCheckFailed {
            iterations,
            recursive_norm,
            true_norm,
            target,
        });
    }
    Ok(implicit_schur::PcgResult {
        solution,
        iterations,
        residual_norm: true_norm,
        target,
    })
}

/// Generic QR PCG recurrence, originally audited against the implicit
/// operator.  The callbacks make it possible to run the exact recurrence
/// against either the implicit action or a borrowed explicit lower-Schur
/// matrix. The existing Schur solver remains unchanged.
pub(super) fn solve_qr_pcg<Apply, Preconditioner>(
    rhs: &DVector<f64>,
    dimension: usize,
    options: implicit_schur::PcgOptions,
    mut apply: Apply,
    mut apply_preconditioner: Preconditioner,
) -> Result<implicit_schur::PcgResult, implicit_schur::ImplicitSchurError>
where
    Apply: FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
    Preconditioner:
        FnMut(&DVector<f64>) -> Result<DVector<f64>, implicit_schur::ImplicitSchurError>,
{
    if rhs.len() != dimension {
        return Err(implicit_schur::ImplicitSchurError::DimensionMismatch {
            expected: dimension,
            actual: rhs.len(),
        });
    }
    if !rhs.iter().all(|value| value.is_finite()) {
        return Err(implicit_schur::ImplicitSchurError::NonFinite(
            "PCG right hand side",
        ));
    }
    if !options.relative_tolerance.is_finite()
        || options.relative_tolerance < 0.0
        || !options.absolute_tolerance.is_finite()
        || options.absolute_tolerance < 0.0
    {
        return Err(implicit_schur::ImplicitSchurError::InvalidTolerance);
    }
    let rhs_norm = rhs.norm();
    let target = options
        .absolute_tolerance
        .max(options.relative_tolerance * rhs_norm);
    if !target.is_finite() {
        return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG target"));
    }
    let mut solution = DVector::zeros(dimension);
    let mut residual = rhs.clone();
    let mut residual_norm = residual.norm();
    if !residual_norm.is_finite() {
        return Err(implicit_schur::ImplicitSchurError::NonFinite(
            "initial residual",
        ));
    }
    if residual_norm <= target {
        return checked_qr_pcg_result(rhs, solution, 0, residual_norm, target, &mut apply);
    }
    if options.max_iterations == 0 {
        return Err(implicit_schur::ImplicitSchurError::MaxIterations {
            iterations: 0,
            recursive_norm: residual_norm,
            residual_norm,
            target,
        });
    }

    let mut preconditioned = apply_preconditioner(&residual)?;
    let mut direction = preconditioned.clone();
    let mut rho = residual.dot(&preconditioned);
    if !rho.is_finite() || rho <= 0.0 {
        return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
    }

    for iteration in 1..=options.max_iterations {
        let applied = apply(&direction)?;
        let curvature = direction.dot(&applied);
        if !curvature.is_finite() || curvature <= 0.0 {
            return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
        }
        let alpha = rho / curvature;
        if !alpha.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG step"));
        }
        solution += alpha * &direction;
        residual -= alpha * applied;
        residual_norm = residual.norm();
        if !solution.iter().all(|value| value.is_finite()) || !residual_norm.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite("PCG iterate"));
        }
        if residual_norm <= target {
            return checked_qr_pcg_result(
                rhs,
                solution,
                iteration,
                residual_norm,
                target,
                &mut apply,
            );
        }
        if iteration == options.max_iterations {
            let true_residual = rhs - &apply(&solution)?;
            let true_norm = true_residual.norm();
            if !true_norm.is_finite() {
                return Err(implicit_schur::ImplicitSchurError::NonFinite(
                    "true PCG residual",
                ));
            }
            if true_norm <= target {
                return Ok(implicit_schur::PcgResult {
                    solution,
                    iterations: iteration,
                    residual_norm: true_norm,
                    target,
                });
            }
            return Err(implicit_schur::ImplicitSchurError::MaxIterations {
                iterations: iteration,
                recursive_norm: residual_norm,
                residual_norm: true_norm,
                target,
            });
        }
        preconditioned = apply_preconditioner(&residual)?;
        let next_rho = residual.dot(&preconditioned);
        if !next_rho.is_finite() || next_rho <= 0.0 {
            return Err(implicit_schur::ImplicitSchurError::NonPositiveCurvature);
        }
        let beta = next_rho / rho;
        if !beta.is_finite() {
            return Err(implicit_schur::ImplicitSchurError::NonFinite(
                "PCG direction",
            ));
        }
        direction = &preconditioned + beta * direction;
        rho = next_rho;
    }
    unreachable!("the max-iteration branch returns above");
}

/// `pub(crate)` (rather than private) solely so
/// `colmap_incremental::rig_ba_solver`'s test suite can cross-check its own
/// (independently, allocation-free re-derived) residual/Jacobian formula
/// against this already-exercised implementation on random inputs. The
/// native solver's hot path does **not** call this (it stays link-independent
/// of `bundle.rs`, per its module doc); only its `#[cfg(test)]` module does.
pub(crate) fn rig_residual_jacobians(
    observation: &BaRigObservation,
    pose: &Pose,
    point_world: &Point3<f64>,
) -> Option<(Vector2<f64>, Matrix2x6<f64>, Matrix2x3<f64>)> {
    let intrinsics = observation.camera.intrinsics()?;
    let point_rig = pose.transform_world_point(point_world);
    let point_sensor = observation.sensor_from_rig.transform_point(&point_rig);
    let predicted = project_pinhole(&intrinsics, &point_sensor)?;
    let residual = predicted - observation.xy;
    let projection = pinhole_projection_jacobian(&intrinsics, &point_sensor)?;
    let rotation_world_to_rig = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation_rig_to_sensor = observation
        .sensor_from_rig
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut d_rig_d_pose = Matrix3x6::<f64>::zeros();
    d_rig_d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_world_to_rig);
    d_rig_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_world_to_rig * skew(&point_world.coords)));
    let j_pose = projection * rotation_rig_to_sensor * d_rig_d_pose;
    let j_landmark = projection * rotation_rig_to_sensor * rotation_world_to_rig;
    Some((residual, j_pose, j_landmark))
}

pub(super) fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Right-Jacobian inverse on SO(3): Jr⁻¹(φ) = I + ½[φ]× + c·[φ]×²,
/// with `c = (1/θ²) · (1 − (θ/2)·cot(θ/2))` for `θ = ‖φ‖`. At `θ → 0`
/// falls back to the leading Taylor expansion `I + ½[φ]× + (1/12)·[φ]×²`.
/// Used by [`build_normal_equations`] to linearise the SO(3) log
/// residual of the Forster IMU factor.
///
/// Identity: `Jr_inv(φ) = Jl_inv(−φ)`, which means relative to the
/// `so3_left_jacobian_inverse` formula in [`visloc_core::geometry`]
/// only the sign of the linear `[φ]×` term flips (the quadratic
/// `[φ]×²` coefficient stays the same).
pub(super) fn right_jacobian_inverse_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = phi.norm_squared();
    let phi_skew = skew(phi);
    if theta_sq < 1e-10 {
        Matrix3::identity() + 0.5 * phi_skew + (1.0 / 12.0) * phi_skew * phi_skew
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5 * theta;
        let c = (1.0 - half_theta * half_theta.cos() / half_theta.sin()) / theta_sq;
        Matrix3::identity() + 0.5 * phi_skew + c * phi_skew * phi_skew
    }
}
