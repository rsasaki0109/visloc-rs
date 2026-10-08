//! Landmark back-substitution and upstream trial projection helpers.

use super::*;

/// Recover a landmark increment from a compact Q1/R payload produced by one
/// [`LandmarkHouseholderF32::factor`] call.  The arithmetic intentionally
/// mirrors `back_substitute_landmark_f32_with_track`: the same row-major
/// GEMV, positive-Q1 RHS, and pinned 3x3 triangular schedule are used.  The
/// compact payload is consumed by the clean solver's one-shot preparation.
pub(crate) fn back_substitute_landmark_compact_f32(
    data: &CompactLandmarkBackSubstitutionF32,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_compact_flat_f32(
        data.landmark_cols,
        data.state_cols,
        &data.storage,
        data.rank,
        data.eligible,
        state_step,
        tolerance,
    )
}

pub(super) fn back_substitute_landmark_compact_entry_f32(
    data: &CompactLandmarkBackSubstitutionEntryF32,
    arena: &[f32],
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    let end = data.storage_offset.checked_add(data.storage_len()?)?;
    let storage = arena.get(data.storage_offset..end)?;
    back_substitute_compact_flat_f32(
        data.landmark_cols,
        data.state_cols,
        storage,
        data.rank,
        data.eligible,
        state_step,
        tolerance,
    )
}

fn back_substitute_compact_flat_f32(
    n: usize,
    state_cols: usize,
    storage: &[f32],
    rank: usize,
    eligible: bool,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    let expected_storage_len = checked_compact_storage_len(n, state_cols)?;
    if !eligible
        || n == 0
        || n > 3
        || rank < n
        || state_cols != state_step.len()
        || storage.len() != expected_storage_len
    {
        return None;
    }

    // This is the same Eigen row-major Packet8/Packet4/scalar association as
    // `eigen_row_major_gemv_f32`, expressed directly over the compact flat
    // storage.  Keeping the fixed three-lane result avoids materializing a
    // temporary DMatrix/DVector for every landmark while preserving every
    // f32 multiply/add and its order.
    let q1_state_step = eigen_compact_q1_state_gemv_f32(state_cols, n, storage, state_step);
    let mut rhs = [0.0_f32; 3];
    for row in 0..n {
        let residual_offset = n * state_cols;
        rhs[row] = storage[residual_offset + row] + q1_state_step[row];
    }
    let threshold = tolerance as f32;
    let mut increment = [0.0_f32; 3];
    let upper_r_offset = n * state_cols + n;
    if n == 3 {
        // Keep this operation order identical to the existing native-f32
        // recovery path.  In particular, row 1 uses one FMA and row 0 folds
        // r02*x2 into r01*x1 with an FMA before the final subtraction.
        let d2 = storage[upper_r_offset + 2 * n + 2];
        let d1 = storage[upper_r_offset + n + 1];
        let d0 = storage[upper_r_offset];
        if d2.abs() <= threshold || d1.abs() <= threshold || d0.abs() <= threshold {
            return None;
        }
        let x2 = rhs[2] / d2;
        let row1 = (-storage[upper_r_offset + n + 2]).mul_add(x2, rhs[1]);
        let x1 = row1 / d1;
        let row0_dot = storage[upper_r_offset + 2].mul_add(x2, storage[upper_r_offset + 1] * x1);
        let row0_numerator = rhs[0] - row0_dot;
        let x0 = row0_numerator / d0;
        increment[0] = -x0;
        increment[1] = -x1;
        increment[2] = -x2;
    } else {
        // Retain the small generic path used by synthetic one/two-column
        // tests; production visual blocks use three landmark columns.
        let mut solved = [0.0_f32; 3];
        for row in (0..n).rev() {
            let mut value = rhs[row];
            let mut row_product = 0.0_f32;
            for column in (row + 1)..n {
                row_product = add_f32_exact(
                    row_product,
                    storage[upper_r_offset + row * n + column] * solved[column],
                );
            }
            value -= row_product;
            let diagonal = storage[upper_r_offset + row * n + row];
            if diagonal.abs() <= threshold {
                return None;
            }
            solved[row] = value / diagonal;
        }
        for row in 0..n {
            increment[row] = -solved[row];
        }
    }
    Some(DVector::from_iterator(
        n,
        increment[..n].iter().copied().map(f64::from),
    ))
}

/// Flat-storage equivalent of [`eigen_row_major_gemv_f32`] for the compact
/// Q1 state rows.  The result is fixed-size because visual landmark blocks
/// have at most three columns; this keeps the per-factor recovery path free
/// of temporary nalgebra allocations.
#[inline]
fn eigen_compact_q1_state_gemv_f32(
    state_cols: usize,
    landmark_cols: usize,
    storage: &[f32],
    state_step: &DVector<f64>,
) -> [f32; 3] {
    let columns = state_cols;
    let full_end = columns / 8 * 8;
    let half_end = columns / 4 * 4;
    let mut result = [0.0_f32; 3];
    for row in 0..landmark_cols {
        let mut lanes = [0.0_f32; 8];
        let mut column = 0;
        while column < full_end {
            for lane in 0..8 {
                lanes[lane] = storage[row * state_cols + column + lane]
                    .mul_add(state_step[column + lane] as f32, lanes[lane]);
            }
            column += 8;
        }
        let q0 = lanes[0] + lanes[4];
        let q1 = lanes[1] + lanes[5];
        let q2 = lanes[2] + lanes[6];
        let q3 = lanes[3] + lanes[7];
        let mut value = (q0 + q2) + (q1 + q3);

        let mut half_lanes = [0.0_f32; 4];
        while column < half_end {
            for lane in 0..4 {
                half_lanes[lane] = storage[row * state_cols + column + lane]
                    .mul_add(state_step[column + lane] as f32, half_lanes[lane]);
            }
            column += 4;
        }
        let h0 = half_lanes[0] + half_lanes[2];
        let h1 = half_lanes[1] + half_lanes[3];
        value += h0 + h1;
        while column < columns {
            value += storage[row * state_cols + column] * state_step[column] as f32;
            column += 1;
        }
        result[row] = value;
    }
    result
}

fn back_substitute_landmark_f32(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_landmark_f32_with_track(data, state_step, tolerance, None)
}

pub(super) fn back_substitute_landmark_f32_with_track(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
    track_id: Option<u64>,
) -> Option<DVector<f64>> {
    let state = as_f32_matrix(&data.state_jacobian);
    let landmark = as_f32_matrix(&data.landmark_jacobian);
    let residual = as_f32_vector(&data.residual);
    let step = as_f32_vector(state_step);
    if data.rank < data.landmark_jacobian.ncols() || landmark.norm() <= tolerance as f32 {
        return None;
    }
    // Upstream transforms the complete row-major [Jp | Jl | r] storage once,
    // then forms Q1r + Q1Jp * pose_inc in source order.  Factoring only Jl
    // with a pre-combined rhs is algebraically equivalent but changes the f32
    // operation path and the resulting landmark increment.
    let qr = LandmarkHouseholderF32::factor(&state, &landmark, &residual)?;
    let transformed_state = qr.transformed_state();
    let transformed_rhs = qr.transformed_residual();
    let r = qr.upper_r();
    let n = landmark.ncols();
    // Keep the positive Q1 right-hand side until after the triangular solve.
    // Eigen evaluates `-Q1Jl.solve(rhs)` as a solve followed by a vector
    // negation.  Negating `rhs` first is algebraically equivalent, but it
    // changes the f32 operation path at each triangular update.
    let rhs = transformed_rhs.rows(0, n).into_owned();
    let q1_state_step = eigen_row_major_gemv_f32(&transformed_state.rows(0, n).into_owned(), &step);
    emit_landmark_backsub_probe(
        track_id,
        &step,
        &transformed_state.rows(0, n).into_owned(),
        &transformed_rhs.rows(0, n).into_owned(),
        &q1_state_step,
        &r,
    );
    let mut rhs = rhs;
    rhs += q1_state_step;
    let mut increment = DVector::<f32>::zeros(n);
    if n == 3 {
        // Pinned Eigen's 3x3 dynamic triangular kernel uses one FMA for the
        // row-1 update.  Row 0 first multiplies r01*x1, then folds r02*x2
        // into that product with an FMA, and only then subtracts the complete
        // dot product from the RHS.  The association is observable in the
        // f32 landmark increment, so do not turn this into two nested FMAs.
        let d2 = r[(2, 2)];
        let d1 = r[(1, 1)];
        let d0 = r[(0, 0)];
        if d2.abs() <= tolerance as f32
            || d1.abs() <= tolerance as f32
            || d0.abs() <= tolerance as f32
        {
            return None;
        }
        let x2 = rhs[2] / d2;
        let row1 = (-r[(1, 2)]).mul_add(x2, rhs[1]);
        let x1 = row1 / d1;
        let row0_dot = r[(0, 2)].mul_add(x2, r[(0, 1)] * x1);
        let row0_numerator = rhs[0] - row0_dot;
        let x0 = row0_numerator / d0;
        increment[0] = -x0;
        increment[1] = -x1;
        increment[2] = -x2;
    } else {
        // Keep the generic fallback for unit-test dimensions other than the
        // production 3x3 landmark block.
        let mut solved = DVector::<f32>::zeros(n);
        for row in (0..n).rev() {
            let mut value = rhs[row];
            let mut row_product = 0.0_f32;
            for column in (row + 1)..n {
                row_product = add_f32_exact(row_product, r[(row, column)] * solved[column]);
            }
            value -= row_product;
            let diagonal = r[(row, row)];
            if diagonal.abs() <= tolerance as f32 {
                return None;
            }
            solved[row] = value / diagonal;
        }
        increment = -solved;
    }
    Some(as_f64_vector(&increment))
}

pub(crate) fn back_substitute_landmark_upstream_f32_with_track(
    factor: &WhitenedFactorRowStack,
    state_step: &DVector<f64>,
    tolerance: f64,
    track_id: Option<u64>,
) -> Option<DVector<f64>> {
    let reduced = reduce_landmark_factors_f32_checked_without_visual_prefix_trace(
        std::slice::from_ref(factor),
        factor.state_jacobian.ncols(),
        tolerance,
    )
    .unwrap_or_else(|error| panic!("invalid f32 landmark reduction factors: {error:?}"));
    let data = reduced.back_substitution.first()?;
    back_substitute_landmark_f32_with_track(data, state_step, tolerance, track_id)
}

pub(crate) fn back_substitute_landmark_upstream_f32(
    factor: &WhitenedFactorRowStack,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    back_substitute_landmark_upstream_f32_with_track(factor, state_step, tolerance, None)
}

/// Recover a landmark increment after the reduced state increment is known.
pub fn back_substitute_landmark(
    data: &LandmarkBackSubstitution,
    state_step: &DVector<f64>,
    tolerance: f64,
) -> Option<DVector<f64>> {
    // The linearized residual convention is r + J_s * dx + J_l * dl = 0.
    // Solve the landmark block for the negative residual after the state
    // increment has been applied.
    let rhs = -(&data.residual + &data.state_jacobian * state_step);
    let landmark_factor = WhitenedFactorRowStack {
        state_jacobian: data.state_jacobian.clone(),
        landmark_jacobian: data.landmark_jacobian.clone(),
        residual: data.residual.clone(),
        objective_cost: 0.0,
        kind: FactorKind::Generic,
        imu_link_offsets: None,
        prior_state_columns: None,
        landmark_metadata: None,
        imu_input_diagnostic: None,
        visual_observation_ids: None,
    };
    let (augmented_state, augmented_landmark, augmented_residual) =
        augmented_landmark_rows(&landmark_factor);
    let mut augmented_rhs = DVector::zeros(augmented_residual.len());
    augmented_rhs.rows_mut(0, rhs.len()).copy_from(&rhs);
    let qr = augmented_landmark.qr();
    if data.rank < data.landmark_jacobian.ncols() {
        return None;
    }
    if data.landmark_jacobian.norm() <= tolerance {
        return None;
    }

    // `nalgebra::QR::solve` intentionally only accepts square systems.  The
    // upstream block solves the first three rows of the Householder-transformed
    // rectangular stack, i.e. R * dl = -(Qᵀ(r + Jp dx))[:3].  Reproduce that
    // triangular backsolve instead of switching to an SVD least-squares path.
    let mut transformed_rhs =
        DMatrix::from_column_slice(augmented_state.nrows(), 1, augmented_rhs.as_slice());
    qr.q_tr_mul(&mut transformed_rhs);
    let r = qr.r();
    let n = data.landmark_jacobian.ncols();
    let mut increment = DVector::zeros(n);
    for row in (0..n).rev() {
        let mut value = transformed_rhs[(row, 0)];
        for column in (row + 1)..n {
            value -= r[(row, column)] * increment[column];
        }
        let diagonal = r[(row, row)];
        if diagonal.abs() <= tolerance {
            return None;
        }
        increment[row] = value / diagonal;
    }
    Some(increment)
}

// Trial computeError's DS visitor uses a different expression schedule from
// the linearized factor objective. Keep this separate until trial wiring.
pub(crate) fn upstream_trial_bearing_f32(direction: Vector2<f32>) -> Vector3<f32> {
    let scale = 2.0_f32 / (direction.x.mul_add(direction.x, direction.y * direction.y) + 1.0_f32);
    Vector3::new(direction.x * scale, direction.y * scale, scale - 1.0_f32)
}

/// Coordinate arithmetic only: the trial evaluator must apply the native
/// camera-domain validity check separately. This does not accept/reject rows.
/// Do not substitute this schedule into the linearization/Jacobian path.
pub(crate) fn upstream_trial_projection_coordinates_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Vector2<f32> {
    let r2 = point.x * point.x + point.y * point.y;
    let d1 = point.z.mul_add(point.z, r2).sqrt();
    let k = (camera.xi as f32).mul_add(d1, point.z);
    let d2 = k.mul_add(k, r2).sqrt();
    let alpha = camera.alpha as f32;
    let denominator = alpha.mul_add(d2, (1.0_f32 - alpha) * k);
    Vector2::new(
        (camera.fx as f32).mul_add(point.x / denominator, camera.cx as f32),
        (camera.fy as f32).mul_add(point.y / denominator, camera.cy as f32),
    )
}

/// Native `linearizePoint` checks finite projected coordinates and the strict
/// DS domain inequality, without image bounds or an epsilon depth cutoff.
pub(crate) fn upstream_trial_project_f32(
    camera: &DoubleSphereCamera,
    point: Vector3<f32>,
) -> Option<Vector2<f32>> {
    let projected = upstream_trial_projection_coordinates_f32(camera, point);
    let r2 = point.x * point.x + point.y * point.y;
    let d1 = point.z.mul_add(point.z, r2).sqrt();
    let alpha = camera.alpha as f32;
    let xi = camera.xi as f32;
    let w1 = if alpha > 0.5_f32 {
        (1.0_f32 - alpha) / alpha
    } else {
        alpha / (1.0_f32 - alpha)
    };
    let w2 = (w1 + xi) / ((w1 + w1).mul_add(xi, xi * xi) + 1.0_f32).sqrt();
    if projected.iter().all(|value| value.is_finite()) && point.z > (-w2) * d1 {
        Some(projected)
    } else {
        None
    }
}

pub(crate) fn upstream_trial_visual_cost_f32(x: f32, y: f32, sigma: f32, huber: f32) -> f32 {
    let norm = (y.mul_add(y, x * x) + 0.0_f32).sqrt();
    let weight = if norm < huber { 1.0_f32 } else { huber / norm };
    let coefficient = (weight / (sigma * sigma)) * (0.5_f32 * (2.0_f32 - weight));
    (coefficient * y).mul_add(y, (coefficient * x) * x)
}

/// Trial computeError constructs one matrix per host/target TimeCamId pair.
/// Callers must supply getPose-equivalent current endpoints, not FEJ poses.
pub(crate) fn upstream_trial_transform_f32(
    host: &SE3,
    target: &SE3,
    host_camera: &SE3,
    target_camera: &SE3,
    same_time_cam_id: bool,
) -> SMatrix<f32, 4, 4> {
    if same_time_cam_id {
        return SMatrix::identity();
    }
    let result = upstream_trial_transform_stages_f32(
        F32Pose::from_se3(host),
        F32Pose::from_se3(target),
        F32Pose::from_se3(host_camera),
        F32Pose::from_se3(target_camera),
    )
    .4;
    eigen_homogeneous_transform_f32(result.rotation, result.translation)
}

/// Evaluate a trial observation without constructing linearization Jacobians.
/// Invalid projection is omitted by computeError; no outlier threshold is
/// applied to its cost. Parameters are native [stereographic u, v, rho].
pub(crate) fn upstream_trial_observation_f32(
    camera: &DoubleSphereCamera,
    transform: SMatrix<f32, 4, 4>,
    parameters: Vector3<f32>,
    pixel: Vector2<f32>,
    config: FactorConfig,
) -> Option<(Vector2<f32>, f32)> {
    let bearing = upstream_trial_bearing_f32(parameters.fixed_rows::<2>(0).into_owned());
    let point = eigen_homogeneous_point_product_f32(transform, bearing, parameters.z);
    let projected = upstream_trial_project_f32(camera, point.fixed_rows::<3>(0).into_owned())?;
    let raw = projected - pixel;
    let cost = upstream_trial_visual_cost_f32(
        raw.x,
        raw.y,
        config.observation_stddev as f32,
        config.huber_delta as f32,
    );
    Some((raw, cost))
}

/// The pinned native trial computeRelPose clone uses a packet difference
/// rotation before the generic camera actions. Keep this separate from the
/// linearization transform: their f32 instruction schedules are different.
pub(super) fn upstream_trial_transform_stages_f32(
    host: F32Pose,
    target: F32Pose,
    host_camera: F32Pose,
    target_camera: F32Pose,
) -> (F32Pose, UnitQuaternion<f32>, F32Pose, F32Pose, F32Pose) {
    let camera_inverse = target_camera.inverse();
    let inverse = sophus_so3_inverse(target.rotation);
    let relative = F32Pose {
        rotation: sophus_quat_product_current_packet_f32(inverse, host.rotation),
        translation: sophus_rotate_difference_f32(inverse, host.translation, target.translation),
    };
    let prefix = F32Pose {
        rotation: sophus_quat_product_camera_prefix_f32(camera_inverse.rotation, relative.rotation),
        translation: sophus_rotate_f32(camera_inverse.rotation, relative.translation)
            + camera_inverse.translation,
    };
    let result = F32Pose {
        rotation: sophus_quat_product_camera_suffix_f32(prefix.rotation, host_camera.rotation),
        translation: sophus_rotate_f32(prefix.rotation, host_camera.translation)
            + prefix.translation,
    };
    (camera_inverse, inverse, relative, prefix, result)
}
