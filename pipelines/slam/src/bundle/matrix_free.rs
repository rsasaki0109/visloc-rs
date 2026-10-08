//! Matrix-free solver support: column equilibration, step-quality diagnostics, adaptive damping and the matrix-free step.

use super::*;

pub(super) const COLUMN_SCALING_MIN_DIAGONAL: f64 = 1.0e-6;
pub(super) const COLUMN_SCALING_MAX_DIAGONAL: f64 = 1.0e32;

#[derive(Debug)]
pub(super) struct ColumnEquilibrationState {
    pub(super) pose_transforms: Vec<Vector6<f64>>,
    pub(super) landmark_transforms: Vec<Vector3<f64>>,
    pub(super) stats: MatrixFreeBaColumnScalingIterationStats,
}

impl ColumnEquilibrationState {
    pub(super) fn unscale_deltas(
        &self,
        delta_poses: &mut DVector<f64>,
        delta_landmarks: &mut DVector<f64>,
    ) -> Result<(), &'static str> {
        let expected_pose = self.pose_transforms.len() * 6;
        let expected_landmark = self.landmark_transforms.len() * 3;
        if delta_poses.len() != expected_pose || delta_landmarks.len() != expected_landmark {
            return Err("scaled delta dimensions do not match the normal system");
        }
        for (pose, transform) in self.pose_transforms.iter().enumerate() {
            for component in 0..6 {
                let index = pose * 6 + component;
                let value = delta_poses[index] * transform[component];
                if !value.is_finite() {
                    return Err("unscaled pose delta is non-finite");
                }
                delta_poses[index] = value;
            }
        }
        for (landmark, transform) in self.landmark_transforms.iter().enumerate() {
            for component in 0..3 {
                let index = landmark * 3 + component;
                let value = delta_landmarks[index] * transform[component];
                if !value.is_finite() {
                    return Err("unscaled landmark delta is non-finite");
                }
                delta_landmarks[index] = value;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MatrixFreeCoordinateQuality {
    /// The undamped squared-cost quadratic prediction in this coordinate
    /// system: `-2 b·δ - δᵀHδ`.  For a scaled solve this is computed from the
    /// rounded scaled normal system and is not an independent reconstruction
    /// of the pre-scaling normal system.
    pub(super) predicted_undamped_squared_decrease: f64,
    /// Normwise backward error in the coordinate system represented by this
    /// report.  The denominator is `||A||F ||δ||2 + ||b||2` for the damped
    /// full pose+landmark system.
    pub(super) normwise_backward_error: f64,
    /// Componentwise backward error `max_i |r_i| / (|A||δ|+|b|)_i`.
    pub(super) componentwise_backward_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct MatrixFreeStepQuality {
    scaled_coordinates: bool,
    coordinate: Option<MatrixFreeCoordinateQuality>,
    physical_equivalent: Option<MatrixFreeCoordinateQuality>,
    coordinate_failure: Option<&'static str>,
    physical_equivalent_failure: Option<&'static str>,
}

#[derive(Clone, Copy)]
pub(super) enum MatrixFreeQualityCoordinate<'a> {
    /// The coordinates currently stored in `system` and supplied to the
    /// matrix-free solve.  This is physical for legacy MF and scaled for the
    /// column-equilibrated path.
    Current,
    /// The physical-equivalent rounded system obtained from a scaled system
    /// by applying `T⁻¹` to rows and columns.  This is intentionally not
    /// called the original normal system: the in-place scaling arithmetic has
    /// already rounded its coefficients.
    PhysicalEquivalent(&'a ColumnEquilibrationState),
}

fn quality_add(total: &mut f64, value: f64) -> Result<(), &'static str> {
    if !value.is_finite() {
        return Err("quality diagnostic encountered a non-finite term");
    }
    *total += value;
    if total.is_finite() {
        Ok(())
    } else {
        Err("quality diagnostic accumulator overflowed")
    }
}

fn quality_add_square(total: &mut f64, value: f64) -> Result<(), &'static str> {
    if !value.is_finite() {
        return Err("quality diagnostic encountered a non-finite value");
    }
    let square = value * value;
    if !square.is_finite() {
        return Err("quality diagnostic square overflowed");
    }
    quality_add(total, square)
}

fn quality_fold_residual_row(
    residual_norm_squared: &mut f64,
    componentwise_backward_error: &mut f64,
    residual: f64,
    denominator: f64,
) -> Result<(), &'static str> {
    if !residual.is_finite() || !denominator.is_finite() {
        return Err("quality diagnostic residual or denominator is non-finite");
    }
    quality_add_square(residual_norm_squared, residual)?;
    if denominator == 0.0 {
        if residual != 0.0 {
            return Err("quality diagnostic has nonzero residual over zero denominator");
        }
        // Zero-over-zero rows are exact zero rows (commonly fixed rotations)
        // and contribute zero to componentwise eta.
    } else {
        let ratio = residual.abs() / denominator;
        if !ratio.is_finite() {
            return Err("quality diagnostic componentwise ratio is non-finite");
        }
        *componentwise_backward_error = (*componentwise_backward_error).max(ratio);
    }
    Ok(())
}

fn quality_pose_factors(
    coordinate: MatrixFreeQualityCoordinate<'_>,
    pose: usize,
    component: usize,
) -> Result<(f64, f64), &'static str> {
    match coordinate {
        MatrixFreeQualityCoordinate::Current => Ok((1.0, 1.0)),
        MatrixFreeQualityCoordinate::PhysicalEquivalent(state) => {
            if component >= 6 {
                return Err("quality diagnostic pose transform component is invalid");
            }
            let transform = state
                .pose_transforms
                .get(pose)
                .ok_or("quality diagnostic pose transform index is invalid")?[component];
            let inverse = transform.recip();
            if !transform.is_finite() || !inverse.is_finite() {
                return Err("quality diagnostic pose transform is non-finite");
            }
            // First factor scales rows/columns of the current system.  The
            // second factor maps the current delta into the target system.
            Ok((inverse, transform))
        }
    }
}

fn quality_landmark_factors(
    coordinate: MatrixFreeQualityCoordinate<'_>,
    landmark: usize,
    component: usize,
) -> Result<(f64, f64), &'static str> {
    match coordinate {
        MatrixFreeQualityCoordinate::Current => Ok((1.0, 1.0)),
        MatrixFreeQualityCoordinate::PhysicalEquivalent(state) => {
            if component >= 3 {
                return Err("quality diagnostic landmark transform component is invalid");
            }
            let transform = state
                .landmark_transforms
                .get(landmark)
                .ok_or("quality diagnostic landmark transform index is invalid")?[component];
            let inverse = transform.recip();
            if !transform.is_finite() || !inverse.is_finite() {
                return Err("quality diagnostic landmark transform is non-finite");
            }
            Ok((inverse, transform))
        }
    }
}

pub(super) fn quality_coordinate_metrics(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    coordinate: MatrixFreeQualityCoordinate<'_>,
) -> Result<MatrixFreeCoordinateQuality, &'static str> {
    if !lambda.is_finite() {
        return Err("quality diagnostic lambda is non-finite");
    }
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        return Err("quality diagnostic requires pose-diagonal normal equations");
    };
    let pose_count = pose_blocks.len();
    let landmark_count = system.landmarks.len();
    if system.b_p.len() != pose_count * 6
        || delta_poses.len() != pose_count * 6
        || delta_landmarks.len() != landmark_count * 3
    {
        return Err("quality diagnostic delta dimensions mismatch");
    }

    // These are O(P) scratch vectors; landmark scratch is streamed and cross
    // grouping is bounded by the maximum track length. Cross terms enter
    // these vectors after same-pose entries are coalesced per landmark, so the
    // componentwise denominator represents the actual matrix rather than a
    // sum of absolute values of duplicate stored entries.
    let mut pose_residual = vec![0.0; pose_count * 6];
    let mut pose_denominator = vec![0.0; pose_count * 6];
    let mut matrix_norm_squared = 0.0;
    let mut delta_norm_squared = 0.0;
    let mut rhs_norm_squared = 0.0;
    let mut gradient_dot = 0.0;
    let mut hessian_quadratic = 0.0;
    let mut residual_norm_squared = 0.0;
    let mut componentwise_backward_error: f64 = 0.0;

    for (pose, block) in pose_blocks.iter().enumerate() {
        for row in 0..6 {
            let (row_scale, row_delta_factor) = quality_pose_factors(coordinate, pose, row)?;
            let row_index = pose * 6 + row;
            let delta_row = delta_poses[row_index] * row_delta_factor;
            let rhs_row = system.b_p[row_index] * row_scale;
            if !delta_row.is_finite() || !rhs_row.is_finite() {
                return Err("quality diagnostic pose delta or rhs is non-finite");
            }
            pose_residual[row_index] = rhs_row;
            pose_denominator[row_index] = rhs_row.abs();
            quality_add_square(&mut delta_norm_squared, delta_row)?;
            quality_add_square(&mut rhs_norm_squared, rhs_row)?;
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;

            for column in 0..6 {
                let (column_scale, column_delta_factor) =
                    quality_pose_factors(coordinate, pose, column)?;
                let column_index = pose * 6 + column;
                let delta_column = delta_poses[column_index] * column_delta_factor;
                let h = block[(row, column)] * row_scale * column_scale;
                let a = h + if row == column {
                    lambda * row_scale * column_scale
                } else {
                    0.0
                };
                if !delta_column.is_finite() || !h.is_finite() || !a.is_finite() {
                    return Err("quality diagnostic pose system is non-finite");
                }
                quality_add_square(&mut matrix_norm_squared, a)?;
                quality_add(&mut pose_residual[row_index], a * delta_column)?;
                quality_add(
                    &mut pose_denominator[row_index],
                    a.abs() * delta_column.abs(),
                )?;
                quality_add(&mut hessian_quadratic, delta_row * h * delta_column)?;
            }
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        // Landmark rows have no coupling to other landmarks, so keep their
        // residual/denominator as a single-landmark scratch array and fold
        // them into the scalar metrics before moving to the next landmark.
        let mut landmark_residual = [0.0; 3];
        let mut landmark_denominator = [0.0; 3];
        for row in 0..3 {
            let (row_scale, row_delta_factor) =
                quality_landmark_factors(coordinate, landmark_index, row)?;
            let row_index = landmark_index * 3 + row;
            let delta_row = delta_landmarks[row_index] * row_delta_factor;
            let rhs_row = landmark.b_l[row] * row_scale;
            if !delta_row.is_finite() || !rhs_row.is_finite() {
                return Err("quality diagnostic landmark delta or rhs is non-finite");
            }
            landmark_residual[row] = rhs_row;
            landmark_denominator[row] = rhs_row.abs();
            quality_add_square(&mut delta_norm_squared, delta_row)?;
            quality_add_square(&mut rhs_norm_squared, rhs_row)?;
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;

            for column in 0..3 {
                let (column_scale, column_delta_factor) =
                    quality_landmark_factors(coordinate, landmark_index, column)?;
                let column_index = landmark_index * 3 + column;
                let delta_column = delta_landmarks[column_index] * column_delta_factor;
                let h = landmark.h_ll[(row, column)] * row_scale * column_scale;
                let a = h + if row == column {
                    lambda * row_scale * column_scale
                } else {
                    0.0
                };
                if !delta_column.is_finite() || !h.is_finite() || !a.is_finite() {
                    return Err("quality diagnostic landmark system is non-finite");
                }
                quality_add_square(&mut matrix_norm_squared, a)?;
                quality_add(&mut landmark_residual[row], a * delta_column)?;
                quality_add(&mut landmark_denominator[row], a.abs() * delta_column.abs())?;
                quality_add(&mut hessian_quadratic, delta_row * h * delta_column)?;
            }
        }

        // The storage can contain multiple sensor contributions for the same
        // pose.  Coalesce each pose before taking absolute values or norms.
        let mut grouped_cross: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
        for (pose, cross) in &landmark.cross {
            if *pose >= pose_count {
                return Err("quality diagnostic cross pose index is invalid");
            }
            grouped_cross
                .entry(*pose)
                .and_modify(|accumulated| *accumulated += *cross)
                .or_insert(*cross);
        }
        for (pose, cross) in grouped_cross {
            for row in 0..6 {
                let (pose_scale, pose_delta_factor) = quality_pose_factors(coordinate, pose, row)?;
                let pose_index = pose * 6 + row;
                let pose_delta = delta_poses[pose_index] * pose_delta_factor;
                for column in 0..3 {
                    let (landmark_scale, landmark_delta_factor) =
                        quality_landmark_factors(coordinate, landmark_index, column)?;
                    let landmark_index_flat = landmark_index * 3 + column;
                    let landmark_delta =
                        delta_landmarks[landmark_index_flat] * landmark_delta_factor;
                    let g = cross[(row, column)] * pose_scale * landmark_scale;
                    if !pose_delta.is_finite() || !landmark_delta.is_finite() || !g.is_finite() {
                        return Err("quality diagnostic cross system is non-finite");
                    }
                    quality_add_square(&mut matrix_norm_squared, g)?;
                    quality_add_square(&mut matrix_norm_squared, g)?;
                    quality_add(&mut pose_residual[pose_index], g * landmark_delta)?;
                    quality_add(&mut landmark_residual[column], g * pose_delta)?;
                    quality_add(
                        &mut pose_denominator[pose_index],
                        g.abs() * landmark_delta.abs(),
                    )?;
                    quality_add(
                        &mut landmark_denominator[column],
                        g.abs() * pose_delta.abs(),
                    )?;
                    // The symmetric cross block appears twice in δᵀHδ.
                    quality_add(
                        &mut hessian_quadratic,
                        2.0 * pose_delta * g * landmark_delta,
                    )?;
                }
            }
        }

        for row in 0..3 {
            quality_fold_residual_row(
                &mut residual_norm_squared,
                &mut componentwise_backward_error,
                landmark_residual[row],
                landmark_denominator[row],
            )?;
        }
    }

    for (residual, denominator) in pose_residual.iter().zip(pose_denominator.iter()) {
        quality_fold_residual_row(
            &mut residual_norm_squared,
            &mut componentwise_backward_error,
            *residual,
            *denominator,
        )?;
    }
    let residual_norm = residual_norm_squared.sqrt();
    let matrix_norm = matrix_norm_squared.sqrt();
    let delta_norm = delta_norm_squared.sqrt();
    let rhs_norm = rhs_norm_squared.sqrt();
    let normwise_denominator = matrix_norm * delta_norm + rhs_norm;
    if !residual_norm.is_finite()
        || !matrix_norm.is_finite()
        || !delta_norm.is_finite()
        || !rhs_norm.is_finite()
        || !normwise_denominator.is_finite()
    {
        return Err("quality diagnostic normwise quantity is non-finite");
    }
    let normwise_backward_error = if normwise_denominator == 0.0 {
        return Err("quality diagnostic normwise denominator is zero");
    } else {
        residual_norm / normwise_denominator
    };
    let predicted_undamped_squared_decrease = -2.0 * gradient_dot - hessian_quadratic;
    if !predicted_undamped_squared_decrease.is_finite()
        || !normwise_backward_error.is_finite()
        || !componentwise_backward_error.is_finite()
    {
        return Err("quality diagnostic result is non-finite");
    }
    Ok(MatrixFreeCoordinateQuality {
        predicted_undamped_squared_decrease,
        normwise_backward_error,
        componentwise_backward_error,
    })
}

/// Compute only the scalar undamped quadratic prediction needed by the
/// adaptive damping policy.  This intentionally does not allocate the
/// residual/denominator scratch used by the opt-in backward-error diagnostic.
/// Same-pose cross blocks are coalesced in insertion order per landmark so
/// duplicate rig-sensor contributions have one bounded local representation.
pub(super) fn matrix_free_undamped_prediction(
    system: &NormalEquationsBa,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
) -> Result<f64, &'static str> {
    let CameraHessian::PoseDiagonal(pose_blocks) = &system.h_pp else {
        return Err("adaptive prediction requires pose-diagonal normal equations");
    };
    if system.b_p.len() != pose_blocks.len() * 6
        || delta_poses.len() != pose_blocks.len() * 6
        || delta_landmarks.len() != system.landmarks.len() * 3
    {
        return Err("adaptive prediction delta dimensions mismatch");
    }

    let mut gradient_dot = 0.0;
    let mut hessian_quadratic = 0.0;
    for (pose, block) in pose_blocks.iter().enumerate() {
        for row in 0..6 {
            let row_index = pose * 6 + row;
            let delta_row = delta_poses[row_index];
            let rhs_row = system.b_p[row_index];
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;
            for column in 0..6 {
                let delta_column = delta_poses[pose * 6 + column];
                quality_add(
                    &mut hessian_quadratic,
                    delta_row * block[(row, column)] * delta_column,
                )?;
            }
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter().enumerate() {
        for row in 0..3 {
            let row_index = landmark_index * 3 + row;
            let delta_row = delta_landmarks[row_index];
            let rhs_row = landmark.b_l[row];
            quality_add(&mut gradient_dot, rhs_row * delta_row)?;
            for column in 0..3 {
                let delta_column = delta_landmarks[landmark_index * 3 + column];
                quality_add(
                    &mut hessian_quadratic,
                    delta_row * landmark.h_ll[(row, column)] * delta_column,
                )?;
            }
        }

        let mut grouped_cross: BTreeMap<usize, Matrix6x3<f64>> = BTreeMap::new();
        for (pose, cross) in &landmark.cross {
            if *pose >= pose_blocks.len() {
                return Err("adaptive prediction cross pose index is invalid");
            }
            if let Some(accumulated) = grouped_cross.get_mut(pose) {
                *accumulated += *cross;
                if !accumulated.iter().all(|value| value.is_finite()) {
                    return Err("adaptive prediction cross accumulation is non-finite");
                }
            } else {
                grouped_cross.insert(*pose, *cross);
            }
        }
        for (pose, cross) in grouped_cross {
            for row in 0..6 {
                let pose_delta = delta_poses[pose * 6 + row];
                for column in 0..3 {
                    let landmark_delta = delta_landmarks[landmark_index * 3 + column];
                    // The symmetric pose/landmark block occurs twice in
                    // delta^T H delta.
                    quality_add(
                        &mut hessian_quadratic,
                        2.0 * pose_delta * cross[(row, column)] * landmark_delta,
                    )?;
                }
            }
        }
    }

    let prediction = -2.0 * gradient_dot - hessian_quadratic;
    if prediction.is_finite() {
        Ok(prediction)
    } else {
        Err("adaptive prediction is non-finite")
    }
}

fn matrix_free_step_quality(
    system: &NormalEquationsBa,
    lambda: f64,
    delta_poses: &DVector<f64>,
    delta_landmarks: &DVector<f64>,
    scaling_state: Option<&ColumnEquilibrationState>,
) -> MatrixFreeStepQuality {
    let scaled_coordinates = scaling_state.is_some();
    let coordinate_result = quality_coordinate_metrics(
        system,
        lambda,
        delta_poses,
        delta_landmarks,
        MatrixFreeQualityCoordinate::Current,
    );
    let coordinate = coordinate_result.as_ref().ok().copied();
    let coordinate_failure = coordinate_result.as_ref().err().copied();
    let physical_result = scaling_state.map(|state| {
        quality_coordinate_metrics(
            system,
            lambda,
            delta_poses,
            delta_landmarks,
            MatrixFreeQualityCoordinate::PhysicalEquivalent(state),
        )
    });
    let physical_equivalent = match physical_result.as_ref() {
        Some(result) => result.as_ref().ok().copied(),
        None => coordinate,
    };
    let physical_equivalent_failure = physical_result
        .as_ref()
        .and_then(|result| result.as_ref().err().copied())
        .or_else(|| {
            if scaling_state.is_none() {
                coordinate_failure
            } else {
                None
            }
        });
    MatrixFreeStepQuality {
        scaled_coordinates,
        coordinate,
        physical_equivalent,
        coordinate_failure,
        physical_equivalent_failure,
    }
}

pub(super) fn matrix_free_quality_actual_and_rho(
    cost_before: f64,
    cost_after: f64,
    coordinate_prediction: Option<f64>,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
) -> (Option<f64>, Option<f64>, &'static str) {
    let actual_cost_decrease = if cost_before.is_finite() && cost_after.is_finite() {
        let decrease = cost_before - cost_after;
        decrease.is_finite().then_some(decrease)
    } else {
        None
    };
    let (rho, rho_reason) = if nonprojectable_before != 0 || nonprojectable_after != 0 {
        (None, "nonprojectable_count_nonzero")
    } else if actual_cost_decrease.is_none() {
        (None, "actual_cost_decrease_nonfinite")
    } else {
        match coordinate_prediction {
            Some(prediction) if prediction.is_finite() && prediction > 0.0 => {
                let rho = actual_cost_decrease.expect("finite actual decrease") / prediction;
                if rho.is_finite() {
                    (Some(rho), "defined")
                } else {
                    (None, "rho_nonfinite")
                }
            }
            Some(_) => (None, "prediction_nonpositive_or_nonfinite"),
            None => (None, "prediction_unavailable"),
        }
    };
    (actual_cost_decrease, rho, rho_reason)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn emit_matrix_free_step_quality(
    iteration: usize,
    solve_lambda: f64,
    quality: MatrixFreeStepQuality,
    cost_before: f64,
    cost_after: f64,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
    cost_gate: bool,
    feasibility_gate: bool,
    step_accepted: bool,
) {
    let coordinate_prediction = quality
        .coordinate
        .map(|metrics| metrics.predicted_undamped_squared_decrease);
    let physical_prediction = quality
        .physical_equivalent
        .map(|metrics| metrics.predicted_undamped_squared_decrease);
    let (actual_cost_decrease, rho, rho_reason) = matrix_free_quality_actual_and_rho(
        cost_before,
        cost_after,
        coordinate_prediction,
        nonprojectable_before,
        nonprojectable_after,
    );
    let coordinate_name = if quality.scaled_coordinates {
        "scaled"
    } else {
        "physical"
    };
    let physical_equivalent_evaluation = if quality.scaled_coordinates {
        "recomputed_from_rounded_scaled_coefficients"
    } else {
        "current_physical_system"
    };
    let coordinate_normwise = quality
        .coordinate
        .map(|metrics| metrics.normwise_backward_error);
    let coordinate_componentwise = quality
        .coordinate
        .map(|metrics| metrics.componentwise_backward_error);
    let physical_normwise = quality
        .physical_equivalent
        .map(|metrics| metrics.normwise_backward_error);
    let physical_componentwise = quality
        .physical_equivalent
        .map(|metrics| metrics.componentwise_backward_error);
    eprintln!(
        concat!(
            "sfm-debug-ba-lm-quality: iteration={} solve_lambda={:.17e} ",
            "coordinates={} prediction_coordinates={} ",
            "physical_equivalent_evaluation={} ",
            "predicted_undamped_squared_decrease={:?} ",
            "physical_equivalent_predicted_undamped_squared_decrease={:?} ",
            "actual_cost_decrease={:?} rho={:?} rho_reason={} ",
            "rho_prediction_coordinates={} actual_cost_scope=existing_ba_cost ",
            "rho_cost_comparable={} nonprojectable_before={} nonprojectable_after={} ",
            "cost_gate={} feasibility_gate={} accepted={} ",
            "coordinate_full_normwise_backward_error={:?} ",
            "coordinate_full_componentwise_eta={:?} ",
            "physical_equivalent_full_normwise_backward_error={:?} ",
            "physical_equivalent_full_componentwise_eta={:?} ",
            "coordinate_quality_failure={:?} physical_equivalent_quality_failure={:?}"
        ),
        iteration,
        solve_lambda,
        coordinate_name,
        coordinate_name,
        physical_equivalent_evaluation,
        coordinate_prediction,
        physical_prediction,
        actual_cost_decrease,
        rho,
        rho_reason,
        coordinate_name,
        nonprojectable_before == 0 && nonprojectable_after == 0,
        nonprojectable_before,
        nonprojectable_after,
        cost_gate,
        feasibility_gate,
        step_accepted,
        coordinate_normwise,
        coordinate_componentwise,
        physical_normwise,
        physical_componentwise,
        quality.coordinate_failure,
        quality.physical_equivalent_failure,
    );
}

pub(super) fn emit_matrix_free_step_quality_failure(
    iteration: usize,
    solve_lambda: f64,
    diagnostic: &str,
) {
    eprintln!(
        "sfm-debug-ba-lm-quality: iteration={} solve_lambda={:.17e} linear_failure={}",
        iteration, solve_lambda, diagnostic
    );
}

fn column_scaling_transform(
    diagonal: f64,
    minimum: &mut f64,
    maximum: &mut f64,
    clamped_to_minimum: &mut usize,
    clamped_to_maximum: &mut usize,
) -> Result<f64, &'static str> {
    if !diagonal.is_finite() {
        return Err("normal diagonal is non-finite");
    }
    if diagonal < 0.0 {
        return Err("normal diagonal is negative");
    }
    let clamped = diagonal.clamp(COLUMN_SCALING_MIN_DIAGONAL, COLUMN_SCALING_MAX_DIAGONAL);
    if clamped == COLUMN_SCALING_MIN_DIAGONAL && diagonal < COLUMN_SCALING_MIN_DIAGONAL {
        *clamped_to_minimum += 1;
    }
    if clamped == COLUMN_SCALING_MAX_DIAGONAL && diagonal > COLUMN_SCALING_MAX_DIAGONAL {
        *clamped_to_maximum += 1;
    }
    *minimum = minimum.min(clamped);
    *maximum = maximum.max(clamped);
    let transform = clamped.sqrt().recip();
    if !transform.is_finite() {
        return Err("column scaling transform is non-finite");
    }
    Ok(transform)
}

pub(super) fn column_equilibrate_normal_system(
    system: &mut NormalEquationsBa,
) -> Result<ColumnEquilibrationState, &'static str> {
    let pose_transforms = match &system.h_pp {
        CameraHessian::PoseDiagonal(blocks) => {
            if blocks.is_empty() || system.b_p.len() != blocks.len() * 6 {
                return Err("pose diagonal dimensions are invalid");
            }
            let mut transforms = Vec::with_capacity(blocks.len());
            for block in blocks {
                let mut transform = Vector6::zeros();
                for component in 0..6 {
                    // The actual aggregate statistics are computed below in
                    // one pass over both pose and landmark columns.
                    transform[component] = block[(component, component)];
                }
                transforms.push(transform);
            }
            transforms
        }
        CameraHessian::Dense(_) => {
            return Err("column scaling requires pose-diagonal normal equations");
        }
    };

    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0;
    let mut clamped_to_minimum = 0;
    let mut clamped_to_maximum = 0;
    let pose_transforms = pose_transforms
        .into_iter()
        .map(|diagonal| {
            let mut transform = Vector6::zeros();
            for component in 0..6 {
                transform[component] = column_scaling_transform(
                    diagonal[component],
                    &mut minimum,
                    &mut maximum,
                    &mut clamped_to_minimum,
                    &mut clamped_to_maximum,
                )?;
            }
            Ok(transform)
        })
        .collect::<Result<Vec<_>, &'static str>>()?;

    let mut landmark_transforms = Vec::with_capacity(system.landmarks.len());
    for landmark in &system.landmarks {
        let mut transform = Vector3::zeros();
        for component in 0..3 {
            transform[component] = column_scaling_transform(
                landmark.h_ll[(component, component)],
                &mut minimum,
                &mut maximum,
                &mut clamped_to_minimum,
                &mut clamped_to_maximum,
            )?;
        }
        landmark_transforms.push(transform);
    }

    let scale_block6 = |block: &mut Matrix6<f64>, transform: &Vector6<f64>| {
        for row in 0..6 {
            for column in 0..6 {
                let value = block[(row, column)] * transform[row] * transform[column];
                if !value.is_finite() {
                    return Err("scaled pose Hessian is non-finite");
                }
                block[(row, column)] = value;
            }
        }
        Ok(())
    };
    let CameraHessian::PoseDiagonal(blocks) = &mut system.h_pp else {
        unreachable!("pose diagonal was checked above");
    };
    for (pose, block) in blocks.iter_mut().enumerate() {
        scale_block6(block, &pose_transforms[pose])?;
        for (component, transform) in pose_transforms[pose].iter().enumerate() {
            let index = pose * 6 + component;
            let value = system.b_p[index] * transform;
            if !value.is_finite() {
                return Err("scaled pose gradient is non-finite");
            }
            system.b_p[index] = value;
        }
    }

    for (landmark_index, landmark) in system.landmarks.iter_mut().enumerate() {
        let transform = landmark_transforms[landmark_index];
        for row in 0..3 {
            for column in 0..3 {
                let value = landmark.h_ll[(row, column)] * transform[row] * transform[column];
                if !value.is_finite() {
                    return Err("scaled landmark Hessian is non-finite");
                }
                landmark.h_ll[(row, column)] = value;
            }
            let value = landmark.b_l[row] * transform[row];
            if !value.is_finite() {
                return Err("scaled landmark gradient is non-finite");
            }
            landmark.b_l[row] = value;
        }
        for (pose, cross) in &mut landmark.cross {
            let Some(pose_transform) = pose_transforms.get(*pose) else {
                return Err("landmark cross pose index is invalid");
            };
            for row in 0..6 {
                for column in 0..3 {
                    let value = cross[(row, column)] * pose_transform[row] * transform[column];
                    if !value.is_finite() {
                        return Err("scaled landmark cross is non-finite");
                    }
                    cross[(row, column)] = value;
                }
            }
        }
    }

    Ok(ColumnEquilibrationState {
        pose_transforms,
        landmark_transforms,
        stats: MatrixFreeBaColumnScalingIterationStats {
            iteration: 0,
            minimum_diagonal: minimum,
            maximum_diagonal: maximum,
            clamped_to_minimum,
            clamped_to_maximum,
        },
    })
}

/// Internal dispatch state for the shared LM loop.  The legacy variant is
/// deliberately the default path; the matrix-free variant is only installed
/// by [`BundleAdjustment::optimize_matrix_free`].
pub(super) enum BaSolveBackend {
    Legacy,
    MatrixFree(MatrixFreeRuntime),
    MatrixFreeColumnScaled(MatrixFreeRuntime),
}

pub(super) struct MatrixFreeRuntime {
    pub(super) landmark_qr: bool,
    pub(super) cluster8: bool,
    pub(super) options: MatrixFreeBaOptions,
    pub(super) iterations: Vec<MatrixFreeBaIterationStats>,
    pub(super) restart_limit: usize,
    pub(super) restart_iterations: Option<Vec<MatrixFreeBaRestartIterationStats>>,
    pub(super) column_scaling_iterations: Option<Vec<MatrixFreeBaColumnScalingIterationStats>>,
    pub(super) adaptive_damping: bool,
    pub(super) adaptive_iterations: Option<Vec<MatrixFreeBaAdaptiveDampingIterationStats>>,
    pub(super) failure: Option<MatrixFreeBaError>,
}

impl MatrixFreeRuntime {
    pub(super) const fn new(options: MatrixFreeBaOptions) -> Self {
        Self {
            landmark_qr: false,
            cluster8: false,
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: None,
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    pub(super) const fn with_restart(options: MatrixFreeBaOptions, restart_limit: usize) -> Self {
        Self {
            landmark_qr: false,
            cluster8: false,
            options,
            iterations: Vec::new(),
            restart_limit,
            restart_iterations: Some(Vec::new()),
            column_scaling_iterations: None,
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    pub(super) const fn with_column_scaling(options: MatrixFreeBaOptions) -> Self {
        Self {
            landmark_qr: false,
            cluster8: false,
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: Some(Vec::new()),
            adaptive_damping: false,
            adaptive_iterations: None,
            failure: None,
        }
    }

    pub(super) const fn with_column_scaling_adaptive(options: MatrixFreeBaOptions) -> Self {
        Self {
            landmark_qr: false,
            cluster8: false,
            options,
            iterations: Vec::new(),
            restart_limit: 0,
            restart_iterations: None,
            column_scaling_iterations: Some(Vec::new()),
            adaptive_damping: true,
            adaptive_iterations: Some(Vec::new()),
            failure: None,
        }
    }
}

const ADAPTIVE_ACCEPT_MIN_LAMBDA_FACTOR: f64 = 1.0 / 3.0;

pub(super) fn bounded_adaptive_lambda(
    lambda: f64,
    multiplier: f64,
    min_lambda: f64,
    max_lambda: f64,
) -> f64 {
    let product = lambda * multiplier;
    if !product.is_finite() {
        max_lambda
    } else {
        product.clamp(min_lambda, max_lambda)
    }
}

pub(super) fn adaptive_accepted_lambda(
    lambda: f64,
    rho: f64,
    min_lambda: f64,
    max_lambda: f64,
) -> Result<f64, &'static str> {
    if !rho.is_finite() || rho <= 0.0 {
        return Err("adaptive rho is non-finite or non-positive");
    }
    // Clamp before cubing so a very large finite rho cannot overflow the
    // policy expression.  This is a fixed policy, not a user-tunable sweep.
    let centered = if rho >= 1.0 {
        1.0
    } else {
        (2.0 * rho - 1.0).max(-1.0)
    };
    let multiplier = (1.0 - centered * centered * centered).max(ADAPTIVE_ACCEPT_MIN_LAMBDA_FACTOR);
    if !multiplier.is_finite() {
        return Err("adaptive lambda multiplier is non-finite");
    }
    Ok(bounded_adaptive_lambda(
        lambda, multiplier, min_lambda, max_lambda,
    ))
}

#[derive(Debug)]
pub(super) struct AdaptiveStepDecision {
    pub(super) prediction: Option<f64>,
    pub(super) actual_cost_decrease: Option<f64>,
    pub(super) rho: Option<f64>,
    pub(super) cost_gate: bool,
    pub(super) feasibility_gate: bool,
    pub(super) accepted: bool,
    pub(super) reason: String,
}

pub(super) fn adaptive_step_decision(
    prediction: Option<Result<f64, &'static str>>,
    cost_before: f64,
    cost_after: f64,
    nonprojectable_before: usize,
    nonprojectable_after: usize,
    cost_gate: bool,
    feasibility_gate: bool,
) -> AdaptiveStepDecision {
    let prediction_value = prediction.as_ref().and_then(|value| match value {
        Ok(value) if value.is_finite() => Some(*value),
        _ => None,
    });
    let prediction_error = match prediction.as_ref() {
        Some(Err(error)) => Some(*error),
        Some(Ok(value)) if !value.is_finite() => Some("adaptive prediction is non-finite"),
        _ => None,
    };
    let (actual_cost_decrease, rho, rho_reason) = matrix_free_quality_actual_and_rho(
        cost_before,
        cost_after,
        prediction_value,
        nonprojectable_before,
        nonprojectable_after,
    );
    let rho_positive = rho.is_some_and(|value| value > 0.0);
    let accepted = cost_gate && feasibility_gate && rho_positive;
    let reason = if accepted {
        "accepted".to_owned()
    } else if !cost_gate && !feasibility_gate {
        "candidate_rejected_cost_and_feasibility_gate".to_owned()
    } else if !cost_gate {
        "candidate_rejected_cost_gate".to_owned()
    } else if !feasibility_gate {
        "candidate_rejected_feasibility_gate".to_owned()
    } else if let Some(error) = prediction_error {
        format!("candidate_rejected_prediction:{error}")
    } else if rho.is_some() && !rho_positive {
        "candidate_rejected_rho_nonpositive".to_owned()
    } else {
        format!("candidate_rejected_rho:{rho_reason}")
    };
    AdaptiveStepDecision {
        prediction: prediction_value,
        actual_cost_decrease,
        rho,
        cost_gate,
        feasibility_gate,
        accepted,
        reason,
    }
}

/// Project a normal-equation system onto the subspace in which selected pose
/// rotations are fixed.  Pose blocks intentionally remain six-dimensional so
/// the existing Schur and linear-solver layouts are unchanged.  The
/// constrained rotation rows/columns are made identity rows with zero right
/// hand side; this is equivalent to removing those variables and is also
/// well-defined for an undamped Gauss--Newton solve.  Landmark cross blocks
/// are cleared for the same rows so the translation/landmark solve cannot
/// use a discarded rotation update.
pub(super) fn constrain_fixed_pose_rotations(
    fixed_rotations: &BTreeSet<u64>,
    pose_index: &BTreeMap<u64, usize>,
    system: &mut NormalEquationsBa,
) {
    for &image_id in fixed_rotations {
        let Some(&pose_slot) = pose_index.get(&image_id) else {
            continue;
        };
        for component in 3..6 {
            let index = pose_slot * 6 + component;
            match &mut system.h_pp {
                CameraHessian::Dense(matrix) => {
                    for column in 0..matrix.ncols() {
                        matrix[(index, column)] = 0.0;
                    }
                    for row in 0..matrix.nrows() {
                        matrix[(row, index)] = 0.0;
                    }
                    matrix[(index, index)] = 1.0;
                }
                CameraHessian::PoseDiagonal(blocks) => {
                    let local = index % 6;
                    for column in 0..6 {
                        blocks[pose_slot][(local, column)] = 0.0;
                    }
                    for row in 0..6 {
                        blocks[pose_slot][(row, local)] = 0.0;
                    }
                    blocks[pose_slot][(local, local)] = 1.0;
                }
            }
            system.b_p[index] = 0.0;
            for landmark in &mut system.landmarks {
                for (landmark_pose_slot, cross) in &mut landmark.cross {
                    if *landmark_pose_slot == pose_slot {
                        for column in 0..3 {
                            cross[(component, column)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

pub(super) struct MatrixFreeStepOutcome {
    pub(super) delta_poses: DVector<f64>,
    pub(super) delta_landmarks: DVector<f64>,
    pub(super) diagnostics: MatrixFreeBaIterationStats,
    pub(super) restart_diagnostics: Option<MatrixFreeBaRestartIterationStats>,
    pub(super) adaptive_prediction: Option<Result<f64, &'static str>>,
    pub(super) quality: Option<MatrixFreeStepQuality>,
}

pub(super) struct MatrixFreeStepError {
    pub(super) diagnostic: String,
    pub(super) pcg_iterations: Option<usize>,
    pub(super) pcg_residual_norm: Option<f64>,
    pub(super) pcg_target: Option<f64>,
    pub(super) restart_diagnostics: Option<MatrixFreeBaRestartIterationStats>,
}

// This private error carries the optional per-solve restart diagnostic along
// with the existing textual PCG failure.  Keeping it inline avoids an
// allocation on the successful/default path; the large-error lint is not an
// API concern here.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(super) fn solve_matrix_free_step(
    system: &NormalEquationsBa,
    lambda: f64,
    options: MatrixFreeBaOptions,
    restart_limit: usize,
    collect_restart_diagnostics: bool,
    scaling_state: Option<&ColumnEquilibrationState>,
    schur_debug_context: Option<SchurBlockDebugContext<'_>>,
    collect_adaptive_prediction: bool,
    cluster8: bool,
) -> Result<MatrixFreeStepOutcome, MatrixFreeStepError> {
    let to_step_error = |error: implicit_schur::ImplicitSchurError| {
        let (pcg_iterations, pcg_residual_norm, pcg_target) = error.diagnostics();
        MatrixFreeStepError {
            diagnostic: format!("{error:?}"),
            pcg_iterations,
            pcg_residual_norm,
            pcg_target,
            restart_diagnostics: collect_restart_diagnostics.then(|| {
                MatrixFreeBaRestartIterationStats {
                    iteration: 0,
                    pcg_iterations,
                    true_residual_rechecks: 0,
                    failed_true_residual_rechecks: 0,
                    restarts: 0,
                    terminal_failure: Some(format!("{error:?}")),
                }
            }),
        }
    };
    let operator = match schur_debug_context {
        Some(context) => {
            implicit_schur::ImplicitSchurOperator::new_with_debug(system, lambda, Some(context))
        }
        None => implicit_schur::ImplicitSchurOperator::new(system, lambda),
    }
    .map_err(to_step_error)?;
    let operator = if cluster8 {
        operator.with_cluster8().map_err(to_step_error)?
    } else {
        operator
    };
    let pcg_options = implicit_schur::PcgOptions {
        max_iterations: options.max_pcg_iterations,
        relative_tolerance: options.pcg_relative_tolerance,
        absolute_tolerance: options.pcg_absolute_tolerance,
    };
    let (pcg, restart_diagnostics) = if restart_limit == 0 && !collect_restart_diagnostics {
        (
            operator
                .solve_pcg(operator.rhs(), pcg_options)
                .map_err(to_step_error)?,
            None,
        )
    } else {
        let run = operator
            .solve_pcg_with_restart(operator.rhs(), pcg_options, restart_limit)
            .map_err(|failure| {
                let implicit_schur::PcgSolveFailure { error, diagnostics } = failure;
                let (pcg_iterations, pcg_residual_norm, pcg_target) = error.diagnostics();
                MatrixFreeStepError {
                    diagnostic: format!("{error:?}"),
                    pcg_iterations,
                    pcg_residual_norm,
                    pcg_target,
                    restart_diagnostics: Some(MatrixFreeBaRestartIterationStats {
                        iteration: 0,
                        pcg_iterations: diagnostics.pcg_iterations,
                        true_residual_rechecks: diagnostics.true_residual_rechecks,
                        failed_true_residual_rechecks: diagnostics.failed_true_residual_rechecks,
                        restarts: diagnostics.restarts,
                        terminal_failure: Some(format!("{error:?}")),
                    }),
                }
            })?;
        (run.result, Some(run.diagnostics))
    };
    let mut delta_poses = pcg.solution;
    let mut delta_landmarks = operator.complete_delta(&delta_poses).map_err(|error| {
        if let Some(diagnostics) = restart_diagnostics {
            let mut restart_diagnostics: MatrixFreeBaRestartIterationStats = diagnostics.into();
            restart_diagnostics.terminal_failure = Some(format!("{error:?}"));
            MatrixFreeStepError {
                diagnostic: format!("{error:?}"),
                pcg_iterations: Some(pcg.iterations),
                pcg_residual_norm: Some(pcg.residual_norm),
                pcg_target: Some(pcg.target),
                restart_diagnostics: Some(restart_diagnostics),
            }
        } else {
            to_step_error(error)
        }
    })?;
    let adaptive_prediction = collect_adaptive_prediction
        .then(|| matrix_free_undamped_prediction(system, &delta_poses, &delta_landmarks));
    let quality = ba_lm_step_quality_debug_enabled().then(|| {
        matrix_free_step_quality(
            system,
            lambda,
            &delta_poses,
            &delta_landmarks,
            scaling_state,
        )
    });
    if let Some(scaling_state) = scaling_state {
        if let Err(error) = scaling_state.unscale_deltas(&mut delta_poses, &mut delta_landmarks) {
            let restart_diagnostics = restart_diagnostics.map(|diagnostics| {
                let mut stats: MatrixFreeBaRestartIterationStats = diagnostics.into();
                stats.terminal_failure = Some(format!("column scaling unscale failed: {error}"));
                stats
            });
            return Err(MatrixFreeStepError {
                diagnostic: format!("column scaling unscale failed: {error}"),
                pcg_iterations: Some(pcg.iterations),
                pcg_residual_norm: Some(pcg.residual_norm),
                pcg_target: Some(pcg.target),
                restart_diagnostics,
            });
        }
    }
    Ok(MatrixFreeStepOutcome {
        delta_poses,
        delta_landmarks,
        diagnostics: MatrixFreeBaIterationStats {
            iteration: 0,
            pcg_iterations: Some(pcg.iterations),
            pcg_residual_norm: Some(pcg.residual_norm),
            pcg_target: Some(pcg.target),
            pcg_failure: None,
        },
        restart_diagnostics: restart_diagnostics.map(Into::into),
        adaptive_prediction,
        quality,
    })
}
