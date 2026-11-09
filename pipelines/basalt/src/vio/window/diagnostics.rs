//! Window diagnostics: LM run summaries, initial / minimal
//! `WindowDiagnostics` and the opt-in detail trace.

use super::*;

pub(super) fn lm_run_diagnostics(pass: &str, result: &LmResult) -> LmRunDiagnostics {
    let accepted = result
        .trace
        .iter()
        .filter(|entry| matches!(entry.decision, super::aom::LmDecision::Accepted))
        .count();
    let rejected = result
        .trace
        .iter()
        .filter(|entry| matches!(entry.decision, super::aom::LmDecision::Rejected))
        .count();
    LmRunDiagnostics {
        pass: pass.to_owned(),
        iterations: result.iterations,
        lambda: result.lambda,
        initial_cost: result
            .trace
            .first()
            .map(|entry| entry.cost_before)
            .unwrap_or(result.cost),
        final_cost: result.cost,
        accepted,
        rejected,
        trace: result.trace.clone(),
        failure: None,
    }
}

pub(super) fn initial_window_diagnostics(
    problem: &WindowProblem,
    state: &DVector<f64>,
) -> WindowDiagnostics {
    let linearization = problem.linearize(state).ok();
    let imu_links = {
        let _capture_scope = InitialImuF64CaptureScope::enter();
        problem
            .imu_links
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(link_index, link)| {
                let _capture_link = InitialImuF64CaptureLink::set(link_index);
                problem
                    .imu_factor_with_trace(state, link)
                    .map(|(_, trace)| trace)
            })
            .collect::<Vec<_>>()
    };
    let (
        prior_cost,
        visual_cost,
        imu_cost,
        bias_cost,
        prior_factor_rows,
        visual_factor_rows,
        imu_factor_rows,
        bias_factor_rows,
    ) =
        linearization
            .as_ref()
            .map(|linearization| {
                let prior_factor_count = if problem.prior.is_some() { 1 } else { 0 };
                let anchor_factor_count = if problem.prior.is_none()
                    && problem.anchor_point.as_ref().is_some_and(|point| {
                        point.len() == NAV_STATE_DOF && !problem.states.is_empty()
                    }) {
                    1
                } else {
                    0
                };
                let prefix =
                    (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
                let imu_factor_count = problem
                    .imu_links
                    .len()
                    .saturating_mul(2)
                    .min(linearization.factors.len().saturating_sub(prefix));
                let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
                let cost = |factors: &[WhitenedFactorRowStack]| {
                    factors
                        .iter()
                        .map(|factor| factor.objective_cost)
                        .sum::<f64>()
                };
                let rows = |factors: &[WhitenedFactorRowStack]| {
                    factors
                        .iter()
                        .map(WhitenedFactorRowStack::rows)
                        .sum::<usize>()
                };
                let prior_cost = cost(&linearization.factors[..prefix]);
                let visual_cost = cost(&linearization.factors[prefix..visual_end]);
                let mut imu_cost = 0.0;
                let mut bias_cost = 0.0;
                for pair in linearization.factors[visual_end..].chunks(2) {
                    if let Some(factor) = pair.first() {
                        imu_cost += factor.objective_cost;
                    }
                    if let Some(factor) = pair.get(1) {
                        bias_cost += factor.objective_cost;
                    }
                }
                (
                    prior_cost,
                    visual_cost,
                    imu_cost,
                    bias_cost,
                    rows(&linearization.factors[..prefix]),
                    rows(&linearization.factors[prefix..visual_end]),
                    linearization.factors[visual_end..]
                        .chunks(2)
                        .filter_map(|pair| pair.first())
                        .map(WhitenedFactorRowStack::rows)
                        .sum(),
                    linearization.factors[visual_end..]
                        .chunks(2)
                        .filter_map(|pair| pair.get(1))
                        .map(WhitenedFactorRowStack::rows)
                        .sum(),
                )
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0));
    if let Some(linearization) = linearization.as_ref() {
        if iteration_trace_path().is_some() {
            emit_iteration_header();
        } else {
            emit_detail_trace(problem, state, linearization);
        }
    }
    WindowDiagnostics {
        attempted: true,
        state_dof: state.len(),
        factor_count: linearization
            .as_ref()
            .map_or(0, |linearization| linearization.factors.len()),
        factor_rows: linearization.as_ref().map_or(0, |linearization| {
            linearization
                .factors
                .iter()
                .map(WhitenedFactorRowStack::rows)
                .sum()
        }),
        landmark_count: problem.landmarks.len(),
        imu_link_count: problem.imu_links.len(),
        prior_rows: problem.prior.as_ref().map_or(0, WindowPrior::rows),
        prior_factor_rows,
        visual_factor_rows,
        imu_factor_rows,
        bias_factor_rows,
        prior_cost,
        visual_cost,
        imu_cost,
        bias_cost,
        imu_links,
        lm: Vec::new(),
        status: "started".into(),
        failure: None,
        state_writeback: false,
        landmark_writeback: 0,
        prior_carry: false,
    }
}

/// Cheap output metadata for the explicit no-MargData/no-trace path.
///
/// The solver still performs all of its normal linearization and LM work, but
/// callers that do not retain MargData or a trace do not need a second
/// diagnostic linearization before that solve.  Keep the public diagnostic
/// shape valid while leaving expensive factor/cost/LM payloads empty.
pub(super) fn minimal_window_diagnostics(
    problem: &WindowProblem,
    state: &DVector<f64>,
) -> WindowDiagnostics {
    WindowDiagnostics {
        attempted: true,
        state_dof: state.len(),
        factor_count: 0,
        factor_rows: 0,
        landmark_count: problem.landmarks.len(),
        imu_link_count: problem.imu_links.len(),
        prior_rows: problem.prior.as_ref().map_or(0, WindowPrior::rows),
        prior_factor_rows: 0,
        visual_factor_rows: 0,
        imu_factor_rows: 0,
        bias_factor_rows: 0,
        prior_cost: 0.0,
        visual_cost: 0.0,
        imu_cost: 0.0,
        bias_cost: 0.0,
        imu_links: Vec::new(),
        lm: Vec::new(),
        status: "started".into(),
        failure: None,
        state_writeback: false,
        landmark_writeback: 0,
        prior_carry: false,
    }
}

/// Populate the four `emit_marg` row counters from a final factor snapshot.
///
/// The lean-with-factors path deliberately skips `initial_window_diagnostics`
/// (which performs a full pre-solve linearization just for metadata).  The
/// factor grouping used here is the same prior/anchor → visual → IMU/bias
/// partition that function applies, so the MargData metadata is populated
/// without the duplicate linearization.
pub(super) fn fill_diagnostic_factor_rows(
    problem: &WindowProblem,
    diagnostics: &mut WindowDiagnostics,
    factors: &[WhitenedFactorRowStack],
) {
    let prior_factor_count = usize::from(problem.prior.is_some());
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(factors.len().saturating_sub(prefix));
    let visual_end = factors.len().saturating_sub(imu_factor_count);
    let rows = |slice: &[WhitenedFactorRowStack]| {
        slice
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>()
    };
    diagnostics.factor_count = factors.len();
    diagnostics.factor_rows = rows(factors);
    diagnostics.prior_factor_rows = rows(&factors[..prefix]);
    diagnostics.visual_factor_rows = rows(&factors[prefix..visual_end]);
    diagnostics.imu_factor_rows = factors[visual_end..]
        .chunks(2)
        .filter_map(|pair| pair.first())
        .map(WhitenedFactorRowStack::rows)
        .sum();
    diagnostics.bias_factor_rows = factors[visual_end..]
        .chunks(2)
        .filter_map(|pair| pair.get(1))
        .map(WhitenedFactorRowStack::rows)
        .sum();
}

/// Emit a complete, reproducible numeric snapshot when explicitly requested.
///
/// The normal estimator path does not write diagnostics.  Setting
/// `VISLOC_BASALT_DETAIL_TRACE` enables one text artifact for the selected
/// solve; `VISLOC_BASALT_DETAIL_FRAME` can restrict it to a final state frame
/// (for example, `4` in the MH01 frame-4 audit).  The snapshot is assembled
/// from the same grouped factors passed to `linearize`, so its row ordering is
/// an assertion target rather than a second implementation of the factor.
fn emit_detail_trace(
    problem: &WindowProblem,
    state: &DVector<f64>,
    linearization: &LmLinearization,
) {
    let Some(path) = diagnostic_env_snapshot().detail_trace.as_ref() else {
        return;
    };
    let target_frame = diagnostic_env_snapshot().detail_frame;
    if let Some(target_frame) = target_frame {
        if problem
            .states
            .last()
            .is_none_or(|state| state.frame_id != target_frame)
        {
            return;
        }
    }

    let mut output = String::new();
    let state_dof = state.len();
    let prior_factor_count = if problem.prior.is_some() { 1 } else { 0 };
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(linearization.factors.len().saturating_sub(prefix));
    let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
    let category_cost = |factors: &[WhitenedFactorRowStack]| {
        factors
            .iter()
            .map(|factor| factor.objective_cost)
            .sum::<f64>()
    };
    let prior_cost = category_cost(&linearization.factors[..prefix]);
    let visual_cost = category_cost(&linearization.factors[prefix..visual_end]);
    let mut imu_cost = 0.0;
    let mut bias_cost = 0.0;
    for pair in linearization.factors[visual_end..].chunks(2) {
        if let Some(factor) = pair.first() {
            imu_cost += factor.objective_cost;
        }
        if let Some(factor) = pair.get(1) {
            bias_cost += factor.objective_cost;
        }
    }

    writeln!(
        &mut output,
        "WINDOW_BEGIN frame {} state_dof {} factors {} rows {}",
        problem.states.last().map_or(0, |state| state.frame_id),
        state_dof,
        linearization.factors.len(),
        linearization
            .factors
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>()
    )
    .ok();
    writeln!(
        &mut output,
        "OBJECTIVE prior {prior_cost:.17e} visual {visual_cost:.17e} imu {imu_cost:.17e} bias {bias_cost:.17e} total {:.17e}",
        linearization.cost
    )
    .ok();
    writeln!(
        &mut output,
        "FACTOR_ORDER prefix_prior {} visual_end {} total {}",
        prefix,
        visual_end,
        linearization.factors.len()
    )
    .ok();
    // The reference traces print upstream `float` values.  Keep the Rust
    // storage/arithmetic types explicit in the component artifact so a
    // numerical mismatch cannot be mistaken for a row-order mismatch.
    writeln!(
        &mut output,
        "NUMERIC_TYPES state f64 pose f64 landmark f64 observation f64 camera f64 intermediate f64 landmark_projection {} upstream_reference float32",
        if problem.scalar_mode == ScalarMode::UpstreamF32 {
            "f32"
        } else {
            "f64"
        }
    )
    .ok();
    for (index, camera) in if problem.cameras.is_empty() {
        vec![(0_u16, problem.camera)]
    } else {
        problem
            .cameras
            .iter()
            .copied()
            .enumerate()
            .map(|(index, camera)| (index as u16, camera))
            .collect()
    } {
        writeln!(
            &mut output,
            "CAMERA index {} fx {:.17e} fy {:.17e} cx {:.17e} cy {:.17e} xi {:.17e} alpha {:.17e} width {} height {}",
            index,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.xi,
            camera.alpha,
            camera.width,
            camera.height
        )
        .ok();
    }
    for (index, transform) in if problem.t_imu_cam.is_empty() {
        vec![(0_u16, SE3::identity())]
    } else {
        problem
            .t_imu_cam
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, transform)| (index as u16, transform))
            .collect()
    } {
        let quaternion = transform.rotation.quaternion();
        writeln!(
            &mut output,
            "EXTRINSIC camera {} translation {:.17e} {:.17e} {:.17e} quaternion_xyzw {:.17e} {:.17e} {:.17e} {:.17e}",
            index,
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w
        )
        .ok();
    }
    for index in 0..(problem.poses.len() + problem.states.len()) {
        let Some(pose) = problem.block_pose(state, index) else {
            continue;
        };
        let Some(frame_id) = problem.block_frame_id(index) else {
            continue;
        };
        let quaternion = pose.rotation.quaternion();
        writeln!(
            &mut output,
            "BLOCK index {} frame {} kind {} translation {:.17e} {:.17e} {:.17e} quaternion_xyzw {:.17e} {:.17e} {:.17e} {:.17e}",
            index,
            frame_id,
            if index < problem.poses.len() { "pose" } else { "state" },
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w
        )
        .ok();
    }
    for (index, factor) in linearization.factors.iter().enumerate() {
        let category = if index < prefix {
            "prior"
        } else if index < visual_end {
            "visual"
        } else if (index - visual_end) % 2 == 0 {
            "imu"
        } else {
            "bias"
        };
        writeln!(
            &mut output,
            "FACTOR index {index} category {category} rows {} state_cols {} landmark_cols {} objective {:.17e}",
            factor.rows(),
            factor.state_jacobian.ncols(),
            factor.landmark_jacobian.ncols(),
            factor.objective_cost
        )
        .ok();
        emit_vector(
            &mut output,
            &format!("FACTOR_RESIDUAL index {index}"),
            &factor.residual,
        );
    }

    let mut visual_factors = Vec::new();
    for landmark_index in 0..problem.landmarks.len() {
        let Some(audit) = problem.visual_factor_with_audit(state, landmark_index) else {
            continue;
        };
        writeln!(
            &mut output,
            "VISUAL_ORDER ordinal {} track_id {} host_frame {} host_cam {} rows {}",
            visual_factors.len(),
            audit.track_id,
            audit.anchor_frame_id,
            audit.anchor_camera_id,
            audit.factor.rows()
        )
        .ok();
        let (reduced_jacobian, reduced_rhs, rank) =
            project_landmark_factor(&audit.factor, 1e-10, problem.scalar_mode);
        writeln!(
            &mut output,
            "LANDMARK_BEGIN track_id {} host_frame {} host_cam {} direction {:.17e} {:.17e} rho {:.17e} observations {} rows {} rank {} objective {:.17e}",
            audit.track_id,
            audit.anchor_frame_id,
            audit.anchor_camera_id,
            audit.direction.xy.x,
            audit.direction.xy.y,
            audit.inverse_distance,
            audit.observations.len(),
            audit.factor.rows(),
            rank,
            audit.factor.objective_cost
        )
        .ok();
        for observation in &audit.observations {
            let factor = &observation.factor;
            writeln!(
                &mut output,
                "OBS target_frame {} target_cam {} pixel {:.17e} {:.17e} raw_residual {:.17e} {:.17e} residual {:.17e} {:.17e} huber_weight {:.17e} sqrt_weight {:.17e} objective {:.17e}",
                problem
                    .block_frame_id(observation.observation.state_index)
                    .unwrap_or_default(),
                observation.observation.camera_id,
                observation.observation.pixel.x,
                observation.observation.pixel.y,
                factor.raw_residual.x,
                factor.raw_residual.y,
                factor.residual.x,
                factor.residual.y,
                factor.huber_weight,
                factor.sqrt_weight,
                factor.objective_cost
            )
            .ok();
            writeln!(
                &mut output,
                "PROJECTION {:.17e} {:.17e}",
                factor.projection.x, factor.projection.y
            )
            .ok();
            emit_fixed_matrix(&mut output, "Jp_anchor", &factor.anchor_pose_jacobian);
            emit_fixed_matrix(&mut output, "Jp_target", &factor.target_pose_jacobian);
            emit_fixed_matrix(&mut output, "Jl", &factor.landmark_jacobian);
        }
        emit_matrix(
            &mut output,
            "LANDMARK_STATE_JACOBIAN",
            &audit.factor.state_jacobian,
        );
        emit_matrix(
            &mut output,
            "LANDMARK_JACOBIAN",
            &audit.factor.landmark_jacobian,
        );
        emit_vector(&mut output, "LANDMARK_RESIDUAL", &audit.factor.residual);
        emit_matrix(&mut output, "REDUCED_ROWS", &reduced_jacobian);
        emit_vector(&mut output, "REDUCED_RHS", &reduced_rhs);
        visual_factors.push(audit.factor);
    }

    // This is the exact reduced system consumed by the LM solve, including
    // prior/IMU/bias rows and visual factors in their linearize order.
    let reduced = reduce_landmark_factors(&linearization.factors, state_dof, 1e-10);
    emit_matrix(&mut output, "REDUCED_H", &reduced.h);
    emit_vector(&mut output, "REDUCED_B", &reduced.b);
    let total_reduced_rows = linearization
        .factors
        .iter()
        .map(WhitenedFactorRowStack::rows)
        .sum::<usize>();
    let mut reduced_rows = DMatrix::zeros(total_reduced_rows, state_dof);
    let mut reduced_rhs = DVector::zeros(total_reduced_rows);
    let mut row = 0;
    for factor in &linearization.factors {
        let (jacobian, rhs, _) = project_landmark_factor(factor, 1e-10, problem.scalar_mode);
        let rows = jacobian.nrows();
        debug_assert_eq!(jacobian.ncols(), state_dof);
        reduced_rows
            .view_mut((row, 0), (rows, state_dof))
            .copy_from(&jacobian);
        reduced_rhs.rows_mut(row, rows).copy_from(&rhs);
        row += rows;
    }
    debug_assert_eq!(row, total_reduced_rows);
    emit_matrix(&mut output, "REDUCED_ROW_STACK", &reduced_rows);
    emit_vector(&mut output, "REDUCED_ROW_RHS", &reduced_rhs);
    writeln!(
        &mut output,
        "LANDMARK_FACTOR_COUNT {}",
        visual_factors.len()
    )
    .ok();
    writeln!(&mut output, "WINDOW_END").ok();

    if let Err(error) = fs::write(&path, output) {
        eprintln!(
            "basalt detail trace: failed to write {}: {error}",
            path.display()
        );
    }
}

fn emit_fixed_matrix<const ROWS: usize, const COLS: usize>(
    output: &mut String,
    label: &str,
    matrix: &nalgebra::SMatrix<f64, ROWS, COLS>,
) {
    writeln!(output, "{label} rows {ROWS} cols {COLS}").ok();
    for row in 0..ROWS {
        for col in 0..COLS {
            write!(output, " {:.17e}", matrix[(row, col)]).ok();
        }
        output.push('\n');
    }
}

fn emit_matrix(output: &mut String, label: &str, matrix: &DMatrix<f64>) {
    writeln!(
        output,
        "{label} rows {} cols {}",
        matrix.nrows(),
        matrix.ncols()
    )
    .ok();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            write!(output, " {:.17e}", matrix[(row, col)]).ok();
        }
        output.push('\n');
    }
}

fn emit_vector(output: &mut String, label: &str, vector: &DVector<f64>) {
    writeln!(output, "{label} len {}", vector.len()).ok();
    for value in vector.iter() {
        writeln!(output, "{value:.17e}").ok();
    }
}
