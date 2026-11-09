//! Opt-in JSON iteration trace: per-iteration records, solver-frontier
//! sidecars and full70 oracle events.

use super::*;

pub(super) fn iteration_trace_path() -> Option<std::path::PathBuf> {
    let policy = diagnostic_env_snapshot();
    policy
        .detail_iterations
        .then(|| policy.detail_trace.clone())
        .flatten()
}

fn iteration_trace_frame_matches(frame_id: u64) -> bool {
    diagnostic_env_snapshot()
        .detail_frame
        .is_none_or(|target| target == frame_id)
}

fn json_number(value: f64) -> JsonValue {
    serde_json::Number::from_f64(value)
        .map(JsonValue::Number)
        .unwrap_or(JsonValue::Null)
}

fn json_vector(values: impl IntoIterator<Item = f64>) -> JsonValue {
    JsonValue::Array(values.into_iter().map(json_number).collect())
}

fn json_matrix(matrix: &DMatrix<f64>) -> JsonValue {
    JsonValue::Array(
        (0..matrix.nrows())
            .map(|row| json_vector((0..matrix.ncols()).map(|column| matrix[(row, column)])))
            .collect(),
    )
}

fn json_visual_chain(chain: &VisualChainF32) -> JsonValue {
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector = |values: &[f32]| {
        JsonValue::Array(
            values
                .iter()
                .copied()
                .map(bits)
                .map(JsonValue::String)
                .collect(),
        )
    };
    json!({
        "target_camera_from_imu_f32_bits": vector(&chain.target_camera_from_imu),
        "target_imu_from_anchor_imu_f32_bits": vector(&chain.target_imu_from_anchor_imu),
        "target_camera_from_anchor_imu_f32_bits": vector(&chain.target_camera_from_anchor_imu),
        "target_camera_from_anchor_imu_fej_f32_bits": vector(&chain.target_camera_from_anchor_imu_fej),
        "target_camera_from_anchor_camera_f32_bits": vector(&chain.target_camera_from_anchor_camera),
        "target_camera_matrix_f32_bits_column_major": vector(&chain.target_camera_matrix),
        "relative_wrt_anchor_f32_bits_column_major": vector(&chain.relative_wrt_anchor),
        "point4_target_f32_bits": vector(&chain.point4_target),
        "point_target_f32_bits": vector(&chain.point_target),
        "projection_f32_bits": vector(&chain.projection),
        "projection_jacobian_f32_bits": vector(&chain.projection_jacobian),
    })
}

fn json_se3(pose: &SE3) -> JsonValue {
    let quaternion = pose.rotation.quaternion();
    let rotation = pose.rotation.to_rotation_matrix();
    let mut homogeneous = Matrix4::identity();
    homogeneous
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(rotation.matrix());
    homogeneous
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&pose.translation);
    json!({
        "translation": json_vector(pose.translation.iter().copied()),
        "quaternion_xyzw": json_vector([
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w,
        ]),
        "matrix": json_matrix(&DMatrix::from_iterator(4, 4, homogeneous.iter().copied())),
    })
}

pub(super) fn block_timestamp(problem: &WindowProblem, index: usize) -> Option<i64> {
    if index < problem.poses.len() {
        problem.poses.get(index).map(|pose| pose.timestamp_ns)
    } else {
        problem
            .states
            .get(index.checked_sub(problem.poses.len())?)
            .map(|state| state.timestamp_ns)
    }
}

fn block_frame_id_for_timestamp(problem: &WindowProblem, index: usize) -> Option<u64> {
    problem.block_frame_id(index)
}

fn json_aom(problem: &WindowProblem) -> JsonValue {
    let mut order = Vec::with_capacity(problem.poses.len() + problem.states.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        order.push(json!({
            "ordinal": index,
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "kind": "pose",
            "offset": problem.pose_offset(index),
            "dof": POSE_DOF,
        }));
    }
    for (index, state) in problem.states.iter().enumerate() {
        order.push(json!({
            "ordinal": problem.poses.len() + index,
            "frame_id": state.frame_id,
            "timestamp_ns": state.timestamp_ns,
            "kind": "state",
            "offset": problem.state_offset(index),
            "dof": NAV_STATE_DOF,
        }));
    }
    JsonValue::Array(order)
}

fn json_trial_visual_costs(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let mut records = Vec::new();
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        let Some(audit) = problem.visual_factor_with_audit(state, index) else {
            continue;
        };
        for observation in &audit.observations {
            let factor = &observation.factor;
            records.push(json!({
                "track_id": landmark.track_id,
                "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
                "host_cam": landmark.anchor_camera_id,
                "target_frame_id": block_frame_id_for_timestamp(problem, observation.observation.state_index),
                "target_cam": observation.observation.camera_id,
                "raw_residual": [factor.raw_residual.x, factor.raw_residual.y],
                "huber_weight": factor.huber_weight,
                "objective": factor.objective_cost,
                "visual_chain": factor.debug_chain.as_ref().map(json_visual_chain),
            }));
        }
    }
    json!({"state_boundary": "candidate_after_step", "observations": records,
        "native_host_order": problem.trial_host_order,
        "trial_geometry": json_trial_native_geometry(problem, state),
        "trial_imu": json_trial_native_imu(problem, state)})
}

fn json_trial_native_imu(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    if problem.scalar_mode != ScalarMode::UpstreamF32 {
        return JsonValue::Null;
    }
    let evaluate = || -> Option<JsonValue> {
        let mut links = problem.imu_links.iter().collect::<Vec<_>>();
        links.sort_by_key(|link| problem.states.get(link.from_index).map(|s| s.timestamp_ns));
        let mut records = Vec::with_capacity(links.len());
        let mut total = 0.0_f32;
        let words = |values: &[f32]| {
            values
                .iter()
                .map(|v| format!("{:08x}", v.to_bits()))
                .collect::<Vec<_>>()
        };
        for link in links {
            if link.delta.delta_time == 0.0 {
                continue;
            }
            let from_state = problem.states.get(link.from_index)?;
            let to_state = problem.states.get(link.to_index)?;
            let from = problem.block_nav(state, problem.poses.len() + link.from_index)?;
            let to = problem.block_nav(state, problem.poses.len() + link.to_index)?;
            let raw = crate::imu::trial_preintegration_residual_f32(
                &from,
                &to,
                &link.delta,
                problem.gravity_world,
            );
            let sqrt =
                crate::imu::sqrt_information_f32(&link.delta.covariance.map(|v| v as f32)).ok()?;
            let (inverse, weighted, cost) = crate::imu::trial_imu_quadratic_stages_f32(&sqrt, &raw);
            let before = total;
            total += cost;
            records.push(json!({"from_timestamp_ns":from_state.timestamp_ns,
                "to_timestamp_ns":to_state.timestamp_ns,"raw":words(raw.as_slice()),
                "sqrt_information":words(sqrt.as_slice()),"covariance_inverse":words(inverse.as_slice()),
                "half_residual_covariance_product":words(weighted.as_slice()),
                "objective":format!("{:08x}",cost.to_bits()),
                "accumulator_before":format!("{:08x}",before.to_bits())}));
        }
        Some(
            json!({"status":"ok","links":records,"cost_f32_bits":format!("{:08x}",total.to_bits())}),
        )
    };
    evaluate().unwrap_or_else(|| json!({"status":"invalid_topology_or_covariance"}))
}

// Candidate-only audit of computeError's geometry. This deliberately does not
// change LM decisions until the complete trial objective is integrated.
fn json_trial_native_geometry(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    if problem.scalar_mode != ScalarMode::UpstreamF32 {
        return JsonValue::Null;
    }
    let evaluate = || -> Option<JsonValue> {
        let ranks = problem
            .trial_host_order
            .iter()
            .enumerate()
            .map(|(rank, &key)| (key, rank))
            .collect::<BTreeMap<_, _>>();
        let mut ordered = BTreeMap::new();
        for (index, landmark) in problem.landmarks.iter().enumerate() {
            let host = (
                block_timestamp(problem, landmark.anchor_state_index)? as u64,
                landmark.anchor_camera_id,
            );
            let rank = *ranks.get(&host)?;
            for observation in &landmark.observations {
                let target = (
                    block_timestamp(problem, observation.state_index)? as u64,
                    observation.camera_id,
                );
                if ordered
                    .insert((rank, target, landmark.track_id), (index, observation))
                    .is_some()
                {
                    return None;
                }
            }
        }
        let mut records = Vec::with_capacity(ordered.len());
        let mut cost = 0.0_f32;
        for ((rank, target, track), (index, observation)) in ordered {
            let landmark = &problem.landmarks[index];
            let host = problem.trial_host_order[rank];
            let host_nav = problem.block_nav_visual_current(state, landmark.anchor_state_index)?;
            let target_nav = problem.block_nav_visual_current(state, observation.state_index)?;
            let matrix = super::aom::upstream_trial_transform_f32(
                &host_nav.imu_to_world,
                &target_nav.imu_to_world,
                &problem.camera_to_imu(host.1)?,
                &problem.camera_to_imu(target.1)?,
                host == target,
            );
            let result = super::aom::upstream_trial_observation_f32(
                problem.camera_model(target.1)?,
                matrix,
                Vector3::new(
                    landmark.direction.xy.x as f32,
                    landmark.direction.xy.y as f32,
                    landmark.inverse_distance as f32,
                ),
                nalgebra::Vector2::new(observation.pixel.x as f32, observation.pixel.y as f32),
                problem.factor_config,
            );
            let raw = result.map(|(raw, _)| {
                [
                    format!("{:08x}", raw.x.to_bits()),
                    format!("{:08x}", raw.y.to_bits()),
                ]
            });
            if let Some((_, objective)) = result {
                cost += objective;
            }
            records.push(json!({"track_id":track,"host_timestamp_ns":host.0,"host_cam":host.1,
                "target_timestamp_ns":target.0,"target_cam":target.1,"valid":result.is_some(),
                "raw_residual_f32_bits":raw,
                "objective_f32_bits":result.map(|(_, v)| format!("{:08x}",v.to_bits())),
                "accumulated_f32_bits":format!("{:08x}",cost.to_bits()),
                "matrix_f32_bits_column_major":matrix.iter().map(|v| format!("{:08x}",v.to_bits())).collect::<Vec<_>>()}));
        }
        Some(
            json!({"status":"ok","observations":records,"cost_f32_bits":format!("{:08x}",cost.to_bits())}),
        )
    };
    evaluate().unwrap_or_else(|| json!({"status":"missing_or_duplicate_topology"}))
}

fn json_block_values(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let mut poses = Vec::with_capacity(problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        let decoded = problem
            .block_pose(state, index)
            .unwrap_or_else(SE3::identity);
        poses.push(json!({
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "kind": "pose",
            "pose": json_se3(&decoded),
        }));
    }
    let mut states = Vec::with_capacity(problem.states.len());
    for (index, window_state) in problem.states.iter().enumerate() {
        let decoded = problem
            .block_nav(state, problem.poses.len() + index)
            .unwrap_or_default();
        states.push(json!({
            "frame_id": window_state.frame_id,
            "timestamp_ns": window_state.timestamp_ns,
            "kind": "state",
            "pose": json_se3(&decoded.imu_to_world),
            "velocity": json_vector(decoded.velocity_world_m_s.iter().copied()),
            "bias_gyro": json_vector(decoded.gyro_bias_rad_s.iter().copied()),
            "bias_accel": json_vector(decoded.accel_bias_m_s2.iter().copied()),
        }));
    }
    json!({"poses": poses, "states": states})
}

fn json_landmark_values(problem: &WindowProblem) -> JsonValue {
    let mut values = problem.landmarks.iter().collect::<Vec<_>>();
    values.sort_by_key(|landmark| landmark.track_id);
    JsonValue::Array(
        values
            .into_iter()
            .map(|landmark| {
                let observations = landmark
                    .observations
                    .iter()
                    .map(|observation| {
                        json!({
                            "target_frame_id": block_frame_id_for_timestamp(problem, observation.state_index),
                            "target_timestamp_ns": block_timestamp(problem, observation.state_index),
                            "target_cam": observation.camera_id,
                            "pixel": json_vector([observation.pixel.x, observation.pixel.y]),
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "track_id": landmark.track_id,
                    "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
                    "host_timestamp_ns": block_timestamp(problem, landmark.anchor_state_index),
                    "host_cam": landmark.anchor_camera_id,
                    "direction": json_vector([
                        landmark.direction.xy.x,
                        landmark.direction.xy.y,
                    ]),
                    "rho": json_number(landmark.inverse_distance),
                    "observations": observations,
                })
            })
            .collect(),
    )
}

fn visual_factor_records(
    problem: &WindowProblem,
    state: &DVector<f64>,
    linearization: &LmLinearization,
) -> (JsonValue, Vec<(usize, usize)>) {
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
    let mut factor_index = prefix;
    let mut reduced_row = 0usize;
    let mut records = Vec::new();
    let mut spans = Vec::new();
    for (landmark_index, landmark) in problem.landmarks.iter().enumerate() {
        let Some(audit) = problem.visual_factor_with_audit(state, landmark_index) else {
            continue;
        };
        if factor_index >= visual_end {
            break;
        }
        let factor = &linearization.factors[factor_index];
        factor_index += 1;
        let (reduced_rows, reduced_rhs, rank) =
            project_landmark_factor(factor, 1e-10, problem.scalar_mode);
        let span = (reduced_row, reduced_rows.nrows());
        reduced_row += reduced_rows.nrows();
        spans.push(span);
        let observations = audit
            .observations
            .iter()
            .map(|observation| {
                let factor = &observation.factor;
                let mut value = json!({
                    "target_frame_id": block_frame_id_for_timestamp(problem, observation.observation.state_index),
                    "target_timestamp_ns": block_timestamp(problem, observation.observation.state_index),
                    "target_cam": observation.observation.camera_id,
                    "pixel": json_vector([
                        observation.observation.pixel.x,
                        observation.observation.pixel.y,
                    ]),
                    "projection": json_vector([factor.projection.x, factor.projection.y]),
                    "raw_residual": json_vector([factor.raw_residual.x, factor.raw_residual.y]),
                    "residual": json_vector([factor.residual.x, factor.residual.y]),
                    "huber_weight": json_number(factor.huber_weight),
                    "sqrt_weight": json_number(factor.sqrt_weight),
                    "objective": json_number(factor.objective_cost),
                    "jp_anchor": json_matrix(&DMatrix::from_iterator(2, 6, factor.anchor_pose_jacobian.iter().copied())),
                    "jp_target": json_matrix(&DMatrix::from_iterator(2, 6, factor.target_pose_jacobian.iter().copied())),
                    "jl": json_matrix(&DMatrix::from_iterator(2, 3, factor.landmark_jacobian.iter().copied())),
                });
                if let Some(chain) = factor.debug_chain.as_ref() {
                    value["visual_chain"] = json_visual_chain(chain);
                }
                value
            })
            .collect::<Vec<_>>();
        records.push(json!({
            "track_id": landmark.track_id,
            "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
            "host_timestamp_ns": block_timestamp(problem, landmark.anchor_state_index),
            "host_cam": landmark.anchor_camera_id,
            "direction": json_vector([landmark.direction.xy.x, landmark.direction.xy.y]),
            "rho": json_number(landmark.inverse_distance),
            "row_span": [span.0, span.1],
            "rank": rank,
            "observations": observations,
            "state_jacobian": json_matrix(&factor.state_jacobian),
            "landmark_jacobian": json_matrix(&factor.landmark_jacobian),
            "residual": json_vector(factor.residual.iter().copied()),
            "reduced_rows": json_matrix(&reduced_rows),
            "reduced_rhs": json_vector(reduced_rhs.iter().copied()),
        }));
    }
    records.sort_by_key(|record| record["track_id"].as_u64().unwrap_or_default());
    (JsonValue::Array(records), spans)
}

pub(super) fn project_landmark_factor(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> (DMatrix<f64>, DVector<f64>, usize) {
    if scalar_mode == ScalarMode::UpstreamF32 {
        let (jacobian, residual, rank) = landmark_nullspace_projection_f32(factor, tolerance);
        let jacobian = DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, column| {
            jacobian[(row, column)] as f64
        });
        let residual =
            DVector::from_iterator(residual.len(), residual.iter().copied().map(f64::from));
        (jacobian, residual, rank)
    } else {
        landmark_nullspace_projection(factor, tolerance)
    }
}

fn global_reduced_rows(
    linearization: &LmLinearization,
    scalar_mode: ScalarMode,
) -> (DMatrix<f64>, DVector<f64>) {
    let state_dof = linearization
        .factors
        .first()
        .map_or(0, |factor| factor.state_jacobian.ncols());
    let projected = linearization
        .factors
        .iter()
        .map(|factor| project_landmark_factor(factor, 1e-10, scalar_mode))
        .collect::<Vec<_>>();
    let rows = projected
        .iter()
        .map(|(jacobian, _, _)| jacobian.nrows())
        .sum();
    let mut jacobian = DMatrix::zeros(rows, state_dof);
    let mut rhs = DVector::zeros(rows);
    let mut row = 0;
    for (factor_jacobian, factor_rhs, _) in projected {
        let count = factor_jacobian.nrows();
        jacobian
            .view_mut((row, 0), (count, state_dof))
            .copy_from(&factor_jacobian);
        rhs.rows_mut(row, count).copy_from(&factor_rhs);
        row += count;
    }
    (jacobian, rhs)
}

/// Return the source-category objective at an opt-in iteration snapshot.
///
/// The normal solver keeps these values only as the scalar LM objective.  The
/// detail trace needs the four terms separately so a native logger can be
/// compared at the same trial boundary.  Re-linearizing the display clone is
/// intentional: for a trial snapshot it includes the candidate navigation
/// and back-substituted landmark state, rather than repeating the iteration's
/// base-state cost for every trial.
fn iteration_category_costs(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let Some(linearization) = problem.linearize(state).ok() else {
        return JsonValue::Null;
    };
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
    let prefix = (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(linearization.factors.len().saturating_sub(prefix));
    let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
    let sum = |factors: &[WhitenedFactorRowStack]| {
        if problem.scalar_mode == ScalarMode::UpstreamF32 {
            factors.iter().fold(0.0_f32, |total, factor| {
                total + factor.objective_cost as f32
            }) as f64
        } else {
            factors
                .iter()
                .map(|factor| factor.objective_cost)
                .sum::<f64>()
        }
    };
    let prior = if prefix != 0 {
        problem.prior.as_ref().map_or_else(
            || sum(&linearization.factors[..prefix]),
            |prior| problem.marginal_prior_reduced_cost(state, prior),
        )
    } else {
        0.0
    };
    let visual = sum(&linearization.factors[prefix..visual_end]);
    let (imu, bias) = if problem.scalar_mode == ScalarMode::UpstreamF32 {
        let mut imu = 0.0_f32;
        let mut bias = 0.0_f32;
        for pair in linearization.factors[visual_end..].chunks(2) {
            if let Some(factor) = pair.first() {
                imu += factor.objective_cost as f32;
            }
            if let Some(factor) = pair.get(1) {
                bias += factor.objective_cost as f32;
            }
        }
        (imu as f64, bias as f64)
    } else {
        let mut imu = 0.0_f64;
        let mut bias = 0.0_f64;
        for pair in linearization.factors[visual_end..].chunks(2) {
            if let Some(factor) = pair.first() {
                imu += factor.objective_cost;
            }
            if let Some(factor) = pair.get(1) {
                bias += factor.objective_cost;
            }
        }
        (imu, bias)
    };
    json!({
        "prior": json_number(prior),
        "visual": json_number(visual),
        "imu": json_number(imu),
        "bias": json_number(bias),
        "total": json_number(linearization.cost),
        "factor_prefix": prefix,
        "visual_factor_end": visual_end,
        "scalar_mode": if problem.scalar_mode == ScalarMode::UpstreamF32 { "f32" } else { "f64" },
    })
}

pub(super) fn emit_iteration_header() {
    let Some(path) = iteration_trace_path() else {
        return;
    };
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_some() {
        return;
    }
    let _ = INITIALIZED.set(());
    {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&path, b"");
    }
    let header = json!({
        "schema": "basalt.vio_iteration.v1",
        "record": "header",
        "source": "rust",
        "scalar_type": "f64",
        "upstream_reference": "float32",
        "cast_points": [
            "pipelines/basalt/src/vio/estimator.rs::process_impl calibrated IMU remains f64",
            "pipelines/basalt/src/vio/window.rs::anchored_visual_reprojection_factor f64",
            "upstream src/vio.cpp::ImuData<double> -> float estimator queue"
        ],
        "symbols": [
            "visloc_basalt::vio::aom::solve_lm",
            "visloc_basalt::vio::window::WindowProblem::linearize",
            "visloc_basalt::vio::window::WindowProblem::landmark_steps"
        ]
    });
    if let Ok(line) = serde_json::to_string(&header) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn write_iteration_record(record: &JsonValue) {
    let Some(path) = iteration_trace_path() else {
        return;
    };
    emit_iteration_header();
    let Ok(line) = serde_json::to_string(record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Serialize the source-order normal-system checkpoints when the f32 reducer
/// was entered with an explicit parity diagnostic enabled.  Every operand in
/// `ReducedNormalSystem::diagnostic_stages` is already an f32 value widened
/// to f64; the explicit casts/adds below therefore preserve the f32 rounding
/// boundary while keeping this helper observation-only.
pub(super) fn diagnostic_normal_system_stages(
    reduced: &ReducedNormalSystem,
    bits: &impl Fn(f32) -> String,
) -> JsonValue {
    let Some(stages) = reduced.diagnostic_stages.as_ref() else {
        return JsonValue::Null;
    };
    let matrix_bits = |matrix: &DMatrix<f64>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value as f32)).collect::<Vec<_>>(),
        })
    };
    let vector_bits = |vector: &DVector<f64>| {
        json!(vector
            .iter()
            .map(|value| bits(*value as f32))
            .collect::<Vec<_>>())
    };
    // Basalt's `get_dense_H_b` checkpoint is captured before the LM solver
    // forms `H_copy = H + Hdiag_lambda`.  The pinned MH01 driver also leaves
    // `setPoseDamping(lambda)` disabled, so its stage-3 pose-damping hook is
    // a no-op for this direct H/b boundary.  `damping_diag` belongs to the
    // separate solver-side H_copy checkpoint and must not be folded into the
    // source normal-system stages.
    let plus_damping_h = stages.visual_imu_h.clone();
    let plus_damping_b = stages.visual_imu_b.clone();
    let mut plus_prior_h = plus_damping_h.clone();
    for col in 0..plus_prior_h.ncols() {
        for row in 0..plus_prior_h.nrows() {
            plus_prior_h[(row, col)] =
                (plus_prior_h[(row, col)] as f32 + stages.prior_h[(row, col)] as f32) as f64;
        }
    }
    let mut plus_prior_b = plus_damping_b.clone();
    for index in 0..plus_prior_b.len() {
        plus_prior_b[index] = (plus_prior_b[index] as f32 + stages.prior_b[index] as f32) as f64;
    }
    json!({
        "stage_order": [
            "visual_accumulator",
            "visual_plus_imu",
            "plus_pose_damping",
            "plus_marginal_prior"
        ],
        "visual_accumulator": {
            "h": matrix_bits(&stages.visual_h),
            "b": vector_bits(&stages.visual_b),
        },
        "visual_plus_imu": {
            "h": matrix_bits(&stages.visual_imu_h),
            "b": vector_bits(&stages.visual_imu_b),
        },
        "plus_pose_damping": {
            "h": matrix_bits(&plus_damping_h),
            "b": vector_bits(&plus_damping_b),
        },
        "plus_marginal_prior": {
            "h": matrix_bits(&plus_prior_h),
            "b": vector_bits(&plus_prior_b),
        },
        // Keep the raw square-root prior contribution alongside the
        // cumulative checkpoint.  The cumulative subtraction is not an
        // invertible f32 operation, so this sidecar is required to localize
        // a marginalization mismatch without changing the solver arithmetic.
        "marginal_prior": {
            "h": matrix_bits(&stages.prior_h),
            "b": vector_bits(&stages.prior_b),
        },
        "production_reduced": {
            "h": matrix_bits(&reduced.h),
            "b": vector_bits(&reduced.b),
        },
    })
}

/// Emit the small, bit-oriented solver frontier sidecar used to compare the
/// Rust LM boundary with Basalt's `sqrt_keypoint_vio.cpp`.  This is separate
/// from the large human-readable iteration trace and is enabled only by an
/// explicit path.  The normal estimator therefore pays no serialization or
/// state-copy cost.
fn emit_solver_frontier_sidecar(
    problem: &WindowProblem,
    applied_problem: &WindowProblem,
    event: &LmDiagnosticEvent<'_>,
) {
    let Some(path) = diagnostic_env_snapshot().solver_frontier_trace.as_ref() else {
        return;
    };
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector_bits = |values: &[f64]| {
        json!(values
            .iter()
            .map(|value| bits(*value as f32))
            .collect::<Vec<_>>())
    };
    let matrix_bits = |matrix: &DMatrix<f64>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value as f32)).collect::<Vec<_>>(),
        })
    };
    let matrix_bits_f32 = |matrix: &DMatrix<f32>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value)).collect::<Vec<_>>(),
        })
    };
    let stage_systems = diagnostic_normal_system_stages(event.reduced, &bits);
    let (h_copy, h_copy_source) = if let Some(h_copy) = event.damped_h {
        // Capture the matrix owned by the active solver.  Rebuilding H_copy
        // from the f64 reduced mirror would conceal any cast/add schedule
        // difference at precisely the boundary under audit.
        (matrix_bits_f32(h_copy), "solver_damped_h_f32_direct")
    } else {
        let mut h_copy = event.reduced.h.clone();
        for index in 0..h_copy.nrows().min(h_copy.ncols()) {
            h_copy[(index, index)] =
                (event.reduced.h[(index, index)] as f32 + event.damping_diag[index] as f32) as f64;
        }
        (matrix_bits(&h_copy), "f64_reduced_plus_f32_diag_fallback")
    };
    let state_record = |state: &BasaltNavState| {
        let q = state.imu_to_world.rotation.quaternion();
        json!({
            "translation_f32_bits": state
                .imu_to_world
                .translation
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "quaternion_xyzw_f32_bits": [
                bits(q.i as f32), bits(q.j as f32), bits(q.k as f32), bits(q.w as f32)
            ],
            "velocity_f32_bits": state
                .velocity_world_m_s
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "bias_gyro_f32_bits": state
                .gyro_bias_rad_s
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "bias_accel_f32_bits": state
                .accel_bias_m_s2
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
        })
    };
    let state_map = |window: &WindowProblem| {
        JsonValue::Array(
            window
                .states
                .iter()
                .map(|state| {
                    json!({
                        "frame_id": state.frame_id,
                        "timestamp_ns": state.timestamp_ns,
                        "linearized": state.linearized,
                        "linearized_state": state_record(&state.linearized_nav),
                        "current_state": state_record(&state.nav),
                        "delta_f32_bits": state
                            .linearized_delta
                            .iter()
                            .map(|value| bits(*value as f32))
                            .collect::<Vec<_>>(),
                    })
                })
                .collect(),
        )
    };
    let step = event
        .step
        .map(|value| value.iter().copied().collect::<Vec<_>>());
    let positive_step = step.as_ref().map(|values| {
        values
            .iter()
            .map(|value| -(*value as f32) as f64)
            .collect::<Vec<_>>()
    });
    let record = json!({
        "schema": "basalt.m7im15_solver_frontier.v1",
        "source": "rust",
        "frame_id": problem.states.last().map(|state| state.frame_id),
        "frame_timestamp_ns": problem.states.last().map(|state| state.timestamp_ns),
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "decision": event.decision,
        "lambda_f32_bits": bits(event.lambda as f32),
        "reduced_h": matrix_bits(&event.reduced.h),
        "reduced_b": vector_bits(event.reduced.b.as_slice()),
        "damping_diag": vector_bits(event.damping_diag.as_slice()),
        "h_copy": h_copy,
        "h_copy_source": h_copy_source,
        "normal_system_stages": stage_systems,
        "step_solved_f32_bits": step
            .as_ref()
            .map(|values| vector_bits(values))
            .unwrap_or(JsonValue::Null),
        "step_positive_equivalent_f32_bits": positive_step
            .as_ref()
            .map(|values| vector_bits(values))
            .unwrap_or(JsonValue::Null),
        "apply_before": state_map(problem),
        "apply_after": state_map(applied_problem),
        "numeric_types": { "normal_system": "f32", "state": "f64_cast_f32" },
    });
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_none() {
        let _ = INITIALIZED.set(());
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, b"");
    }
    if let Ok(line) = serde_json::to_string(&record) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Build the non-factor identity needed to interpret a lossless full-factor
/// oracle.  The factor vector carries absolute state columns and observation
/// state indices; this compact context maps those indices back to the active
/// pose/state timestamps and preserves the stored square-root prior before the
/// generated FEJ-shifted factor RHS.  It is constructed only when the explicit
/// full-factor oracle path is enabled.
fn full70_oracle_window_context(problem: &WindowProblem) -> JsonValue {
    let mut state_blocks = Vec::with_capacity(problem.poses.len() + problem.states.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        state_blocks.push(json!({
            "index": index,
            "kind": "pose",
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "offset": problem.pose_offset(index),
            "dof": POSE_DOF,
            "is_keyframe": pose.is_keyframe,
        }));
    }
    for (index, state) in problem.states.iter().enumerate() {
        state_blocks.push(json!({
            "index": problem.poses.len() + index,
            "kind": "state",
            "frame_id": state.frame_id,
            "timestamp_ns": state.timestamp_ns,
            "offset": problem.state_offset(index),
            "dof": NAV_STATE_DOF,
            "is_keyframe": state.is_keyframe,
            "is_latest": state.is_latest,
            "linearized": state.linearized,
        }));
    }

    let prior_source = problem.prior.as_ref().map_or_else(
        || {
            json!({
                "source": if problem.anchor_point.is_some() { "anchor" } else { "absent" },
                "anchor_point_f32_bits": problem
                    .anchor_point
                    .as_ref()
                    .map(full70_f32_bits_vector)
                    .unwrap_or(JsonValue::Null),
            })
        },
        |prior| {
            json!({
                "source": "stored",
                "frame_ids": prior.frame_ids,
                "block_kinds": prior.block_kinds.iter().map(|kind| match kind {
                    WindowBlockKind::Pose => "pose",
                    WindowBlockKind::StatePose => "state_pose",
                    WindowBlockKind::State => "state",
                }).collect::<Vec<_>>(),
                "jacobian_f32_bits": full70_f32_bits_matrix(&prior.jacobian),
                "rhs_f32_bits": full70_f32_bits_vector(&prior.rhs),
                "fej_point_f32_bits": full70_f32_bits_vector(&prior.fej_point),
            })
        },
    );

    json!({
        "state_blocks": state_blocks,
        "prior_source": prior_source,
        "scalar_mode": format!("{:?}", problem.scalar_mode),
        "state_dof": problem.state_dof(),
    })
}

fn full70_oracle_event_selected(
    policy: &DiagnosticEnvSnapshot,
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
) -> bool {
    // The artifact name and strict schema are frame-4 specific.  With no
    // explicit frame selector, skip earlier events rather than consuming the
    // one-shot slot on a smaller warm-up window.
    if policy.full70_factor_oracle_frame.is_none() && frame_id != 4 {
        return false;
    }
    let frame_matches = match policy.full70_factor_oracle_frame {
        None => true,
        Some(Ok(target)) => target == frame_id,
        Some(Err(())) => false,
    };
    let iteration_matches = match policy.full70_factor_oracle_iteration {
        None => true,
        Some(Ok(target)) => target == event.iteration,
        Some(Err(())) => false,
    };
    let trial_matches = match policy.full70_factor_oracle_trial {
        None => true,
        Some(Ok(target)) => target == event.trial,
        Some(Err(())) => false,
    };
    let phase_matches = match policy.full70_factor_oracle_phase.as_deref() {
        None => event.phase == "iteration_start",
        Some(target) => {
            matches!(
                target,
                "iteration_start" | "trial" | "accepted" | "rejected" | "converged"
            ) && target == event.phase
        }
    };
    if !(frame_matches && iteration_matches && trial_matches && phase_matches) {
        return false;
    }
    // A path names one lossless event bundle.  Without this one-shot guard a
    // selector such as frame=4 would successfully write the first iteration
    // and then repeatedly collide with the same target, producing misleading
    // diagnostic errors.  A caller that needs another event uses a fresh
    // process/path (or a narrower iteration/trial selector).
    static CAPTURED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    CAPTURED
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
}

pub(super) fn emit_iteration_event(problem: &mut WindowProblem, event: LmDiagnosticEvent<'_>) {
    if let Some(frame_id) = problem.states.last().map(|state| state.frame_id) {
        let policy = diagnostic_env_snapshot();
        if let Some(path) = policy.full70_factor_oracle.as_ref() {
            if full70_oracle_event_selected(policy, frame_id, &event) {
                let context = full70_oracle_window_context(problem);
                if let Err(error) = emit_full70_factor_oracle(path, frame_id, &event, &context) {
                    // Diagnostic capture is fail-closed and never changes the
                    // solver result.  Keep the error visible to the harness;
                    // a missing/partial oracle is not silently accepted.
                    eprintln!("full70 factor oracle rejected: {error:?}");
                }
            }
        }
    }
    let Some(path) = iteration_trace_path() else {
        return;
    };
    let Some(frame_id) = problem.states.last().map(|state| state.frame_id) else {
        return;
    };
    if !iteration_trace_frame_matches(frame_id) {
        return;
    }
    let base_state = event.base_state;
    let display_state = if event.phase == "trial" {
        event.trial_state.unwrap_or(event.state)
    } else {
        event.state
    };
    let mut display_problem = problem.clone();
    if event.phase == "trial" {
        if let Some(step) = event.step {
            // Reuse the exact trial clone used by `trial_cost`, including
            // manifold sidecars and eliminated-landmark increments. Keeping
            // one constructor avoids a diagnostic-only prior trajectory that
            // differs from the scalar LM objective.
            display_problem = problem.trial_problem_with_step_owned(base_state, step);
        }
    }
    let (landmark_factors, visual_spans) =
        visual_factor_records(problem, base_state, event.linearization);
    let (global_rows, global_rhs) = global_reduced_rows(event.linearization, problem.scalar_mode);
    let h_copy = event.damped_h.map(|matrix| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "f32_bits": matrix.iter().map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>(),
        })
    });
    let backsub = event.step.map_or_else(
        || JsonValue::Array(Vec::new()),
        |step| {
            JsonValue::Array(
                problem
                    .landmark_steps(base_state, step)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, increment)| {
                        let increment = increment?;
                        Some(json!({
                            "track_id": problem.landmarks.get(index)?.track_id,
                            "delta": json_vector(increment.iter().copied()),
                        }))
                    })
                    .collect(),
            )
        },
    );
    let model_parts = if problem.scalar_mode == ScalarMode::UpstreamF32 && event.phase == "trial" {
        event
            .step
            .zip(event.model_decrease)
            .and_then(|(step, actual)| {
                super::aom::diagnostic_model_decrease_parts_f32(
                    &event.linearization.factors,
                    step,
                    actual,
                )
            })
    } else {
        None
    };
    let record = json!({
        "schema": "basalt.vio_iteration.v1",
        "record": "snapshot",
        "source": "rust",
        "frame_id": frame_id,
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "decision": event.decision,
        "lambda": json_number(event.lambda),
        "lambda_after": json_number(event.lambda_after),
        "cost": {
            "before": json_number(event.cost_before),
            "model": event.model_cost.map_or(JsonValue::Null, json_number),
            "model_decrease": event.model_decrease.map_or(JsonValue::Null, json_number),
            "model_parts": model_parts,
            "actual": event.actual_cost.map_or(JsonValue::Null, json_number),
            "step_norm": event.step_norm.map_or(JsonValue::Null, json_number),
        },
        "aom": json_aom(problem),
        "blocks": json_block_values(&display_problem, display_state),
        "landmarks": json_landmark_values(&display_problem),
        "delta": event.step.map_or(JsonValue::Null, |step| json_vector(step.iter().copied())),
        "backsub_landmark_delta": backsub,
        "damping_diag": json_vector(event.damping_diag.iter().copied()),
        "h_copy": h_copy.unwrap_or(JsonValue::Null),
        "global": {
            "h": json_matrix(&event.reduced.h),
            "b": json_vector(event.reduced.b.iter().copied()),
            "reduced_rows": json_matrix(&global_rows),
            "reduced_rhs": json_vector(global_rhs.iter().copied()),
            "normal_system_stages": diagnostic_normal_system_stages(
                event.reduced,
                &|value: f32| format!("{:08x}", value.to_bits()),
            ),
        },
        "category_costs": iteration_category_costs(&display_problem, display_state),
        "trial_visual_costs": if event.phase == "trial" && diagnostic_env_snapshot().visual_chain_trace {
            json_trial_visual_costs(&display_problem, display_state)
        } else {
            JsonValue::Null
        },
        "visual_row_spans": visual_spans
            .into_iter()
            .map(|(start, count)| json!([start, count]))
            .collect::<Vec<_>>(),
        "landmark_factors": landmark_factors,
        "factor_count": event.linearization.factors.len(),
        "factor_rows": event
            .linearization
            .factors
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>(),
        "numeric_types": {
            "state": "f64",
            "pose": "f64",
            "landmark": "f64",
            "observation": "f64",
            "camera": "f64",
            "landmark_projection": if problem.scalar_mode == ScalarMode::UpstreamF32 {
                "f32"
            } else {
                "f64"
            },
            "upstream_reference": "float32",
        },
        "trace_path": path,
    });
    emit_solver_frontier_sidecar(problem, &display_problem, &event);
    write_iteration_record(&record);
}
