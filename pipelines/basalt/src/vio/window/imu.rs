//! IMU preintegration and bias random-walk factors, plus the IMU factor-input
//! and f64 residual diagnostics.

use super::*;

impl WindowProblem {
    pub(super) fn imu_factor(
        &self,
        state: &DVector<f64>,
        link: WindowImuLink,
    ) -> Option<WhitenedFactorRowStack> {
        self.imu_factor_with_trace(state, link)
            .map(|(factor, _)| factor)
    }

    pub(super) fn imu_factor_with_trace(
        &self,
        state: &DVector<f64>,
        link: WindowImuLink,
    ) -> Option<(WhitenedFactorRowStack, ImuLinkDiagnostics)> {
        if link.from_index >= self.states.len() || link.to_index >= self.states.len() {
            return None;
        }
        let layout = self.layout();
        let from_offset = layout.offset_for_state(link.from_index)?;
        let to_offset = layout.offset_for_state(link.to_index)?;
        let base = (0..self.states.len())
            .map(|index| {
                self.block_nav(state, self.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let mut delta = link.delta.clone();
        // Preintegration normally accumulates this covariance while the
        // estimator supplies calibrated noise.  Keep direct WindowProblem
        // callers deterministic as well by materializing the same diagonal
        // process covariance when they provide a legacy zero-covariance delta.
        if delta.covariance.norm_squared() <= 1e-30 && delta.delta_time.is_finite() {
            let dt = delta.delta_time.max(1e-9);
            for axis in 0..3 {
                // Literal upstream row order is [position, rotation,
                // velocity], matching IntegratedImuMeasurement.
                delta.covariance[(axis, axis)] =
                    self.imu_noise.accel_density.powi(2) * dt.powi(3) / 3.0;
                delta.covariance[(3 + axis, 3 + axis)] = self.imu_noise.gyro_density.powi(2) * dt;
                delta.covariance[(6 + axis, 6 + axis)] = self.imu_noise.accel_density.powi(2) * dt;
            }
        }
        let from = &base[link.from_index];
        let to = &base[link.to_index];
        let jac_from = if self.states[link.from_index].linearized {
            &self.states[link.from_index].linearized_nav
        } else {
            from
        };
        let jac_to = if self.states[link.to_index].linearized {
            &self.states[link.to_index].linearized_nav
        } else {
            to
        };
        let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
            whitened_preintegration_factor_upstream_f32_fej_mode(
                from,
                to,
                jac_from,
                jac_to,
                &delta,
                self.gravity_world,
                self.states[link.from_index].linearized || self.states[link.to_index].linearized,
            )
            .ok()?
        } else {
            whitened_preintegration_factor(from, to, &delta, self.gravity_world).ok()?
        };
        let diagnostic_lm_iteration = active_diagnostic_lm_iteration();
        emit_imu_factor_input_diagnostic(
            &self.states[link.from_index],
            &self.states[link.to_index],
            self.states.last().map(|state| state.frame_id),
            link.from_index,
            link.to_index,
            diagnostic_lm_iteration,
            from,
            to,
            jac_from,
            jac_to,
            &delta,
            self.gravity_world,
        );
        let imu_input_diagnostic = if self.scalar_mode == ScalarMode::UpstreamF32
            && diagnostic_env_snapshot().diagnostic_imu_rows.is_some()
        {
            build_imu_factor_input_diagnostic(
                &self.states[link.from_index],
                &self.states[link.to_index],
                self.states.last().map(|state| state.frame_id),
                link.from_index,
                link.to_index,
                from,
                to,
                jac_from,
                jac_to,
                &delta,
                self.gravity_world,
            )
        } else {
            None
        };
        let (position, rotation, velocity, rotation_product) =
            imu_unwhitened_residual(from, to, &delta, self.gravity_world);
        let gyro_bias_delta = from.gyro_bias_rad_s - delta.bias_gyro;
        let accel_bias_delta = from.accel_bias_m_s2 - delta.bias_accel;
        let bias_rotation_correction = delta.jacobian_rotation_gyro_bias * gyro_bias_delta;
        let bias_velocity_correction = delta.jacobian_velocity_gyro_bias * gyro_bias_delta
            + delta.jacobian_velocity_accel_bias * accel_bias_delta;
        let bias_position_correction = delta.jacobian_position_gyro_bias * gyro_bias_delta
            + delta.jacobian_position_accel_bias * accel_bias_delta;
        emit_imu_unwhitened_residual_f64_diagnostic(
            &self.states[link.from_index],
            &self.states[link.to_index],
            self.states.last().map(|state| state.frame_id),
            link.from_index,
            link.to_index,
            diagnostic_lm_iteration,
            &rotation_product,
            rotation,
        );
        let eigenvalues = SymmetricEigen::new(delta.covariance.clone()).eigenvalues;
        let covariance_eigen_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let covariance_eigen_max = eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut jacobian = DMatrix::zeros(factor.residual.len(), self.state_dof());
        jacobian
            .view_mut(
                (0, self.state_offset(link.from_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
        jacobian
            .view_mut(
                (0, self.state_offset(link.to_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
        let factor = WhitenedFactorRowStack::new(
            jacobian,
            DMatrix::zeros(factor.residual.len(), 0),
            factor.residual,
        )?
        .with_kind(FactorKind::Imu)
        .with_imu_link_offsets(from_offset, to_offset);
        let factor = if let Some(payload) = imu_input_diagnostic {
            factor.with_imu_input_diagnostic(payload)
        } else {
            factor
        };
        let trace = ImuLinkDiagnostics {
            from_frame_id: self.states[link.from_index].frame_id,
            to_frame_id: self.states[link.to_index].frame_id,
            delta_time: delta.delta_time,
            residual_position: vector3_array(position),
            residual_rotation: vector3_array(rotation),
            residual_velocity: vector3_array(velocity),
            whitened_norm: factor.residual.norm(),
            gyro_bias_delta: vector3_array(gyro_bias_delta),
            accel_bias_delta: vector3_array(accel_bias_delta),
            bias_rotation_correction: vector3_array(bias_rotation_correction),
            bias_velocity_correction: vector3_array(bias_velocity_correction),
            bias_position_correction: vector3_array(bias_position_correction),
            covariance_eigen_min,
            covariance_eigen_max,
        };
        Some((factor, trace))
    }

    /// Build an IMU factor from a trial value view.  Diagnostic payloads are
    /// intentionally not produced here: the clean trial path is selected only
    /// when no diagnostic/probe environment is active, while the owned clone
    /// path continues to call `imu_factor_with_trace` above.
    pub(super) fn imu_factor_view<V: WindowValueView>(
        &self,
        values: &V,
        link: WindowImuLink,
    ) -> Option<WhitenedFactorRowStack> {
        let base = values.base();
        if link.from_index >= base.states.len() || link.to_index >= base.states.len() {
            return None;
        }
        let layout = base.layout();
        let from_offset = layout.offset_for_state(link.from_index)?;
        let to_offset = layout.offset_for_state(link.to_index)?;
        let current = (0..base.states.len())
            .map(|index| {
                values
                    .block_nav(base.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let mut delta = link.delta.clone();
        if delta.covariance.norm_squared() <= 1e-30 && delta.delta_time.is_finite() {
            let dt = delta.delta_time.max(1e-9);
            for axis in 0..3 {
                delta.covariance[(axis, axis)] =
                    base.imu_noise.accel_density.powi(2) * dt.powi(3) / 3.0;
                delta.covariance[(3 + axis, 3 + axis)] = base.imu_noise.gyro_density.powi(2) * dt;
                delta.covariance[(6 + axis, 6 + axis)] = base.imu_noise.accel_density.powi(2) * dt;
            }
        }
        let from = &current[link.from_index];
        let to = &current[link.to_index];
        let jac_from_owned = if values.state_is_linearized(link.from_index).unwrap_or(false) {
            values.state_linearized_nav(link.from_index)?
        } else {
            from.clone()
        };
        let jac_to_owned = if values.state_is_linearized(link.to_index).unwrap_or(false) {
            values.state_linearized_nav(link.to_index)?
        } else {
            to.clone()
        };
        let factor = if base.scalar_mode == ScalarMode::UpstreamF32 {
            whitened_preintegration_factor_upstream_f32_fej_mode(
                from,
                to,
                &jac_from_owned,
                &jac_to_owned,
                &delta,
                base.gravity_world,
                values.state_is_linearized(link.from_index).unwrap_or(false)
                    || values.state_is_linearized(link.to_index).unwrap_or(false),
            )
            .ok()?
        } else {
            whitened_preintegration_factor(from, to, &delta, base.gravity_world).ok()?
        };
        let mut jacobian = DMatrix::zeros(factor.residual.len(), base.state_dof());
        jacobian
            .view_mut(
                (0, base.state_offset(link.from_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
        jacobian
            .view_mut(
                (0, base.state_offset(link.to_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
        Some(
            WhitenedFactorRowStack::new(
                jacobian,
                DMatrix::zeros(factor.residual.len(), 0),
                factor.residual,
            )?
            .with_kind(FactorKind::Imu)
            .with_imu_link_offsets(from_offset, to_offset),
        )
    }
}

fn emit_imu_factor_input_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) {
    let policy = diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_factor_inputs.as_ref() else {
        return;
    };
    let target_frame = policy.diagnostic_frame;
    let target_link = policy.diagnostic_imu_link;
    let target_iteration = policy.diagnostic_imu_iteration;
    if target_frame != active_frame_id || from_index != target_link {
        return;
    }
    // Preserve the historical one-shot behavior when the selector is absent.
    // With an explicit selector, only the requested LM linearization is
    // eligible; the create_new open below still prevents duplicate writes
    // from diagnostic-only relinearizations at that same iteration.
    if let Some(target_iteration) = target_iteration {
        if iteration != Some(target_iteration) {
            return;
        }
    }
    let Some(record) = build_imu_factor_input_diagnostic(
        from_state,
        to_state,
        active_frame_id,
        from_index,
        to_index,
        from,
        to,
        jac_from,
        jac_to,
        delta,
        gravity_world,
    ) else {
        return;
    };
    let mut record = record;
    if let Some(iteration) = iteration {
        record["iteration"] = json!(iteration);
    }
    let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(path) else {
        return;
    };
    let _ = serde_json::to_writer_pretty(&mut file, &record);
    let _ = file.write_all(b"\n");
}

const IMU_F64_CAPTURE_MAX_BYTES: usize = 2 * 1024 * 1024;

#[derive(Default)]
struct ImuF64CaptureWriterState {
    initialized: bool,
    bytes_written: usize,
    records_written: usize,
    disabled: bool,
}

static IMU_F64_CAPTURE_WRITER: OnceLock<Mutex<ImuF64CaptureWriterState>> = OnceLock::new();

/// Build the observation-only payload from the values that have already
/// crossed the f64 `scaled_axis` call.  In particular, this function does not
/// call `scaled_axis`, compute a norm, or derive an angle.  Nalgebra does not
/// expose those internal intermediates, so serializing a recomputation as if
/// it were an internal operand would make the capture provenance false.
pub(super) fn build_imu_unwhitened_residual_f64_capture(
    active_frame_id: Option<u64>,
    from_state: &WindowState,
    to_state: &WindowState,
    link_index: usize,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    rotation_product: &UnitQuaternion<f64>,
    rotation: Vector3<f64>,
) -> Option<JsonValue> {
    let active_frame_id = active_frame_id?;
    let product = rotation_product.quaternion();
    let bits = |value: f64| format!("{:016x}", value.to_bits());
    let mut record = json!({
        "schema": "basalt.imu_r_f64_inline_capture.v1",
        "frame_id": active_frame_id,
        "active_frame_id": active_frame_id,
        "link_index": link_index,
        "from_index": from_index,
        "to_index": to_index,
        "from_frame_id": from_state.frame_id,
        "to_frame_id": to_state.frame_id,
        "capture_proof": {
            "origin": "same_inline_computation",
            "producer": "imu_unwhitened_residual",
            "product_source": "same_inline_rotation_product",
            "result_source": "same_inline_scaled_axis_result",
            "product_same_call": true,
            "result_same_call": true,
            "norm_captured": false,
            "angle_captured": false,
            "not_recomputed": true,
        },
        "product_xyzw_f64_bits": [
            bits(product.i),
            bits(product.j),
            bits(product.k),
            bits(product.w),
        ],
        "observed_r_R_f64_bits": [bits(rotation.x), bits(rotation.y), bits(rotation.z)],
    });
    if let Some(iteration) = iteration {
        record["iteration"] = json!(iteration);
    }
    Some(record)
}

/// Append one initial-window f64 IMU observation, bounded so an accidentally
/// broad diagnostic selection cannot consume unbounded disk.  The writer is
/// process-local and only initialized after an explicit opt-in path reaches
/// the initial-diagnostics scope; default solver runs never create the file.
fn emit_imu_unwhitened_residual_f64_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    rotation_product: &UnitQuaternion<f64>,
    rotation: Vector3<f64>,
) {
    let policy = diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_unwhitened_f64.as_ref() else {
        return;
    };
    if !initial_imu_f64_capture_scope_active() {
        return;
    }
    let Some(link_index) = initial_imu_f64_capture_link_index() else {
        return;
    };
    let Some(record) = build_imu_unwhitened_residual_f64_capture(
        active_frame_id,
        from_state,
        to_state,
        link_index,
        from_index,
        to_index,
        iteration,
        rotation_product,
        rotation,
    ) else {
        return;
    };
    let Ok(mut line) = serde_json::to_vec(&record) else {
        return;
    };
    line.push(b'\n');
    if line.len() > IMU_F64_CAPTURE_MAX_BYTES {
        return;
    }

    let writer =
        IMU_F64_CAPTURE_WRITER.get_or_init(|| Mutex::new(ImuF64CaptureWriterState::default()));
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    if writer.disabled {
        return;
    }
    if !writer.initialized {
        if let Some(parent) = path.parent() {
            if fs::create_dir_all(parent).is_err() {
                writer.disabled = true;
                return;
            }
        }
        // Never truncate an existing artifact.  A reused path is treated as
        // a failed capture and the caller must provide a fresh path.
        if OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .is_err()
        {
            writer.disabled = true;
            return;
        }
        writer.initialized = true;
    }
    if writer
        .bytes_written
        .checked_add(line.len())
        .is_none_or(|total| total > IMU_F64_CAPTURE_MAX_BYTES)
    {
        writer.disabled = true;
        return;
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        writer.disabled = true;
        return;
    };
    if file.write_all(&line).is_err() {
        writer.disabled = true;
        return;
    }
    writer.bytes_written += line.len();
    writer.records_written += 1;
}

/// Materialize the complete f32 IMU input boundary for a local-block audit.
/// The current state owns the residual, while the FEJ/linearized state owns
/// the analytic Jacobian whenever its corresponding `linearized` flag is set.
/// This helper is called only when a diagnostic environment variable is set;
/// no payload or stage product is constructed on the normal solver path.
fn build_imu_factor_input_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Option<JsonValue> {
    let Ok((raw, jacobian, whitener, whitened, dense)) = preintegration_f32_stages_fej_mode(
        from,
        to,
        jac_from,
        jac_to,
        delta,
        gravity_world,
        from_state.linearized || to_state.linearized,
    ) else {
        return None;
    };
    // `ImuBlock::linearizeImu` first evaluates the residual/Jacobian at the
    // FEJ states and then, when either endpoint is already linearized,
    // replaces only the residual with the current-state evaluation.  Keep
    // both raw residuals in the payload so the input-stage record can explain
    // that split without re-running or modifying the production factor.
    let raw_fej = preintegration_residual_f32(jac_from, jac_to, delta, gravity_world);
    let bits = |x: f32| format!("{:08x}", x.to_bits());
    let vec_bits = |v: &[f32]| v.iter().copied().map(bits).collect::<Vec<_>>();
    let f64vec_bits = |v: &[f64]| v.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>();
    let mat_bits = |rows: usize, cols: usize, f: &dyn Fn(usize, usize) -> f32| {
        (0..cols)
            .flat_map(|c| (0..rows).map(move |r| bits(f(r, c))))
            .collect::<Vec<_>>()
    };
    let state_bits = |s: &BasaltNavState| {
        let mut v = Vec::with_capacity(16);
        v.extend(s.imu_to_world.translation.iter().map(|x| bits(*x as f32)));
        v.extend(
            [
                s.imu_to_world.rotation.w,
                s.imu_to_world.rotation.i,
                s.imu_to_world.rotation.j,
                s.imu_to_world.rotation.k,
            ]
            .into_iter()
            .map(|x| bits(x as f32)),
        );
        v.extend(s.velocity_world_m_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.gyro_bias_rad_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.accel_bias_m_s2.iter().map(|x| bits(*x as f32)));
        v
    };
    // Keep the historical compact state lane (`wxyz`) above for consumers
    // that already read v2 artifacts, but also expose the native
    // PoseVelBias spelling (`xyzw`) in v3.  The latter matches Basalt's
    // serialized `unit_quaternion()` order and makes the state record
    // self-describing instead of requiring a consumer-side permutation.
    let state_xyzw_bits = |s: &BasaltNavState| {
        let mut v = Vec::with_capacity(16);
        v.extend(s.imu_to_world.translation.iter().map(|x| bits(*x as f32)));
        v.extend(
            [
                s.imu_to_world.rotation.i,
                s.imu_to_world.rotation.j,
                s.imu_to_world.rotation.k,
                s.imu_to_world.rotation.w,
            ]
            .into_iter()
            .map(|x| bits(x as f32)),
        );
        v.extend(s.velocity_world_m_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.gyro_bias_rad_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.accel_bias_m_s2.iter().map(|x| bits(*x as f32)));
        v
    };
    let state_record = |frame_state: &WindowState, state: &BasaltNavState| {
        let q = state.imu_to_world.rotation.quaternion();
        json!({
            "t_ns": frame_state.timestamp_ns,
            "translation_f32_bits": f64vec_bits(state.imu_to_world.translation.as_slice()),
            "quaternion_xyzw_f32_bits": f64vec_bits(&[q.i, q.j, q.k, q.w]),
            "velocity_f32_bits": f64vec_bits(state.velocity_world_m_s.as_slice()),
            "bias_gyro_f32_bits": f64vec_bits(state.gyro_bias_rad_s.as_slice()),
            "bias_accel_f32_bits": f64vec_bits(state.accel_bias_m_s2.as_slice()),
            "f32_bits": state_xyzw_bits(state),
            "scalar_count": 16,
        })
    };
    let matrix3_bits = |m: &Matrix3<f64>| mat_bits(3, 3, &|r, c| m[(r, c)] as f32);
    let delta_rotation_matrix = delta.delta_rotation.to_rotation_matrix().into_inner();
    let delta_rotation_matrix_bits = mat_bits(3, 3, &|r, c| delta_rotation_matrix[(r, c)] as f32);
    let mut delta_dbg = Vec::with_capacity(27);
    let mut delta_dba = Vec::with_capacity(27);
    for col in 0..3 {
        for row in 0..9 {
            delta_dbg.push(match row {
                0..=2 => delta.jacobian_position_gyro_bias[(row, col)],
                3..=5 => delta.jacobian_rotation_gyro_bias[(row - 3, col)],
                _ => delta.jacobian_velocity_gyro_bias[(row - 6, col)],
            });
            delta_dba.push(match row {
                0..=2 => delta.jacobian_position_accel_bias[(row, col)],
                3..=5 => 0.0,
                _ => delta.jacobian_velocity_accel_bias[(row - 6, col)],
            });
        }
    }
    let dt_f32 = delta.delta_time as f32;
    let dt_ns = (dt_f32 * 1.0e9_f32).round() as i64;
    Some(json!({
        "schema": "basalt.imu_factor_input_diagnostic.v3",
        "active_frame_id": active_frame_id,
        "from_index": from_index,
        "to_index": to_index,
        "from_frame_id": from_state.frame_id,
        "to_frame_id": to_state.frame_id,
        "start_t_ns": from_state.timestamp_ns,
        "start_t": from_state.timestamp_ns,
        "end_t_ns": from_state.timestamp_ns.saturating_add(dt_ns),
        "dt_ns": dt_ns,
        "dt": dt_f32,
        "delta_time_bits": bits(dt_f32),
        "current": {
            "from": state_record(from_state, from),
            "to": state_record(to_state, to),
        },
        "fej": {
            "from": state_record(from_state, jac_from),
            "to": state_record(to_state, jac_to),
        },
        "states": {
            "from_fej": state_record(from_state, jac_from),
            "from_current": state_record(from_state, from),
            "to_fej": state_record(to_state, jac_to),
            "to_current": state_record(to_state, to),
        },
        "current_from_state_f32_bits": state_bits(from),
        "current_to_state_f32_bits": state_bits(to),
        "from_current_state_f32_bits": state_bits(from),
        "to_current_state_f32_bits": state_bits(to),
        "from_state_f32_bits": state_bits(from),
        "to_state_f32_bits": state_bits(to),
        "linearized_from_state_f32_bits": state_bits(jac_from),
        "linearized_to_state_f32_bits": state_bits(jac_to),
        "fej_from_state_f32_bits": state_bits(jac_from),
        "fej_to_state_f32_bits": state_bits(jac_to),
        "from_linearized": from_state.linearized,
        "to_linearized": to_state.linearized,
        "from_fej": from_state.linearized,
        "to_fej": to_state.linearized,
        "linearized_flags": {
            "from": from_state.linearized,
            "to": to_state.linearized,
        },
        "preintegrated": {
            "delta_rotation_matrix_f32_bits": delta_rotation_matrix_bits,
            "delta_rotation_f32_bits": f64vec_bits(&[
                delta.delta_rotation.w,
                delta.delta_rotation.i,
                delta.delta_rotation.j,
                delta.delta_rotation.k,
            ]),
            "delta_position_f32_bits": f64vec_bits(delta.delta_position.as_slice()),
            "delta_velocity_f32_bits": f64vec_bits(delta.delta_velocity.as_slice()),
            "bias_gyro_f32_bits": f64vec_bits(delta.bias_gyro.as_slice()),
            "bias_accel_f32_bits": f64vec_bits(delta.bias_accel.as_slice()),
            "jacobian_rotation_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_rotation_gyro_bias),
            "jacobian_velocity_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_velocity_gyro_bias),
            "jacobian_velocity_accel_bias_f32_bits": matrix3_bits(&delta.jacobian_velocity_accel_bias),
            "jacobian_position_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_position_gyro_bias),
            "jacobian_position_accel_bias_f32_bits": matrix3_bits(&delta.jacobian_position_accel_bias),
            "covariance_f32_bits": mat_bits(9, 9, &|r, c| delta.covariance[(r, c)] as f32),
        },
        "delta_t_ns": (dt_f32 * 1.0e9_f32).round() as i64,
        "delta_rotation_matrix_f32_bits": delta_rotation_matrix_bits,
        "delta_rotation_f32_bits": f64vec_bits(&[
            delta.delta_rotation.w,
            delta.delta_rotation.i,
            delta.delta_rotation.j,
            delta.delta_rotation.k,
        ]),
        "delta_velocity_f32_bits": f64vec_bits(delta.delta_velocity.as_slice()),
        "delta_position_f32_bits": f64vec_bits(delta.delta_position.as_slice()),
        "delta_covariance_f32_bits": f64vec_bits(delta.covariance.as_slice()),
        "d_state_d_bg_f32_bits": f64vec_bits(&delta_dbg),
        "d_state_d_ba_f32_bits": f64vec_bits(&delta_dba),
        "delta_d_state_d_bg_f32_bits": f64vec_bits(&delta_dbg),
        "delta_d_state_d_ba_f32_bits": f64vec_bits(&delta_dba),
        "covariance_f32_bits": mat_bits(9, 9, &|r, c| delta.covariance[(r, c)] as f32),
        "d_state_update_trace": delta.update_trace.iter().map(|u| json!({
            "t_ns": u.t_ns,
            "f_f32_bits": u.f.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "g_f32_bits": u.g.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "f_old_bg_f32_bits": u.f_old_bg.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "d_state_d_bg_f32_bits": u.d_state_d_bg.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "d_state_d_ba_f32_bits": u.d_state_d_ba.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "raw_residual_fej_bits": vec_bits(raw_fej.as_slice()),
        "raw_residual_bits": vec_bits(raw.as_slice()),
        "raw_jacobian_bits": mat_bits(9, 30, &|r, c| jacobian[(r, c)]),
        "sqrt_information_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "whitened_residual_bits": vec_bits(whitened.as_slice()),
        "whitened_jacobian_bits": mat_bits(9, 30, &|r, c| dense[(r, c)]),
        "raw_r_bits": vec_bits(raw.as_slice()),
        "raw_j_bits": mat_bits(9, 30, &|r, c| jacobian[(r, c)]),
        "w_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "w_f32_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "raw_r_fej_bits": vec_bits(raw_fej.as_slice()),
        "whitened_r_bits": vec_bits(whitened.as_slice()),
        "whitened_j_bits": mat_bits(9, 30, &|r, c| dense[(r, c)]),
        "native_expr_trace": preintegration_f32_expression_trace(from, to, delta, gravity_world),
        "fej_rotation_producer_trace": preintegration_f32_rotation_producer_trace(
            jac_from, jac_to, delta,
        ),
    }))
}

fn vector3_array(value: Vector3<f64>) -> [f64; 3] {
    [value.x, value.y, value.z]
}

pub(super) const IMU_F64_ATAN_SMALL_MAX_RATIO: f64 = f64::from_bits(0x3eb0_0000_0000_0000);
const IMU_F64_ATAN_SMALL_NORMALIZE_X_FLOOR_EXP: i32 = -500;

const fn imu_f64_atan_small_floor_exponent(value: f64) -> i32 {
    let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent != 0 {
        exponent - 1023
    } else {
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        -1074 + (63 - mantissa.leading_zeros() as i32)
    }
}

fn imu_f64_atan_small_exact_power_of_two(exponent: i32) -> f64 {
    assert!((0..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn imu_f64_atan_small_scaled_operands(y: f64, x: f64) -> (f64, f64, i32) {
    let shift =
        (IMU_F64_ATAN_SMALL_NORMALIZE_X_FLOOR_EXP - imu_f64_atan_small_floor_exponent(x)).max(0);
    let scale = imu_f64_atan_small_exact_power_of_two(shift);
    (y * scale, x * scale, shift)
}

/// Grouped small-angle atan2 control used only by the diagnostic raw f64 IMU
/// rotation observation.  The original nalgebra `scaled_axis` result remains
/// the fallback for every rejected, zero, or non-finite domain.
pub(super) fn imu_f64_atan_small_grouped_positive(y: f64, x: f64) -> Option<f64> {
    if !(y.is_finite() && x.is_finite()) || y < 0.0 || x <= 0.0 {
        return None;
    }
    if y == 0.0 {
        return Some(y);
    }
    let unscaled_ratio = y / x;
    if unscaled_ratio > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return None;
    }
    let (scaled_y, scaled_x, _) = imu_f64_atan_small_scaled_operands(y, x);
    let q_hi = scaled_y / scaled_x;
    if q_hi > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return None;
    }
    let residual = (-q_hi).fused_mul_add(scaled_x, scaled_y);
    let q_lo = residual / scaled_x;
    let q2 = q_hi * q_hi;
    let q3 = q2 * q_hi;
    let q5 = q3 * q2;
    let correction = q_lo - q3 / 3.0 + q5 / 5.0;
    Some(q_hi + correction)
}

/// Reproduce the candidate only for the diagnostic f64 IMU raw rotation.
/// This deliberately preserves nalgebra's axis sign convention (`w >= 0`),
/// uses `abs(w)` for the angle domain, and returns the original nalgebra
/// `scaled_axis` result whenever the candidate cannot accept the domain.
pub(super) fn imu_f64_scaled_axis_candidate(rotation: &UnitQuaternion<f64>) -> Vector3<f64> {
    let mut vector = rotation.quaternion().imag().clone_owned();
    let scalar = rotation.quaternion().scalar();
    if !(scalar >= 0.0) {
        vector = -vector;
    }
    let w = scalar.abs();
    let norm_squared = vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
    let norm = norm_squared.sqrt();
    if norm == 0.0 || norm / w > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return rotation.scaled_axis();
    }
    let Some(half_angle) = imu_f64_atan_small_grouped_positive(norm, w) else {
        return rotation.scaled_axis();
    };
    (vector / norm) * (half_angle * 2.0)
}

/// The literal upstream unwhitened residual contract.  Whitening is kept in
/// `imu::factors::whitened_preintegration_factor`; this adapter mirrors only
/// the raw rows so trace consumers can compare them against the oracle.
pub(super) fn imu_unwhitened_residual(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> (
    Vector3<f64>,
    Vector3<f64>,
    Vector3<f64>,
    UnitQuaternion<f64>,
) {
    let (dr, dv, dp) = delta.corrected(from.gyro_bias_rad_s, from.accel_bias_m_s2);
    let dt = delta.delta_time;
    let r_iw = from.imu_to_world.rotation.inverse();
    // Keep the exact upstream multiplication expression/order intact, but
    // retain its already-computed quaternion so the opt-in f64 observer can
    // serialize the actual product instead of reconstructing it later.
    let rotation_product = r_iw * to.imu_to_world.rotation * dr.inverse();
    let rotation = imu_f64_scaled_axis_candidate(&rotation_product);
    let velocity =
        r_iw * (to.velocity_world_m_s - from.velocity_world_m_s - gravity_world * dt) - dv;
    let position = r_iw
        * (to.imu_to_world.translation
            - from.imu_to_world.translation
            - from.velocity_world_m_s * dt
            - gravity_world * (0.5 * dt * dt))
        - dp;
    (position, rotation, velocity, rotation_product)
}

pub(super) fn bias_walk_factor(
    state: &DVector<f64>,
    link: WindowImuLink,
    state_dof: usize,
    from_offset: Option<usize>,
    to_offset: Option<usize>,
    noise: BiasRandomWalkNoise,
    scalar_mode: ScalarMode,
) -> Option<WhitenedFactorRowStack> {
    let from_offset = from_offset?;
    let to_offset = to_offset?;
    if from_offset + NAV_STATE_DOF > state_dof || to_offset + NAV_STATE_DOF > state_dof {
        return None;
    }
    let from = decode_state_with_mode(state.rows(from_offset, NAV_STATE_DOF), scalar_mode);
    let to = decode_state_with_mode(state.rows(to_offset, NAV_STATE_DOF), scalar_mode);
    let factor = if scalar_mode == ScalarMode::UpstreamF32 {
        whitened_bias_random_walk_factor_upstream_f32(&from, &to, link.delta.delta_time, noise)
            .ok()?
    } else {
        whitened_bias_random_walk_factor(&from, &to, link.delta.delta_time, noise).ok()?
    };
    let mut jacobian = DMatrix::zeros(factor.residual.len(), state_dof);
    jacobian
        .view_mut((0, from_offset), (factor.residual.len(), NAV_STATE_DOF))
        .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
    jacobian
        .view_mut((0, to_offset), (factor.residual.len(), NAV_STATE_DOF))
        .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
    WhitenedFactorRowStack::new(
        jacobian,
        DMatrix::zeros(factor.residual.len(), 0),
        factor.residual,
    )
    .map(|factor| {
        factor
            .with_kind(FactorKind::Bias)
            .with_imu_link_offsets(from_offset, to_offset)
    })
}
