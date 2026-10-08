//! Q2 / full70 oracle payloads, visual-prefix tracing and diagnostic probes.

use super::*;

/// Optional post-QR Q2/ABS sidecar.  This is intentionally attached to the
/// Q2 helper rather than the factor reducer so a capture cannot accidentally
/// report the wrong pre-QR boundary.  It is disabled unless the caller sets
/// `VISLOC_BASALT_Q2_ABS_HB` and never participates in estimator arithmetic.
#[inline]
pub(super) fn emit_q2_abs_hb_diagnostic(
    jacobian: &DMatrix<f32>,
    rhs: &DVector<f32>,
    h: &DMatrix<f32>,
    b: &DVector<f32>,
) {
    let Some(path) = crate::vio::window::diagnostic_env_snapshot()
        .q2_abs_hb
        .as_ref()
    else {
        return;
    };
    static EVENT_COUNT: AtomicUsize = AtomicUsize::new(0);
    let event_ordinal = EVENT_COUNT.fetch_add(1, Ordering::Relaxed);
    let payload = json!({
        "schema": "visloc.m7im15.rust_q2_abs_hb.v1",
        "event_ordinal": event_ordinal,
        "q2_jacobian_f32": diagnostic_f32_matrix(jacobian),
        "q2_rhs_f32": diagnostic_f32_vector(rhs),
        "aom_abs_h_f32": diagnostic_f32_matrix(h),
        "aom_abs_b_f32": b
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    });
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = serde_json::to_writer(&mut file, &payload);
        let _ = file.write_all(b"\n");
    }
}

/// Build the local 15-row/30-column view owned by one semantic IMU+bias pair.
/// The production reducer still consumes the global state-width matrix; this
/// sidecar records the corresponding local dynamic product so padding/state
/// width effects can be compared without a detail trace.
pub(super) fn imu_local_block_diagnostic(
    local_jacobian: &DMatrix<f32>,
    residual: &DVector<f32>,
    global_jacobian: &DMatrix<f32>,
    offsets: ImuLinkOffsets,
    imu_input_diagnostic: Option<serde_json::Value>,
) -> ImuLocalBlockDiagnostic {
    let active_offsets = vec![offsets.start, offsets.end];
    let (local_h, local_b) = local_imu_h_b_15x30(local_jacobian, residual);
    let global_h = global_jacobian.transpose() * global_jacobian;
    let global_b = global_jacobian.transpose() * residual;
    let mut padded_h = DMatrix::zeros(global_jacobian.ncols(), global_jacobian.ncols());
    let mut padded_b = DVector::zeros(global_jacobian.ncols());
    for (local_row_block, &global_row_offset) in active_offsets.iter().enumerate() {
        for (local_column_block, &global_column_offset) in active_offsets.iter().enumerate() {
            padded_h
                .view_mut(
                    (global_row_offset, global_column_offset),
                    (AOM_NAV_DOF, AOM_NAV_DOF),
                )
                .copy_from(&local_h.view(
                    (
                        local_row_block * AOM_NAV_DOF,
                        local_column_block * AOM_NAV_DOF,
                    ),
                    (AOM_NAV_DOF, AOM_NAV_DOF),
                ));
        }
    }
    for (block, &offset) in active_offsets.iter().enumerate() {
        padded_b
            .rows_mut(offset, AOM_NAV_DOF)
            .copy_from(&local_b.rows(block * AOM_NAV_DOF, AOM_NAV_DOF));
    }
    let mut h_mismatches = 0;
    for row in 0..local_h.nrows() {
        for column in 0..local_h.ncols() {
            let global_row = active_offsets[row / AOM_NAV_DOF] + row % AOM_NAV_DOF;
            let global_column = active_offsets[column / AOM_NAV_DOF] + column % AOM_NAV_DOF;
            if local_h[(row, column)].to_bits() != global_h[(global_row, global_column)].to_bits() {
                h_mismatches += 1;
            }
        }
    }
    let mut b_mismatches = 0;
    for column in 0..local_b.len() {
        let global_column = active_offsets[column / AOM_NAV_DOF] + column % AOM_NAV_DOF;
        if local_b[column].to_bits() != global_b[global_column].to_bits() {
            b_mismatches += 1;
        }
    }
    let padded_h_mismatches = (0..global_h.nrows())
        .flat_map(|row| (0..global_h.ncols()).map(move |column| (row, column)))
        .filter(|&(row, column)| {
            padded_h[(row, column)].to_bits() != global_h[(row, column)].to_bits()
        })
        .count();
    let padded_b_mismatches = (0..global_b.len())
        .filter(|&column| padded_b[column].to_bits() != global_b[column].to_bits())
        .count();
    ImuLocalBlockDiagnostic {
        active_offsets,
        local_jacobian: local_jacobian.clone(),
        residual: residual.clone(),
        local_h,
        local_b,
        imu_input_diagnostic,
        local_vs_global_h_mismatches: h_mismatches,
        local_vs_global_b_mismatches: b_mismatches,
        local_vs_global_padded_h_mismatches: padded_h_mismatches,
        local_vs_global_padded_b_mismatches: padded_b_mismatches,
    }
}

pub(super) fn diagnostic_f32_matrix(value: &DMatrix<f32>) -> serde_json::Value {
    json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": value.iter().map(|entry| format!("{:08x}", entry.to_bits())).collect::<Vec<_>>(),
    })
}

pub(super) fn diagnostic_f32_vector(value: &DVector<f32>) -> serde_json::Value {
    json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": value.iter().map(|entry| format!("{:08x}", entry.to_bits())).collect::<Vec<_>>(),
    })
}

/// Serialize an f64-owned matrix/vector at the exact f32 boundary used by the
/// UpstreamF32 reducer.  These helpers are crate-visible only so the active
/// window can put the stored prior/FEJ source beside the ordered factor oracle;
/// they are never called from the nominal solver path.
pub(crate) fn full70_f32_bits_matrix(value: &DMatrix<f64>) -> serde_json::Value {
    json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": value
            .iter()
            .map(|entry| format!("{:08x}", (*entry as f32).to_bits()))
            .collect::<Vec<_>>(),
    })
}

pub(crate) fn full70_f32_bits_vector(value: &DVector<f64>) -> serde_json::Value {
    json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": value
            .iter()
            .map(|entry| format!("{:08x}", (*entry as f32).to_bits()))
            .collect::<Vec<_>>(),
    })
}

const FULL70_ORACLE_REQUIRED_PROVENANCE: &[&str] = &[
    "VISLOC_BASALT_FULL70_EXECUTABLE_SHA256",
    "VISLOC_BASALT_FULL70_SOURCE_SHA256",
    "VISLOC_BASALT_FULL70_CONFIG_SHA256",
    "VISLOC_BASALT_FULL70_CALIBRATION_SHA256",
    "VISLOC_BASALT_FULL70_INPUT_SHA256",
];

pub(super) fn full70_hash_like(value: &str) -> bool {
    (value.len() == 40 || value.len() == 64) && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn full70_oracle_provenance_from_env() -> Result<serde_json::Value, ImuReductionError> {
    let mut required = serde_json::Map::new();
    for &key in FULL70_ORACLE_REQUIRED_PROVENANCE {
        let value = std::env::var(key)
            .ok()
            .filter(|value| full70_hash_like(value))
            .ok_or(ImuReductionError::Full70OracleProvenanceMissing { key })?;
        required.insert(
            key.to_owned(),
            serde_json::Value::String(value.to_ascii_lowercase()),
        );
    }
    let optional = [
        "VISLOC_BASALT_FULL70_FEATURES",
        "VISLOC_BASALT_FULL70_TARGET",
        "VISLOC_BASALT_FULL70_DIRTY_SCOPE_SHA256",
    ]
    .into_iter()
    .filter_map(|key| {
        std::env::var(key)
            .ok()
            .map(|value| (key.to_owned(), json!(value)))
    })
    .collect::<serde_json::Map<_, _>>();
    Ok(json!({
        "status": "declared_external_recompute_required",
        "required_hashes": required,
        "optional": optional,
        "verification": "the harness must recompute every hash from the declared path/source and reject declaration-only evidence",
    }))
}

pub(super) fn full70_checked_f32_bits(
    value: f64,
    index: usize,
) -> Result<String, ImuReductionError> {
    let value = value as f32;
    value
        .is_finite()
        .then(|| format!("{:08x}", value.to_bits()))
        .ok_or(ImuReductionError::Full70OracleInvalid { index })
}

fn full70_checked_matrix_bits(
    value: &DMatrix<f64>,
    index: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let bits = value
        .iter()
        .map(|entry| full70_checked_f32_bits(*entry, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "rows": value.nrows(),
        "cols": value.ncols(),
        "layout": "column_major",
        "bits": bits,
    }))
}

fn full70_checked_vector_bits(
    value: &DVector<f64>,
    index: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let bits = value
        .iter()
        .map(|entry| full70_checked_f32_bits(*entry, index))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "rows": value.len(),
        "cols": 1,
        "layout": "column_major_vector",
        "bits": bits,
    }))
}

fn full70_f32_storage_bits(values: &[f32]) -> serde_json::Value {
    json!({
        "rows": values.len(),
        "cols": 1,
        "layout": "flat_storage",
        "bits": values
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    })
}

pub(super) const fn full70_factor_kind_name(kind: FactorKind) -> &'static str {
    match kind {
        FactorKind::Generic => "Generic",
        FactorKind::Prior => "Prior",
        FactorKind::Visual => "Visual",
        FactorKind::Imu => "Imu",
        FactorKind::Bias => "Bias",
    }
}

/// The full70 capture is intentionally strict: accepting a count-correct but
/// interleaved or metadata-incomplete vector would make the resulting artifact
/// impossible to compare to the native source order.  The checked reducer has
/// more general validation; this additional frame-4 contract is the oracle
/// boundary itself.
pub(super) fn validate_full70_frame4_factors(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<(), ImuReductionError> {
    validate_factor_shapes(factors, state_dof)?;
    validate_imu_pairing(factors, state_dof)?;
    if state_dof != 75 || factors.len() != 70 {
        return Err(ImuReductionError::Full70OracleInvalid {
            index: factors.len(),
        });
    }
    let Some(prior) = factors.first() else {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    };
    if prior.kind != FactorKind::Prior || prior.rows() != 15 || prior.landmark_jacobian.ncols() != 0
    {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    }
    let mut visual_end = 1usize;
    let mut previous_landmark_index = None;
    while visual_end < factors.len() && factors[visual_end].kind == FactorKind::Visual {
        let factor = &factors[visual_end];
        let Some(metadata) = factor.landmark_metadata else {
            return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
        };
        if factor.landmark_jacobian.ncols() != 3
            || previous_landmark_index.is_some_and(|previous| metadata.landmark_index <= previous)
        {
            return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
        }
        previous_landmark_index = Some(metadata.landmark_index);
        visual_end += 1;
    }
    if visual_end - 1 != 61 {
        return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
    }
    let suffix = &factors[visual_end..];
    if suffix.len() != 8
        || suffix
            .chunks_exact(2)
            .any(|pair| pair[0].kind != FactorKind::Imu || pair[1].kind != FactorKind::Bias)
    {
        return Err(ImuReductionError::Full70OracleInvalid { index: visual_end });
    }
    let row_count = factors.iter().try_fold(0usize, |sum, factor| {
        sum.checked_add(factor.rows())
            .ok_or(ImuReductionError::Full70OracleInvalid { index: sum })
    })?;
    if row_count != 1243 {
        return Err(ImuReductionError::Full70OracleInvalid { index: row_count });
    }
    Ok(())
}

fn full70_imu_local_payload(
    imu: &WhitenedFactorRowStack,
    bias: &WhitenedFactorRowStack,
    index: usize,
    state_dof: usize,
) -> Result<serde_json::Value, ImuReductionError> {
    let offsets = imu
        .imu_link_offsets
        .ok_or(ImuReductionError::Full70OracleInvalid { index })?;
    validate_imu_offsets(index, offsets, state_dof)?;
    let mut local_jacobian = DMatrix::<f32>::zeros(IMU_LOCAL_ROWS, IMU_LOCAL_COLS);
    for (local_row, source) in imu.state_jacobian.row_iter().enumerate() {
        for block in 0..2 {
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            for local_column in 0..AOM_NAV_DOF {
                let value = source[global_offset + local_column] as f32;
                if !value.is_finite() {
                    return Err(ImuReductionError::Full70OracleInvalid { index });
                }
                local_jacobian[(local_row, block * AOM_NAV_DOF + local_column)] = value;
            }
        }
    }
    for (local_row, source) in bias.state_jacobian.row_iter().enumerate() {
        for block in 0..2 {
            let global_offset = if block == 0 {
                offsets.start
            } else {
                offsets.end
            };
            for local_column in 0..AOM_NAV_DOF {
                let value = source[global_offset + local_column] as f32;
                if !value.is_finite() {
                    return Err(ImuReductionError::Full70OracleInvalid { index });
                }
                local_jacobian[(9 + local_row, block * AOM_NAV_DOF + local_column)] = value;
            }
        }
    }
    let local_residual = DVector::from_iterator(
        IMU_LOCAL_ROWS,
        imu.residual
            .iter()
            .chain(bias.residual.iter())
            .map(|value| *value as f32),
    );
    if local_residual.iter().any(|value| !value.is_finite()) {
        return Err(ImuReductionError::Full70OracleInvalid { index });
    }
    Ok(json!({
        "offsets": { "start": offsets.start, "end": offsets.end },
        "shape": [IMU_LOCAL_ROWS, IMU_LOCAL_COLS],
        "jacobian_f32": diagnostic_f32_matrix(&local_jacobian),
        "residual_f32": diagnostic_f32_vector(&local_residual),
    }))
}

fn full70_factor_record(
    factor: &WhitenedFactorRowStack,
    ordinal: usize,
    row_start: usize,
    paired_imu: Option<serde_json::Value>,
) -> Result<serde_json::Value, ImuReductionError> {
    let state_jacobian_f32 = full70_checked_matrix_bits(&factor.state_jacobian, ordinal)?;
    let landmark_jacobian_f32 = full70_checked_matrix_bits(&factor.landmark_jacobian, ordinal)?;
    let residual_f32 = full70_checked_vector_bits(&factor.residual, ordinal)?;
    let objective_cost_f32_bits = full70_checked_f32_bits(factor.objective_cost, ordinal)?;
    let visual_metadata = factor.landmark_metadata.map(|metadata| {
        json!({
            "landmark_index": metadata.landmark_index,
            "track_id": metadata.track_id,
        })
    });
    let observation_ids = factor.visual_observation_ids.as_ref().map(|observations| {
        observations
            .iter()
            .map(|&(state_index, camera_id)| json!({ "state_index": state_index, "camera_id": camera_id }))
            .collect::<Vec<_>>()
    });
    let paired_bias_ordinal = (factor.kind == FactorKind::Imu).then_some(ordinal + 1);
    let paired_imu_ordinal = (factor.kind == FactorKind::Bias).then_some(ordinal.saturating_sub(1));
    Ok(json!({
        "ordinal": ordinal,
        "kind": full70_factor_kind_name(factor.kind),
        "row_start": row_start,
        "rows": factor.rows(),
        "state_cols": factor.state_jacobian.ncols(),
        "landmark_cols": factor.landmark_jacobian.ncols(),
        "objective_cost_f32_bits": objective_cost_f32_bits,
        "state_jacobian_f32": state_jacobian_f32,
        "landmark_jacobian_f32": landmark_jacobian_f32,
        "residual_f32": residual_f32,
        "source_layout": "column_major_dmatrix_cast_f32",
        "landmark_metadata": visual_metadata.unwrap_or(serde_json::Value::Null),
        "visual_observation_ids": observation_ids.unwrap_or_default(),
        "imu_link_offsets": factor.imu_link_offsets.map(|offsets| json!({
            "start": offsets.start,
            "end": offsets.end,
        })).unwrap_or(serde_json::Value::Null),
        "paired_imu_ordinal": paired_imu_ordinal,
        "paired_bias_ordinal": paired_bias_ordinal,
        "prior_state_columns": factor.prior_state_columns.clone(),
        "imu_input_diagnostic": factor.imu_input_diagnostic.clone(),
        "imu_local_15x30": paired_imu,
    }))
}

fn full70_stage_json(stage: Option<&DiagnosticNormalSystemF32>) -> serde_json::Value {
    stage.map_or(serde_json::Value::Null, |stage| {
        json!({
            "visual": {
                "h": diagnostic_f32_matrix(&stage.visual_h),
                "b": diagnostic_f32_vector(&stage.visual_b),
            },
            "visual_plus_imu": {
                "h": diagnostic_f32_matrix(&stage.visual_imu_h),
                "b": diagnostic_f32_vector(&stage.visual_imu_b),
            },
            "prior": {
                "h": diagnostic_f32_matrix(&stage.prior_h),
                "b": diagnostic_f32_vector(&stage.prior_b),
            },
        })
    })
}

fn full70_reduced_json(
    reduced: &ReducedNormalSystemF32,
    label: &str,
) -> Result<serde_json::Value, ImuReductionError> {
    let compact = if let Some(batch) = reduced.compact_back_substitution.as_ref() {
        let entries = batch
            .entries
            .iter()
            .map(|entry| {
                let len = entry
                    .storage_len()
                    .ok_or(ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    })?;
                let end = entry.storage_offset.checked_add(len).ok_or(
                    ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    },
                )?;
                if end > batch.storage.len() {
                    return Err(ImuReductionError::Full70OracleInvalid {
                        index: entry.landmark_index,
                    });
                }
                Ok(json!({
                    "landmark_index": entry.landmark_index,
                    "track_id": entry.track_id,
                    "storage_offset": entry.storage_offset,
                    "storage_len": len,
                    "state_cols": entry.state_cols,
                    "landmark_cols": entry.landmark_cols,
                    "rank": entry.rank,
                    "eligible": entry.eligible,
                }))
            })
            .collect::<Result<Vec<_>, ImuReductionError>>()?;
        Some(json!({
            "storage": full70_f32_storage_bits(&batch.storage),
            "entries": entries,
        }))
    } else {
        None
    };
    Ok(json!({
        "label": label,
        "h": diagnostic_f32_matrix(&reduced.h),
        "b": diagnostic_f32_vector(&reduced.b),
        "back_substitution_ranks": reduced
            .back_substitution
            .iter()
            .map(|entry| entry.rank)
            .collect::<Vec<_>>(),
        "compact_back_substitution": compact.unwrap_or(serde_json::Value::Null),
        "diagnostic_stages": full70_stage_json(reduced.diagnostic_stages.as_ref()),
        "model_decrease_payload": reduced.model_decrease_payload.is_some(),
    }))
}

pub(super) fn full70_matrix_bits_match(left: &DMatrix<f32>, right: &DMatrix<f32>) -> bool {
    left.shape() == right.shape()
        && left
            .as_slice()
            .iter()
            .zip(right.as_slice())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

pub(super) fn full70_vector_bits_match(left: &DVector<f32>, right: &DVector<f32>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn full70_factor_input_fingerprint(
    factors: &[WhitenedFactorRowStack],
) -> Result<u64, ImuReductionError> {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    for (ordinal, factor) in factors.iter().enumerate() {
        visual_prefix_hash_mix(&mut hash, ordinal as u64);
        visual_prefix_hash_mix(&mut hash, factor.kind as u64);
        visual_prefix_hash_mix(&mut hash, factor.rows() as u64);
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.ncols() as u64);
        visual_prefix_hash_mix(&mut hash, factor.landmark_jacobian.ncols() as u64);
        for value in factor
            .state_jacobian
            .iter()
            .chain(factor.landmark_jacobian.iter())
            .chain(factor.residual.iter())
        {
            let value = *value as f32;
            if !value.is_finite() {
                return Err(ImuReductionError::Full70OracleInvalid { index: ordinal });
            }
            visual_prefix_hash_mix(&mut hash, u64::from(value.to_bits()));
        }
        if let Some(metadata) = factor.landmark_metadata {
            visual_prefix_hash_mix(&mut hash, metadata.landmark_index as u64);
            visual_prefix_hash_mix(&mut hash, metadata.track_id);
        }
        if let Some(offsets) = factor.imu_link_offsets {
            visual_prefix_hash_mix(&mut hash, offsets.start as u64);
            visual_prefix_hash_mix(&mut hash, offsets.end as u64);
        }
    }
    Ok(hash)
}

fn full70_recovery_json(
    factors: &[WhitenedFactorRowStack],
    legacy: &ReducedNormalSystemF32,
    compact: &ReducedNormalSystemF32,
    state_step: Option<&DVector<f64>>,
    tolerance: f64,
) -> Result<serde_json::Value, ImuReductionError> {
    let Some(state_step) = state_step else {
        return Ok(json!({ "status": "not_exposed_without_solver_step" }));
    };
    let Some(compact_batch) = compact.compact_back_substitution.as_ref() else {
        return Err(ImuReductionError::Full70OracleInvalid { index: 0 });
    };
    let mut records = Vec::with_capacity(compact_batch.entries.len());
    for entry in &compact_batch.entries {
        let Some((factor_index, factor)) = factors.iter().enumerate().find(|(_, factor)| {
            factor.landmark_metadata.is_some_and(|metadata| {
                metadata.landmark_index == entry.landmark_index
                    && metadata.track_id == entry.track_id
            })
        }) else {
            return Err(ImuReductionError::Full70OracleInvalid {
                index: entry.landmark_index,
            });
        };
        let legacy_data = legacy.back_substitution.get(factor_index).ok_or(
            ImuReductionError::Full70OracleInvalid {
                index: factor_index,
            },
        )?;
        let legacy_step = back_substitute_landmark_f32_with_track(
            legacy_data,
            state_step,
            tolerance,
            Some(entry.track_id),
        );
        let compact_step = back_substitute_landmark_compact_entry_f32(
            entry,
            &compact_batch.storage,
            state_step,
            tolerance,
        );
        let legacy_bits = legacy_step.as_ref().map(|step| {
            step.iter()
                .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                .collect::<Vec<_>>()
        });
        let compact_bits = compact_step.as_ref().map(|step| {
            step.iter()
                .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                .collect::<Vec<_>>()
        });
        records.push(json!({
            "factor_index": factor_index,
            "landmark_index": entry.landmark_index,
            "track_id": entry.track_id,
            "eligible": entry.eligible,
            "legacy_f32_bits": legacy_bits,
            "compact_f32_bits": compact_bits,
            "bitwise_equal": legacy_bits == compact_bits,
        }));
        let _ = factor;
    }
    let model = model_cost_decrease_f32(factors, state_step, tolerance)
        .map(|value| format!("{:08x}", (value as f32).to_bits()));
    Ok(json!({
        "status": "captured_from_same_ordered_rows",
        "state_step_f32": full70_checked_vector_bits(state_step, 0)?,
        "landmark_steps": records,
        "model_decrease_f32_row_reference": model,
        "model_decrease_compact_payload": compact.model_decrease_payload.is_some(),
        "model_decrease_note": "current reducer does not expose a compact Q2 model payload; the row reference is captured and compact recovery is compared bitwise",
    }))
}

static FULL70_ORACLE_EVENT: AtomicUsize = AtomicUsize::new(0);

pub(super) fn full70_oracle_payload(
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
    context: &serde_json::Value,
    provenance: &serde_json::Value,
) -> Result<serde_json::Value, ImuReductionError> {
    let factors = &event.linearization.factors;
    let state_dof = event.reduced.h.nrows();
    validate_full70_frame4_factors(factors, state_dof)?;
    let input_fingerprint = full70_factor_input_fingerprint(factors)?;
    let mut row_start = 0usize;
    let mut factor_records = Vec::with_capacity(factors.len());
    let mut visual_count = 0usize;
    for (ordinal, factor) in factors.iter().enumerate() {
        let paired_imu = if factor.kind == FactorKind::Imu {
            let bias = factors
                .get(ordinal + 1)
                .ok_or(ImuReductionError::MissingBias { imu_index: ordinal })?;
            Some(full70_imu_local_payload(factor, bias, ordinal, state_dof)?)
        } else {
            None
        };
        if factor.kind == FactorKind::Visual {
            visual_count += 1;
        }
        factor_records.push(full70_factor_record(
            factor, ordinal, row_start, paired_imu,
        )?);
        row_start = row_start
            .checked_add(factor.rows())
            .ok_or(ImuReductionError::Full70OracleInvalid { index: ordinal })?;
    }
    let legacy = reduce_landmark_factors_f32_checked(factors, state_dof, 1e-10)?;
    let compact = reduce_landmark_factors_f32_checked_with_compact_back_substitution(
        factors, state_dof, 1e-10,
    )?;
    let compact_batch = compact.compact_back_substitution.as_ref().ok_or(
        ImuReductionError::Full70OracleInvalid {
            index: visual_count,
        },
    )?;
    if compact_batch.entries.len() != visual_count {
        return Err(ImuReductionError::InvalidProjectionCount {
            expected: visual_count,
            actual: compact_batch.entries.len(),
        });
    }
    let compact_json = full70_reduced_json(&compact, "compact_f32")?;
    let legacy_json = full70_reduced_json(&legacy, "legacy_f32")?;
    let q2_rows = factors
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            let (jacobian, residual, rank) = landmark_nullspace_projection_f32(factor, 1e-10);
            Ok(json!({
                "factor_index": index,
                "kind": full70_factor_kind_name(factor.kind),
                "rank": rank,
                "jacobian_f32": diagnostic_f32_matrix(&jacobian),
                "residual_f32": diagnostic_f32_vector(&residual),
            }))
        })
        .collect::<Result<Vec<_>, ImuReductionError>>()?;
    let h_equal = full70_matrix_bits_match(&legacy.h, &compact.h);
    let b_equal = full70_vector_bits_match(&legacy.b, &compact.b);
    let recovery = full70_recovery_json(factors, &legacy, &compact, event.step, 1e-10)?;
    let run_id = active_diagnostic_lm_run_id().unwrap_or(0);
    let event_id = FULL70_ORACLE_EVENT.fetch_add(1, Ordering::Relaxed);
    Ok(json!({
        "schema": "basalt.m11.full70_factor_oracle.v1",
        "record": "frame4_factor_event",
        "run_id": format!("{run_id:032x}"),
        "event_id": event_id,
        "frame_id": frame_id,
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "state_dof": state_dof,
        "factor_count": factors.len(),
        "row_count": row_start,
        "visual_factor_count": visual_count,
        "factor_order_fingerprint": format!("{:016x}", visual_prefix_factor_order_fingerprint(factors)),
        "factor_input_fingerprint": format!("{:016x}", input_fingerprint),
        "provenance": provenance,
        "context": context,
        "state_f32": full70_checked_vector_bits(event.state, factors.len())?,
        "base_state_f32": full70_checked_vector_bits(event.base_state, factors.len())?,
        "step_f32": event.step.map(|step| full70_checked_vector_bits(step, factors.len())).transpose()?,
        "trial_state_f32": event.trial_state.map(|state| full70_checked_vector_bits(state, factors.len())).transpose()?,
        "factors": factor_records,
        "q2_rows_recomputed_from_ordered_factors": q2_rows,
        "reducer": {
            "legacy": legacy_json,
            "compact": compact_json,
            "h_bitwise_equal": h_equal,
            "b_bitwise_equal": b_equal,
            "stage_order": ["visual", "visual_plus_imu", "prior", "whole_h_b"],
        },
        "recovery": recovery,
        "solver_frontier": {
            "lambda_f32_bits": format!("{:08x}", (event.lambda as f32).to_bits()),
            "lambda_after_f32_bits": format!("{:08x}", (event.lambda_after as f32).to_bits()),
            "decision": event.decision,
            "model_cost": event.model_cost.map(|value| format!("{:08x}", (value as f32).to_bits())),
            "actual_cost": event.actual_cost.map(|value| format!("{:08x}", (value as f32).to_bits())),
        },
        "status": if h_equal && b_equal { "parity_pass" } else { "parity_mismatch" },
    }))
}

/// Emit one complete frame-4 factor event.  All arithmetic in this function is
/// diagnostic-only: the active solver has already produced its event, and the
/// checked reducer calls below operate on immutable copies of the same factors.
/// The path is created with an atomic temp/rename and an existing target is a
/// hard error, preventing an accidental append or partial fixture.
pub(crate) fn emit_full70_factor_oracle(
    path: &Path,
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
    context: &serde_json::Value,
) -> Result<(), ImuReductionError> {
    let provenance = full70_oracle_provenance_from_env()?;
    let payload = full70_oracle_payload(frame_id, event, context, &provenance)?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::Full70OracleIo)?;
    std::fs::create_dir_all(parent).map_err(|_| ImuReductionError::Full70OracleIo)?;
    let event_id = FULL70_ORACLE_EVENT.load(Ordering::Relaxed);
    let mut temporary = path.to_path_buf();
    let temporary_name = format!(
        ".{}.full70.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .ok_or(ImuReductionError::Full70OracleIo)?,
        std::process::id(),
        event_id
    );
    temporary.set_file_name(temporary_name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|_| ImuReductionError::Full70OracleIo)?;
    let write_result = serde_json::to_writer(&mut file, &payload)
        .map_err(|_| ImuReductionError::Full70OracleIo)
        .and_then(|_| {
            file.write_all(b"\n")
                .map_err(|_| ImuReductionError::Full70OracleIo)
        })
        .and_then(|_| file.flush().map_err(|_| ImuReductionError::Full70OracleIo));
    if let Err(error) = write_result {
        drop(file);
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    drop(file);
    if std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
        return Err(ImuReductionError::Full70OracleIo);
    }
    Ok(())
}

const VISUAL_PREFIX_TRACE_CURRENT_TRIAL: usize = 0;

/// The prefix sidecar is an explicitly diagnostic operation.  Consult the
/// process-lifetime diagnostic snapshot before reading its path so the normal
/// reducer does not enumerate the environment (or allocate a path) on every
/// LM iteration.  The sidecar key is intentionally an unknown diagnostic key
/// to the window allowlist, so it selects the retained diagnostic policy at
/// startup when present.
pub(super) static VISUAL_PREFIX_TRACE_EVENT: AtomicUsize = AtomicUsize::new(0);
#[cfg(test)]
pub(super) static VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
static VISUAL_PREFIX_TRACE_FILTER: OnceLock<Result<VisualPrefixTraceFilter, ()>> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VisualPrefixTraceFilter {
    frame_id: Option<u64>,
    iteration: Option<usize>,
    trial: Option<usize>,
}

impl VisualPrefixTraceFilter {
    #[inline]
    pub(super) fn matches(
        self,
        frame_id: Option<u64>,
        iteration: Option<usize>,
        trial: usize,
    ) -> bool {
        self.frame_id.is_none_or(|target| frame_id == Some(target))
            && self
                .iteration
                .is_none_or(|target| iteration == Some(target))
            && self.trial.is_none_or(|target| trial == target)
    }
}

pub(super) fn visual_prefix_trace_filter_from_cached(
    frame_id: Option<Result<u64, ()>>,
    iteration: Option<Result<usize, ()>>,
    trial: Option<Result<usize, ()>>,
) -> Result<VisualPrefixTraceFilter, ()> {
    Ok(VisualPrefixTraceFilter {
        frame_id: frame_id.transpose()?,
        iteration: iteration.transpose()?,
        trial: trial.transpose()?,
    })
}

fn visual_prefix_trace_filter() -> Result<VisualPrefixTraceFilter, ImuReductionError> {
    match *VISUAL_PREFIX_TRACE_FILTER.get_or_init(|| {
        let policy = crate::vio::window::diagnostic_env_snapshot();
        visual_prefix_trace_filter_from_cached(
            policy.visual_prefix_trace_frame,
            policy.visual_prefix_trace_iteration,
            policy.visual_prefix_trace_trial,
        )
    }) {
        Ok(filter) => Ok(filter),
        Err(()) => Err(ImuReductionError::VisualPrefixTraceInvalid { index: 0 }),
    }
}

fn visual_prefix_trace_path() -> Option<&'static Path> {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    if !policy.active {
        return None;
    }
    policy.visual_prefix_trace.as_deref()
}

#[inline]
const fn visual_prefix_hash_mix(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
}

fn visual_prefix_factor_order_fingerprint(factors: &[WhitenedFactorRowStack]) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    visual_prefix_hash_mix(&mut hash, factors.len() as u64);
    for (factor_index, factor) in factors.iter().enumerate() {
        visual_prefix_hash_mix(&mut hash, factor_index as u64);
        visual_prefix_hash_mix(
            &mut hash,
            match factor.kind {
                FactorKind::Generic => 0,
                FactorKind::Prior => 1,
                FactorKind::Visual => 2,
                FactorKind::Imu => 3,
                FactorKind::Bias => 4,
            },
        );
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.nrows() as u64);
        visual_prefix_hash_mix(&mut hash, factor.state_jacobian.ncols() as u64);
        visual_prefix_hash_mix(&mut hash, factor.landmark_jacobian.ncols() as u64);
        if let Some(metadata) = factor.landmark_metadata {
            visual_prefix_hash_mix(&mut hash, metadata.landmark_index as u64);
            visual_prefix_hash_mix(&mut hash, metadata.track_id);
        } else {
            visual_prefix_hash_mix(&mut hash, u64::MAX);
            visual_prefix_hash_mix(&mut hash, u64::MAX);
        }
        if let Some(observations) = factor.visual_observation_ids.as_ref() {
            visual_prefix_hash_mix(&mut hash, observations.len() as u64);
            for &(state_index, camera_id) in observations {
                visual_prefix_hash_mix(&mut hash, state_index as u64);
                visual_prefix_hash_mix(&mut hash, camera_id as u64);
            }
        } else {
            visual_prefix_hash_mix(&mut hash, u64::MAX);
        }
    }
    hash
}

static VISUAL_PREFIX_TRACE_WRITERS: OnceLock<Mutex<HashSet<std::path::PathBuf>>> = OnceLock::new();

fn visual_prefix_trace_writer_registry() -> &'static Mutex<HashSet<std::path::PathBuf>> {
    VISUAL_PREFIX_TRACE_WRITERS.get_or_init(|| Mutex::new(HashSet::new()))
}

struct VisualPrefixTraceLease {
    key: std::path::PathBuf,
    lock_path: std::path::PathBuf,
    lock_file: Option<std::fs::File>,
}

impl Drop for VisualPrefixTraceLease {
    fn drop(&mut self) {
        // Release the OS handle before removing the adjacent lock file.  A
        // stale lock is never removed by acquisition; only this live lease
        // cleans up the file it successfully created.
        self.lock_file.take();
        let _ = std::fs::remove_file(&self.lock_path);
        let Ok(mut paths) = visual_prefix_trace_writer_registry().lock() else {
            // A poisoned registry is already fail-closed for new writers; do
            // not panic while releasing a diagnostic lease.
            return;
        };
        paths.remove(&self.key);
    }
}

pub(super) fn canonical_visual_prefix_trace_key(
    path: &Path,
) -> Result<std::path::PathBuf, ImuReductionError> {
    if path.as_os_str().is_empty() || path.file_name().is_none() {
        return Err(ImuReductionError::VisualPrefixTraceIo);
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?
            .join(path)
    };
    let parent = absolute
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    std::fs::create_dir_all(parent).map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let canonical_parent =
        std::fs::canonicalize(parent).map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let file_name = absolute
        .file_name()
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    // Resolve an existing file/symlink when possible.  For a new file, the
    // canonical parent plus filename is the stable same-volume key.
    Ok(std::fs::canonicalize(&absolute).unwrap_or_else(|_| canonical_parent.join(file_name)))
}

pub(super) fn visual_prefix_trace_lock_path(
    canonical_target: &Path,
) -> Result<std::path::PathBuf, ImuReductionError> {
    let parent = canonical_target
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    let file_name = canonical_target
        .file_name()
        .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
    let mut lock_name = std::ffi::OsString::from(".");
    lock_name.push(file_name);
    lock_name.push(".lock");
    Ok(parent.join(lock_name))
}

pub(super) fn visual_prefix_path_hash(canonical_target: &Path) -> u64 {
    let mut hash = LM_VECTOR_FNV_OFFSET;
    for byte in canonical_target.to_string_lossy().as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(LM_VECTOR_FNV_PRIME);
    }
    hash
}

fn acquire_visual_prefix_trace_lock(
    canonical_target: &Path,
    run_id: u128,
    event_id: usize,
) -> Result<(std::path::PathBuf, std::fs::File), ImuReductionError> {
    let lock_path = visual_prefix_trace_lock_path(canonical_target)?;
    // `create_new` is the cross-process ownership boundary.  Existing,
    // malformed, or stale locks are deliberately not inspected or removed:
    // the caller must resolve them explicitly.
    let mut lock_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    let payload = json!({
        "schema": "basalt.m11.visual_prefix_trace_lock.v1",
        "run_id": format!("{run_id:032x}"),
        "pid": std::process::id(),
        "event_id": event_id,
        "target_path_hash": format!("{:016x}", visual_prefix_path_hash(canonical_target)),
    });
    let write_result = serde_json::to_writer(&mut lock_file, &payload)
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)
        .and_then(|_| {
            lock_file
                .write_all(b"\n")
                .map_err(|_| ImuReductionError::VisualPrefixTraceIo)
        });
    if let Err(error) = write_result {
        drop(lock_file);
        let _ = std::fs::remove_file(&lock_path);
        return Err(error);
    }
    Ok((lock_path, lock_file))
}

fn reserve_visual_prefix_trace_path(
    path: &Path,
    run_id: u128,
    event_id: usize,
) -> Result<VisualPrefixTraceLease, ImuReductionError> {
    let key = canonical_visual_prefix_trace_key(path)?;
    let mut paths = visual_prefix_trace_writer_registry()
        .lock()
        .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
    if !paths.insert(key.clone()) {
        return Err(ImuReductionError::VisualPrefixTraceIo);
    }
    drop(paths);
    let (lock_path, lock_file) = match acquire_visual_prefix_trace_lock(&key, run_id, event_id) {
        Ok(lock) => lock,
        Err(error) => {
            if let Ok(mut paths) = visual_prefix_trace_writer_registry().lock() {
                paths.remove(&key);
            }
            return Err(error);
        }
    };
    Ok(VisualPrefixTraceLease {
        key,
        lock_path,
        lock_file: Some(lock_file),
    })
}

pub(super) struct VisualPrefixTraceWriter {
    pub(super) file: std::fs::File,
    _lease: VisualPrefixTraceLease,
    event_id: usize,
    run_id: u128,
    record_sequence: usize,
    frame_id: Option<u64>,
    iteration: Option<usize>,
    state_dof: usize,
    pub(super) expected_visual_count: usize,
    pub(super) next_visual_ordinal: usize,
    factor_order_fingerprint: u64,
}

pub(super) struct VisualPrefixPending {
    visual_ordinal: usize,
    factor_index: usize,
    landmark_index: usize,
    track_id: u64,
    observations: Vec<serde_json::Value>,
    row_count: usize,
    landmark_dof: usize,
    rank: usize,
    active_state_columns: Vec<usize>,
    global_h_before: serde_json::Value,
    global_b_before: serde_json::Value,
}

impl VisualPrefixTraceWriter {
    pub(super) fn open(
        path: impl AsRef<Path>,
        factors: &[WhitenedFactorRowStack],
        state_dof: usize,
    ) -> Result<Option<Self>, ImuReductionError> {
        #[cfg(test)]
        VISUAL_PREFIX_TRACE_OPEN_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
        let path = path.as_ref();
        if path.as_os_str().is_empty() || path.file_name().is_none() {
            return Err(ImuReductionError::VisualPrefixTraceIo);
        }
        let event_id = VISUAL_PREFIX_TRACE_EVENT.fetch_add(1, Ordering::Relaxed);
        let run_id = active_diagnostic_lm_run_id().unwrap_or_else(next_diagnostic_lm_run_id);
        let lease = reserve_visual_prefix_trace_path(path, run_id, event_id)?;
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        let frame_id = active_diagnostic_lm_frame();
        let iteration = active_diagnostic_lm_iteration();
        let expected_visual_count = factors
            .iter()
            .filter(|factor| {
                factor.kind == FactorKind::Visual && factor.landmark_jacobian.ncols() != 0
            })
            .count();
        let factor_order_fingerprint = visual_prefix_factor_order_fingerprint(factors);
        let mut writer = Self {
            file,
            _lease: lease,
            event_id,
            run_id,
            record_sequence: 0,
            frame_id,
            iteration,
            state_dof,
            expected_visual_count,
            next_visual_ordinal: 0,
            factor_order_fingerprint,
        };
        let run_id = writer.run_id_string();
        let record_sequence = writer.record_sequence;
        writer.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "header",
            "event_id": event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": frame_id,
            "iteration": iteration,
            "trial": 0,
            "state_dof": state_dof,
            "factor_count": factors.len(),
            "visual_factor_count": expected_visual_count,
            "factor_order_fingerprint": format!("{:016x}", factor_order_fingerprint),
        }))?;
        Ok(Some(writer))
    }

    fn run_id_string(&self) -> String {
        format!("{:032x}", self.run_id)
    }

    pub(super) fn write_record(
        &mut self,
        record: &serde_json::Value,
    ) -> Result<(), ImuReductionError> {
        let next_sequence = self
            .record_sequence
            .checked_add(1)
            .ok_or(ImuReductionError::VisualPrefixTraceIo)?;
        serde_json::to_writer(&mut self.file, record)
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        self.file
            .write_all(b"\n")
            .map_err(|_| ImuReductionError::VisualPrefixTraceIo)?;
        self.record_sequence = next_sequence;
        Ok(())
    }

    pub(super) fn begin_visual_prefix(
        &mut self,
        factor_index: usize,
        factor: &WhitenedFactorRowStack,
        projected_jacobian: &DMatrix<f32>,
        projected_residual: &DVector<f32>,
        rank: usize,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<VisualPrefixPending, ImuReductionError> {
        if factor.kind != FactorKind::Visual
            || factor.landmark_jacobian.ncols() == 0
            || projected_jacobian.ncols() != self.state_dof
            || projected_residual.len() != projected_jacobian.nrows()
            || self.next_visual_ordinal >= self.expected_visual_count
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        }
        let Some(metadata) = factor.landmark_metadata else {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        };
        let Some(observation_ids) = factor.visual_observation_ids.as_ref() else {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: factor_index,
            });
        };
        let observations = observation_ids
            .iter()
            .map(|&(state_index, camera_id)| {
                json!({
                    "state_index": state_index,
                    "camera_id": camera_id,
                })
            })
            .collect();
        let active_state_columns = (0..factor.state_jacobian.ncols())
            .filter(|&column| {
                (0..factor.state_jacobian.nrows())
                    .any(|row| factor.state_jacobian[(row, column)] != 0.0)
            })
            .collect();
        Ok(VisualPrefixPending {
            visual_ordinal: self.next_visual_ordinal,
            factor_index,
            landmark_index: metadata.landmark_index,
            track_id: metadata.track_id,
            observations,
            row_count: projected_jacobian.nrows(),
            landmark_dof: factor.landmark_jacobian.ncols(),
            rank,
            active_state_columns,
            global_h_before: diagnostic_f32_matrix(global_h),
            global_b_before: diagnostic_f32_vector(global_b),
        })
    }

    pub(super) fn finish_visual_prefix(
        &mut self,
        pending: VisualPrefixPending,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if pending.visual_ordinal != self.next_visual_ordinal
            || global_h.nrows() != self.state_dof
            || global_h.ncols() != self.state_dof
            || global_b.len() != self.state_dof
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: pending.factor_index,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "visual_prefix",
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal": pending.visual_ordinal,
            "factor_index": pending.factor_index,
            "landmark_index": pending.landmark_index,
            "track_id": pending.track_id,
            "observations": pending.observations,
            "row_count": pending.row_count,
            "landmark_dof": pending.landmark_dof,
            "rank": pending.rank,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "aom_scatter": {
                "state_dof": self.state_dof,
                "active_state_columns": pending.active_state_columns,
                "landmark_columns_eliminated": pending.landmark_dof,
            },
            "global_h_before": pending.global_h_before,
            "global_b_before": pending.global_b_before,
            "global_h_after": diagnostic_f32_matrix(global_h),
            "global_b_after": diagnostic_f32_vector(global_b),
        }))?;
        self.next_visual_ordinal += 1;
        Ok(())
    }

    pub(super) fn write_stage(
        &mut self,
        stage: &str,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if global_h.nrows() != self.state_dof
            || global_h.ncols() != self.state_dof
            || global_b.len() != self.state_dof
        {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: self.next_visual_ordinal,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "stage",
            "stage": stage,
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal_count": self.next_visual_ordinal,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "global_h": diagnostic_f32_matrix(global_h),
            "global_b": diagnostic_f32_vector(global_b),
        }))
    }

    pub(super) fn write_prior_before(
        &mut self,
        global_h: &DMatrix<f32>,
        global_b: &DVector<f32>,
        prior_h: &DMatrix<f32>,
        prior_b: &DVector<f32>,
    ) -> Result<(), ImuReductionError> {
        if self.next_visual_ordinal != self.expected_visual_count {
            return Err(ImuReductionError::VisualPrefixTraceInvalid {
                index: self.next_visual_ordinal,
            });
        }
        let run_id = self.run_id_string();
        let record_sequence = self.record_sequence;
        self.write_record(&json!({
            "schema": "basalt.m11.absqr.visual_prior_boundary.v1",
            "record": "prior_boundary",
            "stage": "prior_before",
            "event_id": self.event_id,
            "run_id": run_id,
            "record_sequence": record_sequence,
            "frame_id": self.frame_id,
            "iteration": self.iteration,
            "trial": 0,
            "visual_ordinal_count": self.next_visual_ordinal,
            "factor_order_fingerprint": format!("{:016x}", self.factor_order_fingerprint),
            "global_h": diagnostic_f32_matrix(global_h),
            "global_b": diagnostic_f32_vector(global_b),
            "prior_h": diagnostic_f32_matrix(prior_h),
            "prior_b": diagnostic_f32_vector(prior_b),
        }))
    }
}

pub(super) fn visual_prefix_trace_writer(
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<Option<VisualPrefixTraceWriter>, ImuReductionError> {
    if !crate::vio::window::diagnostic_env_active() {
        return Ok(None);
    }
    // Apply the cheap cached selector before reading/cloning the output path
    // or opening the file.  A non-target event therefore allocates neither a
    // header/prefix record nor any sidecar metadata/full H/b payload.  The
    // current prefix records are iteration-level reductions and use trial 0;
    // retaining trial in the selector keeps the contract explicit for future
    // trial-level producers without guessing a trial for this producer.
    let filter = visual_prefix_trace_filter()?;
    let frame_id = active_diagnostic_lm_frame();
    let iteration = active_diagnostic_lm_iteration();
    let trial = VISUAL_PREFIX_TRACE_CURRENT_TRIAL;
    if !filter.matches(frame_id, iteration, trial) {
        return Ok(None);
    }
    let path = visual_prefix_trace_path();
    visual_prefix_trace_writer_selected(
        filter, frame_id, iteration, trial, path, factors, state_dof,
    )
}

/// Apply the cheap event selector before resolving the output path or opening
/// the sidecar.  Keeping this boundary separate makes the no-op contract
/// testable: a non-target event must not invoke [`VisualPrefixTraceWriter`]
/// (and therefore must not copy factor metadata or materialize any H/b JSON).
#[inline]
pub(super) fn visual_prefix_trace_writer_selected(
    filter: VisualPrefixTraceFilter,
    frame_id: Option<u64>,
    iteration: Option<usize>,
    trial: usize,
    path: Option<&Path>,
    factors: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> Result<Option<VisualPrefixTraceWriter>, ImuReductionError> {
    if !filter.matches(frame_id, iteration, trial) {
        return Ok(None);
    }
    let Some(path) = path else {
        return Ok(None);
    };
    VisualPrefixTraceWriter::open(path, factors, state_dof)
}

/// Emit the exact transformed Q1 inputs used by the float32 landmark
/// back-substitution.  This is intentionally opt-in: the production solver
/// never opens a sidecar and the probe is restricted to the two tracks that
/// first exposed the Eigen 3x75 GEMV reduction boundary.
pub(super) fn emit_landmark_backsub_probe(
    track_id: Option<u64>,
    state_step: &DVector<f32>,
    q1_state: &DMatrix<f32>,
    q1_residual: &DVector<f32>,
    q1_state_step: &DVector<f32>,
    upper_r: &DMatrix<f32>,
) {
    let Some(track_id) = track_id else {
        return;
    };
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.landmark_backsub_probe.as_ref() else {
        return;
    };
    let selected = if let Some(filter) = policy.landmark_backsub_probe_tracks.as_deref() {
        filter.iter().any(|value| *value == track_id)
    } else {
        track_id == 19 || track_id == 49
    };
    if !selected {
        return;
    }
    let record = json!({
        "schema": "basalt.m7im15_landmark_backsub_probe.v1",
        "track_id": track_id,
        "state_step": diagnostic_f32_vector(state_step),
        "q1_state": diagnostic_f32_matrix(q1_state),
        "q1_residual": diagnostic_f32_vector(q1_residual),
        "q1_state_step": diagnostic_f32_vector(q1_state_step),
        "upper_r": diagnostic_f32_matrix(upper_r),
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Emit the complete per-landmark QR boundary for the selected diagnostic
/// track.  This reuses the existing back-sub probe path so callers do not
/// need another environment key or a second factor construction.  The hook
/// is called only after the production Householder walk has completed; it
/// never participates in Q2 assembly or landmark acceptance.
pub(super) fn emit_landmark_projection_probe(
    metadata: LandmarkFactorMetadata,
    factor: &WhitenedFactorRowStack,
    state: &DMatrix<f32>,
    landmark: &DMatrix<f32>,
    residual: &DVector<f32>,
    qr: &LandmarkHouseholderF32,
    rank: usize,
    norm_ok: bool,
) {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.landmark_backsub_probe.as_ref() else {
        return;
    };
    let selected = if let Some(filter) = policy.landmark_backsub_probe_tracks.as_deref() {
        filter.iter().any(|value| *value == metadata.track_id)
    } else {
        metadata.track_id == 19 || metadata.track_id == 49
    };
    if !selected {
        return;
    }
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector_bits = |values: &[f32]| json!(values.iter().copied().map(bits).collect::<Vec<_>>());
    let transformed_state = qr.transformed_state();
    let transformed_landmark = qr.transformed_landmark();
    let transformed_residual = qr.transformed_residual();
    let q1_state = DMatrix::from_fn(qr.landmark_cols, qr.state_cols, |row, column| {
        transformed_state[(row, column)]
    });
    let q1_residual = DVector::from_iterator(
        qr.landmark_cols,
        (0..qr.landmark_cols).map(|row| transformed_residual[row]),
    );
    let q2_state = if qr.rows > qr.landmark_cols {
        qr.q2_state()
    } else {
        DMatrix::zeros(0, qr.state_cols)
    };
    let q2_residual = if qr.rows > qr.landmark_cols {
        qr.q2_residual()
    } else {
        DVector::zeros(0)
    };
    let projection_event = active_diagnostic_projection_event();
    let run_id = active_diagnostic_lm_run_id().map(|value| format!("{value:032x}"));
    let observations = factor
        .visual_observation_ids
        .as_ref()
        .map(|values| {
            values
                .iter()
                .map(|(state_index, camera_id)| {
                    json!({"state_index": state_index, "camera_id": camera_id})
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let record = json!({
        "schema": "basalt.m11.track_numeric_projection.v1",
        "source": "rust",
        "stage": "landmark_projection_after_householder",
        "run_id": run_id,
        "frame_id": active_diagnostic_lm_frame(),
        "event_frame_id": projection_event.map(|event| event.frame_id),
        "event_timestamp_ns": projection_event.map(|event| event.timestamp_ns),
        "iteration": active_diagnostic_lm_iteration(),
        "trial": 0,
        "landmark_index": metadata.landmark_index,
        "track_id": metadata.track_id,
        "factor_rows": factor.rows(),
        "state_cols": factor.state_jacobian.ncols(),
        "landmark_cols": factor.landmark_jacobian.ncols(),
        "observations": observations,
        "rank": rank,
        "norm_ok": norm_ok,
        "tolerance_f32_bits": bits(1e-10_f64 as f32),
        "pre_qr": {
            "state": diagnostic_f32_matrix(state),
            "landmark": diagnostic_f32_matrix(landmark),
            "residual": diagnostic_f32_vector(residual),
        },
        "post_qr": {
            "rows": qr.rows,
            "state_cols": qr.state_cols,
            "landmark_cols": qr.landmark_cols,
            "landmark_offset": qr.landmark_offset,
            "residual_offset": qr.residual_offset,
            "transformed_state": diagnostic_f32_matrix(&transformed_state),
            "transformed_landmark": diagnostic_f32_matrix(&transformed_landmark),
            "transformed_residual": diagnostic_f32_vector(&transformed_residual),
            "pivots_f32_bits": vector_bits(&qr.pivots),
            "tau_f32_bits": vector_bits(&qr.tau),
        },
        "q1": {
            "state": diagnostic_f32_matrix(&q1_state),
            "residual": diagnostic_f32_vector(&q1_residual),
            "upper_r": diagnostic_f32_matrix(&qr.upper_r()),
        },
        "q2": {
            "state": diagnostic_f32_matrix(&q2_state),
            "residual": diagnostic_f32_vector(&q2_residual),
        },
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Emit only the selected frame/iterations.  This is JSONL rather than the
/// ordinary detail schema: two frame-8 snapshots are roughly tens of KB and
/// contain no landmarks, observations, or state dumps.
pub(super) fn emit_imu_reduction_diagnostic(
    frame_id: Option<u64>,
    iteration: usize,
    reduced: &ReducedNormalSystemF32,
) {
    let policy = crate::vio::window::diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_rows.as_ref() else {
        return;
    };
    let target_frame = policy.diagnostic_frame;
    if target_frame.is_some() && target_frame != frame_id {
        return;
    }
    if let Some(filter) = policy.diagnostic_imu_iterations.as_deref() {
        let selected = filter.iter().any(|value| *value == iteration);
        if !selected {
            return;
        }
    }
    let Some(imu) = reduced.imu_diagnostic.as_ref() else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_none() {
        let _ = INITIALIZED.set(());
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, b"");
    }
    let local_blocks = imu
        .local_blocks
        .iter()
        .map(|block| {
            let mut record = json!({
                "active_offsets": block.active_offsets,
                "local_jacobian": diagnostic_f32_matrix(&block.local_jacobian),
                "residual": diagnostic_f32_vector(&block.residual),
                "local_h": diagnostic_f32_matrix(&block.local_h),
                "local_b": diagnostic_f32_vector(&block.local_b),
                "local_vs_global_active": {
                    "h_bit_mismatches": block.local_vs_global_h_mismatches,
                    "b_bit_mismatches": block.local_vs_global_b_mismatches,
                },
                "local_vs_global_padded": {
                    "h_bit_mismatches": block.local_vs_global_padded_h_mismatches,
                    "b_bit_mismatches": block.local_vs_global_padded_b_mismatches,
                    "h_total": reduced.h.nrows() * reduced.h.ncols(),
                    "b_total": reduced.b.len(),
                },
            });
            // Keep the stage names at the local-block level so an audit can
            // consume one block without knowing an additional wrapper
            // schema.  Retain the complete nested copy as well for callers
            // that prefer to treat the input boundary as one payload.
            if let Some(input) = block.imu_input_diagnostic.as_ref() {
                if let Some(fields) = input.as_object() {
                    let object = record
                        .as_object_mut()
                        .expect("local IMU diagnostic record is an object");
                    for (key, value) in fields {
                        object.insert(key.clone(), value.clone());
                    }
                    object.insert("factor_input".into(), input.clone());
                }
            }
            record
        })
        .collect::<Vec<_>>();
    let imu_cumulative_stages = imu
        .imu_cumulative_stages
        .iter()
        .map(|stage| {
            let mut record = json!({
                "active_offsets": stage.active_offsets,
                "imu_h": diagnostic_f32_matrix(&stage.imu_h),
                "imu_b": diagnostic_f32_vector(&stage.imu_b),
            });
            if let Some(input) = stage.imu_input_diagnostic.as_ref() {
                if let Some(fields) = input.as_object() {
                    let object = record
                        .as_object_mut()
                        .expect("cumulative IMU diagnostic record is an object");
                    for (key, value) in fields {
                        object.insert(key.clone(), value.clone());
                    }
                    object.insert("factor_input".into(), input.clone());
                }
            }
            record
        })
        .collect::<Vec<_>>();
    let record = json!({
        "schema": "basalt.m7im15_reduction_diagnostic.v3",
        "frame_id": frame_id,
        "iteration": iteration,
        "state_dof": reduced.h.nrows(),
        "local_blocks": local_blocks,
        "imu_cumulative_stages": imu_cumulative_stages,
        "imu_accumulator": {
            "h": diagnostic_f32_matrix(&imu.imu_h),
            "b": diagnostic_f32_vector(&imu.imu_b),
        },
        "full": {
            "h": diagnostic_f32_matrix(&reduced.h),
            "b": diagnostic_f32_vector(&reduced.b),
        },
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}
