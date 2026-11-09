//! Marginal-prior factors, their f32 RHS kernels and the upstream
//! marginal-prior error.

use super::*;

impl WindowProblem {
    pub(super) fn prior_factors(&self, state: &DVector<f64>) -> Vec<WhitenedFactorRowStack> {
        let mut factors = Vec::new();
        if let Some(prior) = &self.prior {
            let mut columns = DMatrix::zeros(prior.rows(), self.state_dof());
            let mut prior_state_columns = Vec::with_capacity(prior.jacobian.ncols());
            let kinds = prior.kinds();
            let mut prior_offset = 0;
            for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
                let kind = kinds
                    .get(prior_index)
                    .copied()
                    .unwrap_or(WindowBlockKind::State);
                let Some(current_index) = self.block_index_for(*frame_id, kind) else {
                    prior_offset += block_dof(kind);
                    continue;
                };
                let dof = block_dof(kind);
                columns
                    .view_mut(
                        (0, self.block_offset(current_index, kind)),
                        (prior.rows(), dof),
                    )
                    .copy_from(&prior.jacobian.columns(prior_offset, dof));
                for local_column in 0..dof {
                    prior_state_columns.push(self.block_offset(current_index, kind) + local_column);
                }
                prior_offset += dof;
            }
            // The stored square-root prior is FEJ-linearized at `fej_point`.
            // Upstream's `get_dense_Q2Jp_Q2r` uses the residual
            //
            //     r_lin = H * delta + b,
            //
            // where `delta` is the current state minus the FEJ state.  The
            // plus sign is important: `b` is the residual at the original
            // linearization point, not a right-hand-side to subtract from the
            // current tangent.  Keeping this operation in one helper also
            // lets the reduced prior cost below use the same delta.
            let current_point = self.prior_current_point(state, prior);
            Self::emit_prior_residual_inputs(prior, &current_point, self.state_dof());
            let rhs = if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.nrows() == 21
                && prior.jacobian.ncols() == 21
                && prior.rhs.len() == 21
                && current_point.len() == 21
            {
                // Eigen's column-major GeneralMatrixVector path handles a
                // 21x21 prior as two Packet8 output blocks, one Packet4
                // block, and one scalar row.  The packet rows accumulate
                // every depth with FMA, then fuse the existing mld.b value
                // in the final pmadd store; row 20 uses separate scalar
                // multiply/add operations throughout.  The ordinary
                // nalgebra product is mathematically identical but rounds
                // four lanes differently at this solver boundary.
                Self::prior_rhs_col_major_21_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (27, 27)
                && prior.rhs.len() == 27
                && current_point.len() == 27
            {
                Self::prior_rhs_col_major_27_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (33, 33)
                && prior.rhs.len() == 33
                && current_point.len() == 33
            {
                Self::prior_rhs_col_major_33_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (39, 39)
                && prior.rhs.len() == 39
                && current_point.len() == 39
            {
                Self::prior_rhs_col_major_39_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (45, 45)
                && prior.rhs.len() == 45
                && current_point.len() == 45
            {
                Self::prior_rhs_col_major_45_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (51, 51)
                && prior.rhs.len() == 51
                && current_point.len() == 51
            {
                Self::prior_rhs_col_major_51_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (57, 57)
                && prior.rhs.len() == 57
                && current_point.len() == 57
            {
                Self::prior_rhs_col_major_57_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else {
                prior.rhs.clone() + &prior.jacobian * current_point
            };
            // Eigen's prior residual starts from a zero-initialized RHS
            // expression.  Keep exact zeros canonical at this producer
            // boundary; nalgebra's subtractive path can otherwise preserve
            // a negative zero, which becomes observable in Q2r although the
            // numerical residual is zero.  Do not alter nonzero values.
            let rhs = DVector::from_iterator(
                rhs.len(),
                rhs.iter().copied().map(|value| {
                    // The capture/solver owns this factor as f32 in the
                    // pinned path.  A tiny f64 carry can cast to -0.0;
                    // canonicalize that representational zero here too.
                    if (value as f32) == 0.0 {
                        0.0
                    } else {
                        value
                    }
                }),
            );
            if let Some(factor) = prior_factor(columns, rhs) {
                // The stored prior is compact in upstream.  Attach the
                // compact-to-global map only when every source column was
                // represented by an active block; incomplete windows keep
                // the legacy global product fallback.
                let factor = if prior_state_columns.len() == prior.jacobian.ncols() {
                    factor.with_prior_state_columns(prior_state_columns)
                } else {
                    factor
                };
                factors.push(factor);
            }
        }
        // The initial anchor row is already folded into the square-root prior
        // on the first window shift.  Re-adding it after that point would pin
        // the *current first state* to the original frame-0 coordinates even
        // after frame 0 has left the window, injecting a large false residual
        // and making the LM model reject every trial.  A fallback prior carries
        // its own gauge row, so the explicit row is needed only before a prior
        // exists.
        if self.prior.is_none() {
            if let Some(anchor) = &self.anchor_point {
                if anchor.len() == NAV_STATE_DOF && !self.states.is_empty() {
                    let mut jacobian = DMatrix::zeros(NAV_STATE_DOF, self.state_dof());
                    let pose_weight = self.initial_pose_weight.sqrt();
                    let accel_bias_weight = self.initial_accel_bias_weight.sqrt();
                    let gyro_bias_weight = self.initial_gyro_bias_weight.sqrt();
                    if !pose_weight.is_finite()
                        || !accel_bias_weight.is_finite()
                        || !gyro_bias_weight.is_finite()
                    {
                        return factors;
                    }
                    for axis in 0..3 {
                        jacobian[(axis, axis)] = pose_weight;
                    }
                    // Basalt fixes yaw while leaving roll/pitch to gravity.
                    jacobian[(5, 5)] = pose_weight;
                    if self.visual_only.is_some() {
                        // No gravity without an IMU: pin roll/pitch too.
                        jacobian[(3, 3)] = pose_weight;
                        jacobian[(4, 4)] = pose_weight;
                    }
                    // Preserve the upstream assignment in
                    // sqrt_keypoint_vio.cpp: vio_init_ba_weight is written to
                    // columns 9..11 and vio_init_bg_weight to columns 12..14.
                    for axis in 0..3 {
                        jacobian[(9 + axis, 9 + axis)] = accel_bias_weight;
                        jacobian[(12 + axis, 12 + axis)] = gyro_bias_weight;
                    }
                    let Some(current_nav) = self.block_nav(state, self.poses.len()) else {
                        return factors;
                    };
                    let anchor_nav = decode_state_with_mode(anchor, self.scalar_mode);
                    let current = if self.scalar_mode == ScalarMode::UpstreamF32 {
                        self.states
                            .first()
                            .filter(|state| state.linearized_delta.len() == NAV_STATE_DOF)
                            .map(|state| state.linearized_delta.clone())
                            .unwrap_or_else(|| local_state_difference(&current_nav, &anchor_nav))
                    } else {
                        local_state_difference(&current_nav, &anchor_nav)
                    };
                    let mut rhs = DVector::zeros(NAV_STATE_DOF);
                    for index in 0..NAV_STATE_DOF {
                        rhs[index] = if self.scalar_mode == ScalarMode::UpstreamF32 {
                            // The pinned anchor factor is built in Scalar=float:
                            // narrow both chart operands before the subtraction
                            // and apply the float diagonal weight there.  Doing
                            // the product in f64 and narrowing afterward moves
                            // the local-bias row by one ulp (Q2r row 90).
                            let weight = jacobian[(index, index)] as f32;
                            let delta = (current[index] as f32) - (anchor[index] as f32);
                            f64::from(weight * delta)
                        } else {
                            jacobian[(index, index)] * (current[index] - anchor[index])
                        };
                    }
                    // The anchor is the first Prior row at the frame-4
                    // boundary.  Match Eigen's f32 zero construction here as
                    // well: tiny f64 products must not survive as -0.0 after
                    // the factor is narrowed for Q2r.
                    for value in rhs.iter_mut() {
                        if (*value as f32) == 0.0 {
                            *value = 0.0;
                        }
                    }
                    if let Some(factor) = prior_factor(jacobian, rhs) {
                        factors.push(factor);
                    }
                }
            }
        }
        if let Some(factor) = self.visual_only_factor(|index| self.block_nav(state, index)) {
            factors.push(factor);
        }
        factors
    }

    pub(super) fn prior_factors_view<V: WindowValueView>(
        &self,
        values: &V,
    ) -> Vec<WhitenedFactorRowStack> {
        let base = values.base();
        let mut factors = Vec::new();
        if let Some(prior) = &base.prior {
            let mut columns = DMatrix::zeros(prior.rows(), base.state_dof());
            let mut prior_state_columns = Vec::with_capacity(prior.jacobian.ncols());
            let kinds = prior.kinds();
            let mut prior_offset = 0;
            for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
                let kind = kinds
                    .get(prior_index)
                    .copied()
                    .unwrap_or(WindowBlockKind::State);
                let Some(current_index) = base.block_index_for(*frame_id, kind) else {
                    prior_offset += block_dof(kind);
                    continue;
                };
                let dof = block_dof(kind);
                let block_offset = base.block_offset(current_index, kind);
                columns
                    .view_mut((0, block_offset), (prior.rows(), dof))
                    .copy_from(&prior.jacobian.columns(prior_offset, dof));
                for local_column in 0..dof {
                    prior_state_columns.push(block_offset + local_column);
                }
                prior_offset += dof;
            }
            let current_point = values.prior_current_point(prior);
            Self::emit_prior_residual_inputs(prior, &current_point, base.state_dof());
            let rhs = if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.nrows() == 21
                && prior.jacobian.ncols() == 21
                && prior.rhs.len() == 21
                && current_point.len() == 21
            {
                Self::prior_rhs_col_major_21_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (27, 27)
                && prior.rhs.len() == 27
                && current_point.len() == 27
            {
                Self::prior_rhs_col_major_27_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (33, 33)
                && prior.rhs.len() == 33
                && current_point.len() == 33
            {
                Self::prior_rhs_col_major_33_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (39, 39)
                && prior.rhs.len() == 39
                && current_point.len() == 39
            {
                Self::prior_rhs_col_major_39_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (45, 45)
                && prior.rhs.len() == 45
                && current_point.len() == 45
            {
                Self::prior_rhs_col_major_45_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (51, 51)
                && prior.rhs.len() == 51
                && current_point.len() == 51
            {
                Self::prior_rhs_col_major_51_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (57, 57)
                && prior.rhs.len() == 57
                && current_point.len() == 57
            {
                Self::prior_rhs_col_major_57_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else {
                prior.rhs.clone() + &prior.jacobian * current_point
            };
            let rhs = DVector::from_iterator(
                rhs.len(),
                rhs.iter()
                    .copied()
                    .map(|value| if (value as f32) == 0.0 { 0.0 } else { value }),
            );
            if let Some(factor) = prior_factor(columns, rhs) {
                let factor = if prior_state_columns.len() == prior.jacobian.ncols() {
                    factor.with_prior_state_columns(prior_state_columns)
                } else {
                    factor
                };
                factors.push(factor);
            }
        }
        if base.prior.is_none() {
            if let Some(anchor) = &base.anchor_point {
                if anchor.len() == NAV_STATE_DOF && !base.states.is_empty() {
                    let mut jacobian = DMatrix::zeros(NAV_STATE_DOF, base.state_dof());
                    let pose_weight = base.initial_pose_weight.sqrt();
                    let accel_bias_weight = base.initial_accel_bias_weight.sqrt();
                    let gyro_bias_weight = base.initial_gyro_bias_weight.sqrt();
                    if !pose_weight.is_finite()
                        || !accel_bias_weight.is_finite()
                        || !gyro_bias_weight.is_finite()
                    {
                        return factors;
                    }
                    for axis in 0..3 {
                        jacobian[(axis, axis)] = pose_weight;
                    }
                    jacobian[(5, 5)] = pose_weight;
                    if base.visual_only.is_some() {
                        jacobian[(3, 3)] = pose_weight;
                        jacobian[(4, 4)] = pose_weight;
                    }
                    for axis in 0..3 {
                        jacobian[(9 + axis, 9 + axis)] = accel_bias_weight;
                        jacobian[(12 + axis, 12 + axis)] = gyro_bias_weight;
                    }
                    let Some(current_nav) = values.block_nav(base.poses.len()) else {
                        return factors;
                    };
                    let anchor_nav = decode_state_with_mode(anchor, base.scalar_mode);
                    let current = if base.scalar_mode == ScalarMode::UpstreamF32 {
                        values
                            .linearized_delta(WindowBlockKind::State, 0)
                            .filter(|delta| delta.len() == NAV_STATE_DOF)
                            .unwrap_or_else(|| local_state_difference(&current_nav, &anchor_nav))
                    } else {
                        local_state_difference(&current_nav, &anchor_nav)
                    };
                    let mut rhs = DVector::zeros(NAV_STATE_DOF);
                    for index in 0..NAV_STATE_DOF {
                        rhs[index] = if base.scalar_mode == ScalarMode::UpstreamF32 {
                            let weight = jacobian[(index, index)] as f32;
                            let delta = (current[index] as f32) - (anchor[index] as f32);
                            f64::from(weight * delta)
                        } else {
                            jacobian[(index, index)] * (current[index] - anchor[index])
                        };
                    }
                    for value in rhs.iter_mut() {
                        if (*value as f32) == 0.0 {
                            *value = 0.0;
                        }
                    }
                    if let Some(factor) = prior_factor(jacobian, rhs) {
                        factors.push(factor);
                    }
                }
            }
        }
        if let Some(factor) = base.visual_only_factor(|index| values.block_nav(index)) {
            factors.push(factor);
        }
        factors
    }

    pub(super) fn marginal_prior_reduced_cost_view<V: WindowValueView>(
        &self,
        values: &V,
        prior: &WindowPrior,
    ) -> f64 {
        upstream_marginal_prior_error(
            &prior.jacobian,
            &prior.rhs,
            &values.prior_current_point(prior),
            values.base().scalar_mode,
        )
    }

    fn emit_prior_residual_inputs(
        prior: &WindowPrior,
        current_point: &DVector<f64>,
        state_dof: usize,
    ) {
        let Some(path) = diagnostic_env_snapshot().diagnostic_prior_input.as_ref() else {
            return;
        };
        let to_bits = |values: &[f64]| -> Vec<String> {
            values
                .iter()
                .map(|v| format!("{:08x}", (*v as f32).to_bits()))
                .collect()
        };
        let record = json!({
            "schema": "basalt.prior_residual_inputs.v1",
            "global_cols": state_dof,
            "rows": prior.jacobian.nrows(),
            "cols": prior.jacobian.ncols(),
            "jacobian_bits": to_bits(prior.jacobian.as_slice()),
            "stored_rhs_bits": to_bits(prior.rhs.as_slice()),
            "delta_bits": to_bits(current_point.as_slice()),
        });
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.with_extension("producer.jsonl"))
        {
            let _ = writeln!(file, "{}", record);
        }
    }

    /// Evaluate `mld.b + mld.H * delta` with Eigen's column-major dynamic
    /// GEMV schedule for the active 21x21 marginal prior.  Rows 0..19 use
    /// Packet8/Packet4 lanes with a fused multiply-add for each depth and a
    /// fused final add into the stored RHS.  The last row is the scalar
    /// remainder and uses ordinary multiply/add operations for both its
    /// depth reduction and final update.
    pub(super) fn prior_rhs_col_major_21_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        current_point: &DVector<f64>,
    ) -> DVector<f64> {
        debug_assert_eq!(jacobian.shape(), (21, 21));
        debug_assert_eq!(stored_rhs.len(), 21);
        debug_assert_eq!(current_point.len(), 21);
        let mut result = DVector::<f64>::zeros(21);
        for row in 0..20 {
            let mut value = 0.0_f32;
            for column in 0..21 {
                value = (jacobian[(row, column)] as f32)
                    .fused_mul_add(current_point[column] as f32, value);
            }
            // Eigen's packet store is `pmadd(c, alpha=1, mld.b)`.
            let value = value.fused_mul_add(1.0_f32, stored_rhs[row] as f32);
            result[row] = value as f64;
        }

        let mut value = 0.0_f32;
        for column in 0..21 {
            value = Self::prior_mul_add_separate_f32(
                value,
                jacobian[(20, column)] as f32,
                current_point[column] as f32,
            );
        }
        // The scalar remainder stores `mld.b + c0` separately.
        result[20] = Self::prior_add_separate_f32(stored_rhs[20] as f32, value) as f64;
        result
    }

    // Pinned native 392310: packet and scalar tails both use depth FMA,
    // followed by an FMA store with alpha=1 and the existing prior RHS.
    pub(super) fn prior_rhs_col_major_39_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (39, 39));
        assert_eq!((stored_rhs.len(), delta.len()), (39, 39));
        DVector::from_fn(39, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..39 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            value.fused_mul_add(1.0, stored_rhs[row] as f32) as f64
        })
    }

    pub(super) fn prior_rhs_col_major_45_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (45, 45));
        assert_eq!((stored_rhs.len(), delta.len()), (45, 45));
        DVector::from_fn(45, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..45 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            f64::from(value.fused_mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    pub(super) fn prior_rhs_col_major_51_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (51, 51));
        assert_eq!((stored_rhs.len(), delta.len()), (51, 51));
        DVector::from_fn(51, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..51 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            f64::from(value.fused_mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    pub(super) fn prior_rhs_col_major_57_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (57, 57));
        assert_eq!((stored_rhs.len(), delta.len()), (57, 57));
        DVector::from_fn(57, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..57 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            f64::from(value.fused_mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    /// Re-anchor a valid 21-row UpstreamF32 prior with the captured
    /// packet/scalar GEMV association.  Rows 0..19 use a zero accumulator,
    /// fused depth products, and a fused final subtraction from the stored
    /// RHS.  Row 20 is the scalar depth/product tail.  Other prior sizes and
    /// ExtendedF64 retain their existing paths.
    #[inline]
    pub(super) fn prior_reanchor_rhs_col_major_21_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        debug_assert_eq!(jacobian.shape(), (21, 21));
        debug_assert_eq!(stored_rhs.len(), 21);
        debug_assert_eq!(keep_delta.len(), 21);
        let mut result = DVector::<f32>::zeros(21);
        for row in 0..20 {
            let mut value = 0.0_f32;
            for column in 0..21 {
                value = jacobian[(row, column)].fused_mul_add(keep_delta[column], value);
            }
            result[row] = value.fused_mul_add(-1.0_f32, stored_rhs[row]);
        }

        let mut value = 0.0_f32;
        for column in 0..21 {
            value += jacobian[(20, column)] * keep_delta[column];
        }
        result[20] = stored_rhs[20] - value;
        result
    }

    // Pinned native reanchor: 24 packet rows followed by three scalar rows.
    // Keep the product accumulation separate from the final subtraction.
    pub(super) fn prior_reanchor_rhs_col_major_27_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        debug_assert_eq!(jacobian.shape(), (27, 27));
        debug_assert_eq!((stored_rhs.len(), keep_delta.len()), (27, 27));
        DVector::from_fn(27, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..27 {
                value = if row < 24 {
                    jacobian[(row, col)].fused_mul_add(keep_delta[col], value)
                } else {
                    Self::prior_mul_add_separate_f32(value, jacobian[(row, col)], keep_delta[col])
                };
            }
            if row < 24 {
                value.fused_mul_add(-1.0, stored_rhs[row])
            } else {
                stored_rhs[row] - value
            }
        })
    }

    // Native 4e11fe -> 49c380 -> 48cee0: for 33 columns, a single
    // depth block uses FMA even in scalar row 32 (48d1c8). The temporary
    // product is then subtracted separately at 49c420/49c480.
    pub(super) fn prior_reanchor_rhs_col_major_33_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (33, 33));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (33, 33));
        DVector::from_fn(33, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..33 {
                value = jacobian[(row, col)].fused_mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value.fused_mul_add(1.0, 0.0)
        })
    }

    // Pinned native frame24 reanchor: all 39 output lanes accumulate
    // depth with FMA before the separate subtraction from the stored RHS.
    pub(super) fn prior_reanchor_rhs_col_major_39_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (39, 39));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (39, 39));
        DVector::from_fn(39, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..39 {
                value = jacobian[(row, col)].fused_mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    // Pinned 45x45 prior created at MH01 frame31: Eigen's
    // column-major GEMV contracts every depth update, then the caller applies
    // the stored-RHS subtraction as a separate scalar operation.
    pub(super) fn prior_reanchor_rhs_col_major_45_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (45, 45));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (45, 45));
        DVector::from_fn(45, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..45 {
                value = jacobian[(row, col)].fused_mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    // Native event33 (MH01 frame37): the 51x51 keep block uses the same
    // single Eigen GEMV depth block as 33/39/45. Every depth update is fused,
    // while the caller's stored-RHS subtraction remains a separate operation.
    pub(super) fn prior_reanchor_rhs_col_major_51_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (51, 51));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (51, 51));
        DVector::from_fn(51, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..51 {
                value = jacobian[(row, col)].fused_mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    pub(super) fn prior_reanchor_rhs_col_major_57_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (57, 57));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (57, 57));
        DVector::from_fn(57, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..57 {
                value = jacobian[(row, col)].fused_mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    #[inline(never)]
    pub(super) fn prior_mul_add_separate_f32(accumulator: f32, left: f32, right: f32) -> f32 {
        let product = left * right;
        accumulator + product
    }

    #[inline(never)]
    pub(super) fn prior_add_separate_f32(left: f32, right: f32) -> f32 {
        left + right
    }

    pub(super) fn prior_rhs_col_major_27_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (27, 27));
        assert_eq!(stored_rhs.len(), 27);
        assert_eq!(delta.len(), 27);
        DVector::from_fn(27, |row, _| {
            let mut value = 0.0_f32;
            // Pinned float GEMV (0x392310): the scalar remainder also uses
            // vfma231ss for every depth (0x3925f8), then vfma213ss (0x392605).
            // With 27 columns, the entire depth is one block.
            for col in 0..27 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            f64::from(value.fused_mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    pub(super) fn prior_rhs_col_major_33_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (33, 33));
        assert_eq!((stored_rhs.len(), delta.len()), (33, 33));
        DVector::from_fn(33, |row, _| {
            let mut value = 0.0_f32;
            // Native 392310, depth <= 127: one FMA block including scalar tails.
            for col in 0..33 {
                value = (jacobian[(row, col)] as f32).fused_mul_add(delta[col] as f32, value);
            }
            f64::from(value.fused_mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    /// Return the FEJ-local tangent of the blocks named by a square-root
    /// prior.  The prior stores one encoded pose/state point per block; the
    /// solver chart is manifold-local, so a raw vector subtraction would be
    /// wrong for the rotation lanes.
    fn prior_current_point(&self, _state: &DVector<f64>, prior: &WindowPrior) -> DVector<f64> {
        let kinds = prior.kinds();
        let mut current_point = DVector::zeros(prior.fej_point.len());
        let mut prior_offset = 0;
        for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
            let kind = kinds
                .get(prior_index)
                .copied()
                .unwrap_or(WindowBlockKind::State);
            let dof = block_dof(kind);
            if let Some(current_index) = self.block_index_for(*frame_id, kind) {
                let delta = match kind {
                    WindowBlockKind::Pose => self
                        .poses
                        .get(current_index)
                        .filter(|pose| pose.linearized_delta.len() == dof)
                        .map(|pose| pose.linearized_delta.clone()),
                    WindowBlockKind::State => self
                        .states
                        .get(current_index.saturating_sub(self.poses.len()))
                        .filter(|state| state.linearized_delta.len() == dof)
                        .map(|state| state.linearized_delta.clone()),
                    WindowBlockKind::StatePose => self
                        .states
                        .get(current_index.saturating_sub(self.poses.len()))
                        .filter(|state| state.linearized_delta.len() >= POSE_DOF)
                        .map(|state| state.linearized_delta.rows(0, POSE_DOF).into_owned()),
                };
                if let Some(delta) = delta {
                    current_point.rows_mut(prior_offset, dof).copy_from(&delta);
                }
            }
            prior_offset += dof;
        }
        current_point
    }

    /// Basalt's `computeMargPriorError` for a square-root prior.  It reports
    /// the linearized quadratic with the FEJ-independent constant
    /// `0.5 * rᵀr` removed:
    ///
    /// `deltaᵀ Jᵀ (0.5 * J * delta + b)`.
    ///
    /// Consequently this value is allowed to be negative.  The complete
    /// positive residual norm is still used by the row stack for H/b; only
    /// the LM cost comparison uses this source-compatible reduced value.
    pub(super) fn marginal_prior_reduced_cost(
        &self,
        state: &DVector<f64>,
        prior: &WindowPrior,
    ) -> f64 {
        let delta = self.prior_current_point(state, prior);
        upstream_marginal_prior_error(&prior.jacobian, &prior.rhs, &delta, self.scalar_mode)
    }
}

/// Evaluate the constant-free marginal-prior error used by Basalt's
/// `computeMargPriorError`/`linearizeMargPrior` contract.
///
/// The square-root prior stores `J` and `b = Jᵀr` at its FEJ point.  At the
/// current point the source evaluates
/// `deltaᵀ Jᵀ (0.5 J delta + b)` and intentionally omits `0.5 rᵀr`.
/// Keep the scalar cast at this helper boundary so the production f32 path
/// cannot accidentally evaluate the reduced error in f64 and round only the
/// final result.
/// Pinned Eigen prior inner product for 21 rows (ELF 0x3b23e1..0x3b2586).
/// Two eight-lane products are reduced before the five scalar FMA tails.
fn prior_inner_product_21_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 21);
    assert_eq!(right.len(), 21);
    let products: [f32; 16] = std::array::from_fn(|i| left[i] * right[i]);
    let lanes: [f32; 8] = std::array::from_fn(|i| products[i] + products[i + 8]);
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
    for i in 16..21 {
        sum = right[i].fused_mul_add(left[i], sum);
    }
    sum
}

/// Candidate for Eigen's 51-element packet dot used by the frame38 marginal
/// prior. Six Packet8 products are combined by the evaluator's recursive
/// packet tree, then the three scalar coefficients are contracted into the
/// reduced packet sum.
fn prior_inner_product_51_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 51);
    assert_eq!(right.len(), 51);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_fifth = left[i + 32].fused_mul_add(right[i + 32], left[i] * right[i]);
        let second_sixth = left[i + 40].fused_mul_add(right[i + 40], left[i + 8] * right[i + 8]);
        first_fifth + ((left[i + 16] * right[i + 16] + left[i + 24] * right[i + 24]) + second_sixth)
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut total = (half[0] + half[2]) + (half[1] + half[3]);
    for i in 48..51 {
        total = right[i].fused_mul_add(left[i], total);
    }
    total
}

fn prior_inner_product_57_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 57);
    assert_eq!(right.len(), 57);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_fifth = left[i + 32].fused_mul_add(right[i + 32], left[i] * right[i]);
        let second_sixth = left[i + 40].fused_mul_add(right[i + 40], left[i + 8] * right[i + 8]);
        let third_seventh = left[i + 48].fused_mul_add(right[i + 48], left[i + 16] * right[i + 16]);
        first_fifth + ((third_seventh + left[i + 24] * right[i + 24]) + second_sixth)
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    right[56].fused_mul_add(left[56], (half[0] + half[2]) + (half[1] + half[3]))
}

pub(super) fn upstream_marginal_prior_error_57_candidate(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
) -> f64 {
    assert_eq!(jacobian.shape(), (57, 57));
    assert_eq!((rhs.len(), delta.len()), (57, 57));
    let mut left = [0.0_f32; 57];
    let mut right = [0.0_f32; 57];
    for row in 0..57 {
        for col in 0..57 {
            let j = jacobian[(row, col)] as f32;
            let d = delta[col] as f32;
            left[row] = j.fused_mul_add(d, left[row]);
            right[row] = (0.5_f32 * j).fused_mul_add(d, right[row]);
        }
        right[row] += rhs[row] as f32;
    }
    f64::from(prior_inner_product_57_f32(&left, &right))
}

pub(super) fn upstream_marginal_prior_error_51_candidate(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
) -> f64 {
    assert_eq!(jacobian.shape(), (51, 51));
    assert_eq!((rhs.len(), delta.len()), (51, 51));
    let mut left = [0.0_f32; 51];
    let mut right = [0.0_f32; 51];
    for row in 0..51 {
        for col in 0..51 {
            let j = jacobian[(row, col)] as f32;
            let d = delta[col] as f32;
            left[row] = j.fused_mul_add(d, left[row]);
            right[row] = (0.5_f32 * j).fused_mul_add(d, right[row]);
        }
        right[row] += rhs[row] as f32;
    }
    f64::from(prior_inner_product_51_f32(&left, &right))
}

pub(super) fn upstream_marginal_prior_error(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
    scalar_mode: ScalarMode,
) -> f64 {
    debug_assert_eq!(jacobian.ncols(), delta.len());
    debug_assert_eq!(jacobian.nrows(), rhs.len());
    if scalar_mode == ScalarMode::UpstreamF32 && jacobian.shape() == (57, 57) {
        return upstream_marginal_prior_error_57_candidate(jacobian, rhs, delta);
    }
    if scalar_mode == ScalarMode::UpstreamF32 && jacobian.shape() == (51, 51) {
        return upstream_marginal_prior_error_51_candidate(jacobian, rhs, delta);
    }
    if scalar_mode == ScalarMode::UpstreamF32
        && matches!(jacobian.shape(), (33, 33) | (39, 39) | (45, 45))
    {
        // Pinned Eigen sqrt-prior evaluator: column-major FMA GEMVs,
        // followed by four Packet8 products and a scalar FMA remainder.
        // Keep the half-scaled matrix product separate from the left GEMV.
        let size = jacobian.nrows();
        let mut left = [0.0_f32; 45];
        let mut right = [0.0_f32; 45];
        for row in 0..size {
            for col in 0..size {
                let j = jacobian[(row, col)] as f32;
                let d = delta[col] as f32;
                left[row] = j.fused_mul_add(d, left[row]);
                right[row] = (0.5_f32 * j).fused_mul_add(d, right[row]);
            }
            right[row] += rhs[row] as f32;
        }
        let lanes: [f32; 8] = if size == 45 {
            // The 45-row Eigen inner-product kernel fuses packet five into packet
            // one, then adds the middle three packets before the Packet8 predux.
            std::array::from_fn(|i| {
                let first_and_fifth = left[i + 32].fused_mul_add(right[i + 32], left[i] * right[i]);
                let middle = left[i + 8] * right[i + 8]
                    + (left[i + 16] * right[i + 16] + left[i + 24] * right[i + 24]);
                first_and_fifth + middle
            })
        } else {
            let products: [f32; 32] = std::array::from_fn(|i| left[i] * right[i]);
            std::array::from_fn(|i| {
                products[i] + ((products[i + 16] + products[i + 24]) + products[i + 8])
            })
        };
        let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
        let mut total = (half[0] + half[2]) + (half[1] + half[3]);
        let tail_start = if size == 45 { 40 } else { 32 };
        for i in tail_start..size {
            total = right[i].fused_mul_add(left[i], total);
        }
        return total as f64;
    }
    if scalar_mode == ScalarMode::UpstreamF32 {
        let jacobian = DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, col| {
            jacobian[(row, col)] as f32
        });
        let delta = DVector::from_iterator(delta.len(), delta.iter().map(|value| *value as f32));
        let rhs = DVector::from_iterator(rhs.len(), rhs.iter().map(|value| *value as f32));
        let j_delta = &jacobian * &delta;
        let linearized_rhs = 0.5_f32 * &j_delta + rhs;
        if j_delta.len() == 21 {
            return prior_inner_product_21_f32(j_delta.as_slice(), linearized_rhs.as_slice())
                as f64;
        }
        (delta.transpose() * jacobian.transpose() * linearized_rhs)[(0, 0)] as f64
    } else {
        let j_delta = jacobian * delta;
        (delta.transpose() * jacobian.transpose() * (0.5 * j_delta + rhs))[(0, 0)]
    }
}
