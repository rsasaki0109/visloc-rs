//! Square-root IMU factors used by the Basalt AOM adapter.
//!
//! Rows are always ordered `[position, rotation, velocity]`.  The covariance
//! is converted to a square-root information matrix before it is applied to
//! both the residual and both navigation-state Jacobian blocks.

use crate::fma::FusedMulAdd;
use crate::imu::ImuPreintegratedDelta;
use crate::vio::landmarks::{sophus_so3_inverse, sophus_so3_product};
use crate::BasaltNavState;
use nalgebra::{DMatrix, DVector, Quaternion, SMatrix, UnitQuaternion, Vector3};
use serde_json::json;
use std::{env, fs::OpenOptions, io::Write as IoWrite, sync::OnceLock};

pub const IMU_RESIDUAL_DOF: usize = 9;
pub const NAV_STATE_DOF: usize = 15;
pub type Matrix9 = SMatrix<f64, 9, 9>;
pub type Matrix9F32 = SMatrix<f32, 9, 9>;
pub type Matrix9x30F32 = SMatrix<f32, 9, 30>;
pub type Matrix30F32 = SMatrix<f32, 30, 30>;
pub type Vector30F32 = SMatrix<f32, 30, 1>;

/// Apply the production fixed-size IMU whitener to already-materialized raw
/// blocks.  This is an audit seam for comparing the Eigen expression
/// `get_sqrt_cov_inv() * value` without rebuilding a full navigation state;
/// it intentionally delegates to the same private helpers used by
/// [`whitened_preintegration_factor_upstream_f32_fej`].
#[doc(hidden)]
pub fn diagnostic_whiten_imu_raw_f32(
    whitener: &Matrix9F32,
    residual: &SMatrix<f32, 9, 1>,
    start: &Matrix9F32,
    gyro_bias: &SMatrix<f32, 9, 3>,
    accel_bias: &SMatrix<f32, 9, 3>,
    end: &Matrix9F32,
) -> (SMatrix<f32, 9, 1>, Matrix9x30F32) {
    let whitened_residual = eigen_whiten_residual_f32(whitener, residual);
    let whitened_start = eigen_whiten_block_f32::<9>(whitener, start);
    let whitened_gyro_bias = eigen_whiten_block_f32::<3>(whitener, gyro_bias);
    let whitened_accel_bias = eigen_whiten_block_f32::<3>(whitener, accel_bias);
    let whitened_end = eigen_whiten_block_f32::<9>(whitener, end);
    let mut whitened_jacobian = Matrix9x30F32::zeros();
    whitened_jacobian
        .fixed_view_mut::<9, 9>(0, 0)
        .copy_from(&whitened_start);
    whitened_jacobian
        .fixed_view_mut::<9, 3>(0, 9)
        .copy_from(&whitened_gyro_bias);
    whitened_jacobian
        .fixed_view_mut::<9, 3>(0, 12)
        .copy_from(&whitened_accel_bias);
    whitened_jacobian
        .fixed_view_mut::<9, 9>(0, 15)
        .copy_from(&whitened_end);
    (whitened_residual, whitened_jacobian)
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhitenedImuFactor {
    /// Columns are `[state_i(15), state_j(15)]`.
    pub state_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
    pub sqrt_information: Matrix9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuFactorError {
    InvalidDelta,
    InvalidCovariance,
}

/// Noise densities for continuous-time gyro/accelerometer bias random walks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasRandomWalkNoise {
    pub gyro_density: f64,
    pub accel_density: f64,
}

impl BiasRandomWalkNoise {
    pub fn is_valid(self) -> bool {
        self.gyro_density.is_finite()
            && self.gyro_density > 0.0
            && self.accel_density.is_finite()
            && self.accel_density > 0.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhitenedBiasWalkFactor {
    /// Columns are `[state_i(15), state_j(15)]`.
    pub state_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
}

/// Build a covariance- and bias-corrected preintegration factor.
pub fn whitened_preintegration_factor(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Result<WhitenedImuFactor, ImuFactorError> {
    if !delta.delta_time.is_finite() || delta.delta_time <= 0.0 {
        return Err(ImuFactorError::InvalidDelta);
    }
    let sqrt_information = sqrt_information(&delta.covariance)?;
    let raw = residual(from, to, delta, gravity_world);
    // Basalt's ImuBlock uses the closed-form Jacobians from
    // IntegratedImuMeasurement::residual.  A finite-difference Jacobian is
    // especially divergent for the f32 port (the perturbation is below the
    // resolution of the state), so keep the source convention here too.
    let jacobian = residual_jacobian(from, to, delta, gravity_world);
    let whitened_fixed = sqrt_information * raw;
    let dense_fixed = sqrt_information * jacobian;
    let whitened = DVector::from_iterator(9, whitened_fixed.iter().copied());
    let dense = DMatrix::from_iterator(9, 2 * NAV_STATE_DOF, dense_fixed.iter().copied());
    if !whitened.iter().all(|x| x.is_finite()) || !dense.iter().all(|x| x.is_finite()) {
        return Err(ImuFactorError::InvalidCovariance);
    }
    Ok(WhitenedImuFactor {
        state_jacobian: dense,
        residual: whitened,
        sqrt_information,
    })
}

/// Build an IMU factor with Basalt's active `Scalar=float` arithmetic at the
/// whitening/product boundary.  The caller stores navigation states as f64,
/// so the unwhitened residual/Jacobian are first narrowed to f32 and then
/// multiplied by the f32 LDLT whitener, matching Eigen's template instantiation
/// before the result is widened for the surrounding Rust API.
pub fn whitened_preintegration_factor_upstream_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Result<WhitenedImuFactor, ImuFactorError> {
    whitened_preintegration_factor_upstream_f32_fej(from, to, from, to, delta, gravity_world)
}

/// Upstream float factor with Basalt's FEJ split: residual is evaluated at
/// `from/to`, while analytic derivatives remain at `jac_from/jac_to` when a
/// state has been linearized.
pub fn whitened_preintegration_factor_upstream_f32_fej(
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Result<WhitenedImuFactor, ImuFactorError> {
    whitened_preintegration_factor_upstream_f32_fej_mode(
        from,
        to,
        jac_from,
        jac_to,
        delta,
        gravity_world,
        false,
    )
}

pub(crate) fn whitened_preintegration_factor_upstream_f32_fej_mode(
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
    recompute_current: bool,
) -> Result<WhitenedImuFactor, ImuFactorError> {
    if !delta.delta_time.is_finite() || delta.delta_time <= 0.0 {
        return Err(ImuFactorError::InvalidDelta);
    }
    let covariance = delta.covariance.map(|value| value as f32);
    let sqrt_information = sqrt_information_f32(&covariance)?;
    // Basalt instantiates the complete residual/Jacobian evaluation with
    // Scalar=float. Narrowing only after an f64 evaluation changes the last
    // bits of the SO(3) log and of the matrix products enough to move the
    // large state/bias row products. Keep a dedicated f32 path here.
    let raw = if recompute_current {
        residual_f32_with_trial_velocity::<false, true>(from, to, delta, gravity_world)
    } else {
        residual_f32(from, to, delta, gravity_world)
    };
    let jacobian = residual_jacobian_f32(jac_from, jac_to, delta, gravity_world);
    let whitened = eigen_whiten_residual_f32(&sqrt_information, &raw);
    // Basalt's production ImuBlock does not form one wide 9x30 product here.
    // It assigns four fixed-size products independently:
    // `W*d_res_d_start`, `W*d_res_d_end`, `W*d_res_d_bg`, and
    // `W*d_res_d_ba`.  Eigen's packet/tail schedule depends on the RHS width,
    // so preserving those block boundaries is observable at the f32 bits.
    let dense = eigen_whiten_jacobian_blockwise_f32(&sqrt_information, &jacobian);
    if !whitened.iter().all(|x| x.is_finite()) || !dense.iter().all(|x| x.is_finite()) {
        return Err(ImuFactorError::InvalidCovariance);
    }
    let residual = DVector::from_iterator(9, whitened.iter().map(|value| *value as f64));
    let state_jacobian = DMatrix::from_fn(9, 30, |row, col| dense[(row, col)] as f64);
    Ok(WhitenedImuFactor {
        state_jacobian,
        residual,
        sqrt_information: sqrt_information.map(|value| value as f64),
    })
}

/// Return Basalt's unwhitened `[position, rotation, velocity]` residual and
/// 9x30 state Jacobian.  This narrow diagnostic seam is intentionally kept
/// free of covariance arithmetic so a scalar-boundary audit can compare the
/// raw contract before whitening and row products.
pub fn preintegration_residual_and_jacobian(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> (DVector<f64>, DMatrix<f64>) {
    (
        residual(from, to, delta, gravity_world),
        residual_jacobian(from, to, delta, gravity_world),
    )
}

/// Return the unwhitened residual evaluated with Basalt's active `Scalar=float`
/// arithmetic.  This is primarily an audit seam: the estimator widens the
/// result only after the f32 residual/Jacobian and LDLT row products have been
/// formed.
pub fn preintegration_residual_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> SMatrix<f32, 9, 1> {
    residual_f32(from, to, delta, gravity_world)
}

/// Return the unwhitened 9x30 Jacobian evaluated with Basalt's active
/// `Scalar=float` arithmetic.
pub fn preintegration_jacobian_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> SMatrix<f32, 9, 30> {
    residual_jacobian_f32(from, to, delta, gravity_world)
}

/// Return every f32 stage at the IMU whitening boundary.  This is an audit
/// seam only: the estimator uses the same helpers below, so exposing the
/// tuple cannot introduce a second arithmetic path.
pub fn preintegration_f32_stages(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Result<
    (
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
        Matrix9F32,
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
    ),
    ImuFactorError,
> {
    preintegration_f32_stages_fej(from, to, from, to, delta, gravity_world)
}

/// Return every f32 stage at the IMU whitening boundary while preserving
/// Basalt's FEJ split: the residual is evaluated at `from/to`, whereas the
/// analytic Jacobian is evaluated at `jac_from/jac_to`.  This is an audit
/// seam only; the production factor uses the same primitive stages above.
pub fn preintegration_f32_stages_fej(
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Result<
    (
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
        Matrix9F32,
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
    ),
    ImuFactorError,
> {
    let covariance = delta.covariance.map(|value| value as f32);
    let whitener = sqrt_information_f32(&covariance)?;
    let raw = residual_f32(from, to, delta, gravity_world);
    let jacobian = residual_jacobian_f32(jac_from, jac_to, delta, gravity_world);
    let whitened = eigen_whiten_residual_f32(&whitener, &raw);
    // Keep the audit stages on the same four fixed-size Eigen products as
    // `whitened_preintegration_factor_upstream_f32_fej`.  The historical
    // wide 9x30 helper remains available for its standalone evaluator audit,
    // but it is not the expression used by Basalt's ImuBlock.
    let dense = eigen_whiten_jacobian_blockwise_f32(&whitener, &jacobian);
    Ok((raw, jacobian, whitener, whitened, dense))
}

/// Diagnostic stages with the same explicit current-recomputation mode as the window.
pub(crate) fn preintegration_f32_stages_fej_mode(
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
    recompute_current: bool,
) -> Result<
    (
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
        Matrix9F32,
        SMatrix<f32, 9, 1>,
        Matrix9x30F32,
    ),
    ImuFactorError,
> {
    let (mut raw, jacobian, whitener, mut whitened, dense) =
        preintegration_f32_stages_fej(from, to, jac_from, jac_to, delta, gravity_world)?;
    if recompute_current {
        raw = residual_f32_with_trial_velocity::<false, true>(from, to, delta, gravity_world);
        whitened = eigen_whiten_residual_f32(&whitener, &raw);
    }
    Ok((raw, jacobian, whitener, whitened, dense))
}

/// Source-order f32 expression trace for comparison with the pinned native
/// `ImuBlock::linearizeImu` boundary. This is diagnostic-only and returns
/// binary32 bit strings in the same field/column-major order as the native
/// logger; it is not used by the estimator's arithmetic path.
pub fn preintegration_f32_expression_trace(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> serde_json::Value {
    let bits = |x: f32| format!("{:08x}", x.to_bits());
    let vec_bits = |v: &[f32]| v.iter().copied().map(bits).collect::<Vec<_>>();
    let mat_bits = |m: &SMatrix<f32, 3, 3>| {
        (0..3)
            .flat_map(|c| (0..3).map(move |r| bits(m[(r, c)])))
            .collect::<Vec<_>>()
    };
    let bg = from.gyro_bias_rad_s.map(|x| x as f32);
    let ba = from.accel_bias_m_s2.map(|x| x as f32);
    let dbg = SMatrix::<f32, 9, 3>::from_fn(|r, c| match r {
        0..=2 => delta.jacobian_position_gyro_bias[(r, c)] as f32,
        3..=5 => delta.jacobian_rotation_gyro_bias[(r - 3, c)] as f32,
        _ => delta.jacobian_velocity_gyro_bias[(r - 6, c)] as f32,
    });
    let dba = SMatrix::<f32, 9, 3>::from_fn(|r, c| match r {
        0..=2 => delta.jacobian_position_accel_bias[(r, c)] as f32,
        3..=5 => 0.0,
        _ => delta.jacobian_velocity_accel_bias[(r - 6, c)] as f32,
    });
    let bg_diff = dbg * (bg - delta.bias_gyro.map(|x| x as f32));
    let ba_diff = dba * (ba - delta.bias_accel.map(|x| x as f32));
    let dr = so3_mul_f32(
        so3_exp_f32(bg_diff.fixed_rows::<3>(3).into_owned()),
        quat_raw_f32(&delta.delta_rotation),
    );
    let dt = delta.delta_time as f32;
    let q0 = quat_state_f32(&from.imu_to_world.rotation);
    let q1 = quat_state_f32(&to.imu_to_world.rotation);
    let r0_inv = eigen_rotation_matrix_f32(&sophus_so3_inverse(q0));
    let gravity = gravity_world.map(|x| x as f32);
    let tmp = eigen_matrix_vector_f32(
        &r0_inv,
        eigen_translation_operand_f32(
            to.imu_to_world.translation.map(|x| x as f32),
            from.imu_to_world.translation.map(|x| x as f32),
            from.velocity_world_m_s.map(|x| x as f32),
            gravity,
            dt,
        ),
    );
    let tmp2 = eigen_matrix_vector_f32(
        &r0_inv,
        eigen_velocity_operand_f32(
            to.velocity_world_m_s.map(|x| x as f32),
            from.velocity_world_m_s.map(|x| x as f32),
            gravity,
            dt,
        ),
    );
    let velocity_operand = eigen_velocity_operand_f32(
        to.velocity_world_m_s.map(|x| x as f32),
        from.velocity_world_m_s.map(|x| x as f32),
        gravity,
        dt,
    );
    let logv = so3_log_f32(so3_mul_f32(so3_mul_f32(dr, sophus_so3_inverse(q1)), q0));
    let jac = residual_jacobian_f32(from, to, delta, gravity_world);
    let block =
        |r: usize, c: usize| -> SMatrix<f32, 3, 3> { SMatrix::from_fn(|i, j| jac[(r + i, c + j)]) };
    json!({
        "schema": "basalt.m7im15.native_expr_trace.v1",
        "dt": bits(dt),
        "translation_delta": vec_bits((to.imu_to_world.translation.map(|x| x as f32) - from.imu_to_world.translation.map(|x| x as f32)).as_slice()),
        "velocity_term": vec_bits((from.velocity_world_m_s.map(|x| x as f32) * dt).as_slice()),
        "gravity_term": vec_bits(((gravity * 0.5) * dt * dt).as_slice()),
        "translation_operand": vec_bits(eigen_translation_operand_f32(
            to.imu_to_world.translation.map(|x| x as f32),
            from.imu_to_world.translation.map(|x| x as f32),
            from.velocity_world_m_s.map(|x| x as f32),
            gravity,
            dt,
        ).as_slice()),
        "velocity_operand": vec_bits(velocity_operand.as_slice()),
        "bg_diff_position": vec_bits(bg_diff.fixed_rows::<3>(0).as_slice()),
        "bg_diff_rotation": vec_bits(bg_diff.fixed_rows::<3>(3).as_slice()),
        "bg_diff_velocity": vec_bits(bg_diff.fixed_rows::<3>(6).as_slice()),
        "ba_diff_position": vec_bits(ba_diff.fixed_rows::<3>(0).as_slice()),
        "ba_diff_velocity": vec_bits(ba_diff.fixed_rows::<3>(6).as_slice()),
        "R0_inv": mat_bits(&r0_inv), "dR0_translation": vec_bits(tmp.as_slice()),
        "dR0_velocity": vec_bits(tmp2.as_slice()), "relative_log": vec_bits(logv.as_slice()),
        "delta_position": vec_bits(delta.delta_position.map(|x| x as f32).as_slice()),
        "delta_velocity": vec_bits(delta.delta_velocity.map(|x| x as f32).as_slice()),
        "J_start_position": mat_bits(&block(0, 0)),
        "J_start_rotation": mat_bits(&block(0, 3)),
        "J_start_rot_bias": mat_bits(&block(3, 9)),
        "J_gyro_bias_rotation": mat_bits(&block(3, 9)),
    })
}

/// Return the named SO(3) producer stages used by the FEJ residual call.
///
/// This is a diagnostic-only view of the same f32 primitives consumed by
/// `residual_f32`: it records the bias correction, `SO3::exp`, the stored
/// delta rotation, each left-to-right quaternion product, and the final log.
/// Keeping this at the factor boundary makes it possible to compare the
/// native Eigen/Sophus operands before Jacobian construction without adding a
/// second numerical path to the estimator.
pub fn preintegration_f32_rotation_producer_trace(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
) -> serde_json::Value {
    let bits = |x: f32| format!("{:08x}", x.to_bits());
    let vec_bits = |v: &[f32]| v.iter().copied().map(bits).collect::<Vec<_>>();
    let quat_bits = |q: &UnitQuaternion<f32>| {
        let q = q.quaternion();
        vec_bits(&[q.i, q.j, q.k, q.w])
    };
    let dbg = SMatrix::<f32, 9, 3>::from_fn(|row, col| match row {
        0..=2 => delta.jacobian_position_gyro_bias[(row, col)] as f32,
        3..=5 => delta.jacobian_rotation_gyro_bias[(row - 3, col)] as f32,
        _ => delta.jacobian_velocity_gyro_bias[(row - 6, col)] as f32,
    });
    let bg = from.gyro_bias_rad_s.map(|value| value as f32);
    let bias_lin = delta.bias_gyro.map(|value| value as f32);
    let bg_diff = eigen_matrix_vector_9x3_f32(&dbg, bg - bias_lin);
    let phi = bg_diff.fixed_rows::<3>(3).into_owned();
    let exp_q = so3_exp_f32(phi);
    let delta_q = quat_raw_f32(&delta.delta_rotation);
    let q0 = quat_state_f32(&from.imu_to_world.rotation);
    let q1 = quat_state_f32(&to.imu_to_world.rotation);
    let q1_inv = sophus_so3_inverse(q1);
    let product_exp_delta = so3_mul_f32(exp_q, delta_q);
    let product_q1 = so3_mul_f32(product_exp_delta, q1_inv);
    let product_q0 = so3_mul_f32(product_q1, q0);
    let rotation_log = so3_log_f32(product_q0);
    let product_q = product_q0.quaternion();
    let scalar_squared_n =
        product_q.i * product_q.i + product_q.j * product_q.j + product_q.k * product_q.k;
    let eigen_squared_n =
        eigen_squared_norm3_f32(Vector3::new(product_q.i, product_q.j, product_q.k));
    let scalar_n = scalar_squared_n.sqrt();
    let eigen_n = eigen_squared_n.sqrt();
    let scalar_atan = scalar_n.atan2(product_q.w);
    let eigen_atan = eigen_n.atan2(product_q.w);
    let ratio = eigen_n / product_q.w;
    let ratio_atan = ratio.atan();
    let scalar_factor = 2.0_f32 * scalar_n.atan2(product_q.w) / scalar_n;
    let eigen_factor = 2.0_f32 * eigen_n.atan2(product_q.w) / eigen_n;
    json!({
        "bg_f32_bits": vec_bits(bg.as_slice()),
        "bias_gyro_lin_f32_bits": vec_bits(bias_lin.as_slice()),
        "bg_diff_f32_bits": vec_bits(bg_diff.as_slice()),
        "bg_diff_rotation_f32_bits": vec_bits(phi.as_slice()),
        "exp_quaternion_xyzw_f32_bits": quat_bits(&exp_q),
        "delta_quaternion_xyzw_f32_bits": quat_bits(&delta_q),
        "q0_quaternion_xyzw_f32_bits": quat_bits(&q0),
        "q1_quaternion_xyzw_f32_bits": quat_bits(&q1),
        "q1_inverse_quaternion_xyzw_f32_bits": quat_bits(&q1_inv),
        "product_exp_delta_xyzw_f32_bits": quat_bits(&product_exp_delta),
        "product_q1_xyzw_f32_bits": quat_bits(&product_q1),
        "product_q0_xyzw_f32_bits": quat_bits(&product_q0),
        "product_q0_scalar_squared_n_bits": bits(scalar_squared_n),
        "product_q0_eigen_squared_n_bits": bits(eigen_squared_n),
        "product_q0_scalar_n_bits": bits(scalar_n),
        "product_q0_eigen_n_bits": bits(eigen_n),
        "product_q0_scalar_atan_bits": bits(scalar_atan),
        "product_q0_eigen_atan_bits": bits(eigen_atan),
        "product_q0_ratio_bits": bits(ratio),
        "product_q0_ratio_atan_bits": bits(ratio_atan),
        "product_q0_scalar_factor_bits": bits(scalar_factor),
        "product_q0_eigen_factor_bits": bits(eigen_factor),
        "rotation_log_f32_bits": vec_bits(rotation_log.as_slice()),
    })
}

/// Build the same unwhitened 9x30 Jacobian as Basalt's
/// `IntegratedImuMeasurement::residual`.
fn residual_jacobian(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> DMatrix<f64> {
    let (dr, _, _) = delta.corrected(from.gyro_bias_rad_s, from.accel_bias_m_s2);
    let dt = delta.delta_time;
    let r0_inv = from
        .imu_to_world
        .rotation
        .inverse()
        .to_rotation_matrix()
        .into_inner();
    let tmp = r0_inv
        * (to.imu_to_world.translation
            - from.imu_to_world.translation
            - from.velocity_world_m_s * dt
            - gravity_world * (0.5 * dt * dt));
    let tmp2 = r0_inv * (to.velocity_world_m_s - from.velocity_world_m_s - gravity_world * dt);
    let rotation =
        (dr * to.imu_to_world.rotation.inverse() * from.imu_to_world.rotation).scaled_axis();
    let right_inv = right_jacobian_inverse_so3(rotation);
    let left_inv = left_jacobian_inverse_so3(rotation);

    let mut jacobian = DMatrix::zeros(IMU_RESIDUAL_DOF, 2 * NAV_STATE_DOF);
    let skew_tmp = hat(tmp);
    let skew_tmp2 = hat(tmp2);

    // State at the beginning of the interval.
    set_block(&mut jacobian, 0, 0, -r0_inv);
    set_block(&mut jacobian, 0, 3, skew_tmp * r0_inv);
    set_block(&mut jacobian, 3, 3, right_inv * r0_inv);
    set_block(&mut jacobian, 6, 3, skew_tmp2 * r0_inv);
    set_block(&mut jacobian, 0, 6, -r0_inv * dt);
    set_block(&mut jacobian, 6, 6, -r0_inv);
    set_block(&mut jacobian, 0, 9, -delta.jacobian_position_gyro_bias);
    set_block(&mut jacobian, 0, 12, -delta.jacobian_position_accel_bias);
    set_block(&mut jacobian, 3, 9, -delta.jacobian_rotation_gyro_bias);
    set_block(&mut jacobian, 6, 9, -delta.jacobian_velocity_gyro_bias);
    set_block(&mut jacobian, 6, 12, -delta.jacobian_velocity_accel_bias);
    // The rotation component of the gyro-bias correction is itself a
    // left-perturbation on SO(3), hence Sophus' left Jacobian inverse.
    set_block(
        &mut jacobian,
        3,
        9,
        left_inv * delta.jacobian_rotation_gyro_bias,
    );

    // State at the end of the interval.
    let end_col = NAV_STATE_DOF;
    set_block(&mut jacobian, 0, end_col, r0_inv);
    set_block(&mut jacobian, 3, end_col + 3, -right_inv * r0_inv);
    set_block(&mut jacobian, 6, end_col + 6, r0_inv);

    // Bias columns above are expressed in the start-state block.  The
    // position/velocity blocks are direct correction Jacobians; the rotation
    // block replaces the direct matrix with the SO(3) left-Jacobian mapping.
    // Keep the current-bias finite-difference-compatible residual convention
    // by retaining the state values in `residual` itself.
    jacobian
}

fn hat(v: Vector3<f64>) -> SMatrix<f64, 3, 3> {
    SMatrix::<f64, 3, 3>::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn so3_jacobian_inverse(phi: Vector3<f64>, sign: f64) -> SMatrix<f64, 3, 3> {
    let theta2 = phi.dot(&phi);
    let a = if theta2 < 1e-12 {
        1.0 / 12.0 + theta2 / 720.0
    } else {
        let theta = theta2.sqrt();
        (1.0 - 0.5 * theta * (theta * 0.5).cos() / (theta * 0.5).sin()) / theta2
    };
    let h = hat(phi);
    SMatrix::identity() + h * (0.5 * sign) + h * h * a
}

fn right_jacobian_inverse_so3(phi: Vector3<f64>) -> SMatrix<f64, 3, 3> {
    so3_jacobian_inverse(phi, 1.0)
}

fn left_jacobian_inverse_so3(phi: Vector3<f64>) -> SMatrix<f64, 3, 3> {
    so3_jacobian_inverse(phi, -1.0)
}

fn set_block(matrix: &mut DMatrix<f64>, row: usize, col: usize, block: SMatrix<f64, 3, 3>) {
    for r in 0..3 {
        for c in 0..3 {
            matrix[(row + r, col + c)] = block[(r, c)];
        }
    }
}

fn set_block_f32(matrix: &mut Matrix9x30F32, row: usize, col: usize, block: SMatrix<f32, 3, 3>) {
    for r in 0..3 {
        for c in 0..3 {
            matrix[(row + r, col + c)] = block[(r, c)];
        }
    }
}

fn hat_f32(v: Vector3<f32>) -> SMatrix<f32, 3, 3> {
    SMatrix::<f32, 3, 3>::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Sophus' SO3::exp operation sequence at Scalar=float.
fn so3_exp_f32(omega: Vector3<f32>) -> UnitQuaternion<f32> {
    let theta_sq = omega.dot(&omega);
    let (imag_factor, real_factor) = if theta_sq < f32::EPSILON * f32::EPSILON {
        let theta_po4 = theta_sq * theta_sq;
        (
            0.5_f32 - (1.0_f32 / 48.0_f32) * theta_sq + (1.0_f32 / 3840.0_f32) * theta_po4,
            1.0_f32 - (1.0_f32 / 8.0_f32) * theta_sq + (1.0_f32 / 384.0_f32) * theta_po4,
        )
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5_f32 * theta;
        (half_theta.sin() / theta, half_theta.cos())
    };
    // Sophus::SO3::exp writes the quaternion coefficients directly.  It does
    // not route this result through the normalizing SO3 constructor.
    UnitQuaternion::from_quaternion(Quaternion::new(
        real_factor,
        imag_factor * omega.x,
        imag_factor * omega.y,
        imag_factor * omega.z,
    ))
}

/// Sophus' explicit quaternion product (Eigen's Quaternion::operator* is not
/// used by Sophus because it must support mixed scalar/JET products).
fn so3_mul_f32(a: UnitQuaternion<f32>, b: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    // Sophus::SO3 multiplication delegates to Eigen's Packet4f quaternion
    // product and packet normalization.  Reuse the audited crate helper so
    // this factor boundary has the same fused lane schedule as the estimator.
    sophus_so3_product(a, b)
}

/// Sophus' atan2-based SO3::log operation sequence at Scalar=float.
///
/// Eigen's pinned Linux build resolves `std::atan2(float, float)` to the
/// glibc `atan2f` implementation.  For finite inputs that implementation
/// first reduces the positive-x quadrant to `atanf(y / x)` and then applies
/// the float quadrant correction.  MSVC's `f32::atan2` follows a different
/// (double-backed) libm path at a few rounding boundaries, so keep the
/// source-level float schedule explicit here.  The branch structure is
/// generic; the IMU rotations normally remain in the positive-x quadrant.
#[inline]
fn native_atan2_f32(y: f32, x: f32) -> f32 {
    if x > 0.0_f32 {
        (y / x).atan()
    } else if x < 0.0_f32 {
        let a = (y / x).abs().atan();
        if y >= 0.0_f32 {
            std::f32::consts::PI - a
        } else {
            a - std::f32::consts::PI
        }
    } else if y > 0.0_f32 {
        std::f32::consts::FRAC_PI_2
    } else if y < 0.0_f32 {
        -std::f32::consts::FRAC_PI_2
    } else {
        y + x
    }
}

fn so3_log_f32(rotation: UnitQuaternion<f32>) -> Vector3<f32> {
    let q = rotation.quaternion();
    let squared_n = eigen_squared_norm3_f32(Vector3::new(q.i, q.j, q.k));
    let w = q.w;
    let two_atan_nbyw_by_n = if squared_n < f32::EPSILON * f32::EPSILON {
        let squared_w = w * w;
        2.0_f32 / w - (2.0_f32 / 3.0_f32) * squared_n / (w * squared_w)
    } else {
        let n = squared_n.sqrt();
        let atan_nbyw = if w < 0.0_f32 {
            native_atan2_f32(-n, -w)
        } else {
            native_atan2_f32(n, w)
        };
        2.0_f32 * atan_nbyw / n
    };
    Vector3::new(
        two_atan_nbyw_by_n * q.i,
        two_atan_nbyw_by_n * q.j,
        two_atan_nbyw_by_n * q.k,
    )
}

fn quat_state_f32(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    // The state already arrives from the f32 Sophus update boundary.  Preserve
    // its binary32 coefficients here; rebuilding a normalized nalgebra
    // quaternion performs a second, different norm reduction before the
    // factor's single Sophus inverse/product normalization.
    UnitQuaternion::new_unchecked(Quaternion::new(
        rotation.w as f32,
        rotation.i as f32,
        rotation.j as f32,
        rotation.k as f32,
    ))
}

/// Convert a normalized quaternion using Eigen's f32 `QuaternionBase::toRotationMatrix`
/// operation schedule.  nalgebra uses an algebraically equivalent square-term
/// schedule (`ww + ii - jj - kk`) whose rounding differs from Eigen's `2*x`
/// temporaries and `1 - (tyy + tzz)` form.  Keep this helper local to the
/// upstream-f32 factor boundary; the surrounding f64 API remains unchanged.
/// Eigen's contracted signed cross terms are spelled explicitly with
/// `mul_add`, while the temporary products and diagonal sums stay separate.
fn eigen_rotation_matrix_f32(rotation: &UnitQuaternion<f32>) -> SMatrix<f32, 3, 3> {
    let q = rotation.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txx = tx * q.i;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;

    SMatrix::<f32, 3, 3>::new(
        1.0_f32 - (tyy + tzz),
        (-tz).fused_mul_add(q.w, txy),
        ty.fused_mul_add(q.w, txz),
        tz.fused_mul_add(q.w, txy),
        1.0_f32 - (txx + tzz),
        (-tx).fused_mul_add(q.w, tyz),
        (-ty).fused_mul_add(q.w, txz),
        tx.fused_mul_add(q.w, tyz),
        1.0_f32 - (txx + tyy),
    )
}

// Pinned out-of-line QuaternionBase::toRotationMatrix (ELF 2b1dd0),
// reached by both primary and current-state residual calls in linearizeImu.
// The separately verified inline trial schedule remains unchanged.
fn eigen_current_rotation_matrix_f32(rotation: &UnitQuaternion<f32>) -> SMatrix<f32, 3, 3> {
    let q = rotation.quaternion();
    let mut result = eigen_rotation_matrix_f32(rotation);
    let tx = q.i + q.i;
    result[(1, 1)] = 1.0 - q.i.fused_mul_add(tx, q.k * (q.k + q.k));
    result[(2, 2)] = 1.0 - q.i.fused_mul_add(tx, q.j * (q.j + q.j));
    result
}

/// Eigen's lazy `Vector3` expression used as the right-hand side of the
/// position residual's `R0_inv * (...)` product.  With the pinned AVX build,
/// the expression evaluator contracts the two subtraction terms into the
/// running difference instead of first materializing `velocity * dt` and the
/// gravity term.  The distinction is one ULP on the live frame-4 ordinal-1
/// and ordinal-2 links, and then reaches the position residual directly.
///
/// Keep this helper separate from the named diagnostic operands above: those
/// operands are materialized source expressions, while this is the actual
/// lazy RHS schedule consumed by the production 3x3 product.
fn eigen_translation_operand_f32(
    translation_to: Vector3<f32>,
    translation_from: Vector3<f32>,
    velocity_from: Vector3<f32>,
    gravity: Vector3<f32>,
    dt: f32,
) -> Vector3<f32> {
    let difference = translation_to - translation_from;
    // Eigen's packet evaluator materializes the gravity half first and only
    // then multiplies it by dt.  `(0.5 * dt) * gravity` is algebraically
    // equivalent but changes the binary32 rounding tree on the live link.
    let half_gravity = gravity * 0.5_f32;
    let gravity_dt = half_gravity * dt;
    Vector3::new(
        (-gravity_dt.x).fused_mul_add(dt, (-dt).fused_mul_add(velocity_from.x, difference.x)),
        (-gravity_dt.y).fused_mul_add(dt, (-dt).fused_mul_add(velocity_from.y, difference.y)),
        (-gravity_dt.z).fused_mul_add(dt, (-dt).fused_mul_add(velocity_from.z, difference.z)),
    )
}

/// Eigen's lazy `Vector3` expression for the velocity residual RHS.  The
/// upstream expression `state1.vel_w_i - state0.vel_w_i - g * dt` contracts
/// each component's gravity multiply/add, while materializing the velocity
/// difference as the addend.  Keep that componentwise FMA schedule explicit
/// at the active f32 boundary.
fn eigen_velocity_operand_f32(
    velocity_to: Vector3<f32>,
    velocity_from: Vector3<f32>,
    gravity: Vector3<f32>,
    dt: f32,
) -> Vector3<f32> {
    Vector3::from_fn(|index, _| {
        (-gravity[index]).fused_mul_add(dt, velocity_to[index] - velocity_from[index])
    })
}

/// Eigen's fixed 3x3 column-major matrix-vector product on the pinned AVX2
/// build.  The coefficient-based product evaluator seeds each three-term
/// reduction with the k=1 product, then contracts k=2 and k=0 into it.  This
/// is the same schedule emitted by Eigen's generated helper:
/// `((a1*x1) + a2*x2) + a0*x0`, with both additions fused.  The seed/FMA
/// order is observable on the live ordinal-0 and ordinal-1 links.
fn eigen_matrix_vector_f32(matrix: &SMatrix<f32, 3, 3>, vector: Vector3<f32>) -> Vector3<f32> {
    let x0 = vector.x;
    let x1 = vector.y;
    let x2 = vector.z;
    let row0 = matrix[(0, 2)].fused_mul_add(x2, matrix[(0, 1)] * x1);
    let row0 = matrix[(0, 0)].fused_mul_add(x0, row0);
    let row1 = matrix[(1, 2)].fused_mul_add(x2, matrix[(1, 1)] * x1);
    let row1 = matrix[(1, 0)].fused_mul_add(x0, row1);
    let row2 = matrix[(2, 2)].fused_mul_add(x2, matrix[(2, 1)] * x1);
    let row2 = matrix[(2, 0)].fused_mul_add(x0, row2);
    Vector3::new(row0, row1, row2)
}

/// Eigen's lazy RHS path for the velocity residual's fixed 3x3 product.
///
/// The source expression is `R0_inv * (v1 - v0 - g * dt)`, so Eigen's
/// fixed-size evaluator keeps the RHS expression inside the coefficient
/// product.  On the pinned clean AVX build its packet reduction seeds the
/// final matrix column and folds columns 1 then 0, which differs by one ULP
/// from the materialized helper above for the frame-4 link 0 velocity lane.
fn eigen_matrix_vector_velocity_f32(
    matrix: &SMatrix<f32, 3, 3>,
    vector: Vector3<f32>,
) -> Vector3<f32> {
    let x0 = vector.x;
    let x1 = vector.y;
    let x2 = vector.z;
    let row0 = matrix[(0, 2)] * x2;
    let row0 = matrix[(0, 1)].fused_mul_add(x1, row0);
    let row0 = matrix[(0, 0)].fused_mul_add(x0, row0);
    let row1 = matrix[(1, 2)] * x2;
    let row1 = matrix[(1, 1)].fused_mul_add(x1, row1);
    let row1 = matrix[(1, 0)].fused_mul_add(x0, row1);
    let row2 = matrix[(2, 2)] * x2;
    let row2 = matrix[(2, 1)].fused_mul_add(x1, row2);
    let row2 = matrix[(2, 0)].fused_mul_add(x0, row2);
    Vector3::new(row0, row1, row2)
}

/// Eigen's fixed 9x3 column-major matrix-vector product at the active f32
/// boundary.  The generated fixed-size evaluator seeds each row with k=0,
/// then fuses k=1 and k=2.  Keeping this schedule
/// explicit is required for the IMU bias Jacobian products: a nalgebra
/// `SMatrix * Vector3` chooses a different reduction tree and changes the
/// residual by an ulp on the second IMU link.
fn eigen_matrix_vector_9x3_f32(
    matrix: &SMatrix<f32, 9, 3>,
    vector: Vector3<f32>,
) -> SMatrix<f32, 9, 1> {
    SMatrix::<f32, 9, 1>::from_fn(|row, _| {
        let value = matrix[(row, 0)] * vector.x;
        let value = matrix[(row, 1)].fused_mul_add(vector.y, value);
        matrix[(row, 2)].fused_mul_add(vector.z, value)
    })
}

/// Eigen's fixed 3x3 packet product seeds each dot product with the k=1
/// product, then contracts k=2 and k=0.  This is the same schedule used by
/// the audited estimator f32 matrix product.
fn eigen_matrix_product_f32(
    left: SMatrix<f32, 3, 3>,
    right: SMatrix<f32, 3, 3>,
) -> SMatrix<f32, 3, 3> {
    let mut result = SMatrix::<f32, 3, 3>::zeros();
    for column in 0..3 {
        for row in 0..3 {
            let mut value = left[(row, 1)] * right[(1, column)];
            value = left[(row, 2)].fused_mul_add(right[(2, column)], value);
            value = left[(row, 0)].fused_mul_add(right[(0, column)], value);
            result[(row, column)] = value;
        }
    }
    result
}

/// Eigen's fixed-size `Vector3f::squaredNorm()` reduction as emitted by the
/// pinned AVX build.  The evaluator seeds with lane 1, folds lane 2 with an
/// FMA, then folds lane 0 with an FMA.  This order is observable near the
/// Sophus small-angle cutoff, where the subsequent coefficient formula
/// subtracts two nearly equal terms.
#[inline]
fn eigen_squared_norm3_f32(value: Vector3<f32>) -> f32 {
    let mut result = value.y * value.y;
    result = value.z.fused_mul_add(value.z, result);
    value.x.fused_mul_add(value.x, result)
}

/// Eigen's AVX column-major 9x9-by-9x1 `GemvProduct` at the active float
/// boundary.  The packet rows accumulate one FMA for each covariance column;
/// keeping the same order also makes the scalar tail explicit and avoids
/// nalgebra's fixed-size product evaluator selecting another reduction tree.
fn eigen_whiten_residual_f32(
    whitener: &Matrix9F32,
    residual: &SMatrix<f32, 9, 1>,
) -> SMatrix<f32, 9, 1> {
    let mut result = SMatrix::<f32, 9, 1>::zeros();
    for row in 0..9 {
        let mut value = 0.0_f32;
        for column in 0..9 {
            value = whitener[(row, column)].fused_mul_add(residual[column], value);
        }
        result[row] = value;
    }
    result
}

/// Eigen's column-major 9x9-by-9x30 `GemmProduct` used for the whitened
/// state Jacobian.  Rows 0..7 are independent packet lanes and therefore
/// reduce each output with the k=0..8 FMA sequence.  The row-8 AVX tail uses
/// the four two-term packet accumulators emitted by the fixed kernel before
/// adding its final depth term; that tree is observable in the last row.
fn eigen_whiten_jacobian_f32(whitener: &Matrix9F32, jacobian: &Matrix9x30F32) -> Matrix9x30F32 {
    let mut result = Matrix9x30F32::zeros();
    for row in 0..8 {
        for column in 0..28 {
            // Eigen's ordinary 1x4 AVX micro-kernel maintains even and odd
            // depth accumulators independently, then performs a packet add
            // before fusing the final depth-8 term.
            let mut even = 0.0_f32;
            let mut odd = 0.0_f32;
            for depth in (0..8).step_by(2) {
                even = whitener[(row, depth)].fused_mul_add(jacobian[(depth, column)], even);
            }
            for depth in (1..8).step_by(2) {
                odd = whitener[(row, depth)].fused_mul_add(jacobian[(depth, column)], odd);
            }
            result[(row, column)] =
                whitener[(row, 8)].fused_mul_add(jacobian[(8, column)], even + odd);
        }
        for column in 28..30 {
            let mut value = 0.0_f32;
            for depth in 0..9 {
                value = whitener[(row, depth)].fused_mul_add(jacobian[(depth, column)], value);
            }
            result[(row, column)] = value;
        }
    }

    // The scalar row uses Eigen's swapped packet tail.  Four columns are
    // interleaved in each two-depth block, then merged across the four blocks
    // before depth 8 is fused.  This is intentionally expressed over all
    // groups rather than as a per-column pair tree.
    for group in 0..7 {
        let base = group * 4;
        let mut blocks = [[0.0_f32; 8]; 4];
        for pair in 0..4 {
            for column in 0..4 {
                let col = base + column;
                let even_depth = 2 * pair;
                let odd_depth = even_depth + 1;
                blocks[pair][column] =
                    whitener[(8, even_depth)].fused_mul_add(jacobian[(even_depth, col)], 0.0_f32);
                blocks[pair][4 + column] =
                    whitener[(8, odd_depth)].fused_mul_add(jacobian[(odd_depth, col)], 0.0_f32);
            }
        }
        for lane in 0..8 {
            let first = blocks[0][lane] + blocks[1][lane];
            let second = blocks[2][lane] + blocks[3][lane];
            blocks[0][lane] = first + second;
        }
        for column in 0..4 {
            let col = base + column;
            let reduced = blocks[0][column] + blocks[0][4 + column];
            result[(8, col)] = whitener[(8, 8)].fused_mul_add(jacobian[(8, col)], reduced);
        }
    }
    for column in 28..30 {
        let mut value = 0.0_f32;
        for depth in 0..9 {
            value = whitener[(8, depth)].fused_mul_add(jacobian[(depth, column)], value);
        }
        result[(8, column)] = value;
    }
    result
}

/// Eigen's fixed-size products used by Basalt's production `ImuBlock::Jp`
/// assignments.  The 9x9 blocks use the AVX GEBP Packet8 row path: columns
/// in complete four-column panels keep independent even/odd depth accumulators
/// and the scalar row uses the swapped packet tail.  The 9x3 bias blocks have
/// no complete four-column panel and consequently take the ordinary FMA
/// remainder path.  This helper intentionally operates on each RHS block
/// separately; applying the same schedule to a synthetic 9x30 RHS is not the
/// source expression evaluated by Basalt.
fn eigen_whiten_block_f32<const C: usize>(
    whitener: &Matrix9F32,
    rhs: &SMatrix<f32, 9, C>,
) -> SMatrix<f32, 9, C> {
    let mut result = SMatrix::<f32, 9, C>::zeros();
    let packet_cols4 = (C / 4) * 4;
    for row in 0..9 {
        for column in 0..C {
            if row < 8 && column < packet_cols4 {
                let mut even = 0.0_f32;
                let mut odd = 0.0_f32;
                for depth in (0..8).step_by(2) {
                    even = whitener[(row, depth)].fused_mul_add(rhs[(depth, column)], even);
                }
                for depth in (1..8).step_by(2) {
                    odd = whitener[(row, depth)].fused_mul_add(rhs[(depth, column)], odd);
                }
                result[(row, column)] =
                    whitener[(row, 8)].fused_mul_add(rhs[(8, column)], even + odd);
            } else if row == 8 && column < packet_cols4 {
                // Eigen's swapped tail loads two depth values into each
                // half-packet.  After four two-depth blocks it merges the
                // even and odd lanes separately, reduces the halves, and
                // finally fuses the depth-8 remainder.
                let terms: [f32; 8] = std::array::from_fn(|depth| {
                    whitener[(row, depth)].fused_mul_add(rhs[(depth, column)], 0.0_f32)
                });
                let even = (terms[0] + terms[2]) + (terms[4] + terms[6]);
                let odd = (terms[1] + terms[3]) + (terms[5] + terms[7]);
                result[(row, column)] =
                    whitener[(row, 8)].fused_mul_add(rhs[(8, column)], even + odd);
            } else {
                let mut value = 0.0_f32;
                for depth in 0..9 {
                    value = whitener[(row, depth)].fused_mul_add(rhs[(depth, column)], value);
                }
                result[(row, column)] = value;
            }
        }
    }
    result
}

/// Native computeImuError evaluates the explicit inverse covariance rather
/// than the squared norm of a whitened residual. Keep linearization unchanged.
pub(crate) fn trial_imu_quadratic_stages_f32(
    sqrt_information: &Matrix9F32,
    raw: &SMatrix<f32, 9, 1>,
) -> (Matrix9F32, SMatrix<f32, 9, 1>, f32) {
    let covariance_inverse =
        eigen_whiten_block_f32(&sqrt_information.transpose(), sqrt_information);
    let dot = |left: &SMatrix<f32, 9, 1>, right: &SMatrix<f32, 9, 1>| {
        let products: [f32; 8] = std::array::from_fn(|i| left[i] * right[i]);
        let half: [f32; 4] = std::array::from_fn(|i| products[i] + products[i + 4]);
        let sum = (half[0] + half[2]) + (half[1] + half[3]);
        left[8].fused_mul_add(right[8], sum)
    };
    let mut weighted = SMatrix::<f32, 9, 1>::zeros();
    for column in 0..9 {
        weighted[column] = 0.5_f32 * dot(&covariance_inverse.column(column).into_owned(), raw);
    }
    let cost = dot(&weighted, raw);
    (covariance_inverse, weighted, cost)
}

/// computeImuError's two independently accumulated bias quadratic forms.
/// `noise` carries the estimator's inverse calibration standard deviations.
/// Unlike whitening, native squares the sqrt weight before dividing by dt.
pub(crate) fn trial_bias_cost_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    dt_ns: i64,
    noise: BiasRandomWalkNoise,
) -> Result<(f32, f32), ImuFactorError> {
    if dt_ns <= 0 || !noise.is_valid() {
        return Err(ImuFactorError::InvalidDelta);
    }
    let dt = dt_ns as f32 * 1.0e-9_f32;
    let cost = |a: Vector3<f64>, b: Vector3<f64>, inverse_std: f64| {
        let sqrt_weight = 1.0_f32 / (1.0_f32 / inverse_std as f32);
        let weight = (sqrt_weight * sqrt_weight) / dt;
        let residual = a.map(|v| v as f32) - b.map(|v| v as f32);
        let coefficient = residual.map(|v| (0.5_f32 * v) * weight);
        coefficient.z.fused_mul_add(
            residual.z,
            coefficient
                .y
                .fused_mul_add(residual.y, coefficient.x * residual.x),
        )
    };
    Ok((
        cost(from.gyro_bias_rad_s, to.gyro_bias_rad_s, noise.gyro_density),
        cost(
            from.accel_bias_m_s2,
            to.accel_bias_m_s2,
            noise.accel_density,
        ),
    ))
}

fn eigen_whiten_jacobian_blockwise_f32(
    whitener: &Matrix9F32,
    jacobian: &Matrix9x30F32,
) -> Matrix9x30F32 {
    let start =
        eigen_whiten_block_f32::<9>(whitener, &jacobian.fixed_view::<9, 9>(0, 0).into_owned());
    let gyro_bias =
        eigen_whiten_block_f32::<3>(whitener, &jacobian.fixed_view::<9, 3>(0, 9).into_owned());
    let accel_bias =
        eigen_whiten_block_f32::<3>(whitener, &jacobian.fixed_view::<9, 3>(0, 12).into_owned());
    let end =
        eigen_whiten_block_f32::<9>(whitener, &jacobian.fixed_view::<9, 9>(0, 15).into_owned());

    let mut result = Matrix9x30F32::zeros();
    result.fixed_view_mut::<9, 9>(0, 0).copy_from(&start);
    result.fixed_view_mut::<9, 3>(0, 9).copy_from(&gyro_bias);
    result.fixed_view_mut::<9, 3>(0, 12).copy_from(&accel_bias);
    result.fixed_view_mut::<9, 9>(0, 15).copy_from(&end);
    result
}

/// Eigen's fixed normal-equation products for one whitened IMU row.  The
/// normal matrix computes every coefficient with the complete k=0..8 FMA
/// sequence.  The gradient follows Eigen's row-major Packet8 reduction and
/// keeps the final scalar depth term as an ordinary multiply/add.
pub fn imu_row_products_f32(
    jacobian: &Matrix9x30F32,
    residual: &SMatrix<f32, 9, 1>,
) -> (Matrix30F32, Vector30F32) {
    let mut hessian = Matrix30F32::zeros();
    for row in 0..30 {
        for column in 0..30 {
            let mut value = 0.0_f32;
            for depth in 0..9 {
                value = jacobian[(depth, row)].fused_mul_add(jacobian[(depth, column)], value);
            }
            hessian[(row, column)] = value;
        }
    }

    let mut gradient = Vector30F32::zeros();
    for column in 0..30 {
        let lane0 = jacobian[(0, column)].fused_mul_add(residual[0], 0.0_f32);
        let lane1 = jacobian[(1, column)].fused_mul_add(residual[1], 0.0_f32);
        let lane2 = jacobian[(2, column)].fused_mul_add(residual[2], 0.0_f32);
        let lane3 = jacobian[(3, column)].fused_mul_add(residual[3], 0.0_f32);
        let lane4 = jacobian[(4, column)].fused_mul_add(residual[4], 0.0_f32);
        let lane5 = jacobian[(5, column)].fused_mul_add(residual[5], 0.0_f32);
        let lane6 = jacobian[(6, column)].fused_mul_add(residual[6], 0.0_f32);
        let lane7 = jacobian[(7, column)].fused_mul_add(residual[7], 0.0_f32);
        let q0 = lane0 + lane4;
        let q1 = lane1 + lane5;
        let q2 = lane2 + lane6;
        let q3 = lane3 + lane7;
        let packet_sum = (q0 + q2) + (q1 + q3);
        gradient[column] = packet_sum + jacobian[(8, column)] * residual[8];
    }
    (hessian, gradient)
}

fn quat_raw_f32(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    UnitQuaternion::from_quaternion(Quaternion::new(
        rotation.w as f32,
        rotation.i as f32,
        rotation.j as f32,
        rotation.k as f32,
    ))
}

fn residual_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> SMatrix<f32, 9, 1> {
    residual_f32_with_trial_velocity::<false, true>(from, to, delta, gravity_world)
}

pub(crate) fn trial_preintegration_residual_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> SMatrix<f32, 9, 1> {
    residual_f32_with_trial_velocity::<true, false>(from, to, delta, gravity_world)
}

fn residual_f32_with_trial_velocity<const TRIAL: bool, const CURRENT_MATRIX: bool>(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> SMatrix<f32, 9, 1> {
    let bg = from.gyro_bias_rad_s.map(|value| value as f32);
    let ba = from.accel_bias_m_s2.map(|value| value as f32);
    let d_bg = bg - delta.bias_gyro.map(|value| value as f32);
    let d_ba = ba - delta.bias_accel.map(|value| value as f32);
    let d_state_bg_full = SMatrix::<f32, 9, 3>::from_fn(|row, col| match row {
        0..=2 => delta.jacobian_position_gyro_bias[(row, col)] as f32,
        3..=5 => delta.jacobian_rotation_gyro_bias[(row - 3, col)] as f32,
        _ => delta.jacobian_velocity_gyro_bias[(row - 6, col)] as f32,
    });
    let d_state_ba_full = SMatrix::<f32, 9, 3>::from_fn(|row, col| match row {
        0..=2 => delta.jacobian_position_accel_bias[(row, col)] as f32,
        3..=5 => 0.0,
        _ => delta.jacobian_velocity_accel_bias[(row - 6, col)] as f32,
    });
    let bg_diff = eigen_matrix_vector_9x3_f32(&d_state_bg_full, d_bg);
    let ba_diff = eigen_matrix_vector_9x3_f32(&d_state_ba_full, d_ba);
    let dr = so3_mul_f32(
        so3_exp_f32(bg_diff.fixed_rows::<3>(3).into_owned()),
        quat_raw_f32(&delta.delta_rotation),
    );
    let dv = delta.delta_velocity.map(|value| value as f32)
        + bg_diff.fixed_rows::<3>(6).into_owned()
        + ba_diff.fixed_rows::<3>(6).into_owned();
    let dp = delta.delta_position.map(|value| value as f32)
        + bg_diff.fixed_rows::<3>(0).into_owned()
        + ba_diff.fixed_rows::<3>(0).into_owned();
    let dt = delta.delta_time as f32;
    let q0 = quat_state_f32(&from.imu_to_world.rotation);
    let q1 = quat_state_f32(&to.imu_to_world.rotation);
    let q0_inv = sophus_so3_inverse(q0);
    let r0_inv = if CURRENT_MATRIX {
        eigen_current_rotation_matrix_f32(&q0_inv)
    } else {
        eigen_rotation_matrix_f32(&q0_inv)
    };
    let gravity = gravity_world.map(|value| value as f32);
    let tmp = eigen_matrix_vector_f32(
        &r0_inv,
        eigen_translation_operand_f32(
            to.imu_to_world.translation.map(|value| value as f32),
            from.imu_to_world.translation.map(|value| value as f32),
            from.velocity_world_m_s.map(|value| value as f32),
            gravity,
            dt,
        ),
    );
    let velocity_operand = eigen_velocity_operand_f32(
        to.velocity_world_m_s.map(|value| value as f32),
        from.velocity_world_m_s.map(|value| value as f32),
        gravity,
        dt,
    );
    // computeImuError<float> ELF470831..4708bc seeds column1,
    // then fuses columns2 and0. Linearization's lazy RHS uses 2,1,0.
    let tmp2 = if TRIAL {
        eigen_matrix_vector_f32(&r0_inv, velocity_operand)
    } else {
        eigen_matrix_vector_velocity_f32(&r0_inv, velocity_operand)
    };
    let rotation = so3_log_f32(so3_mul_f32(so3_mul_f32(dr, sophus_so3_inverse(q1)), q0));
    let position = tmp - dp;
    let velocity = tmp2 - dv;
    let mut result = SMatrix::<f32, 9, 1>::zeros();
    result.fixed_rows_mut::<3>(0).copy_from(&position);
    result.fixed_rows_mut::<3>(3).copy_from(&rotation);
    result.fixed_rows_mut::<3>(6).copy_from(&velocity);
    result
}

fn so3_jacobian_inverse_f32(phi: Vector3<f32>, sign: f32) -> SMatrix<f32, 3, 3> {
    // Sophus::Constants<float>::epsilon() is 1e-5 (not IEEE f32::EPSILON).
    // The IMU rotation residual can fall between those thresholds; matching
    // Sophus therefore keeps this input in its Taylor branch.
    const SOPHUS_EPSILON_F32: f32 = 1.0e-5_f32;
    let theta2 = eigen_squared_norm3_f32(phi);
    let h = hat_f32(phi);
    // Sophus forms Omega*Omega as an Eigen fixed 3x3 product.  Route this
    // intermediate through the pinned product schedule rather than letting
    // nalgebra select its generic reduction tree; the result is observable in
    // the IMU bias-rotation Jacobian at the accepted-state frontier.
    let h2 = eigen_matrix_product_f32(h, h);
    // This is the pinned Sophus right/left inverse-Jacobian expression.  The
    // deployed Basalt header uses the `phi_norm2 > epsilon` cutoff and the
    // full-angle coefficient; preserve that source contract here.  Eigen's
    // packet evaluator emits the identity-plus-half-hat update as a fused
    // multiply-add (`1 + sign * 0.5 * h`) rather than materializing the
    // product and adding it through a matrix `+=`.  Keep that operation local
    // to this helper: other fixed-size products have separately audited
    // reduction schedules.
    let half = 0.5_f32 * sign;
    let mut result = SMatrix::<f32, 3, 3>::from_fn(|row, column| {
        h[(row, column)].fused_mul_add(half, if row == column { 1.0 } else { 0.0 })
    });
    if theta2 > SOPHUS_EPSILON_F32 {
        let theta = theta2.sqrt();
        let coefficient =
            1.0_f32 / theta2 - (1.0_f32 + theta.cos()) / (2.0_f32 * theta * theta.sin());
        result += h2 * coefficient;
    } else {
        // The native Taylor path performs a separate divide followed by an
        // add; do not contract this update with the division.
        for column in 0..3 {
            for row in 0..3 {
                result[(row, column)] += h2[(row, column)] / 12.0_f32;
            }
        }
    }
    result
}

fn residual_jacobian_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Matrix9x30F32 {
    let raw = residual_f32(from, to, delta, gravity_world);
    let dt = delta.delta_time as f32;
    let q0 = quat_state_f32(&from.imu_to_world.rotation);
    let q1 = quat_state_f32(&to.imu_to_world.rotation);
    let q0_inv = sophus_so3_inverse(q0);
    // Primary residual/Jacobian evaluation calls the same out-of-line
    // QuaternionBase::toRotationMatrix as the current-state recomputation.
    let r0_inv = eigen_current_rotation_matrix_f32(&q0_inv);
    let gravity = gravity_world.map(|value| value as f32);
    let tmp = eigen_matrix_vector_f32(
        &r0_inv,
        eigen_translation_operand_f32(
            to.imu_to_world.translation.map(|value| value as f32),
            from.imu_to_world.translation.map(|value| value as f32),
            from.velocity_world_m_s.map(|value| value as f32),
            gravity,
            dt,
        ),
    );
    let tmp2 = eigen_matrix_vector_velocity_f32(
        &r0_inv,
        eigen_velocity_operand_f32(
            to.velocity_world_m_s.map(|value| value as f32),
            from.velocity_world_m_s.map(|value| value as f32),
            gravity,
            dt,
        ),
    );
    let rotation = raw.fixed_rows::<3>(3).into_owned();
    let right_inv = so3_jacobian_inverse_f32(rotation, 1.0);
    let left_inv = so3_jacobian_inverse_f32(rotation, -1.0);
    let mut jacobian = Matrix9x30F32::zeros();
    let delta_pbg = delta.jacobian_position_gyro_bias.map(|value| value as f32);
    let delta_pba = delta.jacobian_position_accel_bias.map(|value| value as f32);
    let delta_rbg = delta.jacobian_rotation_gyro_bias.map(|value| value as f32);
    let delta_vbg = delta.jacobian_velocity_gyro_bias.map(|value| value as f32);
    let delta_vba = delta.jacobian_velocity_accel_bias.map(|value| value as f32);
    let skew_tmp = hat_f32(tmp);
    let skew_tmp2 = hat_f32(tmp2);
    set_block_f32(&mut jacobian, 0, 0, -r0_inv);
    set_block_f32(
        &mut jacobian,
        0,
        3,
        eigen_matrix_product_f32(skew_tmp, r0_inv),
    );
    set_block_f32(
        &mut jacobian,
        3,
        3,
        eigen_matrix_product_f32(right_inv, r0_inv),
    );
    set_block_f32(
        &mut jacobian,
        6,
        3,
        eigen_matrix_product_f32(skew_tmp2, r0_inv),
    );
    set_block_f32(&mut jacobian, 0, 6, -r0_inv * dt);
    set_block_f32(&mut jacobian, 6, 6, -r0_inv);
    set_block_f32(&mut jacobian, 0, 9, -delta_pbg);
    // Basalt writes the complete accelerometer-bias derivative as
    // `-d_state_d_ba_`, including the structurally zero rotation rows. Keep
    // those negative signed zeros observable at the f32 boundary instead of
    // losing them in the zero-initialized destination matrix.
    for row in 0..9 {
        for col in 0..3 {
            let value = match row {
                0..=2 => delta_pba[(row, col)],
                3..=5 => 0.0_f32,
                _ => delta_vba[(row - 6, col)],
            };
            jacobian[(row, 12 + col)] = -value;
        }
    }
    set_block_f32(
        &mut jacobian,
        3,
        9,
        eigen_matrix_product_f32(left_inv, delta_rbg),
    );
    set_block_f32(&mut jacobian, 6, 9, -delta_vbg);
    set_block_f32(&mut jacobian, 6, 12, -delta_vba);
    set_block_f32(&mut jacobian, 0, NAV_STATE_DOF, r0_inv);
    set_block_f32(
        &mut jacobian,
        3,
        NAV_STATE_DOF + 3,
        -eigen_matrix_product_f32(right_inv, r0_inv),
    );
    set_block_f32(&mut jacobian, 6, NAV_STATE_DOF + 6, r0_inv);
    let _ = q1;
    jacobian
}

/// Bias random walk with continuous-time density and `sqrt(dt)` scaling.
pub fn whitened_bias_random_walk_factor(
    from: &BasaltNavState,
    to: &BasaltNavState,
    dt: f64,
    noise: BiasRandomWalkNoise,
) -> Result<WhitenedBiasWalkFactor, ImuFactorError> {
    if !dt.is_finite() || dt <= 0.0 || !noise.is_valid() {
        return Err(ImuFactorError::InvalidDelta);
    }
    // Basalt stores the square-root random-walk weights and divides them by
    // sqrt(dt) for the discrete bias-difference row.
    let gyro_weight = noise.gyro_density / dt.sqrt();
    let accel_weight = noise.accel_density / dt.sqrt();
    let mut jacobian = DMatrix::zeros(6, 2 * NAV_STATE_DOF);
    let mut residual = DVector::zeros(6);
    for axis in 0..3 {
        jacobian[(axis, 9 + axis)] = gyro_weight;
        jacobian[(axis, NAV_STATE_DOF + 9 + axis)] = -gyro_weight;
        residual[axis] = (from.gyro_bias_rad_s[axis] - to.gyro_bias_rad_s[axis]) * gyro_weight;
        jacobian[(axis + 3, 12 + axis)] = accel_weight;
        jacobian[(axis + 3, NAV_STATE_DOF + 12 + axis)] = -accel_weight;
        residual[axis + 3] = (from.accel_bias_m_s2[axis] - to.accel_bias_m_s2[axis]) * accel_weight;
    }
    Ok(WhitenedBiasWalkFactor {
        state_jacobian: jacobian,
        residual,
    })
}

/// Build the bias random-walk rows at Basalt's active `Scalar=float`
/// boundary.
///
/// `ImuBlock<float>::linearizeImu` first narrows the timestamp-derived `dt`
/// to `float`, evaluates `std::sqrt(dt)`, and divides the already-float
/// calibration weights by that value.  The ordinary public helper above is
/// intentionally kept in f64 for callers that request the extended path;
/// applying a final f32 cast to its result is not equivalent at the two
/// alternating EuRoC IMU intervals.
pub fn whitened_bias_random_walk_factor_upstream_f32(
    from: &BasaltNavState,
    to: &BasaltNavState,
    dt: f64,
    noise: BiasRandomWalkNoise,
) -> Result<WhitenedBiasWalkFactor, ImuFactorError> {
    if !dt.is_finite() || dt <= 0.0 || !noise.is_valid() {
        return Err(ImuFactorError::InvalidDelta);
    }

    let dt_f32 = dt as f32;
    let sqrt_dt = dt_f32.sqrt();
    if !sqrt_dt.is_finite() || sqrt_dt <= 0.0 {
        return Err(ImuFactorError::InvalidDelta);
    }

    // The estimator owns the calibration weight as a public f64 scalar, but
    // the pinned native constructor stores `calib.*_bias_std` as Vector3f
    // and computes `.array().inverse()`.  Replaying that inverse from the
    // scalar weight preserves the Eigen float reciprocal (notably
    // `1 / 0.001f == 999.99994f`, while a wide f64 reciprocal cast gives
    // exactly `1000f`).  This only affects the upstream-f32 seam; the
    // extended helper remains a literal f64 implementation.
    let eigen_inverse = |weight: f64| {
        let weight_f32 = weight as f32;
        1.0_f32 / (1.0_f32 / weight_f32)
    };
    let gyro_weight = eigen_inverse(noise.gyro_density) / sqrt_dt;
    let accel_weight = eigen_inverse(noise.accel_density) / sqrt_dt;
    let mut jacobian = DMatrix::zeros(6, 2 * NAV_STATE_DOF);
    let mut residual = DVector::zeros(6);
    for axis in 0..3 {
        jacobian[(axis, 9 + axis)] = gyro_weight as f64;
        jacobian[(axis, NAV_STATE_DOF + 9 + axis)] = -(gyro_weight as f64);
        let bg_diff = (from.gyro_bias_rad_s[axis] as f32) - (to.gyro_bias_rad_s[axis] as f32);
        residual[axis] = (bg_diff * gyro_weight) as f64;
        jacobian[(axis + 3, 12 + axis)] = accel_weight as f64;
        jacobian[(axis + 3, NAV_STATE_DOF + 12 + axis)] = -(accel_weight as f64);
        let ba_diff = (from.accel_bias_m_s2[axis] as f32) - (to.accel_bias_m_s2[axis] as f32);
        residual[axis + 3] = (ba_diff * accel_weight) as f64;
    }
    Ok(WhitenedBiasWalkFactor {
        state_jacobian: jacobian,
        residual,
    })
}

fn residual(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> DVector<f64> {
    let (dr, dv, dp) = delta.corrected(from.gyro_bias_rad_s, from.accel_bias_m_s2);
    let dt = delta.delta_time;
    let r_iw = from.imu_to_world.rotation.inverse();
    // IntegratedImuMeasurement::residual uses the left-perturbation state
    // convention: exp(dtheta_bg) * delta_R * R1^-1 * R0.  `corrected`
    // already applies exp(dtheta_bg) on the left, so keep this order rather
    // than forming the equivalent-looking (but differently tangent-framed)
    // R0^-1 * R1 * delta_R^-1 expression.
    let rotation =
        (dr * to.imu_to_world.rotation.inverse() * from.imu_to_world.rotation).scaled_axis();
    let velocity =
        r_iw * (to.velocity_world_m_s - from.velocity_world_m_s - gravity_world * dt) - dv;
    let position = r_iw
        * (to.imu_to_world.translation
            - from.imu_to_world.translation
            - from.velocity_world_m_s * dt
            - gravity_world * (0.5 * dt * dt))
        - dp;
    DVector::from_iterator(
        9,
        position
            .iter()
            .chain(rotation.iter())
            .chain(velocity.iter())
            .copied(),
    )
}

/// Unwhitened `[position, rotation, velocity]` residual for diagnostics and
/// oracle parity exporters.
pub fn preintegration_residual(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> DVector<f64> {
    residual(from, to, delta, gravity_world)
}

#[cfg(test)]
fn perturb_state(state: &mut BasaltNavState, local: usize, amount: f64) {
    match local {
        0..=2 => state.imu_to_world.translation[local] += amount,
        3..=5 => {
            // Basalt's PoseState::incPose adds translation directly and
            // left-multiplies only the SO(3) part.  A full SE(3) exponential
            // composed on the left would incorrectly rotate the world-frame
            // position around the origin.
            let mut dtheta = Vector3::zeros();
            dtheta[local - 3] = amount;
            state.imu_to_world.rotation =
                nalgebra::UnitQuaternion::from_scaled_axis(dtheta) * state.imu_to_world.rotation;
        }
        6..=8 => state.velocity_world_m_s[local - 6] += amount,
        9..=11 => state.gyro_bias_rad_s[local - 9] += amount,
        12..=14 => state.accel_bias_m_s2[local - 12] += amount,
        _ => unreachable!(),
    }
}

/// Basalt's square-root information uses Eigen's `LDLT` factorization.  Keep
/// the same left whitener (`D^-1/2 L^-1 P`) instead of replacing it with a
/// Cholesky/SVD branch: covariance entries are part of the f32 oracle trace.
pub fn sqrt_information(covariance: &Matrix9) -> Result<Matrix9, ImuFactorError> {
    if !covariance.iter().all(|x| x.is_finite()) {
        return Err(ImuFactorError::InvalidCovariance);
    }
    Ok(ldlt_whitener_f64(covariance))
}

/// f32 counterpart of [`sqrt_information`], used by the upstream scalar path.
pub fn sqrt_information_f32(covariance: &Matrix9F32) -> Result<Matrix9F32, ImuFactorError> {
    if !covariance.iter().all(|x| x.is_finite()) {
        return Err(ImuFactorError::InvalidCovariance);
    }
    Ok(ldlt_whitener_f32(covariance))
}

fn ldlt_whitener_f64(covariance: &Matrix9) -> Matrix9 {
    let covariance = (covariance + covariance.transpose()) * 0.5;
    let mut lower = Matrix9::identity();
    let mut diagonal = [0.0_f64; 9];
    for i in 0..9 {
        let mut value = covariance[(i, i)];
        for k in 0..i {
            value -= lower[(i, k)] * lower[(i, k)] * diagonal[k];
        }
        diagonal[i] = value;
        if value.abs() > f64::MIN_POSITIVE {
            for j in (i + 1)..9 {
                let mut numerator = covariance[(j, i)];
                for k in 0..i {
                    numerator -= lower[(j, k)] * diagonal[k] * lower[(i, k)];
                }
                lower[(j, i)] = numerator / value;
            }
        }
    }
    let mut inverse_lower = Matrix9::zeros();
    for column in 0..9 {
        for row in 0..9 {
            let mut value = if row == column { 1.0 } else { 0.0 };
            for k in 0..row {
                value -= lower[(row, k)] * inverse_lower[(k, column)];
            }
            inverse_lower[(row, column)] = value;
        }
    }
    let mut result = inverse_lower;
    for row in 0..9 {
        // Eigen's source explicitly zeros entries whose D pivot is below
        // numeric_limits<Scalar>::min(), including negative/rank-deficient
        // pivots used by legacy zero-covariance fixtures.
        let scale = if diagonal[row] < f64::MIN_POSITIVE {
            0.0
        } else {
            1.0 / diagonal[row].sqrt()
        };
        for column in 0..9 {
            result[(row, column)] *= scale;
        }
    }
    result
}

fn ldlt_whitener_f32(covariance: &Matrix9F32) -> Matrix9F32 {
    // Eigen's LDLT is pivoted.  It factors the lower triangle as
    // A = P^T L D L^T P, then forms D^-1/2 L^-1 P.  A plain unpivoted LDLT
    // happens to work for well-conditioned toy matrices but is materially
    // different for the rank-deficient IMU covariance (the first six
    // position/rotation pivots are zero after pivoting).
    let covariance = *covariance;
    let mut matrix = covariance;
    let mut transpositions = [0usize; 9];
    let mut diagonal = [0.0_f32; 9];
    let mut temporary = [0.0_f32; 9];
    for k in 0..9 {
        let mut pivot = k;
        let mut biggest = matrix[(k, k)].abs();
        for index in (k + 1)..9 {
            let candidate = matrix[(index, index)].abs();
            if candidate > biggest {
                biggest = candidate;
                pivot = index;
            }
        }
        transpositions[k] = pivot;
        if k != pivot {
            let s = 9 - pivot - 1;
            for col in 0..k {
                let tmp = matrix[(k, col)];
                matrix[(k, col)] = matrix[(pivot, col)];
                matrix[(pivot, col)] = tmp;
            }
            for offset in 0..s {
                let row = pivot + 1 + offset;
                let tmp = matrix[(row, k)];
                matrix[(row, k)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = tmp;
            }
            let tmp = matrix[(k, k)];
            matrix[(k, k)] = matrix[(pivot, pivot)];
            matrix[(pivot, pivot)] = tmp;
            for index in (k + 1)..pivot {
                let tmp = matrix[(index, k)];
                matrix[(index, k)] = matrix[(pivot, index)];
                matrix[(pivot, index)] = tmp;
            }
        }

        let rs = 9 - k - 1;
        if k > 0 {
            for index in 0..k {
                temporary[index] = diagonal[index] * matrix[(k, index)];
            }
            let correction = eigen_ldlt_inner_product_f32(
                (0..k).map(|index| matrix[(k, index)]),
                (0..k).map(|index| temporary[index]),
            );
            matrix[(k, k)] -= correction;
            eigen_ldlt_gemv_sub_f32(&mut matrix, k, rs, &temporary[..k]);
        }
        let pivot_value = matrix[(k, k)];
        diagonal[k] = pivot_value;
        if rs > 0 && pivot_value.abs() > 0.0_f32 {
            for row_offset in 0..rs {
                let row = k + 1 + row_offset;
                matrix[(row, k)] /= pivot_value;
            }
        }
    }

    // P is the transposition sequence returned by Eigen.  Applying the
    // swaps to identity in decomposition order gives transpositionsP().
    let mut result = Matrix9F32::identity();
    for k in 0..9 {
        let pivot = transpositions[k];
        if k != pivot {
            for col in 0..9 {
                let tmp = result[(k, col)];
                result[(k, col)] = result[(pivot, col)];
                result[(pivot, col)] = tmp;
            }
        }
    }
    // L^-1 P.  Eigen's fixed 9x9 RHS takes an eight-row triangular panel,
    // then updates the ninth row through its packet GEMM tail.  Keeping the
    // two stages explicit preserves the AVX2 FMA/reduction order.
    let p_identity = result;
    eigen_ldlt_forward_solve_f32(&mut result, &matrix);
    let pre_scale = result;
    let mut scales = [0.0_f32; 9];
    for row in 0..9 {
        let scale = if diagonal[row] < f32::MIN_POSITIVE {
            0.0_f32
        } else {
            // Pinned Eigen Scalar=float uses the float sqrt overload and
            // binary32 reciprocal at this boundary.
            1.0_f32 / diagonal[row].sqrt()
        };
        scales[row] = scale;
        for column in 0..9 {
            result[(row, column)] *= scale;
        }
    }
    if std::env::var_os("VISLOC_BASALT_M7IM_LDLT").is_some() {
        eprintln!(
            "RUST_M7IM_LDLT covariance={} pivots={} ldlt={} lower={} d={} p_identity={} pre_scale={} scales={} w={}",
            m7im_matrix_bits(&covariance),
            transpositions
                .iter()
                .map(|pivot| pivot.to_string())
                .collect::<Vec<_>>()
                .join(","),
            m7im_matrix_bits(&matrix),
            m7im_lower_bits(&matrix),
            diagonal
                .iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
                .join(","),
            m7im_matrix_bits(&p_identity),
            m7im_matrix_bits(&pre_scale),
            scales.iter().map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>().join(","),
            m7im_matrix_bits(&result),
        );
    }
    result
}

/// Solve a binary32 symmetric system with Eigen's dynamic diagonal-pivoted
/// LDLT schedule.
///
/// This is the dynamic-size counterpart of the fixed 9x9 covariance helper
/// above.  The factorization keeps Eigen's lower-triangle transposition order
/// and the AVX2-sized (eight-lane) rank updates.  The two triangular solves
/// are written in Eigen's panel order as well; this matters for the f32
/// rounding schedule because a panel's trailing update is one GEMV reduction,
/// rather than a sequence of scalar row updates.
pub(crate) fn eigen_ldlt_solve_f32(
    input: &DMatrix<f32>,
    rhs: &DVector<f32>,
) -> Option<DVector<f32>> {
    let size = input.nrows();
    if size == 0
        || input.ncols() != size
        || rhs.len() != size
        || !input.iter().all(|value| value.is_finite())
        || !rhs.iter().all(|value| value.is_finite())
    {
        return None;
    }

    let mut matrix = input.clone();
    let mut transpositions = vec![0usize; size];
    let mut diagonal = vec![0.0_f32; size];
    let mut temporary = vec![0.0_f32; size];

    for k in 0..size {
        let mut pivot = k;
        let mut biggest = matrix[(k, k)].abs();
        for index in (k + 1)..size {
            let candidate = matrix[(index, index)].abs();
            if candidate > biggest {
                biggest = candidate;
                pivot = index;
            }
        }
        transpositions[k] = pivot;
        if pivot != k {
            // This is the lower-triangle-only swap sequence in
            // Eigen::LDLT::unblocked.  In particular, the middle exchange
            // preserves the already factored columns between k and pivot.
            for column in 0..k {
                let value = matrix[(k, column)];
                matrix[(k, column)] = matrix[(pivot, column)];
                matrix[(pivot, column)] = value;
            }
            let tail = size - pivot - 1;
            for offset in 0..tail {
                let row = pivot + 1 + offset;
                let value = matrix[(row, k)];
                matrix[(row, k)] = matrix[(row, pivot)];
                matrix[(row, pivot)] = value;
            }
            let value = matrix[(k, k)];
            matrix[(k, k)] = matrix[(pivot, pivot)];
            matrix[(pivot, pivot)] = value;
            for row in (k + 1)..pivot {
                let value = matrix[(row, k)];
                matrix[(row, k)] = matrix[(pivot, row)];
                matrix[(pivot, row)] = value;
            }
        }

        let rows = size - k - 1;
        if k > 0 {
            for index in 0..k {
                temporary[index] = diagonal[index] * matrix[(k, index)];
            }
            // DMatrix is column-major, so a row is strided.  Build the short
            // A10 vector explicitly; the product/reduction helper itself is
            // shared with the packet-sized fixed helper below.
            let lhs: Vec<f32> = (0..k).map(|index| matrix[(k, index)]).collect();
            // A10 is a dynamic 1xk Block with a strided inner dimension, so
            // Eigen does not expose packet access here.  Keep its scalar
            // left-to-right pmadd reduction; the A20 GEMV below is the
            // packetized part of the dynamic factorization.
            let correction = eigen_ldlt_inner_product_scalar_f32(&lhs, &temporary[..k]);
            matrix[(k, k)] -= correction;
            if rows > 0 {
                eigen_ldlt_gemv_sub_dynamic_f32(&mut matrix, k + 1, rows, 0, &temporary[..k]);
            }
        }

        let pivot_value = matrix[(k, k)];
        if !pivot_value.is_finite() {
            return None;
        }
        diagonal[k] = pivot_value;
        if rows > 0 && pivot_value != 0.0 {
            for row_offset in 0..rows {
                let row = k + 1 + row_offset;
                matrix[(row, k)] /= pivot_value;
            }
        }
    }

    // dst = P*b.  The transposition sequence is deliberately applied in
    // decomposition order, matching Eigen's transpositionsP().
    let mut result = rhs.clone();
    for (k, &pivot) in transpositions.iter().enumerate() {
        if pivot != k {
            let value = result[k];
            result[k] = result[pivot];
            result[pivot] = value;
        }
    }
    let permutation_rhs = result.clone();

    eigen_ldlt_forward_solve_dynamic_f32(&mut result, &matrix);
    let after_forward = result.clone();
    for index in 0..size {
        if diagonal[index].abs() > f32::MIN_POSITIVE {
            result[index] /= diagonal[index];
        } else {
            result[index] = 0.0;
        }
    }
    let after_d = result.clone();
    eigen_ldlt_backward_solve_dynamic_f32(&mut result, &matrix);
    let after_backward = result.clone();

    // dst = P^T * dst.
    for (k, &pivot) in transpositions.iter().enumerate().rev() {
        if pivot != k {
            let value = result[k];
            result[k] = result[pivot];
            result[pivot] = value;
        }
    }
    emit_m7im15_dynamic_ldlt_stage(
        input,
        rhs,
        &matrix,
        &diagonal,
        &transpositions,
        &permutation_rhs,
        &after_forward,
        &after_d,
        &after_backward,
        &result,
    );
    if result.iter().all(|value| value.is_finite()) {
        Some(result)
    } else {
        None
    }
}

/// Scalar Eigen reduction used by dynamic LDLT's strided row blocks.
fn eigen_ldlt_inner_product_scalar_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert!(!lhs.is_empty());
    let mut result = lhs[0] * rhs[0];
    for index in 1..lhs.len() {
        result = lhs[index].fused_mul_add(rhs[index], result);
    }
    result
}

/// Arithmetic choices for the short, internal `L^T` panel correction.  The
/// default is the production choice; the environment override is deliberately
/// diagnostic-only so the four source-level schedules can be compared against
/// a fresh native 75-lane trace without any lane/index special casing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackwardInternalVariant {
    PlainProductsPlainSum,
    FmaProductsPlainSum,
    PlainProductsFmaSubtract,
    FmaProductsFmaSubtract,
}

impl BackwardInternalVariant {
    fn from_diagnostic_env() -> Self {
        match env::var("VISLOC_BASALT_M7IM15_BACKWARD_INTERNAL_VARIANT")
            .ok()
            .as_deref()
        {
            Some("1") | Some("plain_plain") | Some("plain-products-plain-sum") => {
                Self::PlainProductsPlainSum
            }
            Some("2") | Some("fma_plain") | Some("fma-products-plain-sum") => {
                Self::FmaProductsPlainSum
            }
            Some("3") | Some("plain_fma") | Some("plain-products-fma-subtract") => {
                Self::PlainProductsFmaSubtract
            }
            Some("4") | Some("fma_fma") | Some("fma-products-fma-subtract") => {
                Self::FmaProductsFmaSubtract
            }
            // Production is fixed to variant 1.  An absent or unrecognised
            // diagnostic value must not silently select a new arithmetic path.
            _ => Self::PlainProductsPlainSum,
        }
    }

    #[inline]
    const fn use_fma_correction(self) -> bool {
        matches!(
            self,
            Self::FmaProductsPlainSum | Self::FmaProductsFmaSubtract
        )
    }

    #[inline]
    const fn use_fma_subtract(self) -> bool {
        matches!(
            self,
            Self::PlainProductsFmaSubtract | Self::FmaProductsFmaSubtract
        )
    }
}

fn backward_internal_variant() -> BackwardInternalVariant {
    // Keep the normal estimator path fixed and cheap while allowing isolated
    // trace processes to request one of the four diagnostic schedules.
    static VARIANT: OnceLock<BackwardInternalVariant> = OnceLock::new();
    *VARIANT.get_or_init(BackwardInternalVariant::from_diagnostic_env)
}

#[inline]
fn eigen_ldlt_backward_inner_product_f32(
    lhs: &[f32],
    rhs: &[f32],
    variant: BackwardInternalVariant,
) -> f32 {
    if variant.use_fma_correction() {
        eigen_ldlt_inner_product_scalar_f32(lhs, rhs)
    } else {
        eigen_ldlt_inner_product_plain_f32(lhs, rhs)
    }
}

#[inline]
fn eigen_ldlt_backward_apply_f32(
    destination: f32,
    correction: f32,
    variant: BackwardInternalVariant,
) -> f32 {
    if variant.use_fma_subtract() {
        (-1.0_f32).fused_mul_add(correction, destination)
    } else {
        destination - correction
    }
}

#[inline]
fn eigen_ldlt_inner_product_plain_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert!(!lhs.is_empty());
    let mut result = lhs[0] * rhs[0];
    for index in 1..lhs.len() {
        result += lhs[index] * rhs[index];
    }
    result
}

/// Reproduce Eigen's packetized dynamic reduction for an internal backward
/// panel.  The panel has at most seven terms: GCC emits a scalar first term,
/// a four-lane product packet, and a scalar tail.  For an even length the
/// one-term tail is emitted as a fused multiply-add; odd lengths finish with
/// ordinary f32 additions.
#[inline]
fn eigen_ldlt_backward_panel_sum_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert!(!lhs.is_empty());
    debug_assert!(lhs.len() <= 7);

    let product = |index: usize| lhs[index] * rhs[index];
    let first = product(0);
    match lhs.len() {
        1 => first,
        2 => lhs[1].fused_mul_add(rhs[1], first),
        3 => {
            let first_two = first + product(1);
            first_two + product(2)
        }
        4 => {
            let first_three = (first + product(1)) + product(2);
            lhs[3].fused_mul_add(rhs[3], first_three)
        }
        5 => {
            let first_two = first + product(1);
            let first_three = first_two + product(2);
            let first_four = first_three + product(3);
            first_four + product(4)
        }
        6 => {
            let first_two = first + product(1);
            let first_three = first_two + product(2);
            let first_four = first_three + product(3);
            let first_five = first_four + product(4);
            lhs[5].fused_mul_add(rhs[5], first_five)
        }
        7 => {
            let first_two = first + product(1);
            let first_three = first_two + product(2);
            let first_four = first_three + product(3);
            let first_five = first_four + product(4);
            let first_six = first_five + product(5);
            first_six + product(6)
        }
        _ => eigen_ldlt_inner_product_plain_f32(lhs, rhs),
    }
}

/// Dynamic-size lower-triangle GEMV used by both LDLT factorization and the
/// panel triangular solves.  `alpha=-1` is folded into the final store.
fn eigen_ldlt_gemv_sub_dynamic_f32(
    matrix: &mut DMatrix<f32>,
    row_start: usize,
    row_count: usize,
    col_start: usize,
    x: &[f32],
) {
    let mut row_offset = 0;
    while row_count - row_offset >= 8 {
        let mut correction = [0.0_f32; 8];
        for (offset, value) in x.iter().enumerate() {
            let column = col_start + offset;
            for lane in 0..8 {
                let row = row_start + row_offset + lane;
                correction[lane] = matrix[(row, column)].fused_mul_add(*value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            matrix[(row, col_start + x.len())] =
                (-1.0_f32).fused_mul_add(value, matrix[(row, col_start + x.len())]);
        }
        row_offset += 8;
    }
    while row_count - row_offset >= 4 {
        let mut correction = [0.0_f32; 4];
        for (offset, value) in x.iter().enumerate() {
            let column = col_start + offset;
            for lane in 0..4 {
                let row = row_start + row_offset + lane;
                correction[lane] = matrix[(row, column)].fused_mul_add(*value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            matrix[(row, col_start + x.len())] =
                (-1.0_f32).fused_mul_add(value, matrix[(row, col_start + x.len())]);
        }
        row_offset += 4;
    }
    while row_count - row_offset >= 2 {
        let mut correction = [0.0_f32; 2];
        for (offset, value) in x.iter().enumerate() {
            let column = col_start + offset;
            for lane in 0..2 {
                let row = row_start + row_offset + lane;
                correction[lane] = matrix[(row, column)].fused_mul_add(*value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            matrix[(row, col_start + x.len())] =
                (-1.0_f32).fused_mul_add(value, matrix[(row, col_start + x.len())]);
        }
        row_offset += 2;
    }
    while row_offset < row_count {
        let row = row_start + row_offset;
        let mut correction = 0.0_f32;
        for (offset, value) in x.iter().enumerate() {
            correction = matrix[(row, col_start + offset)].fused_mul_add(*value, correction);
        }
        matrix[(row, col_start + x.len())] =
            (-1.0_f32).fused_mul_add(correction, matrix[(row, col_start + x.len())]);
        row_offset += 1;
    }
}

fn eigen_ldlt_gemv_vector_sub_dynamic_f32(
    matrix: &DMatrix<f32>,
    result: &mut DVector<f32>,
    row_start: usize,
    row_count: usize,
    col_start: usize,
    col_count: usize,
    transpose: bool,
) {
    let mut row_offset = 0;
    while row_count - row_offset >= 8 {
        let mut correction = [0.0_f32; 8];
        for offset in 0..col_count {
            let column = col_start + offset;
            let value = result[column];
            for lane in 0..8 {
                let row = row_start + row_offset + lane;
                let coefficient = if transpose {
                    matrix[(column, row)]
                } else {
                    matrix[(row, column)]
                };
                correction[lane] = coefficient.fused_mul_add(value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            result[row] = (-1.0_f32).fused_mul_add(value, result[row]);
        }
        row_offset += 8;
    }
    while row_count - row_offset >= 4 {
        let mut correction = [0.0_f32; 4];
        for offset in 0..col_count {
            let column = col_start + offset;
            let value = result[column];
            for lane in 0..4 {
                let row = row_start + row_offset + lane;
                let coefficient = if transpose {
                    matrix[(column, row)]
                } else {
                    matrix[(row, column)]
                };
                correction[lane] = coefficient.fused_mul_add(value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            result[row] = (-1.0_f32).fused_mul_add(value, result[row]);
        }
        row_offset += 4;
    }
    while row_count - row_offset >= 2 {
        let mut correction = [0.0_f32; 2];
        for offset in 0..col_count {
            let column = col_start + offset;
            let value = result[column];
            for lane in 0..2 {
                let row = row_start + row_offset + lane;
                let coefficient = if transpose {
                    matrix[(column, row)]
                } else {
                    matrix[(row, column)]
                };
                correction[lane] = coefficient.fused_mul_add(value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = row_start + row_offset + lane;
            result[row] = (-1.0_f32).fused_mul_add(value, result[row]);
        }
        row_offset += 2;
    }
    while row_offset < row_count {
        let row = row_start + row_offset;
        let mut correction = 0.0_f32;
        for offset in 0..col_count {
            let column = col_start + offset;
            let coefficient = if transpose {
                matrix[(column, row)]
            } else {
                matrix[(row, column)]
            };
            correction = coefficient.fused_mul_add(result[column], correction);
        }
        result[row] = (-1.0_f32).fused_mul_add(correction, result[row]);
        row_offset += 1;
    }
}

fn eigen_ldlt_forward_solve_dynamic_f32(result: &mut DVector<f32>, matrix: &DMatrix<f32>) {
    let size = result.len();
    let mut panel_start = 0;
    while panel_start < size {
        let panel_width = (size - panel_start).min(8);
        for offset in 0..panel_width {
            let index = panel_start + offset;
            let value = result[index];
            if value != 0.0 {
                for row_offset in (offset + 1)..panel_width {
                    let row = panel_start + row_offset;
                    result[row] = (-value).fused_mul_add(matrix[(row, index)], result[row]);
                }
            }
        }
        let end = panel_start + panel_width;
        if end < size {
            eigen_ldlt_gemv_vector_sub_dynamic_f32(
                matrix,
                result,
                end,
                size - end,
                panel_start,
                panel_width,
                false,
            );
        }
        panel_start = end;
    }
}

fn eigen_ldlt_backward_solve_dynamic_f32(result: &mut DVector<f32>, matrix: &DMatrix<f32>) {
    let trace_path = m7im15_backward_trace_path();
    let trace_enabled = trace_path.is_some();
    let mut trace = Vec::new();
    if trace_enabled {
        m7im15_push_backward_trace(&mut trace, "initial", 0, 0, None, None, None, None, result);
    }
    let size = result.len();
    let mut panel_end = size;
    while panel_end > 0 {
        let panel_width = panel_end.min(8);
        let panel_start = panel_end - panel_width;
        if panel_end < size {
            // `matrixL().transpose()` is a row-major triangular expression in
            // Eigen.  Its dynamic RHS solve therefore dispatches the
            // row-major 8/4/2-row GEMV below; using the column-major helper
            // here changes the packet/reduction order even though the
            // mathematical operation is identical.
            eigen_ldlt_gemv_row_major_sub_dynamic_f32(
                matrix,
                result,
                panel_start,
                panel_width,
                panel_end,
                size - panel_end,
            );
            if trace_enabled {
                m7im15_push_backward_trace(
                    &mut trace,
                    "external",
                    panel_start,
                    panel_width,
                    None,
                    None,
                    None,
                    None,
                    result,
                );
            }
        }
        for offset in 0..panel_width {
            let index = panel_end - offset - 1;
            let known = offset;
            if known > 0 {
                let mut lhs = Vec::with_capacity(known);
                let mut rhs = Vec::with_capacity(known);
                for inner in 0..known {
                    lhs.push(matrix[(index + inner + 1, index)]);
                    rhs.push(result[index + inner + 1]);
                }
                // The transposed row-major panel is at most seven terms after
                // the diagonal.  Eigen's cwiseProduct().sum() computes each
                // product and accumulates it with ordinary f32 operations.
                let correction = eigen_ldlt_backward_panel_sum_f32(&lhs, &rhs);
                result[index] -= correction;
                if trace_enabled {
                    m7im15_push_backward_trace(
                        &mut trace,
                        "internal",
                        panel_start,
                        panel_width,
                        Some(index),
                        Some(correction),
                        Some(&lhs),
                        Some(&rhs),
                        result,
                    );
                }
            }
        }
        if trace_enabled {
            m7im15_push_backward_trace(
                &mut trace,
                "panel",
                panel_start,
                panel_width,
                None,
                None,
                None,
                None,
                result,
            );
        }
        panel_end = panel_start;
    }
    if let Some(path) = trace_path {
        let record = json!({
            "schema": "basalt.m7im15.rust_ldlt_backward_trace.v1",
            "steps": trace,
        });
        if let Ok(line) = serde_json::to_string(&record) {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                let _ = writeln!(file, "{line}");
            }
        }
    }
}

/// Claim the optional first-call backward trace without affecting solver
/// arithmetic or the ordinary stage capture.
fn m7im15_backward_trace_path() -> Option<std::path::PathBuf> {
    static CLAIMED: OnceLock<()> = OnceLock::new();
    let path = env::var_os("VISLOC_BASALT_M7IM15_LDLT_BACKWARD_TRACE")?;
    CLAIMED.set(()).ok()?;
    Some(path.into())
}

fn m7im15_push_backward_trace(
    trace: &mut Vec<serde_json::Value>,
    stage: &str,
    panel_start: usize,
    panel_width: usize,
    index: Option<usize>,
    correction: Option<f32>,
    lhs: Option<&[f32]>,
    rhs: Option<&[f32]>,
    result: &DVector<f32>,
) {
    trace.push(json!({
        "stage": stage,
        "panel_start": panel_start,
        "panel_width": panel_width,
        "index": index,
        "correction_bits": correction.map(|value| format!("{:08x}", value.to_bits())),
        "lhs_bits": lhs.map(|values| values.iter().map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>()),
        "rhs_bits": rhs.map(|values| values.iter().map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>()),
        "f32_bits": result
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
    }));
}

/// Eigen's row-major `general_matrix_vector_product` used by the backward
/// solve of `matrixL().transpose()`.
///
/// The transposed view has rows corresponding to the original matrix columns
/// and contiguous row-major packet lanes across the trailing solve columns.
/// AVX float uses Packet8, Packet4, Packet2, then a scalar tail.  Keep the
/// packet predux tree explicit so this helper has the same f32 schedule as
/// Eigen's `predux(Packet8f/Packet4f/Packet2f)` implementations.
#[inline(never)]
fn eigen_ldlt_gemv_row_major_sub_dynamic_f32(
    matrix: &DMatrix<f32>,
    result: &mut DVector<f32>,
    row_start: usize,
    row_count: usize,
    col_start: usize,
    col_count: usize,
) {
    let coefficient = |row: usize, column: usize| matrix[(column, row)];
    // Eigen's row-major GEMV writes `res += alpha * dot` after the packet
    // reduction.  The scalar alpha multiplication is exact sign negation;
    // keep the final add unfused as in Eigen's `res += alpha*cc` statement.
    let apply = |value: f32, destination: f32| destination - value;

    let predux8 = |packet: &[f32; 8]| {
        let pair0 = packet[0] + packet[4];
        let pair1 = packet[1] + packet[5];
        let pair2 = packet[2] + packet[6];
        let pair3 = packet[3] + packet[7];
        (pair0 + pair2) + (pair1 + pair3)
    };
    let predux4 = |packet: &[f32; 4]| {
        let pair0 = packet[0] + packet[2];
        let pair1 = packet[1] + packet[3];
        pair0 + pair1
    };
    let predux2 = |packet: &[f32; 2]| packet[0] + packet[1];

    let mut row_offset = 0;
    while row_count - row_offset >= 8 {
        let mut packet = [[0.0_f32; 8]; 8];
        let mut column_offset = 0;
        while col_count - column_offset >= 8 {
            for lane in 0..8 {
                let row = row_start + row_offset + lane;
                for depth in 0..8 {
                    let column = col_start + column_offset + depth;
                    packet[lane][depth] =
                        coefficient(row, column).fused_mul_add(result[column], packet[lane][depth]);
                }
            }
            column_offset += 8;
        }
        let mut reduced = [0.0_f32; 8];
        for lane in 0..8 {
            reduced[lane] = predux8(&packet[lane]);
        }
        while column_offset < col_count {
            let column = col_start + column_offset;
            for lane in 0..8 {
                let row = row_start + row_offset + lane;
                reduced[lane] =
                    coefficient(row, column).fused_mul_add(result[column], reduced[lane]);
            }
            column_offset += 1;
        }
        for lane in 0..8 {
            let row = row_start + row_offset + lane;
            result[row] = apply(reduced[lane], result[row]);
        }
        row_offset += 8;
    }

    while row_count - row_offset >= 4 {
        let mut packet = [[0.0_f32; 8]; 4];
        let mut column_offset = 0;
        while col_count - column_offset >= 8 {
            for lane in 0..4 {
                let row = row_start + row_offset + lane;
                for depth in 0..8 {
                    let column = col_start + column_offset + depth;
                    packet[lane][depth] =
                        coefficient(row, column).fused_mul_add(result[column], packet[lane][depth]);
                }
            }
            column_offset += 8;
        }
        let mut reduced = [0.0_f32; 4];
        for lane in 0..4 {
            reduced[lane] = predux8(&packet[lane]);
        }
        while column_offset < col_count {
            let column = col_start + column_offset;
            for lane in 0..4 {
                let row = row_start + row_offset + lane;
                reduced[lane] =
                    coefficient(row, column).fused_mul_add(result[column], reduced[lane]);
            }
            column_offset += 1;
        }
        for lane in 0..4 {
            let row = row_start + row_offset + lane;
            result[row] = apply(reduced[lane], result[row]);
        }
        row_offset += 4;
    }

    while row_count - row_offset >= 2 {
        let mut packet = [[0.0_f32; 8]; 2];
        let mut column_offset = 0;
        while col_count - column_offset >= 8 {
            for lane in 0..2 {
                let row = row_start + row_offset + lane;
                for depth in 0..8 {
                    let column = col_start + column_offset + depth;
                    packet[lane][depth] =
                        coefficient(row, column).fused_mul_add(result[column], packet[lane][depth]);
                }
            }
            column_offset += 8;
        }
        let mut reduced = [0.0_f32; 2];
        for lane in 0..2 {
            reduced[lane] = predux8(&packet[lane]);
        }
        while column_offset < col_count {
            let column = col_start + column_offset;
            for lane in 0..2 {
                let row = row_start + row_offset + lane;
                reduced[lane] =
                    coefficient(row, column).fused_mul_add(result[column], reduced[lane]);
            }
            column_offset += 1;
        }
        for lane in 0..2 {
            let row = row_start + row_offset + lane;
            result[row] = apply(reduced[lane], result[row]);
        }
        row_offset += 2;
    }

    while row_offset < row_count {
        let row = row_start + row_offset;
        let mut reduced8 = [0.0_f32; 8];
        let mut column_offset = 0;
        while col_count - column_offset >= 8 {
            for depth in 0..8 {
                let column = col_start + column_offset + depth;
                reduced8[depth] =
                    coefficient(row, column).fused_mul_add(result[column], reduced8[depth]);
            }
            column_offset += 8;
        }
        let mut reduced = predux8(&reduced8);
        let mut reduced4 = [0.0_f32; 4];
        while col_count - column_offset >= 4 {
            for depth in 0..4 {
                let column = col_start + column_offset + depth;
                reduced4[depth] =
                    coefficient(row, column).fused_mul_add(result[column], reduced4[depth]);
            }
            column_offset += 4;
        }
        reduced += predux4(&reduced4);
        let mut reduced2 = [0.0_f32; 2];
        while col_count - column_offset >= 2 {
            for depth in 0..2 {
                let column = col_start + column_offset + depth;
                reduced2[depth] =
                    coefficient(row, column).fused_mul_add(result[column], reduced2[depth]);
            }
            column_offset += 2;
        }
        reduced += predux2(&reduced2);
        while column_offset < col_count {
            let column = col_start + column_offset;
            reduced += coefficient(row, column) * result[column];
            column_offset += 1;
        }
        result[row] = apply(reduced, result[row]);
        row_offset += 1;
    }
}

/// Eigen's dynamic-size inner product for the LDLT rank update.  With the
/// pinned `-march=skylake` oracle, eight terms use a Packet8f product followed
/// by a balanced Packet4f reduction; shorter tails use scalar `pmadd` (FMA).
fn eigen_ldlt_inner_product_f32<I, J>(lhs: I, rhs: J) -> f32
where
    I: IntoIterator<Item = f32>,
    J: IntoIterator<Item = f32>,
{
    let lhs: Vec<_> = lhs.into_iter().collect();
    let rhs: Vec<_> = rhs.into_iter().collect();
    debug_assert_eq!(lhs.len(), rhs.len());
    if lhs.len() == 8 {
        let products: [f32; 8] = std::array::from_fn(|index| lhs[index] * rhs[index]);
        let pair = [
            products[0] + products[4],
            products[1] + products[5],
            products[2] + products[6],
            products[3] + products[7],
        ];
        return (pair[0] + pair[2]) + (pair[1] + pair[3]);
    }
    debug_assert!(!lhs.is_empty());
    let mut result = lhs[0] * rhs[0];
    for index in 1..lhs.len() {
        result = lhs[index].fused_mul_add(rhs[index], result);
    }
    result
}

/// Eigen's AVX2 `general_matrix_vector_product` tail for
/// `A21.noalias() -= A20 * temporary`.  The row tail is processed as Packet4f,
/// Packet2f, then scalar lanes, with a fused `alpha=-1` store.
fn eigen_ldlt_gemv_sub_f32(matrix: &mut Matrix9F32, pivot: usize, rows: usize, temporary: &[f32]) {
    let mut row_offset = 0;
    while rows - row_offset >= 4 {
        let mut correction = [0.0_f32; 4];
        for (index, value) in temporary.iter().enumerate() {
            for lane in 0..4 {
                let row = pivot + 1 + row_offset + lane;
                correction[lane] = matrix[(row, index)].fused_mul_add(*value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = pivot + 1 + row_offset + lane;
            matrix[(row, pivot)] = (-1.0_f32).fused_mul_add(value, matrix[(row, pivot)]);
        }
        row_offset += 4;
    }
    while rows - row_offset >= 2 {
        let mut correction = [0.0_f32; 2];
        for (index, value) in temporary.iter().enumerate() {
            for lane in 0..2 {
                let row = pivot + 1 + row_offset + lane;
                correction[lane] = matrix[(row, index)].fused_mul_add(*value, correction[lane]);
            }
        }
        for (lane, value) in correction.into_iter().enumerate() {
            let row = pivot + 1 + row_offset + lane;
            matrix[(row, pivot)] = (-1.0_f32).fused_mul_add(value, matrix[(row, pivot)]);
        }
        row_offset += 2;
    }
    while row_offset < rows {
        let row = pivot + 1 + row_offset;
        let mut correction = 0.0_f32;
        for (index, value) in temporary.iter().enumerate() {
            correction = matrix[(row, index)].fused_mul_add(*value, correction);
        }
        matrix[(row, pivot)] = (-1.0_f32).fused_mul_add(correction, matrix[(row, pivot)]);
        row_offset += 1;
    }
}

fn eigen_ldlt_forward_solve_f32(result: &mut Matrix9F32, matrix: &Matrix9F32) {
    for pivot in 0..8 {
        for column in 0..9 {
            let value = result[(pivot, column)];
            for row in (pivot + 1)..8 {
                result[(row, column)] =
                    (-value).fused_mul_add(matrix[(row, pivot)], result[(row, column)]);
            }
        }
    }
    for column in 0..9 {
        let mut correction = 0.0_f32;
        for pivot in 0..8 {
            correction = matrix[(8, pivot)].fused_mul_add(result[(pivot, column)], correction);
        }
        result[(8, column)] = (-1.0_f32).fused_mul_add(correction, result[(8, column)]);
    }
}

fn m7im_matrix_bits(matrix: &Matrix9F32) -> String {
    (0..9)
        .flat_map(|row| (0..9).map(move |column| matrix[(row, column)].to_bits()))
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn m7im_lower_bits(matrix: &Matrix9F32) -> String {
    (0..9)
        .flat_map(|row| {
            (0..9).map(move |column| {
                if column <= row {
                    matrix[(row, column)].to_bits()
                } else {
                    0
                }
            })
        })
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// Optional one-shot dynamic LDLT schedule capture.  This is intentionally an
/// observation-only sidecar: all values are cloned after the production
/// operations have completed and serialized as f32 bit patterns.  The live
/// frame-4 audit enables it with `VISLOC_BASALT_M7IM15_LDLT_STAGE`.
fn emit_m7im15_dynamic_ldlt_stage(
    input: &DMatrix<f32>,
    rhs: &DVector<f32>,
    matrix: &DMatrix<f32>,
    diagonal: &[f32],
    transpositions: &[usize],
    permutation_rhs: &DVector<f32>,
    forward: &DVector<f32>,
    after_d: &DVector<f32>,
    backward: &DVector<f32>,
    post_permutation: &DVector<f32>,
) {
    let Some(path) = env::var_os("VISLOC_BASALT_M7IM15_LDLT_STAGE") else {
        return;
    };
    static EMITTED: OnceLock<()> = OnceLock::new();
    if EMITTED.set(()).is_err() {
        return;
    }
    let matrix_bits = (0..matrix.ncols())
        .flat_map(|column| (0..matrix.nrows()).map(move |row| matrix[(row, column)].to_bits()))
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>();
    let lower_bits = (0..matrix.nrows())
        .flat_map(|row| {
            (0..matrix.ncols()).map(move |column| {
                if column <= row {
                    matrix[(row, column)].to_bits()
                } else {
                    0
                }
            })
        })
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>();
    let vector_bits = |vector: &DVector<f32>| {
        vector
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
    };
    let input_bits = (0..input.ncols())
        .flat_map(|column| (0..input.nrows()).map(move |row| input[(row, column)].to_bits()))
        .map(|bits| format!("{bits:08x}"))
        .collect::<Vec<_>>();
    let record = json!({
        "schema": "basalt.m7im15.rust_ldlt_live.v1",
        "layout": "column_major",
        "input": {
            "rows": input.nrows(),
            "cols": input.ncols(),
            "f32_bits": input_bits,
        },
        "rhs": vector_bits(rhs),
        "matrixLDLT": matrix_bits,
        "matrixLDLT_lower_row_major_bits": lower_bits,
        "D": diagonal
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>(),
        "pivots": transpositions,
        "P_b": vector_bits(permutation_rhs),
        "forward_L": vector_bits(forward),
        "after_D": vector_bits(after_d),
        "backward_LT": vector_bits(backward),
        "P_T": vector_bits(post_permutation),
    });
    let Ok(line) = serde_json::to_string(&record) else {
        return;
    };
    let path = std::path::PathBuf::from(path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires native trial costs and matching candidate states on external SSD"]
    fn m11_native_trial_bias_candidate_exact() {
        use std::io::BufRead;
        let native = std::env::var("M11_NATIVE_COST_PARTS").unwrap();
        let detail = std::env::var("M11_RUST_TRIAL_DETAIL").unwrap();
        let expected = std::io::BufReader::new(std::fs::File::open(native).unwrap())
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(&line.unwrap()).unwrap())
            .collect::<Vec<_>>();
        let mut checked = 0;
        for line in std::io::BufReader::new(std::fs::File::open(detail).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            if row["phase"] != "trial" {
                continue;
            }
            let i = row["iteration"].as_u64().unwrap() as usize;
            // Existing LM diverges after iter5; compare only common candidates.
            if i > 5 {
                continue;
            }
            let mut states = row["blocks"]["states"]
                .as_array()
                .unwrap()
                .iter()
                .collect::<Vec<_>>();
            states.sort_by_key(|s| s["timestamp_ns"].as_i64().unwrap());
            assert_eq!(states.len(), 3);
            let nav = |s: &serde_json::Value| {
                let mut nav = BasaltNavState::default();
                nav.gyro_bias_rad_s = Vector3::from_iterator(
                    s["bias_gyro"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap()),
                );
                nav.accel_bias_m_s2 = Vector3::from_iterator(
                    s["bias_accel"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_f64().unwrap()),
                );
                nav
            };
            let mut sums = (0.0_f32, 0.0_f32);
            for pair in states.windows(2) {
                let dt_ns = pair[1]["timestamp_ns"].as_i64().unwrap()
                    - pair[0]["timestamp_ns"].as_i64().unwrap();
                let costs = trial_bias_cost_f32(
                    &nav(pair[0]),
                    &nav(pair[1]),
                    dt_ns,
                    BiasRandomWalkNoise {
                        gyro_density: 10000.0,
                        accel_density: 1000.0,
                    },
                )
                .unwrap();
                sums.0 += costs.0;
                sums.1 += costs.1;
            }
            assert_eq!(expected[i]["iteration"], i);
            for (name, cost) in [("bg", sums.0), ("ba", sums.1)] {
                let word = u32::from_str_radix(expected[i][name]["f32_bits"].as_str().unwrap(), 16)
                    .unwrap();
                assert_eq!(cost.to_bits(), word, "iteration {i} {name}");
            }
            checked += 1;
        }
        assert_eq!(checked, 6);
    }

    #[test]
    #[ignore = "requires audited native trial IMU capture on external SSD"]
    fn m11_native_trial_imu_quadratic_all_links_exact() {
        use std::io::BufRead;
        let path = std::env::var("M11_NATIVE_TRIAL_IMU_LINKS").unwrap();
        let mut counts = [0_usize; 8];
        for line in std::io::BufReader::new(std::fs::File::open(path).unwrap()).lines() {
            let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            let parse = |v: &serde_json::Value| {
                f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap())
            };
            let s = Matrix9F32::from_iterator(
                row["sqrt_information"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(parse),
            );
            let raw = SMatrix::<f32, 9, 1>::from_iterator(
                row["raw"].as_array().unwrap().iter().map(parse),
            );
            let (inverse, weighted, cost) = trial_imu_quadratic_stages_f32(&s, &raw);
            for (i, value) in inverse.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    parse(&row["covariance_inverse"][i]).to_bits(),
                    "inverse lane {i}"
                );
            }
            for (i, value) in weighted.iter().enumerate() {
                assert_eq!(
                    value.to_bits(),
                    parse(&row["half_residual_covariance_product"][i]).to_bits(),
                    "weighted lane {i}"
                );
            }
            assert_eq!(cost.to_bits(), parse(&row["objective"]).to_bits());
            counts[row["iteration"].as_u64().unwrap() as usize] += 1;
        }
        assert_eq!(counts, [2; 8]);
    }
    use visloc_core::geometry::SE3;

    fn state() -> BasaltNavState {
        BasaltNavState::default()
    }

    #[test]
    fn m11_frame21_primary_rotation_matrix_native_fma_candidate() {
        // Primary residual call 2c3592 also reaches out-of-line 2b1dd0.
        // Frame21 iteration2, interval20->21, matrix captured at 2bb8ba.
        let f = f32::from_bits;
        let q0 = UnitQuaternion::new_unchecked(Quaternion::new(
            f(0x3f08a49d),
            f(0xbd4b8f40),
            f(0xbf5819b2),
            f(0xbbedc3f1),
        ));
        let inverse = sophus_so3_inverse(q0);
        let old = eigen_rotation_matrix_f32(&inverse);
        assert_eq!(old[(2, 2)].to_bits(), 0xbedc34f0);
        let candidate = eigen_current_rotation_matrix_f32(&inverse);
        assert_eq!(
            candidate.map(f32::to_bits).as_slice(),
            [
                0xbed9bb48, 0x3dbbb26c, 0xbf6681f3, 0x3d9bf82e, 0x3f7eb560, 0x3d85bda5, 0x3f66e07b,
                0xbd2720b1, 0xbedc34ec,
            ]
        );
        let operand = Vector3::new(f(0xb7e1b489), f(0xb8757114), f(0x3c646eeb));
        assert_eq!(
            eigen_matrix_vector_f32(&candidate, operand)
                .map(f32::to_bits)
                .as_slice(),
            [0x3c4e2143, 0xba250a45, 0xbbc3d35e]
        );
    }

    #[test]
    fn m11_frame14_current_rotation_matrix_native_fma_candidate() {
        // Pinned native 2b1dd0, captured inside current residual call at
        // frame14 iteration1, link12->13; matrix at 2bb8ba, product at 2bbaa0.
        let f = f32::from_bits;
        let q0 = UnitQuaternion::new_unchecked(Quaternion::new(
            f(0x3f0d8495),
            f(0xbd5bab48),
            f(0xbf54e1d2),
            f(0xbb979cdb),
        ));
        let inverse = sophus_so3_inverse(q0);
        let old = eigen_rotation_matrix_f32(&inverse);
        assert_eq!(old[(2, 2)].to_bits(), 0xbec70d28);
        let q = inverse.quaternion();
        let tx = q.i + q.i;
        let ty = q.j + q.j;
        let tz = q.k + q.k;
        let mut candidate = old;
        // Native 2b1e19 and 2b1e39 fuse x*tx into rounded z*tz/y*ty;
        // 2b1e50/5d then subtract from one. xx diagonal remains separate.
        candidate[(1, 1)] = 1.0 - q.i.mul_add(tx, q.k * tz);
        candidate[(2, 2)] = 1.0 - q.i.mul_add(tx, q.j * ty);
        assert_eq!(candidate, eigen_current_rotation_matrix_f32(&inverse));
        let expected = [
            0xbec420c8, 0x3dc1259a, 0xbf6b3cbc, 0x3dac31a0, 0x3f7e8435, 0x3d89318f, 0x3f6b7dc8,
            0xbd53594d, 0xbec70d24,
        ];
        assert_eq!(candidate.map(f32::to_bits).as_slice(), expected);
        let operand = Vector3::new(f(0xb8bc60e9), f(0x39616b13), f(0x3c5fb24a));
        let product = eigen_matrix_vector_f32(&candidate, operand);
        assert_eq!(
            product.map(f32::to_bits).as_slice(),
            [0x3c4ea2b5, 0xba02df55, 0xbbaac1de]
        );
    }

    #[test]
    fn eigen_f32_rotation_matrix_matches_m7_frame3_probe() {
        // Frame 3 from target/m7_postm7_upstream_frame4_states_20260822.tsv.
        // The expected bits are emitted by the pinned Eigen/Sophus probe in
        // target/m7_state_quat_probe.cpp.  This pins the first same-input
        // factor boundary without depending on a machine-local JSON path.
        let q = UnitQuaternion::new_normalize(Quaternion::new(
            0.589211106300354_f32,
            -0.054311808198690414_f32,
            -0.8061297535896301_f32,
            -0.005946322809904814_f32,
        ));
        let matrix = eigen_rotation_matrix_f32(&q.inverse());
        let expected = [
            0xbe997a48_u32,
            0x3da4fb4e,
            0x3f735afd,
            0x3dc1aef8,
            0x3f7e78bc,
            0xbd5ee280,
            0xbf730654,
            0x3d96b5f7,
            0xbe9c7648,
        ];
        let actual: Vec<_> = matrix.iter().map(|value| value.to_bits()).collect();
        // nalgebra iterates column-major; compare in the row-major order used
        // by the upstream diagnostic and Eigen probe.
        let actual_row_major: Vec<_> = (0..3)
            .flat_map(|row| (0..3).map(move |column| matrix[(row, column)].to_bits()))
            .collect();
        assert_eq!(actual_row_major, expected);
        assert_eq!(actual.len(), 9);
    }

    #[test]
    fn eigen_velocity_gemv_matches_clean_frame4_link0_schedule() {
        // R0_inv and the velocity RHS are the first clean MH-01 frame-4
        // IMU-link fixture.  The expected result pins the Eigen lazy RHS
        // reduction (seed k=2, then FMA k=1 and k=0).  A conventional
        // materialized/vector reduction differs in the x lane by one ULP and
        // changes the downstream velocity residual.
        let matrix = SMatrix::<f32, 3, 3>::from_column_slice(&[
            f32::from_bits(0xbe9341b0),
            f32::from_bits(0x3dad753c),
            f32::from_bits(0xbf743906),
            f32::from_bits(0x3dad753c),
            f32::from_bits(0x3f7e92e4),
            f32::from_bits(0x3d8083fe),
            f32::from_bits(0x3f743906),
            f32::from_bits(0xbd8083fe),
            f32::from_bits(0xbe961be8),
        ]);
        let rhs = Vector3::new(
            f32::from_bits(0x3cabe7c0),
            f32::from_bits(0xbbc3c000),
            f32::from_bits(0x3ed16b71),
        );
        let actual = eigen_matrix_vector_velocity_f32(&matrix, rhs);
        assert_eq!(
            actual.as_slice(),
            &[
                f32::from_bits(0x3ec46fac),
                f32::from_bits(0xbcf45e4f),
                f32::from_bits(0xbe0fadf3),
            ]
        );
    }

    #[test]
    fn frame6_link0_so3_and_bias_block_match_native_fixture() {
        // Frame-6 LM iteration 3 link0 (4 -> 5), interval starting at
        // 1403636579963555584 ns.  The FEJ helper input and J matrices are
        // copied from the native SO3 capture at helper entry 59.  The D block
        // is the direct IntegratedImuMeasurement::d_state_d_bg_ capture, not
        // a reconstructed/manual bias fixture.
        let phi = Vector3::new(
            f32::from_bits(0x3a9c1baf),
            f32::from_bits(0x3a88808d),
            f32::from_bits(0x3b5a01de),
        );
        let h2 = eigen_matrix_product_f32(hat_f32(phi), hat_f32(phi));
        let theta2 = eigen_squared_norm3_f32(phi);
        let right = so3_jacobian_inverse_f32(phi, 1.0);
        let left = so3_jacobian_inverse_f32(phi, -1.0);
        let d_state_d_bg_rotation = SMatrix::<f32, 3, 3>::from_column_slice(&[
            f32::from_bits(0xbd4ccbbb),
            f32::from_bits(0xb8a19f0c),
            f32::from_bits(0xb9824956),
            f32::from_bits(0x38a1712c),
            f32::from_bits(0xbd4ccc91),
            f32::from_bits(0x3799d924),
            f32::from_bits(0x39824c9a),
            f32::from_bits(0xb7956d77),
            f32::from_bits(0xbd4ccbd2),
        ]);
        let bias_product = eigen_matrix_product_f32(left, d_state_d_bg_rotation);
        let row_bits = |m: SMatrix<f32, 3, 3>| {
            (0..3)
                .flat_map(|row| (0..3).map(move |column| m[(row, column)].to_bits()))
                .collect::<Vec<_>>()
        };
        assert_eq!(theta2.to_bits(), 0x3763a5d4);
        assert_eq!(
            row_bits(h2),
            vec![
                0xb74bd964, 0x35a67a32, 0x3684f0b7, 0x35a67a32, 0xb751739e, 0x36687cee, 0x3684f0b7,
                0x36687cee, 0xb627fa97,
            ]
        );
        assert_eq!(
            row_bits(right),
            vec![
                0x3f7ffff0, 0xbad9fe9e, 0x3a089553, 0x3ada051e, 0x3f7ffff0, 0xba1c0985, 0xba086bc7,
                0x3a1c2dd9, 0x3f7ffffd,
            ]
        );
        assert_eq!(
            row_bits(left),
            vec![
                0x3f7ffff0, 0x3ada051e, 0xba086bc7, 0xbad9fe9e, 0x3f7ffff0, 0x3a1c2dd9, 0x3a089553,
                0xba1c0985, 0x3f7ffffd,
            ]
        );
        assert_eq!(
            row_bits(bias_product),
            vec![
                0xbd4ccbae, 0xb6cfe491, 0x398fece8, 0x36c75e41, 0xbd4ccca3, 0xb849636a, 0xb98feba2,
                0x3849ebe8, 0xbd4ccbaa,
            ]
        );
    }

    #[test]
    fn frame6_link0_so3_log_matches_native_atan2f_fixture() {
        // The preceding producer stages are exact.  This is the final
        // quaternion emitted by the pinned native chain; glibc's float
        // atan2f reduces its positive-real branch through atanf(y / x),
        // whereas the target's f32::atan2 follows a double-backed path.
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f7fffe4),
            f32::from_bits(0x3a1c1baa),
            f32::from_bits(0x3a088089),
            f32::from_bits(0x3ada01d7),
        ));
        let actual = so3_log_f32(rotation);
        assert_eq!(
            actual.as_slice(),
            &[
                f32::from_bits(0x3a9c1baf),
                f32::from_bits(0x3a88808d),
                f32::from_bits(0x3b5a01de),
            ]
        );
        let n = eigen_squared_norm3_f32(Vector3::new(
            f32::from_bits(0x3a1c1baa),
            f32::from_bits(0x3a088089),
            f32::from_bits(0x3ada01d7),
        ))
        .sqrt();
        let w = f32::from_bits(0x3f7fffe4);
        assert_eq!(native_atan2_f32(n, w), (n / w).atan());
    }

    #[test]
    fn m7im15_frame4_iter1_accepted_state_inverse_probe() {
        let q0 = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f182ffd),
            f32::from_bits(0xbd582e43),
            f32::from_bits(0xbf4d686f),
            f32::from_bits(0x00000000),
        ));
        let q1 = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f186a45),
            f32::from_bits(0xbd5cff35),
            f32::from_bits(0xbf4d37d1),
            f32::from_bits(0xbb25d808),
        ));
        let bits = |q: UnitQuaternion<f32>| {
            [q.w, q.i, q.j, q.k]
                .into_iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
        };
        let matrix_bits = |m: SMatrix<f32, 3, 3>| {
            m.iter()
                .map(|value| format!("{:08x}", value.to_bits()))
                .collect::<Vec<_>>()
        };
        println!("q0_inv={:?}", bits(sophus_so3_inverse(q0)));
        println!("q1_inv={:?}", bits(sophus_so3_inverse(q1)));
        println!(
            "r0_inv={:?}",
            matrix_bits(eigen_rotation_matrix_f32(&sophus_so3_inverse(q0)))
        );
    }

    #[test]
    fn upstream_mh01_frame1_2_golden_residual_convention() {
        // Values are from target/basalt_upstream_imu_golden_frame1_2.json.
        // Keeping this compact numeric fixture here catches quaternion order,
        // T_w_i convention, gravity sign, dt, and [p,R,v] row regressions
        // without coupling the unit test to a machine-local dataset path.
        let q1 = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            0.5954498648643494,
            -0.05392327159643173,
            -0.801576554775238,
            -0.002603980479761958,
        ));
        let q2 = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            0.5933784246444702,
            -0.05394548922777176,
            -0.8030967712402344,
            -0.00525779603049159,
        ));
        let from = BasaltNavState {
            imu_to_world: SE3::new(
                q1,
                Vector3::new(
                    0.0003828657791018486,
                    -0.0000985765946097672,
                    -0.002031802199780941,
                ),
            ),
            velocity_world_m_s: Vector3::new(
                0.020984530448913574,
                -0.00597381591796875,
                -0.08147591352462769,
            ),
            gyro_bias_rad_s: Vector3::zeros(),
            accel_bias_m_s2: Vector3::zeros(),
        };
        let to = BasaltNavState {
            imu_to_world: SE3::new(
                q2,
                Vector3::new(
                    0.0018608879763633013,
                    -0.0004836908192373812,
                    -0.007697241380810738,
                ),
            ),
            velocity_world_m_s: Vector3::new(
                0.034741103649139404,
                -0.008907575160264969,
                -0.1420118808746338,
            ),
            gyro_bias_rad_s: Vector3::zeros(),
            accel_bias_m_s2: Vector3::zeros(),
        };
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.050000128;
        delta.delta_position = Vector3::new(
            0.010060002095997334,
            -0.0006882318994030356,
            -0.003518919460475445,
        );
        delta.delta_velocity = Vector3::new(
            0.4063984751701355,
            -0.02750103175640106,
            -0.13839077949523926,
        );
        delta.delta_rotation =
            nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                0.9999932050704956,
                -0.002248205477371812,
                -0.0024225490633398294,
                -0.0016497710021212697,
            ));
        let residual = residual(&from, &to, &delta, Vector3::new(0.0, 0.0, -9.81));
        let expected = [
            0.0,
            -5.820766091346741e-11,
            1.3969838619232178e-9,
            -6.982149969303464e-9,
            -6.77322944397929e-8,
            2.8551919162289607e-10,
            -2.9802322387695312e-8,
            3.725290298461914e-9,
            2.9802322387695312e-8,
        ];
        assert!(
            residual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| (actual - expected).abs() < 2.0e-6),
            "residual={residual:?}"
        );

        // The same fixture is also a scalar-boundary check.  These are the
        // pinned upstream float residual values; the tolerance allows only
        // the final platform libm/rotation-matrix rounding, not a change in
        // row order or SO(3) composition.
        let residual_f32 = residual_f32(&from, &to, &delta, Vector3::new(0.0, 0.0, -9.81));
        let expected_f32 = [
            0.0_f32,
            -5.8207661e-11,
            1.3969839e-9,
            -6.9821500e-9,
            -6.7732294e-8,
            2.8551920e-10,
            -2.9802322e-8,
            3.7252903e-9,
            2.9802322e-8,
        ];
        assert!(
            residual_f32
                .iter()
                .zip(expected_f32)
                .all(|(actual, expected)| (*actual - expected).abs() < 2.0e-7),
            "residual_f32={residual_f32:?}"
        );
    }

    #[test]
    fn corrected_delta_uses_left_rotation_bias_perturbation() {
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.2;
        delta.delta_rotation =
            nalgebra::UnitQuaternion::from_scaled_axis(Vector3::new(0.2, -0.1, 0.3));
        delta.jacobian_rotation_gyro_bias =
            nalgebra::Matrix3::new(-0.2, 0.01, 0.0, -0.03, -0.18, 0.02, 0.01, -0.04, -0.21);
        delta.jacobian_velocity_gyro_bias = nalgebra::Matrix3::identity() * 0.4;
        delta.jacobian_velocity_accel_bias = nalgebra::Matrix3::identity() * -0.2;
        delta.jacobian_position_gyro_bias = nalgebra::Matrix3::identity() * 0.03;
        delta.jacobian_position_accel_bias = nalgebra::Matrix3::identity() * -0.04;
        let bg = Vector3::new(0.01, -0.02, 0.03);
        let ba = Vector3::new(-0.04, 0.05, -0.06);
        let (corrected_rotation, corrected_velocity, corrected_position) = delta.corrected(bg, ba);
        let dtheta = delta.jacobian_rotation_gyro_bias * bg;
        let expected_rotation =
            nalgebra::UnitQuaternion::from_scaled_axis(dtheta) * delta.delta_rotation;
        assert!((corrected_rotation.inverse() * expected_rotation).angle() < 1e-14);
        assert_eq!(
            corrected_velocity,
            delta.delta_velocity
                + delta.jacobian_velocity_gyro_bias * bg
                + delta.jacobian_velocity_accel_bias * ba
        );
        assert_eq!(
            corrected_position,
            delta.delta_position
                + delta.jacobian_position_gyro_bias * bg
                + delta.jacobian_position_accel_bias * ba
        );

        // Construct a predicted state with the corrected delta.  The
        // authoritative residual Exp(bg)*DeltaR*R1^-1*R0 then vanishes.
        let from = BasaltNavState {
            imu_to_world: SE3::new(
                nalgebra::UnitQuaternion::from_scaled_axis(Vector3::new(-0.1, 0.2, 0.05)),
                Vector3::new(0.3, -0.2, 0.1),
            ),
            velocity_world_m_s: Vector3::new(0.2, -0.1, 0.4),
            gyro_bias_rad_s: bg,
            accel_bias_m_s2: ba,
        };
        let to = BasaltNavState {
            imu_to_world: SE3::new(
                from.imu_to_world.rotation * corrected_rotation,
                from.imu_to_world.translation
                    + from
                        .imu_to_world
                        .rotation
                        .transform_vector(&corrected_position)
                    + from.velocity_world_m_s * delta.delta_time,
            ),
            velocity_world_m_s: from.velocity_world_m_s
                + from
                    .imu_to_world
                    .rotation
                    .transform_vector(&corrected_velocity),
            gyro_bias_rad_s: bg,
            accel_bias_m_s2: ba,
        };
        let result = residual(&from, &to, &delta, Vector3::zeros());
        assert!(result.norm() < 1e-12, "result={result:?}");
    }

    #[test]
    fn covariance_whitening_is_finite_and_row_ordered() {
        let mut d = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        d.delta_time = 0.01;
        d.covariance = Matrix9::identity() * 4.0;
        let f = whitened_preintegration_factor(&state(), &state(), &d, Vector3::zeros()).unwrap();
        assert_eq!(f.residual.len(), 9);
        assert_eq!(f.state_jacobian.shape(), (9, 30));
        assert!((f.sqrt_information[(0, 0)] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn covariance_svd_fallback_handles_rank_deficient_matrix() {
        let c = Matrix9::zeros();
        let s = sqrt_information(&c).unwrap();
        assert!(s.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn f32_ldlt_whitener_keeps_eigen_pivot_permutation() {
        // The IMU covariance starts with six structurally zero directions.
        // Eigen::LDLT therefore pivots the final three diagonal entries to
        // the front before forming D^-1/2 L^-1 P.  An unpivoted implementation
        // silently leaves every useful whitener row in the wrong location.
        let mut covariance = Matrix9F32::zeros();
        covariance[(6, 6)] = 4.0;
        covariance[(7, 7)] = 9.0;
        covariance[(8, 8)] = 16.0;
        let sqrt_info = sqrt_information_f32(&covariance).unwrap();
        assert!((sqrt_info[(0, 8)] - 0.25).abs() < 1.0e-7);
        assert!((sqrt_info[(1, 7)] - (1.0 / 3.0)).abs() < 1.0e-7);
        assert!((sqrt_info[(2, 6)] - 0.5).abs() < 1.0e-7);
        assert!(sqrt_info.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn eigen_ldlt_backward_panel_sum_matches_native_k1_to_k7() {
        // These are the first seven internal panel terms from the native
        // frame-4 probe.  The rhs slices are taken from the native input
        // vector immediately after each solved index, and the expected bits
        // are Eigen's cwiseProduct().sum() values.
        let fixtures: &[(&[u32], &[u32], u32)] = &[
            (&[0x3c9169fd], &[0x3c6164a5], 0x39800752),
            (
                &[0xbdc85f6f, 0x3d00d3f6],
                &[0x3c4bca74, 0x3c6164a5],
                0xba4d9755,
            ),
            (
                &[0xbd8da28a, 0xbf297a6a, 0x3bc9630c],
                &[0x3e682af6, 0x3c4bca74, 0x3c6164a5],
                0xbcc336cf,
            ),
            (
                &[0x3b9ea3fd, 0x3cad1c4e, 0x3a1612a0, 0xbee0e635],
                &[0x3c37631f, 0x3e682af6, 0x3c4bca74, 0x3c6164a5],
                0xba9c06f2,
            ),
            (
                &[0x3c70c160, 0xbd17ce6e, 0xbeffa32a, 0xbcedda80, 0x3c22fd9f],
                &[0x3c68635a, 0x3c37631f, 0x3e682af6, 0x3c4bca74, 0x3c6164a5],
                0xbde8b894,
            ),
            (
                &[
                    0xbc5a5530, 0x39acfd4b, 0xbeebe38f, 0xbc1efe3a, 0xbeee3a1a, 0x3967fd84,
                ],
                &[
                    0x3e64e96f, 0x3c68635a, 0x3c37631f, 0x3e682af6, 0x3c4bca74, 0x3c6164a5,
                ],
                0xbc84052d,
            ),
            (
                &[
                    0x3828ab65, 0x3b8780ce, 0xbed70da7, 0x3a6205e8, 0x3b41d272, 0xb9615f1c,
                    0xbedc62d7,
                ],
                &[
                    0x3c41a819, 0x3e64e96f, 0x3c68635a, 0x3c37631f, 0x3e682af6, 0x3c4bca74,
                    0x3c6164a5,
                ],
                0xbc285fae,
            ),
        ];

        for (lhs_bits, rhs_bits, expected) in fixtures {
            let lhs: Vec<_> = lhs_bits.iter().copied().map(f32::from_bits).collect();
            let rhs: Vec<_> = rhs_bits.iter().copied().map(f32::from_bits).collect();
            let actual = eigen_ldlt_backward_panel_sum_f32(&lhs, &rhs);
            assert_eq!(actual.to_bits(), *expected, "k={}", lhs.len());
        }
    }

    #[test]
    fn dynamic_ldlt_solve_matches_native_m7ci_fixture() {
        #[derive(serde::Deserialize)]
        struct Fixture {
            hessian_bits: Vec<String>,
            native_inverse_bits: Vec<String>,
        }

        fn bits(value: &str) -> u32 {
            u32::from_str_radix(value.strip_prefix("0x").unwrap_or(value), 16)
                .expect("fixture bit pattern")
        }

        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../tests/fixtures/m7ci_frame4_cam0_ldlt_boundary.json"
        ))
        .expect("m7ci LDLT fixture must parse");
        let hessian = DMatrix::from_row_slice(
            3,
            3,
            &fixture
                .hessian_bits
                .iter()
                .map(|value| f32::from_bits(bits(value)))
                .collect::<Vec<_>>(),
        );
        let expected: Vec<u32> = fixture
            .native_inverse_bits
            .iter()
            .map(|value| bits(value))
            .collect();
        let mut observed = Vec::with_capacity(9);
        for column in 0..3 {
            let mut rhs = DVector::zeros(3);
            rhs[column] = 1.0;
            let solution =
                eigen_ldlt_solve_f32(&hessian, &rhs).expect("native fixture is nonsingular");
            observed.extend(solution.iter().map(|value| value.to_bits()));
        }
        // The inverse fixture is row-major.  The three solves above are
        // column-major, so transpose the observed lane order before checking
        // the native bits.
        let mut observed_row_major = Vec::with_capacity(9);
        for row in 0..3 {
            for column in 0..3 {
                observed_row_major.push(observed[column * 3 + row]);
            }
        }
        assert_eq!(observed_row_major, expected);
    }

    #[test]
    fn covariance_sqrt_information_is_a_left_whitener_for_correlated_noise() {
        let mut covariance = Matrix9::identity() * 2.0;
        covariance[(0, 1)] = 0.4;
        covariance[(1, 0)] = 0.4;
        covariance[(3, 4)] = -0.2;
        covariance[(4, 3)] = -0.2;
        let sqrt_info = sqrt_information(&covariance).unwrap();
        let whitened = sqrt_info * covariance * sqrt_info.transpose();
        assert!((whitened - Matrix9::identity()).norm() < 1e-10);
    }

    #[test]
    fn imu_jacobian_matches_finite_difference_contract() {
        let mut d = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        d.delta_time = 0.1;
        d.covariance = Matrix9::identity();
        let mut to = state();
        to.imu_to_world.translation.x = 0.02;
        let f = whitened_preintegration_factor(&state(), &to, &d, Vector3::new(0.0, 0.0, -9.81))
            .unwrap();
        assert!(f.state_jacobian[(0, 0)].abs() > 0.0);
        assert!(f.state_jacobian[(3, 18)].abs() > 0.0);
    }

    #[test]
    fn analytic_imu_jacobian_matches_left_pose_finite_difference() {
        let from = BasaltNavState {
            imu_to_world: SE3::new(
                nalgebra::UnitQuaternion::from_scaled_axis(Vector3::new(0.15, -0.08, 0.11)),
                Vector3::new(0.2, -0.1, 0.3),
            ),
            velocity_world_m_s: Vector3::new(0.4, -0.2, 0.1),
            gyro_bias_rad_s: Vector3::new(0.01, -0.02, 0.015),
            accel_bias_m_s2: Vector3::new(-0.03, 0.01, 0.02),
        };
        let to = BasaltNavState {
            imu_to_world: SE3::new(
                nalgebra::UnitQuaternion::from_scaled_axis(Vector3::new(0.22, -0.04, 0.16)),
                Vector3::new(0.24, -0.13, 0.27),
            ),
            velocity_world_m_s: Vector3::new(0.45, -0.1, 0.13),
            gyro_bias_rad_s: Vector3::new(0.012, -0.019, 0.014),
            accel_bias_m_s2: Vector3::new(-0.028, 0.012, 0.018),
        };
        let mut delta = ImuPreintegratedDelta::identity(from.gyro_bias_rad_s, from.accel_bias_m_s2);
        delta.delta_time = 0.08;
        delta.delta_rotation =
            nalgebra::UnitQuaternion::from_scaled_axis(Vector3::new(0.05, -0.03, 0.02));
        delta.delta_velocity = Vector3::new(0.03, -0.02, 0.04);
        delta.delta_position = Vector3::new(0.01, -0.015, 0.02);
        delta.jacobian_rotation_gyro_bias =
            nalgebra::Matrix3::new(-0.07, 0.01, 0.02, -0.02, -0.08, 0.01, 0.01, -0.03, -0.06);
        delta.jacobian_position_gyro_bias = nalgebra::Matrix3::identity() * 0.004;
        delta.jacobian_position_accel_bias = nalgebra::Matrix3::identity() * 0.003;
        delta.jacobian_velocity_gyro_bias = nalgebra::Matrix3::identity() * 0.02;
        delta.jacobian_velocity_accel_bias = nalgebra::Matrix3::identity() * 0.03;
        let gravity = Vector3::new(0.1, -0.2, -9.81);
        let analytic = residual_jacobian(&from, &to, &delta, gravity);
        let eps = 1e-7;
        let mut finite = DMatrix::zeros(IMU_RESIDUAL_DOF, 2 * NAV_STATE_DOF);
        for state_index in 0..2 {
            for local in 0..NAV_STATE_DOF {
                let mut plus = [from.clone(), to.clone()];
                let mut minus = [from.clone(), to.clone()];
                perturb_state(&mut plus[state_index], local, eps);
                perturb_state(&mut minus[state_index], local, -eps);
                let rp = residual(&plus[0], &plus[1], &delta, gravity);
                let rm = residual(&minus[0], &minus[1], &delta, gravity);
                for row in 0..IMU_RESIDUAL_DOF {
                    finite[(row, state_index * NAV_STATE_DOF + local)] =
                        (rp[row] - rm[row]) / (2.0 * eps);
                }
            }
        }
        let mut max_diff = 0.0;
        let mut max_at = (0, 0, 0.0, 0.0);
        for row in 0..IMU_RESIDUAL_DOF {
            for col in 0..2 * NAV_STATE_DOF {
                let diff = (analytic[(row, col)] - finite[(row, col)]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    max_at = (row, col, analytic[(row, col)], finite[(row, col)]);
                }
            }
        }
        assert!(
            max_diff < 2e-6,
            "max analytic-vs-fd difference {max_diff} at {max_at:?}"
        );
    }

    #[test]
    fn bias_walk_uses_both_blocks_and_sqrt_dt() {
        let mut to = state();
        to.gyro_bias_rad_s.x = 0.2;
        let f = whitened_bias_random_walk_factor(
            &state(),
            &to,
            0.25,
            BiasRandomWalkNoise {
                gyro_density: 2.0,
                accel_density: 4.0,
            },
        )
        .unwrap();
        assert_eq!(f.state_jacobian.shape(), (6, 30));
        assert!((f.residual[0] + 0.2 * 4.0).abs() < 1e-12);
        assert_eq!(f.state_jacobian[(0, 9)], 4.0);
        assert_eq!(f.state_jacobian[(0, 24)], -4.0);
    }

    #[test]
    fn upstream_bias_walk_matches_native_float_weight_schedule_for_both_euroc_dts() {
        let expected = [
            (0.049999872_f64, 0x472e_b16b_u32, 0x458b_c122_u32),
            (0.050000128_f64, 0x472e_b14e_u32, 0x458b_c10a_u32),
        ];
        for (dt, expected_gyro, expected_accel) in expected {
            let factor = whitened_bias_random_walk_factor_upstream_f32(
                &state(),
                &state(),
                dt,
                BiasRandomWalkNoise {
                    gyro_density: 10_000.0,
                    accel_density: 1_000.0,
                },
            )
            .unwrap();
            assert_eq!(
                (factor.state_jacobian[(0, 9)] as f32).to_bits(),
                expected_gyro
            );
            assert_eq!(
                (factor.state_jacobian[(0, 24)] as f32).to_bits(),
                expected_gyro ^ 0x8000_0000
            );
            assert_eq!(
                (factor.state_jacobian[(3, 12)] as f32).to_bits(),
                expected_accel
            );
            assert_eq!(
                (factor.state_jacobian[(3, 27)] as f32).to_bits(),
                expected_accel ^ 0x8000_0000
            );
        }
    }
}
