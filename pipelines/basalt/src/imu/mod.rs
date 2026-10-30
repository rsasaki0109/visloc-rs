//! Basalt-compatible IMU timestamping and preintegration primitives.
mod factors;
mod preintegration;
mod sampling;

pub use crate::ImuSample;
pub(crate) use factors::eigen_ldlt_solve_f32;
pub(crate) use factors::trial_bias_cost_f32;
pub(crate) use factors::trial_imu_quadratic_stages_f32;
pub(crate) use factors::trial_preintegration_residual_f32;
pub use factors::{
    diagnostic_whiten_imu_raw_f32, imu_row_products_f32, preintegration_f32_expression_trace,
    preintegration_f32_rotation_producer_trace, preintegration_f32_stages,
    preintegration_f32_stages_fej, preintegration_jacobian_f32, preintegration_residual,
    preintegration_residual_and_jacobian, preintegration_residual_f32, sqrt_information,
    sqrt_information_f32, whitened_bias_random_walk_factor,
    whitened_bias_random_walk_factor_upstream_f32, whitened_preintegration_factor,
    whitened_preintegration_factor_upstream_f32, whitened_preintegration_factor_upstream_f32_fej,
    BiasRandomWalkNoise, ImuFactorError, WhitenedBiasWalkFactor, WhitenedImuFactor,
};
pub(crate) use factors::{
    preintegration_f32_stages_fej_mode, whitened_preintegration_factor_upstream_f32_fej_mode,
};
pub use preintegration::{
    ImuDStateUpdateTrace, ImuNoiseModel, ImuPreintegratedDelta, ImuPreintegrator,
};
pub use sampling::{
    integrate_between, integrate_between_with, interpolate_at, SamplingError, TimestampedImu,
};
