use super::{ImuPreintegratedDelta, ImuPreintegrator};
use crate::ImuSample;
use nalgebra::Vector3;
use thiserror::Error;

pub type TimestampedImu = ImuSample;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum SamplingError {
    #[error("timestamps must be strictly increasing")]
    NonIncreasing,
    #[error("camera interval must have positive duration")]
    InvalidInterval,
    #[error("camera endpoint is outside the IMU sample range")]
    EndpointOutside,
    #[error("at least two IMU samples are required")]
    InsufficientSamples,
}

pub fn interpolate_at(
    samples: &[TimestampedImu],
    timestamp_ns: i64,
) -> Result<TimestampedImu, SamplingError> {
    validate(samples)?;
    if samples.len() < 2 {
        return Err(SamplingError::InsufficientSamples);
    }
    if timestamp_ns < samples[0].timestamp_ns
        || timestamp_ns > samples[samples.len() - 1].timestamp_ns
    {
        return Err(SamplingError::EndpointOutside);
    }
    let i = samples.partition_point(|s| s.timestamp_ns < timestamp_ns);
    if i < samples.len() && samples[i].timestamp_ns == timestamp_ns {
        return Ok(samples[i].clone());
    }
    let (a, b) = (&samples[i - 1], &samples[i]);
    let u = (timestamp_ns - a.timestamp_ns) as f64 / (b.timestamp_ns - a.timestamp_ns) as f64;
    Ok(ImuSample::new(
        timestamp_ns,
        a.gyro_rad_s + (b.gyro_rad_s - a.gyro_rad_s) * u,
        a.accel_m_s2 + (b.accel_m_s2 - a.accel_m_s2) * u,
    ))
}

pub fn integrate_between(
    samples: &[TimestampedImu],
    start_ns: i64,
    end_ns: i64,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    noise: Option<super::ImuNoiseModel>,
) -> Result<ImuPreintegratedDelta, SamplingError> {
    integrate_between_with(
        samples, start_ns, end_ns, bias_gyro, bias_accel, noise, false,
    )
}

/// [`integrate_between`], with the noise discretized per step when
/// `noise_per_step` (see `EstimatorConfig::imu_noise_per_step`).
pub fn integrate_between_with(
    samples: &[TimestampedImu],
    start_ns: i64,
    end_ns: i64,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    noise: Option<super::ImuNoiseModel>,
    noise_per_step: bool,
) -> Result<ImuPreintegratedDelta, SamplingError> {
    if end_ns <= start_ns {
        return Err(SamplingError::InvalidInterval);
    }
    let start = interpolate_at(samples, start_ns)?;
    let end = interpolate_at(samples, end_ns)?;
    let mut points = Vec::with_capacity(samples.len() + 2);
    points.push(start);
    for s in samples {
        if s.timestamp_ns > start_ns && s.timestamp_ns < end_ns {
            points.push(s.clone());
        }
    }
    points.push(end);
    let mut pre = ImuPreintegrator::new(bias_gyro, bias_accel).with_per_step_noise(noise_per_step);
    if let Some(n) = noise {
        pre = pre.with_noise(n).ok_or(SamplingError::InvalidInterval)?;
    }
    for pair in points.windows(2) {
        let dt = (pair[1].timestamp_ns - pair[0].timestamp_ns) as f64 / 1e9;
        pre.integrate_sample_at(
            (pair[0].gyro_rad_s + pair[1].gyro_rad_s) * 0.5,
            (pair[0].accel_m_s2 + pair[1].accel_m_s2) * 0.5,
            dt,
            Some(pair[1].timestamp_ns - start_ns),
        );
    }
    Ok(pre.delta().clone())
}

fn validate(samples: &[TimestampedImu]) -> Result<(), SamplingError> {
    if samples.len() < 2 {
        return Err(SamplingError::InsufficientSamples);
    }
    if samples
        .windows(2)
        .any(|p| p[1].timestamp_ns <= p[0].timestamp_ns)
    {
        return Err(SamplingError::NonIncreasing);
    }
    Ok(())
}
