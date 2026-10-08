//! GNSS fix resampling at camera frame times.
//!
//! [`interpolate_gnss_fix`] turns a receiver's fix stream (its own rate and
//! clock) into one ENU position + covariance per camera frame: a fix within
//! [`GnssInterpolationConfig::max_nearest_offset`] is used directly, otherwise
//! the two bracketing fixes are linearly interpolated when they are at most
//! [`GnssInterpolationConfig::max_bracket_gap`] apart. Wider gaps are
//! **dropouts** and yield `None`.
//!
//! Covariances come from [`gnss_fix_covariance`] (full covariance, else the
//! horizontal/vertical accuracies, else defaults). Interpolation blends them
//! *conservatively* -- GNSS errors are strongly time-correlated, so the
//! midpoint of two fixes is not treated as more accurate than either -- and
//! adds an unmodelled-motion term that grows with the distance to the nearest
//! fix.
//!
//! The joint GNSS/VO estimator that consumes these lives in
//! `visloc_slam::gnss_fusion`.

use nalgebra::{Matrix3, Vector3};

use crate::{GnssMeasurement, MeasurementBuffer, TimeDelta, Timestamp};

/// How GNSS fixes are resampled at camera frame times.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssInterpolationConfig {
    /// Longest interval between the two fixes bracketing a frame that may be
    /// interpolated. Wider gaps are treated as dropouts.
    pub max_bracket_gap: TimeDelta,
    /// A fix this close to the frame time is used directly (this also covers
    /// the first/last frames, which have no bracket).
    pub max_nearest_offset: TimeDelta,
    /// Standard deviation (metres per second of distance to the nearest fix)
    /// added for unmodelled motion between fixes.
    pub motion_sigma_per_second: f64,
    /// Horizontal sigma (metres) for fixes that report no accuracy.
    pub default_horizontal_sigma: f64,
    /// Vertical sigma (metres) for fixes that report no accuracy.
    pub default_vertical_sigma: f64,
    /// Lower bound on any per-axis sigma (metres).
    pub min_sigma: f64,
}

impl Default for GnssInterpolationConfig {
    fn default() -> Self {
        Self {
            max_bracket_gap: TimeDelta::from_nanoseconds(1_500_000_000),
            max_nearest_offset: TimeDelta::from_nanoseconds(20_000_000),
            motion_sigma_per_second: 0.5,
            default_horizontal_sigma: 5.0,
            default_vertical_sigma: 10.0,
            min_sigma: 0.02,
        }
    }
}

/// Which fixes produced an [`InterpolatedGnssFix`] (indices into the
/// time-sorted [`MeasurementBuffer`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GnssFixSource {
    /// A single fix within [`GnssInterpolationConfig::max_nearest_offset`].
    Nearest { index: usize },
    /// Linear blend `(1 − alpha)·before + alpha·after`.
    Interpolated {
        before: usize,
        after: usize,
        alpha: f64,
    },
}

/// A GNSS position resampled at a camera frame time.
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolatedGnssFix {
    pub timestamp: Timestamp,
    /// Antenna position in the local ENU frame (metres).
    pub position_enu: Vector3<f64>,
    /// 3×3 ENU covariance (metres²).
    pub covariance: Matrix3<f64>,
    pub source: GnssFixSource,
}

impl InterpolatedGnssFix {
    pub fn horizontal_sigma(&self) -> f64 {
        self.covariance[(0, 0)].max(self.covariance[(1, 1)]).sqrt()
    }

    pub fn vertical_sigma(&self) -> f64 {
        self.covariance[(2, 2)].sqrt()
    }

    /// `(fix index, blend weight)` of the (up to two) source fixes.
    pub fn source_weights(&self) -> [(usize, f64); 2] {
        match self.source {
            GnssFixSource::Nearest { index } => [(index, 1.0), (index, 0.0)],
            GnssFixSource::Interpolated {
                before,
                after,
                alpha,
            } => [(before, 1.0 - alpha), (after, alpha)],
        }
    }
}

/// ENU covariance of a fix: its full covariance if present, else a diagonal
/// from its horizontal/vertical accuracy, else the configured defaults. Each
/// axis sigma is floored at [`GnssInterpolationConfig::min_sigma`].
pub fn gnss_fix_covariance(
    measurement: &GnssMeasurement,
    config: &GnssInterpolationConfig,
) -> Matrix3<f64> {
    let floor = config.min_sigma * config.min_sigma;
    let mut covariance = match &measurement.position_covariance {
        Some(covariance) if covariance.matrix.iter().all(|v| v.is_finite()) => {
            (covariance.matrix + covariance.matrix.transpose()) * 0.5
        }
        _ => {
            let valid = |sigma: Option<f64>| sigma.filter(|s| s.is_finite() && *s > 0.0);
            let h =
                valid(measurement.horizontal_accuracy).unwrap_or(config.default_horizontal_sigma);
            let v = valid(measurement.vertical_accuracy)
                .or(valid(measurement.horizontal_accuracy).map(|h| 2.0 * h))
                .unwrap_or(config.default_vertical_sigma);
            Matrix3::from_diagonal(&Vector3::new(h * h, h * h, v * v))
        }
    };
    for axis in 0..3 {
        covariance[(axis, axis)] = covariance[(axis, axis)].max(floor);
    }
    covariance
}

/// Resample the GNSS stream at `timestamp` (see the module docs). Returns
/// `None` during a dropout: no fix near the frame and no bracketing pair
/// within [`GnssInterpolationConfig::max_bracket_gap`].
pub fn interpolate_gnss_fix(
    buffer: &MeasurementBuffer<GnssMeasurement>,
    timestamp: Timestamp,
    config: &GnssInterpolationConfig,
) -> Option<InterpolatedGnssFix> {
    let fixes = buffer.as_slice();
    let after = fixes.partition_point(|fix| fix.timestamp < timestamp);
    let before = after.checked_sub(1);
    let seconds_between =
        |a: Timestamp, b: Timestamp| (a.as_nanoseconds() - b.as_nanoseconds()).abs() as f64 * 1e-9;
    let motion = |dt: f64| {
        let sigma = config.motion_sigma_per_second * dt;
        Matrix3::identity() * (sigma * sigma)
    };

    // Nearest fix within tolerance.
    let nearest = [before, (after < fixes.len()).then_some(after)]
        .into_iter()
        .flatten()
        .map(|index| (index, seconds_between(fixes[index].timestamp, timestamp)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    if let Some((index, dt)) = nearest {
        if dt <= config.max_nearest_offset.as_seconds_f64() {
            let fix = &fixes[index];
            return Some(InterpolatedGnssFix {
                timestamp,
                position_enu: fix.position_world.coords,
                covariance: gnss_fix_covariance(fix, config) + motion(dt),
                source: GnssFixSource::Nearest { index },
            });
        }
    }

    let before = before?;
    if after >= fixes.len() {
        return None;
    }
    let (a, b) = (&fixes[before], &fixes[after]);
    let span = seconds_between(b.timestamp, a.timestamp);
    if span <= 0.0 || span > config.max_bracket_gap.as_seconds_f64() {
        return None;
    }
    let dt_before = seconds_between(timestamp, a.timestamp);
    let alpha = (dt_before / span).clamp(0.0, 1.0);
    let position_enu = a.position_world.coords * (1.0 - alpha) + b.position_world.coords * alpha;
    // Convex (not squared) blend: GNSS errors are strongly time-correlated, so
    // averaging two fixes does not shrink the error.
    let covariance = gnss_fix_covariance(a, config) * (1.0 - alpha)
        + gnss_fix_covariance(b, config) * alpha
        + motion(dt_before.min(span - dt_before));
    Some(InterpolatedGnssFix {
        timestamp,
        position_enu,
        covariance,
        source: GnssFixSource::Interpolated {
            before,
            after,
            alpha,
        },
    })
}
