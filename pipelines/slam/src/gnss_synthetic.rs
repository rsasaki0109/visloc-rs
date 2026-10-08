//! Deterministic synthetic GNSS + visual-odometry scenarios for tests, demos,
//! and regression checks of [`crate::gnss_fusion`].
//!
//! A vehicle drives a ~1 km closed loop (an oval with wiggles and gentle
//! altitude change). The camera looks forward (`x` right, `y` down, `z`
//! forward). The generator emits:
//!
//! * the ground-truth camera trajectory in ENU,
//! * a drifting VO trajectory in its own map frame (yaw drift, scale error,
//!   per-step noise; optionally monocular with an arbitrary global scale and
//!   scale drift, and an arbitrary non-gravity-aligned map frame),
//! * a GNSS antenna fix stream at its own rate and clock offset, with
//!   horizontal/vertical noise, a dropout window, and multipath outliers.
//!
//! Everything is seeded and dependency-free (xorshift + Box-Muller).

use nalgebra::{Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};

use visloc_fusion::{GnssMeasurement, MeasurementBuffer, Timed, TimedPose, Timestamp};

use crate::gnss_pose_graph::GnssAlignment;

/// Monocular VO parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticMonocular {
    /// VO units per metre at the first frame.
    pub global_scale: f64,
    /// Log-scale drift added per frame (deterministic trend).
    pub log_scale_drift_per_frame: f64,
}

/// A multipath outlier: the fix closest to `time_seconds` (and the following
/// `burst_length − 1` fixes) is offset by `offset_enu` metres.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticOutlier {
    pub time_seconds: f64,
    pub burst_length: usize,
    pub offset_enu: Vector3<f64>,
}

/// Scenario parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticGnssVoConfig {
    pub frame_count: usize,
    pub frame_rate_hz: f64,
    /// Seconds for one lap of the loop.
    pub lap_seconds: f64,
    pub gnss_rate_hz: f64,
    /// GNSS clock offset relative to the first camera frame (seconds).
    pub gnss_time_offset_seconds: f64,
    pub gnss_horizontal_sigma: f64,
    pub gnss_vertical_sigma: f64,
    /// `(start, duration)` seconds with no fixes.
    pub dropout: Option<(f64, f64)>,
    pub outliers: Vec<SyntheticOutlier>,
    /// Antenna position in the camera frame (metres).
    pub lever_arm: Vector3<f64>,
    /// VO heading drift per frame about the up axis (radians).
    pub vo_yaw_drift_per_frame: f64,
    /// Multiplicative VO translation scale error (e.g. `0.02` = 2 % long).
    pub vo_scale_error: f64,
    /// Per-step VO translation noise (metres, per axis).
    pub vo_translation_noise: f64,
    /// Per-step VO rotation noise (radians, per axis).
    pub vo_rotation_noise: f64,
    /// `Some` for monocular VO.
    pub monocular: Option<SyntheticMonocular>,
    /// `true`: the VO map frame is gravity aligned (`+z` up) and related to
    /// ENU by `map_yaw`/`map_translation`. `false`: the map frame is the first
    /// camera frame (camera-only VO).
    pub gravity_aligned_map: bool,
    pub map_yaw: f64,
    pub map_translation: Vector3<f64>,
    pub seed: u64,
}

impl Default for SyntheticGnssVoConfig {
    /// A 120 s, 5 Hz metric gravity-aligned VO run (~1 km loop, ~1° of
    /// heading drift per minute, 2 % scale error) with a 2 Hz receiver, a 15 s
    /// dropout, and three multipath bursts (6 corrupted fixes).
    fn default() -> Self {
        Self {
            frame_count: 600,
            frame_rate_hz: 5.0,
            lap_seconds: 120.0,
            gnss_rate_hz: 2.0,
            gnss_time_offset_seconds: 0.037,
            gnss_horizontal_sigma: 0.8,
            gnss_vertical_sigma: 1.5,
            dropout: Some((55.0, 15.0)),
            outliers: vec![
                SyntheticOutlier {
                    time_seconds: 22.0,
                    burst_length: 1,
                    offset_enu: Vector3::new(18.0, -11.0, 4.0),
                },
                SyntheticOutlier {
                    time_seconds: 41.3,
                    burst_length: 3,
                    offset_enu: Vector3::new(-14.0, 20.0, -6.0),
                },
                SyntheticOutlier {
                    time_seconds: 93.7,
                    burst_length: 2,
                    offset_enu: Vector3::new(25.0, 9.0, 10.0),
                },
            ],
            lever_arm: Vector3::new(0.0, -1.5, -0.8),
            vo_yaw_drift_per_frame: 0.008_f64.to_radians(),
            vo_scale_error: 0.02,
            vo_translation_noise: 0.01,
            vo_rotation_noise: 0.0005,
            monocular: None,
            gravity_aligned_map: true,
            map_yaw: 0.7,
            map_translation: Vector3::new(-35.0, 12.0, 3.0),
            seed: 42,
        }
    }
}

impl SyntheticGnssVoConfig {
    /// Monocular variant: map frame = first camera frame, VO in arbitrary
    /// units (`0.25` per metre) with 10 % scale drift over the run.
    pub fn monocular() -> Self {
        let base = Self::default();
        Self {
            monocular: Some(SyntheticMonocular {
                global_scale: 0.25,
                log_scale_drift_per_frame: 0.1 / base.frame_count as f64,
            }),
            gravity_aligned_map: false,
            ..base
        }
    }
}

/// A generated scenario.
#[derive(Debug, Clone)]
pub struct SyntheticGnssVoScenario {
    /// Ground-truth camera poses (world = ENU).
    pub ground_truth_enu: Vec<TimedPose>,
    /// Drifting VO poses (world = VO map frame).
    pub vo: Vec<TimedPose>,
    pub gnss: MeasurementBuffer<GnssMeasurement>,
    /// Timestamps of the corrupted (outlier) fixes.
    pub outlier_fix_timestamps: Vec<Timestamp>,
    /// True map-to-ENU alignment at the first frame.
    pub true_alignment: GnssAlignment,
    pub lever_arm: Vector3<f64>,
}

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn gaussian3(&mut self, sigma: f64) -> Vector3<f64> {
        Vector3::new(self.gaussian(), self.gaussian(), self.gaussian()) * sigma
    }
}

/// Camera rotation (camera-to-body): camera `z` forward = body `x`, camera
/// `x` right = body `−y`, camera `y` down = body `−z`.
fn body_from_camera() -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(nalgebra::Matrix3::new(
        0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0,
    )))
}

/// Ground-truth camera-to-ENU pose at time `t` (seconds).
fn ground_truth_camera_to_enu(t: f64, lap_seconds: f64) -> SE3 {
    let w = 2.0 * std::f64::consts::PI / lap_seconds;
    let position = Vector3::new(
        150.0 * (w * t).sin() + 20.0 * (3.0 * w * t).sin(),
        100.0 * (1.0 - (w * t).cos()),
        3.0 * (2.0 * w * t).sin(),
    );
    let velocity = Vector3::new(
        150.0 * w * (w * t).cos() + 60.0 * w * (3.0 * w * t).cos(),
        100.0 * w * (w * t).sin(),
        6.0 * w * (2.0 * w * t).cos(),
    );
    let yaw = velocity.y.atan2(velocity.x);
    let pitch = -(velocity.z / velocity.xy().norm().max(1e-6)).atan();
    let body_to_enu = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
        * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), pitch);
    SE3::new(body_to_enu * body_from_camera(), position)
}

fn seconds_to_timestamp(seconds: f64) -> Timestamp {
    Timestamp::from_nanoseconds((seconds * 1e9).round() as i128)
}

/// Generate a scenario.
pub fn generate_gnss_vo_scenario(config: &SyntheticGnssVoConfig) -> SyntheticGnssVoScenario {
    let mut rng = Rng(config.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    let dt = 1.0 / config.frame_rate_hz;
    let truth: Vec<SE3> = (0..config.frame_count)
        .map(|i| ground_truth_camera_to_enu(i as f64 * dt, config.lap_seconds))
        .collect();

    // True alignment (map → ENU) at frame 0 and the VO start pose.
    let global_scale = config.monocular.as_ref().map_or(1.0, |m| m.global_scale);
    let (true_alignment, vo_start) = if config.gravity_aligned_map {
        let alignment =
            GnssAlignment::from_yaw(config.map_yaw, config.map_translation, 1.0 / global_scale);
        let start = SE3::new(
            alignment.rotation.inverse() * truth[0].rotation,
            alignment.inverse_transform_point(&truth[0].translation),
        );
        (alignment, start)
    } else {
        let alignment = GnssAlignment {
            rotation: truth[0].rotation,
            translation: truth[0].translation,
            scale: 1.0 / global_scale,
        };
        (alignment, SE3::identity())
    };

    let mut vo = Vec::with_capacity(config.frame_count);
    let mut current = vo_start;
    let mut log_scale: f64 = 0.0;
    for (i, pose) in truth.iter().enumerate() {
        if i > 0 {
            let relative = truth[i - 1].inverse().compose(pose);
            // Yaw drift about the camera's up axis (−y), plus noise.
            let drift = Vector3::new(0.0, -config.vo_yaw_drift_per_frame, 0.0)
                + rng.gaussian3(config.vo_rotation_noise);
            let step_scale = global_scale * log_scale.exp() * (1.0 + config.vo_scale_error);
            let noisy = SE3::new(
                relative.rotation * UnitQuaternion::from_scaled_axis(drift),
                (relative.translation + rng.gaussian3(config.vo_translation_noise)) * step_scale,
            );
            current = current.compose(&noisy);
            if let Some(mono) = &config.monocular {
                log_scale += mono.log_scale_drift_per_frame;
            }
        }
        vo.push(Timed::new(
            seconds_to_timestamp(i as f64 * dt),
            Pose {
                world_to_camera: current.inverse(),
            },
        ));
    }

    // GNSS fixes at their own clock.
    let duration = (config.frame_count.saturating_sub(1)) as f64 * dt;
    let gnss_dt = 1.0 / config.gnss_rate_hz;
    let mut fix_times = Vec::new();
    let mut t = config.gnss_time_offset_seconds;
    while t <= duration + 1e-9 {
        let in_dropout = config
            .dropout
            .is_some_and(|(start, length)| t >= start && t < start + length);
        if !in_dropout {
            fix_times.push(t);
        }
        t += gnss_dt;
    }
    let mut offsets = vec![Vector3::zeros(); fix_times.len()];
    let mut outlier_fix_timestamps = Vec::new();
    for outlier in &config.outliers {
        let Some(first) = fix_times
            .iter()
            .enumerate()
            .min_by(|a, b| {
                (a.1 - outlier.time_seconds)
                    .abs()
                    .total_cmp(&(b.1 - outlier.time_seconds).abs())
            })
            .map(|(index, _)| index)
        else {
            continue;
        };
        for index in first..(first + outlier.burst_length.max(1)).min(fix_times.len()) {
            offsets[index] += outlier.offset_enu;
            outlier_fix_timestamps.push(seconds_to_timestamp(fix_times[index]));
        }
    }
    let mut gnss = MeasurementBuffer::new();
    for (time, offset) in fix_times.iter().zip(&offsets) {
        let pose = ground_truth_camera_to_enu(*time, config.lap_seconds);
        let antenna = pose.translation + pose.rotation * config.lever_arm;
        let noise = Vector3::new(
            rng.gaussian() * config.gnss_horizontal_sigma,
            rng.gaussian() * config.gnss_horizontal_sigma,
            rng.gaussian() * config.gnss_vertical_sigma,
        );
        gnss.push(
            GnssMeasurement::new(
                seconds_to_timestamp(*time),
                Point3::from(antenna + noise + offset),
            )
            .with_accuracy(
                Some(config.gnss_horizontal_sigma),
                Some(config.gnss_vertical_sigma),
            ),
        );
    }

    SyntheticGnssVoScenario {
        ground_truth_enu: truth
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                Timed::new(
                    seconds_to_timestamp(i as f64 * dt),
                    Pose {
                        world_to_camera: pose.inverse(),
                    },
                )
            })
            .collect(),
        vo,
        gnss,
        outlier_fix_timestamps,
        true_alignment,
        lever_arm: config.lever_arm,
    }
}
