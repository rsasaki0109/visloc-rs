//! The moving-object filter (`visloc_basalt::dynamic_objects`) on a synthetic
//! monocular-inertial replay.
//!
//! A pinhole camera on an IMU moves through a static point scene, as in
//! `mono_inertial_synthetic.rs`. A rigid "companion" object, for example a
//! person walking next to the camera, stays about 3.5 m in front of it and
//! covers part of the left half of the image. Its points are consistent with
//! one another but not with the static scene, which is the case that pulls a
//! VIO off. The tests compare four runs:
//! - the static scene alone;
//! - with the object;
//! - with the object and its boxes given to the filter;
//! - the static scene with a box over static points only (a "parked car"),
//!   which the filter must keep.

use std::collections::BTreeMap;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_basalt::dynamic_objects::{DynamicObjectParams, DynamicObjectStats, ObjectBox};
use visloc_basalt::{
    config::BasaltConfig, vio_estimator_from_calibration, BasaltCalibration, DoubleSphereCamera,
    ImuSample, TrackObservation,
};
use visloc_core::geometry::SE3;

const WIDTH: u32 = 752;
const HEIGHT: u32 = 480;
const FX: f64 = 460.0;
const CX: f64 = 376.0;
const CY: f64 = 240.0;
const GRAVITY: f64 = 9.81;
const START_NS: i64 = 1_000_000_000;
const IMU_PERIOD_NS: i64 = 5_000_000; // 200 Hz
const IMU_PER_FRAME: i64 = 10; // camera at 20 Hz
const OBJECT_ID_BASE: u64 = 1_000_000;

/// Deterministic xorshift64*.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn gaussian3(&mut self, sigma: f64) -> Vector3<f64> {
        Vector3::new(self.gaussian(), self.gaussian(), self.gaussian()) * sigma
    }
}

/// `f(u) = (1 - cos u)^2 / 4` and its derivatives: the body starts at rest.
fn rest_start_profile(u: f64) -> (f64, f64, f64) {
    let (s, c) = u.sin_cos();
    (
        (1.0 - c).powi(2) / 4.0,
        (1.0 - c) * s / 2.0,
        (s * s + (1.0 - c) * c) / 2.0,
    )
}

const AMPLITUDE: [f64; 3] = [0.6, 1.2, 0.5];
const FREQUENCY: [f64; 3] = [1.3, 1.7, 2.1];
const ANGLE_AMPLITUDE: [f64; 3] = [0.08, 0.10, 0.25];
const ANGLE_FREQUENCY: [f64; 3] = [1.1, 1.5, 0.9];

fn position_velocity_acceleration(t: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let (mut p, mut v, mut a) = (Vector3::zeros(), Vector3::zeros(), Vector3::zeros());
    for axis in 0..3 {
        let w = FREQUENCY[axis];
        let (f, df, ddf) = rest_start_profile(w * t);
        p[axis] = AMPLITUDE[axis] * f;
        v[axis] = AMPLITUDE[axis] * w * df;
        a[axis] = AMPLITUDE[axis] * w * w * ddf;
    }
    (p, v, a)
}

fn rotation(t: f64) -> UnitQuaternion<f64> {
    let mut angles = [0.0; 3];
    for (axis, angle) in angles.iter_mut().enumerate() {
        *angle = ANGLE_AMPLITUDE[axis] * rest_start_profile(ANGLE_FREQUENCY[axis] * t).0;
    }
    UnitQuaternion::from_euler_angles(angles[0], angles[1], angles[2])
}

fn body_rate(t: f64) -> Vector3<f64> {
    let h = 1.0e-5;
    (rotation(t - h).inverse() * rotation(t + h)).scaled_axis() / (2.0 * h)
}

fn camera_to_imu() -> SE3 {
    let columns = Matrix3::from_columns(&[
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(1.0, 0.0, 0.0),
    ]);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(columns));
    SE3::new(rotation, Vector3::new(0.06, -0.01, 0.02))
}

fn calibration() -> BasaltCalibration {
    BasaltCalibration {
        t_imu_cam: vec![camera_to_imu()],
        cameras: vec![DoubleSphereCamera::new(FX, FX, CX, CY, 0.0, 0.0, WIDTH, HEIGHT).unwrap()],
        resolutions: vec![(WIDTH, HEIGHT)],
        calib_accel_bias: vec![0.0; 9],
        calib_gyro_bias: vec![0.0; 12],
        imu_update_rate_hz: 200.0,
        accel_noise_std: Vector3::repeat(0.016),
        gyro_noise_std: Vector3::repeat(0.000282),
        accel_bias_std: Vector3::repeat(0.001),
        gyro_bias_std: Vector3::repeat(0.0001),
        t_mocap_world: SE3::identity(),
        t_imu_marker: SE3::identity(),
        mocap_time_offset_ns: 0,
        mocap_to_imu_offset_ns: 0,
        cam_time_offset_ns: 0,
    }
}

fn project(camera_pose_world: &SE3, point_world: &Point3<f64>) -> Option<Point2<f64>> {
    let p = camera_pose_world.inverse().transform_point(point_world);
    if p.z < 0.3 {
        return None;
    }
    let (u, v) = (FX * p.x / p.z + CX, FX * p.y / p.z + CY);
    let margin = 4.0;
    (u >= margin && v >= margin && u <= f64::from(WIDTH) - 5.0 && v <= f64::from(HEIGHT) - 5.0)
        .then(|| Point2::new(u, v))
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    StaticScene,
    WithObject,
    WithObjectAndBoxes,
    StaticSceneWithParkedBox,
}

struct Outcome {
    rms_error: f64,
    stats: Option<DynamicObjectStats>,
}

/// The companion object's points in the camera frame: a 1.0 m × 1.4 m patch
/// about 3.5 m ahead, left of centre, drifting slowly sideways.
fn object_points_camera(rng: &mut Rng, count: usize) -> Vec<Vector3<f64>> {
    (0..count)
        .map(|_| {
            Vector3::new(
                rng.range(-1.6, -0.6),
                rng.range(-0.7, 0.7),
                rng.range(3.3, 3.7),
            )
        })
        .collect()
}

fn run(variant: Variant, seconds: f64) -> Outcome {
    run_with(variant, seconds, DynamicObjectParams::default())
}

fn run_with(variant: Variant, seconds: f64, params: DynamicObjectParams) -> Outcome {
    let mut estimator = vio_estimator_from_calibration(
        &calibration(),
        &BasaltConfig::from_json(include_str!("../../../configs/basalt/euroc_config.json"))
            .unwrap(),
    )
    .unwrap();
    let mut rng = Rng(0x0b1e_c75e_ed00_0002);
    let t_imu_cam = camera_to_imu();
    let frames = (seconds * 20.0).round() as usize;
    let object = object_points_camera(&mut Rng(77), 40);

    // The scene and the object are generated up front, so every variant
    // sees the same static tracks and the same object.
    let camera_pose = |frame: usize| {
        let t = frame as f64 / 20.0;
        SE3::new(rotation(t), position_velocity_acceleration(t).0).compose(&t_imu_cam)
    };
    let object_world = |frame: usize| -> Vec<Point3<f64>> {
        let drift = Vector3::new(0.15 * frame as f64 / 20.0, 0.0, 0.0);
        let pose = camera_pose(frame);
        object
            .iter()
            .map(|p| pose.transform_point(&Point3::from(p + drift)))
            .collect()
    };

    let with_object = matches!(variant, Variant::WithObject | Variant::WithObjectAndBoxes);
    let mut boxes = BTreeMap::new();
    let mut statics: Vec<(u64, Point3<f64>)> = Vec::new();
    let mut next_id = 0_u64;
    let mut scene_rng = Rng(0x5eed_0000_0000_0042);
    // Separate noise for the object, so the static observations are the same
    // in every variant.
    let mut object_rng = Rng(0x0b1e_c700_0000_0003);
    let mut per_frame: Vec<Vec<TrackObservation>> = Vec::with_capacity(frames);
    for frame in 0..frames {
        let timestamp_ns = START_NS + frame as i64 * IMU_PER_FRAME * IMU_PERIOD_NS;
        let pose = camera_pose(frame);
        statics.retain(|(_, p)| project(&pose, p).is_some() && scene_rng.uniform() >= 0.02);
        if statics.len() < 45 {
            while statics.len() < 60 {
                let (u, v) = (scene_rng.range(20.0, 732.0), scene_rng.range(20.0, 460.0));
                let depth = scene_rng.range(2.5, 8.0);
                let p = Point3::new((u - CX) / FX * depth, (v - CY) / FX * depth, depth);
                statics.push((next_id, pose.transform_point(&p)));
                next_id += 1;
            }
        }
        let mut observations = Vec::new();
        let mut push = |track_id: u64, pixel: Point2<f64>, rng: &mut Rng| {
            observations.push(TrackObservation {
                track_id,
                frame_id: frame as u64,
                timestamp_ns,
                camera_id: 0,
                pixel: Point2::new(
                    pixel.x + 0.5 * rng.gaussian(),
                    pixel.y + 0.5 * rng.gaussian(),
                ),
            });
        };
        for (id, p) in &statics {
            push(*id, project(&pose, p).unwrap(), &mut scene_rng);
        }
        let mut frame_boxes = Vec::new();
        if with_object {
            let pixels: Vec<Point2<f64>> = object_world(frame)
                .iter()
                .filter_map(|p| project(&pose, p))
                .collect();
            for (i, pixel) in pixels.iter().enumerate() {
                push(OBJECT_ID_BASE + i as u64, *pixel, &mut object_rng);
            }
            if variant == Variant::WithObjectAndBoxes && !pixels.is_empty() {
                let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
                for p in &pixels {
                    (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
                }
                frame_boxes.push(ObjectBox {
                    object_id: 1,
                    x_min: x0 - 8.0,
                    y_min: y0 - 8.0,
                    x_max: x1 + 8.0,
                    y_max: y1 + 8.0,
                });
            }
        }
        if variant == Variant::StaticSceneWithParkedBox {
            frame_boxes.push(ObjectBox {
                object_id: 2,
                x_min: 480.0,
                y_min: 120.0,
                x_max: 700.0,
                y_max: 400.0,
            });
        }
        if !frame_boxes.is_empty() {
            boxes.insert(timestamp_ns, frame_boxes);
        }
        per_frame.push(observations);
    }
    if matches!(
        variant,
        Variant::WithObjectAndBoxes | Variant::StaticSceneWithParkedBox
    ) {
        estimator.set_object_boxes(boxes, params);
    }

    let mut next_imu_ns = START_NS;
    let mut squared_errors = 0.0;
    for (frame, observations) in per_frame.iter().enumerate() {
        let timestamp_ns = START_NS + frame as i64 * IMU_PER_FRAME * IMU_PERIOD_NS;
        let mut imu = Vec::new();
        while next_imu_ns <= timestamp_ns {
            let t = (next_imu_ns - START_NS) as f64 * 1e-9;
            let (_, _, acceleration) = position_velocity_acceleration(t);
            let specific_force =
                rotation(t).inverse() * (acceleration + Vector3::new(0.0, 0.0, GRAVITY));
            imu.push(ImuSample::new(
                next_imu_ns,
                body_rate(t) + Vector3::new(0.002, -0.0015, 0.001) + rng.gaussian3(0.002),
                specific_force + Vector3::new(0.04, -0.03, 0.05) + rng.gaussian3(0.02),
            ));
            next_imu_ns += IMU_PERIOD_NS;
        }
        let output = estimator
            .process_without_marg_data(frame as u64, timestamp_ns, observations, &imu)
            .unwrap_or_else(|error| panic!("frame {frame}: {error}"));
        let truth = position_velocity_acceleration(frame as f64 / 20.0).0;
        squared_errors += (output.state.imu_to_world.translation - truth).norm_squared();
    }
    Outcome {
        rms_error: (squared_errors / frames as f64).sqrt(),
        stats: estimator.dynamic_object_stats(),
    }
}

/// Runs in a debug build: the filter finds the companion object within the
/// first second and leaves a box over static points alone.
#[test]
fn the_filter_drops_a_moving_object_and_keeps_a_parked_one() {
    let moving = run(Variant::WithObjectAndBoxes, 1.0);
    let stats = moving.stats.expect("boxes were set");
    eprintln!("moving object, 1 s: {stats:?}");
    assert!(
        stats.moving_boxes > 0 && stats.dropped_observations > 0,
        "{stats:?}"
    );

    let parked = run(Variant::StaticSceneWithParkedBox, 1.0);
    let stats = parked.stats.expect("boxes were set");
    eprintln!("parked box, 1 s: {stats:?}");
    assert_eq!(
        (stats.moving_boxes, stats.dropped_observations),
        (0, 0),
        "{stats:?}"
    );
}

#[test]
#[ignore = "release-only: run `cargo test --release -p visloc-basalt --test dynamic_objects_synthetic -- --include-ignored --nocapture`"]
fn dropping_the_moving_object_restores_the_static_scene_accuracy() {
    let seconds = 6.0;
    let clean = run(Variant::StaticScene, seconds);
    let polluted = run(Variant::WithObject, seconds);
    let filtered = run(Variant::WithObjectAndBoxes, seconds);
    let parked = run(Variant::StaticSceneWithParkedBox, seconds);
    eprintln!(
        "rms position error: static {:.4} m, with object {:.4} m, object + filter {:.4} m, \
         parked box {:.4} m; filter {:?}",
        clean.rms_error, polluted.rms_error, filtered.rms_error, parked.rms_error, filtered.stats
    );
    // About 75 times worse with the object. With the filter, within a factor
    // 2 of the static scene: the object is used in the first frames, before
    // it has moved enough to show, and leaving it out changes the keyframes.
    assert!(
        polluted.rms_error > 10.0 * clean.rms_error,
        "the object should hurt"
    );
    assert!(
        filtered.rms_error < 2.0 * clean.rms_error,
        "the filter should undo it"
    );
    assert_eq!(
        parked.rms_error, clean.rms_error,
        "a static box changes nothing"
    );
}
