//! Synthetic divergent two-camera inertial replay (Project Aria geometry)
//! through the Basalt VIO estimator.
//!
//! The rig mirrors the two SLAM cameras of Project Aria glasses as used by
//! the LaMAria benchmark: ideal pinhole cameras (Double Sphere with
//! `xi = alpha = 0`), 758x572, fx = 241.6 (about 115 degrees across the long
//! image axis), whose optical axes diverge by 75 degrees about their common
//! image x axis, with a 0.138 m baseline.  As on Aria the images are "rotated":
//! the long image axis is vertical, so the rig looks front-left (cam0) and
//! front-right (cam1) and the two views share only a ~25 degree band.
//!
//! A closed tiled room provides texture.  In the blank scenario every surface
//! on the left of the room loses its texture for a few seconds, which is the
//! synthetic equivalent of cam0 facing a blank wall: cam0 keeps only a thin
//! textured strip of the shared band while cam1 still sees texture.
//!
//! Three configurations are compared on identical data (same seed):
//!
//! * (a) `Cam0Only`: the pinned config (cam0 detection, same-pixel stereo
//!   seed, cam0-hosted landmarks, cam0 keyframe connectivity);
//! * (b) `StereoSeed`: (a) plus `REPROJ_FIX_DEPTH` stereo seeding;
//! * (c) `MultiCamera`: (b) plus cam1 detection, cam1-hosted landmarks and
//!   all-camera keyframe connectivity.
//!
//! `multicam_inertial_vio_smoke` runs by default (estimator only, ideal
//! tracks, sized for an unoptimized build), as do the rig geometry check and
//! a one-frame rendered stereo-match accuracy check.  The rendered scenarios run
//! the real KLT frontend through `BasaltVioEstimatorAdapter` and are
//! `#[ignore]`d; run them with
//! `cargo test --release -p visloc-basalt --test multicam_inertial_synthetic -- --include-ignored --nocapture`.

use std::path::PathBuf;

use nalgebra::{Matrix3, Point2, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_basalt::{
    config::BasaltConfig, vio::EstimatorOutput, vio_estimator_from_calibration, BasaltCalibration,
    BasaltVioEstimatorAdapter, DoubleSphereCamera, EurocSensorFrame, ImuSample, RawU16Image,
    TrackObservation,
};
use visloc_core::geometry::SE3;

const WIDTH: u32 = 758;
const HEIGHT: u32 = 572;
const FX: f64 = 241.6;
const CX: f64 = 378.5;
const CY: f64 = 285.5;
/// Half the 75 degree divergence: cam0 yaws left, cam1 right.
const HALF_DIVERGENCE_DEG: f64 = 37.5;
const HALF_BASELINE_M: f64 = 0.069;
const GRAVITY: f64 = 9.81;
const START_NS: i64 = 1_000_000_000;
const IMU_PERIOD_NS: i64 = 5_000_000; // 200 Hz
const IMU_PER_FRAME: i64 = 10; // cameras at 20 Hz

/// Deterministic xorshift64* generator; no extra dev-dependency needed.
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

/// `f(u) = (1 - cos u)^2 / 4`: f, f' and f'' vanish at u = 0, so the body
/// starts at rest and without acceleration.
fn rest_start_profile(u: f64) -> (f64, f64, f64) {
    let (s, c) = u.sin_cos();
    let value = (1.0 - c).powi(2) / 4.0;
    let first = (1.0 - c) * s / 2.0;
    let second = (s * s + (1.0 - c) * c) / 2.0;
    (value, first, second)
}

#[derive(Clone, Copy)]
struct Motion {
    amplitude: Vector3<f64>,
    frequency: Vector3<f64>,
    angle_amplitude: Vector3<f64>,
    angle_frequency: Vector3<f64>,
}

impl Motion {
    fn position_velocity_acceleration(&self, t: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let mut p = Vector3::zeros();
        let mut v = Vector3::zeros();
        let mut a = Vector3::zeros();
        for axis in 0..3 {
            let w = self.frequency[axis];
            let (f, df, ddf) = rest_start_profile(w * t);
            p[axis] = self.amplitude[axis] * f;
            v[axis] = self.amplitude[axis] * w * df;
            a[axis] = self.amplitude[axis] * w * w * ddf;
        }
        (p, v, a)
    }

    fn rotation(&self, t: f64) -> UnitQuaternion<f64> {
        let mut angles = [0.0; 3];
        for (axis, angle) in angles.iter_mut().enumerate() {
            let (f, _, _) = rest_start_profile(self.angle_frequency[axis] * t);
            *angle = self.angle_amplitude[axis] * f;
        }
        UnitQuaternion::from_euler_angles(angles[0], angles[1], angles[2])
    }

    fn body_rate(&self, t: f64) -> Vector3<f64> {
        let h = 1.0e-5;
        let delta = self.rotation(t - h).inverse() * self.rotation(t + h);
        delta.scaled_axis() / (2.0 * h)
    }

    fn imu_pose(&self, t: f64) -> SE3 {
        SE3::new(self.rotation(t), self.position_velocity_acceleration(t).0)
    }
}

fn camera() -> DoubleSphereCamera {
    DoubleSphereCamera::new(FX, FX, CX, CY, 0.0, 0.0, WIDTH, HEIGHT).unwrap()
}

/// `T_imu_cam` of camera `index`: the optical axis lies in the IMU x-y plane
/// at yaw `+37.5` (cam0) or `-37.5` (cam1) degrees from IMU +x (forward), the
/// image x axis points along IMU -z (the long image axis is vertical, as on
/// Aria), and the cameras sit 0.138 m apart along IMU y.  The relative
/// rotation is therefore 75 degrees about the shared image x axis.
fn camera_to_imu(index: usize) -> SE3 {
    let sign = if index == 0 { 1.0 } else { -1.0 };
    let yaw = sign * HALF_DIVERGENCE_DEG.to_radians();
    let z = Vector3::new(yaw.cos(), yaw.sin(), 0.0);
    let x = Vector3::new(0.0, 0.0, -1.0);
    let y = z.cross(&x);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        Matrix3::from_columns(&[x, y, z]),
    ));
    SE3::new(rotation, Vector3::new(0.0, sign * HALF_BASELINE_M, 0.0))
}

fn calibration() -> BasaltCalibration {
    BasaltCalibration {
        t_imu_cam: vec![camera_to_imu(0), camera_to_imu(1)],
        cameras: vec![camera(), camera()],
        resolutions: vec![(WIDTH, HEIGHT); 2],
        calib_accel_bias: vec![0.0; 9],
        calib_gyro_bias: vec![0.0; 12],
        imu_update_rate_hz: 200.0,
        // Continuous-time densities from Basalt's EuRoC calibration.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// (a) pinned config: cam0-centric frontend and estimator.
    Cam0Only,
    /// (b) (a) plus the `REPROJ_FIX_DEPTH` stereo seed.
    StereoSeed,
    /// (c) (b) plus cam1 detection, cam1-hosted landmarks and all-camera
    /// keyframe connectivity.
    MultiCamera,
}

impl Variant {
    const ALL: [Self; 3] = [Self::Cam0Only, Self::StereoSeed, Self::MultiCamera];

    fn config(self) -> BasaltConfig {
        let mut config =
            BasaltConfig::from_json(include_str!("../../../configs/basalt/euroc_config.json"))
                .unwrap();
        let mut set = |key: &str, value: serde_json::Value| {
            config.values.insert(key.to_owned(), value);
        };
        if self != Self::Cam0Only {
            set(
                "config.optical_flow_matching_guess_type",
                serde_json::json!("REPROJ_FIX_DEPTH"),
            );
            set(
                "config.optical_flow_matching_default_depth",
                serde_json::json!(2.0),
            );
        }
        if self == Self::MultiCamera {
            set(
                "config.optical_flow_detect_all_cameras",
                serde_json::json!(true),
            );
            set("config.vio_landmarks_all_cameras", serde_json::json!(true));
            set(
                "config.vio_kf_connectivity_all_cameras",
                serde_json::json!(true),
            );
        }
        // Round-trip through the strict parser: every key must be accepted.
        let json = serde_json::json!({ "value0": config.values }).to_string();
        BasaltConfig::from_json(&json).unwrap()
    }
}

/// Closed tiled room around the start pose.
const ROOM_MIN: [f64; 3] = [-4.0, -5.0, -1.6];
const ROOM_MAX: [f64; 3] = [7.0, 5.0, 2.4];
const TILE: f64 = 0.2;
/// While blanking is active, every surface point with world `y` above this
/// (the left of the room, which cam0 faces) is a uniform grey.
const BLANK_Y: f64 = -0.8;
const BLANK_LEVEL: f64 = 128.0;

#[derive(Clone, Copy)]
struct Blank {
    start_s: f64,
    end_s: f64,
}

impl Blank {
    fn active(blank: Option<Self>, t: f64) -> bool {
        blank.is_some_and(|blank| t >= blank.start_s && t < blank.end_s)
    }
}

fn tile_intensity(face: u64, a: f64, b: f64) -> f64 {
    let ia = (a / TILE).floor() as i64 as u64;
    let ib = (b / TILE).floor() as i64 as u64;
    let mut hash = face
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(ia.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(ib.wrapping_mul(0x94d0_49bb_1331_11eb));
    hash ^= hash >> 31;
    hash = hash.wrapping_mul(0xd6e8_feb8_6659_fd93);
    hash ^= hash >> 29;
    30.0 + 195.0 * ((hash >> 11) as f64 / (1u64 << 53) as f64)
}

/// The first room surface hit along a ray, and its radiance.
fn room_hit(
    origin: &Vector3<f64>,
    direction: &Vector3<f64>,
    blank: bool,
) -> Option<(Vector3<f64>, f64)> {
    let mut best: Option<(f64, Vector3<f64>, f64)> = None;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-12 {
            continue;
        }
        let bound = if direction[axis] > 0.0 {
            ROOM_MAX[axis]
        } else {
            ROOM_MIN[axis]
        };
        let distance = (bound - origin[axis]) / direction[axis];
        if distance > 0.0 && best.is_none_or(|(current, _, _)| distance < current) {
            let hit = origin + direction * distance;
            let (a, b) = match axis {
                0 => (hit.y, hit.z),
                1 => (hit.x, hit.z),
                _ => (hit.x, hit.y),
            };
            let face = axis as u64 * 2 + u64::from(direction[axis] > 0.0);
            best = Some((distance, hit, tile_intensity(face, a, b)));
        }
    }
    best.map(|(_, hit, radiance)| {
        if blank && hit.y > BLANK_Y {
            (hit, BLANK_LEVEL)
        } else {
            (hit, radiance)
        }
    })
}

/// Renders one camera with 2x2 supersampling, in the 8-bit-in-u16 layout the
/// EuRoC reader produces.
fn render(camera_pose_world: &SE3, blank: bool) -> RawU16Image {
    let rotation = camera_pose_world.rotation.to_rotation_matrix();
    let origin = camera_pose_world.translation;
    RawU16Image::from_fn(WIDTH as usize, HEIGHT as usize, |x, y| {
        let mut sum = 0.0;
        for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
            let u = x as f64 + sx - 0.5;
            let v = y as f64 + sy - 0.5;
            let ray = rotation * Vector3::new((u - CX) / FX, (v - CY) / FX, 1.0);
            sum += room_hit(&origin, &ray, blank).map_or(0.0, |(_, radiance)| radiance);
        }
        ((sum / 4.0).round().clamp(0.0, 255.0) as u16) << 8
    })
    .unwrap()
}

fn project(camera_pose_world: &SE3, point_world: &Point3<f64>) -> Option<Point2<f64>> {
    let point_camera = camera_pose_world.inverse().transform_point(point_world);
    if point_camera.z < 0.3 {
        return None;
    }
    let u = FX * point_camera.x / point_camera.z + CX;
    let v = FX * point_camera.y / point_camera.z + CY;
    let margin = 20.0;
    (u >= margin
        && v >= margin
        && u <= f64::from(WIDTH) - 1.0 - margin
        && v <= f64::from(HEIGHT) - 1.0 - margin)
        .then(|| Point2::new(u, v))
}

struct Scenario {
    motion: Motion,
    seconds: f64,
    blank: Option<Blank>,
    accel_bias: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    accel_noise: f64,
    gyro_noise: f64,
    pixel_noise: f64,
    track_drop_probability: f64,
    target_tracks: usize,
    replenish_below: usize,
    seed: u64,
}

fn imu_sample(scenario: &Scenario, rng: &mut Rng, timestamp_ns: i64) -> ImuSample {
    let t = (timestamp_ns - START_NS) as f64 * 1e-9;
    let (_, _, acceleration) = scenario.motion.position_velocity_acceleration(t);
    let rotation = scenario.motion.rotation(t);
    let specific_force = rotation.inverse() * (acceleration + Vector3::new(0.0, 0.0, GRAVITY));
    ImuSample::new(
        timestamp_ns,
        scenario.motion.body_rate(t) + scenario.gyro_bias + rng.gaussian3(scenario.gyro_noise),
        specific_force + scenario.accel_bias + rng.gaussian3(scenario.accel_noise),
    )
}

/// One camera event handed to a frame source.
struct FrameEvent {
    frame: usize,
    timestamp_ns: i64,
    t: f64,
    camera_poses: [SE3; 2],
    imu: Vec<ImuSample>,
    initialization_imu: Option<ImuSample>,
}

/// What a source reports per frame besides the estimator output.
struct SourceFrame {
    estimator: EstimatorOutput,
    cam0_observations: usize,
    cam1_observations: usize,
    cam1_hosted_landmarks: usize,
}

/// Ideal-frontend track: a world point with independent per-camera
/// lifetimes, as the KLT frontend keeps one observation map per camera.
struct IdealTrack {
    id: u64,
    point_world: Point3<f64>,
    alive: [bool; 2],
}

/// Estimator-only source with an ideal frontend.  `multi_camera` selects
/// what the multi-camera frontend would deliver (cam0-born tracks also
/// observed in cam1 where visible, plus cam1-born tracks); otherwise only
/// cam0 tracks exist, as with the pinned frontend whose same-pixel stereo
/// seed fails on this rig.
fn ideal_track_source(
    scenario: &Scenario,
    variant: Variant,
) -> impl FnMut(&FrameEvent, &mut Rng) -> SourceFrame {
    let config = variant.config();
    let mut estimator = vio_estimator_from_calibration(&calibration(), &config).unwrap();
    let multi_camera = variant == Variant::MultiCamera;
    let mut tracks: Vec<IdealTrack> = Vec::new();
    let mut next_id = 0_u64;
    let blank = scenario.blank;
    let pixel_noise = scenario.pixel_noise;
    let drop_probability = scenario.track_drop_probability;
    let (target, replenish_below) = (scenario.target_tracks, scenario.replenish_below);
    move |event, rng| {
        let blanked = Blank::active(blank, event.t);
        let textured = |point: &Point3<f64>| !(blanked && point.y > BLANK_Y);
        for track in &mut tracks {
            for camera in 0..2 {
                if track.alive[camera]
                    && (project(&event.camera_poses[camera], &track.point_world).is_none()
                        || !textured(&track.point_world)
                        || rng.uniform() < drop_probability)
                {
                    track.alive[camera] = false;
                }
            }
        }
        tracks.retain(|track| track.alive[0] || track.alive[1]);
        let cameras: &[usize] = if multi_camera { &[0, 1] } else { &[0] };
        for &camera in cameras {
            let alive = tracks.iter().filter(|track| track.alive[camera]).count();
            if alive >= replenish_below {
                continue;
            }
            let mut needed = target - alive;
            let mut attempts = 0;
            while needed > 0 && attempts < 20 * target {
                attempts += 1;
                let pixel = Point2::new(
                    rng.range(30.0, f64::from(WIDTH) - 30.0),
                    rng.range(30.0, f64::from(HEIGHT) - 30.0),
                );
                // Points lie on the room surfaces, as rendered.
                let pose = &event.camera_poses[camera];
                let ray =
                    pose.rotation * Vector3::new((pixel.x - CX) / FX, (pixel.y - CY) / FX, 1.0);
                let Some((hit, _)) = room_hit(&pose.translation, &ray, false) else {
                    continue;
                };
                let point_world = Point3::from(hit);
                if !textured(&point_world) {
                    continue;
                }
                // A new cam0 point is also matched into cam1 where visible
                // (seeded stereo); cam1-born points stay cam1-only.
                let in_cam1 = camera == 1
                    || (multi_camera && project(&event.camera_poses[1], &point_world).is_some());
                tracks.push(IdealTrack {
                    id: next_id,
                    point_world,
                    alive: [camera == 0, in_cam1],
                });
                next_id += 1;
                needed -= 1;
            }
        }
        let mut observations = Vec::new();
        let mut counts = [0_usize; 2];
        for track in &tracks {
            for camera in 0..2 {
                if !track.alive[camera] {
                    continue;
                }
                let pixel = project(&event.camera_poses[camera], &track.point_world).unwrap();
                counts[camera] += 1;
                observations.push(TrackObservation {
                    track_id: track.id,
                    frame_id: event.frame as u64,
                    timestamp_ns: event.timestamp_ns,
                    camera_id: camera as u16,
                    pixel: Point2::new(
                        pixel.x + pixel_noise * rng.gaussian(),
                        pixel.y + pixel_noise * rng.gaussian(),
                    ),
                });
            }
        }
        let output = estimator
            .process_without_marg_data(
                event.frame as u64,
                event.timestamp_ns,
                &observations,
                &event.imu,
            )
            .unwrap_or_else(|error| panic!("frame {}: {error}", event.frame));
        SourceFrame {
            estimator: output,
            cam0_observations: counts[0],
            cam1_observations: counts[1],
            cam1_hosted_landmarks: estimator
                .landmark_count_by_host_camera()
                .get(1)
                .copied()
                .unwrap_or(0),
        }
    }
}

/// Real pipeline: renders both cameras and runs the direct KLT frontend plus
/// the estimator through [`BasaltVioEstimatorAdapter`] with the variant's
/// config.
fn rendered_adapter_source(
    scenario: &Scenario,
    variant: Variant,
) -> impl FnMut(&FrameEvent, &mut Rng) -> SourceFrame {
    let mut adapter =
        BasaltVioEstimatorAdapter::from_config(&calibration(), &variant.config()).unwrap();
    let blank = scenario.blank;
    move |event, _rng| {
        let blanked = Blank::active(blank, event.t);
        let (cam0, cam1) = rayon::join(
            || render(&event.camera_poses[0], blanked),
            || render(&event.camera_poses[1], blanked),
        );
        let frame = EurocSensorFrame {
            frame_id: event.frame as u64,
            timestamp_ns: event.timestamp_ns,
            cam0,
            cam1: Some(cam1),
            cam0_path: PathBuf::new(),
            cam1_path: Some(PathBuf::new()),
            imu: event.imu.clone(),
            initialization_imu: event.initialization_imu,
        };
        let output = adapter
            .process_without_marg_data(frame)
            .unwrap_or_else(|error| panic!("frame {}: {error}", event.frame));
        let count = |camera: u16| {
            output
                .tracks
                .observations
                .iter()
                .filter(|observation| observation.camera_id == camera)
                .count()
        };
        SourceFrame {
            cam0_observations: count(0),
            cam1_observations: count(1),
            cam1_hosted_landmarks: adapter
                .estimator
                .landmark_count_by_host_camera()
                .get(1)
                .copied()
                .unwrap_or(0),
            estimator: output.estimator,
        }
    }
}

struct RunSummary {
    frames: usize,
    keyframes: usize,
    position_errors: Vec<f64>,
    path_length: f64,
    displacement_scale: f64,
    max_cam1_hosted_landmarks: usize,
    mean_cam0_observations: f64,
    mean_cam1_observations: f64,
    /// Frames inside the blank interval (empty without one).
    blank_frames: Vec<usize>,
    /// Frames inside the blank interval with no visual factor row.
    blank_frames_without_vision: usize,
    min_blank_visual_rows: Option<usize>,
}

impl RunSummary {
    fn rms(&self) -> f64 {
        rms(&self.position_errors)
    }

    fn max(&self) -> f64 {
        self.position_errors.iter().copied().fold(0.0, f64::max)
    }

    /// Largest error from the blank start to the end of the run: the drift
    /// accumulated while cam0 had no texture, and how far it got undone.
    fn max_from_blank(&self) -> f64 {
        self.blank_frames.first().map_or(0.0, |&first| {
            self.position_errors[first..]
                .iter()
                .copied()
                .fold(0.0, f64::max)
        })
    }
}

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn run_with<F>(scenario: &Scenario, mut source: F) -> RunSummary
where
    F: FnMut(&FrameEvent, &mut Rng) -> SourceFrame,
{
    let t_imu_cam = [camera_to_imu(0), camera_to_imu(1)];
    let mut rng = Rng(scenario.seed);
    let frames = (scenario.seconds * 20.0).round() as usize;
    let mut summary = RunSummary {
        frames,
        keyframes: 0,
        position_errors: Vec::with_capacity(frames),
        path_length: 0.0,
        displacement_scale: f64::NAN,
        max_cam1_hosted_landmarks: 0,
        mean_cam0_observations: 0.0,
        mean_cam1_observations: 0.0,
        blank_frames: Vec::new(),
        blank_frames_without_vision: 0,
        min_blank_visual_rows: None,
    };
    let mut estimated: Vec<Vector3<f64>> = Vec::with_capacity(frames);
    let mut truth: Vec<Vector3<f64>> = Vec::with_capacity(frames);
    let mut next_imu_ns = START_NS;
    for frame in 0..frames {
        let timestamp_ns = START_NS + frame as i64 * IMU_PER_FRAME * IMU_PERIOD_NS;
        let t = (timestamp_ns - START_NS) as f64 * 1e-9;
        let mut imu = Vec::new();
        while next_imu_ns <= timestamp_ns {
            imu.push(imu_sample(scenario, &mut rng, next_imu_ns));
            next_imu_ns += IMU_PERIOD_NS;
        }
        let imu_pose = scenario.motion.imu_pose(t);
        let event = FrameEvent {
            frame,
            timestamp_ns,
            t,
            camera_poses: [
                imu_pose.compose(&t_imu_cam[0]),
                imu_pose.compose(&t_imu_cam[1]),
            ],
            initialization_imu: (frame == 0).then(|| *imu.last().unwrap()),
            imu,
        };
        let output = source(&event, &mut rng);
        let estimator = &output.estimator;
        summary.keyframes += usize::from(estimator.is_keyframe);
        summary.max_cam1_hosted_landmarks = summary
            .max_cam1_hosted_landmarks
            .max(output.cam1_hosted_landmarks);
        summary.mean_cam0_observations += output.cam0_observations as f64 / frames as f64;
        summary.mean_cam1_observations += output.cam1_observations as f64 / frames as f64;
        if Blank::active(scenario.blank, t) {
            summary.blank_frames.push(frame);
            let rows = estimator.window.visual_factor_rows;
            summary.blank_frames_without_vision += usize::from(rows == 0);
            summary.min_blank_visual_rows = Some(
                summary
                    .min_blank_visual_rows
                    .map_or(rows, |min| min.min(rows)),
            );
        }
        let (true_position, _, _) = scenario.motion.position_velocity_acceleration(t);
        let estimated_position = estimator.state.imu_to_world.translation;
        assert!(estimated_position.iter().all(|value| value.is_finite()));
        summary
            .position_errors
            .push((estimated_position - true_position).norm());
        if let Some(previous) = truth.last() {
            summary.path_length += (true_position - previous).norm();
        }
        estimated.push(estimated_position);
        truth.push(true_position);
    }
    let (mut numerator, mut denominator) = (0.0, 0.0);
    for (estimate, reference) in estimated.iter().zip(&truth) {
        let estimate_delta = estimate - estimated[0];
        let reference_delta = reference - truth[0];
        numerator += estimate_delta.dot(&reference_delta);
        denominator += reference_delta.norm_squared();
    }
    summary.displacement_scale = numerator / denominator;
    summary
}

fn report(name: &str, variant: Variant, summary: &RunSummary) {
    eprintln!(
        "{name} {variant:?}: frames={} keyframes={} path={:.2} m rms_err={:.4} m max_err={:.4} m \
         max_err_from_blank={:.4} m scale={:.4} cam1_hosted_landmarks_max={} \
         mean_obs cam0={:.1} cam1={:.1} blank_frames={} blank_frames_without_vision={} \
         min_blank_visual_rows={:?}",
        summary.frames,
        summary.keyframes,
        summary.path_length,
        summary.rms(),
        summary.max(),
        summary.max_from_blank(),
        summary.displacement_scale,
        summary.max_cam1_hosted_landmarks,
        summary.mean_cam0_observations,
        summary.mean_cam1_observations,
        summary.blank_frames.len(),
        summary.blank_frames_without_vision,
        summary.min_blank_visual_rows,
    );
}

fn moving_scenario(seed: u64) -> Scenario {
    Scenario {
        motion: Motion {
            amplitude: Vector3::new(0.8, 1.0, 0.4),
            frequency: Vector3::new(1.3, 1.7, 2.1),
            angle_amplitude: Vector3::new(0.08, 0.10, 0.25),
            angle_frequency: Vector3::new(1.1, 1.5, 0.9),
        },
        seconds: 8.0,
        blank: None,
        accel_bias: Vector3::new(0.04, -0.03, 0.05),
        gyro_bias: Vector3::new(0.002, -0.0015, 0.001),
        accel_noise: 0.02,
        gyro_noise: 0.002,
        pixel_noise: 0.5,
        track_drop_probability: 0.02,
        target_tracks: 100,
        replenish_below: 75,
        seed,
    }
}

/// Geometry sanity: the rig diverges by 75 degrees with a 0.138 m baseline,
/// the two views share a band, and much of each view is exclusive.
#[test]
fn divergent_rig_matches_aria_geometry() {
    let (cam0, cam1) = (camera_to_imu(0), camera_to_imu(1));
    let relative = cam0.inverse().compose(&cam1);
    assert!((relative.rotation.angle().to_degrees() - 75.0).abs() < 1e-9);
    assert!((relative.translation.norm() - 0.138).abs() < 1e-9);
    // The rotation axis is the shared image x axis.
    let axis = relative.rotation.axis().unwrap();
    assert!(axis.x.abs() > 1.0 - 1e-9);
    let hfov = (f64::from(WIDTH) / 2.0 / FX).atan().to_degrees() * 2.0;
    assert!(hfov > 114.0 && hfov < 116.0, "hfov {hfov}");
    // Fraction of cam1 pixels whose ray (at 5 m) also lands in cam0.
    let pose0 = camera_to_imu(0);
    let pose1 = camera_to_imu(1);
    let (mut shared, mut total) = (0, 0);
    for v in (0..HEIGHT).step_by(13) {
        for u in (0..WIDTH).step_by(13) {
            let ray = Vector3::new((f64::from(u) - CX) / FX, (f64::from(v) - CY) / FX, 1.0);
            let point = pose1.transform_point(&Point3::from(ray.normalize() * 5.0));
            total += 1;
            shared += usize::from(project(&pose0, &point).is_some());
        }
    }
    let fraction = shared as f64 / total as f64;
    assert!(
        fraction > 0.05 && fraction < 0.35,
        "shared fraction {fraction}"
    );
}

/// Always-on smoke version (estimator only, ideal tracks), sized for an
/// unoptimized build: cam0 loses texture after 1.0 s.  The cam0-centric
/// configuration falls back to dead reckoning while the multi-camera one
/// keeps cam1-hosted landmarks in the window.
#[test]
fn multicam_inertial_vio_smoke() {
    let mut scenario = moving_scenario(0x5eed_aa55_0000_0001);
    scenario.seconds = 2.5;
    scenario.target_tracks = 30;
    scenario.replenish_below = 22;
    scenario.blank = Some(Blank {
        start_s: 1.0,
        end_s: 2.5,
    });
    let pinned = run_with(&scenario, ideal_track_source(&scenario, Variant::Cam0Only));
    report("multicam_smoke", Variant::Cam0Only, &pinned);
    let multi = run_with(
        &scenario,
        ideal_track_source(&scenario, Variant::MultiCamera),
    );
    report("multicam_smoke", Variant::MultiCamera, &multi);

    assert_eq!(pinned.max_cam1_hosted_landmarks, 0);
    assert!(multi.max_cam1_hosted_landmarks > 0);
    assert_eq!(multi.blank_frames_without_vision, 0);
    assert!(pinned.blank_frames_without_vision > 0);
    assert!(multi.rms() < 0.05, "multi-camera rms {}", multi.rms());
    assert!(
        multi.max_from_blank() <= pinned.max_from_blank(),
        "multi {} vs pinned {}",
        multi.max_from_blank(),
        pinned.max_from_blank()
    );
}

/// Frontend-only check on one rendered stereo frame: with the pinned
/// same-pixel seed almost no new cam0 keypoint finds its cam1 match on this
/// rig, while the reprojection seed recovers many, and those land on the true
/// correspondence (the cam0 pixel's surface point projected into cam1).
#[test]
fn reprojection_seed_finds_true_stereo_matches_on_rendered_frame() {
    let imu_pose = SE3::identity();
    let poses = [
        imu_pose.compose(&camera_to_imu(0)),
        imu_pose.compose(&camera_to_imu(1)),
    ];
    let (cam0, cam1) = rayon::join(|| render(&poses[0], false), || render(&poses[1], false));
    let stats = |variant: Variant| {
        let config = variant.config();
        let direct = visloc_basalt::direct_klt_config(&config).unwrap();
        let mut stream = visloc_basalt::DirectKltStream::new(calibration(), direct)
            .unwrap()
            .with_multi_camera_options(config.multi_camera_flow_options().unwrap())
            .unwrap();
        let output = stream
            .process_frame(visloc_basalt::StereoFrame::new(
                0,
                START_NS,
                cam0.clone(),
                Some(cam1.clone()),
            ))
            .unwrap();
        let mut errors = Vec::new();
        for observation in output.observations.iter().filter(|o| o.camera_id == 1) {
            let Some(cam0_observation) = output
                .observations
                .iter()
                .find(|o| o.camera_id == 0 && o.track_id == observation.track_id)
            else {
                continue;
            };
            let pixel = cam0_observation.pixel;
            let ray =
                poses[0].rotation * Vector3::new((pixel.x - CX) / FX, (pixel.y - CY) / FX, 1.0);
            let (hit, _) = room_hit(&poses[0].translation, &ray, false).unwrap();
            let point_cam1 = poses[1].inverse().transform_point(&Point3::from(hit));
            let expected = Point2::new(
                FX * point_cam1.x / point_cam1.z + CX,
                FX * point_cam1.y / point_cam1.z + CY,
            );
            errors.push((observation.pixel - expected).norm());
        }
        errors.sort_by(f64::total_cmp);
        errors
    };
    let pinned = stats(Variant::Cam0Only);
    let seeded = stats(Variant::StereoSeed);
    let median = |errors: &[f64]| errors.get(errors.len() / 2).copied().unwrap_or(f64::NAN);
    let within_1px = |errors: &[f64]| errors.iter().filter(|error| **error < 1.0).count();
    eprintln!(
        "stereo matches on frame 0: same-pixel {} (median err {:.2} px, {} within 1 px); \
         reprojection seed {} (median err {:.3} px, {} within 1 px, max {:.2} px)",
        pinned.len(),
        median(&pinned),
        within_1px(&pinned),
        seeded.len(),
        median(&seeded),
        within_1px(&seeded),
        seeded.last().copied().unwrap_or(f64::NAN),
    );
    assert!(within_1px(&pinned) <= 2, "same-pixel matches {pinned:?}");
    assert!(seeded.len() >= 10, "only {} seeded matches", seeded.len());
    assert!(within_1px(&seeded) * 10 >= seeded.len() * 9);
    assert!(
        median(&seeded) < 0.5,
        "median seeded error {}",
        median(&seeded)
    );
}

fn rendered_comparison(name: &str, scenario: &Scenario) -> Vec<(Variant, RunSummary)> {
    Variant::ALL
        .into_iter()
        .map(|variant| {
            let summary = run_with(scenario, rendered_adapter_source(scenario, variant));
            report(name, variant, &summary);
            (variant, summary)
        })
        .collect()
}

#[test]
#[ignore = "release-only: renders two 758x572 cameras per frame; run `cargo test --release -p visloc-basalt --test multicam_inertial_synthetic -- --include-ignored`"]
fn multicam_rendered_textured_room_compares_variants() {
    let scenario = moving_scenario(0x5eed_aa55_0000_0002);
    let results = rendered_comparison("multicam_rendered_textured", &scenario);
    let get = |variant| &results.iter().find(|(v, _)| *v == variant).unwrap().1;
    let (pinned, seeded, multi) = (
        get(Variant::Cam0Only),
        get(Variant::StereoSeed),
        get(Variant::MultiCamera),
    );
    // The reprojection seed recovers stereo matches on the divergent rig.
    assert!(
        seeded.mean_cam1_observations > 2.0 * pinned.mean_cam1_observations.max(1.0),
        "seeded cam1 obs {} vs pinned {}",
        seeded.mean_cam1_observations,
        pinned.mean_cam1_observations
    );
    assert_eq!(pinned.max_cam1_hosted_landmarks, 0);
    assert_eq!(seeded.max_cam1_hosted_landmarks, 0);
    assert!(multi.max_cam1_hosted_landmarks > 10);
    assert!(multi.mean_cam1_observations > seeded.mean_cam1_observations);
    // Warped-seed stereo matches are accurate enough not to bias the solve
    // (observed: 2.8 mm and 3.3 mm RMS vs 7.1 mm for the pinned config).
    assert!(seeded.rms() < 0.02, "seeded rms {}", seeded.rms());
    assert!(multi.rms() < 0.02, "multi rms {}", multi.rms());
    for (variant, summary) in &results {
        assert!(summary.rms() < 0.10, "{variant:?} rms {}", summary.rms());
        assert!(
            (summary.displacement_scale - 1.0).abs() < 0.03,
            "{variant:?} scale {}",
            summary.displacement_scale
        );
    }
}

#[test]
#[ignore = "release-only: renders two 758x572 cameras per frame; run `cargo test --release -p visloc-basalt --test multicam_inertial_synthetic -- --include-ignored`"]
fn multicam_rendered_cam0_blank_wall_keeps_tracking() {
    let mut scenario = moving_scenario(0x5eed_aa55_0000_0003);
    scenario.seconds = 9.0;
    scenario.blank = Some(Blank {
        start_s: 3.0,
        end_s: 6.0,
    });
    let results = rendered_comparison("multicam_rendered_blank", &scenario);
    let get = |variant| &results.iter().find(|(v, _)| *v == variant).unwrap().1;
    let (pinned, multi) = (get(Variant::Cam0Only), get(Variant::MultiCamera));
    assert_eq!(multi.blank_frames_without_vision, 0);
    assert!(multi.max_cam1_hosted_landmarks > 10);
    assert!(
        multi.max_from_blank() < pinned.max_from_blank(),
        "multi {} vs pinned {}",
        multi.max_from_blank(),
        pinned.max_from_blank()
    );
    assert!(multi.rms() < 0.10, "multi rms {}", multi.rms());
}

#[test]
#[ignore = "release-only: estimator-only but long; run `cargo test --release -p visloc-basalt --test multicam_inertial_synthetic -- --include-ignored`"]
fn multicam_ideal_tracks_cam0_blank_wall() {
    let mut scenario = moving_scenario(0x5eed_aa55_0000_0004);
    scenario.seconds = 9.0;
    scenario.blank = Some(Blank {
        start_s: 3.0,
        end_s: 6.0,
    });
    let pinned = run_with(&scenario, ideal_track_source(&scenario, Variant::Cam0Only));
    report("multicam_ideal_blank", Variant::Cam0Only, &pinned);
    let multi = run_with(
        &scenario,
        ideal_track_source(&scenario, Variant::MultiCamera),
    );
    report("multicam_ideal_blank", Variant::MultiCamera, &multi);
    assert_eq!(multi.blank_frames_without_vision, 0);
    assert!(multi.max_from_blank() < pinned.max_from_blank());
    assert!(multi.rms() < 0.05, "multi rms {}", multi.rms());
    assert!((multi.displacement_scale - 1.0).abs() < 0.02);
}
