//! Synthetic monocular-inertial (cam0 + IMU) replay through the Basalt VIO
//! estimator.
//!
//! A pinhole camera (Double Sphere with `xi = alpha = 0`) rigidly mounted on
//! an IMU follows a smooth 3-D trajectory with rotation through a static
//! point scene.  The IMU stream carries constant biases plus white noise; the
//! camera stream carries pixel noise and a frontend-like track lifecycle
//! (tracks die when they leave the image and are replenished with new IDs).
//! The estimator is built from a one-camera [`BasaltCalibration`] and the
//! repository's pinned `configs/basalt/euroc_config.json`, exactly as the
//! `--mono` EuRoC replay builds it, so every new landmark must be seeded by
//! temporal (multi-frame) triangulation and metric scale can only come from
//! the IMU.
//!
//! The trajectory starts at rest (zero velocity and acceleration), which is
//! the assumption behind Basalt's gravity-aligned initialization.

use std::path::PathBuf;

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_basalt::{
    config::BasaltConfig, vio::EstimatorOutput, vio_estimator_from_calibration, BasaltCalibration,
    BasaltVioEstimatorAdapter, DoubleSphereCamera, EurocSensorFrame, ImuSample, RawU16Image,
    TrackObservation,
};
use visloc_core::geometry::SE3;

const WIDTH: u32 = 752;
const HEIGHT: u32 = 480;
const FX: f64 = 460.0;
const FY: f64 = 460.0;
const CX: f64 = 376.0;
const CY: f64 = 240.0;
const GRAVITY: f64 = 9.81;
const START_NS: i64 = 1_000_000_000;
const IMU_PERIOD_NS: i64 = 5_000_000; // 200 Hz
const IMU_PER_FRAME: i64 = 10; // camera at 20 Hz

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

/// `f(u) = (1 - cos u)^2 / 4`: f, f' and f'' all vanish at u = 0, so the
/// body starts at rest and without acceleration.
fn rest_start_profile(u: f64) -> (f64, f64, f64) {
    let (s, c) = u.sin_cos();
    let value = (1.0 - c).powi(2) / 4.0;
    let first = (1.0 - c) * s / 2.0;
    let second = (s * s + (1.0 - c) * c) / 2.0;
    (value, first, second)
}

#[derive(Clone, Copy)]
struct Motion {
    /// Per-axis translation amplitude (m) and angular frequency (rad/s).
    amplitude: Vector3<f64>,
    frequency: Vector3<f64>,
    /// Roll/pitch/yaw amplitude (rad) and angular frequency (rad/s).
    angle_amplitude: Vector3<f64>,
    angle_frequency: Vector3<f64>,
    /// Seconds held exactly at rest before the motion profile starts.
    rest_seconds: f64,
}

impl Motion {
    fn position_velocity_acceleration(&self, t: f64) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let t = (t - self.rest_seconds).max(0.0);
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

    /// `R_w_i(t)`; identity at t = 0 so the true world frame equals the
    /// gravity-aligned frame Basalt initializes from the first accelerometer
    /// sample (up to that sample's noise and bias).
    fn rotation(&self, t: f64) -> UnitQuaternion<f64> {
        let t = (t - self.rest_seconds).max(0.0);
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

struct Track {
    id: u64,
    point_world: Point3<f64>,
}

struct Scenario {
    motion: Motion,
    seconds: f64,
    accel_bias: Vector3<f64>,
    gyro_bias: Vector3<f64>,
    accel_noise: f64,
    gyro_noise: f64,
    pixel_noise: f64,
    /// Per-frame probability that an ideal track is lost.
    track_drop_probability: f64,
    /// Ideal-frontend track budget: replenish to `target_tracks` whenever
    /// fewer than `replenish_below` tracks survive.
    target_tracks: usize,
    replenish_below: usize,
    seed: u64,
}

struct RunSummary {
    frames: usize,
    keyframes: usize,
    mapper_packets: usize,
    position_errors: Vec<f64>,
    dead_reckoning_errors: Vec<f64>,
    displacement_scale: f64,
    path_length: f64,
    max_visual_rows: usize,
    first_visual_frame: Option<usize>,
    final_velocity_error: f64,
}

fn camera() -> DoubleSphereCamera {
    // Double Sphere with xi = alpha = 0 is an ideal pinhole.
    DoubleSphereCamera::new(FX, FY, CX, CY, 0.0, 0.0, WIDTH, HEIGHT).unwrap()
}

/// `T_imu_cam`: the optical axis looks along IMU +x, image x along IMU -y and
/// image y along IMU -z (a forward-looking camera on a z-up body).
fn camera_to_imu() -> SE3 {
    let columns = Matrix3::from_columns(&[
        Vector3::new(0.0, -1.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0),
        Vector3::new(1.0, 0.0, 0.0),
    ]);
    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(columns));
    SE3::new(rotation, Vector3::new(0.06, -0.01, 0.02))
}

fn mono_calibration() -> BasaltCalibration {
    BasaltCalibration {
        t_imu_cam: vec![camera_to_imu()],
        cameras: vec![camera()],
        resolutions: vec![(WIDTH, HEIGHT)],
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

fn euroc_config() -> BasaltConfig {
    BasaltConfig::from_json(include_str!("../../../configs/basalt/euroc_config.json")).unwrap()
}

fn project(camera_pose_world: &SE3, point_world: &Point3<f64>) -> Option<nalgebra::Point2<f64>> {
    let point_camera = camera_pose_world.inverse().transform_point(point_world);
    if point_camera.z < 0.3 {
        return None;
    }
    let u = FX * point_camera.x / point_camera.z + CX;
    let v = FY * point_camera.y / point_camera.z + CY;
    let margin = 4.0;
    (u >= margin
        && v >= margin
        && u <= f64::from(WIDTH) - 1.0 - margin
        && v <= f64::from(HEIGHT) - 1.0 - margin)
        .then(|| nalgebra::Point2::new(u, v))
}

fn spawn_track(rng: &mut Rng, camera_pose_world: &SE3, id: u64) -> Track {
    let u = rng.range(20.0, f64::from(WIDTH) - 20.0);
    let v = rng.range(20.0, f64::from(HEIGHT) - 20.0);
    let depth = rng.range(2.5, 8.0);
    let point_camera = Point3::new((u - CX) / FX * depth, (v - CY) / FY * depth, depth);
    Track {
        id,
        point_world: camera_pose_world.transform_point(&point_camera),
    }
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

/// Bias-unaware strapdown integration of the same noisy IMU stream from the
/// true initial state: the drift the visual factors must remove.
struct DeadReckoning {
    rotation: UnitQuaternion<f64>,
    velocity: Vector3<f64>,
    position: Vector3<f64>,
    last_ns: Option<i64>,
}

impl DeadReckoning {
    fn push(&mut self, sample: &ImuSample) {
        if let Some(last) = self.last_ns {
            let dt = (sample.timestamp_ns - last) as f64 * 1e-9;
            let acceleration = self.rotation * sample.accel_m_s2 - Vector3::new(0.0, 0.0, GRAVITY);
            self.position += self.velocity * dt + 0.5 * acceleration * dt * dt;
            self.velocity += acceleration * dt;
            self.rotation *= UnitQuaternion::from_scaled_axis(sample.gyro_rad_s * dt);
        }
        self.last_ns = Some(sample.timestamp_ns);
    }
}

/// One camera event handed to a frame source: frame index, timestamp, true
/// `T_w_c`, the IMU samples in `(previous frame, this frame]`, and the first
/// sample at or after the first frame (Basalt's initialization sample).
struct FrameEvent<'a> {
    frame: usize,
    timestamp_ns: i64,
    camera_pose: &'a SE3,
    imu: Vec<ImuSample>,
    initialization_imu: Option<ImuSample>,
}

/// Ideal frontend: projects scene points with pixel noise and a KLT-like
/// track lifecycle, then feeds the bare estimator.
fn ideal_track_source(
    scenario: &Scenario,
) -> impl FnMut(FrameEvent<'_>, &mut Rng) -> EstimatorOutput {
    ideal_track_source_with(scenario, false)
}

/// `retain_marg_data` selects the full `process` path, which also builds and
/// validates the MargData mapper packet for every frame.
fn ideal_track_source_with(
    scenario: &Scenario,
    retain_marg_data: bool,
) -> impl FnMut(FrameEvent<'_>, &mut Rng) -> EstimatorOutput {
    let calibration = mono_calibration();
    let mut estimator = vio_estimator_from_calibration(&calibration, &euroc_config()).unwrap();
    let mut next_track_id = 0_u64;
    let mut tracks: Vec<Track> = Vec::new();
    let pixel_noise = scenario.pixel_noise;
    let drop_probability = scenario.track_drop_probability;
    let (target_tracks, replenish_below) = (scenario.target_tracks, scenario.replenish_below);
    move |event, rng| {
        let camera_pose = event.camera_pose;
        // KLT-like loss: tracks leaving the image die, and a small random
        // fraction is lost each frame (occlusion / patch failure). A lost
        // track never comes back under the same ID.
        tracks.retain(|track| {
            project(camera_pose, &track.point_world).is_some() && rng.uniform() >= drop_probability
        });
        if tracks.len() < replenish_below {
            while tracks.len() < target_tracks {
                tracks.push(spawn_track(rng, camera_pose, next_track_id));
                next_track_id += 1;
            }
        }
        let mut observations = Vec::with_capacity(tracks.len());
        for track in &tracks {
            let pixel = project(camera_pose, &track.point_world).unwrap();
            observations.push(TrackObservation {
                track_id: track.id,
                frame_id: event.frame as u64,
                timestamp_ns: event.timestamp_ns,
                camera_id: 0,
                pixel: nalgebra::Point2::new(
                    pixel.x + pixel_noise * rng.gaussian(),
                    pixel.y + pixel_noise * rng.gaussian(),
                ),
            });
        }
        let output = if retain_marg_data {
            estimator.process(
                event.frame as u64,
                event.timestamp_ns,
                &observations,
                &event.imu,
            )
        } else {
            estimator.process_without_marg_data(
                event.frame as u64,
                event.timestamp_ns,
                &observations,
                &event.imu,
            )
        }
        .unwrap_or_else(|error| panic!("frame {}: {error}", event.frame));
        if retain_marg_data {
            if let Err(error) = output.marg_data.validate_contract() {
                panic!("frame {}: invalid mono MargData: {error}", event.frame);
            }
        }
        output
    }
}

/// Closed textured room `x in [-3, 6], y in [-4, 4], z in [-1.6, 2.0]` (m).
/// Every face carries a random-intensity tile pattern (0.12 m tiles) so FAST
/// finds corners and the KLT patches have gradients at every pyramid level.
const ROOM_MIN: [f64; 3] = [-3.0, -4.0, -1.6];
const ROOM_MAX: [f64; 3] = [6.0, 4.0, 2.0];
const TILE: f64 = 0.12;

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

fn room_radiance(origin: &Vector3<f64>, direction: &Vector3<f64>) -> f64 {
    let mut best = (f64::INFINITY, 0.0);
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
        if distance > 0.0 && distance < best.0 {
            let hit = origin + direction * distance;
            let (a, b) = match axis {
                0 => (hit.y, hit.z),
                1 => (hit.x, hit.z),
                _ => (hit.x, hit.y),
            };
            let face = axis as u64 * 2 + u64::from(direction[axis] > 0.0);
            best = (distance, tile_intensity(face, a, b));
        }
    }
    best.1
}

/// Renders cam0 with 2x2 supersampling, in the 8-bit-in-u16 layout the
/// EuRoC reader produces.
fn render(camera_pose_world: &SE3) -> RawU16Image {
    let rotation = camera_pose_world.rotation.to_rotation_matrix();
    let origin = camera_pose_world.translation;
    RawU16Image::from_fn(WIDTH as usize, HEIGHT as usize, |x, y| {
        let mut sum = 0.0;
        for (sx, sy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
            // Pixel centers are integer coordinates in Basalt's convention.
            let u = x as f64 + sx - 0.5;
            let v = y as f64 + sy - 0.5;
            let ray = rotation * Vector3::new((u - CX) / FX, (v - CY) / FY, 1.0);
            sum += room_radiance(&origin, &ray);
        }
        ((sum / 4.0).round().clamp(0.0, 255.0) as u16) << 8
    })
    .unwrap()
}

/// Real pipeline: renders cam0 only and runs the direct KLT frontend plus the
/// estimator through [`BasaltVioEstimatorAdapter`] built from a one-camera
/// calibration, exactly as the `--mono` EuRoC replay does.
fn rendered_adapter_source() -> impl FnMut(FrameEvent<'_>, &mut Rng) -> EstimatorOutput {
    let mut adapter =
        BasaltVioEstimatorAdapter::from_config(&mono_calibration(), &euroc_config()).unwrap();
    move |event, _rng| {
        let frame = EurocSensorFrame {
            frame_id: event.frame as u64,
            timestamp_ns: event.timestamp_ns,
            cam0: render(event.camera_pose),
            cam1: None,
            cam0_path: PathBuf::new(),
            cam1_path: None,
            imu: event.imu,
            initialization_imu: event.initialization_imu,
        };
        let output = adapter
            .process_without_marg_data(frame)
            .unwrap_or_else(|error| panic!("frame {}: {error}", event.frame));
        assert!(output
            .tracks
            .observations
            .iter()
            .all(|observation| observation.camera_id == 0));
        output.estimator
    }
}

fn run(scenario: &Scenario) -> RunSummary {
    run_with(scenario, ideal_track_source(scenario))
}

fn run_with<F>(scenario: &Scenario, mut source: F) -> RunSummary
where
    F: FnMut(FrameEvent<'_>, &mut Rng) -> EstimatorOutput,
{
    let t_imu_cam = camera_to_imu();
    let mut rng = Rng(scenario.seed);
    let mut dead_reckoning = DeadReckoning {
        rotation: UnitQuaternion::identity(),
        velocity: Vector3::zeros(),
        position: Vector3::zeros(),
        last_ns: None,
    };

    let frames = (scenario.seconds * 20.0).round() as usize;
    let mut summary = RunSummary {
        frames,
        keyframes: 0,
        mapper_packets: 0,
        position_errors: Vec::with_capacity(frames),
        dead_reckoning_errors: Vec::with_capacity(frames),
        displacement_scale: f64::NAN,
        path_length: 0.0,
        max_visual_rows: 0,
        first_visual_frame: None,
        final_velocity_error: f64::NAN,
    };
    let mut estimated: Vec<Vector3<f64>> = Vec::with_capacity(frames);
    let mut truth: Vec<Vector3<f64>> = Vec::with_capacity(frames);
    let mut next_imu_ns = START_NS;

    for frame in 0..frames {
        let timestamp_ns = START_NS + frame as i64 * IMU_PER_FRAME * IMU_PERIOD_NS;
        let t = (timestamp_ns - START_NS) as f64 * 1e-9;
        let mut imu = Vec::new();
        while next_imu_ns <= timestamp_ns {
            let sample = imu_sample(scenario, &mut rng, next_imu_ns);
            dead_reckoning.push(&sample);
            imu.push(sample);
            next_imu_ns += IMU_PERIOD_NS;
        }

        let camera_pose = scenario.motion.imu_pose(t).compose(&t_imu_cam);
        let initialization_imu = (frame == 0).then(|| *imu.last().unwrap());
        let output = source(
            FrameEvent {
                frame,
                timestamp_ns,
                camera_pose: &camera_pose,
                imu,
                initialization_imu,
            },
            &mut rng,
        );
        summary.keyframes += usize::from(output.is_keyframe);
        summary.mapper_packets += usize::from(output.marg_data.is_mapper_packet());
        if output.window.visual_factor_rows > 0 && summary.first_visual_frame.is_none() {
            summary.first_visual_frame = Some(frame);
        }
        summary.max_visual_rows = summary
            .max_visual_rows
            .max(output.window.visual_factor_rows);

        let (true_position, true_velocity, _) = scenario.motion.position_velocity_acceleration(t);
        let estimated_position = output.state.imu_to_world.translation;
        assert!(estimated_position.iter().all(|value| value.is_finite()));
        summary
            .position_errors
            .push((estimated_position - true_position).norm());
        summary
            .dead_reckoning_errors
            .push((dead_reckoning.position - true_position).norm());
        summary.final_velocity_error = (output.state.velocity_world_m_s - true_velocity).norm();
        if let Some(previous) = truth.last() {
            summary.path_length += (true_position - previous).norm();
        }
        estimated.push(estimated_position);
        truth.push(true_position);
    }

    // Least-squares scale between estimated and true displacements from the
    // first frame: 1.0 means metric scale was recovered (a monocular-only
    // system would leave it arbitrary).
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

fn rms(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn report(name: &str, summary: &RunSummary) {
    eprintln!(
        "{name}: frames={} keyframes={} mapper_packets={} first_visual_frame={:?} \
         max_visual_rows={} path={:.3} m rms_err={:.4} m max_err={:.4} m final_err={:.4} m \
         final_vel_err={:.4} m/s scale={:.4} dead_reckoning_final_err={:.4} m",
        summary.frames,
        summary.keyframes,
        summary.mapper_packets,
        summary.first_visual_frame,
        summary.max_visual_rows,
        summary.path_length,
        rms(&summary.position_errors),
        summary.position_errors.iter().copied().fold(0.0, f64::max),
        summary.position_errors.last().unwrap(),
        summary.final_velocity_error,
        summary.displacement_scale,
        summary.dead_reckoning_errors.last().unwrap(),
    );
}

fn moving_scenario(seed: u64) -> Scenario {
    Scenario {
        motion: Motion {
            amplitude: Vector3::new(0.6, 1.2, 0.5),
            frequency: Vector3::new(1.3, 1.7, 2.1),
            angle_amplitude: Vector3::new(0.08, 0.10, 0.25),
            angle_frequency: Vector3::new(1.1, 1.5, 0.9),
            rest_seconds: 0.0,
        },
        seconds: 8.0,
        accel_bias: Vector3::new(0.04, -0.03, 0.05),
        gyro_bias: Vector3::new(0.002, -0.0015, 0.001),
        accel_noise: 0.02,
        gyro_noise: 0.002,
        pixel_noise: 0.5,
        track_drop_probability: 0.02,
        target_tracks: 150,
        replenish_below: 110,
        seed,
    }
}

/// Shared acceptance bounds for a run that moves from the first frame.
fn assert_tracks_with_metric_scale(summary: &RunSummary) {
    // Vision must actually participate: temporal triangulation seeded
    // landmarks and their reprojection rows entered the window solve soon
    // after the IMU-propagated baseline passed Basalt's 5 cm gate.
    let first_visual = summary
        .first_visual_frame
        .expect("no visual factor ever entered the monocular window");
    assert!(first_visual < 20, "first visual frame {first_visual}");
    assert!(summary.max_visual_rows > 100);
    assert!(summary.keyframes > 8, "keyframes {}", summary.keyframes);

    let rms_error = rms(&summary.position_errors);
    let max_error = summary.position_errors.iter().copied().fold(0.0, f64::max);
    let final_error = *summary.position_errors.last().unwrap();
    let dead_reckoning_final = *summary.dead_reckoning_errors.last().unwrap();
    assert!(summary.path_length > 3.0, "path {}", summary.path_length);
    // Observed: a few millimetres over a ~5-7 m path; the bounds leave
    // generous margin for platform floating-point differences.
    assert!(rms_error < 0.05, "rms position error {rms_error}");
    assert!(max_error < 0.10, "max position error {max_error}");
    // IMU-only integration of the same biased stream drifts by metres.
    assert!(
        dead_reckoning_final > 1.0 && final_error < 0.05 * dead_reckoning_final,
        "VIO final error {final_error} vs dead reckoning {dead_reckoning_final}"
    );
    assert!(
        (summary.displacement_scale - 1.0).abs() < 0.02,
        "displacement scale {}",
        summary.displacement_scale
    );
    assert!(
        summary.final_velocity_error < 0.10,
        "final velocity error {}",
        summary.final_velocity_error
    );
}

/// Always-on smoke version, sized for unoptimized `cargo test` (the
/// estimator is ~40x slower without optimization): fewer landmarks and a
/// shorter run, still long enough for temporal triangulation to start and
/// vision to bound the IMU drift. The full-length scenarios below are
/// `#[ignore]`d and must be run with `--release ... --include-ignored`.
#[test]
fn monocular_inertial_vio_smoke() {
    let mut scenario = moving_scenario(0x5eed_1234_abcd_0005);
    scenario.seconds = 2.0;
    scenario.target_tracks = 40;
    scenario.replenish_below = 30;
    let summary = run(&scenario);
    report("mono_inertial_smoke", &summary);

    let first_visual = summary
        .first_visual_frame
        .expect("no visual factor ever entered the monocular window");
    assert!(first_visual < 20, "first visual frame {first_visual}");
    let rms_error = rms(&summary.position_errors);
    let final_error = *summary.position_errors.last().unwrap();
    let dead_reckoning_final = *summary.dead_reckoning_errors.last().unwrap();
    assert!(rms_error < 0.05, "rms position error {rms_error}");
    assert!(
        final_error < 0.5 * dead_reckoning_final,
        "VIO final error {final_error} vs dead reckoning {dead_reckoning_final}"
    );
    assert!(
        (summary.displacement_scale - 1.0).abs() < 0.05,
        "displacement scale {}",
        summary.displacement_scale
    );
}

#[test]
#[ignore = "release-only: about 40x slower unoptimized; run `cargo test --release -p visloc-basalt --test mono_inertial_synthetic -- --include-ignored`"]
fn monocular_inertial_vio_tracks_with_metric_scale() {
    let scenario = moving_scenario(0x5eed_1234_abcd_0001);
    let summary = run(&scenario);
    report("mono_inertial_ideal_tracks", &summary);
    assert_tracks_with_metric_scale(&summary);
}

#[test]
#[ignore = "release-only: about 40x slower unoptimized; run `cargo test --release -p visloc-basalt --test mono_inertial_synthetic -- --include-ignored`"]
fn monocular_inertial_margdata_path_emits_valid_mapper_packets() {
    // The full `process` path builds a MargData snapshot every frame and a
    // mapper packet whenever a keyframe is marginalized; the source checks
    // `validate_contract` on each one.
    let mut scenario = moving_scenario(0x5eed_1234_abcd_0004);
    scenario.seconds = 6.0;
    scenario.track_drop_probability = 0.04;
    let summary = run_with(&scenario, ideal_track_source_with(&scenario, true));
    report("mono_inertial_margdata", &summary);
    assert!(summary.mapper_packets > 0, "no keyframe was marginalized");
    assert!(rms(&summary.position_errors) < 0.05);
    assert!((summary.displacement_scale - 1.0).abs() < 0.02);
}

#[test]
#[ignore = "release-only: about 40x slower unoptimized; run `cargo test --release -p visloc-basalt --test mono_inertial_synthetic -- --include-ignored`"]
fn monocular_inertial_adapter_tracks_rendered_images() {
    // End to end: rendered cam0 images through the direct KLT frontend (no
    // cam1 image, no stereo stage) into the estimator via the adapter.
    let mut scenario = moving_scenario(0x5eed_1234_abcd_0003);
    scenario.seconds = 6.0;
    let summary = run_with(&scenario, rendered_adapter_source());
    report("mono_inertial_rendered_klt", &summary);
    assert_tracks_with_metric_scale(&summary);
}

#[test]
#[ignore = "release-only: about 40x slower unoptimized; run `cargo test --release -p visloc-basalt --test mono_inertial_synthetic -- --include-ignored`"]
fn monocular_inertial_static_start_drifts_then_recovers() {
    // Documented limitation, pinned so a change is noticed: while the body
    // is at rest a monocular rig has no parallax, so translation is carried
    // by the IMU alone (as in IMU-only dead reckoning) until motion gives
    // the temporal triangulation real baselines. Basalt's gravity-aligned
    // zero-velocity initialization is still the right prior here; the drift
    // is bias-driven and bounded by the rest duration.
    let mut scenario = moving_scenario(0x5eed_1234_abcd_0002);
    scenario.motion.rest_seconds = 2.0;
    scenario.seconds = 10.0;
    let summary = run(&scenario);
    report("mono_inertial_static_start", &summary);

    let rest_frames = 40;
    let rest_max = summary.position_errors[..rest_frames]
        .iter()
        .copied()
        .fold(0.0, f64::max);
    let moving_tail = &summary.position_errors[rest_frames + 40..];
    let tail_max = moving_tail.iter().copied().fold(0.0, f64::max);
    eprintln!(
        "static_start: rest_max_err={rest_max:.4} m post_motion_tail_max_err={tail_max:.4} m"
    );
    assert!(rest_max < 0.25, "drift at rest {rest_max}");
    assert!(
        tail_max < 0.05,
        "error two seconds after motion onset {tail_max}"
    );
    assert!((summary.displacement_scale - 1.0).abs() < 0.03);
}
