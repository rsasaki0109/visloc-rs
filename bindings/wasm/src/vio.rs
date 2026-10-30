//! A live monocular-inertial VIO session: IMU samples and camera frames go
//! in as they arrive, a pose per frame comes out.
//!
//! The estimator is the Basalt port's (`BasaltVioEstimatorAdapter`: direct
//! KLT frontend, sliding-window VIO), built from a one-camera calibration as
//! `basalt_euroc_vio_demo --mono` builds it. A phone camera is an ideal
//! pinhole here (Double Sphere with `xi = alpha = 0`): its pipeline delivers
//! undistorted frames.
//!
//! Each frame is handed the IMU samples since the previous frame, as the
//! EuRoC reader slices them (strictly after the previous frame, through this
//! one), so frames the caller drops simply lengthen the next interval. A
//! frame is only ready once an IMU sample at or after its timestamp has
//! arrived: that sample both closes the interval and, for the first frame,
//! aligns the initial pose with gravity.
//!
//! Monocular VIO on phone data occasionally diverges for good (on ADVIO's
//! iPhone recordings: a near-static, low-parallax start or a stretch with
//! few landmarks, after which a runaway gyro / accelerometer bias absorbs
//! the visual residuals and the speed grows without bound). A [`Watchdog`]
//! notices the implausible state, restarts the estimator, and anchors the
//! new estimate to the last good position and heading, so the trajectory
//! the caller sees continues in one world frame.

use std::collections::VecDeque;

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use visloc_basalt::config::BasaltConfig;
use visloc_basalt::{
    BasaltAdapterOutput, BasaltCalibration, BasaltFrame, BasaltNavState, BasaltVioEstimatorAdapter,
    DoubleSphereCamera, ImuSample, RawU16Image, StereoFrame, TrackFrameOutput,
};
use visloc_core::geometry::SE3;

use crate::gpu_flow::{GpuContext, GpuFlow};
use crate::sync::{SyncEstimate, TimeSync};

/// The pinned EuRoC configuration the phone settings start from.
const EUROC_CONFIG: &str = include_str!("../../../configs/basalt/euroc_config.json");

/// Feature-grid cells across the frame's short side. The frontend keeps at
/// most one new point per cell: the EuRoC file's 50 px suits its 752 x 480
/// frames, but caps a phone's 360 px frames at about 60 points (median 13
/// landmarks on an iPhone). On 90 s of ADVIO's advio-02 and advio-13, 13
/// cells across (28 px at 360 px) cut the trajectory error by a third and
/// the scale error from 5-8 % to 0.4-2 %, for about 45 % more time per frame.
const GRID_CELLS_ACROSS: f64 = 13.0;

/// Least distance (m) between two observations of a track for a new
/// landmark (`vio_min_triangulation_dist`; the EuRoC file's 5 cm suits a
/// stereo rig on a drone). A hand-held phone that mostly turns rarely moves
/// 5 cm between keyframes: on an iPhone recording, 1 cm raised the median of
/// landmarks in view from 31 to 84 and the frames tracking from 85 % to 95 %;
/// on ADVIO's four walks it left the trajectory error unchanged within the
/// runs' spread and had the fewest resets (3 against 8), for about 40 % more
/// time per frame.
const MIN_TRIANGULATION_M: f64 = 0.01;
/// The still-start gate judges motion over at least this long (every other
/// frame at 60 Hz), so its threshold means the same at either rate.
const MIN_GATE_INTERVAL_NS: i64 = 25_000_000;
/// After landmarks were seen, frames without one for this long report
/// coasting (two frames at 30 Hz, three at 60).
const COAST_AFTER_S: f64 = 0.025;

/// Least frames after a keyframe before the next (`vio_min_frames_after_kf`;
/// 5 in the EuRoC file) and the non-keyframe states kept in the window
/// (`vio_max_states`; 3). A monocular window gets new landmarks only at
/// keyframes, and a phone that turns loses most of its tracks within the
/// EuRoC file's five frames. Together, 2 and 2 raised the median of
/// landmarks in view on three iPhone recordings from 85 to 93, the frames
/// tracking from 90 % to 94 %, and on ADVIO's four walks cut the trajectory
/// error from 3.8 m to 2.6 m and the scale error from 13 % to 4 %, at about
/// the same time per frame (the extra keyframes cost what the smaller window
/// saves). Either alone did worse than both.
const MIN_FRAMES_AFTER_KF: u64 = 2;
const MAX_STATES: u64 = 2;

/// Seconds of motion samples before the first frame whose specific force is
/// averaged for its attitude. Basalt takes roll and pitch from the one
/// sample after the first frame, which suits a still drone at 200 Hz; a
/// hand-held phone that is being moved side to side to start accelerates by
/// 0.5-1 m/s², a tilt error of 3-6° that leaks gravity into the velocity.
const GRAVITY_WINDOW_S: f64 = 0.5;

/// Samples before the first frame kept for it (the estimator only needs the
/// one at or after the frame to initialize).
const MAX_SAMPLES_BEFORE_FIRST_FRAME: usize = 32;

/// A phone's camera and IMU as the estimator sees them.
#[derive(Debug, Clone)]
pub struct PhoneRig {
    /// Size of the frames, pixels.
    pub width: u32,
    pub height: u32,
    /// Pinhole intrinsics at that size.
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Radial-tangential distortion `[k1, k2, p1, p2]` of the frames (all
    /// zero: already undistorted). Frames are remapped to the pinhole
    /// camera before tracking.
    pub distortion: [f64; 4],
    /// `T_imu_cam`: camera axes and origin in the IMU (device) frame.
    pub imu_from_camera: SE3,
    /// Nominal IMU rate, Hz (Basalt discretizes the noise densities with it).
    pub imu_rate_hz: f64,
    /// Continuous-time white-noise densities: m/s²/√Hz and rad/s/√Hz.
    pub accel_noise: f64,
    pub gyro_noise: f64,
    /// Bias random walks: m/s³/√Hz and rad/s²/√Hz.
    pub accel_bias_walk: f64,
    pub gyro_bias_walk: f64,
    /// Frames tracked per second (30 or 60), as the page sets it and the
    /// recordings keep it. The session's own timings are in seconds, so no
    /// setting depends on it.
    pub frame_rate_hz: f64,
}

impl PhoneRig {
    /// The rear camera of a phone held upright in portrait, `offset` (m)
    /// from the IMU in the device frame. The device frame is the W3C one
    /// (and CoreMotion's / Android's): x right, y up the screen, z out of
    /// the screen. The camera looks out of the back (-z), with image x
    /// along device x and image y down the screen (-y).
    pub fn rear_portrait_camera_to_imu(offset: Vector3<f64>) -> SE3 {
        let columns = Matrix3::from_columns(&[
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
        ]);
        SE3::new(
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(columns)),
            offset,
        )
    }

    fn calibration(&self) -> Result<BasaltCalibration, String> {
        let camera = DoubleSphereCamera::new(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            0.0,
            0.0,
            self.width,
            self.height,
        )
        .map_err(|e| format!("camera: {e}"))?;
        Ok(BasaltCalibration {
            t_imu_cam: vec![self.imu_from_camera.clone()],
            cameras: vec![camera],
            resolutions: vec![(self.width, self.height)],
            calib_accel_bias: vec![0.0; 9],
            calib_gyro_bias: vec![0.0; 12],
            imu_update_rate_hz: self.imu_rate_hz,
            accel_noise_std: Vector3::repeat(self.accel_noise),
            gyro_noise_std: Vector3::repeat(self.gyro_noise),
            accel_bias_std: Vector3::repeat(self.accel_bias_walk),
            gyro_bias_std: Vector3::repeat(self.gyro_bias_walk),
            t_mocap_world: SE3::identity(),
            t_imu_marker: SE3::identity(),
            mocap_time_offset_ns: 0,
            mocap_to_imu_offset_ns: 0,
            cam_time_offset_ns: 0,
        })
    }
}

/// The estimator configuration: the pinned EuRoC file with `overrides`
/// (`"config.*"` keys) replacing its values.
pub fn config_with(overrides: &[(String, serde_json::Value)]) -> Result<BasaltConfig, String> {
    let mut json: serde_json::Value =
        serde_json::from_str(EUROC_CONFIG).map_err(|e| format!("config: {e}"))?;
    let values = json
        .get_mut("value0")
        .and_then(|v| v.as_object_mut())
        .ok_or("config: no value0 object")?;
    for (key, value) in overrides {
        values.insert(key.clone(), value.clone());
    }
    BasaltConfig::from_json(&json.to_string()).map_err(|e| format!("config: {e}"))
}

/// When the estimate of a hand-held phone is considered lost.
#[derive(Debug, Clone)]
pub struct Watchdog {
    pub enabled: bool,
    pub max_speed_m_s: f64,
    pub max_gyro_bias_rad_s: f64,
    pub max_accel_bias_m_s2: f64,
    /// Seconds in a row that track at least `min_tracks` points yet have no
    /// landmark connected (the estimator is then integrating the IMU only).
    pub max_unconnected_s: f64,
    pub min_tracks: usize,
    /// On a reset, the new estimate is anchored to the pose this long
    /// before the reset (a divergence builds up over about a second before
    /// it is noticed).
    pub anchor_back_s: f64,
}

impl Default for Watchdog {
    fn default() -> Self {
        Self {
            enabled: true,
            max_speed_m_s: 3.0,
            max_gyro_bias_rad_s: 0.1,
            max_accel_bias_m_s2: 1.0,
            max_unconnected_s: 2.0,
            min_tracks: 20,
            anchor_back_s: 1.0,
        }
    }
}

/// When a (re)started estimator may begin. Basalt starts a window at rest:
/// zero velocity, and roll and pitch from the specific force. A phone that is
/// moving then (the page used to ask for side-to-side motion to start) gets
/// no landmarks for the first second or so, integrates the wrong velocity,
/// triangulates its first landmarks against the drifted poses, and the window
/// diverges: on 60 three-second starts across four iPhone recordings, 27 %
/// reset: by the whole recording's estimate of the speed at the start, one
/// in eight under 0.2 m/s and five in seven above 0.5 m/s. The
/// motion sensors cannot tell a steady pan from rest, so the camera decides,
/// from the tracked points' motion left after the gyroscope's rotation (only
/// moving through space shifts points against the rotation: on two iPhone
/// recordings its median was 0.5-0.7 px a frame below 0.05 m/s, 1 px at
/// 0.1-0.2 m/s and 2.5 px at 0.3-0.5 m/s). The motion is measured over at
/// least 25 ms (every other frame at 60 Hz, which halves it per frame). While
/// the phone moves, each judgement starts a new estimator; after `still_s`
/// of stillness the estimator restarts once more with its attitude from the
/// still stretch alone.
#[derive(Debug, Clone)]
pub struct StillStart {
    pub enabled: bool,
    /// Median motion of the tracked points over the judged interval (one
    /// frame at 30 Hz) after the gyroscope's rotation, pixels.
    pub max_flow_px: f64,
    /// Largest angular rate in the interval, rad/s (above it, timing errors
    /// make the rotation's prediction unreliable).
    pub max_gyro_rad_s: f64,
    /// Points in both frames needed to judge.
    pub min_points: usize,
    /// Seconds of stillness before the estimator starts (six frames at 30
    /// Hz).
    pub still_s: f64,
    /// Seconds to wait for stillness before starting anyway.
    pub max_wait_s: f64,
}

impl Default for StillStart {
    fn default() -> Self {
        Self {
            enabled: true,
            max_flow_px: 1.2,
            max_gyro_rad_s: 1.0,
            min_points: 20,
            still_s: 0.18,
            max_wait_s: 2.0,
        }
    }
}

/// Where the session stands after a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Waiting for the phone to be held still before starting
    /// ([`StillStart`]).
    Waiting,
    /// No landmark in the window yet since the (re)start.
    Initializing,
    Tracking,
    /// No landmark seen in this frame or the one before, after some were:
    /// the pose comes from the IMU alone (fast motion blurs the frames or
    /// moves the points too far for the tracker) until the camera's points
    /// connect again or the watchdog gives up.
    Coasting,
    /// The watchdog restarted the estimator after this frame.
    Reset(ResetReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetReason {
    Speed,
    GyroBias,
    AccelBias,
    NoLandmarks,
    NonFinite,
}

impl ResetReason {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Speed => "speed",
            Self::GyroBias => "gyro-bias",
            Self::AccelBias => "accel-bias",
            Self::NoLandmarks => "no-landmarks",
            Self::NonFinite => "non-finite",
        }
    }
}

/// The rotation about +z closest to `r`. Both worlds are gravity-aligned,
/// so a re-anchored estimate only turns about the vertical and keeps its own
/// roll and pitch.
fn yaw_part(r: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    let m = r.to_rotation_matrix();
    let m = m.matrix();
    let yaw = (m[(1, 0)] - m[(0, 1)]).atan2(m[(0, 0)] + m[(1, 1)]);
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), yaw)
}

/// A frame between its IMU bookkeeping and its tracking.
struct Prepared {
    frame_id: u64,
    timestamp_ns: i64,
    imu: Vec<ImuSample>,
    initialization_imu: Option<ImuSample>,
    imu_samples: usize,
    max_gyro: f64,
    turn: UnitQuaternion<f64>,
}

/// What one processed frame produced.
#[derive(Debug, Clone)]
pub struct FrameUpdate {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    /// IMU pose, velocity and biases after the frame.
    pub state: BasaltNavState,
    /// `T_w_c` of the frame.
    pub camera_to_world: SE3,
    pub is_keyframe: bool,
    /// Tracked points: track id and pixel position.
    pub tracks: Vec<(u64, f32, f32)>,
    /// Tracks connected to a landmark in the window.
    pub connected: usize,
    /// Why the frontend dropped tracks on this frame, and how many.
    pub track_rejects: Vec<(String, usize)>,
    pub imu_samples: usize,
    pub status: Status,
    /// Watchdog restarts so far.
    pub resets: u32,
}

/// Bilinear remap of distorted frames to the rig's pinhole camera: per
/// output pixel, the source pixel's top-left corner index and its 8-bit
/// fractional offsets (`None` where the source falls outside the frame).
struct Undistort {
    width: usize,
    source: Vec<Option<(u32, u8, u8)>>,
}

impl Undistort {
    fn new(rig: &PhoneRig) -> Option<Self> {
        let [k1, k2, p1, p2] = rig.distortion;
        if [k1, k2, p1, p2].iter().all(|&k| k == 0.0) {
            return None;
        }
        let (w, h) = (rig.width as usize, rig.height as usize);
        let source = (0..w * h)
            .map(|i| {
                let x = ((i % w) as f64 - rig.cx) / rig.fx;
                let y = ((i / w) as f64 - rig.cy) / rig.fy;
                let r2 = x * x + y * y;
                let radial = 1.0 + k1 * r2 + k2 * r2 * r2;
                let xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
                let yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
                let (u, v) = (rig.fx * xd + rig.cx, rig.fy * yd + rig.cy);
                if !(u >= 0.0 && v >= 0.0 && u < (w - 1) as f64 && v < (h - 1) as f64) {
                    return None;
                }
                let (u0, v0) = (u.floor(), v.floor());
                Some((
                    (v0 as usize * w + u0 as usize) as u32,
                    ((u - u0) * 256.0).min(255.0) as u8,
                    ((v - v0) * 256.0).min(255.0) as u8,
                ))
            })
            .collect();
        Some(Self { width: w, source })
    }

    /// The frame on the pinhole camera, in the estimator's 8-bit-in-u16
    /// layout; pixels with no source are black.
    fn apply(&self, gray: &[u8]) -> Vec<u16> {
        let w = self.width;
        self.source
            .iter()
            .map(|s| {
                let Some((k, fx, fy)) = *s else { return 0 };
                let k = k as usize;
                let (fx, fy) = (u32::from(fx), u32::from(fy));
                let top = u32::from(gray[k]) * (256 - fx) + u32::from(gray[k + 1]) * fx;
                let bottom = u32::from(gray[k + w]) * (256 - fx) + u32::from(gray[k + w + 1]) * fx;
                // 16 fractional bits; round to 8-bit, stored in the high byte.
                let v = (top * (256 - fy) + bottom * fy + (1 << 15)) >> 16;
                (v.min(255) as u16) << 8
            })
            .collect()
    }
}

/// The first frame's initialization sample: `next` (the first sample after
/// the frame) with its specific force replaced by the mean over the last
/// [`GRAVITY_WINDOW_S`] of samples, each rotated into `next`'s frame by the
/// gyroscope, so the tilt is that of gravity rather than of one sample's
/// gravity plus the hand's acceleration; samples before `since_ns` are left
/// out. `next` itself with fewer than five samples.
fn averaged_gravity<'a>(
    before: impl Iterator<Item = &'a ImuSample>,
    next: ImuSample,
    since_ns: Option<i64>,
) -> ImuSample {
    let start =
        (next.timestamp_ns - (GRAVITY_WINDOW_S * 1e9) as i64).max(since_ns.unwrap_or(i64::MIN));
    let window: Vec<ImuSample> = before
        .filter(|s| s.timestamp_ns >= start && s.timestamp_ns < next.timestamp_ns)
        .copied()
        .chain(std::iter::once(next))
        .collect();
    if window.len() < 5 {
        return next;
    }
    // Attitude of each sample relative to the first (the gyroscope value of a
    // sample covers the interval ending at it, as Basalt integrates it).
    let mut attitude = UnitQuaternion::identity();
    let mut sum = window[0].accel_m_s2;
    for pair in window.windows(2) {
        let dt = (pair[1].timestamp_ns - pair[0].timestamp_ns) as f64 * 1e-9;
        attitude *= UnitQuaternion::from_scaled_axis(pair[1].gyro_rad_s * dt);
        sum += attitude * pair[1].accel_m_s2;
    }
    let mean = sum / window.len() as f64;
    ImuSample::new(
        next.timestamp_ns,
        next.gyro_rad_s,
        attitude.inverse() * mean,
    )
}

/// A running estimator and the IMU samples it has not consumed yet.
pub struct Session {
    rig: PhoneRig,
    undistort: Option<Undistort>,
    overrides: Vec<(String, serde_json::Value)>,
    adapter: BasaltVioEstimatorAdapter,
    imu: VecDeque<ImuSample>,
    /// The last second of samples, consumed or not: a (re)started
    /// estimator's first attitude averages them.
    recent_imu: VecDeque<ImuSample>,
    last_frame_ns: Option<i64>,
    next_frame_id: u64,
    pub watchdog: Watchdog,
    /// Maps the current estimator's world into the caller's.
    world_from_estimator: SE3,
    /// Recent IMU poses in the caller's world with their times, newest
    /// last; the oldest is the newest one `anchor_back_s` old.
    recent: VecDeque<(i64, SE3)>,
    /// Pose the next estimator's first frame is anchored to.
    anchor: Option<SE3>,
    /// Since when no landmark has been connected.
    unconnected_since_ns: Option<i64>,
    /// Run the estimator on every `estimate_every`th frame (1, the default:
    /// all); the others get the last update carried forward by the IMU. At
    /// 60 Hz, 2 tracks points on every frame and estimates at 30 Hz: on
    /// ADVIO's 60 fps walks 3.1 m against 4.3 m for every frame, but on a
    /// phone's 60 fps recording it reset once in three offsets where every
    /// frame did not, and the carried poses jump at each estimate.
    pub estimate_every: u32,
    /// The CPU frontend's frame number, and frames since the (re)start.
    front_seq: u64,
    front_count: u64,
    /// The last estimated frame's update.
    last_update: Option<FrameUpdate>,
    /// The frontend on the GPU, when attached; the time of the frame on it,
    /// and a frame's tracks waiting for the estimator.
    gpu: Option<GpuFlow>,
    gpu_submitted: Option<i64>,
    gpu_ready: Option<(i64, visloc_basalt::TrackFrameOutput)>,
    gpu_seq: u64,
    /// The current estimator has connected landmarks.
    connected_once: bool,
    pub still_start: StillStart,
    /// The current estimator passed the still-start gate (or gave up
    /// waiting).
    committed: bool,
    /// Since when the gate has been waiting, and since when the phone has
    /// been still.
    waiting_since_ns: Option<i64>,
    still_since_ns: Option<i64>,
    /// The rotation (IMU at the newest frame in IMU at the gate's reference
    /// frame) and largest angular rate since the reference frame.
    gate_turn: UnitQuaternion<f64>,
    gate_max_gyro: f64,
    /// The previous frame's tracks, for the gate's point motion.
    last_tracks: std::collections::HashMap<u64, (f32, f32)>,
    last_tracks_ns: i64,
    /// Start of the motion samples the next estimator's attitude averages.
    gravity_since_ns: Option<i64>,
    resets: u32,
    sync: TimeSync,
}

impl Session {
    pub fn new(rig: PhoneRig, overrides: Vec<(String, serde_json::Value)>) -> Result<Self, String> {
        let adapter = Self::adapter(&rig, &overrides)?;
        Ok(Self {
            undistort: Undistort::new(&rig),
            sync: TimeSync::new(
                rig.fx,
                rig.fy,
                rig.cx,
                rig.cy,
                &rig.imu_from_camera.rotation,
            ),
            rig,
            overrides,
            adapter,
            imu: VecDeque::new(),
            recent_imu: VecDeque::new(),
            last_frame_ns: None,
            next_frame_id: 0,
            watchdog: Watchdog::default(),
            world_from_estimator: SE3::identity(),
            recent: VecDeque::new(),
            anchor: None,
            unconnected_since_ns: None,
            estimate_every: 1,
            front_seq: 0,
            front_count: 0,
            last_update: None,
            gpu: None,
            gpu_submitted: None,
            gpu_ready: None,
            gpu_seq: 0,
            connected_once: false,
            still_start: StillStart::default(),
            committed: false,
            waiting_since_ns: None,
            still_since_ns: None,
            gate_turn: UnitQuaternion::identity(),
            gate_max_gyro: 0.0,
            last_tracks: std::collections::HashMap::new(),
            last_tracks_ns: 0,
            gravity_since_ns: None,
            resets: 0,
        })
    }

    /// The estimator for `rig`: the phone defaults (the feature grid for its
    /// frame size, the triangulation baseline, the keyframe spacing and
    /// window), then the caller's `overrides`.
    fn adapter(
        rig: &PhoneRig,
        overrides: &[(String, serde_json::Value)],
    ) -> Result<BasaltVioEstimatorAdapter, String> {
        BasaltVioEstimatorAdapter::from_config(
            &rig.calibration()?,
            &config_with(&Self::config(rig, overrides))?,
        )
        .map_err(|e| e.to_string())
    }

    /// The camera calibration and frontend settings the estimator for `rig`
    /// uses ([`Self::new`]), for running its frontend elsewhere.
    pub fn frontend_config(
        rig: &PhoneRig,
        overrides: &[(String, serde_json::Value)],
    ) -> Result<(BasaltCalibration, visloc_basalt::DirectKltConfig), String> {
        let config = config_with(&Self::config(rig, overrides))?;
        let frontend = visloc_basalt::direct_klt_config(&config).map_err(|e| e.to_string())?;
        Ok((rig.calibration()?, frontend))
    }

    /// The phone defaults, then the caller's `overrides`.
    fn config(
        rig: &PhoneRig,
        overrides: &[(String, serde_json::Value)],
    ) -> Vec<(String, serde_json::Value)> {
        let cell = (f64::from(rig.width.min(rig.height)) / GRID_CELLS_ACROSS)
            .round()
            .max(8.0) as u64;
        let mut config = vec![
            (
                "config.optical_flow_detection_grid_size".to_string(),
                serde_json::json!(cell),
            ),
            (
                "config.vio_min_triangulation_dist".to_string(),
                serde_json::json!(MIN_TRIANGULATION_M),
            ),
            (
                "config.vio_min_frames_after_kf".to_string(),
                serde_json::json!(MIN_FRAMES_AFTER_KF),
            ),
            (
                "config.vio_max_states".to_string(),
                serde_json::json!(MAX_STATES),
            ),
            // A phone's motion events come at about 60 Hz, one or two per
            // frame: the steps at each frame's edges are shorter than a
            // sample period, and the pinned noise model undercounts them
            // (see `EstimatorConfig::imu_noise_per_step`).
            (
                "config.vio_imu_noise_per_step".to_string(),
                serde_json::json!(true),
            ),
        ];
        config.extend_from_slice(overrides);
        config
    }

    pub const fn rig(&self) -> &PhoneRig {
        &self.rig
    }

    /// Starts over from the next frame in a new world (a new estimator;
    /// buffered IMU samples are kept).
    pub fn reset(&mut self) -> Result<(), String> {
        self.restart()?;
        self.waiting_since_ns = None;
        self.anchor = None;
        self.world_from_estimator = SE3::identity();
        self.recent.clear();
        Ok(())
    }

    /// A new estimator from the next frame; the caller's world is kept.
    fn restart(&mut self) -> Result<(), String> {
        self.adapter = Self::adapter(&self.rig, &self.overrides)?;
        self.sync.break_chain();
        self.last_frame_ns = None;
        self.next_frame_id = 0;
        self.unconnected_since_ns = None;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.reset();
        }
        self.gpu_submitted = None;
        self.gpu_ready = None;
        self.front_seq = 0;
        self.front_count = 0;
        self.connected_once = false;
        self.committed = false;
        self.still_since_ns = None;
        self.gate_turn = UnitQuaternion::identity();
        self.gate_max_gyro = 0.0;
        self.last_tracks.clear();
        self.gravity_since_ns = None;
        Ok(())
    }

    /// The still-start gate ([`StillStart`]) after a frame of an estimator
    /// that has not passed it: whether the session is still waiting. Restarts
    /// the estimator while the phone moves, and once more (with the attitude
    /// from the still stretch) when it has been still long enough.
    fn still_gate(
        &mut self,
        timestamp_ns: i64,
        frame_id: u64,
        tracks: &[(u64, f32, f32)],
        max_gyro: f64,
        turn: &UnitQuaternion<f64>,
    ) -> Result<bool, String> {
        let gate = self.still_start.clone();
        let waiting_since = *self.waiting_since_ns.get_or_insert(timestamp_ns);
        if frame_id == 0 {
            self.last_tracks = tracks.iter().map(|&(id, x, y)| (id, (x, y))).collect();
            self.last_tracks_ns = timestamp_ns;
            self.gate_turn = UnitQuaternion::identity();
            self.gate_max_gyro = 0.0;
            return Ok(true);
        }
        self.gate_turn *= turn;
        self.gate_max_gyro = self.gate_max_gyro.max(max_gyro);
        if timestamp_ns - self.last_tracks_ns < MIN_GATE_INTERVAL_NS {
            return Ok(true);
        }
        let (turn, max_gyro) = (self.gate_turn, self.gate_max_gyro);
        self.gate_turn = UnitQuaternion::identity();
        self.gate_max_gyro = 0.0;
        // Where each point of the reference frame would be after the
        // rotation alone (points at infinity), against where it was tracked.
        let r = &self.rig;
        let imu_from_camera = r.imu_from_camera.rotation;
        let last_from_this = imu_from_camera.inverse() * turn * imu_from_camera;
        let this_from_last = last_from_this.inverse();
        let mut flow: Vec<f32> = tracks
            .iter()
            .filter_map(|(id, x, y)| {
                let &(px, py) = self.last_tracks.get(id)?;
                let bearing = Vector3::new(
                    (f64::from(px) - r.cx) / r.fx,
                    (f64::from(py) - r.cy) / r.fy,
                    1.0,
                );
                let b = this_from_last * bearing;
                if b.z <= 0.0 {
                    return None;
                }
                let (qx, qy) = (r.fx * b.x / b.z + r.cx, r.fy * b.y / b.z + r.cy);
                Some((f64::from(*x) - qx).hypot(f64::from(*y) - qy) as f32)
            })
            .collect();
        let previous_ns = self.last_tracks_ns;
        self.last_tracks = tracks.iter().map(|&(id, x, y)| (id, (x, y))).collect();
        self.last_tracks_ns = timestamp_ns;
        let still = flow.len() >= gate.min_points && max_gyro <= gate.max_gyro_rad_s && {
            let mid = flow.len() / 2;
            f64::from(*flow.select_nth_unstable_by(mid, f32::total_cmp).1) <= gate.max_flow_px
        };
        if still {
            let since_ns = *self.still_since_ns.get_or_insert(previous_ns);
            if (timestamp_ns - since_ns) as f64 * 1e-9 >= gate.still_s {
                let since = self.still_since_ns;
                self.restart()?;
                self.committed = true;
                self.gravity_since_ns = since;
                self.waiting_since_ns = None;
            }
            return Ok(true);
        }
        if (timestamp_ns - waiting_since) as f64 * 1e-9 >= gate.max_wait_s {
            // Never still: keep this estimator, as without the gate.
            self.committed = true;
            self.waiting_since_ns = None;
            return Ok(false);
        }
        self.restart()?;
        Ok(true)
    }

    /// Seconds without a connected landmark, as of `timestamp_ns`.
    fn unconnected_s(&self, timestamp_ns: i64) -> f64 {
        self.unconnected_since_ns
            .map_or(0.0, |since| (timestamp_ns - since) as f64 * 1e-9)
    }

    /// Why the state after a frame is implausible, if it is.
    fn verdict(
        &self,
        state: &BasaltNavState,
        tracks: usize,
        timestamp_ns: i64,
    ) -> Option<ResetReason> {
        let w = &self.watchdog;
        let finite = state.imu_to_world.translation.iter().all(|v| v.is_finite())
            && state.velocity_world_m_s.iter().all(|v| v.is_finite());
        if !finite {
            return Some(ResetReason::NonFinite);
        }
        if !w.enabled {
            return None;
        }
        if state.velocity_world_m_s.norm() > w.max_speed_m_s {
            Some(ResetReason::Speed)
        } else if state.gyro_bias_rad_s.norm() > w.max_gyro_bias_rad_s {
            Some(ResetReason::GyroBias)
        } else if state.accel_bias_m_s2.norm() > w.max_accel_bias_m_s2 {
            Some(ResetReason::AccelBias)
        } else if self.unconnected_s(timestamp_ns) > w.max_unconnected_s && tracks >= w.min_tracks {
            Some(ResetReason::NoLandmarks)
        } else {
            None
        }
    }

    /// Frames processed since the start or the last reset.
    pub const fn frames(&self) -> u64 {
        self.next_frame_id
    }

    /// Watchdog restarts so far.
    pub const fn resets(&self) -> u32 {
        self.resets
    }

    /// The residual camera-IMU time offset (and focal length error) seen
    /// over the last seconds of motion, once there is enough of it.
    pub fn time_sync(&self) -> Option<SyncEstimate> {
        self.sync.estimate()
    }

    /// Queued IMU samples, and the oldest and newest one's timestamp.
    pub fn imu_queue(&self) -> (usize, Option<i64>, Option<i64>) {
        (
            self.imu.len(),
            self.imu.front().map(|s| s.timestamp_ns),
            self.imu.back().map(|s| s.timestamp_ns),
        )
    }

    /// The last processed frame's timestamp (none since a restart).
    pub const fn last_frame_ns(&self) -> Option<i64> {
        self.last_frame_ns
    }

    /// Queues one IMU sample; samples must come in time order (others are
    /// dropped, returning false).
    pub fn push_imu(&mut self, sample: ImuSample) -> bool {
        if self
            .imu
            .back()
            .is_some_and(|last| sample.timestamp_ns <= last.timestamp_ns)
        {
            return false;
        }
        let finite = sample.gyro_rad_s.iter().all(|v| v.is_finite())
            && sample.accel_m_s2.iter().all(|v| v.is_finite());
        if !finite {
            return false;
        }
        self.imu.push_back(sample);
        self.recent_imu.push_back(sample);
        while self
            .recent_imu
            .front()
            .is_some_and(|s| sample.timestamp_ns - s.timestamp_ns > 1_000_000_000)
        {
            self.recent_imu.pop_front();
        }
        self.sync.push_gyro(sample.timestamp_ns, sample.gyro_rad_s);
        if self.last_frame_ns.is_none() {
            // Before the first frame only the latest few matter.
            while self.imu.len() > 4 * MAX_SAMPLES_BEFORE_FIRST_FRAME {
                self.imu.pop_front();
            }
        }
        true
    }

    /// Whether a frame at `timestamp_ns` can be processed now: it is after
    /// the previous frame and the IMU has reached it.
    pub fn ready(&self, timestamp_ns: i64) -> bool {
        self.last_frame_ns.is_none_or(|last| timestamp_ns > last)
            && self
                .imu
                .back()
                .is_some_and(|last| last.timestamp_ns >= timestamp_ns)
    }

    /// Tracks one frame (8-bit grey, `rig.width x rig.height`, row-major).
    pub fn process(&mut self, timestamp_ns: i64, gray: &[u8]) -> Result<FrameUpdate, String> {
        if !self.ready(timestamp_ns) {
            return Err("frame is not ready (IMU behind or timestamp not increasing)".into());
        }
        let pixels = self.pixels(gray)?;
        let (w, h) = (self.rig.width as usize, self.rig.height as usize);
        let cam0 = RawU16Image::new(w, h, pixels).map_err(|e| e.to_string())?;
        let tracks = self
            .adapter
            .frontend
            .process_frame(StereoFrame::new(self.front_seq, timestamp_ns, cam0, None))
            .map_err(|e| e.to_string())?;
        self.front_seq += 1;
        self.after_frontend(timestamp_ns, tracks)
    }

    /// A frame's tracks, from either frontend: the estimator's update, or
    /// on a frame [`Self::estimate_every`] skips, the last update carried
    /// to this frame's time by the IMU.
    fn after_frontend(
        &mut self,
        timestamp_ns: i64,
        mut tracks: TrackFrameOutput,
    ) -> Result<FrameUpdate, String> {
        let skip = self.estimate_every > 1
            && !self
                .front_count
                .is_multiple_of(u64::from(self.estimate_every))
            && self.last_update.is_some();
        self.front_count += 1;
        if skip {
            return Ok(self.propagated(timestamp_ns, &tracks));
        }
        let p = self.prepare(timestamp_ns)?;
        // The frontend numbers every frame; the estimator only its own.
        tracks.frame = BasaltFrame::new(p.frame_id, timestamp_ns, 0);
        for observation in &mut tracks.observations {
            observation.frame_id = p.frame_id;
        }
        let output = self
            .adapter
            .process_tracks_no_trace(tracks, &p.imu, p.initialization_imu)
            .map_err(|e| e.to_string())?;
        let update = self.finish(p, output)?;
        self.last_update = Some(update.clone());
        Ok(update)
    }

    /// The last update carried to `timestamp_ns` by the IMU samples since
    /// (gravity-aligned world, biases held), with this frame's tracks.
    fn propagated(&self, timestamp_ns: i64, tracks: &TrackFrameOutput) -> FrameUpdate {
        let last = self.last_update.as_ref().expect("checked by the caller");
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let mut state = last.state.clone();
        let mut since = last.timestamp_ns;
        let mut rate = None;
        let step =
            |state: &mut BasaltNavState, gyro: Vector3<f64>, accel: Vector3<f64>, dt: f64| {
                let rotation = state.imu_to_world.rotation;
                let a = rotation * (accel - state.accel_bias_m_s2) + gravity;
                state.imu_to_world.translation += state.velocity_world_m_s * dt + 0.5 * a * dt * dt;
                state.velocity_world_m_s += a * dt;
                state.imu_to_world.rotation = rotation
                    * UnitQuaternion::from_scaled_axis((gyro - state.gyro_bias_rad_s) * dt);
            };
        for sample in &self.imu {
            if sample.timestamp_ns <= since {
                continue;
            }
            let end = sample.timestamp_ns.min(timestamp_ns);
            step(
                &mut state,
                sample.gyro_rad_s,
                sample.accel_m_s2,
                (end - since) as f64 * 1e-9,
            );
            since = end;
            rate = Some((sample.gyro_rad_s, sample.accel_m_s2));
            if since >= timestamp_ns {
                break;
            }
        }
        if let (Some((gyro, accel)), true) = (rate, since < timestamp_ns) {
            step(
                &mut state,
                gyro,
                accel,
                (timestamp_ns - since) as f64 * 1e-9,
            );
        }
        FrameUpdate {
            frame_id: last.frame_id,
            timestamp_ns,
            camera_to_world: state.imu_to_world.compose(&self.rig.imu_from_camera),
            state,
            is_keyframe: false,
            tracks: tracks
                .observations
                .iter()
                .map(|o| (o.track_id, o.pixel.x as f32, o.pixel.y as f32))
                .collect(),
            connected: last.connected,
            track_rejects: tracks
                .reject_counters
                .iter()
                .map(|(reason, &n)| (format!("{reason:?}"), n))
                .collect(),
            imu_samples: 0,
            status: match last.status {
                Status::Reset(_) => Status::Initializing,
                status => status,
            },
            resets: self.resets,
        }
    }

    /// Tracks on the GPU from now on ([`Self::submit_gpu`]); restarts.
    pub fn attach_gpu(&mut self, ctx: GpuContext) -> Result<(), String> {
        let (_, config) = Self::frontend_config(&self.rig, &self.overrides)?;
        let (w, h) = (self.rig.width as usize, self.rig.height as usize);
        self.gpu = Some(GpuFlow::new(ctx, w, h, config)?);
        self.restart()
    }

    /// The GPU frontend, once attached.
    pub const fn gpu(&self) -> Option<&GpuFlow> {
        self.gpu.as_ref()
    }

    /// [`Self::process`] with the frontend on the GPU, in steps: this sends
    /// a frame to the GPU; once `on_mapped` has run, [`Self::take_gpu`]
    /// collects its tracks and [`Self::estimate_gpu`] runs the estimator on
    /// them. The next frame may be submitted between the two, so the GPU
    /// tracks it while the estimator works: its tracking depends only on the
    /// frame before, not on the estimate. A restart (watchdog, still start,
    /// reset) cancels the frame on the GPU: `on_mapped` then gets an error
    /// and the frame is dropped.
    pub fn submit_gpu<F>(
        &mut self,
        timestamp_ns: i64,
        gray: &[u8],
        on_mapped: F,
    ) -> Result<(), String>
    where
        F: FnOnce(Result<(), wgpu::BufferAsyncError>) + wgpu::WasmNotSend + 'static,
    {
        if self.gpu_submitted.is_some() {
            return Err("a frame is already on the GPU".into());
        }
        let newest = self
            .gpu_ready
            .as_ref()
            .map(|(t, _)| *t)
            .or(self.last_frame_ns);
        if newest.is_some_and(|newest| timestamp_ns <= newest) {
            return Err("frame timestamp is not increasing".into());
        }
        let pixels = self.pixels(gray)?;
        let seq = self.gpu_seq;
        self.gpu.as_mut().ok_or("no GPU frontend")?.submit(
            seq,
            timestamp_ns,
            &pixels,
            on_mapped,
        )?;
        self.gpu_seq += 1;
        self.gpu_submitted = Some(timestamp_ns);
        Ok(())
    }

    /// Collects the tracks of the frame on the GPU (after its `on_mapped`);
    /// false if a restart cancelled it. Then [`Self::estimate_gpu`].
    pub fn take_gpu(&mut self) -> Result<bool, String> {
        let Some(timestamp_ns) = self.gpu_submitted.take() else {
            return Ok(false);
        };
        if self.gpu_ready.is_some() {
            return Err("the previous frame's tracks were not estimated".into());
        }
        match self.gpu.as_mut().ok_or("no GPU frontend")?.finish() {
            Ok(tracks) => {
                self.gpu_ready = Some((timestamp_ns, tracks));
                Ok(true)
            }
            Err(e) => {
                // The CPU frontend takes over from the next frame.
                self.gpu = None;
                self.restart()?;
                Err(format!("{e}; tracking on the CPU from now on"))
            }
        }
    }

    /// Runs the estimator on the tracks [`Self::take_gpu`] collected (None
    /// if there are none).
    pub fn estimate_gpu(&mut self) -> Result<Option<FrameUpdate>, String> {
        let Some((timestamp_ns, tracks)) = self.gpu_ready.take() else {
            return Ok(None);
        };
        self.after_frontend(timestamp_ns, tracks).map(Some)
    }

    /// [`Self::take_gpu`] then [`Self::estimate_gpu`], for a frame
    /// submitted alone.
    pub fn finish_gpu(&mut self) -> Result<FrameUpdate, String> {
        if !self.take_gpu()? {
            return Err("the frame was cancelled by a restart".into());
        }
        self.estimate_gpu()?
            .ok_or_else(|| "no tracks to estimate".into())
    }

    /// Natively, the browser worker's GPU pipeline: collects the frame on
    /// the GPU, submits this one, then estimates the collected frame while
    /// the GPU works. Returns the previous frame's update (None for the
    /// first frame, or one a restart cancelled); [`Self::flush_gpu`] gives
    /// the last.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn process_gpu_pipelined(
        &mut self,
        timestamp_ns: i64,
        gray: &[u8],
    ) -> Result<Option<FrameUpdate>, String> {
        if self.gpu_submitted.is_some() {
            self.gpu.as_ref().ok_or("no GPU frontend")?.wait();
            self.take_gpu()?;
        }
        self.submit_gpu(timestamp_ns, gray, |_| {})?;
        self.estimate_gpu()
    }

    /// The last frame of [`Self::process_gpu_pipelined`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn flush_gpu(&mut self) -> Result<Option<FrameUpdate>, String> {
        if self.gpu_submitted.is_some() {
            self.gpu.as_ref().ok_or("no GPU frontend")?.wait();
            self.take_gpu()?;
        }
        self.estimate_gpu()
    }

    /// A frame's pixels as the frontend takes them (undistorted when the
    /// rig has lens distortion).
    fn pixels(&self, gray: &[u8]) -> Result<Vec<u16>, String> {
        let (w, h) = (self.rig.width as usize, self.rig.height as usize);
        if gray.len() != w * h {
            return Err(format!("frame is {} bytes, expected {w}x{h}", gray.len()));
        }
        Ok(match &self.undistort {
            Some(undistort) => undistort.apply(gray),
            None => gray.iter().map(|&v| u16::from(v) << 8).collect(),
        })
    }

    /// A frame's IMU bookkeeping, before the estimator runs on it.
    fn prepare(&mut self, timestamp_ns: i64) -> Result<Prepared, String> {
        if !self.ready(timestamp_ns) {
            return Err("frame is not ready (IMU behind or timestamp not increasing)".into());
        }
        let first = self.last_frame_ns.is_none();
        let mut imu: Vec<ImuSample> = Vec::new();
        while let Some(sample) = self.imu.front() {
            if sample.timestamp_ns > timestamp_ns {
                break;
            }
            imu.push(*sample);
            self.imu.pop_front();
        }
        if first && imu.len() > MAX_SAMPLES_BEFORE_FIRST_FRAME {
            imu.drain(..imu.len() - MAX_SAMPLES_BEFORE_FIRST_FRAME);
        }
        // Close the interval at the frame with the next sample's values, its
        // time set to the frame's, as upstream Basalt does
        // (sqrt_keypoint_vio.cpp, processing loop). The port otherwise holds
        // the last sample's values to the frame, and an interval with no
        // sample at all (a 60 Hz IMU against 30 fps frames, or the first
        // sample arriving after the second frame) would get no IMU factor,
        // cutting the window in two.
        if !first
            && imu
                .last()
                .is_none_or(|last| last.timestamp_ns < timestamp_ns)
        {
            if let Some(&next) = self.imu.front() {
                imu.push(ImuSample::new(
                    timestamp_ns,
                    next.gyro_rad_s,
                    next.accel_m_s2,
                ));
            }
        }
        let initialization_imu = if first {
            self.imu
                .front()
                .map(|&next| averaged_gravity(self.recent_imu.iter(), next, self.gravity_since_ns))
        } else {
            None
        };
        let frame_id = self.next_frame_id;
        let imu_samples = imu.len();
        let max_gyro = imu.iter().map(|s| s.gyro_rad_s.norm()).fold(0.0, f64::max);
        // The interval's rotation (IMU at this frame in IMU at the last),
        // each sample's rate covering the time since the one before.
        let mut turn = UnitQuaternion::identity();
        let mut since = self.last_frame_ns.unwrap_or(timestamp_ns);
        for sample in &imu {
            if sample.timestamp_ns > since {
                turn *= UnitQuaternion::from_scaled_axis(
                    sample.gyro_rad_s * ((sample.timestamp_ns - since) as f64 * 1e-9),
                );
                since = sample.timestamp_ns;
            }
        }
        Ok(Prepared {
            frame_id,
            timestamp_ns,
            imu,
            initialization_imu,
            imu_samples,
            max_gyro,
            turn,
        })
    }

    /// The rest of a frame after tracking: the still-start gate, the
    /// watchdog and the update.
    fn finish(&mut self, p: Prepared, output: BasaltAdapterOutput) -> Result<FrameUpdate, String> {
        let Prepared {
            frame_id,
            timestamp_ns,
            imu_samples,
            max_gyro,
            turn,
            ..
        } = p;
        self.next_frame_id += 1;
        self.last_frame_ns = Some(timestamp_ns);
        let mut state = output.estimator.state;
        let connected = output.estimator.connected_cam0;
        let tracks: Vec<(u64, f32, f32)> = output
            .tracks
            .observations
            .iter()
            .map(|o| (o.track_id, o.pixel.x as f32, o.pixel.y as f32))
            .collect();
        let track_rejects = output
            .tracks
            .reject_counters
            .iter()
            .map(|(reason, &n)| (format!("{reason:?}"), n))
            .collect();
        self.sync.push_frame(timestamp_ns, &tracks);
        // A restarted estimator's first frame: place its world so this pose
        // has the anchor's position and heading.
        if frame_id == 0 {
            self.world_from_estimator = match &self.anchor {
                Some(anchor) => {
                    let yaw = yaw_part(&(anchor.rotation * state.imu_to_world.rotation.inverse()));
                    let position = anchor.translation - yaw * state.imu_to_world.translation;
                    SE3::new(yaw, position)
                }
                None => SE3::identity(),
            };
        }
        if connected == 0 {
            self.unconnected_since_ns.get_or_insert(timestamp_ns);
        } else {
            self.unconnected_since_ns = None;
        }
        let waiting = self.still_start.enabled
            && !self.committed
            && self.still_gate(timestamp_ns, frame_id, &tracks, max_gyro, &turn)?;
        let verdict = if waiting {
            None
        } else {
            self.verdict(&state, tracks.len(), timestamp_ns)
        };
        state.imu_to_world = self.world_from_estimator.compose(&state.imu_to_world);
        state.velocity_world_m_s = self.world_from_estimator.rotation * state.velocity_world_m_s;
        let status = match verdict {
            None if waiting => Status::Waiting,
            Some(reason) => {
                self.resets += 1;
                self.anchor = self
                    .recent
                    .front()
                    .map(|(_, pose)| pose.clone())
                    .or_else(|| self.anchor.clone());
                self.recent.clear();
                self.restart()?;
                Status::Reset(reason)
            }
            None => {
                self.recent
                    .push_back((timestamp_ns, state.imu_to_world.clone()));
                let back_ns = (self.watchdog.anchor_back_s * 1e9) as i64;
                while self
                    .recent
                    .get(1)
                    .is_some_and(|(t, _)| *t <= timestamp_ns - back_ns)
                {
                    self.recent.pop_front();
                }
                self.connected_once |= connected > 0;
                if connected > 0
                    || (self.connected_once && self.unconnected_s(timestamp_ns) < COAST_AFTER_S)
                {
                    Status::Tracking
                } else if self.connected_once {
                    Status::Coasting
                } else {
                    Status::Initializing
                }
            }
        };
        let camera_to_world = state.imu_to_world.compose(&self.rig.imu_from_camera);
        Ok(FrameUpdate {
            frame_id,
            timestamp_ns,
            camera_to_world,
            is_keyframe: output.estimator.is_keyframe,
            tracks,
            connected,
            track_rejects,
            imu_samples,
            state,
            status,
            resets: self.resets,
        })
    }

    /// World positions of the landmarks in the window.
    pub fn landmarks(&self) -> Vec<[f32; 3]> {
        self.adapter
            .estimator
            .landmark_positions_world()
            .into_iter()
            .map(|(_, p)| {
                let p = self.world_from_estimator.transform_point(&p.into());
                [p.x as f32, p.y as f32, p.z as f32]
            })
            .collect()
    }
}
