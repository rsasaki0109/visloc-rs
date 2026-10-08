use super::aom::{
    q2_f32_normal_system, set_active_diagnostic_projection_event, sophus_rotate_step_packet_f32,
    DiagnosticProjectionEvent, LmConfig, WhitenedFactorRowStack,
};
use super::landmarks::{
    triangulate_dlt, triangulate_dlt_rig_f32, triangulate_dlt_rig_f32_traced, BearingObservation,
    InverseDistanceLandmark, LandmarkStatus, NativeHostOrder, ObservationDb,
    StereographicDirection, TriangulationDltTraceF32,
};
use super::margdata::{
    AomBlockData, FramePoseData, FrameStateData, MargData, MargDataDiagnosticSidecar,
    MarginalizationTargets, MatrixData, NavStateData, OfImageData, OfObservationData,
    PoseStateWithLinData, PoseVelBiasStateWithLinData, PriorData, WindowPolicy,
    MARGDATA_SCHEMA_VERSION,
};
use super::scalar::ScalarMode;
use super::window::{
    diagnostic_env_active, diagnostic_env_snapshot, flatten_nav_with_mode, flatten_pose_with_mode,
    marginalize_mixed_prior_with_mode, projected_rows, projected_rows_native_q2, stack_rows,
    take_diagnostic_prior_pre, visual_state_pose_with_mode, write_blocks_with_mode,
    WindowBlockKind, WindowDiagnostics, WindowImuLink, WindowLandmark, WindowObservation,
    WindowPose, WindowPrior, WindowProblem, WindowSolveResult, WindowState, NAV_STATE_DOF,
    POSE_DOF,
};
use crate::camera::DoubleSphereCamera;
use crate::imu::{
    integrate_between, BiasRandomWalkNoise, ImuDStateUpdateTrace, ImuNoiseModel,
    ImuPreintegratedDelta, ImuPreintegrator,
};
use crate::streaming::{BasaltStream, VisionFrame, WindowPhase};
use crate::{BasaltNavState, ImuSample, TimingBreakdown, TimingBucket, TrackId, TrackObservation};
use nalgebra::{DVector, Matrix3, Point2, Point3, Quaternion, SMatrix, UnitQuaternion, Vector3};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::OpenOptions,
    io::Write as IoWrite,
    path::Path,
};
use visloc_core::geometry::SE3;

type F32Matrix9 = SMatrix<f32, 9, 9>;
type F32Matrix9x3 = SMatrix<f32, 9, 3>;

// Diagnostic-only counter for the native assignment-term comparison.  It is
// read only when VISLOC_BASALT_M7HD_ASSIGN_DUMP is set and does not affect
// the production covariance path.
static M7HD_ASSIGN_DUMP_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatorConfig {
    /// Numeric ownership of the compatibility core.  Public sensor and
    /// trajectory values remain f64 in either mode.
    pub scalar_mode: ScalarMode,
    pub window: WindowPolicy,
    /// Upstream `vio_min_frames_after_kf`.  The pinned implementation uses a
    /// strict `frames_after_kf > value` comparison, so the EuRoC default of 5
    /// produces keyframes no closer than seven frame indices apart.
    pub min_frames_after_kf: u64,
    /// Upstream `vio_new_kf_keypoints_thresh`, applied to camera-0 tracks
    /// that already have a landmark versus all accepted camera-0 tracks.
    pub new_kf_keypoints_threshold: f64,
    /// Opt-in adaptive keyframe spacing (not upstream).  When the connected
    /// ratio falls below `threshold`, a keyframe may be taken once
    /// `frames_after_kf > min_frames_after_kf` instead of the regular
    /// spacing.  `None` keeps the upstream decision unchanged.
    pub urgent_kf: Option<UrgentKeyframePolicy>,
    pub solver: LmConfig,
    /// Basalt's initial square-root prior weights.  The upstream names are
    /// retained even though its `ba`/`bg` assignments follow the state-block
    /// order in `sqrt_keypoint_vio.cpp` (gyro columns 9..11, accel columns
    /// 12..14).
    pub initial_pose_weight: f64,
    pub initial_accel_bias_weight: f64,
    pub initial_gyro_bias_weight: f64,
    /// Opt-in (not upstream; `config.vio_landmarks_all_cameras`).  When a
    /// keyframe is taken, an unconnected track observed in the current frame
    /// by a camera other than cam0 -- and not by cam0 -- also becomes a
    /// landmark candidate, hosted in the lowest such camera and triangulated
    /// from that track's retained multi-frame history.  `false` keeps the
    /// pinned cam0-hosted landmark creation.
    pub landmarks_all_cameras: bool,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            scalar_mode: ScalarMode::default(),
            window: WindowPolicy::default(),
            min_frames_after_kf: 5,
            new_kf_keypoints_threshold: 0.7,
            urgent_kf: None,
            solver: LmConfig::default(),
            initial_pose_weight: 1.0e8,
            initial_accel_bias_weight: 1.0e1,
            initial_gyro_bias_weight: 1.0e2,
            landmarks_all_cameras: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UrgentKeyframePolicy {
    pub threshold: f64,
    pub min_frames_after_kf: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImuPropagationTrace {
    pub interval_start_ns: i64,
    pub interval_end_ns: i64,
    pub sample_timestamps_ns: Vec<i64>,
    pub delta_time: f64,
    pub delta_position: Vector3<f64>,
    pub delta_rotation: UnitQuaternion<f64>,
    pub delta_velocity: Vector3<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorStateTrace {
    /// The latest state before this camera interval was integrated.  It is
    /// absent only for the first camera, where Basalt initializes directly
    /// from the first calibrated IMU packet.
    pub state_from: Option<BasaltNavState>,
    /// The first-camera initialization output, when this is frame zero.
    pub initialization_output: Option<BasaltNavState>,
    /// The newest state inserted into the active window before optimization.
    pub predicted_state: BasaltNavState,
    /// The newest state after LM writeback (and any subsequent window shift).
    pub post_opt_state: BasaltNavState,
    /// Optional live f32 IMU propagation boundary.  This is diagnostic data
    /// only; the solver consumes the same delta regardless of whether callers
    /// serialize this field.
    pub imu_propagation: Option<ImuPropagationTrace>,
    pub imu_integration_fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorOutput {
    pub frame_id: u64,
    pub state: BasaltNavState,
    pub is_keyframe: bool,
    pub connected_cam0: usize,
    pub unconnected_cam0: usize,
    /// Number of post-marginalization 15-DoF navigation states.
    pub active_state_count: usize,
    /// Number of retained 6-DoF keyframe pose blocks after marginalization.
    pub active_pose_count: usize,
    pub state_trace: EstimatorStateTrace,
    pub phases: Vec<WindowPhase>,
    pub marg_data: MargData,
    pub window: WindowDiagnostics,
    /// True when the half-open sensor interval could not be interpolated at
    /// its camera boundary and used the Basalt queue-compatible fallback.
    pub imu_integration_fallback: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct TrackHistory {
    /// Explicit frame id corresponding to upstream `TimeCamId::frame_id`.
    /// The timestamp remains available for the sensor-side ordering/debug
    /// contract, but landmark hosting and active-window membership are keyed
    /// by this id in the Rust window.
    frame_id: u64,
    camera_id: u16,
    timestamp_ns: i64,
    pixel: Point2<f64>,
    /// Raw (non-unit) Double-Sphere unprojection used by Basalt's DLT.
    raw_bearing: Vector3<f64>,
    /// Keep the IMU pose separate from the camera pose.  Upstream forms
    /// `T_i_c[0]^-1 * (T_w_i[0]^-1 * T_w_i[1]) * T_i_c[1]` in Scalar=float;
    /// reconstructing that relative transform from a pre-composed f64 camera
    /// pose changes the f32 operation boundary.
    imu_pose: SE3,
    pose: SE3,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StoredObservation {
    frame_id: u64,
    camera_id: u16,
    pixel: Point2<f64>,
}

/// All fallible work for a newly triangulated track is prepared before the
/// packet's mutable stores are changed.  This keeps a later malformed track
/// from leaving earlier tracks half-committed while preserving the existing
/// candidate and association order at commit time.
#[derive(Debug)]
struct PreparedLandmarkInsertion {
    track_id: TrackId,
    host_timestamp_ns: i64,
    point_world: Point3<f64>,
    landmark: InverseDistanceLandmark,
    factor_observations: Vec<StoredObservation>,
    candidate_bearings: Vec<(TrackHistory, Vector3<f64>)>,
}

/// A diagnostic-only snapshot of the estimator-owned stores. The maps hold
/// only track IDs here; numeric/state values remain in the estimator and are
/// never copied for this sidecar unless the explicit path is configured.
#[derive(Debug, Clone)]
struct LifecycleTraceSnapshot {
    frame_id: u64,
    track_history_ids: BTreeSet<TrackId>,
    factor_ids: BTreeSet<TrackId>,
    landmark_ids: BTreeSet<TrackId>,
    world_ids: BTreeSet<TrackId>,
    track_history_entries: usize,
    track_history_capacity: usize,
    factor_records: usize,
    factor_capacity: usize,
    landmark_count: usize,
    landmark_observation_records: usize,
    landmark_observation_capacity: usize,
    world_entries: usize,
}

#[derive(Debug, Clone, Copy, Default)]
struct LifecycleTracePeak {
    track_history_tracks: usize,
    track_history_entries: usize,
    track_history_capacity: usize,
    factor_tracks: usize,
    factor_records: usize,
    factor_capacity: usize,
    landmark_count: usize,
    landmark_observation_records: usize,
    landmark_observation_capacity: usize,
    world_entries: usize,
}

#[derive(Debug, Clone, Default)]
struct MarginalizationPlan {
    /// Full state IDs dropped entirely.
    drop_states: Vec<u64>,
    /// Keyframe state IDs whose velocity/bias leave but whose pose survives.
    convert_states: Vec<u64>,
    /// Existing pose-only keyframes removed by the max-KF policy.
    drop_poses: Vec<u64>,
    /// The inclusive FEJ boundary block retained in the marginalization AOM.
    /// Upstream excludes every newer state (in particular the just-arrived
    /// latest state) from this AOM, even though those states are present in
    /// the packet's full `frame_states` table.
    boundary_state: Option<u64>,
}

fn json_f32_values(values: &[f32]) -> serde_json::Value {
    json!({
        "values": values
            .iter()
            .map(|value| value.is_finite().then_some(*value))
            .collect::<Vec<_>>(),
        "bits": values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
    })
}

fn json_f64_values(values: &[f64]) -> serde_json::Value {
    json!({
        "values": values
            .iter()
            .map(|value| value.is_finite().then_some(*value))
            .collect::<Vec<_>>(),
        "bits": values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
    })
}

fn json_se3_triangulation_trace(pose: &SE3) -> serde_json::Value {
    let quaternion = pose.rotation.quaternion();
    let translation = [pose.translation.x, pose.translation.y, pose.translation.z];
    let rotation_wxyz = [quaternion.w, quaternion.i, quaternion.j, quaternion.k];
    let translation_f32 = translation.map(|value| value as f32);
    let rotation_f32 = rotation_wxyz.map(|value| value as f32);
    json!({
        "translation": json_f64_values(&translation),
        "rotation_wxyz": json_f64_values(&rotation_wxyz),
        "f32_cast": {
            "translation": json_f32_values(&translation_f32),
            "rotation_wxyz": json_f32_values(&rotation_f32),
        },
    })
}

fn json_track_history_triangulation_trace(history: &TrackHistory) -> serde_json::Value {
    let pixel = [history.pixel.x, history.pixel.y];
    let raw_bearing = [
        history.raw_bearing.x,
        history.raw_bearing.y,
        history.raw_bearing.z,
    ];
    json!({
        "frame_id": history.frame_id,
        "camera_id": history.camera_id,
        "timestamp_ns": history.timestamp_ns,
        "pixel": json_f64_values(&pixel),
        "raw_bearing": json_f64_values(&raw_bearing),
        "imu_pose_current_raw": json_se3_triangulation_trace(&history.imu_pose),
        "camera_pose_effective": json_se3_triangulation_trace(&history.pose),
        "linearized_pose": serde_json::Value::Null,
        "linearized_pose_available": false,
    })
}

fn json_optional_f32_values<const N: usize>(values: Option<&[f32; N]>) -> serde_json::Value {
    values
        .map(|values| json_f32_values(values))
        .unwrap_or(serde_json::Value::Null)
}

fn json_triangulation_dlt_trace(trace: &TriangulationDltTraceF32) -> serde_json::Value {
    json!({
        "input_valid": trace.input_valid,
        "target_bearing": json_f32_values(&trace.target_bearing),
        "candidate_bearing": json_f32_values(&trace.candidate_bearing),
        "relative_imu_qxyzw_t": json_f32_values(&trace.relative_imu_qxyzw_t),
        "camera_prefix_qxyzw_t": json_f32_values(&trace.camera_prefix_qxyzw_t),
        "target_camera_from_candidate_qxyzw_t": json_f32_values(&trace.target_camera_from_candidate_qxyzw_t),
        "candidate_pose_qxyzw_t": json_f32_values(&trace.candidate_pose_qxyzw_t),
        "candidate_rotation_row_major": json_f32_values(&trace.candidate_rotation),
        "candidate_matrix_row_major": json_f32_values(&trace.candidate_matrix),
        "dlt_matrix_row_major": json_f32_values(&trace.dlt_matrix),
        "scale": trace.scale.map(|value| json_f32_values(&[value])),
        "scaled_matrix_row_major": json_optional_f32_values(trace.scaled_matrix.as_ref()),
        "singular_values_scaled": json_optional_f32_values(trace.singular_values_scaled.as_ref()),
        "singular_values": json_optional_f32_values(trace.singular_values.as_ref()),
        "rank_estimate": trace.rank_estimate,
        "homogeneous_raw": json_optional_f32_values(trace.homogeneous_raw.as_ref()),
        "direction_norm": trace
            .direction_norm
            .map(|value| json_f32_values(&[value])),
        "homogeneous_normalized": json_optional_f32_values(
            trace.homogeneous_normalized.as_ref(),
        ),
        "direction_before_sign": json_optional_f32_values(
            trace.direction_before_sign.as_ref(),
        ),
        "rho_before_sign": trace
            .rho_before_sign
            .map(|value| json_f32_values(&[value])),
        "sign_flipped": trace.sign_flipped,
        "direction_after_sign": json_optional_f32_values(trace.direction_after_sign.as_ref()),
        "rho_after_sign": trace
            .rho_after_sign
            .map(|value| json_f32_values(&[value])),
    })
}

fn json_triangulation_result(result: Option<(Vector3<f32>, f32)>) -> serde_json::Value {
    result
        .map(|(direction, rho)| {
            let values = [direction.x, direction.y, direction.z, rho];
            json_f32_values(&values)
        })
        .unwrap_or(serde_json::Value::Null)
}

fn append_triangulation_trace(
    path: &Path,
    event_frame_id: u64,
    event_timestamp_ns: i64,
    track_id: TrackId,
    target_history: &TrackHistory,
    complete_history: &[TrackHistory],
    attempts: &[serde_json::Value],
    selected_attempt: Option<usize>,
    scalar_mode: ScalarMode,
) {
    let record = json!({
        "schema": "basalt.m11.rust_triangulation_attempt.v1",
        "source": "rust",
        "event_frame_id": event_frame_id,
        "event_timestamp_ns": event_timestamp_ns,
        "track_id": track_id,
        "scalar_mode": format!("{scalar_mode:?}"),
        "target_history": json_track_history_triangulation_trace(target_history),
        "complete_history": complete_history
            .iter()
            .map(json_track_history_triangulation_trace)
            .collect::<Vec<_>>(),
        "attempts": attempts,
        "selected_attempt": selected_attempt,
        "capture_contract": {
            "history_order": "timestamp_ns,camera_id,frame_id",
            "target_bearing": "normalized_unprojection_f32",
            "candidate_bearing": "raw_unprojection_f32",
            "pose_fields": "imu_pose_current_raw+camera_pose_effective; linearized pose is not retained at delayed triangulation boundary",
            "dlt_layout": "row_major",
            "singular_value_scale": "both normalized and original matrix magnitudes when SVD succeeds",
            "production_arithmetic_changed": false,
        },
    });
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        if let Ok(line) = serde_json::to_string(&record) {
            let _ = file.write_all(line.as_bytes());
            let _ = file.write_all(b"\n");
        }
    }
}

/// Basalt's active-window estimator boundary.
///
/// The public `states`/`nav` fields retain the existing API. Internally the
/// solver keeps every active navigation state in one N×15 vector and groups
/// all observations of a track before landmark QR elimination.
#[derive(Debug)]
pub struct BasaltVioEstimator {
    pub camera: DoubleSphereCamera,
    pub config: EstimatorConfig,
    pub stream: BasaltStream,
    pub landmarks: ObservationDb,
    pub states: Vec<FrameStateData>,
    pub nav: BasaltNavState,
    camera_models: Vec<DoubleSphereCamera>,
    t_imu_cam: Vec<SE3>,
    imu_noise: ImuNoiseModel,
    bias_walk_noise: BiasRandomWalkNoise,
    calib_accel_bias: Vec<f64>,
    calib_gyro_bias: Vec<f64>,
    last_timestamp_ns: Option<i64>,
    last_rows: Vec<WhitenedFactorRowStack>,
    /// Active pose-only keyframes, ordered before `window_states` in the
    /// mixed absolute order map.
    window_poses: Vec<WindowPose>,
    window_states: Vec<WindowState>,
    /// Raw optical-flow input images retained only while their frame is an
    /// active keyframe.  The adapter moves one copied payload into this map;
    /// no image is copied by the solver or by MargData emission.
    of_images: BTreeMap<u64, Vec<OfImageData>>,
    /// Complete raw feature observations retained alongside each keyframe's
    /// optical-flow images. These mapper payloads are independent of the
    /// landmark subset that remains active in the VIO factor graph.
    of_observations: BTreeMap<u64, Vec<OfObservationData>>,
    window_observations: BTreeMap<TrackId, Vec<StoredObservation>>,
    native_host_order: NativeHostOrder,
    native_host_keys: BTreeMap<(u64, u16), u64>,
    track_history: BTreeMap<TrackId, Vec<TrackHistory>>,
    landmark_world: BTreeMap<TrackId, Point3<f64>>,
    imu_links: Vec<(u64, u64, ImuPreintegratedDelta)>,
    prior: Option<WindowPrior>,
    anchor_point: Option<DVector<f64>>,
    gravity_world: Option<Vector3<f64>>,
    /// Mirrors upstream `SqrtKeypointVioEstimator::opt_started`: the first
    /// four inserted states are prediction/observation accumulation only;
    /// optimization starts once the fifth state makes `frame_states.size() >
    /// 4` at the pinned Basalt commit `0f3b2b52`.
    opt_started: bool,
    /// Mirrors upstream's one-shot `take_kf` flag.  It starts true so frame 0
    /// is a keyframe independently of its connected-track ratio.
    take_kf: bool,
    /// Number of non-keyframes processed since the last keyframe.  This is a
    /// counter rather than a frame-id subtraction because upstream increments
    /// it only in the non-keyframe branch.
    frames_after_kf: u64,
    /// Number of landmarks created while each keyframe was inserted.  The
    /// upstream old-KF policy uses this denominator for its feature ratio.
    num_points_kf: BTreeMap<u64, usize>,
    /// Once explicit no-MargData/no-trace processing starts, older active
    /// keyframes may not have complete `OfImageData` or raw feature records.
    /// Retaining MargData again would therefore emit an incomplete packet, so
    /// that transition is rejected until the estimator is recreated.
    lean_no_output_mode: bool,
    /// Opt-in: when a frame retains MargData, use the compact f32 LM
    /// preparation path (no per-trial landmark re-factorization and no
    /// diagnostic prepass) while still retaining the final factor snapshot the
    /// packet needs.  The accepted trajectory and the MargData bytes are
    /// identical to the diagnostic path (see
    /// `lean_with_factors_matches_retained_diagnostics_exactly`); only the
    /// `window.lm` diagnostic trace payload is omitted.  Default off so
    /// diagnostic/provenance callers keep their existing behavior.
    lean_marg_data: bool,
    /// Marginalization bookkeeping exposed in the next MargData artifact.
    last_kf_to_marg: Vec<(u64, u64)>,
    last_marg_targets: MarginalizationTargets,
    last_lost_landmarks: Vec<u64>,
    /// Captured before the active window is shifted.  Upstream emits
    /// MargData before erasing selected keyframes, so a post-shift snapshot is
    /// semantically wrong even when its dimensions happen to validate.
    pending_marg_data: Option<MargData>,
    /// Prior present immediately before the current marginalization.  This is
    /// populated only for the explicit MargData boundary sidecar, so the
    /// ordinary solver and packet path do not clone a dense prior.
    pending_marg_prior_pre: Option<PriorData>,
    /// Native getPose() values captured from the same pre-shift problem as
    /// the pending MargData packet.  These are diagnostic-only and are kept
    /// outside MargData so packet serialization and its stable hash remain
    /// unchanged.
    pending_marg_effective_pose_fej: BTreeMap<i64, [f64; 7]>,
    pending_marg_effective_state_fej: BTreeMap<i64, [f64; 7]>,
    /// Count of emitted mapper MargData events in this estimator instance.
    /// It is used only as the sidecar join ordinal.
    marg_event_ordinal: usize,
    /// Previous/peak values for the opt-in lifecycle sidecar. These fields
    /// stay inert when `VISLOC_BASALT_LIFECYCLE_TRACE` is not configured.
    lifecycle_trace_previous: Option<LifecycleTraceSnapshot>,
    lifecycle_trace_peak: LifecycleTracePeak,
    /// Optional phase timers merged into the adapter's timing sidecar.
    /// `TimingBreakdown::from_env` is disabled by default and only reads
    /// `VISLOC_BASALT_TIMING_BREAKDOWN` when Cargo feature
    /// `basalt-timing-breakdown` is compiled. `write_json` is a
    /// feature-gated sidecar operation; an environment value alone never
    /// enables timing in the default build.
    timing: TimingBreakdown,
}

impl BasaltVioEstimator {
    fn prune_native_host_order(&mut self) {
        let active = self
            .window_observations
            .iter()
            .filter_map(|(track, observations)| {
                if observations.is_empty() {
                    return None;
                }
                self.landmarks.landmarks.get(track).map(|record| {
                    (
                        record.landmark.anchor_pose,
                        record.landmark.anchor_camera_id,
                    )
                })
            })
            .collect::<BTreeSet<_>>();
        self.native_host_keys.retain(|&(frame, camera), timestamp| {
            if active.contains(&(frame, camera)) {
                return true;
            }
            self.native_host_order.remove(*timestamp, camera);
            false
        });
    }

    pub fn new(camera: DoubleSphereCamera, config: EstimatorConfig) -> Self {
        Self {
            camera,
            config,
            stream: BasaltStream::new(false),
            landmarks: ObservationDb::default(),
            states: Vec::new(),
            nav: BasaltNavState::default(),
            camera_models: vec![camera],
            t_imu_cam: vec![SE3::identity()],
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            calib_accel_bias: vec![0.0; 9],
            calib_gyro_bias: vec![0.0; 12],
            last_timestamp_ns: None,
            last_rows: Vec::new(),
            window_poses: Vec::new(),
            window_states: Vec::new(),
            of_images: BTreeMap::new(),
            of_observations: BTreeMap::new(),
            window_observations: BTreeMap::new(),
            native_host_order: NativeHostOrder::default(),
            native_host_keys: BTreeMap::new(),
            track_history: BTreeMap::new(),
            landmark_world: BTreeMap::new(),
            imu_links: Vec::new(),
            prior: None,
            anchor_point: None,
            // Basalt's VIO constructor uses the fixed world gravity vector;
            // it is not estimated from the first packet.  Keeping it as an
            // option preserves the existing test/configuration seam, while
            // the production default is the upstream constant.
            gravity_world: Some(Vector3::new(0.0, 0.0, -9.81)),
            opt_started: false,
            take_kf: true,
            frames_after_kf: 0,
            num_points_kf: BTreeMap::new(),
            lean_no_output_mode: false,
            lean_marg_data: false,
            last_kf_to_marg: Vec::new(),
            last_marg_targets: MarginalizationTargets::default(),
            last_lost_landmarks: Vec::new(),
            pending_marg_data: None,
            pending_marg_prior_pre: None,
            pending_marg_effective_pose_fej: BTreeMap::new(),
            pending_marg_effective_state_fej: BTreeMap::new(),
            marg_event_ordinal: 0,
            lifecycle_trace_previous: None,
            lifecycle_trace_peak: LifecycleTracePeak::default(),
            timing: TimingBreakdown::from_env(),
        }
    }

    /// Returns the cumulative internal timing snapshot for the adapter sidecar.
    pub(crate) const fn timing_breakdown(&self) -> &TimingBreakdown {
        &self.timing
    }

    /// Enables the compact MargData LM path (byte-identical trajectory and
    /// packet, no `window.lm` diagnostic trace).  See the `lean_marg_data`
    /// field for the contract.
    pub fn set_lean_marg_data(&mut self, enabled: bool) {
        self.lean_marg_data = enabled;
    }

    /// Replaces the compatibility camera-0 setup with the complete calibrated
    /// camera rig.  `t_imu_cam[c]` maps camera-frame points to IMU coordinates;
    /// all visual geometry then uses `T_w_c = T_w_i * T_imu_cam[c]`.
    pub fn with_camera_rig(
        mut self,
        cameras: Vec<DoubleSphereCamera>,
        t_imu_cam: Vec<SE3>,
    ) -> Result<Self, String> {
        if cameras.is_empty() || cameras.len() != t_imu_cam.len() {
            return Err(format!(
                "camera rig requires equal non-empty camera/extrinsic arrays, got {} and {}",
                cameras.len(),
                t_imu_cam.len()
            ));
        }
        self.camera = cameras[0];
        self.camera_models = cameras;
        self.t_imu_cam = t_imu_cam;
        Ok(self)
    }

    /// Sets the calibrated continuous-time measurement and bias random-walk
    /// densities used by preintegration and active-window whitening.
    pub fn with_imu_noise(
        mut self,
        imu_noise: ImuNoiseModel,
        bias_walk_noise: BiasRandomWalkNoise,
    ) -> Result<Self, String> {
        if !imu_noise.is_valid()
            || imu_noise.gyro_density <= 0.0
            || imu_noise.accel_density <= 0.0
            || !bias_walk_noise.is_valid()
        {
            return Err("IMU measurement/bias noise densities must be finite and positive".into());
        }
        self.imu_noise = imu_noise;
        self.bias_walk_noise = bias_walk_noise;
        Ok(self)
    }

    /// Installs Basalt's static IMU calibration polynomials.  The vectors
    /// use the upstream parameter layouts: accel `[b(3), lower-triangular
    /// scale(6)]` and gyro `[b(3), full scale(9)]`.
    pub fn with_imu_calibration(
        mut self,
        accel_bias: Vec<f64>,
        gyro_bias: Vec<f64>,
    ) -> Result<Self, String> {
        if accel_bias.len() != 9 || gyro_bias.len() != 12 {
            return Err(format!(
                "IMU calibration requires 9 accel and 12 gyro parameters, got {} and {}",
                accel_bias.len(),
                gyro_bias.len()
            ));
        }
        if accel_bias
            .iter()
            .chain(gyro_bias.iter())
            .any(|value| !value.is_finite())
        {
            return Err("IMU calibration parameters must be finite".into());
        }
        self.calib_accel_bias = accel_bias;
        self.calib_gyro_bias = gyro_bias;
        Ok(self)
    }

    pub fn active_state_count(&self) -> usize {
        self.window_states.len()
    }

    pub fn active_pose_count(&self) -> usize {
        self.window_poses.len()
    }

    /// Diagnostic: number of landmarks in the database that are not removed,
    /// indexed by host camera (`result[c]` counts landmarks hosted in camera
    /// `c`).  Read-only; it does not touch solver state.
    pub fn landmark_count_by_host_camera(&self) -> Vec<usize> {
        let mut counts = vec![0; self.t_imu_cam.len().max(1)];
        for record in self.landmarks.landmarks.values() {
            if record.status == LandmarkStatus::Removed {
                continue;
            }
            let camera = record.landmark.anchor_camera_id as usize;
            if camera >= counts.len() {
                counts.resize(camera + 1, 0);
            }
            counts[camera] += 1;
        }
        counts
    }

    pub(crate) const fn no_output_mode_active(&self) -> bool {
        self.lean_no_output_mode
    }

    pub fn active_aom_dof(&self) -> usize {
        self.window_poses.len() * POSE_DOF + self.window_states.len() * NAV_STATE_DOF
    }

    pub fn active_state_dof(&self) -> usize {
        self.window_states.len() * NAV_STATE_DOF
    }

    pub const fn has_prior(&self) -> bool {
        self.prior.is_some()
    }

    fn camera_model(&self, camera_id: u16) -> Option<&DoubleSphereCamera> {
        self.camera_models.get(camera_id as usize)
    }

    fn camera_to_imu(&self, camera_id: u16) -> Option<&SE3> {
        self.t_imu_cam.get(camera_id as usize)
    }

    fn camera_pose(&self, imu_pose: &SE3, camera_id: u16) -> Option<SE3> {
        Some(imu_pose.compose(self.camera_to_imu(camera_id)?))
    }

    fn bearing_from_pixel(&self, camera_id: u16, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        let camera = self.camera_model(camera_id)?;
        if self.config.scalar_mode == ScalarMode::UpstreamF32 {
            let pixel_f32 = Point2::new(pixel.x as f32, pixel.y as f32);
            return camera
                .unproject_f32(&pixel_f32)
                .map(|bearing| bearing.map(|value| value as f64));
        }
        camera.unproject(pixel)
    }

    fn raw_bearing_from_pixel(&self, camera_id: u16, pixel: &Point2<f64>) -> Option<Vector3<f64>> {
        let camera = self.camera_model(camera_id)?;
        if self.config.scalar_mode == ScalarMode::UpstreamF32 {
            let pixel_f32 = Point2::new(pixel.x as f32, pixel.y as f32);
            return camera
                .unproject_raw_f32(&pixel_f32)
                .map(|bearing| bearing.map(|value| value as f64));
        }
        camera.unproject_raw(pixel)
    }

    pub fn process(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
    ) -> Result<EstimatorOutput, String> {
        self.process_impl(
            frame_id,
            timestamp_ns,
            observations,
            imu,
            None,
            None,
            true,
            true,
        )
    }

    /// Processes one frame and optionally retains its raw optical-flow input
    /// images for the next MargData snapshot.  The image vector is moved into
    /// the estimator, so the adapter performs the only necessary copy while
    /// handing ownership of the same sensor images to the frontend.
    pub fn process_with_images(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
        images: Option<Vec<OfImageData>>,
    ) -> Result<EstimatorOutput, String> {
        self.process_impl(
            frame_id,
            timestamp_ns,
            observations,
            imu,
            None,
            images,
            true,
            true,
        )
    }

    /// Processes one frame while suppressing the diagnostic MargData snapshot.
    ///
    /// The estimator still performs the same solve, writeback, marginal prior
    /// update, and active-window maintenance.  It suppresses only MargData;
    /// trace diagnostics remain available for compatibility.  Call
    /// `Self::process_without_marg_data_no_trace` when both output classes
    /// are intentionally disabled.
    pub fn process_without_marg_data(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
    ) -> Result<EstimatorOutput, String> {
        self.process_impl(
            frame_id,
            timestamp_ns,
            observations,
            imu,
            None,
            None,
            false,
            true,
        )
    }

    /// Processes one frame without retaining MargData or diagnostic trace
    /// payloads.  This is the explicit low-overhead path used by the demo's
    /// `--no-marg-data --no-trace` mode.  Probe/compatibility environment
    /// variables conservatively fall back to the retained diagnostic path.
    pub(crate) fn process_without_marg_data_no_trace(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
    ) -> Result<EstimatorOutput, String> {
        self.process_impl(
            frame_id,
            timestamp_ns,
            observations,
            imu,
            None,
            None,
            false,
            false,
        )
    }

    pub(crate) fn process_adapter_frame(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
        initialization_imu: Option<ImuSample>,
        images: Option<Vec<OfImageData>>,
        retain_marg_data: bool,
        retain_trace: bool,
    ) -> Result<EstimatorOutput, String> {
        self.process_impl(
            frame_id,
            timestamp_ns,
            observations,
            imu,
            initialization_imu,
            images,
            retain_marg_data,
            retain_trace,
        )
    }

    fn process_impl(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        observations: &[TrackObservation],
        imu: &[ImuSample],
        initialization_imu: Option<ImuSample>,
        images: Option<Vec<OfImageData>>,
        retain_marg_data: bool,
        retain_trace: bool,
    ) -> Result<EstimatorOutput, String> {
        if self.lean_no_output_mode && retain_marg_data {
            return Err(
                "cannot retain MargData after no-output mode: active keyframes may lack complete optical-flow payloads; recreate the estimator"
                    .into(),
            );
        }
        let entering_no_output_mode = !retain_marg_data && !retain_trace;
        let retain_marg_sidecar = diagnostic_env_snapshot().margdata_sidecar.is_some();
        // Any compatibility/probe variable opts out of the lean path.  This
        // keeps all existing capture hooks and their source boundaries active
        // even when a caller selected no-MargData/no-trace for normal runs.
        let retain_trace_payload = retain_trace || diagnostic_env_active();
        if self
            .last_timestamp_ns
            .is_some_and(|previous| timestamp_ns <= previous)
        {
            return Err("non-monotonic frame timestamp".into());
        }
        if let Some(mut images) = images {
            if images.iter().any(|image| {
                image.frame_id != frame_id
                    || image.timestamp_ns != timestamp_ns
                    || !image.is_valid()
            }) {
                return Err("invalid raw optical-flow image payload".into());
            }
            images.sort_by_key(|image| image.camera_id);
            if images
                .windows(2)
                .any(|pair| pair[0].camera_id == pair[1].camera_id)
            {
                return Err("duplicate raw optical-flow camera payload".into());
            }
            self.of_images.insert(frame_id, images);
        }
        let calibrated_imu = imu
            .iter()
            .copied()
            .map(|sample| self.calibrate_imu(sample))
            .collect::<Vec<_>>();
        let calibrated_initialization_imu =
            initialization_imu.map(|sample| self.calibrate_imu(sample));
        for sample in &calibrated_imu {
            self.stream
                .push_imu(*sample)
                .map_err(|error| format!("IMU input: {error:?}"))?;
        }
        self.stream
            .push_vision(VisionFrame {
                frame_id,
                timestamp_ns,
            })
            .map_err(|error| format!("vision input: {error:?}"))?;

        let previous_timestamp = self.last_timestamp_ns;
        let state_from = retain_trace_payload
            .then(|| self.window_states.last().map(|state| state.nav.clone()))
            .flatten();
        let mut previous_nav = self
            .window_states
            .last()
            .map(|state| state.nav.clone())
            .unwrap_or_else(|| self.nav.clone());
        if previous_timestamp.is_none() {
            let initialization_samples = calibrated_initialization_imu
                .as_ref()
                .map(std::slice::from_ref)
                .unwrap_or(&calibrated_imu);
            previous_nav = initial_nav_from_imu(
                timestamp_ns,
                initialization_samples,
                self.config.scalar_mode,
            );
        }
        let initialization_output =
            (retain_trace_payload && previous_timestamp.is_none()).then(|| previous_nav.clone());
        let (imu_delta, imu_integration_fallback) = previous_timestamp
            .map(|previous| {
                integrate_interval(
                    &calibrated_imu,
                    previous,
                    timestamp_ns,
                    // Static sensor calibration is applied above.  Basalt's
                    // IntegratedImuMeasurement is nevertheless linearized at
                    // the current dynamic state biases; these are distinct
                    // from the static sensor calibration offsets.
                    previous_nav.gyro_bias_rad_s,
                    previous_nav.accel_bias_m_s2,
                    self.imu_noise,
                    self.config.scalar_mode,
                )
            })
            .unwrap_or((None, false));
        let predicted_nav = predict_nav(
            &previous_nav,
            imu_delta.as_ref(),
            self.gravity_world
                .unwrap_or_else(|| Vector3::new(0.0, 0.0, -9.81)),
            self.config.scalar_mode,
        );
        let imu_propagation = retain_trace_payload
            .then(|| {
                previous_timestamp.and_then(|interval_start_ns| {
                    imu_delta.as_ref().map(|delta| ImuPropagationTrace {
                        interval_start_ns,
                        interval_end_ns: timestamp_ns,
                        sample_timestamps_ns: calibrated_imu
                            .iter()
                            .filter(|sample| {
                                sample.timestamp_ns > interval_start_ns
                                    && sample.timestamp_ns <= timestamp_ns
                            })
                            .map(|sample| sample.timestamp_ns)
                            .collect(),
                        delta_time: delta.delta_time,
                        delta_position: delta.delta_position,
                        delta_rotation: delta.delta_rotation,
                        delta_velocity: delta.delta_velocity,
                    })
                })
            })
            .flatten();

        if self.window_states.is_empty() {
            self.anchor_point = Some(flatten_nav_with_mode(
                &predicted_nav,
                self.config.scalar_mode,
            ));
        }
        // Upstream computes connectivity before triangulating the unconnected
        // tracks of a newly selected keyframe.  Preserve that ordering here:
        // otherwise a track created by this frame would count as already
        // connected and suppress the keyframe decision.
        let (connected_cam0, unconnected_cam0) = self.cam0_connectivity(observations);
        let is_keyframe = self.decide_keyframe(connected_cam0, unconnected_cam0);
        self.collect_observations(
            frame_id,
            timestamp_ns,
            &predicted_nav,
            observations,
            is_keyframe,
        )?;
        if let Some(delta) = imu_delta {
            if let Some(previous_id) = self.window_states.last().map(|state| state.frame_id) {
                self.imu_links.push((previous_id, frame_id, delta));
            }
        }
        self.window_states.push(WindowState {
            frame_id,
            timestamp_ns,
            nav: predicted_nav.clone(),
            stored_current_nav: predicted_nav.clone(),
            linearized_nav: predicted_nav.clone(),
            linearized_delta: DVector::zeros(NAV_STATE_DOF),
            is_keyframe,
            is_latest: true,
            // Basalt's initialized state starts FEJ-linearized; every
            // subsequently predicted state enters with a mutable point.
            linearized: self.window_states.is_empty(),
        });
        for state in &mut self.window_states {
            state.is_latest = state.frame_id == frame_id;
        }

        let seen = observations
            .iter()
            .map(|observation| observation.track_id)
            .collect::<Vec<_>>();
        // Upstream computes `lost_landmaks` before optimize, but defers the
        // actual LandmarkDatabase removal until marginalization.  Keep lost
        // records visible to the initialization-gated frames and to the
        // first solve; removing them here would make frames 0..3 diverge from
        // Basalt's causal lifecycle.
        self.landmarks.mark_frame_end_deferred(&seen);
        self.last_lost_landmarks = self
            .landmarks
            .landmarks
            .iter()
            .filter(|(_, record)| record.status == super::landmarks::LandmarkStatus::Lost)
            .map(|(track_id, _)| *track_id)
            .collect();
        self.last_lost_landmarks.sort_unstable();
        // This sidecar is deliberately emitted at the causal estimator
        // boundary, after current-frame association/lost marking but before
        // solve or marginalization shifts the window.  It therefore contains
        // frame 0..N observations as well as newly triangulated factors and
        // cannot affect any solver input when the path is unset.
        self.append_visual_factor_identity_trace(frame_id, timestamp_ns);
        self.stream.process_pending();
        self.stream.drain_pending_imu();

        self.last_kf_to_marg.clear();
        self.pending_marg_data = None;
        self.pending_marg_prior_pre = None;
        self.pending_marg_effective_pose_fej.clear();
        self.pending_marg_effective_state_fej.clear();
        self.last_marg_targets = MarginalizationTargets {
            lost_landmarks: self.last_lost_landmarks.clone(),
            ..MarginalizationTargets::default()
        };
        let mut window_diagnostics;
        let mut marginalized = false;
        if !self.opt_started && self.window_states.len() <= 4 {
            // The pinned upstream implementation calls optimize(), but its
            // `opt_started || frame_states.size() > 4` gate is false for
            // frames 0..3.  Keep these states and observations causal, while
            // avoiding both solve and landmark writeback until frame 4.
            self.last_rows.clear();
            window_diagnostics = WindowDiagnostics {
                attempted: false,
                state_dof: self.active_aom_dof(),
                factor_count: 0,
                factor_rows: 0,
                landmark_count: self
                    .landmarks
                    .landmarks
                    .values()
                    .filter(|record| record.status != super::landmarks::LandmarkStatus::Removed)
                    .count(),
                imu_link_count: self.imu_links.len(),
                prior_rows: self.prior.as_ref().map_or(0, WindowPrior::rows),
                prior_factor_rows: 0,
                visual_factor_rows: 0,
                imu_factor_rows: 0,
                bias_factor_rows: 0,
                prior_cost: 0.0,
                visual_cost: 0.0,
                imu_cost: 0.0,
                bias_cost: 0.0,
                imu_links: Vec::new(),
                lm: Vec::new(),
                status: "initialization_gate".into(),
                failure: None,
                state_writeback: false,
                landmark_writeback: 0,
                prior_carry: false,
            };
        } else {
            self.opt_started = true;
            let build_started = self.timing.start();
            let mut problem = self.build_problem();
            self.timing
                .finish(TimingBucket::EstimatorBuildProblem, build_started);
            let initial_state = problem.initial_state();
            let lm_started = self.timing.start();
            let solution_result = if retain_marg_data {
                if self.lean_marg_data {
                    // MargData needs the post-solve factor snapshot, not the
                    // diagnostic LM payloads.  Use the compact f32 preparation
                    // path (byte-identical trajectory, verified by the
                    // retained-vs-lean regression test) instead of the
                    // per-trial landmark re-factorization the diagnostic path
                    // performs.
                    problem.solve_lean_with_factors_with_timing(
                        initial_state,
                        self.config.solver,
                        &mut self.timing,
                    )
                } else {
                    problem.solve_with_timing(initial_state, self.config.solver, &mut self.timing)
                }
            } else if retain_trace_payload {
                problem.solve_without_factors_with_timing(
                    initial_state,
                    self.config.solver,
                    &mut self.timing,
                )
            } else {
                problem.solve_without_diagnostics_with_timing(
                    initial_state,
                    self.config.solver,
                    &mut self.timing,
                )
            };
            self.timing
                .finish(TimingBucket::EstimatorLmSolve, lm_started);
            let solution = solution_result.map_err(|error| error.to_string())?;
            let WindowSolveResult {
                state: solution_state,
                factors: solution_factors,
                diagnostics: solution_diagnostics,
                ..
            } = solution;
            window_diagnostics = solution_diagnostics;
            window_diagnostics.state_writeback = true;
            write_blocks_with_mode(
                &mut problem.poses,
                &mut problem.states,
                &solution_state,
                self.config.scalar_mode,
            );
            // The selection policy uses the solved pose/state values, so give
            // the estimator those snapshots before asking for a plan.  The
            // complete factor-bearing problem is only needed if this frame
            // actually crosses a marginalization boundary; cloning it on
            // every ordinary LM frame needlessly copies all visual/IMU rows
            // and was particularly expensive in no-MargData runs.
            self.window_poses = problem.poses.clone();
            self.window_states = problem.states.clone();
            self.apply_landmark_writeback(&problem.landmarks);
            let plan = self.marginalization_plan(frame_id);
            let needs_marginalization = !plan.drop_states.is_empty()
                || !plan.convert_states.is_empty()
                || !plan.drop_poses.is_empty();
            let marginalization_started = if needs_marginalization {
                self.timing.start()
            } else {
                None
            };
            let solved_problem = needs_marginalization.then(|| problem.clone());
            self.window_poses = problem.poses;
            self.window_states = problem.states;
            self.last_marg_targets = MarginalizationTargets {
                poses_to_marg: plan.drop_poses.clone(),
                states_to_marg_all: plan.drop_states.clone(),
                states_to_marg_vel_bias: plan.convert_states.clone(),
                lost_landmarks: self.last_lost_landmarks.clone(),
            };
            // `kfs_to_marg` is the upstream keyframe-selection set, not the
            // complete set of old navigation states.  In particular, a KF
            // converted from 15 DoF to pose-only remains in this set only if
            // the max-KF policy selected it for removal; ordinary old
            // non-KF states are never reported as keyframes.
            let mut kfs_to_marg = plan.drop_poses.clone();
            kfs_to_marg.sort_unstable();
            kfs_to_marg.dedup();
            self.last_kf_to_marg = kfs_to_marg.into_iter().map(|id| (id, frame_id)).collect();
            if needs_marginalization {
                marginalized = true;
                let had_prior = self.prior.is_some();
                let mut fallback_dropped = plan.drop_states.clone();
                fallback_dropped.extend(plan.convert_states.iter().copied());
                fallback_dropped.extend(plan.drop_poses.iter().copied());
                let solved_problem = solved_problem
                    .as_ref()
                    .expect("marginalization requires a solved problem snapshot");
                let (marg_problem, marg_state, marg_factors) = match self
                    .pre_marginalization_snapshot(
                        solved_problem,
                        &solution_state,
                        plan.boundary_state,
                        &plan.drop_poses,
                        &self.last_lost_landmarks,
                    ) {
                    Ok(snapshot) => snapshot,
                    Err(error) => {
                        self.timing.finish(
                            TimingBucket::EstimatorMarginalization,
                            marginalization_started,
                        );
                        return Err(error);
                    }
                };
                Self::append_q2_boundary_trace(
                    frame_id,
                    timestamp_ns,
                    solved_problem,
                    &marg_problem,
                    plan.boundary_state,
                    &marg_factors,
                );
                self.prior = marginalize_mixed_prior_with_mode(
                    &marg_factors,
                    &marg_problem.poses,
                    &marg_problem.states,
                    &plan.drop_poses,
                    &plan.drop_states,
                    &plan.convert_states,
                    self.config.scalar_mode,
                    &marg_state,
                )
                .or_else(|| self.anchor_prior_for_kept(&fallback_dropped));
                let prior_pre_transition = if retain_marg_sidecar {
                    take_diagnostic_prior_pre().and_then(|prior| Self::prior_data(&prior))
                } else {
                    None
                };
                window_diagnostics.prior_carry = had_prior || self.prior.is_some();
                if retain_marg_sidecar {
                    let (effective_poses, effective_states) =
                        marg_problem.diagnostic_effective_fej_poses();
                    self.pending_marg_effective_pose_fej = effective_poses
                        .into_iter()
                        .map(|(timestamp, pose)| (timestamp, Self::pose_wire(&pose)))
                        .collect();
                    self.pending_marg_effective_state_fej = effective_states
                        .into_iter()
                        .map(|(timestamp, pose)| (timestamp, Self::pose_wire(&pose)))
                        .collect();
                }
                // The packet is captured before any state/pose erasure, just
                // like upstream's `out_marg_queue` branch.  This preserves
                // the dropped keyframe, the newest full state in
                // `frame_states`, and the pre-marginal AOM/J/r together.
                if retain_marg_data {
                    self.pending_marg_prior_pre = prior_pre_transition;
                    self.pending_marg_data = Some(self.emit_pre_marg(
                        &marg_problem,
                        &marg_factors,
                        &plan,
                        &window_diagnostics,
                    ));
                }
                // Upstream freezes exactly the retained boundary after the
                // packet is serialized.  The newest state remains FEJ-false.
                if let Some(boundary_id) = plan.boundary_state {
                    if let Some(boundary) = self
                        .window_states
                        .iter_mut()
                        .find(|state| state.frame_id == boundary_id)
                    {
                        boundary.linearized = true;
                        boundary.linearized_nav = boundary.nav.clone();
                        boundary.stored_current_nav = boundary.nav.clone();
                        boundary.linearized_delta.fill(0.0);
                    }
                }
                for frame_id in &plan.convert_states {
                    if let Some(index) = self
                        .window_states
                        .iter()
                        .position(|state| state.frame_id == *frame_id)
                    {
                        let state = self.window_states.remove(index);
                        let pose = WindowPose {
                            frame_id: state.frame_id,
                            timestamp_ns: state.timestamp_ns,
                            pose: state.nav.imu_to_world.clone(),
                            stored_current_pose: state.nav.imu_to_world.clone(),
                            linearized_pose: state.linearized_nav.imu_to_world.clone(),
                            linearized_delta: state.linearized_delta.rows(0, POSE_DOF).into_owned(),
                            is_keyframe: state.is_keyframe,
                        };
                        let insert_at = self
                            .window_poses
                            .partition_point(|existing| existing.frame_id < pose.frame_id);
                        self.window_poses.insert(insert_at, pose);
                    }
                }
                self.window_poses
                    .retain(|pose| !plan.drop_poses.contains(&pose.frame_id));
                self.window_states
                    .retain(|state| !plan.drop_states.contains(&state.frame_id));
                self.imu_links.retain(|(from, to, _)| {
                    self.window_states
                        .iter()
                        .any(|state| state.frame_id == *from)
                        && self.window_states.iter().any(|state| state.frame_id == *to)
                });
                self.window_observations.retain(|_, records| {
                    records.retain(|record| {
                        self.window_states
                            .iter()
                            .any(|state| state.frame_id == record.frame_id)
                            || self
                                .window_poses
                                .iter()
                                .any(|pose| pose.frame_id == record.frame_id)
                    });
                    !records.is_empty()
                });
                // Basalt deletes landmarks whose host keyframe was selected
                // for marginalization. It does not rehost them at the
                // oldest retained keyframe, because removeKeyframes runs
                // after the pre-removal solve and packet capture.
                self.landmarks.remove_host_landmarks(&plan.drop_poses);
                // `removeLandmark` is part of Basalt's marginalization path,
                // after the solve has used the complete pre-removal window.
                // Prune the Rust-side records before rebuilding the shifted
                // factor snapshot so MargData cannot retain factors for a
                // landmark that has just left the database.
                self.landmarks.remove_deferred_lost();
                // Entries for removed tracks have no later consumer, so drop
                // them instead of retaining one Point3 per historical track
                // for the lifetime of the estimator.
                self.prune_landmark_world_cache();
                // Upstream removes observations attached to poses/full states
                // that left the window.  Converted states remain pose-only
                // blocks, so their observations are deliberately retained.
                let mut dropped_observation_frames = plan.drop_states.clone();
                dropped_observation_frames.extend(plan.drop_poses.iter().copied());
                self.landmarks
                    .remove_frame_observations(&dropped_observation_frames);
                window_diagnostics.landmark_count = self
                    .landmarks
                    .landmarks
                    .values()
                    .filter(|record| record.status != super::landmarks::LandmarkStatus::Removed)
                    .count();
                self.window_observations
                    .retain(|track_id, _| self.landmarks.landmarks.contains_key(track_id));
                self.prune_native_host_order();
                // The packet for a marginalizing frame was already captured
                // from `marg_problem` above, before the shift.  A subsequent
                // post-shift relinearization used to populate `last_rows`,
                // but every following frame overwrites `last_rows` from its
                // own solve (and this frame emits `pending_marg_data`).  It
                // therefore had no consumer and cost a full extra visual /
                // IMU linearization at every keyframe removal.  Leave the
                // previous snapshot untouched; it is never used while a
                // pending packet exists and is replaced before any later
                // state-only packet is emitted.
            } else if retain_marg_data {
                self.last_rows = projected_rows(&solution_factors, 1e-10);
            } else {
                self.last_rows.clear();
            }
            self.timing.finish(
                TimingBucket::EstimatorMarginalization,
                marginalization_started,
            );
        }
        // Basalt's `prev_opt_flow_res` stores optimized active-window poses,
        // not the pre-solve prediction with which a frame first arrived.
        // Refresh clean-room history after writeback and discard entries for
        // states/poses that marginalization erased.
        self.refresh_track_history_poses();
        if marginalized {
            self.prune_track_history();
        }
        self.refresh_public_states();
        let post_opt_state = self
            .window_states
            .last()
            .map(|state| state.nav.clone())
            .unwrap_or_else(|| predicted_nav.clone());
        self.last_timestamp_ns = Some(timestamp_ns);
        let marg = if retain_marg_data {
            self.pending_marg_data
                .take()
                .unwrap_or_else(|| self.emit_marg(&window_diagnostics))
        } else {
            self.pending_marg_data.take();
            MargData::empty()
        };
        if retain_marg_sidecar && marg.is_mapper_packet() {
            let sidecar = marg.diagnostic_sidecar_with_effective(
                self.marg_event_ordinal,
                frame_id,
                timestamp_ns,
                self.pending_marg_prior_pre.take(),
                &self.pending_marg_effective_pose_fej,
                &self.pending_marg_effective_state_fej,
            );
            self.marg_event_ordinal = self.marg_event_ordinal.saturating_add(1);
            Self::append_margdata_sidecar(&sidecar);
        } else {
            self.pending_marg_prior_pre = None;
        }
        self.pending_marg_effective_pose_fej.clear();
        self.pending_marg_effective_state_fej.clear();
        // Keep a packet's dropped keyframe images alive until emit_marg has
        // copied their records into the artifact.  Once the packet is built,
        // only active keyframe images remain resident for the next event.
        let active_keyframes = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(
                self.window_states
                    .iter()
                    .filter(|state| state.is_keyframe)
                    .map(|state| state.frame_id),
            )
            .collect::<std::collections::BTreeSet<_>>();
        self.of_images
            .retain(|frame_id, _| active_keyframes.contains(frame_id));
        self.of_observations
            .retain(|frame_id, _| active_keyframes.contains(frame_id));
        // Commit the one-way mode only after every fallible estimator stage
        // has completed and the output is ready to return.  A failed lean
        // call must leave callers free to retry through the full image-
        // retaining path.
        if entering_no_output_mode {
            self.lean_no_output_mode = true;
        }
        self.append_lifecycle_trace(frame_id, timestamp_ns, is_keyframe, marginalized);
        Ok(EstimatorOutput {
            frame_id,
            state: self.nav.clone(),
            is_keyframe,
            connected_cam0,
            unconnected_cam0,
            active_state_count: self.window_states.len(),
            active_pose_count: self.window_poses.len(),
            state_trace: EstimatorStateTrace {
                state_from,
                initialization_output,
                predicted_state: predicted_nav,
                post_opt_state,
                imu_propagation,
                imu_integration_fallback,
            },
            phases: if retain_trace_payload {
                self.stream.phase_log.clone()
            } else {
                Vec::new()
            },
            marg_data: marg,
            window: window_diagnostics,
            imu_integration_fallback,
        })
    }

    fn collect_observations(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        pose_now: &BasaltNavState,
        observations: &[TrackObservation],
        is_keyframe: bool,
    ) -> Result<(), String> {
        // The current optical-flow packet is a set of TimeCamId identities,
        // not an append-only log. Reject duplicate or stale frame bindings
        // before touching any landmark/history/map state.
        Self::validate_current_observation_identities(frame_id, timestamp_ns, observations)?;

        // First materialize the complete current measurement. Basalt inserts
        // `prev_opt_flow_res[t_ns]` before `addNewKeyframe`, so a same-frame
        // stereo observation is available even if camera 1 precedes camera 0
        // in the Rust input slice.
        let mut current = Vec::with_capacity(observations.len());
        for observation in observations {
            let bearing = self
                .bearing_from_pixel(observation.camera_id, &observation.pixel)
                .ok_or_else(|| "invalid DS observation".to_string())?;
            let raw_bearing = self
                .raw_bearing_from_pixel(observation.camera_id, &observation.pixel)
                .ok_or_else(|| "invalid DS raw observation".to_string())?;
            let camera_pose = self
                .camera_pose(&pose_now.imu_to_world, observation.camera_id)
                .ok_or_else(|| {
                    format!(
                        "camera {} is missing from calibration",
                        observation.camera_id
                    )
                })?;
            current.push((
                observation.clone(),
                bearing,
                TrackHistory {
                    frame_id,
                    camera_id: observation.camera_id,
                    timestamp_ns: observation.timestamp_ns,
                    pixel: observation.pixel,
                    raw_bearing,
                    imu_pose: pose_now.imu_to_world.clone(),
                    pose: camera_pose,
                },
            ));
        }

        // Validate retained history and reject a current identity that was
        // already consumed. This preflight runs before the association and
        // history append loop, so malformed/duplicate input cannot partially
        // mutate an existing landmark's factor-visible records.
        for (observation, _bearing, _history_entry) in &current {
            if let Some(history) = self.track_history.get(&observation.track_id) {
                Self::validate_track_history_identity(history)?;
                if history.iter().any(|entry| {
                    entry.frame_id == observation.frame_id
                        && entry.camera_id == observation.camera_id
                }) {
                    return Err(format!(
                        "duplicate history observation identity for track {} at frame {} camera {}",
                        observation.track_id, observation.frame_id, observation.camera_id
                    ));
                }
            }
        }

        let landmark_exists = |db: &ObservationDb, track_id: TrackId| {
            db.landmarks
                .get(&track_id)
                .is_some_and(|record| record.status != super::landmarks::LandmarkStatus::Removed)
        };

        // Existing native landmarks keep observations on every active pose,
        // including non-keyframes preceding a newly promoted keyframe.
        // The post-solve marginalization boundary removes retired frames.
        let active_factor_frames: BTreeSet<u64> = self
            .window_states
            .iter()
            .map(|state| state.frame_id)
            .chain(self.window_poses.iter().map(|pose| pose.frame_id))
            .chain(std::iter::once(frame_id))
            .collect();

        // Upstream's first pass associates only landmarks that already exist;
        // unknown camera-0 tracks form the keyframe candidate set.  Build the
        // set and all duplicate/map preconditions without mutating any store;
        // the commit loop runs only after every fallible candidate has been
        // prepared.
        for (track_id, records) in &self.window_observations {
            Self::validate_factor_observation_records(
                *track_id,
                records,
                self.track_history.get(track_id).map(Vec::as_slice),
            )?;
        }
        // A factor projection may legitimately be ahead of the delayed
        // history store (for example while a lost track is being rebound).
        // Still reject a current identity that is already present in that
        // projection: without this cross-check, the missing-history case
        // would append a duplicate row after the per-map validation above.
        for (observation, _bearing, _history_entry) in &current {
            if let Some(records) = self.window_observations.get(&observation.track_id) {
                if records.iter().any(|record| {
                    record.frame_id == observation.frame_id
                        && record.camera_id == observation.camera_id
                }) {
                    return Err(format!(
                        "duplicate current factor identity for track {} at frame {} camera {}",
                        observation.track_id, observation.frame_id, observation.camera_id
                    ));
                }
            }
        }
        let mut unconnected_cam0 = super::landmarks::LibstdcxxUnorderedSet::default();
        let mut existing_tracks = BTreeSet::<TrackId>::new();
        for (observation, _bearing, _history_entry) in &current {
            if landmark_exists(&self.landmarks, observation.track_id) {
                existing_tracks.insert(observation.track_id);
            } else if observation.camera_id == 0 {
                unconnected_cam0.insert(observation.track_id);
            }
        }
        // Opt-in (`landmarks_all_cameras`): unconnected tracks that cam0 does
        // not see in this frame, keyed by the lowest other camera that does.
        // They are tried after every cam0 candidate, in track-id order, so
        // the pinned cam0 candidate order is untouched.  Empty by default.
        let mut unconnected_other_host = BTreeMap::<TrackId, u16>::new();
        if self.config.landmarks_all_cameras {
            let seen_by_cam0 = current
                .iter()
                .filter(|(observation, _, _)| observation.camera_id == 0)
                .map(|(observation, _, _)| observation.track_id)
                .collect::<BTreeSet<_>>();
            for (observation, _bearing, _history_entry) in &current {
                if observation.camera_id == 0
                    || seen_by_cam0.contains(&observation.track_id)
                    || landmark_exists(&self.landmarks, observation.track_id)
                {
                    continue;
                }
                unconnected_other_host
                    .entry(observation.track_id)
                    .and_modify(|host| *host = (*host).min(observation.camera_id))
                    .or_insert(observation.camera_id);
            }
        }

        let mut prepared_landmarks = Vec::<PreparedLandmarkInsertion>::new();
        let triangulation_trace_path = diagnostic_env_snapshot().triangulation_trace.clone();
        let triangulation_trace_tracks =
            diagnostic_env_snapshot().triangulation_trace_tracks.clone();
        if is_keyframe {
            // `tcidl = TimeCamId(current_frame, 0)` is the host for every new
            // VIO landmark. Gather each track's complete retained observation
            // set, including the current frame, before trying triangulation.
            let current_cam0 = current
                .iter()
                .filter(|(observation, _, _)| observation.camera_id == 0)
                .map(|(observation, _bearing, history)| {
                    (observation.track_id, (history.raw_bearing, history.clone()))
                })
                .collect::<BTreeMap<_, _>>();
            // Opt-in host observations in cameras other than cam0, keyed by
            // (track, camera).  Empty unless `landmarks_all_cameras` found a
            // candidate above.
            let current_other_host = current
                .iter()
                .filter(|(observation, _, _)| {
                    unconnected_other_host.get(&observation.track_id)
                        == Some(&observation.camera_id)
                })
                .map(|(observation, _bearing, history)| {
                    (
                        (observation.track_id, observation.camera_id),
                        (history.raw_bearing, history.clone()),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let candidate_hosts = unconnected_cam0
                .iter()
                .map(|track_id| (*track_id, 0_u16))
                .chain(
                    unconnected_other_host
                        .iter()
                        .map(|(&track, &host)| (track, host)),
                )
                .collect::<Vec<_>>();

            for (track_id, host_camera_id) in &candidate_hosts {
                let host_camera_id = *host_camera_id;
                let target = if host_camera_id == 0 {
                    current_cam0.get(track_id)
                } else {
                    current_other_host.get(&(*track_id, host_camera_id))
                };
                let Some((target_bearing, target_history)) = target else {
                    continue;
                };

                // Upstream iterates std::map<TimeCamId, ...>, not only the
                // immediate previous same-camera sample. This allows a
                // farther retained keyframe to seed a track when the nearest
                // candidate fails the baseline gate.
                let mut candidates = BTreeMap::<(i64, u16, u64), TrackHistory>::new();
                let mut complete_history = self
                    .track_history
                    .get(&track_id)
                    .cloned()
                    .unwrap_or_default();
                complete_history.extend(
                    current
                        .iter()
                        .filter(|(observation, _, _)| observation.track_id == *track_id)
                        .map(|(_observation, _bearing, history)| history.clone()),
                );
                Self::validate_track_history_identity(&complete_history)?;
                for entry in &complete_history {
                    // TimeCamId is ordered by upstream frame/timestamp first,
                    // then camera id. The Rust frame id is kept as a final
                    // tie-break for malformed synthetic inputs that reuse a
                    // timestamp.
                    if candidates
                        .insert(
                            (entry.timestamp_ns, entry.camera_id, entry.frame_id),
                            entry.clone(),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "duplicate candidate identity for track {} at frame {} camera {}",
                            track_id, entry.frame_id, entry.camera_id
                        ));
                    }
                }

                // Convert every candidate bearing before any world point,
                // LandmarkDb record, factor map entry, or association is
                // inserted. A malformed historical camera therefore fails
                // closed without leaving a half-created landmark.
                let candidate_bearings = candidates
                    .values()
                    .map(|candidate| {
                        self.bearing_from_pixel(candidate.camera_id, &candidate.pixel)
                            .ok_or_else(|| "invalid DS history observation".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if candidate_bearings.len() != candidates.len() {
                    return Err(format!(
                        "candidate bearing count mismatch for track {}: {} != {}",
                        track_id,
                        candidate_bearings.len(),
                        candidates.len()
                    ));
                }

                let capture_triangulation = triangulation_trace_path.is_some()
                    && triangulation_trace_tracks
                        .as_ref()
                        .is_none_or(|tracks| tracks.contains(track_id));
                let mut triangulation_attempts = Vec::new();
                let mut selected_attempt = None;
                let mut triangulated = None;
                for (candidate_index, candidate) in candidates.values().enumerate() {
                    // Basalt applies T_0_1.translation().squaredNorm() to
                    // temporal and same-timestamp stereo candidates alike.
                    let baseline = target_history.pose.translation - candidate.pose.translation;
                    let baseline_norm = if self.config.scalar_mode == ScalarMode::UpstreamF32 {
                        baseline.map(|value| value as f32).norm() as f64
                    } else {
                        baseline.norm()
                    };
                    if baseline_norm < 0.05 {
                        if capture_triangulation {
                            triangulation_attempts.push(json!({
                                "candidate_index": candidate_index,
                                "candidate": json_track_history_triangulation_trace(candidate),
                                "baseline_norm": baseline_norm,
                                "status": "baseline_rejected",
                                "dlt": serde_json::Value::Null,
                                "result": serde_json::Value::Null,
                            }));
                        }
                        continue;
                    }
                    let triangulated_candidate = if self.config.scalar_mode
                        == ScalarMode::UpstreamF32
                    {
                        // `PoseVelBiasStateWithLin<float>` already owns the
                        // upstream Sophus SO3 value.  Widening through the
                        // f64 history and narrowing back must not normalize
                        // that stored quaternion a second time.
                        let target_rotation = UnitQuaternion::new_unchecked(Quaternion::new(
                            target_history.imu_pose.rotation.w as f32,
                            target_history.imu_pose.rotation.i as f32,
                            target_history.imu_pose.rotation.j as f32,
                            target_history.imu_pose.rotation.k as f32,
                        ));
                        let candidate_rotation = UnitQuaternion::new_unchecked(Quaternion::new(
                            candidate.imu_pose.rotation.w as f32,
                            candidate.imu_pose.rotation.i as f32,
                            candidate.imu_pose.rotation.j as f32,
                            candidate.imu_pose.rotation.k as f32,
                        ));
                        // The host camera is cam0 unless the opt-in
                        // `landmarks_all_cameras` selected another camera.
                        let Some(target_extrinsic) = self.camera_to_imu(host_camera_id) else {
                            continue;
                        };
                        let Some(candidate_extrinsic) = self.camera_to_imu(candidate.camera_id)
                        else {
                            continue;
                        };
                        let target_extrinsic_rotation =
                            UnitQuaternion::from_quaternion(Quaternion::new(
                                target_extrinsic.rotation.w as f32,
                                target_extrinsic.rotation.i as f32,
                                target_extrinsic.rotation.j as f32,
                                target_extrinsic.rotation.k as f32,
                            ));
                        let candidate_extrinsic_rotation =
                            UnitQuaternion::from_quaternion(Quaternion::new(
                                candidate_extrinsic.rotation.w as f32,
                                candidate_extrinsic.rotation.i as f32,
                                candidate_extrinsic.rotation.j as f32,
                                candidate_extrinsic.rotation.k as f32,
                            ));
                        let target_imu_translation = target_history
                            .imu_pose
                            .translation
                            .map(|value| value as f32);
                        let target_extrinsic_translation =
                            target_extrinsic.translation.map(|value| value as f32);
                        let target_bearing_f32 = target_bearing.map(|value| value as f32);
                        let candidate_imu_translation =
                            candidate.imu_pose.translation.map(|value| value as f32);
                        let candidate_extrinsic_translation =
                            candidate_extrinsic.translation.map(|value| value as f32);
                        let candidate_bearing_f32 = candidate.raw_bearing.map(|value| value as f32);
                        let target_bearing_trace = [
                            target_bearing_f32.x,
                            target_bearing_f32.y,
                            target_bearing_f32.z,
                        ];
                        let candidate_bearing_normalized =
                            candidate_bearings[candidate_index].map(|value| value as f32);
                        let candidate_bearing_normalized_trace = [
                            candidate_bearing_normalized.x,
                            candidate_bearing_normalized.y,
                            candidate_bearing_normalized.z,
                        ];
                        let (candidate_result, trace_record) = if capture_triangulation {
                            let attempt = triangulate_dlt_rig_f32_traced(
                                &target_rotation,
                                target_imu_translation,
                                &target_extrinsic_rotation,
                                target_extrinsic_translation,
                                target_bearing_f32,
                                &candidate_rotation,
                                candidate_imu_translation,
                                &candidate_extrinsic_rotation,
                                candidate_extrinsic_translation,
                                candidate_bearing_f32,
                            );
                            let result = attempt.result;
                            let status = match result {
                                Some((_direction, inverse_distance))
                                    if inverse_distance.is_finite()
                                        && inverse_distance > 0.0
                                        && inverse_distance < 3.0 =>
                                {
                                    "accepted"
                                }
                                Some(_) => "rho_rejected",
                                None => "dlt_failed",
                            };
                            let result_f64 = result.map(|(direction, inverse_distance)| {
                                (direction.map(|value| value as f64), inverse_distance as f64)
                            });
                            let record = json!({
                                "candidate_index": candidate_index,
                                "candidate": json_track_history_triangulation_trace(candidate),
                                "baseline_norm": baseline_norm,
                                "target_bearing_normalized_f32": json_f32_values(&target_bearing_trace),
                                "candidate_bearing_normalized_f32": json_f32_values(&candidate_bearing_normalized_trace),
                                "target_extrinsic": json_se3_triangulation_trace(target_extrinsic),
                                "candidate_extrinsic": json_se3_triangulation_trace(candidate_extrinsic),
                                "dlt": json_triangulation_dlt_trace(&attempt.dlt),
                                "result": json_triangulation_result(result),
                                "status": status,
                            });
                            (result_f64, Some(record))
                        } else {
                            (
                                triangulate_dlt_rig_f32(
                                    &target_rotation,
                                    target_imu_translation,
                                    &target_extrinsic_rotation,
                                    target_extrinsic_translation,
                                    target_bearing_f32,
                                    &candidate_rotation,
                                    candidate_imu_translation,
                                    &candidate_extrinsic_rotation,
                                    candidate_extrinsic_translation,
                                    candidate_bearing_f32,
                                )
                                .map(
                                    |(direction, inverse_distance)| {
                                        (
                                            direction.map(|value| value as f64),
                                            inverse_distance as f64,
                                        )
                                    },
                                ),
                                None,
                            )
                        };
                        if let Some(record) = trace_record {
                            triangulation_attempts.push(record);
                        }
                        candidate_result
                    } else {
                        if capture_triangulation {
                            triangulation_attempts.push(json!({
                                "candidate_index": candidate_index,
                                "candidate": json_track_history_triangulation_trace(candidate),
                                "baseline_norm": baseline_norm,
                                "status": "extended_f64_untraced",
                                "dlt": serde_json::Value::Null,
                                "result": serde_json::Value::Null,
                            }));
                        }
                        triangulate_dlt(
                            &target_history.pose,
                            *target_bearing,
                            &candidate.pose,
                            candidate.raw_bearing,
                        )
                    };
                    if let Some((direction, inverse_distance)) = triangulated_candidate {
                        // The upstream homogeneous result is accepted only
                        // for finite 0 < rho < 3.
                        if inverse_distance.is_finite()
                            && inverse_distance > 0.0
                            && inverse_distance < 3.0
                        {
                            triangulated = Some((direction, inverse_distance));
                            selected_attempt = Some(candidate_index);
                            break;
                        }
                    }
                }

                if capture_triangulation {
                    if let Some(path) = triangulation_trace_path.as_ref() {
                        append_triangulation_trace(
                            path,
                            frame_id,
                            timestamp_ns,
                            *track_id,
                            target_history,
                            &complete_history,
                            &triangulation_attempts,
                            selected_attempt,
                            self.config.scalar_mode,
                        );
                    }
                }

                let Some((direction, inverse_distance)) = triangulated else {
                    continue;
                };
                let point_anchor = Point3::from(direction / inverse_distance);
                let anchor_distance = point_anchor.coords.norm();
                if !anchor_distance.is_finite() || anchor_distance <= 1e-9 {
                    continue;
                }
                let Some(stereographic_direction) =
                    (if self.config.scalar_mode == ScalarMode::UpstreamF32 {
                        StereographicDirection::from_triangulated_f32(
                            direction.map(|value| value as f32),
                        )
                    } else {
                        StereographicDirection::from_bearing(point_anchor.coords / anchor_distance)
                    })
                else {
                    continue;
                };

                // Native inserts every observation on an active frame, not
                // only the triangulating pair or keyframe observations.
                let factor_observations: Vec<StoredObservation> = candidates
                    .values()
                    .filter(|candidate| active_factor_frames.contains(&candidate.frame_id))
                    .map(|candidate| StoredObservation {
                        frame_id: candidate.frame_id,
                        camera_id: candidate.camera_id,
                        pixel: candidate.pixel,
                    })
                    .collect();
                if factor_observations.is_empty()
                    || !factor_observations
                        .iter()
                        .any(|observation| observation.frame_id == frame_id)
                {
                    return Err(format!(
                        "candidate projection has no current-frame observation for track {}",
                        track_id
                    ));
                }
                let point_world = target_history.pose.transform_point(&point_anchor);
                let candidate_bearings = candidates
                    .values()
                    .cloned()
                    .zip(candidate_bearings.into_iter())
                    .collect::<Vec<_>>();
                prepared_landmarks.push(PreparedLandmarkInsertion {
                    track_id: *track_id,
                    host_timestamp_ns: target_history.timestamp_ns,
                    point_world,
                    landmark: InverseDistanceLandmark {
                        anchor_pose: frame_id,
                        anchor_camera_id: host_camera_id,
                        direction: stereographic_direction,
                        inverse_distance,
                    },
                    factor_observations,
                    candidate_bearings,
                });
            }
        }

        // Commit the packet only after all current records and all candidate
        // insertions have passed their fallible validation. The order of the
        // existing association loop and the source-compatible unordered
        // candidate set is unchanged.
        for (observation, bearing, history_entry) in &current {
            if existing_tracks.contains(&observation.track_id) {
                self.landmarks.associate(BearingObservation {
                    landmark_id: observation.track_id,
                    frame_id,
                    bearing: *bearing,
                });
            }
            self.track_history
                .entry(observation.track_id)
                .or_default()
                .push(history_entry.clone());
            if existing_tracks.contains(&observation.track_id) {
                let records = self
                    .window_observations
                    .entry(observation.track_id)
                    .or_default();
                records.push(StoredObservation {
                    frame_id,
                    camera_id: observation.camera_id,
                    pixel: observation.pixel,
                });
                records.retain(|record| active_factor_frames.contains(&record.frame_id));
            }
        }

        let mut num_points_added = 0usize;
        for prepared in prepared_landmarks {
            // Prepared insertions always have attached observations: this
            // is where native first creates the outer host-map entry.
            let host = (
                prepared.landmark.anchor_pose,
                prepared.landmark.anchor_camera_id,
            );
            let timestamp = prepared.host_timestamp_ns as u64;
            self.native_host_order.insert(timestamp, host.1);
            self.native_host_keys.insert(host, timestamp);
            self.landmark_world
                .insert(prepared.track_id, prepared.point_world);
            self.landmarks.insert(prepared.track_id, prepared.landmark);
            num_points_added += 1;
            self.window_observations
                .insert(prepared.track_id, prepared.factor_observations);
            for (candidate, candidate_bearing) in prepared.candidate_bearings {
                self.landmarks.associate(BearingObservation {
                    landmark_id: prepared.track_id,
                    frame_id: candidate.frame_id,
                    bearing: candidate_bearing,
                });
            }
        }
        if is_keyframe && self.of_images.contains_key(&frame_id) {
            self.of_observations.insert(
                frame_id,
                current
                    .iter()
                    .map(|(observation, _bearing, _history)| OfObservationData {
                        frame_id,
                        track_id: observation.track_id,
                        camera_id: observation.camera_id,
                        x: observation.pixel.x,
                        y: observation.pixel.y,
                    })
                    .collect(),
            );
        }
        if is_keyframe {
            // Basalt's denominator is the number of successful
            // triangulations made by addNewKeyframe, not an active-record
            // count delta. Count at the same point as lmdb.addLandmark.
            self.num_points_kf.insert(frame_id, num_points_added);
        }
        Ok(())
    }

    /// Return the active keyframe IDs in deterministic frame order.  The
    /// current state has not yet been pushed when `collect_observations` runs,
    /// so its keyframe bit is supplied explicitly.  A converted keyframe may
    /// live in `window_poses` rather than `window_states`.
    fn keyframe_frame_ids(&self, current_frame_id: u64, current_is_keyframe: bool) -> Vec<u64> {
        let mut keyframes = self
            .window_states
            .iter()
            .filter(|state| state.is_keyframe)
            .map(|state| state.frame_id)
            .chain(
                self.window_poses
                    .iter()
                    .filter(|pose| pose.is_keyframe)
                    .map(|pose| pose.frame_id),
            )
            .collect::<Vec<_>>();
        if current_is_keyframe {
            keyframes.push(current_frame_id);
        }
        keyframes.sort_unstable();
        keyframes.dedup();
        keyframes
    }

    /// Validate the current packet's frame/camera identity before any
    /// association/history mutation. The frame id carried by an observation
    /// is part of the upstream `TimeCamId` and must agree with the enclosing
    /// camera event.
    fn validate_current_observation_identities(
        current_frame_id: u64,
        current_timestamp_ns: i64,
        observations: &[TrackObservation],
    ) -> Result<(), String> {
        let mut identities = BTreeSet::<(TrackId, u64, u16)>::new();
        for observation in observations {
            if observation.frame_id != current_frame_id {
                return Err(format!(
                    "observation frame {} does not match current frame {} for track {} camera {}",
                    observation.frame_id,
                    current_frame_id,
                    observation.track_id,
                    observation.camera_id
                ));
            }
            if observation.timestamp_ns != current_timestamp_ns {
                return Err(format!(
                    "observation timestamp {} does not match current timestamp {} for track {} frame {} camera {}",
                    observation.timestamp_ns,
                    current_timestamp_ns,
                    observation.track_id,
                    observation.frame_id,
                    observation.camera_id
                ));
            }
            let identity = (
                observation.track_id,
                observation.frame_id,
                observation.camera_id,
            );
            if !identities.insert(identity) {
                return Err(format!(
                    "duplicate current observation identity for track {} at frame {} camera {}",
                    observation.track_id, observation.frame_id, observation.camera_id
                ));
            }
        }
        Ok(())
    }

    /// Validate the one-to-one `(frame,camera)` and `(timestamp,camera)`
    /// bindings used to construct the source-compatible candidate map.
    fn validate_track_history_identity(history: &[TrackHistory]) -> Result<(), String> {
        let mut frame_camera = BTreeSet::<(u64, u16)>::new();
        let mut timestamp_camera = BTreeMap::<(i64, u16), u64>::new();
        for entry in history {
            if !frame_camera.insert((entry.frame_id, entry.camera_id)) {
                return Err(format!(
                    "duplicate track history identity at frame {} camera {}",
                    entry.frame_id, entry.camera_id
                ));
            }
            if let Some(previous_frame) =
                timestamp_camera.insert((entry.timestamp_ns, entry.camera_id), entry.frame_id)
            {
                if previous_frame != entry.frame_id {
                    return Err(format!(
                        "timestamp {} camera {} is bound to frames {} and {}",
                        entry.timestamp_ns, entry.camera_id, previous_frame, entry.frame_id
                    ));
                }
            }
        }
        Ok(())
    }

    /// Validate the factor-visible projection's own identity set before an
    /// existing landmark receives a new association. The projection is
    /// intentionally a subset of complete history, so missing history rows
    /// are allowed; duplicate rows and disagreeing pixels are not.
    fn validate_factor_observation_records(
        track_id: TrackId,
        records: &[StoredObservation],
        history: Option<&[TrackHistory]>,
    ) -> Result<(), String> {
        let mut identities = BTreeSet::<(u64, u16)>::new();
        for record in records {
            if !identities.insert((record.frame_id, record.camera_id)) {
                return Err(format!(
                    "duplicate factor observation identity for track {} at frame {} camera {}",
                    track_id, record.frame_id, record.camera_id
                ));
            }
            if let Some(history) = history {
                if let Some(entry) = history.iter().find(|entry| {
                    entry.frame_id == record.frame_id && entry.camera_id == record.camera_id
                }) {
                    if entry.pixel.x.to_bits() != record.pixel.x.to_bits()
                        || entry.pixel.y.to_bits() != record.pixel.y.to_bits()
                    {
                        return Err(format!(
                            "factor/history pixel mismatch for track {} at frame {} camera {}",
                            track_id, record.frame_id, record.camera_id
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn factor_frame_visible(
        candidate_frame_id: u64,
        current_frame_id: u64,
        keyframe_frames: &[u64],
    ) -> bool {
        if candidate_frame_id == current_frame_id
            || keyframe_frames.binary_search(&candidate_frame_id).is_ok()
        {
            return true;
        }

        // Native keeps non-keyframe observations sampled after the most
        // recent keyframe in the active landmark block until the next
        // keyframe is promoted. This retains frame 50 while solving frame 51,
        // while still dropping frame 48 after KF49. A missing keyframe has no
        // such open interval, so the current-frame branch above remains the
        // only visible sample in that case.
        keyframe_frames.last().is_some_and(|latest| {
            candidate_frame_id > *latest && candidate_frame_id < current_frame_id
        })
    }

    /// Build the factor-visible observation projection from the complete
    /// triangulation candidate history.  The candidate map already carries
    /// Basalt's timestamp/camera/frame order; no floating-point values are
    /// touched here.
    fn project_factor_observations(
        candidates: &BTreeMap<(i64, u16, u64), TrackHistory>,
        current_frame_id: u64,
        keyframe_frames: &[u64],
    ) -> Vec<StoredObservation> {
        candidates
            .values()
            .filter(|candidate| {
                Self::factor_frame_visible(candidate.frame_id, current_frame_id, keyframe_frames)
            })
            .map(|candidate| StoredObservation {
                frame_id: candidate.frame_id,
                camera_id: candidate.camera_id,
                pixel: candidate.pixel,
            })
            .collect()
    }

    /// Refresh stored camera poses from the optimized active window. The
    /// upstream observation database stores pixels separately from the pose
    /// state, and triangulation queries the latest `getPoseStateWithLin()`;
    /// retaining the pre-solve prediction here would silently change which
    /// historical candidate passes the baseline gate.
    fn refresh_track_history_poses(&mut self) {
        let frame_poses = self
            .window_poses
            .iter()
            .map(|pose| (pose.frame_id, pose.pose.clone()))
            .chain(self.window_states.iter().map(|state| {
                (
                    state.frame_id,
                    visual_state_pose_with_mode(state, self.config.scalar_mode),
                )
            }))
            .collect::<BTreeMap<_, _>>();
        let extrinsics = self.t_imu_cam.clone();
        for history in self.track_history.values_mut() {
            for entry in history {
                let Some(imu_pose) = frame_poses.get(&entry.frame_id) else {
                    continue;
                };
                let Some(extrinsic) = extrinsics.get(entry.camera_id as usize) else {
                    continue;
                };
                entry.imu_pose = imu_pose.clone();
                entry.pose = imu_pose.compose(extrinsic);
            }
        }
    }

    /// Keep only observations represented by the active pose/state window,
    /// matching upstream `prev_opt_flow_res` erasure during marginalization.
    fn prune_track_history(&mut self) {
        let active = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(self.window_states.iter().map(|state| state.frame_id))
            .collect::<std::collections::BTreeSet<_>>();
        for history in self.track_history.values_mut() {
            history.retain(|entry| active.contains(&entry.frame_id));
        }
        self.track_history.retain(|_, history| !history.is_empty());
    }

    fn build_problem(&self) -> WindowProblem {
        let pose_index_by_frame = self
            .window_poses
            .iter()
            .enumerate()
            .map(|(index, pose)| (pose.frame_id, index))
            .collect::<BTreeMap<_, _>>();
        let index_by_frame = self
            .window_states
            .iter()
            .enumerate()
            .map(|(index, state)| (state.frame_id, self.window_poses.len() + index))
            .collect::<BTreeMap<_, _>>();
        let state_index_by_frame = self
            .window_states
            .iter()
            .enumerate()
            .map(|(index, state)| (state.frame_id, index))
            .collect::<BTreeMap<_, _>>();
        // `LandmarkDatabase::getLandmarks()` is an aligned std::unordered_map
        // in the upstream implementation.  Its range-for order is part of
        // the floating-point accumulation path, so use the source-compatible
        // order sidecar retained by ObservationDb.
        let landmarks = self
            .landmarks
            .ordered_landmark_ids()
            .filter_map(|track_id| {
                let record = self.landmarks.landmarks.get(&track_id)?;
                if record.status == super::landmarks::LandmarkStatus::Removed {
                    return None;
                }
                let observations = self
                    .window_observations
                    .get(&track_id)?
                    .iter()
                    .filter_map(|observation| {
                        Some(WindowObservation {
                            state_index: pose_index_by_frame
                                .get(&observation.frame_id)
                                .copied()
                                .or_else(|| index_by_frame.get(&observation.frame_id).copied())?,
                            camera_id: observation.camera_id,
                            pixel: observation.pixel,
                        })
                    })
                    .collect::<Vec<_>>();
                if observations.is_empty() {
                    return None;
                }
                let anchor_state_index = pose_index_by_frame
                    .get(&record.landmark.anchor_pose)
                    .copied()
                    .or_else(|| index_by_frame.get(&record.landmark.anchor_pose).copied())?;
                Some(WindowLandmark {
                    track_id,
                    anchor_state_index,
                    anchor_camera_id: record.landmark.anchor_camera_id,
                    direction: record.landmark.direction,
                    inverse_distance: record.landmark.inverse_distance,
                    observations,
                })
            })
            .collect::<Vec<_>>();
        let imu_links = self
            .imu_links
            .iter()
            .filter_map(|(from, to, delta)| {
                Some(WindowImuLink {
                    from_index: *state_index_by_frame.get(from)?,
                    to_index: *state_index_by_frame.get(to)?,
                    delta: delta.clone(),
                })
            })
            .collect();
        WindowProblem {
            trial_host_order: self.native_host_order.keys().collect(),
            camera: self.camera,
            cameras: self.camera_models.clone(),
            t_imu_cam: self.t_imu_cam.clone(),
            poses: self.window_poses.clone(),
            states: self.window_states.clone(),
            landmarks,
            imu_links,
            imu_noise: self.imu_noise,
            bias_walk_noise: self.bias_walk_noise,
            initial_pose_weight: self.config.initial_pose_weight,
            initial_accel_bias_weight: self.config.initial_accel_bias_weight,
            initial_gyro_bias_weight: self.config.initial_gyro_bias_weight,
            prior: self.prior.clone(),
            anchor_point: self.anchor_point.clone(),
            gravity_world: self
                .gravity_world
                .unwrap_or_else(|| Vector3::new(0.0, 0.0, -9.81)),
            scalar_mode: self.config.scalar_mode,
        }
    }

    /// Construct the exact pre-shift problem used by the upstream
    /// marginalization AOM.  The newest state remains in `solved_problem` for
    /// the packet's full frame table, but is excluded from this clone and its
    /// factor rows.  State indices are stable because truncation only removes
    /// the suffix.
    fn pre_marginalization_snapshot(
        &self,
        solved_problem: &WindowProblem,
        solved_state: &DVector<f64>,
        boundary_state: Option<u64>,
        used_frames: &[u64],
        lost_landmarks: &[u64],
    ) -> Result<(WindowProblem, DVector<f64>, Vec<WhitenedFactorRowStack>), String> {
        let mut problem = solved_problem.clone();
        if let Some(boundary_id) = boundary_state {
            let boundary_index = problem
                .states
                .iter()
                .position(|state| state.frame_id == boundary_id)
                .ok_or_else(|| format!("marginalization boundary {boundary_id} is absent"))?;
            problem.states.truncate(boundary_index + 1);
            problem.imu_links.retain(|link| {
                link.from_index <= boundary_index && link.to_index <= boundary_index
            });
            // A host newer than the AOM cannot contribute a valid anchored
            // factor.  Basalt's landmark DB never hosts such a block during
            // this event; filtering it here keeps the Rust clone dimension
            // safe for malformed/synthetic streams without changing normal
            // rows.
            let block_count = problem.poses.len() + problem.states.len();
            problem
                .landmarks
                .retain(|landmark| landmark.anchor_state_index < block_count);
        }
        // `LinearizationAbsQR` constructs LandmarkBlock only for a host in
        // `kfs_to_marg` or a track in `lost_landmaks`; all other tracks are
        // absent from the marginal Q2 stack even when their observations fit
        // the truncated AOM. Match that detached native selection before
        // building grouped visual factors.
        let used_frames = used_frames
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let lost_landmarks = lost_landmarks
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>();
        let selected_landmarks = problem
            .landmarks
            .iter()
            .filter(|landmark| {
                lost_landmarks.contains(&landmark.track_id)
                    || problem
                        .block_frame_id(landmark.anchor_state_index)
                        .is_some_and(|frame_id| used_frames.contains(&frame_id))
            })
            .map(|landmark| landmark.track_id)
            .collect::<std::collections::BTreeSet<_>>();
        problem
            .landmarks
            .retain(|landmark| selected_landmarks.contains(&landmark.track_id));
        let state_dof = problem.state_dof();
        if solved_state.len() < state_dof {
            return Err(format!(
                "marginalization state has {} columns, need {state_dof}",
                solved_state.len()
            ));
        }
        let state = solved_state.rows(0, state_dof).into_owned();
        let factors = problem
            .linearize_snapshot(&state)
            .map_err(|error| format!("window pre-marginalization linearization: {error:?}"))?;
        Ok((problem, state, factors))
    }

    fn apply_landmark_writeback(&mut self, landmarks: &[WindowLandmark]) {
        for landmark in landmarks {
            let Some(anchor_frame_id) = self
                .window_poses
                .get(landmark.anchor_state_index)
                .map(|pose| pose.frame_id)
                .or_else(|| {
                    self.window_states
                        .get(
                            landmark
                                .anchor_state_index
                                .saturating_sub(self.window_poses.len()),
                        )
                        .map(|state| state.frame_id)
                })
            else {
                continue;
            };
            let anchor_imu_pose = self
                .window_poses
                .get(landmark.anchor_state_index)
                .map(|pose| pose.pose.clone())
                .or_else(|| {
                    self.window_states
                        .get(
                            landmark
                                .anchor_state_index
                                .saturating_sub(self.window_poses.len()),
                        )
                        .map(|state| state.nav.imu_to_world.clone())
                });
            let Some(anchor_imu_pose) = anchor_imu_pose else {
                continue;
            };
            let Some(anchor_pose) = self.camera_pose(&anchor_imu_pose, landmark.anchor_camera_id)
            else {
                continue;
            };
            let parameter = InverseDistanceLandmark {
                anchor_pose: anchor_frame_id,
                anchor_camera_id: landmark.anchor_camera_id,
                direction: landmark.direction,
                inverse_distance: landmark.inverse_distance,
            };
            let Some(point_anchor) = parameter.position_in_anchor() else {
                continue;
            };
            let point_world = anchor_pose.transform_point(&Point3::from(point_anchor));
            if point_world.coords.iter().all(|value| value.is_finite()) {
                self.landmark_world.insert(landmark.track_id, point_world);
                if let Some(record) = self.landmarks.landmarks.get_mut(&landmark.track_id) {
                    record.landmark = parameter;
                }
            }
        }
        self.reanchor_landmarks();
    }

    fn reanchor_landmarks(&mut self) {
        let Some((anchor_id, anchor_imu_pose)) = self
            .window_poses
            .first()
            .map(|pose| (pose.frame_id, pose.pose.clone()))
            .or_else(|| {
                self.window_states
                    .first()
                    .map(|state| (state.frame_id, state.nav.imu_to_world.clone()))
            })
        else {
            return;
        };
        // Upstream hosts VIO landmarks in a concrete TimeCamId. Re-host to
        // camera 0 of the oldest retained keyframe when its old host leaves
        // the active window.
        let anchor_camera_id = 0;
        let Some(anchor_pose) = self.camera_pose(&anchor_imu_pose, anchor_camera_id) else {
            return;
        };
        let active_ids = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(self.window_states.iter().map(|state| state.frame_id))
            .collect::<std::collections::BTreeSet<_>>();
        // Opt-in (`landmarks_all_cameras`): a landmark keeps its host camera
        // when it is re-hosted, because a cam1-only point may lie behind
        // cam0.  The default keeps the pinned camera-0 re-host above.
        let host_camera_poses = if self.config.landmarks_all_cameras {
            (0..self.t_imu_cam.len())
                .map(|camera_id| self.camera_pose(&anchor_imu_pose, camera_id as u16))
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        for (track_id, record) in &mut self.landmarks.landmarks {
            if active_ids.contains(&record.landmark.anchor_pose) {
                continue;
            }
            let Some(point_world) = self.landmark_world.get(track_id).copied() else {
                continue;
            };
            if self.config.landmarks_all_cameras {
                let host_camera_id = record.landmark.anchor_camera_id;
                let Some(Some(host_pose)) = host_camera_poses.get(host_camera_id as usize) else {
                    continue;
                };
                record
                    .landmark
                    .reanchor(anchor_id, host_camera_id, host_pose, point_world);
                continue;
            }
            record
                .landmark
                .reanchor(anchor_id, anchor_camera_id, &anchor_pose, point_world);
        }
    }

    /// The world-space cache is only a bridge across reanchoring.  Once the
    /// deferred-removal pass has run, records absent from the active landmark
    /// database cannot be reanchored on a later window shift.
    fn prune_landmark_world_cache(&mut self) {
        self.landmark_world
            .retain(|track_id, _| self.landmarks.landmarks.contains_key(track_id));
    }

    fn anchor_prior_for_kept(&self, dropped: &[u64]) -> Option<WindowPrior> {
        // Fallbacks are intentionally dimension-safe for mixed pose/state
        // windows.  A stale legacy 15-DoF carry cannot be added to a 6+15N
        // AOM; retain a finite gauge row for every surviving block instead.
        let kept_poses = self
            .window_poses
            .iter()
            .filter(|pose| !dropped.contains(&pose.frame_id))
            .cloned()
            .collect::<Vec<_>>();
        let kept_states = self
            .window_states
            .iter()
            .filter(|state| !dropped.contains(&state.frame_id))
            .cloned()
            .collect::<Vec<_>>();
        if kept_states.is_empty() && kept_poses.is_empty() {
            return None;
        }
        let mut kept = kept_poses
            .iter()
            .map(|pose| pose.frame_id)
            .collect::<Vec<_>>();
        kept.extend(
            kept_states
                .iter()
                .map(|state| state.frame_id)
                .collect::<Vec<_>>(),
        );
        let mut block_kinds = vec![WindowBlockKind::Pose; kept_poses.len()];
        block_kinds.extend(vec![WindowBlockKind::State; kept_states.len()]);
        let total_dof = kept_poses.len() * POSE_DOF + kept_states.len() * NAV_STATE_DOF;
        let mut jacobian = nalgebra::DMatrix::zeros(total_dof, total_dof);
        let rhs = DVector::zeros(total_dof);
        let fej_point = {
            let mut point = DVector::zeros(total_dof);
            let mut offset = 0;
            for pose in &kept_poses {
                point
                    .rows_mut(offset, POSE_DOF)
                    .copy_from(&flatten_pose_with_mode(
                        &pose.linearized_pose,
                        self.config.scalar_mode,
                    ));
                offset += POSE_DOF;
            }
            for state in &kept_states {
                point
                    .rows_mut(offset, NAV_STATE_DOF)
                    .copy_from(&flatten_nav_with_mode(
                        &state.linearized_nav,
                        self.config.scalar_mode,
                    ));
                offset += NAV_STATE_DOF;
            }
            point
        };
        for i in 0..total_dof {
            jacobian[(i, i)] = 1.0e3;
        }
        Some(WindowPrior {
            frame_ids: kept,
            block_kinds,
            jacobian,
            rhs,
            fej_point,
        })
    }

    fn calibrate_imu(&self, sample: ImuSample) -> ImuSample {
        if self.config.scalar_mode == ScalarMode::UpstreamF32 {
            let gyro = calibrate_gyro_f32(
                &self.calib_gyro_bias,
                Vector3::new(
                    sample.gyro_rad_s.x as f32,
                    sample.gyro_rad_s.y as f32,
                    sample.gyro_rad_s.z as f32,
                ),
            );
            let accel = calibrate_accel_f32(
                &self.calib_accel_bias,
                Vector3::new(
                    sample.accel_m_s2.x as f32,
                    sample.accel_m_s2.y as f32,
                    sample.accel_m_s2.z as f32,
                ),
            );
            return ImuSample::new(
                sample.timestamp_ns,
                gyro.map(|value| value as f64),
                accel.map(|value| value as f64),
            );
        }
        ImuSample::new(
            sample.timestamp_ns,
            calibrate_gyro(&self.calib_gyro_bias, sample.gyro_rad_s),
            calibrate_accel(&self.calib_accel_bias, sample.accel_m_s2),
        )
    }

    fn pose_for_frame(&self, frame_id: u64) -> Option<SE3> {
        self.window_poses
            .iter()
            .find(|pose| pose.frame_id == frame_id)
            .map(|pose| pose.pose.clone())
            .or_else(|| {
                self.window_states
                    .iter()
                    .find(|state| state.frame_id == frame_id)
                    .map(|state| state.nav.imu_to_world.clone())
            })
    }

    fn connected_ratio_for_kf(&self, frame_id: u64, current_frame_id: u64) -> f64 {
        let denominator = self.num_points_kf.get(&frame_id).copied().unwrap_or(0);
        if denominator == 0 {
            return 0.0;
        }
        // Upstream increments `num_points_connected[host]` inside the camera
        // loop.  A landmark tracked by both stereo cameras therefore
        // contributes two observations, not one unique landmark.  This is
        // observable at the old-KF threshold (for MH_01 frame 51, 8/51 stays
        // above 0.1 while the unique-landmark count 4/51 does not).
        let connected = self
            .landmarks
            .landmarks
            .iter()
            .filter_map(|(track_id, record)| {
                (record.status != super::landmarks::LandmarkStatus::Removed
                    && record.landmark.anchor_pose == frame_id)
                    .then(|| {
                        self.window_observations
                            .get(track_id)
                            .map(|observations| {
                                observations
                                    .iter()
                                    .filter(|observation| observation.frame_id == current_frame_id)
                                    .count()
                            })
                            .unwrap_or(0)
                    })
            })
            .sum::<usize>();
        connected as f64 / denominator as f64
    }

    fn marginalization_plan(&self, current_frame_id: u64) -> MarginalizationPlan {
        let trigger_states = self.window_states.len() >= self.config.window.max_states;
        let trigger_poses = self.window_poses.len() > self.config.window.max_kfs;
        if !trigger_states && !trigger_poses {
            return MarginalizationPlan::default();
        }

        let states_to_remove = if trigger_states {
            self.window_states
                .len()
                .saturating_sub(self.config.window.max_states)
                + 1
        } else {
            0
        };
        let mut plan = MarginalizationPlan::default();
        // `states_to_remove` is the number erased before the retained FEJ
        // boundary, not the boundary's own index.  This is the subtle
        // upstream protocol that keeps the newest state out of the AOM while
        // retaining it in the full frame-state map for the marginal packet.
        plan.boundary_state = trigger_states
            .then(|| {
                self.window_states
                    .get(states_to_remove)
                    .map(|state| state.frame_id)
            })
            .flatten();
        for state in self.window_states.iter().take(states_to_remove) {
            if state.is_keyframe {
                plan.convert_states.push(state.frame_id);
            } else {
                plan.drop_states.push(state.frame_id);
            }
        }

        // The upstream policy considers the newly converted pose before
        // selecting an old keyframe.  Build a deterministic KF list and skip
        // its newest two entries exactly as sqrt_keypoint_vio.cpp does.
        let mut kfs = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(
                self.window_states
                    .iter()
                    .filter(|state| state.is_keyframe)
                    .map(|state| state.frame_id),
            )
            .collect::<Vec<_>>();
        kfs.extend(plan.convert_states.iter().copied());
        kfs.sort_unstable();
        kfs.dedup();
        // Upstream only enters the max-KF selection loop in the same
        // marginalization event that converted an old full keyframe state to
        // a pose-only block.  A newly inserted KF may therefore temporarily
        // make the set eight entries wide until that conversion two frames
        // later.
        if plan.convert_states.is_empty() {
            return plan;
        }
        let mut keep_kfs = kfs.len();
        while keep_kfs > self.config.window.max_kfs {
            if kfs.len() <= 2 {
                break;
            }
            let candidate_end = kfs.len() - 2;
            let mut selected = None;
            for id in &kfs[..candidate_end] {
                if self.connected_ratio_for_kf(*id, current_frame_id)
                    < self.config.window.min_feature_ratio
                {
                    selected = Some(*id);
                    break;
                }
            }
            let selected = selected.or_else(|| {
                let latest = *kfs.last()?;
                let latest_pose = self.pose_for_frame(latest)?;
                kfs[..candidate_end]
                    .iter()
                    .filter_map(|id| {
                        let pose = self.pose_for_frame(*id)?;
                        let denom = kfs[..candidate_end]
                            .iter()
                            .filter_map(|other| self.pose_for_frame(*other))
                            .map(|other_pose| {
                                1.0 / ((pose.translation - other_pose.translation).norm() + 1e-5)
                            })
                            .sum::<f64>();
                        let score =
                            (pose.translation - latest_pose.translation).norm().sqrt() * denom;
                        Some((*id, score))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(id, _)| id)
            });
            let Some(selected) = selected else { break };
            kfs.retain(|id| *id != selected);
            keep_kfs = keep_kfs.saturating_sub(1);
            if self
                .window_poses
                .iter()
                .any(|pose| pose.frame_id == selected)
            {
                plan.drop_poses.push(selected);
            } else if plan.convert_states.iter().any(|id| *id == selected) {
                // Upstream converts every old keyframe state first, then
                // erases selected `poses_to_marg`.  Keeping the conversion
                // entry is therefore essential: dropping it here would
                // incorrectly leave a 15-DoF state in the active AOM.
                plan.drop_poses.push(selected);
            }
        }
        plan
    }

    fn cam0_connectivity(&self, observations: &[TrackObservation]) -> (usize, usize) {
        let mut track_ids = observations
            .iter()
            .filter(|observation| observation.camera_id == 0)
            .map(|observation| observation.track_id)
            .collect::<Vec<_>>();
        track_ids.sort_unstable();
        track_ids.dedup();
        track_ids
            .into_iter()
            .fold((0usize, 0usize), |(connected, unconnected), track_id| {
                let exists = self
                    .landmarks
                    .landmarks
                    .get(&track_id)
                    .is_some_and(|record| {
                        record.status != super::landmarks::LandmarkStatus::Removed
                    });
                if exists {
                    (connected + 1, unconnected)
                } else {
                    (connected, unconnected + 1)
                }
            })
    }

    fn decide_keyframe(&mut self, connected: usize, unconnected: usize) -> bool {
        let total = connected + unconnected;
        if total > 0 {
            let ratio = connected as f64 / total as f64;
            // This strict comparison is intentional and matches
            // sqrt_keypoint_vio.cpp:370-373 at the pinned SHA.
            let regular = ratio < self.config.new_kf_keypoints_threshold
                && self.frames_after_kf > self.config.min_frames_after_kf;
            let urgent = self.config.urgent_kf.is_some_and(|policy| {
                ratio < policy.threshold && self.frames_after_kf > policy.min_frames_after_kf
            });
            if regular || urgent {
                self.take_kf = true;
            }
        }

        if self.take_kf {
            self.take_kf = false;
            self.frames_after_kf = 0;
            true
        } else {
            self.frames_after_kf = self.frames_after_kf.saturating_add(1);
            false
        }
    }

    fn refresh_public_states(&mut self) {
        self.states = self
            .window_states
            .iter()
            .map(|state| {
                let q = state.nav.imu_to_world.rotation.quaternion();
                FrameStateData {
                    frame_id: state.frame_id,
                    timestamp_ns: state.timestamp_ns,
                    pose: [
                        state.nav.imu_to_world.translation.x,
                        state.nav.imu_to_world.translation.y,
                        state.nav.imu_to_world.translation.z,
                        q.w,
                        q.i,
                        q.j,
                        q.k,
                    ],
                    velocity: state.nav.velocity_world_m_s.into(),
                    gyro_bias: state.nav.gyro_bias_rad_s.into(),
                    accel_bias: state.nav.accel_bias_m_s2.into(),
                    linearized: state.linearized,
                    is_keyframe: state.is_keyframe,
                    is_latest: state.is_latest,
                }
            })
            .collect();
        if let Some(latest) = self.window_states.last() {
            self.nav = latest.nav.clone();
        }
    }

    fn factor_row_counts(
        problem: &WindowProblem,
        factors: &[WhitenedFactorRowStack],
    ) -> [usize; 4] {
        let prior_factor_count = if problem.prior.is_some() {
            1
        } else if problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
        {
            1
        } else {
            0
        };
        let prefix = prior_factor_count.min(factors.len());
        let imu_factor_count = problem
            .imu_links
            .len()
            .saturating_mul(2)
            .min(factors.len().saturating_sub(prefix));
        let visual_end = factors.len().saturating_sub(imu_factor_count);
        let rows =
            |slice: &[WhitenedFactorRowStack]| slice.iter().map(WhitenedFactorRowStack::rows).sum();
        let mut imu_rows = 0;
        let mut bias_rows = 0;
        for pair in factors[visual_end..].chunks(2) {
            if let Some(factor) = pair.first() {
                imu_rows += factor.rows();
            }
            if let Some(factor) = pair.get(1) {
                bias_rows += factor.rows();
            }
        }
        [
            rows(&factors[..prefix]),
            rows(&factors[prefix..visual_end]),
            imu_rows,
            bias_rows,
        ]
    }

    fn pose_wire(pose: &SE3) -> [f64; 7] {
        let q = pose.rotation.quaternion();
        [
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            q.w,
            q.i,
            q.j,
            q.k,
        ]
    }

    fn nav_wire(nav: &BasaltNavState) -> NavStateData {
        NavStateData {
            pose: Self::pose_wire(&nav.imu_to_world),
            velocity: nav.velocity_world_m_s.into(),
            gyro_bias: nav.gyro_bias_rad_s.into(),
            accel_bias: nav.accel_bias_m_s2.into(),
        }
    }

    fn delta_wire<const N: usize>(delta: &DVector<f64>, label: &str) -> [f64; N] {
        assert_eq!(
            delta.len(),
            N,
            "{label} must have the upstream fixed tangent dimension"
        );
        let mut out = [0.0; N];
        out.copy_from_slice(delta.as_slice());
        out
    }

    fn pose_fej_wire(pose: &WindowPose) -> PoseStateWithLinData {
        PoseStateWithLinData {
            pose_linearized: Self::pose_wire(&pose.linearized_pose),
            pose_current: Self::pose_wire(&pose.stored_current_pose),
            delta: Self::delta_wire::<POSE_DOF>(&pose.linearized_delta, "pose delta"),
            // A pose-only block enters the upstream mapper as a
            // PoseStateWithLin after a state conversion, which is the frozen
            // (linearized=true) branch.  The effective pose is therefore the
            // explicitly stored current pose.
            linearized: true,
        }
    }

    fn state_fej_wire(state: &WindowState) -> PoseVelBiasStateWithLinData {
        PoseVelBiasStateWithLinData {
            state_linearized: Self::nav_wire(&state.linearized_nav),
            state_current: Self::nav_wire(&state.stored_current_nav),
            delta: Self::delta_wire::<NAV_STATE_DOF>(
                &state.linearized_delta,
                "navigation-state delta",
            ),
            linearized: state.linearized,
        }
    }

    fn prior_data(prior: &WindowPrior) -> Option<PriorData> {
        let block_kinds = prior
            .kinds()
            .into_iter()
            .map(|kind| match kind {
                WindowBlockKind::Pose => "pose".into(),
                WindowBlockKind::StatePose => "state_pose".into(),
                WindowBlockKind::State => "state".into(),
            })
            .collect::<Vec<String>>();
        MatrixData::new(
            prior.jacobian.nrows(),
            prior.jacobian.ncols(),
            prior.jacobian.iter().copied().collect(),
        )
        .map(|jacobian| PriorData {
            frame_ids: prior.frame_ids.clone(),
            block_kinds,
            jacobian,
            rhs: prior.rhs.iter().copied().collect(),
            fej_point: prior.fej_point.iter().copied().collect(),
        })
    }

    /// Serialize prior block kinds against the frame tables carried by a
    /// MargData record.  Internally `WindowBlockKind::Pose` is also used for
    /// the six-DoF prefix retained from a full navigation state; on the wire
    /// that block must be named `state_pose` when its frame resolves through
    /// `frame_states`, otherwise strict schema-4 validation correctly rejects
    /// the record as a pose/state mismatch.
    fn prior_data_for_tables(
        prior: &WindowPrior,
        poses: &[WindowPose],
        states: &[WindowState],
    ) -> Option<PriorData> {
        let pose_ids = poses
            .iter()
            .map(|pose| pose.frame_id)
            .collect::<BTreeSet<_>>();
        let state_ids = states
            .iter()
            .map(|state| state.frame_id)
            .collect::<BTreeSet<_>>();
        let block_kinds = prior
            .kinds()
            .into_iter()
            .zip(prior.frame_ids.iter())
            .map(|(kind, frame_id)| match kind {
                WindowBlockKind::Pose
                    if state_ids.contains(frame_id) && !pose_ids.contains(frame_id) =>
                {
                    "state_pose".into()
                }
                WindowBlockKind::Pose => "pose".into(),
                WindowBlockKind::StatePose => "state_pose".into(),
                WindowBlockKind::State => "state".into(),
            })
            .collect::<Vec<String>>();
        MatrixData::new(
            prior.jacobian.nrows(),
            prior.jacobian.ncols(),
            prior.jacobian.iter().copied().collect(),
        )
        .map(|jacobian| PriorData {
            frame_ids: prior.frame_ids.clone(),
            block_kinds,
            jacobian,
            rhs: prior.rhs.iter().copied().collect(),
            fej_point: prior.fej_point.iter().copied().collect(),
        })
    }

    fn append_margdata_sidecar(sidecar: &MargDataDiagnosticSidecar) {
        let Some(path) = diagnostic_env_snapshot().margdata_sidecar.as_ref() else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };
        if sidecar.write_json(&mut file).is_ok() {
            let _ = file.write_all(b"\n");
        }
    }

    /// Emit the exact boundary at which the detached pre-marginalization AOM
    /// is constructed. Unlike the causal identity sidecar (which is written
    /// before solve), this record is written after solve_lm has completed and
    /// immediately after linearize_snapshot has produced the Q2 input.
    /// It is diagnostic-only and therefore has no effect when the path is
    /// unset.
    fn append_q2_boundary_trace(
        event_frame_id: u64,
        event_timestamp_ns: i64,
        solved_problem: &WindowProblem,
        aom_problem: &WindowProblem,
        boundary_state: Option<u64>,
        factors: &[WhitenedFactorRowStack],
    ) {
        let Some(path) = diagnostic_env_snapshot().q2_boundary_trace.as_ref() else {
            return;
        };
        let frame_timestamp = |problem: &WindowProblem, frame_id: u64| {
            problem
                .poses
                .iter()
                .find(|pose| pose.frame_id == frame_id)
                .map(|pose| pose.timestamp_ns)
                .or_else(|| {
                    problem
                        .states
                        .iter()
                        .find(|state| state.frame_id == frame_id)
                        .map(|state| state.timestamp_ns)
                })
        };
        let identity = |problem: &WindowProblem, index: usize| {
            problem.block_frame_id(index).map(|frame_id| {
                json!({
                    "frame_id": frame_id,
                    "timestamp_ns": frame_timestamp(problem, frame_id),
                    "block_index": index,
                    "kind": if index < problem.poses.len() { "pose" } else { "state" },
                })
            })
        };
        let boundary = boundary_state.map(|frame_id| {
            json!({
                "frame_id": frame_id,
                "timestamp_ns": frame_timestamp(solved_problem, frame_id),
            })
        });
        let solve_frame = solved_problem
            .states
            .last()
            .map(|state| (state.frame_id, state.timestamp_ns))
            .or_else(|| {
                solved_problem
                    .poses
                    .last()
                    .map(|pose| (pose.frame_id, pose.timestamp_ns))
            });
        let solve_entry = json!({
            "frame_id": solve_frame.map(|(frame_id, _)| frame_id),
            "timestamp_ns": solve_frame.map(|(_, timestamp_ns)| timestamp_ns),
            "event_frame_id": event_frame_id,
            "event_timestamp_ns": event_timestamp_ns,
            "active_block_count": solved_problem.poses.len() + solved_problem.states.len(),
            "state_dof": solved_problem.state_dof(),
            "semantic_stage": "after_solve_lm_before_pre_marginalization_snapshot",
        });
        let aom_entry = json!({
            "frame_id": event_frame_id,
            "timestamp_ns": event_timestamp_ns,
            "boundary_state": boundary,
            "active_block_count": aom_problem.poses.len() + aom_problem.states.len(),
            "state_dof": aom_problem.state_dof(),
            "factor_count": factors.len(),
            "factor_rows": factors.iter().map(WhitenedFactorRowStack::rows).sum::<usize>(),
            "semantic_stage": "linearize_snapshot_output_q2_input",
        });
        let solved_blocks = (0..(solved_problem.poses.len() + solved_problem.states.len()))
            .filter_map(|index| identity(solved_problem, index))
            .collect::<Vec<_>>();
        let aom_blocks = (0..(aom_problem.poses.len() + aom_problem.states.len()))
            .filter_map(|index| identity(aom_problem, index))
            .collect::<Vec<_>>();
        let landmarks = aom_problem
            .landmarks
            .iter()
            .enumerate()
            .map(|(landmark_index, landmark)| {
                let observations = landmark
                    .observations
                    .iter()
                    .enumerate()
                    .map(|(ordinal, observation)| {
                        let target_frame_id = solved_problem.block_frame_id(observation.state_index);
                        let in_aom =
                            aom_problem.block_frame_id(observation.state_index).is_some();
                        json!({
                            "ordinal": ordinal,
                            "state_index": observation.state_index,
                            "camera_id": observation.camera_id,
                            "target_frame_id": target_frame_id,
                            "target_timestamp_ns": target_frame_id.and_then(|frame_id| frame_timestamp(solved_problem, frame_id)),
                            "is_current_event_frame": target_frame_id == Some(event_frame_id),
                            "in_solved_window": target_frame_id.is_some(),
                            "in_aom": in_aom,
                            "native_slot_rows": [ordinal.saturating_mul(2), ordinal.saturating_mul(2).saturating_add(2)],
                            "rust_drop_reason": (!in_aom).then_some("outside_truncated_aom"),
                        })
                    })
                    .collect::<Vec<_>>();
                let null_rows = landmark
                    .observations
                    .iter()
                    .filter(|observation| {
                        aom_problem.block_frame_id(observation.state_index).is_none()
                    })
                    .count()
                    .saturating_mul(2);
                json!({
                    "landmark_index": landmark_index,
                    "track_id": landmark.track_id,
                    "host_frame_id": aom_problem.block_frame_id(landmark.anchor_state_index),
                    "host_timestamp_ns": aom_problem.block_frame_id(landmark.anchor_state_index).and_then(|frame_id| frame_timestamp(solved_problem, frame_id)),
                    "observation_count": landmark.observations.len(),
                    "native_null_slot_rows": null_rows,
                    "observations": observations,
                })
            })
            .collect::<Vec<_>>();
        let mut row_counts = BTreeMap::<String, usize>::new();
        let factor_records = factors
            .iter()
            .enumerate()
            .map(|(factor_index, factor)| {
                let kind = format!("{:?}", factor.kind).to_lowercase();
                *row_counts.entry(kind.clone()).or_default() += factor.rows();
                let landmark = factor.landmark_metadata.and_then(|metadata| {
                    aom_problem
                        .landmarks
                        .get(metadata.landmark_index)
                        .map(|landmark| {
                            let null_rows = landmark
                                .observations
                                .iter()
                                .filter(|observation| {
                                    aom_problem.block_frame_id(observation.state_index).is_none()
                                })
                                .count()
                                .saturating_mul(2);
                            json!({
                                "landmark_index": metadata.landmark_index,
                                "track_id": metadata.track_id,
                                "rows": factor.rows(),
                                "native_null_slot_rows": null_rows,
                                "native_expected_rows": landmark.observations.len().saturating_mul(2),
                            })
                        })
                });
                json!({
                    "factor_index": factor_index,
                    "kind": kind,
                    "rows": factor.rows(),
                    "state_columns": factor.state_jacobian.ncols(),
                    "landmark_columns": factor.landmark_jacobian.ncols(),
                    "landmark": landmark,
                })
            })
            .collect::<Vec<_>>();
        let record = json!({
            "schema": "basalt.m11.q2_boundary_trace.v1",
            "source": "rust",
            "event_frame_id": event_frame_id,
            "event_timestamp_ns": event_timestamp_ns,
            "current_observation_frame_id": event_frame_id,
            "boundary_state": boundary_state,
            "solve_lm_entry": solve_entry,
            "q2_construction_entry": aom_entry,
            "solved_window_blocks": solved_blocks,
            "aom_blocks": aom_blocks,
            "row_counts": row_counts,
            "factor_records": factor_records,
            "landmarks": landmarks,
            "capture_contract": {
                "factor_rows_include_out_of_aom_observation_slots": true,
                "out_of_aom_slot_semantics": "native_null_pose_linearization_zero_rows",
                "capture_stage": "post_solve_lm_pre_marginal_prior_shift",
            },
        });
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            if let Ok(line) = serde_json::to_string(&record) {
                let _ = file.write_all(line.as_bytes());
                let _ = file.write_all(b"\n");
            }
        }
    }

    /// Emit the causal visual observation/landmark identity stream used by
    /// cross-implementation track audits.  This intentionally runs only when
    /// its dedicated diagnostic path was captured in the process snapshot. It
    /// reads the already-owned estimator records and does not construct a
    /// WindowProblem, linearize a factor, or mutate solver state.
    fn append_visual_factor_identity_trace(&self, frame_id: u64, timestamp_ns: i64) {
        let policy = diagnostic_env_snapshot();
        let Some(path) = policy.visual_factor_identity_trace.as_ref() else {
            return;
        };

        let mut track_ids = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for track_id in self.landmarks.ordered_landmark_ids() {
            if seen.insert(track_id) {
                track_ids.push(track_id);
            }
        }
        for track_id in self.window_observations.keys().copied() {
            if seen.insert(track_id) {
                track_ids.push(track_id);
            }
        }
        for track_id in self.track_history.keys().copied() {
            if seen.insert(track_id) {
                track_ids.push(track_id);
            }
        }
        if let Some(filter) = policy.visual_factor_identity_tracks.as_ref() {
            track_ids.retain(|track_id| filter.contains(track_id));
        }

        let frame_timestamp = |target_frame: u64| {
            self.window_states
                .iter()
                .find(|state| state.frame_id == target_frame)
                .map(|state| state.timestamp_ns)
                .or_else(|| {
                    self.window_poses
                        .iter()
                        .find(|pose| pose.frame_id == target_frame)
                        .map(|pose| pose.timestamp_ns)
                })
                .or_else(|| (target_frame == frame_id).then_some(timestamp_ns))
        };

        // Mirror build_problem's visual-factor eligibility without creating a
        // second WindowProblem: source order is the ObservationDb order, a
        // host must still be in the active pose/state window, and at least one
        // observation must map to an active frame.  The ordinal is global (not
        // filtered by VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS).
        let active_frames = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(self.window_states.iter().map(|state| state.frame_id))
            .collect::<std::collections::BTreeSet<_>>();
        let mut factor_metadata = BTreeMap::new();
        let mut next_factor_ordinal = 0usize;
        for track_id in self.landmarks.ordered_landmark_ids() {
            let Some(record) = self.landmarks.landmarks.get(&track_id) else {
                continue;
            };
            let valid_observations = self
                .window_observations
                .get(&track_id)
                .map(|observations| {
                    observations
                        .iter()
                        .filter(|observation| active_frames.contains(&observation.frame_id))
                        .count()
                })
                .unwrap_or(0);
            let included = record.status != LandmarkStatus::Removed
                && active_frames.contains(&record.landmark.anchor_pose)
                && valid_observations > 0;
            if included {
                factor_metadata.insert(
                    track_id,
                    (next_factor_ordinal, valid_observations.saturating_mul(2)),
                );
                next_factor_ordinal = next_factor_ordinal.saturating_add(1);
            }
        }

        let mut landmarks = Vec::with_capacity(track_ids.len());
        for track_id in track_ids {
            let landmark_record = self.landmarks.landmarks.get(&track_id);
            let (status, active, delete_reason) = match landmark_record {
                Some(record) => match record.status {
                    LandmarkStatus::Active => ("active", true, None),
                    LandmarkStatus::Lost => {
                        ("lost", false, Some("missed_current_frame".to_string()))
                    }
                    LandmarkStatus::Removed => (
                        "removed",
                        false,
                        Some("removed_from_landmark_db".to_string()),
                    ),
                },
                None => (
                    "untriangulated",
                    false,
                    Some("not_yet_triangulated".to_string()),
                ),
            };

            let (
                host_frame_id,
                host_camera_id,
                host_timestamp_ns,
                direction_xy,
                inverse_distance,
                connected_observations,
                missed_frames,
            ) = landmark_record
                .map(|record| {
                    (
                        Some(record.landmark.anchor_pose),
                        Some(record.landmark.anchor_camera_id),
                        frame_timestamp(record.landmark.anchor_pose),
                        Some([
                            record.landmark.direction.xy.x,
                            record.landmark.direction.xy.y,
                        ]),
                        Some(record.landmark.inverse_distance),
                        Some(record.connected_observations),
                        Some(record.missed_frames),
                    )
                })
                .unwrap_or((None, None, None, None, None, None, None));
            let world_point = self
                .landmark_world
                .get(&track_id)
                .map(|point| [point.x, point.y, point.z]);
            let valid_observation_count = self
                .window_observations
                .get(&track_id)
                .map(|observations| {
                    observations
                        .iter()
                        .filter(|observation| active_frames.contains(&observation.frame_id))
                        .count()
                })
                .unwrap_or(0);
            let (factor_ordinal, factor_row_count) = factor_metadata
                .get(&track_id)
                .copied()
                .map_or((None, None), |(ordinal, rows)| (Some(ordinal), Some(rows)));
            let factor_included = factor_ordinal.is_some();
            let factor_inclusion_reason = if factor_included {
                "host_and_observations_in_active_window"
            } else if landmark_record.is_none() {
                "landmark_not_triangulated"
            } else if landmark_record.is_some_and(|record| record.status == LandmarkStatus::Removed)
            {
                "landmark_record_removed"
            } else if host_frame_id.is_none()
                || !host_frame_id.is_some_and(|host| active_frames.contains(&host))
            {
                "host_outside_active_window"
            } else if valid_observation_count == 0 {
                "no_observations_in_active_window"
            } else {
                "factor_rejected_by_window_mapping"
            };
            let factor_stage = if factor_included {
                "pre_solve_visual_factor"
            } else if status == "lost" {
                "post_association_lost_before_solve"
            } else if status == "untriangulated" {
                "post_association_untriangulated_before_solve"
            } else {
                "post_association_not_in_visual_factor"
            };

            let observations = self
                .window_observations
                .get(&track_id)
                .into_iter()
                .flatten()
                .enumerate()
                .map(|(obs_ordinal, observation)| {
                    let target_timestamp_ns = self
                        .track_history
                        .get(&track_id)
                        .into_iter()
                        .flatten()
                        .find(|entry| {
                            entry.frame_id == observation.frame_id
                                && entry.camera_id == observation.camera_id
                        })
                        .map(|entry| entry.timestamp_ns)
                        .or_else(|| frame_timestamp(observation.frame_id));
                    let in_window = active_frames.contains(&observation.frame_id);
                    json!({
                        "obs_ordinal": obs_ordinal,
                        "target_frame_id": observation.frame_id,
                        "target_timestamp_ns": target_timestamp_ns,
                        "target_camera_id": observation.camera_id,
                        "target_cam": observation.camera_id,
                        "pixel": [observation.pixel.x, observation.pixel.y],
                        "in_window": in_window,
                        "active": active,
                        // ObservationDb has no outlier flag/reason.  Preserve
                        // that absence explicitly instead of claiming a
                        // native outlier decision that Rust never made.
                        "outlier": null,
                        "outlier_reason": null,
                        "delete_reason": delete_reason,
                    })
                })
                .collect::<Vec<_>>();

            landmarks.push(json!({
                "track_id": track_id,
                "factor_present": landmark_record.is_some(),
                "factor_included": factor_included,
                "factor_ordinal": factor_ordinal,
                "factor_row_count": factor_row_count,
                "factor_observation_count": valid_observation_count,
                "factor_inclusion_reason": factor_inclusion_reason,
                "factor_stage": factor_stage,
                "active": active,
                "outlier": null,
                "outlier_reason": null,
                "geometric_outlier": null,
                "geometric_outlier_reason": "not_classified_by_rust_vio_factor_builder",
                "delete_reason": delete_reason,
                "host_frame_id": host_frame_id,
                "host_timestamp_ns": host_timestamp_ns,
                "host_camera_id": host_camera_id,
                "host_cam": host_camera_id,
                "landmark_state": {
                    "status": status,
                    "host_frame_id": host_frame_id,
                    "host_timestamp_ns": host_timestamp_ns,
                    "host_camera_id": host_camera_id,
                    "direction_xy": direction_xy,
                    "inverse_distance": inverse_distance,
                    "world_point": world_point,
                    "connected_observations": connected_observations,
                    "missed_frames": missed_frames,
                    "outlier_tracking": "not_available_in_rust_observation_db",
                },
                "observations": observations,
            }));
        }

        let record = json!({
            "schema": "basalt.rust.visual_factor_identity.v1",
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "boundary": "post_association_pre_solve",
            "observation_storage": "window_observations_insertion_order",
            "track_filter": policy.visual_factor_identity_tracks,
            "landmarks": landmarks,
        });
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let Ok(line) = serde_json::to_string(&record) else {
            return;
        };
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
            return;
        };
        let _ = file.write_all(line.as_bytes());
        let _ = file.write_all(b"\n");
    }

    /// Emit frame-boundary store counts, high-water marks, and ID-level
    /// evictions for an explicit lifecycle audit. This hook is deliberately
    /// called after all estimator mutations for the frame and only performs
    /// work when its dedicated path was captured before the first call. It
    /// never enters the retained diagnostic solver path and does not inspect
    /// floating-point factors.
    fn append_lifecycle_trace(
        &mut self,
        frame_id: u64,
        timestamp_ns: i64,
        is_keyframe: bool,
        marginalized: bool,
    ) {
        let Some(path) = diagnostic_env_snapshot().lifecycle_trace.as_ref() else {
            return;
        };

        let track_history_ids = self.track_history.keys().copied().collect::<BTreeSet<_>>();
        let factor_ids = self
            .window_observations
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        let landmark_ids = self
            .landmarks
            .landmarks
            .keys()
            .copied()
            .collect::<BTreeSet<_>>();
        let world_ids = self.landmark_world.keys().copied().collect::<BTreeSet<_>>();
        let world_entries = world_ids.len();
        let track_history_entries = self.track_history.values().map(Vec::len).sum::<usize>();
        let track_history_capacity = self
            .track_history
            .values()
            .map(Vec::capacity)
            .sum::<usize>();
        let factor_records = self
            .window_observations
            .values()
            .map(Vec::len)
            .sum::<usize>();
        let factor_capacity = self
            .window_observations
            .values()
            .map(Vec::capacity)
            .sum::<usize>();
        let landmark_count = self.landmarks.landmarks.len();
        let landmark_observation_records = self.landmarks.observations.len();
        let landmark_observation_capacity = self.landmarks.observations.capacity();
        let snapshot = LifecycleTraceSnapshot {
            frame_id,
            track_history_ids,
            factor_ids,
            landmark_ids,
            world_ids,
            track_history_entries,
            track_history_capacity,
            factor_records,
            factor_capacity,
            landmark_count,
            landmark_observation_records,
            landmark_observation_capacity,
            world_entries,
        };

        // Take the prior snapshot out first so the ID differences below do
        // not hold an immutable borrow while the peak/last fields update.
        let previous = self.lifecycle_trace_previous.take();
        self.lifecycle_trace_peak.track_history_tracks = self
            .lifecycle_trace_peak
            .track_history_tracks
            .max(snapshot.track_history_ids.len());
        self.lifecycle_trace_peak.track_history_entries = self
            .lifecycle_trace_peak
            .track_history_entries
            .max(snapshot.track_history_entries);
        self.lifecycle_trace_peak.track_history_capacity = self
            .lifecycle_trace_peak
            .track_history_capacity
            .max(snapshot.track_history_capacity);
        self.lifecycle_trace_peak.factor_tracks = self
            .lifecycle_trace_peak
            .factor_tracks
            .max(snapshot.factor_ids.len());
        self.lifecycle_trace_peak.factor_records = self
            .lifecycle_trace_peak
            .factor_records
            .max(snapshot.factor_records);
        self.lifecycle_trace_peak.factor_capacity = self
            .lifecycle_trace_peak
            .factor_capacity
            .max(snapshot.factor_capacity);
        self.lifecycle_trace_peak.landmark_count = self
            .lifecycle_trace_peak
            .landmark_count
            .max(snapshot.landmark_count);
        self.lifecycle_trace_peak.landmark_observation_records = self
            .lifecycle_trace_peak
            .landmark_observation_records
            .max(snapshot.landmark_observation_records);
        self.lifecycle_trace_peak.landmark_observation_capacity = self
            .lifecycle_trace_peak
            .landmark_observation_capacity
            .max(snapshot.landmark_observation_capacity);
        self.lifecycle_trace_peak.world_entries = self
            .lifecycle_trace_peak
            .world_entries
            .max(snapshot.world_entries);

        let removed_ids = |previous: Option<&LifecycleTraceSnapshot>,
                           current: &BTreeSet<TrackId>| {
            previous
                .into_iter()
                .flat_map(|previous| previous.track_history_ids.difference(current))
                .copied()
                .collect::<Vec<_>>()
        };
        let removed_factor_ids_for =
            |previous: Option<&LifecycleTraceSnapshot>, current: &BTreeSet<TrackId>| {
                previous
                    .into_iter()
                    .flat_map(|previous| previous.factor_ids.difference(current))
                    .copied()
                    .collect::<Vec<_>>()
            };
        let removed_landmark_ids_for =
            |previous: Option<&LifecycleTraceSnapshot>, current: &BTreeSet<TrackId>| {
                previous
                    .into_iter()
                    .flat_map(|previous| previous.landmark_ids.difference(current))
                    .copied()
                    .collect::<Vec<_>>()
            };
        let removed_world_ids_for =
            |previous: Option<&LifecycleTraceSnapshot>, current: &BTreeSet<TrackId>| {
                previous
                    .into_iter()
                    .flat_map(|previous| previous.world_ids.difference(current))
                    .copied()
                    .collect::<Vec<_>>()
            };
        let removed_track_ids = removed_ids(previous.as_ref(), &snapshot.track_history_ids);
        let removed_factor_ids = removed_factor_ids_for(previous.as_ref(), &snapshot.factor_ids);
        let removed_landmark_ids =
            removed_landmark_ids_for(previous.as_ref(), &snapshot.landmark_ids);
        let removed_world_ids = removed_world_ids_for(previous.as_ref(), &snapshot.world_ids);
        let net_drop = |previous: Option<&LifecycleTraceSnapshot>,
                        previous_value: fn(&LifecycleTraceSnapshot) -> usize,
                        current: usize| {
            previous
                .map(|previous| previous_value(previous).saturating_sub(current))
                .unwrap_or(0)
        };

        let previous_frame_id = previous.as_ref().map(|previous| previous.frame_id);
        let record = json!({
            "schema": "basalt.rust.lifecycle.v1",
            "frame_id": frame_id,
            "timestamp_ns": timestamp_ns,
            "is_keyframe": is_keyframe,
            "marginalized": marginalized,
            "previous_frame_id": previous_frame_id,
            "stores": {
                "track_history": {
                    "track_count": snapshot.track_history_ids.len(),
                    "entry_count": snapshot.track_history_entries,
                    "capacity_sum": snapshot.track_history_capacity,
                    "capacity_observed": true
                },
                "window_observations_factor_map": {
                    "track_count": snapshot.factor_ids.len(),
                    "record_count": snapshot.factor_records,
                    "capacity_sum": snapshot.factor_capacity,
                    "capacity_observed": true
                },
                "landmark_db": {
                    "record_count": snapshot.landmark_count,
                    "active_count": self.landmarks.landmarks.values().filter(|record| record.status == LandmarkStatus::Active).count(),
                    "lost_count": self.landmarks.landmarks.values().filter(|record| record.status == LandmarkStatus::Lost).count(),
                    "removed_count": self.landmarks.landmarks.values().filter(|record| record.status == LandmarkStatus::Removed).count(),
                    "observation_record_count": snapshot.landmark_observation_records,
                    "observation_capacity": snapshot.landmark_observation_capacity,
                    "capacity_observed": true
                },
                "landmark_world_cache": {
                    "entry_count": snapshot.world_entries,
                    "capacity_observed": false
                },
                "window": {
                    "state_count": self.window_states.len(),
                    "pose_count": self.window_poses.len(),
                    "imu_link_count": self.imu_links.len(),
                    "of_image_frame_count": self.of_images.len()
                }
            },
            "peaks": {
                "track_history_tracks": self.lifecycle_trace_peak.track_history_tracks,
                "track_history_entries": self.lifecycle_trace_peak.track_history_entries,
                "track_history_capacity_sum": self.lifecycle_trace_peak.track_history_capacity,
                "factor_tracks": self.lifecycle_trace_peak.factor_tracks,
                "factor_records": self.lifecycle_trace_peak.factor_records,
                "factor_capacity_sum": self.lifecycle_trace_peak.factor_capacity,
                "landmark_records": self.lifecycle_trace_peak.landmark_count,
                "landmark_observation_records": self.lifecycle_trace_peak.landmark_observation_records,
                "landmark_observation_capacity": self.lifecycle_trace_peak.landmark_observation_capacity,
                "world_entries": self.lifecycle_trace_peak.world_entries
            },
            "eviction": {
                "track_history_track_ids": removed_track_ids,
                "factor_map_track_ids": removed_factor_ids,
                "landmark_db_track_ids": removed_landmark_ids,
                "world_cache_track_ids": removed_world_ids,
                "track_history_entry_net_drop": net_drop(previous.as_ref(), |value| value.track_history_entries, snapshot.track_history_entries),
                "factor_record_net_drop": net_drop(previous.as_ref(), |value| value.factor_records, snapshot.factor_records),
                "landmark_observation_net_drop": net_drop(previous.as_ref(), |value| value.landmark_observation_records, snapshot.landmark_observation_records),
                "record_identity_scope": "track-level exact; row-level net drops are lower bounds when additions and removals share a frame"
            },
            "last_lost_landmarks": self.last_lost_landmarks,
            "num_points_kf": self.num_points_kf.get(&frame_id).copied()
        });
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            if let Ok(line) = serde_json::to_string(&record) {
                let _ = file.write_all(line.as_bytes());
                let _ = file.write_all(b"\n");
            }
        }
        self.lifecycle_trace_previous = Some(snapshot);
    }

    /// Build the on-wire MargData packet from the solved, pre-shift window.
    /// `problem` is the full solved window used for frame tables and KF sets;
    /// `aom_problem` is the boundary-truncated clone used for the matrix.
    fn emit_pre_marg(
        &self,
        aom_problem: &WindowProblem,
        factors: &[WhitenedFactorRowStack],
        plan: &MarginalizationPlan,
        _diagnostics: &WindowDiagnostics,
    ) -> MargData {
        let projection_probe_enabled = diagnostic_env_snapshot().landmark_backsub_probe.is_some();
        if projection_probe_enabled {
            set_active_diagnostic_projection_event(Some(DiagnosticProjectionEvent {
                frame_id: aom_problem
                    .states
                    .last()
                    .map(|state| state.frame_id)
                    .or_else(|| aom_problem.poses.last().map(|pose| pose.frame_id))
                    .unwrap_or_default(),
                timestamp_ns: aom_problem
                    .states
                    .last()
                    .map(|state| state.timestamp_ns)
                    .or_else(|| aom_problem.poses.last().map(|pose| pose.timestamp_ns))
                    .unwrap_or_default(),
            }));
        }
        let projected = projected_rows_native_q2(factors, 1e-10, aom_problem.scalar_mode);
        if projection_probe_enabled {
            set_active_diagnostic_projection_event(None);
        }
        let state_dof = aom_problem.state_dof();
        let (jacobian, rhs_vector) = stack_rows(&projected, state_dof);
        let mut aom_order = Vec::with_capacity(aom_problem.poses.len() + aom_problem.states.len());
        let mut offset = 0usize;
        for pose in &aom_problem.poses {
            aom_order.push(AomBlockData {
                frame_id: pose.frame_id,
                offset,
                dof: POSE_DOF,
                kind: "pose".into(),
            });
            offset += POSE_DOF;
        }
        for state in &aom_problem.states {
            aom_order.push(AomBlockData {
                frame_id: state.frame_id,
                offset,
                dof: NAV_STATE_DOF,
                kind: "state".into(),
            });
            offset += NAV_STATE_DOF;
        }
        let (h, b) = if aom_problem.scalar_mode == ScalarMode::UpstreamF32 {
            // Basalt assigns MargData::abs_H/abs_b from the already reduced
            // Q2 rows, immediately before MargHelper consumes those rows.
            // Do not rebuild the packet from the original factor list: that
            // is an earlier boundary and has a different Eigen reduction
            // tree.  q2_f32_normal_system widens only after its f32 product.
            q2_f32_normal_system(&jacobian, &rhs_vector)
        } else {
            (
                jacobian.transpose() * &jacobian,
                jacobian.transpose() * &rhs_vector,
            )
        };

        let frame_poses = self
            .window_poses
            .iter()
            .map(|pose| {
                let q = pose.pose.rotation.quaternion();
                FramePoseData {
                    frame_id: pose.frame_id,
                    timestamp_ns: pose.timestamp_ns,
                    pose: [
                        pose.pose.translation.x,
                        pose.pose.translation.y,
                        pose.pose.translation.z,
                        q.w,
                        q.i,
                        q.j,
                        q.k,
                    ],
                    is_keyframe: pose.is_keyframe,
                }
            })
            .collect::<Vec<_>>();
        let frame_states = self
            .window_states
            .iter()
            .map(|state| {
                let q = state.nav.imu_to_world.rotation.quaternion();
                FrameStateData {
                    frame_id: state.frame_id,
                    timestamp_ns: state.timestamp_ns,
                    pose: [
                        state.nav.imu_to_world.translation.x,
                        state.nav.imu_to_world.translation.y,
                        state.nav.imu_to_world.translation.z,
                        q.w,
                        q.i,
                        q.j,
                        q.k,
                    ],
                    velocity: state.nav.velocity_world_m_s.into(),
                    gyro_bias: state.nav.gyro_bias_rad_s.into(),
                    accel_bias: state.nav.accel_bias_m_s2.into(),
                    linearized: state.linearized,
                    is_keyframe: state.is_keyframe,
                    is_latest: state.is_latest,
                }
            })
            .collect::<Vec<_>>();
        let frame_poses_fej = self
            .window_poses
            .iter()
            .map(|pose| (pose.timestamp_ns, Self::pose_fej_wire(pose)))
            .collect::<BTreeMap<_, _>>();
        let frame_states_fej = self
            .window_states
            .iter()
            .map(|state| (state.timestamp_ns, Self::state_fej_wire(state)))
            .collect::<BTreeMap<_, _>>();

        let keyframes = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(
                self.window_states
                    .iter()
                    .filter(|state| state.is_keyframe)
                    .map(|state| state.frame_id),
            )
            .collect::<Vec<_>>();
        let mut image_frame_ids = keyframes.clone();
        image_frame_ids.extend(plan.drop_poses.iter().copied());
        image_frame_ids.sort_unstable();
        image_frame_ids.dedup();
        let mut of_images = image_frame_ids
            .iter()
            .filter_map(|frame_id| self.of_images.get(frame_id))
            .flat_map(|images| images.iter().cloned())
            .collect::<Vec<_>>();
        of_images.sort_by_key(|image| (image.frame_id, image.timestamp_ns, image.camera_id));

        let of_observations = self
            .of_observations
            .iter()
            .filter(|(frame_id, _)| image_frame_ids.binary_search(frame_id).is_ok())
            .flat_map(|(_, observations)| observations.iter().cloned())
            .collect::<Vec<_>>();

        let prior = self.prior.as_ref().and_then(|prior| {
            Self::prior_data_for_tables(prior, &self.window_poses, &self.window_states)
        });
        let row_counts = Self::factor_row_counts(aom_problem, factors);
        MargData {
            schema_version: MARGDATA_SCHEMA_VERSION,
            aom_sqrt_jacobian: MatrixData::new(
                jacobian.nrows(),
                jacobian.ncols(),
                jacobian.iter().copied().collect(),
            )
            .expect("pre-marg window Jacobian dimensions"),
            aom_sqrt_rhs: rhs_vector.iter().copied().collect(),
            aom_abs_h: MatrixData::new(h.nrows(), h.ncols(), h.iter().copied().collect()),
            aom_abs_b: Some(b.iter().copied().collect()),
            frame_poses,
            frame_states,
            keyframes: keyframes.clone(),
            kf_to_marg: self.last_kf_to_marg.clone(),
            kfs_all: keyframes,
            kfs_to_marg: self
                .last_kf_to_marg
                .iter()
                .map(|(frame_id, _)| *frame_id)
                .collect(),
            aom_order,
            marginalization: self.last_marg_targets.clone(),
            prior,
            row_counts,
            of_observations,
            of_images,
            frame_poses_fej,
            frame_states_fej,
            fej_complete: true,
            // The pinned VIO path writes `MargData::use_imu = true` for every
            // mapper packet, even when a synthetic packet has no retained
            // link after the active-window shift.  Keep the richer Rust
            // diagnostic value on state-only records, but make the actual
            // queue packet field follow that wire contract exactly.
            used_imu: if self.last_kf_to_marg.is_empty() {
                !self.imu_links.is_empty()
            } else {
                true
            },
            provenance_version: "basalt-0f3b2b52-pre-marg-v4".into(),
        }
    }

    fn emit_marg(&self, diagnostics: &WindowDiagnostics) -> MargData {
        let state_dof = self.active_aom_dof();
        let rows = self
            .last_rows
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum();
        let (jacobian, rhs_vector) = stack_rows(&self.last_rows, state_dof);
        let jacobian_data = jacobian.iter().copied().collect::<Vec<_>>();
        let rhs = rhs_vector.iter().copied().collect::<Vec<_>>();
        let h = jacobian.transpose() * &jacobian;
        let b = jacobian.transpose() * &rhs_vector;
        let frame_poses = self
            .window_poses
            .iter()
            .map(|pose| {
                let q = pose.pose.rotation.quaternion();
                FramePoseData {
                    frame_id: pose.frame_id,
                    timestamp_ns: pose.timestamp_ns,
                    pose: [
                        pose.pose.translation.x,
                        pose.pose.translation.y,
                        pose.pose.translation.z,
                        q.w,
                        q.i,
                        q.j,
                        q.k,
                    ],
                    is_keyframe: pose.is_keyframe,
                }
            })
            .collect::<Vec<_>>();
        let frame_poses_fej = self
            .window_poses
            .iter()
            .map(|pose| (pose.timestamp_ns, Self::pose_fej_wire(pose)))
            .collect::<BTreeMap<_, _>>();
        let frame_states_fej = self
            .window_states
            .iter()
            .map(|state| (state.timestamp_ns, Self::state_fej_wire(state)))
            .collect::<BTreeMap<_, _>>();
        let mut aom_order = Vec::with_capacity(self.window_poses.len() + self.window_states.len());
        let mut offset = 0usize;
        for pose in &self.window_poses {
            aom_order.push(AomBlockData {
                frame_id: pose.frame_id,
                offset,
                dof: POSE_DOF,
                kind: "pose".into(),
            });
            offset += POSE_DOF;
        }
        for state in &self.window_states {
            aom_order.push(AomBlockData {
                frame_id: state.frame_id,
                offset,
                dof: NAV_STATE_DOF,
                kind: "state".into(),
            });
            offset += NAV_STATE_DOF;
        }
        let prior = self.prior.as_ref().and_then(|prior| {
            Self::prior_data_for_tables(prior, &self.window_poses, &self.window_states)
        });
        let keyframes = self
            .window_poses
            .iter()
            .map(|pose| pose.frame_id)
            .chain(
                self.window_states
                    .iter()
                    .filter(|state| state.is_keyframe)
                    .map(|state| state.frame_id),
            )
            .collect::<Vec<_>>();
        let mut image_frame_ids = keyframes.clone();
        // Upstream's MargData packet is assembled before the selected old
        // keyframes are erased.  Include those IDs (and their exact raw
        // images) even though the post-shift frame tables no longer retain
        // them.
        image_frame_ids.extend(self.last_kf_to_marg.iter().map(|(frame_id, _)| *frame_id));
        image_frame_ids.sort_unstable();
        image_frame_ids.dedup();
        let mut of_images = image_frame_ids
            .iter()
            .filter_map(|frame_id| self.of_images.get(frame_id))
            .flat_map(|images| images.iter().cloned())
            .collect::<Vec<_>>();
        of_images.sort_by_key(|image| (image.frame_id, image.timestamp_ns, image.camera_id));
        let of_observations = self
            .of_observations
            .iter()
            .filter(|(frame_id, _)| image_frame_ids.binary_search(frame_id).is_ok())
            .flat_map(|(_, observations)| observations.iter().cloned())
            .collect::<Vec<_>>();
        MargData {
            schema_version: MARGDATA_SCHEMA_VERSION,
            aom_sqrt_jacobian: MatrixData::new(rows, state_dof, jacobian_data)
                .expect("window Jacobian dimensions"),
            aom_sqrt_rhs: rhs,
            aom_abs_h: MatrixData::new(state_dof, state_dof, h.iter().copied().collect()),
            aom_abs_b: Some(b.iter().copied().collect()),
            frame_poses,
            frame_states: self.states.clone(),
            keyframes: keyframes.clone(),
            kf_to_marg: self.last_kf_to_marg.clone(),
            kfs_all: keyframes,
            kfs_to_marg: self
                .last_kf_to_marg
                .iter()
                .map(|(frame_id, _)| *frame_id)
                .collect(),
            aom_order,
            marginalization: self.last_marg_targets.clone(),
            prior,
            row_counts: [
                diagnostics.prior_factor_rows,
                diagnostics.visual_factor_rows,
                diagnostics.imu_factor_rows,
                diagnostics.bias_factor_rows,
            ],
            of_observations,
            of_images,
            frame_poses_fej,
            frame_states_fej,
            fej_complete: true,
            used_imu: !self.imu_links.is_empty(),
            provenance_version: "basalt-0f3b2b52".into(),
        }
    }
}

fn integrate_interval(
    samples: &[ImuSample],
    start_ns: i64,
    end_ns: i64,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    noise: ImuNoiseModel,
    scalar_mode: ScalarMode,
) -> (Option<ImuPreintegratedDelta>, bool) {
    if scalar_mode == ScalarMode::UpstreamF32 {
        return integrate_queue_f32(samples, start_ns, end_ns, bias_gyro, bias_accel, noise);
    }
    match integrate_between(
        samples,
        start_ns,
        end_ns,
        bias_gyro,
        bias_accel,
        Some(noise),
    ) {
        Ok(delta) => (Some(delta), false),
        Err(_) => (
            fallback_integrate(samples, start_ns, end_ns, bias_gyro, bias_accel, noise),
            true,
        ),
    }
}

/// Applies Basalt's static gyro calibration.  The parameter layout is the
/// upstream `[b_x,b_y,b_z,s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9]`, with the
/// scale columns stored contiguously after the three bias entries.
fn calibrate_gyro(params: &[f64], raw: Vector3<f64>) -> Vector3<f64> {
    if params.len() != 12 {
        return raw;
    }
    let scale = Matrix3::from_columns(&[
        Vector3::new(params[3], params[4], params[5]),
        Vector3::new(params[6], params[7], params[8]),
        Vector3::new(params[9], params[10], params[11]),
    ]);
    raw + scale * raw - Vector3::new(params[0], params[1], params[2])
}

/// Applies Basalt's static accelerometer calibration.  Its six scale terms
/// form the lower-triangular matrix described by `CalibAccelBias` in the
/// pinned basalt-headers revision.
fn calibrate_accel(params: &[f64], raw: Vector3<f64>) -> Vector3<f64> {
    if params.len() != 9 {
        return raw;
    }
    let scale = Matrix3::new(
        params[3], 0.0, 0.0, params[4], params[6], 0.0, params[5], params[7], params[8],
    );
    raw + scale * raw - Vector3::new(params[0], params[1], params[2])
}

/// Static calibration in the estimator's upstream `float` scalar.  The
/// public calibration payload is f64, but Basalt casts the calibrated sensor
/// coefficients into `Scalar` before evaluating these matrix products.
fn calibrate_gyro_f32(params: &[f64], raw: Vector3<f32>) -> Vector3<f32> {
    if params.len() != 12 {
        return raw;
    }
    let scale = SMatrix::<f32, 3, 3>::from_columns(&[
        Vector3::new(params[3] as f32, params[4] as f32, params[5] as f32),
        Vector3::new(params[6] as f32, params[7] as f32, params[8] as f32),
        Vector3::new(params[9] as f32, params[10] as f32, params[11] as f32),
    ]);
    raw + scale * raw - Vector3::new(params[0] as f32, params[1] as f32, params[2] as f32)
}

fn calibrate_accel_f32(params: &[f64], raw: Vector3<f32>) -> Vector3<f32> {
    if params.len() != 9 {
        return raw;
    }
    let scale = SMatrix::<f32, 3, 3>::new(
        params[3] as f32,
        0.0,
        0.0,
        params[4] as f32,
        params[6] as f32,
        0.0,
        params[5] as f32,
        params[7] as f32,
        params[8] as f32,
    );
    raw + scale * raw - Vector3::new(params[0] as f32, params[1] as f32, params[2] as f32)
}

/// Reproduces Basalt's first-camera initialization.  The first calibrated
/// IMU packet at or after the camera timestamp defines roll/pitch by rotating
/// its specific-force direction to world +Z; position, velocity, and biases
/// start at zero.  No IMU interval is integrated before this first state.
fn initial_nav_from_imu(
    timestamp_ns: i64,
    samples: &[ImuSample],
    scalar_mode: ScalarMode,
) -> BasaltNavState {
    let sample = samples
        .iter()
        .find(|sample| sample.timestamp_ns >= timestamp_ns)
        .or_else(|| samples.last());
    let rotation = if scalar_mode == ScalarMode::UpstreamF32 {
        sample
            .and_then(|sample| {
                let accel = sample.accel_m_s2.map(|value| value as f32);
                if accel.norm_squared() > 1.0e-18_f32 && accel.iter().all(|value| value.is_finite())
                {
                    // Eigen's Quaternion::FromTwoVectors (used by the
                    // pinned initialize() path) constructs the quaternion
                    // directly from normalized dot/cross terms.  The
                    // nalgebra rotation_between helper takes an
                    // axis-angle route, which is mathematically equivalent
                    // but rounds differently at the f32 boundary.
                    Some(from_two_vectors_eigen_f32(accel, Vector3::z()))
                } else {
                    None
                }
            })
            .map(|q| {
                // `q` already owns the upstream Eigen/Sophus f32
                // normalization.  Widen its stored components without a
                // second f64 normalization: Sophus' cast path preserves the
                // normalized Scalar quaternion at this boundary.
                UnitQuaternion::new_unchecked(Quaternion::new(
                    q.w as f64, q.i as f64, q.j as f64, q.k as f64,
                ))
            })
            .unwrap_or_else(UnitQuaternion::identity)
    } else {
        sample
            .and_then(|sample| {
                let accel = sample.accel_m_s2;
                if accel.norm_squared() > 1.0e-18 && accel.iter().all(|value| value.is_finite()) {
                    UnitQuaternion::rotation_between(&accel.normalize(), &Vector3::z_axis())
                } else {
                    None
                }
            })
            .unwrap_or_else(UnitQuaternion::identity)
    };
    BasaltNavState {
        imu_to_world: SE3::new(rotation, Vector3::zeros()),
        velocity_world_m_s: Vector3::zeros(),
        gyro_bias_rad_s: Vector3::zeros(),
        accel_bias_m_s2: Vector3::zeros(),
    }
}

/// f32 counterpart of Eigen's `Quaternion::FromTwoVectors(a, b)` fast path.
/// The antiparallel SVD branch is not reachable for the IMU gravity
/// initialization (the target is +Z), but a deterministic orthogonal-axis
/// fallback keeps this helper total for synthetic inputs.
fn from_two_vectors_eigen_f32(a: Vector3<f32>, b: Vector3<f32>) -> UnitQuaternion<f32> {
    let v0 = a.normalize();
    let v1 = b.normalize();
    let mut c = v1.dot(&v0);
    if c < -1.0_f32 + 1.0e-5_f32 {
        c = c.max(-1.0_f32);
        let axis = if v0.x.abs() < v0.y.abs() && v0.x.abs() < v0.z.abs() {
            Vector3::new(0.0_f32, -v0.z, v0.y).normalize()
        } else if v0.y.abs() < v0.z.abs() {
            Vector3::new(-v0.z, 0.0_f32, v0.x).normalize()
        } else {
            Vector3::new(-v0.y, v0.x, 0.0_f32).normalize()
        };
        let w2 = (1.0_f32 + c) * 0.5_f32;
        let scale = (1.0_f32 - w2).sqrt();
        return UnitQuaternion::from_quaternion(Quaternion::new(
            w2.sqrt(),
            scale * axis.x,
            scale * axis.y,
            scale * axis.z,
        ));
    }
    let s = ((1.0_f32 + c) * 2.0_f32).sqrt();
    let axis = v0.cross(&v1) / s;
    UnitQuaternion::from_quaternion(Quaternion::new(s * 0.5_f32, axis.x, axis.y, axis.z))
}

fn fallback_integrate(
    samples: &[ImuSample],
    start_ns: i64,
    end_ns: i64,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    noise: ImuNoiseModel,
) -> Option<ImuPreintegratedDelta> {
    if samples.is_empty() || end_ns <= start_ns {
        return None;
    }
    let mut preintegrator = ImuPreintegrator::new(bias_gyro, bias_accel).with_noise(noise)?;
    let selected = samples
        .iter()
        .filter(|sample| sample.timestamp_ns > start_ns && sample.timestamp_ns <= end_ns)
        .collect::<Vec<_>>();
    let Some(_first) = selected.first() else {
        return None;
    };
    let last = selected.last().expect("selected is non-empty");
    let last_gyro = last.gyro_rad_s;
    let last_accel = last.accel_m_s2;
    // Basalt consumes the first packet after the previous camera timestamp
    // with dt = packet.t - start_t.  The half-open reader interval therefore
    // must retain this leading partial interval rather than dropping it.
    let mut previous_timestamp = start_ns;
    for sample in selected {
        let dt = (sample.timestamp_ns - previous_timestamp) as f64 * 1e-9;
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        preintegrator.integrate_sample(sample.gyro_rad_s, sample.accel_m_s2, dt);
        previous_timestamp = sample.timestamp_ns;
    }
    if previous_timestamp < end_ns {
        let dt = (end_ns - previous_timestamp) as f64 * 1e-9;
        if !dt.is_finite() || dt <= 0.0 {
            return None;
        }
        // The final partial packet uses the last packet's measurement.  Using
        // the first packet here duplicates the leading endpoint and creates
        // a visible p/v discontinuity on the next camera interval.
        preintegrator.integrate_sample(last_gyro, last_accel, dt);
    }
    Some(preintegrator.delta().clone())
}

/// Queue-compatible `IntegratedImuMeasurement<float>` shadow.  The public
/// delta remains f64 so existing factor/API contracts do not change, but every
/// state, Jacobian, covariance, and trigonometric intermediate in this path is
/// owned by f32 before it crosses that boundary.
///
/// The boolean result reports whether the interval needed synthetic trailing
/// integration (or had no usable packet).  Exact queue endpoints are the
/// normal path in the Euroc reader and must not be mislabeled as fallback.
fn integrate_queue_f32(
    samples: &[ImuSample],
    start_ns: i64,
    end_ns: i64,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    noise: ImuNoiseModel,
) -> (Option<ImuPreintegratedDelta>, bool) {
    if samples.is_empty() || end_ns <= start_ns {
        return (None, true);
    }
    let selected = samples
        .iter()
        .filter(|sample| sample.timestamp_ns > start_ns && sample.timestamp_ns <= end_ns)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return (None, true);
    }
    let endpoint_complete = selected
        .last()
        .is_some_and(|sample| sample.timestamp_ns == end_ns);
    let last_gyro = selected.last().expect("selected is non-empty").gyro_rad_s;
    let last_accel = selected.last().expect("selected is non-empty").accel_m_s2;
    let mut state = F32ImuState::new(
        bias_gyro.map(|value| value as f32),
        bias_accel.map(|value| value as f32),
        noise,
    );
    let mut update_trace = Vec::new();
    let mut previous_timestamp = start_ns;
    for sample in selected {
        let dt_ns = sample.timestamp_ns - previous_timestamp;
        // This mirrors `Scalar dt = dt_ns * Scalar(1e-9)` in the pinned
        // Basalt header: convert the integer duration to the template scalar
        // before multiplying, rather than computing an f64 duration first.
        let dt = (dt_ns as f32) * 1.0e-9_f32;
        if !dt.is_finite() || dt <= 0.0 {
            return (None, true);
        }
        state.integrate(
            sample.gyro_rad_s.map(|value| value as f32),
            sample.accel_m_s2.map(|value| value as f32),
            dt_ns,
            dt,
        );
        update_trace.push(ImuDStateUpdateTrace {
            t_ns: sample.timestamp_ns - start_ns,
            f: state.last_f.map(f64::from),
            g: state.last_g.map(f64::from),
            f_old_bg: state.last_f_old_bg.map(f64::from),
            d_state_d_bg: state_d_bg_matrix(&state),
            d_state_d_ba: state_d_ba_matrix(&state),
        });
        previous_timestamp = sample.timestamp_ns;
    }
    if previous_timestamp < end_ns {
        let dt = ((end_ns - previous_timestamp) as f32) * 1.0e-9_f32;
        if !dt.is_finite() || dt <= 0.0 {
            return (None, true);
        }
        state.integrate(
            last_gyro.map(|value| value as f32),
            last_accel.map(|value| value as f32),
            end_ns - previous_timestamp,
            dt,
        );
        update_trace.push(ImuDStateUpdateTrace {
            t_ns: end_ns - start_ns,
            f: state.last_f.map(f64::from),
            g: state.last_g.map(f64::from),
            f_old_bg: state.last_f_old_bg.map(f64::from),
            d_state_d_bg: state_d_bg_matrix(&state),
            d_state_d_ba: state_d_ba_matrix(&state),
        });
    }
    (Some(state.into_delta(update_trace)), !endpoint_complete)
}

fn state_d_bg_matrix(state: &F32ImuState) -> SMatrix<f64, 9, 3> {
    let mut out = SMatrix::<f64, 9, 3>::zeros();
    out.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&state.d_position_bg.map(f64::from));
    out.fixed_view_mut::<3, 3>(3, 0)
        .copy_from(&state.d_rotation_bg.map(f64::from));
    out.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&state.d_velocity_bg.map(f64::from));
    out
}

fn state_d_ba_matrix(state: &F32ImuState) -> SMatrix<f64, 9, 3> {
    let mut out = SMatrix::<f64, 9, 3>::zeros();
    out.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&state.d_position_ba.map(f64::from));
    out.fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&state.d_velocity_ba.map(f64::from));
    out
}

#[derive(Clone, Copy)]
struct F32ImuState {
    rotation: UnitQuaternion<f32>,
    velocity: Vector3<f32>,
    position: Vector3<f32>,
    delta_time_ns: i64,
    bias_gyro: Vector3<f32>,
    bias_accel: Vector3<f32>,
    d_rotation_bg: Matrix3<f32>,
    d_velocity_bg: Matrix3<f32>,
    d_velocity_ba: Matrix3<f32>,
    d_position_bg: Matrix3<f32>,
    d_position_ba: Matrix3<f32>,
    covariance: F32Matrix9,
    noise: ImuNoiseModel,
    last_f: SMatrix<f32, 9, 9>,
    last_g: F32Matrix9x3,
    last_f_old_bg: F32Matrix9x3,
}

impl F32ImuState {
    fn new(bias_gyro: Vector3<f32>, bias_accel: Vector3<f32>, noise: ImuNoiseModel) -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            velocity: Vector3::zeros(),
            position: Vector3::zeros(),
            delta_time_ns: 0,
            bias_gyro,
            bias_accel,
            d_rotation_bg: Matrix3::zeros(),
            d_velocity_bg: Matrix3::zeros(),
            d_velocity_ba: Matrix3::zeros(),
            d_position_bg: Matrix3::zeros(),
            d_position_ba: Matrix3::zeros(),
            covariance: F32Matrix9::zeros(),
            noise,
            last_f: SMatrix::zeros(),
            last_g: F32Matrix9x3::zeros(),
            last_f_old_bg: F32Matrix9x3::zeros(),
        }
    }

    fn integrate(&mut self, gyro: Vector3<f32>, accel: Vector3<f32>, dt_ns: i64, dt: f32) {
        let omega = gyro - self.bias_gyro;
        let specific_force = accel - self.bias_accel;
        // Sophus header expression is `exp(omega * dt / 2)`: Eigen first
        // forms the full float product, then divides by two.
        let r_half_q = sophus_quat_product(self.rotation, so3_exp_f32(omega * (0.5_f32 * dt)));
        let r_half = eigen_quaternion_matrix_f32(r_half_q);
        let accel_world = eigen_mat_vec_f32(r_half, specific_force);
        let old_velocity = self.velocity;
        // Eigen's packet evaluator keeps both position multiply/add
        // boundaries fused in the source-order expression
        // `p + v * dt + 0.5 * accel * dt * dt`: first `p_v = fma(v, dt,
        // p)`, then `p_next = fma((0.5 * accel) * dt, dt, p_v)`.  The
        // latter FMA is observable at the final frame-3 -> frame-4 IMU
        // packet (clean `bb7f5a2b` versus a separate-add `bb7f5a2c`).
        // Keep this packet schedule general for every interval; it is not a
        // frame/value-specific correction.
        self.position = integrate_position_f32(self.position, old_velocity, accel_world, dt);
        // Eigen's packet evaluator leaves the source-order vector update
        // grouped by lane.  On the pinned native O3 build this contracts the
        // x lane as accel_world.x * dt + old_velocity.x, while y/z retain the
        // ordinary vector-product association.  Keep this general expression
        // independent of frame/sample values; it is the same kernel for every
        // IMU packet and preserves the frame-1 endpoint.
        self.velocity.x = accel_world.x.mul_add(dt, old_velocity.x);
        self.velocity.y = old_velocity.y + accel_world.y * dt;
        self.velocity.z = old_velocity.z + accel_world.z * dt;
        let dtheta = omega * dt;
        self.rotation = sophus_quat_product(self.rotation, so3_exp_f32(dtheta));
        self.delta_time_ns += dt_ns;

        let mut f = SMatrix::<f32, 9, 9>::identity();
        f.fixed_view_mut::<3, 3>(0, 6)
            .copy_from(&(SMatrix::<f32, 3, 3>::identity() * dt));
        let f_rot = skew_f32(-accel_world * dt);
        f.fixed_view_mut::<3, 3>(6, 3).copy_from(&f_rot);
        f.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&((f_rot * dt) * 0.5_f32));

        let mut a = F32Matrix9x3::zeros();
        a.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(((r_half * 0.5_f32) * dt) * dt));
        a.fixed_view_mut::<3, 3>(6, 0).copy_from(&(r_half * dt));

        let mut g = F32Matrix9x3::zeros();
        let jr = right_jacobian_so3_f32(dtheta);
        let jr2 = right_jacobian_so3_f32(omega * (0.5_f32 * dt));
        let r_new = eigen_quaternion_matrix_f32(self.rotation);
        let g_upper_product = eigen_matrix_product_f32(r_new, jr);
        let g_upper_dt = g_upper_product * dt;
        g.fixed_view_mut::<3, 3>(3, 0).copy_from(&g_upper_dt);
        let g_lower_frot_rhalf = eigen_matrix_product_f32(f_rot, r_half);
        let g_lower_jr2 = eigen_matrix_product_f32(g_lower_frot_rhalf, jr2);
        let g_lower_half = g_lower_jr2 * 0.5_f32;
        let g_lower_dt = g_lower_half * dt;
        g.fixed_view_mut::<3, 3>(6, 0).copy_from(&g_lower_dt);
        let g_position = (g_lower_dt * dt) * 0.5_f32;
        g.fixed_view_mut::<3, 3>(0, 0).copy_from(&g_position);

        if std::env::var_os("VISLOC_BASALT_M7HD_STEP_CURRENT").is_some() {
            eprintln!(
                "RUST_M7HD_STEP_CURRENT dt_ns={} accel={} gyro={} f={} a={} g={}",
                dt_ns,
                v3_bits(accel),
                v3_bits(gyro),
                m9_bits(f),
                m9x3_bits(a),
                m9x3_bits(g)
            );
        }

        let mut old_d_ba = F32Matrix9x3::zeros();
        old_d_ba
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.d_position_ba);
        old_d_ba
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&self.d_velocity_ba);
        let mut old_d_bg = F32Matrix9x3::zeros();
        old_d_bg
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&self.d_position_bg);
        old_d_bg
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&self.d_rotation_bg);
        old_d_bg
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&self.d_velocity_bg);
        let new_d_ba = eigen_matrix_product_9x3_add_f32(f, old_d_ba, a);
        let new_d_bg = eigen_matrix_product_9x3_add_f32(f, old_d_bg, g);
        let f_old_bg = eigen_matrix_product_9x3_f32(f, old_d_bg);
        self.last_f = f;
        self.last_g = g;
        self.last_f_old_bg = f_old_bg;
        self.d_position_ba = new_d_ba.fixed_rows::<3>(0).into_owned();
        self.d_velocity_ba = new_d_ba.fixed_rows::<3>(6).into_owned();
        self.d_position_bg = new_d_bg.fixed_rows::<3>(0).into_owned();
        self.d_rotation_bg = new_d_bg.fixed_rows::<3>(3).into_owned();
        self.d_velocity_bg = new_d_bg.fixed_rows::<3>(6).into_owned();

        // The covariance recursion is the same F/A/G propagation as the
        // pinned IntegratedImuMeasurement::integrate.  The former Rust path
        // used a left-only `r_half * hat(accel)` surrogate here; it is not
        // the header's covariance contract once the body rotates.
        let accel_density = self.noise.accel_density as f32;
        let gyro_density = self.noise.gyro_density as f32;
        let cov_f = f;
        let accel_noise = a;
        let gyro_noise = g;
        // Eigen's fixed-size covariance assignment is evaluated as three
        // in-place GEMMs into one destination: first `F * cov_ * F^T`, then
        // the accelerometer and gyroscope weighted Grams.  The latter GEMMs
        // start with the already accumulated destination, so materializing
        // each term and adding it afterward changes a few FMA boundaries.
        let covariance_before = self.covariance;
        let mut covariance = eigen_covariance_fc_product_f32(cov_f, self.covariance);
        covariance = eigen_matrix_product_9_f32(covariance, cov_f.transpose().into_owned());
        eigen_weighted_gram_accumulate_f32(
            &mut covariance,
            accel_noise,
            accel_density * accel_density,
        );
        eigen_weighted_gram_accumulate_f32(
            &mut covariance,
            gyro_noise,
            gyro_density * gyro_density,
        );
        self.covariance = covariance;
        if std::env::var_os("VISLOC_BASALT_M7HD_NOISE").is_some() {
            let term2_custom =
                eigen_weighted_gram_term_f32(accel_noise, accel_density * accel_density);
            let term3_custom =
                eigen_weighted_gram_term_f32(gyro_noise, gyro_density * gyro_density);
            let term1_custom = eigen_matrix_product_9_f32(
                eigen_covariance_fc_product_f32(cov_f, covariance_before),
                cov_f.transpose().into_owned(),
            );
            eprintln!(
                "RUST_M7HD_NOISE dt_ns={} term1={} term2={} term3={} cov={}",
                dt_ns,
                m9_bits(term1_custom),
                m9_bits(term2_custom),
                m9_bits(term3_custom),
                m9_bits(self.covariance)
            );
        }
    }

    fn into_delta(self, update_trace: Vec<ImuDStateUpdateTrace>) -> ImuPreintegratedDelta {
        let q = self.rotation;
        ImuPreintegratedDelta {
            delta_rotation: UnitQuaternion::new_unchecked(Quaternion::new(
                q.w as f64, q.i as f64, q.j as f64, q.k as f64,
            )),
            delta_velocity: self.velocity.map(|value| value as f64),
            delta_position: self.position.map(|value| value as f64),
            // Basalt stores the elapsed endpoint as an integer nanosecond
            // field and converts it once at each consumer.  Do the same
            // instead of summing rounded f32 per-sample durations.
            delta_time: ((self.delta_time_ns as f32) * 1.0e-9_f32) as f64,
            bias_gyro: self.bias_gyro.map(|value| value as f64),
            bias_accel: self.bias_accel.map(|value| value as f64),
            jacobian_rotation_gyro_bias: self.d_rotation_bg.map(|value| value as f64),
            jacobian_velocity_gyro_bias: self.d_velocity_bg.map(|value| value as f64),
            jacobian_velocity_accel_bias: self.d_velocity_ba.map(|value| value as f64),
            jacobian_position_gyro_bias: self.d_position_bg.map(|value| value as f64),
            jacobian_position_accel_bias: self.d_position_ba.map(|value| value as f64),
            covariance: self.covariance.map(|value| value as f64),
            update_trace,
        }
    }
}

fn skew_f32(v: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn eigen_matrix_product_f32(left: Matrix3<f32>, right: Matrix3<f32>) -> Matrix3<f32> {
    // The pinned Eigen 3x3 packet evaluator seeds each dot product with the
    // k=1 product, then contracts k=2 and k=0 (`vmulps; vfmadd132ps;
    // vfmadd132ps`).  This order is observable in the f32 G Jacobian.
    let mut result = Matrix3::zeros();
    for col in 0..3 {
        for row in 0..3 {
            let mut value = left[(row, 1)] * right[(1, col)];
            value = left[(row, 2)].mul_add(right[(2, col)], value);
            value = left[(row, 0)].mul_add(right[(0, col)], value);
            result[(row, col)] = value;
        }
    }
    result
}

/// Fixed 9x9 product used by Eigen's covariance GEMMs.
///
/// The pinned AVX2 Eigen `gebp` kernel keeps separate even/odd accumulators
/// for the 8-row packet and reduces them before the k=8 tail. Its scalar
/// ninth-row/4-column tail is different again: four packet accumulators each
/// contain one k-pair, and the packet reduction is
/// `((p0+p2)+(p4+p6)) + ((p1+p3)+(p5+p7))`. These association boundaries are
/// observable at f32 and must remain explicit in the Rust port.
fn eigen_covariance_fc_product_f32(left: F32Matrix9, right: F32Matrix9) -> F32Matrix9 {
    let mut result = eigen_matrix_product_9_f32(left, right);
    // Pinned normal/normal gebp 0x490870..0x4908f7: four independent
    // depth-pair packets. 0x490c00 merges halves before the ninth FMA.
    // This is NOT the schedule of the following F*C*F^T assignment.
    for col in 0..8 {
        let p: [f32; 8] = std::array::from_fn(|k| left[(8, k)].mul_add(right[(k, col)], 0.0_f32));
        let even = (p[0] + p[2]) + (p[4] + p[6]);
        let odd = (p[1] + p[3]) + (p[5] + p[7]);
        result[(8, col)] = left[(8, 8)].mul_add(right[(8, col)], odd + even);
    }
    result
}

fn eigen_matrix_product_9_f32(left: F32Matrix9, right: F32Matrix9) -> F32Matrix9 {
    let mut result = F32Matrix9::zeros();
    for col in 0..9 {
        for row in 0..8 {
            let value = if col < 8 {
                let mut even = left[(row, 0)].mul_add(right[(0, col)], 0.0_f32);
                even = left[(row, 2)].mul_add(right[(2, col)], even);
                even = left[(row, 4)].mul_add(right[(4, col)], even);
                even = left[(row, 6)].mul_add(right[(6, col)], even);
                let mut odd = left[(row, 1)].mul_add(right[(1, col)], 0.0_f32);
                odd = left[(row, 3)].mul_add(right[(3, col)], odd);
                odd = left[(row, 5)].mul_add(right[(5, col)], odd);
                odd = left[(row, 7)].mul_add(right[(7, col)], odd);
                left[(row, 8)].mul_add(right[(8, col)], even + odd)
            } else {
                let mut value = left[(row, 0)].mul_add(right[(0, col)], 0.0_f32);
                for k in 1..9 {
                    value = left[(row, k)].mul_add(right[(k, col)], value);
                }
                value
            };
            result[(row, col)] = value;
        }
        // The aliasing assignment path dispatches Eigen's general GEMM
        // kernel.  Its scalar ninth-row tail walks k in source order for
        // every output column; the packet body above retains the even/odd
        // accumulators used by the 8-row kernel.
        let mut value = left[(8, 0)].mul_add(right[(0, col)], 0.0_f32);
        for k in 1..9 {
            value = left[(8, k)].mul_add(right[(k, col)], value);
        }
        result[(8, col)] = value;
    }
    result
}

#[inline(never)]
fn eigen_matrix_product_9x3_f32(left: F32Matrix9, right: F32Matrix9x3) -> F32Matrix9x3 {
    // Keep this boundary explicit for pinned Eigen-style product scheduling.
    let mut result = F32Matrix9x3::zeros();
    for col in 0..3 {
        for row in 0..9 {
            let mut value = 0.0_f32;
            for k in 0..9 {
                value = left[(row, k)].mul_add(right[(k, col)], value);
            }
            result[(row, col)] = value;
        }
    }
    result
}

#[inline(never)]
fn eigen_matrix_product_9x3_add_f32(
    left: F32Matrix9,
    right: F32Matrix9x3,
    subtrahend: F32Matrix9x3,
) -> F32Matrix9x3 {
    let product = eigen_matrix_product_9x3_f32(left, right);
    let mut result = F32Matrix9x3::zeros();
    for col in 0..3 {
        for row in 0..9 {
            let value = product[(row, col)];
            let seed = -subtrahend[(row, col)];
            result[(row, col)] = if row == 8 {
                seed + value
            } else {
                value.mul_add(1.0_f32, seed)
            };
        }
    }
    result
}

/// Accumulate `noise * variance * noise.transpose()` into an existing 9x9
/// covariance matrix with the fixed-size Eigen 9x3 packet/tail schedule.
///
/// Rows 0..7 are the packet body and use the ordinary k=0,1,2 FMA sequence.
/// The ninth row is Eigen's AVX/FMA packet tail.  For the two four-column
/// packets it first computes the k=0 and k=1 products in an 8-lane packet,
/// reduces the low/high halves with an add, and then contracts k=2.  That is
/// observably `(p0 + p1).add(p2)` rather than a scalar k-order permutation.
/// The final one-column packet remains the ordinary k=0,1,2 chain.
///
/// Build the complete term before adding it to `destination`: Eigen's GEMM
/// accumulates this term into the existing destination only after its packet
/// reduction.  Keeping that boundary explicit is required for the final
/// covariance bits.
fn eigen_weighted_gram_accumulate_f32(
    destination: &mut F32Matrix9,
    noise: F32Matrix9x3,
    variance: f32,
) {
    // The observer term above intentionally preserves the pinned trace
    // materialization schedule.  The production covariance expression is a
    // general 9x3 * 3x9 GEMM, whose assignment kernel has a different tail:
    // the four-column packet body is the ordinary k=0,1,2 FMA chain, while
    // the final scalar column reduces three already-materialized products.
    // Keep that assignment schedule separate from the diagnostic term; the
    // distinction is visible in the covariance tail at f32 precision.
    let term = eigen_weighted_gram_assignment_term_f32(noise, variance);
    if std::env::var_os("VISLOC_BASALT_M7HD_ASSIGN_DUMP").is_some() {
        let call = M7HD_ASSIGN_DUMP_CALLS.fetch_add(1, Ordering::Relaxed);
        let limit = std::env::var("VISLOC_BASALT_M7HD_ASSIGN_LIMIT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(2);
        if call < limit {
            let kind = if call.is_multiple_of(2) {
                "accel"
            } else {
                "gyro"
            };
            eprintln!(
                "RUST_M7HD_ASSIGN call={} kind={} variance={} noise={} term={}",
                call,
                kind,
                variance.to_bits(),
                m9x3_bits(noise),
                m9_bits(term)
            );
        }
    }
    for col in 0..9 {
        for row in 0..9 {
            if col < 8 {
                // Eigen's packet `acc` uses a fused alpha*term + destination
                // update when the 4-column GEMM writes into an existing
                // matrix. The row-tail packet follows the same boundary.
                destination[(row, col)] =
                    term[(row, col)].mul_add(1.0_f32, destination[(row, col)]);
            } else {
                // The final scalar column takes Eigen's scalar `+= alpha*C`
                // path rather than the packet FMA accumulator.
                destination[(row, col)] += term[(row, col)];
            }
        }
    }
}

fn eigen_weighted_gram_assignment_term_f32(noise: F32Matrix9x3, variance: f32) -> F32Matrix9 {
    let mut scaled = F32Matrix9x3::zeros();
    for col in 0..3 {
        for row in 0..9 {
            scaled[(row, col)] = noise[(row, col)] * variance;
        }
    }
    let mut term = F32Matrix9::zeros();
    for col in 0..9 {
        for row in 0..9 {
            if col == 8 {
                let p0 = scaled[(row, 0)].mul_add(noise[(col, 0)], 0.0_f32);
                let p1 = scaled[(row, 1)].mul_add(noise[(col, 1)], 0.0_f32);
                // Eigen's one-column packet tail keeps the first two
                // products as independent accumulators, then contracts the
                // third product into their sum.  This is `(p0 + p1) +FMA
                // p2`, rather than adding a separately materialized p2.
                term[(row, col)] = scaled[(row, 2)].mul_add(noise[(col, 2)], p0 + p1);
            } else {
                let mut value = scaled[(row, 0)].mul_add(noise[(col, 0)], 0.0_f32);
                value = scaled[(row, 1)].mul_add(noise[(col, 1)], value);
                value = scaled[(row, 2)].mul_add(noise[(col, 2)], value);
                term[(row, col)] = value;
            }
        }
    }
    term
}

fn eigen_weighted_gram_term_f32(noise: F32Matrix9x3, variance: f32) -> F32Matrix9 {
    let mut scaled = F32Matrix9x3::zeros();
    for col in 0..3 {
        for row in 0..9 {
            scaled[(row, col)] = noise[(row, col)] * variance;
        }
    }
    let mut term = F32Matrix9::zeros();
    for col in 0..9 {
        for row in 0..9 {
            let mut value = scaled[(row, 0)].mul_add(noise[(col, 0)], 0.0);
            if row == 8 && col < 8 {
                // Eigen's 9x3 tail uses Packet8f for the first two depth
                // values, then predux_half_dowto4 (low + high) before the
                // remaining half-packet FMA.
                let p1 = scaled[(row, 1)].mul_add(noise[(col, 1)], 0.0);
                value += p1;
                value = scaled[(row, 2)].mul_add(noise[(col, 2)], value);
            } else {
                for k in 1..3 {
                    value = scaled[(row, k)].mul_add(noise[(col, k)], value);
                }
            }
            term[(row, col)] = value;
        }
    }
    if std::env::var_os("VISLOC_BASALT_M7HD_GRAM_DEBUG").is_some()
        && noise[(0, 0)].to_bits() == 0x3751b5b4
        && variance.to_bits() == 0x3d51b718
    {
        eprintln!(
            "GRAM_DEBUG scaled00={} scaled10={} scaled01={} scaled11={} scaled02={} scaled12={} t00={} t10={} t20={} t80c7={}",
            scaled[(0, 0)].to_bits(),
            scaled[(1, 0)].to_bits(),
            scaled[(0, 1)].to_bits(),
            scaled[(1, 1)].to_bits(),
            scaled[(0, 2)].to_bits(),
            scaled[(1, 2)].to_bits(),
            term[(0, 0)].to_bits(),
            term[(1, 0)].to_bits(),
            term[(2, 0)].to_bits(),
            term[(8, 7)].to_bits()
        );
    }
    term
}

fn m9_bits(value: F32Matrix9) -> String {
    value
        .iter()
        .map(|entry| format!("{:08x}", entry.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn m9x3_bits(value: F32Matrix9x3) -> String {
    value
        .iter()
        .map(|entry| format!("{:08x}", entry.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn v3_bits(value: Vector3<f32>) -> String {
    value
        .iter()
        .map(|entry| format!("{:08x}", entry.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

// Eigen::QuaternionBase::toRotationMatrix operation order.
fn eigen_quaternion_matrix_f32(q: UnitQuaternion<f32>) -> Matrix3<f32> {
    let q = q.quaternion();
    let tx = 2.0_f32 * q.i;
    let ty = 2.0_f32 * q.j;
    let tz = 2.0_f32 * q.k;
    let txx = tx * q.i;
    let txy = ty * q.i;
    let txz = tz * q.i;
    let tyy = ty * q.j;
    let tyz = tz * q.j;
    let tzz = tz * q.k;
    // GCC's pinned Eigen build contracts each off-diagonal source expression
    // into the same scalar FMA schedule as QuaternionBase::toRotationMatrix:
    // e.g. `txz - twy` is `(-w * ty) + txz`, with txz already rounded.
    // Keep the diagonal add/subtract boundaries separate.
    Matrix3::new(
        1.0 - (tyy + tzz),
        tz.mul_add(-q.w, txy),
        ty.mul_add(q.w, txz),
        q.w.mul_add(tz, txy),
        1.0 - (txx + tzz),
        q.w.mul_add(-tx, tyz),
        ty.mul_add(-q.w, txz),
        q.w.mul_add(tx, tyz),
        1.0 - (txx + tyy),
    )
}

fn eigen_mat_vec_f32(m: Matrix3<f32>, v: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(
        m[(0, 0)].mul_add(v.x, m[(0, 2)].mul_add(v.z, m[(0, 1)] * v.y)),
        m[(1, 0)].mul_add(v.x, m[(1, 2)].mul_add(v.z, m[(1, 1)] * v.y)),
        m[(2, 0)].mul_add(v.x, m[(2, 2)].mul_add(v.z, m[(2, 1)] * v.y)),
    )
}

fn sophus_quat_product(a: UnitQuaternion<f32>, b: UnitQuaternion<f32>) -> UnitQuaternion<f32> {
    // The pinned native Eigen/Sophus build contracts this QuaternionProduct on
    // AVX/FMA targets.  Reuse the audited packet lane order and scalar fallback
    // so the IMU state crosses the same f32 arithmetic boundary.
    super::landmarks::sophus_so3_product(a, b)
}

fn sophus_rotate_f32(rotation: UnitQuaternion<f32>, point: Vector3<f32>) -> Vector3<f32> {
    let q = rotation.quaternion();
    // Sophus `SO3Base::operator*(Point)` is `p + q.w() * uv +
    // q.vec().cross(uv)` after `uv = q.vec().cross(p); uv += uv`.  The pinned
    // native Eigen O2/O3 path contracts each cross lane as a fused
    // multiply-subtract and each q.w/point lane as a fused multiply-add;
    // spell those boundaries explicitly so nalgebra cannot choose a
    // different scalar association.
    let uv_x = q.j.mul_add(point.z, -(q.k * point.y));
    let uv_y = q.k.mul_add(point.x, -(q.i * point.z));
    let uv_z = q.i.mul_add(point.y, -(q.j * point.x));
    let uv2_x = uv_x + uv_x;
    let uv2_y = uv_y + uv_y;
    let uv2_z = uv_z + uv_z;

    let cross_x = q.j.mul_add(uv2_z, -(q.k * uv2_y));
    let cross_y = q.k.mul_add(uv2_x, -(q.i * uv2_z));
    let cross_z = q.i.mul_add(uv2_y, -(q.j * uv2_x));

    Vector3::new(
        q.w.mul_add(uv2_x, point.x) + cross_x,
        q.w.mul_add(uv2_y, point.y) + cross_y,
        q.w.mul_add(uv2_z, point.z) + cross_z,
    )
}

/// Eigen's packet evaluator for the pinned `predictState` velocity expression
/// contracts each gravity multiply/add lane before adding the rotated delta:
/// `state0.vel_w_i + g * dt + R0 * delta_v`.  Keeping that boundary explicit
/// is observable on the MH_01 z lane, where a separate `g * dt` temporary is
/// four ULP away from the native O2/O3 result.
fn predict_velocity_f32(
    velocity0: Vector3<f32>,
    gravity: Vector3<f32>,
    dt: f32,
    rotated_delta_velocity: Vector3<f32>,
) -> Vector3<f32> {
    Vector3::new(
        gravity.x.mul_add(dt, velocity0.x) + rotated_delta_velocity.x,
        gravity.y.mul_add(dt, velocity0.y) + rotated_delta_velocity.y,
        gravity.z.mul_add(dt, velocity0.z) + rotated_delta_velocity.z,
    )
}

/// Eigen packet schedule for the position update in Basalt's
/// `IntegratedImuMeasurement::propagateState`.
///
/// The source is written as `p + v * dt + 0.5 * accel * dt * dt`.  The native
/// evaluator contracts both multiply/add boundaries:
///
/// ```text
/// p_v = fma(v, dt, p)
/// p_next = fma((0.5 * accel) * dt, dt, p_v)
/// ```
///
/// Keeping the intermediate products in f32 and spelling both FMAs out is
/// observable on the final frame-3 -> frame-4 preintegration packet: a
/// separate acceleration add yields `bb7f5a2c`, while the clean packet is
/// `bb7f5a2b`.  This helper is used for every IMU packet.
fn integrate_position_f32(
    position0: Vector3<f32>,
    velocity0: Vector3<f32>,
    accel_world: Vector3<f32>,
    dt: f32,
) -> Vector3<f32> {
    let accel_half_dt = (accel_world * 0.5_f32) * dt;
    let position_after_velocity = Vector3::new(
        velocity0.x.mul_add(dt, position0.x),
        velocity0.y.mul_add(dt, position0.y),
        velocity0.z.mul_add(dt, position0.z),
    );
    Vector3::new(
        accel_half_dt.x.mul_add(dt, position_after_velocity.x),
        accel_half_dt.y.mul_add(dt, position_after_velocity.y),
        accel_half_dt.z.mul_add(dt, position_after_velocity.z),
    )
}

/// Sophus `SO3<float>::exp`, including its float small-angle branch.  The
/// nalgebra constructor is mathematically equivalent but its normalization
/// and half-angle path are not the operation sequence used by the pinned
/// Basalt header.
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
    UnitQuaternion::new_unchecked(Quaternion::new(
        real_factor,
        imag_factor * omega.x,
        imag_factor * omega.y,
        imag_factor * omega.z,
    ))
}

/// Sophus `SO3<float>::exp`'s quaternion construction.  Keeping the
/// half-angle factors and small-angle polynomial in the f32 owner avoids
/// routing the upstream float state through nalgebra's generic
/// `from_scaled_axis` implementation.
fn right_jacobian_so3_f32(phi: Vector3<f32>) -> Matrix3<f32> {
    let phi_norm2 = phi.norm_squared();
    let phi_hat = skew_f32(phi);
    let phi_hat2 = phi_hat * phi_hat;
    let mut result = Matrix3::identity();
    // Sophus::Constants<float>::epsilon() is 1e-5f (not machine epsilon).
    // Keep the same Taylor/analytic branch boundary as the pinned header.
    if phi_norm2 > 1.0e-5_f32 {
        let phi_norm = phi_norm2.sqrt();
        let phi_norm3 = phi_norm2 * phi_norm;
        // Preserve Eigen's source grouping: the header spells these as
        // `(phi_hat * numerator) / denominator`, rather than multiplying by
        // a pre-divided scalar.  The intermediate f32 rounding is observable
        // in the gyro-bias Jacobian and covariance recursion.
        result -= (phi_hat * (1.0_f32 - phi_norm.cos())) / phi_norm2;
        result += (phi_hat2 * (phi_norm - phi_norm.sin())) / phi_norm3;
    } else {
        result -= phi_hat * 0.5_f32;
        result += phi_hat2 / 6.0_f32;
    }
    result
}

fn predict_nav(
    previous: &BasaltNavState,
    delta: Option<&ImuPreintegratedDelta>,
    gravity_world: Vector3<f64>,
    scalar_mode: ScalarMode,
) -> BasaltNavState {
    let Some(delta) = delta else {
        return previous.clone();
    };
    if scalar_mode == ScalarMode::UpstreamF32 {
        // The public state widens the upstream f32 quaternion components to
        // f64.  Narrowing those components for the next prediction must not
        // normalize a second time; the source state is already a Sophus SO3
        // unit quaternion at this boundary.
        let q0 = UnitQuaternion::new_unchecked(Quaternion::new(
            previous.imu_to_world.rotation.w as f32,
            previous.imu_to_world.rotation.i as f32,
            previous.imu_to_world.rotation.j as f32,
            previous.imu_to_world.rotation.k as f32,
        ));
        let t0 = previous.imu_to_world.translation.map(|value| value as f32);
        let v0 = previous.velocity_world_m_s.map(|value| value as f32);
        let g = gravity_world.map(|value| value as f32);
        let dt = delta.delta_time as f32;
        let dp = delta.delta_position.map(|value| value as f32);
        let dv = delta.delta_velocity.map(|value| value as f32);
        let dq = UnitQuaternion::new_unchecked(Quaternion::new(
            delta.delta_rotation.w as f32,
            delta.delta_rotation.i as f32,
            delta.delta_rotation.j as f32,
            delta.delta_rotation.k as f32,
        ));
        // Keep the operation order of basalt's pinned
        // `IntegratedImuMeasurement::predictState` literal.  In particular,
        // the gravity terms precede the rotated preintegrated increments;
        // changing only this association is enough to move the f32 result by
        // one or more ulps after each camera interval.
        // The pinned `predictState` instantiation routes both preintegrated
        // vector actions through the out-of-line Packet4f SO3 operator.  Its
        // duplicated cross-product lanes are observable beyond frame 38
        // (frame 39 x differs by one ulp from the scalar-lane helper), so use
        // the same schedule already required by delta velocity.
        let rotated_delta_position = sophus_rotate_step_packet_f32(q0, dp);
        // Eigen's packetized prediction materializes the half-gravity-times-
        // dt operand, then performs the second position update as an FMA:
        // `hdt = (0.5 * g) * dt; p_g = fma(hdt, dt, p_v)`.  Materializing
        // `0.5*g*dt*dt` first changes the final f32 bit on nonzero gravity
        // lanes, even though the real-valued expression is identical.
        let gravity_half_dt = Vector3::new(
            (0.5_f32 * g.x) * dt,
            (0.5_f32 * g.y) * dt,
            (0.5_f32 * g.z) * dt,
        );
        let position_after_velocity = Vector3::new(
            v0.x.mul_add(dt, t0.x),
            v0.y.mul_add(dt, t0.y),
            v0.z.mul_add(dt, t0.z),
        );
        let position_after_gravity = Vector3::new(
            gravity_half_dt.x.mul_add(dt, position_after_velocity.x),
            gravity_half_dt.y.mul_add(dt, position_after_velocity.y),
            gravity_half_dt.z.mul_add(dt, position_after_velocity.z),
        );
        let translation = Vector3::new(
            position_after_gravity.x + rotated_delta_position.x,
            position_after_gravity.y + rotated_delta_position.y,
            position_after_gravity.z + rotated_delta_position.z,
        );
        let velocity = predict_velocity_f32(v0, g, dt, sophus_rotate_step_packet_f32(q0, dv));
        let rotation = sophus_quat_product(q0, dq);
        return BasaltNavState {
            imu_to_world: SE3::new(
                // `rotation` is the already-normalized upstream f32 Sophus
                // product. Widen its components without a second f64
                // normalization, matching the initial state conversion.
                UnitQuaternion::new_unchecked(Quaternion::new(
                    rotation.w as f64,
                    rotation.i as f64,
                    rotation.j as f64,
                    rotation.k as f64,
                )),
                translation.map(|value| value as f64),
            ),
            velocity_world_m_s: velocity.map(|value| value as f64),
            gyro_bias_rad_s: previous.gyro_bias_rad_s.map(|value| (value as f32) as f64),
            accel_bias_m_s2: previous.accel_bias_m_s2.map(|value| (value as f32) as f64),
        };
    }
    let mut predicted = previous.clone();
    let dt = delta.delta_time.max(1e-9);
    predicted.imu_to_world.translation += previous
        .imu_to_world
        .rotation
        .transform_vector(&delta.delta_position)
        + previous.velocity_world_m_s * dt
        + gravity_world * (0.5 * dt * dt);
    predicted.velocity_world_m_s += previous
        .imu_to_world
        .rotation
        .transform_vector(&delta.delta_velocity)
        + gravity_world * dt;
    predicted.imu_to_world.rotation *= delta.delta_rotation;
    predicted
}

#[cfg(test)]
mod tests {
    use super::super::landmarks::triangulate_two_rays;
    use super::*;
    use nalgebra::UnitQuaternion;

    #[test]
    #[ignore = "frame39 native predictState position schedule probe"]
    fn m11_frame39_prediction_position_schedule_probe() {
        let q0 = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f15d17b),
            f32::from_bits(0xbd580cdc),
            f32::from_bits(0xbf4ef897),
            f32::from_bits(0xbd06f2ef),
        ));
        let t0 = Vector3::new(
            f32::from_bits(0xbc41eb34),
            f32::from_bits(0xbbca455c),
            f32::from_bits(0x3d0436e5),
        );
        let v0 = Vector3::new(
            f32::from_bits(0x3caa02c4),
            f32::from_bits(0x3cc220ae),
            f32::from_bits(0xbf240a13),
        );
        let dp = Vector3::new(
            f32::from_bits(0x3c38dfa7),
            f32::from_bits(0x380ec29b),
            f32::from_bits(0xbb7b83eb),
        );
        let dt = f32::from_bits(0x3d4cccaa);
        let gravity = Vector3::new(0.0_f32, 0.0_f32, -9.81_f32);
        let expected = [0xbc2ec678, 0xbb9f8673, 0xb8dc5000];
        let fmt = |value: Vector3<f32>| {
            format!(
                "{:08x} {:08x} {:08x}",
                value.x.to_bits(),
                value.y.to_bits(),
                value.z.to_bits()
            )
        };
        let exact = |value: Vector3<f32>| {
            [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
                .into_iter()
                .zip(expected)
                .filter(|(left, right)| left == right)
                .count()
        };
        for (velocity_name, position_after_velocity) in [
            (
                "fma",
                Vector3::new(
                    v0.x.mul_add(dt, t0.x),
                    v0.y.mul_add(dt, t0.y),
                    v0.z.mul_add(dt, t0.z),
                ),
            ),
            ("separate", t0 + v0 * dt),
        ] {
            let gravity_half_dt = (gravity * 0.5_f32) * dt;
            let position_after_gravity = Vector3::new(
                gravity_half_dt.x.mul_add(dt, position_after_velocity.x),
                gravity_half_dt.y.mul_add(dt, position_after_velocity.y),
                gravity_half_dt.z.mul_add(dt, position_after_velocity.z),
            );
            for (rotation_name, rotated) in [
                ("generic", sophus_rotate_f32(q0, dp)),
                ("step_packet", sophus_rotate_step_packet_f32(q0, dp)),
            ] {
                let result = position_after_gravity + rotated;
                println!(
                    "m11_frame39_prediction velocity={velocity_name} rotation={rotation_name} rotated={} result={} exact={}/3",
                    fmt(rotated), fmt(result), exact(result)
                );
            }
        }
    }

    #[test]
    #[ignore = "requires M11_COV_CAPTURE_ROOT pinned external native capture"]
    fn frame10_covariance_fc_packet_native_stages() {
        let root = std::path::PathBuf::from(std::env::var("M11_COV_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let stages = read("last_cov_stages.jsonl");
        let samples = read("integration_covariance_steps.jsonl");
        assert_eq!(stages.len(), 5);
        assert_eq!(samples.len(), 10);
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        let check = |actual: F32Matrix9, expected: F32Matrix9, stage: &str| {
            for i in 0..81 {
                assert_eq!(
                    actual.as_slice()[i].to_bits(),
                    expected.as_slice()[i].to_bits(),
                    "{stage} lane {i}"
                );
            }
        };
        let last = &samples[9];
        assert_eq!(last["sample_t_ns"].as_i64().unwrap(), 1403636580263555584);
        let f = F32Matrix9::from_column_slice(&parse(&stages[0]["F"]));
        let old = F32Matrix9::from_column_slice(&parse(&last["cov_before_bits"]));
        let fc = eigen_covariance_fc_product_f32(f, old);
        check(
            fc,
            F32Matrix9::from_column_slice(&parse(&stages[1]["tmp_FC"])),
            "FC",
        );
        assert_ne!(
            eigen_matrix_product_9_f32(f, old)[(8, 0)].to_bits(),
            fc[(8, 0)].to_bits()
        );
        let mut cov = eigen_matrix_product_9_f32(fc, f.transpose().into_owned());
        let expected = |i: usize| F32Matrix9::from_row_slice(&parse(&stages[i]["cov_accumulator"]));
        check(cov, expected(2), "FCFt");
        let a = F32Matrix9x3::from_column_slice(&parse(&stages[0]["A"]));
        let g = F32Matrix9x3::from_column_slice(&parse(&stages[0]["G"]));
        eigen_weighted_gram_accumulate_f32(&mut cov, a, parse(&last["accel_cov_bits"])[0]);
        check(cov, expected(3), "accel");
        eigen_weighted_gram_accumulate_f32(&mut cov, g, parse(&last["gyro_cov_bits"])[0]);
        check(cov, expected(4), "gyro");
        check(
            cov,
            F32Matrix9::from_column_slice(&parse(&last["cov_after_bits"])),
            "typed final covariance",
        );
    }

    #[test]
    fn sophus_point_action_matches_pinned_fixtures() {
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f182ffd),
            f32::from_bits(0xbd582e43),
            f32::from_bits(0xbf4d686f),
            f32::from_bits(0x00000000),
        ));
        let fixtures = [
            (
                Vector3::new(
                    f32::from_bits(0x3c1df76d),
                    f32::from_bits(0xba397d97),
                    f32::from_bits(0xbb5ce9dd),
                ),
                [0x39c8bb60, 0xb8cebae8, 0x3c279e64],
            ),
            (
                Vector3::new(
                    f32::from_bits(0x3ec46faa),
                    f32::from_bits(0xbcf45e4c),
                    f32::from_bits(0xbe0fadf7),
                ),
                [0x3cabe7c0, 0xbbc3c000, 0x3ed16b71],
            ),
        ];
        for (point, expected) in fixtures {
            let actual = sophus_rotate_f32(rotation, point);
            assert_eq!(
                [actual.x.to_bits(), actual.y.to_bits(), actual.z.to_bits()],
                expected
            );
        }
    }

    #[test]
    fn frame2_to_frame3_velocity_matches_pinned_predict_state_boundary() {
        // Exact pre-solver frame-2 state and frame-2 -> frame-3 delta from
        // the pinned IntegratedImuMeasurement<float> direct probe.
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f17e7a6),
            f32::from_bits(0xbd5cf5f2),
            f32::from_bits(0xbf4d97c0),
            f32::from_bits(0xbbac4997),
        ));
        let velocity0 = Vector3::new(
            f32::from_bits(0x3d0e4cb0),
            f32::from_bits(0xbc11f114),
            f32::from_bits(0xbe116b90),
        );
        let delta_velocity = Vector3::new(
            f32::from_bits(0x3ecf7735),
            f32::from_bits(0xbcff201e),
            f32::from_bits(0xbe21e85a),
        );
        let gravity = Vector3::new(0.0_f32, 0.0_f32, -9.810000419616699_f32);
        let dt = f32::from_bits(0x3d4cccaa); // 49,999,872 ns * float(1e-9)
        let rotated = sophus_rotate_f32(rotation, delta_velocity);
        assert_eq!(
            [
                rotated.x.to_bits(),
                rotated.y.to_bits(),
                rotated.z.to_bits()
            ],
            [0x3cf777c0, 0xbc212910, 0x3edead68]
        );
        let velocity = predict_velocity_f32(velocity0, gravity, dt, rotated);
        assert_eq!(
            [
                velocity.x.to_bits(),
                velocity.y.to_bits(),
                velocity.z.to_bits()
            ],
            [0x3d850448, 0xbc998d12, 0xbe4a560c]
        );

        // The frame-0 -> frame-1 fixture remains an exact regression guard
        // for the first interval after introducing this gravity FMA boundary.
        let rotation0 = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f182ffd),
            f32::from_bits(0xbd582e43),
            f32::from_bits(0xbf4d686f),
            0.0,
        ));
        let delta_velocity1 = Vector3::new(
            f32::from_bits(0x3ec46faa),
            f32::from_bits(0xbcf45e4c),
            f32::from_bits(0xbe0fadf7),
        );
        let velocity1 = predict_velocity_f32(
            Vector3::zeros(),
            gravity,
            dt,
            sophus_rotate_f32(rotation0, delta_velocity1),
        );
        assert_eq!(
            [
                velocity1.x.to_bits(),
                velocity1.y.to_bits(),
                velocity1.z.to_bits()
            ],
            [0x3cabe7c0, 0xbbc3c000, 0xbda6dcd8]
        );
    }

    #[test]
    fn frame4_to_frame5_step_packet_velocity_matches_pinned_predict_state_boundary() {
        // Exact frame-4 post-optimization state and frame-4 -> frame-5 IMU
        // packet from the max-6 native/Rust comparison.  The native clean
        // RelWithDebInfo target uses -O3 -march=native; its Sophus point
        // action has a call-site-specific Eigen packet schedule captured by
        // aom::sophus_rotate_step_packet_f32.
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f12e4d4),
            f32::from_bits(0xbd5bbe69),
            f32::from_bits(0xbf51341f),
            f32::from_bits(0xbbecff35),
        ));
        let velocity0 = Vector3::new(
            f32::from_bits(0x3c03e9ad),
            f32::from_bits(0xbceb16f0),
            f32::from_bits(0xbf0a7d0f),
        );
        let delta_velocity = Vector3::new(
            f32::from_bits(0x3edaafe5),
            f32::from_bits(0xbd115f51),
            f32::from_bits(0xbe25d144),
        );
        let delta_rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f7fff35),
            f32::from_bits(0x3897b332),
            f32::from_bits(0xbb973edd),
            f32::from_bits(0x3ae013d8),
        ));
        let gravity = Vector3::new(0.0_f32, 0.0_f32, -9.810000419616699_f32);
        let dt = f32::from_bits(0x3d4cccaa); // 49,999,872 ns * float(1e-9)
        let rotated = sophus_rotate_step_packet_f32(rotation, delta_velocity);
        assert_eq!(
            [
                rotated.x.to_bits(),
                rotated.y.to_bits(),
                rotated.z.to_bits()
            ],
            [0x3ba17d00, 0xbc5943f4, 0x3eea7806]
        );
        let velocity = predict_velocity_f32(velocity0, gravity, dt, rotated);
        assert_eq!(
            [
                velocity.x.to_bits(),
                velocity.y.to_bits(),
                velocity.z.to_bits()
            ],
            [0x3c54a82d, 0xbd2bdc75, 0xbf12d25f]
        );
        let predicted_rotation = sophus_quat_product(rotation, delta_rotation);
        let q = predicted_rotation.quaternion();
        assert_eq!(
            [q.i.to_bits(), q.j.to_bits(), q.k.to_bits(), q.w.to_bits()],
            [0xbd616e25, 0xbf51db11, 0xbbc2cc7d, 0x3f11ee3e]
        );
    }

    #[test]
    fn frame3_to_frame4_position_packet_matches_native_fma_schedule() {
        // The final native frame-3 -> frame-4 IMU packet.  These operands are
        // the clean O3 packet boundary from m7ge_native_position_frame3.log;
        // the expected result is the clean preintegrated delta-position word.
        let position0 = Vector3::new(
            f32::from_bits(0x3c104834),
            f32::from_bits(0xba126f9a),
            f32::from_bits(0xbb4e9296),
        );
        let velocity0 = Vector3::new(
            f32::from_bits(0x3ec809c2),
            f32::from_bits(0xbca94c56),
            f32::from_bits(0xbe1019f4),
        );
        let accel_world = Vector3::new(
            f32::from_bits(0x410a149a),
            f32::from_bits(0xbe755aa8),
            f32::from_bits(0xc050382c),
        );
        let actual = integrate_position_f32(
            position0,
            velocity0,
            accel_world,
            f32::from_bits(0x3ba3d8a6), // 5,000,192 ns * float(1e-9)
        );
        assert_eq!(
            [actual.x.to_bits(), actual.y.to_bits(), actual.z.to_bits()],
            [0x3c320e93, 0xba2e4f54, 0xbb7f5a2b]
        );
    }

    #[test]
    fn frame3_to_frame4_point_action_matches_pinned_predict_state_boundary() {
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f16d68a),
            f32::from_bits(0xbd5e760f),
            f32::from_bits(0xbf4e5e85),
            f32::from_bits(0xbbc2d95f),
        ));
        let delta_velocity = Vector3::new(
            f32::from_bits(0x3ede21c0),
            f32::from_bits(0xbcb31cdc),
            f32::from_bits(0xbe20c273),
        );
        let rotated = sophus_rotate_f32(rotation, delta_velocity);
        assert_eq!(
            [
                rotated.x.to_bits(),
                rotated.y.to_bits(),
                rotated.z.to_bits()
            ],
            [0x3c8a91c0, 0x3ada00c0, 0x3eec551f]
        );
        let packet_rotated = sophus_rotate_step_packet_f32(rotation, delta_velocity);
        assert_eq!(
            [
                packet_rotated.x.to_bits(),
                packet_rotated.y.to_bits(),
                packet_rotated.z.to_bits()
            ],
            [0x3c8a91c0, 0x3ada00c0, 0x3eec551f]
        );
    }

    #[test]
    fn eigen_matrix_product_matches_pinned_g_packet_schedule() {
        let left = Matrix3::from_column_slice(&[
            f32::from_bits(0x3f7ffffd),
            f32::from_bits(0xb94c862d),
            f32::from_bits(0xba0caa74),
            f32::from_bits(0x394c40da),
            f32::from_bits(0x3f7ffffe),
            f32::from_bits(0xb9fc5647),
            f32::from_bits(0x3a0cb0bf),
            f32::from_bits(0x39fc483d),
            f32::from_bits(0x3f7ffffc),
        ]);
        let right = Matrix3::from_column_slice(&[
            f32::from_bits(0x3f7fffff),
            f32::from_bits(0x38cc4c68),
            f32::from_bits(0x398cafb3),
            f32::from_bits(0xb8cc7aa0),
            f32::from_bits(0x3f7fffff),
            f32::from_bits(0x397c4a95),
            f32::from_bits(0xb98cab81),
            f32::from_bits(0xb97c53f1),
            f32::from_bits(0x3f7fffff),
        ]);
        let actual = eigen_matrix_product_f32(left, right);
        let expected = [
            0x3f7fffff, 0xb8cc7aa0, 0xb98cab81, 0x38cc4c68, 0x3f7fffff, 0xb97c53f0, 0x398cafb3,
            0x397c4a95, 0x3f7fffff,
        ];
        assert_eq!(
            actual
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
    }

    fn cam() -> DoubleSphereCamera {
        DoubleSphereCamera::new(300., 300., 320., 240., 0.5, 0.7, 640, 480).unwrap()
    }

    fn obs(
        camera: &DoubleSphereCamera,
        id: u64,
        frame_id: u64,
        point: Point3<f64>,
    ) -> TrackObservation {
        TrackObservation {
            track_id: id,
            frame_id,
            timestamp_ns: frame_id as i64 + 1,
            camera_id: 0,
            pixel: camera.project(&point).unwrap(),
        }
    }

    #[test]
    fn known_depth_stereo_triangulation_is_within_one_percent() {
        let point = Vector3::new(0.2, 0.1, 4.0);
        let a = SE3::identity();
        let b = SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0., 0.));
        let ra = (point - a.translation).normalize();
        let rb = (point - b.translation).normalize();
        let (q, _) = triangulate_two_rays(&a, ra, &b, rb).unwrap();
        assert!((q - point).norm() / point.norm() < 0.01);
    }

    #[test]
    fn known_depth_temporal_fallback_is_within_one_percent() {
        let point = Vector3::new(-0.1, 0.2, 3.0);
        let a = SE3::identity();
        let b = SE3::new(UnitQuaternion::identity(), Vector3::new(0.15, 0., 0.));
        let ra = point.normalize();
        let rb = (point - b.translation).normalize();
        let (q, _) = triangulate_two_rays(&a, ra, &b, rb).unwrap();
        assert!((q - point).norm() / point.norm() < 0.01);
    }

    /// Pinhole camera plus a divergent two-camera rig shaped like Project
    /// Aria's SLAM pair: cam1 is cam0 rotated 75 degrees about cam0's x axis
    /// with a 0.138 m baseline.
    fn divergent_rig() -> (DoubleSphereCamera, Vec<SE3>) {
        let camera =
            DoubleSphereCamera::new(241.6, 241.6, 379.0, 286.0, 0.0, 0.0, 758, 572).unwrap();
        let cam1 = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(75_f64.to_radians(), 0.0, 0.0)),
            Vector3::new(0.004, -0.109, -0.085),
        );
        (camera, vec![SE3::identity(), cam1])
    }

    fn rig_estimator(landmarks_all_cameras: bool, scalar_mode: ScalarMode) -> BasaltVioEstimator {
        let (camera, extrinsics) = divergent_rig();
        let config = EstimatorConfig {
            scalar_mode,
            landmarks_all_cameras,
            ..EstimatorConfig::default()
        };
        BasaltVioEstimator::new(camera, config)
            .with_camera_rig(vec![camera, camera], extrinsics)
            .unwrap()
    }

    #[test]
    fn cam1_only_track_is_hosted_in_cam1_only_when_enabled() {
        let (camera, extrinsics) = divergent_rig();
        // In front of cam1 and outside cam0's image.
        let point_world = extrinsics[1].transform_point(&Point3::new(0.3, -0.2, 3.0));
        assert!(camera
            .project(&extrinsics[0].inverse().transform_point(&point_world))
            .is_none_or(|pixel| !camera.contains_pixel(&pixel)));
        // A stereo-visible point (seen by both cameras) for the cam0 path.
        let stereo_world = extrinsics[1].transform_point(&Point3::new(0.1, 1.5, 2.0));
        let nav_at = |x: f64| BasaltNavState {
            imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.02, 0.0)),
            ..BasaltNavState::default()
        };
        let project = |nav: &BasaltNavState, camera_id: usize, point: &Point3<f64>| {
            let camera_pose = nav.imu_to_world.compose(&extrinsics[camera_id]);
            camera
                .project(&camera_pose.inverse().transform_point(point))
                .filter(|pixel| camera.contains_pixel(pixel))
        };
        let frame = |frame_id: u64, nav: &BasaltNavState| {
            let mut observations = vec![TrackObservation {
                track_id: 5,
                frame_id,
                timestamp_ns: frame_id as i64 * 10,
                camera_id: 1,
                pixel: project(nav, 1, &point_world).expect("cam1 sees the point"),
            }];
            for camera_id in [0_u16, 1] {
                observations.push(TrackObservation {
                    track_id: 9,
                    frame_id,
                    timestamp_ns: frame_id as i64 * 10,
                    camera_id,
                    pixel: project(nav, camera_id as usize, &stereo_world)
                        .expect("both cameras see the stereo point"),
                });
            }
            observations
        };

        for scalar_mode in [ScalarMode::ExtendedF64, ScalarMode::UpstreamF32] {
            for enabled in [false, true] {
                let mut estimator = rig_estimator(enabled, scalar_mode);
                let (nav1, nav2) = (nav_at(0.0), nav_at(0.3));
                estimator
                    .collect_observations(1, 10, &nav1, &frame(1, &nav1), false)
                    .unwrap();
                estimator
                    .collect_observations(2, 20, &nav2, &frame(2, &nav2), true)
                    .unwrap();
                // The stereo track is cam0-hosted either way.
                let stereo = &estimator.landmarks.landmarks[&9].landmark;
                assert_eq!(stereo.anchor_camera_id, 0);
                if !enabled {
                    assert!(!estimator.landmarks.landmarks.contains_key(&5));
                    assert_eq!(estimator.num_points_kf.get(&2), Some(&1));
                    continue;
                }
                let landmark = &estimator.landmarks.landmarks[&5].landmark;
                assert_eq!(landmark.anchor_pose, 2);
                assert_eq!(landmark.anchor_camera_id, 1);
                assert_eq!(estimator.num_points_kf.get(&2), Some(&2));
                // Host identity (timestamp, camera 1) joins the native
                // unordered host map next to the cam0 host of this keyframe.
                let mut hosts = estimator.native_host_order.keys().collect::<Vec<_>>();
                hosts.sort_unstable();
                assert_eq!(hosts, vec![(20, 0), (20, 1)]);
                assert_eq!(estimator.native_host_keys.get(&(2, 1)), Some(&20));
                let host_pose = nav2.imu_to_world.compose(&extrinsics[1]);
                let reconstructed = host_pose
                    .transform_point(&Point3::from(landmark.position_in_anchor().unwrap()));
                let tolerance = if scalar_mode == ScalarMode::ExtendedF64 {
                    1e-6
                } else {
                    2e-3
                };
                assert!(
                    (reconstructed - point_world).norm() < tolerance,
                    "{scalar_mode:?}: {reconstructed:?} vs {point_world:?}"
                );
                // Factor-visible observations are the track's cam1 rows.
                let rows = &estimator.window_observations[&5];
                assert!(!rows.is_empty());
                assert!(rows.iter().all(|observation| observation.camera_id == 1));
            }
        }
    }

    #[test]
    fn reanchoring_keeps_the_host_camera_only_when_enabled() {
        let (_camera, extrinsics) = divergent_rig();
        for enabled in [false, true] {
            let mut estimator = rig_estimator(enabled, ScalarMode::ExtendedF64);
            let nav = BasaltNavState {
                imu_to_world: SE3::new(
                    UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.3)),
                    Vector3::new(1.0, 2.0, 0.5),
                ),
                ..BasaltNavState::default()
            };
            estimator.window_states.push(WindowState {
                frame_id: 8,
                timestamp_ns: 80,
                nav: nav.clone(),
                stored_current_nav: nav.clone(),
                linearized_nav: nav.clone(),
                linearized_delta: DVector::zeros(NAV_STATE_DOF),
                is_keyframe: true,
                is_latest: true,
                linearized: true,
            });
            // Host frame 3 has left the window; the point lies in front of
            // cam1 of the new anchor frame.
            let point_world = nav
                .imu_to_world
                .compose(&extrinsics[1])
                .transform_point(&Point3::new(0.1, 0.2, 2.0));
            estimator.landmarks.insert(
                44,
                InverseDistanceLandmark {
                    anchor_pose: 3,
                    anchor_camera_id: 1,
                    direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                    inverse_distance: 0.5,
                },
            );
            estimator.landmark_world.insert(44, point_world);
            estimator.reanchor_landmarks();
            let landmark = estimator.landmarks.landmarks[&44].landmark;
            assert_eq!(landmark.anchor_pose, 8);
            assert_eq!(landmark.anchor_camera_id, u16::from(enabled));
            let host = nav
                .imu_to_world
                .compose(&extrinsics[landmark.anchor_camera_id as usize]);
            let reconstructed =
                host.transform_point(&Point3::from(landmark.position_in_anchor().unwrap()));
            assert!((reconstructed - point_world).norm() < 1e-9);
        }
    }

    #[test]
    fn unconnected_tracks_are_created_only_on_keyframes_and_count_successes() {
        let camera = cam();
        let cam1_extrinsic = SE3::new(UnitQuaternion::identity(), Vector3::new(0.2, 0.0, 0.0));
        let mut config = EstimatorConfig::default();
        config.scalar_mode = ScalarMode::ExtendedF64;
        let mut estimator = BasaltVioEstimator::new(camera, config)
            .with_camera_rig(
                vec![camera, camera],
                vec![SE3::identity(), cam1_extrinsic.clone()],
            )
            .unwrap();
        let nav = BasaltNavState::default();
        let point_world = Point3::new(0.1, -0.05, 4.0);
        let observations = |frame_id: u64, timestamp_ns: i64| {
            [
                TrackObservation {
                    track_id: 77,
                    frame_id,
                    timestamp_ns,
                    camera_id: 0,
                    pixel: camera.project(&point_world).unwrap(),
                },
                TrackObservation {
                    track_id: 77,
                    frame_id,
                    timestamp_ns,
                    camera_id: 1,
                    pixel: camera
                        .project(&cam1_extrinsic.inverse().transform_point(&point_world))
                        .unwrap(),
                },
            ]
        };

        estimator
            .collect_observations(1, 10, &nav, &observations(1, 10), false)
            .unwrap();
        assert!(estimator.landmarks.landmarks.is_empty());

        estimator
            // Camera 1 arrives first here; upstream's current-frame map still
            // makes the camera-0 target and stereo candidate available.
            .collect_observations(
                2,
                20,
                &nav,
                &{
                    let frame = observations(2, 20);
                    [frame[1].clone(), frame[0].clone()]
                },
                true,
            )
            .unwrap();
        let landmark = &estimator.landmarks.landmarks[&77].landmark;
        assert_eq!(landmark.anchor_pose, 2);
        assert_eq!(landmark.anchor_camera_id, 0);
        assert_eq!(estimator.landmarks.landmarks[&77].connected_observations, 4);
        assert_eq!(estimator.num_points_kf.get(&2), Some(&1));
        // Host identity uses the native timestamp, not the logical frame
        // index or the first-arriving camera in the input packet.
        assert_eq!(
            estimator.native_host_order.keys().collect::<Vec<_>>(),
            vec![(20, 0)]
        );
        assert_eq!(estimator.native_host_keys.get(&(2, 0)), Some(&20));
        assert_eq!(estimator.build_problem().trial_host_order, vec![(20, 0)]);
        let reconstructed = landmark.position_in_anchor().unwrap();
        assert!((reconstructed - point_world.coords).norm() < 1e-8);

        // A keyframe candidate with no earlier baseline is not triangulated;
        // its denominator must therefore be zero, rather than an active
        // landmark count delta inherited from the previous window.
        estimator
            .collect_observations(
                3,
                30,
                &nav,
                &[TrackObservation {
                    track_id: 88,
                    frame_id: 3,
                    timestamp_ns: 30,
                    camera_id: 0,
                    pixel: camera.project(&point_world).unwrap(),
                }],
                true,
            )
            .unwrap();
        assert_eq!(estimator.num_points_kf.get(&3), Some(&0));
        assert_eq!(
            estimator.native_host_order.keys().collect::<Vec<_>>(),
            vec![(20, 0)]
        );
        assert_eq!(estimator.native_host_keys.len(), 1);
    }

    #[test]
    fn factor_projection_backfills_native_keyframe_membership_for_1288_and_2018() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        for frame_id in [21_u64, 35, 42, 49] {
            estimator.window_states.push(WindowState {
                frame_id,
                timestamp_ns: frame_id as i64,
                nav: BasaltNavState::default(),
                stored_current_nav: BasaltNavState::default(),
                linearized_nav: BasaltNavState::default(),
                linearized_delta: DVector::zeros(NAV_STATE_DOF),
                is_keyframe: true,
                is_latest: false,
                linearized: true,
            });
        }
        // Frame 28 has already converted to a pose-only keyframe.  The
        // classifier must still recognize it when rebuilding a delayed
        // landmark's factor projection.
        estimator.window_poses.push(WindowPose {
            frame_id: 28,
            timestamp_ns: 28,
            pose: SE3::identity(),
            stored_current_pose: SE3::identity(),
            linearized_pose: SE3::identity(),
            linearized_delta: DVector::zeros(POSE_DOF),
            is_keyframe: true,
        });

        let history = |frame_id: u64, camera_id: u16| TrackHistory {
            frame_id,
            camera_id,
            timestamp_ns: frame_id as i64,
            pixel: Point2::new(frame_id as f64, camera_id as f64),
            raw_bearing: Vector3::z(),
            imu_pose: SE3::identity(),
            pose: SE3::identity(),
        };
        let mut candidates_1288 = BTreeMap::new();
        for frame_id in [21_u64, 22, 27, 28, 35, 42, 48, 49, 50] {
            candidates_1288.insert((frame_id as i64, 0, frame_id), history(frame_id, 0));
        }
        let keyframes = estimator.keyframe_frame_ids(50, false);
        assert_eq!(keyframes, vec![21, 28, 35, 42, 49]);
        let projected_1288 =
            BasaltVioEstimator::project_factor_observations(&candidates_1288, 50, &keyframes);
        assert_eq!(
            projected_1288
                .iter()
                .map(|observation| observation.frame_id)
                .collect::<Vec<_>>(),
            vec![21, 28, 35, 42, 49, 50]
        );
        assert_eq!(candidates_1288.len(), 9);

        let mut candidates_2018 = BTreeMap::new();
        for frame_id in [42_u64, 43, 48, 49, 50] {
            candidates_2018.insert((frame_id as i64, 0, frame_id), history(frame_id, 0));
        }
        let projected_2018 =
            BasaltVioEstimator::project_factor_observations(&candidates_2018, 50, &keyframes);
        assert_eq!(
            projected_2018
                .iter()
                .map(|observation| observation.frame_id)
                .collect::<Vec<_>>(),
            vec![42, 49, 50]
        );
        assert_eq!(candidates_2018.len(), 5);

        // At the following non-keyframe event, the sample after KF49 remains
        // in the native factor block; older non-keyframe samples remain
        // excluded and the current event is included independently.
        let mut candidates_1288_next = candidates_1288.clone();
        candidates_1288_next.insert((51_i64, 0, 51), history(51, 0));
        let projected_1288_next =
            BasaltVioEstimator::project_factor_observations(&candidates_1288_next, 51, &keyframes);
        assert_eq!(
            projected_1288_next
                .iter()
                .map(|observation| observation.frame_id)
                .collect::<Vec<_>>(),
            vec![21, 28, 35, 42, 49, 50, 51]
        );
    }

    #[test]
    fn active_factor_projection_drops_non_keyframe_before_next_solve() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        for frame_id in [21_u64, 28, 35, 42, 49] {
            estimator.window_states.push(WindowState {
                frame_id,
                timestamp_ns: frame_id as i64,
                nav: BasaltNavState::default(),
                stored_current_nav: BasaltNavState::default(),
                linearized_nav: BasaltNavState::default(),
                linearized_delta: DVector::zeros(NAV_STATE_DOF),
                is_keyframe: true,
                is_latest: false,
                linearized: true,
            });
        }
        estimator.landmarks.insert(
            1288,
            InverseDistanceLandmark {
                anchor_pose: 28,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        estimator.window_observations.insert(
            1288,
            [21_u64, 28, 35, 42, 48, 49]
                .into_iter()
                .map(|frame_id| StoredObservation {
                    frame_id,
                    camera_id: 0,
                    pixel: Point2::new(frame_id as f64, 1.0),
                })
                .collect(),
        );
        let nav = BasaltNavState::default();
        let observation = TrackObservation {
            track_id: 1288,
            frame_id: 50,
            timestamp_ns: 50,
            camera_id: 0,
            pixel: camera.project(&Point3::new(0.1, -0.05, 4.0)).unwrap(),
        };
        estimator
            .collect_observations(50, 50, &nav, &[observation], false)
            .unwrap();
        assert_eq!(
            estimator.window_observations[&1288]
                .iter()
                .map(|observation| observation.frame_id)
                .collect::<Vec<_>>(),
            vec![21, 28, 35, 42, 49, 50]
        );
    }

    #[test]
    fn new_keyframe_keeps_existing_landmark_active_nonkeyframe_observations() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        for frame_id in [0, 5, 6] {
            estimator.window_states.push(WindowState {
                frame_id,
                timestamp_ns: frame_id as i64,
                nav: BasaltNavState::default(),
                stored_current_nav: BasaltNavState::default(),
                linearized_nav: BasaltNavState::default(),
                linearized_delta: DVector::zeros(NAV_STATE_DOF),
                is_keyframe: frame_id == 0,
                is_latest: false,
                linearized: false,
            });
        }
        estimator.landmarks.insert(
            118,
            InverseDistanceLandmark {
                anchor_pose: 0,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        estimator.window_observations.insert(
            118,
            [0, 5, 6]
                .into_iter()
                .map(|frame_id| StoredObservation {
                    frame_id,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                })
                .collect(),
        );
        let current = obs(&camera, 118, 7, Point3::new(0.1, -0.05, 4.0));
        let expected_pixel = current.pixel;
        estimator.of_images.insert(
            7,
            vec![OfImageData::new(7, current.timestamp_ns, 0, 1, 1, vec![1]).unwrap()],
        );
        estimator
            .collect_observations(
                7,
                current.timestamp_ns,
                &BasaltNavState::default(),
                &[current],
                true,
            )
            .unwrap();
        assert_eq!(
            estimator.window_observations[&118]
                .iter()
                .map(|r| r.frame_id)
                .collect::<Vec<_>>(),
            vec![0, 5, 6, 7]
        );
        assert_eq!(estimator.of_observations[&7].len(), 1);
        assert_eq!(estimator.of_observations[&7][0].x, expected_pixel.x);
        assert_eq!(estimator.of_observations[&7][0].y, expected_pixel.y);
    }

    #[test]
    fn duplicate_current_observation_fails_before_any_factor_mutation() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let current = obs(&camera, 77, 5, Point3::new(0.1, -0.05, 4.0));

        let error = estimator
            .collect_observations(
                5,
                6,
                &BasaltNavState::default(),
                &[current.clone(), current],
                false,
            )
            .expect_err("duplicate current identity must fail closed");
        assert!(error.contains("duplicate current observation identity"));
        assert!(estimator.landmarks.landmarks.is_empty());
        assert!(estimator.landmarks.observations.is_empty());
        assert!(estimator.window_observations.is_empty());
        assert!(estimator.track_history.is_empty());
        assert!(estimator.landmark_world.is_empty());
    }

    #[test]
    fn current_observation_timestamp_mismatch_fails_without_rewriting_history() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let mut current = obs(&camera, 77, 5, Point3::new(0.1, -0.05, 4.0));
        current.timestamp_ns += 1;

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[current], false)
            .expect_err("current timestamp mismatch must fail closed");
        assert!(error.contains("does not match current timestamp"));
        assert!(estimator.track_history.is_empty());
        assert!(estimator.landmarks.landmarks.is_empty());
        assert!(estimator.landmarks.observations.is_empty());
        assert!(estimator.window_observations.is_empty());
    }

    #[test]
    fn duplicate_history_fails_before_landmark_or_factor_insertion() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let history = TrackHistory {
            frame_id: 4,
            camera_id: 0,
            timestamp_ns: 4,
            pixel: Point2::new(320.0, 240.0),
            raw_bearing: Vector3::z(),
            imu_pose: SE3::identity(),
            pose: SE3::identity(),
        };
        estimator
            .track_history
            .insert(77, vec![history.clone(), history]);
        let current = obs(&camera, 77, 5, Point3::new(0.1, -0.05, 4.0));

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[current], true)
            .expect_err("duplicate retained identity must fail closed");
        assert!(error.contains("duplicate track history identity"));
        assert!(estimator.landmarks.landmarks.is_empty());
        assert!(estimator.landmarks.observations.is_empty());
        assert!(estimator.window_observations.is_empty());
        assert!(estimator.landmark_world.is_empty());
        assert_eq!(estimator.track_history[&77].len(), 2);
    }

    #[test]
    fn invalid_historical_bearing_fails_before_landmark_insertion() {
        let camera = cam();
        let mut config = EstimatorConfig::default();
        config.scalar_mode = ScalarMode::ExtendedF64;
        let mut estimator = BasaltVioEstimator::new(camera, config);
        // ExtendedF64 triangulation can use the stored pose/raw bearing, but
        // this camera id is deliberately absent from the one-camera rig. The
        // all-candidate bearing preflight must reject it before insertion.
        let candidate_pose = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0));
        estimator.track_history.insert(
            77,
            vec![TrackHistory {
                frame_id: 4,
                camera_id: 1,
                timestamp_ns: 4,
                pixel: Point2::new(320.0, 240.0),
                raw_bearing: Vector3::new(0.2, 0.0, 4.0),
                imu_pose: candidate_pose.clone(),
                pose: candidate_pose,
            }],
        );
        let current = obs(&camera, 77, 5, Point3::new(0.0, 0.0, 4.0));

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[current], true)
            .expect_err("invalid historical camera must fail closed");
        assert!(error.contains("invalid DS history observation"));
        assert!(estimator.landmarks.landmarks.is_empty());
        assert!(estimator.landmarks.observations.is_empty());
        assert!(estimator.window_observations.is_empty());
        assert!(estimator.landmark_world.is_empty());
    }

    #[test]
    fn later_track_failure_does_not_commit_earlier_track_mutations() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        // Preserve a populated table (including erased-node history), not
        // merely an empty default, when a later track aborts the packet.
        estimator.native_host_order.insert(100, 0);
        estimator.native_host_order.insert(200, 1);
        estimator.native_host_order.remove(200, 1);
        estimator.native_host_keys.insert((1, 0), 100);
        estimator.landmarks.insert(
            1,
            InverseDistanceLandmark {
                anchor_pose: 1,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        let original_record = estimator.landmarks.landmarks[&1];
        estimator.window_observations.insert(
            1,
            vec![StoredObservation {
                frame_id: 1,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            }],
        );
        let invalid_pose = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0));
        estimator.track_history.insert(
            2,
            vec![TrackHistory {
                frame_id: 4,
                camera_id: 1,
                timestamp_ns: 4,
                pixel: Point2::new(320.0, 240.0),
                raw_bearing: Vector3::new(0.2, 0.0, 4.0),
                imu_pose: invalid_pose.clone(),
                pose: invalid_pose,
            }],
        );
        let original_history = estimator.track_history.clone();
        let original_factor_map = estimator.window_observations.clone();
        let original_hosts = estimator.native_host_order.clone();
        let original_host_keys = estimator.native_host_keys.clone();
        let first = obs(&camera, 1, 5, Point3::new(0.1, -0.05, 4.0));
        let second = obs(&camera, 2, 5, Point3::new(0.0, 0.0, 4.0));

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[first, second], true)
            .expect_err("second track's malformed history must abort the packet");
        assert!(error.contains("invalid DS history observation"));
        assert_eq!(estimator.landmarks.landmarks[&1], original_record);
        assert!(estimator.landmarks.observations.is_empty());
        assert_eq!(estimator.track_history, original_history);
        assert_eq!(estimator.window_observations, original_factor_map);
        assert_eq!(estimator.native_host_order, original_hosts);
        assert_eq!(estimator.native_host_keys, original_host_keys);
        assert!(estimator.landmark_world.is_empty());
    }

    #[test]
    fn native_host_cleanup_waits_for_last_observed_landmark() {
        let mut estimator = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        for (track, host, timestamp) in [(1, 10, 1000), (2, 10, 1000), (3, 20, 2000)] {
            estimator.native_host_order.insert(timestamp, 0);
            estimator.native_host_keys.insert((host, 0), timestamp);
            estimator.landmarks.insert(
                track,
                InverseDistanceLandmark {
                    anchor_pose: host,
                    anchor_camera_id: 0,
                    direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                    inverse_distance: 0.5,
                },
            );
            estimator.window_observations.insert(
                track,
                vec![StoredObservation {
                    frame_id: 30,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                }],
            );
        }
        let before = estimator.native_host_order.keys().collect::<Vec<_>>();
        estimator.window_observations.remove(&1);
        estimator.prune_native_host_order();
        assert_eq!(
            estimator.native_host_order.keys().collect::<Vec<_>>(),
            before
        );
        estimator.window_observations.remove(&2);
        estimator.prune_native_host_order();
        assert_eq!(
            estimator.native_host_order.keys().collect::<Vec<_>>(),
            vec![(2000, 0)]
        );
        assert!(!estimator.native_host_keys.contains_key(&(10, 0)));
        estimator.landmarks.landmarks.remove(&3);
        estimator.prune_native_host_order();
        assert_eq!(estimator.native_host_order.keys().count(), 0);
        assert!(estimator.native_host_keys.is_empty());
    }

    #[test]
    fn duplicate_existing_factor_map_fails_before_association() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        estimator.landmarks.insert(
            7,
            InverseDistanceLandmark {
                anchor_pose: 1,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        estimator.window_observations.insert(
            7,
            vec![
                StoredObservation {
                    frame_id: 1,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                },
                StoredObservation {
                    frame_id: 1,
                    camera_id: 0,
                    pixel: Point2::new(321.0, 240.0),
                },
            ],
        );
        let original_record = estimator.landmarks.landmarks[&7];
        let observation = obs(&camera, 7, 5, Point3::new(0.1, -0.05, 4.0));

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[observation], false)
            .expect_err("duplicate existing factor identity must fail closed");
        assert!(error.contains("duplicate factor observation identity"));
        assert_eq!(estimator.landmarks.landmarks[&7], original_record);
        assert!(estimator.landmarks.observations.is_empty());
        assert_eq!(estimator.window_observations[&7].len(), 2);
        assert!(estimator.track_history.is_empty());
    }

    #[test]
    fn current_factor_identity_collision_fails_without_history() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        estimator.landmarks.insert(
            7,
            InverseDistanceLandmark {
                anchor_pose: 1,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        // Deliberately leave track_history absent: the factor projection can
        // outlive that delayed store during a loss/rebinding boundary.
        estimator.window_observations.insert(
            7,
            vec![StoredObservation {
                frame_id: 5,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            }],
        );
        let original_factor_map = estimator.window_observations.clone();
        let observation = obs(&camera, 7, 5, Point3::new(0.1, -0.05, 4.0));

        let error = estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[observation], false)
            .expect_err("current factor identity collision must fail closed");
        assert!(error.contains("duplicate current factor identity"));
        assert_eq!(estimator.window_observations, original_factor_map);
        assert!(estimator.track_history.is_empty());
        assert!(estimator.landmarks.observations.is_empty());
    }

    #[test]
    fn lost_and_host_drop_boundaries_keep_factor_map_and_database_separate() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let direction = StereographicDirection::from_bearing(Vector3::z()).unwrap();
        estimator.landmarks.insert(
            1288,
            InverseDistanceLandmark {
                anchor_pose: 28,
                anchor_camera_id: 0,
                direction,
                inverse_distance: 0.5,
            },
        );
        estimator.window_observations.insert(
            1288,
            vec![StoredObservation {
                frame_id: 50,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            }],
        );

        // Loss is marked before solve, so the pre-solve factor-visible map
        // still contains the track. Deferred removal then removes the DB
        // record, after which the production retain step drops its map entry.
        estimator.landmarks.mark_frame_end_deferred(&[]);
        assert_eq!(
            estimator.landmarks.landmarks[&1288].status,
            LandmarkStatus::Lost
        );
        assert!(estimator.window_observations.contains_key(&1288));
        assert_eq!(estimator.landmarks.remove_deferred_lost(), 1);
        estimator
            .window_observations
            .retain(|track_id, _| estimator.landmarks.landmarks.contains_key(track_id));
        assert!(!estimator.landmarks.landmarks.contains_key(&1288));
        assert!(!estimator.window_observations.contains_key(&1288));

        // Host deletion is independent of observation-frame cleanup and does
        // not reanchor a surviving landmark.
        estimator.landmarks.insert(
            2018,
            InverseDistanceLandmark {
                anchor_pose: 49,
                anchor_camera_id: 0,
                direction,
                inverse_distance: 0.5,
            },
        );
        assert_eq!(estimator.landmarks.remove_host_landmarks(&[49]), 1);
        assert!(!estimator.landmarks.landmarks.contains_key(&2018));
    }

    #[test]
    fn lost_landmark_reassociation_reactivates_without_duplicate_identity() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        estimator.landmarks.insert(
            1288,
            InverseDistanceLandmark {
                anchor_pose: 28,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        {
            let record = estimator.landmarks.landmarks.get_mut(&1288).unwrap();
            record.status = LandmarkStatus::Lost;
            record.missed_frames = 1;
        }
        estimator.window_observations.insert(
            1288,
            vec![StoredObservation {
                frame_id: 4,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            }],
        );

        let current = obs(&camera, 1288, 5, Point3::new(0.1, -0.05, 4.0));
        estimator
            .collect_observations(5, 6, &BasaltNavState::default(), &[current], false)
            .expect("a previously lost but retained landmark can be observed again");

        let record = &estimator.landmarks.landmarks[&1288];
        assert_eq!(record.status, LandmarkStatus::Active);
        assert_eq!(record.missed_frames, 0);
        assert_eq!(record.connected_observations, 1);
        assert_eq!(estimator.landmarks.observations.len(), 1);
        assert_eq!(estimator.track_history[&1288].len(), 1);
        assert_eq!(estimator.window_observations[&1288].len(), 1);
        assert_eq!(estimator.window_observations[&1288][0].frame_id, 5);
    }

    #[test]
    fn first_camera_initialization_matches_upstream_accel_alignment() {
        let samples = [
            ImuSample::new(90, Vector3::new(1.0, 2.0, 3.0), Vector3::new(1.0, 0.0, 0.0)),
            ImuSample::new(
                100,
                Vector3::new(0.2, -0.1, 0.3),
                Vector3::new(0.0, 2.0, 0.0),
            ),
        ];
        let state = initial_nav_from_imu(100, &samples, ScalarMode::ExtendedF64);
        let aligned = state
            .imu_to_world
            .rotation
            .transform_vector(&Vector3::y_axis());
        assert!((aligned - Vector3::z()).norm() < 1.0e-12);
        assert!(state.imu_to_world.translation.norm() < 1.0e-12);
        assert!(state.velocity_world_m_s.norm() < 1.0e-12);
        assert!(state.gyro_bias_rad_s.norm() < 1.0e-12);
        assert!(state.accel_bias_m_s2.norm() < 1.0e-12);
    }

    #[test]
    fn adapter_bootstrap_uses_first_future_imu_when_frame_interval_is_empty() {
        let mut estimator = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        let bootstrap = ImuSample::new(
            125,
            Vector3::new(0.2, -0.1, 0.3),
            Vector3::new(0.0, 2.0, 0.0),
        );
        let output = estimator
            .process_adapter_frame(0, 100, &[], &[], Some(bootstrap), None, false, true)
            .unwrap();
        let initialized = output.state_trace.initialization_output.unwrap();
        let aligned = initialized
            .imu_to_world
            .rotation
            .transform_vector(&Vector3::y_axis());
        assert!((aligned - Vector3::z()).norm() < 1.0e-6);
        assert_ne!(
            initialized.imu_to_world.rotation,
            UnitQuaternion::identity()
        );
    }

    #[test]
    fn imu_prediction_includes_previous_velocity_in_position_residual() {
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let previous = BasaltNavState {
            imu_to_world: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.2, 0.05)),
                Vector3::new(0.4, -0.3, 0.2),
            ),
            velocity_world_m_s: Vector3::new(1.2, -0.7, 0.35),
            gyro_bias_rad_s: Vector3::new(0.01, -0.02, 0.03),
            accel_bias_m_s2: Vector3::new(-0.2, 0.1, 0.05),
        };
        let noise = ImuNoiseModel {
            gyro_density: 0.02,
            accel_density: 0.3,
        };
        let mut preintegrator =
            ImuPreintegrator::new(previous.gyro_bias_rad_s, previous.accel_bias_m_s2)
                .with_noise(noise)
                .unwrap();
        preintegrator.integrate_sample(
            Vector3::new(0.04, -0.03, 0.02),
            Vector3::new(0.2, -0.1, 9.6),
            0.05,
        );
        let delta = preintegrator.delta().clone();
        let predicted = predict_nav(&previous, Some(&delta), gravity, ScalarMode::ExtendedF64);
        let factor =
            crate::imu::whitened_preintegration_factor(&previous, &predicted, &delta, gravity)
                .expect("synthetic prediction must have a valid covariance");

        assert!(factor.residual.norm() < 1.0e-8);
    }

    #[test]
    fn static_imu_calibration_matches_basalt_headers_parameter_layout() {
        let accel_params = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let gyro_params = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let raw = Vector3::new(10.0, 20.0, 30.0);
        assert!(
            (calibrate_accel(&accel_params, raw) - Vector3::new(10.0, 28.0, 58.0)).norm() < 1e-12
        );
        assert!(
            (calibrate_gyro(&gyro_params, raw) - Vector3::new(39.0, 54.0, 69.0)).norm() < 1e-12
        );
    }

    #[test]
    fn fallback_interval_uses_last_packet_for_trailing_endpoint() {
        let samples = [
            ImuSample::new(10_000_000, Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0)),
            ImuSample::new(20_000_000, Vector3::zeros(), Vector3::new(3.0, 0.0, 0.0)),
        ];
        let noise = ImuNoiseModel {
            gyro_density: 1.0,
            accel_density: 1.0,
        };
        let delta = fallback_integrate(
            &samples,
            0,
            25_000_000,
            Vector3::zeros(),
            Vector3::zeros(),
            noise,
        )
        .unwrap();
        let mut reference = ImuPreintegrator::new(Vector3::zeros(), Vector3::zeros())
            .with_noise(noise)
            .unwrap();
        reference.integrate_sample(Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0), 0.01);
        reference.integrate_sample(Vector3::zeros(), Vector3::new(3.0, 0.0, 0.0), 0.01);
        reference.integrate_sample(Vector3::zeros(), Vector3::new(3.0, 0.0, 0.0), 0.005);
        assert!((delta.delta_position - reference.delta().delta_position).norm() < 1.0e-12);
    }

    #[test]
    fn upstream_f32_interval_keeps_integer_endpoint_and_sophus_exp_order() {
        // Calibrated float samples from the pinned MH_01 frame 1 -> 2 golden
        // (target/basalt_upstream_imu_golden_frame1_2.json).  This is a
        // deliberately small integration oracle: it catches both per-sample
        // f32 accumulation and the SO3 exp implementation before either can
        // be hidden by residual whitening.
        let samples = [
            ImuSample::new(
                1403636579818555392,
                Vector3::new(-0.117891803383827, -0.0232203491032124, -0.0431733503937721),
                Vector3::new(7.73393440246582, -0.577644169330597, -3.00091099739075),
            ),
            ImuSample::new(
                1403636579823555584,
                Vector3::new(-0.115099273622036, -0.0399755090475082, -0.0508527979254723),
                Vector3::new(7.84017324447632, -0.569471955299377, -2.91918873786926),
            ),
            ImuSample::new(
                1403636579828555520,
                Vector3::new(-0.108117960393429, -0.0637119859457016, -0.0613247714936733),
                Vector3::new(8.03630542755127, -0.544955372810364, -2.84563899040222),
            ),
            ImuSample::new(
                1403636579833555456,
                Vector3::new(-0.105325430631638, -0.0853540748357773, -0.0704004839062691),
                Vector3::new(8.13437271118164, -0.520438730716705, -2.78026127815247),
            ),
            ImuSample::new(
                1403636579838555392,
                Vector3::new(-0.0962497219443321, -0.102109231054783, -0.0759855359792709),
                Vector3::new(8.25695514678955, -0.487749874591827, -2.72305583953857),
            ),
            ImuSample::new(
                1403636579843555584,
                Vector3::new(-0.0871740058064461, -0.11397746950388, -0.0780799314379692),
                Vector3::new(8.35502243041992, -0.536783158779144, -2.6903669834137),
            ),
            ImuSample::new(
                1403636579848555520,
                Vector3::new(-0.0760039016604424, -0.119562521576881, -0.0801743268966675),
                Vector3::new(8.29781627655029, -0.471405506134033, -2.62498927116394),
            ),
            ImuSample::new(
                1403636579853555456,
                Vector3::new(-0.0690225809812546, -0.127940103411674, -0.0752874091267586),
                Vector3::new(8.21609401702881, -0.504094302654266, -2.67402267456055),
            ),
            ImuSample::new(
                1403636579858555392,
                Vector3::new(-0.0634375289082527, -0.14190274477005, -0.0669098272919655),
                Vector3::new(8.24878311157227, -0.528610944747925, -2.73940014839172),
            ),
            ImuSample::new(
                1403636579863555584,
                Vector3::new(-0.0606450065970421, -0.150978446006775, -0.0585322454571724),
                Vector3::new(8.11802768707275, -0.561299800872803, -2.84563899040222),
            ),
        ];
        let (delta, fallback) = integrate_queue_f32(
            &samples,
            1403636579813555456,
            1403636579863555584,
            Vector3::zeros(),
            Vector3::zeros(),
            ImuNoiseModel {
                gyro_density: 0.000282 * (200.0_f64).sqrt(),
                accel_density: 0.016 * (200.0_f64).sqrt(),
            },
        );
        let delta = delta.expect("golden interval must integrate");
        assert!(!fallback);
        assert_eq!(
            delta.delta_time.to_bits(),
            (0.050000128_f32 as f64).to_bits()
        );
        let expected_position = Vector3::new(
            0.010060002095997334,
            -0.00068823189940303564,
            -0.0035189196933060884,
        );
        let expected_velocity = Vector3::new(
            0.4063984751701355,
            -0.027501031756401062,
            -0.13839077949523926,
        );
        assert_eq!(delta.delta_position, expected_position);
        assert_eq!(delta.delta_velocity, expected_velocity);
        let q = delta.delta_rotation.quaternion();
        assert_eq!(
            [q.i, q.j, q.k, q.w],
            [
                -0.0022482054773718119,
                -0.0024225490633398294,
                -0.0016497710021212697,
                0.99999320507049561,
            ]
        );
    }

    #[test]
    fn half_open_reader_interval_keeps_leading_imu_dt() {
        let samples = [
            ImuSample::new(10, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81)),
            ImuSample::new(20, Vector3::zeros(), Vector3::new(0.0, 0.0, 9.81)),
        ];
        let delta = fallback_integrate(
            &samples,
            0,
            20,
            Vector3::zeros(),
            Vector3::zeros(),
            ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
        )
        .unwrap();
        assert!((delta.delta_time - 20.0e-9).abs() < 1.0e-15);
        assert!((delta.delta_velocity.z - 9.81 * 20.0e-9).abs() < 1.0e-12);
    }

    #[test]
    fn production_gravity_is_fixed_to_basalt_world_constant() {
        let estimator = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        assert_eq!(estimator.gravity_world, Some(Vector3::new(0.0, 0.0, -9.81)));
    }

    #[test]
    fn upstream_keyframe_counter_uses_strict_min_gap_for_frames_zero_to_ten() {
        let mut estimator = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        let decisions = (0..=10)
            // A low connected ratio makes every frame eligible once the
            // upstream counter gate opens.
            .map(|_| estimator.decide_keyframe(0, 100))
            .collect::<Vec<_>>();
        assert_eq!(
            decisions,
            vec![true, false, false, false, false, false, false, true, false, false, false]
        );
    }

    #[test]
    fn urgent_keyframe_policy_shortens_gap_only_below_its_threshold() {
        let config = EstimatorConfig {
            urgent_kf: Some(UrgentKeyframePolicy {
                threshold: 0.3,
                min_frames_after_kf: 2,
            }),
            ..EstimatorConfig::default()
        };
        // Ratio 0.5 is below the regular threshold only: upstream spacing.
        let mut estimator = BasaltVioEstimator::new(cam(), config);
        let moderate = (0..=10)
            .map(|_| estimator.decide_keyframe(50, 50))
            .collect::<Vec<_>>();
        assert_eq!(
            moderate,
            vec![true, false, false, false, false, false, false, true, false, false, false]
        );
        // Ratio 0.0 is below the urgent threshold: strict gap of two.
        let mut estimator = BasaltVioEstimator::new(cam(), config);
        let severe = (0..=10)
            .map(|_| estimator.decide_keyframe(0, 100))
            .collect::<Vec<_>>();
        assert_eq!(
            severe,
            vec![true, false, false, false, true, false, false, false, true, false, false]
        );
    }

    #[test]
    fn degenerate_pending_track_does_not_create_placeholder_landmark() {
        let mut estimator = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        let observation = obs(&estimator.camera, 42, 0, Point3::new(0.0, 0.0, 2.0));
        estimator.process(0, 1, &[observation], &[]).unwrap();
        assert!(!estimator.landmarks.landmarks.contains_key(&42));
    }

    #[test]
    fn synthetic_window_has_cross_blocks_and_shifts() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let point = Point3::new(0.1, -0.1, 2.0);
        let mut outputs = Vec::new();
        for frame in 0..5 {
            let timestamp_ns = frame as i64 * 1_000_000 + 1;
            let mut observation = obs(&camera, 1, frame, point);
            observation.timestamp_ns = timestamp_ns;
            let imu = if frame == 0 {
                Vec::new()
            } else {
                vec![
                    ImuSample::new(
                        frame as i64 * 1_000_000 - 500_000,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    ImuSample::new(frame as i64 * 1_000_000, Vector3::zeros(), Vector3::zeros()),
                ]
            };
            outputs.push(
                estimator
                    .process(frame, timestamp_ns, &[observation], &imu)
                    .unwrap(),
            );
        }
        assert!(outputs
            .iter()
            .any(|output| output.marg_data.aom_sqrt_jacobian.cols >= 30));
        assert!(outputs
            .last()
            .unwrap()
            .marg_data
            .aom_abs_h
            .as_ref()
            .unwrap()
            .data
            .iter()
            .any(|value| value.abs() > 0.0));
        let shifted = outputs.last().unwrap();
        assert!(
            shifted.marg_data.validate(),
            "synthetic schema4 MargData invalid: {}",
            shifted.marg_data.validate_contract().unwrap_err()
        );
        assert_eq!(
            shifted.marg_data.aom_dof(),
            shifted.marg_data.aom_sqrt_jacobian.cols
        );
        // A marginal packet is captured before the active-window shift, so
        // its full frame tables intentionally differ from the post-shift
        // public counts.  The packet must retain at least the surviving
        // blocks and may additionally contain the selected/dropped blocks.
        assert!(
            shifted.marg_data.frame_poses.len() + shifted.marg_data.frame_states.len()
                >= shifted.active_pose_count + shifted.active_state_count
        );
        assert!(shifted
            .marg_data
            .aom_order
            .windows(2)
            .all(|blocks| blocks[0].offset + blocks[0].dof == blocks[1].offset));
        assert_eq!(
            shifted.marg_data.marginalization.states_to_marg_vel_bias,
            vec![0]
        );
        let solved = outputs
            .iter()
            .find(|output| output.window.attempted)
            .expect("the fifth state starts optimization");
        assert_eq!(
            solved.marg_data.row_counts.iter().sum::<usize>(),
            solved.marg_data.aom_sqrt_jacobian.rows
        );
        assert!(solved.marg_data.row_counts.iter().sum::<usize>() <= solved.window.factor_rows);
        assert!(solved.marg_data.row_counts[2] > 0); // IMU rows
        assert!(solved.marg_data.row_counts[3] > 0); // bias random walk rows
                                                     // Upstream triggers at `size >= max_states` and retains the boundary
                                                     // plus latest state, hence two full states for max_states=3.
        assert_eq!(estimator.active_state_count(), 2);
        assert!(estimator.has_prior());
        let packet_latest = shifted
            .marg_data
            .frame_states
            .iter()
            .find(|state| state.is_latest)
            .expect("pre-marg packet keeps the newest state table entry");
        assert!(!packet_latest.linearized);
        let boundary_id = shifted
            .marg_data
            .aom_order
            .iter()
            .rev()
            .find(|block| block.kind == "state")
            .expect("mixed AOM keeps the FEJ boundary state")
            .frame_id;
        assert!(estimator
            .states
            .iter()
            .find(|state| state.frame_id == boundary_id)
            .is_some_and(|state| state.linearized));
    }

    #[test]
    fn marginal_packet_keeps_images_for_a_keyframe_selected_for_removal() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        estimator.window_poses.push(WindowPose {
            frame_id: 1,
            timestamp_ns: 10,
            pose: SE3::identity(),
            stored_current_pose: SE3::identity(),
            linearized_pose: SE3::identity(),
            linearized_delta: DVector::zeros(POSE_DOF),
            is_keyframe: true,
        });
        estimator.window_states.push(WindowState {
            frame_id: 2,
            timestamp_ns: 20,
            nav: BasaltNavState::default(),
            stored_current_nav: BasaltNavState::default(),
            linearized_nav: BasaltNavState::default(),
            linearized_delta: DVector::zeros(NAV_STATE_DOF),
            is_keyframe: true,
            is_latest: true,
            linearized: false,
        });
        for frame_id in [1_u64, 2, 9] {
            estimator.of_images.insert(
                frame_id,
                vec![
                    OfImageData::new(frame_id, frame_id as i64 * 10, 0, 1, 1, vec![1]).unwrap(),
                    OfImageData::new(frame_id, frame_id as i64 * 10, 1, 1, 1, vec![2]).unwrap(),
                ],
            );
            estimator.of_observations.insert(
                frame_id,
                vec![OfObservationData {
                    frame_id,
                    track_id: 100 + frame_id,
                    camera_id: 0,
                    x: frame_id as f64,
                    y: -(frame_id as f64),
                }],
            );
        }
        // Frame 9 is no longer in the post-shift window, but upstream's
        // marginal packet is assembled before erasing the selected keyframe.
        estimator.last_kf_to_marg = vec![(9, 10)];
        // `emit_marg` normally follows `process`, which refreshes the public
        // compatibility table from `window_states`.  Mirror that lifecycle
        // step here so the v4 FEJ sidecar/table cardinalities are exercised.
        estimator.refresh_public_states();
        let diagnostics = WindowDiagnostics {
            attempted: false,
            state_dof: 0,
            factor_count: 0,
            factor_rows: 0,
            landmark_count: 0,
            imu_link_count: 0,
            prior_rows: 0,
            prior_factor_rows: 0,
            visual_factor_rows: 0,
            imu_factor_rows: 0,
            bias_factor_rows: 0,
            prior_cost: 0.0,
            visual_cost: 0.0,
            imu_cost: 0.0,
            bias_cost: 0.0,
            imu_links: Vec::new(),
            lm: Vec::new(),
            status: "test".into(),
            failure: None,
            state_writeback: false,
            landmark_writeback: 0,
            prior_carry: false,
        };
        let data = estimator.emit_marg(&diagnostics);
        assert_eq!(
            data.of_images
                .iter()
                .map(|image| image.frame_id)
                .collect::<Vec<_>>(),
            vec![1, 1, 2, 2, 9, 9]
        );
        assert_eq!(
            data.of_observations
                .iter()
                .map(|observation| observation.frame_id)
                .collect::<Vec<_>>(),
            vec![1, 2, 9]
        );
        assert!(data.validate());
    }

    #[test]
    fn world_cache_drops_tracks_removed_from_landmark_database() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let landmark = |anchor_pose| InverseDistanceLandmark {
            anchor_pose,
            anchor_camera_id: 0,
            direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
            inverse_distance: 0.5,
        };
        estimator.landmarks.insert(7, landmark(1));
        estimator.landmarks.insert(8, landmark(1));
        estimator
            .landmark_world
            .insert(7, Point3::new(1.0, 2.0, 3.0));
        estimator
            .landmark_world
            .insert(8, Point3::new(4.0, 5.0, 6.0));
        estimator
            .landmark_world
            .insert(99, Point3::new(7.0, 8.0, 9.0));

        estimator.landmarks.mark_frame_end_deferred(&[7]);
        assert_eq!(estimator.landmarks.remove_deferred_lost(), 1);
        estimator.prune_landmark_world_cache();

        assert!(estimator.landmark_world.contains_key(&7));
        assert!(!estimator.landmark_world.contains_key(&8));
        assert!(!estimator.landmark_world.contains_key(&99));
    }

    #[test]
    fn old_kf_connected_ratio_counts_stereo_observations_like_upstream() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        estimator.landmarks.insert(
            7,
            InverseDistanceLandmark {
                anchor_pose: 35,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::z()).unwrap(),
                inverse_distance: 0.5,
            },
        );
        estimator.num_points_kf.insert(35, 51);
        estimator.window_observations.insert(
            7,
            vec![
                StoredObservation {
                    frame_id: 51,
                    camera_id: 0,
                    pixel: Point2::new(10.0, 20.0),
                },
                StoredObservation {
                    frame_id: 51,
                    camera_id: 1,
                    pixel: Point2::new(11.0, 20.0),
                },
                StoredObservation {
                    frame_id: 50,
                    camera_id: 0,
                    pixel: Point2::new(9.0, 20.0),
                },
            ],
        );

        assert_eq!(estimator.connected_ratio_for_kf(35, 51), 2.0 / 51.0);
    }

    #[test]
    fn raw_image_capture_leaves_mapper_off_vio_trace_identical() {
        let camera = cam();
        let mut plain = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let mut captured = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let point = Point3::new(0.1, -0.1, 2.0);
        for frame in 0..5_u64 {
            let timestamp_ns = frame as i64 * 1_000_000 + 1;
            let mut observation = obs(&camera, 1, frame, point);
            observation.timestamp_ns = timestamp_ns;
            let imu = if frame == 0 {
                Vec::new()
            } else {
                vec![
                    ImuSample::new(
                        frame as i64 * 1_000_000 - 500_000,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    ImuSample::new(frame as i64 * 1_000_000, Vector3::zeros(), Vector3::zeros()),
                ]
            };
            let left = plain
                .process(frame, timestamp_ns, &[observation.clone()], &imu)
                .unwrap();
            let right = captured
                .process_with_images(
                    frame,
                    timestamp_ns,
                    &[observation],
                    &imu,
                    Some(vec![OfImageData::new(
                        frame,
                        timestamp_ns,
                        0,
                        1,
                        1,
                        vec![frame as u16],
                    )
                    .unwrap()]),
                )
                .unwrap();
            assert_eq!(left.state, right.state);
            assert_eq!(left.is_keyframe, right.is_keyframe);
            assert_eq!(left.connected_cam0, right.connected_cam0);
            assert_eq!(left.unconnected_cam0, right.unconnected_cam0);
            assert_eq!(left.active_state_count, right.active_state_count);
            assert_eq!(left.active_pose_count, right.active_pose_count);
            assert_eq!(left.state_trace, right.state_trace);
            assert_eq!(left.phases, right.phases);
            assert_eq!(left.window, right.window);
            assert_eq!(
                left.imu_integration_fallback,
                right.imu_integration_fallback
            );
            let mut left_marg = left.marg_data;
            let mut right_marg = right.marg_data;
            left_marg.of_images.clear();
            right_marg.of_images.clear();
            left_marg.of_observations.clear();
            right_marg.of_observations.clear();
            assert_eq!(left_marg, right_marg);
        }
    }

    #[test]
    fn no_marg_data_mode_preserves_solver_and_window_outputs() {
        let camera = cam();
        let mut retained = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let mut suppressed = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let point = Point3::new(0.1, -0.1, 2.0);

        for frame in 0..5_u64 {
            let timestamp_ns = frame as i64 * 1_000_000 + 1;
            let mut observation = obs(&camera, 1, frame, point);
            observation.timestamp_ns = timestamp_ns;
            let imu = if frame == 0 {
                Vec::new()
            } else {
                vec![
                    ImuSample::new(
                        frame as i64 * 1_000_000 - 500_000,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    ImuSample::new(frame as i64 * 1_000_000, Vector3::zeros(), Vector3::zeros()),
                ]
            };
            let expected = retained
                .process(frame, timestamp_ns, &[observation.clone()], &imu)
                .unwrap();
            let actual = suppressed
                .process_without_marg_data(frame, timestamp_ns, &[observation], &imu)
                .unwrap();

            assert_eq!(actual.state, expected.state);
            assert_eq!(actual.is_keyframe, expected.is_keyframe);
            assert_eq!(actual.connected_cam0, expected.connected_cam0);
            assert_eq!(actual.unconnected_cam0, expected.unconnected_cam0);
            assert_eq!(actual.active_state_count, expected.active_state_count);
            assert_eq!(actual.active_pose_count, expected.active_pose_count);
            assert_eq!(actual.state_trace, expected.state_trace);
            assert_eq!(actual.phases, expected.phases);
            assert_eq!(actual.window, expected.window);
            assert_eq!(
                actual.imu_integration_fallback,
                expected.imu_integration_fallback
            );
            assert_eq!(actual.marg_data, MargData::empty());
            assert!(actual.marg_data.validate());
        }
    }

    #[test]
    fn no_trace_path_preserves_solver_and_omits_optional_payloads() {
        let camera = cam();
        let mut retained = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let mut suppressed = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let point = Point3::new(0.1, -0.1, 2.0);
        let probe_fallback = diagnostic_env_active();
        let canonical_harness_env = [
            "VISLOC_BASALT_GT_FREE",
            "VISLOC_BASALT_PROFILE",
            "VISLOC_BASALT_SEQUENCE",
            "VISLOC_BASALT_INPUT_ROOT",
            "VISLOC_BASALT_OUTPUT_ROOT",
            "VISLOC_BASALT_SEED",
            "VISLOC_BASALT_TEMPORAL_SEED",
            "VISLOC_BASALT_THREADS",
        ]
        .into_iter()
        .all(|key| std::env::var_os(key).is_some());
        if canonical_harness_env {
            assert!(!probe_fallback);
        }
        let diagnostic_probe_env = [
            "VISLOC_BASALT_DETAIL_ITERATIONS",
            "VISLOC_BASALT_DIAGNOSTIC_IMU_HB",
        ]
        .into_iter()
        .any(|key| std::env::var_os(key).is_some());
        if diagnostic_probe_env {
            assert!(probe_fallback);
        }

        for frame in 0..5_u64 {
            let timestamp_ns = frame as i64 * 1_000_000 + 1;
            let mut observation = obs(&camera, 1, frame, point);
            observation.timestamp_ns = timestamp_ns;
            let imu = if frame == 0 {
                Vec::new()
            } else {
                vec![
                    ImuSample::new(
                        frame as i64 * 1_000_000 - 500_000,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    ImuSample::new(frame as i64 * 1_000_000, Vector3::zeros(), Vector3::zeros()),
                ]
            };
            let expected = retained
                .process(frame, timestamp_ns, &[observation.clone()], &imu)
                .unwrap();
            let actual = suppressed
                .process_without_marg_data_no_trace(frame, timestamp_ns, &[observation], &imu)
                .unwrap();

            assert_eq!(actual.state, expected.state);
            assert_eq!(actual.is_keyframe, expected.is_keyframe);
            assert_eq!(actual.connected_cam0, expected.connected_cam0);
            assert_eq!(actual.unconnected_cam0, expected.unconnected_cam0);
            assert_eq!(actual.active_state_count, expected.active_state_count);
            assert_eq!(actual.active_pose_count, expected.active_pose_count);
            assert_eq!(
                actual.imu_integration_fallback,
                expected.imu_integration_fallback
            );
            assert_eq!(actual.marg_data, MargData::empty());
            assert!(actual.marg_data.validate());

            if probe_fallback {
                assert_eq!(actual.state_trace, expected.state_trace);
                assert_eq!(actual.phases, expected.phases);
                assert_eq!(actual.window, expected.window);
            } else {
                assert_eq!(
                    actual.state_trace.predicted_state,
                    expected.state_trace.predicted_state
                );
                assert_eq!(
                    actual.state_trace.post_opt_state,
                    expected.state_trace.post_opt_state
                );
                assert!(actual.state_trace.state_from.is_none());
                assert!(actual.state_trace.initialization_output.is_none());
                assert!(actual.state_trace.imu_propagation.is_none());
                assert!(actual.phases.is_empty());

                assert_eq!(actual.window.attempted, expected.window.attempted);
                assert_eq!(actual.window.state_dof, expected.window.state_dof);
                assert_eq!(actual.window.status, expected.window.status);
                assert_eq!(actual.window.factor_count, 0);
                assert_eq!(actual.window.factor_rows, 0);
                assert_eq!(actual.window.prior_factor_rows, 0);
                assert_eq!(actual.window.visual_factor_rows, 0);
                assert_eq!(actual.window.imu_factor_rows, 0);
                assert_eq!(actual.window.bias_factor_rows, 0);
                assert_eq!(actual.window.prior_cost, 0.0);
                assert_eq!(actual.window.visual_cost, 0.0);
                assert_eq!(actual.window.imu_cost, 0.0);
                assert_eq!(actual.window.bias_cost, 0.0);
                assert!(actual.window.imu_links.is_empty());
                assert!(actual.window.lm.is_empty());
            }
        }
    }

    #[test]
    fn no_output_mode_allows_lean_continuation_and_rejects_full_return() {
        let camera = cam();
        let point = Point3::new(0.1, -0.1, 2.0);
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());

        estimator
            .process(0, 1, &[obs(&camera, 1, 0, point)], &[])
            .unwrap();
        estimator
            .process_without_marg_data_no_trace(1, 2, &[obs(&camera, 1, 1, point)], &[])
            .unwrap();
        let continued = estimator
            .process_without_marg_data_no_trace(2, 3, &[obs(&camera, 1, 2, point)], &[])
            .unwrap();
        assert_eq!(continued.marg_data, MargData::empty());
        assert!(continued.marg_data.validate());

        let error = estimator
            .process(3, 4, &[obs(&camera, 1, 3, point)], &[])
            .expect_err("full output must be rejected after lean mode starts");
        assert!(error.contains("optical-flow payloads"));
        assert!(error.contains("recreate"));
    }

    #[test]
    fn failed_lean_call_does_not_latch_full_mode_out() {
        let camera = cam();
        let point = Point3::new(0.1, -0.1, 2.0);
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let mut invalid = obs(&camera, 1, 0, point);
        invalid.camera_id = 7;

        estimator
            .process_without_marg_data_no_trace(0, 1, &[invalid], &[])
            .expect_err("an observation from an unconfigured camera must fail");
        assert!(!estimator.no_output_mode_active());

        let retried = estimator
            .process(1, 2, &[obs(&camera, 1, 1, point)], &[])
            .expect("a failed lean call must not block a full retry");
        assert_eq!(retried.frame_id, 1);
        assert!(!estimator.no_output_mode_active());
    }

    #[test]
    fn upstream_initialization_gate_defers_lm_until_fifth_state() {
        let camera = cam();
        let mut estimator = BasaltVioEstimator::new(camera, EstimatorConfig::default());
        let point = Point3::new(0.1, -0.1, 2.0);
        let mut outputs = Vec::new();
        for frame in 0..5 {
            let timestamp_ns = frame as i64 * 1_000_000 + 1;
            let mut observation = obs(&camera, 1, frame, point);
            observation.timestamp_ns = timestamp_ns;
            let imu = if frame == 0 {
                Vec::new()
            } else {
                vec![
                    ImuSample::new(
                        frame as i64 * 1_000_000 - 500_000,
                        Vector3::zeros(),
                        Vector3::zeros(),
                    ),
                    ImuSample::new(frame as i64 * 1_000_000, Vector3::zeros(), Vector3::zeros()),
                ]
            };
            outputs.push(
                estimator
                    .process(frame, timestamp_ns, &[observation], &imu)
                    .unwrap(),
            );
        }

        for output in outputs.iter().take(4) {
            assert!(!output.window.attempted);
            assert_eq!(
                output.state_trace.predicted_state,
                output.state_trace.post_opt_state
            );
            assert!(!output
                .state_trace
                .post_opt_state
                .imu_to_world
                .translation
                .iter()
                .any(|value| !value.is_finite()));
        }
        assert!(outputs[4].window.attempted);
        assert!(outputs[4].window.lm.iter().any(|run| run.pass == "first"));
    }

    #[test]
    fn synthetic_stereo_inertial_replay_is_finite_bounded_and_deterministic() {
        let mut first = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        let mut second = BasaltVioEstimator::new(cam(), EstimatorConfig::default());
        let point = Point3::new(0.1, 0.0, 2.0);
        let mut last = None;
        for frame in 0..8 {
            let observation = obs(&first.camera, 1, frame, point);
            let imu = [ImuSample::new(
                frame as i64 * 10,
                nalgebra::Vector3::zeros(),
                nalgebra::Vector3::new(0., 0., 9.81),
            )];
            let a = first
                .process(frame, frame as i64 + 1, &[observation.clone()], &imu)
                .unwrap();
            let b = second
                .process(frame, frame as i64 + 1, &[observation.clone()], &imu)
                .unwrap();
            assert_eq!(a.marg_data.stable_hash(), b.marg_data.stable_hash());
            assert!(a
                .state
                .imu_to_world
                .translation
                .iter()
                .all(|v| v.is_finite()));
            assert!(
                a.marg_data.validate(),
                "synthetic schema4 MargData invalid: {}",
                a.marg_data.validate_contract().unwrap_err()
            );
            last = Some(a);
        }
        assert!(first.states.len() <= 3);
        assert!(last.unwrap().marg_data.aom_sqrt_jacobian.cols >= 15);
    }
}
