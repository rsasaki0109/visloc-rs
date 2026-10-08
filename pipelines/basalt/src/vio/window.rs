//! Active-window AOM problem used by the Basalt VIO estimator.
//!
//! Every navigation state owns one contiguous 15-column block in the global
//! system.  Visual factors are grouped by track so all observations of one
//! landmark are eliminated together by ABS_QR; this is important because a
//! single two-row observation cannot constrain a three-parameter landmark.

use super::aom::{
    active_diagnostic_lm_iteration, anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode,
    anchored_visual_reprojection_factor_with_time_cam, back_substitute_landmark,
    back_substitute_landmark_upstream_f32_with_track, emit_full70_factor_oracle,
    full70_f32_bits_matrix, full70_f32_bits_vector, landmark_nullspace_projection,
    landmark_nullspace_projection_f32, lm_trial_vector_fingerprint, prior_factor,
    reduce_landmark_factors, solve_lm_with_timing, AnchoredVisualFactor, FactorConfig, FactorKind,
    LmConfig, LmDiagnosticEvent, LmFailure, LmLinearization, LmProblem, LmResult, LmTraceEntry,
    LmTrialBinding, LmTrialPreparation, LmTrialToken, ReducedNormalSystem, VisualChainF32,
    WhitenedFactorRowStack,
};
#[cfg(test)]
use super::aom::{
    anchored_visual_reprojection_factor_f32_with_time_cam, compact_trial_preparation_for_test,
};
use super::landmarks::{sophus_so3_product, InverseDistanceLandmark, StereographicDirection};
use super::scalar::ScalarMode;
use crate::camera::DoubleSphereCamera;
use crate::imu::{
    preintegration_f32_expression_trace, preintegration_f32_rotation_producer_trace,
    preintegration_f32_stages_fej_mode, preintegration_residual_f32,
    whitened_bias_random_walk_factor, whitened_bias_random_walk_factor_upstream_f32,
    whitened_preintegration_factor, whitened_preintegration_factor_upstream_f32_fej_mode,
    BiasRandomWalkNoise, ImuNoiseModel, ImuPreintegratedDelta,
};
use crate::timing::TimingStart;
use crate::{BasaltNavState, TrackId};
use crate::{TimingBreakdown, TimingBucket};
use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix4, Point2, Quaternion, SymmetricEigen, UnitQuaternion, Vector3,
};
use rayon::prelude::*;
use serde_json::{json, Value as JsonValue};
use std::{
    cell::Cell,
    collections::{BTreeMap, HashMap},
    env,
    ffi::{OsStr, OsString},
    fmt::Write as FmtWrite,
    fs,
    fs::OpenOptions,
    io::Write as IoWrite,
    ops::AddAssign,
    path::PathBuf,
    sync::{Mutex, OnceLock},
};
use visloc_core::geometry::SE3;

pub const NAV_STATE_DOF: usize = 15;
/// A retained keyframe pose has the six Basalt pose tangent coordinates
/// (translation followed by the left-local rotation).  Upstream keeps these
/// blocks after the velocity/bias part of a keyframe has been marginalized.
pub const POSE_DOF: usize = 6;

#[derive(Debug, Clone, PartialEq)]
pub struct WindowState {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub nav: BasaltNavState,
    /// The upstream `PoseVelBiasStateWithLin::state_current` storage.  This
    /// is deliberately separate from `nav`: while `linearized` is false,
    /// upstream `applyInc` advances `state_linearized` but leaves this stored
    /// value untouched.  Keeping it here changes no effective solver value;
    /// it only makes the schema-4 MargData snapshot lossless.
    pub stored_current_nav: BasaltNavState,
    /// The upstream `PoseVelBiasStateWithLin::state_linearized` sidecar.
    /// Non-linearized states move this point together with `nav`; once the
    /// boundary is frozen, accepted increments accumulate in `nav` while
    /// this FEJ point remains fixed.
    pub linearized_nav: BasaltNavState,
    /// Accumulated local increments since `linearized_nav`, equivalent to
    /// upstream `PoseVelBiasStateWithLin::delta`.  This is intentionally not
    /// reconstructed with a box-minus: Sophus applies a sequence of left
    /// SO(3) increments while Basalt stores their f32 sum.
    pub linearized_delta: DVector<f64>,
    pub is_keyframe: bool,
    pub is_latest: bool,
    /// Whether this navigation block has been frozen as a Basalt FEJ
    /// linearization point.  New states enter false; the boundary state is
    /// marked true immediately after its marginalization packet is captured.
    /// Keeping this bit on the internal window state is necessary because
    /// `FrameStateData::linearized` is part of the serialized MargData
    /// contract, and refreshing every public state to `true` loses the
    /// distinction between the newest state and FEJ states.
    pub linearized: bool,
}

/// Pose-only block retained for a marginalized keyframe.  This is deliberately
/// separate from `WindowState`: a pose-only keyframe must not acquire fake
/// velocity or bias variables merely because it remains in the visual window.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowPose {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub pose: SE3,
    /// The upstream `PoseStateWithLin::T_w_i_current` storage.  It can be
    /// stale while a pose is not linearized, so it must not be inferred from
    /// the effective `pose` at serialization time.
    pub stored_current_pose: SE3,
    /// FEJ pose retained by upstream `PoseStateWithLin` after a state is
    /// converted to a pose-only block.
    pub linearized_pose: SE3,
    /// Accumulated pose tangent since `linearized_pose`.
    pub linearized_delta: DVector<f64>,
    pub is_keyframe: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowBlockKind {
    Pose,
    /// The 6-DoF pose prefix of a full 15-DoF navigation state.
    ///
    /// Native Basalt can retain this prefix in a square-root prior after a
    /// state block has been selected for marginalization.  It still resolves
    /// through the state table and the state block's global offset; only the
    /// first six chart lanes participate in the prior columns.
    StatePose,
    State,
}

/// Absolute column layout of the active AOM.
///
/// Basalt keeps pose-only blocks in a prefix and full navigation blocks after
/// that prefix.  IMU metadata must come from this layout rather than deriving
/// a block start from a column remainder: a pose prefix makes `% 15` an
/// invalid ownership test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WindowLayout {
    pose_count: usize,
    state_count: usize,
}

impl WindowLayout {
    const fn new(pose_count: usize, state_count: usize) -> Self {
        Self {
            pose_count,
            state_count,
        }
    }

    fn offset_for_pose(self, index: usize) -> Option<usize> {
        (index < self.pose_count).then_some(index * POSE_DOF)
    }

    fn offset_for_state(self, index: usize) -> Option<usize> {
        (index < self.state_count).then_some(self.pose_count * POSE_DOF + index * NAV_STATE_DOF)
    }

    const fn state_dof(self) -> usize {
        self.pose_count * POSE_DOF + self.state_count * NAV_STATE_DOF
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowObservation {
    pub state_index: usize,
    pub camera_id: u16,
    pub pixel: Point2<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowLandmark {
    pub track_id: TrackId,
    /// Index of the host keyframe state in this active window.
    pub anchor_state_index: usize,
    /// Host camera stream (`TimeCamId::cam_id` upstream).
    pub anchor_camera_id: u16,
    pub direction: StereographicDirection,
    pub inverse_distance: f64,
    pub observations: Vec<WindowObservation>,
}

impl WindowLandmark {
    const fn parameter(&self, anchor_frame_id: u64) -> InverseDistanceLandmark {
        InverseDistanceLandmark {
            anchor_pose: anchor_frame_id,
            anchor_camera_id: self.anchor_camera_id,
            direction: self.direction,
            inverse_distance: self.inverse_distance,
        }
    }

    fn apply_increment(&mut self, increment: Vector3<f64>, anchor_frame_id: u64) -> bool {
        let mut parameter = self.parameter(anchor_frame_id);
        if !parameter.apply_increment(increment) {
            return false;
        }
        self.direction = parameter.direction;
        self.inverse_distance = parameter.inverse_distance;
        true
    }

    fn apply_increment_f32(&mut self, increment: Vector3<f64>, anchor_frame_id: u64) -> bool {
        let mut parameter = self.parameter(anchor_frame_id);
        if !parameter.apply_increment_f32(increment) {
            return false;
        }
        self.direction = parameter.direction;
        self.inverse_distance = parameter.inverse_distance;
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowImuLink {
    pub from_index: usize,
    pub to_index: usize,
    pub delta: ImuPreintegratedDelta,
}

/// A square-root prior whose columns are identified by frame id rather than
/// by an unstable position in the current window.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowPrior {
    pub frame_ids: Vec<u64>,
    /// Block kinds in the same order as `frame_ids`.  `StatePose` is a
    /// prior-only six-DoF prefix resolved through the matching state table;
    /// it is not an active AOM block.  Old callers that build a prior by hand
    /// can leave this empty; that is interpreted as all-15DoF navigation
    /// blocks for backwards compatibility.
    pub block_kinds: Vec<WindowBlockKind>,
    pub jacobian: DMatrix<f64>,
    pub rhs: DVector<f64>,
    pub fej_point: DVector<f64>,
}

impl WindowPrior {
    pub fn rows(&self) -> usize {
        self.rhs.len()
    }

    pub fn kinds(&self) -> Vec<WindowBlockKind> {
        if self.block_kinds.len() == self.frame_ids.len() {
            self.block_kinds.clone()
        } else {
            vec![WindowBlockKind::State; self.frame_ids.len()]
        }
    }

    pub fn dof(&self) -> usize {
        self.kinds()
            .into_iter()
            .map(|kind| match kind {
                WindowBlockKind::Pose => POSE_DOF,
                WindowBlockKind::StatePose => POSE_DOF,
                WindowBlockKind::State => NAV_STATE_DOF,
            })
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowSolveResult {
    pub state: DVector<f64>,
    pub factors: Vec<WhitenedFactorRowStack>,
    pub cost: f64,
    pub iterations: usize,
    pub diagnostics: WindowDiagnostics,
}

/// One LM pass retained for the sensor-only trace.  The complete per-step
/// records are intentionally kept here instead of reduced to a final cost so
/// rejected trial steps and lambda evolution remain auditable.
#[derive(Debug, Clone, PartialEq)]
pub struct LmRunDiagnostics {
    pub pass: String,
    pub iterations: usize,
    pub lambda: f64,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub accepted: usize,
    pub rejected: usize,
    pub trace: Vec<LmTraceEntry>,
    pub failure: Option<String>,
}

/// Literal upstream IMU-factor audit for one active-window link.  The factor
/// itself remains owned by `imu::factors`; this adapter only records the
/// unwhitened contract alongside the square-root row that enters ABS_QR.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuLinkDiagnostics {
    pub from_frame_id: u64,
    pub to_frame_id: u64,
    pub delta_time: f64,
    pub residual_position: [f64; 3],
    pub residual_rotation: [f64; 3],
    pub residual_velocity: [f64; 3],
    pub whitened_norm: f64,
    pub gyro_bias_delta: [f64; 3],
    pub accel_bias_delta: [f64; 3],
    pub bias_rotation_correction: [f64; 3],
    pub bias_velocity_correction: [f64; 3],
    pub bias_position_correction: [f64; 3],
    pub covariance_eigen_min: f64,
    pub covariance_eigen_max: f64,
}

/// Per-frame active-window audit record.  This is emitted by the estimator and
/// serialized by the dedicated EuRoC example; it does not alter the solver's
/// numerical path.
#[derive(Debug, Clone, PartialEq)]
pub struct WindowDiagnostics {
    pub attempted: bool,
    pub state_dof: usize,
    pub factor_count: usize,
    pub factor_rows: usize,
    pub landmark_count: usize,
    pub imu_link_count: usize,
    pub prior_rows: usize,
    pub prior_factor_rows: usize,
    pub visual_factor_rows: usize,
    pub imu_factor_rows: usize,
    pub bias_factor_rows: usize,
    /// Initial residual-cost decomposition.  The category split is used only
    /// for parity diagnostics; LM still consumes the same stacked rows.
    pub prior_cost: f64,
    pub visual_cost: f64,
    pub imu_cost: f64,
    pub bias_cost: f64,
    pub imu_links: Vec<ImuLinkDiagnostics>,
    pub lm: Vec<LmRunDiagnostics>,
    pub status: String,
    pub failure: Option<String>,
    pub state_writeback: bool,
    pub landmark_writeback: usize,
    pub prior_carry: bool,
}

/// One visual observation captured by the opt-in frame-4 numeric audit.
///
/// This deliberately keeps the anchored factor (rather than reconstructing
/// it from an already stacked row) so the audit can distinguish the raw pixel
/// residual, robust objective, and weighted `Jp`/`Jl` values.  None of these
/// records participate in solving.
#[derive(Debug, Clone)]
struct VisualObservationAudit {
    observation: WindowObservation,
    factor: AnchoredVisualFactor,
}

/// One landmark's complete ABS_QR input and reduced rows for the opt-in audit.
#[derive(Debug, Clone)]
struct VisualLandmarkAudit {
    track_id: TrackId,
    anchor_frame_id: u64,
    anchor_camera_id: u16,
    direction: StereographicDirection,
    inverse_distance: f64,
    factor: WhitenedFactorRowStack,
    observations: Vec<VisualObservationAudit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowSolveError {
    pub message: String,
    pub diagnostics: WindowDiagnostics,
}

impl std::fmt::Display for WindowSolveError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for WindowSolveError {}

fn lm_run_diagnostics(pass: &str, result: &LmResult) -> LmRunDiagnostics {
    let accepted = result
        .trace
        .iter()
        .filter(|entry| matches!(entry.decision, super::aom::LmDecision::Accepted))
        .count();
    let rejected = result
        .trace
        .iter()
        .filter(|entry| matches!(entry.decision, super::aom::LmDecision::Rejected))
        .count();
    LmRunDiagnostics {
        pass: pass.to_owned(),
        iterations: result.iterations,
        lambda: result.lambda,
        initial_cost: result
            .trace
            .first()
            .map(|entry| entry.cost_before)
            .unwrap_or(result.cost),
        final_cost: result.cost,
        accepted,
        rejected,
        trace: result.trace.clone(),
        failure: None,
    }
}

fn iteration_trace_path() -> Option<std::path::PathBuf> {
    let policy = diagnostic_env_snapshot();
    policy
        .detail_iterations
        .then(|| policy.detail_trace.clone())
        .flatten()
}

/// These variables are harness metadata or fixture paths.  They are not read
/// by the estimator's numeric/diagnostic hooks, so they do not disable the
/// lean no-output path.  Keep this list explicit: an unknown future
/// `VISLOC_BASALT_*` key must fail closed into the diagnostic path.
const NON_DIAGNOSTIC_ENV_SUFFIXES: &[&str] = &[
    "GT_FREE",
    "PROFILE",
    "SEQUENCE",
    "INPUT_ROOT",
    "OUTPUT_ROOT",
    "SEED",
    "TEMPORAL_SEED",
    "THREADS",
    // Pure timing instrumentation; it does not enable a diagnostic payload.
    "TIMING_BREAKDOWN",
    // Test-only fixture/output paths confirmed by the source-side harnesses.
    "TRACE",
    "M8D_SETUP_INPUT",
    "M8D_SETUP_ORACLE",
    "MH01_ROOT",
    "CALIBRATION",
    "CONFIG",
    // The identity sidecar reads already-owned estimator records at a frame
    // boundary and does not require the retained diagnostic solver path.
    "VISUAL_FACTOR_IDENTITY_TRACE",
    "VISUAL_FACTOR_IDENTITY_TRACKS",
    // The lifecycle sidecar reads store counts/IDs only at a frame boundary;
    // it deliberately remains on the lean solver path.
    "LIFECYCLE_TRACE",
];

/// Diagnostic values which are consulted from the estimator/AOM hot paths.
///
/// The process environment is captured once and converted to owned values so
/// the solver never performs an environment lookup, path allocation, or filter
/// parse while reducing a factor.  The policy is intentionally process scoped:
/// callers must set all diagnostic variables before the first estimator call.
#[derive(Debug, Default)]
pub(crate) struct DiagnosticEnvSnapshot {
    pub(crate) active: bool,
    pub(crate) detail_iterations: bool,
    pub(crate) detail_trace: Option<PathBuf>,
    pub(crate) detail_frame: Option<u64>,
    pub(crate) solver_frontier_trace: Option<PathBuf>,
    pub(crate) diagnostic_imu_rows: Option<PathBuf>,
    pub(crate) diagnostic_imu_factor_inputs: Option<PathBuf>,
    /// JSONL path for the opt-in initial f64 IMU unwhitened-residual capture.
    /// This records only the already-computed quaternion product and raw
    /// rotation residual; it never participates in factor construction.
    pub(crate) diagnostic_imu_unwhitened_f64: Option<PathBuf>,
    pub(crate) diagnostic_frame: Option<u64>,
    pub(crate) diagnostic_imu_link: usize,
    pub(crate) diagnostic_imu_iteration: Option<usize>,
    pub(crate) diagnostic_imu_iterations: Option<Vec<usize>>,
    pub(crate) diagnostic_imu_hb: Option<PathBuf>,
    pub(crate) diagnostic_iteration: Option<usize>,
    pub(crate) diagnostic_normal_stages: bool,
    pub(crate) state_update_trace_frame: Option<u64>,
    pub(crate) marg_trace: Option<PathBuf>,
    pub(crate) marg_trace_event: Option<usize>,
    pub(crate) sqrt_trace: bool,
    pub(crate) sqrt_focus_k12: bool,
    pub(crate) sqrt_dump_workspace_all: bool,
    pub(crate) sqrt_dump_tail: bool,
    pub(crate) sqrt_dump_qj: Option<PathBuf>,
    pub(crate) visual_prefix_trace: Option<PathBuf>,
    /// JSONL visual-factor identity sidecar.  This is intentionally separate
    /// from the reduced-row visual-prefix trace: it records the estimator's
    /// causal observation/landmark lifecycle before solve/marginalization.
    pub(crate) visual_factor_identity_trace: Option<PathBuf>,
    pub(crate) visual_factor_identity_tracks: Option<Vec<u64>>,
    /// Selector values are parsed while the process snapshot is built.  The
    /// inner error is retained so a malformed configured selector fails closed
    /// without making AOM decode the raw environment value again.
    pub(crate) visual_prefix_trace_frame: Option<Result<u64, ()>>,
    pub(crate) visual_prefix_trace_iteration: Option<Result<usize, ()>>,
    pub(crate) visual_prefix_trace_trial: Option<Result<usize, ()>>,
    pub(crate) visual_chain_trace: bool,
    pub(crate) landmark_backsub_probe: Option<PathBuf>,
    pub(crate) landmark_backsub_probe_tracks: Option<Vec<u64>>,
    pub(crate) diagnostic_prior_input: Option<PathBuf>,
    pub(crate) q2_abs_hb: Option<PathBuf>,
    /// JSONL path for the MargData-boundary prior/FEJ sidecar.  This is
    /// intentionally separate from `marg_trace`: the latter records the
    /// reduced Q2 transition, while this path joins pre/post PriorData with
    /// the exact pre-shift schema-4 frame tables.
    pub(crate) margdata_sidecar: Option<PathBuf>,
    /// JSONL path for a frame-boundary store-count/eviction sidecar.  This is
    /// an observation-only hook and must not force retained solver payloads.
    pub(crate) lifecycle_trace: Option<PathBuf>,
    /// JSONL path for the exact pre-marginalization Q2 input boundary.  This
    /// is diagnostic-only and records the source observation slots that are
    /// retained as native null/zero rows after the AOM suffix is truncated.
    pub(crate) q2_boundary_trace: Option<PathBuf>,
    /// JSONL path for the opt-in f32 rig-triangulation attempt trace.  This
    /// key is deliberately diagnostic (rather than a lean metadata suffix),
    /// because it captures DLT/SVD intermediates and candidate history.
    pub(crate) triangulation_trace: Option<PathBuf>,
    pub(crate) triangulation_trace_tracks: Option<Vec<u64>>,
    /// Opt-in lossless ordered-factor oracle.  This is intentionally separate
    /// from the human-readable detail trace: it captures the immutable factor
    /// vector and reducer parity at one selected event.
    pub(crate) full70_factor_oracle: Option<PathBuf>,
    pub(crate) full70_factor_oracle_frame: Option<Result<u64, ()>>,
    pub(crate) full70_factor_oracle_iteration: Option<Result<usize, ()>>,
    pub(crate) full70_factor_oracle_trial: Option<Result<usize, ()>>,
    pub(crate) full70_factor_oracle_phase: Option<String>,
}

const DIAGNOSTIC_VALUE_KEYS: &[&str] = &[
    "VISLOC_BASALT_DETAIL_ITERATIONS",
    "VISLOC_BASALT_DETAIL_TRACE",
    "VISLOC_BASALT_DETAIL_FRAME",
    "VISLOC_BASALT_SOLVER_FRONTIER_TRACE",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_FACTOR_INPUTS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_UNWHITENED_F64",
    "VISLOC_BASALT_DIAGNOSTIC_FRAME",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_LINK",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATION",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATIONS",
    "VISLOC_BASALT_DIAGNOSTIC_IMU_HB",
    "VISLOC_BASALT_DIAGNOSTIC_ITERATION",
    "VISLOC_BASALT_DIAGNOSTIC_NORMAL_STAGES",
    "VISLOC_BASALT_STATE_UPDATE_TRACE_FRAME",
    "VISLOC_BASALT_MARG_TRACE",
    "VISLOC_BASALT_MARG_TRACE_EVENT",
    "VISLOC_BASALT_SQRT_TRACE",
    "VISLOC_BASALT_SQRT_FOCUS_K12",
    "VISLOC_BASALT_SQRT_DUMP_WORKSPACE_ALL",
    "VISLOC_BASALT_SQRT_DUMP_TAIL",
    "VISLOC_BASALT_SQRT_DUMP_QJ",
    "VISLOC_BASALT_SQRT_TRACE_INPUT",
    "VISLOC_BASALT_VISUAL_PREFIX_TRACE",
    "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID",
    "VISLOC_BASALT_VISUAL_PREFIX_ITERATION",
    "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
    "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
    "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
    "VISLOC_BASALT_VISUAL_CHAIN_TRACE",
    "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE",
    "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE_TRACKS",
    "VISLOC_BASALT_DIAGNOSTIC_PRIOR_INPUT",
    "VISLOC_BASALT_Q2_ABS_HB",
    "VISLOC_BASALT_MARGDATA_SIDECAR",
    "VISLOC_BASALT_LIFECYCLE_TRACE",
    "VISLOC_BASALT_Q2_BOUNDARY_TRACE",
    "VISLOC_BASALT_TRIANGULATION_TRACE",
    "VISLOC_BASALT_TRIANGULATION_TRACE_TRACKS",
    "VISLOC_BASALT_FULL70_FACTOR_ORACLE",
    "VISLOC_BASALT_FULL70_FRAME_ID",
    "VISLOC_BASALT_FULL70_ITERATION",
    "VISLOC_BASALT_FULL70_TRIAL",
    "VISLOC_BASALT_FULL70_PHASE",
    "VISLOC_BASALT_FULL70_EXECUTABLE_SHA256",
    "VISLOC_BASALT_FULL70_SOURCE_SHA256",
    "VISLOC_BASALT_FULL70_CONFIG_SHA256",
    "VISLOC_BASALT_FULL70_CALIBRATION_SHA256",
    "VISLOC_BASALT_FULL70_INPUT_SHA256",
    "VISLOC_BASALT_FULL70_FEATURES",
    "VISLOC_BASALT_FULL70_TARGET",
    "VISLOC_BASALT_FULL70_DIRTY_SCOPE_SHA256",
];

fn cached_env_key(key: &OsStr) -> Option<&'static str> {
    let key = key.to_string_lossy();
    DIAGNOSTIC_VALUE_KEYS
        .iter()
        .copied()
        .find(|known| key.eq_ignore_ascii_case(known))
}

fn cached_path(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<PathBuf> {
    values.get(key).map(PathBuf::from)
}

fn cached_u64(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<u64> {
    values
        .get(key)
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<u64>().ok())
}

fn cached_selector_u64(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Result<u64, ()>> {
    values.get(key).map(|value| {
        let value = value.to_str().ok_or(())?;
        if value.is_empty() {
            return Err(());
        }
        value.parse::<u64>().map_err(|_| ())
    })
}

fn cached_selector_usize(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Result<usize, ()>> {
    cached_selector_u64(values, key)
        .map(|value| value.and_then(|value| usize::try_from(value).map_err(|_| ())))
}

fn cached_usize(values: &HashMap<&'static str, OsString>, key: &'static str) -> Option<usize> {
    values
        .get(key)
        .and_then(|value| value.to_str())
        .and_then(|value| value.parse::<usize>().ok())
}

fn cached_usize_filter(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Vec<usize>> {
    values.get(key).map(|value| {
        value
            .to_string_lossy()
            .split(',')
            .filter_map(|entry| entry.trim().parse::<usize>().ok())
            .collect()
    })
}

fn cached_u64_filter(
    values: &HashMap<&'static str, OsString>,
    key: &'static str,
) -> Option<Vec<u64>> {
    values.get(key).map(|value| {
        value
            .to_string_lossy()
            .split(',')
            .filter_map(|entry| entry.trim().parse::<u64>().ok())
            .collect()
    })
}

fn diagnostic_env_snapshot_from_current() -> DiagnosticEnvSnapshot {
    let mut values = HashMap::new();
    let mut active = false;
    for (key, value) in env::vars_os() {
        active |= diagnostic_env_key_requires_fallback(&key);
        if let Some(canonical) = cached_env_key(&key) {
            // Windows environment names are case-insensitive.  If an unusual
            // environment contains the same key twice, the first snapshot
            // value is stable and no hot-path lookup is reintroduced.
            values.entry(canonical).or_insert(value);
        }
    }
    DiagnosticEnvSnapshot {
        active,
        detail_iterations: values.contains_key("VISLOC_BASALT_DETAIL_ITERATIONS"),
        detail_trace: cached_path(&values, "VISLOC_BASALT_DETAIL_TRACE"),
        detail_frame: cached_u64(&values, "VISLOC_BASALT_DETAIL_FRAME"),
        solver_frontier_trace: cached_path(&values, "VISLOC_BASALT_SOLVER_FRONTIER_TRACE"),
        diagnostic_imu_rows: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS"),
        diagnostic_imu_factor_inputs: cached_path(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_FACTOR_INPUTS",
        ),
        diagnostic_imu_unwhitened_f64: cached_path(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_UNWHITENED_F64",
        ),
        diagnostic_frame: cached_u64(&values, "VISLOC_BASALT_DIAGNOSTIC_FRAME"),
        diagnostic_imu_link: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_LINK")
            .unwrap_or(0),
        diagnostic_imu_iteration: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATION"),
        diagnostic_imu_iterations: cached_usize_filter(
            &values,
            "VISLOC_BASALT_DIAGNOSTIC_IMU_ITERATIONS",
        ),
        diagnostic_imu_hb: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_IMU_HB"),
        diagnostic_iteration: cached_usize(&values, "VISLOC_BASALT_DIAGNOSTIC_ITERATION"),
        diagnostic_normal_stages: values.contains_key("VISLOC_BASALT_DIAGNOSTIC_NORMAL_STAGES"),
        state_update_trace_frame: cached_u64(&values, "VISLOC_BASALT_STATE_UPDATE_TRACE_FRAME"),
        marg_trace: cached_path(&values, "VISLOC_BASALT_MARG_TRACE"),
        marg_trace_event: cached_usize(&values, "VISLOC_BASALT_MARG_TRACE_EVENT"),
        sqrt_trace: values.contains_key("VISLOC_BASALT_SQRT_TRACE"),
        sqrt_focus_k12: values.contains_key("VISLOC_BASALT_SQRT_FOCUS_K12"),
        sqrt_dump_workspace_all: values.contains_key("VISLOC_BASALT_SQRT_DUMP_WORKSPACE_ALL"),
        sqrt_dump_tail: values.contains_key("VISLOC_BASALT_SQRT_DUMP_TAIL"),
        sqrt_dump_qj: cached_path(&values, "VISLOC_BASALT_SQRT_DUMP_QJ"),
        visual_prefix_trace: cached_path(&values, "VISLOC_BASALT_VISUAL_PREFIX_TRACE"),
        visual_factor_identity_trace: cached_path(
            &values,
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
        ),
        visual_factor_identity_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
        ),
        visual_prefix_trace_frame: cached_selector_u64(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID",
        ),
        visual_prefix_trace_iteration: cached_selector_usize(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_ITERATION",
        ),
        visual_prefix_trace_trial: cached_selector_usize(
            &values,
            "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
        ),
        visual_chain_trace: values.contains_key("VISLOC_BASALT_VISUAL_CHAIN_TRACE"),
        landmark_backsub_probe: cached_path(&values, "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE"),
        landmark_backsub_probe_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_LANDMARK_BACKSUB_PROBE_TRACKS",
        ),
        diagnostic_prior_input: cached_path(&values, "VISLOC_BASALT_DIAGNOSTIC_PRIOR_INPUT"),
        q2_abs_hb: cached_path(&values, "VISLOC_BASALT_Q2_ABS_HB"),
        margdata_sidecar: cached_path(&values, "VISLOC_BASALT_MARGDATA_SIDECAR"),
        lifecycle_trace: cached_path(&values, "VISLOC_BASALT_LIFECYCLE_TRACE"),
        q2_boundary_trace: cached_path(&values, "VISLOC_BASALT_Q2_BOUNDARY_TRACE"),
        triangulation_trace: cached_path(&values, "VISLOC_BASALT_TRIANGULATION_TRACE"),
        triangulation_trace_tracks: cached_u64_filter(
            &values,
            "VISLOC_BASALT_TRIANGULATION_TRACE_TRACKS",
        ),
        full70_factor_oracle: cached_path(&values, "VISLOC_BASALT_FULL70_FACTOR_ORACLE"),
        full70_factor_oracle_frame: cached_selector_u64(&values, "VISLOC_BASALT_FULL70_FRAME_ID"),
        full70_factor_oracle_iteration: cached_selector_usize(
            &values,
            "VISLOC_BASALT_FULL70_ITERATION",
        ),
        full70_factor_oracle_trial: cached_selector_usize(&values, "VISLOC_BASALT_FULL70_TRIAL"),
        full70_factor_oracle_phase: values
            .get("VISLOC_BASALT_FULL70_PHASE")
            .and_then(|value| value.to_str())
            .map(str::to_owned),
    }
}

// The diagnostic mode is a process-lifetime policy.  Snapshot the environment
// at first use so hot LM/estimator calls do not repeatedly enumerate the
// process environment or parse selector values.  Callers must set diagnostic
// variables before the first window/estimator operation; changing them later
// cannot change the selected path.  The pure key predicate below remains
// separate so tests can exercise the fail-closed allowlist without mutating
// process environment.
static DIAGNOSTIC_ENV_SNAPSHOT: OnceLock<DiagnosticEnvSnapshot> = OnceLock::new();

// Scope the f64 IMU observer to the explicit link walk in
// `initial_window_diagnostics`.  The diagnostic prepass first calls
// `WindowProblem::linearize`, which also reaches `imu_factor_with_trace`;
// that call must not be mistaken for the authoritative one-link capture.
// A thread-local flag keeps this boundary local even if an estimator is
// driven from more than one worker thread.
thread_local! {
    static INITIAL_IMU_F64_CAPTURE_SCOPE: Cell<bool> = const { Cell::new(false) };
    static INITIAL_IMU_F64_CAPTURE_LINK_INDEX: Cell<Option<usize>> = const { Cell::new(None) };
}

struct InitialImuF64CaptureScope {
    previous: bool,
    previous_link_index: Option<usize>,
}

impl InitialImuF64CaptureScope {
    fn enter() -> Self {
        let previous = INITIAL_IMU_F64_CAPTURE_SCOPE.with(|active| {
            let previous = active.get();
            active.set(true);
            previous
        });
        let previous_link_index = INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|link_index| {
            let previous = link_index.get();
            link_index.set(None);
            previous
        });
        Self {
            previous,
            previous_link_index,
        }
    }
}

impl Drop for InitialImuF64CaptureScope {
    fn drop(&mut self) {
        INITIAL_IMU_F64_CAPTURE_SCOPE.with(|active| active.set(self.previous));
        INITIAL_IMU_F64_CAPTURE_LINK_INDEX
            .with(|link_index| link_index.set(self.previous_link_index));
    }
}

fn initial_imu_f64_capture_scope_active() -> bool {
    INITIAL_IMU_F64_CAPTURE_SCOPE.with(Cell::get)
}

struct InitialImuF64CaptureLink {
    previous: Option<usize>,
}

impl InitialImuF64CaptureLink {
    fn set(link_index: usize) -> Self {
        let previous = INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|current| {
            let previous = current.get();
            current.set(Some(link_index));
            previous
        });
        Self { previous }
    }
}

impl Drop for InitialImuF64CaptureLink {
    fn drop(&mut self) {
        INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(|current| current.set(self.previous));
    }
}

fn initial_imu_f64_capture_link_index() -> Option<usize> {
    INITIAL_IMU_F64_CAPTURE_LINK_INDEX.with(Cell::get)
}

// The native prior transition captures the newly reduced square-root prior
// immediately before its f32 RHS shift.  Keep that pre-shift value in a
// process-local diagnostic handoff so the estimator can join it with the
// post-shift WindowPrior without adding fields to the solver state or packet.
static DIAGNOSTIC_PRIOR_PRE: OnceLock<Mutex<Option<WindowPrior>>> = OnceLock::new();

fn remember_diagnostic_prior_pre(prior: &WindowPrior) {
    if diagnostic_env_snapshot().margdata_sidecar.is_none() {
        return;
    }
    if let Ok(mut slot) = DIAGNOSTIC_PRIOR_PRE.get_or_init(|| Mutex::new(None)).lock() {
        *slot = Some(prior.clone());
    }
}

pub(crate) fn take_diagnostic_prior_pre() -> Option<WindowPrior> {
    DIAGNOSTIC_PRIOR_PRE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|mut slot| slot.take())
}

fn diagnostic_env_key_requires_fallback(key: &std::ffi::OsStr) -> bool {
    let key = key.to_string_lossy();
    let prefix = "VISLOC_BASALT_";
    let Some(candidate_prefix) = key.get(..prefix.len()) else {
        return false;
    };
    if !candidate_prefix.eq_ignore_ascii_case(prefix) {
        return false;
    }
    let Some(suffix) = key.get(prefix.len()..) else {
        return false;
    };
    !NON_DIAGNOSTIC_ENV_SUFFIXES
        .iter()
        .any(|allowed| suffix.eq_ignore_ascii_case(allowed))
}

fn diagnostic_env_active_for_keys(keys: impl IntoIterator<Item = std::ffi::OsString>) -> bool {
    keys.into_iter()
        .any(|key| diagnostic_env_key_requires_fallback(&key))
}

/// Returns the process-lifetime diagnostic/probe environment snapshot.
///
/// Canonical harness metadata and source-confirmed fixture paths are
/// deliberately ignored; every other `VISLOC_BASALT_*` key opts back into the
/// diagnostic path so a newly added probe cannot silently lose its boundary.
/// The snapshot is taken on first use, before the first window/estimator
/// operation, and is intentionally not refreshed after that point.
pub(crate) fn diagnostic_env_active() -> bool {
    diagnostic_env_snapshot().active
}

pub(crate) fn diagnostic_env_snapshot() -> &'static DiagnosticEnvSnapshot {
    DIAGNOSTIC_ENV_SNAPSHOT.get_or_init(diagnostic_env_snapshot_from_current)
}

fn iteration_trace_frame_matches(frame_id: u64) -> bool {
    diagnostic_env_snapshot()
        .detail_frame
        .is_none_or(|target| target == frame_id)
}

fn json_number(value: f64) -> JsonValue {
    serde_json::Number::from_f64(value)
        .map(JsonValue::Number)
        .unwrap_or(JsonValue::Null)
}

fn json_vector(values: impl IntoIterator<Item = f64>) -> JsonValue {
    JsonValue::Array(values.into_iter().map(json_number).collect())
}

fn json_matrix(matrix: &DMatrix<f64>) -> JsonValue {
    JsonValue::Array(
        (0..matrix.nrows())
            .map(|row| json_vector((0..matrix.ncols()).map(|column| matrix[(row, column)])))
            .collect(),
    )
}

fn json_visual_chain(chain: &VisualChainF32) -> JsonValue {
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector = |values: &[f32]| {
        JsonValue::Array(
            values
                .iter()
                .copied()
                .map(bits)
                .map(JsonValue::String)
                .collect(),
        )
    };
    json!({
        "target_camera_from_imu_f32_bits": vector(&chain.target_camera_from_imu),
        "target_imu_from_anchor_imu_f32_bits": vector(&chain.target_imu_from_anchor_imu),
        "target_camera_from_anchor_imu_f32_bits": vector(&chain.target_camera_from_anchor_imu),
        "target_camera_from_anchor_imu_fej_f32_bits": vector(&chain.target_camera_from_anchor_imu_fej),
        "target_camera_from_anchor_camera_f32_bits": vector(&chain.target_camera_from_anchor_camera),
        "target_camera_matrix_f32_bits_column_major": vector(&chain.target_camera_matrix),
        "relative_wrt_anchor_f32_bits_column_major": vector(&chain.relative_wrt_anchor),
        "point4_target_f32_bits": vector(&chain.point4_target),
        "point_target_f32_bits": vector(&chain.point_target),
        "projection_f32_bits": vector(&chain.projection),
        "projection_jacobian_f32_bits": vector(&chain.projection_jacobian),
    })
}

fn json_se3(pose: &SE3) -> JsonValue {
    let quaternion = pose.rotation.quaternion();
    let rotation = pose.rotation.to_rotation_matrix();
    let mut homogeneous = Matrix4::identity();
    homogeneous
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(rotation.matrix());
    homogeneous
        .fixed_view_mut::<3, 1>(0, 3)
        .copy_from(&pose.translation);
    json!({
        "translation": json_vector(pose.translation.iter().copied()),
        "quaternion_xyzw": json_vector([
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w,
        ]),
        "matrix": json_matrix(&DMatrix::from_iterator(4, 4, homogeneous.iter().copied())),
    })
}

fn block_timestamp(problem: &WindowProblem, index: usize) -> Option<i64> {
    if index < problem.poses.len() {
        problem.poses.get(index).map(|pose| pose.timestamp_ns)
    } else {
        problem
            .states
            .get(index.checked_sub(problem.poses.len())?)
            .map(|state| state.timestamp_ns)
    }
}

fn block_frame_id_for_timestamp(problem: &WindowProblem, index: usize) -> Option<u64> {
    problem.block_frame_id(index)
}

fn json_aom(problem: &WindowProblem) -> JsonValue {
    let mut order = Vec::with_capacity(problem.poses.len() + problem.states.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        order.push(json!({
            "ordinal": index,
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "kind": "pose",
            "offset": problem.pose_offset(index),
            "dof": POSE_DOF,
        }));
    }
    for (index, state) in problem.states.iter().enumerate() {
        order.push(json!({
            "ordinal": problem.poses.len() + index,
            "frame_id": state.frame_id,
            "timestamp_ns": state.timestamp_ns,
            "kind": "state",
            "offset": problem.state_offset(index),
            "dof": NAV_STATE_DOF,
        }));
    }
    JsonValue::Array(order)
}

fn json_trial_visual_costs(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let mut records = Vec::new();
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        let Some(audit) = problem.visual_factor_with_audit(state, index) else {
            continue;
        };
        for observation in &audit.observations {
            let factor = &observation.factor;
            records.push(json!({
                "track_id": landmark.track_id,
                "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
                "host_cam": landmark.anchor_camera_id,
                "target_frame_id": block_frame_id_for_timestamp(problem, observation.observation.state_index),
                "target_cam": observation.observation.camera_id,
                "raw_residual": [factor.raw_residual.x, factor.raw_residual.y],
                "huber_weight": factor.huber_weight,
                "objective": factor.objective_cost,
                "visual_chain": factor.debug_chain.as_ref().map(json_visual_chain),
            }));
        }
    }
    json!({"state_boundary": "candidate_after_step", "observations": records,
        "native_host_order": problem.trial_host_order,
        "trial_geometry": json_trial_native_geometry(problem, state),
        "trial_imu": json_trial_native_imu(problem, state)})
}

fn json_trial_native_imu(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    if problem.scalar_mode != ScalarMode::UpstreamF32 {
        return JsonValue::Null;
    }
    let evaluate = || -> Option<JsonValue> {
        let mut links = problem.imu_links.iter().collect::<Vec<_>>();
        links.sort_by_key(|link| problem.states.get(link.from_index).map(|s| s.timestamp_ns));
        let mut records = Vec::with_capacity(links.len());
        let mut total = 0.0_f32;
        let words = |values: &[f32]| {
            values
                .iter()
                .map(|v| format!("{:08x}", v.to_bits()))
                .collect::<Vec<_>>()
        };
        for link in links {
            if link.delta.delta_time == 0.0 {
                continue;
            }
            let from_state = problem.states.get(link.from_index)?;
            let to_state = problem.states.get(link.to_index)?;
            let from = problem.block_nav(state, problem.poses.len() + link.from_index)?;
            let to = problem.block_nav(state, problem.poses.len() + link.to_index)?;
            let raw = crate::imu::trial_preintegration_residual_f32(
                &from,
                &to,
                &link.delta,
                problem.gravity_world,
            );
            let sqrt =
                crate::imu::sqrt_information_f32(&link.delta.covariance.map(|v| v as f32)).ok()?;
            let (inverse, weighted, cost) = crate::imu::trial_imu_quadratic_stages_f32(&sqrt, &raw);
            let before = total;
            total += cost;
            records.push(json!({"from_timestamp_ns":from_state.timestamp_ns,
                "to_timestamp_ns":to_state.timestamp_ns,"raw":words(raw.as_slice()),
                "sqrt_information":words(sqrt.as_slice()),"covariance_inverse":words(inverse.as_slice()),
                "half_residual_covariance_product":words(weighted.as_slice()),
                "objective":format!("{:08x}",cost.to_bits()),
                "accumulator_before":format!("{:08x}",before.to_bits())}));
        }
        Some(
            json!({"status":"ok","links":records,"cost_f32_bits":format!("{:08x}",total.to_bits())}),
        )
    };
    evaluate().unwrap_or_else(|| json!({"status":"invalid_topology_or_covariance"}))
}

// Candidate-only audit of computeError's geometry. This deliberately does not
// change LM decisions until the complete trial objective is integrated.
fn json_trial_native_geometry(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    if problem.scalar_mode != ScalarMode::UpstreamF32 {
        return JsonValue::Null;
    }
    let evaluate = || -> Option<JsonValue> {
        let ranks = problem
            .trial_host_order
            .iter()
            .enumerate()
            .map(|(rank, &key)| (key, rank))
            .collect::<BTreeMap<_, _>>();
        let mut ordered = BTreeMap::new();
        for (index, landmark) in problem.landmarks.iter().enumerate() {
            let host = (
                block_timestamp(problem, landmark.anchor_state_index)? as u64,
                landmark.anchor_camera_id,
            );
            let rank = *ranks.get(&host)?;
            for observation in &landmark.observations {
                let target = (
                    block_timestamp(problem, observation.state_index)? as u64,
                    observation.camera_id,
                );
                if ordered
                    .insert((rank, target, landmark.track_id), (index, observation))
                    .is_some()
                {
                    return None;
                }
            }
        }
        let mut records = Vec::with_capacity(ordered.len());
        let mut cost = 0.0_f32;
        for ((rank, target, track), (index, observation)) in ordered {
            let landmark = &problem.landmarks[index];
            let host = problem.trial_host_order[rank];
            let host_nav = problem.block_nav_visual_current(state, landmark.anchor_state_index)?;
            let target_nav = problem.block_nav_visual_current(state, observation.state_index)?;
            let matrix = super::aom::upstream_trial_transform_f32(
                &host_nav.imu_to_world,
                &target_nav.imu_to_world,
                &problem.camera_to_imu(host.1)?,
                &problem.camera_to_imu(target.1)?,
                host == target,
            );
            let result = super::aom::upstream_trial_observation_f32(
                problem.camera_model(target.1)?,
                matrix,
                Vector3::new(
                    landmark.direction.xy.x as f32,
                    landmark.direction.xy.y as f32,
                    landmark.inverse_distance as f32,
                ),
                nalgebra::Vector2::new(observation.pixel.x as f32, observation.pixel.y as f32),
                FactorConfig::default(),
            );
            let raw = result.map(|(raw, _)| {
                [
                    format!("{:08x}", raw.x.to_bits()),
                    format!("{:08x}", raw.y.to_bits()),
                ]
            });
            if let Some((_, objective)) = result {
                cost += objective;
            }
            records.push(json!({"track_id":track,"host_timestamp_ns":host.0,"host_cam":host.1,
                "target_timestamp_ns":target.0,"target_cam":target.1,"valid":result.is_some(),
                "raw_residual_f32_bits":raw,
                "objective_f32_bits":result.map(|(_, v)| format!("{:08x}",v.to_bits())),
                "accumulated_f32_bits":format!("{:08x}",cost.to_bits()),
                "matrix_f32_bits_column_major":matrix.iter().map(|v| format!("{:08x}",v.to_bits())).collect::<Vec<_>>()}));
        }
        Some(
            json!({"status":"ok","observations":records,"cost_f32_bits":format!("{:08x}",cost.to_bits())}),
        )
    };
    evaluate().unwrap_or_else(|| json!({"status":"missing_or_duplicate_topology"}))
}

fn json_block_values(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let mut poses = Vec::with_capacity(problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        let decoded = problem
            .block_pose(state, index)
            .unwrap_or_else(SE3::identity);
        poses.push(json!({
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "kind": "pose",
            "pose": json_se3(&decoded),
        }));
    }
    let mut states = Vec::with_capacity(problem.states.len());
    for (index, window_state) in problem.states.iter().enumerate() {
        let decoded = problem
            .block_nav(state, problem.poses.len() + index)
            .unwrap_or_default();
        states.push(json!({
            "frame_id": window_state.frame_id,
            "timestamp_ns": window_state.timestamp_ns,
            "kind": "state",
            "pose": json_se3(&decoded.imu_to_world),
            "velocity": json_vector(decoded.velocity_world_m_s.iter().copied()),
            "bias_gyro": json_vector(decoded.gyro_bias_rad_s.iter().copied()),
            "bias_accel": json_vector(decoded.accel_bias_m_s2.iter().copied()),
        }));
    }
    json!({"poses": poses, "states": states})
}

fn json_landmark_values(problem: &WindowProblem) -> JsonValue {
    let mut values = problem.landmarks.iter().collect::<Vec<_>>();
    values.sort_by_key(|landmark| landmark.track_id);
    JsonValue::Array(
        values
            .into_iter()
            .map(|landmark| {
                let observations = landmark
                    .observations
                    .iter()
                    .map(|observation| {
                        json!({
                            "target_frame_id": block_frame_id_for_timestamp(problem, observation.state_index),
                            "target_timestamp_ns": block_timestamp(problem, observation.state_index),
                            "target_cam": observation.camera_id,
                            "pixel": json_vector([observation.pixel.x, observation.pixel.y]),
                        })
                    })
                    .collect::<Vec<_>>();
                json!({
                    "track_id": landmark.track_id,
                    "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
                    "host_timestamp_ns": block_timestamp(problem, landmark.anchor_state_index),
                    "host_cam": landmark.anchor_camera_id,
                    "direction": json_vector([
                        landmark.direction.xy.x,
                        landmark.direction.xy.y,
                    ]),
                    "rho": json_number(landmark.inverse_distance),
                    "observations": observations,
                })
            })
            .collect(),
    )
}

fn visual_factor_records(
    problem: &WindowProblem,
    state: &DVector<f64>,
    linearization: &LmLinearization,
) -> (JsonValue, Vec<(usize, usize)>) {
    let prior_factor_count = if problem.prior.is_some() { 1 } else { 0 };
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(linearization.factors.len().saturating_sub(prefix));
    let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
    let mut factor_index = prefix;
    let mut reduced_row = 0usize;
    let mut records = Vec::new();
    let mut spans = Vec::new();
    for (landmark_index, landmark) in problem.landmarks.iter().enumerate() {
        let Some(audit) = problem.visual_factor_with_audit(state, landmark_index) else {
            continue;
        };
        if factor_index >= visual_end {
            break;
        }
        let factor = &linearization.factors[factor_index];
        factor_index += 1;
        let (reduced_rows, reduced_rhs, rank) =
            project_landmark_factor(factor, 1e-10, problem.scalar_mode);
        let span = (reduced_row, reduced_rows.nrows());
        reduced_row += reduced_rows.nrows();
        spans.push(span);
        let observations = audit
            .observations
            .iter()
            .map(|observation| {
                let factor = &observation.factor;
                let mut value = json!({
                    "target_frame_id": block_frame_id_for_timestamp(problem, observation.observation.state_index),
                    "target_timestamp_ns": block_timestamp(problem, observation.observation.state_index),
                    "target_cam": observation.observation.camera_id,
                    "pixel": json_vector([
                        observation.observation.pixel.x,
                        observation.observation.pixel.y,
                    ]),
                    "projection": json_vector([factor.projection.x, factor.projection.y]),
                    "raw_residual": json_vector([factor.raw_residual.x, factor.raw_residual.y]),
                    "residual": json_vector([factor.residual.x, factor.residual.y]),
                    "huber_weight": json_number(factor.huber_weight),
                    "sqrt_weight": json_number(factor.sqrt_weight),
                    "objective": json_number(factor.objective_cost),
                    "jp_anchor": json_matrix(&DMatrix::from_iterator(2, 6, factor.anchor_pose_jacobian.iter().copied())),
                    "jp_target": json_matrix(&DMatrix::from_iterator(2, 6, factor.target_pose_jacobian.iter().copied())),
                    "jl": json_matrix(&DMatrix::from_iterator(2, 3, factor.landmark_jacobian.iter().copied())),
                });
                if let Some(chain) = factor.debug_chain.as_ref() {
                    value["visual_chain"] = json_visual_chain(chain);
                }
                value
            })
            .collect::<Vec<_>>();
        records.push(json!({
            "track_id": landmark.track_id,
            "host_frame_id": problem.block_frame_id(landmark.anchor_state_index),
            "host_timestamp_ns": block_timestamp(problem, landmark.anchor_state_index),
            "host_cam": landmark.anchor_camera_id,
            "direction": json_vector([landmark.direction.xy.x, landmark.direction.xy.y]),
            "rho": json_number(landmark.inverse_distance),
            "row_span": [span.0, span.1],
            "rank": rank,
            "observations": observations,
            "state_jacobian": json_matrix(&factor.state_jacobian),
            "landmark_jacobian": json_matrix(&factor.landmark_jacobian),
            "residual": json_vector(factor.residual.iter().copied()),
            "reduced_rows": json_matrix(&reduced_rows),
            "reduced_rhs": json_vector(reduced_rhs.iter().copied()),
        }));
    }
    records.sort_by_key(|record| record["track_id"].as_u64().unwrap_or_default());
    (JsonValue::Array(records), spans)
}

fn project_landmark_factor(
    factor: &WhitenedFactorRowStack,
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> (DMatrix<f64>, DVector<f64>, usize) {
    if scalar_mode == ScalarMode::UpstreamF32 {
        let (jacobian, residual, rank) = landmark_nullspace_projection_f32(factor, tolerance);
        let jacobian = DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, column| {
            jacobian[(row, column)] as f64
        });
        let residual =
            DVector::from_iterator(residual.len(), residual.iter().copied().map(f64::from));
        (jacobian, residual, rank)
    } else {
        landmark_nullspace_projection(factor, tolerance)
    }
}

fn global_reduced_rows(
    linearization: &LmLinearization,
    scalar_mode: ScalarMode,
) -> (DMatrix<f64>, DVector<f64>) {
    let state_dof = linearization
        .factors
        .first()
        .map_or(0, |factor| factor.state_jacobian.ncols());
    let projected = linearization
        .factors
        .iter()
        .map(|factor| project_landmark_factor(factor, 1e-10, scalar_mode))
        .collect::<Vec<_>>();
    let rows = projected
        .iter()
        .map(|(jacobian, _, _)| jacobian.nrows())
        .sum();
    let mut jacobian = DMatrix::zeros(rows, state_dof);
    let mut rhs = DVector::zeros(rows);
    let mut row = 0;
    for (factor_jacobian, factor_rhs, _) in projected {
        let count = factor_jacobian.nrows();
        jacobian
            .view_mut((row, 0), (count, state_dof))
            .copy_from(&factor_jacobian);
        rhs.rows_mut(row, count).copy_from(&factor_rhs);
        row += count;
    }
    (jacobian, rhs)
}

/// Return the source-category objective at an opt-in iteration snapshot.
///
/// The normal solver keeps these values only as the scalar LM objective.  The
/// detail trace needs the four terms separately so a native logger can be
/// compared at the same trial boundary.  Re-linearizing the display clone is
/// intentional: for a trial snapshot it includes the candidate navigation
/// and back-substituted landmark state, rather than repeating the iteration's
/// base-state cost for every trial.
fn iteration_category_costs(problem: &WindowProblem, state: &DVector<f64>) -> JsonValue {
    let Some(linearization) = problem.linearize(state).ok() else {
        return JsonValue::Null;
    };
    let prior_factor_count = usize::from(problem.prior.is_some());
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(linearization.factors.len().saturating_sub(prefix));
    let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
    let sum = |factors: &[WhitenedFactorRowStack]| {
        if problem.scalar_mode == ScalarMode::UpstreamF32 {
            factors.iter().fold(0.0_f32, |total, factor| {
                total + factor.objective_cost as f32
            }) as f64
        } else {
            factors
                .iter()
                .map(|factor| factor.objective_cost)
                .sum::<f64>()
        }
    };
    let prior = if prefix != 0 {
        problem.prior.as_ref().map_or_else(
            || sum(&linearization.factors[..prefix]),
            |prior| problem.marginal_prior_reduced_cost(state, prior),
        )
    } else {
        0.0
    };
    let visual = sum(&linearization.factors[prefix..visual_end]);
    let (imu, bias) = if problem.scalar_mode == ScalarMode::UpstreamF32 {
        let mut imu = 0.0_f32;
        let mut bias = 0.0_f32;
        for pair in linearization.factors[visual_end..].chunks(2) {
            if let Some(factor) = pair.first() {
                imu += factor.objective_cost as f32;
            }
            if let Some(factor) = pair.get(1) {
                bias += factor.objective_cost as f32;
            }
        }
        (imu as f64, bias as f64)
    } else {
        let mut imu = 0.0_f64;
        let mut bias = 0.0_f64;
        for pair in linearization.factors[visual_end..].chunks(2) {
            if let Some(factor) = pair.first() {
                imu += factor.objective_cost;
            }
            if let Some(factor) = pair.get(1) {
                bias += factor.objective_cost;
            }
        }
        (imu, bias)
    };
    json!({
        "prior": json_number(prior),
        "visual": json_number(visual),
        "imu": json_number(imu),
        "bias": json_number(bias),
        "total": json_number(linearization.cost),
        "factor_prefix": prefix,
        "visual_factor_end": visual_end,
        "scalar_mode": if problem.scalar_mode == ScalarMode::UpstreamF32 { "f32" } else { "f64" },
    })
}

fn emit_iteration_header() {
    let Some(path) = iteration_trace_path() else {
        return;
    };
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_some() {
        return;
    }
    let _ = INITIALIZED.set(());
    {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&path, b"");
    }
    let header = json!({
        "schema": "basalt.vio_iteration.v1",
        "record": "header",
        "source": "rust",
        "scalar_type": "f64",
        "upstream_reference": "float32",
        "cast_points": [
            "pipelines/basalt/src/vio/estimator.rs::process_impl calibrated IMU remains f64",
            "pipelines/basalt/src/vio/window.rs::anchored_visual_reprojection_factor f64",
            "upstream src/vio.cpp::ImuData<double> -> float estimator queue"
        ],
        "symbols": [
            "visloc_basalt::vio::aom::solve_lm",
            "visloc_basalt::vio::window::WindowProblem::linearize",
            "visloc_basalt::vio::window::WindowProblem::landmark_steps"
        ]
    });
    if let Ok(line) = serde_json::to_string(&header) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn write_iteration_record(record: &JsonValue) {
    let Some(path) = iteration_trace_path() else {
        return;
    };
    emit_iteration_header();
    let Ok(line) = serde_json::to_string(record) else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{line}");
    }
}

/// Serialize the source-order normal-system checkpoints when the f32 reducer
/// was entered with an explicit parity diagnostic enabled.  Every operand in
/// `ReducedNormalSystem::diagnostic_stages` is already an f32 value widened
/// to f64; the explicit casts/adds below therefore preserve the f32 rounding
/// boundary while keeping this helper observation-only.
fn diagnostic_normal_system_stages(
    reduced: &ReducedNormalSystem,
    bits: &impl Fn(f32) -> String,
) -> JsonValue {
    let Some(stages) = reduced.diagnostic_stages.as_ref() else {
        return JsonValue::Null;
    };
    let matrix_bits = |matrix: &DMatrix<f64>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value as f32)).collect::<Vec<_>>(),
        })
    };
    let vector_bits = |vector: &DVector<f64>| {
        json!(vector
            .iter()
            .map(|value| bits(*value as f32))
            .collect::<Vec<_>>())
    };
    // Basalt's `get_dense_H_b` checkpoint is captured before the LM solver
    // forms `H_copy = H + Hdiag_lambda`.  The pinned MH01 driver also leaves
    // `setPoseDamping(lambda)` disabled, so its stage-3 pose-damping hook is
    // a no-op for this direct H/b boundary.  `damping_diag` belongs to the
    // separate solver-side H_copy checkpoint and must not be folded into the
    // source normal-system stages.
    let plus_damping_h = stages.visual_imu_h.clone();
    let plus_damping_b = stages.visual_imu_b.clone();
    let mut plus_prior_h = plus_damping_h.clone();
    for col in 0..plus_prior_h.ncols() {
        for row in 0..plus_prior_h.nrows() {
            plus_prior_h[(row, col)] =
                (plus_prior_h[(row, col)] as f32 + stages.prior_h[(row, col)] as f32) as f64;
        }
    }
    let mut plus_prior_b = plus_damping_b.clone();
    for index in 0..plus_prior_b.len() {
        plus_prior_b[index] = (plus_prior_b[index] as f32 + stages.prior_b[index] as f32) as f64;
    }
    json!({
        "stage_order": [
            "visual_accumulator",
            "visual_plus_imu",
            "plus_pose_damping",
            "plus_marginal_prior"
        ],
        "visual_accumulator": {
            "h": matrix_bits(&stages.visual_h),
            "b": vector_bits(&stages.visual_b),
        },
        "visual_plus_imu": {
            "h": matrix_bits(&stages.visual_imu_h),
            "b": vector_bits(&stages.visual_imu_b),
        },
        "plus_pose_damping": {
            "h": matrix_bits(&plus_damping_h),
            "b": vector_bits(&plus_damping_b),
        },
        "plus_marginal_prior": {
            "h": matrix_bits(&plus_prior_h),
            "b": vector_bits(&plus_prior_b),
        },
        // Keep the raw square-root prior contribution alongside the
        // cumulative checkpoint.  The cumulative subtraction is not an
        // invertible f32 operation, so this sidecar is required to localize
        // a marginalization mismatch without changing the solver arithmetic.
        "marginal_prior": {
            "h": matrix_bits(&stages.prior_h),
            "b": vector_bits(&stages.prior_b),
        },
        "production_reduced": {
            "h": matrix_bits(&reduced.h),
            "b": vector_bits(&reduced.b),
        },
    })
}

/// Emit the small, bit-oriented solver frontier sidecar used to compare the
/// Rust LM boundary with Basalt's `sqrt_keypoint_vio.cpp`.  This is separate
/// from the large human-readable iteration trace and is enabled only by an
/// explicit path.  The normal estimator therefore pays no serialization or
/// state-copy cost.
fn emit_solver_frontier_sidecar(
    problem: &WindowProblem,
    applied_problem: &WindowProblem,
    event: &LmDiagnosticEvent<'_>,
) {
    let Some(path) = diagnostic_env_snapshot().solver_frontier_trace.as_ref() else {
        return;
    };
    let bits = |value: f32| format!("{:08x}", value.to_bits());
    let vector_bits = |values: &[f64]| {
        json!(values
            .iter()
            .map(|value| bits(*value as f32))
            .collect::<Vec<_>>())
    };
    let matrix_bits = |matrix: &DMatrix<f64>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value as f32)).collect::<Vec<_>>(),
        })
    };
    let matrix_bits_f32 = |matrix: &DMatrix<f32>| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "bits": matrix.iter().map(|value| bits(*value)).collect::<Vec<_>>(),
        })
    };
    let stage_systems = diagnostic_normal_system_stages(event.reduced, &bits);
    let (h_copy, h_copy_source) = if let Some(h_copy) = event.damped_h {
        // Capture the matrix owned by the active solver.  Rebuilding H_copy
        // from the f64 reduced mirror would conceal any cast/add schedule
        // difference at precisely the boundary under audit.
        (matrix_bits_f32(h_copy), "solver_damped_h_f32_direct")
    } else {
        let mut h_copy = event.reduced.h.clone();
        for index in 0..h_copy.nrows().min(h_copy.ncols()) {
            h_copy[(index, index)] =
                (event.reduced.h[(index, index)] as f32 + event.damping_diag[index] as f32) as f64;
        }
        (matrix_bits(&h_copy), "f64_reduced_plus_f32_diag_fallback")
    };
    let state_record = |state: &BasaltNavState| {
        let q = state.imu_to_world.rotation.quaternion();
        json!({
            "translation_f32_bits": state
                .imu_to_world
                .translation
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "quaternion_xyzw_f32_bits": [
                bits(q.i as f32), bits(q.j as f32), bits(q.k as f32), bits(q.w as f32)
            ],
            "velocity_f32_bits": state
                .velocity_world_m_s
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "bias_gyro_f32_bits": state
                .gyro_bias_rad_s
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
            "bias_accel_f32_bits": state
                .accel_bias_m_s2
                .iter()
                .map(|value| bits(*value as f32))
                .collect::<Vec<_>>(),
        })
    };
    let state_map = |window: &WindowProblem| {
        JsonValue::Array(
            window
                .states
                .iter()
                .map(|state| {
                    json!({
                        "frame_id": state.frame_id,
                        "timestamp_ns": state.timestamp_ns,
                        "linearized": state.linearized,
                        "linearized_state": state_record(&state.linearized_nav),
                        "current_state": state_record(&state.nav),
                        "delta_f32_bits": state
                            .linearized_delta
                            .iter()
                            .map(|value| bits(*value as f32))
                            .collect::<Vec<_>>(),
                    })
                })
                .collect(),
        )
    };
    let step = event
        .step
        .map(|value| value.iter().copied().collect::<Vec<_>>());
    let positive_step = step.as_ref().map(|values| {
        values
            .iter()
            .map(|value| -(*value as f32) as f64)
            .collect::<Vec<_>>()
    });
    let record = json!({
        "schema": "basalt.m7im15_solver_frontier.v1",
        "source": "rust",
        "frame_id": problem.states.last().map(|state| state.frame_id),
        "frame_timestamp_ns": problem.states.last().map(|state| state.timestamp_ns),
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "decision": event.decision,
        "lambda_f32_bits": bits(event.lambda as f32),
        "reduced_h": matrix_bits(&event.reduced.h),
        "reduced_b": vector_bits(event.reduced.b.as_slice()),
        "damping_diag": vector_bits(event.damping_diag.as_slice()),
        "h_copy": h_copy,
        "h_copy_source": h_copy_source,
        "normal_system_stages": stage_systems,
        "step_solved_f32_bits": step
            .as_ref()
            .map(|values| vector_bits(values))
            .unwrap_or(JsonValue::Null),
        "step_positive_equivalent_f32_bits": positive_step
            .as_ref()
            .map(|values| vector_bits(values))
            .unwrap_or(JsonValue::Null),
        "apply_before": state_map(problem),
        "apply_after": state_map(applied_problem),
        "numeric_types": { "normal_system": "f32", "state": "f64_cast_f32" },
    });
    static INITIALIZED: OnceLock<()> = OnceLock::new();
    if INITIALIZED.get().is_none() {
        let _ = INITIALIZED.set(());
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(path, b"");
    }
    if let Ok(line) = serde_json::to_string(&record) {
        if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
            let _ = writeln!(file, "{line}");
        }
    }
}

/// Build the non-factor identity needed to interpret a lossless full-factor
/// oracle.  The factor vector carries absolute state columns and observation
/// state indices; this compact context maps those indices back to the active
/// pose/state timestamps and preserves the stored square-root prior before the
/// generated FEJ-shifted factor RHS.  It is constructed only when the explicit
/// full-factor oracle path is enabled.
fn full70_oracle_window_context(problem: &WindowProblem) -> JsonValue {
    let mut state_blocks = Vec::with_capacity(problem.poses.len() + problem.states.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        state_blocks.push(json!({
            "index": index,
            "kind": "pose",
            "frame_id": pose.frame_id,
            "timestamp_ns": pose.timestamp_ns,
            "offset": problem.pose_offset(index),
            "dof": POSE_DOF,
            "is_keyframe": pose.is_keyframe,
        }));
    }
    for (index, state) in problem.states.iter().enumerate() {
        state_blocks.push(json!({
            "index": problem.poses.len() + index,
            "kind": "state",
            "frame_id": state.frame_id,
            "timestamp_ns": state.timestamp_ns,
            "offset": problem.state_offset(index),
            "dof": NAV_STATE_DOF,
            "is_keyframe": state.is_keyframe,
            "is_latest": state.is_latest,
            "linearized": state.linearized,
        }));
    }

    let prior_source = problem.prior.as_ref().map_or_else(
        || {
            json!({
                "source": if problem.anchor_point.is_some() { "anchor" } else { "absent" },
                "anchor_point_f32_bits": problem
                    .anchor_point
                    .as_ref()
                    .map(full70_f32_bits_vector)
                    .unwrap_or(JsonValue::Null),
            })
        },
        |prior| {
            json!({
                "source": "stored",
                "frame_ids": prior.frame_ids,
                "block_kinds": prior.block_kinds.iter().map(|kind| match kind {
                    WindowBlockKind::Pose => "pose",
                    WindowBlockKind::StatePose => "state_pose",
                    WindowBlockKind::State => "state",
                }).collect::<Vec<_>>(),
                "jacobian_f32_bits": full70_f32_bits_matrix(&prior.jacobian),
                "rhs_f32_bits": full70_f32_bits_vector(&prior.rhs),
                "fej_point_f32_bits": full70_f32_bits_vector(&prior.fej_point),
            })
        },
    );

    json!({
        "state_blocks": state_blocks,
        "prior_source": prior_source,
        "scalar_mode": format!("{:?}", problem.scalar_mode),
        "state_dof": problem.state_dof(),
    })
}

fn full70_oracle_event_selected(
    policy: &DiagnosticEnvSnapshot,
    frame_id: u64,
    event: &LmDiagnosticEvent<'_>,
) -> bool {
    // The artifact name and strict schema are frame-4 specific.  With no
    // explicit frame selector, skip earlier events rather than consuming the
    // one-shot slot on a smaller warm-up window.
    if policy.full70_factor_oracle_frame.is_none() && frame_id != 4 {
        return false;
    }
    let frame_matches = match policy.full70_factor_oracle_frame {
        None => true,
        Some(Ok(target)) => target == frame_id,
        Some(Err(())) => false,
    };
    let iteration_matches = match policy.full70_factor_oracle_iteration {
        None => true,
        Some(Ok(target)) => target == event.iteration,
        Some(Err(())) => false,
    };
    let trial_matches = match policy.full70_factor_oracle_trial {
        None => true,
        Some(Ok(target)) => target == event.trial,
        Some(Err(())) => false,
    };
    let phase_matches = match policy.full70_factor_oracle_phase.as_deref() {
        None => event.phase == "iteration_start",
        Some(target) => {
            matches!(
                target,
                "iteration_start" | "trial" | "accepted" | "rejected" | "converged"
            ) && target == event.phase
        }
    };
    if !(frame_matches && iteration_matches && trial_matches && phase_matches) {
        return false;
    }
    // A path names one lossless event bundle.  Without this one-shot guard a
    // selector such as frame=4 would successfully write the first iteration
    // and then repeatedly collide with the same target, producing misleading
    // diagnostic errors.  A caller that needs another event uses a fresh
    // process/path (or a narrower iteration/trial selector).
    static CAPTURED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    CAPTURED
        .compare_exchange(
            false,
            true,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        )
        .is_ok()
}

fn emit_iteration_event(problem: &mut WindowProblem, event: LmDiagnosticEvent<'_>) {
    if let Some(frame_id) = problem.states.last().map(|state| state.frame_id) {
        let policy = diagnostic_env_snapshot();
        if let Some(path) = policy.full70_factor_oracle.as_ref() {
            if full70_oracle_event_selected(policy, frame_id, &event) {
                let context = full70_oracle_window_context(problem);
                if let Err(error) = emit_full70_factor_oracle(path, frame_id, &event, &context) {
                    // Diagnostic capture is fail-closed and never changes the
                    // solver result.  Keep the error visible to the harness;
                    // a missing/partial oracle is not silently accepted.
                    eprintln!("full70 factor oracle rejected: {error:?}");
                }
            }
        }
    }
    let Some(path) = iteration_trace_path() else {
        return;
    };
    let Some(frame_id) = problem.states.last().map(|state| state.frame_id) else {
        return;
    };
    if !iteration_trace_frame_matches(frame_id) {
        return;
    }
    let base_state = event.base_state;
    let display_state = if event.phase == "trial" {
        event.trial_state.unwrap_or(event.state)
    } else {
        event.state
    };
    let mut display_problem = problem.clone();
    if event.phase == "trial" {
        if let Some(step) = event.step {
            // Reuse the exact trial clone used by `trial_cost`, including
            // manifold sidecars and eliminated-landmark increments. Keeping
            // one constructor avoids a diagnostic-only prior trajectory that
            // differs from the scalar LM objective.
            display_problem = problem.trial_problem_with_step_owned(base_state, step);
        }
    }
    let (landmark_factors, visual_spans) =
        visual_factor_records(problem, base_state, event.linearization);
    let (global_rows, global_rhs) = global_reduced_rows(event.linearization, problem.scalar_mode);
    let h_copy = event.damped_h.map(|matrix| {
        json!({
            "rows": matrix.nrows(),
            "cols": matrix.ncols(),
            "layout": "column_major",
            "f32_bits": matrix.iter().map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>(),
        })
    });
    let backsub = event.step.map_or_else(
        || JsonValue::Array(Vec::new()),
        |step| {
            JsonValue::Array(
                problem
                    .landmark_steps(base_state, step)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(index, increment)| {
                        let increment = increment?;
                        Some(json!({
                            "track_id": problem.landmarks.get(index)?.track_id,
                            "delta": json_vector(increment.iter().copied()),
                        }))
                    })
                    .collect(),
            )
        },
    );
    let model_parts = if problem.scalar_mode == ScalarMode::UpstreamF32 && event.phase == "trial" {
        event
            .step
            .zip(event.model_decrease)
            .and_then(|(step, actual)| {
                super::aom::diagnostic_model_decrease_parts_f32(
                    &event.linearization.factors,
                    step,
                    actual,
                )
            })
    } else {
        None
    };
    let record = json!({
        "schema": "basalt.vio_iteration.v1",
        "record": "snapshot",
        "source": "rust",
        "frame_id": frame_id,
        "iteration": event.iteration,
        "trial": event.trial,
        "phase": event.phase,
        "decision": event.decision,
        "lambda": json_number(event.lambda),
        "lambda_after": json_number(event.lambda_after),
        "cost": {
            "before": json_number(event.cost_before),
            "model": event.model_cost.map_or(JsonValue::Null, json_number),
            "model_decrease": event.model_decrease.map_or(JsonValue::Null, json_number),
            "model_parts": model_parts,
            "actual": event.actual_cost.map_or(JsonValue::Null, json_number),
            "step_norm": event.step_norm.map_or(JsonValue::Null, json_number),
        },
        "aom": json_aom(problem),
        "blocks": json_block_values(&display_problem, display_state),
        "landmarks": json_landmark_values(&display_problem),
        "delta": event.step.map_or(JsonValue::Null, |step| json_vector(step.iter().copied())),
        "backsub_landmark_delta": backsub,
        "damping_diag": json_vector(event.damping_diag.iter().copied()),
        "h_copy": h_copy.unwrap_or(JsonValue::Null),
        "global": {
            "h": json_matrix(&event.reduced.h),
            "b": json_vector(event.reduced.b.iter().copied()),
            "reduced_rows": json_matrix(&global_rows),
            "reduced_rhs": json_vector(global_rhs.iter().copied()),
            "normal_system_stages": diagnostic_normal_system_stages(
                event.reduced,
                &|value: f32| format!("{:08x}", value.to_bits()),
            ),
        },
        "category_costs": iteration_category_costs(&display_problem, display_state),
        "trial_visual_costs": if event.phase == "trial" && diagnostic_env_snapshot().visual_chain_trace {
            json_trial_visual_costs(&display_problem, display_state)
        } else {
            JsonValue::Null
        },
        "visual_row_spans": visual_spans
            .into_iter()
            .map(|(start, count)| json!([start, count]))
            .collect::<Vec<_>>(),
        "landmark_factors": landmark_factors,
        "factor_count": event.linearization.factors.len(),
        "factor_rows": event
            .linearization
            .factors
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>(),
        "numeric_types": {
            "state": "f64",
            "pose": "f64",
            "landmark": "f64",
            "observation": "f64",
            "camera": "f64",
            "landmark_projection": if problem.scalar_mode == ScalarMode::UpstreamF32 {
                "f32"
            } else {
                "f64"
            },
            "upstream_reference": "float32",
        },
        "trace_path": path,
    });
    emit_solver_frontier_sidecar(problem, &display_problem, &event);
    write_iteration_record(&record);
}

fn initial_window_diagnostics(problem: &WindowProblem, state: &DVector<f64>) -> WindowDiagnostics {
    let linearization = problem.linearize(state).ok();
    let imu_links = {
        let _capture_scope = InitialImuF64CaptureScope::enter();
        problem
            .imu_links
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(link_index, link)| {
                let _capture_link = InitialImuF64CaptureLink::set(link_index);
                problem
                    .imu_factor_with_trace(state, link)
                    .map(|(_, trace)| trace)
            })
            .collect::<Vec<_>>()
    };
    let (
        prior_cost,
        visual_cost,
        imu_cost,
        bias_cost,
        prior_factor_rows,
        visual_factor_rows,
        imu_factor_rows,
        bias_factor_rows,
    ) =
        linearization
            .as_ref()
            .map(|linearization| {
                let prior_factor_count = if problem.prior.is_some() { 1 } else { 0 };
                let anchor_factor_count = if problem.prior.is_none()
                    && problem.anchor_point.as_ref().is_some_and(|point| {
                        point.len() == NAV_STATE_DOF && !problem.states.is_empty()
                    }) {
                    1
                } else {
                    0
                };
                let prefix =
                    (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
                let imu_factor_count = problem
                    .imu_links
                    .len()
                    .saturating_mul(2)
                    .min(linearization.factors.len().saturating_sub(prefix));
                let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
                let cost = |factors: &[WhitenedFactorRowStack]| {
                    factors
                        .iter()
                        .map(|factor| factor.objective_cost)
                        .sum::<f64>()
                };
                let rows = |factors: &[WhitenedFactorRowStack]| {
                    factors
                        .iter()
                        .map(WhitenedFactorRowStack::rows)
                        .sum::<usize>()
                };
                let prior_cost = cost(&linearization.factors[..prefix]);
                let visual_cost = cost(&linearization.factors[prefix..visual_end]);
                let mut imu_cost = 0.0;
                let mut bias_cost = 0.0;
                for pair in linearization.factors[visual_end..].chunks(2) {
                    if let Some(factor) = pair.first() {
                        imu_cost += factor.objective_cost;
                    }
                    if let Some(factor) = pair.get(1) {
                        bias_cost += factor.objective_cost;
                    }
                }
                (
                    prior_cost,
                    visual_cost,
                    imu_cost,
                    bias_cost,
                    rows(&linearization.factors[..prefix]),
                    rows(&linearization.factors[prefix..visual_end]),
                    linearization.factors[visual_end..]
                        .chunks(2)
                        .filter_map(|pair| pair.first())
                        .map(WhitenedFactorRowStack::rows)
                        .sum(),
                    linearization.factors[visual_end..]
                        .chunks(2)
                        .filter_map(|pair| pair.get(1))
                        .map(WhitenedFactorRowStack::rows)
                        .sum(),
                )
            })
            .unwrap_or((0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0));
    if let Some(linearization) = linearization.as_ref() {
        if iteration_trace_path().is_some() {
            emit_iteration_header();
        } else {
            emit_detail_trace(problem, state, linearization);
        }
    }
    WindowDiagnostics {
        attempted: true,
        state_dof: state.len(),
        factor_count: linearization
            .as_ref()
            .map_or(0, |linearization| linearization.factors.len()),
        factor_rows: linearization.as_ref().map_or(0, |linearization| {
            linearization
                .factors
                .iter()
                .map(WhitenedFactorRowStack::rows)
                .sum()
        }),
        landmark_count: problem.landmarks.len(),
        imu_link_count: problem.imu_links.len(),
        prior_rows: problem.prior.as_ref().map_or(0, WindowPrior::rows),
        prior_factor_rows,
        visual_factor_rows,
        imu_factor_rows,
        bias_factor_rows,
        prior_cost,
        visual_cost,
        imu_cost,
        bias_cost,
        imu_links,
        lm: Vec::new(),
        status: "started".into(),
        failure: None,
        state_writeback: false,
        landmark_writeback: 0,
        prior_carry: false,
    }
}

/// Cheap output metadata for the explicit no-MargData/no-trace path.
///
/// The solver still performs all of its normal linearization and LM work, but
/// callers that do not retain MargData or a trace do not need a second
/// diagnostic linearization before that solve.  Keep the public diagnostic
/// shape valid while leaving expensive factor/cost/LM payloads empty.
fn minimal_window_diagnostics(problem: &WindowProblem, state: &DVector<f64>) -> WindowDiagnostics {
    WindowDiagnostics {
        attempted: true,
        state_dof: state.len(),
        factor_count: 0,
        factor_rows: 0,
        landmark_count: problem.landmarks.len(),
        imu_link_count: problem.imu_links.len(),
        prior_rows: problem.prior.as_ref().map_or(0, WindowPrior::rows),
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
        status: "started".into(),
        failure: None,
        state_writeback: false,
        landmark_writeback: 0,
        prior_carry: false,
    }
}

/// Populate the four `emit_marg` row counters from a final factor snapshot.
///
/// The lean-with-factors path deliberately skips `initial_window_diagnostics`
/// (which performs a full pre-solve linearization just for metadata).  The
/// factor grouping used here is the same prior/anchor → visual → IMU/bias
/// partition that function applies, so the MargData metadata is populated
/// without the duplicate linearization.
fn fill_diagnostic_factor_rows(
    problem: &WindowProblem,
    diagnostics: &mut WindowDiagnostics,
    factors: &[WhitenedFactorRowStack],
) {
    let prior_factor_count = usize::from(problem.prior.is_some());
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(factors.len().saturating_sub(prefix));
    let visual_end = factors.len().saturating_sub(imu_factor_count);
    let rows = |slice: &[WhitenedFactorRowStack]| {
        slice
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>()
    };
    diagnostics.factor_count = factors.len();
    diagnostics.factor_rows = rows(factors);
    diagnostics.prior_factor_rows = rows(&factors[..prefix]);
    diagnostics.visual_factor_rows = rows(&factors[prefix..visual_end]);
    diagnostics.imu_factor_rows = factors[visual_end..]
        .chunks(2)
        .filter_map(|pair| pair.first())
        .map(WhitenedFactorRowStack::rows)
        .sum();
    diagnostics.bias_factor_rows = factors[visual_end..]
        .chunks(2)
        .filter_map(|pair| pair.get(1))
        .map(WhitenedFactorRowStack::rows)
        .sum();
}

/// Emit a complete, reproducible numeric snapshot when explicitly requested.
///
/// The normal estimator path does not write diagnostics.  Setting
/// `VISLOC_BASALT_DETAIL_TRACE` enables one text artifact for the selected
/// solve; `VISLOC_BASALT_DETAIL_FRAME` can restrict it to a final state frame
/// (for example, `4` in the MH01 frame-4 audit).  The snapshot is assembled
/// from the same grouped factors passed to `linearize`, so its row ordering is
/// an assertion target rather than a second implementation of the factor.
fn emit_detail_trace(
    problem: &WindowProblem,
    state: &DVector<f64>,
    linearization: &LmLinearization,
) {
    let Some(path) = diagnostic_env_snapshot().detail_trace.as_ref() else {
        return;
    };
    let target_frame = diagnostic_env_snapshot().detail_frame;
    if let Some(target_frame) = target_frame {
        if problem
            .states
            .last()
            .is_none_or(|state| state.frame_id != target_frame)
        {
            return;
        }
    }

    let mut output = String::new();
    let state_dof = state.len();
    let prior_factor_count = if problem.prior.is_some() { 1 } else { 0 };
    let anchor_factor_count = if problem.prior.is_none()
        && problem
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !problem.states.is_empty())
    {
        1
    } else {
        0
    };
    let prefix = (prior_factor_count + anchor_factor_count).min(linearization.factors.len());
    let imu_factor_count = problem
        .imu_links
        .len()
        .saturating_mul(2)
        .min(linearization.factors.len().saturating_sub(prefix));
    let visual_end = linearization.factors.len().saturating_sub(imu_factor_count);
    let category_cost = |factors: &[WhitenedFactorRowStack]| {
        factors
            .iter()
            .map(|factor| factor.objective_cost)
            .sum::<f64>()
    };
    let prior_cost = category_cost(&linearization.factors[..prefix]);
    let visual_cost = category_cost(&linearization.factors[prefix..visual_end]);
    let mut imu_cost = 0.0;
    let mut bias_cost = 0.0;
    for pair in linearization.factors[visual_end..].chunks(2) {
        if let Some(factor) = pair.first() {
            imu_cost += factor.objective_cost;
        }
        if let Some(factor) = pair.get(1) {
            bias_cost += factor.objective_cost;
        }
    }

    writeln!(
        &mut output,
        "WINDOW_BEGIN frame {} state_dof {} factors {} rows {}",
        problem.states.last().map_or(0, |state| state.frame_id),
        state_dof,
        linearization.factors.len(),
        linearization
            .factors
            .iter()
            .map(WhitenedFactorRowStack::rows)
            .sum::<usize>()
    )
    .ok();
    writeln!(
        &mut output,
        "OBJECTIVE prior {prior_cost:.17e} visual {visual_cost:.17e} imu {imu_cost:.17e} bias {bias_cost:.17e} total {:.17e}",
        linearization.cost
    )
    .ok();
    writeln!(
        &mut output,
        "FACTOR_ORDER prefix_prior {} visual_end {} total {}",
        prefix,
        visual_end,
        linearization.factors.len()
    )
    .ok();
    // The reference traces print upstream `float` values.  Keep the Rust
    // storage/arithmetic types explicit in the component artifact so a
    // numerical mismatch cannot be mistaken for a row-order mismatch.
    writeln!(
        &mut output,
        "NUMERIC_TYPES state f64 pose f64 landmark f64 observation f64 camera f64 intermediate f64 landmark_projection {} upstream_reference float32",
        if problem.scalar_mode == ScalarMode::UpstreamF32 {
            "f32"
        } else {
            "f64"
        }
    )
    .ok();
    for (index, camera) in if problem.cameras.is_empty() {
        vec![(0_u16, problem.camera)]
    } else {
        problem
            .cameras
            .iter()
            .copied()
            .enumerate()
            .map(|(index, camera)| (index as u16, camera))
            .collect()
    } {
        writeln!(
            &mut output,
            "CAMERA index {} fx {:.17e} fy {:.17e} cx {:.17e} cy {:.17e} xi {:.17e} alpha {:.17e} width {} height {}",
            index,
            camera.fx,
            camera.fy,
            camera.cx,
            camera.cy,
            camera.xi,
            camera.alpha,
            camera.width,
            camera.height
        )
        .ok();
    }
    for (index, transform) in if problem.t_imu_cam.is_empty() {
        vec![(0_u16, SE3::identity())]
    } else {
        problem
            .t_imu_cam
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, transform)| (index as u16, transform))
            .collect()
    } {
        let quaternion = transform.rotation.quaternion();
        writeln!(
            &mut output,
            "EXTRINSIC camera {} translation {:.17e} {:.17e} {:.17e} quaternion_xyzw {:.17e} {:.17e} {:.17e} {:.17e}",
            index,
            transform.translation.x,
            transform.translation.y,
            transform.translation.z,
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w
        )
        .ok();
    }
    for index in 0..(problem.poses.len() + problem.states.len()) {
        let Some(pose) = problem.block_pose(state, index) else {
            continue;
        };
        let Some(frame_id) = problem.block_frame_id(index) else {
            continue;
        };
        let quaternion = pose.rotation.quaternion();
        writeln!(
            &mut output,
            "BLOCK index {} frame {} kind {} translation {:.17e} {:.17e} {:.17e} quaternion_xyzw {:.17e} {:.17e} {:.17e} {:.17e}",
            index,
            frame_id,
            if index < problem.poses.len() { "pose" } else { "state" },
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            quaternion.i,
            quaternion.j,
            quaternion.k,
            quaternion.w
        )
        .ok();
    }
    for (index, factor) in linearization.factors.iter().enumerate() {
        let category = if index < prefix {
            "prior"
        } else if index < visual_end {
            "visual"
        } else if (index - visual_end) % 2 == 0 {
            "imu"
        } else {
            "bias"
        };
        writeln!(
            &mut output,
            "FACTOR index {index} category {category} rows {} state_cols {} landmark_cols {} objective {:.17e}",
            factor.rows(),
            factor.state_jacobian.ncols(),
            factor.landmark_jacobian.ncols(),
            factor.objective_cost
        )
        .ok();
        emit_vector(
            &mut output,
            &format!("FACTOR_RESIDUAL index {index}"),
            &factor.residual,
        );
    }

    let mut visual_factors = Vec::new();
    for landmark_index in 0..problem.landmarks.len() {
        let Some(audit) = problem.visual_factor_with_audit(state, landmark_index) else {
            continue;
        };
        writeln!(
            &mut output,
            "VISUAL_ORDER ordinal {} track_id {} host_frame {} host_cam {} rows {}",
            visual_factors.len(),
            audit.track_id,
            audit.anchor_frame_id,
            audit.anchor_camera_id,
            audit.factor.rows()
        )
        .ok();
        let (reduced_jacobian, reduced_rhs, rank) =
            project_landmark_factor(&audit.factor, 1e-10, problem.scalar_mode);
        writeln!(
            &mut output,
            "LANDMARK_BEGIN track_id {} host_frame {} host_cam {} direction {:.17e} {:.17e} rho {:.17e} observations {} rows {} rank {} objective {:.17e}",
            audit.track_id,
            audit.anchor_frame_id,
            audit.anchor_camera_id,
            audit.direction.xy.x,
            audit.direction.xy.y,
            audit.inverse_distance,
            audit.observations.len(),
            audit.factor.rows(),
            rank,
            audit.factor.objective_cost
        )
        .ok();
        for observation in &audit.observations {
            let factor = &observation.factor;
            writeln!(
                &mut output,
                "OBS target_frame {} target_cam {} pixel {:.17e} {:.17e} raw_residual {:.17e} {:.17e} residual {:.17e} {:.17e} huber_weight {:.17e} sqrt_weight {:.17e} objective {:.17e}",
                problem
                    .block_frame_id(observation.observation.state_index)
                    .unwrap_or_default(),
                observation.observation.camera_id,
                observation.observation.pixel.x,
                observation.observation.pixel.y,
                factor.raw_residual.x,
                factor.raw_residual.y,
                factor.residual.x,
                factor.residual.y,
                factor.huber_weight,
                factor.sqrt_weight,
                factor.objective_cost
            )
            .ok();
            writeln!(
                &mut output,
                "PROJECTION {:.17e} {:.17e}",
                factor.projection.x, factor.projection.y
            )
            .ok();
            emit_fixed_matrix(&mut output, "Jp_anchor", &factor.anchor_pose_jacobian);
            emit_fixed_matrix(&mut output, "Jp_target", &factor.target_pose_jacobian);
            emit_fixed_matrix(&mut output, "Jl", &factor.landmark_jacobian);
        }
        emit_matrix(
            &mut output,
            "LANDMARK_STATE_JACOBIAN",
            &audit.factor.state_jacobian,
        );
        emit_matrix(
            &mut output,
            "LANDMARK_JACOBIAN",
            &audit.factor.landmark_jacobian,
        );
        emit_vector(&mut output, "LANDMARK_RESIDUAL", &audit.factor.residual);
        emit_matrix(&mut output, "REDUCED_ROWS", &reduced_jacobian);
        emit_vector(&mut output, "REDUCED_RHS", &reduced_rhs);
        visual_factors.push(audit.factor);
    }

    // This is the exact reduced system consumed by the LM solve, including
    // prior/IMU/bias rows and visual factors in their linearize order.
    let reduced = reduce_landmark_factors(&linearization.factors, state_dof, 1e-10);
    emit_matrix(&mut output, "REDUCED_H", &reduced.h);
    emit_vector(&mut output, "REDUCED_B", &reduced.b);
    let total_reduced_rows = linearization
        .factors
        .iter()
        .map(WhitenedFactorRowStack::rows)
        .sum::<usize>();
    let mut reduced_rows = DMatrix::zeros(total_reduced_rows, state_dof);
    let mut reduced_rhs = DVector::zeros(total_reduced_rows);
    let mut row = 0;
    for factor in &linearization.factors {
        let (jacobian, rhs, _) = project_landmark_factor(factor, 1e-10, problem.scalar_mode);
        let rows = jacobian.nrows();
        debug_assert_eq!(jacobian.ncols(), state_dof);
        reduced_rows
            .view_mut((row, 0), (rows, state_dof))
            .copy_from(&jacobian);
        reduced_rhs.rows_mut(row, rows).copy_from(&rhs);
        row += rows;
    }
    debug_assert_eq!(row, total_reduced_rows);
    emit_matrix(&mut output, "REDUCED_ROW_STACK", &reduced_rows);
    emit_vector(&mut output, "REDUCED_ROW_RHS", &reduced_rhs);
    writeln!(
        &mut output,
        "LANDMARK_FACTOR_COUNT {}",
        visual_factors.len()
    )
    .ok();
    writeln!(&mut output, "WINDOW_END").ok();

    if let Err(error) = fs::write(&path, output) {
        eprintln!(
            "basalt detail trace: failed to write {}: {error}",
            path.display()
        );
    }
}

fn emit_fixed_matrix<const ROWS: usize, const COLS: usize>(
    output: &mut String,
    label: &str,
    matrix: &nalgebra::SMatrix<f64, ROWS, COLS>,
) {
    writeln!(output, "{label} rows {ROWS} cols {COLS}").ok();
    for row in 0..ROWS {
        for col in 0..COLS {
            write!(output, " {:.17e}", matrix[(row, col)]).ok();
        }
        output.push('\n');
    }
}

fn emit_matrix(output: &mut String, label: &str, matrix: &DMatrix<f64>) {
    writeln!(
        output,
        "{label} rows {} cols {}",
        matrix.nrows(),
        matrix.ncols()
    )
    .ok();
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            write!(output, " {:.17e}", matrix[(row, col)]).ok();
        }
        output.push('\n');
    }
}

fn emit_vector(output: &mut String, label: &str, vector: &DVector<f64>) {
    writeln!(output, "{label} len {}", vector.len()).ok();
    for value in vector.iter() {
        writeln!(output, "{value:.17e}").ok();
    }
}

#[derive(Debug, Clone)]
pub struct WindowProblem {
    /// Persistent native host-map traversal, keyed by (timestamp bits, camera).
    /// Empty for manually constructed windows without native lifecycle history.
    /// This order is distinct from the landmark-index linearization order.
    pub trial_host_order: Vec<(u64, u16)>,
    /// Camera 0 is kept as the compatibility/default model for callers that
    /// construct a window without a full rig.  Production EuRoC replay fills
    /// `cameras` and `t_imu_cam` from the Basalt calibration JSON.
    pub camera: DoubleSphereCamera,
    /// Per-stream Double Sphere intrinsics indexed by `WindowObservation::camera_id`.
    pub cameras: Vec<DoubleSphereCamera>,
    /// `T_imu_cam`: maps a camera-frame point into the IMU frame.  The camera
    /// pose used by visual factors is `T_w_c = T_w_i * T_imu_cam`.
    pub t_imu_cam: Vec<SE3>,
    /// Pose-only keyframes are ordered before full navigation states in the
    /// absolute order map, matching Basalt's `frame_poses` then
    /// `frame_states` construction.
    pub poses: Vec<WindowPose>,
    pub states: Vec<WindowState>,
    pub landmarks: Vec<WindowLandmark>,
    pub imu_links: Vec<WindowImuLink>,
    /// Continuous-time gyro/accelerometer measurement noise densities used
    /// when preintegration covariance is converted to square-root information.
    pub imu_noise: ImuNoiseModel,
    /// Continuous-time gyro/accelerometer bias random-walk densities.
    pub bias_walk_noise: BiasRandomWalkNoise,
    /// Basalt's initial square-root prior weights. Position and yaw plus the
    /// two bias blocks are constrained; velocity and roll/pitch are free.
    pub initial_pose_weight: f64,
    pub initial_accel_bias_weight: f64,
    pub initial_gyro_bias_weight: f64,
    pub prior: Option<WindowPrior>,
    pub anchor_point: Option<DVector<f64>>,
    pub gravity_world: Vector3<f64>,
    /// Internal numeric owner. Public estimator/API values remain f64.
    pub scalar_mode: ScalarMode,
}

const LM_TOKEN_FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
const LM_TOKEN_FNV_PRIME: u64 = 1_099_511_628_211;

#[inline]
const fn lm_token_hash_u64(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(LM_TOKEN_FNV_PRIME);
}

#[inline]
const fn lm_token_hash_usize(hash: &mut u64, value: usize) {
    lm_token_hash_u64(hash, value as u64);
}

#[inline]
const fn lm_token_hash_bool(hash: &mut u64, value: bool) {
    lm_token_hash_u64(hash, value as u64);
}

#[inline]
const fn lm_token_hash_f64(hash: &mut u64, value: f64) {
    lm_token_hash_u64(hash, value.to_bits());
}

#[inline]
fn lm_token_hash_se3(hash: &mut u64, value: &SE3) {
    for component in value.rotation.coords.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.translation.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

#[inline]
fn lm_token_hash_nav(hash: &mut u64, value: &BasaltNavState) {
    lm_token_hash_se3(hash, &value.imu_to_world);
    for component in value.velocity_world_m_s.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.gyro_bias_rad_s.iter() {
        lm_token_hash_f64(hash, *component);
    }
    for component in value.accel_bias_m_s2.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

#[inline]
fn lm_token_hash_vector(hash: &mut u64, value: &DVector<f64>) {
    lm_token_hash_usize(hash, value.len());
    for component in value.iter() {
        lm_token_hash_f64(hash, *component);
    }
}

fn lm_token_camera_fingerprint(hash: &mut u64, camera: &DoubleSphereCamera) {
    lm_token_hash_f64(hash, camera.fx);
    lm_token_hash_f64(hash, camera.fy);
    lm_token_hash_f64(hash, camera.cx);
    lm_token_hash_f64(hash, camera.cy);
    lm_token_hash_f64(hash, camera.xi);
    lm_token_hash_f64(hash, camera.alpha);
    lm_token_hash_u64(hash, camera.width as u64);
    lm_token_hash_u64(hash, camera.height as u64);
}

/// Fingerprint the current window identity/topology and mutable sidecars that
/// can make a previously recovered landmark increment stale.  This is a
/// deterministic, allocation-free FNV-1a pass; the object address is kept as
/// a separate binding field so a clone cannot consume another window's token.
fn lm_token_window_generation(problem: &WindowProblem) -> u64 {
    let mut hash = LM_TOKEN_FNV_OFFSET;
    lm_token_hash_usize(&mut hash, problem.cameras.len());
    lm_token_hash_usize(&mut hash, problem.t_imu_cam.len());
    lm_token_camera_fingerprint(&mut hash, &problem.camera);
    for camera in &problem.cameras {
        lm_token_camera_fingerprint(&mut hash, camera);
    }
    for transform in &problem.t_imu_cam {
        lm_token_hash_se3(&mut hash, transform);
    }

    lm_token_hash_usize(&mut hash, problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, pose.frame_id);
        lm_token_hash_u64(&mut hash, pose.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, pose.is_keyframe);
        lm_token_hash_se3(&mut hash, &pose.pose);
        lm_token_hash_se3(&mut hash, &pose.stored_current_pose);
        lm_token_hash_se3(&mut hash, &pose.linearized_pose);
        lm_token_hash_vector(&mut hash, &pose.linearized_delta);
    }

    lm_token_hash_usize(&mut hash, problem.states.len());
    for (index, state) in problem.states.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, state.frame_id);
        lm_token_hash_u64(&mut hash, state.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, state.is_keyframe);
        lm_token_hash_bool(&mut hash, state.is_latest);
        lm_token_hash_bool(&mut hash, state.linearized);
        lm_token_hash_nav(&mut hash, &state.nav);
        lm_token_hash_nav(&mut hash, &state.stored_current_nav);
        lm_token_hash_nav(&mut hash, &state.linearized_nav);
        lm_token_hash_vector(&mut hash, &state.linearized_delta);
    }

    lm_token_hash_usize(&mut hash, problem.trial_host_order.len());
    for &(timestamp, camera) in &problem.trial_host_order {
        lm_token_hash_u64(&mut hash, timestamp);
        lm_token_hash_u64(&mut hash, camera as u64);
    }
    lm_token_hash_usize(&mut hash, problem.landmarks.len());
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, landmark.track_id);
        lm_token_hash_usize(&mut hash, landmark.anchor_state_index);
        lm_token_hash_u64(&mut hash, landmark.anchor_camera_id as u64);
        lm_token_hash_f64(&mut hash, landmark.direction.xy.x);
        lm_token_hash_f64(&mut hash, landmark.direction.xy.y);
        lm_token_hash_f64(&mut hash, landmark.inverse_distance);
        lm_token_hash_usize(&mut hash, landmark.observations.len());
        for observation in &landmark.observations {
            lm_token_hash_usize(&mut hash, observation.state_index);
            lm_token_hash_u64(&mut hash, observation.camera_id as u64);
            lm_token_hash_f64(&mut hash, observation.pixel.x);
            lm_token_hash_f64(&mut hash, observation.pixel.y);
        }
    }

    lm_token_hash_usize(&mut hash, problem.imu_links.len());
    for link in &problem.imu_links {
        lm_token_hash_usize(&mut hash, link.from_index);
        lm_token_hash_usize(&mut hash, link.to_index);
        lm_token_hash_f64(&mut hash, link.delta.delta_time);
    }

    match &problem.prior {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(prior) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_usize(&mut hash, prior.frame_ids.len());
            for frame_id in &prior.frame_ids {
                lm_token_hash_u64(&mut hash, *frame_id);
            }
            lm_token_hash_usize(&mut hash, prior.block_kinds.len());
            for kind in &prior.block_kinds {
                lm_token_hash_u64(
                    &mut hash,
                    match kind {
                        WindowBlockKind::Pose => 1,
                        WindowBlockKind::StatePose => 3,
                        WindowBlockKind::State => 2,
                    },
                );
            }
            lm_token_hash_usize(&mut hash, prior.jacobian.nrows());
            lm_token_hash_usize(&mut hash, prior.jacobian.ncols());
            lm_token_hash_usize(&mut hash, prior.rhs.len());
            lm_token_hash_usize(&mut hash, prior.fej_point.len());
        }
    }
    match &problem.anchor_point {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(point) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_vector(&mut hash, point);
        }
    }
    hash
}

#[inline]
fn lm_token_hash_f64_iter<'a, I>(hash: &mut u64, values: I)
where
    I: IntoIterator<Item = &'a f64>,
{
    for value in values {
        lm_token_hash_f64(hash, *value);
    }
}

/// Hash the immutable numerical inputs consumed by the active factor plan.
///
/// `lm_token_window_generation` deliberately emphasizes mutable window
/// sidecars used by the accepted-step token.  A preparation also needs a
/// separate factor-plan binding: prior values, noise, preintegrated IMU
/// derivatives, and factor-order metadata can change the rows while the
/// solved state/step fingerprints remain unchanged.  This pass is still
/// allocation-free and does not call `linearize`, so binding cannot alter the
/// solver's arithmetic or factor schedule.
fn lm_token_hash_imu_delta(hash: &mut u64, delta: &ImuPreintegratedDelta) {
    lm_token_hash_f64_iter(hash, delta.delta_rotation.coords.iter());
    lm_token_hash_f64_iter(hash, delta.delta_velocity.iter());
    lm_token_hash_f64_iter(hash, delta.delta_position.iter());
    lm_token_hash_f64(hash, delta.delta_time);
    lm_token_hash_f64_iter(hash, delta.bias_gyro.iter());
    lm_token_hash_f64_iter(hash, delta.bias_accel.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_rotation_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_velocity_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_velocity_accel_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_position_gyro_bias.iter());
    lm_token_hash_f64_iter(hash, delta.jacobian_position_accel_bias.iter());
    lm_token_hash_f64_iter(hash, delta.covariance.iter());
}

fn lm_token_factor_plan_fingerprint(problem: &WindowProblem) -> u64 {
    let mut hash = LM_TOKEN_FNV_OFFSET;
    // Version/tag keeps future additions from silently sharing an old binding
    // domain if the factor construction contract changes.
    lm_token_hash_u64(&mut hash, 0x5142_504c_414e_5631);
    lm_token_hash_u64(
        &mut hash,
        match problem.scalar_mode {
            ScalarMode::UpstreamF32 => 1,
            ScalarMode::ExtendedF64 => 2,
        },
    );
    lm_token_hash_camera_plan(&mut hash, problem);

    lm_token_hash_usize(&mut hash, problem.poses.len());
    for (index, pose) in problem.poses.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, pose.frame_id);
        lm_token_hash_u64(&mut hash, pose.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, pose.is_keyframe);
    }
    lm_token_hash_usize(&mut hash, problem.states.len());
    for (index, state) in problem.states.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, state.frame_id);
        lm_token_hash_u64(&mut hash, state.timestamp_ns as u64);
        lm_token_hash_bool(&mut hash, state.is_keyframe);
        lm_token_hash_bool(&mut hash, state.is_latest);
        lm_token_hash_bool(&mut hash, state.linearized);
    }

    // Visual factor order is landmark-index order, with observation rows in
    // their stored order.  Include both identities and the source geometry so
    // a same-state/step preparation cannot cross a changed visual plan.
    lm_token_hash_usize(&mut hash, problem.trial_host_order.len());
    for &(timestamp, camera) in &problem.trial_host_order {
        lm_token_hash_u64(&mut hash, timestamp);
        lm_token_hash_u64(&mut hash, camera as u64);
    }
    lm_token_hash_usize(&mut hash, problem.landmarks.len());
    for (index, landmark) in problem.landmarks.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_u64(&mut hash, landmark.track_id);
        lm_token_hash_usize(&mut hash, landmark.anchor_state_index);
        lm_token_hash_u64(&mut hash, landmark.anchor_camera_id as u64);
        lm_token_hash_f64_iter(&mut hash, landmark.direction.xy.coords.iter());
        lm_token_hash_f64(&mut hash, landmark.inverse_distance);
        lm_token_hash_usize(&mut hash, landmark.observations.len());
        for (observation_index, observation) in landmark.observations.iter().enumerate() {
            lm_token_hash_usize(&mut hash, observation_index);
            lm_token_hash_usize(&mut hash, observation.state_index);
            lm_token_hash_u64(&mut hash, observation.camera_id as u64);
            lm_token_hash_f64(&mut hash, observation.pixel.x);
            lm_token_hash_f64(&mut hash, observation.pixel.y);
        }
    }

    // Each IMU link contributes a contiguous [Imu9, Bias6] pair.  Hash the
    // complete preintegrated numerical payload (excluding update_trace, which
    // is diagnostic provenance and is not consumed by factor construction).
    lm_token_hash_usize(&mut hash, problem.imu_links.len());
    for (index, link) in problem.imu_links.iter().enumerate() {
        lm_token_hash_usize(&mut hash, index);
        lm_token_hash_usize(&mut hash, link.from_index);
        lm_token_hash_usize(&mut hash, link.to_index);
        lm_token_hash_imu_delta(&mut hash, &link.delta);
    }
    lm_token_hash_f64(&mut hash, problem.imu_noise.gyro_density);
    lm_token_hash_f64(&mut hash, problem.imu_noise.accel_density);
    lm_token_hash_f64(&mut hash, problem.bias_walk_noise.gyro_density);
    lm_token_hash_f64(&mut hash, problem.bias_walk_noise.accel_density);
    lm_token_hash_f64(&mut hash, problem.initial_pose_weight);
    lm_token_hash_f64(&mut hash, problem.initial_accel_bias_weight);
    lm_token_hash_f64(&mut hash, problem.initial_gyro_bias_weight);
    lm_token_hash_f64_iter(&mut hash, problem.gravity_world.iter());

    match &problem.prior {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(prior) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_usize(&mut hash, prior.frame_ids.len());
            for frame_id in &prior.frame_ids {
                lm_token_hash_u64(&mut hash, *frame_id);
            }
            let explicit_kinds = prior.block_kinds.len() == prior.frame_ids.len();
            lm_token_hash_bool(&mut hash, explicit_kinds);
            if explicit_kinds {
                for kind in &prior.block_kinds {
                    lm_token_hash_u64(
                        &mut hash,
                        match kind {
                            WindowBlockKind::Pose => 1,
                            WindowBlockKind::StatePose => 3,
                            WindowBlockKind::State => 2,
                        },
                    );
                }
            } else {
                for _ in &prior.frame_ids {
                    lm_token_hash_u64(&mut hash, 2);
                }
            }
            lm_token_hash_usize(&mut hash, prior.jacobian.nrows());
            lm_token_hash_usize(&mut hash, prior.jacobian.ncols());
            lm_token_hash_f64_iter(&mut hash, prior.jacobian.iter());
            lm_token_hash_usize(&mut hash, prior.rhs.len());
            lm_token_hash_vector(&mut hash, &prior.rhs);
            lm_token_hash_usize(&mut hash, prior.fej_point.len());
            lm_token_hash_vector(&mut hash, &prior.fej_point);
        }
    }
    match &problem.anchor_point {
        None => lm_token_hash_u64(&mut hash, 0),
        Some(point) => {
            lm_token_hash_u64(&mut hash, 1);
            lm_token_hash_vector(&mut hash, point);
        }
    }
    hash
}

fn lm_token_hash_camera_plan(hash: &mut u64, problem: &WindowProblem) {
    lm_token_hash_camera(hash, &problem.camera);
    lm_token_hash_usize(hash, problem.cameras.len());
    for camera in &problem.cameras {
        lm_token_hash_camera(hash, camera);
    }
    lm_token_hash_usize(hash, problem.t_imu_cam.len());
    for transform in &problem.t_imu_cam {
        lm_token_hash_se3(hash, transform);
    }
}

#[inline]
fn lm_token_hash_camera(hash: &mut u64, camera: &DoubleSphereCamera) {
    lm_token_camera_fingerprint(hash, camera);
}

/// The dynamic part of an LM trial.  The static window topology (camera
/// models, observations, IMU links, and the square-root prior) stays borrowed
/// from the accepted [`WindowProblem`].  Only the manifold sidecars that can
/// change during a trial are materialized here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct TrialLandmarkValue {
    direction: StereographicDirection,
    inverse_distance: f64,
}

#[derive(Debug)]
struct WindowTrialView<'problem, 'chart> {
    base: &'problem WindowProblem,
    chart: &'chart DVector<f64>,
    poses: Vec<WindowPose>,
    states: Vec<WindowState>,
    landmarks: Vec<TrialLandmarkValue>,
    /// Exact eliminated-landmark increments produced while constructing this
    /// eager view.  A clean accepted trial moves this vector into the
    /// one-shot LM token instead of recovering it a second time.
    landmark_steps: Vec<Option<Vector3<f64>>>,
    /// FEJ-local point used by the carried prior.  Keeping this alongside the
    /// trial sidecars avoids rebuilding it from the accepted problem and makes
    /// the prior boundary explicit in the view contract.
    prior_fej_delta: Option<DVector<f64>>,
}

/// Timing sink used by [`WindowTrialView::from_step`].  The clean solver's
/// normal path instantiates the no-op implementation, so the generic
/// constructor monomorphizes away the timing calls and keeps the existing
/// disabled path free of `Instant` reads and timing allocations.  The active
/// implementation is selected only by the sidecar-enabled timed hook.
trait TrialConstructTiming {
    fn start(&self) -> Option<TimingStart>;

    fn finish(&mut self, bucket: TimingBucket, started: Option<TimingStart>);
}

#[derive(Debug, Default)]
struct NoTrialConstructTiming;

impl TrialConstructTiming for NoTrialConstructTiming {
    #[inline]
    fn start(&self) -> Option<TimingStart> {
        None
    }

    #[inline]
    fn finish(&mut self, _bucket: TimingBucket, _started: Option<TimingStart>) {}
}

struct EnabledTrialConstructTiming<'timing> {
    timing: &'timing mut TimingBreakdown,
}

impl TrialConstructTiming for EnabledTrialConstructTiming<'_> {
    #[inline]
    fn start(&self) -> Option<TimingStart> {
        self.timing.start()
    }

    #[inline]
    fn finish(&mut self, bucket: TimingBucket, started: Option<TimingStart>) {
        self.timing.finish(bucket, started);
    }
}

/// Add a compact solver increment at Basalt's float boundary without
/// materializing a full trial state.  The returned value is bounded by the
/// navigation/pose block width used by the eager trial view.
fn upstream_f32_accumulated_delta(base: &[f64], increment: &[f64]) -> Option<DVector<f64>> {
    if base.len() != increment.len() {
        return None;
    }
    let mut result = DVector::from_column_slice(base);
    for (lhs, rhs) in result.iter_mut().zip(increment) {
        *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
    }
    Some(result)
}

/// The single source of truth for an UpstreamF32 pose trial.  Both
/// `apply_step_full` (acceptance/diagnostics) and the eager trial view call
/// this exact reconstruction, including the f32 SO(3) product boundary.
fn upstream_f32_pose_from_delta(base_pose: &SE3, delta: &[f64]) -> Option<SE3> {
    if delta.len() < POSE_DOF {
        return None;
    }
    let translation = vec3_f32(base_pose.translation)
        + Vector3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32);
    let rotation = sophus_so3_product(
        so3_exp_f32(Vector3::new(
            delta[3] as f32,
            delta[4] as f32,
            delta[5] as f32,
        )),
        quat_f32_from_f64(&base_pose.rotation),
    );
    Some(SE3::new(
        quat_f64_from_f32(&rotation),
        vec3_f64(translation),
    ))
}

/// Pose returned by native `getPoseStateWithLin().getPose()` for a retained
/// navigation state.  This boundary is also used by delayed triangulation:
/// once a state is FEJ-linearized, constructing the temporary pose applies
/// the accumulated tangent even when it is exactly zero, so the quaternion
/// crosses Sophus' packet normalization once more.  A non-linearized state
/// still exposes its raw `state_linearized.T_w_i` payload.
pub(crate) fn visual_state_pose_with_mode(state: &WindowState, scalar_mode: ScalarMode) -> SE3 {
    if scalar_mode != ScalarMode::UpstreamF32 {
        return state.nav.imu_to_world.clone();
    }
    if !state.linearized {
        return state.linearized_nav.imu_to_world.clone();
    }
    if state.linearized_delta.len() >= POSE_DOF {
        return upstream_f32_pose_from_delta(
            &state.linearized_nav.imu_to_world,
            state.linearized_delta.as_slice(),
        )
        .unwrap_or_else(|| state.nav.imu_to_world.clone());
    }
    state.nav.imu_to_world.clone()
}

/// Apply one full navigation increment using the pinned Basalt float order.
/// `trace_frame_id` is optional so callers can share the arithmetic without
/// duplicating state-update diagnostics; the acceptance path supplies it.
fn upstream_f32_apply_nav_delta(
    nav: &mut BasaltNavState,
    delta: &[f64],
    trace_frame_id: Option<u64>,
) -> bool {
    if delta.len() < NAV_STATE_DOF {
        return false;
    }
    nav.imu_to_world.translation = vec3_f64(
        vec3_f32(nav.imu_to_world.translation)
            + Vector3::new(delta[0] as f32, delta[1] as f32, delta[2] as f32),
    );
    let before_rotation = quat_f32_from_f64(&nav.imu_to_world.rotation);
    let omega = Vector3::new(delta[3] as f32, delta[4] as f32, delta[5] as f32);
    let delta_rotation = so3_exp_f32(omega);
    let updated_rotation = sophus_so3_product(delta_rotation, before_rotation);
    if let Some(frame_id) = trace_frame_id {
        trace_state_update_quaternion(
            frame_id,
            before_rotation,
            omega,
            delta_rotation,
            updated_rotation,
        );
    }
    nav.imu_to_world.rotation = quat_f64_from_f32(&updated_rotation);
    nav.velocity_world_m_s = vec3_f64(
        vec3_f32(nav.velocity_world_m_s)
            + Vector3::new(delta[6] as f32, delta[7] as f32, delta[8] as f32),
    );
    nav.gyro_bias_rad_s = vec3_f64(
        vec3_f32(nav.gyro_bias_rad_s)
            + Vector3::new(delta[9] as f32, delta[10] as f32, delta[11] as f32),
    );
    nav.accel_bias_m_s2 = vec3_f64(
        vec3_f32(nav.accel_bias_m_s2)
            + Vector3::new(delta[12] as f32, delta[13] as f32, delta[14] as f32),
    );
    true
}

/// Value access needed by the factor builders.  Implementations deliberately
/// expose topology through `base()` but own/borrow all mutable trial values;
/// this keeps factor order and the `LmProblem` API unchanged while allowing a
/// clean trial to avoid cloning the whole window.
trait WindowValueView {
    fn base(&self) -> &WindowProblem;
    fn chart(&self) -> &DVector<f64>;
    fn block_nav(&self, index: usize) -> Option<BasaltNavState>;
    fn block_nav_visual_current(&self, index: usize) -> Option<BasaltNavState>;
    fn block_nav_linearized(&self, index: usize) -> Option<BasaltNavState>;
    fn landmark_parameter(&self, index: usize) -> Option<InverseDistanceLandmark>;
    fn state_is_linearized(&self, index: usize) -> Option<bool>;
    fn state_linearized_nav(&self, index: usize) -> Option<BasaltNavState>;
    fn linearized_delta(&self, kind: WindowBlockKind, index: usize) -> Option<DVector<f64>>;

    fn prior_current_point(&self, prior: &WindowPrior) -> DVector<f64>;
}

impl WindowTrialView<'_, '_> {
    fn from_step<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
    ) -> WindowTrialView<'problem, 'chart> {
        let mut timing = NoTrialConstructTiming;
        Self::from_step_with_timing(base, state, step, chart, &mut timing)
    }

    /// Construct an eager trial view and, when timing is enabled, retain the
    /// four non-overlapping construction sub-buckets inside the existing
    /// `lm_trial_construct_step` span. The disabled branch deliberately calls
    /// the no-op constructor above so ordinary solver runs take no timestamps
    /// and allocate no timing state.
    fn from_step_timed<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> WindowTrialView<'problem, 'chart> {
        if !timing.enabled() {
            return Self::from_step(base, state, step, chart);
        }

        let outer_started = timing.start();
        let view = {
            let mut timing_sink = EnabledTrialConstructTiming { timing };
            Self::from_step_with_timing(base, state, step, chart, &mut timing_sink)
        };
        timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
        view
    }

    fn from_step_with_timing<'problem, 'chart, T: TrialConstructTiming>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        timing: &mut T,
    ) -> WindowTrialView<'problem, 'chart> {
        let landmark_recovery_started = timing.start();
        let increments = base.landmark_steps(state, step);
        timing.finish(
            TimingBucket::LmTrialLandmarkRecovery,
            landmark_recovery_started,
        );

        Self::from_step_with_landmark_steps_timing(base, state, step, chart, increments, timing)
    }

    /// Construct a trial view from landmark increments recovered by the clean
    /// reducer.  Keeping this path separate from `from_step_with_timing`
    /// ensures a prepared trial never recomputes `base.landmark_steps`.
    fn from_step_with_landmark_steps<'problem, 'chart>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        increments: Vec<Option<Vector3<f64>>>,
    ) -> WindowTrialView<'problem, 'chart> {
        let mut timing = NoTrialConstructTiming;
        Self::from_step_with_landmark_steps_timing(
            base,
            state,
            step,
            chart,
            increments,
            &mut timing,
        )
    }

    fn from_step_with_landmark_steps_timing<'problem, 'chart, T: TrialConstructTiming>(
        base: &'problem WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        chart: &'chart DVector<f64>,
        increments: Vec<Option<Vector3<f64>>>,
        timing: &mut T,
    ) -> WindowTrialView<'problem, 'chart> {
        let apply_step_started = timing.start();
        let (trial_poses, trial_states) = base.apply_step_full(state, step);
        timing.finish(TimingBucket::LmTrialApplyStepFull, apply_step_started);

        let scalar_mode = base.scalar_mode;
        let state_base_offset = base.poses.len() * POSE_DOF;

        let sidecar_started = timing.start();
        let poses = base
            .poses
            .iter()
            .enumerate()
            .zip(trial_poses)
            .map(|((index, pose), trial_pose)| {
                let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
                let mut linearized_delta = pose.linearized_delta.clone();
                accumulate_linearized_delta(&mut linearized_delta, &increment, scalar_mode);
                WindowPose {
                    frame_id: pose.frame_id,
                    timestamp_ns: pose.timestamp_ns,
                    pose: trial_pose.clone(),
                    stored_current_pose: trial_pose,
                    linearized_pose: pose.linearized_pose.clone(),
                    linearized_delta,
                    is_keyframe: pose.is_keyframe,
                }
            })
            .collect::<Vec<_>>();

        let states = base
            .states
            .iter()
            .enumerate()
            .zip(trial_states)
            .map(|((index, state_base), trial_state)| {
                let mut linearized_delta = state_base.linearized_delta.clone();
                let (linearized_nav, stored_current_nav) = if !state_base.linearized {
                    linearized_delta.fill(0.0);
                    (trial_state.clone(), state_base.stored_current_nav.clone())
                } else {
                    let increment = step
                        .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                        .into_owned();
                    accumulate_linearized_delta(&mut linearized_delta, &increment, scalar_mode);
                    (state_base.linearized_nav.clone(), trial_state.clone())
                };
                WindowState {
                    frame_id: state_base.frame_id,
                    timestamp_ns: state_base.timestamp_ns,
                    nav: trial_state,
                    stored_current_nav,
                    linearized_nav,
                    linearized_delta,
                    is_keyframe: state_base.is_keyframe,
                    is_latest: state_base.is_latest,
                    linearized: state_base.linearized,
                }
            })
            .collect::<Vec<_>>();
        timing.finish(TimingBucket::LmTrialSidecarOther, sidecar_started);

        let landmark_materialization_started = timing.start();

        // Keep observations and anchor metadata borrowed from `base`; only
        // the two numeric landmark coordinates are trial-owned.  A failed
        // increment has exactly the same semantics as
        // `WindowProblem::apply_landmark_steps`: retain the previous value.
        let mut landmarks = base
            .landmarks
            .iter()
            .map(|landmark| TrialLandmarkValue {
                direction: landmark.direction,
                inverse_distance: landmark.inverse_distance,
            })
            .collect::<Vec<_>>();
        for (index, increment) in increments.iter().enumerate() {
            if let Some(increment) = increment {
                if let Some(value) = base.trial_landmark_value(index, *increment) {
                    landmarks[index] = value;
                }
            }
        }
        timing.finish(
            TimingBucket::LmTrialLandmarkMaterialization,
            landmark_materialization_started,
        );

        let sidecar_started = timing.start();
        let mut view = WindowTrialView {
            base,
            chart,
            poses,
            states,
            landmarks,
            landmark_steps: increments,
            prior_fej_delta: None,
        };
        view.prior_fej_delta = base
            .prior
            .as_ref()
            .map(|prior| view.compute_prior_current_point(prior));
        timing.finish(TimingBucket::LmTrialSidecarOther, sidecar_started);
        view
    }

    fn compute_prior_current_point(&self, prior: &WindowPrior) -> DVector<f64> {
        let kinds = prior.kinds();
        let mut current_point = DVector::zeros(prior.fej_point.len());
        let mut prior_offset = 0;
        for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
            let kind = kinds
                .get(prior_index)
                .copied()
                .unwrap_or(WindowBlockKind::State);
            let dof = block_dof(kind);
            if let Some(current_index) = self.base.block_index_for(*frame_id, kind) {
                // `block_index_for` is absolute in the AOM layout (poses are
                // the prefix), whereas the trial sidecars store states in a
                // state-local vector.  Keep this checked conversion at the
                // concrete view boundary; a pose prefix must never be read
                // as a state delta.
                let delta = match kind {
                    WindowBlockKind::Pose => self
                        .poses
                        .get(current_index)
                        .map(|pose| pose.linearized_delta.clone()),
                    WindowBlockKind::State => current_index
                        .checked_sub(self.base.poses.len())
                        .and_then(|local_index| self.states.get(local_index))
                        .and_then(|state| {
                            (state.linearized_delta.len() >= dof)
                                .then(|| state.linearized_delta.rows(0, dof).into_owned())
                        }),
                    WindowBlockKind::StatePose => current_index
                        .checked_sub(self.base.poses.len())
                        .and_then(|local_index| self.states.get(local_index))
                        .and_then(|state| {
                            (state.linearized_delta.len() >= dof)
                                .then(|| state.linearized_delta.rows(0, dof).into_owned())
                        }),
                };
                if let Some(delta) = delta {
                    if delta.len() == dof {
                        current_point.rows_mut(prior_offset, dof).copy_from(&delta);
                    }
                }
            }
            prior_offset += dof;
        }
        current_point
    }

    fn block_pose(&self, index: usize) -> Option<SE3> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => {
                let offset = self.base.pose_offset(index);
                let vector = self.chart.rows(offset, POSE_DOF);
                let mut pose = decode_pose_with_mode(&vector, self.base.scalar_mode);
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.poses[index].pose.rotation,
                    )
                {
                    pose.rotation = self.poses[index].pose.rotation.clone();
                }
                Some(pose)
            }
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                let offset = self.base.state_offset(state_index);
                let vector = self.chart.rows(offset, NAV_STATE_DOF);
                let mut pose = decode_state_with_mode(&vector, self.base.scalar_mode).imu_to_world;
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    pose.rotation = self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(pose)
            }
        }
    }

    fn block_nav_value(&self, index: usize) -> Option<BasaltNavState> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.block_pose(index)?,
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                let offset = self.base.state_offset(state_index);
                let vector = self.chart.rows(offset, NAV_STATE_DOF);
                let mut nav = decode_state_with_mode(&vector, self.base.scalar_mode);
                if self.base.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    nav.imu_to_world.rotation =
                        self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(nav)
            }
        }
    }

    fn block_nav_visual_current_value(&self, index: usize) -> Option<BasaltNavState> {
        if self.base.scalar_mode == ScalarMode::UpstreamF32
            && self.base.block_kind(index) == Some(WindowBlockKind::State)
        {
            let state_index = index.checked_sub(self.base.poses.len())?;
            let state_sidecar = self.states.get(state_index)?;
            if !state_sidecar.linearized {
                return Some(state_sidecar.linearized_nav.clone());
            }
            if state_sidecar.linearized_delta.len() >= POSE_DOF {
                let mut current = state_sidecar.linearized_nav.clone();
                current.imu_to_world =
                    visual_state_pose_with_mode(state_sidecar, self.base.scalar_mode);
                return Some(current);
            }
        }
        self.block_nav_value(index)
    }
}

impl WindowValueView for WindowTrialView<'_, '_> {
    fn base(&self) -> &WindowProblem {
        self.base
    }

    fn chart(&self) -> &DVector<f64> {
        self.chart
    }

    fn block_nav(&self, index: usize) -> Option<BasaltNavState> {
        self.block_nav_value(index)
    }

    fn block_nav_visual_current(&self, index: usize) -> Option<BasaltNavState> {
        self.block_nav_visual_current_value(index)
    }

    fn block_nav_linearized(&self, index: usize) -> Option<BasaltNavState> {
        match self.base.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.poses.get(index)?.linearized_pose.clone(),
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.base.poses.len())?;
                Some(self.states.get(state_index)?.linearized_nav.clone())
            }
        }
    }

    fn landmark_parameter(&self, index: usize) -> Option<InverseDistanceLandmark> {
        let landmark = self.base.landmarks.get(index)?;
        let anchor_frame_id = self.base.block_frame_id(landmark.anchor_state_index)?;
        let value = self.landmarks.get(index)?;
        Some(InverseDistanceLandmark {
            anchor_pose: anchor_frame_id,
            anchor_camera_id: landmark.anchor_camera_id,
            direction: value.direction,
            inverse_distance: value.inverse_distance,
        })
    }

    fn state_is_linearized(&self, index: usize) -> Option<bool> {
        self.states.get(index).map(|state| state.linearized)
    }

    fn state_linearized_nav(&self, index: usize) -> Option<BasaltNavState> {
        self.states
            .get(index)
            .map(|state| state.linearized_nav.clone())
    }

    fn linearized_delta(&self, kind: WindowBlockKind, index: usize) -> Option<DVector<f64>> {
        match kind {
            WindowBlockKind::Pose => self
                .poses
                .get(index)
                .map(|pose| pose.linearized_delta.clone()),
            WindowBlockKind::State => self
                .states
                .get(index)
                .map(|state| state.linearized_delta.clone()),
            WindowBlockKind::StatePose => self.states.get(index).and_then(|state| {
                (state.linearized_delta.len() >= POSE_DOF)
                    .then(|| state.linearized_delta.rows(0, POSE_DOF).into_owned())
            }),
        }
    }

    fn prior_current_point(&self, prior: &WindowPrior) -> DVector<f64> {
        if let (Some(base_prior), Some(delta)) = (&self.base.prior, &self.prior_fej_delta) {
            if std::ptr::eq(base_prior, prior) && delta.len() == prior.fej_point.len() {
                return delta.clone();
            }
        }
        self.compute_prior_current_point(prior)
    }
}

impl WindowProblem {
    /// Shared candidate objective. Pose/landmark accessors refer to either the
    /// owned candidate or the prepared view; never to base linearization rows.
    fn trial_objective_f32(
        &self,
        nav: impl Fn(usize, bool) -> Option<BasaltNavState> + Sync,
        parameter: impl Fn(usize) -> Option<InverseDistanceLandmark> + Sync,
        prior_cost: f64,
    ) -> Result<f64, LmFailure> {
        let ranks = self
            .trial_host_order
            .iter()
            .enumerate()
            .map(|(rank, &key)| (key, rank))
            .collect::<BTreeMap<_, _>>();
        if ranks.len() != self.trial_host_order.len() {
            return Err(LmFailure::LinearSolve);
        }
        let mut observations = BTreeMap::new();
        for (index, landmark) in self.landmarks.iter().enumerate() {
            if landmark.observations.is_empty() {
                continue;
            }
            let host = (
                block_timestamp(self, landmark.anchor_state_index).ok_or(LmFailure::LinearSolve)?
                    as u64,
                landmark.anchor_camera_id,
            );
            let rank = *ranks.get(&host).ok_or(LmFailure::LinearSolve)?;
            for observation in &landmark.observations {
                let target = (
                    block_timestamp(self, observation.state_index).ok_or(LmFailure::LinearSolve)?
                        as u64,
                    observation.camera_id,
                );
                if observations
                    .insert((rank, target, landmark.track_id), (index, observation))
                    .is_some()
                {
                    return Err(LmFailure::LinearSolve);
                }
            }
        }
        // Each observation's reprojection cost depends only on its own
        // landmark/host/target data (all read through `self`, `nav`, and
        // `parameter`, never through another observation's result), so
        // every observation's cost is computed in parallel below. The
        // items are collected from `observations` (a `BTreeMap`, so already
        // in the same deterministic `(rank, target, track_id)` key order
        // the serial loop iterated in) into a `Vec` first specifically so
        // that order survives into the results `Vec` -- `par_iter().map()`
        // preserves input order -- and the fold immediately afterward adds
        // each `Some` cost into `visual` serially, in that same order, via
        // the same `+=` the loop used. A failing item's error is still
        // surfaced at the same position in that order (via `cost?` before
        // the corresponding `visual +=` would have run), matching the
        // original loop's early-return exactly.
        let ordered_observations: Vec<_> = observations.into_iter().collect();
        let observation_costs: Vec<Result<Option<f32>, LmFailure>> = ordered_observations
            .par_iter()
            .map(|((rank, target, _), (index, observation))| {
                let landmark = &self.landmarks[*index];
                let host = self.trial_host_order[*rank];
                let from = nav(landmark.anchor_state_index, true).ok_or(LmFailure::LinearSolve)?;
                let to = nav(observation.state_index, true).ok_or(LmFailure::LinearSolve)?;
                let point = parameter(*index).ok_or(LmFailure::LinearSolve)?;
                let matrix = super::aom::upstream_trial_transform_f32(
                    &from.imu_to_world,
                    &to.imu_to_world,
                    &self.camera_to_imu(host.1).ok_or(LmFailure::LinearSolve)?,
                    &self.camera_to_imu(target.1).ok_or(LmFailure::LinearSolve)?,
                    host == *target,
                );
                let result = super::aom::upstream_trial_observation_f32(
                    self.camera_model(target.1).ok_or(LmFailure::LinearSolve)?,
                    matrix,
                    Vector3::new(
                        point.direction.xy.x as f32,
                        point.direction.xy.y as f32,
                        point.inverse_distance as f32,
                    ),
                    nalgebra::Vector2::new(observation.pixel.x as f32, observation.pixel.y as f32),
                    FactorConfig::default(),
                );
                Ok(result.map(|(_, cost)| cost))
            })
            .collect();
        let mut visual = 0.0_f32;
        for cost in observation_costs {
            if let Some(cost) = cost? {
                visual += cost;
            }
        }
        let mut links = self.imu_links.iter().collect::<Vec<_>>();
        links.sort_by_key(|link| {
            self.states
                .get(link.from_index)
                .map(|state| state.timestamp_ns)
        });
        let (mut imu, mut bg, mut ba) = (0.0_f32, 0.0_f32, 0.0_f32);
        for link in links {
            if link.delta.delta_time == 0.0 {
                continue;
            }
            let from_state = self
                .states
                .get(link.from_index)
                .ok_or(LmFailure::LinearSolve)?;
            let to_state = self
                .states
                .get(link.to_index)
                .ok_or(LmFailure::LinearSolve)?;
            let from =
                nav(self.poses.len() + link.from_index, false).ok_or(LmFailure::LinearSolve)?;
            let to = nav(self.poses.len() + link.to_index, false).ok_or(LmFailure::LinearSolve)?;
            let raw = crate::imu::trial_preintegration_residual_f32(
                &from,
                &to,
                &link.delta,
                self.gravity_world,
            );
            let sqrt = crate::imu::sqrt_information_f32(&link.delta.covariance.map(|v| v as f32))
                .map_err(|_| LmFailure::NonFinite)?;
            imu += crate::imu::trial_imu_quadratic_stages_f32(&sqrt, &raw).2;
            let dt_ns = to_state
                .timestamp_ns
                .checked_sub(from_state.timestamp_ns)
                .ok_or(LmFailure::LinearSolve)?;
            let (gyro, accel) =
                crate::imu::trial_bias_cost_f32(&from, &to, dt_ns, self.bias_walk_noise)
                    .map_err(|_| LmFailure::LinearSolve)?;
            bg += gyro;
            ba += accel;
        }
        let cost = (visual + ((imu + bg) + ba)) + prior_cost as f32;
        if cost.is_finite() {
            Ok(cost as f64)
        } else {
            Err(LmFailure::NonFinite)
        }
    }

    fn owned_trial_objective(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        if self.scalar_mode != ScalarMode::UpstreamF32 {
            return self.cost(state);
        }
        let prior = self.prior.as_ref().map_or_else(
            || {
                self.prior_factors(state)
                    .first()
                    .map_or(0.0, |factor| factor.objective_cost)
            },
            |prior| self.marginal_prior_reduced_cost(state, prior),
        );
        self.trial_objective_f32(
            |index, visual| {
                if visual {
                    self.block_nav_visual_current(state, index)
                } else {
                    self.block_nav(state, index)
                }
            },
            |index| {
                let landmark = self.landmarks.get(index)?;
                Some(landmark.parameter(self.block_frame_id(landmark.anchor_state_index)?))
            },
            prior,
        )
    }

    fn layout(&self) -> WindowLayout {
        WindowLayout::new(self.poses.len(), self.states.len())
    }

    pub fn pose_offset(&self, index: usize) -> usize {
        self.layout()
            .offset_for_pose(index)
            .expect("pose index is outside the active AOM layout")
    }

    pub fn state_offset(&self, index: usize) -> usize {
        self.layout()
            .offset_for_state(index)
            .expect("state index is outside the active AOM layout")
    }

    pub fn state_dof(&self) -> usize {
        self.layout().state_dof()
    }

    pub fn initial_state(&self) -> DVector<f64> {
        flatten_blocks_with_mode(&self.poses, &self.states, self.scalar_mode)
    }

    fn lm_trial_binding(&self, state: &DVector<f64>, step: &DVector<f64>) -> LmTrialBinding {
        LmTrialBinding::new(
            self as *const Self as usize,
            lm_token_window_generation(self),
            self.scalar_mode,
            lm_trial_vector_fingerprint(state),
            lm_trial_vector_fingerprint(step),
        )
    }

    /// Validate and expand the clean reducer's mapped landmark recovery into
    /// the window's full landmark order.  This consumes the opaque
    /// preparation before any trial-side mutation or view construction, so a
    /// stale, duplicated, or mismapped entry fails without changing state.
    fn landmark_steps_from_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        preparation: LmTrialPreparation,
    ) -> Result<Vec<Option<Vector3<f64>>>, LmFailure> {
        let (tolerance_bits, state_fingerprint, step_fingerprint, mapped) =
            preparation.take_landmark_steps();
        if tolerance_bits != 1e-10_f64.to_bits()
            || state_fingerprint != lm_trial_vector_fingerprint(state)
            || step_fingerprint != lm_trial_vector_fingerprint(step)
            || mapped.len() > self.landmarks.len()
        {
            return Err(LmFailure::LinearSolve);
        }

        let mut steps = vec![None; self.landmarks.len()];
        let mut seen = vec![false; self.landmarks.len()];
        for (landmark_index, track_id, step) in mapped {
            let Some(landmark) = self.landmarks.get(landmark_index) else {
                return Err(LmFailure::LinearSolve);
            };
            if seen[landmark_index] || landmark.track_id != track_id {
                return Err(LmFailure::LinearSolve);
            }
            if step.is_some_and(|value| value.iter().any(|component| !component.is_finite())) {
                return Err(LmFailure::NonFinite);
            }
            seen[landmark_index] = true;
            steps[landmark_index] = step;
        }
        Ok(steps)
    }

    /// Linearize an already-solved window without running another LM pass.
    /// Marginalization uses this on the pre-shift AOM (which intentionally
    /// excludes the newest state) so its row ordering and FEJ point are
    /// derived from the same factor builder as the solver.
    pub fn linearize_snapshot(
        &self,
        state: &DVector<f64>,
    ) -> Result<Vec<WhitenedFactorRowStack>, LmFailure> {
        self.linearize(state)
            .map(|linearization| linearization.factors)
    }

    fn block_kind(&self, index: usize) -> Option<WindowBlockKind> {
        if index < self.poses.len() {
            Some(WindowBlockKind::Pose)
        } else if index < self.poses.len() + self.states.len() {
            Some(WindowBlockKind::State)
        } else {
            None
        }
    }

    pub(crate) fn block_frame_id(&self, index: usize) -> Option<u64> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(self.poses.get(index)?.frame_id),
            WindowBlockKind::State | WindowBlockKind::StatePose => self
                .states
                .get(index - self.poses.len())
                .map(|state| state.frame_id),
        }
    }

    fn block_pose(&self, state: &DVector<f64>, index: usize) -> Option<SE3> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => {
                let offset = self.pose_offset(index);
                let vector = state.rows(offset, POSE_DOF);
                let mut pose = decode_pose_with_mode(&vector, self.scalar_mode);
                // Upstream owns the pose as a full Sophus quaternion.  The
                // 15-column solver chart carries only the local SO(3) xyz
                // lanes, so reconstructing w from those lanes here loses the
                // last product/rounding boundary.  Keep the full quaternion
                // in the problem's pose table and use it whenever the chart
                // still names that same pose.
                if self.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.poses[index].pose.rotation,
                    )
                {
                    pose.rotation = self.poses[index].pose.rotation.clone();
                }
                Some(pose)
            }
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let offset = self.state_offset(index - self.poses.len());
                let state_index = index - self.poses.len();
                let vector = state.rows(offset, NAV_STATE_DOF);
                let mut pose = decode_state_with_mode(&vector, self.scalar_mode).imu_to_world;
                if self.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    pose.rotation = self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(pose)
            }
        }
    }

    fn block_nav(&self, state: &DVector<f64>, index: usize) -> Option<BasaltNavState> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.block_pose(state, index)?,
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let offset = self.state_offset(index - self.poses.len());
                let state_index = index - self.poses.len();
                let vector = state.rows(offset, NAV_STATE_DOF);
                let mut nav = decode_state_with_mode(&vector, self.scalar_mode);
                if self.scalar_mode == ScalarMode::UpstreamF32
                    && quaternion_xyz_matches(
                        &[vector[3], vector[4], vector[5]],
                        &self.states[state_index].nav.imu_to_world.rotation,
                    )
                {
                    nav.imu_to_world.rotation =
                        self.states[state_index].nav.imu_to_world.rotation.clone();
                }
                Some(nav)
            }
        }
    }

    /// Return the value-side pose used by Basalt's
    /// `PoseStateWithLin::getPose()` for a navigation state.  The temporary
    /// `PoseStateWithLin` made by `getPoseStateWithLin()` copies
    /// `state_linearized.T_w_i` into `pose_linearized`; for a non-linearized
    /// state `getPose()` returns that raw pose, while a linearized state uses
    /// the separately reconstructed current pose.  In particular, the zero
    /// tangent applied by the temporary constructor only initializes its
    /// unused `T_w_i_current` for the non-linearized case; it must not replace
    /// the raw `pose_linearized` quaternion returned by `getPose()`.
    fn block_nav_visual_current(
        &self,
        state: &DVector<f64>,
        index: usize,
    ) -> Option<BasaltNavState> {
        if self.scalar_mode == ScalarMode::UpstreamF32
            && self.block_kind(index) == Some(WindowBlockKind::State)
        {
            let state_index = index.checked_sub(self.poses.len())?;
            let state_sidecar = self.states.get(state_index)?;
            if !state_sidecar.linearized {
                // Native `PoseVelBiasStateWithLin::getState()` returns
                // `state_linearized` while this flag is false, and the
                // temporary PoseStateWithLin then exposes its raw
                // `pose_linearized.T_w_i` through getPose().
                return Some(state_sidecar.linearized_nav.clone());
            }

            // `getPoseStateWithLin()` materializes a temporary
            // `PoseStateWithLin` from the full navigation state.  Its
            // constructor starts at `state_linearized.T_w_i`, then always
            // calls `PoseState::incPose(delta.head<6>(), T_w_i_current)`;
            // this call is not skipped when delta is zero.  Reconstruct the
            // value-side pose from that same frozen pose and f32 delta rather
            // than returning the stored `state_current`/compact chart.  In
            // particular, the zero SO3 increment still crosses Sophus'
            // float quaternion product/normalization boundary.
            if state_sidecar.linearized_delta.len() >= POSE_DOF {
                let base_pose = &state_sidecar.linearized_nav.imu_to_world;
                let mut current = state_sidecar.linearized_nav.clone();
                let delta = &state_sidecar.linearized_delta;
                current.imu_to_world.translation = vec3_f64(
                    vec3_f32(base_pose.translation)
                        + Vector3::from_iterator(
                            delta.rows(0, 3).iter().copied().map(|value| value as f32),
                        ),
                );
                current.imu_to_world.rotation = quat_f64_from_f32(&sophus_so3_product(
                    so3_exp_f32(Vector3::from_iterator(
                        delta.rows(3, 3).iter().copied().map(|value| value as f32),
                    )),
                    quat_f32_from_f64(&base_pose.rotation),
                ));
                return Some(current);
            }
        }
        self.block_nav(state, index)
    }

    /// Materialize the value-side pose that native
    /// `PoseStateWithLin::getPose()` exposes for the retained FEJ records.
    /// This is used only by the opt-in MargData diagnostic sidecar; factor
    /// evaluation continues through `block_nav_visual_current` above.
    pub(crate) fn diagnostic_effective_fej_poses(
        &self,
    ) -> (BTreeMap<i64, SE3>, BTreeMap<i64, SE3>) {
        let mut poses = BTreeMap::new();
        for pose in &self.poses {
            let effective = if self.scalar_mode == ScalarMode::UpstreamF32 {
                upstream_f32_pose_from_delta(
                    &pose.linearized_pose,
                    pose.linearized_delta.as_slice(),
                )
                .unwrap_or_else(|| pose.pose.clone())
            } else {
                pose.pose.clone()
            };
            poses.insert(pose.timestamp_ns, effective);
        }

        let mut states = BTreeMap::new();
        for state in &self.states {
            let effective = visual_state_pose_with_mode(state, self.scalar_mode);
            states.insert(state.timestamp_ns, effective);
        }
        (poses, states)
    }

    /// Return the pose/state used by Basalt's `getPoseLin()` for visual
    /// relative-pose Jacobians.  The value-side `T_t_h` is still evaluated
    /// from [`Self::block_nav`] at the current state; only a block marked
    /// `linearized` stays on its frozen FEJ point.
    fn visual_block_is_linearized(&self, index: usize) -> Option<bool> {
        if index < self.poses.len() {
            // Pose-only blocks have already been retained by marginalization.
            Some(true)
        } else {
            self.states
                .get(index.checked_sub(self.poses.len())?)
                .map(|s| s.linearized)
        }
    }

    fn block_nav_linearized(&self, _state: &DVector<f64>, index: usize) -> Option<BasaltNavState> {
        match self.block_kind(index)? {
            WindowBlockKind::Pose => Some(BasaltNavState {
                imu_to_world: self.poses.get(index)?.linearized_pose.clone(),
                ..BasaltNavState::default()
            }),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                let state_index = index.checked_sub(self.poses.len())?;
                let state = self.states.get(state_index)?;
                // `getPoseLin()` always names the frozen
                // `state_linearized.T_w_i`, including for a non-linearized
                // state whose value-side temporary receives a zero-inc
                // normalization.  The two paths intentionally differ by
                // those final quaternion bits.
                Some(state.linearized_nav.clone())
            }
        }
    }

    fn upstream_f32_trial_pose(&self, step: &DVector<f64>, index: usize) -> Option<SE3> {
        let pose = self.poses.get(index)?;
        let offset = self.pose_offset(index);
        let end = offset.checked_add(POSE_DOF)?;
        let delta = upstream_f32_accumulated_delta(
            pose.linearized_delta.as_slice(),
            step.as_slice().get(offset..end)?,
        )?;
        upstream_f32_pose_from_delta(&pose.linearized_pose, delta.as_slice())
    }

    fn upstream_f32_apply_state_trial(
        &self,
        index: usize,
        nav: &mut BasaltNavState,
        step: &DVector<f64>,
        trace_frame_id: Option<u64>,
    ) -> bool {
        let state = match self.states.get(index) {
            Some(state) => state,
            None => return false,
        };
        let offset = self.state_offset(index);
        let end = match offset.checked_add(NAV_STATE_DOF) {
            Some(end) => end,
            None => return false,
        };
        let increment = match if state.linearized && state.linearized_delta.len() == NAV_STATE_DOF {
            upstream_f32_accumulated_delta(
                state.linearized_delta.as_slice(),
                match step.as_slice().get(offset..end) {
                    Some(increment) => increment,
                    None => return false,
                },
            )
        } else {
            Some(DVector::from_column_slice(
                match step.as_slice().get(offset..end) {
                    Some(increment) => increment,
                    None => return false,
                },
            ))
        } {
            Some(increment) => increment,
            None => return false,
        };
        if state.linearized && state.linearized_delta.len() == NAV_STATE_DOF {
            *nav = state.linearized_nav.clone();
        }
        upstream_f32_apply_nav_delta(nav, increment.as_slice(), trace_frame_id)
    }

    /// Apply one solver increment while retaining the full pose objects that
    /// own the upstream quaternion payload.  `state` remains a compact
    /// 15-column chart; these vectors are the sidecar used by factor
    /// evaluation and by the trial/accept paths.
    fn apply_step_full(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
    ) -> (Vec<SE3>, Vec<BasaltNavState>) {
        let mut updated_poses = self
            .poses
            .iter()
            .enumerate()
            .map(|(index, _)| self.block_pose(state, index).unwrap_or_default())
            .collect::<Vec<_>>();
        for (index, pose) in updated_poses.iter_mut().enumerate() {
            let offset = self.pose_offset(index);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                // Pose-only keyframes are the Rust counterpart of Basalt's
                // `PoseStateWithLin`.  Keep this reconstruction shared with
                // the eager trial view so acceptance and trial costs cross
                // the same f32 quaternion boundary.
                if let Some(trial_pose) = self.upstream_f32_trial_pose(step, index) {
                    *pose = trial_pose;
                }
            } else {
                pose.translation += Vector3::from_iterator(step.rows(offset, 3).iter().copied());
                pose.rotation = UnitQuaternion::from_scaled_axis(Vector3::from_iterator(
                    step.rows(offset + 3, 3).iter().copied(),
                )) * pose.rotation;
            }
        }

        let mut updated = (0..self.states.len())
            .map(|index| {
                self.block_nav(state, self.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        for (index, nav) in updated.iter_mut().enumerate() {
            let offset = self.state_offset(index);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                let _ = self.upstream_f32_apply_state_trial(
                    index,
                    nav,
                    step,
                    Some(self.states[index].frame_id),
                );
            } else {
                nav.imu_to_world.translation +=
                    Vector3::from_iterator(step.rows(offset, 3).iter().copied());
                nav.imu_to_world.rotation = UnitQuaternion::from_scaled_axis(
                    Vector3::from_iterator(step.rows(offset + 3, 3).iter().copied()),
                ) * nav.imu_to_world.rotation;
                nav.velocity_world_m_s +=
                    Vector3::from_iterator(step.rows(offset + 6, 3).iter().copied());
                nav.gyro_bias_rad_s +=
                    Vector3::from_iterator(step.rows(offset + 9, 3).iter().copied());
                nav.accel_bias_m_s2 +=
                    Vector3::from_iterator(step.rows(offset + 12, 3).iter().copied());
            }
        }
        (updated_poses, updated)
    }

    fn block_index_for(&self, frame_id: u64, kind: WindowBlockKind) -> Option<usize> {
        match kind {
            WindowBlockKind::Pose => self.poses.iter().position(|pose| pose.frame_id == frame_id),
            WindowBlockKind::State | WindowBlockKind::StatePose => self
                .states
                .iter()
                .position(|state| state.frame_id == frame_id)
                .map(|index| self.poses.len() + index),
        }
    }

    fn block_offset(&self, index: usize, kind: WindowBlockKind) -> usize {
        match kind {
            WindowBlockKind::Pose => self.pose_offset(index),
            WindowBlockKind::State | WindowBlockKind::StatePose => {
                self.state_offset(index - self.poses.len())
            }
        }
    }

    fn camera_model(&self, camera_id: u16) -> Option<&DoubleSphereCamera> {
        if let Some(camera) = self.cameras.get(camera_id as usize) {
            Some(camera)
        } else if camera_id == 0 && self.cameras.is_empty() {
            // Preserve the original single-camera construction contract used
            // by the low-level unit tests and downstream adapters.
            Some(&self.camera)
        } else {
            None
        }
    }

    fn camera_to_imu(&self, camera_id: u16) -> Option<SE3> {
        if let Some(transform) = self.t_imu_cam.get(camera_id as usize) {
            Some(transform.clone())
        } else if camera_id == 0 && self.t_imu_cam.is_empty() {
            Some(SE3::identity())
        } else {
            None
        }
    }

    pub fn solve(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_with_options(initial, config, true, true, &mut timing)
    }

    pub(crate) fn solve_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        self.solve_with_options(initial, config, true, true, timing)
    }

    /// Solve without retaining a duplicate final factor snapshot.
    ///
    /// The LM pass already owns the accepted-state cost and all factors needed
    /// to make that decision.  The final factor snapshot is only required by
    /// the diagnostic/retained-MargData path (`last_rows`); explicit
    /// no-MargData callers do not consume it.  Avoiding that post-solve
    /// relinearization removes a full visual/IMU factor build while leaving
    /// the solve, writeback, prior update, and window lifecycle unchanged.
    pub fn solve_without_factors(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_with_options(initial, config, false, true, &mut timing)
    }

    pub(crate) fn solve_without_factors_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        self.solve_with_options(initial, config, false, true, timing)
    }

    /// Solve without retaining factors or building the diagnostic prepass.
    ///
    /// This is reserved for the explicit no-MargData/no-trace estimator path.
    /// Any Basalt probe or compatibility environment variable selects the
    /// existing diagnostic implementation so instrumentation keeps observing
    /// the same boundaries as the retained-output path.
    pub(crate) fn solve_without_diagnostics(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut timing = TimingBreakdown::from_env();
        self.solve_without_diagnostics_with_timing(initial, config, &mut timing)
    }

    pub(crate) fn solve_without_diagnostics_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        if diagnostic_env_active() {
            return self.solve_without_factors_with_timing(initial, config, timing);
        }
        self.solve_with_options(initial, config, false, false, timing)
    }

    /// Solve the active window for its accepted state **and** the final factor
    /// snapshot, while skipping the diagnostic prepass and per-trial
    /// diagnostic payloads.
    ///
    /// This is the online-mapper/MargData fast path.  It has the same LM math
    /// as [`Self::solve_without_diagnostics_with_timing`] (the compact f32
    /// preparation path) but, like [`Self::solve_with_timing`], retains the
    /// post-solve factor snapshot that MargData consumes.  The retained
    /// diagnostic *row counters* are recovered from that same final
    /// linearization, so `emit_marg`'s metadata is populated without the
    /// duplicate pre-solve linearization `initial_window_diagnostics` performs.
    ///
    /// Any Basalt probe or compatibility environment variable falls back to
    /// the fully diagnostic implementation so instrumentation keeps observing
    /// the historical boundaries.
    pub(crate) fn solve_lean_with_factors_with_timing(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        if diagnostic_env_active() {
            return self.solve_with_timing(initial, config, timing);
        }
        self.solve_with_options(initial, config, true, false, timing)
    }

    fn solve_with_options(
        &mut self,
        initial: DVector<f64>,
        config: LmConfig,
        retain_factors: bool,
        retain_diagnostics: bool,
        timing: &mut TimingBreakdown,
    ) -> Result<WindowSolveResult, WindowSolveError> {
        let mut diagnostics = if retain_diagnostics {
            initial_window_diagnostics(self, &initial)
        } else {
            minimal_window_diagnostics(self, &initial)
        };
        if initial.len() != self.state_dof() {
            let message = format!(
                "window state has {} columns, got initial vector of length {}",
                self.state_dof(),
                initial.len()
            );
            diagnostics.status = "invalid_initial_state".into();
            diagnostics.failure = Some(message.clone());
            return Err(WindowSolveError {
                message,
                diagnostics,
            });
        }
        let landmark_before = retain_diagnostics.then(|| self.landmarks.clone());
        let first = match if retain_diagnostics {
            solve_lm_with_timing(self, initial, config, true, true, timing)
        } else {
            solve_lm_with_timing(self, initial, config, false, false, timing)
        } {
            Ok(result) => result,
            Err(error) => {
                let message = format!("window LM: {error:?}");
                diagnostics.status = "first_lm_failed".into();
                diagnostics.failure = Some(message.clone());
                if retain_diagnostics {
                    diagnostics.lm.push(LmRunDiagnostics {
                        pass: "first".into(),
                        iterations: 0,
                        lambda: config.lambda_initial,
                        initial_cost: f64::NAN,
                        final_cost: f64::NAN,
                        accepted: 0,
                        rejected: 0,
                        trace: Vec::new(),
                        failure: Some(format!("{error:?}")),
                    });
                }
                return Err(WindowSolveError {
                    message,
                    diagnostics,
                });
            }
        };
        if retain_diagnostics {
            diagnostics.lm.push(lm_run_diagnostics("first", &first));
            if let Some(landmark_before) = landmark_before {
                diagnostics.landmark_writeback = self
                    .landmarks
                    .iter()
                    .zip(&landmark_before)
                    .filter(|(after, before)| {
                        after.direction != before.direction
                            || after.inverse_distance != before.inverse_distance
                    })
                    .count();
            }
        }

        // Basalt performs one canonical LM solve per active window. Landmark
        // back-substitution updates the 3-D blocks after that solve; emit a
        // fresh factor snapshot for MargData, but do not run a second LM pass
        // over the same frame (which otherwise doubles stale-prior rejects).
        let state = first.state;
        let factors = if retain_factors {
            match self.linearize(&state) {
                Ok(linearization) => {
                    // The diagnostic prepass (`initial_window_diagnostics`) is
                    // skipped on the lean-with-factors path, but `emit_marg`
                    // still reads these four row counters from the packet
                    // metadata.  Derive them from the final linearization we
                    // already hold instead of paying for a second full
                    // pre-solve linearization.
                    if !retain_diagnostics {
                        fill_diagnostic_factor_rows(self, &mut diagnostics, &linearization.factors);
                    }
                    linearization.factors
                }
                Err(error) => {
                    let message = format!("window final linearization: {error:?}");
                    diagnostics.status = "final_linearization_failed".into();
                    diagnostics.failure = Some(message.clone());
                    return Err(WindowSolveError {
                        message,
                        diagnostics,
                    });
                }
            }
        } else {
            Vec::new()
        };
        // `solve_lm` maintains this value at the accepted state. Calling
        // `self.cost(&state)` here would rebuild every factor a second time
        // without changing the returned value.
        let cost = first.cost;
        diagnostics.status = "success".into();
        Ok(WindowSolveResult {
            state,
            factors,
            cost,
            iterations: first.iterations,
            diagnostics,
        })
    }

    fn landmark_steps(
        &self,
        state: &DVector<f64>,
        state_step: &DVector<f64>,
    ) -> Vec<Option<Vector3<f64>>> {
        let mut steps = Vec::with_capacity(self.landmarks.len());
        for landmark_index in 0..self.landmarks.len() {
            let Some(factor) = self.visual_factor(state, landmark_index) else {
                steps.push(None);
                continue;
            };
            let step = if self.scalar_mode == ScalarMode::UpstreamF32 {
                back_substitute_landmark_upstream_f32_with_track(
                    &factor,
                    state_step,
                    1e-10,
                    Some(self.landmarks[landmark_index].track_id),
                )
            } else {
                let reduced =
                    reduce_landmark_factors(std::slice::from_ref(&factor), state.len(), 1e-10);
                reduced
                    .back_substitution
                    .first()
                    .and_then(|data| back_substitute_landmark(data, state_step, 1e-10))
            };
            let Some(step) = step else {
                steps.push(None);
                continue;
            };
            if step.len() == 3 && step.iter().all(|value| value.is_finite()) {
                steps.push(Some(Vector3::new(step[0], step[1], step[2])));
            } else {
                steps.push(None);
            }
        }
        steps
    }

    fn apply_landmark_steps(&mut self, steps: &[Option<Vector3<f64>>]) {
        let anchor_frames = self
            .landmarks
            .iter()
            .map(|landmark| self.block_frame_id(landmark.anchor_state_index))
            .collect::<Vec<_>>();
        for (index, (landmark, increment)) in self.landmarks.iter_mut().zip(steps).enumerate() {
            let Some(increment) = increment else {
                continue;
            };
            let Some(anchor_frame_id) = anchor_frames.get(index).copied().flatten() else {
                continue;
            };
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                landmark.apply_increment_f32(*increment, anchor_frame_id);
            } else {
                landmark.apply_increment(*increment, anchor_frame_id);
            }
        }
    }

    fn trial_landmark_value(
        &self,
        index: usize,
        increment: Vector3<f64>,
    ) -> Option<TrialLandmarkValue> {
        let landmark = self.landmarks.get(index)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let mut parameter = landmark.parameter(anchor_frame_id);
        let applied = if self.scalar_mode == ScalarMode::UpstreamF32 {
            parameter.apply_increment_f32(increment)
        } else {
            parameter.apply_increment(increment)
        };
        applied.then_some(TrialLandmarkValue {
            direction: parameter.direction,
            inverse_distance: parameter.inverse_distance,
        })
    }

    fn visual_factor(
        &self,
        state: &DVector<f64>,
        landmark_index: usize,
    ) -> Option<WhitenedFactorRowStack> {
        let landmark = self.landmarks.get(landmark_index)?;
        let anchor_nav = self.block_nav_visual_current(state, landmark.anchor_state_index)?;
        let anchor_nav_fej = self.block_nav_linearized(state, landmark.anchor_state_index)?;
        let anchor_extrinsic = self.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let parameter = landmark.parameter(anchor_frame_id);
        // The native `LandmarkBlockAbsDynamic` allocates one two-row slot for
        // every stored observation.  When a pre-marginalization AOM has
        // truncated a suffix state, its `pose_lin_vec` entry is null, but the
        // corresponding zero row remains in the packed QR storage.  Keep
        // those holes in source order instead of filtering them out: dropping
        // them changes both the Householder schedule and the subsequent Q2
        // normal system.  In-range observations retain the historical
        // camera/projection filtering semantics.
        let block_count = self.poses.len() + self.states.len();
        let entries = landmark
            .observations
            .iter()
            .filter_map(|observation| {
                if observation.state_index >= block_count {
                    return Some((*observation, None));
                }
                let camera = self.camera_model(observation.camera_id)?;
                let target_extrinsic = self.camera_to_imu(observation.camera_id)?;
                let target_nav = self.block_nav_visual_current(state, observation.state_index)?;
                let target_nav_fej = self.block_nav_linearized(state, observation.state_index)?;
                let target_frame_id = self.block_frame_id(observation.state_index)?;
                // Basalt's absolute landmark block has no pose contribution
                // only for the exact host/target `TimeCamId` identity. A
                // same-timestamp stereo observation executes computeRelPose,
                // so both pose blocks are retained and scattered below.
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.visual_block_is_linearized(landmark.anchor_state_index)?
                            || self.visual_block_is_linearized(observation.state_index)?,
                        FactorConfig::default(),
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        FactorConfig::default(),
                    )
                };
                factor.map(|factor| (*observation, Some(factor)))
            })
            .collect::<Vec<_>>();
        if entries.is_empty() {
            return None;
        }
        let rows = entries
            .iter()
            .map(|(_, factor)| factor.as_ref().map_or(2, |factor| factor.residual.len()))
            .sum::<usize>();
        let mut state_jacobian = DMatrix::zeros(rows, self.state_dof());
        let mut landmark_jacobian = DMatrix::zeros(rows, 3);
        let mut residual = DVector::zeros(rows);
        let mut objective_cost = 0.0;
        let mut row = 0;
        for (observation, factor) in entries {
            let Some(factor) = factor else {
                // This is the native null `pose_lin_vec` row.  The matrices
                // and residual were zero-initialized, so only advance the
                // row cursor; no objective contribution is produced.
                row += 2;
                continue;
            };
            let anchor_offset = self.pose_offset_for_block(landmark.anchor_state_index);
            let target_offset = self.pose_offset_for_block(observation.state_index);
            // A same-timestamp stereo pair shares one navigation block.  The
            // two relative-pose derivatives are accumulated into that block
            // by the same sequential f32 stores as every other shared block.
            // The camera-to-camera transform is mathematically independent
            // of the common IMU pose, but the independently rounded native
            // products can leave a real residual in individual lanes.
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                // The pinned `LinearizationBase<float>` accumulates every
                // visual contribution directly into its float AOM blocks.
                // The compatibility row object is widened to f64 for the
                // public API, so using nalgebra's f64 `add_assign` here would
                // defer the rounding until after host/target contributions
                // have already been summed.  Round each scalar at the same
                // accumulation boundary before it reaches ABS_QR.
                for local_row in 0..factor.residual.len() {
                    for column in 0..6 {
                        let anchor_column = anchor_offset + column;
                        let target_column = target_offset + column;
                        let anchor = state_jacobian[(row + local_row, anchor_column)] as f32
                            + factor.anchor_pose_jacobian[(local_row, column)] as f32;
                        if anchor_column == target_column {
                            // Both TimeCamIds can name the same navigation
                            // block for same-timestamp stereo. Upstream's
                            // two `+=` stores are sequential float adds;
                            // retain that order instead of overwriting the
                            // anchor contribution with an algebraic zero.
                            let combined =
                                anchor + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = combined as f64;
                        } else {
                            let target = state_jacobian[(row + local_row, target_column)] as f32
                                + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = anchor as f64;
                            state_jacobian[(row + local_row, target_column)] = target as f64;
                        }
                    }
                }
            } else {
                state_jacobian
                    .view_mut((row, anchor_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.anchor_pose_jacobian);
                state_jacobian
                    .view_mut((row, target_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.target_pose_jacobian);
            }
            landmark_jacobian
                .view_mut((row, 0), (factor.residual.len(), 3))
                .copy_from(&factor.landmark_jacobian);
            residual
                .rows_mut(row, factor.residual.len())
                .copy_from(&factor.residual);
            if self.scalar_mode == ScalarMode::UpstreamF32 {
                objective_cost = (objective_cost as f32 + factor.objective_cost as f32) as f64;
            } else {
                objective_cost += factor.objective_cost;
            }
            row += factor.residual.len();
        }
        WhitenedFactorRowStack::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Visual,
        )
    }

    /// Trial-view counterpart of [`Self::visual_factor`].  The observation
    /// list and camera topology remain borrowed from the accepted problem;
    /// only the current/FEJ poses and landmark parameter come from `values`.
    /// Keeping this as a separate builder leaves the diagnostic/audit path on
    /// the original owned `WindowProblem` implementation.
    fn visual_factor_view<V: WindowValueView>(
        &self,
        values: &V,
        landmark_index: usize,
    ) -> Option<WhitenedFactorRowStack> {
        let base = values.base();
        let landmark = base.landmarks.get(landmark_index)?;
        let anchor_nav = values.block_nav_visual_current(landmark.anchor_state_index)?;
        let anchor_nav_fej = values.block_nav_linearized(landmark.anchor_state_index)?;
        let anchor_extrinsic = base.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = base.block_frame_id(landmark.anchor_state_index)?;
        let parameter = values.landmark_parameter(landmark_index)?;
        let valid = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index < base.poses.len() + base.states.len())
            .filter_map(|observation| {
                let camera = base.camera_model(observation.camera_id)?;
                let target_extrinsic = base.camera_to_imu(observation.camera_id)?;
                let target_nav = values.block_nav_visual_current(observation.state_index)?;
                let target_nav_fej = values.block_nav_linearized(observation.state_index)?;
                let target_frame_id = base.block_frame_id(observation.state_index)?;
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if base.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        base.visual_block_is_linearized(landmark.anchor_state_index)?
                            || base.visual_block_is_linearized(observation.state_index)?,
                        FactorConfig::default(),
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        FactorConfig::default(),
                    )
                };
                factor.map(|factor| (*observation, factor))
            })
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return None;
        }
        let rows = valid
            .iter()
            .map(|(_, factor)| factor.residual.len())
            .sum::<usize>();
        let mut state_jacobian = DMatrix::zeros(rows, base.state_dof());
        let mut landmark_jacobian = DMatrix::zeros(rows, 3);
        let mut residual = DVector::zeros(rows);
        let mut objective_cost = 0.0;
        let mut row = 0;
        for (observation, factor) in valid {
            let anchor_offset = base.pose_offset_for_block(landmark.anchor_state_index);
            let target_offset = base.pose_offset_for_block(observation.state_index);
            if base.scalar_mode == ScalarMode::UpstreamF32 {
                for local_row in 0..factor.residual.len() {
                    for column in 0..6 {
                        let anchor_column = anchor_offset + column;
                        let target_column = target_offset + column;
                        let anchor = state_jacobian[(row + local_row, anchor_column)] as f32
                            + factor.anchor_pose_jacobian[(local_row, column)] as f32;
                        if anchor_column == target_column {
                            let combined =
                                anchor + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = combined as f64;
                        } else {
                            let target = state_jacobian[(row + local_row, target_column)] as f32
                                + factor.target_pose_jacobian[(local_row, column)] as f32;
                            state_jacobian[(row + local_row, anchor_column)] = anchor as f64;
                            state_jacobian[(row + local_row, target_column)] = target as f64;
                        }
                    }
                }
            } else {
                state_jacobian
                    .view_mut((row, anchor_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.anchor_pose_jacobian);
                state_jacobian
                    .view_mut((row, target_offset), (factor.residual.len(), 6))
                    .add_assign(&factor.target_pose_jacobian);
            }
            landmark_jacobian
                .view_mut((row, 0), (factor.residual.len(), 3))
                .copy_from(&factor.landmark_jacobian);
            residual
                .rows_mut(row, factor.residual.len())
                .copy_from(&factor.residual);
            if base.scalar_mode == ScalarMode::UpstreamF32 {
                objective_cost = (objective_cost as f32 + factor.objective_cost as f32) as f64;
            } else {
                objective_cost += factor.objective_cost;
            }
            row += factor.residual.len();
        }
        WhitenedFactorRowStack::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Visual,
        )
    }

    /// Capture one grouped landmark factor plus its observation-to-row mapping
    /// for the opt-in numeric audit.  The aggregate factor comes directly from
    /// `visual_factor`, preserving the normal solver path; only the audit's
    /// per-observation records are rebuilt when the detail trace is enabled.
    fn visual_factor_with_audit(
        &self,
        state: &DVector<f64>,
        landmark_index: usize,
    ) -> Option<VisualLandmarkAudit> {
        let factor = self.visual_factor(state, landmark_index)?;
        let landmark = self.landmarks.get(landmark_index)?;
        let anchor_nav = self.block_nav_visual_current(state, landmark.anchor_state_index)?;
        let anchor_nav_fej = self.block_nav_linearized(state, landmark.anchor_state_index)?;
        let anchor_extrinsic = self.camera_to_imu(landmark.anchor_camera_id)?;
        let anchor_frame_id = self.block_frame_id(landmark.anchor_state_index)?;
        let parameter = landmark.parameter(anchor_frame_id);
        let valid = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index < self.poses.len() + self.states.len())
            .filter_map(|observation| {
                let camera = self.camera_model(observation.camera_id)?;
                let target_extrinsic = self.camera_to_imu(observation.camera_id)?;
                let target_nav = self.block_nav_visual_current(state, observation.state_index)?;
                let target_nav_fej = self.block_nav_linearized(state, observation.state_index)?;
                let target_frame_id = self.block_frame_id(observation.state_index)?;
                let same_timestamp = anchor_frame_id == target_frame_id;
                let same_time_cam_id =
                    same_timestamp && landmark.anchor_camera_id == observation.camera_id;
                let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
                    anchored_visual_reprojection_factor_f32_with_time_cam_fej_mode(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_nav_fej.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_nav_fej.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        self.visual_block_is_linearized(landmark.anchor_state_index)?
                            || self.visual_block_is_linearized(observation.state_index)?,
                        FactorConfig::default(),
                    )
                } else {
                    anchored_visual_reprojection_factor_with_time_cam(
                        camera,
                        &anchor_nav.imu_to_world,
                        &anchor_extrinsic,
                        &target_nav.imu_to_world,
                        &target_extrinsic,
                        &parameter,
                        observation.pixel,
                        same_timestamp,
                        same_time_cam_id,
                        FactorConfig::default(),
                    )
                };
                factor.map(|factor| (*observation, factor))
            })
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return None;
        }
        let rows = valid
            .iter()
            .map(|(_, factor)| factor.residual.len())
            .sum::<usize>();
        let observations = valid
            .iter()
            .map(|(observation, factor)| VisualObservationAudit {
                observation: *observation,
                factor: factor.clone(),
            })
            .collect::<Vec<_>>();
        let invalid_rows = landmark
            .observations
            .iter()
            .filter(|observation| observation.state_index >= self.poses.len() + self.states.len())
            .count()
            * 2;
        debug_assert_eq!(factor.rows(), rows + invalid_rows);
        Some(VisualLandmarkAudit {
            track_id: landmark.track_id,
            anchor_frame_id,
            anchor_camera_id: landmark.anchor_camera_id,
            direction: landmark.direction,
            inverse_distance: landmark.inverse_distance,
            factor,
            observations,
        })
    }

    fn pose_offset_for_block(&self, index: usize) -> usize {
        match self.block_kind(index) {
            Some(WindowBlockKind::Pose) => self.pose_offset(index),
            Some(WindowBlockKind::State | WindowBlockKind::StatePose) => {
                self.state_offset(index - self.poses.len())
            }
            None => self.state_dof(),
        }
    }

    fn prior_factors(&self, state: &DVector<f64>) -> Vec<WhitenedFactorRowStack> {
        let mut factors = Vec::new();
        if let Some(prior) = &self.prior {
            let mut columns = DMatrix::zeros(prior.rows(), self.state_dof());
            let mut prior_state_columns = Vec::with_capacity(prior.jacobian.ncols());
            let kinds = prior.kinds();
            let mut prior_offset = 0;
            for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
                let kind = kinds
                    .get(prior_index)
                    .copied()
                    .unwrap_or(WindowBlockKind::State);
                let Some(current_index) = self.block_index_for(*frame_id, kind) else {
                    prior_offset += block_dof(kind);
                    continue;
                };
                let dof = block_dof(kind);
                columns
                    .view_mut(
                        (0, self.block_offset(current_index, kind)),
                        (prior.rows(), dof),
                    )
                    .copy_from(&prior.jacobian.columns(prior_offset, dof));
                for local_column in 0..dof {
                    prior_state_columns.push(self.block_offset(current_index, kind) + local_column);
                }
                prior_offset += dof;
            }
            // The stored square-root prior is FEJ-linearized at `fej_point`.
            // Upstream's `get_dense_Q2Jp_Q2r` uses the residual
            //
            //     r_lin = H * delta + b,
            //
            // where `delta` is the current state minus the FEJ state.  The
            // plus sign is important: `b` is the residual at the original
            // linearization point, not a right-hand-side to subtract from the
            // current tangent.  Keeping this operation in one helper also
            // lets the reduced prior cost below use the same delta.
            let current_point = self.prior_current_point(state, prior);
            Self::emit_prior_residual_inputs(prior, &current_point, self.state_dof());
            let rhs = if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.nrows() == 21
                && prior.jacobian.ncols() == 21
                && prior.rhs.len() == 21
                && current_point.len() == 21
            {
                // Eigen's column-major GeneralMatrixVector path handles a
                // 21x21 prior as two Packet8 output blocks, one Packet4
                // block, and one scalar row.  The packet rows accumulate
                // every depth with FMA, then fuse the existing mld.b value
                // in the final pmadd store; row 20 uses separate scalar
                // multiply/add operations throughout.  The ordinary
                // nalgebra product is mathematically identical but rounds
                // four lanes differently at this solver boundary.
                Self::prior_rhs_col_major_21_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (27, 27)
                && prior.rhs.len() == 27
                && current_point.len() == 27
            {
                Self::prior_rhs_col_major_27_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (33, 33)
                && prior.rhs.len() == 33
                && current_point.len() == 33
            {
                Self::prior_rhs_col_major_33_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (39, 39)
                && prior.rhs.len() == 39
                && current_point.len() == 39
            {
                Self::prior_rhs_col_major_39_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (45, 45)
                && prior.rhs.len() == 45
                && current_point.len() == 45
            {
                Self::prior_rhs_col_major_45_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (51, 51)
                && prior.rhs.len() == 51
                && current_point.len() == 51
            {
                Self::prior_rhs_col_major_51_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if self.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (57, 57)
                && prior.rhs.len() == 57
                && current_point.len() == 57
            {
                Self::prior_rhs_col_major_57_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else {
                prior.rhs.clone() + &prior.jacobian * current_point
            };
            // Eigen's prior residual starts from a zero-initialized RHS
            // expression.  Keep exact zeros canonical at this producer
            // boundary; nalgebra's subtractive path can otherwise preserve
            // a negative zero, which becomes observable in Q2r although the
            // numerical residual is zero.  Do not alter nonzero values.
            let rhs = DVector::from_iterator(
                rhs.len(),
                rhs.iter().copied().map(|value| {
                    // The capture/solver owns this factor as f32 in the
                    // pinned path.  A tiny f64 carry can cast to -0.0;
                    // canonicalize that representational zero here too.
                    if (value as f32) == 0.0 {
                        0.0
                    } else {
                        value
                    }
                }),
            );
            if let Some(factor) = prior_factor(columns, rhs) {
                // The stored prior is compact in upstream.  Attach the
                // compact-to-global map only when every source column was
                // represented by an active block; incomplete windows keep
                // the legacy global product fallback.
                let factor = if prior_state_columns.len() == prior.jacobian.ncols() {
                    factor.with_prior_state_columns(prior_state_columns)
                } else {
                    factor
                };
                factors.push(factor);
            }
        }
        // The initial anchor row is already folded into the square-root prior
        // on the first window shift.  Re-adding it after that point would pin
        // the *current first state* to the original frame-0 coordinates even
        // after frame 0 has left the window, injecting a large false residual
        // and making the LM model reject every trial.  A fallback prior carries
        // its own gauge row, so the explicit row is needed only before a prior
        // exists.
        if self.prior.is_none() {
            if let Some(anchor) = &self.anchor_point {
                if anchor.len() == NAV_STATE_DOF && !self.states.is_empty() {
                    let mut jacobian = DMatrix::zeros(NAV_STATE_DOF, self.state_dof());
                    let pose_weight = self.initial_pose_weight.sqrt();
                    let accel_bias_weight = self.initial_accel_bias_weight.sqrt();
                    let gyro_bias_weight = self.initial_gyro_bias_weight.sqrt();
                    if !pose_weight.is_finite()
                        || !accel_bias_weight.is_finite()
                        || !gyro_bias_weight.is_finite()
                    {
                        return factors;
                    }
                    for axis in 0..3 {
                        jacobian[(axis, axis)] = pose_weight;
                    }
                    // Basalt fixes yaw while leaving roll/pitch to gravity.
                    jacobian[(5, 5)] = pose_weight;
                    // Preserve the upstream assignment in
                    // sqrt_keypoint_vio.cpp: vio_init_ba_weight is written to
                    // columns 9..11 and vio_init_bg_weight to columns 12..14.
                    for axis in 0..3 {
                        jacobian[(9 + axis, 9 + axis)] = accel_bias_weight;
                        jacobian[(12 + axis, 12 + axis)] = gyro_bias_weight;
                    }
                    let Some(current_nav) = self.block_nav(state, self.poses.len()) else {
                        return factors;
                    };
                    let anchor_nav = decode_state_with_mode(anchor, self.scalar_mode);
                    let current = if self.scalar_mode == ScalarMode::UpstreamF32 {
                        self.states
                            .first()
                            .filter(|state| state.linearized_delta.len() == NAV_STATE_DOF)
                            .map(|state| state.linearized_delta.clone())
                            .unwrap_or_else(|| local_state_difference(&current_nav, &anchor_nav))
                    } else {
                        local_state_difference(&current_nav, &anchor_nav)
                    };
                    let mut rhs = DVector::zeros(NAV_STATE_DOF);
                    for index in 0..NAV_STATE_DOF {
                        rhs[index] = if self.scalar_mode == ScalarMode::UpstreamF32 {
                            // The pinned anchor factor is built in Scalar=float:
                            // narrow both chart operands before the subtraction
                            // and apply the float diagonal weight there.  Doing
                            // the product in f64 and narrowing afterward moves
                            // the local-bias row by one ulp (Q2r row 90).
                            let weight = jacobian[(index, index)] as f32;
                            let delta = (current[index] as f32) - (anchor[index] as f32);
                            f64::from(weight * delta)
                        } else {
                            jacobian[(index, index)] * (current[index] - anchor[index])
                        };
                    }
                    // The anchor is the first Prior row at the frame-4
                    // boundary.  Match Eigen's f32 zero construction here as
                    // well: tiny f64 products must not survive as -0.0 after
                    // the factor is narrowed for Q2r.
                    for value in rhs.iter_mut() {
                        if (*value as f32) == 0.0 {
                            *value = 0.0;
                        }
                    }
                    if let Some(factor) = prior_factor(jacobian, rhs) {
                        factors.push(factor);
                    }
                }
            }
        }
        factors
    }

    fn prior_factors_view<V: WindowValueView>(&self, values: &V) -> Vec<WhitenedFactorRowStack> {
        let base = values.base();
        let mut factors = Vec::new();
        if let Some(prior) = &base.prior {
            let mut columns = DMatrix::zeros(prior.rows(), base.state_dof());
            let mut prior_state_columns = Vec::with_capacity(prior.jacobian.ncols());
            let kinds = prior.kinds();
            let mut prior_offset = 0;
            for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
                let kind = kinds
                    .get(prior_index)
                    .copied()
                    .unwrap_or(WindowBlockKind::State);
                let Some(current_index) = base.block_index_for(*frame_id, kind) else {
                    prior_offset += block_dof(kind);
                    continue;
                };
                let dof = block_dof(kind);
                let block_offset = base.block_offset(current_index, kind);
                columns
                    .view_mut((0, block_offset), (prior.rows(), dof))
                    .copy_from(&prior.jacobian.columns(prior_offset, dof));
                for local_column in 0..dof {
                    prior_state_columns.push(block_offset + local_column);
                }
                prior_offset += dof;
            }
            let current_point = values.prior_current_point(prior);
            Self::emit_prior_residual_inputs(prior, &current_point, base.state_dof());
            let rhs = if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.nrows() == 21
                && prior.jacobian.ncols() == 21
                && prior.rhs.len() == 21
                && current_point.len() == 21
            {
                Self::prior_rhs_col_major_21_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (27, 27)
                && prior.rhs.len() == 27
                && current_point.len() == 27
            {
                Self::prior_rhs_col_major_27_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (33, 33)
                && prior.rhs.len() == 33
                && current_point.len() == 33
            {
                Self::prior_rhs_col_major_33_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (39, 39)
                && prior.rhs.len() == 39
                && current_point.len() == 39
            {
                Self::prior_rhs_col_major_39_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (45, 45)
                && prior.rhs.len() == 45
                && current_point.len() == 45
            {
                Self::prior_rhs_col_major_45_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (51, 51)
                && prior.rhs.len() == 51
                && current_point.len() == 51
            {
                Self::prior_rhs_col_major_51_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else if base.scalar_mode == ScalarMode::UpstreamF32
                && prior.jacobian.shape() == (57, 57)
                && prior.rhs.len() == 57
                && current_point.len() == 57
            {
                Self::prior_rhs_col_major_57_f32(&prior.jacobian, &prior.rhs, &current_point)
            } else {
                prior.rhs.clone() + &prior.jacobian * current_point
            };
            let rhs = DVector::from_iterator(
                rhs.len(),
                rhs.iter()
                    .copied()
                    .map(|value| if (value as f32) == 0.0 { 0.0 } else { value }),
            );
            if let Some(factor) = prior_factor(columns, rhs) {
                let factor = if prior_state_columns.len() == prior.jacobian.ncols() {
                    factor.with_prior_state_columns(prior_state_columns)
                } else {
                    factor
                };
                factors.push(factor);
            }
        }
        if base.prior.is_none() {
            if let Some(anchor) = &base.anchor_point {
                if anchor.len() == NAV_STATE_DOF && !base.states.is_empty() {
                    let mut jacobian = DMatrix::zeros(NAV_STATE_DOF, base.state_dof());
                    let pose_weight = base.initial_pose_weight.sqrt();
                    let accel_bias_weight = base.initial_accel_bias_weight.sqrt();
                    let gyro_bias_weight = base.initial_gyro_bias_weight.sqrt();
                    if !pose_weight.is_finite()
                        || !accel_bias_weight.is_finite()
                        || !gyro_bias_weight.is_finite()
                    {
                        return factors;
                    }
                    for axis in 0..3 {
                        jacobian[(axis, axis)] = pose_weight;
                    }
                    jacobian[(5, 5)] = pose_weight;
                    for axis in 0..3 {
                        jacobian[(9 + axis, 9 + axis)] = accel_bias_weight;
                        jacobian[(12 + axis, 12 + axis)] = gyro_bias_weight;
                    }
                    let Some(current_nav) = values.block_nav(base.poses.len()) else {
                        return factors;
                    };
                    let anchor_nav = decode_state_with_mode(anchor, base.scalar_mode);
                    let current = if base.scalar_mode == ScalarMode::UpstreamF32 {
                        values
                            .linearized_delta(WindowBlockKind::State, 0)
                            .filter(|delta| delta.len() == NAV_STATE_DOF)
                            .unwrap_or_else(|| local_state_difference(&current_nav, &anchor_nav))
                    } else {
                        local_state_difference(&current_nav, &anchor_nav)
                    };
                    let mut rhs = DVector::zeros(NAV_STATE_DOF);
                    for index in 0..NAV_STATE_DOF {
                        rhs[index] = if base.scalar_mode == ScalarMode::UpstreamF32 {
                            let weight = jacobian[(index, index)] as f32;
                            let delta = (current[index] as f32) - (anchor[index] as f32);
                            f64::from(weight * delta)
                        } else {
                            jacobian[(index, index)] * (current[index] - anchor[index])
                        };
                    }
                    for value in rhs.iter_mut() {
                        if (*value as f32) == 0.0 {
                            *value = 0.0;
                        }
                    }
                    if let Some(factor) = prior_factor(jacobian, rhs) {
                        factors.push(factor);
                    }
                }
            }
        }
        factors
    }

    fn marginal_prior_reduced_cost_view<V: WindowValueView>(
        &self,
        values: &V,
        prior: &WindowPrior,
    ) -> f64 {
        upstream_marginal_prior_error(
            &prior.jacobian,
            &prior.rhs,
            &values.prior_current_point(prior),
            values.base().scalar_mode,
        )
    }

    fn emit_prior_residual_inputs(
        prior: &WindowPrior,
        current_point: &DVector<f64>,
        state_dof: usize,
    ) {
        let Some(path) = diagnostic_env_snapshot().diagnostic_prior_input.as_ref() else {
            return;
        };
        let to_bits = |values: &[f64]| -> Vec<String> {
            values
                .iter()
                .map(|v| format!("{:08x}", (*v as f32).to_bits()))
                .collect()
        };
        let record = json!({
            "schema": "basalt.prior_residual_inputs.v1",
            "global_cols": state_dof,
            "rows": prior.jacobian.nrows(),
            "cols": prior.jacobian.ncols(),
            "jacobian_bits": to_bits(prior.jacobian.as_slice()),
            "stored_rhs_bits": to_bits(prior.rhs.as_slice()),
            "delta_bits": to_bits(current_point.as_slice()),
        });
        if let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.with_extension("producer.jsonl"))
        {
            let _ = writeln!(file, "{}", record);
        }
    }

    /// Evaluate `mld.b + mld.H * delta` with Eigen's column-major dynamic
    /// GEMV schedule for the active 21x21 marginal prior.  Rows 0..19 use
    /// Packet8/Packet4 lanes with a fused multiply-add for each depth and a
    /// fused final add into the stored RHS.  The last row is the scalar
    /// remainder and uses ordinary multiply/add operations for both its
    /// depth reduction and final update.
    fn prior_rhs_col_major_21_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        current_point: &DVector<f64>,
    ) -> DVector<f64> {
        debug_assert_eq!(jacobian.shape(), (21, 21));
        debug_assert_eq!(stored_rhs.len(), 21);
        debug_assert_eq!(current_point.len(), 21);
        let mut result = DVector::<f64>::zeros(21);
        for row in 0..20 {
            let mut value = 0.0_f32;
            for column in 0..21 {
                value =
                    (jacobian[(row, column)] as f32).mul_add(current_point[column] as f32, value);
            }
            // Eigen's packet store is `pmadd(c, alpha=1, mld.b)`.
            let value = value.mul_add(1.0_f32, stored_rhs[row] as f32);
            result[row] = value as f64;
        }

        let mut value = 0.0_f32;
        for column in 0..21 {
            value = Self::prior_mul_add_separate_f32(
                value,
                jacobian[(20, column)] as f32,
                current_point[column] as f32,
            );
        }
        // The scalar remainder stores `mld.b + c0` separately.
        result[20] = Self::prior_add_separate_f32(stored_rhs[20] as f32, value) as f64;
        result
    }

    // Pinned native 392310: packet and scalar tails both use depth FMA,
    // followed by an FMA store with alpha=1 and the existing prior RHS.
    fn prior_rhs_col_major_39_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (39, 39));
        assert_eq!((stored_rhs.len(), delta.len()), (39, 39));
        DVector::from_fn(39, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..39 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            value.mul_add(1.0, stored_rhs[row] as f32) as f64
        })
    }

    fn prior_rhs_col_major_45_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (45, 45));
        assert_eq!((stored_rhs.len(), delta.len()), (45, 45));
        DVector::from_fn(45, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..45 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            f64::from(value.mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    fn prior_rhs_col_major_51_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (51, 51));
        assert_eq!((stored_rhs.len(), delta.len()), (51, 51));
        DVector::from_fn(51, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..51 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            f64::from(value.mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    fn prior_rhs_col_major_57_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (57, 57));
        assert_eq!((stored_rhs.len(), delta.len()), (57, 57));
        DVector::from_fn(57, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..57 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            f64::from(value.mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    /// Re-anchor a valid 21-row UpstreamF32 prior with the captured
    /// packet/scalar GEMV association.  Rows 0..19 use a zero accumulator,
    /// fused depth products, and a fused final subtraction from the stored
    /// RHS.  Row 20 is the scalar depth/product tail.  Other prior sizes and
    /// ExtendedF64 retain their existing paths.
    #[inline]
    fn prior_reanchor_rhs_col_major_21_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        debug_assert_eq!(jacobian.shape(), (21, 21));
        debug_assert_eq!(stored_rhs.len(), 21);
        debug_assert_eq!(keep_delta.len(), 21);
        let mut result = DVector::<f32>::zeros(21);
        for row in 0..20 {
            let mut value = 0.0_f32;
            for column in 0..21 {
                value = jacobian[(row, column)].mul_add(keep_delta[column], value);
            }
            result[row] = value.mul_add(-1.0_f32, stored_rhs[row]);
        }

        let mut value = 0.0_f32;
        for column in 0..21 {
            value += jacobian[(20, column)] * keep_delta[column];
        }
        result[20] = stored_rhs[20] - value;
        result
    }

    // Pinned native reanchor: 24 packet rows followed by three scalar rows.
    // Keep the product accumulation separate from the final subtraction.
    fn prior_reanchor_rhs_col_major_27_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        debug_assert_eq!(jacobian.shape(), (27, 27));
        debug_assert_eq!((stored_rhs.len(), keep_delta.len()), (27, 27));
        DVector::from_fn(27, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..27 {
                value = if row < 24 {
                    jacobian[(row, col)].mul_add(keep_delta[col], value)
                } else {
                    Self::prior_mul_add_separate_f32(value, jacobian[(row, col)], keep_delta[col])
                };
            }
            if row < 24 {
                value.mul_add(-1.0, stored_rhs[row])
            } else {
                stored_rhs[row] - value
            }
        })
    }

    // Native 4e11fe -> 49c380 -> 48cee0: for 33 columns, a single
    // depth block uses FMA even in scalar row 32 (48d1c8). The temporary
    // product is then subtracted separately at 49c420/49c480.
    fn prior_reanchor_rhs_col_major_33_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (33, 33));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (33, 33));
        DVector::from_fn(33, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..33 {
                value = jacobian[(row, col)].mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value.mul_add(1.0, 0.0)
        })
    }

    // Pinned native frame24 reanchor: all 39 output lanes accumulate
    // depth with FMA before the separate subtraction from the stored RHS.
    fn prior_reanchor_rhs_col_major_39_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (39, 39));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (39, 39));
        DVector::from_fn(39, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..39 {
                value = jacobian[(row, col)].mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    // Pinned 45x45 prior created at MH01 frame31: Eigen's
    // column-major GEMV contracts every depth update, then the caller applies
    // the stored-RHS subtraction as a separate scalar operation.
    fn prior_reanchor_rhs_col_major_45_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (45, 45));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (45, 45));
        DVector::from_fn(45, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..45 {
                value = jacobian[(row, col)].mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    // Native event33 (MH01 frame37): the 51x51 keep block uses the same
    // single Eigen GEMV depth block as 33/39/45. Every depth update is fused,
    // while the caller's stored-RHS subtraction remains a separate operation.
    fn prior_reanchor_rhs_col_major_51_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (51, 51));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (51, 51));
        DVector::from_fn(51, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..51 {
                value = jacobian[(row, col)].mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    fn prior_reanchor_rhs_col_major_57_f32(
        jacobian: &DMatrix<f32>,
        stored_rhs: &DVector<f32>,
        keep_delta: &DVector<f32>,
    ) -> DVector<f32> {
        assert_eq!(jacobian.shape(), (57, 57));
        assert_eq!((stored_rhs.len(), keep_delta.len()), (57, 57));
        DVector::from_fn(57, |row, _| {
            let mut value = 0.0_f32;
            for col in 0..57 {
                value = jacobian[(row, col)].mul_add(keep_delta[col], value);
            }
            stored_rhs[row] - value
        })
    }

    #[inline(never)]
    fn prior_mul_add_separate_f32(accumulator: f32, left: f32, right: f32) -> f32 {
        let product = left * right;
        accumulator + product
    }

    #[inline(never)]
    fn prior_add_separate_f32(left: f32, right: f32) -> f32 {
        left + right
    }

    fn prior_rhs_col_major_27_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (27, 27));
        assert_eq!(stored_rhs.len(), 27);
        assert_eq!(delta.len(), 27);
        DVector::from_fn(27, |row, _| {
            let mut value = 0.0_f32;
            // Pinned float GEMV (0x392310): the scalar remainder also uses
            // vfma231ss for every depth (0x3925f8), then vfma213ss (0x392605).
            // With 27 columns, the entire depth is one block.
            for col in 0..27 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            f64::from(value.mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    fn prior_rhs_col_major_33_f32(
        jacobian: &DMatrix<f64>,
        stored_rhs: &DVector<f64>,
        delta: &DVector<f64>,
    ) -> DVector<f64> {
        assert_eq!(jacobian.shape(), (33, 33));
        assert_eq!((stored_rhs.len(), delta.len()), (33, 33));
        DVector::from_fn(33, |row, _| {
            let mut value = 0.0_f32;
            // Native 392310, depth <= 127: one FMA block including scalar tails.
            for col in 0..33 {
                value = (jacobian[(row, col)] as f32).mul_add(delta[col] as f32, value);
            }
            f64::from(value.mul_add(1.0, stored_rhs[row] as f32))
        })
    }

    /// Return the FEJ-local tangent of the blocks named by a square-root
    /// prior.  The prior stores one encoded pose/state point per block; the
    /// solver chart is manifold-local, so a raw vector subtraction would be
    /// wrong for the rotation lanes.
    fn prior_current_point(&self, _state: &DVector<f64>, prior: &WindowPrior) -> DVector<f64> {
        let kinds = prior.kinds();
        let mut current_point = DVector::zeros(prior.fej_point.len());
        let mut prior_offset = 0;
        for (prior_index, frame_id) in prior.frame_ids.iter().enumerate() {
            let kind = kinds
                .get(prior_index)
                .copied()
                .unwrap_or(WindowBlockKind::State);
            let dof = block_dof(kind);
            if let Some(current_index) = self.block_index_for(*frame_id, kind) {
                let delta = match kind {
                    WindowBlockKind::Pose => self
                        .poses
                        .get(current_index)
                        .filter(|pose| pose.linearized_delta.len() == dof)
                        .map(|pose| pose.linearized_delta.clone()),
                    WindowBlockKind::State => self
                        .states
                        .get(current_index.saturating_sub(self.poses.len()))
                        .filter(|state| state.linearized_delta.len() == dof)
                        .map(|state| state.linearized_delta.clone()),
                    WindowBlockKind::StatePose => self
                        .states
                        .get(current_index.saturating_sub(self.poses.len()))
                        .filter(|state| state.linearized_delta.len() >= POSE_DOF)
                        .map(|state| state.linearized_delta.rows(0, POSE_DOF).into_owned()),
                };
                if let Some(delta) = delta {
                    current_point.rows_mut(prior_offset, dof).copy_from(&delta);
                }
            }
            prior_offset += dof;
        }
        current_point
    }

    /// Basalt's `computeMargPriorError` for a square-root prior.  It reports
    /// the linearized quadratic with the FEJ-independent constant
    /// `0.5 * rᵀr` removed:
    ///
    /// `deltaᵀ Jᵀ (0.5 * J * delta + b)`.
    ///
    /// Consequently this value is allowed to be negative.  The complete
    /// positive residual norm is still used by the row stack for H/b; only
    /// the LM cost comparison uses this source-compatible reduced value.
    fn marginal_prior_reduced_cost(&self, state: &DVector<f64>, prior: &WindowPrior) -> f64 {
        let delta = self.prior_current_point(state, prior);
        upstream_marginal_prior_error(&prior.jacobian, &prior.rhs, &delta, self.scalar_mode)
    }

    fn imu_factor(
        &self,
        state: &DVector<f64>,
        link: WindowImuLink,
    ) -> Option<WhitenedFactorRowStack> {
        self.imu_factor_with_trace(state, link)
            .map(|(factor, _)| factor)
    }

    fn imu_factor_with_trace(
        &self,
        state: &DVector<f64>,
        link: WindowImuLink,
    ) -> Option<(WhitenedFactorRowStack, ImuLinkDiagnostics)> {
        if link.from_index >= self.states.len() || link.to_index >= self.states.len() {
            return None;
        }
        let layout = self.layout();
        let from_offset = layout.offset_for_state(link.from_index)?;
        let to_offset = layout.offset_for_state(link.to_index)?;
        let base = (0..self.states.len())
            .map(|index| {
                self.block_nav(state, self.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let mut delta = link.delta.clone();
        // Preintegration normally accumulates this covariance while the
        // estimator supplies calibrated noise.  Keep direct WindowProblem
        // callers deterministic as well by materializing the same diagonal
        // process covariance when they provide a legacy zero-covariance delta.
        if delta.covariance.norm_squared() <= 1e-30 && delta.delta_time.is_finite() {
            let dt = delta.delta_time.max(1e-9);
            for axis in 0..3 {
                // Literal upstream row order is [position, rotation,
                // velocity], matching IntegratedImuMeasurement.
                delta.covariance[(axis, axis)] =
                    self.imu_noise.accel_density.powi(2) * dt.powi(3) / 3.0;
                delta.covariance[(3 + axis, 3 + axis)] = self.imu_noise.gyro_density.powi(2) * dt;
                delta.covariance[(6 + axis, 6 + axis)] = self.imu_noise.accel_density.powi(2) * dt;
            }
        }
        let from = &base[link.from_index];
        let to = &base[link.to_index];
        let jac_from = if self.states[link.from_index].linearized {
            &self.states[link.from_index].linearized_nav
        } else {
            from
        };
        let jac_to = if self.states[link.to_index].linearized {
            &self.states[link.to_index].linearized_nav
        } else {
            to
        };
        let factor = if self.scalar_mode == ScalarMode::UpstreamF32 {
            whitened_preintegration_factor_upstream_f32_fej_mode(
                from,
                to,
                jac_from,
                jac_to,
                &delta,
                self.gravity_world,
                self.states[link.from_index].linearized || self.states[link.to_index].linearized,
            )
            .ok()?
        } else {
            whitened_preintegration_factor(from, to, &delta, self.gravity_world).ok()?
        };
        let diagnostic_lm_iteration = active_diagnostic_lm_iteration();
        emit_imu_factor_input_diagnostic(
            &self.states[link.from_index],
            &self.states[link.to_index],
            self.states.last().map(|state| state.frame_id),
            link.from_index,
            link.to_index,
            diagnostic_lm_iteration,
            from,
            to,
            jac_from,
            jac_to,
            &delta,
            self.gravity_world,
        );
        let imu_input_diagnostic = if self.scalar_mode == ScalarMode::UpstreamF32
            && diagnostic_env_snapshot().diagnostic_imu_rows.is_some()
        {
            build_imu_factor_input_diagnostic(
                &self.states[link.from_index],
                &self.states[link.to_index],
                self.states.last().map(|state| state.frame_id),
                link.from_index,
                link.to_index,
                from,
                to,
                jac_from,
                jac_to,
                &delta,
                self.gravity_world,
            )
        } else {
            None
        };
        let (position, rotation, velocity, rotation_product) =
            imu_unwhitened_residual(from, to, &delta, self.gravity_world);
        let gyro_bias_delta = from.gyro_bias_rad_s - delta.bias_gyro;
        let accel_bias_delta = from.accel_bias_m_s2 - delta.bias_accel;
        let bias_rotation_correction = delta.jacobian_rotation_gyro_bias * gyro_bias_delta;
        let bias_velocity_correction = delta.jacobian_velocity_gyro_bias * gyro_bias_delta
            + delta.jacobian_velocity_accel_bias * accel_bias_delta;
        let bias_position_correction = delta.jacobian_position_gyro_bias * gyro_bias_delta
            + delta.jacobian_position_accel_bias * accel_bias_delta;
        emit_imu_unwhitened_residual_f64_diagnostic(
            &self.states[link.from_index],
            &self.states[link.to_index],
            self.states.last().map(|state| state.frame_id),
            link.from_index,
            link.to_index,
            diagnostic_lm_iteration,
            &rotation_product,
            rotation,
        );
        let eigenvalues = SymmetricEigen::new(delta.covariance.clone()).eigenvalues;
        let covariance_eigen_min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let covariance_eigen_max = eigenvalues
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut jacobian = DMatrix::zeros(factor.residual.len(), self.state_dof());
        jacobian
            .view_mut(
                (0, self.state_offset(link.from_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
        jacobian
            .view_mut(
                (0, self.state_offset(link.to_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
        let factor = WhitenedFactorRowStack::new(
            jacobian,
            DMatrix::zeros(factor.residual.len(), 0),
            factor.residual,
        )?
        .with_kind(FactorKind::Imu)
        .with_imu_link_offsets(from_offset, to_offset);
        let factor = if let Some(payload) = imu_input_diagnostic {
            factor.with_imu_input_diagnostic(payload)
        } else {
            factor
        };
        let trace = ImuLinkDiagnostics {
            from_frame_id: self.states[link.from_index].frame_id,
            to_frame_id: self.states[link.to_index].frame_id,
            delta_time: delta.delta_time,
            residual_position: vector3_array(position),
            residual_rotation: vector3_array(rotation),
            residual_velocity: vector3_array(velocity),
            whitened_norm: factor.residual.norm(),
            gyro_bias_delta: vector3_array(gyro_bias_delta),
            accel_bias_delta: vector3_array(accel_bias_delta),
            bias_rotation_correction: vector3_array(bias_rotation_correction),
            bias_velocity_correction: vector3_array(bias_velocity_correction),
            bias_position_correction: vector3_array(bias_position_correction),
            covariance_eigen_min,
            covariance_eigen_max,
        };
        Some((factor, trace))
    }

    /// Build an IMU factor from a trial value view.  Diagnostic payloads are
    /// intentionally not produced here: the clean trial path is selected only
    /// when no diagnostic/probe environment is active, while the owned clone
    /// path continues to call `imu_factor_with_trace` above.
    fn imu_factor_view<V: WindowValueView>(
        &self,
        values: &V,
        link: WindowImuLink,
    ) -> Option<WhitenedFactorRowStack> {
        let base = values.base();
        if link.from_index >= base.states.len() || link.to_index >= base.states.len() {
            return None;
        }
        let layout = base.layout();
        let from_offset = layout.offset_for_state(link.from_index)?;
        let to_offset = layout.offset_for_state(link.to_index)?;
        let current = (0..base.states.len())
            .map(|index| {
                values
                    .block_nav(base.poses.len() + index)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let mut delta = link.delta.clone();
        if delta.covariance.norm_squared() <= 1e-30 && delta.delta_time.is_finite() {
            let dt = delta.delta_time.max(1e-9);
            for axis in 0..3 {
                delta.covariance[(axis, axis)] =
                    base.imu_noise.accel_density.powi(2) * dt.powi(3) / 3.0;
                delta.covariance[(3 + axis, 3 + axis)] = base.imu_noise.gyro_density.powi(2) * dt;
                delta.covariance[(6 + axis, 6 + axis)] = base.imu_noise.accel_density.powi(2) * dt;
            }
        }
        let from = &current[link.from_index];
        let to = &current[link.to_index];
        let jac_from_owned = if values.state_is_linearized(link.from_index).unwrap_or(false) {
            values.state_linearized_nav(link.from_index)?
        } else {
            from.clone()
        };
        let jac_to_owned = if values.state_is_linearized(link.to_index).unwrap_or(false) {
            values.state_linearized_nav(link.to_index)?
        } else {
            to.clone()
        };
        let factor = if base.scalar_mode == ScalarMode::UpstreamF32 {
            whitened_preintegration_factor_upstream_f32_fej_mode(
                from,
                to,
                &jac_from_owned,
                &jac_to_owned,
                &delta,
                base.gravity_world,
                values.state_is_linearized(link.from_index).unwrap_or(false)
                    || values.state_is_linearized(link.to_index).unwrap_or(false),
            )
            .ok()?
        } else {
            whitened_preintegration_factor(from, to, &delta, base.gravity_world).ok()?
        };
        let mut jacobian = DMatrix::zeros(factor.residual.len(), base.state_dof());
        jacobian
            .view_mut(
                (0, base.state_offset(link.from_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
        jacobian
            .view_mut(
                (0, base.state_offset(link.to_index)),
                (factor.residual.len(), NAV_STATE_DOF),
            )
            .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
        Some(
            WhitenedFactorRowStack::new(
                jacobian,
                DMatrix::zeros(factor.residual.len(), 0),
                factor.residual,
            )?
            .with_kind(FactorKind::Imu)
            .with_imu_link_offsets(from_offset, to_offset),
        )
    }
}

fn emit_imu_factor_input_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) {
    let policy = diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_factor_inputs.as_ref() else {
        return;
    };
    let target_frame = policy.diagnostic_frame;
    let target_link = policy.diagnostic_imu_link;
    let target_iteration = policy.diagnostic_imu_iteration;
    if target_frame != active_frame_id || from_index != target_link {
        return;
    }
    // Preserve the historical one-shot behavior when the selector is absent.
    // With an explicit selector, only the requested LM linearization is
    // eligible; the create_new open below still prevents duplicate writes
    // from diagnostic-only relinearizations at that same iteration.
    if let Some(target_iteration) = target_iteration {
        if iteration != Some(target_iteration) {
            return;
        }
    }
    let Some(record) = build_imu_factor_input_diagnostic(
        from_state,
        to_state,
        active_frame_id,
        from_index,
        to_index,
        from,
        to,
        jac_from,
        jac_to,
        delta,
        gravity_world,
    ) else {
        return;
    };
    let mut record = record;
    if let Some(iteration) = iteration {
        record["iteration"] = json!(iteration);
    }
    let Ok(mut file) = OpenOptions::new().write(true).create_new(true).open(path) else {
        return;
    };
    let _ = serde_json::to_writer_pretty(&mut file, &record);
    let _ = file.write_all(b"\n");
}

const IMU_F64_CAPTURE_MAX_BYTES: usize = 2 * 1024 * 1024;

#[derive(Default)]
struct ImuF64CaptureWriterState {
    initialized: bool,
    bytes_written: usize,
    records_written: usize,
    disabled: bool,
}

static IMU_F64_CAPTURE_WRITER: OnceLock<Mutex<ImuF64CaptureWriterState>> = OnceLock::new();

/// Build the observation-only payload from the values that have already
/// crossed the f64 `scaled_axis` call.  In particular, this function does not
/// call `scaled_axis`, compute a norm, or derive an angle.  Nalgebra does not
/// expose those internal intermediates, so serializing a recomputation as if
/// it were an internal operand would make the capture provenance false.
fn build_imu_unwhitened_residual_f64_capture(
    active_frame_id: Option<u64>,
    from_state: &WindowState,
    to_state: &WindowState,
    link_index: usize,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    rotation_product: &UnitQuaternion<f64>,
    rotation: Vector3<f64>,
) -> Option<JsonValue> {
    let active_frame_id = active_frame_id?;
    let product = rotation_product.quaternion();
    let bits = |value: f64| format!("{:016x}", value.to_bits());
    let mut record = json!({
        "schema": "basalt.imu_r_f64_inline_capture.v1",
        "frame_id": active_frame_id,
        "active_frame_id": active_frame_id,
        "link_index": link_index,
        "from_index": from_index,
        "to_index": to_index,
        "from_frame_id": from_state.frame_id,
        "to_frame_id": to_state.frame_id,
        "capture_proof": {
            "origin": "same_inline_computation",
            "producer": "imu_unwhitened_residual",
            "product_source": "same_inline_rotation_product",
            "result_source": "same_inline_scaled_axis_result",
            "product_same_call": true,
            "result_same_call": true,
            "norm_captured": false,
            "angle_captured": false,
            "not_recomputed": true,
        },
        "product_xyzw_f64_bits": [
            bits(product.i),
            bits(product.j),
            bits(product.k),
            bits(product.w),
        ],
        "observed_r_R_f64_bits": [bits(rotation.x), bits(rotation.y), bits(rotation.z)],
    });
    if let Some(iteration) = iteration {
        record["iteration"] = json!(iteration);
    }
    Some(record)
}

/// Append one initial-window f64 IMU observation, bounded so an accidentally
/// broad diagnostic selection cannot consume unbounded disk.  The writer is
/// process-local and only initialized after an explicit opt-in path reaches
/// the initial-diagnostics scope; default solver runs never create the file.
fn emit_imu_unwhitened_residual_f64_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    iteration: Option<usize>,
    rotation_product: &UnitQuaternion<f64>,
    rotation: Vector3<f64>,
) {
    let policy = diagnostic_env_snapshot();
    let Some(path) = policy.diagnostic_imu_unwhitened_f64.as_ref() else {
        return;
    };
    if !initial_imu_f64_capture_scope_active() {
        return;
    }
    let Some(link_index) = initial_imu_f64_capture_link_index() else {
        return;
    };
    let Some(record) = build_imu_unwhitened_residual_f64_capture(
        active_frame_id,
        from_state,
        to_state,
        link_index,
        from_index,
        to_index,
        iteration,
        rotation_product,
        rotation,
    ) else {
        return;
    };
    let Ok(mut line) = serde_json::to_vec(&record) else {
        return;
    };
    line.push(b'\n');
    if line.len() > IMU_F64_CAPTURE_MAX_BYTES {
        return;
    }

    let writer =
        IMU_F64_CAPTURE_WRITER.get_or_init(|| Mutex::new(ImuF64CaptureWriterState::default()));
    let Ok(mut writer) = writer.lock() else {
        return;
    };
    if writer.disabled {
        return;
    }
    if !writer.initialized {
        if let Some(parent) = path.parent() {
            if fs::create_dir_all(parent).is_err() {
                writer.disabled = true;
                return;
            }
        }
        // Never truncate an existing artifact.  A reused path is treated as
        // a failed capture and the caller must provide a fresh path.
        if OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .is_err()
        {
            writer.disabled = true;
            return;
        }
        writer.initialized = true;
    }
    if writer
        .bytes_written
        .checked_add(line.len())
        .is_none_or(|total| total > IMU_F64_CAPTURE_MAX_BYTES)
    {
        writer.disabled = true;
        return;
    }
    let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) else {
        writer.disabled = true;
        return;
    };
    if file.write_all(&line).is_err() {
        writer.disabled = true;
        return;
    }
    writer.bytes_written += line.len();
    writer.records_written += 1;
}

/// Materialize the complete f32 IMU input boundary for a local-block audit.
/// The current state owns the residual, while the FEJ/linearized state owns
/// the analytic Jacobian whenever its corresponding `linearized` flag is set.
/// This helper is called only when a diagnostic environment variable is set;
/// no payload or stage product is constructed on the normal solver path.
fn build_imu_factor_input_diagnostic(
    from_state: &WindowState,
    to_state: &WindowState,
    active_frame_id: Option<u64>,
    from_index: usize,
    to_index: usize,
    from: &BasaltNavState,
    to: &BasaltNavState,
    jac_from: &BasaltNavState,
    jac_to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> Option<JsonValue> {
    let Ok((raw, jacobian, whitener, whitened, dense)) = preintegration_f32_stages_fej_mode(
        from,
        to,
        jac_from,
        jac_to,
        delta,
        gravity_world,
        from_state.linearized || to_state.linearized,
    ) else {
        return None;
    };
    // `ImuBlock::linearizeImu` first evaluates the residual/Jacobian at the
    // FEJ states and then, when either endpoint is already linearized,
    // replaces only the residual with the current-state evaluation.  Keep
    // both raw residuals in the payload so the input-stage record can explain
    // that split without re-running or modifying the production factor.
    let raw_fej = preintegration_residual_f32(jac_from, jac_to, delta, gravity_world);
    let bits = |x: f32| format!("{:08x}", x.to_bits());
    let vec_bits = |v: &[f32]| v.iter().copied().map(bits).collect::<Vec<_>>();
    let f64vec_bits = |v: &[f64]| v.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>();
    let mat_bits = |rows: usize, cols: usize, f: &dyn Fn(usize, usize) -> f32| {
        (0..cols)
            .flat_map(|c| (0..rows).map(move |r| bits(f(r, c))))
            .collect::<Vec<_>>()
    };
    let state_bits = |s: &BasaltNavState| {
        let mut v = Vec::with_capacity(16);
        v.extend(s.imu_to_world.translation.iter().map(|x| bits(*x as f32)));
        v.extend(
            [
                s.imu_to_world.rotation.w,
                s.imu_to_world.rotation.i,
                s.imu_to_world.rotation.j,
                s.imu_to_world.rotation.k,
            ]
            .into_iter()
            .map(|x| bits(x as f32)),
        );
        v.extend(s.velocity_world_m_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.gyro_bias_rad_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.accel_bias_m_s2.iter().map(|x| bits(*x as f32)));
        v
    };
    // Keep the historical compact state lane (`wxyz`) above for consumers
    // that already read v2 artifacts, but also expose the native
    // PoseVelBias spelling (`xyzw`) in v3.  The latter matches Basalt's
    // serialized `unit_quaternion()` order and makes the state record
    // self-describing instead of requiring a consumer-side permutation.
    let state_xyzw_bits = |s: &BasaltNavState| {
        let mut v = Vec::with_capacity(16);
        v.extend(s.imu_to_world.translation.iter().map(|x| bits(*x as f32)));
        v.extend(
            [
                s.imu_to_world.rotation.i,
                s.imu_to_world.rotation.j,
                s.imu_to_world.rotation.k,
                s.imu_to_world.rotation.w,
            ]
            .into_iter()
            .map(|x| bits(x as f32)),
        );
        v.extend(s.velocity_world_m_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.gyro_bias_rad_s.iter().map(|x| bits(*x as f32)));
        v.extend(s.accel_bias_m_s2.iter().map(|x| bits(*x as f32)));
        v
    };
    let state_record = |frame_state: &WindowState, state: &BasaltNavState| {
        let q = state.imu_to_world.rotation.quaternion();
        json!({
            "t_ns": frame_state.timestamp_ns,
            "translation_f32_bits": f64vec_bits(state.imu_to_world.translation.as_slice()),
            "quaternion_xyzw_f32_bits": f64vec_bits(&[q.i, q.j, q.k, q.w]),
            "velocity_f32_bits": f64vec_bits(state.velocity_world_m_s.as_slice()),
            "bias_gyro_f32_bits": f64vec_bits(state.gyro_bias_rad_s.as_slice()),
            "bias_accel_f32_bits": f64vec_bits(state.accel_bias_m_s2.as_slice()),
            "f32_bits": state_xyzw_bits(state),
            "scalar_count": 16,
        })
    };
    let matrix3_bits = |m: &Matrix3<f64>| mat_bits(3, 3, &|r, c| m[(r, c)] as f32);
    let delta_rotation_matrix = delta.delta_rotation.to_rotation_matrix().into_inner();
    let delta_rotation_matrix_bits = mat_bits(3, 3, &|r, c| delta_rotation_matrix[(r, c)] as f32);
    let mut delta_dbg = Vec::with_capacity(27);
    let mut delta_dba = Vec::with_capacity(27);
    for col in 0..3 {
        for row in 0..9 {
            delta_dbg.push(match row {
                0..=2 => delta.jacobian_position_gyro_bias[(row, col)],
                3..=5 => delta.jacobian_rotation_gyro_bias[(row - 3, col)],
                _ => delta.jacobian_velocity_gyro_bias[(row - 6, col)],
            });
            delta_dba.push(match row {
                0..=2 => delta.jacobian_position_accel_bias[(row, col)],
                3..=5 => 0.0,
                _ => delta.jacobian_velocity_accel_bias[(row - 6, col)],
            });
        }
    }
    let dt_f32 = delta.delta_time as f32;
    let dt_ns = (dt_f32 * 1.0e9_f32).round() as i64;
    Some(json!({
        "schema": "basalt.imu_factor_input_diagnostic.v3",
        "active_frame_id": active_frame_id,
        "from_index": from_index,
        "to_index": to_index,
        "from_frame_id": from_state.frame_id,
        "to_frame_id": to_state.frame_id,
        "start_t_ns": from_state.timestamp_ns,
        "start_t": from_state.timestamp_ns,
        "end_t_ns": from_state.timestamp_ns.saturating_add(dt_ns),
        "dt_ns": dt_ns,
        "dt": dt_f32,
        "delta_time_bits": bits(dt_f32),
        "current": {
            "from": state_record(from_state, from),
            "to": state_record(to_state, to),
        },
        "fej": {
            "from": state_record(from_state, jac_from),
            "to": state_record(to_state, jac_to),
        },
        "states": {
            "from_fej": state_record(from_state, jac_from),
            "from_current": state_record(from_state, from),
            "to_fej": state_record(to_state, jac_to),
            "to_current": state_record(to_state, to),
        },
        "current_from_state_f32_bits": state_bits(from),
        "current_to_state_f32_bits": state_bits(to),
        "from_current_state_f32_bits": state_bits(from),
        "to_current_state_f32_bits": state_bits(to),
        "from_state_f32_bits": state_bits(from),
        "to_state_f32_bits": state_bits(to),
        "linearized_from_state_f32_bits": state_bits(jac_from),
        "linearized_to_state_f32_bits": state_bits(jac_to),
        "fej_from_state_f32_bits": state_bits(jac_from),
        "fej_to_state_f32_bits": state_bits(jac_to),
        "from_linearized": from_state.linearized,
        "to_linearized": to_state.linearized,
        "from_fej": from_state.linearized,
        "to_fej": to_state.linearized,
        "linearized_flags": {
            "from": from_state.linearized,
            "to": to_state.linearized,
        },
        "preintegrated": {
            "delta_rotation_matrix_f32_bits": delta_rotation_matrix_bits,
            "delta_rotation_f32_bits": f64vec_bits(&[
                delta.delta_rotation.w,
                delta.delta_rotation.i,
                delta.delta_rotation.j,
                delta.delta_rotation.k,
            ]),
            "delta_position_f32_bits": f64vec_bits(delta.delta_position.as_slice()),
            "delta_velocity_f32_bits": f64vec_bits(delta.delta_velocity.as_slice()),
            "bias_gyro_f32_bits": f64vec_bits(delta.bias_gyro.as_slice()),
            "bias_accel_f32_bits": f64vec_bits(delta.bias_accel.as_slice()),
            "jacobian_rotation_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_rotation_gyro_bias),
            "jacobian_velocity_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_velocity_gyro_bias),
            "jacobian_velocity_accel_bias_f32_bits": matrix3_bits(&delta.jacobian_velocity_accel_bias),
            "jacobian_position_gyro_bias_f32_bits": matrix3_bits(&delta.jacobian_position_gyro_bias),
            "jacobian_position_accel_bias_f32_bits": matrix3_bits(&delta.jacobian_position_accel_bias),
            "covariance_f32_bits": mat_bits(9, 9, &|r, c| delta.covariance[(r, c)] as f32),
        },
        "delta_t_ns": (dt_f32 * 1.0e9_f32).round() as i64,
        "delta_rotation_matrix_f32_bits": delta_rotation_matrix_bits,
        "delta_rotation_f32_bits": f64vec_bits(&[
            delta.delta_rotation.w,
            delta.delta_rotation.i,
            delta.delta_rotation.j,
            delta.delta_rotation.k,
        ]),
        "delta_velocity_f32_bits": f64vec_bits(delta.delta_velocity.as_slice()),
        "delta_position_f32_bits": f64vec_bits(delta.delta_position.as_slice()),
        "delta_covariance_f32_bits": f64vec_bits(delta.covariance.as_slice()),
        "d_state_d_bg_f32_bits": f64vec_bits(&delta_dbg),
        "d_state_d_ba_f32_bits": f64vec_bits(&delta_dba),
        "delta_d_state_d_bg_f32_bits": f64vec_bits(&delta_dbg),
        "delta_d_state_d_ba_f32_bits": f64vec_bits(&delta_dba),
        "covariance_f32_bits": mat_bits(9, 9, &|r, c| delta.covariance[(r, c)] as f32),
        "d_state_update_trace": delta.update_trace.iter().map(|u| json!({
            "t_ns": u.t_ns,
            "f_f32_bits": u.f.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "g_f32_bits": u.g.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "f_old_bg_f32_bits": u.f_old_bg.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "d_state_d_bg_f32_bits": u.d_state_d_bg.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
            "d_state_d_ba_f32_bits": u.d_state_d_ba.iter().map(|x| bits(*x as f32)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "raw_residual_fej_bits": vec_bits(raw_fej.as_slice()),
        "raw_residual_bits": vec_bits(raw.as_slice()),
        "raw_jacobian_bits": mat_bits(9, 30, &|r, c| jacobian[(r, c)]),
        "sqrt_information_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "whitened_residual_bits": vec_bits(whitened.as_slice()),
        "whitened_jacobian_bits": mat_bits(9, 30, &|r, c| dense[(r, c)]),
        "raw_r_bits": vec_bits(raw.as_slice()),
        "raw_j_bits": mat_bits(9, 30, &|r, c| jacobian[(r, c)]),
        "w_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "w_f32_bits": mat_bits(9, 9, &|r, c| whitener[(r, c)]),
        "raw_r_fej_bits": vec_bits(raw_fej.as_slice()),
        "whitened_r_bits": vec_bits(whitened.as_slice()),
        "whitened_j_bits": mat_bits(9, 30, &|r, c| dense[(r, c)]),
        "native_expr_trace": preintegration_f32_expression_trace(from, to, delta, gravity_world),
        "fej_rotation_producer_trace": preintegration_f32_rotation_producer_trace(
            jac_from, jac_to, delta,
        ),
    }))
}

fn vector3_array(value: Vector3<f64>) -> [f64; 3] {
    [value.x, value.y, value.z]
}

const IMU_F64_ATAN_SMALL_MAX_RATIO: f64 = f64::from_bits(0x3eb0_0000_0000_0000);
const IMU_F64_ATAN_SMALL_NORMALIZE_X_FLOOR_EXP: i32 = -500;

const fn imu_f64_atan_small_floor_exponent(value: f64) -> i32 {
    let bits = value.to_bits() & 0x7fff_ffff_ffff_ffff;
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent != 0 {
        exponent - 1023
    } else {
        let mantissa = bits & 0x000f_ffff_ffff_ffff;
        -1074 + (63 - mantissa.leading_zeros() as i32)
    }
}

fn imu_f64_atan_small_exact_power_of_two(exponent: i32) -> f64 {
    assert!((0..=1023).contains(&exponent));
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

fn imu_f64_atan_small_scaled_operands(y: f64, x: f64) -> (f64, f64, i32) {
    let shift =
        (IMU_F64_ATAN_SMALL_NORMALIZE_X_FLOOR_EXP - imu_f64_atan_small_floor_exponent(x)).max(0);
    let scale = imu_f64_atan_small_exact_power_of_two(shift);
    (y * scale, x * scale, shift)
}

/// Grouped small-angle atan2 control used only by the diagnostic raw f64 IMU
/// rotation observation.  The original nalgebra `scaled_axis` result remains
/// the fallback for every rejected, zero, or non-finite domain.
fn imu_f64_atan_small_grouped_positive(y: f64, x: f64) -> Option<f64> {
    if !(y.is_finite() && x.is_finite()) || y < 0.0 || x <= 0.0 {
        return None;
    }
    if y == 0.0 {
        return Some(y);
    }
    let unscaled_ratio = y / x;
    if unscaled_ratio > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return None;
    }
    let (scaled_y, scaled_x, _) = imu_f64_atan_small_scaled_operands(y, x);
    let q_hi = scaled_y / scaled_x;
    if q_hi > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return None;
    }
    let residual = (-q_hi).mul_add(scaled_x, scaled_y);
    let q_lo = residual / scaled_x;
    let q2 = q_hi * q_hi;
    let q3 = q2 * q_hi;
    let q5 = q3 * q2;
    let correction = q_lo - q3 / 3.0 + q5 / 5.0;
    Some(q_hi + correction)
}

/// Reproduce the candidate only for the diagnostic f64 IMU raw rotation.
/// This deliberately preserves nalgebra's axis sign convention (`w >= 0`),
/// uses `abs(w)` for the angle domain, and returns the original nalgebra
/// `scaled_axis` result whenever the candidate cannot accept the domain.
fn imu_f64_scaled_axis_candidate(rotation: &UnitQuaternion<f64>) -> Vector3<f64> {
    let mut vector = rotation.quaternion().imag().clone_owned();
    let scalar = rotation.quaternion().scalar();
    if !(scalar >= 0.0) {
        vector = -vector;
    }
    let w = scalar.abs();
    let norm_squared = vector.x * vector.x + vector.y * vector.y + vector.z * vector.z;
    let norm = norm_squared.sqrt();
    if norm == 0.0 || norm / w > IMU_F64_ATAN_SMALL_MAX_RATIO {
        return rotation.scaled_axis();
    }
    let Some(half_angle) = imu_f64_atan_small_grouped_positive(norm, w) else {
        return rotation.scaled_axis();
    };
    (vector / norm) * (half_angle * 2.0)
}

/// The literal upstream unwhitened residual contract.  Whitening is kept in
/// `imu::factors::whitened_preintegration_factor`; this adapter mirrors only
/// the raw rows so trace consumers can compare them against the oracle.
fn imu_unwhitened_residual(
    from: &BasaltNavState,
    to: &BasaltNavState,
    delta: &ImuPreintegratedDelta,
    gravity_world: Vector3<f64>,
) -> (
    Vector3<f64>,
    Vector3<f64>,
    Vector3<f64>,
    UnitQuaternion<f64>,
) {
    let (dr, dv, dp) = delta.corrected(from.gyro_bias_rad_s, from.accel_bias_m_s2);
    let dt = delta.delta_time;
    let r_iw = from.imu_to_world.rotation.inverse();
    // Keep the exact upstream multiplication expression/order intact, but
    // retain its already-computed quaternion so the opt-in f64 observer can
    // serialize the actual product instead of reconstructing it later.
    let rotation_product = r_iw * to.imu_to_world.rotation * dr.inverse();
    let rotation = imu_f64_scaled_axis_candidate(&rotation_product);
    let velocity =
        r_iw * (to.velocity_world_m_s - from.velocity_world_m_s - gravity_world * dt) - dv;
    let position = r_iw
        * (to.imu_to_world.translation
            - from.imu_to_world.translation
            - from.velocity_world_m_s * dt
            - gravity_world * (0.5 * dt * dt))
        - dp;
    (position, rotation, velocity, rotation_product)
}

fn quat_f32_from_f64(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    UnitQuaternion::new_unchecked(Quaternion::new(
        rotation.w as f32,
        rotation.i as f32,
        rotation.j as f32,
        rotation.k as f32,
    ))
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
        (
            crate::update::portable_cosf::sinf(half_theta) / theta,
            crate::update::portable_cosf::cosf(half_theta),
        )
    };
    UnitQuaternion::new_unchecked(Quaternion::new(
        real_factor,
        imag_factor * omega.x,
        imag_factor * omega.y,
        imag_factor * omega.z,
    ))
}

fn quat_f64_from_f32(rotation: &UnitQuaternion<f32>) -> UnitQuaternion<f64> {
    UnitQuaternion::new_unchecked(Quaternion::new(
        rotation.w as f64,
        rotation.i as f64,
        rotation.j as f64,
        rotation.k as f64,
    ))
}

#[inline]
fn upstream_f32_zero_inc_pose(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    quat_f64_from_f32(&sophus_so3_product(
        so3_exp_f32(Vector3::zeros()),
        quat_f32_from_f64(rotation),
    ))
}

fn trace_state_update_quaternion(
    frame_id: u64,
    before: UnitQuaternion<f32>,
    omega: Vector3<f32>,
    delta: UnitQuaternion<f32>,
    updated: UnitQuaternion<f32>,
) {
    if diagnostic_env_snapshot().state_update_trace_frame != Some(frame_id) {
        return;
    }
    let qbits = |rotation: &UnitQuaternion<f32>| {
        format!(
            "{:08x},{:08x},{:08x},{:08x}",
            rotation.i.to_bits(),
            rotation.j.to_bits(),
            rotation.k.to_bits(),
            rotation.w.to_bits(),
        )
    };
    eprintln!(
        "STATE_UPDATE frame={} before={} omega={:08x},{:08x},{:08x} delta={} updated={}",
        frame_id,
        qbits(&before),
        omega.x.to_bits(),
        omega.y.to_bits(),
        omega.z.to_bits(),
        qbits(&delta),
        qbits(&updated),
    );
}

fn vec3_f32(value: Vector3<f64>) -> Vector3<f32> {
    value.map(|component| component as f32)
}

fn vec3_f64(value: Vector3<f32>) -> Vector3<f64> {
    value.map(|component| component as f64)
}

impl WindowProblem {
    /// Build the complete candidate problem used by the LM trial objective.
    /// The compact chart alone is insufficient for Basalt's FEJ prior and
    /// quaternion state: both the full manifold objects and their accumulated
    /// linearized deltas must follow the trial. This helper is shared with
    /// the opt-in iteration detail trace so category costs are the same
    /// objective that drives the accept/reject decision.
    fn trial_problem_with_step_owned(&self, state: &DVector<f64>, step: &DVector<f64>) -> Self {
        let increments = self.landmark_steps(state, step);
        let mut trial_problem = self.clone();
        let (trial_poses, trial_states) = self.apply_step_full(state, step);
        let scalar_mode = trial_problem.scalar_mode;
        let state_base_offset = trial_problem.poses.len() * POSE_DOF;
        for (index, (pose, trial_pose)) in
            trial_problem.poses.iter_mut().zip(trial_poses).enumerate()
        {
            pose.pose = trial_pose;
            pose.stored_current_pose = pose.pose.clone();
            let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
            accumulate_linearized_delta(&mut pose.linearized_delta, &increment, scalar_mode);
        }
        for (index, (state, trial_state)) in trial_problem
            .states
            .iter_mut()
            .zip(trial_states)
            .enumerate()
        {
            state.nav = trial_state;
            if !state.linearized {
                state.linearized_nav = state.nav.clone();
                state.linearized_delta.fill(0.0);
            } else {
                let increment = step
                    .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                    .into_owned();
                accumulate_linearized_delta(&mut state.linearized_delta, &increment, scalar_mode);
                state.stored_current_nav = state.nav.clone();
            }
        }
        trial_problem.apply_landmark_steps(&increments);
        trial_problem
    }
}

impl WindowTrialView<'_, '_> {
    fn linearize(&self) -> Result<LmLinearization, LmFailure> {
        self.base.linearize_view(self)
    }

    fn cost(&self) -> Result<f64, LmFailure> {
        if self.chart.len() != self.base.state_dof() {
            return Err(LmFailure::LinearSolve);
        }
        if self.base.scalar_mode == ScalarMode::UpstreamF32 {
            let prior = self.base.prior.as_ref().map_or_else(
                || {
                    self.base
                        .prior_factors_view(self)
                        .first()
                        .map_or(0.0, |factor| factor.objective_cost)
                },
                |prior| self.base.marginal_prior_reduced_cost_view(self, prior),
            );
            return self.base.trial_objective_f32(
                |index, visual| {
                    if visual {
                        self.block_nav_visual_current(index)
                    } else {
                        self.block_nav(index)
                    }
                },
                |index| self.landmark_parameter(index),
                prior,
            );
        }
        checked_lm_linearization_cost(&self.linearize()?)
    }
}

const fn checked_lm_linearization_cost(linearization: &LmLinearization) -> Result<f64, LmFailure> {
    if linearization.cost.is_finite() {
        Ok(linearization.cost)
    } else {
        Err(LmFailure::NonFinite)
    }
}

impl WindowProblem {
    fn linearize_view<V: WindowValueView + Sync>(
        &self,
        values: &V,
    ) -> Result<LmLinearization, LmFailure> {
        let layout = self.layout();
        let mut factors = self.prior_factors_view(values);
        // Each landmark's visual factor depends only on that landmark's own
        // anchor/observation state views (`values`, shared and read-only)
        // and its own landmark index -- there is no shared mutable state
        // between landmarks (`visual_factor_view` builds and returns a fresh
        // local `WhitenedFactorRowStack`, it does not accumulate into any
        // caller-owned state). Computing every landmark's factor in
        // parallel and then pushing the `Some` results into `factors`
        // serially, in the same ascending `landmark_index` order the
        // sequential loop always used, reproduces the identical factor list
        // -- and therefore every downstream f32 rounding tree that depends
        // on factor order, e.g. the landmark-reduction accumulation --
        // bit-for-bit; only *when* each pure per-landmark factor is
        // computed changes, never its value or its position in `factors`.
        let visual_factors: Vec<Option<WhitenedFactorRowStack>> = (0..self.landmarks.len())
            .into_par_iter()
            .map(|landmark_index| self.visual_factor_view(values, landmark_index))
            .collect();
        for factor in visual_factors.into_iter().flatten() {
            factors.push(factor);
        }
        for link in self.imu_links.iter().cloned() {
            if let Some(factor) = self.imu_factor_view(values, link.clone()) {
                factors.push(factor);
            }
            if let Some(factor) = bias_walk_factor(
                values.chart(),
                link.clone(),
                self.state_dof(),
                layout.offset_for_state(link.from_index),
                layout.offset_for_state(link.to_index),
                self.bias_walk_noise,
                self.scalar_mode,
            ) {
                factors.push(factor);
            }
        }
        let prior_prefix = if self.prior.is_some() {
            1
        } else if self
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !self.states.is_empty())
        {
            1
        } else {
            0
        };
        let mut cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
            factors
                .iter()
                .skip(prior_prefix)
                .fold(0.0_f32, |sum, factor| sum + factor.objective_cost as f32) as f64
        } else {
            factors
                .iter()
                .skip(prior_prefix)
                .map(|factor| factor.objective_cost)
                .sum()
        };
        if prior_prefix != 0 {
            let prior_error = self.prior.as_ref().map_or_else(
                || factors.first().map_or(0.0, |factor| factor.objective_cost),
                |prior| self.marginal_prior_reduced_cost_view(values, prior),
            );
            cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
                (cost as f32 + prior_error as f32) as f64
            } else {
                cost + prior_error
            };
        }
        Ok(LmLinearization { factors, cost })
    }
}

impl LmProblem for WindowProblem {
    fn scalar_mode(&self) -> ScalarMode {
        self.scalar_mode
    }

    fn diagnostic_frame_id(&self) -> Option<u64> {
        self.states.last().map(|state| state.frame_id)
    }

    fn diagnostic_patch_reduced_f32(
        &self,
        iteration: usize,
        h: &mut DMatrix<f32>,
        b: &mut DVector<f32>,
    ) {
        let policy = diagnostic_env_snapshot();
        let Some(path) = policy.diagnostic_imu_hb.as_ref() else {
            return;
        };
        let Some(frame_id) = self.states.last().map(|state| state.frame_id) else {
            return;
        };
        let target_frame = policy.diagnostic_frame;
        let target_iteration = policy.diagnostic_iteration;
        if target_frame != Some(frame_id) || target_iteration != Some(iteration) {
            return;
        }
        let Ok(text) = fs::read_to_string(path) else {
            return;
        };
        let Ok(payload) = serde_json::from_str::<JsonValue>(&text) else {
            return;
        };
        let Some(h_delta) = payload.get("h_delta_bits").and_then(JsonValue::as_array) else {
            return;
        };
        let Some(b_delta) = payload.get("b_delta_bits").and_then(JsonValue::as_array) else {
            return;
        };
        if h_delta.len() != h.nrows() * h.ncols() || b_delta.len() != b.len() {
            return;
        }
        for col in 0..h.ncols() {
            for row in 0..h.nrows() {
                let index = col * h.nrows() + row;
                let Some(bits) = h_delta[index].as_str() else {
                    return;
                };
                let Ok(bits) = u32::from_str_radix(bits, 16) else {
                    return;
                };
                h[(row, col)] += f32::from_bits(bits);
            }
        }
        for (index, value) in b_delta.iter().enumerate() {
            let Some(bits) = value.as_str() else {
                return;
            };
            let Ok(bits) = u32::from_str_radix(bits, 16) else {
                return;
            };
            b[index] += f32::from_bits(bits);
        }
    }

    fn diagnostic_lm_event(&mut self, event: LmDiagnosticEvent<'_>) {
        emit_iteration_event(self, event);
    }

    fn apply_step(&self, state: &DVector<f64>, step: &DVector<f64>) -> DVector<f64> {
        if state.len() != self.state_dof() || step.len() != self.state_dof() {
            return state + step;
        }
        let (updated_poses, updated) = self.apply_step_full(state, step);
        flatten_blocks_with_mode(
            &self
                .poses
                .iter()
                .enumerate()
                .map(|(index, _)| WindowPose {
                    frame_id: self.poses[index].frame_id,
                    timestamp_ns: self.poses[index].timestamp_ns,
                    pose: updated_poses[index].clone(),
                    stored_current_pose: updated_poses[index].clone(),
                    linearized_pose: self.poses[index].linearized_pose.clone(),
                    linearized_delta: self.poses[index].linearized_delta.clone(),
                    is_keyframe: self.poses[index].is_keyframe,
                })
                .collect::<Vec<_>>(),
            &updated
                .into_iter()
                .enumerate()
                .map(|(index, nav)| WindowState {
                    frame_id: self.states[index].frame_id,
                    timestamp_ns: self.states[index].timestamp_ns,
                    nav: nav.clone(),
                    stored_current_nav: if self.states[index].linearized {
                        nav.clone()
                    } else {
                        self.states[index].stored_current_nav.clone()
                    },
                    linearized_nav: if self.states[index].linearized {
                        self.states[index].linearized_nav.clone()
                    } else {
                        // `applyInc` on an upstream non-linearized state moves
                        // its linearization point itself, so a trial chart
                        // carries the candidate as its FEJ point.
                        nav
                    },
                    linearized_delta: self.states[index].linearized_delta.clone(),
                    is_keyframe: self.states[index].is_keyframe,
                    is_latest: self.states[index].is_latest,
                    linearized: self.states[index].linearized,
                })
                .collect::<Vec<_>>(),
            self.scalar_mode,
        )
    }

    fn linearize(&self, state: &DVector<f64>) -> Result<LmLinearization, LmFailure> {
        if state.len() != self.state_dof() {
            return Err(LmFailure::LinearSolve);
        }
        let layout = self.layout();
        let mut factors = self.prior_factors(state);
        for landmark_index in 0..self.landmarks.len() {
            if let Some(factor) = self.visual_factor(state, landmark_index) {
                // The clean UpstreamF32 reducer may retain one compact Q1/R
                // payload per visual block.  Carry identity explicitly so
                // that future trial recovery is mapped by landmark rather
                // than inferred from factor order; callers without this
                // optional metadata remain on the legacy path.
                factors.push(
                    factor
                        .with_landmark_metadata(
                            landmark_index,
                            self.landmarks[landmark_index].track_id,
                        )
                        .with_visual_observation_ids(
                            self.landmarks[landmark_index]
                                .observations
                                .iter()
                                .map(|observation| (observation.state_index, observation.camera_id))
                                .collect(),
                        ),
                );
            }
        }
        for link in self.imu_links.iter().cloned() {
            if let Some(factor) = self.imu_factor(state, link.clone()) {
                factors.push(factor);
            }
            if let Some(factor) = bias_walk_factor(
                state,
                link.clone(),
                self.state_dof(),
                layout.offset_for_state(link.from_index),
                layout.offset_for_state(link.to_index),
                self.bias_walk_noise,
                self.scalar_mode,
            ) {
                factors.push(factor);
            }
        }
        // `LinearizationAbsQR::linearizeProblem()` accumulates visual landmark
        // blocks first, then IMU blocks, and finally the marginal-prior error.
        // Rust's factor row stack keeps the prior row first because it must
        // occupy the AOM prefix, so do not fold its positive residual and
        // subtract it afterward: that changes the f32 rounding boundary.
        // Instead omit that row from the ordinary objective fold and append
        // Basalt's constant-free prior error at the source position.
        let prior_prefix = if self.prior.is_some() {
            1
        } else if self
            .anchor_point
            .as_ref()
            .is_some_and(|point| point.len() == NAV_STATE_DOF && !self.states.is_empty())
        {
            1
        } else {
            0
        };
        let mut cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
            factors
                .iter()
                .skip(prior_prefix)
                .fold(0.0_f32, |sum, factor| sum + factor.objective_cost as f32) as f64
        } else {
            factors
                .iter()
                .skip(prior_prefix)
                .map(|factor| factor.objective_cost)
                .sum()
        };
        if prior_prefix != 0 {
            let prior_error = self.prior.as_ref().map_or_else(
                || factors.first().map_or(0.0, |factor| factor.objective_cost),
                |prior| self.marginal_prior_reduced_cost(state, prior),
            );
            cost = if self.scalar_mode == ScalarMode::UpstreamF32 {
                (cost as f32 + prior_error as f32) as f64
            } else {
                cost + prior_error
            };
        }
        Ok(LmLinearization { factors, cost })
    }

    fn cost(&self, state: &DVector<f64>) -> Result<f64, LmFailure> {
        let linearization = self.linearize(state)?;
        // Upstream computeError evaluates the complete objective. Landmark
        // increments are recovered before trial_cost below, so there is no
        // need for the former projected-residual surrogate.
        let cost = linearization.cost;
        if cost.is_finite() {
            Ok(cost)
        } else {
            Err(LmFailure::NonFinite)
        }
    }

    fn trial_cost(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
    ) -> Result<f64, LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            self.trial_problem_with_step_owned(state, step)
                .owned_trial_objective(trial)
        } else {
            WindowTrialView::from_step(self, state, step, trial).cost()
        }
    }

    fn trial_cost_timed(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<f64, LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            // Diagnostic/probe runs retain the historical owned clone so all
            // audit boundaries continue to observe the complete candidate
            // problem and rejected steps cannot overwrite accepted sidecars.
            let trial_problem = timing.measure(TimingBucket::LmTrialConstructStep, || {
                self.trial_problem_with_step_owned(state, step)
            });
            timing.measure(TimingBucket::LmTrialCost, || {
                trial_problem.owned_trial_objective(trial)
            })
        } else {
            let trial_view = WindowTrialView::from_step_timed(self, state, step, trial, timing);
            timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())
        }
    }

    fn trial_cost_timed_with_token(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }
        if diagnostic_env_active() {
            // The clean solver is the only caller of this hook.  Keep a
            // defensive empty-token diagnostic branch so direct callers and
            // future retained paths continue to use the owned fallback.
            let trial_problem = timing.measure(TimingBucket::LmTrialConstructStep, || {
                self.trial_problem_with_step_owned(state, step)
            });
            timing
                .measure(TimingBucket::LmTrialCost, || {
                    trial_problem.owned_trial_objective(trial)
                })
                .map(|cost| (cost, LmTrialToken::default()))
        } else {
            let trial_view = WindowTrialView::from_step_timed(self, state, step, trial, timing);
            let cost = timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())?;
            Ok((
                cost,
                LmTrialToken::with_bound_landmark_steps(
                    trial_view.landmark_steps,
                    self.lm_trial_binding(state, step),
                ),
            ))
        }
    }

    fn trial_cost_timed_with_preparation(
        &self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        trial: &DVector<f64>,
        preparation: Option<LmTrialPreparation>,
        timing: &mut TimingBreakdown,
    ) -> Result<(f64, LmTrialToken), LmFailure> {
        if state.len() != self.state_dof()
            || step.len() != self.state_dof()
            || trial.len() != self.state_dof()
        {
            return Err(LmFailure::LinearSolve);
        }

        // Diagnostic/probe runs intentionally retain the owned clone and the
        // historical token path.  The clean preparation is one-shot, so it
        // must be dropped before delegating rather than accidentally reused.
        if diagnostic_env_active() || self.scalar_mode != ScalarMode::UpstreamF32 {
            drop(preparation);
            return self.trial_cost_timed_with_token(state, step, trial, timing);
        }

        let Some(preparation) = preparation else {
            // Generic/legacy callers may not provide a compact payload.  Keep
            // the established eager recovery behavior in that case.
            return self.trial_cost_timed_with_token(state, step, trial, timing);
        };

        let trial_view = if timing.enabled() {
            let outer_started = timing.start();
            let recovery_started = timing.start();
            let prepared_steps = self.landmark_steps_from_preparation(state, step, preparation);
            timing.finish(TimingBucket::LmTrialLandmarkRecovery, recovery_started);
            let increments = match prepared_steps {
                Ok(increments) => increments,
                Err(error) => {
                    timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
                    return Err(error);
                }
            };
            let view = {
                let mut timing_sink = EnabledTrialConstructTiming { timing };
                WindowTrialView::from_step_with_landmark_steps_timing(
                    self,
                    state,
                    step,
                    trial,
                    increments,
                    &mut timing_sink,
                )
            };
            timing.finish(TimingBucket::LmTrialConstructStep, outer_started);
            view
        } else {
            let increments = self.landmark_steps_from_preparation(state, step, preparation)?;
            WindowTrialView::from_step_with_landmark_steps(self, state, step, trial, increments)
        };

        let cost = timing.measure(TimingBucket::LmTrialCost, || trial_view.cost())?;
        Ok((
            cost,
            LmTrialToken::with_bound_landmark_steps(
                trial_view.landmark_steps,
                self.lm_trial_binding(state, step),
            ),
        ))
    }

    fn accept_step(&mut self, state: &DVector<f64>, step: &DVector<f64>) -> Result<(), LmFailure> {
        let increments = self.landmark_steps(state, step);
        self.accept_step_with_landmark_steps(state, step, &increments)
    }

    fn accept_step_with_token(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        token: LmTrialToken,
    ) -> Result<(), LmFailure> {
        let expected_binding = self.lm_trial_binding(state, step);
        let Some(increments) = token
            .take_landmark_steps_for(expected_binding)
            .map_err(|_| LmFailure::LinearSolve)?
        else {
            return self.accept_step(state, step);
        };
        if increments.len() != self.landmarks.len() {
            return Err(LmFailure::LinearSolve);
        }
        self.accept_step_with_landmark_steps(state, step, &increments)
    }
}

impl WindowProblem {
    fn accept_step_with_landmark_steps(
        &mut self,
        state: &DVector<f64>,
        step: &DVector<f64>,
        increments: &[Option<Vector3<f64>>],
    ) -> Result<(), LmFailure> {
        let (accepted_poses, accepted_states) = self.apply_step_full(state, step);
        let scalar_mode = self.scalar_mode;
        let state_base_offset = self.poses.len() * POSE_DOF;
        self.apply_landmark_steps(increments);
        for (index, (pose, accepted_pose)) in self.poses.iter_mut().zip(accepted_poses).enumerate()
        {
            pose.pose = accepted_pose;
            pose.stored_current_pose = pose.pose.clone();
            let increment = step.rows(index * POSE_DOF, POSE_DOF).into_owned();
            accumulate_linearized_delta(&mut pose.linearized_delta, &increment, scalar_mode);
        }
        for (index, (state, accepted_state)) in
            self.states.iter_mut().zip(accepted_states).enumerate()
        {
            state.nav = accepted_state;
            if !state.linearized {
                state.linearized_nav = state.nav.clone();
                state.linearized_delta.fill(0.0);
            } else {
                let increment = step
                    .rows(state_base_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
                    .into_owned();
                accumulate_linearized_delta(&mut state.linearized_delta, &increment, scalar_mode);
                state.stored_current_nav = state.nav.clone();
            }
        }
        Ok(())
    }
}

/// Evaluate the constant-free marginal-prior error used by Basalt's
/// `computeMargPriorError`/`linearizeMargPrior` contract.
///
/// The square-root prior stores `J` and `b = Jᵀr` at its FEJ point.  At the
/// current point the source evaluates
/// `deltaᵀ Jᵀ (0.5 J delta + b)` and intentionally omits `0.5 rᵀr`.
/// Keep the scalar cast at this helper boundary so the production f32 path
/// cannot accidentally evaluate the reduced error in f64 and round only the
/// final result.
/// Pinned Eigen prior inner product for 21 rows (ELF 0x3b23e1..0x3b2586).
/// Two eight-lane products are reduced before the five scalar FMA tails.
fn prior_inner_product_21_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 21);
    assert_eq!(right.len(), 21);
    let products: [f32; 16] = std::array::from_fn(|i| left[i] * right[i]);
    let lanes: [f32; 8] = std::array::from_fn(|i| products[i] + products[i + 8]);
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
    for i in 16..21 {
        sum = right[i].mul_add(left[i], sum);
    }
    sum
}

/// Candidate for Eigen's 51-element packet dot used by the frame38 marginal
/// prior. Six Packet8 products are combined by the evaluator's recursive
/// packet tree, then the three scalar coefficients are contracted into the
/// reduced packet sum.
fn prior_inner_product_51_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 51);
    assert_eq!(right.len(), 51);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_fifth = left[i + 32].mul_add(right[i + 32], left[i] * right[i]);
        let second_sixth = left[i + 40].mul_add(right[i + 40], left[i + 8] * right[i + 8]);
        first_fifth + ((left[i + 16] * right[i + 16] + left[i + 24] * right[i + 24]) + second_sixth)
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    let mut total = (half[0] + half[2]) + (half[1] + half[3]);
    for i in 48..51 {
        total = right[i].mul_add(left[i], total);
    }
    total
}

fn prior_inner_product_57_f32(left: &[f32], right: &[f32]) -> f32 {
    assert_eq!(left.len(), 57);
    assert_eq!(right.len(), 57);
    let lanes: [f32; 8] = std::array::from_fn(|i| {
        let first_fifth = left[i + 32].mul_add(right[i + 32], left[i] * right[i]);
        let second_sixth = left[i + 40].mul_add(right[i + 40], left[i + 8] * right[i + 8]);
        let third_seventh = left[i + 48].mul_add(right[i + 48], left[i + 16] * right[i + 16]);
        first_fifth + ((third_seventh + left[i + 24] * right[i + 24]) + second_sixth)
    });
    let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
    right[56].mul_add(left[56], (half[0] + half[2]) + (half[1] + half[3]))
}

fn upstream_marginal_prior_error_57_candidate(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
) -> f64 {
    assert_eq!(jacobian.shape(), (57, 57));
    assert_eq!((rhs.len(), delta.len()), (57, 57));
    let mut left = [0.0_f32; 57];
    let mut right = [0.0_f32; 57];
    for row in 0..57 {
        for col in 0..57 {
            let j = jacobian[(row, col)] as f32;
            let d = delta[col] as f32;
            left[row] = j.mul_add(d, left[row]);
            right[row] = (0.5_f32 * j).mul_add(d, right[row]);
        }
        right[row] += rhs[row] as f32;
    }
    f64::from(prior_inner_product_57_f32(&left, &right))
}

fn upstream_marginal_prior_error_51_candidate(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
) -> f64 {
    assert_eq!(jacobian.shape(), (51, 51));
    assert_eq!((rhs.len(), delta.len()), (51, 51));
    let mut left = [0.0_f32; 51];
    let mut right = [0.0_f32; 51];
    for row in 0..51 {
        for col in 0..51 {
            let j = jacobian[(row, col)] as f32;
            let d = delta[col] as f32;
            left[row] = j.mul_add(d, left[row]);
            right[row] = (0.5_f32 * j).mul_add(d, right[row]);
        }
        right[row] += rhs[row] as f32;
    }
    f64::from(prior_inner_product_51_f32(&left, &right))
}

fn upstream_marginal_prior_error(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    delta: &DVector<f64>,
    scalar_mode: ScalarMode,
) -> f64 {
    debug_assert_eq!(jacobian.ncols(), delta.len());
    debug_assert_eq!(jacobian.nrows(), rhs.len());
    if scalar_mode == ScalarMode::UpstreamF32 && jacobian.shape() == (57, 57) {
        return upstream_marginal_prior_error_57_candidate(jacobian, rhs, delta);
    }
    if scalar_mode == ScalarMode::UpstreamF32 && jacobian.shape() == (51, 51) {
        return upstream_marginal_prior_error_51_candidate(jacobian, rhs, delta);
    }
    if scalar_mode == ScalarMode::UpstreamF32
        && matches!(jacobian.shape(), (33, 33) | (39, 39) | (45, 45))
    {
        // Pinned Eigen sqrt-prior evaluator: column-major FMA GEMVs,
        // followed by four Packet8 products and a scalar FMA remainder.
        // Keep the half-scaled matrix product separate from the left GEMV.
        let size = jacobian.nrows();
        let mut left = [0.0_f32; 45];
        let mut right = [0.0_f32; 45];
        for row in 0..size {
            for col in 0..size {
                let j = jacobian[(row, col)] as f32;
                let d = delta[col] as f32;
                left[row] = j.mul_add(d, left[row]);
                right[row] = (0.5_f32 * j).mul_add(d, right[row]);
            }
            right[row] += rhs[row] as f32;
        }
        let lanes: [f32; 8] = if size == 45 {
            // The 45-row Eigen inner-product kernel fuses packet five into packet
            // one, then adds the middle three packets before the Packet8 predux.
            std::array::from_fn(|i| {
                let first_and_fifth = left[i + 32].mul_add(right[i + 32], left[i] * right[i]);
                let middle = left[i + 8] * right[i + 8]
                    + (left[i + 16] * right[i + 16] + left[i + 24] * right[i + 24]);
                first_and_fifth + middle
            })
        } else {
            let products: [f32; 32] = std::array::from_fn(|i| left[i] * right[i]);
            std::array::from_fn(|i| {
                products[i] + ((products[i + 16] + products[i + 24]) + products[i + 8])
            })
        };
        let half: [f32; 4] = std::array::from_fn(|i| lanes[i] + lanes[i + 4]);
        let mut total = (half[0] + half[2]) + (half[1] + half[3]);
        let tail_start = if size == 45 { 40 } else { 32 };
        for i in tail_start..size {
            total = right[i].mul_add(left[i], total);
        }
        return total as f64;
    }
    if scalar_mode == ScalarMode::UpstreamF32 {
        let jacobian = DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, col| {
            jacobian[(row, col)] as f32
        });
        let delta = DVector::from_iterator(delta.len(), delta.iter().map(|value| *value as f32));
        let rhs = DVector::from_iterator(rhs.len(), rhs.iter().map(|value| *value as f32));
        let j_delta = &jacobian * &delta;
        let linearized_rhs = 0.5_f32 * &j_delta + rhs;
        if j_delta.len() == 21 {
            return prior_inner_product_21_f32(j_delta.as_slice(), linearized_rhs.as_slice())
                as f64;
        }
        (delta.transpose() * jacobian.transpose() * linearized_rhs)[(0, 0)] as f64
    } else {
        let j_delta = jacobian * delta;
        (delta.transpose() * jacobian.transpose() * (0.5 * j_delta + rhs))[(0, 0)]
    }
}

fn bias_walk_factor(
    state: &DVector<f64>,
    link: WindowImuLink,
    state_dof: usize,
    from_offset: Option<usize>,
    to_offset: Option<usize>,
    noise: BiasRandomWalkNoise,
    scalar_mode: ScalarMode,
) -> Option<WhitenedFactorRowStack> {
    let from_offset = from_offset?;
    let to_offset = to_offset?;
    if from_offset + NAV_STATE_DOF > state_dof || to_offset + NAV_STATE_DOF > state_dof {
        return None;
    }
    let from = decode_state_with_mode(state.rows(from_offset, NAV_STATE_DOF), scalar_mode);
    let to = decode_state_with_mode(state.rows(to_offset, NAV_STATE_DOF), scalar_mode);
    let factor = if scalar_mode == ScalarMode::UpstreamF32 {
        whitened_bias_random_walk_factor_upstream_f32(&from, &to, link.delta.delta_time, noise)
            .ok()?
    } else {
        whitened_bias_random_walk_factor(&from, &to, link.delta.delta_time, noise).ok()?
    };
    let mut jacobian = DMatrix::zeros(factor.residual.len(), state_dof);
    jacobian
        .view_mut((0, from_offset), (factor.residual.len(), NAV_STATE_DOF))
        .copy_from(&factor.state_jacobian.columns(0, NAV_STATE_DOF));
    jacobian
        .view_mut((0, to_offset), (factor.residual.len(), NAV_STATE_DOF))
        .copy_from(&factor.state_jacobian.columns(NAV_STATE_DOF, NAV_STATE_DOF));
    WhitenedFactorRowStack::new(
        jacobian,
        DMatrix::zeros(factor.residual.len(), 0),
        factor.residual,
    )
    .map(|factor| {
        factor
            .with_kind(FactorKind::Bias)
            .with_imu_link_offsets(from_offset, to_offset)
    })
}

pub fn flatten_states(states: &[WindowState]) -> DVector<f64> {
    let mut out = DVector::zeros(states.len() * NAV_STATE_DOF);
    for (index, state) in states.iter().enumerate() {
        out.rows_mut(index * NAV_STATE_DOF, NAV_STATE_DOF)
            .copy_from(&flatten_nav(&state.nav));
    }
    out
}

/// Flatten the mixed absolute order used by the square-root VIO system:
/// pose-only keyframes first, followed by full navigation states.
pub fn flatten_blocks(poses: &[WindowPose], states: &[WindowState]) -> DVector<f64> {
    flatten_blocks_with_mode(poses, states, ScalarMode::ExtendedF64)
}

/// Flatten the state in the scalar representation owned by the active
/// estimator.  The upstream float estimator keeps an SO3 object as the state
/// block and applies local rotation increments to that object; it does not
/// round-trip the initial quaternion through a double-precision log/exp.
/// Keep the three stored rotation lanes as the f32 quaternion xyz lanes in
/// this internal vector representation so decoding preserves the source f32
/// components exactly.  The LM step still remains a local scaled-axis
/// increment (see `apply_step`).
pub fn flatten_blocks_with_mode(
    poses: &[WindowPose],
    states: &[WindowState],
    scalar_mode: ScalarMode,
) -> DVector<f64> {
    let mut out = DVector::zeros(poses.len() * POSE_DOF + states.len() * NAV_STATE_DOF);
    for (index, pose) in poses.iter().enumerate() {
        out.rows_mut(index * POSE_DOF, POSE_DOF)
            .copy_from(&flatten_pose_with_mode(&pose.pose, scalar_mode));
    }
    let state_offset = poses.len() * POSE_DOF;
    for (index, state) in states.iter().enumerate() {
        out.rows_mut(state_offset + index * NAV_STATE_DOF, NAV_STATE_DOF)
            .copy_from(&flatten_nav_with_mode(&state.nav, scalar_mode));
    }
    out
}

pub fn write_states(states: &mut [WindowState], vector: &DVector<f64>) {
    for (index, state) in states.iter_mut().enumerate() {
        state.nav = decode_state(vector.rows(index * NAV_STATE_DOF, NAV_STATE_DOF));
        if !state.linearized {
            state.linearized_nav = state.nav.clone();
            state.linearized_delta.fill(0.0);
        } else {
            state.stored_current_nav = state.nav.clone();
        }
    }
}

pub fn write_blocks(poses: &mut [WindowPose], states: &mut [WindowState], vector: &DVector<f64>) {
    write_blocks_with_mode(poses, states, vector, ScalarMode::ExtendedF64);
}

pub fn write_blocks_with_mode(
    poses: &mut [WindowPose],
    states: &mut [WindowState],
    vector: &DVector<f64>,
    scalar_mode: ScalarMode,
) {
    for (index, pose) in poses.iter_mut().enumerate() {
        let block = vector.rows(index * POSE_DOF, POSE_DOF);
        let mut decoded = decode_pose_with_mode(&block, scalar_mode);
        // A solved UpstreamF32 block is paired with the full pose sidecar
        // maintained by WindowProblem.  Preserve its propagated w component
        // when the compact xyz lanes identify the same pose; arbitrary
        // callers that provide a different chart still get the historical
        // reconstruction behavior.
        if scalar_mode == ScalarMode::UpstreamF32
            && quaternion_xyz_matches(&[block[3], block[4], block[5]], &pose.pose.rotation)
        {
            decoded.rotation = pose.pose.rotation.clone();
        }
        pose.pose = decoded;
        pose.stored_current_pose = pose.pose.clone();
    }
    let state_offset = poses.len() * POSE_DOF;
    for (index, state) in states.iter_mut().enumerate() {
        let block = vector.rows(state_offset + index * NAV_STATE_DOF, NAV_STATE_DOF);
        let mut decoded = decode_state_with_mode(&block, scalar_mode);
        if scalar_mode == ScalarMode::UpstreamF32
            && quaternion_xyz_matches(
                &[block[3], block[4], block[5]],
                &state.nav.imu_to_world.rotation,
            )
        {
            decoded.imu_to_world.rotation = state.nav.imu_to_world.rotation.clone();
        }
        state.nav = decoded;
        // Upstream `applyInc` mutates `state_linearized` while a block is
        // still non-linearized.  Keep the Rust FEJ sidecar at that moving
        // point; frozen blocks retain their original linearization point.
        if !state.linearized {
            state.linearized_nav = state.nav.clone();
            state.linearized_delta.fill(0.0);
        } else {
            state.stored_current_nav = state.nav.clone();
        }
    }
}

pub fn decode_states(vector: &DVector<f64>, count: usize) -> Vec<BasaltNavState> {
    (0..count)
        .map(|index| decode_state(vector.rows(index * NAV_STATE_DOF, NAV_STATE_DOF)))
        .collect()
}

pub fn flatten_nav(nav: &BasaltNavState) -> DVector<f64> {
    flatten_nav_with_mode(nav, ScalarMode::ExtendedF64)
}

pub fn flatten_nav_with_mode(nav: &BasaltNavState, scalar_mode: ScalarMode) -> DVector<f64> {
    let rotation = if scalar_mode == ScalarMode::UpstreamF32 {
        let quaternion = quat_f32_unchecked_from_f64(&nav.imu_to_world.rotation);
        Vector3::new(
            quaternion.i as f64,
            quaternion.j as f64,
            quaternion.k as f64,
        )
    } else {
        nav.imu_to_world.rotation.scaled_axis()
    };
    let cast = |value: f64| {
        if scalar_mode == ScalarMode::UpstreamF32 {
            (value as f32) as f64
        } else {
            value
        }
    };
    DVector::from_iterator(
        NAV_STATE_DOF,
        nav.imu_to_world
            .translation
            .iter()
            .map(|value| cast(*value))
            .chain(rotation.iter().copied())
            .chain(nav.velocity_world_m_s.iter().copied())
            .chain(nav.gyro_bias_rad_s.iter().copied())
            .chain(nav.accel_bias_m_s2.iter().copied())
            .map(cast),
    )
}

pub fn flatten_pose(pose: &SE3) -> DVector<f64> {
    flatten_pose_with_mode(pose, ScalarMode::ExtendedF64)
}

pub(crate) fn flatten_pose_with_mode(pose: &SE3, scalar_mode: ScalarMode) -> DVector<f64> {
    let rotation = if scalar_mode == ScalarMode::UpstreamF32 {
        let quaternion = quat_f32_unchecked_from_f64(&pose.rotation);
        Vector3::new(
            quaternion.i as f64,
            quaternion.j as f64,
            quaternion.k as f64,
        )
    } else {
        pose.rotation.scaled_axis()
    };
    let cast = |value: f64| {
        if scalar_mode == ScalarMode::UpstreamF32 {
            (value as f32) as f64
        } else {
            value
        }
    };
    DVector::from_iterator(
        POSE_DOF,
        pose.translation
            .iter()
            .map(|value| cast(*value))
            .chain(rotation.iter().copied())
            .map(cast),
    )
}

pub fn decode_pose(vector: impl AsRef<[f64]>) -> SE3 {
    decode_pose_with_mode(vector, ScalarMode::ExtendedF64)
}

fn decode_pose_with_mode(vector: impl AsRef<[f64]>, scalar_mode: ScalarMode) -> SE3 {
    let vector = vector.as_ref();
    SE3::new(
        decode_rotation_with_mode(&vector[3..6], scalar_mode),
        Vector3::new(
            cast_component(vector[0], scalar_mode),
            cast_component(vector[1], scalar_mode),
            cast_component(vector[2], scalar_mode),
        ),
    )
}

pub fn local_pose_difference(current: &SE3, reference: &SE3) -> DVector<f64> {
    // Basalt's PoseState::incPose is not an SE(3) exponential update: the
    // translation is additive in world coordinates and the rotation is a
    // left SO(3) increment.  FEJ must use the inverse of that exact chart,
    // rather than SE3::log(), whose translational component contains the
    // SO(3) left-Jacobian coupling.
    let rotation = (current.rotation * reference.rotation.inverse()).scaled_axis();
    DVector::from_iterator(
        POSE_DOF,
        current
            .translation
            .iter()
            .zip(reference.translation.iter())
            .map(|(current, reference)| current - reference)
            .chain(rotation.iter().copied()),
    )
}

const fn block_dof(kind: WindowBlockKind) -> usize {
    match kind {
        WindowBlockKind::Pose => POSE_DOF,
        WindowBlockKind::StatePose => POSE_DOF,
        WindowBlockKind::State => NAV_STATE_DOF,
    }
}

/// Left-local SE3 boxminus plus additive navigation components. This is the
/// coordinate chart used by visual/IMU pose Jacobians and by FEJ priors.
pub fn local_state_difference(
    current: &BasaltNavState,
    reference: &BasaltNavState,
) -> DVector<f64> {
    let rotation =
        (current.imu_to_world.rotation * reference.imu_to_world.rotation.inverse()).scaled_axis();
    DVector::from_iterator(
        NAV_STATE_DOF,
        current
            .imu_to_world
            .translation
            .iter()
            .zip(reference.imu_to_world.translation.iter())
            .map(|(current, reference)| current - reference)
            .chain(rotation.iter().copied())
            .chain(
                (current.velocity_world_m_s - reference.velocity_world_m_s)
                    .iter()
                    .copied(),
            )
            .chain(
                (current.gyro_bias_rad_s - reference.gyro_bias_rad_s)
                    .iter()
                    .copied(),
            )
            .chain(
                (current.accel_bias_m_s2 - reference.accel_bias_m_s2)
                    .iter()
                    .copied(),
            ),
    )
}

/// Accumulate the tangent exactly at the solver's scalar boundary.  Upstream
/// stores `delta += inc` in `float`; reconstructing it from two poses would
/// replace the source's sum of local rotations with a nonlinear box-minus.
fn accumulate_linearized_delta(
    delta: &mut DVector<f64>,
    increment: &DVector<f64>,
    scalar_mode: ScalarMode,
) {
    if delta.len() != increment.len() {
        return;
    }
    if scalar_mode == ScalarMode::UpstreamF32 {
        for (lhs, rhs) in delta.iter_mut().zip(increment.iter()) {
            *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
        }
    } else {
        *delta += increment;
    }
}

pub fn decode_state(vector: impl AsRef<[f64]>) -> BasaltNavState {
    decode_state_with_mode(vector, ScalarMode::ExtendedF64)
}

fn decode_state_with_mode(vector: impl AsRef<[f64]>, scalar_mode: ScalarMode) -> BasaltNavState {
    let vector = vector.as_ref();
    let translation = Vector3::new(
        cast_component(vector[0], scalar_mode),
        cast_component(vector[1], scalar_mode),
        cast_component(vector[2], scalar_mode),
    );
    let rotation = decode_rotation_with_mode(&vector[3..6], scalar_mode);
    BasaltNavState {
        imu_to_world: SE3::new(rotation, translation),
        velocity_world_m_s: Vector3::new(
            cast_component(vector[6], scalar_mode),
            cast_component(vector[7], scalar_mode),
            cast_component(vector[8], scalar_mode),
        ),
        gyro_bias_rad_s: Vector3::new(
            cast_component(vector[9], scalar_mode),
            cast_component(vector[10], scalar_mode),
            cast_component(vector[11], scalar_mode),
        ),
        accel_bias_m_s2: Vector3::new(
            cast_component(vector[12], scalar_mode),
            cast_component(vector[13], scalar_mode),
            cast_component(vector[14], scalar_mode),
        ),
    }
}

fn cast_component(value: f64, scalar_mode: ScalarMode) -> f64 {
    if scalar_mode == ScalarMode::UpstreamF32 {
        (value as f32) as f64
    } else {
        value
    }
}

fn quat_f32_unchecked_from_f64(rotation: &UnitQuaternion<f64>) -> UnitQuaternion<f32> {
    let q = rotation.quaternion();
    UnitQuaternion::new_unchecked(Quaternion::new(
        q.w as f32, q.i as f32, q.j as f32, q.k as f32,
    ))
}

fn quat_f64_unchecked_from_f32(rotation: &UnitQuaternion<f32>) -> UnitQuaternion<f64> {
    let q = rotation.quaternion();
    UnitQuaternion::new_unchecked(Quaternion::new(
        q.w as f64, q.i as f64, q.j as f64, q.k as f64,
    ))
}

/// Return whether a compact UpstreamF32 block still refers to the quaternion
/// carried by the full state sidecar.  The comparison is deliberately at the
/// source scalar boundary: no norm, log/exp, or epsilon chart can recover the
/// fourth lane once it has been discarded.
fn quaternion_xyz_matches(vector: &[f64], rotation: &UnitQuaternion<f64>) -> bool {
    if vector.len() != 3 {
        return false;
    }
    let quaternion = quat_f32_unchecked_from_f64(rotation);
    vector[0] as f32 == quaternion.i
        && vector[1] as f32 == quaternion.j
        && vector[2] as f32 == quaternion.k
}

fn decode_rotation_with_mode(vector: &[f64], scalar_mode: ScalarMode) -> UnitQuaternion<f64> {
    if scalar_mode != ScalarMode::UpstreamF32 {
        return UnitQuaternion::from_scaled_axis(Vector3::new(vector[0], vector[1], vector[2]));
    }
    let x = vector[0] as f32;
    let y = vector[1] as f32;
    let z = vector[2] as f32;
    let norm_sq = x * x + y * y + z * z;
    let w = (1.0_f32 - norm_sq).max(0.0).sqrt();
    let rotation = UnitQuaternion::new_unchecked(Quaternion::new(w, x, y, z));
    quat_f64_unchecked_from_f32(&rotation)
}

/// Converts local landmark columns to state-only square-root rows for
/// MargData and for Schur-style prior construction.
pub fn projected_rows(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
) -> Vec<WhitenedFactorRowStack> {
    factors
        .iter()
        .filter_map(|factor| {
            let (jacobian, rhs, _) = landmark_nullspace_projection(factor, tolerance);
            WhitenedFactorRowStack::new(jacobian, DMatrix::zeros(rhs.len(), 0), rhs)
                .map(|projected| projected.with_kind(factor.kind))
        })
        .collect()
}

/// Project factor rows at the scalar boundary used by the selected upstream
/// estimator.  The pinned `LinearizationAbsQR<float>` stores each landmark
/// block as f32 before its Householder tail is copied into the global Q2
/// stack; widening a later f64 projection back to f32 changes both the tail
/// bits and the subsequent marginal QR schedule.
pub fn projected_rows_with_mode(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> Vec<WhitenedFactorRowStack> {
    factors
        .iter()
        .filter_map(|factor| {
            let (jacobian, rhs) = if scalar_mode == ScalarMode::UpstreamF32 {
                let (jacobian, rhs, _) = landmark_nullspace_projection_f32(factor, tolerance);
                (
                    DMatrix::from_fn(jacobian.nrows(), jacobian.ncols(), |row, column| {
                        jacobian[(row, column)] as f64
                    }),
                    DVector::from_iterator(rhs.len(), rhs.iter().copied().map(f64::from)),
                )
            } else {
                let (jacobian, rhs, _) = landmark_nullspace_projection(factor, tolerance);
                (jacobian, rhs)
            };
            WhitenedFactorRowStack::new(jacobian, DMatrix::zeros(rhs.len(), 0), rhs)
                .map(|projected| projected.with_kind(factor.kind))
        })
        .collect()
}

/// Build the global Q2 row stack in the order emitted by Basalt's
/// `LinearizationAbsQR::get_dense_Q2Jp_Q2r`: selected visual landmark tails,
/// then IMU/bias rows, then the existing marginal prior.  The public factor
/// list intentionally keeps the prior first for AOM prefix semantics, so the
/// marginalization boundary must restore this source order explicitly.
pub fn projected_rows_native_q2(
    factors: &[WhitenedFactorRowStack],
    tolerance: f64,
    scalar_mode: ScalarMode,
) -> Vec<WhitenedFactorRowStack> {
    let projected = projected_rows_with_mode(factors, tolerance, scalar_mode);
    let mut ordered = Vec::with_capacity(projected.len());
    for phase in 0..3 {
        for (factor, row) in factors.iter().zip(projected.iter()) {
            let is_visual = factor.landmark_jacobian.ncols() != 0;
            let is_imu = matches!(factor.kind, FactorKind::Imu | FactorKind::Bias);
            let selected = match phase {
                0 => is_visual,
                1 => is_imu,
                _ => !is_visual && !is_imu,
            };
            if selected {
                ordered.push(row.clone());
            }
        }
    }
    ordered
}

pub fn stack_rows(
    rows: &[WhitenedFactorRowStack],
    state_dof: usize,
) -> (DMatrix<f64>, DVector<f64>) {
    let total_rows = rows.iter().map(WhitenedFactorRowStack::rows).sum();
    let mut jacobian = DMatrix::zeros(total_rows, state_dof);
    let mut rhs = DVector::zeros(total_rows);
    let mut row = 0;
    for factor in rows {
        if factor.state_jacobian.ncols() != state_dof {
            continue;
        }
        jacobian
            .view_mut((row, 0), (factor.rows(), state_dof))
            .copy_from(&factor.state_jacobian);
        rhs.rows_mut(row, factor.rows()).copy_from(&factor.residual);
        row += factor.rows();
    }
    (jacobian, rhs)
}

pub fn marginalize_prior(
    rows: &[WhitenedFactorRowStack],
    _state_dof: usize,
    states: &[WindowState],
    dropped_ids: &[u64],
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    marginalize_mixed_prior(rows, &[], states, &[], dropped_ids, &[], state)
}

/// Marginalize a mixed Basalt AOM.  `vel_bias_ids` denotes keyframe states
/// whose six pose columns survive as a pose-only block while columns 6..14
/// leave the active navigation state.  The returned prior is ordered as
/// retained pose blocks followed by retained full navigation blocks.
pub fn marginalize_mixed_prior(
    rows: &[WhitenedFactorRowStack],
    poses: &[WindowPose],
    states: &[WindowState],
    dropped_pose_ids: &[u64],
    dropped_state_ids: &[u64],
    vel_bias_ids: &[u64],
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    marginalize_mixed_prior_with_mode(
        rows,
        poses,
        states,
        dropped_pose_ids,
        dropped_state_ids,
        vel_bias_ids,
        ScalarMode::UpstreamF32,
        state,
    )
}

/// Scalar-explicit mixed prior boundary.  The compatibility estimator uses
/// [`ScalarMode::UpstreamF32`] (the pinned `SqrtKeypointVioEstimator<float>`),
/// while the opt-in extended numerical mode retains the f64 reference path.
/// Keeping the old [`marginalize_mixed_prior`] wrapper above preserves the
/// faithful-f32 default for callers that do not carry a scalar mode.
pub fn marginalize_mixed_prior_with_mode(
    rows: &[WhitenedFactorRowStack],
    poses: &[WindowPose],
    states: &[WindowState],
    dropped_pose_ids: &[u64],
    dropped_state_ids: &[u64],
    vel_bias_ids: &[u64],
    scalar_mode: ScalarMode,
    state: &DVector<f64>,
) -> Option<WindowPrior> {
    let state_dof = poses.len() * POSE_DOF + states.len() * NAV_STATE_DOF;
    if state.len() != state_dof {
        return None;
    }
    let projected = projected_rows_native_q2(rows, 1e-10, scalar_mode);
    let (jacobian, rhs) = stack_rows(&projected, state_dof);
    let mut keep_ids = Vec::new();
    let mut block_kinds = Vec::new();
    let mut keep_columns = Vec::new();
    let mut marginal_columns = Vec::new();

    for (index, pose) in poses.iter().enumerate() {
        let columns = (index * POSE_DOF..(index + 1) * POSE_DOF).collect::<Vec<_>>();
        if dropped_pose_ids.contains(&pose.frame_id) {
            marginal_columns.extend(columns);
        } else {
            keep_ids.push(pose.frame_id);
            block_kinds.push(WindowBlockKind::Pose);
            keep_columns.extend(columns);
        }
    }

    let state_offset = poses.len() * POSE_DOF;
    for (index, nav) in states.iter().enumerate() {
        let offset = state_offset + index * NAV_STATE_DOF;
        let all_columns = (offset..offset + NAV_STATE_DOF).collect::<Vec<_>>();
        if dropped_state_ids.contains(&nav.frame_id)
            || (vel_bias_ids.contains(&nav.frame_id) && dropped_pose_ids.contains(&nav.frame_id))
        {
            marginal_columns.extend(all_columns);
        } else if vel_bias_ids.contains(&nav.frame_id) {
            keep_ids.push(nav.frame_id);
            block_kinds.push(WindowBlockKind::Pose);
            keep_columns.extend(offset..offset + POSE_DOF);
            marginal_columns.extend(offset + POSE_DOF..offset + NAV_STATE_DOF);
        } else {
            keep_ids.push(nav.frame_id);
            block_kinds.push(WindowBlockKind::State);
            keep_columns.extend(all_columns);
        }
    }
    if keep_ids.is_empty() {
        return None;
    }
    // Marginalization is performed at the solved current state, but Basalt
    // stores the resulting prior back at each block's original FEJ point.
    // `computeDelta()` followed by `marg_data.b -= marg_data.H * delta` is the
    // source operation.  Preserve both pieces explicitly instead of silently
    // re-anchoring every prior at the latest state (which makes the reduced
    // prior error lose its negative, constant-free value).
    let mut linearized_blocks = DVector::zeros(state_dof);
    for (index, pose) in poses.iter().enumerate() {
        linearized_blocks
            .rows_mut(index * POSE_DOF, POSE_DOF)
            .copy_from(&flatten_pose_with_mode(&pose.linearized_pose, scalar_mode));
    }
    for (index, nav) in states.iter().enumerate() {
        let offset = state_offset + index * NAV_STATE_DOF;
        linearized_blocks
            .rows_mut(offset, NAV_STATE_DOF)
            .copy_from(&flatten_nav_with_mode(&nav.linearized_nav, scalar_mode));
    }
    let fej_point = DVector::from_iterator(
        keep_columns.len(),
        keep_columns.iter().map(|column| linearized_blocks[*column]),
    );
    let mut keep_delta = DVector::zeros(keep_columns.len());
    let mut keep_offset = 0;
    for (_index, pose) in poses.iter().enumerate() {
        if dropped_pose_ids.contains(&pose.frame_id) {
            continue;
        }
        keep_delta
            .rows_mut(keep_offset, POSE_DOF)
            .copy_from(&pose.linearized_delta);
        keep_offset += POSE_DOF;
    }
    for nav in states {
        if dropped_state_ids.contains(&nav.frame_id)
            || (vel_bias_ids.contains(&nav.frame_id) && dropped_pose_ids.contains(&nav.frame_id))
        {
            continue;
        }
        if vel_bias_ids.contains(&nav.frame_id) {
            keep_delta
                .rows_mut(keep_offset, POSE_DOF)
                .copy_from(&nav.linearized_delta.rows(0, POSE_DOF));
            keep_offset += POSE_DOF;
        } else {
            keep_delta
                .rows_mut(keep_offset, NAV_STATE_DOF)
                .copy_from(&nav.linearized_delta);
            keep_offset += NAV_STATE_DOF;
        }
    }
    debug_assert_eq!(keep_offset, keep_columns.len());
    let prior = match scalar_mode {
        ScalarMode::UpstreamF32 => sqrt_marginalize_upstream_f32(
            &jacobian,
            &rhs,
            &keep_columns,
            &marginal_columns,
            fej_point,
        )?,
        ScalarMode::ExtendedF64 => {
            sqrt_marginalize_upstream(&jacobian, &rhs, &keep_columns, &marginal_columns, fej_point)?
        }
    };
    if scalar_mode == ScalarMode::UpstreamF32 {
        remember_diagnostic_prior_pre(&WindowPrior {
            frame_ids: keep_ids.clone(),
            block_kinds: block_kinds.clone(),
            jacobian: prior.jacobian.clone(),
            rhs: prior.rhs.clone(),
            fej_point: prior.fej_point.clone(),
        });
    }
    let adjusted_rhs = if scalar_mode == ScalarMode::UpstreamF32 {
        let jacobian = DMatrix::from_fn(
            prior.jacobian.nrows(),
            prior.jacobian.ncols(),
            |row, col| prior.jacobian[(row, col)] as f32,
        );
        let delta = DVector::from_iterator(
            keep_delta.len(),
            keep_delta.iter().map(|value| *value as f32),
        );
        let stored_rhs =
            DVector::from_iterator(prior.rhs.len(), prior.rhs.iter().map(|value| *value as f32));
        let rhs = if jacobian.nrows() == 21
            && jacobian.ncols() == 21
            && stored_rhs.len() == 21
            && delta.len() == 21
        {
            WindowProblem::prior_reanchor_rhs_col_major_21_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (27, 27) && stored_rhs.len() == 27 && delta.len() == 27 {
            WindowProblem::prior_reanchor_rhs_col_major_27_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (33, 33) && stored_rhs.len() == 33 && delta.len() == 33 {
            WindowProblem::prior_reanchor_rhs_col_major_33_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (39, 39) && stored_rhs.len() == 39 && delta.len() == 39 {
            WindowProblem::prior_reanchor_rhs_col_major_39_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (45, 45) && stored_rhs.len() == 45 && delta.len() == 45 {
            WindowProblem::prior_reanchor_rhs_col_major_45_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (51, 51) && stored_rhs.len() == 51 && delta.len() == 51 {
            WindowProblem::prior_reanchor_rhs_col_major_51_f32(&jacobian, &stored_rhs, &delta)
        } else if jacobian.shape() == (57, 57) && stored_rhs.len() == 57 && delta.len() == 57 {
            WindowProblem::prior_reanchor_rhs_col_major_57_f32(&jacobian, &stored_rhs, &delta)
        } else {
            stored_rhs - jacobian * delta
        };
        DVector::from_iterator(rhs.len(), rhs.iter().copied().map(f64::from))
    } else {
        prior.rhs.clone() - &prior.jacobian * keep_delta.clone()
    };

    // One-shot marginalization boundary capture for the detached native
    // Q2Jp/Q2r audit.  This serializes only already-computed f32 casts and
    // never participates in the production arithmetic.
    if scalar_mode == ScalarMode::UpstreamF32 {
        if let Some(path) = diagnostic_env_snapshot().marg_trace.as_ref() {
            static CAPTURED: OnceLock<()> = OnceLock::new();
            // Diagnostic-only event selector for auditing later marginalization
            // boundaries.  The default remains the historical one-shot capture;
            // when VISLOC_BASALT_MARG_TRACE_EVENT=N is set, count calls in this
            // process and serialize only event N.  This never participates in
            // the production arithmetic.
            static EVENT_COUNT: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);
            let event_ordinal = EVENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let selected_event = diagnostic_env_snapshot().marg_trace_event;
            let capture_this_event = match selected_event {
                Some(wanted) => event_ordinal == wanted,
                None => CAPTURED.set(()).is_ok(),
            };
            if capture_this_event {
                let matrix_bits = |matrix: &DMatrix<f64>| {
                    json!({
                        "rows": matrix.nrows(),
                        "cols": matrix.ncols(),
                        "layout": "column_major",
                        "bits": matrix
                            .iter()
                            .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                            .collect::<Vec<_>>(),
                    })
                };
                let vector_bits = |vector: &DVector<f64>| {
                    json!(vector
                        .iter()
                        .map(|value| format!("{:08x}", (*value as f32).to_bits()))
                        .collect::<Vec<_>>())
                };
                // Keep the Q2 residual source order visible in the detached
                // boundary capture.  This is diagnostic-only metadata: the
                // stack has already been built above, so recording the
                // producer cannot affect the production arithmetic.
                let mut q2_sources = Vec::with_capacity(rhs.len());
                let mut q2_row = 0usize;
                let projected_by_factor = projected_rows_with_mode(rows, 1e-10, scalar_mode);
                // The remaining Q2r frontier is the second IMU link (the
                // sixth factor in the pre-marginalization row list).  When
                // the opt-in IMU audit is enabled, retain exactly that
                // factor's already-materialized source payload here so raw,
                // whitener, and whitened bits can be compared without
                // rebuilding a second arithmetic path.
                let q2_imu_factor6_input =
                    if diagnostic_env_snapshot().diagnostic_imu_rows.is_some() {
                        rows.get(6)
                            .filter(|factor| matches!(factor.kind, FactorKind::Imu))
                            .and_then(|factor| factor.imu_input_diagnostic.clone())
                    } else {
                        None
                    };
                for phase in 0..3 {
                    for (factor_index, factor) in rows.iter().enumerate() {
                        let is_visual = factor.landmark_jacobian.ncols() != 0;
                        let is_imu = matches!(factor.kind, FactorKind::Imu | FactorKind::Bias);
                        let selected = match phase {
                            0 => is_visual,
                            1 => is_imu,
                            _ => !is_visual && !is_imu,
                        };
                        if !selected {
                            continue;
                        }
                        let projected_factor = &projected_by_factor[factor_index];
                        for local_row in 0..projected_factor.residual.len() {
                            q2_sources.push(json!({
                                "q2_row": q2_row,
                                "phase": phase,
                                "factor_index": factor_index,
                                "kind": format!("{:?}", factor.kind),
                                "local_row": local_row,
                                "projected_rhs_f32": format!(
                                    "{:08x}",
                                    (projected_factor.residual[local_row] as f32).to_bits()
                                ),
                                "raw_residual_f32": factor
                                    .residual
                                    .get(local_row)
                                    .map(|value| format!("{:08x}", (*value as f32).to_bits())),
                            }));
                            q2_row += 1;
                        }
                    }
                }
                let payload = json!({
                    "schema": "visloc.m7im15.rust_marg_boundary.v1",
                    "event_ordinal": event_ordinal,
                    "keep_columns": keep_columns,
                    "marginal_columns": marginal_columns,
                    "q2_jacobian_f32": matrix_bits(&jacobian),
                    "q2_rhs_f32": vector_bits(&rhs),
                    "q2_rhs_sources": q2_sources,
                    "q2_imu_factor6_input": q2_imu_factor6_input,
                    "fej_point_f32": vector_bits(&prior.fej_point),
                    "keep_delta_f32": vector_bits(&keep_delta),
                    "prior_jacobian_f32": matrix_bits(&prior.jacobian),
                    "prior_rhs_before_delta_f32": vector_bits(&prior.rhs),
                    "prior_rhs_after_delta_f32": vector_bits(&adjusted_rhs),
                });
                if let Ok(mut file) = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(path)
                {
                    let _ = serde_json::to_writer_pretty(&mut file, &payload);
                    let _ = file.write_all(b"\n");
                }
            }
        }
    }
    Some(WindowPrior {
        frame_ids: keep_ids,
        block_kinds,
        jacobian: prior.jacobian,
        rhs: adjusted_rhs,
        fej_point: prior.fej_point,
    })
}

/// Basalt `MargHelper::marginalizeHelperSqrtToSqrt` in the pinned upstream
/// revision.  The input rows are already square-root/QR rows (including the
/// rows retained after each visual landmark's ABS_QR); no normal matrix is
/// formed here.  Columns are permuted to `[marg | keep]`, then a
/// rank-revealing Householder walk eliminates the leaving block.  Basalt's
/// cutoff is `sqrt(numeric_limits<Scalar>::epsilon())`, not the tighter
/// tolerance used by the landmark QR.
fn sqrt_marginalize_upstream(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    let rows = jacobian.nrows();
    let cols = jacobian.ncols();
    if rows == 0
        || rhs.len() != rows
        || keep_columns.len() + marginal_columns.len() != cols
        || fej_point.len() != keep_columns.len()
    {
        return None;
    }

    // Upstream receives sets, so every index is unique and the two sets form
    // a complete partition.  Keep the same contract at this Rust boundary;
    // silently omitting a column would otherwise change the prior dimension.
    let mut seen = vec![false; cols];
    for &column in keep_columns.iter().chain(marginal_columns.iter()) {
        if column >= cols || seen[column] {
            return None;
        }
        seen[column] = true;
    }
    if seen.iter().any(|present| !present) {
        return None;
    }

    let marg_size = marginal_columns.len();
    let mut permutation = Vec::with_capacity(cols);
    permutation.extend_from_slice(marginal_columns);
    permutation.extend_from_slice(keep_columns);
    let mut qj = DMatrix::zeros(rows, cols);
    for (new_column, &old_column) in permutation.iter().enumerate() {
        qj.column_mut(new_column)
            .copy_from(&jacobian.column(old_column));
    }
    let mut qr = rhs.clone();
    let rank_threshold = f64::EPSILON.sqrt();
    let mut total_rank = 0usize;
    let mut marginal_rank = 0usize;

    for column in 0..cols {
        if total_rank >= rows {
            break;
        }
        let row_start = total_rank;
        let norm = (row_start..rows)
            .map(|row| qj[(row, column)] * qj[(row, column)])
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return None;
        }
        if norm > rank_threshold {
            // This is Eigen's makeHouseholderInPlace convention: beta is the
            // signed pivot and the reflector is H = I - tau*v*v^T.
            let x0 = qj[(row_start, column)];
            let sign = if x0 >= 0.0 { 1.0 } else { -1.0 };
            let beta = -sign * norm;
            let v0 = x0 + sign * norm;
            let v_norm_sq = v0 * v0 + norm * norm - x0 * x0;
            if !v_norm_sq.is_finite() || v_norm_sq <= 0.0 {
                return None;
            }
            let tau = 2.0 / v_norm_sq;

            // Keep the sub-pivot entries in the current column until the RHS
            // reflector has been applied, exactly as upstream does.
            qj[(row_start, column)] = beta;
            for trailing in (column + 1)..cols {
                let mut dot = v0 * qj[(row_start, trailing)];
                for row in (row_start + 1)..rows {
                    dot += qj[(row, column)] * qj[(row, trailing)];
                }
                let scale = tau * dot;
                qj[(row_start, trailing)] -= scale * v0;
                for row in (row_start + 1)..rows {
                    qj[(row, trailing)] -= scale * qj[(row, column)];
                }
            }
            let mut dot = v0 * qr[row_start];
            for row in (row_start + 1)..rows {
                dot += qj[(row, column)] * qr[row];
            }
            let scale = tau * dot;
            qr[row_start] -= scale * v0;
            for row in (row_start + 1)..rows {
                qr[row] -= scale * qj[(row, column)];
            }
            for row in (row_start + 1)..rows {
                qj[(row, column)] = 0.0;
            }
            total_rank += 1;
        } else {
            for row in row_start..rows {
                qj[(row, column)] = 0.0;
            }
        }
        if column + 1 == marg_size {
            marginal_rank = total_rank;
        }
    }
    if marg_size == 0 {
        marginal_rank = 0;
    }
    let keep_rows = total_rank.saturating_sub(marginal_rank).max(1);
    let available_rows = rows.saturating_sub(marginal_rank);
    if available_rows == 0 || keep_rows > available_rows {
        return None;
    }
    let keep_size = keep_columns.len();
    let mut out_j = DMatrix::zeros(keep_rows, keep_size);
    for row in 0..keep_rows {
        out_j
            .row_mut(row)
            .copy_from(&qj.row(marginal_rank + row).columns(marg_size, keep_size));
    }
    let out_rhs =
        DVector::from_iterator(keep_rows, (0..keep_rows).map(|row| qr[marginal_rank + row]));
    Some(super::aom::SqrtPrior {
        jacobian: out_j,
        rhs: out_rhs,
        fej_point,
    })
}

/// Exact scalar-boundary variant of Basalt's SqrtToSqrt marginalization.
///
/// The estimator is instantiated as `MargHelper<float>` upstream.  Keeping the
/// public Rust window representation in `f64` does not make the QR boundary an
/// `f64` operation: Eigen first casts/constructs a float AOM, then
/// `makeHouseholderInPlace` computes the reflector and the rank cutoff in
/// `float`.  This helper makes that boundary explicit and returns values widened
/// back to the surrounding Rust API only after every QR operation has completed.
///
/// The ordering arguments represent upstream `std::set<int>` values and are
/// therefore sorted before the `[marg | keep]` permutation.  The reflector
/// storage and update order mirror Eigen's `makeHouseholderInPlace` plus
/// `applyHouseholderOnTheLeft`; in particular, the essential vector is stored as
/// `tail / (c0 - beta)`, rather than using a normalized `[v0 | tail]` vector.
pub(crate) fn sqrt_marginalize_upstream_f32(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    sqrt_marginalize_upstream_f32_packet(jacobian, rhs, keep_columns, marginal_columns, fej_point)
}

#[derive(Clone, Copy)]
struct UpstreamF32Packet4([f32; 4]);

#[derive(Clone, Copy)]
struct UpstreamF32Packet8([f32; 8]);

#[inline]
fn upstream_f32_bits_hash(values: &[f32]) -> u64 {
    let mut hash = 1_469_598_103_934_665_603_u64;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1_099_511_628_211_u64);
        }
    }
    hash
}

#[inline]
fn upstream_f32_write_qtrace_u32(output: &mut fs::File, value: u32) -> bool {
    output.write_all(&value.to_le_bytes()).is_ok()
}

#[inline]
fn upstream_f32_write_qtrace_f32_bits(output: &mut fs::File, value: f32) -> bool {
    upstream_f32_write_qtrace_u32(output, value.to_bits())
}

fn upstream_f32_write_qtrace_header(
    output: &mut fs::File,
    rows: usize,
    cols: usize,
    rhs_size: usize,
    marginal_size: usize,
    keep_size: usize,
) -> bool {
    output.write_all(b"M7IM15QTRACE1").is_ok()
        && upstream_f32_write_qtrace_u32(output, 1)
        && upstream_f32_write_qtrace_u32(output, rows as u32)
        && upstream_f32_write_qtrace_u32(output, cols as u32)
        && upstream_f32_write_qtrace_u32(output, rhs_size as u32)
        && upstream_f32_write_qtrace_u32(output, marginal_size as u32)
        && upstream_f32_write_qtrace_u32(output, keep_size as u32)
}

fn upstream_f32_write_qtrace_record(
    output: &mut fs::File,
    column: usize,
    rank_before: usize,
    rank_after: usize,
    c0: f32,
    tail_sq_norm: f32,
    beta: f32,
    tau: f32,
    qj: &[f32],
    qr: &[f32],
) -> bool {
    if !upstream_f32_write_qtrace_u32(output, column as u32)
        || !upstream_f32_write_qtrace_u32(output, rank_before as u32)
        || !upstream_f32_write_qtrace_u32(output, rank_after as u32)
        || !upstream_f32_write_qtrace_f32_bits(output, c0)
        || !upstream_f32_write_qtrace_f32_bits(output, tail_sq_norm)
        || !upstream_f32_write_qtrace_f32_bits(output, beta)
        || !upstream_f32_write_qtrace_f32_bits(output, tau)
    {
        return false;
    }
    for &value in qj.iter().chain(qr.iter()) {
        if !upstream_f32_write_qtrace_u32(output, value.to_bits()) {
            return false;
        }
    }
    true
}

#[inline]
fn upstream_f32_packet_load(data: &[f32], index: usize) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        data[index],
        data[index + 1],
        data[index + 2],
        data[index + 3],
    ])
}

#[inline]
fn upstream_f32_packet8_load(data: &[f32], index: usize) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        data[index],
        data[index + 1],
        data[index + 2],
        data[index + 3],
        data[index + 4],
        data[index + 5],
        data[index + 6],
        data[index + 7],
    ])
}

#[inline]
fn upstream_f32_packet_store(data: &mut [f32], index: usize, packet: UpstreamF32Packet4) {
    data[index] = packet.0[0];
    data[index + 1] = packet.0[1];
    data[index + 2] = packet.0[2];
    data[index + 3] = packet.0[3];
}

#[inline]
fn upstream_f32_packet8_store(data: &mut [f32], index: usize, packet: UpstreamF32Packet8) {
    data[index] = packet.0[0];
    data[index + 1] = packet.0[1];
    data[index + 2] = packet.0[2];
    data[index + 3] = packet.0[3];
    data[index + 4] = packet.0[4];
    data[index + 5] = packet.0[5];
    data[index + 6] = packet.0[6];
    data[index + 7] = packet.0[7];
}

#[inline]
fn upstream_f32_packet_add(lhs: UpstreamF32Packet4, rhs: UpstreamF32Packet4) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
    ])
}

#[inline]
fn upstream_f32_packet_mul(lhs: UpstreamF32Packet4, rhs: UpstreamF32Packet4) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0] * rhs.0[0],
        lhs.0[1] * rhs.0[1],
        lhs.0[2] * rhs.0[2],
        lhs.0[3] * rhs.0[3],
    ])
}

#[inline]
fn upstream_f32_packet8_mul(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        lhs.0[0] * rhs.0[0],
        lhs.0[1] * rhs.0[1],
        lhs.0[2] * rhs.0[2],
        lhs.0[3] * rhs.0[3],
        lhs.0[4] * rhs.0[4],
        lhs.0[5] * rhs.0[5],
        lhs.0[6] * rhs.0[6],
        lhs.0[7] * rhs.0[7],
    ])
}

#[inline]
fn upstream_f32_packet_mul_scalar(packet: UpstreamF32Packet4, scalar: f32) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        packet.0[0] * scalar,
        packet.0[1] * scalar,
        packet.0[2] * scalar,
        packet.0[3] * scalar,
    ])
}

#[inline]
fn upstream_f32_packet8_mul_add(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
    accumulator: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
        lhs.0[4].mul_add(rhs.0[4], accumulator.0[4]),
        lhs.0[5].mul_add(rhs.0[5], accumulator.0[5]),
        lhs.0[6].mul_add(rhs.0[6], accumulator.0[6]),
        lhs.0[7].mul_add(rhs.0[7], accumulator.0[7]),
    ])
}

#[inline]
fn upstream_f32_packet8_sub_scaled(data: &mut [f32], start: usize, source: &[f32], scalar: f32) {
    let len = source.len();
    let negative_scalar = -scalar;
    // Eigen's AVX assignment peels each destination vector until the
    // column-tail pointer reaches its 32-byte packet boundary.  The source
    // essential vector starts at the same logical tail offset, so consume the
    // identical scalar prefix before loading Packet8 values.
    let head = ((8 - (start & 7)) & 7).min(len);
    for offset in 0..head {
        // Eigen's scalar peel is emitted as a multiply followed by a
        // subtract; this prefix is retained as the existing scalar peel.
        data[start + offset] -= scalar * source[offset];
    }
    let packet_end = head + ((len - head) / 8) * 8;
    let mut offset = head;
    while offset < packet_end {
        let destination = upstream_f32_packet8_load(data, start + offset);
        let source_packet = upstream_f32_packet8_load(source, offset);
        let updated = UpstreamF32Packet8([
            negative_scalar.mul_add(source_packet.0[0], destination.0[0]),
            negative_scalar.mul_add(source_packet.0[1], destination.0[1]),
            negative_scalar.mul_add(source_packet.0[2], destination.0[2]),
            negative_scalar.mul_add(source_packet.0[3], destination.0[3]),
            negative_scalar.mul_add(source_packet.0[4], destination.0[4]),
            negative_scalar.mul_add(source_packet.0[5], destination.0[5]),
            negative_scalar.mul_add(source_packet.0[6], destination.0[6]),
            negative_scalar.mul_add(source_packet.0[7], destination.0[7]),
        ]);
        upstream_f32_packet8_store(data, start + offset, updated);
        offset += 8;
    }
    // The pinned Eigen AVX masked remainder materializes `tau * essential`
    // before the outer product, then emits a separate multiply by the
    // workspace scalar and a subtract from the destination.  Keep the
    // packet body above fused, but prevent LLVM from contracting this
    // scalar/masked remainder back into an FMA.
    while offset < len {
        let product = upstream_f32_scalar_product(scalar, source[offset]);
        data[start + offset] -= product;
        offset += 1;
    }
}

#[inline]
fn upstream_f32_packet8_sub_scaled_fma_head(
    data: &mut [f32],
    start: usize,
    source: &[f32],
    scalar: f32,
) {
    // The RHS vector assignment uses the same fused scalar multiply/subtract
    // as Eigen's masked packet path, including the alignment-peel head.  The
    // matrix outer update intentionally keeps `upstream_f32_packet8_sub_scaled`
    // because its scalar head is a separate multiply/subtract on this build.
    let len = source.len();
    let negative_scalar = -scalar;
    let head = ((8 - (start & 7)) & 7).min(len);
    for offset in 0..head {
        data[start + offset] = negative_scalar.mul_add(source[offset], data[start + offset]);
    }
    let packet_end = head + ((len - head) / 8) * 8;
    let mut offset = head;
    while offset < packet_end {
        let destination = upstream_f32_packet8_load(data, start + offset);
        let source_packet = upstream_f32_packet8_load(source, offset);
        let updated = UpstreamF32Packet8([
            negative_scalar.mul_add(source_packet.0[0], destination.0[0]),
            negative_scalar.mul_add(source_packet.0[1], destination.0[1]),
            negative_scalar.mul_add(source_packet.0[2], destination.0[2]),
            negative_scalar.mul_add(source_packet.0[3], destination.0[3]),
            negative_scalar.mul_add(source_packet.0[4], destination.0[4]),
            negative_scalar.mul_add(source_packet.0[5], destination.0[5]),
            negative_scalar.mul_add(source_packet.0[6], destination.0[6]),
            negative_scalar.mul_add(source_packet.0[7], destination.0[7]),
        ]);
        upstream_f32_packet8_store(data, start + offset, updated);
        offset += 8;
    }
    while offset < len {
        data[start + offset] = negative_scalar.mul_add(source[offset], data[start + offset]);
        offset += 1;
    }
}

#[inline]
fn upstream_f32_packet8_add(
    lhs: UpstreamF32Packet8,
    rhs: UpstreamF32Packet8,
) -> UpstreamF32Packet8 {
    UpstreamF32Packet8([
        lhs.0[0] + rhs.0[0],
        lhs.0[1] + rhs.0[1],
        lhs.0[2] + rhs.0[2],
        lhs.0[3] + rhs.0[3],
        lhs.0[4] + rhs.0[4],
        lhs.0[5] + rhs.0[5],
        lhs.0[6] + rhs.0[6],
        lhs.0[7] + rhs.0[7],
    ])
}

#[inline]
fn upstream_f32_packet_mul_add(
    lhs: UpstreamF32Packet4,
    rhs: UpstreamF32Packet4,
    accumulator: UpstreamF32Packet4,
) -> UpstreamF32Packet4 {
    UpstreamF32Packet4([
        lhs.0[0].mul_add(rhs.0[0], accumulator.0[0]),
        lhs.0[1].mul_add(rhs.0[1], accumulator.0[1]),
        lhs.0[2].mul_add(rhs.0[2], accumulator.0[2]),
        lhs.0[3].mul_add(rhs.0[3], accumulator.0[3]),
    ])
}

#[inline]
fn upstream_f32_packet_reduce(packet: UpstreamF32Packet4) -> f32 {
    let even = packet.0[0] + packet.0[2];
    let odd = packet.0[1] + packet.0[3];
    even + odd
}

#[inline]
fn upstream_f32_packet8_reduce(packet: UpstreamF32Packet8) -> f32 {
    // Eigen's AVX predux first adds the low and high 128-bit halves, then
    // applies the Packet4 reduction order to that result.
    let pair0 = packet.0[0] + packet.0[4];
    let pair1 = packet.0[1] + packet.0[5];
    let pair2 = packet.0[2] + packet.0[6];
    let pair3 = packet.0[3] + packet.0[7];
    let even = pair0 + pair2;
    let odd = pair1 + pair3;
    even + odd
}

fn upstream_f32_packet_squared_norm(data: &[f32], start: usize, len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }
    // Eigen's `squaredNorm()` uses the linear vectorized reduction for a
    // dynamic column segment.  Its first packet is the first 16-byte aligned
    // address, while scalar coefficients before that packet are folded after
    // the packet reduction.  Starting a packet at `start` changes both the
    // packet grouping and the scalar/packet addition order whenever the
    // Householder tail is not aligned.
    let end = start + len;
    let aligned_start = start + ((4 - (start & 3)) & 3).min(len);
    let aligned_size = ((end - aligned_start) / 4) * 4;
    let aligned_end = aligned_start + aligned_size;
    let mut result;

    if aligned_size != 0 {
        let mut packet_result = upstream_f32_packet_mul(
            upstream_f32_packet_load(data, aligned_start),
            upstream_f32_packet_load(data, aligned_start),
        );
        if aligned_size > 4 {
            let mut packet_result_1 = upstream_f32_packet_mul(
                upstream_f32_packet_load(data, aligned_start + 4),
                upstream_f32_packet_load(data, aligned_start + 4),
            );
            let aligned_end2 = aligned_start + (aligned_size / 8) * 8;
            let mut index = aligned_start + 8;
            while index < aligned_end2 {
                packet_result = upstream_f32_packet_add(
                    packet_result,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, index),
                        upstream_f32_packet_load(data, index),
                    ),
                );
                packet_result_1 = upstream_f32_packet_add(
                    packet_result_1,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, index + 4),
                        upstream_f32_packet_load(data, index + 4),
                    ),
                );
                index += 8;
            }
            packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
            if aligned_end > aligned_end2 {
                packet_result = upstream_f32_packet_add(
                    packet_result,
                    upstream_f32_packet_mul(
                        upstream_f32_packet_load(data, aligned_end2),
                        upstream_f32_packet_load(data, aligned_end2),
                    ),
                );
            }
        }
        result = upstream_f32_packet_reduce(packet_result);
        for index in start..aligned_start {
            result += data[index] * data[index];
        }
        for index in aligned_end..end {
            result += data[index] * data[index];
        }
    } else {
        result = data[start] * data[start];
        for index in (start + 1)..end {
            result += data[index] * data[index];
        }
    }
    result
}

/// Eigen's AVX `squaredNorm()` reduction for a dynamic vector segment.
///
/// This is the same `LinearVectorizedTraversal` schedule as the pinned
/// `redux_impl` for `unaryExpr(squared_norm_functor).sum()`: keep two Packet8
/// accumulators in flight, combine packet products with the contracted
/// multiply-add sequence emitted by the pinned O3 Eigen build, merge them,
/// then reduce with AVX's low/high-half order before adding scalar
/// head/tail coefficients.
#[inline]
fn upstream_f32_packet8_squared_norm(data: &[f32], start: usize, len: usize) -> f32 {
    if len == 0 {
        return 0.0;
    }
    let end = start + len;
    // The pinned Eigen cwiseAbs2 evaluator has no DirectAccessBit, so
    // `first_default_aligned` is the segment origin rather than the parent
    // allocation's physical address.  Keep the packet grouping relative to
    // this VectorBlock segment.
    let aligned_start = start;
    let aligned_size = ((end - aligned_start) / 8) * 8;
    let aligned_end = aligned_start + aligned_size;
    let mut result;
    if aligned_size != 0 {
        let mut packet_result = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(data, aligned_start),
            upstream_f32_packet8_load(data, aligned_start),
        );
        if aligned_size > 8 {
            let mut packet_result_1 = upstream_f32_packet8_mul(
                upstream_f32_packet8_load(data, aligned_start + 8),
                upstream_f32_packet8_load(data, aligned_start + 8),
            );
            let aligned_end2 = aligned_start + (aligned_size / 16) * 16;
            let mut index = aligned_start + 16;
            while index < aligned_end2 {
                let packet = upstream_f32_packet8_load(data, index);
                packet_result = upstream_f32_packet8_mul_add(packet, packet, packet_result);
                let packet_1 = upstream_f32_packet8_load(data, index + 8);
                packet_result_1 = upstream_f32_packet8_mul_add(packet_1, packet_1, packet_result_1);
                index += 16;
            }
            packet_result = upstream_f32_packet8_add(packet_result, packet_result_1);
            if aligned_end > aligned_end2 {
                let packet = upstream_f32_packet8_load(data, aligned_end2);
                packet_result = upstream_f32_packet8_mul_add(packet, packet, packet_result);
            }
        }
        result = upstream_f32_packet8_reduce(packet_result);
        for index in start..aligned_start {
            result = data[index].mul_add(data[index], result);
        }
        // GCC's pinned Eigen build vectorizes a four-coefficient remainder as
        // `vmulps` followed by scalar `vaddss` instructions.  Any remainder
        // after that packet stays on the scalar contracted FMA path.
        let scalar_packet_end = (aligned_end + ((end - aligned_end) / 4) * 4).min(end);
        for index in aligned_end..scalar_packet_end {
            result += data[index] * data[index];
        }
        for index in scalar_packet_end..end {
            result = data[index].mul_add(data[index], result);
        }
    } else {
        result = data[start] * data[start];
        let scalar_packet_end = (start + 1 + ((len - 1) / 4) * 4).min(end);
        for index in (start + 1)..scalar_packet_end {
            result += data[index] * data[index];
        }
        for index in scalar_packet_end..end {
            result = data[index].mul_add(data[index], result);
        }
    }
    result
}

#[inline]
fn upstream_f32_norm_scalar_variant(data: &[f32], start: usize, len: usize) -> f32 {
    let mut result = data[start] * data[start];
    for index in (start + 1)..(start + len) {
        result += data[index] * data[index];
    }
    result
}

#[inline]
fn upstream_f32_norm_packet4_variant(
    data: &[f32],
    start: usize,
    len: usize,
    head: usize,
    two_accumulators: bool,
) -> f32 {
    let head = head.min(len);
    let aligned_start = start + head;
    let aligned_size = ((len - head) / 4) * 4;
    let aligned_end = aligned_start + aligned_size;
    if aligned_size == 0 {
        return upstream_f32_norm_scalar_variant(data, start, len);
    }
    let mut packet_result = upstream_f32_packet_mul(
        upstream_f32_packet_load(data, aligned_start),
        upstream_f32_packet_load(data, aligned_start),
    );
    if two_accumulators && aligned_size > 4 {
        let mut packet_result_1 = upstream_f32_packet_mul(
            upstream_f32_packet_load(data, aligned_start + 4),
            upstream_f32_packet_load(data, aligned_start + 4),
        );
        let aligned_end2 = aligned_start + (aligned_size / 8) * 8;
        let mut index = aligned_start + 8;
        while index < aligned_end2 {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index),
                    upstream_f32_packet_load(data, index),
                ),
            );
            packet_result_1 = upstream_f32_packet_add(
                packet_result_1,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index + 4),
                    upstream_f32_packet_load(data, index + 4),
                ),
            );
            index += 8;
        }
        packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
        if aligned_end > aligned_end2 {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, aligned_end2),
                    upstream_f32_packet_load(data, aligned_end2),
                ),
            );
        }
    } else {
        let mut index = aligned_start + 4;
        while index < aligned_end {
            packet_result = upstream_f32_packet_add(
                packet_result,
                upstream_f32_packet_mul(
                    upstream_f32_packet_load(data, index),
                    upstream_f32_packet_load(data, index),
                ),
            );
            index += 4;
        }
    }
    let mut result = upstream_f32_packet_reduce(packet_result);
    for index in start..aligned_start {
        result = data[index].mul_add(data[index], result);
    }
    for index in aligned_end..(start + len) {
        result = data[index].mul_add(data[index], result);
    }
    result
}

#[inline]
fn upstream_f32_norm_packet8_variant(
    data: &[f32],
    start: usize,
    len: usize,
    head: usize,
    two_accumulators: bool,
) -> f32 {
    let head = head.min(len);
    let aligned_start = start + head;
    let aligned_size = ((len - head) / 8) * 8;
    let aligned_end = aligned_start + aligned_size;
    if aligned_size == 0 {
        return upstream_f32_norm_scalar_variant(data, start, len);
    }
    let mut packet_result = upstream_f32_packet8_mul(
        upstream_f32_packet8_load(data, aligned_start),
        upstream_f32_packet8_load(data, aligned_start),
    );
    if two_accumulators && aligned_size > 8 {
        let mut packet_result_1 = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(data, aligned_start + 8),
            upstream_f32_packet8_load(data, aligned_start + 8),
        );
        let aligned_end2 = aligned_start + (aligned_size / 16) * 16;
        let mut index = aligned_start + 16;
        while index < aligned_end2 {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index),
                    upstream_f32_packet8_load(data, index),
                ),
            );
            packet_result_1 = upstream_f32_packet8_add(
                packet_result_1,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index + 8),
                    upstream_f32_packet8_load(data, index + 8),
                ),
            );
            index += 16;
        }
        packet_result = upstream_f32_packet8_add(packet_result, packet_result_1);
        if aligned_end > aligned_end2 {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, aligned_end2),
                    upstream_f32_packet8_load(data, aligned_end2),
                ),
            );
        }
    } else {
        let mut index = aligned_start + 8;
        while index < aligned_end {
            packet_result = upstream_f32_packet8_add(
                packet_result,
                upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(data, index),
                    upstream_f32_packet8_load(data, index),
                ),
            );
            index += 8;
        }
    }
    let mut result = upstream_f32_packet8_reduce(packet_result);
    for index in start..aligned_start {
        result += data[index] * data[index];
    }
    for index in aligned_end..(start + len) {
        result += data[index] * data[index];
    }
    result
}

fn upstream_f32_packet_dot(
    lhs: &[f32],
    lhs_start: usize,
    rhs: &[f32],
    rhs_start: usize,
    len: usize,
) -> f32 {
    if len == 0 {
        return 0.0;
    }
    let end = lhs_start + len;
    let packet_end = lhs_start + (len / 4) * 4;
    if packet_end == lhs_start {
        let mut result = lhs[lhs_start] * rhs[rhs_start];
        for offset in 1..len {
            result += lhs[lhs_start + offset] * rhs[rhs_start + offset];
        }
        return result;
    }

    let packet_count = len / 4;
    let quad_end = lhs_start + (len / 16) * 16;
    let remainder_packets = (packet_end - quad_end) / 4;
    let mut packet_result = upstream_f32_packet_mul(
        upstream_f32_packet_load(lhs, lhs_start),
        upstream_f32_packet_load(rhs, rhs_start),
    );
    if packet_count >= 2 {
        let mut packet_result_1 = upstream_f32_packet_mul(
            upstream_f32_packet_load(lhs, lhs_start + 4),
            upstream_f32_packet_load(rhs, rhs_start + 4),
        );
        if packet_count >= 3 {
            let mut packet_result_2 = upstream_f32_packet_mul(
                upstream_f32_packet_load(lhs, lhs_start + 8),
                upstream_f32_packet_load(rhs, rhs_start + 8),
            );
            if packet_count >= 4 {
                let mut packet_result_3 = upstream_f32_packet_mul(
                    upstream_f32_packet_load(lhs, lhs_start + 12),
                    upstream_f32_packet_load(rhs, rhs_start + 12),
                );
                let mut index = lhs_start + 16;
                while index < quad_end {
                    packet_result = upstream_f32_packet_add(
                        packet_result,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index),
                            upstream_f32_packet_load(rhs, rhs_start + index - lhs_start),
                        ),
                    );
                    packet_result_1 = upstream_f32_packet_add(
                        packet_result_1,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 4),
                            upstream_f32_packet_load(rhs, rhs_start + index + 4 - lhs_start),
                        ),
                    );
                    packet_result_2 = upstream_f32_packet_add(
                        packet_result_2,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 8),
                            upstream_f32_packet_load(rhs, rhs_start + index + 8 - lhs_start),
                        ),
                    );
                    packet_result_3 = upstream_f32_packet_add(
                        packet_result_3,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, index + 12),
                            upstream_f32_packet_load(rhs, rhs_start + index + 12 - lhs_start),
                        ),
                    );
                    index += 16;
                }
                if remainder_packets >= 1 {
                    packet_result = upstream_f32_packet_add(
                        packet_result,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end - lhs_start),
                        ),
                    );
                }
                if remainder_packets >= 2 {
                    packet_result_1 = upstream_f32_packet_add(
                        packet_result_1,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end + 4),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end + 4 - lhs_start),
                        ),
                    );
                }
                if remainder_packets == 3 {
                    packet_result_2 = upstream_f32_packet_add(
                        packet_result_2,
                        upstream_f32_packet_mul(
                            upstream_f32_packet_load(lhs, quad_end + 8),
                            upstream_f32_packet_load(rhs, rhs_start + quad_end + 8 - lhs_start),
                        ),
                    );
                }
                packet_result_2 = upstream_f32_packet_add(packet_result_2, packet_result_3);
            }
            packet_result_1 = upstream_f32_packet_add(packet_result_1, packet_result_2);
        }
        packet_result = upstream_f32_packet_add(packet_result, packet_result_1);
    }

    let mut result = upstream_f32_packet_reduce(packet_result);
    for index in packet_end..end {
        let offset = index - lhs_start;
        result += lhs[index] * rhs[rhs_start + offset];
    }
    result
}

// Eigen's vector-on-the-left matrix product is evaluated through the
// row-major GEMV kernel.  Unlike the vector inner-product path above, GEMV
// keeps one packet accumulator and advances it in four-element blocks.
#[inline]
fn upstream_f32_packet_gemv_dot(lhs: &[f32], rhs: &[f32], rhs_start: usize, len: usize) -> f32 {
    if len < 4 {
        if len == 0 {
            return 0.0;
        }
        let mut result = lhs[0] * rhs[rhs_start];
        for offset in 1..len {
            result += lhs[offset] * rhs[rhs_start + offset];
        }
        return result;
    }
    let packet_end = (len / 4) * 4;
    let mut packet_result = UpstreamF32Packet4([0.0; 4]);
    let mut offset = 0;
    while offset < packet_end {
        packet_result = upstream_f32_packet_mul_add(
            upstream_f32_packet_load(lhs, offset),
            upstream_f32_packet_load(rhs, rhs_start + offset),
            packet_result,
        );
        offset += 4;
    }
    let mut result = upstream_f32_packet_reduce(packet_result);
    for offset in packet_end..len {
        result += lhs[offset] * rhs[rhs_start + offset];
    }
    result
}

/// Emulate Eigen's AVX row-major GEMV used by
/// `essential.adjoint() * bottom`.  Eigen treats the transposed expression as
/// a row-major matrix whose rows are trailing qj columns and whose columns
/// are the Householder essential vector.  It therefore accumulates eight
/// output rows at once, broadcasting each essential packet across the rows;
/// packetizing the essential dimension instead changes the reduction order.
#[inline(never)]
fn upstream_f32_gemv_scalar_product(lhs: f32, rhs: f32) -> f32 {
    // The pinned Eigen/GCC row-major GEMV emits the scalar remainder as a
    // separate vmulss/vaddss sequence.  Keep the product out of the caller so
    // LLVM cannot contract the source-level `sum += lhs * rhs` back into FMA.
    lhs * rhs
}

#[inline(never)]
fn upstream_f32_scalar_product(lhs: f32, rhs: f32) -> f32 {
    // Eigen's masked packet remainder materializes the product before the
    // subtract (vmulps followed by vsubps), rather than using a scalar FMA.
    // Keeping this operation out of the caller prevents LLVM contraction.
    lhs * rhs
}

#[inline(never)]
fn upstream_f32_gemv_packet4_accumulate(
    mut result: f32,
    lhs: &[f32],
    rhs: &[f32],
    lhs_start: usize,
    rhs_start: usize,
) -> f32 {
    // GCC's AVX row-major GEMV lowers a four-element scalar remainder to one
    // vmulps followed by four ordered vaddss operations into the existing
    // scalar accumulator.  Keep this packet's products separate from the
    // accumulator so it cannot become four FMAs or a tree reduction.
    let product = [
        lhs[lhs_start] * rhs[rhs_start],
        lhs[lhs_start + 1] * rhs[rhs_start + 1],
        lhs[lhs_start + 2] * rhs[rhs_start + 2],
        lhs[lhs_start + 3] * rhs[rhs_start + 3],
    ];
    result += product[0];
    result += product[1];
    result += product[2];
    result += product[3];
    result
}

#[inline]
fn upstream_f32_packet8_gemv_workspace(
    essential: &[f32],
    qj: &[f32],
    rows: usize,
    tail_row_start: usize,
    first_trailing_column: usize,
    trailing_columns: usize,
) -> Vec<f32> {
    let mut workspace = vec![0.0_f32; trailing_columns];
    let len = essential.len();
    let full_col_end = len / 8 * 8;
    let half_col_end = full_col_end + (len - full_col_end) / 4 * 4;

    // The row-major Eigen kernel processes 8, then 4, then 2 output rows.
    // Each packet accumulator is reduced only after the complete essential
    // dimension, matching `predux(Packet8f)` and its low/high-half order.
    let mut row = 0usize;
    while row + 8 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 8];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..8 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..8 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 8;
    }

    while row + 4 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 4];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..4 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..4 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 4;
    }

    while row + 2 <= trailing_columns {
        let mut accumulators = [UpstreamF32Packet8([0.0; 8]); 2];
        let mut offset = 0usize;
        while offset < full_col_end {
            let rhs = upstream_f32_packet8_load(essential, offset);
            for lane in 0..2 {
                let column = first_trailing_column + row + lane;
                let lhs = upstream_f32_packet8_load(qj, column * rows + tail_row_start + offset);
                accumulators[lane] = upstream_f32_packet8_mul_add(lhs, rhs, accumulators[lane]);
            }
            offset += 8;
        }
        for lane in 0..2 {
            let column = first_trailing_column + row + lane;
            let base = column * rows + tail_row_start;
            let mut result = upstream_f32_packet8_reduce(accumulators[lane]);
            if len - full_col_end >= 4 {
                result = upstream_f32_gemv_packet4_accumulate(
                    result,
                    qj,
                    essential,
                    base + full_col_end,
                    full_col_end,
                );
            }
            for offset in (full_col_end + if len - full_col_end >= 4 { 4 } else { 0 })..len {
                result = qj[base + offset].mul_add(essential[offset], result);
            }
            workspace[row + lane] = result;
        }
        row += 2;
    }

    // The final row uses Eigen's full, half, and quarter packet accumulators
    // before the scalar tail.  Packet4's reduction is the same low/high then
    // even/odd order used by `upstream_f32_packet_reduce`.
    while row < trailing_columns {
        let column = first_trailing_column + row;
        let base = column * rows + tail_row_start;
        let mut full = UpstreamF32Packet8([0.0; 8]);
        let mut offset = 0usize;
        while offset < full_col_end {
            full = upstream_f32_packet8_mul_add(
                upstream_f32_packet8_load(qj, base + offset),
                upstream_f32_packet8_load(essential, offset),
                full,
            );
            offset += 8;
        }
        let mut result = upstream_f32_packet8_reduce(full);
        if half_col_end > full_col_end {
            let mut half = UpstreamF32Packet4([0.0; 4]);
            let mut offset = full_col_end;
            while offset < half_col_end {
                half = upstream_f32_packet_mul_add(
                    upstream_f32_packet_load(qj, base + offset),
                    upstream_f32_packet_load(essential, offset),
                    half,
                );
                offset += 4;
            }
            result += upstream_f32_packet_reduce(half);
        }
        for offset in half_col_end..len {
            result = qj[base + offset].mul_add(essential[offset], result);
        }
        workspace[row] = result;
        row += 1;
    }
    workspace
}

/// Eigen's runtime-vector inner product for the one-column RHS update.  The
/// AVX evaluator seeds four Packet8 accumulators, folds complete packets with
/// FMA, merges them left-associatively, and reduces the packet before the
/// scalar FMA tail.  This is distinct from the multi-column GEMV workspace.
#[inline]
fn upstream_f32_packet8_rhs_dot(lhs: &[f32], rhs: &[f32], rhs_start: usize, len: usize) -> f32 {
    // `essential.adjoint() * bottom` is Eigen's dynamic inner-product
    // evaluator, not the one-accumulator GEMV used for matrix workspaces.
    // Its AVX path seeds four Packet8 accumulators, folds complete groups of
    // four packets with packet FMA, merges p2<-p3, p1<-p2, p0<-p1, then
    // preduces.  The scalar remainder is the evaluator's scalar FMA tail.
    if len < 8 {
        if len == 0 {
            return 0.0;
        }
        let mut result = lhs[0] * rhs[rhs_start];
        for offset in 1..len {
            result = lhs[offset].mul_add(rhs[rhs_start + offset], result);
        }
        return result;
    }
    let packet_end = len / 8 * 8;
    let quad_end = len / 32 * 32;
    let packet_count = len / 8;
    let remainder_packets = (packet_end - quad_end) / 8;
    let mut packet0 = upstream_f32_packet8_mul(
        upstream_f32_packet8_load(lhs, 0),
        upstream_f32_packet8_load(rhs, rhs_start),
    );
    if packet_count >= 2 {
        let mut packet1 = upstream_f32_packet8_mul(
            upstream_f32_packet8_load(lhs, 8),
            upstream_f32_packet8_load(rhs, rhs_start + 8),
        );
        if packet_count >= 3 {
            let mut packet2 = upstream_f32_packet8_mul(
                upstream_f32_packet8_load(lhs, 16),
                upstream_f32_packet8_load(rhs, rhs_start + 16),
            );
            if packet_count >= 4 {
                let mut packet3 = upstream_f32_packet8_mul(
                    upstream_f32_packet8_load(lhs, 24),
                    upstream_f32_packet8_load(rhs, rhs_start + 24),
                );
                let mut offset = 32usize;
                while offset < quad_end {
                    packet0 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset),
                        upstream_f32_packet8_load(rhs, rhs_start + offset),
                        packet0,
                    );
                    packet1 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 8),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 8),
                        packet1,
                    );
                    packet2 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 16),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 16),
                        packet2,
                    );
                    packet3 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, offset + 24),
                        upstream_f32_packet8_load(rhs, rhs_start + offset + 24),
                        packet3,
                    );
                    offset += 32;
                }
                if remainder_packets >= 1 {
                    packet0 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end),
                        packet0,
                    );
                }
                if remainder_packets >= 2 {
                    packet1 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end + 8),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end + 8),
                        packet1,
                    );
                }
                if remainder_packets == 3 {
                    packet2 = upstream_f32_packet8_mul_add(
                        upstream_f32_packet8_load(lhs, quad_end + 16),
                        upstream_f32_packet8_load(rhs, rhs_start + quad_end + 16),
                        packet2,
                    );
                }
                packet2 = upstream_f32_packet8_add(packet2, packet3);
            }
            packet1 = upstream_f32_packet8_add(packet1, packet2);
        }
        packet0 = upstream_f32_packet8_add(packet0, packet1);
    }
    let mut result = upstream_f32_packet8_reduce(packet0);
    for offset in packet_end..len {
        result = lhs[offset].mul_add(rhs[rhs_start + offset], result);
    }
    result
}

fn upstream_f32_packet_scale(data: &[f32], scalar: f32) -> Vec<f32> {
    let mut scaled = vec![0.0_f32; data.len()];
    let packet_end = (data.len() / 4) * 4;
    let mut index = 0;
    while index < packet_end {
        let packet = upstream_f32_packet_mul_scalar(upstream_f32_packet_load(data, index), scalar);
        upstream_f32_packet_store(&mut scaled, index, packet);
        index += 4;
    }
    for index in packet_end..data.len() {
        scaled[index] = data[index] * scalar;
    }
    scaled
}

fn upstream_f32_packet_divide_assign(data: &mut [f32], start: usize, len: usize, denominator: f32) {
    let head = ((4 - (start & 3)) & 3).min(len);
    for offset in 0..head {
        data[start + offset] /= denominator;
    }
    let packet_len = ((len - head) / 4) * 4;
    let mut offset = head;
    while offset < head + packet_len {
        let packet = upstream_f32_packet_load(data, start + offset);
        let divided = UpstreamF32Packet4([
            packet.0[0] / denominator,
            packet.0[1] / denominator,
            packet.0[2] / denominator,
            packet.0[3] / denominator,
        ]);
        upstream_f32_packet_store(data, start + offset, divided);
        offset += 4;
    }
    for offset in (head + packet_len)..len {
        data[start + offset] /= denominator;
    }
}

fn upstream_f32_packet_sub_scaled(data: &mut [f32], start: usize, source: &[f32], scalar: f32) {
    let len = source.len();
    let head = ((4 - (start & 3)) & 3).min(len);
    for offset in 0..head {
        data[start + offset] -= source[offset] * scalar;
    }
    let packet_len = ((len - head) / 4) * 4;
    let mut offset = head;
    while offset < head + packet_len {
        let destination = upstream_f32_packet_load(data, start + offset);
        let source_packet = upstream_f32_packet_load(source, offset);
        let scaled = upstream_f32_packet_mul_scalar(source_packet, scalar);
        let updated = UpstreamF32Packet4([
            destination.0[0] - scaled.0[0],
            destination.0[1] - scaled.0[1],
            destination.0[2] - scaled.0[2],
            destination.0[3] - scaled.0[3],
        ]);
        upstream_f32_packet_store(data, start + offset, updated);
        offset += 4;
    }
    for offset in (head + packet_len)..len {
        data[start + offset] -= source[offset] * scalar;
    }
}

#[inline]
fn upstream_f32_householder_beta(c0: f32, tail_sq_norm: f32) -> f32 {
    // Eigen's float makeHouseholderInPlace accumulates the pivot square into
    // the tail norm with a fused multiply-add.  Keep this as one explicit
    // operation at the scalar boundary; the sign convention is applied by
    // the caller after taking the positive square root.
    c0.mul_add(c0, tail_sq_norm).sqrt()
}

/// Safe f32 emulation of Eigen 5.0.1's packetized SqrtToSqrt path.
///
/// Eigen's default x86 build uses four-lane `Packet4f` operations.  The
/// explicit packet helpers above preserve the packet reduction and GEMV
/// operation order without relying on target-specific intrinsics or `unsafe`.
fn sqrt_marginalize_upstream_f32_packet(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Option<super::aom::SqrtPrior> {
    let rows = jacobian.nrows();
    let cols = jacobian.ncols();
    if rows == 0
        || rhs.len() != rows
        || keep_columns.len() + marginal_columns.len() != cols
        || fej_point.len() != keep_columns.len()
    {
        return None;
    }

    // The upstream call receives std::set<int>.  Sort the Rust slices at this
    // boundary so equivalent callers cannot change the Eigen permutation by
    // passing a different input order.
    let mut keep = keep_columns.to_vec();
    let mut marg = marginal_columns.to_vec();
    keep.sort_unstable();
    marg.sort_unstable();
    let mut seen = vec![false; cols];
    for &column in keep.iter().chain(marg.iter()) {
        if column >= cols || seen[column] {
            return None;
        }
        seen[column] = true;
    }
    if seen.iter().any(|present| !present) {
        return None;
    }

    // Cast the complete stack before doing arithmetic.  This is the scalar
    // boundary of SqrtKeypointVioEstimator<float>; casting only final values
    // would retain the wrong reflector and rank decisions.
    let mut qj = vec![0.0_f32; rows * cols];
    for column in 0..cols {
        for row in 0..rows {
            qj[column * rows + row] = jacobian[(row, column)] as f32;
        }
    }
    let mut permutation = marg;
    permutation.extend_from_slice(&keep);
    let mut permuted = vec![0.0_f32; rows * cols];
    for (new_column, &old_column) in permutation.iter().enumerate() {
        for row in 0..rows {
            permuted[new_column * rows + row] = qj[old_column * rows + row];
        }
    }
    qj = permuted;
    let mut qr = rhs.iter().map(|value| *value as f32).collect::<Vec<_>>();

    let marg_size = marginal_columns.len();
    let rank_threshold = f32::EPSILON.sqrt();
    let mut total_rank = 0usize;
    let mut marginal_rank = 0usize;
    let diagnostic_policy = diagnostic_env_snapshot();
    let trace_reflectors = diagnostic_policy.sqrt_trace;
    let trace_focus_k12 = diagnostic_policy.sqrt_focus_k12;
    let trace_workspace_all = diagnostic_policy.sqrt_dump_workspace_all;
    let trace_tail_dump = diagnostic_policy.sqrt_dump_tail;
    let mut trace_qj_dump = diagnostic_policy.sqrt_dump_qj.as_ref().and_then(|path| {
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .ok()
    });
    if let Some(output) = trace_qj_dump.as_mut() {
        if !upstream_f32_write_qtrace_header(output, rows, cols, qr.len(), marg_size, keep.len()) {
            trace_qj_dump = None;
        }
    }
    if trace_reflectors {
        eprintln!("trace_base {}", qj.as_ptr() as usize & 0x0f);
    }

    for column in 0..cols {
        if total_rank >= rows {
            break;
        }
        let row_start = total_rank;
        let remaining_rows = rows - row_start;
        let remaining_columns = cols - column - 1;
        let tail_len = remaining_rows.saturating_sub(1);
        let column_start = column * rows;
        let tail_start = column_start + row_start + 1;
        let c0 = qj[column_start + row_start];
        let tail_sq_norm = upstream_f32_packet8_squared_norm(&qj, tail_start, tail_len);
        if trace_tail_dump && column == 12 {
            eprint!("trace_tail_bits {} {}", column, tail_len);
            for value in &qj[tail_start..tail_start + tail_len] {
                eprint!(" {:08x}", value.to_bits());
            }
            eprintln!();
        }
        let (tau, beta) = if tail_sq_norm <= f32::MIN_POSITIVE {
            (0.0_f32, c0)
        } else {
            let mut beta = upstream_f32_householder_beta(c0, tail_sq_norm);
            if c0 >= 0.0 {
                beta = -beta;
            }
            let denominator = c0 - beta;
            upstream_f32_packet_divide_assign(&mut qj, tail_start, tail_len, denominator);
            let tau = (beta - c0) / beta;
            (tau, beta)
        };

        if trace_reflectors {
            eprintln!(
                "trace_norm_meta {} {} {} {}",
                column,
                tail_len,
                (qj.as_ptr() as usize + tail_start * std::mem::size_of::<f32>()) & 31,
                ((32 - ((qj.as_ptr() as usize + tail_start * std::mem::size_of::<f32>()) & 31))
                    & 31)
                    / std::mem::size_of::<f32>(),
            );
            eprintln!(
                "trace_reflector {} {} {:08x} {:08x} {:08x} {:08x} {:08x} {:08x} {}",
                column,
                total_rank,
                c0.to_bits(),
                tail_sq_norm.to_bits(),
                beta.to_bits(),
                tau.to_bits(),
                (beta.abs() > rank_threshold) as u8,
                rank_threshold.to_bits(),
                tail_len,
            );
        }

        if beta.abs() > rank_threshold {
            qj[column_start + row_start] = beta;
            if tau != 0.0 {
                if remaining_rows == 1 {
                    let scale = 1.0_f32 - tau;
                    for trailing in (column + 1)..(column + 1 + remaining_columns) {
                        qj[trailing * rows + row_start] *= scale;
                    }
                    qr[row_start] *= scale;
                } else {
                    let essential = qj[tail_start..tail_start + tail_len].to_vec();
                    if trace_focus_k12 && column == 15 {
                        eprintln!(
                            "focus_begin {} {} {} {} {:08x} {:08x}",
                            column,
                            row_start,
                            remaining_rows,
                            remaining_columns,
                            tau.to_bits(),
                            beta.to_bits(),
                        );
                        eprint!("focus_essential");
                        for value in &essential {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!("focus_bottom {} {}", trailing_offset, trailing);
                            let base = trailing * rows + row_start + 1;
                            for offset in 0..tail_len {
                                eprint!(" {:08x}", qj[base + offset].to_bits());
                            }
                            eprintln!();
                        }
                    }
                    // ProductEvaluators materializes tau*essential before
                    // applying each scalar workspace to the outer product.
                    let scaled_essential = upstream_f32_packet_scale(&essential, tau);
                    // The pinned Eigen applyHouseholderOnTheLeft dispatches
                    // an exactly one-column block to its dynamic inner product
                    // (3d3cf3 -> 3d43e0), with four Packet8 accumulators. This
                    // is not the final output row of the multi-column GEMV.
                    let workspace = if remaining_columns == 1 {
                        vec![upstream_f32_packet8_rhs_dot(
                            &essential,
                            &qj,
                            (column + 1) * rows + row_start + 1,
                            tail_len,
                        )]
                    } else {
                        upstream_f32_packet8_gemv_workspace(
                            &essential,
                            &qj,
                            rows,
                            row_start + 1,
                            column + 1,
                            remaining_columns,
                        )
                    };
                    if trace_workspace_all {
                        eprint!(
                            "trace_workspace_all {} {} {} {}",
                            column, row_start, remaining_rows, remaining_columns,
                        );
                        for value in &workspace {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                    }
                    if trace_focus_k12 && column == 15 {
                        eprint!("focus_workspace_raw");
                        for value in &workspace {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprint!("focus_row0");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!(" {:08x}", qj[trailing * rows + row_start].to_bits());
                        }
                        eprintln!();
                        eprint!("focus_workspace_plus_row0");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            let value =
                                workspace[trailing_offset] + qj[trailing * rows + row_start];
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprint!("focus_scaled_workspace");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            let value =
                                workspace[trailing_offset] + qj[trailing * rows + row_start];
                            eprint!(" {:08x}", (tau * value).to_bits());
                        }
                        eprintln!();
                    }
                    if trace_reflectors {
                        eprint!("trace_workspace {}", column);
                        for value in workspace.iter().take(8) {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        if column == 9 {
                            eprint!("trace_workspace_full {}", column);
                            for value in &workspace {
                                eprint!(" {:08x}", value.to_bits());
                            }
                            eprintln!();
                        }
                    }
                    if trace_reflectors && column == 9 {
                        let mut workspace_after_add = workspace.clone();
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            workspace_after_add[trailing_offset] += qj[trailing * rows + row_start];
                        }
                        eprintln!(
                            "trace_stage9_workspace_add {:016x}",
                            upstream_f32_bits_hash(&workspace_after_add),
                        );
                        eprint!("trace_stage9_workspace_add_full");
                        for value in &workspace_after_add {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        let diagnostic_lane = 14usize;
                        let trailing = column + 1 + diagnostic_lane;
                        let base = trailing * rows;
                        let workspace_value = workspace_after_add[diagnostic_lane];
                        eprint!("trace_stage9_col14_pre");
                        for value in &qj[base..base + rows] {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                        eprintln!(
                            "trace_stage9_operands {:08x} {:08x} {:08x} {:08x} {:08x}",
                            workspace_value.to_bits(),
                            essential[3].to_bits(),
                            essential[4].to_bits(),
                            scaled_essential[3].to_bits(),
                            scaled_essential[4].to_bits(),
                        );
                        eprint!("trace_stage9_outer_headhash");
                        for requested_head in 0..8 {
                            let mut candidate = qj[base..base + rows].to_vec();
                            candidate[row_start] =
                                (-tau).mul_add(workspace_value, candidate[row_start]);
                            let head = requested_head.min(tail_len);
                            let packet_end = head + ((tail_len - head) / 8) * 8;
                            for offset in 0..head {
                                candidate[row_start + 1 + offset] = (-workspace_value).mul_add(
                                    scaled_essential[offset],
                                    candidate[row_start + 1 + offset],
                                );
                            }
                            let mut offset = head;
                            while offset < packet_end {
                                let destination =
                                    upstream_f32_packet8_load(&candidate, row_start + 1 + offset);
                                let source = upstream_f32_packet8_load(&scaled_essential, offset);
                                let updated = UpstreamF32Packet8([
                                    (-workspace_value).mul_add(source.0[0], destination.0[0]),
                                    (-workspace_value).mul_add(source.0[1], destination.0[1]),
                                    (-workspace_value).mul_add(source.0[2], destination.0[2]),
                                    (-workspace_value).mul_add(source.0[3], destination.0[3]),
                                    (-workspace_value).mul_add(source.0[4], destination.0[4]),
                                    (-workspace_value).mul_add(source.0[5], destination.0[5]),
                                    (-workspace_value).mul_add(source.0[6], destination.0[6]),
                                    (-workspace_value).mul_add(source.0[7], destination.0[7]),
                                ]);
                                upstream_f32_packet8_store(
                                    &mut candidate,
                                    row_start + 1 + offset,
                                    updated,
                                );
                                offset += 8;
                            }
                            while offset < tail_len {
                                candidate[row_start + 1 + offset] = (-workspace_value).mul_add(
                                    scaled_essential[offset],
                                    candidate[row_start + 1 + offset],
                                );
                                offset += 1;
                            }
                            eprint!(" {:016x}", upstream_f32_bits_hash(&candidate));
                        }
                        eprintln!();
                        let mut direct = qj[base..base + rows].to_vec();
                        direct[row_start] = (-tau).mul_add(workspace_value, direct[row_start]);
                        let direct_scale = -tau * workspace_value;
                        for offset in 0..tail_len {
                            direct[row_start + 1 + offset] = direct_scale
                                .mul_add(essential[offset], direct[row_start + 1 + offset]);
                        }
                        eprintln!(
                            "trace_stage9_outer_direct {:016x}",
                            upstream_f32_bits_hash(&direct)
                        );
                    }
                    for trailing_offset in 0..remaining_columns {
                        let trailing = column + 1 + trailing_offset;
                        let trailing_tail_start = trailing * rows + row_start + 1;
                        let mut workspace_value = workspace[trailing_offset];
                        workspace_value += qj[trailing * rows + row_start];
                        qj[trailing * rows + row_start] =
                            (-tau).mul_add(workspace_value, qj[trailing * rows + row_start]);
                        upstream_f32_packet8_sub_scaled(
                            &mut qj,
                            trailing_tail_start,
                            &scaled_essential,
                            workspace_value,
                        );
                    }
                    if trace_focus_k12 && column == 15 {
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!("focus_post {} {}", trailing_offset, trailing);
                            let base = trailing * rows + row_start;
                            for offset in 0..remaining_rows {
                                eprint!(" {:08x}", qj[base + offset].to_bits());
                            }
                            eprintln!();
                        }
                    }
                    if trace_reflectors && column == 9 {
                        eprintln!(
                            "trace_stage9_row0 {:016x}",
                            upstream_f32_bits_hash(
                                &(0..remaining_columns)
                                    .map(|offset| { qj[(column + 1 + offset) * rows + row_start] })
                                    .collect::<Vec<_>>(),
                            ),
                        );
                        eprint!("trace_stage9_colhash");
                        for trailing_offset in 0..remaining_columns {
                            let trailing = column + 1 + trailing_offset;
                            eprint!(
                                " {:016x}",
                                upstream_f32_bits_hash(&qj[trailing * rows..(trailing + 1) * rows])
                            );
                        }
                        eprintln!();
                        let trailing = column + 1 + 14;
                        eprint!("trace_stage9_col14_post");
                        for value in &qj[trailing * rows..(trailing + 1) * rows] {
                            eprint!(" {:08x}", value.to_bits());
                        }
                        eprintln!();
                    }
                    let mut workspace =
                        upstream_f32_packet8_rhs_dot(&essential, &qr, row_start + 1, tail_len);
                    workspace += qr[row_start];
                    qr[row_start] = (-tau).mul_add(workspace, qr[row_start]);
                    let scaled_essential = upstream_f32_packet_scale(&essential, tau);
                    upstream_f32_packet8_sub_scaled_fma_head(
                        &mut qr,
                        row_start + 1,
                        &scaled_essential,
                        workspace,
                    );
                    if trace_reflectors {
                        eprintln!(
                            "trace_rhs {} {:016x} {:08x} {:08x}",
                            column,
                            upstream_f32_bits_hash(&qr),
                            qr[row_start].to_bits(),
                            qr[std::cmp::min(row_start + 1, qr.len() - 1)].to_bits(),
                        );
                    }
                }
            }
            total_rank += 1;
        } else {
            for row in row_start..rows {
                qj[column_start + row] = 0.0;
            }
        }

        // Eigen clears the essential vector after applying the reflector.
        for row in (row_start + 1)..rows {
            qj[column_start + row] = 0.0;
        }
        if trace_reflectors && total_rank > 0 {
            let mut hash = 1_469_598_103_934_665_603_u64;
            for value in &qj {
                for byte in value.to_bits().to_le_bytes() {
                    hash ^= byte as u64;
                    hash = hash.wrapping_mul(1_099_511_628_211_u64);
                }
            }
            eprintln!(
                "trace_after {} {} {:016x} {:08x} {:08x}",
                column,
                total_rank,
                hash,
                qj[column_start + row_start].to_bits(),
                if column + 1 < cols {
                    qj[(column + 1) * rows + row_start].to_bits()
                } else {
                    0
                },
            );
        }
        if let Some(output) = trace_qj_dump.as_mut() {
            if !upstream_f32_write_qtrace_record(
                output,
                column,
                row_start,
                total_rank,
                c0,
                tail_sq_norm,
                beta,
                tau,
                &qj,
                &qr,
            ) {
                trace_qj_dump = None;
            }
        }
        if column + 1 == marg_size {
            marginal_rank = total_rank;
        }
    }
    if marg_size == 0 {
        marginal_rank = 0;
    }

    let keep_rows = total_rank.saturating_sub(marginal_rank).max(1);
    let available_rows = rows.saturating_sub(marginal_rank);
    if available_rows == 0 || keep_rows > available_rows {
        return None;
    }
    let keep_size = keep.len();
    let mut out_j = DMatrix::<f64>::zeros(keep_rows, keep_size);
    for row in 0..keep_rows {
        for column in 0..keep_size {
            out_j[(row, column)] = qj[(marg_size + column) * rows + marginal_rank + row] as f64;
        }
    }
    let out_rhs = DVector::from_iterator(
        keep_rows,
        (0..keep_rows).map(|row| qr[marginal_rank + row] as f64),
    );
    let fej_point = DVector::from_iterator(
        fej_point.len(),
        fej_point.iter().map(|value| *value as f32 as f64),
    );
    Some(super::aom::SqrtPrior {
        jacobian: out_j,
        rhs: out_rhs,
        fej_point,
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn frame16_prior27_scalar_tail_fma_regression() {
        // Native frame16 iteration6 row24, from the pinned prior-stage capture.
        // The upper-triangular prior makes depths0..23 zero for this row.
        let mut j = DMatrix::<f64>::zeros(27, 27);
        let mut b = DVector::<f64>::zeros(27);
        let mut d = DVector::<f64>::zeros(27);
        let j_bits = [0xc2209eaa, 0x4009bc46, 0x41689bab];
        let d_bits = [0x3b8461b3, 0xb785cdd0, 0xbbf14604];
        for k in 0..3 {
            j[(24, 24 + k)] = f64::from(f32::from_bits(j_bits[k]));
            d[24 + k] = f64::from(f32::from_bits(d_bits[k]));
        }
        b[24] = f64::from(f32::from_bits(0xbf587d62));
        let actual = WindowProblem::prior_rhs_col_major_27_f32(&j, &b, &d);
        assert_eq!((actual[24] as f32).to_bits(), 0xbf8eb73b);
        let mut old = 0.0_f32;
        for col in 0..27 {
            old =
                WindowProblem::prior_mul_add_separate_f32(old, j[(24, col)] as f32, d[col] as f32);
        }
        assert_eq!(
            WindowProblem::prior_add_separate_f32(b[24] as f32, old).to_bits(),
            0xbf8eb73a
        );
    }
    #[test]
    #[ignore = "requires M11_PRIOR_CAPTURE_ROOT pinned external capture"]
    fn frame10_prior27_native_adjusted_rhs() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!(inputs.len(), 8);
        assert_eq!(stages.len(), 32);
        let parse = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            let stage = |name: &str| -> Vec<f64> {
                parse(
                    &stages
                        .iter()
                        .find(|s| {
                            s["iteration"].as_u64() == Some(i as u64)
                                && s["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let j = DMatrix::from_column_slice(27, 27, &parse(&input["jacobian_bits"]));
            let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let delta = DVector::from_vec(stage("delta"));
            let expected = stage("adjusted_rhs");
            let actual = WindowProblem::prior_rhs_col_major_27_f32(&j, &b, &delta);
            for lane in 0..27 {
                assert_eq!(
                    (actual[lane] as f32).to_bits(),
                    (expected[lane] as f32).to_bits(),
                    "iteration {i} lane {lane}"
                );
            }
            if i == 0 {
                let old = b + j * delta;
                assert_eq!(
                    old.iter()
                        .zip(expected.iter())
                        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                        .count(),
                    9
                );
            }
        }
    }
    #[test]
    #[ignore = "requires external frame11 native and Rust reanchor captures"]
    fn frame11_prior27_reanchor_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let trace: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("m11_trial_wired_frame10_20260908/r11_frame11_marg6/marg_trace.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let native_text = std::fs::read_to_string(
            root.join("m11_native_frame11_prior_stages_20260909/r1/prior_inputs.jsonl"),
        )
        .unwrap();
        let native: serde_json::Value =
            serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        assert_eq!(trace["event_ordinal"].as_u64(), Some(6));
        assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
        let j = DMatrix::from_column_slice(27, 27, &parse(&native["jacobian_bits"]));
        let b = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
        let d = DVector::from_vec(parse(&trace["keep_delta_f32"]));
        let expected = parse(&native["stored_rhs_bits"]);
        assert_eq!((b.len(), d.len(), expected.len()), (27, 27, 27));
        let old = &b - &j * &d;
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            2
        );
        let actual = WindowProblem::prior_reanchor_rhs_col_major_27_f32(&j, &b, &d);
        for row in 0..27 {
            assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
        }
    }
    #[test]
    #[ignore = "requires external frame17 native and Rust reanchor captures"]
    fn frame17_prior33_reanchor_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let trace: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("m11_trial_wired_frame10_20260908/r16_frame17_marg12/marg_trace.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let native_text = std::fs::read_to_string(
            root.join("m11_native_frame17_reanchor_abi_20260909/r1/prior_inputs.jsonl"),
        )
        .unwrap();
        let native: serde_json::Value =
            serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        assert_eq!(trace["event_ordinal"].as_u64(), Some(12));
        assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
        let j = DMatrix::from_column_slice(33, 33, &parse(&native["jacobian_bits"]));
        let b = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
        let d = DVector::from_vec(parse(&trace["keep_delta_f32"]));
        let expected = parse(&native["stored_rhs_bits"]);
        assert_eq!((b.len(), d.len(), expected.len()), (33, 33, 33));
        let old = &b - &j * &d;
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            5
        );
        let actual = WindowProblem::prior_reanchor_rhs_col_major_33_f32(&j, &b, &d);
        for row in 0..33 {
            assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
        }
    }
    #[test]
    #[ignore = "requires validated native frame17 prior capture"]
    fn frame24_prior39_reanchor_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_REANCHOR_CAPTURE_ROOT").unwrap());
        let trace: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("m11_trial_wired_frame10_20260908/r21_frame24_marg19/marg_trace.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let native_text = std::fs::read_to_string(
            root.join("m11_native_frame24_reanchor_abi_20260909/r2/prior_inputs.jsonl"),
        )
        .unwrap();
        let native: serde_json::Value =
            serde_json::from_str(native_text.lines().next().unwrap()).unwrap();
        let parse = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| f32::from_bits(u32::from_str_radix(x.as_str().unwrap(), 16).unwrap()))
                .collect()
        };
        assert_eq!(trace["event_ordinal"].as_u64(), Some(19));
        assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
        let j = DMatrix::from_column_slice(39, 39, &parse(&native["jacobian_bits"]));
        let b = parse(&trace["prior_rhs_before_delta_f32"]);
        let d = parse(&trace["keep_delta_f32"]);
        let expected = parse(&native["stored_rhs_bits"]);
        assert_eq!((b.len(), d.len(), expected.len()), (39, 39, 39));
        let actual = WindowProblem::prior_reanchor_rhs_col_major_39_f32(
            &j,
            &DVector::from_vec(b.clone()),
            &DVector::from_vec(d.clone()),
        );
        for row in 0..39 {
            assert_eq!(actual[row].to_bits(), expected[row].to_bits(), "lane {row}");
        }
        let old = DVector::from_vec(b) - j * DVector::from_vec(d);
        assert_eq!(
            old.iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count(),
            6
        );
    }
    #[test]
    #[ignore = "requires validated native frame31 prior capture"]
    fn frame31_prior45_reanchor_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME31_REANCHOR_ROOT").expect("frame31 capture root"),
        );
        let trace: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("m11_frame31_marg26_r34_probe_20260912/r1/marg_trace.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let native: serde_json::Value = std::fs::read_to_string(
            root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .find(|record: &serde_json::Value| {
            record["kind"].as_str() == Some("prior_input")
                && record["iteration"].as_u64() == Some(0)
        })
        .unwrap();
        let parse = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap())
                })
                .collect()
        };
        assert_eq!(trace["event_ordinal"].as_u64(), Some(26));
        assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
        let j = DMatrix::from_column_slice(45, 45, &parse(&native["jacobian_bits"]));
        let before = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
        let delta = DVector::from_vec(parse(&trace["keep_delta_f32"]));
        let expected = parse(&native["stored_rhs_bits"]);
        let generic = &before - &j * &delta;
        assert_eq!(
            generic
                .iter()
                .zip(&expected)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count(),
            8
        );
        let candidate = WindowProblem::prior_reanchor_rhs_col_major_45_f32(&j, &before, &delta);
        for row in 0..45 {
            assert_eq!(
                candidate[row].to_bits(),
                expected[row].to_bits(),
                "lane {row}"
            );
        }
    }
    #[test]
    #[ignore = "requires validated native frame38 prior and Rust event33 captures"]
    fn frame38_prior51_reanchor_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
        );
        let trace: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(
                root.join("m11_trial_wired_frame10_20260908/r40_frame38_marg33/marg_trace.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let native: serde_json::Value = std::fs::read_to_string(
            root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .find(|record: &serde_json::Value| {
            record["kind"].as_str() == Some("prior_input")
                && record["iteration"].as_u64() == Some(0)
        })
        .unwrap();
        let parse = |value: &serde_json::Value| -> Vec<f32> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f32::from_bits(u32::from_str_radix(word.as_str().unwrap(), 16).unwrap())
                })
                .collect()
        };
        assert_eq!(trace["event_ordinal"].as_u64(), Some(33));
        assert_eq!(trace["prior_jacobian_f32"]["bits"], native["jacobian_bits"]);
        let j = DMatrix::from_column_slice(51, 51, &parse(&native["jacobian_bits"]));
        let before = DVector::from_vec(parse(&trace["prior_rhs_before_delta_f32"]));
        let delta = DVector::from_vec(parse(&trace["keep_delta_f32"]));
        let expected = parse(&native["stored_rhs_bits"]);
        let generic = &before - &j * &delta;
        assert_eq!(
            generic
                .iter()
                .zip(&expected)
                .filter(|(actual, expected)| actual.to_bits() != expected.to_bits())
                .count(),
            15
        );
        let candidate = WindowProblem::prior_reanchor_rhs_col_major_51_f32(&j, &before, &delta);
        for row in 0..51 {
            assert_eq!(
                candidate[row].to_bits(),
                expected[row].to_bits(),
                "lane {row}"
            );
        }
    }
    #[test]
    #[ignore = "requires validated native frame38 prior capture"]
    fn frame38_prior51_adjusted_rhs_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
        );
        let records: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let parse = |value: &serde_json::Value| -> Vec<f64> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for iteration in 0..8 {
            let input = records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_input")
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap();
            let stage = |name: &str| {
                records
                    .iter()
                    .find(|record| {
                        record["kind"].as_str() == Some("prior_stage")
                            && record["stage"].as_str() == Some(name)
                            && record["iteration"].as_u64() == Some(iteration)
                    })
                    .unwrap()
            };
            let j = DMatrix::from_column_slice(51, 51, &parse(&input["jacobian_bits"]));
            let stored = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
            let expected = parse(&stage("adjusted_rhs")["bits"]);
            let candidate = WindowProblem::prior_rhs_col_major_51_f32(&j, &stored, &delta);
            for row in 0..51 {
                assert_eq!(
                    (candidate[row] as f32).to_bits(),
                    (expected[row] as f32).to_bits(),
                    "iteration {iteration} lane {row}"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires validated native frame31 prior capture"]
    fn frame31_prior45_adjusted_rhs_candidate() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME31_REANCHOR_ROOT").expect("frame31 capture root"),
        );
        let records: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame31_prior_stages_20260912/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let parse = |value: &serde_json::Value| -> Vec<f64> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for iteration in 0..8 {
            let input = records
                .iter()
                .find(|record| {
                    record["kind"].as_str() == Some("prior_input")
                        && record["iteration"].as_u64() == Some(iteration)
                })
                .unwrap();
            let stage = |name: &str| {
                records
                    .iter()
                    .find(|record| {
                        record["kind"].as_str() == Some("prior_stage")
                            && record["stage"].as_str() == Some(name)
                            && record["iteration"].as_u64() == Some(iteration)
                    })
                    .unwrap()
            };
            let j = DMatrix::from_column_slice(45, 45, &parse(&input["jacobian_bits"]));
            let stored = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
            let expected = parse(&stage("adjusted_rhs")["bits"]);
            let candidate = WindowProblem::prior_rhs_col_major_45_f32(&j, &stored, &delta);
            for row in 0..45 {
                assert_eq!(
                    (candidate[row] as f32).to_bits(),
                    (expected[row] as f32).to_bits(),
                    "iteration {iteration} lane {row}"
                );
            }
        }
    }
    #[test]
    #[ignore = "requires validated native frame17 prior capture"]
    fn frame17_prior33_adjusted_rhs_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!((inputs.len(), stages.len()), (8, 32));
        let parse = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(
                (input["rows"].as_u64(), input["cols"].as_u64()),
                (Some(33), Some(33))
            );
            let stage = |name: &str| -> Vec<f64> {
                parse(
                    &stages
                        .iter()
                        .find(|s| {
                            s["iteration"].as_u64() == Some(i as u64)
                                && s["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let j = DMatrix::from_column_slice(33, 33, &parse(&input["jacobian_bits"]));
            let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let d = DVector::from_vec(stage("delta"));
            let expected = stage("adjusted_rhs");
            let actual = WindowProblem::prior_rhs_col_major_33_f32(&j, &b, &d);
            for row in 0..33 {
                assert_eq!(
                    (actual[row] as f32).to_bits(),
                    (expected[row] as f32).to_bits(),
                    "iteration {i} lane {row}"
                );
            }
            if i == 0 {
                let old = b + j * d;
                assert_eq!(
                    old.iter()
                        .zip(&expected)
                        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                        .count(),
                    15
                );
            }
        }
    }
    #[test]
    #[ignore = "requires validated native frame24 prior capture"]
    fn frame24_prior39_adjusted_rhs_candidate() {
        let root = std::path::PathBuf::from(std::env::var("M11_PRIOR_CAPTURE_ROOT").unwrap());
        let read = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect()
        };
        let inputs = read("prior_inputs.jsonl");
        let stages = read("prior_stages.jsonl");
        assert_eq!((inputs.len(), stages.len()), (8, 32));
        let parse = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(x.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(
                (input["rows"].as_u64(), input["cols"].as_u64()),
                (Some(39), Some(39))
            );
            let stage = |name: &str| -> Vec<f64> {
                parse(
                    &stages
                        .iter()
                        .find(|s| {
                            s["iteration"].as_u64() == Some(i as u64)
                                && s["stage"].as_str() == Some(name)
                        })
                        .unwrap()["bits"],
                )
            };
            let j = DMatrix::from_column_slice(39, 39, &parse(&input["jacobian_bits"]));
            let b = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let d = DVector::from_vec(stage("delta"));
            let expected = stage("adjusted_rhs");
            let actual = WindowProblem::prior_rhs_col_major_39_f32(&j, &b, &d);
            for row in 0..39 {
                assert_eq!(
                    (actual[row] as f32).to_bits(),
                    (expected[row] as f32).to_bits(),
                    "iteration {i} lane {row}"
                );
            }
            if i == 0 {
                let old = b + j * d;
                assert_eq!(
                    old.iter()
                        .zip(&expected)
                        .filter(|(a, b)| (**a as f32).to_bits() != (**b as f32).to_bits())
                        .count(),
                    20
                );
            }
        }
    }
    use super::*;
    use crate::TimingStat;
    use nalgebra::{Point3, Vector6};

    #[test]
    fn upstream_f32_householder_beta_uses_fused_pivot_square() {
        // Captured native event-1 column 20 operands.  The fused Eigen
        // association produces 4647bb33; the algebraically equivalent
        // non-fused product/add produces 4647bb34 on the pinned boundary.
        let c0 = f32::from_bits(0xc6452f66);
        let tail_sq_norm = f32::from_bits(0x4a7ca5af);
        let fused = super::upstream_f32_householder_beta(c0, tail_sq_norm);
        let product = std::hint::black_box(c0 * c0);
        let non_fused = std::hint::black_box(product + tail_sq_norm).sqrt();
        assert_eq!(fused.to_bits(), 0x4647bb33);
        assert_eq!(non_fused.to_bits(), 0x4647bb34);
    }

    #[test]
    fn diagnostic_env_allowlist_keeps_harness_fast_and_fails_closed() {
        let canonical = [
            "VISLOC_BASALT_GT_FREE",
            "VISLOC_BASALT_PROFILE",
            "VISLOC_BASALT_SEQUENCE",
            "VISLOC_BASALT_INPUT_ROOT",
            "VISLOC_BASALT_OUTPUT_ROOT",
            "VISLOC_BASALT_SEED",
            "VISLOC_BASALT_TEMPORAL_SEED",
            "VISLOC_BASALT_THREADS",
            "VISLOC_BASALT_TIMING_BREAKDOWN",
            "VISLOC_BASALT_TRACE",
            "VISLOC_BASALT_M8D_SETUP_INPUT",
            "VISLOC_BASALT_M8D_SETUP_ORACLE",
            "VISLOC_BASALT_MH01_ROOT",
            "VISLOC_BASALT_CALIBRATION",
            "VISLOC_BASALT_CONFIG",
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACE",
            "VISLOC_BASALT_VISUAL_FACTOR_IDENTITY_TRACKS",
            "VISLOC_BASALT_LIFECYCLE_TRACE",
        ]
        .into_iter()
        .map(std::ffi::OsString::from)
        .collect::<Vec<_>>();
        assert!(!diagnostic_env_active_for_keys(canonical));

        for key in [
            "visloc_basalt_gt_free",
            "VisLoc_Basalt_Profile",
            "vIsLoC_bAsAlT_tHrEaDs",
            "vIsLoC_bAsAlT_tImInG_bReAkDoWn",
            "VISLOC_basalt_mh01_root",
            "visloc_BASALT_config",
        ] {
            assert!(!diagnostic_env_active_for_keys([std::ffi::OsString::from(
                key
            )]));
        }

        for key in [
            "VISLOC_BASALT_DETAIL_ITERATIONS",
            "VISLOC_BASALT_DIAGNOSTIC_IMU_HB",
            "VISLOC_BASALT_FUTURE_KEY",
        ] {
            assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
                key
            )]));
        }
        for key in [
            "visloc_basalt_detail_iterations",
            "VisLoc_Basalt_Diagnostic_Imu_Hb",
            "vIsLoC_bAsAlT_fUtUrE_kEy",
        ] {
            assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
                key
            )]));
        }
        assert!(!diagnostic_env_active_for_keys([std::ffi::OsString::from(
            "UNRELATED_ENV"
        )]));
    }

    #[test]
    fn visual_prefix_selector_snapshot_parses_typed_values_once_and_fails_closed() {
        let mut values = HashMap::new();
        values.insert("VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID", OsString::from("12"));
        values.insert("VISLOC_BASALT_VISUAL_PREFIX_ITERATION", OsString::from("3"));
        values.insert("VISLOC_BASALT_VISUAL_PREFIX_TRIAL", OsString::from("0"));
        assert_eq!(
            cached_selector_u64(&values, "VISLOC_BASALT_VISUAL_PREFIX_FRAME_ID"),
            Some(Ok(12))
        );
        assert_eq!(
            cached_selector_usize(&values, "VISLOC_BASALT_VISUAL_PREFIX_ITERATION"),
            Some(Ok(3))
        );
        values.insert(
            "VISLOC_BASALT_VISUAL_PREFIX_TRIAL",
            OsString::from("not-a-number"),
        );
        assert_eq!(
            cached_selector_usize(&values, "VISLOC_BASALT_VISUAL_PREFIX_TRIAL"),
            Some(Err(()))
        );
    }

    #[test]
    fn diagnostic_env_snapshot_is_observed_in_fresh_processes() {
        fn run_child(diagnostic_key: Option<&str>, expected: bool) {
            let executable = std::env::current_exe().expect("test executable path");
            let mut command = std::process::Command::new(executable);
            // Do not let the parent test environment decide the child result.
            // Keep ordinary process variables intact, but remove every
            // VISLOC_BASALT_* key before installing the one case under test.
            for (key, _) in std::env::vars_os() {
                if key
                    .to_string_lossy()
                    .get(.."VISLOC_BASALT_".len())
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("VISLOC_BASALT_"))
                {
                    command.env_remove(key);
                }
            }
            command
                .env("VISLOC_DIAGNOSTIC_SNAPSHOT_CHILD", "1")
                .env(
                    "VISLOC_DIAGNOSTIC_SNAPSHOT_EXPECTED",
                    if expected { "true" } else { "false" },
                )
                .args(["diagnostic_env_snapshot_child", "--nocapture"]);
            if let Some(key) = diagnostic_key {
                command.env(key, "1");
            }
            let output = command.output().expect("spawn fresh diagnostic child");
            assert!(
                output.status.success(),
                "child failed: stdout={} stderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Each child snapshots its environment on its first call.  Separate
        // processes make this independent of unit-test ordering and avoid
        // mutating the parent's environment.
        run_child(Some("VISLOC_BASALT_SNAPSHOT_UNKNOWN"), true);
        run_child(Some("vIsLoC_bAsAlT_PrOfIlE"), false);
        run_child(None, false);
    }

    #[test]
    fn diagnostic_env_snapshot_child() {
        if std::env::var_os("VISLOC_DIAGNOSTIC_SNAPSHOT_CHILD").is_none() {
            return;
        }
        let expected = std::env::var("VISLOC_DIAGNOSTIC_SNAPSHOT_EXPECTED")
            .expect("child expected value")
            == "true";
        assert_eq!(diagnostic_env_active(), expected);
        // A second read must return the same process-lifetime value.
        assert_eq!(diagnostic_env_active(), expected);
    }

    #[test]
    fn imu_f64_observer_scope_serializes_actual_product_without_recomputing() {
        assert!(!initial_imu_f64_capture_scope_active());
        let from = nav(10, 0.0);
        let mut to = nav(11, 0.2);
        to.nav.imu_to_world.rotation =
            UnitQuaternion::from_scaled_axis(Vector3::new(0.013, -0.027, 0.041));
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.05;
        delta.delta_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(-0.009, 0.017, 0.023));

        let (_, rotation, _, product) =
            imu_unwhitened_residual(&from.nav, &to.nav, &delta, Vector3::new(0.0, 0.0, -9.81));
        let record = build_imu_unwhitened_residual_f64_capture(
            Some(to.frame_id),
            &from,
            &to,
            0,
            0,
            1,
            None,
            &product,
            rotation,
        )
        .expect("active frame is required for an inline capture");

        let bits = |value: f64| format!("{:016x}", value.to_bits());
        let q = product.quaternion();
        assert_eq!(
            record["product_xyzw_f64_bits"],
            json!([bits(q.i), bits(q.j), bits(q.k), bits(q.w)])
        );
        assert_eq!(
            record["observed_r_R_f64_bits"],
            json!([bits(rotation.x), bits(rotation.y), bits(rotation.z)])
        );
        assert_eq!(record["capture_proof"]["product_same_call"], json!(true));
        assert_eq!(record["capture_proof"]["result_same_call"], json!(true));
        assert_eq!(record["capture_proof"]["norm_captured"], json!(false));
        assert_eq!(record["capture_proof"]["angle_captured"], json!(false));
        assert!(record.get("norm_f64_bits").is_none());
        assert!(record.get("candidate").is_none());
        assert!(record.get("native_oracle").is_none());
        assert!(!initial_imu_f64_capture_scope_active());

        let scope = InitialImuF64CaptureScope::enter();
        assert!(initial_imu_f64_capture_scope_active());
        let first_link = InitialImuF64CaptureLink::set(4);
        assert_eq!(initial_imu_f64_capture_link_index(), Some(4));
        {
            let second_link = InitialImuF64CaptureLink::set(9);
            assert_eq!(initial_imu_f64_capture_link_index(), Some(9));
            drop(second_link);
        }
        assert_eq!(initial_imu_f64_capture_link_index(), Some(4));
        drop(first_link);
        assert_eq!(initial_imu_f64_capture_link_index(), None);
        drop(scope);
        assert!(!initial_imu_f64_capture_scope_active());

        // Constructing the observation must not replace or perturb the
        // production helper's already-computed f64 values.
        let (_, rotation_again, _, product_again) =
            imu_unwhitened_residual(&from.nav, &to.nav, &delta, Vector3::new(0.0, 0.0, -9.81));
        let q_again = product_again.quaternion();
        assert_eq!(q.i.to_bits(), q_again.i.to_bits());
        assert_eq!(q.j.to_bits(), q_again.j.to_bits());
        assert_eq!(q.k.to_bits(), q_again.k.to_bits());
        assert_eq!(q.w.to_bits(), q_again.w.to_bits());
        for (lhs, rhs) in rotation.iter().zip(rotation_again.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }

    fn nav(frame: u64, x: f64) -> WindowState {
        WindowState {
            frame_id: frame,
            timestamp_ns: frame as i64 + 1,
            nav: BasaltNavState {
                imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
                ..BasaltNavState::default()
            },
            stored_current_nav: BasaltNavState {
                imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
                ..BasaltNavState::default()
            },
            linearized_nav: BasaltNavState {
                imu_to_world: SE3::new(UnitQuaternion::identity(), Vector3::new(x, 0.0, 0.0)),
                ..BasaltNavState::default()
            },
            linearized_delta: DVector::zeros(NAV_STATE_DOF),
            is_keyframe: frame == 0,
            is_latest: frame == 2,
            linearized: frame == 0,
        }
    }

    fn test_window(states: Vec<WindowState>, landmarks: Vec<WindowLandmark>) -> WindowProblem {
        // These fresh fixtures insert hosts in landmark order, with no prior
        // erase/rehash history. Production carries its persistent order.
        let mut host_order = super::super::landmarks::NativeHostOrder::default();
        for landmark in &landmarks {
            if !landmark.observations.is_empty() {
                host_order.insert(
                    states[landmark.anchor_state_index].timestamp_ns as u64,
                    landmark.anchor_camera_id,
                );
            }
        }
        WindowProblem {
            trial_host_order: host_order.keys().collect(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states,
            landmarks,
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
        }
    }

    fn assert_window_dynamic_eq(lhs: &WindowProblem, rhs: &WindowProblem) {
        assert_eq!(lhs.poses, rhs.poses);
        assert_eq!(lhs.states, rhs.states);
        assert_eq!(lhs.landmarks, rhs.landmarks);
    }

    fn assert_token_rejected_without_mutation(
        candidate: &mut WindowProblem,
        state: &DVector<f64>,
        step: &DVector<f64>,
        token: LmTrialToken,
    ) {
        let before = candidate.clone();
        let result = candidate.accept_step_with_token(state, step, token);
        assert!(matches!(result, Err(LmFailure::LinearSolve)));
        assert_window_dynamic_eq(candidate, &before);
        assert_eq!(candidate.prior, before.prior);
        assert_eq!(candidate.anchor_point, before.anchor_point);
        assert_eq!(candidate.scalar_mode, before.scalar_mode);
    }

    fn token_binding_window() -> WindowProblem {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        test_window(
            vec![nav(0, 0.0), nav(1, 0.25)],
            vec![WindowLandmark {
                track_id: 23,
                anchor_state_index: 0,
                anchor_camera_id: 0,
                direction,
                inverse_distance: 0.75,
                observations: Vec::new(),
            }],
        )
    }

    fn visual_trial_fixture() -> (
        WindowProblem,
        DVector<f64>,
        DVector<f64>,
        DVector<f64>,
        Vec<Option<Vector3<f64>>>,
    ) {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        let landmark = WindowLandmark {
            track_id: 17,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: vec![
                WindowObservation {
                    state_index: 1,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                },
                WindowObservation {
                    state_index: 2,
                    camera_id: 0,
                    pixel: Point2::new(319.0, 241.0),
                },
            ],
        };
        let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25), nav(2, 0.5)], vec![landmark]);
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[20] = -0.03125;
        step[30] = -0.0234375;
        step[35] = 0.046875;
        let trial = problem.apply_step(&state, &step);
        let landmark_steps = problem.landmark_steps(&state, &step);
        (problem, state, step, trial, landmark_steps)
    }

    #[test]
    fn clean_trial_view_shares_topology_and_preserves_accept_reject_semantics() {
        let problem = test_window(vec![nav(1, 0.0)], Vec::new());
        let before = problem.clone();
        let state = problem.initial_state();
        let mut step = DVector::zeros(NAV_STATE_DOF);
        step[0] = 1.25;
        step[5] = 0.03125;
        let trial = problem.apply_step(&state, &step);
        let view = WindowTrialView::from_step(&problem, &state, &step, &trial);

        assert!(std::ptr::eq(view.base, &problem));
        assert_window_dynamic_eq(&problem, &before);
        assert_ne!(
            view.states[0].nav.imu_to_world.translation.x.to_bits(),
            problem.states[0].nav.imu_to_world.translation.x.to_bits()
        );

        // A rejected trial is dropped without touching the accepted sidecar;
        // an accepted step follows the same candidate manifold values.
        drop(view);
        assert_window_dynamic_eq(&problem, &before);
        let mut accepted = problem.clone();
        accepted.accept_step(&state, &step).unwrap();
        assert_eq!(
            accepted.states[0].nav.imu_to_world.translation.x.to_bits(),
            trial[0].to_bits()
        );
    }

    #[test]
    fn timed_trial_view_subbuckets_preserve_values_and_disabled_path_is_empty() {
        let problem = token_binding_window();
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.25;
        step[20] = -0.03125;
        let trial = problem.apply_step(&state, &step);
        let baseline = WindowTrialView::from_step(&problem, &state, &step, &trial);

        let mut disabled = TimingBreakdown::default();
        let disabled_view =
            WindowTrialView::from_step_timed(&problem, &state, &step, &trial, &mut disabled);
        assert_eq!(disabled_view.poses, baseline.poses);
        assert_eq!(disabled_view.states, baseline.states);
        assert_eq!(disabled_view.landmarks, baseline.landmarks);
        assert_eq!(disabled_view.landmark_steps, baseline.landmark_steps);
        assert_eq!(disabled_view.prior_fej_delta, baseline.prior_fej_delta);
        assert_eq!(disabled.lm_trial_construct_step, TimingStat::default());
        assert_eq!(disabled.lm_trial_landmark_recovery, TimingStat::default());
        assert_eq!(disabled.lm_trial_apply_step_full, TimingStat::default());
        assert_eq!(
            disabled.lm_trial_landmark_materialization,
            TimingStat::default()
        );
        assert_eq!(disabled.lm_trial_sidecar_other, TimingStat::default());

        let mut timing = TimingBreakdown::enabled_for_test();
        let timed = WindowTrialView::from_step_timed(&problem, &state, &step, &trial, &mut timing);
        assert_eq!(timed.poses, baseline.poses);
        assert_eq!(timed.states, baseline.states);
        assert_eq!(timed.landmarks, baseline.landmarks);
        assert_eq!(timed.landmark_steps, baseline.landmark_steps);
        assert_eq!(timed.prior_fej_delta, baseline.prior_fej_delta);
        assert_eq!(
            timed.cost().unwrap().to_bits(),
            baseline.cost().unwrap().to_bits()
        );

        assert_eq!(timing.lm_trial_construct_step.count, 1);
        assert_eq!(timing.lm_trial_landmark_recovery.count, 1);
        assert_eq!(timing.lm_trial_apply_step_full.count, 1);
        assert_eq!(timing.lm_trial_landmark_materialization.count, 1);
        // The faithful source order places pose/state sidecars before the
        // landmark loop and view/prior sidecars after it, hence two sequential
        // samples in this one bucket.
        assert_eq!(timing.lm_trial_sidecar_other.count, 2);
        assert!(
            timing.lm_trial_construct_subbuckets_ns() <= timing.lm_trial_construct_step.total_ns
        );
    }

    #[test]
    fn clean_trial_token_matches_recomputed_landmark_recovery_exactly() {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        let landmark = WindowLandmark {
            track_id: 17,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: vec![
                WindowObservation {
                    state_index: 1,
                    camera_id: 0,
                    pixel: Point2::new(320.0, 240.0),
                },
                WindowObservation {
                    state_index: 2,
                    camera_id: 0,
                    pixel: Point2::new(319.0, 241.0),
                },
            ],
        };
        let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25), nav(2, 0.5)], vec![landmark]);
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[20] = -0.03125;
        step[30] = -0.0234375;
        step[35] = 0.046875;
        let trial = problem.apply_step(&state, &step);
        let expected_steps = problem.landmark_steps(&state, &step);
        assert!(expected_steps.iter().any(Option::is_some));

        let mut optimized = problem.clone();
        let before = optimized.clone();
        let binding = optimized.lm_trial_binding(&state, &step);
        let (optimized_cost, token) = optimized
            .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
            .unwrap();
        let token_steps = token.take_landmark_steps().expect("clean trial token");
        assert_eq!(token_steps.len(), expected_steps.len());
        for (token_step, expected_step) in token_steps.iter().zip(&expected_steps) {
            match (token_step, expected_step) {
                (Some(token_step), Some(expected_step)) => {
                    assert_eq!(token_step.x.to_bits(), expected_step.x.to_bits());
                    assert_eq!(token_step.y.to_bits(), expected_step.y.to_bits());
                    assert_eq!(token_step.z.to_bits(), expected_step.z.to_bits());
                }
                (None, None) => {}
                mismatch => panic!("landmark token mismatch: {mismatch:?}"),
            }
        }
        let token = LmTrialToken::with_bound_landmark_steps(token_steps, binding);
        optimized
            .accept_step_with_token(&state, &step, token)
            .unwrap();

        let mut legacy = problem.clone();
        let legacy_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
            .cost()
            .unwrap();
        legacy.accept_step(&state, &step).unwrap();
        assert_eq!(optimized_cost.to_bits(), legacy_cost.to_bits());
        assert_window_dynamic_eq(&optimized, &legacy);
        assert_eq!(optimized.prior, legacy.prior);
        assert_eq!(optimized.anchor_point, legacy.anchor_point);
        assert_window_dynamic_eq(&before, &problem);
    }

    #[test]
    fn trial_host_order_changes_token_generation_and_factor_plan() {
        let mut problem = token_binding_window();
        problem.trial_host_order = vec![(100, 0), (200, 1)];
        let generation = lm_token_window_generation(&problem);
        let plan = lm_token_factor_plan_fingerprint(&problem);
        problem.trial_host_order.reverse();
        assert_ne!(generation, lm_token_window_generation(&problem));
        assert_ne!(plan, lm_token_factor_plan_fingerprint(&problem));
        problem.trial_host_order.reverse();
        assert_eq!(generation, lm_token_window_generation(&problem));
        assert_eq!(plan, lm_token_factor_plan_fingerprint(&problem));
        problem.trial_host_order[0].1 = 1;
        assert_ne!(generation, lm_token_window_generation(&problem));
        assert_ne!(plan, lm_token_factor_plan_fingerprint(&problem));
    }

    #[test]
    fn prepared_trial_view_matches_legacy_view_and_cost_bitwise() {
        let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
        let legacy = WindowTrialView::from_step(&problem, &state, &step, &trial);
        let prepared = WindowTrialView::from_step_with_landmark_steps(
            &problem,
            &state,
            &step,
            &trial,
            landmark_steps.clone(),
        );

        assert_eq!(prepared.poses, legacy.poses);
        assert_eq!(prepared.states, legacy.states);
        assert_eq!(prepared.landmarks, legacy.landmarks);
        assert_eq!(prepared.landmark_steps, legacy.landmark_steps);
        assert_eq!(prepared.prior_fej_delta, legacy.prior_fej_delta);
        assert_eq!(
            prepared.cost().unwrap().to_bits(),
            legacy.cost().unwrap().to_bits()
        );

        let preparation = LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &step,
            1e-10,
            vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
        );
        let mut candidate = problem.clone();
        let (prepared_cost, token) = candidate
            .trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut TimingBreakdown::default(),
            )
            .unwrap();
        assert_eq!(prepared_cost.to_bits(), legacy.cost().unwrap().to_bits());
        candidate
            .accept_step_with_token(&state, &step, token)
            .unwrap();

        let mut expected = problem.clone();
        expected.accept_step(&state, &step).unwrap();
        assert_window_dynamic_eq(&candidate, &expected);
    }

    #[test]
    fn prepared_trial_timing_keeps_recovery_inside_existing_parent_bucket() {
        let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
        let preparation = LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &step,
            1e-10,
            vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
        );
        let mut timing = TimingBreakdown::enabled_for_test();
        let (_, token) = problem
            .trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut timing,
            )
            .unwrap();
        drop(token);
        assert_eq!(timing.lm_trial_construct_step.count, 1);
        assert_eq!(timing.lm_trial_landmark_recovery.count, 1);
        assert_eq!(timing.lm_trial_apply_step_full.count, 1);
        assert_eq!(timing.lm_trial_landmark_materialization.count, 1);
        assert_eq!(timing.lm_trial_sidecar_other.count, 2);
        assert!(
            timing.lm_trial_construct_subbuckets_ns() <= timing.lm_trial_construct_step.total_ns
        );
    }

    #[test]
    fn missing_preparation_uses_the_legacy_window_token_path() {
        let (problem, state, step, trial, _) = visual_trial_fixture();
        let mut prepared_timing = TimingBreakdown::default();
        let prepared = problem
            .trial_cost_timed_with_preparation(&state, &step, &trial, None, &mut prepared_timing)
            .unwrap();
        let mut legacy_timing = TimingBreakdown::default();
        let legacy = problem
            .trial_cost_timed_with_token(&state, &step, &trial, &mut legacy_timing)
            .unwrap();
        assert_eq!(prepared.0.to_bits(), legacy.0.to_bits());
        assert_eq!(
            prepared.1.take_landmark_steps().unwrap(),
            legacy.1.take_landmark_steps().unwrap()
        );
    }

    #[test]
    fn prepared_landmark_mapping_rejects_bad_inputs_before_mutation() {
        let (problem, state, step, trial, landmark_steps) = visual_trial_fixture();
        let make_preparation = |tolerance: f64, entries: Vec<(usize, u64)>| {
            LmTrialPreparation::from_test_entries_for_state_step(
                &state,
                &step,
                tolerance,
                entries
                    .into_iter()
                    .map(|(index, track_id)| (index, track_id, landmark_steps[0]))
                    .collect(),
            )
        };
        let cases = [
            ("tolerance", make_preparation(1e-9, vec![(0, 17)])),
            ("index", make_preparation(1e-10, vec![(1, 17)])),
            ("track", make_preparation(1e-10, vec![(0, 18)])),
        ];
        for (name, preparation) in cases {
            let candidate = problem.clone();
            let before = candidate.clone();
            let result = candidate.trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut TimingBreakdown::default(),
            );
            assert!(matches!(result, Err(LmFailure::LinearSolve)), "{name}");
            assert_window_dynamic_eq(&candidate, &before);
        }

        let mut stale_state = state.clone();
        stale_state[0] += 0.125;
        let stale_state_preparation = LmTrialPreparation::from_test_entries_for_state_step(
            &stale_state,
            &step,
            1e-10,
            vec![(0, 17, landmark_steps[0])],
        );
        let candidate = problem.clone();
        let before = candidate.clone();
        let result = candidate.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(stale_state_preparation),
            &mut TimingBreakdown::default(),
        );
        assert!(
            matches!(result, Err(LmFailure::LinearSolve)),
            "state fingerprint"
        );
        assert_window_dynamic_eq(&candidate, &before);

        let mut stale_step = step.clone();
        stale_step[0] += 0.125;
        let stale_step_preparation = LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &stale_step,
            1e-10,
            vec![(0, 17, landmark_steps[0])],
        );
        let candidate = problem.clone();
        let before = candidate.clone();
        let result = candidate.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(stale_step_preparation),
            &mut TimingBreakdown::default(),
        );
        assert!(
            matches!(result, Err(LmFailure::LinearSolve)),
            "step fingerprint"
        );
        assert_window_dynamic_eq(&candidate, &before);

        let mut duplicate_problem = problem;
        duplicate_problem
            .landmarks
            .push(duplicate_problem.landmarks[0].clone());
        duplicate_problem.landmarks[1].track_id = 18;
        let duplicate = LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &step,
            1e-10,
            vec![(0, 17, landmark_steps[0]), (0, 17, landmark_steps[0])],
        );
        let before = duplicate_problem.clone();
        let result = duplicate_problem.trial_cost_timed_with_preparation(
            &state,
            &step,
            &trial,
            Some(duplicate),
            &mut TimingBreakdown::default(),
        );
        assert!(matches!(result, Err(LmFailure::LinearSolve)));
        assert_window_dynamic_eq(&duplicate_problem, &before);
    }

    #[test]
    fn prepared_mixed_prior_visual_imu_matches_legacy_coverage_and_accept() {
        let (mut problem, _, _, _, _) = visual_trial_fixture();
        // Keep the initial anchor row and one chronological IMU+bias pair in
        // the same trial as the visual landmark.  This exercises the full
        // factor order that the clean reducer hands to the preparation hook.
        problem.anchor_point = Some(DVector::zeros(NAV_STATE_DOF));
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 1.0;
        problem.imu_links.push(WindowImuLink {
            from_index: 0,
            to_index: 1,
            delta,
        });

        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[20] = -0.03125;
        step[30] = -0.0234375;
        step[35] = 0.046875;
        let trial = problem.apply_step(&state, &step);
        let landmark_steps = problem.landmark_steps(&state, &step);
        let factors = problem.linearize_snapshot(&state).unwrap();
        assert_eq!(
            factors
                .iter()
                .filter(|factor| factor.kind == FactorKind::Prior)
                .count(),
            1
        );
        assert_eq!(
            factors
                .iter()
                .filter(|factor| factor.kind == FactorKind::Visual)
                .count(),
            1
        );
        assert_eq!(
            factors
                .iter()
                .filter(|factor| factor.kind == FactorKind::Imu)
                .count(),
            1
        );
        assert_eq!(
            factors
                .iter()
                .filter(|factor| factor.kind == FactorKind::Bias)
                .count(),
            1
        );

        let preparation = LmTrialPreparation::from_test_entries_for_state_step(
            &state,
            &step,
            1e-10,
            vec![(0, problem.landmarks[0].track_id, landmark_steps[0])],
        );
        let mut prepared = problem.clone();
        let (prepared_cost, prepared_token) = prepared
            .trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut TimingBreakdown::default(),
            )
            .unwrap();
        let mut legacy = problem.clone();
        let (legacy_cost, legacy_token) = legacy
            .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
            .unwrap();
        assert_eq!(prepared_cost.to_bits(), legacy_cost.to_bits());

        prepared
            .accept_step_with_token(&state, &step, prepared_token)
            .unwrap();
        legacy
            .accept_step_with_token(&state, &step, legacy_token)
            .unwrap();
        assert_window_dynamic_eq(&prepared, &legacy);
        assert_eq!(prepared.prior, legacy.prior);
        assert_eq!(prepared.anchor_point, legacy.anchor_point);
    }

    #[test]
    fn compact_reducer_preparation_mixed_window_path_matches_legacy_end_to_end() {
        let (mut problem, _, _, _, _) = visual_trial_fixture();
        problem.anchor_point = Some(DVector::zeros(NAV_STATE_DOF));
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 1.0;
        problem.imu_links.push(WindowImuLink {
            from_index: 0,
            to_index: 1,
            delta,
        });

        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[20] = -0.03125;
        step[30] = -0.0234375;
        step[35] = 0.046875;
        let trial = problem.apply_step(&state, &step);
        let factors = problem.linearize_snapshot(&state).unwrap();
        let preparation = compact_trial_preparation_for_test(&factors, &state, &step, 1e-10)
            .expect("mixed compact reducer must produce Window preparation");

        let mut prepared = problem.clone();
        let (prepared_cost, prepared_token) = prepared
            .trial_cost_timed_with_preparation(
                &state,
                &step,
                &trial,
                Some(preparation),
                &mut TimingBreakdown::default(),
            )
            .unwrap();
        let mut legacy = problem.clone();
        let (legacy_cost, legacy_token) = legacy
            .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
            .unwrap();
        assert_eq!(prepared_cost.to_bits(), legacy_cost.to_bits());

        prepared
            .accept_step_with_token(&state, &step, prepared_token)
            .unwrap();
        legacy
            .accept_step_with_token(&state, &step, legacy_token)
            .unwrap();
        assert_window_dynamic_eq(&prepared, &legacy);
        assert_eq!(prepared.prior, legacy.prior);
        assert_eq!(prepared.anchor_point, legacy.anchor_point);
    }

    #[test]
    fn clean_trial_token_drop_and_malformed_token_leave_window_unchanged() {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        let landmark = WindowLandmark {
            track_id: 19,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: Vec::new(),
        };
        let problem = test_window(vec![nav(0, 0.0)], vec![landmark]);
        let state = problem.initial_state();
        let step = DVector::zeros(problem.state_dof());

        let dropped = problem.clone();
        let trial = dropped.apply_step(&state, &step);
        let (_, token) = dropped
            .trial_cost_timed_with_token(&state, &step, &trial, &mut TimingBreakdown::default())
            .unwrap();
        drop(token);
        assert_window_dynamic_eq(&dropped, &problem);

        let mut malformed = problem.clone();
        let result = malformed.accept_step_with_token(
            &state,
            &step,
            LmTrialToken::with_landmark_steps(Vec::new()),
        );
        assert!(matches!(result, Err(LmFailure::LinearSolve)));
        assert_window_dynamic_eq(&malformed, &problem);
        assert_eq!(malformed.prior, problem.prior);
        assert_eq!(malformed.anchor_point, problem.anchor_point);
    }

    #[test]
    fn empty_trial_token_preserves_legacy_accept_fallback() {
        let problem = test_window(vec![nav(0, 0.0), nav(1, 0.25)], Vec::new());
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.25;
        step[20] = 0.03125;

        let mut fallback = problem.clone();
        fallback
            .accept_step_with_token(&state, &step, LmTrialToken::default())
            .unwrap();
        let mut legacy = problem;
        legacy.accept_step(&state, &step).unwrap();
        assert_window_dynamic_eq(&fallback, &legacy);
    }

    #[test]
    fn clean_trial_token_rejects_cross_context_bindings_before_mutation() {
        let owner = token_binding_window();
        let state = owner.initial_state();
        let mut step = DVector::zeros(owner.state_dof());
        step[15] = 0.25;
        let binding = owner.lm_trial_binding(&state, &step);
        let token_steps = vec![None; owner.landmarks.len()];

        // A clone has the same topology but is a different WindowProblem
        // identity, so a token from the owner must not write into it.
        let mut cross_window = token_binding_window();
        assert_token_rejected_without_mutation(
            &mut cross_window,
            &state,
            &step,
            LmTrialToken::with_bound_landmark_steps(token_steps.clone(), binding),
        );

        let mut state_candidate = token_binding_window();
        let state_candidate_state = state_candidate.initial_state();
        let state_binding = state_candidate.lm_trial_binding(&state_candidate_state, &step);
        let mut changed_state = state_candidate_state;
        changed_state[0] += 0.125;
        assert_token_rejected_without_mutation(
            &mut state_candidate,
            &changed_state,
            &step,
            LmTrialToken::with_bound_landmark_steps(token_steps.clone(), state_binding),
        );

        let mut changed_step = step.clone();
        changed_step[15] += 0.125;
        let mut step_candidate = token_binding_window();
        let step_candidate_state = step_candidate.initial_state();
        let step_binding = step_candidate.lm_trial_binding(&step_candidate_state, &step);
        assert_token_rejected_without_mutation(
            &mut step_candidate,
            &step_candidate_state,
            &changed_step,
            LmTrialToken::with_bound_landmark_steps(token_steps.clone(), step_binding),
        );

        let mut topology_candidate = token_binding_window();
        let topology_state = topology_candidate.initial_state();
        let topology_binding = topology_candidate.lm_trial_binding(&topology_state, &step);
        topology_candidate
            .landmarks
            .push(topology_candidate.landmarks[0].clone());
        assert_token_rejected_without_mutation(
            &mut topology_candidate,
            &topology_state,
            &step,
            LmTrialToken::with_bound_landmark_steps(token_steps.clone(), topology_binding),
        );

        let mut scalar_candidate = token_binding_window();
        let scalar_state = scalar_candidate.initial_state();
        let scalar_binding = scalar_candidate.lm_trial_binding(&scalar_state, &step);
        scalar_candidate.scalar_mode = ScalarMode::ExtendedF64;
        assert_token_rejected_without_mutation(
            &mut scalar_candidate,
            &scalar_state,
            &step,
            LmTrialToken::with_bound_landmark_steps(token_steps, scalar_binding),
        );
    }

    #[test]
    fn failed_landmark_increment_keeps_the_previous_trial_value() {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        let landmark = WindowLandmark {
            track_id: 7,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: Vec::new(),
        };
        let problem = test_window(vec![nav(0, 0.0)], vec![landmark]);
        let original = TrialLandmarkValue {
            direction,
            inverse_distance: 0.75,
        };

        let failed = problem.trial_landmark_value(0, Vector3::new(f64::NAN, 0.0, 0.0));
        assert!(failed.is_none());
        let materialized = failed.unwrap_or(original);
        assert_eq!(materialized, original);

        let state = problem.initial_state();
        let step = DVector::zeros(problem.state_dof());
        let trial = problem.apply_step(&state, &step);
        let view = WindowTrialView::from_step(&problem, &state, &step, &trial);
        let view_value = view.landmark_parameter(0).unwrap();
        assert_eq!(view_value.direction, original.direction);
        assert_eq!(view_value.inverse_distance, original.inverse_distance);

        let accepted = problem
            .trial_landmark_value(0, Vector3::new(0.015625, -0.03125, 0.125))
            .unwrap();
        assert_ne!(accepted, original);
        assert_eq!(problem.landmarks[0].direction, original.direction);
        assert_eq!(
            problem.landmarks[0].inverse_distance,
            original.inverse_distance
        );
    }

    #[test]
    fn diagnostic_trial_path_retains_the_owned_clone_fallback() {
        assert!(diagnostic_env_active_for_keys([std::ffi::OsString::from(
            "VISLOC_BASALT_DETAIL_ITERATIONS"
        )]));
        let problem = test_window(vec![nav(1, 0.0)], Vec::new());
        let before = problem.clone();
        let state = problem.initial_state();
        let mut step = DVector::zeros(NAV_STATE_DOF);
        step[0] = 0.5;
        let owned = problem.trial_problem_with_step_owned(&state, &step);

        assert_window_dynamic_eq(&problem, &before);
        assert!(!std::ptr::eq(&problem, &owned));
        assert_ne!(
            owned.states[0].nav.imu_to_world.translation.x.to_bits(),
            problem.states[0].nav.imu_to_world.translation.x.to_bits()
        );
    }

    #[test]
    fn malformed_trial_chart_is_rejected_before_view_construction() {
        let problem = test_window(vec![nav(1, 0.0)], Vec::new());
        let state = problem.initial_state();
        let step = DVector::zeros(problem.state_dof());
        let malformed = DVector::zeros(problem.state_dof() - 1);
        let result = problem.trial_cost(&state, &step, &malformed);
        assert!(matches!(result, Err(LmFailure::LinearSolve)));
    }

    #[test]
    #[ignore = "requires external pinned-native frame8 capture"]
    fn m11_frame8_prior_error_native_inputs_probe() {
        let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
        let inputs = std::fs::read_to_string(format!("{root}/prior_inputs.jsonl")).unwrap();
        let outputs = std::fs::read_to_string(format!("{root}/lm_eval.jsonl")).unwrap();
        let oracle: Vec<serde_json::Value> = outputs
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let parse = |value: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(value.as_str().unwrap(), 16).unwrap()) as f64
        };
        let mut count = 0;
        for line in inputs.lines() {
            let input: serde_json::Value = serde_json::from_str(line).unwrap();
            let iteration = input["iteration"].as_u64().unwrap() as usize;
            let rows = input["rows"].as_u64().unwrap() as usize;
            let cols = input["cols"].as_u64().unwrap() as usize;
            assert_eq!((rows, cols), (21, 21));
            assert_eq!(iteration, count);
            let j = DMatrix::from_iterator(
                rows,
                cols,
                input["J_column_major"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let b = DVector::from_iterator(rows, input["b"].as_array().unwrap().iter().map(&parse));
            let delta =
                DVector::from_iterator(cols, input["delta"].as_array().unwrap().iter().map(&parse));
            let actual =
                upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
            let expected = u32::from_str_radix(
                oracle[iteration]["after_update_marg_prior_error"]["f32_bits"]
                    .as_str()
                    .unwrap(),
                16,
            )
            .unwrap();
            println!(
                "M11_PRIOR_ERROR iter={iteration} native={expected:08x} rust={:08x} exact={}",
                actual.to_bits(),
                actual.to_bits() == expected
            );
            assert_eq!(
                actual.to_bits(),
                expected,
                "native prior cost iteration {iteration}"
            );
            for packet_gemv in [false, true] {
                let jf = j.map(|v| v as f32);
                let df = delta.map(|v| v as f32);
                let jd = if packet_gemv {
                    WindowProblem::prior_rhs_col_major_21_f32(&j, &DVector::zeros(21), &delta)
                        .map(|v| v as f32)
                } else {
                    &jf * &df
                };
                let mut products = [0.0_f32; 16];
                for k in 0..16 {
                    products[k] = (b[k] as f32 + 0.5_f32 * jd[k]) * jd[k];
                }
                let lanes: [f32; 8] = std::array::from_fn(|k| products[k] + products[k + 8]);
                let half: [f32; 4] = std::array::from_fn(|k| lanes[k] + lanes[k + 4]);
                let mut sum = (half[0] + half[2]) + (half[1] + half[3]);
                for k in 16..21 {
                    sum = (0.5_f32 * jd[k] + b[k] as f32).mul_add(jd[k], sum);
                }
                println!("M11_PRIOR_DOT_CANDIDATE iter={iteration} packet_gemv={packet_gemv} native={expected:08x} candidate={:08x} exact={}", sum.to_bits(), sum.to_bits() == expected);
            }
            count += 1;
        }
        assert_eq!(count, 8);
        // Every captured native output is a mandatory bitwise oracle.
    }

    #[test]
    #[ignore = "requires external pinned-native frame18 prior cost capture"]
    fn m11_frame18_prior33_error_same_inputs_probe() {
        let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
        let records = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(format!("{root}/{name}"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let inputs = records("prior_inputs.jsonl");
        let oracle = records("lm_eval.jsonl");
        assert_eq!(inputs.len(), 8);
        assert_eq!(oracle.len(), 8);
        let parse = |v: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
        };
        let mut exact = 0;
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(oracle[i]["iteration"].as_u64(), Some(i as u64));
            assert_eq!(input["rows"].as_u64(), Some(33));
            assert_eq!(input["cols"].as_u64(), Some(33));
            assert_eq!(input["is_sqrt"].as_bool(), Some(true));
            let j = DMatrix::from_iterator(
                33,
                33,
                input["J_column_major"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let b = DVector::from_iterator(33, input["b"].as_array().unwrap().iter().map(&parse));
            let delta =
                DVector::from_iterator(33, input["delta"].as_array().unwrap().iter().map(&parse));
            let actual =
                upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
            let expected =
                u32::from_str_radix(oracle[i]["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                    .unwrap();
            exact += usize::from(actual.to_bits() == expected);
            println!(
                "M11_PRIOR33_SAME_INPUT iter={i} native={expected:08x} rust={:08x} exact={}",
                actual.to_bits(),
                actual.to_bits() == expected
            );
        }
        // Diagnostic probe: report all cases before enforcing the native oracle.
        assert_eq!(exact, 8, "native prior33 same-input cost parity");
    }

    #[test]
    #[ignore = "requires external pinned-native frame24 prior cost capture"]
    fn m11_frame24_prior39_error_same_inputs_probe() {
        let root = std::env::var("M11_PRIOR_INPUT_ROOT").expect("capture root");
        let records = |name: &str| -> Vec<serde_json::Value> {
            std::fs::read_to_string(format!("{root}/{name}"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        };
        let inputs = records("prior_inputs.jsonl");
        let oracle = records("lm_eval.jsonl");
        assert_eq!(inputs.len(), 8);
        assert_eq!(oracle.len(), 8);
        let parse = |v: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
        };
        let mut exact = 0;
        for (i, input) in inputs.iter().enumerate() {
            assert_eq!(input["iteration"].as_u64(), Some(i as u64));
            assert_eq!(oracle[i]["iteration"].as_u64(), Some(i as u64));
            assert_eq!(input["rows"].as_u64(), Some(39));
            assert_eq!(input["cols"].as_u64(), Some(39));
            assert_eq!(input["is_sqrt"].as_bool(), Some(true));
            let j = DMatrix::from_iterator(
                39,
                39,
                input["J_column_major"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let b = DVector::from_iterator(39, input["b"].as_array().unwrap().iter().map(&parse));
            let delta =
                DVector::from_iterator(39, input["delta"].as_array().unwrap().iter().map(&parse));
            let actual =
                upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
            let expected =
                u32::from_str_radix(oracle[i]["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                    .unwrap();
            exact += usize::from(actual.to_bits() == expected);
            println!(
                "M11_PRIOR39_SAME_INPUT iter={i} native={expected:08x} rust={:08x} exact={}",
                actual.to_bits(),
                actual.to_bits() == expected
            );
        }
        // Diagnostic probe: report all cases before enforcing the native oracle.
        assert_eq!(exact, 8, "native prior39 same-input cost parity");
    }

    #[test]
    #[ignore = "requires external pinned-native frame32 prior cost capture"]
    fn m11_frame32_prior45_error_same_inputs_probe() {
        let root = std::env::var("M11_FRAME32_PRIOR_ROOT").expect("capture root");
        let events: Vec<serde_json::Value> =
            std::fs::read_to_string(format!("{root}/events.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
        let inputs: Vec<_> = events
            .iter()
            .filter(|event| event["kind"].as_str() == Some("prior_input"))
            .collect();
        let deltas: Vec<_> = events
            .iter()
            .filter(|event| {
                event["kind"].as_str() == Some("prior_stage")
                    && event["stage"].as_str() == Some("delta")
            })
            .collect();
        let oracle: Vec<_> = events
            .iter()
            .filter(|event| event["kind"].as_str() == Some("lm_eval"))
            .collect();
        assert_eq!(inputs.len(), 8);
        assert_eq!(deltas.len(), 8);
        assert_eq!(oracle.len(), 8);
        let parse = |v: &serde_json::Value| {
            f32::from_bits(u32::from_str_radix(v.as_str().unwrap(), 16).unwrap()) as f64
        };
        let mut exact = 0;
        // The native prior probe fires at the accepted state entering the next
        // iteration.  LM evaluation N therefore corresponds to prior input N+1
        // only when candidate N was accepted.  Candidate 4 was rejected and
        // candidate 7 has no following iteration in this bounded capture.
        for candidate_iteration in [0_u64, 1, 2, 3, 5, 6] {
            let input_iteration = candidate_iteration + 1;
            let input = inputs
                .iter()
                .find(|event| event["iteration"].as_u64() == Some(input_iteration))
                .unwrap();
            let delta_event = deltas
                .iter()
                .find(|event| event["iteration"].as_u64() == Some(input_iteration))
                .unwrap();
            let expected_event = oracle
                .iter()
                .find(|event| event["iteration"].as_u64() == Some(candidate_iteration))
                .unwrap();
            assert_eq!(input["rows"].as_u64(), Some(45));
            assert_eq!(input["cols"].as_u64(), Some(45));
            let j = DMatrix::from_iterator(
                45,
                45,
                input["jacobian_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let b = DVector::from_iterator(
                45,
                input["stored_rhs_bits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(&parse),
            );
            let delta = DVector::from_iterator(
                45,
                delta_event["bits"].as_array().unwrap().iter().map(&parse),
            );
            let actual =
                upstream_marginal_prior_error(&j, &b, &delta, ScalarMode::UpstreamF32) as f32;
            let expected = u32::from_str_radix(
                expected_event["after_prior"]["f32_bits"].as_str().unwrap(),
                16,
            )
            .unwrap();
            exact += usize::from(actual.to_bits() == expected);
            println!(
                "M11_PRIOR45_SAME_INPUT candidate_iter={candidate_iteration} input_iter={input_iteration} native={expected:08x} rust={:08x} exact={}",
                actual.to_bits(),
                actual.to_bits() == expected
            );
        }
        assert_eq!(exact, 6, "native prior45 accepted same-input cost parity");
    }

    #[test]
    #[ignore = "requires validated native frame38 prior capture"]
    fn m11_frame38_prior51_error_same_inputs_probe() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME38_PRIOR_ROOT").expect("frame38 capture root"),
        );
        let events: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame38_strict_capture_20260913/r2/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let parse = |value: &serde_json::Value| -> Vec<f64> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for candidate_iteration in [0_u64, 1, 2, 3, 4, 6] {
            let input_iteration = candidate_iteration + 1;
            let input = events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("prior_input")
                        && event["iteration"].as_u64() == Some(input_iteration)
                })
                .unwrap();
            let delta = events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("prior_stage")
                        && event["stage"].as_str() == Some("delta")
                        && event["iteration"].as_u64() == Some(input_iteration)
                })
                .unwrap();
            let oracle = events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("lm_eval")
                        && event["iteration"].as_u64() == Some(candidate_iteration)
                })
                .unwrap();
            let j = DMatrix::from_column_slice(51, 51, &parse(&input["jacobian_bits"]));
            let rhs = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let delta = DVector::from_vec(parse(&delta["bits"]));
            let actual = upstream_marginal_prior_error_51_candidate(&j, &rhs, &delta) as f32;
            let expected =
                u32::from_str_radix(oracle["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                    .unwrap();
            assert_eq!(
                actual.to_bits(),
                expected,
                "candidate iteration {candidate_iteration}"
            );
        }
    }

    #[test]
    #[ignore = "requires validated native frame45 prior capture"]
    fn m11_frame45_prior57_same_inputs_probe() {
        let root = std::path::PathBuf::from(
            std::env::var("M11_FRAME45_PRIOR_ROOT").expect("frame45 capture root"),
        );
        let events: Vec<serde_json::Value> = std::fs::read_to_string(
            root.join("m11_native_frame45_strict_capture_20260913/r1/events.jsonl"),
        )
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
        let parse = |value: &serde_json::Value| -> Vec<f64> {
            value
                .as_array()
                .unwrap()
                .iter()
                .map(|word| {
                    f64::from(f32::from_bits(
                        u32::from_str_radix(word.as_str().unwrap(), 16).unwrap(),
                    ))
                })
                .collect()
        };
        for candidate_iteration in 0..7_u64 {
            let iteration = candidate_iteration + 1;
            let input = events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("prior_input")
                        && event["iteration"].as_u64() == Some(iteration)
                })
                .unwrap();
            let stage = |name: &str| {
                events
                    .iter()
                    .find(|event| {
                        event["kind"].as_str() == Some("prior_stage")
                            && event["stage"].as_str() == Some(name)
                            && event["iteration"].as_u64() == Some(iteration)
                    })
                    .unwrap()
            };
            let oracle = events
                .iter()
                .find(|event| {
                    event["kind"].as_str() == Some("lm_eval")
                        && event["iteration"].as_u64() == Some(candidate_iteration)
                })
                .unwrap();
            let j = DMatrix::from_column_slice(57, 57, &parse(&input["jacobian_bits"]));
            let rhs = DVector::from_vec(parse(&input["stored_rhs_bits"]));
            let delta = DVector::from_vec(parse(&stage("delta")["bits"]));
            let adjusted_expected = parse(&stage("adjusted_rhs")["bits"]);
            let adjusted = WindowProblem::prior_rhs_col_major_57_f32(&j, &rhs, &delta);
            for lane in 0..57 {
                assert_eq!(
                    (adjusted[lane] as f32).to_bits(),
                    (adjusted_expected[lane] as f32).to_bits(),
                    "adjusted RHS input iteration {iteration} lane {lane}"
                );
            }
            let actual = upstream_marginal_prior_error_57_candidate(&j, &rhs, &delta) as f32;
            let expected =
                u32::from_str_radix(oracle["after_prior"]["f32_bits"].as_str().unwrap(), 16)
                    .unwrap();
            assert_eq!(
                actual.to_bits(),
                expected,
                "prior model candidate {candidate_iteration} from input {iteration}"
            );
        }
    }

    #[test]
    fn clean_trial_view_prior_cost_is_bit_exact_to_diagnostic_clone() {
        let mut problem = test_window(vec![nav(0, 0.0)], Vec::new());
        problem.prior = Some(WindowPrior {
            frame_ids: vec![0],
            block_kinds: vec![WindowBlockKind::State],
            jacobian: DMatrix::identity(NAV_STATE_DOF, NAV_STATE_DOF),
            rhs: DVector::from_iterator(
                NAV_STATE_DOF,
                (0..NAV_STATE_DOF).map(|index| (index as f64 - 4.0) * 0.125),
            ),
            fej_point: DVector::zeros(NAV_STATE_DOF),
        });
        let state = problem.initial_state();
        let mut step = DVector::zeros(NAV_STATE_DOF);
        step[0] = 0.25;
        step[3] = 0.03125;
        step[9] = -0.0625;
        let trial = problem.apply_step(&state, &step);

        let owned = problem.trial_problem_with_step_owned(&state, &step);
        let owned_cost = owned.owned_trial_objective(&trial).unwrap();
        let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
            .cost()
            .unwrap();
        assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
    }

    #[test]
    fn clean_trial_view_prior_state_index_skips_pose_prefix() {
        let mut problem = test_window(vec![nav(1, 0.0)], Vec::new());
        problem.poses.push(WindowPose {
            frame_id: 0,
            timestamp_ns: 1,
            pose: SE3::identity(),
            stored_current_pose: SE3::identity(),
            linearized_pose: SE3::identity(),
            linearized_delta: DVector::zeros(POSE_DOF),
            is_keyframe: true,
        });
        // The carried prior names the full state block after the pose-only
        // prefix.  Marking it FEJ-linearized makes the trial increment enter
        // the prior current point, exposing the absolute-vs-local index
        // conversion that differs when a pose prefix is present.
        problem.states[0].linearized = true;
        problem.prior = Some(WindowPrior {
            frame_ids: vec![1],
            block_kinds: vec![WindowBlockKind::State],
            jacobian: DMatrix::identity(NAV_STATE_DOF, NAV_STATE_DOF),
            rhs: DVector::from_iterator(
                NAV_STATE_DOF,
                (0..NAV_STATE_DOF).map(|index| (index as f64 - 2.0) * 0.25),
            ),
            fej_point: DVector::zeros(NAV_STATE_DOF),
        });
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[6] = 0.25;
        step[9] = -0.03125;
        let trial = problem.apply_step(&state, &step);

        let owned = problem.trial_problem_with_step_owned(&state, &step);
        let owned_cost = owned.owned_trial_objective(&trial).unwrap();
        let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
            .cost()
            .unwrap();
        assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
    }

    #[test]
    fn clean_trial_view_visual_factor_is_bit_exact_to_diagnostic_clone() {
        let direction = StereographicDirection::from_bearing(Vector3::new(0.1, -0.2, 1.0)).unwrap();
        let landmark = WindowLandmark {
            track_id: 11,
            anchor_state_index: 0,
            anchor_camera_id: 0,
            direction,
            inverse_distance: 0.75,
            observations: vec![WindowObservation {
                state_index: 1,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            }],
        };
        let problem = test_window(vec![nav(0, 0.0), nav(1, 0.0)], vec![landmark]);
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[20] = -0.03125;
        let trial = problem.apply_step(&state, &step);
        let owned = problem.trial_problem_with_step_owned(&state, &step);
        let owned_cost = owned.owned_trial_objective(&trial).unwrap();
        let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
            .cost()
            .unwrap();
        assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
    }

    #[test]
    fn trial_objective_rejects_missing_host_history_without_mutation() {
        let (mut problem, state, step, trial, _) = visual_trial_fixture();
        problem.trial_host_order.clear();
        let before = problem.clone();
        let owned = problem.trial_problem_with_step_owned(&state, &step);
        assert_eq!(
            owned.owned_trial_objective(&trial),
            Err(LmFailure::LinearSolve)
        );
        assert_eq!(
            WindowTrialView::from_step(&problem, &state, &step, &trial).cost(),
            Err(LmFailure::LinearSolve)
        );
        assert_window_dynamic_eq(&problem, &before);
        assert_eq!(problem.trial_host_order, before.trial_host_order);
    }

    #[test]
    fn clean_trial_view_imu_factor_is_bit_exact_to_diagnostic_clone() {
        let mut problem = test_window(vec![nav(0, 0.0), nav(1, 0.1)], Vec::new());
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.01;
        for axis in 0..9 {
            delta.covariance[(axis, axis)] = 1.0;
        }
        problem.imu_links.push(WindowImuLink {
            from_index: 0,
            to_index: 1,
            delta,
        });
        let state = problem.initial_state();
        let mut step = DVector::zeros(problem.state_dof());
        step[15] = 0.015625;
        step[27] = -0.03125;
        let trial = problem.apply_step(&state, &step);
        let owned = problem.trial_problem_with_step_owned(&state, &step);
        let owned_cost = owned.owned_trial_objective(&trial).unwrap();
        let view_cost = WindowTrialView::from_step(&problem, &state, &step, &trial)
            .cost()
            .unwrap();
        assert_eq!(view_cost.to_bits(), owned_cost.to_bits());
    }

    #[test]
    fn global_state_layout_is_fifteen_columns_per_state() {
        let states = vec![nav(0, 0.0), nav(1, 1.0), nav(2, 2.0)];
        let x = flatten_states(&states);
        assert_eq!(x.len(), 45);
        assert_eq!(x[15], 1.0);
        assert_eq!(x[30], 2.0);
    }

    #[test]
    fn upstream_f32_window_state_roundtrip_preserves_so3_components() {
        let rotation = UnitQuaternion::new_unchecked(Quaternion::new(
            0.5944822430610657_f64,
            -0.052778493613004684_f64,
            -0.8023747801780701_f64,
            0.0,
        ));
        let state = BasaltNavState {
            imu_to_world: SE3::new(rotation, Vector3::zeros()),
            ..BasaltNavState::default()
        };
        let encoded = flatten_nav_with_mode(&state, ScalarMode::UpstreamF32);
        let decoded = decode_state_with_mode(&encoded, ScalarMode::UpstreamF32);
        let before = state.imu_to_world.rotation.quaternion();
        let after = decoded.imu_to_world.rotation.quaternion();
        assert_eq!(after.w, before.w);
        assert_eq!(after.i, before.i);
        assert_eq!(after.j, before.j);
        assert_eq!(after.k, before.k);
    }

    #[test]
    fn mixed_pose_then_state_layout_is_six_plus_fifteen() {
        let states = vec![nav(7, 1.0), nav(8, 2.0)];
        let poses = vec![WindowPose {
            frame_id: 0,
            timestamp_ns: 1,
            pose: SE3::identity(),
            stored_current_pose: SE3::identity(),
            linearized_pose: SE3::identity(),
            linearized_delta: DVector::zeros(POSE_DOF),
            is_keyframe: true,
        }];
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses,
            states,
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        assert_eq!(problem.state_dof(), 36);
        let x = problem.initial_state();
        assert_eq!(x.len(), 36);
        assert_eq!(problem.state_offset(0), 6);
        assert_eq!(problem.state_offset(1), 21);
        assert_eq!(problem.block_frame_id(0), Some(0));
        assert_eq!(problem.block_frame_id(1), Some(7));
    }

    #[test]
    fn lm_pose_step_matches_upstream_additive_translation_left_rotation() {
        let mut base = nav(0, 2.0);
        base.nav.imu_to_world.rotation =
            UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.4));
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states: vec![base.clone()],
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        let mut step = DVector::zeros(NAV_STATE_DOF);
        step[0] = 0.3;
        step[5] = 0.2;
        let updated = problem.apply_step(&problem.initial_state(), &step);
        let decoded = decode_state(updated.rows(0, NAV_STATE_DOF));
        let expected_rotation = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, 0.2))
            * base.nav.imu_to_world.rotation;
        assert_eq!(
            decoded.imu_to_world.translation,
            base.nav.imu_to_world.translation + Vector3::new(0.3, 0.0, 0.0)
        );
        assert!((decoded.imu_to_world.rotation.inverse() * expected_rotation).angle() < 1e-12);
    }

    #[test]
    fn upstream_f32_pose_apply_reconstructs_from_linearized_delta() {
        // Pose-only keyframes use Basalt's PoseStateWithLin contract: the
        // accepted increments are accumulated in a binary32 tangent and the
        // candidate pose is rebuilt from the frozen linearization pose.  A
        // direct update of the current pose is observably different once the
        // second rotational increment is applied.
        let linearized_pose = SE3::new(
            UnitQuaternion::from_scaled_axis(Vector3::new(0.17, -0.23, 0.31)),
            Vector3::new(3.8816469, -2.5, 3.75),
        );
        let first_increment = DVector::from_column_slice(&[
            0.8826645_f64,
            -0.375_f64,
            0.625_f64,
            0.013_f64,
            -0.021_f64,
            0.034_f64,
        ]);
        let first_translation = vec3_f32(linearized_pose.translation)
            + Vector3::from_iterator(
                first_increment
                    .rows(0, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            );
        let first_rotation = sophus_so3_product(
            so3_exp_f32(Vector3::from_iterator(
                first_increment
                    .rows(3, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            )),
            quat_f32_from_f64(&linearized_pose.rotation),
        );
        let pose = WindowPose {
            frame_id: 0,
            timestamp_ns: 1,
            pose: SE3::new(
                quat_f64_from_f32(&first_rotation),
                vec3_f64(first_translation),
            ),
            stored_current_pose: SE3::new(
                quat_f64_from_f32(&first_rotation),
                vec3_f64(first_translation),
            ),
            linearized_pose: linearized_pose.clone(),
            linearized_delta: first_increment.clone(),
            is_keyframe: true,
        };
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: vec![pose.clone()],
            states: Vec::new(),
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
        };
        let second_increment = DVector::from_column_slice(&[
            0.8693182_f64,
            0.1875_f64,
            -0.3125_f64,
            -0.019_f64,
            0.027_f64,
            -0.011_f64,
        ]);
        let (updated, _) = problem.apply_step_full(&problem.initial_state(), &second_increment);
        let actual = &updated[0];

        let mut accumulated = first_increment;
        for (lhs, rhs) in accumulated.iter_mut().zip(second_increment.iter()) {
            *lhs = ((*lhs as f32) + (*rhs as f32)) as f64;
        }
        let expected_translation = vec3_f32(linearized_pose.translation)
            + Vector3::from_iterator(
                accumulated
                    .rows(0, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            );
        let expected_rotation = sophus_so3_product(
            so3_exp_f32(Vector3::from_iterator(
                accumulated
                    .rows(3, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            )),
            quat_f32_from_f64(&linearized_pose.rotation),
        );
        let actual_q = actual.rotation.quaternion();
        let expected_q = expected_rotation.quaternion();
        assert_eq!(
            actual.translation.map(|value| (value as f32).to_bits()),
            expected_translation.map(|value| value.to_bits())
        );
        assert_eq!(
            [actual_q.i, actual_q.j, actual_q.k, actual_q.w].map(|value| (value as f32).to_bits()),
            [expected_q.i, expected_q.j, expected_q.k, expected_q.w].map(|value| value.to_bits())
        );

        // This is the pre-fix behavior: apply only the new increment to the
        // already-updated pose.  Keep the assertion explicit so a future
        // simplification cannot silently lose the accumulated-delta rule.
        let naive_translation = vec3_f32(pose.pose.translation)
            + Vector3::from_iterator(
                second_increment
                    .rows(0, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            );
        let naive_rotation = sophus_so3_product(
            so3_exp_f32(Vector3::from_iterator(
                second_increment
                    .rows(3, 3)
                    .iter()
                    .copied()
                    .map(|value| value as f32),
            )),
            quat_f32_from_f64(&pose.pose.rotation),
        );
        let naive_q = naive_rotation.quaternion();
        assert_ne!(
            [actual_q.i, actual_q.j, actual_q.k, actual_q.w].map(|value| (value as f32).to_bits()),
            [naive_q.i, naive_q.j, naive_q.k, naive_q.w].map(|value| value.to_bits())
        );
        assert_ne!(
            actual.translation.map(|value| (value as f32).to_bits()),
            naive_translation.map(|value| value.to_bits())
        );
    }

    #[test]
    fn pose_only_so3_exp_matches_frame174_portable_oracle() {
        // MH_01 frame 174, iteration 0, pose-only block 112. Both targets
        // receive these exact FEJ-base and accumulated-tangent bits.
        let f = f32::from_bits;
        let before = UnitQuaternion::new_unchecked(Quaternion::new(
            f(0x3f16_aba4),
            f(0xbd94_e019),
            f(0xbf4d_7c08),
            f(0xbd82_0a27),
        ));
        let omega = Vector3::new(f(0xbba4_3bd6), f(0xbc41_ea40), f(0xb91b_19cc));
        let delta = so3_exp_f32(omega);
        assert_eq!(
            [delta.i, delta.j, delta.k, delta.w].map(f32::to_bits),
            [0xbb24_3b8c, 0xbbc1_e9e9, 0xb89b_1986, 0x3f7f_fea6]
        );
        let updated = sophus_so3_product(delta, before);
        assert_eq!(
            [updated.i, updated.j, updated.k, updated.w].map(f32::to_bits),
            [0xbd97_3eb7, 0xbf4e_6948, 0xbd7d_c672, 0x3f15_674c]
        );
    }

    #[test]
    fn upstream_f32_state_update_quaternion_matches_native_sophus_bits() {
        let cases = [
            (
                [0xbd5cdea6, 0xbf4d341f, 0xbb2aa78b, 0x3f186f67],
                [0xb92b55aa, 0xb9418699, 0x38359f2f],
                [0xb8ab55aa, 0xb8c18699, 0x37b59f2f, 0x3f800000],
                [0xbd5cff35, 0xbf4d37d1, 0xbb25d808, 0x3f186a45],
            ),
            (
                [0xbd5cf5f2, 0xbf4d97c0, 0xbbac4997, 0x3f17e7a6],
                [0xb84253bc, 0xb99a78ae, 0x387d38e4],
                [0xb7c253bc, 0xb91a78ae, 0x37fd38e4, 0x3f800000],
                [0xbd5cea20, 0xbf4d9d98, 0xbbab59ef, 0x3f17dfd3],
            ),
            (
                [0xbd5e760f, 0xbf4e5e85, 0xbbc2d95f, 0x3f16d68a],
                [0x38781aee, 0xb9dc2926, 0x38c40d56],
                [0x37f81aee, 0xb95c2926, 0x38440d56, 0x3f800000],
                [0xbd5e3af5, 0xbf4e66c7, 0xbbc319ff, 0x3f16cb91],
            ),
            (
                [0xbd5f79e6, 0xbf4f45a2, 0xbbb53f68, 0x3f159719],
                [0x3910e3ec, 0xba0d5f2c, 0x3903baf9],
                [0x3890e3ec, 0xb98d5f2c, 0x3883baf9, 0x3f7fffff],
                [0xbd5f18ac, 0xbf4f5028, 0xbbb65c29, 0x3f15890f],
            ),
        ];
        for (q_bits, omega_bits, exp_bits, result_bits) in cases {
            let q = UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(q_bits[3]),
                f32::from_bits(q_bits[0]),
                f32::from_bits(q_bits[1]),
                f32::from_bits(q_bits[2]),
            ));
            let omega = Vector3::new(
                f32::from_bits(omega_bits[0]),
                f32::from_bits(omega_bits[1]),
                f32::from_bits(omega_bits[2]),
            );
            let delta = so3_exp_f32(omega);
            let result = sophus_so3_product(delta, q);
            let bits = |value: &UnitQuaternion<f32>| {
                [
                    value.i.to_bits(),
                    value.j.to_bits(),
                    value.k.to_bits(),
                    value.w.to_bits(),
                ]
            };
            assert_eq!(bits(&delta), exp_bits);
            assert_eq!(bits(&result), result_bits);
        }
    }

    #[test]
    fn upstream_f32_frame3_iter0_apply_step_matches_native_accepted_state() {
        let f64_from_bits = |bits: u32| f32::from_bits(bits) as f64;
        let pre = BasaltNavState {
            imu_to_world: SE3::new(
                UnitQuaternion::new_unchecked(Quaternion::new(
                    f64_from_bits(0x3f16d68a),
                    f64_from_bits(0xbd5e760f),
                    f64_from_bits(0xbf4e5e85),
                    f64_from_bits(0xbbc2d95f),
                )),
                Vector3::new(
                    f64_from_bits(0x3b8ebfa0),
                    f64_from_bits(0xba9428f4),
                    f64_from_bits(0xbc857642),
                ),
            ),
            velocity_world_m_s: Vector3::new(
                f64_from_bits(0x3d850448),
                f64_from_bits(0xbc998d12),
                f64_from_bits(0xbe4a560c),
            ),
            ..BasaltNavState::default()
        };
        let increment_bits = [
            0xbae43978, 0xbb07ab82, 0xbd09c073, 0x38781648, 0xb9dc3846, 0x38c40d43, 0xbc475478,
            0xbc60489e, 0xbe67ad09, 0x38e89ae4, 0x3ab2b6ec, 0x3ab509db, 0x3a29b4c8, 0xb8b7e658,
            0xb9d1c9ee,
        ];
        let expected_bits = [
            0x3b2b6284, 0xbb51bffc, 0xbd4c7b94, 0xbd5e3af5, 0xbf4e66c7, 0xbbc31a02, 0x3f16cb91,
            0x3d583372, 0xbd04d8b0, 0xbed9018a, 0x38e89ae4, 0x3ab2b6ec, 0x3ab509db, 0x3a29b4c8,
            0xb8b7e658, 0xb9d1c9ee,
        ];
        let state = WindowState {
            frame_id: 3,
            timestamp_ns: 1,
            nav: pre.clone(),
            stored_current_nav: pre.clone(),
            linearized_nav: pre,
            linearized_delta: DVector::zeros(NAV_STATE_DOF),
            is_keyframe: false,
            is_latest: true,
            linearized: false,
        };
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states: vec![state],
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
        };
        let step =
            DVector::from_iterator(NAV_STATE_DOF, increment_bits.into_iter().map(f64_from_bits));
        let (_, updated_states) = problem.apply_step_full(&problem.initial_state(), &step);
        let updated = &updated_states[0];
        let q = updated.imu_to_world.rotation.quaternion();
        let mut actual = Vec::with_capacity(16);
        actual.extend(
            updated
                .imu_to_world
                .translation
                .iter()
                .map(|x| (*x as f32).to_bits()),
        );
        actual.extend([q.i, q.j, q.k, q.w].map(|x| (x as f32).to_bits()));
        actual.extend(
            updated
                .velocity_world_m_s
                .iter()
                .map(|x| (*x as f32).to_bits()),
        );
        actual.extend(
            updated
                .gyro_bias_rad_s
                .iter()
                .map(|x| (*x as f32).to_bits()),
        );
        actual.extend(
            updated
                .accel_bias_m_s2
                .iter()
                .map(|x| (*x as f32).to_bits()),
        );
        assert_eq!(actual, expected_bits);
    }

    #[test]
    fn upstream_f32_current_pose_zero_inc_normalizes_accepted_state_bits() {
        // `BundleAdjustmentBase::getPoseStateWithLin` materializes a
        // non-linearized navigation state as a temporary PoseStateWithLin.
        // Its constructor calls PoseState::incPose with a zero tangent; the
        // Sophus product still crosses the float quaternion normalization
        // boundary.  The accepted compact state below is authoritative
        // frame 3 / frame-4 iter0 from the native GDB capture.
        let accepted = UnitQuaternion::new_unchecked(Quaternion::new(
            f32::from_bits(0x3f16cb91),
            f32::from_bits(0xbd5e3af5),
            f32::from_bits(0xbf4e66c7),
            f32::from_bits(0xbbc31a02),
        ));
        let current = sophus_so3_product(so3_exp_f32(Vector3::zeros()), accepted);
        let q = current.quaternion();
        assert_eq!(
            [q.i, q.j, q.k, q.w].map(f32::to_bits),
            [0xbd5e3af6, 0xbf4e66c8, 0xbbc31a03, 0x3f16cb92]
        );
    }

    #[test]
    fn m11_frame34_boundary_state_history_pose_matches_native_zero_inc_bits() {
        // Native frame34 LM writeback and Rust agree on this raw frame33
        // state.  At the following frame35 measure(), frame33 has become the
        // FEJ boundary and getPoseStateWithLin() applies its zero delta before
        // delayed triangulation.  The resulting native operand was captured
        // at libbasalt 0x4e4960 for track1821.
        let raw_pose = SE3::new(
            UnitQuaternion::new_unchecked(Quaternion::new(
                f32::from_bits(0x3f1b6492) as f64,
                f32::from_bits(0xbd52b57d) as f64,
                f32::from_bits(0xbf4ad7a3) as f64,
                f32::from_bits(0xbd06774d) as f64,
            )),
            Vector3::new(
                f32::from_bits(0xbc8eaa38) as f64,
                f32::from_bits(0xbc2aec38) as f64,
                f32::from_bits(0x3e1236c5) as f64,
            ),
        );
        let state = WindowState {
            frame_id: 33,
            timestamp_ns: 1403636581413555456,
            nav: BasaltNavState {
                imu_to_world: raw_pose.clone(),
                ..BasaltNavState::default()
            },
            stored_current_nav: BasaltNavState {
                imu_to_world: raw_pose.clone(),
                ..BasaltNavState::default()
            },
            linearized_nav: BasaltNavState {
                imu_to_world: raw_pose,
                ..BasaltNavState::default()
            },
            linearized_delta: DVector::zeros(NAV_STATE_DOF),
            is_keyframe: true,
            is_latest: false,
            linearized: true,
        };
        let effective = visual_state_pose_with_mode(&state, ScalarMode::UpstreamF32);
        let q = effective.rotation.quaternion();
        assert_eq!(
            [q.i, q.j, q.k, q.w].map(|value| (value as f32).to_bits()),
            [0xbd52b57b, 0xbf4ad7a1, 0xbd06774c, 0x3f1b6491]
        );
        assert_eq!(
            effective.translation.map(|value| (value as f32).to_bits()),
            Vector3::new(0xbc8eaa38, 0xbc2aec38, 0x3e1236c5)
        );
    }

    #[test]
    fn upstream_f32_visual_get_pose_reconstructs_native_linearized_current() {
        // The pinned native `getPoseStateWithLin()` constructor always
        // applies the stored delta to `pose_linearized`, including a zero
        // delta.  These two cases are copied from
        // `m7im15_gdb_frame6_getpose_frame4_20260828.json`: the first is the
        // frame-6/iter-0 frame-4 state and the second is a later nonzero
        // frame-4 delta.  This test exercises the actual visual accessor,
        // rather than only testing the quaternion helper in isolation.
        let pose_from_bits = |translation: [u32; 3], q_xyzw: [u32; 4]| {
            SE3::new(
                UnitQuaternion::new_unchecked(Quaternion::new(
                    f32::from_bits(q_xyzw[3]) as f64,
                    f32::from_bits(q_xyzw[0]) as f64,
                    f32::from_bits(q_xyzw[1]) as f64,
                    f32::from_bits(q_xyzw[2]) as f64,
                )),
                Vector3::from_iterator(
                    translation
                        .into_iter()
                        .map(|bits| f32::from_bits(bits) as f64),
                ),
            )
        };
        let make_problem = |linearized_pose: SE3, linearized: bool, delta_bits: [u32; 6]| {
            let nav = BasaltNavState {
                imu_to_world: linearized_pose.clone(),
                ..BasaltNavState::default()
            };
            let mut delta = DVector::zeros(NAV_STATE_DOF);
            for (slot, bits) in delta_bits.into_iter().enumerate() {
                delta[slot] = f32::from_bits(bits) as f64;
            }
            WindowProblem {
                trial_host_order: Vec::new(),
                camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                    .unwrap(),
                cameras: Vec::new(),
                t_imu_cam: Vec::new(),
                poses: Vec::new(),
                states: vec![WindowState {
                    frame_id: 4,
                    timestamp_ns: 1403636579963555584,
                    stored_current_nav: nav.clone(),
                    nav,
                    linearized_nav: BasaltNavState {
                        imu_to_world: linearized_pose,
                        ..BasaltNavState::default()
                    },
                    linearized_delta: delta,
                    is_keyframe: false,
                    is_latest: true,
                    linearized,
                }],
                landmarks: Vec::new(),
                imu_links: Vec::new(),
                imu_noise: ImuNoiseModel {
                    gyro_density: 1.0,
                    accel_density: 1.0,
                },
                bias_walk_noise: BiasRandomWalkNoise {
                    gyro_density: 1.0,
                    accel_density: 1.0,
                },
                initial_pose_weight: 1.0,
                initial_accel_bias_weight: 1.0,
                initial_gyro_bias_weight: 1.0,
                prior: None,
                anchor_point: None,
                gravity_world: Vector3::zeros(),
                scalar_mode: ScalarMode::UpstreamF32,
            }
        };
        let raw_pose = pose_from_bits(
            [0xbb21be47, 0xbb8635a5, 0xbdb353f2],
            [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c],
        );

        let zero_problem = make_problem(raw_pose.clone(), true, [0; 6]);
        let zero_current = zero_problem
            .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
            .unwrap();
        let zero_q = zero_current.imu_to_world.rotation.quaternion();
        assert_eq!(
            zero_current
                .imu_to_world
                .translation
                .map(|value| (value as f32).to_bits()),
            Vector3::new(0xbb21be47_u32, 0xbb8635a5_u32, 0xbdb353f2_u32)
        );
        assert_eq!(
            [zero_q.i, zero_q.j, zero_q.k, zero_q.w].map(|value| (value as f32).to_bits()),
            [0xbd5d86c2, 0xbf511115, 0xbbd647b9, 0x3f13148b]
        );

        let nonzero_problem = make_problem(
            raw_pose,
            true,
            [
                0x37bd51ec, 0xb49be800, 0x39cee3c6, 0x3a42b20c, 0xb98b69f0, 0x37b3f87b,
            ],
        );
        let nonzero_current = nonzero_problem
            .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
            .unwrap();
        let nonzero_q = nonzero_current.imu_to_world.rotation.quaternion();
        assert_eq!(
            nonzero_current
                .imu_to_world
                .translation
                .map(|value| (value as f32).to_bits()),
            Vector3::new(0xbb2043a3_u32, 0xbb863815_u32, 0xbdb2850e_u32)
        );
        assert_eq!(
            [nonzero_q.i, nonzero_q.j, nonzero_q.k, nonzero_q.w]
                .map(|value| (value as f32).to_bits()),
            [0xbd5c9cf2, 0xbf5115f9, 0xbbe0405f, 0x3f130ebf]
        );

        // The non-linearized branch is deliberately still the raw frozen
        // pose, even if a stale delta happens to be present in a fixture.
        let false_problem = make_problem(
            pose_from_bits(
                [0xbb21be47, 0xbb8635a5, 0xbdb353f2],
                [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c],
            ),
            false,
            [
                0x37bd51ec, 0xb49be800, 0x39cee3c6, 0x3a42b20c, 0xb98b69f0, 0x37b3f87b,
            ],
        );
        let false_current = false_problem
            .block_nav_visual_current(&DVector::zeros(NAV_STATE_DOF), 0)
            .unwrap();
        let false_q = false_current.imu_to_world.rotation.quaternion();
        assert_eq!(
            [false_q.i, false_q.j, false_q.k, false_q.w].map(|value| (value as f32).to_bits()),
            [0xbd5d86c4, 0xbf511117, 0xbbd647bb, 0x3f13148c]
        );
    }

    #[test]
    fn fej_local_boxminus_is_zero_and_first_order_consistent() {
        let reference = BasaltNavState {
            imu_to_world: SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(0.1, -0.2, 0.3)),
                Vector3::new(1.0, -2.0, 0.5),
            ),
            velocity_world_m_s: Vector3::new(0.2, 0.3, -0.1),
            ..BasaltNavState::default()
        };
        let zero = local_state_difference(&reference, &reference);
        assert!(zero.norm() < 1e-12);
        let increment = Vector6::new(1e-4, -2e-4, 3e-4, 2e-5, -3e-5, 4e-5);
        let mut moved = reference.clone();
        moved.imu_to_world.translation += Vector3::new(increment[0], increment[1], increment[2]);
        moved.imu_to_world.rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
            increment[3],
            increment[4],
            increment[5],
        )) * moved.imu_to_world.rotation;
        let recovered = local_state_difference(&moved, &reference);
        assert!((recovered.rows(0, 6) - increment).norm() < 1e-10);
    }

    #[test]
    fn marginal_prior_error_drops_constant_and_allows_negative_values() {
        let jacobian = DMatrix::from_diagonal(&DVector::from_column_slice(&[2.0, 3.0]));
        let rhs = DVector::from_column_slice(&[-10.0, 1.0]);
        let delta = DVector::from_column_slice(&[1.0, 0.0]);

        // J*delta + r has a positive complete squared cost, while Basalt's
        // linearized error intentionally drops 0.5*rᵀr and is negative.
        let complete = 0.5
            * (&jacobian * &delta + &rhs)
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
        assert_eq!(complete, 32.5);
        assert_eq!(
            upstream_marginal_prior_error(&jacobian, &rhs, &delta, ScalarMode::ExtendedF64,),
            -18.0
        );
        assert_eq!(
            upstream_marginal_prior_error(&jacobian, &rhs, &delta, ScalarMode::UpstreamF32,)
                .to_bits(),
            (-18.0_f32 as f64).to_bits()
        );
    }

    #[test]
    fn upstream_sqrt_marginalization_uses_marg_first_order_and_sqrt_eps_cutoff() {
        let jacobian = DMatrix::from_row_slice(
            4,
            3,
            &[
                1.0, 0.2, 2.0, -0.5, 1.5, 0.3, 0.7, -0.4, 1.2, 0.1, 0.8, -0.9,
            ],
        );
        let rhs = DVector::from_column_slice(&[0.4, -1.2, 0.7, 2.1]);
        // Deliberately non-sorted input exercises the same set-order contract
        // as upstream's std::set indices after the caller has built AOM lists.
        let prior = sqrt_marginalize_upstream(
            &jacobian,
            &rhs,
            &[2],
            &[1, 0],
            DVector::from_element(1, 3.0),
        )
        .expect("full-rank synthetic prior");
        assert_eq!(prior.jacobian.ncols(), 1);
        assert_eq!(prior.jacobian.nrows(), 1);
        assert_eq!(prior.fej_point, DVector::from_element(1, 3.0));

        // Compare the emitted square-root prior's normal terms against the
        // Schur complement of the same un-squared rows. The sign of a
        // Householder row is intentionally free, so compare Gram products.
        let h = jacobian.transpose() * &jacobian;
        let b = jacobian.transpose() * rhs;
        let hmm = h.view((0, 0), (2, 2)).into_owned();
        let hmk = h.view((0, 2), (2, 1)).into_owned();
        let hkm = h.view((2, 0), (1, 2)).into_owned();
        let hkk = h[(2, 2)];
        let bm = b.rows(0, 2).into_owned();
        let bk = b[2];
        let schur_h = hkk - (hkm.clone() * hmm.clone().try_inverse().unwrap() * hmk)[(0, 0)];
        let schur_b = bk - (hkm * hmm.try_inverse().unwrap() * bm)[0];
        assert!((prior.jacobian[(0, 0)].powi(2) - schur_h).abs() < 1e-10);
        assert!((prior.jacobian[(0, 0)] * prior.rhs[0] - schur_b).abs() < 1e-10);

        let tiny_marginal = DMatrix::from_column_slice(3, 1, &[0.5e-8, 0.0, 0.0]);
        let tiny_keep = DMatrix::from_column_slice(3, 1, &[0.0, 2.0, 3.0]);
        let mut stacked = DMatrix::zeros(3, 2);
        stacked.column_mut(0).copy_from(&tiny_marginal.column(0));
        stacked.column_mut(1).copy_from(&tiny_keep.column(0));
        let tiny =
            sqrt_marginalize_upstream(&stacked, &DVector::zeros(3), &[1], &[0], DVector::zeros(1))
                .expect("rank-deficient marginal block is retained as an explicit zero row");
        assert_eq!(tiny.jacobian.nrows(), 1);
        assert!((tiny.jacobian[(0, 0)].abs() - tiny_keep.column(0).norm()).abs() < 1e-12);
    }

    fn assert_f32_golden(actual: f64, expected_bits: u32) {
        let actual_bits = (actual as f32).to_bits();
        assert_eq!(
            actual_bits, expected_bits,
            "actual={actual:.9e} bits=0x{actual_bits:08x}, expected=0x{expected_bits:08x}"
        );
    }

    #[test]
    fn upstream_f32_sqrt_marginalization_matches_eigen_golden_boundaries() {
        // These fixtures are emitted by
        // benchmarks/basalt/upstream_sqrt_marginalization_oracle.cpp against
        // the pinned Basalt Eigen 5.0.1 checkout.  Compare float bit patterns,
        // not widened decimal text, so a change in reflector storage, sign,
        // accumulation precision, or rank cutoff is visible.  The expected
        // values are bound to the pinned clean native O3/AVX build
        // (GCC 11.4, -O3 -march=native, Eigen 5.0.1); the older default
        // SSE2/Packet4 fixture is retained as historical evidence only.
        let cases = [
            (
                "normal",
                DMatrix::from_row_slice(
                    5,
                    4,
                    &[
                        0.75, -1.2, 0.25, -0.7, -0.35, 0.8, 1.3, 0.2, 1.1, 0.15, -0.6, 0.9, -0.45,
                        -0.55, 0.7, -1.4, 0.2, 1.05, -0.9, 0.35,
                    ],
                ),
                DVector::from_column_slice(&[0.4, -1.2, 0.7, 2.1, -0.3]),
                vec![0, 3],
                vec![1, 2],
                (2, 2),
                vec![3214139916, 3208334584, 0, 1064423251],
                vec![1048055211, 3215223833],
            ),
            (
                "near_rank",
                DMatrix::from_row_slice(
                    4,
                    3,
                    &[
                        1.0e-4, 0.4, -0.7, 2.0e-4, -0.2, 0.9, -1.0e-4, 0.8, 0.1, 0.0, -0.5, 0.3,
                    ],
                ),
                DVector::from_column_slice(&[0.25, -0.5, 0.75, -1.0]),
                vec![0, 2],
                vec![1],
                (1, 2),
                vec![0, 3213413843],
                vec![1045480110],
            ),
            (
                "sign",
                DMatrix::from_row_slice(4, 2, &[-0.8, 0.3, 0.25, -1.1, -0.45, 0.6, 0.7, 0.2]),
                DVector::from_column_slice(&[0.6, -0.4, 1.2, 0.1]),
                vec![1],
                vec![0],
                (1, 1),
                vec![1066896430],
                vec![1060968209],
            ),
            (
                "zero",
                DMatrix::from_row_slice(3, 2, &[0.0, 0.5, 0.0, -1.0, 0.0, 0.25]),
                DVector::from_column_slice(&[0.2, -0.3, 0.4]),
                vec![1],
                vec![0],
                (1, 1),
                vec![3214058614],
                vec![3202315395],
            ),
            (
                "denormal",
                DMatrix::from_row_slice(
                    3,
                    2,
                    &[f64::from(f32::from_bits(1)), 0.5, 0.0, -1.0, 0.0, 0.25],
                ),
                DVector::from_column_slice(&[0.2, -0.3, 0.4]),
                vec![1],
                vec![0],
                (1, 1),
                vec![3214058614],
                vec![3202315395],
            ),
        ];

        for (
            name,
            jacobian,
            rhs,
            keep,
            marg,
            (expected_rows, expected_cols),
            expected_j,
            expected_rhs,
        ) in cases
        {
            let prior = sqrt_marginalize_upstream_f32(
                &jacobian,
                &rhs,
                &keep,
                &marg,
                DVector::zeros(keep.len()),
            )
            .unwrap_or_else(|| panic!("{name} fixture unexpectedly rejected"));
            assert_eq!(
                (prior.jacobian.nrows(), prior.jacobian.ncols()),
                (expected_rows, expected_cols),
                "{name} dimensions"
            );
            assert_eq!(prior.rhs.len(), expected_rhs.len(), "{name} rhs dimensions");
            for row in 0..expected_rows {
                for column in 0..expected_cols {
                    assert_f32_golden(
                        prior.jacobian[(row, column)],
                        expected_j[row * expected_cols + column],
                    );
                }
            }
            for (actual, expected) in prior.rhs.iter().zip(expected_rhs) {
                assert_f32_golden(*actual, expected);
            }
        }
    }

    #[test]
    fn upstream_f32_packet8_squared_norm_matches_m7im15_k12_eigen_bits() {
        // Q2 column 12 at the exact 96x60 M7IM15 boundary.  The production
        // Householder call norms the 83-element tail (excluding c0), while
        // the native diagnostic norms the full 84-element column segment.
        let tail_bits: [u32; 83] = [
            0xc3667f7b, 0xc3ab921f, 0xc3878255, 0xc3aaf1f2, 0x4295b589, 0x42a1a78d, 0x41f80a43,
            0x42a054b4, 0x4291e512, 0x428a9698, 0x00000000, 0x00000000, 0x00000000, 0xc30220a8,
            0xc3150c1c, 0x43b06b4d, 0xc3647953, 0x43b0f9bf, 0x41d51baa, 0xc31fd311, 0x00000000,
            0x00000000, 0x00000000, 0xc29dce23, 0xc0fa82ee, 0xc33a4aef, 0xc3462aef, 0xc191cd1f,
            0xc3a20a37, 0xc4aa2e5e, 0x466b58f6, 0x4594b741, 0x43bb27b6, 0x439bd043, 0x432bda22,
            0xc2b7f5db, 0xc2358289, 0xc2b4c9d9, 0x42eb8178, 0x40faa3ee, 0x412a6e92, 0x426e3b4e,
            0x40aefdd3, 0x3e5605f7, 0x44b480e6, 0xc66a71e4, 0xc59a0497, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
            0x3cd8d2c8, 0x3cb4836f, 0x3c47182d, 0xbe529c20, 0xbdcfcdec, 0xbe4efa7d,
        ];
        let tail = tail_bits
            .iter()
            .copied()
            .map(f32::from_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            upstream_f32_packet8_squared_norm(&tail, 0, tail.len()).to_bits(),
            0x4df06330
        );

        let mut full = Vec::with_capacity(tail.len() + 1);
        full.push(f32::from_bits(0x42c19836));
        full.extend_from_slice(&tail);
        assert_eq!(
            upstream_f32_packet8_squared_norm(&full, 0, full.len()).to_bits(),
            0x4df06454
        );
        // Keep this comparison explicitly non-contracted; it documents why
        // the source-level scalar expression is not the Eigen evaluator's
        // packet/FMA result on this boundary.
        let c0_sq = std::hint::black_box(full[0] * full[0]);
        let tail_sum = tail
            .iter()
            .fold(std::hint::black_box(0.0_f32), |sum, value| {
                let square = std::hint::black_box(*value * *value);
                std::hint::black_box(sum + square)
            });
        assert_eq!(std::hint::black_box(c0_sq + tail_sum).to_bits(), 0x4df06456);
    }

    #[test]
    fn upstream_f32_packet8_squared_norm_matches_m7im15_all_reflector_tails() {
        // Each fixture row is the exact pre-reflector essential tail from the
        // 96x60 M7IM15 Q2 replay, followed by Eigen's tail.squaredNorm() bits.
        // Keeping all 60 rows catches packet-unroll and scalar-remainder
        // changes that happen to be invisible on the k12 tail alone.
        let mut lines =
            include_str!("../../../../benchmarks/basalt/m7im15_sqrt_norm_fixture_v1.txt").lines();
        assert_eq!(lines.next(), Some("M7IM15_SQRT_NORM_V1"));
        let dimensions = lines
            .next()
            .expect("M7IM15 norm fixture dimensions")
            .split_whitespace()
            .map(|token| token.parse::<usize>().expect("fixture dimension"))
            .collect::<Vec<_>>();
        assert_eq!(dimensions, vec![60, 96, 60]);

        let mut count = 0usize;
        for line in lines {
            let mut tokens = line.split_whitespace();
            let column = tokens
                .next()
                .expect("fixture column")
                .parse::<usize>()
                .expect("fixture column number");
            let tail_len = tokens
                .next()
                .expect("fixture tail length")
                .parse::<usize>()
                .expect("fixture tail length number");
            let expected = u32::from_str_radix(tokens.next().expect("fixture expected norm"), 16)
                .expect("fixture expected norm bits");
            let tail = tokens
                .map(|token| {
                    f32::from_bits(u32::from_str_radix(token, 16).expect("fixture tail bits"))
                })
                .collect::<Vec<_>>();
            assert_eq!(tail.len(), tail_len, "M7IM15 k{column} tail length");
            assert_eq!(
                upstream_f32_packet8_squared_norm(&tail, 0, tail_len).to_bits(),
                expected,
                "M7IM15 k{column} squared norm"
            );
            count += 1;
        }
        assert_eq!(count, dimensions[0]);
    }

    #[test]
    fn upstream_f32_real_m6_packet_slice_matches_default_eigen() {
        // This is the complete real 69-row, 72-column
        // deterministic slice of the actual MH_01 frame-51 packet
        // (`m6_actual_packet_oracle_v1`, Rust `frame_000051.json`).  The
        // selected rows preserve all 15 leaving columns and reach the same
        // 54-row retained rank as the standalone pinned native O3 Eigen 5.0.1
        // oracle (GCC 11.4, -march=native, AVX Packet8 on the capture host).
        // The production `UpstreamF32` path is a safe packet-order emulation,
        // so this test compares strict f32 bits rather than decimal values.
        // The pinned C++ MargData packet contains only abs_H/abs_b (not sqrt
        // J/r); this fixture is the authoritative standalone SqrtToSqrt
        // boundary for the real packet's AOM rows.
        let mut tokens =
            include_str!("../../../../benchmarks/basalt/m6_sqrt_boundary_frame51_v1.txt")
                .split_whitespace();
        assert_eq!(tokens.next(), Some("M6_FIXTURE_V1"));
        let rows = tokens.next().unwrap().parse::<usize>().unwrap();
        let cols = tokens.next().unwrap().parse::<usize>().unwrap();
        let marg_count = tokens.next().unwrap().parse::<usize>().unwrap();
        let marg = (0..marg_count)
            .map(|_| tokens.next().unwrap().parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let keep_count = tokens.next().unwrap().parse::<usize>().unwrap();
        let keep = (0..keep_count)
            .map(|_| tokens.next().unwrap().parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let selected_count = tokens.next().unwrap().parse::<usize>().unwrap();
        for _ in 0..selected_count {
            let _ = tokens.next().unwrap().parse::<usize>().unwrap();
        }
        let jacobian_values = (0..rows * cols)
            .map(|_| tokens.next().unwrap().parse::<f64>().unwrap())
            .collect::<Vec<_>>();
        let jacobian = DMatrix::from_row_slice(rows, cols, &jacobian_values);
        let rhs = DVector::from_fn(rows, |_, _| tokens.next().unwrap().parse::<f64>().unwrap());
        assert!(tokens.next().is_none(), "fixture has trailing tokens");

        let prior = sqrt_marginalize_upstream_f32(
            &jacobian,
            &rhs,
            &keep,
            &marg,
            DVector::zeros(keep.len()),
        )
        .expect("actual frame-51 slice must produce a retained prior");
        assert_eq!((prior.jacobian.nrows(), prior.jacobian.ncols()), (54, 57));
        assert_eq!(prior.rhs.len(), 54);

        let expected_j = [
            ((0, 0), 1_176_616_427),
            ((0, 1), 1_127_794_637),
            ((0, 56), 3_212_872_857),
            ((1, 0), 0),
            ((27, 28), 3_310_529_033),
            ((53, 56), 3_242_393_197),
        ];
        for &((row, column), bits) in &expected_j {
            assert_f32_golden(prior.jacobian[(row, column)], bits);
        }
        let expected_rhs = [(0, 3_215_659_848), (27, 1_080_990_045), (53, 3_217_203_652)];
        for &(row, bits) in &expected_rhs {
            assert_f32_golden(prior.rhs[row], bits);
        }

        // Hash every retained raw f32 bit, in row-major J order followed by
        // rhs order.  This keeps the full 3132-value gate compact while the
        // selected probes above make a first mismatch easy to diagnose.
        let mut digest = 0xcbf29ce484222325_u64;
        let mut value_count = 0usize;
        for row in 0..prior.jacobian.nrows() {
            for column in 0..prior.jacobian.ncols() {
                for byte in (prior.jacobian[(row, column)] as f32)
                    .to_bits()
                    .to_le_bytes()
                {
                    digest ^= u64::from(byte);
                    digest = digest.wrapping_mul(0x100000001b3);
                }
                value_count += 1;
            }
        }
        for value in prior.rhs.iter() {
            for byte in (*value as f32).to_bits().to_le_bytes() {
                digest ^= u64::from(byte);
                digest = digest.wrapping_mul(0x100000001b3);
            }
            value_count += 1;
        }
        assert_eq!(value_count, 3132);
        assert_eq!(digest, 0xe4fa3a57cf2f00a1);
    }

    #[test]
    fn imu_residual_has_cross_state_jacobian_blocks() {
        let states = vec![nav(0, 0.0), nav(1, 1.0)];
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.01;
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states,
            landmarks: Vec::new(),
            imu_links: vec![WindowImuLink {
                from_index: 0,
                to_index: 1,
                delta,
            }],
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0e8,
            initial_accel_bias_weight: 1.0e1,
            initial_gyro_bias_weight: 1.0e2,
            prior: None,
            anchor_point: Some(DVector::zeros(15)),
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        let x = problem.initial_state();
        let factors = problem.linearize(&x).unwrap().factors;
        let imu = factors.iter().find(|factor| factor.rows() == 9).unwrap();
        assert!(imu.state_jacobian.columns(0, 15).norm() > 0.0);
        assert!(imu.state_jacobian.columns(15, 15).norm() > 0.0);
    }

    #[test]
    fn initial_prior_matches_basalt_pose_yaw_and_bias_columns() {
        let mut problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states: vec![nav(0, 0.0)],
            landmarks: Vec::new(),
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0e8,
            initial_accel_bias_weight: 1.0e1,
            initial_gyro_bias_weight: 1.0e2,
            prior: None,
            anchor_point: Some(DVector::zeros(15)),
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        let factors = problem.linearize(&problem.initial_state()).unwrap().factors;
        let anchor = factors.iter().find(|factor| factor.rows() == 15).unwrap();
        let expected_pose = 1.0e4;
        let expected_accel_bias = 10.0_f64.sqrt();
        let expected_gyro_bias = 1.0e2_f64.sqrt();
        for axis in 0..3 {
            assert!((anchor.state_jacobian[(axis, axis)] - expected_pose).abs() < 1e-12);
            assert!(
                (anchor.state_jacobian[(9 + axis, 9 + axis)] - expected_accel_bias).abs() < 1e-12
            );
            assert!(
                (anchor.state_jacobian[(12 + axis, 12 + axis)] - expected_gyro_bias).abs() < 1e-12
            );
        }
        assert!((anchor.state_jacobian[(5, 5)] - expected_pose).abs() < 1e-12);
        for column in [3, 4, 6, 7, 8] {
            assert_eq!(anchor.state_jacobian[(column, column)], 0.0);
        }
        // Keep this mutable binding so this test also exercises the same
        // linearization path used by LM callers without changing the public
        // constructor contract.
        problem.states[0].is_latest = true;
    }

    #[test]
    fn window_imu_rows_are_whitened_and_bias_walk_uses_both_state_blocks() {
        let states = vec![nav(0, 0.0), nav(1, 1.0)];
        let mut delta = ImuPreintegratedDelta::identity(Vector3::zeros(), Vector3::zeros());
        delta.delta_time = 0.1;
        delta.covariance = nalgebra::SMatrix::<f64, 9, 9>::identity() * 4.0;
        let mut problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera: DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480)
                .unwrap(),
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states,
            landmarks: Vec::new(),
            imu_links: vec![WindowImuLink {
                from_index: 0,
                to_index: 1,
                delta,
            }],
            imu_noise: ImuNoiseModel {
                gyro_density: 2.0,
                accel_density: 3.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 2.0,
                accel_density: 4.0,
            },
            initial_pose_weight: 1.0e8,
            initial_accel_bias_weight: 1.0e1,
            initial_gyro_bias_weight: 1.0e2,
            prior: None,
            anchor_point: Some(DVector::zeros(15)),
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        let initial = problem.initial_state();
        let linearization = problem.linearize(&initial).unwrap();
        let imu = linearization
            .factors
            .iter()
            .find(|factor| factor.rows() == 9)
            .expect("one whitened 9-row IMU factor");
        assert_eq!(imu.state_jacobian.ncols(), 30);
        assert!(imu.state_jacobian.columns(0, 15).norm() > 0.0);
        assert!(imu.state_jacobian.columns(15, 15).norm() > 0.0);
        assert!(imu.residual.iter().all(|value| value.is_finite()));
        // The upstream row contract is [position, rotation, velocity], and
        // the identity delta leaves the one-metre state displacement in
        // position.
        assert!(imu.residual.rows(0, 3).norm() > 0.1);
        assert!(imu.residual.rows(3, 3).norm() < 1e-12);
        assert!(imu.residual.rows(6, 3).norm() < 1e-12);
        // The IMU factor's left-local rotation convention is owned by
        // `imu::factors`; retain the cross-state contract without pinning the
        // pre-m7z tangent coefficient here.
        assert!(imu.state_jacobian[(3, 18)].abs() > 0.0);

        let bias = linearization
            .factors
            .iter()
            .find(|factor| factor.rows() == 6)
            .expect("one whitened 6-row bias walk factor");
        let gyro_weight = 2.0 / 0.1_f64.sqrt();
        let accel_weight = 4.0 / 0.1_f64.sqrt();
        assert!((bias.state_jacobian[(0, 9)] - gyro_weight).abs() < 1e-12);
        assert!((bias.state_jacobian[(0, 24)] + gyro_weight).abs() < 1e-12);
        assert!((bias.state_jacobian[(3, 12)] - accel_weight).abs() < 1e-12);
        assert!((bias.state_jacobian[(3, 27)] + accel_weight).abs() < 1e-12);

        let (_, trace) = problem
            .imu_factor_with_trace(&initial, problem.imu_links[0].clone())
            .expect("IMU adapter trace");
        assert!((trace.delta_time - 0.1).abs() < 1e-12);
        assert!((trace.residual_position[0] - 1.0).abs() < 1e-12);
        assert!(trace
            .residual_rotation
            .iter()
            .all(|value| value.abs() < 1e-12));
        assert!(trace
            .residual_velocity
            .iter()
            .all(|value| value.abs() < 1e-12));
        assert!((trace.whitened_norm - 0.5).abs() < 1e-12);
        assert!((trace.covariance_eigen_min - 4.0).abs() < 1e-12);
        assert!((trace.covariance_eigen_max - 4.0).abs() < 1e-12);

        let solved = problem.solve(initial, LmConfig::default()).unwrap();
        assert!(solved.state.iter().all(|value| value.is_finite()));
        assert!(solved.cost.is_finite());
    }

    #[test]
    fn lean_with_factors_matches_retained_diagnostics_exactly() {
        // The online-mapper/MargData fast path must produce the same accepted
        // state and the same retained factor snapshot as the fully diagnostic
        // implementation, because MargData bytes depend on both.  Build a
        // visual landmark problem with observable geometry so the LM actually
        // iterates, then compare the two solve entry points field by field.
        let camera =
            DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let true_point = Point3::new(0.2, -0.1, 1.5);
        let mut landmarks = Vec::new();
        for track_id in 0..6u64 {
            landmarks.push(WindowLandmark {
                track_id,
                anchor_state_index: 0,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(Vector3::new(
                    0.1 * track_id as f64,
                    -0.05,
                    1.0,
                ))
                .unwrap(),
                inverse_distance: 0.9,
                observations: vec![
                    WindowObservation {
                        state_index: 0,
                        camera_id: 0,
                        pixel: camera.project(&true_point).unwrap(),
                    },
                    WindowObservation {
                        state_index: 1,
                        camera_id: 0,
                        pixel: camera.project(&true_point).unwrap(),
                    },
                ],
            });
        }
        let make = || test_window(vec![nav(0, 0.0), nav(1, 0.2)], landmarks.clone());

        let initial = make().initial_state();
        let config = LmConfig::default();
        let mut retained_timing = TimingBreakdown::default();
        let mut lean_timing = TimingBreakdown::default();
        let retained = make()
            .solve_with_timing(initial.clone(), config, &mut retained_timing)
            .unwrap();
        let lean = make()
            .solve_lean_with_factors_with_timing(initial, config, &mut lean_timing)
            .unwrap();

        assert_eq!(lean.state, retained.state);
        assert_eq!(lean.cost, retained.cost);
        assert_eq!(lean.iterations, retained.iterations);
        assert_eq!(lean.factors.len(), retained.factors.len());
        for (lean_factor, retained_factor) in lean.factors.iter().zip(&retained.factors) {
            assert_eq!(
                lean_factor.state_jacobian.as_slice(),
                retained_factor.state_jacobian.as_slice()
            );
            assert_eq!(
                lean_factor.landmark_jacobian.as_slice(),
                retained_factor.landmark_jacobian.as_slice()
            );
            assert_eq!(
                lean_factor.residual.as_slice(),
                retained_factor.residual.as_slice()
            );
        }
        // The row counters emit_marg reads stay populated on the lean path
        // even though the pre-solve diagnostic linearization is skipped.
        assert_eq!(
            lean.diagnostics.visual_factor_rows,
            retained.diagnostics.visual_factor_rows
        );
        assert_eq!(
            lean.diagnostics.factor_count,
            retained.diagnostics.factor_count
        );
        assert_eq!(
            lean.diagnostics.factor_rows,
            retained.diagnostics.factor_rows
        );
    }

    #[test]
    fn grouped_landmark_qr_backsubstitutes_a_point_update() {
        let camera =
            DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let true_point = Point3::new(0.1, -0.05, 2.0);
        let states = vec![nav(0, 0.0), nav(1, 0.2)];
        let observations = vec![
            WindowObservation {
                state_index: 0,
                camera_id: 0,
                pixel: camera.project(&true_point).unwrap(),
            },
            WindowObservation {
                state_index: 1,
                camera_id: 0,
                pixel: camera
                    .project(
                        &states[1]
                            .nav
                            .imu_to_world
                            .inverse()
                            .transform_point(&true_point),
                    )
                    .unwrap(),
            },
        ];
        let initial_point = true_point + Vector3::new(0.002, -0.001, 0.003);
        let initial_distance = initial_point.coords.norm();
        let mut problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera,
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states,
            landmarks: vec![WindowLandmark {
                track_id: 7,
                anchor_state_index: 0,
                anchor_camera_id: 0,
                direction: StereographicDirection::from_bearing(initial_point.coords).unwrap(),
                inverse_distance: 1.0 / initial_distance,
                observations,
            }],
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0e8,
            initial_accel_bias_weight: 1.0e1,
            initial_gyro_bias_weight: 1.0e2,
            prior: None,
            anchor_point: Some(DVector::zeros(15)),
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::ExtendedF64,
        };
        let before =
            problem.landmarks[0].direction.bearing() / problem.landmarks[0].inverse_distance;
        let initial = problem.initial_state();
        let solved = problem.solve(initial, LmConfig::default()).unwrap();
        assert!(solved.diagnostics.visual_factor_rows > 0);
        assert_eq!(
            solved.diagnostics.prior_factor_rows
                + solved.diagnostics.visual_factor_rows
                + solved.diagnostics.imu_factor_rows
                + solved.diagnostics.bias_factor_rows,
            solved.diagnostics.factor_rows
        );
        let after =
            problem.landmarks[0].direction.bearing() / problem.landmarks[0].inverse_distance;
        assert!((after - before).norm() > 1e-8);
        assert!((after - true_point.coords).norm() < (before - true_point.coords).norm());
    }

    #[test]
    fn cam1_hosted_landmark_rows_use_host_and_target_extrinsics() {
        // Divergent rig like Project Aria's SLAM pair: cam1 = cam0 rotated 75
        // degrees about x with a 0.138 m baseline.  A landmark hosted in cam1
        // is observed by cam1 (host frame and a later frame) and by cam0 at
        // the later frame.  Built from the true point, every visual row must
        // vanish: the window has to use T_imu_cam1 for the host and the
        // observing camera's extrinsic for each target.
        let camera =
            DoubleSphereCamera::new(241.6, 241.6, 379.0, 286.0, 0.0, 0.0, 758, 572).unwrap();
        let t_imu_cam = vec![
            SE3::identity(),
            SE3::new(
                UnitQuaternion::from_scaled_axis(Vector3::new(75_f64.to_radians(), 0.0, 0.0)),
                Vector3::new(0.004, -0.109, -0.085),
            ),
        ];
        let states = vec![nav(0, 0.0), nav(1, 0.3)];
        let point_host = Vector3::new(0.1, 1.5, 2.0);
        let point_world = states[0]
            .nav
            .imu_to_world
            .compose(&t_imu_cam[1])
            .transform_point(&Point3::from(point_host));
        let observe = |state_index: usize, camera_id: u16| {
            let pose = states[state_index]
                .nav
                .imu_to_world
                .compose(&t_imu_cam[camera_id as usize]);
            let pixel = camera
                .project(&pose.inverse().transform_point(&point_world))
                .unwrap();
            assert!(camera.contains_pixel(&pixel));
            WindowObservation {
                state_index,
                camera_id,
                pixel,
            }
        };
        let landmark = WindowLandmark {
            track_id: 7,
            anchor_state_index: 0,
            anchor_camera_id: 1,
            direction: StereographicDirection::from_bearing(point_host.normalize()).unwrap(),
            inverse_distance: 1.0 / point_host.norm(),
            observations: vec![observe(0, 1), observe(1, 1), observe(1, 0)],
        };
        let visual_rows = |anchor_camera_id: u16, scalar_mode: ScalarMode| {
            let mut landmark = landmark.clone();
            landmark.anchor_camera_id = anchor_camera_id;
            let mut problem = test_window(states.clone(), vec![landmark]);
            problem.camera = camera;
            problem.cameras = vec![camera, camera];
            problem.t_imu_cam = t_imu_cam.clone();
            problem.scalar_mode = scalar_mode;
            problem
                .linearize(&problem.initial_state())
                .unwrap()
                .factors
                .into_iter()
                .find(|factor| factor.kind == FactorKind::Visual)
                .expect("one grouped visual factor")
        };
        for (scalar_mode, tolerance) in [
            (ScalarMode::ExtendedF64, 1e-6),
            (ScalarMode::UpstreamF32, 2e-2),
        ] {
            let rows = visual_rows(1, scalar_mode);
            // Two rows per stored observation (the host identity included).
            assert_eq!(rows.residual.len(), 6, "{scalar_mode:?}");
            assert!(
                rows.residual.norm() < tolerance,
                "{scalar_mode:?}: residual {:?}",
                rows.residual
            );
            assert!(rows.state_jacobian.norm() > 1.0);
            assert!(rows.landmark_jacobian.norm() > 1.0);
            // Negative control: the same parameters read in cam0 are wrong.
            let wrong = visual_rows(0, scalar_mode);
            assert!(wrong.residual.norm() > 10.0, "{scalar_mode:?}");
        }
    }

    #[test]
    fn same_timestamp_stereo_rows_accumulate_both_pose_blocks() {
        // Same-camera identity remains zero, while the stereo observation
        // contributes two relative camera terms that cancel on the shared
        // navigation block. The exact zero is a useful guard against
        // independently rounded anchor/target products leaking into AOM.
        let camera =
            DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let point = Vector3::new(0.1, -0.05, 2.0);
        let direction = StereographicDirection::from_bearing(point).unwrap();
        let observations = vec![
            WindowObservation {
                state_index: 0,
                camera_id: 0,
                pixel: Point2::new(320.0, 240.0),
            },
            WindowObservation {
                state_index: 0,
                camera_id: 1,
                pixel: Point2::new(321.0, 241.0),
            },
        ];
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera,
            cameras: vec![camera, camera],
            t_imu_cam: vec![
                SE3::identity(),
                SE3::new(
                    UnitQuaternion::from_scaled_axis(Vector3::new(0.01, -0.02, 0.03)),
                    Vector3::new(0.2, -0.1, 0.05),
                ),
            ],
            poses: Vec::new(),
            states: vec![nav(0, 0.0)],
            landmarks: vec![WindowLandmark {
                track_id: 42,
                anchor_state_index: 0,
                anchor_camera_id: 0,
                direction,
                inverse_distance: 1.0 / point.norm(),
                observations,
            }],
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
        };
        let factor = problem
            .linearize(&problem.initial_state())
            .unwrap()
            .factors
            .into_iter()
            .next()
            .expect("one grouped stereo landmark factor");
        assert_eq!(factor.rows(), 4);
        assert!(factor.state_jacobian.rows(0, 2).columns(0, 6).norm() < 1e-6);
        let landmark = &problem.landmarks[0];
        let nav_state = &problem.states[0].nav;
        let stereo = anchored_visual_reprojection_factor_f32_with_time_cam(
            &problem.cameras[1],
            &nav_state.imu_to_world,
            &problem.t_imu_cam[0],
            &nav_state.imu_to_world,
            &problem.t_imu_cam[1],
            &landmark.parameter(0),
            landmark.observations[1].pixel,
            true,
            false,
            FactorConfig::default(),
        )
        .expect("same-timestamp stereo factor");
        assert!(stereo.anchor_pose_jacobian.norm() > 1e-3);
        assert!(stereo.target_pose_jacobian.norm() > 1e-3);
        assert!(factor
            .state_jacobian
            .rows(2, 2)
            .columns(0, 6)
            .iter()
            .all(|value| (*value as f32).to_bits() == 0));
        assert!(factor.landmark_jacobian.norm() > 1.0);
    }

    #[test]
    fn pre_marginal_aom_holes_keep_native_zero_rows_in_observation_order() {
        // Native LandmarkBlockAbsDynamic reserves two rows for every stored
        // observation even when the pre-marginal AOM has truncated the target
        // state and its pose_lin entry is null.  The zero slot is part of the
        // Householder schedule, so it must remain between the two valid
        // observations rather than being filtered out.
        let camera =
            DoubleSphereCamera::new(300.0, 300.0, 320.0, 240.0, 0.5, 0.7, 640, 480).unwrap();
        let direction =
            StereographicDirection::from_bearing(Vector3::new(0.1, -0.05, 2.0)).unwrap();
        let problem = WindowProblem {
            trial_host_order: Vec::new(),
            camera,
            cameras: Vec::new(),
            t_imu_cam: Vec::new(),
            poses: Vec::new(),
            states: vec![nav(0, 0.0), nav(1, 0.25)],
            landmarks: vec![WindowLandmark {
                track_id: 9001,
                anchor_state_index: 0,
                anchor_camera_id: 0,
                direction,
                inverse_distance: 0.5,
                observations: vec![
                    WindowObservation {
                        state_index: 0,
                        camera_id: 0,
                        pixel: Point2::new(320.0, 240.0),
                    },
                    WindowObservation {
                        state_index: 2,
                        camera_id: 0,
                        pixel: Point2::new(319.0, 241.0),
                    },
                    WindowObservation {
                        state_index: 1,
                        camera_id: 0,
                        pixel: Point2::new(321.0, 241.0),
                    },
                ],
            }],
            imu_links: Vec::new(),
            imu_noise: ImuNoiseModel {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            bias_walk_noise: BiasRandomWalkNoise {
                gyro_density: 1.0,
                accel_density: 1.0,
            },
            initial_pose_weight: 1.0,
            initial_accel_bias_weight: 1.0,
            initial_gyro_bias_weight: 1.0,
            prior: None,
            anchor_point: None,
            gravity_world: Vector3::zeros(),
            scalar_mode: ScalarMode::UpstreamF32,
        };
        let state = problem.initial_state();
        let factor = problem
            .linearize(&state)
            .unwrap()
            .factors
            .into_iter()
            .next()
            .expect("one grouped visual factor");
        assert_eq!(factor.rows(), 6);
        assert!(factor.landmark_jacobian.rows(0, 2).norm() > 0.0);
        assert!(factor.state_jacobian.rows(4, 2).norm() > 0.0);
        assert!(factor
            .state_jacobian
            .rows(2, 2)
            .iter()
            .all(|value| (*value as f32).to_bits() == 0));
        assert!(factor
            .landmark_jacobian
            .rows(2, 2)
            .iter()
            .all(|value| (*value as f32).to_bits() == 0));
        assert!(factor
            .residual
            .rows(2, 2)
            .iter()
            .all(|value| (*value as f32).to_bits() == 0));

        let mut without_hole = problem;
        without_hole.landmarks[0].observations.remove(1);
        let compact_factor = without_hole
            .linearize(&without_hole.initial_state())
            .unwrap()
            .factors
            .into_iter()
            .next()
            .expect("compact grouped visual factor");
        assert_eq!(
            factor.objective_cost.to_bits(),
            compact_factor.objective_cost.to_bits()
        );
    }

    #[test]
    fn normal_system_stage_checkpoint_excludes_lm_solver_damping() {
        // `get_dense_H_b` returns the reduced normal system before the LM
        // driver constructs H_copy.  Keep a non-zero prior and a deliberately
        // unrelated value in the test fixture so accidentally adding the
        // solver's damping diagonal to stage 3 is observable at the bit level.
        let mut visual_h = DMatrix::<f64>::zeros(2, 2);
        visual_h[(0, 0)] = 1.0;
        let mut visual_imu_h = DMatrix::<f64>::zeros(2, 2);
        visual_imu_h[(0, 0)] = 1.5;
        visual_imu_h[(1, 1)] = -2.0;
        let mut prior_h = DMatrix::<f64>::zeros(2, 2);
        prior_h[(0, 0)] = 2.25;
        prior_h[(1, 1)] = 0.25;
        let visual_b = DVector::from_vec(vec![1.0, 2.0]);
        let visual_imu_b = DVector::from_vec(vec![3.0, 4.0]);
        let prior_b = DVector::from_vec(vec![5.0, 6.0]);
        let reduced = ReducedNormalSystem {
            h: visual_imu_h.clone(),
            b: visual_imu_b.clone(),
            back_substitution: Vec::new(),
            diagnostic_stages: Some(super::super::aom::DiagnosticNormalSystem {
                visual_h,
                visual_b,
                visual_imu_h,
                visual_imu_b,
                prior_h,
                prior_b,
            }),
        };
        let payload = diagnostic_normal_system_stages(&reduced, &|value: f32| {
            format!("{:08x}", value.to_bits())
        });
        assert_eq!(payload["plus_pose_damping"]["h"]["bits"][0], "3fc00000");
        assert_eq!(payload["plus_pose_damping"]["h"]["bits"][3], "c0000000");
        assert_eq!(payload["plus_marginal_prior"]["h"]["bits"][0], "40700000");
        assert_eq!(payload["plus_marginal_prior"]["h"]["bits"][3], "bfe00000");
        assert_eq!(payload["plus_pose_damping"]["b"][0], "40400000");
        assert_eq!(payload["plus_marginal_prior"]["b"][0], "41000000");
    }

    fn read_m7im15_marg_capture_for_sqrt_trace(
        path: &std::path::Path,
    ) -> (DMatrix<f64>, DVector<f64>, Vec<usize>, Vec<usize>) {
        let bytes = fs::read(path).expect("M7IM15 MARG capture");
        assert!(bytes.len() >= 11);
        assert_eq!(&bytes[..11], b"M7IM15MARG1");
        let mut offset = 11usize;
        let mut read_u32 = || {
            let end = offset + 4;
            assert!(end <= bytes.len());
            let value = u32::from_le_bytes(bytes[offset..end].try_into().unwrap());
            offset = end;
            value
        };
        assert_eq!(read_u32(), 1);
        let keep_count = read_u32() as usize;
        let marg_count = read_u32() as usize;
        let mut keep = Vec::with_capacity(keep_count);
        for _ in 0..keep_count {
            keep.push(read_u32() as usize);
        }
        let mut marg = Vec::with_capacity(marg_count);
        for _ in 0..marg_count {
            marg.push(read_u32() as usize);
        }
        assert_eq!(read_u32(), 1);
        let rows = read_u32() as usize;
        let cols = read_u32() as usize;
        let mut jacobian = vec![0.0_f64; rows * cols];
        for column in 0..cols {
            for row in 0..rows {
                jacobian[column * rows + row] = f32::from_bits(read_u32()) as f64;
            }
        }
        let rhs_size = read_u32() as usize;
        assert_eq!(rhs_size, rows);
        let mut rhs = Vec::with_capacity(rhs_size);
        for _ in 0..rhs_size {
            rhs.push(f32::from_bits(read_u32()) as f64);
        }
        assert_eq!(offset, bytes.len());
        (
            DMatrix::from_column_slice(rows, cols, &jacobian),
            DVector::from_vec(rhs),
            keep,
            marg,
        )
    }

    fn read_m7im15_k12_focus_fixture() -> (Vec<f32>, Vec<f32>) {
        // Cargo runs package tests with the package directory as cwd when this
        // crate is selected from the workspace, while the captured diagnostic
        // lives in the workspace-level target directory.  Keep the fixture
        // lookup test-only and accept either invocation cwd.
        let relative_paths = [
            std::path::Path::new("target/m7im15_eigen_event2_focus_20260828.stdout.log"),
            std::path::Path::new("../../target/m7im15_eigen_event2_focus_20260828.stdout.log"),
        ];
        let (_path, text) = relative_paths
            .iter()
            .find_map(|path| fs::read_to_string(path).ok().map(|text| (*path, text)))
            .expect("M7IM15 k12 Eigen focus fixture");
        let mut essential = None;
        let mut bottom = None;
        for line in text.lines() {
            let words = line.split_whitespace().collect::<Vec<_>>();
            match words.first().copied() {
                Some("focus_essential") => {
                    essential = Some(
                        words[1..]
                            .iter()
                            .map(|value| f32::from_bits(u32::from_str_radix(value, 16).unwrap()))
                            .collect::<Vec<_>>(),
                    );
                }
                Some("focus_bottom") if words.get(1) == Some(&"0") => {
                    bottom = Some(
                        words[3..]
                            .iter()
                            .map(|value| f32::from_bits(u32::from_str_radix(value, 16).unwrap()))
                            .collect::<Vec<_>>(),
                    );
                }
                _ => {}
            }
        }
        (
            essential.expect("k12 essential"),
            bottom.expect("k12 column 13"),
        )
    }

    fn m7im15_k12_schedule_value(
        essential: &[f32],
        bottom: &[f32],
        packet_fma: bool,
        tail_fma: bool,
        reduction: u8,
        tail_first: bool,
    ) -> f32 {
        assert_eq!(essential.len(), 23);
        assert_eq!(bottom.len(), 23);
        let mut packets = [0.0_f32; 8];
        for offset in [0usize, 8] {
            for lane in 0..8 {
                let lhs = bottom[offset + lane];
                let rhs = essential[offset + lane];
                packets[lane] = if packet_fma {
                    lhs.mul_add(rhs, packets[lane])
                } else {
                    packets[lane] + upstream_f32_gemv_scalar_product(lhs, rhs)
                };
            }
        }
        let reduce = |packet: [f32; 8], mode: u8| -> f32 {
            match mode {
                // Eigen AVX `predux(Packet8f)` low/high then even/odd.
                0 => {
                    let p0 = packet[0] + packet[4];
                    let p1 = packet[1] + packet[5];
                    let p2 = packet[2] + packet[6];
                    let p3 = packet[3] + packet[7];
                    (p0 + p2) + (p1 + p3)
                }
                // Scalar left fold over the packet lanes.
                1 => packet.into_iter().fold(0.0, |sum, value| sum + value),
                // Horizontal pair tree with adjacent lanes.
                2 => {
                    let p0 = packet[0] + packet[1];
                    let p1 = packet[2] + packet[3];
                    let p2 = packet[4] + packet[5];
                    let p3 = packet[6] + packet[7];
                    (p0 + p1) + (p2 + p3)
                }
                // Reverse scalar fold, useful as a control for ordering.
                _ => packet.into_iter().rev().fold(0.0, |sum, value| sum + value),
            }
        };
        let mut result = 0.0_f32;
        let add_tail = |sum: &mut f32, indices: &[usize]| {
            for &offset in indices {
                let lhs = bottom[offset];
                let rhs = essential[offset];
                *sum = if tail_fma {
                    lhs.mul_add(rhs, *sum)
                } else {
                    *sum + upstream_f32_gemv_scalar_product(lhs, rhs)
                };
            }
        };
        if tail_first {
            add_tail(&mut result, &[16, 17, 18, 19, 20, 21, 22]);
            result += reduce(packets, reduction);
        } else {
            result = reduce(packets, reduction);
            add_tail(&mut result, &[16, 17, 18, 19, 20, 21, 22]);
        }
        result
    }

    #[test]
    #[ignore = "requires pinned external k12 Eigen workspace capture"]
    fn enumerate_m7im15_k12_gemv_workspace_schedules() {
        let (essential, bottom) = read_m7im15_k12_focus_fixture();
        let target = 0x35bfb5b6_u32;
        let candidates = [
            ("eigen_packet_fma_tail_fma", true, true, 0, false),
            ("eigen_packet_fma_tail_mul_add", true, false, 0, false),
            ("packet_mul_add_tail_fma", false, true, 0, false),
            ("packet_mul_add_tail_mul_add", false, false, 0, false),
            ("ltr_packet_fma_tail_fma", true, true, 1, false),
            ("adjacent_packet_fma_tail_mul_add", true, false, 2, false),
            ("reverse_packet_fma_tail_mul_add", true, false, 3, false),
            ("tail_first_packet_fma_tail_mul_add", true, false, 0, true),
            ("ltr_packet_mul_add_tail_mul_add", false, false, 1, false),
            (
                "adjacent_packet_mul_add_tail_mul_add",
                false,
                false,
                2,
                false,
            ),
        ];
        let mut best = None;
        for (name, packet_fma, tail_fma, reduction, tail_first) in candidates {
            let value = m7im15_k12_schedule_value(
                &essential, &bottom, packet_fma, tail_fma, reduction, tail_first,
            );
            let bits = value.to_bits();
            let distance = bits.abs_diff(target);
            println!("m7im15_k12_candidate {name} {bits:08x} distance={distance}");
            if best.map_or(true, |(_, best_distance)| distance < best_distance) {
                best = Some((name, distance));
            }
        }
        let (best_name, best_distance) = best.expect("schedule candidates");
        println!("m7im15_k12_best {best_name} distance={best_distance}");
        assert_eq!(target, 0x35bfb5b6);
    }

    #[test]
    #[ignore = "requires external native marginalization column captures"]
    fn m11_marg8_single_column_workspace_native() {
        let root = std::path::PathBuf::from(
            std::env::var_os("M11_QR_COLUMNS_ROOT").expect("native capture root"),
        );
        let read = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(root.join(name)).expect("capture");
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let before = read("event8_loop_k40_J190x42.f32");
        let after = read("event8_loop_k41_J190x42.f32");
        assert_eq!(before.len(), 190 * 42);
        assert_eq!(after.len(), before.len());
        let hh = std::fs::read_to_string(root.join("householder.tsv")).unwrap();
        let fields: Vec<_> = hh
            .lines()
            .find(|line| line.starts_with("40\t"))
            .unwrap()
            .split('\t')
            .collect();
        assert_eq!(fields[1], "40");
        let beta = f32::from_bits(u32::from_str_radix(fields[2], 16).unwrap());
        let tau = f32::from_bits(u32::from_str_radix(fields[3], 16).unwrap());
        let denominator = before[40 * 190 + 40] - beta;
        let essential: Vec<_> = before[40 * 190 + 41..41 * 190]
            .iter()
            .map(|v| *v / denominator)
            .collect();
        let scaled = upstream_f32_packet_scale(&essential, tau);
        let generic = upstream_f32_packet8_gemv_workspace(&essential, &before, 190, 41, 41, 1)[0];
        let candidate = upstream_f32_packet8_rhs_dot(&essential, &before, 41 * 190 + 41, 149);
        for (name, dot, expected_exact) in [
            ("generic", generic, 188),
            ("native_single_column", candidate, 190),
        ] {
            let mut result = before.clone();
            let workspace = dot + before[41 * 190 + 40];
            result[41 * 190 + 40] = (-tau).mul_add(workspace, before[41 * 190 + 40]);
            upstream_f32_packet8_sub_scaled(&mut result, 41 * 190 + 41, &scaled, workspace);
            let exact = result[41 * 190..42 * 190]
                .iter()
                .zip(&after[41 * 190..42 * 190])
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            println!(
                "{name} dot={:08x} workspace={:08x} exact={exact}/190",
                dot.to_bits(),
                workspace.to_bits()
            );
            assert_eq!(exact, expected_exact, "{name}");
        }
    }

    #[test]
    fn dump_m7im15_sqrt_qtrace_from_exact_marg_capture() {
        let Some(input) = std::env::var_os("VISLOC_BASALT_SQRT_TRACE_INPUT") else {
            return;
        };
        let Some(_dump) = std::env::var_os("VISLOC_BASALT_SQRT_DUMP_QJ") else {
            panic!("exact QR replay requires qtrace output");
        };
        let (jacobian, rhs, keep, marg) =
            read_m7im15_marg_capture_for_sqrt_trace(std::path::Path::new(&input));
        let prior = sqrt_marginalize_upstream_f32_packet(
            &jacobian,
            &rhs,
            &keep,
            &marg,
            DVector::zeros(keep.len()),
        )
        .expect("exact M7IM15 Q2 capture should produce a prior");
        assert_eq!(prior.jacobian.ncols(), keep.len());
        assert!(prior.jacobian.nrows() > 0);
    }

    mod m11_imu_r_f64_scaled_axis_candidate_20260908 {
        use super::*;
        use nalgebra::{Quaternion, UnitQuaternion, Vector3};
        use serde_json::Value;
        use std::collections::HashSet;

        const FIXTURE: &str =
            include_str!("../../tests/fixtures/m11_imu_r_f64_scaled_axis_candidate_20260908.json");

        fn string_field<'a>(value: &'a Value, key: &str) -> &'a str {
            value
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_else(|| panic!("missing string field {key}"))
        }

        fn integer_field(value: &Value, key: &str) -> u64 {
            value
                .get(key)
                .and_then(Value::as_u64)
                .unwrap_or_else(|| panic!("missing integer field {key}"))
        }

        fn vector_bits(value: &Value, key: &str, expected_len: usize) -> Vec<u64> {
            let values = value
                .get(key)
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("missing array field {key}"));
            assert_eq!(values.len(), expected_len);
            values
                .iter()
                .map(|entry| {
                    let bits = entry.as_str().expect("f64 bit string");
                    assert_eq!(bits.len(), 16);
                    u64::from_str_radix(bits, 16).expect("f64 bits")
                })
                .collect()
        }

        fn raw_rotation(product_xyzw: [f64; 4]) -> UnitQuaternion<f64> {
            UnitQuaternion::new_unchecked(Quaternion::new(
                product_xyzw[3],
                product_xyzw[0],
                product_xyzw[1],
                product_xyzw[2],
            ))
        }

        fn scaled_axis_bits(value: Vector3<f64>) -> [u64; 3] {
            [value.x.to_bits(), value.y.to_bits(), value.z.to_bits()]
        }

        #[test]
        fn actual_154_quaternions_match_shared_grouped_candidate_and_native_fallback() {
            let fixture: Value = serde_json::from_str(FIXTURE).expect("valid candidate fixture");
            assert_eq!(
                string_field(&fixture, "schema_id"),
                "visloc.basalt.m11.imu_r_f64.scaled_axis_candidate_regression.v1"
            );
            assert_eq!(integer_field(&fixture, "schema_version"), 1);
            assert_eq!(
                string_field(&fixture, "expected_source"),
                "RustLinuxDiagnostic"
            );
            assert!(!fixture["expected_source_is_native_oracle"]
                .as_bool()
                .unwrap());
            assert!(!fixture["native_basalt_oracle_used"].as_bool().unwrap());
            assert_eq!(
                string_field(&fixture, "candidate_scope"),
                "diagnostic_f64_raw_rotation_only"
            );
            assert!(!fixture["production_path_unchanged"].as_bool().unwrap());

            let helper = &fixture["helper_source"];
            assert_eq!(
                string_field(helper, "path"),
                "work/m11_compensated_f64_atan_small_candidate_scale_only_grouped_20260907.rs"
            );
            assert_eq!(
                string_field(helper, "sha256"),
                "5B3DF98640950912EDA558743BD58256862FC29575BDD4F8C9EF4BF2D8E12246"
            );
            let source_hashes = &fixture["source_hashes"];
            for key in [
                "window_rs",
                "linux_actual_capture",
                "msvc_actual_capture",
                "test_module_rs",
            ] {
                let hash = string_field(source_hashes, key);
                assert_eq!(hash.len(), 64, "source hash {key}");
                assert!(hash.chars().all(|character| character.is_ascii_hexdigit()));
            }
            assert_eq!(
                string_field(source_hashes, "linux_actual_capture"),
                "51DFB68440D9393842F3FF98E1CE7F391F15C88CA069F36D8365B85C0B997E83"
            );
            assert_eq!(
                string_field(source_hashes, "msvc_actual_capture"),
                "34D32C64177DCD26200BF165C33EDB3A2C1ECA9E4C68D7F8121F04C7095C3910"
            );

            let capture = &fixture["capture_contract"];
            assert_eq!(string_field(capture, "q_component_order"), "xyzw");
            assert_eq!(string_field(capture, "expected_r_R_component_order"), "xyz");
            assert_eq!(
                string_field(capture, "expected_r_R_source"),
                "RustLinuxDiagnostic"
            );
            assert_eq!(integer_field(capture, "record_count"), 154);
            assert_eq!(integer_field(capture, "composite_link_count"), 154);
            assert!(capture["cross_os_q_f64_bits_equal"].as_bool().unwrap());
            assert!(capture["input_scalars_finite"].as_bool().unwrap());
            assert!(capture["input_w_positive"].as_bool().unwrap());

            let contract = &fixture["candidate_contract"];
            assert_eq!(string_field(contract, "ratio_threshold"), "2^-20");
            assert_eq!(
                string_field(contract, "ratio_threshold_f64_bits"),
                "3eb0000000000000"
            );
            assert_eq!(integer_field(contract, "inside_count"), 79);
            assert_eq!(integer_field(contract, "outside_count"), 75);
            assert_eq!(string_field(contract, "inside_route"), "grouped_scale_only");
            assert_eq!(
                string_field(contract, "outside_route"),
                "platform_std_fallback"
            );

            let records = fixture["records"].as_array().expect("fixture records");
            assert_eq!(records.len(), 154);
            let mut grouped_count = 0;
            let mut fallback_count = 0;
            let mut seen_links = HashSet::new();
            for (ordinal, record) in records.iter().enumerate() {
                assert_eq!(integer_field(record, "ordinal"), ordinal as u64);
                let key = (
                    integer_field(record, "active_frame_id"),
                    integer_field(record, "from_frame_id"),
                    integer_field(record, "to_frame_id"),
                    integer_field(record, "link_index"),
                );
                assert!(seen_links.insert(key));
                let q_bits = vector_bits(record, "product_xyzw_f64_bits", 4);
                let product = [
                    f64::from_bits(q_bits[0]),
                    f64::from_bits(q_bits[1]),
                    f64::from_bits(q_bits[2]),
                    f64::from_bits(q_bits[3]),
                ];
                assert!(product.iter().all(|value| value.is_finite()));
                assert!(product[3] > 0.0);
                let norm_squared =
                    product[0] * product[0] + product[1] * product[1] + product[2] * product[2];
                let route = if norm_squared.sqrt() / product[3] <= IMU_F64_ATAN_SMALL_MAX_RATIO {
                    grouped_count += 1;
                    "grouped_scale_only"
                } else {
                    fallback_count += 1;
                    "platform_std_fallback"
                };
                assert_eq!(string_field(record, "candidate_route"), route);
                let expected_bits = vector_bits(record, "expected_r_R_xyz_f64_bits", 3);
                assert_eq!(
                    scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(product)))
                        .as_slice(),
                    expected_bits.as_slice(),
                    "candidate mismatch at ordinal {ordinal}"
                );
            }
            assert_eq!(seen_links.len(), 154);
            assert_eq!(grouped_count, 79);
            assert_eq!(fallback_count, 75);
        }

        #[test]
        fn shared_candidate_preserves_axis_sign_zero_axis_and_helper_domain_fallback() {
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation([
                    0.0, 0.0, 0.0, 1.0,
                ]))),
                [0.0_f64.to_bits(); 3]
            );
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation([
                    0.0, 0.0, 0.0, -1.0,
                ]))),
                [0.0_f64.to_bits(); 3]
            );

            let positive_w = [1.0e-8, -2.0e-8, 3.0e-8, 1.0];
            let negative_w = positive_w.map(|value| -value);
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(positive_w))),
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&raw_rotation(negative_w)))
            );

            assert_eq!(
                imu_f64_atan_small_grouped_positive(0.0, 1.0)
                    .expect("zero angle")
                    .to_bits(),
                0.0_f64.to_bits()
            );
            assert_eq!(
                imu_f64_atan_small_grouped_positive(-0.0, 1.0)
                    .expect("negative zero angle")
                    .to_bits(),
                (-0.0_f64).to_bits()
            );
            for (y, x) in [
                (-1.0, 1.0),
                (1.0, 0.0),
                (1.0, -0.0),
                (1.0, -1.0),
                (f64::INFINITY, 1.0),
                (1.0, f64::INFINITY),
                (f64::NAN, 1.0),
                (1.0, f64::NAN),
            ] {
                assert!(
                    imu_f64_atan_small_grouped_positive(y, x).is_none(),
                    "accepted invalid helper domain y={y:?} x={x:?}"
                );
            }
            let threshold_bits = IMU_F64_ATAN_SMALL_MAX_RATIO.to_bits();
            assert!(
                imu_f64_atan_small_grouped_positive(f64::from_bits(threshold_bits - 1), 1.0)
                    .is_some()
            );
            assert!(
                imu_f64_atan_small_grouped_positive(IMU_F64_ATAN_SMALL_MAX_RATIO, 1.0).is_some()
            );
            assert!(
                imu_f64_atan_small_grouped_positive(f64::from_bits(threshold_bits + 1), 1.0)
                    .is_none()
            );

            let invalid = raw_rotation([f64::NAN, 0.0, 0.0, 1.0]);
            assert_eq!(
                scaled_axis_bits(imu_f64_scaled_axis_candidate(&invalid)),
                scaled_axis_bits(invalid.scaled_axis())
            );
            for scalar in [0.0, -0.0] {
                let product = raw_rotation([0.25, -0.125, 0.0, scalar]);
                assert_eq!(
                    scaled_axis_bits(imu_f64_scaled_axis_candidate(&product)),
                    scaled_axis_bits(product.scaled_axis())
                );
            }
        }
    }

    // Test-only reanchor schedule candidate.  The production path remains
    // untouched: this isolates the captured 21-row stored-RHS update while
    // its native pre-b/delta operand binding is still incomplete.
    mod m11_prior_reanchor_rhs_schedule_20260908 {
        use super::{DMatrix, DVector, WindowProblem};
        use serde_json::Value;

        const FIXTURE: &str =
            include_str!("../../tests/fixtures/m11_prior_reanchor_rhs_schedule_20260908.json");

        fn bits<const N: usize>(value: &Value, key: &str) -> [u32; N] {
            let values = value
                .get(key)
                .and_then(Value::as_array)
                .unwrap_or_else(|| panic!("missing bit array {key}"));
            assert_eq!(values.len(), N, "bit array length for {key}");
            values
                .iter()
                .map(|entry| {
                    u32::from_str_radix(
                        entry
                            .as_str()
                            .unwrap_or_else(|| panic!("non-string bit entry in {key}")),
                        16,
                    )
                    .unwrap_or_else(|error| panic!("invalid bit entry in {key}: {error}"))
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap_or_else(|_| panic!("wrong bit array length for {key}"))
        }

        // Decode the captured packet and exercise the same private helper as
        // the UpstreamF32 production re-anchor.  The independent witnesses
        // below remain test-local so association changes are visible.
        fn prior_reanchor_rhs_col_major_21_f32_candidate(
            jacobian_bits_col_major: &[u32; 441],
            stored_rhs_bits: &[u32; 21],
            keep_delta_bits: &[u32; 21],
        ) -> [u32; 21] {
            let jacobian = DMatrix::from_column_slice(
                21,
                21,
                &jacobian_bits_col_major
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>(),
            );
            let stored_rhs =
                DVector::from_iterator(21, stored_rhs_bits.iter().copied().map(f32::from_bits));
            let keep_delta =
                DVector::from_iterator(21, keep_delta_bits.iter().copied().map(f32::from_bits));
            WindowProblem::prior_reanchor_rhs_col_major_21_f32(&jacobian, &stored_rhs, &keep_delta)
                .iter()
                .copied()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
                .try_into()
                .expect("21-row candidate output")
        }

        // Algebraically equivalent, but not the native association: round
        // every depth product before the ordinary subtraction from b.
        #[inline(never)]
        fn separate_depth_reanchor_rhs(
            jacobian_bits_col_major: &[u32; 441],
            stored_rhs_bits: &[u32; 21],
            keep_delta_bits: &[u32; 21],
        ) -> [u32; 21] {
            let mut result = [0_u32; 21];
            for row in 0..21 {
                let mut value = 0.0_f32;
                for column in 0..21 {
                    let product = f32::from_bits(jacobian_bits_col_major[column * 21 + row])
                        * f32::from_bits(keep_delta_bits[column]);
                    value += product;
                }
                result[row] = (f32::from_bits(stored_rhs_bits[row]) - value).to_bits();
            }
            result
        }

        // Another plausible expression starts the accumulator at b and fuses
        // each negative product into it.  Keep it as a negative witness so a
        // future change cannot collapse the tested association to this form.
        #[inline(never)]
        fn negative_fma_into_stored_rhs(
            jacobian_bits_col_major: &[u32; 441],
            stored_rhs_bits: &[u32; 21],
            keep_delta_bits: &[u32; 21],
        ) -> [u32; 21] {
            let mut result = [0_u32; 21];
            for row in 0..20 {
                let mut value = f32::from_bits(stored_rhs_bits[row]);
                for column in 0..21 {
                    value = f32::from_bits(jacobian_bits_col_major[column * 21 + row])
                        .mul_add(-f32::from_bits(keep_delta_bits[column]), value);
                }
                result[row] = value.to_bits();
            }

            let mut value = 0.0_f32;
            for column in 0..21 {
                let product = f32::from_bits(jacobian_bits_col_major[column * 21 + 20])
                    * f32::from_bits(keep_delta_bits[column]);
                value += product;
            }
            result[20] = (f32::from_bits(stored_rhs_bits[20]) - value).to_bits();
            result
        }

        fn mismatch_indices(actual: &[u32; 21], expected: &[u32; 21]) -> Vec<usize> {
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .filter_map(|(index, (actual, expected))| (actual != expected).then_some(index))
                .collect()
        }

        #[test]
        fn captured_prior_reanchor_schedule_has_native_and_negative_witnesses() {
            let fixture: Value = serde_json::from_str(FIXTURE).expect("valid reanchor fixture");
            assert_eq!(fixture["shape"]["rows"].as_u64(), Some(21));
            assert_eq!(fixture["shape"]["cols"].as_u64(), Some(21));
            assert_eq!(
                fixture["source_basis"]["matrix_layout"].as_str(),
                Some("column_major_f32")
            );

            let jacobian = bits::<441>(&fixture["inputs"], "jacobian_f32_bits_col_major");
            let stored_rhs = bits::<21>(&fixture["inputs"], "stored_rhs_f32_bits");
            let keep_delta = bits::<21>(&fixture["inputs"], "keep_delta_f32_bits");
            let expected = bits::<21>(&fixture["expected"], "native_mld_b_after_f32_bits");

            let actual =
                prior_reanchor_rhs_col_major_21_f32_candidate(&jacobian, &stored_rhs, &keep_delta);
            assert_eq!(actual, expected, "candidate schedule must match captured b");
            assert_eq!(actual[0], 0xbf4b05b0);
            assert_eq!(actual[2], 0x3ee2e4f5);

            let separated = separate_depth_reanchor_rhs(&jacobian, &stored_rhs, &keep_delta);
            assert_eq!(
                mismatch_indices(&separated, &expected),
                vec![0, 2],
                "separate product/add witness changed unexpectedly"
            );
            assert_eq!(separated[0], 0xbf4b05af);
            assert_eq!(separated[2], 0x3ee2e4f7);

            let negative_fma = negative_fma_into_stored_rhs(&jacobian, &stored_rhs, &keep_delta);
            assert_eq!(
                mismatch_indices(&negative_fma, &expected),
                vec![3],
                "negative-FMA-into-b witness changed unexpectedly"
            );
            assert_eq!(negative_fma[3], 0x41841148);
        }

        #[test]
        fn prior_reanchor_candidate_keeps_scalar_row20_tail_contract() {
            let mut jacobian = [0_u32; 441];
            let mut keep_delta = [0_u32; 21];
            let stored_rhs = [0_u32; 21];
            // A cancellation witness: separate f32 products round the
            // second product to one and produce zero, while an all-FMA tail
            // retains -2^-46.  The scalar row must take the former path.
            jacobian[20] = 1.0_f32.to_bits();
            jacobian[21 + 20] = 0x3f800001;
            keep_delta[0] = (-1.0_f32).to_bits();
            keep_delta[1] = 0x3f7ffffe;

            let actual =
                prior_reanchor_rhs_col_major_21_f32_candidate(&jacobian, &stored_rhs, &keep_delta);
            assert_eq!(actual[20], 0.0_f32.to_bits());

            let mut all_fma_sum = 0.0_f32;
            for column in 0..21 {
                all_fma_sum = f32::from_bits(jacobian[column * 21 + 20])
                    .mul_add(f32::from_bits(keep_delta[column]), all_fma_sum);
            }
            assert_eq!(all_fma_sum.to_bits(), 0xa8800000);
            assert_eq!(
                all_fma_sum
                    .mul_add(-1.0_f32, f32::from_bits(stored_rhs[20]))
                    .to_bits(),
                0x28800000
            );
        }
    }
}
