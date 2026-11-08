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
use crate::fma::FusedMulAdd;
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

mod diagnostic_env;
mod diagnostics;
mod imu;
mod iteration_trace;
mod lm;
mod lm_token;
mod marginalization;
mod packet;
mod prior;
mod problem;
mod state_vector;
mod trial_view;
mod visual;

use super::aom;
pub(crate) use diagnostic_env::*;
use diagnostics::*;
use imu::*;
use iteration_trace::*;
use lm_token::*;
pub use marginalization::*;
use packet::*;
#[cfg(test)]
use prior::*;
pub use state_vector::*;
pub(crate) use trial_view::*;

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
    /// Visual residual weighting (`vio_obs_std_dev`, `vio_obs_huber_thresh`).
    pub factor_config: FactorConfig,
    /// Visual-only (no IMU) gauge rows; `None` on every IMU path, which then
    /// builds exactly the factors it always did.
    pub visual_only: Option<VisualOnlyPrior>,
}

/// Visual-only monocular mode (`config.vio_visual_only`, not upstream; see
/// `docs/mono_visual_only_vio.md`). Without an IMU the window's gauge is 7-DoF:
/// position, full rotation and scale. The initial anchor therefore also pins
/// roll/pitch, and while no marginal prior exists one row pins the scale-frame
/// position along `scale_direction` (the two-view baseline). Velocity and bias
/// columns of every state get a weak prior towards zero: without IMU factors
/// nothing else touches them, so the prior keeps them non-singular and does
/// not move any pose.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualOnlyPrior {
    /// Second bootstrap keyframe; its position carries the scale gauge.
    pub scale_frame_id: u64,
    pub scale_position: Vector3<f64>,
    /// Unit vector along the bootstrap baseline.
    pub scale_direction: Vector3<f64>,
}

/// Square-root weight of the visual-only velocity/bias rows.
const VISUAL_ONLY_DECOUPLED_WEIGHT: f64 = 1.0;

#[cfg(test)]
mod tests;
