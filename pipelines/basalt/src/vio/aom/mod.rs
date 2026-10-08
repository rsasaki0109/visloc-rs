//! Basalt ABS_QR-style AOM reduction core.
//!
//! The ordering follows the upstream fixed-SHA contract: pose (6), velocity
//! (3), gyro bias (3), accelerometer bias (3). This module is deliberately a
//! numeric core and has no dependency on the repository bundle optimizer.
use crate::camera::DoubleSphereCamera;
use crate::imu::{eigen_ldlt_solve_f32, ImuPreintegratedDelta};
use crate::timing::{TimingBreakdown, TimingBucket};
use crate::vio::landmarks::{sophus_so3_inverse, sophus_so3_product, InverseDistanceLandmark};
use crate::vio::scalar::ScalarMode;
use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix3, Matrix3x2, Matrix6, Point2, Point3, Quaternion, SMatrix,
    UnitQuaternion, Vector2, Vector3, Vector6,
};
use rayon::prelude::*;
use serde_json::json;
use std::{
    cell::Cell,
    collections::HashSet,
    fs::OpenOptions,
    io::Write as IoWrite,
    path::Path,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
};
use visloc_core::geometry::SE3;

mod back_substitution;
mod diagnostics;
mod eigen_kernels;
mod householder;
mod lm;
mod model_decrease;
mod oracle_trace;
mod reduction;
mod visual;

pub use back_substitution::*;
pub(crate) use diagnostics::*;
use eigen_kernels::*;
pub(crate) use householder::*;
pub use lm::*;
pub use model_decrease::*;
pub(crate) use oracle_trace::*;
pub use reduction::*;
pub use visual::*;

/// Upstream AOM navigation block: pose6, velocity3, gyro-bias3, accel-bias3.
pub const AOM_NAV_DOF: usize = 15;

#[cfg(test)]
mod m11_frame39_translation_probe;

#[derive(Debug, Clone, PartialEq)]
pub struct SqrtPrior {
    pub jacobian: DMatrix<f64>,
    pub rhs: DVector<f64>,
    pub fej_point: DVector<f64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginalizationError {
    InvalidColumns,
    RankDeficient,
}

impl SqrtPrior {
    pub fn re_reference(
        &mut self,
        new_fej_point: &DVector<f64>,
    ) -> Result<(), MarginalizationError> {
        if new_fej_point.len() != self.fej_point.len() {
            return Err(MarginalizationError::InvalidColumns);
        }
        self.rhs += &self.jacobian * (&self.fej_point - new_fej_point);
        self.fej_point = new_fej_point.clone();
        Ok(())
    }
}

/// Eliminate `marginal_columns` from a whitened square-root system and emit a
/// new square-root prior over `keep_columns`. QR is used both for the
/// landmark/marginal nullspace projection and for the final SqrtToSqrt factor.
pub fn sqrt_to_sqrt_marginalize(
    jacobian: &DMatrix<f64>,
    rhs: &DVector<f64>,
    keep_columns: &[usize],
    marginal_columns: &[usize],
    fej_point: DVector<f64>,
) -> Result<SqrtPrior, MarginalizationError> {
    let n = jacobian.ncols();
    if rhs.len() != jacobian.nrows()
        || keep_columns.iter().chain(marginal_columns).any(|&c| c >= n)
        || keep_columns.iter().any(|c| marginal_columns.contains(c))
        || fej_point.len() != keep_columns.len()
    {
        return Err(MarginalizationError::InvalidColumns);
    }
    let mut jm = DMatrix::zeros(jacobian.nrows(), marginal_columns.len());
    for (k, &c) in marginal_columns.iter().enumerate() {
        jm.set_column(k, &jacobian.column(c));
    }
    let projector = if marginal_columns.is_empty() {
        DMatrix::identity(jacobian.nrows(), jacobian.nrows())
    } else {
        let qr = jm.clone().qr();
        let _q = qr.q();
        let gram = jm.transpose() * &jm;
        if let Some(inv) = gram.try_inverse() {
            DMatrix::identity(jacobian.nrows(), jacobian.nrows()) - &jm * inv * jm.transpose()
        } else {
            return Err(MarginalizationError::RankDeficient);
        }
    };
    let mut jk = DMatrix::zeros(jacobian.nrows(), keep_columns.len());
    for (k, &c) in keep_columns.iter().enumerate() {
        jk.set_column(k, &jacobian.column(c));
    }
    let projected = projector.clone() * jk;
    let projected_rhs = projector * rhs;
    let qr = projected.clone().qr();
    let rank = qr.r().diagonal().iter().filter(|x| x.abs() > 1e-10).count();
    if rank == 0 && !keep_columns.is_empty() {
        return Err(MarginalizationError::RankDeficient);
    }
    let q = qr.q();
    let out_j = q.transpose() * projected;
    let out_r = q.transpose() * projected_rhs;
    Ok(SqrtPrior {
        jacobian: out_j,
        rhs: out_r,
        fej_point,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorConfig {
    pub observation_stddev: f64,
    pub huber_delta: f64,
    pub outlier_threshold: f64,
}
impl Default for FactorConfig {
    fn default() -> Self {
        Self {
            observation_stddev: 0.5,
            huber_delta: 1.0,
            outlier_threshold: 3.0,
        }
    }
}

fn robust_whiten(residual: &DVector<f64>, config: FactorConfig) -> Option<DVector<f64>> {
    if !config.observation_stddev.is_finite() || config.observation_stddev <= 0.0 {
        return None;
    }
    let scaled = residual / config.observation_stddev;
    if scaled.norm() > config.outlier_threshold {
        return None;
    }
    let norm = scaled.norm();
    let weight = if norm > config.huber_delta {
        (config.huber_delta / norm).sqrt()
    } else {
        1.0
    };
    Some(scaled * weight)
}

/// Basalt's robust visual objective for one residual row block.
///
/// `huber_weight` is the unsquared Huber weight used to build the whitened
/// row (`sqrt(huber_weight) * raw / sigma`).  Keeping the objective formula
/// explicit prevents the common but incorrect `||whitened_row||²` shortcut
/// for outliers.
fn robust_objective_from_raw(
    residual_squared: f64,
    huber_weight: f64,
    observation_stddev: f64,
) -> f64 {
    0.5 * (2.0 - huber_weight) * huber_weight * residual_squared
        / (observation_stddev * observation_stddev)
}

pub fn visual_reprojection_factor(
    camera: &DoubleSphereCamera,
    pose_world_camera: &SE3,
    point_world: Point3<f64>,
    observation: Point2<f64>,
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let point_camera = pose_world_camera.inverse().transform_point(&point_world);
    let predicted = camera.project(&point_camera)?;
    let raw = DVector::from_vec(vec![
        observation.x - predicted.x,
        observation.y - predicted.y,
    ]);
    let residual = robust_whiten(&raw, config)?;
    let eps = 1e-6;
    let mut js = DMatrix::zeros(2, AOM_NAV_DOF);
    let mut jl = DMatrix::zeros(2, 3);
    for k in 0..6 {
        let mut d = Vector6::zeros();
        d[k] = eps;
        let plus = SE3::exp(&d).compose(pose_world_camera);
        d[k] = -eps;
        let minus = SE3::exp(&d).compose(pose_world_camera);
        let rp = projection_residual(camera, &plus, &point_world, observation)?;
        let rm = projection_residual(camera, &minus, &point_world, observation)?;
        for row in 0..2 {
            js[(row, k)] = (rp[row] - rm[row]) / (2.0 * eps) / config.observation_stddev;
        }
    }
    for k in 0..3 {
        let mut p = point_world;
        p.coords[k] += eps;
        let rp = projection_residual(camera, pose_world_camera, &p, observation)?;
        p.coords[k] -= 2.0 * eps;
        let rm = projection_residual(camera, pose_world_camera, &p, observation)?;
        for row in 0..2 {
            jl[(row, k)] = (rp[row] - rm[row]) / (2.0 * eps) / config.observation_stddev;
        }
    }
    let scaled_norm = raw.norm() / config.observation_stddev;
    if scaled_norm > config.huber_delta {
        let w = (config.huber_delta / scaled_norm).sqrt();
        js *= w;
        jl *= w;
    }
    let huber_weight = if scaled_norm > config.huber_delta {
        config.huber_delta / scaled_norm
    } else {
        1.0
    };
    Some(WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        jl,
        residual,
        robust_objective_from_raw(raw.norm_squared(), huber_weight, config.observation_stddev),
        FactorKind::Visual,
    )?)
}
fn projection_residual(
    camera: &DoubleSphereCamera,
    pose: &SE3,
    point: &Point3<f64>,
    obs: Point2<f64>,
) -> Option<Vector3<f64>> {
    let p = camera.project(&pose.inverse().transform_point(point))?;
    Some(Vector3::new(obs.x - p.x, obs.y - p.y, 0.0))
}

pub fn stereo_reprojection_factor(
    camera_left: &DoubleSphereCamera,
    pose_left: &SE3,
    camera_right: &DoubleSphereCamera,
    pose_right: &SE3,
    point: Point3<f64>,
    observations: (Point2<f64>, Point2<f64>),
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let a = visual_reprojection_factor(camera_left, pose_left, point, observations.0, config)?;
    let b = visual_reprojection_factor(camera_right, pose_right, point, observations.1, config)?;
    let mut js = DMatrix::zeros(4, AOM_NAV_DOF);
    let mut jl = DMatrix::zeros(4, 3);
    js.rows_mut(0, 2).copy_from(&a.state_jacobian);
    js.rows_mut(2, 2).copy_from(&b.state_jacobian);
    jl.rows_mut(0, 2).copy_from(&a.landmark_jacobian);
    jl.rows_mut(2, 2).copy_from(&b.landmark_jacobian);
    let mut r = DVector::zeros(4);
    r.rows_mut(0, 2).copy_from(&a.residual);
    r.rows_mut(2, 2).copy_from(&b.residual);
    WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        jl,
        r,
        a.objective_cost + b.objective_cost,
        FactorKind::Visual,
    )
}

pub fn imu_preintegration_factor(
    delta: &ImuPreintegratedDelta,
    config: FactorConfig,
) -> Option<WhitenedFactorRowStack> {
    let mut js = DMatrix::zeros(9, AOM_NAV_DOF);
    js.view_mut((0, 9), (3, 3))
        .copy_from(&delta.jacobian_rotation_gyro_bias);
    js.view_mut((3, 9), (3, 3))
        .copy_from(&delta.jacobian_velocity_gyro_bias);
    js.view_mut((3, 12), (3, 3))
        .copy_from(&delta.jacobian_velocity_accel_bias);
    js.view_mut((6, 9), (3, 3))
        .copy_from(&delta.jacobian_position_gyro_bias);
    js.view_mut((6, 12), (3, 3))
        .copy_from(&delta.jacobian_position_accel_bias);
    let raw = DVector::from_iterator(
        9,
        delta
            .delta_velocity
            .iter()
            .chain(delta.delta_position.iter())
            .chain(delta.delta_rotation.scaled_axis().iter())
            .copied(),
    );
    let residual = robust_whiten(&raw, config)?;
    WhitenedFactorRowStack::with_objective_cost_kind(
        js,
        DMatrix::zeros(9, 0),
        residual.clone(),
        0.5 * residual.norm_squared(),
        FactorKind::Imu,
    )
}

pub fn bias_random_walk_factor(
    delta_gyro: Vector3<f64>,
    delta_accel: Vector3<f64>,
    stddev: f64,
) -> Option<WhitenedFactorRowStack> {
    if !stddev.is_finite() || stddev <= 0.0 {
        return None;
    }
    let mut j = DMatrix::zeros(6, AOM_NAV_DOF);
    j.view_mut((0, 9), (3, 3)).fill(-1.0 / stddev);
    j.view_mut((3, 12), (3, 3)).fill(-1.0 / stddev);
    let r = DVector::from_iterator(
        6,
        delta_gyro
            .iter()
            .chain(delta_accel.iter())
            .map(|x| x / stddev),
    );
    WhitenedFactorRowStack::with_objective_cost_kind(
        j,
        DMatrix::zeros(6, 0),
        r.clone(),
        0.5 * r.norm_squared(),
        FactorKind::Bias,
    )
}

pub fn prior_factor(
    jacobian: DMatrix<f64>,
    residual: DVector<f64>,
) -> Option<WhitenedFactorRowStack> {
    WhitenedFactorRowStack::with_objective_cost_kind(
        jacobian,
        DMatrix::zeros(residual.len(), 0),
        residual.clone(),
        0.5 * residual.norm_squared(),
        FactorKind::Prior,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AomBlock {
    Pose6,
    Velocity3,
    GyroBias3,
    AccelBias3,
}

impl AomBlock {
    pub const fn offset(self) -> usize {
        match self {
            Self::Pose6 => 0,
            Self::Velocity3 => 6,
            Self::GyroBias3 => 9,
            Self::AccelBias3 => 12,
        }
    }
    pub const fn dof(self) -> usize {
        match self {
            Self::Pose6 => 6,
            _ => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AomState {
    pub pose6: DVector<f64>,
    pub velocity: DVector<f64>,
    pub gyro_bias: DVector<f64>,
    pub accel_bias: DVector<f64>,
}
impl Default for AomState {
    fn default() -> Self {
        Self {
            pose6: DVector::zeros(6),
            velocity: DVector::zeros(3),
            gyro_bias: DVector::zeros(3),
            accel_bias: DVector::zeros(3),
        }
    }
}
impl AomState {
    pub fn flatten(&self) -> DVector<f64> {
        let mut x = DVector::zeros(AOM_NAV_DOF);
        x.rows_mut(0, 6).copy_from(&self.pose6);
        x.rows_mut(6, 3).copy_from(&self.velocity);
        x.rows_mut(9, 3).copy_from(&self.gyro_bias);
        x.rows_mut(12, 3).copy_from(&self.accel_bias);
        x
    }
}

/// Semantic owner of a whitened row stack.
///
/// The row shape is not sufficient to identify an IMU block: a square-root
/// marginal prior can legitimately have nine rows, while the upstream IMU
/// `DenseAccumulator` receives a 15-row `[imu9 | gyro_bias3 | accel_bias3]`
/// block.  Keep this tag beside the rows so the f32 reduction never routes an
/// accidental nine-row prior (or a synthetic test factor) through the IMU
/// packet schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorKind {
    /// A factor assembled by a generic caller or a numerical test.
    Generic,
    /// A square-root marginal/anchor prior.
    Prior,
    /// A grouped visual landmark factor, before landmark elimination.
    Visual,
    /// The nine residual rows of one preintegrated IMU link.
    Imu,
    /// The six gyro/accelerometer bias random-walk rows paired with an IMU
    /// link.
    Bias,
}

/// Absolute state-column ownership of one chronological IMU link.
///
/// Basalt's `ImuBlock` receives two adjacent navigation blocks as a local
/// 30-column matrix.  The blocks are not necessarily at columns `0` and
/// `15`: pose-only keyframes are kept in the prefix of the AOM order.  Keep
/// those absolute offsets on the factor pair instead of recovering them from
/// a row/column count (or from `% 15`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImuLinkOffsets {
    pub start: usize,
    pub end: usize,
}

/// Optional identity carried by a grouped visual landmark factor.  The
/// compact UpstreamF32 landmark-recovery plan uses this identity to move a
/// precomputed increment back to the owning landmark without depending on
/// factor position.  It is deliberately an adapter rather than a required
/// constructor argument so generic/external factor builders keep their
/// historical legacy path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LandmarkFactorMetadata {
    pub landmark_index: usize,
    pub track_id: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhitenedFactorRowStack {
    pub state_jacobian: DMatrix<f64>,
    pub landmark_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
    /// Semantic source used by the f32 normal-equation reduction.  This is
    /// deliberately explicit rather than inferred from row/column counts.
    pub kind: FactorKind,
    /// Explicit absolute columns for an `Imu`/`Bias` pair.  Ordinary factors
    /// leave this unset; constructors for the active-window pair attach it
    /// after scattering their local 15-column blocks into the AOM.
    pub imu_link_offsets: Option<ImuLinkOffsets>,
    /// Absolute state columns corresponding to the compact columns of a
    /// square-root marginal prior.  Upstream evaluates the stored prior as
    /// its compact `H` matrix before scattering the resulting normal block
    /// into the active AOM.  Ordinary factors leave this unset and retain the
    /// historical global-width product path.
    pub prior_state_columns: Option<Vec<usize>>,
    /// Optional identity of the landmark represented by this grouped visual
    /// factor.  Missing metadata selects the legacy back-substitution path.
    pub landmark_metadata: Option<LandmarkFactorMetadata>,
    /// Objective contribution associated with these whitened rows.
    ///
    /// Most factors are Gaussian and use `0.5 * ||residual||²`.  Robust
    /// visual factors override this with Basalt's exact Huber objective.
    pub objective_cost: f64,
    /// Optional IMU input-stage payload used only by the f32 reduction audit.
    /// Keeping the payload on the semantic IMU row lets the reducer attach
    /// the exact factor inputs to the matching 15x30 local packet without
    /// changing any production arithmetic or relying on row ordering.
    pub imu_input_diagnostic: Option<serde_json::Value>,
    /// Optional `(state_index, camera_id)` identities for the observations
    /// belonging to a grouped visual factor.  This is metadata for the
    /// visual-prefix sidecar only; the reducer never reads it on the normal
    /// path.
    visual_observation_ids: Option<Vec<(usize, u16)>>,
}
impl WhitenedFactorRowStack {
    pub fn new(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
    ) -> Option<Self> {
        let objective_cost = 0.5 * residual.norm_squared();
        Self::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Generic,
        )
    }

    pub fn with_objective_cost(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
        objective_cost: f64,
    ) -> Option<Self> {
        Self::with_objective_cost_kind(
            state_jacobian,
            landmark_jacobian,
            residual,
            objective_cost,
            FactorKind::Generic,
        )
    }

    /// Construct a factor while retaining its semantic source tag.
    pub fn with_objective_cost_kind(
        state_jacobian: DMatrix<f64>,
        landmark_jacobian: DMatrix<f64>,
        residual: DVector<f64>,
        objective_cost: f64,
        kind: FactorKind,
    ) -> Option<Self> {
        if state_jacobian.nrows() != landmark_jacobian.nrows()
            || residual.len() != state_jacobian.nrows()
            || !objective_cost.is_finite()
        {
            None
        } else {
            Some(Self {
                state_jacobian,
                landmark_jacobian,
                residual,
                objective_cost,
                kind,
                imu_link_offsets: None,
                prior_state_columns: None,
                landmark_metadata: None,
                imu_input_diagnostic: None,
                visual_observation_ids: None,
            })
        }
    }

    /// Assign a semantic source to a factor returned by an existing
    /// constructor.  Keeping this small adapter preserves the historical
    /// constructor API for downstream synthetic callers.
    pub const fn with_kind(mut self, kind: FactorKind) -> Self {
        self.kind = kind;
        self
    }

    /// Attach the absolute start/end navigation-block columns owned by an
    /// IMU link.  This is deliberately a separate adapter so legacy factor
    /// constructors remain source-compatible.
    pub const fn with_imu_link_offsets(mut self, start: usize, end: usize) -> Self {
        self.imu_link_offsets = Some(ImuLinkOffsets { start, end });
        self
    }

    /// Attach the active absolute AOM columns represented by a compact
    /// square-root prior.  The f32 reducer uses this metadata to reproduce
    /// Eigen's compact `H.transpose() * H` schedule before scattering it into
    /// the global state matrix.
    pub fn with_prior_state_columns(mut self, columns: Vec<usize>) -> Self {
        self.prior_state_columns = Some(columns);
        self
    }

    /// Attach the optional landmark identity used by the clean UpstreamF32
    /// compact recovery plan.  Callers that do not provide it retain the
    /// historical per-factor recovery fallback.
    pub const fn with_landmark_metadata(mut self, landmark_index: usize, track_id: u64) -> Self {
        self.landmark_metadata = Some(LandmarkFactorMetadata {
            landmark_index,
            track_id,
        });
        self
    }

    /// Attach an already materialized diagnostic-only IMU input payload.
    /// Production callers leave this unset; the active-window factor builder
    /// populates it only when `VISLOC_BASALT_DIAGNOSTIC_IMU_ROWS` is enabled.
    pub fn with_imu_input_diagnostic(mut self, payload: serde_json::Value) -> Self {
        self.imu_input_diagnostic = Some(payload);
        self
    }

    /// Attach observation identities for the opt-in visual-prefix capture.
    /// The compact pair keeps this adapter independent of the active-window
    /// observation type and has no effect on numeric factor evaluation.
    pub fn with_visual_observation_ids(mut self, observations: Vec<(usize, u16)>) -> Self {
        self.visual_observation_ids = Some(observations);
        self
    }
    pub fn rows(&self) -> usize {
        self.residual.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LandmarkBackSubstitution {
    pub state_jacobian: DMatrix<f64>,
    pub landmark_jacobian: DMatrix<f64>,
    pub residual: DVector<f64>,
    pub rank: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReducedNormalSystem {
    pub h: DMatrix<f64>,
    pub b: DVector<f64>,
    pub back_substitution: Vec<LandmarkBackSubstitution>,
    /// Optional f32 normal-system checkpoints used by the frame-level
    /// parity logger.  This is populated only when an explicit diagnostic
    /// environment variable is set; the production solver still consumes
    /// only `h`/`b`.
    pub diagnostic_stages: Option<DiagnosticNormalSystem>,
}

/// Source-order normal-system checkpoints for the pinned float32 path.
///
/// Basalt's `LinearizationAbsQR::get_dense_H_b` has four observable phases:
/// visual landmark reduction, the separate IMU `DenseAccumulator`, pose
/// damping, and finally the marginal prior.  Keeping these snapshots as
/// f64 containers is intentional: every value is converted from an already
/// rounded f32 operand, so the logger can serialize the original f32 bits
/// without changing the active arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticNormalSystem {
    pub visual_h: DMatrix<f64>,
    pub visual_b: DVector<f64>,
    pub visual_imu_h: DMatrix<f64>,
    pub visual_imu_b: DVector<f64>,
    pub prior_h: DMatrix<f64>,
    pub prior_b: DVector<f64>,
}

#[cfg(test)]
mod tests;
