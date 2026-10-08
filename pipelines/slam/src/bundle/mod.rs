//! Bundle adjustment with Schur-complement landmark elimination.
//!
//! Optimizes camera poses jointly with landmark positions to minimize the
//! sum of squared 2D reprojection residuals. Pinhole intrinsics are held
//! fixed; the variables are pose `T_world_to_camera` (6 DoF, right
//! perturbation `T 竊・T ﾂｷ Exp(ﾎｾ)` with `ﾎｾ = [ﾏ・ ﾏ云`) per non-fixed pose
//! and `X_w` (3 DoF) per non-fixed landmark.
//!
//! The Schur complement of the block-diagonal landmark Hessian `H_LL`
//! reduces the linear system to one of pose-only size `(6P) ﾃ・(6P)` per
//! iteration, regardless of how many landmarks the scene has, then
//! back-substitutes for the landmark updates. Each iteration is a
//! Levenberg-Marquardt step with optional cost-rejection.
//!
//! Gauge fixing is the caller's responsibility: monocular BA has 7 DoF
//! gauge freedom (6 SE(3) + 1 scale). At minimum fix the first pose
//! (anchor) and one of the following to remove scale: a second pose, a
//! second landmark, or a known-distance pair. Rectified-stereo BA (any
//! [`BaStereoObservation`] present) has only 6 DoF gauge freedom 窶・the
//! baseline anchors metric scale 窶・so a single fixed pose is enough.
//!
//! # Parallelism
//!
//! [`BaConfig::parallel`] (default `false`, so [`BundleAdjustment::optimize`]
//! and friends are unchanged unless a caller opts in) parallelizes the three
//! per-item hot loops of [`BundleAdjustment::optimize_weighted`]'s
//! Levenberg-Marquardt iteration with `rayon`:
//!
//! - **Assembly** (`build_normal_equations`'s monocular observation loop):
//!   each observation's residual/Jacobian is a pure function of the current
//!   pose and landmark estimate 窶・it touches no shared state 窶・so it is
//!   computed on the rayon pool in fixed-size chunks
//!   ([`PARALLEL_OBSERVATION_CHUNK`]), collected into a plain per-chunk
//!   `Vec`; the actual `+=` scatter into `h_pp` / `b_p` / the per-landmark
//!   blocks stays a single serial pass over each chunk's precomputed
//!   contributions, *in the original per-observation order*.
//! - **Schur reduction** (`solve_step`'s per-landmark `S -= H_PL H_LL竅ｻﾂｹ
//!   H_PL盞` loop): each landmark's `3ﾃ・` factorization is independent and
//!   computed directly in parallel (disjoint output slots, no merge needed);
//!   the pose-pair contributions it produces are computed the same chunked
//!   way as assembly, collecting each landmark's `(Vec<(p,q,block)>,
//!   Vec<(p,upd)>)` pair per chunk and then flattening/merging into the
//!   shared reduced system `s` / `b_reduced` by a serial pass over each
//!   chunk, in landmark-ascending order. Pure-visual sparse BA instead keeps
//!   this stage serial so parallel observation assembly does not force a
//!   dense camera Hessian or change the block-sparse O(nnz) memory bound.
//! - **Back-substitution** (`solve_step`'s per-landmark `δ_L` loop): each
//!   landmark writes only its own 3 rows of `delta_l`, so this is
//!   embarrassingly parallel with no merge step at all.
//!
//! Unlike [`crate::block_cholesky`]'s intra-column path 窶・which reassociates
//! a floating-point sum across contributors and is therefore only
//! deterministic *to rounding* 窶・every merge here reproduces the exact
//! summation order the serial code would have used, so the parallel path is
//! bit-identical to the serial one at any thread count or chunk size; the
//! chunk constants below exist only to cap peak memory (a full-sequence BA
//! can carry tens of millions of observations, so materializing one
//! contribution per observation up front is not an option) and to amortize
//! the per-dispatch rayon overhead, never to change the result. Each path is
//! also work-gated ([`PARALLEL_MIN_OBSERVATIONS`], [`PARALLEL_MIN_LANDMARKS`])
//! so small problems stay on the plain serial loop even with the flag on,
//! matching `block_cholesky`'s `PARALLEL_MIN_BLOCKS` precedent.
//!
//! Not parallelized: [`BundleAdjustment::optimize_joint_intrinsics`]'s own
//! Schur reduction (a separate, less-used code path 窶・self-calibration BA is
//! opt-in and typically run on far smaller problems than a full-sequence
//! pose/structure solve) and the cost-evaluation passes (`robust_cost_weighted`
//! / `reprojection_squared_residuals`, shared by many callers beyond the LM
//! loop, so gating them on `BaConfig` would require threading the flag
//! through call sites that have nothing to do with this optimizer).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::ops::{Index, IndexMut};

use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix2x4, Matrix2x6, Matrix3, Matrix3x4, Matrix3x6, Matrix4x3,
    Matrix4x6, Matrix6, Matrix6x3, Point2, Point3, Vector2, Vector3, Vector4, Vector6,
};

use visloc_core::geometry::{Pose, SE3, SO3};
use visloc_core::types::{Camera, CameraModel, VisualMap};
use visloc_mapping::{
    LocalMapWindow, LocalRefinementReason, LocalRefinementResult, LocalRefiner, StagedMapUpdate,
};

use crate::gnc::{GncConfig, GncState};
use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::process_memory::log as log_process_memory;
use crate::{solve_normal_equations, LinearSolver, PoseGraphError, RobustKernel};

mod debug;
mod jacobians;
mod joint_intrinsics;
mod lm;
mod matrix_free;
mod normal_equations;
mod optimize;
mod problem;
mod sqrt_qr;

use debug::*;
pub(crate) use jacobians::*;
use lm::*;
use matrix_free::*;
use normal_equations::*;
pub use sqrt_qr::*;

#[cfg(test)]
mod schur_block_debug_tests;

#[cfg(test)]
mod generalized_rig_factor_tests;

/// Convert optional normalized matcher confidences into relative BA
/// information weights without changing the visual factor group's mean scale.
///
/// Learned match probabilities are not calibrated inverse variances. Feeding
/// them directly into tight VI-BA would weaken the entire visual block against
/// the physically whitened IMU block. Explicit scores are therefore divided
/// by their own finite mean; observations without a score stay at `1`.
/// Returns `None` when no valid confidence signal is present.
pub(crate) fn relative_observation_confidence_weights(
    confidences: impl IntoIterator<Item = Option<f32>>,
) -> Option<Vec<f64>> {
    let confidences = confidences
        .into_iter()
        .map(|confidence| {
            confidence.filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        })
        .collect::<Vec<_>>();
    let explicit = confidences.iter().flatten().copied().collect::<Vec<_>>();
    if explicit.is_empty() {
        return None;
    }
    let mean = explicit.iter().map(|value| *value as f64).sum::<f64>() / explicit.len() as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return None;
    }
    Some(
        confidences
            .into_iter()
            .map(|confidence| confidence.map_or(1.0, |value| value as f64 / mean))
            .collect(),
    )
}

/// FEJ-style dense Gaussian prior over one or more navigation states.
///
/// Each keyframe contributes `[pose(6), velocity(3), bias(6)]` in that order.
/// `information`, `gradient`, and `constant_cost` describe the quadratic at
/// `reference`: `c + 2 g^T dx + dx^T H dx`. Pose deltas use the same right
/// perturbation as BA, `T = T_ref Exp(dx)`. Keeping the reference fixed avoids
/// silently changing the linearisation point as a fixed-lag window slides.
#[derive(Debug, Clone, PartialEq)]
pub struct NavigationStatePrior {
    pub keyframe_ids: Vec<u64>,
    pub reference_poses: BTreeMap<u64, Pose>,
    pub reference_velocities: BTreeMap<u64, Vector3<f64>>,
    pub reference_biases: BTreeMap<u64, Vector6<f64>>,
    pub information: DMatrix<f64>,
    pub gradient: DVector<f64>,
    pub constant_cost: f64,
}

impl NavigationStatePrior {
    pub fn is_well_formed(&self) -> bool {
        let dim = self.keyframe_ids.len() * 15;
        self.information.nrows() == dim
            && self.information.ncols() == dim
            && self.gradient.len() == dim
            && self.constant_cost.is_finite()
            && self.information.iter().all(|value| value.is_finite())
            && self.gradient.iter().all(|value| value.is_finite())
            && self.keyframe_ids.iter().all(|id| {
                self.reference_poses.contains_key(id)
                    && self.reference_velocities.contains_key(id)
                    && self.reference_biases.contains_key(id)
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NavigationLinearization {
    pub pose_ids: Vec<u64>,
    pub velocity_ids: Vec<u64>,
    pub bias_ids: Vec<u64>,
    pub information: DMatrix<f64>,
    pub gradient: DVector<f64>,
}

/// One 2D image-point measurement linking a keyframe to a landmark.
#[derive(Debug, Clone, PartialEq)]
pub struct BaObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    /// Pixel coordinates `(u, v)` in the keyframe's image.
    pub xy: Point2<f64>,
}

/// One rectified-stereo measurement linking a keyframe to a landmark. The
/// keyframe's pose is the LEFT camera's `T_world_to_camera`. The right camera
/// is assumed rectified: shared intrinsics, optical axes parallel, and image
/// rows aligned, so the right pixel only needs its horizontal coordinate
/// (`v_r = v_l`). The shared baseline lives on [`BundleAdjustment`].
///
/// Compared with two independent [`BaObservation`]s for the left and right
/// pixel, a single [`BaStereoObservation`] (i) avoids carrying a separate
/// right-camera pose (it is implicitly the left's translated by `bﾂｷxﾌＡ) and
/// (ii) couples the two residuals through the same landmark variable, which
/// is the standard rectified-stereo BA formulation.
#[derive(Debug, Clone, PartialEq)]
pub struct BaStereoObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    /// Left-image pixel coordinates `(u_l, v_l)`.
    pub xy: Point2<f64>,
    /// Right-image horizontal pixel coordinate `u_r`. The vertical coordinate
    /// `v_r` is taken to equal `xy.y` (rectified-stereo assumption).
    pub u_right: f64,
}

/// One calibrated, non-rectified stereo observation. The keyframe pose is the
/// left camera's `T_left<-world`; `left_to_right` is the fixed rig transform
/// `T_right<-left`. Unlike [`BaStereoObservation`], both right-image
/// coordinates and the right camera intrinsics are retained, so rigs with a
/// rotational cam0/cam1 extrinsic (including EuRoC) contribute their true
/// four-dimensional reprojection residual.
#[derive(Debug, Clone, PartialEq)]
pub struct BaGeneralStereoObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    pub xy_left: Point2<f64>,
    pub xy_right: Point2<f64>,
    pub right_camera: Camera,
    pub left_to_right: SE3,
}

/// One observation from an arbitrary sensor rigidly attached to a shared rig
/// frame. `poses[keyframe_id]` stores `T_rig<-world`; the fixed extrinsic is
/// `T_sensor<-rig`. This is the non-central counterpart of
/// [`BaObservation`] and allows every sensor pixel to constrain one body pose
/// without requiring a same-landmark observation in a designated left camera.
#[derive(Debug, Clone, PartialEq)]
pub struct BaRigObservation {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    pub xy: Point2<f64>,
    pub camera: Camera,
    pub sensor_from_rig: SE3,
}

/// Rotation-alignment gravity prior on every non-fixed pose.
///
/// Adds a 3-vector residual `r = R_wc ﾂｷ g_world 竏・g_camera_observed` per
/// pose, where `R_wc` is the pose's world-to-camera rotation and the two
/// gravity vectors are caller-supplied. The most common use is a level
/// prior: set both vectors to the same down-direction (e.g.
/// `(0, 9.81, 0)` for a KITTI-style y-down camera that starts level)
/// and the optimiser will resist pitch / roll drift that re-projection
/// residuals cannot disambiguate on coplanar-feature scenes.
///
/// This prior constrains ROTATION only. Pure-translation drift (such as
/// the structural vertical bias on KITTI sequence 08, where the camera
/// rotation already matches ground truth) is NOT corrected by this
/// prior 窶・that would require a translation/altitude prior fed from
/// IMU velocity or GNSS, which lives outside [`BundleAdjustment`] in
/// its current form.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityPrior {
    /// Gravity direction in world frame. Magnitude defines the
    /// residual's natural scale; using the physical 9.81 m/sﾂｲ keeps the
    /// per-pose residual in the same order of magnitude as a pixel
    /// reprojection residual, so a default Huber `delta 竕・3` does not
    /// over- or under-weight the prior.
    pub g_world: Vector3<f64>,
    /// Gravity direction observed (or assumed) in camera frame for
    /// every pose. For a level prior this matches the camera-frame
    /// direction of `g_world` at the anchor pose, e.g. `(0, 9.81, 0)`.
    pub g_camera_observed: Vector3<f64>,
    /// Scalar weight applied to the gravity contribution. The cost
    /// added per pose is `weight ﾂｷ 窶睦窶鳴ｲ` and the normal-equations
    /// contribution is `weight ﾂｷ J盞 J` / `weight ﾂｷ J盞 r`. A weight of
    /// `1.0` makes a 9.81 m/sﾂｲ gravity residual count comparably to a
    /// single 9.81 px reprojection residual; lower this for a softer
    /// prior, raise it for a stiffer one.
    pub weight: f64,
}

/// Per-keyframe observation of the gravity direction in camera
/// coordinates. Each entry constrains
/// `R_wc ﾂｷ g_world 竕・g_camera_observed` at the named keyframe; the
/// residual and Jacobian shape are identical to [`GravityPrior`]'s
/// global pose-independent variant, except the observation is sourced
/// per-keyframe rather than shared across all poses.
///
/// The intended source of `g_camera_observed` is an accelerometer
/// sample (or a low-pass-filtered window of samples) at the keyframe
/// timestamp, rotated into the camera frame via the body竊団amera
/// extrinsic. Unlike [`PositionPrior`], which can leak ground-truth
/// poses when fed from GNSS/INS-fused trajectories, a properly-
/// generated per-keyframe gravity prior is a true online sensor
/// observation 窶・the same signal a deployed VIO would consume.
#[derive(Debug, Clone, PartialEq)]
pub struct PerPoseGravityObservation {
    /// Keyframe whose pose rotation is being constrained. The pose
    /// must already be added to [`BundleAdjustment`]. Fixed poses
    /// still contribute to the cost report but generate no Jacobian
    /// rows because they have no Hessian slot.
    pub keyframe_id: u64,
    /// Observed gravity direction in camera frame at this keyframe.
    /// Magnitude should match [`PerPoseGravityPrior::g_world`] (e.g.
    /// `9.81 m/sﾂｲ` for a physical accelerometer-derived observation),
    /// so the per-pose residual stays in the same order of magnitude
    /// as a pixel reprojection residual.
    pub g_camera_observed: Vector3<f64>,
    /// Per-observation stiffness multiplier applied on top of the
    /// global [`PerPoseGravityPrior::weight`]. `1.0` is neutral;
    /// raise to up-weight a high-confidence sample, lower to soften a
    /// motion-contaminated one. Setting to `0.0` mutes the
    /// observation entirely (useful for keeping all keyframe slots
    /// while gating obviously bad samples).
    pub weight: f64,
}

impl PerPoseGravityObservation {
    /// Build an observation with the default neutral per-obs weight
    /// (`1.0`). Use the public `weight` field directly when emitting
    /// per-sample stiffness from a sensor model.
    pub const fn new(keyframe_id: u64, g_camera_observed: Vector3<f64>) -> Self {
        Self {
            keyframe_id,
            g_camera_observed,
            weight: 1.0,
        }
    }
}

/// Per-keyframe gravity-alignment prior. Each
/// [`PerPoseGravityObservation`] adds a rotation-domain residual at
/// its keyframe; the prior as a whole shares a single world-frame
/// gravity vector and stiffness.
///
/// This is the online-friendly companion to [`GravityPrior`] (single
/// observation shared across all poses) 窶・it accepts per-keyframe
/// observations rather than baking in a single "level-world" assumption.
/// Use it when the body's pitch/roll varies meaningfully along the
/// trajectory (climbing/descending on a slope, banking on a curve,
/// etc.) so the gravity-in-camera-frame direction is no longer
/// constant.
///
/// Like [`GravityPrior`], the prior constrains ROTATION only. Pure-
/// translation drift (such as the structural vertical bias on KITTI
/// sequence 08, where the camera rotation already matches ground
/// truth) is NOT corrected by this prior.
#[derive(Debug, Clone, PartialEq)]
pub struct PerPoseGravityPrior {
    /// Per-keyframe observations. May contain at most one entry per
    /// `keyframe_id`; duplicates are accepted but each contributes
    /// independently (the optimiser does not deduplicate).
    pub observations: Vec<PerPoseGravityObservation>,
    /// Gravity direction in world frame, shared across all
    /// observations. Magnitude defines the residual's natural scale.
    pub g_world: Vector3<f64>,
    /// Global scalar weight applied to every observation's
    /// contribution, multiplied with each observation's
    /// [`PerPoseGravityObservation::weight`]. `weight = 1.0` plus
    /// per-obs `1.0` makes a 9.81 m/sﾂｲ gravity residual count
    /// comparably to a single 9.81 px reprojection residual; lower
    /// the global scale for a softer prior, raise for a stiffer one.
    /// The per-observation field stays neutral unless the upstream
    /// sensor model emits inverse-variance weights.
    pub weight: f64,
}

impl PerPoseGravityPrior {
    pub const fn new(g_world: Vector3<f64>, weight: f64) -> Self {
        Self {
            observations: Vec::new(),
            g_world,
            weight,
        }
    }

    pub fn push(&mut self, observation: PerPoseGravityObservation) {
        self.observations.push(observation);
    }
}

/// One absolute position measurement for a single keyframe. The
/// expected world-frame camera centre is compared against the BA's
/// current estimate of `竏坦盞 ﾂｷ t`. Designed for translation-domain
/// priors fed from GNSS, an external altimeter, or 窶・in evaluation
/// scenarios 窶・ground-truth poses; the prior constrains TRANSLATION
/// only, complementing [`GravityPrior`] which constrains ROTATION
/// only.
///
/// `axis_weights` enables per-axis stiffness: a per-pose altitude
/// constraint sets `axis_weights = (0, w, 0)` so only the vertical
/// component is anchored (the most common shape for fixing seq08-style
/// vertical drift without claiming horizontal GNSS accuracy).
#[derive(Debug, Clone, PartialEq)]
pub struct PositionPriorObservation {
    /// Keyframe whose world camera centre is being constrained. The
    /// pose must already be added to [`BundleAdjustment`]. Fixed poses
    /// still contribute to the cost (for diagnostics) but generate no
    /// Jacobian rows because they have no Hessian slot.
    pub keyframe_id: u64,
    /// Expected world-frame camera centre. For a level KITTI-style
    /// y-down camera this is the same coordinate frame as
    /// `pose.camera_center_world()`.
    pub camera_center_world: Point3<f64>,
    /// Per-axis weights in the cost `ﾎ｣ w盞｢ ﾂｷ (C盞｢ 竏・target盞｢)ﾂｲ` and the
    /// normal-equations contribution. A zero entry removes that axis
    /// from the prior entirely; mixed positive entries pin a subset of
    /// axes with different stiffnesses (`(0, w, 0)` for altitude-only).
    pub axis_weights: Vector3<f64>,
}

/// A relative-pose constraint between two BA keyframes, e.g. an IMU
/// pre-integration delta, a wheel-odometry tick, or an external
/// pose-graph edge being lifted into BA.
///
/// At convergence the measurement equals the BA-implied relative pose
/// `T_j ﾂｷ T_i竅ｱ` (`world_to_camera_j` of the "to" keyframe composed with
/// the inverse of the "from" keyframe). The residual is the SE(3) log
/// of the disagreement:
///
/// ```text
/// r = log(measurement竅ｻﾂｹ ﾂｷ T_j ﾂｷ T_i竅ｱ)  竏・邃昶・
/// ```
///
/// Jacobians under right-perturbation `T 竊・T ﾂｷ exp(ﾎｴ)`:
///
/// - `竏Ｓ / 竏ばｴ_j =  Ad(T_i)`
/// - `竏Ｓ / 竏ばｴ_i = 竏但d(T_i)`
///
/// This is the same Jacobian shape used by
/// `PoseGraph::optimize_se3_iterative`; the factor lifts those edges
/// into [`BundleAdjustment`] so visual residuals and external-sensor
/// pose deltas can be jointly optimised in a single LM solve. Full
/// IMU pre-integration with velocity/bias states is a future
/// extension; this v1 factor assumes the pre-integrator has already
/// produced a single `(ﾎ廃, ﾎ燃)` pair plus a scalar weight.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwisePoseFactor {
    /// "From" keyframe id (the one Ad(T_from) is computed about).
    pub keyframe_id_from: u64,
    /// "To" keyframe id.
    pub keyframe_id_to: u64,
    /// Measured relative pose `T_meas` such that, at convergence,
    /// `T_meas = T_j ﾂｷ T_i竅ｱ` where `T_i` and `T_j` are the BA poses
    /// for `keyframe_id_from` and `keyframe_id_to` respectively.
    pub measurement: Pose,
    /// Scalar weight (sqrt-information squared). The cost added is
    /// `weight ﾂｷ 窶睦窶鳴ｲ` so `weight = 1 / ﾏδｲ` for an isotropic
    /// measurement with standard deviation `ﾏチ (per-axis). Anisotropic
    /// 6ﾃ・ sqrt-information matrices are deferred to a future
    /// extension.
    pub weight: f64,
}

/// Bias random-walk factor between two keyframes' 6-vector IMU
/// biases. Adds the residual `r = b_j 竏・b_i` with independent gyro and
/// accelerometer weights. Use this to keep neighbouring
/// keyframes' biases close to each other when the IMU factor's data-
/// driven Jacobian leaves some bias DoFs unobservable in isolation
/// (e.g., gyro biases on a straight-line trajectory).
///
/// Both endpoint biases must be registered via
/// [`BundleAdjustment::add_bias`] for the factor to contribute. If
/// either side has a non-fixed bias slot, the factor adds its 6ﾃ・
/// Jacobian (`J_i = 竏棚`, `J_j = I`) to the normal equations; fully-
/// fixed endpoints still contribute to the cost report but no
/// Jacobian rows.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasRandomWalkFactor {
    /// "From" keyframe id (the bias on the `竏棚` side of the Jacobian).
    pub keyframe_id_from: u64,
    /// "To" keyframe id (the bias on the `+I` side).
    pub keyframe_id_to: u64,
    /// Gyroscope-bias sqrt-information squared. A typical value is
    /// `1 / (ﾏダbgﾂｲ ﾂｷ ﾎ杯_ij)` for continuous random-walk density `ﾏダbg`.
    pub weight_gyro: f64,
    /// Accelerometer-bias sqrt-information squared. A typical value is
    /// `1 / (ﾏダbaﾂｲ ﾂｷ ﾎ杯_ij)` for continuous random-walk density `ﾏダba`.
    pub weight_accel: f64,
}

/// A bundle of per-keyframe absolute position constraints.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionPrior {
    pub observations: Vec<PositionPriorObservation>,
    /// When true, the camera-centre residual uses the full Jacobian
    /// `[-I | [C_w]_x]`, so pose rotation updates can also move the
    /// constrained centre. When false, the residual uses `[-I | 0]`
    /// and acts as a translation-only centre prior. Keep this true for
    /// the historical BA semantics; turn it off for sensor height/grade
    /// priors that should not pull rotation away from visual evidence.
    pub couple_rotation: bool,
}

impl PositionPrior {
    pub const fn new() -> Self {
        Self {
            observations: Vec::new(),
            couple_rotation: true,
        }
    }

    pub const fn with_rotation_coupling(mut self, couple_rotation: bool) -> Self {
        self.couple_rotation = couple_rotation;
        self
    }

    pub fn push(&mut self, observation: PositionPriorObservation) {
        self.observations.push(observation);
    }
}

impl Default for PositionPrior {
    fn default() -> Self {
        Self::new()
    }
}

/// Bundle-adjustment problem: poses, landmarks, observations, plus a single
/// shared pinhole camera (multi-camera support is left as a future extension).
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustment {
    pub poses: BTreeMap<u64, Pose>,
    pub landmarks: BTreeMap<u64, Point3<f64>>,
    pub observations: Vec<BaObservation>,
    /// Rectified-stereo observations sharing [`Self::stereo_baseline`]. They
    /// reference the same `poses` / `landmarks` collections as
    /// [`Self::observations`], so a single landmark can have both monocular
    /// and stereo evidence.
    pub stereo_observations: Vec<BaStereoObservation>,
    /// Calibrated non-rectified stereo observations. These use [`Self::camera`]
    /// as the left camera and carry their right camera/extrinsic explicitly.
    pub general_stereo_observations: Vec<BaGeneralStereoObservation>,
    /// Arbitrary calibrated rig-sensor observations sharing the frame poses.
    pub rig_observations: Vec<BaRigObservation>,
    pub camera: Camera,
    /// Pose ids whose `Pose` is held constant during optimization.
    pub fixed_poses: BTreeSet<u64>,
    /// Pose ids whose rotation is held constant while their translation may
    /// still be optimized.  The six-dimensional pose slot is retained for
    /// the Schur system, but the rotation rows/columns are constrained to
    /// zero by [`Self::optimize`].  An empty set preserves the historical
    /// pose/structure solve exactly.
    pub fixed_pose_rotations: BTreeSet<u64>,
    /// Landmark ids whose `Point3` is held constant during optimization.
    pub fixed_landmarks: BTreeSet<u64>,
    /// Rectified-stereo baseline in metric units. The right camera is at
    /// `+stereo_baseline ﾂｷ xﾌＡ of the left in the left-camera frame. Required
    /// (positive, finite) when [`Self::stereo_observations`] is non-empty;
    /// ignored otherwise. `None` means "monocular BA".
    pub stereo_baseline: Option<f64>,
    /// Optional rotation-alignment gravity prior. When `Some`, every
    /// non-fixed pose contributes a 3-vector gravity-alignment residual
    /// (see [`GravityPrior`]). Fixed poses are still included in the
    /// cost report but do not generate Jacobian rows.
    pub gravity_prior: Option<GravityPrior>,
    /// Optional per-keyframe gravity-alignment prior. Like
    /// [`Self::gravity_prior`] but the `g_camera_observed` varies per
    /// observation, so the prior accepts e.g. accelerometer-derived
    /// per-keyframe observations rather than a single shared level-
    /// world assumption. See [`PerPoseGravityPrior`].
    pub per_pose_gravity_prior: Option<PerPoseGravityPrior>,
    /// Optional per-keyframe absolute position prior. Each observation
    /// adds an axis-weighted residual `(C_w 竏・target)` with Jacobian
    /// `[竏棚 | [C_w]_ﾃ余` (right perturbation, xi-order `[ﾏ・ ﾏ云`). See
    /// [`PositionPrior`].
    pub position_prior: Option<PositionPrior>,
    /// Pairwise relative-pose factors. Each factor lifts an external
    /// relative-pose measurement (IMU pre-integration, wheel odometry,
    /// loop-closure verification, etc.) into the BA solve. See
    /// [`PairwisePoseFactor`].
    pub pairwise_pose_factors: Vec<PairwisePoseFactor>,
    /// Per-keyframe world-frame velocity state. Populated for the
    /// keyframes that participate in any [`ImuPreintegrationFactor`]; the
    /// optimiser jointly refines pose + velocity. Keyframes without an
    /// IMU factor referencing them can leave their velocity slot empty
    /// (the reprojection / pairwise pose / prior factors don't read it).
    pub velocities: BTreeMap<u64, Vector3<f64>>,
    /// Velocity ids held constant during optimisation (mirrors
    /// [`Self::fixed_poses`] / [`Self::fixed_landmarks`]).
    pub fixed_velocities: BTreeSet<u64>,
    /// On-manifold IMU pre-integration factors. Each factor carries a
    /// gravity-compensated `(ﾎ燃, ﾎ牌, ﾎ廃)` produced by
    /// [`crate::imu_preintegration::ImuPreintegrator`] and binds two
    /// keyframes' `(pose, velocity)` states with a 9-vector residual
    /// `[r_R; r_v; r_p]` (Forster 2017 eq. 45-47). The optimiser
    /// linearises the rotation residual via the SO(3) right-Jacobian
    /// inverse.
    pub imu_factors: Vec<ImuPreintegrationFactor>,
    /// Rigid transform from the tracked camera/sensor frame into the IMU
    /// body frame (`T_b<-c`, EuRoC `T_BS`). Visual residuals continue to use
    /// the stored camera poses; IMU residuals compose this extrinsic to obtain
    /// body poses. Identity preserves the historical co-located rig behavior.
    pub imu_body_to_camera: SE3,
    /// Per-keyframe IMU bias state, packing `(bias_gyro, bias_acc)` as a
    /// 6-vector. Populated for the keyframes whose
    /// [`ImuPreintegrationFactor`] should be bias-corrected (the
    /// integration window from `i` to `j` uses `bias[i]` for its
    /// first-order correction). Keyframes without an IMU factor
    /// referencing them, or whose bias has not been registered, fall
    /// back to using the integrator's linearisation bias (no
    /// correction).
    pub biases: BTreeMap<u64, Vector6<f64>>,
    /// Bias ids held constant during optimisation (mirrors
    /// [`Self::fixed_poses`] / [`Self::fixed_velocities`]).
    pub fixed_biases: BTreeSet<u64>,
    /// Bias random-walk priors between consecutive keyframes. See
    /// [`BiasRandomWalkFactor`]; the cost contribution is
    /// `weight ﾂｷ 窶肪_j 竏・b_i窶鳴ｲ` and the Jacobian places `ﾂｱI` against
    /// each non-fixed bias slot.
    pub bias_random_walk_factors: Vec<BiasRandomWalkFactor>,
    /// Dense fixed-lag prior carried from the preceding VI window.
    pub navigation_state_prior: Option<NavigationStatePrior>,
}

/// Configuration for [`BundleAdjustment::optimize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaConfig {
    /// Reuse a rolled-back Legacy sparse pose-diagonal linearization. Keeps
    /// one normal system plus O(P) undamped pose blocks; excludes dense paths.
    pub reuse_rejected_pose_diagonal: bool,
    pub max_iterations: usize,
    pub initial_lambda: Option<f64>,
    pub lambda_increase_factor: f64,
    pub lambda_decrease_factor: f64,
    pub max_lambda: f64,
    pub min_lambda: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
    /// Optional relative accepted-cost-decrease stopping threshold.
    ///
    /// An accepted iteration converges when
    /// `(cost_before - cost_after) / max(abs(cost_before), epsilon)` is below
    /// this value. `None` preserves the historical absolute-cost and step-only
    /// stopping rules.
    pub relative_cost_tolerance: Option<f64>,
    /// Linear-solver backend for the Schur-reduced pose system. The
    /// landmark elimination is always done analytically via per-landmark
    /// `3ﾃ・` block inversion (since `H_LL` is block-diagonal).
    pub linear_solver: LinearSolver,
    /// Robust IRLS kernel applied per-observation to its squared
    /// reprojection residual. [`RobustKernel::None`] runs standard
    /// non-robust BA; `Huber` / `Cauchy` down-weight outliers so a small
    /// number of bad correspondences cannot pull the solution away from
    /// the inlier consensus.
    pub robust_kernel: RobustKernel,
    /// Also refine the shared pinhole intrinsics `(fx, fy, cx, cy)` **jointly**:
    /// when set, [`BundleAdjustment::optimize`] carries the 4 intrinsics as extra
    /// unknowns inside the Schur-complement camera system, co-estimated with the
    /// poses and (eliminated) landmarks 窶・the COLMAP self-calibration formulation.
    /// This is the lever for unknown / inaccurate calibration: a wrong fixed focal
    /// forces a residual onto the poses, and the joint solve lets the camera absorb
    /// it. (The coupled, landmark-eliminated gradient is what makes this work; an
    /// alternating refinement against converged structure cannot move a wrong focal,
    /// because the structure-fixed gradient is ~0.) Only [`CameraModel::Pinhole`]
    /// and [`CameraModel::OpenCv`] (whose lens terms are held fixed unless
    /// [`Self::refine_distortion`] is set) are refined; any other model falls
    /// back to the pose/structure-only solve, which still uses its full lens
    /// model. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub refine_intrinsics: bool,
    /// Additionally self-calibrate the two radial-distortion coefficients
    /// `(k1, k2)` jointly with the intrinsics (the camera block grows from 4 to 6).
    /// Requires `refine_intrinsics`; only applies to a **monocular** pinhole
    /// reconstruction (rectified stereo is already undistorted). The coefficients
    /// are appended to `Camera::params` as `[fx, fy, cx, cy, k1, k2]`. **`false`
    /// by default.**
    ///
    /// An [`CameraModel::OpenCv`] camera (`[fx, fy, cx, cy, k1, k2, p1, p2]`) is
    /// refined in the same layout: its `(k1, k2)` are refined and its tangential
    /// `(p1, p2)` stay fixed unless [`Self::refine_tangential_distortion`] is set.
    pub refine_distortion: bool,
    /// Additionally self-calibrate the tangential coefficients `(p1, p2)` (the
    /// camera block grows from 6 to 8). Requires `refine_intrinsics` and
    /// `refine_distortion`, and is ignored without them. A `Pinhole` camera is
    /// promoted to [`CameraModel::OpenCv`] `[fx, fy, cx, cy, k1, k2, p1, p2]`
    /// (starting from `p1 = p2 = 0`, which projects exactly like the radial
    /// pinhole), since that is the layout that carries tangential terms.
    /// Decentering is weakly observable and trades against the principal point
    /// on small or forward-moving scenes, so it is opt-in. **`false` by
    /// default** (the joint solve is then bit-identical to before).
    pub refine_tangential_distortion: bool,
    /// Constrain the refined focal length to `fx == fy` (one shared focal, as
    /// COLMAP's `SIMPLE_*` / `RADIAL` models and every square-pixel sensor do).
    /// Without it `fx` and `fy` move independently, which lets weakly
    /// observable motion (e.g. near-pure forward translation) trade a spurious
    /// aspect ratio against structure. The constraint is applied to the reduced
    /// camera system (`fx`/`fy` rows and columns summed, the shared update
    /// written to both), so the Jacobian builder is unchanged. Requires
    /// `refine_intrinsics`; a camera whose `fx != fy` starts from their mean.
    /// **`false` by default** (the joint solve is then bit-identical to before).
    pub shared_focal: bool,
    /// Run the per-observation assembly, per-landmark Schur reduction, and
    /// back-substitution loops of [`BundleAdjustment::optimize_weighted`]'s
    /// Levenberg-Marquardt iteration on the `rayon` pool (see the module's
    /// "Parallelism" section). The result is bit-identical to the serial
    /// path at any thread count 窶・this only changes *how* the normal
    /// equations are computed, never the summation order 窶・so it is safe to
    /// flip independently of everything else in this config. Small problems
    /// stay serial even when this is set (see `PARALLEL_MIN_OBSERVATIONS` /
    /// `PARALLEL_MIN_LANDMARKS`). Only consumed by `optimize_weighted`
    /// (the plain pose/structure/IMU solve); `optimize_joint_intrinsics`
    /// ignores it. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub parallel: bool,
    /// Solve the pure-visual bundle adjustment with the matrix-free
    /// implicit-Schur PCG backend (see [`BundleAdjustment::optimize_matrix_free`])
    /// instead of the dense/sparse block-Cholesky Schur reduction.
    ///
    /// The default reduced pose system fills in when tracks are long, so one LM
    /// linear solve becomes `O(pose^3)` and a large reconstruction can spend
    /// hours in `linear_solve`. The matrix-free operator eliminates the
    /// landmarks implicitly and its cost tracks the observation count.
    /// Ineligible problems (intrinsics/distortion refinement, non-visual states
    /// or priors, a fully fixed pose set, or a missing gauge anchor) fall back to
    /// the ordinary solver, which keeps `true` safe as a general default-off
    /// opt-in. **`false` by default** (the ordinary solve is then unchanged).
    pub matrix_free_ba: bool,
}

impl Default for BaConfig {
    fn default() -> Self {
        Self {
            reuse_rejected_pose_diagonal: false,
            max_iterations: 20,
            initial_lambda: Some(1e-4),
            lambda_increase_factor: 10.0,
            lambda_decrease_factor: 0.1,
            max_lambda: 1e12,
            min_lambda: 1e-9,
            step_tolerance: 1e-7,
            cost_tolerance: 1e-9,
            relative_cost_tolerance: None,
            linear_solver: LinearSolver::Dense,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            refine_distortion: false,
            refine_tangential_distortion: false,
            shared_focal: false,
            parallel: false,
            matrix_free_ba: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaIterationStats {
    pub iteration: usize,
    pub cost_before: f64,
    pub cost_after: f64,
    pub max_pose_step: f64,
    pub max_landmark_step: f64,
    pub lambda: f64,
    pub step_accepted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BaResult {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<BaIterationStats>,
    pub converged: bool,
}

/// Options for the opt-in matrix-free pure-visual BA entry point.
///
/// This is deliberately separate from [`BaConfig`]: adding a solver variant
/// there would change the default path and would make existing callers
/// accidentally opt into a new numerical backend.  The matrix-free entry
/// point requires a finite, positive LM start value and uses PCG for each
/// reduced pose solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixFreeBaOptions {
    /// Maximum PCG iterations for one LM linear solve.
    pub max_pcg_iterations: usize,
    /// Relative PCG residual tolerance.
    pub pcg_relative_tolerance: f64,
    /// Absolute PCG residual tolerance.
    pub pcg_absolute_tolerance: f64,
}

impl Default for MatrixFreeBaOptions {
    fn default() -> Self {
        Self {
            max_pcg_iterations: 128,
            pcg_relative_tolerance: 1.0e-12,
            pcg_absolute_tolerance: 1.0e-12,
        }
    }
}

/// Options for the opt-in column-equilibrated matrix-free entry point.
///
/// The diagonal bounds are deliberately private fixed policy constants.  The
/// nested PCG options reuse the existing matrix-free option shape without
/// changing its defaults or adding a scaling switch to the legacy API.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MatrixFreeBaColumnScalingOptions {
    pub pcg: MatrixFreeBaOptions,
}

/// Per-LM-iteration diagnostics returned by [`BundleAdjustment::optimize_matrix_free`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaIterationStats {
    pub iteration: usize,
    pub pcg_iterations: Option<usize>,
    pub pcg_residual_norm: Option<f64>,
    pub pcg_target: Option<f64>,
    pub pcg_failure: Option<String>,
}

/// Result of the opt-in matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaResult {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: Vec<BaIterationStats>,
    pub matrix_free_iterations: Vec<MatrixFreeBaIterationStats>,
    pub converged: bool,
}

/// Scalar accounting for one normal-system equilibration.  PCG residuals in
/// the nested [`MatrixFreeBaResult`] are in scaled coordinates; physical pose
/// and landmark step norms remain in its ordinary LM iteration statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatrixFreeBaColumnScalingIterationStats {
    pub iteration: usize,
    /// Minimum of the clamped normal-equation diagonal values `d_j`, not of
    /// the transforms `1/sqrt(d_j)`.
    pub minimum_diagonal: f64,
    /// Maximum of the clamped normal-equation diagonal values `d_j`, not of
    /// the transforms `1/sqrt(d_j)`.
    pub maximum_diagonal: f64,
    pub clamped_to_minimum: usize,
    pub clamped_to_maximum: usize,
}

/// Result of the opt-in column-equilibrated matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaColumnScalingResult {
    pub ba: MatrixFreeBaResult,
    pub scaling_iterations: Vec<MatrixFreeBaColumnScalingIterationStats>,
}

/// Scalar per-LM-iteration accounting for adaptive column-scaled damping.
/// Prediction is evaluated in the current scaled coordinates; it is not a
/// full backward-error diagnostic and does not retain normal-equation state.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaAdaptiveDampingIterationStats {
    pub iteration: usize,
    pub solve_lambda: f64,
    pub next_lambda: f64,
    pub predicted_undamped_squared_decrease: Option<f64>,
    pub actual_cost_decrease: Option<f64>,
    pub rho: Option<f64>,
    pub cost_gate: Option<bool>,
    pub feasibility_gate: Option<bool>,
    pub nonprojectable_before: usize,
    pub nonprojectable_after: Option<usize>,
    pub accepted: bool,
    pub reason: String,
}

/// Result of the opt-in adaptive column-scaled matrix-free BA run.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaAdaptiveDampingResult {
    pub ba: MatrixFreeBaResult,
    pub scaling_iterations: Vec<MatrixFreeBaColumnScalingIterationStats>,
    pub adaptive_iterations: Vec<MatrixFreeBaAdaptiveDampingIterationStats>,
}

/// Additive options for the bounded true-residual restart diagnostic.
///
/// `max_restarts_per_solve` is intentionally limited to zero or one.  The
/// PCG iteration budget in [`MatrixFreeBaOptions`] is global to each reduced
/// solve and is never reset by a restart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatrixFreeBaRestartOptions {
    pub max_restarts_per_solve: usize,
}

/// Per-LM-iteration diagnostics for the bounded restart entry point.
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaRestartIterationStats {
    pub iteration: usize,
    /// Total alpha iterations used by this PCG solve.  This value is never
    /// reset when a residual restart occurs.
    pub pcg_iterations: Option<usize>,
    /// Number of explicit true-residual checks performed.
    pub true_residual_rechecks: usize,
    /// Number of those checks whose norm exceeded the configured target.  This
    /// includes checks at the iteration cap; it is not a count of
    /// restart-eligible checks.
    pub failed_true_residual_rechecks: usize,
    pub restarts: usize,
    pub terminal_failure: Option<String>,
}

/// Result of [`BundleAdjustment::optimize_matrix_free_with_restart`].
#[derive(Debug, Clone, PartialEq)]
pub struct MatrixFreeBaRestartResult {
    pub ba: MatrixFreeBaResult,
    pub restart_iterations: Vec<MatrixFreeBaRestartIterationStats>,
}

/// Failure from the opt-in matrix-free pure-visual BA entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixFreeBaError {
    /// The problem contains a factor/state that the pure-visual operator does
    /// not assemble (for example IMU, navigation, or a non-visual prior).
    Ineligible(&'static str),
    /// The matrix-free LM/PCG options are not finite or cannot make progress.
    InvalidConfiguration(&'static str),
    /// The existing BA input validation rejected the problem.
    Ba(BaError),
    /// The reduced operator or PCG solve failed.  The textual payload is a
    /// deterministic diagnostic representation of the private numerical error.
    LinearSolve {
        iteration: usize,
        diagnostic: String,
    },
}

impl std::fmt::Display for MatrixFreeBaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ineligible(reason) => write!(f, "matrix-free BA is ineligible: {reason}"),
            Self::InvalidConfiguration(reason) => {
                write!(f, "invalid matrix-free BA configuration: {reason}")
            }
            Self::Ba(error) => write!(f, "matrix-free BA input error: {error}"),
            Self::LinearSolve {
                iteration,
                diagnostic,
            } => write!(
                f,
                "matrix-free reduced solve failed at iteration {iteration}: {diagnostic}"
            ),
        }
    }
}

impl std::error::Error for MatrixFreeBaError {}

#[cfg(test)]
mod column_scaling_tests;

#[cfg(test)]
mod lm_step_quality_tests;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaCostBreakdown {
    pub total: f64,
    pub visual: f64,
    pub imu: f64,
    pub bias_random_walk: f64,
    pub navigation_prior: f64,
    pub other_structural: f64,
    pub imu_normalized_squared_residual_per_dof: Option<f64>,
    pub imu_rotation_residual_rms_rad: Option<f64>,
    pub imu_velocity_residual_rms_mps: Option<f64>,
    pub imu_position_residual_rms_meters: Option<f64>,
}

/// Result of [`BundleAdjustment::optimize_gnc`].
#[derive(Debug, Clone, PartialEq)]
pub struct BaGncResult {
    /// Non-robust reprojection cost at the input estimate.
    pub initial_cost: f64,
    /// GNC-weighted reprojection cost at the recovered estimate (every
    /// observation scaled by its final `w`).
    pub final_cost: f64,
    /// Reprojection cost over the classified inliers only (outliers
    /// contribute nothing), using the `0.5` weight threshold.
    pub inlier_cost: f64,
    /// The inlier scale `c` (pixels) the solve actually used: the configured
    /// [`GncConfig::c`] verbatim, or — under [`GncConfig::auto_scale`] — the
    /// MAD estimate (floored at the configured `c`).
    pub inlier_scale: f64,
    /// Number of reprojection observations (monocular + stereo) the weight
    /// vector covers.
    pub observation_count: usize,
    /// GNC outer (μ) levels actually executed.
    pub outer_iterations: usize,
    /// Whether the μ schedule reached its terminal level.
    pub converged: bool,
    /// Final per-observation Black-Rangarajan weight `w ∈ [0,1]`, indexed
    /// monocular, rectified stereo, then general stereo. `NaN` marks an observation that could
    /// not be evaluated at the recovered estimate. Near-zero finite entries
    /// are the rejected outliers.
    pub observation_weights: Vec<f64>,
}

impl BaGncResult {
    /// Count of observations classified as inliers (`w ≥ threshold`).
    /// `NaN` (un-evaluable) observations are excluded from both counts.
    pub fn inlier_count(&self, threshold: f64) -> usize {
        self.observation_weights
            .iter()
            .filter(|w| w.is_finite() && **w >= threshold)
            .count()
    }

    /// Count of observations classified as outliers (`w < threshold`).
    pub fn outlier_count(&self, threshold: f64) -> usize {
        self.observation_weights
            .iter()
            .filter(|w| w.is_finite() && **w < threshold)
            .count()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaError {
    NoPoses,
    NoLandmarks,
    NoObservations,
    /// Every pose AND every landmark is fixed, so there is nothing to
    /// optimize. (Pose-only or landmark-only BA is allowed.)
    AllPosesFixed,
    MissingPose(u64),
    MissingLandmark(u64),
    /// Camera model is not pinhole (multi-model BA is a future extension).
    UnsupportedCameraModel,
    /// One or more [`BaStereoObservation`]s were added but
    /// [`BundleAdjustment::stereo_baseline`] is missing or non-positive.
    MissingStereoBaseline,
    /// The external visual-weight vector does not match the flattened visual
    /// observation count (mono, rectified stereo, general stereo).
    ObservationWeightCount {
        expected: usize,
        actual: usize,
    },
    /// An external visual weight is negative, NaN, or infinite.
    InvalidObservationWeight(usize),
    /// A fixed-rotation diagnostic supplied a vector that is not aligned with
    /// the pose vector being optimized.
    InvalidFixedRotationCount {
        expected: usize,
        actual: usize,
    },
    /// Confidence-weighted joint intrinsics refinement is not implemented.
    ObservationWeightsWithIntrinsicsRefinement,
    /// Reduced camera system was singular even after λ damping. Usually
    /// means the gauge is under-fixed (e.g., monocular without enough
    /// fixed poses or landmarks to remove scale).
    SingularSystem,
}

impl std::fmt::Display for BaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BaError::NoPoses => write!(f, "bundle adjustment has no poses"),
            BaError::NoLandmarks => write!(f, "bundle adjustment has no landmarks"),
            BaError::NoObservations => write!(f, "bundle adjustment has no observations"),
            BaError::AllPosesFixed => write!(f, "every pose is fixed; nothing to optimize"),
            BaError::MissingPose(id) => write!(f, "observation references unknown pose {id}"),
            BaError::MissingLandmark(id) => {
                write!(f, "observation references unknown landmark {id}")
            }
            BaError::UnsupportedCameraModel => {
                write!(f, "only pinhole camera models are supported")
            }
            BaError::MissingStereoBaseline => {
                write!(f, "stereo observations require a positive stereo_baseline")
            }
            BaError::ObservationWeightCount { expected, actual } => write!(
                f,
                "observation weight count mismatch: expected {expected}, got {actual}"
            ),
            BaError::InvalidObservationWeight(index) => write!(
                f,
                "observation weight at index {index} must be finite and non-negative"
            ),
            BaError::InvalidFixedRotationCount { expected, actual } => write!(
                f,
                "fixed-rotation pose count mismatch: expected {expected}, got {actual}"
            ),
            BaError::ObservationWeightsWithIntrinsicsRefinement => write!(
                f,
                "observation weights are not supported with joint intrinsics refinement"
            ),
            BaError::SingularSystem => write!(f, "reduced camera system is singular"),
        }
    }
}

impl std::error::Error for BaError {}

#[cfg(test)]
mod mono_projection_tests;

#[test]
fn qr_shadow_caps_reject_oversized_diagnostics() {
    assert!(qr_shadow_dimensions_allowed(60, 64, 1024, 20000));
    for dimensions in [
        (61, 64, 1024, 20000),
        (60, 65, 1024, 20000),
        (60, 64, 1025, 20000),
        (60, 64, 1024, 20001),
        (60, 64, 1024, 0),
        (usize::MAX, usize::MAX, usize::MAX, usize::MAX),
    ] {
        assert!(!qr_shadow_dimensions_allowed(
            dimensions.0,
            dimensions.1,
            dimensions.2,
            dimensions.3
        ));
    }
}

/// `LocalRefiner` implementation that runs windowed bundle adjustment on
/// a staged map update. Existing keyframes and landmarks in the window
/// (already in `VisualMap`) are added as fixed gauge; the newly-staged
/// keyframe poses and landmark positions are the BA variables. Observations
/// from both the existing window and the staged update feed the residual.
///
/// Refined poses / landmarks are written back into the staged update so
/// subsequent `apply_to(&mut map)` lands the BA-corrected values. The
/// existing map is never mutated by this refiner.
#[derive(Debug, Clone, PartialEq)]
pub struct BundleAdjustmentRefiner {
    pub config: BaConfig,
}

impl BundleAdjustmentRefiner {
    pub const fn new(config: BaConfig) -> Self {
        Self { config }
    }
}

impl Default for BundleAdjustmentRefiner {
    fn default() -> Self {
        Self::new(BaConfig::default())
    }
}

impl LocalRefiner for BundleAdjustmentRefiner {
    fn refine(
        &self,
        map: &VisualMap,
        local_window: &LocalMapWindow,
        staged_update: &mut StagedMapUpdate,
    ) -> LocalRefinementResult {
        // Pick a camera. Prefer the camera attached to the first staged
        // keyframe; fall back to any camera in the existing window.
        let camera = staged_update
            .keyframes
            .iter()
            .find_map(|kf| map.cameras.get(&kf.frame.camera_id).cloned())
            .or_else(|| {
                local_window
                    .keyframe_ids
                    .iter()
                    .find_map(|id| map.keyframes.get(id))
                    .and_then(|kf| map.cameras.get(&kf.frame.camera_id).cloned())
            });
        let Some(camera) = camera else {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        };

        let mut ba = BundleAdjustment::new(camera);

        // Treat staged keyframes / landmarks as variable. Everything else
        // in the window is already in `map` and serves as a fixed gauge.
        // The local window typically already includes the newly-staged
        // keyframe (the local-mapping pipeline inserts it into a working
        // map before computing the window), so we must skip those when
        // adding fixed poses — otherwise the BA variable would become a
        // fixed gauge and never move.
        let staged_kf_ids: std::collections::BTreeSet<u64> = staged_update
            .keyframes
            .iter()
            .map(|kf| kf.frame.id)
            .collect();
        let staged_lm_ids: std::collections::BTreeSet<u64> =
            staged_update.landmarks.iter().map(|lm| lm.id).collect();

        // Fixed gauge: window keyframes that are NOT in the staged update.
        for &kf_id in &local_window.keyframe_ids {
            if staged_kf_ids.contains(&kf_id) {
                continue;
            }
            let Some(kf) = map.keyframes.get(&kf_id) else {
                continue;
            };
            let Some(pose) = kf.frame.pose.clone() else {
                continue;
            };
            ba.add_pose(kf_id, pose);
            ba.fix_pose(kf_id);
        }
        // Variable: newly-staged keyframe poses.
        for kf in &staged_update.keyframes {
            let id = kf.frame.id;
            let Some(pose) = kf.frame.pose.clone() else {
                continue;
            };
            ba.add_pose(id, pose);
        }

        // Fixed gauge: window landmarks that are NOT in the staged update.
        for &lm_id in &local_window.landmark_ids {
            if staged_lm_ids.contains(&lm_id) {
                continue;
            }
            let Some(lm) = map.landmarks.get(&lm_id) else {
                continue;
            };
            ba.add_landmark(lm_id, lm.position);
            ba.fix_landmark(lm_id);
        }
        // Variable: newly-staged landmarks.
        for lm in &staged_update.landmarks {
            ba.add_landmark(lm.id, lm.position);
        }

        // Observations: existing keyframes' observations of fixed landmarks
        // (anchor the gauge), plus new staged observations.
        for &kf_id in &local_window.keyframe_ids {
            let Some(kf) = map.keyframes.get(&kf_id) else {
                continue;
            };
            for obs in &kf.observations {
                if ba.poses.contains_key(&obs.frame_id)
                    && ba.landmarks.contains_key(&obs.landmark_id)
                {
                    ba.add_observation(BaObservation {
                        keyframe_id: obs.frame_id,
                        landmark_id: obs.landmark_id,
                        xy: obs.xy,
                    });
                }
            }
        }
        for obs in &staged_update.observations {
            if ba.poses.contains_key(&obs.frame_id) && ba.landmarks.contains_key(&obs.landmark_id) {
                ba.add_observation(BaObservation {
                    keyframe_id: obs.frame_id,
                    landmark_id: obs.landmark_id,
                    xy: obs.xy,
                });
            }
        }

        // No observations to optimize against → nothing to refine.
        if ba.observations.is_empty() {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }
        // Every variable pose is fixed (or no variable poses exist) AND
        // every variable landmark is fixed → nothing to optimize. The BA
        // would still report `AllPosesFixed`; just skip cleanly.
        let has_variable_pose = ba.poses.keys().any(|id| !ba.fixed_poses.contains(id));
        let has_variable_landmark = ba
            .landmarks
            .keys()
            .any(|id| !ba.fixed_landmarks.contains(id));
        if !has_variable_pose && !has_variable_landmark {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }

        if ba.optimize(&self.config).is_err() {
            return LocalRefinementResult::skipped(LocalRefinementReason::Noop);
        }

        // Write back refined values into staged_update. Fixed entries are
        // never modified.
        let mut keyframe_count = 0usize;
        for kf in staged_update.keyframes.iter_mut() {
            let id = kf.frame.id;
            if ba.fixed_poses.contains(&id) {
                continue;
            }
            if let Some(refined) = ba.poses.get(&id).cloned() {
                kf.frame.pose = Some(refined);
                keyframe_count += 1;
            }
        }
        let mut landmark_count = 0usize;
        for lm in staged_update.landmarks.iter_mut() {
            if ba.fixed_landmarks.contains(&lm.id) {
                continue;
            }
            if let Some(refined) = ba.landmarks.get(&lm.id).copied() {
                lm.position = refined;
                landmark_count += 1;
            }
        }

        LocalRefinementResult {
            refined: keyframe_count > 0 || landmark_count > 0,
            reason: LocalRefinementReason::Refined,
            keyframe_count,
            landmark_count,
        }
    }
}

/// Private matrix-free Schur backend.
///
/// The backend consumes an assembled pure-visual `PoseDiagonal` system and
/// never constructs pose-pair blocks, triplets, or a Cholesky factor.  Its
/// public entry point remains on [`BundleAdjustment`]; keeping the numerical
/// implementation here avoids a second operator implementation in tests.
mod implicit_schur;

#[cfg(test)]
mod matrix_free_ba_api_tests;

#[cfg(test)]
mod matrix_free_real_oracle_tests;

#[cfg(test)]
mod visual_jacobian_audit_tests;

#[cfg(test)]
mod imu_gradient_tests;

/// Tests for [`BaConfig::parallel`] (see the module's "Parallelism"
/// section): the serial and parallel assembly / Schur-reduction /
/// back-substitution paths must agree, and the parallel path must be
/// deterministic. The synthetic problem is sized past
/// `PARALLEL_MIN_OBSERVATIONS` / `PARALLEL_MIN_LANDMARKS` so these tests
/// actually exercise the parallel dispatch rather than falling through the
/// work gate to the serial loops.
#[cfg(test)]
mod parallel_ba_tests;
