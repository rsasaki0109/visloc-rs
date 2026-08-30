//! Bundle adjustment with Schur-complement landmark elimination.
//!
//! Optimizes camera poses jointly with landmark positions to minimize the
//! sum of squared 2D reprojection residuals. Pinhole intrinsics are held
//! fixed; the variables are pose `T_world_to_camera` (6 DoF, right
//! perturbation `T ← T · Exp(ξ)` with `ξ = [ρ; ω]`) per non-fixed pose
//! and `X_w` (3 DoF) per non-fixed landmark.
//!
//! The Schur complement of the block-diagonal landmark Hessian `H_LL`
//! reduces the linear system to one of pose-only size `(6P) × (6P)` per
//! iteration, regardless of how many landmarks the scene has, then
//! back-substitutes for the landmark updates. Each iteration is a
//! Levenberg-Marquardt step with optional cost-rejection.
//!
//! Gauge fixing is the caller's responsibility: monocular BA has 7 DoF
//! gauge freedom (6 SE(3) + 1 scale). At minimum fix the first pose
//! (anchor) and one of the following to remove scale: a second pose, a
//! second landmark, or a known-distance pair. Rectified-stereo BA (any
//! [`BaStereoObservation`] present) has only 6 DoF gauge freedom — the
//! baseline anchors metric scale — so a single fixed pose is enough.
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
//!   pose and landmark estimate — it touches no shared state — so it is
//!   computed on the rayon pool in fixed-size chunks
//!   ([`PARALLEL_OBSERVATION_CHUNK`]), collected into a plain per-chunk
//!   `Vec`; the actual `+=` scatter into `h_pp` / `b_p` / the per-landmark
//!   blocks stays a single serial pass over each chunk's precomputed
//!   contributions, *in the original per-observation order*.
//! - **Schur reduction** (`solve_step`'s per-landmark `S -= H_PL H_LL⁻¹
//!   H_PLᵀ` loop): each landmark's `3×3` factorization is independent and
//!   computed directly in parallel (disjoint output slots, no merge needed);
//!   the pose-pair contributions it produces are computed the same chunked
//!   way as assembly, collecting each landmark's `(Vec<(p,q,block)>,
//!   Vec<(p,upd)>)` pair per chunk and then flattening/merging into the
//!   shared reduced system `s` / `b_reduced` by a serial pass over each
//!   chunk, in landmark-ascending order.
//! - **Back-substitution** (`solve_step`'s per-landmark `δ_L` loop): each
//!   landmark writes only its own 3 rows of `delta_l`, so this is
//!   embarrassingly parallel with no merge step at all.
//!
//! Unlike [`crate::block_cholesky`]'s intra-column path — which reassociates
//! a floating-point sum across contributors and is therefore only
//! deterministic *to rounding* — every merge here reproduces the exact
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
//! Schur reduction (a separate, less-used code path — self-calibration BA is
//! opt-in and typically run on far smaller problems than a full-sequence
//! pose/structure solve) and the cost-evaluation passes (`robust_cost_weighted`
//! / `reprojection_squared_residuals`, shared by many callers beyond the LM
//! loop, so gating them on `BaConfig` would require threading the flag
//! through call sites that have nothing to do with this optimizer).

use std::collections::{BTreeMap, BTreeSet};

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
use crate::{solve_normal_equations, LinearSolver, PoseGraphError, RobustKernel};

/// Keep solver-step diagnostics opt-in even when a caller already enables a
/// higher-level SFM trace.  This flag is intentionally read here rather than
/// threaded through [`BaConfig`], so the public/default optimizer state stays
/// byte-identical and a diagnostic cannot accidentally become a production
/// behavior switch.
fn ba_step_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
        && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
}

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
/// right-camera pose (it is implicitly the left's translated by `b·x̂`) and
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

/// Rotation-alignment gravity prior on every non-fixed pose.
///
/// Adds a 3-vector residual `r = R_wc · g_world − g_camera_observed` per
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
/// prior — that would require a translation/altitude prior fed from
/// IMU velocity or GNSS, which lives outside [`BundleAdjustment`] in
/// its current form.
#[derive(Debug, Clone, PartialEq)]
pub struct GravityPrior {
    /// Gravity direction in world frame. Magnitude defines the
    /// residual's natural scale; using the physical 9.81 m/s² keeps the
    /// per-pose residual in the same order of magnitude as a pixel
    /// reprojection residual, so a default Huber `delta ≈ 3` does not
    /// over- or under-weight the prior.
    pub g_world: Vector3<f64>,
    /// Gravity direction observed (or assumed) in camera frame for
    /// every pose. For a level prior this matches the camera-frame
    /// direction of `g_world` at the anchor pose, e.g. `(0, 9.81, 0)`.
    pub g_camera_observed: Vector3<f64>,
    /// Scalar weight applied to the gravity contribution. The cost
    /// added per pose is `weight · ‖r‖²` and the normal-equations
    /// contribution is `weight · Jᵀ J` / `weight · Jᵀ r`. A weight of
    /// `1.0` makes a 9.81 m/s² gravity residual count comparably to a
    /// single 9.81 px reprojection residual; lower this for a softer
    /// prior, raise it for a stiffer one.
    pub weight: f64,
}

/// Per-keyframe observation of the gravity direction in camera
/// coordinates. Each entry constrains
/// `R_wc · g_world ≈ g_camera_observed` at the named keyframe; the
/// residual and Jacobian shape are identical to [`GravityPrior`]'s
/// global pose-independent variant, except the observation is sourced
/// per-keyframe rather than shared across all poses.
///
/// The intended source of `g_camera_observed` is an accelerometer
/// sample (or a low-pass-filtered window of samples) at the keyframe
/// timestamp, rotated into the camera frame via the body→camera
/// extrinsic. Unlike [`PositionPrior`], which can leak ground-truth
/// poses when fed from GNSS/INS-fused trajectories, a properly-
/// generated per-keyframe gravity prior is a true online sensor
/// observation — the same signal a deployed VIO would consume.
#[derive(Debug, Clone, PartialEq)]
pub struct PerPoseGravityObservation {
    /// Keyframe whose pose rotation is being constrained. The pose
    /// must already be added to [`BundleAdjustment`]. Fixed poses
    /// still contribute to the cost report but generate no Jacobian
    /// rows because they have no Hessian slot.
    pub keyframe_id: u64,
    /// Observed gravity direction in camera frame at this keyframe.
    /// Magnitude should match [`PerPoseGravityPrior::g_world`] (e.g.
    /// `9.81 m/s²` for a physical accelerometer-derived observation),
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
    pub fn new(keyframe_id: u64, g_camera_observed: Vector3<f64>) -> Self {
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
/// observation shared across all poses) — it accepts per-keyframe
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
    /// per-obs `1.0` makes a 9.81 m/s² gravity residual count
    /// comparably to a single 9.81 px reprojection residual; lower
    /// the global scale for a softer prior, raise for a stiffer one.
    /// The per-observation field stays neutral unless the upstream
    /// sensor model emits inverse-variance weights.
    pub weight: f64,
}

impl PerPoseGravityPrior {
    pub fn new(g_world: Vector3<f64>, weight: f64) -> Self {
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
/// current estimate of `−Rᵀ · t`. Designed for translation-domain
/// priors fed from GNSS, an external altimeter, or — in evaluation
/// scenarios — ground-truth poses; the prior constrains TRANSLATION
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
    /// Per-axis weights in the cost `Σ wᵢ · (Cᵢ − targetᵢ)²` and the
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
/// `T_j · T_iⁱ` (`world_to_camera_j` of the "to" keyframe composed with
/// the inverse of the "from" keyframe). The residual is the SE(3) log
/// of the disagreement:
///
/// ```text
/// r = log(measurement⁻¹ · T_j · T_iⁱ)  ∈ ℝ⁶
/// ```
///
/// Jacobians under right-perturbation `T ← T · exp(δ)`:
///
/// - `∂r / ∂δ_j =  Ad(T_i)`
/// - `∂r / ∂δ_i = −Ad(T_i)`
///
/// This is the same Jacobian shape used by
/// `PoseGraph::optimize_se3_iterative`; the factor lifts those edges
/// into [`BundleAdjustment`] so visual residuals and external-sensor
/// pose deltas can be jointly optimised in a single LM solve. Full
/// IMU pre-integration with velocity/bias states is a future
/// extension; this v1 factor assumes the pre-integrator has already
/// produced a single `(Δp, ΔR)` pair plus a scalar weight.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwisePoseFactor {
    /// "From" keyframe id (the one Ad(T_from) is computed about).
    pub keyframe_id_from: u64,
    /// "To" keyframe id.
    pub keyframe_id_to: u64,
    /// Measured relative pose `T_meas` such that, at convergence,
    /// `T_meas = T_j · T_iⁱ` where `T_i` and `T_j` are the BA poses
    /// for `keyframe_id_from` and `keyframe_id_to` respectively.
    pub measurement: Pose,
    /// Scalar weight (sqrt-information squared). The cost added is
    /// `weight · ‖r‖²` so `weight = 1 / σ²` for an isotropic
    /// measurement with standard deviation `σ` (per-axis). Anisotropic
    /// 6×6 sqrt-information matrices are deferred to a future
    /// extension.
    pub weight: f64,
}

/// Bias random-walk factor between two keyframes' 6-vector IMU
/// biases. Adds the residual `r = b_j − b_i` with independent gyro and
/// accelerometer weights. Use this to keep neighbouring
/// keyframes' biases close to each other when the IMU factor's data-
/// driven Jacobian leaves some bias DoFs unobservable in isolation
/// (e.g., gyro biases on a straight-line trajectory).
///
/// Both endpoint biases must be registered via
/// [`BundleAdjustment::add_bias`] for the factor to contribute. If
/// either side has a non-fixed bias slot, the factor adds its 6×6
/// Jacobian (`J_i = −I`, `J_j = I`) to the normal equations; fully-
/// fixed endpoints still contribute to the cost report but no
/// Jacobian rows.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasRandomWalkFactor {
    /// "From" keyframe id (the bias on the `−I` side of the Jacobian).
    pub keyframe_id_from: u64,
    /// "To" keyframe id (the bias on the `+I` side).
    pub keyframe_id_to: u64,
    /// Gyroscope-bias sqrt-information squared. A typical value is
    /// `1 / (σ_bg² · Δt_ij)` for continuous random-walk density `σ_bg`.
    pub weight_gyro: f64,
    /// Accelerometer-bias sqrt-information squared. A typical value is
    /// `1 / (σ_ba² · Δt_ij)` for continuous random-walk density `σ_ba`.
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
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            couple_rotation: true,
        }
    }

    pub fn with_rotation_coupling(mut self, couple_rotation: bool) -> Self {
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
    /// `+stereo_baseline · x̂` of the left in the left-camera frame. Required
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
    /// adds an axis-weighted residual `(C_w − target)` with Jacobian
    /// `[−I | [C_w]_×]` (right perturbation, xi-order `[ρ; ω]`). See
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
    /// gravity-compensated `(ΔR, Δv, Δp)` produced by
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
    /// `weight · ‖b_j − b_i‖²` and the Jacobian places `±I` against
    /// each non-fixed bias slot.
    pub bias_random_walk_factors: Vec<BiasRandomWalkFactor>,
    /// Dense fixed-lag prior carried from the preceding VI window.
    pub navigation_state_prior: Option<NavigationStatePrior>,
}

impl BundleAdjustment {
    pub fn new(camera: Camera) -> Self {
        Self {
            poses: BTreeMap::new(),
            landmarks: BTreeMap::new(),
            observations: Vec::new(),
            stereo_observations: Vec::new(),
            general_stereo_observations: Vec::new(),
            camera,
            fixed_poses: BTreeSet::new(),
            fixed_pose_rotations: BTreeSet::new(),
            fixed_landmarks: BTreeSet::new(),
            stereo_baseline: None,
            gravity_prior: None,
            per_pose_gravity_prior: None,
            position_prior: None,
            pairwise_pose_factors: Vec::new(),
            velocities: BTreeMap::new(),
            fixed_velocities: BTreeSet::new(),
            imu_factors: Vec::new(),
            imu_body_to_camera: SE3::identity(),
            biases: BTreeMap::new(),
            fixed_biases: BTreeSet::new(),
            bias_random_walk_factors: Vec::new(),
            navigation_state_prior: None,
        }
    }

    pub fn set_navigation_state_prior(&mut self, prior: NavigationStatePrior) {
        self.navigation_state_prior = Some(prior);
    }

    /// Linearize the navigation-only portion at the current estimate.
    /// Used by the fixed-lag VI stage to Schur-marginalize a state leaving
    /// the window. Landmark-bearing problems are deliberately rejected here:
    /// the boundary prior owns only inertial-chain information, preventing
    /// retained visual observations from being counted both in the prior and
    /// again in the next window.
    pub(crate) fn linearized_navigation_system(&self) -> Option<NavigationLinearization> {
        if !self.landmarks.is_empty()
            || !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty()
        {
            return None;
        }
        let intrinsics = self.intrinsics()?;
        let mut pose_index = BTreeMap::new();
        for &id in self.poses.keys() {
            if !self.fixed_poses.contains(&id) {
                let next = pose_index.len();
                pose_index.insert(id, next);
            }
        }
        let mut velocity_index = BTreeMap::new();
        for &id in self.velocities.keys() {
            if !self.fixed_velocities.contains(&id) {
                let next = velocity_index.len();
                velocity_index.insert(id, next);
            }
        }
        let mut bias_index = BTreeMap::new();
        for &id in self.biases.keys() {
            if !self.fixed_biases.contains(&id) {
                let next = bias_index.len();
                bias_index.insert(id, next);
            }
        }
        let system = build_normal_equations(
            self,
            &intrinsics,
            &pose_index,
            &BTreeMap::new(),
            &velocity_index,
            &bias_index,
            &RobustKernel::None,
            None,
            false,
        );
        let ordered_ids = |index: &BTreeMap<u64, usize>| {
            let mut ids: Vec<(usize, u64)> = index.iter().map(|(id, slot)| (*slot, *id)).collect();
            ids.sort_unstable();
            ids.into_iter().map(|(_, id)| id).collect()
        };
        Some(NavigationLinearization {
            pose_ids: ordered_ids(&pose_index),
            velocity_ids: ordered_ids(&velocity_index),
            bias_ids: ordered_ids(&bias_index),
            information: system.h_pp,
            gradient: system.b_p,
        })
    }

    /// Append a relative-pose factor between two keyframes. See
    /// [`PairwisePoseFactor`] for semantics.
    pub fn add_pairwise_pose_factor(&mut self, factor: PairwisePoseFactor) {
        self.pairwise_pose_factors.push(factor);
    }

    /// Register an initial world-frame velocity for the given keyframe.
    /// Required for any keyframe referenced by an
    /// [`ImuPreintegrationFactor`] — the velocity becomes a BA variable
    /// (unless also passed to [`Self::fix_velocity`]).
    pub fn add_velocity(&mut self, id: u64, velocity: Vector3<f64>) {
        self.velocities.insert(id, velocity);
    }

    /// Pin the velocity of `id` so it does not change during
    /// [`Self::optimize`]. Useful for anchoring the initial keyframe's
    /// velocity to a measured value when the IMU factor would otherwise
    /// leave it under-constrained.
    pub fn fix_velocity(&mut self, id: u64) {
        self.fixed_velocities.insert(id);
    }

    /// Append an on-manifold IMU pre-integration factor between two
    /// keyframes. Both keyframes must have a [`Self::add_velocity`]
    /// entry; otherwise the factor is silently skipped during build
    /// (the rest of the BA still runs).
    pub fn add_imu_factor(&mut self, factor: ImuPreintegrationFactor) {
        self.imu_factors.push(factor);
    }

    /// Set the calibrated camera/sensor-to-body transform used only by IMU
    /// residuals. Reprojection residuals always retain the camera pose state.
    pub fn set_imu_body_to_camera(&mut self, body_to_camera: SE3) {
        self.imu_body_to_camera = body_to_camera;
    }

    /// Register an initial IMU bias state for the given keyframe,
    /// packing the gyro bias in the first 3 components and the
    /// accelerometer bias in the last 3. The bias becomes a BA
    /// variable (unless also passed to [`Self::fix_bias`]). Required
    /// for any keyframe that should provide the bias-correction term
    /// for its outgoing [`ImuPreintegrationFactor`]; keyframes without
    /// a registered bias use the factor's linearisation bias
    /// (no correction) and do not contribute a bias Jacobian column.
    pub fn add_bias(&mut self, id: u64, bias: Vector6<f64>) {
        self.biases.insert(id, bias);
    }

    /// Pin the bias of `id` so it does not change during
    /// [`Self::optimize`]. The bias is still used for the residual's
    /// first-order correction (so the integration's linearisation
    /// point and the BA-side bias estimate stay decoupled), but no
    /// bias Jacobian column is added to the normal equations.
    pub fn fix_bias(&mut self, id: u64) {
        self.fixed_biases.insert(id);
    }

    /// Append a bias random-walk prior. Both endpoints should have
    /// been registered via [`Self::add_bias`]; the factor pulls
    /// `bias[keyframe_id_to]` toward `bias[keyframe_id_from]` with
    /// a weight of `weight`. See [`BiasRandomWalkFactor`].
    pub fn add_bias_random_walk_factor(&mut self, factor: BiasRandomWalkFactor) {
        self.bias_random_walk_factors.push(factor);
    }

    /// Install (or replace) the gravity prior used by [`Self::optimize`]
    /// and [`Self::robust_cost`]. See [`GravityPrior`] for semantics.
    pub fn set_gravity_prior(&mut self, prior: GravityPrior) {
        self.gravity_prior = Some(prior);
    }

    /// Install (or replace) the per-keyframe gravity prior. See
    /// [`PerPoseGravityPrior`] for semantics; this is the online-
    /// friendly companion to [`Self::set_gravity_prior`] that accepts
    /// per-keyframe `g_camera_observed` observations.
    pub fn set_per_pose_gravity_prior(&mut self, prior: PerPoseGravityPrior) {
        self.per_pose_gravity_prior = Some(prior);
    }

    /// Install (or replace) the absolute position prior. See
    /// [`PositionPrior`] for semantics.
    pub fn set_position_prior(&mut self, prior: PositionPrior) {
        self.position_prior = Some(prior);
    }

    pub fn add_pose(&mut self, id: u64, pose: Pose) {
        self.poses.insert(id, pose);
    }

    pub fn fix_pose(&mut self, id: u64) {
        self.fixed_poses.insert(id);
    }

    /// Pin only the rotation of `id` during bundle adjustment.  Translation
    /// remains a variable, which is useful for diagnostic decompositions that
    /// separate rotational from translational error.  Calling this for a
    /// fully fixed pose is harmless.
    pub fn fix_pose_rotation(&mut self, id: u64) {
        self.fixed_pose_rotations.insert(id);
    }

    pub fn add_landmark(&mut self, id: u64, xyz: Point3<f64>) {
        self.landmarks.insert(id, xyz);
    }

    pub fn fix_landmark(&mut self, id: u64) {
        self.fixed_landmarks.insert(id);
    }

    pub fn add_observation(&mut self, obs: BaObservation) {
        self.observations.push(obs);
    }

    /// Append a rectified-stereo observation. Caller must call
    /// [`Self::set_stereo_baseline`] (with the same baseline used to
    /// triangulate `landmark_id`) before [`Self::optimize`], or the optimizer
    /// returns [`BaError::MissingStereoBaseline`].
    pub fn add_stereo_observation(&mut self, obs: BaStereoObservation) {
        self.stereo_observations.push(obs);
    }

    pub fn add_general_stereo_observation(&mut self, obs: BaGeneralStereoObservation) {
        self.general_stereo_observations.push(obs);
    }

    /// Set the rectified-stereo baseline (positive, metric, in the units of
    /// the landmark coordinates). Required when any
    /// [`Self::stereo_observations`] are present.
    pub fn set_stereo_baseline(&mut self, baseline: f64) {
        self.stereo_baseline = Some(baseline);
    }

    /// Sum of squared reprojection residuals `Σ ||π(K · T · X_w) − u||²`.
    /// Observations whose camera-frame point falls behind the camera are
    /// skipped (cost is reported as if those observations are absent).
    /// Equivalent to [`Self::robust_cost`] called with [`RobustKernel::None`].
    pub fn cost(&self) -> f64 {
        self.robust_cost(&RobustKernel::None)
    }

    /// Number of visual observations that cannot currently be projected.
    ///
    /// Reprojection cost and Jacobian assembly intentionally skip points on
    /// or behind the camera plane.  Without a separate feasibility check an
    /// LM step can therefore lower its reported cost merely by moving hard
    /// observations behind a camera.  Candidate steps are only accepted when
    /// this count does not increase.
    fn nonprojectable_observation_count(&self) -> usize {
        let mono = self.observations.iter().filter(|obs| {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || self.camera.project(&xc).is_none()
        });

        let intrinsics = self.intrinsics();
        let baseline_valid = self
            .stereo_baseline
            .is_some_and(|baseline| baseline.is_finite() && baseline > 0.0);
        let stereo = self.stereo_observations.iter().filter(|obs| {
            let (Some(intrinsics), true, Some(pose), Some(point)) = (
                intrinsics,
                baseline_valid,
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                return true;
            };
            let xc = pose.transform_world_point(point);
            xc.z <= 0.0 || project_pinhole(&intrinsics, &xc).is_none()
        });

        let general_stereo = self.general_stereo_observations.iter().filter(|obs| {
            let (Some(pose), Some(point), Some(left_intrinsics)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
                self.intrinsics(),
            ) else {
                return true;
            };
            general_stereo_residual_jacobians(&left_intrinsics, obs, pose, point).is_none()
        });

        mono.count() + stereo.count() + general_stereo.count()
    }

    /// Robust reprojection cost: `Σ ρ(||r||²)` where `ρ` is the supplied
    /// [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::cost`]. Stereo observations contribute a 3-vector residual
    /// `(u_l_pred − u_l_meas, v_l_pred − v_l_meas, u_r_pred − u_r_meas)`
    /// where `u_r_pred = u_l_pred − fx · b / Z` (rectified-stereo assumption,
    /// see [`BaStereoObservation`]).
    pub fn robust_cost(&self, kernel: &RobustKernel) -> f64 {
        self.robust_cost_weighted(kernel, None)
    }

    /// Additive cost decomposition at the current linearisation point.
    ///
    /// `imu_normalized_squared_residual_per_dof` is the mean whitened
    /// squared IMU residual (NIS / 9 DoF per preintegration factor). It is
    /// useful for detecting a visual/inertial consistency failure that a
    /// single aggregate BA cost would otherwise hide.
    pub fn cost_breakdown(&self, kernel: &RobustKernel) -> BaCostBreakdown {
        self.cost_breakdown_weighted(kernel, None)
    }

    /// Confidence-weighted counterpart of [`Self::cost_breakdown`]. Visual
    /// terms use the same flattened weight vector as
    /// [`Self::optimize_with_observation_weights`]; IMU, navigation and other
    /// structural terms retain their physical information matrices.
    pub fn cost_breakdown_with_observation_weights(
        &self,
        kernel: &RobustKernel,
        observation_weights: &[f64],
    ) -> Result<BaCostBreakdown, BaError> {
        self.validate_observation_weights(observation_weights)?;
        Ok(self.cost_breakdown_weighted(kernel, Some(observation_weights)))
    }

    fn cost_breakdown_weighted(
        &self,
        kernel: &RobustKernel,
        observation_weights: Option<&[f64]>,
    ) -> BaCostBreakdown {
        let total = self.robust_cost_weighted(kernel, observation_weights);

        let mut visual_problem = self.clone();
        visual_problem.gravity_prior = None;
        visual_problem.per_pose_gravity_prior = None;
        visual_problem.position_prior = None;
        visual_problem.pairwise_pose_factors.clear();
        visual_problem.bias_random_walk_factors.clear();
        visual_problem.imu_factors.clear();
        visual_problem.navigation_state_prior = None;
        let visual = visual_problem.robust_cost_weighted(kernel, observation_weights);

        let mut imu_problem = self.clone();
        clear_visual_and_structural_costs(&mut imu_problem);
        imu_problem.bias_random_walk_factors.clear();
        imu_problem.navigation_state_prior = None;
        let imu = imu_problem.robust_cost(kernel);

        let mut bias_problem = self.clone();
        clear_visual_and_structural_costs(&mut bias_problem);
        bias_problem.imu_factors.clear();
        bias_problem.navigation_state_prior = None;
        let bias_random_walk = bias_problem.robust_cost(kernel);

        let mut navigation_problem = self.clone();
        clear_visual_and_structural_costs(&mut navigation_problem);
        navigation_problem.imu_factors.clear();
        navigation_problem.bias_random_walk_factors.clear();
        let navigation_prior = navigation_problem.robust_cost(kernel);

        let other_structural = total - visual - imu - bias_random_walk - navigation_prior;
        let imu_normalized_squared_residual_per_dof =
            (!self.imu_factors.is_empty()).then_some(imu / (9.0 * self.imu_factors.len() as f64));
        let (
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        ) = self
            .imu_raw_residual_rms()
            .map_or((None, None, None), |(rotation, velocity, position)| {
                (Some(rotation), Some(velocity), Some(position))
            });
        BaCostBreakdown {
            total,
            visual,
            imu,
            bias_random_walk,
            navigation_prior,
            other_structural,
            imu_normalized_squared_residual_per_dof,
            imu_rotation_residual_rms_rad,
            imu_velocity_residual_rms_mps,
            imu_position_residual_rms_meters,
        }
    }

    /// Unwhitened per-axis RMS of the rotation, velocity, and position IMU
    /// residual blocks at the current state, in physical units.
    pub fn imu_raw_residual_rms(&self) -> Option<(f64, f64, f64)> {
        let mut rotation_squared = 0.0;
        let mut velocity_squared = 0.0;
        let mut position_squared = 0.0;
        let mut evaluated = 0usize;
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j), Some(v_i), Some(v_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rotation, r_velocity, r_position] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bias_gyro: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let bias_accel: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(
                        &r_i,
                        &p_i,
                        v_i,
                        &r_j,
                        &p_j,
                        v_j,
                        &bias_gyro,
                        &bias_accel,
                    )
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            rotation_squared += r_rotation.norm_squared();
            velocity_squared += r_velocity.norm_squared();
            position_squared += r_position.norm_squared();
            evaluated += 1;
        }
        (evaluated > 0).then(|| {
            let denominator = 3.0 * evaluated as f64;
            (
                (rotation_squared / denominator).sqrt(),
                (velocity_squared / denominator).sqrt(),
                (position_squared / denominator).sqrt(),
            )
        })
    }

    /// Like [`Self::robust_cost`] but multiplies each reprojection
    /// observation's contribution by an external per-observation weight
    /// (the Graduated Non-Convexity Black-Rangarajan weight `w ∈ [0,1]`).
    /// `gnc_weights` is indexed monocular-observations-first
    /// (`0 .. observations.len()`), rectified stereo, then general stereo
    /// (`observations.len() .. + stereo_observations.len()`). `None`
    /// reproduces [`Self::robust_cost`] exactly. Structural and inertial
    /// terms (gravity / position priors, pairwise pose, bias random-walk,
    /// IMU) are never reweighted — only outlier-prone feature
    /// reprojections are, so a wrong correspondence is the only thing GNC
    /// can switch off.
    fn robust_cost_weighted(&self, kernel: &RobustKernel, gnc_weights: Option<&[f64]>) -> f64 {
        let intrinsics = match self.intrinsics() {
            Some(k) => k,
            None => return 0.0,
        };
        let mut total = 0.0;
        for (obs_idx, obs) in self.observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            // Distortion-aware projection (identical to `project_pinhole` when the
            // camera carries no distortion, so all existing callers are unchanged).
            if let Some(predicted) = self.camera.project(&xc) {
                let r = predicted - obs.xy;
                let s = r.x * r.x + r.y * r.y;
                let w = gnc_weights.map_or(1.0, |gw| gw[obs_idx]);
                total += w * kernel.cost(s);
            }
        }
        if let Some(baseline) = self.stereo_baseline {
            if baseline.is_finite() && baseline > 0.0 {
                let (fx, _fy, _cx, _cy) = intrinsics;
                let stereo_offset = self.observations.len();
                for (st_idx, obs) in self.stereo_observations.iter().enumerate() {
                    let (Some(pose), Some(point)) = (
                        self.poses.get(&obs.keyframe_id),
                        self.landmarks.get(&obs.landmark_id),
                    ) else {
                        continue;
                    };
                    let xc = pose.transform_world_point(point);
                    if xc.z <= 0.0 {
                        continue;
                    }
                    if let Some(predicted) = project_pinhole(&intrinsics, &xc) {
                        let u_r_pred = predicted.x - fx * baseline / xc.z;
                        let dx = predicted.x - obs.xy.x;
                        let dy = predicted.y - obs.xy.y;
                        let dr = u_r_pred - obs.u_right;
                        let s = dx * dx + dy * dy + dr * dr;
                        let w = gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);
                        total += w * kernel.cost(s);
                    }
                }
            }
        }
        let general_offset = self.observations.len() + self.stereo_observations.len();
        for (index, obs) in self.general_stereo_observations.iter().enumerate() {
            let (Some(pose), Some(point)) = (
                self.poses.get(&obs.keyframe_id),
                self.landmarks.get(&obs.landmark_id),
            ) else {
                continue;
            };
            let Some((residual, _, _)) =
                general_stereo_residual_jacobians(&intrinsics, obs, pose, point)
            else {
                continue;
            };
            let w = gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
            total += w * kernel.cost(residual.norm_squared());
        }
        if let Some(prior) = &self.gravity_prior {
            for pose in self.poses.values() {
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - prior.g_camera_observed;
                let s = r_vec.norm_squared();
                // Gravity prior uses an L2 (non-robust) contribution: the
                // measurement is global per pose, not a per-feature
                // outlier-prone observation, so a robust kernel here would
                // hide rather than down-weight prior–data conflicts.
                total += prior.weight * s;
            }
        }
        if let Some(prior) = &self.per_pose_gravity_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let r_mat = pose
                    .world_to_camera
                    .rotation
                    .to_rotation_matrix()
                    .into_inner();
                let r_vec: Vector3<f64> = r_mat * prior.g_world - obs.g_camera_observed;
                let s = r_vec.norm_squared();
                // Same L2 (non-robust) reasoning as [`Self::gravity_prior`].
                total += prior.weight * obs.weight * s;
            }
        }
        if let Some(prior) = &self.position_prior {
            for obs in &prior.observations {
                let Some(pose) = self.poses.get(&obs.keyframe_id) else {
                    continue;
                };
                let c_world = pose.camera_center_world();
                let r_vec = c_world - obs.camera_center_world;
                // Axis-weighted L2 cost: Σ wᵢ · rᵢ². Zero weight axes
                // contribute nothing, so an altitude-only prior with
                // `axis_weights = (0, w, 0)` is exact.
                let s = obs.axis_weights.x * r_vec.x * r_vec.x
                    + obs.axis_weights.y * r_vec.y * r_vec.y
                    + obs.axis_weights.z * r_vec.z * r_vec.z;
                total += s;
            }
        }
        for factor in &self.pairwise_pose_factors {
            let (Some(from), Some(to)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let predicted = to.world_to_camera.compose(&from.world_to_camera.inverse());
            let r = factor
                .measurement
                .world_to_camera
                .inverse()
                .compose(&predicted)
                .log();
            total += factor.weight * r.norm_squared();
        }
        for factor in &self.bias_random_walk_factors {
            let (Some(b_i), Some(b_j)) = (
                self.biases.get(&factor.keyframe_id_from),
                self.biases.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let r: Vector6<f64> = b_j - b_i;
            total += factor.weight_gyro.max(0.0) * r.fixed_rows::<3>(0).norm_squared()
                + factor.weight_accel.max(0.0) * r.fixed_rows::<3>(3).norm_squared();
        }
        // IMU pre-integration factors: 9-vector residual [r_R; r_v; r_p]
        // weighted axis-wise. The factor's `residual` helper takes the
        // body-to-world rotation and world-frame body centre obtained by
        // composing the camera pose with the calibrated T_b<-c extrinsic.
        // When `self.biases` carries a bias for the factor's "from"
        // keyframe, the bias-corrected residual is used (Forster eq. 44).
        for factor in &self.imu_factors {
            let (Some(pose_i), Some(pose_j)) = (
                self.poses.get(&factor.keyframe_id_from),
                self.poses.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let (Some(v_i), Some(v_j)) = (
                self.velocities.get(&factor.keyframe_id_from),
                self.velocities.get(&factor.keyframe_id_to),
            ) else {
                continue;
            };
            let body_i = self.imu_body_to_camera.compose(&pose_i.world_to_camera);
            let body_j = self.imu_body_to_camera.compose(&pose_j.world_to_camera);
            let r_i = SO3::from_quaternion(body_i.rotation.inverse());
            let r_j = SO3::from_quaternion(body_j.rotation.inverse());
            let p_i: Vector3<f64> = body_i.inverse().translation;
            let p_j: Vector3<f64> = body_j.inverse().translation;
            let [r_rot, r_vel, r_pos] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let ba: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(&r_i, &p_i, v_i, &r_j, &p_j, v_j, &bg, &ba)
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            if let Some(whitener) = factor.covariance_sqrt_information() {
                let mut residual = nalgebra::SVector::<f64, 9>::zeros();
                residual.fixed_rows_mut::<3>(0).copy_from(&r_rot);
                residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
                residual.fixed_rows_mut::<3>(6).copy_from(&r_pos);
                total += (whitener * residual).norm_squared();
            } else {
                total += factor.weight_rotation * r_rot.norm_squared()
                    + factor.weight_velocity * r_vel.norm_squared()
                    + factor.weight_position * r_pos.norm_squared();
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            if let Some(delta) = navigation_prior_delta(self, prior) {
                let quadratic = delta.dot(&(&prior.information * &delta));
                total += prior.constant_cost + 2.0 * prior.gradient.dot(&delta) + quadratic;
            }
        }
        total
    }

    fn intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        match self.camera.model {
            CameraModel::Pinhole | CameraModel::SimplePinhole => self.camera.intrinsics(),
            _ => None,
        }
    }

    /// Per-observation squared reprojection residual `s = ‖r‖²` (pixel²),
    /// evaluated at the current state and aligned to the GNC weight layout
    /// used everywhere in this file: monocular observations first
    /// (`0 .. observations.len()`), then rectified stereo, then general
    /// stereo. An observation that cannot
    /// be evaluated now (missing pose / landmark, behind the camera, or
    /// non-projectable, or — for stereo — no usable baseline) is reported
    /// as `f64::NAN`, so it neither sets the GNC inlier scale nor is
    /// classified as an inlier or outlier.
    fn reprojection_squared_residuals(&self) -> Vec<f64> {
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len();
        let mut out = Vec::with_capacity(n);
        let Some(intrinsics) = self.intrinsics() else {
            out.resize(n, f64::NAN);
            return out;
        };
        for obs in &self.observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = project_pinhole(&intrinsics, &xc)?;
                let r = predicted - obs.xy;
                Some(r.x * r.x + r.y * r.y)
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        let baseline = match self.stereo_baseline {
            Some(b) if b.is_finite() && b > 0.0 => Some(b),
            _ => None,
        };
        let (fx, _, _, _) = intrinsics;
        for obs in &self.stereo_observations {
            let s = baseline.and_then(|baseline| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let xc = pose.transform_world_point(point);
                if xc.z <= 0.0 {
                    return None;
                }
                let predicted = project_pinhole(&intrinsics, &xc)?;
                let u_r_pred = predicted.x - fx * baseline / xc.z;
                let dx = predicted.x - obs.xy.x;
                let dy = predicted.y - obs.xy.y;
                let dr = u_r_pred - obs.u_right;
                Some(dx * dx + dy * dy + dr * dr)
            });
            out.push(s.unwrap_or(f64::NAN));
        }
        for obs in &self.general_stereo_observations {
            let s = (|| {
                let pose = self.poses.get(&obs.keyframe_id)?;
                let point = self.landmarks.get(&obs.landmark_id)?;
                let (residual, _, _) =
                    general_stereo_residual_jacobians(&intrinsics, obs, pose, point)?;
                Some(residual.norm_squared())
            })();
            out.push(s.unwrap_or(f64::NAN));
        }
        out
    }

    /// Run Levenberg-Marquardt bundle adjustment with Schur-complement
    /// landmark elimination. Returns iteration trace and final cost.
    pub fn optimize(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        if !config.refine_intrinsics || !self.general_stereo_observations.is_empty() {
            return self.optimize_weighted(config, None);
        }
        // Joint pose + structure + intrinsics refinement (the COLMAP self-
        // calibration formulation). Falls back to the pose/structure-only solve
        // for non-pinhole cameras, which carry no refinable 4-parameter intrinsics.
        if self.camera.model != CameraModel::Pinhole || self.camera.intrinsics().is_none() {
            return self.optimize_weighted(config, None);
        }
        self.optimize_joint_intrinsics(config)
    }

    /// Run pose/structure bundle adjustment with one external confidence
    /// weight per visual observation.
    ///
    /// The flattened order is monocular observations first, followed by
    /// rectified stereo and then calibrated general stereo. A weight of `1`
    /// preserves the ordinary BA contribution and `0` disables that visual
    /// residual. Intermediate values provide the confidence-weighted least
    /// squares used by learned correspondence update operators such as
    /// DROID-SLAM. The external weight multiplies the configured robust-kernel
    /// weight; inertial and structural priors are never reweighted.
    ///
    /// Joint intrinsics refinement is deliberately rejected because its
    /// separate normal-equation builder does not yet consume these weights.
    pub fn optimize_with_observation_weights(
        &mut self,
        config: &BaConfig,
        observation_weights: &[f64],
    ) -> Result<BaResult, BaError> {
        self.validate_observation_weights(observation_weights)?;
        if config.refine_intrinsics {
            return Err(BaError::ObservationWeightsWithIntrinsicsRefinement);
        }
        self.optimize_weighted(config, Some(observation_weights))
    }

    fn validate_observation_weights(&self, observation_weights: &[f64]) -> Result<(), BaError> {
        let expected = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len();
        if observation_weights.len() != expected {
            return Err(BaError::ObservationWeightCount {
                expected,
                actual: observation_weights.len(),
            });
        }
        if let Some((index, _)) = observation_weights
            .iter()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite() || **weight < 0.0)
        {
            return Err(BaError::InvalidObservationWeight(index));
        }
        Ok(())
    }

    /// Bundle adjustment that carries the shared pinhole intrinsics
    /// `(fx, fy, cx, cy)` as four extra unknowns **inside** the Schur-complement
    /// camera system, jointly with the poses and (eliminated) landmarks.
    ///
    /// This is the difference that matters versus an *alternating* refinement
    /// (update the intrinsics by Gauss-Newton against a *converged* structure,
    /// then re-solve): there the structure-fixed gradient `∂cost/∂K` is ≈ 0 (the
    /// structure has already absorbed any focal error), so it cannot move a wrong
    /// focal. The joint solve uses the **coupled** gradient — the
    /// reduced-camera gradient *after* landmark elimination — which is non-zero,
    /// so it pulls the intrinsics and poses together toward the true calibration.
    ///
    /// SfM-only: handles monocular + rectified-stereo reprojection observations
    /// and ignores IMU / velocity / bias / gravity / position-prior factors (which
    /// SfM intrinsics refinement does not use). The intrinsics are always a free
    /// block; the caller fixes poses (anchor + farthest, or ≥2 stereo observers) to
    /// pin the remaining gauge. Writes refined poses, landmarks, and intrinsics
    /// into `self`.
    fn optimize_joint_intrinsics(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        let kernel = config.robust_kernel;

        // Variable layout: non-fixed poses occupy `6·p .. 6·p+6`; the 4 shared
        // intrinsics occupy the final block `k_off .. k_off+4`. Fixed poses and
        // fixed landmarks contribute residuals but get no variable slot.
        let mut pose_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if self.fixed_poses.contains(&id) {
                continue;
            }
            let next = pose_index.len();
            pose_index.insert(id, next);
        }
        let mut landmark_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.landmarks.keys() {
            if self.fixed_landmarks.contains(&id) {
                continue;
            }
            let next = landmark_index.len();
            landmark_index.insert(id, next);
        }
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        // Also self-calibrate radial distortion (k1, k2) when asked — but only on
        // a monocular reconstruction (rectified stereo is already undistorted, and
        // its baseline term does not carry a distortion model). The two coefficients
        // get appended to the camera block, so `k_dim` is 6 instead of 4.
        let refine_dist = config.refine_distortion
            && self.camera.model == CameraModel::Pinhole
            && self.stereo_observations.is_empty();
        if refine_dist {
            // Ensure the camera carries the two distortion slots (start at 0).
            while self.camera.params.len() < 6 {
                self.camera.params.push(0.0);
            }
        }
        let k_dim = if refine_dist { 6 } else { 4 };
        let cam_dim = k_off + k_dim;

        let initial_cost = self.robust_cost_weighted(&kernel, None);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;

        for iteration in 0..config.max_iterations {
            // Current distortion (reflects the running k1, k2 estimate) drives the
            // distortion-aware projection / Jacobians inside the build.
            let dist = self.camera.radial_distortion();
            let (cam_dim_n, h_cc, b_c, lm_blocks) = self.build_joint_intrinsics_system(
                &pose_index,
                &landmark_index,
                &kernel,
                k_dim,
                dist,
            );
            debug_assert_eq!(cam_dim_n, cam_dim);

            // Damped Schur reduction (Levenberg I·λ on both the camera and the
            // landmark diagonals, exactly as `solve_step`).
            let mut s = h_cc.clone();
            if lambda > 0.0 {
                for d in 0..cam_dim {
                    s[(d, d)] += lambda;
                }
            }
            let mut b_reduced = -&b_c;
            let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(lm_blocks.len());
            for lm in &lm_blocks {
                let mut h_ll = lm.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let inv = h_ll.try_inverse();
                h_ll_inv_cache.push(inv);
                let Some(inv) = inv else { continue };
                // S -= Σ cross_a^T · H_ll^{-1} · cross_b ; b += cross_a^T H_ll^{-1} b_l.
                for (cs_a, a) in &lm.cross {
                    let ah = a * inv; // (rows_a × 3)
                    for (cs_b, b) in &lm.cross {
                        let block = &ah * b.transpose(); // (rows_a × rows_b)
                        for r in 0..a.nrows() {
                            for c in 0..b.nrows() {
                                s[(cs_a + r, cs_b + c)] -= block[(r, c)];
                            }
                        }
                    }
                    let upd = &ah * lm.b_l; // (rows_a)
                    for r in 0..a.nrows() {
                        b_reduced[cs_a + r] += upd[r];
                    }
                }
            }

            let delta_cam = match solve_normal_equations(&s, &b_reduced) {
                Ok(d) => d,
                Err(_) => {
                    lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                    iterations.push(BaIterationStats {
                        iteration,
                        cost_before: current_cost,
                        cost_after: current_cost,
                        max_pose_step: 0.0,
                        max_landmark_step: 0.0,
                        lambda,
                        step_accepted: false,
                    });
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
            };

            // Back-substitute landmark updates: δ_L = H_ll^{-1}(−b_l − Σ crossᵀ δ_cam).
            let mut delta_lm: BTreeMap<u64, Vector3<f64>> = BTreeMap::new();
            for (lm, inv) in lm_blocks.iter().zip(&h_ll_inv_cache) {
                let Some(inv) = inv else { continue };
                let mut acc = -lm.b_l;
                for (cs, a) in &lm.cross {
                    let mut dcam = DVector::<f64>::zeros(a.nrows());
                    for r in 0..a.nrows() {
                        dcam[r] = delta_cam[cs + r];
                    }
                    acc -= a.transpose() * dcam;
                }
                delta_lm.insert(lm.id, inv * acc);
            }

            // Tentative update (save → apply → cost → accept/reject).
            let saved_poses = self.poses.clone();
            let saved_landmarks = self.landmarks.clone();
            let saved_params = self.camera.params.clone();
            let cost_before = current_cost;

            let mut max_pose_step = 0.0f64;
            for (&id, &p) in &pose_index {
                let xi: Vector6<f64> = delta_cam.fixed_rows::<6>(p * 6).into_owned();
                max_pose_step = max_pose_step.max(xi.norm());
                let pose = self.poses.get_mut(&id).expect("pose exists");
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }
            let mut max_landmark_step = 0.0f64;
            for (&id, dl) in &delta_lm {
                max_landmark_step = max_landmark_step.max(dl.norm());
                let pt = self.landmarks.get_mut(&id).expect("landmark exists");
                *pt = Point3::from(pt.coords + dl);
            }
            // Intrinsics (and, when k_dim == 6, distortion) block.
            for j in 0..k_dim {
                self.camera.params[j] += delta_cam[k_off + j];
            }

            let cost_after = self.robust_cost_weighted(&kernel, None);
            let nonprojectable_after = self.nonprojectable_observation_count();
            let cost_accepted = match config.initial_lambda {
                None => true,
                Some(_) => cost_after < cost_before,
            };
            let step_accepted = cost_accepted && nonprojectable_after <= current_nonprojectable;
            if !step_accepted {
                self.poses = saved_poses;
                self.landmarks = saved_landmarks;
                self.camera.params = saved_params;
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(BaIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_pose_step,
                    max_landmark_step,
                    lambda,
                    step_accepted: false,
                });
                if config.initial_lambda.is_none() {
                    break;
                }
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(BaIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_pose_step,
                max_landmark_step,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            current_nonprojectable = nonprojectable_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }
            if max_pose_step < config.step_tolerance && max_landmark_step < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
            if config.relative_cost_tolerance.is_some_and(|tolerance| {
                tolerance.is_finite()
                    && tolerance >= 0.0
                    && (cost_before - cost_after) / cost_before.abs().max(f64::EPSILON) < tolerance
            }) {
                converged = true;
                break;
            }
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }

    /// Assemble the raw (un-damped) joint normal equations for
    /// [`Self::optimize_joint_intrinsics`]: the camera-block Hessian `H_cc`
    /// (poses then the 4 intrinsics) and gradient `b_c`, plus per-landmark
    /// `{H_ll, b_l, cross}` blocks where `cross` maps each touching camera-block
    /// column-start to `Jᵀ_cam · J_lm`. Mirrors `build_normal_equations`'
    /// reprojection Jacobians, extended with the intrinsics columns
    /// `J_K = ∂(predicted)/∂(fx, fy, cx, cy)`.
    fn build_joint_intrinsics_system(
        &self,
        pose_index: &BTreeMap<u64, usize>,
        landmark_index: &BTreeMap<u64, usize>,
        kernel: &RobustKernel,
        k_dim: usize,
        dist: Option<(f64, f64)>,
    ) -> (usize, DMatrix<f64>, DVector<f64>, Vec<JointLandmarkBlock>) {
        let intrinsics = self.intrinsics().expect("pinhole checked by caller");
        let (fx, fy, cx, cy) = intrinsics;
        let p_count = pose_index.len();
        let k_off = p_count * 6;
        let cam_dim = k_off + k_dim;
        let mut h_cc = DMatrix::<f64>::zeros(cam_dim, cam_dim);
        let mut b_c = DVector::<f64>::zeros(cam_dim);
        let mut lm_blocks: Vec<JointLandmarkBlock> = landmark_index
            .iter()
            .map(|(&id, _)| JointLandmarkBlock {
                id,
                h_ll: Matrix3::zeros(),
                b_l: Vector3::zeros(),
                cross: BTreeMap::new(),
            })
            .collect();

        // Accumulate a camera×camera block (rows_a × cols_b) at (row_start, col_start).
        let mut add_cc = |rs: usize, cs: usize, blk: &DMatrix<f64>| {
            for r in 0..blk.nrows() {
                for c in 0..blk.ncols() {
                    h_cc[(rs + r, cs + c)] += blk[(r, c)];
                }
            }
        };

        // Monocular observations.
        for obs in &self.observations {
            let pose = &self.poses[&obs.keyframe_id];
            let point = &self.landmarks[&obs.landmark_id];
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            let z_inv = 1.0 / xc.z;
            let x = xc.x * z_inv;
            let y = xc.y * z_inv;
            let r2 = x * x + y * y;
            // Radial distortion factor d = 1 + k1·r² + k2·r⁴ and its radial
            // derivative helper g = k1 + 2·k2·r² (d=1, g=0 when distortion-free).
            let (k1, k2) = dist.unwrap_or((0.0, 0.0));
            let d = 1.0 + k1 * r2 + k2 * r2 * r2;
            let g = k1 + 2.0 * k2 * r2;
            let (xd, yd) = (x * d, y * d);
            let predicted = Point2::new(fx * xd + cx, fy * yd + cy);
            let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            // J_π = diag(fx, fy) · D · ∂(x, y)/∂X_c, where the distortion Jacobian
            //   D = [[d + 2x²g, 2xyg], [2xyg, d + 2y²g]]  (= I when distortion-free)
            // and ∂(x, y)/∂X_c = (1/Z)·[[1, 0, -x], [0, 1, -y]].
            let d11 = d + 2.0 * x * x * g;
            let d12 = 2.0 * x * y * g;
            let d22 = d + 2.0 * y * y * g;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * d11 * z_inv;
            j_pi[(0, 1)] = fx * d12 * z_inv;
            j_pi[(0, 2)] = -fx * (d11 * x + d12 * y) * z_inv;
            j_pi[(1, 0)] = fy * d12 * z_inv;
            j_pi[(1, 1)] = fy * d22 * z_inv;
            j_pi[(1, 2)] = -fy * (d12 * x + d22 * y) * z_inv;
            let mut dx_dxi = Matrix3x6::<f64>::zeros();
            dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
            dx_dxi
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r_mat * skew(&point.coords)));
            let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
            let j_lm: Matrix2x3<f64> = j_pi * r_mat;
            // ∂(predicted)/∂K with K = (fx, fy, cx, cy[, k1, k2]) (2×k_dim).
            let mut j_k = DMatrix::<f64>::zeros(2, k_dim);
            j_k[(0, 0)] = xd;
            j_k[(0, 2)] = 1.0;
            j_k[(1, 1)] = yd;
            j_k[(1, 3)] = 1.0;
            if k_dim == 6 {
                j_k[(0, 4)] = fx * x * r2;
                j_k[(0, 5)] = fx * x * r2 * r2;
                j_k[(1, 4)] = fy * y * r2;
                j_k[(1, 5)] = fy * y * r2 * r2;
            }

            let s = residual.x * residual.x + residual.y * residual.y;
            let w = kernel.weight(s);
            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();

            // Dynamic-sized residual / pose for the K-coupled products.
            let res2 = DVector::from_column_slice(&[residual.x, residual.y]);
            let jkt = j_k.transpose(); // k_dim×2

            // K-K and K gradient (intrinsics are always variable).
            add_cc(k_off, k_off, &(w * (&jkt * &j_k)));
            let bk = w * (&jkt * &res2);
            for j in 0..k_dim {
                b_c[k_off + j] += bk[j];
            }
            if let Some(p) = i_pose {
                let hpp = w * (j_pose.transpose() * j_pose);
                add_cc(p * 6, p * 6, &DMatrix::from_fn(6, 6, |r, c| hpp[(r, c)]));
                let bp = w * (j_pose.transpose() * residual);
                for r in 0..6 {
                    b_c[p * 6 + r] += bp[r];
                }
                // pose-K coupling (and its transpose).
                let jp_dyn = DMatrix::from_iterator(2, 6, j_pose.iter().copied());
                let hpk = w * (jp_dyn.transpose() * &j_k); // 6×k_dim
                add_cc(p * 6, k_off, &hpk);
                add_cc(k_off, p * 6, &hpk.transpose());
            }
            if let Some(l) = i_lm {
                lm_blocks[l].h_ll += w * (j_lm.transpose() * j_lm);
                lm_blocks[l].b_l += w * (j_lm.transpose() * residual);
                if let Some(p) = i_pose {
                    let cr = w * (j_pose.transpose() * j_lm); // 6×3
                    add_cross(
                        &mut lm_blocks[l].cross,
                        p * 6,
                        6,
                        &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                    );
                }
                let jl_dyn = DMatrix::from_iterator(2, 3, j_lm.iter().copied());
                let crk = w * (&jkt * &jl_dyn); // k_dim×3
                add_cross(&mut lm_blocks[l].cross, k_off, k_dim, &crk);
            }
        }

        // Rectified-stereo observations (3D residual u_l, v_l, u_r).
        if !self.stereo_observations.is_empty() {
            if let Some(baseline) = self.stereo_baseline {
                if baseline.is_finite() && baseline > 0.0 {
                    for obs in &self.stereo_observations {
                        let pose = &self.poses[&obs.keyframe_id];
                        let point = &self.landmarks[&obs.landmark_id];
                        let xc = pose.transform_world_point(point);
                        if xc.z <= 0.0 {
                            continue;
                        }
                        let Some(predicted) = project_pinhole(&intrinsics, &xc) else {
                            continue;
                        };
                        let z_inv = 1.0 / xc.z;
                        let z_inv2 = z_inv * z_inv;
                        let u_r_pred = predicted.x - fx * baseline * z_inv;
                        let residual = Vector3::new(
                            predicted.x - obs.xy.x,
                            predicted.y - obs.xy.y,
                            u_r_pred - obs.u_right,
                        );
                        let r_mat = pose
                            .world_to_camera
                            .rotation
                            .to_rotation_matrix()
                            .into_inner();
                        let mut j_pi = Matrix3::<f64>::zeros();
                        j_pi[(0, 0)] = fx * z_inv;
                        j_pi[(0, 2)] = -fx * xc.x * z_inv2;
                        j_pi[(1, 1)] = fy * z_inv;
                        j_pi[(1, 2)] = -fy * xc.y * z_inv2;
                        j_pi[(2, 0)] = fx * z_inv;
                        j_pi[(2, 2)] = -fx * (xc.x - baseline) * z_inv2;
                        let mut dx_dxi = Matrix3x6::<f64>::zeros();
                        dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
                        dx_dxi
                            .fixed_view_mut::<3, 3>(0, 3)
                            .copy_from(&(-r_mat * skew(&point.coords)));
                        let j_pose: Matrix3x6<f64> = j_pi * dx_dxi;
                        let j_lm: Matrix3<f64> = j_pi * r_mat;
                        // u_r = fx·(X−b)/Z + cx, so ∂u_r/∂fx = (X−b)/Z, ∂u_r/∂cx = 1.
                        let mut j_k = Matrix3x4::<f64>::zeros();
                        j_k[(0, 0)] = xc.x * z_inv;
                        j_k[(0, 2)] = 1.0;
                        j_k[(1, 1)] = xc.y * z_inv;
                        j_k[(1, 3)] = 1.0;
                        j_k[(2, 0)] = (xc.x - baseline) * z_inv;
                        j_k[(2, 2)] = 1.0;

                        let s = residual.norm_squared();
                        let w = kernel.weight(s);
                        let i_pose = pose_index.get(&obs.keyframe_id).copied();
                        let i_lm = landmark_index.get(&obs.landmark_id).copied();

                        add_cc(
                            k_off,
                            k_off,
                            &DMatrix::from_fn(4, 4, |r, c| w * (j_k.transpose() * j_k)[(r, c)]),
                        );
                        let bk = w * (j_k.transpose() * residual);
                        for j in 0..4 {
                            b_c[k_off + j] += bk[j];
                        }
                        if let Some(p) = i_pose {
                            let hpp = w * (j_pose.transpose() * j_pose);
                            add_cc(p * 6, p * 6, &DMatrix::from_fn(6, 6, |r, c| hpp[(r, c)]));
                            let bp = w * (j_pose.transpose() * residual);
                            for r in 0..6 {
                                b_c[p * 6 + r] += bp[r];
                            }
                            let hpk = w * (j_pose.transpose() * j_k);
                            add_cc(p * 6, k_off, &DMatrix::from_fn(6, 4, |r, c| hpk[(r, c)]));
                            add_cc(k_off, p * 6, &DMatrix::from_fn(4, 6, |r, c| hpk[(c, r)]));
                        }
                        if let Some(l) = i_lm {
                            lm_blocks[l].h_ll += w * (j_lm.transpose() * j_lm);
                            lm_blocks[l].b_l += w * (j_lm.transpose() * residual);
                            if let Some(p) = i_pose {
                                let cr = w * (j_pose.transpose() * j_lm);
                                add_cross(
                                    &mut lm_blocks[l].cross,
                                    p * 6,
                                    6,
                                    &DMatrix::from_fn(6, 3, |r, c| cr[(r, c)]),
                                );
                            }
                            let crk = w * (j_k.transpose() * j_lm);
                            add_cross(
                                &mut lm_blocks[l].cross,
                                k_off,
                                4,
                                &DMatrix::from_fn(4, 3, |r, c| crk[(r, c)]),
                            );
                        }
                    }
                }
            }
        }

        (cam_dim, h_cc, b_c, lm_blocks)
    }

    /// Outlier-robust bundle adjustment via Graduated Non-Convexity (GNC).
    ///
    /// A local M-estimator (`RobustKernel::Huber` / `Cauchy`) only
    /// down-weights gross reprojection errors *near* the current estimate,
    /// so a cluster of wrong correspondences that the initialisation
    /// already believes can capture the solution in a bad basin. GNC
    /// instead anneals a control parameter `μ` from a convex surrogate
    /// (every observation trusted — ordinary least squares) toward the true
    /// non-convex robust cost, recomputing the per-observation
    /// Black-Rangarajan weight `w ∈ [0,1]` at each level. Each level is a
    /// bounded weighted-LS solve reusing the same Schur-complement assembly
    /// as [`Self::optimize`] with `RobustKernel::None` (GNC supersedes the
    /// M-estimator). See [`crate::gnc`] for the surrogate math.
    ///
    /// `config` drives the inner LM solve (linear solver, λ schedule);
    /// `config.robust_kernel` is ignored — GNC sets the weights. `gnc.c` is
    /// the inlier reprojection scale **in pixels** (so `c²` is the squared-
    /// residual band): pick it from the expected inlier reprojection error,
    /// e.g. `c ≈ 3` for ~1 px noise. The returned
    /// [`BaGncResult::observation_weights`] gives the final per-observation
    /// weight (monocular, rectified stereo, then general stereo; `NaN` for un-evaluable
    /// observations); near-zero entries are the rejected outliers.
    pub fn optimize_gnc(
        &mut self,
        config: &BaConfig,
        gnc: &GncConfig,
    ) -> Result<BaGncResult, BaError> {
        let kernel_none = RobustKernel::None;
        let initial_cost = self.robust_cost(&kernel_none);
        let n = self.observations.len()
            + self.stereo_observations.len()
            + self.general_stereo_observations.len();

        // GNC inlier scale: largest residual seeds the convex μ₀; the same
        // residuals optionally drive the MAD auto-estimate of `c` (with the
        // configured `c` as a floor) so the pixel threshold tracks the actual
        // reprojection noise instead of a hand-set value.
        let squared_residuals = self.reprojection_squared_residuals();
        let s_max = squared_residuals
            .iter()
            .copied()
            .filter(|s| s.is_finite())
            .fold(0.0_f64, f64::max);
        let effective_gnc = match gnc.auto_scale {
            Some(k) => {
                let c = crate::gnc::estimate_scale_mad(&squared_residuals, k)
                    .map_or(gnc.c, |est| est.max(gnc.c));
                GncConfig { c, ..*gnc }
            }
            None => *gnc,
        };
        let mut inlier_scale = effective_gnc.c;
        let mut state = GncState::new(&effective_gnc, s_max);

        // Inner solve: a short weighted LM with no M-estimator (the GNC
        // weights are the only robustification) restarted at each μ level.
        let mut inner = *config;
        inner.robust_kernel = RobustKernel::None;
        inner.max_iterations = gnc.inner_iterations.max(1);

        let mut weights = vec![1.0_f64; n];
        let mut converged = false;
        let mut outer_iterations = 0usize;
        for _ in 0..gnc.max_outer.max(1) {
            outer_iterations += 1;
            // The terminal level (μ at its recovered extreme) reproduces the
            // true robust cost; we run it, then stop.
            let terminal_level = state.is_terminal();
            let residuals = self.reprojection_squared_residuals();
            // Adaptive inlier scale: re-derive `c` from the current residuals
            // each level (configured `c` as a floor). Level 0 reproduces the
            // one-shot estimate; later levels tighten as the surrogate
            // suppresses outliers and inlier residuals shrink.
            if gnc.auto_scale_readapt {
                if let Some(k) = gnc.auto_scale {
                    if let Some(est) = crate::gnc::estimate_scale_mad(&residuals, k) {
                        let c = est.max(gnc.c);
                        state.set_inlier_scale(c);
                        inlier_scale = c;
                    }
                }
            }
            for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
                *w = if s.is_finite() { state.weight(s) } else { 1.0 };
            }
            self.optimize_weighted(&inner, Some(&weights))?;
            if terminal_level {
                converged = true;
                break;
            }
            state.anneal();
        }

        // Final per-observation weights at the recovered estimate (NaN for
        // observations that cannot be evaluated, matching the result
        // contract and skipped by the weighted cost anyway).
        let residuals = self.reprojection_squared_residuals();
        for (w, &s) in weights.iter_mut().zip(residuals.iter()) {
            *w = if s.is_finite() {
                state.weight(s)
            } else {
                f64::NAN
            };
        }
        let final_cost = self.robust_cost_weighted(&kernel_none, Some(&weights));

        // Inlier-only cost: hard 0/1 mask at the classification threshold,
        // so the reported cost reflects what survives outlier rejection.
        const INLIER_THRESHOLD: f64 = 0.5;
        let inlier_mask: Vec<f64> = weights
            .iter()
            .map(|&w| {
                if w.is_finite() {
                    if w >= INLIER_THRESHOLD {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            })
            .collect();
        let inlier_cost = self.robust_cost_weighted(&kernel_none, Some(&inlier_mask));

        Ok(BaGncResult {
            initial_cost,
            final_cost,
            inlier_cost,
            inlier_scale,
            observation_count: n,
            outer_iterations,
            converged,
            observation_weights: weights,
        })
    }

    /// Levenberg-Marquardt bundle adjustment with optional per-observation
    /// GNC weights folded into every reprojection contribution. `None`
    /// runs standard (optionally `RobustKernel`-IRLS) BA, bit-identical to
    /// the public [`Self::optimize`]; `Some(weights)` is the inner solve of
    /// [`Self::optimize_gnc`], where `weights` are the current
    /// Graduated-Non-Convexity surrogate weights and the cost used for the
    /// LM accept / reject test is correspondingly reweighted. `weights` is
    /// indexed monocular, rectified stereo, then general stereo (see
    /// [`Self::robust_cost_weighted`]).
    fn optimize_weighted(
        &mut self,
        config: &BaConfig,
        gnc_weights: Option<&[f64]>,
    ) -> Result<BaResult, BaError> {
        let intrinsics = self.intrinsics().ok_or(BaError::UnsupportedCameraModel)?;
        if self.poses.is_empty() {
            return Err(BaError::NoPoses);
        }
        let has_visual_observations = !self.observations.is_empty()
            || !self.stereo_observations.is_empty()
            || !self.general_stereo_observations.is_empty();
        let has_imu_factors = !self.imu_factors.is_empty();
        if !has_visual_observations && !has_imu_factors {
            return Err(BaError::NoObservations);
        }
        // Visual residuals require landmarks; inertial-only solves (used by
        // the motion-based VI initialiser's VIBA1 stage) do not.
        if has_visual_observations && self.landmarks.is_empty() {
            return Err(BaError::NoLandmarks);
        }
        for obs in &self.observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
        }
        if !self.stereo_observations.is_empty() {
            match self.stereo_baseline {
                Some(b) if b.is_finite() && b > 0.0 => {}
                _ => return Err(BaError::MissingStereoBaseline),
            }
            for obs in &self.stereo_observations {
                if !self.poses.contains_key(&obs.keyframe_id) {
                    return Err(BaError::MissingPose(obs.keyframe_id));
                }
                if !self.landmarks.contains_key(&obs.landmark_id) {
                    return Err(BaError::MissingLandmark(obs.landmark_id));
                }
            }
        }
        for obs in &self.general_stereo_observations {
            if !self.poses.contains_key(&obs.keyframe_id) {
                return Err(BaError::MissingPose(obs.keyframe_id));
            }
            if !self.landmarks.contains_key(&obs.landmark_id) {
                return Err(BaError::MissingLandmark(obs.landmark_id));
            }
            if obs.right_camera.intrinsics().is_none() {
                return Err(BaError::UnsupportedCameraModel);
            }
        }

        // Variable layout: only non-fixed entries get a slot in the linear
        // system. Fixed poses / landmarks contribute residuals but no Hessian
        // or gradient block.
        let mut pose_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.poses.keys() {
            if self.fixed_poses.contains(&id) {
                continue;
            }
            let next = pose_index.len();
            pose_index.insert(id, next);
        }
        let mut landmark_index: BTreeMap<u64, usize> = BTreeMap::new();
        for &id in self.landmarks.keys() {
            if self.fixed_landmarks.contains(&id) {
                continue;
            }
            let next = landmark_index.len();
            landmark_index.insert(id, next);
        }
        // Velocity slots: non-fixed velocities that appear on at least
        // one IMU factor. We DO NOT add a slot for every registered
        // velocity — only the ones the factor touches — so a stray
        // `add_velocity` without a corresponding `add_imu_factor` does
        // not introduce an unconstrained DoF that would singularise the
        // system.
        let mut velocity_index: BTreeMap<u64, usize> = BTreeMap::new();
        for factor in &self.imu_factors {
            for kf_id in [factor.keyframe_id_from, factor.keyframe_id_to] {
                if !self.velocities.contains_key(&kf_id) {
                    continue;
                }
                if self.fixed_velocities.contains(&kf_id) {
                    continue;
                }
                if velocity_index.contains_key(&kf_id) {
                    continue;
                }
                let next = velocity_index.len();
                velocity_index.insert(kf_id, next);
            }
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                if self.velocities.contains_key(&kf_id)
                    && !self.fixed_velocities.contains(&kf_id)
                    && !velocity_index.contains_key(&kf_id)
                {
                    let next = velocity_index.len();
                    velocity_index.insert(kf_id, next);
                }
            }
        }
        // Bias slots: non-fixed biases registered on the "from" side of
        // an IMU factor OR on either side of a bias random-walk factor.
        // Same singularity guard as `velocity_index` — a stray
        // `add_bias` without a matching factor does not introduce an
        // unconstrained DoF.
        let mut bias_index: BTreeMap<u64, usize> = BTreeMap::new();
        let register_bias_slot = |kf_id: u64, idx: &mut BTreeMap<u64, usize>| {
            if !self.biases.contains_key(&kf_id) {
                return;
            }
            if self.fixed_biases.contains(&kf_id) {
                return;
            }
            if idx.contains_key(&kf_id) {
                return;
            }
            let next = idx.len();
            idx.insert(kf_id, next);
        };
        for factor in &self.imu_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
        }
        for factor in &self.bias_random_walk_factors {
            register_bias_slot(factor.keyframe_id_from, &mut bias_index);
            register_bias_slot(factor.keyframe_id_to, &mut bias_index);
        }
        if let Some(prior) = &self.navigation_state_prior {
            for &kf_id in &prior.keyframe_ids {
                register_bias_slot(kf_id, &mut bias_index);
            }
        }
        if pose_index.is_empty()
            && landmark_index.is_empty()
            && velocity_index.is_empty()
            && bias_index.is_empty()
        {
            return Err(BaError::AllPosesFixed);
        }

        let kernel = config.robust_kernel;
        let initial_cost = self.robust_cost_weighted(&kernel, gnc_weights);
        let mut iterations: Vec<BaIterationStats> = Vec::with_capacity(config.max_iterations);
        let mut current_cost = initial_cost;
        let mut current_nonprojectable = self.nonprojectable_observation_count();
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;

        for iteration in 0..config.max_iterations {
            let mut system = build_normal_equations(
                self,
                &intrinsics,
                &pose_index,
                &landmark_index,
                &velocity_index,
                &bias_index,
                &kernel,
                gnc_weights,
                config.parallel,
            );
            constrain_fixed_pose_rotations(&self.fixed_pose_rotations, &pose_index, &mut system);

            // Build the reduced (Schur-complement) camera system. λ is added
            // to both the pose and landmark diagonals before reduction so the
            // augmented system stays SPD when the un-damped one is rank-
            // deficient (as monocular BA generally is).
            let saved_poses = self.poses.clone();
            let saved_landmarks = self.landmarks.clone();
            let saved_velocities = self.velocities.clone();
            let saved_biases = self.biases.clone();
            let cost_before = current_cost;

            let (delta_poses, delta_landmarks) = match solve_step(
                &system,
                pose_index.len(),
                landmark_index.len(),
                velocity_index.len(),
                bias_index.len(),
                lambda,
                config.linear_solver,
                config.parallel,
            ) {
                Ok(d) => d,
                Err(BaError::SingularSystem) => {
                    // Treat singular system the same as a rejected LM step:
                    // bump λ and retry.
                    lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                    iterations.push(BaIterationStats {
                        iteration,
                        cost_before,
                        cost_after: cost_before,
                        max_pose_step: 0.0,
                        max_landmark_step: 0.0,
                        lambda,
                        step_accepted: false,
                    });
                    if lambda >= config.max_lambda {
                        break;
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };

            // Apply tentative update. `delta_poses` packs pose slots
            // first (`i * 6 .. i * 6 + 6`), then velocity slots
            // (`6P + v * 3 .. 6P + v * 3 + 3`), then bias slots
            // (`6P + 3V + b * 6 .. 6P + 3V + b * 6 + 6`).
            let mut max_pose_step: f64 = 0.0;
            let vel_offset_in_delta = pose_index.len() * 6;
            let bias_offset_in_delta = vel_offset_in_delta + velocity_index.len() * 3;
            for (&id, &i) in &pose_index {
                let mut xi = delta_poses.fixed_rows::<6>(i * 6).into_owned();
                if self.fixed_pose_rotations.contains(&id) {
                    xi[3] = 0.0;
                    xi[4] = 0.0;
                    xi[5] = 0.0;
                }
                let xi_vec: Vector6<f64> = xi;
                let step = xi_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let pose = self.poses.get_mut(&id).expect("pose exists");
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi_vec));
            }
            for (&id, &v) in &velocity_index {
                let dv = delta_poses
                    .fixed_rows::<3>(vel_offset_in_delta + v * 3)
                    .into_owned();
                let dv_vec: Vector3<f64> = dv;
                let step = dv_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let v_ref = self.velocities.get_mut(&id).expect("velocity exists");
                *v_ref += dv_vec;
            }
            for (&id, &b) in &bias_index {
                let db = delta_poses
                    .fixed_rows::<6>(bias_offset_in_delta + b * 6)
                    .into_owned();
                let db_vec: Vector6<f64> = db;
                let step = db_vec.norm();
                if step > max_pose_step {
                    max_pose_step = step;
                }
                let b_ref = self.biases.get_mut(&id).expect("bias exists");
                *b_ref += db_vec;
            }
            let mut max_landmark_step: f64 = 0.0;
            for (&id, &i) in &landmark_index {
                let dx = delta_landmarks.fixed_rows::<3>(i * 3).into_owned();
                let v: Vector3<f64> = dx;
                let step = v.norm();
                if step > max_landmark_step {
                    max_landmark_step = step;
                }
                let pt = self.landmarks.get_mut(&id).expect("landmark exists");
                *pt = Point3::from(pt.coords + v);
            }

            let cost_after = self.robust_cost_weighted(&kernel, gnc_weights);
            let nonprojectable_after = self.nonprojectable_observation_count();
            let cost_accepted = match config.initial_lambda {
                None => true, // Pure GN: accept unconditionally.
                Some(_) => cost_after < cost_before,
            };
            let step_accepted = cost_accepted && nonprojectable_after <= current_nonprojectable;

            // In particular, expose the two independent acceptance gates for
            // camera-fixed (landmark-only) solves.  A high robust cost can be
            // dominated by observations that are already down-weighted; a
            // candidate can also lower that cost while making more points
            // non-projectable, in which case the feasibility gate correctly
            // rejects it.  This line is diagnostic-only and is never emitted
            // unless both explicit BA step environment flags are set.
            if ba_step_debug_enabled() && (pose_index.is_empty() || !step_accepted) {
                eprintln!(
                    concat!(
                        "sfm-debug-ba-step-detail: poses={} landmarks={} iteration={} ",
                        "accepted={} cost_gate={} feasibility_gate={} ",
                        "nonprojectable={}->{} cost={:.9e}->{:.9e} lambda={:.3e}"
                    ),
                    pose_index.len(),
                    landmark_index.len(),
                    iteration,
                    step_accepted,
                    cost_accepted,
                    nonprojectable_after <= current_nonprojectable,
                    current_nonprojectable,
                    nonprojectable_after,
                    cost_before,
                    cost_after,
                    lambda,
                );
            }

            if !step_accepted {
                self.poses = saved_poses;
                self.landmarks = saved_landmarks;
                self.velocities = saved_velocities;
                self.biases = saved_biases;
                lambda = (lambda * config.lambda_increase_factor).min(config.max_lambda);
                iterations.push(BaIterationStats {
                    iteration,
                    cost_before,
                    cost_after,
                    max_pose_step,
                    max_landmark_step,
                    lambda,
                    step_accepted: false,
                });
                if config.initial_lambda.is_none() {
                    break;
                }
                if lambda >= config.max_lambda {
                    break;
                }
                continue;
            }

            iterations.push(BaIterationStats {
                iteration,
                cost_before,
                cost_after,
                max_pose_step,
                max_landmark_step,
                lambda,
                step_accepted: true,
            });
            current_cost = cost_after;
            current_nonprojectable = nonprojectable_after;
            if config.initial_lambda.is_some() {
                lambda = (lambda * config.lambda_decrease_factor).max(config.min_lambda);
            }

            if max_pose_step < config.step_tolerance && max_landmark_step < config.step_tolerance {
                converged = true;
                break;
            }
            if (cost_before - cost_after).abs() < config.cost_tolerance {
                converged = true;
                break;
            }
            if config.relative_cost_tolerance.is_some_and(|tolerance| {
                tolerance.is_finite()
                    && tolerance >= 0.0
                    && (cost_before - cost_after) / cost_before.abs().max(f64::EPSILON) < tolerance
            }) {
                converged = true;
                break;
            }
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
}

fn clear_visual_and_structural_costs(ba: &mut BundleAdjustment) {
    ba.observations.clear();
    ba.stereo_observations.clear();
    ba.general_stereo_observations.clear();
    ba.gravity_prior = None;
    ba.per_pose_gravity_prior = None;
    ba.position_prior = None;
    ba.pairwise_pose_factors.clear();
}

/// Configuration for [`BundleAdjustment::optimize`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaConfig {
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
    /// `3×3` block inversion (since `H_LL` is block-diagonal).
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
    /// poses and (eliminated) landmarks — the COLMAP self-calibration formulation.
    /// This is the lever for unknown / inaccurate calibration: a wrong fixed focal
    /// forces a residual onto the poses, and the joint solve lets the camera absorb
    /// it. (The coupled, landmark-eliminated gradient is what makes this work; an
    /// alternating refinement against converged structure cannot move a wrong focal,
    /// because the structure-fixed gradient is ~0.) Only the 4-parameter
    /// [`CameraModel::Pinhole`] is refined; any other model falls back to the
    /// pose/structure-only solve. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub refine_intrinsics: bool,
    /// Additionally self-calibrate the two radial-distortion coefficients
    /// `(k1, k2)` jointly with the intrinsics (the camera block grows from 4 to 6).
    /// Requires `refine_intrinsics`; only applies to a **monocular** pinhole
    /// reconstruction (rectified stereo is already undistorted). The coefficients
    /// are appended to `Camera::params` as `[fx, fy, cx, cy, k1, k2]`. **`false`
    /// by default.**
    pub refine_distortion: bool,
    /// Run the per-observation assembly, per-landmark Schur reduction, and
    /// back-substitution loops of [`BundleAdjustment::optimize_weighted`]'s
    /// Levenberg-Marquardt iteration on the `rayon` pool (see the module's
    /// "Parallelism" section). The result is bit-identical to the serial
    /// path at any thread count — this only changes *how* the normal
    /// equations are computed, never the summation order — so it is safe to
    /// flip independently of everything else in this config. Small problems
    /// stay serial even when this is set (see `PARALLEL_MIN_OBSERVATIONS` /
    /// `PARALLEL_MIN_LANDMARKS`). Only consumed by `optimize_weighted`
    /// (the plain pose/structure/IMU solve); `optimize_joint_intrinsics`
    /// ignores it. **`false` by default** (the public
    /// [`BundleAdjustment::optimize`] is then bit-identical to before).
    pub parallel: bool,
}

impl Default for BaConfig {
    fn default() -> Self {
        Self {
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
            parallel: false,
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

/// Per-landmark contribution: the `H_ll` block, the `b_l` gradient, and
/// the `H_pl` cross blocks (one per pose that observed this landmark, in
/// arbitrary order). This is the only place the cross blocks are stored —
/// there is no full `H_PL` matrix.
struct LandmarkBlock {
    /// `3×3` Hessian summed over observations. Includes any λ damping.
    h_ll: Matrix3<f64>,
    /// `3-vec` gradient summed over observations.
    b_l: Vector3<f64>,
    /// `(pose_idx, J_pose^T · J_lm)` per observation that touches a
    /// non-fixed pose. Shape `6×3`.
    cross: Vec<(usize, Matrix6x3<f64>)>,
}

/// Per-landmark block for the joint pose+structure+intrinsics solve
/// ([`BundleAdjustment::optimize_joint_intrinsics`]). Unlike [`LandmarkBlock`]
/// the cross blocks are keyed by the touching camera-block column-start (a pose
/// block of width 6, or the shared 4-wide intrinsics block) so a single map
/// holds both the pose and intrinsics couplings, and observations sharing a
/// camera block (e.g. the intrinsics, seen by every observation) accumulate.
struct JointLandmarkBlock {
    id: u64,
    h_ll: Matrix3<f64>,
    b_l: Vector3<f64>,
    /// `column_start → Σ_obs Jᵀ_cam · J_lm` (`rows × 3`, `rows ∈ {6, 4}`).
    cross: BTreeMap<usize, DMatrix<f64>>,
}

/// Accumulate a `rows × 3` cross block into the per-landmark map at `col_start`.
fn navigation_prior_delta(
    ba: &BundleAdjustment,
    prior: &NavigationStatePrior,
) -> Option<DVector<f64>> {
    if !prior.is_well_formed() {
        return None;
    }
    let mut delta = DVector::zeros(prior.keyframe_ids.len() * 15);
    for (slot, id) in prior.keyframe_ids.iter().enumerate() {
        let pose = ba.poses.get(id)?;
        let velocity = ba.velocities.get(id)?;
        let bias = ba.biases.get(id)?;
        let reference_pose = prior.reference_poses.get(id)?;
        let reference_velocity = prior.reference_velocities.get(id)?;
        let reference_bias = prior.reference_biases.get(id)?;
        let pose_delta = reference_pose
            .world_to_camera
            .inverse()
            .compose(&pose.world_to_camera)
            .log();
        delta.fixed_rows_mut::<6>(slot * 15).copy_from(&pose_delta);
        delta
            .fixed_rows_mut::<3>(slot * 15 + 6)
            .copy_from(&(velocity - reference_velocity));
        delta
            .fixed_rows_mut::<6>(slot * 15 + 9)
            .copy_from(&(bias - reference_bias));
    }
    Some(delta)
}

fn add_cross(
    cross: &mut BTreeMap<usize, DMatrix<f64>>,
    col_start: usize,
    rows: usize,
    blk: &DMatrix<f64>,
) {
    *cross
        .entry(col_start)
        .or_insert_with(|| DMatrix::zeros(rows, 3)) += blk;
}

/// Output of the per-iteration normal-equations build.
struct NormalEquationsBa {
    /// Pose-pose Hessian, dense `(6P) × (6P)`.
    h_pp: DMatrix<f64>,
    /// Pose gradient, dense `6P`.
    b_p: DVector<f64>,
    /// Landmark blocks indexed by landmark variable index.
    landmarks: Vec<LandmarkBlock>,
}

// --- Parallel assembly / Schur-reduction support (see the module's
// "Parallelism" section). ---

/// Below this many observations, [`assemble_mono_observations_parallel`] is
/// not dispatched even when [`BaConfig::parallel`] is set: a full-sequence
/// BA's per-observation cost dwarfs the rayon per-chunk overhead, but small
/// problems (a handful of keyframes) do not, so they stay on the plain
/// serial loop. Mirrors `block_cholesky::PARALLEL_MIN_BLOCKS`.
const PARALLEL_MIN_OBSERVATIONS: usize = 4_096;
/// Observations computed per rayon chunk in the parallel assembly path.
/// Bounds the transient `Vec<Option<MonoObsContribution>>` buffer to a few
/// megabytes regardless of the total observation count — a full-sequence BA
/// can carry tens of millions of observations, so materializing one
/// contribution per observation up front (rather than chunk by chunk) would
/// multiply the assembly's peak memory several-fold. The value does not
/// affect the result at all (see the merge comment on
/// [`assemble_mono_observations_parallel`]), so it is chosen purely for that
/// memory/dispatch-overhead trade-off.
const PARALLEL_OBSERVATION_CHUNK: usize = 65_536;

/// Below this many landmarks, the parallel Schur-reduction and back-
/// substitution paths in [`solve_step`] are not dispatched. Mirrors
/// [`PARALLEL_MIN_OBSERVATIONS`].
const PARALLEL_MIN_LANDMARKS: usize = 2_048;
/// Landmarks processed per rayon chunk in the parallel Schur reduction.
/// Bounds the transient per-chunk `(pose, pose, block)` triplet buffer the
/// same way [`PARALLEL_OBSERVATION_CHUNK`] bounds the assembly path's; it
/// does not affect the result (same reasoning as that constant).
const PARALLEL_LANDMARK_CHUNK: usize = 16_384;

/// Precomputed contribution of one monocular observation to the normal
/// equations: the weighted `H_pp` / `b_p` block for the observing pose (if
/// non-fixed), the weighted `H_ll` / `b_l` block for the observed landmark
/// (if non-fixed), and their cross term (if both are). `None` when the
/// observation is skipped exactly as the serial loop in
/// [`build_normal_equations`] skips it — behind the camera or not
/// projectable. This is the unit of work
/// [`assemble_mono_observations_parallel`] farms out to the rayon pool: it
/// is a pure function of the current pose/landmark estimate, so many can be
/// computed concurrently with no shared mutable state.
struct MonoObsContribution {
    pose: Option<(usize, Matrix6<f64>, Vector6<f64>)>,
    landmark: Option<(usize, Matrix3<f64>, Vector3<f64>)>,
    cross: Option<(usize, usize, Matrix6x3<f64>)>,
}

/// Compute one observation's [`MonoObsContribution`]. Deliberately kept
/// byte-for-byte in step with the inline loop body in
/// [`build_normal_equations`] (same operations in the same order, so the
/// weighted blocks are bitwise identical to what that loop would compute) —
/// if the residual/Jacobian model there ever changes, this must change with
/// it.
#[allow(clippy::too_many_arguments)]
fn compute_mono_contribution(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    obs_idx: usize,
    obs: &BaObservation,
) -> Option<MonoObsContribution> {
    let pose = &ba.poses[&obs.keyframe_id];
    let point = &ba.landmarks[&obs.landmark_id];
    let r_mat = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let xc = pose.transform_world_point(point);
    if xc.z <= 0.0 {
        return None;
    }
    let predicted = project_pinhole(intrinsics, &xc)?;
    let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

    let (fx, fy, _, _) = *intrinsics;
    let z_inv = 1.0 / xc.z;
    let mut j_pi = Matrix2x3::<f64>::zeros();
    j_pi[(0, 0)] = fx * z_inv;
    j_pi[(0, 1)] = 0.0;
    j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
    j_pi[(1, 0)] = 0.0;
    j_pi[(1, 1)] = fy * z_inv;
    j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;

    let xw_skew = skew(&point.coords);
    let mut dx_dxi = nalgebra::Matrix3x6::<f64>::zeros();
    dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
    dx_dxi
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_mat * xw_skew));
    let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
    let j_lm: Matrix2x3<f64> = j_pi * r_mat;

    let s = residual.x * residual.x + residual.y * residual.y;
    let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[obs_idx]);

    let i_pose = pose_index.get(&obs.keyframe_id).copied();
    let i_lm = landmark_index.get(&obs.landmark_id).copied();

    let pose_update = i_pose.map(|p| {
        let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
        let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
        (p, w * h_pp_block, w * b_p_block)
    });
    let landmark_update = i_lm.map(|l| {
        let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
        let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
        (l, w * h_ll_block, w * b_l_block)
    });
    let cross_update = match (i_pose, i_lm) {
        (Some(p), Some(l)) => {
            let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
            Some((p, l, w * cross))
        }
        _ => None,
    };

    Some(MonoObsContribution {
        pose: pose_update,
        landmark: landmark_update,
        cross: cross_update,
    })
}

/// Scatter one [`MonoObsContribution`] into the shared accumulators, in
/// exactly the pose-then-landmark-then-cross order the serial loop in
/// [`build_normal_equations`] uses. Never called concurrently on the same
/// `h_pp` / `b_p` / `landmarks` — see the serial merge loop in
/// [`assemble_mono_observations_parallel`].
fn apply_mono_contribution(
    contribution: MonoObsContribution,
    h_pp: &mut DMatrix<f64>,
    b_p: &mut DVector<f64>,
    landmarks: &mut [LandmarkBlock],
) {
    if let Some((p, h_pp_block, b_p_block)) = contribution.pose {
        for r in 0..6 {
            for c in 0..6 {
                h_pp[(p * 6 + r, p * 6 + c)] += h_pp_block[(r, c)];
            }
            b_p[p * 6 + r] += b_p_block[r];
        }
    }
    if let Some((l, h_ll_block, b_l_block)) = contribution.landmark {
        landmarks[l].h_ll += h_ll_block;
        landmarks[l].b_l += b_l_block;
    }
    if let Some((p, l, cross)) = contribution.cross {
        landmarks[l].cross.push((p, cross));
    }
}

#[allow(clippy::too_many_arguments)]
/// Parallel counterpart of the monocular loop in [`build_normal_equations`].
///
/// Processes observations in fixed-size chunks
/// ([`PARALLEL_OBSERVATION_CHUNK`]): within a chunk, every observation's
/// [`MonoObsContribution`] is computed concurrently on the rayon pool (pure
/// function, no shared state) and collected into a `Vec` that preserves the
/// original index order exactly like the serial loop would produce it; the
/// chunk's contributions are then scattered into `h_pp` / `b_p` /
/// `landmarks` by a single serial pass, *in ascending observation-index
/// order*, before the next chunk starts. Because the scatter order is
/// therefore always identical to what the plain serial loop would produce —
/// only the (order-independent) computation of each contribution moved to
/// the pool — the result is bit-identical to the serial path at any thread
/// count or chunk size, unlike a reassociating parallel reduction.
fn assemble_mono_observations_parallel(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    h_pp: &mut DMatrix<f64>,
    b_p: &mut DVector<f64>,
    landmarks: &mut [LandmarkBlock],
) {
    use rayon::prelude::*;

    let mut start = 0;
    while start < ba.observations.len() {
        let end = (start + PARALLEL_OBSERVATION_CHUNK).min(ba.observations.len());
        let chunk = &ba.observations[start..end];
        let contributions: Vec<Option<MonoObsContribution>> = chunk
            .par_iter()
            .enumerate()
            .map(|(offset, obs)| {
                compute_mono_contribution(
                    ba,
                    intrinsics,
                    kernel,
                    gnc_weights,
                    pose_index,
                    landmark_index,
                    start + offset,
                    obs,
                )
            })
            .collect();
        for contribution in contributions.into_iter().flatten() {
            apply_mono_contribution(contribution, h_pp, b_p, landmarks);
        }
        start = end;
    }
}

#[allow(clippy::too_many_arguments)]
fn build_normal_equations(
    ba: &BundleAdjustment,
    intrinsics: &(f64, f64, f64, f64),
    pose_index: &BTreeMap<u64, usize>,
    landmark_index: &BTreeMap<u64, usize>,
    velocity_index: &BTreeMap<u64, usize>,
    bias_index: &BTreeMap<u64, usize>,
    kernel: &RobustKernel,
    gnc_weights: Option<&[f64]>,
    parallel: bool,
) -> NormalEquationsBa {
    let p_count = pose_index.len();
    let l_count = landmark_index.len();
    let v_count = velocity_index.len();
    let b_count = bias_index.len();
    // Joint pose+velocity+bias Hessian. Pose slots occupy rows/cols
    // `0 .. 6P`; velocity slots occupy `6P .. 6P + 3V`; bias slots
    // occupy `6P + 3V .. 6P + 3V + 6B`. When IMU factors are absent
    // (`V = 0, B = 0`) the matrix layout is identical to the legacy
    // pose-only system, so the existing call sites that pass empty
    // velocity / bias indices see no change.
    let pose_dim = p_count * 6;
    let vel_offset = pose_dim;
    let bias_offset = pose_dim + v_count * 3;
    let total_dim = pose_dim + v_count * 3 + b_count * 6;
    let mut h_pp = DMatrix::<f64>::zeros(total_dim, total_dim);
    let mut b_p = DVector::<f64>::zeros(total_dim);
    let mut landmarks: Vec<LandmarkBlock> = (0..l_count)
        .map(|_| LandmarkBlock {
            h_ll: Matrix3::zeros(),
            b_l: Vector3::zeros(),
            cross: Vec::new(),
        })
        .collect();

    // Flag-gated, work-gated parallel assembly (see the module's
    // "Parallelism" section): bit-identical to the plain loop below at any
    // thread count, so the branch only changes how the contributions are
    // computed, never the result.
    if parallel && ba.observations.len() >= PARALLEL_MIN_OBSERVATIONS {
        assemble_mono_observations_parallel(
            ba,
            intrinsics,
            pose_index,
            landmark_index,
            kernel,
            gnc_weights,
            &mut h_pp,
            &mut b_p,
            &mut landmarks,
        );
    } else {
        for (obs_idx, obs) in ba.observations.iter().enumerate() {
            let pose = &ba.poses[&obs.keyframe_id];
            let point = &ba.landmarks[&obs.landmark_id];
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            // Predicted pixel - measured pixel.
            let predicted = match project_pinhole(intrinsics, &xc) {
                Some(p) => p,
                None => continue,
            };
            let residual = Vector2::new(predicted.x - obs.xy.x, predicted.y - obs.xy.y);

            // Projection Jacobian J_π (2×3) at X_c = (X, Y, Z):
            //   J_π = (1/Z) [[fx, 0, -fx X/Z], [0, fy, -fy Y/Z]]
            let (fx, fy, _, _) = *intrinsics;
            let z_inv = 1.0 / xc.z;
            let mut j_pi = Matrix2x3::<f64>::zeros();
            j_pi[(0, 0)] = fx * z_inv;
            j_pi[(0, 1)] = 0.0;
            j_pi[(0, 2)] = -fx * xc.x * z_inv * z_inv;
            j_pi[(1, 0)] = 0.0;
            j_pi[(1, 1)] = fy * z_inv;
            j_pi[(1, 2)] = -fy * xc.y * z_inv * z_inv;

            // Right perturbation pose Jacobian:
            //   ∂X_c / ∂[ρ; ω] = [R, -R · [X_w]_×]   (3×6)
            let xw_skew = skew(&point.coords);
            let mut dx_dxi = nalgebra::Matrix3x6::<f64>::zeros();
            dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
            dx_dxi
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-r_mat * xw_skew));
            let j_pose: Matrix2x6<f64> = j_pi * dx_dxi;
            // Landmark Jacobian: ∂X_c / ∂X_w = R, so J_lm = J_π · R (2×3).
            let j_lm: Matrix2x3<f64> = j_pi * r_mat;

            // IRLS weight applied per-observation. With `RobustKernel::None`
            // this is `1.0` and the build matches plain Gauss-Newton; with
            // Huber / Cauchy the weight shrinks for large residuals so a
            // single bad observation cannot dominate the normal equations.
            let s = residual.x * residual.x + residual.y * residual.y;
            let w = kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[obs_idx]);

            let i_pose = pose_index.get(&obs.keyframe_id).copied();
            let i_lm = landmark_index.get(&obs.landmark_id).copied();

            if let Some(p) = i_pose {
                let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
                let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
                for r in 0..6 {
                    for c in 0..6 {
                        h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                    }
                    b_p[p * 6 + r] += w * b_p_block[r];
                }
            }
            if let Some(l) = i_lm {
                let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
                let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
                landmarks[l].h_ll += w * h_ll_block;
                landmarks[l].b_l += w * b_l_block;
            }
            if let (Some(p), Some(l)) = (i_pose, i_lm) {
                let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
                landmarks[l].cross.push((p, w * cross));
            }
        }
    }

    // Stereo observations: 3D residual `(u_l, v_l, u_r)` with
    // `u_r_pred = u_l_pred − fx · b / Z`. Jacobian of the residual w.r.t.
    // `X_c = (X, Y, Z)` is the 3×3 matrix
    //   J_π_st = [[fx/Z, 0,    -fx·X/Z²       ],
    //             [0,    fy/Z, -fy·Y/Z²       ],
    //             [fx/Z, 0,    -fx·(X-b)/Z²   ]].
    // The pose / landmark Jacobians have shape 3×6 / 3×3, but the
    // accumulated `H_pp = J^T J` and `H_pl = J^T J_lm` blocks have the
    // same 6×6 / 6×3 / 3×3 shapes as the monocular path so the rest of
    // the pipeline (Schur complement, back-substitution) is unchanged.
    if !ba.stereo_observations.is_empty() {
        if let Some(baseline) = ba.stereo_baseline {
            if baseline.is_finite() && baseline > 0.0 {
                let (fx, fy, _, _) = *intrinsics;
                let stereo_offset = ba.observations.len();
                for (st_idx, obs) in ba.stereo_observations.iter().enumerate() {
                    let pose = &ba.poses[&obs.keyframe_id];
                    let point = &ba.landmarks[&obs.landmark_id];
                    let r_mat = pose
                        .world_to_camera
                        .rotation
                        .to_rotation_matrix()
                        .into_inner();
                    let xc = pose.transform_world_point(point);
                    if xc.z <= 0.0 {
                        continue;
                    }
                    let predicted = match project_pinhole(intrinsics, &xc) {
                        Some(p) => p,
                        None => continue,
                    };
                    let u_r_pred = predicted.x - fx * baseline / xc.z;
                    let residual = Vector3::new(
                        predicted.x - obs.xy.x,
                        predicted.y - obs.xy.y,
                        u_r_pred - obs.u_right,
                    );

                    let z_inv = 1.0 / xc.z;
                    let z_inv2 = z_inv * z_inv;
                    let mut j_pi = Matrix3::<f64>::zeros();
                    j_pi[(0, 0)] = fx * z_inv;
                    j_pi[(0, 2)] = -fx * xc.x * z_inv2;
                    j_pi[(1, 1)] = fy * z_inv;
                    j_pi[(1, 2)] = -fy * xc.y * z_inv2;
                    j_pi[(2, 0)] = fx * z_inv;
                    j_pi[(2, 2)] = -fx * (xc.x - baseline) * z_inv2;

                    let xw_skew = skew(&point.coords);
                    let mut dx_dxi = Matrix3x6::<f64>::zeros();
                    dx_dxi.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_mat);
                    dx_dxi
                        .fixed_view_mut::<3, 3>(0, 3)
                        .copy_from(&(-r_mat * xw_skew));
                    let j_pose: Matrix3x6<f64> = j_pi * dx_dxi;
                    let j_lm: Matrix3<f64> = j_pi * r_mat;

                    let s = residual.norm_squared();
                    let w =
                        kernel.weight(s) * gnc_weights.map_or(1.0, |gw| gw[stereo_offset + st_idx]);

                    let i_pose = pose_index.get(&obs.keyframe_id).copied();
                    let i_lm = landmark_index.get(&obs.landmark_id).copied();

                    if let Some(p) = i_pose {
                        let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
                        let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
                        for r in 0..6 {
                            for c in 0..6 {
                                h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                            }
                            b_p[p * 6 + r] += w * b_p_block[r];
                        }
                    }
                    if let Some(l) = i_lm {
                        let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
                        let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
                        landmarks[l].h_ll += w * h_ll_block;
                        landmarks[l].b_l += w * b_l_block;
                    }
                    if let (Some(p), Some(l)) = (i_pose, i_lm) {
                        let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
                        landmarks[l].cross.push((p, w * cross));
                    }
                }
            }
        }
    }

    // Calibrated non-rectified stereo: four residual rows `(u_l, v_l, u_r,
    // v_r)`. The helper composes `T_right<-left` before right projection and
    // stacks both cameras' Jacobians into the same pose/landmark blocks, so
    // the Schur structure remains unchanged.
    let general_offset = ba.observations.len() + ba.stereo_observations.len();
    for (index, obs) in ba.general_stereo_observations.iter().enumerate() {
        let pose = &ba.poses[&obs.keyframe_id];
        let point = &ba.landmarks[&obs.landmark_id];
        let Some((residual, j_pose, j_lm)) =
            general_stereo_residual_jacobians(intrinsics, obs, pose, point)
        else {
            continue;
        };
        let s = residual.norm_squared();
        let w =
            kernel.weight(s) * gnc_weights.map_or(1.0, |weights| weights[general_offset + index]);
        let i_pose = pose_index.get(&obs.keyframe_id).copied();
        let i_lm = landmark_index.get(&obs.landmark_id).copied();
        if let Some(p) = i_pose {
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * residual;
            for r in 0..6 {
                for c in 0..6 {
                    h_pp[(p * 6 + r, p * 6 + c)] += w * h_pp_block[(r, c)];
                }
                b_p[p * 6 + r] += w * b_p_block[r];
            }
        }
        if let Some(l) = i_lm {
            let h_ll_block: Matrix3<f64> = j_lm.transpose() * j_lm;
            let b_l_block: Vector3<f64> = j_lm.transpose() * residual;
            landmarks[l].h_ll += w * h_ll_block;
            landmarks[l].b_l += w * b_l_block;
        }
        if let (Some(p), Some(l)) = (i_pose, i_lm) {
            let cross: Matrix6x3<f64> = j_pose.transpose() * j_lm;
            landmarks[l].cross.push((p, w * cross));
        }
    }

    // Optional per-pose absolute-position prior. Residual per
    // observation: r = C_w − target, where C_w = −Rᵀt is the world
    // camera centre. From the gravity-prior derivation:
    //   d C_w / d ρ = −I,   d C_w / d ω = [C_w]_×.
    // Per-axis stiffness is applied as a 3×3 diagonal scaling so a
    // zero entry collapses that row to zero — i.e. drops it from the
    // normal equations entirely.
    if let Some(prior) = &ba.position_prior {
        for obs in &prior.observations {
            let Some(&p) = pose_index.get(&obs.keyframe_id) else {
                continue;
            };
            let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
                continue;
            };
            let c_world = pose.camera_center_world();
            let r_vec: Vector3<f64> = c_world - obs.camera_center_world;
            let c_skew = skew(&c_world.coords);
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            j_pose
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&-Matrix3::identity());
            if prior.couple_rotation {
                j_pose.fixed_view_mut::<3, 3>(0, 3).copy_from(&c_skew);
            }
            // Apply √w on each residual row so JᵀJ and Jᵀr come out
            // axis-weighted exactly as in the cost.
            let sqrt_w_x = obs.axis_weights.x.max(0.0).sqrt();
            let sqrt_w_y = obs.axis_weights.y.max(0.0).sqrt();
            let sqrt_w_z = obs.axis_weights.z.max(0.0).sqrt();
            let mut weighted_j = j_pose;
            weighted_j.row_mut(0).scale_mut(sqrt_w_x);
            weighted_j.row_mut(1).scale_mut(sqrt_w_y);
            weighted_j.row_mut(2).scale_mut(sqrt_w_z);
            let mut weighted_r = r_vec;
            weighted_r.x *= sqrt_w_x;
            weighted_r.y *= sqrt_w_y;
            weighted_r.z *= sqrt_w_z;
            let h_pp_block: Matrix6<f64> = weighted_j.transpose() * weighted_j;
            let b_p_block: Vector6<f64> = weighted_j.transpose() * weighted_r;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += b_p_block[rr];
            }
        }
    }

    // Pairwise relative-pose factors. For each factor:
    //   r = log(meas⁻¹ · T_j · T_iⁱ)
    //   ∂r/∂δ_j =  Ad(T_i)   (6×6)
    //   ∂r/∂δ_i = -Ad(T_i)
    // Hessian/gradient contributions per factor (with scalar weight w):
    //   H[j,j] += w · AdᵀAd     b[j] += w · Adᵀ · r
    //   H[i,i] += w · AdᵀAd     b[i] -= w · Adᵀ · r
    //   H[j,i] -= w · AdᵀAd     H[i,j] = (H[j,i])ᵀ
    // Diagonal block AdᵀAd is shared (same magnitude on both sides);
    // the off-diagonal block carries a minus sign because the from
    // Jacobian is the negative of the to Jacobian.
    for factor in &ba.pairwise_pose_factors {
        let (Some(t_from_pose), Some(t_to_pose)) = (
            ba.poses.get(&factor.keyframe_id_from),
            ba.poses.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let t_from = &t_from_pose.world_to_camera;
        let t_to = &t_to_pose.world_to_camera;
        let predicted = t_to.compose(&t_from.inverse());
        let r = factor
            .measurement
            .world_to_camera
            .inverse()
            .compose(&predicted)
            .log();
        let ad_from = t_from.adjoint();
        let ata: Matrix6<f64> = ad_from.transpose() * ad_from;
        let atr: Vector6<f64> = ad_from.transpose() * r;
        let w = factor.weight;
        let i_to = pose_index.get(&factor.keyframe_id_to).copied();
        let i_from = pose_index.get(&factor.keyframe_id_from).copied();
        if let Some(j) = i_to {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(j * 6 + rr, j * 6 + cc)] += w * ata[(rr, cc)];
                }
                b_p[j * 6 + rr] += w * atr[rr];
            }
        }
        if let Some(i) = i_from {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(i * 6 + rr, i * 6 + cc)] += w * ata[(rr, cc)];
                }
                b_p[i * 6 + rr] -= w * atr[rr];
            }
        }
        if let (Some(j), Some(i)) = (i_to, i_from) {
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(j * 6 + rr, i * 6 + cc)] -= w * ata[(rr, cc)];
                    h_pp[(i * 6 + rr, j * 6 + cc)] -= w * ata[(cc, rr)];
                }
            }
        }
    }

    // Optional gravity-alignment prior on every non-fixed pose.
    //
    // Residual per pose: r = R_wc · g_world − g_camera_observed (3-vec).
    // Under right perturbation T_new = T_old · exp([ρ; ω]):
    //     R_new = R_old · R(ω) ≈ R_old · (I + [ω]_×)
    //     R_new · g_w ≈ R_old · g_w − R_old · [g_w]_× · ω
    // so the Jacobian w.r.t. xi = [ρ; ω] is J = [0_3×3 | −R_old · [g_w]_×].
    // Translation does not appear, which leaves the gauge of horizontal
    // translation completely unconstrained — this prior fixes rotation
    // ambiguity only, not translation drift.
    if let Some(prior) = &ba.gravity_prior {
        for (&pose_id, pose) in &ba.poses {
            let Some(&p) = pose_index.get(&pose_id) else {
                continue;
            };
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_vec: Vector3<f64> = r_mat * prior.g_world - prior.g_camera_observed;
            let g_skew = skew(&prior.g_world);
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            // ∂r/∂ρ = 0, ∂r/∂ω = −R · [g_w]_×.
            j_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-(r_mat * g_skew)));
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * r_vec;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += prior.weight * h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += prior.weight * b_p_block[rr];
            }
        }
    }

    // Per-keyframe gravity-alignment prior. Same residual / Jacobian
    // shape as the global gravity prior above, but the observation is
    // sourced per-keyframe (and pose entries not in the observations
    // list contribute nothing).
    if let Some(prior) = &ba.per_pose_gravity_prior {
        let g_skew = skew(&prior.g_world);
        for obs in &prior.observations {
            let Some(pose) = ba.poses.get(&obs.keyframe_id) else {
                continue;
            };
            let Some(&p) = pose_index.get(&obs.keyframe_id) else {
                continue;
            };
            // Effective stiffness combines the prior's global scale and the
            // per-observation multiplier; zero per-obs weight mutes the
            // observation without removing its slot.
            let stiffness = prior.weight * obs.weight;
            if stiffness == 0.0 {
                continue;
            }
            let r_mat = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let r_vec: Vector3<f64> = r_mat * prior.g_world - obs.g_camera_observed;
            let mut j_pose = nalgebra::Matrix3x6::<f64>::zeros();
            j_pose
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-(r_mat * g_skew)));
            let h_pp_block: Matrix6<f64> = j_pose.transpose() * j_pose;
            let b_p_block: Vector6<f64> = j_pose.transpose() * r_vec;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += stiffness * h_pp_block[(rr, cc)];
                }
                b_p[p * 6 + rr] += stiffness * b_p_block[rr];
            }
        }
    }

    // Forster 2017 IMU pre-integration factors.
    //
    // The factor binds (pose, velocity) at keyframe i and j via the
    // gravity-compensated relative measurement (ΔR, Δv, Δp). Residual
    // and Jacobians follow Forster eq. 45-47 with the BA right-perturbation
    // convention (T_new = T_old · exp([ρ; ω])):
    //
    //   r_R = log(ΔR.T · R_iᵀ · R_j)            (Forster's R_i = R_wbᵢ)
    //   r_v = R_bwᵢ · (v_j − v_i − g·Δt) − Δv
    //   r_p = R_bwᵢ · (B_j − B_i − v_i·Δt − ½ g Δt²) − Δp
    //
    // Right-perturbation Jacobians (ρ = translation perturbation, ω =
    // rotation perturbation, both 3-vec; world body centre
    // B = −Rᵀt so ∂B/∂ρ = −I, ∂B/∂ω = [B]×). Since
    // T_bw = T_bc T_cw, a right perturbation of T_cw is the same right
    // perturbation of T_bw, so no additional adjoint is required:
    //
    //   ∂r_R/∂ω_i =  Jr_inv(r_R) · R_bwⱼ
    //   ∂r_R/∂ω_j = −Jr_inv(r_R) · R_bwⱼ
    //   ∂r_v/∂ω_i = −R_bwᵢ · [v_j − v_i − g·Δt]×
    //   ∂r_v/∂v_i = −R_bwᵢ
    //   ∂r_v/∂v_j =  R_bwᵢ
    //   ∂r_p/∂ρ_i =  R_bwᵢ            ∂r_p/∂ρ_j = −R_bwᵢ
    //   ∂r_p/∂ω_i = −R_bwᵢ · [B_j − v_i·Δt − ½ g Δt²]×
    //   ∂r_p/∂ω_j =  R_bwᵢ · [B_j]×
    //   ∂r_p/∂v_i = −Δt · R_bwᵢ
    //
    // The 9-vector residual is stacked [r_R; r_v; r_p] with axis-wise
    // weights `sqrt(weight_rotation, weight_velocity, weight_position)`
    // applied as a per-block √w scaling so JᵀJ / Jᵀr come out
    // axis-weighted.
    for factor in &ba.imu_factors {
        let (Some(pose_i), Some(pose_j)) = (
            ba.poses.get(&factor.keyframe_id_from),
            ba.poses.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let (Some(v_i_w), Some(v_j_w)) = (
            ba.velocities.get(&factor.keyframe_id_from),
            ba.velocities.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let body_i = ba.imu_body_to_camera.compose(&pose_i.world_to_camera);
        let body_j = ba.imu_body_to_camera.compose(&pose_j.world_to_camera);
        let r_wc_i = body_i.rotation.to_rotation_matrix().into_inner();
        let r_wc_j = body_j.rotation.to_rotation_matrix().into_inner();
        let c_i: Vector3<f64> = body_i.inverse().translation;
        let c_j: Vector3<f64> = body_j.inverse().translation;
        let dt = factor.delta.delta_time;
        let g = factor.gravity_world;

        // Residual (use the same formulation as the factor's residual()
        // helper so the cost evaluation and Jacobian linearise at the
        // same point). When the "from" keyframe has a registered bias,
        // apply the first-order bias correction; otherwise fall back
        // to the un-corrected residual (the linearisation bias is
        // implicit in the integrated delta).
        let r_i_so3 = SO3::from_quaternion(body_i.rotation.inverse());
        let r_j_so3 = SO3::from_quaternion(body_j.rotation.inverse());
        let bias_for_factor = ba.biases.get(&factor.keyframe_id_from);
        let [r_rot, r_vel, r_pos] = if let Some(bias) = bias_for_factor {
            let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
            let ba_acc: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
            factor.residual_with_bias_correction(
                &r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w, &bg, &ba_acc,
            )
        } else {
            factor.residual(&r_i_so3, &c_i, v_i_w, &r_j_so3, &c_j, v_j_w)
        };

        // Full preintegration-covariance whitening when available; legacy
        // hand-tuned block weights remain the exact zero-covariance fallback.
        let whitener = factor.covariance_sqrt_information().unwrap_or_else(|| {
            let mut diagonal = nalgebra::SVector::<f64, 9>::zeros();
            diagonal
                .fixed_rows_mut::<3>(0)
                .fill(factor.weight_rotation.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(3)
                .fill(factor.weight_velocity.max(0.0).sqrt());
            diagonal
                .fixed_rows_mut::<3>(6)
                .fill(factor.weight_position.max(0.0).sqrt());
            crate::imu_preintegration::Matrix9::from_diagonal(&diagonal)
        });

        let q_diff = v_j_w - v_i_w - g * dt;
        let q_pos_i = c_j - v_i_w * dt - 0.5 * g * dt * dt;
        let jr_inv = right_jacobian_inverse_so3(&r_rot);
        let jr_inv_rwc_j = jr_inv * r_wc_j;

        // Build the 9×N Jacobian blocks per side, with √w scaling
        // baked into each row block so the JᵀJ / Jᵀr accumulation does
        // the right axis-weighting automatically.
        // J_pose_i: 9×6, columns [ρ_i | ω_i].
        let mut j_pose_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        // r_R block: ω_i column = Jr_inv · R_wc_j.
        j_pose_i
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&jr_inv_rwc_j);
        // r_v block: ω_i column = −R_wcᵢ · [q_diff]×.
        j_pose_i
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-r_wc_i * skew(&q_diff)));
        // r_p block: ρ_i = R_wcᵢ, ω_i = −R_wcᵢ · [q_pos_i]×.
        j_pose_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_wc_i);
        j_pose_i
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-r_wc_i * skew(&q_pos_i)));

        // J_pose_j: 9×6, columns [ρ_j | ω_j].
        let mut j_pose_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        j_pose_j
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-jr_inv_rwc_j));
        // r_v has no R_j / p_j dependence (block stays zero).
        // r_p block: ρ_j = −R_wcᵢ, ω_j = R_wcᵢ · [C_j]×.
        j_pose_j.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-r_wc_i));
        j_pose_j
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(r_wc_i * skew(&c_j)));

        // J_vel_i / J_vel_j: 9×3 each.
        let mut j_vel_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_i.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_wc_i));
        j_vel_i
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-dt * r_wc_i));
        let mut j_vel_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_wc_i);

        // Residual before applying the common full-row whitening transform.
        let mut r_stack = nalgebra::SVector::<f64, 9>::zeros();
        r_stack.fixed_rows_mut::<3>(0).copy_from(&r_rot);
        r_stack.fixed_rows_mut::<3>(3).copy_from(&r_vel);
        r_stack.fixed_rows_mut::<3>(6).copy_from(&r_pos);

        // Bias Jacobian (9×6, columns [δb_g | δb_a]) at keyframe i:
        //
        //   ∂r_R/∂δb_g = −Jr⁻¹(r_R) · Exp(−r_R) · J_R_bg
        //   ∂r_R/∂δb_a = 0
        //   ∂r_v/∂δb_g = −J_v_bg                 ∂r_v/∂δb_a = −J_v_ba
        //   ∂r_p/∂δb_g = −J_p_bg                 ∂r_p/∂δb_a = −J_p_ba
        //
        // Forster eq. 159 (simplified by dropping the `Jr(J_R · δb)` factor,
        // which equals identity at the linearisation point and is ~I for
        // the small `|J_R · δb|` regime where biases live in practice).
        // The `J_*_b*` matrices are the bias Jacobians stored on the
        // pre-integrated delta (Forster eq. 35-39); see
        // [`crate::imu_preintegration`].
        let i_bias = bias_index.get(&factor.keyframe_id_from).copied();
        let j_bias_block = if i_bias.is_some() {
            let neg_r_rot_mat: Matrix3<f64> = nalgebra::Rotation3::from_scaled_axis(-r_rot).into();
            let lhs_rot = -jr_inv * neg_r_rot_mat;
            let mut j_bias = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
            j_bias
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(lhs_rot * factor.delta.j_rotation_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-factor.delta.j_velocity_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(-factor.delta.j_velocity_ba));
            j_bias
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(-factor.delta.j_position_bg));
            j_bias
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-factor.delta.j_position_ba));
            Some(j_bias)
        } else {
            None
        };

        let j_pose_i = whitener * j_pose_i;
        let j_pose_j = whitener * j_pose_j;
        let j_vel_i = whitener * j_vel_i;
        let j_vel_j = whitener * j_vel_j;
        let j_bias_block = j_bias_block.map(|jacobian| whitener * jacobian);
        let r_stack = whitener * r_stack;

        let i_pose = pose_index.get(&factor.keyframe_id_from).copied();
        let j_pose = pose_index.get(&factor.keyframe_id_to).copied();
        let i_vel = velocity_index.get(&factor.keyframe_id_from).copied();
        let j_vel = velocity_index.get(&factor.keyframe_id_to).copied();

        // Helper: accumulate block A^T B into h_pp at (row_block, col_block).
        // Compute the (6 or 3)-row / (6 or 3)-col block and accumulate.
        // (Done inline below per block pair.)

        if let Some(p) = i_pose {
            let blk: Matrix6<f64> = j_pose_i.transpose() * j_pose_i;
            let bblk: Vector6<f64> = j_pose_i.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[p * 6 + rr] += bblk[rr];
            }
        }
        if let Some(p) = j_pose {
            let blk: Matrix6<f64> = j_pose_j.transpose() * j_pose_j;
            let bblk: Vector6<f64> = j_pose_j.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(p * 6 + rr, p * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[p * 6 + rr] += bblk[rr];
            }
        }
        if let (Some(pi), Some(pj)) = (i_pose, j_pose) {
            let blk: Matrix6<f64> = j_pose_i.transpose() * j_pose_j;
            for rr in 0..6 {
                for cc in 0..6 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, pj * 6 + cc)] += v;
                    h_pp[(pj * 6 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let Some(v) = i_vel {
            let blk: Matrix3<f64> = j_vel_i.transpose() * j_vel_i;
            let bblk: Vector3<f64> = j_vel_i.transpose() * r_stack;
            for rr in 0..3 {
                for cc in 0..3 {
                    h_pp[(vel_offset + v * 3 + rr, vel_offset + v * 3 + cc)] += blk[(rr, cc)];
                }
                b_p[vel_offset + v * 3 + rr] += bblk[rr];
            }
        }
        if let Some(v) = j_vel {
            let blk: Matrix3<f64> = j_vel_j.transpose() * j_vel_j;
            let bblk: Vector3<f64> = j_vel_j.transpose() * r_stack;
            for rr in 0..3 {
                for cc in 0..3 {
                    h_pp[(vel_offset + v * 3 + rr, vel_offset + v * 3 + cc)] += blk[(rr, cc)];
                }
                b_p[vel_offset + v * 3 + rr] += bblk[rr];
            }
        }
        if let (Some(vi), Some(vj)) = (i_vel, j_vel) {
            let blk: Matrix3<f64> = j_vel_i.transpose() * j_vel_j;
            for rr in 0..3 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(vel_offset + vi * 3 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, vel_offset + vi * 3 + rr)] += v;
                }
            }
        }
        if let (Some(pi), Some(vi)) = (i_pose, i_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_i.transpose() * j_vel_i;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, vel_offset + vi * 3 + cc)] += v;
                    h_pp[(vel_offset + vi * 3 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pi), Some(vj)) = (i_pose, j_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_i.transpose() * j_vel_j;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pi * 6 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, pi * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pj), Some(vi)) = (j_pose, i_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_j.transpose() * j_vel_i;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pj * 6 + rr, vel_offset + vi * 3 + cc)] += v;
                    h_pp[(vel_offset + vi * 3 + cc, pj * 6 + rr)] += v;
                }
            }
        }
        if let (Some(pj), Some(vj)) = (j_pose, j_vel) {
            let blk: nalgebra::Matrix6x3<f64> = j_pose_j.transpose() * j_vel_j;
            for rr in 0..6 {
                for cc in 0..3 {
                    let v = blk[(rr, cc)];
                    h_pp[(pj * 6 + rr, vel_offset + vj * 3 + cc)] += v;
                    h_pp[(vel_offset + vj * 3 + cc, pj * 6 + rr)] += v;
                }
            }
        }

        // Bias contributions: J_bias_i (9×6) accumulates against itself
        // and against every other side (pose_i, pose_j, vel_i, vel_j).
        if let (Some(b), Some(jb)) = (i_bias, j_bias_block) {
            let blk: Matrix6<f64> = jb.transpose() * jb;
            let bblk: Vector6<f64> = jb.transpose() * r_stack;
            for rr in 0..6 {
                for cc in 0..6 {
                    h_pp[(bias_offset + b * 6 + rr, bias_offset + b * 6 + cc)] += blk[(rr, cc)];
                }
                b_p[bias_offset + b * 6 + rr] += bblk[rr];
            }
            if let Some(p) = i_pose {
                let cross: Matrix6<f64> = j_pose_i.transpose() * jb;
                for rr in 0..6 {
                    for cc in 0..6 {
                        let v = cross[(rr, cc)];
                        h_pp[(p * 6 + rr, bias_offset + b * 6 + cc)] += v;
                        h_pp[(bias_offset + b * 6 + cc, p * 6 + rr)] += v;
                    }
                }
            }
            if let Some(p) = j_pose {
                let cross: Matrix6<f64> = j_pose_j.transpose() * jb;
                for rr in 0..6 {
                    for cc in 0..6 {
                        let v = cross[(rr, cc)];
                        h_pp[(p * 6 + rr, bias_offset + b * 6 + cc)] += v;
                        h_pp[(bias_offset + b * 6 + cc, p * 6 + rr)] += v;
                    }
                }
            }
            if let Some(v) = i_vel {
                let cross: nalgebra::Matrix3x6<f64> = j_vel_i.transpose() * jb;
                for rr in 0..3 {
                    for cc in 0..6 {
                        let val = cross[(rr, cc)];
                        h_pp[(vel_offset + v * 3 + rr, bias_offset + b * 6 + cc)] += val;
                        h_pp[(bias_offset + b * 6 + cc, vel_offset + v * 3 + rr)] += val;
                    }
                }
            }
            if let Some(v) = j_vel {
                let cross: nalgebra::Matrix3x6<f64> = j_vel_j.transpose() * jb;
                for rr in 0..3 {
                    for cc in 0..6 {
                        let val = cross[(rr, cc)];
                        h_pp[(vel_offset + v * 3 + rr, bias_offset + b * 6 + cc)] += val;
                        h_pp[(bias_offset + b * 6 + cc, vel_offset + v * 3 + rr)] += val;
                    }
                }
            }
        }
    }

    // Bias random-walk factors: residual `r = b_j − b_i`, Jacobian
    // `J_i = −I, J_j = I`, with separate gyro/accel weights. The
    // factor only touches the bias slots (no pose / velocity coupling),
    // so `Jᵀ J` lands as `+w · I` on each diagonal bias block, `−w · I`
    // on the off-diagonal cross block, and `Jᵀ r` distributes
    // `−w · r` to `b_i` and `+w · r` to `b_j`.
    for factor in &ba.bias_random_walk_factors {
        let (Some(b_i), Some(b_j)) = (
            ba.biases.get(&factor.keyframe_id_from),
            ba.biases.get(&factor.keyframe_id_to),
        ) else {
            continue;
        };
        let r: Vector6<f64> = b_j - b_i;
        let weights = [
            factor.weight_gyro.max(0.0),
            factor.weight_gyro.max(0.0),
            factor.weight_gyro.max(0.0),
            factor.weight_accel.max(0.0),
            factor.weight_accel.max(0.0),
            factor.weight_accel.max(0.0),
        ];
        if weights.iter().all(|weight| *weight <= 0.0) {
            continue;
        }
        let i_slot = bias_index.get(&factor.keyframe_id_from).copied();
        let j_slot = bias_index.get(&factor.keyframe_id_to).copied();
        if let Some(i) = i_slot {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + i * 6 + k, bias_offset + i * 6 + k)] += w;
                b_p[bias_offset + i * 6 + k] += -w * r[k];
            }
        }
        if let Some(j) = j_slot {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + j * 6 + k, bias_offset + j * 6 + k)] += w;
                b_p[bias_offset + j * 6 + k] += w * r[k];
            }
        }
        if let (Some(i), Some(j)) = (i_slot, j_slot) {
            for k in 0..6 {
                let w = weights[k];
                h_pp[(bias_offset + i * 6 + k, bias_offset + j * 6 + k)] += -w;
                h_pp[(bias_offset + j * 6 + k, bias_offset + i * 6 + k)] += -w;
            }
        }
    }

    // Dense FEJ navigation prior. Its Jacobian is frozen to identity in the
    // stored right-perturbation coordinates; only the residual displacement
    // from the fixed reference changes. Fixed variables still affect the
    // current gradient through H*delta, but receive no solver rows.
    if let Some(prior) = &ba.navigation_state_prior {
        if let Some(delta) = navigation_prior_delta(ba, prior) {
            let current_gradient = &prior.information * delta + &prior.gradient;
            let mut global_indices: Vec<Option<usize>> =
                Vec::with_capacity(prior.keyframe_ids.len() * 15);
            for id in &prior.keyframe_ids {
                let pose_slot = pose_index.get(id).copied();
                for component in 0..6 {
                    global_indices.push(pose_slot.map(|slot| slot * 6 + component));
                }
                let velocity_slot = velocity_index.get(id).copied();
                for component in 0..3 {
                    global_indices
                        .push(velocity_slot.map(|slot| vel_offset + slot * 3 + component));
                }
                let bias_slot = bias_index.get(id).copied();
                for component in 0..6 {
                    global_indices.push(bias_slot.map(|slot| bias_offset + slot * 6 + component));
                }
            }
            for (prior_row, global_row) in global_indices.iter().enumerate() {
                let Some(global_row) = *global_row else {
                    continue;
                };
                b_p[global_row] += current_gradient[prior_row];
                for (prior_col, global_col) in global_indices.iter().enumerate() {
                    let Some(global_col) = *global_col else {
                        continue;
                    };
                    h_pp[(global_row, global_col)] += prior.information[(prior_row, prior_col)];
                }
            }
        }
    }

    NormalEquationsBa {
        h_pp,
        b_p,
        landmarks,
    }
}

/// Project a normal-equation system onto the subspace in which selected pose
/// rotations are fixed.  Pose blocks intentionally remain six-dimensional so
/// the existing Schur and linear-solver layouts are unchanged.  The
/// constrained rotation rows/columns are made identity rows with zero right
/// hand side; this is equivalent to removing those variables and is also
/// well-defined for an undamped Gauss--Newton solve.  Landmark cross blocks
/// are cleared for the same rows so the translation/landmark solve cannot
/// use a discarded rotation update.
fn constrain_fixed_pose_rotations(
    fixed_rotations: &BTreeSet<u64>,
    pose_index: &BTreeMap<u64, usize>,
    system: &mut NormalEquationsBa,
) {
    for &image_id in fixed_rotations {
        let Some(&pose_slot) = pose_index.get(&image_id) else {
            continue;
        };
        for component in 3..6 {
            let index = pose_slot * 6 + component;
            for column in 0..system.h_pp.ncols() {
                system.h_pp[(index, column)] = 0.0;
            }
            for row in 0..system.h_pp.nrows() {
                system.h_pp[(row, index)] = 0.0;
            }
            system.h_pp[(index, index)] = 1.0;
            system.b_p[index] = 0.0;
            for landmark in &mut system.landmarks {
                for (landmark_pose_slot, cross) in &mut landmark.cross {
                    if *landmark_pose_slot == pose_slot {
                        for column in 0..3 {
                            cross[(component, column)] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_step(
    system: &NormalEquationsBa,
    p_count: usize,
    l_count: usize,
    v_count: usize,
    b_count: usize,
    lambda: f64,
    linear_solver: LinearSolver,
    parallel: bool,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    // Landmark-only BA: H_LL is block-diagonal so each landmark gets an
    // independent 3×3 solve. No Schur complement needed. Every landmark's
    // solve is independent and writes only its own 3 rows of `delta_l`, so
    // the parallel path is a direct embarrassingly-parallel dispatch with no
    // merge step: bit-identical to the serial loop below at any thread
    // count.
    if p_count == 0 && v_count == 0 && b_count == 0 {
        let mut delta_l = DVector::<f64>::zeros(l_count * 3);
        if parallel && l_count >= PARALLEL_MIN_LANDMARKS {
            use rayon::prelude::*;
            let solved: Result<Vec<Vector3<f64>>, BaError> = system
                .landmarks
                .par_iter()
                .map(|landmark| {
                    let mut h_ll = landmark.h_ll;
                    if lambda > 0.0 {
                        h_ll[(0, 0)] += lambda;
                        h_ll[(1, 1)] += lambda;
                        h_ll[(2, 2)] += lambda;
                    }
                    let h_ll_inv = h_ll.try_inverse().ok_or(BaError::SingularSystem)?;
                    Ok(-(h_ll_inv * landmark.b_l))
                })
                .collect();
            for (l, dl) in solved?.into_iter().enumerate() {
                for k in 0..3 {
                    delta_l[l * 3 + k] = dl[k];
                }
            }
        } else {
            for (l, landmark) in system.landmarks.iter().enumerate() {
                let mut h_ll = landmark.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let h_ll_inv = h_ll.try_inverse().ok_or(BaError::SingularSystem)?;
                let dl: Vector3<f64> = -(h_ll_inv * landmark.b_l);
                for k in 0..3 {
                    delta_l[l * 3 + k] = dl[k];
                }
            }
        }
        return Ok((DVector::<f64>::zeros(0), delta_l));
    }

    // λ damping on the joint pose+velocity+bias diagonal. The first 6P
    // rows are pose perturbations; the next 3V are velocity
    // perturbations; the final 6B are bias perturbations.
    let pose_dim = p_count * 6;
    let total_dim = pose_dim + v_count * 3 + b_count * 6;
    let mut s = system.h_pp.clone();
    if lambda > 0.0 {
        for k in 0..total_dim {
            s[(k, k)] += lambda;
        }
    }
    let mut b_reduced = -&system.b_p;

    // Per-landmark Schur reduction. Each landmark contributes:
    //   S -= H_PL_l · H_LL_l^{-1} · H_PL_l^T
    //   b_S = -b_P + H_PL_l · H_LL_l^{-1} · b_l
    // (the RHS starts at -b_P, so each landmark contributes with a plus
    // sign). Both updates only touch the rows/cols of S corresponding to poses
    // that observed this landmark, so we never materialize the full H_PL.
    // Flag-gated, work-gated parallel reduction (see the module's
    // "Parallelism" section): bit-identical to the plain loop below at any
    // thread count / chunk size.
    let (h_ll_inv_cache, b_l_cache): (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) =
        if parallel && system.landmarks.len() >= PARALLEL_MIN_LANDMARKS {
            schur_reduce_parallel(system, lambda, &mut s, &mut b_reduced)
        } else {
            let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> =
                Vec::with_capacity(system.landmarks.len());
            let mut b_l_cache: Vec<Vector3<f64>> = Vec::with_capacity(system.landmarks.len());
            for landmark in &system.landmarks {
                let mut h_ll = landmark.h_ll;
                if lambda > 0.0 {
                    h_ll[(0, 0)] += lambda;
                    h_ll[(1, 1)] += lambda;
                    h_ll[(2, 2)] += lambda;
                }
                let h_ll_inv = h_ll.try_inverse();
                h_ll_inv_cache.push(h_ll_inv);
                b_l_cache.push(landmark.b_l);
                let h_ll_inv = match h_ll_inv {
                    Some(h) => h,
                    None => continue,
                };
                // Update S:
                //   for (p, A) in cross, for (q, B) in cross:
                //     S[p, q] -= A · h_ll_inv · B^T
                for (p, a) in &landmark.cross {
                    // Precompute A · h_ll_inv (6×3) once per outer pose.
                    let a_h: Matrix6x3<f64> = a * h_ll_inv;
                    for (q, b) in &landmark.cross {
                        let block: Matrix6<f64> = a_h * b.transpose();
                        for r in 0..6 {
                            for c in 0..6 {
                                s[(p * 6 + r, q * 6 + c)] -= block[(r, c)];
                            }
                        }
                    }
                    // Update reduced rhs. Schur derivation:
                    //   S · δ_p = -g_p + H_PL · H_LL^{-1} · g_l,
                    // so each landmark `l` contributes `+ A · h_ll_inv · b_l`
                    // to `b_reduced` (which already starts at `-g_p = -b_p`).
                    let upd: Vector6<f64> = a_h * landmark.b_l;
                    for k in 0..6 {
                        b_reduced[p * 6 + k] += upd[k];
                    }
                }
            }
            (h_ll_inv_cache, b_l_cache)
        };

    // Solve the reduced pose system. The dense path uses Cholesky/LU; the
    // sparse path goes through CscCholesky on S as a CSC matrix.
    let delta_p = match linear_solver {
        LinearSolver::Dense => match solve_normal_equations(&s, &b_reduced) {
            Ok(d) => d,
            Err(PoseGraphError::SingularSystem) => return Err(BaError::SingularSystem),
            Err(_) => return Err(BaError::SingularSystem),
        },
        LinearSolver::Sparse => {
            let dim = total_dim;
            // Collect the structural nonzeros of the reduced system once. Both
            // back-ends consume the same triplet list; its sparsity pattern is
            // dictated by which pose pairs share a landmark observation.
            let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
            for c in 0..dim {
                for r in 0..dim {
                    let v = s[(r, c)];
                    if v != 0.0 {
                        triplets.push((r, c, v));
                    }
                }
            }
            let rhs = DMatrix::from_column_slice(dim, 1, b_reduced.as_slice());

            // Pose and IMU-bias variables are 6×6 diagonal blocks, so when the
            // system carries no 3-DOF velocity blocks (the common pure-visual
            // BA case) the reduced matrix tiles cleanly into 6×6 blocks and the
            // block Cholesky back-end — the same one the pose graph uses —
            // factors it without the scalar gather/scatter bookkeeping. λ is
            // already folded into `s`, so we pass `lambda = 0`. Visual-inertial
            // systems interleave 3-DOF velocities, breaking the uniform tiling,
            // so they fall back to the scalar `CscCholesky` factorization.
            let sol = if v_count == 0 {
                crate::block_cholesky::solve_spd_block(&triplets, dim, 6, &rhs, 0.0)
                    .map_err(|_| BaError::SingularSystem)?
            } else {
                use nalgebra_sparse::{factorization::CscCholesky, CooMatrix, CscMatrix};
                let mut coo = CooMatrix::<f64>::new(dim, dim);
                for &(r, c, v) in &triplets {
                    coo.push(r, c, v);
                }
                let csc = CscMatrix::from(&coo);
                let chol = CscCholesky::factor(&csc).map_err(|_| BaError::SingularSystem)?;
                chol.solve(&rhs)
            };
            DVector::from_column_slice(sol.as_slice())
        }
    };

    // Back-substitute landmark updates:
    //   δ_L[l] = h_ll_inv · (-b_l - Σ_p H_pl^T · δ_p)
    // Each landmark writes only its own 3 rows of `delta_l` and only reads
    // the (already solved, read-only from here on) `delta_p` and its own
    // cache entries, so — like the landmark-only branch above — the
    // parallel path is a direct embarrassingly-parallel dispatch with no
    // merge step: bit-identical to the serial loop below at any thread
    // count.
    let mut delta_l = DVector::<f64>::zeros(system.landmarks.len() * 3);
    if parallel && system.landmarks.len() >= PARALLEL_MIN_LANDMARKS {
        use rayon::prelude::*;
        delta_l
            .as_mut_slice()
            .par_chunks_mut(3)
            .zip(system.landmarks.par_iter())
            .zip(h_ll_inv_cache.par_iter())
            .zip(b_l_cache.par_iter())
            .for_each(|(((out, landmark), h_ll_inv), b_l)| {
                let Some(h_ll_inv) = h_ll_inv else {
                    return;
                };
                let mut acc = -*b_l;
                for (p, a) in &landmark.cross {
                    let dp = delta_p.fixed_rows::<6>(p * 6);
                    let dp_vec: Vector6<f64> = dp.into_owned();
                    let sub: Vector3<f64> = a.transpose() * dp_vec;
                    acc -= sub;
                }
                let dl = *h_ll_inv * acc;
                out.copy_from_slice(dl.as_slice());
            });
    } else {
        for (l, landmark) in system.landmarks.iter().enumerate() {
            let h_ll_inv = match h_ll_inv_cache[l] {
                Some(h) => h,
                None => continue,
            };
            let mut acc = -b_l_cache[l];
            for (p, a) in &landmark.cross {
                let dp = delta_p.fixed_rows::<6>(p * 6);
                let dp_vec: Vector6<f64> = dp.into_owned();
                let sub: Vector3<f64> = a.transpose() * dp_vec;
                acc -= sub;
            }
            let dl = h_ll_inv * acc;
            for k in 0..3 {
                delta_l[l * 3 + k] = dl[k];
            }
        }
    }

    Ok((delta_p, delta_l))
}

/// Parallel counterpart of the per-landmark Schur reduction in
/// [`solve_step`] (the `p_count > 0` branch's main loop). Fills
/// `h_ll_inv_cache` / `b_l_cache` directly in parallel — each landmark's
/// `3×3` factorization only touches its own output slot, no merge needed —
/// then updates `s` / `b_reduced` from fixed-size landmark chunks
/// ([`PARALLEL_LANDMARK_CHUNK`]) the same way
/// [`assemble_mono_observations_parallel`] updates the assembly
/// accumulators: within a chunk, every landmark's pose-pair contributions
/// (its `S[p, q] -= …` blocks and its `b_reduced[p] += …` update) are
/// computed concurrently — a pure function of that landmark's cached
/// inverse, `b_l`, and cross blocks, collected into a `(Vec<(p, q, block)>,
/// Vec<(p, upd)>)` pair per landmark — and then folded into `s` /
/// `b_reduced` by a single serial pass, landmark by landmark in ascending
/// index order, before the next chunk starts. `s` and `b_reduced` are
/// disjoint arrays, so grouping a landmark's `S` updates before its
/// `b_reduced` update (rather than interleaving them per pose as the serial
/// loop does) does not change either accumulator's own summation order —
/// only operations that target the *same* memory location are
/// order-sensitive for floating point, and each one's order here is exactly
/// the serial loop's. The result is therefore bit-identical to the serial
/// path at any thread count or chunk size.
fn schur_reduce_parallel(
    system: &NormalEquationsBa,
    lambda: f64,
    s: &mut DMatrix<f64>,
    b_reduced: &mut DVector<f64>,
) -> (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) {
    use rayon::prelude::*;

    let (h_ll_inv_cache, b_l_cache): (Vec<Option<Matrix3<f64>>>, Vec<Vector3<f64>>) = system
        .landmarks
        .par_iter()
        .map(|landmark| {
            let mut h_ll = landmark.h_ll;
            if lambda > 0.0 {
                h_ll[(0, 0)] += lambda;
                h_ll[(1, 1)] += lambda;
                h_ll[(2, 2)] += lambda;
            }
            (h_ll.try_inverse(), landmark.b_l)
        })
        .unzip();

    let mut start = 0;
    while start < system.landmarks.len() {
        let end = (start + PARALLEL_LANDMARK_CHUNK).min(system.landmarks.len());

        #[allow(clippy::type_complexity)]
        let (s_updates, b_updates): (
            Vec<Vec<(usize, usize, Matrix6<f64>)>>,
            Vec<Vec<(usize, Vector6<f64>)>>,
        ) = system.landmarks[start..end]
            .par_iter()
            .zip(h_ll_inv_cache[start..end].par_iter())
            .map(|(landmark, h_ll_inv)| {
                let mut s_acc = Vec::new();
                let mut b_acc = Vec::new();
                let Some(h_ll_inv) = h_ll_inv else {
                    return (s_acc, b_acc);
                };
                for (p, a) in &landmark.cross {
                    let a_h: Matrix6x3<f64> = a * h_ll_inv;
                    for (q, b) in &landmark.cross {
                        let block: Matrix6<f64> = a_h * b.transpose();
                        s_acc.push((*p, *q, block));
                    }
                    let upd: Vector6<f64> = a_h * landmark.b_l;
                    b_acc.push((*p, upd));
                }
                (s_acc, b_acc)
            })
            .unzip();
        let s_updates: Vec<(usize, usize, Matrix6<f64>)> =
            s_updates.into_iter().flatten().collect();
        let b_updates: Vec<(usize, Vector6<f64>)> = b_updates.into_iter().flatten().collect();

        for (p, q, block) in s_updates {
            for r in 0..6 {
                for c in 0..6 {
                    s[(p * 6 + r, q * 6 + c)] -= block[(r, c)];
                }
            }
        }
        for (p, upd) in b_updates {
            for k in 0..6 {
                b_reduced[p * 6 + k] += upd[k];
            }
        }

        start = end;
    }

    (h_ll_inv_cache, b_l_cache)
}

fn project_pinhole(intrinsics: &(f64, f64, f64, f64), xc: &Point3<f64>) -> Option<Point2<f64>> {
    if xc.z <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = *intrinsics;
    Some(Point2::new(fx * xc.x / xc.z + cx, fy * xc.y / xc.z + cy))
}

/// Maximum absolute and relative (symmetric, Frobenius-normalized) error
/// between two small Jacobians.  This is intentionally a diagnostic helper,
/// not part of the optimizer: it reports both an absolute error (important
/// when a derivative is close to zero) and a scale-free error (important when
/// comparing translation, rotation, point, and intrinsics columns).
fn jacobian_error<I>(pairs: I) -> (f64, f64)
where
    I: IntoIterator<Item = (f64, f64)>,
{
    let mut max_abs = 0.0_f64;
    let mut diff_squared = 0.0_f64;
    let mut scale_squared = 0.0_f64;
    for (analytic, numerical) in pairs {
        if !analytic.is_finite() || !numerical.is_finite() {
            return (f64::INFINITY, f64::INFINITY);
        }
        let diff = analytic - numerical;
        max_abs = max_abs.max(diff.abs());
        diff_squared += diff * diff;
        let scale = analytic.abs().max(numerical.abs());
        scale_squared += scale * scale;
    }
    let scale = scale_squared.sqrt().max(1.0e-15);
    (max_abs, diff_squared.sqrt() / scale)
}

/// Finite-difference audit of the visual pinhole residual Jacobians for one
/// observation.  The production assembly has two deliberately duplicated
/// fast paths (serial and rayon), so this helper mirrors their formulas while
/// evaluating the residual through the public `Camera::project` API.  It is
/// only called by the explicit `VISLOC_SFM_DEBUG_BA_JACOBIANS` diagnostic and
/// by focused unit tests; it never participates in normal BA.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BaVisualJacobianCase {
    pub residual_norm: f64,
    pub depth: f64,
    pub pose_max_abs: f64,
    pub pose_relative: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative: f64,
}

/// Compare the analytic right-pose, world-landmark, and pinhole-intrinsics
/// Jacobians to central differences at one state.  Intrinsics are reported for
/// the four-parameter pinhole/OpenCV layout; radial distortion is deliberately
/// rejected because the ordinary BA path uses a separate distortion-aware
/// Jacobian in its joint-intrinsics solver.
pub(crate) fn audit_visual_jacobian_case(
    camera: &Camera,
    pose: &Pose,
    point: &Point3<f64>,
    measured: &Point2<f64>,
    epsilon: f64,
) -> Option<BaVisualJacobianCase> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return None;
    }
    if !matches!(camera.model, CameraModel::Pinhole | CameraModel::OpenCv)
        || camera.params.len() < 4
        || camera
            .radial_distortion()
            .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0)
    {
        return None;
    }
    let intrinsics = camera.intrinsics()?;
    let point_camera = pose.transform_world_point(point);
    let predicted = camera.project(&point_camera)?;
    let residual = predicted - *measured;
    let j_projection = pinhole_projection_jacobian(&intrinsics, &point_camera)?;
    let rotation = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut dpoint_dpose = Matrix3x6::<f64>::zeros();
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation);
    dpoint_dpose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation * skew(&point.coords)));
    let analytic_pose = j_projection * dpoint_dpose;
    let analytic_landmark = j_projection * rotation;
    let x = point_camera.x / point_camera.z;
    let y = point_camera.y / point_camera.z;
    let analytic_intrinsics = Matrix2x4::new(x, 0.0, 1.0, 0.0, 0.0, y, 0.0, 1.0);

    let mut numerical_pose = Matrix2x6::<f64>::zeros();
    for axis in 0..6 {
        let mut plus = pose.clone();
        let mut minus = pose.clone();
        let mut delta = Vector6::<f64>::zeros();
        delta[axis] = epsilon;
        plus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        delta[axis] = -epsilon;
        minus.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&delta));
        let plus = camera.project(&plus.transform_world_point(point))?;
        let minus = camera.project(&minus.transform_world_point(point))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_pose[(0, axis)] = derivative.x;
        numerical_pose[(1, axis)] = derivative.y;
    }

    let mut numerical_landmark = Matrix2x3::<f64>::zeros();
    for axis in 0..3 {
        let mut plus = point.coords;
        let mut minus = point.coords;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        let plus = camera.project(&pose.transform_world_point(&Point3::from(plus)))?;
        let minus = camera.project(&pose.transform_world_point(&Point3::from(minus)))?;
        let derivative = (plus - minus) / (2.0 * epsilon);
        numerical_landmark[(0, axis)] = derivative.x;
        numerical_landmark[(1, axis)] = derivative.y;
    }

    let mut numerical_intrinsics = Matrix2x4::<f64>::zeros();
    for axis in 0..4 {
        let parameter_epsilon = epsilon * camera.params[axis].abs().max(1.0);
        let mut plus = camera.clone();
        let mut minus = camera.clone();
        plus.params[axis] += parameter_epsilon;
        minus.params[axis] -= parameter_epsilon;
        let plus = plus.project(&point_camera)?;
        let minus = minus.project(&point_camera)?;
        let derivative = (plus - minus) / (2.0 * parameter_epsilon);
        numerical_intrinsics[(0, axis)] = derivative.x;
        numerical_intrinsics[(1, axis)] = derivative.y;
    }

    let (pose_max_abs, pose_relative) = jacobian_error(
        analytic_pose
            .iter()
            .copied()
            .zip(numerical_pose.iter().copied()),
    );
    let pose_error = |columns: std::ops::Range<usize>| {
        jacobian_error((0..2).flat_map(|row| {
            columns
                .clone()
                .map(move |column| (analytic_pose[(row, column)], numerical_pose[(row, column)]))
        }))
    };
    let (pose_translation_max_abs, pose_translation_relative) = pose_error(0..3);
    let (pose_rotation_max_abs, pose_rotation_relative) = pose_error(3..6);
    let (landmark_max_abs, landmark_relative) = jacobian_error(
        analytic_landmark
            .iter()
            .copied()
            .zip(numerical_landmark.iter().copied()),
    );
    let (intrinsics_max_abs, intrinsics_relative) = jacobian_error(
        analytic_intrinsics
            .iter()
            .copied()
            .zip(numerical_intrinsics.iter().copied()),
    );
    Some(BaVisualJacobianCase {
        residual_norm: residual.norm(),
        depth: point_camera.z,
        pose_max_abs,
        pose_relative,
        pose_translation_max_abs,
        pose_translation_relative,
        pose_rotation_max_abs,
        pose_rotation_relative,
        landmark_max_abs,
        landmark_relative,
        intrinsics_max_abs,
        intrinsics_relative,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianBucket {
    pub samples: usize,
    pub pose_max_abs: f64,
    pub pose_relative_max: f64,
    pub pose_translation_max_abs: f64,
    pub pose_translation_relative_max: f64,
    pub pose_rotation_max_abs: f64,
    pub pose_rotation_relative_max: f64,
    pub landmark_max_abs: f64,
    pub landmark_relative_max: f64,
    pub intrinsics_max_abs: f64,
    pub intrinsics_relative_max: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct BaVisualJacobianAudit {
    pub observations_seen: usize,
    pub samples_audited: usize,
    pub invalid_samples: usize,
    pub normal: BaVisualJacobianBucket,
    pub far_depth: BaVisualJacobianBucket,
    pub low_parallax: BaVisualJacobianBucket,
    pub high_residual: BaVisualJacobianBucket,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaJacobianBucketKind {
    Normal,
    FarDepth,
    LowParallax,
    HighResidual,
}

fn update_jacobian_bucket(bucket: &mut BaVisualJacobianBucket, case: &BaVisualJacobianCase) {
    bucket.samples += 1;
    bucket.pose_max_abs = bucket.pose_max_abs.max(case.pose_max_abs);
    bucket.pose_relative_max = bucket.pose_relative_max.max(case.pose_relative);
    bucket.pose_translation_max_abs = bucket
        .pose_translation_max_abs
        .max(case.pose_translation_max_abs);
    bucket.pose_translation_relative_max = bucket
        .pose_translation_relative_max
        .max(case.pose_translation_relative);
    bucket.pose_rotation_max_abs = bucket.pose_rotation_max_abs.max(case.pose_rotation_max_abs);
    bucket.pose_rotation_relative_max = bucket
        .pose_rotation_relative_max
        .max(case.pose_rotation_relative);
    bucket.landmark_max_abs = bucket.landmark_max_abs.max(case.landmark_max_abs);
    bucket.landmark_relative_max = bucket.landmark_relative_max.max(case.landmark_relative);
    bucket.intrinsics_max_abs = bucket.intrinsics_max_abs.max(case.intrinsics_max_abs);
    bucket.intrinsics_relative_max = bucket.intrinsics_relative_max.max(case.intrinsics_relative);
}

fn observation_jacobian_bucket(
    ba: &BundleAdjustment,
    obs_idx: usize,
    point_camera: &Point3<f64>,
    residual_norm: f64,
) -> BaJacobianBucketKind {
    // A large residual is the most useful disjoint bucket for checking the
    // robust-weighting path.  For geometric conditioning, estimate the widest
    // ray angle to another observation of the same landmark.  This is a
    // deterministic, diagnostic-only proxy; no optimizer decision uses it.
    if residual_norm > 10.0 {
        return BaJacobianBucketKind::HighResidual;
    }
    let observation = &ba.observations[obs_idx];
    let Some(anchor_pose) = ba.poses.get(&observation.keyframe_id) else {
        return BaJacobianBucketKind::Normal;
    };
    let point = &ba.landmarks[&observation.landmark_id];
    let anchor_ray = point.coords - anchor_pose.camera_center_world().coords;
    let mut max_angle = None;
    for (other_idx, other) in ba.observations.iter().enumerate() {
        if other_idx == obs_idx || other.landmark_id != observation.landmark_id {
            continue;
        }
        let Some(other_pose) = ba.poses.get(&other.keyframe_id) else {
            continue;
        };
        let other_ray = point.coords - other_pose.camera_center_world().coords;
        let (Some(a), Some(b)) = (
            anchor_ray.try_normalize(1.0e-15),
            other_ray.try_normalize(1.0e-15),
        ) else {
            continue;
        };
        let cosine = a.dot(&b).clamp(-1.0, 1.0);
        let angle = cosine.acos();
        if angle.is_finite() {
            max_angle = Some(max_angle.map_or(angle, |current: f64| current.max(angle)));
        }
    }
    if max_angle.is_some_and(|angle| angle.to_degrees() < 1.0) {
        BaJacobianBucketKind::LowParallax
    } else if point_camera.z > 100.0 {
        BaJacobianBucketKind::FarDepth
    } else {
        BaJacobianBucketKind::Normal
    }
}

/// Audit a deterministic, small sample from a live BA state.  The function is
/// intentionally `pub(crate)` so the incremental SFM diagnostic can invoke it
/// without exposing a new public solver API.  It returns no result used by the
/// optimizer and is never called unless the explicit debug environment flag is
/// enabled by the caller.
pub(crate) fn audit_bundle_visual_jacobians(
    ba: &BundleAdjustment,
    max_samples: usize,
) -> BaVisualJacobianAudit {
    let mut report = BaVisualJacobianAudit {
        observations_seen: ba.observations.len(),
        ..BaVisualJacobianAudit::default()
    };
    if max_samples == 0 {
        return report;
    }

    // Reserve an equal deterministic quota for each conditioning bucket, so a
    // long track ordered entirely by one region cannot hide the other cases.
    let quota = max_samples.div_ceil(4);
    let mut candidates: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    for (obs_idx, observation) in ba.observations.iter().enumerate() {
        let (Some(pose), Some(point)) = (
            ba.poses.get(&observation.keyframe_id),
            ba.landmarks.get(&observation.landmark_id),
        ) else {
            continue;
        };
        let point_camera = pose.transform_world_point(point);
        let Some(predicted) = ba.camera.project(&point_camera) else {
            continue;
        };
        let residual_norm = (predicted - observation.xy).norm();
        let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, residual_norm);
        let slot = match kind {
            BaJacobianBucketKind::Normal => 0,
            BaJacobianBucketKind::FarDepth => 1,
            BaJacobianBucketKind::LowParallax => 2,
            BaJacobianBucketKind::HighResidual => 3,
        };
        if candidates[slot].len() < quota {
            candidates[slot].push(obs_idx);
        }
    }

    for indices in candidates {
        for obs_idx in indices {
            if report.samples_audited >= max_samples {
                break;
            }
            let observation = &ba.observations[obs_idx];
            let (Some(pose), Some(point)) = (
                ba.poses.get(&observation.keyframe_id),
                ba.landmarks.get(&observation.landmark_id),
            ) else {
                report.invalid_samples += 1;
                continue;
            };
            let Some(case) =
                audit_visual_jacobian_case(&ba.camera, pose, point, &observation.xy, 1.0e-6)
            else {
                report.invalid_samples += 1;
                continue;
            };
            let point_camera = pose.transform_world_point(point);
            let kind = observation_jacobian_bucket(ba, obs_idx, &point_camera, case.residual_norm);
            match kind {
                BaJacobianBucketKind::Normal => update_jacobian_bucket(&mut report.normal, &case),
                BaJacobianBucketKind::FarDepth => {
                    update_jacobian_bucket(&mut report.far_depth, &case)
                }
                BaJacobianBucketKind::LowParallax => {
                    update_jacobian_bucket(&mut report.low_parallax, &case)
                }
                BaJacobianBucketKind::HighResidual => {
                    update_jacobian_bucket(&mut report.high_residual, &case)
                }
            }
            report.samples_audited += 1;
        }
    }
    report
}

fn pinhole_projection_jacobian(
    intrinsics: &(f64, f64, f64, f64),
    point: &Point3<f64>,
) -> Option<Matrix2x3<f64>> {
    if point.z <= 0.0 {
        return None;
    }
    let (fx, fy, _, _) = *intrinsics;
    let z_inv = point.z.recip();
    let z_inv2 = z_inv * z_inv;
    Some(Matrix2x3::new(
        fx * z_inv,
        0.0,
        -fx * point.x * z_inv2,
        0.0,
        fy * z_inv,
        -fy * point.y * z_inv2,
    ))
}

fn general_stereo_residual_jacobians(
    left_intrinsics: &(f64, f64, f64, f64),
    observation: &BaGeneralStereoObservation,
    pose: &Pose,
    point_world: &Point3<f64>,
) -> Option<(Vector4<f64>, Matrix4x6<f64>, Matrix4x3<f64>)> {
    let right_intrinsics = observation.right_camera.intrinsics()?;
    let point_left = pose.transform_world_point(point_world);
    let point_right = observation.left_to_right.transform_point(&point_left);
    let predicted_left = project_pinhole(left_intrinsics, &point_left)?;
    let predicted_right = project_pinhole(&right_intrinsics, &point_right)?;
    let residual = Vector4::new(
        predicted_left.x - observation.xy_left.x,
        predicted_left.y - observation.xy_left.y,
        predicted_right.x - observation.xy_right.x,
        predicted_right.y - observation.xy_right.y,
    );

    let j_left_projection = pinhole_projection_jacobian(left_intrinsics, &point_left)?;
    let j_right_projection = pinhole_projection_jacobian(&right_intrinsics, &point_right)?;
    let rotation_world_to_left = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let rotation_left_to_right = observation
        .left_to_right
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let mut d_left_d_pose = Matrix3x6::<f64>::zeros();
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&rotation_world_to_left);
    d_left_d_pose
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-rotation_world_to_left * skew(&point_world.coords)));

    let j_left_pose = j_left_projection * d_left_d_pose;
    let j_right_pose = j_right_projection * rotation_left_to_right * d_left_d_pose;
    let j_left_landmark = j_left_projection * rotation_world_to_left;
    let j_right_landmark = j_right_projection * rotation_left_to_right * rotation_world_to_left;
    let mut j_pose = Matrix4x6::<f64>::zeros();
    j_pose.fixed_rows_mut::<2>(0).copy_from(&j_left_pose);
    j_pose.fixed_rows_mut::<2>(2).copy_from(&j_right_pose);
    let mut j_landmark = Matrix4x3::<f64>::zeros();
    j_landmark
        .fixed_rows_mut::<2>(0)
        .copy_from(&j_left_landmark);
    j_landmark
        .fixed_rows_mut::<2>(2)
        .copy_from(&j_right_landmark);
    Some((residual, j_pose, j_landmark))
}

fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Right-Jacobian inverse on SO(3): Jr⁻¹(φ) = I + ½[φ]× + c·[φ]×²,
/// with `c = (1/θ²) · (1 − (θ/2)·cot(θ/2))` for `θ = ‖φ‖`. At `θ → 0`
/// falls back to the leading Taylor expansion `I + ½[φ]× + (1/12)·[φ]×²`.
/// Used by [`build_normal_equations`] to linearise the SO(3) log
/// residual of the Forster IMU factor.
///
/// Identity: `Jr_inv(φ) = Jl_inv(−φ)`, which means relative to the
/// `so3_left_jacobian_inverse` formula in [`visloc_core::geometry`]
/// only the sign of the linear `[φ]×` term flips (the quadratic
/// `[φ]×²` coefficient stays the same).
fn right_jacobian_inverse_so3(phi: &Vector3<f64>) -> Matrix3<f64> {
    let theta_sq = phi.norm_squared();
    let phi_skew = skew(phi);
    if theta_sq < 1e-10 {
        Matrix3::identity() + 0.5 * phi_skew + (1.0 / 12.0) * phi_skew * phi_skew
    } else {
        let theta = theta_sq.sqrt();
        let half_theta = 0.5 * theta;
        let c = (1.0 - half_theta * half_theta.cos() / half_theta.sin()) / theta_sq;
        Matrix3::identity() + 0.5 * phi_skew + c * phi_skew * phi_skew
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
    pub fn new(config: BaConfig) -> Self {
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

#[cfg(test)]
mod visual_jacobian_audit_tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    fn audit_camera() -> Camera {
        Camera::pinhole(1, 1600, 1066, 879.4, 879.4, 803.4, 532.6)
    }

    fn audit_pose() -> Pose {
        Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.08, -0.11, 0.17),
            Vector3::new(0.24, -0.13, 0.31),
        )
    }

    fn assert_case_is_accurate(label: &str, case: BaVisualJacobianCase) {
        eprintln!(
            "ba-jacobian-test: case={label} residual={:.3e} depth={:.3e} pose=(abs {:.3e},rel {:.3e}; trans {:.3e}/{:.3e}; rot {:.3e}/{:.3e}) landmark=(abs {:.3e},rel {:.3e}) intrinsics=(abs {:.3e},rel {:.3e})",
            case.residual_norm,
            case.depth,
            case.pose_max_abs,
            case.pose_relative,
            case.pose_translation_max_abs,
            case.pose_translation_relative,
            case.pose_rotation_max_abs,
            case.pose_rotation_relative,
            case.landmark_max_abs,
            case.landmark_relative,
            case.intrinsics_max_abs,
            case.intrinsics_relative,
        );
        assert!(
            case.pose_max_abs < 1.0e-5 && case.pose_relative < 1.0e-6,
            "{label} pose Jacobian mismatch: {:?}",
            case
        );
        assert!(
            case.landmark_max_abs < 1.0e-5 && case.landmark_relative < 1.0e-6,
            "{label} landmark Jacobian mismatch: {:?}",
            case
        );
        assert!(
            case.intrinsics_max_abs < 1.0e-6 && case.intrinsics_relative < 1.0e-8,
            "{label} intrinsics Jacobian mismatch: {:?}",
            case
        );
    }

    #[test]
    fn analytic_visual_jacobians_match_finite_differences_across_regimes() {
        let camera = audit_camera();
        let pose = audit_pose();
        let normal_point = Point3::new(0.45, -0.35, 4.8);
        let normal_measurement = camera
            .project(&pose.transform_world_point(&normal_point))
            .unwrap();
        assert_case_is_accurate(
            "normal",
            audit_visual_jacobian_case(&camera, &pose, &normal_point, &normal_measurement, 1.0e-6)
                .unwrap(),
        );

        let far_point = Point3::new(15.0, -8.0, 10_000.0);
        let far_measurement = camera
            .project(&pose.transform_world_point(&far_point))
            .unwrap();
        assert_case_is_accurate(
            "far-depth",
            audit_visual_jacobian_case(&camera, &pose, &far_point, &far_measurement, 1.0e-6)
                .unwrap(),
        );

        // A very small camera baseline relative to depth is the low-parallax
        // regime that made the captured 27-camera point block ill-conditioned.
        // The per-observation Jacobian itself remains well-defined, so this
        // case checks that no special-case branch changes its numerical value.
        let low_parallax_point = Point3::new(-0.15, 0.12, 100.0);
        let low_parallax_measurement = camera
            .project(&pose.transform_world_point(&low_parallax_point))
            .unwrap();
        assert_case_is_accurate(
            "low-parallax",
            audit_visual_jacobian_case(
                &camera,
                &pose,
                &low_parallax_point,
                &low_parallax_measurement,
                1.0e-6,
            )
            .unwrap(),
        );

        let high_residual_measurement = normal_measurement + Vector2::new(80.0, -55.0);
        assert_case_is_accurate(
            "high-residual",
            audit_visual_jacobian_case(
                &camera,
                &pose,
                &normal_point,
                &high_residual_measurement,
                1.0e-6,
            )
            .unwrap(),
        );
    }

    #[test]
    fn bundle_audit_reports_low_parallax_and_high_residual_buckets() {
        let camera = audit_camera();
        let pose0 = Pose::identity();
        let pose1 =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.01, 0.0, 0.0));
        let point = Point3::new(0.2, -0.1, 100.0);
        let mut ba = BundleAdjustment::new(camera.clone());
        ba.add_pose(0, pose0.clone());
        ba.add_pose(1, pose1);
        ba.add_landmark(0, point);
        let exact0 = camera
            .project(&pose0.transform_world_point(&point))
            .unwrap();
        let exact1 = camera
            .project(&ba.poses[&1].transform_world_point(&point))
            .unwrap();
        ba.add_observation(BaObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy: exact0,
        });
        ba.add_observation(BaObservation {
            keyframe_id: 1,
            landmark_id: 0,
            xy: exact1 + Vector2::new(25.0, 0.0),
        });
        let report = audit_bundle_visual_jacobians(&ba, 16);
        assert_eq!(report.observations_seen, 2);
        assert_eq!(report.samples_audited, 2);
        assert_eq!(report.low_parallax.samples, 1);
        assert_eq!(report.high_residual.samples, 1);
        assert_eq!(report.invalid_samples, 0);
    }

    #[test]
    fn huber_weight_is_the_derivative_of_the_squared_residual_cost() {
        let kernel = RobustKernel::Huber { delta: 3.0 };
        for squared_residual in [1.0, 4.0, 16.0, 100.0] {
            let epsilon = 1.0e-6 * squared_residual;
            let numerical = (kernel.cost(squared_residual + epsilon)
                - kernel.cost(squared_residual - epsilon))
                / (2.0 * epsilon);
            let analytic = kernel.weight(squared_residual);
            assert!(
                (analytic - numerical).abs() < 1.0e-8,
                "s={squared_residual}: rho'={analytic}, finite difference={numerical}"
            );
        }
    }

    #[test]
    fn schur_rhs_and_back_substitution_match_the_full_normal_system() {
        let mut h_pp = DMatrix::<f64>::zeros(6, 6);
        for i in 0..6 {
            h_pp[(i, i)] = 10.0 + i as f64;
        }
        let mut h_ll = Matrix3::<f64>::zeros();
        h_ll[(0, 0)] = 4.0;
        h_ll[(1, 1)] = 5.0;
        h_ll[(2, 2)] = 6.0;
        let cross = Matrix6x3::<f64>::from_fn(|r, c| 0.03 * (r as f64 + 1.0) * (c as f64 + 2.0));
        let b_p = DVector::from_iterator(6, (0..6).map(|i| 0.2 * (i as f64 + 1.0)));
        let b_l = Vector3::new(-0.4, 0.3, 0.2);
        let system = NormalEquationsBa {
            h_pp: h_pp.clone(),
            b_p: b_p.clone(),
            landmarks: vec![LandmarkBlock {
                h_ll,
                b_l,
                cross: vec![(0, cross)],
            }],
        };

        let (delta_p, delta_l) =
            solve_step(&system, 1, 1, 0, 0, 0.0, LinearSolver::Dense, false).unwrap();

        let mut full_h = DMatrix::<f64>::zeros(9, 9);
        full_h.view_mut((0, 0), (6, 6)).copy_from(&h_pp);
        for r in 0..6 {
            for c in 0..3 {
                full_h[(r, 6 + c)] = cross[(r, c)];
                full_h[(6 + c, r)] = cross[(r, c)];
            }
        }
        full_h.view_mut((6, 6), (3, 3)).copy_from(&h_ll);
        let mut full_rhs = DVector::<f64>::zeros(9);
        for i in 0..6 {
            full_rhs[i] = -b_p[i];
        }
        for i in 0..3 {
            full_rhs[6 + i] = -b_l[i];
        }
        let full_delta = solve_normal_equations(&full_h, &full_rhs).unwrap();
        assert!((delta_p - full_delta.rows(0, 6)).norm() < 1.0e-10);
        assert!((delta_l - full_delta.rows(6, 3)).norm() < 1.0e-10);
    }
}

#[cfg(test)]
mod imu_gradient_tests {
    use super::*;
    use crate::imu_preintegration::ImuPreintegrator;
    use nalgebra::UnitQuaternion;

    fn make_problem() -> BundleAdjustment {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut ba = BundleAdjustment::new(camera);
        let pose_i = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.12, -0.08, 0.2),
            Vector3::new(0.3, -0.2, 0.1),
        );
        let pose_j = Pose::from_world_to_camera(
            UnitQuaternion::from_euler_angles(0.17, -0.03, 0.27),
            Vector3::new(0.42, -0.15, 0.18),
        );
        ba.add_pose(10, pose_i);
        ba.add_pose(20, pose_j);
        ba.add_velocity(10, Vector3::new(0.4, -0.1, 0.2));
        ba.add_velocity(20, Vector3::new(0.35, 0.05, 0.1));
        ba.add_bias(10, Vector6::new(0.002, -0.001, 0.003, 0.02, -0.01, 0.03));
        ba.add_bias(20, Vector6::zeros());
        ba.set_imu_body_to_camera(SE3::new(
            UnitQuaternion::from_euler_angles(-0.04, 0.03, -0.02),
            Vector3::new(-0.02, -0.06, 0.01),
        ));
        let mut preintegrator = ImuPreintegrator::new();
        for _ in 0..20 {
            preintegrator.integrate_sample(
                Vector3::new(0.03, -0.02, 0.04),
                Vector3::new(0.2, -0.1, 9.7),
                0.01,
            );
        }
        ba.add_imu_factor(ImuPreintegrationFactor {
            keyframe_id_from: 10,
            keyframe_id_to: 20,
            delta: preintegrator.delta(),
            gravity_world: Vector3::new(0.0, 0.0, -9.81),
            weight_position: 1.3,
            weight_velocity: 0.8,
            weight_rotation: 1.1,
        });
        ba
    }

    fn perturb(problem: &mut BundleAdjustment, coordinate: usize, step: f64) {
        match coordinate {
            0..=11 => {
                let pose_slot = coordinate / 6;
                let component = coordinate % 6;
                let id = [10_u64, 20][pose_slot];
                let mut xi = Vector6::zeros();
                xi[component] = step;
                let pose = problem.poses.get_mut(&id).unwrap();
                pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&xi));
            }
            12..=17 => {
                let velocity_slot = (coordinate - 12) / 3;
                let component = (coordinate - 12) % 3;
                let id = [10_u64, 20][velocity_slot];
                problem.velocities.get_mut(&id).unwrap()[component] += step;
            }
            18..=29 => {
                let bias_slot = (coordinate - 18) / 6;
                let component = (coordinate - 18) % 6;
                let id = [10_u64, 20][bias_slot];
                problem.biases.get_mut(&id).unwrap()[component] += step;
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn analytic_imu_gradient_matches_central_difference_with_extrinsic() {
        let problem = make_problem();
        let linearized = problem.linearized_navigation_system().unwrap();
        assert_eq!(linearized.information.shape(), (30, 30));
        let epsilon = 1.0e-6;
        for coordinate in 0..30 {
            let mut plus = problem.clone();
            let mut minus = problem.clone();
            perturb(&mut plus, coordinate, epsilon);
            perturb(&mut minus, coordinate, -epsilon);
            let numerical = (plus.robust_cost(&RobustKernel::None)
                - minus.robust_cost(&RobustKernel::None))
                / (4.0 * epsilon);
            let analytic = linearized.gradient[coordinate];
            let tolerance = 2.0e-4 * analytic.abs().max(numerical.abs()).max(1.0);
            assert!(
                (analytic - numerical).abs() <= tolerance,
                "coordinate {coordinate}: analytic={analytic} numerical={numerical} tolerance={tolerance}"
            );
        }
    }
}

/// Tests for [`BaConfig::parallel`] (see the module's "Parallelism"
/// section): the serial and parallel assembly / Schur-reduction /
/// back-substitution paths must agree, and the parallel path must be
/// deterministic. The synthetic problem is sized past
/// `PARALLEL_MIN_OBSERVATIONS` / `PARALLEL_MIN_LANDMARKS` so these tests
/// actually exercise the parallel dispatch rather than falling through the
/// work gate to the serial loops.
#[cfg(test)]
mod parallel_ba_tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    /// Deterministic `[0, 1)` pseudo-random value (GLSL-style sine hash) —
    /// avoids pulling in a `rand` dependency just to scatter synthetic
    /// points/poses reproducibly.
    fn pseudo_rand(seed: u64) -> f64 {
        let x = (seed as f64 + 1.0) * 12.9898;
        let y = x.sin() * 43758.5453;
        y - y.floor()
    }

    /// Build a synthetic multi-camera, multi-landmark monocular BA problem:
    /// `num_cameras` cameras translated along a horizontal baseline (plus a
    /// small yaw each) all observe every one of `num_landmarks` landmarks
    /// scattered in front of them, so every camera/landmark pair is a valid,
    /// positive-depth observation. The first two poses are fixed (anchor +
    /// scale, per the module doc's gauge-fixing rule); every other pose and
    /// every landmark is then nudged away from the ground truth that
    /// generated the observations by a small deterministic offset, so LM has
    /// a real (if easy, well-conditioned) problem to converge on rather than
    /// starting already at the optimum.
    fn build_synthetic_ba(num_cameras: usize, num_landmarks: usize) -> BundleAdjustment {
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut ba = BundleAdjustment::new(camera.clone());

        let center = (num_cameras as f64 - 1.0) / 2.0;
        let gt_poses: Vec<Pose> = (0..num_cameras)
            .map(|i| {
                let baseline = (i as f64 - center) * 0.4;
                let yaw = (i as f64 - center) * 0.02;
                let rotation = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
                let translation = Vector3::new(-baseline, 0.0, 0.0);
                Pose::from_world_to_camera(rotation, translation)
            })
            .collect();
        let gt_points: Vec<Point3<f64>> = (0..num_landmarks)
            .map(|j| {
                let x = (pseudo_rand(j as u64 * 3) - 0.5) * 6.0;
                let y = (pseudo_rand(j as u64 * 3 + 1) - 0.5) * 6.0;
                let z = 8.0 + pseudo_rand(j as u64 * 3 + 2) * 4.0;
                Point3::new(x, y, z)
            })
            .collect();

        for (i, pose) in gt_poses.iter().enumerate() {
            ba.add_pose(i as u64, pose.clone());
        }
        for (j, point) in gt_points.iter().enumerate() {
            ba.add_landmark(1_000_000 + j as u64, *point);
        }
        for (i, pose) in gt_poses.iter().enumerate() {
            for (j, point) in gt_points.iter().enumerate() {
                let xc = pose.transform_world_point(point);
                let xy = camera
                    .project(&xc)
                    .expect("synthetic landmarks stay in front of every camera");
                ba.add_observation(BaObservation {
                    keyframe_id: i as u64,
                    landmark_id: 1_000_000 + j as u64,
                    xy,
                });
            }
        }

        ba.fix_pose(0);
        ba.fix_pose(1);

        // Nudge every non-fixed pose off ground truth.
        for i in 2..num_cameras {
            let dxi = Vector6::new(
                (pseudo_rand(i as u64 * 7) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 1) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 2) - 0.5) * 0.02,
                (pseudo_rand(i as u64 * 7 + 3) - 0.5) * 0.01,
                (pseudo_rand(i as u64 * 7 + 4) - 0.5) * 0.01,
                (pseudo_rand(i as u64 * 7 + 5) - 0.5) * 0.01,
            );
            let pose = ba.poses.get_mut(&(i as u64)).expect("pose was just added");
            pose.world_to_camera = pose.world_to_camera.compose(&SE3::exp(&dxi));
        }
        // Nudge every landmark off ground truth.
        for j in 0..num_landmarks {
            let d = Vector3::new(
                (pseudo_rand(j as u64 * 11) - 0.5) * 0.05,
                (pseudo_rand(j as u64 * 11 + 1) - 0.5) * 0.05,
                (pseudo_rand(j as u64 * 11 + 2) - 0.5) * 0.05,
            );
            let point = ba
                .landmarks
                .get_mut(&(1_000_000 + j as u64))
                .expect("landmark was just added");
            *point = Point3::from(point.coords + d);
        }

        ba
    }

    /// `num_landmarks` clears `PARALLEL_MIN_LANDMARKS` and `num_cameras *
    /// num_landmarks` clears `PARALLEL_MIN_OBSERVATIONS`, so every parallel
    /// path in the module (assembly, Schur reduction, back-substitution) is
    /// actually dispatched by these tests instead of falling through the
    /// work gate.
    const TEST_CAMERAS: usize = 6;
    const TEST_LANDMARKS: usize = 2_500;

    #[test]
    fn parallel_config_defaults_to_off() {
        assert!(!BaConfig::default().parallel);
    }

    #[test]
    fn serial_and_parallel_converge_to_the_same_result() {
        let mut ba_serial = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
        let mut ba_parallel = ba_serial.clone();

        let serial_config = BaConfig {
            max_iterations: 8,
            parallel: false,
            ..BaConfig::default()
        };
        let parallel_config = BaConfig {
            parallel: true,
            ..serial_config
        };

        let result_serial = ba_serial
            .optimize(&serial_config)
            .expect("serial BA should solve the synthetic problem");
        let result_parallel = ba_parallel
            .optimize(&parallel_config)
            .expect("parallel BA should solve the synthetic problem");

        assert!(result_serial.converged, "serial run should converge");
        assert!(result_parallel.converged, "parallel run should converge");

        // The parallel assembly / Schur-reduction / back-substitution paths
        // change only *how* the normal equations are computed, never the
        // summation order (see the module's "Parallelism" section), so the
        // two runs must land on bit-identical states -- a far tighter check
        // than the "~1e-9 relative" bar a reassociating design would need.
        assert_eq!(
            result_serial.final_cost, result_parallel.final_cost,
            "final cost must match exactly"
        );
        assert_eq!(
            result_serial.iterations.len(),
            result_parallel.iterations.len(),
            "iteration count must match exactly"
        );
        assert_eq!(
            ba_serial.poses, ba_parallel.poses,
            "poses must match exactly"
        );
        assert_eq!(
            ba_serial.landmarks, ba_parallel.landmarks,
            "landmarks must match exactly"
        );
    }

    #[test]
    fn parallel_path_is_deterministic_across_runs() {
        let ba = build_synthetic_ba(TEST_CAMERAS, TEST_LANDMARKS);
        let mut ba_run_a = ba.clone();
        let mut ba_run_b = ba.clone();

        let config = BaConfig {
            max_iterations: 8,
            parallel: true,
            ..BaConfig::default()
        };

        let result_a = ba_run_a
            .optimize(&config)
            .expect("parallel BA should solve the synthetic problem");
        let result_b = ba_run_b
            .optimize(&config)
            .expect("parallel BA should solve the synthetic problem");

        assert_eq!(
            result_a.final_cost, result_b.final_cost,
            "repeated parallel runs must produce bitwise-identical cost"
        );
        assert_eq!(
            ba_run_a.poses, ba_run_b.poses,
            "repeated parallel runs must produce bitwise-identical poses"
        );
        assert_eq!(
            ba_run_a.landmarks, ba_run_b.landmarks,
            "repeated parallel runs must produce bitwise-identical landmarks"
        );
    }
}
