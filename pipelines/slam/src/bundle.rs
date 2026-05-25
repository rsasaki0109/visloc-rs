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

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{
    DMatrix, DVector, Matrix2x3, Matrix2x6, Matrix3, Matrix3x6, Matrix6, Matrix6x3, Point2, Point3,
    Vector2, Vector3, Vector6,
};

use visloc_core::geometry::{Pose, SE3, SO3};
use visloc_core::types::{Camera, CameraModel, VisualMap};
use visloc_mapping::{
    LocalMapWindow, LocalRefinementReason, LocalRefinementResult, LocalRefiner, StagedMapUpdate,
};

use crate::gnc::{GncConfig, GncState};
use crate::imu_preintegration::ImuPreintegrationFactor;
use crate::{solve_normal_equations, LinearSolver, PoseGraphError, RobustKernel};

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
/// biases. Adds the residual `r = b_j − b_i` weighted by the scalar
/// `weight` (per-axis isotropic). Use this to keep neighbouring
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
    /// Scalar weight (sqrt-information squared). The cost added is
    /// `weight · ‖b_j − b_i‖²` so `weight = 1 / σ²` for an isotropic
    /// per-axis bias random-walk noise `σ`. A typical value is
    /// `1 / (σ_bw² · Δt_{ij})` where `σ_bw` is the gyro/accel bias
    /// random-walk noise density and `Δt_{ij}` is the inter-keyframe
    /// time — but the caller chooses the units.
    pub weight: f64,
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
    pub camera: Camera,
    /// Pose ids whose `Pose` is held constant during optimization.
    pub fixed_poses: BTreeSet<u64>,
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
}

impl BundleAdjustment {
    pub fn new(camera: Camera) -> Self {
        Self {
            poses: BTreeMap::new(),
            landmarks: BTreeMap::new(),
            observations: Vec::new(),
            stereo_observations: Vec::new(),
            camera,
            fixed_poses: BTreeSet::new(),
            fixed_landmarks: BTreeSet::new(),
            stereo_baseline: None,
            gravity_prior: None,
            per_pose_gravity_prior: None,
            position_prior: None,
            pairwise_pose_factors: Vec::new(),
            velocities: BTreeMap::new(),
            fixed_velocities: BTreeSet::new(),
            imu_factors: Vec::new(),
            biases: BTreeMap::new(),
            fixed_biases: BTreeSet::new(),
            bias_random_walk_factors: Vec::new(),
        }
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

    /// Robust reprojection cost: `Σ ρ(||r||²)` where `ρ` is the supplied
    /// [`RobustKernel`]. With [`RobustKernel::None`] this matches
    /// [`Self::cost`]. Stereo observations contribute a 3-vector residual
    /// `(u_l_pred − u_l_meas, v_l_pred − v_l_meas, u_r_pred − u_r_meas)`
    /// where `u_r_pred = u_l_pred − fx · b / Z` (rectified-stereo assumption,
    /// see [`BaStereoObservation`]).
    pub fn robust_cost(&self, kernel: &RobustKernel) -> f64 {
        self.robust_cost_weighted(kernel, None)
    }

    /// Like [`Self::robust_cost`] but multiplies each reprojection
    /// observation's contribution by an external per-observation weight
    /// (the Graduated Non-Convexity Black-Rangarajan weight `w ∈ [0,1]`).
    /// `gnc_weights` is indexed monocular-observations-first
    /// (`0 .. observations.len()`) then stereo
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
            if let Some(predicted) = project_pinhole(&intrinsics, &xc) {
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
            total += factor.weight * r.norm_squared();
        }
        // IMU pre-integration factors: 9-vector residual [r_R; r_v; r_p]
        // weighted axis-wise. The factor's `residual` helper takes the
        // body-to-world rotation (= world_to_camera.inverse()) and the
        // world-frame camera centre, both pulled from the BA pose state.
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
            let r_i = SO3::from_quaternion(pose_i.world_to_camera.rotation.inverse());
            let r_j = SO3::from_quaternion(pose_j.world_to_camera.rotation.inverse());
            let p_i: Vector3<f64> = pose_i.camera_center_world().coords;
            let p_j: Vector3<f64> = pose_j.camera_center_world().coords;
            let [r_rot, r_vel, r_pos] =
                if let Some(bias) = self.biases.get(&factor.keyframe_id_from) {
                    let bg: Vector3<f64> = bias.fixed_rows::<3>(0).into_owned();
                    let ba: Vector3<f64> = bias.fixed_rows::<3>(3).into_owned();
                    factor.residual_with_bias_correction(&r_i, &p_i, v_i, &r_j, &p_j, v_j, &bg, &ba)
                } else {
                    factor.residual(&r_i, &p_i, v_i, &r_j, &p_j, v_j)
                };
            total += factor.weight_rotation * r_rot.norm_squared()
                + factor.weight_velocity * r_vel.norm_squared()
                + factor.weight_position * r_pos.norm_squared();
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
    /// (`0 .. observations.len()`), then stereo. An observation that cannot
    /// be evaluated now (missing pose / landmark, behind the camera, or
    /// non-projectable, or — for stereo — no usable baseline) is reported
    /// as `f64::NAN`, so it neither sets the GNC inlier scale nor is
    /// classified as an inlier or outlier.
    fn reprojection_squared_residuals(&self) -> Vec<f64> {
        let n = self.observations.len() + self.stereo_observations.len();
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
        out
    }

    /// Run Levenberg-Marquardt bundle adjustment with Schur-complement
    /// landmark elimination. Returns iteration trace and final cost.
    pub fn optimize(&mut self, config: &BaConfig) -> Result<BaResult, BaError> {
        self.optimize_weighted(config, None)
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
    /// weight (monocular-first then stereo, `NaN` for un-evaluable
    /// observations); near-zero entries are the rejected outliers.
    pub fn optimize_gnc(
        &mut self,
        config: &BaConfig,
        gnc: &GncConfig,
    ) -> Result<BaGncResult, BaError> {
        let kernel_none = RobustKernel::None;
        let initial_cost = self.robust_cost(&kernel_none);
        let n = self.observations.len() + self.stereo_observations.len();

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
        let inlier_scale = effective_gnc.c;
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
    /// indexed monocular-first then stereo (see
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
        let has_visual_observations =
            !self.observations.is_empty() || !self.stereo_observations.is_empty();
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
        let mut lambda = config.initial_lambda.unwrap_or(0.0);
        let mut converged = false;

        for iteration in 0..config.max_iterations {
            let system = build_normal_equations(
                self,
                &intrinsics,
                &pose_index,
                &landmark_index,
                &velocity_index,
                &bias_index,
                &kernel,
                gnc_weights,
            );

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
                let xi = delta_poses.fixed_rows::<6>(i * 6).into_owned();
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
            let step_accepted = match config.initial_lambda {
                None => true, // Pure GN: accept unconditionally.
                Some(_) => cost_after < cost_before,
            };

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
        }

        Ok(BaResult {
            initial_cost,
            final_cost: current_cost,
            iterations,
            converged,
        })
    }
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
            linear_solver: LinearSolver::Dense,
            robust_kernel: RobustKernel::None,
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
    /// monocular-first then stereo. `NaN` marks an observation that could
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

/// Output of the per-iteration normal-equations build.
struct NormalEquationsBa {
    /// Pose-pose Hessian, dense `(6P) × (6P)`.
    h_pp: DMatrix<f64>,
    /// Pose gradient, dense `6P`.
    b_p: DVector<f64>,
    /// Landmark blocks indexed by landmark variable index.
    landmarks: Vec<LandmarkBlock>,
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
    //   r_R = log(ΔR.T · R_iᵀ · R_j)            (Forster's R_i = R_wcᵢ.T)
    //   r_v = R_wcᵢ · (v_j − v_i − g·Δt) − Δv
    //   r_p = R_wcᵢ · (C_j − C_i − v_i·Δt − ½ g Δt²) − Δp
    //
    // Right-perturbation Jacobians (ρ = translation perturbation, ω =
    // rotation perturbation, both 3-vec; world camera centre
    // C = −Rᵀt so ∂C/∂ρ = −I, ∂C/∂ω = [C]×):
    //
    //   ∂r_R/∂ω_i =  Jr_inv(r_R) · R_wcⱼ
    //   ∂r_R/∂ω_j = −Jr_inv(r_R) · R_wcⱼ
    //   ∂r_v/∂ω_i = −R_wcᵢ · [v_j − v_i − g·Δt]×
    //   ∂r_v/∂v_i = −R_wcᵢ
    //   ∂r_v/∂v_j =  R_wcᵢ
    //   ∂r_p/∂ρ_i =  R_wcᵢ            ∂r_p/∂ρ_j = −R_wcᵢ
    //   ∂r_p/∂ω_i = −R_wcᵢ · [C_j − v_i·Δt − ½ g Δt²]×
    //   ∂r_p/∂ω_j =  R_wcᵢ · [C_j]×
    //   ∂r_p/∂v_i = −Δt · R_wcᵢ
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
        let r_wc_i = pose_i
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let r_wc_j = pose_j
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        let c_i: Vector3<f64> = pose_i.camera_center_world().coords;
        let c_j: Vector3<f64> = pose_j.camera_center_world().coords;
        let dt = factor.delta.delta_time;
        let g = factor.gravity_world;

        // Residual (use the same formulation as the factor's residual()
        // helper so the cost evaluation and Jacobian linearise at the
        // same point). When the "from" keyframe has a registered bias,
        // apply the first-order bias correction; otherwise fall back
        // to the un-corrected residual (the linearisation bias is
        // implicit in the integrated delta).
        let r_i_so3 = SO3::from_quaternion(pose_i.world_to_camera.rotation.inverse());
        let r_j_so3 = SO3::from_quaternion(pose_j.world_to_camera.rotation.inverse());
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

        // Axis-wise √w. Weights are nonnegative; clamp to zero just in case.
        let sqrt_w_r = factor.weight_rotation.max(0.0).sqrt();
        let sqrt_w_v = factor.weight_velocity.max(0.0).sqrt();
        let sqrt_w_p = factor.weight_position.max(0.0).sqrt();

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
            .copy_from(&(sqrt_w_r * jr_inv_rwc_j));
        // r_v block: ω_i column = −R_wcᵢ · [q_diff]×.
        j_pose_i
            .fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(-sqrt_w_v * r_wc_i * skew(&q_diff)));
        // r_p block: ρ_i = R_wcᵢ, ω_i = −R_wcᵢ · [q_pos_i]×.
        j_pose_i
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(sqrt_w_p * r_wc_i));
        j_pose_i
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(-sqrt_w_p * r_wc_i * skew(&q_pos_i)));

        // J_pose_j: 9×6, columns [ρ_j | ω_j].
        let mut j_pose_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U6, _>::zeros();
        j_pose_j
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-sqrt_w_r * jr_inv_rwc_j));
        // r_v has no R_j / p_j dependence (block stays zero).
        // r_p block: ρ_j = −R_wcᵢ, ω_j = R_wcᵢ · [C_j]×.
        j_pose_j
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-sqrt_w_p * r_wc_i));
        j_pose_j
            .fixed_view_mut::<3, 3>(6, 3)
            .copy_from(&(sqrt_w_p * r_wc_i * skew(&c_j)));

        // J_vel_i / J_vel_j: 9×3 each.
        let mut j_vel_i = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_i
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(-sqrt_w_v * r_wc_i));
        j_vel_i
            .fixed_view_mut::<3, 3>(6, 0)
            .copy_from(&(-sqrt_w_p * dt * r_wc_i));
        let mut j_vel_j = nalgebra::Matrix::<f64, nalgebra::U9, nalgebra::U3, _>::zeros();
        j_vel_j
            .fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(sqrt_w_v * r_wc_i));

        // √w-scaled residual.
        let mut r_stack = nalgebra::SVector::<f64, 9>::zeros();
        r_stack
            .fixed_rows_mut::<3>(0)
            .copy_from(&(sqrt_w_r * r_rot));
        r_stack
            .fixed_rows_mut::<3>(3)
            .copy_from(&(sqrt_w_v * r_vel));
        r_stack
            .fixed_rows_mut::<3>(6)
            .copy_from(&(sqrt_w_p * r_pos));

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
                .copy_from(&(sqrt_w_r * lhs_rot * factor.delta.j_rotation_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 0)
                .copy_from(&(-sqrt_w_v * factor.delta.j_velocity_bg));
            j_bias
                .fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(-sqrt_w_v * factor.delta.j_velocity_ba));
            j_bias
                .fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(-sqrt_w_p * factor.delta.j_position_bg));
            j_bias
                .fixed_view_mut::<3, 3>(6, 3)
                .copy_from(&(-sqrt_w_p * factor.delta.j_position_ba));
            Some(j_bias)
        } else {
            None
        };

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
    // `J_i = −I, J_j = I`, weight scaling `w = factor.weight`. The
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
        let w = factor.weight.max(0.0);
        if w <= 0.0 {
            continue;
        }
        let i_slot = bias_index.get(&factor.keyframe_id_from).copied();
        let j_slot = bias_index.get(&factor.keyframe_id_to).copied();
        if let Some(i) = i_slot {
            for k in 0..6 {
                h_pp[(bias_offset + i * 6 + k, bias_offset + i * 6 + k)] += w;
                b_p[bias_offset + i * 6 + k] += -w * r[k];
            }
        }
        if let Some(j) = j_slot {
            for k in 0..6 {
                h_pp[(bias_offset + j * 6 + k, bias_offset + j * 6 + k)] += w;
                b_p[bias_offset + j * 6 + k] += w * r[k];
            }
        }
        if let (Some(i), Some(j)) = (i_slot, j_slot) {
            for k in 0..6 {
                h_pp[(bias_offset + i * 6 + k, bias_offset + j * 6 + k)] += -w;
                h_pp[(bias_offset + j * 6 + k, bias_offset + i * 6 + k)] += -w;
            }
        }
    }

    NormalEquationsBa {
        h_pp,
        b_p,
        landmarks,
    }
}

fn solve_step(
    system: &NormalEquationsBa,
    p_count: usize,
    l_count: usize,
    v_count: usize,
    b_count: usize,
    lambda: f64,
    linear_solver: LinearSolver,
) -> Result<(DVector<f64>, DVector<f64>), BaError> {
    // Landmark-only BA: H_LL is block-diagonal so each landmark gets an
    // independent 3×3 solve. No Schur complement needed.
    if p_count == 0 && v_count == 0 && b_count == 0 {
        let mut delta_l = DVector::<f64>::zeros(l_count * 3);
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
    //   b_S -= H_PL_l · H_LL_l^{-1} · b_l
    // Both updates only touch the rows/cols of S corresponding to poses
    // that observed this landmark, so we never materialize the full H_PL.
    let mut h_ll_inv_cache: Vec<Option<Matrix3<f64>>> = Vec::with_capacity(system.landmarks.len());
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
            // so each landmark `l` contributes `+ A · h_ll_inv · b_l` to
            // `b_reduced` (which already starts at `-g_p = -b_p`).
            let upd: Vector6<f64> = a_h * landmark.b_l;
            for k in 0..6 {
                b_reduced[p * 6 + k] += upd[k];
            }
        }
    }

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
    let mut delta_l = DVector::<f64>::zeros(system.landmarks.len() * 3);
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

    Ok((delta_p, delta_l))
}

fn project_pinhole(intrinsics: &(f64, f64, f64, f64), xc: &Point3<f64>) -> Option<Point2<f64>> {
    if xc.z <= 0.0 {
        return None;
    }
    let (fx, fy, cx, cy) = *intrinsics;
    Some(Point2::new(fx * xc.x / xc.z + cx, fy * xc.y / xc.z + cy))
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
