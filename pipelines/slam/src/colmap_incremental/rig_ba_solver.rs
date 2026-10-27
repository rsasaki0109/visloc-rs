//! C2.5: fast native bundle-adjustment backend for
//! `bundle_adjustment.rs::solve`'s rig-reprojection problem
//! (`bundle::BundleAdjustment` populated with `add_pose`/`add_landmark`/
//! `add_rig_observation`/`fix_pose`/`fix_landmark` only — the exact subset
//! `bundle_adjustment.rs::solve` builds).
//!
//! Replaces `bundle::BundleAdjustment::optimize`'s `BTreeMap`-indexed,
//! serially-assembled, densely-Schur-solved LM loop (measured ~340ms/iter at
//! 97 frames / 51k obs / 1.7k points; the dense 7.5k×7.5k reduced-camera
//! solve for a 1250-frame problem is already the dominant cost and is
//! infeasible past ~10k frames) with:
//! - contiguous `Vec`-of-struct observation/frame/point storage (§ below),
//!   sorted by point so per-point Schur elimination reads a contiguous slice
//!   — no map lookups in the per-observation/per-point hot loops;
//! - `rayon`-parallel Jacobian evaluation and per-point Schur elimination,
//!   reduced deterministically (every parallel step is either a positional
//!   `collect()`, whose result order is independent of the thread count by
//!   construction of `rayon`'s `IndexedParallelIterator`, or a serial fold
//!   over that already-ordered `Vec` — never a `rayon` `.sum()`/`.reduce()`,
//!   whose split structure this module does not want to depend on);
//! - the reduced (pose-only) camera system solved as a 6×6-block-sparse
//!   system via [`crate::block_cholesky::solve_spd_blocks6_cached`] (already
//!   in this crate; reused here rather than re-implemented, per the task
//!   brief's "reuse `block_cholesky.rs` if it fits" — it does: the module's
//!   own doc frames it as "the BA Schur solve" as one of its two intended
//!   callers). No separate dense path is implemented for small frame counts
//!   (a documented deviation from the task's "dense fallback ≤200 frames"
//!   suggestion): `block_cholesky`'s own doc states it is the *general*
//!   production path (dense-at-block-granularity even for small block
//!   counts, see its "Why not supernodal?" section), so a second,
//!   independently-tested dense code path would duplicate risk without a
//!   measured benefit — verified instead by comparing Native's output
//!   directly against `BundleAdjustment::optimize` (`Legacy`, itself a dense
//!   solver) on identical small synthetic problems (`native_vs_legacy_*`
//!   tests below).
//!
//! ## Residual model
//! Identical formula to `bundle.rs`'s `rig_residual_jacobians` (the existing,
//! already-exercised port of COLMAP's `RigReprojErrorConstantRigCostFunctor`,
//! `cost_functions/reprojection_error.h:389-417`, documented in
//! `bundle_adjustment.rs`'s module doc citation): pinhole intrinsics
//! `(fx, fy, cx, cy)`, `sensor_from_rig`, variable `rig_from_world` pose and
//! variable landmark `xyz`. The intrinsics and `sensor_from_rig` are fixed
//! data unless [`ExtraBlocks`] makes them variables:
//! ```text
//! point_rig    = rig_from_world · point_world
//! point_sensor = sensor_from_rig · point_rig
//! residual     = (fx·x/z + cx, fy·y/z + cy) − observed_xy     (point_sensor = (x,y,z))
//! ```
//! Robust loss (`SoftLOneLoss`/`CauchyLoss`, selectable per
//! [`super::bundle_adjustment::LossFunction`]) is supported per the C2.7 task
//! brief's finding that COLMAP's control is **not** trivial-loss everywhere:
//! `LocalBundleAdjustment()` sets `SOFT_L1` (scale 1.0,
//! `controllers/incremental_pipeline.cc:217-219`), `GlobalBundleAdjustment()`
//! stays `TRIVIAL` (`:267-268`), and `IterativeLocalRefinement` downgrades to
//! `TRIVIAL` after its first iteration ("Only use robust cost function for
//! first iteration", `sfm/incremental_mapper.cc:1277-1281`) — see
//! `bundle_adjustment.rs`'s module doc and `mapper.rs::iterative_local_refinement`
//! for how those defaults/the per-iteration switch are threaded through.
//! [`Corrector`] below is a direct, 1:1 port of Ceres' own residual/Jacobian
//! robustification (`internal/ceres/corrector.{h,cc}`), including the exact
//! call order and cost convention `residual_block.cc:161-197` establishes
//! (`cost = 0.5·rho(s)`; Jacobian corrected *before* the residual, both using
//! the *uncorrected* residual) — this module's `linearize_point`/
//! `evaluate_cost` mirror that order exactly. No deviation was needed: the
//! algorithm is closed-form (no root-finding, no auxiliary randomness).
//! [`residual_and_jacobians`] below is an independent, allocation-free
//! re-derivation of the same formula (not a call into `bundle.rs`, keeping
//! this module link-independent); `tests::matches_bundle_rs_formula` checks
//! the two agree bit-for-bit-close on random inputs, and
//! `tests::jacobian_matches_finite_difference` checks both partials
//! numerically.
//!
//! ## Pose parameterization
//! `rig_from_world` is `Pose { world_to_camera: SE3 }` (`SE3 { rotation:
//! UnitQuaternion, translation }`). The local update is this crate's
//! existing SE(3) convention — the same one every other `bundle.rs` LM loop
//! already uses (e.g. `bundle.rs:2894,3906,16075,16205`):
//! `T_new = T_old.compose(SE3::exp(xi))`, `xi = [ρ; ω] ∈ R^6` (translation
//! tangent first, then rotation axis-angle) — a *right* (local, body-frame)
//! 6-DoF manifold update. COLMAP's own Ceres setup instead uses
//! `EigenQuaternionManifold` for the rotation (a separate 4-parameter block
//! with its own local parameterization) plus a plain unconstrained R³ block
//! for the translation. Both are minimal 6-DoF local parameterizations of
//! the same SE(3) manifold at the same linearization point; the converged
//! optimum (a stationary point of the same reprojection-error cost) and the
//! Gauss-Newton/LM trust-region geometry near it coincide to first order —
//! the difference is bookkeeping (one coupled 6-vector update vs. two
//! independent blocks), not a modeling difference. This port keeps this
//! crate's existing convention rather than introducing a second one.
//!
//! ## Trust region (Ceres `LEVENBERG_MARQUARDT` defaults, as COLMAP's control
//! configures them — `bundle_adjustment_ceres.cc:103-105` sets
//! `function_tolerance=0`, `parameter_tolerance=0`; `gradient_tolerance` is
//! left at Ceres' own default `1e-4`; the trust-region strategy and its
//! `initial_trust_region_radius=1e4` are Ceres solver defaults, not
//! overridden by COLMAP)
//! - Diagonal (Jacobi) scaling: each parameter block's damping uses
//!   `mu · clamp(diag(H_block), 1e-6, 1e32)` (`mu = 1/radius`), added to that
//!   block's own diagonal only — the well-known
//!   `LevenbergMarquardtStrategy` clamp constants from the public Ceres
//!   Solver source (`internal/ceres/levenberg_marquardt_strategy.cc`).
//! - Step quality `rho = actual_cost_decrease / predicted_cost_decrease`
//!   (both on the Ceres-internal `0.5·Σr²` cost scale, so the gradient
//!   `g = Jᵀr` and the `gradient_tolerance` check line up with Ceres';
//!   [`BaResult`]'s reported `initial_cost`/`final_cost` are rescaled back to
//!   `Σr²` to match `BundleAdjustment::cost()`'s convention for the
//!   Native/Legacy parity test). `predicted_decrease = ½·dx·(b + μ·D·dx)`,
//!   the standard trust-region model-decrease identity for a step solving
//!   `(H + μD)dx = b`.
//! - Accept iff `rho > 1e-3`; on accept `radius ← radius / max(1/3, 1 −
//!   (2ρ−1)³)`, reset the decrease factor to 2; on reject `radius ←
//!   radius / decrease_factor`, then `decrease_factor *= 2`.
//! - Terminate on `max_num_iterations` (from
//!   [`super::bundle_adjustment::BundleAdjustmentOptions`]), on the gradient
//!   criterion (checked once per trial, at the current linearization), or —
//!   a deliberate deviation from COLMAP's `function_tolerance=0`, see
//!   [`FUNCTION_TOLERANCE`] — once an accepted step lowers the cost by less
//!   than `1e-5` of it (`parameter_tolerance=0` stays disabled).
//! - A trial whose linear solve fails is an invalid step, as in Ceres: it is
//!   rejected (the trust region shrinks) and the solve continues;
//!   [`MAX_CONSECUTIVE_INVALID_STEPS`] in a row end it with
//!   `BaError::SingularSystem`.
//! - Jacobians are only re-evaluated after an **accepted** step (mirroring
//!   Ceres' `TrustRegionMinimizer`); a rejected trial only re-solves the
//!   reduced system at the new damping and re-evaluates the (Jacobian-free)
//!   candidate cost.
//!
//! ## Gauge / constant handling
//! Frames in `ba.fixed_poses` and points in `ba.fixed_landmarks` are never
//! given a variable slot (their observations still constrain the *other*
//! side of the residual — see [`Problem::free_frame_slot`] / point handling
//! in [`linearize_point`]) — this is `bundle_adjustment.rs`'s existing
//! `BundleAdjustmentConfig`/`Gauge` mechanism (`set_constant_rig_from_world_pose`,
//! `add_constant_point`, `fix_gauge`), unchanged by this backend swap.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use nalgebra::{
    DMatrix, DVector, Matrix3, Matrix6, Point2, Point3, SMatrix, Vector2, Vector3, Vector6,
};
use rayon::prelude::*;

use visloc_core::geometry::SE3;
use visloc_core::types::{Camera, CameraModel};

use crate::ba_accel::ReducedCameraSystem;
use crate::block_cholesky::{solve_spd_blocks6_cached, BlockSymbolic};
use crate::bundle::{BaError, BaIterationStats, BaResult, BaRigObservation, BundleAdjustment};

use super::bundle_adjustment::LossFunction;

/// Ceres `LevenbergMarquardtStrategy` defaults (public Ceres Solver source,
/// `internal/ceres/levenberg_marquardt_strategy.cc` /
/// `trust_region_minimizer.cc`); see module doc.
const INITIAL_RADIUS: f64 = 1e4;
const MIN_RELATIVE_DECREASE: f64 = 1e-3;
const GRADIENT_TOLERANCE: f64 = 1e-4;
const MIN_DIAGONAL: f64 = 1e-6;
const MAX_DIAGONAL: f64 = 1e32;

/// One rig-reprojection observation: its frame, its pixel and its camera
/// (an index into the solve's [`ObsCamera`]s) — no `Camera`/`SE3` lookups by
/// id anywhere in the per-observation hot loop. Grouped by point in
/// [`Problem::obs`] so a point's observations are a contiguous slice. The
/// intrinsics and `sensor_from_rig` used to be copied into every
/// observation, 112 bytes against these 24: 1.3 GB of SmallCity's last
/// bundle adjustment (15M observations, one camera).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Obs {
    frame_idx: u32,
    camera: u32,
    xy: Point2<f64>,
}

impl Obs {
    /// An observation of frame `frame_idx` (index into
    /// [`RigBaInput::frame_ids`]) through `cameras[camera]`
    /// ([`PointMajorObs::cameras`]).
    pub(crate) const fn new(frame_idx: u32, xy: Point2<f64>, camera: u32) -> Self {
        Self {
            frame_idx,
            camera,
            xy,
        }
    }
}

/// The pinhole intrinsics and `sensor_from_rig` of observations
/// ([`Obs::camera`]).
#[derive(Debug, Clone)]
pub(crate) struct ObsCamera {
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: SE3,
    /// Index into [`ExtraBlocks::sensors`] when this `sensor_from_rig` is
    /// refined, else [`NO_SENSOR`].
    sensor: u32,
}

/// [`ObsCamera::sensor`] of a camera whose `sensor_from_rig` is held fixed.
const NO_SENSOR: u32 = u32::MAX;

/// The intrinsics tuple of an equirectangular panorama of `width x height`
/// px: `fx == 0` marks it, `(cx, cy)` carry the size (see
/// [`project_sensor_point`]).
pub(crate) const fn equirectangular_intrinsics(width: f64, height: f64) -> (f64, f64, f64, f64) {
    (0.0, 0.0, width, height)
}

/// Pixel of a sensor-frame point and its `2×3` Jacobian, for the
/// intrinsics `(fx, fy, cx, cy)`: a pinhole, or with `fx == 0`
/// ([`equirectangular_intrinsics`]) an equirectangular panorama of
/// `cx x cy` px (`visloc_core`'s `CameraModel::Equirectangular`), which
/// projects every direction. `None` behind a pinhole sensor or on the
/// panorama's vertical axis.
#[inline]
fn project_sensor_point(
    (fx, fy, cx, cy): (f64, f64, f64, f64),
    point_sensor: &Point3<f64>,
) -> Option<(Point2<f64>, SMatrix<f64, 2, 3>)> {
    if fx == 0.0 {
        return project_equirectangular(cx, cy, point_sensor);
    }
    if point_sensor.z <= 0.0 {
        return None;
    }
    let z_inv = point_sensor.z.recip();
    let z_inv2 = z_inv * z_inv;
    let predicted = Point2::new(
        fx * point_sensor.x * z_inv + cx,
        fy * point_sensor.y * z_inv + cy,
    );
    let projection = SMatrix::<f64, 2, 3>::new(
        fx * z_inv,
        0.0,
        -fx * point_sensor.x * z_inv2,
        0.0,
        fy * z_inv,
        -fy * point_sensor.y * z_inv2,
    );
    Some((predicted, projection))
}

/// [`project_sensor_point`]'s panorama: azimuth `atan2(x, z)` across the
/// width, elevation `atan2(-y, hypot(x, z))` down the height.
#[cold]
fn project_equirectangular(
    width: f64,
    height: f64,
    p: &Point3<f64>,
) -> Option<(Point2<f64>, SMatrix<f64, 2, 3>)> {
    use std::f64::consts::PI;
    let rho2 = p.x * p.x + p.z * p.z;
    if rho2 < 1.0e-24 {
        return None;
    }
    let rho = rho2.sqrt();
    let r2 = rho2 + p.y * p.y;
    let predicted = Point2::new(
        (p.x.atan2(p.z) / (2.0 * PI) + 0.5) * width,
        (0.5 - (-p.y).atan2(rho) / PI) * height,
    );
    let (su, sv) = (width / (2.0 * PI), -height / PI);
    let jacobian = SMatrix::<f64, 2, 3>::new(
        su * p.z / rho2,
        0.0,
        -su * p.x / rho2,
        sv * p.y * p.x / (rho * r2),
        -sv * rho / r2,
        sv * p.y * p.z / (rho * r2),
    );
    Some((predicted, jacobian))
}

/// `predicted - xy`, with a panorama's horizontal residual wrapped into
/// `[-width / 2, width / 2]`: its left and right edges are one meridian.
#[inline]
fn pixel_residual(fx: f64, cx: f64, predicted: Point2<f64>, xy: Point2<f64>) -> Vector2<f64> {
    let mut residual = Vector2::new(predicted.x - xy.x, predicted.y - xy.y);
    if fx == 0.0 {
        residual.x -= cx * (residual.x / cx).round();
    }
    residual
}

impl ObsCamera {
    /// Pinhole `intrinsics` behind a fixed `sensor_from_rig`.
    pub(crate) const fn new((fx, fy, cx, cy): (f64, f64, f64, f64), sensor_from_rig: SE3) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            sensor_from_rig,
            sensor: NO_SENSOR,
        }
    }

    /// The same camera with its `sensor_from_rig` refined as
    /// [`ExtraBlocks::sensors`]`[sensor]`.
    pub(crate) const fn with_sensor(mut self, sensor: u32) -> Self {
        self.sensor = sensor;
        self
    }

    /// Every field's bits: equal keys, identical cameras.
    fn key(&self) -> [u64; 12] {
        let (q, t) = (
            self.sensor_from_rig.rotation.coords,
            self.sensor_from_rig.translation,
        );
        [
            self.fx.to_bits(),
            self.fy.to_bits(),
            self.cx.to_bits(),
            self.cy.to_bits(),
            q.x.to_bits(),
            q.y.to_bits(),
            q.z.to_bits(),
            q.w.to_bits(),
            t.x.to_bits(),
            t.y.to_bits(),
            t.z.to_bits(),
            u64::from(self.sensor),
        ]
    }
}

/// Distinct [`ObsCamera`]s in first-use order, for
/// [`PointMajorObs::cameras`]: cameras with the same bits share an index.
#[derive(Debug, Default)]
pub(crate) struct ObsCameraTable {
    cameras: Vec<ObsCamera>,
    index: std::collections::HashMap<[u64; 12], u32>,
}

impl ObsCameraTable {
    /// `camera`'s index, added if new.
    pub(crate) fn index_of(&mut self, camera: ObsCamera) -> u32 {
        let cameras = &mut self.cameras;
        *self.index.entry(camera.key()).or_insert_with(|| {
            cameras.push(camera);
            (cameras.len() - 1) as u32
        })
    }

    pub(crate) fn into_cameras(self) -> Vec<ObsCamera> {
        self.cameras
    }
}

/// A solve's observations already grouped by point: `obs[start..end]` for
/// `point_obs_range[p] = (start, end)`, `p` indexing
/// [`RigBaInput::point_ids`] — the layout [`build_problem`] otherwise
/// derives from image-ordered [`RigObs`] by a stable sort on the point —
/// and the cameras they index.
pub(crate) struct PointMajorObs {
    pub obs: Vec<Obs>,
    pub point_obs_range: Vec<(u32, u32)>,
    pub cameras: Vec<ObsCamera>,
}

/// Pinhole intrinsics shared by every observation of a solve and refined
/// with the poses and points: COLMAP's `ba_refine_focal_length` and
/// `ba_refine_principal_point` (`bundle_adjustment_ceres.cc`, the camera
/// block with a `SubsetManifold` over the fixed parameters). Deviation: one
/// focal step moves `fx` and `fy` together (square pixels, as COLMAP's
/// `SIMPLE_PINHOLE`), where COLMAP's `PINHOLE` refines them independently.
/// The observations' own intrinsics are ignored while it is set (see
/// [`ExtraBlocks`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SharedIntrinsics {
    /// `[fx, fy, cx, cy]`.
    pub params: [f64; 4],
    pub refine_focal_length: bool,
    pub refine_principal_point: bool,
}

impl SharedIntrinsics {
    /// Which of the camera block's `[focal, cx, cy]` steps are free.
    const fn free(&self) -> [bool; 3] {
        [
            self.refine_focal_length,
            self.refine_principal_point,
            self.refine_principal_point,
        ]
    }

    fn stepped(&self, dk: &Vector6<f64>) -> Self {
        let mut next = *self;
        next.params[0] += dk[0];
        next.params[1] += dk[0];
        next.params[2] += dk[1];
        next.params[3] += dk[2];
        next
    }
}

/// Parameter blocks refined with the frame poses and points beyond them:
/// the one camera of every observation ([`SharedIntrinsics`]) and the
/// `sensor_from_rig` poses that cameras name with [`ObsCamera::with_sensor`]
/// (COLMAP's `ba_refine_sensor_from_rig`; a sensor pose steps as
/// `sensor_from_rig ∘ exp(ξ)`, like the frame poses). Each is one more 6-dim
/// node of the reduced system after the free frames (the camera's
/// `[focal, cx, cy]` padded with three fixed slots): an observation couples
/// its frame, camera and sensor nodes directly, and its point couples them
/// through the elimination, so the direct and iterative solves take them
/// like frames. Fixed parameters keep a zero row and column (identity on
/// the diagonal), hence a zero step.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct ExtraBlocks {
    pub camera: Option<SharedIntrinsics>,
    pub sensors: Vec<SE3>,
}

impl ExtraBlocks {
    fn len(&self) -> usize {
        usize::from(self.camera.is_some()) + self.sensors.len()
    }

    /// Which of extra node `e`'s 6 steps are free.
    fn free(&self, e: usize) -> [bool; 6] {
        match (e, &self.camera) {
            (0, Some(c)) => {
                let [f, cx, cy] = c.free();
                [f, cx, cy, false, false, false]
            }
            _ => [true; 6],
        }
    }

    /// The blocks after extra node `e` stepped by `dx[e]`.
    fn stepped(&self, dx: &[Vector6<f64>]) -> Self {
        let offset = usize::from(self.camera.is_some());
        Self {
            camera: self.camera.map(|c| c.stepped(&dx[0])),
            sensors: self
                .sensors
                .iter()
                .enumerate()
                .map(|(s, pose)| pose.compose(&SE3::exp(&dx[offset + s])))
                .collect(),
        }
    }

    /// `obs`'s intrinsics and `sensor_from_rig` under these blocks, its
    /// camera `cameras[obs.camera]`.
    fn camera_of<'a>(
        &'a self,
        o: &Obs,
        cameras: &'a [ObsCamera],
    ) -> ((f64, f64, f64, f64), &'a SE3) {
        let camera = &cameras[o.camera as usize];
        let k = match &self.camera {
            Some(c) => (c.params[0], c.params[1], c.params[2], c.params[3]),
            None => (camera.fx, camera.fy, camera.cx, camera.cy),
        };
        let sensor = if camera.sensor == NO_SENSOR {
            &camera.sensor_from_rig
        } else {
            &self.sensors[camera.sensor as usize]
        };
        (k, sensor)
    }
}

/// The reduced-system nodes an observation touches, as frame indices (extra
/// nodes numbered after the `n_frames` frames, see [`ExtraBlocks`]): its
/// frame, the camera and its sensor.
fn obs_nodes(
    o: &Obs,
    cameras: &[ObsCamera],
    n_frames: u32,
    has_camera: bool,
) -> impl Iterator<Item = u32> {
    let sensor_base = n_frames + u32::from(has_camera);
    let sensor = cameras[o.camera as usize].sensor;
    [
        Some(o.frame_idx),
        has_camera.then_some(n_frames),
        (sensor != NO_SENSOR).then(|| sensor_base + sensor),
    ]
    .into_iter()
    .flatten()
}

struct Problem {
    /// `Some(slot)` for free frames (`slot` indexes the reduced camera
    /// system, `0..n_free_frames`), `None` for fixed frames.
    free_frame_slot: Vec<Option<u32>>,
    n_free_frames: usize,
    poses: Vec<SE3>,

    point_fixed: Vec<bool>,
    points: Vec<Point3<f64>>,

    obs: Vec<Obs>,
    /// `[start, end)` into `obs` for each point (`obs` is grouped by point).
    point_obs_range: Vec<(u32, u32)>,
    /// What [`Obs::camera`] indexes.
    cameras: Vec<ObsCamera>,

    /// Performance fix (see [`build_reduced_system_pattern`]'s doc): the
    /// distinct *free* frames observing each point, ascending by
    /// `frame_idx` — a property of the observation topology alone
    /// (independent of pose/point values, hence independent of the current
    /// LM linearization), computed once here rather than rediscovered by
    /// scanning Jacobian-evaluated observations on every accepted step.
    point_free_frames: Vec<Vec<u32>>,
    /// Where each point's blocks start in [`Linearization::hcp`]: the
    /// prefix sum of `point_free_frames` lengths (`n_points + 1` entries).
    hcp_off: Vec<usize>,
    /// Deduplicated, ascending `(free_slot_a, free_slot_b)` pairs (`a < b`)
    /// across every point's `point_free_frames` — the reduced camera
    /// system's off-diagonal sparsity pattern.
    edges: Vec<(u32, u32)>,
    /// Source of [`Problem::point_pair_edge_idx`].
    pairs: PairIndex,
    /// Performance fix #3 (see [`eliminate_and_accumulate`]'s doc): point
    /// index range `[shard_boundaries[s], shard_boundaries[s+1])` for shard
    /// `s`, balanced by *cumulative pair count* (`Σ n·(n-1)/2`), not point
    /// count — a real (corridor-revisit) track-length distribution is
    /// heavy-tailed, so an equal-point-count split can leave one shard
    /// doing most of the O(pairs) work while the others idle. Computed once
    /// here (pair counts are fixed for the whole `optimize()` call) from
    /// `point_free_frames`, reused unchanged by every trial and every
    /// accepted step.
    shard_boundaries: Vec<usize>,
    /// Built for large problems — from [`ITERATIVE_MIN_FREE_FRAMES`] free
    /// frames or [`BY_OUTPUT_MIN_EDGES`] blocks, or when the memory cap
    /// leaves [`elimination_shard_count`] below 8: the elimination then runs
    /// in parallel over its outputs instead (see [`ContributionIndex`]),
    /// with the same arithmetic. On a 1,375-frame / 338k-block synthetic
    /// city problem that is 3.2x faster than 8 shards each zeroing and
    /// filling an `edges_len` buffer; on the few-hundred-frame Courthouse
    /// problems the shards stay faster.
    contributions: Option<ContributionIndex>,
    /// Frames proper (`poses.len()`); `free_frame_slot` and the frame
    /// indices in `point_free_frames` continue with the extra nodes.
    n_frames: usize,
    /// The refined blocks beyond the frames ([`ExtraBlocks`]); their nodes
    /// are the last `extras.len()` slots.
    extras: ExtraBlocks,
    /// Free frames from which the reduced system is solved iteratively
    /// ([`iterative_min_free_frames`]; tests lower it).
    iterative_min_free_frames: usize,
    /// Unique per problem: lets a [`ReducedSolveAccelerator`] keep what it
    /// derived from the structure across the trials of one solve.
    structure_id: u64,
    /// The free slot of each `point_free_frames` entry, flattened like
    /// [`Linearization::hcp`]; built only for a registered accelerator's
    /// point elimination (empty otherwise).
    point_nodes: Vec<u32>,
}

/// Source of [`Problem::structure_id`] and of the linearization ids given to
/// a [`ReducedSolveAccelerator`].
static NEXT_ACCELERATOR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_accelerator_id() -> u64 {
    NEXT_ACCELERATOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// Which points observe each free frame of the reduced camera system, in
/// ascending point order, plus where each frame's off-diagonal row starts
/// in `edges`. Lets [`eliminate_and_accumulate`] compute every row of the
/// reduced system independently, in parallel, with exactly the additions
/// (and their order) of the sharded loop.
struct ContributionIndex {
    /// `slot[slot_off[s]..slot_off[s + 1]]`: `(point, k)` for free slot `s`,
    /// `k` the position of `s`'s frame in `point_free_frames[point]`.
    slot_off: Vec<usize>,
    slot: Vec<[u32; 2]>,
    /// `edges[row_off[s]..row_off[s + 1]]` are the edges `(s, _)`.
    row_off: Vec<usize>,
    /// Shard of each point (the sharded loop's summation grouping).
    point_shard: Vec<u32>,
}

impl ContributionIndex {
    #[cfg(test)]
    fn new(
        point_free_frames: &[Vec<u32>],
        free_frame_slot: &[Option<u32>],
        edges: &[(u32, u32)],
        n_free: usize,
        shard_boundaries: &[usize],
    ) -> Self {
        Self::from_slots(
            slot_index(point_free_frames, free_frame_slot, n_free),
            point_free_frames.len(),
            edges,
            shard_boundaries,
        )
    }

    /// The index over the [`slot_index`] the pattern was built from.
    fn from_slots(
        (slot_off, slot): SlotIndex,
        n_points: usize,
        edges: &[(u32, u32)],
        shard_boundaries: &[usize],
    ) -> Self {
        let n_free = slot_off.len() - 1;
        let row_off = (0..=n_free)
            .map(|s| edges.partition_point(|&(a, _)| (a as usize) < s))
            .collect();
        let mut point_shard = vec![0u32; n_points];
        for s in 0..shard_boundaries.len().saturating_sub(1) {
            for v in &mut point_shard[shard_boundaries[s]..shard_boundaries[s + 1]] {
                *v = s as u32;
            }
        }
        Self {
            slot_off,
            slot,
            row_off,
            point_shard,
        }
    }
}

/// What [`Problem::point_pair_edge_idx`] is built from, and the index once
/// built. Only the sharded elimination uses the index, and only problems
/// without a [`ContributionIndex`] build it: the elimination by output and
/// the direct terms find their blocks from the rows of `edges` instead (the
/// index is O(pairs), 1.9 GB on SmallCity's last bundle adjustment).
struct PairIndex {
    /// `edges[row_off[a]..row_off[a + 1]]` are the edges `(a, _)`.
    row_off: Vec<usize>,
    rank: Option<EdgeRank>,
    idx: std::sync::OnceLock<Vec<Vec<u32>>>,
}

impl Problem {
    /// Per point, the `edges`-index of each of its
    /// `point_free_frames[p].len() * (len-1) / 2` distinct frame pairs, in
    /// the order `for i in 0..n { for j in i + 1..n }` over
    /// `point_free_frames[p]` ([`pair_slot`]).
    fn point_pair_edge_idx(&self) -> &[Vec<u32>] {
        self.pairs.idx.get_or_init(|| {
            pair_edge_index(
                &self.point_free_frames,
                &self.free_frame_slot,
                &self.edges,
                &self.pairs.row_off,
                self.pairs.rank.as_ref(),
            )
        })
    }
}

/// [`Problem::point_pair_edge_idx`] from the pattern
/// [`build_reduced_system_pattern`] returns.
fn pair_edge_index(
    point_free_frames: &[Vec<u32>],
    free_frame_slot: &[Option<u32>],
    edges: &[(u32, u32)],
    row_off: &[usize],
    rank: Option<&EdgeRank>,
) -> Vec<Vec<u32>> {
    let slots_of = |frames: &[u32]| -> Vec<u32> {
        frames
            .iter()
            .map(|&f| free_frame_slot[f as usize].expect("free"))
            .collect()
    };
    point_free_frames
        .par_iter()
        .map(|frames| {
            if frames.len() < 2 {
                return Vec::new();
            }
            // Slots ascend with frames, so the pairs `(i, j > i)` walk
            // edge row `slots[i]` forwards.
            let slots = slots_of(frames);
            let n = slots.len();
            let mut idxs = vec![0u32; n * (n - 1) / 2];
            for i in 0..n {
                let a = slots[i] as usize;
                if let Some(rank) = rank {
                    for j in i + 1..n {
                        let e = row_off[a] + rank.position(a, slots[j]);
                        debug_assert_eq!(edges[e], (slots[i], slots[j]), "pair is an edge");
                        idxs[pair_slot(n, i, j)] = e as u32;
                    }
                    continue;
                }
                let row = &edges[row_off[a]..row_off[a + 1]];
                let mut pos = 0usize;
                for j in i + 1..n {
                    pos += row[pos..].partition_point(|&(_, b)| b < slots[j]);
                    debug_assert_eq!(row[pos], (slots[i], slots[j]), "pair is an edge");
                    idxs[pair_slot(n, i, j)] = (row_off[a] + pos) as u32;
                }
            }
            idxs
        })
        .collect()
}

/// Performance fix (profiled on a 300-frame/2500-point/289k-observation
/// synthetic problem sized like the tier-1000 global BA call that measured
/// 2.9s/iteration): the reduced camera system's off-diagonal blocks come
/// from O(track_len²) frame pairs per point — real tracks (corridor
/// revisits) run ~70 observations/point, so this is millions of pairs
/// total. Two things had to change from a first pass that computed this
/// per accepted LM step (still O(pairs) but only recomputed a few times
/// instead of every trial): (1) it must be a `BTreeMap`-free flat-array
/// merge (`solve_step`'s hot path), and (2) — the fix here — the pair
/// *pattern* itself (which frames co-observe which point) is a property of
/// the **observation topology alone**, not of the current pose/point
/// values, so it is computed **once per `optimize()` call**, not once per
/// accepted step: re-deriving and re-sorting ~6M pairs on every accept
/// (even only ~8 times in the profiling run) still cost ~200ms each. Using
/// `problem.obs`/`point_obs_range` (raw observation topology, ignoring the
/// z<=0 cheirality guard `residual_and_jacobians` applies — a frame/point
/// pair failing that check is a numerical edge case, not a topology
/// change) instead of `PointLin.frames` (Jacobian-evaluation output)
/// removes the coupling to relinearization entirely.
#[allow(clippy::type_complexity)]
fn build_reduced_system_pattern(
    point_obs_range: &[(u32, u32)],
    (obs, cameras): (&[Obs], &[ObsCamera]),
    frame_fixed: &[bool],
    free_frame_slot: &[Option<u32>],
    (n_frames, has_camera): (u32, bool),
) -> (Vec<Vec<u32>>, Vec<(u32, u32)>, PairIndex, SlotIndex) {
    // Per-point work runs on rayon; the outputs (sorted unique edge list,
    // per-point free frames and pair -> edge indices) are pure functions of
    // the input, so they match the serial construction exactly.
    let point_free_frames: Vec<Vec<u32>> = point_obs_range
        .par_iter()
        .map(|&(start, end)| {
            let mut frames: Vec<u32> = obs[start as usize..end as usize]
                .iter()
                .flat_map(|o| obs_nodes(o, cameras, n_frames, has_camera))
                .filter(|&f| !frame_fixed[f as usize])
                .collect();
            frames.sort_unstable();
            frames.dedup();
            frames
        })
        .collect();
    let n_slots = free_frame_slot
        .iter()
        .flatten()
        .map(|&s| s as usize + 1)
        .max()
        .unwrap_or(0);
    // The edges row by row: row `a` holds the slots `b > a` that share a
    // point with `a`, sorted and deduplicated — the sorted unique pair list
    // without materializing every pair. Each row gathers from the points
    // observing its frame (`(point, position)` per slot).
    let slots = slot_index(&point_free_frames, free_frame_slot, n_slots);
    let (slot_off, slot_points) = (&slots.0, &slots.1);
    // Up to `EDGE_RANK_MAX_SLOTS` frames the rows are bitmaps over the
    // slots ([`EdgeRank`]): marking every pair's bit leaves each row sorted
    // and unique without a sort, and a pair's edge index below is its row's
    // offset plus one popcount. Past that, each row gathers and sorts its
    // partners, and the pairs search their rows.
    let (row_off, edges, rank) = if n_slots <= EDGE_RANK_MAX_SLOTS {
        let rank = EdgeRank::new(n_slots, |a, bits| {
            for &[p, k] in &slot_points[slot_off[a]..slot_off[a + 1]] {
                for &f in &point_free_frames[p as usize][k as usize + 1..] {
                    let b = free_frame_slot[f as usize].expect("free");
                    bits[b as usize / 64] |= 1 << (b % 64);
                }
            }
        });
        let (row_off, edges) = rank.edges();
        (row_off, edges, Some(rank))
    } else {
        let rows: Vec<Vec<u32>> = (0..n_slots)
            .into_par_iter()
            .map(|a| {
                let mut row: Vec<u32> = Vec::new();
                for &[p, k] in &slot_points[slot_off[a]..slot_off[a + 1]] {
                    row.extend(
                        point_free_frames[p as usize][k as usize + 1..]
                            .iter()
                            .map(|&f| free_frame_slot[f as usize].expect("free")),
                    );
                }
                row.sort_unstable();
                row.dedup();
                row
            })
            .collect();
        let mut row_off = vec![0usize; n_slots + 1];
        for (a, row) in rows.iter().enumerate() {
            row_off[a + 1] = row_off[a] + row.len();
        }
        let mut edges: Vec<(u32, u32)> = Vec::with_capacity(row_off[n_slots]);
        for (a, row) in rows.iter().enumerate() {
            edges.extend(row.iter().map(|&b| (a as u32, b)));
        }
        (row_off, edges, None)
    };

    (
        point_free_frames,
        edges,
        PairIndex {
            row_off,
            rank,
            idx: std::sync::OnceLock::new(),
        },
        slots,
    )
}

/// Per free slot `s`, `(point, k)` for every point observing its frame,
/// ascending in point (`k` the frame's position in
/// `point_free_frames[point]`): `slot[slot_off[s]..slot_off[s + 1]]`, as
/// `(slot_off, slot)`.
type SlotIndex = (Vec<usize>, Vec<[u32; 2]>);

/// [`SlotIndex`] of `point_free_frames` over `n_slots` free slots. Large
/// inputs are grouped by slot in contiguous point chunks in parallel, then
/// each slot's runs are concatenated in chunk order: the layout of a serial
/// scatter in point order, at any thread count.
fn slot_index(
    point_free_frames: &[Vec<u32>],
    free_frame_slot: &[Option<u32>],
    n_slots: usize,
) -> SlotIndex {
    let scatter = |points: std::ops::Range<usize>| -> SlotIndex {
        let slot_of = |f: u32| free_frame_slot[f as usize].expect("free") as usize;
        let mut off = vec![0usize; n_slots + 1];
        for frames in &point_free_frames[points.clone()] {
            for &f in frames {
                off[slot_of(f) + 1] += 1;
            }
        }
        for i in 1..off.len() {
            off[i] += off[i - 1];
        }
        let mut out = vec![[0u32; 2]; off[n_slots]];
        let mut fill = off.clone();
        for p in points {
            for (k, &f) in point_free_frames[p].iter().enumerate() {
                let s = slot_of(f);
                out[fill[s]] = [p as u32, k as u32];
                fill[s] += 1;
            }
        }
        (off, out)
    };
    let n_points = point_free_frames.len();
    if n_points < 1 << 15 {
        return scatter(0..n_points);
    }
    const CHUNKS: usize = 64;
    let local: Vec<SlotIndex> = (0..CHUNKS)
        .into_par_iter()
        .map(|c| scatter(c * n_points / CHUNKS..(c + 1) * n_points / CHUNKS))
        .collect();
    let mut slot_off = vec![0usize; n_slots + 1];
    for s in 0..n_slots {
        let len: usize = local.iter().map(|(off, _)| off[s + 1] - off[s]).sum();
        slot_off[s + 1] = slot_off[s] + len;
    }
    let mut slot = vec![[0u32; 2]; slot_off[n_slots]];
    let mut rows: Vec<&mut [[u32; 2]]> = Vec::with_capacity(n_slots);
    let mut rest = slot.as_mut_slice();
    for s in 0..n_slots {
        let (row, tail) = rest.split_at_mut(slot_off[s + 1] - slot_off[s]);
        rows.push(row);
        rest = tail;
    }
    rows.into_par_iter().enumerate().for_each(|(s, row)| {
        let mut at = 0;
        for (off, out) in &local {
            let run = &out[off[s]..off[s + 1]];
            row[at..at + run.len()].copy_from_slice(run);
            at += run.len();
        }
    });
    (slot_off, slot)
}

/// Frame slots up to which [`EdgeRank`] indexes the edge rows: its bitmaps
/// take `n² / 8` bytes and their counts half that again (6 MB at SmallCity's
/// 5,800 frames, 48 MB here).
const EDGE_RANK_MAX_SLOTS: usize = 16_384;

/// The reduced camera system's edge rows as bitmaps over the frame slots,
/// with each 64-bit word's count of the row's earlier columns, so a column's
/// position in its row is one popcount. Building a point's pair -> edge
/// indices by searching each row instead cost most of a large global BA's
/// setup (tracks of tens of frames have hundreds of pairs each).
struct EdgeRank {
    words: usize,
    bits: Vec<u64>,
    before: Vec<u32>,
}

impl EdgeRank {
    /// `mark(a, bits)` sets row `a`'s bits: the slots `b > a` sharing a
    /// point with `a`.
    fn new(n_slots: usize, mark: impl Fn(usize, &mut [u64]) + Sync) -> Self {
        let words = n_slots.div_ceil(64).max(1);
        let mut bits = vec![0u64; n_slots * words];
        let mut before = vec![0u32; n_slots * words];
        bits.par_chunks_mut(words)
            .zip(before.par_chunks_mut(words))
            .enumerate()
            .for_each(|(a, (bits, before))| {
                mark(a, bits);
                let mut count = 0u32;
                for (word, before) in bits.iter().zip(before.iter_mut()) {
                    *before = count;
                    count += word.count_ones();
                }
            });
        Self {
            words,
            bits,
            before,
        }
    }

    /// The row offsets and the edges `(a, b)`, ascending.
    fn edges(&self) -> (Vec<usize>, Vec<(u32, u32)>) {
        let n_rows = self.bits.len() / self.words;
        let mut row_off = vec![0usize; n_rows + 1];
        for a in 0..n_rows {
            let last = (a + 1) * self.words - 1;
            row_off[a + 1] =
                row_off[a] + self.before[last] as usize + self.bits[last].count_ones() as usize;
        }
        let mut edges = vec![(0u32, 0u32); row_off[n_rows]];
        let mut rows: Vec<&mut [(u32, u32)]> = Vec::with_capacity(n_rows);
        let mut rest = edges.as_mut_slice();
        for a in 0..n_rows {
            let (row, tail) = rest.split_at_mut(row_off[a + 1] - row_off[a]);
            rows.push(row);
            rest = tail;
        }
        rows.into_par_iter().enumerate().for_each(|(a, row)| {
            let mut out = row.iter_mut();
            let words = &self.bits[a * self.words..(a + 1) * self.words];
            for (w, &word) in words.iter().enumerate() {
                let mut word = word;
                while word != 0 {
                    let b = (w * 64) as u32 + word.trailing_zeros();
                    *out.next().expect("counted") = (a as u32, b);
                    word &= word - 1;
                }
            }
        });
        (row_off, edges)
    }

    /// The position of column `b` in row `a`: the row's columns below `b`.
    #[inline]
    fn position(&self, a: usize, b: u32) -> usize {
        let w = a * self.words + b as usize / 64;
        let below = self.bits[w] & ((1u64 << (b % 64)) - 1);
        self.before[w] as usize + below.count_ones() as usize
    }
}

/// Position of the pair `(i, j)`, `i < j < n`, among a point's `n (n - 1) / 2`
/// frame pairs ordered `for i in 0..n { for j in i + 1..n }`: one frame's
/// pairs with the frames after it are contiguous, in the order the sharded
/// elimination walks them.
#[inline]
fn pair_slot(n: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j && j < n);
    i * (2 * n - i - 1) / 2 + (j - i - 1)
}

/// Most point shards a linearization is split into ([`linearize`]); a
/// problem has one per 2,048 points up to this.
const LINEARIZE_SHARDS: usize = 64;

/// Reduced camera systems with at least this many free frames are solved
/// iteratively ([`pcg_blocks6`]) instead of by block Cholesky — COLMAP's own
/// switch from Ceres' `SPARSE_SCHUR` to `ITERATIVE_SCHUR` past 1,000 images.
/// The factor of a dense covisibility graph (e.g. a city block revisited in
/// several passes) costs seconds to tens of seconds per LM step at that
/// size, while a preconditioned CG solve stays linear in the number of
/// camera pairs.
const ITERATIVE_MIN_FREE_FRAMES: usize = 1000;
/// [`ITERATIVE_MIN_FREE_FRAMES`], overridable with
/// `VISLOC_PORT_ITERATIVE_MIN_FRAMES` for diagnostics / A-B runs.
fn iterative_min_free_frames() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("VISLOC_PORT_ITERATIVE_MIN_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ITERATIVE_MIN_FREE_FRAMES)
    })
}
/// Ceres-style `function_tolerance`: an LM solve stops once an accepted
/// step lowers the cost by less than this fraction of it. COLMAP configures
/// Ceres with `function_tolerance=0`, so this is a deliberate deviation:
/// on SmallCity (5,583 images) several global BAs crept on to the
/// 50-iteration cap at ~1e-6 relative decrease per step, about a third of
/// the mapper's time, for no visible change. Overridable with
/// `VISLOC_PORT_BA_FUNCTION_TOLERANCE` (`0` disables the stop).
///
/// A solve with the absolute gradient criterion (the mapper's final
/// converged BA) stops only when the model's predicted decrease is that small
/// as well. On SmallCity (mapper seed 2) the final BA otherwise stopped after
/// a step that lowered the cost by 4% of its predicted decrease, at 0.711 m
/// median centre error, while the solve run on reaches 0.650 m.
const FUNCTION_TOLERANCE: f64 = 1e-5;
fn function_tolerance() -> f64 {
    static TOL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *TOL.get_or_init(|| {
        std::env::var("VISLOC_PORT_BA_FUNCTION_TOLERANCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(FUNCTION_TOLERANCE)
    })
}
/// Ceres' `max_num_consecutive_invalid_steps`: a step whose linear solve
/// fails is rejected (the trust region shrinks) and the solve goes on;
/// only this many failures in a row end it with an error.
const MAX_CONSECUTIVE_INVALID_STEPS: usize = 5;
/// Relative residual at which the bundle-adjustment PCG solve stops (`‖r‖ ≤ tol · ‖b‖`).
/// An inexact LM step is fine (the trust region absorbs it); Ceres'
/// `ITERATIVE_SCHUR` default is a far looser `eta = 0.1`. On a SmallCity
/// mapper replay (5.6k images in the main model) 1e-2 needs 190 PCG
/// iterations per global solve instead of 1e-3's 356 and the mapper takes
/// 1,184 s instead of 1,686 s, with 5,599 images registered instead of
/// 5,606 and a median camera-centre difference of 0.2 % of the scene
/// extent. That is within the run-to-run spread of other solver changes
/// (3e-3: 5,581 images, 0.28 %). With the iterative solve forced on from
/// 100 free frames, a 553-image Courthouse replay at 1e-3 keeps all 543
/// images within 0.08 % of the block-Cholesky run, and at 1e-2 drops one
/// image and is within 0.6 %. Without that override Courthouse never
/// reaches the iterative solve.
const PCG_RELATIVE_TOLERANCE: f64 = 1e-2;
/// [`PCG_RELATIVE_TOLERANCE`], overridable with `VISLOC_PORT_PCG_TOL`
/// (e.g. `1e-3`, the previous default) for A-B runs.
fn pcg_relative_tolerance() -> f64 {
    static TOL: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *TOL.get_or_init(|| {
        std::env::var("VISLOC_PORT_PCG_TOL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(PCG_RELATIVE_TOLERANCE)
    })
}
/// Edge chunks of the bundle-adjustment PCG product, each with its own
/// accumulator; fixed so the summation order does not depend on the thread
/// count.
const PCG_MATVEC_CHUNKS: usize = 64;
/// Whether the bundle-adjustment PCG product reads its off-diagonal blocks
/// in f32 (the default) instead of f64; `VISLOC_PORT_PCG_F64_BLOCKS=1`
/// keeps f64. The product is memory-bound, so halving the bytes nearly
/// halves its time. The diagonal, the preconditioner and every vector stay
/// f64, and the per-entry rounding (~6e-8 relative) is far below the 1e-2
/// residual the solve stops at. On a SmallCity mapper replay a PCG
/// iteration took 5.6 ms instead of 11.8 ms. Iteration counts matched
/// while the runs stayed on the same path, and the mapper took 836 s
/// instead of 1,122 s for the f64 product that read both triangles. It
/// registered 5,581 images instead of 5,599, with a median camera-centre
/// difference of 0.19 % of the scene extent.
fn pcg_f32_blocks() -> bool {
    static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F32.get_or_init(|| std::env::var("VISLOC_PORT_PCG_F64_BLOCKS").map_or(true, |v| v != "1"))
}
/// Free frames per cluster of the edge-list PCG's preconditioner
/// ([`cluster_preconditioner`]); `VISLOC_PORT_PCG_CLUSTER=k` overrides it,
/// `1` restores block-Jacobi. Frames are in frame-id (capture) order, and
/// neighbours in capture order share most points, so a dense inverse over
/// each run of consecutive frames captures the strongest couplings that
/// block-Jacobi drops. On a Courthouse replay forced onto PCG, the same
/// systems took 62 iterations a solve instead of 114 (k=16: 70, k=64: 60).
/// Each frame's run inverse adds `6 x 6k` values to a product that reads
/// `edges / frames` 6x6 blocks per frame (~390 on SmallCity), so the
/// per-iteration cost barely moves.
const PCG_CLUSTER_FRAMES: usize = 32;
fn pcg_cluster_size() -> Option<usize> {
    static K: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        Some(
            std::env::var("VISLOC_PORT_PCG_CLUSTER")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(PCG_CLUSTER_FRAMES),
        )
        .filter(|&k| k > 1)
    })
}

/// Inverses of the reduced camera matrix restricted to runs of `k`
/// consecutive free frames (diagonal blocks plus the edges inside a run),
/// as `(first frame, inverse)`. `None` when a run is not positive definite;
/// the caller then falls back to block-Jacobi.
fn cluster_preconditioner(
    diag: &[Matrix6<f64>],
    edges: &[(u32, u32)],
    offdiag: &[Matrix6<f64>],
    k: usize,
) -> Option<Vec<(usize, DMatrix<f64>)>> {
    cluster_inverses(diag, k, edges.iter().copied().zip(offdiag))
}

/// [`cluster_preconditioner`] from `((a, b), block)` pairs (blocks of edges
/// outside the runs are ignored).
fn cluster_inverses<'a>(
    diag: &[Matrix6<f64>],
    k: usize,
    blocks: impl Iterator<Item = ((u32, u32), &'a Matrix6<f64>)>,
) -> Option<Vec<(usize, DMatrix<f64>)>> {
    let n = diag.len();
    let starts: Vec<usize> = (0..n).step_by(k).collect();
    let mut mats: Vec<DMatrix<f64>> = starts
        .iter()
        .map(|&s| {
            let m = k.min(n - s);
            let mut mat = DMatrix::<f64>::zeros(m * 6, m * 6);
            for i in 0..m {
                mat.view_mut((i * 6, i * 6), (6, 6)).copy_from(&diag[s + i]);
            }
            mat
        })
        .collect();
    for ((a, b), blk) in blocks {
        let (a, b) = (a as usize, b as usize);
        if a / k != b / k {
            continue;
        }
        let (c, ia, ib) = (a / k, a % k, b % k);
        // `blk` is the (row b, column a) block.
        mats[c].view_mut((ib * 6, ia * 6), (6, 6)).copy_from(blk);
        mats[c]
            .view_mut((ia * 6, ib * 6), (6, 6))
            .copy_from(&blk.transpose());
    }
    starts
        .into_par_iter()
        .zip(mats)
        .map(|(s, mat)| mat.cholesky().map(|c| (s, c.inverse())))
        .collect()
}

/// Free frames per aggregate of the PCG's coarse correction
/// ([`coarse_correction`]); `VISLOC_PORT_PCG_COARSE=g` overrides it, `0`
/// or `1` turns the correction off. Larger problems use larger aggregates,
/// at most [`PCG_COARSE_MAX_AGGREGATES`] of them.
const PCG_COARSE_FRAMES: usize = 32;
/// Cap on the coarse level's aggregates: its dense factor (up to 1,536
/// rows) then costs at most ~0.2 s a solve and its triangular solves ~1 ms
/// a PCG iteration. SmallCity's 5,816 frames stay at 32 a group (1,092
/// rows: 55 ms and 0.5 ms, against ~26 ms for one PCG product).
const PCG_COARSE_MAX_AGGREGATES: usize = 256;
/// The coarse correction applies once the LM damping `mu = 1/radius` is at
/// most this, from the fourth accepted step of a solve on (Ceres' initial
/// radius 1e4, tripling per good step). Heavier damping holds the smooth
/// modes down by itself; the one-to-three-step global BAs of the
/// incremental loop then solve exactly as without it. On SmallCity,
/// applying it from the first step made those BAs run up to 13 steps
/// (their relative gradient stop rarely fired) and changed the
/// reconstruction's path.
const PCG_COARSE_MAX_MU: f64 = 1e-5;
fn pcg_coarse_size() -> Option<usize> {
    static G: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        Some(
            std::env::var("VISLOC_PORT_PCG_COARSE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(PCG_COARSE_FRAMES),
        )
        .filter(|&g| g > 1)
    })
}

/// The coarse level of the PCG preconditioner: the reduced camera matrix
/// restricted to one shared 6-vector step per aggregate of `group`
/// consecutive frames (`Pᵀ A P`, `P` stacking identity blocks), factored.
/// A frame's step `ξ` acts as `pose ∘ exp(ξ)`, a world-frame motion, so a
/// shared step moves the aggregate rigidly. Smooth deformations of a long
/// capture, such as scale drifting along a street, are close to rigid at
/// that scale; they are the low-curvature modes the cluster level cannot
/// see and conjugate gradients converge on slowest.
struct CoarseCorrection {
    group: usize,
    /// Leading free frames covered; the extra nodes after them are not.
    n_frames: usize,
    chol: nalgebra::Cholesky<f64, nalgebra::Dyn>,
}

impl CoarseCorrection {
    /// `z += P (Pᵀ A P)⁻¹ Pᵀ r`.
    fn add_to(&self, r: &[Vector6<f64>], z: &mut [Vector6<f64>]) {
        let m = self.chol.l_dirty().nrows();
        let mut rc = DVector::<f64>::zeros(m);
        for (i, ri) in r.iter().enumerate().take(self.n_frames) {
            let g = i / self.group * 6;
            for k in 0..6 {
                rc[g + k] += ri[k];
            }
        }
        let yc = self.chol.solve(&rc);
        for (i, zi) in z.iter_mut().enumerate().take(self.n_frames) {
            let g = i / self.group * 6;
            for k in 0..6 {
                zi[k] += yc[g + k];
            }
        }
    }

    /// `(Pᵀ A P)⁻¹`, for an accelerator that applies the coarse level as a
    /// dense product (column solves in parallel).
    fn inverse(&self) -> DMatrix<f64> {
        let m = self.chol.l_dirty().nrows();
        let columns: Vec<DVector<f64>> = (0..m)
            .into_par_iter()
            .map(|j| {
                let mut e = DVector::<f64>::zeros(m);
                e[j] = 1.0;
                self.chol.solve(&e)
            })
            .collect();
        DMatrix::from_columns(&columns)
    }
}

/// [`CoarseCorrection`] over the first `n_frames` free frames, or `None`
/// for fewer than two aggregates or a coarse matrix that is not positive
/// definite.
fn coarse_correction(
    diag: &[Matrix6<f64>],
    edges: &[(u32, u32)],
    offdiag: &[Matrix6<f64>],
    n_frames: usize,
    group: usize,
) -> Option<CoarseCorrection> {
    let group = group.max(n_frames.div_ceil(PCG_COARSE_MAX_AGGREGATES));
    let m = n_frames.div_ceil(group);
    if m < 2 {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(m * 6, m * 6);
    for (i, d) in diag.iter().enumerate().take(n_frames) {
        let g = i / group * 6;
        let mut view = a.view_mut((g, g), (6, 6));
        view += d;
    }
    for (&(ea, eb), blk) in edges.iter().zip(offdiag) {
        let (ea, eb) = (ea as usize, eb as usize);
        if ea >= n_frames || eb >= n_frames {
            continue;
        }
        let (ga, gb) = (ea / group * 6, eb / group * 6);
        // `blk` is the (row b, column a) block.
        let mut view = a.view_mut((gb, ga), (6, 6));
        view += blk;
        let mut view = a.view_mut((ga, gb), (6, 6));
        view += blk.transpose();
    }
    a.cholesky().map(|chol| CoarseCorrection {
        group,
        n_frames,
        chol,
    })
}

/// [`coarse_correction`] from per-aggregate-pair block sums
/// (`sums[g * m + h]`, `g <= h`: the blocks of edges from aggregate `g` to
/// `h`), as [`crate::ba_accel::EliminatedBlocks::coarse_sums`] returns them;
/// `group` is already [`coarse_correction`]'s adjusted aggregate size.
fn coarse_from_sums(
    diag: &[Matrix6<f64>],
    sums: &[Matrix6<f64>],
    n_frames: usize,
    group: usize,
) -> Option<CoarseCorrection> {
    let m = n_frames.div_ceil(group);
    if m < 2 || sums.len() != m * m {
        return None;
    }
    let mut a = DMatrix::<f64>::zeros(m * 6, m * 6);
    for (i, d) in diag.iter().enumerate().take(n_frames) {
        let g = i / group * 6;
        let mut view = a.view_mut((g, g), (6, 6));
        view += d;
    }
    for g in 0..m {
        for h in g..m {
            let sum = &sums[g * m + h];
            // `sum` adds to the (row h, column g) block, its transpose to
            // the (row g, column h) block.
            let mut view = a.view_mut((h * 6, g * 6), (6, 6));
            view += sum;
            let mut view = a.view_mut((g * 6, h * 6), (6, 6));
            view += sum.transpose();
        }
    }
    a.cholesky().map(|chol| CoarseCorrection {
        group,
        n_frames,
        chol,
    })
}

/// One off-diagonal block's share of the PCG product: `(Bᵀ x_b, B x_a)`
/// for the column-major (row b, column a) block `B`, read once, in f64
/// whatever the block's storage type.
#[inline(always)]
fn edge_product<T: Copy + Into<f64>>(
    blk: &[T],
    xa: &Vector6<f64>,
    xb: &Vector6<f64>,
) -> (Vector6<f64>, Vector6<f64>) {
    let mut to_a = [0.0f64; 6];
    let mut to_b = [0.0f64; 6];
    for (c, column) in blk.chunks_exact(6).enumerate() {
        let xac = xa[c];
        let mut dot = 0.0f64;
        for r in 0..6 {
            let v: f64 = column[r].into();
            to_b[r] = v.mul_add(xac, to_b[r]);
            dot = v.mul_add(xb[r], dot);
        }
        to_a[c] = dot;
    }
    (Vector6::from(to_a), Vector6::from(to_b))
}

/// Iteration cap of [`pcg_blocks6`] (Ceres' `max_linear_solver_iterations`).
const PCG_MAX_ITERATIONS: usize = 500;

/// Solve the SPD block system given as lower block columns (`columns[j]`
/// maps block row `i >= j` to `A_ij`, diagonal included) for the single
/// right-hand side `rhs` to relative residual `tol` by conjugate gradients
/// with a block-Jacobi
/// (inverse 6×6 diagonal) preconditioner — the camera-block analogue of
/// Ceres' `SCHUR_JACOBI`. The matrix-vector product runs on rayon over
/// symmetric block rows and every reduction is summed in block order, so
/// the result does not depend on the thread count. `None` when a diagonal
/// block is not positive definite or the iteration breaks down.
fn pcg_blocks6(
    columns: &[BTreeMap<usize, Matrix6<f64>>],
    rhs: &DMatrix<f64>,
    tol: f64,
) -> Option<DMatrix<f64>> {
    let n = columns.len();
    // Symmetric block rows: row i holds (j, A_ij) for every stored block,
    // ascending by j (blocks left of the diagonal come from earlier
    // columns, then the diagonal and the rest of column i).
    let mut rows: Vec<Vec<(usize, Matrix6<f64>)>> = vec![Vec::new(); n];
    for (j, col) in columns.iter().enumerate() {
        for (&i, blk) in col {
            rows[i].push((j, *blk));
            if i != j {
                rows[j].push((i, blk.transpose()));
            }
        }
    }
    let mut row_off = Vec::with_capacity(n + 1);
    row_off.push(0);
    for row in &rows {
        row_off.push(row_off.last().copied().unwrap_or(0) + row.len());
    }
    let entries: Vec<(usize, Matrix6<f64>)> = rows.into_iter().flatten().collect();
    pcg_csr6(&row_off, &entries, rhs, tol)
}

/// [`pcg_blocks6`] on the symmetric block rows in CSR form: row `i` is
/// `entries[row_off[i]..row_off[i + 1]]`, `(column, A_ij)` ascending by
/// column with the diagonal block included.
fn pcg_csr6(
    row_off: &[usize],
    entries: &[(usize, Matrix6<f64>)],
    rhs: &DMatrix<f64>,
    tol: f64,
) -> Option<DMatrix<f64>> {
    let n = row_off.len() - 1;
    let diag: Vec<Matrix6<f64>> = (0..n)
        .map(|i| {
            entries[row_off[i]..row_off[i + 1]]
                .iter()
                .find(|(j, _)| *j == i)
                .map(|(_, d)| *d)
        })
        .collect::<Option<Vec<_>>>()?;
    let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                entries[row_off[i]..row_off[i + 1]]
                    .iter()
                    .fold(Vector6::zeros(), |acc, (j, blk)| acc + blk * x[*j])
            })
            .collect()
    };
    pcg_core(&diag, matvec, rhs, tol, None, None).map(|(x, _)| x)
}

/// [`pcg_blocks6`] with the matrix given by its diagonal blocks and a
/// matrix-vector product.
fn pcg_core(
    diag: &[Matrix6<f64>],
    matvec: impl Fn(&[Vector6<f64>]) -> Vec<Vector6<f64>>,
    rhs: &DMatrix<f64>,
    tol: f64,
    clusters: Option<&[(usize, DMatrix<f64>)]>,
    coarse: Option<&CoarseCorrection>,
) -> Option<(DMatrix<f64>, usize)> {
    let n = diag.len();
    let precond: Vec<Matrix6<f64>> = diag
        .par_iter()
        .map(|d| d.cholesky().map(|c| c.inverse()))
        .collect::<Option<Vec<_>>>()?;
    // z = M⁻¹ r: block-Jacobi, or per-cluster dense inverses when given,
    // plus the coarse correction when given (additive two-level).
    let apply = |r: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
        let mut z: Vec<Vector6<f64>> = match clusters {
            None => r.iter().zip(&precond).map(|(r, m)| m * r).collect(),
            Some(clusters) => {
                let parts: Vec<Vec<Vector6<f64>>> = clusters
                    .par_iter()
                    .map(|(start, inv)| {
                        let m = inv.nrows() / 6;
                        let rc = DVector::<f64>::from_fn(m * 6, |k, _| r[start + k / 6][k % 6]);
                        let zc = inv * rc;
                        (0..m)
                            .map(|i| Vector6::from_fn(|k, _| zc[i * 6 + k]))
                            .collect()
                    })
                    .collect();
                parts.into_iter().flatten().collect()
            }
        };
        if let Some(coarse) = coarse {
            coarse.add_to(r, &mut z);
        }
        z
    };
    let dot = |a: &[Vector6<f64>], b: &[Vector6<f64>]| -> f64 {
        let partial: Vec<f64> = a.par_iter().zip(b).map(|(x, y)| x.dot(y)).collect();
        partial.iter().sum()
    };
    let b: Vec<Vector6<f64>> = (0..n)
        .map(|i| Vector6::from_fn(|k, _| rhs[(i * 6 + k, 0)]))
        .collect();
    let b_norm = dot(&b, &b).sqrt();
    let mut x = vec![Vector6::<f64>::zeros(); n];
    if b_norm == 0.0 {
        return Some((DMatrix::zeros(n * 6, 1), 0));
    }
    let mut r = b;
    let mut z: Vec<Vector6<f64>> = apply(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut iterations = 0;
    for _ in 0..PCG_MAX_ITERATIONS {
        iterations += 1;
        let ap = matvec(&p);
        let pap = dot(&p, &ap);
        if !pap.is_finite() || pap <= 0.0 {
            return None;
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        if dot(&r, &r).sqrt() <= tol * b_norm {
            break;
        }
        z = apply(&r);
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    let mut out = DMatrix::<f64>::zeros(n * 6, 1);
    for (i, xi) in x.iter().enumerate() {
        for k in 0..6 {
            out[(i * 6 + k, 0)] = xi[k];
        }
    }
    Some((out, iterations))
}

/// Reduced camera systems with at least this many free frames are factored
/// in a fill-reducing block order ([`crate::reordering`]); smaller ones (every
/// local BA) keep the natural order, where fill is negligible.
const REORDER_MIN_FREE_FRAMES: usize = 64;

/// One rig observation as [`build_problem`] reads it: a
/// [`BaRigObservation`] with the camera reduced to its pinhole intrinsics
/// (`None` for a model this solver does not support), so building one costs
/// no allocation. Global BAs carry millions of observations, and cloning a
/// `Camera` (a heap-allocated parameter vector) per observation, then
/// freeing it, cost as much as a third of the solve.
#[derive(Clone, Debug)]
pub(crate) struct RigObs {
    pub keyframe_id: u64,
    pub landmark_id: u64,
    pub xy: Point2<f64>,
    pub intrinsics: Option<(f64, f64, f64, f64)>,
    pub sensor_from_rig: SE3,
}

impl From<&BaRigObservation> for RigObs {
    fn from(o: &BaRigObservation) -> Self {
        Self {
            keyframe_id: o.keyframe_id,
            landmark_id: o.landmark_id,
            xy: o.xy,
            intrinsics: distortion_free_intrinsics(&o.camera),
            sensor_from_rig: o.sensor_from_rig.clone(),
        }
    }
}

/// The pinhole intrinsics of a camera without lens distortion; `None` for
/// any other, so a distorted sensor is reported as unsupported instead of
/// being solved as a pinhole.
fn distortion_free_intrinsics(camera: &Camera) -> Option<(f64, f64, f64, f64)> {
    let first_lens_param = match camera.model {
        CameraModel::Pinhole | CameraModel::SimplePinhole => return camera.intrinsics(),
        CameraModel::SimpleRadial | CameraModel::Radial => 3,
        CameraModel::OpenCv | CameraModel::FullOpenCv => 4,
        _ => return None,
    };
    camera
        .params
        .iter()
        .skip(first_lens_param)
        .all(|&value| value == 0.0)
        .then(|| camera.intrinsics())
        .flatten()
}

/// Frames and points of a solve as flat arrays, ids ascending: what
/// [`build_problem`] reads from a [`BundleAdjustment`]'s maps, so a caller
/// with the data in hand need not build those maps (on a global BA, a
/// `BTreeMap` of a million landmarks that is converted straight back).
/// [`optimize_rig_input`] leaves the optimized poses and points here.
pub(crate) struct RigBaInput {
    pub frame_ids: Vec<u64>,
    pub frame_fixed: Vec<bool>,
    pub poses: Vec<SE3>,
    pub point_ids: Vec<u64>,
    pub point_fixed: Vec<bool>,
    pub points: Vec<Point3<f64>>,
}

impl RigBaInput {
    /// The poses and landmarks of `ba` (its maps iterate ids ascending).
    fn from_ba(ba: &BundleAdjustment) -> Self {
        let frame_ids: Vec<u64> = ba.poses.keys().copied().collect();
        let point_ids: Vec<u64> = ba.landmarks.keys().copied().collect();
        Self {
            frame_fixed: frame_ids
                .iter()
                .map(|id| ba.fixed_poses.contains(id))
                .collect(),
            poses: ba
                .poses
                .values()
                .map(|p| p.world_to_camera.clone())
                .collect(),
            point_fixed: point_ids
                .iter()
                .map(|id| ba.fixed_landmarks.contains(id))
                .collect(),
            points: ba.landmarks.values().copied().collect(),
            frame_ids,
            point_ids,
        }
    }
}

/// The solver's problem from `input`'s frames and points and the
/// observations `rig_obs`.
fn build_problem(input: &RigBaInput, rig_obs: &[RigObs]) -> Result<Problem, BaError> {
    if input.frame_ids.is_empty() {
        return Err(BaError::NoPoses);
    }
    if input.point_ids.is_empty() {
        return Err(BaError::NoLandmarks);
    }
    if rig_obs.is_empty() {
        return Err(BaError::NoObservations);
    }
    if input.frame_fixed.iter().all(|&fixed| fixed) {
        return Err(BaError::AllPosesFixed);
    }

    let t_start = Instant::now();
    // Ids are ascending, so they map to indices by binary search.
    let frame_ids = &input.frame_ids;
    let point_ids = &input.point_ids;
    // (frame index, point index) per observation, in parallel; the first
    // failing observation (in input order) decides the error, as before.
    let indices: Vec<Result<(u32, u32), BaError>> = rig_obs
        .par_iter()
        .map(|o| {
            let frame_idx = frame_ids
                .binary_search(&o.keyframe_id)
                .map_err(|_| BaError::MissingPose(o.keyframe_id))?;
            let point_idx = point_ids
                .binary_search(&o.landmark_id)
                .map_err(|_| BaError::MissingLandmark(o.landmark_id))?;
            o.intrinsics.ok_or(BaError::UnsupportedCameraModel)?;
            Ok((frame_idx as u32, point_idx as u32))
        })
        .collect();
    let indices: Vec<(u32, u32)> = indices.into_iter().collect::<Result<_, _>>()?;
    // Each observation's camera, deduplicated bit for bit (consecutive
    // observations mostly share one, so the previous one is tried first).
    let mut table = ObsCameraTable::default();
    let mut previous: Option<([u64; 12], u32)> = None;
    let camera_of_obs: Vec<u32> = rig_obs
        .iter()
        .map(|o| {
            let camera = ObsCamera::new(
                o.intrinsics.expect("checked above"),
                o.sensor_from_rig.clone(),
            );
            let key = camera.key();
            match previous {
                Some((last, index)) if last == key => index,
                _ => {
                    let index = table.index_of(camera);
                    previous = Some((key, index));
                    index
                }
            }
        })
        .collect();

    // Stable counting sort by point index: observations of a point keep the
    // caller's (`rig_obs`) order, exactly like the stable
    // `sort_by_key` this replaces, without moving the large `Obs` records.
    let mut point_obs_range = vec![(0u32, 0u32); point_ids.len()];
    for &(_, p) in &indices {
        point_obs_range[p as usize].1 += 1;
    }
    let mut start = 0u32;
    for range in point_obs_range.iter_mut() {
        let count = range.1;
        *range = (start, start + count);
        start += count;
    }
    let mut order = vec![0u32; indices.len()];
    let mut next: Vec<u32> = point_obs_range.iter().map(|r| r.0).collect();
    for (i, &(_, p)) in indices.iter().enumerate() {
        order[next[p as usize] as usize] = i as u32;
        next[p as usize] += 1;
    }
    let obs: Vec<Obs> = order
        .par_iter()
        .map(|&i| {
            let (frame_idx, _) = indices[i as usize];
            Obs::new(frame_idx, rig_obs[i as usize].xy, camera_of_obs[i as usize])
        })
        .collect();
    let obs_ms = t_start.elapsed().as_secs_f64() * 1e3;
    assemble_problem(
        input,
        PointMajorObs {
            obs,
            point_obs_range,
            cameras: table.into_cameras(),
        },
        obs_ms,
        ExtraBlocks::default(),
    )
}

/// The solver's problem from `input`'s frames and points and observations
/// already grouped by point; `obs_ms` is the time spent grouping them, for
/// the `BUILD_PARTS` line.
fn assemble_problem(
    input: &RigBaInput,
    point_major: PointMajorObs,
    obs_ms: f64,
    extras: ExtraBlocks,
) -> Result<Problem, BaError> {
    if input.frame_ids.is_empty() {
        return Err(BaError::NoPoses);
    }
    if input.point_ids.is_empty() {
        return Err(BaError::NoLandmarks);
    }
    let PointMajorObs {
        obs,
        point_obs_range,
        cameras,
    } = point_major;
    if obs.is_empty() {
        return Err(BaError::NoObservations);
    }
    debug_assert_eq!(point_obs_range.len(), input.point_ids.len());

    let t_start = Instant::now();
    // The extra nodes follow the frames, always free.
    let n_frames = input.frame_ids.len();
    let mut frame_fixed: Vec<bool> = input.frame_fixed.clone();
    frame_fixed.resize(n_frames + extras.len(), false);
    let mut free_frame_slot = vec![None; frame_fixed.len()];
    let mut n_free_frames = 0usize;
    for (i, &fixed) in frame_fixed.iter().enumerate() {
        if !fixed {
            free_frame_slot[i] = Some(n_free_frames as u32);
            n_free_frames += 1;
        }
    }
    if n_free_frames == extras.len() {
        return Err(BaError::AllPosesFixed);
    }
    let poses: Vec<SE3> = input.poses.clone();
    let point_fixed: Vec<bool> = input.point_fixed.clone();
    let points: Vec<Point3<f64>> = input.points.clone();

    let setup_ms = t_start.elapsed().as_secs_f64() * 1e3;
    let (point_free_frames, edges, pairs, slots) = build_reduced_system_pattern(
        &point_obs_range,
        (&obs, &cameras),
        &frame_fixed,
        &free_frame_slot,
        (n_frames as u32, extras.camera.is_some()),
    );
    let pattern_ms = t_start.elapsed().as_secs_f64() * 1e3 - setup_ms;
    let n_shards = elimination_shard_count(edges.len());
    let shard_boundaries = balanced_shard_boundaries(&point_free_frames, n_shards);
    let contributions = (n_shards < 8
        || n_free_frames >= ITERATIVE_MIN_FREE_FRAMES
        || edges.len() >= BY_OUTPUT_MIN_EDGES)
        .then(|| {
            ContributionIndex::from_slots(slots, point_free_frames.len(), &edges, &shard_boundaries)
        });
    // The pair index serves the sharded elimination only; the elimination
    // by output finds each pair's block without it.
    if contributions.is_none() {
        let idx = pair_edge_index(
            &point_free_frames,
            &free_frame_slot,
            &edges,
            &pairs.row_off,
            pairs.rank.as_ref(),
        );
        pairs.idx.set(idx).expect("fresh index");
    }

    if n_free_frames >= 1000 {
        let total_ms = t_start.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "BUILD_PARTS setup_ms={setup_ms:.1} obs_ms={obs_ms:.1} pattern_ms={pattern_ms:.1} index_ms={:.1}",
            total_ms - setup_ms - pattern_ms
        );
    }

    let mut hcp_off = Vec::with_capacity(point_free_frames.len() + 1);
    hcp_off.push(0);
    for frames in &point_free_frames {
        hcp_off.push(hcp_off.last().copied().unwrap_or(0) + frames.len());
    }
    // Only the solves from `iterative_min_free_frames` hand the points to
    // an accelerator.
    let point_nodes: Vec<u32> = if contributions.is_some()
        && n_free_frames >= iterative_min_free_frames()
        && crate::ba_accel::reduced_solve_accelerator().is_some()
    {
        point_free_frames
            .iter()
            .flatten()
            .map(|&f| free_frame_slot[f as usize].expect("free"))
            .collect()
    } else {
        Vec::new()
    };

    Ok(Problem {
        free_frame_slot,
        n_free_frames,
        poses,
        point_fixed,
        points,
        obs,
        point_obs_range,
        cameras,
        point_free_frames,
        hcp_off,
        edges,
        pairs,
        shard_boundaries,
        contributions,
        n_frames,
        extras,
        iterative_min_free_frames: iterative_min_free_frames(),
        structure_id: next_accelerator_id(),
        point_nodes,
    })
}

#[inline]
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Residual `r = π(sensor_from_rig · pose · point_world) − xy` and its
/// Jacobians w.r.t. the pose's `[ρ; ω]` local update (`2×6`) and the point
/// (`2×3`). `None` when the point is behind the sensor (matches
/// `bundle.rs::cost()`'s "skip nonprojectable observations" convention — see
/// module doc). See module doc for the formula citation and
/// `tests::matches_bundle_rs_formula` / `tests::jacobian_matches_finite_difference`
/// for verification.
#[inline]
#[allow(clippy::too_many_arguments)]
fn residual_and_jacobians(
    fx: f64,
    fy: f64,
    cx: f64,
    cy: f64,
    sensor_from_rig: &SE3,
    pose: &SE3,
    point_world: &Point3<f64>,
    xy: Point2<f64>,
) -> Option<(Vector2<f64>, SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>)> {
    let point_rig = pose.transform_point(point_world);
    let point_sensor = sensor_from_rig.transform_point(&point_rig);
    let (predicted, projection) = project_sensor_point((fx, fy, cx, cy), &point_sensor)?;
    let residual = pixel_residual(fx, cx, predicted, xy);
    let r_wc = pose.rotation.to_rotation_matrix().into_inner();
    let r_rs = sensor_from_rig.rotation.to_rotation_matrix().into_inner();
    // d(rig point)/d(pose) = [R | -R [p]x], so the pose Jacobian is
    // [J_point | -J_point [p]x] with J_point = P R_rs R.
    let j_point = (projection * r_rs) * r_wc;
    let mut j_pose = SMatrix::<f64, 2, 6>::zeros();
    j_pose.fixed_view_mut::<2, 3>(0, 0).copy_from(&j_point);
    j_pose
        .fixed_view_mut::<2, 3>(0, 3)
        .copy_from(&-(j_point * skew(&point_world.coords)));
    Some((residual, j_pose, j_point))
}

/// `rho(s), rho'(s), rho''(s)` for one residual block's squared norm `s`.
/// Port of `ceres::{TrivialLoss,SoftLOneLoss,CauchyLoss}::Evaluate`
/// (`internal/ceres/loss_function.cc:46-50` (Trivial: `rho=[s,1,0]`),
/// `:68-75` (`SoftLOneLoss`), `:77-84` (`CauchyLoss`)), with the
/// scale-squaring constructors `b_ = a*a`, `c_ = 1/b_` from
/// `include/ceres/loss_function.h:190-217` (`a` = this module's `scale`).
/// `f64::MIN_POSITIVE` matches `std::numeric_limits<double>::min()`'s
/// smallest-positive-normal value used to clamp `rho[1]` away from zero.
#[inline]
fn evaluate_loss(loss: LossFunction, s: f64) -> [f64; 3] {
    match loss {
        LossFunction::Trivial => [s, 1.0, 0.0],
        LossFunction::SoftL1(a) => {
            let b = a * a;
            let c = 1.0 / b;
            let sum = 1.0 + s * c;
            let tmp = sum.sqrt();
            let rho0 = 2.0 * b * (tmp - 1.0);
            let rho1 = (1.0 / tmp).max(f64::MIN_POSITIVE);
            let rho2 = -(c * rho1) / (2.0 * sum);
            [rho0, rho1, rho2]
        }
        LossFunction::Cauchy(a) => {
            let b = a * a;
            let c = 1.0 / b;
            let sum = 1.0 + s * c;
            let inv = 1.0 / sum;
            let rho0 = b * sum.ln();
            let rho1 = inv.max(f64::MIN_POSITIVE);
            let rho2 = -c * (inv * inv);
            [rho0, rho1, rho2]
        }
    }
}

/// Direct port of `ceres::internal::Corrector` (`internal/ceres/corrector.cc:41-155`):
/// rescales one residual block's residual and Jacobian(s) so that a plain
/// Gauss-Newton/LM step on the *corrected* quantities reproduces Ceres'
/// robustified step (the BAMS — "Bundle Adjustment: A Modern Synthesis" —
/// second-order correction in the common case, falling back to a first-order
/// rescale when `rho'' >= 0` or `s == 0`, `corrector.cc:45-93`).
struct Corrector {
    /// Multiplies each Jacobian block (`corrector.cc:108,127,150-152`).
    sqrt_rho1: f64,
    /// Multiplies the residual (`corrector.cc:108,115`) — equals `sqrt_rho1`
    /// in the common (`rho'' <= 0`) case, differs only in the rarer
    /// `rho'' > 0` branch.
    residual_scaling: f64,
    /// `alpha / sq_norm` (`corrector.cc:109`); `0.0` selects the fast
    /// (no-curvature-correction) path in [`Corrector::correct_jacobian`].
    alpha_sq_norm: f64,
}

impl Corrector {
    /// Port of `Corrector::Corrector` (`corrector.cc:41-110`). `rho` is
    /// `[rho(s), rho'(s), rho''(s)]` for this residual block's squared norm
    /// `sq_norm`.
    fn new(sq_norm: f64, rho: [f64; 3]) -> Self {
        let sqrt_rho1 = rho[1].max(0.0).sqrt();
        // `corrector.cc:81-85`: the common case (and `sq_norm == 0`) skips
        // the second-order curvature correction entirely.
        if sq_norm == 0.0 || rho[2] <= 0.0 {
            return Self {
                sqrt_rho1,
                residual_scaling: sqrt_rho1,
                alpha_sq_norm: 0.0,
            };
        }
        // `corrector.cc:93-109`. `rho[1] > 0` is guaranteed here (Ceres
        // `CHECK_GT`; our loss evaluators clamp `rho[1]` to
        // `f64::MIN_POSITIVE`, never non-positive) so this division is safe.
        let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
        let alpha = 1.0 - d.sqrt();
        Self {
            sqrt_rho1,
            residual_scaling: sqrt_rho1 / (1.0 - alpha),
            alpha_sq_norm: alpha / sq_norm,
        }
    }

    /// Port of `Corrector::CorrectJacobian` (`corrector.cc:118-155`), applied
    /// to one parameter block's `2×C` Jacobian (`C` = 6 for the pose block,
    /// 3 for the point block — each corrected independently, both from the
    /// *same*, still-uncorrected `r`, matching `residual_block.cc:179-191`'s
    /// per-parameter-block loop that runs entirely before
    /// [`Corrector::correct_residual`]).
    #[inline]
    fn correct_jacobian<const C: usize>(&self, r: &Vector2<f64>, j: &mut SMatrix<f64, 2, C>) {
        if self.alpha_sq_norm == 0.0 {
            // `corrector.cc:126-129`.
            *j *= self.sqrt_rho1;
            return;
        }
        // `corrector.cc:131-154`: `J = sqrt_rho1 * (J - alpha_sq_norm * r * (rᵀJ))`.
        let r_transpose_j = r.transpose() * *j;
        *j = (*j - r * (self.alpha_sq_norm * r_transpose_j)) * self.sqrt_rho1;
    }

    /// Port of `Corrector::CorrectResiduals` (`corrector.cc:112-116`). Must
    /// be called *after* every [`Corrector::correct_jacobian`] call for this
    /// residual block (they need the uncorrected `r`).
    #[inline]
    fn correct_residual(&self, r: &mut Vector2<f64>) {
        *r *= self.residual_scaling;
    }
}

/// Per-point linearization output (Stage A — recomputed only after an
/// accepted step). `frames` are the point's *free* observing frames only
/// (ascending, deduplicated); fixed frames contribute to `hpp`/`bp` (they
/// still constrain the point) but never get a `diag`/`bc`/`hcp` slot (they
/// have no variable to receive one).
/// `f64`s per (point, free frame) block in [`linearize`]'s chunk buffer:
/// the frame's `JᵀJ` (6×6, column-major) then its `−Jᵀr`.
const FRAME_BLOCK: usize = 42;

struct PointLin {
    /// `JᵀJ` for the point block, only meaningful if `free`.
    hpp: Matrix3<f64>,
    /// `−Jᵀr` for the point block, only meaningful if `free`.
    bp: Vector3<f64>,
    free: bool,
}

/// Every point's linearization. The `Jpose^T Jpoint` blocks of all points
/// share one array: point `p`'s, one per `point_free_frames[p]` (only
/// meaningful if the point is free), are `hcp[problem.hcp_off[p]..
/// problem.hcp_off[p + 1]]`. One array instead of one vector per point
/// spares hundreds of thousands of allocations per linearization of a
/// global BA, and is handed back to the next linearization for reuse.
struct Linearization {
    points: Vec<PointLin>,
    hcp: Vec<SMatrix<f64, 6, 3>>,
}

impl Linearization {
    fn hcp(&self, problem: &Problem, p: usize) -> &[SMatrix<f64, 6, 3>] {
        &self.hcp[problem.hcp_off[p]..problem.hcp_off[p + 1]]
    }
}

/// `d(u, v)/d(sensor_from_rig)` for the update `sensor_from_rig ∘ exp(ξ)`,
/// `ξ = [ρ; ω]`: the sensor point is `R (p + ρ + ω × p) + t` to first order
/// for the rig point `p`.
fn sensor_jacobian(
    intrinsics: (f64, f64, f64, f64),
    sensor_from_rig: &SE3,
    point_rig: &Point3<f64>,
) -> SMatrix<f64, 2, 6> {
    let point_sensor = sensor_from_rig.transform_point(point_rig);
    // Only called for observations `residual_and_jacobians` projected.
    let Some((_, projection)) = project_sensor_point(intrinsics, &point_sensor) else {
        return SMatrix::zeros();
    };
    let r_sr = sensor_from_rig.rotation.to_rotation_matrix().into_inner();
    let mut d_sensor = SMatrix::<f64, 3, 6>::zeros();
    d_sensor.fixed_view_mut::<3, 3>(0, 0).copy_from(&r_sr);
    d_sensor
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-r_sr * skew(&point_rig.coords)));
    projection * d_sensor
}

/// Stage A: evaluate residual + Jacobians for every observation of one point
/// and fold them into that point's local blocks. Returns `(cost_contribution,
/// PointLin, direct)`; `cost_contribution = Σ rho(s_i)` (full, un-halved —
/// matches `BundleAdjustment::cost()`'s convention; `rho(s_i) = s_i =
/// ‖r_i‖²` for `LossFunction::Trivial`, so this is exactly the old "Σ ‖r_i‖²"
/// convention when the loss is trivial) over this point's projectable
/// observations, with `hpp`/`bp`/`bc`/`diag`/`hcp` folded from the
/// robust-loss-*corrected* `(r, j_pose, j_point)` (see
/// `evaluate_loss`/`Corrector` above) rather than the raw ones.
/// `target_frames` (== `problem.point_free_frames[p]`, ascending) fixes
/// `frames`/`bc`/`diag`/`hcp`'s slot layout up front — an observation's
/// frame is placed by `binary_search` (`target_frames` is sorted), not a
/// linear scan that grows the list — see [`build_reduced_system_pattern`]'s
/// doc for why the frame *set* itself is precomputed once per `optimize()`
/// call rather than rediscovered here.
///
/// With [`ExtraBlocks`], an observation also touches the camera and sensor
/// nodes (`target_frames` lists them after the frames); `direct` returns the
/// `J_bᵀ J_a` blocks between two nodes of one observation, keyed by the
/// point's local pair index `b(b-1)/2 + a` (positions `a < b` in
/// `target_frames`), for [`linearize`] to add to those edges.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn linearize_point(
    (obs_slice, cameras): (&[Obs], &[ObsCamera]),
    point_free: bool,
    point_xyz: &Point3<f64>,
    poses: &[SE3],
    target_frames: &[u32],
    loss: LossFunction,
    (extras, n_frames): (&ExtraBlocks, u32),
    frame_blocks: &mut [f64],
    hcp: &mut [SMatrix<f64, 6, 3>],
    full: bool,
) -> (f64, PointLin, Vec<([u32; 2], Matrix6<f64>)>) {
    frame_blocks.fill(0.0);
    hcp.fill(SMatrix::<f64, 6, 3>::zeros());
    let mut cost = 0.0;
    let mut hpp = Matrix3::zeros();
    let mut bp = Vector3::zeros();
    let mut direct: Vec<([u32; 2], Matrix6<f64>)> = Vec::new();
    let has_extras = extras.len() > 0;
    let camera_node = n_frames;
    let sensor_base = n_frames + u32::from(extras.camera.is_some());

    for o in obs_slice {
        let pose = &poses[o.frame_idx as usize];
        let ((fx, fy, cx, cy), sensor_from_rig) = extras.camera_of(o, cameras);
        let sensor = cameras[o.camera as usize].sensor;
        let Some((mut r, mut j_pose, mut j_point)) =
            residual_and_jacobians(fx, fy, cx, cy, sensor_from_rig, pose, point_xyz, o.xy)
        else {
            continue;
        };
        // The observation's extra nodes and their Jacobians (from the raw
        // residual): the camera's d(u, v)/d(focal, cx, cy) =
        // [[x/z, 1, 0], [y/z, 0, 1]] (u = r.x + x_obs = fx x/z + cx), the
        // sensor's [`sensor_jacobian`].
        let mut j_extra: [Option<(u32, SMatrix<f64, 2, 6>)>; 2] = [None, None];
        if has_extras {
            if let Some(c) = &extras.camera {
                let free = c.free();
                let pick = |i: usize, v: f64| if free[i] { v } else { 0.0 };
                let (xn, yn) = ((r.x + o.xy.x - cx) / fx, (r.y + o.xy.y - cy) / fy);
                let mut j = SMatrix::<f64, 2, 6>::zeros();
                j[(0, 0)] = pick(0, xn);
                j[(1, 0)] = pick(0, yn);
                j[(0, 1)] = pick(1, 1.0);
                j[(1, 2)] = pick(2, 1.0);
                j_extra[0] = Some((camera_node, j));
            }
            if sensor != NO_SENSOR {
                let point_rig = pose.transform_point(point_xyz);
                j_extra[1] = Some((
                    sensor_base + sensor,
                    sensor_jacobian((fx, fy, cx, cy), sensor_from_rig, &point_rig),
                ));
            }
        }
        // Robust-loss correction (`residual_block.cc:161-197`'s exact order:
        // evaluate rho at the *raw* squared norm, correct both Jacobian
        // blocks from the raw residual, *then* correct the residual itself)
        // — a no-op for `LossFunction::Trivial` (`rho=[s,1,0]` makes
        // `Corrector::new` take the `sqrt_rho1=1, alpha_sq_norm=0` fast
        // path), skipped outright here to avoid the `Corrector`
        // construction/multiply overhead on the by-far most common case
        // (global BA is always `Trivial`; local BA is `Trivial` after its
        // first iteration too, see `mapper.rs::iterative_local_refinement`).
        let rho0 = if matches!(loss, LossFunction::Trivial) {
            r.norm_squared()
        } else {
            let s = r.norm_squared();
            let rho = evaluate_loss(loss, s);
            let corrector = Corrector::new(s, rho);
            corrector.correct_jacobian(&r, &mut j_pose);
            corrector.correct_jacobian(&r, &mut j_point);
            for (_, j) in j_extra.iter_mut().flatten() {
                corrector.correct_jacobian(&r, j);
            }
            corrector.correct_residual(&mut r);
            rho[0]
        };
        cost += rho0;
        if point_free {
            if full {
                hpp += j_point.transpose() * j_point;
            }
            bp += -(j_point.transpose() * r);
        }
        // The observation's nodes in `target_frames` (fixed frames have
        // none), frame first.
        let mut touched: [Option<(usize, SMatrix<f64, 2, 6>)>; 3] = [None, None, None];
        let nodes = std::iter::once((o.frame_idx, j_pose)).chain(j_extra.into_iter().flatten());
        for (t, (node, j)) in nodes.enumerate() {
            let Ok(slot) = target_frames.binary_search(&node) else {
                continue;
            };
            let block = &mut frame_blocks[slot * FRAME_BLOCK..(slot + 1) * FRAME_BLOCK];
            let (diag, bc) = block.split_at_mut(36);
            let g = -(j.transpose() * r);
            for (acc, v) in bc.iter_mut().zip(g.as_slice()) {
                *acc += v;
            }
            if !full {
                continue;
            }
            let jtj = j.transpose() * j;
            for (acc, v) in diag.iter_mut().zip(jtj.as_slice()) {
                *acc += v;
            }
            if point_free {
                hcp[slot] += j.transpose() * j_point;
            }
            touched[t] = Some((slot, j));
        }
        if has_extras && full {
            let touched: Vec<(usize, SMatrix<f64, 2, 6>)> = touched.into_iter().flatten().collect();
            for (i, &(a, ja)) in touched.iter().enumerate() {
                for &(b, jb) in &touched[i + 1..] {
                    // Nodes ascend in `target_frames`: frame < camera < sensor.
                    let (lo, hi, j_lo, j_hi) = if a < b {
                        (a, b, ja, jb)
                    } else {
                        (b, a, jb, ja)
                    };
                    let key = [lo as u32, hi as u32];
                    let term = j_hi.transpose() * j_lo;
                    match direct.iter_mut().find(|(k, _)| *k == key) {
                        Some((_, acc)) => *acc += term,
                        None => direct.push((key, term)),
                    }
                }
            }
        }
    }

    (
        cost,
        PointLin {
            hpp,
            bp,
            free: point_free,
        },
        direct,
    )
}

/// Stage A driver: relinearize every point in parallel (deterministic —
/// `par_iter().collect()` on an `IndexedParallelIterator` preserves input
/// order regardless of thread count), then serially merge each point's
/// per-frame contribution into the (mu-independent) raw pose-block
/// accumulators. Returns `(full_cost, points_lin, frame_diag_raw,
/// frame_bc_raw, offdiag_raw)`; `offdiag_raw` (per edge, only with
/// [`ExtraBlocks`]) holds the direct couplings of the nodes one observation
/// touches. The reduced-system sparsity pattern
/// (`problem.edges`/`problem.point_pair_edge_idx`) is *not* recomputed here
/// — see [`build_reduced_system_pattern`]'s doc for why it is a
/// `build_problem`-time, not a per-linearization, computation.
#[allow(clippy::type_complexity)]
fn linearize(
    problem: &Problem,
    loss: LossFunction,
    mut hcp: Vec<SMatrix<f64, 6, 3>>,
    full: bool,
) -> (
    f64,
    Linearization,
    Vec<Matrix6<f64>>,
    Vec<Vector6<f64>>,
    Option<Vec<Matrix6<f64>>>,
) {
    // Points are linearized in parallel in a fixed number of point-index
    // shards, balanced by observations (fixed by the problem, not by the
    // thread count, so every sum below is the same at any thread count).
    // Each shard adds its points' per-frame `JᵀJ` / `−Jᵀr` straight into its
    // own frame accumulators, in point order, and the shards' accumulators
    // and costs are added in shard order. An earlier version had every
    // point write its blocks to a buffer (336 bytes per observation) and
    // folded them serially; that fold was a third of a linearization.
    let n_points = problem.points.len();
    let n_free = problem.n_free_frames;
    let n_shards = (n_points / 2048).clamp(1, LINEARIZE_SHARDS);
    let mut bounds = Vec::with_capacity(n_shards + 1);
    {
        let total: usize = problem.point_free_frames.iter().map(Vec::len).sum();
        let mut acc = 0usize;
        bounds.push(0);
        for (p, frames) in problem.point_free_frames.iter().enumerate() {
            if bounds.len() < n_shards && acc * n_shards >= total * bounds.len() {
                bounds.push(p);
            }
            acc += frames.len();
        }
        while bounds.len() <= n_shards {
            bounds.push(n_points);
        }
    }
    // `hcp` (the previous linearization's array, reused) split by shard;
    // a gradient-only pass leaves it alone. A solve's first linearization
    // starts from an empty array: it is zeroed in parallel, which spreads
    // the first touch of its pages over the threads (a serial `resize` of
    // a global BA's 2 GB was most of that linearization).
    if full {
        let len = problem.hcp_off[n_points];
        if hcp.len() < len {
            let missing = len - hcp.len();
            hcp.par_extend(rayon::iter::repeat_n(
                SMatrix::<f64, 6, 3>::zeros(),
                missing,
            ));
        } else {
            hcp.truncate(len);
        }
    }
    let mut hcp_shards = Vec::with_capacity(n_shards);
    let mut rest = hcp.as_mut_slice();
    for shard in 0..n_shards {
        let len = if full {
            problem.hcp_off[bounds[shard + 1]] - problem.hcp_off[bounds[shard]]
        } else {
            0
        };
        let (head, tail) = rest.split_at_mut(len);
        hcp_shards.push(head);
        rest = tail;
    }
    #[allow(clippy::type_complexity)]
    let shards: Vec<(
        f64,
        Vec<PointLin>,
        Vec<Matrix6<f64>>,
        Vec<Vector6<f64>>,
        Vec<(usize, Vec<([u32; 2], Matrix6<f64>)>)>,
    )> = hcp_shards
        .into_par_iter()
        .enumerate()
        .map(|(shard, shard_hcp)| {
            let range = bounds[shard]..bounds[shard + 1];
            let shard_start = problem.hcp_off[range.start];
            let mut diag = vec![Matrix6::<f64>::zeros(); n_free];
            let mut bc = vec![Vector6::<f64>::zeros(); n_free];
            let mut cost = 0.0;
            let mut lin = Vec::with_capacity(range.len());
            let mut direct_terms = Vec::new();
            let mut frame_blocks: Vec<f64> = Vec::new();
            for p in range {
                let frames = &problem.point_free_frames[p];
                frame_blocks.resize(frames.len() * FRAME_BLOCK, 0.0);
                let (start, end) = problem.point_obs_range[p];
                let (point_cost, pl, direct) = linearize_point(
                    (&problem.obs[start as usize..end as usize], &problem.cameras),
                    !problem.point_fixed[p],
                    &problem.points[p],
                    &problem.poses,
                    frames,
                    loss,
                    (&problem.extras, problem.n_frames as u32),
                    &mut frame_blocks,
                    if full {
                        &mut shard_hcp
                            [problem.hcp_off[p] - shard_start..problem.hcp_off[p + 1] - shard_start]
                    } else {
                        &mut []
                    },
                    full,
                );
                for (&f, block) in frames.iter().zip(frame_blocks.chunks_exact(FRAME_BLOCK)) {
                    let slot = problem.free_frame_slot[f as usize].expect("free") as usize;
                    if full {
                        diag[slot] += Matrix6::from_column_slice(&block[..36]);
                    }
                    bc[slot] += Vector6::from_column_slice(&block[36..]);
                }
                cost += point_cost;
                lin.push(pl);
                if !direct.is_empty() {
                    direct_terms.push((p, direct));
                }
            }
            (cost, lin, diag, bc, direct_terms)
        })
        .collect();

    let mut full_cost = 0.0;
    let mut points_lin = Vec::with_capacity(n_points);
    let mut frame_diag_raw = vec![Matrix6::<f64>::zeros(); n_free];
    let mut frame_bc_raw = vec![Vector6::<f64>::zeros(); n_free];
    let mut offdiag_raw =
        (problem.extras.len() > 0).then(|| vec![Matrix6::<f64>::zeros(); problem.edges.len()]);
    for (cost, lin, diag, bc, direct_terms) in shards {
        full_cost += cost;
        points_lin.extend(lin);
        for (total, d) in frame_diag_raw.iter_mut().zip(diag) {
            *total += d;
        }
        for (total, b) in frame_bc_raw.iter_mut().zip(bc) {
            *total += b;
        }
        if let Some(offdiag) = offdiag_raw.as_mut() {
            for (p, direct) in direct_terms {
                let frames = &problem.point_free_frames[p];
                for ([lo, hi], term) in direct {
                    offdiag[direct_edge(problem, frames[lo as usize], frames[hi as usize])] += term;
                }
            }
        }
    }

    (
        full_cost,
        Linearization {
            points: points_lin,
            hcp,
        },
        frame_diag_raw,
        frame_bc_raw,
        offdiag_raw,
    )
}

/// The edge of a direct term between the nodes `lo < hi` of one
/// observation: `hi` is an extra node, and the extra nodes take the last
/// slots, so it is among the last few entries of `lo`'s row.
fn direct_edge(problem: &Problem, lo: u32, hi: u32) -> usize {
    let slot = |node: u32| problem.free_frame_slot[node as usize].expect("free");
    let (a, b) = (slot(lo) as usize, slot(hi));
    let row_off = &problem.pairs.row_off;
    let row = &problem.edges[row_off[a]..row_off[a + 1]];
    row_off[a]
        + row
            .iter()
            .rposition(|&(_, partner)| partner == b)
            .expect("the observation's nodes share an edge")
}

#[inline]
fn clamp_diag6(h: &Matrix6<f64>) -> Vector6<f64> {
    Vector6::from_iterator((0..6).map(|i| h[(i, i)].clamp(MIN_DIAGONAL, MAX_DIAGONAL)))
}

#[inline]
fn clamp_diag3(h: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::from_iterator((0..3).map(|i| h[(i, i)].clamp(MIN_DIAGONAL, MAX_DIAGONAL)))
}

/// Damped 3×3 point-block inverse at the given `mu = 1/radius`, or `None` if
/// the damped block is (numerically) singular.
fn damped_hpp_inverse(hpp: &Matrix3<f64>, mu: f64) -> Option<Matrix3<f64>> {
    let diag = clamp_diag3(hpp);
    let mut damped = *hpp;
    for i in 0..3 {
        damped[(i, i)] += mu * diag[i];
    }
    damped.try_inverse()
}

/// Per-phase wall time accumulated across every trial of one `optimize()`
/// call, printed as `BA_PHASES` alongside `BA_SOLVE` (task item: "Add timing
/// instrumentation inside the native solver per phase"). Phases match the
/// task brief's list: `eliminate` = per-point Schur elimination,
/// `assemble` = merging eliminated contributions into the reduced-system
/// blocks (+ pose damping + `columns`/`rhs` construction), `linsolve` =
/// [`crate::block_cholesky::solve_spd_blocks6_cached`], `backsub` =
/// free-point back-substitution + predicted-decrease bookkeeping,
/// `evaluate_cost` = the trial-step residual-only cost re-evaluation.
/// `linearize` is timed separately in [`optimize`] (only runs on an
/// accepted step, so its count usually differs from the trial count).
#[derive(Default, Debug, Clone, Copy)]
struct PhaseTimings {
    linearize: Duration,
    /// Gradient-only passes deciding the gradient stop ([`linearize`]).
    gradient: Duration,
    eliminate: Duration,
    assemble: Duration,
    linsolve: Duration,
    backsub: Duration,
    evaluate_cost: Duration,
    linearize_calls: u32,
    trials: u32,
    reorder: Duration,
    pcg_iterations: usize,
}

/// Performance fix #2 (see [`build_reduced_system_pattern`] for fix #1):
/// profiling after fix #1 still showed `eliminate_ms`+`assemble_ms`
/// dominating (~1.4s of ~2.2s total at 300f/2500pt/289k-obs, 7 trials) even
/// with the flat-`Vec` merge and the precomputed edge pattern. Two things
/// remained: (a) `eliminate_point` still heap-allocated an
/// O(track_len²)-sized `Vec<Matrix6>` *per point, per trial* (its `pairs`
/// output) purely to hand values to a *separate* merge step — at ~2500
/// points × up to `~2500` pairs each, that is gigabytes of transient
/// allocation per trial; (b) the merge then wrote those values into
/// `flat_offdiag` at effectively random indices (`pair_edge_idx` scatters
/// across the whole edge set), which is cache-unfriendly and was serial
/// (one core). This function fuses per-point elimination and the
/// flat-buffer write into one step with **no** per-point/per-pair heap
/// allocation (only a small O(track_len) `hcp_scaled` scratch buffer,
/// reused — not reallocated — across every point in a shard) and shards
/// the O(total pairs) work across a *fixed* number of point-index ranges
/// (independent of `RAYON_NUM_THREADS`) so it runs in parallel while
/// staying bit-identical at any thread count (module doc): each shard
/// writes into its own private `edges.len()`-sized buffer (no cross-thread
/// contention), and the buffers are summed in a fixed shard-index order
/// afterward.
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)]
fn eliminate_and_accumulate(
    problem: &Problem,
    lin: &Linearization,
    edges_len: usize,
    mu: f64,
) -> (Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    let n_free = problem.n_free_frames;
    if let Some(index) = &problem.contributions {
        return eliminate_by_output(problem, index, lin, edges_len, mu);
    }
    let n_shards = problem.shard_boundaries.len().saturating_sub(1).max(1);

    let shard_results: Vec<(Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>)> = (0
        ..n_shards)
        .into_par_iter()
        .map(|shard| {
            let lo = problem.shard_boundaries[shard];
            let hi = problem.shard_boundaries[shard + 1];
            let mut local_offdiag = vec![Matrix6::<f64>::zeros(); edges_len];
            let mut local_diag = vec![Matrix6::<f64>::zeros(); n_free];
            let mut local_bc = vec![Vector6::<f64>::zeros(); n_free];
            let mut hcp_scaled: Vec<SMatrix<f64, 6, 3>> = Vec::new();
            // Both `hcp` and `problem.point_pair_edge_idx`
            // are indexed by the same `p`, and the pair loop below indexes
            // `hcp_scaled`/`hcp` at two *different* offsets (`i`, `j`) in
            // the same iteration — an index loop reads more directly than
            // threading multiple `.zip()`s through both (hence the
            // `needless_range_loop` allow above).
            for p in lo..hi.max(lo) {
                let pl = &lin.points[p];
                let hcp = lin.hcp(problem, p);
                if !pl.free || hcp.is_empty() {
                    continue;
                }
                let n = hcp.len();
                let Some(hpp_inv) = damped_hpp_inverse(&pl.hpp, mu) else {
                    // Degenerate point block: contribute nothing rather than
                    // propagate a solve failure (see the original
                    // `eliminate_point` doc, same policy).
                    continue;
                };
                hcp_scaled.clear();
                hcp_scaled.extend((0..n).map(|k| hcp[k] * hpp_inv));
                for k in 0..n {
                    let slot = problem.free_frame_slot[problem.point_free_frames[p][k] as usize]
                        .expect("free") as usize;
                    local_diag[slot] -= hcp_scaled[k] * hcp[k].transpose();
                    local_bc[slot] -= hcp_scaled[k] * pl.bp;
                }
                let pair_edge_idx = &problem.point_pair_edge_idx()[p];
                let mut local_pair = 0usize;
                for i in 0..n {
                    for j in i + 1..n {
                        sub_outer_6x3(
                            &mut local_offdiag[pair_edge_idx[local_pair] as usize],
                            &hcp[j],
                            &hcp_scaled[i],
                        );
                        local_pair += 1;
                    }
                }
            }
            (local_offdiag, local_diag, local_bc)
        })
        .collect();

    let mut flat_offdiag = vec![Matrix6::<f64>::zeros(); edges_len];
    let mut diag_delta = vec![Matrix6::<f64>::zeros(); n_free];
    let mut bc_delta = vec![Vector6::<f64>::zeros(); n_free];
    // Fixed shard-index order (0..n_shards, not runtime scheduling order) —
    // bit-identical at any thread count, matching `block_cholesky`'s
    // documented "fixed chunk, folded in order" determinism pattern.
    for (local_offdiag, local_diag, local_bc) in shard_results {
        for (a, b) in flat_offdiag.iter_mut().zip(local_offdiag) {
            *a += b;
        }
        for (a, b) in diag_delta.iter_mut().zip(local_diag) {
            *a += b;
        }
        for (a, b) in bc_delta.iter_mut().zip(local_bc) {
            *a += b;
        }
    }
    (flat_offdiag, diag_delta, bc_delta)
}

/// `acc -= a bᵀ`, the Schur complement's per-pair term (`b` is the
/// partner block already scaled by the point's inverse Hessian): three
/// fused multiply-adds per entry straight into the accumulator, column by
/// column, which the compiler vectorizes (a fifth off the elimination of
/// the `large_global_ba` bench against the generic product and a separate
/// subtraction). Both elimination paths use it, so they stay bit-identical.
#[inline(always)]
fn sub_outer_6x3(acc: &mut Matrix6<f64>, a: &SMatrix<f64, 6, 3>, b: &SMatrix<f64, 6, 3>) {
    let (a, b) = (a.as_slice(), b.as_slice());
    for (c, column) in acc.as_mut_slice().chunks_exact_mut(6).enumerate() {
        let (b0, b1, b2) = (-b[c], -b[6 + c], -b[12 + c]);
        for (r, value) in column.iter_mut().enumerate() {
            *value = a[12 + r].mul_add(b2, a[6 + r].mul_add(b1, a[r].mul_add(b0, *value)));
        }
    }
}

/// [`eliminate_and_accumulate`] computed per output block, in parallel:
/// each off-diagonal block, diagonal block and right-hand side sums its
/// points' contributions in ascending point order, starting a fresh partial
/// sum at every shard boundary and adding the partial sums in shard order,
/// which is exactly the sharded loop's arithmetic (it subtracts each
/// shard's points in order into a zeroed per-shard buffer, then adds the
/// buffers in shard order). Used when the per-shard buffers would not fit
/// in memory more than a few at a time, where the sharded loop ran on as
/// few as 2 threads.
#[allow(clippy::type_complexity)]
fn eliminate_by_output(
    problem: &Problem,
    index: &ContributionIndex,
    lin: &Linearization,
    edges_len: usize,
    mu: f64,
) -> (Vec<Matrix6<f64>>, Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    let n_free = problem.n_free_frames;
    let hpp_inv = point_inverses(problem, lin, mu);
    // Row `s` (edges `(s, _)`), the diagonal block and the right-hand side
    // of free frame `s` all come from the points observing `s`, in
    // ascending order, so one pass computes them, each row into its own
    // slice of the block array (zeroed in parallel: building each row in
    // its own buffer and concatenating held every block twice). A pair's
    // block in the row is found through `position`, the row's position of
    // each partner slot, filled per row: an index over every point's pairs
    // instead took 1.9 GB on SmallCity's last bundle adjustment, read once
    // per elimination.
    let mut flat_offdiag: Vec<Matrix6<f64>> = (0..edges_len)
        .into_par_iter()
        .map(|_| Matrix6::zeros())
        .collect();
    let mut rows: Vec<&mut [Matrix6<f64>]> = Vec::with_capacity(n_free);
    let mut rest = flat_offdiag.as_mut_slice();
    for s in 0..n_free {
        let (row, tail) = rest.split_at_mut(index.row_off[s + 1] - index.row_off[s]);
        rows.push(row);
        rest = tail;
    }
    let per_row: Vec<(Matrix6<f64>, Vector6<f64>)> = rows
        .into_par_iter()
        .enumerate()
        .map_init(
            || (Vec::new(), vec![0u32; n_free]),
            |(acc, position), (s, total)| {
                let len = total.len();
                acc.clear();
                acc.resize(len, Matrix6::<f64>::zeros());
                for (i, &(_, b)) in problem.edges[index.row_off[s]..index.row_off[s + 1]]
                    .iter()
                    .enumerate()
                {
                    position[b as usize] = i as u32;
                }
                let (mut diag_total, mut diag_acc) = (Matrix6::<f64>::zeros(), Matrix6::zeros());
                let (mut bc_total, mut bc_acc) = (Vector6::<f64>::zeros(), Vector6::zeros());
                // One shard's partial sums at a time (points ascend, so each
                // shard's points are contiguous), added to the totals when
                // the shard ends. Blocks the shard did not touch add zero,
                // which leaves a total unchanged (it starts at +0, never -0).
                let mut current_shard = None;
                let mut flush = |acc: &mut Vec<Matrix6<f64>>,
                                 diag_acc: &mut Matrix6<f64>,
                                 bc_acc: &mut Vector6<f64>| {
                    for (t, a) in total.iter_mut().zip(acc.iter_mut()) {
                        *t += *a;
                        *a = Matrix6::zeros();
                    }
                    diag_total += *diag_acc;
                    bc_total += *bc_acc;
                    *diag_acc = Matrix6::zeros();
                    *bc_acc = Vector6::zeros();
                };
                for &[p, k] in &index.slot[index.slot_off[s]..index.slot_off[s + 1]] {
                    let (p, k) = (p as usize, k as usize);
                    let Some(inv) = hpp_inv[p] else {
                        continue;
                    };
                    let pl = &lin.points[p];
                    let hcp = lin.hcp(problem, p);
                    let point_shard = Some(index.point_shard[p]);
                    if current_shard != point_shard {
                        if current_shard.is_some() {
                            flush(acc, &mut diag_acc, &mut bc_acc);
                        }
                        current_shard = point_shard;
                    }
                    let scaled = hcp[k] * inv;
                    diag_acc -= scaled * hcp[k].transpose();
                    bc_acc -= scaled * pl.bp;
                    // `k`'s pairs with the point's later frames (none for the
                    // last frame).
                    let later = &problem.point_free_frames[p][k + 1..];
                    for (block, &f) in hcp[k + 1..].iter().zip(later) {
                        let b = problem.free_frame_slot[f as usize].expect("free");
                        sub_outer_6x3(&mut acc[position[b as usize] as usize], block, &scaled);
                    }
                }
                if current_shard.is_some() {
                    flush(acc, &mut diag_acc, &mut bc_acc);
                }
                (diag_total, bc_total)
            },
        )
        .collect();
    let (diag_delta, bc_delta) = per_row.into_iter().unzip();
    (flat_offdiag, diag_delta, bc_delta)
}

/// Each point's damped inverse Hessian block, `None` for a point that
/// contributes nothing to the elimination (fixed, unobserved by free frames,
/// or degenerate).
fn point_inverses(problem: &Problem, lin: &Linearization, mu: f64) -> Vec<Option<Matrix3<f64>>> {
    lin.points
        .par_iter()
        .zip(&problem.point_free_frames)
        .map(|(pl, frames)| {
            if !pl.free || frames.is_empty() {
                None
            } else {
                damped_hpp_inverse(&pl.hpp, mu)
            }
        })
        .collect()
}

/// The diagonal blocks and right-hand side of [`eliminate_by_output`]: the
/// same sums, in the same order (shard partial sums included), without the
/// off-diagonal pairs.
fn eliminate_diagonal(
    problem: &Problem,
    index: &ContributionIndex,
    lin: &Linearization,
    hpp_inv: &[Option<Matrix3<f64>>],
) -> (Vec<Matrix6<f64>>, Vec<Vector6<f64>>) {
    (0..problem.n_free_frames)
        .into_par_iter()
        .map(|s| {
            let (mut diag_total, mut diag_acc) = (Matrix6::<f64>::zeros(), Matrix6::zeros());
            let (mut bc_total, mut bc_acc) = (Vector6::<f64>::zeros(), Vector6::zeros());
            let mut current_shard = None;
            for &[p, k] in &index.slot[index.slot_off[s]..index.slot_off[s + 1]] {
                let (p, k) = (p as usize, k as usize);
                let Some(inv) = hpp_inv[p] else {
                    continue;
                };
                let point_shard = Some(index.point_shard[p]);
                if current_shard != point_shard {
                    if current_shard.is_some() {
                        diag_total += diag_acc;
                        bc_total += bc_acc;
                        diag_acc = Matrix6::zeros();
                        bc_acc = Vector6::zeros();
                    }
                    current_shard = point_shard;
                }
                let hcp = lin.hcp(problem, p);
                let scaled = hcp[k] * inv;
                diag_acc -= scaled * hcp[k].transpose();
                bc_acc -= scaled * lin.points[p].bp;
            }
            if current_shard.is_some() {
                diag_total += diag_acc;
                bc_total += bc_acc;
            }
            (diag_total, bc_total)
        })
        .unzip()
}

/// [`solve_step`]'s iterative branch with the point elimination's pairs on
/// a [`crate::ba_accel::ReducedSolveAccelerator`]: the CPU forms the
/// diagonal blocks and the right-hand side ([`eliminate_diagonal`], the CPU
/// path's arithmetic), the accelerator the off-diagonal blocks and the
/// conjugate gradients, with the preconditioner factored here from the
/// blocks it returns. `None`: the caller eliminates on the CPU.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn solve_step_device_elimination(
    problem: &Problem,
    index: &ContributionIndex,
    lin: &Linearization,
    linearization_id: u64,
    accelerator: &dyn crate::ba_accel::ReducedSolveAccelerator,
    frame_diag_raw: &[Matrix6<f64>],
    frame_bc_raw: &[Vector6<f64>],
    offdiag_raw: Option<&[Matrix6<f64>]>,
    mu: f64,
    timings: &mut PhaseTimings,
) -> Option<(Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64)> {
    if problem.point_nodes.is_empty() {
        return None;
    }
    let n_free = problem.n_free_frames;
    let edges = &problem.edges;
    let first_extra = n_free - problem.extras.len();
    let t = Instant::now();
    let hpp_inv = point_inverses(problem, lin, mu);
    let (diag_delta, bc_delta) = eliminate_diagonal(problem, index, lin, &hpp_inv);
    let scale: Vec<Vector6<f64>> = diag_delta
        .iter()
        .map(|d| Vector6::from_fn(|i, _| (-d[(i, i)]).max(0.0).sqrt()))
        .collect();
    // Direct couplings only join an extra node (the last nodes, so the
    // larger end of an edge).
    let extra: Vec<(u32, Matrix6<f64>)> = offdiag_raw
        .map(|raw| {
            edges
                .iter()
                .zip(raw)
                .enumerate()
                .filter(|(_, (&(_, b), m))| {
                    b as usize >= first_extra && m.iter().any(|&v| v != 0.0)
                })
                .map(|(e, (_, m))| (e as u32, *m))
                .collect()
        })
        .unwrap_or_default();
    let cluster_k = pcg_cluster_size();
    let run_edges: Vec<u32> = cluster_k.map_or_else(Vec::new, |k| {
        (0..edges.len() as u32)
            .filter(|&e| {
                let (a, b) = edges[e as usize];
                a as usize / k == b as usize / k
            })
            .collect()
    });
    // `coarse_correction`'s aggregate size and coverage.
    let coarse_spec = pcg_coarse_size()
        .filter(|_| mu <= PCG_COARSE_MAX_MU)
        .map(|g| {
            (
                g.max(first_extra.div_ceil(PCG_COARSE_MAX_AGGREGATES)),
                first_extra,
            )
        })
        .filter(|&(group, nodes)| nodes.div_ceil(group) >= 2);
    let blocks = accelerator.eliminate(&crate::ba_accel::PointElimination {
        structure_id: problem.structure_id,
        linearization_id,
        edges,
        row_off: &index.row_off,
        slot_off: &index.slot_off,
        slots: &index.slot,
        point_off: &problem.hcp_off,
        point_nodes: &problem.point_nodes,
        hcp: &lin.hcp,
        point_inverse: &hpp_inv,
        scale: &scale,
        extra: &extra,
        run_edges: &run_edges,
        coarse: coarse_spec,
    })?;
    timings.eliminate += t.elapsed();

    let t = Instant::now();
    let mut frame_diag: Vec<Matrix6<f64>> = frame_diag_raw.to_vec();
    let mut frame_bc: Vec<Vector6<f64>> = frame_bc_raw.to_vec();
    for a in 0..n_free {
        frame_diag[a] += diag_delta[a];
        frame_bc[a] += bc_delta[a];
        let d = clamp_diag6(&frame_diag_raw[a]);
        for i in 0..6 {
            frame_diag[a][(i, i)] += mu * d[i];
        }
    }
    for e in 0..problem.extras.len() {
        let free = problem.extras.free(e);
        for i in (0..6).filter(|&i| !free[i]) {
            frame_diag[first_extra + e][(i, i)] = 1.0;
        }
    }
    timings.assemble += t.elapsed();

    let t = Instant::now();
    let clusters = cluster_k.and_then(|k| {
        cluster_inverses(
            &frame_diag,
            k,
            run_edges
                .iter()
                .map(|&e| edges[e as usize])
                .zip(&blocks.run_blocks),
        )
    });
    let coarse = coarse_spec.and_then(|(group, nodes)| {
        coarse_from_sums(&frame_diag, &blocks.coarse_sums, nodes, group)
    });
    let coarse_inverse = coarse.as_ref().map(CoarseCorrection::inverse);
    let (dx_frames, iterations) = accelerator.solve(&ReducedCameraSystem {
        structure_id: problem.structure_id,
        diag: &frame_diag,
        edges,
        offdiag: &[],
        rhs: &frame_bc,
        clusters: clusters.as_deref(),
        coarse: coarse
            .as_ref()
            .zip(coarse_inverse.as_ref())
            .map(|(c, inverse)| (c.group, c.n_frames, inverse)),
        tol: pcg_relative_tolerance(),
        max_iterations: PCG_MAX_ITERATIONS,
    })?;
    timings.pcg_iterations += iterations;
    timings.linsolve += t.elapsed();
    Some(back_substitute(
        problem,
        lin,
        frame_diag_raw,
        frame_bc_raw,
        dx_frames,
        mu,
        timings,
    ))
}

/// Reduced systems with at least this many off-diagonal blocks are
/// eliminated by output ([`ContributionIndex`]) whatever their frame count.
/// The sharded loop's 8 buffers of every block took 825 MB at 358k blocks
/// (995 frames), the peak of SmallCity pass 1; by output, the eliminations
/// from 20k to 400k blocks there took 0.59 to 0.90 of the time, with the
/// same arithmetic.
const BY_OUTPUT_MIN_EDGES: usize = 20_000;

/// Shard count for [`eliminate_and_accumulate`]: capped so the transient
/// per-shard `edges_len`-sized buffers (`3 shards worth` — see its `local_*`
/// locals — of `Matrix6` = 288 bytes each) never exceed roughly 256MB in
/// aggregate, and bounded to `[1, 8]` (the pool this crate targets). A pure
/// function of `edges_len` (fixed for the whole `optimize()` call — see
/// `build_reduced_system_pattern`), so the shard count — hence the fixed
/// reduction order in [`eliminate_and_accumulate`] — never depends on
/// `RAYON_NUM_THREADS`.
fn elimination_shard_count(edges_len: usize) -> usize {
    // Real (corridor-revisit) global-BA windows can have `edges_len` well
    // into the hundreds of thousands (dense frame-frame co-observation from
    // long, slowly-moving tracks) — e.g. `edges_len=357k` (≈ `C(846,2)`,
    // observed on the real tier-1000 run's largest global-BA window) costs
    // ~103MB *per shard*. At the previous 256MB cap that collapsed to just
    // 2 shards, discarding most of the intended 8x parallelism exactly on
    // the heaviest problems — plausibly the dominant reason a same-sized
    // synthetic profile (with a more modest `edges_len`, hence 8 shards)
    // measured far faster than the real run's global-BA `BA_PHASES` lines.
    // 1.5GB keeps 8 shards up to `edges_len≈650k` (the real run's peak RSS
    // for the *entire* mapper process was 1.15GB, so this has headroom).
    const MAX_TOTAL_BYTES: usize = 1536 * 1024 * 1024;
    let per_shard_bytes = edges_len.max(1) * std::mem::size_of::<Matrix6<f64>>();
    (MAX_TOTAL_BYTES / per_shard_bytes).clamp(1, 8)
}

/// Performance fix #3: [`elimination_shard_count`] fixes *how many* shards,
/// this fixes *which points* land in each — balanced by cumulative pair
/// count (`Σ n·(n-1)/2` over each shard's points), not point count. A real
/// (corridor-revisit) track-length distribution is heavy-tailed: a handful
/// of long-revisited points can carry most of the O(pairs) work, and an
/// equal-point-count split (the original scheme) can land all of them in
/// one shard while the other 7 threads idle — losing most of the intended
/// 8x speedup exactly on the problems that need it most. Computed once (pair
/// counts, hence the optimal boundaries, are fixed for the whole
/// `optimize()` call) via a prefix sum over `point_free_frames` and a
/// `partition_point` binary search per boundary — O(n_points log n_points),
/// negligible next to the O(pairs) work it schedules.
fn balanced_shard_boundaries(point_free_frames: &[Vec<u32>], n_shards: usize) -> Vec<usize> {
    let n_points = point_free_frames.len();
    let n_shards = n_shards.max(1);
    let mut prefix: Vec<u64> = Vec::with_capacity(n_points + 1);
    prefix.push(0);
    for frames in point_free_frames {
        let n = frames.len() as u64;
        prefix.push(prefix.last().copied().unwrap_or(0) + n * n.saturating_sub(1) / 2);
    }
    let total = *prefix.last().unwrap_or(&0);
    let mut boundaries = Vec::with_capacity(n_shards + 1);
    boundaries.push(0usize);
    for s in 1..n_shards {
        let target = total * s as u64 / n_shards as u64;
        let idx = prefix.partition_point(|&v| v < target);
        let idx = idx.clamp(*boundaries.last().expect("non-empty"), n_points);
        boundaries.push(idx);
    }
    boundaries.push(n_points);
    boundaries
}

/// Stage B driver: assemble the damped, Schur-reduced 6×6-block camera
/// system at the given `mu` and solve it via
/// [`crate::block_cholesky::solve_spd_blocks6_cached`], then back-substitute
/// the free-point steps. Returns `(dx_frames, dx_points, predicted_decrease)`
/// (all on the free-slot / point-index space); `None` if the reduced system
/// was singular even after damping (`BaError::SingularSystem`).
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn solve_step(
    problem: &Problem,
    lin: &Linearization,
    linearization_id: u64,
    edges: &[(u32, u32)],
    frame_diag_raw: &[Matrix6<f64>],
    frame_bc_raw: &[Vector6<f64>],
    offdiag_raw: Option<&[Matrix6<f64>]>,
    mu: f64,
    cache: &mut Option<BlockSymbolic>,
    order: Option<&[u32]>,
    timings: &mut PhaseTimings,
) -> Option<(Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64)> {
    let n_free = problem.n_free_frames;
    // The iterative branch with the points eliminated by a registered
    // accelerator; when it declines, the CPU eliminates below.
    if n_free >= problem.iterative_min_free_frames && order.is_none() {
        if let (Some(accelerator), Some(index)) = (
            crate::ba_accel::reduced_solve_accelerator(),
            &problem.contributions,
        ) {
            if let Some(step) = solve_step_device_elimination(
                problem,
                index,
                lin,
                linearization_id,
                accelerator,
                frame_diag_raw,
                frame_bc_raw,
                offdiag_raw,
                mu,
                timings,
            ) {
                return Some(step);
            }
        }
    }
    let t = Instant::now();
    let (mut flat_offdiag, diag_delta, bc_delta) =
        eliminate_and_accumulate(problem, lin, edges.len(), mu);
    // The extra nodes' direct couplings ([`linearize`]).
    if let Some(raw) = offdiag_raw {
        for (acc, term) in flat_offdiag.iter_mut().zip(raw) {
            *acc += term;
        }
    }
    timings.eliminate += t.elapsed();

    let t = Instant::now();
    let mut frame_diag: Vec<Matrix6<f64>> = frame_diag_raw.to_vec();
    let mut frame_bc: Vec<Vector6<f64>> = frame_bc_raw.to_vec();
    for a in 0..n_free {
        frame_diag[a] += diag_delta[a];
        frame_bc[a] += bc_delta[a];
    }
    // Pose-block Jacobi damping, applied to the *raw* (pre-elimination)
    // diagonal — matches Ceres' `LevenbergMarquardtStrategy` (module doc).
    for a in 0..n_free {
        let d = clamp_diag6(&frame_diag_raw[a]);
        for i in 0..6 {
            frame_diag[a][(i, i)] += mu * d[i];
        }
    }
    // Fixed parameters of the extra nodes: a zero row and column, so an
    // identity diagonal gives them a zero step.
    let first_extra = n_free - problem.extras.len();
    for e in 0..problem.extras.len() {
        let free = problem.extras.free(e);
        for i in (0..6).filter(|&i| !free[i]) {
            frame_diag[first_extra + e][(i, i)] = 1.0;
        }
    }

    if n_free >= problem.iterative_min_free_frames && order.is_none() {
        // Iterative solve straight from the edge list: each off-diagonal
        // block is read once per product, contributing to both of its
        // rows (`y_b += A x_a`, `y_a += Aᵀ x_b`). The product is limited by
        // memory traffic, and storing both triangles read every block
        // twice. The edges are cut into a fixed number of chunks, each
        // accumulating into its own vector, summed in chunk order, so the
        // result does not depend on the thread count. The blocks are read
        // in f32 unless `VISLOC_PORT_PCG_F64_BLOCKS=1` ([`pcg_f32_blocks`]).
        // One f32 copy serves both a registered accelerator (its input) and
        // the CPU product.
        let accelerator = crate::ba_accel::reduced_solve_accelerator();
        let blocks32: Option<Vec<f32>> = (accelerator.is_some() || pcg_f32_blocks()).then(|| {
            let mut out = vec![0.0f32; flat_offdiag.len() * 36];
            out.par_chunks_mut(36)
                .zip(flat_offdiag.par_iter())
                .for_each(|(out, blk)| {
                    for (o, &v) in out.iter_mut().zip(blk.as_slice()) {
                        *o = v as f32;
                    }
                });
            out
        });
        let cpu_blocks32 = blocks32.as_deref().filter(|_| pcg_f32_blocks());
        let n_chunks = PCG_MATVEC_CHUNKS.min(edges.len().max(1));
        let bounds: Vec<usize> = (0..=n_chunks).map(|c| c * edges.len() / n_chunks).collect();
        let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
            let parts: Vec<Vec<Vector6<f64>>> = (0..n_chunks)
                .into_par_iter()
                .map(|c| {
                    let mut y = vec![Vector6::zeros(); n_free];
                    for e in bounds[c]..bounds[c + 1] {
                        let (a, b) = (edges[e].0 as usize, edges[e].1 as usize);
                        let (ya, yb) = match cpu_blocks32 {
                            Some(blocks) => {
                                edge_product(&blocks[36 * e..36 * e + 36], &x[a], &x[b])
                            }
                            None => edge_product(flat_offdiag[e].as_slice(), &x[a], &x[b]),
                        };
                        y[a] += ya;
                        y[b] += yb;
                    }
                    y
                })
                .collect();
            (0..n_free)
                .into_par_iter()
                .map(|i| {
                    parts
                        .iter()
                        .fold(frame_diag[i] * x[i], |acc, part| acc + part[i])
                })
                .collect()
        };
        let mut rhs = DMatrix::<f64>::zeros(n_free * 6, 1);
        for a in 0..n_free {
            for i in 0..6 {
                rhs[(a * 6 + i, 0)] = frame_bc[a][i];
            }
        }
        timings.assemble += t.elapsed();
        let t = Instant::now();
        let clusters = pcg_cluster_size()
            .and_then(|k| cluster_preconditioner(&frame_diag, edges, &flat_offdiag, k));
        let coarse = pcg_coarse_size()
            .filter(|_| mu <= PCG_COARSE_MAX_MU)
            .and_then(|g| coarse_correction(&frame_diag, edges, &flat_offdiag, first_extra, g));
        // A registered accelerator (e.g. visloc-ba-gpu's) runs the same
        // iteration; it may decline, and then the CPU solve below runs.
        if let (Some(accelerator), Some(offdiag)) = (accelerator, blocks32.as_deref()) {
            let coarse_inverse = coarse.as_ref().map(CoarseCorrection::inverse);
            let system = ReducedCameraSystem {
                structure_id: problem.structure_id,
                diag: &frame_diag,
                edges,
                offdiag,
                rhs: &frame_bc,
                clusters: clusters.as_deref(),
                coarse: coarse
                    .as_ref()
                    .zip(coarse_inverse.as_ref())
                    .map(|(c, inverse)| (c.group, c.n_frames, inverse)),
                tol: pcg_relative_tolerance(),
                max_iterations: PCG_MAX_ITERATIONS,
            };
            if let Some((dx_frames, iterations)) = accelerator.solve(&system) {
                timings.pcg_iterations += iterations;
                timings.linsolve += t.elapsed();
                return Some(back_substitute(
                    problem,
                    lin,
                    frame_diag_raw,
                    frame_bc_raw,
                    dx_frames,
                    mu,
                    timings,
                ));
            }
        }
        let (solved, iterations) = pcg_core(
            &frame_diag,
            matvec,
            &rhs,
            pcg_relative_tolerance(),
            clusters.as_deref(),
            coarse.as_ref(),
        )?;
        timings.pcg_iterations += iterations;
        let mut dx_frames = vec![Vector6::<f64>::zeros(); n_free];
        for a in 0..n_free {
            for i in 0..6 {
                dx_frames[a][i] = solved[(a * 6 + i, 0)];
            }
        }
        timings.linsolve += t.elapsed();
        return Some(back_substitute(
            problem,
            lin,
            frame_diag_raw,
            frame_bc_raw,
            dx_frames,
            mu,
            timings,
        ));
    }

    // Free frame `a` sits at block `pos(a)` of the factored system: the
    // fill-reducing position when an order is given (large systems, see
    // `REORDER_MIN_FREE_FRAMES`), else `a` itself.
    let pos = |a: usize| order.map_or(a, |o| o[a] as usize);
    let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n_free];
    for (a, &diag) in frame_diag.iter().enumerate().take(n_free) {
        columns[pos(a)].insert(pos(a), diag);
    }
    // `val` is the (row b, column a) block, a < b. Keep the lower triangle
    // of the permuted matrix: transpose when the permutation flips the pair.
    for (&(a, b), val) in edges.iter().zip(flat_offdiag) {
        let (pa, pb) = (pos(a as usize), pos(b as usize));
        if pb >= pa {
            columns[pa].insert(pb, val);
        } else {
            columns[pb].insert(pa, val.transpose());
        }
    }
    let mut rhs = DMatrix::<f64>::zeros(n_free * 6, 1);
    for a in 0..n_free {
        for i in 0..6 {
            rhs[(pos(a) * 6 + i, 0)] = frame_bc[a][i];
        }
    }
    timings.assemble += t.elapsed();

    let t = Instant::now();
    let solved = if n_free >= problem.iterative_min_free_frames {
        pcg_blocks6(&columns, &rhs, pcg_relative_tolerance())?
    } else {
        solve_spd_blocks6_cached(cache, columns, &rhs).ok()?
    };
    let mut dx_frames = vec![Vector6::<f64>::zeros(); n_free];
    for a in 0..n_free {
        for i in 0..6 {
            dx_frames[a][i] = solved[(pos(a) * 6 + i, 0)];
        }
    }
    timings.linsolve += t.elapsed();
    Some(back_substitute(
        problem,
        lin,
        frame_diag_raw,
        frame_bc_raw,
        dx_frames,
        mu,
        timings,
    ))
}

/// Free-point steps from the camera steps, and the LM model's predicted
/// cost decrease, for [`solve_step`].
fn back_substitute(
    problem: &Problem,
    lin: &Linearization,
    frame_diag_raw: &[Matrix6<f64>],
    frame_bc_raw: &[Vector6<f64>],
    dx_frames: Vec<Vector6<f64>>,
    mu: f64,
    timings: &mut PhaseTimings,
) -> (Vec<Vector6<f64>>, Vec<Vector3<f64>>, f64) {
    let n_free = problem.n_free_frames;
    let t = Instant::now();
    let dx_points: Vec<Vector3<f64>> = lin
        .points
        .par_iter()
        .zip(&problem.point_free_frames)
        .enumerate()
        .map(|(p, (pl, frames))| {
            if !pl.free {
                return Vector3::zeros();
            }
            let Some(hpp_inv) = damped_hpp_inverse(&pl.hpp, mu) else {
                return Vector3::zeros();
            };
            let mut rhs = pl.bp;
            for (&fidx, hcp) in frames.iter().zip(lin.hcp(problem, p)) {
                let slot = problem.free_frame_slot[fidx as usize].expect("free") as usize;
                rhs -= hcp.transpose() * dx_frames[slot];
            }
            hpp_inv * rhs
        })
        .collect();

    let mut predicted = 0.0;
    for a in 0..n_free {
        let d = clamp_diag6(&frame_diag_raw[a]);
        let dx = dx_frames[a];
        let damped_dx = Vector6::from_iterator((0..6).map(|i| mu * d[i] * dx[i]));
        predicted += dx.dot(&(frame_bc_raw[a] + damped_dx));
    }
    for (p, pl) in lin.points.iter().enumerate() {
        if !pl.free {
            continue;
        }
        let d = clamp_diag3(&pl.hpp);
        let dx = dx_points[p];
        let damped_dx = Vector3::from_iterator((0..3).map(|i| mu * d[i] * dx[i]));
        predicted += dx.dot(&(pl.bp + damped_dx));
    }
    predicted *= 0.5;
    timings.backsub += t.elapsed();
    timings.trials += 1;

    (dx_frames, dx_points, predicted)
}

/// Residual-only (no Jacobian) total cost at a candidate `(poses, points)` —
/// `Σ ‖r_i‖²`, same convention as [`linearize`]'s `full_cost` /
/// `BundleAdjustment::cost()`. Used for a rejected trial's cheap re-check
/// (module doc: Jacobians are only re-evaluated after an accepted step).
///
/// `skip_behind_now` also leaves out the observations whose point is behind
/// its sensor at `problem`'s current state, which the linearization (and so
/// the LM model) left out. Otherwise a point that a step has carried behind a
/// camera and the next trial brings back adds a residual the model never
/// saw, hundreds of pixels near the camera centre.
fn evaluate_cost(
    problem: &Problem,
    poses: &[SE3],
    points: &[Point3<f64>],
    extras: &ExtraBlocks,
    loss: LossFunction,
    skip_behind_now: bool,
) -> f64 {
    let per_point: Vec<f64> = problem
        .points
        .par_iter()
        .enumerate()
        .map(|(p, _)| {
            let (start, end) = problem.point_obs_range[p];
            let mut c = 0.0;
            for o in &problem.obs[start as usize..end as usize] {
                if skip_behind_now {
                    let ((fx_now, ..), sensor_now) = problem.extras.camera_of(o, &problem.cameras);
                    let point_now =
                        problem.poses[o.frame_idx as usize].transform_point(&problem.points[p]);
                    // A panorama sees behind itself.
                    if fx_now != 0.0 && sensor_now.transform_point(&point_now).z <= 0.0 {
                        continue;
                    }
                }
                let pose = &poses[o.frame_idx as usize];
                let ((fx, fy, cx, cy), sensor_from_rig) = extras.camera_of(o, &problem.cameras);
                let point_rig = pose.transform_point(&points[p]);
                let point_sensor = sensor_from_rig.transform_point(&point_rig);
                let s = if fx == 0.0 {
                    let Some((predicted, _)) = project_equirectangular(cx, cy, &point_sensor)
                    else {
                        continue;
                    };
                    pixel_residual(fx, cx, predicted, o.xy).norm_squared()
                } else {
                    if point_sensor.z <= 0.0 {
                        continue;
                    }
                    let z_inv = point_sensor.z.recip();
                    let dx = fx * point_sensor.x * z_inv + cx - o.xy.x;
                    let dy = fy * point_sensor.y * z_inv + cy - o.xy.y;
                    dx * dx + dy * dy
                };
                // Residual-only re-check (no Jacobian) — `rho[0]` alone
                // reproduces the robustified cost, matching `linearize_point`'s
                // `cost += rho0` convention; `LossFunction::Trivial` short
                // circuits `rho[0] == s` without allocating the `[f64;3]`.
                c += if matches!(loss, LossFunction::Trivial) {
                    s
                } else {
                    evaluate_loss(loss, s)[0]
                };
            }
            c
        })
        .collect();
    // Serial fold over an already-materialized `Vec` — see `linearize`.
    per_point.iter().sum()
}

fn apply_step(
    problem: &Problem,
    dx_frames: &[Vector6<f64>],
    dx_points: &[Vector3<f64>],
) -> (Vec<SE3>, Vec<Point3<f64>>, ExtraBlocks) {
    let poses: Vec<SE3> = problem
        .poses
        .iter()
        .enumerate()
        .map(|(i, pose)| match problem.free_frame_slot[i] {
            Some(slot) => pose.compose(&SE3::exp(&dx_frames[slot as usize])),
            None => pose.clone(),
        })
        .collect();
    let points: Vec<Point3<f64>> = problem
        .points
        .iter()
        .enumerate()
        .map(|(p, xyz)| {
            if problem.point_fixed[p] {
                *xyz
            } else {
                xyz + dx_points[p]
            }
        })
        .collect();
    let extras = if problem.extras.len() > 0 {
        problem
            .extras
            .stepped(&dx_frames[problem.n_free_frames - problem.extras.len()..])
    } else {
        ExtraBlocks::default()
    };
    (poses, points, extras)
}

/// The LM loop's gradient stop, once the initial gradient is known:
/// `grad_norm` at or below the absolute [`GRADIENT_TOLERANCE`], or below
/// `rel` times the initial gradient.
fn gradient_converged(grad_norm: f64, rel: Option<f64>, initial: Option<f64>) -> bool {
    match (rel, initial) {
        (None, _) => grad_norm <= GRADIENT_TOLERANCE,
        (Some(rel), Some(g0)) => grad_norm <= rel * g0,
        (Some(_), None) => false,
    }
}

fn gradient_inf_norm(points_lin: &[PointLin], frame_bc_raw: &[Vector6<f64>]) -> f64 {
    let mut m: f64 = 0.0;
    for v in frame_bc_raw {
        for i in 0..6 {
            m = m.max(v[i].abs());
        }
    }
    for pl in points_lin {
        if !pl.free {
            continue;
        }
        for i in 0..3 {
            m = m.max(pl.bp[i].abs());
        }
    }
    m
}

/// Native `BaBackend` entry point: solves `ba`'s rig-reprojection problem in
/// place (writing the optimized poses/landmarks back into `ba.poses`/
/// `ba.landmarks`) and returns the same [`BaResult`]
/// `bundle_adjustment.rs::solve` already expects from the `Legacy` backend.
/// See module doc for the full algorithm.
///
/// `gradient_tolerance_rel` adds Ceres' **relative** `gradient_tolerance`
/// stopping criterion: terminate once `‖g‖` has fallen to or below
/// `rel * ‖g_initial‖` (the first iteration only records the initial norm, so
/// at least one step is attempted). COLMAP sets this to `1.0` for global BA
/// and `10.0` for
/// local BA (`incremental_pipeline.cc:236-283`/`:192-235`); real Ceres then
/// terminates after ~1-2 iterations, whereas this port's legacy absolute
/// `‖g‖_∞ <= 1e-4` runs to the iteration cap. Passing `None` preserves the
/// legacy absolute behaviour (used by every existing caller/test).
pub(crate) fn optimize_with_tolerance(
    ba: &mut BundleAdjustment,
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
) -> Result<BaResult, BaError> {
    let rig_obs: Vec<RigObs> = ba.rig_observations.par_iter().map(RigObs::from).collect();
    optimize_rig_obs(
        ba,
        &rig_obs,
        max_num_iterations,
        loss,
        gradient_tolerance_rel,
    )
}

/// [`optimize_with_tolerance`] with the observations given as `rig_obs`;
/// `ba.rig_observations` is ignored.
fn optimize_rig_obs(
    ba: &mut BundleAdjustment,
    rig_obs: &[RigObs],
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
) -> Result<BaResult, BaError> {
    let mut input = RigBaInput::from_ba(ba);
    let result = optimize_rig_input(
        &mut input,
        rig_obs,
        max_num_iterations,
        loss,
        gradient_tolerance_rel,
    )?;
    for (id, pose) in input.frame_ids.iter().zip(input.poses) {
        ba.poses
            .get_mut(id)
            .expect("frame id present")
            .world_to_camera = pose;
    }
    for (id, point) in input.point_ids.iter().zip(input.points) {
        *ba.landmarks.get_mut(id).expect("point id present") = point;
    }
    Ok(result)
}

/// One `BA_MEMORY` line: the main working sets of a solve, in MB, so a
/// large run's memory can be attributed without a heap profiler. Sizes
/// count elements, not allocator slack. `elimination` is the reduced
/// system's off-diagonal blocks (f64) built on every trial, `pcg_f32` their
/// f32 copy on the PCG path.
fn log_memory(problem: &Problem, rig_obs_bytes: usize, lin: &Linearization) {
    const MB: f64 = 1024.0 * 1024.0;
    // Zero while the index has not been built (an accelerator eliminates).
    let pair_idx: usize = problem
        .pairs
        .idx
        .get()
        .map_or(0, |idx| idx.iter().map(Vec::len).sum());
    let free_frames: usize = problem.point_free_frames.iter().map(Vec::len).sum();
    let lin_hcp = lin.hcp.len();
    let contributions = problem
        .contributions
        .as_ref()
        .map_or(0, |c| c.slot.len() * std::mem::size_of::<[u32; 2]>());
    let edges = problem.edges.len();
    eprintln!(
        "BA_MEMORY n_free_frames={} obs={} points={} edges={} rig_obs_mb={:.0} problem_obs_mb={:.0} point_free_frames_mb={:.0} point_pair_edge_idx_mb={:.0} (pairs={pair_idx}) contributions_mb={:.0} points_lin_mb={:.0} elimination_mb={:.0} pcg_f32_mb={:.0}",
        problem.n_free_frames,
        problem.obs.len(),
        problem.points.len(),
        edges,
        rig_obs_bytes as f64 / MB,
        std::mem::size_of_val(problem.obs.as_slice()) as f64 / MB,
        (free_frames * 4) as f64 / MB,
        (pair_idx * 4) as f64 / MB,
        contributions as f64 / MB,
        (std::mem::size_of_val(lin.points.as_slice())
            + lin_hcp * std::mem::size_of::<SMatrix<f64, 6, 3>>()) as f64
            / MB,
        (edges * std::mem::size_of::<Matrix6<f64>>()) as f64 / MB,
        (edges * std::mem::size_of::<nalgebra::Matrix6<f32>>()) as f64 / MB,
    );
}

/// [`optimize_with_tolerance`] on frames and points given as arrays
/// ([`RigBaInput`]) and observations `rig_obs`; the optimized poses and
/// points replace `input.poses` and `input.points`.
pub(crate) fn optimize_rig_input(
    input: &mut RigBaInput,
    rig_obs: &[RigObs],
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
) -> Result<BaResult, BaError> {
    let t_build = Instant::now();
    let mut problem = build_problem(input, rig_obs)?;
    optimize_problem(
        input,
        &mut problem,
        t_build,
        std::mem::size_of_val(rig_obs),
        max_num_iterations,
        loss,
        gradient_tolerance_rel,
    )
}

/// [`optimize_rig_input`] with the observations already grouped by point
/// ([`PointMajorObs`]), so the caller's own per-point pass replaces the
/// image-ordered [`RigObs`] list and its regrouping.
///
/// `extras` are the blocks refined beyond the frames ([`ExtraBlocks`]),
/// updated in place.
pub(crate) fn optimize_point_major(
    input: &mut RigBaInput,
    point_major: PointMajorObs,
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
    extras: &mut ExtraBlocks,
) -> Result<BaResult, BaError> {
    let t_build = Instant::now();
    let mut problem = assemble_problem(input, point_major, 0.0, extras.clone())?;
    let result = optimize_problem(
        input,
        &mut problem,
        t_build,
        0,
        max_num_iterations,
        loss,
        gradient_tolerance_rel,
    );
    *extras = problem.extras;
    result
}

/// The LM loop of [`optimize_rig_input`] on a built `problem`;
/// `rig_obs_bytes` only feeds the `BA_MEMORY` line.
fn optimize_problem(
    input: &mut RigBaInput,
    problem: &mut Problem,
    t_build: Instant,
    rig_obs_bytes: usize,
    max_num_iterations: usize,
    loss: LossFunction,
    gradient_tolerance_rel: Option<f64>,
) -> Result<BaResult, BaError> {
    let build_ms = t_build.elapsed().as_secs_f64() * 1e3;
    let mut timings = PhaseTimings::default();
    let mut apply_ms = 0.0;

    let (mut half_cost, mut lin, mut frame_diag_raw, mut frame_bc_raw, mut offdiag_raw) = {
        let t = Instant::now();
        let (full, pl, fd, fb, cl) = linearize(problem, loss, Vec::new(), true);
        timings.linearize += t.elapsed();
        timings.linearize_calls += 1;
        (0.5 * full, pl, fd, fb, cl)
    };
    let mut linearization_id = next_accelerator_id();
    let initial_cost = 2.0 * half_cost;
    if problem.n_free_frames >= problem.iterative_min_free_frames
        || std::env::var_os("VISLOC_BA_MEMORY").is_some()
    {
        log_memory(problem, rig_obs_bytes, &lin);
    }

    let mut radius = INITIAL_RADIUS;
    let mut decrease_factor = 2.0;
    let mut cache: Option<BlockSymbolic> = None;
    // Fill-reducing block order of the reduced camera system (its pattern,
    // `problem.edges`, is fixed for this solve). Natural order otherwise.
    let order: Option<Vec<u32>> = (problem.n_free_frames >= REORDER_MIN_FREE_FRAMES
        && problem.n_free_frames < problem.iterative_min_free_frames)
        .then(|| {
            let t = Instant::now();
            let mut adjacency = vec![Vec::new(); problem.n_free_frames];
            for &(a, b) in &problem.edges {
                adjacency[a as usize].push(b as usize);
                adjacency[b as usize].push(a as usize);
            }
            for neighbours in adjacency.iter_mut() {
                neighbours.sort_unstable();
            }
            let elimination = crate::reordering::fast_fill_reducing_block_order(&adjacency);
            let mut pos = vec![0u32; problem.n_free_frames];
            for (k, &block) in elimination.iter().enumerate() {
                pos[block] = k as u32;
            }
            timings.reorder += t.elapsed();
            pos
        });
    let mut iterations: Vec<BaIterationStats> = Vec::new();
    let mut trace_lines: Vec<String> = Vec::new();
    let mut converged = false;
    let mut initial_grad_norm: Option<f64> = None;
    let mut consecutive_invalid_steps = 0usize;

    for it in 0..max_num_iterations {
        let grad_norm = gradient_inf_norm(&lin.points, &frame_bc_raw);
        match gradient_tolerance_rel {
            // Legacy absolute criterion (unchanged).
            None => {
                if grad_norm <= GRADIENT_TOLERANCE {
                    converged = true;
                    break;
                }
            }
            // Ceres-relative criterion: stop once the gradient has decreased
            // by the requested factor; always attempt at least one step.
            Some(rel) => match initial_grad_norm {
                None => initial_grad_norm = Some(grad_norm),
                Some(g0) => {
                    // `<=` (not `<`): if the first step was rejected the
                    // gradient is unchanged, and a strict comparison would
                    // never fire and run to the iteration cap (observed as a
                    // single 865s global BA at 5k). Ceres stops after 1-2
                    // iterations with these tolerances.
                    if grad_norm <= rel * g0 {
                        converged = true;
                        break;
                    }
                }
            },
        }
        let mu = 1.0 / radius;
        let Some((dx_frames, dx_points, predicted)) = solve_step(
            problem,
            &lin,
            linearization_id,
            &problem.edges,
            &frame_diag_raw,
            &frame_bc_raw,
            offdiag_raw.as_deref(),
            mu,
            &mut cache,
            order.as_deref(),
            &mut timings,
        ) else {
            // An invalid step (Ceres `StepIsInvalid`): rejected like a step
            // that did not lower the cost.
            consecutive_invalid_steps += 1;
            trace_lines.push(format!(
                "BA_TRACE it={it} cost={:.6} radius={radius:.6e} grad_inf_norm={grad_norm:.6e} invalid_step",
                2.0 * half_cost
            ));
            if consecutive_invalid_steps >= MAX_CONSECUTIVE_INVALID_STEPS {
                eprintln!(
                    "BA_FAILED singular_system it={it} n_free_frames={} radius={radius:.6e} linsolve_ms={:.1} pcg_iterations={}",
                    problem.n_free_frames,
                    timings.linsolve.as_secs_f64() * 1e3,
                    timings.pcg_iterations,
                );
                for line in &trace_lines {
                    eprintln!("{line}");
                }
                return Err(BaError::SingularSystem);
            }
            radius /= decrease_factor;
            decrease_factor *= 2.0;
            continue;
        };
        consecutive_invalid_steps = 0;
        let t_apply = Instant::now();
        let (trial_poses, trial_points, trial_extras) = apply_step(problem, &dx_frames, &dx_points);
        apply_ms += t_apply.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        // A solve meant to converge takes many steps at a large trust
        // radius, where a point can cross a camera's plane and come back;
        // the one- or two-step incremental BAs keep the plain evaluation.
        let trial_half_cost = 0.5
            * evaluate_cost(
                problem,
                &trial_poses,
                &trial_points,
                &trial_extras,
                loss,
                gradient_tolerance_rel.is_none(),
            );
        timings.evaluate_cost += t.elapsed();
        let actual = half_cost - trial_half_cost;
        let rho = if predicted > 0.0 && predicted.is_finite() {
            actual / predicted
        } else {
            f64::NEG_INFINITY
        };

        let step_accepted = rho > MIN_RELATIVE_DECREASE;
        iterations.push(BaIterationStats {
            iteration: it,
            cost_before: 2.0 * half_cost,
            cost_after: 2.0
                * (if step_accepted {
                    trial_half_cost
                } else {
                    half_cost
                }),
            max_pose_step: dx_frames.iter().map(|v| v.norm()).fold(0.0, f64::max),
            max_landmark_step: dx_points.iter().map(|v| v.norm()).fold(0.0, f64::max),
            lambda: mu,
            step_accepted,
        });
        // Buffered (only flushed — see below — for solves with >= 30
        // iterations, per the task brief) per-trial diagnostic: cost, rho,
        // radius, gradient max-norm, accepted. `radius` here is the value
        // this trial was attempted *at* (i.e. before the accept/reject
        // update below), matching what determined this trial's `mu`.
        trace_lines.push(format!(
            "BA_TRACE it={it} cost={:.6} rho={rho:.6} radius={radius:.6e} grad_inf_norm={grad_norm:.6e} accepted={step_accepted}",
            2.0 * half_cost
        ));

        if step_accepted {
            // A solve meant to converge (the absolute gradient criterion,
            // `gradient_tolerance_rel == None`) also needs the model to
            // predict a small decrease: a step that barely passed the
            // acceptance test lowers the cost little however far the
            // optimum is (see `FUNCTION_TOLERANCE`).
            let decrease = match gradient_tolerance_rel {
                None => actual.max(predicted),
                Some(_) => actual,
            };
            let small_decrease = decrease < function_tolerance() * half_cost;
            problem.poses = trial_poses;
            problem.points = trial_points;
            problem.extras = trial_extras;
            radius /= (1.0 / 3.0_f64).max(1.0 - (2.0 * rho - 1.0).powi(3));
            decrease_factor = 2.0;
            if small_decrease {
                // Ceres stops on the function tolerance before evaluating
                // the Jacobian at the accepted point; so does this, saving
                // a linearization (about a sixth of a one-step global BA).
                half_cost = trial_half_cost;
                converged = true;
                break;
            }
            // Nearly every solve then stops on the gradient tolerance at the
            // top of the next iteration, which reads only `Jᵀr`. A
            // gradient-only pass (the same sums, in the same order, as a
            // full linearization) decides that first; the `JᵀJ` blocks are
            // built only when the solve goes on. The last iteration needs
            // neither.
            let t = Instant::now();
            let (g_full, g_lin, _, g_bc, _) = linearize(problem, loss, Vec::new(), false);
            timings.gradient += t.elapsed();
            let last = it + 1 == max_num_iterations;
            if last
                || gradient_converged(
                    gradient_inf_norm(&g_lin.points, &g_bc),
                    gradient_tolerance_rel,
                    initial_grad_norm,
                )
            {
                half_cost = 0.5 * g_full;
                converged = !last;
                break;
            }
            let t = Instant::now();
            let (full, pl, fd, fb, cl) =
                linearize(problem, loss, std::mem::take(&mut lin.hcp), true);
            timings.linearize += t.elapsed();
            timings.linearize_calls += 1;
            half_cost = 0.5 * full;
            lin = pl;
            linearization_id = next_accelerator_id();
            frame_diag_raw = fd;
            frame_bc_raw = fb;
            offdiag_raw = cl;
        } else {
            radius /= decrease_factor;
            decrease_factor *= 2.0;
        }
    }

    input.poses = problem.poses.clone();
    input.points = problem.points.clone();

    eprintln!(
        "BA_PHASES trials={} linearize_calls={} n_free_frames={} n_points={} edges={} shards={} linearize_ms={:.1} eliminate_ms={:.1} assemble_ms={:.1} linsolve_ms={:.1} backsub_ms={:.1} evaluate_cost_ms={:.1} build_ms={build_ms:.1} apply_ms={apply_ms:.1} reorder_ms={:.1} pcg_iterations={} gradient_ms={:.1}",
        timings.trials,
        timings.linearize_calls,
        problem.n_free_frames,
        problem.points.len(),
        problem.edges.len(),
        problem.shard_boundaries.len().saturating_sub(1),
        timings.linearize.as_secs_f64() * 1e3,
        timings.eliminate.as_secs_f64() * 1e3,
        timings.assemble.as_secs_f64() * 1e3,
        timings.linsolve.as_secs_f64() * 1e3,
        timings.backsub.as_secs_f64() * 1e3,
        timings.evaluate_cost.as_secs_f64() * 1e3,
        timings.reorder.as_secs_f64() * 1e3,
        timings.pcg_iterations,
        timings.gradient.as_secs_f64() * 1e3,
    );
    if iterations.len() >= 30 {
        for line in &trace_lines {
            eprintln!("{line}");
        }
    }

    Ok(BaResult {
        initial_cost,
        final_cost: 2.0 * half_cost,
        iterations,
        converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    use crate::bundle::{
        rig_residual_jacobians as legacy_rig_residual_jacobians, BaRigObservation,
    };
    use visloc_core::geometry::Pose;
    use visloc_core::types::Camera;

    fn sample_inputs() -> (f64, f64, f64, f64, SE3, SE3, Point3<f64>, Point2<f64>) {
        let fx = 480.0;
        let fy = 470.0;
        let cx = 320.0;
        let cy = 240.0;
        let sensor_from_rig = SE3::new(
            UnitQuaternion::from_euler_angles(0.01, 0.02, -0.01),
            Vector3::new(-0.12, 0.005, 0.01),
        );
        let pose = SE3::new(
            UnitQuaternion::from_euler_angles(0.05, -0.1, 0.03),
            Vector3::new(0.3, -0.2, 1.1),
        );
        let point = Point3::new(0.4, -0.15, 3.2);
        let xy = Point2::new(315.0, 238.0);
        (fx, fy, cx, cy, sensor_from_rig, pose, point, xy)
    }

    /// C2.5 task item 6(a): analytic Jacobian vs central finite differences.
    #[test]
    fn jacobian_matches_finite_difference() {
        let (fx, fy, cx, cy, sensor_from_rig, pose, point, xy) = sample_inputs();
        let (r0, j_pose, j_point) =
            residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &point, xy)
                .expect("projectable");

        let eps = 1e-6;
        for k in 0..6 {
            let mut d = Vector6::<f64>::zeros();
            d[k] = eps;
            let pose_plus = pose.compose(&SE3::exp(&d));
            let pose_minus = pose.compose(&SE3::exp(&(-d)));
            let r_plus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose_plus, &point, xy)
                    .unwrap()
                    .0;
            let r_minus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose_minus, &point, xy)
                    .unwrap()
                    .0;
            let numeric = (r_plus - r_minus) / (2.0 * eps);
            let analytic = j_pose.column(k);
            assert!(
                (numeric - analytic).norm() < 1e-5,
                "pose col {k}: analytic {analytic:?} vs numeric {numeric:?}"
            );
        }
        for k in 0..3 {
            let mut d = Vector3::<f64>::zeros();
            d[k] = eps;
            let r_plus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &(point + d), xy)
                    .unwrap()
                    .0;
            let r_minus =
                residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &(point - d), xy)
                    .unwrap()
                    .0;
            let numeric = (r_plus - r_minus) / (2.0 * eps);
            let analytic = j_point.column(k);
            assert!(
                (numeric - analytic).norm() < 1e-5,
                "point col {k}: analytic {analytic:?} vs numeric {numeric:?}"
            );
        }
        assert!(r0.norm() > 0.0, "sample residual should be nonzero");
    }

    /// The equirectangular branch: a point behind the sensor projects, the
    /// Jacobians match central differences, and a prediction across the
    /// panorama's left/right seam from its observation leaves a small
    /// residual.
    #[test]
    fn equirectangular_jacobians_match_finite_difference() {
        let (_, _, _, _, sensor_from_rig, pose, _, _) = sample_inputs();
        let (fx, fy, cx, cy) = equirectangular_intrinsics(2048.0, 1024.0);
        // Behind the sensor (z < 0) and off the seam.
        let point = Point3::new(1.4, -0.6, -2.5);
        let point_sensor = sensor_from_rig.transform_point(&pose.transform_point(&point));
        assert!(point_sensor.z < 0.0);
        let (predicted, _) = project_sensor_point((fx, fy, cx, cy), &point_sensor).unwrap();
        let xy = predicted + Vector2::new(3.0, -2.0);
        let residual = |sensor: &SE3, pose: &SE3, point: &Point3<f64>| {
            residual_and_jacobians(fx, fy, cx, cy, sensor, pose, point, xy).expect("projectable")
        };
        let (r0, j_pose, j_point) = residual(&sensor_from_rig, &pose, &point);
        assert!((r0 - Vector2::new(-3.0, 2.0)).norm() < 1e-9);
        let eps = 1e-6;
        let j_sensor = sensor_jacobian(
            (fx, fy, cx, cy),
            &sensor_from_rig,
            &pose.transform_point(&point),
        );
        for k in 0..6 {
            let mut d = Vector6::<f64>::zeros();
            d[k] = eps;
            let numeric = (residual(&sensor_from_rig, &pose.compose(&SE3::exp(&d)), &point).0
                - residual(&sensor_from_rig, &pose.compose(&SE3::exp(&(-d))), &point).0)
                / (2.0 * eps);
            assert!((numeric - j_pose.column(k)).norm() < 1e-4, "pose col {k}");
            let numeric = (residual(&sensor_from_rig.compose(&SE3::exp(&d)), &pose, &point).0
                - residual(&sensor_from_rig.compose(&SE3::exp(&(-d))), &pose, &point).0)
                / (2.0 * eps);
            assert!(
                (numeric - j_sensor.column(k)).norm() < 1e-4,
                "sensor col {k}"
            );
        }
        for k in 0..3 {
            let mut d = Vector3::<f64>::zeros();
            d[k] = eps;
            let numeric = (residual(&sensor_from_rig, &pose, &(point + d)).0
                - residual(&sensor_from_rig, &pose, &(point - d)).0)
                / (2.0 * eps);
            assert!((numeric - j_point.column(k)).norm() < 1e-4, "point col {k}");
        }
        // Predicted just right of the seam (x near 0), observed 2 px to
        // its left (x near 2048).
        let seam = Point3::new(-1.0e-3, 0.2, -1.0);
        let (at_seam, _) = project_sensor_point((fx, fy, cx, cy), &seam).unwrap();
        assert!(at_seam.x < 1.0, "{at_seam:?}");
        let r = pixel_residual(
            fx,
            cx,
            at_seam,
            Point2::new(at_seam.x + 2048.0 - 2.0, at_seam.y),
        );
        assert!((r.x - 2.0).abs() < 1e-9, "{r:?}");
    }

    /// C2.5 task item 6(a) (second half): this module's independent
    /// re-derivation agrees with `bundle.rs`'s already-exercised
    /// `rig_residual_jacobians` (see module doc).
    #[test]
    fn matches_bundle_rs_formula() {
        let (fx, fy, cx, cy, sensor_from_rig, pose, point, xy) = sample_inputs();
        let camera = Camera::pinhole(1, 640, 480, fx, fy, cx, cy);
        let legacy_pose = Pose {
            world_to_camera: pose.clone(),
        };
        let obs = BaRigObservation {
            keyframe_id: 0,
            landmark_id: 0,
            xy,
            camera,
            sensor_from_rig: sensor_from_rig.clone(),
        };
        let (r_legacy, j_pose_legacy, j_point_legacy) =
            legacy_rig_residual_jacobians(&obs, &legacy_pose, &point).expect("projectable");
        let (r_native, j_pose_native, j_point_native) =
            residual_and_jacobians(fx, fy, cx, cy, &sensor_from_rig, &pose, &point, xy)
                .expect("projectable");

        assert!((r_legacy - r_native).norm() < 1e-12);
        assert!((j_pose_legacy - j_pose_native).norm() < 1e-12);
        assert!((j_point_legacy - j_point_native).norm() < 1e-12);
    }

    fn build_synthetic_recon_for_ba() -> (
        crate::colmap_incremental::reconstruction::Reconstruction,
        super::super::bundle_adjustment::BundleAdjustmentConfig,
    ) {
        use super::super::bundle_adjustment::BundleAdjustmentConfig;
        use crate::colmap_incremental::pipeline::reconstruction_from_cache;
        use crate::colmap_incremental::reconstruction::TrackElement;
        use crate::colmap_incremental::test_support::build_synthetic_rig_scene;

        let scene = build_synthetic_rig_scene(5, 3);
        let mut recon = reconstruction_from_cache(&scene.db);
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }
        let mut point3d_ids = Vec::new();
        for (j, gt_xyz) in scene.ground_truth_points.iter().enumerate() {
            let mut track = Vec::new();
            for &(i1, i2) in &scene.images_per_frame {
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: j,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: j,
                });
            }
            point3d_ids.push(recon.add_point3d(*gt_xyz, track));
        }
        let anchors = [0u64, (scene.images_per_frame.len() - 1) as u64];
        let noise = Vector3::new(0.03, -0.02, 0.015);
        for &frame_id in scene.ground_truth_rig_from_world.keys() {
            if anchors.contains(&frame_id) {
                continue;
            }
            let mut pose = recon.frame(frame_id).rig_from_world().clone();
            pose.translation += noise;
            recon.frame_mut(frame_id).set_rig_from_world(pose);
        }
        for &pid in &point3d_ids {
            let xyz = recon.point3d(pid).xyz;
            recon.point3d_mut(pid).xyz = xyz + Vector3::new(0.02, -0.015, 0.01);
        }

        let mut config = BundleAdjustmentConfig::new();
        for &(i1, i2) in &scene.images_per_frame {
            config.add_image(i1);
            config.add_image(i2);
        }
        for &pid in &point3d_ids {
            config.add_variable_point(pid);
        }
        for &frame_id in &anchors {
            config.set_constant_rig_from_world_pose(frame_id);
        }
        (recon, config)
    }

    /// C2.5 perf-regression fix follow-up: profile `optimize` on a
    /// synthetic problem sized like the tier-1000 real-data global BA call
    /// that measured 2.9s/iteration (`frames=333 obs=179623 landmarks=2541`,
    /// `BA_SOLVE` log line from the killed run) — `#[ignore]`d (prints to
    /// stderr, run explicitly: `cargo test --release -p visloc-slam --lib
    /// rig_ba_solver::tests::profile_synthetic_large_problem -- --ignored
    /// --nocapture`).
    ///
    /// The task's <=100ms/iteration target at this size was **not** fully
    /// met: the O(track_len²)-per-point allocation/`BTreeMap` blowup that
    /// caused the original 2.9s/iteration is fixed (per-point elimination
    /// no longer allocates or touches a map; the frame-pair *pattern* is
    /// computed once per `optimize()` call, not once per accepted step; the
    /// O(pairs) work is sharded across 8 threads with a fixed, deterministic
    /// reduction — see `build_reduced_system_pattern` and
    /// `eliminate_and_accumulate`'s docs for the two fixes in order), taking
    /// per-iteration cost from ~2.9s to ~0.13-0.14s (~21x) on this
    /// synthetic problem — but ~140ms measured here is still ~40% over the
    /// 100ms target. This synthetic case's 2 cameras × 70-frame span gives
    /// every point ~140 observations across ~71 *distinct frames* (≈2485
    /// pairs/point); the real killed run's `obs/point` ratio (70.7) is
    /// consistent with a *shorter* per-point frame span (its 2 cameras
    /// would give ~35 frames/point for the same ratio, ≈595 pairs/point,
    /// ~4x fewer pairs) — so this synthetic profile is plausibly
    /// pessimistic relative to the real corridor tracks it was sized to
    /// resemble, not a confirmed shortfall against them. The asserted bound
    /// below is therefore the actually-measured value (with headroom) —
    /// a regression guard, not a claim the 100ms target is met — and the
    /// real tier-1000 rerun (§ task item 3) is the authoritative check.
    #[test]
    #[ignore]
    fn profile_synthetic_large_problem() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(300, 2500, 70);
        // Perturb every free pose/point so the solve actually has work to
        // do (built from exact ground-truth projections, so an
        // unperturbed solve converges in 0 iterations and never exercises
        // the LM loop at all).
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            pose.world_to_camera.translation += Vector3::new(0.02, -0.015, 0.01);
        }
        for xyz in ba.landmarks.values_mut() {
            *xyz += Vector3::new(0.015, 0.01, -0.02);
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_large_problem: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
        assert!(
            per_iter_ms <= 200.0,
            "per-iteration cost {per_iter_ms:.1}ms regressed past the measured ~140ms baseline \
             (task target is 100ms; see this test's doc comment for why 200ms is the guard here)"
        );
    }

    /// Lead-directed follow-up: profile at the larger size the Lead's own
    /// synthetic run used (~440 frames / 2.8k points / 255k observations,
    /// measured 1.08s/iteration pre-fix), target <=150ms/iteration at 8
    /// threads. `#[ignore]`d — run explicitly: `cargo test --release -p
    /// visloc-slam --lib rig_ba_solver::tests::profile_synthetic_440f
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_synthetic_440f() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(440, 2800, 45);
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            pose.world_to_camera.translation += Vector3::new(0.02, -0.015, 0.01);
        }
        for xyz in ba.landmarks.values_mut() {
            *xyz += Vector3::new(0.015, 0.01, -0.02);
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_440f: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
        assert!(
            per_iter_ms <= 150.0,
            "per-iteration cost {per_iter_ms:.1}ms exceeds the 150ms/iteration target at 440f/2800pt"
        );
    }

    /// Reproduces the real tier-1000 run's *largest* global-BA window
    /// (`BA_PHASES` grep: `n_free_frames≈846 n_points≈2900`, `eliminate_ms`
    /// ≈600ms/trial, `linsolve_ms`≈371ms/trial ⇒ ≈1s/iteration, vs this
    /// same-frame-count synthetic case's much smaller `edges_len` — see
    /// `elimination_shard_count`'s doc for why that gap is believed to be
    /// the previous 256MB shard-memory cap collapsing 8 shards down to ~2
    /// on a wide `edges_len`). `#[ignore]`d — run explicitly: `cargo test
    /// --release -p visloc-slam --lib rig_ba_solver::tests::profile_synthetic_846f
    /// -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn profile_synthetic_846f() {
        use crate::colmap_incremental::test_support::build_large_synthetic_ba_problem;

        let mut ba = build_large_synthetic_ba_problem(846, 2900, 200);
        // Harder, non-uniform (per-frame-varying, includes rotation) noise
        // than the other profiling tests' plain uniform shift: a uniform
        // shift of every free pose by the same vector is largely gauge/
        // structure-absorbed in a couple of LM steps and converges fast (as
        // the other tests here show, 7-9 iterations even at similar scale),
        // which is not representative of the real run's harder-to-correct,
        // per-frame-varying incremental drift that hits the 50-iteration
        // cap. `id` seeds a simple deterministic pseudo-random perturbation
        // (no RNG dependency) so every frame gets a different, harder-to-
        // cancel error.
        let fixed_poses = ba.fixed_poses.clone();
        for (&id, pose) in ba.poses.iter_mut() {
            if fixed_poses.contains(&id) {
                continue;
            }
            let s = id as f64;
            pose.world_to_camera.translation += Vector3::new(
                0.02 * (s * 0.7).sin(),
                0.015 * (s * 1.3).cos(),
                0.012 * (s * 0.4).sin(),
            );
            let d = UnitQuaternion::from_euler_angles(
                0.004 * (s * 0.9).cos(),
                0.004 * (s * 0.5).sin(),
                0.004 * (s * 1.1).cos(),
            );
            pose.world_to_camera.rotation = d * pose.world_to_camera.rotation;
        }
        for (id, xyz) in ba.landmarks.iter_mut() {
            let s = *id as f64;
            *xyz += Vector3::new(
                0.015 * (s * 0.6).sin(),
                0.015 * (s * 1.7).cos(),
                0.015 * (s * 0.3).sin(),
            );
        }
        eprintln!(
            "synthetic problem: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .expect("thread pool");
        let started = Instant::now();
        let result = pool
            .install(|| optimize_with_tolerance(&mut ba, 50, LossFunction::Trivial, None))
            .expect("synthetic BA solve should succeed");
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let per_iter_ms = elapsed_ms / result.iterations.len().max(1) as f64;
        eprintln!(
            "profile_synthetic_846f: iterations={} total_ms={elapsed_ms:.1} per_iter_ms={per_iter_ms:.1}",
            result.iterations.len()
        );
    }

    /// SmallCity-shaped global BA (the shape that dominates the mapper on
    /// large scenes): 5,500 frames driving down a street, 950k points each
    /// seen from ~8 frames spread over up to a few hundred frames, so the
    /// reduced camera system has ~2M off-diagonal blocks and runs PCG.
    /// `VISLOC_BENCH_SCALE` (default 1) scales frames and points.
    /// `#[ignore]`d — run explicitly: `cargo test --release -p visloc-slam
    /// --lib rig_ba_solver::tests::profile_synthetic_city -- --ignored
    /// --nocapture`, and read its `BA_PHASES` line.
    #[test]
    #[ignore]
    fn profile_synthetic_city() {
        let scale: f64 = std::env::var("VISLOC_BENCH_SCALE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let num_frames = (5500.0 * scale) as usize;
        let num_points = (950_000.0 * scale) as usize;
        let camera = Camera::pinhole(1, 1024, 768, 487.0, 487.0, 512.0, 384.0);
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut ba = BundleAdjustment::new(camera.clone());
        // Frame k sits at z = 0.5 k looking down +z, swaying a little.
        let poses: Vec<SE3> = (0..num_frames)
            .map(|k| {
                let z = 0.5 * k as f64;
                let rotation = UnitQuaternion::from_euler_angles(0.0, 0.05 * (z * 0.01).sin(), 0.0);
                let centre = Vector3::new(0.8 * (z * 0.02).sin(), 0.0, z);
                SE3::new(rotation, -(rotation * centre))
            })
            .collect();
        for (k, pose) in poses.iter().enumerate() {
            let mut noisy = pose.clone();
            if k > 0 {
                noisy.translation += Vector3::new(0.01, -0.008, 0.012);
            }
            ba.add_pose(
                k as u64,
                Pose {
                    world_to_camera: noisy,
                },
            );
        }
        ba.fix_pose(0);
        let max_z = 0.5 * num_frames as f64;
        for j in 0..num_points {
            let point = Point3::new(
                (rnd() - 0.5) * 40.0,
                (rnd() - 0.5) * 12.0,
                rnd() * (max_z + 60.0) - 10.0,
            );
            // Frames that see it in front of them and inside the image.
            let first = ((point.z - 150.0) / 0.5).max(0.0) as usize;
            let last = (((point.z - 2.0) / 0.5).max(0.0) as usize).min(num_frames);
            if last <= first + 1 {
                continue;
            }
            let mut observed = 0;
            for _ in 0..8 {
                let k = first + (rnd() * (last - first) as f64) as usize;
                let p_cam = poses[k].transform_point(&point);
                if p_cam.z <= 1.0 {
                    continue;
                }
                let Some(xy) = camera.project(&p_cam) else {
                    continue;
                };
                if xy.x < 0.0 || xy.x >= 1024.0 || xy.y < 0.0 || xy.y >= 768.0 {
                    continue;
                }
                if observed == 0 {
                    ba.add_landmark(j as u64, point + Vector3::new(0.02, 0.01, -0.02));
                }
                observed += 1;
                ba.add_rig_observation(BaRigObservation {
                    keyframe_id: k as u64,
                    landmark_id: j as u64,
                    xy: xy + nalgebra::Vector2::new(rnd() - 0.5, rnd() - 0.5),
                    camera: camera.clone(),
                    sensor_from_rig: SE3::identity(),
                });
            }
        }
        eprintln!(
            "synthetic city: poses={} landmarks={} rig_observations={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.rig_observations.len()
        );
        let started = Instant::now();
        let result = optimize_with_tolerance(&mut ba, 4, LossFunction::Trivial, Some(1.0))
            .expect("synthetic city BA");
        eprintln!(
            "profile_synthetic_city: iterations={} total_ms={:.1}",
            result.iterations.len(),
            started.elapsed().as_secs_f64() * 1e3
        );
    }

    /// C2.5 task item 6(e): the whole pipeline (build_problem → linearize →
    /// solve_step, including the block-Cholesky reduced-camera solve) is
    /// bit-identical at `RAYON_NUM_THREADS=1` vs `8`.
    #[test]
    fn deterministic_across_thread_counts() {
        use super::super::bundle_adjustment::{BundleAdjustmentOptions, Gauge};

        let run = |threads: usize| {
            let (mut recon, mut config) = build_synthetic_recon_for_ba();
            config.fix_gauge(Gauge::Unspecified);
            let mut options = BundleAdjustmentOptions::global();
            options.max_num_iterations = 12;
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let ok = pool
                .install(|| super::super::bundle_adjustment::solve(&options, &config, &mut recon));
            assert!(ok, "BA solve failed at {threads} threads");
            recon
        };

        let recon1 = run(1);
        let recon8 = run(8);

        for &frame_id in recon1.reg_frame_ids() {
            let a = recon1.frame(frame_id).rig_from_world();
            let b = recon8.frame(frame_id).rig_from_world();
            assert_eq!(a, b, "frame {frame_id} pose differs across thread counts");
        }
    }

    /// C2.7 task item: `rho'(s)`/`rho''(s)` match central finite differences
    /// of `rho(s)`/`rho'(s)` respectively, for both `SoftL1` and `Cauchy`,
    /// across a range of `s` (including small and large squared residuals).
    #[test]
    fn loss_derivatives_match_finite_difference() {
        let eps = 1.0e-6;
        for loss in [LossFunction::SoftL1(1.7), LossFunction::Cauchy(2.3)] {
            for &s in &[1.0e-3, 0.1, 1.0, 4.0, 25.0, 100.0] {
                let rho = evaluate_loss(loss, s);
                let rho_at = |s: f64| evaluate_loss(loss, s);

                let fd_rho1 = (rho_at(s + eps)[0] - rho_at(s - eps)[0]) / (2.0 * eps);
                let rho1_tol = 1.0e-5 * rho[1].abs().max(1.0);
                assert!(
                    (fd_rho1 - rho[1]).abs() < rho1_tol,
                    "{loss:?} s={s}: rho'(s) analytic={} fd={fd_rho1}",
                    rho[1]
                );

                let fd_rho2 = (rho_at(s + eps)[1] - rho_at(s - eps)[1]) / (2.0 * eps);
                let rho2_tol = 1.0e-5 * rho[2].abs().max(1.0);
                assert!(
                    (fd_rho2 - rho[2]).abs() < rho2_tol,
                    "{loss:?} s={s}: rho''(s) analytic={} fd={fd_rho2}",
                    rho[2]
                );
            }
        }
    }

    /// C2.7 task item: `Corrector` output equals a hand-computed case
    /// (SoftLOneLoss(a=1) at `r=(3,4)`, so `s=25`) — the *common* branch
    /// (`rho''(s) <= 0`, always true for `SoftL1`/`Cauchy`, see
    /// `evaluate_loss`'s doc), where `Corrector` reduces to a uniform
    /// `sqrt(rho'(s))` rescale of both the residual and every Jacobian
    /// block, no curvature term.
    #[test]
    fn corrector_matches_hand_computed_common_branch_case() {
        // rho(s) = 2*(sqrt(1+s) - 1), rho'(s) = 1/sqrt(1+s), rho''(s) =
        // -rho'(s)/(2*(1+s)) for SoftLOneLoss(a=1) (b=c=1) — computed here
        // independently of `evaluate_loss`.
        let s = 25.0_f64;
        let sqrt26 = 26.0_f64.sqrt();
        let hand_rho0 = 2.0 * (sqrt26 - 1.0);
        let hand_rho1 = 1.0 / sqrt26;
        let hand_rho2 = -hand_rho1 / (2.0 * 26.0);

        let rho = evaluate_loss(LossFunction::SoftL1(1.0), s);
        assert!((rho[0] - hand_rho0).abs() < 1.0e-12);
        assert!((rho[1] - hand_rho1).abs() < 1.0e-12);
        assert!((rho[2] - hand_rho2).abs() < 1.0e-12);
        assert!(rho[2] <= 0.0, "expected the common (no-curvature) branch");

        let sqrt_rho1 = hand_rho1.sqrt();
        let corrector = Corrector::new(s, rho);
        assert!((corrector.sqrt_rho1 - sqrt_rho1).abs() < 1.0e-12);
        assert!((corrector.residual_scaling - sqrt_rho1).abs() < 1.0e-12);
        assert_eq!(corrector.alpha_sq_norm, 0.0);

        let mut r = Vector2::new(3.0_f64, 4.0);
        let orig_j = SMatrix::<f64, 2, 6>::from_row_slice(&[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.,
        ]);
        let mut j = orig_j;
        corrector.correct_jacobian(&r, &mut j);
        for row in 0..2 {
            for col in 0..6 {
                let expected = orig_j[(row, col)] * sqrt_rho1;
                assert!((j[(row, col)] - expected).abs() < 1.0e-10);
            }
        }
        corrector.correct_residual(&mut r);
        assert!((r.x - 3.0 * sqrt_rho1).abs() < 1.0e-12);
        assert!((r.y - 4.0 * sqrt_rho1).abs() < 1.0e-12);
    }

    /// C2.7 task item: `Corrector` output equals a hand-computed case for the
    /// *rare* branch (`rho''(s) > 0`) — never produced by `SoftL1`/`Cauchy`
    /// (both always have `rho'' <= 0`, see `evaluate_loss`'s doc), but
    /// `Corrector::new` must still implement Ceres' generic
    /// `corrector.cc:93-109` curvature-correction math faithfully. Uses a
    /// synthetic `rho = [_, 2.0, 0.5]` at `s=4.0` (`rho[0]` is unused by
    /// `Corrector`'s own math, only `rho[1]`/`rho[2]` matter).
    #[test]
    fn corrector_matches_hand_computed_rare_branch_case() {
        let s = 4.0_f64;
        let rho = [0.0_f64, 2.0, 0.5];
        // corrector.cc:100-109.
        let d = 1.0 + 2.0 * s * rho[2] / rho[1]; // 1 + 2*4*0.5/2 = 3.0
        let alpha = 1.0 - d.sqrt(); // 1 - sqrt(3)
        let hand_sqrt_rho1 = rho[1].sqrt(); // sqrt(2)
        let hand_residual_scaling = hand_sqrt_rho1 / (1.0 - alpha);
        let hand_alpha_sq_norm = alpha / s;

        let corrector = Corrector::new(s, rho);
        assert!((corrector.sqrt_rho1 - hand_sqrt_rho1).abs() < 1.0e-12);
        assert!((corrector.residual_scaling - hand_residual_scaling).abs() < 1.0e-12);
        assert!((corrector.alpha_sq_norm - hand_alpha_sq_norm).abs() < 1.0e-12);
        assert_ne!(
            corrector.alpha_sq_norm, 0.0,
            "expected the curvature branch"
        );

        // r = (0, 2), so r.norm_squared() == s == 4, consistent with the
        // `Corrector` this `r` is used with.
        let mut r = Vector2::new(0.0_f64, 2.0);
        let orig_j = SMatrix::<f64, 2, 1>::new(1.0, 2.0);
        let mut j = orig_j;
        corrector.correct_jacobian(&r, &mut j);
        // r_transpose_j = r . j_col0 = 0*1 + 2*2 = 4.
        let r_transpose_j = 4.0_f64;
        let expected_j0 =
            hand_sqrt_rho1 * (orig_j[(0, 0)] - hand_alpha_sq_norm * 0.0 * r_transpose_j);
        let expected_j1 =
            hand_sqrt_rho1 * (orig_j[(1, 0)] - hand_alpha_sq_norm * 2.0 * r_transpose_j);
        assert!((j[(0, 0)] - expected_j0).abs() < 1.0e-10);
        assert!((j[(1, 0)] - expected_j1).abs() < 1.0e-10);

        corrector.correct_residual(&mut r);
        assert!((r.x - 0.0 * hand_residual_scaling).abs() < 1.0e-12);
        assert!((r.y - 2.0 * hand_residual_scaling).abs() < 1.0e-10);
    }

    /// Deterministic-across-thread-counts, repeated for a non-trivial loss
    /// (`SoftL1`) — the `Corrector` path (per-observation branch on
    /// `alpha_sq_norm`, `rayon`-parallel across points in `linearize`) must
    /// stay bit-identical at any thread count too, not just the
    /// `LossFunction::Trivial` fast path `deterministic_across_thread_counts`
    /// already covers.
    #[test]
    fn deterministic_across_thread_counts_with_soft_l1_loss() {
        use super::super::bundle_adjustment::{BundleAdjustmentOptions, Gauge};

        let run = |threads: usize| {
            let (mut recon, mut config) = build_synthetic_recon_for_ba();
            config.fix_gauge(Gauge::Unspecified);
            let mut options = BundleAdjustmentOptions::global();
            options.max_num_iterations = 12;
            options.loss_function = LossFunction::SoftL1(0.5);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let ok = pool
                .install(|| super::super::bundle_adjustment::solve(&options, &config, &mut recon));
            assert!(ok, "BA solve failed at {threads} threads");
            recon
        };

        let recon1 = run(1);
        let recon8 = run(8);

        for &frame_id in recon1.reg_frame_ids() {
            let a = recon1.frame(frame_id).rig_from_world();
            let b = recon8.frame(frame_id).rig_from_world();
            assert_eq!(
                a, b,
                "frame {frame_id} pose differs across thread counts (SoftL1)"
            );
        }
    }

    /// Builds a `num_frames`-frame synthetic rig reconstruction (all
    /// `ground_truth_points` visible from every frame, per
    /// `build_synthetic_rig_scene`'s doc) with ~10% of its 2D observations
    /// corrupted into gross pixel outliers (deterministic: every 10th
    /// observation in `(frame, camera, point)` iteration order), frame poses
    /// and point positions perturbed from ground truth identically to
    /// `build_synthetic_recon_for_ba`, and every point requested variable —
    /// used by `soft_l1_recovers_gross_outlier_scene_trivial_does_not` to
    /// compare `LossFunction::SoftL1`'s outlier robustness against
    /// `LossFunction::Trivial`'s from an identical starting state.
    fn build_outlier_recon_for_ba(
        num_frames: usize,
    ) -> (
        crate::colmap_incremental::reconstruction::Reconstruction,
        super::super::bundle_adjustment::BundleAdjustmentConfig,
        Vec<u64>,
        std::collections::BTreeMap<u64, Point3<f64>>,
    ) {
        use super::super::bundle_adjustment::BundleAdjustmentConfig;
        use crate::colmap_incremental::pipeline::reconstruction_from_cache;
        use crate::colmap_incremental::reconstruction::TrackElement;
        use crate::colmap_incremental::test_support::build_synthetic_rig_scene;
        use std::collections::BTreeMap;

        let scene = build_synthetic_rig_scene(num_frames, num_frames.saturating_sub(1));
        let mut recon = reconstruction_from_cache(&scene.db);
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }

        let mut point3d_ids = Vec::new();
        let mut gt_by_id = BTreeMap::new();
        for (j, gt_xyz) in scene.ground_truth_points.iter().enumerate() {
            let mut track = Vec::new();
            for &(i1, i2) in &scene.images_per_frame {
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: j,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: j,
                });
            }
            let pid = recon.add_point3d(*gt_xyz, track);
            point3d_ids.push(pid);
            gt_by_id.insert(pid, *gt_xyz);
        }

        let mut obs_idx = 0usize;
        for &(i1, i2) in &scene.images_per_frame {
            for &iid in &[i1, i2] {
                for j in 0..scene.ground_truth_points.len() {
                    if obs_idx.is_multiple_of(10) {
                        let image = recon.image_mut(iid);
                        image.points2d[j].xy.x += 400.0;
                        image.points2d[j].xy.y -= 320.0;
                    }
                    obs_idx += 1;
                }
            }
        }

        let anchors = [0u64, (scene.images_per_frame.len() - 1) as u64];
        let noise = Vector3::new(0.03, -0.02, 0.015);
        for &frame_id in scene.ground_truth_rig_from_world.keys() {
            if anchors.contains(&frame_id) {
                continue;
            }
            let mut pose = recon.frame(frame_id).rig_from_world().clone();
            pose.translation += noise;
            recon.frame_mut(frame_id).set_rig_from_world(pose);
        }
        for &pid in &point3d_ids {
            let xyz = recon.point3d(pid).xyz;
            recon.point3d_mut(pid).xyz = xyz + Vector3::new(0.02, -0.015, 0.01);
        }

        let mut config = BundleAdjustmentConfig::new();
        for &(i1, i2) in &scene.images_per_frame {
            config.add_image(i1);
            config.add_image(i2);
        }
        for &pid in &point3d_ids {
            config.add_variable_point(pid);
        }
        for &frame_id in &anchors {
            config.set_constant_rig_from_world_pose(frame_id);
        }
        (recon, config, point3d_ids, gt_by_id)
    }

    /// C2.7 task item: on a scene with ~10% gross pixel-outlier
    /// observations, `SoftL1` recovers points close to ground truth while
    /// `Trivial` (dragged by the outliers) does not.
    #[test]
    fn soft_l1_recovers_gross_outlier_scene_trivial_does_not() {
        use super::super::bundle_adjustment::BundleAdjustmentOptions;

        let (mut recon_soft, config, point3d_ids, gt_by_id) = build_outlier_recon_for_ba(8);
        let mut recon_trivial = recon_soft.clone();

        let mut soft_options = BundleAdjustmentOptions::global();
        soft_options.max_num_iterations = 60;
        soft_options.loss_function = LossFunction::SoftL1(1.0);
        let mut trivial_options = soft_options;
        trivial_options.loss_function = LossFunction::Trivial;

        assert!(
            super::super::bundle_adjustment::solve(&soft_options, &config, &mut recon_soft),
            "SoftL1 solve failed"
        );
        assert!(
            super::super::bundle_adjustment::solve(&trivial_options, &config, &mut recon_trivial),
            "Trivial solve failed"
        );

        let median_point_error =
            |recon: &crate::colmap_incremental::reconstruction::Reconstruction| -> f64 {
                let mut errs: Vec<f64> = point3d_ids
                    .iter()
                    .map(|&pid| (recon.point3d(pid).xyz - gt_by_id[&pid]).norm())
                    .collect();
                errs.sort_by(|a, b| a.total_cmp(b));
                errs[errs.len() / 2]
            };

        let soft_err = median_point_error(&recon_soft);
        let trivial_err = median_point_error(&recon_trivial);
        assert!(
            soft_err < 0.02,
            "SoftL1 median point error too large: {soft_err}"
        );
        assert!(
            trivial_err > soft_err * 2.0,
            "expected Trivial (no outlier down-weighting) to be pulled off further than \
             SoftL1: soft={soft_err} trivial={trivial_err}"
        );
    }

    /// A 12-frame scene seen through `truth`, with frames 0 and 1 fixed (the
    /// gauge, scale included): observations, the true poses and points, and
    /// a perturbed start.
    fn intrinsics_scene(truth: [f64; 4]) -> (RigBaInput, PointMajorObs, Vec<Point3<f64>>) {
        let [fx, fy, cx, cy] = truth;
        let poses: Vec<SE3> = (0..12)
            .map(|i| {
                let t = i as f64 / 11.0 - 0.5;
                SE3::new(
                    UnitQuaternion::from_euler_angles(0.08 * t, -0.25 * t, 0.05 * t),
                    Vector3::new(-1.5 * t, 0.2 * t * t, 0.4 * t),
                )
            })
            .collect();
        let points: Vec<Point3<f64>> = (0..300)
            .map(|p| {
                let (a, b, c) = (p % 10, (p / 10) % 10, p / 100);
                Point3::new(
                    a as f64 * 0.4 - 1.8,
                    b as f64 * 0.3 - 1.4,
                    4.0 + c as f64 * 1.5 + 0.1 * a as f64,
                )
            })
            .collect();
        let mut obs = Vec::new();
        let mut ranges = Vec::new();
        for point in &points {
            let start = obs.len() as u32;
            for (f, pose) in poses.iter().enumerate() {
                let x = pose.transform_point(point);
                let xy = Point2::new(fx * x.x / x.z + cx, fy * x.y / x.z + cy);
                // Wrong intrinsics on the observations' camera: the camera
                // block must not read them.
                obs.push(Obs::new(f as u32, xy, 0));
            }
            ranges.push((start, obs.len() as u32));
        }
        let start_poses = poses
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                if i < 2 {
                    pose.clone()
                } else {
                    let d = Vector6::new(0.01, -0.01, 0.02, 0.002, -0.001, 0.0015);
                    pose.compose(&SE3::exp(&(d * (i as f64 / 12.0))))
                }
            })
            .collect();
        let input = RigBaInput {
            frame_ids: (0..12).collect(),
            frame_fixed: (0..12).map(|i| i < 2).collect(),
            poses: start_poses,
            point_ids: (0..points.len() as u64).collect(),
            point_fixed: vec![false; points.len()],
            points: points
                .iter()
                .map(|p| p + Vector3::new(0.02, -0.01, 0.03))
                .collect(),
        };
        (
            input,
            PointMajorObs {
                obs,
                point_obs_range: ranges,
                cameras: vec![ObsCamera::new((1.0, 1.0, 0.0, 0.0), SE3::identity())],
            },
            points,
        )
    }

    /// The fused PCG edge product is `(Bᵀ x_b, B x_a)`, for f64 and f32
    /// blocks alike.
    #[test]
    fn edge_product_matches_the_matrix_products() {
        let blk = Matrix6::<f64>::from_fn(|r, c| (r * 7 + c * 3) as f64 * 0.125 - 1.5);
        let xa = Vector6::new(0.5, -1.0, 2.0, 0.25, -0.75, 1.5);
        let xb = Vector6::new(-2.0, 1.0, 0.5, 3.0, -0.5, 0.125);
        let (to_a, to_b) = edge_product(blk.as_slice(), &xa, &xb);
        assert!((to_a - blk.tr_mul(&xb)).amax() < 1e-12);
        assert!((to_b - blk * xa).amax() < 1e-12);
        let blk32 = blk.cast::<f32>();
        let (to_a, to_b) = edge_product(blk32.as_slice(), &xa, &xb);
        assert!((to_a - blk.tr_mul(&xb)).amax() < 1e-12);
        assert!((to_b - blk * xa).amax() < 1e-12);
    }

    /// The chunked, parallel slot index lays its entries out like one serial
    /// scatter in point order.
    #[test]
    fn slot_index_matches_the_serial_scatter() {
        let n_frames = 300usize;
        // Every seventh frame fixed; the others numbered in order.
        let mut free_frame_slot = vec![None; n_frames];
        let mut n_slots = 0u32;
        for (f, slot) in free_frame_slot.iter_mut().enumerate() {
            if f % 7 != 3 {
                *slot = Some(n_slots);
                n_slots += 1;
            }
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |m: usize| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize % m
        };
        let point_free_frames: Vec<Vec<u32>> = (0..40_000)
            .map(|_| {
                let first = next(n_frames - 8);
                let mut frames: Vec<u32> = (0..1 + next(6))
                    .map(|_| (first + next(8)) as u32)
                    .filter(|&f| free_frame_slot[f as usize].is_some())
                    .collect();
                frames.sort_unstable();
                frames.dedup();
                frames
            })
            .collect();
        let n_slots = n_slots as usize;
        let mut slot_off = vec![0usize; n_slots + 1];
        let mut slot = vec![Vec::new(); n_slots];
        for (p, frames) in point_free_frames.iter().enumerate() {
            for (k, &f) in frames.iter().enumerate() {
                slot[free_frame_slot[f as usize].unwrap() as usize].push([p as u32, k as u32]);
            }
        }
        for s in 0..n_slots {
            slot_off[s + 1] = slot_off[s] + slot[s].len();
        }
        let (off, entries) = slot_index(&point_free_frames, &free_frame_slot, n_slots);
        assert!(point_free_frames.len() >= 1 << 15, "takes the chunked path");
        assert_eq!(off, slot_off);
        assert_eq!(entries, slot.concat());
    }

    /// The per-output elimination (iterative path, tight memory) and the
    /// sharded per-point loop add the same terms in the same order.
    #[test]
    fn elimination_paths_are_bit_identical() {
        let (input, point_major, _) = intrinsics_scene([500.0, 500.0, 322.0, 238.0]);
        let mut problem =
            assemble_problem(&input, point_major, 0.0, ExtraBlocks::default()).expect("problem");
        // Several shards, so the per-output path restarts partial sums.
        problem.shard_boundaries = balanced_shard_boundaries(&problem.point_free_frames, 3);
        let index = ContributionIndex::new(
            &problem.point_free_frames,
            &problem.free_frame_slot,
            &problem.edges,
            problem.n_free_frames,
            &problem.shard_boundaries,
        );
        let (_, lin, _, _, _) = linearize(&problem, LossFunction::Trivial, Vec::new(), true);
        problem.contributions = None;
        let sharded = eliminate_and_accumulate(&problem, &lin, problem.edges.len(), 1e-3);
        problem.contributions = Some(index);
        let by_output = eliminate_and_accumulate(&problem, &lin, problem.edges.len(), 1e-3);
        assert!(sharded.0.iter().any(|b| b.norm() > 0.0));
        assert_eq!(sharded, by_output);
    }

    /// A converging solve's trial cost leaves out an observation whose point
    /// is behind the camera at the current state (the linearization skipped
    /// it), even when the trial brings the point back in front.
    #[test]
    fn trial_cost_skips_observations_behind_the_camera_now() {
        let k = (500.0, 500.0, 320.0, 240.0);
        let project = |pose: &SE3, x: &Point3<f64>| {
            let q = pose.transform_point(x);
            Point2::new(k.0 * q.x / q.z + k.2, k.1 * q.y / q.z + k.3)
        };
        // Frame 1 sits at z = 1 looking back down the z axis.
        let back = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, std::f64::consts::PI, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let poses = vec![SE3::identity(), back];
        let now = Point3::new(0.1, 0.2, 3.0);
        let trial = Point3::new(0.1, 0.2, 0.5);
        assert!(poses[1].transform_point(&now).z < 0.0);
        assert!(poses[1].transform_point(&trial).z > 0.0);
        let xy0 = project(&poses[0], &now);
        let xy1 = Point2::new(300.0, 250.0);
        let input = RigBaInput {
            frame_ids: vec![0, 1],
            frame_fixed: vec![true, false],
            poses: poses.clone(),
            point_ids: vec![0],
            point_fixed: vec![false],
            points: vec![now],
        };
        let point_major = PointMajorObs {
            obs: vec![Obs::new(0, xy0, 0), Obs::new(1, xy1, 0)],
            point_obs_range: vec![(0, 2)],
            cameras: vec![ObsCamera::new(k, SE3::identity())],
        };
        let problem =
            assemble_problem(&input, point_major, 0.0, ExtraBlocks::default()).expect("problem");
        let extras = ExtraBlocks::default();
        let cost = |skip| {
            evaluate_cost(
                &problem,
                &poses,
                &[trial],
                &extras,
                LossFunction::Trivial,
                skip,
            )
        };
        let r0 = (project(&poses[0], &trial) - xy0).norm_squared();
        let r1 = (project(&poses[1], &trial) - xy1).norm_squared();
        assert!((cost(true) - r0).abs() < 1e-9 * r0);
        assert!((cost(false) - (r0 + r1)).abs() < 1e-9 * (r0 + r1));
    }

    #[test]
    fn camera_block_recovers_focal_and_principal_point() {
        let truth = [500.0, 500.0, 322.0, 238.0];
        let (mut input, point_major, points) = intrinsics_scene(truth);
        let mut extras = ExtraBlocks {
            camera: Some(SharedIntrinsics {
                params: [510.0, 510.0, 315.0, 245.0],
                refine_focal_length: true,
                refine_principal_point: true,
            }),
            sensors: Vec::new(),
        };
        let result = optimize_point_major(
            &mut input,
            point_major,
            100,
            LossFunction::Trivial,
            None,
            &mut extras,
        )
        .expect("solve");
        let camera = extras.camera.expect("camera");
        for (got, want) in camera.params.iter().zip(truth) {
            assert!(
                (got - want).abs() < 1e-4,
                "{:?} vs {truth:?}",
                camera.params
            );
        }
        assert!(result.final_cost < 1e-12, "cost {}", result.final_cost);
        for (got, want) in input.points.iter().zip(&points) {
            assert!((got - want).norm() < 1e-6);
        }
    }

    #[test]
    fn camera_block_recovers_intrinsics_on_the_iterative_path() {
        let truth = [500.0, 500.0, 322.0, 238.0];
        let (mut input, point_major, _) = intrinsics_scene(truth);
        let extras = ExtraBlocks {
            camera: Some(SharedIntrinsics {
                params: [510.0, 510.0, 315.0, 245.0],
                refine_focal_length: true,
                refine_principal_point: true,
            }),
            sensors: Vec::new(),
        };
        let mut problem = assemble_problem(&input, point_major, 0.0, extras).expect("problem");
        // Every solve takes the edge-list PCG, the camera as its extra node.
        problem.iterative_min_free_frames = 1;
        let result = optimize_problem(
            &mut input,
            &mut problem,
            Instant::now(),
            0,
            100,
            LossFunction::Trivial,
            None,
        )
        .expect("solve");
        let camera = problem.extras.camera.expect("camera");
        for (got, want) in camera.params.iter().zip(truth) {
            assert!(
                (got - want).abs() < 1e-3,
                "{:?} vs {truth:?}",
                camera.params
            );
        }
        assert!(result.final_cost < 1e-8, "cost {}", result.final_cost);
    }

    #[test]
    fn camera_block_keeps_fixed_parameters() {
        let truth = [500.0, 500.0, 322.0, 238.0];
        let (mut input, point_major, _) = intrinsics_scene(truth);
        let mut extras = ExtraBlocks {
            camera: Some(SharedIntrinsics {
                params: [510.0, 510.0, 322.0, 238.0],
                refine_focal_length: true,
                refine_principal_point: false,
            }),
            sensors: Vec::new(),
        };
        optimize_point_major(
            &mut input,
            point_major,
            100,
            LossFunction::Trivial,
            None,
            &mut extras,
        )
        .expect("solve");
        let camera = extras.camera.expect("camera");
        assert_eq!(camera.params[2..], [322.0, 238.0]);
        assert!(
            (camera.params[0] - 500.0).abs() < 1e-4,
            "{:?}",
            camera.params
        );
        assert_eq!(camera.params[0], camera.params[1]);
    }

    #[test]
    fn sensor_jacobian_matches_finite_difference() {
        let (fx, fy, cx, cy, sensor_from_rig, pose, point, xy) = sample_inputs();
        let point_rig = pose.transform_point(&point);
        let analytic = sensor_jacobian((fx, fy, cx, cy), &sensor_from_rig, &point_rig);
        let residual = |sensor: &SE3| {
            residual_and_jacobians(fx, fy, cx, cy, sensor, &pose, &point, xy)
                .expect("projectable")
                .0
        };
        let eps = 1e-6;
        for k in 0..6 {
            let mut d = Vector6::<f64>::zeros();
            d[k] = eps;
            let plus = residual(&sensor_from_rig.compose(&SE3::exp(&d)));
            let minus = residual(&sensor_from_rig.compose(&SE3::exp(&(-d))));
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (numeric - analytic.column(k)).norm() < 1e-5,
                "col {k}: {:?} vs {numeric:?}",
                analytic.column(k)
            );
        }
    }

    /// A 3-sensor rig over 10 frames (frames 0 and 1 fixed): sensor 0 is
    /// the reference, sensors 1 and 2 are refined from a perturbed start.
    /// Returns the input, observations and true `sensor_from_rig` of 1, 2.
    fn rig_scene() -> (RigBaInput, PointMajorObs, Vec<SE3>) {
        let k = (500.0, 500.0, 320.0, 240.0);
        let sensors = vec![
            SE3::new(
                UnitQuaternion::from_euler_angles(0.02, 0.15, -0.01),
                Vector3::new(-0.3, 0.02, 0.05),
            ),
            SE3::new(
                UnitQuaternion::from_euler_angles(-0.03, -0.12, 0.02),
                Vector3::new(0.35, -0.04, 0.0),
            ),
        ];
        let poses: Vec<SE3> = (0..10)
            .map(|i| {
                let t = i as f64 / 9.0 - 0.5;
                SE3::new(
                    UnitQuaternion::from_euler_angles(0.05 * t, -0.2 * t, 0.03 * t),
                    Vector3::new(-1.2 * t, 0.1 * t * t, 0.3 * t),
                )
            })
            .collect();
        let points: Vec<Point3<f64>> = (0..240)
            .map(|p| {
                let (a, b, c) = (p % 12, (p / 12) % 10, p / 120);
                Point3::new(
                    a as f64 * 0.5 - 2.8,
                    b as f64 * 0.3 - 1.4,
                    5.0 + c as f64 * 2.0 + 0.1 * b as f64,
                )
            })
            .collect();
        let mut obs = Vec::new();
        let mut ranges = Vec::new();
        for point in &points {
            let start = obs.len() as u32;
            for (f, pose) in poses.iter().enumerate() {
                for s in 0..3 {
                    let x_rig = pose.transform_point(point);
                    let x = if s == 0 {
                        x_rig
                    } else {
                        sensors[s - 1].transform_point(&x_rig)
                    };
                    if x.z <= 0.1 {
                        continue;
                    }
                    let xy = Point2::new(k.0 * x.x / x.z + k.2, k.1 * x.y / x.z + k.3);
                    obs.push(Obs::new(f as u32, xy, s as u32));
                }
            }
            ranges.push((start, obs.len() as u32));
        }
        let input = RigBaInput {
            frame_ids: (0..10).collect(),
            frame_fixed: (0..10).map(|i| i < 2).collect(),
            poses: poses
                .iter()
                .enumerate()
                .map(|(i, pose)| {
                    let d = Vector6::new(0.01, -0.01, 0.02, 0.002, -0.001, 0.0015);
                    if i < 2 {
                        pose.clone()
                    } else {
                        pose.compose(&SE3::exp(&(d * (i as f64 / 10.0))))
                    }
                })
                .collect(),
            point_ids: (0..points.len() as u64).collect(),
            point_fixed: vec![false; points.len()],
            points: points
                .iter()
                .map(|p| p + Vector3::new(0.02, -0.01, 0.03))
                .collect(),
        };
        let camera = ObsCamera::new(k, SE3::identity());
        (
            input,
            PointMajorObs {
                obs,
                point_obs_range: ranges,
                cameras: vec![
                    camera.clone(),
                    camera.clone().with_sensor(0),
                    camera.with_sensor(1),
                ],
            },
            sensors,
        )
    }

    fn perturbed(sensors: &[SE3]) -> ExtraBlocks {
        ExtraBlocks {
            camera: None,
            sensors: sensors
                .iter()
                .map(|s| {
                    s.compose(&SE3::exp(&Vector6::new(
                        0.03, -0.02, 0.01, 0.01, -0.02, 0.015,
                    )))
                })
                .collect(),
        }
    }

    fn assert_sensors_recovered(got: &[SE3], want: &[SE3], tol: f64) {
        for (got, want) in got.iter().zip(want) {
            let d = got.inverse().compose(want).log();
            assert!(d.norm() < tol, "sensor off by {d:?}");
        }
    }

    #[test]
    fn sensor_blocks_recover_the_rig_calibration() {
        let (mut input, point_major, truth) = rig_scene();
        let mut extras = perturbed(&truth);
        let result = optimize_point_major(
            &mut input,
            point_major,
            100,
            LossFunction::Trivial,
            None,
            &mut extras,
        )
        .expect("solve");
        assert_sensors_recovered(&extras.sensors, &truth, 1e-6);
        assert!(result.final_cost < 1e-9, "cost {}", result.final_cost);
    }

    #[test]
    fn sensor_blocks_recover_the_rig_calibration_on_the_iterative_path() {
        let (mut input, point_major, truth) = rig_scene();
        let mut problem =
            assemble_problem(&input, point_major, 0.0, perturbed(&truth)).expect("problem");
        problem.iterative_min_free_frames = 1;
        let result = optimize_problem(
            &mut input,
            &mut problem,
            Instant::now(),
            0,
            100,
            LossFunction::Trivial,
            None,
        )
        .expect("solve");
        assert_sensors_recovered(&problem.extras.sensors, &truth, 1e-4);
        assert!(result.final_cost < 1e-8, "cost {}", result.final_cost);
    }
}

#[cfg(test)]
mod iterative_bench {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector2, Vector3};

    /// A reduced camera system shaped like a city block revisited in several
    /// passes: each camera couples to its ±150 capture-order neighbours and
    /// to 100 cameras of the other passes at the same place. Diagonally
    /// dominant SPD by construction.
    fn dense_covisibility_system(n: usize) -> Vec<BTreeMap<usize, Matrix6<f64>>> {
        let mut columns: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n];
        let mut add = |a: usize, b: usize, k: f64| {
            let (lo, hi) = (a.min(b), a.max(b));
            if lo != hi {
                let blk =
                    Matrix6::from_fn(|r, c| ((r * 7 + c * 3 + lo + hi) % 11) as f64 * 1e-3 * k);
                columns[lo].insert(hi, blk);
            }
        };
        for i in 0..n {
            for d in 1..=150 {
                if i + d < n {
                    add(i, i + d, 1.0);
                }
            }
            let place = i % (n / 4);
            for k in 0..100 {
                add(
                    i,
                    ((i / (n / 4) + 1 + k % 3) % 4) * (n / 4) + (place + k / 3) % (n / 4),
                    0.5,
                );
            }
        }
        for (i, col) in columns.iter_mut().enumerate() {
            col.insert(
                i,
                Matrix6::identity() * 10.0
                    + Matrix6::from_fn(|r, c| if r == c { (i % 5) as f64 } else { 0.0 }),
            );
        }
        columns
    }

    /// The run-of-frames preconditioner converges to the same solution as a
    /// dense Cholesky solve, in fewer iterations than block-Jacobi.
    #[test]
    fn cluster_preconditioned_pcg_matches_cholesky() {
        let n = 120;
        let columns = dense_covisibility_system(n);
        let mut diag = Vec::with_capacity(n);
        let mut edges = Vec::new();
        let mut offdiag = Vec::new();
        let mut dense = DMatrix::<f64>::zeros(n * 6, n * 6);
        for (j, col) in columns.iter().enumerate() {
            for (&i, blk) in col {
                dense.view_mut((i * 6, j * 6), (6, 6)).copy_from(blk);
                if i == j {
                    diag.push(*blk);
                } else {
                    dense
                        .view_mut((j * 6, i * 6), (6, 6))
                        .copy_from(&blk.transpose());
                    // Edge-list convention: `(a, b)` holds the (row b, column a) block.
                    edges.push((j as u32, i as u32));
                    offdiag.push(*blk);
                }
            }
        }
        let rhs = DMatrix::<f64>::from_fn(n * 6, 1, |i, _| ((i * 13) % 17) as f64 - 8.0);
        let exact = dense.clone().cholesky().expect("SPD").solve(&rhs);
        let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
            let xs = DVector::<f64>::from_fn(n * 6, |k, _| x[k / 6][k % 6]);
            let y = &dense * xs;
            (0..n)
                .map(|i| Vector6::from_fn(|k, _| y[i * 6 + k]))
                .collect()
        };
        let (jacobi, it_jacobi) = pcg_core(&diag, matvec, &rhs, 1e-10, None, None).expect("pcg");
        let clusters = cluster_preconditioner(&diag, &edges, &offdiag, 16).expect("SPD runs");
        let (clustered, it_clustered) =
            pcg_core(&diag, matvec, &rhs, 1e-10, Some(&clusters), None).expect("pcg");
        let scale = exact.norm();
        assert!((&jacobi - &exact).norm() < 1e-6 * scale);
        assert!((&clustered - &exact).norm() < 1e-6 * scale);
        assert!(
            it_clustered < it_jacobi,
            "clustered {it_clustered} vs block-Jacobi {it_jacobi} iterations"
        );
    }

    /// Diagonal blocks, edges and their off-diagonal blocks.
    type EdgeSystem = (Vec<Matrix6<f64>>, Vec<(u32, u32)>, Vec<Matrix6<f64>>);

    /// A reduced camera system shaped like a long open capture: each frame
    /// couples to its 8 capture-order neighbours on each side, as a graph
    /// Laplacian plus a little diagonal damping. Its smooth modes (one end
    /// moving against the other) have tiny eigenvalues, the modes a drifting
    /// street capture leaves to the solver.
    fn chain_system(n: usize) -> EdgeSystem {
        let k = 8;
        let mut diag: Vec<Matrix6<f64>> = (0..n)
            .map(|i| {
                Matrix6::from_diagonal(&Vector6::from_fn(|r, _| 1e-4 * (1 + (i + r) % 3) as f64))
            })
            .collect();
        let mut edges = Vec::new();
        let mut offdiag = Vec::new();
        for a in 0..n {
            for b in a + 1..(a + k + 1).min(n) {
                let w = 1.0 + ((a * 7 + b * 3) % 5) as f64 * 0.1;
                let blk = Matrix6::identity() * -w;
                diag[a] += Matrix6::identity() * w;
                diag[b] += Matrix6::identity() * w;
                edges.push((a as u32, b as u32));
                offdiag.push(blk);
            }
        }
        (diag, edges, offdiag)
    }

    /// The coarse correction keeps the solution and takes the PCG through
    /// the chain's smooth modes in fewer iterations.
    #[test]
    fn coarse_corrected_pcg_matches_cholesky() {
        let n = 600;
        let (diag, edges, offdiag) = chain_system(n);
        let mut dense = DMatrix::<f64>::zeros(n * 6, n * 6);
        for (i, d) in diag.iter().enumerate() {
            dense.view_mut((i * 6, i * 6), (6, 6)).copy_from(d);
        }
        for (&(a, b), blk) in edges.iter().zip(&offdiag) {
            let (a, b) = (a as usize, b as usize);
            dense.view_mut((b * 6, a * 6), (6, 6)).copy_from(blk);
            dense
                .view_mut((a * 6, b * 6), (6, 6))
                .copy_from(&blk.transpose());
        }
        let rhs = DMatrix::<f64>::from_fn(n * 6, 1, |i, _| ((i * 13) % 17) as f64 - 8.0);
        let exact = dense.clone().cholesky().expect("SPD").solve(&rhs);
        let matvec = |x: &[Vector6<f64>]| -> Vec<Vector6<f64>> {
            let xs = DVector::<f64>::from_fn(n * 6, |k, _| x[k / 6][k % 6]);
            let y = &dense * xs;
            (0..n)
                .map(|i| Vector6::from_fn(|k, _| y[i * 6 + k]))
                .collect()
        };
        let clusters = cluster_preconditioner(&diag, &edges, &offdiag, 32).expect("SPD runs");
        let coarse = coarse_correction(&diag, &edges, &offdiag, n, 32).expect("SPD coarse");
        let (one, it_one) =
            pcg_core(&diag, matvec, &rhs, 1e-8, Some(&clusters), None).expect("pcg");
        let (two, it_two) =
            pcg_core(&diag, matvec, &rhs, 1e-8, Some(&clusters), Some(&coarse)).expect("pcg");
        let scale = exact.norm();
        assert!((&one - &exact).norm() < 1e-4 * scale);
        assert!((&two - &exact).norm() < 1e-4 * scale);
        // 33 against 58 when written.
        assert!(
            3 * it_two < 2 * it_one,
            "two-level {it_two} vs cluster-only {it_one} iterations"
        );
    }

    /// A global BA shaped like a long street capture: `n` frames along a
    /// line, points on a facade with heavy-tailed track lengths (most 3-15
    /// frames, one in twenty 40-120), noisy observations and a perturbed
    /// start. Prints the solver's `BA_PHASES` line.
    ///
    /// `VISLOC_BENCH_FRAMES=1200 [VISLOC_BENCH_PASSES=4] cargo test --release \
    ///     -p visloc-slam --lib large_global_ba -- --ignored --nocapture`
    /// The scene of [`large_global_ba`]: `n` frames, about 2,500
    /// observations each.
    fn street_scene(n: usize) -> (RigBaInput, PointMajorObs) {
        street_scene_passes(n, 1)
    }

    /// [`street_scene`] driven `passes` times down the same street (`n`
    /// frames in all, capture order pass by pass, each pass a little to the
    /// side), as SmallCity's four loops: a point seen in one pass is seen by
    /// the same stretch of each other pass with probability 0.7, so frames
    /// far apart in capture order share many points.
    fn street_scene_passes(n: usize, passes: usize) -> (RigBaInput, PointMajorObs) {
        let per_pass = n / passes;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut random = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let (f, c) = (600.0, (320.0, 240.0));
        let poses: Vec<SE3> = (0..n)
            .map(|i| {
                let (pass, s) = (i / per_pass, (i % per_pass) as f64);
                let yaw = 0.05 * (s * 0.01).sin() + 0.02 * pass as f64;
                let rotation = UnitQuaternion::from_euler_angles(0.0, yaw, 0.0);
                let centre =
                    Vector3::new(s * 0.1, 0.05 * (s * 0.03).cos() + 0.3 * pass as f64, 0.0);
                SE3::new(rotation, -(rotation * centre))
            })
            .collect();
        let mut points = Vec::new();
        let mut obs = Vec::new();
        let mut ranges = Vec::new();
        while obs.len() < n * 2500 {
            let track = if random() < 0.05 {
                40 + (random() * 80.0) as usize
            } else {
                3 + (random() * random() * 13.0) as usize
            };
            let first = (random() * per_pass.saturating_sub(track) as f64) as usize;
            let stretch = first..(first + track).min(per_pass);
            let mid = (stretch.start + stretch.end) as f64 * 0.05;
            let seen_pass = if passes > 1 {
                (random() * passes as f64) as usize
            } else {
                0
            };
            let frames: Vec<usize> = (0..passes)
                .filter(|&p| passes == 1 || p == seen_pass || random() < 0.7)
                .flat_map(|p| stretch.clone().map(move |s| p * per_pass + s))
                .collect();
            let point = Point3::new(
                mid + (random() - 0.5) * 2.0,
                (random() - 0.5) * 4.0,
                6.0 + random() * 4.0,
            );
            let start = obs.len() as u32;
            for frame in frames {
                let x = poses[frame].transform_point(&point);
                let noise = Vector2::new(random() - 0.5, random() - 0.5);
                let xy = Point2::new(f * x.x / x.z + c.0, f * x.y / x.z + c.1) + noise;
                obs.push(Obs::new(frame as u32, xy, 0));
            }
            ranges.push((start, obs.len() as u32));
            points.push(point);
        }
        let start_poses = poses
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                let d = Vector6::from_fn(|_, _| (random() - 0.5) * 2e-3);
                if i < 2 {
                    pose.clone()
                } else {
                    pose.compose(&SE3::exp(&d))
                }
            })
            .collect();
        let input = RigBaInput {
            frame_ids: (0..n as u64).collect(),
            frame_fixed: (0..n).map(|i| i < 2).collect(),
            poses: start_poses,
            point_ids: (0..points.len() as u64).collect(),
            point_fixed: vec![false; points.len()],
            points: points
                .iter()
                .map(|p| p + Vector3::from_fn(|_, _| (random() - 0.5) * 0.02))
                .collect(),
        };
        (
            input,
            PointMajorObs {
                obs,
                point_obs_range: ranges,
                cameras: vec![ObsCamera::new((f, f, c.0, c.1), SE3::identity())],
            },
        )
    }

    #[test]
    #[ignore]
    fn large_global_ba() {
        let n: usize = std::env::var("VISLOC_BENCH_FRAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1200);
        let passes: usize = std::env::var("VISLOC_BENCH_PASSES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        let (mut input, point_major) = street_scene_passes(n, passes);
        let (n_points, n_obs) = (input.points.len(), point_major.obs.len());
        let t = Instant::now();
        let result = optimize_point_major(
            &mut input,
            point_major,
            50,
            LossFunction::Trivial,
            None,
            &mut ExtraBlocks::default(),
        )
        .expect("solve");
        eprintln!(
            "{n} frames, {n_points} points, {n_obs} observations: {:?}, {} iterations, cost {:.1} -> {:.1}",
            t.elapsed(),
            result.iterations.len(),
            result.initial_cost,
            result.final_cost
        );
    }

    /// A linearization split into several point shards gives the same sums
    /// at any thread count.
    #[test]
    fn sharded_linearize_is_deterministic_across_thread_counts() {
        let (input, point_major) = street_scene(20);
        let problem =
            assemble_problem(&input, point_major, 0.0, ExtraBlocks::default()).expect("problem");
        assert!(
            problem.points.len() >= 2 * 2048,
            "{} points",
            problem.points.len()
        );
        let run = |threads: usize| {
            let (cost, _, diag, bc, _) = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(|| linearize(&problem, LossFunction::Trivial, Vec::new(), true));
            (cost, diag, bc)
        };
        assert_eq!(run(1), run(3));
    }

    /// Best-of-five times of one linearization and one elimination of the
    /// [`large_global_ba`] problem at its start.
    #[test]
    #[ignore]
    fn large_global_ba_phases() {
        let (input, point_major) = street_scene(1200);
        let problem =
            assemble_problem(&input, point_major, 0.0, ExtraBlocks::default()).expect("problem");
        let best = |mut run: Box<dyn FnMut()>| {
            (0..5)
                .map(|_| {
                    let t = Instant::now();
                    run();
                    t.elapsed()
                })
                .min()
                .expect("runs")
        };
        // Each linearization reuses the previous one's block array, as the
        // LM loop does.
        let mut reuse = Vec::new();
        let t_linearize = best(Box::new(|| {
            let (_, lin, _, _, _) = linearize(
                &problem,
                LossFunction::Trivial,
                std::mem::take(&mut reuse),
                true,
            );
            reuse = std::hint::black_box(lin).hcp;
        }));
        let (_, lin, _, _, _) = linearize(&problem, LossFunction::Trivial, Vec::new(), true);
        let t_eliminate = best(Box::new(|| {
            std::hint::black_box(eliminate_and_accumulate(
                &problem,
                &lin,
                problem.edges.len(),
                1e-4,
            ));
        }));
        eprintln!("linearize {t_linearize:?}, eliminate {t_eliminate:?}");
    }

    /// `cargo test --release -p visloc-slam --lib iterative_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn pcg_vs_block_cholesky_on_a_dense_covisibility_graph() {
        let n = 2500;
        let columns = dense_covisibility_system(n);
        let edges: usize = columns.iter().map(|c| c.len() - 1).sum();
        let rhs = DMatrix::<f64>::from_fn(n * 6, 1, |i, _| ((i * 13) % 17) as f64 - 8.0);

        let t = Instant::now();
        let x_pcg = pcg_blocks6(&columns, &rhs, 1e-3).expect("pcg");
        let t_pcg = t.elapsed();

        let t = Instant::now();
        let mut adjacency = vec![Vec::new(); n];
        for (j, col) in columns.iter().enumerate() {
            for &i in col.keys().filter(|&&i| i != j) {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }
        }
        for a in adjacency.iter_mut() {
            a.sort_unstable();
        }
        let elimination = crate::reordering::fast_fill_reducing_block_order(&adjacency);
        let mut pos = vec![0usize; n];
        for (k, &b) in elimination.iter().enumerate() {
            pos[b] = k;
        }
        let mut permuted: Vec<BTreeMap<usize, Matrix6<f64>>> = vec![BTreeMap::new(); n];
        for (j, col) in columns.iter().enumerate() {
            for (&i, blk) in col {
                let (pi, pj) = (pos[i], pos[j]);
                if pi >= pj {
                    permuted[pj].insert(pi, *blk);
                } else {
                    permuted[pi].insert(pj, blk.transpose());
                }
            }
        }
        let mut prhs = DMatrix::<f64>::zeros(n * 6, 1);
        for i in 0..n {
            for k in 0..6 {
                prhs[(pos[i] * 6 + k, 0)] = rhs[(i * 6 + k, 0)];
            }
        }
        let t_order = t.elapsed();
        let t = Instant::now();
        let mut cache = None;
        let x_chol = solve_spd_blocks6_cached(&mut cache, permuted, &prhs).expect("cholesky");
        let t_chol = t.elapsed();
        let mut max_diff = 0.0f64;
        for i in 0..n {
            for k in 0..6 {
                max_diff =
                    max_diff.max((x_chol[(pos[i] * 6 + k, 0)] - x_pcg[(i * 6 + k, 0)]).abs());
            }
        }
        eprintln!(
            "{n} cameras, {edges} camera pairs: pcg {t_pcg:?}; order {t_order:?} + block cholesky {t_chol:?}; max |dx| diff {max_diff:.2e}"
        );
        assert!(max_diff < 1e-3 * x_pcg.amax().max(1.0));
    }

    /// A smooth scale drift along a long street (the start scaled from 1
    /// at one end to 1 + 8% at the other, as an incremental reconstruction
    /// drifts) survives a COLMAP-style global BA, which stops on the
    /// relative gradient tolerance after one LM step: the step's damping
    /// (Ceres' initial trust radius) holds back that low-curvature mode.
    /// Run to convergence, the trust radius grows and the drift goes.
    /// About 25 s in release:
    ///
    /// `cargo test --release -p visloc-slam --lib global_ba_drift -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn global_ba_drift_needs_a_converged_solve() {
        let n = 1500;
        let solve = |gradient_tolerance_rel: Option<f64>| {
            let (mut input, point_major) = street_scene(n);
            let len = 0.1 * n as f64;
            let warp = |p: Vector3<f64>| {
                let k = 1.0 + 0.08 * p.x / len;
                Vector3::new(p.x + 0.04 * p.x * p.x / len, k * p.y, k * p.z)
            };
            for pose in input.poses.iter_mut().skip(2) {
                let centre = -(pose.rotation.inverse() * pose.translation);
                pose.translation = -(pose.rotation * warp(centre));
            }
            for p in &mut input.points {
                *p = Point3::from(warp(p.coords));
            }
            optimize_point_major(
                &mut input,
                point_major,
                50,
                LossFunction::Trivial,
                gradient_tolerance_rel,
                &mut ExtraBlocks::default(),
            )
            .expect("solve");
            // Length of the first and the last tenth of the path (the very
            // end, seen by few points, left out) over its true length.
            let centre = |i: usize| {
                let p = &input.poses[i];
                -(p.rotation.inverse() * p.translation)
            };
            let w = (n - 60) / 10;
            let scale = |a: usize| (centre(a + w) - centre(a)).norm() / (0.1 * w as f64);
            scale(9 * w) / scale(0) - 1.0
        };
        let one_step = solve(Some(1.0));
        let converged = solve(None);
        eprintln!("drift left: one step {one_step:.4}, converged {converged:.4}");
        assert!(one_step > 0.05, "one step {one_step}");
        assert!(converged.abs() < 0.005, "converged {converged}");
    }
}
