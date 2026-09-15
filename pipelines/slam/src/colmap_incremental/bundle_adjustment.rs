//! Faithful (C2-scoped) port of `estimators/bundle_adjustment.{h,cc}`'s
//! `BundleAdjustmentConfig` and the driver COLMAP wires around a Ceres
//! problem in `bundle_adjustment_ceres.cc`.
//!
//! Ported from (commit `64805cb870b574a569dccc34918d95a2db2b2fee`, pinned by
//! `docs/colmap_rig_mapper_port_plan.md` §0):
//! - `src/colmap/estimators/bundle_adjustment.h:77-150` — config surface
//!   ([`BundleAdjustmentConfig`]: `add_image`, `set_constant_rig_from_world_pose`,
//!   `add_variable_point`/`add_constant_point`, `fix_gauge`).
//! - `src/colmap/estimators/cost_functions/reprojection_error.h:389-417` —
//!   `RigReprojErrorConstantRigCostFunctor`, the control-configuration
//!   residual (fixed `sensor_from_rig` baked in as data, variable
//!   `rig_from_world` + `point3D`, fixed camera intrinsics). This is
//!   **reused, not re-derived**: `pipelines/slam/src/bundle.rs`'s
//!   `BaRigObservation`/`add_rig_observation` already implements exactly
//!   this formulation (verified against `bundle.rs`'s `rig_residual_jacobians`
//!   by a research pass during this port — `point_rig = pose.transform(X)`,
//!   `point_sensor = sensor_from_rig.transform(point_rig)`,
//!   `residual = project(point_sensor) - xy`, `sensor_from_rig`/intrinsics
//!   fixed, `pose`/`X` variable), per §3.3's explicit reuse instruction.
//! - `bundle_adjustment_ceres.cc:300-409` (`FixGaugeWithTwoCamsFromWorld`)
//!   and `:262-293` (`FixGaugeWithThreePoints`) — gauge fixing, mapped onto
//!   `bundle.rs`'s `fix_pose`/`fix_landmark` (whole-pose / whole-point
//!   fixes; see module doc below for why this port fixes *whole* poses/
//!   points rather than COLMAP's finer per-DoF gauge fix).
//!
//! ## Deviations from COLMAP, documented per the C2 task brief
//!
//! 1. **Per-image, not per-pose-DoF, config granularity.** COLMAP's
//!    `BundleAdjustmentConfig` is keyed by `image_t`/`rig_t`/`camera_t`
//!    independently. This port's callers (`mapper.rs`) always add or
//!    constant-fix a frame's images as a whole unit (never a lone camera of
//!    a 2-camera rig), so [`BundleAdjustmentConfig`] tracks `image_ids`
//!    (residual inclusion, exactly matching COLMAP) plus `constant_frame_ids`
//!    (pose gauge, one level coarser than COLMAP's per-sensor
//!    `SetConstantSensorFromRigPose` — irrelevant here since
//!    `ba_refine_sensor_from_rig=0` means `sensor_from_rig` is *never* a
//!    Ceres parameter block in this port at all, matching COLMAP's own
//!    control-configuration behavior exactly, see the reprojection_error.h
//!    citation above).
//! 2. **No intrinsics refinement code path at all** — the control config
//!    (`ba_refine_focal_length/principal_point/extra_params = 0`, plan
//!    §0.1/§1.2) never varies camera intrinsics, and
//!    `BaRigObservation`/`rig_residual_jacobians` in `bundle.rs` never
//!    exposes them as parameters, so there is nothing to wire up.
//! 3. **Point variable/constant policy now faithfully ports
//!    `ParameterizePoints` (`bundle_adjustment_ceres.cc:538-555`) plus
//!    `AddPointToProblem` (`bundle_adjustment_ceres.cc:819-879`).** COLMAP's
//!    rule: a point is held constant iff `track.Length() >
//!    num_observations_added` (i.e. **not every** observation of that point
//!    was added as a residual to *this* problem) or it is in
//!    `ConstantPoints()`; `VariablePoints()` (only — see below for why
//!    `ConstantPoints()` needs no pull-in) works by additionally pulling in
//!    the point's *remaining* observations from images outside
//!    `config.Images()` as extra residuals with that image's pose baked in
//!    as fixed data (`AddPointToProblem`), so `track.Length() ==
//!    num_observations` holds and it becomes free. This port implements
//!    that pull-in (see [`solve`]'s "Deviation 3" comment) by adding the
//!    outside image's *frame* to the `bundle.rs` problem via `add_pose` +
//!    `fix_pose` (a frame added and immediately fixed is bit-for-bit
//!    equivalent to COLMAP's `ReprojErrorConstantPoseCostFunctor`/
//!    `RigReprojErrorConstantRigCostFunctor` with a baked-constant pose: the
//!    residual formula and its point-Jacobian are identical whether the pose
//!    value is "a Ceres parameter block that happens to be constant" or "not
//!    a parameter block at all", and `bundle.rs`'s existing fixed-pose
//!    handling already excludes it from the reduced camera system) —
//!    relying on this port's calling convention (`mapper.rs` always adds or
//!    constant-fixes a frame's images as a whole unit, module doc point 1)
//!    to guarantee an outside track element's frame is never *also* a
//!    variable frame already in `config.Images()` under a different image id
//!    of the same frame (COLMAP itself does not enforce this and would, in
//!    that corner case, add a second residual that treats the same physical
//!    pose as a frozen snapshot disconnected from its live parameter block —
//!    not reachable by this port's callers).
//!    **`ConstantPoints()` does *not* need the same pull-in**: COLMAP does
//!    call `AddPointToProblem` for them too, but since this control never
//!    refines camera intrinsics (deviation 2) every pulled-in residual for a
//!    constant point would have *both* its point (`SetParameterBlockConstant`
//!    via `ParameterizePoints`) and its camera params
//!    (`SetParameterBlockConstant` via `ParameterizeCameras`,
//!    `constant_camera` is always true here) held constant — i.e. zero free
//!    parameters, hence zero contribution to the Jacobian/gradient/Hessian
//!    of the reduced system either way. Skipping it is therefore a provably
//!    numerically-inert simplification for this control, not a deviation in
//!    the solved system.
//! 4. **Gauge fixing is whole-pose / whole-point**, not COLMAP's per-DoF
//!    `SetParameterization` trick (fixing e.g. only the X-translation
//!    component of one frame while leaving its other 5 DoF free).
//!    `bundle.rs` has no partial-DoF pose fix (confirmed by a research pass
//!    during this port); `Gauge::TwoFramesFromWorld` fixes two whole frame
//!    poses (12 numbers, redundantly over-determining the 7-DoF similarity
//!    gauge — generically forces it to identity, same practical effect as
//!    COLMAP's `TWO_CAMS_FROM_WORLD`) and `Gauge::ThreePoints` fixes three
//!    whole 3D points (9 numbers, same over-determination argument as
//!    COLMAP's `THREE_POINTS`). Numerically these are *stricter* than
//!    COLMAP's fix (more parameters pinned to exact values instead of one
//!    scalar per anchor), which only affects convergence rate/conditioning,
//!    not the solution's metric correctness (§1.6 of the port plan: gauge
//!    fixing is orthogonal to the metric-scale guarantee, which comes from
//!    the fixed `sensor_from_rig` baked into every residual, unaffected by
//!    this port's choice of *which* extra DoF to pin numerically).
//! 5. **Convergence criteria differ from Ceres.** COLMAP's
//!    `BundleAdjustmentOptions` sets Ceres `function_tolerance=0`,
//!    `gradient_tolerance` (10.0 local / 1.0 global, i.e. effectively
//!    "run to `max_num_iterations`"), `parameter_tolerance=0`
//!    (`incremental_pipeline.cc:201-210,244-249`) — i.e. COLMAP's control
//!    essentially disables early Ceres convergence checks and always runs
//!    the full iteration budget (local 25, global 50, doubled + halved
//!    tolerances for the first <10 frames). `bundle.rs`'s LM loop instead
//!    uses `step_tolerance`/`cost_tolerance`/`relative_cost_tolerance`
//!    (different quantities: parameter-step norm and absolute/relative
//!    cost decrease, not a gradient-norm test) — this port sets
//!    `max_iterations` from COLMAP's local/global defaults and otherwise
//!    keeps `BaConfig::default()`'s already-tight `step_tolerance=1e-7`/
//!    `cost_tolerance=1e-9`, which in practice also drives most solves to
//!    (or near) the iteration cap on non-trivial problems; exact
//!    iteration-by-iteration parity with Ceres is not claimed (§6 risk in
//!    the port plan).

use std::collections::BTreeSet;

use nalgebra::Point3;

use crate::bundle::{BaConfig, BaRigObservation, BundleAdjustment};
use visloc_core::geometry::{Pose, SE3};

use super::reconstruction::Reconstruction;
use super::types::{FrameT, ImageT, Point3DT, SensorT};

/// Port of `BundleAdjustmentGauge` (`bundle_adjustment.h:47-48`). See module
/// doc deviation 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Gauge {
    #[default]
    Unspecified,
    TwoFramesFromWorld,
    ThreePoints,
}

/// Port of `BundleAdjustmentConfig` (`bundle_adjustment.h:77-150`). See
/// module doc deviation 1 for the per-image (not per-sensor) granularity.
#[derive(Debug, Clone, Default)]
pub struct BundleAdjustmentConfig {
    image_ids: BTreeSet<ImageT>,
    constant_frame_ids: BTreeSet<FrameT>,
    constant_point3d_ids: BTreeSet<Point3DT>,
    variable_point3d_ids: BTreeSet<Point3DT>,
    gauge: Gauge,
}

impl BundleAdjustmentConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_image(&mut self, image_id: ImageT) {
        self.image_ids.insert(image_id);
    }

    /// Convenience matching mapper.rs's usual call pattern: add every image
    /// of `frame_id`.
    pub fn add_frame(&mut self, recon: &Reconstruction, frame_id: FrameT) {
        for image_id in recon.frame(frame_id).image_ids() {
            self.add_image(image_id);
        }
    }

    pub fn images(&self) -> &BTreeSet<ImageT> {
        &self.image_ids
    }

    pub fn num_images(&self) -> usize {
        self.image_ids.len()
    }

    /// Port of `SetConstantRigFromWorldPose` (`bundle_adjustment.h`).
    pub fn set_constant_rig_from_world_pose(&mut self, frame_id: FrameT) {
        self.constant_frame_ids.insert(frame_id);
    }

    pub fn add_variable_point(&mut self, point3d_id: Point3DT) {
        self.variable_point3d_ids.insert(point3d_id);
    }

    pub fn add_constant_point(&mut self, point3d_id: Point3DT) {
        self.constant_point3d_ids.insert(point3d_id);
    }

    pub fn fix_gauge(&mut self, gauge: Gauge) {
        self.gauge = gauge;
    }
}

/// Port of `BundleAdjustmentOptions`'s control-relevant subset
/// (`bundle_adjustment_ceres.h`, `incremental_pipeline.cc:192-282`'s
/// `LocalBundleAdjustment`/`GlobalBundleAdjustment` factories). See module
/// doc deviation 5 for why only `max_iterations` is threaded through to
/// `bundle.rs`'s `BaConfig`.
/// C2.5: which BA solver backend `solve` dispatches to. `Native`
/// ([`super::rig_ba_solver`]) is a from-scratch, contiguous-`Vec`,
/// `rayon`-parallel, block-Cholesky-Schur Levenberg-Marquardt solver written
/// to replace `Legacy`'s measured ~340ms/iteration BTreeMap-indexed, dense-
/// Schur `bundle::BundleAdjustment::optimize` path (infeasible past ~10k
/// frames at global-BA scale). `Legacy` is kept selectable for
/// regression/parity checks (`rig_ba_solver`'s `native_vs_legacy_*` tests)
/// and as an escape hatch. See `super::rig_ba_solver`'s module doc for the
/// Native solver's full design (residual model, parameterization, trust
/// region, citations).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BaBackend {
    #[default]
    Native,
    Legacy,
}

/// A/B switch for the "Deviation 3" pull-in behavior documented above
/// (`AddPointToProblem`, `bundle_adjustment_ceres.cc:819-879`). Added after
/// a 5k-frame real-data A/B (lead review) showed the literal COLMAP pull-in
/// regressing ATE relative to the pre-pull-in snapshot at that scale (0.273m
/// -> 0.699m holding the DLT pose solver fixed) despite this port's pull-in
/// mechanics re-auditing clean against COLMAP line-for-line (see this
/// module's `solve()` doc comment on the re-audit) — kept as a live A/B
/// lever pending further investigation, not because a coding bug was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalBaPointPolicy {
    /// Faithful `AddPointToProblem` pull-in (this module's "Deviation 3"
    /// fix): a point explicitly requested variable
    /// (`BundleAdjustmentConfig::add_variable_point`) gets its *entire*
    /// track pulled in, including observations from frames outside the
    /// window (added with their pose held fixed).
    #[default]
    Colmap,
    /// A point is variable iff its *entire* track is already inside
    /// `config.image_ids` (`track_fully_in_window`) — an explicit
    /// `add_variable_point` request is honored only when the track is
    /// *already* fully in the window, otherwise the point falls back to
    /// constant. **Not** the pre-C2.6 rule (confirmed by a 2.5k real-data
    /// A/B: this policy underperforms the pre-C2.6 snapshot, 0.198m vs
    /// 0.105m ATE with the DLT pose solver held fixed) — see
    /// [`LocalBaPointPolicy::VariableWithoutPullIn`] for that. Kept as a
    /// distinct, already-tested A/B point rather than removed.
    WindowOnly,
    /// The actual pre-C2.6 rule, restored verbatim from
    /// `git show 053f6e4:.../bundle_adjustment.rs`: a point is variable iff
    /// `add_variable_point` was called for it *or* its track is fully in the
    /// window — same variable/constant *decision* as [`LocalBaPointPolicy::Colmap`]
    /// — but, unlike `Colmap`, an explicitly-variable point whose track
    /// leaves the window is **never pulled in**: it is optimized as a free
    /// point using only whichever of its observations happen to already be
    /// residuals in this problem (i.e. only its in-window observations —
    /// the rest are simply omitted, not added with a fixed pose). This is
    /// what `add_variable_point` alone did before this session's
    /// COLMAP-faithful `AddPointToProblem` pull-in fix.
    VariableWithoutPullIn,
}

/// Port of `CeresBundleAdjustmentOptions::LossFunctionType`
/// (`bundle_adjustment_ceres.h`) restricted to the three variants COLMAP's
/// control actually reaches (`TRIVIAL`, `SOFT_L1`, `CAUCHY` — `HUBER` is
/// never selected by any call site this port implements, see [`solve`]'s
/// doc for the full per-call-site audit, so it is not ported). The `f64`
/// payload on `SoftL1`/`Cauchy` is Ceres' loss-function `scale` parameter
/// `a` (`CeresBundleAdjustmentOptions::loss_function_scale`, or
/// `AbsolutePoseRefinementOptions::loss_function_scale` for the
/// GP3P-refinement call site — see [`super::rig_ba_solver::evaluate_loss`]
/// for where `a` enters the `rho(s)` formula). Evaluated/applied by
/// [`super::rig_ba_solver`]'s `Corrector` (a direct port of Ceres'
/// `internal/ceres/corrector.{h,cc}`); see that module's doc.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LossFunction {
    #[default]
    Trivial,
    SoftL1(f64),
    Cauchy(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BundleAdjustmentOptions {
    pub max_num_iterations: usize,
    pub backend: BaBackend,
    pub local_ba_point_policy: LocalBaPointPolicy,
    pub loss_function: LossFunction,
}

impl BundleAdjustmentOptions {
    /// `kDefaultCeresLocalMaxNumIterations` (`incremental_pipeline.cc:47`).
    /// `loss_function`: `LocalBundleAdjustment()` sets `SOFT_L1` with
    /// `loss_function_scale = 1.0` (`incremental_pipeline.cc:217-219`).
    /// Callers driving `IterativeLocalRefinement`'s multi-iteration loop
    /// (`mapper.rs::iterative_local_refinement`) must downgrade this to
    /// `Trivial` after the first iteration themselves — see that function's
    /// doc for the exact COLMAP citation
    /// (`sfm/incremental_mapper.cc:1277-1281`, "Only use robust cost
    /// function for first iteration"); `local()` alone always returns the
    /// first-iteration (robust) configuration.
    pub fn local() -> Self {
        Self {
            max_num_iterations: 25,
            backend: BaBackend::default(),
            local_ba_point_policy: LocalBaPointPolicy::default(),
            loss_function: LossFunction::SoftL1(1.0),
        }
    }
    /// `kDefaultCeresGlobalMaxNumIterations` (`incremental_pipeline.cc:48`).
    /// `local_ba_point_policy` is moot for global BA: `adjust_global_bundle`
    /// never calls `add_variable_point` (every point's track is trivially
    /// "fully in window" once every registered frame is in the config), so
    /// the pull-in branch never fires there either way — kept in sync with
    /// `local()` purely so a single CLI flag can set both without the field
    /// being silently ignored for one of the two call sites.
    /// `loss_function`: `GlobalBundleAdjustment()` sets `TRIVIAL`
    /// (`incremental_pipeline.cc:267-268`) — every `IterativeGlobalRefinement`
    /// iteration uses the *same* `ba_options` unchanged
    /// (`sfm/incremental_mapper.cc:1286-1317`; unlike local refinement there
    /// is no per-iteration loss switch for global BA).
    pub fn global() -> Self {
        Self {
            max_num_iterations: 50,
            backend: BaBackend::default(),
            local_ba_point_policy: LocalBaPointPolicy::default(),
            loss_function: LossFunction::Trivial,
        }
    }
}

fn sensor_from_rig_for_image(recon: &Reconstruction, image_id: ImageT) -> SE3 {
    let image = recon.image(image_id);
    let frame = recon.frame(image.frame_id);
    let rig = recon.rig(frame.rig_id());
    let sensor_id = SensorT::camera(image.camera_id);
    if rig.is_ref_sensor(sensor_id) {
        SE3::identity()
    } else {
        rig.sensor_from_rig(sensor_id)
    }
}

/// Port of `CreateDefaultBundleAdjuster(...)->Solve()` for this control's
/// exact case (fixed `sensor_from_rig`, fixed intrinsics, trivial loss —
/// see module doc). Builds one `bundle.rs::BundleAdjustment` problem from
/// every image in `config`, solves it, and writes optimized frame poses /
/// point positions back into `recon`. Returns `true` iff the solve
/// succeeded and produced a usable solution (mirrors
/// `summary->IsSolutionUsable()`).
///
/// ## Pull-in re-audit (5k real-data regression, lead review)
/// A 5k-frame real-data A/B showed `Colmap`'s literal pull-in regressing ATE
/// relative to the pre-pull-in snapshot (holding the DLT pose solver fixed:
/// 0.273m -> 0.699m; COLMAP itself: 0.123m at this scale). Re-audited every
/// mechanical detail against COLMAP with fresh eyes; found no coding bug:
/// - **(a) Which points are pulled in?** Only `config.VariablePoints()`
///   (`bundle_adjustment_ceres.h:616-618`'s constructor loop calling
///   `AddPointToProblem` for `config_.VariablePoints()`), i.e. only the
///   caller-chosen set — `AdjustLocalBundle` populates it from its
///   `point3D_ids` argument (the triangulator's `ModifiedPoints3D()`, *not*
///   "every point observed by a window image"), filtered to
///   `!point3D.HasError() || track.Length() <= 15`
///   (`incremental_mapper.cc:1067-1080`, "make sure we refine all new and
///   short-track 3D points, no matter if they are fully contained in the
///   local image set or not" — pull-in for exactly these points is the
///   documented *intent*, not a side effect). `mapper.rs::adjust_local_bundle`
///   already reproduces this filter faithfully (its own `MAX_TRACK_LENGTH`
///   constant) — confirmed unchanged by this session's work.
/// - **(b) Must pulled-in images be registered?** `AddPointToProblem` never
///   checks `HasPose()`/`IsRegistered()` explicitly — it relies on the
///   invariant that a `Point3D`'s track only ever contains observations from
///   currently-registered images (deregistration removes the observation,
///   `DeRegisterFrameEvent`/`ObservationManager`). This port maintains the
///   same invariant (`observation_manager.rs`'s deregister path removes
///   track elements); a pulled-in image's frame is therefore always
///   currently registered, but its **pose value may be stale** relative to
///   the *last* global BA — this is true of COLMAP's own pull-in too (it
///   reads `image.FramePtr()->RigFromWorld()`, whatever is currently
///   stored, `bundle_adjustment_ceres.cc:852,863`), so staleness itself is
///   not a deviation from COLMAP, but see the hypothesis below.
/// - **(c) Do pulled-in images contribute their *other* residuals?** No —
///   `AddPointToProblem` adds exactly one residual per track element (the
///   *specific* `(point3D_id, track_el.image_id)` observation), never the
///   pulled-in image's other 2D points against other 3D points
///   (`bundle_adjustment_ceres.cc:839-878`). This port's pull-in loop adds
///   exactly one `BaRigObservation` per track element for the same reason —
///   confirmed matching.
/// - **(d) How is the pulled-in image's pose held constant?** Not via
///   `SetParameterBlockConstant` on an existing block — the pose is never a
///   Ceres parameter for that residual at all; `ReprojErrorConstantPoseCostFunctor`
///   bakes `cam_from_world` as plain data (`bundle_adjustment_ceres.cc:852-859`,
///   `:865-870`). This port's `ba.add_pose(..); ba.fix_pose(..)` is the
///   bit-for-bit equivalent given `bundle.rs`'s existing (pre-dating this
///   port) fixed-pose handling: a fixed frame is excluded from
///   `free_frame_slot` entirely (`rig_ba_solver.rs`'s `Problem::free_frame_slot`),
///   so it never gets a column/row in the reduced camera system — confirmed
///   by `native_constant_frames_and_points_unchanged` and this session's new
///   `local_ba_pulls_in_out_of_window_observation_for_variable_point` test
///   (outside pose asserted bit-identical after solve).
/// - **(e) Schur elimination when one fixed frame is shared by many pulled-in
///   points?** Re-checked `rig_ba_solver.rs::linearize_point`: a fixed
///   frame's observations still fold into `hpp`/`bp` (point block,
///   unconditional) but never receive a `bc`/`diag`/`hcp` slot (guarded by
///   `target_frames.binary_search`, and `target_frames` is built from
///   *free* frames only — `point_free_frames`, `build_reduced_system_pattern`).
///   Multiple points sharing the same fixed frame each independently fold
///   their own contribution into their own `hpp`/`bp`; there is no shared,
///   frame-indexed accumulator a fixed frame could corrupt. No indexing bug
///   found.
/// - **Bug hunt (task 3):** re-checked for (i) double-adding a window-image
///   observation via pull-in — guarded by `config.image_ids.contains(...)`
///   `continue`, confirmed exercised by the existing test; (ii) wrong
///   `sensor_from_rig` for the pulled-in image — uses the same
///   `sensor_from_rig_for_image` helper as every other call site; (iii) a
///   pulled-in frame being eligible for `Gauge::TwoFramesFromWorld` — that
///   gauge is only used by `adjust_global_bundle`, which never populates
///   `variable_point3d_ids` at all (every point's track is trivially "fully
///   in window" there), so pull-in structurally cannot interact with it;
///   `Gauge::ThreePoints` (used by local BA) selects from `added_points`
///   filtered to not-yet-fixed, matching COLMAP's own "any unfixed point
///   qualifies" `FixGaugeWithThreePoints` rule exactly. **No coding bug
///   found in (a)-(e) or the bug-hunt list.**
/// - **Leading hypothesis (untested against real data, per instructions):**
///   the regression is a genuine large-scale interaction between two
///   *independently* faithful, already-documented pieces: pulling in a
///   revisited old point's full (possibly wide-track, temporally-distant)
///   track as hard constraints compounds when combined with this module's
///   own **deviation 4** (whole-pose/whole-point `Gauge::ThreePoints`
///   gauge-fixing is strictly more rigid than COLMAP's per-DoF
///   `SetParameterization`) and **deviation 5**/6 (this port's LM
///   convergence criteria differ from Ceres', plausibly letting a
///   pulled-in-biased local window run further from its window-only optimum
///   in the allotted iterations than COLMAP's `gradient_tolerance=1e-4`
///   Ceres solve would). Both are pre-existing, independently-documented
///   deviations, not introduced by the pull-in fix itself — flagged for
///   follow-up, not fixed here. [`LocalBaPointPolicy`] is the requested A/B
///   lever pending the lead's own real-data A/B.
pub fn solve(
    options: &BundleAdjustmentOptions,
    config: &BundleAdjustmentConfig,
    recon: &mut Reconstruction,
) -> bool {
    if config.image_ids.is_empty() {
        return false;
    }

    let default_camera = recon
        .camera(
            recon
                .image(*config.image_ids.iter().next().unwrap())
                .camera_id,
        )
        .clone();
    let mut ba = BundleAdjustment::new(default_camera);

    let mut frame_ids: BTreeSet<FrameT> = BTreeSet::new();
    for &image_id in &config.image_ids {
        frame_ids.insert(recon.image(image_id).frame_id);
    }
    for &frame_id in &frame_ids {
        let rig_from_world = recon.frame(frame_id).rig_from_world().clone();
        ba.add_pose(
            frame_id,
            Pose {
                world_to_camera: rig_from_world,
            },
        );
        if config.constant_frame_ids.contains(&frame_id) {
            ba.fix_pose(frame_id);
        }
    }

    let mut added_points: BTreeSet<Point3DT> = BTreeSet::new();
    for &image_id in &config.image_ids {
        let image = recon.image(image_id);
        let frame_id = image.frame_id;
        let camera = recon.camera(image.camera_id).clone();
        let sensor_from_rig = sensor_from_rig_for_image(recon, image_id);
        for point2d in &image.points2d {
            let Some(point3d_id) = point2d.point3d_id else {
                continue;
            };
            added_points.insert(point3d_id);
            ba.add_rig_observation(BaRigObservation {
                keyframe_id: frame_id,
                landmark_id: point3d_id,
                xy: point2d.xy,
                camera: camera.clone(),
                sensor_from_rig: sensor_from_rig.clone(),
            });
        }
    }
    // Port of `DefaultBundleAdjuster`'s constructor calling `AddPointToProblem`
    // for every point in `config.VariablePoints()`/`config.ConstantPoints()`
    // (`bundle_adjustment_ceres.cc:616-621`), independent of whether that
    // point had any observation among `config.Images()` at all.
    for &pid in &config.variable_point3d_ids {
        added_points.insert(pid);
    }
    for &pid in &config.constant_point3d_ids {
        added_points.insert(pid);
    }

    // Frames pulled in purely to hold a fixed, baked-constant pose for a
    // variable point's out-of-window observations (see module doc deviation
    // 3 / `AddPointToProblem`). Never written back to `recon` below (only
    // `frame_ids`, computed above from `config.image_ids` alone, is).
    let mut pulled_frame_ids: BTreeSet<FrameT> = BTreeSet::new();

    for &point3d_id in &added_points {
        let xyz = recon.point3d(point3d_id).xyz;
        ba.add_landmark(point3d_id, xyz);
        // Deviation 3: see module doc for the exact `ParameterizePoints`
        // policy this reproduces.
        let track_fully_in_window = recon
            .point3d(point3d_id)
            .track
            .iter()
            .all(|el| config.image_ids.contains(&el.image_id));
        // Three-way `LocalBaPointPolicy` branch (see each variant's doc):
        // `Colmap` and `VariableWithoutPullIn` both honor an explicit
        // `add_variable_point` request unconditionally (the *variable vs.
        // constant decision* is identical); `WindowOnly` only honors it when
        // the track is already fully in the window. Only `Colmap` then
        // pulls in the rest of the track as extra fixed-pose residuals;
        // `VariableWithoutPullIn` (the actual pre-C2.6 rule, restored from
        // `git show 053f6e4`) leaves the point free but optimizes it from
        // only whichever observations already happen to be in this problem
        // (its in-window observations) — the outside ones are simply never
        // added, not pulled in with a fixed pose.
        let explicit_variable = config.variable_point3d_ids.contains(&point3d_id);
        let honor_explicit_variable = matches!(
            options.local_ba_point_policy,
            LocalBaPointPolicy::Colmap | LocalBaPointPolicy::VariableWithoutPullIn
        );
        let variable = !config.constant_point3d_ids.contains(&point3d_id)
            && (track_fully_in_window || (honor_explicit_variable && explicit_variable));
        if !variable {
            ba.fix_landmark(point3d_id);
            continue;
        }
        if options.local_ba_point_policy == LocalBaPointPolicy::Colmap
            && explicit_variable
            && !track_fully_in_window
        {
            // `AddPointToProblem` (`bundle_adjustment_ceres.cc:819-879`):
            // pull in every remaining track observation from images outside
            // `config.Images()`, with that image's pose baked in as fixed
            // data, so the point sees its *entire* track (matching
            // `ParameterizePoints`'s `track.Length() == num_observations`
            // free condition exactly instead of only seeing the in-window
            // subset).
            let track = recon.point3d(point3d_id).track.clone();
            for el in &track {
                if config.image_ids.contains(&el.image_id) {
                    continue; // already added above (`AddImageToProblem`).
                }
                let image = recon.image(el.image_id);
                let frame_id = image.frame_id;
                if !frame_ids.contains(&frame_id) && pulled_frame_ids.insert(frame_id) {
                    let rig_from_world = recon.frame(frame_id).rig_from_world().clone();
                    ba.add_pose(
                        frame_id,
                        Pose {
                            world_to_camera: rig_from_world,
                        },
                    );
                    ba.fix_pose(frame_id);
                }
                let camera = recon.camera(image.camera_id).clone();
                let sensor_from_rig = sensor_from_rig_for_image(recon, el.image_id);
                let xy = image.points2d[el.point2d_idx].xy;
                ba.add_rig_observation(BaRigObservation {
                    keyframe_id: frame_id,
                    landmark_id: point3d_id,
                    xy,
                    camera,
                    sensor_from_rig,
                });
            }
        }
    }

    if ba.rig_observations.is_empty() || ba.landmarks.is_empty() {
        return false;
    }

    match config.gauge {
        Gauge::Unspecified => {}
        Gauge::TwoFramesFromWorld => {
            let candidates: Vec<FrameT> = frame_ids
                .iter()
                .copied()
                .filter(|id| !ba.fixed_poses.contains(id))
                .take(2)
                .collect();
            for id in candidates {
                ba.fix_pose(id);
            }
        }
        Gauge::ThreePoints => {
            let candidates: Vec<Point3DT> = added_points
                .iter()
                .copied()
                .filter(|id| !ba.fixed_landmarks.contains(id))
                .take(3)
                .collect();
            for id in candidates {
                ba.fix_landmark(id);
            }
        }
    }

    let ba_config = BaConfig {
        max_iterations: options.max_num_iterations,
        // Deviation 6: COLMAP stops Ceres on gradient_tolerance=1e-4
        // (function_tolerance=0). bundle.rs has no gradient criterion, so a
        // relative cost tolerance stands in for it; without it every solve
        // runs to max_iterations while the cost changes by <1e-6.
        relative_cost_tolerance: Some(1.0e-6),
        ..BaConfig::default()
    };

    let started = std::time::Instant::now();
    let n_obs = ba.rig_observations.len();
    let n_lm = ba.landmarks.len();
    let n_fixed_lm = ba.fixed_landmarks.len();
    let solved = match options.backend {
        BaBackend::Legacy => {
            // `Legacy` (`bundle::BundleAdjustment::optimize`) predates
            // `LossFunction` and does not implement Ceres' `Corrector`
            // (`super::rig_ba_solver`'s port) — it always solves the
            // trivial-loss problem regardless of `options.loss_function`.
            // Not wired up: `Legacy` is a regression/parity escape hatch,
            // never this port's default backend (`BaBackend::default() ==
            // Native`), and every test/call site that needs robust loss
            // uses `Native`.
            if options.loss_function != LossFunction::Trivial {
                eprintln!(
                    "BA_SOLVE WARNING backend=Legacy loss_function={:?} is ignored \
                     (Legacy is trivial-loss only); use backend=Native for robust loss",
                    options.loss_function
                );
            }
            ba.optimize(&ba_config)
        }
        BaBackend::Native => super::rig_ba_solver::optimize(
            &mut ba,
            options.max_num_iterations,
            options.loss_function,
        ),
    };
    let Ok(result) = solved else {
        return false;
    };
    eprintln!(
        "BA_SOLVE backend={:?} frames={} obs={n_obs} landmarks={n_lm} fixed_landmarks={n_fixed_lm} iterations={} elapsed_ms={}",
        options.backend,
        frame_ids.len(),
        result.iterations.len(),
        started.elapsed().as_millis()
    );

    for &frame_id in &frame_ids {
        let pose = &ba.poses[&frame_id];
        recon
            .frame_mut(frame_id)
            .set_rig_from_world(pose.world_to_camera.clone());
    }
    for &point3d_id in &added_points {
        let xyz: Point3<f64> = ba.landmarks[&point3d_id];
        recon.point3d_mut(point3d_id).xyz = xyz;
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    use crate::colmap_incremental::pipeline::reconstruction_from_cache;
    use crate::colmap_incremental::reconstruction::TrackElement;
    use crate::colmap_incremental::test_support::build_synthetic_rig_scene;

    /// C2 task item 7: "BA on a synthetic rig scene converges to ground
    /// truth". Seeds exact ground-truth poses/points, perturbs every frame
    /// except two anchors and every point, then checks `solve` converges
    /// back within tight tolerances — including translation (hence scale),
    /// which the fixed `sensor_from_rig` baseline should pin exactly per
    /// `docs/colmap_rig_mapper_port_plan.md` §1.6.
    #[test]
    fn solve_converges_synthetic_rig_scene() {
        let scene = build_synthetic_rig_scene(4, 3);
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

        let mut options = BundleAdjustmentOptions::global();
        options.max_num_iterations = 50;
        assert!(solve(&options, &config, &mut recon), "BA solve failed");

        for (j, gt_xyz) in scene.ground_truth_points.iter().enumerate() {
            let err = (recon.point3d(point3d_ids[j]).xyz - gt_xyz).norm();
            assert!(err < 0.01, "point {j} error {err} too large after BA");
        }
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            let got = recon.frame(frame_id).rig_from_world();
            let dt = (got.translation - gt.translation).norm();
            assert!(
                dt < 0.01,
                "frame {frame_id} translation error {dt} too large after BA"
            );
        }
    }

    /// Shared setup for the Native/Legacy-parity and constant-frame/point
    /// tests below: same perturbed synthetic rig scene as
    /// `solve_converges_synthetic_rig_scene`, factored out so both backends
    /// run from an identical starting point.
    fn build_scene(
        num_frames: usize,
    ) -> (
        crate::colmap_incremental::reconstruction::Reconstruction,
        BundleAdjustmentConfig,
        Vec<u64>,
        [u64; 2],
    ) {
        let scene = build_synthetic_rig_scene(num_frames, 3);
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
        (recon, config, point3d_ids, anchors)
    }

    /// C2.5 task item 6(c): `Native` and `Legacy` solve the *same* problem
    /// (identical starting poses/points, gauge, iteration budget) to the
    /// same final cost and the same poses/points, within the task's 1e-6
    /// relative tolerance.
    #[test]
    fn native_vs_legacy_same_problem_same_result() {
        let (mut recon_native, config_native, point_ids, _anchors) = build_scene(5);
        let (mut recon_legacy, config_legacy, _point_ids2, _anchors2) = build_scene(5);

        let mut native_options = BundleAdjustmentOptions::global();
        native_options.max_num_iterations = 30;
        native_options.backend = BaBackend::Native;
        let mut legacy_options = native_options;
        legacy_options.backend = BaBackend::Legacy;

        assert!(solve(&native_options, &config_native, &mut recon_native));
        assert!(solve(&legacy_options, &config_legacy, &mut recon_legacy));

        for &pid in &point_ids {
            let a = recon_native.point3d(pid).xyz;
            let b = recon_legacy.point3d(pid).xyz;
            let diff = (a - b).norm();
            let scale = b.coords.norm().max(1.0);
            assert!(
                diff / scale < 1e-6,
                "point {pid} native {a:?} vs legacy {b:?} (relative diff {})",
                diff / scale
            );
        }
        for &frame_id in recon_native.reg_frame_ids() {
            let a = recon_native.frame(frame_id).rig_from_world();
            let b = recon_legacy.frame(frame_id).rig_from_world();
            let dt = (a.translation - b.translation).norm();
            assert!(
                dt < 1e-6 * b.translation.norm().max(1.0),
                "frame {frame_id} native translation {a:?} vs legacy {b:?}"
            );
        }
    }

    /// C3 task item A: a point explicitly requested variable
    /// (`config.add_variable_point`, mirroring `AdjustLocalBundle`'s
    /// "recently modified" point set,
    /// `incremental_mapper.cc:1072-1080`) but whose track has an
    /// observation from a frame *outside* the local window must still be
    /// pulled in fully (`AddPointToProblem`,
    /// `bundle_adjustment_ceres.cc:819-879`) and move under optimization —
    /// contrasted with a second point with the same out-of-window shape
    /// that is *not* explicitly requested variable, which must stay exactly
    /// constant (`ParameterizePoints`, `bundle_adjustment_ceres.cc:538-555`:
    /// `track.Length() > num_observations` for the implicit/default case).
    /// The outside frame pulled in purely to supply the fixed pose for the
    /// extra residual must itself stay bit-identical (never a free
    /// parameter — `AddPointToProblem` bakes it as constant data).
    #[test]
    fn local_ba_pulls_in_out_of_window_observation_for_variable_point() {
        let scene = build_synthetic_rig_scene(6, 5);
        let mut recon = reconstruction_from_cache(&scene.db);
        for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
            recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
            recon.register_frame(frame_id);
        }

        // Two points, each observed by every frame 0..=5 (full track); each
        // uses its own `point2d_idx` (`0`/`1`, matching its position in
        // `ground_truth_points`, which `build_synthetic_rig_scene` uses as
        // every image's `points2d` order).
        let gt_a = scene.ground_truth_points[0];
        let gt_b = scene.ground_truth_points[1];
        let track_for = |idx: usize| {
            let mut track = Vec::new();
            for &(i1, i2) in &scene.images_per_frame {
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: idx,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: idx,
                });
            }
            track
        };
        let p_a = recon.add_point3d(gt_a, track_for(0));
        let p_b = recon.add_point3d(gt_b, track_for(1));

        let noise = Vector3::new(0.05, -0.03, 0.02);
        let perturbed_a = gt_a + noise;
        let perturbed_b = gt_b + noise;
        recon.point3d_mut(p_a).xyz = perturbed_a;
        recon.point3d_mut(p_b).xyz = perturbed_b;

        // Window = frames 0..=3; frames 4,5 are outside the window and
        // never added to `config.Images()`. Both points' tracks include
        // observations from the outside frames (every frame, by
        // construction above).
        let window_frames: [u64; 4] = [0, 1, 2, 3];
        let outside_frame = 5u64;
        let outside_pose_before = recon.frame(outside_frame).rig_from_world().clone();

        let mut config = BundleAdjustmentConfig::new();
        for &frame_id in &window_frames {
            let (i1, i2) = scene.images_per_frame[frame_id as usize];
            config.add_image(i1);
            config.add_image(i2);
        }
        config.set_constant_rig_from_world_pose(window_frames[0]);
        config.set_constant_rig_from_world_pose(*window_frames.last().unwrap());
        // Only p_a is explicitly requested variable; p_b is left to the
        // default policy (track not fully in window -> stays constant).
        config.add_variable_point(p_a);

        let options = BundleAdjustmentOptions::global();
        assert!(solve(&options, &config, &mut recon), "BA solve failed");

        let moved = (recon.point3d(p_a).xyz - perturbed_a).norm();
        assert!(
            moved > 1e-4,
            "explicitly-variable out-of-window point p_a did not move ({moved})"
        );
        let err_a = (recon.point3d(p_a).xyz - gt_a).norm();
        assert!(
            err_a < 0.02,
            "p_a did not converge using its full (pulled-in) track: error {err_a}"
        );

        assert_eq!(
            recon.point3d(p_b).xyz,
            perturbed_b,
            "p_b (not explicitly variable, track not fully in window) must stay exactly constant"
        );

        assert_eq!(
            recon.frame(outside_frame).rig_from_world(),
            &outside_pose_before,
            "outside frame pulled in only to supply a fixed pose must stay bit-identical"
        );
    }

    /// C2.5 task item 6(d): frames in `constant_frame_ids` and points in
    /// `constant_point3d_ids` are left exactly unchanged by `Native`.
    #[test]
    fn native_constant_frames_and_points_unchanged() {
        let (mut recon, mut config, point_ids, anchors) = build_scene(5);
        // Anchors are already constant (gauge); additionally fix one more
        // frame and one more (otherwise-variable) point explicitly.
        let extra_constant_frame = 2u64;
        config.set_constant_rig_from_world_pose(extra_constant_frame);
        let extra_constant_point = point_ids[0];
        config.add_constant_point(extra_constant_point);

        let before_frame = recon.frame(extra_constant_frame).rig_from_world().clone();
        let before_point = recon.point3d(extra_constant_point).xyz;
        let before_anchor0 = recon.frame(anchors[0]).rig_from_world().clone();

        let options = BundleAdjustmentOptions::global();
        assert!(solve(&options, &config, &mut recon), "BA solve failed");

        assert_eq!(
            recon.frame(extra_constant_frame).rig_from_world(),
            &before_frame,
            "explicitly-constant frame moved"
        );
        assert_eq!(
            recon.point3d(extra_constant_point).xyz,
            before_point,
            "explicitly-constant point moved"
        );
        assert_eq!(
            recon.frame(anchors[0]).rig_from_world(),
            &before_anchor0,
            "gauge-anchor frame moved"
        );
    }

    /// Total squared pixel reprojection error over exactly `window_images`'
    /// own observations (never a pulled-in residual), evaluated at `recon`'s
    /// *current* state — used by
    /// `local_ba_point_policy_matches_colmap_local_bundle_scenario` to
    /// compare `Colmap` vs `WindowOnly` on a common, policy-independent cost.
    fn window_reprojection_cost(recon: &Reconstruction, window_images: &[ImageT]) -> f64 {
        let mut cost = 0.0;
        for &iid in window_images {
            let image = recon.image(iid);
            let rig_from_world = recon.frame(image.frame_id).rig_from_world().clone();
            let sensor_from_rig = sensor_from_rig_for_image(recon, iid);
            let camera = recon.camera(image.camera_id);
            for point2d in &image.points2d {
                let Some(pid) = point2d.point3d_id else {
                    continue;
                };
                let xyz = recon.point3d(pid).xyz;
                let p_rig = rig_from_world.transform_point(&xyz);
                let p_sensor = sensor_from_rig.transform_point(&p_rig);
                cost += match camera.project(&p_sensor) {
                    Some(proj) => (proj - point2d.xy).norm_squared(),
                    None => 1.0e6,
                };
            }
        }
        cost
    }

    /// C2.6 task (lead review after the 5k and 2.5k real-data A/Bs): a small
    /// `AdjustLocalBundle`-shaped scenario — 12 frames, a 6-frame local
    /// window, points seen both inside and outside the window — exercising
    /// all three [`LocalBaPointPolicy`] variants side by side from an
    /// identical starting state. Checks: (1) frames outside the window stay
    /// bit-identical under *every* policy (never a free parameter either way
    /// — `Colmap` adds them fixed for pull-in, `WindowOnly`/
    /// `VariableWithoutPullIn` never touch them at all); (2) a point
    /// explicitly requested variable with a track leaving the window
    /// converges close to ground truth under `Colmap` (pull-in sees its
    /// whole track) and under `VariableWithoutPullIn` (free, but optimized
    /// from only its in-window observations — sufficient here since the
    /// window alone already over-determines the point), but stays *exactly*
    /// at its perturbed value under `WindowOnly` (falls back to constant);
    /// (3) a point whose track is already fully inside the window converges
    /// under all three; (4) on this noiseless, fully-consistent synthetic
    /// scene (a single shared ground truth, so every policy's problem shares
    /// the same zero-cost global optimum for in-window residuals), `Colmap`'s
    /// final window-only reprojection cost is not worse than `WindowOnly`'s.
    #[test]
    fn local_ba_point_policy_matches_colmap_local_bundle_scenario() {
        let scene = build_synthetic_rig_scene(12, 11);
        let window_frames: [u64; 6] = [3, 4, 5, 6, 7, 8];
        let outside_frames: [u64; 6] = [0, 1, 2, 9, 10, 11];

        let build_recon = || {
            let mut recon = reconstruction_from_cache(&scene.db);
            for (&frame_id, gt) in &scene.ground_truth_rig_from_world {
                recon.frame_mut(frame_id).set_rig_from_world(gt.clone());
                recon.register_frame(frame_id);
            }
            recon
        };
        let track_for = |frames: &[u64], idx: usize| -> Vec<TrackElement> {
            let mut track = Vec::new();
            for &f in frames {
                let (i1, i2) = scene.images_per_frame[f as usize];
                track.push(TrackElement {
                    image_id: i1,
                    point2d_idx: idx,
                });
                track.push(TrackElement {
                    image_id: i2,
                    point2d_idx: idx,
                });
            }
            track
        };
        let all_frames: Vec<u64> = (0u64..12).collect();

        // 3 "gauge fodder" points: track confined to 2 window frames, added
        // *before* the real test points so `Gauge::ThreePoints`'s smallest-id
        // scan (`added_points` is a `BTreeSet`, ascending numeric order,
        // matching COLMAP's own "any unfixed point qualifies"
        // `FixGaugeWithThreePoints`) consumes exactly these 3, leaving every
        // other variable point below genuinely free to assert on.
        let noise = Vector3::new(0.02, -0.015, 0.01);
        let mut recon = build_recon();
        let mut gauge_fodder = Vec::new();
        for j in 0..3 {
            let gt = scene.ground_truth_points[j];
            let id = recon.add_point3d(gt, track_for(&[4, 5], j));
            gauge_fodder.push(id);
        }
        // 4 points with a *full* 12-frame track, explicitly requested
        // variable (mirrors a revisited old landmark newly re-observed by
        // the just-registered frame) — the pull-in-vs-constant contrast.
        let mut p_pulled = Vec::new();
        let mut p_pulled_gt = Vec::new();
        for j in 3..7 {
            let gt = scene.ground_truth_points[j];
            let id = recon.add_point3d(gt + noise, track_for(&all_frames, j));
            p_pulled.push(id);
            p_pulled_gt.push(gt);
        }
        // 2 points whose track never leaves the window (freshly triangulated
        // by the local window itself) — variable under both policies.
        let mut p_window_only = Vec::new();
        let mut p_window_only_gt = Vec::new();
        for j in 7..9 {
            let gt = scene.ground_truth_points[j];
            let id = recon.add_point3d(gt + noise, track_for(&window_frames, j));
            p_window_only.push(id);
            p_window_only_gt.push(gt);
        }
        // Remaining points: full 12-frame track, never requested variable —
        // stay constant under both policies, providing direct (unconditional
        // `bc`/`diag`) constraints on the window frames' poses.
        let mut p_control = Vec::new();
        for j in 9..scene.ground_truth_points.len() {
            let gt = scene.ground_truth_points[j];
            let id = recon.add_point3d(gt, track_for(&all_frames, j));
            p_control.push(id);
        }

        let mut config = BundleAdjustmentConfig::new();
        for &f in &window_frames {
            config.add_frame(&recon, f);
        }
        config.fix_gauge(Gauge::ThreePoints);
        for &id in gauge_fodder.iter().chain(&p_pulled).chain(&p_window_only) {
            config.add_variable_point(id);
        }
        let window_images: Vec<ImageT> = window_frames
            .iter()
            .flat_map(|&f| {
                let (i1, i2) = scene.images_per_frame[f as usize];
                [i1, i2]
            })
            .collect();

        let outside_poses_before: Vec<_> = outside_frames
            .iter()
            .map(|&f| recon.frame(f).rig_from_world().clone())
            .collect();

        let mut options = BundleAdjustmentOptions::global();
        options.max_num_iterations = 60;

        // --- Colmap policy ---
        options.local_ba_point_policy = LocalBaPointPolicy::Colmap;
        assert!(
            solve(&options, &config, &mut recon),
            "Colmap-policy solve failed"
        );
        for (&f, before) in outside_frames.iter().zip(&outside_poses_before) {
            assert_eq!(
                recon.frame(f).rig_from_world(),
                before,
                "outside frame {f} moved under Colmap policy"
            );
        }
        for (j, &pid) in p_pulled.iter().enumerate() {
            let err = (recon.point3d(pid).xyz - p_pulled_gt[j]).norm();
            assert!(
                err < 0.01,
                "Colmap policy: pulled-in point {j} did not converge (error {err})"
            );
        }
        for (j, &pid) in p_window_only.iter().enumerate() {
            let err = (recon.point3d(pid).xyz - p_window_only_gt[j]).norm();
            assert!(
                err < 0.01,
                "Colmap policy: window-only point {j} did not converge (error {err})"
            );
        }
        let colmap_cost = window_reprojection_cost(&recon, &window_images);

        // --- WindowOnly policy, from an identical starting state ---
        let mut recon2 = build_recon();
        for j in 0..3 {
            recon2.add_point3d(scene.ground_truth_points[j], track_for(&[4, 5], j));
        }
        for j in 3..7 {
            recon2.add_point3d(
                scene.ground_truth_points[j] + noise,
                track_for(&all_frames, j),
            );
        }
        for j in 7..9 {
            recon2.add_point3d(
                scene.ground_truth_points[j] + noise,
                track_for(&window_frames, j),
            );
        }
        for j in 9..scene.ground_truth_points.len() {
            recon2.add_point3d(scene.ground_truth_points[j], track_for(&all_frames, j));
        }
        options.local_ba_point_policy = LocalBaPointPolicy::WindowOnly;
        assert!(
            solve(&options, &config, &mut recon2),
            "WindowOnly-policy solve failed"
        );
        for &f in &outside_frames {
            assert_eq!(
                recon2.frame(f).rig_from_world(),
                recon.frame(f).rig_from_world(),
                "outside frame {f} moved under WindowOnly policy"
            );
        }
        for (j, &pid) in p_pulled.iter().enumerate() {
            let perturbed = p_pulled_gt[j] + noise;
            assert_eq!(
                recon2.point3d(pid).xyz,
                perturbed,
                "WindowOnly policy: out-of-window point {j} should stay exactly constant"
            );
        }
        for (j, &pid) in p_window_only.iter().enumerate() {
            let err = (recon2.point3d(pid).xyz - p_window_only_gt[j]).norm();
            assert!(
                err < 0.01,
                "WindowOnly policy: window-only point {j} did not converge (error {err})"
            );
        }
        let window_only_cost = window_reprojection_cost(&recon2, &window_images);

        // `WindowOnly` cannot reach zero window-only cost here: `p_pulled`'s
        // points stay frozen at their perturbed positions (constant), yet
        // *are* observed by window images (their track spans all 12
        // frames), so those specific residuals carry irreducible error the
        // window frames' poses cannot compensate away. `Colmap` *can* move
        // those points (pull-in sees their whole track), so on this
        // noiseless, single-ground-truth scene it reaches (about) zero
        // window-only cost — strictly better than (never worse than)
        // `WindowOnly`'s.
        assert!(
            colmap_cost <= window_only_cost + 1.0e-6,
            "Colmap policy window-only cost {colmap_cost} exceeds WindowOnly's {window_only_cost}"
        );
        assert!(
            colmap_cost < 1.0e-6,
            "Colmap policy window-only cost {colmap_cost} not near zero"
        );
        assert!(
            window_only_cost > 1.0,
            "expected WindowOnly's frozen out-of-window points to leave a non-trivial \
             window-only residual (cost {window_only_cost}) — otherwise this test isn't \
             exercising the intended contrast"
        );

        // --- VariableWithoutPullIn policy (the actual pre-C2.6 rule),
        // from an identical starting state ---
        let mut recon3 = build_recon();
        for j in 0..3 {
            recon3.add_point3d(scene.ground_truth_points[j], track_for(&[4, 5], j));
        }
        for j in 3..7 {
            recon3.add_point3d(
                scene.ground_truth_points[j] + noise,
                track_for(&all_frames, j),
            );
        }
        for j in 7..9 {
            recon3.add_point3d(
                scene.ground_truth_points[j] + noise,
                track_for(&window_frames, j),
            );
        }
        for j in 9..scene.ground_truth_points.len() {
            recon3.add_point3d(scene.ground_truth_points[j], track_for(&all_frames, j));
        }
        options.local_ba_point_policy = LocalBaPointPolicy::VariableWithoutPullIn;
        assert!(
            solve(&options, &config, &mut recon3),
            "VariableWithoutPullIn-policy solve failed"
        );
        for &f in &outside_frames {
            assert_eq!(
                recon3.frame(f).rig_from_world(),
                recon.frame(f).rig_from_world(),
                "outside frame {f} moved under VariableWithoutPullIn policy"
            );
        }
        // Unlike `WindowOnly`, these points are free (not frozen at the
        // perturbed value) — optimized from only their in-window
        // observations, which on this noiseless scene are already
        // sufficient (6 window frames x 2 cams) to pin the point back near
        // ground truth without needing the pulled-in outside observations.
        for (j, &pid) in p_pulled.iter().enumerate() {
            let perturbed = p_pulled_gt[j] + noise;
            let moved = (recon3.point3d(pid).xyz - perturbed).norm();
            assert!(
                moved > 1.0e-4,
                "VariableWithoutPullIn policy: out-of-window point {j} did not move (was constant before)"
            );
            let err = (recon3.point3d(pid).xyz - p_pulled_gt[j]).norm();
            assert!(
                err < 0.01,
                "VariableWithoutPullIn policy: out-of-window point {j} did not converge \
                 using in-window observations only (error {err})"
            );
        }
        for (j, &pid) in p_window_only.iter().enumerate() {
            let err = (recon3.point3d(pid).xyz - p_window_only_gt[j]).norm();
            assert!(
                err < 0.01,
                "VariableWithoutPullIn policy: window-only point {j} did not converge (error {err})"
            );
        }

        let _ = p_control; // kept alive: constrains the window frames above.
    }
}
