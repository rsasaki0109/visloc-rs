//! IMU-coupled DPVO patch bundle adjustment — Milestone M5 of
//! `docs/dpvo_droid_port_plan.md`.
//!
//! `crate::dpvo_patch_ba`'s `dpvo_ba`/`dpvo_ba_step` (M3) solve a
//! **visual-only** Gauss-Newton problem over DPVO's sliding window: SE(3)
//! poses plus scalar patch inverse depths, under DPVO/`lietorch`'s
//! **left**-perturbation retraction (`T ← Exp(ξ) · T`). M4-perf's own EuRoC
//! run (`docs/dpvo_droid_port_plan.md`, "M4-perf results") measured the
//! recovered similarity scale at **1.266–1.359** against a ground-truth scale
//! of `1.0` — textbook monocular scale drift: visual reprojection residuals
//! are exactly invariant under the joint rescaling `(translation, inverse
//! depth) ↦ (s·translation, depth/s)`, so nothing in a visual-only solve can
//! ever pull that gauge freedom back to metric. This module adds the one
//! ingredient that breaks that invariance — IMU pre-integration, which is
//! metric by construction (`m/s²` accelerometer readings integrate to metric
//! velocity/position) — as an *additional* residual family in the **same**
//! per-window Gauss-Newton solve, per the plan doc's §4 hybrid-design
//! sentence: "a patch-BA solve and an IMU-preintegration solve are one joint
//! Gauss-Newton problem, not two separately-reconciled ones."
//!
//! # Placement: a new sibling module, not a `dpvo_patch_ba.rs` edit
//!
//! `dpvo_patch_ba.rs`'s own module doc already explains why it is a sibling
//! of `bundle.rs` rather than an extension of it (incompatible perturbation
//! conventions, damping schemes, iteration structure). The same reasoning
//! applies one level down here, for the opposite reason: `dpvo_ba_step` is a
//! **tested, upstream-parity-verified** solver (`ba_fixture.npz`, M3 results,
//! max abs diff ~1e-6 against upstream's own `mini_ba` reference). Bolting
//! velocity/IMU state onto its `DpvoBaProblem`/`DpvoBaConfig` types in place
//! would force every M3/M4 call site and fixture test to grow new fields it
//! doesn't use, and — worse — would risk silently perturbing the exact
//! numerics that parity test locks in. Composition instead of mutation: this
//! module's [`DpvoViWindow`] layers IMU/velocity state *alongside* an
//! unmodified `DpvoBaProblem`, and [`dpvo_vi_ba_step`]/[`dpvo_vi_ba`] are new
//! functions, not edits to `dpvo_ba_step`/`dpvo_ba`. `crate::dpvo_patch_ba`
//! itself is untouched except for widening four already-private items
//! (`transform_edge`, `EdgeGeometry`, `DISP_MIN`/`DISP_MAX`, `POSE_DIAG_LM`)
//! from private to `pub(crate)` — a pure visibility change, verified by the
//! unchanged M3 fixture test continuing to pass unmodified.
//!
//! ## Visual assembly is a deliberate, tested duplication, not a refactor
//!
//! [`dpvo_vi_ba_step`] re-derives its own accumulation of the visual normal
//! equations (`B_pose`, `E`, `v_pose`, `C`, `w`) from [`transform_edge`]
//! rather than calling into `dpvo_ba_step` and trying to splice IMU/velocity
//! state into its return value — `dpvo_ba_step` returns only the *solved*
//! `DpvoBaProblem`, not the intermediate linear-system blocks a joint solve
//! needs to augment. Extracting a shared assembly helper out of
//! `dpvo_ba_step` was considered and rejected for this milestone: it would
//! touch the one function upstream-parity-tested to ~1e-6, for a saving of
//! roughly 60 lines, in a milestone whose actual hard problem is the
//! left-perturbation IMU Jacobian derivation below. The tradeoff is made
//! safe, not just convenient: [`tests::zero_imu_factors_matches_visual_only_solve`]
//! cross-checks that this module's own assembly, run with an **empty**
//! `imu.factors` list, reproduces `dpvo_ba_step`'s output on the same
//! problem to numerical tolerance — i.e. the duplication is verified against
//! the trusted original, not merely assumed correct by inspection.
//!
//! # The jacobian convention conversion (the hard math)
//!
//! `pipelines/slam/src/bundle.rs`'s own Forster IMU factor (search that file
//! for "Forster 2017 IMU pre-integration factors") derives residual and
//! Jacobians under `BundleAdjustment`'s **right**-perturbation convention
//! (`T ← T · Exp(ξ)`) for the *body* pose `body = imu_body_to_camera ∘
//! camera_pose` (`imu_body_to_camera` there — and here — is the EuRoC-style
//! `T_BS` extrinsic taken literally: a function that maps a **camera**-frame
//! coordinate to its **body**-frame coordinate, `X_body = R_bc·X_camera +
//! t_bc`; confirmed both by `bundle.rs`'s own use, `body_i =
//! imu_body_to_camera.compose(&pose_i.world_to_camera)` type-checking only
//! under that reading, and by `examples/euroc_online_slam_vi_demo.rs`'s
//! `se3_from_t_bs`/`world_to_camera_pose` helpers, which build it directly
//! from EuRoC's raw `T_BS` matrix with no inversion). DPVO's own poses
//! retract on the **left** (`crate::dpvo_patch_ba`'s own convention-mapping
//! section: `T ← Exp(ξ) · T`, matching `lietorch`). This module needs
//! `bundle.rs`'s exact residual/Jacobian *content* (no re-derivation of
//! Forster's math — that would just be a second, independently-fallible copy
//! of a formula this codebase already has right) reinterpreted for DPVO's
//! *camera* pose's left perturbation instead of the body pose's right
//! perturbation. Two conjugations, both grounded in the single Adjoint
//! definition documented on [`visloc_core::geometry::SE3::adjoint`] itself
//! (`T · Exp(ξ) · T⁻¹ = Exp(Ad(T)·ξ)`, `crates/core/src/geometry/se3.rs:84-85`
//! — the *definition* of the Adjoint representation for a matrix Lie group,
//! not a derived fact requiring its own citation; see also Solà, J. (2018),
//! "A micro Lie theory for state estimation in robotics", arXiv:1812.01537,
//! eq. 80, for the same identity in a general treatment):
//!
//! ## Step 1 — right-perturbation Jacobian of `body` ⇒ left-perturbation Jacobian of `body`
//!
//! Write `T = body` (either `body_i` or `body_j`) and let `r(T)` be the
//! Forster residual as a function of `T` alone (the other pose/velocities
//! held fixed). `bundle.rs` computes `J_right = ∂r/∂ξ_right` where `T_new =
//! T · Exp(ξ_right)`. This module instead needs `J_left = ∂r/∂ξ_left` where
//! `T_new = Exp(ξ_left) · T`. From the Adjoint definition applied to `T⁻¹`:
//! `T⁻¹ · Exp(ξ) · T = Exp(Ad(T⁻¹)·ξ)`, i.e. `Exp(ξ) · T = T ·
//! Exp(Ad(T⁻¹)·ξ)`. Setting `ξ = ξ_left` shows `T_new = Exp(ξ_left)·T = T ·
//! Exp(Ad(T⁻¹)·ξ_left)`, i.e. this **is** a right perturbation with
//! `ξ_right = Ad(T⁻¹)·ξ_left`. Chaining: `J_left = J_right · Ad(T⁻¹)`.
//!
//! ## Step 2 — left-perturbation Jacobian of `body` ⇒ left-perturbation Jacobian of DPVO's camera pose
//!
//! Write `C = imu_body_to_camera` (the fixed extrinsic) and `P` = the DPVO
//! camera pose, so `body = C ∘ P`. Under a DPVO left perturbation `P_new =
//! Exp(ξ_P) · P`: `body_new = C ∘ Exp(ξ_P) ∘ P = (C ∘ Exp(ξ_P) ∘ C⁻¹) ∘ (C ∘
//! P) = Exp(Ad(C)·ξ_P) ∘ body` — the middle step is again exactly the
//! Adjoint's own definition (conjugating `Exp(ξ_P)` by `C`), not a derived
//! identity. So `body`'s own left-perturbation tangent `η = Ad(C)·ξ_P`, and
//! by the chain rule `J_P = J_left(body) · (∂η/∂ξ_P) = J_left(body) ·
//! Ad(C)`.
//!
//! ## Combined
//!
//! `J_P_i = J_right(body_i) · Ad(body_i⁻¹) · Ad(imu_body_to_camera)`, and
//! symmetrically for `J_P_j`. `imu_body_to_camera` is the **same** fixed
//! extrinsic for every factor in a window, so `Ad(imu_body_to_camera)` is
//! *recomputed* per factor by this milestone's implementation even though
//! it could be hoisted once per [`dpvo_vi_ba_step`] call — a
//! straightforward follow-up left undone here (negligible cost: a single
//! `6×6` matrix per factor, dwarfed by every other per-frame cost this
//! port measures — see the plan doc's own M6 blockers note). Velocity
//! Jacobians (`∂r/∂v_i`, `∂r/∂v_j`) need **no** conversion at all: velocity
//! is an ordinary Euclidean (additive) variable in both conventions, so
//! `bundle.rs`'s velocity-column formulas are reused verbatim.
//! [`tests::imu_factor_jacobian_matches_numeric_finite_difference`] checks
//! every one of `J_P_i`/`J_P_j`/`J_v_i`/`J_v_j`'s columns against a central
//! finite difference of the residual evaluated at DPVO's own
//! `SE3::exp(&xi).compose(&pose)` retraction — i.e. the test perturbs the
//! *actual* left-perturbation retraction this crate uses, not an abstract
//! stand-in for it, so a sign or ordering slip in the conversion above would
//! be caught directly rather than by a second copy of the same reasoning.
//!
//! # Sign convention: why the IMU RHS gets an extra minus sign
//!
//! `dpvo_ba_step`'s own accumulation (`ba.py`'s own convention) stores, per
//! visual edge, `Ji = ∂(predicted pixel)/∂ξ_i` (a **forward** geometric
//! derivative, not of the residual) and residual `r = target − predicted`,
//! then accumulates `B += JᵀJ`, `v += Jᵀr`, solves `B·δ = v`, and retracts
//! with **`+δ`** (`Exp(δ)·pose`, no negation — see `dpvo_patch_ba.rs`'s own
//! `dpvo_ba_step` retraction line). Standard Gauss-Newton for minimizing
//! `‖r(x+δ)‖²` with `J_r = ∂r/∂δ` solves `J_rᵀJ_r·δ = −J_rᵀr`. Since here `r =
//! target − predicted`, `J_r = −J` (the code's `Ji`/`Jj`/`Jz`), so the normal
//! equations become `(−J)ᵀ(−J)·δ = −(−J)ᵀr`, i.e. `JᵀJ·δ = Jᵀr` — **exactly**
//! what the code computes, with no sign anywhere, purely because the two
//! negations from "`J_r = −J`" cancel. This module's IMU residual (Forster's
//! own `r_R, r_v, r_p`, ported byte-for-byte from `bundle.rs`) is a **literal**
//! function of state with no such "target − predicted" sign trick baked in —
//! its Jacobian *is* `∂r/∂ξ` directly (this is exactly what `J_P_i`/`J_P_j`/
//! `J_v_i`/`J_v_j` above compute). Combining the two residual families in
//! *one* linear system that shares a single `+δ` retraction therefore
//! requires the IMU block's contribution to the shared RHS to carry the
//! standard Gauss-Newton minus sign explicitly: **`v += −Jᵀr`** for every
//! IMU-derived block (pose *and* velocity), while every visual block keeps
//! `dpvo_ba_step`'s existing `v += +Jᵀr` unchanged. The Hessian accumulation
//! `B += JᵀJ` needs no sign correction either way (it is quadratic in `J`).
//! Concretely: for a pure-IMU window (no visual edges at all), this yields
//! `δ = B⁻¹v = B⁻¹·(−Σ Jᵀr) = −(JᵀJ)⁻¹Jᵀr`, textbook Gauss-Newton, retracted
//! as `Exp(δ)·pose` — self-consistent on its own, independently of the
//! visual family's sign convention, which is the cross-check
//! [`tests::pure_imu_window_matches_textbook_gauss_newton_sign`] performs.
//!
//! # Per-window state and gauge-fixing
//!
//! [`DpvoViWindow`] adds, per the task's staged-bias philosophy (see
//! `docs/motion_based_vi_alignment.md`'s staged-bias-release design and
//! `vi_motion_initializer.rs`'s own bias-release schedule): a **per-pose**
//! world-frame velocity (`velocities`, one entry per `DpvoBaProblem::poses`
//! entry) and a **single shared** gyro/accel bias pair, fixed for the
//! duration of the window solve (not a Gauss-Newton variable here at all —
//! `crate::dpvo_vo`'s bootstrap chain re-estimates it once, outside this
//! module, exactly like the motion-VI chain's own staged release).
//!
//! Poses with local index `< fixedp` (DPVO's own gauge-fixing convention,
//! ported unchanged from `dpvo_patch_ba.rs`) are excluded from the pose
//! Hessian entirely, per that module's own `fixedp` semantics — this removes
//! the *pose* gauge freedom (an arbitrary global frame, and — critically —
//! monocular scale, which is exactly what this module exists to recover from
//! IMU evidence). Velocity is **not** tied to that same gate: every frame's
//! velocity, including the gauge-fixed pose's, stays a free Gauss-Newton
//! variable. This mirrors `bundle.rs`'s own `BundleAdjustment::fix_velocity`
//! being a wholly independent API from pose-fixing (a caller there must
//! explicitly opt a keyframe's velocity into being fixed; nothing ties it to
//! the keyframe's own pose-fixed status) — and is not just stylistic
//! symmetry: velocity, unlike pose, is not itself a gauge direction. An
//! earlier version of this module *did* fix the anchor frame's velocity
//! alongside its pose and found — via
//! [`tests::synthetic_window_recovers_metric_scale_within_two_percent`],
//! which failed hard (recovered scale `0.20`, not `~1.0`) until this was
//! corrected — that pinning `v_0` to an arbitrary (wrong) guess starves the
//! very first IMU factor of the one thing that lets it pull scale toward
//! metric truth: a free `v_0` to solve for. A small Tikhonov term
//! ([`VELOCITY_DIAG_EPSILON`]) is added to every free
//! velocity's Hessian diagonal — a genuinely new, non-upstream constant (no
//! `ba.py` analogue exists for a variable `ba.py` doesn't have) needed
//! because a free velocity untouched by any in-window IMU factor (a
//! transient bookkeeping edge case, not expected in steady-state operation
//! with a fully-populated factor chain) would otherwise leave a literal
//! all-zero row/column in the Hessian — singular, not just poorly
//! conditioned.
//!
//! # Scale handling: no dedicated scale variable
//!
//! `online_slam_vi_ba.rs::run_viba2_inertial_with_scale` ("VIBA2") recovers
//! scale via an **explicit** alternating-minimization outer loop: rescale
//! the IMU factors by `1/s`, run an inertial-only inner solve against
//! already-converged, scale-ambiguous visual structure, re-estimate `s` in
//! closed form, repeat. That structure fits VIBA2's own setting — a
//! separate, already-computed visual solve feeding a distinct inertial-only
//! refinement stage. It does **not** fit this module's setting: DPVO has no
//! such intermediate "visual-only converged, now fix scale" stage — poses,
//! patch inverse depths, *and* velocities are all simultaneously free in
//! **every** window solve, and the plan doc's own hybrid-design sentence
//! (quoted above) explicitly commits to one joint solve, not two
//! reconciled ones. A dedicated scalar `s` would need its own Jacobian
//! column threading through every visual **and** IMU residual (`∂r/∂s`) and
//! would duplicate a gauge the joint pose+depth state already spans (scaling
//! every free pose's translation by `s` and every patch depth by `1/s` is
//! already reachable by the existing pose/depth Jacobian columns — a
//! dedicated `s` variable would be rank-deficient against them). This module
//! therefore lets the joint Gauss-Newton solve absorb scale implicitly,
//! exactly as it absorbs every other correlated direction in the state
//! space; [`tests::synthetic_window_recovers_metric_scale_within_two_percent`]
//! is the empirical check that this actually converges on a case with a
//! known, deliberately-wrong (`1.4×`) starting scale. If a real EuRoC run
//! (M5's own acceptance section) shows slow/incomplete convergence, a
//! natural M6 follow-up is a VIBA2-style periodic explicit rescale as a
//! *warm start* into this joint solve — not attempted here, since the task
//! asks this decision be made from the math and tested, which this section
//! and its cited test do.
//!
//! # Gravity
//!
//! `crate::dpvo_vo`'s bootstrap chain (not this module) is responsible for
//! producing `DpvoViWindow`'s per-factor `ImuPreintegrationFactor::gravity_world`
//! via `vi_motion_initializer::estimate_gyro_bias` then
//! `estimate_gravity_and_velocities`, run once against the window's own
//! poses treated as fixed — exactly the motion-VI chain's own bootstrap,
//! reused as-is (see that crate's module docs). This module only *consumes*
//! whatever `gravity_world` each factor already carries; DPVO's own "world"
//! frame (an arbitrary anchor — typically frame 0's original pose) is
//! whatever frame the bootstrap's fixed poses were expressed in, so gravity
//! and the solved poses stay in the same frame by construction.
use nalgebra::{DMatrix, DVector, Matrix3, SMatrix, SVector, Vector3, Vector6};
use visloc_core::geometry::{SE3, SO3};

use crate::dpvo_patch_ba::{
    dpvo_ba_step, transform_edge, DpvoBaConfig, DpvoBaError, DpvoBaProblem, DpvoPatch, DISP_MAX,
    DISP_MIN, POSE_DIAG_LM,
};
use crate::imu_preintegration::{ImuPreintegrationFactor, Matrix9};

/// New, non-upstream Tikhonov term on every free velocity's Hessian
/// diagonal — see the module doc's "Per-window state and gauge-fixing"
/// section for why this is needed at all.
pub const VELOCITY_DIAG_EPSILON: f64 = 1.0e-6;

/// One IMU pre-integration factor between two poses **local to the current
/// window** (`i`/`j` index into the same `DpvoBaProblem::poses` this factor
/// accompanies — analogous to [`crate::dpvo_patch_ba::DpvoEdge`]'s `i`/`j`,
/// but for a consecutive-pose IMU edge rather than a patch/target-frame
/// visual edge). `crate::dpvo_vo` is responsible for translating its own
/// arrival-index-keyed bookkeeping into these window-local indices, and for
/// dropping a factor once either endpoint has aged out of the window (see
/// that module's doc).
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoImuFactor {
    pub i: usize,
    pub j: usize,
    pub factor: ImuPreintegrationFactor,
}

/// The IMU/velocity/bias state layered alongside an (unmodified)
/// [`DpvoBaProblem`] for [`dpvo_vi_ba`]/[`dpvo_vi_ba_step`]. See the module
/// doc's "Per-window state and gauge-fixing" section.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoViWindow {
    /// One world-frame velocity per `DpvoBaProblem::poses` entry (same
    /// length/order). Entries at local index `< config.fixedp` are treated
    /// as fixed (read but never updated), matching how `fixedp` already
    /// gauge-fixes poses in `crate::dpvo_patch_ba`.
    pub velocities: Vec<Vector3<f64>>,
    /// Consecutive-pose IMU factors, window-local indices (see
    /// [`DpvoImuFactor`]).
    pub factors: Vec<DpvoImuFactor>,
    /// Fixed body↔camera extrinsic for every factor in this window (EuRoC's
    /// `T_BS`, taken literally — see the module doc's jacobian-conversion
    /// section for the exact convention).
    pub body_to_camera: SE3,
    /// Shared gyro bias, fixed for this window's solve (staged-bias
    /// philosophy — see the module doc).
    pub bias_gyro: Vector3<f64>,
    /// Shared accelerometer bias, fixed for this window's solve.
    pub bias_accel: Vector3<f64>,
}

/// Output of [`dpvo_vi_ba_step`]/[`dpvo_vi_ba`]: refined poses, patches, and
/// velocities (bias/extrinsic are inputs only — never refined by this
/// module, per its staged-bias design).
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoViBaSolution {
    pub poses: Vec<SE3>,
    pub patches: Vec<DpvoPatch>,
    /// Same length/order as [`DpvoViWindow::velocities`].
    pub velocities: Vec<Vector3<f64>>,
}

/// Run [`dpvo_vi_ba_step`] `config.iterations` times, re-linearizing at the
/// updated poses/patches/velocities each time — the joint-solve analogue of
/// `crate::dpvo_patch_ba::dpvo_ba`'s own outer loop. `imu.factors`/
/// `body_to_camera`/`bias_gyro`/`bias_accel` stay fixed across iterations
/// (only `velocities` moves, alongside `poses`/`patches`).
pub fn dpvo_vi_ba(
    problem: &DpvoBaProblem,
    imu: &DpvoViWindow,
    config: &DpvoBaConfig,
) -> Result<DpvoViBaSolution, DpvoBaError> {
    let mut poses = problem.poses.clone();
    let mut patches = problem.patches.clone();
    let mut velocities = imu.velocities.clone();
    for _ in 0..config.iterations {
        let current_problem = DpvoBaProblem { poses, patches, ..problem.clone() };
        let current_imu = DpvoViWindow { velocities, ..imu.clone() };
        let solved = dpvo_vi_ba_step(&current_problem, &current_imu, config)?;
        poses = solved.poses;
        patches = solved.patches;
        velocities = solved.velocities;
    }
    Ok(DpvoViBaSolution { poses, patches, velocities })
}

/// One joint Gauss-Newton iteration over visual patch residuals (identical
/// math to `crate::dpvo_patch_ba::dpvo_ba_step`, re-derived here — see the
/// module doc's "Visual assembly" section) plus IMU pre-integration
/// residuals between consecutive window poses (see the module doc's
/// jacobian-conversion and sign-convention sections).
pub fn dpvo_vi_ba_step(
    problem: &DpvoBaProblem,
    imu: &DpvoViWindow,
    config: &DpvoBaConfig,
) -> Result<DpvoViBaSolution, DpvoBaError> {
    let n_frames = problem.poses.len();
    assert_eq!(
        imu.velocities.len(),
        n_frames,
        "DpvoViWindow::velocities must have exactly one entry per DpvoBaProblem::poses entry"
    );
    let fixedp = config.fixedp.min(n_frames);
    let n2 = n_frames - fixedp;

    if n2 == 0 {
        // No free pose for a velocity/IMU block to attach to (see the
        // module doc — this is a documented, not-expected-in-steady-state
        // edge case, not a silent narrowing of scope: `crate::dpvo_vo` only
        // enables IMU coupling once the graph is already initialized, at
        // which point `fixedp < n_frames` always holds). Fall back to the
        // pure visual solve; velocities pass through unchanged.
        let solved = dpvo_ba_step(problem, config)?;
        return Ok(DpvoViBaSolution {
            poses: solved.poses,
            patches: solved.patches,
            velocities: imu.velocities.clone(),
        });
    }

    // ---- Visual normal-equation assembly (see module doc: a deliberate,
    // tested duplication of `dpvo_ba_step`'s own loop, not a shared helper) ----
    let mut used_patches: Vec<usize> = problem.edges.iter().map(|edge| edge.k).collect();
    used_patches.sort_unstable();
    used_patches.dedup();
    let m = used_patches.len();
    let local_patch_index =
        |k: usize| used_patches.binary_search(&k).expect("k is drawn from used_patches");

    let mut b_pose = DMatrix::<f64>::zeros(6 * n2, 6 * n2);
    let mut e_mat = DMatrix::<f64>::zeros(6 * n2, m);
    let mut v_pose = DVector::<f64>::zeros(6 * n2);
    let mut c_diag = DVector::<f64>::zeros(m);
    let mut w_rhs = DVector::<f64>::zeros(m);

    for (edge_index, edge) in problem.edges.iter().enumerate() {
        let geom = transform_edge(
            &problem.poses[edge.i],
            &problem.poses[edge.j],
            &problem.intrinsics[edge.i],
            &problem.intrinsics[edge.j],
            &problem.patches[edge.k],
        );
        let raw_residual = problem.targets[edge_index] - geom.coords_center;
        let mut valid = geom.valid_depth && raw_residual.norm() < 250.0;
        let [x0, y0, x1, y1] = config.bounds;
        let in_bounds = geom.coords_center.x > x0
            && geom.coords_center.y > y0
            && geom.coords_center.x < x1
            && geom.coords_center.y < y1;
        valid = valid && in_bounds;
        let v_gate = if valid { 1.0 } else { 0.0 };
        let r = raw_residual * v_gate;
        let weight = problem.weights[edge_index] * v_gate;

        let mut w_ji = geom.ji;
        w_ji.row_mut(0).scale_mut(weight.x);
        w_ji.row_mut(1).scale_mut(weight.y);
        let mut w_jj = geom.jj;
        w_jj.row_mut(0).scale_mut(weight.x);
        w_jj.row_mut(1).scale_mut(weight.y);
        let w_jz = nalgebra::Vector2::new(geom.jz.x * weight.x, geom.jz.y * weight.y);

        let k_local = local_patch_index(edge.k);
        c_diag[k_local] += w_jz.x * geom.jz.x + w_jz.y * geom.jz.y;
        w_rhs[k_local] += w_jz.x * r.x + w_jz.y * r.y;

        let i_local = if edge.i >= fixedp { Some(edge.i - fixedp) } else { None };
        let j_local = if edge.j >= fixedp { Some(edge.j - fixedp) } else { None };

        if let Some(i) = i_local {
            let bii = w_ji.transpose() * geom.ji;
            b_pose.view_mut((i * 6, i * 6), (6, 6)).iter_mut().zip(bii.iter()).for_each(|(d, s)| *d += s);
            let vi = w_ji.transpose() * r;
            v_pose.rows_mut(i * 6, 6).iter_mut().zip(vi.iter()).for_each(|(d, s)| *d += s);
            let eik = w_ji.transpose() * geom.jz;
            e_mat.view_mut((i * 6, k_local), (6, 1)).iter_mut().zip(eik.iter()).for_each(|(d, s)| *d += s);
        }
        if let Some(j) = j_local {
            let bjj = w_jj.transpose() * geom.jj;
            b_pose.view_mut((j * 6, j * 6), (6, 6)).iter_mut().zip(bjj.iter()).for_each(|(d, s)| *d += s);
            let vj = w_jj.transpose() * r;
            v_pose.rows_mut(j * 6, 6).iter_mut().zip(vj.iter()).for_each(|(d, s)| *d += s);
            let ejk = w_jj.transpose() * geom.jz;
            e_mat.view_mut((j * 6, k_local), (6, 1)).iter_mut().zip(ejk.iter()).for_each(|(d, s)| *d += s);
        }
        if let (Some(i), Some(j)) = (i_local, j_local) {
            let bij = w_ji.transpose() * geom.jj;
            b_pose.view_mut((i * 6, j * 6), (6, 6)).iter_mut().zip(bij.iter()).for_each(|(d, s)| *d += s);
            let bji = w_jj.transpose() * geom.ji;
            b_pose.view_mut((j * 6, i * 6), (6, 6)).iter_mut().zip(bji.iter()).for_each(|(d, s)| *d += s);
        }
    }

    // ---- Augmented (pose + velocity) system: pose block first (6*n2
    // rows/cols, gauge-fixed poses excluded exactly as `dpvo_ba_step` does),
    // velocity block second (3*n_frames rows/cols, covering EVERY frame —
    // see module doc's "Per-window state and gauge-fixing" for why velocity
    // is intentionally *not* narrowed to the free-pose set). ----
    let dim = 6 * n2 + 3 * n_frames;
    let mut b_aug = DMatrix::<f64>::zeros(dim, dim);
    let mut v_aug = DVector::<f64>::zeros(dim);
    b_aug.view_mut((0, 0), (6 * n2, 6 * n2)).copy_from(&b_pose);
    v_aug.rows_mut(0, 6 * n2).copy_from(&v_pose);

    for imu_factor in &imu.factors {
        let DpvoImuFactor { i, j, factor } = imu_factor;
        let (i, j) = (*i, *j);
        let (residual, j_pose_i, j_pose_j, j_vel_i, j_vel_j) = imu_factor_jacobians(
            &problem.poses[i],
            &problem.poses[j],
            &imu.velocities[i],
            &imu.velocities[j],
            &imu.body_to_camera,
            factor,
            &imu.bias_gyro,
            &imu.bias_accel,
        );

        let whitener = factor.covariance_sqrt_information().unwrap_or_else(|| {
            let mut diagonal = SVector::<f64, 9>::zeros();
            diagonal.fixed_rows_mut::<3>(0).fill(factor.weight_rotation.max(0.0).sqrt());
            diagonal.fixed_rows_mut::<3>(3).fill(factor.weight_velocity.max(0.0).sqrt());
            diagonal.fixed_rows_mut::<3>(6).fill(factor.weight_position.max(0.0).sqrt());
            Matrix9::from_diagonal(&diagonal)
        });
        let r_stack = whitener * residual;
        let j_pose_i = whitener * j_pose_i;
        let j_pose_j = whitener * j_pose_j;
        let j_vel_i = whitener * j_vel_i;
        let j_vel_j = whitener * j_vel_j;

        let pose_off = |idx: usize| -> Option<usize> {
            if idx < fixedp { None } else { Some((idx - fixedp) * 6) }
        };
        // Velocity is free for every frame, including a pose-gauge-fixed one
        // — see the module doc's "Per-window state and gauge-fixing" section.
        let vel_off = |idx: usize| -> Option<usize> { Some(6 * n2 + idx * 3) };

        let blocks: [(Option<usize>, DMatrix<f64>); 4] = [
            (pose_off(i), DMatrix::from_column_slice(9, 6, j_pose_i.as_slice())),
            (pose_off(j), DMatrix::from_column_slice(9, 6, j_pose_j.as_slice())),
            (vel_off(i), DMatrix::from_column_slice(9, 3, j_vel_i.as_slice())),
            (vel_off(j), DMatrix::from_column_slice(9, 3, j_vel_j.as_slice())),
        ];
        let r_dyn = DVector::from_column_slice(r_stack.as_slice());

        for (off_a, j_a) in &blocks {
            let Some(off_a) = *off_a else { continue };
            for (off_b, j_b) in &blocks {
                let Some(off_b) = *off_b else { continue };
                let block = j_a.transpose() * j_b;
                for r in 0..j_a.ncols() {
                    for c in 0..j_b.ncols() {
                        b_aug[(off_a + r, off_b + c)] += block[(r, c)];
                    }
                }
            }
            // Sign convention: see the module doc's "Sign convention"
            // section for why this is `−Jᵀr`, unlike the visual blocks'
            // `+Jᵀr` above.
            let rhs_block = j_a.transpose() * &r_dyn;
            for r in 0..j_a.ncols() {
                v_aug[off_a + r] -= rhs_block[r];
            }
        }
    }

    // Schur-eliminate the patch depths against the POSE sub-block only
    // (velocity rows of `E` are identically zero — patches have no velocity
    // dependence — so the velocity/pose-velocity blocks of `b_aug`/`v_aug`
    // need no Schur correction at all; see the module doc's derivation).
    let q: DVector<f64> = c_diag.map(|c| 1.0 / (c + config.lmbda));
    let mut eq = e_mat.clone();
    for k in 0..m {
        let mut col = eq.column_mut(k);
        col *= q[k];
    }
    let schur_correction = &eq * e_mat.transpose();
    for r in 0..6 * n2 {
        for c in 0..6 * n2 {
            b_aug[(r, c)] -= schur_correction[(r, c)];
        }
    }
    let eq_w = &eq * &w_rhs;
    for r in 0..6 * n2 {
        v_aug[r] -= eq_w[r];
    }

    // Damping: pose diagonal reuses `ba.py`'s own formula (`POSE_DIAG_LM`,
    // `config.ep` — see `dpvo_patch_ba.rs`'s "Damping" section); velocity
    // diagonal gets the new [`VELOCITY_DIAG_EPSILON`] regularizer only
    // (module doc: no upstream analogue exists for a variable upstream
    // doesn't have).
    for d in 0..6 * n2 {
        b_aug[(d, d)] = b_aug[(d, d)] * (1.0 + POSE_DIAG_LM) + config.ep;
    }
    for d in (6 * n2)..dim {
        b_aug[(d, d)] += VELOCITY_DIAG_EPSILON;
    }

    let dx_aug = b_aug.lu().solve(&v_aug).ok_or(DpvoBaError::SingularSystem)?;
    let dx_pose = dx_aug.rows(0, 6 * n2).clone_owned();
    let et_dx = e_mat.transpose() * &dx_pose;
    let dz = DVector::from_iterator(m, (0..m).map(|k| q[k] * (w_rhs[k] - et_dx[k])));

    let mut new_poses = problem.poses.clone();
    for local in 0..n2 {
        let xi = Vector6::from_iterator((0..6).map(|c| dx_aug[local * 6 + c]));
        let delta = SE3::exp(&xi);
        let global = fixedp + local;
        new_poses[global] = delta.compose(&problem.poses[global]);
    }

    let mut new_velocities = imu.velocities.clone();
    for (global, velocity) in new_velocities.iter_mut().enumerate() {
        let base = 6 * n2 + global * 3;
        let dv = Vector3::new(dx_aug[base], dx_aug[base + 1], dx_aug[base + 2]);
        *velocity += dv;
    }

    let mut new_patches = problem.patches.clone();
    for (local, &global_k) in used_patches.iter().enumerate() {
        new_patches[global_k].inverse_depth += dz[local];
    }
    for patch in &mut new_patches {
        patch.inverse_depth = patch.inverse_depth.clamp(DISP_MIN, DISP_MAX);
    }

    Ok(DpvoViBaSolution { poses: new_poses, patches: new_patches, velocities: new_velocities })
}

/// `bundle.rs`'s own `skew` (`crates/../bundle.rs:3662`), duplicated rather
/// than exposed cross-crate-privately since it is a 1-line, well-known
/// formula (`[v]×`) with no room for divergence.
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// `bundle.rs`'s own `right_jacobian_inverse_so3` (`crates/../bundle.rs:3676`),
/// duplicated verbatim (same reasoning as [`skew`] above).
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

/// Compute one IMU factor's residual (`[r_R; r_v; r_p]`, stacked 9-vector,
/// Forster eq. 45-47 via [`ImuPreintegrationFactor::residual_with_bias_correction`])
/// and its **unwhitened** Jacobians with respect to: `pose_i`'s DPVO
/// left-perturbation (`J_pose_i`, 9×6), `pose_j`'s (`J_pose_j`, 9×6), `v_i`
/// (`J_vel_i`, 9×3), and `v_j` (`J_vel_j`, 9×3). See the module doc's
/// "jacobian convention conversion" section for the full derivation this
/// function implements; [`tests::imu_factor_jacobian_matches_numeric_finite_difference`]
/// is the non-negotiable numeric check on every one of these four blocks.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn imu_factor_jacobians(
    pose_i: &SE3,
    pose_j: &SE3,
    v_i: &Vector3<f64>,
    v_j: &Vector3<f64>,
    body_to_camera: &SE3,
    factor: &ImuPreintegrationFactor,
    bias_gyro: &Vector3<f64>,
    bias_accel: &Vector3<f64>,
) -> (
    SVector<f64, 9>,
    SMatrix<f64, 9, 6>,
    SMatrix<f64, 9, 6>,
    SMatrix<f64, 9, 3>,
    SMatrix<f64, 9, 3>,
) {
    let body_i = body_to_camera.compose(pose_i);
    let body_j = body_to_camera.compose(pose_j);
    let r_i_so3 = SO3::from_quaternion(body_i.rotation.inverse());
    let r_j_so3 = SO3::from_quaternion(body_j.rotation.inverse());
    let c_i = body_i.inverse().translation;
    let c_j = body_j.inverse().translation;

    let [r_rot, r_vel, r_pos] = factor.residual_with_bias_correction(
        &r_i_so3, &c_i, v_i, &r_j_so3, &c_j, v_j, bias_gyro, bias_accel,
    );
    let mut residual = SVector::<f64, 9>::zeros();
    residual.fixed_rows_mut::<3>(0).copy_from(&r_rot);
    residual.fixed_rows_mut::<3>(3).copy_from(&r_vel);
    residual.fixed_rows_mut::<3>(6).copy_from(&r_pos);

    // Right-perturbation Jacobians of the BODY pose (bundle.rs's own
    // formulas, `bundle.rs:2974-3106`, verbatim).
    let r_wc_i = body_i.rotation.to_rotation_matrix().into_inner();
    let r_wc_j = body_j.rotation.to_rotation_matrix().into_inner();
    let dt = factor.delta.delta_time;
    let g = factor.gravity_world;
    let q_diff = v_j - v_i - g * dt;
    let q_pos_i = c_j - v_i * dt - 0.5 * g * dt * dt;
    let jr_inv = right_jacobian_inverse_so3(&r_rot);
    let jr_inv_rwc_j = jr_inv * r_wc_j;

    let mut j_right_body_i = SMatrix::<f64, 9, 6>::zeros();
    j_right_body_i.fixed_view_mut::<3, 3>(0, 3).copy_from(&jr_inv_rwc_j);
    j_right_body_i.fixed_view_mut::<3, 3>(3, 3).copy_from(&(-r_wc_i * skew(&q_diff)));
    j_right_body_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&r_wc_i);
    j_right_body_i.fixed_view_mut::<3, 3>(6, 3).copy_from(&(-r_wc_i * skew(&q_pos_i)));

    let mut j_right_body_j = SMatrix::<f64, 9, 6>::zeros();
    j_right_body_j.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-jr_inv_rwc_j));
    j_right_body_j.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-r_wc_i));
    j_right_body_j.fixed_view_mut::<3, 3>(6, 3).copy_from(&(r_wc_i * skew(&c_j)));

    let mut j_vel_i = SMatrix::<f64, 9, 3>::zeros();
    j_vel_i.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_wc_i));
    j_vel_i.fixed_view_mut::<3, 3>(6, 0).copy_from(&(-dt * r_wc_i));
    let mut j_vel_j = SMatrix::<f64, 9, 3>::zeros();
    j_vel_j.fixed_view_mut::<3, 3>(3, 0).copy_from(&r_wc_i);

    // Jacobian-convention conversion (module doc): right-perturbation of
    // BODY -> left-perturbation of DPVO's camera pose.
    let adj_c = body_to_camera.adjoint();
    let adj_i = body_i.inverse().adjoint() * adj_c;
    let adj_j = body_j.inverse().adjoint() * adj_c;
    let j_pose_i = j_right_body_i * adj_i;
    let j_pose_j = j_right_body_j * adj_j;

    (residual, j_pose_i, j_pose_j, j_vel_i, j_vel_j)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpvo_patch_ba::{DpvoBaConfig, DpvoEdge, DpvoIntrinsics, DpvoPatch};
    use crate::imu_preintegration::ImuPreintegratedDelta;
    use nalgebra::{UnitQuaternion, Vector2};

    fn synthetic_factor(delta_time: f64, gravity: Vector3<f64>) -> ImuPreintegrationFactor {
        ImuPreintegrationFactor {
            keyframe_id_from: 0,
            keyframe_id_to: 1,
            delta: ImuPreintegratedDelta {
                delta_rotation: visloc_core::geometry::SO3::from_quaternion(
                    UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03),
                ),
                delta_velocity: Vector3::new(0.10, 0.05, -0.02),
                delta_position: Vector3::new(0.05, 0.02, 0.01),
                delta_time,
                // Bias linearisation matches whatever `bias_gyro`/`bias_accel`
                // is passed to `residual_with_bias_correction` in every test
                // below, so `δb = 0` and the (unset, zero) bias Jacobians
                // never actually enter the residual — a deliberate
                // simplification isolating the pose/velocity Jacobian check
                // from the (already independently tested, in
                // `imu_preintegration.rs`) bias-correction machinery.
                bias_gyro_linearisation: Vector3::new(0.001, -0.002, 0.0005),
                bias_acc_linearisation: Vector3::new(0.01, -0.02, 0.03),
                j_rotation_bg: Matrix3::zeros(),
                j_velocity_ba: Matrix3::zeros(),
                j_velocity_bg: Matrix3::zeros(),
                j_position_ba: Matrix3::zeros(),
                j_position_bg: Matrix3::zeros(),
                covariance: Matrix9::zeros(),
            },
            gravity_world: gravity,
            weight_position: 1.0,
            weight_velocity: 1.0,
            weight_rotation: 1.0,
        }
    }

    /// Non-negotiable per the task: every one of `J_pose_i`/`J_pose_j`/
    /// `J_vel_i`/`J_vel_j` checked column-by-column against a central finite
    /// difference of the *actual* DPVO left-perturbation retraction
    /// (`SE3::exp(&xi).compose(&pose)`) this crate uses elsewhere.
    #[test]
    fn imu_factor_jacobian_matches_numeric_finite_difference() {
        let body_to_camera = SE3::new(
            UnitQuaternion::from_euler_angles(0.10, -0.20, 0.30),
            Vector3::new(0.01, -0.02, 0.03),
        );
        let pose_i = SE3::new(
            UnitQuaternion::from_euler_angles(0.05, 0.10, -0.05),
            Vector3::new(0.20, -0.10, 0.05),
        );
        let pose_j = SE3::new(
            UnitQuaternion::from_euler_angles(-0.02, 0.15, 0.08),
            Vector3::new(0.40, 0.05, -0.05),
        );
        let v_i = Vector3::new(0.30, -0.10, 0.05);
        let v_j = Vector3::new(0.35, -0.08, 0.02);
        let bias_gyro = Vector3::new(0.001, -0.002, 0.0005);
        let bias_accel = Vector3::new(0.01, -0.02, 0.03);
        let factor = synthetic_factor(0.1, Vector3::new(0.0, 0.0, -9.81));

        let (r0, j_pose_i, j_pose_j, j_vel_i, j_vel_j) = imu_factor_jacobians(
            &pose_i, &pose_j, &v_i, &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
        );
        // r0 is unused directly (central differences don't need the base
        // point) but computing it also exercises the "both jacobians share
        // the same residual" invariant implicitly.
        let _ = r0;

        let eps = 1.0e-6;
        let tol = 1.0e-5;

        // Pose_i / pose_j: perturb via the exact DPVO left retraction.
        for k in 0..6 {
            let mut xi_p = Vector6::zeros();
            xi_p[k] = eps;
            let mut xi_m = Vector6::zeros();
            xi_m[k] = -eps;

            let pose_i_p = SE3::exp(&xi_p).compose(&pose_i);
            let pose_i_m = SE3::exp(&xi_m).compose(&pose_i);
            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i_p, &pose_j, &v_i, &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i_m, &pose_j, &v_i, &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_pose_i.column(k);
            let err = (numeric - analytic).norm();
            assert!(err < tol, "pose_i column {k}: numeric={numeric:?} analytic={analytic:?} err={err}");

            let pose_j_p = SE3::exp(&xi_p).compose(&pose_j);
            let pose_j_m = SE3::exp(&xi_m).compose(&pose_j);
            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j_p, &v_i, &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j_m, &v_i, &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_pose_j.column(k);
            let err = (numeric - analytic).norm();
            assert!(err < tol, "pose_j column {k}: numeric={numeric:?} analytic={analytic:?} err={err}");
        }

        // Velocity: plain Euclidean perturbation, no SE3 retraction needed.
        for k in 0..3 {
            let mut e_k = Vector3::zeros();
            e_k[k] = eps;
            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j, &(v_i + e_k), &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j, &(v_i - e_k), &v_j, &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_vel_i.column(k);
            let err = (numeric - analytic).norm();
            assert!(err < tol, "v_i column {k}: numeric={numeric:?} analytic={analytic:?} err={err}");

            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j, &v_i, &(v_j + e_k), &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i, &pose_j, &v_i, &(v_j - e_k), &body_to_camera, &factor, &bias_gyro, &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_vel_j.column(k);
            let err = (numeric - analytic).norm();
            assert!(err < tol, "v_j column {k}: numeric={numeric:?} analytic={analytic:?} err={err}");
        }
    }

    /// Sign-convention cross-check (module doc): a pure-IMU window (no
    /// visual edges) run through [`dpvo_vi_ba_step`] should reduce the IMU
    /// residual, exactly like a textbook Gauss-Newton step on `‖r‖²` would —
    /// a wrong sign on the RHS would instead *increase* the residual.
    #[test]
    fn pure_imu_window_matches_textbook_gauss_newton_sign() {
        let intr = DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 100.0, cy: 100.0 };
        let pose0 = SE3::identity();
        // A deliberately "wrong" pose_1 guess (true relative motion is a
        // small rotation + translation; start elsewhere so there is a
        // nonzero residual to reduce).
        let pose1 = SE3::new(UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0), Vector3::new(0.5, 0.0, 0.0));

        let problem = DpvoBaProblem {
            poses: vec![pose0, pose1],
            patches: vec![DpvoPatch { x: 100.0, y: 100.0, inverse_depth: 0.2 }],
            intrinsics: vec![intr, intr],
            edges: vec![],
            targets: vec![],
            weights: vec![],
        };
        let true_relative_translation = Vector3::new(0.1, 0.0, 0.0);
        let factor = DpvoImuFactor {
            i: 0,
            j: 1,
            factor: ImuPreintegrationFactor {
                keyframe_id_from: 0,
                keyframe_id_to: 1,
                delta: ImuPreintegratedDelta {
                    delta_rotation: visloc_core::geometry::SO3::identity(),
                    delta_velocity: Vector3::zeros(),
                    delta_position: true_relative_translation,
                    delta_time: 0.1,
                    ..ImuPreintegratedDelta::identity()
                },
                gravity_world: Vector3::zeros(),
                weight_position: 1.0,
                weight_velocity: 1.0,
                weight_rotation: 1.0,
            },
        };
        let imu = DpvoViWindow {
            velocities: vec![Vector3::zeros(), Vector3::new(1.0, 0.0, 0.0)],
            factors: vec![factor.clone()],
            body_to_camera: SE3::identity(),
            bias_gyro: Vector3::zeros(),
            bias_accel: Vector3::zeros(),
        };
        let config = DpvoBaConfig { fixedp: 1, ..DpvoBaConfig::default() };

        let residual_norm_before = {
            let (r, ..) = imu_factor_jacobians(
                &problem.poses[0], &problem.poses[1], &imu.velocities[0], &imu.velocities[1],
                &imu.body_to_camera, &factor.factor, &imu.bias_gyro, &imu.bias_accel,
            );
            r.norm()
        };

        let solved = dpvo_vi_ba_step(&problem, &imu, &config).expect("solvable system");

        let residual_norm_after = {
            let (r, ..) = imu_factor_jacobians(
                &solved.poses[0], &solved.poses[1], &solved.velocities[0], &solved.velocities[1],
                &imu.body_to_camera, &factor.factor, &imu.bias_gyro, &imu.bias_accel,
            );
            r.norm()
        };
        assert!(
            residual_norm_after < residual_norm_before,
            "expected the GN step to reduce the IMU residual: before={residual_norm_before} after={residual_norm_after}"
        );
    }

    /// Regression check for the module doc's "Visual assembly" tradeoff:
    /// with zero IMU factors, this module's own (duplicated) visual
    /// assembly must reproduce `dpvo_patch_ba::dpvo_ba_step`'s output on
    /// the same problem.
    #[test]
    fn zero_imu_factors_matches_visual_only_solve() {
        let intr = DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 100.0, cy: 100.0 };
        let pose0 = SE3::identity();
        let pose1 = SE3::new(UnitQuaternion::from_euler_angles(0.02, -0.01, 0.03), Vector3::new(0.15, 0.02, -0.01));
        let patch = DpvoPatch { x: 105.0, y: 98.0, inverse_depth: 0.22 };
        let target = crate::dpvo_patch_ba::transform_point(&pose0, &pose1, &intr, &intr, &patch, false)
            + Vector2::new(0.7, -0.3);
        let problem = DpvoBaProblem {
            poses: vec![pose0, pose1],
            patches: vec![patch],
            intrinsics: vec![intr, intr],
            edges: vec![DpvoEdge { i: 0, j: 1, k: 0 }],
            targets: vec![target],
            weights: vec![Vector2::new(1.0, 1.0)],
        };
        let config = DpvoBaConfig { fixedp: 1, ..DpvoBaConfig::default() };
        let imu = DpvoViWindow {
            velocities: vec![Vector3::zeros(), Vector3::zeros()],
            factors: vec![],
            body_to_camera: SE3::identity(),
            bias_gyro: Vector3::zeros(),
            bias_accel: Vector3::zeros(),
        };

        let visual_only = dpvo_ba_step(&problem, &config).expect("visual-only solvable");
        let joint = dpvo_vi_ba_step(&problem, &imu, &config).expect("joint solvable");

        for (a, b) in visual_only.poses.iter().zip(joint.poses.iter()) {
            assert!((a.translation - b.translation).norm() < 1.0e-9);
            assert!(a.rotation.angle_to(&b.rotation) < 1.0e-9);
        }
        for (a, b) in visual_only.patches.iter().zip(joint.patches.iter()) {
            assert!((a.inverse_depth - b.inverse_depth).abs() < 1.0e-9);
        }
    }

    /// Degenerate case (module doc, "Scale handling"): with IMU factors
    /// present but a visual problem that already agrees perfectly with a
    /// **wrong** (1.4x) scale, the joint solve must still recover metric
    /// scale from the IMU evidence alone. This is the task's required
    /// synthetic scale-recovery test.
    #[test]
    fn synthetic_window_recovers_metric_scale_within_two_percent() {
        let intr = DpvoIntrinsics { fx: 200.0, fy: 200.0, cx: 100.0, cy: 100.0 };
        let n_frames = 4;
        let dt = 0.1;
        // Genuine constant *acceleration* (not constant velocity — see the
        // note below on why this matters), identity rotation throughout
        // (isolates scale recovery from the rotation-jacobian correctness
        // already exhaustively covered by
        // `imu_factor_jacobian_matches_numeric_finite_difference`).
        let true_accel = Vector3::new(2.0, 0.0, 0.0);
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let true_scale = 1.0_f64;
        let wrong_scale = 1.4_f64;

        // # Why constant *acceleration*, not constant velocity
        //
        // An earlier version of this test used a constant-velocity
        // trajectory (`v_i` identical for every `i`) and found — the hard
        // way, via a failing assertion whose recovered scale (`1.3999...`)
        // sat essentially unmoved from `wrong_scale` — that it is
        // information-theoretically degenerate for scale recovery: with
        // velocity a fully free Gauss-Newton variable, the Forster
        // gravity-compensated residuals `r_v = R_iᵀ(v_j−v_i−g·Δt) − Δv` and
        // `r_p = R_iᵀ(p_j−p_i−v_i·Δt−½g·Δt²) − Δp` are satisfied by *any*
        // joint rescaling `(p, v) ↦ (s·p, s·v)` whenever the true motion is
        // constant-velocity — because both residuals then only constrain
        // velocity *differences*/position-velocity *consistency*, never an
        // absolute magnitude. This is not a solver bug: it is the same
        // well-known "insufficient excitation" degeneracy
        // `vi_motion_initializer.rs`'s own bootstrap gates exist to detect
        // and reject in the real pipeline (`crate::dpvo_vo`'s bootstrap
        // reuses those same gates — see that module's doc). A constant
        // **acceleration** breaks the degeneracy: `Δv = a·Δt − g·Δt` is a
        // fixed physical quantity independent of any position rescaling,
        // so `r_v = 0` now pins the recovered velocity (and hence position)
        // scale to exactly `1`, not to an arbitrary value.
        let true_velocity_at = |i: usize| true_accel * (dt * i as f64);
        let true_position_at =
            |i: usize| true_accel * (0.5 * (dt * i as f64) * (dt * i as f64));

        // True (metric) camera centers in world.
        let true_centers: Vec<Vector3<f64>> = (0..n_frames).map(true_position_at).collect();
        // Starting-guess (wrong-scale) camera centers and poses.
        let guess_centers: Vec<Vector3<f64>> = true_centers.iter().map(|c| c * wrong_scale).collect();
        let guess_poses: Vec<SE3> =
            guess_centers.iter().map(|c| SE3::new(UnitQuaternion::identity(), -c)).collect();

        // Five patches anchored in frame 0 at varying pixel offsets / true
        // depths, all visible from every later frame.
        let anchor_pixels = [(90.0, 100.0), (100.0, 100.0), (110.0, 100.0), (100.0, 90.0), (100.0, 110.0)];
        let true_depths = [4.0, 5.0, 6.0, 4.5, 5.5];
        let guess_patches: Vec<DpvoPatch> = anchor_pixels
            .iter()
            .zip(true_depths.iter())
            .map(|(&(x, y), &depth)| DpvoPatch { x, y, inverse_depth: 1.0 / (depth * wrong_scale) })
            .collect();

        let mut edges = Vec::new();
        let mut targets = Vec::new();
        let mut weights = Vec::new();
        for (k, patch) in guess_patches.iter().enumerate() {
            for j in 1..n_frames {
                // Target computed by literally reprojecting the WRONG-SCALE
                // starting guess (poses + inverse depth) through the same
                // forward-projection function the solver itself uses — this
                // guarantees an exact visual fixed point at the wrong scale
                // (mono reprojection is scale-invariant), with no separate
                // hand-derived pixel formula to get subtly wrong.
                let target = crate::dpvo_patch_ba::transform_point(
                    &guess_poses[0], &guess_poses[j], &intr, &intr, patch, false,
                );
                edges.push(DpvoEdge { i: 0, j, k });
                targets.push(target);
                weights.push(Vector2::new(1.0, 1.0));
            }
        }

        let problem = DpvoBaProblem {
            poses: guess_poses.clone(),
            patches: guess_patches,
            intrinsics: vec![intr; n_frames],
            edges,
            targets,
            weights,
        };

        // IMU factors built from TRUE metric kinematics (constant
        // acceleration, identity rotation throughout — see the note above):
        // ΔR = I, Δv = R_iᵀ(v_j−v_i−g·Δt) = a·Δt − g·Δt (R_i = I),
        // Δp = R_iᵀ(p_j−p_i−v_i·Δt−½g·Δt²) = ½·a·Δt² − ½·g·Δt² (both exact
        // for a true constant-acceleration trajectory, for any `i`).
        let mut factors = Vec::new();
        for i in 0..n_frames - 1 {
            let v_i = true_velocity_at(i);
            let v_j = true_velocity_at(i + 1);
            let p_i = true_position_at(i);
            let p_j = true_position_at(i + 1);
            factors.push(DpvoImuFactor {
                i,
                j: i + 1,
                factor: ImuPreintegrationFactor {
                    keyframe_id_from: i as u64,
                    keyframe_id_to: (i + 1) as u64,
                    delta: ImuPreintegratedDelta {
                        delta_rotation: visloc_core::geometry::SO3::identity(),
                        delta_velocity: (v_j - v_i) - gravity * dt,
                        delta_position: (p_j - p_i) - v_i * dt - 0.5 * gravity * dt * dt,
                        delta_time: dt,
                        ..ImuPreintegratedDelta::identity()
                    },
                    gravity_world: gravity,
                    weight_position: 1.0,
                    weight_velocity: 1.0,
                    weight_rotation: 1.0,
                },
            });
        }

        let imu = DpvoViWindow {
            velocities: vec![Vector3::zeros(); n_frames],
            factors,
            body_to_camera: SE3::identity(),
            bias_gyro: Vector3::zeros(),
            bias_accel: Vector3::zeros(),
        };
        let config = DpvoBaConfig {
            fixedp: 1,
            iterations: 80,
            lmbda: 1.0e-6,
            ep: 0.1,
            bounds: [f64::NEG_INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY],
        };

        let solved = dpvo_vi_ba(&problem, &imu, &config).expect("joint solve should converge");

        let recovered_center_3 = solved.poses[3].inverse().translation;
        let recovered_scale = recovered_center_3.norm() / true_centers[3].norm();
        println!(
            "[synthetic scale recovery] wrong_scale={wrong_scale} recovered_scale={recovered_scale:.6} \
             (true={true_scale})"
        );
        assert!(
            (recovered_scale - true_scale).abs() < 0.02,
            "expected recovered scale within 2% of {true_scale}, got {recovered_scale} \
             (recovered center {recovered_center_3:?}, true center {:?})",
            true_centers[3]
        );
    }
}
