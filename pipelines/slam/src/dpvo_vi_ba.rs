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
//!
//! # Milestone M5b: a monocular-aware bootstrap (`docs/dpvo_droid_port_plan.md`,
//! "M5 results" / "M5b results")
//!
//! M5's own honest-negative finding: `vi_motion_initializer::estimate_gyro_bias`/
//! `estimate_gravity_and_velocities` (the "Gravity" section above) were
//! designed for, and everywhere else in this codebase are run against, an
//! **already metric** set of visual poses. Run against DPVO's own
//! reconstruction — which is in an arbitrary monocular scale (`0.5×`–`1.3×`
//! per M4-perf's own measurement) whenever the bootstrap chain first has
//! enough evidence to fire — `estimate_gravity_and_velocities`'s linear
//! system silently absorbs that unmodeled scale error into its recovered
//! gravity/velocity estimate, and the M5 "staged, fixed-forever" bias/
//! gravity design then poisons the rest of the run on a single bad
//! bootstrap. [`estimate_mono_vi_alignment`] below is the fix M5's own
//! blockers list called for: option (a), an explicit scale unknown in the
//! alignment itself, rather than reusing an estimator that assumes it away.
//!
//! ## Formulation: VINS-Mono monocular VI alignment, adapted
//!
//! Unknowns `x = [v_1 .. v_N (world), g (world), s]` — the same per-window
//! world-frame velocities and gravity vector `estimate_gravity_and_velocities`
//! solves for, plus one **scalar monocular scale** `s` such that
//! `metric_position = s · visual_position` (DPVO's own non-metric camera/body
//! center, read directly off its own poses — no `VisualMap` needed, unlike
//! the metric estimators, since this function takes DPVO's poses directly).
//! Per consecutive-pose factor `i → j` (`R_bw_i` = keyframe `i`'s
//! world-to-body rotation, `Δt` = `factor.delta.delta_time`, `Δv_ij`/`Δp_ij`
//! = the bias-corrected preintegrated deltas at the caller's `bias_gyro`/
//! `bias_accel`):
//!
//! * Velocity (scale-free, identical to the metric estimator):
//!   `Δv_ij = R_bw_i · (v_j − v_i − g·Δt)`
//! * Position (the one line that changes — visual positions scaled by `s`
//!   before differencing):
//!   `Δp_ij = R_bw_i · (s·(p_j − p_i) − v_i·Δt − ½·g·Δt²)`
//!
//! Both are linear in `(v_i, v_j, g, s)` (rotations come from the fixed
//! visual poses and are themselves scale-invariant — a rigid rotation
//! doesn't care what units its own translation is in — so no scale unknown
//! is needed there): moving every unknown to the left of `=` gives
//!
//! `v_i·Δt + ½g·Δt² − s·(p_j−p_i) = −R_bw_i·Δp_ij` (3 rows)
//! `−v_i + v_j − g·Δt = R_bw_i·Δv_ij` (3 rows)
//!
//! solved by the same dense SVD least-squares + VINS-Mono tangent-space
//! magnitude-constrained refinement `estimate_gravity_and_velocities` uses
//! (Qin & Shen 2018, eq. 7-9) — [`mono_vi_tangent_basis`] is a verbatim
//! duplicate of that module's private `tangent_basis` (same "duplicate a
//! well-known formula rather than expose cross-module-privately" reasoning
//! as [`skew`]/[`right_jacobian_inverse_so3`] below), extended with `s` as
//! an additional free scalar the refinement re-solves for every iteration
//! (unlike gravity, `s` needs no norm constraint — nothing pins its
//! magnitude except the IMU evidence itself, exactly like every free
//! velocity).
//!
//! ## Observability gates (see [`DpvoMonoViAlignmentGates`])
//!
//! Three checks, in order, any of which returns `Err(`[`DpvoMonoViAlignmentRejection`]`)`
//! identifying exactly which one tripped:
//!
//! 1. **Degrees of freedom**: `6 · factors.len() ≥ 3 · N + 4` (the unknown
//!    count) is a hard mathematical necessity, not a tunable gate — an
//!    underdetermined system's SVD solve returns a minimum-norm solution
//!    that satisfies the equations but is not the physically meaningful
//!    answer (unconstrained directions get silently zeroed rather than
//!    flagged), so this is checked unconditionally before the solve, not
//!    exposed as a config knob to loosen.
//! 2. **Excitation / conditioning**: the unconstrained solve's condition
//!    number (`σ_max/σ_min` of the assembled system) gated by
//!    [`DpvoMonoViAlignmentGates::max_condition_number`]. A window with
//!    real, multi-directional acceleration is well-conditioned; a
//!    constant-velocity (or near-stationary) window is close to
//!    rank-deficient in exactly the `(v, s)` joint direction — the same
//!    degeneracy [`tests::synthetic_window_recovers_metric_scale_within_two_percent`]'s
//!    own doc comment already documents for the *coupled* solve, now
//!    checked directly on the *bootstrap's own* linear system before any
//!    bad scale ever reaches `dpvo_vi_ba`. Default (`1e8`, see
//!    `crate::dpvo_vo::DpvoImuConfig::max_mono_alignment_condition_number`)
//!    calibrated empirically against this file's own two tests' MEASURED
//!    numbers:
//!    [`tests::estimate_mono_vi_alignment_recovers_scale_from_constant_acceleration_window`]'s
//!    genuinely-3D-excited window measures `≈361` (comfortably below the
//!    default), while
//!    [`tests::estimate_mono_vi_alignment_rejects_constant_velocity_window`]'s
//!    constant-velocity window measures a literal `min_sv = 0.0` (condition
//!    number `∞`, rejected regardless of how loose the default is) — a wide
//!    margin between "well-conditioned" and "structurally degenerate", not
//!    a knife-edge tuning.
//! 3. **Gravity-norm deviation** and **scale plausibility**
//!    (`raw_gravity_norm` within [`DpvoMonoViAlignmentGates::gravity_norm_deviation_ratio`]
//!    of the expected magnitude; recovered `s` within `[min_scale,
//!    max_scale]`) — the same style of physically-motivated sanity bound
//!    `crate::dpvo_vo::DpvoImuConfig::gravity_norm_deviation_ratio` already
//!    uses, plus the task's own `[0.05, 20]` bound on a monocular scale
//!    factor (a real rig's true scale error is never that extreme; a
//!    solve landing outside that range is reporting numerical garbage, not
//!    a plausible-if-large scale correction).
//!
//! ## Sequencing: gyro bias first, mono alignment second
//!
//! `crate::dpvo_vo::DpvoOdometry::try_imu_bootstrap` runs
//! `vi_motion_initializer::estimate_gyro_bias` **before** this function,
//! unchanged — gyro-bias recovery from ROTATION alignment alone is
//! genuinely scale-invariant (a rotation doesn't know or care what units
//! the translations around it are in), so reusing that metric-agnostic
//! estimator as-is is correct, *not* an oversight this milestone needs to
//! fix. What M5 got wrong there was not the estimator's math but its
//! **gate**: accepting whatever bias came back unconditionally. See
//! `crate::dpvo_vo::DpvoImuConfig`'s new `max_gyro_bias_magnitude_rad_s`/
//! `gyro_bias_max_rms_after`/`gyro_bias_max_rms_fraction` fields and
//! `try_imu_bootstrap`'s own doc for the added rms-based rejection this
//! milestone layers on top — deliberately NOT fixing a bias until it
//! passes, retrying with a growing window on every subsequent frame
//! instead (mirroring this function's own "gated `None`, retry later"
//! contract).
//!
//! ## Applying the recovered scale (`crate::dpvo_vo::DpvoOdometry::try_imu_bootstrap`)
//!
//! Once both gates pass, `s` must be applied to every LIVE frame/patch
//! before IMU coupling turns on, or the newly-seeded metric velocities
//! would be solving against still-non-metric positions. A uniform scaling
//! of world coordinates (`X_world_new = s · X_world_old`, keeping every
//! camera's orientation and every pixel's reprojection identical — a pure
//! similarity transform, not a re-optimization) requires: pose translation
//! `t_new = s · t_old` (since `t = −R·C` for camera center `C`, and
//! `C_new = s·C_old` under a uniform world rescale, so `t_new = −R·(s·C_old)
//! = s·t_old` — rotation is untouched), and patch `inverse_depth_new =
//! inverse_depth_old / s` (depth is a translation-like length, so
//! `depth_new = s · depth_old`, hence the reciprocal). Both are the
//! textbook VINS-Mono/ORB-SLAM3 "apply the recovered scale" step, ported
//! here rather than reused from anywhere (no prior milestone had a scale
//! unknown to apply).
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
        let current_problem = DpvoBaProblem {
            poses,
            patches,
            ..problem.clone()
        };
        let current_imu = DpvoViWindow {
            velocities,
            ..imu.clone()
        };
        let solved = dpvo_vi_ba_step(&current_problem, &current_imu, config)?;
        poses = solved.poses;
        patches = solved.patches;
        velocities = solved.velocities;
    }
    Ok(DpvoViBaSolution {
        poses,
        patches,
        velocities,
    })
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
    let local_patch_index = |k: usize| {
        used_patches
            .binary_search(&k)
            .expect("k is drawn from used_patches")
    };

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

        let i_local = if edge.i >= fixedp {
            Some(edge.i - fixedp)
        } else {
            None
        };
        let j_local = if edge.j >= fixedp {
            Some(edge.j - fixedp)
        } else {
            None
        };

        if let Some(i) = i_local {
            let bii = w_ji.transpose() * geom.ji;
            b_pose
                .view_mut((i * 6, i * 6), (6, 6))
                .iter_mut()
                .zip(bii.iter())
                .for_each(|(d, s)| *d += s);
            let vi = w_ji.transpose() * r;
            v_pose
                .rows_mut(i * 6, 6)
                .iter_mut()
                .zip(vi.iter())
                .for_each(|(d, s)| *d += s);
            let eik = w_ji.transpose() * geom.jz;
            e_mat
                .view_mut((i * 6, k_local), (6, 1))
                .iter_mut()
                .zip(eik.iter())
                .for_each(|(d, s)| *d += s);
        }
        if let Some(j) = j_local {
            let bjj = w_jj.transpose() * geom.jj;
            b_pose
                .view_mut((j * 6, j * 6), (6, 6))
                .iter_mut()
                .zip(bjj.iter())
                .for_each(|(d, s)| *d += s);
            let vj = w_jj.transpose() * r;
            v_pose
                .rows_mut(j * 6, 6)
                .iter_mut()
                .zip(vj.iter())
                .for_each(|(d, s)| *d += s);
            let ejk = w_jj.transpose() * geom.jz;
            e_mat
                .view_mut((j * 6, k_local), (6, 1))
                .iter_mut()
                .zip(ejk.iter())
                .for_each(|(d, s)| *d += s);
        }
        if let (Some(i), Some(j)) = (i_local, j_local) {
            let bij = w_ji.transpose() * geom.jj;
            b_pose
                .view_mut((i * 6, j * 6), (6, 6))
                .iter_mut()
                .zip(bij.iter())
                .for_each(|(d, s)| *d += s);
            let bji = w_jj.transpose() * geom.ji;
            b_pose
                .view_mut((j * 6, i * 6), (6, 6))
                .iter_mut()
                .zip(bji.iter())
                .for_each(|(d, s)| *d += s);
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

        let whitener = imu_factor_whitener(factor);
        let r_stack = whitener * residual;
        let j_pose_i = whitener * j_pose_i;
        let j_pose_j = whitener * j_pose_j;
        let j_vel_i = whitener * j_vel_i;
        let j_vel_j = whitener * j_vel_j;

        let pose_off = |idx: usize| -> Option<usize> {
            if idx < fixedp {
                None
            } else {
                Some((idx - fixedp) * 6)
            }
        };
        // Velocity is free for every frame, including a pose-gauge-fixed one
        // — see the module doc's "Per-window state and gauge-fixing" section.
        let vel_off = |idx: usize| -> Option<usize> { Some(6 * n2 + idx * 3) };

        let blocks: [(Option<usize>, DMatrix<f64>); 4] = [
            (
                pose_off(i),
                DMatrix::from_column_slice(9, 6, j_pose_i.as_slice()),
            ),
            (
                pose_off(j),
                DMatrix::from_column_slice(9, 6, j_pose_j.as_slice()),
            ),
            (
                vel_off(i),
                DMatrix::from_column_slice(9, 3, j_vel_i.as_slice()),
            ),
            (
                vel_off(j),
                DMatrix::from_column_slice(9, 3, j_vel_j.as_slice()),
            ),
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

    let dx_aug = b_aug
        .lu()
        .solve(&v_aug)
        .ok_or(DpvoBaError::SingularSystem)?;
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

    Ok(DpvoViBaSolution {
        poses: new_poses,
        patches: new_patches,
        velocities: new_velocities,
    })
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
// `pub(crate)`, not private: Milestone M5b's `crate::dpvo_vo` rollback
// monitor needs this factor's residual too (via [`imu_factor_nis`] below) —
// a pure visibility widening, same reasoning `dpvo_patch_ba.rs`'s own
// `pub(crate)` items already used for M5's cross-module reuse.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn imu_factor_jacobians(
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
    j_right_body_i
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&jr_inv_rwc_j);
    j_right_body_i
        .fixed_view_mut::<3, 3>(3, 3)
        .copy_from(&(-r_wc_i * skew(&q_diff)));
    j_right_body_i
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&r_wc_i);
    j_right_body_i
        .fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(-r_wc_i * skew(&q_pos_i)));

    let mut j_right_body_j = SMatrix::<f64, 9, 6>::zeros();
    j_right_body_j
        .fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-jr_inv_rwc_j));
    j_right_body_j
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-r_wc_i));
    j_right_body_j
        .fixed_view_mut::<3, 3>(6, 3)
        .copy_from(&(r_wc_i * skew(&c_j)));

    let mut j_vel_i = SMatrix::<f64, 9, 3>::zeros();
    j_vel_i.fixed_view_mut::<3, 3>(3, 0).copy_from(&(-r_wc_i));
    j_vel_i
        .fixed_view_mut::<3, 3>(6, 0)
        .copy_from(&(-dt * r_wc_i));
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

/// The whitened square-root-information matrix for one IMU factor —
/// `covariance_sqrt_information()` when the delta carries a real propagated
/// covariance, else a diagonal fallback from the factor's own scalar
/// `weight_*` fields. Extracted out of [`dpvo_vi_ba_step`]'s inline block
/// (same computation, same call site — a pure refactor, not a behavior
/// change) so [`imu_factor_nis`] (Milestone M5b's rollback monitor) can
/// reuse the exact same fallback formula rather than re-deriving it a
/// second time.
///
/// `pub(crate)`, not private: Milestone M7's `crate::dpvo_scale_coupling`
/// reuses this exact whitening (same reasoning as [`imu_factor_jacobians`]'s
/// own `pub(crate)` widening for M5b's rollback monitor) for its
/// finite-difference gentle-scale-correction Jacobian — see that module's
/// doc.
pub(crate) fn imu_factor_whitener(factor: &ImuPreintegrationFactor) -> Matrix9 {
    factor.covariance_sqrt_information().unwrap_or_else(|| {
        let mut diagonal = SVector::<f64, 9>::zeros();
        diagonal
            .fixed_rows_mut::<3>(0)
            .fill(factor.weight_rotation.max(0.0).sqrt());
        diagonal
            .fixed_rows_mut::<3>(3)
            .fill(factor.weight_velocity.max(0.0).sqrt());
        diagonal
            .fixed_rows_mut::<3>(6)
            .fill(factor.weight_position.max(0.0).sqrt());
        Matrix9::from_diagonal(&diagonal)
    })
}

/// Milestone M5b (see the module doc's own section): the whitened
/// Normalized-Innovation-Squared (`‖whitener · residual‖²`, 9 scalar
/// components) of one IMU factor at a given `(pose_i, pose_j, v_i, v_j)`
/// solution. `crate::dpvo_vo`'s post-bootstrap rollback monitor calls this
/// once per in-window IMU factor after every `dpvo_vi_ba` solve; a solve
/// that keeps fighting its own IMU evidence because a bootstrap landed on a
/// bad scale/gravity/bias shows up here as a persistently large NIS — the
/// "after the fact" detector M5's own "poisoned forever" failure had no
/// equivalent of (see `crate::dpvo_vo::DpvoOdometry::rollback_imu_bootstrap`'s
/// doc for what happens once this trips).
#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn imu_factor_nis(
    pose_i: &SE3,
    pose_j: &SE3,
    v_i: &Vector3<f64>,
    v_j: &Vector3<f64>,
    body_to_camera: &SE3,
    factor: &ImuPreintegrationFactor,
    bias_gyro: &Vector3<f64>,
    bias_accel: &Vector3<f64>,
) -> f64 {
    let (residual, ..) = imu_factor_jacobians(
        pose_i,
        pose_j,
        v_i,
        v_j,
        body_to_camera,
        factor,
        bias_gyro,
        bias_accel,
    );
    let whitener = imu_factor_whitener(factor);
    (whitener * residual).norm_squared()
}

/// Gates for [`estimate_mono_vi_alignment`] — Milestone M5b. See the module
/// doc's "Observability gates" section for what each one guards against and
/// how its default was chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpvoMonoViAlignmentGates {
    /// Expected local gravity magnitude (m/s²) — same role as
    /// `crate::dpvo_vo::DpvoImuConfig::gravity_magnitude`.
    pub expected_gravity_magnitude: f64,
    /// Same semantics as `crate::dpvo_vo::DpvoImuConfig::gravity_norm_deviation_ratio`,
    /// applied to this function's own unconstrained-solve `raw_gravity_norm`.
    pub gravity_norm_deviation_ratio: f64,
    /// Lower bound on the recovered monocular scale `s` (task-specified
    /// default `0.05`).
    pub min_scale: f64,
    /// Upper bound on the recovered monocular scale `s` (task-specified
    /// default `20.0`).
    pub max_scale: f64,
    /// Upper bound on the unconstrained linear system's condition number
    /// (`σ_max/σ_min`) — the excitation/degeneracy gate. See the module
    /// doc's "Observability gates" section for how the default was
    /// calibrated against this file's own synthetic well-conditioned vs.
    /// degenerate tests.
    pub max_condition_number: f64,
}

/// Why [`estimate_mono_vi_alignment`] rejected a window — Milestone M5b's
/// answer to the task's own "if it still fails, isolate which gate" demand:
/// every rejection carries the SPECIFIC gate that tripped and the value(s)
/// that tripped it, so `crate::dpvo_vo::DpvoOdometry`'s bootstrap-rejection
/// diagnostics can report a real breakdown (not just "rejected N times")
/// against a live EuRoC run. Every variant is still a rejection — this
/// enum changes nothing about acceptance behavior, only what a caller can
/// observe about why.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DpvoMonoViAlignmentRejection {
    /// Fewer than 2 poses, or fewer than 2 usable in-window factors
    /// (missing `delta_time > 0`, or an out-of-range index).
    NotEnoughFactors,
    /// Degrees-of-freedom gate (module doc, "Observability gates" #1):
    /// `6 * usable_factors < 3 * n + 4`.
    Underdetermined {
        usable_factors: usize,
        n_poses: usize,
    },
    /// Excitation/conditioning gate (module doc, "Observability gates" #2).
    IllConditioned {
        condition_number: f64,
        max_condition_number: f64,
    },
    /// The unconstrained SVD solve itself failed, or produced a
    /// non-finite/near-zero gravity vector or non-finite scale — a
    /// numerically degenerate system distinct from a merely
    /// poorly-conditioned one.
    DegenerateSolve,
    /// Gravity-norm gate (module doc, "Observability gates" #3a).
    GravityNormDeviation {
        raw_gravity_norm: f64,
        deviation_ratio: f64,
        max_deviation_ratio: f64,
    },
    /// Scale-plausibility gate (module doc, "Observability gates" #3b).
    ScaleOutOfRange {
        /// Magnitude-constrained refinement result that tripped the gate.
        scale: f64,
        /// Unconstrained linear solve's scale before tangent refinement.
        raw_scale: f64,
        min_scale: f64,
        max_scale: f64,
        /// Observability context from the same rejected solve. These fields
        /// are diagnostic only; adding them does not alter gate ordering or
        /// acceptance.
        raw_gravity_norm: f64,
        condition_number: f64,
        min_singular_value: f64,
        window_frames: usize,
    },
}

/// Outcome of [`estimate_mono_vi_alignment`]: everything
/// `crate::dpvo_vo::DpvoOdometry::try_imu_bootstrap` needs to promote a
/// non-metric DPVO window to metric in one step — see the module doc's
/// "Milestone M5b" section.
#[derive(Debug, Clone, PartialEq)]
pub struct DpvoMonoViAlignment {
    /// Recovered monocular scale: `metric_position = scale · visual_position`.
    pub scale: f64,
    /// Recovered world-frame gravity, magnitude-constrained to the gate's
    /// `expected_gravity_magnitude` (or the plain scale-to-norm fallback —
    /// see the module doc's refinement description).
    pub gravity_world: Vector3<f64>,
    /// Norm of the UNCONSTRAINED solve's gravity estimate, before magnitude
    /// refinement — the observability signal
    /// [`DpvoMonoViAlignmentGates::gravity_norm_deviation_ratio`] gates on
    /// (mirrors `vi_motion_initializer::GravityVelocityAlignment::raw_gravity_norm`
    /// exactly).
    pub raw_gravity_norm: f64,
    /// Per-window-pose-index world-frame (metric) velocity, same
    /// length/order as the `poses` slice passed in.
    pub velocities: Vec<Vector3<f64>>,
    /// RMS of the stacked (scale-substituted) position + velocity
    /// preintegration residual at the final `(gravity_world, velocities,
    /// scale)` estimate — a relative diagnostic, mirroring
    /// `GravityVelocityAlignment::mean_residual_after`.
    pub mean_residual_after: f64,
    /// Condition number of the unconstrained solve's system matrix — the
    /// excitation gate's own diagnostic value, echoed here so a caller can
    /// log *how* well-conditioned an accepted alignment actually was, not
    /// just that it cleared the gate.
    pub condition_number: f64,
    /// Number of poses (`= poses.len()`) this alignment was solved over.
    pub window_frames: usize,
    /// Milestone M7 (`docs/dpvo_droid_port_plan.md`): the unconstrained
    /// solve's smallest singular value (`svd.singular_values.min()`),
    /// echoed alongside `condition_number` so a caller can derive an
    /// honest per-window scale-measurement VARIANCE proxy without
    /// re-running the SVD a second time. See
    /// `crate::dpvo_scale_coupling::scale_measurement_from_alignment`'s
    /// module doc for the exact formula
    /// (`variance ≈ mean_residual_after² / min_singular_value²`, the
    /// classical worst-case bound on a linear-least-squares parameter's
    /// own marginal variance, `σ²·[(AᵀA)⁻¹]_{ss} ≤ σ²/σ_min²`).
    pub min_singular_value: f64,
}

/// Recover monocular scale, gravity, and per-pose velocities from a window
/// of DPVO's own (non-metric) poses and IMU pre-integration factors — the
/// monocular-aware replacement for reusing
/// `vi_motion_initializer::estimate_gravity_and_velocities` against
/// still-non-metric visual poses (Milestone M5b; see the module doc's own
/// section for the full derivation and the gates this applies, in order).
///
/// `poses`/`factors` use the SAME window-local indexing convention as
/// [`DpvoViWindow`] (`factors[k].i`/`.j` index into `poses`) — the caller
/// (`crate::dpvo_vo::DpvoOdometry::try_imu_bootstrap`) is responsible for
/// translating its own arrival-index-keyed bootstrap history into this
/// local indexing, exactly as `update_step` already does for `DpvoViWindow`
/// itself. `bias_gyro` is expected to already be the caller's OWN
/// accepted/gated estimate (this function does not estimate gyro bias —
/// see the module doc's "Sequencing" section for why that stays a separate,
/// prior step); `bias_accel` is typically `0.0` (M5's own documented
/// narrowing, unchanged by this milestone).
///
/// Returns `Err(`[`DpvoMonoViAlignmentRejection`]`)` identifying the
/// SPECIFIC gate (of the three in the module doc's "Observability gates"
/// section) that rejected this window, or outright degeneracy (fewer than 2
/// poses, fewer than 2 usable in-window factors, or a singular/non-finite
/// unconstrained solve).
pub fn estimate_mono_vi_alignment(
    poses: &[SE3],
    factors: &[DpvoImuFactor],
    body_to_camera: &SE3,
    bias_gyro: Vector3<f64>,
    bias_accel: Vector3<f64>,
    gates: &DpvoMonoViAlignmentGates,
) -> Result<DpvoMonoViAlignment, DpvoMonoViAlignmentRejection> {
    const REFINEMENT_ITERATIONS: usize = 4;

    let n = poses.len();
    if n < 2 {
        return Err(DpvoMonoViAlignmentRejection::NotEnoughFactors);
    }
    if !gates.expected_gravity_magnitude.is_finite() || gates.expected_gravity_magnitude <= 0.0 {
        return Err(DpvoMonoViAlignmentRejection::DegenerateSolve);
    }

    // Body-to-world rotation and non-metric (visual-scale) body/camera
    // center in world, per window-local pose index — the module doc's
    // `R_bw_i`/`p_i`, read directly off DPVO's own poses (no `VisualMap`
    // needed here, unlike the metric estimators this replaces).
    let world_from_body: Vec<SE3> = poses
        .iter()
        .map(|p| body_to_camera.compose(p).inverse())
        .collect();

    struct Row {
        idx_from: usize,
        idx_to: usize,
        delta_time: f64,
        rhs_velocity: Vector3<f64>,
        rhs_position: Vector3<f64>,
        position_vis_diff: Vector3<f64>,
    }
    let mut rows: Vec<Row> = Vec::new();
    for imu_factor in factors {
        let (i, j) = (imu_factor.i, imu_factor.j);
        if i >= n || j >= n {
            continue;
        }
        let delta_time = imu_factor.factor.delta.delta_time;
        // Equivalent to `!(delta_time > 0.0)` (rejects non-positive AND
        // NaN alike) without clippy's `neg_cmp_op_on_partial_ord` lint,
        // which the equivalent negated-comparison form trips.
        if !delta_time.is_finite() || delta_time <= 0.0 {
            continue;
        }
        let r_i = world_from_body[i].rotation;
        let p_i = world_from_body[i].translation;
        let p_j = world_from_body[j].translation;
        let (_, delta_velocity, delta_position) =
            imu_factor.factor.delta.corrected(&bias_gyro, &bias_accel);
        rows.push(Row {
            idx_from: i,
            idx_to: j,
            delta_time,
            rhs_velocity: r_i.transform_vector(&delta_velocity),
            rhs_position: r_i.transform_vector(&delta_position),
            position_vis_diff: p_j - p_i,
        });
    }
    if rows.len() < 2 {
        return Err(DpvoMonoViAlignmentRejection::NotEnoughFactors);
    }

    let num_rows = 6 * rows.len();
    let num_unknowns = 3 * n + 4; // v_1..v_n (3 each), g (3), s (1).
    if num_rows < num_unknowns {
        // Degrees-of-freedom gate (module doc, "Observability gates" #1):
        // a hard mathematical necessity, not a tunable — see that section.
        return Err(DpvoMonoViAlignmentRejection::Underdetermined {
            usable_factors: rows.len(),
            n_poses: n,
        });
    }
    let g_off = 3 * n;
    let s_off = 3 * n + 3;

    let mut a = DMatrix::<f64>::zeros(num_rows, num_unknowns);
    let mut b = DVector::<f64>::zeros(num_rows);
    for (r, row) in rows.iter().enumerate() {
        let base = 6 * r;
        for k in 0..3 {
            // Position row (module doc): v_i·Δt + ½g·Δt² − s·(p_j−p_i) = −R_bw_i·Δp_ij
            a[(base + k, 3 * row.idx_from + k)] = row.delta_time;
            a[(base + k, g_off + k)] = 0.5 * row.delta_time * row.delta_time;
            a[(base + k, s_off)] = -row.position_vis_diff[k];
            b[base + k] = -row.rhs_position[k];

            // Velocity row (module doc, unchanged from the metric estimator):
            // −v_i + v_j − g·Δt = R_bw_i·Δv_ij
            a[(base + 3 + k, 3 * row.idx_from + k)] = -1.0;
            a[(base + 3 + k, 3 * row.idx_to + k)] = 1.0;
            a[(base + 3 + k, g_off + k)] = -row.delta_time;
            b[base + 3 + k] = row.rhs_velocity[k];
        }
    }

    let svd = a.svd(true, true);
    let singular_values = svd.singular_values.clone();
    let max_sv = singular_values.max();
    let min_sv = singular_values.min();
    let condition_number = if min_sv > 0.0 && max_sv.is_finite() {
        max_sv / min_sv
    } else {
        f64::INFINITY
    };
    if !condition_number.is_finite() || condition_number > gates.max_condition_number {
        // Excitation/conditioning gate (module doc, "Observability gates" #2).
        return Err(DpvoMonoViAlignmentRejection::IllConditioned {
            condition_number,
            max_condition_number: gates.max_condition_number,
        });
    }
    let Ok(solution) = svd.solve(&b, 1.0e-9) else {
        return Err(DpvoMonoViAlignmentRejection::DegenerateSolve);
    };

    let raw_gravity = Vector3::new(solution[g_off], solution[g_off + 1], solution[g_off + 2]);
    let raw_gravity_norm = raw_gravity.norm();
    if !raw_gravity_norm.is_finite() || raw_gravity_norm < 1.0e-6 {
        return Err(DpvoMonoViAlignmentRejection::DegenerateSolve);
    }
    let raw_scale = solution[s_off];
    if !raw_scale.is_finite() {
        return Err(DpvoMonoViAlignmentRejection::DegenerateSolve);
    }
    let deviation_ratio = (raw_gravity_norm - gates.expected_gravity_magnitude).abs()
        / gates.expected_gravity_magnitude;
    if !deviation_ratio.is_finite() || deviation_ratio > gates.gravity_norm_deviation_ratio {
        // Gravity-norm gate (module doc, "Observability gates" #3a).
        return Err(DpvoMonoViAlignmentRejection::GravityNormDeviation {
            raw_gravity_norm,
            deviation_ratio,
            max_deviation_ratio: gates.gravity_norm_deviation_ratio,
        });
    }

    // Magnitude-constrained refinement (module doc): the same VINS-Mono
    // tangent-space iteration `estimate_gravity_and_velocities` uses,
    // extended with the extra free scalar `s` (unlike gravity, `s` gets no
    // norm constraint — nothing pins its magnitude but the IMU evidence
    // itself, exactly like every free velocity).
    let mag = gates.expected_gravity_magnitude;
    let mut g_hat = raw_gravity / raw_gravity_norm;
    let mut velocities_final: Vec<Vector3<f64>> = (0..n)
        .map(|k| Vector3::new(solution[3 * k], solution[3 * k + 1], solution[3 * k + 2]))
        .collect();
    let mut scale_final = raw_scale;
    let mut refined = false;
    for _ in 0..REFINEMENT_ITERATIONS {
        let (b1, b2) = mono_vi_tangent_basis(&g_hat);
        let g0 = mag * g_hat;
        let w1_off = 3 * n;
        let w2_off = 3 * n + 1;
        let s2_off = 3 * n + 2;
        let refine_unknowns = 3 * n + 3; // v_1..v_n, w1, w2, s.
        let mut a2 = DMatrix::<f64>::zeros(num_rows, refine_unknowns);
        let mut b2vec = DVector::<f64>::zeros(num_rows);
        for (r, row) in rows.iter().enumerate() {
            let base = 6 * r;
            for k in 0..3 {
                a2[(base + k, 3 * row.idx_from + k)] = row.delta_time;
                a2[(base + k, w1_off)] = 0.5 * row.delta_time * row.delta_time * mag * b1[k];
                a2[(base + k, w2_off)] = 0.5 * row.delta_time * row.delta_time * mag * b2[k];
                a2[(base + k, s2_off)] = -row.position_vis_diff[k];
                b2vec[base + k] =
                    -row.rhs_position[k] - 0.5 * row.delta_time * row.delta_time * g0[k];

                a2[(base + 3 + k, 3 * row.idx_from + k)] = -1.0;
                a2[(base + 3 + k, 3 * row.idx_to + k)] = 1.0;
                a2[(base + 3 + k, w1_off)] = -row.delta_time * mag * b1[k];
                a2[(base + 3 + k, w2_off)] = -row.delta_time * mag * b2[k];
                b2vec[base + 3 + k] = row.rhs_velocity[k] + row.delta_time * g0[k];
            }
        }
        let Ok(solution2) = a2.svd(true, true).solve(&b2vec, 1.0e-9) else {
            break;
        };
        let w1 = solution2[w1_off];
        let w2 = solution2[w2_off];
        let s_candidate = solution2[s2_off];
        if !w1.is_finite() || !w2.is_finite() || !s_candidate.is_finite() {
            break;
        }
        let candidate = g_hat + w1 * b1 + w2 * b2;
        let candidate_norm = candidate.norm();
        if !candidate_norm.is_finite() || candidate_norm < 1.0e-9 {
            break;
        }
        g_hat = candidate / candidate_norm;
        velocities_final = (0..n)
            .map(|k| Vector3::new(solution2[3 * k], solution2[3 * k + 1], solution2[3 * k + 2]))
            .collect();
        scale_final = s_candidate;
        refined = true;
    }

    let gravity_world = if refined {
        mag * g_hat
    } else {
        mag * (raw_gravity / raw_gravity_norm)
    };
    let scale = if refined { scale_final } else { raw_scale };
    if !scale.is_finite() || scale < gates.min_scale || scale > gates.max_scale {
        // Scale-plausibility gate (module doc, "Observability gates" #3b).
        return Err(DpvoMonoViAlignmentRejection::ScaleOutOfRange {
            scale,
            raw_scale,
            min_scale: gates.min_scale,
            max_scale: gates.max_scale,
            raw_gravity_norm,
            condition_number,
            min_singular_value: min_sv,
            window_frames: n,
        });
    }

    let mut residual_sum_sq = 0.0;
    for row in &rows {
        let v_i = velocities_final[row.idx_from];
        let v_j = velocities_final[row.idx_to];
        let r_pos = v_i * row.delta_time + 0.5 * row.delta_time * row.delta_time * gravity_world
            - scale * row.position_vis_diff
            + row.rhs_position;
        let r_vel = (-v_i + v_j - row.delta_time * gravity_world) - row.rhs_velocity;
        residual_sum_sq += r_pos.norm_squared() + r_vel.norm_squared();
    }
    let mean_residual_after = (residual_sum_sq / (6.0 * rows.len() as f64)).sqrt();

    Ok(DpvoMonoViAlignment {
        scale,
        gravity_world,
        raw_gravity_norm,
        velocities: velocities_final,
        mean_residual_after,
        condition_number,
        window_frames: n,
        min_singular_value: min_sv,
    })
}

/// Orthonormal basis `(b1, b2)` tangent to the unit vector `g_hat` — a
/// verbatim duplicate of `vi_motion_initializer.rs`'s own private
/// `tangent_basis` (see the module doc's "Formulation" section for why this
/// is a deliberate duplication, matching [`skew`]/[`right_jacobian_inverse_so3`]'s
/// own precedent elsewhere in this file).
fn mono_vi_tangent_basis(g_hat: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let reference = if g_hat.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let b1 = g_hat.cross(&reference).normalize();
    let b2 = g_hat.cross(&b1).normalize();
    (b1, b2)
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
            &pose_i,
            &pose_j,
            &v_i,
            &v_j,
            &body_to_camera,
            &factor,
            &bias_gyro,
            &bias_accel,
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
                &pose_i_p,
                &pose_j,
                &v_i,
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i_m,
                &pose_j,
                &v_i,
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_pose_i.column(k);
            let err = (numeric - analytic).norm();
            assert!(
                err < tol,
                "pose_i column {k}: numeric={numeric:?} analytic={analytic:?} err={err}"
            );

            let pose_j_p = SE3::exp(&xi_p).compose(&pose_j);
            let pose_j_m = SE3::exp(&xi_m).compose(&pose_j);
            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j_p,
                &v_i,
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j_m,
                &v_i,
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_pose_j.column(k);
            let err = (numeric - analytic).norm();
            assert!(
                err < tol,
                "pose_j column {k}: numeric={numeric:?} analytic={analytic:?} err={err}"
            );
        }

        // Velocity: plain Euclidean perturbation, no SE3 retraction needed.
        for k in 0..3 {
            let mut e_k = Vector3::zeros();
            e_k[k] = eps;
            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j,
                &(v_i + e_k),
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j,
                &(v_i - e_k),
                &v_j,
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_vel_i.column(k);
            let err = (numeric - analytic).norm();
            assert!(
                err < tol,
                "v_i column {k}: numeric={numeric:?} analytic={analytic:?} err={err}"
            );

            let (r_p, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j,
                &v_i,
                &(v_j + e_k),
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let (r_m, _, _, _, _) = imu_factor_jacobians(
                &pose_i,
                &pose_j,
                &v_i,
                &(v_j - e_k),
                &body_to_camera,
                &factor,
                &bias_gyro,
                &bias_accel,
            );
            let numeric = (r_p - r_m) / (2.0 * eps);
            let analytic = j_vel_j.column(k);
            let err = (numeric - analytic).norm();
            assert!(
                err < tol,
                "v_j column {k}: numeric={numeric:?} analytic={analytic:?} err={err}"
            );
        }
    }

    /// Sign-convention cross-check (module doc): a pure-IMU window (no
    /// visual edges) run through [`dpvo_vi_ba_step`] should reduce the IMU
    /// residual, exactly like a textbook Gauss-Newton step on `‖r‖²` would —
    /// a wrong sign on the RHS would instead *increase* the residual.
    #[test]
    fn pure_imu_window_matches_textbook_gauss_newton_sign() {
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 100.0,
            cy: 100.0,
        };
        let pose0 = SE3::identity();
        // A deliberately "wrong" pose_1 guess (true relative motion is a
        // small rotation + translation; start elsewhere so there is a
        // nonzero residual to reduce).
        let pose1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0),
            Vector3::new(0.5, 0.0, 0.0),
        );

        let problem = DpvoBaProblem {
            poses: vec![pose0, pose1],
            patches: vec![DpvoPatch {
                x: 100.0,
                y: 100.0,
                inverse_depth: 0.2,
            }],
            intrinsics: vec![intr, intr],
            edges: vec![],
            targets: vec![],
            weights: vec![],
            depth_damping: None,
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
        let config = DpvoBaConfig {
            fixedp: 1,
            ..DpvoBaConfig::default()
        };

        let residual_norm_before = {
            let (r, ..) = imu_factor_jacobians(
                &problem.poses[0],
                &problem.poses[1],
                &imu.velocities[0],
                &imu.velocities[1],
                &imu.body_to_camera,
                &factor.factor,
                &imu.bias_gyro,
                &imu.bias_accel,
            );
            r.norm()
        };

        let solved = dpvo_vi_ba_step(&problem, &imu, &config).expect("solvable system");

        let residual_norm_after = {
            let (r, ..) = imu_factor_jacobians(
                &solved.poses[0],
                &solved.poses[1],
                &solved.velocities[0],
                &solved.velocities[1],
                &imu.body_to_camera,
                &factor.factor,
                &imu.bias_gyro,
                &imu.bias_accel,
            );
            r.norm()
        };
        assert!(
            residual_norm_after < residual_norm_before,
            "expected the GN step to reduce the IMU residual: before={residual_norm_before} after={residual_norm_after}"
        );
    }

    /// Milestone M5b: [`imu_factor_nis`] must report a large value for an
    /// obviously inconsistent factor (a huge unexplained relative
    /// translation with a tight, confident covariance) and a small value
    /// for a factor whose state already satisfies it — the signal
    /// `crate::dpvo_vo`'s rollback monitor's own
    /// [`crate::dpvo_vo::rollback_monitor_step`] threshold ultimately acts
    /// on, checked directly at the source rather than assumed meaningful.
    #[test]
    fn imu_factor_nis_is_large_for_an_obviously_inconsistent_factor_and_small_for_a_consistent_one()
    {
        let pose0 = SE3::identity();
        // A large, obviously-implausible relative displacement/velocity for
        // a 0.1s IMU interval (50 m/s over 0.1s would be a 5m jump) — large
        // enough that even the position row alone (`r_p`) dwarfs `100`.
        let pose1 = SE3::new(UnitQuaternion::identity(), Vector3::new(5.0, 0.0, 0.0));
        let v0 = Vector3::zeros();
        let v1 = Vector3::new(50.0, 0.0, 0.0);

        // A tight, confident (identity) covariance so the whitener doesn't
        // wash out the residual magnitude.
        let mut factor = synthetic_factor(0.1, Vector3::new(0.0, 0.0, -9.81));
        factor.delta.covariance = Matrix9::identity();
        factor.delta.delta_position = Vector3::zeros();
        factor.delta.delta_velocity = Vector3::zeros();
        factor.delta.delta_rotation = visloc_core::geometry::SO3::identity();

        // Consistent: with pose_i=pose_j=identity and v_i=v_j=0, gravity
        // still contributes to both r_v/r_p (Forster's residual is
        // gravity-compensated, not gravity-free) — set the delta to
        // exactly what a truly-at-rest body's preintegration would report
        // over this dt, so the residual is genuinely (near) zero.
        let gravity = factor.gravity_world;
        let dt = factor.delta.delta_time;
        factor.delta.delta_velocity = -gravity * dt;
        factor.delta.delta_position = -0.5 * gravity * dt * dt;
        let consistent_nis = imu_factor_nis(
            &pose0,
            &pose0,
            &Vector3::zeros(),
            &Vector3::zeros(),
            &SE3::identity(),
            &factor,
            &Vector3::zeros(),
            &Vector3::zeros(),
        );

        // Inconsistent: a large relative translation/velocity the (still
        // zeroed) delta does not predict at all.
        let inconsistent_nis = imu_factor_nis(
            &pose0,
            &pose1,
            &v0,
            &v1,
            &SE3::identity(),
            &factor,
            &Vector3::zeros(),
            &Vector3::zeros(),
        );

        println!("[imu nis] consistent={consistent_nis:.6e} inconsistent={inconsistent_nis:.6e}");
        assert!(
            consistent_nis < 1.0e-6,
            "expected a near-zero NIS for a self-consistent factor, got {consistent_nis}"
        );
        assert!(
            inconsistent_nis > 100.0,
            "expected a large NIS for an obviously inconsistent factor, got {inconsistent_nis}"
        );
    }

    /// Regression check for the module doc's "Visual assembly" tradeoff:
    /// with zero IMU factors, this module's own (duplicated) visual
    /// assembly must reproduce `dpvo_patch_ba::dpvo_ba_step`'s output on
    /// the same problem.
    #[test]
    fn zero_imu_factors_matches_visual_only_solve() {
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 100.0,
            cy: 100.0,
        };
        let pose0 = SE3::identity();
        let pose1 = SE3::new(
            UnitQuaternion::from_euler_angles(0.02, -0.01, 0.03),
            Vector3::new(0.15, 0.02, -0.01),
        );
        let patch = DpvoPatch {
            x: 105.0,
            y: 98.0,
            inverse_depth: 0.22,
        };
        let target =
            crate::dpvo_patch_ba::transform_point(&pose0, &pose1, &intr, &intr, &patch, false)
                + Vector2::new(0.7, -0.3);
        let problem = DpvoBaProblem {
            poses: vec![pose0, pose1],
            patches: vec![patch],
            intrinsics: vec![intr, intr],
            edges: vec![DpvoEdge { i: 0, j: 1, k: 0 }],
            targets: vec![target],
            weights: vec![Vector2::new(1.0, 1.0)],
            depth_damping: None,
        };
        let config = DpvoBaConfig {
            fixedp: 1,
            ..DpvoBaConfig::default()
        };
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
        let intr = DpvoIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 100.0,
            cy: 100.0,
        };
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
        let true_position_at = |i: usize| true_accel * (0.5 * (dt * i as f64) * (dt * i as f64));

        // True (metric) camera centers in world.
        let true_centers: Vec<Vector3<f64>> = (0..n_frames).map(true_position_at).collect();
        // Starting-guess (wrong-scale) camera centers and poses.
        let guess_centers: Vec<Vector3<f64>> =
            true_centers.iter().map(|c| c * wrong_scale).collect();
        let guess_poses: Vec<SE3> = guess_centers
            .iter()
            .map(|c| SE3::new(UnitQuaternion::identity(), -c))
            .collect();

        // Five patches anchored in frame 0 at varying pixel offsets / true
        // depths, all visible from every later frame.
        let anchor_pixels = [
            (90.0, 100.0),
            (100.0, 100.0),
            (110.0, 100.0),
            (100.0, 90.0),
            (100.0, 110.0),
        ];
        let true_depths = [4.0, 5.0, 6.0, 4.5, 5.5];
        let guess_patches: Vec<DpvoPatch> = anchor_pixels
            .iter()
            .zip(true_depths.iter())
            .map(|(&(x, y), &depth)| DpvoPatch {
                x,
                y,
                inverse_depth: 1.0 / (depth * wrong_scale),
            })
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
                    &guess_poses[0],
                    &guess_poses[j],
                    &intr,
                    &intr,
                    patch,
                    false,
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
            depth_damping: None,
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
            bounds: [
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::INFINITY,
            ],
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

    /// Shared synthetic-window builder for
    /// `estimate_mono_vi_alignment_recovers_scale_from_constant_acceleration_window`:
    /// `n_frames` poses on a true metric trajectory built by integrating
    /// `accel_per_segment` (one 3D acceleration vector per consecutive pair,
    /// `n_frames - 1` of them, each held constant WITHIN its own segment —
    /// see the note below on why this must vary in DIRECTION across
    /// segments, not just be "constant acceleration" in the single-vector
    /// sense `synthetic_window_recovers_metric_scale_within_two_percent`
    /// above uses), identity rotation throughout (rotation does not appear
    /// in this linear system's LEFT-hand-side coefficients at all — only in
    /// the right-hand-side constant term — so, unlike a naive first guess,
    /// rotation diversity cannot fix a genuine column-space degeneracy; see
    /// below), with DPVO's own (non-metric) visual poses scaled by
    /// `visual_scale` from that true trajectory (`p_vis = visual_scale ·
    /// p_true`, so the correct recovered `s` is `1 / visual_scale`), and IMU
    /// factors built from the exact per-segment kinematics.
    ///
    /// # Why acceleration must change DIRECTION across the window
    ///
    /// An earlier version of this helper used one constant `true_accel`
    /// vector for the whole window (mirroring
    /// `synthetic_window_recovers_metric_scale_within_two_percent`'s own
    /// simplification above) and found — via this exact test failing with
    /// `min_sv` EXACTLY `0.0` (a genuine structural rank deficiency, not a
    /// numerical-precision artifact; confirmed by inspecting the SVD's own
    /// null right-singular-vector, which was proportional to `true_accel`
    /// across every `v_i`/`g`/`s` unknown) — that a single-direction
    /// acceleration is ALSO degenerate for THIS alignment's explicit-scale
    /// linear system specifically: every position row's `s` column
    /// (`−(p_j−p_i)_vis`) is then always parallel to that one fixed
    /// direction, so the component of gravity ALONG that same direction
    /// becomes exactly interchangeable with a joint `(v, s)` rescale along
    /// it — the IMU evidence alone cannot tell "gravity has a component
    /// here" from "my scale/velocity drifts here" when the trajectory never
    /// explores a second, independent direction. (A rotation-diversity fix
    /// was tried first and found NOT to help — confirmed by re-running with
    /// varying per-frame rotation and observing byte-identical singular
    /// values — because rotation here only scales the right-hand-side
    /// constant term, never the left-hand-side unknown coefficients that
    /// determine rank.) This mirrors real VINS-Mono practice: mono-VI
    /// initialization windows need genuinely 3D motion, not a single
    /// accelerating direction — MH_01's own real trajectory (turns, height
    /// change) satisfies this trivially; a synthetic single-vector
    /// acceleration does not. Multiple DIFFERENT segment accelerations
    /// break the tie completely (`min_sv` moves from exactly `0.0` to a
    /// well-conditioned finite value — see the test's own logged
    /// `condition_number`).
    fn synthetic_mono_window(
        dt: f64,
        accel_per_segment: &[Vector3<f64>],
        gravity: Vector3<f64>,
        visual_scale: f64,
    ) -> (Vec<SE3>, Vec<DpvoImuFactor>) {
        let n_frames = accel_per_segment.len() + 1;
        // Integrate the (per-segment-constant) accelerations into true
        // metric velocity/position waypoints via exact constant-acceleration
        // kinematics per segment (`v(i+1) = v(i) + a_i·Δt`, `p(i+1) = p(i) +
        // v(i)·Δt + ½·a_i·Δt²`) — exact for ANY sequence of per-segment
        // accelerations, unlike a single closed-form quadratic that only
        // holds for one constant vector over the whole window.
        let mut velocities = vec![Vector3::<f64>::zeros(); n_frames];
        let mut positions = vec![Vector3::<f64>::zeros(); n_frames];
        for (i, &a_i) in accel_per_segment.iter().enumerate() {
            positions[i + 1] = positions[i] + velocities[i] * dt + 0.5 * a_i * dt * dt;
            velocities[i + 1] = velocities[i] + a_i * dt;
        }

        let poses: Vec<SE3> = (0..n_frames)
            .map(|i| SE3::new(UnitQuaternion::identity(), -(positions[i] * visual_scale)))
            .collect();

        let mut factors = Vec::new();
        for i in 0..n_frames - 1 {
            let v_i = velocities[i];
            let v_j = velocities[i + 1];
            let p_i = positions[i];
            let p_j = positions[i + 1];
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
        (poses, factors)
    }

    /// The task's own required synthetic check: a non-metric window scaled
    /// `0.5×` from true metric (so the correct recovered `s` is exactly
    /// `2.0`) with genuine multi-DIRECTIONAL acceleration excitation (see
    /// [`synthetic_mono_window`]'s own doc for why the acceleration
    /// direction must change across the window, not just be nonzero)
    /// recovers `s` and `g` within 1%. Also records the condition number
    /// this well-conditioned case produces —
    /// [`DpvoMonoViAlignmentGates::max_condition_number`]'s default
    /// (`crate::dpvo_vo::DpvoImuConfig::max_mono_alignment_condition_number`)
    /// is calibrated to sit comfortably above this test's own measured
    /// value and comfortably below
    /// [`estimate_mono_vi_alignment_rejects_constant_velocity_window`]'s.
    #[test]
    fn estimate_mono_vi_alignment_recovers_scale_from_constant_acceleration_window() {
        let dt = 0.1;
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let visual_scale = 0.5;
        // Four segments, each a DIFFERENT acceleration direction — see
        // `synthetic_mono_window`'s own doc for why direction diversity
        // (not just nonzero acceleration) is what this test needs.
        let accel_per_segment = [
            Vector3::new(2.0, 1.0, -0.5),
            Vector3::new(-1.0, 2.0, 0.5),
            Vector3::new(0.5, -1.5, 1.0),
            Vector3::new(1.0, 0.5, -1.0),
        ];

        let (poses, factors) = synthetic_mono_window(dt, &accel_per_segment, gravity, visual_scale);
        let gates = DpvoMonoViAlignmentGates {
            expected_gravity_magnitude: 9.81,
            gravity_norm_deviation_ratio: 0.3,
            min_scale: 0.05,
            max_scale: 20.0,
            max_condition_number: 1.0e9,
        };

        let alignment = estimate_mono_vi_alignment(
            &poses,
            &factors,
            &SE3::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            &gates,
        )
        .expect("well-conditioned window must recover an alignment");

        println!(
            "[mono-vi recover] scale={:.6} gravity={:?} raw_gravity_norm={:.4} condition_number={:.3e} \
             mean_residual_after={:.6e}",
            alignment.scale,
            alignment.gravity_world,
            alignment.raw_gravity_norm,
            alignment.condition_number,
            alignment.mean_residual_after,
        );

        let expected_scale = 1.0 / visual_scale;
        assert!(
            (alignment.scale - expected_scale).abs() / expected_scale < 0.01,
            "expected scale within 1% of {expected_scale}, got {}",
            alignment.scale
        );
        let gravity_error = (alignment.gravity_world - gravity).norm() / gravity.norm();
        assert!(
            gravity_error < 0.01,
            "expected gravity within 1%, got error {gravity_error} ({:?})",
            alignment.gravity_world
        );

        // Rejection must retain the solve context needed to distinguish a
        // genuinely unobservable window from a well-conditioned estimate
        // that merely lies outside a configured physical range.
        let rejecting_gates = DpvoMonoViAlignmentGates {
            max_scale: 1.5,
            ..gates
        };
        let rejected = estimate_mono_vi_alignment(
            &poses,
            &factors,
            &SE3::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            &rejecting_gates,
        );
        match rejected {
            Err(DpvoMonoViAlignmentRejection::ScaleOutOfRange {
                scale,
                raw_scale,
                raw_gravity_norm,
                condition_number,
                min_singular_value,
                window_frames,
                ..
            }) => {
                assert!(scale > rejecting_gates.max_scale);
                assert!(raw_scale.is_finite());
                assert!(raw_gravity_norm.is_finite());
                assert!(condition_number.is_finite());
                assert!(min_singular_value > 0.0);
                assert_eq!(window_frames, poses.len());
            }
            other => panic!("expected diagnostic-rich scale rejection, got {other:?}"),
        }
    }

    /// Degenerate case (mirrors this file's own coupled-BA degeneracy note
    /// on `synthetic_window_recovers_metric_scale_within_two_percent`, now
    /// checked directly on the bootstrap's own linear system): a
    /// constant-*velocity* window (zero acceleration) must be REJECTED by
    /// the excitation/conditioning gate, not silently return a wrong scale.
    #[test]
    fn estimate_mono_vi_alignment_rejects_constant_velocity_window() {
        let n_frames = 5;
        let dt = 0.1;
        // Zero acceleration: velocity is the same at every frame. This
        // degenerate/no-excitation case is the whole point of the test.
        let true_accel = Vector3::<f64>::zeros();
        let gravity = Vector3::new(0.0, 0.0, -9.81);
        let visual_scale = 0.5;

        // Constant velocity (`(1,0,0) m/s`, not `true_accel`-derived since
        // `synthetic_mono_window`'s own `true_velocity_at` would be
        // identically zero with `true_accel=0` — a completely stationary
        // window is a degenerate case too, but the classically-cited one is
        // constant NONZERO velocity with zero acceleration).
        let dt_local = dt;
        let v_const = Vector3::new(1.0, 0.0, 0.0);
        let poses: Vec<SE3> = (0..n_frames)
            .map(|i| {
                SE3::new(
                    UnitQuaternion::identity(),
                    -(v_const * (dt_local * i as f64) * visual_scale),
                )
            })
            .collect();
        let mut factors = Vec::new();
        for i in 0..n_frames - 1 {
            factors.push(DpvoImuFactor {
                i,
                j: i + 1,
                factor: ImuPreintegrationFactor {
                    keyframe_id_from: i as u64,
                    keyframe_id_to: (i + 1) as u64,
                    delta: ImuPreintegratedDelta {
                        delta_rotation: visloc_core::geometry::SO3::identity(),
                        // Δv_ij = R_iᵀ(v_j−v_i−gΔt) = −gΔt (v_j=v_i=v_const).
                        delta_velocity: -gravity * dt_local,
                        // Δp_ij = R_iᵀ(p_j−p_i−v_iΔt−½gΔt²); with
                        // p_j−p_i = v_const·Δt exactly (constant velocity),
                        // the `v_const·Δt` terms cancel — this is the
                        // textbook constant-velocity degeneracy, not a
                        // typo: the position delta carries NO trace of
                        // `v_const` at all.
                        delta_position: -0.5 * gravity * dt_local * dt_local,
                        delta_time: dt_local,
                        ..ImuPreintegratedDelta::identity()
                    },
                    gravity_world: gravity,
                    weight_position: 1.0,
                    weight_velocity: 1.0,
                    weight_rotation: 1.0,
                },
            });
        }
        let _ = true_accel;

        let gates = DpvoMonoViAlignmentGates {
            expected_gravity_magnitude: 9.81,
            gravity_norm_deviation_ratio: 0.3,
            min_scale: 0.05,
            max_scale: 20.0,
            max_condition_number: 1.0e9,
        };

        let result = estimate_mono_vi_alignment(
            &poses,
            &factors,
            &SE3::identity(),
            Vector3::zeros(),
            Vector3::zeros(),
            &gates,
        );
        match &result {
            Ok(alignment) => println!(
                "[mono-vi degenerate] UNEXPECTEDLY accepted: scale={:.6} condition_number={:.3e}",
                alignment.scale, alignment.condition_number
            ),
            Err(reason) => println!("[mono-vi degenerate] rejected: {reason:?}"),
        }
        assert!(
            result.is_err(),
            "expected the constant-velocity window to be rejected by the excitation gate"
        );
        assert!(
            matches!(
                result,
                Err(DpvoMonoViAlignmentRejection::IllConditioned { .. })
                    | Err(DpvoMonoViAlignmentRejection::Underdetermined { .. })
            ),
            "expected rejection specifically from the DOF/conditioning gates, got {result:?}"
        );
    }
}
