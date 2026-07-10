# Phase 2 Design — Tight Visual-Inertial Coupling (V2_03 Blackout Survival)

Status: DESIGN DRAFT, 2026-07-09. Not yet implemented. Produced as the "A" arm of
the SOTA push (the "B" arm is the SP-ONNX live-frontend A/B). Supersedes nothing;
companion strategy context in `PLAN.md` §"Strategy 2026-06-12 — Visual-Inertial
SLAM" (Phase 2 there is this document).

## 0. Scope-clarifying finding — read this before doing anything else

The repo contains **two structurally different "online SLAM" pipelines**, and the Phase 2 brief's acceptance criteria (V2_03 completes, MH numbers frozen at .052–.065) come from **the pipeline this design is not primarily about**. This must be resolved (or explicitly accepted) before writing code.

1. **`OnlineStereoVoBa`** (`pipelines/slam/src/online_stereo_vo_ba.rs`, used by `examples/stereo_vo_external_deep_files.rs`) — a feature-file-backed (offline SuperPoint/LightGlue) stereo VO + periodic trailing-window Schur BA + loop closure. This is what `scripts/run_euroc_loop_closure_benchmark.sh` runs, and it is the pipeline that produced the headline `.052/.054/.054/.106†/.065` MH numbers in `docs/euroc_loop_closure_benchmark.md` and the V2_03 DNF description ("triangulated-pair count collapses 59 → 5"). **It has no `Tracker`, no `TrackingState`, no relocalization, no `MotionModel` — no "state machine" for tracking loss at all.** It does already carry `OnlineStereoVoBaConfig::imu_input` / velocity / bias plumbing, but nobody runs it with IMU in the benchmark.
2. **`OnlineSlamPipeline`** (`pipelines/slam/src/online_slam.rs`) — the `Tracker`/`TrackingState`/relocalization/`MotionModel` state machine (the "Online SLAM Visual-Inertial (EuRoC) Workstream", PLAN.md; full log in `docs/motion_based_vi_alignment.md`, Phases 7–27). Per that doc's own Phase-21 diagnostic ("universal tracker cliff") and every subsequent phase through Phase-27, this pipeline currently dies at frame 84–1069 out of 2700+-frame EuRoC sequences on every configuration tried. It has never been run to completion end-to-end on any EuRoC sequence, let alone matched the `.052–.065` MH figures.

**Consequence:** the two pipelines are not interchangeable, and "task C" ("VIO == BA-only to 6 sig figs" on MH) is neither of these — it is an unlabeled prior offline-batch experiment (almost certainly `refine_stereo_vo_with_ba`/`StereoVoBaImuInput`; "to 6 sig figs" only makes sense for a converged batch BA). That conclusion says nothing directly about the *causal, no-loop-closure* streaming regime Phase 2 targets.

**What this design assumes, and what the first delivery step must verify (Step 0 below):** `OnlineSlamPipeline` is the target. **The "zero-regression on MH" gate must therefore be redefined**: there is no existing full-sequence MH number for this pipeline to protect. The real zero-regression baseline is *this pipeline's own current best-known config* (Phase-26/27 recommended flags), measured fresh, per-frame-success-rate and rigid ATE over whatever prefix it currently survives on MH_01/02/03/05.

---

## 1. State augmentation: velocity + bias

**The state augmentation types already exist.** `BundleAdjustment` (`pipelines/slam/src/bundle.rs`) already carries, alongside `poses`/`landmarks`:

- `velocities: BTreeMap<u64, Vector3<f64>>` + `fixed_velocities` (bundle.rs:393-396), one 3-DoF vertex **per keyframe id**.
- `biases: BTreeMap<u64, Vector6<f64>>` + `fixed_biases` (bundle.rs:405-416), one 6-DoF `(b_g, b_a)` vertex **per keyframe id**.
- `imu_factors: Vec<ImuPreintegrationFactor>` (bundle.rs:404), the 9-residual `[r_R; r_v; r_p]` binding two keyframes' `(pose, velocity, bias_from)`.
- `bias_random_walk_factors` / `BiasRandomWalkFactor` (bundle.rs:294-306): a 6-residual `‖b_j − b_i‖²` tie between two keyframes' bias vertices, **already implemented in the solver but never constructed by the online path** (`grep add_bias_random_walk_factor pipelines/slam/src/online_slam_vi_ba.rs` → zero hits). This is the missing piece for §4.

The online-window glue also already exists: `OnlineSlamLocalBaState::keyframe_state: BTreeMap<u64, KeyframeImuState>` (`online_slam_vi_ba.rs:242`, `KeyframeImuState { velocity_world, bias_gyro, bias_acc }` at line 231), and `run_local_vi_ba` (online_slam_vi_ba.rs:492-790) already builds the joint (pose, velocity, bias, landmark) BA per trailing window, with the first window keyframe gauge-fixed. **State lives per-keyframe, not per-frame** — matches ORB-SLAM3-VI; no new vertex type required.

**Confirmed, load-bearing property to keep:** `run_local_vi_ba`'s `active_landmarks` set can be empty (zero visual observations in-window) without the function bailing — it only requires `≥2` window keyframes and `≥1` in-window IMU factor (online_slam_vi_ba.rs:501-519). **The existing function already degenerates cleanly to an inertial-only solve when vision is absent** — the single most important reuse fact for the blackout bridge (§3). `run_inertial_only_vi_ba` (online_slam_vi_ba.rs:806-928) is also available but requires keyframe poses present in `map.keyframes`.

**Initialization (gravity/scale):** already solved (`OnlineSlamViInitConfig` stationary-window init + `MotionBasedViInitializer`); scale is inherent from stereo. **One explicit non-goal in `docs/vi_initialization_integration.md` is exactly Phase 2's subject and must be overridden deliberately:** *"Online re-initialisation after a long tracking loss... is the relocalisation layer's responsibility."* Phase 2 does not need to re-run gravity init after blackout (gravity in world frame doesn't change), but it needs an explicit bias-handling policy for the gap (§4).

---

## 2. Factor graph shape of the online window

Today's wiring (`OnlineSlamPipeline::process_frame`, online_slam.rs:1508-1615):

1. `tracker.track_frame` → `maybe_run_relocalization` (only on primary-tracking failure).
2. On success → `keyframe_from_tracking_result` → `mapper.process_keyframe` → `applied_update`.
3. `stage_imu_factor_on_new_keyframe` (online_slam.rs:2553-2611): **only fires when `applied_update.keyframe_count > 0`**. Snapshots the running `ImuPreintegrator`, emits an `ImuPreintegrationFactor { keyframe_id_from: prev_id, keyframe_id_to: new_keyframe_id }`, resets the integrator. Consecutive-keyframe chain — already the ORB-SLAM3-VI shape.
4. `maybe_run_local_vi_ba` (online_slam.rs:2348-2372) → `run_local_vi_ba`, gated by `OnlineSlamLocalBaConfig::trigger_every` (default 1). Window = trailing `window_size` (default 5) keyframes; `factor_history` capped at `4 * window_size` (online_slam_vi_ba.rs:276-289). **This is today's fixed-prefix-equivalent for VI-BA: a hard-fixed first-window-keyframe gauge + a bounded history ring buffer — no marginalization prior carried forward** (contrast `PoseGraph::marginalize_oldest*`, pose_graph.rs:1745-1770, used only by the loop-closure stage). A real, pre-existing gap; acceptable to leave as-is for Phase 2, flag as known approximation.
5. `covisibility_local_ba` (visual-only, no IMU) can run in the **same** `process_frame` call *after* `local_vi_ba` (online_slam.rs:1580-1585) and writes poses back independently — **the two BA passes can fight each other, and nothing indicates the combination was ever measured together.** Phase 2 recommendation: forbid enabling both simultaneously (assert in `OnlineSlamConfig::validate`).

**What "permanently in the online-BA window" requires that doesn't exist yet:**
- **Δt-scaled factor weights.** `ImuPreintegrationFactor::weight_position/velocity/rotation` are single scalars set once from `OnlineSlamImuConfig` (online_slam.rs:1206-1221), used verbatim in the residual cost (bundle.rs:749-751, 2432-2434) — **not scaled by `factor.delta.delta_time`**. A 0.05 s factor and a multi-second blackout factor get identical confidence today. Must be fixed as part of Phase 2 (Step 1) — blackout-spanning factors are the first with wildly non-uniform Δt.
- **Bias random-walk ties between consecutive keyframes** (§4).

---

## 3. The blackout bridge: state machine

### What already exists (mechanically works, never validated for multi-second gaps)

- `ImuPredictiveMotionModel::observe` (pipelines/tracking/src/motion.rs:421-458) is a no-op on tracking failure — pending IMU samples are not drained, `Tracker::last_successful_pose` (tracker.rs:509) stays frozen, and `predict_pose` (motion.rs:376-419) re-integrates the *entire* accumulated blackout window every frame (forward-Euler strapdown, frozen biases, pre-blackout `velocity_world`). **The mechanical dead-reckoning fallback already exists** — it needs to be made trustworthy (§4), committed into shared state (below), and gated correctly on re-acquisition.
- **Do not use `AdaptiveImuPoseMotionModel`** (motion.rs:652-820) for this: it switches to `ConstantPoseMotionModel` after only 2 consecutive failures (default, motion.rs:621) — exactly wrong for a multi-second blackout. Keep plain `--motion-model imu`.
- `stage_imu_factor_on_new_keyframe` naturally spans the blackout: no keyframe registers while lost, the `ImuPreintegrator` keeps accumulating, and the next registered keyframe closes **one** factor whose `delta_time` equals the whole gap.
- Existing re-acquisition gates are the wrong *shape* for IMU-propagated priors: `OnlineSlamRelocalizationConfig::max_translation_from_imu_prediction_meters` (online_slam.rs:300-312) is a fixed absolute radius; `TrackingConfig::pose_jump_gap_scaling` (tracker.rs:581-602) scales linearly in *frame count* and its doc rationale ("the prior hasn't moved") assumes a frozen prior — **conceptually mismatched with `--motion-model imu`**, where true error grows ~quadratically in Δt for bias-driven drift. It is accidentally-adequate (capped multiplier, generous base radius), not principled.

### Recommended design

1. **Uncertainty growth without a full EKF.** Maintain a closed-form scalar "predicted position 1-σ radius" from the already-accumulated `ImuPreintegratedDelta::j_position_ba/bg`, `j_velocity_ba/bg` Jacobians (imu_preintegration.rs:43-52) plus a configured bias-noise density (dominant term ~`‖j_position_ba‖·σ_ba·Δt` — the term that blew up the 2026-06 loose bridge). Replace the fixed radius with `radius(Δt) = base_radius + k · Δt` or the Jacobian-derived proxy.
2. **Bridge keyframes = "propagation with committed v/b".** `Keyframe { frame, observations }` is trivially constructible with zero observations (`Frame::new(id, camera_id)` + `pose: Some(propagated_pose)`). Add an `OnlineSlamPipeline` insertion path (parallel to `mapper.process_keyframe`) that — **only while `TrackingState::Lost`** — periodically (every `bridge_keyframe_interval_frames` or `bridge_keyframe_interval_seconds`, whichever first) inserts a synthetic keyframe at the IMU-propagated pose into `self.map.keyframes`, tagged in a new `OnlineSlamImuState::bridge_keyframe_ids: HashSet<u64>` so every consumer of `map.keyframes` (covisibility BA window selection, loop-closure candidates, relocalization descriptor stores) is **audited and explicitly made to skip these ids**. Generalize `stage_imu_factor_on_new_keyframe`'s gate to fire on bridge insertion. Every blackout segment becomes a short chain of ordinary `ImuPreintegrationFactor`s between bridge keyframes, each with its own `KeyframeImuState` velocity/bias vertex — v/b are BA state at every bridge checkpoint, not a silently-propagated side variable. Uses `run_local_vi_ba`'s degenerate-to-inertial-only behavior (§1); no new BA code.
3. **Re-acquisition gate composition.** Accept a PnP/relocalization recovery once it clears (a) the existing inlier/ratio/reprojection gates AND (b) the Δt-scaled covariance gate from (1) evaluated against the *last bridge keyframe's* pose. On acceptance, close the final bridge→real factor, fold the recovered keyframe into the normal window (mixed tracked/bridge/recovered keyframes need no special-casing), stop inserting bridge keyframes.
4. **Relocalization interaction.** Bridge insertion runs independently of `OnlineSlamConfig::relocalization`; `maybe_run_relocalization` continues as today but warm-starts against the bridge chain's prediction (it already supports `pose_prior_candidate_radius_meters`), with the Δt-scaled gate in addition to `max_translation_from_imu_prediction_meters`.

---

## 4. Bias observability during blackout

Biases are **not observable** from IMU alone (pure inertial links with free velocity have as many unknowns as equations — LM can zero any link's residual with arbitrary bias; a no-op, not a correction). Neither freeze (reproduces the loose-bridge failure) nor free-float (absorbs drift, produces nonsense).

**Recommendation: wire the already-implemented `BiasRandomWalkFactor` (bundle.rs:293-306) into `run_local_vi_ba`'s `build_ba` closure** (online_slam_vi_ba.rs:603-657), one factor per consecutive in-window keyframe pair (bridge and real alike), `weight = 1.0 / (sigma_bias_random_walk² · Δt_ij)` per the factor's own doc-comment formula, using each factor's own `delta.delta_time`. Short Δt → stiff tie; long Δt (blackout) → loose tie, never fully unconstrained. Starting σ values for the EuRoC ADIS16448: ~1e-4 rad/s/√s gyro, ~1e-3 m/s²/√s accel; tune against Step 3's gate. Unit-testable in isolation (two-keyframe chain, big Δt: refined bias must move less than unconstrained control, more than frozen control).

---

## 5. Incremental delivery plan (PR-sized steps)

Each step runs `scripts/check.sh` and updates `CHANGELOG.md`/`docs/motion_based_vi_alignment.md`.

**Step 0 — Baseline measurement + scope lock (no production code).**
Run `euroc_online_slam_vi_image_demo` with the Phase-26/27 recommended config (+ `--local-vi-ba`) fresh on MH_01/02/03/05: record `last_tracked_frame`, success rate, rigid ATE over achieved prefix — the zero-regression baseline for Steps 1+. Run V2_03 once unmodified: record where/how it fails (frame, `TrackingState` transition, stereo-pair count) as the "before" reference. **Must check whether V2_03 dies from the universal tracker cliff before reaching the blackout — if so, Phase 2's scope must expand before the bridge can be exercised (§6).**
*Gate:* table committed to `docs/motion_based_vi_alignment.md`.

**Step 1 — Δt-aware IMU factor weighting (bundle.rs + imu_preintegration.rs).**
`weight_x(Δt) = base_weight_x / f(Δt)` with `f` from propagated-Jacobian magnitude or a documented Δt/Δt² heuristic.
*Gate:* unit tests ("10× Δt ⇒ ≤1/10 weight"); MH_01/02/03/05 with `--local-vi-ba` bit-identical-or-noise-floor vs Step 0.

**Step 2 — Bias random-walk factors in the online window (online_slam_vi_ba.rs).**
§4; new `OnlineSlamLocalBaConfig::bias_random_walk_noise_density: Option<(f64, f64)>`, `None` = today's behavior.
*Gate:* §4 unit test; with default `None`, MH bit-identical to Step 1.

**Step 3 — Bridge-keyframe blackout bridge (online_slam.rs + new `online_slam_imu_bridge.rs`).**
§3: insertion while Lost, tagged-id skip-list audited across covisibility_ba.rs / vo_loop_closure.rs / loop_gating.rs / relocalization stores; generalized factor staging; Δt-scaled gate additive in `maybe_run_relocalization`.
*Gate:* unit tests (bridge KFs excluded from all consumers; chain closes factors both ends; reset clears state). V2_03 progresses past Step 0's failure point. MH bit-identical with the feature disabled (opt-in flag, default off).

**Step 4 — V2_03 tuning pass (config only).**
Sweep `bridge_keyframe_interval_*`, gate constant `k`, bias noise densities on V2_03.
*Gate:* V2_03 completes, lands in the 0.05–0.11 m causal-VIO band. MH unchanged; run V1_03/V2_01/V2_02 and record (no formal gate).

**Step 5 (stretch) — covisibility/local-vi-BA mutual exclusion + docs closeout.**
`OnlineSlamConfig::validate` rejects both enabled simultaneously (or fold IMU factors into covisibility BA if headroom). Update `docs/next_development_plan.md` row D + closing Phase-2 summary.

---

## 6. Risks / loose-bridge failure modes to avoid

- **Loose-bridge root cause** (drifted 32–285 m): propagation without v/b co-estimation. Avoided structurally: every bridge checkpoint is a BA vertex written back into `KeyframeImuState`/`BundleAdjustment.{velocities,biases}` (online_slam_vi_ba.rs:755-768), never a private integrator field.
- **Zero-observation keyframes are a structural novelty** — audit every `map.keyframes` consumer; the tagged-id skip-list is mandatory, not hardening. Covisibility-graph shared-landmark counting could otherwise select a degenerate anchor.
- **Δt-unaware IMU weights are a pre-existing latent bug** that Phase 2 is first to expose — fix (Step 1) *before* the bridge (Step 3) so Step 3/4 gates measure the bridge, not weighting artifacts.
- **`pose_jump_gap_scaling` is written for frozen priors** — do not extend it; add the Δt/covariance gate as a parallel, IMU-aware gate.
- **`covisibility_local_ba` + `local_vi_ba` uncoordinated in one `process_frame`** — resolve (mutual exclusion or measured combination) before/alongside Step 3, or gains can be invisibly erased.
- **Scope risk (§0):** if the universal tracker cliff kills V2_03 before the blackout, blackout-bridging alone cannot complete it — Step 0 must determine this first. Note: the "B" arm (SP-ONNX live frontend) attacks exactly the cliff/coverage problem; sequencing B-wins into the Step 0 baseline config may be the cheapest cliff fix.

## 7. Rust constraints / reuse

- Both `pipelines/slam` and `pipelines/tracking` are `#![forbid(unsafe_code)]`; all math nalgebra. No new deps: velocity/bias vertices, IMU factor, bias random-walk factor, Schur BA + GNC robust kernels all exist in bundle.rs. Phase 2 is glue + one small bridge-policy module.
- Extend `ImuPreintegratedDelta` only additively if full covariance propagation replaces the Δt heuristic (check `docs/api_stability.md` before altering public fields).

### Critical files

- `pipelines/slam/src/online_slam.rs`
- `pipelines/slam/src/online_slam_vi_ba.rs`
- `pipelines/slam/src/imu_preintegration.rs`
- `pipelines/slam/src/bundle.rs`
- `pipelines/tracking/src/motion.rs`
- `pipelines/tracking/src/tracker.rs`
- `examples/euroc_online_slam_vi_image_demo.rs`
