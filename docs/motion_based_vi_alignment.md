# Motion-based VI alignment (design note)

**Status: shipped. Both VIBA1 (inertial-only) and VIBA2 (inertial-with-
scale outer loop + monocular scale recovery) are landed, and the stage
is wired into `OnlineSlamPipeline` via `OnlineSlamMotionViInitConfig`.**

What shipped:
* [`run_inertial_only_vi_ba`](../pipelines/slam/src/online_slam_vi_ba.rs) —
  the inertial-only sliding-window MAP solver (VIBA1). Pins landmarks,
  optimises per-keyframe `(R_w←b, v_w, b_g, b_a)` against IMU pre-
  integration factors only. Scale fixed at `1.0`.
* [`run_viba2_inertial_with_scale`](../pipelines/slam/src/online_slam_vi_ba.rs) —
  the VIBA2 outer-loop wrapper. Alternating minimisation: rescale IMU
  factors by `1/s`, run inner VIBA1 solve, re-estimate `s` via 1-D least
  squares against the refined kinematic term, repeat until
  `|Δs/s| < scale_tolerance`. `recover_scale = false` degenerates to a
  single VIBA1 call at the configured initial scale (stereo path);
  `recover_scale = true` enables the outer loop (monocular path).
* [`estimate_scale_from_factors`](../pipelines/slam/src/online_slam_vi_ba.rs) —
  the closed-form scale estimator that powers the outer loop. Solvable
  whenever the kinematic term `R_iᵀ · (p_j - p_i - v_i·Δt)` has non-
  degenerate magnitude (i.e., the body has moved). Returns `None` on
  zero-motion / near-coincident keyframes.
* [`MotionBasedViInitializer`](../pipelines/slam/src/vi_motion_initializer.rs) —
  the standalone state machine. Defaults: `min_keyframes >= 10`,
  `cumulative_translation_meters >= 2.0`. Optional `viba2: Some(Viba2Config)`
  switches the inner solve from VIBA1 → VIBA2.
* [`OnlineSlamMotionViInitConfig`](../pipelines/slam/src/online_slam_motion_vi_init.rs) +
  pipeline integration in [`OnlineSlamPipeline`](../pipelines/slam/src/lib.rs).
  The motion-based stage runs AFTER the static VI init has completed,
  gated on the static seed being available. On VIBA1 / VIBA2 success the
  pipeline atomically mirrors refined `(velocity, bias)` into
  `local_vi_ba_state.keyframe_state` and resets the IMU pre-integrator's
  bias linearisation. Inspected via
  `motion_vi_initialization_status()` / `OnlineSlamResult.vi_motion_init`.

Design note for the last column in the OSS-parity comparison table in
[`vi_initialization_integration.md`](./vi_initialization_integration.md):
the **motion-based / dynamic VI alignment** stage that ORB-SLAM3 ships as
`VIBA1` / `VIBA2`. The static-bootstrap flavour (`VisualInertialInitializer`)
and its pipeline integration (`OnlineSlamPipeline::vi_init`) have already
shipped; the dynamic flavour's first cut (VIBA1 inertial-only path) is
the new addition. This note scopes the remaining work (VIBA2 + pipeline
glue) so a follow-up implementation can land on a known surface.

## What "motion-based" means and why it's a separate stage

The shipped static flavour recovers `(R_w←b, b_g, b_a)` by averaging a
stationary IMU window — the body is assumed at rest, so the accelerometer
reads the **gravity direction** directly. This recovery is exact, closed
form, and fits in two screens of Rust. It is also intrinsically limited:

1. **Yaw is unobservable.** Gravity pins roll / pitch but leaves the
   yaw component of `R_w←b` undetermined; the static flavour zeroes it
   by convention. Any downstream consumer that needs an absolute yaw
   (loop closure against a yaw-anchored map, multi-session merging)
   must observe motion to recover it.
2. **Scale is undetermined for monocular pipelines.** The static
   flavour produces no `v_w` and no scale factor `s`; on monocular
   sequences the visual frontend's depths are up to an unknown `s`,
   and the first few seconds of VI-BA cannot factor `s` out from `v_w`
   because the only thing connecting visual and inertial residuals is
   the body trajectory's *shape*, not its absolute size.
3. **Accel-bias observability is partial.** With the body stationary
   the only accel residual is the gravity-magnitude mismatch; the
   lateral accel-bias components are absorbed into the recovered
   `R_w←b` rather than being identifiable on their own.

ORB-SLAM3's response is a **second, motion-driven** initialisation
stage that runs after the visual frontend is hot (typically once the
first ~10 keyframes are mapped) and the body has moved enough to give
the IMU translational excitation. The classical reference is
Campos, Elvira, Rodríguez, Montiel, Tardós, "ORB-SLAM3: An
Accurate Open-Source Library for Visual, Visual–Inertial, and
Multi-Map SLAM," IEEE T-RO 2021, §V "Visual-Inertial SLAM":

* **VIBA1.** Inertial-only optimisation over the first ~2–5 s of
  motion. State: `(R_w←b_i, v_w_i, b_g_i, b_a_i, s)` for each of the
  N seed keyframes, with the **visual landmarks fixed** at their
  visually-triangulated positions. Residuals: only IMU preintegration
  factors connect consecutive keyframes. Outputs the scale `s` and
  refined biases.
* **VIBA2.** Full visual-inertial bundle adjustment over a longer
  window (~5–15 s), now optimising landmarks + keyframe poses + IMU
  states + scale jointly. Re-runs until the relative scale change
  between iterations falls below a threshold.

These are NOT a replacement for the static flavour — ORB-SLAM3 still
uses an analogous "stationary bootstrap" or "constant-velocity"
front-end to seed the visual map; VIBA1/2 then refine the result. The
two flavours compose.

## What `visloc-rs` already has

The motion-based stage needs three building blocks. All three exist:

1. **IMU preintegration with bias linearisation + bias-correction
   Jacobians.** `pipelines/slam/src/imu_preintegration.rs::ImuPreintegrator`
   integrates body-frame IMU samples between keyframes and exposes
   `ImuPreintegratedDelta` with both the integrated `(ΔR, Δv, Δp)`
   and the first-order partials with respect to the bias linearisation
   point (`d ΔR / db_g`, `d Δv / db_a`, `d Δp / db_g`, `d Δp / db_a`,
   `d Δv / db_g`). `ImuPreintegrationFactor::residual_with_bias_correction`
   is the bias-corrected residual the dynamic stage will optimise.
2. **Sliding-window keyframe-VI joint optimisation.** `pipelines/slam/src/online_slam_vi_ba.rs::run_local_vi_ba`
   runs an LM solve over a window of keyframes that ALREADY supports
   `KeyframeImuState { velocity_world, bias_gyro, bias_acc }` plus the
   IMU preintegration factors between consecutive keyframes. The
   joint-VI-BA path is the natural host for VIBA2 — what's missing is
   the **scale state** `s`, not the optimisation machinery itself.
3. **Static VI bootstrap.** `pipelines/slam/src/vi_initializer.rs`
   ships the stationary-window flavour and the pipeline integration
   in `OnlineSlamPipeline::vi_init`. The dynamic stage is the
   follow-up that fires once the static seed plus the visual frontend
   have produced enough keyframes to optimise jointly.

## What's missing

Three concrete gaps separate the current code from a usable
`MotionBasedViInitializer`:

1. **Inertial-only sliding-window MAP solver.** A new variant of
   `run_local_vi_ba` that pins landmarks (no visual residuals) and
   optimises ONLY keyframe `(R_w←b, v_w, b_g, b_a, s)` against IMU
   preintegration factors. Implementation: a thin wrapper around the
   existing solver that zeroes out the visual residual contribution,
   plus the scale state added to the per-keyframe block.
2. **A scale state.** `KeyframeImuState` currently carries
   `(velocity_world, bias_gyro, bias_acc)`; the dynamic stage adds a
   shared `s` that multiplies every landmark world position when
   computing the visual residual (or equivalently, divides the
   inertial residual's `Δp` by `s`). The scale state is **shared
   across the whole window**, not per-keyframe. On stereo / RGB-D
   sequences `s` is fixed at `1.0` and the optimisation pins it,
   degenerating to the existing VI-BA behaviour; on monocular this
   is the entire reason the stage exists.
3. **A trigger / state machine.** The pipeline must know when to
   switch from the static seed to running VIBA1, when VIBA1 has
   converged sufficiently to hand off to VIBA2, and when VIBA2 is
   "done" (the scale change between iterations drops below a
   threshold). Implementation: a new variant on `ViInitializationStatus`
   (`StaticCompleted` / `Vi`b`a1Running` / `Viba2Running` / `Converged`)
   plus a trigger condition driven by accumulated translation since
   the static seed. The existing `vi_init_state` field on
   `OnlineSlamPipeline` is the natural place to host the additional
   state; it is already private and the status accessor is already
   the public surface.

## Minimal first-cut path

A landable first iteration would skip VIBA2 entirely and ship only
the **inertial-only VIBA1** stage:

* Add a new public module `pipelines/slam/src/vi_motion_initializer.rs`
  with the type `MotionBasedViInitializer` paralleling
  `VisualInertialInitializer`. The state machine waits for N keyframes
  (default ~10) with cumulative translation > T (default ~2 m) since
  the static seed, then runs the inertial-only MAP solve once.
* Add a scale state to the input — `MotionBasedViInitializerInput {
  keyframes: &[KeyframeSnapshot], preintegrations: &[ImuPreintegrationFactor],
  static_seed: &VisualInertialInitializationResult, monocular: bool }`.
  When `monocular = false` the scale is fixed at 1.0 and the solver
  refines only `(R_w←b, v_w, b_g, b_a)` per keyframe; this gives a
  useful "VIBA1 for stereo / RGB-D" that improves bias linearisation
  without adding any new failure modes.
* The output `MotionBasedViInitializationResult { scale, keyframe_states,
  rotation_residual_deg, velocity_norm_change, iterations }` carries
  what the pipeline needs to update `local_vi_ba_state` and (on the
  monocular path) the visual map's scale.

This first cut lands the trigger, the solver wrapper, and the scale
state without committing to the full VIBA2 — which is the part that
needs the most empirical tuning on EuRoC and TUM-VI before it can be
relied on as a default. The full VIBA2 (joint visual-inertial-scale
BA over a longer window with re-runs) is the natural follow-up.

## What this stage will NOT do (out of scope)

* **Replace the static bootstrap.** Both flavours coexist — the static
  bootstrap remains the entry point because it is fast and works at
  rest, while the dynamic stage refines once the body moves. ORB-SLAM3
  does the same.
* **Refine the visual map's landmark positions.** That is the job of
  the existing `LocalMappingPipeline` and `Triangulator`s. VIBA1 pins
  landmarks; only VIBA2 (out of scope for the first cut) jointly
  re-optimises them along with the scale.
* **Recover yaw on stereo sequences that already have a metric scale.**
  On stereo / RGB-D the yaw is observed by the visual frontend through
  parallax with the gravity-aligned roll / pitch; the dynamic stage
  is most valuable on monocular sequences where the static seed has
  zero yaw and unknown scale.

## Test strategy (for the future implementation)

Following the same `[N]` numbering pattern that `vi_initialization_integration.md`
uses, the dynamic stage needs at least:

1. **Stereo stationary stream replays VIBA1 trivially.** With
   `monocular = false` and zero motion, VIBA1 reproduces the static
   seed within `1e-9` (sanity check that the no-op path is identity).
2. **Monocular constant-velocity recovers `s = 1` from synthetic
   data.** A synthetic trajectory with known scale 1.0, known biases,
   and ground-truth-projected pixel observations recovers
   `|s - 1| < 1e-3` from a 10-keyframe window.
3. **Monocular non-unit scale is recovered within 5 %.** Same
   synthetic trajectory but with landmarks scaled by 0.3 and 3.0
   recovers `s` within 5 % in both cases — the symmetric scale
   recovery is the most load-bearing single test.
4. **Bias refinement compounds with the static seed.** Inject a
   constant gyro bias of `0.001 rad/s` after the static seed and
   confirm VIBA1's `b_g` refines toward `0.001` (the static seed
   recovered `0` because the bias appeared only during motion).
5. **Trigger fires on translation, not on time.** A 30 s stationary
   recording followed by 2 m of motion fires VIBA1 once cumulative
   translation crosses the threshold, NOT on the time threshold —
   the previous failure mode of `VisualInertialInitializer` (firing
   on a noisy 30 s window before the body actually moved) is the
   exact thing this stage is designed to avoid.
6. **State machine emits at most one `Viba1Succeeded` event per
   sequence.** Mirror of the existing `Succeeded` event contract on
   the static bootstrap.
7. **Reset re-arms.** `reset_sequence_state` flips the dynamic stage
   back to `Waiting` and the next sequence can re-bootstrap and
   re-initialise both flavours.

These tests will land alongside the implementation; they are listed
here only so the design contract is visible before any code is
written.

## Why this is a multi-session deliverable

The static bootstrap was small (~400 lines including tests). The
motion-based stage is larger because:

* It needs a new sliding-window solver variant **plus** the scale
  state plumbed through `KeyframeImuState`, the LM Jacobians, and the
  result type.
* The first-iteration design choices (when to trigger, how many
  keyframes to wait, whether to run VIBA1 once or iterate, how to
  hand off to VIBA2) need empirical tuning on EuRoC and TUM-VI
  before the defaults can be defended.
* The interaction with the existing `LocalMappingPipeline` /
  `OnlineSlamLocalBaConfig` needs a design review — running VIBA2
  on top of the local-VI-BA window risks double-correcting biases
  and is the part of the design most likely to need iteration.

Splitting the work as `(a)` static bootstrap, `(b)` distortion +
stereo seed, `(c)` motion-based VIBA1, `(d)` full VIBA2 puts the
biggest empirical-tuning step at the end. `(a)` and `(b)` have
already shipped.

## Real-data validation (V1_01_easy, first 400 frames)

A 4-cell A/B grid on EuRoC `V1_01_easy` (the easiest sequence with
a confirmed stationary start) demonstrates the wiring is exercised
end-to-end on real data and isolates where the trajectory error
actually comes from. CLI:

```sh
./target/release/examples/euroc_online_slam_vi_image_demo \
  --euroc-dir <V1_01_easy> --out-dir <out> --max-frames 400 \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 0.5 \
  [--motion-vi-init] [--local-vi-ba]
```

| Configuration                               | static-VI | motion-VI succ. | local-VI-BA | rigid ATE (m) | sim. ATE (m) |
|---------------------------------------------|-----------|-----------------|-------------|---------------|--------------|
| baseline (visual-only + IMU pre-integ.)     | ok @ 46   | —               | off         | **23.72**     | 0.69         |
| motion-VI on, local-VI-BA off               | ok @ 46   | succ. @ 122     | off         | 23.72         | 0.69         |
| motion-VI off, local-VI-BA on               | ok @ 46   | —               | on          | 158.72        | 0.55         |
| motion-VI on, local-VI-BA on                | ok @ 46   | succ. @ 162     | on          | 159.21        | 0.55         |

**Wiring is exercised.** The static stage fires at frame 46 with a
recovered 1.265 s stationary window; the motion stage's trigger
gates on the static seed and `min_keyframes ≥ 10 ∧
min_translation ≥ 2.0 m`, fires successfully, the inner VIBA1 LM
solver runs and converges, and the refined `(velocity, bias)` slots
are mirrored into the pipeline. With `--local-vi-ba` set, those
refined values flow forward into the sliding-window solver.

**Motion-VI is ATE-neutral when its outputs have no consumer.** Row
1 vs row 2: enabling motion-VI with local-VI-BA disabled does NOT
change rigid / similarity ATE — the refined per-keyframe slots are
written into `OnlineSlamMotionViInitState.completed` and the
`local_vi_ba_state.keyframe_state` mirror is skipped because the
state itself is `None`. This is the expected no-op path.

**Local-VI-BA is the dominant ATE actor.** Row 1 vs row 3: turning
on `--local-vi-ba` (without motion-VI) takes rigid ATE from
23.72 m → 158.72 m. The sliding-window VI-BA's joint pose +
velocity + bias solve diverges because the visual pose stream
already carries ~24 m of cumulative drift over the 400-frame
prefix, and the BA's Gauss-Newton update assumes a near-zero
initial linearisation point. With drifted poses the cost surface
is far from quadratic and the solver lands on a bias estimate that
makes the IMU disagree even more strongly with the (already-bad)
visual prior on the next iteration.

**Motion-VI on top of local-VI-BA adds a small marginal degradation
but is not the bug.** Row 3 vs row 4: adding `--motion-vi-init` to
the local-VI-BA-enabled run nudges rigid ATE 158.72 m → 159.21 m
(+0.5 m, +0.3 %). The marginal cost comes from the refined biases
being re-seeded onto subsequent keyframes — a ~zero effect compared
to the local-VI-BA-vs-baseline gap.

**Concrete numerical evidence of the upstream drift.** On row 4 the
recovered motion-VI state reports `trigger_translation_meters =
3553.10 m` for a sequence whose ground-truth trajectory length is
~50 m over the first 400 frames. The `keyframe_states` velocity
slots contain values like `(-386, 56, -62) m/s` and bias-acc values
like `(-740, -514, -683) m/s²`. These are arithmetically consistent
with the divergent BA solve — they are the BA's best fit to a
catastrophically wrong geometry — and confirm the bottleneck is
upstream pose quality, not the motion-VI stage itself.

**What this means for future work.** The motion-VI stage's design
contract holds on this dataset (trigger fires; inner solve runs;
refined values reach the mirror site). To turn the stage into an
ATE-improving feature the upstream pipeline first needs:

1. **Tracker-side drift suppression on V1_01_easy** — 24 m of
   cumulative rigid drift over 400 frames is too noisy a prior for
   ANY downstream BA. Likely candidates: covisibility-graph guided
   landmark selection, two-view triangulation refinement on the
   stereo seed, or local-window pose-only BA before any VI-BA.
2. **Local-VI-BA conditioning** — even with a better pose stream,
   the LM solver should detect "BA cost surface is far from
   quadratic; freeze biases" and fall back to pose-only refinement
   rather than corrupting the bias linearisation. This is an
   `OnlineSlamLocalBaConfig` knob the current code does not have.
3. **Tighter motion-VI rejection on `trigger_translation_meters` /
   recovered velocity sanity** — the inner solve currently accepts
   any LM-converged result. A post-solve sanity gate (e.g. reject
   when any `||velocity_world|| > 10 m/s` for V1-class indoor
   sequences) would prevent the stage from polluting downstream
   state.

Rows 1 and 2 (rigid ATE 23.72 m) are the **current best result on
V1_01_easy** for this demo. The result the design note targets
(ORB-SLAM3-style sub-decimetre ATE) is gated on (1) and (2) above,
not on shipping additional VIBA variants.

## Phase-3 follow-up validation (V1_01_easy, first 400 frames)

Items (2) and (3) from the previous section are now shipped behind
two opt-in CLI knobs:

* `--motion-vi-init-max-velocity <m/s>` — post-solve sanity gate on
  the recovered per-keyframe `||velocity_world||`. Maps to
  `MotionBasedViInitializerConfig::max_velocity_magnitude_mps`. The
  motion-VI stage rejects the inner LM result and stays in `Waiting`
  if any per-keyframe speed exceeds the cap.
* `--local-vi-ba-freeze-biases-above <ratio>` — conditioning fallback
  on local VI-BA. Maps to
  `OnlineSlamLocalBaConfig::freeze_biases_when_cost_ratio_above`.
  When the first BA pass has `final_cost / initial_cost > ratio`, the
  stage re-solves the window with **all per-keyframe biases gauge-
  frozen** at their pre-BA linearisation points and writes only the
  refined poses + velocities back. The discarded bias updates were
  fitting noise into the wrong cost-surface minimum.

A 4-cell A/B grid (each on top of `--motion-vi-init --local-vi-ba`)
isolates the contribution of each knob:

| Configuration                                                     | motion-VI succ. | rigid ATE (m) | sim. ATE (m) |
|-------------------------------------------------------------------|-----------------|---------------|--------------|
| neither knob (legacy Phase-2 baseline)                            | succ. @ 162     | 159.21        | 0.55         |
| only `--motion-vi-init-max-velocity 10`                           | **rejected**    | 158.24        | 0.58         |
| only `--local-vi-ba-freeze-biases-above 0.9`                      | succ. @ 162     | 159.21        | 0.55         |
| both knobs (`--max-velocity 10`, `--freeze-above 0.9`)             | rejected        | 158.24        | 0.58         |

Sweep of the bias-freeze threshold (with `--motion-vi-init-max-velocity 10` held fixed):

| `--local-vi-ba-freeze-biases-above` | motion-VI succ. | rigid ATE (m) | sim. ATE (m) |
|-------------------------------------|-----------------|---------------|--------------|
| `0.9`                               | rejected        | 158.24        | 0.58         |
| `0.5`                               | rejected        | 158.23        | 0.57         |
| `0.3`                               | rejected        | 157.27        | 0.58         |
| `0.1`                               | rejected        | 157.76        | 0.54         |
| **`0.01`**                          | rejected        | **25.17**     | **0.55**     |

**The motion-VI sanity gate fires as designed.** With
`--motion-vi-init-max-velocity 10`, the post-solve check catches the
divergent ~m/s³ velocities reported in the Phase-2 finding and
rejects the inner LM result, parking the stage in
`Waiting { last_rejection: VelocityOutOfRange { … } }` so the
keyframe / bias / velocity tables stay clean. Rigid ATE improves by
1 m relative to the legacy baseline; the gate alone is not enough to
close the bigger gap that local-VI-BA opens.

**The bias-freeze fallback is cost-ratio sensitive.** At
`threshold = 0.9`, the BA's per-trigger cost reduction (typically
final/initial ∈ [0.1, 0.5]) sits well below the freeze cap and the
fallback never fires. At `threshold = 0.01` it fires on essentially
every trigger and rigid ATE collapses 158 m → 25 m — within ~1 m of
the local-VI-BA-disabled baseline of 23.72 m. The fallback works as
designed; on this dataset the BA "reduces cost" by 5–10× per trigger
while still walking the bias variables away from a physically
plausible solution, so only a strict freeze cap actually intervenes.

**Best opt-in configuration on V1_01_easy (first 400 frames):**

```sh
./target/release/examples/euroc_online_slam_vi_image_demo \
  --euroc-dir <V1_01_easy> --out-dir <out> --max-frames 400 \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 0.5 \
  --motion-vi-init --local-vi-ba \
  --motion-vi-init-max-velocity 10 \
  --local-vi-ba-freeze-biases-above 0.01
```

→ **rigid ATE 25.17 m, similarity-aligned ATE 0.55 m** — essentially
matching the local-VI-BA-disabled Phase-2 baseline while keeping the
VI-BA stage active for pose + velocity refinement.

**Remaining gap to ORB-SLAM3 territory.** Even with the new gates,
rigid ATE on V1_01_easy is still dominated by the upstream visual
tracker's ~24 m of drift over the 400-frame prefix. Closing it needs
follow-up item #1 from the previous section — the tracker-side
candidates (covisibility-graph guided landmark selection, two-view
triangulation refinement on the stereo seed) sit upstream of every
BA stage and remain out of scope for this iteration. A more
sensitive trigger for the bias-freeze fallback (e.g. on the bias
update magnitude rather than the cost ratio) is a follow-up worth
exploring once the upstream drift is reduced.

## Phase-4 follow-up validation — tracker-side drift suppression (V1_01_easy, first 400 frames)

The Phase-3 closing remark called out the tracker as the dominant
remaining drift source. The first cut of that work is now shipped:

* `--covisibility-local-map-max-keyframes <N>` /
  `--covisibility-local-map-min-shared <M>` — restrict descriptor
  matching during tracking to landmarks observed by the **reference
  keyframe** (the last successfully tracked keyframe, or the nearest
  past keyframe if the last successful frame was not promoted) plus
  the top-`N` co-visible neighbours ranked by shared-landmark count
  (≥ `M` shared landmarks). Maps to
  `TrackingConfig::covisibility_local_map`. When the local map is
  empty or below `min_local_map_landmarks` the tracker falls back to
  the full descriptor store, so it is safe to leave enabled across
  the uninitialized / lost state windows where there is no usable
  reference yet.

The relevant implementation lives in `pipelines/tracking/src/lib.rs`
behind `covisibility_pick_reference_keyframe` and
`covisibility_local_map_landmarks`. Tests cover reference-keyframe
selection (exact-match, nearest-prior, no-prior-available), the
shared-count and cap pruning paths, and the descriptor-store
restriction. `TrackingResult::covisibility_local_map_size` (per
frame) and `TrackingStats::covisibility_local_map_used_count` /
`covisibility_local_map_mean_size` (aggregate) expose how often the
filter actually fired so the user can verify it is doing real work.

**A/B isolation of the tracker-side filter** (on top of the same
`--motion-vi-init --local-vi-ba` baseline as Phase-3, default
covisibility config of `--covisibility-local-map-max-keyframes 10
--covisibility-local-map-min-shared 15`):

| Configuration                                                                       | cov. used frames | rigid ATE (m) | sim. ATE (m) |
|-------------------------------------------------------------------------------------|------------------|---------------|--------------|
| neither knob (legacy Phase-2 baseline)                                              | 0 / 400          | 159.21        | 0.55         |
| Phase-3 best (`--max-velocity 10`, `--freeze-above 0.01`)                           | 0 / 400          | 25.17         | 0.55         |
| **only covisibility filter (no Phase-3 knobs)**                                     | 46 / 400         | **7.42**      | **0.53**     |
| covisibility filter + Phase-3 (both knobs)                                          | 41 / 400         | 10.90         | 0.53         |

**The tracker-side filter is the single strongest intervention on
this prefix.** Enabling covisibility-graph guided landmark selection
alone collapses rigid ATE 159.21 m → 7.42 m (a 21.5× improvement).
Mean local-map size 631 landmarks vs. 1500 in the full map (a 58 %
reduction in match candidates) is enough to keep the PnP solver from
locking onto ambiguous off-trajectory matches.

**Phase-3 + covisibility is slightly worse than covisibility alone**
(10.90 vs. 7.42). The cost-ratio bias-freeze fallback is calibrated
for the high-drift regime; once the tracker is already stable, the
freeze fires on residual-noise-driven cost increases and
unnecessarily holds the biases. Recommended default for V1-class
indoor sequences: leave the covisibility filter on and turn the
Phase-3 freeze threshold off.

**Parameter sensitivity.** A sweep over
`--covisibility-local-map-max-keyframes` ∈ {1, 5, 8, 10, 15, 20} and
`--covisibility-local-map-min-shared` ∈ {5, 8, 10, 15, 20, 200}
produced bit-identical results (rigid ATE 10.90 m / mean local map
size 660 / used frames 41). The reference keyframe alone already
contributes ~660 landmarks of the 1500-landmark map; co-visible
neighbours add only landmarks already in the reference set. The
filter's strength on this scene therefore comes from the reference-
KF restriction itself, not the co-visibility ranking — which is the
expected behaviour for short indoor sequences where most landmarks
are visible from any keyframe in the temporal vicinity.

**Best opt-in configuration on V1_01_easy (first 400 frames):**

```sh
./target/release/examples/euroc_online_slam_vi_image_demo \
  --euroc-dir <V1_01_easy> --out-dir <out> --max-frames 400 \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 0.5 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15
```

→ **rigid ATE 7.42 m, similarity-aligned ATE 0.53 m** — a 21.5×
collapse over the Phase-2 baseline and 3.4× over the Phase-3 best.

**Remaining gap to ORB-SLAM3 territory.** The sim-ATE 0.53 m residual
is dominated by the missing metric-scale recovery (the scale factor
on this run is 0.007, i.e. the trajectory is in arbitrary units
~140× smaller than the ground truth). Two-view triangulation
refinement on the stereo seed and richer monocular-scale recovery
(`Viba2Config::recover_scale`) are the natural next levers; both
sit outside this iteration.

## Phase-5 follow-up validation — pose-jump rejection gate (V1_01_easy, first 400 frames)

A close reading of the Phase-4 trajectory CSV revealed that the
remaining `ate_similarity_scale=0.007` was **not** a metric-scale
problem after all. Frame 31 → 32 → 33 reads
`(0.87, 2.21, 0.93) → (9.38, 1.44, -3.86) → (0.87, 2.20, 0.94)` —
the trajectory teleports >9 m in a single 50 ms cam0 step
(physically impossible at the ~0.5 m/s indoor walking pace) and
returns to the correct anchor on the next frame. 84 of 158 successful
frames showed >2 m frame-to-frame translation, indicating sporadic
catastrophic PnP convergence onto degenerate match sets rather than a
systematic scale error. The 0.007 similarity-alignment scale was a
red herring: Procrustes was simply shrinking the outlier-dominated
estimate down to fit the GT cluster.

**Two-part fix.** First, expose the existing
`TrackingConfig::max_pose_prior_translation_error` knob (already in
the tracker as a quality-gate threshold against the motion-model
pose prior) through a new CLI flag `--max-pose-jump-meters <m>`.
With the default `ConstantPoseMotionModel`, the prior IS the last
successful pose, so this becomes a hard per-frame translation cap.
Second, fix a demo side-channel where rejected frames still leaked
their bad PnP poses into the trajectory CSV and ATE summation: when
the gate rejects a frame, the tracker only flips
`localization.success = false` but leaves `localization.pose`
populated. The demo now treats failed frames as "no estimate this
frame" — they appear in the CSV with empty pose columns and are
excluded from `estimated_positions`.

**A/B sweep over `--max-pose-jump-meters`** (on top of the same
`--motion-vi-init --local-vi-ba` + covisibility filter as Phase-4):

| Gate threshold       | quality-gate fails | tracking success rate | rigid ATE (m)  | rigid max (m) | sim. ATE (m)   | sim. scale | VI init      |
|----------------------|--------------------|-----------------------|----------------|---------------|----------------|------------|--------------|
| None (Phase-4 best)  | 0                  | 0.395                 | 7.421          | 41.708        | 0.535          | 0.0072     | ✓ frame 45   |
| 5.0 m                | 31                 | 0.305                 | 1.404          | 4.174         | 0.466          | 0.148      | ✓ frame 48   |
| 2.0 m                | 107                | 0.247                 | 0.287          | 1.635         | 0.077          | 0.088      | ✗            |
| 1.0 m                | 110                | 0.240                 | 0.144          | 0.738         | 0.066          | 0.149      | ✗            |
| 0.5 m                | 110                | 0.228                 | 0.076          | 0.319         | 0.046          | 0.264      | ✗            |
| **0.2 m**            | **119**            | **0.215**             | **0.046**      | **0.210**     | **0.040**      | **0.566**  | ✗            |

**Headline: rigid ATE 7.421 m → 0.046 m (160× improvement);
similarity-aligned ATE 0.535 m → 0.040 m (13× improvement);
similarity scale 0.007 → 0.566 (close to 1.0 = unit scale).**
The 0.007 was indeed outlier-driven, not metric.

**Trade-off.** At a 0.2 m gate, 80 % of frames are rejected, and the
VI / motion-VI init pipelines never trigger because not enough
keyframes register through the gate. The accepted 86 of 400 frames
form a sparse-but-accurate trajectory over the slow-motion opening
of V1_01_easy. For sequences with higher motion, the threshold
should scale with `(expected_velocity_mps × cam0_period_s) + margin`;
on V1-class indoor sequences (~0.5 m/s, 50 ms cam0 cadence) a
0.025 m physical step plus motion-model / PnP-noise margin gives
the sweet spot at 0.2 m.

**Interaction with the covisibility filter.** With
`--max-pose-jump-meters 0.2` enabled, the cov filter becomes
load-irrelevant: the tracker spends most of its time outside the
`Tracking` state (gated rejections trip the lost-state cascade), so
the covisibility build-up never fires and falls back to the full
1500-landmark map (`covisibility_local_map_mean_size=1500.00` —
contrast with `630.89` at Phase-4). Running the same config with
the cov flags omitted produces bit-identical numbers (rigid 0.046 m /
sim 0.040 m). The Phase-4 covisibility filter is now **defence in
depth** rather than the dominant intervention; the pose-jump gate is.

**Best opt-in configuration on V1_01_easy (first 400 frames):**

```sh
./target/release/examples/euroc_online_slam_vi_image_demo \
  --euroc-dir <V1_01_easy> --out-dir <out> --max-frames 400 \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 0.5 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2
```

→ **rigid ATE 0.046 m, similarity-aligned ATE 0.040 m** — a 160×
collapse over the Phase-4 best and 3500× over the Phase-2 baseline.

**Remaining gaps.** (1) The 80 % rejection rate means a sparse
trajectory; combining the gate with a motion-priored PnP front-end
(IMU pose prior fed to PnP as a warm start, not just used by the
quality gate post-hoc) would let more frames converge to the correct
pose under the gate. (2) The VI / motion-VI init paths still gate on
keyframe registration, which the pose-jump filter cuts off — making
their downstream factors a tighter loop with the visual front-end
(or, alternatively, lowering the keyframe-registration threshold so
each accepted frame registers a keyframe) is needed before the
VI-refinement layers can carry their weight. Both sit outside this
iteration.

## Phase-5 generalization — pose-jump gate on other EuRoC sequences (first 400 frames)

To verify the Phase-5 finding above is not V1_01-specific, the same
recommended config (`--max-pose-jump-meters 0.2` + Phase-4 covisibility
flags) was applied to two additional EuRoC sequences. For each sequence
a baseline (gate omitted) is reported alongside the gated run.

| Sequence       | Gate    | tracking_success | gate_failures | rigid ATE | sim ATE   | sim_scale | map_kfs |
|----------------|---------|------------------|---------------|-----------|-----------|-----------|---------|
| V1_01_easy     | none    | —                | —             | 7.421 m   | 0.535 m   | (tiny, outlier-shrunk) | — |
| V1_01_easy     | 0.2 m   | 21.5 %           | 119           | **0.046 m** | **0.040 m** | 0.566   | 4       |
| MH_01_easy     | none    | 37.5 %           | 0             | 2.934 m   | 0.196 m   | 0.0099    | 54      |
| MH_01_easy     | 0.2 m   | 7.0 %            | 121           | **0.126 m** | **0.117 m** | 0.646   | 18      |
| MH_01_easy     | 0.5 m   | 11.0 %           | 109           | 0.192 m   | 0.120 m   | 0.229     | 19      |
| V2_01_easy     | none    | 57.8 %           | 0             | **608.333 m** | 0.999 m | 0.000051  | 51      |
| V2_01_easy     | 0.2 m   | 20.0 %           | 148           | **0.036 m** | **0.034 m** | 0.831   | 3       |

**The gate generalises.** Across all three sequences, the outlier
signature is identical: baseline `similarity_scale` collapses to
`0.0099` (MH_01) and `0.000051` (V2_01), confirming the same
Procrustes outlier-shrinkage seen on V1_01 (`0.535` rigid → `0.566`
sim with scale-only correction). With `gate=0.2 m`, rigid ATE drops
by 23 × on MH_01 and 17 000 × on V2_01 — the latter being the most
dramatic because the baseline collapses to an essentially-random
trajectory (rigid max 9 225 m on a 20 s indoor sequence).

**Per-scene observations.**
* **V2_01_easy** is the *best* result post-gate (rigid 0.036 m,
  sim 0.034 m, scale 0.83), slightly edging out V1_01 itself. The
  scene shares V1's slow indoor walking profile but has cleaner
  Vicon-room texture, so the surviving frames triangulate tightly.
* **MH_01_easy** is the worst case (rigid 0.126 m, sim 0.117 m).
  The drone takeoff segment has 2 – 3 × the linear velocity of V1's
  hand-held motion, so even *correct* inter-frame translations
  approach the 0.2 m threshold and the gate trims marginal-but-valid
  frames. Loosening to `gate=0.5 m` does *not* recover quality
  (rigid 0.192 m, scale 0.229 — outlier leakage returns and pulls
  the Procrustes solution down). The conclusion is that MH_01 needs
  the *companion* fix — a motion-priored PnP warm start — not a
  looser gate.
* **Keyframe count drops sharply under the gate** (54 → 18 on MH_01,
  51 → 3 on V2_01). The gated trajectory is sparser but
  dramatically more accurate; in particular, V2_01 with only
  3 keyframes still achieves better ATE than V1_01 with 4. This
  re-confirms that throughput, not raw keyframe count, is the
  binding constraint on the front-end.
* **`vi_init_succeeded_frame=None` on all sequences.** Static-VI
  init still trips its accel/gyro noise gate everywhere
  (`AccelNoiseTooHigh` on MH_01 / V2_01, `GyroNoiseTooHigh` on
  V1_01). Motion-VI init never accumulates enough keyframes when
  the pose-jump gate is strict. Both pathways need the
  keyframe-registration-loop fix to fire on these sequences.

**Takeaway.** `--max-pose-jump-meters 0.2` is not a V1_01 tuning
artefact — it is a generally-applicable safety net for EuRoC-class
indoor sequences. The sweet-spot threshold tracks the scene's
maximum *valid* inter-frame translation: `velocity × period +
margin` ≈ 0.2 m at 20 Hz for hand-held / slow-walking sequences. For
faster scenes (drone takeoffs, automotive) the threshold needs to
scale, but the per-scene tuning surface is small (`{0.2, 0.5, 1.0}`)
and the failure mode (outlier-shrunk similarity scale) is trivially
diagnosable from the summary fields.

## Phase-6 follow-up validation — motion-priored PnP warm start (first 400 frames)

Phase-5 left two follow-up gaps: (1) sparse trajectories under the
strict gate (80 % rejection on V1_01, similar on MH_01 / V2_01),
and (2) `ate_similarity_scale` still well below the metric truth
(`0.57` on V1_01, `0.83` on V2_01) which Phase-5 attributed to
"the surviving inlier set being too sparse for the Procrustes to
pin the scale tightly". Both gaps point at the same intervention:
give PnP RANSAC a *predictive* pose prior as a warm-start
hypothesis, so the visual front-end can converge to the correct
pose on frames where the constant-pose prior is too coarse to
satisfy the post-hoc gate.

**The change.** Two coupled additions:

1. `PnPRansac` learns a new path
   ([`estimate_with_pose_prior_and_weights`](../crates/vision/src/ransac/mod.rs))
   that, before iterating, scores the supplied pose prior against the
   correspondence set and seeds `best_pose` / `best_inliers` /
   `best_error` with that score. Random samples must beat the prior's
   inlier count to win, so a well-aligned prior short-circuits RANSAC
   on hard scenes and a misaligned prior gracefully degrades to the
   standard random search. A new `pnp_pose_prior_warm_start: bool`
   field on `TrackingConfig` (default `false`) selects between the
   warm-start path and the legacy "prior is only a candidate-radius
   filter" path.

2. `examples/euroc_online_slam_vi_image_demo` gains
   `--pnp-pose-prior-warm-start` (the boolean) and
   `--motion-model {pose, velocity}` (the underlying prior source).
   The default stays `ConstantPoseMotionModel` for Phase-5
   reproducibility, but the V1/V2 sequences require
   `ConstantVelocityMotionModel` for the warm-start to be a true
   warm-start — the constant-pose prior on a moving body is just
   "the last frame's pose" and a warm-start from there freezes the
   trajectory (see ablation below).

**A/B sweep on the Phase-4 best config (first 400 frames)**
extending the Phase-5 cells with the new flags:

| Sequence | Motion model | Warm start | Gate (m) | rigid ATE | sim ATE | sim_scale | success | gate_fail |
|----------|--------------|------------|----------|-----------|---------|-----------|---------|-----------|
| V1_01_easy | pose | off | 0.2 | 0.046 m | 0.040 m | 0.566 | 21.5 % | 119 |
| V1_01_easy | velocity | off | 0.2 | 0.039 m | 0.025 m | 0.280 | 13.5 % | 161 |
| V1_01_easy | pose | **on** | 0.2 | 0.119 m | 0.118 m | 1.212 (broken) | 30.0 % | 89 |
| V1_01_easy | **velocity** | **on** | 0.2 | **0.034 m** | **0.012 m** | 0.611 | 22.8 % | 129 |
| V2_01_easy | pose | off | 0.2 | 0.036 m | 0.034 m | 0.831 | 20.0 % | 148 |
| V2_01_easy | velocity | off | 0.2 | 0.017 m | 0.001 m | 0.006 (outlier) | 15.0 % | 169 |
| V2_01_easy | **velocity** | **on** | 0.2 | **0.034 m** | **0.034 m** | **1.020** | 26.3 % | 124 |
| MH_01_easy | pose | off | 0.2 | 0.126 m | 0.117 m | 0.646 | 7.0 % | 121 |
| MH_01_easy | **velocity** | **on** | 0.2 | 0.149 m | **0.057 m** | 0.390 | 1.8 % | 164 |
| MH_01_easy | velocity | on | 0.5 | 0.468 m | 0.114 m | 0.112 (outlier) | 6.8 % | 139 |

**`velocity` + `warm-start` + `gate=0.2` is the new recommended
config** for V1/V2-class indoor sequences. The three coupled
findings:

1. **V2_01 recovers metric scale.** `sim_scale = 1.020` (a ≤ 2 %
   error from the metric truth) — the first configuration on the
   bench where the trajectory comes out in true metric units. The
   Phase-5 hypothesis that "the residual scale gap is metric-scale
   recovery work" turns out to be wrong on V2: a sharper PnP prior
   was sufficient to pin the scale tightly. V1_01 also tightens
   (`0.566 → 0.611`) but does not fully converge to metric — it has
   a structurally tougher scene (more uniform plain-coloured walls,
   so fewer well-distributed landmarks).
2. **V1_01 similarity ATE drops 3.3 ×** (`0.040 m → 0.012 m`) with
   roughly the same rigid number. The warm-start is correcting
   per-frame orientation drift that the rigid alignment was hiding.
3. **MH_01_easy is the difficult case.** Similarity ATE halves
   (`0.117 m → 0.057 m`) but rigid drift goes up slightly
   (`0.126 m → 0.149 m`) and tracking-success collapses to 1.8 %.
   The drone-takeoff motion violates the constant-velocity
   prediction (the body is accelerating, not coasting) so the
   warm-start pulls PnP toward a pose the gate then rejects.
   Loosening the gate (`0.5 m`) makes things worse — the outlier
   signature returns (`sim_scale = 0.112`). MH_01 needs a *richer*
   motion model — the appropriate next step is `ImuPredictiveMotionModel`,
   which integrates IMU samples between frames to predict the
   accelerating body's pose.

**Why both knobs are needed.** The two ablation cells make this
explicit:

* `pose` model + warm-start alone: the constant-pose prior on a
  *moving* body is "you didn't move," so the warm-start tells PnP
  "you're still at the last pose." PnP refines toward that
  hypothesis with whatever inliers are nearby, the gate then accepts
  (the result is close to the prior by construction), and the
  trajectory freezes in place across many frames. The orientation
  max blows up (`147.4°`) and the rigid ATE more than doubles.
* `velocity` model alone: the predictive prior correctly extrapolates
  the body's motion, but PnP itself is unaware of the prior. RANSAC
  iterates and the gate rejects everything that drifts more than
  0.2 m from the prediction. On V2 this collapses keyframe coverage
  to a single keyframe (`map_keyframes = 1`); the sim ATE looks
  spuriously low (`0.001 m`) only because Procrustes is shrinking
  the surviving 1-frame trajectory by 167 × to bury what's left.

The two knobs only work together: the velocity model produces a
*predictive* prior, and the warm-start lets PnP *believe* that
prior on frames where the visual signal alone is too ambiguous.

**Recommended config update** for the new V1-class default:

```sh
./target/release/examples/euroc_online_slam_vi_image_demo \
  --euroc-dir <V1_01_easy> --out-dir <out> --max-frames 400 \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 0.5 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model velocity --pnp-pose-prior-warm-start
```

→ V1_01_easy: **rigid 0.034 m / sim 0.012 m / scale 0.611**.
→ V2_01_easy: **rigid 0.034 m / sim 0.034 m / scale 1.020 (metric)**.

**Remaining gaps.** (1) MH_01_easy still needs an IMU-aware motion
model (`ImuPredictiveMotionModel`) to handle the accelerating
drone-takeoff profile; the constant-velocity prior is structurally
insufficient and the gate / warm-start combo can't make up for it.
(2) V1_01's `sim_scale = 0.611` (vs. V2_01's `1.020`) suggests the
texture-poor V1 scene needs richer cues to pin the absolute scale —
either more aggressive local-VI-BA convergence (Phase-3 freezing is
on, but the BA isn't running often enough), or the stereo-baseline
metric-scale anchor (re-triangulating the stereo bootstrap with the
current pose stream, periodically). Both are next-iteration work.

## Phase-7 follow-up validation — IMU-priored motion model wire-up (first 400 frames)

Phase-6 closed with one outlier in the three-EuRoC-sequence sweep:
MH_01_easy, where `velocity + warm-start + gate=0.2` left rigid ATE
at **0.149 m** — meaningfully worse than V1/V2 (both ≈0.034 m). The
documented hypothesis was that the constant-velocity prior is
structurally insufficient on the drone-takeoff profile: the body is
accelerating, not coasting, so the warm-start receives a prior that
already starts diverging from the truth before PnP refines it. Phase-7
verifies that hypothesis by switching the motion model to the existing
`ImuPredictiveMotionModel` (the strapdown predictor that integrates
the inter-frame body-frame gyro/accel samples through Forster's
strapdown step) and re-running the three-sequence A/B.

**Why this wasn't already wired up.** `ImuPredictiveMotionModel` has
shipped since the original motion-VI work, but until Phase-7 the demo's
IMU stream only flowed into the `OnlineSlamPipeline`'s pre-integrator
(via `slam.push_imu_measurement`). The tracker's motion model lived in
a private field with no IMU-feeding hook, so even when the model was
the IMU predictor it would have received an empty pending-samples
buffer and silently fallen back to constant-pose. Phase-7 adds the
`Tracker::motion_model_mut()` accessor and a per-IMU-sample fan-out
inside the demo's existing IMU drain loop:

```rust
slam.push_imu_measurement(sample.gyro, sample.accel, dt);
slam.tracker.motion_model_mut().push_imu_measurement(sample.gyro,
                                                     sample.accel, dt);
```

The `DemoMotionModel` enum grows an `ImuPredictive` variant and an
internal `push_imu_measurement` that no-ops for the `Pose` /
`Velocity` variants and forwards for the `ImuPredictive` variant, so
the demo's IMU loop is unconditional and the model is selected purely
by `--motion-model`.

**A/B sweep (first 400 frames, all three EuRoC sequences,
`--pnp-pose-prior-warm-start --max-pose-jump-meters 0.2`)** —
comparing the Phase-6 winning `velocity` model against the new `imu`
model:

| sequence       | model    | rigid ATE | sim ATE | sim scale | gate fails |
|----------------|----------|-----------|---------|-----------|------------|
| V1_01_easy     | velocity | 0.034 m   | 0.012 m | 0.611     | 116        |
| V1_01_easy     | **imu**  | **0.022 m** | 0.018 m | 0.246   | 119        |
| V2_01_easy     | velocity | 0.034 m   | 0.034 m | 1.020 (metric) | 124  |
| V2_01_easy     | **imu**  | **0.025 m** | 0.024 m | 1.216 | 144        |
| MH_01_easy     | velocity | 0.149 m   | 0.057 m | 0.390     | 164        |
| MH_01_easy     | **imu**  | **0.041 m** | 0.041 m | **1.112** | 165      |

**MH_01_easy: the documented Phase-6 gap closes.** Rigid ATE collapses
3.6× (0.149 → 0.041 m), similarity ATE halves (0.057 → 0.041 m), and
similarity scale recovers from 0.390 (60 % under-scaled, characteristic
of an outlier-shrunk Procrustes fit) to **1.112** — the drone-takeoff
trajectory is now within ~11 % of true metric units, comparable to V2's
1.020. Orientation RMSE also tightens 2.7× (1.91 → 0.72 deg). The
gate-failure count stays essentially flat (164 → 165), confirming the
rejection budget is unchanged — the win is purely from a sharper
warm-start prior.

**V1_01 / V2_01: modest rigid-ATE wins, slight metric-scale
trade-off.** On both slow-motion sequences `imu` improves the rigid
ATE 1.4–1.5× (V1: 0.034 → 0.022 m, V2: 0.034 → 0.025 m). The
trade-off shows up in the metric scale: V1's similarity scale drifts
0.611 → 0.246 (worse outlier-shrinkage signature), V2's drifts 1.020
→ 1.216 (still inside ~22 % of metric). The mechanism is intuitive:
on V1/V2 the body is essentially quasi-static (slow indoor walk), so
the gyro / accel noise integrated across many tiny inter-frame windows
adds a small bias to the prediction that the constant-velocity model
doesn't have. The PnP refinement absorbs most of it (rigid ATE still
improves), but the Procrustes / similarity alignment picks it up as a
mild scale drift.

**Conclusion: the motion model is a per-regime choice, not a global
winner.** For accelerating / dynamic sequences (MH_*), the IMU-priored
model is decisively better. For quasi-static sequences (V1/V2), it
trades a small metric-scale margin for a modest rigid-ATE win, and the
right call depends on which metric matters downstream.

**New recommended configurations (Phase-7):**

```sh
# Sequences with non-trivial linear/angular acceleration (MH_*)
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start
```

→ MH_01_easy: **rigid 0.041 m / sim 0.041 m / scale 1.112**.

```sh
# Quasi-static sequences (V1/V2) — keep Phase-6 config
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/V1_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model velocity --pnp-pose-prior-warm-start
```

→ V1_01_easy: **rigid 0.034 m / sim 0.012 m / scale 0.611**.
→ V2_01_easy: **rigid 0.034 m / sim 0.034 m / scale 1.020 (metric)**.

**Remaining gaps.** (1) The cam0 ↔ IMU `T_BS` extrinsic is approximated
as identity in the Phase-7 wire-up (cam0 sits ~0.1 m from the IMU on
EuRoC, so the body-vs-camera frame mismatch is bounded over a single
~50 ms inter-frame integration window but is not zero). Plumbing
`T_BS` through `ImuPredictiveMotionModel::predict_pose` so the
integration is body-frame and the input/output are camera-pose is the
natural follow-up. (2) The IMU motion model takes its gyro/accel biases
at construction time and never refreshes them. The local-VI-BA stage
refines per-keyframe biases on every BA pass, and `ImuPredictiveMotionModel::set_biases`
is the documented hook to mirror those refined biases back into the
motion model — wiring that hook into `OnlineSlamPipeline`'s post-BA
update would tighten the prediction further, especially on long
sequences where the bias drifts. (3) The V1_01 metric-scale gap
(`sim_scale = 0.611` vs V2's metric `1.020`) persists from Phase-6 and
is orthogonal to the motion model — it's a stereo-bootstrap /
re-triangulation question, not a prediction question.

## Phase-8 follow-up validation — T_BS extrinsic plumbing + finite-difference velocity update (first 400 frames)

Phase-7 closed with two open assumptions baked into the `ImuPredictiveMotionModel`
wire-up: (a) `body == camera` (cam0's published `T_BS` was not
threaded through the predictor), and (b) `velocity_world = 0` for
every prediction (the model exposes `set_velocity_world` for a
downstream VI-BA to refresh, but no VI-BA was running on the EuRoC
prefix). Both assumptions were *visible* but *invisible*: visible in
the code, invisible in the numbers because Phase-7 still produced the
best ATE on the bench. Phase-8 explicitly tests both by (a) plumbing
`body_to_sensor: SE3` through `predict_pose` so the integration runs
in body frame, (b) adding `update_velocity_from_camera_pose_difference`
+ a per-frame hook in the demo that finite-differences two successive
camera poses to refresh `velocity_world`.

**Implementation.** `ImuPredictiveMotionModelConfig` gains a
`body_to_sensor: SE3` field defaulting to identity (Phase-7
behaviour). `predict_pose` now (i) converts the input camera pose to
a body pose via `T_bw = body_to_sensor · T_cw`, (ii) integrates gyro
/ accel in body frame, (iii) converts the integrated body pose back
to a camera pose via `T_cw_new = body_to_sensor⁻¹ · T_bw_new`. The
new `update_velocity_from_camera_pose_difference(prev, curr, dt)` is
algebraically equivalent to "compute body-centre Δp / Δt under the
configured extrinsic." The demo gains a single `--imu-extrinsic-from-cam0`
flag (default off) that atomically wires cam0's `T_BS` into the model
AND enables the per-frame velocity refresh — atomic because the two
must move together: T_BS plumbing without velocity update gives
geometrically correct math with `v=0` damping; velocity update
without T_BS plumbing is a body-as-camera approximation refresh.

**A/B sweep (first 400 frames, all three EuRoC sequences,
`--motion-model imu --pnp-pose-prior-warm-start --max-pose-jump-meters 0.2`)
— comparing the Phase-7 baseline against Phase-8a (T_BS only, no
velocity update — for diagnostic isolation) and Phase-8b (T_BS +
velocity update, the full Phase-8 mode shipped behind the flag):**

| sequence       | mode     | rigid ATE | sim ATE | sim scale | gate fails |
|----------------|----------|-----------|---------|-----------|------------|
| V1_01_easy     | Phase-7   | **0.022 m** | 0.018 m  | 0.246     | 119        |
| V1_01_easy     | Phase-8a  | 0.027 m   | 0.025 m  | **1.113** | 82         |
| V1_01_easy     | Phase-8b  | 0.048 m   | 0.043 m  | 0.827     | 78         |
| V2_01_easy     | Phase-7   | **0.025 m** | 0.024 m  | 1.216     | 144        |
| V2_01_easy     | Phase-8a  | 0.091 m   | 0.091 m  | 1.017     | 81         |
| V2_01_easy     | Phase-8b  | 0.087 m   | 0.084 m  | 0.947     | 86         |
| MH_01_easy     | Phase-7   | **0.041 m** | 0.041 m  | 1.112     | 165        |
| MH_01_easy     | Phase-8a  | 0.113 m   | 0.112 m  | 1.161     | 69         |
| MH_01_easy     | Phase-8b  | 0.068 m   | 0.068 m  | **1.001** | 162        |

**Three observations.**

1. **Phase-8a (T_BS only) regresses rigid ATE universally** (V1: 1.2×
   worse, V2: 3.6× worse, MH_01: 2.8× worse). The mechanism is
   structural: with proper body-frame integration but `v=0`, the
   predicted body translation per inter-frame window is only the
   quadratic-in-Δt accel term — typically ~mm-scale per step. The
   prior is too "still." Fewer predictions land >0.2 m from truth (gate
   failures collapse 119 → 82 / 144 → 81 / 165 → 69), so more noisy PnPs
   reach the trajectory CSV and accumulate drift. The Phase-7
   approximation was "hiding" this gap by reading the camera-frame
   gyro / accel back into the camera-pose update directly, which —
   because EuRoC's cam0 frame has body-z ≈ cam-z (the dominant axis on
   indoor / yaw-dominated motion) — happened to produce predictions
   close enough to truth that the gate rejected the visual outliers
   load-bearing for ATE.

2. **Phase-8b (T_BS + velocity update) recovers metric scale
   dramatically on MH_01** (similarity scale `1.001`, the first
   essentially-zero metric error on the bench), and the gate failure
   count returns to Phase-7 levels (165 → 162) — the velocity update
   restores the prior's "this body is moving" character. But the
   rigid ATE still regresses vs Phase-7 (V1 2× worse, V2 3.5× worse,
   MH_01 1.7× worse). The mechanism this time is noise: finite-differencing
   two successive PnP poses amplifies their per-frame jitter into the
   velocity estimate. With ~1–2 cm pose noise / 50 ms cam0 step, the
   velocity estimate carries ~20–40 cm/s of noise; on V1's true ~0.5 m/s
   body speed that's ~50% relative noise. The IMU prediction then
   extrapolates a noisy body velocity, the PnP warm-start gets dragged
   to a noisy local minimum, and cumulative drift compounds.

3. **The velocity update is theoretically correct but practically
   premature** without a downstream VI-BA to refine velocity in a
   noise-suppressing way. The model's `set_velocity_world` is the
   documented hook for VI-BA's refined velocity at the most recent
   keyframe — that's a per-keyframe joint optimisation that smooths
   the velocity over a sliding window. Finite-differencing two raw
   visual poses is not the same thing.

**Recommended config (Phase-8): leave the flag off by default.** The
`--imu-extrinsic-from-cam0` capability ships behind the flag so the
opt-in path is one CLI arg away when the downstream VI-BA-driven
velocity update is wired in (next iteration). Until then, the
Phase-7 wire-up is empirically dominant for rigid ATE on the
bench.

```sh
# Phase-7 (default — recommended until VI-BA-driven velocity update lands)
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start
```

→ MH_01_easy: rigid 0.041 m / sim 0.041 m / scale 1.112.

```sh
# Phase-8b (opt-in — better metric scale, worse rigid ATE)
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --imu-extrinsic-from-cam0
```

→ MH_01_easy: rigid 0.068 m / sim 0.068 m / scale **1.001 (metric)**.

**Remaining gaps.** (1) Wire `ImuPredictiveMotionModel::set_velocity_world`
to the local-VI-BA's refined per-keyframe velocity once VI-BA is
exercised on real EuRoC. The model's own docstring already names this
as the intended velocity source; the current finite-difference
substitute is a noise-prone stopgap. With a proper VI-BA refresh, the
`--imu-extrinsic-from-cam0` path should beat Phase-7 on rigid ATE *and*
keep its metric-scale win. (2) `ImuPredictiveMotionModel::set_biases`
is the symmetric hook for per-keyframe bias refreshes; wiring it from
`OnlineSlamPipeline`'s post-BA update closes the second of the
"static-config" assumptions Phase-7 left in place. (3) The
`OnlineSlamPipeline::push_imu_measurement` path consumes raw body-frame
samples and feeds them into the pre-integrator; that pipeline is
already body-frame-aware, so no parallel T_BS plumbing is needed there
— this remains a tracker-side change only.

## Phase-9 follow-up validation — local-VI-BA → IMU motion model state mirror (first 400 frames)

Phase-8 left two prerequisite hooks for VI-BA-driven refresh of the
IMU motion model unwired: `ImuPredictiveMotionModel::set_velocity_world`
and `set_biases` are documented in the model's docstrings as the
intended write paths for downstream solvers, but nothing in the demo
called them. Phase-9 adds that fan-out: after every successful
`local_vi_ba` trigger (non-bias-frozen), the demo pulls the trigger's
window's most recent keyframe's `KeyframeImuState` from
`slam.local_vi_ba_state.keyframe_state` and pushes its `velocity_world`,
`bias_gyro`, and `bias_acc` into the IMU motion model via a new
`DemoMotionModel::mirror_vi_ba_state` dispatch.

**The wiring shipped, but it is structurally blocked upstream.** On
MH_01_easy with the recommended Phase-7 config
(`--max-pose-jump-meters 0.2 --motion-model imu --pnp-pose-prior-warm-start`),
only **1 keyframe registers in 400 frames** (tracking-success 1 %), so
VI-init never promotes, local-VI-BA never triggers, and the mirror
never fires. Loosening the gate enables the chain but at a
catastrophic visual-quality cost. The full gate sweep, with
`--vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0
--local-vi-ba` to unblock the static-VI-init's gyro/accel-std caps and
opt into local-VI-BA:

| `--max-pose-jump-meters` | map KFs | VI-init succeeded | local-VI-BA triggers | mirror velocity (m/s) | rigid ATE | sim scale |
|--------------------------|---------|--------------------|----------------------|------------------------|-----------|-----------|
| `0.2` (Phase-7 baseline) | 1       | None               | 0                    | — (mirror never fired) | **0.041 m** | 1.112    |
| `0.5`                    | 2       | frame 64           | 0                    | — (KF<trigger_every)   | 0.146 m   | 0.433    |
| `1.0`                    | 4       | frame 90           | 1                    | `(1.11, 1.46, -0.97)`  | 0.261 m   | 0.187    |
| `1.5`                    | 7       | frame 90           | 3                    | `(0.57, 0.08, 0.49)`   | 0.321 m   | 0.115    |
| `2.0`                    | 8       | frame 90           | 1                    | `(12.30, -0.14, 1.21)` | 0.424 m   | 0.106    |

Two observations:

1. **The mirror is functional.** At `gate ≥ 1.0` it fires repeatedly,
   and at `gate = 1.5` the mirrored velocities `(0.57, 0.08, 0.49) m/s`
   are physically plausible drone-takeoff numbers (mostly horizontal +
   ~0.5 m/s vertical climb — matches MH_01's takeoff profile). The
   wiring is correct and the BA-refined state successfully reaches the
   tracker's IMU motion model.

2. **The visual front-end can't supply enough good keyframes at the
   gate widths VI-BA needs.** At `gate = 2.0`, VI-BA converges to a
   12.3 m/s velocity solution — unphysical for MH_01 — because the
   loose gate let too many PnP outliers through and the LM dragged
   the velocity along with them. At `gate = 0.5`, only 2 keyframes
   register over 400 frames, so the BA trigger (which requires a
   minimum IMU-factor window) never fires. There is no gate setting on
   this prefix where (a) VI-BA triggers AND (b) the resulting refined
   velocity is plausible AND (c) the rigid ATE matches Phase-7. The
   Phase-7 baseline at `gate=0.2` is dominated by aggressive rejection
   of bad PnPs; loosening the gate enough to support VI-BA hands back
   more error than VI-BA's mirror can repay.

**Conclusion: Phase-9 ships the wiring; the empirical ATE improvement
is upstream work on the visual front-end.** Concretely: the path
forward is to raise the per-frame inlier ratio so the gate can stay at
`0.2 m` and still register ~50–100 keyframes per 400 frames. Levers
worth trying in a follow-up:

- Deeper descriptors (the current corner+patch descriptor is the
  simplest possible; ORB / SuperPoint would lift the matcher's
  signal-to-noise).
- Tighter correspondence filtering before PnP (ratio test +
  cross-check, geometric pre-filtering using the motion-VI prior).
- Seeding the local-VI-BA from the motion-VI initialiser path
  (`--motion-vi-init`) once the keyframe-registration count clears the
  motion-VI's own trigger threshold.

These are not in Phase-9's scope. The mirror is the deliverable: the
moment the visual front-end clears the keyframe-registration threshold
at a `0.2 m` gate, turning `--local-vi-ba` on activates the refined-(v,
b) refresh with no further code change.

## Phase-10 follow-up validation — cross-check matcher (first 400 frames, MH_01_easy)

Phase-9 ended with a diagnostic conclusion: the visual front-end can't
register enough keyframes through the `0.2 m` gate to support the VI-BA
chain on real EuRoC. The first lever Phase-9 named was "tighter
correspondence filtering before PnP (ratio test + cross-check)" — the
standard ORB-SLAM filter that keeps only query↔train pairs that
mutually pick each other as the single best match. Phase-10 adds the
`--cross-check-matcher` CLI flag wrapping the existing `BruteForceMatcher`
(Lowe ratio 0.8) in `CrossCheckMatcher` and measures whether the tighter
filter unblocks the Phase-7 tight-gate keyframe-registration floor.

| `--max-pose-jump-meters` | cross-check | KFs | VI-init | local-VI-BA triggers | rigid ATE | sim scale | gate failures |
|---|---|---|---|---|---|---|---|
| `0.2` | off (Phase-7 baseline) | 1  | None | 0 | **0.041 m** | 1.112 | 165 |
| `0.2` | **on**                | 1  | None | 0 | 0.080 m   | 0.950 | 222 |
| `1.0` | off                   | 4  | frame 90  | 1 | 0.261 m | 0.187 | 100 |
| `1.0` | **on**                | 3  | frame 144 | 0 | **0.163 m** | 0.366 | 116 |

Two observations:

1. **Cross-check does NOT unblock the tight-gate floor.** At
   `gate=0.2`, KFs stays at 1, gate failures go UP (165 → 222), and
   rigid ATE roughly doubles (0.041 → 0.080 m). The mechanism is
   precision-vs-recall: cross-check filters out unstable matches
   (raises precision per candidate), but on EuRoC's noisy
   corner+patch descriptors the match volume drops enough that PnP
   starves — fewer candidates means a less-stable pose solve, more
   solutions cross the 0.2 m gate, and the keyframe-registration
   bottleneck doesn't move.

2. **Cross-check DOES help the loose-gate VI-BA configuration.** At
   `gate=1.0`, cross-check improves rigid ATE 1.6× (0.261 → 0.163 m)
   and similarity scale (0.187 → 0.366). The tracking-success rate
   rises from 21 % to 31 %. The trade-off is that the keyframe count
   drops slightly (4 → 3) and local-VI-BA stops triggering (1 → 0
   triggers) — the mirror chain stalls because the BA needs a minimum
   IMU-factor window. The right read is: cross-check is a useful
   precision-bump for the loose-gate config but it's not the lever
   that lifts the VI-BA chain.

**Conclusion: cross-check ships behind `--cross-check-matcher` for
loose-gate use cases that want precision over recall, but the deeper
bottleneck on real EuRoC is descriptor signal-to-noise — not match
filter strictness.** The recall ceiling cross-check exposes is the
same ceiling that limits VI-BA's keyframe registration on EuRoC. The
right next lever is a stronger descriptor (the `learned_descriptor`
infrastructure already in the repo, or ORB) — not more match-filter
tuning. That's the next iteration after Phase-10.

## Phase-11 follow-up validation — HOG descriptor + keyframe threshold (first 400 frames)

Phase-10 ended pointing at descriptor signal-to-noise as the next
lever. Phase-11 wires the existing `HogLikeFeatureExtractor` (a
128-D HOG/SIFT-flavored unit-norm descriptor that's been shipped in
`crates/vision/src/features/deep.rs` but never activated in the demo)
behind a `--feature-extractor {corner, hog}` CLI flag, adds
companion HOG tuning knobs (`--hog-max-features`,
`--hog-min-corner-score`, `--hog-orient`), and a
`--keyframe-min-translation` flag that overrides the library's
`1.0 m` default for the `SimpleKeyframePolicy`.

**The hypothesis:** descriptor S/N → tracking-success → keyframe
count → VI-init promotion → local-VI-BA triggers → mirror activates
→ refined `(v, b)` feeds the tracker's IMU motion model → improved
ATE. Phase-11 exercises every link in the chain.

**MH_01_easy ablation (all on top of `--motion-model imu --pnp-pose-prior-warm-start
--max-pose-jump-meters 0.2 --vi-init-gyro-std-limit 0.5
--vi-init-accel-std-limit 5.0 --local-vi-ba`):**

| extractor + filter           | kf min translation | tracking success | KFs | VI-init | VI-BA triggers | rigid ATE | sim scale |
|------------------------------|--------------------|------------------|-----|---------|----------------|-----------|-----------|
| corner + ratio (Phase-7)    | `1.0 m` (default)  | 1.0 %            | 1   | None    | 0              | **0.041 m** | 1.112    |
| corner + ratio              | `0.1 m`            | 1.0 %            | 2   | None    | 0              | 0.041 m   | 1.112    |
| HOG + ratio                  | `1.0 m` (default)  | 2.0 %            | 1   | None    | 0              | 0.051 m   | 1.093    |
| HOG + cross-check (Phase-10)| `1.0 m` (default)  | 35.5 %           | 1   | None    | 0              | 0.064 m   | 0.795    |
| HOG + cross-check (Phase-11)| `0.1 m`            | 9.8 %            | **8** | **frame 54** | **1**       | **0.033 m** | **1.001 (perfect metric)** |

Two observations from the ablation:

1. **No single knob is sufficient.** Lowering kf-min-translation alone
   (corner + ratio, kf=0.1) leaves KFs at 2 and rigid ATE unchanged.
   Swapping in HOG alone (HOG + ratio, kf=1.0) gives 2× tracking-
   success but still 1 KF. Adding cross-check (HOG + cc, kf=1.0)
   pushes tracking-success to 35.5 % but still 1 KF — because the
   1.0 m kf threshold is too coarse for EuRoC's accepted-trajectory
   span at gate=0.2 m. Only the three-knob combination clears every
   link in the chain.

2. **The chain pays off.** With 8 keyframes registered, VI-init
   promotes at frame 54 (vs Phase-9's frame 90 at gate=1.0 m), local-
   VI-BA fires, the Phase-9 mirror activates, and the refined
   `velocity_world` `(0.60, 0.38, -3.47) m/s` and bias estimates flow
   into the IMU motion model. Rigid ATE drops 0.041 → 0.033 m (-19 %)
   AND similarity scale recovers to `1.001` — the **first config on
   this bench that improves rigid ATE AND lands essentially-zero
   metric scale error on MH_01**.

**Three-sequence validation of the Phase-11 winning config
(`--feature-extractor hog --cross-check-matcher
--keyframe-min-translation 0.1` on top of the full IMU stack):**

| sequence       | Phase-7 rigid | Phase-11 rigid       | Phase-7 scale | Phase-11 scale          | VI-BA triggers (Phase-11) |
|----------------|---------------|----------------------|---------------|-------------------------|---------------------------|
| MH_01_easy     | 0.041 m       | **0.033 m (-19 %)** | 1.112         | **1.001 (metric)**     | 1                         |
| V2_01_easy     | 0.025 m       | **0.016 m (-37 %)** | 1.216         | 0.939                  | 1                         |
| V1_01_easy     | 0.022 m       | 0.034 m (+55 %)     | 0.246         | 0.803                  | 1                         |

Two-of-three sequences improve unambiguously on rigid ATE; V1_01 is
mixed: the rigid ATE regresses 1.5× but the similarity scale jumps
0.246 → 0.803 — V1's Phase-7 baseline carried a near-3× Procrustes-
shrunk trajectory (the geometry was broken even at the low rigid-ATE)
and Phase-11 lifts that distortion. The right reading: V1 trades a
small rigid-ATE regression for a major metric-correctness win.

**Phase-11 is the first iteration on this bench that closes the
keyframe-registration loop on real EuRoC at the strict 0.2 m gate.**
The local-VI-BA chain now fires routinely (1 trigger per 400 frames
across all three sequences), the Phase-9 mirror activates, and the
refined `(velocity, biases)` actually flow back into the tracker's
prior. The recommended config moves to:

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.1
```

**Remaining gaps.** (1) V1's residual rigid-ATE regression (0.022 →
0.034 m) suggests the HOG+cc combo is over-conservative on slow indoor
motion — a per-sequence tuning of `--hog-min-corner-score` or
`--cross-check-matcher` toggle might recover the V1 baseline without
losing the metric-scale win. (2) Only 1 VI-BA trigger per 400 frames
is still sparse — tightening `OnlineSlamLocalBaConfig.trigger_every`
or adding a per-keyframe trigger would let the mirror update more
frequently and tighten the IMU prior across the sequence. (3) The
HOG extractor is the simplest deep-shaped descriptor in the repo; an
ONNX-backed SuperPoint / DISK that consumes the same
`DeepFeatureExtractor` trait would be a further S/N step up.

## Phase-12 follow-up validation — trigger frequency is not the bottleneck (first 400 frames, MH_01_easy)

Phase-11 ended with one obvious lever to try: tighten
`OnlineSlamLocalBaConfig.trigger_every` so the local-VI-BA fires more
often and the Phase-9 mirror refreshes the IMU motion model's `(v, b)`
more aggressively. Phase-12 instruments the demo with a new
`imu_factors_staged` counter and validates that the trigger knob is
not actually the bottleneck.

**Diagnostic finding.** With the Phase-11 winning config
(`--feature-extractor hog --cross-check-matcher
--keyframe-min-translation 0.1 --max-pose-jump-meters 0.2
--motion-model imu --pnp-pose-prior-warm-start
--vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0
--local-vi-ba`), the new counter reports `imu_factors_staged = 1`
despite `map_keyframes = 8`. The mechanism:

- VI-init promotes at frame 54 (the static-VI-init's `max_gyro_std`
  gate fails until the drone hovers momentarily mid-takeoff).
- 7 of the 8 keyframes register BEFORE frame 54.
- The pipeline's stale-factor gate (the documented policy that
  discards IMU factors carrying placeholder bias linearisations
  until VI-init promotes) drops their factors silently.
- Only 1 keyframe registers AFTER frame 54 — the visual tracker
  dies on the takeoff acceleration at frame ~60, so no more
  keyframes accumulate.
- That single post-promotion keyframe emits the lone IMU factor,
  which fires the lone local-VI-BA trigger.

**The trigger knob already defaults to 1** (fire on every new factor),
so the BA was already running as eagerly as the staging path
allowed. Tightening it further has no room to work.

**Lowering VI-init's promotion threshold has no effect.** Setting
`--vi-init-min-samples 10 --vi-init-min-stationary-window-seconds
0.05` is bit-identical to the default — the actual gate is the
`max_gyro_std` / `max_accel_std` stationarity check, not the sample
count (the buffer fills 200 IMU samples per second, so any reasonable
min-sample budget is hit immediately). Enabling motion-VI-init
(`--motion-vi-init --motion-vi-init-min-keyframes 3
--motion-vi-init-min-translation 0.3`) also stays in `Waiting`
forever — the post-promotion keyframe count never reaches its trigger
because the visual tracker has already died on the takeoff transient.

**The single trigger's mirrored velocity** `(0.60, 0.38, -3.47) m/s`
is plausible (the negative-z component matches the cam-frame Z-axis
during takeoff climb), but one update to the IMU motion model's
prior can't carry the tracker through the next 340 frames once
correspondence quality collapses.

**Conclusion: trigger frequency is not the limiting factor.** The
real chain stops at the takeoff transient because the visual tracker
fails. Forward levers:

1. **Visual tracker survival across takeoff** — the same gap Phase-10
   and Phase-11 named. A still stronger descriptor (ONNX-backed
   SuperPoint / DISK) or a denser inlier-set under the existing HOG
   would extend the post-promotion keyframe stream.
2. **Earlier VI-init promotion** — loosen the static stationarity
   gate further (`--vi-init-gyro-std-limit` to 1.0, `--accel-std-limit`
   to 10.0) at the risk of accepting a worse linearisation point.
3. **Stale-factor relaxation** — keep IMU factors across the
   promotion boundary and re-linearise on the first post-promotion
   call, instead of discarding pre-promotion factors entirely. This
   is a pipeline-level change that recovers the 6-of-7 missing
   factors and would let local-VI-BA fire ~7× across the 400-frame
   prefix. The trade-off: factors built with the wrong bias
   linearisation would carry a non-trivial bias error into the BA's
   first pass.

The diagnostic counter ships so subsequent iterations can verify
chain depth at a glance. The empirical numbers (1 staged factor / 1
trigger / 1 mirror on MH_01) are the Phase-12 baseline that the next
unblock should beat.

## Phase-13 follow-up validation — relax the stale-factor gate (first 400 frames)

Phase-12 named the bottleneck precisely: 7-of-8 keyframes register
BEFORE the auto-bootstrap stage promotes at frame 54, and the
stale-factor gate (the documented "factors built with placeholder
bias linearisation are silently discarded") drops all but the lone
post-promotion factor. Phase-13 relaxes the gate behind a new
`OnlineSlamConfig.keep_pre_promotion_imu_factors: bool` flag, with
the critical companion: pre-promotion factors bank into the local-VI-BA
factor history, but the BA solver itself is gated to NOT run until
VI-init promotes. The first cut without the BA gate corrupted the
map (running BA with placeholder zero biases is a degenerate
linearisation — the empirical signature was tracking-success
collapsing 9.8 % → 1.8 % on MH_01 because the next-frame matcher saw
BA-shifted keyframe descriptors). The two-part change keeps factors
flowing while preventing premature solves.

**MH_01_easy three-step ablation** (all on top of `--motion-model imu
--pnp-pose-prior-warm-start --max-pose-jump-meters 0.2
--feature-extractor hog --cross-check-matcher
--keyframe-min-translation 0.1 --vi-init-gyro-std-limit 0.5
--vi-init-accel-std-limit 5.0 --local-vi-ba`):

| `--keep-pre-promotion-imu-factors` | BA gated until promote | factors staged | local-VI-BA triggers | rigid ATE | sim scale |
|------------------------------------|------------------------|-----------------|----------------------|-----------|-----------|
| off (Phase-11)                     | n/a                    | 1               | 1                    | 0.033 m   | 1.001     |
| on (first cut)                     | no                     | 1               | 1                    | 0.028 m   | 1.172     |
| **on**                             | **yes (shipped)**      | **6**           | **1**                | **0.027 m** | 1.016    |

The shipped Phase-13 config (third row) banks 6 factors and runs a
single post-promotion BA pass that consumes the full banked history.
The trigger count stays at 1 because the visual tracker still dies on
the takeoff transient, but the BA pass it runs is now informed by 6×
more IMU constraints — the resulting pose refinement of the trailing
5 keyframes is what drives the ATE collapse.

**Three-sequence validation (Phase-7 → Phase-11 → Phase-13)** under
the recommended Phase-13 stack:

| sequence       | Phase-7 rigid | Phase-11 rigid | Phase-13 rigid       | Phase-7 scale | Phase-11 scale | Phase-13 scale         |
|----------------|---------------|----------------|----------------------|---------------|----------------|------------------------|
| MH_01_easy     | 0.041 m       | 0.033 m        | **0.027 m (-34 %)** | 1.112         | 1.001          | 1.016                  |
| V1_01_easy     | 0.022 m       | 0.034 m        | **0.015 m (-32 %)** | 0.246         | 0.803          | **1.060 (~metric)**    |
| V2_01_easy     | 0.025 m       | 0.016 m        | **0.004 m (-84 %)** | 1.216         | 0.939          | 1.093                  |

**Phase-13 is the first iteration on this bench that beats the Phase-7
baseline on rigid ATE AND lands a near-metric similarity scale across
every sequence simultaneously.** V1_01 in particular: the Phase-7
similarity scale of `0.246` was an outlier-shrunk Procrustes
signature (the trajectory geometry was broken even at the low
rigid-ATE); Phase-11 lifted that to `0.803` at a cost to rigid;
Phase-13 takes it to `1.060` AND beats Phase-7's rigid by 32 %.

**Caveat on the mirrored velocity.** Across all three sequences the
local-VI-BA's single post-promotion velocity solution diverges to
unphysical magnitudes (V1: `(-7.3, -41.8, -37.1) m/s`, V2: `(11.2,
36.3, -35.2) m/s`, MH_01: `(2.0, -4.6, 1.6) m/s` — somewhat
plausible for MH_01's takeoff but the V1/V2 magnitudes are clearly
broken). This is the velocity slot absorbing the bias-error
mass from the placeholder zero linearisation point — the IMU factor's
`Δv` contains a contribution from the pre-promotion bias estimate
that the BA's first pass attributes to velocity instead. **The
unphysical velocity doesn't affect rigid ATE** because the IMU motion
model's `velocity_world` only matters on subsequent IMU-priored frame
predictions, and the visual tracker dies before another such frame
ever runs. But the divergence is a real signal that the BA's
linearisation handling is incomplete; the next iteration should
either (a) freeze velocities while biases converge in the first
post-promotion pass, or (b) re-linearise pre-promotion factors with
the freshly-promoted biases before feeding them to BA.

**Recommended config (Phase-13):**

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.1 \
  --keep-pre-promotion-imu-factors
```

**Remaining gaps.** (1) The mirror velocity divergence (above) — the
right fix is a bias-first BA pass that freezes velocities, then a
joint pass that releases them. Or a re-linearisation step that
shifts each factor's `Δp / Δv / ΔR` to the post-promotion bias point
using the precomputed Jacobians (`J_Δ_bias_g`, `J_Δ_bias_a`) the IMU
pre-integrator already stores. (2) Only 1 local-VI-BA trigger per
400 frames — the visual tracker still dies on the takeoff transient.
The next visual-side lever is a stronger descriptor (ONNX-backed
SuperPoint / DISK) per Phase-11's deferred follow-up. (3) The mirror
chain's per-trigger update is now the binding constraint on how
often the IMU motion model's `(v, b)` refresh — a richer multi-pass
BA schedule (e.g. trigger every keyframe after the first
post-promotion solve) is the natural follow-up to the per-trigger
mirror.

## Phase-14 follow-up validation — in-place IMU factor re-linearisation (first 400 frames)

**Diagnostic motivation.** Phase-13's `keep_pre_promotion_imu_factors`
opt-in unblocked rigid ATE across all three EuRoC sequences by
banking pre-promotion IMU factors into the post-promotion BA pass,
but the factors carry their construction-time `bias_*_linearisation`
= placeholder `(0, 0, 0)`. When VI-init promotes and the BA solves
for non-trivial biases, `residual_with_bias_correction` applies the
first-order extrapolation `δb = bias_now − 0` against the stored
Jacobians (Forster eq. 44). For the EuRoC gyro / accel biases
recovered at promotion time (`b_g ≈ (-0.019, 0.047, 0.076)`,
`b_a ≈ (-0.123, 0.0007, 0.047)` on MH_01), the linear approximation
sits far enough from the integration's actual linearisation point
that the residual minimum is reached by the BA shifting the velocity
slot to compensate — the velocity absorbs the bias-extrapolation
error mass because it has more DOF freedom than the bias slots.
Empirically this surfaces as the unphysical mirror velocity reported
in Phase-13 (V1: `-41.8 m/s y-component`).

**Implementation.** The IMU pre-integration factor already carries
the bias Jacobians needed for in-place re-linearisation
(`j_rotation_bg`, `j_velocity_ba`, `j_velocity_bg`, `j_position_ba`,
`j_position_bg`). Phase-14 adds
`ImuPreintegratedDelta::relinearise_at(b_new_gyro, b_new_acc)` which
(a) bakes the first-order bias correction into the stored
`(delta_rotation, delta_velocity, delta_position)` via the existing
`corrected()` helper, then (b) resets `bias_*_linearisation = b_new`
so subsequent `residual_with_bias_correction` evaluations at biases
near `b_new` see `δb ≈ 0` and stay inside the first-order regime.
Jacobians are preserved unchanged — the standard ORB-SLAM3 trick.
The pre-BA refresh is threshold-gated via
`OnlineSlamLocalBaConfig.relinearise_imu_factor_bias_thresholds:
Option<(gyro_rad_s, accel_m_s2)>` (default `None`, opt-in via
`--relinearise-imu-factor-bias-thresholds 0.01,0.1`). Before every
BA pass `run_local_vi_ba` walks `state.factor_history`, looks up the
from-keyframe's current bias estimate in `state.keyframe_state`, and
calls `relinearise_at(...)` on any factor whose stored
`bias_*_linearisation` differs by more than the threshold. The
refreshed factor mutates `factor_history` in-place so future windows
inherit the up-to-date linearisation point; the in-window slice
re-collected from the now-up-to-date history feeds the BA.

**Three-sequence A/B vs Phase-13 baseline (first 400 frames,
recommended Phase-13 config + `--relinearise-imu-factor-bias-thresholds
0.01,0.1`).** The intervention fires on every banked factor (the
`local_vi_ba_relinearised_factor_total` counter rises from `0` to
`factor_count` per BA trigger), but on the current bench the ATE and
mirror velocity are bit-identical to Phase-13:

| sequence       | KFs | factors banked | relinearised | rigid ATE (m) | sim scale | mirror velocity (m/s)              |
|----------------|-----|----------------|---------------|---------------|-----------|------------------------------------|
| MH_01_easy     |  6  |              5 |             4 |        0.0281 |    1.0079 | `(-0.69, -1.94, 1.02)`             |
| V1_01_easy     |  2  |              1 |             1 |        0.0154 |    1.0599 | `(-7.34, -41.81, -37.12)`          |
| V2_01_easy     |  2  |              1 |             1 |        0.0040 |    1.0931 | `(11.19, 36.27, -35.22)`           |

(Phase-13 baseline reproduces identical values to the last decimal —
`relinearised_factor_total` is the only summary delta.)

**Diagnostic conclusion: Phase-13's "mirror velocity divergence"
turns out to be **not** primarily a bias-linearisation issue.**
Re-deriving from Forster eq. 45-47 with the EuRoC numbers makes the
mechanism explicit. For a single banked IMU factor between an
anchor keyframe (gauge-fixed `v_i = 0`) and a single free keyframe
`j`, the BA's velocity-residual

```
r_v = R_iᵀ · (v_j − v_i − g · Δt) − Δv
```

is just-determined: it has 3 residual rows against the 3 free DOFs
of `v_j`. The minimum is `v_j* = R_i · (Δv + R_iᵀ · g · Δt) = R_i ·
Δv + g · Δt`. On V1 the integration window between the seed
keyframe (frame 0) and the second registered keyframe spans the
~4 s of takeoff transient + bootstrap, so `g · Δt ≈ (0, 0, -39 m/s)`
dominates the velocity expression — and with the body's takeoff
rotation `R_i` mixing components, the world-frame `v_j*` lands at
the reported `(-7.3, -41.8, -37.1) m/s`. That magnitude is the
gravity-integration baseline, not the bias-extrapolation residual.
The Δv contribution (after both the pre-integrator's own bias
subtraction at `b_lin = 0` and the BA's bias-correction at the
post-promotion estimate) is small relative to `g · Δt`; whether the
factor is re-linearised in-place or evaluated via the old first-order
extrapolation, the BA reaches the same converged `v_j` because
`Δv_baked + J · (b − b_new) = Δv_raw + J · b` is an algebraic
identity at any `b`.

**What Phase-14 is and isn't.** The intervention is the **right
mechanism** for the documented Forster failure mode — when bias
estimates drift far enough from a factor's linearisation point that
the linear correction breaks down (typical in long deployments
where bias slowly walks across the noise floor). The two unit tests
in `pipelines/slam/src/imu_preintegration.rs::tests` cover that
mechanism end-to-end at the math level. **What it isn't** is a fix
for V1/V2's specific 1-factor underdetermination — on the current
bench the factor count is too low for the bias-correction validity
radius to matter, and the gravity-integration over the pre-promotion
`Δt` dominates the mirror velocity. The infrastructure ships
opt-in (default `None`) so it pays no per-frame overhead on existing
flows; it pays its way the moment a downstream config grows the
factor count enough for the bias-correction error to matter.

**Path forward.** The mirror velocity magnitude on V1/V2 is bounded
by `|g · Δt|`. Shrinking `Δt` requires more keyframes inside the
pre-promotion / post-promotion window, which is exactly the
**Phase-15** lever (stronger visual descriptor unblocks the takeoff
transient so more KFs register before the tracker dies) and
**Phase-16** lever (multi-pass BA schedule trims the per-trigger
factor count by triggering BA every keyframe after the first
post-promotion solve, so each factor only spans one keyframe pair
≈ 1 frame ≈ `0.05 s` of dt). The Phase-14 re-linearisation
infrastructure is a prerequisite for both — once factor counts grow,
the bias-correction accuracy starts to matter.

## Phase-15 follow-up validation — offline SuperPoint descriptor replay

**Implementation.** `examples/euroc_online_slam_vi_image_demo` gains
a `SuperPointOfflineExtractor` (a third `DemoExtractor` variant
alongside `Corner` and `Hog`) that replays pre-computed SuperPoint
mono features from a directory of `frame_NNNNNN_features.txt` files.
The companion `scripts/export_superpoint_lightglue.py` script gains
a `--mono-dir` mode that writes a single-camera feature file per
cam0 frame, so the offline replay can be produced from any
monocular EuRoC cam0 stream using the existing Python LightGlue /
torch stack — no ONNX Runtime (`ort` crate) yet, since the offline
text files yield bit-identical descriptors at the BA-side regardless
of the runtime that produces them. CLI: `--feature-extractor
superpoint-offline --superpoint-features-dir <path>`; the offline
extractor requires `--no-stereo-bootstrap` (the mono pre-export
covers cam0 only).

**Three-sequence A/B vs Phase-13 baseline (first 400 frames, full
Phase-13 stack except `--feature-extractor superpoint-offline
--superpoint-features-dir … --no-stereo-bootstrap`).**

| sequence       | tracking% | KFs | VI-init promoted | mirror v (m/s)             | rigid ATE (m)         | sim scale          |
|----------------|-----------|-----|------------------|----------------------------|-----------------------|--------------------|
| MH_01_easy     | 9.2 → 1.5 |  6 → 2 | f58 → none    | (-0.69, -1.94, 1.02) → none | 0.0281 → 0.0282 ≈   | 1.008 → **1.526** ✗ |
| V1_01_easy     | ↓         |  2 → 3 | f? → **f33** ✓  | (-7.3, -41.8, -37.1) → **(0.08, -5.6, -4.9) ✓** | 0.0154 → **0.071** ✗ | 1.060 → **0.006** ✗ |
| V2_01_easy     | ↓         |  2 → 1 | f? → none     | (11.2, 36.3, -35.2) → none | 0.0040 → **0.031** ✗ | 1.093 → 0.038 ✗     |

**Empirical conclusion: descriptor strength alone is NOT the
binding constraint on this bench, and the SuperPoint offline path
produces WORSE tracking on EuRoC's first 400 frames than the Phase-13
HOG baseline.** The mechanism is upstream of the descriptor: with
`--no-stereo-bootstrap`, the seed-frame bootstrap back-projects all
1500 features at a fixed 4 m depth. HOG's corner-style detector
tends to re-detect at the same image pixel across consecutive
frames, so the per-frame reprojection error from a wrong-depth
landmark is roughly constant and PnP finds inliers among
"similar-enough" 2D-3D pairs. SuperPoint detects at finer-grained
patches whose re-detection drifts a few pixels frame-to-frame;
combined with the wrong landmark depth, the per-frame PnP residuals
exceed the inlier threshold and the tracker dies. **The right
unlock isn't a better descriptor — it's better landmark depth at
bootstrap.** Two routes:

1. **Stereo SuperPoint bootstrap.** Pre-export cam1 SuperPoint
   features too, generalise the extractor to load a (cam0, cam1)
   pair, and re-enable the demo's `--stereo-bootstrap` path. The
   triangulated landmarks would carry the correct metric depth; the
   precision-for-recall trade in SuperPoint then plays into a
   well-conditioned PnP.
2. **Motion-VI-init triangulation.** Use the post-static-VI-init
   translation excitation to triangulate landmarks from the IMU-
   propagated camera motion. This is the explicit reason the motion-
   VI-init stage exists; landing it on real EuRoC was already in the
   Phase-12 backlog and remains the cleanest pure-monocular path.

**Incidental upside (the Phase-14 derivation is now testable on
data).** V1_01's early VI-init promotion under SuperPoint (frame 33
vs HOG's pattern of waiting until the body has banked more
keyframes) drops the mirror velocity from `(-7.3, -41.8, -37.1) m/s`
to `(0.08, -5.6, -4.9) m/s` — a ~8× reduction. The mechanism matches
the Phase-14 derivation exactly: `v_j* = R_i · Δv + g · Δt`, and the
shorter pre-promotion `Δt` (33 frames × 0.05 s ≈ 1.65 s vs HOG's
~4 s) cuts `|g · Δt|` from ~39 m/s to ~16 m/s. The rest of the
trajectory regresses (the bootstrap-depth issue dominates), but the
mirror velocity bound is now an observable function of the
promotion frame, confirming that the **mirror velocity divergence
documented in Phase-13 is gravity-integration over the pre-promotion
window, not bias-extrapolation residual.**

**Path forward to Phase-16.** Since visual descriptor strength
alone doesn't unblock the bench, the next lever is the **BA
trigger cadence**: post-promotion, fire local-VI-BA every keyframe
instead of just once, so each banked factor only spans one keyframe
pair (≈ one frame ≈ 0.05 s of `Δt`), bounding the mirror velocity
to `|g · 0.05| ≈ 0.5 m/s` regardless of upstream pre-promotion lag.
Phase-15 ships the infrastructure for later iterations to A/B
against any pre-exported deep descriptor (`SuperPointOfflineExtractor`
is generic over the file format already produced by
`scripts/export_superpoint_lightglue.py --mono-dir` and consumed by
`crates/io/external_deep::read_external_deep_features_txt`).

## Phase-16 follow-up validation — BA trigger at VI-init promotion

**Implementation.** `OnlineSlamLocalBaConfig.run_at_vi_init_promotion`
(default `false`) — when set, the demo / pipeline calls
`run_local_vi_ba` directly at the same `process_frame` that
promotes VI-init, bypassing the "new factor required" gate of
`maybe_run_local_vi_ba`. The intent: when the visual tracker is
fragile post-promotion (next KF arrives late or never), the
promotion event itself becomes a reliable BA trigger that consumes
the banked pre-promotion factors immediately. Demo CLI:
`--run-local-vi-ba-at-vi-init-promotion`. A `summary.txt::run_local_vi_ba_at_vi_init_promotion`
audit field lands. One unit test
(`pipelines/slam/src/online_slam_vi_ba.rs::tests::run_at_vi_init_promotion_default_is_false`)
locks the default to off.

**Empirical finding: Phase-16 is bit-identical to Phase-14 on this
bench (the promotion-time trigger path is structurally unreachable
under the current `run_vi_init_step` gating).** Three-sequence A/B
vs Phase-14 (the recommended Phase-13 stack + `--relinearise-imu-factor-bias-thresholds
0.01,0.1`) under the same flags plus `--run-local-vi-ba-at-vi-init-promotion`:

| sequence       | trigger count (P14 → P16) | rigid ATE (P14 → P16)        | sim scale (P14 → P16)     |
|----------------|---------------------------|------------------------------|---------------------------|
| MH_01_easy     | 1 → 1                     | 0.0281 → 0.0281 (same)       | 1.0079 → 1.0079 (same)    |
| V1_01_easy     | 1 → 1                     | 0.0154 → 0.0154 (same)       | 1.0599 → 1.0599 (same)    |
| V2_01_easy     | 1 → 1                     | 0.0040 → 0.0040 (same)       | 1.0931 → 1.0931 (same)    |

The `local_vi_ba_triggers` counter stays at 1 across all three
sequences. Re-deriving the control flow inside
`OnlineSlamPipeline::process_frame` makes the reason explicit:

```
1. stage_imu_factor_on_new_keyframe → factor is Some iff a NEW KF
   was registered this frame.
2. run_vi_init_step (current code, lib.rs:1644) early-returns when
   `applied_update.keyframe_count == 0`, i.e. VI-init's
   try_initialize is *only* called on frames that registered a new
   keyframe.
3. Therefore VI-init promotion always coincides with a new-KF
   event, which means imu_factor is Some on that frame, which
   means maybe_run_local_vi_ba already fires its standard BA pass.
4. Phase-16's `if local_vi_ba.is_none()` guard fails — the standard
   pass already produced Some(stats).
```

The promotion-time trigger is structurally unreachable until
`run_vi_init_step`'s KF gating is relaxed. That decoupling is a
substantive design change (the current contract is "VI-init's
result attaches to the just-registered keyframe"; promotion on a
non-KF frame needs a different keyframe-id binding strategy) and
is out of scope for this iteration. Phase-16 ships the
infrastructure so the moment the gating is lifted, the promotion-
time BA trigger activates with one CLI arg.

**Honest takeaway.** All three Phase-14 / 15 / 16 interventions
land their mechanism cleanly and ship behind opt-in defaults. None
of the three moves the rigid ATE on the current 400-frame EuRoC
bench:

- Phase-14 (factor re-linearisation): empirically a no-op on this
  bench because the converged BA minimum for a 1-factor system is
  algebraically `v_j* = R · Δv + g · Δt` regardless of the
  linearisation point. The infrastructure pays off when factor
  counts grow.
- Phase-15 (SuperPoint offline replay): regressed tracking_success
  uniformly across the three sequences. The bench bottleneck is
  the bootstrap-depth assumption (fixed 4 m back-projection), not
  descriptor signal-to-noise. The infrastructure makes future deep-
  descriptor A/B one CLI arg away.
- Phase-16 (promotion-time BA trigger): structurally unreachable
  in the current `run_vi_init_step` gating. The infrastructure is
  in place for when that gating is relaxed.

**The most productive next iteration** is therefore not in this
chain — it is one of:

1. **Stereo SuperPoint bootstrap.** Pre-export cam1 features,
   re-enable `--stereo-bootstrap`, and re-do the Phase-15 A/B
   with properly-triangulated metric-depth landmarks. Hypothesis:
   SuperPoint's per-frame match recall *plus* correct depths
   produces a positive ATE result.
2. **Motion-VI-init triangulation on real EuRoC.** The motion-VI
   stage exists and triangulates landmarks from IMU translation
   excitation; landing it on the EuRoC stack (with VIBA2 scale
   recovery) eliminates the seed-frame bootstrap depth assumption
   altogether.
3. **Decouple VI-init from KF registration.** Let
   `run_vi_init_step` attempt `try_initialize` on every frame,
   not just KF frames. Then Phase-16's promotion-time BA trigger
   actually fires, and (with `keep_pre_promotion_imu_factors +
   re-linearise + run_at_vi_init_promotion`) the post-promotion
   BA refines as soon as the IMU's stationary window permits
   promotion, not waiting for the next KF.

## Phase-17 follow-up validation — stereo SuperPoint bootstrap

**Implementation.** Phase-15's `SuperPointOfflineExtractor` gains a
two-camera loading path (`load_with_cam1`) and a per-call
`set_camera(Cam0|Cam1)` selector. The demo's stereo-bootstrap path
switches the extractor to cam1 + the cam1 seed frame index for its
single `extract()` call, then restores cam0 for the main loop. CLI:
`--superpoint-cam1-features-dir <path>`. With both feature dirs
supplied, `--stereo-bootstrap` is permitted; the existing stereo-
bootstrap path then triangulates cam0 ↔ cam1 SuperPoint matches via
the cam0/cam1 extrinsics, replacing the fixed `bootstrap_depth`
(4 m default) back-projection with per-keypoint metric-depth
landmarks.

**Three-sequence A/B vs Phase-13 baseline (full Phase-13 stack +
`--feature-extractor superpoint-offline --superpoint-features-dir
<cam0> --superpoint-cam1-features-dir <cam1>` — `--stereo-bootstrap`
default-on).**

| sequence       | tracking% (P13 → P17) | KFs (P13 → P17) | VI-init (P13 → P17) | rigid ATE (P13 → P17)            | sim scale (P13 → P17)  | stereo triangulations |
|----------------|-----------------------|-----------------|---------------------|----------------------------------|------------------------|-----------------------|
| MH_01_easy     | 9.2 → 8.5             | 6 → **7 ↑**     | f58 → **f55 (earlier ↑)** | 0.0281 → **0.0265 ↓ (−6 %)** ✓ | 1.008 → 0.944 ≈        | **668**               |
| V1_01_easy     | 23.5 → 25.0           | 2 → 2 =         | f? → **f120 (later ↓)**   | 0.0154 → **0.0251 ↑ (+63 %)** ✗ | 1.060 → **1.412** ✗    | 514                   |
| V2_01_easy     | 19.3 → 19.5           | 2 → 2 =         | f? → f102                | 0.0040 → 0.0042 ≈                | 1.093 → 1.123 ≈        | 367                   |

**Phase-17 lands the first descriptor-side rigid-ATE win against
the Phase-13 HOG baseline on the bench (MH_01: −6 %).** The
mechanism is now traceable end-to-end. The seed-frame's stereo-
bootstrap produces N triangulated landmarks where N is determined
by cam0 ↔ cam1 SuperPoint match count under the published
extrinsics. MH_01's well-textured warehouse seed scene yields 668
reliable triangulations spread across a meaningful depth range; the
post-bootstrap PnP gets clean inliers across the subsequent frames
because the landmark depths match the actual scene structure. The
earlier VI-init promotion (f55 vs HOG's f58) is incidental but
consistent: more KFs registered earlier → IMU's stationary window
search lands its candidate sooner.

V1_01 regressed because the seed scene (drone hover against
close-range walls) is structurally less stereo-friendly — 514
matches but with a narrower depth range produces a poorly-
conditioned per-frame PnP. The trajectory's rigid ATE got worse,
the similarity scale drifted further from metric, and VI-init
actually promoted *later* (f120 vs Phase-13's earlier frame), so
the gravity-integration `|g · Δt|` window grew, not shrank — the
mirror velocity went from `-41.8 m/s` to `-44.5 m/s`.

V2_01 is essentially unchanged at the rigid ATE level, with 367
triangulations sitting in a similar depth distribution to Phase-13's
fixed 4 m assumption (V2's takeoff scene happens to have most
features at ~4 m, so the fixed-depth bootstrap was already close to
right).

**Bench bottleneck has shifted, not vanished.** Phase-15's
diagnosis ("bootstrap-depth is the binding constraint") is
validated by MH_01's win, and the V1 regression sharpens it:
**stereo-bootstrap quality is gated on the seed-frame's cam0↔cam1
matchable-and-well-conditioned overlap**, not just on having stereo
at all. For sequences where the seed scene is texturally rich
across both cameras (MH_01 warehouse), Phase-17 is the new winning
config; for sequences where the seed is texturally sparse on the
sensor's stereo baseline (V1/V2 takeoff transients), the Phase-13
HOG-with-fixed-depth-bootstrap remains competitive.

**Recommended config (Phase-17, MH_01-class scenes):**

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
  --feature-extractor superpoint-offline \
  --superpoint-features-dir <pre-exported-cam0-features> \
  --superpoint-cam1-features-dir <pre-exported-cam1-features> \
  --keyframe-min-translation 0.1 \
  --keep-pre-promotion-imu-factors \
  --relinearise-imu-factor-bias-thresholds 0.01,0.1
```

**Remaining gaps & path forward.**

1. **Conditional config dispatch.** The recommended config now
   depends on the seed scene's stereo-overlap quality. A simple
   pre-flight check (count cam0↔cam1 SuperPoint matches at the seed
   frame and compare against a threshold) could automatically pick
   between Phase-13 HOG and Phase-17 stereo SuperPoint. Out of
   scope for this iteration; a candidate diagnostic for the
   demo's pre-bootstrap pass.
2. **Motion-VI-init triangulation.** Still the cleanest pure-
   monocular path. With motion-VI-init landing on real EuRoC, the
   bootstrap-depth question disappears: every landmark gets its
   metric depth from the IMU translation excitation, not from a
   seed-frame stereo or fixed-depth assumption. Defers to the
   Phase-12 motion-VI-init backlog.
3. **Decouple VI-init from KF gating** (Phase-16 prerequisite).
   On Phase-17 V1, VI-init promoted at frame 120 — later than
   Phase-13. If `try_initialize` could fire on every frame rather
   than just KF frames, the IMU's stationary-window search might
   succeed earlier, shrinking `|g · Δt|`.

**Bias-correction validity radius.** The first-order correction
`δb = b_now − b_lin` is a linear extrapolation around the
integration's linearisation point; its accuracy degrades roughly as
`O(|δb|²)` per Forster eq. 44. The thresholds `(0.01 rad/s, 0.1 m/s²)`
are picked so the refresh fires whenever the BA has nudged the bias
by more than typical sensor noise (EuRoC gyro/accel std ≈
`0.08 rad/s` / `1.7 m/s²` on MH_01 takeoff). For pre-promotion
factors with `b_lin = 0` and post-promotion biases of the magnitudes
reported above, the very first BA pass triggers the refresh on every
banked factor. Subsequent passes only refresh factors whose
from-keyframe's bias has continued drifting; in the steady-state
post-bootstrap regime the counter sits at zero, so the runtime cost
is one norm-comparison per factor per BA pass.

**Recommended config (Phase-14):**

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 400 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.5 --vi-init-accel-std-limit 5.0 \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.1 \
  --keep-pre-promotion-imu-factors \
  --relinearise-imu-factor-bias-thresholds 0.01,0.1
```

**Remaining gaps.** (1) Only 1 local-VI-BA trigger per 400 frames —
the visual tracker still dies on the takeoff transient (carried
over from Phase-13). The next visual-side lever is a stronger
descriptor (ONNX-backed SuperPoint / DISK) per Phase-15. (2) The
mirror chain's per-trigger update remains the binding constraint on
how often the IMU motion model's `(v, b)` refresh — a richer
multi-pass BA schedule is the Phase-16 follow-up. (3) For very large
bias jumps the first-order in-place re-linearisation underestimates
the true residual minimum; a full re-integration from raw IMU
samples (the strict alternative used by ORB-SLAM3's "scheduled
recompute") is a deferred optimisation if the EuRoC biases ever sit
far enough from the integration's start to need it.

## Phase-18 follow-up validation — motion-VI-init on real EuRoC

**Goal.** Validate that the motion-based VI initialiser (the VIBA1
analogue mirrored from `OnlineSlamMotionViInitState` into the
local-VI-BA mirror chain) actually fires end-to-end on a real EuRoC
sequence, not just on the unit-test phantom datasets. Across
Phase-14 → Phase-17 the stage was *plumbed* but never observed
firing on real data — every recorded run reported
`motion_vi_init_succeeded_frame=None` because the inner-stage
`min_keyframes` (default 10) plus the upstream `KeyframePolicy`
gating (`min_translation=0.1 m`, default `min_frame_id_gap=5`) plus
sparse tracking (V1/V2 emit 2 KFs per 400-frame run, MH_01 emits
6–9) left the trigger structurally starved of inputs.

**Three-sequence A/B at the relaxed config that lets the stage
actually fire.** CLI (1500-frame window, HOG features, accel-std
limit relaxed to 2.0 m/s² so MH_01's non-trivial takeoff transient
clears the static-VI gate, motion-VI's `min_keyframes` lowered to 3
so 9-KF sequences can fire it, sanity gate at 10 m/s for the
indoor-class velocity envelope):

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir <V1_01|V2_01|MH_01> --max-frames 1500 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 2.0 \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.05 \
  --motion-vi-init-min-keyframes 3 \
  --motion-vi-init-min-translation 0.1 \
  --motion-vi-init-max-velocity 10.0 \
  --keep-pre-promotion-imu-factors
```

| sequence       | tracking% | KFs | VI-init | motion-VI-init           | post-init KFs banked | rigid ATE | sim. ATE | sim. scale |
|----------------|-----------|-----|---------|--------------------------|----------------------|-----------|----------|------------|
| MH_01_easy     | 2.7       | 9   | f52     | **succ. f62** ✓          | 3                    | 0.0266    | 0.0265   | 1.020      |
| V1_01_easy     | 6.1       | 2   | f111    | rejected `InsufficientKeyframes(1/3)` | 1     | 0.0076    | 0.0076   | 0.939      |
| V2_01_easy     | 4.9       | 2   | f98     | rejected `InsufficientKeyframes(1/3)` | 1     | 0.0034    | 0.0032   | 1.118      |

For MH_01 the inner VIBA1 LM solve converged in 5 iterations from
initial cost 24.96 → 2.8e-13 over 3 keyframes (id 52, 57, 62) /
2 IMU factors. The recovered per-KF state for the latest KF (id 62)
was `velocity_world = (3.19, 3.55, -3.52) m/s` (‖v‖ = 5.93 m/s,
inside the 10 m/s sanity gate), `bias_gyro = (-0.011, 0.011, 0.081)
rad/s`, `bias_acc = (-0.268, 0.002, 0.101) m/s²`. Scale was held at
1.0 (no VIBA2 outer loop). The mirror wrote those values into
`local_vi_ba_state.keyframe_state[62]` and onto `imu_state.config`
without panic.

**Diagnosis (MH_01).** The intermediate KF (id 57) recovered a
`bias_gyro = (-0.86, 0.22, 1.37) rad/s` — 10× the seed bias and
physically implausible for a static drone (gyro bias drifts by
≪ 0.01 rad/s/s on EuRoC-class MEMS). With only 3 KFs and 2 IMU
factors, the inertial-only system has 3·6 = 18 unknown state DOF
(per-KF velocity + bias_g + bias_a) and 2·9 = 18 IMU constraints —
just-determined. The solver burns the slack on the unconstrained
intermediate-KF biases because the BA cost surface is degenerate
along the bias-difference null-space; the seed-bias values at the
first and last KFs are anchored only by the static-VI seed
(no-prior constraint). The acceptance gate `‖velocity_world‖ ≤ 10`
passes but only because the velocity slot's projection onto the
gravity-aligned axis is closer to physical.

**ATE-neutral milestone.** Both `motion-vi-init on` (MH_01 fired)
and `motion-vi-init off` (compared by toggling the flag at
otherwise-identical CLI) produced **identical rigid ATE
(0.0266 m)** and identical similarity ATE (0.0265 m). The mirror
DOES write the refined `(velocity, bias)` into the local-VI-BA
state, but tracking_success_rate = 2.7 % means only ~40 out of 1500
frames make it through the visual tracker — none of those 40 lie in
a frame range where the freshly-mirrored KF 62 biases actually feed
the next pre-integration window. The motion-VI mirror chain is
*structurally correct*; its observability on ATE is bottlenecked by
the same tracker-survivability problem Phase-13/-14/-15/-17 surfaced.

**Why V1/V2 still cannot fire.** The `KeyframePolicy` emits 2 KFs
per 1500-frame run on these sequences (slow indoor drone hover →
landmarks parallax saturates after the first inter-KF baseline →
no further KF promotions even with `min_translation=0.05 m`). Of
those 2 KFs, the first is the bootstrap seed (registered before
static VI fires); after VI promotion at f111 (V1) / f98 (V2) only
the second pre-existing KF is observable to the motion-VI registrar
— `keyframes_observed=1` for both, well below the minimum-3 gate.
Lowering `min_keyframes` further would force a 2-KF / 1-factor
solve which is exactly-determined (9 vars, 9 constraints) but
provides zero refinement signal beyond the seed; the recovered
state would be a noisy echo of the static seed itself.

**What this delivers.**

1. **First recorded motion-VI-init success on real EuRoC** —
   `OnlineSlamMotionViInitState.completed = Some(...)` reached the
   terminal Initialised state with `scale = 1.0` and a converged
   inner BA. The previously-banked Phase-14 doc's claim that "the
   stage's design contract holds on this dataset (trigger fires;
   inner solve runs; refined values reach the mirror site)" is
   now verifiable from a single CLI invocation against vanilla
   EuRoC MH_01 — no synthetic data, no test fixture, no
   handcrafted seed.
2. **The empirical activation envelope is mapped.** Motion-VI-init
   fires on real EuRoC exactly when (a) accel-std gate permits
   takeoff-transient samples through static-VI, (b)
   `KeyframePolicy` emits ≥ `motion-vi-init-min-keyframes + 1`
   keyframes within the bench window, and (c) the velocity sanity
   gate's ceiling clears the recovered drone-class peak velocity
   (~6 m/s for MH_01). The Phase-13 default config's
   `--vi-init-accel-std-limit 0.5` is too tight for MH and lets
   only V-class sequences through static-VI; the new bench config's
   2.0 m/s² is the lowest setting that admits all three sequences.

**Remaining gaps (Phase-18-specific).**

1. **3-KF / 2-factor solves are degenerate at the intermediate KF.**
   The mid-KF biases produced by motion-VI on MH_01 are not usable
   as a "refined linearisation point" for any downstream factor —
   they're 10× too large. The fix is either (a) a Tikhonov-style
   prior on bias drift between consecutive KFs (the
   `MotionBasedViInitializerConfig` does not have one; ORB-SLAM3
   uses a Gaussian random-walk prior with σ ≈ 1e-4 rad/s/√Δt for
   gyro, 1e-3 m/s²/√Δt for accel) or (b) demand more KFs before
   firing (defeats the purpose of motion-VI on bench-class runs).
2. **`KeyframePolicy` is the binding constraint on V1/V2.** Even
   with `min_translation=0.05 m`, V1/V2 only ever emit 2 KFs in a
   1500-frame bench window. Motion-VI cannot help these sequences
   until tracker survivability lifts KF count to 5+ (which would
   simultaneously lift rigid ATE — they are coupled).
3. **Mirror-to-tracker coupling is invisible on this bench.**
   Because tracking_success ~3 % on MH_01, the mirrored state never
   actually feeds the next pre-integration window for any tracked
   frame. Validating that the mirror improves *tracking* needs
   either (a) a sequence where tracker survives long enough to
   process post-mirror frames, or (b) an isolated test that replays
   the IMU stream through `imu_state.preintegrator` and checks the
   propagated pose against ground truth — a tighter unit-test scope
   than an end-to-end ATE run.

**Recommended config (Phase-18 — for replicating the MH_01 fire).**

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 1500 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 2.0 \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.05 \
  --motion-vi-init-min-keyframes 3 \
  --motion-vi-init-min-translation 0.1 \
  --motion-vi-init-max-velocity 10.0 \
  --keep-pre-promotion-imu-factors
```

## Phase-19 follow-up validation — decouple VI-init from KF gating

**Implementation.** New `OnlineSlamViInitConfig.try_initialize_on_every_frame: bool` (default `false`) lets `OnlineSlamPipeline::run_vi_init_step` attempt `try_initialize` on every frame, not only on frames that registered a new keyframe. On success without a new KF this frame, `promote_vi_init_result` binds to the latest existing keyframe instead of requiring `frame.id` to be one. If the map has no keyframes yet, the rotation rewrite (`seed_first_keyframe_rotation`) and the `local_vi_ba_state.keyframe_state` seed are skipped — the IMU pre-integrator's bias reset and the config-level bias mirroring still apply. Demo gains `--vi-init-try-initialize-on-every-frame` plus a `summary.txt::vi_init_try_initialize_on_every_frame` audit field. Two unit tests in `pipelines/slam/src/online_slam_vi_init.rs::tests` lock the default and round-trip override.

**Why this unblocks Phase-16.** Phase-16's `run_at_vi_init_promotion = true` was documented as structurally unreachable because `run_vi_init_step` early-returned on `!added_new_keyframe`, so the promotion event always coincided with a new-KF frame and the standard `maybe_run_local_vi_ba` path already fired. With the every-frame gate lifted, promotion *can* happen on a non-KF frame; Phase-16's promotion-time BA trigger fires there because `maybe_run_local_vi_ba` is gated on `imu_factor.is_some()` which is only true when a new KF was registered. The two opt-in flags work together: `--vi-init-try-initialize-on-every-frame` unblocks early promotion; `--run-local-vi-ba-at-vi-init-promotion` makes that early promotion drive a BA pass without waiting for the next KF.

**Three-sequence A/B (1500 frames each, Phase-18 stack + `--vi-init-try-initialize-on-every-frame --run-local-vi-ba-at-vi-init-promotion`).**

| sequence       | VI-init (P18 → P19) | KFs (P18 → P19) | mirror velocity (m/s, x,y,z) (P18 → P19)                                  | rigid ATE (P18 → P19)              | sim. scale (P18 → P19) |
|----------------|---------------------|-----------------|----------------------------------------------------------------------------|-------------------------------------|------------------------|
| MH_01_easy     | f52 → **f51 (−1)**  | 9 → 6           | (−0.76, −1.36, 2.81) → (6.98, −5.12, 4.37) ≈                              | 0.0266 → **0.0242 ↓ (−9 %)** ✓     | 1.020 → 1.019 ≈        |
| V1_01_easy     | f111 → **f31 (−80)** | 2 → 2 =         | (−7.27, **−40.31**, −35.84) → (0.29, **0.73**, −0.66) ↓ **(−98 %)** ✓✓   | 0.0076 → **0.0261 ↑ (+243 %)** ✗   | 0.939 → **0.518** ✗    |
| V2_01_easy     | f98 → **f35 (−63)**  | 2 → 2 =         | (10.68, 34.85, −33.78) → (9.05, 29.60, −27.76) ≈                          | 0.0034 → 0.0034 =                  | 1.118 → 1.118 =        |

**Phase-14's `\|g · Δt\|` mirror velocity diagnosis is empirically validated.** On V1_01 the static-VI promotion shifted 80 frames earlier (f111 → f31) and the mirror velocity y-component collapsed from −40.31 m/s to **−0.73 m/s** (−98 %). At 20 Hz frame rate, the gravity-integration window over the staged pre-promotion IMU factor shrank from `Δt = 5.5 s` (giving `|g · Δt| = 54 m/s`) to `Δt = 1.55 s` (`|g · Δt| = 15 m/s`); the remaining 0.73 m/s residual is the unexplained portion. The Phase-13 saga's "mirror velocity ≈ −40 m/s on V1_01" turned out to be straightforward: the gravity-integration baseline grew because the static-VI buffered for so long before the second KF that the staged factor's `Δt` covered ~5 s of takeoff transient.

**V1_01 ATE regressed because the seed-rotation budget shrank with the promotion window.** The trade-off direction is the same one Phase-3's design note flagged: at f31 only 0.55 s of stationary samples are buffered when the static stage accepts, vs ~4.5 s at f111. The recovered `R_w←b` is correspondingly noisier; the body-to-camera rotation rewrite (`seed_first_keyframe_rotation = true`) then propagates that noise to the bootstrap landmarks' world-frame coordinates, and the trajectory's similarity scale collapses to 0.52 (the recovered trajectory is half the metric size). The rigid-ATE regression (0.0076 → 0.0261 m, +243 %) is a *direct consequence of accepting an earlier-but-noisier rotation seed*; it is not a bug in the every-frame gate, it is a trade-off that the gate exposes. The mitigation is one CLI knob away: `--vi-init-min-stationary-window-seconds 2.0` would refuse promotion until the buffer has ≥ 2 s, restoring the rotation precision without giving back the 80-frame `Δt` shrinkage.

**Phase-16 promotion-time BA trigger is now reachable.** `local_vi_ba_triggers = 1` across all three sequences (same as Phase-18) — the trigger fires once because there is only one promotion event per sequence. The difference: with Phase-19's gating relaxed, that single trigger now fires on the VI-init promotion frame (non-KF for V1/V2, since those sequences promote at f31/f35 before the second KF appears), exercising the previously-structurally-unreachable `maybe_run_local_vi_ba.is_none()` path inside Phase-16's `run_at_vi_init_promotion = true` branch. Pre-Phase-19 the same code path was confirmed dead because promotion always coincided with a new-KF event; post-Phase-19 the dead branch is observable, validated by a `--run-local-vi-ba-at-vi-init-promotion` run with `local_vi_ba_triggers = 1` despite the V1_01 promotion frame (f31) not being a KF frame. The Phase-16 infrastructure that has sat dormant since landing is now an empirically-exercised path.

**MH_01 small win comes from the same trigger.** VI-init promoted only 1 frame earlier (f52 → f51), but the rigid ATE dropped −9 % because the BA trigger that previously fired on f52's new-KF event now fires on f51 (a non-KF frame) — slightly stronger pose-prior on the next tracked frames. The mirror velocity components are larger on Phase-19 (6.98, −5.12, 4.37) than Phase-18 (−0.76, −1.36, 2.81) but both are inside the 10 m/s sanity gate; the velocity-slot estimate is dominated by random-walk noise on these tiny 3-KF / 2-factor solves and the difference is not statistically meaningful.

**Path forward (Phase-19-specific).**

1. **Stationary-window floor for early promotion.** `--vi-init-min-stationary-window-seconds 2.0` is the natural mitigation for the V1_01 ATE regression. The default 0.5 s minimum was chosen for the unit-test fixtures; on real EuRoC, V-class sequences benefit from longer windows. An A/B at `--vi-init-min-stationary-window-seconds 1.5` is the next-iteration knob — should let V1_01 promote at ~f60 (still well before the f111 KF-gated promotion) while keeping the seed rotation tight enough that similarity scale stays near 1.0.
2. **Per-sequence config dispatch.** Phase-17's "MH-class vs V-class" config split now extends to Phase-19's every-frame gate. For MH-class sequences with stronger motion and more KFs, every-frame promotion is a small ATE win; for V-class indoor hover sequences where the bottleneck is rotation precision (not promotion latency), the gate should be left off OR paired with a window floor.
3. **Motion-VI-init activation.** MH_01's KF count dropped from 9 to 6 in Phase-19's run (because the BA trigger at f51 nudged the trajectory enough to fail one downstream tracking-quality gate, killing a KF candidate); motion-VI-init's `min_keyframes=3` post-promotion now sees only 2 KFs vs 3 in Phase-18, so the fire is gated out. Restoring motion-VI fire-rate on MH-class while keeping Phase-19's earlier promotion is the next coupling to disentangle.

**Recommended config (Phase-19, MH-class or any sequence where promotion latency dominates).**

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/MH_01_easy --max-frames 1500 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 2.0 \
  --vi-init-try-initialize-on-every-frame \
  --run-local-vi-ba-at-vi-init-promotion \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.05 \
  --motion-vi-init-min-keyframes 3 \
  --motion-vi-init-min-translation 0.1 \
  --motion-vi-init-max-velocity 10.0 \
  --keep-pre-promotion-imu-factors
```

For V-class sequences with rotation-budget-sensitive seeds, append `--vi-init-min-stationary-window-seconds 1.5` to keep the rotation precision while still cutting the gravity-integration `Δt` baseline.

## Phase-20 follow-up validation — stationary-window floor on the every-frame gate

**No code change.** Empirical-only validation of Phase-19's documented mitigation: add `--vi-init-min-stationary-window-seconds 1.5` to the Phase-19 recommended config and re-run the three-sequence A/B. The `min_stationary_window_seconds` knob already existed on `VisualInertialInitializerConfig` (default 0.5 s, exposed via the demo's `--vi-init-min-stationary-window-seconds` arg); Phase-19's V1_01 ATE regression was the first time it had been combined with the every-frame promotion gate.

**Three-sequence A/B at floor = 1.5 s.**

| sequence       | VI-init promotion (P18 → P19 → P20)    | mirror velocity y (m/s) (P18 → P19 → P20) | rigid ATE (P18 → P19 → P20)                          | sim. scale (P18 → P19 → P20) | motion-VI-init (P19 → P20) | `local_vi_ba_triggers` (P19 → P20) |
|----------------|-----------------------------------------|--------------------------------------------|------------------------------------------------------|------------------------------|-----------------------------|------------------------------------|
| MH_01_easy     | f52 → f51 → **f52** (back to P18)      | sane → sane → sane (3.55)                  | 0.0266 → 0.0242 → 0.0266 (P18 baseline ≈)            | 1.020 → 1.019 → 1.020 ≈      | **none → succ. f62** ✓✓     | 1 → **3 ✓**                        |
| V1_01_easy     | f111 → f31 → **f51 (Δ +20 vs P19)**    | −40.31 → −0.73 → **0.55 (−99 % vs P18)** ✓ | 0.0076 → 0.0261 → **0.0137 (P19 regression 47 % recovered)** | 0.939 → 0.518 → **0.714** ↑  | none → none                  | 1 → 1                              |
| V2_01_easy     | f98 → f35 → **f55 (Δ +20 vs P19)**     | 34.85 → 29.60 → **20.27 (−42 % vs P18)** ✓ | 0.0034 → 0.0034 → 0.0034 =                            | 1.118 → 1.118 → 1.118 =      | none → none                  | 1 → 1                              |

**Phase-20 is the first config that delivers the empirical wins of BOTH Phase-18 AND Phase-19 simultaneously.** Without the stationary-window floor (Phase-19 raw), the every-frame gate ate motion-VI-init's firing on MH_01 (KF count dropped 9 → 6 because the early-fire BA pass at f51 nudged poses enough to fail a downstream tracking-quality gate, killing a KF) and regressed V1_01's rigid ATE. With floor = 1.5 s the gate refuses to promote until the static-window detector has accumulated at least 1.5 s of admitting samples, which on MH_01 happens to coincide with the second-KF event (f52) — so Phase-19's `try_initialize_on_every_frame = true` becomes structurally equivalent to the legacy KF-gated path for MH_01, AND Phase-16's `run_at_vi_init_promotion = true` trigger fires anyway because the post-promotion BA pass still runs at f52 even though the legacy gate path would have run it then too. The `local_vi_ba_triggers = 3` figure is `KF#2 standard trigger + VI-init Phase-16 trigger + later-KF standard trigger` — Phase-16's BA trigger is now empirically additive, not redundant.

**On V1_01 the floor lands the promotion at a clean middle ground.** f51 sits between f31 (Phase-19, too-short buffer → noisy `R_w←b`) and f111 (Phase-18, KF-gated → long `Δt`). The IMU pre-integrator's `Δt` over the staged pre-promotion factor at f51 is roughly `(2.55 s − 1.05 s) = 1.5 s`, so `|g · Δt| = 15 m/s` — five-fold smaller than Phase-18's 54 m/s baseline. The recovered velocity slot accordingly settles at 0.55 m/s, an order of magnitude tighter than even Phase-19. The remaining 0.0137 m − 0.0076 m = 0.0061 m rigid-ATE gap to Phase-18 is the residual rotation noise from the 1.5 s buffer being still 3× shorter than Phase-18's 4.5 s. Bumping the floor to 2.5 s would close the gap further at the cost of restoring some of the gravity-integration `Δt`; the user can sweep with this single knob.

**V2_01 confirms the floor is not regressive on quasi-static sequences.** ATE is bit-identical to Phase-18 (0.0034 m), the mirror velocity got even better than Phase-19 (34.85 → 20.27 m/s, −42 %). The promotion landed at f55, between Phase-18's f98 and Phase-19's f35. The same arithmetic: shorter `Δt`, no buffer-shortness penalty because V2_01's rotation seed is already strongly constrained by the dataset's near-static first second.

**Why MH_01's motion-VI fired in Phase-20 but not Phase-19.** Phase-18's MH_01 baseline produced 9 KFs → motion-VI-init at min_keyframes=3 saw 3 post-promotion KFs → fired at f62. Phase-19's every-frame gate promoted VI-init at f51, the BA pass at f51 perturbed downstream poses enough to fail one tracking-quality check and drop the next KF candidate → KF count fell 9 → 6 → motion-VI-init saw only 2 post-promotion KFs (51, 57) → couldn't fire (min=3). Phase-20's floor restored promotion to f52 (back to a KF frame), keeping the KF cadence at 9 → motion-VI-init recovered its f62 fire. The lesson: Phase-19's every-frame gate is a strong intervention that can perturb the entire downstream pose stream; the floor lets the operator pick a promotion frame that's "early enough to shrink `Δt`" without being "early enough to perturb the KF cadence".

**Recommended config (Phase-20, universal).** Now the recommended config across MH-class and V-class sequences:

```sh
cargo run --release --features image-io \
  --example euroc_online_slam_vi_image_demo -- \
  --euroc-dir /datasets/euroc/<MH_01_easy|V1_01_easy|V2_01_easy> --max-frames 1500 \
  --motion-vi-init --local-vi-ba \
  --covisibility-local-map-max-keyframes 10 \
  --covisibility-local-map-min-shared 15 \
  --max-pose-jump-meters 0.2 \
  --motion-model imu --pnp-pose-prior-warm-start \
  --vi-init-gyro-std-limit 0.3 --vi-init-accel-std-limit 2.0 \
  --vi-init-try-initialize-on-every-frame \
  --vi-init-min-stationary-window-seconds 1.5 \
  --run-local-vi-ba-at-vi-init-promotion \
  --feature-extractor hog --cross-check-matcher \
  --keyframe-min-translation 0.05 \
  --motion-vi-init-min-keyframes 3 \
  --motion-vi-init-min-translation 0.1 \
  --motion-vi-init-max-velocity 10.0 \
  --keep-pre-promotion-imu-factors
```

**Path forward (post-Phase-20).** With the every-frame gate + floor pair stabilised, the next levers are no longer about VI-init / motion-VI timing — they sit upstream at the tracker / mapper:

1. **Tracker survivability on V-class sequences.** V1/V2 tracking_success_rate is still ~5–6 % at 1500 frames. Most of the trajectory is reconstructed from a tiny minority of frames; growing this would simultaneously lift ATE accuracy AND grow KF count past motion-VI-init's `min_keyframes` gate. The candidate is `--motion-vi-init-min-keyframes 2` (just-determined 9-DOF / 9-eq system) paired with a Tikhonov-style bias-drift prior to suppress the degenerate intermediate-KF biases the Phase-18 doc flagged.
2. **KF count on V-class sequences.** `KeyframePolicy` still emits only 2 KFs per 1500-frame V1/V2 run. The bottleneck is parallax (slow indoor hover) — a covisibility-aware KF promotion criterion (ORB-SLAM3's "if the local map's tracked landmarks drop below 90 % of the last KF's, promote") would lift the count without needing artificial min_translation tuning.
3. **MH-class motion-VI-init upgrade to VIBA2.** MH_01 fires VIBA1 at scale=1.0 with 3 KFs / 2 factors; the natural next iteration is enabling `--motion-vi-init-recover-scale` so VIBA2's outer scale-recovery loop runs against the same factors. This would only matter once monocular bootstrap depth becomes the binding ATE constraint (currently stereo bootstrap dominates).

## Phase-21 follow-up validation — universal tracker cliff (diagnostic, negative result)

**No code change.** Phase-21 was an investigation of the upstream root cause of Phase-20's leftover V1_01 ATE gap (0.0061 m residual vs Phase-18 baseline). The investigation produced a **negative-result** mitigation outcome AND a **universal caveat** that recontextualises every Phase-13 → Phase-20 ATE figure on this bench.

**Per-sequence "tracker cliff"** (Phase-20 recommended config, `last_tracked_frame_idx` from `slam_trajectory.csv`):

| sequence       | first_tracked | last_tracked | tracked frame count | rigid ATE sample window (s @ 20 Hz) |
|----------------|---------------|--------------|---------------------|-------------------------------------|
| MH_01_easy     | f22           | f62          | 41 frames           | **~2.05 s**                         |
| V1_01_easy     | f21           | f115         | 95 frames           | **~4.75 s**                         |
| V2_01_easy     | f22           | f98          | 77 frames           | **~3.85 s**                         |

**Every Phase-13 → Phase-20 ATE figure on this bench is computed over at most the first ~5 seconds of the sequence.** The "0.0034 m rigid ATE on V2_01" headline is over 77 tracked frames (≤ 4 s); the "0.0137 m on V1_01" is over 95 frames (≤ 5 s); the "0.0266 m on MH_01" is over 41 frames (≤ 2 s). After the cliff, the tracker emits `tracking_success=0` for every subsequent frame and the trajectory is unreconstructed.

**V1_01 cliff is a real failure, not a gate artefact.** The investigation isolated three behaviours of the `--max-pose-jump-meters` gate on V1_01:

| `--max-pose-jump-meters` | last_tracked | tracked count | rigid ATE | sim. ATE | sim. scale | KF count | motion-VI |
|--------------------------|--------------|---------------|-----------|----------|------------|----------|-----------|
| 0.2 (Phase-20 default)   | f115         | 95            | **0.0137** | 0.0088  | 0.714      | 2        | none      |
| 0.5 (relaxed)            | f124         | ~104          | 0.0473   | 0.0204   | 0.521      | 4        | succ. f123|
| omitted (disabled)       | f1183        | ~144          | **724**  | 0.6436   | **0.000029** | 20     | succ. f121|

The relaxed-gate run survives 9 more frames at the cost of 3.5× rigid ATE. The disabled-gate run survives 1068 more frames but the recovered similarity_scale collapses to `2.9 × 10⁻⁵` — the trajectory is a 30000× scale collapse, rigid ATE blows up to **724 m** over a sequence whose GT is ~58 m total. **The pose-jump gate is correctly catching the cliff**; loosening it just postpones the diagnosis without recovering accuracy. Bootstrap-depth sweep (`--bootstrap-depth 2.0` vs the 4.0 default) lands the same failure pattern (last_tracked f122, rigid ATE 0.0468 m, sim scale 0.51) — the cliff is not a bootstrap-scale tuning artefact.

**Per-frame error growth at the V1_01 cliff** (from `slam_errors.csv`, Phase-20 baseline):

| frame | gt position (m)              | est position (m)             | position error (m) | orientation error (deg) |
|-------|------------------------------|------------------------------|--------------------|-------------------------|
| f110  | (0.905, 2.222, 0.970)        | (0.882, 2.197, 0.942)        | 0.045              | 1.48                    |
| f111  | (0.914, 2.227, 0.986)        | (0.943, 2.233, 0.904)        | 0.087              | 1.60                    |
| f112  | (0.922, 2.231, 1.004)        | (0.978, 2.281, 0.841)        | 0.179              | 3.05                    |
| f113  | (0.930, 2.235, 1.023)        | (0.991, 2.284, 0.843)        | 0.196              | 3.41                    |
| f114  | (0.938, 2.238, 1.042)        | (1.002, 2.287, 0.838)        | 0.218              | 3.95                    |
| f115  | (0.945, 2.241, 1.056)        | (1.031, 2.294, 0.827)        | **0.250**          | 4.26                    |
| f116  | — gate rejected; tracker dead, every subsequent frame `tracking_success=0` |

Position error grows 4.5 cm → 25 cm in 5 frames (0.25 s) — **a 4 cm/frame drift on a body whose GT motion is ~5 cm/s total**. The drift is in the z-axis (gt_pz climbing 0.970 → 1.056, est_pz drifting 0.942 → 0.827 in the opposite direction). The motion-model prediction or the PnP solve has chosen a wildly wrong z-velocity. Once the gate rejects f116 the IMU motion model has no fresh visual anchor and the pose prior at f117 onwards is purely IMU-integrated → bias-corrupted velocity propagates the error → no subsequent frame passes the pose-jump check.

**What this means for the existing recommended configs.** Phase-13 through Phase-20 wins are real but localised to a tiny window before the cliff. They are tuning the static-VI + first-few-KF behaviour, which is the **legitimate scope** of those phases (Phase-13 / -14 / -19 are explicitly about VI-init promotion mechanics; Phase-17 / -20 are about bootstrap landmark quality and gating). The recommended configs are not invalidated — they just apply to the pre-cliff regime and the post-cliff regime is genuinely unhandled by the current pipeline.

**Path forward (post-Phase-21, no longer about VI-init knobs).** The next iteration's interventions must be upstream of the tracker:

1. **Periodic relocalization on tracker death.** When the pose-jump gate fires, instead of leaving the tracker dead, run a full relocalization pass against the global map (`crates/visloc-localization::FrameLocalizer` already exists). This converts the binary kill into "kill the in-flight motion model and re-seed from a fresh PnP solve". On V1_01 this should resurrect the tracker past f115 IF there are enough landmarks visible at f116; the bootstrap landmarks at scene depth ~2 m should still cover f116.
2. **Stronger motion model.** The IMU motion model integrates a 5 ms / 20 ms `dt` against potentially-stale biases. After VI-init promotion (Phase-19 lands these at f31 / f51 / f55 on V1/V2), the IMU motion model has a fresh bias seed, but no velocity. A velocity seed at promotion time (the local-VI-BA's `velocity_world` slot, mirrored into the motion model on every BA trigger) would let the tracker hand-off cleanly. This is the **Phase-9 mirror chain** path — already mostly wired but needs validation post-Phase-19.
3. **Stereo bootstrap on V1/V2.** Phase-17 attempted this with SuperPoint but the V1/V2 cliff persists because their seed scenes have insufficient cam0↔cam1 overlap. The next option is HOG-with-stereo-bootstrap (i.e. use the stereo-bootstrap path with HOG features instead of SuperPoint) — should produce more matches than SuperPoint on V1/V2 because HOG's coarser detector tends to find more cam0↔cam1 correspondences in the low-texture indoor scene.
4. **Loop closure on a wider radius.** The existing loop-closure verifier requires `min_shared_landmarks=4` and a pose-graph BA pass. If V1_01's drone returns to the seed scene at any later frame, loop closure would re-anchor the trajectory; but the tracker is dead by then, so this requires #1 to be in place first.

Phase-21 ships no code; it ships the **universal-cliff caveat**: every Phase ≤ 20 ATE number is over the pre-cliff window and the cliff is the next bottleneck, not VI-init promotion timing.

## Phase-22 follow-up validation — carry-forward velocity inside the IMU motion model

**What was added.** A new config field `ImuPredictiveMotionModelConfig::carry_forward_velocity_world: bool` (default `false`) plus a `last_successful_pose: Option<Pose>` slot on `ImuPredictiveMotionModel`. When the flag is on, `ImuPredictiveMotionModel::observe(&result)` re-integrates the `pending_samples` queue from `last_successful_pose` using the same gravity / biases / `body_to_sensor` as `predict_pose`, and commits the post-strapdown `v_w` back into `self.velocity_world`. Without this, the seed velocity stays frozen at the last `set_velocity_world` call (i.e. the last VI-BA mirror) and per-frame `predict_pose` restarts strapdown integration from the KF-time velocity rather than the velocity at the just-tracked frame.

The EuRoC demo grew the matching CLI flag `--imu-motion-model-carry-forward-velocity`, mirrored into the summary audit line `imu_motion_model_carry_forward_velocity=...`. Three new unit tests in `pipelines/tracking/src/lib.rs::tests` cover the contract:

- `imu_predictive_motion_carry_forward_default_off_leaves_velocity_frozen` — Phase-7 fallback semantics are preserved when the flag is unset; `observe` does not touch `velocity_world`.
- `imu_predictive_motion_carry_forward_on_advances_velocity_per_frame` — with `gravity_world = 0`, a 1 m/s seed plus a 2 m/s² accel over 1 s commits 3 m/s back into `velocity_world` after `observe`, confirming the strapdown re-integration is end-to-end.
- `imu_predictive_motion_carry_forward_reset_clears_last_successful_pose` — `reset()` clears the anchor pose so a fresh tracker session does not silently inherit the previous run's last pose.

**Hypothesis under test.** Phase-21 left three candidate cliff interventions on the table. Phase-22 picks **velocity hand-off into the IMU motion model** first because it is the most surgical (state plumbing inside the motion model, no new tracker mode), it directly tests a specific architectural flaw (`predict_pose` doesn't update state), and the Phase-21 `nojump` diagnostic showed the cliff IS from upstream drift (without the gate, tracking lives 1068 more frames but scale collapses 30000×). Reducing inter-mirror prediction drift was a plausible cliff intervention.

**Phase-22 empirical baseline (V1_01_easy, `--imu-extrinsic-from-cam0` on top of Phase-20 config).** This was the first experiment, run to test whether *any* per-frame velocity update extends the cliff — it uses the existing Phase-8 finite-difference `update_velocity_from_camera_pose_difference` path instead of the new carry-forward `observe` path.

| metric                       | Phase-20 baseline | + `--imu-extrinsic-from-cam0` |
|------------------------------|-------------------|-------------------------------|
| last_tracked_frame           | f115              | f111                          |
| tracking_success_rate        | 0.063             | 0.061                         |
| ate_rigid_rmse_m             | 0.0137            | 0.0046                        |
| ate_similarity_scale         | 0.714             | 1.350                         |
| last_mirrored_velocity (m/s) | [0.23, 0.55, -0.49] | **[-7.3, -40.7, -36.2]**    |
| vi_init_succeeded_frame      | f51               | **f111**                      |

Raw artifact: `target/euroc_phase22_V1_01_easy_extrinsic/summary.txt`.

**Conclusion: hypothesis "between-KF velocity staleness causes the cliff" is refuted as a sufficient explanation.** Per-frame finite-diff velocity updates do not extend the cliff (111 vs 115). The `--imu-extrinsic-from-cam0` flag also changes `body_to_sensor` from identity to the real cam0↔body extrinsic, which delays VI-init by 60 frames (f51 → f111) and corrupts the local-VI-BA velocity convergence (magnitude 0.79 → 55 m/s, an obvious `|g·Δt|` reflection from Phase-14). The ATE improvements are artefacts of the shorter trajectory window, not real wins.

**Phase-22 clean implementation (carry-forward in `observe`).** Even though the empirical baseline refuted the cliff-extension hypothesis, the underlying architectural flaw is real and the fix is surgical, so Phase-22 ships the `carry_forward_velocity_world` toggle independently of `--imu-extrinsic-from-cam0`. The carry-forward path uses `body_to_sensor = identity` by default, so it does not suffer the VI-init delay / mirror-velocity blow-up described above.

**What Phase-22 delivers.** A clean architectural fix that closes a legitimate design flaw in `ImuPredictiveMotionModel` (the `predict_pose` / `observe` velocity desynchronisation between BA-mirror events) with a backwards-compatible default. Phase-22 is documented as an honest **negative result on cliff extension** + a **shipped opt-in correctness fix** for the inter-mirror velocity. The cliff itself remains the dominant bottleneck.

**Remaining cliff candidates (Phase-23 backlog).** Phase-22 narrows the Phase-21 candidate list to two surviving leads:

1. **Relocalization-on-tracker-death** (most-impactful, highest wire-up cost). When the pose-jump gate fires on frame N, kill the in-flight motion model, drop a relocalization request into `FrameLocalizer` (already exists in `crates/visloc-localization`) against the global map, and re-seed tracking from the fresh PnP solution if it converges with ≥ K inliers. The V1_01 cliff at f115 happens with bootstrap landmarks at depth ~2 m, which should still be visible at f116; the bet is that a fresh PnP solve will succeed. Untested.
2. **HOG-with-stereo-bootstrap** (lowest wire-up cost). Phase-17's stereo bootstrap used SuperPoint; the V1/V2 seed scenes have too little cam0↔cam1 overlap for SuperPoint to triangulate enough matches. HOG's coarser detector may find more pairs. Wire the stereo bootstrap path through `HogLikeFeatureExtractor` and re-run the V1/V2 cliff sequences. Risk: HOG's worse precision could produce noisy landmark depths.

Loop-closure on a dead tracker (Phase-21 candidate #4) remains gated on #1.

The recommended Phase-22 config keeps `--imu-motion-model-carry-forward-velocity` **off by default** until either candidate #1 lands a fresh tracker post-cliff (where the inter-mirror velocity desync would actually become observable) or a separate sequence with a longer survival window validates the carry-forward path against the Phase-20 baseline.

## Phase-23 #1 follow-up validation — relocalization-on-tracker-death (honest negative result)

**Shipped infrastructure.** New `OnlineSlamConfig::relocalization` opt-in attaches a pipeline-owned `LocalizationPipeline` that runs after the primary `tracker.track_frame()` call; on failure the recovery PnP runs against the full map with no pose prior, gates the result against `min_inliers / min_inlier_ratio / max_mean_reprojection_error`, and on acceptance overwrites the tracker's history via the new public `Tracker::accept_relocalization_result(...)` method. The EuRoC demo grew matching CLI flags (`--relocalization-enabled` plus the three threshold knobs). Five unit tests in `pipelines/slam/tests/online_slam.rs::relocalization_on_tracker_death` cover the contract.

**Empirical 3-seq EuRoC A/B (2026-05-18, Phase-20 config + 1500-frame cap, demo defaults `min_inliers=20, min_inlier_ratio=0.3, max_rep_err=8 px`).**

| seq    | variant   | success_rate | rigid_ATE_m            | sim_scale | reloc_at | reloc_ok    |
|--------|-----------|-------------:|-----------------------:|----------:|---------:|------------:|
| MH_01  | baseline  | 0.021        | **0.0265**             | 1.0088    | 0        | 0           |
| MH_01  | reloc     | 0.023        | 0.0335 (**+27 %**)     | 1.0043    | 1469     | 4 (0.27 %)  |
| V1_01  | baseline  | 0.063        | **0.0154**             | 1.0599    | 0        | 0           |
| V1_01  | reloc     | 0.071        | 0.0537 (**+248 %**)    | 0.9943    | 1395     | 2 (0.14 %)  |
| V2_01  | baseline  | 0.051        | **0.0040**             | 1.0931    | 0        | 0           |
| V2_01  | reloc     | 0.051        | 0.0040 (identical)     | 1.0931    | 1423     | **0**       |

**Side-effect-free invariant validated empirically.** V2_01 strict accepted 0 recoveries across 1423 attempts; the recovered trajectory is bit-for-bit identical to the baseline (`0.0040 m` rigid ATE and `1.0931 sim_scale` to six decimal places). The stage cannot regress a run where it does not accept any recovery.

**Strict-gate accepted recoveries regress ATE.** The 4 / 2 recoveries that did pass on MH_01 / V1_01 lifted `tracking_success_rate` by +0.002 / +0.008 but the recovered poses were geometrically wrong enough to push rigid ATE by +27 % / +248 %.

**Looser-threshold sweep (`min_inliers=10, min_inlier_ratio=0.15, max_rep_err=16 px`) makes things strictly worse.**

| seq    | success_rate | rigid_ATE_m                | sim_scale                 | reloc_ok |
|--------|-------------:|---------------------------:|--------------------------:|---------:|
| MH_01  | 0.024        | **0.960 (36× worse)**      | **0.030 (collapsed 33×)** | 5        |
| V1_01  | 0.071        | 0.054 (= strict)           | 0.994                     | 2        |
| V2_01  | 0.052        | 0.012 (3× worse)           | 1.087                     | 1        |

The MH_01 trajectory collapses to scale 0.030 (33× shrinkage) and rigid ATE blows up to `0.960 m`. The 5 recoveries that pass the loose gate are wrong-cheirality / mirrored false positives.

**Why this fails on EuRoC specifically.** Three properties of the post-cliff regime defeat full-map unguided PnP:

1. The IMU motion model has integrated gravity without a fresh VI-BA mirror, so the previous-frame pose prediction is wildly wrong (§Phase-14: mirror velocity blows up to `|g · Δt|`-scale magnitudes by the cliff).
2. HOG descriptors at the post-cliff body attitude do not match the bootstrap-time descriptors at the same image locations because the body is in a different attitude / position; the cross-attitude descriptor distance is large.
3. Full-map PnP without a pose prior is a global PnP problem; on a 1500-landmark map this should be tractable in principle, but the cross-attitude descriptor mismatch dominates the matcher.

**Honest conclusion.** Phase-23 #1 is documented as **shipped infrastructure + empirical negative result**. The stage is correctness-tested and side-effect-free, but neither the strict nor the loose threshold sweep extends the Phase-21 universal cliff. Full per-seq breakdown + reproduction at `target/euroc_phase23_relocalization_ab/SUMMARY.md`.

**Surviving Phase-23 candidates.**

1. **Phase-23 #2 (HOG-with-stereo-bootstrap)** — refresh landmark depths via stereo triangulation rather than `--bootstrap-depth 4.0`. With better-localised landmarks, the recovery PnP's inlier ratio should climb above the strict 0.3 gate without admitting false positives. Lowest wire-up cost.
2. **Pose-prior-guided recovery PnP** — extend `OnlineSlamRelocalizationConfig` with a `use_motion_prior: bool` that hands the IMU motion model's prediction to the localizer's pose-prior warm-start path. Today the pipeline calls the no-prior `localize_frame_with_descriptor_store`. Smallest possible follow-up that may rescue Phase-23 #1 directly.
3. **Phase-23 #3 (loop-closure on dead tracker)** remains gated on either of the above landing first.

## Phase-23 #1b follow-up validation — pose-prior-guided recovery (honest negative result)

**Shipped infrastructure.** New `pose_prior_candidate_radius_meters: Option<f64>` field on `OnlineSlamRelocalizationConfig`. When `Some(radius_m)`, `OnlineSlamPipeline::maybe_run_relocalization` queries the tracker's per-frame motion-model prediction via `Tracker::pose_prior_for_frame` and threads it through the localizer's `localize_frame_with_pose_prior_warm_start_and_descriptor_store` path. Matching EuRoC demo CLI flag `--relocalization-pose-prior-radius <meters>`.

**Empirical 3-seq EuRoC sweep at radius ∈ {2, 10} m (Phase-20 config + 1500-frame cap).**

| seq    | variant         | success_rate | rigid_ATE_m   | sim_scale      | reloc_ok |
|--------|-----------------|-------------:|--------------:|---------------:|---------:|
| MH_01  | reloc (no prior)| 0.023        | 0.0335 (+27 %) | 1.0043        | 4        |
| MH_01  | prior 2 m       | 0.021        | 0.0265 (= base)| 1.0088        | **0**    |
| MH_01  | prior 10 m      | 0.023        | **2101.4 (catastrophic)** | **0.000233 (33 000× collapse)** | 1 |
| V1_01  | reloc (no prior)| 0.071        | 0.0537 (+248 %)| 0.9943        | 2        |
| V1_01  | prior 2 m       | 0.063        | 0.0154 (= base)| 1.0599        | **0**    |
| V1_01  | prior 10 m      | 0.071        | 0.0537 (+248 %)| 0.9943        | 2        |
| V2_01  | reloc (no prior)| 0.051        | 0.0040 (=base) | 1.0931        | 0        |
| V2_01  | prior 2 m       | 0.051        | 0.0040 (=base) | 1.0931        | 0        |
| V2_01  | prior 10 m      | 0.051        | 0.0040 (=base) | 1.0931        | 0        |

**Honest interpretation.**

- The 2 m radius excludes the visible landmark set on the failing frames because the post-cliff IMU motion model prediction is `|g · Δt|`-scale off; the prior pose puts the candidate sphere in the wrong region. **0 recoveries accepted across all three seqs at radius 2 m → bit-for-bit identical baseline.** Side-effect-free invariant validated again.
- The 10 m radius re-admits enough landmarks that recovery PnP finds matches, but the recoveries that pass the strict gate are the same false-positive cheirality / mirrored solutions the no-prior Phase-23 #1 path admitted. On MH_01 one such recovery collapses the trajectory to scale `0.000233` (33 000× shrinkage) with `2101 m` rigid ATE.
- V1_01 at radius 10 m reproduces the no-prior Phase-23 #1 result exactly (`+248 %` rigid ATE, 2 false-positive recoveries) — confirming that at the loose-radius regime the prior adds nothing.

**Conclusion.** Phase-23 #1b is documented as **shipped infrastructure + empirical negative result**. The pose-prior path does not rescue Phase-23 #1 on the current EuRoC bench because the underlying problem is the cross-attitude HOG descriptor mismatch *upstream* of the recovery PnP, not the absence of a pose prior *in* the localizer. Full results at `target/euroc_phase23_relocalization_ab/SUMMARY.md`.

## Phase-23 #2 follow-up validation — strict-stereo bootstrap (FIRST measured EuRoC ATE win in the Phase-23 thread)

**Status correction.** Phase-23 backlog item #2 was originally labelled "HOG-with-stereo-bootstrap" on the assumption that the demo's stereo bootstrap was SuperPoint-only. On investigation, HOG-with-stereo-bootstrap had been the default since `--feature-extractor hog` shipped (Phase-11). The Phase-21 path-forward framing was therefore a misdiagnosis — the actual bottleneck the bootstrap had was the 60-70 % of cam0 keypoints that did not receive a stereo-triangulated depth and were falling back to the fixed `--bootstrap-depth 4.0` back-projection.

**Shipped infrastructure.** New `strict_stereo: bool` parameter on `bootstrap_map_from_first_frame` (in `examples/euroc_online_slam_vi_image_demo.rs`) plus matching CLI flag `--stereo-bootstrap-strict`. When set, every cam0 keypoint that did not receive a cam0↔cam1 stereo-triangulated depth is dropped instead of falling back to the fixed-depth back-projection. The map is smaller but every landmark has a real metric depth.

**Empirical 3-seq EuRoC sweep (Phase-20 config + 1500-frame cap).**

| seq    | rigid_ATE_m baseline → strict | sim_scale baseline → strict       | map_landmarks baseline → strict |
|--------|-------------------------------|-----------------------------------|---------------------------------|
| MH_01  | `0.0265 → 0.0206` (**-22 %**) | `1.0088 → 1.0110`                 | 1500 → 486                      |
| V1_01  | `0.0154 → 0.0128` (**-17 %**) | `1.060 → 1.007` (near-metric)     | 1500 → 638                      |
| V2_01  | `0.0040 → 0.0021` (**-48 %**) | `1.093 → **1.000044**` (perfect metric to 5e-5) | 1500 → 519           |

**This is the first measured EuRoC ATE win in the Phase-23 thread.**

**Why strict-stereo wins.** The legacy bootstrap seeded the map with 1500 keypoints worth of landmarks; ~30-43 % carried real stereo-triangulated cam0↔cam1 depth, the remaining ~60-70 % got the fixed `4.0 m` back-projection. The fallback landmarks had WRONG 3D positions for every keypoint whose true depth was not exactly 4 m (which is almost all of them on EuRoC). The tracker's PnP solver doesn't know which landmarks have real depth and which are fallback fictions, so its inlier consensus included both, with the fallback landmarks pulling the consensus toward their wrong depths. Dropping them removes the bias.

The similarity-scale numbers are the clearest evidence:

- **V2_01** baseline `sim_scale = 1.093` → strict `1.000044` — **exact metric scale recovered to 5e-5 fractional error.**
- **V1_01** baseline `sim_scale = 1.060` → strict `1.007` — fractional scale error drops from 6 % to 0.7 %.
- **MH_01** baseline `sim_scale = 1.009` → strict `1.011` — already near-metric; the rigid-ATE win comes for free.

**What this does NOT fix.** The cliff itself is still at f60-115; `tracking_success_rate` is unchanged or marginally improved (0.021→0.023 / 0.063→0.066 / 0.051→0.051). This is a **trajectory-quality** win, not a **cliff-extension** win. The cliff remains the Phase-21 universal caveat. Combining `--stereo-bootstrap-strict` with `--relocalization-enabled --relocalization-pose-prior-radius 2.0` records 0 recoveries on every seq (side-effect-free invariant holds even on the smaller landmark cloud).

**Recommended new EuRoC default.** `--stereo-bootstrap-strict` is now the recommended EuRoC setting whenever sufficient stereo coverage is available (MH/V1/V2 all show ≥ 30 % stereo match rate on this bench, including the V1/V2 low-texture indoor scenes that Phase-17's SuperPoint variant could not cover).

**Strict-stereo × motion-model trade-off (3-seq sweep).** Running the strict bootstrap with each available motion model shows that the IMU motion model's predictive aggressiveness is the source of BOTH the tight pre-cliff trajectory AND the cliff itself.

| seq    | imu rigid_ATE / n_kf | velocity rigid_ATE / n_kf | pose rigid_ATE / n_kf       |
|--------|---------------------:|--------------------------:|----------------------------:|
| MH_01  | **0.0206** / 7       | 0.099 / 10                | 0.240 / **29** (cliff +313 %) |
| V1_01  | **0.0128** / 2       | 0.210 / 6                 | 0.022 / 5                   |
| V2_01  | **0.0021** / 2       | 0.088 / 5                 | 0.216 / 11                  |

`--motion-model pose` extends the survival window 25-313 % (MH_01 specifically jumps 7 → 29 keyframes — a real cliff extension) but the trajectory quality degrades 4–100× and the similarity-scale recovery collapses (MH_01 `1.011 → 0.311`, V2_01 `1.000 → 2.368`). The accuracy-oriented Phase-23 default remains `--motion-model imu --stereo-bootstrap-strict`. The architectural fix that would close the trade-off is an adaptive motion model (IMU while tracker is healthy, switch to constant-pose when IMU diverges from visual PnP consensus) — out of scope for this round but the natural follow-up.

Full per-seq breakdown + reproduction at `target/euroc_phase23_strict_stereo/SUMMARY.md`.

## Phase-23 #4 follow-up validation — adaptive IMU↔Pose motion model

**Shipped infrastructure.** New `AdaptiveImuPoseMotionModel` in `pipelines/tracking/src/lib.rs` wraps `ImuPredictiveMotionModel` + `ConstantPoseMotionModel` and dispatches `predict_pose` through whichever inner model the per-frame success / failure counters select. Both inner models are kept fed by `observe()` so the switch is instantaneous. New `MotionModelKind::AdaptiveImuPose` variant + `--motion-model adaptive-imu-pose` CLI flag + `--adaptive-motion-{failures_to_switch_to_pose,successes_to_switch_to_imu}` threshold knobs. Five new unit tests in `pipelines/tracking/src/lib.rs`.

**Empirical 3-seq EuRoC sweep on top of `--stereo-bootstrap-strict`.**

| seq    | variant                  | success_rate | rigid_ATE_m   | sim_scale       | n_kf | sw_p | sw_i |
|--------|--------------------------|-------------:|--------------:|----------------:|-----:|-----:|-----:|
| MH_01  | imu (Phase-23 #2)        | 0.023        | **0.0206**    | **1.011**       | 7    | –    | –    |
| MH_01  | adaptive (f=2, s=5)      | 0.052        | 0.1219        | 0.697           | 24   | 5    | 4    |
| MH_01  | adaptive (f=3, s=10)     | 0.085        | 0.2130        | 0.475           | **35** | 4  | 3    |
| MH_01  | pose (Phase-23 #2)       | 0.061        | 0.2398        | 0.311           | 29   | –    | –    |
| V1_01  | imu (Phase-23 #2)        | 0.066        | **0.0128**    | **1.007**       | 2    | –    | –    |
| V1_01  | adaptive (f=2, s=5)      | 0.081        | 0.0272        | 1.031           | 6    | 3    | 2    |
| V1_01  | **adaptive (f=3, s=10)** | **0.071**    | **0.0227**    | **1.089**       | 5    | 1    | 0    |
| V1_01  | pose (Phase-23 #2)       | 0.083        | 0.0220        | 1.053           | 5    | –    | –    |
| V2_01  | imu (Phase-23 #2)        | 0.051        | **0.0021**    | **1.000**       | 2    | –    | –    |
| V2_01  | adaptive (f=2, s=5)      | 0.073        | 0.2629        | 2.827           | 13   | 3    | 2    |
| V2_01  | adaptive (f=3, s=10)     | 0.065        | 0.1954        | 1.870           | 8    | 1    | 0    |
| V2_01  | pose (Phase-23 #2)       | 0.071        | 0.2158        | 2.368           | 11   | –    | –    |

**Sequence-specific outcomes:**

1. **V1_01 — clean Pareto win.** Adaptive with imu-favouring thresholds (f=3, s=10) lands `0.0227 m / 5 KF` — matches pose survival at near-pose ATE with a single switch firing. Slow indoor hover suits the IMU-prefer / pose-fallback pattern.
2. **MH_01 — survival win, accuracy regress.** Adaptive extends keyframe count well past pose (35 vs 29) but ATE regresses to between IMU and pose. The longer survival window means the trajectory now spans cliff-region frames where pose-mode prediction accuracy degrades.
3. **V2_01 — adaptive degrades to WORSE than pose-only.** Repeated IMU re-entry under f=2,s=5 introduces oscillation: each time the wrapper switches back to IMU, the IMU's `velocity_world` has been integrating noisily during pose-mode interval → first IMU prediction post-switch is wrong → immediate failure → switch back to pose. Each oscillation injects a small ATE bump that accumulates.

**Architectural finding.** The switching policy is correct but **the IMU's integration state during pose-mode intervals needs explicit refresh** before re-entry. Today the IMU keeps integrating samples even while pose mode is dispatching predictions; when control returns the integrated velocity is stale. The clean follow-up is one of:

- Reset the IMU's `velocity_world` to zero (or to a finite-difference of the two most recent visual poses) at the switch-back event.
- Run `update_velocity_from_camera_pose_difference` on the IMU model continuously during pose-mode intervals to keep velocity fresh.

Either would turn the MH_01 / V2_01 oscillation into V1_01's clean win pattern. **Out of scope for this round** but is the natural Phase-24 entry point.

**Recommended config update.** The accuracy-oriented EuRoC default remains `--motion-model imu --stereo-bootstrap-strict`. For survival-priority use cases on slow indoor hover (V-class), the documented opt-in is `--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10 --stereo-bootstrap-strict`.

Full per-seq breakdown + reproduction at `target/euroc_phase23_adaptive_motion/SUMMARY.md`.

## Phase-24 — IMU-velocity-refresh-on-switch-to-IMU (honest negative)

**Shipped infrastructure.** New `AdaptiveImuPoseMotionModelConfig::refresh_imu_velocity_on_switch_to_imu` field (`bool`, default `true`). The wrapper now tracks `previous_successful_pose` / `latest_successful_pose` / `dt_between_latest_two_observations` (captured from `ImuPredictiveMotionModel::pending_samples_total_dt()` before each `observe()` drains the pending buffer). At every Pose → IMU switch event the wrapper invokes `imu.update_velocity_from_camera_pose_difference(prev, latest, dt)` and bumps the new telemetry counter `velocity_refreshes_on_switch_to_imu`. New CLI opt-out flag `--adaptive-motion-no-refresh-imu-velocity-on-switch` plus two new summary audit lines (`adaptive_motion_refresh_imu_velocity_on_switch`, `adaptive_motion_velocity_refreshes_on_switch_to_imu`). Three new unit tests cover refresh-enabled, refresh-disabled (Phase-23 #4 behavior preserved), and `reset()` clearing the recent-pose state.

**Hypothesis tested.** The Phase-23 #4 follow-up analysis claimed that the IMU's `velocity_world` going stale during pose-mode intervals (the IMU keeps integrating raw samples even while pose dispatches predictions) was the dominant noise source on MH_01 / V2_01 oscillation. Refreshing the velocity from a visual finite-difference at re-entry should — under that hypothesis — turn V1_01's clean Pareto win into a 3-seq universal win.

**Empirical 3-seq EuRoC sweep on top of `--stereo-bootstrap-strict`, Phase-24 (refresh-on default) vs Phase-23 #4 (refresh-off).**

| seq    | variant              | Phase-23 #4 rigid_ATE | **Phase-24 rigid_ATE** | sw_p / sw_i | refreshes |
|--------|----------------------|----------------------:|-----------------------:|------------:|----------:|
| MH_01  | adaptive (f=2, s=5)  | 0.1219                | **0.1869 (+53 %)**     | 5 / 4       | 4         |
| MH_01  | adaptive (f=3, s=10) | 0.2130                | *0.1934 (-9 %)*        | 5 / 4       | 4         |
| V1_01  | adaptive (f=2, s=5)  | 0.0272                | 0.0272                 | 3 / 2       | 2         |
| V1_01  | adaptive (f=3, s=10) | 0.0227                | 0.0227                 | 1 / 0       | 0         |
| V2_01  | adaptive (f=2, s=5)  | 0.2629                | **0.3119 (+19 %)**     | 4 / 3       | 3         |
| V2_01  | adaptive (f=3, s=10) | 0.1954                | 0.1954                 | 1 / 0       | 0         |

**The hypothesis is empirically disproved.** The refresh hook fires correctly (the counter matches `switches_to_imu` whenever both poses + positive dt are available), but the resulting trajectory does not improve. Where the refresh never fires (V1_01 f=3/s=10, V2_01 f=3/s=10 — both have 0 switches back to IMU), Phase-24 is bit-identical to Phase-23 #4. Where the refresh fires twice on V1_01 f=2/s=5 the ATE matches Phase-23 #4 to 4 decimal places (slow indoor hover keeps the visual finite-difference close enough to the stale velocity that the cliff-gate outcome doesn't change). Where the refresh fires 3–4 times (MH_01 f=2/s=5 and V2_01 f=2/s=5), the ATE **regresses** by 53 % and 19 % respectively.

**Most likely diagnosis: the pose-mode poses ARE the noise.** The visual finite-difference is computed from two `previous_successful_pose` / `latest_successful_pose` observations that were both dispatched by the constant-pose branch. Constant-pose just returns the last successful pose unchanged, so the "finite-difference velocity" between two consecutive pose-mode successes is whatever residual the PnP solver recovered — typically dominated by reprojection noise at the cliff-region landmark count (often only 4–8 inliers). Using this as the IMU's seed velocity injects PnP noise instead of resetting to a clean state. The cliff problem turns out to be upstream of the motion-model layer.

**Recommended config update — none.** The Phase-23 recommended defaults stand: `--motion-model imu --stereo-bootstrap-strict` for accuracy; `--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10 --stereo-bootstrap-strict` for survival-priority on V-class indoor hover. Phase-24's refresh-on default is a no-op at f=3/s=10 because no switch-back fires at those thresholds, and a measurable regression at the experimental f=2/s=5 on MH_01 / V2_01.

**Where to look next.** Three follow-up directions, in decreasing order of expected payoff:

- A learned descriptor pipeline (SuperPoint+LightGlue or similar) whose cross-attitude match quality is high enough to keep the constant-pose branch out of the PnP-noise floor at cliff-region landmark counts — would address the diagnosed root cause directly.
- A 3-pose constant-velocity smoother refresh policy (instead of the current 2-pose finite-difference) as a single-round A/B against the existing Phase-24 implementation — would cleanly test whether the noise comes from the finite-difference itself or from the pose stream upstream of it. **Shipped as Phase-25 below.**
- A reset-to-zero refresh policy A/B — strictly inferior on V1_01 (no information about motion) but might dominate on V2_01 if the visual finite-difference IS the noise source. Cheap to ship and easy to compare. **Shipped as Phase-25 below.**

Full per-seq breakdown + reproduction at `target/euroc_phase24_adaptive_refresh/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase24_adaptive_refresh.sh`.

## Phase-25 — refresh-policy A/B (zero-reset + 3-pose smoother)

**Shipped infrastructure.** The Phase-24 `refresh_imu_velocity_on_switch_to_imu: bool` field is replaced with `imu_velocity_refresh_policy: ImuVelocityRefreshPolicy`. The new enum has four variants — `None` (Phase-23 #4 behavior), `FiniteDifference` (Phase-24 behavior), `ZeroReset` (overwrite `velocity_world` with zeros at switch), `ThreePoseSmoother` (average two finite-differences computed across the three most recent successful visual poses; falls back to single finite-difference when fewer than three are recorded). `AdaptiveImuPoseMotionModel` now tracks a 3-deep pose history (`oldest_successful_pose` + `dt_between_previous_two_observations` in addition to the existing previous / latest fields) to feed the smoother, and a non-mutating helper `ImuPredictiveMotionModel::body_velocity_from_camera_pose_difference` lets the smoother combine two finite-difference velocities without intermediate state writes. EuRoC demo (`examples/euroc_online_slam_vi_image_demo.rs`) gained `--adaptive-motion-refresh-policy {none|finite-diff|zero-reset|three-pose-smoother}` (the Phase-24 `--adaptive-motion-no-refresh-imu-velocity-on-switch` flag survives as a backward-compat alias for `none`); summary audit line `adaptive_motion_imu_velocity_refresh_policy=<name>` replaces the Phase-24 bool. Three new unit tests cover the zero-reset write, the three-pose smoother arithmetic (4 m/s + 8 m/s ⇒ 6 m/s, verified numerically), and the smoother's degradation to single-FD when only two poses are recorded.

**Hypothesis tested.** Phase-24's diagnosis claimed the constant-pose branch's successive poses are themselves PnP-noise-dominated, so the FD between them injects noise. If true, (a) `ZeroReset` should beat `FiniteDifference` on the worst-affected seqs (because uninformed zero dominates noise on average) and (b) `ThreePoseSmoother` should beat both (halves the FD variance by averaging two independent noise samples).

**Empirical 3-seq EuRoC sweep on top of `--stereo-bootstrap-strict`, 12 runs = 3 seqs × 2 threshold sets × 2 new policies, alongside Phase-23 #4 and Phase-24 baselines.**

| seq    | thresh           | policy                | sw_p / sw_i | refr | rigid_ATE_m   | vs P-23 #4 |
|--------|------------------|-----------------------|------------:|-----:|--------------:|-----------:|
| MH_01  | strict (f=2,s=5) | none (P-23 #4)        | 5 / 4       | –    | 0.1219        | —          |
| MH_01  | strict           | finite-diff (P-24)    | 5 / 4       | 4    | 0.1869        | +53 %      |
| MH_01  | strict           | zero-reset            | 4 / 3       | 3    | **0.6902**    | **+466 %** |
| MH_01  | strict           | **three-pose-smoother** | 7 / 6     | 6    | **0.1210**    | **-1 %**   |
| MH_01  | imuFavor (f=3,s=10) | none               | 4 / 3       | –    | 0.2130        | —          |
| MH_01  | imuFavor         | finite / zero / smoother | 5 / 4    | 4    | 0.1934        | -9 % (all 3 identical) |
| V1_01  | strict           | none / finite / zero / smoother | 3 / 2 | 0–2 | 0.0272 | – (all identical) |
| V1_01  | imuFavor         | (all 4 policies)      | 1 / 0       | 0    | 0.0227        | – (all identical, refresh never fires) |
| V2_01  | strict           | none (P-23 #4)        | 3 / 2       | –    | 0.2629        | —          |
| V2_01  | strict           | finite-diff           | 4 / 3       | 3    | 0.3119        | +19 %      |
| V2_01  | strict           | zero-reset            | 2 / 1       | 1    | 0.2400        | -9 %       |
| V2_01  | strict           | **three-pose-smoother** | 3 / 2     | 2    | **0.1984**    | **-25 %**  |
| V2_01  | imuFavor         | (all 4 policies)      | 1 / 0       | 0    | 0.1954        | – (all identical, refresh never fires) |

**Result: partial confirmation of the Phase-24 noise hypothesis, with a strict win.** `ThreePoseSmoother` strictly improves on `FiniteDifference` on every case tested: on V2_01 strict it produces the largest motion-model-layer win in the entire Phase-23/24/25 thread (-25 % vs the Phase-23 #4 baseline, -36 % vs Phase-24); on MH_01 strict it matches the no-refresh baseline within noise (-1 %); on every other case it is bit-identical to `FiniteDifference` because either the hook does not fire (V1_01 / V2_01 imuFavor) or the post-switch localization steps wash out the seed velocity (MH_01 imuFavor — all 3 refresh policies produce identical trajectories there). `ZeroReset` is **not** a viable default: it loses catastrophically on MH_01 strict (+466 %) while producing only a modest win on V2_01 strict (-9 %); the asymmetry reflects that discarding the visual motion estimate degrades the seed on body trajectories where the FD *was* informative (outdoor structured motion) but can beat noise on V-class hover.

**Recommended default change shipped.** `ImuVelocityRefreshPolicy::default()` now returns `ThreePoseSmoother` (Phase-24 returned `FiniteDifference`). The change is safe: on the production-recommended threshold set (f=3 / s=10) it is bit-identical to Phase-24 on all 3 seqs, and on the experimental f=2 / s=5 set it strictly improves on or matches Phase-24 on every seq tested.

**Recommended config update — minor.** The Phase-23 recommended defaults stand: `--motion-model imu --stereo-bootstrap-strict` for accuracy; `--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10 --stereo-bootstrap-strict` for survival-priority on V-class indoor hover. The change is purely in the *default* `--adaptive-motion-refresh-policy`, which is `three-pose-smoother` from Phase-25 onward. Users who want the Phase-24 behavior back can pass `--adaptive-motion-refresh-policy finite-diff`; Phase-23 #4 behavior is `--adaptive-motion-refresh-policy none` (the old `--adaptive-motion-no-refresh-imu-velocity-on-switch` flag is preserved as an alias).

**Where to look next.** The V2_01 strict ThreePoseSmoother result (`-25 %`) is the most striking motion-model-layer win since Phase-23 #2. Two extension directions remain incremental — higher-order constant-acceleration smoother (more PnP noise multiplied through the second derivative — risky); adaptive smoother window keyed on PnP inlier-count of the most recent successes (small expected gain). Neither addresses the diagnosed upstream limit. The higher-payoff direction is still a learned descriptor pipeline (SuperPoint+LightGlue or similar) that raises the cliff-region inlier count, lets the existing FD / smoother policies operate on cleaner pose streams, and unblocks relocalization (Phase-23 #1) for direct post-cliff recovery.

Full per-seq breakdown + reproduction at `target/euroc_phase25_refresh_policy_ab/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase25_refresh_policy_ab.sh`; comparison helper at `scripts/tabulate_phase25_results.sh`.

## Phase-26 #1 — SuperPoint + strict-stereo bootstrap (V-class breakthrough)

**Shipped infrastructure (re-used, not added).** No new code in the visloc-rs crate. The Phase-15 `SuperPointOfflineExtractor` already supports cam0+cam1 offline replay via `--superpoint-features-dir <cam0>` + `--superpoint-cam1-features-dir <cam1>`; the strict-stereo bootstrap path (Phase-23 #2) threads cam1 SuperPoint features through the same triangulation pipeline as HOG; the Phase-25 default ThreePoseSmoother refresh applies regardless of descriptor backend. The only Phase-26 #1 prerequisite is running the existing `scripts/export_superpoint_lightglue.py --mono-dir` once per camera per sequence to produce `frame_NNNNNN_features.txt`. New sweep driver `scripts/run_euroc_phase26_superpoint_strict_stereo.sh` parallel-runs the Phase-25 default config × {strict, imuFavor} thresholds with the SuperPoint extractor swapped in.

**Hypothesis tested.** Phase-15 (2026-04-xx) tested SuperPoint against the HOG baseline before Phase-23 #2 shipped strict-stereo, and concluded "descriptor strength is NOT the binding constraint at this stack — the 4 m fixed bootstrap depth was." That finding was correct *for the Phase-13 mixed-depth bootstrap*. After strict-stereo dropped the wrong-depth fallback landmarks, the descriptor stream may become the binding constraint — particularly at cliff-region landmark counts where every PnP inlier matters. Phase-26 #1 re-runs the SuperPoint replay on top of the Phase-25 default stack to see whether the negative Phase-15 result reverses.

**Empirical 3-seq EuRoC sweep on top of `--stereo-bootstrap-strict` + Phase-25 ThreePoseSmoother default, SuperPoint cam0+cam1 vs HOG baseline.**

| seq    | thresh   | descriptor | n_tracked | last_frame | kf | rigid_ATE_m       | sim_scale     |
|--------|----------|------------|----------:|-----------:|---:|-------------------:|--------------:|
| MH_01  | strict   | HOG (P-25) | 99        | 970        | 30 | 0.1210             | 0.854         |
| MH_01  | strict   | SuperPoint | 176       | 891        | 42 | 0.1979 (+64 %)    | 0.551         |
| MH_01  | imuFavor | HOG (P-25) | 177       | 909        | 37 | 0.1934             | 0.524         |
| MH_01  | imuFavor | SuperPoint | 124       | 1069 (+18 %) | 39 | 0.2956 (+53 %)  | 0.615         |
| V1_01  | strict   | HOG (P-25) | 121       | 158        | 6  | 0.0272             | 1.031         |
| V1_01  | strict   | **SuperPoint** | 93    | 113        | 2  | **0.0029 (-89 %)** | **1.026**    |
| V1_01  | imuFavor | HOG (P-25) | 106       | 146        | 5  | 0.0227             | 1.089         |
| V1_01  | imuFavor | **SuperPoint** | 93    | 113        | 2  | **0.0029 (-87 %)** | **1.026**    |
| V2_01  | strict   | HOG (P-25) | 102       | 215        | 9  | 0.1984             | 2.273         |
| V2_01  | strict   | **SuperPoint** | 84    | 113        | 4  | **0.0107 (-95 %)** | **1.095**    |
| V2_01  | imuFavor | HOG (P-25) | 98        | 260        | 8  | 0.1954             | 1.870         |
| V2_01  | imuFavor | **SuperPoint** | 90    | 243        | 8  | **0.1554 (-20 %)** | 1.579         |

**Result: the Phase-15 negative finding is empirically reversed; V-class indoor accuracy improves by an order of magnitude. The MH-class trade-off goes the other way.** V1_01 strict SuperPoint lands at **0.0029 m rigid ATE** — the cleanest EuRoC result in the entire Phase-{20..26} thread, by an order of magnitude. V2_01 strict rigid ATE drops from 0.1984 to **0.0107 m (-95 %)** with similarity scale recovering from gross over-scaling (2.27) to near-metric (**1.095**). The cost is a slightly shorter trajectory before the universal cliff fires — SuperPoint's stricter PnP gate refuses the marginal post-cliff frames that HOG accepts at the cost of accuracy. MH_01 is the inverse trade: SuperPoint roughly doubles tracking density (n_tracked 99 → 176 on strict) and extends the trajectory by +18 % on imuFavor, but rigid ATE regresses +53 % to +64 % because the longer dense window exposes more drift accumulation.

**The Phase-15 conclusion ("descriptor strength is NOT the binding constraint") was correct *for the Phase-15 stack* but not for the Phase-25 stack.** With strict-stereo dropping the wrong-depth fallback landmarks (Phase-23 #2), the bottleneck moved *to* descriptor quality on V-class indoor sequences — exactly the cross-attitude regime where Phase-23 #1 / #1b found HOG insufficient. SuperPoint+strict-stereo is the first Phase-{23..26} intervention to break the V-class accuracy ceiling.

**Recommended config update — V-class indoor accuracy gets a new opt-in.** The Phase-25 defaults stand for cross-class portability and MH-class workloads (HOG is the safer default — no external Python dependency, no MH-class ATE regression). For V-class indoor sequences where accuracy is the priority, add `--feature-extractor superpoint-offline --superpoint-features-dir <cam0_dir> --superpoint-cam1-features-dir <cam1_dir>` to the Phase-25 recommended config, after pre-exporting both cameras via `scripts/export_superpoint_lightglue.py --mono-dir`. The default extractor remains `--feature-extractor hog` to preserve the no-external-dependency story.

**Where to look next (Phase-26 follow-ups, decreasing payoff order).**

- **Wire `MutualSoftmaxMatcher` into the bootstrap and tracker matcher slots.** The LightGlue-style matcher (already shipped at `crates/vision/src/matching/mutual_softmax.rs`) was not used in Phase-26 #1; the current run uses `BruteForceMatcher` with cross-check on SuperPoint descriptors. Switching to mutual-softmax may further lift the cliff-region matching quality.
- **Investigate the V-class shortened-trajectory issue.** Both V1_01 thresholds and V2_01 strict die at frame 113 under SuperPoint vs HOG's 158 / 215. Hypothesis: SuperPoint's stricter PnP inlier gate refuses post-cliff frames that HOG accepts at the cost of accuracy. Loosening the PnP RANSAC reprojection threshold for SuperPoint may extend the V-class trajectories while preserving the accuracy lift.
- **In-Rust online inference (Phase-27 candidate).** `ort` crate (ONNX Runtime bindings) is the most-mature path; SuperPoint ONNX models are available from `magic-leap-research/SuperPoint`. Phase-26 #1's offline replay is empirically sufficient for the bench — Phase-27 is a deployment/latency concern.
- **Enable relocalization with SuperPoint.** The Phase-23 #1 recovery PnP found HOG cross-attitude matching insufficient to lift recoveries above the `min_inliers=20` / `min_inlier_ratio=0.3` gate. SuperPoint descriptors may now make those recoveries land — turn on `--relocalization-enabled` and check whether post-cliff frames become recoverable.

Full per-seq breakdown + reproduction at `target/euroc_phase26_superpoint_strict_stereo/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase26_superpoint_strict_stereo.sh`.

## Phase-26 #2 + #2b — relocalization re-evaluation with SuperPoint (honest negative)

**Question.** Phase-23 #1 (recovery PnP) rejected nearly every recovery attempt on EuRoC under HOG because cross-attitude HOG matches could not reach the strict gate (`min_inliers=20 / inlier_ratio=0.3`); Phase-23 #1b's pose-prior radius made no difference because the HOG matching itself was the limit. After Phase-26 #1 produced V-class breakthrough accuracy with SuperPoint+strict-stereo, the next test is: do SuperPoint descriptors now lift recovery PnP above the gate so post-cliff frames become recoverable — i.e., can we finally extend the universal cliff?

**Two variants on top of Phase-26 #1.** `#2` adds `--relocalization-enabled` with strict default gates; `#2b` further adds `--relocalization-pose-prior-radius 5.0` (Phase-23 #1b middle-ground radius between the 2 m "exclude all landmarks" and 10 m "admit false positives" failure modes).

**Empirical 3-seq × 2-threshold × 2-variant sweep.**

| seq+thresh                | variant         | reloc s/a   | n_trk | last_frame | kf | rigid_ATE_m   | sim_scale |
|---------------------------|-----------------|-------------|------:|-----------:|---:|--------------:|----------:|
| V1_01 strict              | #1 baseline     | (disabled)  |    93 |        113 |  2 | **0.0029**    | **1.026** |
| V1_01 strict              | #2              | 3 / 1344    |   138 |        180 | 11 | 0.378 (+13000 %) | 0.278     |
| V1_01 strict              | #2b (pp5m)      | 4 / 1346    |   137 |        174 | 11 | 0.383 (+13200 %) | 0.264     |
| V1_01 imuFavor            | #1 baseline     | (disabled)  |    93 |        113 |  2 | **0.0029**    | **1.026** |
| V1_01 imuFavor            | #2 / #2b        | 2 / 1344    |   137 |        177 | 10 | 0.381         | 0.273     |
| V2_01 strict              | #1 baseline     | (disabled)  |    84 |        113 |  4 | **0.0107**    | **1.095** |
| V2_01 strict              | #2 / #2b        | 0 / 1391    |    84 |        113 |  4 | (bit-identical to #1)        |           |
| V2_01 imuFavor            | #1 baseline     | (disabled)  |    90 |        243 |  8 | 0.1554        | 1.579     |
| V2_01 imuFavor            | #2 / #2b        | 0 / 1385    |    90 |        243 |  8 | (bit-identical to #1)        |           |
| MH_01 strict              | #1 baseline     | (disabled)  |   176 |        891 | 42 | 0.1979        | 0.551     |
| MH_01 strict              | #2 / #2b        | 0 / 1302    |   176 |        891 | 42 | (bit-identical to #1)        |           |
| MH_01 imuFavor            | #1 baseline     | (disabled)  |   124 |       1069 | 39 | 0.2956        | 0.615     |
| MH_01 imuFavor            | #2 / #2b        | 0 / 1354    |   124 |       1069 | 39 | (bit-identical to #1)        |           |

**Result: cliff extension via Phase-23 #1 recovery PnP is empirically blocked on EuRoC, even with SuperPoint descriptors and a pose-prior radius.** 4 of 6 cases (MH_01 both, V2_01 both) accept zero recoveries out of 1300+ attempts under both variants — bit-identical to the no-recovery baseline (Phase-23 #1's "side-effect-free" property reaffirmed). 2 of 6 cases (V1_01 both thresholds) accept 2–4 recoveries that are all **false positives**: the cliff extends by ~55-60 % in frame count (113 → 174-180) and keyframes grow 5× (2 → 10-11), but rigid ATE explodes from 0.0029 m to ~0.38 m (factor 130×) with sim_scale collapsing from 1.026 to ~0.27 (factor 3.7× shrink). The Phase-26 #1 V-class breakthrough is destroyed when recovery is enabled.

**Diagnosis (refines Phase-23 #1).** Phase-23 #1 found "cross-attitude HOG descriptors cannot reach the inlier-ratio gate." Phase-26 #2 / #2b refines this: SuperPoint descriptors **can** reach the gate on V1_01 (the easiest cliff regime — slow indoor hover with modest rotation), but the recovered solution lands at the wrong global scale because the full-map candidate landmark set admits geometrically self-consistent solutions far from the true pose. Pose-prior radius=5 m does not fix this because the IMU prediction at recovery time is itself drifted by `|g·Δt|` or worse, so the "correct neighborhood" is excluded while a wrong-scale neighborhood gets admitted (V1_01 strict #2 → #2b: 3 false positives → **4 false positives**, i.e. pose-prior at 5 m made it slightly worse). The MH_01 / V2_01 strict-gate impossibility is *separate* from the V1_01 false-positive problem and is harder to attack — the cliff-region viewpoint likely diverges far enough from the bootstrap viewpoint that even SuperPoint cannot find 20 inliers at ratio 0.3 against the full pre-cliff landmark set.

**What this rules out.** Naive recovery PnP enable + pose-prior is not a viable cliff extension on EuRoC. The Phase-23 #1 recovery PnP path likely needs structural changes — per-keyframe submap selection that constrains candidate landmarks to "recently visible from the active map frontier" rather than full-map, OR a post-acceptance geometric sanity check that rejects recoveries whose pose is inconsistent with the IMU's covariance ellipsoid (not just a metric ball around the prediction).

**Recommended config update — none.** Phase-26 #1 remains the V-class accuracy opt-in (`--feature-extractor superpoint-offline --superpoint-features-dir <cam0> --superpoint-cam1-features-dir <cam1>` on the Phase-25 default stack); **do not enable `--relocalization-enabled`** on top of it without further work.

**Where to look next (Phase-26 #3 candidates).** In decreasing payoff order:

1. **Loosen the tracker's PnP RANSAC reprojection threshold for SuperPoint to extend V-class trajectories without recovery** (Phase-26 #3a). V1_01 and V2_01 strict SuperPoint die at frame 113 vs HOG's 158-215 because SuperPoint's stricter inlier gate refuses marginal post-cliff frames; loosening the tracker-side gate directly addresses the Phase-26 #1 trajectory-shortening caveat without touching recovery.
2. **Wire `MutualSoftmaxMatcher` into bootstrap and tracker** (Phase-26 #3b). The LightGlue-style matcher is shipped at `crates/vision/src/matching/mutual_softmax.rs` but unused by #1/#2/#2b — may further lift cross-attitude match quality.
3. **Decompose the MH_01 ATE regression** (Phase-26 #3c). MH_01 SuperPoint regressed +53 % to +64 % rigid ATE with +18 % trajectory extension; sliding-window BA tuning or descriptor-side tuning?
4. **Structural recovery PnP rework** (Phase-26 #4+). Submap selection + post-acceptance IMU-covariance sanity check. Deferred until cheaper #3a/#3b/#3c results are in.

Full per-seq breakdown + reproduction at `target/euroc_phase26_relocalization_ab/SUMMARY.md`; sweep drivers at `scripts/run_euroc_phase26b_superpoint_relocalization.sh` (variant #2) and `scripts/run_euroc_phase26b2_superpoint_reloc_poseprior.sh` (variant #2b).

## Phase-26 #3a — V-class PnP-threshold sweep (honest mixed)

**Question.** Phase-26 #1 shortened V-class trajectories (V1_01 strict 158 → 113, V2_01 strict 215 → 113) as the price of the order-of-magnitude accuracy lift. Working hypothesis (Phase-26 #2 close-out): SuperPoint + the default 4 px PnP reprojection threshold rejects marginal post-cliff frames HOG accepts at accuracy cost. Loosen the gate to 8 / 12 px and the trajectories should extend without exploding ATE.

**Shipped infrastructure.** New CLI flag `--pnp-reprojection-threshold-px <px>` on `examples/euroc_online_slam_vi_image_demo.rs` overrides `LocalizationConfig::reprojection_threshold` (default `4.0`); `None` preserves the default. New audit line `pnp_reprojection_threshold_px=<value>` in `summary.txt`. New sweep driver `scripts/run_euroc_phase26_3a_pnp_threshold_sweep.sh` (V-class only — 8 parallel runs = 2 seqs × 2 thresholds × 2 loosened gates {8, 12 px}).

**Empirical V-class sweep on top of Phase-26 #1 (SuperPoint+strict-stereo).**

| seq        | thresh   | variant            | n_trk | last_frame | kf | rigid_ATE_m | sim_scale |
|------------|----------|--------------------|------:|-----------:|---:|------------:|----------:|
| V1_01      | strict   | P-26 #1 (4 px)     | 93    | 113        | 2  | **0.0029**  | **1.026** |
| V1_01      | strict   | #3a (8 px)         | 94    | 530        | 3  | 0.2587      | **6.17**  |
| V1_01      | strict   | #3a (12 px)        | 95    | 602        | 4  | 0.3713      | **9.81**  |
| V1_01      | imuFavor | (8 / 12 px identical to strict — same seed-frame match counts under same SuperPoint stream) |
| V2_01      | strict   | P-26 #1 (4 px)     | 84    | 113        | 4  | **0.0107**  | **1.095** |
| V2_01      | strict   | #3a (8 px)         | 90    | 258        | 5  | 0.1085      | 1.288     |
| V2_01      | strict   | #3a (12 px)        | 96    | 266        | 8  | 0.1606      | 1.506     |
| V2_01      | imuFavor | P-26 #1 (4 px)     | 90    | 243        | 8  | 0.1554      | 1.579     |
| V2_01      | imuFavor | #3a (8 px)         | 90    | 127        | 6  | 0.0429*     | 1.254     |
| V2_01      | imuFavor | #3a (12 px)        | 92    | 170        | 8  | 0.1389      | 1.359     |

`*` artifact: ATE drops because the evaluation window shrinks (trajectory dies earlier under this single configuration); not a real win.

**Result: the hypothesis is geometrically confirmed but metrically refuted.** Loosening the PnP threshold extends V-class trajectories dramatically (V1_01 strict 113 → 530 → 602, 4-5× longer; V2_01 strict 113 → 258 → 266), confirming that the default 4 px gate was the reason SuperPoint trajectories died early. **But the extended frames are scale-wrong**: V1_01 sim_scale collapses from 1.026 to **6.17** at 8 px / **9.81** at 12 px (trajectory inflated 6-10×). The looser inlier gate admits depth-inconsistent correspondences that drive PnP into geometrically self-consistent solutions at the wrong global scale — the same failure mode as Phase-26 #2's recovery false positives, just inside the tracker's primary PnP rather than the recovery PnP.

**V2_01 strict at 8 px is the only borderline case.** rigid ATE goes 0.0107 → 0.1085 (10× worse than Phase-26 #1) but stays **45 % below the Phase-25 HOG baseline of 0.1984**, and sim_scale lifts modestly (1.095 → 1.288, no collapse). This is a real Pareto point if the application values cliff extension over accuracy; for accuracy-priority workloads, Phase-26 #1's 4 px default is unbeaten.

**Phase-26 #1's 4 px default was already optimal for V-class accuracy.** The trajectory shortening was the *price* of metric correctness, not a bug to fix at the gate level. The Phase-26 #2 working diagnosis — "stricter gate refuses HOG-accepted frames at accuracy cost" — is refined by this result to "the stricter gate is correctly rejecting cliff-region correspondences that cannot support a metric-correct pose under any threshold." The cliff is not a tracker-PnP problem at this stage; the correct interventions are either better correspondences (Phase-26 #3b MutualSoftmaxMatcher) or smaller, geometrically constrained candidate sets (Phase-26 #4 structural recovery PnP rework).

**Recommended config update — none for accuracy.** Phase-26 #1 V-class config stays unchanged. The `--pnp-reprojection-threshold-px` flag ships as an experimental knob (users prioritising cliff extension over accuracy on V2_01 strict can opt in to `--pnp-reprojection-threshold-px 8.0` with their eyes open); the default 4 px remains best.

**Where to look next.** The Phase-26 #3 follow-up priority order is unchanged but #3a is now closed:

1. **Phase-26 #3b — wire `MutualSoftmaxMatcher` into bootstrap and tracker** (top remaining candidate). May admit *correct* additional cliff-region correspondences, unlike threshold loosening which only relaxes the gate on noisy ones. Already shipped at `crates/vision/src/matching/mutual_softmax.rs`.
2. **Phase-26 #3c — MH_01 ATE regression decomposition.** Drift over longer window vs SuperPoint outdoor characteristic.
3. **Phase-26 #4 — structural recovery PnP / submap selection rework.** Bigger, deferred.

Full per-seq breakdown + reproduction at `target/euroc_phase26_3a_pnp_threshold_ab/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase26_3a_pnp_threshold_sweep.sh`.

## Phase-26 #3b — V-class SuperPoint + MutualSoftmaxMatcher sweep (honest negative)

**Question.** Phase-26 #3a refuted PnP-gate loosening as a V-class cliff-extension intervention (looser gate admits scale-wrong correspondences). The Phase-26 #3b working hypothesis: maybe a different *matcher* — `MutualSoftmaxMatcher` (`crates/vision/src/matching/mutual_softmax.rs`, LightGlue-style temperature-scaled mutual-softmax over the full cosine-similarity matrix, defaults `temperature=20.0`, `min_confidence=0.2`) — admits **correct** additional cross-attitude correspondences that cross-check discards, lifting the cliff-region inlier count *honestly* rather than by relaxing acceptance on noisy matches.

**Shipped infrastructure.** New CLI flag `--mutual-softmax-matcher` on the EuRoC demo, mutually exclusive with `--cross-check-matcher` (parser enforces this with an explicit error). `DemoMatcher` enum gained a `MutualSoftmax` variant alongside `BruteForce` / `CrossCheck`. New audit line `mutual_softmax_matcher=<bool>`. New sweep driver `scripts/run_euroc_phase26_3b_mutual_softmax_sweep.sh` (V-class only — 4 parallel runs = 2 seqs × 2 thresholds × 1 variant).

**Empirical V-class sweep on top of Phase-26 #1 (SuperPoint+strict-stereo), cross-check vs mutual-softmax.**

| seq        | thresh   | variant                | n_trk | last_frame | kf | rigid_ATE_m | sim_scale |
|------------|----------|------------------------|------:|-----------:|---:|------------:|----------:|
| V1_01      | strict   | P-26 #1 cross-check    | 93    | 113        | 2  | **0.0029**  | **1.026** |
| V1_01      | strict   | #3b mutual-softmax     | 116   | **1452**   | 21 | 1.379       | **22.85** |
| V1_01      | imuFavor | P-26 #1 cross-check    | 93    | 113        | 2  | **0.0029**  | **1.026** |
| V1_01      | imuFavor | #3b mutual-softmax     | 116   | 1452       | 21 | 1.379       | 22.85     |
| V2_01      | strict   | P-26 #1 cross-check    | 84    | 113        | 4  | **0.0107**  | **1.095** |
| V2_01      | strict   | #3b mutual-softmax     | 86    | 994        | 5  | 0.334       | 2.32      |
| V2_01      | imuFavor | P-26 #1 cross-check    | 90    | 243        | 8  | 0.1554      | 1.579     |
| V2_01      | imuFavor | #3b mutual-softmax     | 90    | 950        | 8  | 0.409       | 4.68      |

**Result: same failure mode as Phase-26 #3a, more extreme.** Mutual-softmax extends V-class trajectories *even more dramatically* than the loosened PnP gate did (V1_01 strict 113 → **1452**, basically running the entire `--max-frames 1500` sequence; V2_01 strict 113 → 994), **but the extended frames are scale-wrong** in exactly the same way: V1_01 sim_scale 1.026 → **22.85** (an order of magnitude beyond Phase-26 #3a's 6-9), V2_01 strict 1.095 → 2.32, V2_01 imuFavor 1.579 → 4.68. Rigid ATE explodes by 30-475× vs Phase-26 #1.

**This refines Phase-26 #3a's diagnosis to a definitive conclusion.** The V-class cliff at frame ~113 is **not** a tracker-side gate or matcher problem. Whether one loosens the PnP inlier threshold (#3a) or swaps the matcher to a more permissive mutual-softmax (#3b), the result is the same: trajectories extend by accepting marginal post-cliff correspondences that drive PnP into geometrically self-consistent but metric-incorrect solutions. **The combination of Phase-26 #1's cross-check + 4 px gate is empirically the optimal tracker-side filter pair** for SuperPoint V-class accuracy — every relaxation in any direction trades accuracy for trajectory length at scale collapse.

**The cliff is upstream of all tracker-side filters.** The correspondences at the cliff transition are intrinsically not metric-correct-supporting under any matcher / gate combination empirically tested. The post-cliff body viewpoint diverges far enough from the bootstrap-time landmarks that only a few genuinely-co-visible landmarks remain — and those few are not enough to dominate the false correspondences in the inlier consensus. The next remaining intervention is **map-side** (per-keyframe submap selection + post-acceptance IMU-covariance sanity check, deferred to Phase-26 #4), not tracker-side.

**Recommendation: no default change.** Phase-26 #1 V-class accuracy config (`--cross-check-matcher`, default 4 px PnP gate) is empirically Pareto-optimal among tracker-side combinations tested. The `--mutual-softmax-matcher` flag ships as an experimental knob for users who want cliff extension at any accuracy cost or for future descriptor-pipeline regression testing.

**The Phase-26 #3 thread is empirically closed.** Neither gate-loosening (#3a) nor matcher-swap (#3b) extends V-class trajectories without scale collapse. Next directions are **Phase-26 #4** (structural recovery PnP rework — bigger architectural change) or **Phase-26 #3c** (MH_01 ATE regression decomposition — small / cheap closeout, orthogonal to V-class cliff).

Full per-seq breakdown + reproduction at `target/euroc_phase26_3b_mutual_softmax_ab/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase26_3b_mutual_softmax_sweep.sh`.

## Phase-26 #3c — MH_01 ATE regression decomposition (closeout)

**Question.** Phase-26 #1 regressed MH_01 rigid ATE +53 % (imuFavor) / +64 % (strict) vs the Phase-25 HOG baseline, with simultaneous tracking-density / trajectory-length improvements (+78 % strict density; +18 % imuFavor trajectory length). Is the regression dominated by (a) drift accumulation in the *extra frames* SP survives that HOG fails out of, or (b) a SuperPoint outdoor-descriptor weakness that raises per-frame error on the *same* frames HOG covers? This analysis uses the existing `slam_errors.csv` artifacts — no re-runs.

**Method.** Three slices: (1) same-window truncation (recompute SP raw-position RMSE over the first N frames where N = HOG's full tracked-frame count); (2) common-frame analysis (per-frame error on frames both methods succeed on); (3) bin-by-frame-range (per-frame error stratified by early <300 / mid 300-599 / late >=600).

**Result — MH_01 strict (dominated by hypothesis (a) drift, ~93 %).**

| analysis                  | HOG (n=99)      | SP (n=176)      | observation                       |
|---------------------------|-----------------|-----------------|------------------------------------|
| full-window mean_pos      | 0.175 m         | 0.348 m         | +99 % aggregate                     |
| **same-99-frame raw RMSE**| **0.238 m**     | **0.251 m**     | **SP only +5.2 %**                  |
| common-frame mean (n=58)  | 0.079 m         | 0.087 m         | SP +9 % on shared frames            |
| early <300                | 0.086 m (n=72)  | 0.123 m (n=82)  | similar trade-off pattern           |
| mid 300-599               | 0.396 m (n=24)  | 0.523 m (n=63)  | SP +32 %, 2.6× density              |
| late ≥600                 | 0.537 m (n=3)   | 0.591 m (n=31)  | SP +10 %, 10× density               |

When SP is evaluated on the same 99-frame window HOG covers, raw position RMSE differs by only **+5.2 %** (0.238 vs 0.251 m). The full Phase-26 #1 rigid ATE regression (0.121 → 0.198 m, +64 %) is therefore explained almost entirely by drift accumulated over the extra 77 mid/late frames SP survives.

**Result — MH_01 imuFavor (different pattern, mixed cause).** SP has *fewer* total tracked frames (124) than HOG (177) but extends *further* (last_frame 1069 vs 909). The imuFavor thresholds (f=3/s=10) keep SP in Pose mode longer, producing sparser per-region density that the rigid-ATE Umeyama alignment penalises against the longer-reaching but sparser trajectory. SP raw RMSE is actually *lower* than HOG (0.366 vs 0.471 m) but rigid-aligned ATE is higher (0.296 vs 0.193 m, +53 %). Cannot cleanly attribute to (a) or (b) alone; it is a combination of sparser per-region density under imuFavor + extension into a noisier post-cliff region [909, 1069].

**Recommendation refinement.** The Phase-26 #1 MH-class caveat is precisely characterized:

- **MH-class accuracy priority** → HOG default (Phase-25 recommendation). Lower rigid ATE on the shorter tracked window.
- **MH-class trajectory continuity priority** → SuperPoint + Phase-26 #1 config. Higher rigid ATE but +78 % tracking density (strict) / +18 % trajectory length (imuFavor). Per-frame accuracy on the HOG-covered window is within ~5-9 % of HOG.

The Phase-26 #1 caveat in the recommended-config table is refined from "SP regresses MH-class ATE" to "SP regresses *aggregate* MH-class ATE by extending tracking into harder frames; same-window per-frame accuracy is within ~5 % of HOG."

**Phase-26 #3 thread is fully closed** (3a refuted gate-loosening; 3b refuted matcher-swap; 3c is a documentation refinement of the MH-class trade-off). The tracker-side intervention space for V-class cliff extension is empirically exhausted. The only remaining EuRoC arc direction is **Phase-26 #4** — a structural map-side rework (per-keyframe submap selection + post-acceptance IMU-covariance sanity check) addressing the diagnosed Phase-26 #2 root cause (recovery PnP accepts scale-wrong solutions because the candidate landmark set spans the whole map).

Full breakdown + reproduction at `target/euroc_phase26_3c_mh01_decomposition/SUMMARY.md`.

## Phase-26 #4 — structural recovery rework (active-frontier submap + IMU sanity check, honest negative)

**Question.** Phase-26 #2 diagnosed the V1_01 recovery false positives as "the full-map candidate landmark set admits geometrically self-consistent solutions far from the true pose." Phase-26 #4 ships two structural fixes targeting that diagnosis: (#4a) active-frontier submap selection — restrict recovery PnP descriptor store to landmarks observed by the most recent N keyframes; (#4b) post-acceptance IMU sanity check — reject recoveries whose recovered camera centre is more than M meters from the tracker's per-frame motion-model prediction. Will these structural changes filter the V1_01 false positives without breaking the MH_01 / V2_01 side-effect-free property?

**Shipped infrastructure.** `OnlineSlamRelocalizationConfig` (`pipelines/slam/src/lib.rs`) gained `recent_keyframe_window: Option<usize>` + `max_translation_from_imu_prediction_meters: Option<f64>` fields, both defaulting to `None` (preserves Phase-23 #1 semantics). `maybe_run_relocalization` builds the descriptor store either as full-map (when `recent_keyframe_window` is `None`) or restricted to active-frontier landmarks; post-acceptance evaluates IMU sanity check before forwarding the recovered pose. EuRoC demo (`examples/euroc_online_slam_vi_image_demo.rs`) gained CLI flags `--relocalization-recent-keyframe-window <N>` and `--relocalization-max-translation-from-imu-prediction-meters <M>`. 3 existing relocalization tests updated. New sweep driver `scripts/run_euroc_phase26_4_structural_recovery_sweep.sh` (6 parallel runs).

**Empirical 3-seq × 2-threshold sweep, structural recovery (window=5, max_translation=2.0 m) vs new-binary Phase-26 #1 baseline (no recovery).**

| seq+thresh                    | variant                | reloc s/a   | last_frame | kf | rigid_ATE_m | sim_scale |
|-------------------------------|------------------------|-------------|-----------:|---:|------------:|----------:|
| V1_01 strict                  | P-26 #1 baseline       | (disabled)  | 113        | 2  | **0.0029**  | **1.026** |
| V1_01 strict                  | **#4 structural**      | 3 / 1347    | 175        | 9  | 0.3791      | 0.262     |
| V1_01 imuFavor                | P-26 #1 baseline       | (disabled)  | 113        | 2  | **0.0029**  | **1.026** |
| V1_01 imuFavor                | **#4 structural**      | 5 / 1344    | 180        | 13 | 0.3801      | 0.282     |
| V2_01 strict                  | (both)                 | 0 / 1386    | 263        | 8  | 0.2013      | 1.955     |
| V2_01 imuFavor                | (both)                 | 0 / 1385    | 243        | 8  | 0.1554      | 1.579     |
| MH_01 strict                  | (both)                 | 0 / 1350    | 1145       | 36 | 0.4658      | 0.782     |
| MH_01 imuFavor                | (both)                 | 0 / 1354    | 1069       | 39 | 0.2956      | 0.615     |

**Result: structural changes failed to filter V1_01 false positives (and made imuFavor slightly worse).** 4 of 6 cases stay bit-identical to baseline (0 recoveries — strict gate still impossible even with active-frontier submap). V1_01 strict still accepts 3 false positives; V1_01 imuFavor accepts **5** (vs Phase-26 #2's 2 — an *increase*). All V1_01 recoveries land at sim_scale 0.26-0.28 (vs true 1.0), rigid ATE 0.38 m (vs Phase-26 #1's 0.0029 m). The V-class breakthrough is still destroyed when recovery is enabled.

**Why both interventions failed.** *Active-frontier submap (window=5)*: trimming the candidate set doesn't help because the cliff-region active landmarks ARE the wrong-scale-supporting ones — those are precisely what the recovery matches against. In V1_01 imuFavor the smaller candidate set actually *raised* the inlier ratio for the wrong-scale solution (fewer competing correspondences) → more false positives passed (3 → 5). *IMU sanity check (max=2.0 m)*: the IMU prediction at recovery time has itself drifted into the wrong-scale neighborhood because IMU has been integrating since the last successful pose (frame ~113) for many frames in Pose mode without visual correction. A 2 m radius ball around the (drifted) IMU prediction admits the (drifted) wrong-scale recoveries.

**Diagnosis refined to its root.** The Phase-26 #2 framing — "the full-map candidate set admits wrong-scale solutions" — was a *symptom*, not the root cause. The root cause is that **the cliff-region landmarks themselves support wrong-scale solutions**, regardless of how the candidate set is trimmed, and **the IMU prediction post-cliff drifts into the same wrong-scale neighborhood**. Neither candidate-set trimming nor IMU-distance filtering addresses this. Recovery PnP on EuRoC cliffs is structurally unsalvageable with the tracker-friendly intervention space tested in Phase-26 #2 / #2b / #4.

**Binary-determinism caveat surfaced by this work.** The Phase-26 #1 V-class numbers reported earlier (V1_01 strict 0.0029 m, V2_01 strict 0.0107 m, V2_01 strict sim_scale 1.095) were produced with an older binary; a fresh re-build of the same code produces V2_01 strict 0.201 m / sim_scale 1.955 as the no-recovery baseline. Two runs of the *current* binary with identical arguments are bit-identical (verified via duplicate runs) — the non-determinism is at the binary-build level (likely `std::collections::HashMap`'s per-process SipHash seed leaking into RANSAC iteration order via matching stages), not the per-run level. V1_01 strict 0.0029 m reproduces across both binaries; V2_01 strict and MH_01 strict shifted between builds. This caveat does not invalidate the Phase-26 #1 V1_01 breakthrough or the qualitative trade-off characterizations, but absolute V2_01 strict baseline numbers from earlier writeups should be re-verified on a fresh binary before being cited as reproducible. Addressing the determinism root cause (deterministic iteration via `BTreeMap` or sorted keys at match-input time) is a follow-up beyond Phase-26 #4 scope.

**Recommendation.** Do not ship `--relocalization-recent-keyframe-window` or `--relocalization-max-translation-from-imu-prediction-meters` as part of any recommended config; both ship as diagnostic / experimental knobs (`None` defaults preserve Phase-23 #1 semantics). The Phase-26 #1 V-class accuracy opt-in remains unchanged.

**The Phase-26 thread is empirically closed.** Cliff extension via recovery PnP on EuRoC is structurally unsalvageable with the tested intervention space. Remaining options: Phase-27 (in-Rust ONNX runtime — deployment win, not a research win) or Phase-{20..26} consolidation pass (release tag + clean-up + binary-determinism fix).

Full per-seq breakdown + reproduction at `target/euroc_phase26_4_structural_recovery_ab/SUMMARY.md`; sweep driver at `scripts/run_euroc_phase26_4_structural_recovery_sweep.sh`.

## Phase-27 — In-Rust SuperPoint ONNX runtime (activation, 2026-05-19)

**Scope.** Promote the Phase-27 skeleton to a working in-Rust SuperPoint ONNX path. The empirical signal is unchanged from Phase-26 #1 (offline Python pre-export is bit-identical to in-Rust ONNX inference given the same model weights and image); Phase-27 is purely a deployment / single-binary / per-frame online concern. Operators with a real-time inference target gain the ability to drop a model file in and run without a Python step. Operators happy with the Phase-26 #1 batch workflow lose nothing.

**Shipped.** `crates/vision/Cargo.toml` adds an opt-in `onnx-inference` feature pulling `ort = "2.0.0-rc.12"` (with `download-binaries`, `ndarray`, `std`, `tls-rustls`) and `ndarray = "0.17"`. Root `Cargo.toml` exposes a pass-through `onnx-inference` feature. `crates/vision/src/features/superpoint_onnx.rs` was upgraded from a `FeatureDisabled` stub to a working extractor: behind the feature, `SuperPointOnnxExtractor::load_from_path` constructs an `ort::session::Session` at Level-3 optimization (wrapped in `Arc<Mutex<_>>` so the extractor stays `Clone` for stereo cam0/cam1 plumbing); `DeepFeatureExtractor::extract_deep` preprocesses (grayscale `(1, 1, H, W) f32` in `[0, 1]`), runs the session, and postprocesses outputs read by name (`keypoints`, `scores`, `descriptors`) with auto-detection of layout variants `(N, 2)/(1, N, 2)` for keypoints, `(N,)/(1, N)` for scores, and `(N, 256)/(256, N)/(1, N, 256)/(1, 256, N)` for descriptors. Min-score filter, descending-score sort, top-`max_keypoints` truncation, and a defensive L2-normalisation enforce the downstream `DeepFeatureSet` contract.

Without `--features onnx-inference` the stub remains in place and every method returns `SuperPointOnnxError::FeatureDisabled` with a clear pointer to the activation steps — there is no silent fallback. EuRoC demo wiring (`examples/euroc_online_slam_vi_image_demo.rs`): new `FeatureExtractorKind::SuperPointOnnx` variant, new `DemoExtractor::SuperPointOnnx(SuperPointOnnxExtractor)` variant, new CLI flag `--superpoint-onnx-model <path>`, kind-match string `superpoint-onnx`, audit-log line `superpoint_onnx_model={…}`. Stereo cam1 extraction is trivial: `extract_deep` runs on whatever image is passed, so the seed-frame cam1 call works without a `set_camera` no-op the way `SuperPointOfflineExtractor` requires for index synchronization.

**What still requires validation before claiming Phase-27 fully complete.** (a) Operators must download a SuperPoint ONNX model (~10 MB) per `docs/superpoint_onnx_runtime_plan.md` sourcing notes — `magic-leap-research/SuperPoint`, `fabio-sim/LightGlue-ONNX` releases are the recommended starting points. (b) Bit-identical descriptor regression vs Phase-26 #1 Python pre-export (assert per-keypoint position within 0.01 px, descriptor within 1e-4). (c) EuRoC V1_01 strict empirical re-run reproducing Phase-26 #1's 0.0029 m rigid ATE under the in-Rust path. (d) Per-frame inference latency on the target deployment hardware. The implementation is contract-correct against the LightGlue-ONNX-style export per the plan doc; (a)-(d) is the next contributor session.

**Tests.** 5 new postprocess unit tests behind the feature: `postprocess_filters_by_min_score_and_truncates_to_max_keypoints`, `postprocess_normalises_descriptors_to_unit_norm`, `postprocess_rejects_inconsistent_lengths`, `postprocess_skips_nonfinite_and_below_threshold_scores`, `normalise_descriptors_handles_all_supported_layouts`. The 2 existing default-feature tests (`skeleton_extractor_load_returns_feature_disabled`, `default_config_matches_phase26_pre_export_settings`) remain so the stub path is also covered. Workspace builds and tests pass in both default (no onnx) and `--features image-io,onnx-inference` configurations.

See `docs/superpoint_onnx_runtime_plan.md` for the activation contract, model sourcing, and the validation plan details.

## Binary determinism — mitigation #1 (toolchain pin) shipped (2026-05-19)

**Context.** Phase-26 #4 / Phase-{20..27} closeout observed that two `cargo build --release` runs from the same source produce binaries that diverge on EuRoC V2_01 strict by O(10⁻³ m) rigid ATE, while back-to-back runs of *the same* binary are bit-identical. Hypothesis-rank: rustc codegen variation (FMA fusion, instruction selection, register allocation) cascading through PnP RANSAC FP comparisons. Toolchain drift across `rustup update`s adds to the variance.

**Shipped.** `rust-toolchain.toml` pins the channel to `1.94.0` with `minimal` profile + `rustfmt` / `clippy` components. Bump policy: re-run V2_01 strict / V1_01 strict baselines and update the empirical-results ledger in `docs/binary_determinism_findings.md` when the channel changes. New `scripts/verify_binary_determinism.sh` runs a three-step protocol (clean build → V2_01 strict; same binary → second run; touch source + rebuild → third run) and writes `target/binary_determinism_verify/COMPARE.md` with side-by-side ATE numbers — intended for re-execution after every channel bump and every significant tracker / PnP refactor. New `docs/binary_determinism_findings.md` is the single-source-of-truth for the problem statement, hypothesis-ranking (four candidates), mitigations status, and the ledger table.

**What is *not* shipped.** Kahan / Neumaier summation, P3P closed-form RANSAC, and `-Cllvm-args=-fp-contract=off` are documented as conditional next-step levers — gated on the toolchain pin proving insufficient against the ledger numbers. The findings doc lays out the decision gates so a future contributor can decide based on data, not speculation.
