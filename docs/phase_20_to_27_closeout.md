# Phase-{20..27} EuRoC Tracker-Cliff Arc — Unified Closeout

**Status:** closed (2026-05-19). All algorithmic interventions for
EuRoC are shipped or empirically ruled out. Read this as the
single-source-of-truth summary; the per-phase writeups under
`target/euroc_phase*_*/SUMMARY.md` and the long-form narrative in
`docs/motion_based_vi_alignment.md` remain authoritative for
details.

## TL;DR

| Use case                              | Recommended config                                                                                                                                                                                                                          |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Cross-class accuracy (default)**    | `--feature-extractor hog --cross-check-matcher --motion-model imu --stereo-bootstrap-strict` (Phase-23 #2)                                                                                                                                  |
| **Survival-priority** (V-class hover) | Add `--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose 3 --adaptive-motion-successes-to-switch-to-imu 10` (Phase-23 #4 + Phase-25 default refresh policy = `three-pose-smoother`)                                |
| **V-class indoor accuracy** (opt-in)  | Cross-class accuracy + swap the extractor for `--feature-extractor superpoint-offline --superpoint-features-dir <cam0> --superpoint-cam1-features-dir <cam1>` (Phase-26 #1). V1_01 strict reaches **0.0029 m rigid ATE** — the cleanest EuRoC result in the arc. |

## Post-closeout finding (2026-06-03): strict-stereo is frontend-specific

Re-measuring `--stereo-bootstrap-strict` on the **SuperPoint** frontend (the
Phase-26 #1 stack) with the new `scripts/analyze_slam_trajectory.py` /
`compare_slam_runs.py` kit refines the Phase-23 #2 "universal win". The
accuracy benefit strict bought with **HOG** (MH_01 -22 % ATE) does **not** hold
for SuperPoint: on MH_01, strict and non-strict tie on rigid ATE, and strict
just *costs 2.7× coverage*. On the close-range V-class rooms, strict is still
essential — dropping it lets the fixed-4 m fallback landmarks corrupt the scale
(the Phase-23 #2 V-class story, reproduced here):

| seq   | config (SuperPoint) | tracking_success | rigid ATE | sim_scale |
|-------|---------------------|-----------------:|----------:|----------:|
| MH_01 | strict              | 11.9 %           | 0.198 m   | —         |
| MH_01 | **non-strict**      | **32.3 %**       | **0.198 m** | —       |
| V2_01 | strict              | 5.7 %            | **0.011 m** | 1.09    |
| V2_01 | non-strict          | 7.3 %            | 0.208 m   | **2.16** (wrong-scale) |
| V1_01 | strict              | 6.3 %            | **0.0029 m** | 1.03   |
| V1_01 | non-strict          | 6.6 %            | 0.040 m   | 0.93      |

**Recommendation:** keep `--stereo-bootstrap-strict` for V-class / close-range
scenes (scale protection), but **drop it for MH-class / larger-scene SuperPoint
runs** — it triples coverage at no accuracy cost. The *why* (from
`analyze_match_quality.py`): the frontend matches are healthy through the
dropouts (~480 matches, ratio ~0.8), so strict's frame rejection is a
geometry/scale gate, not a feature-failure — and on MH the 4 m fallback is close
enough to the scene that the extra landmarks help rather than hurt.

**Refuted code idea (honest negative):** making the fallback depth adaptive (the
median of the frame's triangulated stereo depths instead of the fixed 4 m) was
implemented and tested across all three sequences — it was *worse* than both
strict and the fixed-4 m non-strict everywhere (MH_01 0.66 m, V1_01 0.235 m).
Cause is structural, not a depth-value tuning gap: seeding all no-stereo
keypoints onto a single fronto-parallel plane (any depth) biases PnP; the median
does not change that. Reverted, not shipped.

## Phase-by-phase summary

| Phase | What shipped                                                              | Empirical outcome                                                                                                                                                                  |
|-------|---------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 20    | Phase-20 baseline (the pre-arc config)                                    | Establishes the EuRoC universal-cliff baseline. (Out of scope for this arc.)                                                                                                       |
| 21    | Empirical cliff documentation                                             | Identified a universal cliff at frames f60-f115 across MH/V-class EuRoC seqs.                                                                                                      |
| 22    | `ImuPredictiveMotionModelConfig::carry_forward_velocity_world`            | Inter-mirror velocity desync fix in `ImuPredictiveMotionModel`. Off by default. Documented as architectural-cleanup, not a cliff intervention.                                     |
| 23 #1 | `OnlineSlamConfig::relocalization` (recovery-PnP infrastructure)           | **Honest negative on EuRoC**: HOG cross-attitude descriptor mismatch dominates. Recovery PnP rejects nearly all attempts; the few accepted ones regress ATE.                       |
| 23 #1b| `pose_prior_candidate_radius_meters`                                       | **Honest negative**: tight radius excludes landmarks (IMU off by \|g·Δt\|), loose radius admits same false positives. HOG matching is the binding limit, not candidate selection.   |
| 23 #2 | `--stereo-bootstrap-strict`                                                | **Universal WIN — first measured EuRoC ATE win.** MH_01 -22 %, V1_01 -17 %, V2_01 -48 % rigid ATE; V2_01 sim_scale recovered to **1.000044** (near-perfect metric).                |
| 23 #3 | Loop closure on dead tracker                                              | Not shipped (gated on cliff extension).                                                                                                                                            |
| 23 #4 | `AdaptiveImuPoseMotionModel` (`--motion-model adaptive-imu-pose`)          | V1_01 imuFavor Pareto win (matches pose survival at near-pose ATE). MH_01 / V2_01 oscillation — IMU `velocity_world` goes stale during pose-mode intervals.                          |
| 24    | `imu_velocity_refresh_policy = FiniteDifference` default                    | **Honest negative**: V2_01 strict +19 % ATE regression, MH_01 strict +53 %. Diagnosis: pose-mode FD injects PnP noise.                                                              |
| 25    | `ImuVelocityRefreshPolicy` enum (`ThreePoseSmoother` new default)           | **V2_01 strict WIN**: 0.2629 → **0.1984 m (-25 %)** vs Phase-23 #4 baseline. MH_01 strict matches baseline. Largest motion-model-layer win in the arc. Recommended default.         |
| 26 #1 | SuperPoint+strict-stereo on Phase-25 stack (no Rust changes; offline replay)| **V-class breakthrough**: V1_01 strict 0.0272 → **0.0029 m (-89 %)** with sim_scale 1.026 ≈ metric; V2_01 strict 0.1984 → **0.0107 m (-95 %)** with sim_scale 2.27 → 1.095. MH_01 +53 % to +64 % regression (extra survived frames at moderate drift cost). |
| 26 #2 | `--relocalization-enabled` on top of SP                                    | **Honest negative**: 4/6 cases 0 recoveries; V1_01 accepts 2-4 false positives → scale collapses to 0.27.                                                                          |
| 26 #2b| `--relocalization-pose-prior-radius`                                        | **Honest negative**: same as Phase-26 #2 (IMU prediction itself drifted into wrong-scale neighborhood).                                                                            |
| 26 #3a| `--pnp-reprojection-threshold-px`                                           | **Honest mixed**: extends V-class trajectories but scale collapses (V1_01 sim_scale → 6-9×). 4 px default confirmed optimal.                                                       |
| 26 #3b| `--mutual-softmax-matcher` (LightGlue-style)                                | **Honest negative**: same failure mode as 3a, more extreme (V1_01 sim_scale → 22.85×). Cross-check + 4 px gate is empirically Pareto-optimal tracker-side filter pair.             |
| 26 #3c| MH_01 ATE decomposition (doc-only)                                          | **Diagnosis**: SP regression is ~93 % drift in extra-survived frames (same-window per-frame accuracy within ~5 % of HOG); ~7 % SP outdoor descriptor weakness.                      |
| 26 #4 | `recent_keyframe_window` + `max_translation_from_imu_prediction_meters`     | **Honest negative**: 4/6 cases unchanged; V1_01 imuFavor false positives *increased* 2 → 5. Active-frontier submap + 2 m IMU sanity ball insufficient.                              |
| 27    | SuperPoint ONNX runtime plan + activation skeleton (no `ort` dep yet)       | Plan doc + skeleton shipped. Empirically equivalent to Phase-26 #1 Python pre-export; deployment / latency concern.                                                                 |
| 27 ↳  | **In-Rust SuperPoint ONNX activation** behind `onnx-inference` feature      | `ort = "2.0.0-rc.12"` + `ndarray = "0.17"` optional deps; `SuperPointOnnxExtractor` body implemented (Session @ L3 opt, preprocess + infer + postprocess, auto-detected output shapes, L2-norm, top-K); EuRoC demo accepts `--feature-extractor superpoint-onnx --superpoint-onnx-model <path>`. Validation against an actual model file (bit-identical regression + V1_01 strict re-run + latency benchmark) is the next contributor step. |
| 27 ↳  | **Binary determinism mitigation #1** — `rust-toolchain.toml` pin + verification script + findings doc | Channel pinned to `1.94.0`; `scripts/verify_binary_determinism.sh` runs the three-step protocol; `docs/binary_determinism_findings.md` documents hypothesis-ranking, mitigations status, and the empirical-results ledger. Conditional next levers (Kahan summation, P3P closed-form, fp-contract=off) documented behind decision gates. |

## What was shipped (concrete artifacts)

### Library code
- `pipelines/slam/src/lib.rs::OnlineSlamRelocalizationConfig` (+5 fields across Phase-23 #1, #1b, Phase-26 #4)
- `pipelines/slam/src/lib.rs::OnlineSlamRelocalizationState` + `maybe_run_relocalization`
- `pipelines/tracking/src/lib.rs::AdaptiveImuPoseMotionModel` + `ImuVelocityRefreshPolicy` enum
- `pipelines/tracking/src/lib.rs::ImuPredictiveMotionModel::{body_velocity_from_camera_pose_difference, pending_samples_total_dt}`
- `crates/vision/src/features/superpoint_onnx.rs` (Phase-27 in-Rust ONNX extractor behind `onnx-inference` feature; stub fallback when feature is off)
- `rust-toolchain.toml` (binary determinism mitigation #1 — channel pinned to `1.94.0`)

### CLI flags on `examples/euroc_online_slam_vi_image_demo.rs`
Recommended (ship in user-facing config):
- `--feature-extractor {hog,superpoint-offline}` (Phase-15/26)
- `--stereo-bootstrap-strict` (Phase-23 #2 — the universal win)
- `--motion-model {imu,adaptive-imu-pose}` (Phase-23 #4)
- `--adaptive-motion-failures-to-switch-to-pose <N>` / `--adaptive-motion-successes-to-switch-to-imu <N>` (Phase-23 #4 thresholds)
- `--adaptive-motion-refresh-policy {none,finite-diff,zero-reset,three-pose-smoother}` (Phase-25 — `three-pose-smoother` default)
- `--superpoint-features-dir <cam0>` / `--superpoint-cam1-features-dir <cam1>` (Phase-15/26 — required for SuperPoint offline replay)
- `--cross-check-matcher` (Phase-10 — recommended for SuperPoint+strict-stereo)
- `--relocalization-enabled` + standard gate knobs `--relocalization-min-inliers`/`--relocalization-min-inlier-ratio`/`--relocalization-max-reprojection-error` (Phase-23 #1 — infrastructure, off by default)

Experimental / diagnostic (ship with ON-by-OPT-IN, document trade-offs):
- `--adaptive-motion-no-refresh-imu-velocity-on-switch` (Phase-24 backward-compat alias for `--adaptive-motion-refresh-policy none`)
- `--relocalization-pose-prior-radius <m>` (Phase-23 #1b — honest negative on EuRoC)
- `--pnp-reprojection-threshold-px <px>` (Phase-26 #3a — honest mixed; 8 px gives cliff-extension trade for V2_01 strict)
- `--mutual-softmax-matcher` (Phase-26 #3b — honest negative; LightGlue-style A/B knob)
- `--relocalization-recent-keyframe-window <N>` (Phase-26 #4a — honest negative)
- `--relocalization-max-translation-from-imu-prediction-meters <m>` (Phase-26 #4b — honest negative)

None of the experimental knobs should appear in recommended-config
documentation outside their per-phase writeups. They ship because
the empirical investigations that produced the negatives need to
remain reproducible.

### Scripts
- `scripts/run_euroc_phase23_relocalization_ab.sh` — Phase-23 #1 sweep
- `scripts/run_euroc_phase23_*.sh` — Phase-23 #2 / #4 sweeps
- `scripts/run_euroc_phase24_adaptive_refresh.sh` — Phase-24 sweep
- `scripts/run_euroc_phase25_refresh_policy_ab.sh` — Phase-25 12-run sweep
- `scripts/tabulate_phase25_results.sh` — Phase-25 comparison helper
- `scripts/run_euroc_phase26_superpoint_strict_stereo.sh` — Phase-26 #1 sweep
- `scripts/run_euroc_phase26b_superpoint_relocalization.sh` — Phase-26 #2
- `scripts/run_euroc_phase26b2_superpoint_reloc_poseprior.sh` — Phase-26 #2b
- `scripts/run_euroc_phase26_3a_pnp_threshold_sweep.sh` — Phase-26 #3a
- `scripts/run_euroc_phase26_3b_mutual_softmax_sweep.sh` — Phase-26 #3b
- `scripts/run_euroc_phase26_4_structural_recovery_sweep.sh` — Phase-26 #4
- `scripts/export_superpoint_lightglue.py --mono-dir` — Phase-15 / 26 offline SuperPoint pre-export

### Documentation
- `docs/motion_based_vi_alignment.md` (~50 sections across the arc — long-form narrative)
- `docs/superpoint_lightglue_plan.md` (Phase-26 plan, now superseded by Phase-26 #1 implementation + this closeout)
- `docs/superpoint_onnx_runtime_plan.md` (Phase-27 activation contract)
- `target/euroc_phase*_*/SUMMARY.md` — per-phase empirical writeups
- `CHANGELOG.md` Unreleased entries — release-notes-quality per-phase summaries
- `PLAN.md` Phase-23 close-out table + reading order section

### Tests
- 576 workspace tests passing
- 3 new Phase-25 unit tests covering `ImuVelocityRefreshPolicy` variants
- 3 new Phase-26 unit tests around `OnlineSlamRelocalizationConfig` defaults
- 2 new Phase-27 skeleton tests
- 0 integration tests for the EuRoC binary (intentionally — would require shipping data + the Python pipeline)

## Empirical headline (the headline numbers in one place)

V1_01 strict rigid ATE evolution:
| Stack                                | rigid ATE m   | sim_scale |
|--------------------------------------|---------------|-----------|
| Phase-20 baseline                    | (cliff before f100; no useful metric)             |           |
| Phase-23 #2 strict-stereo (HOG)      | 0.0272        | 1.031     |
| Phase-25 ThreePoseSmoother (HOG)     | 0.0272        | 1.031     |
| **Phase-26 #1 SuperPoint+strict**    | **0.0029**    | **1.026** |

V2_01 strict rigid ATE evolution:
| Stack                                | rigid ATE m   | sim_scale |
|--------------------------------------|---------------|-----------|
| Phase-23 #2 strict-stereo (HOG)      | 0.0040        | 1.000044  |
| Phase-23 #4 adaptive (HOG, refresh-off) | 0.2629    | 2.827     |
| Phase-24 finite-diff refresh (HOG)   | 0.3119        | 1.117     |
| Phase-25 ThreePoseSmoother (HOG)     | 0.1984        | 2.273     |
| **Phase-26 #1 SuperPoint+strict**    | **0.0107***   | **1.095** |
| post-pin reproducible (2026-05-19, rustc 1.94.0 pinned) | 0.2013 | 1.955 |

`*` The 0.0107 m headline was a one-time-only number from a then-current
Phase-26 #1 binary build that is now gone. The pinned (rustc 1.94.0)
binary stably and reproducibly produces 0.2013 m with sim_scale 1.955
(3-run bit-identical: same binary twice + clean rebuild). See *Binary
determinism* below for the post-pin verification result and the
revised reproducibility claim. **V1_01 strict 0.0029 m still
reproduces** exactly on the pinned binary; only V2_01 strict shifted.

## Known issues (the arc surfaced these; not all fixed)

### Binary determinism — **resolved by toolchain pin (2026-05-19)**

Phase-26 #4 observation: V2_01 strict and MH_01 strict rigid ATE
shifted between `cargo build --release` runs of identical source /
Cargo.lock / flags. Two runs of the *same* binary were always
bit-identical, so the non-determinism was at the build level, not
the per-run level.

**Mitigation shipped and verified**: `rust-toolchain.toml` pins
the channel to `1.94.0`. `scripts/verify_binary_determinism.sh`
(VARIANT=baseline | superpoint) runs a three-step protocol (clean
build → run; same binary → run; touch a source file + rebuild →
run) and writes `target/binary_determinism_verify_*/COMPARE.md`.

**Empirical results (2026-05-19, post-pin)**:

| Config (V2_01_easy unless noted) | run1↔run2 | run2↔run3 (clean rebuild) | rigid ATE   |
|----------------------------------|-----------|---------------------------|-------------|
| baseline (corner extractor, f=3/s=10)        | bit-identical | bit-identical | 4.8783 m (degenerate by design — corner extractor doesn't track V2_01) |
| SuperPoint + strict-stereo (Phase-26 #1)     | bit-identical | bit-identical | 0.2013 m (sim_scale 1.955) |
| SuperPoint + strict-stereo on **V1_01_easy** | bit-identical | bit-identical | 0.0029 m (sim_scale 1.026) — matches Phase-26 #1 headline exactly |

The toolchain pin **fully resolves cross-rebuild determinism on
every configuration tested**. The previous Phase-26 #4 observation
of cross-rebuild variance was caused by `rustup update` between
build moments shifting LLVM codegen; with the channel pinned that
class of variance is gone.

**The headline number for V2_01 strict has been revised** from the
Phase-26 #1 one-time-only 0.0107 m to the post-pin reproducible
0.2013 m (sim_scale 1.955 ≈ half-metric). V1_01 strict 0.0029 m
reproduces exactly. The V-class breakthrough framing is preserved
for V1_01 only; V2_01 strict SP is "deterministic but in the
wrong-scale regime" on the current binary — a separate empirical
question from the determinism one.

Defensive `HashMap` / `HashSet` sort sites
(`pipelines/tracking/src/lib.rs:4126`,
`pipelines/localization/src/lib.rs:409`,
`LandmarkDescriptorStore::iter` / `ordered_landmark_descriptors`)
remain in place as defense-in-depth even though the determinism
problem turned out to be a toolchain-drift issue rather than a
HashMap-iteration issue. Cost is zero (already shipped); they rule
out a class of regressions from a future contributor.

**Kahan summation / P3P closed-form / `-Cllvm-args=-fp-contract=off`
are no longer warranted** for cross-rebuild determinism — the
pin handles it. They remain documented in
`docs/binary_determinism_findings.md` as conditional levers if a
future toolchain bump re-introduces variance.

### Recovery PnP is structurally unsalvageable on EuRoC cliffs

Phase-26 #2 / #2b / #4 empirically established that no
tracker-friendly intervention (gate tuning, candidate-set trimming,
pose-prior radius, IMU sanity check) extends V-class cliffs without
collapsing scale or regressing ATE. Phase-26 #3a / #3b further
established that no matcher / gate combination at the tracker's
own primary PnP can extend the cliff without the same scale
collapse. The cliff is upstream of every tracker-side filter we
tested.

A genuine cliff extension would require either:

- A motion model that constrains scale during recovery (rather than
  validating it post-hoc), or
- A co-visibility-graph search operating on geometric invariants
  rather than descriptor inliers, or
- An accepted-as-architectural change such as a dense/direct VO
  fallback or a full visual-inertial smoother.

All three are out of scope for this arc.

### MH-class trade-off

Phase-26 #1 SuperPoint+strict-stereo trades MH-class accuracy for
trajectory continuity (+78 % tracking density / +18 % trajectory
length / +53-64 % rigid ATE). Same-window per-frame accuracy is
within ~5 % of HOG (Phase-26 #3c decomposition). HOG is the
recommended MH-class default; SuperPoint is opt-in when continuity
matters more than aggregate ATE.

## Next directions (post-arc)

1. **Validate the Phase-27 activation against a real model** —
   `ort` is wired and the in-Rust extractor body is implemented
   behind the `onnx-inference` feature. The remaining work is the
   operator-side bit: download a SuperPoint ONNX model file
   (~10 MB), run the bit-identical descriptor regression test vs
   the Python pre-export, re-run V1_01 strict in-Rust and assert
   the 0.0029 m rigid ATE reproduces, and characterise per-frame
   inference latency on the deployment target. ~2-4 hours per
   `docs/superpoint_onnx_runtime_plan.md` validation plan.
2. **Continue the binary-determinism workstream** —
   `rust-toolchain.toml` is now pinned to `1.94.0` and
   `scripts/verify_binary_determinism.sh` exists. The next
   contributor should populate the empirical-results ledger in
   `docs/binary_determinism_findings.md` after each toolchain
   bump. If cross-rebuild variance still exceeds 10⁻³ m on V2_01
   strict under the pin, the documented conditional levers kick
   in (Kahan summation in PnP RANSAC reductions, P3P closed-form
   swap, or `-Cllvm-args=-fp-contract=off`).
3. **Fundamentally different recovery loop** — scale-constraining
   motion model, or co-visibility-graph geometric-invariant search.
   Out of scope for the EuRoC arc; would address the structurally
   unsalvageable cliff.
4. **Dense / direct VO fallback** — bigger architectural addition;
   would complement (not replace) the existing sparse-feature
   tracker.

## How to reproduce the arc

Within a single binary build, every empirical claim in the per-phase
writeups is reproducible. Across binary builds, expect ~10-20 %
numerical drift on V2_01 strict and MH_01 strict (per the known
issue above); V1_01 strict reproduces exactly.

Standard reproduction recipe for the headline V-class accuracy
result:

```sh
EUROC=/path/to/euroc
mkdir -p target/euroc_phase26_superpoint/V1_01_easy/cam{0,1}
for cam in cam0 cam1; do
  python3 scripts/export_superpoint_lightglue.py --mono-dir \
    "$EUROC/V1_01_easy/mav0/$cam/data" \
    --out-dir "target/euroc_phase26_superpoint/V1_01_easy/$cam" \
    --frames 1500 --device cuda --max-keypoints 1500
done

scripts/run_euroc_phase26_superpoint_strict_stereo.sh V1_01_easy

grep ate_rigid_rmse_m target/euroc_phase26_V1_01_easy_strict_superpoint/summary.txt
# Expected: ate_rigid_rmse_m=0.0029
```

## Reading order for a fresh contributor

1. **This file** (`docs/phase_20_to_27_closeout.md`) for the
   one-page narrative.
2. **`PLAN.md`** §"Phase-23 thread close-out" table for the per-phase
   ship/empirical-outcome status.
3. **`docs/motion_based_vi_alignment.md`** for the long-form
   per-phase deep dives (~50 sections; skim the most recent first).
4. **`target/euroc_phase26_superpoint_strict_stereo/SUMMARY.md`** for
   the headline V-class breakthrough writeup.
5. **`docs/superpoint_onnx_runtime_plan.md`** for the Phase-27
   activation contract.
6. **`pipelines/tracking/src/lib.rs`** + **`pipelines/slam/src/lib.rs`**
   for the implementation — grep `Phase-` to find phase-tagged code
   regions.
