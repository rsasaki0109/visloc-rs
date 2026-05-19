# SuperPoint + LightGlue descriptor pipeline — Phase-26 plan

**Status:** planning (2026-05-19). No code yet. Phase-25's V2_01 strict
ThreePoseSmoother win (`-25 %` rigid ATE) captured the
motion-model-layer ceiling. The next ceiling is descriptor quality at
cliff-region landmark counts.

## Why now

The Phase-23/24/25 thread has empirically narrowed the EuRoC cliff
problem to **descriptor mismatch under cross-attitude conditions**:

- Phase-23 #1 (relocalization-on-tracker-death): infrastructure
  shipped, but full-map BruteForce PnP cannot match cross-attitude
  HOG descriptors well enough to reach the inlier-ratio gate.
- Phase-23 #1b (pose-prior-guided recovery): tight radius excludes
  landmarks, loose radius admits the same false positives Phase-23 #1
  had — descriptor matching is the bottleneck, not the candidate
  selection.
- Phase-23 #2 (strict-stereo bootstrap): the ONLY universal EuRoC win.
  Dropped the wrong 4 m fixed-depth fallback landmarks; clean metric
  depths now flow through the map.
- Phase-23 #4 + Phase-24 + Phase-25: adaptive motion model
  (IMU↔Pose) + refresh policies. Phase-25 ThreePoseSmoother is the
  current best, with a -25 % V2_01 strict win. The residual MH_01
  strict oscillation reduces to "the constant-pose branch's poses
  are themselves PnP-noise-dominated at cliff-region landmark counts
  (4-8 inliers)" — i.e. upstream of every motion-model intervention.

The remaining lever is the descriptor stream. The Phase-15 offline
SuperPoint replay (Phase-15, CHANGELOG 2026-04-xx) tested SuperPoint
descriptors *before* Phase-23 #2 was shipped, and concluded
"descriptor strength is NOT the binding constraint at this stack
— the 4 m fixed bootstrap depth was." That finding was correct **for
the Phase-13 mixed-depth bootstrap**. With strict-stereo (real metric
depths) shipped, the prior null result becomes inconclusive — the
SuperPoint replay needs to be re-tested on top of the current
Phase-25 default stack.

## What is already shipped that we will reuse

- **`DeepFeatureExtractor` trait** (`crates/vision/src/features/deep/`)
  with `DeepFeatureSet { keypoints, scores, descriptors }`.
- **`HogLikeFeatureExtractor`** — classical-descriptor stand-in
  implementing both `FeatureExtractor` and `DeepFeatureExtractor`,
  used as the placeholder until real SuperPoint inference lands.
- **`MutualSoftmaxMatcher`** — LightGlue-style matcher (temperature
  20.0, mutual-NN, min_confidence 0.2). Already implements the
  `Matcher` trait and is used in `deep_frontend_two_view_demo`.
- **`SuperPointOfflineExtractor`** (`examples/euroc_online_slam_vi_image_demo.rs`)
  — replays pre-exported `frame_NNNNNN_features.txt` files via the
  existing `FeatureExtractor` trait. Mono only; cam0 features
  required, optional cam1 features for stereo bootstrap.
- **`scripts/export_superpoint_lightglue.py --mono-dir`** — Python
  bridge that produces the per-frame feature files. Requires a
  working PyTorch + LightGlue install.
- **`crates/io/src/external_deep.rs`** — text-file format for deep
  feature files. Stable across phases.
- **`OnlineSlamConfig::relocalization`** — Phase-23 #1 recovery
  stage. Currently underwater because of HOG descriptor mismatch
  under cross-attitude; the obvious customer of better descriptors.

## Phase-26 scope (one round)

### #1 — Re-run SuperPoint offline-replay on the Phase-25 default stack (cheapest, highest-information per dollar)

Pre-export cam0 SuperPoint features for MH_01_easy / V1_01_easy /
V2_01_easy (1500 frames each) with `scripts/export_superpoint_lightglue.py
--mono-dir`. Then re-run the Phase-25 recommended config with
`--feature-extractor superpoint-offline --superpoint-features-dir <path>
--no-stereo-bootstrap` against the Phase-25 HOG baseline.

**Limitations of #1:** `--no-stereo-bootstrap` is forced because
Phase-15 only exports cam0. Without stereo bootstrap, strict-stereo
mode (`--stereo-bootstrap-strict`) is meaningless and the map will
fall back to wrong-depth fallback landmarks — which is precisely the
Phase-15 failure mode. So #1 is expected to reproduce Phase-15's
negative finding *unless* the demo's existing motion-VI-init scale
recovery happens to compensate.

**Expected outcome:** likely negative, but cheaply confirms that we
need cam1 features to make the SuperPoint path actually competitive.

### #2 — Cam1 SuperPoint offline-replay + strict-stereo SuperPoint bootstrap

Extend `scripts/export_superpoint_lightglue.py --mono-dir` to also
accept a cam1 directory (or add `--stereo-dirs cam0:<path>,cam1:<path>`)
and emit `frame_NNNNNN_features_cam1.txt`. Extend
`SuperPointOfflineExtractor` to load both cameras (much of the
plumbing already exists via the `--superpoint-cam1-features-dir`
CLI flag — verify completeness).

Then run the strict-stereo bootstrap with SuperPoint descriptors on
both cameras. Cam0↔cam1 matching at bootstrap time should use the
existing `BruteForceMatcher` (descriptors are L2-normalised, so
inner-product = cosine similarity); if that produces low inlier
rates, swap in `MutualSoftmaxMatcher` (which is what LightGlue
actually does).

**Expected outcome:** the real test of the SuperPoint hypothesis.
If strict-stereo + SuperPoint flips the cliff-region inlier count
high enough that the existing relocalization (Phase-23 #1) starts
accepting recoveries, this is the breakthrough.

### #3 — Online inference (deferred to Phase-27)

`ort` crate (ONNX Runtime bindings) is the most mature path; SuperPoint
ONNX models are available (e.g. `magic-leap-research/SuperPoint`),
LightGlue has community-exported ONNX. Pure-Rust alternatives
(`candle`, `burn`) are immature for these specific architectures.

**This phase does NOT ship online inference.** The offline replay is
empirically sufficient for the EuRoC bench, and the binary signal
(do better descriptors fix the cliff?) is identical between offline
and online paths. Online inference is a deployment / latency concern,
not a research one.

## API and integration surface (no changes for #1, additive for #2)

For #1 (cam0-only):
- **No new code** in the visloc-rs crate. The demo already wires
  `SuperPointOfflineExtractor` via `--feature-extractor
  superpoint-offline --superpoint-features-dir <path>`.

For #2 (cam1 + strict-stereo SuperPoint):
- `SuperPointOfflineExtractor::load_cam1_features_dir` — likely
  already exists per the Phase-15 entry mentioning
  `--superpoint-cam1-features-dir`; verify and ship any gaps.
- Bootstrap path (`bootstrap_map_from_first_frame`) needs to accept
  a `DeepFeatureSet` for cam1 as well as cam0 and forward both
  through the existing stereo triangulation pipeline. The existing
  strict-stereo code path drops keypoints lacking cam1 matches, so
  it should slot in unchanged once both cameras supply SuperPoint
  features.
- A new `Matcher` knob on bootstrap (`--stereo-bootstrap-matcher
  {brute|mutual-softmax}`) is optional — start with brute-force
  cosine, drop in mutual-softmax only if inlier rates are visibly
  low.
- New CLI flag `--feature-extractor superpoint-offline-stereo`
  (sets both cam0 and cam1 SuperPoint paths in one shot, so the
  user doesn't have to remember to pair `--superpoint-features-dir`
  with `--superpoint-cam1-features-dir`).
- Summary audit lines: `superpoint_cam1_features_dir`,
  `bootstrap_matcher`.

## Empirical evaluation plan

Same 3-seq EuRoC sweep (MH_01_easy / V1_01_easy / V2_01_easy,
1500 frames each) on top of the Phase-25 default config:
`--motion-model adaptive-imu-pose --adaptive-motion-failures-to-switch-to-pose
3 --adaptive-motion-successes-to-switch-to-imu 10
--stereo-bootstrap-strict --adaptive-motion-refresh-policy three-pose-smoother`
(the Phase-25 default).

Two variants per seq for #1 and #2 each:
- HOG baseline (Phase-25 default)
- SuperPoint replay (cam0 only for #1, cam0+cam1 for #2)

Key metrics to watch (in order of importance):
1. `cliff-region tracking_success` (the per-frame success rate in
   the f60-115 universal cliff window, currently `0`) — if
   SuperPoint pushes it above zero, this is the breakthrough.
2. `relocalization_successes` — Phase-23 #1's gate
   (`min_inliers=20 min_inlier_ratio=0.3`) currently rejects every
   recovery on EuRoC. If SuperPoint + cross-attitude matching
   produces inliers above the gate, recovery becomes useful.
3. `map_keyframes` and rigid `ate_rigid_rmse_m` as secondary
   sanity checks.

Comparison table format identical to Phase-25's
`comparison_table.md` — sequences × variants, sorted by ATE within
each sequence.

## Out of scope

- Online inference (Phase-27 if Phase-26 is positive).
- A LightGlue-the-graph-neural-network reimplementation in Rust
  (`MutualSoftmaxMatcher` is the cheap LightGlue-style stand-in
  and is empirically sufficient on the existing two-view demo).
- DISK / R2D2 / ALIKED variants of SuperPoint (one descriptor
  family at a time).
- Anything outside the EuRoC bench. KITTI / 7-Scenes /
  COLMAP-mapping are downstream of the Rust online inference path.

## Decision gates

- After #1: if SuperPoint cam0-only + no-stereo regression matches
  Phase-15's, skip directly to #2. If it's a surprise win, write up
  and stop the phase early.
- After #2: if strict-stereo SuperPoint produces a measurable
  cliff-region inlier-count lift but the trajectory does not
  improve, the bottleneck has moved downstream (e.g. PnP RANSAC
  parameter tuning, descriptor-bank submap selection); spawn a
  Phase-27 sub-thread targeting that specific bottleneck.
- After #2: if no cliff-region lift is measurable, the descriptor
  hypothesis is also empirically disproved at the Phase-25 stack,
  and the EuRoC arc is best paused. The remaining options are
  out-of-scope architectural rewrites (e.g. dense / direct VO,
  full visual-inertial smoother).

## References

- `docs/motion_based_vi_alignment.md` §Phase-23 #2 (strict-stereo,
  the precondition that makes SuperPoint testing meaningful), §Phase-24,
  §Phase-25 (the just-shipped motion-model ceiling).
- `target/euroc_phase25_refresh_policy_ab/SUMMARY.md` — the
  baseline numbers Phase-26 must improve on.
- `pipelines/tracking/src/lib.rs` — `ImuVelocityRefreshPolicy`,
  `AdaptiveImuPoseMotionModel`; no changes needed for Phase-26.
- `examples/euroc_online_slam_vi_image_demo.rs` — `SuperPointOfflineExtractor`
  (around lines 387–540 in the current file); the CLI surface
  Phase-26 extends.
- `scripts/export_superpoint_lightglue.py` — the Python bridge;
  Phase-26 extends `--mono-dir` to `--stereo-dirs` or equivalent.
- `crates/vision/src/features/deep/` — `DeepFeatureExtractor` trait
  and `HogLikeFeatureExtractor` placeholder.
- `crates/vision/src/matching/mutual_softmax.rs` — `MutualSoftmaxMatcher`
  (the LightGlue-style matcher available out of the box).
