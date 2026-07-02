# Progress

This file tracks project milestones and remaining engineering work.

## Current Development State

The file-backed two-view match VO path drives a short tracking sequence
end-to-end, a classical essential-matrix RANSAC frontend recovers metric
relative pose from the same external correspondences, loop-closure candidates
are geometrically verified through that classical frontend, verified candidates
lift into a `LoopClosureConstraint`, and a sparse `PoseGraph` consumes
sequential and loop-closure edges. The graph exposes both a translation-only
Gauss-Newton step (fast linear baseline) and a full SE(3) iterative
Gauss-Newton solver (right-perturbation + first-order BCH) that corrects
rotations alongside translations. The end-to-end six-keyframe loop demo
(`online_slam_pose_graph_loop_demo`) drives the whole tracking + verifier +
pose-graph stack on a self-contained synthetic sequence with measured drift
correction for both pure-translation and combined translation + rotation drift.
Current work should be evaluated by public-data metrics, documented
limitations, and runnable tests rather than by a completion percentage.

Completed pieces:

- Adaptive stereo depth gate evidence is now registry-backed for a minimal
  KITTI seq00 diagnostic smoke: adaptive default and legacy fixed-3m replay
  both record `effective_min_depth_m_mean=3 m` on the same far-field subset,
  with per-frame diagnostics summarized in
  `docs/generated/kitti_adaptive_depth_gate_smoke.md`; the reduced-feature
  6-frame adaptive failure is retained in the same registry-backed summary.
  This is non-regression evidence for the gate policy, not a trajectory
  benchmark claim.
- `VisualOdometryFrontend` and `VisualOdometryPriorProvider` boundaries exist.
- VO-derived pose priors can narrow tracking candidates.
- Externally generated two-view match files can be parsed.
- External two-view correspondences can produce a lightweight translation-only
  VO prior through `TwoViewMatchVisualOdometryFrontend`.
- File-backed two-view match files can drive `TwoViewMatchVisualOdometryFrontend`
  through `read_two_view_matches_txt`, and the resulting VO priors feed the
  external-prior tracking path on a short multi-frame sequence in the
  `track_sequence_with_two_view_match_vo_prior` example.
- A classical essential-matrix two-view geometry pipeline now lives in
  `visloc-vision::two_view` with a Hartley-normalized 8-point estimator,
  Sampson-distance scored RANSAC, and 4-fold cheirality disambiguation.
- `EssentialMatrixVisualOdometryFrontend` exposes that pipeline as a
  `VisualOdometryFrontend`, returning a full SE3 relative pose with
  caller-supplied translation scale; `two_view_vo_compare` runs the new
  frontend alongside the flow-only adapter on the same synthetic three-frame
  sequence to make the difference visible.
- A classical-geometry `EssentialMatrixLoopClosureVerifier` consumes the
  same essential-matrix RANSAC and reports `LoopClosureVerification` with
  inlier count, inlier ratio, mean Sampson error, score, recovered relative
  pose, and an enumerated failure reason. `verify_loop_closure_candidates`
  plus `correspondences_for_loop_candidate` plumb shared landmarks from the
  current frame's tracking inliers and an older keyframe's observations into
  the verifier without requiring `OnlineSlamPipeline` callers to change.
- `LoopClosureConstraint` (with `from_verified_candidate` /
  `loop_closure_constraints_from_candidates`) lifts each verified candidate
  into a stand-alone constraint (`from_keyframe_id`, `to_keyframe_id`,
  `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`,
  `score`) that the pose-graph / BA consumers can consume.
- `PoseGraph` skeleton (nodes = `BTreeMap<u64, Pose>`, edges =
  `PoseGraphEdge { from, to, measurement, kind, weight }`,
  anchor = `Option<u64>`) plus `PoseGraphEdgeKind::{Sequential, LoopClosure}`,
  builders (`add_pose`, `add_sequential_edge`, `add_loop_closure_constraint`,
  `anchor`, `relative_world_to_camera`), `translation_cost`, and a single
  translation-only Gauss-Newton step `optimize_translations_once` that holds
  rotations fixed and returns a `PoseGraphOptimizationStep` diagnostic. The
  step is exact for translation-only residuals.
- `online_slam_loop_candidate_with_verifier_dummy` example now also builds
  per-frame `LoopClosureConstraint`s and prints the recovered relative
  translation; the loop HTML/SVG report surfaces a separate Loop Closure
  Constraints table alongside the candidate diagnostics. The example also
  injects a small drift into the most recent keyframe, builds a `PoseGraph`,
  runs `optimize_translations_once`, and prints the cost / mean-correction /
  max-correction diagnostics so the loop drift correction is visible.
- `online_slam_pose_graph_loop_demo` example exercises the full pipeline on
  a six-keyframe synthetic loop: classical-tracker localization, verifier
  validation of the closed loop with the matching translation scale,
  `PoseGraph` construction with five sequential edges plus the verified
  loop-closure constraint, a translation-only Gauss-Newton step that
  pulls a `[0.06, 0.03, -0.05]` injected drift back to the loop-closed truth
  in one solve (`cost_before=0.105 → cost_after=0.000`, all six keyframes at
  `err=0.0`), and a follow-up full SE(3) iterative Gauss-Newton run that
  recovers from a combined `[0.04, 0, -0.03]` translation drift plus a
  `0.18 rad` yaw drift on the most recent keyframe in 2 iterations
  (`se3_cost_before=0.557 → 0.000`).
- `PoseGraph::optimize_se3_iterative` (with `PoseGraphSe3Config`,
  `PoseGraphSe3IterationStats`, and `PoseGraphSe3Result`) runs full SE(3)
  Gauss-Newton with right-perturbation updates `T_i ← T_i · Exp(δ_i)`,
  per-edge residual `r = log(meas⁻¹ · T_to · T_from⁻¹)`, and
  Jacobians `Ad(T_from)` (to-node) and `-Ad(T_from)` (from-node) under a
  first-order BCH approximation. `PoseGraph::se3_cost` reports the matching
  cost; `optimize_translations_once` remains the fast linear baseline.
- SE(3) Lie-group helpers (`SE3::log`, `SE3::exp`, `SE3::adjoint`,
  `so3_left_jacobian`, `so3_left_jacobian_inverse`) live in
  `visloc-core::geometry::se3` with Taylor fallbacks for small angles and
  `exp ∘ log` round-trip + adjoint-conjugation tests.
- An end-to-end real-image VO + loop-closure demo,
  `online_slam_image_vo_loop_demo` (gated behind the `image-io` feature),
  reads a KITTI-format grayscale image sequence + `calib.txt`, extracts
  `CornerFeatureExtractor` features per frame, matches them with
  `CrossCheckMatcher<BruteForceMatcher>`, recovers each consecutive
  pair's relative SE(3) via 8-point essential-matrix RANSAC, integrates
  the trajectory, runs the same pipeline between the first and last
  frames as the loop closure constraint, and corrects the chain with
  `PoseGraph::optimize_se3_iterative`. No simulated drift and no GT
  poses are used — the drifted trajectory is what monocular essential-
  matrix VO actually produces from the pixel data.
- A consensus loop-closure verifier, `HybridLoopClosureVerifier`, runs both
  the essential-matrix and PnP backends on the same candidate and accepts
  only when both verify AND their recovered relative poses agree within
  configurable rotation / translation-direction tolerances. Disagreement
  surfaces as `LoopClosureVerificationFailureReason::PoseDisagreement`.
  This catches ambiguity where the 2D-2D essential fit looks plausible but
  conflicts with the 3D map structure (or vice versa) without forcing
  callers to glue two verifier outputs together by hand.
- A second loop-closure verifier path, `PnPLoopClosureVerifier`, reuses
  `visloc-vision::ransac::PnPRansac` to re-localize the current frame
  against the candidate keyframe's landmarks. It operates on 2D-3D
  correspondences (built via `correspondences_2d3d_for_loop_candidate`
  from the current frame's tracking inliers intersected with the older
  keyframe's observed landmarks) and returns a metric relative SE(3)
  directly — no externally supplied translation scale required. The
  `online_slam_pnp_loop_demo` example runs both the essential-matrix and
  PnP verifiers on the same candidate and prints the diagnostics
  side-by-side.
- `optimize_se3_iterative` now also runs Levenberg-Marquardt with optional
  Huber / Cauchy robust kernels (`RobustKernel::{None, Huber{delta},
  Cauchy{c}}`). `PoseGraphSe3Config` gains `robust_kernel`, `initial_lambda`,
  `lambda_increase_factor`, `lambda_decrease_factor`, `max_lambda`, and
  `min_lambda`. `PoseGraphSe3IterationStats` records per-attempt `lambda`
  and `step_accepted` so the LM trajectory is inspectable. The dense
  normal-equations solve now prefers Cholesky on the SPD system and falls
  back to LU on ill-conditioned cases. `PoseGraph::robust_se3_cost` reports
  the kernel-shaped objective. The new `pose_graph_robust_demo` example
  shows that a wildly wrong outlier loop closure that drags KF30 ~0.20 m
  off truth under pure Gauss-Newton is suppressed to ~0.002 m drift under
  LM + Huber.
- A second loop-closure demo, `online_slam_public_loop_demo`, ingests a
  COLMAP-text-format sparse reconstruction from disk (defaulting to a
  synthesized 12-keyframe / 60-landmark orbit fixture written via
  `write_colmap_text_model`) and drives the full SLAM pipeline on the
  loaded data. With `--colmap-path <dir>` it loads any user-supplied
  reconstruction, reporting `se3_cost_before ≈ 8.3 → ≈ 1e-4` in 3
  iterations on a combined `[0.05, 0, -0.04]` translation + `0.18 rad`
  yaw drift. Synthetic per-landmark descriptors are generated when no
  `landmark_descriptors.txt` is supplied so the demo stays runnable on
  any registered COLMAP model.
- Online SLAM composition exists over tracking and local mapping.
- Loop-closure candidates can be detected from shared verified landmarks.
- Loop-candidate HTML/SVG reporting exists for synthetic sequence demos.
- `visloc_io::kitti_imu` loads KITTI raw OXTS / IMU recordings.
  `read_kitti_oxts_dir(<sequence>/oxts)` parses the 30-field
  `oxts/data/<10-digit>.txt` rows together with the textual
  `oxts/timestamps.txt` and returns `Vec<KittiOxtsRecord>` carrying
  wall-clock nanoseconds plus a typed `KittiOxtsSample` (geodetic +
  RPY + body/nav velocities + body/nav accel + body/nav angular rate
  + accuracy / mode fields). The body-frame `acceleration_body_mps2`
  and `angular_rate_body_rps` triplets are directly consumable by
  `ImuPreintegrator::integrate_sample`. Timestamp parsing is
  chrono-free / timezone-free (private Howard Hinnant
  `days_from_civil` helper) so the loader is pure-text and sits
  outside the `image-io` feature gate. `visloc_slam::slice_imu_samples_for_keyframes`
  buckets the resulting timestamp / gyro / accel slices into the
  per-keyframe `Vec<Vec<StereoVoBaImuSample>>` layout expected by
  `StereoVoBaImuInput.windows`: each keyframe pair collects samples
  with `kf[i] < t <= kf[i+1]`, `dt` is anchored at the previous
  timestamp inside the window (first sample anchored at `kf[i]`), and
  a trailing zero-order-hold step closes the interval when the last
  sample stops short of `kf[i+1]`. Windows without IMU coverage stay
  empty so `refine_stereo_vo_with_ba` silently skips wiring an IMU
  factor on that segment. Twelve new tests cover the OXTS parser
  (full 30-field row, comments / blank lines, too-few / non-numeric
  fields, timestamp with / without fractional seconds, full
  `read_kitti_oxts_dir` round-trip, data/timestamp count-mismatch
  detection — 9 in `crates/io/tests/kitti_imu.rs`) and the slicer
  (mixed-coverage 4-sample / 3-window bucketing with total integrated
  `Δt` matching the keyframe span, empty-window-on-no-coverage, and
  length / monotonicity validation — 3 in
  `pipelines/slam/src/stereo_vo_ba.rs::tests`). The slicer is now
  wired into `examples/stereo_vo_external_deep_files` via
  `--kitti-oxts-dir <dir>` + `--kitti-image-timestamps <path>`, which
  reads `<sequence>/oxts` and the image-stream `timestamps.txt`,
  builds the per-keyframe windows, and feeds them into
  `StereoVoBaImuInput.windows` on the post-process BA path (requires
  `--enable-ba`, conflicts with `--online-ba` and `--imu-windows-dir`).
  Two integration tests in `tests/kitti_oxts_imu_pipeline.rs` exercise
  the loader + slicer through the `visloc_rs` facade.
- 3D Gaussian Splatting / NeRF ingestion path lands as a
  COLMAP-text exporter. `visloc_io::colmap::write_colmap_text_model_for_3dgs`
  takes `(camera, poses, left_features, stereo_per_frame,
  image_name_fn)` and writes `cameras.txt` / `images.txt` /
  `points3D.txt` under a target directory: one shared PINHOLE
  camera, one image entry per pose (`world_to_camera` quaternion +
  translation, NAME = `<prefix><6-digit-frame-idx><suffix>`), and a
  sparse `points3D.txt` lifted from every stereo feature's
  left-camera `point_cam` through `pose.camera_to_world()`. The
  CLI `examples/stereo_vo_external_deep_files --colmap-export
  <dir>` invokes this after VO + (optional) BA so the refined
  poses / triangulated points are what gets exported, and the
  resulting directory drops straight into `nerfstudio ns-train
  splatfacto --data <dir>` / Inria gaussian-splatting's
  `convert.py --skip-matching` workflow. A binary counterpart
  `write_colmap_binary_model_for_3dgs` writes `cameras.bin` /
  `images.bin` / `points3D.bin` in the little-endian COLMAP
  layout for trainers that prefer the binary form;
  `--colmap-export-binary <dir>` on the CLI is independent of
  `--colmap-export` so a single VO run can emit both formats.
  The same four flags (`--colmap-export`, `--colmap-export-binary`,
  `--colmap-image-prefix`, `--colmap-image-suffix`) are also on
  `examples/online_slam_stereo_vo_kitti_demo`, so the real-data
  KITTI stereo VO demo can produce a 3DGS bootstrap directly from
  the VO + (optional) BA pipeline (image suffix defaults to `.png`
  to match the KITTI filenames; the writer uses the BA-refined
  poses when stereo BA ran, otherwise the raw VO poses).
  Nine tests in `crates/io/tests/colmap_export.rs` round-trip
  the text and binary exports through `read_colmap_text_model`
  / `read_colmap_binary_model`, verify length-mismatch rejection
  on each path, pin text-vs-binary writer parity (both formats,
  fed the same synthetic input, read back as maps with the same
  camera intrinsics, keyframe poses, and landmark world positions
  — `< 1e-9` tolerance on all numeric fields), and pin
  symmetric validation: both writers reject `CameraModel::Unknown`
  names that have no COLMAP model id, and a single shared
  `validate_colmap_image_name` makes both writers reject any
  `image_name(frame_idx)` containing NUL / space / tab / LF / CR
  (NUL terminates the binary NAME field, whitespace breaks the
  text format's space-separated tokens, LF would inject a
  spurious image record into `images.txt`). The validation
  symmetry means a caller driving both writers off the same
  input either gets both files or the same structured
  `ColmapError::InvalidExportInput` from each surface.
  A `scripts/run_kitti_3dgs_smoke.sh` harness fetches a small
  stride-4 KITTI subset, runs the KITTI stereo VO demo with
  both `--colmap-export` and `--colmap-export-binary`, and loads
  BOTH writer surfaces back through their in-repo readers
  (`examples/inspect_colmap_text_model` + `examples/inspect_colmap_binary_model`,
  which print `cameras / keyframes / landmarks / observations`
  and exit non-zero on reader failure or empty cameras /
  keyframes), so writer ↔ reader divergence on real driving
  data is caught in CI rather than only on the synthetic
  unit-test fixtures. The script then grep/awks the four count
  fields out of the two inspect logs and aborts on any
  disagreement OR on any missing field (an empty value would
  otherwise silently match another empty value if one inspect
  tool's output schema drifted), and additionally requires
  `landmarks > 0` so a run where every stereo feature was
  rejected upstream cannot pass parity trivially with both
  formats reporting zero structure. The inspect tools and the
  optional `ns-train splatfacto` step are invoked with `cmd > log`
  redirection (not `cmd | tee log`), because POSIX sh / dash do
  not have `set -o pipefail` and `tee`'s zero exit would
  otherwise silently mask an upstream failure. An optional
  `--run-ns-train` flag chains `ns-train splatfacto --data
  <colmap_text_dir>` when the trainer is available; the trainer
  step is otherwise skipped so the smoke stays tractable without
  a CUDA Python environment, leaving the full
  KITTI → COLMAP → 3DGS pipeline reachable from a single
  in-repo entry point.
- `visloc_slam::VisualInertialInitializer` closes the last
  remaining OSS-parity gap above the dead-reckoning baseline. Real
  VIO front-ends (ORB-SLAM3, VINS-Mono, OKVIS, Kimera-VIO) all gate
  the joint estimator on a "VI initialisation" stage that turns the
  raw IMU stream into `(R_w←b, v_w, b_g, b_a)` before any image
  constraint is consumed; without it every state has to be seeded
  from ground truth (cheating) or accumulate hundreds of metres of
  dead-reckoning drift before the first keyframe. The new module
  ships the stationary-window flavour: detect a window where the
  gyro and accel signals are statistically consistent with a body
  at rest, then read out (a) `b_g = ω̄` from the gyro mean, (b)
  `R_w←b` from the shortest rotation that lifts the mean specific-
  force direction into the world "up" direction `-g_w / ‖g_w‖` (yaw
  is unobservable from gravity alone and is left at zero), and (c)
  `b_a` from the magnitude residual after the rotation has absorbed
  the direction. Stationary detection guards the read-out with
  three thresholds — per-axis gyro std-dev, per-axis accel std-dev,
  magnitude error vs `‖g_w‖` — plus a minimum sample count and
  window duration. The defaults match the EuRoC stationary holding
  period (z-up gravity, `0.5 s` of quiet samples,
  `0.05 rad/s` / `0.5 m/s²` / `0.5 m/s²` noise thresholds). The
  `examples/euroc_imu_dead_reckon_demo` harness gained
  `--run-vi-init` / `--seed-from-vi-init` /
  `--vi-init-window-seconds` /
  `--vi-init-{gyro,accel}-std-limit` flags so the same end-to-end
  example validates the initialiser against ground truth (logs
  `gravity_alignment_residual_deg`, `bias_gyro_residual_vs_gt`,
  `bias_acc_residual_vs_gt` for the recovered bootstrap) AND runs
  the propagator with the honest "no GT cheating" seed. Empirical
  validation on the five EuRoC sequences in the existing IMU
  baseline table, with loose thresholds (`--vi-init-gyro-std-limit
  0.3 --vi-init-accel-std-limit 1.0`):

  | Sequence       | VI init status      | Gravity alignment residual | Gyro-bias residual (rad/s) | Accel-bias residual (m/s²) | Dead-reckon ATE (VI-init seed, rigid RMSE) | Dead-reckon ATE (GT seed, rigid RMSE) |
  |----------------|---------------------|----------------------------|----------------------------|-----------------------------|---------------------------------------------|----------------------------------------|
  | MH_01_easy     | rejected (accel std `1.33` > `1.0`) | —      | —      | —      | —      | 1355.93 |
  | MH_02_easy     | success             | `1.95°`                    | `0.114`                    | `0.484`                     | (skipped — bias residual indicates real motion in leading window) | 241.60  |
  | MH_03_medium   | success             | `0.68°`                    | `0.0019`                   | `0.133`                     | `928.99`                                    | `208.85` |
  | V1_01_easy     | rejected (accel std `1.08` > `1.0`) | —      | —      | —      | —      | 756.99  |
  | V2_01_easy     | success             | `0.97°`                    | `0.0017`                   | `0.139`                     | `273.51`                                    | `258.34` |

  Two facts this validation pins: (a) on EuRoC sequences with a
  genuine stationary holding period (MH_03, V2_01), the
  initialiser recovers gyro bias within `~0.002 rad/s` — at the
  random-walk floor of the EuRoC IMU — and aligns gravity within
  `~1°`, so `V2_01_easy`'s dead-reckon ATE under the VI-init seed
  (`274 m`) is statistically indistinguishable from the same run
  under the GT-cheating seed (`258 m`); (b) the gap on MH_03
  (`929 m` vs `209 m`) is concentrated in the orientation track,
  but the attribution is ambiguous: the observed `14°` rotation
  residual is *also* explained by the recovered gyro bias residual
  integrated over the full sequence duration
  (`0.0019 rad/s × 132.6 s ≈ 0.252 rad ≈ 14.4°`), and the
  `0.68°` gravity-alignment residual alone — when projected as a
  roll/pitch gravity leakage over `132.6 s` — sits at the right
  order of magnitude to explain the `929 m` drift. V2_01 has a
  similar gyro-bias residual (`0.0017 rad/s × 112.7 s ≈ 11°`)
  but does **not** show the same ATE penalty, suggesting motion
  profile or alignment-after-rigid-rotation matters more than the
  raw rotation residual. A `--seed-rotation-source` /
  `--seed-bias-source` / `--seed-velocity-source` ablation harness
  for `euroc_imu_dead_reckon_demo` is the right next step to
  attribute MH_03's drift cleanly between yaw gauge, gyro bias
  integration, roll/pitch gravity leakage, and initial-velocity
  error.
  Tightening yaw will require either visual aiding (one keyframe
  with a yaw-resolving feature constraint) or motion-based
  initialisation (the ORB-SLAM3 "wait-for-translation, then run
  motion-only optimisation" flow), both of which fall to the next
  follow-up step. With this in place, every entry in the OSS
  feature-coverage matrix that previously read "×" for visloc-rs
  now reads "○" — the remaining gap to ORB-SLAM3 et al. on
  EuRoC is no longer a missing component but a closed-loop
  benchmarking exercise. The follow-up that turns the standalone
  initialiser into a `OnlineSlamPipeline` pre-tracking bootstrap
  stage (so callers stop buffering the leading IMU window out of
  band) is designed in
  [vi_initialization_integration.md](vi_initialization_integration.md).
- Pure-IMU baseline ATE is now measured on five EuRoC MAV
  sequences using the `euroc_imu_dead_reckon_demo` harness with
  Umeyama-aligned ATE (the new
  `TrajectoryAlignment::Umeyama` /
  `TrajectoryAlignment::UmeyamaWithScale` variants drive a
  closed-form `umeyama_similarity_transform` SVD solver, so the
  baseline produces directly comparable numbers against published
  ORB-SLAM3 / VINS-Mono / OKVIS / Kimera-VIO ATE on EuRoC). Each
  sequence is seeded from the first ground-truth state inside the
  IMU window (position, velocity, orientation, gyro / accel biases),
  then forward-Euler integrated for the full cam0 cadence with no
  visual aiding. The numbers below are unaided IMU drift only — they
  are the lower bound that any visual-inertial pipeline added on
  top of `ImuPredictiveMotionModel` + `OnlineSlamPipeline.local_vi_ba`
  is expected to beat by three to four orders of magnitude. They are
  the same metric ORB-SLAM3 / VINS-Mono / OKVIS / Kimera-VIO report,
  so they slot into the comparison table immediately.

  | Sequence       | Duration | ATE raw RMSE (m) | ATE rigid-aligned RMSE (m) | ATE Sim(3)-aligned RMSE (m) | Sim(3) scale |
  |----------------|----------|------------------|----------------------------|------------------------------|--------------|
  | MH_01_easy     | 183.0 s  | 1905.34          | 1355.93                    | 4.25                         | 5.44e-4      |
  | MH_02_easy     | 151.1 s  | 321.47           | 241.60                     | 4.50                         | 4.83e-3      |
  | MH_03_medium   | 132.6 s  | 297.30           | 208.85                     | 3.57                         | 1.29e-3      |
  | V1_01_easy     | 144.5 s  | 1142.80          | 756.99                     | 1.79                         | 6.31e-4      |
  | V2_01_easy     | 112.7 s  | 394.48           | 258.34                     | 2.07                         | 3.90e-3      |

  Two pieces of context this table buys: (a) the recovered Sim(3)
  scale factor is `~10⁻³` across every sequence — the IMU-integrated
  trajectory has expanded by a factor of `~10³` against the bounded
  GT, so the Sim(3) ATE compresses the explosion into the scale
  parameter rather than reflecting a real "good shape" match;
  (b) the rigid-aligned RMSE in the 200–1400 m range is the honest
  IMU-drift number — three to four orders of magnitude above the
  reference OSS systems (ORB-SLAM3 `~0.035 m`, VINS-Mono / OKVIS
  `~0.23 m`, Kimera-VIO `~0.14 m`), which captures both the missing
  visual aiding AND the missing VI initialisation (stationary
  detection + gravity alignment + velocity / bias bootstrap). All
  five `summary.txt` / `imu_dead_reckon.csv` /
  `imu_dead_reckon_errors.csv` outputs are reproducible from
  `cargo run --release --example euroc_imu_dead_reckon_demo --
  --euroc-dir <MH_xx_easy> --out-dir <out>` given a local EuRoC
  clone.
- `visloc_tracking::ImuPredictiveMotionModel` provides a
  loosely-coupled inertial pose predictor that drops into
  `Tracker`'s `MotionModel` slot in place of the rotation-static
  `ConstantPoseMotionModel` or the position-only
  `ConstantVelocityMotionModel`. The user pushes inter-frame IMU
  samples via `push_imu_measurement`, sets the world-frame initial
  velocity (and optionally bias linearisation) via setters, and the
  next `predict_pose` call forward-Euler integrates `(R, v, p)`
  through the buffer using the strapdown equations (`R_{k+1} = R_k ·
  Exp((ω − b_g)·Δt)`, `v_{k+1} = v_k + (R_k·(a − b_a) + g_world)·Δt`,
  `p_{k+1} = p_k + v_k·Δt + 0.5·(R_k·(a − b_a) + g_world)·Δt²`) and
  returns a `world_to_camera` pose prior; `observe` drains the
  buffer on success so the next window starts fresh. The predictor
  is opt-in (`Tracker::with_motion_model(...)`) and ships outside
  `pipelines/slam` so it can be paired with either
  `OnlineSlamPipeline` or `OnlineStereoVoBa`. Combined with the
  EuRoC dead-reckoning baseline + the `OnlineSlamPipeline` local
  VI-BA stage above, every level of inertial integration (rotation
  prior → joint VI-BA → measurement harness) is now reachable from
  a single workspace; benchmarking the predictor against
  `ConstantPoseMotionModel` on EuRoC sequences is the remaining
  follow-up step toward concrete OSS-parity numbers.
- `OnlineSlamPipeline` now optionally runs a sliding-window local
  visual-inertial BA stage on the critical path, closing the loop
  between the previously-staged IMU pre-integration factor and the
  appearance-only tracker / mapper. `OnlineSlamConfig.local_vi_ba:
  Option<OnlineSlamLocalBaConfig>` (default `None`) opts the pipeline
  into per-keyframe `(velocity, bias_gyro, bias_acc)` book-keeping plus
  a trailing-window BA solve every `trigger_every` IMU factors
  (default `1`), refining the last `window_size` keyframes (default
  `5`) over their `min_observations_per_landmark`-filtered (default
  `2`) landmark observations + every stored IMU factor whose `from /
  to` keyframe pair both sit inside the window. The first in-window
  keyframe's pose / velocity / bias are gauge-fixed; everything else
  is free. The refined poses + landmarks are written back into
  `map.keyframes[*].frame.pose` / `map.landmarks[*].position`, and the
  refined `(velocity, bias)` state is written back into the running
  `keyframe_state` table so the next trigger starts from the new
  linearisation point. `OnlineSlamResult.local_vi_ba` carries the
  per-trigger window ids, landmark / observation / IMU-factor counts,
  and the inner `BaResult` (LM trace + initial / final cost) for
  observability. Initial velocity for a newly-promoted keyframe is
  seeded from the inter-keyframe camera-centre displacement divided
  by the connecting factor's `delta_time` (a clean first guess when
  GT velocity is unknown); biases start at the configured
  linearisation point. The new `crate::OnlineSlamLocalBaState` carries
  the rolling state with the recent-factor history capped at `4 ×
  window_size` so memory is bounded. `reset_sequence_state` also
  clears the VI-BA state. Closes the heaviest of the three OSS-parity
  gaps (visual-inertial joint optimisation inside `OnlineSlamPipeline`
  itself, not just hint-emitting for downstream BA glue); a full
  benchmark against ORB-SLAM3 / VINS-Mono on EuRoC sequences is the
  follow-up step now that both the IMU dead-reckoning baseline and
  the VI-BA path are landed.
- EuRoC MAV stereo-inertial dataset is now a first-class input. The
  new `visloc_io::euroc` module reads `mav0/cam0|cam1/data.csv`
  (timestamp + filename image manifests), `mav0/imu0/data.csv` (200 Hz
  body-frame gyro + accel rows), the matching `cam*/sensor.yaml` /
  `imu0/sensor.yaml` calibration blobs (`T_BS` 4×4 body-to-sensor
  extrinsics, pinhole intrinsics, radial-tangential distortion, IMU
  noise + random-walk densities), and the optional
  `state_groundtruth_estimate0/data.csv` in all three published layouts
  (8-column pose-only, 13-column pose + velocity, 17-column pose +
  velocity + biases). A small line-based YAML extractor specialised for
  the EuRoC `key: value` + `key: [a, b, c]` + indented `T_BS:` block
  grammar avoids pulling in a generic YAML dependency. A composite
  `read_euroc_dataset_dir(dir)` returns the aggregate `EurocDataset`,
  and the `euroc_imu_dead_reckon_demo` example seeds a forward-Euler
  strapdown propagator from the first GT row inside the IMU window
  (position / velocity / orientation / biases) and integrates `(R, v, p)`
  through every IMU sample to produce a per-cam0-frame ATE CSV plus a
  summary `(rmse_position_m, max_position_m, rmse_orientation_deg,
  max_orientation_deg)`. The example is the explicit "lower bound"
  baseline — no visual aiding, no bias re-estimation, no zero-velocity
  updates — so any VIO pipeline layered on top can be quantified against
  the same number on the same recording. Smoke-tested on a synthetic
  hovering EuRoC fixture (10 cam0 frames @ 20 Hz, 100 IMU samples @
  200 Hz, accel exactly cancelling gravity) with ATE = 0 m / 0 deg,
  pinning down that the propagator + nearest-neighbour ATE wiring is
  correct before any real-data evaluation run.
- `OnlineSlamPipeline` exposes an optional IMU pre-integration hook
  on its keyframe stream. `OnlineSlamConfig.imu: Option<OnlineSlamImuConfig>`
  (gravity-world, bias linearisation, position / velocity / rotation
  weights) opts the pipeline into per-keyframe IMU bookkeeping;
  `push_imu_measurement(gyro, accel, dt)` folds a body-frame sample
  into the running `ImuPreintegrator`, and on every `process_frame`
  call that registers a new keyframe (`applied_update.keyframe_count
  > 0`) the pipeline snapshots the running delta into an
  `ImuPreintegrationFactor { keyframe_id_from = prev_kf,
  keyframe_id_to = frame.id, delta, gravity_world, weight_* }`,
  resets the integrator, and exposes the factor on
  `OnlineSlamResult.imu_factor` plus the `take_pending_imu_factor()`
  getter. The factor is a hint for downstream pose-graph / BA glue
  (`BundleAdjustment::add_imu_factor`,
  `ImuPreintegrationFactor::residual`) — the pipeline itself stays
  appearance-driven on the critical path (no per-frame BA, no
  velocity / bias state on the tracker). Default config leaves
  `imu = None` so existing tests / examples / `OnlineSlamPipeline::new`
  signature are unchanged; every in-repo `OnlineSlamConfig { ... }`
  struct literal (4 examples + 2 tests) gained a trailing
  `..OnlineSlamConfig::default()` for forward compatibility.
  `reset_sequence_state` also clears the integrator and the
  `last_keyframe_id` window anchor so a per-sequence reset is
  symmetric with the existing tracker / mapper reset. `ImuPreintegrator`
  gained a `PartialEq` derive so `OnlineSlamPipeline`'s `PartialEq`
  derive still compiles with the new `imu_state` field. Seven tests
  in `pipelines/slam/tests/online_slam.rs` lock in the contract: (a)
  factor emission between two keyframes 1.5 m apart (above the
  `SimpleKeyframePolicy::min_translation = 1.0` so the mapper
  actually registers a second keyframe) with the expected
  `keyframe_id_from / keyframe_id_to / delta_time = 1.0 s` book-keeping
  and `take_pending_imu_factor()` semantics, (b) IMU-free default config
  leaves every result's `imu_factor = None` and `push_imu_measurement` a
  no-op, (c) `reset_sequence_state` clears the IMU window state while
  preserving the map, (d) post-emit integrator reset — a 3-sample
  window pushed AFTER the first factor reads `delta_time = 0.3 s`, not
  `1.3 s`, so the running pre-integrator genuinely restarts from each
  keyframe, (e) full config propagation — non-default gravity,
  weights, and bias linearisation all flow into the emitted factor
  bit-for-bit (guards the wiring against weight / bias copy-paste
  regressions that would silently degrade downstream BA conditioning),
  (f) non-keyframe frames keep the window open — an intermediate
  frame that tracks but is rejected by the mapper (translation below
  the keyframe-policy threshold) does NOT emit a factor and does NOT
  reset the integrator, so all samples spanning the rejected frame
  flow into the next genuine-keyframe factor (`delta_time = 1.0 s`
  across a 4 + 6 sample split), and (g) three-keyframe chain — two
  back-to-back factors `KF1→KF2 (0.5 s)` and `KF2→KF3 (0.7 s)` emit
  with the second factor's shorter window proving the post-reset
  accumulator is genuinely starting from zero, and
  `take_pending_imu_factor` returns the latest staged factor exactly
  once. Full BA against the emitted factor remains a downstream caller
  responsibility.
- Streaming IMU pre-integration through `OnlineStereoVoBa` is now
  wired end-to-end. `OnlineStereoVoBaConfig.imu_input` carries a
  global IMU input spanning the full trajectory and the wrapper
  slices `windows[start..end - 1]` to align with each trigger's
  trailing BA window; gravity / bias linearisation / weights /
  fix-first flags are passed through verbatim so the Forster IMU
  factor stack inside `refine_stereo_vo_with_ba` works identically
  on the sliding window. The inner `ba_config.imu_input` must stay
  `None` (setting both at once returns a structured
  `InvalidImuInput`), and an `imu_input.windows.len()` that does not
  cover the trailing window also surfaces an `InvalidImuInput` in
  the trigger history instead of panicking. Two new tests in
  `pipelines/slam/src/online_stereo_vo_ba.rs::tests` cover the
  end-to-end synthetic refinement (per-keyframe velocity within
  `0.05 m/s` of truth) and the window-too-short error path. The CLI
  follows suit: `examples/stereo_vo_external_deep_files
  --online-ba` now accepts `--imu-windows-dir` or `--kitti-oxts-dir`
  + `--kitti-image-timestamps`, loads the IMU windows once at
  startup, and routes them through `OnlineStereoVoBaConfig.imu_input`
  for streaming BA (or through the post-process path when
  `--enable-ba` is set instead). The per-trigger refined IMU state
  can be dumped via `write_online_ba_imu_state_csv(path,
  &trigger_history)` (re-exported from the facade) or, on the CLI,
  via `--online-ba-imu-csv <path>` (requires `--online-ba`). Schema:
  one header line plus `(trigger_idx, window_start, window_end,
  window_kf_offset, vx, vy, vz, bg_x, bg_y, bg_z, ba_x, ba_y, ba_z)`
  per (trigger, in-window keyframe); triggers that ran visual-only
  or returned `Err` contribute no rows. The absolute frame id is
  `window_start + window_kf_offset`. Two new tests in
  `pipelines/slam/src/online_stereo_vo_ba.rs::tests` cover the
  one-row-per-keyframe layout and the header-only output on
  visual-only trigger history.
- `StereoVoBaConfig.imu_input` exposes the Forster IMU factor stack
  through `refine_stereo_vo_with_ba` and through the
  `examples/stereo_vo_external_deep_files` CLI. The post-process BA
  path accepts per-edge `StereoVoBaImuSample { dt, gyro, accel }`
  windows via `StereoVoBaImuInput { windows, gravity_world,
  bias_*_init, weight_*, bias_random_walk_weight, fix_first_bias,
  fix_first_velocity }`; the refiner pre-integrates each window with
  `ImuPreintegrator::new_with_bias`, seeds per-keyframe velocity from
  the inter-keyframe pose-centre delta over the integrated `Δt`,
  registers a bias slot at the supplied linearisation point, and ties
  consecutive biases with a `BiasRandomWalkFactor` when requested.
  Sliding-window BA + IMU is rejected with a structured
  `StereoVoBaError::InvalidImuInput`. The refinement now returns
  `imu_refinement: Option<StereoVoBaImuRefinement { refined_velocities,
  refined_bias_gyro, refined_bias_acc }>` so callers can observe the
  joint pose+velocity+bias state. A new free function
  `parse_stereo_vo_imu_samples_txt` parses one
  `dt gyro_x gyro_y gyro_z accel_x accel_y accel_z` per line (gravity
  NOT pre-subtracted from accel; `#` comments and blank lines tolerated).
  The CLI adds `--imu-windows-dir <dir>` (one `frame_NNNNNN_imu.txt`
  per inter-frame window, naming mirrors `frame_NNNNNN_temporal_matches.txt`;
  missing files are treated as empty windows), `--imu-gravity gx,gy,gz`
  (default `0,9.81,0` KITTI y-down), `--imu-weight-{position,velocity,
  rotation}`, `--imu-bias-{gyro,acc}-init x,y,z`,
  `--imu-bias-random-walk-weight`, `--imu-fix-first-bias on|off`
  (default on), `--imu-fix-first-velocity on|off` (default off);
  requires `--enable-ba` and conflicts with `--online-ba`. On success
  the CLI writes `<out>/ba_imu_state.csv` with `(id, vx, vy, vz,
  bg_x, bg_y, bg_z, ba_x, ba_y, ba_z)` per keyframe. Four new tests in
  `stereo_vo_ba::tests` cover (a) wiring on a 3-frame +2 m/s
  constant-velocity scene recovering refined velocities within
  `0.05 m/s` of truth and leaving poses within `5 mm` of input, (b)
  validation errors for wrong window count and sliding-window + IMU,
  and (c, d) parser round-trip / bad-line rejection. Streaming IMU
  through `OnlineStereoVoBa` is now wired (see the bullet on
  `OnlineStereoVoBaConfig.imu_input` above); the gyro-bias
  observability claim is pinned by
  `ba_with_imu_input_recovers_gyro_bias_under_rotation` — a 4-keyframe
  rotation-only scene with a hidden `+0.015 rad/s` y-axis gyro bias,
  `bias_gyro_init = 0`, and `fix_first_bias = false` recovers every
  refined gyro bias slot within `5e-3` of truth while accel bias and
  refined poses stay essentially unchanged.
- `BundleAdjustment` now jointly optimises per-keyframe pose,
  world-frame velocity, AND IMU bias state — the full Forster
  ImuFactor stack. `velocities` / `fixed_velocities` / `imu_factors`
  ship the velocity state and `biases: BTreeMap<u64, Vector6<f64>>` /
  `fixed_biases` ship the gyro+accel bias state, with `add_velocity` /
  `fix_velocity` / `add_imu_factor` / `add_bias` / `fix_bias` as the
  construction hooks. `ImuPreintegratedDelta` carries the Forster
  eq. 35-39 first-order bias Jacobians (`j_rotation_bg`,
  `j_velocity_{ba,bg}`, `j_position_{ba,bg}`) plus the
  linearisation-point biases used during integration, all propagated
  online by `ImuPreintegrator::integrate_sample`;
  `ImuPreintegratedDelta::corrected(b_g, b_a)` returns the first-order
  corrected `(ΔR, Δv, Δp)` and
  `ImuPreintegrationFactor::residual_with_bias_correction` lifts that
  into the Forster residual. The Schur-reduced linear system grows
  from `(6P) × (6P)` to `((6P + 3V + 6B)) × ((6P + 3V + 6B))` exactly
  when an IMU factor binds two velocities (and optionally a bias on
  the "from" side); when `V = 0, B = 0` the legacy reprojection-only
  layout is bit-identical, so every pre-existing BA test still passes.
  The Forster 2017 `[r_R; r_v; r_p]` 9-vector residual is linearised
  with all six analytical right-perturbation pose/velocity Jacobians
  (`∂r_R/∂ω = ±Jr⁻¹·R_wcⱼ`, `∂r_v/∂{ω_i, v_i, v_j}`,
  `∂r_p/∂{ρ_i, ρ_j, ω_i, ω_j, v_i}`) AND the bias Jacobian column
  (`∂r_R/∂δb_g = −Jr⁻¹(r_R) · Exp(−r_R) · J_R_bg`,
  `∂r_v/∂δb_g = −J_v_bg`, `∂r_v/∂δb_a = −J_v_ba`,
  `∂r_p/∂δb_g = −J_p_bg`, `∂r_p/∂δb_a = −J_p_ba`; Forster eq. 159
  simplified by dropping the `Jr(J_R·δb)` factor that is ≈I in the
  small-bias regime). Private `right_jacobian_inverse_so3` and
  `right_jacobian_so3` helpers cover the SO(3) right-Jacobian /
  inverse needed by the rotation residual and bias propagation.
  A bias random-walk prior factor (`BiasRandomWalkFactor`) ties two
  keyframes' bias slots with cost `weight · ‖b_j − b_i‖²` and the
  linear `±I` Jacobian, so a keyframe whose bias is unobservable
  through its own IMU window (gyro bias on straight motion, etc.)
  still gets pulled along by its neighbour. Eleven integration tests
  across `imu_preintegration::tests` and `bundle_adjustment.rs` cover
  (a) zero IMU cost on a constant-accel
  truth configuration, (b) a single IMU factor pulling a 1 m/s `v_0`
  drift back to truth with both poses + `v_1` fixed, (c) BA + two IMU
  factors pulling a 0.2 m lateral pose drift on the middle of a
  3-keyframe constant-velocity scene, (d) finite-difference
  verification of the bias Jacobians (corrected delta matches a
  re-integrated delta to `< 1e-4` for `|δb| ~ 1e-3` motion),
  (e) bias-correction no-op at the linearisation point, (f) zero cost
  with a registered zero bias, (g) BA recovering a hidden `+0.5 m/s²`
  accel bias from an inconsistent pre-integrated delta with poses +
  velocities fixed, (h) fixed-bias correction-only path, (i) zero
  cost on a 10↔20 random-walk tie sitting on top of an IMU factor,
  (j) random-walk pull alone bringing two non-fixed bias slots
  together (no IMU factor present), and (k) random-walk factor
  propagating an observable accel bias from KF10 (anchored by its
  IMU factor) to KF20 (which has no factor of its own) so both bias
  estimates converge within `1e-2` of the hidden truth.
- `visloc_slam::imu_preintegration` (`ImuPreintegrator`,
  `ImuPreintegratedDelta`, `ImuPreintegrationFactor`) ships the on-
  manifold Forster 2017 pre-integration primitive: accumulate
  body-frame `(gyro, accel)` samples between two keyframes into the
  gravity-free `(ΔR, Δv, Δp)` triplet plus the first-order
  bias-Jacobians `j_rotation_bg` / `j_velocity_{ba,bg}` /
  `j_position_{ba,bg}` (Forster eq. 35-39), then evaluate the
  gravity-compensated 9-vector residual `[r_R; r_v; r_p]` against
  world-frame `(R_i, p_i, v_i)` and `(R_j, p_j, v_j)`. Nine unit
  tests cover zero motion, constant linear acceleration / yaw rate,
  bias cancellation, body-frame rotated acceleration mapping to a
  world-z velocity, residual-vanishing under both gravity-free and
  pure-gravity motion, finite-difference verification of the bias
  Jacobians, and the bias-correction-no-op-at-linearisation-point
  property. The BA-side velocity + bias state + Jacobian wiring
  landed in the follow-up bullet above.
- `StereoVoBaConfig.gravity_prior` and the
  `examples/stereo_vo_external_deep_files --ba-gravity-prior-weight <w>`
  CLI flag thread the `GravityPrior` through the stereo VO BA
  refiner (both single-shot and sliding-window paths). The flag is
  off by default (`weight = 0` disables); a positive weight installs
  the KITTI y-down level-world prior `g_world = g_camera_observed =
  (0, 9.81, 0)`. A wiring test on a well-conditioned synthetic
  stereo scene confirms the refinement still converges at parity with
  the no-prior baseline. Combined with the stacked-factor integration
  test (reprojection + gravity + position + pairwise on a 4-keyframe
  bundle, each pose injected with a different drift mode only the
  matching prior can fix), this closes the "task #24 factor stack"
  loop: every BA-side factor type now has both a unit-level recovery
  test and an integration test proving they do not interfere inside
  the same LM solve.
- `LoopClosureConstraint::to_pairwise_pose_factor(weight)` and
  `pairwise_pose_factors_from_loop_closures(&[…], weight)` lift verified
  loop-closure edges into BA-ready `PairwisePoseFactor`s. The
  constraint's `relative_pose` (already in `T_to · T_fromⁱ`
  convention) is reused verbatim, so a verified loop now feeds
  straight into a unified BA solve alongside reprojection +
  `GravityPrior` + `PositionPrior` without a separate post-VO pose-
  graph stage. Two tests cover (a) adapter preserving ids /
  measurement / weight and dropping verifier metadata, and (b) end-
  to-end drift correction on a 3-keyframe bundle where KF40's
  translation drift is reduced to < 10 % of its input magnitude
  after BA + the lifted loop edge.
- `visloc_slam::PairwisePoseFactor` lifts external relative-pose
  measurements (IMU pre-integration, wheel odometry, verified loop
  closures) into `BundleAdjustment`. Residual
  `r = log(meas⁻¹ · T_to · T_fromⁱ)`; Jacobians `Ad(T_from)` /
  `−Ad(T_from)` reuse the `PoseGraph::optimize_se3_iterative` shape,
  with cross-pose Hessian blocks populated symmetrically. Three new
  tests cover zero-cost-at-truth, translation correction with a fixed
  anchor, and yaw correction. With `GravityPrior` (rotation) +
  `PositionPrior` (per-pose translation) + `PairwisePoseFactor`
  (pairwise relative pose), `BundleAdjustment` now covers all three
  external-sensor → BA factor shapes commonly seen in non-GPL VIO
  stacks (Kimera-VIO BSD-2, MAPLAB Apache 2.0). Full Forster on-
  manifold pre-integration with velocity / bias states is a future
  follow-up.
- Real-data loop-closure validation on KITTI 00 long-revisit segments
  via `examples/kitti_revisit_scanner_demo --frontend classical`. With
  the 30-frame start subset + the 30-frame revisit subset around frame
  4500, the appearance scanner + essential-matrix verifier detects 8
  cross-segment loop candidates with strongest pair `(KF 12, KF 4527)`
  at `25 inliers`, `inlier_ratio = 0.50`, `mean_sampson_error = 0.00057`,
  `score = 21997`. The candidate cluster `(12-18, 4511-4527)` matches
  the actual trajectory overlap. End-to-end VO + loop + PGO across the
  4500-frame gap is deferred to a future deliverable that pulls down
  the full KITTI 00 archive; for the loop-closure layer itself this
  confirms the pairwise scanner is real-data-ready.
- `visloc_slam::PositionPrior` adds per-keyframe absolute camera-centre
  priors to `BundleAdjustment`. Each `PositionPriorObservation` carries
  a target camera centre and a 3-vector `axis_weights` so an
  altitude-only GNSS / GT prior is `(0, w, 0)`. The BA Jacobian under
  right-perturbation `xi = [ρ; ω]` is `J = [−I | [C_w]_×]`; LM uses
  axis-weighted `√w` row scaling so zero weights drop their rows
  exactly. On a synthetic 3-frame bundle this recovers pure-vertical
  translation drift (the seq08-shaped failure mode where
  `GravityPrior` is provably out of scope by symmetry).
- File-backed VO + online sliding-window BA via
  `examples/stereo_vo_external_deep_files --online-ba` validates the
  `OnlineStereoVoBa` wrapper on real KITTI 00 / 900 frames: with
  `window=30, trigger_every=10` it reaches `t_rel = 1.4590 %`,
  `max_t_rel = 3.7561 %` (-41.8 % / -38.4 % vs no-BA, -28.6 % / -25.1 %
  vs single-shot post-process BA at the same window size). The
  interleaved sliding-window approach beats global post-process BA on
  this scale because each trigger sees a fresh boundary anchor and the
  refiner avoids the local minimum that global BA drifts into on a
  noisy 900-pose trajectory.
- `visloc_slam::OnlineStereoVoBa` wraps a `StereoVoFrontend` and triggers
  `refine_stereo_vo_with_ba` every `trigger_every_frames` processed pairs
  over the trailing `window_size` frames, then writes the refined poses
  back into the frontend. `StereoVoFrontend` gained a
  `temporal_matches_per_pair` field that captures filtered temporal
  matches, so the wrapper does not have to re-run the matcher. Keeping
  the BA hook outside `visloc-vision` avoids a vision → slam cycle.
- `visloc_slam::GravityPrior` adds a rotation-alignment gravity prior to
  `BundleAdjustment`. Per non-fixed pose it contributes a 3-vector
  residual `R_wc · g_world − g_camera_observed` (L2, non-robust) and a
  `J^T J / J^T r` block in `build_normal_equations` under the right
  perturbation `xi = [ρ; ω]` (`J = [0 | −R · [g_w]_×]`). The prior is
  rotation-only by design — pure translation drift (e.g. KITTI seq08's
  per-pair `Δy ≈ +0.176 m` with rotation matching GT) is documented as
  out of scope; closing that gap needs an IMU velocity or GNSS altitude
  prior.
- On the local 900-frame KITTI 00 stride-1 stereo subset (long-revisit
  triage), file-backed SP/LG VO drops from `t_rel = 2.5074 %` (no BA)
  to **`2.0432 %`** (-18.5 %) and `max_t_rel` from `6.0994 %` to
  **`5.0180 %`** (-17.7 %) with the v0.5 BA refiner (`--ba-max-init-
  residual 3 --ba-min-track-count 2000 --ba-huber-delta 3`); BA refined
  12,400 tracks / 54,608 observations in 11 LM iterations on this slice.
- EuRoC MH_05 covisibility local BA now has an opt-in pre-solve
  boundary-support gate that rejects large optimized windows when too few
  fixed boundary keyframes anchor them. The diagnostic
  [boundary-support sweep](generated/euroc_covisibility_mh05_boundary_support_gate_sweep.md)
  keeps the 400-frame MH_05 quality-gate configuration fixed and compares
  `none/0`, `7/2`, and `10/2`: `7/2` is rejected because it loses tracking
  (`0.215` vs `0.265`), while `10/2` is a candidate because it preserves the
  quality-gate-only tracking and ATE while converting two post-solve quality
  rejects into cheaper boundary-support rejects and reducing mean trigger time
  from `304.495 ms` to `254.946 ms`. This is exploratory evidence for one
  failure mode, not a default-policy or headline benchmark claim.
- `visloc_slam::refine_stereo_vo_with_ba` provides a multi-frame Schur BA
  refiner that lifts a stereo VO trajectory by chaining per-pair temporal
  matches into forward feature tracks, initialising each landmark from its
  first stereo observation, and running sparse-Cholesky LM BA over all
  poses (pose 0 fixed) with a Huber kernel. `StereoVoBaConfig` exposes
  pre-BA quality filters (`max_init_residual_px` reprojection gate,
  `min_track_count` auto-skip), a sliding-window mode (`window_size`),
  and a multi-view DLT landmark-init option (`landmark_init`). On the
  local 00-10 / 260-frame SP/LG benchmark this improves `mean_t_rel`
  from 1.4685 % (tuned SP/LG, no BA) to **1.3403 %** (-8.7 %) and
  `mean_max_t_rel` from 3.4228 % to **3.1354 %** (-8.4 %); 10 of 11
  sequences now beat the HOG/MutualSoftmax reference end-to-end.

The milestone is feature-complete for its MVP scope. Stretch tasks include:

- A public-data loop demo on real imagery (KITTI / COLMAP South Building) to
  replace the synthetic six-keyframe sequence.
- Levenberg-Marquardt damping plus robust kernels (Huber / Cauchy) on top of
  `optimize_se3_iterative`, and a sparse Cholesky / Schur-complement solver
  path so the optimizer scales beyond a handful of keyframes.
- Verifier reuse from PnP / tracking inliers via `PnPRansac` so candidates can
  be checked against the 3D map structure as well as essential-matrix
  two-view geometry.

## Rubric

- **0-20%:** map-based localization only; no sequence, VO, or loop direction.
- **20-40%:** tracking, local mapping, and SLAM composition boundaries exist.
- **40-60%:** VO frontend boundaries, external match IO, loop-candidate
  detection, and visible loop reports exist.
- **60-80%:** real external classical or learned frontend drives sequence
  tracking, and public demos show correspondences, pose continuity, and loop
  candidates clearly.
- **80-100%:** loop constraints, pose-graph hooks, regression datasets, and
  stable interfaces exist, while heavy runtimes remain optional integrations.
