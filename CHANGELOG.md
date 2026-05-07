# Changelog

All notable changes to `visloc-rs` will be documented here.

## Unreleased

### Added

- PnP-based loop-closure verifier `PnPLoopClosureVerifier` (with `PnPLoopClosureVerifierConfig`, `correspondences_2d3d_for_loop_candidate`, and `verify_loop_closure_candidates_pnp` runner) that re-localizes the current frame against the candidate keyframe's landmarks via `PnPRansac`. Returns metric relative poses directly (the keyframe pose carries world scale, no `default_translation_scale` parameter needed) and reports inlier count / inlier ratio / mean reprojection error in pixels. `LoopClosureVerification` now also carries an optional `mean_reprojection_error_px` field so essential-matrix and PnP verifications can be told apart at a glance; the HTML report renders whichever metric is populated.
- `online_slam_pnp_loop_demo` example runs both verifiers side-by-side on the same loop candidate, printing the essential-matrix path (needs an externally supplied translation scale) alongside the PnP path (metric pose recovered directly) and the truth relative SE(3) for comparison.
- Levenberg-Marquardt damping plus Huber / Cauchy robust kernels on top of `PoseGraph::optimize_se3_iterative`. New `RobustKernel` enum (`None` / `Huber { delta }` / `Cauchy { c }`) selects the per-edge IRLS cost; `PoseGraphSe3Config` now exposes `robust_kernel`, `initial_lambda`, `lambda_increase_factor`, `lambda_decrease_factor`, `max_lambda`, and `min_lambda`. With `initial_lambda: None` the solver runs pure Gauss-Newton (every step accepted); with `Some(λ₀)` it runs LM (`(H + λI) δ = -g`, accept on cost decrease, otherwise revert and grow `λ`). `PoseGraphSe3IterationStats` records `lambda` and `step_accepted` per attempt. `PoseGraph::robust_se3_cost` reports the matching minimization objective.
- The pose-graph SE(3) inner solve now prefers Cholesky and falls back to LU on ill-conditioned systems, which is faster for SPD normal equations and survives `H + λI` damping reliably.
- `pose_graph_robust_demo` example runs the same outlier-prone three-keyframe loop through pure Gauss-Newton (KF30 drifts ~0.20 m off truth) and LM + Huber (`delta=0.05`, `λ₀=1e-4`); LM + Huber recovers KF30 to within ~2 mm of truth and prints the full per-iteration `λ` / accept-reject trace.
- `online_slam_public_loop_demo` example ingests a COLMAP-text-format reconstruction from disk and drives the full tracking + verifier + pose-graph SE(3) Gauss-Newton stack on it, exercising the same I/O path that real public-data reconstructions (e.g., COLMAP South Building or KITTI-derived sparse models) would take. Without flags it synthesizes a 12-keyframe / 60-landmark orbit fixture, writes it via `write_colmap_text_model` plus a paired `landmark_descriptors.txt`, reads it back, and reports `se3_cost_before ≈ 8.31 → ≈ 0.0001` (3 iterations) on a combined `[0.05, 0, -0.04]` translation + `0.18 rad` yaw drift injected on the loop-closing keyframe. With `--colmap-path <dir>` it loads a user-supplied sparse reconstruction instead, and `--descriptors-path <file>` lets callers pin landmark descriptors (otherwise synthetic per-landmark descriptors are generated so the demo stays runnable on any registered reconstruction).
- Full SE(3) Gauss-Newton pose-graph optimizer in `visloc-slam`: `PoseGraph::se3_cost`, `PoseGraph::optimize_se3_iterative`, plus `PoseGraphSe3Config` (max iterations, step / cost tolerances), per-iteration `PoseGraphSe3IterationStats`, and a `PoseGraphSe3Result` summary. Uses right-perturbation updates `T_i ← T_i · Exp(δ_i)` with a first-order BCH approximation (`J_r⁻¹ ≈ I`) so each edge contributes `r = log(meas⁻¹ · T_to · T_from⁻¹)` together with `∂r/∂δ_to = Ad(T_from)` and `∂r/∂δ_from = -Ad(T_from)`. Anchors stay fixed; rotations are now corrected alongside translations.
- SE(3) Lie-group helpers in `visloc-core::geometry::se3`: `SE3::log` / `SE3::exp` (right-perturbation `[ρ; ω]` tangent layout), `SE3::adjoint`, plus public `so3_left_jacobian` and `so3_left_jacobian_inverse` (with Taylor fallbacks for small angles). Exercised by `exp ∘ log` round-trip tests and an `Ad(T) · ξ` ↔ conjugation consistency test.
- `online_slam_pose_graph_loop_demo` example now also injects a combined `[0.04, 0, -0.03]` translation drift plus a `0.18 rad` yaw drift on the most recent keyframe and runs `optimize_se3_iterative`, taking `se3_cost_before=0.557` down to `0.000` in 2 iterations and printing per-iteration `cost_before / cost_after / max_step` together with each keyframe's post-optimization translation and rotation error.
- Deep VO / loop-close milestone completion increased to 100% to reflect the full SE(3) Gauss-Newton solver and rotation-aware demo.
- `online_slam_pose_graph_loop_demo` example exercises the full tracking + verifier + pose-graph stack on a six-keyframe synthetic loop: classical localization, verified loop-closure constraint with the matching translation scale, sparse `PoseGraph` with five sequential edges plus the loop edge, a `[0.06, 0.03, -0.05]` injected drift on the last keyframe, and a single translation-only Gauss-Newton step that takes `cost_before=0.105` down to `cost_after=0.000` and reports each keyframe's post-optimization error against the truth path. With `--out-dir` it writes `loop_demo_report.html`.
- Deep VO / loop-close milestone completion increased to 90% to reflect the end-to-end loop demo.
- Sparse `PoseGraph` skeleton with `PoseGraphEdge`, `PoseGraphEdgeKind::{Sequential, LoopClosure}`, builders (`add_pose`, `add_sequential_edge`, `add_loop_closure_constraint`, `anchor`), `translation_cost`, and a single translation-only `optimize_translations_once` Gauss-Newton step that holds rotations fixed and returns `PoseGraphOptimizationStep` diagnostics.
- `relative_world_to_camera` helper turns two `Pose`s into a `previous_to_current` SE3 measurement for `PoseGraphEdge`.
- `online_slam_loop_candidate_with_verifier_dummy` example now also injects a small drift into the most recent keyframe, builds a `PoseGraph`, and runs `optimize_translations_once` so the loop drift correction is visible: cost goes from 0.0585 to 0.0 with mean translation correction ~0.034 m.
- Deep VO / loop-close milestone completion increased to 80% to reflect the pose-graph skeleton and translation-only solver.
- `LoopClosureConstraint` plus `LoopClosureConstraint::from_verified_candidate` and `loop_closure_constraints_from_candidates` lift a verified `LoopClosureCandidate` into a stand-alone constraint (`from_keyframe_id`, `to_keyframe_id`, `relative_pose`, `inlier_count`, `inlier_ratio`, `mean_sampson_error`, `score`) ready for a future pose-graph backend; no solver lives in the crate yet.
- `LoopClosureVerification` now carries the recovered `relative_pose: Option<SE3>` so callers can build constraints (or apply their own scale) without re-running the essential-matrix RANSAC, and `LoopClosureVerifierConfig` adds `default_translation_scale` for caller-controlled translation scale.
- `online_slam_loop_candidate_with_verifier_dummy` example now also builds and prints `LoopClosureConstraint`s; the loop HTML/SVG report renders a separate Loop Closure Constraints table next to the candidate diagnostics.
- Deep VO / loop-close milestone completion increased to 70% to reflect the constraint type and verifier-output enrichment.
- `LoopClosureVerifier` trait, `EssentialMatrixLoopClosureVerifier`, `LoopClosureVerifierConfig`, `LoopClosureVerification`, and `LoopClosureVerificationFailureReason` give loop-closure candidates a classical-geometry verifier built on `visloc-vision::two_view`'s essential-matrix RANSAC, with explicit inlier count, inlier ratio, mean Sampson error, score, and enumerated failure reasons.
- `correspondences_for_loop_candidate` and `verify_loop_closure_candidates` plumb the current frame's tracking inliers and an older keyframe's observations into the verifier without forcing `OnlineSlamPipeline` callers to change their constructors.
- `LoopClosureCandidate.verification` now optionally carries the verifier's output; `geometrically_verified` is updated in place when the verifier rejects a candidate.
- `online_slam_loop_candidate_with_verifier_dummy` example demonstrates the candidate-detection plus geometric-verification path on a 12-landmark synthetic sequence; the loop HTML/SVG report adds verifier inlier counts, inlier ratio, mean Sampson error, score, and failure-reason columns.
- Deep VO / loop-close milestone completion increased to 65% to reflect the loop-closure verifier and verifier-aware demo.
- `visloc-vision::two_view` module with `TwoViewCorrespondence`, a Hartley-normalized 8-point `EightPointEssentialMatrixEstimator`, Sampson-distance-scored `EssentialRansac`, 4-fold `recover_relative_pose` cheirality decomposition, and a composing `RelativePoseEstimator` that applies a caller-supplied translation scale.
- `EssentialMatrixVisualOdometryFrontend` and `EssentialMatrixVisualOdometryConfig` expose the classical-geometry pipeline as a `VisualOdometryFrontend`, returning a full SE3 relative pose plus inlier/Sampson diagnostics and supporting per-pair translation-scale overrides.
- `two_view_vo_compare` example runs the classical essential-matrix frontend alongside the flow-only `TwoViewMatchVisualOdometryFrontend` on the same synthetic three-frame sequence to make the structural difference visible; with `--out-dir` it writes a per-frame text report.
- Deep VO / loop-close milestone completion increased to 60% to reflect the classical two-view geometry pipeline and demo.
- `track_sequence_with_two_view_match_vo_prior` example reads per-pair two-view match text files with `read_two_view_matches_txt`, populates `TwoViewMatchVisualOdometryFrontend`, and feeds the resulting VO priors through `track_frame_with_localization_prior_submap_provider` for a short three-frame sequence; with `--out-dir` it writes the generated input match files plus a per-frame text report.
- `tests/two_view_vo.rs` now covers the file-backed two-view match VO path across consecutive frame pairs to guard the `read_two_view_matches_txt` → `TwoViewMatchVisualOdometryFrontend` → `VisualOdometryPriorProvider` chain.
- Documentation now clarifies that `VisualOdometryEstimate::mean_reprojection_error` stores the mean inlier two-view flow residual in pixels when produced by `TwoViewMatchVisualOdometryFrontend`, and recommends labeling the field as `mean_flow_residual_px` in user-facing logs/reports for that case.
- Deep VO / loop-close milestone completion increased to 55% to reflect the file-backed two-view VO sequence path.
- `PoseTrajectory` and `TrajectorySample` helpers for extracting successful tracking poses, camera centers, path length, mean reprojection error, CSV output, KITTI-style 3x4 pose rows, and TUM-style trajectory rows from sequence-localization results.
- KITTI- and TUM-style trajectory parsers and file readers for reading pose rows back into `PoseTrajectory`.
- `TrajectorySummary` helper and JSON summary export for sequence-localization demos and downstream visualization scripts.
- `TrajectoryErrorSummary` and per-frame translation-error helpers for comparing estimated trajectories against reference poses.
- Optional first-matched-frame translation alignment for trajectory-error reports.
- Self-contained HTML / SVG trajectory-evaluation reports for quick visual inspection.
- Self-contained HTML / SVG single-trajectory reports for sequence-localization demos.
- Self-contained HTML tracking reports for frame-by-frame state, failures, priors, and inlier diagnostics.
- CSV export for frame-by-frame tracking state, localization counts, failures, priors, and map stats.
- `TrackingStats` JSON export for aggregate tracking summaries.
- Tracking diagnostics now distinguish motion pose priors from external localization priors such as GNSS-derived submap narrowing.
- Tracking HTML reports now summarize motion-prior and external-localization-prior usage in the top-level metrics.
- `TrackingStats::from_results` helper for rebuilding sequence diagnostics from stored tracking outputs.
- Trajectory-evaluation example showing frame-id matched translation errors, CSV output, and JSON summary output.
- File-based TUM trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based KITTI trajectory evaluation example with optional CSV / JSON / HTML output directory.
- File-based sequence localization example that tracks query feature files and prints or writes CSV / KITTI / TUM trajectory exports plus `summary.json`, `tracking.csv`, `tracking_summary.json`, `trajectory_report.html`, and `tracking_report.html`.
- Tracking sequence example with optional `tracking.csv`, `tracking_summary.json`, `tracking_report.html`, and `trajectory_report.html` output directory.
- Moving-camera GNSS-prior tracking example with optional tracking diagnostics plus `trajectory.csv`, `poses.txt`, `trajectory_tum.txt`, `trajectory_summary.json`, and `trajectory_report.html` output directory.
- GNSS-prior tracking demo output now includes an `index.html` dashboard linking the tracking report, trajectory report, CSVs, KITTI/TUM poses, and JSON summaries.
- GNSS-prior tracking demo now exports a synthetic reference trajectory plus translation-error CSV, JSON summary, and trajectory-comparison HTML report.
- GNSS-prior tracking demo output now includes `manifest.json` with generated file names and top-level tracking / trajectory / error metrics.
- Local quality checks now include a GNSS demo output smoke test for the dashboard, manifest, trajectory exports, and error reports.
- CI now runs the GNSS demo output smoke test in addition to the regular example suite.
- CI now uploads the checked GNSS demo dashboard and export directory as a `gnss-demo-outputs` artifact.
- Documentation now includes a GNSS-prior tracking demo guide with dashboard, report, export, and expected-metric notes.
- GitHub issue templates, a pull request template, contribution guide, and security policy now document the project scope and local quality gate.
- Dependabot now checks Rust crate and GitHub Actions dependencies weekly.
- CI now verifies the declared Rust 1.82 MSRV with `cargo check --workspace --all-targets`.
- Trajectory evaluation now has reusable pass/fail threshold types, evaluator CLI threshold flags, `evaluation_result.json` export, and a local trajectory-evaluation smoke check.
- Tracking statistics now have reusable pass/fail threshold types, and the GNSS-prior demo exports `tracking_evaluation.json` for tracking smoke checks.
- `GrayscaleImage` and `CornerFeatureExtractor` provide a dependency-free image feature extraction smoke path, with a new `localize_with_corner_extractor` example.
- PGM grayscale image IO now supports dependency-free image fixtures and the `localize_from_pgm` example.
- Optional `image-io` feature support for PNG/JPEG grayscale loading and the `localize_from_common_image` example.
- Optional common-image sequence loading and the `track_image_sequence_from_common_images` example.
- Common-image sequence summaries and dimension validation for image-sequence tracking inputs.
- Optional nanosecond timestamps and timestamp validation for common-image sequence inputs.
- Timestamped common-image sequence tracking example with GNSS-derived localization priors.
- Timestamp text parsing for image-sequence datasets with separate image folders and timestamp files.
- GNSS text/CSV parsing for timestamped world-position priors used by sequence localization demos.
- KITTI-style camera calibration parser for turning projection rows such as `P2` into `Camera::pinhole` inputs for automotive sequence demos.
- KITTI-style image sequence loader that combines image frames, optional timestamp files, calibration, and validation summaries.
- KITTI-style image sequence loader example that writes a small automotive-like image folder, timestamps, and calibration before reading them back.
- Local and CI smoke checks now verify KITTI-style image sequence demo outputs and upload them as a CI artifact.
- Documentation now includes a KITTI-style image sequence demo guide covering generated images, timestamps, calibration, output logs, and CI artifacts.
- Local and CI checks now verify local README/docs markdown links and anchors.
- Local and CI MSRV checks now cover all workspace targets and all features through `scripts/check_msrv.sh`.
- docs.rs metadata now builds every publishable crate with all features enabled so optional APIs are included in hosted documentation, and `scripts/package_check.sh` verifies the metadata is present.
- Local and CI checks now verify release metadata consistency across Cargo manifests, docs.rs settings, publish docs, and documented CI demo artifacts.
- README first-view copy and imagery now highlight the real public-data localization demo, robotics use case, current inputs/outputs, working demos, and explicit non-goals for readers evaluating the project quickly.
- README public-data demo assets now include a feature-rich variant with many detected image features and highlighted pose-link overlays for a clearer visual-localization first impression.
- Demo guidance now calls out feature-rich visualization and the future path for learned feature/matcher integrations without implying bundled deep models.
- Roadmap and demo strategy now make deep visual odometry and loop-closure candidate detection explicit next technical targets.
- `visloc-slam` now reports lightweight loop-closure candidates from shared verified landmarks between the current frame and older keyframes.
- `VisualOdometryFrontend`, `VisualOdometryEstimate`, and `NoopVisualOdometryFrontend` now provide the first tracking-level boundary for optional classical or learned two-frame VO integrations.
- `VisualOdometryPriorProvider` and `VisualOdometryPosePrior` convert two-frame VO estimates into current-frame pose priors.
- Two-view match text parsing supports external learned/classical matcher outputs for VO frontend experiments.
- `read_two_view_matches_dummy` demonstrates the external two-view match text bridge.
- Deep VO / loop-close milestone completion is now tracked in docs and surfaced in the README.
- `TwoViewMatchVisualOdometryFrontend` converts external two-view correspondences into a lightweight translation-only VO prior.
- `two_view_match_vo_prior_dummy` demonstrates the first file-backed bridge from external matches to `VisualOdometryPriorProvider`.
- `PLAN.md` now captures a detailed development handoff for the next Deep VO / loop-closure milestones.
- `visual_odometry_prior_dummy` demonstrates the VO-prior adapter path without bundling a model runtime.
- `track_sequence_with_visual_odometry_prior` demonstrates using a VO-derived external prior to narrow localization candidates during tracking.
- `online_slam_loop_candidate_dummy` demonstrates loop-candidate reporting on a tiny synthetic sequence.
- Online SLAM results can now be exported to a self-contained HTML/SVG loop-candidate report.
- Large README animation GIFs are excluded from the root crates.io package while remaining available on GitHub.
- `FramePriorSyncSummary` diagnostics for checking external measurement coverage against frame timestamps.
- `FramePriorSyncEvaluationConfig` and pass/fail sync evaluation for CI-checkable external sensor coverage.
- JSON export for frame-prior sync evaluation results and the timestamped image GNSS-prior demo.
- Local and CI smoke checks now verify timestamped image GNSS-prior demo outputs and sync evaluation JSON.
- CI now uploads timestamped image GNSS-prior demo outputs as a separate artifact.
- Documentation now includes a timestamped image GNSS-prior demo guide covering generated images, timestamp/GNSS files, sync evaluation JSON, and CI artifacts.

## 0.1.0 - 2026-05-07

### Added

- Workspace split into core, vision, IO, localization pipeline, and tracking pipeline crates.
- Core visual localization types: `Frame`, `Keyframe`, `VisualMap`, `Landmark`, `Observation`, `Camera`, `Pose`, and `LocalizationResult`.
- `SO3` / `SE3` pose wrappers and reprojection utilities built on `nalgebra`.
- Brute-force descriptor matching with L2 distance, ratio test, optional cross-checking, and match diagnostics.
- Minimal DLT PnP estimator, PnP RANSAC, pose-estimation diagnostics, and optional Gauss-Newton pose refinement.
- COLMAP text and binary map parsers for `cameras`, `images`, and `points3D`.
- Text parsers for landmark descriptors and query features.
- Localization pipeline over query descriptors and visual-map landmarks.
- Map providers, submap selectors, priors, localization quality gates, and map validation reports.
- Tracking skeleton with motion models, state transitions, and sequence examples.
- Local mapping skeleton with keyframe policy, local map windows, landmark candidates, linear triangulation, staged map updates, and local refinement hooks.
- Online SLAM MVP composition that combines tracking and local mapping without loop closure or global optimization.
- COLMAP text model writer for saving reusable sparse maps.
- Sensor-fusion foundation crate with timestamped frames/poses, GNSS/pose/IMU measurements, covariance types, measurement buffers, frame prior sources, and external localization-prior tracking hooks.
- GNSS-prior tracking example showing radius-submap narrowing before localization.
- COLMAP compatibility notes covering supported sparse model inputs, descriptor handling, writer behavior, and current limitations.
- Root crate prelude and top-level re-exports for common application-facing localization APIs.
- Pre-1.0 to v1.0 migration guide covering recommended imports, localization boundaries, COLMAP descriptor handling, tracking priors, and experimental layers.
- Package metadata and crate-content checks in the local quality gate and CI.
- crates.io package metadata now includes project homepage and repository URLs.
- Workspace member crates now use crate-specific descriptions and docs.rs URLs.
- Publishing guide documenting workspace crate publish order and package-check workflow.
- Examples, integration tests, design docs, local check script, and GitHub Actions CI.

### Not Yet Implemented

- Full Visual SLAM.
- Full SfM.
- Loop closure.
- Dense mapping.
- Full bundle adjustment.
- Full tightly-coupled visual-inertial or GNSS/INS fusion.
