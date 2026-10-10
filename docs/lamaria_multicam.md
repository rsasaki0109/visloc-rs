# Multi-camera VIO for divergent rigs (LaMAria / Project Aria)

**Status: implemented, opt-in, measured on three LaMAria training sequences
(2026-10-09):** it raises Score on the two longer sequences (sequence_1_19
17.16 → 31.55, R_12_10cp 28.85 → 33.46) and lowers it on the shortest
(R_11_5cp 62.87 → 59.78); see
[the measured table](#measured-on-lamaria-training-sequences-2026-10-09).
With the new keys absent, the Basalt port is bit-for-bit unchanged.

## Diagnosis: the port is cam0-centric

Project Aria's two SLAM cameras (LaMAria ASL release: pinhole-undistorted
758×572, fx ≈ 241, ~115° across the long image axis) are **rotated about 75°
relative to each other** with a 0.138 m baseline (from
`configs/basalt/variants/lamaria/R_01_easy_calib.json`: relative rotation
75.3° about cam0's x axis). They share only a ~25° band; most of what cam1
sees, cam0 never sees.

The pinned Basalt VIO (commit `0f3b2b52`) and this port assume a
fronto-parallel stereo pair such as EuRoC's:

| Stage | Pinned behaviour | Effect on a 75° rig |
| --- | --- | --- |
| Frontend detection (`stream.rs`) | FAST + grid replenishment in **cam0 only**; tracks start with no cam1 observation. | cam1's exclusive view is never tracked. |
| New-keypoint stereo KLT (`stream.rs`) | Seeds the cam1 search **at the cam0 pixel coordinates** (`SAME_PIXEL`). | The true match is hundreds of pixels away (e.g. cam0 (600, 120) lands near cam1 (650, 530) at 2–5 m), far outside the KLT's convergence basin, so almost every stereo match fails or is wrong. |
| Temporal cam1 tracking | Continues only cam1 observations that were born by stereo. | Few cam1 tracks exist at all. |
| Landmark creation (`vio/estimator.rs`) | Candidates are unconnected **cam0** tracks; every landmark is hosted in `TimeCamId(kf, 0)`; re-hosting goes to cam0. | A track seen only by cam1 can never become a landmark. |
| Keyframe decision | Connected/unconnected counts over **cam0** tracks. | Ignores cam1 entirely. |

So on LaMAria the "stereo-inertial" port is effectively monocular-inertial on
cam0, and loses vision whenever cam0 faces a texture-poor surface. A
competitor (RoboCap, see the leaderboard row below) attributes its LaMAria lead
to fusing each camera mono-inertially plus stereo where the views overlap.

## What changed (all opt-in)

The geometry code (factors, linearization, AOM, marginalization) was already
camera-generic: every visual factor reads `T_imu_cam` of the host camera
(`WindowLandmark::anchor_camera_id`) and of the observing camera, the native
host order is keyed by `(timestamp, camera)`, and same-`TimeCamId` rows are
detected per camera. The changes are in the frontend and in the estimator's
landmark/keyframe bookkeeping.

1. **Stereo seed from extrinsics** (`StereoMatchingGuess::ReprojectFixedDepth`,
   `stream::reprojected_stereo_seed`). Follows newer upstream Basalt's
   `REPROJ_FIX_DEPTH`: unproject the cam0 keypoint to a unit bearing, place
   the point `optical_flow_matching_default_depth` metres along it, transform
   with `T_cam1_cam0 = T_imu_cam1⁻¹ · T_imu_cam0`, project into cam1, and
   start the forward KLT there. A seed outside cam1 is rejected
   (`RejectReason::StereoSeedOutOfView`) without searching. Because the two
   views differ by a strong perspective change (the local cam0 → cam1 map has
   scale and shear, not just rotation), the cam1 patch is sampled through the
   local warp of that reprojection (central differences over ±1 px), and the
   SE(2) search only absorbs the residual rotation and translation. The
   backward (cam1 → cam0) search of a seeded match starts at the cam0
   keypoint with the inverse warp, and the FB² and essential-matrix gates are
   unchanged. On a rendered frame of the synthetic rig the warp matters: with
   a translation-only seed the matches were off by 0.62 px (median, max
   1.55 px, 12 matches); with the warp, 0.16 px (max 0.34 px, 22 matches).
   The same-pixel seed found 1 "match", 2168 px from the true one. In the
   rendered textured-room run, the translation-only seed's sub-pixel bias
   showed up as a ~1 % scale error (scale 1.0097, RMS 9.5 mm vs 7.1 mm for
   the pinned config); the warped seed removes it (table below). Upstream
   Basalt seeds the translation only; the warp is this port's addition and
   only runs on the opt-in path. Upstream's
   `REPROJ_AVG_DEPTH` (seed at the estimator's mean landmark depth) is not
   implemented and is rejected by the config parser.
2. **Independent cam1 detection** (`MultiCameraFlowOptions::detect_all_cameras`).
   After the essential filter, grid FAST replenishment also runs on cam1 with
   its own occupancy grid, which counts every surviving cam1 point:
   temporally tracked ones, fresh stereo matches and earlier cam1-born tracks.
   cam1-born tracks get fresh ids from the same counter (after the frame's
   cam0-born ids), have no cam0 observation, and are tracked temporally in
   cam1 by the existing per-camera map. The trace gains
   `TrackStage::Cam1GridFastReplenish` only in this mode.
3. **cam1-hosted landmarks** (`EstimatorConfig::landmarks_all_cameras`). On a
   keyframe, an unconnected track that cam0 does not see in the current frame
   but another camera does becomes a candidate hosted in the lowest such
   camera (`anchor_camera_id = 1` on a two-camera rig). These candidates are
   tried after every cam0 candidate (whose order is unchanged), and
   triangulation uses the same code: the track's full retained multi-camera,
   multi-frame history, the same 0.05 m baseline gate and the same
   `0 < ρ < 3` acceptance, with the host camera's extrinsic in the DLT. In
   practice this is temporal triangulation from earlier cam1 frames. When a
   landmark is re-hosted, it keeps its host camera (a cam1-only point may lie
   behind cam0). Verified paths:
   - `vio::window` builds every row with the host and target cameras'
     extrinsics (`cam1_hosted_landmark_rows_use_host_and_target_extrinsics`:
     zero residual at the true point in f64 and f32, large residual if the
     same parameters are read in cam0);
   - the anchored factor's analytic Jacobians match central differences for
     cam1→cam1 and cam1→cam0 on the 75° rig
     (`cam1_hosted_factor_matches_finite_difference_on_divergent_rig`);
   - the LM trial cost (`trial_objective_f32`) and the native host order use
     `(timestamp, camera)` hosts;
   - marginalization selects landmarks by host frame, so cam1-hosted
     landmarks of a marginalized keyframe are handled exactly like cam0-hosted
     ones;
   - lost-landmark detection uses observations from every camera.
4. **Keyframe decision over all cameras**
   (`EstimatorConfig::kf_connectivity_all_cameras`). The connected ratio
   counts distinct tracks seen by any camera (a stereo track counts once)
   instead of cam0 tracks only. Without it, keyframes are still driven by cam0
   alone, so while cam0 is blind no keyframe would host new cam1 landmarks
   after the minimum spacing.

Also fixed: `config.optical_flow_imu_seed_rotation` was read by
`adapter::direct_klt_config` but `BasaltConfig::from_json` rejected it as an
unknown key, so only programmatic callers (the online SLAM demo, the ROS 2
node) could set it. It is now an accepted optional key (default `false`).

## Config keys

All optional; absent means the pinned behaviour.

| Key | Values | Default | Effect |
| --- | --- | --- | --- |
| `config.optical_flow_matching_guess_type` | `"SAME_PIXEL"`, `"REPROJ_FIX_DEPTH"` | `"SAME_PIXEL"` | Stereo seed (1). |
| `config.optical_flow_matching_default_depth` | metres, > 0 | `2.0` (upstream default) | Depth along the cam0 ray for `REPROJ_FIX_DEPTH`. |
| `config.optical_flow_detect_all_cameras` | bool | `false` | cam1 FAST replenishment (2). |
| `config.vio_landmarks_all_cameras` | bool | `false` | cam1-hosted landmarks (3). |
| `config.vio_kf_connectivity_all_cameras` | bool | `false` | All-camera keyframe ratio (4). |
| `config.optical_flow_imu_seed_rotation` | bool | `false` | Existing raw-gyro temporal cam0 KLT seed, now settable from JSON. |

`configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json` is
`euroc_config_big_window.json` (10 states / 30 keyframes) with the five
multi-camera keys on (`REPROJ_FIX_DEPTH` at 2 m).

On the default depth: with fx ≈ 241 and a 0.138 m baseline, the disparity
term along the epipolar line is at most about f·b/d ≈ 33 px·m / d, so a 2 m
seed is within ~17 px of the true match for any depth from 1 m to infinity,
inside the KLT's three-level pyramid basin. The detection grid
(`optical_flow_detection_grid_size` 50) is left unchanged: 758×572 gives
15×11 cells per camera, close to EuRoC's 15×9, so there is no a-priori reason
to change it; it is a lever to measure, not a tuned value.

## Synthetic verification

`pipelines/basalt/tests/multicam_inertial_synthetic.rs`:

- **Rig:** two ideal pinhole cameras (Double Sphere, `xi = alpha = 0`),
  758×572, fx = 241.6, optical axes yawed ±37.5° (75° apart) about their
  shared image x axis, which is vertical as on Aria, and 0.138 m apart. About
  a fifth of cam1's view at 5 m is also in cam0 (checked by
  `divergent_rig_matches_aria_geometry`).
- **Scene:** a closed 11 m × 10 m × 4 m room tiled with random-intensity
  0.2 m tiles. In the *blank* scenario every surface left of `y = −0.8 m`
  turns uniform grey for 3 s: cam0, which looks front-left, keeps only a thin
  textured strip near the shared band, while cam1 still sees texture.
- **Motion and IMU:** a smooth 3-D path with roll/pitch/yaw from rest; IMU at
  200 Hz with constant biases (accel ≈ 0.07 m/s², gyro ≈ 0.0027 rad/s) and
  white noise; cameras at 20 Hz.
- **Variants on identical data (same seed):** (a) `Cam0Only` = pinned
  `configs/basalt/euroc_config.json`; (b) `StereoSeed` = (a) +
  `REPROJ_FIX_DEPTH` at 2 m; (c) `MultiCamera` = (b) + cam1 detection +
  cam1-hosted landmarks + all-camera keyframe ratio. The rendered scenarios
  run the real `DirectKltStream` through `BasaltVioEstimatorAdapter`; the
  ideal-track scenarios feed the estimator directly. In those, (a) gets cam0
  tracks only and (c) also gets cam1 observations of cam0 tracks in the
  shared band plus cam1-born tracks.

Results (release build; errors are raw, without alignment, because the
estimator starts in the true gravity-aligned frame):

| Scenario (data) | Variant | RMS / max position error | Max error from blank start | Scale | cam1 obs/frame | cam1-hosted | Blind frames |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| Textured room, rendered, 8 s, 6.1 m | (a) cam0-only | 7.1 / 12.5 mm | — | 0.9994 | 0.1 | 0 | — |
| | (b) stereo seed | **2.8** / 7.4 mm | — | 1.0017 | 94 | 0 | — |
| | (c) multi-camera | 3.3 / **6.4** mm | — | 1.0031 | 355 | 295 | — |
| cam0 blank 3–6 s, rendered, 9 s, 7.2 m | (a) cam0-only | 23.5 / 75.8 mm | 75.8 mm | 0.9924 | 0.1 | 0 | 59 / 60 |
| | (b) stereo seed | 11.6 / 43.5 mm | 43.5 mm | 0.9997 | 39 | 0 | 0 / 60 (min 26 rows) |
| | (c) multi-camera | **5.1 / 11.2 mm** | **11.2 mm** | 1.0040 | 270 | 434 | 0 / 60 (min 1244 rows) |
| cam0 blank 3–6 s, ideal tracks, 9 s | (a) cam0-only | 23.3 / 53.3 mm | 53.3 mm | 1.0081 | 0 | 0 | 59 / 60 |
| | (c) multi-camera | **4.0 / 8.5 mm** | **7.3 mm** | 0.9982 | 89 | 90 | 0 / 60 |
| Smoke (debug default): blank 1.0–2.5 s, ideal, 30 tracks/camera | (a) cam0-only | 45.8 / 101.9 mm | 101.9 mm | 0.9636 | 0 | 0 | 29 / 30 |
| | (c) multi-camera | **8.4 / 22.9 mm** | 10.5 mm | 0.9982 | 27 | 29 | 0 / 30 |

Reading the table: with full texture every variant tracks to millimetres
(the IMU is good and cam0 alone sees plenty), and the stereo seed alone
already more than halves the error because the overlap band adds metric
stereo constraints. When cam0 goes blind, the pinned config has no visual row
for 59 of 60 frames and runs on the IMU; the seed alone keeps a few dozen
stereo rows from the overlap band; the full multi-camera config keeps
hundreds of cam1-hosted landmarks and its error stays at the textured-room
level. The smoke result is bit-identical between debug and release.

"cam1 obs/frame" is the mean number of cam1 observations the frontend
emitted per frame; "cam1-hosted" is the largest number of live landmarks
hosted in cam1; "blind frames" counts frames in the blank interval whose
window solve had no visual row.

Commands:

`reprojection_seed_finds_true_stereo_matches_on_rendered_frame` (runs by
default) checks the frontend alone on one rendered frame: every cam1 match
of a new cam0 keypoint is compared with the true correspondence (the cam0
pixel's surface point projected into cam1).

```bash
# Default (debug) run: smoke test, geometry check, rendered stereo-match check.
cargo test -p visloc-basalt --test multicam_inertial_synthetic
# Full scenarios.
cargo test --release -p visloc-basalt --test multicam_inertial_synthetic -- --include-ignored --nocapture
```

Unit tests: `stream::tests::reprojected_stereo_seed_matches_independent_pinhole_projection`,
`stream::tests::cam1_replenishment_fills_only_cells_free_of_stereo_matches`,
`stream::tests::multi_camera_options_validate_the_default_depth`,
`vio::estimator::tests::cam1_only_track_is_hosted_in_cam1_only_when_enabled`,
`vio::estimator::tests::keyframe_connectivity_counts_all_cameras_only_when_enabled`,
`vio::estimator::tests::reanchoring_keeps_the_host_camera_only_when_enabled`,
`vio::window::tests::cam1_hosted_landmark_rows_use_host_and_target_extrinsics`,
`vio::aom::tests::cam1_hosted_factor_matches_finite_difference_on_divergent_rig`,
`config::tests::multi_camera_keys_default_off_and_parse`,
`config::tests::lamaria_multicam_variant_parses`,
`adapter::tests::imu_seed_rotation_is_settable_from_a_config_file`.

## Measured on LaMAria training sequences (2026-10-09)

Run on the official LaMAria ASL training data with the unmodified `cvg/lamaria`
evaluator (commit `238b6ca`, 2026-10-07): `evaluate_wrt_control_points`
(Score, CP@1m) and `evaluate_wrt_pgt` (pose recall at 1 m / 5 m),
`--corresponding_sensor imu`. Every row uses the same release binary (AVX2/FMA,
`basalt-lm-workspace-reuse`), the variant-A calibration, `--pipeline
--threads 4 --no-trace --no-marg-data`, VIO only (no mapper), on a 4-core
cloud container. One run per cell.

| Sequence (control points) | Config | Score | CP@1m | pGT R@1m | pGT R@5m | Wall | RT factor |
|---|---|---:|---:|---:|---:|---:|---:|
| R_11_5cp (5) | big window (baseline) | **62.87** | 60 % | **51.1 %** | 100 % | 761 s | 0.63× |
| R_11_5cp (5) | multicam (all keys) | 59.78 | 40 % | 39.3 % | 100 % | 1112 s | 0.43× |
| R_11_5cp (5) | multicam, no stereo seed | 56.85 | 40 % | 37.8 % | 100 % | 1103 s | 0.43× |
| R_11_5cp (5) | multicam, no keyframe change | 60.26 | 60 % | 39.7 % | 100 % | 1100 s | 0.43× |
| R_11_5cp (5) | stereo seed only (2 m) | 62.78 | 40 % | 24.8 % | 100 % | 744 s | 0.64× |
| R_11_5cp (5) | stereo seed only (10 m) | 53.46 | 20 % | 14.2 % | 100 % | 756 s | 0.63× |
| R_12_10cp (10) | big window (baseline) | 28.85 | 10 % | 8.0 % | **65.7 %** | 1547 s | 0.66× |
| R_12_10cp (10) | multicam (all keys) | 33.46 | 10 % | **10.2 %** | 65.4 % | 2199 s | 0.46× |
| R_12_10cp (10) | multicam, no stereo seed | **34.19** | 10 % | 9.3 % | 64.7 % | 2192 s | 0.46× |
| sequence_1_19 (14) | big window (baseline) | 17.16 | 7.1 % | 4.6 % | 30.0 % | 1484 s | 0.62× |
| sequence_1_19 (14) | multicam (all keys) | **31.55** | 7.1 % | **5.0 %** | **57.0 %** | 2206 s | 0.42× |
| sequence_1_19 (14) | multicam, no stereo seed | 31.46 | 7.1 % | 5.1 % | 56.9 % | 2232 s | 0.41× |

RT factor = sensor duration (frames / 20 Hz) / wall time on this 4-core host.

Reading:

- **The multicam config wins on the two longer sequences and loses on the
  shortest.** Score +4.6 on R_12_10cp and +14.4 (×1.8) on sequence_1_19,
  where pose recall within 5 m nearly doubles (30 % → 57 %): using cam1's
  field of view mostly cuts long-range drift. On R_11_5cp (5 control points)
  every variant scores at or below the baseline, and pGT R@1m drops
  51 % → 39 %.
- **The stereo seed alone hurts fine accuracy on R_11_5cp** (pGT R@1m
  51 % → 25 %; 10 m default depth is worse still), but removing it from the
  full multicam config does not help there either (56.85). On R_12_10cp the
  multicam configs with and without the seed are within run-to-run noise of
  each other.
- **The all-camera keyframe rule makes no measurable difference** on R_11_5cp.
- **Noise caveat.** One run per cell. R_11_5cp has only five control points,
  so one point flipping in or out of the 1 m band moves Score by many points;
  pGT recall (thousands of poses) is the steadier signal there.
- **Score vs Stage 0.** These Scores are not comparable with the numbers in
  [`lamaria_stage0.md`](lamaria_stage0.md): the trajectories reproduce
  (sequence_1_19 baseline pGT R@1m/5m 4.6 % / 30.0 %, identical to Stage 0),
  but the current evaluator scores the same sequence_1_19 baseline 17.16
  instead of 27.09. Compare only within this table.
- **Cost.** The multicam config is ~1.45× slower (about 2× the observations
  per frame); on this 4-core host neither config is real time.

## Gyro bias random walk: the lever that closes the gap to OpenVINS (2026-10-09)

The LaMAria demo archive ships an OpenVINS estimate for sequence_1_19. Scored
with the same evaluator it reaches **49.86** (CP@1m 28.6 %, pGT R@5m 99.7 %),
well above the multicam result above. Comparing positions against the
pseudo-GT (Sim(3)-aligned 60 s windows, `yaw range` = spread of the best local
yaw correction over the run) showed why: our trajectories rotate steadily in
yaw (multicam 14.2°, cam0-only 25.1° over 15 min) while OpenVINS stays within
8.4°, and the drift has the same shape in every config — a systematic error,
not noise.

Sweeping only `gyro_bias_std` in the variant-A calibration (Basalt's default
`1e-4`; Aria's factory value is `2.44e-4`) with the multicam config:

| sequence_1_19, multicam | Score | CP@1m | pGT R@1m | pGT R@5m | Yaw range | Sim(3) ATE |
|---|---:|---:|---:|---:|---:|---:|
| `gyro_bias_std` 5e-4 | 20.35 | 7.1 % | 4.7 % | 38.6 % | 19.9° | — |
| 2.44e-4 (Aria factory) | 24.70 | 7.1 % | 4.8 % | 45.0 % | 17.7° | — |
| 1e-4 (variant A, current) | 31.55 | 7.1 % | 5.0 % | 57.0 % | 14.2° | 4.72 m |
| 5e-5 | 37.58 | 14.3 % | 11.2 % | 88.3 % | 10.8° | 3.72 m |
| 2e-5 | 43.90 | 14.3 % | 15.5 % | 94.0 % | 7.6° | 2.95 m |
| 1e-5 | 45.66 | 28.6 % | 20.3 % | 98.4 % | 6.4° | 2.67 m |
| 5e-6 | 46.89 | 28.6 % | 21.9 % | **100 %** | 5.9° | 2.51 m |
| 2e-6 | 49.00 | 28.6 % | 23.0 % | **100 %** | 5.4° | 2.31 m |
| 1e-6 | **50.04** | 28.6 % | 23.8 % | **100 %** | **5.3°** | **2.23 m** |
| OpenVINS (LaMAria demo estimate) | 49.86 | 28.6 % | **24.2 %** | 99.7 % | 8.4° | 2.32 m |

Held-out check on R_11_5cp (5e-6): Score **63.71** (big-window baseline
62.87, multicam 59.78), CP@1m 60 %, pGT R@1m 43.9 %, Sim(3) ATE 1.08 m
(baseline 1.26 m) — the lever also fixes the one sequence where multicam lost.
At 2e-6 R_11_5cp scores 63.19 (Sim(3) ATE 1.09 m), so it is flat across 2e-6–5e-6.

Held-out check on R_12_10cp (10 control points):

| R_12_10cp | Score | CP@1m | pGT R@1m | pGT R@5m | Sim(3) ATE |
|---|---:|---:|---:|---:|---:|
| big window, `gyro_bias_std` 1e-4 (baseline) | 28.85 | 10 % | 8.0 % | 65.7 % | 4.60 m |
| multicam, 1e-4 | 33.46 | 10 % | 10.2 % | 65.4 % | 7.53 m |
| multicam, 2e-6 | 39.62 | 10 % | 11.4 % | 81.6 % | 5.02 m |
| multicam, 1e-6 | **40.11** | 10 % | **12.0 %** | **82.4 %** | 4.91 m |

So on all three training sequences multicam + a tight gyro bias beats the
big-window baseline: sequence_1_19 17.16 → 50.04, R_12_10cp 28.85 → 40.11,
R_11_5cp 62.87 → 63.19–63.71. 1e-6 edges out 2e-6 on both sequences where
both were run. `scripts/run_lamaria_test_submission.py` now defaults to the
multicam config with `--gyro-bias-std 1e-6`. (Superseded: with the
factory-rectified IMU the sweep continues to 1e-7, now the default; see
[`lamaria_imu_rectification.md`](lamaria_imu_rectification.md#re-tuning-gyro_bias_std).)

At 1e-6 sequence_1_19 now matches OpenVINS on Score (50.04 vs 49.86, a gap
within run-to-run noise) and beats it on yaw drift (5.3° vs 8.4°), Sim(3) ATE
(2.23 m vs 2.32 m) and pGT R@5m (100 % vs 99.7 %); OpenVINS keeps a slight
edge on pGT R@1m (24.2 % vs 23.8 %).

Reading: letting the gyro bias wander lets the window explain systematic
rotation error as bias and integrate it into yaw. Tightening the bias random
walk well below Basalt's default pins the bias and roughly halves the yaw
drift. This is a calibration-noise setting, not new code. It was tuned on
sequence_1_19; R_11_5cp is the first held-out check, R_12_10cp is next.

## README animation

`docs/assets/hero_vislam_lamaria.gif` is rendered from the measured runs
above (sequence_1_19): visloc-rs = multicam config + `gyro_bias_std` 1e-6
(Score 50.04), OpenVINS = the estimate in the LaMAria demo archive (49.86),
Basalt upstream configuration = `configs/basalt/variants/lamaria/euroc_config.json`
(3-state / 7-keyframe window, cam0-centric defaults) with the variant-A
calibration, run through this port (Score 7.72, Sim(3) ATE 13.8 m, yaw spread
40.6°). All three are Sim(3)-aligned to the pseudo-GT for display, as the
evaluator does.

```bash
basalt_euroc_vio_demo --euroc-dir sequence_1_19 \
  --calibration sequence_1_19_calib_gb001.json \
  --config configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json \
  --out-dir out/gifbest --pipeline --no-trace --no-marg-data \
  --dump-tracks gif_tracks.csv
python scripts/render_vislam_gif.py --images sequence_1_19 --tracks gif_tracks.csv \
  --gt sequence_1_19_pgt.txt \
  --traj "visloc-rs=out/gifbest/sequence_1_19.txt@50.0" \
  --traj "OpenVINS=demo/estimate/sequence_1_19.txt@49.9" \
  --traj "Basalt (upstream config)=out/basalt_default/sequence_1_19.txt@7.7" \
  --title "LaMAria sequence_1_19 · Project Aria · 1.0 km" \
  --out docs/assets/hero_vislam_lamaria.gif
```

## Measuring on LaMAria

Use the training sequences of [`lamaria_stage0.md`](lamaria_stage0.md)
(`R_11_5cp`, `sequence_1_19`) with the variant-A calibration and compare the
big-window config with and without the multi-camera keys. Everything else
(binary, flags, calibration, evaluator) must be identical.

```bash
cargo build --release --example basalt_euroc_vio_demo
for seq in R_11_5cp sequence_1_19; do
  for cfg in euroc_config_big_window euroc_config_big_window_multicam; do
    target/release/examples/basalt_euroc_vio_demo \
      --euroc-dir /data/lamaria/training/$seq \
      --calibration configs/basalt/variants/lamaria/${seq}_calib_variantA_default_noise.json \
      --config configs/basalt/variants/lamaria/$cfg.json \
      --out-dir target/lamaria_multicam/$seq/$cfg \
      --pipeline --threads 12 --no-trace --no-marg-data
    python scripts/basalt_tum_to_lamaria_estimate.py \
      --in-tum target/lamaria_multicam/$seq/$cfg/trajectory.tum \
      --out-estimate target/lamaria_multicam/$seq/$cfg/${seq}_estimate.txt
  done
done
```

Then score each `*_estimate.txt` with the unmodified `cvg/lamaria` evaluator
exactly as in Stage 0: `evaluate_wrt_control_points` (Score, CP@1m) and
`evaluate_wrt_pgt` (pose recall at 1 m / 5 m with the control-point Sim(3)),
`--corresponding_sensor imu`. (Check the evaluator's `--help` for the input
paths of your checkout; Stage 0 ran it from a WSL venv.)

Record per sequence: Score, CP@1m, pGT R@1m/R@5m, wall time, and from the
demo summary the frame count. Also worth recording: the mean number of cam1
observations per frame and the number of cam1-hosted landmarks (both visible
through `BasaltAdapterOutput::tracks` and
`BasaltVioEstimator::landmark_count_by_host_camera`). Because one control
point can swing the Score by ~14 points, rank on both sequences together and
on CP@1m / pGT recall, not on a single Score. A useful ablation is the
stereo seed alone (only the two `optical_flow_matching_*` keys), which
corresponds to synthetic variant (b).

## Limits and open items

1. **No LaMAria measurement yet.** The synthetic rig matches Aria's geometry
   but not its imagery (rolling shutter, exposure changes, low light, motion
   blur, real texture statistics) or its 10 Hz-class camera rate.
2. **Runtime.** cam1 detection roughly doubles the frontend's tracked points
   and adds cam1-hosted landmarks to the window; with the 10/30 window,
   which already costs ~5× the default, real-time headroom on long sequences
   must be re-measured.
3. **No cam1 → cam0 matching.** cam1-born points are not searched in cam0, so
   a point first detected in cam1 inside the shared band stays cam1-only
   (its cam1 temporal track still gives it a landmark). The cam1 occupancy
   grid avoids most duplicates because stereo matches of cam0 points occupy
   the shared band first.
4. **Seed uses a fixed depth.** `REPROJ_AVG_DEPTH` (feeding the estimator's
   landmark depth back to the frontend) and using a known landmark's depth are
   not implemented.
5. **Temporal cam1 KLT has no IMU rotation seed.**
   `optical_flow_imu_seed_rotation` seeds cam0 only; seeding cam1 as well is a
   natural follow-up for fast head rotation.
6. **Only two cameras exercised.** The estimator code is written for any
   camera count (host = lowest observing camera), but the frontend detects
   only in cam0 and cam1.
7. **Mapper unchanged.** The NFR mapper still uses its own stereo matching;
   MargData carries cam1 observations and cam1-hosted landmarks are
   marginalized like cam0 ones, but the mapper has not been evaluated on such
   packets.
