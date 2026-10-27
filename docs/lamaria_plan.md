# LaMAria plan — from first official-metric scores to the leaderboard

Goal: reach **#1 on the [LaMAria leaderboard](https://lamaria.ethz.ch/leaderboard)**
(ETH, egocentric city-scale VI-SLAM on Project Aria glasses) with this
repository's Basalt Rust port, while keeping it real time. This document is the
milestone/lever plan that follows Stage 0 (the first official-metric scores,
[`lamaria_stage0.md`](lamaria_stage0.md)). Numbers here come from artifacts
named inline; nothing is tuned against ground truth unless stated.

## 1. Benchmark and metric

- **Sensors**: Project Aria, 2 SLAM cameras (ASL release: pinhole-undistorted
  ~758×572, fx≈241 → ~115° HFOV) + IMU 1 kHz. Sequences run 0.5–2.9 km / 5–48
  min; the test set is 63 sequences (Short 18, Medium 10, Long 16, Low light 9,
  Moving platform 10). The raw VRS recordings also carry 1 Hz GPS fixes
  (2.5 m median error against the pseudo-GT on sequence_4_10); by decision we
  do not use them, nor any other sensor beyond the two SLAM cameras and the
  IMU, so that the method stays visual-inertial like the leaderboard's.
- **Metric** (`cvg/lamaria` evaluator, unmodified): a robust Sim(3) alignment to
  surveyed control points, then a per-control-point piecewise score (20 points
  below 5 cm, decaying to 0 at 10 m). Report: **Score** 0–100, **CP@1m** recall,
  and `evaluate_wrt_pgt` pose recall at 1 m/5 m (pseudo-GT), using the same
  Sim(3). Submission is TUM `ns`, `world_from_imu`, all image timestamps, zipped
  as `slam/<seq>.txt`, with a 24 h submission rate limit.
- **Headlines on the leaderboard (Score per challenge, test set, as shown on
  2026-10-17)**. Aria's SLAM is the LaMAria team's run of the closed-source
  Aria system, from which the pseudo-GT pipeline is initialized. The page
  recorded on 2026-09-15 listed microSLAM at 64.5–85.6; it now shows the
  numbers below. Our rows are training sequences (batch BA with the
  revisits, lever 14), not a submission: one per track, the ones the method
  was developed on, and in the last row both per track:

  | Method | Sensors | Short | Medium | Long | Low light | Moving | Mean |
  |---|---|---:|---:|---:|---:|---:|---:|
  | Aria's SLAM (closed source) | bino + IMU | 90.7 | 78.5 | 70.9 | 84.2 | 55.0 | 75.9 |
  | GroundedSLAM | bino + IMU | 80.2 | 61.6 | 59.9 | 67.7 | 41.9 | 62.3 |
  | AnonSLAM | mono + IMU | 43.3 | 33.6 | 30.0 | 41.7 | 19.1 | 33.5 |
  | Mighty Camera | mono + IMU | 57.1 | 38.8 | 18.5 | 36.9 | 15.8 | 33.4 |
  | OpenVINS+Maplab | bino + IMU | 27.7 | 23.4 | 12.8 | 19.8 | 13.9 | 19.5 |
  | microSLAM | mono | 35.8 | 20.3 | 10.0 | 22.9 | 4.9 | 18.8 |
  | OKVIS2 | bino + IMU | 20.0 | 11.6 | 2.6 | 14.5 | 4.7 | 10.7 |
  | Ours, eight rotation segments (training: 1_19, 2_11, 3_17, 4_10, 5_11) | bino + IMU | 87.22 | 77.31 | 60.00 | 49.52 | 57.15 | 66.24 |
  | Ours, + focal lengths, keypoint sizes, low-light denoising (training) | bino + IMU | 84.14 | 81.26 | 65.38 | 63.99 | 57.70 | 70.49 |
  | Ours, + matches up to 128 s apart (training) | bino + IMU | 83.15 | 83.83 | 65.41 | 67.81 | 57.17 | 71.47 |
  | **Ours, + 60 solver iterations (30 in low light; training)** | bino + IMU | 88.16 | **85.46** | **71.43** | 67.81 | **60.32** | 74.64 |
  | Ours, the same, two training sequences per track | bino + IMU | 68.58 | 53.14 | 53.78 | 62.15 | 34.03 | 54.34 |

- Test data must be **streamed and deleted** (~862 GB total; server throttles
  per connection — use `aria2c -x16`, ~57 MB/s, from the official file list).

## 2. Where we stand (2026-09-16)

Training-sequence results, all from the official evaluator
(`docs/lamaria_stage0.md`, PR #156):

| Sequence | Track | Stage | Score | CP@1m | Pose R@5m | ATE | Sim3 scale |
|---|---|---|---|---|---|---|---|
| R_11_5cp | training | VIO only | 49.65 | 20.0 % | 100 % | 1.94 m | 0.965 |
| R_11_5cp | training | + offline mapper | 57.66 | 20.0 % | 100 % | 1.37 m | 0.960 |
| R_11_5cp | training | VIO, larger window (10 states / 30 kfs) | **63.19** | 60.0 % | 100 % | — | — |
| sequence_1_19 | Short (1.5 km, 15 min) | VIO only | 12.75 | 7.1 % | 5.4 % | — | — |
| sequence_1_19 | Short (1.5 km, 15 min) | + offline mapper | 16.98 | 0.0 % | 37.1 % | — | — |
| sequence_1_19 | Short (1.5 km, 15 min) | VIO, larger window (10 states / 30 kfs) | **27.09** | 14.3 % | 30.0 % | — | — |

Infrastructure already in place: real-time two-thread VIO pipeline (RTF 1.07
EuRoC / 1.10 R_01 near-uncontended, PR #153), online NFR mapper that keeps pace
with the VIO on EuRoC (8/11 vs ORB-SLAM3, PR #155), offline-mapper O(N²)
matching and RSS fixes (PR #156), and the streaming-trace fix that unblocked
long LaMAria sequences at 18k+ frames (PR #158, in review).

The two open gaps that the numbers expose:

1. **VIO-only drift over km / minutes** — with the default window,
   sequence_1_19 VIO-only is 12.75 and R_11_5cp 49.65; enlarging the sliding
   window to 10 states / 30 keyframes lifts them to **27.09** and **63.19**
   respectively. The mapper is still needed for Medium/Long.
2. **Mapper cost at city scale** — the online mapper runs ~0.42–0.55 s/packet
   of synchronous work vs the ~0.4 s real-time keyframe interval (still the
   real-time blocker; PR #160 parallelises detection and match/RANSAC for
   ~1.5× on this host), and the offline mapper's second global optimisation
   dominates: R_11_5cp took 44 min for 1,357 packets, `sequence_1_19` took
   **5 h 30 min for 2,615 packets**. A full test sweep (~22 h of sensor data)
   needs this before it is affordable.

## 3. Milestones

Each milestone has an explicit exit criterion and is validated on **training**
sequences only (test scores are only obtainable by submission).

| # | Milestone | Exit criterion | Status |
|---|---|---|---|
| M0 | Stage 0: first official scores; mapper usable at scale | R_11_5cp 49.65 VIO / 57.66 mapper; sequence_1_19 VIO 12.75; mapper perf/memory fixes merged | **done** (PR #156) |
| M1 | Beat the best academic baseline on Short | mapper-scored sequences ≥ 27.7, toward GroundedSLAM's 80.2 — already **R_11_5cp VIO-only 63.19** (CP@1m 60 %) with the larger window, and sequence_1_19 VIO-only 27.09 (default window: R_11 49.65 / seq_1_19 12.75) | in progress |
| M2 | Medium/Long consistency | loop-closure/mapper holds at km scale; mean Short/Medium/Long Score clearly above 27.7; no >2 km drift blow-up | not started |
| M3 | Low-light robustness | low-light training sequences score comparably to Short; no tracking loss | not started |
| M4 | Moving-platform handling | moving-platform training sequences tracked without divergence | not started |
| M5 | Full test submission | all 63 sequences processed at ≥1× RT, `slam/<seq>.txt` built, submitted within the 24 h rate limit | **done**: three submissions on GitHub Actions, mean 51.5 → 52.9, fourth on the leaderboard ([lamaria_test_results.md](lamaria_test_results.md)) |

## 4. Levers (a-priori, ranked)

These are the candidate changes, ordered by expected payoff per unit of risk.
None may be tuned against ground truth; any accuracy claim is a same-protocol
before/after on training sequences.

1. **Mapper throughput and scaling.** The online mapper's per-packet cost
   (~2.4 s vs the ~0.4 s keyframe interval) and the offline mapper's second
   global optimisation (1,151 s / 1,357 packets) are the gating costs for a
   full test sweep. Levers: parallelise per-pair matching/RANSAC and per-image
   detection, cap BoW candidates, and expose iteration/window caps for the
   second optimisation. Gate: deterministic output order and ATE within 5 % of
   the unoptimised baseline. `BASALT_ONLINE_MAPPER_TRACE=1` gives per-packet
   cost splits.
2. **Global consistency / loop closure at km scale.** sequence_1_19 VIO-only
   (1.5 km) loses the pGT recall entirely; the mapper plus loop closure must
   recover it. Levers: the inverted BoW index (already merged, but weaker on
   repetitive wide-FOV texture), loop-verification thresholds, and the
   persistent-map/pose-graph path explored on EuRoC
   ([`vi_slam_global_consistency_plan.md`](vi_slam_global_consistency_plan.md)).
3. **VIO tracking robustness / sliding window.** **Confirmed lever (config-only):**
   Basalt's default window (`vio_max_states` 3, `vio_max_kfs` 7) is far too
   small for LaMAria — raising it to 10/30 lifts R_11_5cp VIO-only CP Score
   49.65 → **63.19** (CP@1m 20 % → 60 %, pGT R@1m 16 % → 51.8 %) and
   sequence_1_19 12.75 → **27.09** (pGT R@5m 5.4 % → 30.0 %). Cost: 12 → 58 min
   for 18,352 frames; note the marginalization packets are ~3.7× larger
   (73.5 MB), so the offline file-based mapper dump is impractical at this
   window size. Also: wide-FOV optical-flow settings
   (`optical_flow_detection_grid_size`, pyramid `levels`, `pattern`,
   `vio_obs_std_dev`) — currently EuRoC values on a ~115° camera, unverified.
   **Examined and refuted:** gravity/init misalignment is *not* the dominant
   error — an opt-in gyro-compensated initial-gravity window
   (`BASALT_INIT_GRAVITY_WINDOW_MS`, branch `exp/lamaria-moving-start-init`)
   changed seq_1_19 SE3 ATE only 19.304 → 19.244 m, and loosening
   `vio_init_pose_weight` 1e8 → 1e2 did nothing.
4. **Use both cameras of the divergent Aria rig.** Status: **implemented and
   measured on three training sequences (2026-10-09)**. Combined with a
   tight gyro bias random walk (`gyro_bias_std` 1e-6 instead of Basalt's
   1e-4, which removed a systematic yaw drift) it scores sequence_1_19
   17.16 → 50.04 (OpenVINS demo estimate: 49.86), R_12_10cp 28.85 → 40.11 and
   R_11_5cp 62.87 → 63.19–63.71, at ~1.45× the cost. Now the submission
   driver's default
   ([`lamaria_multicam.md`](lamaria_multicam.md#measured-on-lamaria-training-sequences-2026-10-09)).
   The port is cam0-centric: it detects only in cam0, seeds the cam0→cam1
   stereo KLT at the same pixel (hundreds of pixels off on Aria's 75°-rotated
   pair, so stereo mostly fails), hosts every landmark in cam0 and counts only
   cam0 tracks for keyframes — effectively mono-inertial on cam0. Opt-in keys
   now add a `REPROJ_FIX_DEPTH` stereo seed from the extrinsics, cam1 FAST
   replenishment, cam1-hosted landmarks and all-camera keyframe connectivity
   (`configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json`).
   On a synthetic Aria-geometry rig it keeps vision in the window while cam0
   faces a blank wall, where the cam0-only config falls back to dead
   reckoning. RoboCap (GroundedSI) attributes its LaMAria lead to this kind of
   per-camera mono-inertial plus overlap-stereo fusion. Measure: big window
   vs big window + multicam on R_11_5cp and sequence_1_19 (commands in the
   linked doc); watch runtime, since cam1 roughly doubles the tracked points.
5. **Online focal-length estimation.** Aria's focal length changes ~0.11 %
   over a session; the current calibration is fixed per sequence. An online
   focal estimate removes a systematic scale/drift source.
6. **Low-light and moving-platform robustness.** Distinct failure modes
   (feature starvation; tram/motion dynamics). Likely needs frontend-level
   work rather than mapper tuning.
7. **Runtime headroom.** 22 h of test data at ≥1× RT is required. VIO is
   already ~1.1× (near-uncontended); the mapper (lever 1) is the open item.
8. **Aria factory IMU rectification.** Status: **measured on four
   training sequences (2026-10-10); on by default in the submission driver.** The
   ASL release ships raw IMU samples; applying the factory scale/misalignment
   and bias from the `.vrs` header raises the Score on all four non-Short
   categories (Medium 37.72 → 39.47, Long 24.14 → 28.89, Low light
   20.42 → 26.42, Moving 27.64 → 32.56; mean +4.35) with the multicam
   defaults unchanged
   ([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md)). The
   factory IMU time offset (±4.1 ms) gave no consistent gain on sequence_2_11
   and is not applied. The VIO is deterministic (a rerun is bit-identical).
   With the rectified IMU, `gyro_bias_std` 1e-7 beats 1e-6 on average over
   one sequence per track (+2.53; Long +6.87, Low light +6.00, Short +1.15,
   Medium −1.11, Moving −0.29) and is now the driver default.
9. **Online mapper on the rectified IMU.** Status: **measured on one training
   sequence per track (2026-10-12); available in the submission driver via
   `--slam-exe`.** With the rectified IMU and 1e-7, the online mapper adds
   +5.44 Score on average over the VIO (Short +1.66, Medium +13.05, Long
   +4.82, Low light +3.77, Moving +3.93; mean 38.21 → 43.66) and lowers ATE on
   every sequence with pseudo-GT, for roughly 1.15–1.4× the run time
   ([`lamaria_online_mapper.md`](lamaria_online_mapper.md)). At 1e-8 its
   effect is erratic (one wrong loop costs up to 10 points), and joint VI-BA
   does not beat the default mapper. A frame mix-up in the mapper's loop
   rotation check (IMU vs camera frame) was fixed; with a 5° gate the mapper
   mean rises to 44.95 (+1.29; Medium +4.94, Long +1.03, Short +0.47, no
   track worse).
10. **Remaining error: slow heading and scale drift.** Status: **cause found
   and corrected per device (2026-10-14); measured with the mapper on one
   training sequence per track (2026-10-15).** Fitting each sequence's CP residuals with a yaw drift and a
   scale drift linear in time brings the median CP error below 1 m on Short,
   Long and Moving. The heading drift comes from the factory IMU-camera
   calibration: the gyroscope lags the images by ~4.6 ms (factory 4.1 ms) and
   the IMU-to-camera rotation is off by 0.3-0.7 deg; the two errors partly
   cancel, so correcting one alone is worse. A per-device refinement fitted on
   training pseudo-GT only (both devices of the test set are covered) raises
   the VIO-only Score on sequence_2_11 38.36 -> 49.68 and on the held-out
   sequence_5_11 32.27 -> 43.14 and sequence_1_20 33.18 -> 38.10 (time
   alignment only on that device, whose fitted rotation overshoots). The
   submission driver applies it by default
   ([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#imu-camera-calibration)).
   Over one sequence per track it raises the VIO mean 38.21 → 49.87 and the
   mapper mean 44.95 → 50.44, but the mapper now loses to the VIO on Short,
   Long and Low light: it bends the long-range shape that the refined VIO
   gets right
   ([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper)).
   Earth-rotation compensation and a VIO outlier filter did not help and stay
   opt-in.
11. **Online mapper on the refined VIO.** Status: **studied on four tracks
   (2026-10-15); the submission driver now picks VIO or mapper per track.**
   The mapper's relative-pose factors come from the window estimates at
   marginalization, and with the big window each keyframe gets about 30 of
   them. Without loops they chain up to a trajectory up to 3 m from the
   VIO's. That chain is better than the VIO output on Medium and Moving
   platform and worse on Short, Long and Low light. The loops are short
   revisits that change little. Neither a higher relative-pose weight, fewer
   factors per keyframe, nor factors measured on the VIO output
   (`--relative-pose-max-neighbors`, `--mapper-vio-output-poses`, both
   opt-in) beats the VIO on every track
   ([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper)).
   The driver therefore submits the VIO on Short, Long and Low light and the
   mapper on Medium and Moving platform: 53.27 on one training sequence per
   track, against 51.44 with the mapper everywhere.
12. **KLT round-trip threshold in low light.** Status: **measured on four
   tracks (2026-10-16); Low light only.** 0.5 px instead of 0.2 px raises
   the low-light VIO from 34.49 to 39.00 but costs 11 points on Medium
   ([`lamaria_klt_threshold.md`](lamaria_klt_threshold.md)).
13. **Remaining scale error on Moving platform.** sequence_5_11's VIO
   measures the route about 6 % short, with or without the funicular ride.
   Removing its fitted scale drift alone would be worth about +15 there, the
   largest gain left on any track; the cause is still open. The batch BA
   (lever 14) halves the fitted scale drift (−5.4 % → −2.5 % without PB2).
14. **Offline batch VI bundle adjustment.** Status: **measured on one
   training sequence per track and on sequence_1_20 (2026-10-17).**
   `scripts/lamaria_batch_ba.py` refines the VIO trajectory afterwards: SIFT
   on 2 Hz keyframes, matches up to 128 s apart (32 s at first) under the
   VIO's epipolar geometry, and one Levenberg-Marquardt solve over every
   keyframe (pose, velocity and IMU biases with GTSAM's preintegrated
   factors), every landmark and each camera's IMU-camera rotation, with no
   ground truth. On
   1WM093701G1276 both cameras turn by about 0.1° against the IMU during a
   recording, so the solve estimates the rotations in eight time segments.
   It raises the mean over the five tracks to **64.36** (2026-10-17; 61.41
   with one rotation per camera, 59.81 before the 3 px outlier threshold of
   its last iterations), against 50.77 for the refined VIO and 53.27 for the
   per-track VIO or mapper; sequence_3_17 gains 9.9 and sequence_5_11 loses
   3.6 from the segments. Closing the revisits with a pose graph
   (`loop-correct`) and solving again from the corrected trajectory brings
   the mean to **66.24** (sequence_2_11: 71.18 → 77.31; 63.12 with one
   rotation per camera). It keeps the slow heading drift and the global
   scale of its starting trajectory: these are flat directions of its
   objective. `--modes yaw` corrects a start that drifts by degrees
   (sequence_1_20: 44.4 → 76.1) but costs points where the start is already
   good, so it stays opt-in. The G1276 sequences keep 1.4–3.5 % of scale
   drift; on sequence_2_11 the BA itself creates it (a solve started at the
   pseudo-GT drifts as much, the VIO does not). Keyframes at 4 Hz, estimated
   camera translations, Earth's rotation, a loose first rotation prior and
   leaving out the cross-camera matches did not raise the mean. The drift
   follows the ratio of the image's angular scale to the gyroscope's:
   estimating each camera's focal length per time segment (`--est-focal`;
   +0.3–0.4 % on 1WM093701G1276 by day, −0.04 % at night, −0.1 % on
   1WM09370081264) with the pixel noise growing with the SIFT keypoint size,
   and non-local means before CLAHE in low light (sequence_4_10:
   49.52 → 63.99, scale drift +3.5 % → +0.8 %), raise the mean to **70.49**
   (2026-10-17); sequence_2_11 reaches 81.26 with a pose recall of 98 %. The
   focal lengths go wrong on the moving platform, which keeps the
   calibration. Matching keyframes up to 256 steps (128 s) and 200 m apart
   instead of 64 steps and 80 m raises the mean to **71.47** (sequence_4_10:
   67.81; the second Low light sequence, sequence_4_11, 54.09 → 56.49).
   Letting each solve run 60 iterations instead of 30 (30 in low light)
   raises it to **74.64** (2026-10-18): the extra iterations move the
   solution along nearly flat drift directions, which helps the seven day
   and moving-platform sequences by 1.4–6.0 and costs the two low-light ones
   4.2–8.9. On the second training sequence of each track the mean is only
   34.03: sequence_1_20, 2_12 and 3_18 keep their VIO's heading drift
   of 4–8°, which these one-way routes give no revisit to correct and their
   far-apart views too few matches to measure; a start with that drift
   removed (from the pseudo-GT, diagnosis only) scores 55 instead of 21 on
   sequence_2_12, but no pGT-free change tried reaches it. sequence_5_12
   starts with a tram ride that the VIO and the BA halve (7.73)
   ([`lamaria_batch_ba.md`](lamaria_batch_ba.md)). An online mode
   (`online`, 2026-10-18) solves the latest 240 s every 30 s, each window
   warm-started from the one before and seeing no later data: it raises the
   mean over the ten training sequences from 54.34 to **56.42**
   (sequence_1_20 49.00 → 65.37, sequence_3_18 36.13 → 57.87, both low-light
   ones +2.0 and +4.5), with the heading error of the one-way routes a half
   to a third of the batch's; the revisit-rich sequence_1_19 drops from
   88.16 to 75.73, sequence_2_12 and the moving platform stay. Its windows
   keep up with the data on 4 cores, and with Basalt's default
   configuration the whole chain (VIO, SIFT, matching, revisits at stride
   4, windows) takes about 3.9 cores: sequence_2_11 scores 72.48 that way,
   against 83.12 from the big-window VIO, which runs at 0.42 times real time
   ([Online mode](lamaria_batch_ba.md#online-mode)). The batch steps run
   once more after the recording from the online trajectory keep its
   heading and add the revisits: sequence_1_20 73.03, sequence_3_18 67.39,
   sequence_3_17 75.57. In low light that pass drifts again (4_10: 69.79 →
   51.44), so low light keeps the online result; chosen by track, the mean
   over the ten sequences is **59.70** (Short 76.12, Medium 51.06, Long
   71.48, Low light 65.41, Moving platform 34.41;
   [Online, then one batch pass](lamaria_batch_ba.md#online-then-one-batch-pass)).
   The whole chain from Basalt's default VIO (`euroc_config`, in low light
   `euroc_config_lowlight`), causal and within 4 cores, averages **52.19**
   over the ten sequences (2026-10-19), against 54.34 for the batch pipeline
   after the recording; with the pass after the recording 55.34. That VIO
   keeps sequence_2_12's heading (online 56.41, against 19.87 from the big
   window) but fails on sequence_3_17 (11.20). Averaging the two chains'
   results helps where they agree (sequence_2_11 91.52) and lands between
   them where one fails, but needs both chains and is not used ([the
   real-time chain on all ten
   sequences](lamaria_batch_ba.md#the-real-time-chain-on-all-ten-sequences)).
   That chain failed on sequence_3_17 because its matching followed the
   drifting VIO: the epipolar band (at most 6 px) let the guide pick the
   matches that agree with its drift. With `match --tau-slope 0.3 --tau-max
   20` it averages **60.54** online (2026-10-19), above the big-window
   pipeline after the recording (59.70), and **61.19** with the pass after
   the recording (sequence_1_20 81.53, sequence_2_11 89.84, sequence_3_17
   74.95, sequence_4_11 71.07); sequence_2_12, whose VIO heading was good,
   falls to 37.75 ([Matching guided by a drifting
   VIO](lamaria_batch_ba.md#matching-guided-by-a-drifting-vio)). What the
   band changed were the matches 96–256 keyframes apart, and nearly all of
   them are wrong (2026-10-20: no peak at the pseudo-GT's epipolar lines on
   six sequences): matching only up to 64 keyframes, sequence_3_17 scores
   76.8 with either band and the windows hold the heading within ±0.55° on
   the five day sequences whose VIO drifts, but the wrong matches had held
   the starts of sequence_2_12 and 4_11, and the mean falls to 55.20 /
   56.76. A looser gyro bias random walk mends those starts and costs the
   others; a prior on the VIO's rotation changes nothing. The wide band with
   the far offsets stays ([Matches far apart are
   wrong](lamaria_batch_ba.md#matches-far-apart-are-wrong)).
   What the far matches had held was the warm-up: the gyro bias moves in
   the first minutes of a recording. A random walk ten times looser at the
   start, decaying over 300 s (`--gyro-rw-warmup`), mends sequence_2_12's
   start (online 37.75 → 56.00) and, by day with the pass (whose random
   walk is ten times looser throughout), the chain averages **64.35**
   (2026-10-20; Short 83.78, Medium 69.81, Long 67.40), above
   GroundedSLAM's 62.3; low light keeps the online result without it
   ([The first minutes, online](lamaria_batch_ba.md#the-first-minutes-online)).

## 5. Test-set operational pipeline

1. Download one sequence with `aria2c -x16` from the official test file list.
   `scripts/run_lamaria_test_submission.py --split training --sequences ...`
   runs the same steps on training sequences, so the result can be scored.
2. Run VIO (`--pipeline --threads 12`, variant-A default IMU noise), then the
   mapper (`--temporal-seed 7`; reduced-points config if it exceeds ~1 h or
   ~12 GB RSS).
3. Propagate corrections (`scripts/propagate_basalt_mapper_corrections.py`) and
   convert to the submission format
   (`scripts/basalt_tum_to_lamaria_estimate.py`). For the batch BA (lever
   14), start `scripts/lamaria_batch_ba.py` from the VIO trajectory:
   `features`, `match`, `loops`, `solve --save-states`, `loop-correct`, and
   a second `solve --loops` from the corrected trajectory, with the options
   of [its Usage](lamaria_batch_ba.md#usage) (low light: denoised features
   and 30 iterations; moving platform: no `--est-focal`). On 4 cores that
   takes about 15 min for the features (30 with the denoising), 2–7 min for
   the matching, 3–6 min for `loops`, and 10–40 min per solve at 30
   iterations, up to twice that at the default 60.
4. Delete the sequence data; add `slam/<seq>.txt` to the submission zip.
5. Submit via <https://lamaria.ethz.ch/login> (24 h rate limit).

## 6. Caveats and risks

- **R_11_5cp has only 5 control points**, so its Score/CP@1m has high variance
  and is a genuine-but-single data point, not a stable estimate.
- **All numbers to date are training sequences**, not the held-out test set;
  only a submission produces a leaderboard number.
- **Sim(3) alignment forgives global scale**, so metric-scale bias shows up as
  lower Score rather than as an obvious trajectory error.
- The evaluation is robust to a few outliers by construction; a single bad
  control point cannot dominate a sequence score.
- Compute budget: full test sweep is ~22 h of sensor data; mapper throughput
  (lever 1) must be resolved before it is affordable.

## See also

- [`lamaria_stage0.md`](lamaria_stage0.md) — method, results, mapper fixes.
- [`lamaria_multicam.md`](lamaria_multicam.md) — multi-camera VIO for the
  divergent Aria rig (lever 4).
- [`lamaria_imu_rectification.md`](lamaria_imu_rectification.md) — Aria
  factory IMU rectification and per-category results (lever 8).
- [`lamaria_online_mapper.md`](lamaria_online_mapper.md) — online mapper on
  the rectified IMU, per-track results and cost (lever 9).
- [`lamaria_drift_analysis.md`](lamaria_drift_analysis.md) — CP error
  analysis: funicular ride, heading and scale drift (lever 10).
- [`lamaria_batch_ba.md`](lamaria_batch_ba.md) — offline batch VI bundle
  adjustment, per-camera rotations, slow heading drift (lever 14).
- [`vi_slam_benchmarks.md`](vi_slam_benchmarks.md) — EuRoC VI-SLAM results.
- [`vi_slam_global_consistency_plan.md`](vi_slam_global_consistency_plan.md) —
  global-consistency evidence and staged plan.
- [`basalt_online_mapper_design.md`](basalt_online_mapper_design.md) — online
  mapper architecture.
