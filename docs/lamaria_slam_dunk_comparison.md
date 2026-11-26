# SLAM Dunk and visloc-rs, side by side

SLAM Dunk (Van Meter and Paul, "SLAM Dunk: A recipe for long-horizon egocentric SLAM in the wild", 2026,
[alphaXiv 2611.slam-dunk](https://www.alphaxiv.org/abs/2611.slam-dunk), code
[jacobvm04/slam-dunk](https://github.com/jacobvm04/slam-dunk), AGPL-3.0) models two effects in an offline
bundle adjustment: the camera calibration drifting over time and the gyroscope's 3×3 intrinsics. This page
sets the paper's numbers and settings next to what visloc-rs measured when it tried the same ideas. It is
written from the paper alone; no SLAM Dunk code was read or copied.

## The numbers

The paper scores 13 training recordings with control points (R_11_5cp, R_12_10cp, R_13_15cp and the ten
sequence_* walks; visloc-rs has scored twelve, without R_13_15cp) with the official evaluator, and the ATE on 21 with dense references.

| | Paper (13 recordings) | visloc-rs |
|---|---:|---:|
| Baseline | 69.1 (64.9 on the ten walks) | 68.14 on twelve on Actions (the chain now, [The twelve on Actions](lamaria_batch_ba.md#the-twelve-on-actions)); the paper quotes 63.4 for visloc-rs on the ten walks (commit d47cdd8) |
| + gyroscope matrix | 71.5 (+2.3, 7 of 13 gain) | not kept by day (sequence_2_12 −1.4); +1.3 on sequence_1_19, −0.2 on sequence_3_17 |
| + time-varying camera | 77.1 (+8.0, 9 of 13) | −3 to −6 on sequence_3_17 in every variant tried |
| both | **82.4** (+13.3, 11 of 13; ATE 1.02 → 0.43 m) | sequence_3_17: CP 79.9 → 73.1, ATE 1.48 → 1.05 m |
| Test leaderboard mean | 75.5 | 58.7 (submission 12) |

The paper's removal study (each term taken out of the full system): camera path −11.0 CP, gyroscope matrix
−5.3, accelerometer intrinsics −3.3, clock offsets −2.3, thermal terms −0.5. Its per-recording plot puts
sequence_3_17 among those that gain with both additions; sequence_1_20 loses 4.8 and sequence_5_12 stays
unsolved (7 → 6), as it does here (6.4).

So the two baselines are level, and the paper's gain is the camera path. visloc-rs's camera path lowers
the ATE as the paper's does (sequence_3_17: 1.48 → 1.05 m against the paper's 1.02 → 0.43 m on average)
but loses on the control points, where the paper gains.

## Where the two differ

| | SLAM Dunk (paper, Table 5 and Section 4) | visloc-rs |
|---|---|---|
| Images and camera model | the factory **fisheye** model, focal lengths free | the ASL release's **rectified pinhole** images (115° × 100° of the view), focal scale on the pinhole K |
| Focal parameters | f_x and f_y per camera per 5 s chunk; walk 0.035 px, absolute 0.5√K px, f_x − f_y 0.05 px | one log scale (f_x and f_y together) per camera per segment; walk 0.035 px converted on the pinhole K, absolute 1 % |
| Stereo | cam0's transform to the IMU fixed; cam1 relative to cam0 per 5 s chunk: rotation (1 mrad), translation (0.5 mm), baseline length 0.5 mm | the same structure (`--stereo-reference 0`), translation optional (`--fix-stereo-translation`) |
| Biases | per 5 s chunk; gyroscope walk 2×10⁻⁵ rad/s around the temperature trend, absolute 0.002√K; accelerometer walk 0.005 m/s² | GTSAM combined IMU factors per keyframe, random walk from the calibration (gyro 1×10⁻⁷, ×10 in the pass) |
| Gyroscope matrix | I + G, σ_G 0.002 per entry, one per recording | the same (`--est-gyro-m`) |
| Accelerometers | both IMUs fused; misalignment 0.01 rad, scale 0.003 / 0.001 | the right IMU; `--est-accel-m`; fusion tried on the benchmark branch |
| Clock | IMU–camera and accelerometer–gyroscope offsets per recording, 1 ms prior | per-device delays fitted on the training pseudo-GT; a sweep found the accelerometer within 1 ms |
| Keyframes | every max(0.6 s, T / 1500) (1.2 s for 30 min) | 2 Hz |
| Front end | 900 RootSIFT per image, lags 1–32 and stereo, ratio 0.78, essential-matrix RANSAC; short optical-flow tracks; revisits by PnP within 25 m | SIFT matched within the VIO's epipolar band; revisits within 15 m and 60° |
| Robust passes | Huber 1 px, drop > 3 px; adaptive Cauchy 0.45–1.5 px; a last pass with the calibration frozen, tracks recovered within 0.8 px, SIFT-scale weights | Huber with keypoint-size sigmas, outliers over 10 px; the frozen-calibration pass tried (−0.10) |
| Solver | Ceres, up to 250 iterations a pass | own Levenberg–Marquardt with a banded Schur complement, 60 a pass |

## What the comparison suggests

1. **The camera model is the first suspect.** The paper estimates the focal lengths of the fisheye model
   on the raw images. visloc-rs estimates a scale of the rectified pinhole images. A change of the fisheye's
   focal length moves a pinhole keypoint radially by f (tan θ' − tan θ), 2.1 times a pinhole scale of the
   same size at 57° off axis ([The raw fisheye images](lamaria_batch_ba.md#the-raw-fisheye-images)). A
   pinhole scale cannot follow it, so the 5 s focal walk fits something else. That would explain a lower
   reprojection cost and ATE with worse control points on sequence_3_17's last ten minutes.
2. **f_x and f_y apart.** The paper keeps them separate, tied by 0.05 px. visloc-rs ties them into one
   scale.
3. **The bias model.** The paper's gyroscope bias is a walk per 5 s chunk around a temperature trend. Its
   σ_b is 2×10⁻⁵ rad/s per chunk, about 9×10⁻⁶ rad/s/√s, looser than visloc-rs's pass, and it is tied to
   the thermal slope. visloc-rs tightening the walk (×3) lost 0.62; loosening it with the thermal trend as
   the mean was not tried.
4. **The terms come as a set.** In the paper the extras that add 0.5 CP to the baseline cost 5.6 when they
   are taken out of the full system: the camera path and G need the accelerometer intrinsics and the clock
   offsets. visloc-rs added the camera path to a baseline without the clock offsets or the dual
   accelerometers.

The next experiment, by that reading: the focal walk on the fisheye model of the raw images (f_x and f_y
apart, the paper's priors), then the clock offsets per recording, measured with the twelve-sequence
workflow ([The twelve on Actions](lamaria_batch_ba.md#the-twelve-on-actions)).

## The paper's recipe on the thirteen, as far as visloc-rs has it

The training evaluation ([run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45155545321), commit
`a2a26f6`) paired the default chain with an experimental pass that takes the paper's camera and IMU model
as far as visloc-rs implements it: the factory fisheye model with the fisheye's focal lengths
(`--ba-fisheye-model`), cam0 fixed and cam1's rotation and translation per 5 s chunk against it, both focal
lengths per chunk with the paper's walks (1 mrad, 0.5 mm, 0.035 px; `--ba-stereo-reference 0
--ba-stereo-paper-priors`), and the gyroscope and accelerometer matrices. The temperature terms and the
rematch run in both. Not in it: the clock offsets per recording, the two accelerometers fused, f_x and f_y
apart, the paper's bias model and robust passes.

| Recording | Default | Paper's model | Change | Paper, both additions |
|---|---:|---:|---:|---|
| R_11_5cp | 87.90 | 86.00 | −1.90 | gains |
| R_12_10cp | 63.74 | 57.23 | −6.51 | gains |
| R_13_15cp | 78.10 | 75.59 | −2.51 | gains |
| sequence_1_19 | 85.34 | 83.93 | −1.42 | gains |
| sequence_1_20 | 86.38 | 81.47 | **−4.90** | **−4.8** |
| sequence_2_11 | 88.47 | 84.21 | −4.26 | gains |
| sequence_2_12 | 51.02 | 61.33 | +10.31 | gains |
| sequence_3_17 | 80.38 | 73.93 | −6.45 | gains |
| sequence_3_18 | 57.78 | 53.94 | −3.84 | gains |
| sequence_4_10 | 68.18 | 67.66 | −0.52 | gains |
| sequence_4_11 | 76.05 | 75.98 | −0.07 | gains |
| sequence_5_11 | 66.10 | 68.04 | +1.94 | gains |
| sequence_5_12 | 5.62 | 5.76 | +0.15 | unsolved (7 → 6) |
| Mean of thirteen | 68.85 | 67.31 | −1.54 | 69.1 → 82.4 |

Without sequence_2_12 (±5 between runs) the twelve lose 2.52 and two gain: not kept. The paper's one loss,
sequence_1_20 (−4.8), comes out the same here (−4.90), but where the paper gains on the other eleven, this
pass loses on nine of them, most on R_12_10cp and sequence_3_17. The fisheye model therefore does not
explain the difference by itself. What remains between the two is the clock offsets per recording (the paper
loses 2.3 CP without them), the fused accelerometers, f_x and f_y apart, the bias walk per 5 s chunk around the
temperature trend, the robust passes, the keyframe spacing and the front end. The next measurement is the
fisheye model alone, without the time-varying calibration, to see whether the camera model or the camera
path costs these points.

## The fisheye model alone

The same evaluation with only `--ba-fisheye-model` in the experimental pass, the default calibration
otherwise (both cameras' rotations and focal lengths in 8 segments; [run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45228319544),
commit `7ea111b`). R_12_10cp's scoring step failed on a timeout fetching the evaluator's pycolmap from the ETH
server; its two estimates were scored locally with the same evaluator.

| Recording | Default | Fisheye model | Change | Paper's model (above) |
|---|---:|---:|---:|---:|
| R_11_5cp | 87.90 | 85.93 | −1.97 | −1.90 |
| R_12_10cp | 63.74 | 60.28 | −3.46 | −6.51 |
| R_13_15cp | 78.14 | 78.28 | +0.14 | −2.51 |
| sequence_1_19 | 85.34 | 84.64 | −0.70 | −1.42 |
| sequence_1_20 | 86.70 | 85.86 | −0.84 | −4.90 |
| sequence_2_11 | 88.56 | 80.05 | −8.50 | −4.26 |
| sequence_2_12 | 51.02 | 61.45 | +10.43 | +10.31 |
| sequence_3_17 | 80.35 | 78.98 | −1.37 | −6.45 |
| sequence_3_18 | 38.86 | 39.14 | +0.28 | −3.84 |
| sequence_4_10 | 68.18 | 65.68 | −2.50 | −0.52 |
| sequence_4_11 | 76.05 | 74.75 | −1.30 | −0.07 |
| sequence_5_11 | 66.92 | 66.92 | 0.00 | +1.94 |
| sequence_5_12 | 6.33 | 6.49 | +0.17 | +0.15 |
| Mean of the twelve without 2_12 | | | **−1.67** | **−2.52** |

The fisheye model alone costs 1.67 on the twelve, four of them gaining; the paper's time-varying path on top
of it costs another 0.85. In this pipeline the camera model the paper uses is itself the larger loss, so the
gap to the paper is not in how the calibration varies over time. Both runs put sequence_2_12 at about +10,
the same pass both times: its default chain is the one that varies (51 here and in every run so far against
61 with either model).

The default chain's sequence_3_18 fell from 57.78 to 38.86 in this run because its IMU was not rectified:
fetching the head of the `.vrs` hit "Connection reset by peer" and the driver went on with the raw IMU
(both chains of the pair share it, so the pair still compares like with like). Apart from that, the chain is
reproducible on Actions: across four runs the default chain repeats within about 0.3 on most recordings
(sequence_2_12 at 51.02 every time; sequence_5_11 moves by up to 2.3). The driver now retries its downloads
(5, 15, 45 and 120 s apart), and the training evaluation fails a job that ran on the raw IMU instead of scoring
it; the evaluator's packages are fetched with retries too.

## The clock offsets per recording

`solve --est-time-offsets` estimates the IMU's clock against the cameras' and the accelerometer's against the
gyroscope's (prior 1 ms; the paper loses 2.3 CP without them), as two more entries of the calibration vector
that carries the gyroscope matrix. On a window of sequence_3_17 with its IMU shifted by 3 ms on purpose, the
estimate moves by 3.00 ms. On the thirteen ([run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45283299894),
commit `aff5383`), now decided on all thirteen since the chain repeats within about 0.3:

| Recording | Default | Clock offsets | Change | IMU–camera (ms) | Accelerometer–gyroscope (ms) |
|---|---:|---:|---:|---:|---:|
| R_11_5cp | 88.16 | 86.04 | −2.12 | −0.69 | −0.48 |
| R_12_10cp | 64.07 | 65.09 | +1.02 | −0.81 | −0.46 |
| R_13_15cp | 78.10 | 78.33 | +0.23 | −0.37 | −0.53 |
| sequence_1_19 | 85.34 | 84.61 | −0.73 | −0.21 | +0.01 |
| sequence_1_20 | 86.70 | 84.90 | −1.80 | −0.18 | +0.16 |
| sequence_2_11 | 88.49 | 85.73 | −2.76 | −0.72 | −0.55 |
| sequence_2_12 | 51.02 | 42.80 | −8.22 | −1.20 | −0.43 |
| sequence_3_17 | 80.33 | 73.50 | −6.84 | −0.98 | −0.30 |
| sequence_3_18 | 57.76 | 54.37 | −3.39 | −1.21 | −0.45 |
| sequence_4_10 | 68.22 | 68.95 | +0.73 | −0.20 | −0.62 |
| sequence_4_11 | 76.05 | 74.35 | −1.70 | −0.18 | −0.78 |
| sequence_5_11 | 66.10 | 68.84 | +2.74 | −1.34 | −1.03 |
| sequence_5_12 | 6.19 | 6.21 | +0.02 | −0.72 | −0.83 |
| Mean of thirteen | 68.96 | 67.21 | −1.76 | | |

Five of thirteen gain: not kept. The offsets come out small and alike (−0.2 to −1.3 ms between the IMU and
the cameras, about −0.5 ms between the accelerometer and the gyroscope), so the per-device delays the driver
already applies leave little to find, and what the solve finds costs control points.

## What the five measurements say

Every term of the paper tried here lowers the solve's cost and, where measured, the ATE against the
pseudo-GT, and every one of them loses on the control points (camera path with the fisheye model −2.52, the
fisheye model −1.67, the clock offsets −1.76, the accelerometer matrix and the 5 s focal lengths on
sequence_3_17). The temperature terms by day are the exception. The paper's terms work inside its own
objective: its inertial residuals are whitened with fixed sigmas and additive floors (accelerometer noise
3×10⁻³ m/s²/√Hz against visloc-rs's 1.6×10⁻², the gyroscope 5×10⁻⁴ rad/s/√Hz against 2.8×10⁻⁴), its biases
walk per 5 s chunk, and its robust passes end with the calibration frozen. In visloc-rs's objective the same
freedom fits the pseudo-GT better and the surveyed points worse, the pattern sequence_3_18's drift showed
before ([Where the second sequences' heading drift comes from](lamaria_batch_ba.md#where-the-second-sequences-heading-drift-comes-from)).
Adopting the paper's terms would mean adopting its objective first.

## The paper's IMU noise

The first step towards the paper's objective: its noise densities and bias walks in the pass
(`--accel-noise 3e-3 --gyro-noise 5e-4 --accel-bias-rw 2.2e-3 --gyro-bias-rw 8.9e-6`, the walks per 5 s chunk
taken per √s, the pass's ×10 off; [run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45329522916),
commit `b8471fc`):

| Recording | Default | Paper's IMU noise | Change |
|---|---:|---:|---:|
| R_11_5cp | 87.75 | 83.11 | −4.63 |
| R_12_10cp | 63.88 | 66.77 | +2.90 |
| R_13_15cp | 78.10 | 76.98 | −1.11 |
| sequence_1_19 | 85.34 | 91.34 | +6.00 |
| sequence_1_20 | 86.38 | 90.65 | +4.27 |
| sequence_2_11 | 88.47 | 84.10 | −4.37 |
| sequence_2_12 | 51.02 | 43.45 | −7.57 |
| sequence_3_17 | 80.33 | 79.95 | −0.38 |
| sequence_3_18 | 57.78 | 56.80 | −0.98 |
| sequence_4_10 | 68.18 | 64.61 | −3.57 |
| sequence_4_11 | 76.05 | 72.14 | −3.92 |
| sequence_5_11 | 69.02 | 66.50 | −2.52 |
| sequence_5_12 | 6.18 | 6.44 | +0.26 |
| Mean of thirteen | 69.11 | 67.91 | −1.20 |

Four of thirteen gain: not kept. It moves the recordings more than any term so far, both ways: the two
Short walks of the second device gain 4–6 (sequence_1_19 reaches 91.3, its best), R_12_10cp gains 2.9, and
the longer and the low-light recordings lose up to 7.6. The noise is half of the objective's IMU side; the
paper's also has bias states per 5 s chunk with an absolute prior and fixed whitening with additive floors,
which GTSAM's preintegrated factors here do not have, so this run does not yet put the paper's camera terms in
the paper's objective.

## The paper's IMU model: biases per 5 s chunk

`solve --bias-chunk-seconds 5` holds the biases nearly constant within 5 s chunks and lets them step by the
paper's sigmas between them (0.005 m/s², 2×10⁻⁵ rad/s; a thousandth of that inside a chunk), through the
per-factor bias covariance of the combined IMU factors. With the paper's noise densities as above
([run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45384406106), commit `39841c6`):

| Recording | Default | Paper's IMU model | Change | Noise only (above) |
|---|---:|---:|---:|---:|
| R_11_5cp | 88.16 | 82.05 | −6.12 | −4.63 |
| R_12_10cp | 63.74 | 69.45 | +5.71 | +2.90 |
| R_13_15cp | 78.10 | 77.58 | −0.52 | −1.11 |
| sequence_1_19 | 85.34 | 90.66 | +5.32 | +6.00 |
| sequence_1_20 | 86.69 | 85.35 | −1.35 | +4.27 |
| sequence_2_11 | 88.47 | 87.46 | −1.02 | −4.37 |
| sequence_2_12 | 51.02 | 41.10 | −9.92 | −7.57 |
| sequence_3_17 | 80.35 | 80.27 | −0.07 | −0.38 |
| sequence_3_18 | 57.76 | 59.35 | +1.58 | −0.98 |
| sequence_4_10 | 68.23 | 63.45 | −4.78 | −3.57 |
| sequence_4_11 | 76.06 | 73.21 | −2.84 | −3.92 |
| sequence_5_11 | 65.44 | 67.54 | +2.10 | −2.52 |
| sequence_5_12 | 6.11 | 6.54 | +0.43 | +0.26 |
| Mean of thirteen | 68.88 | 68.00 | −0.88 | −1.20 |

Five of thirteen gain: not kept, but the closest of the paper's terms so far. Without the two low-light
recordings (whose pass has its own settings) and sequence_2_12, the ten gain 0.61 on average, five of them.
sequence_2_12 loses 7.6–9.9 under every run with the paper's noise; R_12_10cp, sequence_1_19 and 3_18
gain with the chunked biases.

## The paper's camera path on the paper's IMU model

Both chains on the paper's IMU model (its noise, biases per 5 s chunk; the driver's `--ba-common-pass-args`),
the experiment adding the paper's camera path as before (fisheye model, cam1 per 5 s against cam0, both focal
lengths per 5 s, the gyroscope and accelerometer matrices;
[run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45438023214), commit `52632ba`):

| Recording | Paper's IMU model | With the camera path | Change | Same camera path, default IMU (above) |
|---|---:|---:|---:|---:|
| R_11_5cp | 82.05 | 84.20 | +2.15 | −1.90 |
| R_12_10cp | 69.45 | 56.30 | −13.15 | −6.51 |
| R_13_15cp | 77.58 | 78.93 | +1.35 | −2.51 |
| sequence_1_19 | 90.65 | 87.34 | −3.31 | −1.42 |
| sequence_1_20 | 85.37 | 69.45 | −15.92 | −4.90 |
| sequence_2_11 | 87.42 | 82.26 | −5.16 | −4.26 |
| sequence_2_12 | 41.09 | 56.14 | +15.05 | +10.31 |
| sequence_3_17 | 80.10 | 73.72 | −6.38 | −6.45 |
| sequence_3_18 | 59.41 | 52.59 | −6.82 | −3.84 |
| sequence_4_10 | 63.36 | 61.73 | −1.63 | −0.52 |
| sequence_4_11 | 73.24 | 70.76 | −2.48 | −0.07 |
| sequence_5_11 | 68.02 | 70.12 | +2.10 | +1.94 |
| sequence_5_12 | 6.68 | 6.49 | −0.18 | +0.15 |
| Mean of thirteen | 68.03 | 65.39 | **−2.64** | |

Four of thirteen gain: not kept. With the IMU side of the objective moved to the paper's, the camera terms
lose more than on visloc-rs's own IMU model, and lose most on R_12_10cp and sequence_1_20 (−13, −16). The
IMU model alone reproduces its earlier run (68.03 against 68.00). So the paper's hypothesis, that the
remaining gap is the camera calibration's drift and the gyroscope's intrinsics, does not carry over to this
pipeline even with the paper's camera model, priors, IMU noise and bias model: whatever makes those terms
pay in SLAM Dunk lies in the parts not reproduced here (the robust passes and the frozen-calibration
refinement, the fixed whitening floors, the 0.6–1.2 s keyframes, the RootSIFT front end with flow tracks,
the fused accelerometers, Ceres).

## SLAM Dunk itself, config by config

The [`slam_dunk_reference`](../.github/workflows/slam_dunk_reference.yml) workflow runs SLAM Dunk as published
(commit `38896a3` of jacobvm04/slam-dunk, its Docker image, its own extraction from the `.vrs`; only executed,
nothing copied) with each of its configs end to end (`run`; this release's `refine` rejects the packets its
own `prepare` writes), and the trajectories are scored with the same evaluator
([run](https://github.com/rsasaki0109/visloc-rs/actions/runs/45550732914); scored locally, the job's
evaluator lacked `tqdm`).

| SLAM Dunk config | sequence_3_17 | R_12_10cp | sequence_1_20 |
|---|---:|---:|---:|
| baseline (no camera path, no G; with the accelerometer, clock and thermal terms) | 52.65 | 81.09 | 95.12 |
| camera_only | 64.26 | 78.42 | 90.73 |
| gyro_only | 61.32 | 80.56 | 94.72 |
| **full** | **87.64** | **82.13** | **90.83** |
| no_clock (full without the clock offsets) | 71.60 | 81.70 | 89.99 |
| no_accelerometer | 69.56 | 81.04 | 89.62 |
| no_thermal | 86.96 | 82.01 | 91.20 |
| no_final_refinement | 86.51 | 80.95 | 89.27 |
| ladder_0 / ladder_1 / ladder_2 | 56.70 / 48.36 / 52.32 | 79.43 / 81.03 / 81.28 | 94.30 / 94.94 / 95.48 |
| no_auxiliary | 54.75 | 80.94 | 89.24 |
| visloc-rs, default chain | 80.35 | 63.88 | 86.70 |

Two things follow:

- **On sequence_3_17 the terms pay only together.** SLAM Dunk's own baseline scores 52.65, far below
  visloc-rs's 80.35; the camera path alone or G alone adds 9–12; all of them together reach 87.64, and taking
  out just the clock offsets (71.60) or the accelerometer intrinsics (69.56) loses most of it again.
  visloc-rs's "paper's model" pass above had no clock offsets and scored 73.93 there, next to SLAM Dunk's
  no_clock 71.60. The terms were never measured here all at once.
- **sequence_1_20 is the paper's one loss, and it reproduces.** SLAM Dunk's baseline scores 95.12 and its
  full model 90.83 (−4.29; the paper: −4.8), against visloc-rs's 86.70; G alone keeps it at 94.72, the camera
  path costs it.
- **On R_12_10cp the gap is not the model.** SLAM Dunk's baseline already scores 81.09 against visloc-rs's
  63.88, and its full model adds 1.0. What makes the difference there, and on sequence_1_20 (95.12 against 86.70), is in the baseline
  pipeline: front end, keyframes, robust passes or solver. On sequence_3_17 the same baseline is 28 points
  below visloc-rs's.

## All of the paper's terms at once

The pass SLAM Dunk's full config corresponds to, as far as visloc-rs has it: the fisheye model, the camera
path with the paper's priors (cam0 as the reference), the gyro and accelerometer intrinsics and the clock
offsets, all in one pass, paired against the default chain from the same VIO, online windows and first matching
on the thirteen (`EXPERIMENT_ARGS="--ba-fisheye-model --ba-stereo-reference 0 --ba-stereo-paper-priors
--ba-est-gyro-m --ba-est-accel-m --ba-est-time-offsets"`, run 45573280189).

| Recording | Default CP | All terms | Change | Recall at 1 m, default → all (%) |
|---|---:|---:|---:|---:|
| R_11_5cp | 87.90 | 85.77 | −2.13 | 100.0 → 100.0 |
| R_12_10cp | 63.74 | 57.96 | −5.78 | 60.0 → 40.0 |
| R_13_15cp | 78.10 | 78.30 | +0.20 | 93.3 → 93.3 |
| sequence_1_19 | 85.34 | 83.36 | −1.99 | 100.0 → 100.0 |
| sequence_1_20 | 86.70 | 82.78 | −3.92 | 100.0 → 100.0 |
| sequence_2_11 | 88.47 | 79.03 | −9.44 | 100.0 → 100.0 |
| sequence_2_12 | 51.16 | 60.45 | +9.29 | 25.0 → 60.0 |
| sequence_3_17 | 80.33 | 75.21 | −5.12 | 100.0 → 70.4 |
| sequence_3_18 | 57.78 | 54.65 | −3.13 | 41.7 → 33.3 |
| sequence_4_10 | 68.23 | 67.81 | −0.42 | 81.2 → 75.0 |
| sequence_4_11 | 76.05 | 75.39 | −0.66 | 100.0 → 100.0 |
| sequence_5_11 | 65.03 | 71.32 | +6.29 | 71.4 → 78.6 |
| sequence_5_12 | 6.34 | 6.30 | −0.05 | 0.0 → 0.0 |

Mean of the thirteen 68.86 → 67.56 (**−1.30**), 3 of 13 gain: not kept. On sequence_3_17 the pass scores 75.21,
below the default chain's 80.33 and far from SLAM Dunk's full 87.64; it sits with SLAM Dunk's no_clock (71.60)
and visloc-rs's earlier pass without the clock offsets (73.93), so the clock offsets add little here. The terms
that lift SLAM Dunk's own baseline by 35 points on sequence_3_17 do not lift visloc-rs's; together with
R_12_10cp and sequence_1_20, where SLAM Dunk's baseline alone is 17 and 8 points ahead, the gap points at the
pipeline the terms sit on (front end, keyframes, robust passes) rather than at the calibration model.

## The paper's front end in the pass: RootSIFT and the keyframe spacing

With the calibration terms ruled out as the gap, the pass by day gets keypoints of its own
(`--ba-experiment-pass-features`, lamaria_ba_chain.py `--pass-features sift|rootsift` with `--pass-feature-args`
and `--pass-match-args`), paired against the default chain from the same VIO and online windows. The online
windows keep the default SIFT; in low light (track 4) the pass keeps the low-light keypoints, so there the
experiment is the default chain (change 0; the first runs failed there on a rematch bug, fixed in 21bdd1c).

**900 RootSIFT keypoints per image, ratio test 0.78** (the paper's count and ratio; RootSIFT's distance limit
225; run 45714906572):

| Recording | Default CP | RootSIFT 900 | Change | Recall at 1 m, default → RootSIFT (%) |
|---|---:|---:|---:|---:|
| R_11_5cp | 87.90 | 85.10 | −2.80 | 100.0 → 100.0 |
| R_12_10cp | 63.74 | 74.72 | +10.98 | 60.0 → 80.0 |
| R_13_15cp | 78.14 | 78.97 | +0.83 | 93.3 → 100.0 |
| sequence_1_19 | 85.34 | 91.30 | +5.96 | 100.0 → 100.0 |
| sequence_1_20 | 86.69 | 87.55 | +0.85 | 100.0 → 100.0 |
| sequence_2_11 | 88.47 | 83.03 | −5.45 | 100.0 → 100.0 |
| sequence_2_12 | 51.02 | 46.11 | −4.91 | 25.0 → 30.0 |
| sequence_3_17 | 80.33 | 77.71 | −2.62 | 100.0 → 92.6 |
| sequence_3_18 | 57.76 | 63.76 | +6.00 | 41.7 → 45.8 |
| sequence_4_10, sequence_4_11 | low light: the default chain | | 0 | |
| sequence_5_11 | 68.38 | 73.81 | +5.43 | 71.4 → 92.9 |
| sequence_5_12 | 6.53 | 5.70 | −0.83 | 0.0 → 0.0 |

Mean change over the thirteen **+1.03**, 6 of the 11 it applies to gain. The largest gain is on R_12_10cp, where
SLAM Dunk's baseline is 17 points ahead; this is the first change measured here that closes part of the
baseline-pipeline gap. One run is not yet a decision (sequence_2_12 and the Moving recordings move by several
points between runs): a repeat (run 45751972024) and 1,500 RootSIFT keypoints at ratio 0.78 (run 45751990291)
follow.

**Keyframes 1 s apart** (`--kf-step 20`, every second keyframe of the online windows; the paper spaces them
max(0.6 s, T/1500); run 45714886254): R_11_5cp −11.46, R_12_10cp −5.55, R_13_15cp −1.32, sequence_1_19 −8.59,
sequence_1_20 −13.31, sequence_2_11 −16.37, sequence_2_12 −10.40, sequence_3_17 −12.49, sequence_3_18 +2.92,
sequence_5_11 −11.67, sequence_5_12 −0.10; mean over the thirteen −6.80. The 0.5 s keyframes stay.
