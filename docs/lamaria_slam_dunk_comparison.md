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
