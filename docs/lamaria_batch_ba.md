# LaMAria: batch visual-inertial bundle adjustment

The refined Basalt VIO still drifts slowly in heading and scale
([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md)). Its sliding
window only sees short KLT tracks: on sequence_1_19, half of the tracks
last one frame and 90 % at most six, and an established track dies with a
probability of 10 % per frame. Points that stay in view for tens of seconds
are therefore re-detected as new landmarks many times, and nothing ties
keyframes more than a few seconds apart together.

[`scripts/lamaria_batch_ba.py`](../scripts/lamaria_batch_ba.py) refines the
VIO trajectory afterwards with one batch optimization over the whole
sequence. It uses no ground truth.

The sections below follow the development in order. The current pipeline and
its results on ten training sequences are in [Two sequences per
track](#two-sequences-per-track) (mean 74.64 over the five sequences it was
developed on, 54.34 over all ten), the commands in [Usage](#usage). Run on
windows of the latest 240 s as the data comes in, then once more over the
whole recording, it reaches 59.70 (see [Online, then one batch
pass](#online-then-one-batch-pass)). From Basalt's default VIO, causal and
within 4 cores, with a wider epipolar band for the matching, it averages
60.54, and 61.19 with the batch pass after the recording (see [Matching
guided by a drifting VIO](#matching-guided-by-a-drifting-vio)). With the gyro
bias random walk loosened in the first minutes of a recording (day tracks),
it averages 61.40 online and **64.35** with the pass (its random walk ten
times looser), above GroundedSLAM's 62.3 on the leaderboard (see [The first
minutes, online](#the-first-minutes-online)).

## Method

1. **Keyframes.** Every 10th frame of the VIO trajectory (2 Hz).
2. **Features.** SIFT on both cameras (OpenCV, 1,500 per image, contrast
   threshold 0.01).
3. **Matching.** Each keyframe is matched against the keyframes 1, 2, 3, 4,
   6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192 and 256 steps later (0.5 to
   128 s), within and across the two cameras, if their optical axes differ
   by less than 75° and the VIO puts them less than 200 m apart (see
   [Matches further apart](#matches-further-apart)). A
   candidate must lie within an epipolar band around the VIO's relative pose:
   2 px plus 0.1 px per keyframe step, at most 6 px. The descriptor ratio
   test (0.8) and a mutual check run among those candidates.
4. **Tracks.** Connected components of the match graph. Components with two
   features in one image are dropped. Each track is triangulated with the
   VIO poses (midpoint). Observations more than 4 px off are dropped, and the
   triangulation angle must reach 1°.
5. **Solve.** Levenberg-Marquardt over:
   - every keyframe's pose, velocity and IMU biases, with GTSAM's
     preintegrated IMU factor between consecutive keyframes (the
     calibration's noise densities);
   - every landmark, with Huber reprojection terms (σ 1 px, threshold 2 px);
   - each camera's IMU-camera rotation, one per eighth of the sequence (see
     [During a recording](#during-a-recording)).

   The landmarks are eliminated by the Schur complement. A landmark keeps
   its observations within 64 keyframes of its first one (`--band`), so the
   reduced system is banded and is factored by LAPACK's banded Cholesky.
   The second solve of the pipeline keeps all of them (see
   [Revisits](#revisits)).
   The camera rotations form a dense border. A prior on the first pose fixes
   the gauge.

   Each iteration first drops the observations more than 10 px off, then
   the landmarks left with a single observation. A landmark's damping is its
   Hessian diagonal floored at 1 % of its largest entry. Without both, a
   landmark seen with little parallax could take a step of hundreds of
   metres, and every damped step of an iteration raised the cost.

   From the 8th iteration on the threshold tightens to 3 px. By then the
   solution has settled and the median error is about 0.3 px, so an
   observation still 3 px off is almost always a wrong match. Huber alone
   lets each such observation pull like a 2 px residual; the 1–2 % of them
   bend the trajectory against the IMU (see [Results](#results)).

   A solve runs at most 60 iterations (`--iterations`) and stops earlier
   once an iteration lowers the cost by less than 1e-7 of it (`--rel-tol`);
   see [Iterations](#iterations).
6. **Output.** The VIO's per-frame trajectory, each frame moved by the
   correction of the surrounding keyframes (interpolated).
7. **Revisits (optional).** `loops` matches places visited again, and
   `loop-correct` turns those matches into relative poses for a pose graph
   over the solved keyframes. A second `solve` then starts from the
   corrected trajectory and adds the revisit matches (see
   [Revisits](#revisits)).

## The IMU-camera rotation of each camera

The batch problem observes each camera's rotation relative to the IMU.

### Device 1WM09370081264

On sequence_1_19 only cam0 is off:

| Estimate | cam0 (in the IMU frame) | cam1 |
|---|---|---:|
| Batch BA, first 75 s | 0.60° | 0.03° |
| Batch BA, whole sequence | [0.49, −0.26, 0.17]° (0.58°) | 0.04° |
| Gyroscope vs pseudo-GT, cam0 ([device table](lamaria_drift_analysis.md#per-device-refinement)) | [0.57, −0.29, 0.37]° (0.74°) | — |

The device table measured the cam0 rotation against the pseudo-GT, which is
given in the cam0 frame. It then rotated the whole rig, which also turned
cam1 by 0.74°. That explains why the table's rotation "overshoots" on this
device.

With cam0's rotation set to the BA estimate, the VIO alone gives these
results on sequence_1_19:

| Metric | Factory rotation | Corrected cam0 |
|---|---:|---:|
| Pose recall at 1 m | 76.3 % | 85.4 % |
| Sim(3) RMSE against the pseudo-GT | 0.88 m | 0.78 m |
| CP Score | 71.96 | 71.25 (14 control points) |

### Device 1WM093701G1276

On sequence_2_11, three 75 s windows each estimate about
[0.14, 0.13, −0.03]° (0.19°) in the IMU frame, for both cameras. The two
cameras agree to within 0.05–0.08°. The device table applies 0.36°.

### During a recording

The rotations also change while the glasses are worn. `--ext-segments 8`
(the default) estimates each camera's rotation separately in eight equal
time segments of the keyframes (2–4 min each), with a prior of 0.05°
(`--ext-seg-sigma`) on the change between consecutive segments. On
1WM093701G1276 both cameras then turn together relative to the IMU, by
about 0.1° over a recording. On 1WM09370081264 the segments stay within
0.02–0.05° of each other:

| Sequence | Device | Range of the common rotation | Correlation of the two cameras |
|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | 0.05° (y) | 0.65 |
| sequence_1_20 | 1WM09370081264 | 0.02° (z) | 0.34 |
| sequence_2_11 | 1WM093701G1276 | 0.11° (y) | 0.94 |
| sequence_3_17 | 1WM093701G1276 | 0.09° (y) | 0.97 |
| sequence_4_10 | 1WM093701G1276 | 0.09° (x) | 0.97 |
| sequence_5_11 | 1WM093701G1276 | 0.10° (y) | 0.90 |

The common rotation is the mean of the two cameras' segment estimates, in
the IMU frame. The range is that of its component that varies most, and the
correlation is that of the two cameras' estimates in that component.

The number of segments matters little between four and eight. With the
prior scaled to the same smoothness per unit of time (0.07° for four
segments, 0.035° for sixteen), the first solve scores:

| Sequence | One rotation | 4 segments | 8 segments | 16 segments |
|---|---:|---:|---:|---:|
| sequence_2_11 | 68.88 | **71.84** | 71.18 | 69.97 |
| sequence_3_17 | 50.32 | 59.57 | **60.25** | 59.23 |
| sequence_5_11 | **59.81** | 56.83 | 56.22 | |

## Results

Official evaluator, one training sequence per track, plus sequence_1_20, a
second Short sequence of the device of sequence_1_19. The VIO and mapper
columns are the per-device-refined runs of
[`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper);
on sequence_1_20 the VIO has only the per-device time alignment, and there is
no mapper run. The batch BA starts from that VIO trajectory and uses the
factory camera rig. On sequence_4_10 the VIO is the low-light configuration
(0.5 px KLT threshold) and the BA's SIFT runs on CLAHE-equalized images. The
first version of the solver had neither the outlier rejection nor the
landmark damping floor; the 10 px column has both but keeps 10 px to the end;
the 3 px column also tightens it to 3 px from the 8th iteration on (see
[Method](#method)). The last column also estimates the camera
rotations in eight time segments (see [During a
recording](#during-a-recording)).

| Sequence | Track | Device | VIO | VIO + mapper | Batch BA, first version | Batch BA, 10 px | Batch BA, 3 px | Batch BA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | Short | 1WM09370081264 | 71.96 | 68.54 | 83.96 | 83.02 | 80.63 | **86.53** |
| sequence_2_11 | Medium | 1WM093701G1276 | 49.68 | 55.51 | 52.29 | 61.03 | 68.88 | **71.18** |
| sequence_3_17 | Long | 1WM093701G1276 | 50.06 | 46.68 | 49.59 | 50.14 | 50.32 | **60.25** |
| sequence_4_10 | Low light | 1WM093701G1276 | 39.00 | 36.65 | 44.28 | 45.79 | 47.43 | **47.65** |
| sequence_5_11 | Moving platform | 1WM093701G1276 | 43.14 | 49.84 | 55.70 | 59.06 | **59.81** | 56.22 |
| **Mean of the five tracks** | | | 50.77 | 51.44 | 57.16 | 59.81 | 61.41 | **64.36** |
| sequence_1_20 | Short | 1WM09370081264 | 38.10 | | 43.48 | 43.39 | 44.43 (**76.13** with `--modes yaw`, see [below](#slow-heading-drift)) | 44.41 |

The 3 px stage drops 1–2 % of the observations (11,400 of 874,000 on
sequence_2_11). The IMU then fits better: on sequence_2_11 the whitened
residual per factor falls from 1.61 to 0.91 for the position and from 6.60
to 4.94 for the velocity, and the fitted scale drift from +2.8 % to +1.8 %.
The trajectory improves on every sequence with pseudo-GT (table below); the
CP Score of sequence_1_19 falls by 2.4 even so, its CP errors growing from
0.33 to 0.37 m at the median, where the score falls steeply.

The eight rotation segments act on the slow drift, not on the local
accuracy. The RMSE of 60 s segments stays the same (table below), while the
Sim(3) RMSE falls on four of the five sequences with pseudo-GT: on
sequence_3_17 from 1.97 to 1.31 m, and its pose recall at 1 m rises from
17.0 % to 56.0 %. The fitted scale drift falls from +2.3 % to +1.4 % on
sequence_3_17 and from +1.9 % to +1.4 % on sequence_2_11 (drift table
below). sequence_5_11 loses 3.6 points: most of its CP errors grow by
0.1–0.6 m (1.09 → 1.23 m at the median, in a 2D similarity fit).

The same runs against the pseudo-GT:

| Sequence | Metric | VIO | VIO + mapper | Batch BA, 10 px | Batch BA, 3 px | Batch BA |
|---|---|---:|---:|---:|---:|---:|
| sequence_1_19 | Pose recall at 1 m | 76.3 % | 60.6 % | 91.5 % | 91.7 % | **94.0 %** |
| sequence_1_19 | Sim(3) RMSE | 0.88 m | 0.96 m | 0.55 m | 0.55 m | **0.44 m** |
| sequence_2_11 | Pose recall at 1 m | 30.9 % | 39.4 % | 51.3 % | 70.2 % | **76.8 %** |
| sequence_2_11 | Sim(3) RMSE | 2.31 m | 1.92 m | 1.51 m | 1.01 m | **0.87 m** |
| sequence_3_17 | Pose recall at 1 m | 24.7 % | 21.7 % | 16.3 % | 17.0 % | **56.0 %** |
| sequence_3_17 | Sim(3) RMSE | 2.07 m | 2.48 m | 2.02 m | 1.97 m | **1.31 m** |
| sequence_4_10 | Pose recall at 1 m | 14.1 % | | 19.3 % | **21.5 %** | 20.9 % |
| sequence_4_10 | Sim(3) RMSE | 3.11 m | | 2.36 m | 2.11 m | **2.05 m** |
| the four above | RMSE of 60 s segments, each Sim(3)-aligned | 0.151 / 0.264 / 0.182 / 0.381 m | | 0.127 / 0.206 / 0.147 / 0.336 m | 0.125 / 0.169 / 0.141 / 0.315 m | 0.125 / 0.169 / 0.141 / 0.317 m |
| sequence_1_20 | Pose recall at 1 m | 3.8 % | | 11.8 % | 14.4 % (**100 %** with `--modes yaw`) | 16.4 % |
| sequence_1_20 | Sim(3) RMSE | 3.03 m | | 2.41 m | **2.31 m** | 2.35 m |

On sequence_5_11 (no pseudo-GT) the BA brings AA2976 from 0.94 m (VIO) to
0.07 m and the fitted scale drift (without PB2) from −5.4 % to −2.5 %. PB2,
the only CP before the funicular ride, stays 19 m off.

The batch BA is the most accurate locally on every sequence with pseudo-GT.
The slow drift that remains, fitted to the CP residuals as a heading and a
scale drift linear in time:

| Sequence | Heading drift, 10 px | Scale drift, 10 px | Heading drift, 3 px | Scale drift, 3 px | Heading drift | Scale drift |
|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | −0.14° | −0.6 % | −0.30° | −0.6 % | −0.19° | −0.5 % |
| sequence_2_11 | −0.34° | +2.8 % | −0.47° | +1.9 % | −0.50° | +1.4 % |
| sequence_3_17 | +0.82° | +2.3 % | +0.81° | +2.3 % | +0.48° | +1.4 % |
| sequence_4_10 | −0.58° | +4.0 % | −0.92° | +3.6 % | −0.86° | +3.5 % |
| sequence_5_11 (without PB2) | −0.35° | −2.5 % | −0.22° | −2.3 % | +0.25° | −2.4 % |

The sequences of 1WM093701G1276 keep 1.4–3.5 % of scale drift (2–4 % with
one rotation per camera), and their local scale (Sim(3) fits in 60 s
windows) varies by ±2–4 %, against ±1 % on sequence_1_19. Their IMU factors
also fit worse: the whitened position residual per factor is 1.6–1.8,
against 0.4–0.5 on 1WM09370081264 (10 px solve). These do not remove it
(sequence_2_11, 10 px, one rotation per camera, CP Score 61.03):

| Change | CP Score | Note |
|---|---:|---|
| Accelerometer y axis ×1.03 / ×0.97 | 60.40 / 56.48 | The IMU cost nearly doubles (20,600 → 39,100 / 36,700): the BA pins each axis' scale to well under 1 % |
| Accelerometer delayed by −2 / +2 ms | 61.14 / 61.14 | Final cost +1,100 / −90 |
| Accelerometer bias random walk ×10 | 60.49 | |
| IMU noise densities ×2 | 58.02 | |
| Keyframes at 4 Hz (every 5th frame), matches and band up to 128 keyframes | 60.05 | Scale drift +2.8 → +1.8 %, heading drift −0.35 → −0.02°, IMU position residual 1.61 → 0.75; the CPs of the first half gain what those of the second half lose |
| Camera translations estimated too (prior 1 cm) | 60.21 | They move by 2–4 mm; the scale stays (1.026 against the pseudo-GT) |
| Vertical Earth rotation in the IMU factors (`--omega-coriolis`) | 60.99 | The gyro bias absorbed it already |
| Cross-camera matches left out | 64.89 | With the first rotation prior loose as well, −2.2 on sequence_1_19, −2.0 on sequence_4_10 and −7.9 on sequence_5_11 against the 10 px solve; the stereo constraints carry sequence_5_11's funicular ride |

A fit of the accelerometer against the pseudo-GT's second differences had
suggested a y scale of 0.97 on this device; the BA rules that out. The
camera exposure does not explain the local scale either: it correlates
weakly with it on sequence_2_11 (−0.33), and on sequence_4_10 the exposure
stays at 14 ms while the local scale varies by ±7 %.

The IMU is not what shrinks the trajectory. With the BA's rotations fixed and
its positions replaced by the pseudo-GT's (aligned rigidly per 60 s window),
the scale the preintegrated IMU prefers is 1.006 ± 0.004 on sequence_2_11
(window to window ±1.7 %). The BA instead keeps the global scale of its
starting point, like its heading drift: the VIO is 2.4 % short against the
pseudo-GT, the BA 2.7 %, and a start stretched by 2.5 % ends 0.9 % short at a
slightly lower cost (229,625 against 229,821, with the first rotation prior
loose, see [below](#the-first-keyframes-rotation-prior)). The scale is a weak
direction of the objective; within a 60 s window the IMU constrains it to
only about ±1.7 %.

### Where the scale drift comes from

The scale drift of sequence_2_11 is the BA's own. In Sim(3) fits of 60 s
windows against the pseudo-GT, the VIO's local scale has no trend, the BA's
grows by 2.5 % per 1,000 s, and a solve started at the pseudo-GT drifts as
much: the objective prefers the drift. Changing one ingredient at a time
(eight segments unless noted):

| Run on sequence_2_11 | CP Score | Local scale, trend per 1,000 s |
|---|---:|---:|
| VIO | 49.68 | −0.3 % |
| Batch BA | 71.18 | +2.5 % |
| Started at the pseudo-GT (one rotation per camera, 10 px; diagnosis only) | 54.46 | +2.5 % |
| cam0 only | 64.75 | +2.9 % |
| cam1 only | 60.55 | +3.1 % |
| Cross-camera matches left out | **72.70** | +1.7 % |
| Matches at most 16 keyframes (8 s) apart | 68.19 | +1.7 % |
| IMU noise densities ×0.5 | 66.66 | +3.3 % |
| Intrinsics (fx, fy, cx, cy, k1, k2) estimated per segment too | 66.60 | **+1.2 %** |
| Intrinsics estimated once per camera (one rotation per camera) | 62.19 | +2.3 % |

Each camera alone drifts, so the rig does not cause it. Leaving out the
cross-camera matches or the long matches takes a third of it away, but
neither removes it. The solve leaves a small radial residual on
1WM093701G1276: observations lie 0.02–0.04 px further out than their
projections at 150–300 px from the principal point, on both cameras, against
at most 0.02 px inwards on 1WM09370081264. Estimating the intrinsics finds a
focal length 0.1–0.3 % longer and k1 ≈ +0.002 on both cameras of
1WM093701G1276 (−0.1 % and 0 on sequence_1_19), but that reduces the drift
only when the intrinsics may change per segment, and it costs points either
way.

The radial residual grows with the SIFT keypoint size, on both devices. At
200–250 px from the principal point (about 45° off axis, where the pinhole
image stretches fastest radially), by keypoint octave:

| Sequence | Octave −1 (upsampled) | Octave 0 | Octave 1 | Octave 2 |
|---|---:|---:|---:|---:|
| sequence_2_11 | +0.024 px | +0.038 px | +0.058 px | +0.075 px |
| sequence_1_19 | −0.007 px | −0.001 px | +0.012 px | +0.051 px |

Only the offset common to all octaves differs between the devices. Moving
every keypoint by its octave's and radius's mean residual (a table from the
first solve) leaves the local scale trend of
sequence_2_11 at +2.55 % per 1,000 s (+2.53 % with three times the
correction; CP Score 71.75 and 68.01), and keeping only matches between
keypoints of similar size raises it (+3.22 % within a size ratio of 1.25,
+2.82 % within 1.5). Neither the detector nor the residual distortion drives
the drift.

The ratio of the image's angular scale to the gyroscope's does. Changing the
focal lengths linearly over the recording (the mean unchanged; first solve,
eight segments) moves the trend in proportion:

| Focal length change from the first to the last segment | CP Score | Pose recall at 1 m | Local scale trend per 1,000 s |
|---|---:|---:|---:|
| −0.4 % | 63.42 | 55.8 % | +3.80 % |
| 0 (the calibration) | 71.18 | 76.8 % | +2.48 % |
| +0.4 % | **74.93** | **84.5 %** | +1.32 % |
| +1.2 % | 70.59 | 81.5 % | (fitted scale drift −1.2 %) |

Without outlier rejection, so that every run keeps the same observations,
the objective itself is lowest near +0.4 %: final costs 256,222 (−0.4 %),
254,336 (0), 253,950 (+0.4 %) and 255,206 (+0.8 %). Scaling the gyroscope
instead changes the scale drift alike but turns the heading: a gyroscope
scale from 1.003 to 0.997 over the recording raises the trend to +3.78 %
(CP Score 65.82), the opposite to +1.73 % (60.37, heading drift +0.21°). A
constant gyroscope scale of 1.003 gives 72.79 (pose recall 83.2 %) at a much
higher cost (190,737 against 181,127; the rotation residual per factor goes
from 7.6 to 13.4), so the gyroscope's own scale is right.

The same +0.4 % does not suit every sequence (first solve):

| Sequence | Device | Calibration | Focal length +0.4 % over the recording |
|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | **86.53** | 81.24 |
| sequence_2_11 | 1WM093701G1276 | 71.18 | **74.93** |
| sequence_4_10 | 1WM093701G1276 | 47.65 | **50.97** |
| sequence_5_11 | 1WM093701G1276 | **56.22** | 51.27 |
| sequence_1_20 | 1WM09370081264 | 44.41 | **45.29** |

`--est-focal` estimates the focal lengths instead: one scale per camera and
time segment, next to its rotation, with priors on its change between
segments (0.05 %, `--focal-seg-sigma`) and on its value (1 %). The fit
improves on both sides: on sequence_2_11 the reprojection cost falls from
164,733 to 160,559 and the IMU cost from 16,386 to 15,489, and the IMU
position, velocity and rotation residuals per factor all shrink. The
estimates are consistent by device and conditions, which suggests the
temperature:

| Sequence | Device | Conditions | Focal length against the calibration, cam0 / cam1 (mean of the segments) | First and last segment, cam0 |
|---|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | day | −0.11 % / −0.08 % | −0.15 % .. −0.15 % |
| sequence_1_20 | 1WM09370081264 | day | −0.11 % / −0.06 % | −0.13 % .. −0.10 % |
| sequence_2_11 | 1WM093701G1276 | day | +0.34 % / +0.26 % | +0.18 % .. +0.36 % |
| sequence_3_17 | 1WM093701G1276 | day | +0.41 % / +0.37 % | +0.32 % .. +0.45 % |
| sequence_4_10 | 1WM093701G1276 | night | −0.04 % / −0.04 % | −0.08 % .. +0.01 % |
| sequence_5_11 | 1WM093701G1276 | day, funicular | +0.57 % / +0.53 % | +0.41 % .. +0.59 % |

The pipeline with the revisits (first solve, `loop-correct`, second solve,
each with `--est-focal`):

| Sequence | First solve | `--est-focal` | Pipeline | `--est-focal` | Pose recall at 1 m, pipeline / `--est-focal` |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | **86.53** | 83.28 | 87.22 | | 96.5 % / |
| sequence_2_11 | **71.18** | 68.74 | 77.31 | **77.72** | 75.8 % / **94.1 %** |
| sequence_3_17 | 60.25 | **65.96** | 60.00 | **66.16** | 55.5 % / **63.4 %** |
| sequence_4_10 | 47.65 | **48.86** | 49.52 | **50.75** | 24.1 % / **25.8 %** |
| sequence_5_11 | **56.22** | 46.96 | **57.15** | 48.02 | |
| sequence_1_20 | 44.41 | **46.18** | 43.41 | **44.79** | 12.7 % / **14.1 %** |

On the moving platform the estimate goes wrong: there the IMU feels the
funicular, which the scene inside the car does not show, and the focal
lengths absorb the conflict (the fitted scale drift grows from −6.5 % to
−8.6 %). Pinning the mean of each camera's scales to the calibration
(`--focal-mean-sigma 0.00001`) does not save it (48.14).

Elsewhere the mean is what helps: with it pinned, sequence_3_17's first solve
scores 56.49, against 65.96 with the focal lengths estimated and 60.25 with
the calibration's.

### Keypoint size

A SIFT keypoint detected at a coarser scale is located less precisely. The
residuals of the first solve (robust spread, 1.4826 × MAD of the x residual)
by keypoint octave:

| Sequence | Octave −1 (2.7 px) | Octave 0 (4.8 px) | Octave 1 (9.3 px) | Octave 2 (18 px) | Octave 3 (37 px) |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | 0.19 px | 0.23 px | 0.32 px | 0.49 px | 0.59 px |
| sequence_2_11 | 0.24 px | 0.30 px | 0.40 px | 0.51 px | 0.59 px |
| sequence_4_10 | 0.42 px | 0.43 px | 0.51 px | 0.60 px | 0.66 px |

About ×1.27 per octave, that is σ ∝ size^0.35, except at the finest scale in
low light, where noise dominates. `--sigma-size 2.7 --sigma-size-pow 0.35`
gives each observation σ = 1 px × max(1, size / 2.7 px)^0.35 (first solve):

| Sequence | σ = 1 px | σ by keypoint size |
|---|---:|---:|
| sequence_1_19 | 86.53 | **87.46** |
| sequence_2_11 | 71.18 | **72.57** |
| sequence_3_17 | **60.25** | 60.09 |
| sequence_4_10 | **47.65** | 46.65 |
| sequence_5_11 | 56.22 | **56.54** |
| **Mean of the five tracks** | 64.36 | **64.66** |
| sequence_1_20 | 44.41 | **44.94** |

### Low light: features after denoising

On sequence_4_10 the BA keeps 191 observations per keyframe, against 360–500
on the other sequences, and its 60 s segments are twice as far off (0.32 m
RMSE). The images are dark (median grey level 10–30) and noisy at gain 15,
and the CLAHE equalization amplifies the noise: half the matches between
consecutive keyframes of the other sequences survive (137 instead of 259–362
per image pair), and 72 % of the keypoints come from the upsampled octave,
where the residuals are no smaller than one octave up (0.42 px MAD on
sequence_4_10, against 0.24 px on sequence_2_11).

Non-local means denoising (OpenCV, h = 7) before the equalization
(`features --denoise 7`) leaves fewer keypoints, but the ones that match agree
with the IMU. Each variant below runs features, matching and a solve on one
200 s window of keyframes; the RMSE is that of a Sim(3) fit to the pseudo-GT
over the window, and the IMU column the whitened position residual per
factor:

| Variant | Keypoints per image | Observations | RMSE, keyframes 1000–1400 | IMU | RMSE, 200–600 | RMSE, 2200–2600 |
|---|---:|---:|---:|---:|---:|---:|
| CLAHE (the default for low light) | 1,500 | 87,593 | 0.890 m | 2.09 | 0.706 m | **0.440 m** |
| No equalization | 1,403 | 81,460 | 0.809 m | 1.80 | | |
| CLAHE, clip limit 1.5 instead of 3 | 1,500 | 85,757 | 0.904 m | 2.27 | | |
| Gaussian blur (σ 1 px) before CLAHE | 1,500 | 95,751 | 0.889 m | 2.20 | | |
| 3,000 keypoints | 3,000 | 127,258 | 0.864 m | 4.58 | | |
| Non-local means, h = 4 | 1,083 | 82,983 | 0.476 m | 1.16 | | |
| Non-local means, h = 7 | 679 | 62,415 | **0.399 m** | 0.69 | **0.462 m** | 0.471 m |
| Non-local means, h = 10 | 508 | 50,101 | 0.428 m | 0.43 | | |
| Non-local means, h = 7, contrast threshold 0.005 | 1,326 | 74,238 | 0.900 m | 1.96 | | |

The weak keypoints are the problem: with the denoising but a lower contrast
threshold, the window is as far off as without it.

On the whole sequence the denoised features (556 per image, 2.6 million
matches against 4.1 million) take the fitted scale drift from +3.5 % to
+0.8 % and most CP errors under a metre:

| sequence_4_10 | First solve | + `loop-correct` | + second solve | Pose recall at 1 m |
|---|---:|---:|---:|---:|
| CLAHE | 47.65 | 47.54 | 49.52 | 24.1 % |
| Non-local means (h = 7), then CLAHE | 63.31 | 62.52 | 63.73 | **68.0 %** |
| The same, with `--est-focal` | **63.64** | **63.26** | **64.30** | 66.9 % |

What is left comes from the observations, not from the optimizer: the same
first solve started at the pseudo-GT moves away from it and ends at 60.10
(pose recall 45.9 %), below the solve started at the VIO. The local error
of 60 s segments is still about twice that of the day sequences (0.15–0.43 m),
the heading wanders by ±1° over the recording, and 2D errors of 1.5–2 m
remain at 60–180 s, 840 s and 1,020–1,080 s.

Other preprocessing and detector settings on top of the denoising, on four
200 s windows (Sim(3) RMSE against the pseudo-GT; one IMU-camera rotation
per camera):

| Variant | Keyframes 200–600 | 1000–1400 | 1600–2000 | 2200–2600 |
|---|---:|---:|---:|---:|
| Non-local means (h = 7), then CLAHE | 0.462 m | 0.399 m | 0.619 m | **0.471 m** |
| The same, SIFT contrast threshold 0.02 instead of 0.01 | **0.429 m** | **0.340 m** | **0.457 m** | 0.523 m |
| The same, contrast threshold 0.03 | 0.449 m | | | |
| Non-local means h = 10, contrast threshold 0.02 | 0.518 m | | | |
| CLAHE clip limit 2 instead of 3 | 0.484 m | 0.382 m | 0.578 m | 0.562 m |
| Gamma 0.5 instead of CLAHE | 0.516 m | 0.344 m | | 0.478 m |
| Keyframes at 4 Hz (offsets and band doubled) | | 0.344 m | | |
| Keyframes at 4 Hz, contrast threshold 0.02 | | 0.342 m | | |

No equalization, CLAHE tiles of 4 × 4, dropping the keypoints on dark
ground (mean grey level under 10) or larger than 20 px, and RootSIFT
descriptors (0.809 m) do not beat the denoised baseline on the 1000–1400
window either. The contrast threshold of 0.02 wins three windows of four, but
on the whole sequence (402 keypoints per image, 2.1 million matches) it ends
where the baseline does: 62.97 after the first solve and **64.35** after the
second (pose recall 62.1 %). The 200 s windows measure the local shape, while
the CP Score of this one-way route follows the drift over minutes.

The VIO's KLT tracks cannot fill in either. `basalt_euroc_vio_demo
--dump-tracks` writes the tracked points of every frame. A track lives a
median of one frame, 1.8 frames on average on sequence_4_11 at night and
2.5 by day (the first 2,000 frames of sequence_3_17); sampled at the 2 Hz
keyframes, 20–28 tracks per keyframe are seen twice or more, against 45–162
SIFT landmarks.

Learned features do not help on this window either. XFeat (1,500 keypoints
per image) and ALIKED-n16 (score threshold 0.2, 494 keypoints per image), on
the raw images and matched by the same VIO-guided ratio test, end at 0.833 m
and 0.836 m on the 1000–1400 window, against 0.399 m for denoised SIFT. They
locate their keypoints less precisely (median reprojection error 0.60 and
0.52 px, against 0.47 px), and the solve bends against the IMU (position
residual 4.4 and 1.6 per factor, against 0.69).

### The first minutes of a recording

The gyroscope disagrees with the solved rotations at the start of every
recording. The whitened rotation residual per IMU factor (median over 15 s,
eight-segment solves) falls to its usual 1–2 within about two minutes:

| Sequence | 0–15 s | 15–30 s | 30–45 s | 45–60 s | 60–75 s | 75–90 s | 90–105 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | 25.9 | 13.6 | 2.4 | 2.3 | 1.9 | 6.2 | 1.9 |
| sequence_2_11 | 96.8 | 10.9 | 9.3 | 7.7 | 4.6 | 3.9 | 3.2 |
| sequence_3_17 | 11.7 | 8.1 | 5.7 | 2.9 | 2.6 | 3.7 | 1.1 |
| sequence_4_10 | 305.6 | 16.0 | 16.3 | 9.9 | 10.6 | 4.6 | 3.6 |

On sequence_4_10 the residual of the first two minutes does not depend on
how far the head turns between keyframes (about 10 for turns of 2–5° and of
more than 20°), which points at a bias rather than a scale: the gyro bias
still settles while the device warms up, and the tight bias random walk
(1e-7) keeps it constant. The recordings carry no IMU temperature (the field
is NaN in the VRS records). Letting the bias move does not help in general
(first solve):

| Run | sequence_2_11 | sequence_4_10 |
|---|---:|---:|
| Default | **71.18** | 47.65 |
| Gyro bias random walk ×100 | | 49.38 |
| Gyro bias random walk ×1000 | 66.56 | 50.53 |
| ×1000 at the start, decaying with a time constant of 120 s | 67.51 | **50.72** |
| The same with 300 s | | 49.29 |
| Accelerometer bias random walk ×30 at the start, 120 s | 68.36 | |

sequence_4_11, the second low-light sequence, shows how much is at stake.
Its VIO drifts in heading (−1.5° fitted to the CPs), and the BA keeps most
of it (−1.3°, CP Score 55.30 after the first solve). With the gyro bias
random walk ×100, the gyro bias of its first keyframe triples (2.8 mrad/s on
the y axis, then 0.45 mrad/s after nine minutes), the heading drift falls to
−0.5° and the first solve scores **70.47** (pose recall 28 % → 60 %). The
other sequences of 1WM093701G1276 let their gyro bias move as much with the
looser random walk (by 1–5 mrad/s on the y axis), and lose; the stable
1WM09370081264 gains a little. First solve with `--est-focal --sigma-size 2.7
--sigma-size-pow 0.35` (sequence_5_11 without `--est-focal`):

| Gyro bias random walk | sequence_1_19 | sequence_1_20 | sequence_2_11 | sequence_3_17 | sequence_4_10 | sequence_4_11 | sequence_5_11 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ×1 (1e-7, the default) | 84.12 | 46.64 | **70.93** | 65.24 | 62.18 | 55.30 | **56.54** |
| ×10 | **86.36** | **47.22** | 68.27 | **66.12** | **62.49** | 54.67 | 53.91 |
| ×100 | 85.64 | 46.66 | 64.10 | 61.47 | 55.21 | **70.47** | 55.30 |

Neither the costs nor the IMU residuals tell the cases apart: the cost falls
by 900–1,050 with ×100 on both sequence_4_11 and sequence_2_11, and
sequence_4_11 has the smallest whitened gyroscope residual of all with the
default (1.7 per factor, against 3.5–8.1 elsewhere). The default stays.

### What the gain on sequence_1_19 comes from

First version of the solver:

| Run | CP Score | Pose recall at 1 m |
|---|---:|---:|
| VIO | 71.96 | 76.3 % |
| VIO with the BA's cam0 rotation | 71.25 | 85.4 % |
| Batch BA, matches at most 4 keyframes (2 s) apart | 70.87 | 85.8 % |
| Batch BA, matches up to 64 keyframes (32 s) apart | **83.96** | 91.4 % |
| Batch BA, matches up to 128 keyframes (64 s) apart | 80.52 | **92.4 %** |

The camera rotation accounts for the first step in pose recall. The CP Score
gain comes from the long-range matches.

### Revisits

`loops` matches keyframes that are more than 64 keyframes apart but within
15 m (by the VIO) and look in similar directions. It matches them without a
pose prior, keeps a pair only if an essential matrix finds at least 25
inliers, and checks the rotation against the VIO (10°). On sequence_2_11 it
accepts 1,566 keyframe pairs:

- They lie 5.1 m apart in the pseudo-GT at the median and 19.3 m at most,
  so there is no false revisit.
- 750 of them are the revisit 120–240 s later between 570 s and 820 s.

Landmarks seen on both visits couple keyframes outside the band. The solver
then runs conjugate gradients on the full reduced system, preconditioned by
the banded factor, to a relative residual of 1e-4 (`--pcg-tol`, at most 200
iterations). Solving each step to 1e-9 instead took 5,816 iterations over the
30 steps of sequence_2_11's second solve, against 3,156, for the same result
(CP Score 77.31 and 77.30).

Added straight to the solve, the revisits do not close the loops. The
solved sequence_2_11 is about 1.3 m off between its two visits, and a revisit
observation that far off fails the 4 px triangulation check or the 10 px
rejection of the first iterations. What survives already agrees with the
trajectory: the solve keeps fewer observations than without the revisits
(845,000 against 874,000, as merged tracks conflict more often), stops at its
iteration limit and scores 53.15, against 68.88 without them. On
sequence_1_20 the 10 px solve fell from 43.39 to 38.97 the same way.

`loop-correct` closes the loops before a solve sees them:

1. It reads a finished solve (`solve --save-states`). For each revisit image
   pair (a, b), the features of a that are observations of a solved landmark
   give 3D points of a's visit. PnP with RANSAC (OpenCV, 3 px, at least 15
   inliers) places b's camera against them, and a's against b's visit. Each
   placement is a relative pose between keyframes a and b.
2. An edge that disagrees with the solve by more than 0.3 m plus 1 cm per
   second between the visits, or by more than 2°, is dropped: so much drift
   is not plausible.
3. The rest join a pose graph (GTSAM) whose odometry edges are the solve's
   consecutive relative poses (2e-4 rad and 2 cm per keyframe step); the loop
   edges are Cauchy-robust (0.005 rad, 0.1 m). The graph's keyframe
   corrections are spread over every frame, as `solve` spreads its own.
4. A second solve starts from that trajectory (as `--vio`) with `--loops`.
   The triangulation check and the outlier rejection now measure the revisit
   observations against the corrected trajectory.

On sequence_2_11 PnP places 1,160 edges, 1,050 within the gate. They disagree
with the solve by 1.29 m at the median, and the 941 edges of the revisit
120–240 s later agree with each other (1.67 m at the 90th percentile). The
graph moves the keyframes by 0.48 m at the median and 3.0 m at most, and the
revisited CP AA1812 goes from 1.8 to 1.1 m. With one rotation per camera
(the 3 px solve):

| Sequence | Revisit image pairs | Loop edges in the gate | Batch BA, 3 px | + `loop-correct` | + second solve |
|---|---:|---:|---:|---:|---:|
| sequence_1_19 | 966 | 220 | 80.63 | **81.79** | 81.76 |
| sequence_2_11 | 1,922 | 1,050 | 68.88 | 71.53 | **73.34** |
| sequence_3_17 | 415 | 45 | **50.32** | 50.15 | 50.06 |
| sequence_4_10 | 540 | 31 | 47.43 | 47.29 | **49.24** |
| sequence_5_11 | 529 | 84 | 59.81 | **61.31** | 61.20 |
| **Mean of the five tracks** | | | 61.41 | 62.41 | **63.12** |
| sequence_1_20 | 1,052 | 266 | **44.43** | 43.33 | 43.35 |

With the default eight rotation segments, in both solves:

| Sequence | Loop edges in the gate | Batch BA | + `loop-correct` | + second solve |
|---|---:|---:|---:|---:|
| sequence_1_19 | 219 | 86.53 | **87.35** | 87.22 |
| sequence_2_11 | 1,065 | 71.18 | 75.59 | **77.31** |
| sequence_3_17 | 46 | **60.25** | 60.03 | 60.00 |
| sequence_4_10 | 31 | 47.65 | 47.54 | **49.52** |
| sequence_5_11 | 84 | 56.22 | **57.45** | 57.15 |
| **Mean of the five tracks** | | 64.36 | 65.59 | **66.24** |
| sequence_1_20 | 270 | **44.41** | 43.44 | 43.41 |

The segmented solve of sequence_2_11 disagrees less with its revisits
(0.94 m at the median instead of 1.29 m), and the loops add 6.1 points
instead of 4.5.

On sequence_1_20 the revisits lie mostly under a minute apart and the solve
agrees with them already (0.10–0.18 m at the median). They cannot correct its
slow heading drift, and the 10 % of edges that disagree by more than a metre
cost a point.

### With the focal lengths, the keypoint sizes and denoising

The same pipeline (first solve, `loop-correct`, second solve), each solve
with the focal lengths estimated per segment (`--est-focal`, see [Where the
scale drift comes from](#where-the-scale-drift-comes-from)) and the pixel
noise by keypoint size (`--sigma-size 2.7 --sigma-size-pow 0.35`, see
[Keypoint size](#keypoint-size)); on sequence_4_10 the features are denoised
first ([Low light](#low-light-features-after-denoising)), and on the moving
platform the focal lengths stay at the calibration. Final CP Score (pose
recall at 1 m):

| Sequence | Track | Before (eight segments) | Focal lengths | Focal lengths and keypoint sizes |
|---|---|---:|---:|---:|
| sequence_1_19 | Short | **87.22** (96.5 %) | 84.47 (95.9 %) | 84.14 (**99.3 %**) |
| sequence_2_11 | Medium | 77.31 (75.8 %) | 77.72 (94.1 %) | **81.26** (**98.3 %**) |
| sequence_3_17 | Long | 60.00 (55.5 %) | **66.16** (**63.4 %**) | 65.38 (61.1 %) |
| sequence_4_10 | Low light | 49.52 (24.1 %) | **64.30** (66.9 %), denoised | 63.99 (59.6 %), denoised |
| sequence_5_11 | Moving platform | 57.15 | 57.15 | **57.70** (keypoint sizes only) |
| **Mean of the five tracks** | | 66.24 | 69.96 | **70.49** |
| sequence_1_20 | Short | 43.41 (12.7 %) | 44.79 (14.1 %) | **45.36** (**17.0 %**) |

The denoised features alone, without the focal lengths, give sequence_4_10
63.73 (68.0 %).

### Matches further apart

`match` used to stop at 64 keyframes (32 s) and 80 m. Matching also the
keyframes 96, 128, 192 and 256 steps later (up to 128 s), up to 200 m apart,
adds 23–24 % image pairs and 4–6 % matches (sequence_2_11: 81,565 pairs and
6.30 million matches, against 66,322 and 6.04 million) and takes up to
twice as long. The solve must take them in: its `--max-offset` was 64 as
well, which dropped them again and gave the same result as before.

The first solve still keeps a landmark's observations within 64 keyframes
of its first one; there the longer matches only merge tracks. The second
solve, which keeps every observation of a landmark, gains the most
(sequence_4_10: 62.18 → 63.44 after the first solve, 63.99 → 67.81 after the
second). The merged tracks are fewer and longer, 8–11 % fewer observations
pass the triangulation check against the starting trajectory, and the
solves take no longer. Final CP Score (pose recall at 1 m), the pipeline and
options of the table above:

| Sequence | Track | Up to 64 keyframes, 80 m | Up to 256 keyframes, 200 m |
|---|---|---:|---:|
| sequence_1_19 | Short | **84.14** (**99.3 %**) | 83.15 (93.4 %) |
| sequence_2_11 | Medium | 81.26 (98.3 %) | **83.83** (**99.3 %**) |
| sequence_3_17 | Long | 65.38 (61.1 %) | **65.41** (**61.7 %**) |
| sequence_4_10 | Low light | 63.99 (59.6 %) | **67.81** (**68.4 %**) |
| sequence_5_11 | Moving platform | **57.70** | 57.17 |
| **Mean of the five tracks** | | 70.49 | **71.47** |
| sequence_1_20 | Short | 45.36 (17.0 %) | **46.76** (**23.0 %**) |
| sequence_4_11 | Low light | 54.09 (27.2 %) | **56.49** (**37.6 %**) |

sequence_4_11 starts from the VIO of the test-submission pipeline
(`scripts/run_lamaria_test_submission.py --split training`), with denoised
features. These are now the defaults (`match --offsets`, `match
--max-dist`, `solve --max-offset`). Nearly all the matches beyond 64
keyframes are wrong, though; what they change is how far the solve stays
from its start (see [Matches far apart are wrong](#matches-far-apart-are-wrong)).

### Two sequences per track

The other five training sequences, through the test-submission pipeline's
VIO (`scripts/run_lamaria_test_submission.py --split training`, the
VIO-only build) and the same batch BA (the defaults above, `--est-focal
--sigma-size 2.7 --sigma-size-pow 0.35`, no `--est-focal` on the moving
platform). Final CP Score (pose recall at 1 m):

| Track | Sequence | VIO | Batch BA | Sequence | VIO | Batch BA | Mean |
|---|---|---:|---:|---|---:|---:|---:|
| Short | sequence_1_19 | 71.96 | 88.16 (95.8 %) | sequence_1_20 | 38.32 | 49.00 (27.8 %) | 68.58 |
| Medium | sequence_2_11 | 49.68 | 85.46 (99.5 %) | sequence_2_12 | 18.03 | 20.82 (0.4 %) | 53.14 |
| Long | sequence_3_17 | 50.06 | 71.43 (72.3 %) | sequence_3_18 | 34.68 | 36.13 (8.5 %) | 53.78 |
| Low light | sequence_4_10 | 38.99 | 67.81 (68.4 %)¹ | sequence_4_11 | 36.37 | 56.49 (37.6 %)¹ | 62.15 |
| Moving platform | sequence_5_11 | 43.14 | 60.32 | sequence_5_12 | 4.98 | 7.73 | 34.03 |
| **Mean** | | | **74.64** | | | 34.03 | **54.34** |

¹ 30 iterations per solve, as recommended in low light (see
[Iterations](#iterations)).

The first sequence of each track is the one the method was developed on.
On the second ones the BA gains far less. sequence_2_12, 3_18 and 1_20 keep
the heading drift of their VIO: see [A heading drift that is not linear:
sequence_2_12](#a-heading-drift-that-is-not-linear-sequence_2_12). The VIO
of sequence_3_18 runs +5.3° … −2.2° per 300 s window, and the BA +6.3° …
−2.1°.

sequence_5_12 starts on a moving platform. Between its first two control
points (19 s and 209 s) the route rises 23.4 m and runs 257 m in a straight
line, while the solve rises 9.6 m and runs 128 m. After that, its
control-point-to-control-point distances agree to 0–6 % for 15 minutes. The
first control point is 153 m off after the Sim(3) fit and the others 2–36 m,
until the last four minutes, where the solve runs 22–51 % too far.

The platform is a tram. From 75 s to 165 s the walking signature leaves the
accelerometer (the standard deviation of its norm falls from 1–4 to
0.13–0.34 m/s²), and the images show a tram's windows, ceiling and seats.
There, the solve's speed falls from 2.4 m/s to 1.2 m/s. Unguided matches
between consecutive keyframes split cleanly. On cam1, 100–224 of 300–500 are
explained by the gyroscope's rotation alone within 2 px: the cabin. An
essential matrix on the rest points the outside motion within 2–10° of the
solve's direction, so the direction holds and the speed does not. Starts
whose ride is 1.5, 2 and 2.5 times as fast end 2,000–3,200 lower in cost
than the VIO start (first solve), but score 5.21, 4.44 and 3.67 against
6.53. The first control point comes closer (153 → 102 m at twice the speed),
while the others drift too.

### Iterations

The solves used to stop at 30 iterations, most of them while the cost still
fell by 20–150 per iteration. At 60 (and `--rel-tol 1e-7`), in both solves
of the pipeline, final CP Score (pose recall at 1 m):

| Sequence | 30 iterations | 60 iterations |
|---|---:|---:|
| sequence_1_19 | 83.15 (93.4 %) | **88.16** (**95.8 %**) |
| sequence_1_20 | 46.31 (22.2 %) | **49.00** (**27.8 %**) |
| sequence_2_11 | 83.83 (99.3 %) | **85.46** (**99.5 %**) |
| sequence_3_17 | 65.41 (61.7 %) | **71.43** (**72.3 %**) |
| sequence_3_18 | 34.75 (8.7 %) | **36.13** (8.5 %) |
| sequence_4_10 | **67.81** (**68.4 %**) | 58.90 (43.1 %) |
| sequence_4_11 | **56.49** (**37.6 %**) | 52.27 (30.9 %) |
| sequence_5_11 | 57.17 | **60.32** |
| sequence_5_12 | 6.37 | **7.73** |

The extra iterations lower the cost by a few hundred at most (sequence_4_10,
first solve: 69,862 → 69,717, with 40 observations fewer), and yet move the
score by up to 9 points. They move the solution along the nearly flat
directions of a slow heading and scale drift. On sequence_4_10 the heading
error of the last 300 s grows from +1.1° to +1.8°. The seven day and
moving-platform sequences gain 1.4–6.0 points, the two low-light ones lose
4.2 and 8.9, where vision holds those directions least. 60 iterations are
the default; in low light the pipeline keeps 30 (`--iterations 30 --rel-tol
1e-5`). The second solve alone at 60 iterations, from the first solve's
30, scored 70.08 on sequence_4_10, so the first solve is where the extra
iterations hurt there.

Averaging the 30- and 60-iteration solutions frame by frame lands between
the two on every sequence (sequence_4_10: 65.35, sequence_3_17: 69.61). The
extra iterations move the solution one way, not around the truth.

In low light the error left is mostly local scale. Per 60 s window against
the pseudo-GT, the solve's scale is off by 2.4 % rms on sequence_4_10 and
2.2 % on sequence_4_11, against 0.7–1.4 % by day; the heading stays within
±0.9°. The VIO is worse still (5.4 % and 3.9 %). Keyframes at 10 Hz instead
of 2 Hz, on the 200 s window from keyframe 1000 of sequence_4_10 (the same
images, denoised SIFT, offsets scaled to the same times), do not tighten
it: 0.378 m against 0.389 m after a Sim(3) fit of the window, with sub-window
scales off by 4.8 % rms per 40 s against 4.1 %. The 2 Hz keyframes sample
the walking cadence at about one phase, but the scale does not hinge on
that.

### What is left on sequence_3_17

The CP errors left on the Long sequence are not a slow drift: fitting a
heading or scale drift to them leaves the median at 0.80 m (0.77 m without).
Against the pseudo-GT, the heading stays within ±0.7° for 1,500 s, then turns
by +0.7° between 1,500 and 1,590 s, and the position is 1.4 m further off
between 1,410 and 1,500 s (the CPs ETH2 and ETH1 there are 2.8 m and 2.0 m
off). That stretch crosses an open area against the sun, with dark ground and
a car passing close: the keyframes keep 160–170 observations (64 at the
least), against about 500 elsewhere. Over 1,380–1,580 s the VIO itself is
0.80 m off after a Sim(3) fit, its two halves 2.2 % apart in scale, and the
pipeline is 0.92 m off. Contrast equalization does not help these day
images on that window (one rotation per camera): 0.63 m without, 0.66 m with
CLAHE, 0.60 m with a clip limit of 1.5, 0.75 m with a gamma of 0.7.

In every image region the BA rejects about 1 % of the observations, so the
wearer's feet and the passers-by in the lower part of the images are already
left out by the matching.

### IMU-camera time offset

The BA reports the whitened IMU residual per factor. A sweep of the gyroscope
time offset on top of the per-device alignment (±2 ms, 150 s windows)
minimizes the total cost within 0.5 ms of the device table on both devices
(−0.4 ms on sequence_1_19, −0.5 ms on sequence_2_11). The time alignment
is therefore not what limits the sequences of 1WM093701G1276.

Nor does the exposure shift it. sequence_4_10's SLAM cameras expose for
13.98 ms throughout (gain 15), sequence_2_11's for 0.08 ms at the median (the
VRS image records), so a timestamp at the start or the end of the exposure
would move sequence_4_10 by 7 ms against the device table. Shifting both IMU
channels instead raises the cost of 150 s windowed solves (keyframes 100–400
and 1500–1800, one rotation per camera) on both sides:

| IMU shifted by | 0 ms | −6 ms | +6 ms | −12 ms | +12 ms |
|---|---:|---:|---:|---:|---:|
| Final cost, window 1 | **23,201** | 30,381 | 30,663 | 40,583 | 40,596 |
| Final cost, window 2 | **19,785** | 24,655 | 24,846 | 31,228 | 30,672 |

The timestamps are the middle of the exposure.

## Slow heading drift

### The solution keeps its starting point's drift

On sequence_1_20 the VIO trajectory drifts −5.3° in heading over 17
minutes, and the batch BA still ends at −4.6° (fitted to the CPs, CP Score
43.4). The drift is linear in time and does not follow the turns.

The usual suspects do not cause it:

| Run | Heading drift after the BA (fitted to the CPs) | CP Score |
|---|---:|---:|
| Both cameras | −4.64° | 43.4 |
| cam0 only | −5.99° | 43.4 |
| cam1 only | −4.55° | 44.1 |
| Gyro bias random walk ×1000 | −4.92° | 41.6 |
| Focal lengths, principal points and radial distortion estimated too | −4.42° | 45.2 |

The estimated focal lengths change by −0.13 % and the radial terms stay
near zero.

The starting point decides instead. Given the same observations, the BA
ends near the drift it starts from:

| Starting trajectory | Its heading drift (against the pseudo-GT) | After the BA (fitted to the CPs) | CP Score | Final cost |
|---|---:|---:|---:|---:|
| The VIO | −5.3° | −4.6° | 43.4 | 198,233 |
| A VIO with the device table's rig rotation | +10.5° | +7.2° | 29.3 | 200,686 |
| The pseudo-GT (diagnosis only) | 0° | −1.0° | 74.4 | (fewer observations) |
| The VIO, turned by −0.31°/min (diagnosis only) | −0.4° | −0.7° | 74.6 | 198,185 |

The objective barely tells these solutions apart: the drift-free one
started from the turned VIO ends 48 below the drifted one, out of 198,000.
Nothing separates them either. Along the straight path from the solution
started at the VIO to the one started at the pseudo-GT the cost falls
monotonically (by 12,400, most of it local differences), and it is nearly
flat at the first end. A heading drift that grows linearly in time moves
every keyframe and landmark a little. The gyroscope hardly sees it, because
one constant bias offset absorbs it; vision sees it only through the long
chain of overlapping windows. The damped Levenberg-Marquardt steps do not
follow such a long, flat direction: each one moves it by a fraction of a
degree, and the solve stops at its relative tolerance first.

### Deformation modes

`--modes yaw` adds that direction to the reduced system as one more border
unknown, next to the camera rotations: keyframe k turns about the vertical
by s_k times the mode's amplitude (s from 0 at the first keyframe to 1 at
the last). Positions are re-integrated from the turned displacements,
velocities turn, every landmark moves with its first keyframe, and the
gyro bias takes one offset (the mean body-frame vertical times the rate),
so that the preintegrated rotations still agree.

- The landmark coupling of the mode is eliminated with the landmarks, so
  they follow it optimally.
- The mode is undamped until a step fails; a rejected step damps it
  relative to its own curvature, and LM falls back to plain steps.
- Each step of the mode is capped at 0.01 rad (a trust region solved on
  the border alone).
- The mode's share of a step is applied as the exact finite deformation.
  GTSAM's Pose3 retraction couples rotation and translation, and a 0.09 rad
  turn through it would move far keyframes by decimetres.
- The mode joins at the fourth iteration, after the large first steps.

`--modes scale` stretches the displacements linearly in time instead.

On sequence_1_20 the yaw mode settles at −0.090 rad after 17 iterations:
CP Score 74.8 instead of 43.4, pose recall at 1 m 100 % instead of 12 %, a
fitted heading drift of −0.5°, from the VIO and without any ground truth.

Where the VIO starts with little drift, the mode moves the heading by a few
tenths of a degree, mostly the wrong way:

| Sequence | Without | `--modes yaw` | Fitted heading drift without / with |
|---|---:|---:|---:|
| sequence_1_19 | **83.02** | 74.81 | −0.14° / +0.46° |
| sequence_2_11 | **61.03** | 59.35 | −0.34° / −0.60° |
| sequence_3_17 | 50.14 | **53.64** | +0.82° / −0.05° |
| sequence_4_10 | **45.79** | 40.30 | −0.58° / −1.31° |
| sequence_5_11 | **59.06** | 56.19 | −1.88° / −2.25° (with PB2) |
| sequence_1_20 | 43.39 | **74.84** | −4.64° / −0.49° |

The objective is nearly flat along this mode: on sequence_1_20 the drifted
and the drift-free solutions end within 0–1,000 of each other out of
198,000. Its minimum is therefore only good to a few tenths of a degree,
which costs more than it brings when the start is already that good. The
mode stays off by default; it is the remedy for a start that drifts by
degrees, such as a VIO without the per-device refinement.

The tighter outlier threshold does not change that (3 px solve with one
rotation per camera, with and without the mode): sequence_1_20 goes from
44.43 to **76.13**, but sequence_1_19 from 80.63 to 74.87, sequence_2_11 from
68.88 to 59.17 and sequence_4_10 from 47.43 to 40.42. The cost cannot pick the cases either: the
mode lowers it by 126 on sequence_1_20 and by 107 on sequence_4_10, where it
turns the heading the wrong way, out of about 150,000–200,000. Neither does
Earth's rotation: with its full vector in the IMU factors (`--omega-coriolis`,
the north taken from the control points' LV95 frame for this check only), the
mode still ends at +0.50° on sequence_1_19 (74.50).

The current solver (eight rotation segments, the 3 px threshold) does the
same (first solve):

| Sequence | Without | `--modes yaw` | Cost on the observations active in both, without / with |
|---|---:|---:|---:|
| sequence_1_19 | **86.53** | 76.25 | |
| sequence_2_11 | **71.18** | 61.01 | 180,923 / 180,903 |
| sequence_4_10 (denoised features) | **63.31** | 51.50 | |
| sequence_1_20 | 44.41 | **77.74** (pose recall 99.7 %) | 147,923 / 147,685 |

On sequence_2_11 the mode also lowers the cost, by 20, and loses 10 points.

The scale mode, where the accelerometer does observe the mode, helps four
of the five tracks a little, but the low-light one, whose vision is weak,
loses 7 points:

| Sequence | Without | `--modes scale` | Mode at the end |
|---|---:|---:|---:|
| sequence_1_19 | 83.02 | **83.52** | −1.3 % |
| sequence_2_11 | 61.03 | **63.02** | +0.4 % |
| sequence_3_17 | 50.14 | **50.32** | −0.3 % |
| sequence_4_10 | **45.79** | 38.75 | −5.5 % |
| sequence_5_11 | 59.06 | **60.22** | +2.0 % |
| **Mean** | **59.81** | 59.17 | |

It stays off by default as well.

### A heading drift that is not linear: sequence_2_12

sequence_2_12 (Medium, 28 minutes) starts from a VIO whose heading error,
against the pseudo-GT per 120 s window, runs −4.1° … +1.6° … −0.4°: it is
neither linear in time nor a single step. The batch BA keeps it (−5.4° …
+1.2°): CP Score 21.0, pose recall 0 % (the VIO: 18.0). Locally it is
fine, with its 60 s segments 0.25 m off.

Started from the same VIO with only the heading error removed (from the
pseudo-GT, per 60 s window; diagnosis only), the first solve keeps the
heading within ±0.4° and scores 55.23 (pose recall 34.7 %, Sim(3) RMSE
1.67 m against 5.56 m). It also ends lower, at 172,310 against 173,335 with
0.3 % fewer observations, and its IMU cost is lower too, but both solves
stop at their 30 iterations still descending by about 40 per iteration, so
the difference does not show which minimum is lower. Nothing tried so far
gets there without the pseudo-GT (first solve, CP Score):

| Run | CP Score | Heading error per 300 s (°) |
|---|---:|---|
| Default | 20.98 | |
| Gyro bias random walk ×100 | 21.77 | −3.32 −1.58 −0.15 +0.58 +0.73 +0.18 |
| Gyro bias random walk ×1000 at the start, decaying over 300 s | 20.90 | −2.80 −1.58 −0.14 +0.66 +0.70 +0.10 |
| `--modes yaw` | 24.14 | −3.41 −1.49 −0.31 +0.42 +0.72 +0.49 |
| `--modes yaw8` (eight time pieces) | 17.65 | −3.03 −1.92 −0.57 +0.79 +2.18 +2.47 |
| IMU noise halved (`--imu-noise-scale 0.5`) | 19.22 | −4.24 −2.09 −0.40 +0.43 +0.73 +0.72 |
| No focal lengths, no keypoint sizes | 18.09 | −4.21 −2.12 −0.44 +0.40 +0.65 +0.47 |
| The same, one rotation per camera | 18.43 | −4.32 −2.21 −0.52 +0.40 +0.78 +0.64 |
| Triangulation check 15 px, outliers 30 px (the far observations stay in) | 19.99 | −4.09 −1.89 −0.45 +0.48 +0.82 +0.78 |
| The same, 3 px only from iteration 15, 40 iterations | 21.09 | −4.07 −1.79 −0.44 +0.52 +1.17 +1.28 |
| Started with the heading removed (pseudo-GT, diagnosis) | **55.23** | −0.09 −0.09 −0.24 −0.12 +0.40 +0.41 |
| Started at the pseudo-GT (diagnosis; 30 % of the observations) | **71.82** | +0.28 −0.01 −0.07 −0.18 +0.10 −0.12 |

`--modes yawN` splits the heading mode into N time pieces. Each piece turns
the heading over its own piece only, and the gyro bias takes the piece's turn
spread over the whole recording: a constant bias absorbs the pieces' mean
rate, and the gyroscope sees how they differ. Its steps on sequence_2_12
change sign from one iteration to the next, and it stops higher than the
default. Shapes chosen without ground truth do not help either. Starts with
an exponential heading change A (1 − e^(−t/300 s)) cost more than the VIO
after eight iterations for A = −7.5° (where the pseudo-GT puts it) and only
slightly less for −2.5° (230,149 against 230,437). Turning the solve to
the heading the gyroscope integrates with the solve's bias doubles the
error: that heading drifts +5° the other way.

The VIO does not cause it at its start either. With the solve's gyro bias
subtracted from the start (`calib_gyro_bias`), with a 100 times looser gyro
bias random walk, or with the solve's IMU-camera rotations instead of the
per-device rig rotation, its first five minutes keep the same heading error
(−3.6°, −4.6° and −4.2° in the first 30 s, against −4.0°). The online demo's
`trajectory_vio.tum` is the same as the VIO-only demo's there, so the
VIO-only runs of the second sequences of each track (sequence_1_20, 2_12,
3_18, 4_11) are not the difference either.

The far-apart views do not hold the heading either. On sequence_3_18, the
image pairs of the same camera with at least 40 matches number 1,152 at 24
keyframes apart, 282 at 64, 89 at 96 and 9 at 192: the 100,000 matches per
offset beyond 64 keyframes are spread a few per pair. An essential matrix
(RANSAC, 1 px) on those pairs measures the relative heading to about 0.3°.
That is more than the solve's heading error over such spans (0.11° at the
median against the pseudo-GT, on the 36 pairs it covers), and it does not
follow that error (correlation 0.05). Rotation averaging over these pairs
would know less than the BA already does.

Revisits cannot fix it on these routes. By the pseudo-GT, the keyframes
that come back within 10 m more than 120 s later are 3.8 % on sequence_2_12
(204 s at most), 3.5 % on sequence_1_20, and none on sequence_3_18,
sequence_4_11 and sequence_3_17, against 9.8 % on sequence_2_11. The
heading over a one-way route is the odometry's alone. sequence_3_17 and
sequence_2_11 start from VIOs that stay within ±1.5° and ±0.5° of it;
sequence_2_12, 3_18 and 4_11 do not. Within 30–90 s the
heading changes the solve, the gyroscope and the pseudo-GT measure between
pseudo-GT timestamps agree to 0.2–0.4° per interval. The solve still turns
1.5° more than the other two over the minute.

### The first keyframe's rotation prior

The gauge prior pins the first keyframe's pose, rotation included, at the
VIO's. On sequence_2_11 the first 30 IMU factors (15 s) then carry half of
the whole sequence's gyroscope residual and 44 % of its velocity residual:
the first keyframe has no visual observation, the VIO initialized while the
wearer walked, and its tilt is off. `--prior-rot-sigma 0.05` (rad) lets the
solve fix the tilt; gravity observes roll and pitch, and the 0.05 rad prior
still fixes the heading's gauge. The IMU cost halves on every sequence
(sequence_2_11: 20,600 → 11,700), but the CP Scores split (with the 3 px
threshold, prior loose minus pinned): +0.6, +0.1, −1.0, −4.6 and +4.2 on
sequence_1_19, sequence_1_20, sequence_2_11, sequence_4_10 and
sequence_5_11. A different start shifts the gyro bias and with it the
heading drift, along the same flat direction as above. It stays at 1e-4 by
default.

### Starting from the online mapper

On Medium and Moving platform the online mapper's trajectory beats the VIO
([`lamaria_plan.md`](lamaria_plan.md), lever 11). Starting the BA from it
(the observations still selected with the VIO) gives:

| Sequence | Start: VIO | Start: mapper | Final cost VIO / mapper start | Fitted heading drift VIO / mapper start |
|---|---:|---:|---:|---:|
| sequence_2_11 | **61.03** | 54.41 | 240,741 / 228,230 | −0.34° / +0.67° |
| sequence_5_11 | 59.06 | **62.84** | 294,609 / 293,437 | |

On sequence_2_11 the mapper start ends better by every measure but the
Score: a lower cost, half the IMU cost (11,300 against 20,600), the same
Sim(3) RMSE against the pseudo-GT (1.50 m) and a lower 60 s segment RMSE
(0.199 m against 0.206 m). It keeps the mapper's heading drift, though, and
that is what the CPs see. The lowest cost does not pick the better start.

### Where the second sequences' heading drift comes from

Against the pseudo-GT, per window, the heading change of the solve, of the
VIO and of the gyroscope integrated with the solve's biases (first solve):

| Sequence | Solve, whole sequence | Of it in the first ~6.5 min | VIO | Gyroscope |
|---|---:|---:|---:|---:|
| sequence_2_11 | within ±1.2° | −0.1° | −1.3° | +3.3° (+2.7° in the first 2 min) |
| sequence_2_12 | +9.0° | +5.5° | +6.8° (+5.1° in the first minute) | +14.0° |
| sequence_3_18 | −9.5° | −3.7° | −10.1° | −6.2° |

The gyroscope with the solve's biases drifts steadily: on sequence_2_12 its
vertical bias is 0.15 mrad/s off on average, whether the wearer walks or
not. The solve follows its errors there: the 30 s windows' heading errors,
with the trend removed, correlate with the gyroscope's at +0.76, against
+0.39 on sequence_2_11.

Without ground truth nothing pins that bias. On sequence_3_18 the first
solve started from the VIO with its heading error removed (pseudo-GT,
diagnosis only) keeps it and scores 74.97 instead of 36.01 (pose recall
89.4 % against 9.1 %), but ends at a higher cost (197,486 against 196,792):
its reprojection cost per observation is higher in 12 of the 15 windows of
120 s (by up to 0.7 %), the IMU cost the same. Both solves have converged. The objective
prefers the drift slightly, so neither more iterations nor another optimizer
would fix it. Nor do these:

- A third solve from the pipeline's result: 36.13, unchanged.
- Matching guided by the solve instead of the VIO: 8.6 % more matches, three
  quarters of them at offsets 4–32; at 64 and more, 0–4 % more.
- Revisits up to 100 m apart, at least 128 s later, with the optical axes
  within 45°: of 6,490 image pairs, none is verified.
- Longer tracks: a landmark of the solve projects into the image of 564,507
  keyframes within 32 of its observed span, but is observed in 8.9 % of them,
  and a feature with a similar descriptor lies within 3 px in another 1.5 %.
  SIFT does not find the same point again as the view changes.
- Far points: without the 1° triangulation angle the solve gains 69,537
  landmarks, 124 of them over 64 keyframes or more.
- The stops: sequence_3_18 has one of at least 4 s.
- A prior from other recordings: on 1WM093701G1276 the vertical component of
  the solved gyro bias ranges from −0.29 to +0.36 mrad/s between them.
- Horizontal building lines as a compass: by the pseudo-GT, the dominant
  direction of the frame's horizontal lines stays within 2° for at least 20 s
  in 11 stretches (285 s of 1,689 s), to 0.37° per frame. The shift between two
  frames' line-direction histograms 8–40 s apart follows the true change of
  the solve's heading error at a correlation of 0.35 at most, 0.34–0.71° off at
  the median, and its mean rate over the sequence misses the drift. On
  training R_12_10cp (cam0 at 2 Hz) the histograms of 10 s windows 30–180 s
  apart differ by 1.5–2.6° at the median even by the pseudo-GT, against
  0.15–0.18° of change in the heading error, and by the solve their shift
  does not follow it (correlation −0.19 … 0.00).

Over a one-way route, the heading beyond a minute is the integrated
gyroscope's, and the cameras and the IMU leave its bias undetermined.

### Far points out of view

Long tracks would hold the heading, but a head-worn camera keeps far points
in view only briefly. Along the BA's trajectory of sequence_3_18 (300–420 s
and 900–1,020 s), a direction on the horizon is in cam0's or cam1's view
(rectified, 758 × 572) 22–24 % of the time, for a median of 1.0 s at a time,
4.4 s at the 90th percentile and never 30 s. Pyramidal KLT at 20 Hz
(forward-backward within 0.5 px) comes close to that limit: on 300–330 s its
tracks last a median of 0.65 s, 2.7 s at the 90th percentile and 11 s at
most, and 55 % of those lasting 2 s or more triangulate within 2 px RMS along
the BA's trajectory (median depth 12 m). A better tracker cannot make them
much longer.

Nor would the fisheye's whole image. The rectified images keep 115° × 100°
of the SLAM cameras' view, about 152° × 114° on the 640 × 480 fisheye at the
same focal length. Over the whole fisheye a horizon direction is in cam0's view
29–30 % of the time instead of 21–23 %, and in both cameras' 8.1–8.9 % instead
of 3.7–4.4 %, but still for a median of 1.1 s at a time (4.0–4.3 s at the
90th percentile).

Nor do the points come back recognizably. For 351 such landmarks at least
20 m away (300–480 s), each return into view was searched for by warping the
patch around the landmark's middle observation into the new image (the plane
through the point facing that camera) and by normalized cross-correlation
within ±30 px of its projection. A find counts as right within 2 px of the
epipolar line, which a random one is in 5–8 % of the cases:

| Out of view for | Returns | Right among the finds (NCC ≥ 0.7) | Right among the returns |
|---|---:|---:|---:|
| 0 (the track's own ends) | 368 | 82 % | 76 % |
| 1–5 s | 438 | 56 % | 52 % |
| 5–20 s | 770 | 33 % | 26 % |
| 20–60 s | 1,558 | 20 % | 15 % |
| 60 s or more | 2,447 | 9 % | 3 % |

Walking for 20–60 s moves the camera 25–80 m, as far as the "far" points
themselves, so their look changes as much as their place. The links within
20 s the chain already has from its matches up to 64 keyframes apart.

The pseudo-GT could not serve as the reference here. Its orientations are
those of another body frame, 105° from the IMU's, and at 3.3 Hz it cannot be
interpolated to every image while the head turns. The BA's trajectory was
used instead; its own error may count some right finds after long gaps as
wrong.

### The gyroscope's share of the drift

Against the pseudo-GT (its orientations turned into the IMU's frame by the
rotation fitted to their changes), the chain's heading error changes over
256 s by 0.00° on average on sequence_1_20 and +0.03° on 2_11, with a spread
of 0.29–0.30° as over 1 s (the pseudo-GT's own noise); by −0.30° on 1_19 and
by −0.71° on 3_18 (spread 0.66°). Where the chain loses, the mean grows in
proportion to the time: a drift, not a random walk of noisy rotations.

On sequence_3_18 the pass solve once more, with the gyroscope's noise density
10 and 100 times the calibration's (the accelerometer's unchanged), so that
the cameras decide the rotations:

| Trajectory | Heading error at the end | Change over 256 s (mean) | CP Score |
|---|---:|---:|---:|
| The VIO | −26.9° | −6.80° | |
| The online result (the pass's start) | −3.1° | −0.93° | |
| The pass solve | −2.6° | −0.71° | **54.35** |
| Gyroscope noise ×10 | −2.1° | −0.64° | 52.09 |
| Gyroscope noise ×100 | −1.3° | −0.51° | 54.07 |

Without the gyroscope the drift shrinks by a third and the score stays. The
gyroscope's model does not make the drift. The cameras' constraints are too
weak over minutes to take out what the solve starts from ([The solution keeps
its starting point's drift](#the-solution-keeps-its-starting-points-drift)),
and a head-worn camera does not give them longer ones.

### A learned dense front end, and the VIO's calibration

Measured below at the poses whose timestamps match the pseudo-GT's within
2 ms, without interpolation (a keyframe trajectory cannot be interpolated
while the head turns): the body rotation fitted to the rotations' changes
between consecutive matches, the world rotation to the first 60 s, then the
error's rotation about the pseudo-GT's vertical. It counts the other way round
from the section above (the pass solve drifts by + here).

DROID-SLAM (monocular, its TartanAir weights, the frontend without the final
global BA) on sequence_3_18's rectified cam0, 300–700 s, at 10 Hz and
384 × 504:

| Trajectory | Rotation change over 1–3 s, off the pseudo-GT's (median) | Heading change over 256 s (mean) |
|---|---:|---:|
| The chain's pass solve | **0.07°** | **+1.34°** |
| The VIO | 0.07° | +7.04° |
| DROID-SLAM | 0.26° | +14.1° |

At 5 Hz and 240 × 312 (300–1,020 s) DROID-SLAM is 1.9–3.0° off over 1–3 s on
either camera and turns by 22–34° per 256 s. A learned dense front end does
not hold the heading better here; DBA-Fusion, which fuses DROID-SLAM's dense
BA with an IMU (monocular, GPL-3.0), was not pursued.

The VIO's drift takes the device's sign. Heading change over 256 s (mean),
whole sequences:

| Sequence | Device | VIO | Chain |
|---|---|---:|---:|
| sequence_1_19 | 1WM09370081264 | −1.27° | +0.19° |
| sequence_1_20 | 1WM09370081264 | −2.79° | +0.02° |
| R_11_5cp | 1WM093701G1276 | +1.12° | +1.74° |
| R_12_10cp | 1WM093701G1276 | +3.55° | +0.31° |
| sequence_2_11 | 1WM093701G1276 | +1.88° | +0.01° |
| sequence_3_18 | 1WM093701G1276 | +4.24° | +0.49° |
| sequence_4_10 (night) | 1WM093701G1276 | +0.57° | +0.13° |
| sequence_4_11 (night) | 1WM093701G1276 | −1.78° | −0.19° |

The chain takes most of it out; what it leaves is on 1WM093701G1276. Its own
calibration explains little of the VIO's drift, though. sequence_3_18's VIO
again, with the chain's estimate of each camera's rotation (mean of the eight
segments) and of the focal lengths (a calibration that knows the answer,
diagnosis only):

| VIO calibration | Heading change over 256 s | Heading error at the end | CP Score (VIO alone) |
|---|---:|---:|---:|
| The device table's | +4.24° | +26.6° | 19.94 |
| The chain's camera rotations | +3.93° | +24.5° | 22.41 |
| The chain's camera rotations and focal lengths | +3.20° | +20.1° | 24.61 |

A quarter of the drift goes, the rest stays: it is the VIO's short window,
which the chain's matches and revisits take out, not its calibration.

### A device's gyroscope correction

The rectified gyroscope still leaves a correction that depends on the device.
Fitted against the pseudo-GT's rotation changes over about 1 s (diagnosis
only): a symmetric correction, ω ← (I + δM) ω, with three axis scales and
three non-orthogonality terms. It is linearized around the measured
integration, with a bias per 60 s and a time offset. The antisymmetric part
is left out, since the fixed rotation between the IMU and the pseudo-GT's body
absorbs it.

| Sequence | Device | Scale x / y / z | Non-orthogonality xy / xz | Time offset | Residual, before → after |
|---|---|---|---|---:|---|
| sequence_1_19 | 1WM09370081264 | +0.13 / −0.01 / **+0.22 %** | −1.04 / −0.84 mrad | +0.6 ms | 0.229° → 0.115° |
| sequence_1_20 | 1WM09370081264 | +0.12 / −0.07 / **+0.19 %** | −1.12 / −0.64 mrad | +0.5 ms | 0.219° → 0.106° |
| R_11_5cp | 1WM093701G1276 | −0.23 / −0.21 / **−0.13 %** | −0.99 / −0.89 mrad | +1.0 ms | 0.287° → 0.134° |
| R_12_10cp | 1WM093701G1276 | −0.19 / −0.20 / **−0.15 %** | −0.85 / −0.75 mrad | +1.1 ms | 0.259° → 0.125° |
| sequence_2_11 | 1WM093701G1276 | −0.02 / −0.04 / **−0.13 %** | −0.48 / −0.90 mrad | +1.1 ms | 0.212° → 0.120° |
| sequence_3_18 | 1WM093701G1276 | +0.01 / −0.03 / **−0.13 %** | −0.54 / −0.94 mrad | +1.4 ms | 0.252° → 0.130° |
| sequence_4_10 | 1WM093701G1276 | −0.12 / −0.15 / **−0.13 %** | −0.63 / −0.32 mrad | +0.4 ms | 0.180° → 0.145° |
| sequence_4_11 | 1WM093701G1276 | −0.16 / −0.17 / **−0.12 %** | −0.49 / −0.72 mrad | +0.4 ms | 0.183° → 0.128° |

The z axis's scale splits by device: +0.19 … +0.22 % on 1WM09370081264,
−0.12 … −0.15 % on all six sequences of 1WM093701G1276. The biases and the
time offset alone take most of the residual down (sequence_3_18: 0.133°,
sequence_1_20: 0.136°).

The VIO takes the correction well. With a correction fitted on the device's
other sequences (leave one out: sequence_1_19's for 1_20, the mean of the
other five for 1WM093701G1276), CP Score of the VIO alone and its heading
change over 256 s:

| Sequence | Without | With the device's correction |
|---|---:|---:|
| sequence_1_20 | 33.55 (−2.79°) | **40.12** (−1.56°) |
| sequence_3_18 | 19.94 (+4.24°) | **26.38** (+2.79°) |
| sequence_4_11 | 41.95 (−1.78°) | **53.83** (−0.99°) |

The chain does not. With the correction in the VIO that starts it, in the IMU
it integrates, or in both (CP Score; the chain repeats its results exactly for
the same inputs, so every difference is the correction's):

| Sequence | Neither | The VIO | The chain's IMU | Both |
|---|---:|---:|---:|---:|
| sequence_1_20 | **88.65** | 77.94 | 77.62 | 79.04 |
| sequence_2_11 | **89.25** | 84.03 | 83.37 | 84.24 |
| sequence_3_18 | 54.27 | 53.83 | 58.15 | **58.18** |
| sequence_4_10 | **59.80** | 47.53 | 57.87 | 56.43 |
| sequence_4_11 | 71.07 | 66.51 | **73.40** | 69.87 |
| **Mean** | **72.61** | 65.97 | 70.08 | 69.55 |

The device's correction is not kept. What it shows is how far the chain moves
when its input moves a little. A sequence changes by up to 11 points.
sequence_1_20 and 2_11, near 89, lose 5–11 whatever changes, and
sequence_3_18 gains. On five to eight training sequences, gains of 2–3 points
in the mean cannot be told from that (and submission 6's did not carry over to
the test set, [lamaria_test_results.md](lamaria_test_results.md)).

### A ride the VIO halves

Keyframes on a moving platform are easy to find without ground truth: the
accelerometer norm varies by less than 0.5 m/s² over 4 s while the
trajectory still moves faster than 0.5 m/s. That marks 83–166 s of
sequence_5_12 (the tram; and 10 s around it), 119–188 s of sequence_5_11 (the
funicular) and nothing on sequence_3_18. The tracks of the cabin, whose observations the
rotation alone explains within 3 px, are already gone, though: dropping
them leaves sequence_5_12's landmarks the same (310,282 against 310,284),
and of sequence_5_11's 1,047 only 78 pass the triangulation check. The BA
keeps the ride the VIO gives it. Starts with the ride's displacements scaled
(first solve, sequence_5_12):

| Ride ×  | Ride length against the survey | Final cost | CP Score |
|---|---:|---:|---:|
| 1 (60 iterations, the others 30) | 0.50 | 131,722 | 8.05 |
| 1.5 | 0.70 | 130,613 | 5.21 |
| 2 | 0.92 | 130,122 | 4.44 |
| 2.5 | 1.15 | 129,381 | 3.67 |

The cost falls past the true length, so it cannot choose the factor, and the
score is set elsewhere: after the ride the CP errors (pipeline) stay at 2–16 m
for 13 minutes and grow to 24–37 m in the last five, where the distances
between the CPs come out 15–42 % too long after the Sim(3) fit.

Nor is the ride's length an accelerometer bias the solve made up. Over the
ride the first solve's accelerometer bias changes by 0.07 m/s² on one axis,
seven times its random walk's 1σ; with a random walk ten times stiffer
(`--accel-rw-scale 0.1`) it stays within 0.01 m/s², but the ride's
displacement from 80 s to 170 s stays 157 m and the score 8.11 (4.73 from
the ride doubled). The directions between consecutive CPs show what is left: in
every solution (pipeline, online, ride doubled) the first one, across the
ride, points 16–20° off, the next twelve 0.7–5.4° off, and the last three
+21°, −68° and +23° off.

### The rides, again

With the chain of [The first minutes, online](#the-first-minutes-online)
(pass with the looser random walk), sequence_5_11 scores 58.88 and
sequence_5_12 7.41. On sequence_5_11 every CP after the funicular is within
1.6 m and only the one before it misses (32.8 m); lengthening the ride by
1.19 or 1.25 (diagnosis, displacements rescaled) gives 61.17 and 63.62.

On sequence_5_12 the ride is not halved any more but folded. The solve
runs forward about 220 m at 3–6 m/s from 80 s to 130 s, stops at 132 s and
goes back 65 m by 186 s, so the first two CPs (19 s and 208 s) come out 151 m
apart against 257 m. The tram does not stop there: in consecutive frames
1 s apart at 146 s and 152 s the street outside shifts as much as at 95 s,
and the outer camera's parallax after removing the rotation stays at 4–8°
as on the ride (the inner camera, which sees the cabin and an advert inside
it, 1–2°). The solve does not take that turn from the cameras. Without any
camera observation from 78 s to 188 s, with the accelerometer bias held at
zero (prior 0.003 m/s², random walk ×0.01), or with the triangulation gate
at 20 px (and a 0.3° minimum angle), its velocities over the ride stay within
0.5 m/s of the pass and the score at 7.36–8.00. Over 110 s the IMU alone
cannot hold a ride: a tilt 0.3° off (a gyro bias 0.05 mrad/s off) leaks
0.05 m/s² of gravity into the direction of travel, a few m/s after a minute,
and the rectified IMU dead-reckoned from 78 s ends 808 m away. The solve
keeps the ride it starts from.

The last four minutes are not a scale error. Counted on the accelerometer
(peaks of the 1–3.5 Hz band at least 3 m/s² prominent, windows of 30 s at
1.5–2.2 steps/s), the pseudo-GT's stride varies by 4.7–11 % over the
steady-walking windows of six day sequences, and the solve's stride on
sequence_5_12 stays at 0.81 m (4.1 %) to the end; the CP chords there are
off by the turns (+21°, −68°, +23°). Rescaling the end by 0.8 (diagnosis)
gives 14.18: the walk after the ride still drifts from 2 m to 15 m.

### Finding the rides the VIO missed

On the test set, sequence_5_5 and 5_6 still score 3.1 and 2.4 with both
cameras' features (submission 4, [lamaria_test_results.md](lamaria_test_results.md)):
through a door's windows too few street tracks last, and the VIO and the BA
keep the cabin's standstill (sequence_5_5: about 35 m from 10.5 to 12.7 min).
The IMU shows such a ride without ground truth. Over 10 s windows, compare
the trajectory's velocity change with the accelerometer integrated in the
trajectory's orientation, its body-frame bias and the gravity vector fitted by
least squares on the windows outside the runs without steps (the 1.4–2.6 Hz
band of the accelerometer norm under 0.05 (m/s²)² for 30 s or more). On the
walks the two agree within 0.23–0.29 m/s at the median. In 3–4 windows of a
missed ride they differ by more than 1 m/s (sequence_5_5, 5_6, and 5_9 in
submission 3); in none of a ride the trajectory keeps (sequence_5_9 in
submission 4, training sequence_5_11 and 5_12).

Bridging such a ride with the IMU is not accurate enough, though.
Dead-reckoned from the trajectory's state where the run starts, the
gyroscope bias fitted to meet its orientation where the run ends and a
constant acceleration to meet its velocity there, the ride between training
sequence_5_12's CPs at 18 s and 208 s comes out 433 m against 257 m surveyed
(the trajectory: 263 m), and sequence_5_11's between 9 s and 264 s 167 m
against 203 m (the trajectory: 173 m). About 1° of tilt while the vehicle
accelerates is enough ([The rides, again](#the-rides-again)). The
trajectory's own orientation is no better: inside a missed ride the VIO tilts
to explain the vehicle's acceleration away, and dead reckoning in it gives
sequence_5_9's ride 265 m, against the 447 m of the run that kept it. Nor does
the velocity check see a wrong speed held constant, which is how
sequence_5_11's trajectory stays 15 % short and still passes. The chain does
not bridge rides.

### The platform filter where it stays off

The platform filter ([`platform.rs`](../pipelines/basalt/src/platform.rs))
turns on where enough tracks move beyond the gyroscope's rotation with one
common translation: at least 8 of them and 15 % of those it evaluates, in 14 of
20 frames, while the accelerometer is quiet. Its test, run on KLT tracks of the
rides (forward-backward within 0.5 px; they last longer than the VIO's, so the
counts are an upper bound), per camera (cam0 / cam1):

| Ride | Translation inliers per frame (median) | Their share | 20-frame windows that turn the filter on |
|---|---:|---:|---:|
| training sequence_5_12, 83–166 s | 76 / 40 | 34 % / 19 % | 83 % / 64 % |
| test sequence_5_5, 10.5–12.7 min (missed) | 7 / 8 | 7 % / 7 % | 21 % / 12 % |
| test sequence_5_6, 14.0–15.5 min (missed) | 4 / 6 | 2 % / 3 % | 7 % / 11 % |
| test sequence_5_9, 5.2–7.0 min (kept since submission 4) | 19 / 25 | 19 % / 22 % | 48 % / 56 % |

On a walk (sequence_5_12, 20–70 s) 2 % of cam0's windows would, as the
accelerometer is rarely quiet there.

Looser thresholds (`VISLOC_PLATFORM_PARAMS=min_inliers=5,min_inlier_share=0.05,evidence_on=10`)
let the VIO move through sequence_5_5's tram ride: 201 m of path and 165 m
net from 10.5 to 12.7 min, against 56 m and 3 m in submission 6, with the
walks before and after within 1 % of its path. They find 55 rides there,
though, and false ones on the training sequences, whose VIO loses (CP Score
of the VIO alone):

| Sequence | Default thresholds | Looser | Rides (frames), default → looser |
|---|---:|---:|---|
| sequence_5_12 | **6.40** | 5.05 | 10 (1,843) → 16 (1,933) |
| sequence_5_11 | **39.34** | 33.59 | 13 (1,533) → 21 (1,892) |

The defaults stay. Loosening only while the accelerometer disagrees with the
VIO's velocity ([Finding the rides the VIO missed](#finding-the-rides-the-vio-missed))
would leave the training sequences as they are, but nothing in the training
set could show what it does on a missed ride.

Nor does a missed ride made on purpose. With a training ride frozen in the
VIO (standing still through it, the rest moved back by its displacement), the
chain keeps the standstill whatever matches its pass:

| Sequence | Chain from the real VIO | Ride frozen: SIFT | ALIKED, essential matrix | ALIKED, two essential matrices¹ |
|---|---:|---:|---:|---:|
| sequence_5_12 (83–166 s) | 5.80 | 6.62 | 6.62 | 6.93 |
| sequence_5_11 (119–188 s) | **51.46** | 44.47 | 44.44 | 44.33 |

¹ A second essential matrix fitted to the first one's outliers (the street
outside the cabin); an experiment, not kept.

On sequence_5_12 the chain from the real VIO covers 267 m between the CPs at
19 s and 208 s (surveyed: 257 m), and the frozen ones 1 m with SIFT and 13 m
with the two essential matrices. The solve moves a trajectory only near where
it starts, and the street's observations do not triangulate from a standstill.

### Low light: matching guided by the solve

Matching the low-light sequences again, guided by the pipeline's final
trajectory instead of the VIO, and running the pipeline once more from the
VIO (30 iterations):

| Sequence | First solve | + `loop-correct` | + second solve |
|---|---:|---:|---:|
| sequence_4_10, VIO-guided matches | 63.44 | 63.09 | 67.81 |
| sequence_4_10, solve-guided matches | **69.11** | **69.57** | **71.13** (pose recall 74.5 %) |
| sequence_4_11, VIO-guided matches | **57.22** | **56.47** | **56.49** |
| sequence_4_11, solve-guided matches | 54.47 | 54.05 | 53.97 |

The track mean moves by 0.4: within the spread of the flat directions.

### Online mode

`online` runs the solve as the data comes in. Every 60 keyframes (30 s) it
solves the latest 480 (240 s) from the trajectory so far, with 10
iterations, two rotation segments and the revisits inside the window, and
carries the window's last correction to the frames after it. A match or a
revisit between keyframes i < j uses the VIO poses of i and j only, so the
batch matches and revisits are what a system running alongside the VIO
would have found by the time of j, and no window sees data after its last
keyframe. The window's first keyframe stays at its earlier estimate (the
solve's gauge prior).

Each window starts from the biases, velocities, camera rotations and focal
lengths of the one before. From zero the gyro bias needs about 30
iterations to settle in a 480-keyframe window (sequence_1_19), and with 10
the windows let it wander and the heading drift: started fresh, the online
sequence_1_19 scored 54.67 and its heading ran −1.9° … +2.9°; warm-started,
it stays within ±0.65°.

From the batch pipeline's VIO, with its matches and revisits, `--loops`,
the keypoint-size noise and `--est-focal` (not on the moving platform); two
sequences at a time on 4 cores, two threads each:

| Sequence | Batch pipeline | Online | Pose recall @ 1 m, online | Online run / recording |
|---|---:|---:|---:|---:|
| sequence_1_19 | **88.16** | 75.73 | 86.9 % | 0.69 |
| sequence_1_20 | 49.00 | **65.37** | 60.5 % | 0.76 |
| sequence_2_11 | **85.46** | 83.12 | 100.0 % | 0.67 |
| sequence_2_12 | **20.82** | 19.87 | 1.6 % | 1.34¹ |
| sequence_3_17 | **71.43** | 70.33 | 80.3 % | 0.73 |
| sequence_3_18 | 36.13 | **57.87** | 51.2 % | 0.76 |
| sequence_4_10 | 67.81 | **69.79** | 72.4 % | 0.40 |
| sequence_4_11 | 56.49 | **61.03** | 55.0 % | 0.42 |
| sequence_5_11 | **60.32** | 55.95 | – | 0.65 |
| sequence_5_12 | **7.73** | 5.17 | – | 0.58 |
| **Mean** | 54.34 | **56.42** | | |

¹ Alongside four other jobs; the others two at a time.

The one-way routes gain the most. Against the pseudo-GT after the
evaluator's Sim(3) fit, per 120 s window:

| Sequence | Heading error, range: batch | online | Sim(3) scale: batch | online | Mean 2D error: batch | online |
|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | 1.26° | 1.00° | 0.9973 | 0.9999 | 0.34 m | 0.55 m |
| sequence_1_20 | 4.14° | 1.83° | 1.0056 | 1.0036 | 1.67 m | 0.90 m |
| sequence_2_11 | 0.57° | 0.26° | 1.0061 | 1.0030 | 0.36 m | 0.39 m |
| sequence_3_18 | 8.09° | 2.40° | 1.0210 | 1.0048 | 4.09 m | 1.48 m |

The batch solve keeps the heading drift of its starting trajectory over the
whole recording (see [The solution keeps its starting point's
drift](#the-solution-keeps-its-starting-points-drift)); on sequence_3_18 it
starts 6.3° off like the VIO. Online, the VIO's drift comes in 30 s at a
time, and each window starts from a first keyframe at the earlier windows'
estimate and from their gyro bias, so only a little of it stays. On
sequence_2_12 the heading error builds up in the first minutes, while the
windows are still short: −6.0° in the first 120 s window against the
median (batch −5.6°, VIO −4.1°). Where revisits tie the whole recording
together (sequence_1_19, sequence_2_11), the batch solve is better: a
window sees the revisits within its 240 s only. Low light gains too, as
with fewer iterations (see [Iterations](#iterations)). The moving platform
does not.

### Online, then one batch pass

After the recording, the batch pipeline's steps (`solve --save-states`,
`loop-correct`, `solve --loops`, the same options and iterations) can start
from the online trajectory instead of the VIO:

| Sequence | Batch pipeline | Online | Online + batch pass (pose recall @ 1 m) |
|---|---:|---:|---:|
| sequence_1_19 | **88.16** | 75.73 | 79.20 (89.4 %) |
| sequence_1_20 | 49.00 | 65.37 | **73.03** (90.2 %) |
| sequence_2_11 | 85.46 | 83.12 | **85.55** (100.0 %) |
| sequence_2_12 | **20.82** | 19.87 | 16.57 (0.0 %) |
| sequence_3_17 | 71.43 | 70.33 | **75.57** (79.4 %) |
| sequence_3_18 | 36.13 | 57.87 | **67.39** (68.2 %) |
| sequence_4_10 | 67.81 | **69.79** | 51.44 (22.1 %) |
| sequence_4_11 | 56.49 | **61.03** | 52.33 (39.1 %) |
| sequence_5_11 | 60.32 | 55.95 | **61.16** |
| sequence_5_12 | **7.73** | 5.17 | 7.66 |
| **Mean** | 54.34 | 56.42 | 56.99 |

By day the pass keeps the online heading and adds the revisits across the
whole recording: after its first solve the heading error of sequence_3_18
spans 2.13° per 120 s window (online 2.40°, batch 8.09°) and the mean 2D
error falls to 0.88 m (1.48 m, 4.09 m); on sequence_1_20, 1.23° and 0.59 m.
In low light it drifts again, as with more iterations: sequence_4_10's
heading error runs −1.6° … +2.4° after its first solve, against −1.2° …
+0.6° online. The online result in low light and the pass elsewhere,
chosen by track and not by sequence, average **59.70**:

| Track | Batch pipeline | Online, then the pass (low light: online) |
|---|---:|---:|
| Short | 68.58 | **76.12** |
| Medium | **53.14** | 51.06 |
| Long | 53.78 | **71.48** |
| Low light | 62.15 | **65.41** |
| Moving platform | 34.03 | **34.41** |
| **Mean** | 54.34 | **59.70** |

### Real time

A window finished within the 30 s of data the next one waits for (12–23 s
on average per sequence, at most 28.0 s), except 6 of the 34 of
sequence_1_20 (up to 34.8 s; the next ones caught up) and those of
sequence_2_12, which ran alongside four other jobs. What each stage takes
at the recording's pace, in CPU seconds per second of recording (4 cores;
the VIO from its CPU time while it waited for the frames as they arrived):

| Stage | Cores |
|---|---:|
| VIO, Basalt `euroc_config` | 1.47 |
| SIFT on the 2 Hz stereo keyframes | 0.85 |
| The same in low light, with non-local means (h = 7, search window 21) | 1.83 |
| The same, h = 11, search window 11 (`--denoise-window 11`) | 1.09 |
| `match` (`--tau-slope 0.3 --tau-max 20`) | 0.22 (0.27) |
| `loops` (candidate stride 2; stride 4) | 1.08; 0.26 |
| `online` (two threads) | 1.1 |
| **Sum, by day, `loops --step 4`** | **3.9** (4.0) |

The VIO of every result above, `euroc_config_big_window_multicam`, runs at
0.42 times real time on these 4 cores; `euroc_config` at 1.95 times. From
it, sequence_2_11 scores 72.50 online (the VIO alone 29.67), against 83.12
from the big-window VIO; with `loops --step 4` 72.48, without the revisits
66.99. Matching and revisits use only the VIO poses of the two keyframes,
and of the 1,579 verified revisits of sequence_2_11 all but one are within
480 keyframes, so both can run on each new keyframe as it arrives. The
search window 11 is as accurate on two 200 s windows of sequence_4_10
(Sim(3) RMSE 0.352 m and 0.516 m, against 0.389 m and 0.486 m with h = 7
and window 21; window 7: 0.835 m).

### The real-time chain on all ten sequences

Basalt's `euroc_config` (in low light `euroc_config_lowlight`, the same with
the 0.5 px KLT threshold), fed from the remote zip as the frames arrive;
matching and revisits (`loops --step 4`) guided by it, the batch features,
then `online`, and by day the batch pass after the recording. In brackets
the same steps from the big-window VIO; the last column is the default
VIO's CPU seconds per second of recording:

| Sequence | VIO | Online | Online + pass | VIO cores |
|---|---:|---:|---:|---:|
| sequence_1_19 | 63.44 (71.96) | 78.20 (75.73) | 79.78 (79.20) | 1.30 |
| sequence_1_20 | 33.55 (38.32) | 56.92 (65.37) | 69.13 (73.03) | 1.34 |
| sequence_2_11 | 29.67 (49.68) | 72.48 (83.12) | 86.50 (85.55) | 1.47 |
| sequence_2_12 | 31.45 (18.03) | **56.41** (19.87) | 47.45 (16.57) | 1.43 |
| sequence_3_17 | 11.20 (50.06) | 37.58 (70.33) | 43.55 (75.57) | 1.57 |
| sequence_3_18 | 19.94 (34.68) | 43.61 (57.87) | 51.33 (67.39) | 1.41 |
| sequence_4_10 | 35.66 (38.99) | 63.58 (69.79) | | 1.49 |
| sequence_4_11 | 41.95 (36.37) | 53.58 (61.03) | | 1.48 |
| sequence_5_11 | 38.55 (43.14) | 52.68 (55.95) | 50.97 (61.16) | 1.37 |
| sequence_5_12 | 10.76 (4.98) | 6.89 (5.17) | 7.54 (7.66) | 2.20 |
| **Mean** | 31.62 (38.62) | **52.19** (56.42) | 55.34 (59.70)¹ | |

¹ Low light: the online result.

On 4 cores and causal, the chain scores 52.19 on average, against 54.34
for the batch pipeline from the big-window VIO after the recording. The
default VIO keeps its heading better on sequence_2_12 (within −1.4° …
+1.1° per 120 s window for the first 22 minutes, against −4.1° in the first
window for the big window), so there the online result is the best of all (56.41,
pose recall 38.8 %) and the batch pass loses 9 points of it; on
sequence_3_17 the default VIO fails (11.20). On the tram of sequence_5_12
it needs 2.2 cores.

Averaging the two chains' results, the second turned onto the first by a
rigid fit, positions averaged and rotations met halfway, beats both where
they agree: sequence_2_11 91.52, sequence_1_19 81.38, sequence_1_20 73.73
(rigid fit residual 0.6–0.9 m). Where one fails the average lands between
them (sequence_2_12 27.77, sequence_3_17 63.36, sequence_3_18 57.23,
sequence_5_11 58.99; residual 2.2–5.2 m); in low light the averaged online
results stay slightly below the big window's (69.43 and 59.26; 1.2–1.3 m).
Averaging below a residual of 2 m and keeping the big-window result above
it gives 60.37 against 59.70, with the 2 m chosen on these sequences. It
needs both chains, twice the VIO and the solves, so it is not part of the
pipeline.

### Matching guided by a drifting VIO

On sequence_3_17 the default VIO's heading drifts steadily, +8.8° … −8.2°
per 120 s window against the pseudo-GT (0.16 mrad/s), and the chain scores
37.58 online. The matches cause it, not the VIO the windows start from:
with the big-window chain's matches, the default VIO's start and its own
revisits, `online` scores 72.00. With the default VIO's matches the windows
end at the same heading however they start: rotations of the new frames
from the gyroscope instead of the VIO (38.77), `--modes yaw` (40.02), 20
iterations per window (42.07) and revisits at candidate stride 2 (37.66)
barely move it. Both chains have as many matches per keyframe offset, but
those only the default VIO's guidance finds lie further off the pseudo-GT's
epipolar geometry (offsets 192–256: median 4.1 px against 3.2 px): the band
(2 px + 0.1 px per keyframe offset, at most 6 px) lets a drifting guide pick
the matches that agree with its drift (at those offsets nearly all of them
are wrong: see [Matches far apart are
wrong](#matches-far-apart-are-wrong)). A wider band, `match --tau-slope 0.3
--tau-max 20`, on the online result of the chain:

| Band | sequence_3_17 | sequence_2_12 | Mean of ten |
|---|---:|---:|---:|
| 2 px + 0.1 px per offset, at most 6 px (the default) | 37.58 | **56.41** | 52.19 |
| + 0.2 px per offset, at most 12 px | 60.64 | | |
| + 0.1 px per offset, at most 20 px | 66.37 | 53.22 | 57.60 |
| + 0.3 px per offset, at most 20 px | **74.95** | 37.75 | **60.54** |

Matching each new keyframe guided by the windows' trajectory as it stands
instead of the VIO raises the default band's sequence_3_17 to 51.46 and
sequence_2_11 to 77.01, but with the wide band it does worse than the VIO
(68.98, and 33.65 on sequence_2_12). An essential matrix fitted to each
pair's wide-band matches by RANSAC keeps fewer good ones (71.28 and 31.11)
and makes the matching five times slower.

The chain with the wide band, online and with the batch pass after the
recording (by day), against the default band:

| Sequence | Online | Online + pass |
|---|---:|---:|
| sequence_1_19 | 76.75 (78.20) | 79.62 (79.78) |
| sequence_1_20 | **81.53** (56.92) | 81.71 (69.13) |
| sequence_2_11 | 79.97 (72.48) | **89.84** (86.50) |
| sequence_2_12 | 37.75 (56.41) | 27.15 (47.45) |
| sequence_3_17 | **74.95** (37.58) | 74.04 (43.55) |
| sequence_3_18 | 54.40 (43.61) | 59.16 (51.33) |
| sequence_4_10 | 64.18 (63.58) | |
| sequence_4_11 | **71.07** (53.58) | |
| sequence_5_11 | 57.56 (52.68) | 57.53 (50.97) |
| sequence_5_12 | 7.24 (6.89) | 7.57 (7.54) |
| **Mean** | **60.54** (52.19) | **61.19** (55.34)¹ |

¹ Low light: the online result.

Causal and within 4 cores, the chain now beats the big-window pipeline run
after the recording (59.70); so do sequence_1_20 (81.53, pose recall
99.3 %, against 73.03) and sequence_4_11 (71.07 against 61.03), and
sequence_3_17 comes within 0.6 of it (74.95 against 75.57). Only
sequence_2_12 loses: there the default VIO's heading was already good, and
the wide band's extra matches turn its first two minutes 3.3° off (1.3°
with the default band). The matching takes about 0.27 cores instead of
0.22.

### Matches far apart are wrong

The matches between keyframes 96 to 256 apart are nearly all wrong. Their
signed distance from the pseudo-GT's epipolar lines (image b; 600 image
pairs per range of offsets) has a sharp peak at zero up to 48 keyframes
apart, a trace of one at 64 and 96, and none at 192–256, where the matches
spread evenly over the band.
Density in % per px on sequence_2_12, wide band:

| Offset (keyframes) | −20 … −8 px | −2 … −1 | −1 … 0 | 0 … 1 | 1 … 2 | 8 … 20 px |
|---|---:|---:|---:|---:|---:|---:|
| 1–4 | 0.2 | 9.1 | 27.7 | 31.4 | 11.9 | 0.2–0.3 |
| 24–48 | 1.0–1.2 | 7.6 | 13.9 | 17.3 | 9.8 | 0.9–1.3 |
| 64 | 1.7–2.2 | 3.5 | 5.2 | 4.1 | 3.6 | 2.3–2.8 |
| 96 | 2.0–2.5 | 3.1 | 3.8 | 4.0 | 2.7 | 2.4–2.6 |
| 192–256 | 2.4–2.7 | 2.9 | 2.0 | 2.4 | 2.3 | 2.1–2.8 |

With the default band they fill its ±6 px the same way, on the real-time
chain's matches of sequence_2_12 and 3_17 and the big-window chain's of
sequence_1_19, 2_11, 3_18 and 4_10 (a small peak at 96–128 on
sequence_2_11 and 4_10 only). Their SIFT distances are those of the right
ones (median 254 against 254 at 192–256 on sequence_2_12), so neither the
descriptor nor the band tells them apart. Within a window they mostly merge
tracks: the triangulation against the trajectory so far (4 px) decides
which observations of a merged track stay, and `--band` drops those more
than 64 keyframes after its first. What the band changed on sequence_3_17
was these matches, not the ones nearer in time. The real-time chain matched
only up to 64 keyframes apart (`--offsets 1,2,3,4,6,8,12,16,24,32,48,64`),
online:

| Sequence | Default band, to 256 | Wide band, to 256 | Default band, to 64 | Wide band, to 64 |
|---|---:|---:|---:|---:|
| sequence_1_19 | **78.20** | 76.75 | 76.07 | 73.28 |
| sequence_1_20 | 56.92 | 81.53 | 61.58 | **85.71** |
| sequence_2_11 | 72.48 | **79.97** | 76.99 | 79.85 |
| sequence_2_12 | **56.41** | 37.75 | 23.18 | 10.58 |
| sequence_3_17 | 37.58 | 74.95 | **76.82** | 76.62 |
| sequence_3_18 | 43.61 | 54.40 | 61.32 | **64.70** |
| sequence_4_10 | 63.58 | **64.18** | 62.20 | 54.63 |
| sequence_4_11 | 53.58 | **71.07** | 50.97 | 55.45 |
| sequence_5_11 | 52.68 | 57.56 | 54.45 | **58.73** |
| sequence_5_12 | 6.89 | 7.24 | **8.43** | 8.07 |
| **Mean** | 52.19 | **60.54** | 55.20 | 56.76 |

Without the far matches the windows hold the heading on their own: per
120 s window against the pseudo-GT, within ±0.55° on sequence_1_19, 1_20,
2_11, 3_17 and 3_18 (wide band), where the VIO's error reaches 2.3°, 5.7°,
5.2°, 8.8° and 13.0°. Where they lose is the start of sequence_2_12
(−7.05° in the first two minutes, 9.4° of turn over the first 22 minutes)
and of sequence_4_11 (−3.47°), and low light on sequence_4_10 (−1.2° …
+2.2°). There the far matches pull the solve back: on sequence_2_12 to the
VIO's heading, which stays within ±1.1° for 20 minutes (the default band
−1.32° … +0.84°, the wide band −3.33° … +0.89°); on sequence_4_11 the wide
band starts at −0.67°, better than the VIO's −3.29°. On the mean the wide
band with the far offsets stays ahead, and stays the chain's.

Two ways to get the pull without the wrong matches do not. A looser gyro
bias random walk online (wide band to 64) mends the starts (sequence_2_12
−2.66° in the first two minutes, sequence_4_11 −0.58°) but turns the others
off (sequence_3_18 +1.24° against +0.23°). A prior that keeps the rotation
change over every 64 keyframes within 0.3° or 1° of the VIO's changes
nothing: over 32 s the solve's own matches fix the rotation far more
tightly.

| Sequence | Wide band, to 64 | Gyro bias random walk ×10 | ×100 | VIO rotation prior 1° | 0.3° |
|---|---:|---:|---:|---:|---:|
| sequence_2_12 | 10.58 | 30.17 | 30.05 | 10.67 | 12.48 |
| sequence_4_11 | 55.45 | 65.71 | **65.94** | 55.28 | 53.74 |
| sequence_3_18 | **64.70** | 58.58 | 56.83 | 64.67 | 63.08 |
| sequence_1_20 | 85.71 | 78.98 | 73.71 | **85.83** | 84.63 |
| sequence_3_17 | 76.62 | | | 77.16 | **80.10** |

### The first minutes, online

The gyroscope disagrees with the solved rotations in the first minutes of
every recording while the device warms up ([The first minutes of a
recording](#the-first-minutes-of-a-recording)), and the windows' random walk
(1e-7) keeps their bias constant there. Online, sequence_2_12 starts 3.33°
off the pseudo-GT's heading and comes to +0.89° after 22 minutes. Ten times
looser everywhere, its first two minutes are −1.27° off, but the heading
wanders later in low light (sequence_4_10: 3.6° of spread per 120 s window
against 2.9°). `--gyro-rw-warmup 10 --gyro-rw-warmup-s 300` loosens it at
the start only: by 1 + 9 e^(−t / 300 s) at t seconds into the recording
(10 at the start, 4.3 after five minutes, 2.2 after ten). sequence_2_12's
heading then spans −1.33° … +0.56° online (−3.33° … +0.89° without) and
−1.73° … +1.23° after the pass (−3.45° … +1.91°). The wide-band chain,
online and with the batch pass after the recording, both solves of the pass
with the same warm-up:

| Sequence | Online | Random walk ×10 | Warm-up | Online + pass | Warm-up, + pass | Warm-up, + pass with ×10 |
|---|---:|---:|---:|---:|---:|---:|
| sequence_1_19 | 76.75 | **78.16** | 77.52 | 79.62 | 79.48 | **81.35** |
| sequence_1_20 | 81.53 | 81.01 | **81.79** | 81.71 | 81.71 | **86.20** |
| sequence_2_11 | **79.97** | 76.57 | 78.39 | 89.84 | **90.13** | 87.65 |
| sequence_2_12 | 37.75 | 51.51 | **56.00** | 27.15 | 43.07 | **51.96** |
| sequence_3_17 | **74.95** | 70.91 | 72.63 | 74.04 | 76.37 | **80.47** |
| sequence_3_18 | **54.40** | 53.11 | 53.93 | **59.16** | 58.94 | 54.32 |
| sequence_4_10 | **64.18** | 52.68 | 61.09 | | | |
| sequence_4_11 | 71.07 | **71.43** | 68.47 | | | |
| sequence_5_11 | **57.56** | 56.80 | 56.98 | 57.53 | 58.02 | **58.88** |
| sequence_5_12 | **7.24** | 7.12 | 7.15 | **7.57** | 7.32 | 7.41 |
| **Mean** | 60.54 | 59.93 | **61.40** | 61.19¹ | 63.03² | **64.35**² |

¹ Low light: the online result. ² Low light: the online result without the
warm-up, which costs both low-light sequences (−3.1 and −2.6; with it the
mean is 62.46). In low light the matches pin a moving bias less well, as
with the looser random walk throughout.

By track, with the leaderboard's GroundedSLAM (test set, 2026-10-17):

| Track | Online + pass | Warm-up, + pass | Warm-up, + pass with ×10 | GroundedSLAM |
|---|---:|---:|---:|---:|
| Short | 80.67 | 80.60 | **83.78** | 80.2 |
| Medium | 58.50 | 66.60 | **69.81** | 61.6 |
| Long | 66.60 | **67.66** | 67.40 | 59.9 |
| Low light (online, no warm-up) | 67.63 | 67.63 | 67.63 | **67.7** |
| Moving platform | 32.55 | 32.67 | 33.15 | **41.9** |
| **Mean** | 61.19 | 63.03 | **64.35** | 62.3 |

The pass with the warm-up alone still costs sequence_2_12 13 points (56.00
online): its solves keep one bias for the whole 28 minutes after the
warm-up, where each online window has its own. With the random walk ten
times looser in the pass (`--gyro-rw-scale 10` in both solves, on top of the
warm-up) it loses 4 (51.96), and six of the eight day sequences gain
(sequence_1_20 +4.5, 3_17 +4.1; 2_11 −2.5 and 3_18 −4.6).

In low light the pass still loses to the online result. From the wide-band
online results (no warm-up), sequence_4_10 and 4_11 score 54.40 and 62.27
after the pass, and 66.27 and 67.27 with its random walk ten times looser,
against 64.18 and 71.07 online (mean 66.77 against 67.63), so low light
keeps the online result. [`scripts/lamaria_ba_chain.py`](../scripts/lamaria_ba_chain.py)
runs these steps per track and reproduces the scratch runs: sequence_1_20
online 81.79, after the pass 86.20.

### Matching again after the pass

The Short test sequences furthest behind GroundedSLAM are fast walks (1.1–1.4
m/s) along open streets, 270–670 m across, where the sequences close to it
mostly wander within 160–280 m at 0.75–1.16 m/s
([lamaria_test_results.md](lamaria_test_results.md#against-groundedslam-sequence-by-sequence)).
Training R_12_10cp (1WM093701G1276, 17 minutes, 604 m across, 1.18 m/s, wide
streets and distant buildings) is one of them. Through the test pipeline:

| Stage | CP Score | CP recall at 1 m | Pose recall at 5 m |
|---|---:|---:|---:|
| VIO | 22.85 | 0 % | 43.8 % |
| Online | 60.75 | 40 % | 100 % |
| Loop-corrected after the pass's first solve | 62.82 | 50 % | 100 % |
| The pass with the revisits (submitted) | 61.09 | 40 % | 100 % |

The VIO's heading drifts linearly, +13.1° over 1,011 s (0.78°/min; the VIO
of sequence_1_20, 2_12, 3_18 and 4_11 drifts 0.14–0.32°/min), since the
windows do not settle the gyroscope bias. The pass's bias, [3.3, 20.6,
4.2]·10⁻⁴ rad/s, matches one fitted against the pseudo-GT ([3.6, 20.9,
3.8]·10⁻⁴ over 120 s windows on average), and the chain leaves 1.5° of the
heading drift. What is left are slow swings of the local scale (0.979–1.026
in 120 s windows) and of the heading (−1.2° … +0.7°); within a window the
error is 0.18 m. Corrected per 60 s
window from the pseudo-GT (diagnosis only), the scale alone gives 68.69, the
heading alone 65.82, both 96.09.

Matching once more, guided by the pass's trajectory (the same band and
revisit search), then one more solve with the revisits, started from the
pass's result:

| Matches guided by | CP Score |
|---|---:|
| The VIO (as in the chain; the extra solve alone) | 60.48 |
| The pass's trajectory | **64.30** |
| The pseudo-GT, filled in between its 5 Hz poses with the VIO (diagnosis only) | 64.53 |

The band lets the VIO's drift choose matches that agree with it ([Matching
guided by a drifting VIO](#matching-guided-by-a-drifting-vio)); the pass's
trajectory does as well as the pseudo-GT, so better guidance gets no more
here. The accelerometer bias moves by 0.06 m/s² (z) over the first two
minutes, but loosening its random walk there like the gyroscope's (×10,
decaying over 300 s) does not help (64.08).

`lamaria_ba_chain.py --rematch` adds these steps by day (work/rematch, result
ba3.tum), and the submission driver runs them by default. On the eight day
training sequences, one GitHub Actions run (the pass, then matched again
from the same VIO):

| Sequence | The pass | Matched again |
|---|---:|---:|
| sequence_1_19 | 81.75 | 82.25 |
| sequence_1_20 | 86.21 | 86.98 |
| sequence_2_11 | 88.84 | 89.02 |
| sequence_2_12 | 51.96 | 51.97 |
| sequence_3_17 | 79.88 | 80.03 |
| sequence_3_18 | 54.33 | 54.28 |
| sequence_5_11 | 66.27 | 66.77 |
| sequence_5_12 | 6.77 | 6.76 |
| **Mean** | 64.50 | **64.76** |

None loses more than 0.05; the gain is large only where the VIO's heading
drifts fast, as on R_12_10cp. sequence_2_12, whose heading goes wrong in its
first minutes, does not move.

### Learned features

SIFT rarely finds a landmark again as the view changes (see [Where the second
sequences' heading drift comes from](#where-the-second-sequences-heading-drift-comes-from)).
`features --extractor aliked|disk|superpoint` replaces it with one of
[LightGlue](https://github.com/cvg/LightGlue)'s learned extractors on the GPU,
and `match` and `loops` then match with LightGlue instead of the ratio test.
The VIO's epipolar band stays as the filter of LightGlue's matches, and `loops`
keeps its essential-matrix check. The learned keypoints have no size, so their
observations keep `--pixel-sigma`. (SuperPoint's weights are licensed for
non-commercial use only; ALIKED's and DISK's are not restricted.)

Each extractor keeps its 1,024 best keypoints per image (`--num-features 1024
--detection-threshold 0`). With the same number in every image, LightGlue
matches 16 image pairs per call: one pair per call takes about 27 ms on an 8 GB
GPU whatever the number of keypoints, 16 pairs of 1,024 keypoints 9 ms per pair
in half precision (98.8 % of the matches of LightGlue one pair at a time).
sequence_3_18's 97,000–103,000 image pairs take 15–16 min, against 73 s for
SIFT's ratio test on eight cores.

[`scripts/lamaria_track_stats.py`](../scripts/lamaria_track_stats.py) counts
how often a solve's landmarks are found again. It projects each landmark into
both cameras of every keyframe within 32 of its observed span; of the keyframes
where it lands inside an image, it counts those that observe it. For SIFT on
sequence_3_18 that is 7.8 % (the 8.9 % above came from a smaller count, 564,507
keyframes against 17.2 million). Each variant below runs the pass's first solve
(`ba1`) from the chain's online result, with the chain's options and its own
features and VIO-guided matches. (On this machine the chain itself ends at
54.27 on sequence_3_18, 64.61 on R_12_10cp and 88.65 on sequence_1_20, against
54.28, 64.30 and 86.98 on GitHub Actions.)

| sequence_3_18 | SIFT | ALIKED | DISK | SuperPoint |
|---|---:|---:|---:|---:|
| Keyframes that see a landmark and observe it | 7.8 % | 7.8 % | 7.8 % | 7.5 % |
| Keyframes per landmark, mean (99th percentile) | 3.07 (14) | 2.83 (10) | 2.96 (11) | 2.72 (10) |
| Landmarks | 439,618 | 387,907 | 467,008 | 343,315 |
| Tracks dropped with two features in one image | 12,095 | 54,182 | 35,068 | 36,811 |
| Not observed, a keypoint within 3 px | 23.8 % | 12.9 % | 13.2 % | 15.3 % |
| Cost per observation | 0.14 | 0.32 | 0.37 | 0.45 |
| CP Score | **54.35** | 49.02 | 48.75 | 49.22 |
| Pose recall at 1 m | **41.2 %** | 32.0 % | 29.3 % | 35.8 % |

| R_12_10cp | SIFT | ALIKED | DISK | SuperPoint |
|---|---:|---:|---:|---:|
| Keyframes that see a landmark and observe it | 7.0 % | 7.1 % | 7.1 % | 6.7 % |
| Keyframes per landmark, mean (99th percentile) | 3.03 (14) | 2.91 (11) | 3.08 (12) | 2.67 (9) |
| Landmarks | 179,017 | 181,794 | 223,316 | 158,822 |
| Tracks dropped with two features in one image | 6,540 | 24,726 | 17,692 | 17,928 |
| Not observed, a keypoint within 3 px | 25.1 % | 14.0 % | 11.8 % | 16.2 % |
| Cost per observation | 0.14 | 0.30 | 0.35 | 0.45 |
| CP Score | 63.27 | **68.15** | 62.02 | 64.42 |
| Pose recall at 1 m | 51.3 % | **84.3 %** | 51.8 % | 59.6 % |

"Not observed, a keypoint within 3 px" is the share of the images that see a
landmark without observing it where a keypoint lies within 3 px of its
projection; at their densities, chance alone gives about 9 % for SIFT and 7 %
for the learned keypoints.

None of them finds the landmarks again more often, and the longest tracks get
shorter. The learned detectors fire at the same point less often than SIFT as
the view changes; their keypoints are less precise (2.1–3.3 times SIFT's cost
per observation, as on sequence_4_10's window in [Low light: features after
denoising](#low-light-features-after-denoising)); and LightGlue's matches
disagree more between image pairs, so 2.7–4.5 times as many tracks are
dropped. ALIKED on sequence_1_20 gives the same picture (7.4 % against 7.5 %,
first solve 87.53 against 88.00).

The first solve's score follows none of this: ALIKED loses 5.3 on
sequence_3_18 and gains 4.9 on R_12_10cp, where DISK loses 1.3 and SuperPoint
gains 1.2. As the tracks are no longer than SIFT's, the eight day sequences were
not run.

ALIKED's gain on R_12_10cp is spread over the whole sequence. Against the
pseudo-GT (one similarity for the whole trajectory), its first solve is 0.77 m
off on average, against 0.98 m for SIFT's. The scale and heading of each 60 s
window, fitted to the pseudo-GT after that alignment, wander a little less:
scale 0.986–1.036 against 0.977–1.038 (standard deviation 0.012 against
0.015), heading standard deviation 0.57° against 0.64°. With the error that
close to the evaluation's 1 m, the small gain moves the pose recall at 1 m from
51 % to 84 %. On sequence_3_18 it goes the other way (1.73 m against 1.47 m).

R_11_5cp, the other Short training sequence that is not a `sequence_1_*`,
gains the same way: the first solve is 0.38 m off on average against 0.57 m,
its 60 s windows' heading has a standard deviation of 0.31° against 0.74°, and
it scores 85.23 against 81.32 (the chain with SIFT ends at 85.81).

The whole chain was then run two ways, both against the chain with SIFT on
this machine:

- with ALIKED throughout (`lamaria_ba_chain.py --features aliked`), the online
  windows included;
- with SIFT online and ALIKED in the pass after it (`--pass-features aliked`).
  The pass reads ALIKED's keypoints, matches and revisits, all guided by the VIO
  like the online windows' own, and the rematch goes there too.

| Sequence | Device | SIFT | ALIKED | SIFT online, ALIKED pass |
|---|---|---:|---:|---:|
| R_11_5cp | 1WM093701G1276 | 85.81 | 92.12 | **92.20** |
| R_12_10cp | 1WM093701G1276 | 64.61 | 72.03 | **73.96** |
| sequence_1_19 | 1WM09370081264 | **82.25** | 73.34 | 81.00 |
| sequence_1_20 | 1WM09370081264 | **88.65** | 63.13 | 86.95 |
| sequence_2_11 | 1WM093701G1276 | **89.25** | | 76.56 |
| sequence_3_18 | 1WM093701G1276 | **54.27** | | 49.23 |

With ALIKED in the online windows, sequence_1_20's heading turns by 2° over its
17 minutes (+0.9° … −1.2° per 60 s window against the pseudo-GT). Its online
result is 1.36 m off on average, against 0.38 m with SIFT. With SIFT online,
the pass gains 6.4 and 9.3 on the two R sequences and loses 1.2–12.7 on the
four others. Neither variant goes into the chain's defaults, and the remaining
day sequences were not run.

The device does not explain the split: sequence_2_11 and 3_18 come from the
R sequences' device too. Nothing else measured sets the R sequences apart:

- SIFT finds as many keypoints there (1,358–1,370 per image, against
  1,221–1,432 on the four others).
- ALIKED and LightGlue find as many matches per image pair (128–130, against
  112–128).
- The first solve's cost per observation is that of sequence_2_11 and 3_18
  (0.13 with SIFT, 0.29–0.32 with ALIKED).

Only their images are a little darker: a mean grey level of 68–69, against
75–82.

The pass with ALIKED shortens every trajectory, by 0.25–0.82 % (the scale of
one similarity to the pseudo-GT). The evaluation fits a similarity to the
control points as well, so that scale costs nothing either way. What the
scores follow is the shape left after the fit: the mean error after one
similarity, and the spread of the heading over 60 s windows fitted after it.

| Sequence | Mean error, SIFT | SIFT online, ALIKED pass | Heading spread, SIFT | SIFT online, ALIKED pass |
|---|---:|---:|---:|---:|
| R_11_5cp | 0.58 m | **0.24 m** | 0.88° | **0.25°** |
| R_12_10cp | 0.89 m | **0.60 m** | 0.58° | **0.49°** |
| sequence_1_19 | **0.44 m** | 0.46 m | **0.19°** | 0.29° |
| sequence_1_20 | **0.28 m** | 0.38 m | 0.21° | 0.21° |
| sequence_2_11 | **0.30 m** | 0.59 m | **0.19°** | 0.45° |
| sequence_3_18 | **1.46 m** | 1.71 m | 1.07° | 1.07° |

(Heading spread: the standard deviation over the 60 s windows.) ALIKED
steadies the heading where SIFT's wanders, as on the two R sequences. Where
SIFT's heading already holds within 0.2°, ALIKED unsettles it, as on
sequence_2_11.

In low light the chain ends with its online windows, so ALIKED goes there too
(`--features aliked`, after the same CLAHE and non-local means as SIFT;
learned matching had not been tried in low light, see [Low light: features
after denoising](#low-light-features-after-denoising)). It loses on both
low-light training sequences, both chains on this machine from the same VIO:

| Sequence | SIFT | ALIKED | Mean error, SIFT | ALIKED | Heading spread, SIFT | ALIKED |
|---|---:|---:|---:|---:|---:|---:|
| sequence_4_10 | **59.80** | 49.25 | **1.33 m** | 1.90 m | **1.10°** | 1.96° |
| sequence_4_11 | **71.07** | 48.26 | **0.77 m** | 3.15 m | **1.11°** | 2.38° |

As on sequence_1_20 by day, the online windows with ALIKED let the heading
wander: per 60 s window it reaches +6.5° on sequence_4_10 and +4.4° on 4_11.

### Learned matches without the VIO's band

The VIO's epipolar band lets a drifting VIO pick the matches ([Matching guided
by a drifting VIO](#matching-guided-by-a-drifting-vio)). LightGlue matches
without any guidance, so the pass with ALIKED can keep its matches by other
means:

- `match --verify essential` keeps those an essential matrix explains,
  checked as a revisit is (RANSAC at 1 px, at least 20 inliers, the rotation
  within 10° of the VIO's), whatever the VIO's translation.
- `match --vio online.tum` keeps those within the online result's band instead
  of the VIO's (`lamaria_ba_chain.py --pass-match online`).

Each runs with SIFT online, ALIKED's revisits (searched near the VIO) and the
rematch guided by the pass, on this machine:

| Sequence | SIFT | The VIO's band | Essential matrix | The online result's band |
|---|---:|---:|---:|---:|
| R_11_5cp | 85.81 | 92.20 | 96.98 | **97.02** |
| R_12_10cp | 64.61 | 73.96 | 75.43 | **79.31** |
| sequence_1_19 | **82.25** | 81.00 | 68.16 | 70.16 |
| sequence_1_20 | **88.65** | 86.95 | 86.15 | 80.78 |
| sequence_2_11 | **89.25** | 76.56 | 83.14 | 82.43 |
| sequence_3_18 | 54.27 | 49.23 | **62.12** | 53.92 |
| **Mean** | 77.47 | 76.65 | **78.66** | 77.27 |
| The two R sequences | 75.21 | 83.08 | 86.21 | **88.17** |
| The four others | **78.61** | 73.44 | 74.89 | 71.82 |

None of the variants beats SIFT outside the R sequences, and none of them
keeps its place across all six sequences. sequence_3_18 gains 7.8 with the
essential matrix, and sequence_1_19 loses 14.1 with it.

- **The essential matrix and the far pairs.** It keeps few pairs far apart. On
  sequence_1_19, at 128–256 keyframes apart, it keeps 3–142 image pairs per
  offset, against about 2,000 in the band. The few matches of such a pair rarely
  reach 20 inliers. Without them the heading of sequence_1_19's first solve
  spreads 0.67° over the 60 s windows, against 0.27° in the VIO's band. The
  online result's band keeps the far pairs and spreads 0.65° all the same.
- **The revisits.** On sequence_2_11 the first solve with the essential matrix
  scores 88.41, against 77.04 for SIFT's. The solve with the revisits then
  loses 5.4, where SIFT's gains 12.0. With ALIKED, 40,553 tracks are dropped for
  two features in one image (SIFT 7,669), and the revisits leave 7,857
  keyframe blocks outside the band (SIFT 36,461).
  - Dropping only the conflicting images' features of such a track helps
    R_12_10cp (77.16) and loses on sequence_2_11 (78.51) and 3_18 (57.53).
  - A pass without the revisits (the rematch guided by its first solve, no
    `--loops`) loses on all three tried. With the online result's band,
    sequence_1_19 scores 68.31, 2_11 71.24 and R_12_10cp 73.57.

The online result's band does best on the two R sequences. It replaced the
VIO's for the five Short test sequences of device 1WM093701G1276 in
submission 6, and four of them lost: 60.0 on average, against 61.5 with the
VIO's band ([lamaria_test_results.md](lamaria_test_results.md)).

### Revisits for place recognition

`loops` looks for revisits near the trajectory that guides it (the VIO, and
the pass's `ba2` when matching again): keyframes more than 64 apart, within
15 m and with camera axes within 60° ([Revisits](#revisits)). A global image
descriptor (NetVLAD, MixVPR, SALAD) would find a revisit however far the
trajectory has drifted, but only the revisits outside that gate need one. In
the pseudo-GT, at the keyframes' 2 Hz, a keyframe is on a revisit when it lies
within 15 m and 60° of heading of another after at least 50 m walked in
between (so standing at a crossing does not count):

| Sequence | Minutes | Walked | Keyframes on a revisit | Longest time between the visits |
|---|---:|---:|---:|---:|
| R_11_5cp | 7.9 | 459 m | 23.2 % | 136 s |
| R_12_10cp | 16.9 | 984 m | 0 % | |
| sequence_1_19 | 15.3 | 975 m | 0.2 % | 54 s |
| sequence_1_20 | 16.9 | 1,117 m | 15.0 % | 131 s |
| sequence_2_11 | 19.8 | 1,261 m | 13.2 % | 221 s |
| sequence_2_12 | 27.9 | 2,033 m | 15.9 % | 211 s |
| sequence_3_17 | 29.8 | 1,946 m | 4.4 % | 66 s |
| sequence_3_18 | 28.2 | 1,651 m | 3.7 % | 70 s |
| sequence_4_10 | 23.6 | 1,410 m | 10.5 % | 76 s |
| sequence_4_11 | 18.8 | 1,062 m | 2.0 % | 74 s |

Every revisit comes within four minutes of the first visit, too soon for the
VIO to drift out of the gate. On sequence_1_20 and 3_18 (the chain's first
`loops`, guided by the VIO), each revisit of the pseudo-GT (at least 10
keyframe pairs) has 71–100 % of its keyframe pairs within 15 m in the VIO as
well. `loops` verifies pairs on 7 of sequence_1_20's 9 revisits (93 % of their
keyframe pairs) and on 2 of sequence_3_18's 3 (96 %). The ones it misses are
13–14 m apart in the VIO at the median, inside the gate: their images fail the
essential-matrix check, so a global descriptor would propose them again to no
effect. The sequences furthest
behind, R_12_10cp and 3_18, barely revisit at all, and sequence_2_12's
revisits come after its first ten minutes, too late for the heading that goes
wrong at its start. Place recognition was not tried.

## Usage

[`scripts/lamaria_ba_chain.py`](../scripts/lamaria_ba_chain.py) runs the real-time chain behind a VIO
trajectory, with the options of each track, and resumes where an earlier run stopped:

```bash
S=sequence_2_11
python scripts/lamaria_ba_chain.py --mav0 data/$S/mav0 --vio vio/$S.tum \
  --imu data/$S/mav0/imu0/data.csv --calib calib/$S.json --track 2 --work work/$S --out out/$S.tum
```

`scripts/run_lamaria_test_submission.py --ba-chain` runs it on every downloaded sequence (the VIO with
`euroc_config`, `euroc_config_lowlight` in low light) and packages its trajectories. The steps one by one:

```bash
pip install numpy scipy opencv-python gtsam numba
S=sequence_1_19
python scripts/lamaria_batch_ba.py features --mav0 data/$S/mav0 --vio vio/$S.tum --work work/$S
python scripts/lamaria_batch_ba.py match --work work/$S --vio vio/$S.tum --calib calib/$S.json
python scripts/lamaria_batch_ba.py solve --work work/$S --vio vio/$S.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.tum
```

With the revisits:

```bash
python scripts/lamaria_batch_ba.py loops --work work/$S --vio vio/$S.tum --calib calib/$S.json
python scripts/lamaria_batch_ba.py solve --work work/$S --vio vio/$S.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.ba1.tum --save-states
python scripts/lamaria_batch_ba.py loop-correct --work work/$S --calib calib/$S.json \
  --states out/$S.ba1_states.npz --tum out/$S.ba1.tum --out out/$S.lc.tum
python scripts/lamaria_batch_ba.py solve --work work/$S --vio out/$S.lc.tum --calib calib/$S.json \
  --imu data/$S/mav0/imu0/data.csv --out out/$S.tum --loops
```

- **IMU.** `--imu` is the IMU as the VIO read it, with the factory
  rectification and the per-device time alignment.
- **Calibration.** `--calib` is the Basalt calibration of the ASL pinhole
  images, with the factory camera rig: the BA estimates the camera rotations
  itself.
- **Output.** `solve` writes the corrected trajectory and a JSON summary
  next to it: the camera rotations, the biases, the costs and the whitened
  IMU residual per factor.
- **Robustness.** `--outlier-px` (10) and `--lm-damp-floor` (0.01) are the
  outlier rejection and the landmark damping floor; 0 turns them off.
  `--outlier-px2` (3) is the tighter threshold from iteration
  `--outlier-start2` (8) on.
- **Camera rotations.** `--ext-segments` (8) is the number of time segments
  with their own IMU-camera rotations, `--ext-seg-sigma` (0.05°) the prior
  between consecutive ones; `--ext-segments 1` keeps one rotation per camera.
  `--est-extrinsic` (01) picks the cameras whose rotations are estimated.
- **Revisits.** `solve --save-states` keeps what `loop-correct` needs.
  `loop-correct --gate BASE,RATE,MAX_ROT` (0.3 m, 0.01 m/s, 2°) sets the
  drift a loop edge may disagree by.
- **Other options.** `--prior-rot-sigma` loosens the first keyframe's
  rotation prior (1e-4 rad). `--omega-coriolis x,y,z` puts Earth's rotation
  (rad/s, in the VIO's world frame) into the IMU factors. `--imu-noise-scale`,
  `--gyro-rw-scale` and `--accel-rw-scale` multiply the calibration's noise
  densities and bias random walks. `--kf-range START:END` solves a window of
  keyframes only.
- **Online.** `online` takes the solve's options plus `--window` (480
  keyframes) and `--step` (60): every 60 keyframes it solves the latest 480
  (10 iterations and two rotation segments per window by default), each
  window starting from the biases, velocities, camera rotations and focal
  lengths of the one before, and writes the stitched trajectory to `--out`
  (see [Online mode](#online-mode)). By day (not in low light), the batch
  steps with the revisits from that trajectory, `solve --vio online.tum
  --save-states`, `loop-correct` and `solve --loops`, improve it after the
  recording (see [Online, then one batch pass](#online-then-one-batch-pass)).
  From Basalt's default VIO (`euroc_config`, the real-time chain), match
  with `--tau-slope 0.3 --tau-max 20` (see [Matching guided by a drifting
  VIO](#matching-guided-by-a-drifting-vio)); by day, `online` and both solves
  of the pass with `--gyro-rw-warmup 10 --gyro-rw-warmup-s 300`, the pass
  also with `--gyro-rw-scale 10` (see [The first minutes,
  online](#the-first-minutes-online)).
  `BASALT_EUROC_WAIT_S=120` lets the VIO demo wait for frames that a
  streaming reader has not written yet.
- **Heading drift.** `--modes yaw` lets the solve correct a slow heading
  drift of its starting trajectory (see [Slow heading drift](#slow-heading-drift));
  `--modes yawN` splits it into N time pieces.
- **Low light.** `features --clahe` equalizes the contrast before SIFT. On
  sequence_4_10 frames it raises the detections from 130–300 to 1,100–1,500
  per image. `--denoise 7` runs non-local means first (see [Low
  light](#low-light-features-after-denoising)); `--denoise 11
  --denoise-window 11` takes 40 % less time (see [Real time](#real-time)).
- **Focal lengths.** `solve --est-focal` estimates each camera's focal
  length per time segment (`--focal-seg-sigma`, `--focal-sigma`,
  `--focal-mean-sigma`); leave it off on a moving platform. `loop-correct`
  takes the estimates from the states.
- **Keypoint sizes.** `solve --sigma-size 2.7 --sigma-size-pow 0.35` gives
  larger keypoints a larger pixel sigma (`features` stores the sizes).
- **Learned features.** `features --extractor aliked|disk|superpoint
  --num-features 1024 --detection-threshold 0` needs torch and LightGlue (tried
  with cvg/LightGlue 5a650c4 and torch 2.14.1 on CUDA 13.0); `match` and `loops`
  then match with LightGlue (`--lg-batch` pairs per call, `--lg-threshold`,
  `--no-lg-mp` for single precision; `--matcher ratio` for the ratio test, whose
  `--max-desc` is then unlimited). `lamaria_ba_chain.py --features` passes the
  extractor to the features step (1,024 keypoints at detection threshold 0 by
  default), and `--pass-features` gives the pass by day an extractor of its own
  in `work/<extractor>`. They do not find the landmarks again more often (see
  [Learned features](#learned-features)).
- **Learned matches without the VIO's band.** `match --verify essential`
  keeps LightGlue's matches that an essential matrix explains (`--min-inliers`
  20, `--max-rot` 10°, fitted on `--jobs` threads) instead of those within the
  band. `lamaria_ba_chain.py --pass-match online|essential` picks the pass's
  matching: the online result's band, or the essential matrix (default `vio`,
  the VIO's band). See [Learned matches without the VIO's
  band](#learned-matches-without-the-vios-band).
- **Track statistics.** `scripts/lamaria_track_stats.py --states
  ba1_states.npz --calib CALIB --work WORK` prints how often a solve's landmarks
  are observed where they are visible, and the track lengths.
- **Recommended.** The [table above](#with-the-focal-lengths-the-keypoint-sizes-and-denoising)
  uses `--est-focal --sigma-size 2.7 --sigma-size-pow 0.35` in both solves,
  `features --clahe --denoise 7` in low light, and no `--est-focal` on the
  moving platform. In low light, both solves also take `--iterations 30
  --rel-tol 1e-5` (see [Iterations](#iterations)).
- **Cost.** On this 4-core container, a 15-minute sequence takes about
  15 min for the features (2 processes; non-local means adds about as much)
  and 2–7 min for the matching (with numba). The solve uses
  `NUMBA_NUM_THREADS` threads for the reprojection terms and the BLAS
  threads for the banded factor: on the 20-minute sequence_2_11 the first
  solve takes 269 s on one thread and 181 s on four (the same result), the
  second solve with the revisits 483 s on four, at 30 iterations; the
  default 60 take up to twice as long. `loops` takes 3–6 min with
  four processes and `loop-correct` a minute. sequence_3_17's solves need
  about 6.5 GB of memory.
- **Removed options.** The options of some experiments above that did not
  help are gone: the keypoint position correction, the focal length trend,
  the keypoint size ratio filter, an earlier warm-up random walk of the batch
  solve (`--gyro-rw-warmup` now does it, online), and the `sizes` stage for
  features written without sizes. Commit `adfe888` still has them.
