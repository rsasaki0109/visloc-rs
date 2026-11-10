# LaMAria: where the remaining error comes from

This page analyses the per-control-point (CP) errors that the current
submission setup leaves on the training sequences. The setup is the rectified
IMU, `gyro_bias_std` 1e-7, the multicam big-window config, the online mapper
and the 5° loop gate ([`lamaria_online_mapper.md`](lamaria_online_mapper.md)).
It started from the Moving-platform track, which had the largest gap to the
leaderboard (36.19 on sequence_5_11), and ended at a cause shared by all
tracks: the factory IMU-camera calibration, whose time and rotation errors
drive the VIO's heading drift
([IMU-camera calibration](#imu-camera-calibration)).

The Score uses the horizontal (2D) error of each CP after the evaluator's
robust Sim(3) alignment. Points fall from 20 below 5 cm to 0 at 10 m
(`lamaria/utils/metrics.py`). A CP at 1 m earns 12 points, at 2 m 8, and at
4 m about 5.

## Moving platform: sequence_5_11

### The CP errors

| Time | CP | Horizontal error | Points |
|---:|---|---:|---:|
| 12 s | PB2 | 20.81 m | 0.0 |
| 264 s | AA2918 | 3.81 m | 5.6 |
| 323 s | AA2846 | 2.59 m | 7.2 |
| 389 s | AA2976 | 2.86 m | 6.9 |
| 469 s | AA2917 | 0.59 m | 14.4 |
| 515 s | AA2916 | 1.32 m | 10.7 |
| 573 s | AA3739 | 2.98 m | 6.7 |
| 626 s | AA1708 | 3.40 m | 6.1 |
| 753 s | AA1726 | 4.20 m | 5.1 |
| 836 s | AA1143 | 2.67 m | 7.1 |
| 919 s | AA1144 | 2.16 m | 7.8 |
| 966 s | AA1133 | 1.04 m | 11.9 |
| 1052 s | AA4030 | 3.47 m | 6.0 |
| 1100 s | AA4031 | 3.61 m | 5.9 |

Five of the 14 CPs carry no height (their vertical variance is 1e18). Their
3D errors of 21–31 m are vertical only and do not count.

### The platform: a funicular ride

PB2 is the only CP before a ride on the Polybahn funicular, from 115 s to
190 s. The cameras see the cabin interior, the passengers and, through the
windows, the slope outside. The VIO estimates the ride at 133 m horizontally
and 30 m up. Shifting the part before the ride until PB2 fits gives 157 m and
38.6 m. The VIO therefore underestimates the ride by 15–22 %. Its speed
settles at 2.0 m/s after the acceleration phase, while integrating the IMU
alone over the acceleration phase reaches 2.5 m/s.

With the pre-ride part shifted onto PB2 (an upper bound for any fix of the
ride), the Score rises from 36.19 to 41.37.

The IMU cannot observe a constant cruise speed, so the cruise speed comes
from vision, which mixes the static cabin with the moving exterior. A
post-solve outlier filter (see [VIO outlier filter](#vio-outlier-filter))
does not change the ride: 133.7 m / 30.7 m at 3 px and 134.8 m / 31.0 m at
2 px.

### The other 13 CPs

The remaining CPs are 0.6–4.2 m off, and their residual vectors vary
smoothly along the route. A 2D similarity fit with a yaw drift and a scale
drift that are linear in time explains them (next section).

The ride does not cause that scale error. With the per-device refinement
and the 0.5 px KLT threshold
([`lamaria_klt_threshold.md`](lamaria_klt_threshold.md)), a VIO started at
200 s, after the ride, measures the route as short as one that rode the
funicular:

| VIO on sequence_5_11 | Full sequence | Started after the ride |
|---|---:|---:|
| Scale of the evaluator's Sim(3) alignment | 1.065 | 1.063 |
| Distance ratio of CP pairs, 264–626 s | 0.916 | 0.925 |
| Distance ratio of CP pairs, 626–1100 s | 0.948 | 0.946 |
| Fitted scale drift over the CP span | −5.1 % | −4.6 % |

The ratios compare the estimated and surveyed distances of CP pairs at most
250 s apart. The same device measures sequence_4_10 at a Sim(3) scale of
0.998, so the device calibration alone does not explain it either.

## Drift models fitted to the CP residuals

The fits take the triangulated CPs from the evaluator's output and fit them
to the surveyed positions with a 2D similarity, then add a yaw drift, then
a yaw and a scale drift, both linear in time (robust loss). Each drift adds
one parameter. PB2 is excluded on sequence_5_11.

| Sequence | Median, similarity | + yaw drift | + yaw + scale drift | Fitted yaw / scale drift over the CP span |
|---|---:|---:|---:|---|
| sequence_1_19 | 1.57 m | 0.76 m | **0.46 m** | −2.5° / −2.5 % |
| sequence_2_11 | 0.76 m | 0.90 m | 0.89 m | −1.2° / +0.4 % |
| sequence_3_17 | 1.85 m | 1.26 m | **0.92 m** | −2.2° / −2.1 % |
| sequence_4_10 | 2.47 m | 1.97 m | 2.28 m | +1.8° / +1.2 % |
| sequence_5_11 | 2.88 m | 2.52 m | **0.70 m** | −2.3° / −7.6 % |

On Short, Long and Moving, two drift parameters bring the median CP error
below 1 m, which would score about 12–15 points per CP instead of 5–8. A
slow drift in heading and scale is therefore the main error left on these
sequences, not a local failure. On sequence_5_11 the scale term dominates.
Medium is already accurate except for a few CPs, and Low light has errors
that a smooth drift does not explain.

### Scale

Sim(3) fits against the pseudo-GT in 60 s windows show the local scale
varying by ±2–5 % around the global value on every sequence with pseudo-GT.

The global scale grows with exposure:

| Sequence | Global Sim(3) scale (estimate → GT) |
|---|---:|
| sequence_1_19 | 0.960 |
| sequence_2_11 | 0.972 |
| sequence_3_17 | 0.987 |
| sequence_4_10 (low light, 14 ms exposure) | 0.919 |

Shifting the IMU by −3 ms and −6 ms on sequence_4_10 moves the scale to
0.957 and 1.015, so an IMU–camera time offset of about half the exposure
sets the global scale. Sim(3) absorbs a constant scale. The local
variations, however, do not change with the shift.

On sequence_5_11 the exposure (read from the `.vrs` image records) is
0.02 ms for most of the route after 240 s. Exposure changes therefore do not
explain its −7.6 % scale drift.

### Yaw

Against the pseudo-GT, the VIO's heading drifts in the same direction on all
four sequences:

| Sequence | Yaw at end | Rate |
|---|---:|---:|
| sequence_1_19 | −4.3° over 916 s | −0.28°/min |
| sequence_2_11 | −3.1° over 1185 s | −0.15°/min |
| sequence_3_17 | −6.9° over 1790 s | −0.23°/min |
| sequence_4_10 | −1.4° over 1418 s | −0.06°/min |

The mean, −0.18°/min, equals the vertical component of Earth's rotation at
Zurich (0.184°/min) in size and sign. Compensating it (see below) changes
sequence_5_11's VIO only slightly: Score 32.27 → 32.98, and the fitted yaw
drift goes from −3.18° to −2.87° instead of the 2.6° reduction the
hypothesis predicts. The body-frame projection of a vertical rate depends
only on roll and pitch. With the wearer's head roughly upright, the gyro bias
estimate absorbs most of it. The heading drift comes from the IMU-camera
calibration instead (next section).

## IMU-camera calibration

The VIO's heading drift is very sensitive to the IMU-camera time offset and
rotation. On sequence_2_11, shifting the whole IMU by −4.1, 0 and +4.1 ms
gives a yaw drift of −8.1°, −2.6° and +1.8° after 20 minutes
(`gyro_bias_std` 1e-6). The factory calibration in the `.vrs` header gets
both slightly wrong.

### Measuring both against the pseudo-GT

The rectified gyroscope is integrated between consecutive pseudo-GT
keyframes (about 150 ms apart), shifted by a trial time offset, and compared
with the pseudo-GT relative rotation. A constant gyro bias and the gyro-to-
camera rotation are fitted at each offset. The pseudo-GT poses are in the
camera-slam-left frame, whose factory orientation relative to the IMU is
exactly the one in the pinhole calibration.

| Sequence | Device | Best gyro delay | Residual (median per pair) | Rig rotation correction | Angle |
|---|---|---:|---:|---|---:|
| sequence_2_11 | 1WM093701G1276 | 4.75 ms | 0.066° | [0.21, 0.20, −0.07]° | 0.30° |
| sequence_3_17 | 1WM093701G1276 | 4.5 ms | 0.072° | [0.28, 0.24, −0.10]° | 0.38° |
| sequence_4_10 | 1WM093701G1276 | 4.5 ms | 0.080° | [0.34, 0.21, −0.06]° | 0.41° |
| sequence_1_19 | 1WM09370081264 | 4.5 ms | 0.069° | [0.57, −0.29, 0.37]° | 0.74° |
| sequence_1_20 | 1WM09370081264 | 4.5 ms | 0.068° | [0.60, −0.26, 0.32]° | 0.73° |

- **Time.** The gyroscope stamps its samples 4.5–4.75 ms after the image they
  belong to; the factory `TimeOffsetSec_Device_Gyro` is 4.1 ms. Aligning
  them divides the per-pair residual by three (0.228° → 0.066° on
  sequence_2_11). The delay is the same on both devices and does not depend
  on exposure: sequence_4_10 is exposed for 14 ms and sequence_2_11 for
  0.08 ms (median).
- **Rotation.** The fitted gyro-to-camera rotation differs from the factory
  one by 0.3–0.4° on 1WM093701G1276 and 0.73° on 1WM09370081264. Within a
  device the correction agrees to within 0.15°.
- **Rectification.** The factory rectification `R^-1 (raw − b)` fits better
  than dropping or flipping its rotation part (0.38°, 0.63° and 0.88° from the
  factory rotation on sequence_3_17). A general 3×3 gyro correction improves
  the residual by only 1–5 %: the gyro scale is right to 0.2–0.3 %.

Every test sequence was recorded with one of these two devices
(1WM093701G1276: 34 sequences, 1WM09370081264: 29).

### Partial corrections make it worse

The two errors partly cancel in the VIO, so correcting one of them alone
is worse than correcting neither (VIO only, `gyro_bias_std` 1e-7):

| Run | Score | Yaw drift at the end | Tilt error, max |
|---|---:|---:|---:|
| sequence_2_11 baseline | 38.36 | −3.05° | 1.30° |
| sequence_2_11, time only | 35.61 | −10.18° | 0.68° |
| sequence_3_17 baseline | 35.76 | −6.92° | 1.65° |
| sequence_3_17, rotation only (own fit) | 34.09 | +4.55° | 1.62° |
| sequence_3_17, time + rotation (own fit) | **49.23** | **+0.25°** | **0.82°** |

With both corrected from sequence_3_17's own fit, its Pose recall rises from
4.2 % to 27.1 % at 1 m and from 86.2 % to 100 % at 5 m.

### Per-device refinement

[`scripts/aria_device_refinement.py`](../scripts/aria_device_refinement.py)
stores one refinement per device, fitted on training sequences only
(1WM093701G1276: sequence_2_11, sequence_3_17 and sequence_4_10;
1WM09370081264: sequence_1_19). The test-set ground truth is never used. The
gyro delay is 4.6 ms on both devices. The accelerometer delay keeps the
factory gyroscope-accelerometer difference: 3.58 ms and 3.14 ms. VIO only:

| Sequence | Device | Refinement | Score | Pose R@5m | Yaw drift at the end |
|---|---|---|---|---|---|
| sequence_2_11 | G1276 | time + rotation | 38.36 → **49.68** | 89.8 → 100 % | −3.05° → +0.74° |
| sequence_5_11 (held out) | G1276 | time + rotation | 32.27 → **43.14** | n/a | −3.18° → −0.69° (CP fit) |
| sequence_1_20 (held out) | 81264 | time + rotation | 33.18 → 27.32 | 88.0 → 74.8 % | −6.00° → +10.45° |
| sequence_1_20 (held out) | 81264 | time only | 33.18 → **38.10** | 88.0 → 94.2 % | −6.00° → −5.27° |

On 1WM09370081264 the fitted 0.73° rotation overshoots, although it is
reproduced to 0.07° by sequence_1_20's own fit. Rotating only the gyroscope
(leaving the accelerometer-camera rotation at the factory value) gives the
same overshoot over the first 8 minutes of sequence_1_20 (yaw +4.5° against
+4.4° for the rig rotation, −2.1° with the time alignment alone), so the
accelerometer is not the cause. That device therefore gets the time
alignment only. The time alignment alone also brings sequence_1_20's global
Sim(3) scale from 0.983 to 1.001 and its first-8-minute ATE from 1.26 m to
0.73 m.

The submission driver applies the refinement by default
(`--no-device-refinement` turns it off). It shifts the gyroscope and
accelerometer samples while rectifying the IMU, and rotates the camera rig in
the Basalt calibration, keeping the stereo extrinsics.

### With the online mapper

The mapper protocol of
[`lamaria_online_mapper.md`](lamaria_online_mapper.md#loop-rotation-gate)
(5° loop gate), rerun with the refinement from the device table. Each run
writes both the VIO and the mapper trajectory, so both columns come from the
same run:

| Sequence | Track | VIO: factory → refined | Mapper: factory → refined |
|---|---|---|---|
| sequence_1_19 | Short | 52.26 → **71.96** | 54.39 → 68.54 |
| sequence_2_11 | Medium | 38.36 → 49.68 | 56.35 → **55.51** |
| sequence_3_17 | Long | 35.76 → **50.06** | 41.61 → 46.68 |
| sequence_4_10 | Low light | 32.42 → **34.49** | 36.20 → 31.62 |
| sequence_5_11 | Moving platform | 32.27 → 43.14 | 36.19 → **49.84** |
| **Mean** | | 38.21 → 49.87 (+11.66) | 44.95 → **50.44** (+5.49) |

Bold marks the better of the two refined columns. sequence_3_17 scores 50.06
with the device table, against 49.23 with its own fit.

The refinement raises the VIO on all five tracks, but the mapper keeps only
half of the gain. It now beats the VIO on Medium (+5.83) and Moving platform
(+6.70) and loses on Short (−3.42), Long (−3.38) and Low light (−2.87). Pose
recall against the pseudo-GT (official evaluator, VIO / mapper) shows the
same split:

| Sequence | Pose R@1m, factory | Pose R@1m, refined | Pose R@5m, refined |
|---|---|---|---|
| sequence_1_19 | 26.5 / 30.8 % | **76.3** / 60.6 % | 100 / 100 % |
| sequence_2_11 | 5.2 / 47.9 % | 30.9 / **39.4** % | 100 / 100 % |
| sequence_3_17 | 4.2 / 9.5 % | **24.7** / 21.7 % | **100** / 97.5 % |
| sequence_4_10 | 5.7 / 6.2 % | **5.5** / 0.0 % | **86.5** / 75.7 % |

Where the mapper loses, it bends a trajectory that no longer needs it.

- On sequence_1_19 it moves the VIO by 0.2–0.6 m. After the global Sim(3)
  alignment the 2D error rises from 0.87 m to 0.96 m RMSE.
- On sequence_4_10 it moves the VIO by up to 3.5 m in the first 12 minutes.
  Its final bundle adjustment keeps only about 930 landmarks there, against
  5,500–9,600 on the other tracks. The VIO relative-pose factors dominate the
  final cost: 3.0e5, against 1.0e5 for the reprojection terms.

Locally the mapper is still better. Aligning each 60 s segment separately
(Sim(3)), its RMSE is lower on all four sequences with pseudo-GT, for example
0.135 m against 0.151 m on sequence_1_19 and 0.421 m against 0.442 m on
sequence_4_10. So the loss is in the long-range shape, which the refined VIO
now gets better than the mapper.

Weighting the VIO relative-pose factors ten times higher
(`--relative-pose-weight 10`) changes almost nothing: the mapper scores
31.79 instead of 31.62 on sequence_4_10 and 68.32 instead of 68.54 on
sequence_1_19. On sequence_4_10 it still moves the VIO by up to 3.5 m. In
units of weight 1, the relative-pose residuals shrink only from 3.0e5 to
2.3e5, while the roll-pitch residuals grow from 3.9e4 to 3.0e5 and the
reprojection residuals from 1.0e5 to 3.0e5. The relative-pose factors
therefore disagree with both the roll and pitch the VIO measures at each
keyframe and the mapper's landmarks. The roll-pitch measurements come from
the VIO's own keyframe poses, so it is the relative-pose factors, not the
reprojection terms, that pull the mapper away from the VIO.

Each marginalized keyframe contributes a relative-pose factor to every other
keyframe in the VIO window, about 30 with the big-window config. The
information of each factor is the marginal information of that one relative
pose, so the same IMU and visual information is counted up to 30 times.
`--relative-pose-max-neighbors 2` (opt-in) keeps only the factors to the two
keyframes nearest in time. Crossed with loop matching on and off
(`--no-loop-matching`), on sequence_4_10:

| Mapper on sequence_4_10 | Loops on | Loops off |
|---|---|---|
| Factors to every keyframe in the window (default) | 31.62 (R@5m 75.7 %) | 31.70 (76.0 %) |
| Factors to the 2 nearest keyframes | **34.83 (88.8 %)** | 31.65 (78.2 %) |

The VIO alone scores 34.49 (86.5 %).

- **Loops off:** both versions move the VIO by the same 3 m. The relative
  poses recovered from the marginalization data do not chain up to the VIO
  trajectory; the loops are not what bends it.
- **Default with loops:** the 13 loops of sequence_4_10 change nothing,
  since the 30-fold relative-pose chain is too stiff to move.
- **2 nearest keyframes with loops:** the loops pull the trajectory back. The
  relative-pose and roll-pitch residuals at the optimum fall 20- to 30-fold
  (9.7e3 and 1.9e3, against 3.0e5 and 3.9e4). This is the first mapper
  setting that beats the VIO on a track where the default mapper loses.

The limit does not carry over to sequence_1_19 (with the 0.5 px KLT
threshold of [`lamaria_klt_threshold.md`](lamaria_klt_threshold.md)): the
mapper falls to 45.87, against 70.08 with the default factors and 73.79 for
the VIO. Its heading drifts by up to 4° from the VIO's, which stays within
±1° of the pseudo-GT. The 222 accepted loops are again short revisits that
the pseudo-GT confirms (median distance error 0.08 m). With two factors
per keyframe the relative-pose chain no longer holds the heading over the
whole sequence, and short loops cannot replace it.

`--mapper-vio-output-poses` takes the factor measurements from the VIO's
per-frame output instead of the window estimates (the factor information is
unchanged). With the default factors, on sequence_5_11 with the 0.5 px KLT
threshold, the mapper then reproduces the VIO: 44.69 against 44.73, a
constant 0.12 m apart. Its 198 loops move nothing, because the 30-fold
relative-pose chain is too stiff for them. The default mapper's 52.99 on the
same run therefore owes its gain over the VIO to the window estimates,
which on sequence_5_11 chain up to a better trajectory than the VIO output,
and not to its loops.

Softer chains with the VIO output as measurements, on sequence_1_19 (0.5 px
threshold; VIO 73.79, default mapper 70.08):

| Relative-pose factors per keyframe | Window estimates | VIO output |
|---|---:|---:|
| 2 nearest | 45.87 | 66.60 |
| 4 nearest | n/a | 69.71 |

On sequence_4_10 (0.5 px threshold, VIO 39.00) the same two settings score
38.77 and 39.07: within 0.35 m of the VIO everywhere, since its 13 loops
change little.

The VIO output removes most of the 2-neighbor heading loss, but neither
version reaches the VIO. Both rotate the first 90 s by about 3° (2D error
2.1–2.3 m there, against 1.15 m for the VIO) and follow the VIO elsewhere.
sequence_1_19 was recorded by 1WM09370081264, whose 0.73° IMU-camera
rotation stays uncorrected (see [Per-device refinement](#per-device-refinement)).
The VIO tolerates it, but the mapper's reprojection terms link keyframes
seen from directions up to 90° apart, where it does not cancel.

All 13 accepted loops are short revisits that the pseudo-GT confirms: the
two keyframes lie 0.1–7.7 m apart and the map distance matches within
0.6 m. With the 2-neighbor limit the mapper's final optimization takes
136 s instead of 1,368 s, and the peak memory is 761 MB instead of 1.7 GB.

### Calibration without ground truth (not usable yet)

- **Camera-only rotations** (KLT + essential matrix on cam0, 150 ms pairs)
  recover the 5 ms delay, but their rotation axes are biased: 0.66° per pair
  against the pseudo-GT, and the fitted gyro-to-camera rotation is about 4°
  off. The bias does not average out, which points to forward walking
  coupling into the rotation estimate.
- **Stereo pairs** (SIFT across the 75° camera divergence) give too few
  matches: 4–6 usable frames and ±2–3° scatter.
- **Gravity** (gyro-propagated, accelerometer-observed) estimates the
  gyro-to-accelerometer rotation with ~1° scatter between sequences.

## Changes in the VIO (both opt-in, default off)

### VIO outlier filter

Upstream's `sqrt_keypoint_vio.cpp` leaves `filterOutliers` as a TODO, and
the Rust port follows it: a visual observation stays in every later solve,
bounded only by its Huber weight. `config.vio_filter_outliers: true` adds
the filter of upstream's older `keypoint_vio.cpp`.

- After each window solve, it removes the observations whose reprojection
  error exceeds `vio_outlier_threshold` (3 px).
- It removes landmarks left with fewer than four observations.
- It then re-solves for the three iterations after `vio_filter_iteration`.

The default trajectory is bit-identical. On the ride above the filter changes
nothing. On whole sequences it helps the VIO alone on the night walk
(sequence_4_10, 34.69 to 57.41), but the chain from the filtered VIO loses 4.0
on the mean of nine training sequences, so it stays off ([The VIO's outlier
filter](lamaria_batch_ba.md#the-vios-outlier-filter)).

### Earth-rotation compensation

`config.vio_earth_rotation_latitude_deg` subtracts the vertical component
of Earth's rotation, `R_wb^T [0, 0, w_e sin(lat)]`, from every gyro sample.
`R_wb` is the newest window state. Measured on sequence_5_11 (VIO only):

| Run | Score | Fitted yaw drift |
|---|---:|---:|
| Baseline | 32.27 | −3.18° |
| Gyro corrected offline from the baseline's attitude | 32.98 | −2.87° |
| `config.vio_earth_rotation_latitude_deg: 47.376` (online) | 33.00 | −2.88° |

The online correction matches the offline one. A single sequence cannot
separate +0.7 from the Score's sensitivity to small input changes
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md#imu-time-offset)),
so it stays off.

## Open items

1. **Mapper on the refined VIO.** The mapper now loses to the VIO on three
   of five tracks ([With the online mapper](#with-the-online-mapper)). On
   sequence_4_10, limiting the relative-pose factors to the 2 nearest
   keyframes turns that into a small gain, but on sequence_1_19 it loses
   the heading. Without loops, the recovered relative poses drift 3 m away
   from the VIO; taking the factor measurements from the VIO trajectory
   itself (`--mapper-vio-output-poses`) is being measured.
2. **Calibration without the device table.** The refinement relies on
   both devices having training pseudo-GT. Estimating the gyro delay and the
   IMU-camera rotation online in the VIO would remove that dependency and
   may explain why 1WM09370081264 rejects its fitted rotation.
3. **Local scale drift.** On sequence_5_11 it is the largest error term.
   The online mapper's joint VI-BA (`--joint-vi-ba-weight 10`) leaves the
   local scale of sequence_3_17 unchanged, so the IMU factors carry too
   little weight there.
4. **The funicular ride.** It is worth at most +5.2 on sequence_5_11. The
   test set's Moving sequences are 11–24 GB, against 10 GB for
   sequence_5_11, and may contain longer rides.
5. **The scale of sequence_5_11.** The VIO measures the route about 6 %
   short with or without the ride. Removing the fitted scale drift alone
   would lift its Score from about 50 to 65 (VIO) and 57 to 75 (mapper),
   the largest gain left on any track.
