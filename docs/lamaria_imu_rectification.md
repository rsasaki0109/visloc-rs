# LaMAria: Aria factory IMU rectification

LaMAria's ASL export ships the **raw** Project Aria IMU samples. Aria's factory
calibration models each IMU sensor as `raw = R · real + b` (a 3×3
rectification matrix for scale and misalignment, plus a constant bias, for the
gyroscope and the accelerometer separately). The ASL release does not apply it,
and the per-sequence pinhole calibration JSON does not carry it. It lives in the
JSON calibration record in the header of the raw `.vrs` recording, which an
HTTP range request for the first 4 MiB retrieves without downloading the
multi-GB file.

[`scripts/aria_factory_imu_rectify.py`](../scripts/aria_factory_imu_rectify.py)
extracts that record (`extract`) and rewrites `mav0/imu0/data.csv` as
`real = R⁻¹ (raw − b)` for the `imu-right` IMU (`rectify`), which is the IMU
that LaMAria's `imu0` stream and `T_b_s` frames refer to. Images are untouched.

## Protocol

- **Sequences**: one training sequence per non-Short category: `sequence_2_11`
  (Medium), `sequence_3_17` (Long), `sequence_4_10` (Low light), and
  `sequence_5_11` (Moving platform).
- **Both arms** use the submission defaults of
  `scripts/run_lamaria_test_submission.py`: VIO only (no mapper), config
  `configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json`,
  variant-A noise, `gyro_bias_std = 1e-6` (the default at the time; see
  [Re-tuning](#re-tuning-gyro_bias_std) for 1e-7), 2 threads, as fast as
  possible. The arms differ **only** in `imu0/data.csv` (raw vs rectified).
- **Scoring**: the official `cvg/lamaria` evaluator, unmodified. Score and
  CP@1m come from the control points; pose recall comes from the pseudo-dense
  GT (not published for `sequence_5_11`).
- One run per arm and sequence. The VIO is deterministic: re-running the
  rectified `sequence_2_11` arm under a different CPU load (wall time 6,360 s
  vs 7,448 s) gave a bit-identical `trajectory.tum`. Repeat runs therefore add
  nothing. The remaining uncertainty is how sensitive the score is to small
  input changes (see [IMU time offset](#imu-time-offset)). Nothing was tuned
  against ground truth: the factory model is applied as published.

## Results

| Sequence | Category | Frames | Score raw → rect | CP@1m raw → rect | Pose R@1m raw → rect | Pose R@5m raw → rect |
|---|---|---:|---|---|---|---|
| sequence_2_11 | Medium | 23,748 | 37.72 → **39.47** (+1.74) | 16.7 → 11.1 % | 16.3 → 5.6 % | 81.6 → **90.3 %** |
| sequence_3_17 | Long | 35,842 | 24.14 → **28.89** (+4.74) | 3.7 → **7.4 %** | 0.4 → **2.8 %** | 53.5 → **65.5 %** |
| sequence_4_10 | Low light | 28,410 | 20.42 → **26.42** (+6.01) | 0.0 → 0.0 % | 0.0 → **0.8 %** | 48.8 → **66.6 %** |
| sequence_5_11 | Moving platform | 22,483 | 27.64 → **32.56** (+4.92) | 0.0 → 0.0 % | n/a | n/a |
| **Mean** | | | 27.48 → **31.83** (+4.35) | | | |

The score improves on all four sequences, and pose recall at 5 m improves by
9–18 points on the three sequences with pseudo-GT. The one regression is
`sequence_2_11`'s fraction within 1 m (CP@1m 16.7 → 11.1 %, pose R@1m
16.3 → 5.6 %). There the better global shape (see below) trades against a
section that the raw run happened to fit more tightly.

### Where the gain comes from (`sequence_2_11`)

Trajectory diagnostics against the pseudo-GT (Sim(3)-aligned):

| Run | ATE Sim3 RMSE | max | yaw drift range (60 s windows) | error, first decile |
|---|---:|---:|---:|---:|
| VIO, raw IMU | 4.01 m | 8.51 m | 12.1° | 6.6 m |
| VIO, raw IMU + online NFR mapper | 3.71 m | 7.28 m | — | 5.9 m |
| **VIO, rectified IMU** | **3.10 m** | 6.07 m | **8.0°** | **2.8 m** |

Rectification cuts ATE by 23 % and the yaw drift range by a third, and halves
the error in the first tenth of the sequence, so the raw IMU hurts most around
initialisation. Rectification alone does more than adding the online mapper to
the raw-IMU run (Score 38.10, ATE 3.71 m). Only 8.6 % of `sequence_2_11`'s
poses have a revisit within 5 m more than 60 s apart, so loop closure has
little to correct on this sequence. Reducing VIO drift is the lever for
Medium/Long.

## Re-tuning `gyro_bias_std`

The earlier default of 1e-6 came from a sweep on the **raw** IMU
([`lamaria_multicam.md`](lamaria_multicam.md)) that improved monotonically
from 5e-4 down to 1e-6, the smallest value tried. With the rectified IMU, the
sweep was extended to both sides of 1e-6, with the same protocol as above and
one training sequence per track:

| Sequence | Track | Score 1e-5 | Score 1e-6 | Score **1e-7** | ATE 1e-6 → 1e-7 | Pose R@5m 1e-6 → 1e-7 |
|---|---|---:|---:|---:|---|---|
| sequence_1_19 | Short | — | 51.10 | **52.26** | 2.25 → **2.17 m** | 100 → 100 % |
| sequence_2_11 | Medium | 39.22 | **39.47** | 38.36 | **3.10** → 3.21 m | **90.3** → 89.8 % |
| sequence_3_17 | Long | — | 28.89 | **35.76** | 6.72 → **4.63 m** | 65.5 → **86.2 %** |
| sequence_4_10 | Low light | 28.93 | 26.42 | **32.42** | 4.63 → **3.58 m** | 66.6 → **86.2 %** |
| sequence_5_11 | Moving platform | — | **32.56** | 32.27 | n/a | n/a |
| **Mean** | | | 35.69 | **38.21 (+2.53)** | | |

1e-7 gains the most on the long and dark sequences, where it is also a large
trajectory improvement (ATE −31 % / −23 %, pose recall at 5 m +21 / +20
points). That is well beyond the few-point Score sensitivity seen in the
time-offset test below. On Medium and Moving it is within about a point,
and the trajectory metrics move by less than 0.11 m. 1e-5 is worse than 1e-7
on trajectory error on both sequences where it was run (ATE 3.51 vs 3.21 m
and 4.52 vs 3.58 m), so the looser side was not pursued. **1e-7 is now the
driver default.**

On sequence_1_19, rectification at 1e-6 scores 51.10 against 50.04 for the
raw IMU in `lamaria_multicam.md`. That raw run used an older build, so the
+1.06 is indicative only. Values below 1e-7 were not tried.

## IMU time offset

The factory record also gives per-sensor time offsets for `imu-right`
(`TimeOffsetSec_Device_Gyro` 4.1 ms, `TimeOffsetSec_Device_Accel` 3.1 ms;
`projectaria_tools` documents them only as "time offset device to gyroscope",
without a sign convention). LaMAria's own tooling does not apply them: both
`tools/vrs_to_asl_folder.py` (which writes the ASL IMU CSV) and the VI
optimisation in `lamaria/utils/aria.py` take IMU timestamps as `DEVICE_TIME`
unchanged. The VI optimisation does apply `raw_to_rectified_{accel,gyro}`,
the same `R⁻¹ (raw − b)` model used here.

Both signs were measured on `sequence_2_11` (rectified IMU, otherwise the
same protocol). The shift moves every IMU timestamp; 4,145,618 ns is the gyro
offset.

| IMU shift | Score | CP@1m | Pose R@1m | Pose R@5m | ATE Sim3 RMSE | SE3 RMSE | Sim3 scale |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 (default) | 39.47 | 11.1 % | 5.6 % | 90.3 % | **3.10 m** | **3.86 m** | **0.981** |
| +4.1 ms | **45.13** | **27.8 %** | **20.3 %** | 85.9 % | 3.94 m | 7.29 m | 0.951 |
| −4.1 ms | 38.02 | 11.1 % | 14.5 % | **93.2 %** | 3.23 m | 4.16 m | 1.023 |

The two signs disagree with each other, and the metrics disagree on which
shift is best. +4.1 ms raises the Score by 5.7 because 5 rather than 2 of
the 18 control points fall within 1 m. The same shift also gives the worst
global trajectory: ATE +27 %, SE3 error nearly doubled, and the scale
estimate 5 % short. −4.1 ms is close to the unshifted run on every metric.

A ±4 ms perturbation thus moves the Score by several points in either
direction without a consistent gain. That is also a measure of how sensitive
a single-sequence Score is to small input changes. **The shift stays off**
(the driver applies none). Adopting it would need a multi-sequence result
that improves the trajectory, not only the control-point count.

## Reproducing

```sh
B=https://cvg-data.inf.ethz.ch/lamaria
curl -fsS -r 0-4194303 -o seq.head $B/raw_data/training/$SEQ.vrs
python3 scripts/aria_factory_imu_rectify.py extract seq.head factory_calib.json
mkdir -p $SEQ_rect/mav0/imu0
cp -al $SEQ/mav0/cam0 $SEQ/mav0/cam1 $SEQ_rect/mav0/      # images shared
python3 scripts/aria_factory_imu_rectify.py rectify factory_calib.json \
    $SEQ/mav0/imu0/data.csv $SEQ_rect/mav0/imu0/data.csv
```

Then run the VIO on `$SEQ_rect` exactly as on `$SEQ`.

## Applicability to the test set

The test set's raw recordings are published too (`raw_data/test/<seq>.vrs`),
and their headers carry the same record. Test sequences `sequence_1_1` and
`sequence_3_1` and every training sequence above come from the same device
(serial `1WM093701G1276`) with identical factory gyro/accel biases. The step is
therefore available for a submission without touching ground truth.

`scripts/run_lamaria_test_submission.py` applies the rectification by default
(`--no-imu-rectify` turns it off). It fetches `raw_data/test/<seq>.vrs`'s first
4 MiB, rewrites `imu0/data.csv` in place, and falls back to the raw IMU with a
`WARN` line if the header cannot be fetched or parsed.

## Open items

1. **More sequences per track.** One sequence per track cannot separate a
   real gain from the Score's sensitivity to small changes (see the
   time-offset table). The evidence that rectification and 1e-7 are real is
   their direction across tracks together with their trajectory
   improvement.
2. **Below 1e-7.** The sweep stops at 1e-7; smaller values were not tried.
