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
  variant-A noise, `gyro_bias_std = 1e-6`, 2 threads, as fast as possible. The
  arms differ **only** in `imu0/data.csv` (raw vs rectified).
- **Scoring**: the official `cvg/lamaria` evaluator, unmodified. Score and
  CP@1m come from the control points; pose recall comes from the pseudo-dense
  GT (not published for `sequence_5_11`).
- One run per arm and sequence. Nothing was tuned against ground truth: the
  factory model is applied as published.

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
section that the raw run happened to fit more tightly. With one run per arm,
this cannot be told apart from run-to-run variation.

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

1. **IMU time offset.** The factory record also gives per-sensor time offsets
   for `imu-right` (`TimeOffsetSec_Device_Gyro` 4.1 ms,
   `TimeOffsetSec_Device_Accel` 3.1 ms). It is not known whether the ASL
   timestamps already include them. `rectify --shift-ns` can apply a shift, but
   it has not been measured.
2. **Re-tune `gyro_bias_std`.** The default of 1e-6 was chosen on raw IMU data;
   with the bias removed up front, a looser value may now be better.
3. **Short track and repeat runs.** No Short sequence was measured, and each
   arm ran once.
