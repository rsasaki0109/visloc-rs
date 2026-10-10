# LaMAria test-set submission pipeline

[LaMAria](https://lamaria.ethz.ch/leaderboard) (ETH, egocentric city-scale
VI-SLAM on Project Aria glasses) is scored by uploading a zip of
`slam/<sequence>.txt` trajectories.  Each file is TUM with **nanosecond**
timestamps and poses in `world_from_imu`, and must cover the image timestamps.

[`scripts/run_lamaria_test_submission.py`](../scripts/run_lamaria_test_submission.py)
drives the whole per-sequence loop for one or more tracks:

```
download ASL zip + pinhole calibration
  -> verify + extract -> rename `aria/` to `mav0/`
  -> apply Aria's factory IMU calibration to `imu0/data.csv`
     (first 4 MiB of the raw `.vrs`; `--no-imu-rectify` skips it)
  -> pinhole -> Basalt Double-Sphere calibration (variant-A IMU noise)
  -> run the Basalt VIO
  -> convert the trajectory to the submission estimate
  -> append `slam/<sequence>.txt`, then delete the sequence data
```

It is resumable: a sequence whose estimate already exists in `--slam-dir` is
skipped, so an interrupted run can be restarted unchanged.

## Track layout

| track | leaderboard track | sequences | count |
|---|---|---|---|
| 1 | Short | `sequence_1_1` .. `sequence_1_18` | 18 |
| 2 | Medium | `sequence_2_1` .. `sequence_2_10` | 10 |
| 3 | Long | `sequence_3_1` .. `sequence_3_16` | 16 |
| 4 | Low light | `sequence_4_1` .. `sequence_4_9` | 9 |
| 5 | Moving platform | `sequence_5_1` .. `sequence_5_10` | 10 |

## Usage

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" \
  cargo build --release --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse

python scripts/run_lamaria_test_submission.py \
  --tracks 1 \
  --vio-exe target/release/examples/basalt_euroc_vio_demo \
  --config configs/basalt/variants/lamaria/euroc_config_big_window_multicam.json \
  --gyro-bias-std 1e-7 \
  --work-dir /path/to/lamaria_submission/work \
  --slam-dir /path/to/lamaria_submission/slam \
  --threads 12
```

`--config` defaults to the multi-camera big-window LaMAria variant
(`euroc_config_big_window_multicam.json`) and `--gyro-bias-std` to 1e-7, the
best measured setup ([`lamaria_multicam.md`](lamaria_multicam.md) for the
config, [`lamaria_imu_rectification.md`](lamaria_imu_rectification.md#re-tuning-gyro_bias_std)
for the value, which assumes the rectified IMU); it is about
1.45× slower than `euroc_config_big_window.json`.  `aria2c` is used with 16 connections when available
(the server throttles a single connection), otherwise `urllib` is the fallback.

IMU rectification is on by default
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md); +4.35 Score on
average across one training sequence per non-Short track). It fetches only the
first 4 MiB of `raw_data/test/<sequence>.vrs` with an HTTP range request, never
the multi-GB recording, and saves the parsed record as
`<work-dir>/calibrations/<sequence>_factory_calib.json`. If that fails, the
sequence runs on the raw IMU and the log shows a `WARN` line.

On completion the driver writes `submission.zip` next to `--slam-dir` (i.e.
`<work-dir>/../submission.zip`) containing `slam/<sequence>.txt` for every
estimate collected so far.  Upload it at <https://lamaria.ethz.ch/login>
(24 h submission rate limit).

## Notes

- The test set is ~862 GB across 63 sequences; each sequence's data is deleted
  after its estimate is written, so plan for the largest single sequence plus
  working space, not the whole set.
- Runtime is dominated by the VIO.  With the larger window the demo runs at
  roughly 0.4x real time, so a 48-minute sequence takes a few hours; a full
  track is an overnight job, and the full set is a multi-day job.
- The driver only reads sensor data: ground truth is never used.
