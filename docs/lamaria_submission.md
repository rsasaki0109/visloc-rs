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
     and the per-device IMU time alignment
  -> pinhole -> Basalt Double-Sphere calibration (variant-A IMU noise,
     camera rig rotated by the per-device refinement)
  -> run the Basalt VIO
  -> convert the trajectory to the submission estimate
  -> append `slam/<sequence>.txt`, then delete the sequence data
```

It is resumable: a sequence whose estimate already exists in `--slam-dir` is
skipped, so an interrupted run can be restarted unchanged.

The test-set scores of the submissions so far, and what changed between them,
are in [lamaria_test_results.md](lamaria_test_results.md).

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
  cargo build --release --example basalt_euroc_online_slam_demo --features basalt-lm-workspace-reuse

python scripts/run_lamaria_test_submission.py \
  --tracks 1 \
  --slam-exe target/release/examples/basalt_euroc_online_slam_demo \
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

`--slam-exe` runs the VIO plus the online mapper, which writes both the
full-frame mapper trajectory `trajectory_online.tum` and the VIO's own
`trajectory_vio.tum`. Which one is submitted depends on the track. With the
per-device refinement, one training sequence per track scores:

| Track | VIO | VIO + mapper | Submitted by default |
|---|---:|---:|---|
| Short (sequence_1_19) | **71.96** | 68.54 | VIO |
| Medium (sequence_2_11) | 49.68 | **55.51** | mapper |
| Long (sequence_3_17) | **50.06** | 46.68 | VIO |
| Low light (sequence_4_10, low-light config) | **39.00** | 36.65 | VIO |
| Moving platform (sequence_5_11) | 43.14 | **49.84** | mapper |
| **Mean** | 50.77 | 51.44 | **53.27** |

(see [`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#with-the-online-mapper)).
`--vio-tracks` sets the tracks that submit the VIO (default `1 3 4`; with no
values the mapper is submitted everywhere). If `--vio-exe`
(`basalt_euroc_vio_demo`) is passed as well, those tracks run the faster
VIO-only build. `--vio-exe` alone runs the VIO only on every track. Each
choice rests on one training sequence per track, so the gain over a uniform
setup (51.44 with the mapper everywhere) will be smaller on the test set.

Low light sequences (track 4) run
[`euroc_config_big_window_multicam_lowlight.json`](../configs/basalt/variants/lamaria/euroc_config_big_window_multicam_lowlight.json),
whose KLT round-trip threshold is 0.5 px instead of 0.2 px. It is worth +4.5
on sequence_4_10 but −11.0 on the Medium sequence
([`lamaria_klt_threshold.md`](lamaria_klt_threshold.md)). `--low-light-config
none` uses `--config` there too.

Not every Low light sequence is dark. The two low-light training sequences
average a grey level of 26-29 on cam0 (60 images spread over the recording),
the eight day training sequences 74-88, and test sequence_4_1, 4_4, 4_8 and
4_9, shot at dusk or under an overcast sky, 74-75. A Low light sequence that
averages 70 or more (`DAYLIT_BRIGHTNESS`) runs the day options: `--config`
for the VIO, and the BA chain's day steps (plain SIFT, the warm-up, the batch
pass). By day both are measured to help: the 0.5 px KLT threshold costs the
Medium sequence 11 points, and the pass adds 4-9 points on four of the six
walking day sequences ([lamaria_batch_ba.md](lamaria_batch_ba.md#the-first-minutes-online)).
The other test sequences of the track (grey level 27-60) keep the low-light
options.

Moving-platform sequences (track 5) run the VIO with the platform filter
(`config.vio_platform_filter`, [`platform.rs`](../pipelines/basalt/src/platform.rs);
`--platform-tracks` with no values turns it off). On a tram the cabin's
features say the head stands still and outweigh the street seen through the
windows, so the VIO shortens or stops the ride. When the accelerometer shows
no steps and enough tracks move with one translation beyond the gyro's
rotation, only those tracks reach the estimator. With the BA chain (GitHub
Actions, same commit, filter off / on): sequence_5_11 60.29 / **66.38**,
sequence_5_12 **7.41** / 5.98. On sequence_5_12 the chord across the tram
(CPs at 18 s and 208 s) comes out at 0.59 / **0.95** of the survey; its score
stays set by the last four minutes, whose chords are 21-50 % too long either
way.

On the test set's trams seen through a door's windows the street is often in
cam1's view, and the default VIO tracks cam0's features only: on
sequence_5_9 the filter kept the cabin out for 499 of the ride's ~1,900
frames and the submitted trajectory has no ride. With both cameras' features
(`MULTICAM`, `--platform-multicam`, on by default on the platform tracks) it
keeps it out for 1,283 frames on a window around the ride, and after the BA
chain the 5.2-7.0 min ride covers 450 m (chord 332 m) instead of 142 m (91
m). sequence_5_5 and 5_6 do not change: through their door windows too few
street tracks last even with both cameras (sequence_5_5: 29 m over its
90 s ride), and holding the ride until the steps come back adds motion
after it that is not there. On the training sequences (GitHub Actions,
before / after): sequence_5_11 66.77 / 65.94, sequence_5_12 6.76 / 6.27,
within a rerun's spread. On the test set it takes the track from 26.9 to 30.6
(submission 4, [lamaria_test_results.md](lamaria_test_results.md#what-changed-between-the-submissions)),
though it costs sequence_5_4 its first ride.

By day the BA chain matches the keyframes once more after its batch pass,
guided by the pass's trajectory instead of the VIO, and solves again
(`--ba-rematch`, on by default; `--no-ba-rematch` skips it). The matching
band admits the matches that agree with the guide's drift, and the pass's
heading drifts far less than the VIO's. On the eight day training sequences
(GitHub Actions, one run, before / after): mean 64.50 / 64.76, no sequence
down by more than 0.05; on training R_12_10cp, a fast walk along open streets
like the Short test sequences that score lowest, 61.09 / 64.30
([lamaria_batch_ba.md](lamaria_batch_ba.md#matching-again-after-the-pass)). It
adds 10-20 minutes per sequence on a runner. On the test set it moves the day
trajectories by 0.3 m or less (submission 4).

IMU rectification is on by default
([`lamaria_imu_rectification.md`](lamaria_imu_rectification.md); +4.35 Score on
average across one training sequence per non-Short track). It fetches only the
first 4 MiB of `raw_data/test/<sequence>.vrs` with an HTTP range request, never
the multi-GB recording, and saves the parsed record as
`<work-dir>/calibrations/<sequence>_factory_calib.json`. If that fails, the
sequence runs on the raw IMU and the log shows a `WARN` line.

The per-device IMU-camera refinement is on by default as well
([`lamaria_drift_analysis.md`](lamaria_drift_analysis.md#imu-camera-calibration);
`--no-device-refinement` turns it off). The test set was recorded with the two
devices that also recorded the training set. For each, the refinement aligns
the gyroscope and accelerometer in time (gyroscope 4.6 ms) and, on
1WM093701G1276, corrects the IMU-to-camera rotation by 0.36 deg. Both values
were fitted on training pseudo-GT only. A sequence from another device keeps
the factory calibration, with a `WARN` line.

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
