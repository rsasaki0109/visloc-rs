# LaMAria with one camera (mono + IMU)

The LaMAria leaderboard lists each method's sensors: mono, bino, imu. The
visloc-rs entry so far is bino + IMU (both SLAM cameras). This page runs the
same pipeline on the left SLAM camera (cam0) and the IMU alone, for a
mono + IMU entry, listed as "visloc-rs (mono)" (submission 77, uploaded
2026-11-25). On the test set it scores **52.9** on average over the five
challenges, against 57.8 for the bino entry
([Test set](#test-set)).

## What changed

`scripts/run_lamaria_test_submission.py --mono`:

- The VIO runs monocular-inertial (`basalt_euroc_vio_demo --mono`,
  [mono_inertial_vio.md](mono_inertial_vio.md)): cam0 and the IMU, every
  cam0 frame.
- The fallbacks that add the second camera are off. A diverged VIO goes
  straight to the late starts. The Moving platform track runs without
  `--platform-multicam`; its platform filter stays on.
- The BA chain runs on cam0's keypoints. `lamaria_batch_ba.py features
  --cameras 1` (passed by `lamaria_ba_chain.py --cameras 1`) records the
  camera count in `features.npz`. Every later step reads it back and uses
  the calibration's first camera. With two cameras, the default, the images
  are indexed as before.

The GitHub Actions workflow takes it through its `driver_args` input
(`--mono`). GPS bridging ([lamaria_gps.md](lamaria_gps.md)) is a separate
step and was not used here: a mono + IMU entry keeps to the camera and the
IMU.

Running cam0 alone found one bug, which also affects the two-camera runs:
on sequence_5_11 a degenerate image pair made OpenCV's essential-matrix
solver assert, and the `loops` step failed, so the chain fell back to the
VIO. Such a pair now just gives no revisit. The PnP of `loop-correct` is
guarded the same way.

## Results on the training sequences

One training sequence per track, run locally (4 cores) with the workflow's
settings (`--ba-chain --threads 4 --ba-jobs 4 --ba-threads 2`, rematch on),
scored by the official evaluator (`cvg/lamaria` at 238b6ca). CP Score;
pose recall at 1 m against the pseudo-GT in parentheses (sequence_5_11 has
no pseudo-GT).

| Sequence (track) | Mono VIO alone | **Mono + BA chain** | Bino + BA chain |
|---|---:|---:|---:|
| sequence_1_19 (Short) | 62.59 (65.8%) | **75.22** (99.7%) | 82.46 (100%); documented 82.25 |
| sequence_2_11 (Medium) | | **76.72** (89.3%) | 89.02 |
| sequence_3_17 (Long) | | **56.30** (35.8%) | 80.03 |
| sequence_4_10 (Low light) | | **37.92** (20.2%) | 67.81 |
| sequence_5_11 (Moving platform) | 47.25 (the chain had failed, before the fix) | **55.77** | 66.77 |
| Mean | | **60.39** | 77.18 |

The bino column is the documented chain on GitHub Actions
([lamaria_gpu_handover.md](lamaria_gpu_handover.md)). That table has no
sequence_4_10; its 67.81 is an earlier version of the chain
([lamaria_plan.md](lamaria_plan.md), whose same row has 88.16, 85.46, 71.43
and 60.32 on the other four), so the mean's bino side is approximate. sequence_1_19 was rerun here with the
changed scripts, at 82.46 against the documented 82.25, so the two-camera
path is unchanged.

- **One camera costs 7–30 points per sequence, and 17 on average.** The
  loss is small on the Short and Medium tracks (7 and 12 points). It is
  largest in low light (30 points): there cam0 alone often sees too little
  texture. On the Long track it loses 24 points.
- **The test set did not show this gap** ([below](#test-set)): there mono is
  within about a point of bino on three of the five challenges. One
  training sequence per track was too few to predict it.
- The chain matters more with one camera: on sequence_1_19 it lifts the
  monocular VIO from 62.6 to 75.2.

## Test set

[LaMAria submission run 35](https://github.com/rsasaki0109/visloc-rs/actions/runs/45524186572)
(`--mono`, all 63 test sequences, every job succeeded), scored by the
leaderboard. Mean score per challenge:

| | Short | Medium | Long | Low light | Moving platform | Mean |
|---|---:|---:|---:|---:|---:|---:|
| **Mono + IMU** | **65.3** | **56.2** | **55.2** | **61.6** | **25.9** | **52.9** |
| Bino + IMU, submission 4 (CPU only, no GPS) | 68.9 | 56.8 | 54.1 | 60.6 | 30.6 | 54.2 |
| Bino + IMU, submission 10 (the bino entry) | 69.7 | 56.8 | 54.1 | 61.1 | 47.3 | 57.8 |

Submission 4 is the closest comparison: the same pipeline on CPU, before
five Short sequences were rerun with learned features on a GPU (submission
5) and before the Moving platform track was bridged with GPS (submission 7);
see [lamaria_test_results.md](lamaria_test_results.md).

- **One camera costs 1.3 points on average.** Medium, Long and Low light
  are within a point of bino (Long +1.1, Low light +1.0 above submission 4).
  The loss is on the Short track (−3.6) and the Moving platform (−4.7).
- **The training sequences overstated the loss** (17 points there). Each
  track's training sequence is the one the bino pipeline was developed on,
  so bino scores high there; the test set behaves like the second training
  sequences ([lamaria_batch_ba.md](lamaria_batch_ba.md#two-sequences-per-track)).
- **A handful of sequences fail outright** (score below 10): sequence_1_2
  (7.5), sequence_4_3 (4.7), and sequence_5_5, 5_6 and 5_9 (1.4–2.0). They
  are where to look next: a mono VIO that diverges, or a ride the platform
  filter does not keep out.

Per sequence (CP score, CP recall at 1 m, pose recall at 5 m; the Moving
platform track has no pseudo-GT):

| Sequence | Score | CP@1m | R@5m | Sequence | Score | CP@1m | R@5m |
|---|---:|---:|---:|---|---:|---:|---:|
| sequence_1_1 | 67.8 | 75.0 | 100.0 | sequence_3_5 | 77.1 | 88.9 | 100.0 |
| sequence_1_2 | 7.5 | 0.0 | 13.2 | sequence_3_6 | 52.5 | 25.9 | 99.9 |
| sequence_1_3 | 42.9 | 15.4 | 94.6 | sequence_3_7 | 73.2 | 92.0 | 100.0 |
| sequence_1_4 | 52.3 | 20.0 | 99.6 | sequence_3_8 | 48.0 | 13.3 | 92.1 |
| sequence_1_5 | 81.5 | 100.0 | 100.0 | sequence_3_9 | 73.9 | 96.2 | 100.0 |
| sequence_1_6 | 78.0 | 100.0 | 100.0 | sequence_3_10 | 72.6 | 89.3 | 100.0 |
| sequence_1_7 | 74.1 | 85.7 | 100.0 | sequence_3_11 | 73.9 | 88.9 | 100.0 |
| sequence_1_8 | 75.3 | 92.9 | 99.8 | sequence_3_12 | 38.1 | 10.7 | 90.8 |
| sequence_1_9 | 92.2 | 100.0 | 100.0 | sequence_3_13 | 55.9 | 30.8 | 100.0 |
| sequence_1_10 | 73.5 | 93.3 | 100.0 | sequence_3_14 | 65.3 | 73.1 | 100.0 |
| sequence_1_11 | 82.3 | 100.0 | 100.0 | sequence_3_15 | 51.6 | 34.6 | 99.9 |
| sequence_1_12 | 69.1 | 84.6 | 100.0 | sequence_3_16 | 28.9 | 3.7 | 76.0 |
| sequence_1_13 | 63.5 | 57.1 | 95.7 | sequence_4_1 | 69.9 | 83.3 | 100.0 |
| sequence_1_14 | 82.8 | 100.0 | 100.0 | sequence_4_2 | 56.8 | 37.5 | 99.1 |
| sequence_1_15 | 81.4 | 100.0 | 99.8 | sequence_4_3 | 4.7 | 0.0 | 0.2 |
| sequence_1_16 | 33.0 | 14.3 | 67.6 | sequence_4_4 | 49.1 | 25.0 | 94.8 |
| sequence_1_17 | 60.7 | 53.3 | 100.0 | sequence_4_5 | 54.9 | 40.0 | 95.5 |
| sequence_1_18 | 57.6 | 60.0 | 100.0 | sequence_4_6 | 82.7 | 100.0 | 100.0 |
| sequence_2_1 | 45.3 | 19.0 | 92.6 | sequence_4_7 | 72.8 | 93.8 | 100.0 |
| sequence_2_2 | 35.7 | 0.0 | 75.4 | sequence_4_8 | 78.0 | 78.6 | 100.0 |
| sequence_2_3 | 45.2 | 23.5 | 83.8 | sequence_4_9 | 85.7 | 100.0 | 100.0 |
| sequence_2_4 | 45.2 | 21.1 | 99.9 | sequence_5_1 | 40.8 | 20.0 | – |
| sequence_2_5 | 40.7 | 5.0 | 99.7 | sequence_5_2 | 55.3 | 33.3 | – |
| sequence_2_6 | 62.7 | 56.3 | 99.8 | sequence_5_3 | 24.8 | 3.7 | – |
| sequence_2_7 | 57.9 | 37.5 | 100.0 | sequence_5_4 | 31.4 | 13.6 | – |
| sequence_2_8 | 79.3 | 93.8 | 100.0 | sequence_5_5 | 1.4 | 0.0 | – |
| sequence_2_9 | 66.1 | 77.8 | 100.0 | sequence_5_6 | 1.4 | 0.0 | – |
| sequence_2_10 | 84.1 | 100.0 | 100.0 | sequence_5_7 | 26.2 | 6.7 | – |
| sequence_3_1 | 41.3 | 14.8 | 94.4 | sequence_5_8 | 60.0 | 60.0 | – |
| sequence_3_2 | 60.2 | 38.5 | 100.0 | sequence_5_9 | 2.0 | 0.0 | – |
| sequence_3_3 | 19.2 | 0.0 | 39.4 | sequence_5_10 | 16.1 | 0.0 | – |
| sequence_3_4 | 51.6 | 29.6 | 100.0 |  | | |  |

## Low light: what did not help

On sequence_4_10 the chain lifts the monocular VIO only from 35.47 to
37.92: in low light it runs the online windows and no batch pass
([lamaria_batch_ba.md](lamaria_batch_ba.md#low-light-features-after-denoising)).
Two tries, by hand on the same run, did not help:

| sequence_4_10, cam0 + IMU | CP Score | Pose recall @ 1 m |
|---|---:|---:|
| VIO alone | 35.47 | 11.6% |
| **The chain (online windows)** | **37.92** | 20.2% |
| + the day's batch pass after it (first solve) | 28.11 | 0.4% |
| + its second solve with the revisits | 28.31 | 0.4% |
| VIO alone on cam1 instead (`config.vio_mono_camera: 1`) | 6.25 | 2.7% |

- The batch pass with one camera's denoised low-light keypoints pulls the
  trajectory further off than the online windows leave it, so the chain's
  rule (no pass in low light) holds for mono too.
- On cam1 the monocular VIO diverges (3,400 km of trajectory over a 1.5 km
  walk). cam0 stays the camera.

## A bigger VIO window: mixed

sequence_1_2 (test, 7.5 on the leaderboard against bino's 29.0) does not
diverge. Against the recording's own GPS (diagnosis only, not used for the
submission) its chain result sits 11.4 m from the fixes at the median,
against 2–3 m for sequences that score well, with the gap opening after
about four minutes. With the VIO's window enlarged to 10 states and 30
keyframes (`euroc_config_big_window.json`; a low-light copy of it on track
4), sequence_1_2 comes to 5.8 m. On the training sequences, scored against
the pseudo-GT, it is a gain on one and a loss on two:

| | Default window | 10 states / 30 keyframes |
|---|---:|---:|
| sequence_1_2 (test), median distance to GPS | 11.4 m | **5.8 m** |
| sequence_1_19, CP Score | 75.22 | **78.98** |
| sequence_2_11, CP Score | **76.72** | 57.67 |
| sequence_4_10, CP Score | **37.92** | 20.60 |

On sequence_2_11 the loss is already in the VIO: the pass before the rematch
scores 54.60. The window stays at the default; sequence_3_17 and 5_11 were
not run. Windowed scale against GPS is no measure here: on training
sequences whose scale the pseudo-GT shows within 1 %, 120 s windows of GPS
swing it by 10–27 %.

## Low light: a window that jumps gates early

sequence_4_3 (test, 4.7 on the leaderboard, pose recall at 5 m 0.2 %; bino
37.0) is a 23-minute night walk of 1.3 km. The VIO alone keeps it within
7.6 m of the recording's GPS at the median (diagnosis only); the online
windows leave the first six minutes 25 m off (90th percentile 25.7 m). One
window does it: keyframes 300:780, whose new keyframes (360–390 s) bring an
initial cost per observation five times the window before's. Its first step,
taken before the 3 px outlier gate (from iteration 8), shortens the walk of
360–380 s from the 26 m GPS shows (the VIO has 18 m) to 8 m. The revisits of
the sequence are all short (about 40 s apart) and agree with the VIO; none is
false.

The online windows' `--jump-gate 3` (on for a monocular run in low light):
a window whose initial cost per observation exceeds three times the window
before's gates at 3 px from its first iteration. It fires once on
sequence_4_3 and never on the two Low light training sequences, whose largest
jumps are 1.5× and 2.4×:

| | sequence_4_3, distance to GPS (median / 90 %) | sequence_4_10, CP Score | sequence_4_11, CP Score |
|---|---|---:|---:|
| Online windows | 6.3 m / 25.7 m | 37.92 | 50.76 |
| **+ `--jump-gate 3`** | **3.1 m / 7.4 m** | 37.92 | 50.78 |
| Every window gated from iteration 0, 3 px | 3.7 m / 7.4 m | 3.58 | |
| — 5 px | 3.0 m / 6.8 m | 36.31 | 44.45 |
| — 6 px | 3.1 m / 7.6 m | 41.52 | 43.10 |
| — 7 px | 8.3 m / 13.3 m | 39.88 | 49.10 |

Gating every window early is no rule: in the first windows of sequence_4_10
the VIO is still far off, a 3 px gate drops a fifth to a third of the
observations, and the solve comes apart; the thresholds that help one
training sequence cost the other.

## Reproduce

```bash
cargo build --locked --profile release-rt --example basalt_euroc_vio_demo
python scripts/run_lamaria_test_submission.py --split training --sequences sequence_1_19 --mono \
  --vio-exe target/release-rt/examples/basalt_euroc_vio_demo --ba-chain \
  --threads 4 --ba-jobs 4 --ba-threads 2 --work-dir work --slam-dir slam --log-dir logs
# scored as in lamaria_gpu_handover.md (evaluate_wrt_control_points.py, evaluate_wrt_pgt.py)
```

The test set: Actions > LaMAria submission > Run workflow, split `test`,
`driver_args` `--mono`. The zip goes to the leaderboard by hand, as a
separate method whose sensors are mono + imu. The run's `submission`
artifact downloads as a zip around `submission.zip`: upload the inner one,
whose root holds only `slam/`. The outer one fails the evaluation.

## Limits

- One training sequence per track, each run once; they overstated the
  loss from one camera (17 points against 1.3 on the test set).
- Nothing was tuned for one camera: the VIO config, the chain's options and
  the low-light options are the bino ones.
