# LaMAria with one camera (mono + IMU)

The LaMAria leaderboard lists each method's sensors: mono, bino, imu. The
visloc-rs entry so far is bino + IMU (both SLAM cameras). This page runs the
same pipeline on the left SLAM camera (cam0) and the IMU alone, for a
mono + IMU entry. On 2026-10-17 the best mono + IMU methods on the
leaderboard averaged 33.5 (AnonSLAM) and 33.4 (Mighty Camera) over the five
challenges ([lamaria_plan.md](lamaria_plan.md)).

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
- **Mono + IMU is still far above the leaderboard's mono + IMU entries.**
  The bino chain scored 64.35 per track on the ten training sequences and
  57.8 on the test set. If mono loses on the test set the 17 points it
  loses here, a mono + IMU submission would land around 41, against 33.5
  for the best mono + IMU method listed. Only a test-set run says how it does there.
- The chain matters more with one camera: on sequence_1_19 it lifts the
  monocular VIO from 62.6 to 75.2.

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
separate method whose sensors are mono + imu.

## Limits

- One training sequence per track, each run once. The bino chain's
  training scores ran 6–7 points above its test scores.
- Nothing was tuned for one camera: the VIO config, the chain's options and
  the low-light options are the bino ones.
