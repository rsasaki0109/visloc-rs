# Stereo-inertial VIO in a boreal forest, summer and winter (FoMo)

Measured 2026-11-28 on [FoMo](https://fomo.norlab.ulaval.ca) (Foret
Montmorency; Boxan et al., IJRR 2026; CC BY 4.0, public on AWS Open Data):
a wheeled robot driving the same forest trails over a year, from snow and
-19 C in January to dense vegetation in August. Here: the ~555 m "blue"
loop (forest trails, an open snowfield or meadow, buildings at the start and
end) in four seasons, ZED X stereo (rectified, downscaled to 960x600, about
9 Hz) + VectorNav IMU (200 Hz), against the PPK-GNSS ground truth.

## Results

VIO only (`basalt_euroc_vio_demo`, no mapper, no loop closure). ATE is the
RMSE of base_link positions after an SE(3) fit to the ground truth; drift
is the end-point error after aligning only the first 10% of the run, as a
share of the distance driven. Every run covers > 99.8% of the recording.
Runs are deterministic (the same command gives the same number).

| Sequence | Conditions | Length | Baseline ATE | **Forest config** ATE | Baseline drift | Forest drift |
|---|---|---:|---:|---:|---:|---:|
| 2025-01-30 (tuned on) | deep snow, low sun into the camera | 569 m | 4.76 m | **1.41 m** | 1.67% | 0.44% |
| 2025-08-20 (tuned on) | summer, gravel and grass | 554 m | **1.22 m** | 1.44 m | 0.25% | 0.21% |
| 2025-03-10 (held out) | late-winter snow | 559 m | 1.74 m | **1.44 m** | 0.46% | 0.45% |
| 2025-10-14 (held out) | autumn, no snow | 553 m | **0.68 m** | 0.88 m | 0.28% | 0.31% |
| mean of the four | | | 2.10 m | **1.29 m** | | |

- **Baseline:** `configs/basalt/variants/fomo/fomo_config_baseline.json`,
  Basalt's EuRoC config unchanged.
- **Forest config:** `configs/basalt/variants/fomo/fomo_config_forest.json`.
  It changes two things:
  - KLT pyramid `optical_flow_levels` 3 -> 4.
  - KLT forward-backward check `optical_flow_max_recovered_dist2` 0.04 -> 0.25
    (0.2 px -> 0.5 px, the same value LaMAria's low-light config uses).

Read honestly, the forest config is a **snow** config. On both snow runs it
lowers the error, by 3.4x on the tuned January run and by 17% on the
held-out March run. On both snow-free runs it costs 0.2 m. The worst case
over the four seasons drops from 4.76 m to 1.44 m. With snow on the ground
use the forest config; without it the default config is slightly better.

Why snow needs it: in the open snowfield (January, 280-360 s) the bottom
half of the image is textureless snow, and the sun shines straight into
the lens. Tracks survive only on the band of trees, about 50% of tracked
points are rejected there, and the baseline under-estimates the distance
driven by about 10%. With one more pyramid level, KLT follows the large
image motion of the near ground. The looser check keeps the low-contrast
tracks that the 0.2 px test drops.

## What did not help (January / August, ATE SE(3), baseline 4.76 / 1.22 m)

| Lever | January | August |
|---|---:|---:|
| Big window (`vio_max_states` 10, `vio_max_kfs` 30), which helped on LaMAria | 6.75 | 1.14 |
| Detection grid 50 -> 30 px | 4.71 | 1.33 |
| `vio_init_accel_average_s` 0.5 | 4.86 | 1.26 |
| `optical_flow_imu_seed_rotation` | 4.85 | 1.25 |
| Pyramid 5 levels (vs 4) | 2.26 | 1.49 |
| 4 levels + patch pattern 52 | 5.37 | 1.40 |
| CLAHE on the images (8x8 tiles, clip 2), baseline config | 5.57 | 1.27 |
| CLAHE + forest config | 1.74 | 1.40 |
| Forest config + `optical_flow_max_iterations` 7 | 1.39 | 1.37 |
| Forest config + `vio_obs_std_dev` 1.0 | 1.32 | 1.40 |

The last two are within 0.1 m of the forest config, so they are not
adopted. On snow, CLAHE amplifies sensor noise along with the faint texture
of the snow, and it is worse with either config.

## A calibration trap: the published stereo rotation

FoMo's `calib/transforms.json` gives the ZED X right camera a 0.5 deg
rotation (mostly yaw) from the left. The images are already rectified, and
that rotation is the factory one from before rectification. Applied on top
of the rectified images, it shifts every disparity by about 3 px at 960 px
width. The VIO then comes out 1.4x too long on every segment: on August,
Sim(3) scale 0.71 and ATE 28.5 m instead of 1.22 m. The IMU cannot correct
it: switching to the Xsens IMU (scale 0.71) or giving the IMU 100x noise
(scale 0.68) changes nothing, because the stereo depth itself is wrong.

`scripts/fomo_to_euroc.py` therefore puts the right camera at the
published baseline's length (0.1197 m) along +x with no rotation. Two
independent checks hold with that:
- Stereo block matching puts the ground 1.08 m below base_link, where the
  published footprint is 1.114 m.
- Stereo + PnP displacements match the ground truth at a median ratio of
  0.996.

`--published-stereo-rotation` restores the published rotation, for
comparison only.

The remaining extrinsics are used as published, after two checks:
- **Camera-IMU rotation:** a hand-eye check of camera rotations (essential
  matrix) against the VectorNav gyro agrees to 1.5 deg, with a rate ratio
  of 0.98.
- **Frame convention:** a `transforms.json` entry {to: A, from: B} is
  T_B_A. With that reading, gravity puts the camera 18 deg pitched down,
  which matches the horizon in the images.

## Reproduce

```bash
# Convert (streams from s3://fomo-dataset over HTTPS, no account; about 20 min per run)
python3 scripts/fomo_to_euroc.py --trajectory data/2025-01-29/blue_2025-01-30-10-08 --out /data/fomo/winter_blue
python3 scripts/fomo_to_euroc.py --trajectory data/2025-08-20/blue_2025-08-20-11-25 --out /data/fomo/summer_blue
python3 scripts/fomo_to_euroc.py --trajectory data/2025-03-10/blue_2025-03-10-16-59 --out /data/fomo/march_blue
python3 scripts/fomo_to_euroc.py --trajectory data/2025-10-14/blue_2025-10-14-11-46 --out /data/fomo/october_blue
python3 scripts/fomo_to_euroc.py --trajectory data/2025-11-03/blue_2025-11-03-18-37 --out /data/fomo/night_blue
# Other routes: data/2025-01-29/green_2025-01-30-14-45, data/2025-08-20/green_2025-08-20-11-39,
#               data/2025-01-29/magenta_2025-01-30-15-03, data/2025-08-20/magenta_2025-08-20-12-41

cargo build --release --example basalt_euroc_vio_demo --features basalt-lm-workspace-reuse
S=/data/fomo/winter_blue
target/release/examples/basalt_euroc_vio_demo --euroc-dir $S --calibration $S/calibration.json \
  --config configs/basalt/variants/fomo/fomo_config_forest.json \
  --out-dir target/fomo_winter --no-trace --no-marg-data
python3 scripts/fomo_eval.py --sequence $S --estimate target/fomo_winter/trajectory.tum
```

A run is faster than real time on 4 CPU cores (August: 160 s for 350 s).
Each converted run is 2.5-3.5 GB of grey PNGs. The dataset images stay
outside the repository and remain under FoMo's CC BY 4.0 terms.

## Online mapper / loop closure

`basalt_euroc_online_slam_demo` (VIO + online NFR mapper + pose-graph loop
closure), `--optimize-every-k 100 --periodic-iterations 4`, scoring
`trajectory_online.tum`. Online ATE SE(3) in m, accepted loops in brackets.

| Sequence | Baseline | Forest | Baseline + loose loops | **Forest + loose loops** |
|---|---:|---:|---:|---:|
| 2025-01-30 snow | 3.03 (32) | 0.99 (33) | 2.95 (98) | **0.95** (98) |
| 2025-08-20 summer | 1.27 (0) | 1.49 (0) | 1.27 (6) | **1.38** (7) |
| 2025-03-10 snow | 1.14 (4) | 1.68 (4) | 1.10 (9) | **1.66** (5) |
| 2025-10-14 autumn | 0.72 (0) | 0.92 (0) | 0.49 (24) | **0.70** (13) |
| mean | 1.54 | 1.27 | 1.45 | **1.17** |

- **Loose loops** = `configs/basalt/variants/fomo/fomo_config_forest_loop.json`
  plus `--match-top-k 15`. Over the forest config it changes the mapper:
  - `mapper_frames_to_match_threshold` 0.04 -> 0.01 (BoW score),
  - `mapper_min_matches` 20 -> 15,
  - `mapper_detection_num_points` 800 -> 1500.
- The default thresholds close **no** loop in August or October. On those
  runs the robot comes back to the start only in the last 4-7 s, so there
  are only a few revisit keyframes, and their BoW scores and match counts
  fall just under the defaults. `--match-top-k 15` alone does not help.
- The loose thresholds found no false loop: across the eight runs, 2 of 262
  geometrically verified candidates were rejected, and every run is equal or
  better than with the default thresholds. End-point drift is 0.03-0.15% on three runs (March with
  the forest config: 0.57%).
- Forest + loose loops is the best single setting on blue (mean 1.17 m,
  worst 1.66 m). On green and magenta it does not hold (see below). Per season, the baseline config with loose loops is better
  on March (1.10 m) and October (0.49 m), so the snow/no-snow choice is not
  clean once the mapper runs.

```bash
cargo build --release --example basalt_euroc_online_slam_demo --features basalt-lm-workspace-reuse
target/release/examples/basalt_euroc_online_slam_demo --euroc-dir $S --calibration $S/calibration.json \
  --config configs/basalt/variants/fomo/fomo_config_forest_loop.json --match-top-k 15 \
  --optimize-every-k 100 --periodic-iterations 4 --out-dir target/fomo_winter_slam
python3 scripts/fomo_eval.py --sequence $S --estimate target/fomo_winter_slam/trajectory_online.tum
```

## Night and rain (2025-11-03, 18:37)

The same blue loop driven after dark in the rain: 583 m, 344 s. The only
light is the robot's headlights. The near ground is motion-blurred,
everything beyond about 15 m is black, and large raindrops sit on the
lens (a different set at the start and at the end).

| Run | ATE SE(3) | Drift | Loops |
|---|---:|---:|---:|
| VIO, baseline config | 4.13 m | 2.26% | - |
| **VIO, forest config** | **1.35 m** | **0.56%** | - |
| Online SLAM, baseline + loose loops | 1.96 m | 0.90% | 0 |
| Online SLAM, forest + loose loops | 1.57 m | 0.89% | 0 |

- The forest config cuts the error by 3x, as on the January snowfield. The
  shared trait is a scene where good texture sits only in part of the
  image and the ground moves fast through it. So it is a hard-conditions
  config more than a snow config.
- No loop closes, not even a verified candidate. The robot returns within
  0.5 m of the start, as in August. The night revisit frames share little
  besides a lit fence and building corner: the headlights light different
  ground, and different raindrops cover the middle of the image.
- Without loops the online trajectory is 0.26 m worse than the VIO it is
  built from. At night, use the VIO output.

## Other routes: green and magenta (held out)

Two more forest-heavy routes, January and August, with no further tuning.
VIO ATE SE(3):

| Route | Length | Returns to start | Baseline | Forest config |
|---|---:|---|---:|---:|
| green 2025-01-30 | 216 m | 7 m away | 1.49 m | 1.44 m |
| green 2025-08-20 | 816 m | 6 m away | 1.59 m | 1.63 m |
| magenta 2025-01-30 | 868 m | no (one-way) | 2.23 m | 2.23 m |
| magenta 2025-08-20 | 810 m | 5 m away | 2.11 m | **1.60 m** |

Over all nine runs on this page (blue x4, blue at night, green x2, magenta
x2), the forest config averages **1.49 m** against **2.22 m** for the
baseline. It is never worse by more than 0.22 m, and its worst run is 2.23 m
against 4.76 m. This makes it the recommended default for FoMo-like
forest driving, not only for snow.

Online SLAM on these routes (forest config throughout; loops accepted in
brackets):

| Route | VIO in the SLAM run | Default loop thresholds | Loose loops | Loose + `--loop-min-travelled-m 100` |
|---|---:|---:|---:|---:|
| green January | 1.58 m | - | 1.58 m (7) | 1.56 m (0) |
| green August | 1.64 m | - | 1.65 m (0) | - |
| magenta January | 2.22 m | 2.24 m (99) | 2.31 m (563) | 2.37 m (7) |
| magenta August | 1.83 m | 1.84 m (0) | 1.93 m (7) | 1.91 m (0) |
| blue January | 1.32 m | 0.99 m (33) | 0.95 m (98) | 0.95 m (93) |
| blue August | 1.56 m | 1.49 m (0) | 1.38 m (7) | 1.60 m (5) |

(The VIO inside the SLAM demo is not bit-identical to `basalt_euroc_vio_demo`,
so compare each row with its own VIO column.)

- **On green and magenta the mapper does not help:** with default
  thresholds the online trajectory equals the VIO, and the loose
  thresholds cost 0.07-0.10 m on magenta. The trade is still worth it:
  loose thresholds gain 0.11 m on blue August and 0.22 m on blue October.
- **Most loose "loops" on magenta are near-sequential:** pairs a median 26 m
  of driving apart, just past the 30-keyframe loop gap, along straight
  trail (January: 563 accepted).
- **The distance gate is implemented but not recommended:**
  `--loop-min-travelled-m`, default off, rejects such pairs as
  `[loop] reject-short`. It removes them (January magenta: 563 -> 7 loops),
  but the error does not drop, and on blue August it rejects 2 pairs that
  were helping (1.38 -> 1.60 m). So the short pairs were not what cost
  accuracy. Leave the gate off.
- **Correction:** an earlier version of this page put the magenta loss at
  2.89 / 2.58 m and blamed the short loops. Those runs used the *baseline*
  config with loose loops, so they are not comparable with this table.

## Open
## Open

- **Startup run-away:** on a 10 s slice from rest, the VIO's speed overshoots
  by up to 2x in the first seconds of hard acceleration. Over the whole run
  it does not show, but `StaticHold` (only wired into
  `basalt_euroc_localize_demo`) is the candidate fix.
- **Other routes:** orange and yellow (2-2.7 km). The 2025-11-03 night
  deployment has all six routes.
- **Night loop closure:** the BoW place recognition does not survive
  headlights + rain. That needs a different descriptor, not looser
  thresholds.
