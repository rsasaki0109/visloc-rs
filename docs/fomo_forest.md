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

## Open

- **Online mapper / loop closure:** blue is a loop, so the online mapper
  (`basalt_euroc_online_slam_demo`) should close it. Not measured yet.
- **Startup run-away:** on a 10 s slice from rest, the VIO's speed overshoots
  by up to 2x in the first seconds of hard acceleration. Over the whole run
  it does not show, but `StaticHold` (only wired into
  `basalt_euroc_localize_demo`) is the candidate fix.
- **Other routes and night:** the other FoMo routes (green and magenta are
  more forest; orange and yellow are 2-2.7 km) and the 2025-11-03 night/rain
  deployment.
