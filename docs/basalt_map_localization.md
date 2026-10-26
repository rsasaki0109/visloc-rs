# Localizing against a saved VI-SLAM map (EuRoC)

`basalt_euroc_online_slam_demo --save-map` saves the stereo-inertial SLAM's
final map; `basalt_euroc_localize_demo` then localizes each cam0 image of a
sequence against it (mapper ORB frontend, Hamming matching, P3P RANSAC). This
page measures that path on EuRoC Machine Hall: a map built from **MH_01**,
queried with MH_01 itself and with **MH_02**, a different flight through the
same hall.

**Headline.** Against the MH_01 map, MH_02 localizes **87.4%** of its 3,040
images at **2.0 cm** RMSE (1.2 cm median) against ground truth, with
`--min-inliers 12` **94.1%** at 2.2 cm. MH_01 against its own map localizes
99.6% and agrees with the SLAM trajectory to 2.4 mm (median), so its 1.6 cm
error against ground truth is the map's own accuracy.

## Setup

- **Data.** EuRoC `MH_01_easy` and `MH_02_easy`, ASL format, from the official
  `machine_hall.zip`. ETH's Research Collection rate-limited this machine, so
  the archive was read from the Hugging Face copy
  `GlowBond/EuRoC_MAV_Dataset`, whose LFS SHA-256
  `5ed7d07903f8d19b6c8808e2ae8a0872b281f6e34ef5497023b8ac58c3de0f6f` is the
  official archive's (see [nonregression_20260830.md](nonregression_20260830.md));
  only the two nested sequence zips were fetched (HTTP range requests), and
  `unzip -t` passed on both.
- **Code.** `main` at `bd09614` (PR #343), release build. Map: README
  configuration (`configs/basalt/variants/official_euroc_ds/`), default
  online-mapper settings.
- **Machine.** 4-core Intel Xeon @ 2.10 GHz, 15 GB RAM. At this core count the
  SLAM ran at 0.84× real time (MH_01: 310 s); the localizer's times below are
  for the same machine.

## Map

| | |
|---|---|
| Keyframes / images | 566 / 1,132 (both cameras) |
| Landmarks / observations | 15,483 / 271,091 |
| Size on disk | 20 MB (8.0 MB descriptors, 9.4 MB `images.txt`) |
| `OPENCV_FISHEYE` fit to Double Sphere | ≤ 0.0003 px (cam0), ≤ 0.000003 px (cam1) |
| SLAM trajectory vs GT (SE(3) fit) | 1.51 cm RMSE, 1.18 cm median |

## Results

Every pose is an IMU pose. **Fixed alignment**: the map→ground-truth
transform is fitted once, on the MH_01 SLAM trajectory against MH_01 ground
truth, and applied unchanged to the localizations; nothing is fitted to the
scored poses. (Fitting on the MH_02 localizations instead gives the same
2.0 cm, so the two ground-truth sequences share a frame and the fixed
transform adds no bias.)

| Query → MH_01 map | Localized | vs | Translation RMSE / median / p99 / max | Rotation median |
|---|---|---|---|---|
| MH_01 | 3,666 / 3,682 (99.6%) | SLAM trajectory, no alignment | 0.58 / 0.24 / 2.3 / 7.2 cm | 0.03° |
| MH_01 | 3,666 / 3,682 (99.6%) | ground truth, fixed alignment | 1.59 / 1.21 / 4.6 / 7.2 cm | — |
| **MH_02** | **2,656 / 3,040 (87.4%)** | ground truth, fixed alignment | **2.01 / 1.24 / 5.3 / 30 cm** | **0.43°** |

The MH_01 rotation against ground truth is left out: the SLAM trajectory
itself is 0.9° off there after a position-only fit, so it measures the
alignment, not the localizer.

Latency, MH_02: 74 ms median, 89 ms p90 per image (global matching against all
15,483 landmarks every frame; ORB extraction included).

### Parameter sweep (MH_02 → MH_01 map)

| Options | Localized | RMSE / median / p99 / max | > 5 cm | > 10 cm | > 25 cm | Latency median |
|---|---|---|---|---|---|---|
| defaults (`--min-inliers 20`) | 87.4% | 2.01 / 1.24 / 5.3 / 30 cm | 37 | 3 | 1 | 74 ms |
| `--min-inliers 12` | **94.1%** | 2.15 / 1.31 / 6.3 / 30 cm | 59 | 4 | 1 | 75 ms |
| `--prior-radius-m 5` | 88.0% | 2.14 / 1.39 / 5.8 / 30 cm | 48 | 6 | 1 | 72 ms |
| `--prior-radius-m 5 --min-inliers 12` | 94.7% | 3.92 / 1.49 / 11.9 / 84 cm | 139 | 37 | 10 | 46 ms |

- `--min-inliers 12` recovers 206 more images for one more pose over 10 cm;
  the extra misses are mostly 5–10 cm. It is the better setting on this pair.
  The default stays at 20 until more sequences confirm it.
- `--prior-radius-m` does not pay off: matching only the landmarks near the
  previous pose barely raises the localized rate, and combined with a low
  inlier threshold it accepts wrong poses (up to 84 cm). With fewer
  candidates the nearest/second-nearest ratio test also passes wrong
  matches, and 12 inliers no longer reject them.

### Where MH_02 fails

The 384 failures form 35 contiguous stretches, the longest 114 frames
(5.7 s, around t = 85 s), typically with 10–14 inliers. Measured with ground
truth, the failing images are farther from any MH_01 pose looking the same
way (camera axes within 30°): 0.55 m median / 1.89 m p90, against 0.29 m /
0.77 m for localized images. Most failures are therefore viewpoints the map
did not cover; the single-scale corner/ORB frontend shared with the mapper
probably adds to this under viewpoint change (not isolated here).

## Reproduce

```bash
D=/path/to/euroc     # MH_01_easy/ and MH_02_easy/, each with mav0/
cargo build --release --features basalt-lm-workspace-reuse \
  --example basalt_euroc_online_slam_demo --example basalt_euroc_localize_demo

target/release/examples/basalt_euroc_online_slam_demo \
  --euroc-dir $D/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir runs/mh01_slam --save-map runs/mh01_map

target/release/examples/basalt_euroc_localize_demo \
  --map runs/mh01_map --euroc-dir $D/MH_02_easy --out-dir runs/mh02_on_mh01

python3 scripts/evaluate_map_localization.py --mode fixed \
  --estimate runs/mh02_on_mh01/localization.tum \
  --reference $D/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/mh01_slam/trajectory_online.tum \
  --align-ground-truth $D/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --queries 3040
```

`--mode direct --reference runs/mh01_slam/trajectory_online.tum` compares a
localization with the SLAM trajectory in the map frame without alignment.

## Limits

- One map and one cross-session query, both Machine Hall "easy" flights; no
  Vicon-room or "difficult" sequences yet.
- cam0 only; each image is localized independently (no tracking or motion
  model between frames).
- The online mapper's background optimization depends on thread timing, so
  a rerun may build a slightly different map.
