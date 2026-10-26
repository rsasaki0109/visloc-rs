# Localizing against a saved VI-SLAM map (EuRoC)

`basalt_euroc_online_slam_demo --save-map` saves the stereo-inertial SLAM's
final map; `basalt_euroc_localize_demo` then localizes each cam0 image of a
sequence against it (mapper ORB frontend, Hamming matching, P3P RANSAC, a pose
accepted with at least `--min-inliers` inliers, default 30). This page
measures that path on all of EuRoC: one map per environment (MH_01, V1_01,
V2_01), queried with the other eight sequences of the same environment.

**Headline.** With the default threshold of 30 inliers, no accepted pose in
any of the eight query sequences is more than 25 cm off; across all of them
0.3% are more than 10 cm off. MH_02 against the MH_01 map localizes **81.2%**
of its images at **1.8 cm** RMSE against ground truth. How many images
localize is set mostly by how much of the query flight the map covers: from
81% (MH_02) down to under 10% for flights through parts of the room the map
never saw. MH_01 against its own map localizes 98.4% and agrees with the SLAM
trajectory to 2.4 mm (median).

## Setup

- **Data.** EuRoC ASL format. ETH's Research Collection rate-limited this
  machine, so the sequences were read from the Hugging Face copy
  `GlowBond/EuRoC_MAV_Dataset`; only the nested sequence zips were fetched
  (HTTP range requests) and `unzip -t` passed on every one.
  - Machine Hall: the copy's `machine_hall.zip` has the official archive's
    SHA-256 `5ed7d079…de0f6f` (see
    [nonregression_20260830.md](nonregression_20260830.md)).
  - Vicon rooms: no official hash is on record. The nested `V1_01_easy.zip`
    inflated from the copy's `vicon_room1.zip` has the same SHA-256
    (`a920fe5b…ba45fa`) as the separately uploaded
    `pepijn223/euroc-mirror`.
- **Code.** `main` at `1d655cd`, release build. Maps: README configuration
  (`configs/basalt/variants/official_euroc_ds/`), default online-mapper
  settings.
- **Machine.** 4-core Intel Xeon @ 2.10 GHz, 15 GB RAM.
- **Scoring.** Every pose is an IMU pose; translation is compared with ground
  truth. The map→ground-truth transform is fitted once, on the map's own SLAM
  trajectory against its ground truth, and applied unchanged to the
  localizations (`--mode fixed`): nothing is fitted to the scored poses.
  Rotation against ground truth is not used: the SLAM trajectories
  themselves differ from the ground-truth orientation by a median 0.9°
  (MH_01) and 5.5° (V1_01) after alignment, so it measures the reference,
  not the localizer.

## Maps

| Map | Keyframes / images | Landmarks | On disk | SLAM ATE (RMSE) | SLAM time |
|---|---|---|---|---|---|
| MH_01 | 566 / 1,132 | 15,483 | 20 MB | 1.51 cm | 310 s (0.84× real time) |
| V1_01 | 416 / 832 | 14,053 | 15 MB | 3.54 cm | 169 s (1.07×) |
| V2_01 | 343 / 686 | 8,242 | 9.3 MB | 1.54 cm | 124 s (1.07×) |

The cameras are saved as COLMAP `OPENCV_FISHEYE` fitted to the Double Sphere
calibration, at most 0.0003 px off over the 57° half field of view.

## Results (default `--min-inliers 30`)

| Query → map | Localized | RMSE / median / max | Query flight within 0.5 m of a map viewpoint | Latency (median) |
|---|---|---|---|---|
| MH_01 → MH_01 (same flight) | 98.4% of 3,682 | 1.56 / 1.21 / 7.2 cm | — | 77 ms |
| MH_02 → MH_01 | **81.2%** of 3,040 | 1.83 / 1.19 / 16.8 cm | 66% | 75 ms |
| MH_03 → MH_01 | 47.4% of 2,700 | 1.98 / 1.23 / 10.3 cm | 33% | 73 ms |
| MH_04 → MH_01 | 4.6% of 2,033 | 4.51 / 3.12 / 11.9 cm | 0.1% | 69 ms |
| MH_05 → MH_01 | 7.8% of 2,273 | 5.12 / 4.78 / 17.1 cm | 0.1% | 67 ms |
| V1_02 → V1_01 | 28.5% of 1,710 | 2.84 / 2.47 / 6.9 cm | 48% | 62 ms |
| V1_03 → V1_01 | 10.9% of 2,149 | 4.02 / 2.30 / 21.4 cm | 39% | 62 ms |
| V2_02 → V2_01 | 20.3% of 2,348 | 2.14 / 1.57 / 9.0 cm | 36% | 48 ms |
| V2_03 → V2_01 | 0.4% of 1,922 | 3.96 / 2.55 / 7.5 cm | 30% | 43 ms |

MH_01 against its own map agrees with the SLAM trajectory to 0.58 cm RMSE
(2.4 mm median) without any alignment; its 1.56 cm against ground truth is
the map's own accuracy. Latency includes ORB extraction and global matching
against every landmark of the map, so it scales with the map size.

**Coverage.** "Within 0.5 m of a map viewpoint" is the share of the query's
ground-truth poses that have a map-flight pose within 0.5 m looking the same
way (camera axes within 30°). MH_04 and MH_05 fly through parts of the hall
MH_01 never saw (median 7 m from any MH_01 viewpoint), and their few
localized images are the ones near it. Coverage does not explain the
"difficult" Vicon flights: V1_03 and V2_03 come near map viewpoints for
30–39% of the flight but localize 11% and 0.4%; their fast motion and blur
are the likely cause (not isolated here).

## Choosing the inlier threshold

Each query was run once at `--min-inliers 12`. Without `--prior-radius-m`
every image is localized independently with a fixed RANSAC seed, so that run
contains the result of every higher threshold: a pose accepted at N is one
whose logged inlier count is at least N. (On MH_02, filtering the 12-inlier
run at 20 reproduces a separate 20-inlier run exactly: the same 2,656 poses.)

All eight cross-session queries, by threshold:

| `--min-inliers` | 12 | 20 | 25 | **30** | 40 |
|---|---|---|---|---|---|
| Localized | 40.6% | 34.0% | 31.1% | **28.7%** | 23.8% |
| Poses > 10 cm off | 243 | 103 | 38 | **15** | 8 |
| Poses > 25 cm off | 53 | 12 | 2 | **0** | 0 |

And by the accepted pose's inlier count:

| Inliers | Poses | > 10 cm off | > 25 cm off |
|---|---|---|---|
| 12–19 | 1,196 | 11.7% | 3.4% |
| 20–29 | 950 | 9.3% | 1.3% |
| 30–49 | 1,586 | 0.9% | 0 |
| ≥ 50 | 3,471 | 0.03% | 0 |

Wrong poses are those supported by few inliers. Below 30 they reach metres:
at the previous default of 20, MH_03 accepted a pose 1.6 m off and V2_02 one
3.6 m off (RMSE 6.6 cm and 15.0 cm, against 2.0 cm and 2.1 cm at 30). The
default is therefore 30. It localizes fewer images (MH_02: 81.2% instead of
87.4%), but a localizer that returns no pose is safer than one that
confidently returns the wrong place.

An earlier version of this page, measured on MH_02 alone, preferred
`--min-inliers 12` (94.1% of MH_02 at 2.2 cm). The other sequences reverse
that: at 12, MH_03 accepts 29 poses more than 25 cm off (up to 2.1 m) and
V2_02 20 (up to 4.7 m).

### `--prior-radius-m` (MH_02, previous thresholds)

Matching only the landmarks within 5 m of the previous pose was measured on
MH_02 at 20 and 12 inliers:

| Options | Localized | RMSE / max | > 10 cm | > 25 cm | Latency median |
|---|---|---|---|---|---|
| `--min-inliers 20` | 87.4% | 2.01 / 30 cm | 3 | 1 | 74 ms |
| `--min-inliers 20 --prior-radius-m 5` | 88.0% | 2.14 / 30 cm | 6 | 1 | 72 ms |
| `--min-inliers 12 --prior-radius-m 5` | 94.7% | 3.92 / 84 cm | 37 | 10 | 46 ms |

It barely raises the localized rate. Combined with a low inlier threshold it
accepts wrong poses, since with fewer candidates the nearest /
second-nearest ratio test also passes wrong matches.

## Reproduce

```bash
D=/path/to/euroc     # MH_01_easy/, MH_02_easy/, ... each with mav0/
cargo build --release --features basalt-lm-workspace-reuse \
  --example basalt_euroc_online_slam_demo --example basalt_euroc_localize_demo

# Map (once per environment: MH_01_easy, V1_01_easy, V2_01_easy)
target/release/examples/basalt_euroc_online_slam_demo \
  --euroc-dir $D/MH_01_easy \
  --calibration configs/basalt/variants/official_euroc_ds/euroc_ds_calib.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir runs/mh01_slam --save-map runs/mh01_map

# Query, once at a low threshold
target/release/examples/basalt_euroc_localize_demo \
  --map runs/mh01_map --euroc-dir $D/MH_02_easy --out-dir runs/mh02_on_mh01 \
  --min-inliers 12

# Score at any threshold >= 12
python3 scripts/evaluate_map_localization.py --mode fixed \
  --estimate runs/mh02_on_mh01/localization.tum \
  --per-frame runs/mh02_on_mh01/per_frame.csv --min-inliers 30 \
  --reference $D/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/mh01_slam/trajectory_online.tum \
  --align-ground-truth $D/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv \
  --queries 3040
```

`--mode direct --reference runs/mh01_slam/trajectory_online.tum` compares a
localization with the SLAM trajectory in the map frame, without alignment.

## Limits

- One map per environment, built from its "easy" flight; a map merged from
  several flights would cover more of each room (MH_04/MH_05 especially).
- cam0 only; each image is localized independently (no tracking or motion
  model between frames).
- The online mapper's background optimization depends on thread timing, so
  a rerun may build a slightly different map.
