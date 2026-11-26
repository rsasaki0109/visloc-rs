# Map localization among moving people: measurements

Results page of [dynamic_obstacle_localization_plan.md](dynamic_obstacle_localization_plan.md).
This is Phase 0: today's localization, measured indoors with people walking
in front of the camera, before any lever. Nothing in the localization has
changed; only measurement was added.

**Headline.** With people walking through the view (TUM RGB-D fr3 walking),
single-image localization accepts many wrong poses: **22 to 145 fixes more
than 10 cm off** per sequence, none on the sequence where they sit still.
Most wrong fixes have their inliers bunched in a few image cells: a fix with
inliers in at most 4 of 16 cells is more than 10 cm off **56%** of the
time, one with 8 or more only **5%**. With an odometry, the correction's
gate keeps the map pose within 2–4 cm on three of the four walking
sequences, but still takes in 3–27 wrong fixes, and on walking_rpy the map
pose is 10 cm RMSE (20 cm worst). The walking people hide a third of the
image cells where the map has landmarks (matched in 58–67% of them, against
92% with the people seated).

## What was added (measurement only)

- `visloc_localization::{ExpectedVisibility, ImageGrid, view_diagnostics}`
  and `basalt_euroc_localize_demo --diagnostics FILE [--diag-grid N]
  [--eval-mask-dir DIR]`: per attempt,
  - `inlier_cells`: cells of a 4 × 4 image grid holding an inlier;
  - `expected`: the landmarks the map's 5 nearest keyframes (within 1 m and
    30° of the camera's axis) saw, projected into the image, from the
    predicted pose (with an odometry) or the solved one;
  - `expected_cells` / `missing_cells`: cells holding 3 or more of them,
    and of those the cells holding no inlier at all;
  - with masks: keypoints and inliers on moving things.
  Every other output is byte-identical with and without `--diagnostics`
  (checked on walking_xyz in both modes; the latency column aside). Cost:
  0.1 ms median per attempt (0.3 ms worst), against 18–60 ms of
  localization.
- Landmark counts are of limited use on these maps: a map built along a
  sequence keeps a landmark per stretch of keyframes that tracked a point,
  so one physical point has several. The cell counts do not care, and are
  what the tables use.
- `scripts/tum_rgbd_to_euroc.py`, `scripts/synth_drift_odometry.py`,
  `scripts/eval_dynamic_localization.py`.

## Setup

- **Data.** TUM RGB-D Freiburg 3, from `cvg.cit.tum.de` (CC BY 4.0):
  `sitting_xyz` and `sitting_halfsphere` (two people seated at a desk,
  moving a little; handheld camera moving) for the map; queries
  `sitting_static` (people seated, camera still) as the control and
  `walking_static`, `walking_xyz`, `walking_halfsphere`, `walking_rpy`
  (two people walking around the desk and in front of the camera). All were
  recorded in one session in one motion-capture frame. SHA-256 of the
  archives:

  | Archive | SHA-256 |
  | --- | --- |
  | `rgbd_dataset_freiburg3_walking_xyz.tgz` | `1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d` |
  | `rgbd_dataset_freiburg3_walking_static.tgz` | `8aca832bf01746aa95fcc0b15bf42d78ad3bb8edef672ba4cc94a404b194f6c1` |
  | `rgbd_dataset_freiburg3_walking_halfsphere.tgz` | `c4984f4894724398590dcf18c3d9d54dbb76cce925d51695f44ad8e697fc8458` |
  | `rgbd_dataset_freiburg3_walking_rpy.tgz` | `8727e72a01da0150996aa98a63682e1798e4d6d196c30d35e8f693b33fba3d5d` |
  | `rgbd_dataset_freiburg3_sitting_static.tgz` | `6239d4b7f6edcab719b51c76434acdc238d03dad666758b51e6046992a7c3914` |
  | `rgbd_dataset_freiburg3_sitting_xyz.tgz` | `05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205` |
  | `rgbd_dataset_freiburg3_sitting_halfsphere.tgz` | `ba9f0fab0d07e22f04fbfae16eb4e3fb44088a32c920ad36c782b5024ed4b767` |

- **Map.** `tum_rgbd_save_map --keep-world` along the ground truth of
  `sitting_xyz` and `sitting_halfsphere` concatenated (their stamps do not
  overlap): 663 keyframe images, 40,242 landmarks. The map is in the
  ground truth's frame, so poses are scored directly, with nothing fitted.
  The seated people are in it (F3 is part of what is measured).
- **Why not a static-camera map.** A map from `sitting_static` alone (the
  camera never moves) localized 0–6 images of each walking sequence: their
  views start 25 cm or more away, and that map has one viewpoint.
- **Query settings.** `--ratio none --max-hamming 60 --min-inliers 20`.
  With the default ratio test (tuned on EuRoC), even `sitting_static`
  localized only 139 of 707 images against a `sitting_xyz` map; without it,
  707. The many near-duplicate landmarks of an RGB-D map along a sequence
  make the second-best distance close to the best.
- **Modes.**
  - *Single image*: every image localized on its own. Shows the raw fixes.
  - *Odometry*: `--odometry` with a drifting odometry made from the ground
    truth (`synth_drift_odometry.py`: 2% scale, 3°/m of yaw, step noise;
    roll and pitch kept, as an IMU would), `--localize-every 3` (10 Hz).
    Freiburg 3 records no IMU. Three seeds.
- **Machine.** 4-core container, release build, `--threads 4`.

## Single image

| Query | Localized | Longest gap | Fix error RMSE / median / max | Fixes > 10 cm | Inlier cells (median) | Mapped cells matched (median) | Time / image |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| sitting_static (control) | 707/707 (100%) | 0.1 s | 2.9 / 2.2 / 8.5 cm | **0** | 12 | 92% | 50 ms |
| walking_static | 730/743 (98.3%) | 0.3 s | 4.2 / 2.8 / 25.3 cm | **22** | 8 | 67% | 57 ms |
| walking_xyz | 710/859 (82.7%) | 1.2 s | 7.1 / 3.4 / 59.1 cm | **70** | 7 | 62% | 60 ms |
| walking_halfsphere | 602/1067 (56.4%) | 3.4 s | 10.3 / 4.5 / 84.8 cm | **95** | 8 | 67% | 58 ms |
| walking_rpy | 385/910 (42.3%) | 2.7 s | 16.2 / 7.0 / 129.2 cm | **145** | 7 | 67% | 57 ms |

The runs are deterministic (the same output bytes on a rerun).

**Inlier spread against error**, all accepted fixes of the four walking
sequences:

| Inlier cells (of 16) | Fixes | > 10 cm off |
| --- | ---: | ---: |
| ≤ 4 | 93 | 52 (**56%**) |
| 5–6 | 540 | 145 (27%) |
| 7 | 483 | 67 (14%) |
| ≥ 8 | 1,295 | 68 (**5%**) |

On `sitting_static`, every fix has 5 or more inlier cells (median 12) and
none is 10 cm off.

![The four worst fixes of walking_xyz (59, 49, 46 and 38 cm off) and two good ones (1 cm), with their inliers in red. The bad ones have their inliers on the posters on the partition, bunched in one or two places; the good ones have them over the desk, the monitors and the partitions.](assets/dynamic_tum_wrong_fixes.jpg)

The four worst fixes of walking_xyz (top row and bottom left: 59, 49, 46,
38 cm) and two good ones (1 cm), inliers in red. The bad ones are not on
the people: their inliers sit on the posters of the partition, in one or two
blobs, seen from farther back than the map was. The good ones spread over
desk, monitors and partitions. Images: TUM RGB-D, CC BY 4.0.

## With an odometry (three seeds)

Median of the three seeds, range in brackets.

| Query | Images localized | Map pose RMSE | Map pose max | Wrong fixes used (> 10 cm) | Resets |
| --- | --- | --- | --- | --- | --- |
| sitting_static (control) | 100% | 1.5 (1.3–1.8) cm | 2.8 (2.4–3.3) cm | 0 | 0 |
| walking_static | 98.8% | 2.2 (1.8–2.3) cm | 4.0 (2.7–4.4) cm | 3 (2–4) | 0 |
| walking_xyz | 87.8% | 2.0 (1.7–4.8) cm | 5.9 (3.7–12.9) cm | 12 (11–21) | 1 (0–1) |
| walking_halfsphere | 67.7% | 3.1 (2.8–4.4) cm | 9.2 (9.2–10.4) cm | 13 (11–16) | 3 (2–3) |
| walking_rpy | 55.6% | **10.0** (10.0–10.1) cm | **20.2** (19.7–20.6) cm | **27** (23–27) | 1 |

"Wrong fixes used" are fixes more than 10 cm off that the correction
accepted, initialized or reset from. Its gate (15 cm and 3°, widened with
travel) lets through fixes that are wrong by less than the gate, and they
are averaged into the correction.

## Failure modes (F1–F5 of the plan)

- **F2/F5, bunched inliers: the main source of wrong fixes.** Wrong fixes
  have their inliers in few cells, on one textured patch (posters), often
  from a view farther back than the map's. The people take away the rest
  of the view, so what is left is a small, nearly planar patch, and P3P
  RANSAC on it is poorly conditioned. Movers' own keypoints are rarely the
  inliers in these images. Lever D1 (inlier spread) addresses this
  directly.
- **F1, occlusion.** The walking people hide about a third of the mapped
  cells (58–67% matched against 92%). Fewer images localize (42–98%), but
  the longest gap without a fix stays at 3.7 s or less.
- **F5 with an odometry.** 3–27 wrong fixes pass the gate per sequence; on
  walking_rpy (fast rotation) they hold the map pose 10 cm off.
- **F3, people in the map.** The map holds the seated people. Not separated
  out yet: no masks for TUM.
- **F4, dirty odometry.** Not tested: the odometry here is synthetic.

## Not yet measured

- **Bonn RGB-D Dynamic.** Downloaded (crowd, crowd2, person_tracking,
  moving/removing_obstructing_box, static_close_far). Its `groundtruth.txt`
  is the motion-capture markers' pose, and the published marker-to-camera
  transform is not a rotation, so a map cannot be built along it as is. It
  needs its own marker-to-camera calibration first.
- **OpenLORIS-Scene cafe**, the Gazebo actors and masks: not started.
- **Detector cost on a Jetson:** needs hardware.

## Proposed next step and targets

D1 first (inlier spread as a fix-quality test), measured on these tables.
Proposed targets on the walking sequences, against the tables above:

- single image: fixes more than 10 cm off down by at least half, losing at
  most a tenth of the good fixes;
- with an odometry: no fix more than 10 cm off used, walking_rpy's map pose
  under 5 cm RMSE, the control unchanged;
- under 1 ms added per localized image.

## Reproduce

```bash
D=/data/tum   # rgbd_dataset_freiburg3_{sitting,walking}_*
W=/work
cargo build --release --example basalt_euroc_localize_demo
cargo build --release --features image-io --example tum_rgbd_save_map
# Map: sitting_xyz + sitting_halfsphere, concatenated (rgb.txt, depth.txt and
# groundtruth.txt merged and sorted by stamp; rgb/ and depth/ linked).
target/release/examples/tum_rgbd_save_map --seq-dir $W/tum_combined/sitting_moving \
  --trajectory $W/tum_combined/sitting_moving/groundtruth.txt --out $W/maps/sitting_moving \
  --keep-world --keyframe-m 0.02 --keyframe-deg 2 \
  --fx 535.4 --fy 539.2 --cx 320.1 --cy 247.6 --k1 0 --k2 0 --p1 0 --p2 0 --k3 0
Q="--ratio none --max-hamming 60 --min-inliers 20 --threads 4"
for s in sitting_static walking_static walking_xyz walking_halfsphere walking_rpy; do
  python3 scripts/tum_rgbd_to_euroc.py --seq-dir $D/rgbd_dataset_freiburg3_$s --out $W/tum/$s
  # Single image
  target/release/examples/basalt_euroc_localize_demo --map $W/maps/sitting_moving \
    --euroc-dir $W/tum/$s --out-dir $W/q/sm_$s $Q --diagnostics $W/q/sm_$s/diagnostics.csv
  python3 scripts/eval_dynamic_localization.py \
    --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt --label $s $W/q/sm_$s
  # With a drifting odometry, seeds 1-3
  for seed in 1 2 3; do
    python3 scripts/synth_drift_odometry.py --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt \
      --out $W/odom/$s.seed$seed.tum --seed $seed
    target/release/examples/basalt_euroc_localize_demo --map $W/maps/sitting_moving \
      --euroc-dir $W/tum/$s --out-dir $W/q/od${seed}_$s $Q \
      --odometry $W/odom/$s.seed$seed.tum --localize-every 3 \
      --diagnostics $W/q/od${seed}_$s/diagnostics.csv
    python3 scripts/eval_dynamic_localization.py \
      --gt $D/rgbd_dataset_freiburg3_$s/groundtruth.txt --label "$s s$seed" $W/q/od${seed}_$s
  done
done
```
