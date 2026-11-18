# TUM RGB-D Benchmark (indoor, handheld)

The indoor handheld counterpart to the [KITTI](kitti_loop_closure_benchmark.md)
(ground vehicle) and [EuRoC](euroc_loop_closure_benchmark.md) (aerial)
loop-closure benchmarks. [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
is the classic indoor SLAM benchmark — a hand-held Kinect through an office, with
motion-capture ground truth — on which ORB-SLAM2, ElasticFusion, and DVO-SLAM
publish ATE.

It also exercises a different sensor than the rest of the suite. TUM RGB-D has
**no stereo pair**: each frame is a single RGB image plus a registered depth map
(16-bit PNG, `depth_m = pixel / 5000`). Rather than add a separate RGB-D
frontend, the same metric stereo VO / online-BA / loop-closure backend
(`examples/stereo_vo_external_deep_files.rs`,
`pipelines/slam/src/vo_loop_closure.rs`) is driven by **virtual stereo**.

## Virtual stereo: depth as a synthetic right image

The stereo backend only ever uses the right image to triangulate metric 3D
points; everything downstream (PnP / Kabsch relative pose, online BA, loop
closure) runs on the resulting camera-frame points. So `depth` can stand in for
the right view directly. For each depth-valid SuperPoint keypoint
`(u, v)` with depth `d`, `scripts/export_tum_rgbd_virtual_stereo.py` emits a
synthetic right keypoint at the disparity it would have at a chosen virtual
baseline `b`, carrying the same descriptor:

```
disparity = b * fx / d            u_right = u - disparity,   v_right = v
```

and a 1:1 stereo match. The Rust triangulator inverts this exactly
(`d = b * fx / disparity`), so the recovered depth — and hence the **metric
scale** — is the true TUM depth, *independent of the (arbitrary) baseline*, as
long as the same `--baseline` is passed to the binary. Temporal (frame-to-frame)
matches come from LightGlue between consecutive left frames, exactly as in the
stereo path.

The payoff: the entire stereo VO / BA / loop stack runs on RGB-D **with zero Rust
changes** — the new sensor is handled entirely in the feature exporter. Metric
scale is recovered directly from depth, with no scale drift to absorb (the SE(3)
and Sim(3) ATE numbers below agree to ~1 mm).

## Results (Freiburg1, full sequences)

SuperPoint (2048 kpts) + LightGlue, virtual-stereo depth lift, Freiburg1
intrinsics (`fx 517.3 fy 516.5 cx 318.6 cy 255.3`), baseline 0.1 m. ATE rmse via
`evo_ape` under SE(3) (rigid, the metric-frame number) and Sim(3) (scale-absorbed)
alignment, associated against the motion-capture `groundtruth.txt` by timestamp.

The full pipeline visloc ships — incremental window BA (`--online-ba`) + metric
loop closure (`--loop-closure`) + two-view loop-edge BA (`--loop-two-view-ba`) —
is the `full_tv` row.

### fr1_xyz (798 frames, translation-dominant)

| trajectory                         | ATE rmse SE(3) | ATE rmse Sim(3) | loops |
| ---------------------------------- | -------------: | --------------: | ----: |
| open VO                            |       0.0313 m |        0.0280 m |     – |
| + loop closure                     |       0.0244 m |        0.0213 m |    63 |
| + loop closure + two-view BA       |       0.0207 m |        0.0165 m |    63 |
| window BA + loop                   |       0.0205 m |        0.0200 m |    89 |
| **window BA + loop + two-view BA** |   **0.0140 m** |    **0.0129 m** |    89 |

![TUM fr1_xyz virtual-stereo VO + loop closure](assets/tum_fr1_xyz_loop_closure.png)

### fr1_desk (596 frames, rotation-heavy)

| trajectory                         | ATE rmse SE(3) | ATE rmse Sim(3) | loops |
| ---------------------------------- | -------------: | --------------: | ----: |
| open VO                            |       0.1596 m |        0.1573 m |     – |
| window BA + loop                   |       0.0363 m |        0.0264 m |    52 |
| **window BA + loop + two-view BA** |   **0.0262 m** |    **0.0215 m** |    52 |

## Where a vision-only RGB-D stack stands

The best config lands at **14.0 mm (fr1_xyz)** and **26.2 mm (fr1_desk)** ATE
rmse, **~1.3–1.6× ORB-SLAM2 RGB-D** (which publishes ≈ 0.009–0.011 m on fr1_xyz
and ≈ 0.016 m on fr1_desk) — competitive for a vision-only stack that treats
depth purely as a virtual right image and reuses an outdoor-stereo-tuned VO
backend unchanged.

Two levers carry the indoor result, the same ones the
[EuRoC benchmark](euroc_loop_closure_benchmark.md) identified:

- **Loop closure is the dominant lever**, far more so than on the largely
  monotonic aerial flights. The desk scene revisits the same workspace
  repeatedly, so the open VO's accumulated drift (159.6 mm) collapses to 36.3 mm
  once 52 revisits are tied back — a **6× reduction** from loop closure alone.
- **Two-view loop BA** (`--loop-two-view-ba`) sharpens it further on both
  sequences (fr1_xyz 20.5 → 14.0 mm, fr1_desk 36.3 → 26.2 mm). Each verified
  loop edge is re-ground as a minimal two-view bundle adjustment before it enters
  the pose graph, removing the depth-anchoring bias PnP leaves in the raw edge —
  the same ~30% lever measured on EuRoC.

A note on window BA in isolation: on `fr1_xyz`, `--online-ba` *without* loop
closure slightly worsens the open VO (31.3 → 50.8 mm) because the temporal
sliding window, tuned for fast KITTI/EuRoC motion, over-constrains the slow
hand-held trajectory. Its value here is indirect: the locally-tighter map detects
more revisits (89 vs 63 loops), and once those loops plus two-view BA are applied
the combination is the best of all (14.0 mm). The frontend stage and the loop
stage are complementary, not independently additive.

## Several sessions in one map

![Three TUM RGB-D sessions of one office, each mapped in its own frame, merged into one map](assets/map_merge_tum_rgbd.gif)

<sub>Data: [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) fr1_room, fr1_desk2 and fr1_desk (Sturm et al., IROS 2012), CC BY 4.0. Colours are the sequences' RGB images placed by their depth.</sub>

The fr1 sequences film one office. Each is mapped on its own, in its own
frame, and `basalt_map_merge` merges the maps, as it merges the
stereo-inertial SLAM's maps of EuRoC and OpenLORIS-Scene
([basalt_map_localization.md](basalt_map_localization.md#merging-maps-of-several-flights)):

1. **Trajectory.** The virtual-stereo VO above (`full_tv`, here with 1,024
   SuperPoint keypoints on the CPU): ATE 5.33 cm on fr1_room, 4.67 cm on
   fr1_desk2 and 2.27 cm on fr1_desk. Its frame is the first camera's.
2. **Map.** `examples/tum_rgbd_save_map.rs` turns the RGB images and depth
   into a map in the layout `basalt_euroc_online_slam_demo --save-map`
   writes: keyframes every 5 cm or 5°, the Basalt mapper's ORB on each
   (Shi-Tomasi corners, 256-bit descriptors), and a landmark at each
   corner's depth, tracked over the last 10 keyframes by projection and
   descriptor. Landmarks seen once are dropped.
3. **Gravity.** The merge registers maps by a yaw and a translation, so each
   map's world must have z up. The Kinect's accelerometer gives it: each
   sample is turned into the camera frame (its x and y point the other way
   from the optical frame's, which was read once off fr1_room's ground
   truth), then into the world by the trajectory, and averaged over the
   sequence. On fr1_room and fr1_desk, the ground truth's vertical is
   1.8° and 3.1° from this up.
4. **Merge.** `basalt_map_merge` as for the SLAM's maps. The colour clouds
   of the GIF come from `scripts/rgbd_map_cloud.py --tum`.

| Added map (in order) | Its frame vs fr1_room's | Keyframes localized | Residual |
|---|---|---|---|
| fr1_desk2 | 71.6°, 248 cm | 20 of 209 | 8.6 cm |
| fr1_desk | 72.1°, 227 cm | 17 of 171 | 4.0 cm |

The merged map holds 751 keyframe images and 53,063 landmarks. Aligned to
the ground truth by fr1_room's keyframes alone, the keyframe centres are
at 7.4 cm RMSE (24.5 cm max): fr1_room 5.4, fr1_desk2 10.2 and fr1_desk
7.0 cm, against 5.4, 4.6 and 2.4 cm for each trajectory aligned on its own.

Few keyframes localize: TUM's handheld Kinect blurs most images, and a
keyframe offers only its tracked corners. fr1_360 (turning in place, VO
ATE 9.95 cm) and fr1_plant (close to one plant) register in none of the
other maps, in any order.

```bash
D=/path/to/tum   # rgbd_dataset_freiburg1_{room,desk2,desk}
for s in room desk2 desk; do
  TUM_MAX_KEYPOINTS=1024 scripts/run_tum_rgbd_benchmark.sh --seq-dir $D/rgbd_dataset_freiburg1_$s \
    --out-dir runs/tum_$s --device cpu
  target/release/examples/tum_rgbd_save_map --seq-dir $D/rgbd_dataset_freiburg1_$s \
    --trajectory runs/tum_$s/full_tv/est.tum --out runs/fr1_${s}_map \
    --keyframe-m 0.05 --keyframe-deg 5 --match-radius-px 8 --track-keyframes 10
  python3 scripts/rgbd_map_cloud.py --map runs/fr1_${s}_map \
    --tum $D/rgbd_dataset_freiburg1_$s --out runs/fr1_${s}_cloud.npz --every 3 --max-depth 3
done
target/release/examples/basalt_map_merge --base runs/fr1_room_map \
  --add runs/fr1_desk2_map --add runs/fr1_desk_map --out runs/fr1_merged
python3 scripts/render_map_merge_gif.py --merged runs/fr1_merged \
  $(for s in room desk2 desk; do echo --map runs/fr1_${s}_map --cloud runs/fr1_${s}_cloud.npz \
    --label fr1_$s; done) \
  --max-points 60000 --point-size 2 --unit session --place office --fill 0.6 \
  --title "Multi-session mapping: three RGB-D sessions, one map" \
  --out docs/assets/map_merge_tum_rgbd.gif
```

## Reproduce

Download a sequence (e.g. `rgbd_dataset_freiburg1_xyz`) from the
[TUM RGB-D page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download),
then:

```bash
scripts/run_tum_rgbd_benchmark.sh --seq-dir /path/to/rgbd_dataset_freiburg1_xyz
```

This exports virtual-stereo SuperPoint/LightGlue features
(`scripts/export_tum_rgbd_virtual_stereo.py`, requires `torch` + `lightglue` +
`pillow`), runs the three conditions above, converts each VO trajectory to TUM
(`scripts/kitti_poses_to_tum.py`), and scores with `evo_ape`
(`pip install evo`). For Freiburg2/3 pass the matching intrinsics with
`--fx/--fy/--cx/--cy`. The trajectory figure is regenerated with:

```bash
scripts/plot_tum_trajectory.py \
  --gt rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --est target/tum_rgbd_benchmark/full_tv/est.tum \
  --out docs/assets/tum_fr1_xyz_loop_closure.png \
  --title "TUM fr1_xyz: virtual-stereo VO + window BA + loop closure"
```
