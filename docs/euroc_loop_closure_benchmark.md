# EuRoC Loop-Closure Benchmark (UAV, 6-DOF)

The aerial counterpart to the [KITTI loop-closure
benchmark](kitti_loop_closure_benchmark.md). KITTI is a ground vehicle on a
plane; EuRoC MH_03 is a **micro aerial vehicle on a 6-DOF flight** through a
GPS-denied machine hall with Vicon/Leica ground truth. The same comparison
holds: an open stereo-VO trajectory accumulates unbounded drift, and a loop
closure — the one constraint that ties a revisited place back to its earlier
observation — removes it.

The pipeline is identical to KITTI's
(`pipelines/slam/src/vo_loop_closure.rs`: VLAD appearance → PnP metric
verification → robust GNC SE(3) pose-graph optimization), run over the same
file-backed SuperPoint/LightGlue stereo features twice (open vs
`--loop-closure`). Only the front matter differs:

- **Rectification.** EuRoC's cam0/cam1 are radtan-distorted (unlike KITTI's
  pre-rectified odometry images), so `scripts/rectify_euroc_stereo.py`
  undistorts+rectifies to a pinhole stereo pair and writes a KITTI-format
  `calib.txt` + a `timestamps.txt`.
- **Timestamped ground truth.** EuRoC GT is a ~200 Hz timestamped trajectory,
  so VO poses are converted to TUM (`scripts/kitti_poses_to_tum.py`, using the
  rectifier's `timestamps.txt`) and scored with `evo_ape`, which associates by
  timestamp.

## Result (EuRoC MH_03_medium, 2700 frames)

SuperPoint (2048 kpts) + LightGlue stereo/temporal matching, PnP relative pose.
ATE rmse via `evo_ape` under SE(3) (rigid, the metric-frame number) and Sim(3)
(scale-absorbed) alignment.

| trajectory     | ATE rmse SE(3) | ATE rmse Sim(3) |
| -------------- | -------------: | --------------: |
| open VO        |        2.462 m |         2.203 m |
| + loop closure |        0.464 m |         0.443 m |

**5.3× lower SE(3) ATE rmse (2.46 m → 0.46 m)** from 307 verified loops (5051
appearance candidates), GNC pose-graph cost `1.05e6 → 230`. Unlike the KITTI
stride-2 contrast, **both alignments improve cleanly** (SE(3) 5.3×, Sim(3) 5.0×)
because the metric stereo frontend keeps the scale correct here — the open VO
integrates to 123.9 m against the true ~131 m path — so the metric PnP loop
edges and the odometry edges agree and the pose graph closes without distorting
the metric frame.

![EuRoC MH_03 open VO vs loop closure](assets/euroc_mh03_loop_closure.png)

The open VO (left, red) drifts off ground truth over the flight; loop closure
(right, green) pulls the revisited places back onto the Vicon/Leica trajectory.

## The frame-gap gate matters more in the air

EuRoC flies at 20 Hz, and the MAV spends time hovering/slow-maneuvering. A small
`--loop-min-frame-gap` then matches only slow-motion temporal neighbours that
are already odometry-consistent and contribute no drift correction. Measured on
MH_03:

| `--loop-min-frame-gap` | meaning  | ATE rmse SE(3) |
| ---------------------: | -------- | -------------: |
| 30                     | ~1.5 s   |     2.46 m (unchanged) |
| 200                    | ~10 s    |     0.46 m |

A loop only corrects drift if enough drift accumulated between its two frames,
and drift grows with *distance travelled*, not frame index — so the path-length
gate (`--loop-min-path-length`, metres) is the frame-rate-independent version of
the same idea. The default is `--loop-min-frame-gap 200`.

## Reproduce

Fetch the EuRoC `MH_03_medium` ASL-format zip (the one with a `mav0/` folder) —
EuRoC is hosted on the ETH Research Collection — then:

```sh
scripts/run_euroc_loop_closure_benchmark.sh \
  --mav0 /path/to/MH_03_medium/mav0 \
  --frames 2700
```

The script rectifies the stereo pair, exports SuperPoint/LightGlue features once
(needs torch+lightglue+opencv), runs the VO twice (open / `--loop-closure`),
derives the GT TUM from `mav0/state_groundtruth_estimate0`, converts both VO
trajectories to TUM, scores them with `evo_ape` (needs `pip install evo`), and
writes `summary.md` with the table above. `--rect-dir`/`--feat-dir`/`--gt-tum`
reuse already-prepared artifacts; `--device cpu` runs the export without CUDA.
