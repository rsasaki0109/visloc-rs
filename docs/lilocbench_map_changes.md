# LILocBench corridor change evaluation

This small holdout uses the front RGB-D camera from LILocBench's public `mapping` and `lt_changes_0` recordings. Their published reference trajectories start 10 h 36 min apart. The evaluation windows cover the same corridor on independent robot passes about 10 h 24 min apart for the box and 10 h 34 min apart for the doors. The examples and sampled annotations stay within one building. [LILocBench data and calibration](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/) · [Previous Bonn and OpenLORIS study](loop_influence_and_motion_pose.md)

## Protocol

In one corridor interval, the later visit sees a white cardboard box on previously clear floor. The second interval shows an open door leaf in the mapping run and the same door closed in the later run. The second score treats the occupied surface at the old, open location as removed. Rectified RGB polygons specify changed surfaces, unchanged walls and doors, and unchanged floor. The `box` key in the shared annotation schema denotes the changed surface in both cases. Fourteen annotated frames are withheld from map construction, pose estimation and depth selection. The ground-truth voxel support uses the published poses to back-project depth within those polygons. The occluded floor subset is determined from the annotated unchanged floor and changed surface. [Polygons](../benchmarks/lilocbench/corridor_labels.json)

Depth is rectified and projected into the RGB optical frame through the published sensor transform tree, using the closest matching depth timestamp (maximum observed skew below 0.06 ms). Z-buffering keeps the nearest depth at each pixel. The prediction uses the same 7 cm voxels, 10 cm visibility margin, 0.3–4 m depth support and minimum of two views as the earlier benchmarks. The front camera is sampled at 4 Hz for map building. Change scoring uses every other eligible observation. For the door, the base scoring timestamps are staggered relative to the annotation frames so the other nearby observations still participate in the score.

Estimated maps use the recorded wheel odometry and the calibrated robot-to-camera transform, with visual RGB-D landmarks and Basalt's cross-session registration. The corridor yielded insufficient depth-backed RGB feature matches for an RGB-D-only SIFT/PnP replay, so all estimated conditions share the published wheel odometry. The camera pose at the first usable base frame fixes a common gauge to compare trajectory and voxel errors. Reference poses also provide a separate oracle score. The gauge is a single common rigid transform and does not change the relative depth agreement used by selection.

The Basalt keyframe gate uses the default 35 cm minimum spacing. The depth selector uses the previously fixed rule on co-view pairs from outside the scoring and loop-endpoint frames. It requires at least five pairs and an increase of 0.5 percentage points in mean bidirectional agreement within 7 cm. The selector compares complete solutions and uses neither the manual labels nor the reference pose errors. The validation pairs are drawn from frames available to the reconstructed maps.

## Results

| Case | Pose source | Change F1 | Revisit position RMSE | Occluded floor marked changed |
| --- | --- | ---: | ---: | ---: |
| Box addition | Published reference | 47.7% | — | 0 / 261 |
| Box addition | Session registration | 29.6% | 1.96 cm | 0 / 261 |
| Box addition | All six keyframe constraints | **31.0%** | 1.74 cm | 1 / 261 |
| Open door surface removal | Published reference | 37.9% | — | 0 / 18 |
| Open door surface removal | Session registration | 19.1% | 2.03 cm | 0 / 18 |
| Open door surface removal | All five keyframe constraints | **25.4%** | 2.18 cm | 0 / 18 |

| Case | Validation pairs | Session registration depth agreement | All-loops depth agreement | Selected |
| --- | ---: | ---: | ---: | --- |
| Box addition | 15 | 77.42% | 78.45% | All loops |
| Open door surface removal | 11 | 89.92% | 89.89% | Session registration |

The selected box solution cut total stable false positives from 192 to 173 and detected 44 of the 67 labeled changed voxels. It also introduced one false change on the occluded floor. For the door the selector kept the more accurate revisit trajectory, with lower change F1. Even the reference-pose run missed 431 of 574 labeled old-door voxels, including 339 absent from the sampled surface candidate set. The door enters the 4 m range late in the pass. The results quantify the mismatch between the scene-wide depth criterion and local change accuracy in these two sampled corridor intervals. [Full metrics and diagnostics](../benchmarks/lilocbench/corridor_results.json)

## Reproduce

The fetcher uses ZIP byte ranges and verifies checksums of archive members. It retrieves about 130 MB of RGB and depth images plus metadata, with raw images kept outside the repository. The two published reference trajectory SHA-256 digests are `dfc8b3c4abf717abdd9826ba3ca67978cc00c8986ef038390cc1ac3ad73efd2a` for `mapping` and `32c1f451eced24dbd4891771c113aaf2dccfb9d67abf54036c3a2b0f42bcb461` for `lt_changes_0`.

```bash
python -m pip install numpy scipy pillow pyyaml opencv-python-headless
cargo build --release --example basalt_map_merge --example tum_rgbd_save_map

python scripts/fetch_lilocbench_windows.py \
  --annotations benchmarks/lilocbench/corridor_labels.json --out /path/to/liloc
OPENBLAS_NUM_THREADS=1 python scripts/eval_lilocbench_changes.py \
  --data /path/to/liloc --annotations benchmarks/lilocbench/corridor_labels.json \
  --map-tools target/release/examples --out runs/lilocbench_corridor.json
```
