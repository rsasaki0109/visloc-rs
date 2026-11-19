# Bonn motion and map update follow-up

The original three recordings are the development set. Three distinct Bonn recordings with the `2` suffix are the holdout. They show the same room and the same type of task at different capture times. The holdout uses 19 moving-object annotation frames and 11 map annotation frames. The same thresholds and motion variants are run on both sets. [Database and calibration](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) · [Initial benchmark and scoring protocol](bonn_labeled_map_evaluation.md) · [Holdout annotations](../benchmarks/bonn/holdout_labels.json) · [Development results](../benchmarks/bonn/development_followup_results.json) · [Holdout results](../benchmarks/bonn/holdout_followup_results.json)

## 1. Motion confirmation

The replay can export per-proposal sample counts, median residuals, evidence and final decisions via `--object-decisions FILE`. The motion benchmark records those decisions at the annotated frames. Three runs compare the original sparse proposals, linearly interpolated proposals with the default filter, and the same interpolated proposals with a bounded continuity hold. Interpolation uses adjacent annotations carrying the same object ID, with a maximum interval of 1.35 s. Only manually annotated frames are scored. The boxes and IDs remain oracle input throughout.

The continuity hold allows an undecided proposal to inherit a previous confirmation for up to 2 s. It requires the same ID, proposal gaps of at most the normal 0.5 s hold, and no intervening decision with sufficient residual evidence below threshold. The original 0.5 s path is the default.

| Recording | Proposals and hold | Precision | Recall | Static retention |
| --- | --- | ---: | ---: | ---: |
| Development | Sparse, default | 91.7% | 28.3% | 98.0% |
| Development | Interpolated, default | 92.7% | 15.1% | 99.1% |
| Development | Interpolated, continuity hold | 92.7% | 15.1% | 99.1% |
| Holdout | Sparse, default | 92.5% | 19.4% | 98.2% |
| Holdout | Interpolated, default | 99.2% | 2.0% | 100.0% |
| Holdout | Interpolated, continuity hold | 99.2% | 2.0% | 100.0% |

Development sparse-proposal misses by first decision evidence were 7 below threshold, 7 missing background. Holdout sparse-proposal misses by first decision evidence were 8 below threshold, 11 missing background.

## 2. Map update

The diagnostic assigns each false negative to the first applicable reason. It checks candidate generation, then projection and occupied evidence in the known occupied visit, then projection and free-space evidence in the known clear visit. The clear-side breakdown includes missing depth, 3 × 3 patch rejection, nearer surfaces and conflicting votes. The `outside_*_view` labels mean fewer than two projected views at the benchmark's valid image/depth bounds. `insufficient_occupied_votes` means the occupied vote count or its ratio to free votes failed, after the geometric and valid-depth checks passed.

The ablations use the same labeled voxels and accepted RGB-D frames. They compare a denser candidate stride (4 px instead of 8 px) and depth margins of 7, 10 and 13 cm. Occluded floor remains a separately scored subset of the reference static voxels.

| Set | Change | Occupancy F1 | Default F1 | Stride 4 F1 | Margin 7 cm F1 | Margin 13 cm F1 | Occluded floor FP (XOR → default) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | addition | 37.6% | 75.1% | 74.8% | 57.0% | 75.2% | 215 → 0 |
| Development | removal | 45.1% | 80.2% | 80.9% | 73.6% | 89.0% | 60 → 0 |
| Holdout | addition | 46.7% | 83.3% | 83.2% | 72.7% | 91.1% | 143 → 0 |
| Holdout | removal | 47.0% | 54.2% | 53.6% | 54.5% | 70.3% | 114 → 0 |

Development had 48 default misses. Their causes were 7 missing surface candidate, 40 insufficient occupied votes, 1 invalid clear depth. Holdout had 61 default misses. Their causes were 5 missing surface candidate, 36 insufficient occupied votes, 3 invalid clear depth, 7 occluded in clear visit, 7 insufficient clear votes, 3 conflicting clear votes.

## 3. Estimated pose comparison

The pose benchmark uses calibrated RGB-D SIFT matches and metric PnP odometry to build independent visit trajectories. It exports maps using `tum_rgbd_save_map` and runs `basalt_map_merge --optimize-sessions --refine-keyframes`. The tracking windows extend the original scoring windows to give registration more viewpoints. The manually labeled frames are excluded from both tracking and map prediction. The change layer is scored on the original scoring windows using the same reference voxels and frame IDs as the mocap run.

One rigid transform aligns the first estimated pose of the base visit to its mocap pose. That single transform applies to both visits and every subsequent frame, so drift and relative registration errors remain in the measurements. The benchmark records accepted session registrations and the separate count of keyframe loop factors. Each comparison rebuilds its two session maps once and repeats the merge three times to expose variation in registration. F1 is reported as the median and range across the three trials.

| Set | Change | Mocap F1 | Estimated F1 median [range] | Session registrations per trial | Keyframe loops per trial |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | addition | 75.1% | 16.4% [16.4%–16.4%] | 1, 1, 1 | 0, 0, 0 |
| Development | removal | 80.2% | 62.3% [62.3%–62.3%] | 1, 1, 1 | 0, 0, 0 |
| Holdout | addition | 83.3% | 53.0% [53.0%–53.0%] | 1, 1, 1 | 0, 0, 0 |
| Holdout | removal | 54.2% | 35.0% [35.0%–35.0%] | 1, 1, 1 | 0, 0, 0 |

Each trial accepted the cross-session registration. The keyframe refinement verifier selected zero loop factors, so the estimated scores measure session-level alignment. The 13 cm margin produced zero occluded-floor false positives in all four mocap cases. It improved F1 in this benchmark, with a loss of precision in the development addition case (77.6% to 69.1%).

The scope is one environment. Motion scores are conditional on manually supplied boxes and IDs, and the pose study measures the combination of RGB-D odometry and multi-session registration. The voxel scores cover annotated changes and static surfaces on short revisits.

The next evaluation identifies the keyframe rejection gates and measures short-baseline refinement and a depth fallback on the same recordings. [Loop and depth evaluation](bonn_loop_and_depth_evaluation.md)

## Reproduce

Download and extract the original three Bonn archives and the additional `moving_obstructing_box2`, `placing_nonobstructing_box2` and `removing_nonobstructing_box2` archives to the same directory. See the initial report for dependencies and the original hashes. [Dataset downloads](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) · [Initial report](bonn_labeled_map_evaluation.md)

```text
cc5d3ec67cb9dadd9905de3ea9b72120f228e0bd4da865858fe44cc2a1ccd643  rgbd_bonn_moving_obstructing_box2.zip
ea48ae0c78ed07d3404e5ec6e17723bef1500581245984e8afd5843445e95089  rgbd_bonn_placing_nonobstructing_box2.zip
02d51c5c26a740f69ceb1c269b1874d5304c5a435cfe0102c0971a7475eb3d52  rgbd_bonn_removing_nonobstructing_box2.zip
```

```bash
cargo build --release --example basalt_euroc_vio_demo \
  --example tum_rgbd_save_map --example basalt_map_merge

for labels in map_change holdout; do
  python scripts/eval_bonn_map_changes.py \
    --data /path/to/extracted/bonn \
    --basalt target/release/examples/basalt_euroc_vio_demo \
    --annotations "benchmarks/bonn/${labels}_labels.json" \
    --out "runs/bonn_${labels}_followup.json" \
    --motion-ablation --map-ablation --map-tools target/release/examples --map-repeats 3
done
```
