# Seasonal alignment and conservative map clearance

## Seasonal correspondence and reconstruction

ROVER Garden Small records the same site in summer and snow, 148 days apart. The fixed correspondence evaluation uses four pairs with visually confirmed overlap, six pairs without overlap, and three same-season controls. The learned variant uses the outdoor EfficientLoFTR checkpoint. It filters matches below confidence 0.2 and requires at least 15 epipolar inliers, an inlier fraction of at least 35%, and inlier support in at least four cells of a 4 × 4 grid in each image. Both methods use 2 px epipolar RANSAC after undistortion.

| Method | Seasonal overlapping pairs accepted | Seasonal non-overlap pairs accepted | Same-season controls accepted |
| --- | ---: | ---: | ---: |
| Mutual SIFT | 0/4 | 0/6 | 3/3 |
| EfficientLoFTR | 2/4 | 0/6 | 3/3 |

The accepted pairs show the shed corner and the house facade. The follow-up reconstructs six nearby views from each season around the facade. It joins geometrically screened correspondences in a COLMAP model and performs bundle adjustment with fixed camera calibration. The reconstructed positions are in a common map frame. The summer reference prism distances and the published prism-to-camera offset set one metric scale. The winter reference prism distances provide a separate check of that scale.

One model registered five images from each season. It contains 1,487 reconstructed points, including 1,010 tracks observed in both seasons, with mean reprojection error of 0.78 px. The prism-distance RMSE was 7.1 cm in summer and 4.9 cm on the ten held-out winter image-pair distances. The winter distances share five images. The recorded relative pose and distance-scale check cover the selected facade window.

Absolute cross-season pose error requires a surveyed cross-season reference frame or independent control points. Site-wide loop detection and a seasonal occupancy score would also require larger sampling and 3D change labels. [ROVER data and calibration](https://iis-esslingen.github.io/rover/pages/download/) · [Pair results](../benchmarks/rover/seasonal_learned_results.json) · [Geometry results](../benchmarks/rover/seasonal_geometry_results.json)

## Persisted map clearance

All policies start from baseline clearance candidates supported by at least two views with a 3 × 3 depth patch and 10 cm clearance margin. The wider-patch policy uses a 5 × 5 patch and 15 cm margin to confirm each deletion. The separated-view policy also requires at least two supporting camera origins 25 cm apart. For visit three, an optional history policy defers candidates with occupied evidence in both earlier visits. It uses the earlier RGB-D and estimated poses to save that evidence alongside the map before visit three. The replay writes the updated geometry and reopens the file to count actual deletions. Labels only score the changes.

| LILocBench policy | Stable voxels wrongly deleted on visit 2 | Stable voxels wrongly deleted on visit 3 | Box voxels deleted on visit 3 |
| --- | ---: | ---: | ---: |
| Baseline | 197/1,834 | 118/1,639 | 47/47 |
| Wider patch | 94/1,834 | 85/1,742 | 43/47 |
| Wider patch with separated views | 89/1,834 | 79/1,747 | 43/47 |
| Wider patch and visit history | 94/1,834 | 60/1,742 | 43/47 |
| Wider patch, separated views and visit history | 89/1,834 | 55/1,747 | 43/47 |

Visit 2 added the same 749 surface voxels for all policies. Visit 3 uses visual fixes with wheel propagation. The visit 3 stable denominators reflect each map's saved footprint from visit 2. The history variant retains repeatedly seen surfaces through this update. It takes further visits to validate when persistent clearance should supersede that history.

The Bonn box-removal holdout tests the two per-visit guards in another environment. Using short-baseline optimized poses, the baseline removes 92 true and 3 stable voxels. The wider patch removes 89 true and 3 stable voxels. The separated-view guard defers all removals with these estimated poses. With the published reference poses, it preserves all 93 baseline true removals and reduces false removals from 5 to 0. This shows the need to account for available camera motion and pose quality before requiring separated views. The two-visit Bonn case provides no history condition. [Clearance metrics](../benchmarks/lilocbench/clearance_results.json)

## Reproduce

Prepare the LILocBench, Bonn, and ROVER directories and map tools as described in the [public revisit evaluation](public_lifelong_slam_followup.md). For the seasonal runs, install `onnxruntime==1.23.2` and `pycolmap==4.2.1`. Use `eloftr_outdoor_full.onnx` with SHA-256 `e6478683562e0e4052af3a0bfdbc20dd08bfbd907ec8e4e5eff83a1adfca5fcf`. The geometry run additionally reads each date's `groundtruth.txt`. [Pinned ONNX model](https://huggingface.co/SpatialHub/efficient-loftr-onnx/blob/0a7cadfb9834380495f902d17a39bdb6b63e29e8/eloftr_outdoor_full.onnx) · [EfficientLoFTR source](https://github.com/zju3dv/EfficientLoFTR)

```bash
export OPENBLAS_NUM_THREADS=1
rover_summer=/path/to/extracted/2023-08-18
rover_winter=/path/to/extracted/2024-01-13
rover_calibration=/path/to/calibration/calib_d435i.yaml
rover_model=/path/to/eloftr_outdoor_full.onnx
python scripts/eval_rover_seasonal_overlap.py \
  --summer "$rover_summer" --winter "$rover_winter" --calibration "$rover_calibration" \
  --efficient-loftr "$rover_model" --annotations benchmarks/rover/seasonal_pairs.json \
  --out runs/rover_seasonal_learned.json
python scripts/eval_rover_seasonal_geometry.py \
  --summer "$rover_summer" --winter "$rover_winter" --calibration "$rover_calibration" \
  --efficient-loftr "$rover_model" --out runs/rover_seasonal_geometry.json

for policy in baseline patch separated; do
  history=()
  if [[ "$policy" != baseline ]]; then history+=(--keep-reobserved); fi
  python scripts/eval_lilocbench_reuse.py \
    --data /path/to/liloc --annotations benchmarks/lilocbench/corridor_labels.json \
    --map-tools target/release/examples --maps-out "runs/lilocbench_${policy}_maps" \
    --clearance-policy "$policy" --save-third-update "${history[@]}" \
    --out "runs/lilocbench_${policy}_clearance.json"
done
python scripts/eval_bonn_loop_closure.py \
  --data /path/to/extracted/bonn --annotations benchmarks/bonn/holdout_labels.json \
  --pair holdout_box_removed --map-tools target/release/examples --level-from-depth \
  --compare-removals --out runs/bonn_clearance.json
```
