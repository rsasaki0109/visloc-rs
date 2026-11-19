# Public revisit evaluation of loop selection, map reuse, and person exclusion

## Loop selection near map changes

The experiment compares global held-out depth agreement with confirmation of signed changes near candidate surfaces. It alternates the existing held-out RGB-D pairs between proposal and confirmation. Proposal depth builds a shared 7 cm surface grid from both pose hypotheses and marks additions and removals with at least two views and a 10 cm depth margin. Candidate regions with conflicting signs are excluded. The other split measures the fraction of candidates with the same sign under each hypothesis. The selector needs four pairs per split, 32 candidates, and a gain of at least 0.5 percentage points to prefer loop refinement. With insufficient pairs or candidates it uses the global depth selector. Annotation polygons and reference pose errors are used in scoring.

| Public data | Global selection F1 | Local selection F1 | Local decision |
| --- | ---: | ---: | --- |
| LILocBench box added | 31.0% | 29.6% | Session registration, 636 vs 580 of 1,269 candidate voxels confirmed |
| LILocBench old door surface removed | 19.1% | 25.4% | All loops, 3 vs 1 of 89 candidate voxels confirmed |
| Bonn holdout box added | 31.1% | 31.1% | Global fallback, three confirmation pairs |
| Bonn holdout box removed | 72.4% | 67.4% | Session registration, 361 vs 264 of 738 candidate voxels confirmed |

The local score favored the better F1 in the door case and missed the better solution in two cases. The door selection rested on two additional confirming voxels. These results support retaining the global rule for the existing benchmark selection. [Loop-selection results](../benchmarks/lilocbench/loop_selection_results.json)

## Saving and reusing a map on a third visit

LILocBench captures the same corridor first without the box, then with it about 10 h 24 min later, then after it was removed about 21 h 42 min later. The replay builds a sparse visual map from the first visit and saves a second version after merging the second visit with six keyframe loops. It also writes a versioned occupancy snapshot beside each sparse map. The occupancy update added 749 and removed 990 voxels. It incorrectly removed 197 of the 1,834 annotated stable voxels that were present in the initial snapshot. Two annotated stable voxels were added.

A fresh localizer process opens each saved sparse map. Its visual poses position the independent third visit's RGB-D frames when the replay reopens the occupancy snapshot and checks for free space at previously occupied sites. One common transform from the first usable frame aligns the world gauge to the reference labels. The visual map's landmark lifecycle and the occupancy layer are evaluated separately. Sparse localization uses image landmarks, and the occupancy update supplies the changed geometry to the free-space check.

| Saved map and pose source on visit 3 | Visual fixes or propagated poses | Position RMSE on available estimates | Box voxels observed free | Stable voxels marked free |
| --- | ---: | ---: | ---: | ---: |
| Initial, visual fixes | 15/49 | 1.63 cm | No stored box voxels | 293/1,834 |
| Updated, visual fixes | 12/49 | 2.00 cm | 16/47 | 130/1,639 |
| Updated, visual fixes with wheel fallback | 49/49 | 2.23 cm | 47/47 | 118/1,639 |
| Updated, published reference poses | 49 reference poses | — | 43/47 | 112/1,639 |

Position errors use the frames for which each method has a pose. Stable voxel fractions use each saved map's footprint and each method's available observations. Geometry warnings on the third visit represent candidate removals. The experiment measures these candidates before another saved-map update. [Three-visit results](../benchmarks/lilocbench/third_visit_results.json) · [LILocBench data](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/)

## Person exclusion under occlusion

Eight rectified front-camera frames from the third visit have hand-traced silhouettes. Four show the walking person crossing the camera view, and four show a standing person near the end of the corridor. The evaluation uses the labeled silhouette and nearby background, masks an eight-pixel band around the boundary, and scores pixels with valid 0.3–4 m depth. It checks these points against the first visit's background. Missing poses leave the pixels unfiltered. The wheel fallback propagates forward from the most recent successful visual pose and resets on subsequent fixes.

| Query pose | Labeled frames with poses | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Visual fixes | 2/8 | 98.0% | 40.8% | 57.6% |
| Visual fixes with wheel fallback | 8/8 | 73.0% | 96.9% | 83.3% |
| Published reference poses | 8/8 | 70.6% | 97.0% | 81.8% |

The reference-pose run also excludes background pixels, which shows that errors persist with the reference trajectory. These figures measure exclusion within annotated regions over one short corridor pass. Autonomous person detection and extended wheel-only operation would need separate evaluation. [Silhouette annotations](../benchmarks/lilocbench/third_visit_motion.json)

## 148-day revisit in a second environment

ROVER Garden Small provides summer and snowy winter captures on 2023-08-18 and 2024-01-13. Four manually reviewed overlapping image pairs, six non-overlap pairs, and three same-season controls test correspondence verification on actual RGB frames. SIFT uses mutual 0.75-ratio matches, followed by a 2 px epipolar RANSAC threshold. Acceptance requires at least 15 inliers and an inlier fraction of at least 35%.

The verifier accepted all three same-season controls with 67–136 inliers and rejected all ten season-crossing pairs. The strongest cross-season pair had nine inliers on the house facade. This experiment reached correspondence verification. A seasonal pose-graph or map-update score requires metric cross-season alignment and change labels. The public total-station files give prism positions with empty orientation values. [ROVER download and calibration](https://iis-esslingen.github.io/rover/pages/download/) · [Public archive mirror](https://huggingface.co/datasets/iis-esslingen/ROVER) · [ROVER sample labels and results](../benchmarks/rover/seasonal_results.json)

A subsequent evaluation adds a learned matcher, reconstructs the facade across both seasons, and tests saved-map clearance with observation history. [Seasonal alignment and map clearance](seasonal_alignment_and_map_clearance.md)

## Reproduce

Use the dependencies and two-window extraction from the [LILocBench corridor evaluation](lilocbench_map_changes.md). Build the map tools with the `image-io` feature.

```bash
cargo build --release --features image-io --example basalt_map_merge \
  --example tum_rgbd_save_map --example basalt_euroc_localize_demo
export OPENBLAS_NUM_THREADS=1

python scripts/fetch_lilocbench_windows.py \
  --annotations benchmarks/lilocbench/third_visit.json --out /path/to/liloc
python scripts/eval_lilocbench_changes.py \
  --data /path/to/liloc --annotations benchmarks/lilocbench/corridor_labels.json \
  --map-tools target/release/examples --out runs/lilocbench_loop_selection.json
python scripts/eval_lilocbench_reuse.py \
  --data /path/to/liloc --annotations benchmarks/lilocbench/corridor_labels.json \
  --motion-annotations benchmarks/lilocbench/third_visit_motion.json \
  --map-tools target/release/examples --maps-out runs/lilocbench_saved_maps \
  --out runs/lilocbench_third_visit.json

python scripts/eval_bonn_loop_closure.py \
  --data /path/to/extracted/bonn --annotations benchmarks/bonn/holdout_labels.json \
  --map-tools target/release/examples --level-from-depth --leave-one-out \
  --out runs/bonn_loop_selection.json

python scripts/eval_rover_seasonal_overlap.py \
  --summer /path/to/extracted/2023-08-18 --winter /path/to/extracted/2024-01-13 \
  --calibration /path/to/calibration/calib_d435i.yaml \
  --annotations benchmarks/rover/seasonal_pairs.json --out runs/rover_seasonal.json
```

Each ROVER date directory contains `realsense_D435i/rgb.txt` and the corresponding images. The script takes the nearest timestamp to each labeled offset from the start of its RGB stream. Raw captures and saved benchmark maps are kept outside Git.
