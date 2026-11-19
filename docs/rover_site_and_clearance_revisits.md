# ROVER site alignment and persistent clearance across revisits

## Seasonal site alignment

The wider ROVER Garden Small run samples the full summer and winter traversals every 3 seconds. DINOv2 ranks cross-season and within-season loop candidates. Consecutive frames also qualify for verification. SIFT checks within-season geometry, with EfficientLoFTR used for weak adjacent pairs and cross-season candidates. Image matching and joint reconstruction use RGB and camera calibration.

| Result | Summer | Winter |
| --- | ---: | ---: |
| Images sampled | 114 | 116 |
| Images registered in at least one image-only submap | 56 | 72 |
| Published positions held out of the seasonal bridge fit | 102 | 108 |
| Median distance to the other surveyed path after frame alignment | 0.72 m | 0.66 m |

The image-only run verified 451 pairs, including 40 cross-season pairs, and produced 15 disconnected reconstructions. Every reconstruction contains images from a single season. The site-scale experiment therefore uses the previously verified facade reconstruction to bridge the two surveyed frames. The bridge fits yaw and translation in each surveyed frame, using the metric scale from summer and a shared vertical estimate from the facade cameras. The median nearest-path distance before applying the bridge was 7.25 m for summer and 7.02 m for winter. The two published paths are from different traversals, so path proximity measures agreement of traversed areas. A survey-only refinement initialized by that bridge aligns the first 210 seconds of each path using robust bidirectional nearest-neighbor matching in the horizontal plane. On the segment after 225 seconds, excluding the facade fit window, winter-to-summer median path distance fell from 0.72 m to 0.33 m. The reverse direction fell from 0.70 m to 0.24 m. Absolute cross-season trajectory accuracy requires shared survey control. [Image-only metrics](../benchmarks/rover/site_image_only_results.json)

## Saved sparse map update

The site experiment fits each single-season submap to its synchronized published prism positions. It keeps points observed by at least three cameras whose prism fit is within 30 cm and requires at least four such cameras per submap. One submap per season passes, each with ten supported camera poses. In the refined site frame, the map update keeps the 250 summer landmarks and adds 116 of the 150 winter landmarks whose nearest summer point is more than 1 m away. The saved map reopens with 366 landmarks. Seven winter landmarks are within 25 cm of a summer landmark. These counts characterize sparse map overlap and additions under the surveyed initialization.

The facade provides a second geometric check. In the joint visual model, 30 tracks had enough parallax to triangulate separately from summer and winter images. Their median 3D disagreement was 11.7 cm, with 25 within 25 cm. For winter-only tracks, the replay saved 36 new landmarks after fitting from earlier observations and checking the last observation at a 2 px threshold. Their median held-out reprojection error was 1.33 px. The bundle-adjusted camera poses are shared, so this check evaluates landmark support conditional on those poses.

Dense seasonal change precision and recall still require depth-to-color calibration with reliable cross-season registration and independent physical change labels. The saved ROVER snapshots store sparse landmarks. [Site reference metrics](../benchmarks/rover/site_reference_results.json) · [Facade map metrics](../benchmarks/rover/facade_sparse_update_results.json) · [ROVER data](https://iis-esslingen.github.io/rover/pages/download/)

## Clearing pending removals on independent LILocBench visits

The LILocBench replay loads the saved map with the placed corridor box. It processes two later recordings in timestamp order, `lt_changes_dynamics_1` followed by `lt_changes_dynamics_0` about 1 h 43 min later. Review of the front RGB images shows the original box location clear on both visits. The earlier revisit has no published reference pose. The saved map provides 25/45 direct visual fixes and 36/45 poses with causal wheel propagation. The later revisit provides 12/49 direct fixes and 49/49 propagated poses.

The replay uses a 5 × 5 depth patch with a 15 cm margin. The first independent clear observation saves a voxel as pending. A later independent clear observation deletes it, occupied evidence resets it, and uncertainty preserves it. Retrying a visit ID cannot advance its own pending entries. The replay writes and reopens the pending state between the two visits.

| Policy across both revisits | Removed box voxels | Wrongly removed stable voxels |
| --- | ---: | ---: |
| Remove on the first clear visit | 47/47 | 150/1,742 |
| Remove after two clear visits | 43/47 | 57/1,742 |

The first clear visit left all 47 box voxels pending. The second cleared 43 of those and kept four pending, while its occupied evidence reset 34 pending annotated stable voxels. Both policies start from the same saved footprint and use the same observations. [Clearance results](../benchmarks/lilocbench/clearance_revisit_results.json) · [LILocBench data](https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/)

## Reproduce

Install the Python and Basalt dependencies from the [previous evaluation](seasonal_alignment_and_map_clearance.md). The site run uses the outdoor EfficientLoFTR model from that report and the uint8 DINOv2 Small ONNX model with SHA-256 `6d10d812e13012b8a0c9f29c04db0b06d6704270649cea734443ed0a9c764312`. The range fetcher retrieves the site and facade samples from the pinned public archive. [Pinned DINOv2 model](https://huggingface.co/Xenova/dinov2-small/blob/c2bb04a51fab207c420665f1946016107bffc701/onnx/model_uint8.onnx)

```bash
export OPENBLAS_NUM_THREADS=1
python scripts/fetch_rover_rgb_samples.py --out runs/rover \
  --facade-window benchmarks/rover/facade_window.json
summer=runs/rover/garden_small_summer_2023-08-18
winter=runs/rover/garden_small_winter_2024-01-13
calibration=/path/to/calib_d435i.yaml
loftr=/path/to/eloftr_outdoor_full.onnx

python scripts/eval_rover_site_alignment.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" --dinov2 /path/to/model_uint8.onnx \
  --models-out runs/site_models --out runs/site_alignment.json
python scripts/eval_rover_seasonal_geometry.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" \
  --models-out runs/facade_models --out runs/facade_alignment.json
scale=$(jq -r '.models[] | select(.meters_per_model_unit != null) | .meters_per_model_unit' runs/facade_alignment.json)
python scripts/eval_rover_reference_alignment.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --facade-model runs/facade_models/0 --facade-scale "$scale" \
  --site-models runs/site_models --maps-out runs/rover_reference_maps --out runs/rover_reference_site.json
python scripts/eval_rover_sparse_update.py --model runs/facade_models/0 --summer-frames 6 \
  --meters-per-unit "$scale" --maps-out runs/rover_facade_maps --out runs/rover_facade_update.json

liloc=/path/to/liloc
python scripts/fetch_lilocbench_windows.py \
  --annotations benchmarks/lilocbench/clearance_revisit.json --out "$liloc"
python scripts/eval_lilocbench_clearance_revisits.py --data "$liloc" \
  --annotations benchmarks/lilocbench/corridor_labels.json --saved-maps runs/lilocbench_patch_maps \
  --map-tools target/release/examples --work runs/lilocbench_clearance_revisits \
  --out runs/lilocbench_clearance_across_visits.json
```

The LILocBench command uses the `patch` map and the other three recordings prepared in the preceding evaluation.

The next evaluation tests image-only links between seasonal submaps and measures real building changes against public Rotterdam imagery and airborne LiDAR. [Outdoor changes and visual bridges](outdoor_changes_and_visual_bridges.md)
