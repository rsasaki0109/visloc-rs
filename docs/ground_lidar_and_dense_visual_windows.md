# Ground LiDAR occupancy and dense visual bridge evaluation

## KITTI ground LiDAR occupancy

KITTI Tracking sequence 0011 provides synchronized Velodyne scans, OXTS poses, and annotated 3D cuboids for a moving vehicle and two stationary vehicles. The initial map uses five scans. Four interleaved scans provide voxel labels from the annotated cuboids. The clearance replay uses eight later scans, starting 5.5 seconds after the last map scan. At the start of replay, the moving car's annotated center is more than 5 m from its last position in the map window. The replay represents a short outdoor revisit within one drive.

The evaluator creates 35 cm voxels from scan endpoints and checks measured rays against the exact voxel bounds. A valid free-space observation crosses the voxel, ends beyond it, and has no conflicting nearer return in the view. A removal requires at least two free-space scans and at least twice as many free votes as hit votes. The final update also retains voxels with a measured occupied endpoint in any replay scan. All tested policies share the same cuboid-derived labels. The pose correction fits a bounded point-to-plane transform using only LiDAR outside the region that is scored.

| Pose source and removal rule | Moved-vehicle voxels removed | Stationary voxels falsely removed | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| OXTS, occupancy difference | 134/134 | 56/117 | 70.5% | 100.0% |
| OXTS, rays plus occupied-endpoint veto | 119/134 | 47/117 | 71.7% | 88.8% |
| OXTS and background correction, occupancy difference | 131/134 | 4/117 | 97.0% | 97.8% |
| OXTS and background correction, ray evidence | 131/134 | 7/117 | 94.9% | 97.8% |
| OXTS and background correction, rays plus occupied-endpoint veto | 131/134 | 0/117 | 100.0% | 97.8% |

Background residuals fell from 40.7 cm to 2.0 cm. The measured correction at the sample centroid includes a vertical shift of 42.9 cm. This explains why registration quality dominates the raw clearance results. The saved map is reloaded for the final policy. Four of the 138 labeled moved-vehicle voxels were outside the initial occupancy, so the table's recall is conditional on the occupied map.

The insertion replay uses four later scans and labels from six separate scans. Newly visible stationary-vehicle surfaces are positives. Voxels from the moving vehicle are negatives. The applied rule requires occupation in at least two scans and excludes voxels with contradictory free-space evidence.

| Insertion rule with background correction | New stationary voxels added | Moving voxels incorrectly added | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| All returns | 36/58 | 184/343 | 16.4% | 62.1% |
| At least two scans | 22/58 | 45/343 | 32.8% | 37.9% |
| Two scans, without contradictory free evidence | 10/58 | 7/343 | 58.8% | 17.2% |

The strict insertion rule reduces moving-vehicle points at the cost of stationary-surface coverage. Six hundred and seven voxels were inserted in the cropped map, including voxels outside the cuboid labels.

These figures cover a single scene and three identities. They measure local map updates from ground LiDAR, with pose initialization from OXTS. Long-interval outdoor 3D map updates require independent revisits with physical change labels. [KITTI Tracking data and labels](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) · [Sequence protocol](../benchmarks/kitti/lidar_clearance_0011.json) · [Corrected replay results](../benchmarks/kitti/clearance_0011_refined.json) · [OXTS replay results](../benchmarks/kitti/clearance_0011_oxts.json)

## ROVER visual integration

A denser attempt sampled 13 RGB frames per season around a candidate outdoor approach. Its two partial reconstructions used shared exposures on only one side of the proposed bridge. A separate summer-only transfer used 57 images, at one-second intervals over most of the gap. It produced four disconnected pieces. The two end pieces shared five and six exposures with their respective existing submaps but none with the opposite submap.

A second diagnostic estimated submap similarities directly from calibrated two-view constraints. It initializes rotation from essential matrices, uses multiple baseline directions to initialize scale and translation, then minimizes robust Sampson error. Each fold withholds whole image pairs from the fit. The west-side summer/winter candidate produced usable hypotheses in only two of four folds. All three evaluated holdout pairs had zero matches within 3 px and median errors of 21.4–60.5 px. The known connected house pair had median errors below 4 px on every held-out pair, while a smaller potential connection fit in only two folds. The prior image-only component of 22 summer and 21 winter poses remains the largest supported join. [Dense window results](../benchmarks/rover/dense_transfer_results.json) · [Epipolar holdout results](../benchmarks/rover/epipolar_submap_results.json)

## Reproduce

The LiDAR fetcher verifies ZIP member checksums and fetches only the listed scans. Use Python with NumPy and SciPy installed.

```bash
export OPENBLAS_NUM_THREADS=1
python scripts/fetch_kitti_tracking_clearance.py \
  --annotations benchmarks/kitti/lidar_clearance_0011.json --out runs/kitti_clearance
python scripts/eval_kitti_lidar_clearance.py --data runs/kitti_clearance \
  --annotations benchmarks/kitti/lidar_clearance_0011.json --refine-background \
  --map-out runs/kitti_clearance/updated.npz --out runs/kitti_clearance/refined.json
```

Omit `--refine-background` and use a different output path to repeat the OXTS-only trial. Use the ROVER extracts, calibration, ONNX matcher and original site submaps from the [site evaluation](rover_site_and_clearance_revisits.md).

```bash
python scripts/fetch_rover_rgb_samples.py --out runs/rover \
  --dense-windows benchmarks/rover/dense_transfer_windows.json
python scripts/eval_rover_dense_windows.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" --site-results runs/site_alignment.json \
  --windows benchmarks/rover/dense_transfer_windows.json --local-models runs/dense_rover_models \
  --out runs/dense_windows.json
python scripts/eval_rover_epipolar_submaps.py --summer "$summer" --winter "$winter" \
  --calibration "$calibration" --efficient-loftr "$loftr" --site-results runs/site_alignment.json \
  --site-models runs/site_models --out runs/epipolar_submaps.json
```
