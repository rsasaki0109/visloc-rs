# Bonn keyframe loop and motion depth evaluation

This evaluation uses the original and holdout Bonn RGB-D recordings from the labeled map benchmark. It measures why keyframe loop constraints are rejected, what admitted loops change in the map, and how much recently observed depth can recover from Basalt's background-support failures. [Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html) · [Annotations and previous measurements](bonn_map_followup.md)

## Loop rejection and map updates

The loop report counts posed query frames, successful registrations, anchored candidates and the first rejection reason for each candidate. The gates cover session agreement, anchor inliers, PnP reprojection, rig-frame availability, relative translation and rotation residuals, and spatial separation in both visits. Each pair still needs at least two separated measurements. A graph update is applied only when its gravity-preserving result lowers cost.

The comparison uses identical RGB-D prediction frames and labeled voxels within each case. An SE(3) transform from the first base-visit pose places both estimated sessions in the mocap world. Translation and rotation RMSE are scored at the predicted map frames. The standard map exporter fits a horizontal plane from sparse landmarks. It now gives that fit a stable keyframe and landmark order. The alternative fits a dominant near-horizontal plane per depth image from the lower image region, requires consistent normals across at least five frames, and rotates the per-visit VO trajectory before export. Both orientations are estimated without mocap.

We run the session registration alone, the standard 35 cm loop gate, and an experimental 15 cm gate for the short repeat paths. Matching, reprojection, and placement residual thresholds are shared.

| Leveling | Case | Mocap F1 | Session-only F1 | 35 cm F1 | 15 cm F1 | 15 cm applied loops | 15 cm vs. session-only occluded floor FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse | Development addition | 75.1% | 63.3% | 63.3% | 63.3% | 0 | 0 → 0 |
| Sparse | Development removal | 80.2% | 30.6% | 30.6% | 30.6% | 0 | 1 → 1 |
| Sparse | Holdout addition | 83.3% | 55.9% | 55.9% | 55.9% | 0 | 0 → 0 |
| Sparse | Holdout removal | 54.2% | 34.1% | 34.1% | 34.1% | 0 | 1 → 1 |
| Depth floor | Development addition | 75.1% | 60.7% | 60.7% | 60.7% | 0 | 0 → 0 |
| Depth floor | Development removal | 80.2% | 67.1% | 67.1% | 69.6% | 2 | 0 → 0 |
| Depth floor | Holdout addition | 83.3% | 31.1% | 31.1% | 25.3% | 3 | 1 → 1 |
| Depth floor | Holdout removal | 54.2% | 67.4% | 67.4% | 72.4% | 2 | 0 → 0 |

With sparse leveling at 35 cm, all 204 posed queries had anchored candidates. The development addition had 55 candidates rejected for separation after the first choice. Translation and rotation residuals accounted for the other three pairs. Depth-floor leveling removed the rotation rejections on these pairs. One translation rejection remained in each holdout pair.

With depth-floor leveling, the 15 cm gate applied 2, 3 and 2 loop constraints on three pairs. The holdout addition is a regression. For its second visit, translational RMSE increased from 13.1 cm to 13.5 cm even as the gravity-preserving graph cost decreased. The holdout removal illustrates that the voxel F1 can rise even with higher pose error. Candidate discretization, the fixed tolerance and the set of annotated surfaces also influence this score.

[Sparse leveling counts and pose errors](../benchmarks/bonn/keyframe_sparse_alignment_results.json) · [Depth-floor leveling counts and pose errors](../benchmarks/bonn/keyframe_floor_alignment_results.json)

## Depth fallback for motion

The fallback runs on annotated proposal frames where Basalt could not decide because of missing background support. It preserves Basalt's confirmed pixels. Within the undecided boxes, it suppresses a depth-backed pixel when at least two previous views within five seconds had valid 3 × 3 depth neighborhoods at least 15 cm behind that pixel, and free-space votes outnumber surface votes by at least two to one. Earlier candidate boxes are masked when available. The two pose sources are motion capture and RGB-D PnP against previous background observations, with up to 0.35 s of velocity extrapolation. The latter uses only measured poses for references.

Metrics use the same masks and border handling as the earlier motion benchmark, on a shared 4 px scoring grid. Proposal boxes and object IDs still come from the hand annotations. This experiment calls the RGB-D visibility routine for evidence. It evaluates motion pixel exclusion with the fallback in the benchmark.

| Recording | Pose source for fallback | Precision | Recall | Static retention |
| --- | --- | ---: | ---: | ---: |
| Development | Basalt only | 91.7% | 28.4% | 97.95% |
| Development | Estimated pose | 91.7% | 28.4% | 97.95% |
| Development | Mocap | 92.9% | 34.3% | 97.90% |
| Holdout | Basalt only | 92.4% | 19.3% | 98.16% |
| Holdout | Estimated pose | 92.1% | 19.7% | 98.05% |
| Holdout | Mocap | 94.4% | 27.0% | 98.16% |

With inferred poses, every fallback target was unlocalized on the development sequence. On the holdout all seven fallback frames used extrapolated poses. The holdout estimate recovered 718 additional labeled motion samples and suppressed 164 additional static samples. Motion capture recovered 7,218 development and 13,118 holdout motion samples. The fallback remains a benchmark experiment and requires reliable query poses.

[Development pixel counts](../benchmarks/bonn/development_depth_motion_results.json) · [Holdout pixel counts](../benchmarks/bonn/holdout_depth_motion_results.json)

## Reproduce

Use the archives and dependencies listed in the previous Bonn evaluation. Build the map exporter and the map merge example. [Input hashes and downloads](bonn_map_followup.md#reproduce)

```bash
cargo build --release --example basalt_map_merge --example tum_rgbd_save_map

for mode in sparse depth; do
  options=()
  if [ "$mode" = depth ]; then options+=(--level-from-depth); fi
  OPENBLAS_NUM_THREADS=1 python scripts/eval_bonn_loop_closure.py \
    --data /path/to/extracted/bonn --map-tools target/release/examples \
    --annotations benchmarks/bonn/map_change_labels.json \
    --annotations benchmarks/bonn/holdout_labels.json \
    --out "runs/bonn_loop_${mode}.json" "${options[@]}"
done

for set in development holdout; do
  labels=map_change
  if [ "$set" = holdout ]; then labels=holdout; fi
  python scripts/eval_bonn_depth_motion.py --data /path/to/extracted/bonn \
    --annotations "benchmarks/bonn/${labels}_labels.json" \
    --baseline "benchmarks/bonn/${set}_followup_results.json" \
    --out "runs/bonn_depth_motion_${set}.json"
done
```
