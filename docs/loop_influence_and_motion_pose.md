# Loop influence and motion pose follow-up

The experiments reuse the manually labeled Bonn holdout and the OpenLORIS office visits `office1-1` and `office1-6`. Loop endpoint depth and extra validation frames supply the selection signals. Bonn map labels and both datasets' published reference poses are reserved for evaluation. [Prior Bonn evaluation](bonn_loop_and_depth_evaluation.md) · [OpenLORIS-Scene](https://lifelong-robotic-vision.github.io/dataset/scene)

## Removing one loop at a time

Bonn's holdout addition uses depth-floor leveling and the experimental 15 cm spacing. Each trial excludes one selected loop by its directed pair of source image IDs, after candidate ranking and spatial separation. The two other loops retain their original weights and measurements. The first base pose defines the same gauge for all trials. The table uses the same 140 changed and 1,017 unchanged reference voxels.

| Conditions | F1 | True positives | False positives | Missed changes | Second visit position RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Session registration only | 31.1% | 33 | 39 | 107 | 13.09 cm |
| All three loops | 25.3% | 25 | 33 | 115 | 13.50 cm |
| Exclude base 48, query 16 | 30.8% | 32 | 36 | 108 | 13.13 cm |
| Exclude base 24, query 12 | 25.7% | 26 | 36 | 114 | 13.74 cm |
| Exclude base 8, query 1 | 26.0% | 26 | 34 | 114 | 13.56 cm |

All single-exclusion trial updates were applied. The 48→16 edge had the largest influence on the regression. The paired poses, rotation errors and selected edges are in the report. [Bonn measurements](../benchmarks/bonn/loop_influence_results.json)

## Selecting a result from depth

The endpoint gate projects observed depth into the other image in both directions through each visual loop transform. It requires 100 comparable rays in each direction and 70% within 7 cm in both directions. Unavailable depth leaves the visual decision in place. The most influential Bonn loop passed with 87.3% agreement in its weaker direction. The gate excluded only the 8→1 edge, yielding 26.0% F1. On OpenLORIS it rejected eight of eight loops even though the query visit benefited from the full refinement in translation RMSE.

The second experiment uses a fixed set of co-view frame pairs chosen with the session-only poses, subsampled in time and excluding all loop endpoints. The Bonn probe also excludes the map scoring frames and annotated frames. These probes use available depth from the reconstructed visits. Each candidate solution is ranked by mean bidirectional agreement on the same pairs. A candidate needs at least five pairs and a 0.5 percentage-point gain over session registration to take precedence. Both datasets use the same depth tolerance and thresholds. This selects a complete pose solution before scoring its effect against the reference.

| Dataset | Frame pairs | Session-only agreement | All-loops agreement | Selected by depth | Reference result |
| --- | ---: | ---: | ---: | --- | --- |
| Bonn holdout addition | 7 | 57.1% | 49.0% | Session registration | Map F1 31.1% vs. 25.3% with all loops |
| OpenLORIS | 16 | 50.0% | 52.5% | All eight loops | Query translation RMSE 4.01 cm vs. 4.50 cm with session registration |

On OpenLORIS, base translation RMSE increased from 2.81 cm to 2.98 cm, and query rotation RMSE increased from 2.11° to 2.18°. The external check uses published reference trajectories. Bonn supplies the labeled change measurements. These two cases are an exploratory check of graph-level depth selection. The 15 cm separation and both depth checks remain experimental. [OpenLORIS measurements](../benchmarks/openloris/loop_depth_validation.json)

## Re-estimating camera poses for motion exclusion

The depth fallback uses prior reference depth. In the offline pose experiment, an undecided query can interpolate between earlier and later measured camera poses when the gap is at most five seconds and endpoint motion is limited to 0.5 m and 45°. A second pass tries to relocalize against two measured background frames, requiring agreement within 4 cm and 2°. The comparisons use the same scoring grid, motion boxes and depth vote thresholds as the previous Bonn test.

| Recording | Basalt | Causal estimated pose + depth | Offline estimated pose + depth | Mocap + depth |
| --- | ---: | ---: | ---: | ---: |
| Development recall | 28.36% | 28.36% | 28.36% | 34.35% |
| Holdout recall | 19.32% | 19.74% | 19.85% | 26.98% |

All seven holdout fallback queries received an interpolated pose. The two-view relocalizer added zero poses. At those queries, position RMSE was 46.8 cm for the causal estimate and 46.7 cm for the interpolated estimate. Holdout static retention was 98.16% for Basalt alone and 98.05% with offline depth. All five eligible development queries remained unlocalized. The experiments measure the benefit of depth with independent poses and the gap that remains with RGB-D tracking through occlusions. [Development motion counts](../benchmarks/bonn/development_offline_motion_results.json) · [Holdout motion counts](../benchmarks/bonn/holdout_offline_motion_results.json)

## Reproduce

Build the same examples and use the dataset layout from the prior Bonn and OpenLORIS benchmarks. The OpenLORIS script uses the two equalized source maps and aligned depth, and reports pose scores with a single alignment fit to the unmerged base trajectory. [Bonn inputs](bonn_map_followup.md#reproduce) · [OpenLORIS map preparation](openloris_map_localization.md)

```bash
cargo build --release --example basalt_map_merge --example tum_rgbd_save_map

OPENBLAS_NUM_THREADS=1 python scripts/eval_bonn_loop_closure.py \
  --data /path/to/bonn --map-tools target/release/examples \
  --annotations benchmarks/bonn/holdout_labels.json --pair holdout_box_added \
  --level-from-depth --leave-one-out --out runs/bonn_loop_influence.json

OPENBLAS_NUM_THREADS=1 python scripts/eval_openloris_loop_depth.py \
  --data /path/to/openloris --merge target/release/examples/basalt_map_merge \
  --out runs/openloris_loop_depth.json

for set in development holdout; do
  labels=map_change
  if [ "$set" = holdout ]; then labels=holdout; fi
  python scripts/eval_bonn_depth_motion.py --data /path/to/bonn \
    --annotations "benchmarks/bonn/${labels}_labels.json" \
    --baseline "benchmarks/bonn/${set}_followup_results.json" \
    --out "runs/bonn_offline_motion_${set}.json"
done
```
