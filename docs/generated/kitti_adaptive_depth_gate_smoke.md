# KITTI Adaptive Depth Gate Smoke

Generated from benchmark-registry run manifests. This is a diagnostic
A/B smoke for the rectified-stereo depth gate, not a trajectory benchmark
claim. The run uses a stride-20 KITTI seq00 subset, `--max-frames 2`,
`--frontend deep`, `--deep-max-features 300`, and disables stereo BA.

Dataset checksum: `sha256_tree_v1 892e15f703c484dc01cff76896586532d5bc96a2008863f08bb4f90280c07e94`.

| variant | frames | effective min depth m | candidates mean | accepted mean | depth quantile mean m | VO ATE RMSE m | diagnostics artifact | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| adaptive | 2 | 3 | 219 | 206.5 | 14.381 | 16.4839 | yes | kitti-adaptive-depth-gate-smoke-00-adaptive-20260620T020000Z |
| fixed | 2 | 3 | 219 | 206.5 |  | 16.4839 | yes | kitti-adaptive-depth-gate-smoke-00-fixed3-20260620T020001Z |

Interpretation: on this far-field KITTI smoke subset, the adaptive
policy remains at the bounded `3 m` effective lower-depth floor, so it
matches the legacy fixed-3m replay while still recording per-frame
adaptive diagnostics for audit.

## Recorded Failures

| variant | frames requested | status | failure reason | run id |
| --- | ---: | --- | --- | --- |
| adaptive | 6 | failure | KabschFailed { pair_index: 1, correspondence_count: 81, min_inliers: 8 } during 6-frame adaptive smoke | kitti-adaptive-depth-gate-smoke-00-adaptive-failure-6frames-20260620T020002Z |
