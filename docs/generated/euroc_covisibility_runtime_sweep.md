# EuRoC Covisibility BA Runtime Sweep

Generated from benchmark-registry run manifests. This smoke sweep keeps
`--max-frames 80`,
`--covisibility-local-ba-max-neighbor-keyframes 10`,
`--covisibility-local-ba-max-boundary-keyframes 10`,
`--covisibility-local-ba-min-active-observations 20`,
remove-outliers `False`,
fallback boundary selection `none`,
max-outlier observation ratio `none`,
and boundary support gate `none/0`; only the local BA
`--covisibility-local-ba-max-landmarks` cap changes.

| sequence | max landmarks | tracking | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | max ms | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MH_03_medium | 100 | 0.900 | 0.0714 | 0.0709 | 7 | 0 |  |  | 350.970 | 598.861 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T192148Z |
| MH_03_medium | 200 | 0.938 | 0.0544 | 0.0512 | 8 | 0 |  |  | 685.996 | 1188.856 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T191258Z |
| MH_03_medium | 400 | 0.975 | 0.0588 | 0.0508 | 8 | 0 |  |  | 707.277 | 1309.790 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T192301Z |

Notes:

- Runtime is wall-clock milliseconds measured inside `OnlineSlamPipeline` around each triggered covisibility local BA attempt.
- `mean ms` averages every trigger, including selection failures; `max ms` exposes single-frame BA spikes.
- This is smoke evidence for choosing an opt-in BA window budget, not a headline benchmark claim.
