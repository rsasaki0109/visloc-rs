# EuRoC Covisibility BA Window Sweep

Generated from benchmark-registry run manifests. This sweep keeps
`--max-frames 80`,
`--covisibility-local-ba-max-landmarks 200`,
`--covisibility-local-ba-min-keyframes 3`,
`--covisibility-local-ba-trigger-every 1`,
`--covisibility-local-ba-min-active-observations 20`,
remove-outliers `False`,
fallback boundary selection `none`,
max-outlier observation ratio `none`,
and boundary support gate `none/0`; only the local BA
neighbor/boundary keyframe caps change.

| sequence | neighbor KF | boundary KF | tracking | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | max ms | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MH_01_easy | 5 | 5 | 1.000 | 0.0181 | 0.0158 | 12 | 0 |  |  | 768.583 | 1205.720 | euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T200124Z |
| MH_01_easy | 10 | 10 | 1.000 | 0.0173 | 0.0158 | 12 | 0 |  |  | 1097.608 | 2446.577 | euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T200340Z |
| MH_01_easy | 15 | 15 | 1.000 | 0.0173 | 0.0141 | 12 | 0 |  |  | 1231.720 | 2830.529 | euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T200602Z |
| MH_03_medium | 5 | 5 | 0.975 | 0.0558 | 0.0522 | 8 | 0 |  |  | 522.977 | 662.327 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T195041Z |
| MH_03_medium | 10 | 10 | 0.938 | 0.0544 | 0.0512 | 8 | 0 |  |  | 685.996 | 1188.856 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T191258Z |
| MH_03_medium | 15 | 15 | 0.938 | 0.0544 | 0.0512 | 8 | 0 |  |  | 641.184 | 988.181 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T195153Z |
| MH_05_difficult | 5 | 5 | 0.825 | 0.0502 | 0.0471 | 1 | 0 |  |  | 75.621 | 75.621 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T200124Z |
| MH_05_difficult | 10 | 10 | 0.825 | 0.0502 | 0.0471 | 1 | 0 |  |  | 76.215 | 76.215 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T200340Z |
| MH_05_difficult | 15 | 15 | 0.825 | 0.0502 | 0.0471 | 1 | 0 |  |  | 73.244 | 73.244 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T200602Z |

Notes:

- Runtime is wall-clock milliseconds measured inside `OnlineSlamPipeline` around each triggered covisibility local BA attempt.
- The landmark cap is fixed, so deltas mainly reflect keyframe-window selection and observation count.
- This is registry-backed evidence for choosing an opt-in BA window budget, not a headline benchmark claim.
