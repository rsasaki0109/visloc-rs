# EuRoC Covisibility BA Window Sweep

Generated from benchmark-registry run manifests. This sweep keeps
`--max-frames 400`,
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
| MH_01_easy | 5 | 5 | 0.357 | 0.0785 | 0.0769 | 16 | 10 |  |  | 586.744 | 1420.501 | euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T201509Z |
| MH_01_easy | 10 | 10 | 0.585 | 0.0607 | 0.0593 | 28 | 13 |  |  | 1100.109 | 3227.432 | euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T203510Z |
| MH_03_medium | 5 | 5 | 0.920 | 0.0639 | 0.0580 | 31 | 3 |  |  | 1200.106 | 1865.623 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T201509Z |
| MH_03_medium | 10 | 10 | 0.973 | 0.0394 | 0.0386 | 24 | 2 |  |  | 1132.111 | 2610.739 | euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T205812Z |
| MH_05_difficult | 5 | 5 | 0.343 | 0.1725 | 0.0971 | 26 | 8 |  |  | 271.650 | 477.573 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T201509Z |
| MH_05_difficult | 10 | 10 | 0.220 | 0.1683 | 0.0888 | 6 | 11 |  |  | 62.554 | 356.136 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T205812Z |

Notes:

- Runtime is wall-clock milliseconds measured inside `OnlineSlamPipeline` around each triggered covisibility local BA attempt.
- The landmark cap is fixed, so deltas mainly reflect keyframe-window selection and observation count.
- This is registry-backed evidence for choosing an opt-in BA window budget, not a headline benchmark claim.
