# EuRoC MH_05 Covisibility BA Mitigation

Generated from benchmark-registry run manifests. This targeted sweep keeps
`--max-frames 400`, neighbor/boundary `10/10`,
landmark cap `200`, active-observation floor `20`,
remove-outliers `False`, fallback boundary selection `none`,
max-outlier observation ratio `0.3`,
and boundary support gate `10/2`; only BA start/trigger cadence changes.

| config | tracking | rigid ATE m | sim ATE m | map keyframes | BA triggers | BA success | BA fail | quality reject | boundary support | no-local-landmarks | mean ms | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| disabled | 0.565 | 0.1139 | 0.1118 | 42 | 0 | 0 | 0 | 0 | 0 | 0 |  | euroc-covisibility-local-ba-MH_05_difficult-disabled-20260703T170345Z |
| enabled min3/every1 boundary10 | 0.265 | 0.1614 | 0.1003 | 27 | 25 | 12 | 13 | 2 | 2 | 9 | 254.946 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260620T011742Z |

Notes:

- `min3/every1` is the original 400-frame 10/10 enabled row that regressed MH_05.
- Later starts and less frequent triggers recover tracking stability but do not beat the disabled baseline yet.
- This is diagnostic evidence for the opt-in covisibility BA path, not a default-policy claim.
