# EuRoC MH_05 Covisibility BA Mitigation

Generated from benchmark-registry run manifests. This targeted sweep keeps
`--max-frames 400`, neighbor/boundary `10/10`,
landmark cap `200`, active-observation floor `20`,
remove-outliers `False`, fallback boundary selection `none`,
max-outlier observation ratio `none`,
and boundary support gate `none/0`; only BA start/trigger cadence changes.

| config | tracking | rigid ATE m | sim ATE m | map keyframes | BA triggers | BA success | BA fail | quality reject | boundary support | no-local-landmarks | mean ms | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| disabled | 0.565 | 0.1139 | 0.1118 | 42 | 0 | 0 | 0 |  |  | 0 |  | euroc-covisibility-local-ba-MH_05_difficult-disabled-20260619T211409Z |
| enabled min3/every1 | 0.220 | 0.1683 | 0.0888 | 19 | 17 | 6 | 11 |  |  | 11 | 62.554 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T205812Z |
| enabled min6/every3 | 0.445 | 0.1142 | 0.1065 | 42 | 13 | 12 | 1 |  |  | 1 | 1054.491 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T215803Z |
| enabled min10/every5 | 0.525 | 0.1255 | 0.1243 | 43 | 7 | 7 | 0 |  |  | 0 | 1175.659 | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T220343Z |

Notes:

- `min3/every1` is the original 400-frame 10/10 enabled row that regressed MH_05.
- Later starts and less frequent triggers recover tracking stability but do not beat the disabled baseline yet.
- This is diagnostic evidence for the opt-in covisibility BA path, not a default-policy claim.
