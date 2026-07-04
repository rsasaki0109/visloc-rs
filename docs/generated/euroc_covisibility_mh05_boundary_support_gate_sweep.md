# EuRoC MH_05 Boundary Support Gate Sweep

Generated from benchmark-registry run manifests. This diagnostic table keeps
`--max-frames 400`, min-keyframes `3`, trigger-every `1`,
neighbor/boundary `10/10`,
landmark cap `200`, active-observation floor `20`,
max-outlier observation ratio `0.3`,
and varies only the pre-solve boundary-support gate.

| config | gate min opt/fixed | tracking | tracking delta vs qg | rigid ATE m | sim ATE m | BA success | BA fail | quality reject | boundary support | no-local-landmarks | mean ms | mean ms delta vs qg | verdict | run id |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| disabled | disabled | 0.565 |  | 0.1139 | 0.1118 | 0 | 0 | 0 | 0 | 0 |  |  |  | euroc-covisibility-local-ba-MH_05_difficult-disabled-20260703T170345Z |
| quality-gate only | none/0 | 0.265 | 0.000 | 0.1614 | 0.1003 | 12 | 13 | 4 |  | 9 | 304.495 | 0.000 | neutral | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T224943Z |
| boundary7/2 | 7/2 | 0.215 | -0.050 | 0.1535 | 0.0910 | 5 | 9 | 0 | 4 | 5 | 74.806 | -229.689 | reject | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260620T003448Z |
| boundary10/2 | 10/2 | 0.265 | 0.000 | 0.1614 | 0.1003 | 12 | 13 | 2 | 2 | 9 | 254.946 | -49.549 | candidate | euroc-covisibility-local-ba-MH_05_difficult-enabled-20260620T011742Z |

Notes:

- `qg` is the quality-gate-only enabled row with no pre-solve boundary-support gate.
- Negative `mean ms delta vs qg` means the pre-solve gate reduced average covisibility-BA trigger time.
- `candidate` preserves quality-gate-only tracking while reducing mean trigger time; `reject` loses tracking.
- This is diagnostic evidence for one MH_05 failure mode, not a default-policy claim.
