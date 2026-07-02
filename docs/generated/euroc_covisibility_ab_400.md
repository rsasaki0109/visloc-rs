# EuRoC Covisibility BA A/B

Generated from benchmark-registry run manifests. This comparison keeps
`--max-frames 400` and compares the disabled baseline against
enabled covisibility local BA with neighbor/boundary `10/10`,
min-keyframes `3`,
trigger-every `1`,
landmark cap `200`,
active-observation floor `20`,
remove-outliers `False`,
fallback boundary selection `none`,
max-outlier observation ratio `none`,
and boundary support gate `none/0`.

| sequence | disabled tracking | enabled tracking | tracking delta | disabled rigid ATE m | enabled rigid ATE m | rigid improvement m | disabled sim ATE m | enabled sim ATE m | BA success | BA fail | quality reject | boundary support | mean ms | verdict | run ids |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MH_01_easy | 0.380 | 0.585 | 0.205 | 0.0642 | 0.0607 | 0.0035 | 0.0617 | 0.0593 | 28 | 13 |  |  | 1100.109 | win | euroc-covisibility-local-ba-MH_01_easy-disabled-20260619T211409Z<br>euroc-covisibility-local-ba-MH_01_easy-enabled-20260619T203510Z |
| MH_03_medium | 0.865 | 0.973 | 0.108 | 0.0648 | 0.0394 | 0.0254 | 0.0629 | 0.0386 | 24 | 2 |  |  | 1132.111 | win | euroc-covisibility-local-ba-MH_03_medium-disabled-20260619T211409Z<br>euroc-covisibility-local-ba-MH_03_medium-enabled-20260619T205812Z |
| MH_05_difficult | 0.565 | 0.220 | -0.345 | 0.1139 | 0.1683 | -0.0544 | 0.1118 | 0.0888 | 6 | 11 |  |  | 62.554 | regress | euroc-covisibility-local-ba-MH_05_difficult-disabled-20260619T211409Z<br>euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T205812Z |

Notes:

- Positive `tracking delta` means enabled BA tracked more frames than the disabled baseline.
- Positive `rigid improvement` means enabled BA reduced rigid ATE RMSE.
- This is scoped A/B evidence for the opt-in covisibility local BA path, not a headline benchmark claim.
