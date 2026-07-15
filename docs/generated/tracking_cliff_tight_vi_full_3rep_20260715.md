# Tracking-cliff tight-VI full-sequence matrix

Runs: 18 (3 sequences × 2 variants × 3 counterbalanced repetitions).
Overall gate: **PASS**.

| Sequence | Variant | Tracking | Longest | Rigid ATE m | d1 m / deg | d10 m / deg | Loop precision | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MH_01_easy | control | 0.905 | 558 | 2.9272 | 0.2538 / 4.7720 | 1.5217 / 21.1040 | 1.000 | 798.3 [763.7, 819.2] |
| MH_01_easy | candidate | 0.905 | 558 | 2.9272 | 0.2538 / 4.7720 | 1.5217 / 21.1040 | 1.000 | 1062.9 [926.6, 1083.5] |
| MH_03_medium | control | 0.767 | 215 | 3.6816 | 0.4357 / 6.0837 | 3.2276 / 33.9538 | 1.000 | 604.8 [516.0, 671.6] |
| MH_03_medium | candidate | 0.767 | 215 | 3.6816 | 0.4357 / 6.0837 | 3.2276 / 33.9538 | 1.000 | 705.3 [660.1, 796.4] |
| MH_05_difficult | control | 0.843 | 144 | 6.4940 | 0.4730 / 5.0329 | 3.7482 / 27.3704 | 1.000 | 352.0 [322.5, 356.7] |
| MH_05_difficult | candidate | 0.884 | 391 | 6.3923 | 0.4506 / 5.0747 | 3.5838 / 27.8397 | 1.000 | 390.5 [326.9, 416.1] |

## Candidate delta from control

- MH_01_easy: tracking +0.00%, longest +0.00%, ATE +0.00%, d1 trans/rot +0.00%/+0.00%, d10 trans/rot +0.00%/+0.00%.
- MH_03_medium: tracking +0.00%, longest +0.00%, ATE +0.00%, d1 trans/rot +0.00%/+0.00%, d10 trans/rot +0.00%/+0.00%.
- MH_05_difficult: tracking +4.86%, longest +171.53%, ATE -1.57%, d1 trans/rot -4.74%/+0.83%, d10 trans/rot -4.39%/+1.71%.

Gate: no tracking/continuity/loop-precision regression; each ATE/RPE metric within 2%; runtime is report-only; at least one cliff sequence must improve tracking and longest continuity.
