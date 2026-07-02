# EuRoC Covisibility BA Active-Observation Sweep

Generated from benchmark-registry run manifests. The sweep keeps covisibility BA enabled,
`--covisibility-local-ba-max-landmarks 200`, fallback boundary selection disabled,
`--keyframe-tracked-landmark-ratio 0.9`, and `--max-frames 400`; only
`--covisibility-local-ba-min-active-observations` changes.

Recommendation from this sweep: use `20` as the MH smoke-run opt-in value.
`50` is more conservative, but it drops MH_05 tracked-drop continuity while not
improving the cross-sequence picture enough to justify the stricter gate.

| floor | sequence | variant | tracking | rigid ATE m | sim ATE m | map KF | BA success | BA fail | active gate | no local | solver fail | run id |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 20 | MH_01_easy | fixed | 0.420 | 2.2880 | 0.1686 | 60 | 10 | 48 | 5 | 43 | 0 | euroc-keyframe-tracked-landmark-drop-MH_01_easy-fixed-20260619T125631Z |
| 20 | MH_01_easy | tracked_drop | 0.412 | 2.5556 | 0.1850 | 54 | 12 | 40 | 3 | 37 | 0 | euroc-keyframe-tracked-landmark-drop-MH_01_easy-tracked_drop-20260619T125631Z |
| 20 | MH_03_medium | fixed | 0.820 | 1.4787 | 0.1588 | 65 | 29 | 34 | 4 | 25 | 0 | euroc-keyframe-tracked-landmark-drop-MH_03_medium-fixed-20260619T125631Z |
| 20 | MH_03_medium | tracked_drop | 0.802 | 1.5885 | 0.1487 | 66 | 35 | 29 | 4 | 24 | 0 | euroc-keyframe-tracked-landmark-drop-MH_03_medium-tracked_drop-20260619T125631Z |
| 20 | MH_05_difficult | fixed | 0.297 | 6.9441 | 0.1445 | 46 | 3 | 41 | 1 | 38 | 0 | euroc-keyframe-tracked-landmark-drop-MH_05_difficult-fixed-20260619T125631Z |
| 20 | MH_05_difficult | tracked_drop | 0.323 | 6.0872 | 0.1453 | 50 | 3 | 45 | 1 | 44 | 0 | euroc-keyframe-tracked-landmark-drop-MH_05_difficult-tracked_drop-20260619T125631Z |
| 50 | MH_01_easy | fixed | 0.470 | 2.1526 | 0.1792 | 57 | 12 | 43 | 6 | 37 | 0 | euroc-keyframe-tracked-landmark-drop-MH_01_easy-fixed-20260619T125821Z |
| 50 | MH_01_easy | tracked_drop | 0.445 | 2.3990 | 0.1859 | 58 | 10 | 46 | 5 | 41 | 0 | euroc-keyframe-tracked-landmark-drop-MH_01_easy-tracked_drop-20260619T125821Z |
| 50 | MH_03_medium | fixed | 0.823 | 1.5865 | 0.1540 | 73 | 25 | 46 | 18 | 27 | 0 | euroc-keyframe-tracked-landmark-drop-MH_03_medium-fixed-20260619T125821Z |
| 50 | MH_03_medium | tracked_drop | 0.805 | 1.5851 | 0.1462 | 71 | 31 | 38 | 13 | 25 | 0 | euroc-keyframe-tracked-landmark-drop-MH_03_medium-tracked_drop-20260619T125821Z |
| 50 | MH_05_difficult | fixed | 0.297 | 6.9441 | 0.1445 | 46 | 3 | 41 | 1 | 38 | 0 | euroc-keyframe-tracked-landmark-drop-MH_05_difficult-fixed-20260619T125821Z |
| 50 | MH_05_difficult | tracked_drop | 0.265 | 5.6797 | 0.1311 | 37 | 2 | 33 | 3 | 30 | 0 | euroc-keyframe-tracked-landmark-drop-MH_05_difficult-tracked_drop-20260619T125821Z |

Notes:

- `fixed` leaves tracked-landmark keyframe promotion disabled.
- `tracked_drop` enables `--keyframe-tracked-landmark-ratio 0.9` with a count floor of 20.
- `no local` means the selected local BA window had no eligible landmarks after the strict boundary threshold.
- `solver fail` stayed at zero in the recorded sweep; the failures are selection/gating diagnostics, not optimizer crashes.
