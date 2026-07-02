# EuRoC Tight VIO Gate Smoke

Generated from benchmark-registry run manifests. This table compares
`--max-frames 1500` local VI-BA smoke runs with the same HOG/cross-check
front-end and motion-IMU warm start; only local VI-BA writeback gates change.

Recommendation from this smoke: prefer `adaptive_velocity` over a raw fixed
velocity cap. The adaptive gate is non-interfering on the recorded MH rows
and avoids the fixed 10 m/s false rejection on MH_03. The 20 m/s fixed cap is
also non-interfering here, but remains a scene-scale safety ceiling rather than
a primary policy. The 1 m/s cap is an intentional tripwire.

| sequence | variant | velocity cap m/s | adaptive threshold m/s | cost-ratio cap | rejects | velocity rejects | adaptive rejects | mirrors | tracking | tracking delta | map KF | rigid ATE m | rigid delta m | sim ATE m | sim scale | verdict | run id |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| MH_03_medium | baseline | none |  | none | 0 | 0 | 0 | 1 | 0.015 | 0.000 | 2 | 0.0203 | 0.0000 | 0.0198 | 0.839159 | baseline | euroc-tight-vio-gate-MH_03_medium-baseline-1500-20260620 |
| MH_03_medium | adaptive_velocity | none | 36.216 | 1.00 | 0 | 0 | 0 | 1 | 0.015 | 0.000 | 2 | 0.0203 | 0.0000 | 0.0198 | 0.839159 | non-interfering | euroc-tight-vio-gate-MH_03_medium-adaptive_velocity-1500-20260620 |

Notes:

- Negative `rigid delta` means the gated run reduced rigid ATE versus the per-sequence baseline.
- `rejects` is the combined local VI-BA writeback quality gate counter; `velocity rejects` is the subset triggered by the velocity cap.
- `adaptive threshold` is the final per-trigger adaptive velocity threshold from the run summary when that gate was enabled.
- `mirrors` counts accepted local VI-BA velocity/bias writebacks mirrored into the IMU motion model.
- Missing cells mean that optional exploratory cap was not run for that sequence yet.
