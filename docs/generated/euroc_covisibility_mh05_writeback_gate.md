# EuRoC Covisibility-BA Write-Back Gate Verification (Verified Negative)

Generated from benchmark-registry run manifests. PR #37 added two opt-in write-back conditioning gates on covisibility local BA: `--covisibility-local-ba-max-behind-camera-ratio 0.3` and `--covisibility-local-ba-min-fixed-to-optimized-ratio 0.34`. This `--max-frames 400` A/B/C across MH_01/MH_03/MH_05 checks whether those gates let covisibility local BA beat the disabled baseline on all three sequences at once. It does not: MH_05 still regresses even with the strictest useful gate setting, so covisibility local BA remains an explicit opt-in feature, not a default-safe one.

| sequence | config | tracking | rigid ATE m | sim ATE m | map keyframes | BA triggers | BA success | BA fail | behind-cam reject | fixed-ratio reject | run id |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MH_01_easy | disabled | 0.380 | 0.0642 | 0.0617 | 27 | 0 | 0 | 0 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_disabled-MH_01_easy-20260703T000000Z |
| MH_01_easy | enabled, no gate | 0.585 | 0.0607 | 0.0593 | 43 | 41 | 28 | 13 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_enabled_nogate-MH_01_easy-20260703T000000Z |
| MH_01_easy | enabled + gate | 0.705 | 0.0540 | 0.0521 | 52 | 50 | 20 | 30 | 0 | 12 | euroc-covisibility-local-ba-writeback_gate_enabled_gate-MH_01_easy-20260703T000000Z |
| MH_03_medium | disabled | 0.865 | 0.0648 | 0.0629 | 24 | 0 | 0 | 0 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_disabled-MH_03_medium-20260703T000000Z |
| MH_03_medium | enabled, no gate | 0.973 | 0.0394 | 0.0386 | 28 | 26 | 24 | 2 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_enabled_nogate-MH_03_medium-20260703T000000Z |
| MH_03_medium | enabled + gate | 0.882 | 0.0577 | 0.0577 | 28 | 26 | 13 | 13 | 0 | 10 | euroc-covisibility-local-ba-writeback_gate_enabled_gate-MH_03_medium-20260703T000000Z |
| MH_05_difficult | disabled | 0.565 | 0.1139 | 0.1118 | 42 | 0 | 0 | 0 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_disabled-MH_05_difficult-20260703T000000Z |
| MH_05_difficult | enabled, no gate | 0.220 | 0.1683 | 0.0888 | 19 | 17 | 6 | 11 | 0 | 0 | euroc-covisibility-local-ba-writeback_gate_enabled_nogate-MH_05_difficult-20260703T000000Z |
| MH_05_difficult | enabled + gate | 0.258 | 0.1024 | 0.1014 | 16 | 14 | 7 | 7 | 0 | 5 | euroc-covisibility-local-ba-writeback_gate_enabled_gate-MH_05_difficult-20260703T000000Z |

## Headline

- MH_01: disabled `0.380` / enabled-no-gate `0.585` / enabled+gate `0.705` (WIN).
- MH_03: disabled `0.865` / enabled-no-gate `0.973` / enabled+gate `0.882` (marginal win).
- MH_05: disabled `0.565` / enabled-no-gate `0.220` / enabled+gate `0.258` (FAIL, far below the `0.565` disabled baseline).

The disabled arm reproduces prior `euroc-covisibility-local-ba` disabled history for each sequence exactly, which validates this A/B setup.

## Caveats

- **The behind-camera gate never fires.** At `max_behind_camera_ratio=0.3` it rejects zero triggers on every sequence in this sweep. The MH_05-corrupting solves keep low post-BA reprojection error (roughly `0.2` to `0.5` px), so the collapse is global drift from locally-consistent solves, not behind-camera degeneracy that this gate can detect. Only the fixed-ratio gate ever rejects anything here.
- **MH_05 only matches disabled at a no-op gate setting.** MH_05 reaches `0.565` (matching disabled) only when the fixed-ratio gate is strict enough that 100% of solves are rejected (`fixed_ratio=2.0` is a true no-op point). `fixed_ratio=1.0` gets MH_05 to `0.448`, but the same setting drops MH_03 to about `0.860` -- below its own `0.865` disabled baseline -- wiping out the MH_03 win.
- **Run-to-run nondeterminism was observed.** The same MH_01 + `fixed_ratio=1.0` configuration produced `0.458` in one run and `0.642` in another. Gated numbers in this table (and in this sweep generally) carry noise; the disabled and enabled-no-gate arms above are the reproducible anchor, and gated numbers should be read as single-run, not as a stable measurement.

## Conclusion

Covisibility local BA stays an honest opt-in feature. The write-back gates added in PR #37 cannot make it safe to turn on by default: they detect a different failure mode (behind-camera degeneracy) than the one that actually drives the MH_05 regression (locally-consistent but globally-drifting solves), and the one gate that does reject anything on MH_05 only helps at settings that also erase the MH_03 win. The MH_05 regression stays visible and documented here, not swept away.
