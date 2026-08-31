# Electro quality-gated BA schedule audit

The frozen 1,200-image Electro snapshot was replayed with the M3 direct-block
sparse solver. Candidate pairs, verified correspondences, cap96 mapper input,
seed policy, calibration, and bounded post-refinement registration were held
fixed. Only the maximum follow-up refinement rounds and the per-solve LM
iteration budget changed.

The previous schedule always ran six final global solves because its
filter/re-triangulation churn stayed between 5.9% and 8.9%, far above the 0.05%
convergence threshold. Those late rounds changed support only slightly. Setting
the follow-up cap to zero retains the initial solve after the nine missing
cameras register, while removing five repeated solves.

| Schedule | Global solves | Core | Wall | Registered | RMSE | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20 iterations, 5 follow-ups | 9 | 1310.11 s | 1366.75 s | 1200/1200 | **3.224 cm** | quality baseline |
| 20 iterations, 0 follow-ups | 4 | 575.98 s | 641.80 s | 1200/1200 | 3.394 cm | passes quality |
| 10 iterations, 0 follow-ups | 4 | 263.66 s | 320.73 s | 1200/1200 | 3.487 cm | passes quality/core |
| **8 iterations, 0 follow-ups** | **4** | **212.93 s median** | **268.49 s median** | **1200/1200** | **3.501 cm** | **accepted** |
| 5 iterations, 0 follow-ups | 3 | 102.41 s | 156.72 s | 1200/1200 | 6.788 cm | rejected |

The accepted 8-iteration configuration was repeated independently. Its three
COLMAP text files were byte-identical, core timings were 211.847 and 214.003 s,
and wall timings were 267.56 and 269.42 s. The 268.49 s wall median is 18.36x
faster than the same-pair COLMAP mapper's 4929.56 s while retaining lower
camera-centre RMSE (3.501 cm versus 4.679 cm). The median BA solve fell from
130.681 to 51.227 s.

Five iterations is deliberately not accepted despite its speed. All three
solves accepted zero LM steps, RMSE exceeded the predeclared 6 cm Competitive
limit, and one camera had 1.87 m error. This is the quality cliff that bounds
the chosen schedule.

At the time of this schedule audit, peak RSS remained effectively flat at
4,011,160 KiB median versus COLMAP's 1,255,996 KiB. The subsequent
[`snapshot-keypoints-only` audit](electro_snapshot_memory_audit.md) retained the
exact model while reducing the two-run median peak to 1,459,194 KiB, closing
the 2 GiB M3 memory target.

Commands, both run arrays, scores, hashes, the rejected control, and external
artifact roots are recorded in
[`ba-schedule-audit.json`](../benchmarks/electro/ba-schedule-audit.json).
The README PNG and GIF are reproducibly rendered from the accepted visloc
model, the COLMAP control, and the official reference by
[`generate_electro_readme_visuals.py`](../scripts/generate_electro_readme_visuals.py);
their dimensions, frame count, and hashes are stored in the same ledger.
