# Electro pure-visual block BA A/B

The frozen 1,200-image Electro quality champion was rerun without changing its
12,000 candidate pairs, verified snapshot, cap96 mapper policy, sparse solver,
or bounded post-registration pass. The implementation replaces the initial
dense camera Hessian with deterministic 6x6 pose blocks, stores only the lower
Schur triangle, transfers those blocks directly into block Cholesky, and reuses
symbolic analysis when the pattern is unchanged.

| Measure | M2 dense/scalar bridge | M3 direct blocks + cache | Change |
| --- | ---: | ---: | ---: |
| Median BA, nine 20-iteration solves | 153.764 s | **130.681 s** | **-15.0%** |
| Mapper core | 1425.985 s | **1310.111 s** | -8.1% |
| External mapper wall | 1490.07 s | **1366.75 s** | -8.3% |
| Peak RSS | 4,016,264 KiB | 4,016,460 KiB | flat |
| Registered cameras | 1200/1200 | 1200/1200 | identical |

All three exported model files are SHA-256 identical to the M2 champion. The
15.0% median BA reduction passes the roadmap's per-PR 10% gate. Peak RSS does
not improve: the 3.83 GiB high-water mark is dominated outside the logged
steady-state Schur buffers, so feature/snapshot lifetime and factor transients
remain explicit follow-up work rather than being hidden behind the speed win.

Machine-readable values, all nine timings, hashes, and the external artifact
root are in
[`ba-block-system.json`](../benchmarks/electro/ba-block-system.json).
