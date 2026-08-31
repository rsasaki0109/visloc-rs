# Electro memory-bounded snapshot replay audit

The frozen 1,200-image Electro snapshot was replayed twice with the accepted
cap96, sparse-BA, four-solve schedule. `--snapshot-keypoints-only` changes only
feature ownership: it parses one file at a time, retains keypoints and row
shape, releases descriptor payloads, and re-reads one file at a time after
per-image calibration to reproduce the original descriptor-bound manifest
hash. The imported pair stream, mapper decisions, and output model are
unchanged.

| Result | Run 1 | Run 2 | Two-run result |
| --- | ---: | ---: | ---: |
| Registered | 1200/1200 | 1200/1200 | parity |
| Tracks / observations | 83,840 / 431,833 | 83,840 / 431,833 | identical |
| Mapper core | 225.656 s | 234.947 s | 230.302 s median |
| Snapshot-to-model wall | 324.61 s | 349.18 s | 336.90 s median |
| Peak RSS | 1,459,516 KiB | 1,458,872 KiB | **1,459,194 KiB (1.39 GiB)** |
| Camera-centre RMSE | 3.501 cm | 3.501 cm | identical |

The median peak is 63.6% below the prior descriptor-resident champion's
4,011,160 KiB and passes the predeclared 2 GiB M3 gate. It is 1.16x COLMAP's
1,255,996 KiB peak. The extra bounded descriptor-integrity pass makes the
conservative snapshot-to-model wall 25.5% slower than the 268.49 s speed-only
champion, but it remains 14.63x faster than the same-pair COLMAP mapper's
4,929.56 s.

Both runs produced the accepted model hashes exactly:

```text
cameras.txt  f9e722f47df961c0cf3ff7414e52f5bad2ebbc60a10a5f99d8c77e2cda880b17
images.txt   b7a4d3c9f160340232e29c0258e07045acfba358bdf8f48e97c88bda086d0099
points3D.txt e73ad63944034d43f24a94408d8cc3c0b11e1d95bfe025d0d6a32e8788929fbe
```

The mode is default-off and deliberately limited to file-backed features, an
imported verified-pair snapshot, and the plain incremental mapper. Feature or
snapshot export, coordinate override, row reordering, descriptor-based model
diagnostics, stable descriptor tie-breaking, and alternate mapper modes fail
closed. Source mutation between the two descriptor passes is also rejected.

Machine-readable timings, memory checkpoints, scores, hashes, command inputs,
and external artifact locations are recorded in
[`snapshot-memory-audit.json`](../benchmarks/electro/snapshot-memory-audit.json).
The README PNG and GIF were regenerated from the measured models and metrics;
two generations were byte-identical.
