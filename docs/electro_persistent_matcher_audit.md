# Electro persistent matcher audit

The M4 persistent worker loads the frozen 1,200-image descriptor bank once,
then processes 375 ordered 32-pair shards in one Rust process. Each shard is
published atomically and remains independently hash-valid for restart. The
final merge moves owned pair payloads instead of cloning the complete decoded
stream.

The accepted implementation also computes cross-check matching from one GEMM
and one column-major scan that updates both directional top-two states. Safe
RANSAC scoring exits only when the unvisited suffix cannot strictly beat the
incumbent. Neither optimization changes sampling, strict tie order, accepted
indices, pair order, or snapshot bytes.

| Frozen 12,000-pair result | Run 1 | Run 2 | Run 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Persistent worker wall | 439.01 s | 399.69 s | 442.62 s | **439.01 s** |
| Shard elapsed sum | 378.04 s | 344.97 s | 381.48 s | **378.04 s** |
| Worker peak RSS | 1,981,752 KiB | 1,981,352 KiB | 1,969,708 KiB | **1.89 GiB** |
| Owned merge wall | 3.29 s | 3.57 s | 3.62 s | **3.57 s** |
| Owned merge peak RSS | 1,632,432 KiB | 1,632,656 KiB | 1,632,484 KiB | **1.56 GiB** |

COLMAP 3.9.1 CPU matched the same frozen candidate manifest in 471.37 s.
The persistent visloc worker is therefore **1.074x faster** (6.9% less wall
time) and 4.75x faster than the former 2,085.89 s multi-process visloc path.
COLMAP's matching memory remains substantially lower (270,248 KiB versus
1.89 GiB).

## CPU8 end-to-end result

Eight fixed 150-image workers re-extracted the 1,200-image bank in 721.42 s.
All 1,200 feature files and 1,200 locus files match the frozen bank byte for
byte; their sorted SHA-256 list hashes to
`00d54f69ca7fd89a3e8ea016ce3d2c94eb6988e81b7d4048814373b0432cd9c6`.
The maximum single extractor-worker peak was 200,388 KiB. Regenerating the
candidate manifest from that bank also reproduced the frozen manifest SHA
`1b5cece...51186` exactly.

| End-to-end CPU8 phase | visloc-rs | COLMAP 3.9.1 CPU |
| --- | ---: | ---: |
| Feature extraction | 721.42 s | **304.12 s** |
| Manifest validation + candidate generation/sharding | 148.58 s | frozen manifest supplied |
| Exact-pair matching + merge | **442.58 s** | 471.37 s |
| Mapper / model writing | **336.90 s** | 4,929.56 s |
| Conservative total | **1,649.48 s** | 5,705.05 s |

The complete visloc pipeline is therefore **3.46x faster** while retaining
1,200/1,200 cameras, 3.50 cm centre RMSE, the accepted model hashes, and the
bounded-memory mapper. The accounting is conservative: visloc is charged
12.78 s of manifest validation and 135.80 s of candidate generation/sharding,
while the COLMAP exact-pair control consumes the already frozen identical
candidate manifest. Feature extraction alone remains 2.37x slower than
COLMAP and is the next optimization target.

All three runs produced the canonical merged snapshot SHA-256:

```text
55b2c1d9ec30df502e14d8f2de44b80042742428df4a62efbd903cd2850051f3
```

It contains 11,625 verified pairs and 7,475,384 accepted correspondences. A
single-shard old/new comparison also reproduced
`dd483410581c977ab86c3c60668d271cb6e5b80346c14c368eff95f713cb0367`.
The older full snapshot hash `93bf...a86` embedded an absolute candidate/output
path in its diagnostic effective config; pair-by-pair comparison was equal,
and the merge now canonicalizes both forms to the path-independent verifier
config.

The owned merge reduced its measured peak from 2,102,056 KiB to a 1,632,484
KiB median (22.3%) while preserving the output SHA. Worker and merge therefore
both pass the predeclared 2 GiB M4 ceiling. Failed 4- and 2-thread parallel
text-loader probes were rejected because they exceeded the ceiling; the
accepted worker retains the exact serial compatibility loader.

Machine-readable commands, hashes, timings, memory, negative-control outcome,
and external artifact roots are recorded in
[`persistent-matcher-audit.json`](../benchmarks/electro/persistent-matcher-audit.json).
