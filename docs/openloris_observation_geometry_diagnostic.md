# Frozen 1k observation-geometry diagnostic

Status: predeclared, read-only; no candidate optimization or GT-based selection.

## Why this diagnostic

The [frozen Ceres experiment](../benchmarks/electro/m8-openloris-ceres-reference-solve-v1.json)
reduced reprojection cost while worsening trajectory. This motivates checking
where cost reduction and state motion concentrate; it does not establish a
cause. This step changes no observations, weights, calibration or solver.

## Fixed metric and populations

Use the original frozen 1k model for all memberships. For each landmark X
and all its observing camera centres C_i (including anchor observations), set
u_i = (X-C_i)/||X-C_i|| and
A = max_{i<j} min(acos(clamp(u_i dot u_j, -1, 1)),
pi-acos(clamp(u_i dot u_j, -1, 1))).
Zero-length or nonfinite rays fail closed.

This follows the pair-angle convention and the “at least one sufficient
pair” test in [COLMAP 4.2.0 geometry](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/geometry/triangulation.cc)
and [triangulation estimator](https://github.com/colmap/colmap/blob/4.2.0/src/colmap/estimators/triangulation.cc).
It is a geometric descriptor, not a full BA condition number, nor proof of
the exact installed COLMAP development binary's implementation.
Using the minimum across pairs would misclassify tracks containing adjacent
views. Even the maximum depends on track length, so always stratify by length.

Fixed half-open angle bins in degrees: [0,0.1), [0.1,1), [1,5), [5,90].
Track-length strata: 2–3, 4–8, 9–16, >=17. Report all 16 cells, including empty
ones. No post-solve reclassification, threshold tuning, sampling or GT access.

## Comparisons and output

Read the initial model and exactly one existing candidate per invocation:
initial self-control, legacy matrix-free, adaptive scaled LM, or Ceres.
Use the already recorded PR #87 and #89 output hashes. No solver reruns.
Verify camera bytes, image IDs/order/names/camera assignment, every keypoint
and association, point IDs/RGB/order/track membership. Reject mismatches;
do not silently intersect point or observation sets.

For each cell report point/observation counts; initial and candidate full
squared reprojection costs; separately summed positive cost reductions and
cost increases at observation level; mean/max reprojection errors; sum/mean/max
landmark displacement; and observation-weighted mean/max camera-centre
displacement. Also report global image-weighted camera motion separately.
Use raw fixed-anchor coordinates, not a fitted alignment.
Empty means are null. Include global totals and verify bin sums reconstruct
them. Displacement associations are descriptive, not causal attribution.

## Resource and correctness gates

This is a bounded 1k diagnostic, not the scalable mapper implementation.
Maximum 1,024 images, 8,192 landmarks, 262,144 observations, 1,000,000 total
keypoints, 1,024 observations per track and 16 cameras. Before parser reuse, enforce <=32 MiB per file, <=64 MiB per model,
<=2 MiB per line, and the record/keypoint caps in a bounded preflight.
Require regular files, reject symlinks, and verify pre/post file hashes.
These model comparisons do not independently re-certify rig extrinsics:
use the already rig-audited inputs. Motion here means camera centre, not
inferred rig-body centre. Output JSON to stdout only. Read at most initial plus one
candidate model, never all arms together.

Preflight all tracks before pair traversal:
W2 = sum(k*k) <= 64,000,000; also record sum(k*(k-1)/2), sum(k), max(k).
Use all anchor observations. The expected frozen input counts are 130,900
observations, W2=28,705,634 and 14,287,367 pairs; independently verify these.
No global pair graph and no k-by-k allocation: pair scratch O(max track length),
aggregate state fixed to 16 cells. Exceeding a cap is an explicit error,
not a fallback to approximate angles. Record wall time and peak RSS separately
from solver performance; target <=512 MiB for this diagnostic.

Tests must cover known parallel/orthogonal/antiparallel angles, a long track
with a close pair but another wide pair, rigid-frame invariance, bin boundaries,
initial-only membership, positive-depth and identity failures, caps before pair
execution, cost-increase/decrease separation and deterministic output.
Real output must reconcile with the existing independent cost audit and repeat
exactly excluding wall/RSS metadata. No model files are written.

## CLI and implementation checkpoint

Implementation `5300634` uses the existing strict model parser after bounded
preflight, and the independent model auditor's normalized rotation convention
for both projection and camera centres. Inputs must remain stable regular files:
pre/post hashes detect observed mutation, but this is not a race-safe snapshot
reader. Do not run it against models being written concurrently.

```bash
python3 scripts/audit_ba_geometry_bins.py \
  --initial-model /path/to/frozen-1k-initial/model \
  --candidate-model /path/to/already-audited-1k-candidate/model \
  --candidate-label legacy
```

JSON is emitted on stdout only. The CLI additionally requires the frozen
130,900 observations, W2=28,705,634 and 14,287,367 pairs in both inputs;
the Python API permits small synthetic fixtures for tests. Counts alone are
not provenance: compare output file hashes with the
[frozen input certificate](../benchmarks/electro/m8-openloris-observation-geometry-preflight-v1.json).
Runtime metadata is excluded only when checking deterministic repeats.

Root re-ran 11 diagnostic tests, six publisher tests and 14 independent model
auditor tests after Luna Max implementation. Initial review caught and fixed
the PINHOLE row-layout check, invalid synthetic track references and a
tautological bin-total check before any real diagnostic run.

## Result (2026-09-08): large point motion is localized, cost reduction is not

The [five-run evidence](../benchmarks/electro/m8-openloris-observation-geometry-v1.json)
records initial self-control, legacy, adaptive, Ceres and a Ceres diagnostic
repeat. All exit successfully in 19.53–21.29 s with peak RSS
220,860–221,040 KiB (about 216 MiB); this is diagnostic-only process cost,
not optimizer or mapper timing. The repeat report is exact after removing
runtime metadata. All 12 input model files remain hash-identical.

Root independently recomputed every track's bin using a minimum-absolute-dot
calculation and camera centres from a linear solve. All 16 populations match:
4,716 points, 130,900 observations, 1,000 images, 361,170 keypoints. The initial
self-comparison has exactly zero cost change and motion. Candidate costs
match earlier evidence within 7.3e-10; the serialized initial model differs
from the original rig fixture by 7.44e-7, while its mean error agrees.

The fixed angle-below-0.1-degree group has 39 points (0.83%) and 949
observations (0.72%). These are descriptive sums of displacement, not a sum
of trajectory errors or an influence measure:

| Existing arm | All-point displacement sum (m) | Below-0.1° displacement sum (m) | Share of displacement | Share of net cost reduction |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 3.148973 | 0.032029 | 1.02% | -0.26% |
| Adaptive | 886.053895 | 636.707437 | 71.86% | 6.28% |
| Ceres | 1020.052384 | 749.779231 | 73.50% | 6.54% |

The legacy group's negative cost-reduction share means its cost increased;
the evidence keeps increases and decreases separate. Conversely, tracks of
length at least 17 contain 83.80% of observations and account for 88.06% /
87.77% of adaptive / Ceres net cost reduction. Their large absolute contribution
must be interpreted alongside this large observation share.

Thus weak-angle tracks explain much of the landmark displacement, but not
most of the objective reduction. This does not establish that freezing,
dropping or reweighting those points would repair trajectory accuracy. Do
not promote a new solver, discard support or change the predeclared bins.
Any intervention needs a separately fixed hypothesis and support/quality
non-regression test; GT must remain post-only. Native E2E and the full 10k
quality goal remain unmet.
