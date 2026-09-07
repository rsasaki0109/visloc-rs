# OpenLORIS rig-atlas diagnostic

Status: M8 remains incomplete. These outputs contain trajectories only, with
empty observation rows. They do not establish merged-model reprojection quality,
mapper-only performance, or native end-to-end performance.

## Frozen experiment inputs

The September 4 overlap experiments use
`/tmp/visloc-m8-hierarchical-500x250/nodes-shifted-bridges.tsv`
(SHA-256 `4dfb8283d0cd9693768555636e433a4933b153b5bbcd6d32ecc428cc6a705366`).
They publish 9,996 images in two components, containing 4,493 and 505 rig
frames. Frames 4,493 and 4,494 are absent. The COLMAP target requires 9,998
images, a largest component of at least 4,494 frames, RMSE at most 0.384307 m,
p95 at most 0.638669 m, and mean reprojection at most 0.903003 px.

Ground truth is loaded only by the post-map scorer. Window selection and seam
estimation use reconstructed poses and shared frame identities. Nevertheless,
the successive arms are development experiments evaluated repeatedly against
this sequence; they are not independent held-out validation.

## Retained September 4 results

| Diagnostic arm | Transform graph | ATE RMSE (m) | ATE p95 (m) |
| --- | --- | ---: | ---: |
| H, earliest overlap sample | traversal | 0.540386 | 1.119452 |
| H, earliest overlap sample | quality forest | 0.650178 | 1.128827 |
| I, uniform overlap sample | quality forest | 0.717315 | 1.613746 |
| J, relaxed mean-residual gate | traversal | 0.486770 | 0.933776 |
| K, unit-scale fixed-rotation estimator with generic fallback | traversal | 0.478419 | 0.894231 |

All arms fail registration and trajectory acceptance. Quality-forest ordering
and uniform sampling were not promoted. H traversal was reproduced after
restoring the explicit traversal policy. K still accepts a generic fallback
with scale 0.775097 on the node 3 / node 21 seam, so it does not enforce unit
scale throughout the atlas.

These numbers describe the old per-camera publication rule. On September 7,
an independent calibration audit found that this rule scales the stereo
baseline when applying a non-unit Sim(3) to the two sensor centres separately:

| Artifact, dominant component | Minimum baseline (m) | Median baseline (m) | Maximum baseline (m) |
| --- | ---: | ---: | ---: |
| Input window 0 | 0.063977924049 | 0.063977924051 | 0.063977924053 |
| H traversal output | 0.044581955118 | 0.046855061714 | 0.063977924052 |
| K traversal output | 0.049589122861 | 0.049589122893 | 0.063977924059 |

The audit pairs sensor names through the rig manifest's `F` rows and computes
each camera centre as `-R.transpose() * t`. Numeric image suffixes differ
between the two sensors and must not be used as the pairing key.

The required publication rule is to transform one rig pose into the atlas,
then compose each unchanged, manifest-bound `sensor_from_rig` transform with
that rig pose. This preserves the supplied physical calibration. It changes
sensor centres relative to a naive Sim(3) of the complete local model, so any
future landmark integration must re-evaluate observations and triangulation;
it cannot inherit the local model's reprojection score.

## Calibration-preserving publication verified (September 7)

The exporter now binds each image to the manifest's fixed `sensor_from_rig`
and composes it with the transformed rig pose. A pose-composition unit test
covers scales 0.7 and 1.3, nontrivial rig motion, and distinct sensor rotations.
The example suite passes all 24 tests; formatting and example clippy with
warnings denied also pass.

The same H and K inputs were replayed twice into
`/tmp/visloc-m8-rig-calibration-20260907-xW3PlM`. Both component files were
byte-identical across repetitions for each arm. An independent numeric audit
checked all 4,998 sensor pairs against the manifest: maximum relative-rotation
matrix error was below `5e-15` (Frobenius norm), relative-translation error
below `1.7e-13 m`, and every baseline was within `1.3e-13 m` of
`0.063977924051 m`.

| Corrected arm | Images | Components | ATE RMSE (m) | ATE p95 (m) |
| --- | ---: | ---: | ---: | ---: |
| H traversal | 9,996 | 2 | 0.541224502 | 1.123657290 |
| K traversal | 9,996 | 2 | 0.478844237 | 0.897778825 |

The calibration defect is fixed, but trajectory and registration still fail.
These scores supersede the old per-camera publication scores for future
comparisons. The correction does not justify another threshold relaxation.

Repeat-verified `images.txt` SHA-256 values:

| Arm | Component | SHA-256 |
| --- | --- | --- |
| H | 000 | `f0635f72f7b120992afea84beaf50a022dc1e1d23bc56c9f951a2edff73b1967` |
| H | 001 | `c830546d1549080aeb6536155b4aba8f54c16524b0c750c5eab628d05cd7f649` |
| K | 000 | `5a6ec16c01ef94cb79ff2be23737b14435cdcfab53650b45d1e763ef29229322` |
| K | 001 | `44005eeee990cb8b1fb10f9e253d64d789e9f66b0e99502dd4e530f579f77ca0` |

## Further registration and seam experiments

### Boundary correspondence audit (September 7)

Decoding the frozen targeted7 base snapshot validates 10,000 images, 60,347
verified pairs, and 3,352,705 accepted correspondences. Using its unchanged
59,961-pair structure prefix and the rig manifest's image assignments gives:

| Target frame / sensor | Stage | Other endpoint frame / sensor | Accepted correspondences |
| --- | --- | --- | ---: |
| 4493 / 1 | structure | 4491 / 1 | 18 |
| 4493 / 1 | structure | 4492 / 1 | 21 |
| 4493 / 1 | deferred | 4488 / 0 | 17 |
| 4493 / 1 | deferred | 4489 / 0 | 17 |
| 4493 / 1 | deferred | 4490 / 0 | 25 |
| 4493 / 1 | deferred | 4490 / 1 | 15 |
| 4494 / 0 | structure | 4495 / 0 | 15 |
| 4494 / 0 | deferred | 4497 / 1 | 15 |

These are all incident pairs in the base snapshot, before the separate dense
overlay. Every target observation for frame 4493 is on sensor 1, so this input
alone cannot satisfy the normal two-sensor PnP gate. The configured deferred
one-sensor path is necessary to test next. Pair counts do not establish usable
3-D support or successful PnP; source-track triangulation and per-target
registration diagnostics must be inspected before assigning the failure cause.

The historical strong-structure registration experiment used a direct-stereo
PnP fallback at frame gap 2 with one sensor allowed. The current local-window
command disables that fallback but enables deferred one-sensor registration.
This is a measured configuration difference, not yet proof that enabling the
old fallback will recover the frame in the current snapshot and local map.

The dense overlay source has **zero** incident verified pairs for either
4493 or 4494, before temporal-gap filtering. Dense feature volume therefore
does not add registration support for these two frames in this frozen input.

### Boundary registration recovered with a shifted window

Replaying the saved window command with start 4200 and count 500, without
changing thresholds, recovers 4493 through deferred PnP. The local model has
499/500 frames, two components, and 0.558604912 px weighted reprojection.
Mapper time is 22.431005 s, process wall time 24.82 s, and peak RSS
340,056 KiB. Inputs, debug logs, model files, and the timed command are at
`/tmp/visloc-m8-boundary-window-4200-debug.AqO1ao`.

The remaining frame 4494 has zero usable deferred correspondences in the
first component and two in the second (six required). Direct and motion
bridge visits remain zero. This local recovery does not require reopening
either bridge mechanism or relaxing a PnP gate.

Replacing window node 18 with the new node 27 recovers registration but
regresses atlas RMSE/p95 to 0.527112/1.097068 m. Keeping node 18 and adding
node 27 instead preserves its existing overlap coverage and yields:

| K traversal with added boundary node | Result | COLMAP gate |
| --- | ---: | ---: |
| Registered images | 9,998 | 9,998 |
| Largest component, rig frames | 4,494 | 4,494 |
| Components | 2 | at most 2 |
| ATE RMSE | 0.481208093 m | at most 0.384307 m |
| ATE p95 | 0.905394066 m | at most 0.638669 m |

Output root: `/tmp/visloc-m8-boundary-added-atlas-20260907-vVLgiV`.
Node manifest: `/tmp/visloc-m8-boundary-added-nodes-20260907.tsv`.
Component 000 `images.txt` SHA-256:
`e131fa0bde1684be822a6d10a4c23f4583534ff43e3b7ff4357d08121c27b74d`.
Component 001 is unchanged from corrected K
(`44005eeee990cb8b1fb10f9e253d64d789e9f66b0e99502dd4e530f579f77ca0`).
Registration gates pass for this trajectory diagnostic; accuracy and merged
landmark reprojection remain unproven. These variants are single decision
runs, not the required final three-run performance comparison.

### Additional overlap around the non-unit seam

An accepted-edge connectivity audit identifies node 3 / node 21 as a graph
bridge before adding any new window. Its K fixed-rotation unit-scale attempt
rejects with 22/64 inliers; the generic fallback accepts scale 0.775097405.
There is no alternative accepted route across that cut, so cycle-based
optimization alone has no independent constraint to correct this edge.

Replaying frames 650--1149 with the unchanged 500-frame mapper configuration
produces 500/500 registered rig frames in one model, 0.674879864 px mean
reprojection, 27.669512 s mapper time, 30.07 s process wall time, and
340,020 KiB peak RSS. All frames register during normal PnP. Output:
`/tmp/visloc-m8-window-650.hSRTGo/model/component-000`.

Adding it as node 28 creates accepted unit-scale connections to nodes 1, 21,
3, and 4. The original scale-0.775 edge is consequently no longer the only
route. With all earlier nodes retained, K traversal improves to 0.397456745 m
RMSE and 0.642896173 m p95 at 9,998 images / two components. Both metrics still
miss COLMAP. K quality-forest on these same inputs is worse at
0.451738027 / 0.714085127 m; the corresponding pre-addition quality control
was 0.536545315 / 0.933283174 m. These paired controls distinguish new overlap
support from changing the graph policy.

The K traversal output was repeated byte-for-byte for both components:
`/tmp/visloc-m8-seam650-atlas-mtHTsS/traversal` and
`/tmp/visloc-m8-seam650-repeat-L6XfF4/model`. Its dominant-component
`images.txt` SHA-256 is
`f61c9f2eaa0deaf5a1ca7d423dd2e3a1d82a811dd9d31bb3bee07fa5cfacbb8e`;
the tail component remains unchanged. The node manifest is
`/tmp/visloc-m8-seam650-nodes-20260907.tsv`, SHA-256
`564f0ac7f31fcac0e3995a65cfe3fe7711ebdef100b30f28f8970c81c1cea0ce`.

## Remaining acceptance checks

The strict metric L arm uses K's exact thresholds and earliest sampling, but
disables generic scale-changing fallback. On the new overlap graph it still
publishes 9,998 images / two components and scores 0.391778248 m RMSE /
0.643697863 m p95. It rejects the old non-unit edge while preserving the new
unit-scale route. Both trajectory gates remain unmet; the modest RMSE gain
does not offset the slightly worse p95 versus K.

Output: `/tmp/visloc-m8-seam650-strict-metric-8a9UaT`.
Dominant `images.txt` SHA-256:
`a26a3a4444dd73be2e707d5c89c3ebce845fd92a34eb8c18a0d2e6833e489ed3`.
The parser/config tests cover L's sole configuration difference; the example
suite passes 28 tests after the owner-policy experiment below; example clippy
passes with warnings denied.

### Overlap frame ownership control

With the same 23-node manifest, L seams, and traversal transforms, the
`--frame-owner-policy interior` diagnostic chooses the healthy window with
the greatest distance from its first/last registered frame. Only windows
actually containing the frame are eligible; ties use newest start and node
ID. This uses no ground truth. The default remains `newest`.

| Owner policy | Registered images | ATE RMSE (m) | ATE p95 (m) |
| --- | ---: | ---: | ---: |
| newest, repeat control | 9,998 | 0.391778248 | 0.643697863 |
| interior, rejected | 9,998 | 0.431936060 | 0.712670307 |

Both retain two components. Interior worsens both target metrics and is not
promoted. Newest reproduces both strict-metric component hashes exactly.
Outputs and post-map scores: `/tmp/visloc-m8-owner-ab-nzxeyZ/{newest,interior}`.
Exact metrics, hashes, gates, and test results are retained in the repository's
[owner A/B evidence](../benchmarks/electro/m8-openloris-atlas-owner-ab.json).
Interior component 000 SHA-256:
`5b25803b144fb16df686a31d490cb1138ba0a4ed5ad49550337a5406fbb51c59`;
component 001:
`4ace6d25135c0d4baa401b9a56f3f2771b44875548de49f0f0d572c2cd85d569`.
The next control below tests whether a scale-fixed sparse pose graph can
use the newly available redundant seams; no trajectory threshold is relaxed.

### Equal-weight metric SE3 cycle optimization (rejected)

The opt-in `--metric-se3` diagnostic uses all accepted L seams, one anchored
SE3 graph per component, unit edge weights, no robust kernel, sparse Cholesky,
LM initial damping 0.001, and no chordal initialization. Stored node poses are
`node_from_atlas`; measurements retain `target_from_source`. Scale is fixed
at one. The default remains unchanged, and the quality-forest combination is
rejected because it would remove the redundant cycle constraints being tested.

On the same frozen 23-node input, the dominant component's objective drops
from 1.118173145605 to 0.260483497520 in nine iterations (converged). However,
ATE RMSE/p95 worsen from 0.391778248/0.643697863 m to
0.431011205/0.717679163 m. Both arms retain 9,998 images / two components.
Thus reducing internal seam inconsistency does not establish improved scene
accuracy; this arm is not promoted and no score-informed weight sweep follows.

Output root: `/tmp/visloc-m8-metric-se3-ab-SWSGIA`, with separate `control`
and `metric` directories, post-map `score.json`, and run/time logs. Control
reproduces both previous L hashes. Metric component 000 SHA-256:
`b462a17319866b9212c721ecbe4df8b148f4f1eb20919ceb15cc4d99ce63deb9`.
Both metric components repeat byte-for-byte in `metric-repeat`. Exact metrics
and the decision are retained in the
[metric SE3 A/B evidence](../benchmarks/electro/m8-openloris-atlas-metric-se3-ab.json).
The two-node tail is byte-identical to L; its numerically zero-cost solve
reports `converged=false` after 15 iterations, so convergence is not claimed
for the entire atlas. The time logs describe concurrent trajectory-only
diagnostics, not a comparative mapper performance measurement.

The next required work is real landmark integration and observation-based
validation/refinement under the final calibrated rig poses. Further improvements
must be supported by image geometry, not merely a lower pose-graph objective.

Preserve the old outputs as diagnostic evidence. M8 still needs the
trajectory gates, registration in a real merged landmark model, and its
reprojection evaluation before performance and release closure.

### Real-model integration requirements (implementation audit)

The existing `generalized_rig_sfm` exporter retains each image's full
keypoint ordering, but image IDs are local to each exported window. A future
landmark integration must therefore join observations by manifest-bound image
name/global ID plus original keypoint index, never by a window-local image or
point ID. Confirm matching pixel coordinates when joining reused indices;
different feature-bank generations must fail closed rather than merge silently.

A read-only audit of all 23 frozen source windows finds 21,566 image rows,
9,998 unique image names, and 8,482 shared names (at most three occurrences).
SHA-256 signatures of each complete ordered float64 `(x, y)` array agree for
every repeated image: zero coordinate/ordering mismatches. The sources contain
3,364,422 landmark observation references including overlap duplicates.
Original window `images.txt` observation rows and sibling `cameras.txt` /
`points3D.txt` exist; only the stitched trajectory output lacks landmarks.

Naively unioning local tracks through shared observations is not safe: a
read-only transitive-union diagnostic finds 803,676 source tracks, 1,750,273
unique observations, and 380,669 unions, of which 833 contain multiple
keypoints from the same image (10,611 excess same-image references). The
largest union has 1,264 observations. This is a rejected diagnostic, not an
accepted merge policy. Integration must detect these conflicts before mutating
ownership and report rejected support, rather than silently combining them.

All source windows also pass an independent bidirectional-track/projection
audit: 803,676 source points, 3,364,422 observations, two consistent camera
definitions across windows, no non-positive-depth observations, mean
0.6777286687176332 px and maximum 3.9999762218186277 px reprojection error.
These are source-window statistics with overlap duplicates, not integrated
model quality. Exact audit results and provenance are retained in
[source audit evidence](../benchmarks/electro/m8-openloris-atlas-source-audit.json).

For continued integration, the 69 source model files and L newest trajectory
are copied under
`/home/sasaki/datasets/openloris/corridor1-1-m8-atlas-landmarks-v1`.
Every source copy is byte-identical; `nodes.tsv` changes paths only. Replaying
the trajectory stitcher with these relocated inputs reproduces both L output
hashes exactly. This removes the source models' dependency on `/tmp`, but is
not the complete workflow's restart/stress acceptance test.

`RigSfmResult` contains poses and tracks, but the public fixed-rotation
refinement entry point adjusts translations and landmarks of an existing
result; it is not a bounded, fixed-pose atlas triangulation/import API. Calling
it does not by itself establish correct integrated geometry or the memory gate.

The existing COLMAP text writer emits zero in the points3D `ERROR` field.
Consequently, a merged model's quality must be recomputed from actual camera
projection and observation pixels, not read from those placeholder errors.
COLMAP's [upstream `UpdatePoint3DErrors` implementation](https://github.com/colmap/colmap/blob/main/src/colmap/scene/reconstruction.cc)
stores the arithmetic mean of Euclidean reprojection errors per point, not
the RMS. Its [format documentation](https://colmap.github.io/format.html#points3d-txt)
also notes that stored errors are updated after global BA. The independent
frozen-control audit gives 0.9030031049631768 px directly versus
0.9030031518617252 px from stored per-point values; the historical gate is
unchanged, and both comparisons should be reported for integrated output.
An independent read-only projection audit of window 650 confirms this:
40,754 points / 162,892 observations have bidirectionally consistent tracks,
zero non-positive-depth observations, and mean error
0.6748798644580134 px (p95 2.0428205985713275 px, maximum
3.9951008058508584 px), matching the mapper's reported mean even though all
exported `ERROR` fields are zero. This validates that individual window only,
not the atlas or a merged 10k model.
Keep physical rig extrinsics fixed, triangulate/refine using the final poses,
validate positive depth and observation/track referential integrity, and report
observation-weighted reprojection together with retained track/observation
counts. Low error obtained only by discarding support is not sufficient.

Plan the integration as streamed window ingestion with a sparse observation
index and bounded local refinement. Do not load all descriptor banks or make
an image-by-image matrix. Measure process/cgroup peak RSS for the complete
mapper workflow, including window overlap work, integration, and export; the
small trajectory-publication process is not a mapper performance substitute.
