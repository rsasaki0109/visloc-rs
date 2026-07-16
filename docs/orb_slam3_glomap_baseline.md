# ORB-SLAM3 / GLOMAP official baseline protocol

This protocol defines the external baselines for the online SLAM and offline
global-SfM tracks. It is an evidence protocol, not a paper-number comparison.

## Competitors

- Online: the official ORB-SLAM3 `stereo_inertial_euroc` executable, built from
  a recorded official revision.
- Offline: COLMAP `global_mapper`. The standalone GLOMAP repository was archived
  in March 2026 after its implementation moved into COLMAP; the source revision
  must therefore identify the tested COLMAP commit.

Two learned systems are additional tracking references, not substitutes for the
visual-inertial/multi-map acceptance gate:

- DPVO / DPV-SLAM supplies a lightweight learned patch tracker, differentiable
  patch BA and an optional loop backend. Its official EuRoC protocol uses five
  trials. The source is MIT-licensed. Research snapshot (2026-07-15):
  `859bbbfdac6c6185f345003b3c473901fcd13ace`.
- DROID-SLAM supplies the dense recurrent update operator, dense BA and
  keyframe/factor-graph policy. It supports EuRoC monocular and stereo runs but
  requires a CUDA GPU with roughly 11 GiB for inference. The source is
  BSD-3-Clause. Research snapshot (2026-07-15):
  `2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99`.

Both can therefore be ported or wrapped with attribution. Initial integration
should expose their relative-pose/covariance result as an A/B motion prior while
visloc-rs retains metric stereo depth, IMU preintegration, relocalization and
multi-map ownership.

### Learned-frontend transfer experiments

The learned systems are decomposed into independently switchable experiments;
they are not accepted as an opaque end-to-end replacement:

| Experiment | Reference idea | visloc-rs invariant | Primary cliff metric |
| --- | --- | --- | --- |
| learned correlation | DPVO patch correlation or DROID dense correlation pyramid | calibrated stereo geometry remains authoritative | inlier survival under blur/rotation |
| iterative correspondence | recurrent residual/confidence update | pose and depth are updated through geometric residuals | consecutive lost frames |
| weighted BA | DROID confidence-weighted dense BA, initially sampled sparsely | IMU preintegration and fixed left/right extrinsics remain factors | recovery rate and ATE/RPE |
| dynamic frame graph | optical-flow/covisibility edges and long-range edges | submap ownership and map-merge verification stay explicit | relocalization latency |
| asynchronous backend | local frontend plus history-wide optimization | realtime deadline is measured on the frontend path | p50/p95 frame time |

DROID-SLAM's inference recipe optimizes per-frame pose and inverse depth,
predicts correspondence revisions and confidence, and maps those revisions to
Gauss-Newton updates through a Schur-complement dense-BA layer. For stereo it
adds left/right graph edges while fixing their calibrated relative pose. The
first implementation target here is therefore a bounded, confidence-weighted
correspondence factor inside the existing Stereo-VI window—not a dense CUDA
clone. Ablations must compare classical-only, learned-correlation-only,
learned-weighting-only, and combined modes on identical frame ranges.

The first backend slice is implemented in `BundleAdjustment` as
`optimize_with_observation_weights`. It accepts a flattened mono / rectified-
stereo / general-stereo confidence vector, multiplies it with the selected
robust-kernel IRLS weight in both the Schur normal equations and LM acceptance
cost, and leaves IMU and structural factors untouched. Zero-confidence
observations are fully muted; invalid lengths, negative values and non-finite
values are rejected. Joint intrinsics refinement is rejected explicitly until
its separate normal-equation path supports the same weighting contract. The
current regression ablation confirms that a zero-confidence 50-pixel false
correspondence does not move an otherwise exact pose.

Frontend confidence now survives PnP inlier selection, accepted-keyframe map
insertion and both covisibility and tight Stereo-VI local-BA builders. Missing
classical confidence remains uniform weight `1`; mixed learned/classical sets no
longer accidentally mute classical correspondences. Because mutual-softmax
probability is not a calibrated inverse variance learned jointly with the
optimizer, explicit confidences are divided by their window mean before BA.
This preserves the visual block's average information scale against physically
whitened IMU factors while retaining relative suppression/emphasis. Weighted visual cost
breakdowns use the same vector as optimization, while IMU NIS and structural
terms stay unweighted. The EuRoC image demo records
`map_observation_confidences`, confidence min/mean/max, and
`observation_confidence_ba_enabled` in its summary. The uniform-weight path is
the default; add `--observation-confidence-ba` for the experimental weighted
arm of an otherwise identical run. The next experiment is therefore a measured
soft-weighting A/B on tracking-cliff frame ranges, rather than more plumbing.

#### Initial confidence-weighting smoke (not an acceptance result)

On 2026-07-16, the GT-free MH01 view was run for the first 100 cam0 frames
with mutual-softmax matching, strict stereo bootstrap/replenishment,
projection-guided tracking and calibrated-stereo covisibility BA. Both arms
used identical arguments; the weighted arm alone added
`--observation-confidence-ba`. Both engines finished before the external
evaluator opened the original EuRoC GT.

| Metric | relative confidence | uniform | Direction |
| --- | ---: | ---: | --- |
| tracked frames | 100/100 | 100/100 | tie |
| engine wall time (s) | 11.260 | 9.614 | uniform |
| local-BA success | 45/46 | 45/46 | tie |
| live SE3 ATE RMSE (78 associated poses, m) | 0.2494 | 0.2262 | uniform |
| live consecutive translation RPE RMSE (m) | 0.0618 | 0.0857 | confidence |
| live consecutive rotation RPE RMSE (deg) | 1.346 | 1.117 | uniform |
| final-KF SE3 ATE RMSE (37 poses, m) | 0.2814 | 0.2773 | near tie |
| final-KF Sim3 ATE RMSE (m) | 0.1426 | 0.1455 | confidence |

Stored matcher confidence spanned `0.279..1.0` with mean about `0.977`.
Relative weighting increased surviving map landmarks (1,266 vs 1,086) and
improved translational continuity, but did not improve the aggregate live ATE
or rotation RPE. It therefore remains opt-in. These are one-run, short-prefix
diagnostics—not an ORB-SLAM3 comparison and not evidence of a tracking-cliff
win. Artifacts and post-run evaluations are under
`E:/visloc_archive/confidence_ba_ab_100_20260716_v2`. This rerun used the
current release binary, whose default is uniform and whose weighted arm alone
receives the opt-in flag above.

Extending the identical GT-free run to 300 frames reproduced the actual
tracking-cliff pathology rather than fixing it. Both arms reported 300/300
tracking, but metric scale collapsed:

| Metric (278 associated live poses) | relative confidence | uniform |
| --- | ---: | ---: |
| SE3 ATE RMSE (m) | 0.6935 | 0.5041 |
| Sim3 ATE RMSE (m) | 0.1770 | 0.1781 |
| Sim3 scale | 0.0615 | 0.0775 |
| translation RPE RMSE (m) | 0.1021 | 0.0902 |
| rotation RPE RMSE (deg) | 1.556 | 1.370 |

Thus frontend survival is not metric trajectory survival: mutual-softmax keeps
the sparse tracker alive while both estimates shrink by roughly 13--16x, and
the current confidence weighting makes the metric result worse. Artifacts are
under `E:/visloc_archive/confidence_ba_ab_300_20260716`. This closes raw matcher
confidence as a sufficient DROID transfer. The next DROID-derived experiment
must jointly revise correspondence targets and pose/depth over repeated
geometric updates, with a stereo/IMU metric-consistency gate; merely increasing
the confidence exponent would optimize the same wrong correspondences harder.

#### Sparse iterative-correspondence and metric-gate experiment

The next DROID-derived slice rebuilds a fresh one-to-one projected
correspondence set around the latest pose, permits bounded query-to-landmark
reassignment, and alternates this with PnP for up to three rounds (8, 4 and 2
pixel windows). A conservative mode also fixes an existing union bug where one
query keypoint could enter PnP against multiple landmarks; matcher confidence
is retained on carried inliers. Unchanged assignments now converge as a no-op
instead of sampling PnP again. Revised rounds are monotonic in inlier count and
reprojection error and expose optional inlier-pair-retention and pose-correction
trust regions. All features are opt-in; the legacy one-round behavior remains
the default.

On the same 300-frame GT-free MH01 input, local-map iterative matching without
a total motion gate still accepted a scale-wrong trajectory. An 80% pair
retention gate plus 5 cm / 2 degree per-round trust region was also insufficient:
small biased updates accumulated. Adding the existing total pose-prior motion
gate produced the following post-run evaluation:

| GT-free engine configuration | tracking | SE3 ATE (m) | Sim3 scale | trans. RPE (m) | rot. RPE (deg) |
| --- | ---: | ---: | ---: | ---: | ---: |
| iterative, no total gate | 96.3% | 1.324 | 0.031 | 0.287 | 12.034 |
| + 0.2 m/frame gate | 94.0% | 0.334 | 0.126 | 0.0520 | 1.332 |
| + 0.1 m/frame gate | 89.0% | 0.209 | 0.359 | 0.0439 | 1.243 |
| + 0.05 m/frame gate | 77.7% | 0.189 | 0.488 | 0.0394 | 1.360 |
| 0.1 m times failure-gap (velocity gate) | 89.7% | 0.198 | 0.360 | 0.0397 | 1.390 |
| 0.1 m + sparse stereo rebootstrap | 89.0% | 0.203 | 0.329 | 0.0379 | 1.140 |

The motion gate converts silent scale collapse into explicit tracking failures
and gives a real accuracy/coverage Pareto improvement, but it does not recover
metric scale or meet ORB-SLAM3: the official MH01 Stereo-Inertial reference is
0.042 m ATE at about 99.89% coverage. Sparse rebootstrap fired once and improved
RPE, but its current implementation appends stereo landmarks into the same map
gauge; it is not yet independent multi-map recovery. Artifacts are under
`E:/visloc_archive/iterative_correspondence_*_300_20260716*`. The next required
step is therefore an independent submap with a calibrated stereo scale and a
verified SE3 map-merge edge, plus a MAP visual-inertial initializer; relaxing
the motion gate would merely hide the cliff again.

The current benchmark host has a 6 GiB GTX 1660 Ti, below the official
DROID-SLAM inference requirement of roughly 11 GiB. Official DROID-SLAM is
therefore recorded as an external research reference on this host, not used for
same-hardware pass/fail claims. ORB-SLAM3 and COLMAP remain the executable
same-hardware gates.

### Pinned benchmark-host installation

The first official-baseline host installation is pinned as follows:

- COLMAP 4.1.0 CPU Windows build, commit `fa8e3b3`, executable archive SHA-256
  `dc8179bb4f3f48edec683bcec7627176b66e53a33ef0e2aa98d487f45873af5f`.
- Official South Building archive from the COLMAP 3.11.1 release assets,
  419,421,847 bytes, locally verified SHA-256
  `d210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1`.
  It contains 128 images and the supplied 221,487,104-byte feature/match
  database. The database SHA-256 is
  `702074ab5da8cfc7e7a53a1e3e4a49a0e3d18b88094d8b94b5ca972f2e08665a`;
  the 128-image tree SHA-256 is
  `c7016b9943e26148e2e7547ce3fd5ff55c367c351bbe12943e1294efff7314ec`.
  The mapper must consume that database unchanged except for its
  per-repetition copy and view-graph calibration.
- ORB-SLAM3 official master snapshot
  `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`.
- OpenCV 4.4.0 tag target `c3bb57afeaf030f10939204d48d7c2a3842f4293`
  and Pangolin v0.6 tag target
  `dd801d244db3a8e27b7fe8020cd751404aa818fd`, built externally. Pangolin's
  optional Python console/module is disabled; its GUI/OpenGL library remains
  enabled. ORB-SLAM3 itself is not patched.

All source, build trees, datasets and output artifacts above reside on the
external SSD. They are not vendored into this repository.

`scripts/run_official_baselines.py` records the executable, vocabulary,
calibration, timestamps and database SHA-256 values; the source revision; the
complete command; host information; wall time; peak resident memory; logs; and
raw trajectory or sparse-model artifacts. Each repetition gets a fresh output
directory. Each Global Mapper repetition also starts from a byte-identical copy
of the input COLMAP database. Dataset directories receive a deterministic tree
SHA-256 over relative paths, sizes and file contents, so a reused path with
changed images cannot masquerade as the same input.

## Leakage boundary

Ground truth is never an argument to either competing engine. The baseline
runner has no ground-truth option and records
`protocol.ground_truth_available_to_engine=false`. Trajectories are aligned and
scored only after the engine exits. The same external evaluator must score
ORB-SLAM3 and visloc-rs outputs.

For visloc-rs, the EuRoC runtime must work when
`mav0/state_groundtruth_estimate0` is physically absent. The image demo now
anchors its gauge at the first camera frame, uses strict measured stereo depth
by default, and uses the last successful estimated pose for a segment restart.
Fixed-depth bootstrap is available only through the explicitly diagnostic
`--allow-fixed-depth-bootstrap` switch and is disallowed for parity claims.

The GT-free smoke evidence from 2026-07-15 used a dataset view containing only
`cam0`, `cam1`, `imu0`, and `body.yaml`. A 30-frame real-image run tracked 30/30
frames, initialized 235 strict-stereo landmarks and ended with 6 keyframes and
504 landmarks. Its artifacts are under
`E:/visloc_archive/gt_free_smoke_20260715_01`; the follow-up semantic-output
check is under `E:/visloc_archive/gt_free_smoke_20260715_02` and records all
ATE/RPE fields as `None`, not false zeroes.

## ORB-SLAM3 run

Run the Python script in the same Linux environment used to build the official
executable. On Windows, running it inside WSL keeps Linux paths and dynamic
libraries unambiguous. Put large outputs on the external SSD.

```sh
python3 scripts/run_official_baselines.py orb-slam3 \
  --executable /opt/ORB_SLAM3/Examples/Stereo-Inertial/stereo_inertial_euroc \
  --vocabulary /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt \
  --settings /opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC.yaml \
  --sequence-dir /datasets/euroc/MH01 \
  --timestamps /opt/ORB_SLAM3/Examples/Stereo-Inertial/EuRoC_TimeStamps/MH01.txt \
  --sequence MH_01_easy \
  --source-revision v1.0-release \
  --out-root /mnt/e/visloc_archive/official/orb_slam3/MH_01_easy \
  --repetitions 5
```

Repeat for every declared EuRoC sequence. Do not compare one successful run
against a visloc median.

### Measured ORB-SLAM3 MH01 baseline

The Windows checkout's official timestamp text had CRLF endings, which the
official example retained as part of each image basename. A value-identical LF
copy was therefore created outside the repository (3,682 lines, SHA-256
`9354db11b423f7169eda7a3048d9594ee14572c6017e1439dee7861261b065fb`).
The executable SHA-256 is
`822407e7ea16eef36c9d0d60921a8f08d0723eefbbc48552e9bdeea383de4d44`;
the GT-free MH01 sequence tree SHA-256 is
`4b686cfc7fa302dc92e61f3c1965ba70ff0d667df815c824e098688452b4212e`.

The five official stereo-inertial runs completed as follows. Times include
vocabulary loading, tracking, mapping, shutdown and trajectory export; the
official example already constructs `System` with its viewer disabled.

| Run | wall time (s) | peak RSS (bytes) | frame poses | coverage | keyframes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 482.547 | 862,392,320 | 3,678/3,682 | 99.8914% | 129 |
| 2 | 484.569 | 853,491,712 | 3,678/3,682 | 99.8914% | 126 |
| 3 | 504.590 | 868,945,920 | 3,678/3,682 | 99.8914% | 128 |
| 4 | 484.393 | 854,134,784 | 3,678/3,682 | 99.8914% | 128 |
| 5 | 499.573 | 844,255,232 | 3,678/3,682 | 99.8914% | 128 |

The official MH01 gate is therefore median wall time 484.569 s, median peak
RSS 854,134,784 bytes and 99.8914% trajectory coverage. MH01 spans about 182 s,
so this official build is about 2.66 times slower than the recorded 20 Hz
stream on this host. The manifest is stored externally at
`E:\visloc_archive\official_baselines\orb_slam3_4452a3c_mh01_gtfree_lf_5x_20260715\manifest.json`.

Post-run evaluation with `scripts/evaluate_euroc_trajectory.py` associated
3,638 poses per run to the EuRoC body ground truth (maximum timestamp delta
256 ns), then applied metric SE(3) Umeyama alignment. Per-run translation ATE
RMSE was 0.04813, 0.04014, 0.04443, 0.04012 and 0.04202 m; the median is
0.04202 m. Median consecutive-pose translation RPE RMSE is 0.003463 m and
rotation RPE RMSE is 0.02012 degrees. Diagnostic Sim(3) scales range from
1.00766 to 1.00994. Evaluation reads GT only after each engine process has
exited; the full result is the adjacent external `evaluation.json`.

## COLMAP Global Mapper run

Feature extraction and matching happen before this command. To isolate the SfM
backend, visloc-rs and Global Mapper must consume the same image set and the
same feature/match database. View-graph calibration is enabled by default,
matching current COLMAP guidance; disable it only when the experiment declares
fixed calibrated intrinsics for both competitors.

```sh
python3 scripts/run_official_baselines.py colmap-global \
  --executable colmap \
  --database /datasets/south-building/database.db \
  --images /datasets/south-building/images \
  --sequence south-building \
  --source-revision 43dd3bb2 \
  --out-root /mnt/e/visloc_archive/official/colmap_global/south-building \
  --repetitions 5
```

Mapper settings can be pinned with repeated `--mapper-option NAME=VALUE`; view
graph settings use `--calibrator-option NAME=VALUE`. The manifest extracts
registered images, points, observations, mean track length and mean reprojection
error from `colmap model_analyzer`.

### Measured COLMAP 4.1.0 baseline

On 2026-07-15 the pinned CPU-only Global Mapper completed five fresh South
Building runs. The supplied database is already calibrated for this protocol,
so these runs use `--skip-view-graph-calibrator`: COLMAP 4.1.0's calibrator
aborts on this older official database because an `UNCALIBRATED` two-view
geometry lacks an F matrix. The failed calibrator attempt is retained as a
separate manifest; it is not included in the timing sample.

| Run | mapper time (s) | peak RSS (bytes) | registered | points | reprojection (px) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 110.073 | 745,357,312 | 128/128 | 57,944 | 0.522334 |
| 2 | 111.746 | 750,571,520 | 128/128 | 57,938 | 0.521210 |
| 3 | 116.363 | 749,838,336 | 128/128 | 57,941 | 0.522119 |
| 4 | 118.450 | 743,518,208 | 128/128 | 57,934 | 0.521324 |
| 5 | 131.697 | 742,633,472 | 128/128 | 57,947 | 0.522244 |

The official gate is therefore 100% image registration, median mapper time
116.363 s and median peak RSS 745,357,312 bytes. The mapper is slightly
nondeterministic in its accepted point set, so accuracy comparisons use the
per-run model metrics and their median/range rather than requiring identical
point counts. The manifest is stored externally at
`E:\visloc_archive\official_baselines\colmap_global_4.1.0_south_building_skip_calibrator_5x_20260715\manifest.json`.

The Global Mapper implementation may be ported directly into Rust from its
BSD-3-Clause COLMAP/GLOMAP sources as long as the source copyright and license
notices remain attached to derived code. ORB-SLAM3 is GPLv3, so its source is a
runtime baseline only; visloc-rs implements the corresponding ideas from the
papers and equations without copying GPL implementation code.

## DROID-SLAM design audit

The official [DROID-SLAM paper](https://arxiv.org/abs/2108.10869) and
[implementation](https://github.com/princeton-vl/DROID-SLAM) were audited on
2026-07-16. The repository is BSD-3-Clause, but the reference inference path is
not a drop-in Rust dependency: it uses PyTorch, custom CUDA, `lietorch`, learned
weights, and about 11 GB of GPU memory according to its README.

The useful architectural units are more precise than “iterate matching”:

- A dynamic directed frame graph contains temporal-neighborhood, proximity,
  and stereo self-edges. Duplicate edges are rejected; old edges become
  inactive; low-confidence nonlocal edges are removed.
- Each update reprojects the current pose/inverse-depth state, samples a
  multi-level correlation volume, and feeds correlation plus current flow and
  residual into a recurrent update operator. The operator jointly predicts a
  2D target correction, anisotropic confidence, and damping.
- Dense BA then updates camera poses and per-pixel inverse depth. The frontend
  runs a bounded local graph; a lower-memory backend periodically rebuilds a
  broader proximity graph and runs global BA. Non-keyframe poses are filled
  after optimization rather than being treated as independent keyframes.

The current visloc implementation adopts only the parts supported by its
sparse geometric backend: iterative projection/correspondence rebuilding,
query-to-landmark reassignment, confidence-weighted BA, radius tightening,
factor-pair retention, and metric pose trust regions. It does **not** yet
implement DROID-SLAM's dense all-pairs correlation, learned ConvGRU update,
joint pixelwise inverse-depth BA, or global learned factor graph, so results
must not be labelled “DROID-SLAM equivalent.” The next faithful sparse step is
an age/confidence-managed keyframe factor graph feeding local and periodic
global sparse BA; a future GPU backend can supply learned correlation targets
through the same factor interface.

The DROID-SLAM robustness lesson also changes cliff recovery. Continuing from
the last accepted pose can preserve a bad scale branch. The EuRoC image demo
therefore offers `--rebootstrap-independent-submap`: after the configured lost
frame threshold, it snapshots the old map into `MapAtlas`, resets all per-map
tracking/mapping/VI/pose-graph state, and starts calibrated stereo in an
identity local gauge. `submap_trajectory.csv` records local continuity. Until
a geometrically verified metric SE(3) bridge aligns that submap, its poses and
landmarks are excluded from `slam_trajectory.csv`, final keyframe ATE, and the
materialized global map. Summary output reports local tracking success and
globally aligned coverage separately.

A GT-free 12-frame structural smoke run is stored externally at
`E:\visloc_archive\atlas_independent_smoke_12_20260716`. An intentionally tiny
`1e-6 m` motion gate forced six stereo restarts. The run produced seven owned
submaps (one aligned root and six independent maps), local tracking success
`7/12`, and globally aligned success `1/12`. All twelve local rows appear in
`submap_trajectory.csv`, while `slam_trajectory.csv` marks the eleven
unverified-gauge rows unsuccessful. The aligned materialization contains only
the root's 235 landmarks and one keyframe; the 1,814 landmarks in the six
unverified maps are retained by the Atlas but excluded from global outputs.
This is a structure/isolation test, not an accuracy result.

The persistence smoke at
`E:\visloc_archive\atlas_independent_smoke_4_20260716_v2` additionally checks
the audit exports. `atlas_submaps.csv` contains one aligned and two independent
maps; `atlas_submap_landmarks.csv` contains 235, 259, and 317 local landmarks
respectively, with Atlas coordinates blank for the two unverified maps. Global
trajectory success is one of four rows.

Cross-submap merge is now GT-free and metric rather than a pose-only stitch.
For each independent keyframe, aligned target keyframes are ranked by mean
descriptor cosine similarity. A dedicated mutual cross-check matcher builds
one-to-one query/target-landmark correspondences; PnP estimates
`T_camera<-target`, which combines with the source keyframe's stored
`T_camera<-source` to recover `T_target<-source`. The same PnP inliers are
lifted to source/target 3D landmark pairs. Pair-distance ratios provide a
median/MAD metric-scale estimate, and transformed 3D residuals decide which
same-point relations are safe to weld. PnP therefore cannot hide a scale
branch: a synthetic case with identical pixels but source geometry scaled by
1.5 passes PnP and is rejected at estimated scale 0.667.

The real GT-free merge smoke at
`E:\visloc_archive\atlas_weld_smoke_4_20260716_v5` forced two cliffs. The two
bridges had 111/104 and 76/69 correspondence/inlier counts, mean reprojection
errors 0.803/0.859 px, and scale estimates 1.00135/0.99349. All three submaps
became globally aligned. `materialize_aligned` uses union-find over verified
landmark equivalences, so chained bridges share one deterministic output id;
observations, stereo measurements, confidence, descriptors, and rotated
covariances survive. Of 811 owned landmarks, 77 passed the stricter transformed
3D residual gate and were welded, leaving 734 global landmarks. The focused
two-frame persistence check at
`E:\visloc_archive\atlas_weld_smoke_2_20260716_v6` records 45 welded pairs in
both `atlas_merge_attempts.csv` and `summary.txt`.

The 300-frame GT-free run at
`E:\visloc_archive\atlas_projection_bridge_metric_gate_300_20260716_v3`
tests the same bridge without forcing a synthetic cliff. Four real tracking
cliffs created five owned submaps. Projection harvesting used only the last
accepted visual pose transported through the old aligned submap as a search
prior; it did not authorize a merge. PnP, metric scale consensus, and 3D
residual gates independently verified two bridges:

| source -> target | frame | PnP corr./inliers | reprojection (px) | scale | scale inlier ratio | welded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 -> 0 | 6 | 179/74 | 2.190 | 0.97866 | 0.6389 | 17 |
| 2 -> 0 | 59 | 140/80 | 1.378 | 0.98291 | 0.6633 | 21 |

Local tracking remained 269/300 (89.7%), but only 73/300 poses (24.3%) were in
a verified global gauge: the final two submaps produced repeated appearance
candidates but no acceptable PnP solution. Post-exit evaluation associated 56
of the 73 exported poses and measured SE(3) translation ATE 0.19348 m,
consecutive translation RPE 0.06509 m, and rotation RPE 1.2900 degrees. The
prior same-gauge rebootstrap run had SE(3) ATE 0.20286 m, but 89% coverage, so
the small ATE improvement is not a win: global recovery coverage regressed by
design because unverified gauges are now honestly excluded.

This failure localizes the next DROID-derived task. A single current-keyframe
bridge attempt is insufficient after appearance and viewpoint move away from
the boundary. The sparse frontend should retain temporal, proximity, and
stereo factors across a bounded active window; periodically rescore inactive
and nonlocal factors; remove persistently low-confidence factors; and run a
broader recovery/global pass before discarding the geometric overlap. Factor
targets and anisotropic information must be explicit data so a later learned
GPU update operator can replace the geometric proposer without changing the
metric PnP/scale acceptance gates.

That sparse factor lifecycle is now implemented in
`pipelines/slam/src/sparse_factor_graph.rs` and is opt-in in the EuRoC runner
with `--sparse-factor-graph`. Each directed factor has a stable key, temporal /
proximity / stereo kind, active or inactive state, support, confidence, 2D
target correction, anisotropic information, damping, age, and update count.
The geometric proposer creates bidirectional temporal and proximity edges plus
stereo self-edges. Active-window age, confidence patience, and an active-edge
budget retire factors without destroying them; explicit broader recovery can
reactivate retained factors. Inactive reasons distinguish low confidence,
window age, and budget pressure, preventing budget-retired factors from
thrashing back into the next active set. The same measurement record can later
be updated by a learned recurrent frontend without replacing the metric
backend.

The factor graph is not diagnostic-only: when covisibility BA is enabled, its
active temporal/proximity neighbors form the allowlist of variable keyframe
poses. Keyframes outside that set remain eligible as fixed boundary anchors.
Manual calibrated-stereo segment restarts explicitly synchronize their seed
keyframe after stereo sidecars are committed, so bootstrap stereo factors are
not lost by bypassing the normal `process_frame` mapping path.

A 20-frame GT-free integration smoke is stored at
`E:\visloc_archive\sparse_factor_graph_ba_smoke_20_20260716_v4`. It retained
24 active temporal, four proximity, and one stereo factor. All five local BA
triggers succeeded and reported two to five graph neighbors; tracking and
verified Atlas coverage both remained 16/20. This proves the graph controls
real optimizer windows, but it is not an accuracy gate.

The final 300-frame single-run comparison is stored at
`E:\visloc_archive\sparse_factor_graph_ba_metric_gate_300_20260716_v2`.
Ground truth remained absent from the engine process and was read only by the
post-exit evaluator. The reference arm is the otherwise identical
`atlas_projection_bridge_metric_gate_300_20260716_v3` run above:

| Metric | Atlas bridge reference | Sparse factor graph + graph-scoped BA |
| --- | ---: | ---: |
| local tracking | 269/300 (89.7%) | 275/300 (91.7%) |
| verified global poses | 73/300 (24.3%) | 60/300 (20.0%) |
| verified submap merges | 2/4 restarts | 1/3 restarts |
| SE(3) ATE (associated exported poses) | 0.19348 m | 0.12043 m |
| Sim(3) diagnostic scale | 0.19775 | 1.11843 |
| consecutive translation RPE | 0.06509 m | 0.04570 m |
| consecutive rotation RPE | 1.2900 deg | 1.1641 deg |
| wall time | 104.82 s | 68.45 s |
| per-frame p95 / p99 | 1429 / 2017 ms | 659 / 706 ms |

The graph arm performed 120 lifecycle updates, added 1,854 factors, retained a
bounded final active set of 72 temporal and 184 proximity factors, and ran 112
successful graph-scoped BA solves. After inactive-reason hysteresis,
ordinary rescoring caused zero automatic reactivations; 1,157 edges retired by
window age and 46 by the active budget. The ATE, scale, local tracking, and
latency changes are a strong single-run signal, not an ORB-SLAM3 win: verified
global recovery regressed by 4.3 percentage points and remains the binding
failure. The next stage must preserve cross-submap boundary factors and run a
broader Atlas-level recovery pass; a per-submap local graph cannot reconnect
geometry it no longer owns.

That next stage was tested rather than assumed. `MapAtlas` now retains a
deterministic source window (current, early boundary, then recent keyframes),
and the EuRoC runner can periodically search it with
`--atlas-broader-recovery-max-source-keyframes` and
`--atlas-broader-recovery-interval-attempts`. The 300-frame run at
`E:\visloc_archive\atlas_broader_recovery_metric_gate_300_20260716_v2`
executed 17 broader cycles and 203 source-view attempts. It produced exactly
the same 60/300 globally aligned poses, one merge, ATE and RPE as the
single-view factor baseline, while wall time rose from 68.45 s to 121.23 s
and p99 from 0.71 s to 4.46 s. The broader pass is therefore rejected as a
default; its source-window limit now defaults to one and remains available for
explicit experiments.

The failure is not just insufficient keyframe search. The online keyframe
policy had discarded the last successfully tracked non-keyframe before each
cliff. Atlas snapshots now preserve that frame and its landmark observations.
A DROID-style adjacent-boundary factor then directly matches the old last-good
view to the new calibrated-stereo seed, adds mutual 2D-flow proposals, and
applies deterministic 3D RANSAC, metric-scale consensus, target-view
reprojection, welding residuals, and the unchanged Atlas merge gate. The
factor interface and all diagnostics are exported in
`atlas_boundary_factors.csv`; no ground truth participates.

The final single-run gate is
`E:\visloc_archive\atlas_boundary_factor_metric_gate_300_20260716_v2`.
Its three boundary factors proposed 12/37/37 initial metric correspondences.
A causal last-good-pose prior was used only to rebuild 12/32/33 projected
correspondences; transforms were still re-estimated from 3D and the prior
could not authorize a merge. The factors found only 6/4/6 rigid 3D inliers
and were correctly rejected by the 30-inlier safety gate. The fallback
reproduced the factor baseline exactly:
local tracking 275/300 (91.7%), verified global coverage 60/300 (20.0%), one
merge, SE(3) ATE 0.12043 m, translation RPE 0.04570 m and rotation RPE
1.1641 degrees. Runtime was 62.30 s with p50/p95/p99 85/583/614 ms versus
68.45 s and 98/659/706 ms for the preceding factor run. This is a useful
single-run latency signal, but not a recovery or real-time win: verified
global coverage is unchanged and remains far below the 20 Hz requirement.
Lowering the merge threshold to turn four inliers into a reported success
would be an unsafe benchmark-specific change.

This is faithful to the control structure in the official
[DROID-SLAM frontend](https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/droid_frontend.py),
[backend](https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/droid_backend.py),
and [factor graph](https://github.com/princeton-vl/DROID-SLAM/blob/main/droid_slam/factor_graph.py):
aged edges move to an inactive store, proximity edges are added, inactive
edges can participate in updates, and low-memory backend updates run BA. It is
still only a sparse geometric analogue. The measured 4--6 boundary inliers
show that the next justified lever is a learned correlation/flow updater (or
materially denser stereo-observed landmark support), not weaker metric merge
verification.

The source audit also gives four more precise implementation constraints. The
official motion filter admits a frame only after one recurrent update predicts
enough mean flow; the frontend caps the active graph at 48 factors, retires
edges older than 20 updates into an inactive store, adds NMS-suppressed
proximity edges, and uses inactive factors during each update. It then removes
a redundant interior keyframe using a bidirectional learned-motion distance.
The backend rebuilds a broader proximity graph and performs low-memory
updates, again with inactive factors enabled. Therefore the next DROID-inspired
work is not to copy its CUDA/PyTorch runtime wholesale. The compatible path is:

1. add an image-motion admission score separate from the current pose/inlier
   keyframe policy;
2. preserve factor measurements when an edge becomes inactive and allow them
   to constrain recovery updates;
3. replace Euclidean camera-centre proximity with a bidirectional
   correspondence/flow distance plus NMS; and
4. run bounded recurrent correspondence correction before metric BA/PnP, while
   retaining calibrated stereo and IMU as the final scale/acceptance authority.

This keeps the useful DROID-SLAM control structure while avoiding a false
claim that the current zero-correction sparse factors are equivalent to its
learned dense update operator.

## Win gates

ORB-SLAM3 parity requires five-run medians on identical sequence data and host:
rigid ATE/RPE, tracked-frame coverage, longest continuous segment, recovery
rate, wall time and p95 per-frame latency. The runtime must not read ground
truth, and a 20 Hz EuRoC stream must not accumulate an unbounded backlog.

Global Mapper parity requires identical images, intrinsics policy, features and
matches: registered-image ratio, pose accuracy where reference poses exist,
reprojection error, failure rate, wall time and peak memory. A win requires
non-inferior reconstruction quality plus a material speed/memory advantage, or
materially better quality without an unacceptable resource regression.
