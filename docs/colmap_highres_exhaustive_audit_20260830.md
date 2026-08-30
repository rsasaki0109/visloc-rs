# High-resolution exhaustive COLMAP matching audit (2026-08-30)

This is a non-destructive control for the original-resolution ETH3D courtyard
images.  It removes the earlier overlap-3 candidate restriction, but does not
change the default visloc candidate schedule or feed official camera poses into
matching, merging, or reconstruction.  The evaluator used here is the local
official-calibration/`gt` proxy described in
[`evaluator_audit_20260830.md`](evaluator_audit_20260830.md); an independent
laser-camera pose file is not present in the local dataset.

## Inputs and provenance

- Images: `/home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted`
  (38 `.JPG` files).
- Per-image calibration: `/home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted`
  (`cameras.txt`, `images.txt`, `points3D.txt`).
- External feature directory:
  `/tmp/colmap_official_highres_8192_20260830/features_sixcol`.
  It contains the six-column conversion of the official CPU SIFT output,
  439,481 rows in 38 files.  The sorted filename/bytes manifest is
  `506395c2cfc165865bc2d02a5960c2e3bd55475c625efd2d473463bb415fcd9d`;
  per-image counts range from 9,663 to 17,689.
- Extraction database:
  `/tmp/colmap_official_highres_8192_20260830/database.db`, SHA-256
  `b676d6fcde13ff3e44b38a125348f251b8159b01860e9dbc404eaf0086a7acb1`.
- COLMAP container: `colmap/colmap:latest`, reported version
  `4.2.0.dev0`, image digest
  `sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`.

The extraction settings were CPU SIFT, `first_octave=-1`, four octaves,
`max_num_features=8192`, peak threshold `0.00667`, edge threshold `10`, and
maximum orientations `2`.  No feature extraction was repeated for the
visloc run below.

## Exhaustive COLMAP matching

The fresh exhaustive database was
`/tmp/colmap_official_highres_8192_exhaustive_v2_20260830/database_exhaustive.db`,
SHA-256
`9bddce4ba9ac5e1bd4426f8542d933f4b1414a6c5650ab6cee3bbdf11908a2ef`.
The matching command was:

```text
docker run --rm -v "$OUT:/out" colmap/colmap:latest colmap exhaustive_matcher \
  --database_path /out/database_exhaustive.db \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 0 \
  --FeatureMatching.num_threads 8 --FeatureMatching.guided_matching 1 \
  --FeatureMatching.max_num_matches 32768 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 --SiftMatching.cpu_brute_force_matcher 1 \
  --TwoViewGeometry.max_error 4 --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.random_seed 0 --ExhaustiveMatching.block_size 50
```

It completed successfully in `2:00:55` with eight matching workers (observed
host RSS was approximately 5.7 GB).  All 703 image pairs have a raw match row;
the raw total is 306,324.  The first verification pass retained 426 geometry
rows and 433,248 inlier rows (`config=3`: 243, `config=6`: 183).  Because the
copied database initially had `prior_focal_length=0`, those rows carried F/H
models without E/relative-pose blobs.  This was corrected without rematching:
setting `prior_focal_length=1` and rerunning `colmap geometric_verifier` took
12.996 s and produced
`/tmp/colmap_official_highres_8192_exhaustive_calibrated_20260830/database.db`,
SHA-256
`f64fbd7da339d4023c63190bd0d46a9c1cb81cfe815a012e19b42cada3e11810`.
The calibrated database has 428 geometry rows and 433,279 inliers:
`config=2`: 2, `config=3`: 243, `config=6`: 183.  Matching contents were
unchanged; only the two newly calibrated rows were added.

## Cross-component graph inventory

The earlier overlap-3 official model split into model 0 (`DSC_0286..0308`,
23 images) and model 1 (`DSC_0308..0323`, 16 images), sharing only `DSC_0308`.
This audit uses that membership only to define the diagnostic 22-by-15
exclusive pair set; it is not used as a reconstruction constraint.

- All 330 exclusive cross-component pairs have raw rows (75,428 raw matches).
- The calibrated exhaustive graph has 198 geometry rows / 119,595 inliers on
  this cross set (`config=6`: 107, `config=3`: 89, `config=2`: 2).
- The 37 exclusive vertices form one connected component.  In particular,
  the four pairs that were present as raw-only rows in the old overlap-3
  database are no longer a structural limitation; the exact old examples
  (0305-0309, 0306-0310, 0307-0309, 0307-0311) had only 0 verified rows there,
  while the exhaustive graph supplies many other nonlocal bridges.
- For the 59 F-to-E candidates with support at least 100 and p25
  triangulation angle at least 1 degree, the translation-direction constraint
  matrix has rank 4 and singular values
  `17.577, 7.679, 6.832, 2.147` (condition number about `8.19`).  A least
  squares scale fit is `1.9333`; direction residuals have median about
  `1.05 deg`, p90 about `2.62 deg`, and a maximum about `53.55 deg`.
  Thus the newly observed graph is sufficiently connected/conditioned for a
  relative Sim(3) diagnostic, although it contains clear outlier edges.

## Official mapper controls

The single-model and multiple-model COLMAP controls used the same exhaustive
database, authoritative per-image PINHOLE calibration, CPU mapper, and the
following relevant settings: `multiple_models` as indicated, `init_num_trials
200`, `init_min_num_inliers 100`, `init_max_error 4`, `init_min_tri_angle 16`,
`abs_pose_min_num_inliers 30`, `abs_pose_max_error 12`,
`filter_max_reproj_error 4`, `filter_min_tri_angle 1.5`, `num_threads 8`, and
`random_seed 0`.

Using the initial exhaustive database (with calibrated camera priors), both
single and multiple mapper controls reached 38/38 with 38,451 points and
169,639 observations, mean point error 0.743985 px (median 0.62194, p95
1.81917), and calibration-proxy centre RMSE **1.896 cm**.  The multiple-model
run produced only one model, so there was no merge to perform.

Using the geometrically reverified calibrated database, both controls again
reached 38/38 and produced the same model statistics: 38,422 points, 169,590
observations, mean point error 0.744758 px (median 0.622468, p95 1.82057), and
calibration-proxy centre RMSE **1.6166 cm**.  The calibrated single-model
output is in
`/tmp/colmap_official_highres_8192_exhaustive_calibrated_mapper_20260830/sparse_single_txt`.
For the earlier overlap-3 comparison, the independent 23-image left component
scored 2.964 cm on its matched subset and the independent 16-image right
component scored 2.466 cm; those values are not scores of a jointly aligned
38-camera model.

## visloc all-pairs control

The raw matches were exported with the existing
`scripts/export_colmap_matches.py` (703 pairs) and imported without
re-extraction.  The all-pairs run used:

```text
RAYON_NUM_THREADS=1 VISLOC_SFM_DEBUG=1 VISLOC_SFM_DEBUG_IMAGES=20,21 \
target/release/examples/unordered_sfm_demo \
  --feature-extractor files \
  --features-dir /tmp/colmap_official_highres_8192_20260830/features_sixcol \
  --feature-suffix _features.txt --image-suffix .JPG \
  --images-dir /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted \
  --input-colmap-calibration /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted \
  --import-matches-file /tmp/colmap_official_highres_8192_exhaustive_v2_20260830/matches_import.txt \
  --exhaustive --min-matches 20 --verification-mode full --mapper incremental \
  --pnp-max-iterations 100000 --min-pnp-inliers 8 \
  --geometry-guided-conflict-recovery --post-refinement-registration \
  --final-iterative-refinement \
  --out-colmap /tmp/colmap_official_highres_8192_visloc_exhaustive_allpairs_20260830
```

The internal view graph contains all 703 imported pairs.  Verification retained
**366/703 pairs** and **261,724 inlier correspondences** (701 classified rows:
57 calibrated, 314 uncalibrated, 2 planar, 328 degenerate).  Growth initially
reached 37/38; post-registration admitted image 28 with 3,313 correspondences
and 2,962 inliers, yielding **38/38**.  The final model contains 43,852 tracks
and 152,432 observations with mean reported reprojection error **0.579 px**.
The calibration-proxy score is **0.5379 cm RMSE**, median 0.3042 cm, with
Sim(3) scale 2.474044.  Output is in
`/tmp/colmap_official_highres_8192_visloc_exhaustive_allpairs_20260830`.

For comparison, omitting `--exhaustive` left the internal VLAD candidate list
at 259 pairs: 195/259 verified, 200,290 inliers, 26/38 registered, 29,639
tracks/80,632 observations, 0.451 px, and 3.612 cm.  The difference is the
candidate schedule, not a feature re-extraction.

The all-pairs visloc command was repeated in
`/tmp/colmap_official_highres_8192_visloc_exhaustive_allpairs_repeat_20260830`.
It reproduced 366/703, 38/38, 43,852/152,432, 0.579 px, and 0.5379 cm; the
`cameras.txt`, `images.txt`, and `points3D.txt` output hashes were byte-
identical to the first run (`76fc7583...`, `a14ac6b9...`, and `d7b680e6...`,
respectively).

## Merge decision

The old overlap-3 two-submap outputs were also transformed using the new
cross-edge translation directions as a diagnostic.  A non-robust 59-edge
least-squares merge scored 9.8599 cm; deterministic two-edge robust selection
reduced this to 5.37345 cm (58/59 retained), still worse than the calibrated
official all-pair mapper (1.6166 cm) and the visloc all-pair result (0.5379 cm).
The graph is now observable, but merging already-diverged submaps is not a
competitive path.  No default-off production merge code was added.  The
evidence-backed next control is the direct exhaustive graph, which is already
complete and reproducible; further merge work should require a separate
quality gate and a demonstrated improvement over these controls.

All reconstruction and merge fitting above used image features, verified
matches, and supplied calibration only.  Official model poses/GT were used
after fitting for scoring and for the explicitly labelled merge/evaluator
diagnostics.
