# Reproducibility and CI closure (2026-08-30)

This document closes the current high-resolution courtyard control and records
the cross-dataset non-regression blockers.  The repository was not committed;
existing user changes remain in the worktree.  All reconstruction fitting used
image features, verified matches, and supplied calibration only.  The local
`gt` file is the ETH3D calibration model symlink, so the courtyard score below
is a calibration/pose proxy, not an independent laser-camera score (see
[`evaluator_audit_20260830.md`](evaluator_audit_20260830.md)).

## Durable courtyard control

The non-recreatable control is preserved at:

```text
/media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830
```

It contains the converted 38-image feature files, extraction and exhaustive
matching databases/logs, calibrated official mapper output, the first visloc
model, and its repeat model.  The directory is about 465 MiB.  Its current
`SHA256SUMS` has 64 entries and digest
`12e91cd3a2e595625ef167d8cd8a2af6310d3ea3cd1e3b1a0c2f8264004fa96b`; running
`sha256sum -c` from `/` reports `OK` for every entry.  Important hashes are:

| artifact | SHA-256 |
|---|---|
| extraction DB | `b676d6fcde13ff3e44b38a125348f251b8159b01860e9dbc404eaf0086a7acb1` |
| exhaustive raw-match DB | `9bddce4ba9ac5e1bd4426f8542d933f4b1414a6c5650ab6cee3bbdf11908a2ef` |
| calibrated geometry DB | `f64fbd7da339d4023c63190bd0d46a9c1cb81cfe815a012e19b42cada3e11810` |
| feature `MANIFEST.tsv` | `030b02982e263f3b5e3d94d1edd873b757894b4c20a9609feef264cadffd0d2a` |
| first `visloc_model/cameras.txt` | `76fc758375228319300a5c076b6bcc84a88413d69cf3613eb40c980b60e8cc9c` |
| first `visloc_model/images.txt` | `a14ac6b958bf09481bfcc0ae72b59671d8e6405cd880e4711ff14c5c9432852e` |
| first `visloc_model/points3D.txt` | `d7b680e6d51a403a4962920e8f0e0615f382646a1ac9836f160ecb44622a3293` |
| repeat `images.txt` | same as first |
| repeat `points3D.txt` | same as first |

The complete per-file manifest, source provenance, and earlier overlap-3
comparison are in
[`colmap_highres_exhaustive_audit_20260830.md`](colmap_highres_exhaustive_audit_20260830.md).

### Exact extraction and matching

The source images are
`/home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted` and
the four-camera PINHOLE calibration is
`/home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted`.
The extraction used Docker `colmap/colmap:latest`, reported COLMAP
`4.2.0.dev0`, image digest
`sha256:b809882552887b6471094dcadd2f2eb01656b010663564c43a5e7f04c0a08f2f`:

```bash
docker run --rm \
  -v /home/sasaki/datasets/eth3d/courtyard/images/dslr_images_undistorted:/images:ro \
  -v /tmp/colmap_official_highres_8192_20260830:/out \
  colmap/colmap:latest colmap feature_extractor \
  --database_path /out/database.db --image_path /images \
  --ImageReader.camera_model PINHOLE --ImageReader.single_camera 0 \
  --FeatureExtraction.type SIFT --FeatureExtraction.use_gpu 0 \
  --FeatureExtraction.num_threads 1 \
  --SiftExtraction.max_num_features 8192 \
  --SiftExtraction.first_octave -1 --SiftExtraction.num_octaves 4 \
  --SiftExtraction.octave_resolution 3 \
  --SiftExtraction.peak_threshold 0.00667 \
  --SiftExtraction.edge_threshold 10 \
  --SiftExtraction.max_num_orientations 2 \
  --SiftExtraction.upright 0 --SiftExtraction.estimate_affine_shape 0 \
  --SiftExtraction.domain_size_pooling 0
```

The extraction produced 439,481 rows.  The exact conversion to the visloc
six-column text format was:

```bash
python3 /media/sasaki/aiueo1/visloc-rs/scripts/export_colmap_sift_features.py \
  --database /tmp/colmap_official_highres_8192_20260830/database.db \
  --out-dir /tmp/colmap_official_highres_8192_20260830/features_sixcol
```

All row coordinates/descriptors were checked against the database blobs.  The
all-pairs raw matcher command was:

```bash
docker run --rm \
  -v /tmp/colmap_official_highres_8192_exhaustive_v2_20260830:/out \
  colmap/colmap:latest colmap exhaustive_matcher \
  --database_path /out/database_exhaustive.db \
  --FeatureMatching.type SIFT_BRUTEFORCE --FeatureMatching.use_gpu 0 \
  --FeatureMatching.num_threads 8 --FeatureMatching.guided_matching 1 \
  --FeatureMatching.max_num_matches 32768 \
  --SiftMatching.max_ratio 0.8 --SiftMatching.max_distance 0.7 \
  --SiftMatching.cross_check 1 --SiftMatching.cpu_brute_force_matcher 1 \
  --TwoViewGeometry.max_error 4 --TwoViewGeometry.min_num_inliers 15 \
  --TwoViewGeometry.random_seed 0 --ExhaustiveMatching.block_size 50
```

The copied DB camera rows were set to `prior_focal_length=1` before the
calibrated re-verification.  The exact postprocess command was:

```bash
docker run --rm \
  -v /tmp/colmap_official_highres_8192_exhaustive_calibrated_20260830:/out \
  colmap/colmap:latest colmap geometric_verifier \
  --database_path /out/database.db \
  --TwoViewGeometry.min_num_inliers 15 --TwoViewGeometry.multiple_models 0 \
  --TwoViewGeometry.compute_relative_pose 1 --TwoViewGeometry.max_error 4 \
  --TwoViewGeometry.confidence 0.999 --TwoViewGeometry.max_num_trials 10000 \
  --TwoViewGeometry.random_seed 0 --batch_size 1000 --num_threads 8
```

This gives 703 raw pair rows and 428 calibrated geometry rows / 433,279
inlier rows.  It is an exhaustive candidate set: there are
`38*37/2 = 703` unordered image pairs.  In the visloc run, `--exhaustive`
selects all 703 candidates and `--import-matches-file` supplies the already
computed raw index pairs; it does not re-extract features or silently replace
the imported indices.  Verification subsequently retains 366/703 pairs and
261,724 inlier correspondences.

### Exact mapping and evaluation

The first mapping command (the repeat changes only `--out-colmap`) was:

```bash
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

The scorer command was:

```bash
python3 /media/sasaki/aiueo1/visloc-rs/scripts/score_umeyama_centers.py \
  --est /media/sasaki/aiueo1/visloc-rs/eth3d/courtyard/artifacts/colmap_highres_exhaustive_allpairs_20260830/visloc_model/images.txt \
  --gt /home/sasaki/datasets/eth3d/courtyard/dslr_calibration_undistorted/images.txt
```

It reports 38/38, Sim(3) scale `2.474044`, RMSE **0.5379 cm**, median
**0.3042 cm**, and mean reprojection **0.579 px** for 43,852 tracks / 152,432
observations.  The all-38 aligned centre residual distribution (cm) is:

| p25 | p50 | p75 | p90 | p95 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2235 | 0.3042 | 0.5017 | 0.9629 | 1.0336 | 1.2526 | 1.2737 |

The repeat run reproduced the 366/703 verification count, 38/38 registration,
43,852/152,432 support, 0.579 px reprojection, score 0.5379 cm, and byte-
identical `cameras.txt`, `images.txt`, and `points3D.txt` hashes.  The logged
effective-config hashes differ (`ff262e1392434ffb` and `927c2bea62efd815`)
only because the output path is included in the diagnostic argument snapshot;
the semantic flags and all input bytes are unchanged.  The durable copy also
contains both logs/models and the score text.

## Cross-dataset regression closure

The downloaded inputs, extraction attempts, and current results are detailed
in [`nonregression_20260830.md`](nonregression_20260830.md).  An isolated
external environment now supplies CPU torch/LightGlue/evo (Python 3.12.3,
torch 2.3.1+cpu, torchvision 0.18.1+cpu, LightGlue commit
`eb42fee2d71449efb0aa5c10549752b5d75384d8`, evo 1.31.1; pip-freeze digest
`dae41bf42ceedf9a214cd040d002f941d88f8dcc036d5ecd1dc637808dad8f9f`).  The
South, terrace, and office archives are SHA-256 verified under
`/media/sasaki/aiueo1/visloc-rs/eth3d/nonregression_20260830`.

The comparison is **partially measured, not fully passed**:

| suite | archived acceptance control | current session |
|---|---|---|
| South Building | 128/128, 1.09 cm (documented default; 0.40 cm archived colmap-style arm) | default **127/128, 0.74 cm** (repeat byte-identical; missing `P1180163.JPG`); opt-in colmap-style **128/128, 0.40 cm** |
| ETH3D terrace | 23/23, 12.37 cm laser-GT RMSE | external 3200 frontend **23/23, 132.07 cm**; current full-res helper **23/23, 130.62 cm**; historical cached features unavailable, so incomparable |
| ETH3D office | 18/26, 0.37 cm laser-GT RMSE | external 3200 frontend **18/26, 1.28 cm**; current full-res helper **17/26, 0.35 cm**; historical cached features unavailable, so incomparable |
| EuRoC MH_03 | open/loop 2.462/2.203 m ATE RMSE (SE3), 0.464/0.443 m (Sim3) | not measured; no valid `mav0`; official URLs timeout/403 and archive metadata is 12,096.15 MB |

The durable South/terrace/office outputs, EuRoC download evidence, and all CI
logs are under `.../nonregression_20260830/runs/`.  South's default arm has a
one-image registration regression candidate despite lower reference RMSE.
Terrace/office results cannot be compared to the archived cached-feature
controls; EuRoC remains blocked rather than substituted.

## CI and default-off audit

On Linux, the complete repository gate set passed on the current dirty tree:

| gate | result |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| same clippy command with `--features image-io` | pass |
| `cargo test --workspace --all-targets` | 1,397 passed, 0 failed, 11 ignored (95 test binaries) |
| same tests with `--features image-io` | 1,455 passed, 0 failed, 11 ignored (108 test binaries) |
| `python3 -m unittest discover -s tests -p 'test_*.py'` | 241 passed, 8 skipped |
| `sh scripts/check.sh` | pass |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` | pass |
| `sh scripts/package_check.sh` | pass |
| `cargo test --test api_stability -- --nocapture` | 4 passed |
| feature matrix, MSRV, registry, docs links/release, examples, trajectory, GNSS/timestamped/KITTI output checks | pass |
| `git diff --check` | pass |

The only compilation-quality correction made during the gate was a one-line
`enumerate()` rewrite in the already-dirty `global_sfm.rs` test to remove a
`needless_range_loop` warning; behavior is unchanged.  No tracked user file
was reset and no commit was created.  Snapshot round-trip, float-bit/order,
checksum, and schema-rejection tests are included in the workspace suite;
verified-pair snapshot imports and all experimental mapper/detector flags
remain explicit opt-ins.  The high-resolution champion itself intentionally
enables only its recorded recovery/post/final controls; no new default was
enabled by this closure.

Windows-only gates were not run on this Linux host.  The external artifact and
non-regression archive copies are non-destructive and separate from the repo;
the next actionable steps are to resolve South's default growth shortfall,
recover the exact historical terrace/office feature caches, and provision
EuRoC `mav0` before claiming closure.
