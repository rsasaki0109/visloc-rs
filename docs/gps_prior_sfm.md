# GPS priors in incremental SfM

The incremental photo mapper (`visloc_slam::incremental_sfm`, behind
`unordered_sfm_demo` and the `colmap` CLI shim) can use a GPS fix per image,
the way COLMAP's `pose_prior_mapper` (`use_prior_position`, COLMAP ≥ 3.10, and
the [colmap-gps](https://github.com/Vincentqyw/colmap-gps) fork) does. The
model then comes out metric and georeferenced in a local East-North-Up frame,
and the fixes correct the drift that a monocular reconstruction accumulates.

## How it works

Each image may carry a prior camera centre `p` (metres, local ENU) with a
per-axis standard deviation `σ`. Once at least three registered images have
one, every global and periodic bundle adjustment:

1. **Aligns** the model to the priors with a robust Sim(3): deterministic
   RANSAC over three-point Umeyama fits, then a refit on the inliers. The
   inlier threshold is `3 · mean σ` unless set. This moves the model out of
   the arbitrary frame and scale of its seed pair.
2. **Adds a prior** `(C − p) / σ` on each prior image's camera centre (weight
   `1/σ²` per axis, in the same squared-pixel units as the reprojection
   terms). The priors fix all 7 gauge freedoms, so the ordinary BA's anchor
   and scale-anchor poses are left free.

Every prior enters the BA, including the alignment's outliers, as in COLMAP.
When the model disagrees with a fix, the model is often what is wrong (drift,
a bad registration), and that prior is what pulls it back. Two opt-ins handle
bad fixes:

- a Cauchy loss on the normalised residual, COLMAP's
  `use_robust_loss_on_prior_position` (scale `sqrt(χ²₀.₉₅(3)) ≈ 2.8` by
  default). It is applied by reweighting each BA's prior at its current
  residual, so it works as IRLS across the mapper's BA calls;
- dropping the alignment's outliers from the BA.

With real onboard GNSS (the gravel-mine set below) both opt-ins are worse
than plain least squares, so neither is the default. On simulated GPS with
30 m jumps (ETH3D courtyard) they trade a worse median for fewer cameras
far off.

Fewer than three registered prior images leave every solve unchanged, so
the model is byte-identical to a run without priors. When the inlier priors
are close to collinear (a straight drive), GPS does not observe the roll
about that line. The first registered pose's rotation is then held fixed in
the BA, and the model's roll about the line is arbitrary.

## Inputs

- **EXIF GPS** (`GPSLatitude`/`GPSLongitude` with their refs,
  `GPSAltitude`/`GPSAltitudeRef`): `visloc_io::gps_prior::read_exif_gps`. A
  fix without an altitude gets an unweighted vertical axis. A `0, 0` fix
  counts as no fix.
- **A prior file**: one line per image, `#` comments, whitespace or commas:

  ```text
  # NAME          LAT          LON          ALT    [STD_H [STD_V]]
  IMG_0001.JPG    35.681236    139.767125   40.2   3.0    6.0
  ```

  `NAME` matches the image's file name, a path ending in it, or its stem.
  With the `cartesian` format the three numbers are `X Y Z` in a metric
  frame, used as they are (COLMAP `model_aligner`'s `ref_images` layout).

WGS84 fixes are converted exactly (WGS84 → ECEF → ENU) to the tangent frame
at the mean fix. The mappers write that origin to `enu_origin.txt` next to
the model. EXIF altitude is above sea level, not the ellipsoid. The
difference is a constant offset over a capture and does not matter.

## Use

`unordered_sfm_demo` (incremental mapper only):

```bash
cargo run --release --features image-io --example unordered_sfm_demo -- \
  --feature-extractor sift --images-dir photos/ --image-suffix .jpg \
  --width 4000 --height 3000 --fx 3000 --fy 3000 --cx 2000 --cy 1500 \
  --gps-priors-from-exif --gps-prior-std 1 --gps-prior-std-vertical 2 \
  --out-colmap sparse/
```

| Flag | Meaning |
|---|---|
| `--gps-priors FILE` | priors from a file (instead of EXIF) |
| `--gps-priors-format wgs84\|cartesian` | how the file's three numbers read (default `wgs84`) |
| `--gps-priors-from-exif` | priors from the images' EXIF GPS |
| `--gps-prior-std M`, `--gps-prior-std-vertical M` | σ where a fix states none (default 1 m each, as COLMAP; see "Choosing σ") |
| `--gps-prior-max-error M` | inlier threshold of the alignment (default 3 σ) |
| `--gps-prior-robust-loss`, `--gps-prior-loss-scale C` | Cauchy loss on the priors |
| `--gps-prior-reject-outliers` | drop the alignment's outliers from the BA |

`colmap` CLI shim, mirroring COLMAP's command surface:

```bash
colmap feature_extractor --database_path db.db --image_path images/   # stores EXIF GPS
colmap exhaustive_matcher --database_path db.db
colmap pose_prior_mapper --database_path db.db --output_path out/ \
  --prior_position_std_x 1 --prior_position_std_y 1 --prior_position_std_z 2
```

`mapper --Mapper.use_prior_position 1` is the same as `pose_prior_mapper`.
Also accepted: `--overwrite_priors_covariance 1`,
`--use_robust_loss_on_prior_position 1`, `--prior_position_loss_scale C`,
`--Mapper.prior_position_ransac_max_error M`, and this shim's
`--pose_priors_path FILE` / `--pose_priors_format cartesian`.

Both CLIs end by printing how well the model agrees with its priors, for
example `gps priors: 18/23 registered prior images agree, RMSE 1.180 m
(residual scale 1.0021)`.

Library:

```rust
use visloc_rs::io::gps_prior::{read_exif_gps, GpsPriorSet, PriorCoordinates};
use visloc_rs::slam::incremental_sfm::SfmPositionPriors;

let mut set = GpsPriorSet::new(PriorCoordinates::Wgs84);
for name in &image_names {
    if let Some(fix) = read_exif_gps(&image_dir.join(name)) {
        set.insert_geodetic(name.clone(), fix);
    }
}
let (local, enu_frame) = set.to_local_enu(&image_names, None, Vector3::new(1.0, 1.0, 1.0));
let config = IncrementalSfmConfig {
    position_priors: Some(SfmPositionPriors::from_local_priors(&local)),
    ..IncrementalSfmConfig::default()
};
let result = incremental_sfm(&camera, &features, &pairwise, &config)?;
// result.poses are now camera poses in `enu_frame`.
```

## Measured: ETH3D courtyard with simulated GPS

This uses the 38 bundled courtyard photos (`bindings/sfm-web/web/samples/courtyard`,
800×534) and ETH3D's ground-truth camera centres
(`dslr_calibration_undistorted/images.txt` from `courtyard_dslr_undistorted.7z`).
GPS fixes are simulated from the ground truth: Gaussian noise of σ 0.5 m
east/north and 1.0 m up (1.10 m RMS), plus 30 m jumps on DSC_0289 and
DSC_0301. They are written into the JPEGs' EXIF and read back by
`--gps-priors-from-exif --gps-prior-std 0.5 --gps-prior-std-vertical 1.0`.
The mapper flags are `--exhaustive --sift-max-keypoints 8192
--verification-mode full --min-matches 15 --min-pnp-inliers 10`. The error is
each registered camera centre's distance to its ground truth in the ENU
frame, with no alignment after mapping.

| Run | Registered | RMSE | Median | Max |
|---|---:|---:|---:|---:|
| No priors, then Sim(3)-aligned to the GPS (the jumps removed by hand) | 23 | 2.20 m | 1.50 m | 5.95 m |
| **GPS priors in the BA (default)** | 25 | 3.15 m | **0.52 m** | 8.65 m |
| GPS priors, Cauchy loss | 26 | 1.60 m | 1.19 m | 3.67 m |
| GPS priors, alignment outliers dropped | 21 | **1.44 m** | 0.80 m | 4.54 m |

- **Default.** 15 of the 25 cameras end closer to the truth than their own
  GPS fix (1.1 m RMS).
  - DSC_0296 (6.0 m) and one cluster, DSC_0314–0316 (7–9 m), stay
    misplaced: the 800-pixel visual model puts them wrong, and the priors
    do not pull them all the way back.
  - The 30 m jump on DSC_0289 leaves its camera 0.33 m from the truth: the
    visual terms outweigh one bad fix.
- **The two opt-ins** weaken every prior that disagrees with the model.
  - No camera ends as far off (at most 3.7 m and 4.5 m).
  - Fewer cameras beat their fix: 11 of 26 and 11 of 21.
- **This small set is fragile.** Which cameras register, and which of them
  go wrong, change from run to run. With the SIFT before its rotation fix
  (see the gravel section), the default had the best RMSE and both opt-ins
  the worst. The default is chosen on real GNSS below.

`colmap` shim on the same fixes (images upscaled 2× to the shim's ETH3D
1600-pixel camera, `pose_prior_mapper --prior_position_std_x 0.5
--prior_position_std_y 0.5 --prior_position_std_z 1`): its plain matcher
registers only 11–12 of 38. Against the ground truth that gives RMSE
0.77 m with `exhaustive_matcher` and 0.45 m with `spatial_matcher`
(10 neighbours). Those cameras span only a few metres, so 1 m GPS noise
leaves the scale 16–22 % off even where the shape is right (0.04 m after
the best similarity). **GPS fixes the scale only when the cameras spread
much farther than σ.**

Reproduce (needs numpy and Pillow, plus the 500 MB ETH3D archive for the
ground truth):

```bash
TRUTH_SCALE=1 python3 scripts/gps_prior/make_exif_gps.py \
  ETH3D/courtyard/dslr_calibration_undistorted \
  bindings/sfm-web/web/samples/courtyard gps_images truth.csv
cargo run --release --features image-io --example unordered_sfm_demo -- \
  --feature-extractor sift --images-dir gps_images --image-suffix .jpg \
  --width 800 --height 534 --fx 440 --fy 440 --cx 400 --cy 267 --exhaustive \
  --sift-max-keypoints 8192 --verification-mode full --min-matches 15 \
  --min-pnp-inliers 10 --gps-priors-from-exif --gps-prior-std 0.5 \
  --gps-prior-std-vertical 1.0 --out-colmap out/
python3 scripts/gps_prior/eval_centres.py out truth.csv
```

`scripts/gps_prior/posthoc_alignment.py` scores a model built without priors
after a Sim(3) to the GPS. `upscale_keep_exif.py` makes the shim's input.

## Measured: real onboard GNSS (gravel-mine UAV)

This is the open-pit gravel-mine set of Shahbazi et al., "Unmanned aerial
image dataset: Ready for 3D reconstruction", *Data in Brief* 25 (2019),
CC BY 4.0. A 16 MP camera at about 70 m took 158 nadir images.

- **GPS:** `DirectEOPs.txt`, the camera centres from the drone's
  low-cost GNSS/IMU.
- **Reference:** `RefinedEOPs.txt`, centres bundle-adjusted against 109
  RTK/total-station control points (2–5 mm).

Against the reference, the onboard GNSS is off by RMSE 3.4 m (registered
images). Most of that is a constant bias of (1.0, 1.5, 2.4) m; without it,
1.7 m RMSE remains.

`scripts/gps_prior/gravel_prepare.py` undistorts the images with the
dataset's calibration and scales them by 1/4 to 1218×812 pinhole JPEGs.
`gravel_model_check.py` checks the calibration convention: it reprojects
the dataset's own tie points at 0.29 px median.

The mapper runs with `--sift-max-keypoints 8192 --verification-mode full
--match-ratio 0.9 --min-matches 15 --min-pnp-inliers 10`, candidate pairs
from `--pair-source spatial` (or `--exhaustive` where marked), and the GNSS
as `--gps-priors gps_priors.txt --gps-priors-format cartesian`.
`scripts/gps_prior/eval_metric.py` scores it.

Camera-centre error against the reference, in metres. "Bias removed" takes
out the best translation, "shape" the best similarity. The priors runs
register 147 of 158 images; the GNSS row is scored on those 147.

| Run | Absolute RMSE | Bias removed RMSE / median / max | Shape RMSE |
|---|---:|---:|---:|
| Onboard GNSS itself | 3.45 | 1.71 / 1.58 / 2.86 | – |
| No priors (exhaustive, 142 images), Sim(3)-aligned to the GNSS afterwards | 3.51 | 1.79 / 1.06 / 13.04 | 1.78 |
| GPS priors, σ 0.5 / 1 m | 3.03 | **0.44** / 0.35 / 1.69 | 0.41 |
| GPS priors, σ 1 / 2 m | 3.08 | 0.71 / 0.45 / 4.28 | 0.69 |
| GPS priors, σ 1 / 2 m, exhaustive pairs | 3.06 | 0.62 / 0.47 / 3.89 | 0.60 |
| GPS priors, σ 1 / 2 m, Cauchy loss | 3.51 | 1.80 / 0.55 / 14.69 | 1.79 |
| GPS priors, σ 1 / 2 m, alignment outliers dropped | 3.14 | 0.86 / 0.55 / 6.48 | 0.84 |
| GPS priors, σ 2 / 4 m | 3.18 | 1.07 / 0.69 / 6.18 | 1.06 |
| GPS priors, σ 5 / 10 m | 3.32 | 1.43 / 0.79 / 7.17 | 1.42 |

- **With priors at a realistic σ the shape is right** to 0.4–0.7 m, 2.4–3.9×
  better than the GNSS's own scatter.
  - The absolute error (about 3 m) is the GNSS's constant bias. Position
    priors cannot see a bias: correcting it needs control points.
  - Without priors, aligning afterwards leaves a camera 13 m off and a
    bias-removed error no better than the GNSS itself.
- **Plain least squares is the right default here.** The Cauchy loss lets
  one camera drift 14.7 m from a fix that was right, and dropping outliers
  is worse at every statistic.
- **This set first exposed a SIFT bug.** The strips alternate heading by
  about 165°, and the built-in SIFT was not rotation invariant (fixed: see
  the CHANGELOG).
  - Before the fix only 53–54 images registered: the westward strips, which
    could not match the eastward ones.
  - The prior-free model of those images was bent by 14 m (40 m at worst).
    With σ 1 / 2 m priors their shape error was 0.63 m.

### Choosing σ

A σ that is too large costs more than one that is too small.
- At σ 0.5–1× the GNSS's real scatter the priors do best.
- At 2–5× they still beat aligning afterwards, but give back most of the
  gain (1.1 m, then 1.4 m). With the buggy SIFT, whose model was bent much
  harder, 5× cost 8 m.

Set σ to the receiver's real per-fix accuracy, or a little below it. The
CLIs default to 1 m on every axis, as COLMAP's `pose_prior_mapper` does.
A constant bias does not bend the model, so it is no reason to raise σ.

Reproduce: download `mmc2.zip` (the text files) from the article's
supplementary material and `Part3-RawImages.zip` (5.9 GB) from its Google
Drive link, then:

```bash
python3 scripts/gps_prior/gravel_prepare.py supp/ Part3-RawImages.zip gravel/ 4
cargo run --release --features image-io --example unordered_sfm_demo -- \
  --feature-extractor sift --images-dir gravel/images --image-suffix .jpg \
  --width 1218 --height 812 --fx 1216.5643 --fy 1216.5643 --cx 609 --cy 406 \
  --exhaustive --sift-max-keypoints 8192 --verification-mode full \
  --match-ratio 0.9 --min-matches 15 --min-pnp-inliers 10 \
  --gps-priors gravel/gps_priors.txt --gps-priors-format cartesian \
  --gps-prior-std 1 --gps-prior-std-vertical 2 --out-colmap out_s1/
python3 scripts/gps_prior/eval_metric.py gravel/truth.csv out_nopri:posthoc out_s1
```

## GPS for pair selection (spatial matching)

With fixes available, candidate pairs can come from them instead of from
image retrieval, as in COLMAP's `spatial_matcher`. Each image with a fix
is paired with up to 50 other fixes within 100 m, by horizontal distance.
Images without a fix get their VLAD top-k pairs.

```bash
unordered_sfm_demo ... --gps-priors-from-exif --pair-source spatial \
  [--spatial-max-neighbors 50] [--spatial-max-distance 100] [--spatial-use-z]
colmap spatial_matcher --database_path db.db \
  [--SpatialMatching.max_num_neighbors 50] [--SpatialMatching.max_distance 100] \
  [--SpatialMatching.ignore_z 1]
```

Gravel-mine set, same flags as above, GPS priors at σ 1 / 2 m:

| Candidate pairs | Candidates | Verified | Verification time | Registered | Bias removed RMSE |
|---|---:|---:|---:|---:|---:|
| Exhaustive | 12,403 | 1,280 | 96.1 s | 147 | 0.62 m |
| **Spatial, 50 neighbours / 100 m (default)** | 4,632 | 1,279 | 40.8 s | 147 | 0.71 m |
| Spatial, 20 neighbours | 1,827 | 1,118 | 23.6 s | 147 | 1.02 m |
| VLAD top-12 (the demo's default) | 1,198 | 810 | 19.2 s | 140 | 0.59 m (on its 140) |

- **Spatial pairs keep what exhaustive matching finds.** With the defaults
  they find 1,279 of the 1,280 pairs that exhaustive matching verifies,
  from 37 % of the candidates, and register the same 147 images.
  - The 0.62 against 0.71 m is run-to-run variation from that one pair
    and the mapper's order, not a loss of matches.
- **20 neighbours** loses 160 verified pairs and leaves a worse model.
- **VLAD retrieval** is the cheapest, but drops 470 of the verified pairs
  and 7 images.
  - With the SIFT before its rotation fix, VLAD registered 41 images where
    spatial pairs registered 54.
- The candidate count grows linearly with the number of images, where
  exhaustive matching grows quadratically, so the saving grows with the
  block.

## Limits

- Incremental mapper only. `unordered_sfm_demo` refuses the GPS flags with
  `--mapper global|hybrid`.
- The COLMAP-port rig mapper (`colmap_incremental`) does not take priors.
- Positions only: no orientation priors (compass, IMU attitude) and no full
  3×3 covariance (per-axis σ only).
- Rolling-shutter or time-offset effects between the GPS fix and the
  exposure are not modelled.
