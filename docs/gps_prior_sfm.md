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
a bad registration), and that prior is what pulls it back.

Bad fixes (a GPS jump) are handled by a Cauchy loss on the normalised
residual: COLMAP's `use_robust_loss_on_prior_position`, scale
`sqrt(χ²₀.₉₅(3)) ≈ 2.8`.
- It is applied by reweighting each BA's prior at its current residual, so
  it works as IRLS across the mapper's BA calls.
- **It is on by default, unlike COLMAP.** With COLMAP-compatible features:
  - on real onboard GNSS (the gravel-mine set below) it equals least
    squares;
  - on simulated GPS with 30 m jumps (ETH3D courtyard) it keeps the model
    at 0.30 m where least squares leaves 1.21 m.
- `--gps-prior-least-squares` (`--use_robust_loss_on_prior_position 0` in
  the shim) turns it off.
- With the legacy SIFT, whose models are much weaker, least squares was the
  better choice: a weak model needs the full pull of every fix.

Dropping the alignment's outliers from the BA instead is a further opt-in.
It was never the best in these measurements.

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

The demo's built-in SIFT is the COLMAP-compatible one by default
(`--sift-legacy` for the old extractor, whose integer-pixel keypoints give
far weaker models: see the measurements below).

| Flag | Meaning |
|---|---|
| `--gps-priors FILE` | priors from a file (instead of EXIF) |
| `--gps-priors-format wgs84\|cartesian` | how the file's three numbers read (default `wgs84`) |
| `--gps-priors-from-exif` | priors from the images' EXIF GPS |
| `--gps-prior-std M`, `--gps-prior-std-vertical M` | σ where a fix states none (default 1 m each, as COLMAP; see "Choosing σ") |
| `--gps-prior-max-error M` | inlier threshold of the alignment (default 3 σ) |
| `--gps-prior-loss-scale C` | scale of the Cauchy loss on the priors (default 2.8) |
| `--gps-prior-least-squares` | plain least squares instead of the Cauchy loss |
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
| No priors, then Sim(3)-aligned to the GPS (the jumps removed by hand) | 38 | **0.26 m** | 0.23 m | 0.61 m |
| **GPS priors in the BA (default: Cauchy loss)** | 38 | 0.30 m | **0.22 m** | 0.74 m |
| GPS priors, least squares | 36 | 1.21 m | 0.92 m | 3.13 m |
| GPS priors, alignment outliers dropped | 37 | 0.86 m | 0.55 m | 3.19 m |
| GPS priors, least squares, `--sift-legacy` (the old defaults) | 25 | 3.15 m | 0.52 m | 8.65 m |

- **The visual model is very good.** Without priors its shape error after
  the best similarity is 0.088 m (2.15 m with the legacy SIFT). So
  aligning it afterwards to the GPS, with the two jumps removed by hand,
  is hard to beat.
- **The Cauchy default gets within 4 cm of that without being told about
  the jumps.** DSC_0289 and DSC_0301, whose fixes are 30 m off, end 0.27 m
  and 0.21 m from the truth.
- **Least squares lets the two jumps bend the model** (1.21 m) and loses
  two cameras. Dropping the alignment's outliers does better, but not well.
- **With the legacy SIFT the model itself was the problem**: 25 of 38
  cameras and a 7–9 m misplaced cluster.

`colmap` shim on the same fixes (images upscaled 2× to the shim's ETH3D
1600-pixel camera, `pose_prior_mapper --prior_position_std_x 0.5
--prior_position_std_y 0.5 --prior_position_std_z 1`):
- It registers 37 of 38, against 11–12 with the legacy SIFT.
- Against the ground truth that is RMSE 1.48 m, median 0.78 m.
- The shim's simpler mapper and fixed camera are the limit.
- `spatial_matcher` proposes all 703 pairs here (the scene is smaller than
  100 m), so it gives the same model.

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

The mapper runs with the default (COLMAP-compatible) SIFT and
`--sift-max-keypoints 8192 --verification-mode full --min-matches 15
--min-pnp-inliers 10`. Candidate pairs come from `--pair-source spatial`
(or `--exhaustive` where marked), and the GNSS from `--gps-priors
gps_priors.txt --gps-priors-format cartesian` at σ 1 / 2 m unless noted.
`scripts/gps_prior/eval_metric.py` scores the result.

Camera-centre error against the reference, in metres. "Bias removed" takes
out the best translation, "shape" the best similarity. Every run registers
all 158 images.

| Run | Absolute RMSE | Bias removed RMSE / median / max | Shape RMSE |
|---|---:|---:|---:|
| Onboard GNSS itself | 3.44 | 1.72 / 1.60 / 2.88 | – |
| No priors (exhaustive), Sim(3)-aligned to the GNSS afterwards | 3.02 | 0.50 / 0.38 / 1.38 | 0.47 |
| **GPS priors (default: Cauchy loss)** | 2.98 | **0.18** / 0.14 / 0.36 | **0.025** |
| GPS priors, least squares | 2.98 | 0.18 / 0.15 / 0.35 | 0.025 |
| GPS priors, σ 5 / 10 m | 2.98 | 0.18 / 0.15 / 0.35 | 0.025 |

- **The priors make the block's shape right to 2.5 cm.** Without them it is
  0.47 m off, as monocular nadir blocks bend.
- **The camera positions land 10× closer than the GNSS's own scatter**
  (0.18 m against 1.72 m, bias removed).
- **The absolute error (about 3 m) is the GNSS's constant bias.** Position
  priors cannot see a bias; correcting it needs control points.
- **The loss and σ hardly matter here.** The receiver has no jumps, and the
  visual model is strong enough that a loose σ still straightens it.

**With the legacy SIFT** (`--sift-legacy`, and `--match-ratio 0.9`) the same
block behaves very differently.
- The block exposed its rotation bug. The strips alternate heading by about
  165°, and before the fix only the 53–54 images of the westward strips
  registered; their prior-free model was bent by 14 m.
- After the fix 147 images register, and the GPS priors reach a shape error
  of 0.41–0.69 m (σ 0.5 / 1 m to 1 / 2 m).
- That weaker model makes σ and the loss matter:

| Legacy SIFT, σ | Least squares | Cauchy | Outliers dropped |
|---|---:|---:|---:|
| 0.5 / 1 m | 0.44 | | |
| 1 / 2 m | 0.71 | 1.80 | 0.86 |
| 2 / 4 m | 1.07 | | |
| 5 / 10 m | 1.43 | | |

(bias-removed RMSE in metres; no priors gives 1.79 m)

**Matcher ratio.** Keep the default 0.8 with the COLMAP-compatible SIFT. At
`--match-ratio 0.9` this repetitive gravel texture passed false matches:
9,232 of 12,403 exhaustive pairs "verified", and the prior-free model came
out 50 m off.

### Choosing σ

Set σ to the receiver's real per-fix accuracy, or a little below it. The
CLIs default to 1 m on every axis, as COLMAP's `pose_prior_mapper` does.
- With a strong visual model σ barely matters.
- With a weak one, a σ well above the real scatter gives back most of the
  priors' gain (legacy-SIFT table above), while a smaller σ does not hurt.
- A constant bias does not bend the model, so it is no reason to raise σ.

Reproduce: download `mmc2.zip` (the text files) from the article's
supplementary material and `Part3-RawImages.zip` (5.9 GB) from its Google
Drive link, then:

```bash
python3 scripts/gps_prior/gravel_prepare.py supp/ Part3-RawImages.zip gravel/ 4
cargo run --release --features image-io --example unordered_sfm_demo -- \
  --feature-extractor sift --images-dir gravel/images --image-suffix .jpg \
  --width 1218 --height 812 --fx 1216.5643 --fy 1216.5643 --cx 609 --cy 406 \
  --pair-source spatial --sift-max-keypoints 8192 --verification-mode full \
  --min-matches 15 --min-pnp-inliers 10 \
  --gps-priors gravel/gps_priors.txt --gps-priors-format cartesian \
  --gps-prior-std 1 --gps-prior-std-vertical 2 --out-colmap out_s1/
python3 scripts/gps_prior/eval_metric.py gravel/truth.csv out_s1
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

Gravel-mine set, same flags as above (default SIFT), GPS priors at σ 1 / 2 m:

| Candidate pairs | Candidates | Verified | Verification time | Registered | Bias removed RMSE |
|---|---:|---:|---:|---:|---:|
| Exhaustive (no priors) | 12,403 | 2,781 | – | 158 | – |
| **Spatial, 50 neighbours / 100 m (default)** | 4,632 | 2,748 | 386 s | 158 | 0.18 m |
| VLAD top-12 (the demo's default) | 1,192 | 1,005 | 121 s | 158 | 0.18 m |

- **Spatial pairs keep what exhaustive matching finds.** They recover
  2,748 of the 2,781 pairs it verifies, from 37 % of its candidates.
- **On this block VLAD retrieval is as good and cheaper.** It verifies far
  fewer pairs, but enough: the same 158 images and the same accuracy.
  - With the legacy SIFT it was not (140 images against 147, and 41
    against 54 before the rotation fix).
  - Spatial pairs are the choice when retrieval is unreliable: weak or
    repetitive texture, or images that look alike far apart.
- The exhaustive run's time is left out: it ran alongside other jobs.
  Candidates are a fair proxy for matching cost, and they grow linearly
  with the number of images for spatial pairs, quadratically for
  exhaustive matching.

## Limits

- Incremental mapper only. `unordered_sfm_demo` refuses the GPS flags with
  `--mapper global|hybrid`.
- The COLMAP-port rig mapper (`colmap_incremental`) does not take priors.
- Positions only: no orientation priors (compass, IMU attitude) and no full
  3×3 covariance (per-axis σ only).
- Rolling-shutter or time-offset effects between the GPS fix and the
  exposure are not modelled.
