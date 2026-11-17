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

On ETH3D courtyard (below) both opt-ins are worse than plain least squares,
so neither is the default.

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
  --gps-priors-from-exif --gps-prior-std 5 --gps-prior-std-vertical 10 \
  --out-colmap sparse/
```

| Flag | Meaning |
|---|---|
| `--gps-priors FILE` | priors from a file (instead of EXIF) |
| `--gps-priors-format wgs84\|cartesian` | how the file's three numbers read (default `wgs84`) |
| `--gps-priors-from-exif` | priors from the images' EXIF GPS |
| `--gps-prior-std M`, `--gps-prior-std-vertical M` | σ where a fix states none (default 5 m / 10 m) |
| `--gps-prior-max-error M` | inlier threshold of the alignment (default 3 σ) |
| `--gps-prior-robust-loss`, `--gps-prior-loss-scale C` | Cauchy loss on the priors |
| `--gps-prior-reject-outliers` | drop the alignment's outliers from the BA |

`colmap` CLI shim, mirroring COLMAP's command surface:

```bash
colmap feature_extractor --database_path db.db --image_path images/   # stores EXIF GPS
colmap exhaustive_matcher --database_path db.db
colmap pose_prior_mapper --database_path db.db --output_path out/ \
  --prior_position_std_x 5 --prior_position_std_y 5 --prior_position_std_z 10
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
let (local, enu_frame) = set.to_local_enu(&image_names, None, Vector3::new(5.0, 5.0, 10.0));
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
| No priors, then Sim(3)-aligned to the GPS (the jumps removed by hand) | 23 | 2.52 m | 1.66 m | 6.63 m |
| **GPS priors in the BA (default)** | 23 | **1.66 m** | **0.48 m** | 6.52 m |
| GPS priors, Cauchy loss | 25 | 19.49 m | 0.69 m | 96.17 m |
| GPS priors, alignment outliers dropped | 24 | 6.33 m | 1.14 m | 30.18 m |

- With the priors in the BA, 16 of the 23 cameras are closer to the truth
  than their own GPS fix. The model without priors is bent: its error grows
  along the capture from 1 m to 6 m, even after the best similarity.
  Alignment after mapping cannot undo that bend.
- The 30 m jump on DSC_0289 leaves its camera 0.35 m from the truth: the
  visual terms outweigh one bad fix.
- The remaining large errors (DSC_0294–0296, up to 6.5 m) are cameras the
  800-pixel visual model places badly. The no-prior model has them wrong
  too.
- Both opt-ins lose here, on the same camera. DSC_0309's fix is 0.9 m
  from the truth, but the growing model placed the camera badly and so
  disagreed with it. The Cauchy loss then all but ignores that fix (the
  camera ends 96 m off), and rejection drops it (30 m off). In the default
  run DSC_0309 does not register at all. Registration order changes with
  the priors, so the three runs register different sets (23–25 cameras).

`colmap` shim on the same fixes (images upscaled 2× to the shim's ETH3D
1600-pixel camera, `pose_prior_mapper --prior_position_std_x 0.5
--prior_position_std_y 0.5 --prior_position_std_z 1`): it registered 12/38
(the shim's plain matcher) with RMSE 0.62 m and median 0.59 m against the
ground truth. Those 12 cameras span only a few metres, so 1 m GPS noise
leaves the scale about 25 % off even though the shape is right (0.07 m after
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

## Limits

- Incremental mapper only. `unordered_sfm_demo` refuses the GPS flags with
  `--mapper global|hybrid`.
- The COLMAP-port rig mapper (`colmap_incremental`) does not take priors.
- Positions only: no orientation priors (compass, IMU attitude) and no full
  3×3 covariance (per-axis σ only).
- Rolling-shutter or time-offset effects between the GPS fix and the
  exposure are not modelled.
