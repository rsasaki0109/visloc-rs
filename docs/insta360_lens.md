# Insta360 lens calibration from the `.insv` (X5)

`gsplat_photos --insv` reconstructs an Insta360 video from its raw
dual-fisheye frames, and `insv_stitch` stitches them into equirectangular
(360°) frames, both with no Insta360 Studio export in between
([photos_to_splat.md](photos_to_splat.md#360-cameras),
[below](#stitching)). Both need the factory calibration of each lens,
which the camera writes into the `.insv`. This page records how that
calibration was read and checked, on Insta360 X5 videos (four cameras,
firmware v1.10.11 and v1.11.6) from
[OverMaps360_50](https://huggingface.co/datasets/OverTheReality/OverMaps360_50)
(walks 0313afa5, 0c05c38a, 3b81dbc3, 9d2fe9b1 and 5bb3848a; CC BY-NC 4.0;
nothing from it is in the repository). Code: `visloc_io::insv` (parsing)
and `visloc_gsplat_train::dual_fisheye` (the model, the faces, the
stitching).

## Where it is

The `.insv` is an MP4 with a trailer appended (layout in
[insta360_imu_scale.md](insta360_imu_scale.md)). The trailer's metadata
record (id `0x101`) holds the camera model, the firmware and several
calibration strings, numbers joined by `_`:

- **`p2_` string** (`n2_` on some cameras), per lens: a radius, `cx`,
  `cy` and three angles; then
  the calibration image's size (10752 x 5376, the two lenses side by side,
  5376 px square each).
- **Unified-model string**: `2_`, then per lens 19 fields:
  `xi fx fy cx cy`, three angles (degrees), a translation (metres),
  `k1 k2 k3 p1 p2`, and the calibration image's width, height and a
  constant (113); then one trailing field. The back lens's `cx` is in the
  side-by-side image (about 8079 = 5376 + 2703), and its translation is
  about 3.2 cm along the optical axis.
- Two more strings (`2_<radius>_…` with a four-term polynomial, and a
  unified model with 13 distortion terms) are not used.

`visloc_io::insv::read_insv_telemetry` returns the unified model of both
lenses with the `p2_` radius as `InsvLens`. Other cameras or firmware may
lay these out differently; `gsplat_photos --insv` then stops with an error.

## The model

A ray `(x, y, z)` in the lens frame (x right, y down, z along the optical
axis) goes to `(x, y) / (z + xi |ray|)` (the unified, or Mei, model;
`xi = 2` on the X5), then through radial distortion
`1 + k1 r² + k2 r⁴ + k3 r⁶` and OpenCV's tangential `p1, p2`, then to
pixels with `fx, fy, cx, cy`. These are OpenCV's `omnidir` conventions.

**From calibration pixels to the video's pixels.** The video has one 3840 px
square stream per lens. It is not the 5376 px calibration image scaled by
5376 / 3840 = 1.4. It is the square centred on the lens's `(cx, cy)` that
spans the 200° image circle edge to edge:
`pixel = (calibration - (cx, cy)) / s + 1919.5`, with
`s = 2 radius / 3840 = 1.3784`. Four checks support this:

1. **Centred on the calibrated centre.** The frames' black circle edge (the
   camera blacks out past about 98°) is centred within 0.4 px of the image
   centre in a frame of each lens (1919.4, 1919.1 and 1919.5, 1919.2). The
   calibrated centre is 11 px from the 5376 px image's middle.
2. **The radius is 100°.** Under each lens's unified model, the `p2_` radius
   (2646.5 and 2647.8 px) lies at 99.95° off the axis: the 200° circle.
3. **The scale, fitted.** OverMaps360 ships perspective views cut from the
   same frames (1600 px, f = 800, five per lens image). Fitting the model
   with a free scale and centre to SIFT matches between a frame and a view
   near the back lens's axis (14–42° off it) gives s = 1.37874 against the
   predicted 1.37906, centre (1919.8, 1918.0), 0.29 px median residual: the
   factory model at the predicted scale fits to SIFT noise. Views further
   out (to 79°) give s = 1.3765–1.3815 at about 1 px, where the dataset's
   views follow a slightly different model. The rotations fitted for two
   views of one lens agree with the dataset's rig to 0.10°.
4. **The two lenses meet.** Rays through the lenses' overlap (90–100° off
   each axis, 1,753 matches in six frames) agree best near s = 1.38–1.39,
   and are clearly worse at 1.40 or 1.36. The lenses are 3.2 cm apart, so
   nearby ground points shift this estimate upwards.

An SfM control does not separate the two scales by reprojection error.
`gsplat_photos --insv` ran on 200 frames of walk 0313afa5 (see
[photos_to_splat.md](photos_to_splat.md#metric-scale-from-the-cameras-imu)).
The control ran on the same frames, each lens image magnified by
1.4 / 1.3784 first, which emulates s = 1.4.

| | s = 1.3784 | s = 1.4 (control) |
| --- | ---: | ---: |
| Mean reprojection error | 0.584 px | 0.582 px |
| Back lens behind the front one (refined) | 3.22 cm | 14.4 cm |
| Back lens past the half turn | 1.17° | 1.23° |
| m per dataset unit (dataset: 1.0401) | 1.0504 (+1.0%) | 1.0368 (−0.3%) |
| Median distance to the dataset's track | 15 cm | 9.6 cm |

- **Reprojection error.** Every face's pose on the rig is refined, so the
  faces turn and absorb most of the scale error.
- **The back lens.** On this walk it settles where the factory calibration
  puts it (3.23 cm) at s = 1.3784, and far off at s = 1.4. On four other
  walks it lands 2.1–6.8 cm behind at s = 1.3784, though, so the offset is
  only weakly determined. It is a hint, not a proof; the image fits above
  are the evidence.
- **The dataset's track.** Over these 133 s both runs drift from the
  dataset's track by more than the two differ, so it does not decide
  between them.

**The lenses' centres relative to each other.** Where the lenses see
the same things (90 ± 8° off their axes), the back lens sits 3.2 cm behind
the front one, so a point near the camera shows it a direction shifted
towards the lenses' axes; across that direction the two lenses must agree
at any distance. On SIFT matches there (40 frames 3.3 s apart per walk),
with the back lens's rotation fitted, they do not quite:

| Walk (camera) | Median disagreement across the parallax | With the back lens's optical centre moved too | The move |
| --- | ---: | ---: | --- |
| 0313afa5 (A) | 0.217° | TBD | TBD |
| 0c05c38a (B) | 0.114° | TBD | TBD |
| 3b81dbc3 (B) | 0.111° | TBD | TBD |
| 9d2fe9b1 (C) | 0.148° | TBD | TBD |
| 5bb3848a (D) | 0.145° | TBD | TBD |

The two optical centres are 5–21 px apart from where the model puts them
relative to each other. The calibration predicts that offset to within
3 px if the video held each lens's calibration image upside down and
centred on the middle of its 5376 px square rather than on `(cx, cy)`.
The matches fix only the relative offset, though, and moving each lens's
centre to where that reading puts it (2–15 px from the middle) makes SfM
worse: on walk 9d2fe9b1 the metric scale goes from +1.7% to +12.5% of the
dataset's (640 px faces; +1.8% to +13.1% at 1024 px), with |g| 9.6, and
the other four walks move 0.8–2.0 points further from it at 1024 px. So
`gsplat_photos --insv` keeps both centres in the middle, and `insv_stitch`
moves the back lens's by the offset it measures along the seam
([below](#stitching)). The black edge of check 1 shows where the camera
masks the image, which need not be the optics.

## The lenses on the camera

The three angles per lens describe how the lens sits on the camera. Only
the third is decoded: the lens's turn about its own axis, 90° plus up to
a degree and a half. The back lens's turn about the axis, beyond the half
turn, is the sum of the two third angles minus 180°:

| Camera (walk) | Recorded angles | SfM rig (`gsplat_photos --insv`, 1024 px) | Along the seam (`insv_stitch`) |
| --- | ---: | ---: | ---: |
| A (0313afa5) | +0.94° | +0.87° | TBD |
| B (0c05c38a) | −1.63° | −1.64° | TBD |
| B (3b81dbc3) | −1.63° | −1.63° | TBD |
| C (9d2fe9b1) | −1.52° | −1.43° | TBD |
| D (5bb3848a) | +0.70° | +0.64° | TBD |

The first two angles are not decoded: the reading of them as Euler angles
that fits the four cameras' SfM rigs best still misses by 0.5° (RMS). The back
lens's tilt (its turn about the other two axes) reaches 1.1° in the rigs,
more than camera D's first two angles could give together (0.6°).
Without the tilt the seam shows things twice, a degree apart.

- **`gsplat_photos --insv`** solves a frame's ten faces as one rig in the
  front lens's frame, the back lens's pose on it starting from the half
  turn with no offset, and refined by the mapper's bundle adjustments
  (COLMAP's `ba_refine_sensor_from_rig`). On the five walks it settles
  1.2–1.7° past the half turn, 2.1–6.8 cm behind the front lens (the
  calibration says 3.2 cm: the offset is only weakly determined); the front
  lens's faces, refined too, move by at most 0.034° and 5 mm. It writes the
  refined pose to `<out>/back_lens.txt`.
- **`insv_stitch`** fits the back lens along the seam ([below](#stitching)),
  or takes `back_lens.txt` or the recorded angles alone.

## Stitching

```text
insv_stitch VID.insv --out pano            # pano/NNNNNN.jpg, every frame
insv_stitch VID.insv --video pano.mp4      # one equirectangular video (ffmpeg)
```

`cargo run --release -p visloc-gsplat-train --features euroc --example
insv_stitch -- …`; ffmpeg on `PATH` decodes the frames, both lenses side by
side, and encodes `--video`. `--step N` takes every N-th frame, `--frame`
stitches images of the two lenses side by side instead.

- **Panorama.** Equirectangular, the front lens's axis at the centre
  column (the convention `gsplat_photos --equirect` reads), 6,496 x 3,248
  by default: the lenses' resolution on their axes.
- **The back lens.** Before stitching, `insv_stitch` cuts eight views
  across the seam from both lenses in 16 frames spread over the video,
  matches them with SIFT, and fits the back lens's rotation and optical
  centre to the matches: across the parallax exactly, along it up to one
  shift all matches share, robustly (`dual_fisheye::fit_back_lens`).
- **Distance.** Where the lenses meet, things at different distances
  cannot all line up: at 2 m they are 0.9° apart. The shared shift gives
  the scene's typical distance along the seam, and the back lens samples
  where it sees a sphere that far away (`--distance` sets it).
- **Seam.** Each direction takes the lens whose axis is nearer, the two
  mixed linearly over `--blend` (2°) where they meet. There is no exposure
  matching or optical flow.

How far apart the two lenses put the same thing along the seam (the
median over all matches in 40 frames per walk; the fit used 20 of them):

TBD-TABLE
