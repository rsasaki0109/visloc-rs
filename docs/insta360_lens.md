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

| Walk (camera) | Median disagreement across the parallax | With the back lens's optical centre moved too | The move (3840 px lens image) |
| --- | ---: | ---: | --- |
| 0313afa5 (A) | 0.217° | 0.045° | (+20.5, +5.2) px |
| 0c05c38a (B) | 0.114° | 0.026° | (+5.7, −4.7) px |
| 3b81dbc3 (B) | 0.111° | 0.030° | (+4.2, −4.5) px |
| 9d2fe9b1 (C) | 0.148° | 0.029° | (+14.8, −3.2) px |
| 5bb3848a (D) | 0.145° | 0.022° | (+17.3, +1.9) px |

The back lens's optical centre sits 5–21 px from where the model puts it
relative to the front lens's (the same for both walks of camera B). The
calibration predicts that offset to within 3.5 px if the video held each
lens's calibration image upside down and centred on the middle of its
5376 px square rather than on `(cx, cy)`. The matches fix only the
relative offset, though, and moving each lens's centre to where that
reading puts it (2–15 px from the middle) makes SfM worse: on walk
9d2fe9b1 the metric scale goes from +1.7% to +11.9% of the dataset's
(640 px faces), with |g| 9.56. With a scale of 1.386 calibration px per
video px for every camera as well, it goes from +1.8% to +13.1% at
1024 px, and the other four walks move 0.8–2.0 points further from it. So
`gsplat_photos --insv` keeps both centres in the middle unless it calibrates
them from the video itself ([below](#calibrating-both-lenses)), and
`insv_stitch` moves the back lens's by the offset it measures along the seam
([below](#stitching)). The black edge of check 1 shows where the camera
masks the image, which need not be the optics.

## The lenses on the camera

The three angles per lens describe how the lens sits on the camera. Only
the third is decoded: the lens's turn about its own axis, 90° plus up to
a degree and a half. The back lens's turn about the axis, beyond the half
turn, is the sum of the two third angles minus 180°:

| Camera (walk) | Recorded angles | SfM rig (`gsplat_photos --insv`, 1024 px) | Along the seam (`insv_stitch`) |
| --- | ---: | ---: | ---: |
| A (0313afa5) | +0.94° | +0.87° | +0.92° |
| B (0c05c38a) | −1.63° | −1.64° | −1.61° |
| B (3b81dbc3) | −1.63° | −1.63° | −1.62° |
| C (9d2fe9b1) | −1.52° | −1.43° | −1.47° |
| D (5bb3848a) | +0.70° | +0.64° | +0.68° |

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

## Calibrating both lenses

```text
gsplat_photos --insv VID.insv --out run --insv-calibrate
```

`--insv-calibrate` reconstructs twice. The first run goes into
`run/calibration`. Each lens's optical centre and focal lengths are
calibrated from it and written to `run/lenses.txt`, and the second run
cuts the faces from the calibrated lenses. `--insv-lenses lenses.txt` and
`insv_stitch --lenses lenses.txt` reuse a calibration.

- **The fit** (`photos::calibrate_insv_lenses`,
  `dual_fisheye::calibrate_lenses`). Every observation in the first model
  goes back to its lens image, through the model its face was cut with. A
  bundle adjustment in those pixels, through the unified model, then
  refines per lens an optical-centre offset and a focal-length factor,
  with the back lens's pose on the camera, the frames (at most 200) and the
  points (at most 20,000). It is robust (soft-L1 at 1 px), with the
  points eliminated, and takes about a minute.
- **The seam pins it.** The faces show each lens only within 90° of its
  axis. There, both centres moving together looks almost like a turn of
  the whole camera, which the frames absorb. Without more, which frames
  are used moved the front lens's centre by up to 12 px on walk 5bb3848a
  (5.7–17.8 px, 100 to 199 of its frames), at almost the same cost. So SIFT
  matches along the seam in 16 of the frames join the fit, each a point
  both lenses of a frame see. With them, 120 or 200 frames agree within
  2.3 px on two walks; on walk 9d2fe9b1 the back lens's centre moves by
  4.6 px.
- **The seam agrees.** On walk 0313afa5, `insv_stitch --lenses` then finds
  the back lens 2 px from where the calibration put it, against 20.5 px
  without it.

The calibration on each walk (200 frames, 640 px faces; offsets in a 3840
px lens image):

| Walk (camera) | Front lens's centre | Back lens's centre | Focal lengths |
| --- | --- | --- | --- |
| 0313afa5 (A) | (+9.2, −5.4) px | (+10.0, −0.6) px | x0.9964, x0.9942 |
| 9d2fe9b1 (C) | (+7.9, −4.8) px | (+3.5, −7.0) px | x1.0019, x0.9991 |
| 5bb3848a (D) | (+8.1, +1.3) px | (+8.3, +3.6) px | x0.9955, x0.9968 |

What the second run gives, against the first. Track shape is the median
distance from the dataset's camera track after a similarity fit; the scale
is against the dataset's IMU scale.

| Walk | Metric scale, first → second | Track shape, first → second | Reprojection (median) |
| --- | ---: | ---: | ---: |
| 0313afa5 | +1.0% → +0.1% | 0.148 → 0.109 m | 0.422 → 0.406 px |
| 9d2fe9b1 | PARK_SCALE | PARK_TRACK | PARK_REPROJ |
| 5bb3848a | +12.0% → +8.1% | 0.343 → 0.085 m | 0.475 → 0.469 px |

The tracks come closer to the dataset's on every walk. The scale moves by
up to PARK_SCALE_MOVE points, in either direction. The calibration takes
one more reconstruction, so `--insv-calibrate` is not the default.

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
  `--lenses lenses.txt` starts it from a calibration of both lenses
  ([above](#calibrating-both-lenses)).
- **Distance.** Where the lenses meet, things at different distances
  cannot all line up on one sphere: at 2 m they are 0.9° apart. The
  shared shift gives the scene's typical distance along the seam, and the
  back lens samples where it sees a sphere that far away (`--distance M`
  sets one instead).
- **Seam.** Each direction takes the lens whose axis is nearer. Over a
  band 6° wide where they meet (`--blend`), the two lenses of each frame
  are matched and one morphed into the other, so what is nearer or
  farther than that sphere shows once ([below](#the-morph)). `--seam
  blend` only mixes them, over 2°. There is no exposure matching.

How far apart the two lenses put the same thing along the seam (the
median over all matches in 40 frames per walk; the fit used 20 of them):

| Walk | Recorded angles, far away | Fitted: rotation, centre, shift | Shift along the parallax |
| --- | ---: | ---: | --- |
| 0313afa5 | 0.89° | 0.17° | −0.18° |
| 0c05c38a | 0.85° | 0.17° | +0.74° (2.4 m) |
| 3b81dbc3 | 0.90° | 0.18° | +1.03° (1.8 m) |
| 9d2fe9b1 | 0.89° | 0.12° | +0.58° (3.0 m) |
| 5bb3848a | 1.15° | 0.15° | −0.17° |

0.17° is about 3 px in a 6,496 px panorama. The rotation alone, without
the centre, leaves 0.33–0.93°. On two walks the shared shift points away
from the axes, which no distance explains: the lens models disagree a
little on the scale at the seam there, and the stitch takes that shift all
the same. On 4 CPU cores the fit takes 1.5–3 minutes for its 16 frames,
and stitching a frame about 1 s, 1.8 s with the morph.

### The morph

The lenses sit 3.2 cm apart along their axes, so a thing nearer or
farther than the stitching sphere moves only towards or away from the
axes between their views. Around the baseline the seam is a rectified
stereo pair (`dual_fisheye::SeamDisparity`):

- **Strips.** Both lenses are rendered 7° either side of the seam, at half
  their axial resolution (9 samples per degree on the X5), each row a
  circle about the baseline.
- **Matching.** 7 x 7 census costs, aggregated along four paths
  (semi-global matching), for disparities up to 3.5° (half a metre
  away), refined to a fraction of a sample; then a 5 x 5 median and a
  light blur. Where a lens does not see, the neighbours fill in.
- **Morph.** Across the band each lens samples part of the way towards
  where the other sees the same thing, the front lens's share falling from
  1 to 0, so a thing slides from where the front lens sees it to where the
  back lens does.

How far apart the lenses put the same thing within 5° of the seam, in the
20 frames per walk the fit did not use (SIFT matches, median / 90th
percentile; OpenCV's DIS optical flow on the same strips for reference):

| Walk | Matches | One distance | Measured disparity | DIS |
| --- | ---: | ---: | ---: | ---: |
| 0313afa5 | 4,640 | 0.175° / 0.42° | 0.038° / 0.20° | 0.040° / 0.17° |
| 0c05c38a | 7,027 | 0.147° / 0.85° | 0.034° / 0.14° | 0.036° / 0.13° |
| 3b81dbc3 | 4,199 | 0.189° / 0.60° | 0.035° / 0.17° | 0.036° / 0.14° |
| 9d2fe9b1 | 7,724 | 0.108° / 0.35° | 0.034° / 0.13° | 0.041° / 0.13° |
| 5bb3848a | 3,475 | 0.119° / 0.39° | 0.033° / 0.15° | 0.037° / 0.16° |

- **Seen.** 0.035° is about 0.6 px of the panorama. On a painted ceiling
  about 2 m away (0c05c38a) the plain blend shows its patterns twice,
  the morph once.
- **Video.** Over 30 consecutive frames of the park walk, the panorama
  changes from frame to frame 1.1% more in the band than beside it with
  the morph, 2.0% more with the plain blend: the morph adds no flicker.
- **Exposure.** In the overlap the two lenses' brightness agrees within
  1.7% per colour (front over back 0.986–1.017, the median over the
  overlap per walk). Across the overlap the ratio drifts by up to 1.7% per
  degree, each lens darker towards its edge, which the band spreads out.
- **Test.** A synthetic world 1 m away above the horizon and 4 m below
  it: the measured disparity is within 0.09° of the true one (median),
  and the band twice as close to a single view of the world as with any
  one distance.
