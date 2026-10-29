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
(walks 0313afa5, 0c05c38a, 3b81dbc3, 5bb3848a and 9d2fe9b1; CC BY-NC 4.0;
nothing from it is in the repository). Code: `visloc_io::insv` (parsing)
and `visloc_gsplat_train::dual_fisheye` (the model, the faces, the
stitching).

## Where it is

The `.insv` is an MP4 with a trailer appended (layout in
[insta360_imu_scale.md](insta360_imu_scale.md)). The trailer's metadata
record (id `0x101`) holds the camera model, the firmware and several
calibration strings, numbers joined by `_`:

- **`p2_` string** (`n2_` on some cameras), per lens: a radius, `cx`, `cy`
  and three angles; then the calibration image's size (10752 x 5376, the
  two lenses side by side, 5376 px square each).
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

**From calibration pixels to the video's pixels.** The video has one
3840 px square stream per lens. It holds the calibration's lens image
upside down, around the middle of the lens's 5376 px calibration square,
at 1.386 calibration pixels per video pixel: video pixel `(u, v)` is
calibration pixel

```text
(C + 1.386 (u - 1919.5),  2687.5 - 1.386 (v - 1919.5))
```

with `C = 2687.5` for the front lens and 8063.5 for the back one (the
right half of the side-by-side image). In the video's own frame (x right,
y down) that is the same model with `fx, fy` divided by 1.386, `p1`'s sign
flipped, and the optical centre at
`(1919.5 + (cx - C) / 1.386, 1919.5 - (cy - 2687.5) / 1.386)`: 2–15 px
from the middle of the image on these cameras, not at it
(`FisheyeLens::from_insv`).

**Measured where the two lenses see the same things.** The lenses see
90 ± 8° off their axes twice. SIFT matches there (8 to 19 thousand per
walk, 40 frames 3.3 s apart) give two checks that do not depend on the
scene:

1. **Across the parallax.** The back lens sits 3.2 cm behind the front
   one, so a point near the camera shows it a direction shifted towards
   the lenses' axes, and across that direction the two lenses must agree
   at any distance. With the back lens's rotation fitted, the median
   disagreement across it:

   | Walk (camera) | Optical centre in the middle, scaled by the image circle (before) | This model |
   | --- | ---: | ---: |
   | 0313afa5 (A) | 0.217° | 0.044° |
   | 0c05c38a (B) | 0.114° | 0.025° |
   | 3b81dbc3 (B) | 0.111° | 0.031° |
   | 9d2fe9b1 (C) | 0.148° | 0.030° |
   | 5bb3848a (D) | 0.145° | 0.046° |

   Fitted freely instead, the offset between the two lenses' optical
   centres lands within 1–3 px of this model's on every walk, its
   vertical part only with the image upside down (and `p1` flipped).
2. **The scale.** Along the shift, parallax adds to any scale error, but
   it can only push the back lens's view towards the axes, and by the
   baseline over the distance. With 1.386 for every camera, the shift the
   matches share puts the scene along the seam 1.8–2.7 m away on all five
   walks (the seam runs through the ground beside and below the camera).
   With the scale the image circles' radii would give (`2 r / 3840`,
   1.378–1.389 per lens), two walks would need the shift to point the
   other way, as if the scene were beyond infinity. The scale is the
   least certain part: a few thousandths either way move the seam's
   distance, which the stitcher fits anyway, and barely change SfM.

Three earlier checks seemed to put the optical centre in the middle, with
the image circle edge to edge (`2 r / 3840`), and do not:

- **The black edge.** The camera blacks out past about 98° in a circle
  centred in the video's square: centred on the square, not on the optics.
- **OverMaps360's perspective views** fit that model to SIFT noise
  (0.29 px): they were cut with it, so the fit recovers how they were cut.
- **SfM** barely tells the two apart. A shift of both lenses' centres is
  close to a turn of the whole camera, and the refined back lens absorbs
  most of the rest: on walks 0313afa5 and 5bb3848a the mean reprojection
  error is TBD before and TBD after (640 px faces).

## The lenses on the camera

The three angles per lens describe how the lens sits on the camera. Only
the third is decoded: the lens's turn about its own axis, 90° plus up to
a degree and a half. The back lens's turn about the axis, beyond the half
turn, is the sum of the two third angles minus 180°:

| Camera (walk) | Recorded angles | Along the seam (`insv_stitch`) | SfM rig (`gsplat_photos --insv`, 1024 px) |
| --- | ---: | ---: | ---: |
| A (0313afa5) | +0.94° | +0.91° | TBD |
| B (0c05c38a) | −1.63° | −1.60° | TBD |
| B (3b81dbc3) | −1.63° | −1.60° | TBD |
| C (9d2fe9b1) | −1.52° | −1.47° | TBD |
| D (5bb3848a) | +0.70° | +0.68° | TBD |

The first two angles are not decoded. The back lens's tilt, its turn
about the other two axes, is 0.6–2.0° on these cameras, more than any
reading of the angles allows (at most 0.6° on any of them); without it the
seam shows things twice, a degree apart. So:

- `gsplat_photos --insv` solves a frame's ten faces as one rig with the
  back lens's pose refined (COLMAP's `ba_refine_sensor_from_rig`),
  starting from the plain half turn. It writes the refined pose to
  `<out>/back_lens.txt`.
- `insv_stitch` fits it along the seam ([below](#stitching)), or takes
  `back_lens.txt` or the recorded angles alone.

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
  column (the convention `gsplat_photos --equirect` reads), 6,464 x 3,232
  by default: the lenses' resolution on their axes.
- **The back lens.** Before stitching, `insv_stitch` cuts eight views
  across the seam from both lenses in 24 frames spread over the video,
  matches them with SIFT, and fits the back lens's rotation to the
  matches: across the parallax exactly, along it up to one shift all
  matches share, robustly (`dual_fisheye::fit_back_lens_rotation`).
- **Distance.** Where the lenses meet, things at different distances
  cannot all line up: at 2 m they are 0.9° apart. The shared shift gives
  the scene's typical distance along the seam, and the back lens samples
  where it sees a sphere that far away (`--distance` sets it).
- **Seam.** Each direction takes the lens whose axis is nearer, the two
  mixed linearly over `--blend` (2°) where they meet. There is no exposure
  matching or optical flow.

How far apart the two lenses put the same thing along the seam (the
median over all matches in 40 frames per walk; the fit used 20 of them):

| Walk | Recorded angles, far away | Recorded angles, at the fit's distance | Fitted, far away | Fitted, at its distance |
| --- | ---: | ---: | ---: | ---: |
| 0313afa5 | 1.06° | 1.10° | 0.81° | 0.19° (2.2 m) |
| 0c05c38a | 1.07° | 0.51° | 0.82° | 0.20° (2.2 m) |
| 3b81dbc3 | 1.01° | 0.48° | 1.09° | 0.19° (1.8 m) |
| 9d2fe9b1 | 1.07° | 0.39° | 0.78° | 0.13° (2.3 m) |
| 5bb3848a | 1.03° | 1.23° | 0.69° | 0.17° (2.7 m) |

0.2° is about 3.6 px in a 6,464 px panorama. Both the rotation and the
distance are needed: the recorded angles miss the tilt, and stitched far
away the ground beside the camera, 2 m off, doubles.

On 4 CPU cores, the fit takes TBD for its 16 frames and stitching a frame
about 2.5 s.
