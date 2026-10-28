# Insta360 lens calibration from the `.insv` (X5)

`gsplat_photos --insv` reconstructs an Insta360 video from its raw
dual-fisheye frames, with no Insta360 Studio export in between
([photos_to_splat.md](photos_to_splat.md#360-cameras)). It needs the
factory calibration of each lens, which the camera writes into the
`.insv`. This page records how that calibration was read and checked, on
an Insta360 X5 (firmware v1.11.6) video from
[OverMaps360_50](https://huggingface.co/datasets/OverTheReality/OverMaps360_50)
(walk 0313afa5, CC BY-NC 4.0; nothing from it is in the repository).
Code: `visloc_io::insv` (parsing) and `visloc_gsplat_train::dual_fisheye`
(the model and the faces).

## Where it is

The `.insv` is an MP4 with a trailer appended (layout in
[insta360_imu_scale.md](insta360_imu_scale.md)). The trailer's metadata
record (id `0x101`) holds the camera model, the firmware and several
calibration strings, numbers joined by `_`:

- **`p2_` string**, per lens: a radius, `cx`, `cy` and three angles; then
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
- **The back lens.** It does not absorb the error: it settles where the
  factory calibration puts it (3.23 cm) only at s = 1.3784.
- **The dataset's track.** Over these 133 s both runs drift from the
  dataset's track by more than the two differ, so it does not decide
  between them.

## The lenses on the camera

The three angles per lens describe how the lens sits on the camera, but
their convention was not found. Three estimates from the data agree on the
back lens's rotation: a half turn about the vertical axis, then about 1° more.

| Estimate | Beyond the half turn |
| --- | ---: |
| OverMaps360's rig (its perspective views, through the per-view fits above) | 1.01° |
| SfM of the two lenses as separate cameras, mean over 200 frames of walk 0313afa5 | 1.01° (0.16° from the rig's) |
| SfM with one rig per frame, the back lens refined (same frames) | 1.13° |

No reading of the angles as Euler angles comes closer than 0.30° to the
dataset's rig. A fit from the lenses' overlap alone is biased: ground 2 m
below the camera shifts a point by up to 0.9° between the lenses (they are
3.2 cm apart), and along the overlap that cannot be told apart from a tilt.

So `gsplat_photos --insv` does not stitch, and does not rely on the angles.

- **Faces.** Each lens is cut into five 90° faces within 90° of its axis.
- **Rig.** A frame's ten faces are one rig in the front lens's frame. The
  back lens's pose on it starts from the half turn with no offset.
- **Refinement.** The mapper's bundle adjustments refine that pose (COLMAP's
  `ba_refine_sensor_from_rig`). On those 200 frames it settles 1.13° from
  the half turn and 3.6 cm behind the front lens; the calibration says 3.2 cm.
  The front lens's faces, refined too, move by at most 0.034° and 5 mm.
