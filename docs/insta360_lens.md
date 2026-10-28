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
   with a free scale to SIFT matches between a frame and its views gives
   s = 1.377–1.385 (eight fits, all four 90° orientations of the sensor
   axes). The rotations fitted for two views of one lens agree with the
   dataset's rig to 0.10°.
4. **The two lenses meet.** Rays through the lenses' overlap (90–100° off
   each axis, 1,753 matches in six frames) agree best near s = 1.38–1.39,
   and are clearly worse at 1.40 or 1.36. The lenses are 3.2 cm apart, so
   nearby ground points shift this estimate upwards.

The dataset's own views follow a slightly different model: fitting one
freely gives `fx` and `fy` 0.25% apart, and the factory model with the best
scale stays about 1 px off. So they show the scale, but they are not a
pixel-exact reference.

## What is not decoded

**The lens-to-lens rotation.** The three angles per lens describe how the
lens sits on the camera, but their convention was not found. Fitted from
the overlap, the back lens's rotation is a 180° turn about y, then
1.1–1.6° more depending on which matches are used. No reading of the angles
as Euler angles comes closer than 0.57° to that.

Ground 2 m below the camera shifts a point by up to 0.9° between the lenses
(they are 3.2 cm apart). Along the overlap, that shift cannot be told apart
from a tilt. A stitched panorama needs this rotation to about 0.1°.

So `gsplat_photos --insv` does not stitch. Each lens is cut into five 90°
faces that stay within 90° of its axis, and the faces of a lens are solved
as one rig. The two lenses are separate cameras, tied together only by the
points they share.
