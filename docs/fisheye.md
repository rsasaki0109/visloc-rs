# Fisheye and distorted cameras

The goal is to run SfM and 3DGS on fisheye images directly, without
resampling them to pinhole views first. This page covers what is done and
what is still open.

## 3DGS: drawing through the lens

A camera's intrinsics carry a lens (`visloc_gsplat_core::lens::Lens`):

- **Projection.** Either the pinhole's `(x, y) / z` or the equidistant
  `θ (x, y) / r`, where `r = |(x, y)|` and `θ = atan2(r, z)`. The equidistant
  form is the Kannala-Brandt / OpenCV fisheye base, and it images rays past
  90° as well.
- **Distortion.** COLMAP's distortion of the projected coordinates
  `(u, v)`, with `ρ² = u² + v²`:
  `u' = u + u (k1 ρ² + k2 ρ⁴ + k3 ρ⁶ + k4 ρ⁸) + 2 p1 u v + p2 (ρ² + 2 u²) + sx1 ρ²`,
  and `v'` likewise, with `p1`/`p2` and `sx1`/`sy1` swapped.

COLMAP's `SIMPLE_RADIAL`, `RADIAL`, `OPENCV` and `FULL_OPENCV` without its
rational denominator are pinhole lenses of this form. `SIMPLE_RADIAL_FISHEYE`,
`RADIAL_FISHEYE`, `OPENCV_FISHEYE` and `THIN_PRISM_FISHEYE` are equidistant
ones (`colmap_scene::lens_of`). A COLMAP dataset with any of these models
now trains on its images as they are. Before, `SIMPLE_RADIAL`, `RADIAL` and
`OPENCV` loaded with their distortion dropped, and the fisheye models did
not load at all.

The EWA splatting linearises the lens at each gaussian's centre. The 2×3
Jacobian `J` of the lens replaces the pinhole one in `Σ' = J W Σ Wᵀ Jᵀ`:

- **CPU reference** (`cpu_render`) and **f64 gradient oracle**
  (`backward`). The lens math is written once, generic over the scalar
  (`lens::LensScalar`). On the oracle's dual numbers, the Jacobian's own
  derivatives, which the backward pass needs, come out exactly.
- **CUDA** (`native/gsplat_cuda/lens.cuh`, ABI 11) and **wgpu**
  (`common.wgsl`). The same math, run in the backward pass on 3-component
  dual numbers in the camera-space point. The undistorted pinhole keeps its
  original code path in all four.
- **Culling and depth.** A fisheye draws what lies 0.1 from its centre
  within the wider of `max_theta` (below) and 90°, even behind the image
  plane, and sorts by distance. Its depth channels hold the distance from
  the camera centre. A distorted pinhole draws what lies 0.1 in front of
  it, as the undistorted one does.
- **Past the image.** A lens is trusted out to `max_theta`, fitted to the
  image: along 64 directions, the angle where a ray leaves the image grown
  by 10%, but never past an angle where the distortion folds back. Past it
  the polynomial is not calibrated and can blow up. A gaussian centred
  there is drawn through the lens's first-order extension at the cone
  instead (`lens::extended_lens_coords`). Its centre is turned onto the
  cone at the same distance and azimuth, giving `q`. It then projects to
  `m(q) + J(q) (p − q)`, and its footprint uses `J(q)`. This is the lens
  counterpart of a pinhole's 1.3× frustum clamp: a gaussian off the
  frustum still reaches into the image.
- **Near the axis.** The equidistant scale `θ / r` and its derivative cancel
  catastrophically in f32. Below `(r / z)² = 0.01` they come from their
  series.
- **Pinhole only, for now.** The TSDF meshing, the planar median depth and
  the depth-normal loss back-project pixels through a pinhole. They take a
  distorted pinhole without its distortion, as they did before, and refuse
  fisheye views (`CameraView::pinhole_geometry`).

### Verified so far

These checks run on the CPU (`cargo test -p visloc-gsplat-core`):

- The lens Jacobian against central differences, on pinhole, distorted,
  equidistant and thin-prism lenses, including points within 1e-4 of the
  axis.
- The series and closed forms agree at the switch.
- ETH3D's thin-prism lens against `visloc-core`'s `ThinPrismFisheye`, which
  is COLMAP's formula.
- The f64 oracle against the f32 CPU renderer, and its parameter gradients
  against finite differences, on seven views of one scene:
  - a pinhole
  - a distorted pinhole
  - a fisheye seeing the scene ~50° off its axis
  - ETH3D's thin-prism lens
  - a fisheye seeing it past 90°, behind its image plane
  - a narrow fisheye with every gaussian centred past its `max_theta`
  - a narrow distorted pinhole with every gaussian centred past its
    `max_theta`
- The extension meets the lens at `max_theta` and keeps moving points
  outwards, on ETH3D's narrowest rig lens, whose polynomial blows up past
  the image.
- Every wgpu kernel validates with naga (`shaders::every_kernel_validates`),
  and the uniforms match the WGSL layout.

On a GPU, the CUDA `lens_*` tests (`crates/gsplat-cuda/tests/oracle.rs`) and
the wgpu `gpu_lens_*` tests compare the forward pass, the screen gradients
and the parameter gradients against the oracle. They use six lens views of
a 64-gaussian scene:

- a distorted pinhole
- a fisheye ~35° off its axis
- ETH3D's thin-prism lens
- a fisheye past 90°
- a narrow fisheye with much of the scene past its `max_theta`
- a narrow distorted pinhole with much of the scene past its `max_theta`

CUDA agrees with the oracle within 1e-6 in the forward pass and within
1.2e-5 of each gradient's largest value. wgpu, run on lavapipe, agrees
within 2e-5 and 2.1e-4.

### ETH3D rig: delivery_area

The scene has 948 images from four rig cameras, of which 829 train and every
8th of the sorted names (119) is held out. Each variant trains for 30k steps
with the CUDA trainer:

| Variant | Images | Camera | PSNR | SSIM |
|---|---|---|---|---|
| `dist_lens` | distorted | `THIN_PRISM_FISHEYE`, as calibrated | 26.17 | 0.916 |
| `dist_pinhole` | distorted | `PINHOLE`, distortion dropped | 22.10 | 0.822 |
| `undist` | ETH3D's undistorted, resized to 752×480 | `PINHOLE` | 33.29 | 0.964 |

On the same images, modelling the lens gains 4.1 dB. `undist` is not
comparable: its images are resampled, and so smoother.

ETH3D's undistortion and this lens model agree. Warping a distorted image
onto its undistorted camera with `visloc-core`'s `ThinPrismFisheye` gives
59 dB against ETH3D's own undistorted image, on all four cameras.

Scoring one splat on the other dataset's views (`gsplat_eval`) is a harsher
test:

- The `dist_lens` splat scores 15.4 dB on the undistorted views.
- The `undist` splat scores 18.7 dB through the lens.

Both renders line up with the ground truth, but differ in brightness. The
gap comes from the few huge gaussians that cover the background. In the
`dist_lens` splat, 0.8% of the gaussians have a scale of 2 m or more. Without
them, the lens and pinhole renders of one frame agree within about 1 dB. EWA
linearises a gaussian at its centre, so a gaussian metres wide gets a
footprint that depends on the projection, and the two sort keys (distance
vs `z`) order it differently against the walls in front. A fix would be a
scale cap in training, or projecting with an unscented transform as 3DGUT
does.

An earlier version culled every gaussian centred past `max_theta`. On these
images it scored 9.6 dB on the undistorted views of the two narrow cameras.
Training had left big dark gaussians just outside their fields of view,
where the lens never drew them.

## SfM: mapping through the lens

`visloc-core` has the camera models: `OpenCvFisheye`, `SimpleRadialFisheye`,
`RadialFisheye`, `Fov`, `DoubleSphere` and COLMAP's `ThinPrismFisheye`. Each
has projection, unprojection to a unit ray, and the point Jacobian. The
COLMAP-port mapper uses those for registration, triangulation and its
filters, and now for bundle adjustment too:

- **Bundle adjustment.** The native solver (`rig_ba_solver`) projects a
  camera with distortion or a fisheye projection through its `Lens`. The
  lens gives the normalized coordinates `m = (pixel − c) / f` and their
  Jacobian, from `Camera::project_with_point_jacobian`. The solve's
  intrinsics then place them at `(fx m.x + cx, fy m.y + cy)`. Undistorted
  pinholes keep the direct path. Before, the solver took any camera with a
  focal length for an undistorted pinhole, without a warning. On a synthetic
  Kannala-Brandt sequence the frame centres came out 3.7 cm off; through the
  lens they are within 1 mm.
- **Intrinsics refinement.** Focal length and principal point refinement
  (`ba_refine_focal_length`, `ba_refine_principal_point`) works in front of
  a lens, with the distortion fixed. `Camera::set_intrinsics` writes the
  refined values back and keeps the distortion coefficients.
- **Rig manifest.** A sensor row can carry any COLMAP model, with its
  parameters after the pose, for example
  `S 0 1 752 480 541.9 541.6 413.8 216.0 1 0 0 0 0 0 0 THIN_PRISM_FISHEYE 541.9 …`.
  `run_colmap_port` writes such a row for a lens camera.
- **`gsplat_photos`, `gsplat_photos_cuda`.** `--camera "MODEL W H PARAMS..."`
  takes the photos' calibrated camera, as a `cameras.txt` row without its
  id. It is scaled to the working images and held fixed. SfM and training
  then follow its lens on the images as they are. `sparse/0/cameras.txt`
  keeps the model. The mesh step is skipped for a fisheye.
- **Two-view verification.** It fits E on normalized coordinates
  (`normalize_pixel`) and F and H on raw pixels, as COLMAP does. With strong
  distortion, E wins and the pair is classified as calibrated.

Still open:

- Refining the distortion itself (COLMAP's `ba_refine_extra_params`).
- A rig of differently calibrated cameras in `gsplat_photos`, such as
  ETH3D's four.
- Rays past 180°. `normalize_pixel` drops pixels whose ray has `z ≤ 0`, so
  E and PnP never see them.

### ETH3D rig: SfM on one camera

`gsplat_photos_cuda` ran on one rig camera's images at a time: 237 images
for delivery_area, 165 for terrains, 752×480 grey. The run's camera
centres are aligned to the ground truth by a similarity transform and
scored by RMSE. Centres only: on these near-straight paths, a centre
alignment leaves the roll about the path open, so rotation errors are not
comparable. Two runs per camera:

- **lens**: the distorted images with `--camera` and their
  `THIN_PRISM_FISHEYE` calibration.
- **undistorted**: ETH3D's undistorted images with their `PINHOLE`
  calibration. This baseline the old pipeline could already run.

| Scene | Camera | lens | undistorted |
|---|---|---|---|
| delivery_area | cam4 | 11.65 cm | 1.91 cm |
| delivery_area | cam5 | 1.54 cm | 112.6 cm (reversed) |
| delivery_area | cam6 | 58.0 cm (reversed) | 1.74 cm |
| delivery_area | cam7 | 1.84 cm | 1.32 cm |
| terrains | cam4 | 0.40 cm | 0.39 cm |
| terrains | cam5 | 0.34 cm | 0.36 cm |
| terrains | cam6 | 1.99 cm | 1.36 cm |
| terrains | cam7 | 1.29 cm | 2.61 cm |

Leaving out one failed run on each side, both median about 1.4 cm: the
distorted images through their lens map as well as the undistorted ones.
On delivery_area cam4's distorted images, two other runs fail:

- the calibration with its distortion dropped: 119.5 cm
- no calibration, the self-calibrated pinhole prior: 103.7 cm

The two failures, about 1 m off with rotations near 180°, are a depth
reversal of the whole reconstruction. It happens on the undistorted,
pinhole side too, so it is the mapper's, not the lens's, and is still
open. delivery_area cam4's 11.65 cm looks like sensitivity rather than a
model error, since the other cameras map well with the same model. This
sequence of forward motion with little parallax has a soft bending mode:
moving the principal point by half a pixel either way gives 1.2 or 7.9 cm.

## Validation data

ETH3D's low-res many-view training scenes (`<scene>_rig.7z`) ship the
distorted images of four `THIN_PRISM_FISHEYE` cameras, 752×480 grey, with
their calibration and poses, beside the undistorted images and the laser
scans. They serve both halves:

- 3DGS on the distorted images against 3DGS on the undistorted ones
  (above).
- SfM poses against the ground truth (to do).
