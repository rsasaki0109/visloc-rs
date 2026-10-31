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
- **Culling and depth.** A fisheye draws what lies 0.1 from its centre,
  even behind the image plane, and sorts by distance. Its depth channels
  hold the distance from the camera centre. Every lens stops at
  `max_theta`, fitted to the image: along 64 directions, the angle where a
  ray leaves the image grown by 10%, but never past an angle where the
  distortion folds back.
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
  against finite differences, on five views of one scene: a pinhole, a
  distorted pinhole, a fisheye seeing the scene ~50° off its axis, ETH3D's
  thin-prism lens, and a fisheye seeing it past 90°, behind its image
  plane.
- Every wgpu kernel validates with naga (`shaders::every_kernel_validates`),
  and the uniforms match the WGSL layout.

Still to run on a GPU:

- `cargo test -p visloc-gsplat-render --features gpu gpu_lens`
- the CUDA `lens_*` tests in `crates/gsplat-cuda/tests/oracle.rs`, which
  compare the GPU forward pass, screen gradients and parameter gradients on
  four lens views against the oracle

## SfM

`visloc-core` has the camera models: `OpenCvFisheye`, `SimpleRadialFisheye`,
`RadialFisheye`, `Fov`, `DoubleSphere` and, new here, COLMAP's
`ThinPrismFisheye`. Each has projection, unprojection to a unit ray, and
the point Jacobian. The COLMAP-port mapper reaches cameras through those.
Two pieces are still missing:

- Its bundle adjustment (`rig_ba_solver`) solves undistorted pinholes and
  equirectangular panoramas only.
- `gsplat_photos` builds a pinhole camera.

## Validation data

ETH3D's low-res many-view training scenes (`<scene>_rig.7z`) ship the
distorted images of four `THIN_PRISM_FISHEYE` cameras, 752×480 grey, with
their calibration and poses, beside the undistorted images and the laser
scans. They will serve both halves:

- 3DGS on the distorted images against 3DGS on the undistorted ones.
- SfM poses against the ground truth.
