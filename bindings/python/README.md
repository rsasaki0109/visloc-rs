# visloc (Python bindings)

Python bindings for [visloc-rs](https://github.com/rsasaki0109/visloc-rs), a
pure-Rust visual localization / SfM library, built with
[pyo3](https://pyo3.rs) and [maturin](https://www.maturin.rs). Arrays go in and
out as NumPy arrays; no GPU features are compiled in.

The bindings cover the stable-intent surface of the Rust crates
(`docs/api_stability.md`):

| Python | Rust |
| --- | --- |
| `Camera` (COLMAP models + Double Sphere): `project`, `unproject`, `normalize`, `calibration_matrix` | `visloc_core::types::Camera` |
| `Pose` (alias `SE3`): rotation / quaternion / translation, `compose` (`@`), `inverse`, `transform`, `exp` / `log`, `camera_center` | `visloc_core::geometry::{SE3, Pose}` |
| `Reconstruction`, `Image`: COLMAP text + binary read/write, cameras / images / points3D as NumPy arrays | `visloc_io::colmap` |
| `localize`: 2-D keypoints + descriptors against a map with per-point descriptors (brute-force matching, PnP + RANSAC, refinement) | `visloc_localization::LocalizationPipeline` |
| `estimate_pose_pnp_ransac`: pose from 2D-3D correspondences | `visloc_vision::ransac::PnPRansac` |
| `evaluate_ate`, `evaluate_rpe`, `umeyama_alignment`: trajectory evaluation with SE(3) / Sim(3) alignment | `visloc_tracking::{PoseTrajectory, umeyama_similarity_transform}` |

Conventions match the Rust library and COLMAP: a camera pose is the
**world-to-camera** transform `T_cw` (`x_cam = R x_world + t`), quaternions are
`(w, x, y, z)` unless `scalar_first=False`, and pixel coordinates follow the
camera model's principal point.

## Build and install

Requirements: a Rust toolchain (the repository pins it in `rust-toolchain.toml`),
Python >= 3.9, and NumPy.

```sh
cd bindings/python
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy

# Editable development install into the active virtualenv:
maturin develop --release

# Or build a wheel:
maturin build --release --out dist && pip install dist/*.whl

# Tests (use the repository's COLMAP fixtures when present):
pytest -q
```

This crate is intentionally **not** a member of the root Cargo workspace (it is
listed under `[workspace] exclude` and has its own `Cargo.lock`): a pyo3
extension module leaves libpython symbols to be resolved at import time, so it
must not take part in `cargo test --workspace` at the repository root. Run
`cargo fmt` / `cargo clippy -- -D warnings` from this directory.

## Examples

### Cameras and poses

```python
import numpy as np
import visloc

camera = visloc.Camera("OPENCV", 640, 480, [500, 500, 320, 240, -0.1, 0.01, 0.0, 0.0])
pose = visloc.Pose.from_quaternion([1, 0, 0, 0], [0.0, 0.0, 0.5])  # world-to-camera

world_points = np.random.default_rng(0).uniform(-1, 1, (100, 3)) + [0, 0, 5]
pixels = camera.project(world_points, pose=pose)  # (100, 2), NaN where not visible
rays = camera.unproject(pixels)                   # (100, 3) unit camera-frame rays

relative = pose.inverse() @ visloc.Pose.exp([0.1, 0, 0, 0, 0.05, 0])
print(relative.matrix(), relative.quaternion(), pose.camera_center())
```

### COLMAP models

```python
model = visloc.Reconstruction.read("sparse/0")  # binary if cameras.bin exists, else text
print(model)                                     # Reconstruction(cameras=..., images=..., points3d=...)

xyz = model.points3d                             # (M, 3) float64
ids = model.point3d_ids                          # (M,) uint64, aligned with xyz
for image_id, image in model.images.items():
    T_cw = image.pose.matrix()                   # 4x4 world-to-camera
    uv = image.keypoints                         # (N, 2)
    observed = image.point3d_ids                 # (N,) int64, -1 = untriangulated

model.write_binary("sparse_out")                 # or write_text(...)
```

Models can also be built from NumPy data with `add_camera`, `add_image`
(`visloc.Image(id, camera_id, pose, keypoints, point3d_ids)`), and
`set_points3d(ids, xyz, descriptors=None)`. Point tracks in `points3D` are
derived from the image observations when writing. visloc's map model does not
keep image names, point colors, or reprojection errors: written models use
`image_<id>.jpg`, white points, and zero error.

### Localization

```python
# Attach one descriptor per 3-D point: from a `LANDMARK_ID D0 D1 ...` text file...
model.load_point3d_descriptors("landmark_descriptors.txt")
# ...or directly: model.set_points3d(ids, xyz, descriptors)  # (M, D)

result = visloc.localize(
    model.cameras[1], query_keypoints, query_descriptors, model,  # (N, 2), (N, D)
    ratio=0.8, ransac_iterations=256, reprojection_threshold=4.0, min_inliers=12,
)
if result.success:
    print(result.pose.matrix(), result.inlier_count, result.mean_reprojection_error)
else:
    print(result.failure_reason)  # e.g. "no_descriptor_matches", "quality_gate_failed"

# With known 2D-3D correspondences:
pnp = visloc.estimate_pose_pnp_ransac(camera, pixels, world_points, reprojection_threshold=2.0)
if pnp is not None:
    print(pnp.pose, pnp.inliers)
```

Note that `min_inliers` defaults to `0`, mirroring the Rust
`LocalizationConfig`; set a floor (for example 8-12) for real data.

### Trajectory evaluation

```python
# (N, 3) camera centers, (N, 7) [tx ty tz qx qy qz qw] (TUM), or (N, 4, 4) camera-to-world.
ate = visloc.evaluate_ate(estimated, ground_truth, alignment="sim3")  # "se3" (default), "sim3", "first", "none"
print(ate.rmse, ate.mean, ate.max, ate.alignment.scale)
per_frame = ate.errors

# Match by frame id instead of by index:
ate = visloc.evaluate_ate(estimated, ground_truth, estimated_ids=est_ids, reference_ids=gt_ids)

rpe = visloc.evaluate_rpe(estimated_poses, gt_poses, delta=10)
print(rpe.translation["rmse"], rpe.rotation_deg["rmse"])

T = visloc.umeyama_alignment(source_xyz, target_xyz, with_scale=True)
aligned = T.apply(source_xyz)
```

## Typing

The package ships `py.typed` and a `_visloc.pyi` stub describing every class
and function, so editors and type checkers see the full API.
