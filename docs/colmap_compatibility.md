# COLMAP Compatibility

`visloc-rs` can reuse sparse COLMAP/SfM maps for map-based visual localization. The IO layer is intentionally focused on sparse visual maps: cameras, registered images, sparse 3D points, observations, and optional descriptors supplied outside COLMAP.

## Supported Inputs

### Text Models

`visloc_io::colmap::read_colmap_text_model` expects a directory containing:

- `cameras.txt`
- `images.txt`
- `points3D.txt`

`ColmapMapProvider::from_text_model_dir` wraps this as a `MapProvider`. Use `from_text_model_dir_validated` when callers want structural validation during loading.

### Binary Models

`visloc_io::colmap::read_colmap_binary_model` expects a directory containing:

- `cameras.bin`
- `images.bin`
- `points3D.bin`

`ColmapMapProvider::from_binary_model_dir` wraps this as a `MapProvider`. Use `from_binary_model_dir_validated` for structural validation.

## Camera Models

Text parsing maps the COLMAP camera model name to a `CameraModel`:

- Pinhole family: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`
- Fisheye family: `OPENCV_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`, `FOV`
- `Unknown(String)` for other text model names (e.g. `THIN_PRISM_FISHEYE`)

Binary parsing recognizes COLMAP camera model ids 0 through 10; ids 5
(`OPENCV_FISHEYE`), 6 (`FULL_OPENCV`), 7 (`FOV`), 8 (`SIMPLE_RADIAL_FISHEYE`)
and 9 (`RADIAL_FISHEYE`) decode to their real `CameraModel` variants, and id 10
(`THIN_PRISM_FISHEYE`) remains `Unknown`.

`Camera::intrinsics` reads the shared `[fx, fy, cx, cy]` layout (pinhole family,
`OPENCV_FISHEYE`, `FOV`, `DoubleSphere`) or `[f, cx, cy]` (the radial-fisheye
family). `Camera::project` / `normalize_pixel` / `unit_ray_from_pixel` dispatch
on the model:

- pinhole family: pinhole projection plus the optional radial `(k1, k2)` term
- `OPENCV_FISHEYE` / `SIMPLE_RADIAL_FISHEYE` / `RADIAL_FISHEYE`: Kannala-Brandt
  equidistant, `theta_d = theta · (1 + k1·θ² + k2·θ⁴ + k3·θ⁶ + k4·θ⁸)`
- `FOV`: Devernay's FOV model
- `DoubleSphere`: the Double Sphere model (shared with the Basalt VI-SLAM path)

`normalize_pixel` returns the `(x, y, 1)` form of the recovered ray, so it is
only defined while the ray points forward; `unit_ray_from_pixel` returns the
full unit ray and is the robust entry point for fisheye fields of view wider
than 90°.

## Map Semantics

COLMAP images become `Keyframe` values:

- image id -> `Frame.id`
- camera id -> `Frame.camera_id`
- COLMAP world-to-camera quaternion and translation -> `Frame.pose`
- 2D points -> `Frame.keypoints`
- 2D points with non-negative `POINT3D_ID` -> `Observation`

COLMAP points become `Landmark` values:

- point id -> `Landmark.id`
- xyz -> `Landmark.position`

COLMAP RGB, reprojection error, and track metadata are parsed only as needed to advance through the file. They are not currently represented in `Landmark`.

## Descriptor Handling

COLMAP sparse models do not contain the local feature descriptors needed by `visloc-rs` matching. Use one of these paths:

- Embed descriptors in `Landmark.descriptor` when constructing a `VisualMap` in memory.
- Load external landmark descriptors with `visloc_io::descriptors::read_landmark_descriptors_txt`.
- Use `ColmapMapProvider::from_text_model_dir_with_descriptors` or its validated variant.

The descriptor text format is documented in [interfaces.md](interfaces.md#descriptor-store-text-format).

## Writing Maps

`visloc_io::colmap::write_colmap_text_model` writes:

- `cameras.txt`
- `images.txt`
- `points3D.txt`

This is intended for sparse map reuse after local/online updates. The writer emits deterministic text sorted by ids and fills unsupported visual fields conservatively:

- point RGB is written as `255 255 255`
- point error is written as `0`
- generated image names use `image_<FRAME_ID>.jpg`
- missing keypoint observations are written with `POINT3D_ID = -1`

The writer does not write feature descriptors.

`visloc_io::colmap::write_colmap_binary_model` writes the same records as
`cameras.bin`, `images.bin`, and `points3D.bin` (same camera-model mapping,
generated image names, `-1` placeholders, and white/zero-error points). Camera
and image ids, and track entries, must fit COLMAP's 32-bit fields; otherwise the
writer returns `ColmapError::InvalidExportInput` before writing any file.

## 3DGS / NeRF Bootstrap Export

`visloc_io::colmap::write_colmap_text_model_for_3dgs` and `write_colmap_binary_model_for_3dgs` materialise the COLMAP triple (`cameras.{txt,bin}`, `images.{txt,bin}`, `points3D.{txt,bin}`) directly from a stereo VO output — `(camera, &[Pose], &[FeatureSet], &[Vec<StereoFeature>], image_name: Fn(usize) -> String)` — for the explicit purpose of bootstrapping a downstream 3D Gaussian Splatting / NeRF trainer (drop the directory into `nerfstudio ns-train splatfacto --data <dir>` or Inria gaussian-splatting's `convert.py --skip-matching` flow).

The two writers are a symmetric pair so a single VO run can emit both formats from the same input. Concretely, any `(camera, image_name)` accepted by one writer is also accepted by the other:

- `image_name(frame_idx)` is checked for characters that would corrupt either format — NUL terminates the binary NAME field early, and ASCII space / tab / LF / CR break the text format's space-separated tokens (LF would inject a spurious image record into `images.txt`).
- `camera.model` must be a COLMAP-recognised name (`PINHOLE` / `SIMPLE_PINHOLE` / `SIMPLE_RADIAL` / `RADIAL` / `OPENCV` / `OPENCV_FISHEYE` / `FULL_OPENCV` / `FOV` / `SIMPLE_RADIAL_FISHEYE` / `RADIAL_FISHEYE` / `THIN_PRISM_FISHEYE`); the binary writer needs a numeric model id, and the text writer enforces the same set so unencodable `CameraModel::Unknown(name)` values are rejected up-front by both surfaces.

Both writers return a structured `ColmapError::InvalidExportInput` for inputs they refuse rather than writing partial files.

Cross-format equivalence (same input → maps that re-read to the same camera intrinsics, keyframe poses, and landmark world positions within `1e-9`) is pinned by `crates/io/tests/colmap_export.rs::write_colmap_text_and_binary_models_for_3dgs_emit_equivalent_maps`. Real-data parity on a small KITTI stride-4 subset is exercised by `scripts/run_kitti_3dgs_smoke.sh`, which runs `examples/inspect_colmap_text_model` and `examples/inspect_colmap_binary_model` against the live writer output and aborts on any per-format count disagreement.

### Consuming a COLMAP model in Rust (3DGS)

The `visloc-gsplat-core` crate reads a COLMAP model back into a 3DGS scene seed:
`load_colmap_scene(dir, seed_log_scale)` turns the sparse landmarks into degree-0
gaussians and each registered image into a render camera, and its CPU reference
rasterizer renders a frame without a GPU. This is the stage-0 foundation of the
Rust 3DGS effort; the plan and staged roadmap are in
[rust_3dgs_plan.md](rust_3dgs_plan.md).

## Validation

Use map validation before localization:

- `VisualMap::validate` checks structural references.
- `VisualMap::validate_with_descriptors` also checks descriptor availability.
- `ColmapMapProvider::validate_map` and `validate_for_localization` expose those checks for loaded COLMAP models.
- The `*_validated` provider constructors return an error when validation fails.

## Current Non-Goals

- Dense COLMAP outputs.
- Full COLMAP database parsing.
- Feature extraction from COLMAP databases.
- Bundle adjustment compatibility.
- Distortion-aware projection during PnP.
- Generic binary COLMAP writing from a `VisualMap`. The 3DGS-shaped binary writer (`write_colmap_binary_model_for_3dgs`) covers the bootstrap path but there is no `write_colmap_binary_model(&VisualMap, path)` mirror of the generic text writer yet (no caller needs it).

These can be added later without changing the map-based localization boundary.
