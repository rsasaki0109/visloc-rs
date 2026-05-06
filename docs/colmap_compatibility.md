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

Text parsing preserves the COLMAP camera model name as one of:

- `SIMPLE_PINHOLE`
- `PINHOLE`
- `SIMPLE_RADIAL`
- `RADIAL`
- `OPENCV`
- `Unknown(String)` for other text model names

Binary parsing recognizes COLMAP camera model ids 0 through 10. Models that are not directly represented in `CameraModel` are stored as `Unknown(String)` with the corresponding COLMAP name.

Current projection and normalization use the pinhole intrinsics subset:

- `PINHOLE` and `OPENCV`: `fx, fy, cx, cy`
- `SIMPLE_PINHOLE`, `SIMPLE_RADIAL`, and `RADIAL`: `f, cx, cy`

Distortion parameters are preserved in `Camera.params`, but projection currently ignores distortion. For high-distortion cameras, callers should undistort features/images before localization or provide a custom camera/projection path in a future extension.

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

The writer does not emit binary COLMAP models and does not write feature descriptors.

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
- Binary COLMAP writing.

These can be added later without changing the map-based localization boundary.
