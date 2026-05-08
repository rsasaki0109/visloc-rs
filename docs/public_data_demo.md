# Public Data Demo

The README localization image uses public COLMAP South Building data rather than generated artwork.

## Source

- Dataset: COLMAP South Building example dataset
- Download used for the demo: `https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip`
- Input images used for the small reconstruction: `P1180141.JPG` through `P1180149.JPG`

The original archive is not committed because it is about 400 MiB. The repository only stores the derived README assets.

## Reconstruction

The demo was built from a small subset of the public images:

1. Download `south-building.zip`.
2. Extract the first 9 South Building images.
3. Resize the images to 1024x768 for a compact README-oriented reconstruction.
4. Run `pycolmap.extract_features`, `pycolmap.match_exhaustive`, and `pycolmap.incremental_mapping`.
5. Export the sparse model to COLMAP text format.

The generated model registered 9 cameras and reconstructed 1,428 sparse 3D points.

## README Assets

- `docs/assets/south-building-query.jpg`: real query image from the public dataset subset.
- `docs/assets/south-building-localization.png`: final frame of the public-data sequence demo, with the current real image, reliable 2D-3D matches, sparse SfM map, and localized camera path.
- `docs/assets/south-building-localization.gif`: animated README sequence demo showing 9 real images localized frame by frame against the same reusable visual map.
- `docs/assets/south-building-localization-rich.png`: feature-rich README variant built from the same final frame, with additional detected image-feature overlays and highlighted pose-link visualization for a clearer first impression.
- `docs/assets/south-building-localization-rich.gif`: feature-rich animated README variant built from the same real-image sequence.

The feature-rich README variants are generated with:

```sh
python3 scripts/build_rich_readme_demo.py
```

This helper uses Python with Pillow and OpenCV. It is an asset-generation tool,
not part of the Rust runtime or CI quality gate.

The visualization is intentionally a lightweight README artifact. It is not a benchmark and does not claim full SLAM; it demonstrates the library direction in a readable way: Visual Localization first, using a reusable visual map and per-frame camera poses for a short image sequence.

## Public-Data Loop-Closure Demo

The `online_slam_public_loop_demo` example reads a COLMAP-text-format sparse
reconstruction (`cameras.txt`, `images.txt`, `points3D.txt`) from disk and
runs the full tracking + verifier + pose-graph SE(3) Gauss-Newton stack on
the loaded keyframes. Without flags it synthesizes a 12-keyframe orbit
fixture so the demo stays runnable in CI:

```sh
cargo run --example online_slam_public_loop_demo
```

To point it at a real reconstruction (e.g., a COLMAP run on the South
Building subset above):

```sh
cargo run --example online_slam_public_loop_demo -- \
    --colmap-path path/to/sparse/0
```

If the COLMAP folder contains a sibling `landmark_descriptors.txt` (one row
per landmark: `LANDMARK_ID D0 D1 ...`), it is consumed automatically;
otherwise the demo generates synthetic per-landmark descriptors so it can
run unmodified on any registered COLMAP reconstruction. Override the
descriptor file explicitly with `--descriptors-path <file>`. Use
`--out-dir <dir>` to copy the generated synthetic fixture out of the
temporary directory for inspection.

Loop closure is triggered on the keyframe pair with the largest frame-id
gap so the verifier scale calibrated for that pair is not contaminated by
intermediate matches; the demo prints the chosen `min_frame_id_gap`,
per-iteration cost, and per-keyframe translation / rotation error against
the loaded ground-truth poses.
