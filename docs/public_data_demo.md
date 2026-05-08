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

## Real-Image Visual Odometry + Loop Closure Demo

The `online_slam_image_vo_loop_demo` example consumes a KITTI-format
grayscale image sequence directly — no synthetic poses, no toy data — and
runs the full visual SLAM loop in Rust:

1. `read_kitti_image_sequence_dir` loads the `image_0` folder plus the
   matching `calib.txt` (`P0` projection by default).
2. Per-frame `CornerFeatureExtractor` features (Shi-Tomasi-style corners
   with intensity-difference patch descriptors) feed a
   `CrossCheckMatcher<BruteForceMatcher>` between consecutive frames.
3. `RelativePoseEstimator` (8-point essential matrix RANSAC + cheirality
   recovery) returns a `previous_to_current` SE(3) for every adjacent
   pair, which is integrated into a monocular VO trajectory.
4. The same essential-matrix pipeline runs once more between the first and
   last frames; if it produces enough RANSAC inliers, the recovered
   relative pose is kept as a loop-closure constraint.
5. `PoseGraph::optimize_se3_iterative` (Levenberg-Marquardt + Cholesky)
   pulls the chain back along the loop edge; the example writes `vo.csv`
   and `corrected.csv` (id, x, y, z) to the output directory.

Monocular essential-matrix VO is scale-ambiguous; this demo uses unit
translation per pair, so the resulting trajectory is in arbitrary units.
The loop-closure constraint, computed from the same essential-matrix
path, is what closes the chain.

To run on KITTI 00 (download the official odometry grayscale dataset
first; the file layout expected here is the same as KITTI's
`sequences/00/`):

```sh
cargo run --release --features image-io \
    --example online_slam_image_vo_loop_demo -- \
    --image-dir /path/to/KITTI_odometry/sequences/00/image_0 \
    --calib    /path/to/KITTI_odometry/sequences/00/calib.txt \
    --max-frames 200 \
    --frame-stride 4 \
    --out-dir target/kitti_image_vo_loop_demo
```

## KITTI Loop-Closure Asset (README)

The README hero (`docs/assets/kitti_loop_closure.{png,gif}`) is generated
end-to-end from the **real-image** `online_slam_image_vo_loop_demo`
pipeline on KITTI 00. Pipeline:

1. Stream a stride-subsampled subset of KITTI odometry seq 00 `image_0`
   directly from the public S3 archive (no full 23 GB download required):

   ```sh
   python3 scripts/fetch_kitti_seq00_images.py \
       --stride 4 --max-frames 1140 --workers 8 --also-fetch-poses \
       --out-dir ~/datasets/kitti_seq00_subset
   ```

2. Run the real-image VO + loop-closure example. Trim to where the loop
   actually closes (KITTI 00 first revisits start within ~1 m at original
   frame ~4447, i.e., subsample index 1111 at stride 4):

   ```sh
   cargo run --release --features image-io \
       --example online_slam_image_vo_loop_demo -- \
       --image-dir ~/datasets/kitti_seq00_subset/image_0 \
       --calib    ~/datasets/kitti_seq00_subset/calib.txt \
       --max-frames 1112 --frame-stride 1 \
       --out-dir target/kitti_image_vo_loop_demo
   ```

   Sample run on the 1112-frame subset: 195.9 mean RANSAC inliers per
   sequential edge, loop edge KF 0 ↔ KF 1111 verified with 40 inliers,
   translation PGO collapses cost 1.6 M → 36.1, SE(3) refine drives it
   down to 0.02.

3. Render the README asset. The Python helper applies start-anchored
   similarity Procrustes alignment so the unit-scale monocular VO and
   the corrected trajectory share a common metric frame with truth:

   ```sh
   python3 scripts/build_kitti_loop_asset.py --mode real-vo \
       --input-dir target/kitti_image_vo_loop_demo \
       --truth-kitti-poses ~/datasets/kitti_seq00_subset/poses_00.txt \
       --gt-stride 4 --out-dir docs/assets
   ```

   Sample run: VO endpoint drift ~548 m (Procrustes-aligned), corrected
   endpoint error ~5 m after a single loop edge.

The Python helper is asset-generation only — not part of the Rust
runtime or the CI gate.

## Legacy GT-Pose-Based KITTI Asset

A simpler `online_slam_kitti_loop_demo` example (Rust-only, no images)
is kept around for environments without the KITTI image archive. It
reads `<KITTI>/poses/<seq>.txt` directly, fabricates a per-edge yaw
drift on the truth trajectory, adds a single truth-relative loop edge
between the first and last keyframes, runs `optimize_se3_iterative`
(LM + Cholesky), and writes the same `truth.csv`/`drifted.csv`/
`corrected.csv` triple. The `--mode gt-drift` path of
`scripts/build_kitti_loop_asset.py` renders that variant.

(Original GT-pose pipeline:)

1. Run the Rust demo against the KITTI odometry pose file:

   ```sh
   cargo run --release --example online_slam_kitti_loop_demo -- \
       --kitti-poses /path/to/KITTI_odometry/poses/00.txt \
       --keyframe-stride 30 \
       --max-keyframes 200 \
       --out-dir target/kitti_loop_demo
   ```

   The demo subsamples the GT trajectory to ~150 keyframes, fabricates a
   realistic per-edge yaw drift on the sequential odometry (so the chained
   estimate diverges hundreds of metres by the end of the loop), adds a
   single truth-relative loop-closure constraint between the first and last
   keyframes, and runs `PoseGraph::optimize_se3_iterative` (Levenberg-
   Marquardt + Cholesky). It writes `truth.csv`, `drifted.csv`, and
   `corrected.csv` (id, x, y, z) to the output directory. Sample run:
   `endpoint err: drifted=160.9 m → corrected=0.007 m` after 12 LM
   iterations.

2. Render the README asset:

   ```sh
   python3 scripts/build_kitti_loop_asset.py \
       --input-dir target/kitti_loop_demo \
       --out-dir docs/assets
   ```

   This emits `kitti_loop_closure.png` (three-panel: truth / drifted /
   corrected, top-down XZ view) and `kitti_loop_closure.gif` (drifted →
   corrected morph animation). The Python helper is asset-generation only,
   not part of the Rust runtime or the CI gate.
