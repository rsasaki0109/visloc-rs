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
- `docs/assets/south-building-deep-vs-classical-matches.jpg`: real classical-vs-deep COLMAP localization match overlay on the `P1180141 → P1180144` pair, produced by `examples/deep_localization_demo.rs --out-dir ...` + `scripts/render_deep_localization_matches.py`. Top row shows classical Corner+BF inliers (132); bottom row shows deep HogLike+MutualSoftmax inliers (289). The match lines and counts are exactly what the Rust pipeline reports — not a Python-side overlay.

The deep-vs-classical comparison asset is generated with:

```sh
# 1. Run the demo with --out-dir to write correspondences.json + summary.txt
cargo run --release --features image-io --example deep_localization_demo -- \
    --root ~/datasets/south-building/south-building \
    --map-image P1180141.JPG --query-image P1180144.JPG \
    --out-dir target/deep_localization_real

# 2. Render the side-by-side PNG/JPG
python3 scripts/render_deep_localization_matches.py \
    --correspondences target/deep_localization_real/correspondences.json \
    --images-dir ~/datasets/south-building/south-building/images \
    --output docs/assets/south-building-deep-vs-classical-matches.jpg \
    --image-width 700 --max-lines 180
```

The renderer needs only Pillow; no OpenCV. The numbers in the title bar
(matches / inliers) are read straight from the demo's JSON output, so
the asset cannot drift away from the actual pipeline behaviour without
re-running both steps.

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

The current README KITTI VO hero (`docs/assets/kitti_deep_vo.{png,gif}`) is
generated from the deep-frontend metric stereo `online_slam_stereo_vo_kitti_demo`
pipeline on a 260-frame KITTI 00 seq00 subset. Pipeline:

1. Stream a 260-frame stereo subset of KITTI odometry seq 00
   directly from the public S3 archive (no full 23 GB download required):

   ```sh
   python3 scripts/fetch_kitti_seq00_images.py \
       --stride 1 --max-frames 260 --workers 8 --also-fetch-poses \
       --cameras image_0,image_1 \
       --out-dir ~/datasets/kitti_seq00_stride1_subset
   ```

2. Run the deep metric stereo VO path with the adopted leaderboard-oriented
   settings:

   ```sh
   scripts/run_kitti_deep_vo_smoke.sh --skip-fetch \
       --out-dir target/kitti_deep_vo_refine_guard_pnp332_260 \
       --max-frames 260 \
       --deep-max-features 1500 \
       --deep-descriptor-clip 0.2 \
       --deep-temperature 25.0 \
       --pnp-reprojection-threshold 3.32 \
       --relative-pose-iterations 1000
   ```

3. Render the README asset:

   ```sh
   python3 scripts/build_kitti_loop_asset.py --mode stereo-vo \
       --frontend-label "deep stereo VO" \
       --input-dir target/kitti_deep_vo_refine_guard_pnp332_260_early_stop \
       --out-dir docs/assets
   ```

   Sample 260-frame run: raw deep VO mean/RMSE/max ATE
   2.49 / 2.63 / 4.18 m, with KITTI-style 100 m windows
   `t_rel = 0.675%` and `r_rel = 0.0146 deg/m`.
   The demo also writes KITTI-style pose rows (`vo_poses.txt`,
   `gt_poses.txt`) plus
   `relative_pose_errors.csv` when GT is available, so per-frame
   translation-magnitude and rotation drift can be inspected before the
   leaderboard-style relative-motion evaluator is run on the same output:

   ```sh
   cargo run --example evaluate_kitti_odometry_benchmark -- \
       target/kitti_deep_vo_refine_guard_pnp332_260/vo_poses.txt \
       target/kitti_deep_vo_refine_guard_pnp332_260/gt_poses.txt
   ```

The older loop-closure hero (`docs/assets/kitti_loop_closure.{png,gif}`) can
still be regenerated with `--mode stereo` from a 50-frame
`--synthetic-loop-closure` run when the goal is to visualize the BA/PGO
correction chain instead of raw VO quality.

   Sample short-window run: 122 windows, mean `t_rel = 3.41%`,
   mean `r_rel = 0.0447 deg/m`. Running without `--lengths` uses KITTI's
   current public benchmark lengths (`100,200,...,800 m`); this 45.70 m
   slice is too short for those windows, so it intentionally produces no
   leaderboard-style segments unless you pass shorter development lengths.

   For a longer seq00 smoke test, run the helper that fetches/reuses 260
   stride-1 stereo frames, runs the same deep frontend with a smaller
   relative-pose RANSAC budget, and writes both current public-length and 100 m
   KITTI summaries:

   ```sh
   scripts/run_kitti_deep_vo_smoke.sh
   ```

   To avoid overfitting seq00 before performance tuning, use the training
   benchmark runner to execute the same smoke path over KITTI odometry
   sequences 00-10 and collect one sequence-level report:

   ```sh
   scripts/run_kitti_deep_vo_train_benchmark.sh --max-frames 260
   ```

   It writes `target/kitti_deep_vo_train_benchmark/summary.csv` and
   `summary.md` with ATE, public-length `t_rel` / `r_rel`, fallback counts,
   the worst relative-pose pair, the worst KITTI segment, and links to each
   sequence's `slam_debug` report.

   To compare a new benchmark run against a previous root, pass
   `--compare-root <old-benchmark-root>`. The runner writes per-sequence
   `slam_debug_compare.md` links into the consolidated `summary.md` whenever
   both sequence directories are present:

   ```sh
   scripts/run_kitti_deep_vo_train_benchmark.sh \
       --out-dir target/kitti_deep_vo_train_benchmark_new \
       --compare-root target/kitti_deep_vo_train_benchmark
   ```

   Sample 260-frame seq00 run: GT length 183.11 m, raw deep VO mean/max ATE
   2.49 / 4.18 m, and KITTI current public lengths report `t_rel = 0.675%`,
   `r_rel = 0.0146 deg/m` over 97 windows with the leaderboard-oriented
   3.32 px PnP reprojection gate, guarded PnP refinement, high-consensus
   PnP early-stop, lazy Kabsch fallback, adaptive 60 m depth rescue,
   conservative motion-scale band rescue, p75 scale-target rescue,
   translation-direction rescue, rotation-spike rescue, and automatic stereo
   translation refinement for fast low-consensus pairs. The rescue gates leave
   the seq00 260-frame result unchanged; on the seq01 260-frame highway slice
   they improve public-length `t_rel` from `20.230%` to `4.984%` and `r_rel`
   from `0.0313` to `0.0203 deg/m`. The older
   short-window development set
   (`--lengths 5,10,50,100,150,200,250,300,350,400`) reports `t_rel =
   1.97%`, `r_rel = 0.0438 deg/m` over 816 windows on the same run.
   This is an open seq00 subset diagnostic, not a held-out KITTI
   leaderboard submission.

   The smoke script also writes a visual-SLAM debug bundle under
   `target/kitti_stereo_vo_deep_260/slam_debug/`:

   - `slam_debug_report.md` / `slam_debug_report.html`: headline ATE,
     source counts, worst translation pairs, worst rotation pairs, weakest
     inlier-ratio pairs, and worst KITTI segments.
   - `slam_debug_summary.json`: machine-readable version of the same report.
   - `slam_debug_worst_pairs.csv`: compact triage table with tags such as
     `fallback`, `weak_pnp`, `scale`, and `rotation`.

   You can regenerate the report for any existing stereo VO output directory:

   ```sh
   scripts/visual_slam_debug_report.py target/kitti_stereo_vo_deep_260
   ```

   To compare two runs, pass the older run as `--compare`. The candidate run is
   the positional argument, so negative deltas on error metrics mean the
   candidate improved:

   ```sh
   scripts/visual_slam_debug_report.py \
       target/kitti_deep_vo_train_seq01_direction_rescue10 \
       --compare target/kitti_deep_vo_train_seq01_scale_ratio080
   ```

   This additionally writes `slam_debug_compare.md`,
   `slam_debug_compare.html`, `slam_debug_compare.json`, and
   `slam_debug_compare_metrics.csv`, including KITTI segment-level deltas such
   as `205→243: 35.949% → 4.933%`.

   To run the long VO smoke and the revisit scanner smoke together, use:

   ```sh
   scripts/run_kitti_deep_stack_smoke.sh
   ```

   This writes `target/kitti_deep_stack_smoke/deep_stack_smoke_summary.txt`
   and `target/kitti_deep_stack_smoke/deep_stack_smoke_summary.json` plus
   separate `vo/` and `revisit/` output folders. The JSON summary exposes the
   headline ATE, per-frame relative-pose diagnostics, mean/max KITTI
   relative-motion, and strongest revisit-loop metrics for regression
   comparisons.

The Python helper is asset-generation only — not part of the Rust
runtime or the CI gate.

The helper still supports `--mode real-vo` for the older monocular
`online_slam_image_vo_loop_demo` asset path, but the README hero uses the
deep stereo output because it shows the deep-style matcher in a metrically
meaningful VO → BA → PGO chain without Procrustes scale alignment.

## KITTI Revisit Scanner Smoke

The real-image appearance-loop smoke test uses the same deep-style frontend on
two KITTI 00 slices: frames `0-49` from the start and frames `4500-4529` from
the major revisit area. The helper fetches/reuses both `image_0` subsets, runs
`kitti_revisit_scanner_demo`, requires a strongest loop pair, and writes
`deep_revisit_smoke_summary.txt`:

```sh
scripts/run_kitti_deep_vo_revisit_smoke.sh --frontend deep
```

Sample run: 62 cross-segment candidates, strongest pair `(49, 4500)` with
152 inliers, inlier ratio 0.749, and score 76574.96. Use `--frontend both` to
rerun the documented classical-vs-deep comparison.

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
