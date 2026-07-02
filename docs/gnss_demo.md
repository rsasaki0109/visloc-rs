# GNSS-Prior Tracking Demo

This demo is the current small, reproducible vertical slice for sequence localization with an external position prior. It does not implement online SLAM yet. It shows a moving camera localizing against a reusable sparse visual map while a GNSS-like prior narrows the landmark search region before matching and PnP.

Run it with:

```bash
cargo run --example track_sequence_with_gnss_prior -- --out-dir target/visloc_gnss_tracking_demo
```

Open `target/visloc_gnss_tracking_demo/index.html` in a browser first. It links the tracking report, trajectory report, trajectory evaluation, CSV exports, JSON summaries, and KITTI/TUM pose files from one page.

## What To Look For

The demo is intentionally tiny so each file is inspectable:

- `index.html` is the dashboard entry point.
- `tracking_report.html` shows frame-by-frame localization status, inlier counts, reprojection error, map sizes, and prior usage.
- `tracking_evaluation.json` records whether tracking-level smoke thresholds passed.
- `trajectory_report.html` shows the estimated moving-camera path.
- `trajectory_evaluation.html` compares the estimated trajectory against the synthetic reference trajectory.
- `tracking.csv` is the per-frame tracking log.
- `trajectory.csv` is the estimated camera-center trajectory.
- `translation_errors.csv` is the per-frame translation error against the reference trajectory.
- `manifest.json` lists all generated files plus headline tracking, trajectory, and error metrics.

For a successful smoke run, the dashboard should describe a 3-frame sequence with all frames localized, 3 external localization priors used, and a path length of about `0.91 m`. The synthetic reference is noise-free, so mean translation error should be near zero.

## Why This Matters

The visual-localization core is still the important part:

1. A visual map contains reusable 3D landmarks and descriptors.
2. Each query frame provides 2D keypoints and descriptors.
3. A GNSS-like external prior selects a smaller candidate submap.
4. Descriptor matching builds 2D-3D correspondences.
5. PnP + RANSAC estimates the camera pose.
6. Reports expose tracking state, priors, inliers, reprojection error, and trajectory output.

This is a practical bridge toward automotive or UAV localization demos: the
sequence and prior plumbing exists, while production full SLAM and
tightly-coupled fusion remain intentionally out of scope for this demo. Loop
closure, pose-graph, and BA experiments live in the separate SLAM examples and
benchmark docs, not in the GNSS-prior tracking smoke path.

## CI Artifact

CI runs `scripts/check_gnss_demo_outputs.sh` and uploads the checked output directory as the `gnss-demo-outputs` artifact. Download that artifact from a GitHub Actions run to inspect the exact dashboard and reports produced by CI.
