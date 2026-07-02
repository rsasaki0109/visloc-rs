# Timestamped Image GNSS-Prior Demo

This demo is the smallest file-backed image-sequence localization path in `visloc-rs`. It creates a short PNG sequence, a separate nanosecond timestamp file, a GNSS-like world-position log, then localizes each image frame against a reusable sparse visual map with an external localization prior.

Run it with:

```bash
cargo run --features image-io --example track_timestamped_image_sequence_with_gnss_prior
```

The output directory is:

```text
target/visloc_timestamped_image_sequence_demo
```

## What It Writes

- `0000.png`, `0001.png`, and `0002.png` are the grayscale image sequence frames.
- `timestamps_ns.txt` stores one image timestamp per line in nanoseconds.
- `gnss_world.txt` stores one GNSS-like prior row per timestamp: `timestamp_ns x y z horizontal_accuracy vertical_accuracy`.
- `gnss_sync_evaluation.json` records whether the external measurements cover every image frame timestamp.

For a successful smoke run, `gnss_sync_evaluation.json` should contain:

```json
{
  "passed": true,
  "summary": {
    "frame_count": 3,
    "measurement_count": 3,
    "matched_frame_count": 3,
    "missing_measurement_count": 0,
    "matched_frame_ratio": 1
  },
  "failures": []
}
```

The console output should show three successful localized frames, a GNSS sync ratio of `1.000`, and `external_prior_rate=1.000`.

## Why This Matters

Real automotive and UAV datasets often keep images, image timestamps, and GNSS logs as separate files. This demo exercises that data shape without adding a full robotics log reader:

1. `read_common_image_sequence_dir_with_timestamp_file` loads ordered images and attaches timestamps.
2. `read_gnss_measurements_txt` loads external position priors.
3. `FramePriorSource` matches image frame timestamps to nearest GNSS measurements.
4. `FramePriorSyncEvaluationConfig` checks whether the sync coverage is good enough before tracking.
5. `ImageTracker` localizes each image using a GNSS-derived submap prior.

This is still Visual Localization, not full SLAM. The purpose is to make real
dataset plumbing visible and testable while keeping SLAM-stage optimization and
tightly-coupled fusion experiments in their own examples and benchmark records.

## CI Artifact

CI runs `scripts/check_timestamped_gnss_image_demo_outputs.sh` and uploads the checked output directory as the `timestamped-gnss-image-demo-outputs` artifact. Download that artifact from a GitHub Actions run to inspect the generated images, timestamp file, GNSS prior log, and sync evaluation JSON.
