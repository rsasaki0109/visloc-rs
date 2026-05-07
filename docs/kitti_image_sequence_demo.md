# KITTI Image Sequence Demo

This demo is the smallest automotive-style dataset loader path in `visloc-rs`.
It creates a KITTI-like image folder, a nanosecond timestamp file, and a
`calib.txt` file with projection rows, then reads them back as a validated
`KittiImageSequence`.

Run it with:

```bash
cargo run --features image-io --example load_kitti_image_sequence
```

The output directory is:

```text
target/visloc_kitti_image_sequence_demo
```

## What It Writes

- `image_2/000000.png`, `image_2/000001.png`, and `image_2/000002.png` are a short automotive-like grayscale image sequence.
- `times_ns.txt` stores one image timestamp per line in nanoseconds.
- `calib.txt` stores KITTI-style projection rows. The demo reads `P2`.
- `output.log` is written by the local/CI smoke check and records the loader summary.

For a successful smoke run, the output should include:

```text
camera id=2 size=64x48 intrinsics=Some((710.0, 705.0, 32.0, 24.0))
frames=3 timestamps=3 timestamp_valid=true dimension_issues=0 timestamp_issues=0
```

## Why This Matters

Automotive datasets often start as separate image directories, timestamp files,
and calibration files. This demo makes that data shape testable before adding a
larger KITTI/Oxford/nuScenes localization example:

1. `write_png_gray` creates a small image sequence for the smoke test.
2. `read_kitti_image_sequence_dir_with_timestamp_file` loads images, timestamps, and calibration.
3. `read_kitti_pinhole_camera` converts the selected KITTI projection row into `Camera::pinhole`.
4. `KittiImageSequence` returns the camera, loaded frames, sequence summary, dimension issues, and timestamp issues.

This is dataset plumbing for Visual Localization, not a SLAM backend. The next
step is to feed the loaded frames into feature extraction, tracking, and
map-based localization against a reusable visual map.

## CI Artifact

CI runs `scripts/check_kitti_image_sequence_demo_outputs.sh` and uploads the
checked output directory as the `kitti-image-sequence-demo-outputs` artifact.
Download that artifact from a GitHub Actions run to inspect the generated image
sequence, timestamps, calibration file, and `output.log`.
