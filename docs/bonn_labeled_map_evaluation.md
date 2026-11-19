# Labeled Bonn RGB-D map evaluation

This pilot uses manual annotations on three real Bonn RGB-D Dynamic recordings. It evaluates Basalt's motion-confirmation gate on an RGB replay and the RGB-D visibility classifier on the addition and removal of a box between separate recordings in one room. The annotated polygons and raw confusion counts are versioned with the benchmark. [Annotations](../benchmarks/bonn/map_change_labels.json) · [Results](../benchmarks/bonn/map_change_results.json) · [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)

## Motion exclusion

The 16 labeled RGB frames show a person and a second person carrying a large box past the camera, with four frames of static background and static object proposals. The annotations mark moving silhouettes and the ambiguous region at the left edge of frame 197. The evaluator scores rectified RGB pixels away from the 10 px silhouette boundary band and the outer 10 px image border.

The replay runs the production `basalt_euroc_vio_demo --mono-vo` on all 590 RGB frames, using zero IMU samples. On the labeled frames, bounding rectangles of the annotations serve as oracle proposals with manually assigned object IDs. Pixels within the exported motion-confirmed rectangles count as excluded. The same rectangles without motion confirmation form an aggressive comparison. Their recall is 100% by construction. Precision, recall and F1 are micro-averaged over the labeled pixels. Static retention is the fraction of labeled static pixels outside the excluded rectangles.

| Exclusion | Precision | Recall | F1 | Static retention |
| --- | ---: | ---: | ---: | ---: |
| Off | — | 0.0% | 0.0% | 100.0% |
| All proposal rectangles | 82.8% | 100.0% | 90.6% | 83.5% |
| Basalt motion confirmation | 91.7% | 28.3% | 43.2% | 98.0% |

Basalt evaluated 23 proposals, confirmed 6, and recorded 8 as undecided. It dropped 701 feature observations. Frame 235 was filled by the carried box, leaving no background reference for its motion decision. None of the proposals from frame 197 through 348 were confirmed. The gate preserves static pixels better than removing every supplied rectangle, at a substantial cost in moving-pixel recall on this sequence.

## Map update

Both comparisons use different recordings, `placing_nonobstructing_box` and `removing_nonobstructing_box`. The addition compares a window before placement with a window before removal. The removal compares a window after placement with a window after removal. The windows exclude the manipulation itself. The evaluator rectifies registered RGB and depth, converts TUM depth units to millimetres, uses Bonn's published marker-to-camera calibration and interpolates the public mocap poses in the common coordinate frame. The rotational block of the published calibration is projected to the closest orthonormal rotation.

Eleven labeled RGB-D frames are held out from prediction. Conservative interior polygons cover the box and unchanged wall, furniture and floor. Valid depth samples inside the polygons generate 7 cm reference voxels. Voxels with conflicting manual labels are excluded. The compared methods share the same reference voxels, including any voxels absent from both predicted candidate sets. A positive label is a physical surface belonging to the placed or removed box. A labeled floor voxel counts as occluded if its center projects into a held-out box polygon and at least 10 cm behind the observed box depth in one of those views.

The occupancy baseline marks voxels sampled in exactly one visit as changed. The production visibility classifier uses a 10 cm depth margin and two supporting views, and treats occlusion as an absence of free-space evidence. Unknown and intermittent predictions count as missed changes on positive labels. The candidate set samples every fifth depth frame and every eighth pixel along both image axes. One sampled depth filename is missing from the official placement archive and is recorded as skipped.

| Box change | Method | Precision | Recall | F1 | Occluded floor marked changed |
| --- | --- | ---: | ---: | ---: | ---: |
| Addition | Occupancy difference | 23.4% | 96.5% | 37.6% | 215/227 |
| Addition | Visibility | 77.6% | 72.8% | 75.1% | 0/227 |
| Removal | Occupancy difference | 29.4% | 96.5% | 45.1% | 60/71 |
| Removal | Visibility | 80.2% | 80.2% | 80.2% | 0/71 |

For addition, 25 and 24 synchronized depth frames produced predictions. Visibility detected 83 of 114 labeled changed voxels and assigned the correct direction to all 83. It labeled 708 of 1185 unchanged voxels stable. For removal, 11 and 11 synchronized depth frames produced predictions. Visibility detected 69 of 86 labeled changed voxels and assigned the correct direction to all 69. It labeled 794 of 1152 unchanged voxels stable.

## Scope

The sample covers one room, one moving-object recording, two kinds of box changes and a single annotator. The pixel counts describe these frames, whose observations are correlated. The motion results are conditional on sparse manual box proposals and visual-only tracking. They do not measure detector recall, inertial VIO or sparse-map landmark removal during session merge. The map numbers isolate classification given mocap poses. Evaluating registration error, long gaps between revisits, or persistent map writes needs a separate experiment.

## Reproduce

Download and extract `rgbd_bonn_moving_obstructing_box.zip`, `rgbd_bonn_placing_nonobstructing_box.zip` and `rgbd_bonn_removing_nonobstructing_box.zip` from the dataset page into the same directory. The evaluation used the following archive SHA-256 hashes. [Dataset downloads](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html)

```text
d0fc8c33d896b6a4c4a506508f44ddec63bf7dc37c8f58524826cbca1bbba689  rgbd_bonn_moving_obstructing_box.zip
8eda7eabf1f71217d202d12f438201146a1488c1eb3664e21f490701934b9bcc  rgbd_bonn_placing_nonobstructing_box.zip
a1d4731b6d0a60d50009c4c401336a23957c5182e3be79fd8e7028e9dde1db8d  rgbd_bonn_removing_nonobstructing_box.zip
```

From the repository root, with Python 3.10 or later and the Rust toolchain installed, run the following.

```bash
python -m pip install numpy scipy pillow opencv-python-headless
cargo build --release --example basalt_euroc_vio_demo
python scripts/eval_bonn_map_changes.py \
  --data /path/to/extracted/bonn \
  --basalt target/release/examples/basalt_euroc_vio_demo \
  --annotations benchmarks/bonn/map_change_labels.json \
  --out runs/bonn_map_change_results.json
```

The annotation frame indices are zero-based rows in `depth.txt` after comments. Polygons use rectified, 640 × 480 image coordinates. Moving polygons and stationary proposal regions are listed in the same order as their persistent object IDs. Each map visit has an inclusive window and separately listed holdout frames.
