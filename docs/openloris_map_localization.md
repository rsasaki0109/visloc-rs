# Map-based localization on a real wheeled robot (OpenLORIS-Scene)

The EuRoC results in [basalt_map_localization.md](basalt_map_localization.md)
come from a drone in a static room, flown minutes apart. This page runs
the same pipeline on a wheeled robot over four days in an office whose
furniture and lighting changed between sessions. It uses the stereo-inertial
SLAM for the maps, `basalt_euroc_localize_demo` to localize against them,
and the VIO or the robot's own wheel odometry, corrected into the map frame
by those localizations.

**Summary.**

- **Same lighting:** within the same evening, and 2.8 days later with the
  office rearranged, the VIO corrected into the map frame gives a map pose for 79–99%
  of the frames at 2.9–3.5 cm RMSE. This holds although only 3–37% of the
  images localize on their own.
- **Wheel odometry:** correcting the robot's wheel odometry instead gives
  7–12 cm. It drifts by up to 17 cm in an in-place turn, which a few map
  fixes cannot correct.
- **Daylight:** the two sessions recorded in daylight with the ceiling
  lights off never localize, in any map. ORB features do not survive that
  change.

## Data

[OpenLORIS-Scene](https://lifelong-robotic-vision.github.io/dataset/scene)
records a wheeled robot with a RealSense T265 (two fisheye cameras and an
IMU), a D435i, wheel odometry, and motion-capture ground truth of
`base_link` in the office scene. The seven office sequences, taken from the
dataset's Hugging Face mirror (`package/office1-1_7-package.tar`, 10.6 GB):

| Sequence | Recorded (UTC+8) | Length | Lighting |
|---|---|---|---|
| office1-1 | Jun 8, 21:21 | 27 s | ceiling lights on |
| office1-2 | Jun 8, 21:22 | 30 s | ceiling lights on |
| office1-3 | Jun 8, 21:21 | 12 s | ceiling lights on |
| office1-4 | Jun 9, 11:44 | 29 s | daylight, lights off |
| office1-5 | Jun 10, 05:46 | 53 s | dawn, lights off |
| office1-6 | Jun 11, 17:11 | 36 s | ceiling lights on, furniture moved |
| office1-7 | Jun 11, 17:22 | 38 s | ceiling lights on, furniture moved |

All sequences follow roughly the same 6 m path through the room.
office1-1 and office1-6 drive it in one direction. office1-2 and office1-7
drive it in the other. office1-3 mostly turns in place. office1-4 and
office1-5 drive it both ways. The camera faces forward, so opposite
directions see opposite sides of the room.

`scripts/openloris_to_euroc.py` converts a sequence (its `.7z`, or the
extracted directory) to the EuRoC layout the Basalt examples read:

- **Images.** The Basalt port models cameras as Double Sphere only. No
  Double Sphere camera reproduces the T265's Kannala-Brandt lenses over
  their whole field of view: they reach 97° off-axis at the image edge,
  and a fit out to 85° leaves 1.9 px RMS. So the images are resampled to a
  Double Sphere camera fitted within 60° (0.15 px RMS). The calibration
  is then exact by construction and the whole field of view is kept.
- **IMU.** The T265 accelerometer runs at 62.5 Hz. It is interpolated to
  the 200 Hz gyro stamps.
- **Ground truth and odometry.** The ground truth (`base_link`, not the
  mocap marker) is moved to the T265 IMU. Read as the marker's pose,
  office1-1's VIO looks 14 cm off instead of 2.7 cm. The wheel odometry is
  written as a TUM file of `base_link`.

## Maps

Each map is the stereo-inertial SLAM of one sequence
(`basalt_euroc_online_slam_demo --save-map`, EuRoC configuration), at about
0.5× real time on 4 cores (848×800 images).

| Map | From | SLAM ATE | Scale vs ground truth | Keyframe images / landmarks |
|---|---|---|---|---|
| A | office1-1 | 2.75 cm | 0.983 | 332 / 4,027 |
| B | office1-2 | 1.58 cm | 0.991 | 356 / 3,969 |
| — | office1-3 | 0.94 cm | 0.965 | 170 / 1,298 |

The SLAM comes out 1–3.5% short in scale on every sequence. This points to
the T265 calibration shipped with the dataset (`trans_matrix.yaml`).
Because of it, a fix is 3–4 cm off on average however well it matches.

A and B were driven in opposite directions, and `basalt_map_merge` cannot
register them: none of B's keyframes localizes in A. office1-3 turns in
place and sees part of both:

- office1-3 registers in A: 91 of its 170 keyframes localize, with a 0.4 cm
  keyframe residual.
- Then B registers in A plus office1-3: 121 of its 356 keyframes localize,
  with a 1.2 cm residual.

The merged map (858 keyframe images, 9,158 landmarks) covers both
directions. It is in A's frame and is scored with A's alignment to the
ground truth.

## Results

Each query is localized against a map built from another session, with
these settings:

- Localization every 5th image. The VIO runs on every frame; the wheel
  odometry is interpolated at every image.
- Projection matching (`--projection-radius-px 20`).
- Scoring with the map's fixed alignment to the mocap frame: the
  "fixed" mode of `scripts/evaluate_map_localization.py`.

"Alone" is each odometry's own error after a rigid fit to the ground truth.
It is in the odometry's own frame, not the map's.

| Query → map | Images that localize alone | VIO + map: frames, RMSE / max | Wheel odometry + map: frames, RMSE / max | VIO alone | Wheel alone |
|---|---|---|---|---|---|
| office1-3 → A (minutes later) | 36.7% | 98.6%, 2.85 / 3.9 cm | 96.9%, 11.7 / 19.6 cm | 0.83 cm | 0.97 cm |
| office1-6 → A (2.8 days, moved furniture) | 6.7% | 98.6%, 3.21 / 4.3 cm | 98.6%, 7.19 / 15.7 cm | 3.24 cm | 4.80 cm |
| office1-6 → merged | 2.9% | 94.4%, 3.30 / 4.7 cm | 94.4%, 7.30 / 15.8 cm | | |
| office1-7 → B or merged | 0% | no map pose | no map pose | 5.29 cm | 7.90 cm |
| office1-7 → merged, `--min-inliers 15` | | 79.4%, 3.50 / 6.0 cm | | | |
| office1-4 → A, B or merged (daylight) | 0% | no map pose | no map pose | | 8.34 cm |
| office1-5 → merged (dawn) | 0% | no map pose, also at 15 inliers | no map pose | 5.20 cm | 9.40 cm |

What this shows:

- **The correction keeps the robot in the map frame although few images
  localize.** The robot needs map-frame poses across sessions, which VIO
  alone does not give: it has its own frame each time. On office1-6, 2.8
  days later with the furniture moved, fewer than 7% of the images
  localize alone. The corrected VIO still gets a map pose for 98.6% of the
  frames, about as accurate as the VIO's own rigid-fit error. On these
  6 m paths the VIO barely drifts, so the map's own few-cm error (its
  scale above) is what bounds the result.
- **Projection matching.** Matching around the corrected VIO's prediction
  raises office1-6's fixes from 16 to 137 of its 216 attempts. The
  VIO + map RMSE falls from 3.93 to 3.21 cm.
- **The wheel odometry is the weaker partner on this robot.** Its heading
  agrees with the ground truth within 0.5°. In office1-3's final 150°
  in-place turn, though, it reports `base_link` moving along a different
  path than the mocap does: 16.7 cm apart after 7.7 s. The camera turns
  away from the mapped view in the same turn, so no fix corrects it.
  Rigid-fit numbers (0.97 cm) hide this, because the fit rotates the turn
  into place. Where fixes keep coming, the corrected wheel odometry stays
  within a few cm of the VIO's result.
- **Merging both directions costs some matches.** office1-6 localizes
  alone in 2.9% of its images against the merged map, against 6.7% in A.
  The many near-identical landmarks make the ratio test reject more
  matches (`--duplicate-radius-m 0.1` did not change this run). Once the
  VIO has a map pose, projection matching makes up for it.
- **office1-7 gets map poses only below the default inlier threshold.** Its best
  map-wide match has 21 inliers against B and 17 against the merged map. At `--min-inliers 15` the first fixes come
  through, and no accepted fix is more than 10 cm off. On EuRoC, though,
  20 inliers already admits poses metres off, so 15 is not a safe default.
  The correction's two-fix agreement check is what keeps this run clean.
- **Lighting is the hard limit.** office1-4 and office1-5 are recorded
  in daylight with the ceiling lights off. Their best images reach at
  most 12 inliers against maps recorded at night. No map, threshold or odometry
  gives them map poses: ORB features on these images do not match across
  that change.

## Reproduce

```bash
# Once per sequence (needs numpy, Pillow and py7zr): office1-1 .. office1-7
U=https://huggingface.co/datasets/shixuesong/openloris-scene/resolve/cbc03108723d08322b23d0338680bffa9404cce9/package
# office1-1 is the first member of the tar: bytes 512 .. 512+1365585538-1
curl -L -r 512-1365586049 -o office1-1.7z $U/office1-1_7-package.tar
python3 scripts/openloris_to_euroc.py --sequence office1-1.7z --out data/office1-1

# Map
target/release/examples/basalt_euroc_online_slam_demo --euroc-dir data/office1-1 \
  --calibration data/office1-1/calibration.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --out-dir runs/o11_slam --save-map runs/o11_map

# Query: VIO corrected into the map frame
target/release/examples/basalt_euroc_localize_demo --map runs/o11_map \
  --euroc-dir data/office1-6 --out-dir runs/o16_vio --vio \
  --calibration data/office1-6/calibration.json \
  --config configs/basalt/variants/official_euroc_ds/euroc_config.json \
  --localize-every 5 --projection-radius-px 20

# Query: the wheel odometry corrected into the map frame. --imu-from-body is
# base_link in the T265 IMU frame, the inverse of T_base_imu in
# data/office1-6/extrinsics.json.
target/release/examples/basalt_euroc_localize_demo --map runs/o11_map \
  --euroc-dir data/office1-6 --out-dir runs/o16_wheel \
  --odometry data/office1-6/odom.tum \
  --imu-from-body 0.0306,-0.8937,-0.2452,-0.5030,-0.4927,-0.5074,0.4967 \
  --localize-every 5 --projection-radius-px 20

# Score against the ground truth, through the map's own alignment
python3 scripts/evaluate_map_localization.py --mode fixed \
  --estimate runs/o16_vio/trajectory_map.tum \
  --reference data/office1-6/mav0/state_groundtruth_estimate0/data.csv \
  --align-estimate runs/o11_slam/trajectory_online.tum \
  --align-ground-truth data/office1-1/mav0/state_groundtruth_estimate0/data.csv \
  --queries 1080
```

The byte ranges of the other sequences are in the tar's headers. The
dataset is CC BY-ND 4.0: keep the converted images out of the repository.

## Limits

- **One scene, short paths.** Seven sequences, all in one office, each at
  most 11 m. They test change over days, not large areas.
- **ORB only.** Across the day/night change no image localizes. Learned
  features would be the next thing to try.
- **The T265's shipped calibration.** It leaves the maps 1–3.5% short in
  scale, so every map pose carries a few cm of error. Recalibrating would
  likely remove most of it.
- **The wheel odometry is planar and its frame is assumed level.** The
  office floor is level, which this needs.
